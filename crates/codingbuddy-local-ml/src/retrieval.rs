use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::chunker::{Chunk, ChunkConfig, ChunkManifest};
use crate::embeddings::EmbeddingsBackend;
use crate::reranker::RerankerBackend;
use crate::vector_index::{BruteForceBackend, SearchFilter, VectorIndex, VectorIndexBackend};

/// A single retrieval result combining vector and BM25 scores.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub chunk: Chunk,
    pub vector_score: f32,
    pub bm25_score: f32,
    pub hybrid_score: f32,
    pub rank: usize,
}

/// Report from building a full index.
#[derive(Debug, Clone, Default)]
pub struct IndexBuildReport {
    pub chunks_indexed: usize,
    pub files_processed: usize,
    pub duration_ms: u64,
}

/// Report from an incremental index update.
#[derive(Debug, Clone, Default)]
pub struct IndexUpdateReport {
    pub chunks_added: usize,
    pub chunks_removed: usize,
    pub files_changed: usize,
    pub duration_ms: u64,
}

/// Hybrid retriever combining vector similarity search with BM25 text search.
///
/// Uses Reciprocal Rank Fusion (RRF) to combine rankings from both methods.
/// RRF is preferred over linear score blending because BM25 and cosine similarity
/// scores are on incomparable scales.
pub struct HybridRetriever {
    vector_index: VectorIndex,
    embeddings: Arc<dyn EmbeddingsBackend>,
    tantivy_index: Option<codingbuddy_index::IndexService>,
    reranker: Option<Arc<dyn RerankerBackend>>,
    blend_alpha: f32,
    chunk_config: ChunkConfig,
    manifest: ChunkManifest,
    /// Number of candidates to pass to cross-encoder reranker (default 20).
    rerank_top_n: usize,
}

impl HybridRetriever {
    /// Create a new hybrid retriever.
    ///
    /// - `blend_alpha`: 0.0 = pure BM25, 1.0 = pure vector search
    pub fn new(
        index_path: &Path,
        embeddings: Arc<dyn EmbeddingsBackend>,
        tantivy_index: Option<codingbuddy_index::IndexService>,
        blend_alpha: f32,
        chunk_config: ChunkConfig,
    ) -> Result<Self> {
        let dimension = embeddings.dimension();
        let backend = Box::new(BruteForceBackend::new(dimension));
        let vector_index = VectorIndex::create(index_path, backend)?;

        Ok(Self {
            vector_index,
            embeddings,
            tantivy_index,
            reranker: None,
            blend_alpha,
            chunk_config,
            manifest: ChunkManifest::default(),
            rerank_top_n: 20,
        })
    }

    /// Create a hybrid retriever with a custom vector index backend.
    pub fn new_with_backend(
        index_path: &Path,
        embeddings: Arc<dyn EmbeddingsBackend>,
        backend: Box<dyn VectorIndexBackend>,
        tantivy_index: Option<codingbuddy_index::IndexService>,
        blend_alpha: f32,
        chunk_config: ChunkConfig,
    ) -> Result<Self> {
        let vector_index = VectorIndex::create(index_path, backend)?;

        Ok(Self {
            vector_index,
            embeddings,
            tantivy_index,
            reranker: None,
            blend_alpha,
            chunk_config,
            manifest: ChunkManifest::default(),
            rerank_top_n: 20,
        })
    }

    /// Set an optional cross-encoder reranker for a final precision pass.
    ///
    /// After RRF fusion, the top `rerank_top_n` candidates are re-scored
    /// by the cross-encoder. This significantly improves precision@k.
    pub fn set_reranker(&mut self, reranker: Arc<dyn RerankerBackend>) {
        self.reranker = Some(reranker);
    }

    /// Set how many candidates to pass to the reranker (default 20).
    pub fn set_rerank_top_n(&mut self, n: usize) {
        self.rerank_top_n = n;
    }

    /// Search using hybrid retrieval (vector + BM25 with RRF).
    ///
    /// Combines vector similarity search with BM25 text search using Reciprocal
    /// Rank Fusion (RRF). BM25 results are matched to vector chunks by file path
    /// and overlapping line ranges, since BM25 returns (path, line) tuples while
    /// vector results use SHA-256 chunk IDs.
    pub fn search(
        &self,
        query: &str,
        k: usize,
        filter: &SearchFilter,
    ) -> Result<Vec<RetrievalResult>> {
        let query_vec = self.embeddings.embed(query)?;

        // Vector search
        let vector_results = self.vector_index.search(&query_vec, k * 2, filter)?;

        // BM25 search: prefer chunk-level index (exact chunk_id matching) over
        // file-level index (requires spatial line-range matching).
        let scope = if filter.languages.len() == 1 {
            Some(filter.languages[0].as_str())
        } else {
            None
        };

        let chunk_bm25_results = if let Some(ref tantivy) = self.tantivy_index {
            tantivy.query_chunks(query, k * 2, scope).ok()
        } else {
            None
        };

        let file_bm25_results = if chunk_bm25_results.is_none() {
            if let Some(ref tantivy) = self.tantivy_index {
                tantivy.query(query, k * 2, scope).ok()
            } else {
                None
            }
        } else {
            None
        };

        // Map chunk_id → BM25 rank
        let mut bm25_rank_by_chunk: HashMap<String, usize> = HashMap::new();
        // Track BM25-only results (not matched by any vector chunk)
        let mut bm25_only_chunks: Vec<(usize, Chunk)> = Vec::new();

        let vector_chunk_ids: std::collections::HashSet<String> =
            vector_results.iter().map(|r| r.chunk.id.clone()).collect();

        if let Some(ref chunk_results) = chunk_bm25_results {
            // Chunk-level BM25: chunk IDs match directly with vector results
            for (bm25_rank, result) in chunk_results.iter().enumerate() {
                if vector_chunk_ids.contains(&result.chunk_id) {
                    bm25_rank_by_chunk
                        .entry(result.chunk_id.clone())
                        .or_insert(bm25_rank + 1);
                } else {
                    // BM25-only chunk (vector search didn't find it)
                    bm25_only_chunks.push((
                        bm25_rank + 1,
                        Chunk {
                            id: result.chunk_id.clone(),
                            file_path: PathBuf::from(&result.path),
                            start_line: result.start_line,
                            end_line: result.end_line,
                            content: String::new(), // Will be populated from Tantivy if needed
                            content_hash: String::new(),
                            language: String::new(),
                        },
                    ));
                }
            }
        } else if let Some(ref file_results) = file_bm25_results {
            // File-level BM25 fallback: match by spatial line-range overlap
            let chunk_spatial_index: Vec<(PathBuf, usize, usize, String)> = vector_results
                .iter()
                .map(|r| {
                    (
                        r.chunk.file_path.clone(),
                        r.chunk.start_line,
                        r.chunk.end_line,
                        r.chunk.id.clone(),
                    )
                })
                .collect();

            for (bm25_rank, result) in file_results.results.iter().enumerate() {
                let bm25_path = PathBuf::from(&result.path);
                let bm25_line = result.line;

                let matched =
                    chunk_spatial_index
                        .iter()
                        .find(|(file_path, start_line, end_line, _)| {
                            path_matches(file_path, &bm25_path)
                                && bm25_line >= *start_line
                                && bm25_line <= *end_line
                        });

                if let Some((_, _, _, chunk_id)) = matched {
                    bm25_rank_by_chunk
                        .entry(chunk_id.clone())
                        .or_insert(bm25_rank + 1);
                } else {
                    bm25_only_chunks.push((
                        bm25_rank + 1,
                        Chunk {
                            id: format!("bm25:{}:{}", result.path, result.line),
                            file_path: PathBuf::from(&result.path),
                            start_line: result.line,
                            end_line: result.line,
                            content: result.excerpt.clone(),
                            content_hash: String::new(),
                            language: String::new(),
                        },
                    ));
                }
            }
        }

        // Reciprocal Rank Fusion
        let k_rrf: f32 = 60.0;
        let mut fused_scores: HashMap<String, (f32, f32, f32)> = HashMap::new();

        for result in &vector_results {
            let chunk_id = &result.chunk.id;
            let vector_rrf = 1.0 / (result.rank as f32 + k_rrf);
            let bm25_rrf = bm25_rank_by_chunk
                .get(chunk_id)
                .map(|rank| 1.0 / (*rank as f32 + k_rrf))
                .unwrap_or(0.0);

            let hybrid = self.blend_alpha * vector_rrf + (1.0 - self.blend_alpha) * bm25_rrf;
            fused_scores.insert(chunk_id.clone(), (result.score, bm25_rrf, hybrid));
        }

        // Build result set from vector results with fused scores
        let mut scored_chunks: Vec<_> = vector_results
            .into_iter()
            .map(|r| {
                let (vs, bs, hs) = fused_scores
                    .get(&r.chunk.id)
                    .copied()
                    .unwrap_or((r.score, 0.0, r.score));
                (r.chunk, vs, bs, hs)
            })
            .collect();

        // Include BM25-only results that vector search missed.
        // These get vector_score=0 and only BM25 contributes to hybrid score.
        for (bm25_rank, chunk) in &bm25_only_chunks {
            let bm25_rrf = 1.0 / (*bm25_rank as f32 + k_rrf);
            let hybrid = (1.0 - self.blend_alpha) * bm25_rrf;
            scored_chunks.push((chunk.clone(), 0.0, bm25_rrf, hybrid));
        }

        // Sort by hybrid score descending
        scored_chunks.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        // Optional cross-encoder reranking pass for highest precision
        if let Some(ref reranker) = self.reranker {
            let rerank_n = self.rerank_top_n.min(scored_chunks.len());
            if rerank_n > 0 {
                let docs: Vec<&str> = scored_chunks[..rerank_n]
                    .iter()
                    .map(|(chunk, _, _, _)| chunk.content.as_str())
                    .collect();

                if let Ok(rerank_scores) = reranker.rerank(query, &docs) {
                    // Re-sort the top-N by cross-encoder scores
                    let mut indexed: Vec<(usize, f32)> =
                        rerank_scores.into_iter().enumerate().collect();
                    indexed
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                    let mut reranked_top: Vec<_> = indexed
                        .into_iter()
                        .map(|(orig_idx, rerank_score)| {
                            let (chunk, vs, bs, _hs) = scored_chunks[orig_idx].clone();
                            (chunk, vs, bs, rerank_score)
                        })
                        .collect();

                    // Append remaining non-reranked results
                    let remaining: Vec<_> = scored_chunks[rerank_n..].to_vec();
                    reranked_top.extend(remaining);
                    scored_chunks = reranked_top;
                }
            }
        }

        scored_chunks.truncate(k);

        let results = scored_chunks
            .into_iter()
            .enumerate()
            .map(|(rank, (chunk, vs, bs, hs))| RetrievalResult {
                chunk,
                vector_score: vs,
                bm25_score: bs,
                hybrid_score: hs,
                rank: rank + 1,
            })
            .collect();

        Ok(results)
    }

    /// Build a full index from workspace.
    pub fn build_index(&mut self, workspace: &Path) -> Result<IndexBuildReport> {
        let start = std::time::Instant::now();

        let chunks = crate::chunker::chunk_workspace(workspace, &self.chunk_config)?;
        let files: std::collections::HashSet<String> = chunks
            .iter()
            .map(|c| c.file_path.to_string_lossy().to_string())
            .collect();

        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        let vectors = self.embeddings.embed_batch(&texts)?;

        for (chunk, vector) in chunks.iter().zip(vectors.iter()) {
            self.vector_index.insert(chunk, vector)?;
        }

        // Build chunk-level Tantivy index for proper BM25↔vector fusion
        if let Some(ref tantivy) = self.tantivy_index {
            let chunk_entries: Vec<codingbuddy_index::ChunkEntry> = chunks
                .iter()
                .map(|c| codingbuddy_index::ChunkEntry {
                    chunk_id: c.id.clone(),
                    path: c.file_path.to_string_lossy().to_string(),
                    content: c.content.clone(),
                    start_line: c.start_line,
                    end_line: c.end_line,
                })
                .collect();
            // Best-effort: chunk-level index is an enhancement, not required
            let _ = tantivy.build_chunk_index(&chunk_entries);
        }

        let report = IndexBuildReport {
            chunks_indexed: chunks.len(),
            files_processed: files.len(),
            duration_ms: start.elapsed().as_millis() as u64,
        };

        Ok(report)
    }

    /// Update the index incrementally (only changed files).
    pub fn update_index(&mut self, workspace: &Path) -> Result<IndexUpdateReport> {
        let start = std::time::Instant::now();

        let (new_chunks, new_manifest) = crate::chunker::chunk_workspace_incremental(
            workspace,
            &self.chunk_config,
            &self.manifest,
        )?;

        // Find removed files
        let removed_files: Vec<String> = self
            .manifest
            .file_hashes
            .keys()
            .filter(|f| !new_manifest.file_hashes.contains_key(*f))
            .cloned()
            .collect();

        let files_changed = new_chunks
            .iter()
            .map(|c| c.file_path.to_string_lossy().to_string())
            .collect::<std::collections::HashSet<_>>()
            .len()
            + removed_files.len();

        // Index new/changed chunks
        let texts: Vec<&str> = new_chunks.iter().map(|c| c.content.as_str()).collect();
        let vectors = self.embeddings.embed_batch(&texts)?;
        for (chunk, vector) in new_chunks.iter().zip(vectors.iter()) {
            self.vector_index.insert(chunk, vector)?;
        }

        self.manifest = new_manifest;

        let report = IndexUpdateReport {
            chunks_added: new_chunks.len(),
            chunks_removed: removed_files.len(),
            files_changed,
            duration_ms: start.elapsed().as_millis() as u64,
        };

        Ok(report)
    }
}

/// Check if two paths refer to the same file, handling relative vs absolute paths.
///
/// BM25 results from Tantivy use workspace-relative paths while vector chunks
/// may use absolute paths. This function normalizes by comparing file names and
/// checking suffix containment.
fn path_matches(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    // Check if one is a suffix of the other (handles relative vs absolute)
    a.ends_with(b) || b.ends_with(a)
}

/// Reciprocal Rank Fusion: combines two ranked lists into a single fused ranking.
///
/// For each item, the fused score is:
///   score = alpha * (1 / (rank_vector + k)) + (1 - alpha) * (1 / (rank_bm25 + k))
///
/// where k (default 60) is a smoothing constant that reduces the impact of high rankings.
pub fn reciprocal_rank_fusion(
    vector_results: &[(String, f32)],
    bm25_results: &[(String, f32)],
    alpha: f32,
    k_rrf: usize,
) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();

    for (rank, (id, _)) in vector_results.iter().enumerate() {
        let rrf_score = alpha / (rank as f32 + k_rrf as f32);
        *scores.entry(id.clone()).or_default() += rrf_score;
    }

    for (rank, (id, _)) in bm25_results.iter().enumerate() {
        let rrf_score = (1.0 - alpha) / (rank as f32 + k_rrf as f32);
        *scores.entry(id.clone()).or_default() += rrf_score;
    }

    let mut fused: Vec<(String, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::MockEmbeddings;

    #[test]
    fn path_matches_exact() {
        assert!(path_matches(
            Path::new("src/main.rs"),
            Path::new("src/main.rs")
        ));
    }

    #[test]
    fn path_matches_relative_vs_absolute() {
        assert!(path_matches(
            Path::new("/home/user/project/src/main.rs"),
            Path::new("src/main.rs")
        ));
        assert!(path_matches(
            Path::new("src/main.rs"),
            Path::new("/home/user/project/src/main.rs")
        ));
    }

    #[test]
    fn path_matches_different_files() {
        assert!(!path_matches(
            Path::new("src/main.rs"),
            Path::new("src/lib.rs")
        ));
    }

    #[test]
    fn rrf_fuses_both_sources() {
        let vec_results = vec![("a".to_string(), 0.9), ("b".to_string(), 0.8)];
        let bm25_results = vec![("b".to_string(), 5.0), ("c".to_string(), 3.0)];
        let fused = reciprocal_rank_fusion(&vec_results, &bm25_results, 0.7, 60);
        // "b" should rank highest because it appears in both lists
        assert_eq!(fused[0].0, "b");
        // "c" is BM25-only, should still appear
        assert!(fused.iter().any(|(id, _)| id == "c"));
    }

    #[test]
    fn hybrid_search_with_bm25_contributes_score() {
        // Set up a retriever with mock embeddings and a tantivy index
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();

        // Create a test file
        std::fs::write(
            workspace.join("util.rs"),
            "fn calculate_score(x: i32) -> i32 {\n    x * 2\n}\n\nfn helper() {\n    println!(\"helper\");\n}\n",
        )
        .unwrap();

        // Build tantivy index
        let session = codingbuddy_core::Session {
            session_id: uuid::Uuid::now_v7(),
            workspace_root: workspace.to_string_lossy().to_string(),
            baseline_commit: None,
            status: codingbuddy_core::SessionState::Idle,
            budgets: codingbuddy_core::SessionBudgets {
                per_turn_seconds: 10,
                max_think_tokens: 1000,
            },
            active_plan_id: None,
        };
        let tantivy = codingbuddy_index::IndexService::new(&workspace).unwrap();
        tantivy.build(&session).unwrap();

        // Create retriever with tantivy
        let embeddings = Arc::new(MockEmbeddings::new(64));
        let index_path = tmp.path().join("index");
        let mut retriever = HybridRetriever::new(
            &index_path,
            embeddings,
            Some(tantivy),
            0.7, // 70% vector, 30% BM25
            ChunkConfig {
                chunk_lines: 10,
                chunk_overlap: 2,
                ..Default::default()
            },
        )
        .unwrap();

        // Build the vector index
        retriever.build_index(&workspace).unwrap();

        // Search for something that both vector and BM25 can find
        let results = retriever
            .search("calculate_score", 5, &SearchFilter::default())
            .unwrap();

        assert!(!results.is_empty(), "should find results");

        // At least one result should have non-zero bm25_score
        // (proving BM25 actually contributes now)
        let has_bm25 = results.iter().any(|r| r.bm25_score > 0.0);
        assert!(
            has_bm25,
            "BM25 should contribute non-zero scores. Scores: {:?}",
            results
                .iter()
                .map(|r| (
                    &r.chunk.file_path,
                    r.vector_score,
                    r.bm25_score,
                    r.hybrid_score
                ))
                .collect::<Vec<_>>()
        );
    }
}
