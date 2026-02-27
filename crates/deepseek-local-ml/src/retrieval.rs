use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::chunker::{Chunk, ChunkConfig, ChunkManifest};
use crate::embeddings::EmbeddingsBackend;
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
    tantivy_index: Option<deepseek_index::IndexService>,
    blend_alpha: f32,
    chunk_config: ChunkConfig,
    manifest: ChunkManifest,
}

impl HybridRetriever {
    /// Create a new hybrid retriever.
    ///
    /// - `blend_alpha`: 0.0 = pure BM25, 1.0 = pure vector search
    pub fn new(
        index_path: &Path,
        embeddings: Arc<dyn EmbeddingsBackend>,
        tantivy_index: Option<deepseek_index::IndexService>,
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
            blend_alpha,
            chunk_config,
            manifest: ChunkManifest::default(),
        })
    }

    /// Create a hybrid retriever with a custom vector index backend.
    pub fn new_with_backend(
        index_path: &Path,
        embeddings: Arc<dyn EmbeddingsBackend>,
        backend: Box<dyn VectorIndexBackend>,
        tantivy_index: Option<deepseek_index::IndexService>,
        blend_alpha: f32,
        chunk_config: ChunkConfig,
    ) -> Result<Self> {
        let vector_index = VectorIndex::create(index_path, backend)?;

        Ok(Self {
            vector_index,
            embeddings,
            tantivy_index,
            blend_alpha,
            chunk_config,
            manifest: ChunkManifest::default(),
        })
    }

    /// Search using hybrid retrieval (vector + BM25 with RRF).
    pub fn search(
        &self,
        query: &str,
        k: usize,
        filter: &SearchFilter,
    ) -> Result<Vec<RetrievalResult>> {
        let query_vec = self.embeddings.embed(query)?;

        // Vector search
        let vector_results = self.vector_index.search(&query_vec, k * 2, filter)?;

        // BM25 search via tantivy (if available)
        let bm25_results = if let Some(ref tantivy) = self.tantivy_index {
            let scope = if filter.languages.len() == 1 {
                Some(filter.languages[0].as_str())
            } else {
                None
            };
            tantivy.query(query, k * 2, scope).ok()
        } else {
            None
        };

        // Build vector score map
        let mut vector_scores: HashMap<String, (f32, usize)> = HashMap::new();
        for (rank, result) in vector_results.iter().enumerate() {
            vector_scores.insert(result.chunk.id.clone(), (result.score, rank + 1));
        }

        // Build BM25 score map (if available)
        let mut bm25_scores: HashMap<String, (f32, usize)> = HashMap::new();
        if let Some(ref bm25) = bm25_results {
            for (rank, result) in bm25.results.iter().enumerate() {
                // Create a pseudo chunk ID from path + line for matching
                let pseudo_id = format!("{}:{}", result.path, result.line);
                bm25_scores.insert(pseudo_id, (1.0 / (rank as f32 + 1.0), rank + 1));
            }
        }

        // Reciprocal Rank Fusion
        let k_rrf = 60;
        let mut fused_scores: HashMap<String, (f32, f32, f32)> = HashMap::new();

        for result in &vector_results {
            let chunk_id = &result.chunk.id;
            let vector_rrf = 1.0 / (result.rank as f32 + k_rrf as f32);
            let bm25_rrf = bm25_scores
                .get(chunk_id)
                .map(|(_, rank)| 1.0 / (*rank as f32 + k_rrf as f32))
                .unwrap_or(0.0);

            let hybrid = self.blend_alpha * vector_rrf + (1.0 - self.blend_alpha) * bm25_rrf;
            fused_scores.insert(chunk_id.clone(), (result.score, bm25_rrf, hybrid));
        }

        // Sort by hybrid score descending
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

        scored_chunks.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
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
