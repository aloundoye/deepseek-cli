use anyhow::Result;
use rusqlite::Connection;
use std::path::{Path, PathBuf};

use crate::chunker::Chunk;

/// Backend trait for vector similarity search.
pub trait VectorIndexBackend: Send + Sync {
    /// Insert a vector with its chunk ID.
    fn insert(&mut self, chunk_id: &str, vector: &[f32]) -> Result<()>;

    /// Remove a vector by chunk ID.
    fn remove(&mut self, chunk_id: &str) -> Result<()>;

    /// Search for the top-k nearest vectors to the query.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>>;

    /// Get index statistics.
    fn stats(&self) -> IndexStats;
}

/// Filter criteria for vector search.
#[derive(Debug, Clone, Default)]
pub struct SearchFilter {
    pub file_globs: Vec<String>,
    pub languages: Vec<String>,
    pub max_results: usize,
}

/// A single search result with chunk and score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub score: f32,
    pub rank: usize,
}

/// Statistics about the vector index.
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    pub chunk_count: usize,
    pub dimension: usize,
    pub file_count: usize,
    pub size_bytes: u64,
}

/// Vector index combining a vector backend with SQLite metadata storage.
pub struct VectorIndex {
    backend: Box<dyn VectorIndexBackend>,
    db: Connection,
    db_path: PathBuf,
}

impl VectorIndex {
    /// Create a new vector index at the given path.
    pub fn create(path: &Path, backend: Box<dyn VectorIndexBackend>) -> Result<Self> {
        std::fs::create_dir_all(path)?;
        let db_path = path.join("chunks.db");
        let db = Connection::open(&db_path)?;

        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                language TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_lang ON chunks(language);",
        )?;

        Ok(Self {
            backend,
            db,
            db_path,
        })
    }

    /// Open an existing vector index.
    pub fn open(path: &Path, backend: Box<dyn VectorIndexBackend>) -> Result<Self> {
        let db_path = path.join("chunks.db");
        let db = Connection::open(&db_path)?;
        Ok(Self {
            backend,
            db,
            db_path,
        })
    }

    /// Insert a chunk and its embedding vector.
    pub fn insert(&mut self, chunk: &Chunk, vector: &[f32]) -> Result<()> {
        self.db.execute(
            "INSERT OR REPLACE INTO chunks (chunk_id, file_path, start_line, end_line, content, content_hash, language)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                chunk.id,
                chunk.file_path.to_string_lossy().to_string(),
                chunk.start_line as i64,
                chunk.end_line as i64,
                chunk.content,
                chunk.content_hash,
                chunk.language,
            ],
        )?;
        self.backend.insert(&chunk.id, vector)?;
        Ok(())
    }

    /// Remove a chunk by ID.
    pub fn remove(&mut self, chunk_id: &str) -> Result<()> {
        self.db
            .execute("DELETE FROM chunks WHERE chunk_id = ?1", [chunk_id])?;
        self.backend.remove(chunk_id)?;
        Ok(())
    }

    /// Search for similar chunks, with optional filtering.
    pub fn search(&self, query: &[f32], k: usize, filter: &SearchFilter) -> Result<Vec<SearchResult>> {
        let max_k = if filter.max_results > 0 {
            filter.max_results
        } else {
            k
        };

        // Over-fetch from vector backend to account for filtering
        let fetch_k = max_k * 3;
        let candidates = self.backend.search(query, fetch_k)?;

        let mut results = Vec::new();
        let mut rank = 1;

        for (chunk_id, score) in candidates {
            if results.len() >= max_k {
                break;
            }

            // Look up chunk metadata
            let chunk = self.get_chunk(&chunk_id)?;
            if let Some(chunk) = chunk {
                // Apply filters
                if !filter.languages.is_empty()
                    && !filter.languages.contains(&chunk.language)
                {
                    continue;
                }
                if !filter.file_globs.is_empty() {
                    let path_str = chunk.file_path.to_string_lossy();
                    let matches = filter.file_globs.iter().any(|g| {
                        glob::Pattern::new(g)
                            .map(|p| p.matches(&path_str))
                            .unwrap_or(false)
                    });
                    if !matches {
                        continue;
                    }
                }

                results.push(SearchResult { chunk, score, rank });
                rank += 1;
            }
        }

        Ok(results)
    }

    /// Get index statistics.
    pub fn stats(&self) -> Result<IndexStats> {
        let mut backend_stats = self.backend.stats();

        let chunk_count: i64 = self.db.query_row(
            "SELECT COUNT(*) FROM chunks",
            [],
            |row| row.get(0),
        )?;

        let file_count: i64 = self.db.query_row(
            "SELECT COUNT(DISTINCT file_path) FROM chunks",
            [],
            |row| row.get(0),
        )?;

        let size_bytes = if self.db_path.exists() {
            std::fs::metadata(&self.db_path)
                .map(|m| m.len())
                .unwrap_or(0)
        } else {
            0
        };

        backend_stats.chunk_count = chunk_count as usize;
        backend_stats.file_count = file_count as usize;
        backend_stats.size_bytes = size_bytes;

        Ok(backend_stats)
    }

    fn get_chunk(&self, chunk_id: &str) -> Result<Option<Chunk>> {
        let mut stmt = self.db.prepare(
            "SELECT chunk_id, file_path, start_line, end_line, content, content_hash, language
             FROM chunks WHERE chunk_id = ?1",
        )?;

        let result = stmt.query_row([chunk_id], |row| {
            Ok(Chunk {
                id: row.get(0)?,
                file_path: PathBuf::from(row.get::<_, String>(1)?),
                start_line: row.get::<_, i64>(2)? as usize,
                end_line: row.get::<_, i64>(3)? as usize,
                content: row.get(4)?,
                content_hash: row.get(5)?,
                language: row.get(6)?,
            })
        });

        match result {
            Ok(chunk) => Ok(Some(chunk)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

/// Brute-force O(n) vector search backend. Suitable for < 10K chunks.
///
/// Stores all vectors in memory and computes cosine similarity on each search.
/// No external dependencies needed.
pub struct BruteForceBackend {
    vectors: Vec<(String, Vec<f32>)>,
    dimension: usize,
}

impl BruteForceBackend {
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dimension,
        }
    }
}

impl VectorIndexBackend for BruteForceBackend {
    fn insert(&mut self, chunk_id: &str, vector: &[f32]) -> Result<()> {
        // Remove existing entry if present
        self.vectors.retain(|(id, _)| id != chunk_id);
        self.vectors.push((chunk_id.to_string(), vector.to_vec()));
        Ok(())
    }

    fn remove(&mut self, chunk_id: &str) -> Result<()> {
        self.vectors.retain(|(id, _)| id != chunk_id);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        let mut scored: Vec<(String, f32)> = self
            .vectors
            .iter()
            .map(|(id, vec)| {
                let sim = cosine_similarity(query, vec);
                (id.clone(), sim)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }

    fn stats(&self) -> IndexStats {
        IndexStats {
            chunk_count: self.vectors.len(),
            dimension: self.dimension,
            file_count: 0, // Filled by VectorIndex
            size_bytes: 0,
        }
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(feature = "local-ml")]
pub struct UsearchBackend {
    // Placeholder for usearch::Index wrapper
    _dimension: usize,
}
