use anyhow::Result;
use sha2::{Digest, Sha256};

/// Backend trait for computing text embeddings.
///
/// Implementations convert text into dense floating-point vectors suitable for
/// semantic similarity search. The trait is object-safe so backends can be
/// swapped at runtime (mock for testing, Candle for production).
pub trait EmbeddingsBackend: Send + Sync {
    /// Embed a single text input into a dense vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed a batch of text inputs. Default implementation calls `embed` per item.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Dimensionality of the output vectors.
    fn dimension(&self) -> usize;

    /// Model identifier string (e.g. "all-MiniLM-L6-v2").
    fn model_id(&self) -> &str;
}

/// Deterministic mock embeddings backend for testing.
///
/// Produces vectors by hashing the input text with SHA-256, then distributing
/// the hash bytes across the configured dimension. This guarantees:
/// - Same input always produces the same vector (deterministic).
/// - Different inputs produce different vectors (collision-resistant).
/// - No ML dependencies required.
pub struct MockEmbeddings {
    dimension: usize,
}

impl MockEmbeddings {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl EmbeddingsBackend for MockEmbeddings {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let hash = Sha256::digest(text.as_bytes());
        let hash_bytes = hash.as_slice();
        let mut vector = Vec::with_capacity(self.dimension);
        for i in 0..self.dimension {
            // Cycle through hash bytes and normalize to [-1, 1]
            let byte = hash_bytes[i % hash_bytes.len()] as f32;
            vector.push((byte / 127.5) - 1.0);
        }
        // L2-normalize for consistent cosine similarity behavior
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vector {
                *v /= norm;
            }
        }
        Ok(vector)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_id(&self) -> &str {
        "mock-embeddings"
    }
}

#[cfg(feature = "local-ml")]
pub mod candle_backend;
