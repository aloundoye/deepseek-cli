pub mod chunker;
pub mod completion;
pub mod embeddings;
mod model_manager;
pub mod privacy;
pub mod retrieval;
pub mod vector_index;

pub use chunker::{Chunk, ChunkConfig, ChunkManifest};
pub use completion::{GenOpts, LocalGenBackend, MockGenerator};
pub use embeddings::{EmbeddingsBackend, MockEmbeddings};
pub use model_manager::{ModelInfo, ModelManager, ModelStatus};
pub use privacy::{PrivacyConfig, PrivacyPolicy, PrivacyResult, PrivacyRouter, SensitiveMatch};
pub use retrieval::{HybridRetriever, IndexBuildReport, IndexUpdateReport, RetrievalResult};
pub use vector_index::{
    BruteForceBackend, IndexStats, SearchFilter, SearchResult, VectorIndex, VectorIndexBackend,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_embeddings_deterministic() {
        let emb = MockEmbeddings::new(128);
        let v1 = emb.embed("hello world").unwrap();
        let v2 = emb.embed("hello world").unwrap();
        assert_eq!(v1, v2, "same input must produce identical vectors");
    }

    #[test]
    fn mock_embeddings_correct_dimension() {
        let dim = 384;
        let emb = MockEmbeddings::new(dim);
        let v = emb.embed("test input").unwrap();
        assert_eq!(v.len(), dim, "vector length must equal configured dimension");
    }

    #[test]
    fn mock_generator_returns_output() {
        let generator = MockGenerator::new("fixed response".to_string());
        let opts = GenOpts::default();
        let result = generator.generate("prompt", &opts).unwrap();
        assert_eq!(result, "fixed response");
    }

    #[test]
    fn model_manager_status_not_downloaded() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ModelManager::new(dir.path().to_path_buf());
        let status = mgr.status("all-MiniLM-L6-v2");
        assert!(
            matches!(status, ModelStatus::NotDownloaded),
            "new model must be NotDownloaded"
        );
    }

    #[test]
    fn embeddings_batch_default_impl() {
        let emb = MockEmbeddings::new(64);
        let inputs = vec!["hello", "world", "test"];
        let batch = emb.embed_batch(&inputs).unwrap();
        assert_eq!(batch.len(), 3);
        // Each result should match individual embed
        for (i, input) in inputs.iter().enumerate() {
            let single = emb.embed(input).unwrap();
            assert_eq!(batch[i], single, "batch[{}] must match individual embed", i);
        }
    }
}
