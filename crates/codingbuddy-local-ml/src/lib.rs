pub mod chunker;
pub mod completion;
pub mod embeddings;
pub mod hardware;
mod model_manager;
pub mod model_registry;
pub mod privacy;
pub mod reranker;
pub mod retrieval;
pub mod speculative;
pub mod vector_index;

pub use chunker::{Chunk, ChunkConfig, ChunkManifest, ChunkStrategy, chunk_workspace_metadata};
pub use completion::{GenOpts, LocalGenBackend, MockGenerator};
pub use embeddings::{EmbeddingsBackend, MockEmbeddings};
pub use model_manager::{ModelInfo, ModelManager, ModelManifest, ModelStatus};
pub use privacy::{PrivacyConfig, PrivacyPolicy, PrivacyResult, PrivacyRouter, SensitiveMatch};
pub use reranker::{MockReranker, RerankerBackend};
pub use retrieval::{HybridRetriever, IndexBuildReport, IndexUpdateReport, RetrievalResult};
pub use vector_index::{
    BruteForceBackend, IndexStats, SearchFilter, SearchResult, VectorIndex, VectorIndexBackend,
};

#[cfg(feature = "local-ml")]
pub use completion::candle_backend::CandleCompletion;
#[cfg(feature = "local-ml")]
pub use embeddings::candle_backend::CandleEmbeddings;
#[cfg(feature = "local-ml")]
pub use reranker::candle_backend::CandleReranker;
#[cfg(feature = "local-ml")]
pub use vector_index::UsearchBackend;

/// Parse a device string ("cpu", "cuda", "metal") into a candle Device.
#[cfg(feature = "local-ml")]
pub fn parse_device(s: &str) -> candle_core::Device {
    match s {
        "metal" => candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu),
        "cuda" => candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu),
        _ => candle_core::Device::Cpu,
    }
}

/// Resolve a device string to a candle Device with logging.
///
/// Handles `"auto"` by detecting hardware, and logs fallbacks on init failure.
#[cfg(feature = "local-ml")]
pub fn resolve_device(requested: &str) -> (candle_core::Device, hardware::DetectedDevice) {
    let target = if requested == "auto" {
        let hw = hardware::detect_hardware();
        eprintln!(
            "[codingbuddy] auto-detected device: {}, RAM: {} MB (available for models: {} MB)",
            hw.device, hw.total_ram_mb, hw.available_for_models_mb
        );
        hw.device
    } else {
        match requested {
            "metal" => hardware::DetectedDevice::Metal,
            "cuda" => hardware::DetectedDevice::Cuda,
            _ => hardware::DetectedDevice::Cpu,
        }
    };

    match target {
        hardware::DetectedDevice::Metal => match candle_core::Device::new_metal(0) {
            Ok(d) => (d, hardware::DetectedDevice::Metal),
            Err(e) => {
                eprintln!("[codingbuddy] Metal init failed ({e}), falling back to CPU");
                (candle_core::Device::Cpu, hardware::DetectedDevice::Cpu)
            }
        },
        hardware::DetectedDevice::Cuda => match candle_core::Device::new_cuda(0) {
            Ok(d) => (d, hardware::DetectedDevice::Cuda),
            Err(e) => {
                eprintln!("[codingbuddy] CUDA init failed ({e}), falling back to CPU");
                (candle_core::Device::Cpu, hardware::DetectedDevice::Cpu)
            }
        },
        hardware::DetectedDevice::Cpu => (candle_core::Device::Cpu, hardware::DetectedDevice::Cpu),
    }
}

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
        assert_eq!(
            v.len(),
            dim,
            "vector length must equal configured dimension"
        );
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
