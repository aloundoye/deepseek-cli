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

/// Strategy for loading a model based on available system memory.
///
/// Returned by [`determine_load_strategy`] to guide callers on how (or whether)
/// to proceed with model loading given current memory pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadStrategy {
    /// Comfortable margin: available >= 2x model size. Load at full context.
    Full,
    /// Tight fit: available >= model size but < 2x. Halve the context window.
    ReducedContext,
    /// Very tight: available >= half model size but < model size. CPU-only mode.
    CpuOnly,
    /// Insufficient memory: fall back to mock / API.
    Skip,
}

/// Determine the safest model-loading strategy given the model's on-disk size
/// and the current available system memory.
///
/// Thresholds:
/// - `available >= model_size * 2` -> [`LoadStrategy::Full`]
/// - `available >= model_size`     -> [`LoadStrategy::ReducedContext`]
/// - `available >= model_size / 2` -> [`LoadStrategy::CpuOnly`]
/// - otherwise                     -> [`LoadStrategy::Skip`]
pub fn determine_load_strategy(model_size_mb: u64, available_mb: u64) -> LoadStrategy {
    if available_mb >= model_size_mb.saturating_mul(2) {
        LoadStrategy::Full
    } else if available_mb >= model_size_mb {
        LoadStrategy::ReducedContext
    } else if available_mb >= model_size_mb / 2 {
        LoadStrategy::CpuOnly
    } else {
        LoadStrategy::Skip
    }
}

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
    fn test_load_strategy_full() {
        // Plenty of memory (2x or more) -> Full
        assert_eq!(
            determine_load_strategy(1000, 2000),
            LoadStrategy::Full,
            "exactly 2x should be Full"
        );
        assert_eq!(
            determine_load_strategy(1000, 5000),
            LoadStrategy::Full,
            "5x should be Full"
        );
    }

    #[test]
    fn test_load_strategy_reduced() {
        // Tight but >= model size -> ReducedContext
        assert_eq!(
            determine_load_strategy(1000, 1000),
            LoadStrategy::ReducedContext,
            "exactly 1x should be ReducedContext"
        );
        assert_eq!(
            determine_load_strategy(1000, 1500),
            LoadStrategy::ReducedContext,
            "1.5x should be ReducedContext"
        );
        assert_eq!(
            determine_load_strategy(1000, 1999),
            LoadStrategy::ReducedContext,
            "just under 2x should be ReducedContext"
        );
    }

    #[test]
    fn test_load_strategy_cpu() {
        // Very tight: >= half but < model size -> CpuOnly
        assert_eq!(
            determine_load_strategy(1000, 500),
            LoadStrategy::CpuOnly,
            "exactly half should be CpuOnly"
        );
        assert_eq!(
            determine_load_strategy(1000, 999),
            LoadStrategy::CpuOnly,
            "just under model size should be CpuOnly"
        );
    }

    #[test]
    fn test_load_strategy_skip() {
        // Insufficient memory -> Skip
        assert_eq!(
            determine_load_strategy(1000, 499),
            LoadStrategy::Skip,
            "under half should be Skip"
        );
        assert_eq!(
            determine_load_strategy(1000, 0),
            LoadStrategy::Skip,
            "zero memory should be Skip"
        );
    }

    #[test]
    fn test_load_strategy_edge_cases() {
        // Zero-size model: any memory is fine
        assert_eq!(
            determine_load_strategy(0, 0),
            LoadStrategy::Full,
            "zero model + zero mem should be Full (0 >= 0*2)"
        );
        assert_eq!(
            determine_load_strategy(0, 100),
            LoadStrategy::Full,
            "zero model + any mem should be Full"
        );
        // Very small model
        assert_eq!(
            determine_load_strategy(1, 2),
            LoadStrategy::Full,
            "tiny model with 2x mem should be Full"
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
