//! Model registry — catalogs supported model architectures for local inference.
//!
//! Provides metadata about each model family so the system can select the right
//! loading path, tokenizer, and generation strategy.

/// Architecture family for completion models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionArchitecture {
    /// DeepSeek-Coder family (GGUF, LLaMA-based)
    CodingBuddyCoder,
    /// Qwen2.5-Coder family (GGUF, LLaMA-based)
    Qwen2Coder,
    /// Phi-3 family (GGUF, LLaMA-based)
    Phi3,
    /// TinyLLaMA (GGUF)
    TinyLlama,
    /// Generic GGUF quantized LLaMA-compatible model
    GenericGguf,
}

/// Architecture family for embedding models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingArchitecture {
    /// Sentence-Transformers (BERT-based, SafeTensors)
    SentenceTransformers,
    /// BGE family (BERT-based, 768d)
    Bge,
    /// Jina Code Embeddings (BERT-based, 768d, code-optimized)
    JinaCode,
}

/// Metadata about a supported model.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    /// Internal model identifier (used for cache directory naming).
    pub model_id: &'static str,
    /// HuggingFace repository to download from.
    pub hf_repo: &'static str,
    /// Human-readable name.
    pub display_name: &'static str,
    /// Parameter count (approximate, in billions).
    pub params_b: f32,
    /// Whether this is a code-specialized model.
    pub code_specialized: bool,
    /// Context window size in tokens.
    pub context_tokens: u32,
    /// Expected output quality tier (1=basic, 2=good, 3=excellent).
    pub quality_tier: u8,
    /// GGUF weight filename for quantized completion models. `None` for SafeTensors models.
    pub gguf_filename: Option<&'static str>,
    /// Estimated VRAM/RAM required in MB (Q4_K_M quantization).
    pub estimated_vram_mb: u32,
}

/// Default files to download for SafeTensors (non-GGUF) models.
pub const DEFAULT_SAFETENSORS_FILES: &[&str] =
    &["config.json", "tokenizer.json", "model.safetensors"];

impl ModelEntry {
    /// Returns the list of files to download from HuggingFace.
    pub fn download_files(&self) -> Vec<&str> {
        if let Some(gguf) = self.gguf_filename {
            vec![gguf, "tokenizer.json"]
        } else {
            DEFAULT_SAFETENSORS_FILES.to_vec()
        }
    }
}

/// Registry of supported completion models (ordered by quality tier descending).
pub const COMPLETION_MODELS: &[ModelEntry] = &[
    ModelEntry {
        model_id: "qwen2.5-coder-7b",
        hf_repo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        display_name: "Qwen2.5 Coder 7B",
        params_b: 7.0,
        code_specialized: true,
        context_tokens: 32768,
        quality_tier: 3,
        gguf_filename: Some("qwen2.5-coder-7b-instruct-q4_k_m.gguf"),
        estimated_vram_mb: 4800,
    },
    ModelEntry {
        model_id: "qwen2.5-coder-3b",
        hf_repo: "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        display_name: "Qwen2.5 Coder 3B",
        params_b: 3.0,
        code_specialized: true,
        context_tokens: 32768,
        quality_tier: 3,
        gguf_filename: Some("qwen2.5-coder-3b-instruct-q4_k_m.gguf"),
        estimated_vram_mb: 2200,
    },
    ModelEntry {
        model_id: "qwen2.5-coder-1.5b",
        hf_repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        display_name: "Qwen2.5 Coder 1.5B",
        params_b: 1.5,
        code_specialized: true,
        context_tokens: 32768,
        quality_tier: 2,
        gguf_filename: Some("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
        estimated_vram_mb: 1100,
    },
    ModelEntry {
        model_id: "phi-3-mini-4k",
        hf_repo: "microsoft/Phi-3-mini-4k-instruct-gguf",
        display_name: "Phi-3 Mini 4K",
        params_b: 3.8,
        code_specialized: false,
        context_tokens: 4096,
        quality_tier: 3,
        gguf_filename: Some("Phi-3-mini-4k-instruct-q4.gguf"),
        estimated_vram_mb: 2600,
    },
    ModelEntry {
        model_id: "tinyllama-1.1b-chat",
        hf_repo: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        display_name: "TinyLLaMA 1.1B Chat",
        params_b: 1.1,
        code_specialized: false,
        context_tokens: 2048,
        quality_tier: 1,
        gguf_filename: Some("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
        estimated_vram_mb: 800,
    },
];

/// Registry of supported embedding models.
pub const EMBEDDING_MODELS: &[ModelEntry] = &[
    ModelEntry {
        model_id: "sentence-transformers/all-MiniLM-L6-v2",
        hf_repo: "sentence-transformers/all-MiniLM-L6-v2",
        display_name: "MiniLM L6 v2 (384d)",
        params_b: 0.022,
        code_specialized: false,
        context_tokens: 512,
        quality_tier: 2,
        gguf_filename: None,
        estimated_vram_mb: 100,
    },
    ModelEntry {
        model_id: "BAAI/bge-base-en-v1.5",
        hf_repo: "BAAI/bge-base-en-v1.5",
        display_name: "BGE Base EN v1.5 (768d)",
        params_b: 0.110,
        code_specialized: false,
        context_tokens: 512,
        quality_tier: 3,
        gguf_filename: None,
        estimated_vram_mb: 450,
    },
    ModelEntry {
        model_id: "jinaai/jina-embeddings-v2-base-code",
        hf_repo: "jinaai/jina-embeddings-v2-base-code",
        display_name: "Jina Code v2 (768d)",
        params_b: 0.137,
        code_specialized: true,
        context_tokens: 8192,
        quality_tier: 3,
        gguf_filename: None,
        estimated_vram_mb: 550,
    },
];

/// Look up a completion model by ID.
pub fn find_completion_model(model_id: &str) -> Option<&'static ModelEntry> {
    COMPLETION_MODELS.iter().find(|m| m.model_id == model_id)
}

/// Look up an embedding model by ID.
pub fn find_embedding_model(model_id: &str) -> Option<&'static ModelEntry> {
    EMBEDDING_MODELS.iter().find(|m| m.model_id == model_id)
}

/// Returns the recommended default embedding model for local retrieval.
pub fn default_embedding_model() -> &'static ModelEntry {
    // Jina Code v2 — code-optimized, high quality, long context
    &EMBEDDING_MODELS[2]
}

/// Returns the recommended default completion model for local ghost text.
pub fn default_completion_model() -> &'static ModelEntry {
    find_completion_model("qwen2.5-coder-3b")
        .expect("qwen2.5-coder-3b must exist in COMPLETION_MODELS")
}

/// Recommend the best completion model that fits within `available_mb` of RAM/VRAM.
///
/// Returns `None` if no model fits (caller should disable local completion).
/// Prefers code-specialized models and higher quality tiers.
pub fn recommend_completion_model(available_mb: u64) -> Option<&'static ModelEntry> {
    // COMPLETION_MODELS is ordered by quality (best first), so the first
    // code-specialized model that fits is the best choice.
    COMPLETION_MODELS
        .iter()
        .find(|m| m.code_specialized && u64::from(m.estimated_vram_mb) <= available_mb)
        .or_else(|| {
            // Fall back to any model that fits
            COMPLETION_MODELS
                .iter()
                .find(|m| u64::from(m.estimated_vram_mb) <= available_mb)
        })
}

/// Check if a model fits within the available memory before loading.
///
/// Returns `Ok(())` if the model fits, or `Err` with a suggestion if it doesn't.
pub fn check_model_fits(model_id: &str, available_mb: u64) -> Result<(), String> {
    let Some(entry) = find_completion_model(model_id) else {
        // Unknown model — can't check, let it proceed
        return Ok(());
    };
    let required = u64::from(entry.estimated_vram_mb);
    if required <= available_mb {
        return Ok(());
    }
    let suggestion = recommend_completion_model(available_mb)
        .map(|m| format!(" Try '{}' instead.", m.model_id))
        .unwrap_or_default();
    Err(format!(
        "model '{}' requires ~{} MB but only {} MB available.{}",
        model_id, required, available_mb, suggestion
    ))
}

/// Detect the architecture from a model ID.
#[cfg(any(test, feature = "local-ml"))]
pub fn detect_completion_architecture(model_id: &str) -> CompletionArchitecture {
    let lower = model_id.to_ascii_lowercase();
    if lower.contains("deepseek") {
        CompletionArchitecture::CodingBuddyCoder
    } else if lower.contains("qwen") {
        CompletionArchitecture::Qwen2Coder
    } else if lower.contains("phi") {
        CompletionArchitecture::Phi3
    } else if lower.contains("tinyllama") {
        CompletionArchitecture::TinyLlama
    } else {
        CompletionArchitecture::GenericGguf
    }
}

/// Detect the embedding architecture from a model ID.
#[cfg(any(test, feature = "local-ml"))]
pub fn detect_embedding_architecture(model_id: &str) -> EmbeddingArchitecture {
    let lower = model_id.to_ascii_lowercase();
    if lower.contains("bge") {
        EmbeddingArchitecture::Bge
    } else if lower.contains("jina") {
        EmbeddingArchitecture::JinaCode
    } else {
        EmbeddingArchitecture::SentenceTransformers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_known_completion_model() {
        let entry = find_completion_model("qwen2.5-coder-3b");
        assert!(entry.is_some());
        assert!(entry.unwrap().code_specialized);
    }

    #[test]
    fn find_known_embedding_model() {
        let entry = find_embedding_model("sentence-transformers/all-MiniLM-L6-v2");
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().params_b, 0.022);
    }

    #[test]
    fn detect_codingbuddy_architecture() {
        assert_eq!(
            detect_completion_architecture("deepseek-coder-6.7b"),
            CompletionArchitecture::CodingBuddyCoder
        );
    }

    #[test]
    fn detect_qwen_architecture() {
        assert_eq!(
            detect_completion_architecture("qwen2.5-coder-7b"),
            CompletionArchitecture::Qwen2Coder
        );
    }

    #[test]
    fn detect_unknown_falls_back() {
        assert_eq!(
            detect_completion_architecture("some-random-model"),
            CompletionArchitecture::GenericGguf
        );
    }

    #[test]
    fn detect_jina_embedding() {
        assert_eq!(
            detect_embedding_architecture("jinaai/jina-embeddings-v2-base-code"),
            EmbeddingArchitecture::JinaCode
        );
    }

    #[test]
    fn default_embedding_model_is_code_specialized() {
        let model = default_embedding_model();
        assert!(
            model.code_specialized,
            "default embedding should be code-specialized"
        );
        assert!(model.model_id.contains("jina"), "should be Jina Code");
    }

    #[test]
    fn default_completion_model_is_code_specialized() {
        let model = default_completion_model();
        assert!(
            model.code_specialized,
            "default completion should be code-specialized"
        );
        assert_eq!(model.model_id, "qwen2.5-coder-3b");
    }

    #[test]
    fn find_7b_model() {
        let entry = find_completion_model("qwen2.5-coder-7b");
        assert!(entry.is_some(), "7B model must exist in registry");
        let e = entry.unwrap();
        assert_eq!(e.params_b, 7.0);
        assert_eq!(e.estimated_vram_mb, 4800);
        assert!(e.code_specialized);
    }

    #[test]
    fn recommend_selects_best_fitting_model() {
        // Enough for 7B
        let m = recommend_completion_model(5000).unwrap();
        assert_eq!(m.model_id, "qwen2.5-coder-7b");

        // Enough for 3B but not 7B
        let m = recommend_completion_model(3000).unwrap();
        assert_eq!(m.model_id, "qwen2.5-coder-3b");

        // Enough for 1.5B but not 3B
        let m = recommend_completion_model(1500).unwrap();
        assert_eq!(m.model_id, "qwen2.5-coder-1.5b");
    }

    #[test]
    fn recommend_returns_none_when_too_small() {
        assert!(
            recommend_completion_model(500).is_none(),
            "should return None when no model fits"
        );
    }

    #[test]
    fn recommend_falls_back_to_non_code_model() {
        // TinyLLaMA needs 800 MB — at exactly 800, it should fit
        // but no code-specialized model fits at 800
        let m = recommend_completion_model(800).unwrap();
        assert_eq!(m.model_id, "tinyllama-1.1b-chat");
    }

    #[test]
    fn check_model_fits_passes_when_enough_ram() {
        assert!(check_model_fits("qwen2.5-coder-3b", 5000).is_ok());
    }

    #[test]
    fn check_model_fits_fails_with_suggestion() {
        let err = check_model_fits("qwen2.5-coder-7b", 3000).unwrap_err();
        assert!(err.contains("4800 MB"), "should mention required RAM");
        assert!(
            err.contains("qwen2.5-coder-3b"),
            "should suggest smaller model"
        );
    }

    #[test]
    fn check_model_fits_unknown_model_passes() {
        assert!(
            check_model_fits("unknown-model-xyz", 1000).is_ok(),
            "unknown models should pass (can't check)"
        );
    }

    #[test]
    fn all_models_have_vram_estimate() {
        for m in COMPLETION_MODELS {
            assert!(
                m.estimated_vram_mb > 0,
                "model {} missing VRAM estimate",
                m.model_id
            );
        }
        for m in EMBEDDING_MODELS {
            assert!(
                m.estimated_vram_mb > 0,
                "model {} missing VRAM estimate",
                m.model_id
            );
        }
    }
}
