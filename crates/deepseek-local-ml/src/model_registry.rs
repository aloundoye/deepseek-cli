//! Model registry — catalogs supported model architectures for local inference.
//!
//! Provides metadata about each model family so the system can select the right
//! loading path, tokenizer, and generation strategy.

/// Architecture family for completion models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionArchitecture {
    /// DeepSeek-Coder family (GGUF, LLaMA-based)
    DeepSeekCoder,
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
    /// HuggingFace model ID or local path identifier.
    pub model_id: &'static str,
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
}

/// Registry of supported completion models.
pub const COMPLETION_MODELS: &[ModelEntry] = &[
    ModelEntry {
        model_id: "deepseek-coder-1.3b",
        display_name: "DeepSeek Coder 1.3B",
        params_b: 1.3,
        code_specialized: true,
        context_tokens: 16384,
        quality_tier: 2,
    },
    ModelEntry {
        model_id: "qwen2.5-coder-1.5b",
        display_name: "Qwen2.5 Coder 1.5B",
        params_b: 1.5,
        code_specialized: true,
        context_tokens: 32768,
        quality_tier: 2,
    },
    ModelEntry {
        model_id: "phi-3-mini-4k",
        display_name: "Phi-3 Mini 4K",
        params_b: 3.8,
        code_specialized: false,
        context_tokens: 4096,
        quality_tier: 3,
    },
    ModelEntry {
        model_id: "tinyllama-1.1b-chat",
        display_name: "TinyLLaMA 1.1B Chat",
        params_b: 1.1,
        code_specialized: false,
        context_tokens: 2048,
        quality_tier: 1,
    },
];

/// Registry of supported embedding models.
pub const EMBEDDING_MODELS: &[ModelEntry] = &[
    ModelEntry {
        model_id: "sentence-transformers/all-MiniLM-L6-v2",
        display_name: "MiniLM L6 v2 (384d)",
        params_b: 0.022,
        code_specialized: false,
        context_tokens: 512,
        quality_tier: 2,
    },
    ModelEntry {
        model_id: "BAAI/bge-base-en-v1.5",
        display_name: "BGE Base EN v1.5 (768d)",
        params_b: 0.110,
        code_specialized: false,
        context_tokens: 512,
        quality_tier: 3,
    },
    ModelEntry {
        model_id: "jinaai/jina-embeddings-v2-base-code",
        display_name: "Jina Code v2 (768d)",
        params_b: 0.137,
        code_specialized: true,
        context_tokens: 8192,
        quality_tier: 3,
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
    // DeepSeek Coder 1.3B — best code quality at small size
    &COMPLETION_MODELS[0]
}

/// Detect the architecture from a model ID.
#[cfg(any(test, feature = "local-ml"))]
pub fn detect_completion_architecture(model_id: &str) -> CompletionArchitecture {
    let lower = model_id.to_ascii_lowercase();
    if lower.contains("deepseek") {
        CompletionArchitecture::DeepSeekCoder
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
        let entry = find_completion_model("deepseek-coder-1.3b");
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
    fn detect_deepseek_architecture() {
        assert_eq!(
            detect_completion_architecture("deepseek-coder-1.3b"),
            CompletionArchitecture::DeepSeekCoder
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
        assert!(
            model.model_id.contains("deepseek"),
            "should be DeepSeek Coder"
        );
    }
}
