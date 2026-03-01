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
}

/// Default files to download for SafeTensors (non-GGUF) models.
pub const DEFAULT_SAFETENSORS_FILES: &[&str] = &["config.json", "tokenizer.json", "model.safetensors"];

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

/// Registry of supported completion models.
pub const COMPLETION_MODELS: &[ModelEntry] = &[
    ModelEntry {
        model_id: "qwen2.5-coder-3b",
        hf_repo: "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        display_name: "Qwen2.5 Coder 3B",
        params_b: 3.0,
        code_specialized: true,
        context_tokens: 32768,
        quality_tier: 3,
        gguf_filename: Some("qwen2.5-coder-3b-instruct-q4_k_m.gguf"),
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
    // Qwen2.5 Coder 3B — best code quality at small size
    &COMPLETION_MODELS[0]
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
        assert!(
            model.model_id.contains("qwen"),
            "should be Qwen2.5 Coder"
        );
    }
}
