use crate::{CODINGBUDDY_V32_CHAT_MODEL, CODINGBUDDY_V32_REASONER_MODEL};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderKind {
    Deepseek,
    OpenAiCompatible,
    Ollama,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Deepseek,
    Qwen,
    Gemini,
    OpenAi,
    Llama,
    Mistral,
    Generic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreferredEditTool {
    FsEdit,
    MultiEdit,
    PatchDirect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCapabilities {
    pub provider: ProviderKind,
    pub family: ModelFamily,
    pub supports_tool_calling: bool,
    pub supports_tool_choice: bool,
    pub supports_parallel_tool_calls: bool,
    pub supports_reasoning_mode: bool,
    pub supports_thinking_config: bool,
    pub supports_streaming_tool_deltas: bool,
    pub supports_fim: bool,
    pub max_safe_tool_count: usize,
    pub preferred_edit_tool: PreferredEditTool,
}

#[must_use]
pub fn normalize_provider_kind(name: &str) -> Option<ProviderKind> {
    match name.trim().to_ascii_lowercase().as_str() {
        "deepseek" => Some(ProviderKind::Deepseek),
        "openai-compatible" | "openai-compat" | "openai_compat" | "openai" | "custom" | "local" => {
            Some(ProviderKind::OpenAiCompatible)
        }
        "ollama" => Some(ProviderKind::Ollama),
        _ => None,
    }
}

#[must_use]
pub fn detect_model_family(model: &str) -> ModelFamily {
    let lower = model.trim().to_ascii_lowercase();
    if lower.contains("qwen") {
        ModelFamily::Qwen
    } else if lower.contains("gemini") {
        ModelFamily::Gemini
    } else if lower.contains("deepseek")
        || lower == CODINGBUDDY_V32_CHAT_MODEL
        || lower == CODINGBUDDY_V32_REASONER_MODEL
    {
        ModelFamily::Deepseek
    } else if lower.starts_with("gpt-")
        || lower.starts_with("o1")
        || lower.starts_with("o3")
        || lower.starts_with("o4")
    {
        ModelFamily::OpenAi
    } else if lower.contains("llama") {
        ModelFamily::Llama
    } else if lower.contains("mistral") {
        ModelFamily::Mistral
    } else {
        ModelFamily::Generic
    }
}

#[must_use]
pub fn model_capabilities(provider: ProviderKind, model: &str) -> ModelCapabilities {
    let family = detect_model_family(model);
    match provider {
        ProviderKind::Deepseek => {
            let is_reasoner = crate::is_reasoner_model(model);
            ModelCapabilities {
                provider,
                family,
                supports_tool_calling: true,
                supports_tool_choice: !is_reasoner,
                supports_parallel_tool_calls: !is_reasoner,
                supports_reasoning_mode: is_reasoner,
                supports_thinking_config: !is_reasoner,
                supports_streaming_tool_deltas: true,
                supports_fim: true,
                max_safe_tool_count: if is_reasoner { 18 } else { 24 },
                preferred_edit_tool: PreferredEditTool::FsEdit,
            }
        }
        ProviderKind::OpenAiCompatible => ModelCapabilities {
            provider,
            family,
            supports_tool_calling: true,
            supports_tool_choice: true,
            supports_parallel_tool_calls: true,
            supports_reasoning_mode: false,
            supports_thinking_config: false,
            supports_streaming_tool_deltas: true,
            supports_fim: false,
            max_safe_tool_count: 18,
            preferred_edit_tool: PreferredEditTool::PatchDirect,
        },
        ProviderKind::Ollama => ModelCapabilities {
            provider,
            family,
            supports_tool_calling: true,
            supports_tool_choice: true,
            supports_parallel_tool_calls: false,
            supports_reasoning_mode: false,
            supports_thinking_config: false,
            supports_streaming_tool_deltas: true,
            supports_fim: false,
            max_safe_tool_count: 12,
            preferred_edit_tool: PreferredEditTool::FsEdit,
        },
    }
}
