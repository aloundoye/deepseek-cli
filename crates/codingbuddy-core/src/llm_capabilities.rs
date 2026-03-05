use crate::{CODINGBUDDY_V32_CHAT_MODEL, CODINGBUDDY_V32_REASONER_MODEL};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderKind {
    Deepseek,
    OpenAiCompatible,
    Ollama,
}

impl ProviderKind {
    #[must_use]
    pub fn as_key(self) -> &'static str {
        match self {
            ProviderKind::Deepseek => "deepseek",
            ProviderKind::OpenAiCompatible => "openai-compatible",
            ProviderKind::Ollama => "ollama",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelFamily {
    Deepseek,
    Qwen,
    Gemini,
    OpenAi,
    Llama,
    Mistral,
    Generic,
}

impl ModelFamily {
    #[must_use]
    pub fn as_key(self) -> &'static str {
        match self {
            ModelFamily::Deepseek => "deepseek",
            ModelFamily::Qwen => "qwen",
            ModelFamily::Gemini => "gemini",
            ModelFamily::OpenAi => "openai",
            ModelFamily::Llama => "llama",
            ModelFamily::Mistral => "mistral",
            ModelFamily::Generic => "generic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
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
    /// Whether chat payloads may include image inputs.
    pub supports_image_input: bool,
    /// Whether outbound chat messages should be strictly filtered for empty content.
    pub strict_empty_content_filtering: bool,
    /// Whether tool call ids should be normalized to provider-safe identifiers.
    pub normalize_tool_call_ids: bool,
    pub max_safe_tool_count: usize,
    pub preferred_edit_tool: PreferredEditTool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(default)]
pub struct CapabilityOverride {
    pub supports_tool_calling: Option<bool>,
    pub supports_tool_choice: Option<bool>,
    pub supports_parallel_tool_calls: Option<bool>,
    pub supports_reasoning_mode: Option<bool>,
    pub supports_thinking_config: Option<bool>,
    pub supports_streaming_tool_deltas: Option<bool>,
    pub supports_fim: Option<bool>,
    pub supports_image_input: Option<bool>,
    pub strict_empty_content_filtering: Option<bool>,
    pub normalize_tool_call_ids: Option<bool>,
    pub max_safe_tool_count: Option<usize>,
    pub preferred_edit_tool: Option<PreferredEditTool>,
}

impl CapabilityOverride {
    fn apply_to(&self, capabilities: &mut ModelCapabilities) {
        if let Some(value) = self.supports_tool_calling {
            capabilities.supports_tool_calling = value;
        }
        if let Some(value) = self.supports_tool_choice {
            capabilities.supports_tool_choice = value;
        }
        if let Some(value) = self.supports_parallel_tool_calls {
            capabilities.supports_parallel_tool_calls = value;
        }
        if let Some(value) = self.supports_reasoning_mode {
            capabilities.supports_reasoning_mode = value;
        }
        if let Some(value) = self.supports_thinking_config {
            capabilities.supports_thinking_config = value;
        }
        if let Some(value) = self.supports_streaming_tool_deltas {
            capabilities.supports_streaming_tool_deltas = value;
        }
        if let Some(value) = self.supports_fim {
            capabilities.supports_fim = value;
        }
        if let Some(value) = self.supports_image_input {
            capabilities.supports_image_input = value;
        }
        if let Some(value) = self.strict_empty_content_filtering {
            capabilities.strict_empty_content_filtering = value;
        }
        if let Some(value) = self.normalize_tool_call_ids {
            capabilities.normalize_tool_call_ids = value;
        }
        if let Some(value) = self.max_safe_tool_count {
            capabilities.max_safe_tool_count = value.max(1);
        }
        if let Some(value) = self.preferred_edit_tool {
            capabilities.preferred_edit_tool = value;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(default)]
pub struct CapabilityRegistryOverrides {
    /// Family-level overrides keyed by either:
    /// - `<family>` (e.g. `qwen`)
    /// - `<provider>@<family>` (e.g. `ollama@qwen`)
    pub families: BTreeMap<String, CapabilityOverride>,
    /// Model-level overrides keyed by either:
    /// - exact model id (e.g. `qwen2.5-coder:7b`)
    /// - prefix wildcard (e.g. `qwen2.5-coder:*`)
    /// - scoped exact/prefix (e.g. `ollama@qwen2.5-coder:*`)
    pub models: BTreeMap<String, CapabilityOverride>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityResolution {
    pub capabilities: ModelCapabilities,
    pub applied_rules: Vec<String>,
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
    if lower.contains("qwen") || lower.contains("qwq") {
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
pub fn resolve_model_capabilities(
    provider: ProviderKind,
    model: &str,
    registry: Option<&CapabilityRegistryOverrides>,
) -> CapabilityResolution {
    let family = detect_model_family(model);
    let mut capabilities = base_capabilities(provider, model, family);
    let mut applied_rules = vec![format!("base:{}:{}", provider.as_key(), family.as_key())];

    if let Some((rule, family_override)) = built_in_family_override(provider, family) {
        family_override.apply_to(&mut capabilities);
        applied_rules.push(rule.to_string());
    }

    if let Some((rule, model_override)) = built_in_model_override(provider, model) {
        model_override.apply_to(&mut capabilities);
        applied_rules.push(rule.to_string());
    }

    if let Some(registry) = registry {
        for (rule, override_entry) in config_family_overrides(registry, provider, family) {
            override_entry.apply_to(&mut capabilities);
            applied_rules.push(rule);
        }
        for (rule, override_entry) in config_model_overrides(registry, provider, model) {
            override_entry.apply_to(&mut capabilities);
            applied_rules.push(rule);
        }
    }

    CapabilityResolution {
        capabilities,
        applied_rules,
    }
}

#[must_use]
pub fn model_capabilities(provider: ProviderKind, model: &str) -> ModelCapabilities {
    resolve_model_capabilities(provider, model, None).capabilities
}

#[must_use]
pub fn model_capabilities_with_registry(
    provider: ProviderKind,
    model: &str,
    registry: &CapabilityRegistryOverrides,
) -> ModelCapabilities {
    resolve_model_capabilities(provider, model, Some(registry)).capabilities
}

fn base_capabilities(
    provider: ProviderKind,
    model: &str,
    family: ModelFamily,
) -> ModelCapabilities {
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
                supports_image_input: !is_reasoner,
                strict_empty_content_filtering: false,
                normalize_tool_call_ids: false,
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
            supports_image_input: true,
            strict_empty_content_filtering: true,
            normalize_tool_call_ids: false,
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
            supports_image_input: false,
            strict_empty_content_filtering: true,
            normalize_tool_call_ids: true,
            max_safe_tool_count: 12,
            preferred_edit_tool: PreferredEditTool::FsEdit,
        },
    }
}

fn built_in_family_override(
    provider: ProviderKind,
    family: ModelFamily,
) -> Option<(&'static str, CapabilityOverride)> {
    match (provider, family) {
        (ProviderKind::Ollama, ModelFamily::Qwen) => Some((
            "builtin_family:ollama@qwen",
            CapabilityOverride {
                max_safe_tool_count: Some(14),
                preferred_edit_tool: Some(PreferredEditTool::MultiEdit),
                supports_parallel_tool_calls: Some(false),
                ..CapabilityOverride::default()
            },
        )),
        (ProviderKind::Ollama, ModelFamily::Llama) => Some((
            "builtin_family:ollama@llama",
            CapabilityOverride {
                supports_tool_choice: Some(false),
                max_safe_tool_count: Some(9),
                preferred_edit_tool: Some(PreferredEditTool::PatchDirect),
                ..CapabilityOverride::default()
            },
        )),
        (ProviderKind::Ollama, ModelFamily::Deepseek) => Some((
            "builtin_family:ollama@deepseek",
            CapabilityOverride {
                supports_tool_choice: Some(false),
                supports_parallel_tool_calls: Some(false),
                supports_reasoning_mode: Some(true),
                supports_thinking_config: Some(false),
                max_safe_tool_count: Some(8),
                preferred_edit_tool: Some(PreferredEditTool::PatchDirect),
                ..CapabilityOverride::default()
            },
        )),
        (ProviderKind::OpenAiCompatible, ModelFamily::Gemini) => Some((
            "builtin_family:openai-compatible@gemini",
            CapabilityOverride {
                supports_parallel_tool_calls: Some(false),
                max_safe_tool_count: Some(14),
                ..CapabilityOverride::default()
            },
        )),
        (ProviderKind::OpenAiCompatible, ModelFamily::Qwen) => Some((
            "builtin_family:openai-compatible@qwen",
            CapabilityOverride {
                supports_parallel_tool_calls: Some(false),
                max_safe_tool_count: Some(16),
                preferred_edit_tool: Some(PreferredEditTool::FsEdit),
                ..CapabilityOverride::default()
            },
        )),
        (ProviderKind::OpenAiCompatible, ModelFamily::Llama) => Some((
            "builtin_family:openai-compatible@llama",
            CapabilityOverride {
                supports_parallel_tool_calls: Some(false),
                max_safe_tool_count: Some(12),
                preferred_edit_tool: Some(PreferredEditTool::FsEdit),
                ..CapabilityOverride::default()
            },
        )),
        _ => None,
    }
}

fn built_in_model_override(
    provider: ProviderKind,
    model: &str,
) -> Option<(&'static str, CapabilityOverride)> {
    let lower = model.trim().to_ascii_lowercase();
    if provider != ProviderKind::Deepseek
        && (lower.contains("deepseek-r1") || lower.contains("deepseek-reasoner"))
    {
        return Some((
            "builtin_model:deepseek-r1",
            CapabilityOverride {
                supports_tool_choice: Some(false),
                supports_parallel_tool_calls: Some(false),
                supports_reasoning_mode: Some(true),
                supports_thinking_config: Some(false),
                max_safe_tool_count: Some(8),
                preferred_edit_tool: Some(PreferredEditTool::PatchDirect),
                ..CapabilityOverride::default()
            },
        ));
    }
    if lower.contains("qwen2.5-coder:1.5b") || lower.contains("qwen2.5-coder-1.5b") {
        return Some((
            "builtin_model:qwen2.5-coder-1.5b",
            CapabilityOverride {
                max_safe_tool_count: Some(10),
                ..CapabilityOverride::default()
            },
        ));
    }
    if lower.contains("qwen2.5-coder:3b") || lower.contains("qwen2.5-coder-3b") {
        return Some((
            "builtin_model:qwen2.5-coder-3b",
            CapabilityOverride {
                max_safe_tool_count: Some(12),
                ..CapabilityOverride::default()
            },
        ));
    }
    if lower.contains("qwen2.5-coder:7b") || lower.contains("qwen2.5-coder-7b") {
        return Some((
            "builtin_model:qwen2.5-coder-7b",
            CapabilityOverride {
                max_safe_tool_count: Some(14),
                preferred_edit_tool: Some(PreferredEditTool::MultiEdit),
                ..CapabilityOverride::default()
            },
        ));
    }
    if lower.contains("llama3.1:8b") || lower.contains("llama3:8b") {
        return Some((
            "builtin_model:llama3-8b",
            CapabilityOverride {
                max_safe_tool_count: Some(8),
                supports_tool_choice: Some(false),
                ..CapabilityOverride::default()
            },
        ));
    }
    if lower.contains("gemini-2.0-flash") {
        return Some((
            "builtin_model:gemini-2.0-flash",
            CapabilityOverride {
                max_safe_tool_count: Some(12),
                ..CapabilityOverride::default()
            },
        ));
    }
    if lower.contains("llava")
        || lower.contains("bakllava")
        || lower.contains("moondream")
        || lower.contains("vision")
    {
        return Some((
            "builtin_model:vision-family",
            CapabilityOverride {
                supports_image_input: Some(true),
                ..CapabilityOverride::default()
            },
        ));
    }
    None
}

fn config_family_overrides(
    registry: &CapabilityRegistryOverrides,
    provider: ProviderKind,
    family: ModelFamily,
) -> Vec<(String, &CapabilityOverride)> {
    let mut out = Vec::new();
    let generic_key = family.as_key().to_string();
    if let Some(override_entry) = registry.families.get(&generic_key) {
        out.push((format!("config_family:{generic_key}"), override_entry));
    }
    let scoped_key = format!("{}@{}", provider.as_key(), family.as_key());
    if let Some(override_entry) = registry.families.get(&scoped_key) {
        out.push((format!("config_family:{scoped_key}"), override_entry));
    }
    out
}

fn config_model_overrides<'a>(
    registry: &'a CapabilityRegistryOverrides,
    provider: ProviderKind,
    model: &str,
) -> Vec<(String, &'a CapabilityOverride)> {
    let model_norm = model.trim().to_ascii_lowercase();
    let mut matches = Vec::new();
    for (raw_key, override_entry) in &registry.models {
        if let Some(rank) = model_override_match_rank(raw_key, provider, &model_norm) {
            matches.push((rank, raw_key.clone(), override_entry));
        }
    }
    matches.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    matches
        .into_iter()
        .map(|(_, key, override_entry)| (format!("config_model:{key}"), override_entry))
        .collect()
}

fn model_override_match_rank(key: &str, provider: ProviderKind, model_norm: &str) -> Option<u8> {
    let (scoped_provider, pattern) = split_provider_scope(key);
    if let Some(scope) = scoped_provider
        && scope != provider
    {
        return None;
    }
    let pattern = pattern.trim();
    if pattern.is_empty() {
        return None;
    }
    let wildcard = pattern.ends_with('*');
    let matched = if wildcard {
        let prefix = pattern.trim_end_matches('*').trim();
        !prefix.is_empty() && model_norm.starts_with(prefix)
    } else {
        model_norm == pattern
    };
    if !matched {
        return None;
    }
    let rank = match (scoped_provider.is_some(), wildcard) {
        (false, true) => 0,
        (false, false) => 1,
        (true, true) => 2,
        (true, false) => 3,
    };
    Some(rank)
}

fn split_provider_scope(key: &str) -> (Option<ProviderKind>, String) {
    let normalized = key.trim().to_ascii_lowercase();
    let Some((provider_raw, pattern_raw)) = normalized.split_once('@') else {
        return (None, normalized);
    };
    let Some(provider) = normalize_provider_kind(provider_raw) else {
        return (None, normalized);
    };
    (Some(provider), pattern_raw.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_qwen_family_override_applies() {
        let caps = model_capabilities(ProviderKind::Ollama, "qwen2.5-coder:7b");
        assert_eq!(caps.family, ModelFamily::Qwen);
        assert_eq!(caps.preferred_edit_tool, PreferredEditTool::MultiEdit);
        assert_eq!(caps.max_safe_tool_count, 14);
        assert!(!caps.supports_image_input);
        assert!(caps.strict_empty_content_filtering);
        assert!(caps.normalize_tool_call_ids);
    }

    #[test]
    fn ollama_deepseek_r1_model_override_disables_tool_choice() {
        let caps = model_capabilities(ProviderKind::Ollama, "deepseek-r1:14b");
        assert!(!caps.supports_tool_choice);
        assert!(caps.supports_reasoning_mode);
        assert_eq!(caps.preferred_edit_tool, PreferredEditTool::PatchDirect);
    }

    #[test]
    fn preferred_edit_tool_contracts_match_provider_and_model() {
        let cases = [
            (
                ProviderKind::Deepseek,
                "deepseek-chat",
                PreferredEditTool::FsEdit,
            ),
            (
                ProviderKind::OpenAiCompatible,
                "gpt-4o-mini",
                PreferredEditTool::PatchDirect,
            ),
            (
                ProviderKind::OpenAiCompatible,
                "qwen2.5-coder:7b",
                PreferredEditTool::MultiEdit,
            ),
            (
                ProviderKind::Ollama,
                "qwen2.5-coder:7b",
                PreferredEditTool::MultiEdit,
            ),
            (
                ProviderKind::Ollama,
                "llama3.1:8b",
                PreferredEditTool::PatchDirect,
            ),
        ];

        for (provider, model, expected) in cases {
            let caps = model_capabilities(provider, model);
            assert_eq!(
                caps.preferred_edit_tool,
                expected,
                "unexpected preferred edit tool for provider={} model={}",
                provider.as_key(),
                model
            );
        }
    }

    #[test]
    fn config_overrides_apply_in_specificity_order() {
        let mut registry = CapabilityRegistryOverrides::default();
        registry.families.insert(
            "qwen".to_string(),
            CapabilityOverride {
                max_safe_tool_count: Some(11),
                ..CapabilityOverride::default()
            },
        );
        registry.families.insert(
            "ollama@qwen".to_string(),
            CapabilityOverride {
                max_safe_tool_count: Some(13),
                ..CapabilityOverride::default()
            },
        );
        registry.models.insert(
            "qwen2.5-coder:*".to_string(),
            CapabilityOverride {
                max_safe_tool_count: Some(9),
                ..CapabilityOverride::default()
            },
        );
        registry.models.insert(
            "ollama@qwen2.5-coder:7b".to_string(),
            CapabilityOverride {
                max_safe_tool_count: Some(7),
                supports_tool_choice: Some(false),
                ..CapabilityOverride::default()
            },
        );

        let resolved =
            resolve_model_capabilities(ProviderKind::Ollama, "qwen2.5-coder:7b", Some(&registry));
        assert_eq!(resolved.capabilities.max_safe_tool_count, 7);
        assert!(!resolved.capabilities.supports_tool_choice);
        assert!(
            resolved
                .applied_rules
                .iter()
                .any(|rule| rule.contains("config_model:ollama@qwen2.5-coder:7b"))
        );
    }

    #[test]
    fn scoped_override_ignores_other_provider() {
        let mut registry = CapabilityRegistryOverrides::default();
        registry.models.insert(
            "deepseek@qwen2.5-coder:*".to_string(),
            CapabilityOverride {
                max_safe_tool_count: Some(3),
                ..CapabilityOverride::default()
            },
        );

        let caps =
            model_capabilities_with_registry(ProviderKind::Ollama, "qwen2.5-coder:7b", &registry);
        assert_ne!(caps.max_safe_tool_count, 3);
    }

    #[test]
    fn vision_family_model_override_enables_image_input() {
        let caps = model_capabilities(ProviderKind::Ollama, "llava:13b");
        assert!(caps.supports_image_input);
    }

    #[test]
    fn transform_flags_can_be_overridden_from_registry() {
        let mut registry = CapabilityRegistryOverrides::default();
        registry.models.insert(
            "ollama@qwen2.5-coder:7b".to_string(),
            CapabilityOverride {
                supports_image_input: Some(true),
                strict_empty_content_filtering: Some(false),
                normalize_tool_call_ids: Some(false),
                ..CapabilityOverride::default()
            },
        );
        let caps =
            model_capabilities_with_registry(ProviderKind::Ollama, "qwen2.5-coder:7b", &registry);
        assert!(caps.supports_image_input);
        assert!(!caps.strict_empty_content_filtering);
        assert!(!caps.normalize_tool_call_ids);
    }
}
