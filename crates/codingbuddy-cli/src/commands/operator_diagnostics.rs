use codingbuddy_core::{AppConfig, AppliedCompatibility, ModelFamily, ProviderKind};
use codingbuddy_local_ml::{RunnerLifecycleState, RuntimeLifecycleSnapshot};
use serde::Serialize;
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Default)]
pub(crate) struct ProviderCompatibilityDiagnostics {
    pub provider: String,
    pub family: String,
    pub summary: String,
    pub active_transforms: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub(crate) struct RuntimeOperatorDiagnostics {
    pub summary: String,
    pub highlights: Vec<String>,
}

pub(crate) fn summarize_applied_compatibility(
    compatibility: Option<&AppliedCompatibility>,
) -> Option<String> {
    let compatibility = compatibility?;
    if compatibility.transforms.is_empty() && compatibility.degraded_inputs.is_empty() {
        return None;
    }
    let mut parts = Vec::new();
    if !compatibility.transforms.is_empty() {
        parts.push(format!("applied={}", compatibility.transforms.join(",")));
    }
    if !compatibility.degraded_inputs.is_empty() {
        parts.push(format!(
            "degraded={}",
            compatibility.degraded_inputs.join(",")
        ));
    }
    Some(parts.join(" "))
}

pub(crate) fn provider_compatibility_diagnostics(
    cfg: &AppConfig,
    model: &str,
) -> Option<ProviderCompatibilityDiagnostics> {
    let resolution = cfg.llm.capability_resolution_for_model(model)?;
    let caps = resolution.capabilities;
    let provider = cfg.llm.active_provider();
    let mut active = Vec::new();

    active.push("tool-name-repair".to_string());

    if caps.normalize_tool_call_ids {
        active.push("tool-id-normalization".to_string());
    }
    if caps.strict_empty_content_filtering {
        active.push("strict-empty-filtering".to_string());
    }

    match caps.provider {
        ProviderKind::OpenAiCompatible => {
            active.push("thinking->reasoning_effort".to_string());
            if prefers_max_completion_tokens(model) {
                active.push("max_tokens->max_completion_tokens".to_string());
                active.push("sampling-strip-on-reasoning".to_string());
            }
            if caps.family == ModelFamily::Gemini {
                active.push("gemini-schema-sanitize".to_string());
                active.push("required->auto-tool_choice".to_string());
                active.push("max_output_tokens-alias".to_string());
            }
            if looks_like_litellm_proxy(&provider.base_url, &cfg.llm.endpoint) {
                active.push("litellm-placeholder-tool".to_string());
            }
        }
        ProviderKind::Ollama => {
            active.push("required->auto-tool_choice".to_string());
            active.push("max_tokens->options.num_predict".to_string());
        }
        ProviderKind::Deepseek => {}
    }

    let summary = active.join(", ");
    Some(ProviderCompatibilityDiagnostics {
        provider: caps.provider.as_key().to_string(),
        family: caps.family.as_key().to_string(),
        summary,
        active_transforms: active,
    })
}

pub(crate) fn runtime_operator_diagnostics(
    snapshot: &RuntimeLifecycleSnapshot,
) -> RuntimeOperatorDiagnostics {
    let warm = snapshot.warm_models.len();
    let cap = snapshot.max_loaded_models.max(1);
    let metrics = &snapshot.metrics;
    let now_epoch_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let live_is_stale = snapshot.live.updated_at_epoch_secs == 0
        || now_epoch_secs.saturating_sub(snapshot.live.updated_at_epoch_secs) > 10;
    let last_event = snapshot
        .recent_events
        .last()
        .map(|event| match event.model_id.as_deref() {
            Some(model_id) => format!("{}:{model_id}", event.kind),
            None => event.kind.clone(),
        })
        .unwrap_or_else(|| "none".to_string());
    let active_requests = if live_is_stale {
        0
    } else {
        snapshot.live.active_requests
    };
    let queued_requests = if live_is_stale {
        0
    } else {
        snapshot.live.queued_requests
    };
    let loading_waiters = if live_is_stale {
        0
    } else {
        snapshot.live.loading_waiters
    };

    let mut highlights = Vec::new();
    highlights.push(if live_is_stale {
        "live=stale".to_string()
    } else {
        "live=fresh".to_string()
    });
    if let Some(model) = snapshot.live.loading_model.as_ref()
        && !live_is_stale
    {
        highlights.push(format!("loading={model}"));
    }
    if queued_requests > 0 {
        highlights.push(format!("queued={queued_requests}"));
    }
    if loading_waiters > 0 {
        highlights.push(format!("load_waiters={loading_waiters}"));
    }
    if !snapshot.runner_states.is_empty() {
        let counts = summarize_runner_states(&snapshot.runner_states);
        highlights.extend(counts);
    }
    if metrics.total_runner_load_waits > 0 {
        highlights.push(format!("load_waits={}", metrics.total_runner_load_waits));
    }
    if metrics.total_memory_pressure_evictions > 0 {
        highlights.push(format!(
            "memory_pressure_evictions={}",
            metrics.total_memory_pressure_evictions
        ));
    }
    if metrics.total_memory_admission_denied > 0 {
        highlights.push(format!(
            "memory_denied={}",
            metrics.total_memory_admission_denied
        ));
    }
    if metrics.total_runner_load_failures > 0 {
        highlights.push(format!(
            "load_failures={}",
            metrics.total_runner_load_failures
        ));
    }
    if metrics.total_runner_reloads > 0 {
        highlights.push(format!("reloads={}", metrics.total_runner_reloads));
    }
    if highlights.is_empty() {
        highlights.push("steady".to_string());
    }

    let summary = format!(
        "warm={warm}/{cap} active={active_requests} queued={queued_requests} loading={} waiters={} live={} queue_peak={} load_waits={} last={last_event}",
        snapshot
            .live
            .loading_model
            .clone()
            .unwrap_or_else(|| "none".to_string()),
        loading_waiters,
        if live_is_stale { "stale" } else { "fresh" },
        metrics.max_observed_queue_depth,
        metrics.total_runner_load_waits
    );

    RuntimeOperatorDiagnostics {
        summary,
        highlights,
    }
}

fn prefers_max_completion_tokens(model: &str) -> bool {
    let lower = model.trim().to_ascii_lowercase();
    lower.starts_with("o1")
        || lower.starts_with("o3")
        || lower.starts_with("o4")
        || lower.contains("reasoning")
}

fn looks_like_litellm_proxy(base_url: &str, endpoint: &str) -> bool {
    let lower_base = base_url.to_ascii_lowercase();
    let lower_endpoint = endpoint.to_ascii_lowercase();
    lower_base.contains("litellm") || lower_endpoint.contains("litellm")
}

fn summarize_runner_states(states: &BTreeMap<String, RunnerLifecycleState>) -> Vec<String> {
    let mut loading = 0usize;
    let mut busy = 0usize;
    let mut reloading = 0usize;
    let mut failed = 0usize;
    for state in states.values() {
        match state {
            RunnerLifecycleState::Loading => loading += 1,
            RunnerLifecycleState::Busy => busy += 1,
            RunnerLifecycleState::Reloading => reloading += 1,
            RunnerLifecycleState::Failed => failed += 1,
            RunnerLifecycleState::Warm | RunnerLifecycleState::Expiring => {}
        }
    }
    let mut summary = Vec::new();
    if loading > 0 {
        summary.push(format!("loading_states={loading}"));
    }
    if busy > 0 {
        summary.push(format!("busy_states={busy}"));
    }
    if reloading > 0 {
        summary.push(format!("reloading_states={reloading}"));
    }
    if failed > 0 {
        summary.push(format!("failed_states={failed}"));
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;
    use codingbuddy_core::AppConfig;
    use codingbuddy_local_ml::RuntimeLifecycleMetrics;

    #[test]
    fn provider_diagnostics_include_openai_reasoning_and_litellm_shims() {
        let mut cfg = AppConfig::default();
        cfg.llm.provider = "openai-compatible".to_string();
        if let Some(provider) = cfg.llm.providers.get_mut("openai-compatible") {
            provider.base_url = "https://litellm.internal".to_string();
            provider.models.chat = "o3-mini".to_string();
        }
        cfg.llm.endpoint = "https://litellm.internal/v1/chat/completions".to_string();

        let diagnostics = provider_compatibility_diagnostics(&cfg, "o3-mini").expect("diagnostics");
        assert!(diagnostics.summary.contains("thinking->reasoning_effort"));
        assert!(
            diagnostics
                .active_transforms
                .iter()
                .any(|item| item == "max_tokens->max_completion_tokens")
        );
        assert!(
            diagnostics
                .active_transforms
                .iter()
                .any(|item| item == "litellm-placeholder-tool")
        );
    }

    #[test]
    fn runtime_diagnostics_surface_pressure_signals() {
        let now_epoch_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_secs();
        let snapshot = RuntimeLifecycleSnapshot {
            max_loaded_models: 2,
            keep_warm_secs: 300,
            aggressive_eviction: false,
            scheduler: Default::default(),
            warm_models: vec!["model-a".to_string()],
            live: codingbuddy_local_ml::RuntimeSchedulerLiveSnapshot {
                active_requests: 1,
                queued_requests: 2,
                loading_model: Some("model-b".to_string()),
                loading_waiters: 1,
                loaded_runners: vec!["model-a".to_string()],
                updated_at_epoch_secs: now_epoch_secs,
            },
            runner_states: BTreeMap::from([
                ("model-a".to_string(), RunnerLifecycleState::Warm),
                ("model-b".to_string(), RunnerLifecycleState::Loading),
                ("model-c".to_string(), RunnerLifecycleState::Failed),
            ]),
            metrics: RuntimeLifecycleMetrics {
                total_runner_load_waits: 2,
                total_memory_pressure_evictions: 1,
                total_memory_admission_denied: 1,
                total_runner_load_failures: 1,
                max_observed_queue_depth: 3,
                ..RuntimeLifecycleMetrics::default()
            },
            recent_events: vec![codingbuddy_local_ml::RuntimeLifecycleEvent {
                kind: "runner_load_wait".to_string(),
                model_id: Some("model-b".to_string()),
                at_epoch_secs: 42,
                detail: None,
            }],
        };

        let diagnostics = runtime_operator_diagnostics(&snapshot);
        assert!(diagnostics.summary.contains("warm=1/2"));
        assert!(diagnostics.summary.contains("active=1"));
        assert!(diagnostics.summary.contains("queued=2"));
        assert!(diagnostics.summary.contains("loading=model-b"));
        assert!(diagnostics.summary.contains("queue_peak=3"));
        assert!(diagnostics.summary.contains("load_waits=2"));
        assert!(
            diagnostics
                .highlights
                .iter()
                .any(|item| item == "live=fresh")
        );
        assert!(
            diagnostics
                .highlights
                .iter()
                .any(|item| item == "loading_states=1")
        );
        assert!(
            diagnostics
                .highlights
                .iter()
                .any(|item| item == "memory_pressure_evictions=1")
        );
    }
}
