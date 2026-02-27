use anyhow::Result;
use deepseek_core::{AppConfig, SessionState};
use deepseek_mcp::McpManager;
use deepseek_store::Store;
use deepseek_tools::PluginManager;
use deepseek_ui::UiStatus;
use serde_json::json;
use std::path::Path;
use uuid::Uuid;

use crate::UsageArgs;
use crate::output::*;
use crate::util::*;

pub(crate) fn current_ui_status(
    cwd: &Path,
    cfg: &AppConfig,
    force_max_think: bool,
) -> Result<UiStatus> {
    let store = Store::new(cwd)?;
    let session = store.load_latest_session()?;
    let projection = if let Some(session) = &session {
        store.rebuild_from_events(session.session_id)?
    } else {
        Default::default()
    };
    let usage = store.usage_summary(session.as_ref().map(|s| s.session_id), None)?;
    let autopilot_running = store
        .load_latest_autopilot_run()?
        .is_some_and(|run| run.status == "running");
    let background_jobs = store
        .list_background_jobs()?
        .into_iter()
        .filter(|job| job.status == "running")
        .count();
    let pending_approvals = projection
        .tool_invocations
        .len()
        .saturating_sub(projection.approved_invocations.len());
    let pr_review_status = resolve_pr_review_status(
        &store,
        session.as_ref().map(|s| s.session_id),
        projection.review_ids.len(),
    )?;
    let estimated_cost_usd = (usage.input_tokens as f64 / 1_000_000.0)
        * cfg.usage.cost_per_million_input
        + (usage.output_tokens as f64 / 1_000_000.0) * cfg.usage.cost_per_million_output;

    // Estimate current context window usage from transcript content size.
    // Each token is ~4 chars on average. This approximates how much of the
    // context window the next LLM call would consume (not cumulative API usage).
    // If the session is already completed/failed, show 0 — a new session will start.
    let is_terminal_session = session
        .as_ref()
        .is_some_and(|s| matches!(s.status, SessionState::Completed | SessionState::Failed));
    let estimated_context_tokens = if is_terminal_session {
        0
    } else {
        let transcript_chars: u64 = projection.transcript.iter().map(|t| t.len() as u64).sum();
        transcript_chars / 4
    };

    Ok(UiStatus {
        model: if force_max_think {
            format!("{} (thinking)", cfg.llm.base_model)
        } else {
            cfg.llm.base_model.clone()
        },
        pending_approvals,
        estimated_cost_usd,
        background_jobs,
        autopilot_running,
        permission_mode: projection
            .permission_mode
            .clone()
            .unwrap_or_else(|| cfg.policy.permission_mode.to_string()),
        active_tasks: projection.task_ids.len(),
        context_used_tokens: estimated_context_tokens,
        context_max_tokens: cfg.llm.context_window_tokens,
        session_turns: projection.transcript.len(),
        working_directory: cwd.display().to_string(),
        pr_review_status,
        pr_url: None,
        agent_mode: String::new(),
    })
}

fn resolve_pr_review_status(
    store: &Store,
    session_id: Option<Uuid>,
    started_reviews: usize,
) -> Result<Option<String>> {
    if let Ok(raw) = std::env::var("DEEPSEEK_PR_REVIEW_STATUS")
        && let Some(status) = normalize_review_status(&raw)
    {
        return Ok(Some(status.to_string()));
    }

    let Some(session_id) = session_id else {
        return Ok(None);
    };
    let reviews = store.list_review_runs(session_id)?;
    if started_reviews > reviews.len() {
        return Ok(Some("pending".to_string()));
    }
    let Some(latest) = reviews.first() else {
        return Ok(None);
    };
    let target = latest.target.to_ascii_lowercase();
    if target.contains("merged") {
        return Ok(Some("merged".to_string()));
    }
    if target.contains("draft") {
        return Ok(Some("draft".to_string()));
    }
    if latest.critical_count > 0 || latest.findings_count > 0 {
        return Ok(Some("changes_requested".to_string()));
    }
    Ok(Some("approved".to_string()))
}

fn normalize_review_status(value: &str) -> Option<&'static str> {
    let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "approved" => Some("approved"),
        "pending" => Some("pending"),
        "changes_requested" => Some("changes_requested"),
        "draft" => Some("draft"),
        "merged" => Some("merged"),
        _ => None,
    }
}

pub(crate) fn run_status(cwd: &Path, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;
    let store = Store::new(cwd)?;
    let session = store.load_latest_session()?;
    let plugin_manager = PluginManager::new(cwd)?;
    let plugins = plugin_manager.list().unwrap_or_default();
    let mcp_servers = McpManager::new(cwd)
        .and_then(|manager| manager.list_servers())
        .unwrap_or_default();

    let payload = if let Some(session) = session {
        let projection = store.rebuild_from_events(session.session_id)?;
        let usage = store.usage_summary(Some(session.session_id), None)?;
        let max_tokens = session.budgets.max_think_tokens.max(1) as f64;
        let context_usage_pct =
            (((usage.input_tokens + usage.output_tokens) as f64 / max_tokens) * 100.0).min(100.0);
        let pending_approvals = projection
            .tool_invocations
            .len()
            .saturating_sub(projection.approved_invocations.len());
        let latest_autopilot = store.load_latest_autopilot_run()?;
        json!({
            "session_id": session.session_id,
            "state": session.status,
            "active_plan_id": session.active_plan_id,
            "model": {
                "profile": cfg.llm.profile,
                "base": cfg.llm.base_model,
                "max_think": cfg.llm.base_model,
                "thinking_mode": "auto",
            },
            "context_usage_percent": context_usage_pct,
            "pending_approvals": pending_approvals,
            "plugins": {
                "installed": plugins.len(),
                "enabled": plugins.iter().filter(|p| p.enabled).count(),
            },
            "permissions": {
                "approve_bash": cfg.policy.approve_bash,
                "approve_edits": cfg.policy.approve_edits,
                "sandbox_mode": cfg.policy.sandbox_mode,
                "allowlist_entries": cfg.policy.allowlist.len(),
            },
            "mcp_servers": mcp_servers.len(),
            "autopilot": latest_autopilot.map(|run| json!({
                "run_id": run.run_id,
                "status": run.status,
                "completed_iterations": run.completed_iterations,
                "failed_iterations": run.failed_iterations,
            })),
        })
    } else {
        json!({
            "session_id": null,
            "state": "none",
            "model": {
                "profile": cfg.llm.profile,
                "base": cfg.llm.base_model,
                "max_think": cfg.llm.base_model,
                "thinking_mode": "auto",
            },
            "context_usage_percent": 0.0,
            "pending_approvals": 0,
            "plugins": {
                "installed": plugins.len(),
                "enabled": plugins.iter().filter(|p| p.enabled).count(),
            },
            "permissions": {
                "approve_bash": cfg.policy.approve_bash,
                "approve_edits": cfg.policy.approve_edits,
                "sandbox_mode": cfg.policy.sandbox_mode,
                "allowlist_entries": cfg.policy.allowlist.len(),
            },
            "mcp_servers": mcp_servers.len(),
            "autopilot": null,
        })
    };

    if json_mode {
        print_json(&payload)?;
    } else {
        println!(
            "session={} state={} model={}/{}/{} context={:.1}% pending_approvals={} plugins={}/{}",
            payload["session_id"].as_str().unwrap_or("none"),
            payload["state"].as_str().unwrap_or("unknown"),
            payload["model"]["profile"].as_str().unwrap_or_default(),
            payload["model"]["base"].as_str().unwrap_or_default(),
            payload["model"]["max_think"].as_str().unwrap_or_default(),
            payload["context_usage_percent"].as_f64().unwrap_or(0.0),
            payload["pending_approvals"].as_u64().unwrap_or(0),
            payload["plugins"]["enabled"].as_u64().unwrap_or(0),
            payload["plugins"]["installed"].as_u64().unwrap_or(0),
        );
        println!(
            "mcp_servers={}",
            payload["mcp_servers"].as_u64().unwrap_or(0)
        );
        println!(
            "permissions bash={} edits={} sandbox={} allowlist={}",
            payload["permissions"]["approve_bash"]
                .as_str()
                .unwrap_or_default(),
            payload["permissions"]["approve_edits"]
                .as_str()
                .unwrap_or_default(),
            payload["permissions"]["sandbox_mode"]
                .as_str()
                .unwrap_or_default(),
            payload["permissions"]["allowlist_entries"]
                .as_u64()
                .unwrap_or(0),
        );
        if !payload["autopilot"].is_null() {
            println!(
                "autopilot run={} status={} completed={} failed={}",
                payload["autopilot"]["run_id"].as_str().unwrap_or_default(),
                payload["autopilot"]["status"].as_str().unwrap_or_default(),
                payload["autopilot"]["completed_iterations"]
                    .as_u64()
                    .unwrap_or(0),
                payload["autopilot"]["failed_iterations"]
                    .as_u64()
                    .unwrap_or(0),
            );
        }
    }

    Ok(())
}

pub(crate) fn run_usage(cwd: &Path, args: UsageArgs, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;
    let store = Store::new(cwd)?;
    let session_id = if args.session {
        store.load_latest_session()?.map(|s| s.session_id)
    } else {
        None
    };
    let lookback_hours = if args.day { Some(24) } else { None };
    let usage = store.usage_summary(session_id, lookback_hours)?;
    let compactions = store.list_context_compactions(session_id)?;
    let input_cost = (usage.input_tokens as f64 / 1_000_000.0) * cfg.usage.cost_per_million_input;
    let output_cost =
        (usage.output_tokens as f64 / 1_000_000.0) * cfg.usage.cost_per_million_output;
    let rate_limit_events = estimate_rate_limit_events(cwd);
    let payload = json!({
        "scope": {
            "session": session_id,
            "last_hours": lookback_hours,
        },
        "input_tokens": usage.input_tokens,
        "cache_hit_tokens": usage.cache_hit_tokens,
        "cache_miss_tokens": usage.cache_miss_tokens,
        "output_tokens": usage.output_tokens,
        "records": usage.records,
        "estimated_cost_usd": input_cost + output_cost,
        "compactions": compactions.len(),
        "rate_limit_events": rate_limit_events,
    });

    if json_mode {
        print_json(&payload)?;
    } else {
        println!(
            "input_tokens={} (cache={:.1}%) output_tokens={} estimated_cost_usd={:.6} compactions={} rate_limits={}",
            usage.input_tokens,
            if usage.input_tokens > 0 {
                (usage.cache_hit_tokens as f64 / usage.input_tokens as f64) * 100.0
            } else {
                0.0
            },
            usage.output_tokens,
            input_cost + output_cost,
            compactions.len(),
            rate_limit_events,
        );
    }

    Ok(())
}

pub(crate) fn run_context(cwd: &Path, json_mode: bool) -> Result<()> {
    let config = AppConfig::load(cwd).unwrap_or_default();
    let store = Store::new(cwd)?;

    let context_window = config.llm.context_window_tokens;
    let compact_threshold = config.context.auto_compact_threshold;

    // Load latest session to compute token usage with per-unit breakdown
    let session = store.load_latest_session()?;
    let (session_tokens, compactions, unit_breakdown) = if let Some(ref s) = session {
        let usage = store.usage_summary(Some(s.session_id), None)?;
        let compactions = store.list_context_compactions(Some(s.session_id))?;
        let by_unit = store.usage_by_unit(s.session_id)?;
        (
            usage.input_tokens + usage.output_tokens,
            compactions.len(),
            by_unit,
        )
    } else {
        (0, 0, Vec::new())
    };

    // Compute memory token estimate from DEEPSEEK.md content
    let memory_tokens = {
        let mem = deepseek_memory::MemoryManager::new(cwd).ok();
        let text = mem
            .and_then(|m| m.read_combined_memory().ok())
            .unwrap_or_default();
        // Rough estimate: ~4 characters per token for English text
        (text.len() as u64) / 4
    };

    // Estimate system prompt tokens from config (model instructions, tool definitions, etc.)
    // Count declared tools from config + built-in set to approximate system prompt size
    let system_prompt_tokens = {
        let base_instructions: u64 = 800; // core system instructions
        let tool_defs: u64 = config.policy.allowlist.len() as u64 * 40; // ~40 tokens per tool definition
        let safety_rules: u64 = 400; // permission/safety rules
        base_instructions + tool_defs + safety_rules
    };

    // Compute per-unit tokens (Planner vs Executor)
    let planner_tokens: u64 = unit_breakdown
        .iter()
        .filter(|u| u.unit.contains("Planner"))
        .map(|u| u.input_tokens + u.output_tokens)
        .sum();
    let executor_tokens: u64 = unit_breakdown
        .iter()
        .filter(|u| u.unit.contains("Executor"))
        .map(|u| u.input_tokens + u.output_tokens)
        .sum();

    // Conversation tokens = total minus system/memory overhead
    let conversation_tokens = session_tokens.saturating_sub(system_prompt_tokens + memory_tokens);

    let utilization = if context_window > 0 {
        (session_tokens as f64 / context_window as f64) * 100.0
    } else {
        0.0
    };

    if json_mode {
        print_json(&json!({
            "context_window_tokens": context_window,
            "auto_compact_threshold": compact_threshold,
            "session_tokens_used": session_tokens,
            "utilization_pct": format!("{utilization:.1}"),
            "compactions": compactions,
            "breakdown": {
                "system_prompt": system_prompt_tokens,
                "conversation": conversation_tokens,
                "memory": memory_tokens,
                "planner": planner_tokens,
                "executor": executor_tokens,
            }
        }))?;
    } else {
        render_context_bar(&ContextBreakdown {
            total: context_window,
            system: system_prompt_tokens,
            memory: memory_tokens,
            conversation: conversation_tokens,
            planner: planner_tokens,
            executor: executor_tokens,
            compactions: compactions as u64,
            compact_threshold,
        });
    }
    Ok(())
}

struct ContextBreakdown {
    total: u64,
    system: u64,
    memory: u64,
    conversation: u64,
    planner: u64,
    executor: u64,
    compactions: u64,
    compact_threshold: f32,
}

/// Render a colored ASCII bar graph for context window usage.
#[allow(clippy::too_many_arguments)]
fn render_context_bar(b: &ContextBreakdown) {
    let ContextBreakdown {
        total,
        system,
        memory,
        conversation,
        planner,
        executor,
        compactions,
        compact_threshold,
    } = *b;
    let bar_width: u64 = 40;
    let used = system + memory + conversation;
    let available = total.saturating_sub(used);

    let pct = if total > 0 {
        (used as f64 / total as f64 * 100.0) as u64
    } else {
        0
    };

    // Compute character widths for each segment
    let sys_chars = if total > 0 {
        (system as f64 / total as f64 * bar_width as f64).round() as u64
    } else {
        0
    };
    let mem_chars = if total > 0 {
        (memory as f64 / total as f64 * bar_width as f64).round() as u64
    } else {
        0
    };
    let conv_chars = if total > 0 {
        (conversation as f64 / total as f64 * bar_width as f64).round() as u64
    } else {
        0
    };
    let filled = sys_chars + mem_chars + conv_chars;
    let avail_chars = bar_width.saturating_sub(filled);

    // Use ANSI colors: blue=system, cyan=memory, green=conversation, dark_gray=available
    let sys_bar = "█".repeat(sys_chars as usize);
    let mem_bar = "█".repeat(mem_chars as usize);
    let conv_bar = "█".repeat(conv_chars as usize);
    let avail_bar = "░".repeat(avail_chars as usize);

    println!("Context Window Inspector");
    println!("========================");
    println!(
        "\n  [\x1b[34m{sys_bar}\x1b[36m{mem_bar}\x1b[32m{conv_bar}\x1b[90m{avail_bar}\x1b[0m] {pct}% ({used}/{}K tokens)",
        total / 1000
    );
    println!(
        "  \x1b[34m████\x1b[0m System prompt:    ~{}K",
        system / 1000
    );
    println!(
        "  \x1b[36m████\x1b[0m Memory:           ~{}K",
        memory / 1000
    );
    println!(
        "  \x1b[32m████\x1b[0m Conversation:     ~{}K",
        conversation / 1000
    );
    if planner > 0 {
        println!("       Planner:          {planner}");
    }
    if executor > 0 {
        println!("       Executor:         {executor}");
    }
    println!(
        "  \x1b[90m░░░░\x1b[0m Available:        ~{}K",
        available / 1000
    );
    println!("\n  Compactions: {compactions}");

    let threshold_pct = compact_threshold as f64 * 100.0;
    if (pct as f64) > threshold_pct {
        println!(
            "\n  \x1b[33m⚠ Context is above compact threshold ({threshold_pct:.0}%). Use /compact to free space.\x1b[0m"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_grid_renders_bar() {
        // Verify rendering doesn't panic with typical inputs
        render_context_bar(&ContextBreakdown {
            total: 128_000,
            system: 1_200,
            memory: 500,
            conversation: 10_000,
            planner: 0,
            executor: 0,
            compactions: 0,
            compact_threshold: 0.95,
        });
    }

    #[test]
    fn context_grid_percentages() {
        // Zero total should not panic
        render_context_bar(&ContextBreakdown {
            total: 0,
            system: 0,
            memory: 0,
            conversation: 0,
            planner: 0,
            executor: 0,
            compactions: 0,
            compact_threshold: 0.95,
        });
        // Full usage triggers threshold warning
        render_context_bar(&ContextBreakdown {
            total: 100_000,
            system: 50_000,
            memory: 10_000,
            conversation: 40_000,
            planner: 20_000,
            executor: 20_000,
            compactions: 3,
            compact_threshold: 0.50,
        });
    }
}
