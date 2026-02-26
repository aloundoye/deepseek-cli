use anyhow::{Result, anyhow};
use base64::Engine;
use chrono::Utc;
use deepseek_agent::{AgentEngine, ChatMode, ChatOptions};
use deepseek_chrome::{ChromeSession, ScreenshotFormat};
use deepseek_context::ContextManager;
use deepseek_core::{
    AppConfig, ApprovedToolCall, EventKind, StreamChunk, ToolCall, ToolHost, runtime_dir,
    stream_chunk_to_event_json,
};
use deepseek_mcp::McpManager;
use deepseek_memory::{ExportFormat, MemoryManager};
use deepseek_policy::PolicyEngine;
use deepseek_skills::SkillManager;
use deepseek_store::{Store, SubagentRunRecord};
use deepseek_tools::LocalToolHost;
use deepseek_ui::{
    KeyBindings, SlashCommand, TuiStreamEvent, TuiTheme, load_keybindings, render_statusline,
    run_tui_shell_with_bindings,
};
use serde_json::json;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use uuid::Uuid;

// Shared helpers
use crate::context::*;
use crate::output::*;
use crate::util::*;

// CLI types
use crate::{
    Cli, CompactArgs, ConfigCmd, DoctorArgs, DoctorModeArg, ExportArgs, McpCmd, McpGetArgs,
    McpRemoveArgs, MemoryCmd, MemoryEditArgs, MemoryShowArgs, MemorySyncArgs, RewindArgs, RunArgs,
    SearchArgs, SkillRunArgs, SkillsCmd, UsageArgs,
};

// Commands that chat dispatches to
use crate::commands::admin::doctor_payload;
use crate::commands::admin::run_config;
use crate::commands::admin::run_doctor;
use crate::commands::admin::{parse_permissions_cmd, permissions_payload, run_permissions};
use crate::commands::background::background_payload;
use crate::commands::background::{parse_background_cmd, run_background};
use crate::commands::compact::{compact_now, rewind_now, run_compact, run_rewind};
use crate::commands::git::{
    git_commit_interactive, git_commit_staged, git_diff as git_diff_output, git_stage,
    git_status_summary, git_unstage,
};
use crate::commands::mcp::run_mcp;
use crate::commands::memory::{run_export, run_memory};
use crate::commands::remote_env::{parse_remote_env_cmd, remote_env_now, run_remote_env};
use crate::commands::search::run_search;
use crate::commands::skills::run_skills;
use crate::commands::status::{current_ui_status, run_context, run_status, run_usage};
use crate::commands::teleport::{parse_teleport_args, run_teleport, teleport_now};
use crate::commands::visual::{parse_visual_cmd, run_visual, visual_payload};

fn is_max_think_selection(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    lower.contains("reasoner") || lower.contains("max") || lower.contains("high")
}

fn truncate_inline(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        text.to_string()
    } else {
        format!("{}...", &text[..text.floor_char_boundary(max_chars)])
    }
}

fn parse_commit_message(args: &[String]) -> Result<Option<String>> {
    if args.is_empty() {
        return Ok(None);
    }
    if matches!(
        args.first().map(String::as_str),
        Some("-m") | Some("--message")
    ) {
        let message = args[1..].join(" ").trim().to_string();
        if message.is_empty() {
            return Err(anyhow!("commit message cannot be empty"));
        }
        return Ok(Some(message));
    }
    Ok(Some(args.join(" ")))
}

fn parse_stage_args(args: &[String]) -> (bool, Vec<String>) {
    let mut all = false;
    let mut files = Vec::new();
    for arg in args {
        if matches!(arg.as_str(), "--all" | "-A") {
            all = true;
        } else {
            files.push(arg.clone());
        }
    }
    (all, files)
}

fn parse_diff_args(args: &[String]) -> (bool, bool) {
    let staged = args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--staged" | "--cached" | "-s"));
    let stat = args.iter().any(|arg| arg == "--stat");
    (staged, stat)
}

fn parse_chat_mode_name(raw: &str) -> Option<ChatMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "ask" => Some(ChatMode::Ask),
        "code" => Some(ChatMode::Code),
        "architect" | "plan" => Some(ChatMode::Architect),
        "context" => Some(ChatMode::Context),
        "agent" => Some(ChatMode::Agent),
        _ => None,
    }
}

fn chat_mode_name(mode: ChatMode) -> &'static str {
    match mode {
        ChatMode::Ask => "ask",
        ChatMode::Code => "code",
        ChatMode::Architect => "architect",
        ChatMode::Context => "context",
        ChatMode::Agent => "agent",
    }
}

fn resolve_chat_profile_path(cwd: &Path, args: &[String]) -> PathBuf {
    if let Some(first) = args.first() {
        resolve_additional_dir(cwd, first)
    } else {
        runtime_dir(cwd).join("chat_profile.json")
    }
}

fn slash_save_profile_output(
    cwd: &Path,
    args: &[String],
    mode: ChatMode,
    read_only: bool,
    thinking_enabled: bool,
    additional_dirs: &[PathBuf],
) -> Result<String> {
    let path = resolve_chat_profile_path(cwd, args);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let payload = json!({
        "schema": "deepseek.chat_profile.v1",
        "mode": chat_mode_name(mode),
        "read_only": read_only,
        "thinking_enabled": thinking_enabled,
        "additional_dirs": additional_dirs.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>(),
    });
    std::fs::write(&path, serde_json::to_vec_pretty(&payload)?)?;
    Ok(format!("saved chat profile: {}", path.display()))
}

fn slash_load_profile_output(
    cwd: &Path,
    args: &[String],
) -> Result<(ChatMode, bool, bool, Vec<PathBuf>, String)> {
    let path = resolve_chat_profile_path(cwd, args);
    let raw = std::fs::read_to_string(&path)?;
    let payload: serde_json::Value = serde_json::from_str(&raw)?;
    let mode = payload
        .get("mode")
        .and_then(|v| v.as_str())
        .and_then(parse_chat_mode_name)
        .unwrap_or(ChatMode::Code);
    let read_only = payload
        .get("read_only")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let thinking_enabled = payload
        .get("thinking_enabled")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let additional_dirs = payload
        .get("additional_dirs")
        .and_then(|v| v.as_array())
        .map(|rows| {
            rows.iter()
                .filter_map(|row| row.as_str())
                .map(PathBuf::from)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let output = format!(
        "loaded chat profile: {} mode={} read_only={} thinking={}",
        path.display(),
        chat_mode_name(mode),
        read_only,
        if thinking_enabled { "enabled" } else { "auto" }
    );
    Ok((mode, read_only, thinking_enabled, additional_dirs, output))
}

fn slash_git_output(cwd: &Path, args: &[String]) -> Result<String> {
    if args.is_empty() {
        return run_process(cwd, "git", &["status", "--short"]);
    }
    let command = args.iter().map(String::as_str).collect::<Vec<_>>();
    run_process(cwd, "git", &command)
}

fn slash_voice_output(args: &[String]) -> Result<String> {
    if args.first().is_some_and(|v| v == "status") {
        return Ok(format!(
            "voice status: ffmpeg={} sox={} arecord={}",
            command_exists("ffmpeg"),
            command_exists("sox"),
            command_exists("arecord")
        ));
    }
    Ok(
        "voice scaffold: not configured for capture in this build. Use text input or `/voice status` for capability checks."
            .to_string(),
    )
}

fn scan_watch_comment_payload(cwd: &Path) -> Option<(u64, String)> {
    let output = run_process(
        cwd,
        "rg",
        &[
            "-n",
            "--hidden",
            "--glob",
            "!.git/**",
            "--glob",
            "!target/**",
            "--glob",
            "!.deepseek/**",
            "--ignore-file",
            ".deepseekignore",
            "TODO\\(ai\\)|FIXME\\(ai\\)|AI:",
            ".",
        ],
    )
    .ok()?;
    let trimmed = output.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut hasher = DefaultHasher::new();
    trimmed.hash(&mut hasher);
    let digest = hasher.finish();
    Some((digest, trimmed.to_string()))
}

fn slash_stage_output(cwd: &Path, args: &[String]) -> Result<String> {
    let (all, files) = parse_stage_args(args);
    let summary = git_stage(cwd, all, &files)?;
    Ok(format!(
        "staged changes: staged={} unstaged={} untracked={}",
        summary.staged, summary.unstaged, summary.untracked
    ))
}

fn slash_unstage_output(cwd: &Path, args: &[String]) -> Result<String> {
    let (all, files) = parse_stage_args(args);
    let summary = git_unstage(cwd, all, &files)?;
    Ok(format!(
        "unstaged changes: staged={} unstaged={} untracked={}",
        summary.staged, summary.unstaged, summary.untracked
    ))
}

fn slash_diff_output(cwd: &Path, args: &[String]) -> Result<String> {
    let (staged, stat) = parse_diff_args(args);
    let output = git_diff_output(cwd, staged, stat)?;
    if output.trim().is_empty() {
        Ok("(no diff)".to_string())
    } else {
        Ok(output)
    }
}

fn slash_commit_output(cwd: &Path, cfg: &AppConfig, args: &[String]) -> Result<String> {
    let summary = git_status_summary(cwd)?;
    if summary.staged == 0 {
        return Err(anyhow!("no staged changes to commit. Run /stage first."));
    }
    if let Some(message) = parse_commit_message(args)? {
        let output = git_commit_staged(cwd, cfg, &message)?;
        return Ok(if output.trim().is_empty() {
            format!("committed staged changes: {message}")
        } else {
            output
        });
    }
    git_commit_interactive(cwd, cfg)?;
    Ok("committed staged changes".to_string())
}

fn slash_add_dirs(cwd: &Path, dirs: &mut Vec<PathBuf>, args: &[String]) -> Result<String> {
    if args.is_empty() {
        return Err(anyhow!("usage: /add <path> [path ...]"));
    }
    let mut added = Vec::new();
    for raw in args {
        let path = resolve_additional_dir(cwd, raw);
        if path.exists() && path.is_dir() && !dirs.contains(&path) {
            dirs.push(path.clone());
            added.push(path);
        }
    }
    if added.is_empty() {
        return Ok("no new directories added".to_string());
    }
    Ok(format!(
        "added: {}",
        added
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

fn slash_drop_dirs(cwd: &Path, dirs: &mut Vec<PathBuf>, args: &[String]) -> Result<String> {
    if args.is_empty() {
        return Err(anyhow!("usage: /drop <path> [path ...]"));
    }
    let mut removed = Vec::new();
    for raw in args {
        let target = resolve_additional_dir(cwd, raw);
        if let Some(pos) = dirs.iter().position(|dir| dir == &target) {
            removed.push(dirs.remove(pos));
        }
    }
    if removed.is_empty() {
        return Ok("no matching directories were active".to_string());
    }
    Ok(format!(
        "dropped: {}",
        removed
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

fn local_tool_output(cwd: &Path, call: ToolCall) -> Result<serde_json::Value> {
    let cfg = AppConfig::load(cwd).unwrap_or_default();
    let policy = PolicyEngine::from_app_config(&cfg.policy);
    let tool_host = LocalToolHost::new(cwd, policy)?;
    let proposal = tool_host.propose(call);
    let result = tool_host.execute(ApprovedToolCall {
        invocation_id: proposal.invocation_id,
        call: proposal.call,
    });
    if !result.success {
        return Err(anyhow!(
            "{}",
            result
                .output
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("tool execution failed")
        ));
    }
    Ok(result.output)
}

fn slash_run_output(cwd: &Path, args: &[String]) -> Result<String> {
    if args.is_empty() {
        return Err(anyhow!("usage: /run <command>"));
    }
    let command = args.join(" ");
    let output = local_tool_output(
        cwd,
        ToolCall {
            name: "bash.run".to_string(),
            args: json!({
                "cmd": command,
                "timeout": 120,
            }),
            requires_approval: false,
        },
    )?;
    let stdout = output
        .get("stdout")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let stderr = output
        .get("stderr")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let mut out = String::new();
    if !stdout.trim().is_empty() {
        out.push_str(stdout.trim());
    }
    if !stderr.trim().is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(stderr.trim());
    }
    if out.is_empty() {
        Ok("command completed with no output".to_string())
    } else {
        Ok(out)
    }
}

fn inferred_test_command(cwd: &Path) -> Option<String> {
    if cwd.join("Cargo.toml").exists() {
        return Some("cargo test -q".to_string());
    }
    if cwd.join("package.json").exists() {
        return Some("npm test -- --runInBand".to_string());
    }
    if cwd.join("pyproject.toml").exists() || cwd.join("requirements.txt").exists() {
        return Some("pytest -q".to_string());
    }
    None
}

fn inferred_lint_command(cwd: &Path) -> Option<String> {
    if cwd.join("Cargo.toml").exists() {
        return Some("cargo clippy --all-targets -- -D warnings".to_string());
    }
    if cwd.join("package.json").exists() {
        return Some("npm run lint".to_string());
    }
    if cwd.join("pyproject.toml").exists() {
        return Some("ruff check .".to_string());
    }
    None
}

fn slash_test_output(cwd: &Path, args: &[String]) -> Result<String> {
    let command = if args.is_empty() {
        inferred_test_command(cwd).ok_or_else(|| {
            anyhow!("could not infer a test command for this repo; pass one: /test <command>")
        })?
    } else {
        args.join(" ")
    };
    slash_run_output(cwd, &[command])
}

fn slash_lint_output(cwd: &Path, args: &[String], cfg: Option<&AppConfig>) -> Result<String> {
    if !args.is_empty() {
        return slash_run_output(cwd, &[args.join(" ")]);
    }

    // Try configured lint commands from AppConfig.agent_loop.lint (derive per changed files)
    if let Some(cfg) = cfg {
        let lint_cfg = &cfg.agent_loop.lint;
        if lint_cfg.enabled && !lint_cfg.commands.is_empty() {
            let changed = changed_file_list(cwd);
            let commands =
                deepseek_agent::linter::derive_lint_commands(lint_cfg, &changed);
            if !commands.is_empty() {
                let mut results = Vec::new();
                for cmd in &commands {
                    let output = slash_run_output(cwd, &[cmd.clone()])?;
                    results.push(format!("$ {cmd}\n{output}"));
                }
                return Ok(results.join("\n\n"));
            }
        }
    }

    // Fallback to inferred command
    let command = inferred_lint_command(cwd).ok_or_else(|| {
        anyhow!("could not infer a lint command for this repo; pass one: /lint <command>")
    })?;
    slash_run_output(cwd, &[command])
}

/// List changed file paths (staged + unstaged) for lint command derivation.
fn changed_file_list(cwd: &Path) -> Vec<String> {
    let Ok(output) = run_process(cwd, "git", &["status", "--porcelain"]) else {
        return Vec::new();
    };
    output
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.len() < 3 {
                return None;
            }
            Some(trimmed[3..].trim().to_string())
        })
        .collect()
}

fn slash_tokens_output(cwd: &Path, cfg: &AppConfig) -> Result<String> {
    let context_window = cfg.llm.context_window_tokens;

    // System prompt estimate (model instructions, tool definitions, etc.)
    let system_prompt_est: u64 = 800;

    // Memory (DEEPSEEK.md)
    let memory_tokens = {
        let md_path = cwd.join("DEEPSEEK.md");
        if md_path.exists() {
            let content = std::fs::read_to_string(&md_path).unwrap_or_default();
            estimate_tokens(&content)
        } else {
            0
        }
    };

    // Conversation history from store
    let conversation_tokens: u64 = {
        let store = deepseek_store::Store::new(cwd).ok();
        store
            .and_then(|s| {
                let session = s.load_latest_session().ok()??;
                let projection = s.rebuild_from_events(session.session_id).ok()?;
                let chars: u64 = projection
                    .transcript
                    .iter()
                    .map(|t: &String| t.len() as u64)
                    .sum();
                Some(chars / 4)
            })
            .unwrap_or(0)
    };

    // Repo map estimate
    let repo_map_tokens: u64 = 400; // typical bootstrap overhead

    let total_used = system_prompt_est + memory_tokens + conversation_tokens + repo_map_tokens;
    let reserved = cfg.context.reserved_overhead_tokens + cfg.context.response_budget_tokens;
    let available = context_window.saturating_sub(total_used).saturating_sub(reserved);
    let utilization = if context_window > 0 {
        (total_used as f64 / context_window as f64) * 100.0
    } else {
        0.0
    };

    // Cost estimate (based on tokens used so far)
    let input_cost = (total_used as f64 / 1_000_000.0) * cfg.usage.cost_per_million_input;

    Ok(format!(
        "token breakdown:\n  system_prompt:  ~{system_prompt_est}\n  memory:         ~{memory_tokens}\n  conversation:   ~{conversation_tokens}\n  repo_map:       ~{repo_map_tokens}\n  ────────────────────\n  total_used:     ~{total_used}\n  reserved:       ~{reserved} (overhead={} response={})\n  available:      ~{available}\n  context_window: {context_window}\n  utilization:    {utilization:.1}%\n  est_input_cost: ${input_cost:.4}",
        cfg.context.reserved_overhead_tokens,
        cfg.context.response_budget_tokens,
    ))
}

fn slash_web_output(cwd: &Path, args: &[String]) -> Result<String> {
    if args.is_empty() {
        return Err(anyhow!("usage: /web <query|url>"));
    }

    let first = args.first().map(String::as_str).unwrap_or_default();
    if first.starts_with("http://") || first.starts_with("https://") {
        let output = local_tool_output(
            cwd,
            ToolCall {
                name: "web.fetch".to_string(),
                args: json!({
                    "url": first,
                    "max_bytes": 120000,
                    "timeout": 20,
                }),
                requires_approval: false,
            },
        )?;
        return Ok(render_web_fetch_markdown(first, &output, 140));
    }

    let query = args.join(" ");
    let output = local_tool_output(
        cwd,
        ToolCall {
            name: "web.search".to_string(),
            args: json!({
                "query": query,
                "max_results": 5,
            }),
            requires_approval: false,
        },
    )?;
    let results = output
        .get("results")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut top_extract = None;
    for row in results.iter().take(3) {
        let url = row.get("url").and_then(|v| v.as_str()).unwrap_or_default();
        if !url.starts_with("http://") && !url.starts_with("https://") {
            continue;
        }
        if let Ok(fetch) = local_tool_output(
            cwd,
            ToolCall {
                name: "web.fetch".to_string(),
                args: json!({
                    "url": url,
                    "max_bytes": 64000,
                    "timeout": 18,
                }),
                requires_approval: false,
            },
        ) {
            let preview = fetch_preview_text(&fetch, 28);
            if !preview.is_empty() {
                top_extract = Some((url.to_string(), preview));
                break;
            }
        }
    }

    Ok(render_web_search_markdown(&query, &results, top_extract))
}

fn render_web_fetch_markdown(url: &str, output: &serde_json::Value, max_lines: usize) -> String {
    let status = output.get("status").and_then(|v| v.as_u64()).unwrap_or(0);
    let content_type = output
        .get("content_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let truncated = output
        .get("truncated")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let bytes = output.get("bytes").and_then(|v| v.as_u64()).unwrap_or(0);
    let preview = fetch_preview_text(output, max_lines);

    let mut lines = vec![
        "# Web Fetch".to_string(),
        format!("- URL: {url}"),
        format!("- Status: {status}"),
        format!("- Content-Type: {content_type}"),
        format!("- Bytes: {bytes}"),
        format!("- Truncated: {truncated}"),
        String::new(),
        "## Extract".to_string(),
    ];

    if preview.is_empty() {
        lines.push("(empty)".to_string());
    } else {
        lines.push("```text".to_string());
        lines.push(preview);
        lines.push("```".to_string());
    }

    lines.join("\n")
}

fn render_web_search_markdown(
    query: &str,
    results: &[serde_json::Value],
    top_extract: Option<(String, String)>,
) -> String {
    let mut lines = vec![
        "# Web Search".to_string(),
        format!("- Query: {query}"),
        format!("- Results: {}", results.len()),
        String::new(),
        "## Top Results".to_string(),
    ];

    if results.is_empty() {
        lines.push("(no results)".to_string());
    } else {
        for (idx, row) in results.iter().enumerate() {
            let title = row
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("untitled");
            let url = row.get("url").and_then(|v| v.as_str()).unwrap_or_default();
            let snippet = row
                .get("snippet")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            lines.push(format!("{}. {}", idx + 1, title));
            if !url.is_empty() {
                lines.push(format!("   - {url}"));
            }
            if !snippet.is_empty() {
                let compact = snippet.split_whitespace().collect::<Vec<_>>().join(" ");
                lines.push(format!("   - {}", truncate_inline(&compact, 220)));
            }
        }
    }

    if let Some((url, preview)) = top_extract {
        lines.push(String::new());
        lines.push(format!("## Extract ({url})"));
        lines.push("```text".to_string());
        lines.push(preview);
        lines.push("```".to_string());
    }

    lines.join("\n")
}

fn fetch_preview_text(output: &serde_json::Value, max_lines: usize) -> String {
    let content = output
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    normalize_web_content(content, max_lines)
}

fn normalize_web_content(content: &str, max_lines: usize) -> String {
    let mut out = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let cleaned = trimmed.split_whitespace().collect::<Vec<_>>().join(" ");
        if cleaned.is_empty() {
            continue;
        }
        out.push(cleaned);
        if out.len() >= max_lines {
            break;
        }
    }
    let mut joined = out.join("\n");
    if joined.len() > 12000 {
        joined.truncate(joined.floor_char_boundary(12000));
    }
    joined
}

fn slash_map_output(cwd: &Path, args: &[String], additional_dirs: &[PathBuf]) -> Result<String> {
    let query = if args.is_empty() {
        "project map".to_string()
    } else {
        args.join(" ")
    };
    let mut manager = ContextManager::new(cwd)?;
    manager.analyze_workspace()?;
    let suggestions = manager.suggest_relevant_files(&query, 20);
    let mut lines = vec![format!(
        "repo map: indexed_files={} query=\"{}\"",
        manager.file_count(),
        query
    )];
    if suggestions.is_empty() {
        lines.push("(no scored files)".to_string());
    } else {
        for item in suggestions {
            lines.push(format!(
                "- {} score={:.2} {}",
                item.path.strip_prefix(cwd).unwrap_or(&item.path).display(),
                item.score,
                truncate_inline(&item.reasons.join(" | "), 120)
            ));
        }
    }
    if !additional_dirs.is_empty() {
        lines.push("additional dirs:".to_string());
        for dir in additional_dirs {
            lines.push(format!("- {}", dir.display()));
        }
    }
    Ok(lines.join("\n"))
}

fn agents_payload(cwd: &Path, limit: usize) -> Result<serde_json::Value> {
    let store = Store::new(cwd)?;
    let session_id = store
        .load_latest_session()?
        .map(|session| session.session_id);
    let runs = store.list_subagent_runs(session_id, limit)?;
    Ok(json!({
        "schema": "deepseek.chat.agents.v1",
        "session_id": session_id.map(|id| id.to_string()),
        "count": runs.len(),
        "agents": runs,
    }))
}

fn mission_control_payload(cwd: &Path, limit: usize) -> Result<serde_json::Value> {
    let store = Store::new(cwd)?;
    let session_id = store
        .load_latest_session()?
        .map(|session| session.session_id);
    let tasks = store.list_tasks(session_id)?;
    let subagents = store.list_subagent_runs(session_id, limit)?;
    let running_subagents = subagents
        .iter()
        .filter(|run| run.status.eq_ignore_ascii_case("running"))
        .count();
    let failed_subagents = subagents
        .iter()
        .filter(|run| run.status.eq_ignore_ascii_case("failed"))
        .count();
    Ok(json!({
        "schema": "deepseek.chat.mission_control.v1",
        "session_id": session_id.map(|id| id.to_string()),
        "tasks": tasks,
        "subagents": subagents,
        "summary": {
            "task_count": tasks.len(),
            "subagent_count": subagents.len(),
            "running_subagents": running_subagents,
            "failed_subagents": failed_subagents,
        }
    }))
}

fn render_agents_payload(payload: &serde_json::Value) -> String {
    let runs = payload
        .get("agents")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    if runs.is_empty() {
        return "No subagent runs recorded in this session.".to_string();
    }
    let parsed_runs = runs
        .into_iter()
        .filter_map(|row| serde_json::from_value::<SubagentRunRecord>(row).ok())
        .collect::<Vec<_>>();
    if parsed_runs.is_empty() {
        return "No subagent runs recorded in this session.".to_string();
    }
    let mut lines = vec![format!("Subagents ({} total):", parsed_runs.len())];
    for run in parsed_runs {
        let detail = run
            .output
            .as_deref()
            .or(run.error.as_deref())
            .unwrap_or_default()
            .replace('\n', " ");
        let detail = truncate_inline(&detail, 120);
        lines.push(format!(
            "- {} [{}] {} — {}",
            run.name, run.status, run.run_id, detail
        ));
    }
    lines.join("\n")
}

fn render_mission_control_payload(payload: &serde_json::Value) -> String {
    let tasks = payload
        .get("tasks")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    let subagents = payload
        .get("subagents")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    let mut lines = vec![format!(
        "Mission Control: {} task(s), {} subagent run(s)",
        tasks.len(),
        subagents.len()
    )];
    if tasks.is_empty() {
        lines.push("- Tasks: none".to_string());
    } else {
        lines.push("- Tasks:".to_string());
        for task in tasks.iter().take(10) {
            let title = task.get("title").and_then(|v| v.as_str()).unwrap_or("task");
            let status = task
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let priority = task
                .get("priority")
                .and_then(|v| v.as_u64())
                .unwrap_or_default();
            lines.push(format!("  - {title} [{status}] priority={priority}"));
        }
    }
    if subagents.is_empty() {
        lines.push("- Subagents: none".to_string());
    } else {
        lines.push("- Subagents:".to_string());
        for run in subagents.iter().take(10) {
            let name = run
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("subagent");
            let status = run
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let run_id = run
                .get("run_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            lines.push(format!("  - {name} [{status}] {run_id}"));
        }
    }
    lines.join("\n")
}

pub(crate) fn run_chat(
    cwd: &Path,
    json_mode: bool,
    json_events: bool,
    allow_tools: bool,
    tui: bool,
    cli: Option<&crate::Cli>,
) -> Result<()> {
    use std::io::{IsTerminal, Write, stdin, stdout};

    let cfg = AppConfig::ensure(cwd)?;
    ensure_llm_ready_with_cfg(Some(cwd), &cfg, json_mode)?;
    let mut engine = AgentEngine::new(cwd)?;
    if let Some(cli) = cli {
        apply_cli_flags(&mut engine, cli);
    }
    // Validate API key works before entering chat loop
    engine.validate_api_key()?;
    let mut force_max_think = cli
        .and_then(|v| v.model.as_deref())
        .map(is_max_think_selection)
        .unwrap_or(false);
    let force_execute = cli.map(|v| v.force_execute).unwrap_or(false);
    let force_plan_only = cli.map(|v| v.plan_only).unwrap_or(false);
    let teammate_mode = cli.and_then(|v| v.teammate_mode.clone());
    let repo_root_override = cli.and_then(|v| v.repo.clone());
    let watch_files_enabled = cli.map(|value| value.watch_files).unwrap_or(false);
    let detect_urls = cli.map(|value| value.detect_urls).unwrap_or(false);
    let debug_context = cli.map(|v| v.debug_context).unwrap_or(false)
        || std::env::var("DEEPSEEK_DEBUG_CONTEXT")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
    let interactive_tty = stdin().is_terminal() && stdout().is_terminal();
    if !json_mode && (tui || cfg.ui.enable_tui) && interactive_tty {
        return run_chat_tui(
            cwd,
            allow_tools,
            &cfg,
            force_max_think,
            force_execute,
            force_plan_only,
            teammate_mode.clone(),
            repo_root_override.clone(),
            debug_context,
            detect_urls,
            watch_files_enabled,
        );
    }
    if tui && !interactive_tty {
        return Err(anyhow!("--tui requires an interactive terminal"));
    }
    let mut last_assistant_response: Option<String> = None;
    let mut additional_dirs = cli.map(|value| value.add_dir.clone()).unwrap_or_default();
    let mut read_only_mode = false;
    let mut active_chat_mode = ChatMode::Code;
    let mut last_watch_digest: Option<u64> = None;
    let mut pending_images: Vec<deepseek_core::ImageContent> = vec![];
    if !json_mode {
        println!("deepseek chat (type 'exit' to quit)");
        println!(
            "model: {} thinking=auto approvals: bash={} edits={} tools={}",
            cfg.llm.base_model,
            cfg.policy.approve_bash,
            cfg.policy.approve_edits,
            if allow_tools {
                "enabled"
            } else {
                "approval-gated"
            }
        );
        if watch_files_enabled {
            println!("watch mode: enabled (scans TODO(ai)/FIXME(ai)/AI: hints each turn)");
        }
    }
    loop {
        if !json_mode {
            print!("> ");
            stdout().flush()?;
        }
        let mut line = String::new();
        stdin().read_line(&mut line)?;
        let raw_prompt = line.trim();
        if raw_prompt == "exit" {
            break;
        }
        if raw_prompt.is_empty() {
            continue;
        }

        // Expand @file mentions into inline file content
        let mut prompt_owned = deepseek_ui::expand_at_mentions(raw_prompt);
        if watch_files_enabled
            && let Some((digest, hints)) = scan_watch_comment_payload(cwd)
            && last_watch_digest != Some(digest)
        {
            last_watch_digest = Some(digest);
            prompt_owned
                .push_str("\n\nAUTO_WATCH_CONTEXT_V1\nDetected comment hints in workspace:\n");
            prompt_owned.push_str(&hints);
            prompt_owned.push_str("\nAUTO_WATCH_CONTEXT_END");
        }
        let prompt = prompt_owned.as_str();

        if let Some(cmd) = SlashCommand::parse(prompt) {
            match cmd {
                SlashCommand::Help => {
                    let message = json!({
                        "commands": [
                            "/help",
                            "/ask",
                            "/code",
                            "/architect",
                            "/chat-mode",
                            "/init",
                            "/clear",
                            "/compact",
                            "/memory",
                            "/config",
                            "/model",
                            "/cost",
                            "/mcp",
                            "/rewind",
                            "/export",
                            "/plan",
                            "/teleport",
                            "/remote-env",
                            "/status",
                            "/add",
                            "/drop",
                            "/read-only",
                            "/map",
                            "/map-refresh",
                            "/run",
                            "/test",
                            "/lint",
                            "/web",
                            "/diff",
                            "/stage",
                            "/unstage",
                            "/commit",
                            "/undo",
                            "/effort",
                            "/skills",
                            "/permissions",
                            "/background",
                            "/visual",
                            "/vim",
                            "/copy",
                            "/paste",
                            "/git",
                            "/settings",
                            "/load",
                            "/save",
                            "/voice",
                            "/debug",
                            "/desktop",
                            "/todos",
                            "/chrome",
                            "/exit",
                            "/hooks",
                            "/rename",
                            "/resume",
                            "/stats",
                            "/statusline",
                            "/theme",
                            "/usage",
                            "/add-dir",
                            "/bug",
                            "/pr_comments",
                            "/release-notes",
                            "/login",
                            "/logout",
                        ],
                    });
                    if json_mode {
                        print_json(&message)?;
                    } else {
                        println!("slash commands:");
                        for command in message["commands"].as_array().into_iter().flatten() {
                            if let Some(name) = command.as_str() {
                                println!("- {name}");
                            }
                        }
                    }
                }
                SlashCommand::Ask(_args) => {
                    active_chat_mode = ChatMode::Ask;
                    if json_mode {
                        print_json(&json!({"mode": "ask"}))?;
                    } else {
                        println!("chat mode set to ask");
                    }
                }
                SlashCommand::Code(_args) => {
                    active_chat_mode = ChatMode::Code;
                    if json_mode {
                        print_json(&json!({"mode": "code"}))?;
                    } else {
                        println!("chat mode set to code");
                    }
                }
                SlashCommand::Architect(_args) => {
                    active_chat_mode = ChatMode::Architect;
                    if json_mode {
                        print_json(&json!({"mode": "architect"}))?;
                    } else {
                        println!("chat mode set to architect");
                    }
                }
                SlashCommand::ChatMode(mode) => {
                    let Some(raw_mode) = mode else {
                        if json_mode {
                            print_json(&json!({"mode": chat_mode_name(active_chat_mode)}))?;
                        } else {
                            println!("current chat mode: {}", chat_mode_name(active_chat_mode));
                        }
                        continue;
                    };
                    if let Some(parsed) = parse_chat_mode_name(&raw_mode) {
                        active_chat_mode = parsed;
                        if json_mode {
                            print_json(&json!({"mode": chat_mode_name(active_chat_mode)}))?;
                        } else {
                            println!("chat mode set to {}", chat_mode_name(active_chat_mode));
                        }
                    } else if json_mode {
                        print_json(
                            &json!({"error": format!("unsupported chat mode: {raw_mode}")}),
                        )?;
                    } else {
                        println!(
                            "unsupported chat mode: {raw_mode} (use ask|code|architect|context)"
                        );
                    }
                }
                SlashCommand::Init => {
                    let manager = MemoryManager::new(cwd)?;
                    let path = manager.ensure_initialized()?;
                    let version_id = manager.sync_memory_version("init")?;
                    append_control_event(
                        cwd,
                        EventKind::MemorySyncedV1 {
                            version_id,
                            path: path.to_string_lossy().to_string(),
                            note: "init".to_string(),
                        },
                    )?;
                    if json_mode {
                        print_json(&json!({
                            "initialized": true,
                            "path": path,
                            "version_id": version_id,
                        }))?;
                    } else {
                        println!("initialized memory at {}", path.display());
                    }
                }
                SlashCommand::Clear => {
                    if json_mode {
                        print_json(&json!({"cleared": true}))?;
                    } else {
                        println!("chat buffer cleared");
                    }
                }
                SlashCommand::Compact(focus) => {
                    run_compact(
                        cwd,
                        CompactArgs {
                            from_turn: None,
                            yes: false,
                            focus,
                        },
                        json_mode,
                    )?;
                }
                SlashCommand::Memory(args) => {
                    if args.is_empty() || args[0].eq_ignore_ascii_case("show") {
                        run_memory(cwd, MemoryCmd::Show(MemoryShowArgs {}), json_mode)?;
                    } else if args[0].eq_ignore_ascii_case("edit") {
                        run_memory(cwd, MemoryCmd::Edit(MemoryEditArgs {}), json_mode)?;
                    } else if args[0].eq_ignore_ascii_case("sync") {
                        let note = args.get(1).cloned();
                        run_memory(cwd, MemoryCmd::Sync(MemorySyncArgs { note }), json_mode)?;
                    } else if json_mode {
                        print_json(&json!({"error":"unknown /memory subcommand"}))?;
                    } else {
                        println!("unknown /memory subcommand");
                    }
                }
                SlashCommand::Config => {
                    run_config(cwd, ConfigCmd::Show, json_mode)?;
                }
                SlashCommand::Model(model) => {
                    if let Some(model) = model {
                        force_max_think = is_max_think_selection(&model);
                    }
                    if json_mode {
                        print_json(&json!({
                            "force_max_think": force_max_think,
                            "model": cfg.llm.base_model,
                            "thinking_enabled": force_max_think,
                        }))?;
                    } else if force_max_think {
                        println!("model mode: thinking-enabled ({})", cfg.llm.base_model);
                    } else {
                        println!(
                            "model mode: auto ({} thinking=on-demand)",
                            cfg.llm.base_model
                        );
                    }
                }
                SlashCommand::Cost => {
                    run_usage(
                        cwd,
                        UsageArgs {
                            session: true,
                            day: false,
                        },
                        json_mode,
                    )?;
                }
                SlashCommand::Tokens => {
                    let output = slash_tokens_output(cwd, &cfg)?;
                    if json_mode {
                        print_json(&json!({"tokens": output}))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::Mcp(args) => {
                    if args.is_empty() || args[0].eq_ignore_ascii_case("list") {
                        run_mcp(cwd, McpCmd::List, json_mode)?;
                    } else if args[0].eq_ignore_ascii_case("get") && args.len() > 1 {
                        run_mcp(
                            cwd,
                            McpCmd::Get(McpGetArgs {
                                server_id: args[1].clone(),
                            }),
                            json_mode,
                        )?;
                    } else if args[0].eq_ignore_ascii_case("remove") && args.len() > 1 {
                        run_mcp(
                            cwd,
                            McpCmd::Remove(McpRemoveArgs {
                                server_id: args[1].clone(),
                            }),
                            json_mode,
                        )?;
                    } else if args[0].eq_ignore_ascii_case("prompt") {
                        // H8: MCP prompts as slash commands
                        let mcp = deepseek_mcp::McpManager::new(cwd)?;
                        if args.len() < 3 {
                            // List available prompts across all servers
                            let servers = mcp.list_servers()?;
                            if json_mode {
                                let prompts: Vec<serde_json::Value> = servers.iter().map(|s| {
                                    json!({"server": s.id, "hint": format!("/mcp prompt {} <name> [args...]", s.id)})
                                }).collect();
                                print_json(&json!({"mcp_prompts": prompts}))?;
                            } else {
                                println!("MCP prompts — usage: /mcp prompt <server> <name> [args...]");
                                for s in &servers {
                                    println!("  server: {}", s.id);
                                }
                            }
                        } else {
                            // Invoke: /mcp prompt <server_id> <prompt_name> [arg=val ...]
                            let server_id = &args[1];
                            let prompt_name = &args[2];
                            let prompt_args: serde_json::Value = if args.len() > 3 {
                                let mut map = serde_json::Map::new();
                                for kv in &args[3..] {
                                    if let Some((k, v)) = kv.split_once('=') {
                                        map.insert(k.to_string(), serde_json::Value::String(v.to_string()));
                                    }
                                }
                                serde_json::Value::Object(map)
                            } else {
                                json!({})
                            };
                            let result = deepseek_mcp::execute_mcp_stdio_request(
                                &mcp.get_server(server_id)?
                                    .ok_or_else(|| anyhow::anyhow!("MCP server not found: {server_id}"))?,
                                "prompts/get",
                                json!({"name": prompt_name, "arguments": prompt_args}),
                                3,
                                std::time::Duration::from_secs(10),
                            )?;
                            if json_mode {
                                print_json(&result)?;
                            } else {
                                if let Some(messages) = result.pointer("/result/messages").and_then(|v| v.as_array()) {
                                    for msg in messages {
                                        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("?");
                                        let text = msg.pointer("/content/text").and_then(|v| v.as_str()).unwrap_or("");
                                        println!("[{role}] {text}");
                                    }
                                } else {
                                    println!("{}", serde_json::to_string_pretty(&result)?);
                                }
                            }
                        }
                    } else if json_mode {
                        print_json(&json!({"error":"use /mcp list|get <id>|remove <id>|prompt <server> <name>"}))?;
                    } else {
                        println!("use /mcp list|get <id>|remove <id>|prompt <server> <name>");
                    }
                }
                SlashCommand::Rewind(args) => {
                    let to_checkpoint = args
                        .iter()
                        .find(|arg| !arg.starts_with('-'))
                        .map(ToString::to_string);
                    let yes = true;
                    run_rewind(cwd, RewindArgs { to_checkpoint, yes }, json_mode)?;
                }
                SlashCommand::Export(args) => {
                    let format = args.first().cloned().unwrap_or_else(|| "json".to_string());
                    let output = args.get(1).cloned();
                    run_export(
                        cwd,
                        ExportArgs {
                            session: None,
                            format,
                            output,
                        },
                        json_mode,
                    )?;
                }
                SlashCommand::Plan => {
                    force_max_think = true;
                    if json_mode {
                        print_json(&json!({"plan_mode": true, "thinking_enabled": true}))?;
                    } else {
                        println!(
                            "plan mode active; prompts will prefer structured planning with thinking enabled."
                        );
                    }
                }
                SlashCommand::Teleport(args) => match parse_teleport_args(args) {
                    Ok(teleport_args) => run_teleport(cwd, teleport_args, json_mode)?,
                    Err(err) => {
                        if json_mode {
                            print_json(&json!({"error": err.to_string()}))?;
                        } else {
                            println!("teleport parse error: {err}");
                        }
                    }
                },
                SlashCommand::RemoteEnv(args) => match parse_remote_env_cmd(args) {
                    Ok(command) => run_remote_env(cwd, command, json_mode)?,
                    Err(err) => {
                        if json_mode {
                            print_json(&json!({"error": err.to_string()}))?;
                        } else {
                            println!("remote-env parse error: {err}");
                        }
                    }
                },
                SlashCommand::Add(args) => {
                    let output = slash_add_dirs(cwd, &mut additional_dirs, &args)?;
                    if json_mode {
                        print_json(&json!({"add": output}))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::Drop(args) => {
                    let output = slash_drop_dirs(cwd, &mut additional_dirs, &args)?;
                    if json_mode {
                        print_json(&json!({"drop": output}))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::ReadOnly(args) => {
                    let action = args
                        .first()
                        .map(|v| v.to_ascii_lowercase())
                        .unwrap_or_else(|| "toggle".to_string());
                    match action.as_str() {
                        "on" | "true" | "1" => read_only_mode = true,
                        "off" | "false" | "0" => read_only_mode = false,
                        "status" => {}
                        _ => read_only_mode = !read_only_mode,
                    }
                    let output = format!(
                        "read-only mode: {}",
                        if read_only_mode {
                            "enabled"
                        } else {
                            "disabled"
                        }
                    );
                    if json_mode {
                        print_json(&json!({"read_only": read_only_mode, "message": output}))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::Map(args) => {
                    let output = slash_map_output(cwd, &args, &additional_dirs)?;
                    if json_mode {
                        print_json(&json!({"map": output}))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::MapRefresh(args) => {
                    deepseek_agent::clear_tag_cache();
                    let output = slash_map_output(cwd, &args, &additional_dirs)?;
                    if json_mode {
                        print_json(&json!({"map": output, "refreshed": true}))?;
                    } else {
                        println!("(cache cleared)");
                        println!("{output}");
                    }
                }
                SlashCommand::Run(args) => {
                    let output = slash_run_output(cwd, &args)?;
                    if json_mode {
                        print_json(&json!({"run": output}))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::Test(args) => {
                    let output = slash_test_output(cwd, &args)?;
                    if json_mode {
                        print_json(&json!({"test": output}))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::Lint(args) => {
                    let output = slash_lint_output(cwd, &args, Some(&cfg))?;
                    if json_mode {
                        print_json(&json!({"lint": output}))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::Web(args) => {
                    let output = slash_web_output(cwd, &args)?;
                    if json_mode {
                        print_json(&json!({"web": output}))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::Diff(args) => {
                    let output = slash_diff_output(cwd, &args)?;
                    if json_mode {
                        print_json(&json!({"diff": output}))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::Stage(args) => {
                    let output = slash_stage_output(cwd, &args)?;
                    if json_mode {
                        print_json(&json!({"stage": output}))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::Unstage(args) => {
                    let output = slash_unstage_output(cwd, &args)?;
                    if json_mode {
                        print_json(&json!({"unstage": output}))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::Commit(args) => {
                    let output = slash_commit_output(cwd, &cfg, &args)?;
                    if json_mode {
                        print_json(&json!({"commit": output}))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::Undo(_args) => {
                    let checkpoint_msg = match crate::commands::compact::rewind_now(cwd, None) {
                        Ok(checkpoint) => format!(
                            "rewound to checkpoint {} (files={})",
                            checkpoint.checkpoint_id, checkpoint.files_count
                        ),
                        Err(e) => format!("no files rewound ({e})"),
                    };
                    crate::context::append_control_event(
                        cwd,
                        deepseek_core::EventKind::TurnRevertedV1 { turns_dropped: 1 },
                    )?;
                    let output = format!("Reverted 1 conversation turn. {checkpoint_msg}");
                    if json_mode {
                        crate::output::print_json(&serde_json::json!({
                            "undo": true,
                            "msg": output
                        }))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::Status => run_status(cwd, json_mode)?,
                SlashCommand::Effort(level) => {
                    let level = level.unwrap_or_else(|| "medium".to_string());
                    let normalized = level.to_ascii_lowercase();
                    force_max_think = matches!(normalized.as_str(), "high" | "max");
                    append_control_event(
                        cwd,
                        EventKind::EffortChangedV1 {
                            level: normalized.clone(),
                        },
                    )?;
                    if json_mode {
                        print_json(&json!({
                            "effort": normalized,
                            "thinking_enabled": force_max_think
                        }))?;
                    } else {
                        println!(
                            "effort={} thinking={}",
                            normalized,
                            if force_max_think { "enabled" } else { "auto" }
                        );
                    }
                }
                SlashCommand::Skills(args) => {
                    if args.is_empty() || args[0].eq_ignore_ascii_case("list") {
                        run_skills(cwd, SkillsCmd::List, json_mode)?;
                    } else if args[0].eq_ignore_ascii_case("reload") {
                        run_skills(cwd, SkillsCmd::Reload, json_mode)?;
                    } else if args[0].eq_ignore_ascii_case("run") && args.len() > 1 {
                        let input = if args.len() > 2 {
                            Some(args[2..].join(" "))
                        } else {
                            None
                        };
                        run_skills(
                            cwd,
                            SkillsCmd::Run(SkillRunArgs {
                                skill_id: args[1].clone(),
                                input,
                                execute: false,
                            }),
                            json_mode,
                        )?;
                    } else if json_mode {
                        print_json(&json!({"error":"use /skills list|reload|run <id> [input]"}))?;
                    } else {
                        println!("use /skills list|reload|run <id> [input]");
                    }
                }
                SlashCommand::Permissions(args) => match parse_permissions_cmd(args) {
                    Ok(command) => run_permissions(cwd, command, json_mode)?,
                    Err(err) => {
                        if json_mode {
                            print_json(&json!({"error": err.to_string()}))?;
                        } else {
                            println!("permissions parse error: {err}");
                        }
                    }
                },
                SlashCommand::Background(args) => match parse_background_cmd(args) {
                    Ok(command) => run_background(cwd, command, json_mode)?,
                    Err(err) => {
                        if json_mode {
                            print_json(&json!({"error": err.to_string()}))?;
                        } else {
                            println!("background parse error: {err}");
                        }
                    }
                },
                SlashCommand::Visual(args) => match parse_visual_cmd(args) {
                    Ok(command) => run_visual(cwd, command, json_mode)?,
                    Err(err) => {
                        if json_mode {
                            print_json(&json!({"error": err.to_string()}))?;
                        } else {
                            println!("visual parse error: {err}");
                        }
                    }
                },
                SlashCommand::Context => {
                    run_context(cwd, json_mode)?;
                }
                SlashCommand::Sandbox(args) => {
                    if json_mode {
                        print_json(
                            &json!({"sandbox_mode": cfg.policy.sandbox_mode, "args": args}),
                        )?;
                    } else {
                        println!("Sandbox mode: {}", cfg.policy.sandbox_mode);
                        if !args.is_empty() {
                            println!("(sandbox config changes not yet implemented in REPL)");
                        }
                    }
                }
                SlashCommand::Agents => {
                    let payload = agents_payload(cwd, 20)?;
                    if json_mode {
                        print_json(&payload)?;
                    } else {
                        println!("{}", render_agents_payload(&payload));
                    }
                }
                SlashCommand::Tasks(_args) => {
                    let payload = mission_control_payload(cwd, 20)?;
                    if json_mode {
                        print_json(&payload)?;
                    } else {
                        println!("{}", render_mission_control_payload(&payload));
                    }
                }
                SlashCommand::Review(args) => {
                    if json_mode {
                        print_json(
                            &json!({"review": "use 'deepseek review' subcommand", "args": args}),
                        )?;
                    } else {
                        println!("Use 'deepseek review [--diff|--staged|--pr N]' for code review.");
                        println!("Presets: security, perf, style, PR-ready");
                    }
                }
                SlashCommand::Search(args) => {
                    let query = args.join(" ");
                    if query.is_empty() {
                        println!("Usage: /search <query>");
                    } else {
                        run_search(
                            cwd,
                            SearchArgs {
                                query,
                                max_results: 10,
                            },
                            json_mode,
                        )?;
                    }
                }
                SlashCommand::Vim(args) => {
                    let mode = if args.is_empty() {
                        "toggle"
                    } else {
                        args.first().map(String::as_str).unwrap_or("toggle")
                    };
                    if json_mode {
                        print_json(&json!({
                            "vim": "tui_only",
                            "mode": mode
                        }))?;
                    } else {
                        println!("vim mode is handled in TUI only; requested mode={mode}");
                    }
                }
                SlashCommand::TerminalSetup => {
                    if json_mode {
                        print_json(&json!({"terminal_setup": "configured"}))?;
                    } else {
                        println!("Terminal setup:");
                        println!(
                            "  Shell: {}",
                            std::env::var("SHELL").unwrap_or_else(|_| "unknown".to_string())
                        );
                        println!(
                            "  TERM:  {}",
                            std::env::var("TERM").unwrap_or_else(|_| "unknown".to_string())
                        );
                        let (cols, rows) = std::process::Command::new("stty")
                            .arg("size")
                            .stderr(std::process::Stdio::inherit())
                            .output()
                            .ok()
                            .and_then(|o| {
                                let s = String::from_utf8_lossy(&o.stdout);
                                let mut parts = s.split_whitespace();
                                let r: u16 = parts.next()?.parse().ok()?;
                                let c: u16 = parts.next()?.parse().ok()?;
                                Some((c, r))
                            })
                            .unwrap_or((80, 24));
                        println!("  Cols:  {}", cols);
                        println!("  Rows:  {}", rows);
                    }
                }
                SlashCommand::Keybindings => {
                    let kb_path = AppConfig::keybindings_path().unwrap_or_else(|| {
                        std::path::PathBuf::from("~/.deepseek/keybindings.json")
                    });
                    if json_mode {
                        print_json(&json!({"keybindings_path": kb_path.to_string_lossy()}))?;
                    } else {
                        println!("Keybindings: {}", kb_path.display());
                        if kb_path.exists() {
                            let content = std::fs::read_to_string(&kb_path)?;
                            println!("{content}");
                        } else {
                            println!("(no custom keybindings configured)");
                        }
                    }
                }
                SlashCommand::Doctor => {
                    run_doctor(cwd, DoctorArgs::default(), json_mode)?;
                }
                SlashCommand::Copy => {
                    // Copy last assistant response to clipboard
                    if let Some(ref last) = last_assistant_response {
                        copy_to_clipboard(last);
                        if !json_mode {
                            println!("Copied to clipboard.");
                        }
                    } else if !json_mode {
                        println!("No assistant response to copy.");
                    }
                }
                SlashCommand::Paste => {
                    // Check for image data first
                    if let Some(img_bytes) = read_image_from_clipboard() {
                        use base64::Engine;
                        let b64 = base64::engine::general_purpose::STANDARD.encode(&img_bytes);
                        pending_images.push(deepseek_core::ImageContent {
                            mime: "image/png".to_string(),
                            base64_data: b64,
                        });
                        if json_mode {
                            print_json(&json!({"pasted_image": true, "size_bytes": img_bytes.len()}))?;
                        } else {
                            println!("Image pasted ({} bytes). It will be included in your next prompt.", img_bytes.len());
                        }
                    } else {
                        let pasted = read_from_clipboard().unwrap_or_default();
                        if pasted.trim().is_empty() {
                            if json_mode {
                                print_json(&json!({"pasted": "", "empty": true}))?;
                            } else {
                                println!("clipboard is empty or unavailable");
                            }
                        } else if json_mode {
                            print_json(&json!({"pasted": pasted}))?;
                        } else {
                            println!("{pasted}");
                        }
                    }
                }
                SlashCommand::Git(args) => {
                    let output = slash_git_output(cwd, &args)?;
                    if json_mode {
                        print_json(&json!({"git": output}))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::Settings => {
                    run_config(cwd, ConfigCmd::Show, json_mode)?;
                }
                SlashCommand::Load(args) => {
                    let (mode, read_only, thinking, dirs, output) =
                        slash_load_profile_output(cwd, &args)?;
                    active_chat_mode = mode;
                    read_only_mode = read_only;
                    force_max_think = thinking;
                    additional_dirs = dirs;
                    if json_mode {
                        print_json(&json!({
                            "loaded": true,
                            "mode": chat_mode_name(active_chat_mode),
                            "read_only": read_only_mode,
                            "thinking_enabled": force_max_think,
                            "additional_dirs": additional_dirs.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>()
                        }))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::Save(args) => {
                    let output = slash_save_profile_output(
                        cwd,
                        &args,
                        active_chat_mode,
                        read_only_mode,
                        force_max_think,
                        &additional_dirs,
                    )?;
                    if json_mode {
                        print_json(&json!({"saved": true, "message": output}))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::Voice(args) => {
                    let output = slash_voice_output(&args)?;
                    if json_mode {
                        print_json(&json!({"voice": output}))?;
                    } else {
                        println!("{output}");
                    }
                }
                SlashCommand::Debug(args) => match parse_debug_analysis_args(&args)? {
                    Some(doctor_args) => {
                        run_doctor(cwd, doctor_args, json_mode)?;
                    }
                    None => {
                        let desc = if args.is_empty() {
                            "general".to_string()
                        } else {
                            args.join(" ")
                        };
                        let log_dir = deepseek_core::runtime_dir(cwd).join("logs");
                        if json_mode {
                            print_json(
                                &json!({"debug": desc, "log_dir": log_dir.to_string_lossy()}),
                            )?;
                        } else {
                            println!("Debug: {desc}");
                            println!("Logs: {}", log_dir.display());
                        }
                    }
                },
                SlashCommand::Exit => {
                    break;
                }
                SlashCommand::Hooks(args) => {
                    let hooks_config = &cfg.hooks;
                    if json_mode {
                        print_json(hooks_config)?;
                    } else if args.first().is_some_and(|a| a == "list") {
                        println!("Hooks configuration:");
                        println!(
                            "{}",
                            serde_json::to_string_pretty(hooks_config).unwrap_or_default()
                        );
                    } else {
                        println!("Usage: /hooks list");
                        println!("Configure hooks in .deepseek/settings.json under \"hooks\" key.");
                    }
                }
                SlashCommand::Rename(name) => {
                    if let Some(name) = name {
                        // Store session rename in metadata
                        if json_mode {
                            print_json(&json!({"renamed": name}))?;
                        } else {
                            println!("Session renamed to: {name}");
                        }
                    } else {
                        println!("Usage: /rename <name>");
                    }
                }
                SlashCommand::Resume(session_id) => {
                    if let Some(id) = session_id {
                        println!("Use 'deepseek --resume {id}' to resume a session.");
                    } else {
                        println!("Use 'deepseek --continue' or 'deepseek --resume <id>'.");
                    }
                }
                SlashCommand::Stats => {
                    let store = Store::new(cwd)?;
                    let usage = store.usage_summary(None, Some(24))?;
                    if json_mode {
                        print_json(&json!({
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "records": usage.records,
                        }))?;
                    } else {
                        println!("Last 24h usage:");
                        println!("  Input tokens:  {}", usage.input_tokens);
                        println!("  Output tokens: {}", usage.output_tokens);
                        println!("  Records: {}", usage.records);
                    }
                }
                SlashCommand::Statusline(args) => {
                    if json_mode {
                        print_json(&json!({"statusline": args}))?;
                    } else if args.is_empty() {
                        println!("Configure status line in settings: \"statusLine\" key.");
                    } else {
                        println!("Statusline: {}", args.join(" "));
                    }
                }
                SlashCommand::Theme(name) => {
                    if let Some(t) = name {
                        if json_mode {
                            print_json(&json!({"theme": t}))?;
                        } else {
                            println!("Theme set to: {t}");
                        }
                    } else {
                        println!("Available themes: default, dark, light\nUsage: /theme <name>");
                    }
                }
                SlashCommand::Usage => {
                    run_usage(
                        cwd,
                        UsageArgs {
                            session: true,
                            day: false,
                        },
                        json_mode,
                    )?;
                }
                SlashCommand::AddDir(args) => {
                    match slash_add_dirs(cwd, &mut additional_dirs, &args) {
                        Ok(message) => {
                            if json_mode {
                                print_json(&json!({
                                    "message": message,
                                    "active": additional_dirs
                                        .iter()
                                        .map(|p| p.to_string_lossy().to_string())
                                        .collect::<Vec<_>>(),
                                }))?;
                            } else {
                                println!("{message}");
                            }
                        }
                        Err(err) => {
                            if json_mode {
                                print_json(&json!({"error": err.to_string()}))?;
                            } else {
                                println!("{err}");
                            }
                        }
                    }
                }
                SlashCommand::Bug => {
                    let log_dir = deepseek_core::runtime_dir(cwd).join("logs");
                    let config_dir = deepseek_core::runtime_dir(cwd);
                    if json_mode {
                        print_json(&json!({
                            "log_dir": log_dir.to_string_lossy(),
                            "config_dir": config_dir.to_string_lossy(),
                            "report_url": "https://github.com/anthropics/deepseek-cli/issues"
                        }))?;
                    } else {
                        println!("Bug report info:");
                        println!("  Logs: {}", log_dir.display());
                        println!("  Config: {}", config_dir.display());
                        println!("  Report: https://github.com/anthropics/deepseek-cli/issues");
                    }
                }
                SlashCommand::PrComments(args) => {
                    let pr_num = args.first().cloned().unwrap_or_default();
                    let output_path = args.get(1).cloned();
                    if pr_num.is_empty() {
                        if json_mode {
                            print_json(
                                &json!({"error":"usage: /pr_comments <PR_NUMBER> [OUTPUT_JSON]"}),
                            )?;
                        } else {
                            println!("Usage: /pr_comments <PR_NUMBER> [OUTPUT_JSON]");
                        }
                    } else {
                        let payload = pr_comments_payload(cwd, &pr_num, output_path.as_deref())?;
                        if json_mode {
                            print_json(&payload)?;
                        } else {
                            if let Some(path) = payload["saved_to"].as_str() {
                                println!("PR comments saved to {path}");
                            }
                            if let Some(summary) = payload["summary"].as_str() {
                                println!("{summary}");
                            }
                        }
                    }
                }
                SlashCommand::ReleaseNotes(args) => {
                    let range = args
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "HEAD~10..HEAD".to_string());
                    let output_path = args.get(1).cloned();
                    let payload = release_notes_payload(cwd, &range, output_path.as_deref())?;
                    if json_mode {
                        print_json(&payload)?;
                    } else {
                        println!("Release notes for {range}:");
                        if let Some(lines) = payload["lines"].as_array() {
                            for line in lines.iter().filter_map(|value| value.as_str()) {
                                println!("{line}");
                            }
                        }
                        if let Some(path) = payload["saved_to"].as_str() {
                            println!("Saved to {path}");
                        }
                    }
                }
                SlashCommand::Login => {
                    let payload = login_payload(cwd)?;
                    if json_mode {
                        print_json(&payload)?;
                    } else {
                        println!(
                            "{}",
                            payload["message"].as_str().unwrap_or("login complete")
                        );
                    }
                }
                SlashCommand::Logout => {
                    let payload = logout_payload(cwd)?;
                    if json_mode {
                        print_json(&payload)?;
                    } else {
                        println!(
                            "{}",
                            payload["message"].as_str().unwrap_or("logout complete")
                        );
                    }
                }
                SlashCommand::Desktop(args) => {
                    let payload = desktop_payload(cwd, &args)?;
                    if json_mode {
                        print_json(&payload)?;
                    } else {
                        println!("{}", serde_json::to_string_pretty(&payload)?);
                    }
                }
                SlashCommand::Todos(args) => {
                    let payload = todos_payload(cwd, &args)?;
                    if json_mode {
                        print_json(&payload)?;
                    } else {
                        println!(
                            "TODO scan: {} result(s)",
                            payload["count"].as_u64().unwrap_or(0)
                        );
                        if let Some(rows) = payload["items"].as_array() {
                            for row in rows.iter().take(20) {
                                println!(
                                    "- {}:{} {}",
                                    row["path"].as_str().unwrap_or_default(),
                                    row["line"].as_u64().unwrap_or(0),
                                    row["text"].as_str().unwrap_or_default()
                                );
                            }
                        }
                    }
                }
                SlashCommand::Chrome(args) => {
                    let payload = chrome_payload(cwd, &args)?;
                    if json_mode {
                        print_json(&payload)?;
                    } else {
                        println!("{}", serde_json::to_string_pretty(&payload)?);
                    }
                }
                SlashCommand::Unknown { name, args } => {
                    // Try custom commands from .deepseek/commands/
                    let custom_cmds = deepseek_skills::load_custom_commands(cwd);
                    if let Some(cmd) = custom_cmds.iter().find(|c| c.name == name) {
                        let rendered = deepseek_skills::render_custom_command(
                            cmd,
                            &args.join(" "),
                            cwd,
                            &Uuid::now_v7().to_string(),
                        );
                        if cmd.disable_model_invocation {
                            println!("{rendered}");
                        } else {
                            // Feed rendered prompt into the agent
                            let output = engine.chat_with_options(
                                &rendered,
                                ChatOptions {
                                    tools: allow_tools && !read_only_mode,
                                    force_max_think,
                                    additional_dirs: additional_dirs.clone(),
                                    repo_root_override: repo_root_override.clone(),
                                    debug_context,
                                    mode: active_chat_mode,
                                    force_execute,
                                    force_plan_only,
                                    teammate_mode: teammate_mode.clone(),
                                    detect_urls,
                                    watch_files: watch_files_enabled,
                                    ..Default::default()
                                },
                            )?;
                            last_assistant_response = Some(output);
                        }
                    } else if json_mode {
                        print_json(&json!({"error": format!("unknown slash command: /{name}")}))?;
                    } else {
                        println!("unknown slash command: /{name}");
                    }
                }
            }
            continue;
        }

        // Set up streaming callback for real-time token output
        if json_events {
            engine.set_stream_callback(std::sync::Arc::new(|chunk: deepseek_core::StreamChunk| {
                use std::io::Write as _;
                let out = std::io::stdout();
                let mut handle = out.lock();
                
                let val = serde_json::json!({
                    "ts": chrono::Utc::now().to_rfc3339(),
                    // Event stream payload. Exclude big texts or format appropriately.
                    "type": match &chunk {
                        deepseek_core::StreamChunk::ContentDelta(_) => "ContentDelta",
                        deepseek_core::StreamChunk::ReasoningDelta(_) => "ReasoningDelta",
                        deepseek_core::StreamChunk::ArchitectStarted { .. } => "ArchitectStarted",
                        deepseek_core::StreamChunk::ArchitectCompleted { .. } => "ArchitectCompleted",
                        deepseek_core::StreamChunk::EditorStarted { .. } => "EditorStarted",
                        deepseek_core::StreamChunk::EditorCompleted { .. } => "EditorCompleted",
                        deepseek_core::StreamChunk::ApplyStarted { .. } => "ApplyStarted",
                        deepseek_core::StreamChunk::ApplyCompleted { .. } => "ApplyCompleted",
                        deepseek_core::StreamChunk::VerifyStarted { .. } => "VerifyStarted",
                        deepseek_core::StreamChunk::VerifyCompleted { .. } => "VerifyCompleted",
                        deepseek_core::StreamChunk::LintStarted { .. } => "LintStarted",
                        deepseek_core::StreamChunk::LintCompleted { .. } => "LintCompleted",
                        deepseek_core::StreamChunk::CommitProposal { .. } => "CommitProposal",
                        deepseek_core::StreamChunk::CommitCompleted { .. } => "CommitCompleted",
                        deepseek_core::StreamChunk::CommitSkipped => "CommitSkipped",
                        deepseek_core::StreamChunk::ToolCallStart { .. } => "ToolCallStart",
                        deepseek_core::StreamChunk::ToolCallEnd { .. } => "ToolCallEnd",
                        deepseek_core::StreamChunk::ModeTransition { .. } => "ModeTransition",
                        deepseek_core::StreamChunk::SubagentSpawned { .. } => "SubagentSpawned",
                        deepseek_core::StreamChunk::SubagentCompleted { .. } => "SubagentCompleted",
                        deepseek_core::StreamChunk::SubagentFailed { .. } => "SubagentFailed",
                        deepseek_core::StreamChunk::ImageData { .. } => "ImageData",
                        deepseek_core::StreamChunk::WatchTriggered { .. } => "WatchTriggered",
                        deepseek_core::StreamChunk::ClearStreamingText => "ClearStreamingText",
                        deepseek_core::StreamChunk::Done { .. } => "Done",
                    },
                    "payload": match &chunk {
                        deepseek_core::StreamChunk::ContentDelta(text) | deepseek_core::StreamChunk::ReasoningDelta(text) => {
                            serde_json::json!({ "text": text })
                        },
                        deepseek_core::StreamChunk::ArchitectStarted { iteration } => serde_json::json!({ "iteration": iteration }),
                        deepseek_core::StreamChunk::ArchitectCompleted { iteration, files, no_edit } => serde_json::json!({ "iteration": iteration, "files": files, "no_edit": no_edit }),
                        deepseek_core::StreamChunk::EditorStarted { iteration, files } => serde_json::json!({ "iteration": iteration, "files": files }),
                        deepseek_core::StreamChunk::EditorCompleted { iteration, status } => serde_json::json!({ "iteration": iteration, "status": status }),
                        deepseek_core::StreamChunk::ApplyStarted { iteration } => serde_json::json!({ "iteration": iteration }),
                        deepseek_core::StreamChunk::ApplyCompleted { iteration, success, summary } => serde_json::json!({ "iteration": iteration, "success": success, "summary": summary }),
                        deepseek_core::StreamChunk::VerifyStarted { iteration, commands } => serde_json::json!({ "iteration": iteration, "commands": commands }),
                        deepseek_core::StreamChunk::VerifyCompleted { iteration, success, summary } => serde_json::json!({ "iteration": iteration, "success": success, "summary": summary }),
                        deepseek_core::StreamChunk::LintStarted { iteration, commands } => serde_json::json!({ "iteration": iteration, "commands": commands }),
                        deepseek_core::StreamChunk::LintCompleted { iteration, success, fixed, remaining } => serde_json::json!({ "iteration": iteration, "success": success, "fixed": fixed, "remaining": remaining }),
                        deepseek_core::StreamChunk::CommitProposal { files, touched_files, loc_delta, verify_commands, verify_status, suggested_message } => serde_json::json!({ "files": files, "touched_files": touched_files, "loc_delta": loc_delta, "verify_commands": verify_commands, "verify_status": verify_status, "suggested_message": suggested_message }),
                        deepseek_core::StreamChunk::CommitCompleted { sha, message } => serde_json::json!({ "sha": sha, "message": message }),
                        deepseek_core::StreamChunk::ToolCallStart { tool_name, args_summary } => serde_json::json!({ "tool_name": tool_name, "args_summary": args_summary }),
                        deepseek_core::StreamChunk::ToolCallEnd { tool_name, duration_ms, success, summary } => serde_json::json!({ "tool_name": tool_name, "duration_ms": duration_ms, "success": success, "summary": summary }),
                        deepseek_core::StreamChunk::ModeTransition { from, to, reason } => serde_json::json!({ "from": from, "to": to, "reason": reason }),
                        deepseek_core::StreamChunk::SubagentSpawned { run_id, name, goal } => serde_json::json!({ "run_id": run_id, "name": name, "goal": goal }),
                        deepseek_core::StreamChunk::SubagentCompleted { run_id, name, summary } => serde_json::json!({ "run_id": run_id, "name": name, "summary": summary }),
                        deepseek_core::StreamChunk::SubagentFailed { run_id, name, error } => serde_json::json!({ "run_id": run_id, "name": name, "error": error }),
                        deepseek_core::StreamChunk::ImageData { label, .. } => serde_json::json!({ "label": label }),
                        deepseek_core::StreamChunk::WatchTriggered { digest, comment_count } => serde_json::json!({ "digest": digest, "comment_count": comment_count }),
                        _ => serde_json::json!({}),
                    }
                });
                
                let _ = writeln!(handle, "{}", serde_json::to_string(&val).unwrap());
                let _ = handle.flush();
            }));
        } else if !json_mode {
            let md_renderer = std::sync::Arc::new(std::sync::Mutex::new(
                crate::md_render::StreamingMdRenderer::new(),
            ));
            let md_clone = md_renderer.clone();
            engine.set_stream_callback(std::sync::Arc::new(move |chunk: deepseek_core::StreamChunk| {
                match chunk {
                    deepseek_core::StreamChunk::ContentDelta(text) => {
                        if let Ok(mut renderer) = md_clone.lock() {
                            renderer.push(&text);
                        }
                    }
                    deepseek_core::StreamChunk::ReasoningDelta(_) => {}
                    deepseek_core::StreamChunk::ArchitectStarted { iteration: _ } => {
                        crate::md_render::print_phase("◉", "Planning", "understanding context and defining plan");
                    }
                    deepseek_core::StreamChunk::ArchitectCompleted {
                        files,
                        no_edit,
                        ..
                    } => {
                        let detail = if no_edit { "no changes needed".to_string() } else { format!("{files} file(s) selected") };
                        crate::md_render::print_phase_done(true, "Plan complete", &detail);
                    }
                    deepseek_core::StreamChunk::EditorStarted { files, .. } => {
                        crate::md_render::print_phase("◉", "Editing", &format!("{files} file(s)"));
                    }
                    deepseek_core::StreamChunk::EditorCompleted { status, .. } => {
                        let success = status != "error";
                        crate::md_render::print_phase_done(success, "Edit complete", &status);
                    }
                    deepseek_core::StreamChunk::ApplyStarted { .. } => {
                        crate::md_render::print_phase("◉", "Applying", "writing changes to disk");
                    }
                    deepseek_core::StreamChunk::ApplyCompleted {
                        success,
                        ..
                    } => {
                        crate::md_render::print_phase_done(success, "Apply complete", "");
                    }
                    deepseek_core::StreamChunk::VerifyStarted {
                        commands,
                        ..
                    } => {
                        if !commands.is_empty() {
                            crate::md_render::print_phase("◉", "Verifying", &commands.join(", "));
                        }
                    }
                    deepseek_core::StreamChunk::VerifyCompleted {
                        success,
                        ..
                    } => {
                        let label = if success { "Verification passed" } else { "Verification failed" };
                        crate::md_render::print_phase_done(success, label, "");
                    }
                    deepseek_core::StreamChunk::LintStarted {
                        commands,
                        ..
                    } => {
                        if !commands.is_empty() {
                            crate::md_render::print_phase("◉", "Linting", &commands.join(", "));
                        }
                    }
                    deepseek_core::StreamChunk::LintCompleted {
                        success,
                        ..
                    } => {
                        crate::md_render::print_phase_done(success, "Lint complete", "");
                    }
                    deepseek_core::StreamChunk::CommitProposal {
                        files,
                        touched_files,
                        loc_delta,
                        suggested_message,
                        ..
                    } => {
                        use std::io::Write as _;
                        let out = std::io::stdout();
                        let mut handle = out.lock();
                        let _ = writeln!(
                            handle,
                            "\n  \x1b[33m◆\x1b[0m  \x1b[1mCommit Proposal\x1b[0m — {} file(s), +{} LoC",
                            touched_files, loc_delta
                        );
                        for f in &files {
                            let _ = writeln!(handle, "     \x1b[90m•\x1b[0m {f}");
                        }
                        let _ = writeln!(
                            handle,
                            "     \x1b[90mmessage:\x1b[0m \"{suggested_message}\""
                        );
                        let _ = handle.flush();
                    }
                    deepseek_core::StreamChunk::CommitCompleted { sha, message } => {
                        crate::md_render::print_phase_done(true, "Committed", &format!("{} — {}", &sha[..7.min(sha.len())], message));
                    }
                    deepseek_core::StreamChunk::CommitSkipped => {
                        use std::io::Write as _;
                        let out = std::io::stdout();
                        let mut handle = out.lock();
                        let _ = writeln!(handle, "  \x1b[90m◦  Commit skipped\x1b[0m");
                        let _ = handle.flush();
                    }
                    deepseek_core::StreamChunk::ToolCallStart { tool_name, args_summary } => {
                        crate::md_render::print_phase("⚡", &tool_name, &args_summary);
                    }
                    deepseek_core::StreamChunk::ToolCallEnd {
                        tool_name,
                        duration_ms,
                        success,
                        summary,
                    } => {
                        let detail = if summary.is_empty() {
                            format!("{duration_ms}ms")
                        } else {
                            format!("{duration_ms}ms — {}", summary.replace('\n', " "))
                        };
                        crate::md_render::print_phase_done(success, &tool_name, &detail);
                    }
                    deepseek_core::StreamChunk::ModeTransition { from, to, reason } => {
                        use std::io::Write as _;
                        let out = std::io::stdout();
                        let mut handle = out.lock();
                        let _ = writeln!(
                            handle,
                            "  \x1b[35m→\x1b[0m  Mode: {from} → {to} \x1b[90m({reason})\x1b[0m"
                        );
                        let _ = handle.flush();
                    }
                    deepseek_core::StreamChunk::SubagentSpawned {
                        run_id: _,
                        name,
                        goal,
                    } => {
                        crate::md_render::print_phase("◉", &format!("Subagent: {name}"), &goal);
                    }
                    deepseek_core::StreamChunk::SubagentCompleted {
                        run_id: _,
                        name,
                        summary,
                    } => {
                        crate::md_render::print_phase_done(true, &format!("Subagent: {name}"), &summary.replace('\n', " "));
                    }
                    deepseek_core::StreamChunk::SubagentFailed {
                        run_id: _,
                        name,
                        error,
                    } => {
                        crate::md_render::print_phase_done(false, &format!("Subagent: {name}"), &error.replace('\n', " "));
                    }
                    deepseek_core::StreamChunk::ImageData { label, .. } => {
                        use std::io::Write as _;
                        let out = std::io::stdout();
                        let mut handle = out.lock();
                        let _ = writeln!(handle, "  \x1b[90m[image: {label}]\x1b[0m");
                        let _ = handle.flush();
                    }
                    deepseek_core::StreamChunk::WatchTriggered { comment_count, .. } => {
                        use std::io::Write as _;
                        let out = std::io::stdout();
                        let mut handle = out.lock();
                        let _ = writeln!(handle, "  \x1b[35m◉\x1b[0m  \x1b[1mWatch triggered\x1b[0m \x1b[90m({comment_count} comment(s))\x1b[0m");
                        let _ = handle.flush();
                    }
                    deepseek_core::StreamChunk::ClearStreamingText => {}
                    deepseek_core::StreamChunk::Done { .. } => {
                        // Flush any remaining partial line from the renderer
                        if let Ok(mut renderer) = md_clone.lock() {
                            renderer.flush_remaining();
                        }
                    }
                }
            }));
        }

        let images_for_turn = std::mem::take(&mut pending_images);
        if !json_mode && !json_events {
            crate::md_render::print_role_header("assistant", &cfg.llm.base_model);
        }
        let output = engine.chat_with_options(
            prompt,
            ChatOptions {
                tools: allow_tools && !read_only_mode,
                force_max_think,
                additional_dirs: additional_dirs.clone(),
                repo_root_override: repo_root_override.clone(),
                debug_context,
                mode: active_chat_mode,
                force_execute,
                force_plan_only,
                teammate_mode: teammate_mode.clone(),
                detect_urls,
                watch_files: watch_files_enabled,
                images: images_for_turn,
                ..Default::default()
            },
        )?;
        last_assistant_response = Some(output.clone());
        if !json_mode && !json_events {
            crate::md_render::print_role_footer();
        }
        let ui_status = current_ui_status(cwd, &cfg, force_max_think)?;
        if json_mode {
            let suggestions = generate_prompt_suggestions(&output);
            print_json(&json!({"output": output, "statusline": render_statusline(&ui_status), "suggestions": suggestions}))?;
        } else {
            println!("[status] {}", render_statusline(&ui_status));
            // Show follow-up prompt suggestions
            let suggestions = generate_prompt_suggestions(&output);
            if !suggestions.is_empty() {
                println!("\x1b[2m  suggestions: {}\x1b[0m", suggestions.join(" | "));
            }
        }

        // Watch auto-execute: if digest changed after agent turn, auto-dispatch
        if watch_files_enabled {
            let mut auto_watch_turns: usize = 0;
            const MAX_WATCH_AUTO_TURNS: usize = 3;
            while auto_watch_turns < MAX_WATCH_AUTO_TURNS {
                if let Some((digest, hints)) = scan_watch_comment_payload(cwd) {
                    if last_watch_digest == Some(digest) {
                        break; // no change
                    }
                    last_watch_digest = Some(digest);
                    auto_watch_turns += 1;
                    let comment_count = hints.lines().count();
                    if !json_mode {
                        println!("[watch: auto-trigger {auto_watch_turns}/{MAX_WATCH_AUTO_TURNS} — {comment_count} comment(s)]");
                    }
                    let mut auto_prompt =
                        "Resolve the following TODO/FIXME/AI comments detected in the workspace:".to_string();
                    auto_prompt.push_str("\n\nAUTO_WATCH_CONTEXT_V1\n");
                    auto_prompt.push_str(&hints);
                    auto_prompt.push_str("\nAUTO_WATCH_CONTEXT_END");

                    let auto_output = engine.chat_with_options(
                        &auto_prompt,
                        ChatOptions {
                            tools: allow_tools && !read_only_mode,
                            force_max_think,
                            additional_dirs: additional_dirs.clone(),
                            repo_root_override: repo_root_override.clone(),
                            debug_context,
                            mode: active_chat_mode,
                            force_execute,
                            force_plan_only,
                            teammate_mode: teammate_mode.clone(),
                            detect_urls,
                            watch_files: true,
                            ..Default::default()
                        },
                    )?;
                    last_assistant_response = Some(auto_output);
                } else {
                    break;
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_chat_tui(
    cwd: &Path,
    allow_tools: bool,
    cfg: &AppConfig,
    initial_force_max_think: bool,
    force_execute: bool,
    force_plan_only: bool,
    teammate_mode: Option<String>,
    repo_root_override: Option<PathBuf>,
    debug_context: bool,
    detect_urls: bool,
    watch_files_enabled: bool,
) -> Result<()> {
    let engine = Arc::new(AgentEngine::new(cwd)?);
    wire_subagent_worker(&engine, cwd);
    let force_max_think = Arc::new(AtomicBool::new(initial_force_max_think));
    let additional_dirs = Arc::new(std::sync::Mutex::new(Vec::<PathBuf>::new()));
    let read_only_mode = Arc::new(AtomicBool::new(false));
    let active_chat_mode = Arc::new(std::sync::Mutex::new(ChatMode::Code));
    let last_watch_digest = Arc::new(std::sync::Mutex::new(None::<u64>));
    let pending_images = Arc::new(std::sync::Mutex::new(Vec::<deepseek_core::ImageContent>::new()));

    // Create the channel for TUI stream events.
    let (tx, rx) = mpsc::channel::<TuiStreamEvent>();

    // Set approval handler that routes through the TUI channel.
    {
        let approval_tx = tx.clone();
        engine.set_approval_handler(Box::new(move |call| {
            let (resp_tx, resp_rx) = mpsc::channel();
            let compact_args = serde_json::to_string(&call.args)
                .unwrap_or_else(|_| "<unserializable>".to_string());
            let _ = approval_tx.send(TuiStreamEvent::ApprovalNeeded {
                tool_name: call.name.clone(),
                args_summary: compact_args,
                response_tx: resp_tx,
            });
            // Block agent thread waiting for TUI user response.
            resp_rx
                .recv()
                .map_err(|e| anyhow!("approval channel closed: {e}"))
        }));
    }

    let status = current_ui_status(cwd, cfg, force_max_think.load(Ordering::Relaxed))?;
    let bindings = load_tui_keybindings(cwd, cfg);
    let theme = TuiTheme::from_config(&cfg.theme.primary, &cfg.theme.secondary, &cfg.theme.error);
    let fmt_refresh = Arc::clone(&force_max_think);
    let additional_dirs_for_closure = Arc::clone(&additional_dirs);
    let read_only_for_closure = Arc::clone(&read_only_mode);
    let active_mode_for_closure = Arc::clone(&active_chat_mode);
    let watch_digest_for_closure = Arc::clone(&last_watch_digest);
    run_tui_shell_with_bindings(
        status,
        bindings,
        theme,
        cfg.ui.reduced_motion,
        rx,
        |prompt| {
            // Handle slash commands synchronously, sending result via channel.
            if let Some(cmd) = SlashCommand::parse(prompt) {
                let result: Result<String> = (|| {
                    let out = match cmd {
                SlashCommand::Help => "commands: /help /ask /code /architect /chat-mode /init /clear /compact /memory /config /model /cost /mcp /rewind /export /plan /teleport /remote-env /status /add /drop /read-only /map /map-refresh /run /test /lint /web /diff /stage /unstage /commit /undo /effort /skills /permissions /background /visual /git /settings /load /save /paste /voice /desktop /todos /chrome /vim".to_string(),
                SlashCommand::Ask(_) => {
                    if let Ok(mut guard) = active_mode_for_closure.lock() {
                        *guard = ChatMode::Ask;
                    }
                    "chat mode set to ask".to_string()
                }
                SlashCommand::Code(_) => {
                    if let Ok(mut guard) = active_mode_for_closure.lock() {
                        *guard = ChatMode::Code;
                    }
                    "chat mode set to code".to_string()
                }
                SlashCommand::Architect(_) => {
                    if let Ok(mut guard) = active_mode_for_closure.lock() {
                        *guard = ChatMode::Architect;
                    }
                    "chat mode set to architect".to_string()
                }
                SlashCommand::ChatMode(mode) => {
                    if let Some(raw_mode) = mode {
                        if let Some(parsed) = parse_chat_mode_name(&raw_mode) {
                            if let Ok(mut guard) = active_mode_for_closure.lock() {
                                *guard = parsed;
                            }
                            format!("chat mode set to {}", chat_mode_name(parsed))
                        } else {
                            format!("unsupported chat mode: {raw_mode} (ask|code|architect|context)")
                        }
                    } else if let Ok(guard) = active_mode_for_closure.lock() {
                        format!("current chat mode: {}", chat_mode_name(*guard))
                    } else {
                        "current chat mode: code".to_string()
                    }
                }
                SlashCommand::Init => {
                    let manager = MemoryManager::new(cwd)?;
                    let path = manager.ensure_initialized()?;
                    format!("initialized memory at {}", path.display())
                }
                SlashCommand::Clear => "cleared".to_string(),
                SlashCommand::Compact(focus) => {
                    let summary = compact_now(cwd, None, focus.as_deref())?;
                    format!(
                        "compacted turns {}..{} summary_id={} token_delta={}",
                        summary.from_turn,
                        summary.to_turn,
                        summary.summary_id,
                        summary.token_delta_estimate
                    )
                }
                SlashCommand::Memory(args) => {
                    if args.is_empty() || args[0].eq_ignore_ascii_case("show") {
                        MemoryManager::new(cwd)?.read_memory()?
                    } else if args[0].eq_ignore_ascii_case("edit") {
                        let manager = MemoryManager::new(cwd)?;
                        let path = manager.ensure_initialized()?;
                        let checkpoint = manager.create_checkpoint("memory_edit")?;
                        append_control_event(
                            cwd,
                            EventKind::CheckpointCreatedV1 {
                                checkpoint_id: checkpoint.checkpoint_id,
                                reason: checkpoint.reason.clone(),
                                files_count: checkpoint.files_count,
                                snapshot_path: checkpoint.snapshot_path.clone(),
                            },
                        )?;
                        let editor =
                            std::env::var("EDITOR").unwrap_or_else(|_| default_editor().to_string());
                        let status = std::process::Command::new(editor).arg(&path).status()?;
                        if !status.success() {
                            return Err(anyhow!("editor exited with non-zero status"));
                        }
                        let version_id = manager.sync_memory_version("edit")?;
                        append_control_event(
                            cwd,
                            EventKind::MemorySyncedV1 {
                                version_id,
                                path: path.to_string_lossy().to_string(),
                                note: "edit".to_string(),
                            },
                        )?;
                        format!("memory edited at {}", path.display())
                    } else if args[0].eq_ignore_ascii_case("sync") {
                        let note = args
                            .get(1)
                            .cloned()
                            .unwrap_or_else(|| "tui-sync".to_string());
                        let manager = MemoryManager::new(cwd)?;
                        let version_id = manager.sync_memory_version(&note)?;
                        append_control_event(
                            cwd,
                            EventKind::MemorySyncedV1 {
                                version_id,
                                path: manager.memory_path().to_string_lossy().to_string(),
                                note,
                            },
                        )?;
                        format!("memory synced: {version_id}")
                    } else {
                        "unknown /memory subcommand".to_string()
                    }
                }
                SlashCommand::Config => format!(
                    "config file: {}",
                    AppConfig::project_settings_path(cwd).display()
                ),
                SlashCommand::Model(model) => {
                    if let Some(model) = model {
                        force_max_think.store(is_max_think_selection(&model), Ordering::Relaxed);
                    }
                    if force_max_think.load(Ordering::Relaxed) {
                        format!(
                            "model mode: thinking-enabled ({})",
                            cfg.llm.base_model
                        )
                    } else {
                        format!(
                            "model mode: auto ({} thinking=on-demand)",
                            cfg.llm.base_model
                        )
                    }
                }
                SlashCommand::Cost => {
                    let store = Store::new(cwd)?;
                    let usage = store.usage_summary(None, Some(24))?;
                    format!(
                        "24h usage input={} output={}",
                        usage.input_tokens, usage.output_tokens
                    )
                }
                SlashCommand::Tokens => slash_tokens_output(cwd, cfg)?,
                SlashCommand::Mcp(args) => {
                    let manager = McpManager::new(cwd)?;
                    if args.is_empty() || args[0].eq_ignore_ascii_case("list") {
                        let servers = manager.list_servers()?;
                        format!("mcp servers: {}", servers.len())
                    } else if args[0].eq_ignore_ascii_case("get") && args.len() > 1 {
                        match manager.get_server(&args[1])? {
                            Some(server) => format!(
                                "mcp {} transport={:?} enabled={}",
                                server.id, server.transport, server.enabled
                            ),
                            None => format!("mcp server not found: {}", args[1]),
                        }
                    } else if args[0].eq_ignore_ascii_case("remove") && args.len() > 1 {
                        let removed = manager.remove_server(&args[1])?;
                        format!("mcp remove {} -> {}", args[1], removed)
                    } else {
                        "use /mcp list|get <id>|remove <id>".to_string()
                    }
                }
                SlashCommand::Rewind(args) => {
                    if args.is_empty() {
                        // Show rewind picker (checkpoint list + action menu).
                        let mem = MemoryManager::new(cwd)?;
                        let checkpoints = mem.list_checkpoints().unwrap_or_default();
                        if checkpoints.is_empty() {
                            "no checkpoints available".to_string()
                        } else {
                            let picker = deepseek_ui::RewindPickerState::new(checkpoints);
                            // Display as numbered list for user to choose
                            let mut lines: Vec<String> = vec!["Rewind checkpoints:".to_string()];
                            for (i, cp) in picker.checkpoints.iter().enumerate() {
                                lines.push(format!(
                                    "  {}. [{}] {} ({} files)",
                                    i + 1, cp.created_at, cp.reason, cp.files_count
                                ));
                            }
                            lines.push(String::new());
                            lines.push("Use /rewind <number> to rewind to a checkpoint.".to_string());
                            lines.join("\n")
                        }
                    } else {
                        let to_checkpoint = args.first().cloned();
                        let checkpoint = rewind_now(cwd, to_checkpoint)?;
                        format!("rewound to checkpoint {}", checkpoint.checkpoint_id)
                    }
                }
                SlashCommand::Export(_) => {
                    let record = MemoryManager::new(cwd)?.export_transcript(
                        ExportFormat::Json,
                        None,
                        None,
                    )?;
                    format!("exported transcript {}", record.output_path)
                }
                SlashCommand::Plan => {
                    force_max_think.store(true, Ordering::Relaxed);
                    "plan mode enabled (thinking enabled)".to_string()
                }
                SlashCommand::Teleport(args) => {
                    match parse_teleport_args(args) {
                        Ok(teleport_args) => {
                            let teleport = teleport_now(cwd, teleport_args)?;
                            match teleport.mode.as_str() {
                                "import" => format!(
                                    "imported teleport bundle {}",
                                    teleport.imported.unwrap_or_default()
                                ),
                                "link" => format!(
                                    "handoff link {}",
                                    teleport.link_url.unwrap_or_default()
                                ),
                                "consume" => format!(
                                    "consumed handoff {}",
                                    teleport.handoff_id.unwrap_or_default()
                                ),
                                _ => format!(
                                    "teleport bundle {} -> {}",
                                    teleport.bundle_id.unwrap_or_default(),
                                    teleport.path.unwrap_or_default()
                                ),
                            }
                        }
                        Err(err) => format!("teleport parse error: {err}"),
                    }
                }
                SlashCommand::RemoteEnv(args) => {
                    match parse_remote_env_cmd(args) {
                        Ok(cmd) => {
                            let out = remote_env_now(cwd, cmd)?;
                            serde_json::to_string(&out)?
                        }
                        Err(err) => format!("remote-env parse error: {err}"),
                    }
                }
                SlashCommand::Add(args) => {
                    let mut guard = additional_dirs_for_closure
                        .lock()
                        .map_err(|_| anyhow!("failed to access additional dir state"))?;
                    slash_add_dirs(cwd, &mut guard, &args)?
                }
                SlashCommand::Drop(args) => {
                    let mut guard = additional_dirs_for_closure
                        .lock()
                        .map_err(|_| anyhow!("failed to access additional dir state"))?;
                    slash_drop_dirs(cwd, &mut guard, &args)?
                }
                SlashCommand::ReadOnly(args) => {
                    let action = args
                        .first()
                        .map(|v| v.to_ascii_lowercase())
                        .unwrap_or_else(|| "toggle".to_string());
                    match action.as_str() {
                        "on" | "true" | "1" => {
                            read_only_for_closure.store(true, Ordering::Relaxed);
                        }
                        "off" | "false" | "0" => {
                            read_only_for_closure.store(false, Ordering::Relaxed);
                        }
                        "status" => {}
                        _ => {
                            let next = !read_only_for_closure.load(Ordering::Relaxed);
                            read_only_for_closure.store(next, Ordering::Relaxed);
                        }
                    }
                    format!(
                        "read-only mode: {}",
                        if read_only_for_closure.load(Ordering::Relaxed) {
                            "enabled"
                        } else {
                            "disabled"
                        }
                    )
                }
                SlashCommand::Map(args) => {
                    let guard = additional_dirs_for_closure
                        .lock()
                        .map_err(|_| anyhow!("failed to access additional dir state"))?;
                    slash_map_output(cwd, &args, &guard)?
                }
                SlashCommand::MapRefresh(args) => {
                    deepseek_agent::clear_tag_cache();
                    let guard = additional_dirs_for_closure
                        .lock()
                        .map_err(|_| anyhow!("failed to access additional dir state"))?;
                    format!("(cache cleared)\n{}", slash_map_output(cwd, &args, &guard)?)
                }
                SlashCommand::Run(args) => slash_run_output(cwd, &args)?,
                SlashCommand::Test(args) => slash_test_output(cwd, &args)?,
                SlashCommand::Lint(args) => slash_lint_output(cwd, &args, Some(cfg))?,
                SlashCommand::Web(args) => slash_web_output(cwd, &args)?,
                SlashCommand::Diff(args) => slash_diff_output(cwd, &args)?,
                SlashCommand::Stage(args) => slash_stage_output(cwd, &args)?,
                SlashCommand::Unstage(args) => slash_unstage_output(cwd, &args)?,
                SlashCommand::Commit(args) => {
                    if parse_commit_message(&args)?.is_none() {
                        "commit requires a message in TUI mode: /commit -m \"your message\"".to_string()
                    } else {
                        slash_commit_output(cwd, cfg, &args)?
                    }
                }
                SlashCommand::Undo(_args) => {
                    let checkpoint_msg = match crate::commands::compact::rewind_now(cwd, None) {
                        Ok(checkpoint) => format!(
                            "rewound to checkpoint {} (files={})",
                            checkpoint.checkpoint_id, checkpoint.files_count
                        ),
                        Err(e) => format!("no files rewound ({e})"),
                    };
                    crate::context::append_control_event(
                        cwd,
                        deepseek_core::EventKind::TurnRevertedV1 { turns_dropped: 1 },
                    )?;
                    format!("Reverted 1 conversation turn. {checkpoint_msg}")
                }
                SlashCommand::Status => {
                    let status = current_ui_status(cwd, cfg, force_max_think.load(Ordering::Relaxed))?;
                    render_statusline(&status)
                }
                SlashCommand::Effort(level) => {
                    let level = level.unwrap_or_else(|| "medium".to_string());
                    let normalized = level.to_ascii_lowercase();
                    force_max_think.store(
                        matches!(normalized.as_str(), "high" | "max"),
                        Ordering::Relaxed,
                    );
                    format!(
                        "effort={} thinking={}",
                        normalized,
                        if force_max_think.load(Ordering::Relaxed) {
                            "enabled"
                        } else {
                            "auto"
                        }
                    )
                }
                SlashCommand::Skills(args) => {
                    let manager = SkillManager::new(cwd)?;
                    let paths = cfg
                        .skills
                        .paths
                        .iter()
                        .map(|path| expand_tilde(path))
                        .collect::<Vec<_>>();
                    if args.is_empty() || args[0].eq_ignore_ascii_case("list") {
                        let skills = manager.list(&paths)?;
                        if skills.is_empty() {
                            "no skills found".to_string()
                        } else {
                            skills
                                .into_iter()
                                .map(|skill| format!("{} - {}", skill.id, skill.summary))
                                .collect::<Vec<_>>()
                                .join("\n")
                        }
                    } else if args[0].eq_ignore_ascii_case("reload") {
                        let loaded = manager.reload(&paths)?;
                        format!("reloaded {} skills", loaded.len())
                    } else if args[0].eq_ignore_ascii_case("run") && args.len() > 1 {
                        let input = if args.len() > 2 {
                            Some(args[2..].join(" "))
                        } else {
                            None
                        };
                        let rendered = manager.run(&args[1], input.as_deref(), &paths)?;
                        format!("{}\n{}", rendered.skill_id, rendered.rendered_prompt)
                    } else {
                        "use /skills list|reload|run <id> [input]".to_string()
                    }
                }
                SlashCommand::Permissions(args) => match parse_permissions_cmd(args) {
                    Ok(cmd) => serde_json::to_string_pretty(&permissions_payload(cwd, cmd)?)?,
                    Err(err) => format!("permissions parse error: {err}"),
                },
                SlashCommand::Background(args) => match parse_background_cmd(args) {
                    Ok(cmd) => serde_json::to_string_pretty(&background_payload(cwd, cmd)?)?,
                    Err(err) => format!("background parse error: {err}"),
                },
                SlashCommand::Visual(args) => match parse_visual_cmd(args) {
                    Ok(cmd) => serde_json::to_string_pretty(&visual_payload(cwd, cmd)?)?,
                    Err(err) => format!("visual parse error: {err}"),
                },
                SlashCommand::Git(args) => slash_git_output(cwd, &args)?,
                SlashCommand::Settings => format!(
                    "config file: {}",
                    AppConfig::project_settings_path(cwd).display()
                ),
                SlashCommand::Load(args) => {
                    let (mode, read_only, thinking, dirs, output) =
                        slash_load_profile_output(cwd, &args)?;
                    if let Ok(mut guard) = active_mode_for_closure.lock() {
                        *guard = mode;
                    }
                    read_only_for_closure.store(read_only, Ordering::Relaxed);
                    force_max_think.store(thinking, Ordering::Relaxed);
                    if let Ok(mut guard) = additional_dirs_for_closure.lock() {
                        *guard = dirs;
                    }
                    output
                }
                SlashCommand::Save(args) => {
                    let mode = active_mode_for_closure
                        .lock()
                        .map(|g| *g)
                        .unwrap_or(ChatMode::Code);
                    let dirs = additional_dirs_for_closure
                        .lock()
                        .map(|d| d.clone())
                        .unwrap_or_default();
                    slash_save_profile_output(
                        cwd,
                        &args,
                        mode,
                        read_only_for_closure.load(Ordering::Relaxed),
                        force_max_think.load(Ordering::Relaxed),
                        &dirs,
                    )?
                }
                SlashCommand::Context => {
                    let ctx_cfg = AppConfig::load(cwd).unwrap_or_default();
                    let ctx_store = Store::new(cwd)?;
                    let context_window = ctx_cfg.llm.context_window_tokens;
                    let compact_threshold = ctx_cfg.context.auto_compact_threshold;
                    let session = ctx_store.load_latest_session()?;
                    let (session_tokens, compactions) = if let Some(ref s) = session {
                        let usage = ctx_store.usage_summary(Some(s.session_id), None)?;
                        let compactions = ctx_store.list_context_compactions(Some(s.session_id))?;
                        (usage.input_tokens + usage.output_tokens, compactions.len())
                    } else {
                        (0, 0)
                    };
                    let memory_tokens = {
                        let mem = deepseek_memory::MemoryManager::new(cwd).ok();
                        let text = mem.and_then(|m| m.read_combined_memory().ok()).unwrap_or_default();
                        (text.len() as u64) / 4
                    };
                    let system_prompt_tokens: u64 = 800 + ctx_cfg.policy.allowlist.len() as u64 * 40 + 400;
                    let conversation_tokens = session_tokens.saturating_sub(system_prompt_tokens + memory_tokens);
                    let utilization = if context_window > 0 {
                        (session_tokens as f64 / context_window as f64) * 100.0
                    } else { 0.0 };
                    let mut out = format!(
                        "Context Window Inspector\n========================\nWindow size:       {} tokens\nCompact threshold: {:.0}%\nSession tokens:    {}\nUtilization:       {:.1}%\nCompactions:       {}\n\nBreakdown:\n  System prompt:        ~{} tokens\n  Memory (DEEPSEEK.md): ~{} tokens\n  Conversation:         ~{} tokens",
                        context_window, compact_threshold * 100.0, session_tokens,
                        utilization, compactions, system_prompt_tokens, memory_tokens, conversation_tokens,
                    );
                    if utilization > (compact_threshold as f64 * 100.0) {
                        out.push_str("\n\nContext is above compact threshold. Use /compact to free space.");
                    }
                    out
                }
                SlashCommand::Sandbox(_) => format!("Sandbox mode: {}", AppConfig::load(cwd).unwrap_or_default().policy.sandbox_mode),
                SlashCommand::Agents => {
                    let payload = agents_payload(cwd, 20)?;
                    render_agents_payload(&payload)
                }
                SlashCommand::Tasks(_) => {
                    let payload = mission_control_payload(cwd, 20)?;
                    render_mission_control_payload(&payload)
                }
                SlashCommand::Review(_) => "Use 'deepseek review' subcommand for code review.".to_string(),
                SlashCommand::Search(args) => {
                    let query = args.join(" ");
                    if query.is_empty() {
                        "Usage: /search <query>".to_string()
                    } else {
                        format!("Search '{}': use 'deepseek search' subcommand.", query)
                    }
                }
                SlashCommand::Vim(args) => {
                    if args.is_empty() {
                        "vim mode toggled in the TUI input layer".to_string()
                    } else {
                        format!("vim mode command received: {}", args.join(" "))
                    }
                }
                SlashCommand::TerminalSetup => "Use /terminal-setup in interactive mode.".to_string(),
                SlashCommand::Keybindings => {
                    let path = AppConfig::keybindings_path().unwrap_or_default();
                    format!("Keybindings: {}", path.display())
                }
                SlashCommand::Doctor => serde_json::to_string_pretty(&doctor_payload(cwd, &DoctorArgs::default())?)?,
                SlashCommand::Copy => "Copied last response to clipboard.".to_string(),
                SlashCommand::Paste => {
                    if let Some(img_bytes) = read_image_from_clipboard() {
                        use base64::Engine;
                        let b64 = base64::engine::general_purpose::STANDARD.encode(&img_bytes);
                        if let Ok(mut imgs) = pending_images.lock() {
                            imgs.push(deepseek_core::ImageContent {
                                mime: "image/png".to_string(),
                                base64_data: b64,
                            });
                        }
                        format!("Image pasted ({} bytes). It will be included in your next prompt.", img_bytes.len())
                    } else {
                        let pasted = read_from_clipboard().unwrap_or_default();
                        if pasted.is_empty() {
                            "clipboard is empty or unavailable".to_string()
                        } else {
                            pasted
                        }
                    }
                }
                SlashCommand::Voice(args) => slash_voice_output(&args)?,
                SlashCommand::Debug(args) => match parse_debug_analysis_args(&args)? {
                    Some(doctor_args) => {
                        serde_json::to_string_pretty(&doctor_payload(cwd, &doctor_args)?)?
                    }
                    None => format!(
                        "Debug: {}",
                        if args.is_empty() {
                            "general".to_string()
                        } else {
                            args.join(" ")
                        }
                    ),
                },
                SlashCommand::Exit => "Exiting...".to_string(),
                SlashCommand::Hooks(_) => {
                    let hooks = &cfg.hooks;
                    serde_json::to_string_pretty(hooks).unwrap_or_else(|_| "no hooks configured".to_string())
                }
                SlashCommand::Rename(name) => {
                    if let Some(n) = name { format!("Session renamed to: {n}") } else { "Usage: /rename <name>".to_string() }
                }
                SlashCommand::Resume(id) => {
                    if let Some(id) = id { format!("Use 'deepseek --resume {id}' to resume.") } else { "Usage: /resume <session-id>".to_string() }
                }
                SlashCommand::Stats => {
                    let store = Store::new(cwd)?;
                    let usage = store.usage_summary(None, Some(24))?;
                    format!("24h: input={} output={} records={}", usage.input_tokens, usage.output_tokens, usage.records)
                }
                SlashCommand::Statusline(_) => "Configure status line in settings.json".to_string(),
                SlashCommand::Theme(t) => {
                    if let Some(t) = t { format!("Theme: {t}") } else { "Available: default, dark, light".to_string() }
                }
                SlashCommand::Usage => {
                    let store = Store::new(cwd)?;
                    let usage = store.usage_summary(None, None)?;
                    format!("Usage: input={} output={}", usage.input_tokens, usage.output_tokens)
                }
                SlashCommand::AddDir(args) => {
                    let mut guard = additional_dirs_for_closure
                        .lock()
                        .map_err(|_| anyhow!("failed to access additional dir state"))?;
                    slash_add_dirs(cwd, &mut guard, &args)?
                }
                SlashCommand::Bug => format!("Report bugs at https://github.com/anthropics/deepseek-cli/issues\nLogs: {}", deepseek_core::runtime_dir(cwd).join("logs").display()),
                SlashCommand::PrComments(args) => {
                    if let Some(pr) = args.first() {
                        let payload = pr_comments_payload(cwd, pr, args.get(1).map(|s| s.as_str()))?;
                        serde_json::to_string_pretty(&payload)?
                    } else {
                        "Usage: /pr_comments <number> [output.json]".to_string()
                    }
                }
                SlashCommand::ReleaseNotes(args) => {
                    let range = args
                        .first()
                        .map(|s| s.as_str())
                        .unwrap_or("HEAD~10..HEAD");
                    let payload =
                        release_notes_payload(cwd, range, args.get(1).map(|s| s.as_str()))?;
                    serde_json::to_string_pretty(&payload)?
                }
                SlashCommand::Login => {
                    let payload = login_payload(cwd)?;
                    serde_json::to_string_pretty(&payload)?
                }
                SlashCommand::Logout => {
                    let payload = logout_payload(cwd)?;
                    serde_json::to_string_pretty(&payload)?
                }
                SlashCommand::Desktop(args) => {
                    let payload = desktop_payload(cwd, &args)?;
                    serde_json::to_string_pretty(&payload)?
                }
                SlashCommand::Todos(args) => {
                    let payload = todos_payload(cwd, &args)?;
                    serde_json::to_string_pretty(&payload)?
                }
                SlashCommand::Chrome(args) => {
                    let payload = chrome_payload(cwd, &args)?;
                    serde_json::to_string_pretty(&payload)?
                }
                SlashCommand::Unknown { name, args } => {
                    let custom_cmds = deepseek_skills::load_custom_commands(cwd);
                    if let Some(cmd) = custom_cmds.iter().find(|c| c.name == name) {
                        deepseek_skills::render_custom_command(
                            cmd, &args.join(" "), cwd, &uuid::Uuid::now_v7().to_string(),
                        )
                    } else {
                        format!("unknown slash command: /{name}")
                    }
                }
                    };
                    Ok(out)
                })();
                match result {
                    Ok(output) => {
                        let _ = tx.send(TuiStreamEvent::Done(output));
                    }
                    Err(e) => {
                        let _ = tx.send(TuiStreamEvent::Error(e.to_string()));
                    }
                }
                return;
            }

            // Agent prompt — expand @file mentions and set stream callback.
            let engine_clone = Arc::clone(&engine);
            let mut prompt = deepseek_ui::expand_at_mentions(prompt);
            if watch_files_enabled
                && let Some((digest, hints)) = scan_watch_comment_payload(cwd)
                && let Ok(mut guard) = watch_digest_for_closure.lock()
                && *guard != Some(digest)
            {
                *guard = Some(digest);
                prompt
                    .push_str("\n\nAUTO_WATCH_CONTEXT_V1\nDetected comment hints in workspace:\n");
                prompt.push_str(&hints);
                prompt.push_str("\nAUTO_WATCH_CONTEXT_END");
            }
            let max_think = force_max_think.load(Ordering::Relaxed);
            let tx_stream = tx.clone();
            let tx_done = tx.clone();
            let read_only_for_turn = read_only_for_closure.load(Ordering::Relaxed);
            let mode_for_turn = active_mode_for_closure
                .lock()
                .map(|mode| *mode)
                .unwrap_or(ChatMode::Code);
            let prompt_additional_dirs = additional_dirs_for_closure
                .lock()
                .map(|dirs| dirs.clone())
                .unwrap_or_default();
            let images_for_turn = pending_images
                .lock()
                .map(|mut imgs| std::mem::take(&mut *imgs))
                .unwrap_or_default();
            let teammate_mode_for_turn = teammate_mode.clone();

            engine.set_stream_callback(std::sync::Arc::new(move |chunk| match chunk {
                StreamChunk::ContentDelta(s) => {
                    let _ = tx_stream.send(TuiStreamEvent::ContentDelta(s));
                }
                StreamChunk::ReasoningDelta(s) => {
                    let _ = tx_stream.send(TuiStreamEvent::ReasoningDelta(s));
                }
                StreamChunk::ArchitectStarted { iteration } => {
                    let _ = tx_stream.send(TuiStreamEvent::ArchitectStarted { iteration });
                }
                StreamChunk::ArchitectCompleted {
                    iteration,
                    files,
                    no_edit,
                } => {
                    let _ = tx_stream.send(TuiStreamEvent::ArchitectCompleted {
                        iteration,
                        files,
                        no_edit,
                    });
                }
                StreamChunk::EditorStarted { iteration, files } => {
                    let _ = tx_stream.send(TuiStreamEvent::EditorStarted { iteration, files });
                }
                StreamChunk::EditorCompleted { iteration, status } => {
                    let _ = tx_stream.send(TuiStreamEvent::EditorCompleted { iteration, status });
                }
                StreamChunk::ApplyStarted { iteration } => {
                    let _ = tx_stream.send(TuiStreamEvent::ApplyStarted { iteration });
                }
                StreamChunk::ApplyCompleted {
                    iteration,
                    success,
                    summary,
                } => {
                    let _ = tx_stream.send(TuiStreamEvent::ApplyCompleted {
                        iteration,
                        success,
                        summary,
                    });
                }
                StreamChunk::VerifyStarted {
                    iteration,
                    commands,
                } => {
                    let _ = tx_stream.send(TuiStreamEvent::VerifyStarted {
                        iteration,
                        commands,
                    });
                }
                StreamChunk::VerifyCompleted {
                    iteration,
                    success,
                    summary,
                } => {
                    let _ = tx_stream.send(TuiStreamEvent::VerifyCompleted {
                        iteration,
                        success,
                        summary,
                    });
                }
                StreamChunk::LintStarted {
                    iteration,
                    commands,
                } => {
                    let _ = tx_stream.send(TuiStreamEvent::LintStarted {
                        iteration,
                        commands,
                    });
                }
                StreamChunk::LintCompleted {
                    iteration,
                    success,
                    fixed,
                    remaining,
                } => {
                    let _ = tx_stream.send(TuiStreamEvent::LintCompleted {
                        iteration,
                        success,
                        fixed,
                        remaining,
                    });
                }
                StreamChunk::CommitProposal {
                    files,
                    touched_files,
                    loc_delta,
                    verify_commands,
                    verify_status,
                    suggested_message,
                } => {
                    let _ = tx_stream.send(TuiStreamEvent::CommitProposal {
                        files,
                        touched_files,
                        loc_delta,
                        verify_commands,
                        verify_status,
                        suggested_message,
                    });
                }
                StreamChunk::CommitCompleted { sha, message } => {
                    let _ = tx_stream.send(TuiStreamEvent::CommitCompleted { sha, message });
                }
                StreamChunk::CommitSkipped => {
                    let _ = tx_stream.send(TuiStreamEvent::CommitSkipped);
                }
                StreamChunk::ToolCallStart {
                    tool_name,
                    args_summary,
                } => {
                    let _ = tx_stream.send(TuiStreamEvent::ToolCallStart {
                        tool_name,
                        args_summary,
                    });
                }
                StreamChunk::ToolCallEnd {
                    tool_name,
                    duration_ms,
                    success,
                    summary,
                } => {
                    let _ = tx_stream.send(TuiStreamEvent::ToolCallEnd {
                        tool_name,
                        duration_ms,
                        summary,
                        success,
                    });
                }
                StreamChunk::ModeTransition { from, to, reason } => {
                    let _ = tx_stream.send(TuiStreamEvent::ModeTransition { from, to, reason });
                }
                StreamChunk::SubagentSpawned { run_id, name, goal } => {
                    let _ = tx_stream.send(TuiStreamEvent::SubagentSpawned { run_id, name, goal });
                }
                StreamChunk::SubagentCompleted {
                    run_id,
                    name,
                    summary,
                } => {
                    let _ = tx_stream.send(TuiStreamEvent::SubagentCompleted {
                        run_id,
                        name,
                        summary,
                    });
                }
                StreamChunk::SubagentFailed {
                    run_id,
                    name,
                    error,
                } => {
                    let _ = tx_stream.send(TuiStreamEvent::SubagentFailed {
                        run_id,
                        name,
                        error,
                    });
                }
                StreamChunk::ImageData { data, label } => {
                    let _ = tx_stream.send(TuiStreamEvent::ImageDisplay { data, label });
                }
                StreamChunk::WatchTriggered { digest, comment_count } => {
                    let _ = tx_stream.send(TuiStreamEvent::WatchTriggered { digest, comment_count });
                }
                StreamChunk::ClearStreamingText => {
                    let _ = tx_stream.send(TuiStreamEvent::ClearStreamingText);
                }
                StreamChunk::Done { .. } => {}
            }));

            let prompt_repo_root_override = repo_root_override.clone();
            thread::spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    engine_clone.chat_with_options(
                        &prompt,
                        ChatOptions {
                            tools: allow_tools && !read_only_for_turn,
                            force_max_think: max_think,
                            additional_dirs: prompt_additional_dirs,
                            repo_root_override: prompt_repo_root_override.clone(),
                            debug_context,
                            mode: mode_for_turn,
                            force_execute,
                            force_plan_only,
                            teammate_mode: teammate_mode_for_turn.clone(),
                            detect_urls,
                            watch_files: watch_files_enabled,
                            images: images_for_turn,
                            ..Default::default()
                        },
                    )
                }));
                match result {
                    Ok(Ok(output)) => {
                        let _ = tx_done.send(TuiStreamEvent::Done(output));
                    }
                    Ok(Err(e)) => {
                        let _ = tx_done.send(TuiStreamEvent::Error(e.to_string()));
                    }
                    Err(_) => {
                        let _ = tx_done
                            .send(TuiStreamEvent::Error("agent thread panicked".to_string()));
                    }
                }
            });
        },
        move || current_ui_status(cwd, cfg, fmt_refresh.load(Ordering::Relaxed)).ok(),
    )
}

fn resolve_additional_dir(cwd: &Path, raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    }
}

fn pr_comments_payload(
    cwd: &Path,
    pr_number: &str,
    output_path: Option<&str>,
) -> Result<serde_json::Value> {
    let gh_available = std::process::Command::new("gh")
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false);
    if !gh_available {
        return Err(anyhow!(
            "GitHub CLI ('gh') is required for /pr_comments. Install gh and authenticate first."
        ));
    }

    let output = std::process::Command::new("gh")
        .current_dir(cwd)
        .args([
            "pr",
            "view",
            pr_number,
            "--json",
            "number,title,url,author,comments",
        ])
        .output()?;
    if !output.status.success() {
        return Err(anyhow!(
            "failed to fetch PR comments: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let comments_count = parsed["comments"]
        .as_array()
        .map(|rows| rows.len())
        .unwrap_or(0);

    let mut saved_to = None;
    if let Some(path) = output_path {
        let destination = resolve_additional_dir(cwd, path);
        if let Some(parent) = destination.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&destination, serde_json::to_vec_pretty(&parsed)?)?;
        saved_to = Some(destination.to_string_lossy().to_string());
    }

    Ok(json!({
        "schema": "deepseek.pr_comments.v1",
        "ok": true,
        "pr": pr_number,
        "summary": format!("Fetched {} comment(s) for PR #{}", comments_count, pr_number),
        "saved_to": saved_to,
        "data": parsed,
    }))
}

fn release_notes_payload(
    cwd: &Path,
    range: &str,
    output_path: Option<&str>,
) -> Result<serde_json::Value> {
    let output = std::process::Command::new("git")
        .current_dir(cwd)
        .args(["log", "--no-merges", "--pretty=format:%h %s", range])
        .output()?;
    if !output.status.success() {
        return Err(anyhow!(
            "failed to generate release notes: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    let mut saved_to = None;
    if let Some(path) = output_path {
        let destination = resolve_additional_dir(cwd, path);
        if let Some(parent) = destination.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let mut rendered = format!("# Release Notes ({range})\n\n");
        for line in &lines {
            rendered.push_str("- ");
            rendered.push_str(line);
            rendered.push('\n');
        }
        std::fs::write(&destination, rendered)?;
        saved_to = Some(destination.to_string_lossy().to_string());
    }

    Ok(json!({
        "schema": "deepseek.release_notes.v1",
        "ok": true,
        "range": range,
        "count": lines.len(),
        "lines": lines,
        "saved_to": saved_to,
    }))
}

fn login_payload(cwd: &Path) -> Result<serde_json::Value> {
    let cfg = AppConfig::ensure(cwd)?;
    let env_key = if cfg.llm.api_key_env.trim().is_empty() {
        "DEEPSEEK_API_KEY".to_string()
    } else {
        cfg.llm.api_key_env.clone()
    };
    let token = std::env::var(&env_key).unwrap_or_default();
    if token.trim().is_empty() {
        return Err(anyhow!(
            "missing {}. export the key first, then run /login",
            env_key
        ));
    }

    let runtime_auth = runtime_dir(cwd).join("auth").join("session.json");
    if let Some(parent) = runtime_auth.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mask = format!(
        "***{}",
        token
            .chars()
            .rev()
            .take(4)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
    );
    let session_payload = json!({
        "provider": "deepseek",
        "api_key_env": env_key,
        "masked": mask,
        "created_at": Utc::now().to_rfc3339(),
    });
    std::fs::write(&runtime_auth, serde_json::to_vec_pretty(&session_payload)?)?;

    let local_path = AppConfig::project_local_settings_path(cwd);
    if let Some(parent) = local_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut root = if local_path.exists() {
        let raw = std::fs::read_to_string(&local_path)?;
        serde_json::from_str::<serde_json::Value>(&raw).unwrap_or_else(|_| json!({}))
    } else {
        json!({})
    };
    if !root.is_object() {
        root = json!({});
    }
    let map = root
        .as_object_mut()
        .ok_or_else(|| anyhow!("settings.local.json root must be an object"))?;
    let llm_value = map.entry("llm".to_string()).or_insert_with(|| json!({}));
    if !llm_value.is_object() {
        *llm_value = json!({});
    }
    if let Some(llm) = llm_value.as_object_mut() {
        llm.insert("api_key".to_string(), json!(token));
        llm.insert("api_key_env".to_string(), json!(env_key));
    }
    std::fs::write(&local_path, serde_json::to_vec_pretty(&root)?)?;

    Ok(json!({
        "schema": "deepseek.auth.v1",
        "logged_in": true,
        "session_path": runtime_auth.to_string_lossy().to_string(),
        "settings_path": local_path.to_string_lossy().to_string(),
        "message": "Login successful. Workspace auth session and settings.local.json updated.",
    }))
}

fn logout_payload(cwd: &Path) -> Result<serde_json::Value> {
    let cfg = AppConfig::ensure(cwd)?;
    let env_key = if cfg.llm.api_key_env.trim().is_empty() {
        "DEEPSEEK_API_KEY".to_string()
    } else {
        cfg.llm.api_key_env.clone()
    };
    let runtime_auth = runtime_dir(cwd).join("auth").join("session.json");
    let session_removed = if runtime_auth.exists() {
        std::fs::remove_file(&runtime_auth)?;
        true
    } else {
        false
    };

    let local_path = AppConfig::project_local_settings_path(cwd);
    let mut settings_updated = false;
    if local_path.exists() {
        let raw = std::fs::read_to_string(&local_path)?;
        let mut root =
            serde_json::from_str::<serde_json::Value>(&raw).unwrap_or_else(|_| json!({}));
        if let Some(llm) = root.get_mut("llm").and_then(|entry| entry.as_object_mut())
            && llm.remove("api_key").is_some()
        {
            settings_updated = true;
        }
        std::fs::write(&local_path, serde_json::to_vec_pretty(&root)?)?;
    }

    // SAFETY: called on explicit user command from main thread.
    unsafe { std::env::remove_var(env_key) };

    Ok(json!({
        "schema": "deepseek.auth.v1",
        "logged_in": false,
        "session_removed": session_removed,
        "settings_updated": settings_updated,
        "message": "Logged out. Session file removed and workspace API key unset.",
    }))
}

fn desktop_payload(cwd: &Path, args: &[String]) -> Result<serde_json::Value> {
    let teleport_args = if args.is_empty() {
        parse_teleport_args(vec!["link".to_string()]).unwrap_or_default()
    } else {
        parse_teleport_args(args.to_vec()).unwrap_or_default()
    };
    let execution = teleport_now(cwd, teleport_args)?;
    let session_id = Store::new(cwd)?
        .load_latest_session()?
        .map(|session| session.session_id.to_string());
    Ok(json!({
        "schema": "deepseek.desktop_handoff.v2",
        "mode": execution.mode,
        "bundle_id": execution.bundle_id,
        "handoff_id": execution.handoff_id,
        "link_url": execution.link_url,
        "token": execution.token,
        "path": execution.path.or(execution.imported),
        "session_id": session_id,
        "resume_command": session_id.map(|id| format!("deepseek --resume {id}")),
    }))
}

/// Generate 2-3 context-aware follow-up prompt suggestions from the assistant response.
fn generate_prompt_suggestions(response: &str) -> Vec<String> {
    let lower = response.to_ascii_lowercase();
    let mut suggestions = Vec::new();

    // Detect edits → suggest test/review/commit
    if lower.contains("applied") || lower.contains("modified") || lower.contains("created") {
        suggestions.push("run tests".to_string());
        suggestions.push("/diff".to_string());
        if lower.contains("created") {
            suggestions.push("document this change".to_string());
        }
    }

    // Detect errors → suggest debug/fix
    if lower.contains("error") || lower.contains("failed") || lower.contains("panic") {
        suggestions.push("fix the error".to_string());
        suggestions.push("show the full stack trace".to_string());
    }

    // Detect test results → suggest coverage
    if lower.contains("test") && (lower.contains("passed") || lower.contains("ok")) {
        suggestions.push("check test coverage".to_string());
    }

    // Detect refactoring → suggest verification
    if lower.contains("refactor") || lower.contains("renamed") || lower.contains("moved") {
        suggestions.push("verify no regressions".to_string());
    }

    // Detect explanations → suggest deeper dives
    if lower.contains("because") || lower.contains("reason") || lower.contains("architecture") {
        suggestions.push("explain in more detail".to_string());
    }

    // Always cap at 3 suggestions
    suggestions.truncate(3);

    // Fallback if nothing triggered
    if suggestions.is_empty() {
        suggestions.push("/compact".to_string());
        suggestions.push("/cost".to_string());
    }

    suggestions
}
fn todos_payload(cwd: &Path, args: &[String]) -> Result<serde_json::Value> {
    let mut max_results = 100usize;
    let mut query = None;
    if let Some(first) = args.first() {
        if let Ok(parsed) = first.parse::<usize>() {
            max_results = parsed.clamp(1, 2000);
            query = args.get(1).cloned();
        } else {
            query = Some(first.clone());
        }
    }
    let query_lower = query.as_deref().map(|value| value.to_ascii_lowercase());

    let output = std::process::Command::new("rg")
        .current_dir(cwd)
        .args([
            "--line-number",
            "--no-heading",
            "--hidden",
            "--glob",
            "!.git/*",
            "--glob",
            "!target/*",
            "--glob",
            "!node_modules/*",
            "TODO|FIXME",
            ".",
        ])
        .output();

    let mut items = Vec::new();
    if let Ok(out) = output {
        let stdout = String::from_utf8_lossy(&out.stdout);
        for line in stdout.lines() {
            let mut parts = line.splitn(3, ':');
            let path = parts.next().unwrap_or_default();
            let line_no = parts
                .next()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
            let text = parts.next().unwrap_or_default().trim().to_string();
            if let Some(filter) = query_lower.as_deref()
                && !text.to_ascii_lowercase().contains(filter)
            {
                continue;
            }
            items.push(json!({
                "path": path,
                "line": line_no,
                "text": text,
            }));
            if items.len() >= max_results {
                break;
            }
        }
    }

    Ok(json!({
        "schema": "deepseek.todos.v1",
        "count": items.len(),
        "query": query,
        "items": items,
    }))
}

fn chrome_payload(cwd: &Path, args: &[String]) -> Result<serde_json::Value> {
    let mut idx = 0usize;
    let subcommand = args
        .first()
        .cloned()
        .unwrap_or_else(|| "status".to_string())
        .to_ascii_lowercase();
    if !args.is_empty() {
        idx = 1;
    }

    let port = std::env::var("DEEPSEEK_CHROME_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(9222);
    let cfg = AppConfig::load(cwd).unwrap_or_default();
    let mut session = ChromeSession::new(port)?;
    session.set_allow_stub_fallback(cfg.tools.chrome.allow_stub_fallback);
    let debug_url = session.debug_url().to_string();

    match subcommand.as_str() {
        "status" => {
            let status = session.connection_status()?;
            Ok(json!({
                "schema": "deepseek.chrome.v1",
                "action": "status",
                "port": port,
                "connected": status.connected,
                "debug_url": debug_url,
                "status": status,
            }))
        }
        "reconnect" => {
            let status = session.reconnect(true)?;
            Ok(json!({
                "schema": "deepseek.chrome.v1",
                "action": "reconnect",
                "port": port,
                "connected": status.connected,
                "debug_url": debug_url,
                "status": status,
            }))
        }
        "tabs" => match session.list_tabs() {
            Ok(tabs) => Ok(json!({
                "schema": "deepseek.chrome.v1",
                "action": "tabs",
                "port": port,
                "count": tabs.len(),
                "tabs": tabs,
            })),
            Err(err) => Ok(chrome_error_payload("tabs", port, &debug_url, &err)),
        },
        "tab" => {
            let Some(tab_action) = args.get(idx).map(|v| v.to_ascii_lowercase()) else {
                return Err(anyhow!("usage: /chrome tab [new <url>|focus <target_id>]"));
            };
            idx += 1;
            match tab_action.as_str() {
                "new" => {
                    let url = args.get(idx).map(String::as_str).unwrap_or("about:blank");
                    match session.create_tab(url) {
                        Ok(tab) => Ok(json!({
                            "schema": "deepseek.chrome.v1",
                            "action": "tab.new",
                            "port": port,
                            "ok": true,
                            "tab": tab,
                        })),
                        Err(err) => Ok(chrome_error_payload("tab.new", port, &debug_url, &err)),
                    }
                }
                "focus" | "activate" => {
                    let Some(target_id) = args.get(idx) else {
                        return Err(anyhow!("usage: /chrome tab focus <target_id>"));
                    };
                    match session.activate_tab(target_id) {
                        Ok(_) => Ok(json!({
                            "schema": "deepseek.chrome.v1",
                            "action": "tab.focus",
                            "port": port,
                            "ok": true,
                            "target_id": target_id,
                        })),
                        Err(err) => Ok(chrome_error_payload("tab.focus", port, &debug_url, &err)),
                    }
                }
                _ => Err(anyhow!("usage: /chrome tab [new <url>|focus <target_id>]")),
            }
        }
        "navigate" => {
            let Some(url) = args.get(idx) else {
                return Err(anyhow!("usage: /chrome navigate <url>"));
            };
            if let Err(err) = session.ensure_live_connection() {
                return Ok(chrome_error_payload("navigate", port, &debug_url, &err));
            }
            match session.navigate(url) {
                Ok(result) => Ok(json!({
                    "schema": "deepseek.chrome.v1",
                    "action": "navigate",
                    "port": port,
                    "url": url,
                    "ok": result.error.is_none(),
                    "error": result.error.map(|e| e.message),
                })),
                Err(err) => Ok(chrome_error_payload("navigate", port, &debug_url, &err)),
            }
        }
        "click" => {
            let Some(selector) = args.get(idx) else {
                return Err(anyhow!("usage: /chrome click <css-selector>"));
            };
            if let Err(err) = session.ensure_live_connection() {
                return Ok(chrome_error_payload("click", port, &debug_url, &err));
            }
            match session.click(selector) {
                Ok(result) => Ok(json!({
                    "schema": "deepseek.chrome.v1",
                    "action": "click",
                    "port": port,
                    "selector": selector,
                    "ok": result.error.is_none(),
                    "error": result.error.map(|e| e.message),
                })),
                Err(err) => Ok(chrome_error_payload("click", port, &debug_url, &err)),
            }
        }
        "type" => {
            let Some(selector) = args.get(idx) else {
                return Err(anyhow!("usage: /chrome type <css-selector> <text>"));
            };
            let Some(text) = args.get(idx + 1) else {
                return Err(anyhow!("usage: /chrome type <css-selector> <text>"));
            };
            if let Err(err) = session.ensure_live_connection() {
                return Ok(chrome_error_payload("type", port, &debug_url, &err));
            }
            match session.type_text(selector, text) {
                Ok(result) => Ok(json!({
                    "schema": "deepseek.chrome.v1",
                    "action": "type",
                    "port": port,
                    "selector": selector,
                    "text": text,
                    "ok": result.error.is_none(),
                    "error": result.error.map(|e| e.message),
                })),
                Err(err) => Ok(chrome_error_payload("type", port, &debug_url, &err)),
            }
        }
        "evaluate" => {
            let expression = args[idx..].join(" ");
            if expression.trim().is_empty() {
                return Err(anyhow!("usage: /chrome evaluate <javascript-expression>"));
            }
            if let Err(err) = session.ensure_live_connection() {
                return Ok(chrome_error_payload("evaluate", port, &debug_url, &err));
            }
            match session.evaluate(&expression) {
                Ok(value) => Ok(json!({
                    "schema": "deepseek.chrome.v1",
                    "action": "evaluate",
                    "port": port,
                    "expression": expression,
                    "ok": true,
                    "value": value,
                })),
                Err(err) => Ok(chrome_error_payload("evaluate", port, &debug_url, &err)),
            }
        }
        "record" => {
            let duration_seconds = args
                .get(idx)
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(3)
                .clamp(1, 60);
            let output_path = args
                .get(idx + 1)
                .cloned()
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    runtime_dir(cwd)
                        .join("chrome")
                        .join("recordings")
                        .join(format!("recording-{}.gif", Utc::now().timestamp()))
                });
            if let Err(err) = session.ensure_live_connection() {
                return Ok(chrome_error_payload("record", port, &debug_url, &err));
            }
            let frames_dir = runtime_dir(cwd)
                .join("chrome")
                .join("recordings")
                .join(format!("frames-{}", Utc::now().timestamp_millis()));
            std::fs::create_dir_all(&frames_dir)?;
            let start = std::time::Instant::now();
            let mut frame_count = 0usize;
            while start.elapsed().as_secs() < duration_seconds {
                let base64_png = session.screenshot(ScreenshotFormat::Png)?;
                let bytes = base64::engine::general_purpose::STANDARD.decode(base64_png)?;
                let frame_path = frames_dir.join(format!("frame-{:04}.png", frame_count));
                std::fs::write(&frame_path, bytes)?;
                frame_count += 1;
                std::thread::sleep(std::time::Duration::from_millis(250));
            }
            if frame_count == 0 {
                return Err(anyhow!("recording produced no frames"));
            }

            if let Some(parent) = output_path.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent)?;
            }

            let ffmpeg_result = std::process::Command::new("ffmpeg")
                .args([
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-framerate",
                    "4",
                    "-i",
                    "frame-%04d.png",
                    output_path.to_string_lossy().as_ref(),
                ])
                .current_dir(&frames_dir)
                .output();

            let mut export_mode = "frames_only".to_string();
            let mut export_error = None;
            if let Ok(output) = ffmpeg_result {
                if output.status.success() {
                    export_mode = "gif".to_string();
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                    if !stderr.is_empty() {
                        export_error = Some(stderr);
                    }
                }
            } else {
                export_error = Some("ffmpeg unavailable; kept PNG frame sequence".to_string());
            }

            Ok(json!({
                "schema": "deepseek.chrome.v1",
                "action": "record",
                "port": port,
                "ok": true,
                "duration_seconds": duration_seconds,
                "frame_count": frame_count,
                "frames_dir": frames_dir.to_string_lossy().to_string(),
                "output_path": if export_mode == "gif" {
                    output_path.to_string_lossy().to_string()
                } else {
                    String::new()
                },
                "export_mode": export_mode,
                "export_error": export_error,
            }))
        }
        "screenshot" => {
            let format = match args.get(idx).map(|v| v.to_ascii_lowercase()) {
                Some(value) if value == "jpeg" || value == "jpg" => ScreenshotFormat::Jpeg,
                Some(value) if value == "webp" => ScreenshotFormat::Webp,
                _ => ScreenshotFormat::Png,
            };
            let output_path = args
                .get(idx + 1)
                .cloned()
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    let ext = match format {
                        ScreenshotFormat::Png => "png",
                        ScreenshotFormat::Jpeg => "jpg",
                        ScreenshotFormat::Webp => "webp",
                    };
                    runtime_dir(cwd).join("chrome").join(format!(
                        "screenshot-{}.{}",
                        Utc::now().timestamp(),
                        ext
                    ))
                });
            if let Some(parent) = output_path.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent)?;
            }
            if let Err(err) = session.ensure_live_connection() {
                return Ok(chrome_error_payload("screenshot", port, &debug_url, &err));
            }
            match session.screenshot(format) {
                Ok(data) => {
                    let bytes = base64::engine::general_purpose::STANDARD.decode(data)?;
                    std::fs::write(&output_path, bytes)?;
                    Ok(json!({
                        "schema": "deepseek.chrome.v1",
                        "action": "screenshot",
                        "port": port,
                        "path": output_path.to_string_lossy().to_string(),
                        "ok": true,
                    }))
                }
                Err(err) => Ok(chrome_error_payload("screenshot", port, &debug_url, &err)),
            }
        }
        "console" => match session
            .ensure_live_connection()
            .and_then(|_| session.read_console())
        {
            Ok(entries) => Ok(json!({
                "schema": "deepseek.chrome.v1",
                "action": "console",
                "port": port,
                "count": entries.len(),
                "entries": entries,
            })),
            Err(err) => Ok(chrome_error_payload("console", port, &debug_url, &err)),
        },
        _ => Err(anyhow!(
            "usage: /chrome [status|reconnect|tabs|tab new <url>|tab focus <target_id>|navigate <url>|click <selector>|type <selector> <text>|evaluate <js>|record [seconds] [output.gif]|screenshot [png|jpeg|webp] [output]|console]"
        )),
    }
}

fn chrome_error_payload(
    action: &str,
    port: u16,
    debug_url: &str,
    err: &anyhow::Error,
) -> serde_json::Value {
    let message = err.to_string();
    let lower = message.to_ascii_lowercase();
    let (kind, hints): (&str, Vec<&str>) =
        if lower.contains("endpoint_unreachable") || lower.contains("connection refused") {
            (
                "endpoint_unreachable",
                vec![
                    "Start Chrome with --remote-debugging-port=9222",
                    "Verify DEEPSEEK_CHROME_PORT points to the active debugging port",
                ],
            )
        } else if lower.contains("no_page_targets") || lower.contains("no debuggable page target") {
            (
                "no_page_targets",
                vec![
                    "Run /chrome reconnect to create a recovery tab",
                    "Open at least one browser tab in the target Chrome profile",
                ],
            )
        } else if lower.contains("timed out") {
            (
                "endpoint_timeout",
                vec![
                    "Confirm local firewall/proxy rules are not blocking localhost debug traffic",
                    "Retry /chrome reconnect once the browser is responsive",
                ],
            )
        } else if lower.contains("cdp error") {
            (
                "cdp_command_failed",
                vec![
                    "Retry the command after /chrome reconnect",
                    "Use /chrome tabs to confirm the active page target",
                ],
            )
        } else {
            (
                "chrome_error",
                vec![
                    "Run /chrome status for live connection diagnostics",
                    "Run /chrome reconnect to recover stale sessions",
                ],
            )
        };

    json!({
        "schema": "deepseek.chrome.v1",
        "action": action,
        "port": port,
        "ok": false,
        "debug_url": debug_url,
        "error": {
            "kind": kind,
            "message": message,
            "hints": hints,
        }
    })
}

fn parse_debug_analysis_args(args: &[String]) -> Result<Option<DoctorArgs>> {
    if args.is_empty() {
        return Ok(None);
    }

    let mut idx = 0usize;
    if args[0].eq_ignore_ascii_case("analyze") {
        idx = 1;
        if idx >= args.len() {
            return Err(anyhow!(
                "usage: /debug analyze <auto|runtime|test|performance> <file-or-text>"
            ));
        }
    }

    let Some(mode) = parse_debug_mode_token(&args[idx]) else {
        if idx == 0 {
            return Ok(None);
        }
        return Err(anyhow!(
            "usage: /debug analyze <auto|runtime|test|performance> <file-or-text>"
        ));
    };
    idx += 1;

    let remaining = &args[idx..];
    if remaining.is_empty() {
        return Err(anyhow!(
            "missing debug input. provide a file path or inline text after the mode"
        ));
    }

    let mut doctor_args = DoctorArgs {
        mode,
        ..DoctorArgs::default()
    };
    if remaining.len() == 1 && Path::new(&remaining[0]).exists() {
        doctor_args.analyze_file = Some(remaining[0].clone());
    } else {
        doctor_args.analyze_text = Some(remaining.join(" "));
    }
    Ok(Some(doctor_args))
}

fn parse_debug_mode_token(token: &str) -> Option<DoctorModeArg> {
    match token.to_ascii_lowercase().as_str() {
        "auto" => Some(DoctorModeArg::Auto),
        "runtime" => Some(DoctorModeArg::Runtime),
        "test" | "tests" => Some(DoctorModeArg::Test),
        "performance" | "perf" => Some(DoctorModeArg::Performance),
        _ => None,
    }
}

pub(crate) fn load_tui_keybindings(cwd: &Path, cfg: &AppConfig) -> KeyBindings {
    let mut candidates = Vec::new();
    if !cfg.ui.keybindings_path.trim().is_empty() {
        candidates.push(std::path::PathBuf::from(expand_tilde(
            &cfg.ui.keybindings_path,
        )));
    }
    if let Some(path) = AppConfig::keybindings_path() {
        candidates.push(path);
    }
    candidates.push(runtime_dir(cwd).join("keybindings.json"));
    candidates.dedup();

    for path in candidates {
        if !path.exists() {
            continue;
        }
        if let Ok(bindings) = load_keybindings(&path) {
            return bindings;
        }
    }
    KeyBindings::default()
}

pub(crate) fn run_resume(cwd: &Path, args: RunArgs, json_mode: bool) -> Result<String> {
    if let Some(session_id) = args.session_id {
        let session_id = Uuid::parse_str(&session_id)?;
        let store = Store::new(cwd)?;
        let projection = store.rebuild_from_events(session_id)?;
        return Ok(format!(
            "resumed session={} turns={} steps={}",
            session_id,
            projection.transcript.len(),
            projection.step_status.len()
        ));
    }
    ensure_llm_ready(cwd, json_mode)?;
    #[allow(deprecated)]
    {
        AgentEngine::new(cwd)?.resume()
    }
}

pub(crate) fn run_print_mode(cwd: &Path, cli: &Cli) -> Result<()> {
    use deepseek_core::StreamChunk;
    use std::io::{IsTerminal, Read, Write as _, stdin, stdout};

    let prompt = if !cli.prompt_args.is_empty() {
        cli.prompt_args.join(" ")
    } else if !stdin().is_terminal() {
        let mut buf = String::new();
        stdin().read_to_string(&mut buf)?;
        buf.trim().to_string()
    } else {
        return Err(anyhow!(
            "-p/--print requires a prompt argument or stdin input"
        ));
    };

    if prompt.is_empty() {
        return Err(anyhow!("empty prompt"));
    }

    let json_mode = cli.json || cli.output_format == "json" || cli.output_format == "stream-json";
    let is_stream_json = cli.output_format == "stream-json";
    let is_text = !json_mode;
    ensure_llm_ready(cwd, json_mode)?;
    let mut engine = AgentEngine::new(cwd)?;
    apply_cli_flags(&mut engine, cli);
    wire_subagent_worker(&engine, cwd);

    // Handle --no-input: auto-deny all approval prompts
    if cli.no_input {
        engine.set_approval_handler(Box::new(|_call| Ok(false)));
    }

    // Handle --from-pr: fetch PR diff and prepend to prompt
    let prompt = if let Some(pr_number) = cli.from_pr {
        let diff = std::process::Command::new("gh")
            .args(["pr", "diff", &pr_number.to_string()])
            .output()?;
        let pr_context = String::from_utf8_lossy(&diff.stdout);
        format!("PR #{pr_number} diff:\n```\n{pr_context}\n```\n\n{prompt}")
    } else {
        prompt
    };

    // Set up streaming callback for real-time output
    if is_text || is_stream_json {
        let stream_json = is_stream_json;
        engine.set_stream_callback(std::sync::Arc::new(move |chunk: StreamChunk| {
            let out = stdout();
            let mut handle = out.lock();
            if stream_json {
                let event = stream_chunk_to_event_json(&chunk);
                let _ = serde_json::to_writer(&mut handle, &event);
                let _ = writeln!(handle);
                let _ = handle.flush();
                return;
            }
            match chunk {
                StreamChunk::ContentDelta(text) => {
                    let _ = write!(handle, "{text}");
                    let _ = handle.flush();
                }
                StreamChunk::ReasoningDelta(text) => {
                    let _ = text;
                    // In text mode, reasoning is not shown
                }
                StreamChunk::ArchitectStarted { iteration } => {
                    let _ = writeln!(handle, "\n[phase] architect started (iter {iteration})");
                    let _ = handle.flush();
                }
                StreamChunk::ArchitectCompleted {
                    iteration,
                    files,
                    no_edit,
                } => {
                    let _ = writeln!(
                        handle,
                        "[phase] architect completed (iter {iteration}) files={files} no_edit={no_edit}"
                    );
                    let _ = handle.flush();
                }
                StreamChunk::EditorStarted { iteration, files } => {
                    let _ = writeln!(handle, "[phase] editor started (iter {iteration}) files={files}");
                    let _ = handle.flush();
                }
                StreamChunk::EditorCompleted { iteration, status } => {
                    let _ = writeln!(handle, "[phase] editor completed (iter {iteration}) status={status}");
                    let _ = handle.flush();
                }
                StreamChunk::ApplyStarted { iteration } => {
                    let _ = writeln!(handle, "[phase] apply started (iter {iteration})");
                    let _ = handle.flush();
                }
                StreamChunk::ApplyCompleted {
                    iteration,
                    success,
                    summary,
                } => {
                    let _ = writeln!(
                        handle,
                        "[phase] apply {} (iter {iteration}) {}",
                        if success { "ok" } else { "failed" },
                        summary.replace('\n', " ")
                    );
                    let _ = handle.flush();
                }
                StreamChunk::VerifyStarted {
                    iteration,
                    commands,
                } => {
                    let _ = writeln!(
                        handle,
                        "[phase] verify started (iter {iteration}) {}",
                        commands.join(" | ")
                    );
                    let _ = handle.flush();
                }
                StreamChunk::VerifyCompleted {
                    iteration,
                    success,
                    summary,
                } => {
                    let _ = writeln!(
                        handle,
                        "[phase] verify {} (iter {iteration}) {}",
                        if success { "ok" } else { "failed" },
                        summary.replace('\n', " ")
                    );
                    let _ = handle.flush();
                }
                StreamChunk::LintStarted {
                    iteration,
                    commands,
                } => {
                    let _ = writeln!(
                        handle,
                        "[phase] lint started (iter {iteration}) {}",
                        commands.join(" | ")
                    );
                    let _ = handle.flush();
                }
                StreamChunk::LintCompleted {
                    iteration,
                    success,
                    fixed,
                    remaining,
                } => {
                    let _ = writeln!(
                        handle,
                        "[phase] lint {} (iter {iteration}) fixed={fixed} remaining={remaining}",
                        if success { "ok" } else { "failed" },
                    );
                    let _ = handle.flush();
                }
                StreamChunk::CommitProposal {
                    files,
                    touched_files,
                    loc_delta,
                    verify_commands,
                    verify_status,
                    suggested_message,
                } => {
                    let _ = verify_commands;
                    let _ = verify_status;
                    let _ = writeln!(
                        handle,
                        "[commit] ready files={} touched={} loc={} message=\"{}\"",
                        files.join(","),
                        touched_files,
                        loc_delta,
                        suggested_message
                    );
                    let _ = writeln!(
                        handle,
                        "✅ Verify passed. Run /commit to save changes, /diff to review, /undo to revert."
                    );
                    let _ = handle.flush();
                }
                StreamChunk::CommitCompleted { sha, message } => {
                    let _ = writeln!(handle, "[commit] completed sha={sha} message=\"{message}\"");
                    let _ = handle.flush();
                }
                StreamChunk::CommitSkipped => {
                    let _ = writeln!(handle, "[commit] skipped by user");
                    let _ = handle.flush();
                }
                StreamChunk::ToolCallStart {
                    tool_name,
                    args_summary,
                } => {
                    let _ = writeln!(handle, "\n[tool: {tool_name}] {args_summary}");
                    let _ = handle.flush();
                }
                StreamChunk::ToolCallEnd {
                    tool_name,
                    duration_ms,
                    success,
                    summary,
                } => {
                    let status = if success { "ok" } else { "error" };
                    let _ = writeln!(handle, "[tool: {tool_name}] {status} ({duration_ms}ms) {summary}");
                    let _ = handle.flush();
                }
                StreamChunk::ModeTransition { from, to, reason } => {
                    let _ = writeln!(handle, "\n[mode: {from} -> {to}: {reason}]");
                    let _ = handle.flush();
                }
                StreamChunk::SubagentSpawned { run_id, name, goal } => {
                    let _ = writeln!(handle, "[subagent:spawned] {name} ({run_id}) {goal}");
                    let _ = handle.flush();
                }
                StreamChunk::SubagentCompleted {
                    run_id,
                    name,
                    summary,
                } => {
                    let _ = writeln!(
                        handle,
                        "[subagent:completed] {name} ({run_id}) {}",
                        summary.replace('\n', " ")
                    );
                    let _ = handle.flush();
                }
                StreamChunk::SubagentFailed {
                    run_id,
                    name,
                    error,
                } => {
                    let _ = writeln!(
                        handle,
                        "[subagent:failed] {name} ({run_id}) {}",
                        error.replace('\n', " ")
                    );
                    let _ = handle.flush();
                }
                StreamChunk::ImageData { label, .. } => {
                    let _ = writeln!(handle, "[image: {label}]");
                    let _ = handle.flush();
                }
                StreamChunk::WatchTriggered { comment_count, .. } => {
                    let _ = writeln!(handle, "[watch: {comment_count} comment(s) detected]");
                    let _ = handle.flush();
                }
                StreamChunk::ClearStreamingText => {
                    // In non-TUI mode, nothing to clear — text already
                    // written to stdout.
                }
                StreamChunk::Done { .. } => {
                    let _ = writeln!(handle);
                    let _ = handle.flush();
                }
            }
        }));
    }

    let options = chat_options_from_cli(cli, true, ChatMode::Code);
    let output = engine.chat_with_options(&prompt, options)?;

    match cli.output_format.as_str() {
        "json" => {
            let session_id = Store::new(cwd)?
                .load_latest_session()?
                .map(|s| s.session_id.to_string())
                .unwrap_or_default();
            print_json(&json!({
                "output": output,
                "session_id": session_id,
                "model": AppConfig::load(cwd).unwrap_or_default().llm.base_model,
            }))?;
        }
        "stream-json" => {
            // Streaming was already output via callback; emit final summary
            let session_id = Store::new(cwd)?
                .load_latest_session()?
                .map(|s| s.session_id.to_string())
                .unwrap_or_default();
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "type": "result",
                    "output": output,
                    "session_id": session_id,
                    "model": AppConfig::load(cwd).unwrap_or_default().llm.base_model,
                }))?
            );
        }
        _ => {
            // Text was already streamed to stdout via callback; output is the session summary
            // (only print if there was no streaming, e.g., from cache hit)
        }
    }
    Ok(())
}

pub(crate) fn run_continue_session(
    cwd: &Path,
    json_mode: bool,
    _model: Option<&str>,
) -> Result<()> {
    let store = Store::new(cwd)?;
    let session = store
        .load_latest_session()?
        .ok_or_else(|| anyhow!("no previous session to continue"))?;
    let projection = store.rebuild_from_events(session.session_id)?;
    if !json_mode {
        println!(
            "resuming session {} ({} turns, state={:?})",
            session.session_id,
            projection.transcript.len(),
            session.status
        );
    }
    // Enter chat mode with the continued session context
    run_chat(cwd, json_mode, false, true, false, None)
}

pub(crate) fn run_resume_specific(
    cwd: &Path,
    session_id: &str,
    json_mode: bool,
    _model: Option<&str>,
) -> Result<()> {
    let store = Store::new(cwd)?;
    let uuid =
        Uuid::parse_str(session_id).map_err(|_| anyhow!("invalid session ID: {session_id}"))?;
    let session = store
        .load_session(uuid)?
        .ok_or_else(|| anyhow!("session not found: {session_id}"))?;
    let projection = store.rebuild_from_events(session.session_id)?;
    if !json_mode {
        println!(
            "resuming session {} ({} turns, state={:?})",
            session.session_id,
            projection.transcript.len(),
            session.status
        );
    }
    run_chat(cwd, json_mode, false, true, false, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn chrome_error_payload_classifies_endpoint_unreachable() {
        let payload = chrome_error_payload(
            "status",
            9222,
            "http://127.0.0.1:9222",
            &anyhow::anyhow!("Connection refused (os error 61)"),
        );
        assert_eq!(payload["error"]["kind"], "endpoint_unreachable");
    }

    #[test]
    fn chrome_error_payload_classifies_no_page_targets() {
        let payload = chrome_error_payload(
            "status",
            9222,
            "http://127.0.0.1:9222",
            &anyhow::anyhow!("no_page_targets: no debuggable page target available"),
        );
        assert_eq!(payload["error"]["kind"], "no_page_targets");
    }

    #[test]
    fn render_web_fetch_markdown_contains_metadata_and_extract_block() {
        let output = serde_json::json!({
            "status": 200,
            "content_type": "text/html",
            "truncated": false,
            "bytes": 320,
            "content": "Title\n\nFirst paragraph.\nSecond paragraph."
        });
        let rendered = render_web_fetch_markdown("https://example.com", &output, 6);
        assert!(rendered.contains("# Web Fetch"));
        assert!(rendered.contains("- URL: https://example.com"));
        assert!(rendered.contains("- Status: 200"));
        assert!(rendered.contains("## Extract"));
        assert!(rendered.contains("```text"));
        assert!(rendered.contains("First paragraph."));
    }

    #[test]
    fn render_web_search_markdown_formats_results_and_extract() {
        let results = vec![
            serde_json::json!({
                "title": "DeepSeek CLI docs",
                "url": "https://example.com/docs",
                "snippet": "A deterministic coding agent runtime."
            }),
            serde_json::json!({
                "title": "Architecture notes",
                "url": "https://example.com/notes",
                "snippet": "Architect editor apply verify."
            }),
        ];
        let rendered = render_web_search_markdown(
            "deepseek cli",
            &results,
            Some((
                "https://example.com/docs".to_string(),
                "DeepSeek CLI is terminal-native.".to_string(),
            )),
        );
        assert!(rendered.contains("# Web Search"));
        assert!(rendered.contains("- Query: deepseek cli"));
        assert!(rendered.contains("1. DeepSeek CLI docs"));
        assert!(rendered.contains("## Extract (https://example.com/docs)"));
        assert!(rendered.contains("DeepSeek CLI is terminal-native."));
    }

    #[test]
    fn parse_chat_mode_name_supports_expected_aliases() {
        assert_eq!(parse_chat_mode_name("ask"), Some(ChatMode::Ask));
        assert_eq!(parse_chat_mode_name("code"), Some(ChatMode::Code));
        assert_eq!(parse_chat_mode_name("architect"), Some(ChatMode::Architect));
        assert_eq!(parse_chat_mode_name("plan"), Some(ChatMode::Architect));
        assert_eq!(parse_chat_mode_name("context"), Some(ChatMode::Context));
        assert_eq!(parse_chat_mode_name("invalid"), None);
    }

    #[test]
    fn watch_scan_returns_digest_and_payload() -> Result<()> {
        // rg must be a real binary on PATH (not just a shell alias)
        let rg_available = std::process::Command::new("rg")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success());
        if !rg_available {
            eprintln!("skipping watch_scan test: rg not found on PATH");
            return Ok(());
        }
        let dir = tempdir()?;
        let root = dir.path();
        fs::write(
            root.join("notes.md"),
            "todo list\nTODO(ai): inspect runtime flow\n",
        )?;
        let result = scan_watch_comment_payload(root);
        assert!(result.is_some());
        let (digest, payload) = result.unwrap();
        assert!(digest > 0);
        assert!(payload.contains("TODO(ai): inspect runtime flow"));
        assert!(payload.contains("notes.md"));
        Ok(())
    }

    #[test]
    fn profile_save_and_load_roundtrip() -> Result<()> {
        let dir = tempdir()?;
        let root = dir.path();
        let additional_dirs = vec![root.join("src"), root.join("docs")];
        let save = slash_save_profile_output(
            root,
            &[String::from("roundtrip")],
            ChatMode::Architect,
            true,
            true,
            &additional_dirs,
        )?;
        assert!(save.contains("saved chat profile"));

        let (mode, read_only, thinking, dirs, load_msg) =
            slash_load_profile_output(root, &[String::from("roundtrip")])?;
        assert_eq!(mode, ChatMode::Architect);
        assert!(read_only);
        assert!(thinking);
        assert_eq!(dirs, additional_dirs);
        assert!(load_msg.contains("loaded chat profile"));
        Ok(())
    }

    #[test]
    fn slash_git_without_args_returns_usage() -> Result<()> {
        let dir = tempdir()?;
        run_process(dir.path(), "git", &["init"])?;
        let output = slash_git_output(dir.path(), &[])?;
        assert!(output.trim().is_empty());
        Ok(())
    }

    #[test]
    fn slash_voice_status_reports_capability_probe() -> Result<()> {
        let output = slash_voice_output(&[String::from("status")])?;
        assert!(output.contains("voice status:"));
        Ok(())
    }
}
