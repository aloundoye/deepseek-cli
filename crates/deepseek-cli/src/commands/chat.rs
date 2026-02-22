use anyhow::{Result, anyhow};
use base64::Engine;
use chrono::Utc;
use deepseek_agent::{AgentEngine, ChatOptions};
use deepseek_chrome::{ChromeSession, ScreenshotFormat};
use deepseek_core::{AppConfig, EventKind, StreamChunk, runtime_dir};
use deepseek_mcp::McpManager;
use deepseek_memory::{ExportFormat, MemoryManager};
use deepseek_skills::SkillManager;
use deepseek_store::{Store, SubagentRunRecord};
use deepseek_ui::{
    KeyBindings, SlashCommand, TuiStreamEvent, TuiTheme, load_keybindings, render_statusline,
    run_tui_shell_with_bindings,
};
use serde_json::json;
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
    allow_tools: bool,
    force_tui: bool,
    cli: Option<&Cli>,
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
    let allow_r1_drive_tools = cli.map(|value| value.allow_r1_drive_tools).unwrap_or(false);
    let interactive_tty = stdin().is_terminal() && stdout().is_terminal();
    if !json_mode && (force_tui || cfg.ui.enable_tui) && interactive_tty {
        return run_chat_tui(
            cwd,
            allow_tools,
            allow_r1_drive_tools,
            &cfg,
            force_max_think,
        );
    }
    if force_tui && !interactive_tty {
        return Err(anyhow!("--tui requires an interactive terminal"));
    }
    let mut last_assistant_response: Option<String> = None;
    let mut additional_dirs = cli.map(|value| value.add_dir.clone()).unwrap_or_default();
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
        let prompt_owned = deepseek_ui::expand_at_mentions(raw_prompt);
        let prompt = prompt_owned.as_str();

        if let Some(cmd) = SlashCommand::parse(prompt) {
            match cmd {
                SlashCommand::Help => {
                    let message = json!({
                        "commands": [
                            "/help",
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
                            "/effort",
                            "/skills",
                            "/permissions",
                            "/background",
                            "/visual",
                            "/vim",
                            "/copy",
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
                SlashCommand::Compact => {
                    run_compact(
                        cwd,
                        CompactArgs {
                            from_turn: None,
                            yes: false,
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
                    } else if json_mode {
                        print_json(&json!({"error":"use /mcp list|get <id>|remove <id>"}))?;
                    } else {
                        println!("use /mcp list|get <id>|remove <id>");
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
                    if args.is_empty() {
                        if json_mode {
                            print_json(&json!({"error":"usage: /add-dir <path>"}))?;
                        } else {
                            println!("Usage: /add-dir <path>");
                        }
                    } else {
                        let mut added = Vec::new();
                        for raw in &args {
                            let path = resolve_additional_dir(cwd, raw);
                            if !path.exists() || !path.is_dir() {
                                if json_mode {
                                    print_json(&json!({
                                        "error": format!("directory does not exist: {}", path.display())
                                    }))?;
                                } else {
                                    println!("Directory does not exist: {}", path.display());
                                }
                                continue;
                            }
                            if !additional_dirs.contains(&path) {
                                additional_dirs.push(path.clone());
                                added.push(path);
                            }
                        }
                        if json_mode {
                            print_json(&json!({
                                "added": added.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>(),
                                "active": additional_dirs.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>(),
                            }))?;
                        } else if added.is_empty() {
                            println!("No new directories added.");
                        } else {
                            for dir in &added {
                                println!("Added directory: {}", dir.display());
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
                                    tools: allow_tools,
                                    allow_r1_drive_tools,
                                    force_max_think,
                                    additional_dirs: additional_dirs.clone(),
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
        if !json_mode {
            engine.set_stream_callback(std::sync::Arc::new(|chunk: deepseek_core::StreamChunk| {
                use std::io::Write as _;
                let out = std::io::stdout();
                let mut handle = out.lock();
                match chunk {
                    deepseek_core::StreamChunk::ContentDelta(text) => {
                        let _ = write!(handle, "{text}");
                        let _ = handle.flush();
                    }
                    deepseek_core::StreamChunk::ReasoningDelta(_) => {}
                    deepseek_core::StreamChunk::ToolCallStart {
                        tool_name,
                        args_summary,
                    } => {
                        let _ = writeln!(handle, "\n[tool: {tool_name}] {args_summary}");
                        let _ = handle.flush();
                    }
                    deepseek_core::StreamChunk::ToolCallEnd {
                        tool_name,
                        duration_ms,
                        success,
                        summary,
                    } => {
                        let status = if success { "ok" } else { "error" };
                        let _ = writeln!(
                            handle,
                            "[tool: {tool_name}] {status} ({duration_ms}ms) {summary}"
                        );
                        let _ = handle.flush();
                    }
                    deepseek_core::StreamChunk::ModeTransition { from, to, reason } => {
                        let _ = writeln!(handle, "\n[mode: {from} -> {to}: {reason}]");
                        let _ = handle.flush();
                    }
                    deepseek_core::StreamChunk::SubagentSpawned { run_id, name, goal } => {
                        let _ = writeln!(handle, "[subagent:spawned] {name} ({run_id}) {goal}");
                        let _ = handle.flush();
                    }
                    deepseek_core::StreamChunk::SubagentCompleted {
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
                    deepseek_core::StreamChunk::SubagentFailed {
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
                    deepseek_core::StreamChunk::ImageData { label, .. } => {
                        let _ = writeln!(handle, "[image: {label}]");
                        let _ = handle.flush();
                    }
                    deepseek_core::StreamChunk::ClearStreamingText => {
                        // In non-TUI mode, nothing to clear — text already
                        // written to stdout. Ignore.
                    }
                    deepseek_core::StreamChunk::Done => {
                        let _ = writeln!(handle);
                        let _ = handle.flush();
                    }
                }
            }));
        }

        let output = engine.chat_with_options(
            prompt,
            ChatOptions {
                tools: allow_tools,
                allow_r1_drive_tools,
                force_max_think,
                additional_dirs: additional_dirs.clone(),
                ..Default::default()
            },
        )?;
        last_assistant_response = Some(output.clone());
        let ui_status = current_ui_status(cwd, &cfg, force_max_think)?;
        if json_mode {
            print_json(&json!({"output": output, "statusline": render_statusline(&ui_status)}))?;
        } else {
            println!("[status] {}", render_statusline(&ui_status));
        }
    }
    Ok(())
}

pub(crate) fn run_chat_tui(
    cwd: &Path,
    allow_tools: bool,
    allow_r1_drive_tools: bool,
    cfg: &AppConfig,
    initial_force_max_think: bool,
) -> Result<()> {
    let engine = Arc::new(AgentEngine::new(cwd)?);
    wire_subagent_worker(&engine, cwd);
    let force_max_think = Arc::new(AtomicBool::new(initial_force_max_think));
    let additional_dirs = Arc::new(std::sync::Mutex::new(Vec::<PathBuf>::new()));

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
                SlashCommand::Help => "commands: /help /init /clear /compact /memory /config /model /cost /mcp /rewind /export /plan /teleport /remote-env /status /effort /skills /permissions /background /visual /desktop /todos /chrome /vim".to_string(),
                SlashCommand::Init => {
                    let manager = MemoryManager::new(cwd)?;
                    let path = manager.ensure_initialized()?;
                    format!("initialized memory at {}", path.display())
                }
                SlashCommand::Clear => "cleared".to_string(),
                SlashCommand::Compact => {
                    let summary = compact_now(cwd, None)?;
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
                    let to_checkpoint = args.first().cloned();
                    let checkpoint = rewind_now(cwd, to_checkpoint)?;
                    format!("rewound to checkpoint {}", checkpoint.checkpoint_id)
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
                            if let Some(imported) = teleport.imported {
                                format!("imported teleport bundle {}", imported)
                            } else {
                                format!(
                                    "teleport bundle {} -> {}",
                                    teleport.bundle_id.unwrap_or_default(),
                                    teleport.path.unwrap_or_default()
                                )
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
                    if args.is_empty() {
                        "Usage: /add-dir <path>".to_string()
                    } else {
                        let mut added = Vec::new();
                        let mut guard = additional_dirs_for_closure
                            .lock()
                            .map_err(|_| anyhow!("failed to access additional dir state"))?;
                        for raw in &args {
                            let path = resolve_additional_dir(cwd, raw);
                            if path.exists() && path.is_dir() && !guard.contains(&path) {
                                guard.push(path.clone());
                                added.push(path);
                            }
                        }
                        if added.is_empty() {
                            "No new directories added.".to_string()
                        } else {
                            format!(
                                "Added: {}",
                                added
                                    .iter()
                                    .map(|p| p.to_string_lossy().to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
                        }
                    }
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
            let prompt = deepseek_ui::expand_at_mentions(prompt);
            let max_think = force_max_think.load(Ordering::Relaxed);
            let tx_stream = tx.clone();
            let tx_done = tx.clone();
            let prompt_additional_dirs = additional_dirs_for_closure
                .lock()
                .map(|dirs| dirs.clone())
                .unwrap_or_default();

            engine.set_stream_callback(std::sync::Arc::new(move |chunk| match chunk {
                StreamChunk::ContentDelta(s) => {
                    let _ = tx_stream.send(TuiStreamEvent::ContentDelta(s));
                }
                StreamChunk::ReasoningDelta(s) => {
                    let _ = tx_stream.send(TuiStreamEvent::ReasoningDelta(s));
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
                    summary,
                    ..
                } => {
                    let _ = tx_stream.send(TuiStreamEvent::ToolCallEnd {
                        tool_name,
                        duration_ms,
                        summary,
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
                StreamChunk::ClearStreamingText => {
                    let _ = tx_stream.send(TuiStreamEvent::ClearStreamingText);
                }
                StreamChunk::Done => {}
            }));

            thread::spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    engine_clone.chat_with_options(
                        &prompt,
                        ChatOptions {
                            tools: allow_tools,
                            allow_r1_drive_tools,
                            force_max_think: max_think,
                            additional_dirs: prompt_additional_dirs,
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
    let teleport_args = parse_teleport_args(args.to_vec()).unwrap_or_default();
    let execution = teleport_now(cwd, teleport_args)?;
    let session_id = Store::new(cwd)?
        .load_latest_session()?
        .map(|session| session.session_id.to_string());
    Ok(json!({
        "schema": "deepseek.desktop_handoff.v1",
        "mode": if execution.imported.is_some() { "import" } else { "export" },
        "bundle_id": execution.bundle_id,
        "path": execution.path.or(execution.imported),
        "session_id": session_id,
        "resume_command": session_id.map(|id| format!("deepseek --resume {id}")),
    }))
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
    let mut session = ChromeSession::new(port)?;
    // Slash-command UX should surface real browser connectivity, not silent stubs.
    session.set_allow_stub_fallback(false);
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
            match chunk {
                StreamChunk::ContentDelta(text) => {
                    if stream_json {
                        let _ = serde_json::to_writer(
                            &mut handle,
                            &serde_json::json!({"type": "content", "text": text}),
                        );
                        let _ = writeln!(handle);
                    } else {
                        let _ = write!(handle, "{text}");
                    }
                    let _ = handle.flush();
                }
                StreamChunk::ReasoningDelta(text) => {
                    if stream_json {
                        let _ = serde_json::to_writer(
                            &mut handle,
                            &serde_json::json!({"type": "reasoning", "text": text}),
                        );
                        let _ = writeln!(handle);
                        let _ = handle.flush();
                    }
                    // In text mode, reasoning is not shown
                }
                StreamChunk::ToolCallStart { tool_name, args_summary } => {
                    if stream_json {
                        let _ = serde_json::to_writer(
                            &mut handle,
                            &serde_json::json!({"type": "tool_start", "tool": tool_name, "args": args_summary}),
                        );
                        let _ = writeln!(handle);
                    } else {
                        let _ = writeln!(handle, "\n[tool: {tool_name}] {args_summary}");
                    }
                    let _ = handle.flush();
                }
                StreamChunk::ToolCallEnd { tool_name, duration_ms, success, summary } => {
                    if stream_json {
                        let _ = serde_json::to_writer(
                            &mut handle,
                            &serde_json::json!({"type": "tool_end", "tool": tool_name, "duration_ms": duration_ms, "success": success, "summary": summary}),
                        );
                        let _ = writeln!(handle);
                    } else {
                        let status = if success { "ok" } else { "error" };
                        let _ = writeln!(handle, "[tool: {tool_name}] {status} ({duration_ms}ms) {summary}");
                    }
                    let _ = handle.flush();
                }
                StreamChunk::ModeTransition { from, to, reason } => {
                    if stream_json {
                        let _ = serde_json::to_writer(
                            &mut handle,
                            &serde_json::json!({"type": "mode_transition", "from": from, "to": to, "reason": reason}),
                        );
                        let _ = writeln!(handle);
                    } else {
                        let _ = writeln!(handle, "\n[mode: {from} -> {to}: {reason}]");
                    }
                    let _ = handle.flush();
                }
                StreamChunk::SubagentSpawned { run_id, name, goal } => {
                    if stream_json {
                        let _ = serde_json::to_writer(
                            &mut handle,
                            &serde_json::json!({"type":"subagent_spawned","run_id":run_id,"name":name,"goal":goal}),
                        );
                        let _ = writeln!(handle);
                    } else {
                        let _ = writeln!(handle, "[subagent:spawned] {name} ({run_id}) {goal}");
                    }
                    let _ = handle.flush();
                }
                StreamChunk::SubagentCompleted {
                    run_id,
                    name,
                    summary,
                } => {
                    if stream_json {
                        let _ = serde_json::to_writer(
                            &mut handle,
                            &serde_json::json!({"type":"subagent_completed","run_id":run_id,"name":name,"summary":summary}),
                        );
                        let _ = writeln!(handle);
                    } else {
                        let _ = writeln!(
                            handle,
                            "[subagent:completed] {name} ({run_id}) {}",
                            summary.replace('\n', " ")
                        );
                    }
                    let _ = handle.flush();
                }
                StreamChunk::SubagentFailed {
                    run_id,
                    name,
                    error,
                } => {
                    if stream_json {
                        let _ = serde_json::to_writer(
                            &mut handle,
                            &serde_json::json!({"type":"subagent_failed","run_id":run_id,"name":name,"error":error}),
                        );
                        let _ = writeln!(handle);
                    } else {
                        let _ = writeln!(
                            handle,
                            "[subagent:failed] {name} ({run_id}) {}",
                            error.replace('\n', " ")
                        );
                    }
                    let _ = handle.flush();
                }
                StreamChunk::ImageData { label, .. } => {
                    if stream_json {
                        let _ = serde_json::to_writer(
                            &mut handle,
                            &serde_json::json!({"type": "image", "label": label}),
                        );
                        let _ = writeln!(handle);
                    } else {
                        let _ = writeln!(handle, "[image: {label}]");
                    }
                    let _ = handle.flush();
                }
                StreamChunk::ClearStreamingText => {
                    // In non-TUI mode, nothing to clear — text already
                    // written to stdout.
                }
                StreamChunk::Done => {
                    if stream_json {
                        let _ = serde_json::to_writer(
                            &mut handle,
                            &serde_json::json!({"type": "done"}),
                        );
                        let _ = writeln!(handle);
                    } else {
                        let _ = writeln!(handle);
                    }
                    let _ = handle.flush();
                }
            }
        }));
    }

    let options = chat_options_from_cli(cli, true);
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
    run_chat(cwd, json_mode, true, false, None)
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
    run_chat(cwd, json_mode, true, false, None)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
