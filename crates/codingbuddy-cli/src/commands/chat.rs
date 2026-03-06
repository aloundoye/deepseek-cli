use anyhow::{Result, anyhow};
use base64::Engine;
use chrono::Utc;
use codingbuddy_agent::{AgentEngine, ChatMode, ChatOptions};
use codingbuddy_chrome::{ChromeSession, ScreenshotFormat};
use codingbuddy_context::ContextManager;
use codingbuddy_core::{
    AppConfig, ApprovedToolCall, EventKind, StreamChunk, ToolCall, ToolHost, runtime_dir,
    stream_chunk_to_event_json,
};
use codingbuddy_mcp::McpManager;
use codingbuddy_memory::{ExportFormat, MemoryManager};
use codingbuddy_policy::PolicyEngine;
use codingbuddy_skills::SkillManager;
use codingbuddy_store::{SessionTodoRecord, Store, SubagentRunRecord};
use codingbuddy_tools::LocalToolHost;
use codingbuddy_ui::{
    KeyBindings, SlashCommand, TuiStreamEvent, TuiTheme, load_keybindings, render_statusline,
    run_tui_shell_with_bindings, slash_command_catalog_entries,
};
use serde_json::json;
use std::collections::{HashMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;
use uuid::Uuid;

// Shared helpers
use crate::context::*;
use crate::output::*;
use crate::util::*;

// CLI types
use crate::{
    Cli, CompactArgs, ConfigCmd, DoctorArgs, ExportArgs, McpCmd, McpGetArgs, McpRemoveArgs,
    MemoryCmd, MemoryEditArgs, MemoryShowArgs, MemorySyncArgs, RewindArgs, RunArgs, SearchArgs,
    SkillRunArgs, SkillsCmd, UsageArgs,
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
use crate::commands::plan::{
    current_plan_payload, handle_plan_slash, plan_state_label, render_plan_notice_lines,
    workflow_phase_label,
};
use crate::commands::search::run_search;
use crate::commands::skills::run_skills;
use crate::commands::status::{current_ui_status, run_context, run_usage};
use crate::commands::tasks::handle_tasks_slash;

mod args;
#[path = "chat_lifecycle.rs"]
mod chat_lifecycle;
mod payloads;
mod slash;
mod tui;
mod web;
pub(crate) use args::is_max_think_selection;
use args::{
    chat_mode_name, format_provider_info, parse_chat_mode_name, parse_commit_message,
    parse_diff_args, parse_stage_args, truncate_inline,
};
use payloads::*;
use slash::*;
use tui::{ChatTuiArgs, run_chat_tui};
use web::fetch_preview_text;
pub(crate) use web::{render_web_fetch_markdown, render_web_search_markdown};

pub(crate) fn run_chat(
    cwd: &Path,
    json_mode: bool,
    json_events: bool,
    allow_tools: bool,
    tui: bool,
    cli: Option<&crate::Cli>,
    initial_session_id: Option<Uuid>,
) -> Result<()> {
    use std::io::{IsTerminal, Write, stdin, stdout};

    let mut cfg = AppConfig::ensure(cwd)?;
    // Run first-time setup wizard once (provider, API key, local ML, privacy)
    if !json_mode {
        match super::setup::maybe_first_time_setup(cwd, &cfg) {
            Ok(true) => cfg = AppConfig::ensure(cwd)?,
            Ok(false) => {}
            Err(e) => eprintln!("setup skipped: {e}"),
        }
    }
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
    let teammate_mode = cli.and_then(|v| v.teammate_mode.clone());
    let repo_root_override = cli.and_then(|v| v.repo.clone());
    let watch_files_enabled = cli.map(|value| value.watch_files).unwrap_or(false);
    let detect_urls = cli.map(|value| value.detect_urls).unwrap_or(false);
    let debug_context = cli.map(|v| v.debug_context).unwrap_or(false)
        || std::env::var("CODINGBUDDY_DEBUG_CONTEXT")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
    let interactive_tty = stdin().is_terminal() && stdout().is_terminal();
    if !json_mode && (tui || cfg.ui.enable_tui) && interactive_tty {
        return run_chat_tui(ChatTuiArgs {
            cwd,
            allow_tools,
            cfg: &cfg,
            initial_force_max_think: force_max_think,
            teammate_mode: teammate_mode.clone(),
            repo_root_override: repo_root_override.clone(),
            debug_context,
            detect_urls,
            watch_files_enabled,
            initial_session_id,
        });
    }
    if tui && !interactive_tty {
        return Err(anyhow!("--tui requires an interactive terminal"));
    }
    let mut last_assistant_response: Option<String> = None;
    let mut additional_dirs = cli.map(|value| value.add_dir.clone()).unwrap_or_default();
    let mut read_only_mode = false;
    let mut active_chat_mode = ChatMode::Code;
    let mut last_watch_digest: Option<u64> = None;
    let mut pending_images: Vec<codingbuddy_core::ImageContent> = vec![];
    let mut selected_session_id = initial_session_id;
    let mut lifecycle_notice_watermarks = HashMap::<Uuid, u64>::new();
    if !json_mode {
        println!("deepseek chat (type 'exit' to quit)");
        println!(
            "model: {} thinking=auto approvals: bash={} edits={} tools={}",
            cfg.llm.active_base_model(),
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
            for notice in poll_session_lifecycle_notices(
                cwd,
                selected_session_id,
                &mut lifecycle_notice_watermarks,
            )? {
                println!("{}", notice.line);
            }
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
        let mut prompt_owned = codingbuddy_ui::expand_at_mentions(raw_prompt);
        if watch_files_enabled
            && let Some((digest, hints)) = scan_watch_comment_payload(cwd)
            && last_watch_digest != Some(digest)
        {
            last_watch_digest = Some(digest);
            prompt_owned.push_str("\n\nAUTO_WATCH_CONTEXT\nDetected comment hints in workspace:\n");
            prompt_owned.push_str(&hints);
            prompt_owned.push_str("\nAUTO_WATCH_CONTEXT_END");
        }
        let prompt = prompt_owned.as_str();

        if let Some(cmd) = SlashCommand::parse(prompt) {
            match cmd {
                SlashCommand::Help => {
                    let payload = slash_help_payload();
                    if json_mode {
                        print_json(&payload)?;
                    } else {
                        println!("{}", render_slash_help(&payload));
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
                        println!("unsupported chat mode: {raw_mode} (use ask|code|context)");
                    }
                }
                SlashCommand::Init => {
                    let manager = MemoryManager::new(cwd)?;
                    let path = manager.ensure_initialized()?;
                    let version_id = manager.sync_memory_version("init")?;
                    append_control_event(
                        cwd,
                        EventKind::MemorySynced {
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
                    if args.is_empty() || args[0].eq_ignore_ascii_case("edit") {
                        // Default: open editor (was "show")
                        run_memory(cwd, MemoryCmd::Edit(MemoryEditArgs {}), json_mode)?;
                    } else if args[0].eq_ignore_ascii_case("show") {
                        run_memory(cwd, MemoryCmd::Show(MemoryShowArgs {}), json_mode)?;
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
                            "model": if force_max_think {
                                cfg.llm.active_reasoner_model()
                            } else {
                                cfg.llm.active_base_model()
                            },
                            "thinking_enabled": force_max_think,
                        }))?;
                    } else if force_max_think {
                        println!(
                            "model mode: thinking-enabled ({})",
                            cfg.llm.active_reasoner_model()
                        );
                    } else {
                        println!(
                            "model mode: auto ({} thinking=on-demand)",
                            cfg.llm.active_base_model()
                        );
                    }
                }
                SlashCommand::Provider(provider) => {
                    if json_mode {
                        let active = cfg.llm.active_provider();
                        if let Some(ref name) = provider {
                            if let Some(p) = cfg.llm.providers.get(name) {
                                print_json(&json!({
                                    "provider": name,
                                    "base_url": p.base_url,
                                    "model": p.models.chat,
                                }))?;
                            } else {
                                print_json(&json!({"error": format!("unknown provider: {name}")}))?;
                            }
                        } else {
                            let names: Vec<_> = cfg.llm.providers.keys().cloned().collect();
                            print_json(&json!({
                                "provider": cfg.llm.provider,
                                "base_url": active.base_url,
                                "model": active.models.chat,
                                "available": names,
                            }))?;
                        }
                    } else {
                        println!("{}", format_provider_info(&cfg, provider));
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
                    if args.is_empty() && !json_mode {
                        // Interactive menu: show numbered server list with transport/status
                        let mcp_mgr = codingbuddy_mcp::McpManager::new(cwd)?;
                        let servers = mcp_mgr.list_servers()?;
                        if servers.is_empty() {
                            println!("No MCP servers configured. Use /mcp add to add one.");
                        } else {
                            println!("MCP Servers:");
                            for (i, server) in servers.iter().enumerate() {
                                let transport = match server.transport {
                                    codingbuddy_mcp::McpTransport::Stdio => "stdio",
                                    codingbuddy_mcp::McpTransport::Http => "http",
                                    codingbuddy_mcp::McpTransport::Sse => "sse",
                                };
                                let status = if server.enabled {
                                    "enabled"
                                } else {
                                    "disabled"
                                };
                                let endpoint = server
                                    .command
                                    .as_deref()
                                    .or(server.url.as_deref())
                                    .unwrap_or("-");
                                println!(
                                    "  {}. {} ({transport}, {status}) {endpoint}",
                                    i + 1,
                                    server.id
                                );
                            }
                            println!(
                                "\nCommands: /mcp list, /mcp get <id>, /mcp add, /mcp remove <id>, /mcp prompt"
                            );
                        }
                    } else if args.is_empty() || args[0].eq_ignore_ascii_case("list") {
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
                        let mcp = codingbuddy_mcp::McpManager::new(cwd)?;
                        if args.len() < 3 {
                            // List available prompts across all servers
                            let servers = mcp.list_servers()?;
                            if json_mode {
                                let prompts: Vec<serde_json::Value> = servers.iter().map(|s| {
                                    json!({"server": s.id, "hint": format!("/mcp prompt {} <name> [args...]", s.id)})
                                }).collect();
                                print_json(&json!({"mcp_prompts": prompts}))?;
                            } else {
                                println!(
                                    "MCP prompts — usage: /mcp prompt <server> <name> [args...]"
                                );
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
                                        map.insert(
                                            k.to_string(),
                                            serde_json::Value::String(v.to_string()),
                                        );
                                    }
                                }
                                serde_json::Value::Object(map)
                            } else {
                                json!({})
                            };
                            let result = codingbuddy_mcp::execute_mcp_stdio_request(
                                &mcp.get_server(server_id)?.ok_or_else(|| {
                                    anyhow::anyhow!("MCP server not found: {server_id}")
                                })?,
                                "prompts/get",
                                json!({"name": prompt_name, "arguments": prompt_args}),
                                3,
                                std::time::Duration::from_secs(10),
                            )?;
                            if json_mode {
                                print_json(&result)?;
                            } else if let Some(messages) = result
                                .pointer("/result/messages")
                                .and_then(|v| v.as_array())
                            {
                                for msg in messages {
                                    let role =
                                        msg.get("role").and_then(|v| v.as_str()).unwrap_or("?");
                                    let text = msg
                                        .pointer("/content/text")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    println!("[{role}] {text}");
                                }
                            } else {
                                println!("{}", serde_json::to_string_pretty(&result)?);
                            }
                        }
                    } else if json_mode {
                        print_json(
                            &json!({"error":"use /mcp list|get <id>|remove <id>|prompt <server> <name>"}),
                        )?;
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
                SlashCommand::Plan(args) => {
                    if args.is_empty() {
                        let store = Store::new(cwd)?;
                        let session = if let Some(session_id) = selected_session_id {
                            store.load_session(session_id)?
                        } else {
                            store.load_latest_session()?
                        };
                        if current_plan_payload(&store, session.as_ref())?.is_some() {
                            let response =
                                handle_plan_slash(cwd, &["show".to_string()], selected_session_id)?;
                            if let Some(session_id) = response.session_switch {
                                selected_session_id = Some(session_id);
                            }
                            if json_mode {
                                print_json(&response.payload)?;
                            } else {
                                println!("{}", response.text);
                            }
                        } else {
                            force_max_think = true;
                            if json_mode {
                                print_json(&json!({"plan_mode": true, "thinking_enabled": true}))?;
                            } else {
                                println!(
                                    "plan mode active; prompts will prefer structured planning with thinking enabled. Use /plan show|approve|reject <feedback> for the persisted review flow."
                                );
                            }
                        }
                    } else {
                        let response = handle_plan_slash(cwd, &args, selected_session_id)?;
                        if let Some(session_id) = response.session_switch {
                            selected_session_id = Some(session_id);
                        }
                        if json_mode {
                            print_json(&response.payload)?;
                        } else {
                            println!("{}", response.text);
                        }
                    }
                }
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
                    codingbuddy_agent::clear_tag_cache();
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
                        codingbuddy_core::EventKind::TurnReverted { turns_dropped: 1 },
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
                SlashCommand::Status => {
                    let status =
                        current_ui_status(cwd, &cfg, force_max_think, selected_session_id)?;
                    if json_mode {
                        print_json(&serde_json::to_value(&status)?)?;
                    } else {
                        println!("{}", render_statusline(&status));
                    }
                }
                SlashCommand::Effort(level) => {
                    let level = level.unwrap_or_else(|| "medium".to_string());
                    let normalized = level.to_ascii_lowercase();
                    force_max_think = matches!(normalized.as_str(), "high" | "max");
                    append_control_event(
                        cwd,
                        EventKind::EffortChanged {
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
                SlashCommand::Tasks(args) => {
                    let response = handle_tasks_slash(cwd, &args, selected_session_id)?;
                    if let Some(session_id) = response.session_switch {
                        selected_session_id = Some(session_id);
                    }
                    if json_mode {
                        print_json(&response.payload)?;
                    } else {
                        println!("{}", response.text);
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
                        std::path::PathBuf::from("~/.codingbuddy/keybindings.json")
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
                        pending_images.push(codingbuddy_core::ImageContent {
                            mime: "image/png".to_string(),
                            base64_data: b64,
                        });
                        if json_mode {
                            print_json(
                                &json!({"pasted_image": true, "size_bytes": img_bytes.len()}),
                            )?;
                        } else {
                            println!(
                                "Image pasted ({} bytes). It will be included in your next prompt.",
                                img_bytes.len()
                            );
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
                        let log_dir = codingbuddy_core::runtime_dir(cwd).join("logs");
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
                        println!(
                            "Configure hooks in .codingbuddy/settings.json under \"hooks\" key."
                        );
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
                        let session_id = Uuid::parse_str(&id)
                            .map_err(|_| anyhow!("invalid session ID: {id}"))?;
                        let payload = session_focus_payload(cwd, session_id)?;
                        selected_session_id = Some(session_id);
                        if json_mode {
                            print_json(&payload)?;
                        } else {
                            println!(
                                "{}",
                                payload["message"]
                                    .as_str()
                                    .unwrap_or("active chat session updated")
                            );
                        }
                    } else {
                        let message = selected_session_id
                            .map(|id| format!("current chat session: {id}"))
                            .unwrap_or_else(|| {
                                "Use 'codingbuddy --continue' or 'codingbuddy --resume <id>'."
                                    .to_string()
                            });
                        if json_mode {
                            print_json(&json!({
                                "session_id": selected_session_id.map(|id| id.to_string()),
                                "message": message,
                            }))?;
                        } else {
                            println!("{message}");
                        }
                    }
                }
                SlashCommand::Stats => {
                    let store = Store::new(cwd)?;
                    let usage = store.usage_summary(None, Some(24))?;
                    let sessions = store.session_history(7)?;
                    if json_mode {
                        print_json(&json!({
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "records": usage.records,
                            "sessions": sessions,
                        }))?;
                    } else {
                        println!("Last 24h usage:");
                        println!("  Input tokens:  {}", usage.input_tokens);
                        println!("  Output tokens: {}", usage.output_tokens);
                        println!("  Records: {}", usage.records);
                        if !sessions.is_empty() {
                            println!("\nSession history (last 7 days):");
                            for s in &sessions {
                                let total_tokens = s.input_tokens + s.output_tokens;
                                // Rough cost estimate: $0.27/M input, $1.10/M output (DeepSeek V3)
                                let cost = (s.input_tokens as f64 * 0.27
                                    + s.output_tokens as f64 * 1.10)
                                    / 1_000_000.0;
                                println!(
                                    "  {} | {}…  | {} turns | {} tokens | ${:.4}",
                                    &s.started_at[..19.min(s.started_at.len())],
                                    &s.session_id[..8.min(s.session_id.len())],
                                    s.turn_count,
                                    total_tokens,
                                    cost
                                );
                            }
                        }
                    }
                }
                SlashCommand::Statusline(args) => {
                    let status =
                        current_ui_status(cwd, &cfg, force_max_think, selected_session_id)?;
                    let rendered = render_statusline(&status);
                    let note = "statusline shortcut is deprecated; use /status for state and settings.json for formatting.";
                    if json_mode {
                        print_json(&json!({
                            "statusline": rendered,
                            "deprecated": true,
                            "note": note,
                            "args": args
                        }))?;
                    } else {
                        println!("{rendered}");
                        println!("{note}");
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
                    let log_dir = codingbuddy_core::runtime_dir(cwd).join("logs");
                    let config_dir = codingbuddy_core::runtime_dir(cwd);
                    if json_mode {
                        print_json(&json!({
                            "log_dir": log_dir.to_string_lossy(),
                            "config_dir": config_dir.to_string_lossy(),
                            "report_url": "https://github.com/anthropics/codingbuddy-cli/issues"
                        }))?;
                    } else {
                        println!("Bug report info:");
                        println!("  Logs: {}", log_dir.display());
                        println!("  Config: {}", config_dir.display());
                        println!("  Report: https://github.com/anthropics/codingbuddy-cli/issues");
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
                    let payload = todos_payload(cwd, selected_session_id, &args)?;
                    if json_mode {
                        print_json(&payload)?;
                    } else {
                        println!("{}", render_todos_payload(&payload));
                    }
                }
                SlashCommand::CommentTodos(args) => {
                    let payload = comment_todos_payload(cwd, &args)?;
                    if json_mode {
                        print_json(&payload)?;
                    } else {
                        println!("{}", render_comment_todos_payload(&payload));
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
                    // Try custom commands from .codingbuddy/commands/
                    let custom_cmds = codingbuddy_skills::load_custom_commands(cwd);
                    if let Some(cmd) = custom_cmds.iter().find(|c| c.name == name) {
                        let rendered = codingbuddy_skills::render_custom_command(
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
                                    teammate_mode: teammate_mode.clone(),
                                    detect_urls,
                                    watch_files: watch_files_enabled,
                                    session_id: selected_session_id,
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
            engine.set_stream_callback(std::sync::Arc::new(|chunk: codingbuddy_core::StreamChunk| {
                use std::io::Write as _;
                let out = std::io::stdout();
                let mut handle = out.lock();

                let val = serde_json::json!({
                    "ts": chrono::Utc::now().to_rfc3339(),
                    // Event stream payload. Exclude big texts or format appropriately.
                    "type": match &chunk {
                        codingbuddy_core::StreamChunk::ContentDelta(_) => "ContentDelta",
                        codingbuddy_core::StreamChunk::ReasoningDelta(_) => "ReasoningDelta",
                        codingbuddy_core::StreamChunk::ToolCallStart { .. } => "ToolCallStart",
                        codingbuddy_core::StreamChunk::ToolCallEnd { .. } => "ToolCallEnd",
                        codingbuddy_core::StreamChunk::ModeTransition { .. } => "ModeTransition",
                        codingbuddy_core::StreamChunk::SubagentSpawned { .. } => "SubagentSpawned",
                        codingbuddy_core::StreamChunk::SubagentCompleted { .. } => "SubagentCompleted",
                        codingbuddy_core::StreamChunk::SubagentFailed { .. } => "SubagentFailed",
                        codingbuddy_core::StreamChunk::ImageData { .. } => "ImageData",
                        codingbuddy_core::StreamChunk::WatchTriggered { .. } => "WatchTriggered",
                        codingbuddy_core::StreamChunk::SecurityWarning { .. } => "SecurityWarning",
                        codingbuddy_core::StreamChunk::UsageUpdate { .. } => "UsageUpdate",
                        codingbuddy_core::StreamChunk::ClearStreamingText => "ClearStreamingText",
                        codingbuddy_core::StreamChunk::SnapshotRecorded { .. } => "SnapshotRecorded",
                        codingbuddy_core::StreamChunk::Error { .. } => "Error",
                        codingbuddy_core::StreamChunk::PhaseTransition { .. } => "PhaseTransition",
                        codingbuddy_core::StreamChunk::ModelChanged { .. } => "ModelChanged",
                        codingbuddy_core::StreamChunk::Done { .. } => "Done",
                    },
                    "payload": match &chunk {
                        codingbuddy_core::StreamChunk::ContentDelta(text) | codingbuddy_core::StreamChunk::ReasoningDelta(text) => {
                            serde_json::json!({ "text": text })
                        },
                        codingbuddy_core::StreamChunk::ToolCallStart { tool_name, args_summary } => serde_json::json!({ "tool_name": tool_name, "args_summary": args_summary }),
                        codingbuddy_core::StreamChunk::ToolCallEnd { tool_name, duration_ms, success, summary } => serde_json::json!({ "tool_name": tool_name, "duration_ms": duration_ms, "success": success, "summary": summary }),
                        codingbuddy_core::StreamChunk::ModeTransition { from, to, reason } => serde_json::json!({ "from": from, "to": to, "reason": reason }),
                        codingbuddy_core::StreamChunk::SubagentSpawned { run_id, name, goal } => serde_json::json!({ "run_id": run_id, "name": name, "goal": goal }),
                        codingbuddy_core::StreamChunk::SubagentCompleted { run_id, name, summary } => serde_json::json!({ "run_id": run_id, "name": name, "summary": summary }),
                        codingbuddy_core::StreamChunk::SubagentFailed { run_id, name, error } => serde_json::json!({ "run_id": run_id, "name": name, "error": error }),
                        codingbuddy_core::StreamChunk::ImageData { label, .. } => serde_json::json!({ "label": label }),
                        codingbuddy_core::StreamChunk::WatchTriggered { digest, comment_count } => serde_json::json!({ "digest": digest, "comment_count": comment_count }),
                        codingbuddy_core::StreamChunk::Error { message, recoverable } => serde_json::json!({ "message": message, "recoverable": recoverable }),
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
            engine.set_stream_callback(std::sync::Arc::new(move |chunk: codingbuddy_core::StreamChunk| {
                match chunk {
                    codingbuddy_core::StreamChunk::ContentDelta(text) => {
                        if let Ok(mut renderer) = md_clone.lock() {
                            renderer.push(&text);
                        }
                    }
                    codingbuddy_core::StreamChunk::ReasoningDelta(_) => {}
                    codingbuddy_core::StreamChunk::ToolCallStart { tool_name, args_summary } => {
                        crate::md_render::print_phase("⚡", &tool_name, &args_summary);
                    }
                    codingbuddy_core::StreamChunk::ToolCallEnd {
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
                    codingbuddy_core::StreamChunk::ModeTransition { from, to, reason } => {
                        use std::io::Write as _;
                        let out = std::io::stdout();
                        let mut handle = out.lock();
                        let _ = writeln!(
                            handle,
                            "  \x1b[35m→\x1b[0m  Mode: {from} → {to} \x1b[90m({reason})\x1b[0m"
                        );
                        let _ = handle.flush();
                    }
                    codingbuddy_core::StreamChunk::SubagentSpawned {
                        run_id: _,
                        name,
                        goal,
                    } => {
                        crate::md_render::print_phase("◉", &format!("Subagent: {name}"), &goal);
                    }
                    codingbuddy_core::StreamChunk::SubagentCompleted {
                        run_id: _,
                        name,
                        summary,
                    } => {
                        crate::md_render::print_phase_done(true, &format!("Subagent: {name}"), &summary.replace('\n', " "));
                    }
                    codingbuddy_core::StreamChunk::SubagentFailed {
                        run_id: _,
                        name,
                        error,
                    } => {
                        crate::md_render::print_phase_done(false, &format!("Subagent: {name}"), &error.replace('\n', " "));
                    }
                    codingbuddy_core::StreamChunk::ImageData { label, .. } => {
                        use std::io::Write as _;
                        let out = std::io::stdout();
                        let mut handle = out.lock();
                        let _ = writeln!(handle, "  \x1b[90m[image: {label}]\x1b[0m");
                        let _ = handle.flush();
                    }
                    codingbuddy_core::StreamChunk::WatchTriggered { comment_count, .. } => {
                        use std::io::Write as _;
                        let out = std::io::stdout();
                        let mut handle = out.lock();
                        let _ = writeln!(handle, "  \x1b[35m◉\x1b[0m  \x1b[1mWatch triggered\x1b[0m \x1b[90m({comment_count} comment(s))\x1b[0m");
                        let _ = handle.flush();
                    }
                    codingbuddy_core::StreamChunk::SecurityWarning { message } => {
                        use std::io::Write as _;
                        let out = std::io::stdout();
                        let mut handle = out.lock();
                        let _ = writeln!(handle, "  \x1b[33m⚠ Security:\x1b[0m {message}");
                        let _ = handle.flush();
                    }
                    codingbuddy_core::StreamChunk::Error { message, .. } => {
                        use std::io::Write as _;
                        let err = std::io::stderr();
                        let mut handle = err.lock();
                        let _ = writeln!(handle, "  \x1b[31m✗ Error:\x1b[0m {message}");
                        let _ = handle.flush();
                    }
                    codingbuddy_core::StreamChunk::PhaseTransition { from, to } => {
                        use std::io::Write as _;
                        let out = std::io::stdout();
                        let mut handle = out.lock();
                        let _ = writeln!(handle, "  \x1b[36m⟳ Phase:\x1b[0m {from} → {to}");
                        let _ = handle.flush();
                    }
                    codingbuddy_core::StreamChunk::ModelChanged { model } => {
                        use std::io::Write as _;
                        let out = std::io::stdout();
                        let mut handle = out.lock();
                        let _ = writeln!(handle, "  \x1b[36m⟳ Model:\x1b[0m {model}");
                        let _ = handle.flush();
                    }
                    codingbuddy_core::StreamChunk::UsageUpdate { .. } => {}
                    codingbuddy_core::StreamChunk::ClearStreamingText => {}
                    codingbuddy_core::StreamChunk::SnapshotRecorded { .. } => {}
                    codingbuddy_core::StreamChunk::Done { .. } => {
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
            crate::md_render::print_role_header("assistant", &cfg.llm.active_base_model());
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
                teammate_mode: teammate_mode.clone(),
                detect_urls,
                watch_files: watch_files_enabled,
                images: images_for_turn,
                session_id: selected_session_id,
                ..Default::default()
            },
        )?;
        if selected_session_id.is_none() {
            selected_session_id = Store::new(cwd)?
                .load_latest_session()?
                .map(|session| session.session_id);
        }
        last_assistant_response = Some(output.clone());
        if !json_mode && !json_events {
            crate::md_render::print_role_footer();
        }
        let ui_status = current_ui_status(cwd, &cfg, force_max_think, selected_session_id)?;
        if json_mode {
            let suggestions = generate_prompt_suggestions(&output);
            print_json(
                &json!({"output": output, "statusline": render_statusline(&ui_status), "suggestions": suggestions}),
            )?;
        } else {
            println!("[status] {}", render_statusline(&ui_status));
            // Show follow-up prompt suggestions
            let suggestions = generate_prompt_suggestions(&output);
            if !suggestions.is_empty() {
                println!("\x1b[2m  suggestions: {}\x1b[0m", suggestions.join(" | "));
            }
            for notice in poll_session_lifecycle_notices(
                cwd,
                selected_session_id,
                &mut lifecycle_notice_watermarks,
            )? {
                println!("{}", notice.line);
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
                        println!(
                            "[watch: auto-trigger {auto_watch_turns}/{MAX_WATCH_AUTO_TURNS} — {comment_count} comment(s)]"
                        );
                    }
                    let mut auto_prompt =
                        "Resolve the following TODO/FIXME/AI comments detected in the workspace:"
                            .to_string();
                    auto_prompt.push_str("\n\nAUTO_WATCH_CONTEXT\n");
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
                            teammate_mode: teammate_mode.clone(),
                            detect_urls,
                            watch_files: true,
                            session_id: selected_session_id,
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

fn resolve_additional_dir(cwd: &Path, raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        cwd.join(path)
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
    use codingbuddy_core::StreamChunk;
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
                    let _ = writeln!(
                        handle,
                        "[tool: {tool_name}] {status} ({duration_ms}ms) {summary}"
                    );
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
                StreamChunk::SecurityWarning { message } => {
                    let _ = writeln!(handle, "[security warning: {message}]");
                    let _ = handle.flush();
                }
                StreamChunk::Error { message, .. } => {
                    let _ = writeln!(handle, "[error: {message}]");
                    let _ = handle.flush();
                }
                StreamChunk::PhaseTransition { from, to } => {
                    let _ = writeln!(handle, "[phase: {from} -> {to}]");
                    let _ = handle.flush();
                }
                StreamChunk::ModelChanged { model } => {
                    let _ = writeln!(handle, "[model changed: {model}]");
                    let _ = handle.flush();
                }
                StreamChunk::UsageUpdate { .. } => {}
                StreamChunk::ClearStreamingText => {
                    // In non-TUI mode, nothing to clear — text already
                    // written to stdout.
                }
                StreamChunk::SnapshotRecorded { .. } => {}
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
                "model": AppConfig::load(cwd)
                    .unwrap_or_default()
                    .llm
                    .active_base_model(),
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
                    "model": AppConfig::load(cwd)
                        .unwrap_or_default()
                        .llm
                        .active_base_model(),
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
    run_chat(
        cwd,
        json_mode,
        false,
        true,
        false,
        None,
        Some(session.session_id),
    )
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
    run_chat(
        cwd,
        json_mode,
        false,
        true,
        false,
        None,
        Some(session.session_id),
    )
}

#[cfg(test)]
mod tests;
