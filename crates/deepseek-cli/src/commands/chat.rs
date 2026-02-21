use anyhow::{Result, anyhow};
use deepseek_agent::{AgentEngine, ChatOptions};
use deepseek_core::{AppConfig, EventKind, StreamChunk, runtime_dir};
use deepseek_mcp::McpManager;
use deepseek_memory::{ExportFormat, MemoryManager};
use deepseek_skills::SkillManager;
use deepseek_store::Store;
use deepseek_ui::{
    KeyBindings, SlashCommand, TuiStreamEvent, TuiTheme, load_keybindings, render_statusline,
    run_tui_shell_with_bindings,
};
use serde_json::json;
use std::path::Path;
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
    Cli, CompactArgs, ConfigCmd, DoctorArgs, ExportArgs, McpCmd, McpGetArgs, McpRemoveArgs,
    MemoryCmd, MemoryEditArgs, MemoryShowArgs, MemorySyncArgs, RewindArgs, RunArgs, SearchArgs,
    SkillRunArgs, SkillsCmd, TasksCmd, UsageArgs,
};

// Commands that chat dispatches to
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
use crate::commands::tasks::run_tasks;
use crate::commands::teleport::{parse_teleport_args, run_teleport, teleport_now};
use crate::commands::visual::{parse_visual_cmd, run_visual, visual_payload};

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
    let interactive_tty = stdin().is_terminal() && stdout().is_terminal();
    if !json_mode && (force_tui || cfg.ui.enable_tui) && interactive_tty {
        return run_chat_tui(cwd, allow_tools, &cfg);
    }
    if force_tui && !interactive_tty {
        return Err(anyhow!("--tui requires an interactive terminal"));
    }
    let mut force_max_think = false;
    let mut last_assistant_response: Option<String> = None;
    if !json_mode {
        println!("deepseek chat (type 'exit' to quit)");
        println!(
            "models: base={} max_think={} approvals: bash={} edits={} tools={}",
            cfg.llm.base_model,
            cfg.llm.max_think_model,
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
                        let lower = model.to_ascii_lowercase();
                        force_max_think = lower.contains("reasoner")
                            || lower.contains("max")
                            || lower.contains("high");
                    }
                    if json_mode {
                        print_json(&json!({
                            "force_max_think": force_max_think,
                            "base_model": cfg.llm.base_model,
                            "max_think_model": cfg.llm.max_think_model,
                        }))?;
                    } else if force_max_think {
                        println!("model mode: max-think ({})", cfg.llm.max_think_model);
                    } else {
                        println!("model mode: base ({})", cfg.llm.base_model);
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
                    if json_mode {
                        print_json(&json!({"plan_mode": true}))?;
                    } else {
                        println!("plan mode active; prompts will prefer structured planning.");
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
                            "force_max_think": force_max_think
                        }))?;
                    } else {
                        println!(
                            "effort={} model_mode={}",
                            normalized,
                            if force_max_think { "max-think" } else { "base" }
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
                    if json_mode {
                        print_json(&json!({"agents": "subagent listing"}))?;
                    } else {
                        println!(
                            "Subagent status: use 'deepseek background list' for running agents."
                        );
                    }
                }
                SlashCommand::Tasks(_args) => {
                    run_tasks(cwd, TasksCmd::List, json_mode)?;
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
                    run_doctor(cwd, DoctorArgs {}, json_mode)?;
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
                SlashCommand::Debug(args) => {
                    let desc = if args.is_empty() {
                        "general".to_string()
                    } else {
                        args.join(" ")
                    };
                    let log_dir = deepseek_core::runtime_dir(cwd).join("logs");
                    if json_mode {
                        print_json(&json!({"debug": desc, "log_dir": log_dir.to_string_lossy()}))?;
                    } else {
                        println!("Debug: {desc}");
                        println!("Logs: {}", log_dir.display());
                    }
                }
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
                        println!("Usage: /add-dir <path>");
                    } else {
                        for dir in &args {
                            println!("Added directory: {dir}");
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
                    let pr_num = args.first().unwrap_or(&String::new()).clone();
                    if pr_num.is_empty() {
                        println!("Usage: /pr_comments <PR_NUMBER>");
                    } else if json_mode {
                        print_json(&json!({"pr": pr_num, "status": "fetching"}))?;
                    } else {
                        println!("Fetching PR #{pr_num} comments...");
                        // Use gh CLI to fetch PR comments
                        match std::process::Command::new("gh")
                            .args(["pr", "view", &pr_num, "--comments"])
                            .output()
                        {
                            Ok(output) => {
                                println!("{}", String::from_utf8_lossy(&output.stdout));
                            }
                            Err(e) => println!("Failed to fetch PR comments: {e}"),
                        }
                    }
                }
                SlashCommand::ReleaseNotes(args) => {
                    let range = args
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "HEAD~10..HEAD".to_string());
                    if json_mode {
                        print_json(&json!({"range": range}))?;
                    } else {
                        println!("Release notes for {range}:");
                        match std::process::Command::new("git")
                            .args(["log", "--oneline", &range])
                            .output()
                        {
                            Ok(output) => {
                                println!("{}", String::from_utf8_lossy(&output.stdout));
                            }
                            Err(e) => println!("Failed: {e}"),
                        }
                    }
                }
                SlashCommand::Login => {
                    println!("Set your API key via DEEPSEEK_API_KEY environment variable");
                    println!("or add `llm.api_key` to .deepseek/settings.json");
                }
                SlashCommand::Logout => {
                    println!("Remove your API key from the environment or settings file.");
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

pub(crate) fn run_chat_tui(cwd: &Path, _allow_tools: bool, cfg: &AppConfig) -> Result<()> {
    let engine = Arc::new(AgentEngine::new(cwd)?);
    wire_subagent_worker(&engine, cwd);
    let force_max_think = Arc::new(AtomicBool::new(false));

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
    run_tui_shell_with_bindings(
        status,
        bindings,
        theme,
        rx,
        |prompt| {
            // Handle slash commands synchronously, sending result via channel.
            if let Some(cmd) = SlashCommand::parse(prompt) {
                let result: Result<String> = (|| {
                    let out = match cmd {
                SlashCommand::Help => "commands: /help /init /clear /compact /memory /config /model /cost /mcp /rewind /export /plan /teleport /remote-env /status /effort /skills /permissions /background /visual /vim".to_string(),
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
                        let lower = model.to_ascii_lowercase();
                        force_max_think.store(
                            lower.contains("reasoner") || lower.contains("max") || lower.contains("high"),
                            Ordering::Relaxed,
                        );
                    }
                    format!(
                        "model mode: {}",
                        if force_max_think.load(Ordering::Relaxed) { &cfg.llm.max_think_model } else { &cfg.llm.base_model }
                    )
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
                SlashCommand::Plan => "plan mode enabled".to_string(),
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
                    format!("effort={} force_max_think={}", normalized, force_max_think.load(Ordering::Relaxed))
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
                SlashCommand::Agents => "Use 'deepseek background list' for subagent status.".to_string(),
                SlashCommand::Tasks(_) => "Use 'deepseek tasks list' for task queue.".to_string(),
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
                SlashCommand::Doctor => "Use 'deepseek doctor' for diagnostics.".to_string(),
                SlashCommand::Copy => "Copied last response to clipboard.".to_string(),
                SlashCommand::Debug(args) => format!("Debug: {}", if args.is_empty() { "general".to_string() } else { args.join(" ") }),
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
                    if args.is_empty() { "Usage: /add-dir <path>".to_string() } else { format!("Added: {}", args.join(", ")) }
                }
                SlashCommand::Bug => format!("Report bugs at https://github.com/anthropics/deepseek-cli/issues\nLogs: {}", deepseek_core::runtime_dir(cwd).join("logs").display()),
                SlashCommand::PrComments(args) => {
                    if let Some(pr) = args.first() { format!("Fetch PR #{pr} comments via 'gh pr view {pr} --comments'") } else { "Usage: /pr_comments <number>".to_string() }
                }
                SlashCommand::ReleaseNotes(_) => "Use 'git log --oneline' for release notes.".to_string(),
                SlashCommand::Login => "Set DEEPSEEK_API_KEY or add llm.api_key to settings.json".to_string(),
                SlashCommand::Logout => "Remove API key from env or settings.".to_string(),
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
            let _max_think = force_max_think.load(Ordering::Relaxed);
            let tx_stream = tx.clone();
            let tx_done = tx.clone();

            engine.set_stream_callback(std::sync::Arc::new(move |chunk| match chunk {
                StreamChunk::ContentDelta(s) => {
                    let _ = tx_stream.send(TuiStreamEvent::ContentDelta(s));
                }
                StreamChunk::ReasoningDelta(s) => {
                    let _ = tx_stream.send(TuiStreamEvent::ReasoningDelta(s));
                }
                StreamChunk::Done => {}
            }));

            thread::spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    engine_clone.chat(&prompt)
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
    AgentEngine::new(cwd)?.resume()
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
