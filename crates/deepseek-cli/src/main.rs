use anyhow::{Result, anyhow};
use chrono::Utc;
use clap::{Args, Parser, Subcommand};
use deepseek_agent::AgentEngine;
use deepseek_core::{
    AppConfig, EventEnvelope, EventKind, Session, SessionBudgets, SessionState, runtime_dir,
};
use deepseek_diff::PatchStore;
use deepseek_index::IndexService;
use deepseek_store::Store;
use deepseek_tools::PluginManager;
use serde::Serialize;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "deepseek")]
#[command(about = "DeepSeek CLI coding agent", long_about = None)]
struct Cli {
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Chat(ChatArgs),
    Autopilot(AutopilotArgs),
    Ask(AskArgs),
    Plan(PromptArg),
    Run,
    Diff,
    Apply(ApplyArgs),
    Index {
        #[command(subcommand)]
        command: IndexCmd,
    },
    Config {
        #[command(subcommand)]
        command: ConfigCmd,
    },
    Plugins {
        #[command(subcommand)]
        command: PluginCmd,
    },
    Clean(CleanArgs),
}

#[derive(Args)]
struct AskArgs {
    prompt: String,
    #[arg(long)]
    tools: bool,
}

#[derive(Args, Default)]
struct ChatArgs {
    #[arg(long)]
    tools: bool,
}

#[derive(Args)]
struct AutopilotArgs {
    prompt: String,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    tools: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    max_think: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    continue_on_error: bool,
    #[arg(long)]
    max_iterations: Option<u64>,
    #[arg(long)]
    duration_seconds: Option<u64>,
    #[arg(long)]
    hours: Option<f64>,
    #[arg(long)]
    forever: bool,
    #[arg(long, default_value_t = 0)]
    sleep_seconds: u64,
    #[arg(long, default_value_t = 2)]
    retry_delay_seconds: u64,
    #[arg(long)]
    stop_file: Option<String>,
    #[arg(long)]
    heartbeat_file: Option<String>,
    #[arg(long, default_value_t = 10)]
    max_consecutive_failures: u64,
}

#[derive(Args)]
struct PromptArg {
    prompt: String,
}

#[derive(Args)]
struct ApplyArgs {
    #[arg(long)]
    patch_id: Option<String>,
    #[arg(long)]
    yes: bool,
}

#[derive(Args)]
struct CleanArgs {
    #[arg(long)]
    dry_run: bool,
}

#[derive(Subcommand)]
enum IndexCmd {
    Build,
    Update,
    Status,
    Query {
        q: String,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
    },
}

#[derive(Subcommand)]
enum ConfigCmd {
    Edit,
    Show,
}

#[derive(Subcommand)]
enum PluginCmd {
    List(PluginListArgs),
    Install(PluginInstallArgs),
    Remove(PluginIdArgs),
    Enable(PluginIdArgs),
    Disable(PluginIdArgs),
    Inspect(PluginIdArgs),
    Run(PluginRunArgs),
}

#[derive(Args)]
struct PluginListArgs {
    #[arg(long)]
    discover: bool,
}

#[derive(Args)]
struct PluginInstallArgs {
    source: String,
}

#[derive(Args)]
struct PluginIdArgs {
    plugin_id: String,
}

#[derive(Args)]
struct PluginRunArgs {
    plugin_id: String,
    command_name: String,
    #[arg(long)]
    input: Option<String>,
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    tools: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    max_think: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cwd = std::env::current_dir()?;
    let command = cli.command.unwrap_or(Commands::Chat(ChatArgs::default()));

    match command {
        Commands::Chat(args) => run_chat(&cwd, cli.json, args.tools),
        Commands::Autopilot(args) => run_autopilot(&cwd, args, cli.json),
        Commands::Ask(args) => {
            let engine = AgentEngine::new(&cwd)?;
            let output = engine.run_once(&args.prompt, args.tools)?;
            if cli.json {
                print_json(&json!({"output": output}))?;
            } else {
                println!("{output}");
            }
            Ok(())
        }
        Commands::Plan(args) => {
            let engine = AgentEngine::new(&cwd)?;
            let plan = engine.plan_only(&args.prompt)?;
            if cli.json {
                print_json(&plan)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&plan)?);
            }
            Ok(())
        }
        Commands::Run => {
            let engine = AgentEngine::new(&cwd)?;
            let output = engine.resume()?;
            if cli.json {
                print_json(&json!({"output": output}))?;
            } else {
                println!("{output}");
            }
            Ok(())
        }
        Commands::Diff => run_diff(&cwd, cli.json),
        Commands::Apply(args) => run_apply(&cwd, args, cli.json),
        Commands::Index { command } => run_index(&cwd, command, cli.json),
        Commands::Config { command } => run_config(&cwd, command, cli.json),
        Commands::Plugins { command } => run_plugins(&cwd, command, cli.json),
        Commands::Clean(args) => run_clean(&cwd, args, cli.json),
    }
}

fn run_autopilot(cwd: &Path, args: AutopilotArgs, json_mode: bool) -> Result<()> {
    if let Some(max_iterations) = args.max_iterations
        && max_iterations == 0
    {
        return Err(anyhow!("--max-iterations must be greater than 0"));
    }
    if args.forever && (args.duration_seconds.is_some() || args.hours.is_some()) {
        return Err(anyhow!(
            "--forever cannot be combined with --duration-seconds or --hours"
        ));
    }
    if args.max_consecutive_failures == 0 {
        return Err(anyhow!("--max-consecutive-failures must be greater than 0"));
    }

    let engine = AgentEngine::new(cwd)?;
    let started = Instant::now();
    let deadline = autopilot_deadline(&args)?;
    let stop_file = args
        .stop_file
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| runtime_dir(cwd).join("autopilot.stop"));
    let heartbeat_file = args
        .heartbeat_file
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| runtime_dir(cwd).join("autopilot.heartbeat.json"));

    if !json_mode {
        let runtime = if args.forever {
            "indefinite".to_string()
        } else if let Some(hours) = args.hours {
            format!("{hours:.2} hours")
        } else if let Some(seconds) = args.duration_seconds {
            format!("{seconds} seconds")
        } else {
            "7200 seconds (default)".to_string()
        };
        println!(
            "autopilot started: tools={} max_think={} runtime={} max_iterations={:?} stop_file={} heartbeat_file={}",
            args.tools,
            args.max_think,
            runtime,
            args.max_iterations,
            stop_file.display(),
            heartbeat_file.display(),
        );
    }

    let mut completed_iterations = 0_u64;
    let mut failed_iterations = 0_u64;
    let mut consecutive_failures = 0_u64;
    let mut last_error: Option<String> = None;
    write_autopilot_heartbeat(
        &heartbeat_file,
        &json!({
            "status": "started",
            "at": Utc::now().to_rfc3339(),
            "completed_iterations": completed_iterations,
            "failed_iterations": failed_iterations,
            "consecutive_failures": consecutive_failures,
            "stop_file": stop_file,
        }),
    )?;
    let stop_reason = loop {
        if let Some(max_iterations) = args.max_iterations
            && completed_iterations + failed_iterations >= max_iterations
        {
            break "max_iterations_reached".to_string();
        }
        if let Some(deadline) = deadline
            && Instant::now() >= deadline
        {
            break "duration_elapsed".to_string();
        }
        if stop_file.exists() {
            break "stop_file_detected".to_string();
        }

        let iteration_no = completed_iterations + failed_iterations + 1;
        if !json_mode {
            println!("autopilot iteration {iteration_no}");
        }

        match engine.run_once_with_mode(&args.prompt, args.tools, args.max_think) {
            Ok(output) => {
                completed_iterations += 1;
                consecutive_failures = 0;
                if !json_mode {
                    println!("{output}");
                }
                if args.sleep_seconds > 0 {
                    thread::sleep(Duration::from_secs(args.sleep_seconds));
                }
            }
            Err(err) => {
                failed_iterations += 1;
                consecutive_failures += 1;
                let err_text = err.to_string();
                last_error = Some(err_text.clone());
                if !json_mode {
                    println!("autopilot iteration failed: {err_text}");
                }
                if !args.continue_on_error {
                    break "stopped_on_error".to_string();
                }
                if consecutive_failures >= args.max_consecutive_failures {
                    break "max_consecutive_failures_reached".to_string();
                }
                if args.retry_delay_seconds > 0 {
                    thread::sleep(Duration::from_secs(args.retry_delay_seconds));
                }
            }
        }
        write_autopilot_heartbeat(
            &heartbeat_file,
            &json!({
                "status": "running",
                "at": Utc::now().to_rfc3339(),
                "completed_iterations": completed_iterations,
                "failed_iterations": failed_iterations,
                "consecutive_failures": consecutive_failures,
                "last_error": last_error,
            }),
        )?;
    };

    let summary = json!({
        "stop_reason": stop_reason,
        "elapsed_seconds": started.elapsed().as_secs(),
        "completed_iterations": completed_iterations,
        "failed_iterations": failed_iterations,
        "consecutive_failures": consecutive_failures,
        "tools": args.tools,
        "max_think": args.max_think,
        "continue_on_error": args.continue_on_error,
        "stop_file": stop_file,
        "heartbeat_file": heartbeat_file,
        "last_error": last_error,
    });
    write_autopilot_heartbeat(
        &heartbeat_file,
        &json!({
            "status": "stopped",
            "at": Utc::now().to_rfc3339(),
            "summary": summary,
        }),
    )?;

    if json_mode {
        print_json(&summary)?;
    } else {
        println!(
            "autopilot stopped: {} (completed={} failed={} elapsed={}s)",
            summary["stop_reason"].as_str().unwrap_or_default(),
            completed_iterations,
            failed_iterations,
            summary["elapsed_seconds"].as_u64().unwrap_or(0)
        );
    }

    let stop_reason = summary["stop_reason"].as_str().unwrap_or_default();
    if stop_reason == "stopped_on_error" || stop_reason == "max_consecutive_failures_reached" {
        return Err(anyhow!(
            "autopilot stopped on error: {}",
            last_error.unwrap_or_else(|| "unknown error".to_string())
        ));
    }

    Ok(())
}

fn autopilot_deadline(args: &AutopilotArgs) -> Result<Option<Instant>> {
    if args.forever {
        return Ok(None);
    }

    let seconds = if let Some(seconds) = args.duration_seconds {
        seconds
    } else if let Some(hours) = args.hours {
        if !(hours.is_finite() && hours > 0.0) {
            return Err(anyhow!("--hours must be a positive finite value"));
        }
        (hours * 3600.0).round() as u64
    } else {
        2 * 3600
    };

    Ok(Some(Instant::now() + Duration::from_secs(seconds.max(1))))
}

fn write_autopilot_heartbeat(path: &Path, payload: &serde_json::Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(payload)?)?;
    Ok(())
}

fn run_chat(cwd: &Path, json_mode: bool, allow_tools: bool) -> Result<()> {
    use std::io::{Write, stdin, stdout};

    let engine = AgentEngine::new(cwd)?;
    let cfg = AppConfig::ensure(cwd)?;
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
        let prompt = line.trim();
        if prompt == "exit" {
            break;
        }
        if prompt.is_empty() {
            continue;
        }
        let output = engine.run_once(prompt, allow_tools)?;
        if json_mode {
            print_json(&json!({"output": output, "router_hint": "auto-max-think"}))?;
        } else {
            println!("[status] router=auto-max-think");
            println!("{output}");
        }
    }
    Ok(())
}

fn run_diff(cwd: &Path, json_mode: bool) -> Result<()> {
    let patches = PatchStore::new(cwd)?.list()?;
    if json_mode {
        print_json(&patches)?;
        return Ok(());
    }
    if patches.is_empty() {
        println!("No staged patches.");
        return Ok(());
    }
    for p in patches {
        println!(
            "patch_id={} applied={} created_at={}",
            p.patch_id, p.applied, p.created_at
        );
        println!("{}", p.unified_diff);
    }
    Ok(())
}

fn run_apply(cwd: &Path, args: ApplyArgs, json_mode: bool) -> Result<()> {
    let store = PatchStore::new(cwd)?;
    let patches = store.list()?;
    let patch = if let Some(id) = args.patch_id {
        let uid = Uuid::parse_str(&id)?;
        patches
            .into_iter()
            .find(|p| p.patch_id == uid)
            .ok_or_else(|| anyhow!("patch_id not found"))?
    } else {
        patches
            .into_iter()
            .last()
            .ok_or_else(|| anyhow!("no staged patch found"))?
    };

    if !args.yes {
        return Err(anyhow!("approval required: pass --yes to apply"));
    }

    let (applied, conflicts) = store.apply(cwd, patch.patch_id)?;
    if json_mode {
        print_json(
            &json!({"patch_id": patch.patch_id, "applied": applied, "conflicts": conflicts}),
        )?;
        return Ok(());
    }
    if applied {
        println!("Applied patch {}", patch.patch_id);
    } else {
        println!("Failed to apply patch {}", patch.patch_id);
        for c in conflicts {
            println!("conflict: {c}");
        }
    }
    Ok(())
}

fn run_index(cwd: &Path, cmd: IndexCmd, json_mode: bool) -> Result<()> {
    let service = IndexService::new(cwd)?;
    let session = Store::new(cwd)?.load_latest_session()?.unwrap_or(Session {
        session_id: Uuid::now_v7(),
        workspace_root: cwd.to_string_lossy().to_string(),
        baseline_commit: None,
        status: SessionState::Idle,
        budgets: SessionBudgets {
            per_turn_seconds: 120,
            max_think_tokens: 8192,
        },
        active_plan_id: None,
    });

    match cmd {
        IndexCmd::Build => {
            let manifest = service.build(&session)?;
            if json_mode {
                print_json(&manifest)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&manifest)?);
            }
        }
        IndexCmd::Update => {
            let manifest = service.update(&session)?;
            if json_mode {
                print_json(&manifest)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&manifest)?);
            }
        }
        IndexCmd::Status => {
            let status = service.status()?;
            if json_mode {
                print_json(&status)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&status)?);
            }
        }
        IndexCmd::Query { q, top_k } => {
            let result = service.query(&q, top_k)?;
            if json_mode {
                print_json(&result)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
        }
    }
    Ok(())
}

fn run_config(cwd: &Path, cmd: ConfigCmd, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;
    let cfg_path = AppConfig::config_path(cwd);

    match cmd {
        ConfigCmd::Show => {
            if json_mode {
                print_json(&cfg)?;
            } else {
                println!("{}", fs::read_to_string(cfg_path)?);
            }
        }
        ConfigCmd::Edit => {
            let editor = std::env::var("EDITOR").unwrap_or_else(|_| default_editor().to_string());
            let status = Command::new(editor).arg(&cfg_path).status()?;
            if !status.success() {
                return Err(anyhow!("editor exited with non-zero status"));
            }
        }
    }
    Ok(())
}

fn run_plugins(cwd: &Path, cmd: PluginCmd, json_mode: bool) -> Result<()> {
    let manager = PluginManager::new(cwd)?;
    match cmd {
        PluginCmd::List(args) => {
            if args.discover {
                let cfg = AppConfig::load(cwd)?;
                let found = manager.discover(&cfg.plugins.search_paths)?;
                if json_mode {
                    print_json(&found)?;
                } else {
                    for p in found {
                        println!(
                            "{} {} {}",
                            p.manifest.id,
                            p.manifest.version,
                            p.root.display()
                        );
                    }
                }
            } else {
                let installed = manager.list()?;
                if json_mode {
                    print_json(&installed)?;
                } else {
                    for p in installed {
                        let state = if p.enabled { "enabled" } else { "disabled" };
                        println!("{} {} ({})", p.manifest.id, p.manifest.version, state);
                    }
                }
            }
        }
        PluginCmd::Install(args) => {
            let info = manager.install(Path::new(&args.source))?;
            append_control_event(
                cwd,
                EventKind::PluginInstalledV1 {
                    plugin_id: info.manifest.id.clone(),
                    version: info.manifest.version.clone(),
                },
            )?;
            if json_mode {
                print_json(&info)?;
            } else {
                println!("installed {} {}", info.manifest.id, info.manifest.version);
            }
        }
        PluginCmd::Remove(args) => {
            manager.remove(&args.plugin_id)?;
            append_control_event(
                cwd,
                EventKind::PluginRemovedV1 {
                    plugin_id: args.plugin_id.clone(),
                },
            )?;
            if json_mode {
                print_json(&json!({"removed": args.plugin_id}))?;
            } else {
                println!("removed {}", args.plugin_id);
            }
        }
        PluginCmd::Enable(args) => {
            manager.enable(&args.plugin_id)?;
            append_control_event(
                cwd,
                EventKind::PluginEnabledV1 {
                    plugin_id: args.plugin_id.clone(),
                },
            )?;
            if json_mode {
                print_json(&json!({"enabled": args.plugin_id}))?;
            } else {
                println!("enabled {}", args.plugin_id);
            }
        }
        PluginCmd::Disable(args) => {
            manager.disable(&args.plugin_id)?;
            append_control_event(
                cwd,
                EventKind::PluginDisabledV1 {
                    plugin_id: args.plugin_id.clone(),
                },
            )?;
            if json_mode {
                print_json(&json!({"disabled": args.plugin_id}))?;
            } else {
                println!("disabled {}", args.plugin_id);
            }
        }
        PluginCmd::Inspect(args) => {
            let info = manager.inspect(&args.plugin_id)?;
            if json_mode {
                print_json(&info)?;
            } else {
                println!(
                    "id={} version={} root={}",
                    info.manifest.id,
                    info.manifest.version,
                    info.root.display()
                );
                println!(
                    "commands={} agents={} skills={} hooks={}",
                    info.commands.len(),
                    info.agents.len(),
                    info.skills.len(),
                    info.hooks.len()
                );
            }
        }
        PluginCmd::Run(args) => {
            let rendered = manager.render_command_prompt(
                &args.plugin_id,
                &args.command_name,
                args.input.as_deref(),
            )?;
            let engine = AgentEngine::new(cwd)?;
            let output = engine.run_once_with_mode(&rendered.prompt, args.tools, args.max_think)?;
            if json_mode {
                print_json(&json!({
                    "plugin_id": rendered.plugin_id,
                    "command_name": rendered.command_name,
                    "source_path": rendered.source_path,
                    "tools": args.tools,
                    "max_think": args.max_think,
                    "output": output
                }))?;
            } else {
                println!(
                    "plugin command {}:{} ({})",
                    rendered.plugin_id,
                    rendered.command_name,
                    rendered.source_path.display()
                );
                println!("{output}");
            }
        }
    }
    Ok(())
}

fn run_clean(cwd: &Path, args: CleanArgs, json_mode: bool) -> Result<()> {
    let candidates = vec![
        cwd.join(".deepseek/patches"),
        cwd.join(".deepseek/observe.log"),
        cwd.join(".deepseek/index/tantivy"),
    ];
    let mut removed = Vec::new();
    for path in candidates {
        if !path.exists() {
            continue;
        }
        if !args.dry_run {
            if path.is_dir() {
                fs::remove_dir_all(&path)?;
            } else {
                fs::remove_file(&path)?;
            }
        }
        removed.push(path.to_string_lossy().to_string());
    }

    if json_mode {
        print_json(&json!({"dry_run": args.dry_run, "removed": removed}))?;
    } else if args.dry_run {
        println!("would remove:\n{}", removed.join("\n"));
    } else {
        println!("removed:\n{}", removed.join("\n"));
    }
    Ok(())
}

fn default_editor() -> &'static str {
    if cfg!(target_os = "windows") {
        "notepad"
    } else {
        "nano"
    }
}

fn print_json<T: Serialize>(value: &T) -> Result<()> {
    println!("{}", serde_json::to_string(value)?);
    Ok(())
}

fn append_control_event(cwd: &Path, kind: EventKind) -> Result<()> {
    let store = Store::new(cwd)?;
    let session = if let Some(existing) = store.load_latest_session()? {
        existing
    } else {
        let session = Session {
            session_id: Uuid::now_v7(),
            workspace_root: cwd.to_string_lossy().to_string(),
            baseline_commit: None,
            status: SessionState::Idle,
            budgets: SessionBudgets {
                per_turn_seconds: 120,
                max_think_tokens: 8192,
            },
            active_plan_id: None,
        };
        store.save_session(&session)?;
        session
    };
    let event = EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: Utc::now(),
        session_id: session.session_id,
        kind,
    };
    store.append_event(&event)?;
    Ok(())
}
