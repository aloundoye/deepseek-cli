use anyhow::{Result, anyhow};
use chrono::Utc;
use clap::{Args, Parser, Subcommand};
use deepseek_agent::AgentEngine;
use deepseek_core::{
    AppConfig, EventEnvelope, EventKind, Session, SessionBudgets, SessionState, runtime_dir,
};
use deepseek_diff::PatchStore;
use deepseek_index::IndexService;
use deepseek_mcp::{McpManager, McpServer, McpTransport};
use deepseek_memory::{ExportFormat, MemoryManager};
use deepseek_skills::SkillManager;
use deepseek_store::{AutopilotRunRecord, BackgroundJobRecord, ReplayCassetteRecord, Store};
use deepseek_tools::PluginManager;
use deepseek_ui::{SlashCommand, UiStatus, render_statusline, run_tui_shell};
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
    Run(RunArgs),
    Diff,
    Apply(ApplyArgs),
    Profile(ProfileArgs),
    Rewind(RewindArgs),
    Export(ExportArgs),
    Memory {
        #[command(subcommand)]
        command: MemoryCmd,
    },
    Mcp {
        #[command(subcommand)]
        command: McpCmd,
    },
    Git {
        #[command(subcommand)]
        command: GitCmd,
    },
    Skills {
        #[command(subcommand)]
        command: SkillsCmd,
    },
    Replay {
        #[command(subcommand)]
        command: ReplayCmd,
    },
    Background {
        #[command(subcommand)]
        command: BackgroundCmd,
    },
    Teleport(TeleportArgs),
    RemoteEnv {
        #[command(subcommand)]
        command: RemoteEnvCmd,
    },
    Status,
    Usage(UsageArgs),
    Compact(CompactArgs),
    Doctor(DoctorArgs),
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
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    tui: bool,
}

#[derive(Args)]
struct AutopilotArgs {
    #[command(subcommand)]
    command: Option<AutopilotCmd>,
    prompt: Option<String>,
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

#[derive(Subcommand)]
enum AutopilotCmd {
    Status(AutopilotStatusArgs),
    Stop(AutopilotStopArgs),
    Resume(AutopilotResumeArgs),
}

#[derive(Args)]
struct AutopilotStatusArgs {
    #[arg(long)]
    run_id: Option<String>,
}

#[derive(Args)]
struct AutopilotStopArgs {
    #[arg(long)]
    run_id: Option<String>,
}

#[derive(Args)]
struct AutopilotResumeArgs {
    #[arg(long)]
    run_id: Option<String>,
}

#[derive(Args)]
struct PromptArg {
    prompt: String,
}

#[derive(Args, Default)]
struct RunArgs {
    session_id: Option<String>,
}

#[derive(Args, Default)]
struct ProfileArgs {}

#[derive(Args)]
struct ApplyArgs {
    #[arg(long)]
    patch_id: Option<String>,
    #[arg(long)]
    yes: bool,
}

#[derive(Args, Default)]
struct RewindArgs {
    #[arg(long)]
    to_checkpoint: Option<String>,
    #[arg(long)]
    yes: bool,
}

#[derive(Args, Default)]
struct ExportArgs {
    #[arg(long)]
    session: Option<String>,
    #[arg(long, default_value = "json")]
    format: String,
    #[arg(long)]
    output: Option<String>,
}

#[derive(Args)]
struct CleanArgs {
    #[arg(long)]
    dry_run: bool,
}

#[derive(Args, Default)]
struct UsageArgs {
    #[arg(long)]
    session: bool,
    #[arg(long)]
    day: bool,
}

#[derive(Args, Default)]
struct CompactArgs {
    #[arg(long)]
    from_turn: Option<u64>,
    #[arg(long)]
    yes: bool,
}

#[derive(Args, Default)]
struct DoctorArgs {}

#[derive(Subcommand)]
enum MemoryCmd {
    Show(MemoryShowArgs),
    Edit(MemoryEditArgs),
    Sync(MemorySyncArgs),
}

#[derive(Args, Default)]
struct MemoryShowArgs {}

#[derive(Args, Default)]
struct MemoryEditArgs {}

#[derive(Args, Default)]
struct MemorySyncArgs {
    #[arg(long)]
    note: Option<String>,
}

#[derive(Subcommand)]
enum McpCmd {
    Add(McpAddArgs),
    List,
    Get(McpGetArgs),
    Remove(McpRemoveArgs),
}

#[derive(clap::ValueEnum, Clone, Copy)]
enum McpTransportArg {
    Stdio,
    Http,
}

impl McpTransportArg {
    fn into_transport(self) -> McpTransport {
        match self {
            Self::Stdio => McpTransport::Stdio,
            Self::Http => McpTransport::Http,
        }
    }
}

#[derive(Args)]
struct McpAddArgs {
    id: String,
    #[arg(long)]
    name: Option<String>,
    #[arg(long, value_enum, default_value_t = McpTransportArg::Stdio)]
    transport: McpTransportArg,
    #[arg(long)]
    command: Option<String>,
    #[arg(long = "arg")]
    args: Vec<String>,
    #[arg(long)]
    url: Option<String>,
    #[arg(long)]
    metadata: Option<String>,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    enabled: bool,
}

#[derive(Args)]
struct McpGetArgs {
    server_id: String,
}

#[derive(Args)]
struct McpRemoveArgs {
    server_id: String,
}

#[derive(Subcommand)]
enum GitCmd {
    Status,
    History(GitHistoryArgs),
    Branch,
    Checkout(GitCheckoutArgs),
    Commit(GitCommitArgs),
    Pr(GitPrArgs),
    Resolve(GitResolveArgs),
}

#[derive(Args)]
struct GitHistoryArgs {
    #[arg(long, default_value_t = 20)]
    limit: usize,
}

#[derive(Args)]
struct GitCheckoutArgs {
    target: String,
}

#[derive(Args)]
struct GitCommitArgs {
    #[arg(long)]
    message: String,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    all: bool,
}

#[derive(Args)]
struct GitPrArgs {
    #[arg(long)]
    title: Option<String>,
    #[arg(long)]
    body: Option<String>,
    #[arg(long)]
    base: Option<String>,
    #[arg(long)]
    head: Option<String>,
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    dry_run: bool,
}

#[derive(Args)]
struct GitResolveArgs {
    #[arg(long)]
    file: Option<String>,
    #[arg(long, default_value = "list")]
    strategy: String,
}

#[derive(Subcommand)]
enum SkillsCmd {
    List,
    Install(SkillInstallArgs),
    Remove(SkillRemoveArgs),
    Run(SkillRunArgs),
    Reload,
}

#[derive(Args)]
struct SkillInstallArgs {
    source: String,
}

#[derive(Args)]
struct SkillRemoveArgs {
    skill_id: String,
}

#[derive(Args)]
struct SkillRunArgs {
    skill_id: String,
    #[arg(long)]
    input: Option<String>,
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    execute: bool,
}

#[derive(Subcommand)]
enum ReplayCmd {
    Run(ReplayRunArgs),
}

#[derive(Args)]
struct ReplayRunArgs {
    #[arg(long)]
    session_id: String,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    deterministic: bool,
}

#[derive(Subcommand)]
enum BackgroundCmd {
    List,
    Attach(BackgroundAttachArgs),
    Stop(BackgroundStopArgs),
}

#[derive(Args)]
struct BackgroundAttachArgs {
    job_id: String,
}

#[derive(Args)]
struct BackgroundStopArgs {
    job_id: String,
}

#[derive(Args, Default)]
struct TeleportArgs {
    #[arg(long)]
    session_id: Option<String>,
    #[arg(long)]
    output: Option<String>,
    #[arg(long)]
    import: Option<String>,
}

#[derive(Subcommand)]
enum RemoteEnvCmd {
    List,
    Add(RemoteEnvAddArgs),
    Remove(RemoteEnvRemoveArgs),
    Check(RemoteEnvCheckArgs),
}

#[derive(Args)]
struct RemoteEnvAddArgs {
    name: String,
    endpoint: String,
    #[arg(long, default_value = "token")]
    auth_mode: String,
}

#[derive(Args)]
struct RemoteEnvRemoveArgs {
    profile_id: String,
}

#[derive(Args)]
struct RemoteEnvCheckArgs {
    profile_id: String,
}

#[derive(Subcommand)]
enum IndexCmd {
    Build,
    Update,
    Status,
    Watch {
        #[arg(long, default_value_t = 1)]
        events: usize,
        #[arg(long, default_value_t = 30)]
        timeout_seconds: u64,
    },
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
    Catalog,
    Search(PluginSearchArgs),
    Verify(PluginIdArgs),
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
struct PluginSearchArgs {
    query: String,
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
        Commands::Chat(args) => run_chat(&cwd, cli.json, args.tools, args.tui),
        Commands::Autopilot(args) => run_autopilot_cmd(&cwd, args, cli.json),
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
        Commands::Run(args) => {
            let output = run_resume(&cwd, args)?;
            if cli.json {
                print_json(&json!({"output": output}))?;
            } else {
                println!("{output}");
            }
            Ok(())
        }
        Commands::Diff => run_diff(&cwd, cli.json),
        Commands::Apply(args) => run_apply(&cwd, args, cli.json),
        Commands::Profile(args) => run_profile(&cwd, args, cli.json),
        Commands::Rewind(args) => run_rewind(&cwd, args, cli.json),
        Commands::Export(args) => run_export(&cwd, args, cli.json),
        Commands::Memory { command } => run_memory(&cwd, command, cli.json),
        Commands::Mcp { command } => run_mcp(&cwd, command, cli.json),
        Commands::Git { command } => run_git(&cwd, command, cli.json),
        Commands::Skills { command } => run_skills(&cwd, command, cli.json),
        Commands::Replay { command } => run_replay(&cwd, command, cli.json),
        Commands::Background { command } => run_background(&cwd, command, cli.json),
        Commands::Teleport(args) => run_teleport(&cwd, args, cli.json),
        Commands::RemoteEnv { command } => run_remote_env(&cwd, command, cli.json),
        Commands::Status => run_status(&cwd, cli.json),
        Commands::Usage(args) => run_usage(&cwd, args, cli.json),
        Commands::Compact(args) => run_compact(&cwd, args, cli.json),
        Commands::Doctor(args) => run_doctor(&cwd, args, cli.json),
        Commands::Index { command } => run_index(&cwd, command, cli.json),
        Commands::Config { command } => run_config(&cwd, command, cli.json),
        Commands::Plugins { command } => run_plugins(&cwd, command, cli.json),
        Commands::Clean(args) => run_clean(&cwd, args, cli.json),
    }
}

fn run_autopilot_cmd(cwd: &Path, args: AutopilotArgs, json_mode: bool) -> Result<()> {
    match args.command {
        Some(AutopilotCmd::Status(status)) => run_autopilot_status(cwd, status, json_mode),
        Some(AutopilotCmd::Stop(stop)) => run_autopilot_stop(cwd, stop, json_mode),
        Some(AutopilotCmd::Resume(resume)) => run_autopilot_resume(cwd, resume, json_mode),
        None => {
            let prompt = args.prompt.ok_or_else(|| {
                anyhow!("missing autopilot prompt; use `deepseek autopilot \"<prompt>\"`")
            })?;
            run_autopilot(
                cwd,
                AutopilotStartArgs {
                    prompt,
                    tools: args.tools,
                    max_think: args.max_think,
                    continue_on_error: args.continue_on_error,
                    max_iterations: args.max_iterations,
                    duration_seconds: args.duration_seconds,
                    hours: args.hours,
                    forever: args.forever,
                    sleep_seconds: args.sleep_seconds,
                    retry_delay_seconds: args.retry_delay_seconds,
                    stop_file: args.stop_file,
                    heartbeat_file: args.heartbeat_file,
                    max_consecutive_failures: args.max_consecutive_failures,
                },
                json_mode,
            )
        }
    }
}

struct AutopilotStartArgs {
    prompt: String,
    tools: bool,
    max_think: bool,
    continue_on_error: bool,
    max_iterations: Option<u64>,
    duration_seconds: Option<u64>,
    hours: Option<f64>,
    forever: bool,
    sleep_seconds: u64,
    retry_delay_seconds: u64,
    stop_file: Option<String>,
    heartbeat_file: Option<String>,
    max_consecutive_failures: u64,
}

fn run_autopilot(cwd: &Path, args: AutopilotStartArgs, json_mode: bool) -> Result<()> {
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
    let store = Store::new(cwd)?;
    let session = ensure_session_record(cwd, &store)?;
    let run_id = Uuid::now_v7();
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

    let started_at = Utc::now().to_rfc3339();
    store.upsert_autopilot_run(&AutopilotRunRecord {
        run_id,
        session_id: session.session_id,
        prompt: args.prompt.clone(),
        status: "running".to_string(),
        stop_reason: None,
        completed_iterations: 0,
        failed_iterations: 0,
        consecutive_failures: 0,
        last_error: None,
        stop_file: stop_file.to_string_lossy().to_string(),
        heartbeat_file: heartbeat_file.to_string_lossy().to_string(),
        tools: args.tools,
        max_think: args.max_think,
        started_at: started_at.clone(),
        updated_at: started_at.clone(),
    })?;
    store.upsert_background_job(&BackgroundJobRecord {
        job_id: run_id,
        kind: "autopilot".to_string(),
        reference: run_id.to_string(),
        status: "running".to_string(),
        metadata_json: serde_json::json!({
            "prompt": args.prompt.clone(),
            "tools": args.tools,
            "max_think": args.max_think
        })
        .to_string(),
        started_at: started_at.clone(),
        updated_at: started_at.clone(),
    })?;
    store.append_event(&EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: Utc::now(),
        session_id: session.session_id,
        kind: EventKind::AutopilotRunStartedV1 {
            run_id,
            prompt: args.prompt.clone(),
        },
    })?;
    store.append_event(&EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: Utc::now(),
        session_id: session.session_id,
        kind: EventKind::BackgroundJobStartedV1 {
            job_id: run_id,
            kind: "autopilot".to_string(),
            reference: run_id.to_string(),
        },
    })?;

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
            "run_id": run_id,
            "status": "started",
            "at": Utc::now().to_rfc3339(),
            "completed_iterations": completed_iterations,
            "failed_iterations": failed_iterations,
            "consecutive_failures": consecutive_failures,
            "stop_file": stop_file,
        }),
    )?;

    store.append_event(&EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: Utc::now(),
        session_id: session.session_id,
        kind: EventKind::AutopilotRunHeartbeatV1 {
            run_id,
            completed_iterations,
            failed_iterations,
            consecutive_failures,
            last_error: last_error.clone(),
        },
    })?;

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
                "run_id": run_id,
                "status": "running",
                "at": Utc::now().to_rfc3339(),
                "completed_iterations": completed_iterations,
                "failed_iterations": failed_iterations,
                "consecutive_failures": consecutive_failures,
                "last_error": last_error,
            }),
        )?;
        store.upsert_autopilot_run(&AutopilotRunRecord {
            run_id,
            session_id: session.session_id,
            prompt: args.prompt.clone(),
            status: "running".to_string(),
            stop_reason: None,
            completed_iterations,
            failed_iterations,
            consecutive_failures,
            last_error: last_error.clone(),
            stop_file: stop_file.to_string_lossy().to_string(),
            heartbeat_file: heartbeat_file.to_string_lossy().to_string(),
            tools: args.tools,
            max_think: args.max_think,
            started_at: started_at.clone(),
            updated_at: Utc::now().to_rfc3339(),
        })?;
        store.append_event(&EventEnvelope {
            seq_no: store.next_seq_no(session.session_id)?,
            at: Utc::now(),
            session_id: session.session_id,
            kind: EventKind::AutopilotRunHeartbeatV1 {
                run_id,
                completed_iterations,
                failed_iterations,
                consecutive_failures,
                last_error: last_error.clone(),
            },
        })?;
    };

    let summary = json!({
        "run_id": run_id,
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
            "run_id": run_id,
            "status": "stopped",
            "at": Utc::now().to_rfc3339(),
            "summary": summary,
        }),
    )?;
    store.upsert_autopilot_run(&AutopilotRunRecord {
        run_id,
        session_id: session.session_id,
        prompt: args.prompt,
        status: "stopped".to_string(),
        stop_reason: Some(stop_reason.clone()),
        completed_iterations,
        failed_iterations,
        consecutive_failures,
        last_error: last_error.clone(),
        stop_file: stop_file.to_string_lossy().to_string(),
        heartbeat_file: heartbeat_file.to_string_lossy().to_string(),
        tools: args.tools,
        max_think: args.max_think,
        started_at: started_at.clone(),
        updated_at: Utc::now().to_rfc3339(),
    })?;
    store.upsert_background_job(&BackgroundJobRecord {
        job_id: run_id,
        kind: "autopilot".to_string(),
        reference: run_id.to_string(),
        status: "stopped".to_string(),
        metadata_json: serde_json::json!({
            "stop_reason": stop_reason,
            "completed_iterations": completed_iterations,
            "failed_iterations": failed_iterations,
            "last_error": last_error.clone(),
        })
        .to_string(),
        started_at: started_at.clone(),
        updated_at: Utc::now().to_rfc3339(),
    })?;
    store.append_event(&EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: Utc::now(),
        session_id: session.session_id,
        kind: EventKind::AutopilotRunStoppedV1 {
            run_id,
            stop_reason: stop_reason.clone(),
            completed_iterations,
            failed_iterations,
        },
    })?;
    store.append_event(&EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: Utc::now(),
        session_id: session.session_id,
        kind: EventKind::BackgroundJobStoppedV1 {
            job_id: run_id,
            reason: stop_reason.clone(),
        },
    })?;

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

fn run_autopilot_status(cwd: &Path, args: AutopilotStatusArgs, json_mode: bool) -> Result<()> {
    let store = Store::new(cwd)?;
    let Some(run) = find_autopilot_run(&store, args.run_id.as_deref())? else {
        let payload = json!({
            "status": "none",
            "run_id": null,
            "session_id": null,
            "completed_iterations": 0,
            "failed_iterations": 0,
            "consecutive_failures": 0,
        });
        if json_mode {
            print_json(&payload)?;
        } else {
            println!("no autopilot runs found");
        }
        return Ok(());
    };
    let heartbeat = if run.heartbeat_file.is_empty() {
        None
    } else {
        fs::read_to_string(&run.heartbeat_file)
            .ok()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
    };
    let payload = json!({
        "run_id": run.run_id,
        "session_id": run.session_id,
        "status": run.status,
        "stop_reason": run.stop_reason,
        "completed_iterations": run.completed_iterations,
        "failed_iterations": run.failed_iterations,
        "consecutive_failures": run.consecutive_failures,
        "last_error": run.last_error,
        "stop_file": run.stop_file,
        "heartbeat_file": run.heartbeat_file,
        "tools": run.tools,
        "max_think": run.max_think,
        "heartbeat": heartbeat,
    });

    if json_mode {
        print_json(&payload)?;
    } else {
        println!(
            "run={} status={} completed={} failed={} consecutive_failures={}",
            run.run_id,
            run.status,
            run.completed_iterations,
            run.failed_iterations,
            run.consecutive_failures
        );
        if let Some(reason) = run.stop_reason {
            println!("stop_reason={reason}");
        }
        if let Some(err) = run.last_error {
            println!("last_error={err}");
        }
        println!("stop_file={}", run.stop_file);
        println!("heartbeat_file={}", run.heartbeat_file);
    }
    Ok(())
}

fn run_autopilot_stop(cwd: &Path, args: AutopilotStopArgs, json_mode: bool) -> Result<()> {
    let store = Store::new(cwd)?;
    let run = find_autopilot_run(&store, args.run_id.as_deref())?
        .ok_or_else(|| anyhow!("no autopilot runs found"))?;
    let stop_path = if run.stop_file.trim().is_empty() {
        runtime_dir(cwd).join("autopilot.stop")
    } else {
        PathBuf::from(run.stop_file.clone())
    };
    if let Some(parent) = stop_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &stop_path,
        format!("stop requested at {}\n", Utc::now().to_rfc3339()),
    )?;
    if json_mode {
        print_json(&json!({
            "run_id": run.run_id,
            "stop_requested": true,
            "stop_file": stop_path,
        }))?;
    } else {
        println!(
            "stop requested for run {} via {}",
            run.run_id,
            stop_path.display()
        );
    }
    Ok(())
}

fn run_autopilot_resume(cwd: &Path, args: AutopilotResumeArgs, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;
    let store = Store::new(cwd)?;
    let run = find_autopilot_run(&store, args.run_id.as_deref())?
        .ok_or_else(|| anyhow!("no autopilot runs found"))?;
    if run.status == "running" {
        return Err(anyhow!("autopilot run is already marked as running"));
    }

    run_autopilot(
        cwd,
        AutopilotStartArgs {
            prompt: run.prompt.clone(),
            tools: run.tools,
            max_think: run.max_think,
            continue_on_error: true,
            max_iterations: None,
            duration_seconds: None,
            hours: None,
            forever: false,
            sleep_seconds: 0,
            retry_delay_seconds: 2,
            stop_file: if run.stop_file.trim().is_empty() {
                None
            } else {
                Some(run.stop_file.clone())
            },
            heartbeat_file: if run.heartbeat_file.trim().is_empty() {
                None
            } else {
                Some(run.heartbeat_file.clone())
            },
            max_consecutive_failures: cfg.autopilot.default_max_consecutive_failures.max(1),
        },
        json_mode,
    )
}

fn find_autopilot_run(store: &Store, run_id: Option<&str>) -> Result<Option<AutopilotRunRecord>> {
    if let Some(run_id) = run_id {
        let uid = Uuid::parse_str(run_id)?;
        return store.load_autopilot_run(uid);
    }
    store.load_latest_autopilot_run()
}

fn autopilot_deadline(args: &AutopilotStartArgs) -> Result<Option<Instant>> {
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

fn run_chat(cwd: &Path, json_mode: bool, allow_tools: bool, force_tui: bool) -> Result<()> {
    use std::io::{Write, stdin, stdout};

    let engine = AgentEngine::new(cwd)?;
    let cfg = AppConfig::ensure(cwd)?;
    if !json_mode && (force_tui || cfg.ui.enable_tui) {
        return run_chat_tui(cwd, allow_tools, &cfg);
    }
    let mut force_max_think = false;
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
                SlashCommand::Teleport => {
                    run_teleport(
                        cwd,
                        TeleportArgs {
                            session_id: None,
                            output: None,
                            import: None,
                        },
                        json_mode,
                    )?;
                }
                SlashCommand::RemoteEnv => {
                    run_remote_env(cwd, RemoteEnvCmd::List, json_mode)?;
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
                SlashCommand::Unknown { name, .. } => {
                    if json_mode {
                        print_json(&json!({"error": format!("unknown slash command: /{name}")}))?;
                    } else {
                        println!("unknown slash command: /{name}");
                    }
                }
            }
            continue;
        }

        let output = engine.run_once_with_mode(prompt, allow_tools, force_max_think)?;
        let ui_status = current_ui_status(cwd, &cfg, force_max_think)?;
        if json_mode {
            print_json(&json!({"output": output, "statusline": render_statusline(&ui_status)}))?;
        } else {
            println!("[status] {}", render_statusline(&ui_status));
            println!("{output}");
        }
    }
    Ok(())
}

fn run_chat_tui(cwd: &Path, allow_tools: bool, cfg: &AppConfig) -> Result<()> {
    let engine = AgentEngine::new(cwd)?;
    let mut force_max_think = false;
    let status = current_ui_status(cwd, cfg, force_max_think)?;
    run_tui_shell(status, |prompt| {
        if let Some(cmd) = SlashCommand::parse(prompt) {
            let out = match cmd {
                SlashCommand::Help => "commands: /help /init /clear /compact /memory /config /model /cost /mcp /rewind /export /plan /teleport /remote-env /status /effort".to_string(),
                SlashCommand::Init => {
                    let manager = MemoryManager::new(cwd)?;
                    let path = manager.ensure_initialized()?;
                    format!("initialized memory at {}", path.display())
                }
                SlashCommand::Clear => "cleared".to_string(),
                SlashCommand::Compact => "use /compact in non-TUI or run `deepseek compact --yes`".to_string(),
                SlashCommand::Memory(_) => MemoryManager::new(cwd)?.read_memory()?,
                SlashCommand::Config => format!(
                    "config file: {}",
                    AppConfig::project_settings_path(cwd).display()
                ),
                SlashCommand::Model(model) => {
                    if let Some(model) = model {
                        let lower = model.to_ascii_lowercase();
                        force_max_think =
                            lower.contains("reasoner") || lower.contains("max") || lower.contains("high");
                    }
                    format!(
                        "model mode: {}",
                        if force_max_think { &cfg.llm.max_think_model } else { &cfg.llm.base_model }
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
                SlashCommand::Mcp(_) => {
                    let servers = McpManager::new(cwd)?.list_servers()?;
                    format!("mcp servers: {}", servers.len())
                }
                SlashCommand::Rewind(_) => "run `deepseek rewind --yes`".to_string(),
                SlashCommand::Export(_) => {
                    let record = MemoryManager::new(cwd)?.export_transcript(
                        ExportFormat::Json,
                        None,
                        None,
                    )?;
                    format!("exported transcript {}", record.output_path)
                }
                SlashCommand::Plan => "plan mode enabled".to_string(),
                SlashCommand::Teleport => {
                    let bundle_id = Uuid::now_v7();
                    let output = runtime_dir(cwd)
                        .join("teleport")
                        .join(format!("{bundle_id}.json"));
                    if let Some(parent) = output.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    fs::write(&output, "{}")?;
                    format!("teleport bundle: {}", output.display())
                }
                SlashCommand::RemoteEnv => {
                    let profiles = Store::new(cwd)?.list_remote_env_profiles()?;
                    format!("remote profiles: {}", profiles.len())
                }
                SlashCommand::Status => {
                    let status = current_ui_status(cwd, cfg, force_max_think)?;
                    render_statusline(&status)
                }
                SlashCommand::Effort(level) => {
                    let level = level.unwrap_or_else(|| "medium".to_string());
                    force_max_think = matches!(level.as_str(), "high" | "max");
                    format!("effort={} force_max_think={}", level, force_max_think)
                }
                SlashCommand::Unknown { name, .. } => format!("unknown slash command: /{name}"),
            };
            return Ok(out);
        }
        engine.run_once_with_mode(prompt, allow_tools, force_max_think)
    })
}

fn current_ui_status(cwd: &Path, cfg: &AppConfig, force_max_think: bool) -> Result<UiStatus> {
    let store = Store::new(cwd)?;
    let session = store.load_latest_session()?;
    let projection = if let Some(session) = &session {
        store.rebuild_from_events(session.session_id)?
    } else {
        Default::default()
    };
    let usage = store.usage_summary(session.map(|s| s.session_id), None)?;
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
    let estimated_cost_usd = (usage.input_tokens as f64 / 1_000_000.0)
        * cfg.usage.cost_per_million_input
        + (usage.output_tokens as f64 / 1_000_000.0) * cfg.usage.cost_per_million_output;
    Ok(UiStatus {
        model: if force_max_think {
            cfg.llm.max_think_model.clone()
        } else {
            cfg.llm.base_model.clone()
        },
        pending_approvals,
        estimated_cost_usd,
        background_jobs,
        autopilot_running,
    })
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

    let checkpoint = MemoryManager::new(cwd)?.create_checkpoint("patch_apply")?;
    append_control_event(
        cwd,
        EventKind::CheckpointCreatedV1 {
            checkpoint_id: checkpoint.checkpoint_id,
            reason: checkpoint.reason.clone(),
            files_count: checkpoint.files_count,
            snapshot_path: checkpoint.snapshot_path.clone(),
        },
    )?;

    let (applied, conflicts) = store.apply(cwd, patch.patch_id)?;
    if json_mode {
        print_json(&json!({
            "patch_id": patch.patch_id,
            "applied": applied,
            "conflicts": conflicts,
            "checkpoint_id": checkpoint.checkpoint_id
        }))?;
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

fn run_profile(cwd: &Path, _args: ProfileArgs, json_mode: bool) -> Result<()> {
    let started = Instant::now();
    let cfg = AppConfig::ensure(cwd)?;
    let store = Store::new(cwd)?;
    let session = ensure_session_record(cwd, &store)?;
    let usage = store.usage_summary(Some(session.session_id), Some(24))?;
    let compactions = store.list_context_compactions(Some(session.session_id))?;
    let autopilot = store.load_latest_autopilot_run()?;
    let index = IndexService::new(cwd)?.status()?;
    let estimated_cost_usd = (usage.input_tokens as f64 / 1_000_000.0)
        * cfg.usage.cost_per_million_input
        + (usage.output_tokens as f64 / 1_000_000.0) * cfg.usage.cost_per_million_output;
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let profile_id = Uuid::now_v7();
    let summary = format!(
        "tokens={} cost_usd={:.6} compactions={} index_fresh={} autopilot={}",
        usage.input_tokens + usage.output_tokens,
        estimated_cost_usd,
        compactions.len(),
        index.as_ref().is_some_and(|m| m.fresh),
        autopilot
            .as_ref()
            .map(|run| run.status.as_str())
            .unwrap_or("none")
    );
    store.insert_profile_run(&deepseek_store::ProfileRunRecord {
        profile_id,
        session_id: session.session_id,
        summary: summary.clone(),
        elapsed_ms,
        created_at: Utc::now().to_rfc3339(),
    })?;
    append_control_event(
        cwd,
        EventKind::ProfileCapturedV1 {
            profile_id,
            summary: summary.clone(),
            elapsed_ms,
        },
    )?;

    let payload = json!({
        "profile_id": profile_id,
        "session_id": session.session_id,
        "summary": summary,
        "elapsed_ms": elapsed_ms,
        "usage": usage,
        "estimated_cost_usd": estimated_cost_usd,
        "compactions": compactions.len(),
        "autopilot": autopilot.map(|run| json!({
            "run_id": run.run_id,
            "status": run.status,
            "completed_iterations": run.completed_iterations,
            "failed_iterations": run.failed_iterations,
        })),
        "index": index,
    });

    if json_mode {
        print_json(&payload)?;
    } else {
        println!(
            "profile={} elapsed_ms={} cost_usd={:.6} tokens={} compactions={}",
            profile_id,
            elapsed_ms,
            estimated_cost_usd,
            usage.input_tokens + usage.output_tokens,
            compactions.len()
        );
    }
    Ok(())
}

fn run_rewind(cwd: &Path, args: RewindArgs, json_mode: bool) -> Result<()> {
    let memory = MemoryManager::new(cwd)?;
    let checkpoint_id = if let Some(value) = args.to_checkpoint.as_deref() {
        Uuid::parse_str(value)?
    } else {
        let checkpoints = memory.list_checkpoints()?;
        if checkpoints.is_empty() {
            let payload = json!({"rewound": false, "reason": "no_checkpoints"});
            if json_mode {
                print_json(&payload)?;
            } else {
                println!("no checkpoints available");
            }
            return Ok(());
        }
        checkpoints
            .into_iter()
            .next()
            .expect("non-empty checkpoints")
            .checkpoint_id
    };
    if !args.yes {
        return Err(anyhow!(
            "rewind requires --yes to confirm (target checkpoint: {})",
            checkpoint_id
        ));
    }
    let checkpoint = memory.rewind_to_checkpoint(checkpoint_id)?;
    append_control_event(
        cwd,
        EventKind::CheckpointRewoundV1 {
            checkpoint_id: checkpoint.checkpoint_id,
            reason: checkpoint.reason.clone(),
        },
    )?;
    let payload = json!({
        "checkpoint_id": checkpoint.checkpoint_id,
        "reason": checkpoint.reason,
        "snapshot_path": checkpoint.snapshot_path,
        "files_count": checkpoint.files_count,
        "created_at": checkpoint.created_at,
        "rewound": true,
    });
    if json_mode {
        print_json(&payload)?;
    } else {
        println!(
            "rewound to checkpoint {} (files={})",
            payload["checkpoint_id"].as_str().unwrap_or_default(),
            payload["files_count"].as_u64().unwrap_or(0)
        );
    }
    Ok(())
}

fn run_export(cwd: &Path, args: ExportArgs, json_mode: bool) -> Result<()> {
    let format = ExportFormat::parse(&args.format)
        .ok_or_else(|| anyhow!("unsupported format '{}'; expected json|md", args.format))?;
    let explicit_session = args.session.as_deref().map(Uuid::parse_str).transpose()?;
    let session = if explicit_session.is_none() {
        let store = Store::new(cwd)?;
        Some(ensure_session_record(cwd, &store)?.session_id)
    } else {
        explicit_session
    };
    let output = args.output.as_deref().map(PathBuf::from);
    let memory = MemoryManager::new(cwd)?;
    let record = memory.export_transcript(format, output.as_deref(), session)?;
    append_control_event(
        cwd,
        EventKind::TranscriptExportedV1 {
            export_id: record.export_id,
            format: record.format.clone(),
            output_path: record.output_path.clone(),
        },
    )?;
    if json_mode {
        print_json(&record)?;
    } else {
        println!(
            "exported transcript {} ({}) to {}",
            record.export_id, record.format, record.output_path
        );
    }
    Ok(())
}

fn run_memory(cwd: &Path, cmd: MemoryCmd, json_mode: bool) -> Result<()> {
    let manager = MemoryManager::new(cwd)?;
    match cmd {
        MemoryCmd::Show(_) => {
            let path = manager.ensure_initialized()?;
            let content = manager.read_memory()?;
            if json_mode {
                print_json(&json!({
                    "path": path,
                    "content": content,
                }))?;
            } else {
                println!("{}", content);
            }
        }
        MemoryCmd::Edit(_) => {
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
            let editor = std::env::var("EDITOR").unwrap_or_else(|_| default_editor().to_string());
            let status = Command::new(editor).arg(&path).status()?;
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
            if json_mode {
                print_json(&json!({
                    "edited": true,
                    "path": path,
                    "version_id": version_id,
                    "checkpoint_id": checkpoint.checkpoint_id
                }))?;
            } else {
                println!("updated {}", path.display());
            }
        }
        MemoryCmd::Sync(args) => {
            let path = manager.ensure_initialized()?;
            let note = args.note.unwrap_or_else(|| "sync".to_string());
            let version_id = manager.sync_memory_version(&note)?;
            append_control_event(
                cwd,
                EventKind::MemorySyncedV1 {
                    version_id,
                    path: path.to_string_lossy().to_string(),
                    note: note.clone(),
                },
            )?;
            if json_mode {
                print_json(&json!({
                    "synced": true,
                    "path": path,
                    "version_id": version_id,
                    "note": note,
                }))?;
            } else {
                println!("memory synced version={} note={}", version_id, note);
            }
        }
    }
    Ok(())
}

fn run_mcp(cwd: &Path, cmd: McpCmd, json_mode: bool) -> Result<()> {
    let manager = McpManager::new(cwd)?;
    match cmd {
        McpCmd::Add(args) => {
            let metadata = if let Some(metadata) = args.metadata.as_deref() {
                serde_json::from_str(metadata)?
            } else {
                serde_json::Value::Null
            };
            let server = McpServer {
                id: args.id.clone(),
                name: args.name.unwrap_or_else(|| args.id.clone()),
                transport: args.transport.into_transport(),
                command: args.command,
                args: args.args,
                url: args.url,
                enabled: args.enabled,
                metadata,
            };
            let endpoint = server
                .command
                .clone()
                .or_else(|| server.url.clone())
                .unwrap_or_default();
            let transport = match server.transport {
                McpTransport::Stdio => "stdio",
                McpTransport::Http => "http",
            };
            manager.add_server(server.clone())?;
            append_control_event(
                cwd,
                EventKind::McpServerAddedV1 {
                    server_id: server.id.clone(),
                    transport: transport.to_string(),
                    endpoint,
                },
            )?;

            let (discovered, refreshes) = manager.refresh_tools()?;
            let discovered_for_server = discovered
                .iter()
                .filter(|tool| tool.server_id == server.id)
                .cloned()
                .collect::<Vec<_>>();
            emit_mcp_discovery_events(cwd, &refreshes)?;

            if json_mode {
                print_json(&json!({
                    "added": server,
                    "discovered_tools": discovered_for_server,
                }))?;
            } else {
                println!("added mcp server {} (transport={})", args.id, transport);
            }
        }
        McpCmd::List => {
            let (_, refreshes, notice) =
                manager.discover_tools_with_notice(read_mcp_fingerprint(cwd)?.as_deref())?;
            write_mcp_fingerprint(cwd, &notice.fingerprint)?;
            emit_mcp_discovery_events(cwd, &refreshes)?;
            let servers = manager.list_servers()?;
            if json_mode {
                print_json(&servers)?;
            } else if servers.is_empty() {
                println!("no mcp servers configured");
            } else {
                for server in servers {
                    println!(
                        "{} {} enabled={} endpoint={}",
                        server.id,
                        match server.transport {
                            McpTransport::Stdio => "stdio",
                            McpTransport::Http => "http",
                        },
                        server.enabled,
                        server
                            .command
                            .as_deref()
                            .or(server.url.as_deref())
                            .unwrap_or_default()
                    );
                }
            }
        }
        McpCmd::Get(args) => {
            let server = manager
                .get_server(&args.server_id)?
                .ok_or_else(|| anyhow!("mcp server not found: {}", args.server_id))?;
            if json_mode {
                print_json(&server)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&server)?);
            }
        }
        McpCmd::Remove(args) => {
            let removed = manager.remove_server(&args.server_id)?;
            if removed {
                append_control_event(
                    cwd,
                    EventKind::McpServerRemovedV1 {
                        server_id: args.server_id.clone(),
                    },
                )?;
            }
            if json_mode {
                print_json(&json!({
                    "server_id": args.server_id,
                    "removed": removed,
                }))?;
            } else if removed {
                println!("removed mcp server {}", args.server_id);
            } else {
                println!("mcp server not found: {}", args.server_id);
            }
        }
    }
    Ok(())
}

fn mcp_fingerprint_path(cwd: &Path) -> PathBuf {
    runtime_dir(cwd).join("mcp").join("tools_fingerprint.txt")
}

fn read_mcp_fingerprint(cwd: &Path) -> Result<Option<String>> {
    let path = mcp_fingerprint_path(cwd);
    if !path.exists() {
        return Ok(None);
    }
    let value = fs::read_to_string(path)?;
    let value = value.trim();
    if value.is_empty() {
        return Ok(None);
    }
    Ok(Some(value.to_string()))
}

fn write_mcp_fingerprint(cwd: &Path, fingerprint: &str) -> Result<()> {
    let path = mcp_fingerprint_path(cwd);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, fingerprint)?;
    Ok(())
}

fn emit_mcp_discovery_events(cwd: &Path, refreshes: &[deepseek_mcp::McpToolRefresh]) -> Result<()> {
    for refresh in refreshes {
        for tool_name in &refresh.added {
            append_control_event(
                cwd,
                EventKind::McpToolDiscoveredV1 {
                    server_id: refresh.server_id.clone(),
                    tool_name: tool_name.clone(),
                },
            )?;
        }
    }
    Ok(())
}

fn run_status(cwd: &Path, json_mode: bool) -> Result<()> {
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
                "base": cfg.llm.base_model,
                "max_think": cfg.llm.max_think_model,
            },
            "context_usage_percent": context_usage_pct,
            "pending_approvals": pending_approvals,
            "plugins": {
                "installed": plugins.len(),
                "enabled": plugins.iter().filter(|p| p.enabled).count(),
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
                "base": cfg.llm.base_model,
                "max_think": cfg.llm.max_think_model,
            },
            "context_usage_percent": 0.0,
            "pending_approvals": 0,
            "plugins": {
                "installed": plugins.len(),
                "enabled": plugins.iter().filter(|p| p.enabled).count(),
            },
            "mcp_servers": mcp_servers.len(),
            "autopilot": null,
        })
    };

    if json_mode {
        print_json(&payload)?;
    } else {
        println!(
            "session={} state={} model={}/{} context={:.1}% pending_approvals={} plugins={}/{}",
            payload["session_id"].as_str().unwrap_or("none"),
            payload["state"].as_str().unwrap_or("unknown"),
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

fn run_usage(cwd: &Path, args: UsageArgs, json_mode: bool) -> Result<()> {
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
            "input_tokens={} output_tokens={} estimated_cost_usd={:.6} compactions={} rate_limits={}",
            usage.input_tokens,
            usage.output_tokens,
            input_cost + output_cost,
            compactions.len(),
            rate_limit_events,
        );
    }

    Ok(())
}

fn run_compact(cwd: &Path, args: CompactArgs, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;
    let store = Store::new(cwd)?;
    let session = store
        .load_latest_session()?
        .ok_or_else(|| anyhow!("no session found to compact"))?;
    let projection = store.rebuild_from_events(session.session_id)?;
    if projection.transcript.is_empty() {
        let payload = json!({"status":"no_op", "reason":"empty_transcript"});
        if json_mode {
            print_json(&payload)?;
        } else {
            println!("no transcript to compact");
        }
        return Ok(());
    }

    let from_turn = args.from_turn.unwrap_or(1).max(1);
    let transcript_len = projection.transcript.len() as u64;
    if from_turn > transcript_len {
        return Err(anyhow!(
            "--from-turn {} exceeds transcript length {}",
            from_turn,
            transcript_len
        ));
    }

    let selected = projection
        .transcript
        .iter()
        .skip((from_turn - 1) as usize)
        .cloned()
        .collect::<Vec<_>>();
    let summary_id = Uuid::now_v7();
    let full_text = selected.join("\n");
    let before_tokens = estimate_tokens(&full_text);
    let summary_lines = selected
        .iter()
        .take(12)
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.len() > 200 {
                format!("- {}...", &trimmed[..200])
            } else {
                format!("- {trimmed}")
            }
        })
        .collect::<Vec<_>>();
    let summary = format!(
        "Compaction summary {}\nfrom_turn: {}\nto_turn: {}\n\n{}",
        summary_id,
        from_turn,
        transcript_len,
        summary_lines.join("\n")
    );
    let after_tokens = estimate_tokens(&summary);
    let token_delta_estimate = before_tokens as i64 - after_tokens as i64;
    let replay_pointer = format!(".deepseek/compactions/{summary_id}.md");
    let payload = json!({
        "summary_id": summary_id,
        "from_turn": from_turn,
        "to_turn": transcript_len,
        "token_delta_estimate": token_delta_estimate,
        "replay_pointer": replay_pointer,
    });

    if cfg.context.compact_preview && !args.yes {
        if json_mode {
            print_json(&json!({
                "preview": true,
                "persisted": false,
                "summary": summary,
                "result": payload,
            }))?;
        } else {
            println!("compaction preview (not persisted):");
            println!("{summary}");
            println!("rerun with --yes to persist");
        }
        return Ok(());
    }

    let summary_path = cwd.join(&replay_pointer);
    if let Some(parent) = summary_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&summary_path, summary)?;
    append_control_event(
        cwd,
        EventKind::ContextCompactedV1 {
            summary_id,
            from_turn,
            to_turn: transcript_len,
            token_delta_estimate,
            replay_pointer: replay_pointer.clone(),
        },
    )?;

    if json_mode {
        print_json(&json!({
            "preview": false,
            "persisted": true,
            "result": payload,
        }))?;
    } else {
        println!(
            "compacted turns {}..{} summary_id={} token_delta_estimate={} replay_pointer={}",
            from_turn, transcript_len, summary_id, token_delta_estimate, replay_pointer
        );
    }

    Ok(())
}

fn run_doctor(cwd: &Path, _args: DoctorArgs, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;
    let plugin_manager = PluginManager::new(cwd)?;
    let plugins = plugin_manager.list().unwrap_or_default();

    let runtime = runtime_dir(cwd);
    fs::create_dir_all(&runtime)?;
    let rustc = run_capture("rustc", &["--version"]);
    let cargo = run_capture("cargo", &["--version"]);
    let shell = std::env::var("SHELL")
        .ok()
        .or_else(|| std::env::var("ComSpec").ok())
        .unwrap_or_else(|| "unknown".to_string());
    let api_key_set = std::env::var(&cfg.llm.api_key_env)
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);

    let checks = json!({
        "git": command_exists("git"),
        "rg": command_exists("rg"),
        "cargo": command_exists("cargo"),
        "shell": command_exists(shell.split(std::path::MAIN_SEPARATOR).next_back().unwrap_or("sh")),
    });

    let mut warnings = Vec::new();
    if !api_key_set && !cfg.llm.offline_fallback {
        warnings.push(format!(
            "{} not set and offline_fallback=false",
            cfg.llm.api_key_env
        ));
    }
    if checks["git"].as_bool() != Some(true) {
        warnings.push("git not found in PATH".to_string());
    }
    if checks["cargo"].as_bool() != Some(true) {
        warnings.push("cargo not found in PATH".to_string());
    }

    let payload = json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "shell": shell,
        "workspace": cwd,
        "runtime_dir": runtime,
        "config_path": AppConfig::config_path(cwd),
        "config_paths": {
            "user": AppConfig::user_settings_path(),
            "project": AppConfig::project_settings_path(cwd),
            "project_local": AppConfig::project_local_settings_path(cwd),
            "legacy_toml": AppConfig::legacy_toml_path(cwd),
            "keybindings": AppConfig::keybindings_path(),
        },
        "binary_path": std::env::current_exe().ok(),
        "toolchain": {
            "rustc": rustc,
            "cargo": cargo,
        },
        "llm": {
            "endpoint": cfg.llm.endpoint,
            "api_key_env": cfg.llm.api_key_env,
            "api_key_set": api_key_set,
            "offline_fallback": cfg.llm.offline_fallback,
            "base_model": cfg.llm.base_model,
            "max_think_model": cfg.llm.max_think_model,
        },
        "plugins": {
            "installed": plugins.len(),
            "enabled": plugins.iter().filter(|p| p.enabled).count(),
        },
        "checks": checks,
        "warnings": warnings,
    });

    if json_mode {
        print_json(&payload)?;
    } else {
        println!(
            "doctor: os={} arch={} shell={}",
            payload["os"].as_str().unwrap_or_default(),
            payload["arch"].as_str().unwrap_or_default(),
            payload["shell"].as_str().unwrap_or_default()
        );
        println!(
            "toolchain: {} | {}",
            payload["toolchain"]["rustc"]
                .as_str()
                .unwrap_or("unavailable"),
            payload["toolchain"]["cargo"]
                .as_str()
                .unwrap_or("unavailable")
        );
        println!(
            "llm: base={} max={} endpoint={} api_key_set={} offline_fallback={}",
            payload["llm"]["base_model"].as_str().unwrap_or_default(),
            payload["llm"]["max_think_model"]
                .as_str()
                .unwrap_or_default(),
            payload["llm"]["endpoint"].as_str().unwrap_or_default(),
            payload["llm"]["api_key_set"].as_bool().unwrap_or(false),
            payload["llm"]["offline_fallback"]
                .as_bool()
                .unwrap_or(false)
        );
        println!(
            "plugins: enabled={} installed={}",
            payload["plugins"]["enabled"].as_u64().unwrap_or(0),
            payload["plugins"]["installed"].as_u64().unwrap_or(0)
        );
        if let Some(warnings) = payload["warnings"].as_array()
            && !warnings.is_empty()
        {
            println!("warnings:");
            for warning in warnings {
                if let Some(text) = warning.as_str() {
                    println!("- {text}");
                }
            }
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
        IndexCmd::Watch {
            events,
            timeout_seconds,
        } => {
            let manifest = service.watch_and_update(
                &session,
                events.max(1),
                Duration::from_secs(timeout_seconds.max(1)),
            )?;
            if json_mode {
                print_json(&manifest)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&manifest)?);
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
        PluginCmd::Catalog => {
            let cfg = AppConfig::load(cwd)?;
            let catalog = manager
                .sync_catalog(&cfg.plugins.catalog)
                .or_else(|_| manager.search_catalog("", &cfg.plugins.catalog))?;
            append_control_event(
                cwd,
                EventKind::PluginCatalogSyncedV1 {
                    source: cfg.plugins.catalog.index_url,
                    total: catalog.len(),
                    verified_count: catalog.iter().filter(|p| p.verified).count(),
                },
            )?;
            if json_mode {
                print_json(&catalog)?;
            } else {
                for item in catalog {
                    println!(
                        "{} {} ({}) {}",
                        item.plugin_id,
                        item.version,
                        if item.verified {
                            "verified"
                        } else {
                            "unverified"
                        },
                        item.source
                    );
                }
            }
        }
        PluginCmd::Search(args) => {
            let cfg = AppConfig::load(cwd)?;
            let matches = manager.search_catalog(&args.query, &cfg.plugins.catalog)?;
            if json_mode {
                print_json(&matches)?;
            } else if matches.is_empty() {
                println!("no catalog results for '{}'", args.query);
            } else {
                for item in matches {
                    println!(
                        "{} {} ({}) - {}",
                        item.plugin_id,
                        item.version,
                        if item.verified {
                            "verified"
                        } else {
                            "unverified"
                        },
                        item.description
                    );
                }
            }
        }
        PluginCmd::Verify(args) => {
            let cfg = AppConfig::load(cwd)?;
            let result = manager.verify_catalog_plugin(&args.plugin_id, &cfg.plugins.catalog)?;
            append_control_event(
                cwd,
                EventKind::PluginVerifiedV1 {
                    plugin_id: result.plugin_id.clone(),
                    verified: result.verified,
                    reason: result.reason.clone(),
                },
            )?;
            if json_mode {
                print_json(&result)?;
            } else {
                println!(
                    "{} verified={} reason={} source={}",
                    result.plugin_id, result.verified, result.reason, result.source
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

fn run_resume(cwd: &Path, args: RunArgs) -> Result<String> {
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
    AgentEngine::new(cwd)?.resume()
}

fn run_git(cwd: &Path, cmd: GitCmd, json_mode: bool) -> Result<()> {
    match cmd {
        GitCmd::Status => {
            let output = run_process(cwd, "git", &["status", "--short"])?;
            if json_mode {
                print_json(&json!({"command":"git status --short", "output": output}))?;
            } else {
                println!("{output}");
            }
        }
        GitCmd::History(args) => {
            let output = run_process(
                cwd,
                "git",
                &["log", "--oneline", "-n", &args.limit.to_string()],
            )?;
            if json_mode {
                print_json(&json!({"limit": args.limit, "output": output}))?;
            } else {
                println!("{output}");
            }
        }
        GitCmd::Branch => {
            let output = run_process(cwd, "git", &["branch", "--all", "--verbose"])?;
            if json_mode {
                print_json(&json!({"output": output}))?;
            } else {
                println!("{output}");
            }
        }
        GitCmd::Checkout(args) => {
            let output = run_process(cwd, "git", &["checkout", &args.target])?;
            if json_mode {
                print_json(&json!({"target": args.target, "output": output}))?;
            } else {
                println!("{output}");
            }
        }
        GitCmd::Commit(args) => {
            let mut logs = Vec::new();
            if args.all {
                logs.push(run_process(cwd, "git", &["add", "-A"])?);
            }
            logs.push(run_process(cwd, "git", &["commit", "-m", &args.message])?);
            if json_mode {
                print_json(&json!({"message": args.message, "output": logs}))?;
            } else {
                println!("{}", logs.join("\n"));
            }
        }
        GitCmd::Pr(args) => {
            let gh_available = command_exists("gh");
            if !gh_available || args.dry_run {
                let payload = json!({
                    "available": gh_available,
                    "dry_run": args.dry_run,
                    "suggested_command": format!(
                        "gh pr create{}{}{}{}",
                        args.title.as_deref().map(|title| format!(" --title \"{title}\"")).unwrap_or_default(),
                        args.body.as_deref().map(|body| format!(" --body \"{body}\"")).unwrap_or_default(),
                        args.base.as_deref().map(|base| format!(" --base {base}")).unwrap_or_default(),
                        args.head.as_deref().map(|head| format!(" --head {head}")).unwrap_or_default(),
                    )
                });
                if json_mode {
                    print_json(&payload)?;
                } else {
                    println!("{}", serde_json::to_string_pretty(&payload)?);
                }
            } else {
                let mut cmd = Command::new("gh");
                cmd.current_dir(cwd).arg("pr").arg("create");
                if let Some(title) = args.title {
                    cmd.arg("--title").arg(title);
                }
                if let Some(body) = args.body {
                    cmd.arg("--body").arg(body);
                }
                if let Some(base) = args.base {
                    cmd.arg("--base").arg(base);
                }
                if let Some(head) = args.head {
                    cmd.arg("--head").arg(head);
                }
                let output = cmd.output()?;
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                if !output.status.success() {
                    return Err(anyhow!(
                        "gh pr create failed: {}\n{}",
                        output.status,
                        stderr
                    ));
                }
                if json_mode {
                    print_json(&json!({"stdout": stdout, "stderr": stderr}))?;
                } else {
                    println!("{stdout}");
                }
            }
        }
        GitCmd::Resolve(args) => {
            let strategy = args.strategy.to_ascii_lowercase();
            if strategy == "list" {
                let output = run_process(cwd, "git", &["diff", "--name-only", "--diff-filter=U"])?;
                if json_mode {
                    print_json(&json!({"conflicts": output.lines().collect::<Vec<_>>() }))?;
                } else {
                    println!("{output}");
                }
            } else {
                let file = args
                    .file
                    .ok_or_else(|| anyhow!("--file is required for strategy '{}'", strategy))?;
                if strategy != "ours" && strategy != "theirs" {
                    return Err(anyhow!("unsupported strategy '{}'", strategy));
                }
                let output = run_process(
                    cwd,
                    "git",
                    &["checkout", &format!("--{strategy}"), "--", &file],
                )?;
                if json_mode {
                    print_json(&json!({"strategy": strategy, "file": file, "output": output}))?;
                } else {
                    println!("{output}");
                }
            }
        }
    }
    Ok(())
}

fn run_skills(cwd: &Path, cmd: SkillsCmd, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;
    let manager = SkillManager::new(cwd)?;
    let paths = cfg
        .skills
        .paths
        .iter()
        .map(|path| expand_tilde(path))
        .collect::<Vec<_>>();
    let store = Store::new(cwd)?;
    match cmd {
        SkillsCmd::List => {
            let skills = manager.list(&paths)?;
            if json_mode {
                print_json(&skills)?;
            } else if skills.is_empty() {
                println!("no skills found");
            } else {
                for skill in skills {
                    println!("{} {} ({})", skill.id, skill.name, skill.path.display());
                }
            }
        }
        SkillsCmd::Install(args) => {
            let installed = manager.install(Path::new(&args.source))?;
            store.set_skill_registry(&deepseek_store::SkillRegistryRecord {
                skill_id: installed.id.clone(),
                name: installed.name.clone(),
                path: installed.path.to_string_lossy().to_string(),
                enabled: true,
                metadata_json: serde_json::json!({"summary": installed.summary}).to_string(),
                updated_at: Utc::now().to_rfc3339(),
            })?;
            append_control_event(
                cwd,
                EventKind::SkillLoadedV1 {
                    skill_id: installed.id.clone(),
                    source_path: installed.path.to_string_lossy().to_string(),
                },
            )?;
            if json_mode {
                print_json(&installed)?;
            } else {
                println!(
                    "installed skill {} ({})",
                    installed.id,
                    installed.path.display()
                );
            }
        }
        SkillsCmd::Remove(args) => {
            manager.remove(&args.skill_id)?;
            store.remove_skill_registry(&args.skill_id)?;
            if json_mode {
                print_json(&json!({"removed": args.skill_id}))?;
            } else {
                println!("removed skill {}", args.skill_id);
            }
        }
        SkillsCmd::Run(args) => {
            let run = manager.run(&args.skill_id, args.input.as_deref(), &paths)?;
            if args.execute {
                let output = AgentEngine::new(cwd)?.run_once(&run.rendered_prompt, false)?;
                if json_mode {
                    print_json(&json!({"skill": run, "output": output}))?;
                } else {
                    println!("{output}");
                }
            } else if json_mode {
                print_json(&run)?;
            } else {
                println!("{}", run.rendered_prompt);
            }
        }
        SkillsCmd::Reload => {
            let loaded = manager.reload(&paths)?;
            for skill in &loaded {
                store.set_skill_registry(&deepseek_store::SkillRegistryRecord {
                    skill_id: skill.id.clone(),
                    name: skill.name.clone(),
                    path: skill.path.to_string_lossy().to_string(),
                    enabled: true,
                    metadata_json: serde_json::json!({"summary": skill.summary}).to_string(),
                    updated_at: Utc::now().to_rfc3339(),
                })?;
            }
            if json_mode {
                print_json(&json!({"reloaded": loaded.len(), "skills": loaded}))?;
            } else {
                println!("reloaded {} skills", loaded.len());
            }
        }
    }
    Ok(())
}

fn run_replay(cwd: &Path, cmd: ReplayCmd, json_mode: bool) -> Result<()> {
    match cmd {
        ReplayCmd::Run(args) => {
            let cfg = AppConfig::ensure(cwd)?;
            if cfg.replay.strict_mode && !args.deterministic {
                return Err(anyhow!(
                    "replay.strict_mode=true requires --deterministic=true"
                ));
            }
            let session_id = Uuid::parse_str(&args.session_id)?;
            let store = Store::new(cwd)?;
            let events = read_session_events(cwd, session_id)?;
            let projection = store.rebuild_from_events(session_id)?;
            let events_replayed = events.len() as u64;
            let tool_results_replayed = events
                .iter()
                .filter(|event| matches!(event.kind, EventKind::ToolResultV1 { .. }))
                .count() as u64;
            let payload = json!({
                "session_id": session_id,
                "deterministic": args.deterministic,
                "strict_mode": cfg.replay.strict_mode,
                "events_replayed": events_replayed,
                "tool_results_replayed": tool_results_replayed,
                "turns": projection.transcript.len(),
                "steps": projection.step_status.len(),
                "router_models": projection.router_models,
            });
            store.insert_replay_cassette(&ReplayCassetteRecord {
                cassette_id: Uuid::now_v7(),
                session_id,
                deterministic: args.deterministic,
                events_count: events_replayed,
                payload_json: payload.to_string(),
                created_at: Utc::now().to_rfc3339(),
            })?;
            append_control_event(
                cwd,
                EventKind::ReplayExecutedV1 {
                    session_id,
                    deterministic: args.deterministic,
                    events_replayed,
                },
            )?;
            if json_mode {
                print_json(&payload)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&payload)?);
            }
        }
    }
    Ok(())
}

fn run_background(cwd: &Path, cmd: BackgroundCmd, json_mode: bool) -> Result<()> {
    let store = Store::new(cwd)?;
    match cmd {
        BackgroundCmd::List => {
            let jobs = store.list_background_jobs()?;
            if json_mode {
                print_json(&jobs)?;
            } else if jobs.is_empty() {
                println!("no background jobs");
            } else {
                for job in jobs {
                    println!(
                        "{} {} {} {}",
                        job.job_id, job.kind, job.status, job.reference
                    );
                }
            }
        }
        BackgroundCmd::Attach(args) => {
            let job_id = Uuid::parse_str(&args.job_id)?;
            let job = store
                .load_background_job(job_id)?
                .ok_or_else(|| anyhow!("background job not found: {}", args.job_id))?;
            append_control_event(
                cwd,
                EventKind::BackgroundJobResumedV1 {
                    job_id,
                    reference: job.reference.clone(),
                },
            )?;
            let payload = json!({
                "job_id": job_id,
                "kind": job.kind,
                "status": job.status,
                "reference": job.reference,
                "metadata": serde_json::from_str::<serde_json::Value>(&job.metadata_json).unwrap_or(serde_json::Value::Null)
            });
            if json_mode {
                print_json(&payload)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&payload)?);
            }
        }
        BackgroundCmd::Stop(args) => {
            let job_id = Uuid::parse_str(&args.job_id)?;
            let mut job = store
                .load_background_job(job_id)?
                .ok_or_else(|| anyhow!("background job not found: {}", args.job_id))?;
            if job.kind == "autopilot"
                && let Ok(run_id) = Uuid::parse_str(&job.reference)
                && let Some(run) = store.load_autopilot_run(run_id)?
            {
                let stop_path = if run.stop_file.trim().is_empty() {
                    runtime_dir(cwd).join("autopilot.stop")
                } else {
                    PathBuf::from(run.stop_file)
                };
                if let Some(parent) = stop_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(
                    &stop_path,
                    format!("stop requested at {}\n", Utc::now().to_rfc3339()),
                )?;
            }
            job.status = "stopped".to_string();
            job.updated_at = Utc::now().to_rfc3339();
            job.metadata_json = serde_json::json!({"reason":"manual_stop"}).to_string();
            store.upsert_background_job(&job)?;
            append_control_event(
                cwd,
                EventKind::BackgroundJobStoppedV1 {
                    job_id,
                    reason: "manual_stop".to_string(),
                },
            )?;
            if json_mode {
                print_json(&json!({"job_id": job_id, "stopped": true}))?;
            } else {
                println!("stopped background job {}", job_id);
            }
        }
    }
    Ok(())
}

fn run_teleport(cwd: &Path, args: TeleportArgs, json_mode: bool) -> Result<()> {
    if let Some(import_path) = args.import {
        let raw = fs::read_to_string(&import_path)?;
        let value: serde_json::Value = serde_json::from_str(&raw)?;
        if json_mode {
            print_json(&json!({"imported": true, "bundle": value}))?;
        } else {
            println!("imported teleport bundle from {}", import_path);
        }
        return Ok(());
    }

    let store = Store::new(cwd)?;
    let session_id = if let Some(session_id) = args.session_id {
        Uuid::parse_str(&session_id)?
    } else {
        ensure_session_record(cwd, &store)?.session_id
    };
    let projection = store.rebuild_from_events(session_id)?;
    let bundle_id = Uuid::now_v7();
    let output_path = args.output.map(PathBuf::from).unwrap_or_else(|| {
        runtime_dir(cwd)
            .join("teleport")
            .join(format!("{bundle_id}.json"))
    });
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let payload = json!({
        "bundle_id": bundle_id,
        "session_id": session_id,
        "created_at": Utc::now().to_rfc3339(),
        "turns": projection.transcript,
        "steps": projection.step_status,
    });
    fs::write(&output_path, serde_json::to_vec_pretty(&payload)?)?;
    append_control_event(
        cwd,
        EventKind::TeleportBundleCreatedV1 {
            bundle_id,
            path: output_path.to_string_lossy().to_string(),
        },
    )?;
    if json_mode {
        print_json(&json!({"bundle_id": bundle_id, "path": output_path}))?;
    } else {
        println!("teleport bundle created at {}", output_path.display());
    }
    Ok(())
}

fn run_remote_env(cwd: &Path, cmd: RemoteEnvCmd, json_mode: bool) -> Result<()> {
    let store = Store::new(cwd)?;
    match cmd {
        RemoteEnvCmd::List => {
            let profiles = store.list_remote_env_profiles()?;
            if json_mode {
                print_json(&profiles)?;
            } else if profiles.is_empty() {
                println!("no remote environment profiles configured");
            } else {
                for profile in profiles {
                    println!(
                        "{} {} {}",
                        profile.profile_id, profile.name, profile.endpoint
                    );
                }
            }
        }
        RemoteEnvCmd::Add(args) => {
            let profile_id = Uuid::now_v7();
            store.upsert_remote_env_profile(&deepseek_store::RemoteEnvProfileRecord {
                profile_id,
                name: args.name.clone(),
                endpoint: args.endpoint.clone(),
                auth_mode: args.auth_mode.clone(),
                metadata_json: "{}".to_string(),
                updated_at: Utc::now().to_rfc3339(),
            })?;
            append_control_event(
                cwd,
                EventKind::RemoteEnvConfiguredV1 {
                    profile_id,
                    name: args.name,
                    endpoint: args.endpoint,
                },
            )?;
            if json_mode {
                print_json(&json!({"profile_id": profile_id, "configured": true}))?;
            } else {
                println!("remote profile configured: {}", profile_id);
            }
        }
        RemoteEnvCmd::Remove(args) => {
            let profile_id = Uuid::parse_str(&args.profile_id)?;
            store.remove_remote_env_profile(profile_id)?;
            if json_mode {
                print_json(&json!({"profile_id": profile_id, "removed": true}))?;
            } else {
                println!("remote profile removed: {}", profile_id);
            }
        }
        RemoteEnvCmd::Check(args) => {
            let profile_id = Uuid::parse_str(&args.profile_id)?;
            let profile = store
                .load_remote_env_profile(profile_id)?
                .ok_or_else(|| anyhow!("remote profile not found: {}", profile_id))?;
            let payload = json!({
                "profile_id": profile.profile_id,
                "name": profile.name,
                "endpoint": profile.endpoint,
                "auth_mode": profile.auth_mode,
                "reachable": true,
            });
            if json_mode {
                print_json(&payload)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&payload)?);
            }
        }
    }
    Ok(())
}

fn run_process(cwd: &Path, program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).current_dir(cwd).args(args).output()?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        return Err(anyhow!(
            "{} {:?} failed with status {}: {}{}",
            program,
            args,
            output.status,
            stdout,
            stderr
        ));
    }
    Ok(format!("{stdout}{stderr}").trim().to_string())
}

fn read_session_events(cwd: &Path, session_id: Uuid) -> Result<Vec<EventEnvelope>> {
    let path = runtime_dir(cwd).join("events.jsonl");
    let Ok(raw) = fs::read_to_string(path) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for line in raw.lines() {
        let Ok(event) = serde_json::from_str::<EventEnvelope>(line) else {
            continue;
        };
        if event.session_id == session_id {
            out.push(event);
        }
    }
    Ok(out)
}

fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = std::env::var("HOME")
            .ok()
            .or_else(|| std::env::var("USERPROFILE").ok())
    {
        return Path::new(&home).join(rest).to_string_lossy().to_string();
    }
    path.to_string()
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
    let session = ensure_session_record(cwd, &store)?;
    let event = EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: Utc::now(),
        session_id: session.session_id,
        kind,
    };
    store.append_event(&event)?;
    Ok(())
}

fn ensure_session_record(cwd: &Path, store: &Store) -> Result<Session> {
    if let Some(existing) = store.load_latest_session()? {
        return Ok(existing);
    }
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
    Ok(session)
}

fn estimate_tokens(text: &str) -> u64 {
    (text.chars().count() as u64).div_ceil(4)
}

fn estimate_rate_limit_events(cwd: &Path) -> u64 {
    let path = runtime_dir(cwd).join("observe.log");
    let Ok(raw) = fs::read_to_string(path) else {
        return 0;
    };
    raw.lines()
        .filter(|line| {
            let lower = line.to_ascii_lowercase();
            lower.contains("429") || lower.contains("rate limit")
        })
        .count() as u64
}

fn run_capture(cmd: &str, args: &[&str]) -> Option<String> {
    Command::new(cmd).args(args).output().ok().and_then(|out| {
        if out.status.success() {
            Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
        } else {
            None
        }
    })
}

fn command_exists(name: &str) -> bool {
    if name.trim().is_empty() {
        return false;
    }
    let checker = if cfg!(target_os = "windows") {
        ("where", vec![name])
    } else {
        ("which", vec![name])
    };
    Command::new(checker.0)
        .args(checker.1)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}
