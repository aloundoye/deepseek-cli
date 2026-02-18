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
use deepseek_ui::{
    KeyBindings, SlashCommand, UiStatus, load_keybindings, render_statusline,
    run_tui_shell_with_bindings,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
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
    Benchmark {
        #[command(subcommand)]
        command: BenchmarkCmd,
    },
    Permissions {
        #[command(subcommand)]
        command: PermissionsCmd,
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
    pause_file: Option<String>,
    #[arg(long)]
    heartbeat_file: Option<String>,
    #[arg(long, default_value_t = 10)]
    max_consecutive_failures: u64,
}

#[derive(Subcommand)]
enum AutopilotCmd {
    Status(AutopilotStatusArgs),
    Pause(AutopilotPauseArgs),
    Stop(AutopilotStopArgs),
    Resume(AutopilotResumeArgs),
}

#[derive(Args)]
struct AutopilotStatusArgs {
    #[arg(long)]
    run_id: Option<String>,
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    follow: bool,
    #[arg(long)]
    samples: Option<u64>,
    #[arg(long, default_value_t = 2)]
    interval_seconds: u64,
}

#[derive(Args)]
struct AutopilotStopArgs {
    #[arg(long)]
    run_id: Option<String>,
}

#[derive(Args)]
struct AutopilotPauseArgs {
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
struct ProfileArgs {
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    benchmark: bool,
    #[arg(long, default_value_t = 5)]
    benchmark_cases: usize,
    #[arg(long)]
    benchmark_seed: Option<u64>,
    #[arg(long)]
    benchmark_suite: Option<String>,
    #[arg(long)]
    benchmark_pack: Option<String>,
    #[arg(long, default_value = "DEEPSEEK_BENCHMARK_SIGNING_KEY")]
    benchmark_signing_key_env: String,
    #[arg(long)]
    benchmark_min_success_rate: Option<f64>,
    #[arg(long)]
    benchmark_min_quality_rate: Option<f64>,
    #[arg(long)]
    benchmark_max_p95_ms: Option<u64>,
    #[arg(long)]
    benchmark_baseline: Option<String>,
    #[arg(long)]
    benchmark_max_regression_ms: Option<u64>,
    #[arg(long = "benchmark-compare")]
    benchmark_compare: Vec<String>,
    #[arg(long)]
    benchmark_output: Option<String>,
}

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
enum BenchmarkCmd {
    ListPacks,
    ShowPack(BenchmarkShowPackArgs),
    ImportPack(BenchmarkImportPackArgs),
}

#[derive(Args)]
struct BenchmarkShowPackArgs {
    name: String,
}

#[derive(Args)]
struct BenchmarkImportPackArgs {
    name: String,
    source: String,
}

#[derive(Subcommand)]
enum PermissionsCmd {
    Show,
    Set(PermissionsSetArgs),
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum PermissionModeArg {
    Ask,
    Always,
    Never,
}

impl PermissionModeArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::Always => "always",
            Self::Never => "never",
        }
    }
}

#[derive(Args, Default)]
struct PermissionsSetArgs {
    #[arg(long)]
    approve_bash: Option<PermissionModeArg>,
    #[arg(long)]
    approve_edits: Option<PermissionModeArg>,
    #[arg(long)]
    sandbox_mode: Option<String>,
    #[arg(long = "allow")]
    allow: Vec<String>,
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    clear_allowlist: bool,
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
            ensure_llm_ready(&cwd, cli.json)?;
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
            ensure_llm_ready(&cwd, cli.json)?;
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
            let output = run_resume(&cwd, args, cli.json)?;
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
        Commands::Benchmark { command } => run_benchmark(cwd.as_path(), command, cli.json),
        Commands::Permissions { command } => run_permissions(&cwd, command, cli.json),
        Commands::Plugins { command } => run_plugins(&cwd, command, cli.json),
        Commands::Clean(args) => run_clean(&cwd, args, cli.json),
    }
}

fn run_autopilot_cmd(cwd: &Path, args: AutopilotArgs, json_mode: bool) -> Result<()> {
    match args.command {
        Some(AutopilotCmd::Status(status)) => run_autopilot_status(cwd, status, json_mode),
        Some(AutopilotCmd::Pause(pause)) => run_autopilot_pause(cwd, pause, json_mode),
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
                    pause_file: args.pause_file,
                    heartbeat_file: args.heartbeat_file,
                    max_consecutive_failures: args.max_consecutive_failures,
                },
                json_mode,
            )
        }
    }
}

fn ensure_llm_ready(cwd: &Path, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;
    ensure_llm_ready_with_cfg(&cfg, json_mode)
}

fn ensure_llm_ready_with_cfg(cfg: &AppConfig, json_mode: bool) -> Result<()> {
    use std::io::IsTerminal;

    let provider = cfg.llm.provider.trim().to_ascii_lowercase();
    if provider != "deepseek" && provider != "openai" {
        return Err(anyhow!(
            "unsupported llm.provider='{}' (supported: deepseek, openai)",
            cfg.llm.provider
        ));
    }
    if cfg.llm.offline_fallback {
        return Ok(());
    }

    let env_key = cfg.llm.api_key_env.trim();
    if env_key.is_empty() {
        return Err(anyhow!(
            "llm.api_key_env is empty; set it in .deepseek/settings.json"
        ));
    }

    if std::env::var(env_key)
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
    {
        return Ok(());
    }

    let interactive_tty = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    if json_mode || !interactive_tty {
        return Err(anyhow!(
            "{} is required when llm.offline_fallback=false. Set it and retry.",
            env_key
        ));
    }

    eprintln!(
        "API key is required to use provider '{}' (offline fallback is disabled).",
        cfg.llm.provider
    );
    let prompt = format!("Enter {}: ", env_key);
    let key = rpassword::prompt_password(prompt)?;
    let trimmed = key.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("received empty API key"));
    }
    // SAFETY: We set process-local environment for this CLI process before worker threads start.
    unsafe {
        std::env::set_var(env_key, trimmed);
    }
    Ok(())
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
    pause_file: Option<String>,
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
    ensure_llm_ready(cwd, json_mode)?;

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
    let pause_file = args
        .pause_file
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| autopilot_pause_path(&stop_file));
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
            "autopilot started: tools={} max_think={} runtime={} max_iterations={:?} stop_file={} pause_file={} heartbeat_file={}",
            args.tools,
            args.max_think,
            runtime,
            args.max_iterations,
            stop_file.display(),
            pause_file.display(),
            heartbeat_file.display(),
        );
    }

    let mut completed_iterations = 0_u64;
    let mut failed_iterations = 0_u64;
    let mut consecutive_failures = 0_u64;
    let mut last_error: Option<String> = None;
    let mut paused_state = false;

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
            "pause_file": pause_file,
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
        if pause_file.exists() {
            if !paused_state {
                paused_state = true;
                store.upsert_autopilot_run(&AutopilotRunRecord {
                    run_id,
                    session_id: session.session_id,
                    prompt: args.prompt.clone(),
                    status: "paused".to_string(),
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
                store.upsert_background_job(&BackgroundJobRecord {
                    job_id: run_id,
                    kind: "autopilot".to_string(),
                    reference: run_id.to_string(),
                    status: "paused".to_string(),
                    metadata_json: serde_json::json!({
                        "completed_iterations": completed_iterations,
                        "failed_iterations": failed_iterations,
                        "last_error": last_error.clone(),
                        "pause_file": pause_file,
                    })
                    .to_string(),
                    started_at: started_at.clone(),
                    updated_at: Utc::now().to_rfc3339(),
                })?;
                if !json_mode {
                    println!(
                        "autopilot paused; remove {} to continue",
                        pause_file.display()
                    );
                }
            }
            write_autopilot_heartbeat(
                &heartbeat_file,
                &json!({
                    "run_id": run_id,
                    "status": "paused",
                    "at": Utc::now().to_rfc3339(),
                    "completed_iterations": completed_iterations,
                    "failed_iterations": failed_iterations,
                    "consecutive_failures": consecutive_failures,
                    "last_error": last_error,
                    "pause_file": pause_file,
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
            thread::sleep(Duration::from_secs(1));
            continue;
        }
        if paused_state {
            paused_state = false;
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
            store.upsert_background_job(&BackgroundJobRecord {
                job_id: run_id,
                kind: "autopilot".to_string(),
                reference: run_id.to_string(),
                status: "running".to_string(),
                metadata_json: serde_json::json!({
                    "completed_iterations": completed_iterations,
                    "failed_iterations": failed_iterations,
                    "last_error": last_error.clone(),
                })
                .to_string(),
                started_at: started_at.clone(),
                updated_at: Utc::now().to_rfc3339(),
            })?;
            if !json_mode {
                println!("autopilot resumed");
            }
        }

        let iteration_no = completed_iterations + failed_iterations + 1;
        if !json_mode {
            println!("autopilot iteration {iteration_no}");
        }

        match engine.run_once_with_mode_and_priority(&args.prompt, args.tools, args.max_think, true)
        {
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
        "pause_file": pause_file,
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
    if args.follow {
        let max_samples = args.samples.unwrap_or(10).max(1);
        let interval = Duration::from_secs(args.interval_seconds.max(1));
        let mut samples = Vec::new();
        let mut current = run;
        for idx in 0..max_samples {
            let mut snapshot = autopilot_status_payload(&current);
            if let Some(obj) = snapshot.as_object_mut() {
                obj.insert("sample_index".to_string(), json!(idx + 1));
                obj.insert("sampled_at".to_string(), json!(Utc::now().to_rfc3339()));
            }
            samples.push(snapshot.clone());
            if !current.status.eq_ignore_ascii_case("running") {
                break;
            }
            if idx + 1 >= max_samples {
                break;
            }
            thread::sleep(interval);
            current = store
                .load_autopilot_run(current.run_id)?
                .unwrap_or_else(|| current.clone());
        }
        let payload = json!({
            "run_id": current.run_id,
            "follow": true,
            "interval_seconds": args.interval_seconds.max(1),
            "samples_collected": samples.len(),
            "samples": samples,
        });
        if json_mode {
            print_json(&payload)?;
        } else {
            for sample in payload["samples"].as_array().into_iter().flatten() {
                println!(
                    "sample#{} at={} status={} completed={} failed={} paused={}",
                    sample["sample_index"].as_u64().unwrap_or(0),
                    sample["sampled_at"].as_str().unwrap_or_default(),
                    sample["status"].as_str().unwrap_or_default(),
                    sample["completed_iterations"].as_u64().unwrap_or(0),
                    sample["failed_iterations"].as_u64().unwrap_or(0),
                    sample["paused"].as_bool().unwrap_or(false),
                );
            }
        }
    } else {
        let payload = autopilot_status_payload(&run);
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
            println!("paused={}", payload["paused"].as_bool().unwrap_or(false));
            if let Some(reason) = run.stop_reason {
                println!("stop_reason={reason}");
            }
            if let Some(err) = run.last_error {
                println!("last_error={err}");
            }
            println!("stop_file={}", run.stop_file);
            println!(
                "pause_file={}",
                payload["pause_file"].as_str().unwrap_or_default()
            );
            println!("heartbeat_file={}", run.heartbeat_file);
        }
    }
    Ok(())
}

fn autopilot_status_payload(run: &AutopilotRunRecord) -> serde_json::Value {
    let heartbeat = if run.heartbeat_file.is_empty() {
        None
    } else {
        fs::read_to_string(&run.heartbeat_file)
            .ok()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
    };
    let pause_file = autopilot_pause_path(&PathBuf::from(&run.stop_file));
    let paused = pause_file.exists();
    json!({
        "run_id": run.run_id,
        "session_id": run.session_id,
        "status": run.status,
        "paused": paused,
        "stop_reason": run.stop_reason,
        "completed_iterations": run.completed_iterations,
        "failed_iterations": run.failed_iterations,
        "consecutive_failures": run.consecutive_failures,
        "last_error": run.last_error,
        "stop_file": run.stop_file,
        "pause_file": pause_file,
        "heartbeat_file": run.heartbeat_file,
        "tools": run.tools,
        "max_think": run.max_think,
        "heartbeat": heartbeat,
    })
}

fn run_autopilot_pause(cwd: &Path, args: AutopilotPauseArgs, json_mode: bool) -> Result<()> {
    let store = Store::new(cwd)?;
    let run = find_autopilot_run(&store, args.run_id.as_deref())?
        .ok_or_else(|| anyhow!("no autopilot runs found"))?;
    let pause_path = autopilot_pause_path(&PathBuf::from(&run.stop_file));
    if let Some(parent) = pause_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &pause_path,
        format!("pause requested at {}\n", Utc::now().to_rfc3339()),
    )?;
    if json_mode {
        print_json(&json!({
            "run_id": run.run_id,
            "pause_requested": true,
            "pause_file": pause_path,
        }))?;
    } else {
        println!(
            "pause requested for run {} via {}",
            run.run_id,
            pause_path.display()
        );
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
    let pause_path = autopilot_pause_path(&stop_path);
    if pause_path.exists() {
        let _ = fs::remove_file(&pause_path);
    }
    if json_mode {
        print_json(&json!({
            "run_id": run.run_id,
            "stop_requested": true,
            "stop_file": stop_path,
            "pause_file": pause_path,
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
    let pause_path = autopilot_pause_path(&PathBuf::from(&run.stop_file));
    if pause_path.exists() {
        fs::remove_file(&pause_path)?;
        if run.status == "running" || run.status == "paused" {
            if json_mode {
                print_json(&json!({
                    "run_id": run.run_id,
                    "resumed_live": true,
                    "pause_file": pause_path,
                }))?;
            } else {
                println!("live resume requested for run {}", run.run_id);
            }
            return Ok(());
        }
    }
    if run.status == "running" {
        return Err(anyhow!(
            "autopilot run is already marked as running (no pause file present)"
        ));
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
            pause_file: None,
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

fn autopilot_pause_path(stop_file: &Path) -> PathBuf {
    if let Some(ext) = stop_file.extension()
        && ext == "stop"
    {
        return stop_file.with_extension("pause");
    }
    let file_name = stop_file
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| format!("{name}.pause"))
        .unwrap_or_else(|| "autopilot.pause".to_string());
    stop_file.with_file_name(file_name)
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
    use std::io::{IsTerminal, Write, stdin, stdout};

    let cfg = AppConfig::ensure(cwd)?;
    ensure_llm_ready_with_cfg(&cfg, json_mode)?;
    let engine = AgentEngine::new(cwd)?;
    let interactive_tty = stdin().is_terminal() && stdout().is_terminal();
    if !json_mode && (force_tui || cfg.ui.enable_tui) && interactive_tty {
        return run_chat_tui(cwd, allow_tools, &cfg);
    }
    if force_tui && !interactive_tty {
        return Err(anyhow!("--tui requires an interactive terminal"));
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
                            "/skills",
                            "/permissions",
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
    let bindings = load_tui_keybindings(cwd, cfg);
    run_tui_shell_with_bindings(status, bindings, |prompt| {
        if let Some(cmd) = SlashCommand::parse(prompt) {
            let out = match cmd {
                SlashCommand::Help => "commands: /help /init /clear /compact /memory /config /model /cost /mcp /rewind /export /plan /teleport /remote-env /status /effort /skills /permissions".to_string(),
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
                    let status = current_ui_status(cwd, cfg, force_max_think)?;
                    render_statusline(&status)
                }
                SlashCommand::Effort(level) => {
                    let level = level.unwrap_or_else(|| "medium".to_string());
                    let normalized = level.to_ascii_lowercase();
                    force_max_think = matches!(normalized.as_str(), "high" | "max");
                    format!("effort={} force_max_think={}", normalized, force_max_think)
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
                SlashCommand::Unknown { name, .. } => format!("unknown slash command: /{name}"),
            };
            return Ok(out);
        }
        engine.run_once_with_mode(prompt, allow_tools, force_max_think)
    })
}

fn load_tui_keybindings(cwd: &Path, cfg: &AppConfig) -> KeyBindings {
    let mut candidates = Vec::new();
    if !cfg.ui.keybindings_path.trim().is_empty() {
        candidates.push(PathBuf::from(expand_tilde(&cfg.ui.keybindings_path)));
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

#[derive(Default)]
struct TeleportExecution {
    bundle_id: Option<Uuid>,
    path: Option<String>,
    imported: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct CompactSummary {
    summary_id: Uuid,
    from_turn: u64,
    to_turn: u64,
    token_delta_estimate: i64,
}

fn parse_teleport_args(args: Vec<String>) -> Result<TeleportArgs> {
    if args.is_empty() {
        return Ok(TeleportArgs::default());
    }
    if args.len() >= 2 && args[0].eq_ignore_ascii_case("import") {
        return Ok(TeleportArgs {
            session_id: None,
            output: None,
            import: Some(args[1].clone()),
        });
    }
    let mut parsed = TeleportArgs::default();
    let mut idx = 0usize;
    while idx < args.len() {
        let token = &args[idx];
        if token.eq_ignore_ascii_case("session") && idx + 1 < args.len() {
            parsed.session_id = Some(args[idx + 1].clone());
            idx += 2;
            continue;
        }
        if token.eq_ignore_ascii_case("output") && idx + 1 < args.len() {
            parsed.output = Some(args[idx + 1].clone());
            idx += 2;
            continue;
        }
        if token.eq_ignore_ascii_case("import") && idx + 1 < args.len() {
            parsed.import = Some(args[idx + 1].clone());
            idx += 2;
            continue;
        }
        if parsed.output.is_none() {
            parsed.output = Some(token.clone());
        }
        idx += 1;
    }
    Ok(parsed)
}

fn parse_remote_env_cmd(args: Vec<String>) -> Result<RemoteEnvCmd> {
    if args.is_empty() || args[0].eq_ignore_ascii_case("list") {
        return Ok(RemoteEnvCmd::List);
    }
    let sub = args[0].to_ascii_lowercase();
    match sub.as_str() {
        "add" => {
            if args.len() < 3 {
                return Err(anyhow!(
                    "usage: /remote-env add <name> <endpoint> [auth_mode]"
                ));
            }
            Ok(RemoteEnvCmd::Add(RemoteEnvAddArgs {
                name: args[1].clone(),
                endpoint: args[2].clone(),
                auth_mode: args.get(3).cloned().unwrap_or_else(|| "token".to_string()),
            }))
        }
        "remove" => {
            if args.len() < 2 {
                return Err(anyhow!("usage: /remote-env remove <profile_id>"));
            }
            Ok(RemoteEnvCmd::Remove(RemoteEnvRemoveArgs {
                profile_id: args[1].clone(),
            }))
        }
        "check" => {
            if args.len() < 2 {
                return Err(anyhow!("usage: /remote-env check <profile_id>"));
            }
            Ok(RemoteEnvCmd::Check(RemoteEnvCheckArgs {
                profile_id: args[1].clone(),
            }))
        }
        _ => Err(anyhow!("unknown /remote-env subcommand: {sub}")),
    }
}

fn parse_permissions_cmd(args: Vec<String>) -> Result<PermissionsCmd> {
    if args.is_empty() || args[0].eq_ignore_ascii_case("show") {
        return Ok(PermissionsCmd::Show);
    }

    let first = args[0].to_ascii_lowercase();
    if first == "bash" {
        let mode = args
            .get(1)
            .ok_or_else(|| anyhow!("usage: /permissions bash <ask|always|never>"))?;
        return Ok(PermissionsCmd::Set(PermissionsSetArgs {
            approve_bash: Some(parse_permission_mode(mode)?),
            ..PermissionsSetArgs::default()
        }));
    }
    if first == "edits" {
        let mode = args
            .get(1)
            .ok_or_else(|| anyhow!("usage: /permissions edits <ask|always|never>"))?;
        return Ok(PermissionsCmd::Set(PermissionsSetArgs {
            approve_edits: Some(parse_permission_mode(mode)?),
            ..PermissionsSetArgs::default()
        }));
    }
    if first == "sandbox" {
        let mode = args
            .get(1)
            .ok_or_else(|| anyhow!("usage: /permissions sandbox <mode>"))?;
        return Ok(PermissionsCmd::Set(PermissionsSetArgs {
            sandbox_mode: Some(mode.clone()),
            ..PermissionsSetArgs::default()
        }));
    }
    if first == "allow" {
        let entry = args
            .iter()
            .skip(1)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(" ");
        if entry.trim().is_empty() {
            return Err(anyhow!("usage: /permissions allow <command-prefix>"));
        }
        return Ok(PermissionsCmd::Set(PermissionsSetArgs {
            allow: vec![entry],
            ..PermissionsSetArgs::default()
        }));
    }
    if first == "clear-allowlist" {
        return Ok(PermissionsCmd::Set(PermissionsSetArgs {
            clear_allowlist: true,
            ..PermissionsSetArgs::default()
        }));
    }

    if first != "set" {
        return Err(anyhow!(
            "use /permissions show|set|bash|edits|sandbox|allow|clear-allowlist"
        ));
    }

    let mut parsed = PermissionsSetArgs::default();
    let mut idx = 1usize;
    while idx < args.len() {
        let token = args[idx].to_ascii_lowercase();
        match token.as_str() {
            "--approve-bash" | "approve-bash" => {
                idx += 1;
                let value = args.get(idx).ok_or_else(|| {
                    anyhow!("usage: /permissions set --approve-bash <ask|always|never>")
                })?;
                parsed.approve_bash = Some(parse_permission_mode(value)?);
            }
            "--approve-edits" | "approve-edits" => {
                idx += 1;
                let value = args.get(idx).ok_or_else(|| {
                    anyhow!("usage: /permissions set --approve-edits <ask|always|never>")
                })?;
                parsed.approve_edits = Some(parse_permission_mode(value)?);
            }
            "--sandbox-mode" | "sandbox-mode" => {
                idx += 1;
                let value = args
                    .get(idx)
                    .ok_or_else(|| anyhow!("usage: /permissions set --sandbox-mode <mode>"))?;
                parsed.sandbox_mode = Some(value.clone());
            }
            "--allow" | "allow" => {
                idx += 1;
                let value = args
                    .get(idx)
                    .ok_or_else(|| anyhow!("usage: /permissions set --allow <command-prefix>"))?;
                parsed.allow.push(value.clone());
            }
            "--clear-allowlist" | "clear-allowlist" => {
                parsed.clear_allowlist = true;
            }
            _ => {
                return Err(anyhow!(
                    "unknown permissions option: {} (expected --approve-bash/--approve-edits/--sandbox-mode/--allow/--clear-allowlist)",
                    args[idx]
                ));
            }
        }
        idx += 1;
    }

    Ok(PermissionsCmd::Set(parsed))
}

fn parse_permission_mode(value: &str) -> Result<PermissionModeArg> {
    match value.to_ascii_lowercase().as_str() {
        "ask" => Ok(PermissionModeArg::Ask),
        "always" => Ok(PermissionModeArg::Always),
        "never" => Ok(PermissionModeArg::Never),
        _ => Err(anyhow!(
            "invalid permission mode '{}' (expected ask|always|never)",
            value
        )),
    }
}

fn compact_now(cwd: &Path, from_turn: Option<u64>) -> Result<CompactSummary> {
    let store = Store::new(cwd)?;
    let session = store
        .load_latest_session()?
        .ok_or_else(|| anyhow!("no session found to compact"))?;
    let projection = store.rebuild_from_events(session.session_id)?;
    if projection.transcript.is_empty() {
        return Err(anyhow!("no transcript to compact"));
    }
    let from_turn = from_turn.unwrap_or(1).max(1);
    let transcript_len = projection.transcript.len() as u64;
    if from_turn > transcript_len {
        return Err(anyhow!(
            "from_turn {} exceeds transcript length {}",
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
    let token_delta_estimate = before_tokens as i64 - estimate_tokens(&summary) as i64;
    let replay_pointer = format!(".deepseek/compactions/{summary_id}.md");
    let summary_path = cwd.join(&replay_pointer);
    if let Some(parent) = summary_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(summary_path, summary)?;
    append_control_event(
        cwd,
        EventKind::ContextCompactedV1 {
            summary_id,
            from_turn,
            to_turn: transcript_len,
            token_delta_estimate,
            replay_pointer,
        },
    )?;
    Ok(CompactSummary {
        summary_id,
        from_turn,
        to_turn: transcript_len,
        token_delta_estimate,
    })
}

fn rewind_now(
    cwd: &Path,
    to_checkpoint: Option<String>,
) -> Result<deepseek_store::CheckpointRecord> {
    let memory = MemoryManager::new(cwd)?;
    let checkpoint_id = if let Some(value) = to_checkpoint.as_deref() {
        Uuid::parse_str(value)?
    } else {
        memory
            .list_checkpoints()?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no checkpoints available"))?
            .checkpoint_id
    };
    let checkpoint = memory.rewind_to_checkpoint(checkpoint_id)?;
    append_control_event(
        cwd,
        EventKind::CheckpointRewoundV1 {
            checkpoint_id: checkpoint.checkpoint_id,
            reason: checkpoint.reason.clone(),
        },
    )?;
    Ok(checkpoint)
}

fn teleport_now(cwd: &Path, args: TeleportArgs) -> Result<TeleportExecution> {
    if let Some(import_path) = args.import {
        let raw = fs::read_to_string(&import_path)?;
        let _: serde_json::Value = serde_json::from_str(&raw)?;
        return Ok(TeleportExecution {
            imported: Some(import_path),
            ..TeleportExecution::default()
        });
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
    Ok(TeleportExecution {
        bundle_id: Some(bundle_id),
        path: Some(output_path.to_string_lossy().to_string()),
        imported: None,
    })
}

fn remote_env_now(cwd: &Path, cmd: RemoteEnvCmd) -> Result<serde_json::Value> {
    let store = Store::new(cwd)?;
    match cmd {
        RemoteEnvCmd::List => {
            let profiles = store.list_remote_env_profiles()?;
            Ok(json!({"profiles": profiles}))
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
            Ok(json!({"profile_id": profile_id, "configured": true}))
        }
        RemoteEnvCmd::Remove(args) => {
            let profile_id = Uuid::parse_str(&args.profile_id)?;
            store.remove_remote_env_profile(profile_id)?;
            Ok(json!({"profile_id": profile_id, "removed": true}))
        }
        RemoteEnvCmd::Check(args) => {
            let profile_id = Uuid::parse_str(&args.profile_id)?;
            let profile = store
                .load_remote_env_profile(profile_id)?
                .ok_or_else(|| anyhow!("remote profile not found: {}", profile_id))?;
            Ok(json!({
                "profile_id": profile.profile_id,
                "name": profile.name,
                "endpoint": profile.endpoint,
                "auth_mode": profile.auth_mode,
                "reachable": true,
            }))
        }
    }
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

fn run_profile(cwd: &Path, args: ProfileArgs, json_mode: bool) -> Result<()> {
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

    let payload = if args.benchmark {
        ensure_llm_ready_with_cfg(&cfg, json_mode)?;
        let engine = AgentEngine::new(cwd)?;
        let suite_path = args.benchmark_suite.as_deref().map(Path::new);
        if suite_path.is_some() && args.benchmark_pack.is_some() {
            return Err(anyhow!(
                "use either --benchmark-suite or --benchmark-pack, not both"
            ));
        }
        let bench = run_profile_benchmark(
            &engine,
            args.benchmark_cases.max(1),
            args.benchmark_seed,
            suite_path,
            args.benchmark_pack.as_deref(),
            cwd,
            &args.benchmark_signing_key_env,
        );
        match bench {
            Ok(bench) => {
                if let Some(path) = args.benchmark_output.as_deref() {
                    let output_path = PathBuf::from(path);
                    if let Some(parent) = output_path.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    fs::write(&output_path, serde_json::to_vec_pretty(&bench)?)?;
                }
                if let Some(min_success_rate) = args.benchmark_min_success_rate {
                    let executed = bench
                        .get("executed_cases")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as f64;
                    let succeeded =
                        bench.get("succeeded").and_then(|v| v.as_u64()).unwrap_or(0) as f64;
                    let success_rate = if executed <= f64::EPSILON {
                        0.0
                    } else {
                        succeeded / executed
                    };
                    if success_rate < min_success_rate {
                        return Err(anyhow!(
                            "benchmark success rate {:.3} below required minimum {:.3}",
                            success_rate,
                            min_success_rate
                        ));
                    }
                }
                if let Some(max_p95_ms) = args.benchmark_max_p95_ms {
                    let p95 = bench
                        .get("p95_latency_ms")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(u64::MAX);
                    if p95 > max_p95_ms {
                        return Err(anyhow!(
                            "benchmark p95 latency {}ms above allowed maximum {}ms",
                            p95,
                            max_p95_ms
                        ));
                    }
                }
                if let Some(min_quality_rate) = args.benchmark_min_quality_rate {
                    let quality_rate = benchmark_quality_rate(&bench);
                    if quality_rate < min_quality_rate {
                        return Err(anyhow!(
                            "benchmark quality rate {:.3} below required minimum {:.3}",
                            quality_rate,
                            min_quality_rate
                        ));
                    }
                }

                let baseline_comparison = if let Some(path) = args.benchmark_baseline.as_deref() {
                    let baseline_raw = fs::read_to_string(path)?;
                    let baseline_value: serde_json::Value = serde_json::from_str(&baseline_raw)?;
                    let baseline_bench = baseline_value.get("benchmark").unwrap_or(&baseline_value);
                    let comparison = compare_benchmark_runs(&bench, baseline_bench)?;
                    if let Some(max_regression_ms) = args.benchmark_max_regression_ms
                        && comparison
                            .get("p95_regression_ms")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0)
                            > max_regression_ms
                    {
                        return Err(anyhow!(
                            "benchmark p95 regression {}ms above allowed maximum {}ms",
                            comparison
                                .get("p95_regression_ms")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0),
                            max_regression_ms
                        ));
                    }
                    Some(comparison)
                } else {
                    None
                };
                let peer_comparison = if args.benchmark_compare.is_empty() {
                    None
                } else {
                    Some(compare_benchmark_with_peers(
                        &bench,
                        &args.benchmark_compare,
                    )?)
                };

                let mut object = payload.as_object().cloned().unwrap_or_default();
                object.insert("benchmark".to_string(), bench.clone());
                if let Some(comparison) = baseline_comparison {
                    object.insert("benchmark_comparison".to_string(), comparison);
                }
                if let Some(comparison) = peer_comparison {
                    object.insert("benchmark_peer_comparison".to_string(), comparison);
                }
                serde_json::Value::Object(object)
            }
            Err(err) => {
                let mut object = payload.as_object().cloned().unwrap_or_default();
                object.insert(
                    "benchmark".to_string(),
                    json!({
                        "ok": false,
                        "error": err.to_string(),
                    }),
                );
                serde_json::Value::Object(object)
            }
        }
    } else {
        payload
    };

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

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BenchmarkCase {
    #[serde(default)]
    case_id: String,
    prompt: String,
    #[serde(default)]
    expected_keywords: Vec<String>,
    #[serde(default)]
    min_steps: Option<usize>,
    #[serde(default)]
    min_verification_steps: Option<usize>,
}

fn built_in_benchmark_cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase {
            case_id: "refactor-plan".to_string(),
            prompt: "Plan a safe refactor for a router module with rollback strategy.".to_string(),
            expected_keywords: vec!["refactor".to_string(), "verify".to_string()],
            min_steps: Some(2),
            min_verification_steps: Some(1),
        },
        BenchmarkCase {
            case_id: "test-triage".to_string(),
            prompt: "Create a stepwise plan to investigate flaky tests and isolate nondeterminism."
                .to_string(),
            expected_keywords: vec!["test".to_string()],
            min_steps: Some(2),
            min_verification_steps: Some(1),
        },
        BenchmarkCase {
            case_id: "security-hardening".to_string(),
            prompt:
                "Plan command-injection hardening for shell execution with verification criteria."
                    .to_string(),
            expected_keywords: vec!["verify".to_string(), "bash".to_string()],
            min_steps: Some(2),
            min_verification_steps: Some(1),
        },
        BenchmarkCase {
            case_id: "git-recovery".to_string(),
            prompt: "Plan recovery from a failed merge with conflict-resolution checkpoints."
                .to_string(),
            expected_keywords: vec!["git".to_string()],
            min_steps: Some(2),
            min_verification_steps: Some(1),
        },
        BenchmarkCase {
            case_id: "index-performance".to_string(),
            prompt: "Plan improvements for index query latency and deterministic cache behavior."
                .to_string(),
            expected_keywords: vec!["index".to_string()],
            min_steps: Some(2),
            min_verification_steps: Some(1),
        },
        BenchmarkCase {
            case_id: "mcp-reliability".to_string(),
            prompt: "Plan reliability improvements for MCP tool discovery and reconnect flows."
                .to_string(),
            expected_keywords: vec!["mcp".to_string()],
            min_steps: Some(2),
            min_verification_steps: Some(1),
        },
        BenchmarkCase {
            case_id: "autopilot-ops".to_string(),
            prompt: "Plan operational guardrails for long-running autopilot loops.".to_string(),
            expected_keywords: vec!["autopilot".to_string()],
            min_steps: Some(2),
            min_verification_steps: Some(1),
        },
        BenchmarkCase {
            case_id: "release-readiness".to_string(),
            prompt: "Plan release readiness checks across tests, lint, and packaging artifacts."
                .to_string(),
            expected_keywords: vec!["verify".to_string()],
            min_steps: Some(2),
            min_verification_steps: Some(1),
        },
    ]
}

fn load_benchmark_cases(path: &Path) -> Result<Vec<BenchmarkCase>> {
    let raw = fs::read_to_string(path)?;
    let mut cases = if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("jsonl"))
    {
        let mut parsed = Vec::new();
        for (line_no, line) in raw.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let case: BenchmarkCase = serde_json::from_str(trimmed).map_err(|err| {
                anyhow!(
                    "invalid benchmark case at {}:{}: {err}",
                    path.display(),
                    line_no + 1
                )
            })?;
            parsed.push(case);
        }
        parsed
    } else {
        let value: serde_json::Value = serde_json::from_str(&raw)?;
        let items = if let Some(arr) = value.as_array() {
            arr.clone()
        } else if let Some(arr) = value.get("cases").and_then(|v| v.as_array()) {
            arr.clone()
        } else {
            return Err(anyhow!(
                "benchmark suite must be a JSON array or an object with `cases` array"
            ));
        };
        let mut parsed = Vec::new();
        for (idx, item) in items.into_iter().enumerate() {
            let case: BenchmarkCase = serde_json::from_value(item).map_err(|err| {
                anyhow!(
                    "invalid benchmark case at {}:{}: {err}",
                    path.display(),
                    idx + 1
                )
            })?;
            parsed.push(case);
        }
        parsed
    };

    for (idx, case) in cases.iter_mut().enumerate() {
        case.prompt = case.prompt.trim().to_string();
        case.expected_keywords = case
            .expected_keywords
            .iter()
            .map(|kw| kw.trim().to_ascii_lowercase())
            .filter(|kw| !kw.is_empty())
            .collect();
        if case.case_id.trim().is_empty() {
            case.case_id = format!("case-{}", idx + 1);
        }
    }
    cases.retain(|case| !case.prompt.is_empty());
    if cases.is_empty() {
        return Err(anyhow!("benchmark suite has no valid cases"));
    }
    Ok(cases)
}

fn built_in_benchmark_pack(name: &str) -> Option<(&'static str, Vec<BenchmarkCase>)> {
    match name.to_ascii_lowercase().as_str() {
        "core" => Some(("builtin-core", built_in_benchmark_cases())),
        "smoke" => Some((
            "builtin-smoke",
            built_in_benchmark_cases().into_iter().take(3).collect(),
        )),
        "ops" => Some((
            "builtin-ops",
            built_in_benchmark_cases()
                .into_iter()
                .filter(|case| {
                    matches!(
                        case.case_id.as_str(),
                        "mcp-reliability" | "autopilot-ops" | "release-readiness"
                    )
                })
                .collect(),
        )),
        _ => None,
    }
}

fn benchmark_pack_dir(cwd: &Path) -> PathBuf {
    runtime_dir(cwd).join("benchmark-packs")
}

fn sanitize_pack_name(name: &str) -> String {
    let sanitized = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string();
    if sanitized.is_empty() {
        "pack".to_string()
    } else {
        sanitized
    }
}

fn local_benchmark_pack_path(cwd: &Path, name: &str) -> PathBuf {
    benchmark_pack_dir(cwd).join(format!("{}.json", sanitize_pack_name(name)))
}

fn load_benchmark_pack(cwd: &Path, name: &str) -> Result<serde_json::Value> {
    if let Some((corpus_id, cases)) = built_in_benchmark_pack(name) {
        return Ok(json!({
            "name": name,
            "kind": "builtin",
            "source": corpus_id,
            "cases": cases,
        }));
    }
    let path = local_benchmark_pack_path(cwd, name);
    if !path.exists() {
        return Err(anyhow!("benchmark pack not found: {}", name));
    }
    let raw = fs::read_to_string(&path)?;
    let mut value: serde_json::Value = serde_json::from_str(&raw)?;
    if value.get("name").is_none() {
        value["name"] = json!(name);
    }
    if value.get("kind").is_none() {
        value["kind"] = json!("imported");
    }
    if value.get("source").is_none() {
        value["source"] = json!(path.display().to_string());
    }
    Ok(value)
}

fn load_benchmark_pack_cases(cwd: &Path, name: &str) -> Result<(String, Vec<BenchmarkCase>)> {
    if let Some((corpus_id, cases)) = built_in_benchmark_pack(name) {
        return Ok((corpus_id.to_string(), cases));
    }
    let pack = load_benchmark_pack(cwd, name)?;
    let cases_value = pack
        .get("cases")
        .and_then(|v| v.as_array())
        .cloned()
        .ok_or_else(|| anyhow!("benchmark pack {} missing cases array", name))?;
    let mut cases = Vec::new();
    for (idx, item) in cases_value.into_iter().enumerate() {
        let case: BenchmarkCase = serde_json::from_value(item).map_err(|err| {
            anyhow!(
                "invalid benchmark case in pack {} at index {}: {}",
                name,
                idx + 1,
                err
            )
        })?;
        cases.push(case);
    }
    let corpus_id = pack
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or(name)
        .to_string();
    Ok((corpus_id, cases))
}

fn list_benchmark_packs(cwd: &Path) -> Result<Vec<serde_json::Value>> {
    let mut out = Vec::new();
    for (name, source) in [
        ("core", "builtin-core"),
        ("smoke", "builtin-smoke"),
        ("ops", "builtin-ops"),
    ] {
        let count = built_in_benchmark_pack(name)
            .map(|(_, cases)| cases.len())
            .unwrap_or(0);
        out.push(json!({
            "name": name,
            "kind": "builtin",
            "source": source,
            "cases": count,
        }));
    }

    let dir = benchmark_pack_dir(cwd);
    if dir.exists() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            let raw = fs::read_to_string(&path)?;
            let value: serde_json::Value = serde_json::from_str(&raw).unwrap_or_default();
            let count = value
                .get("cases")
                .and_then(|v| v.as_array())
                .map(|rows| rows.len())
                .unwrap_or(0);
            let source = value
                .get("source")
                .and_then(|v| v.as_str())
                .map(ToString::to_string)
                .unwrap_or_else(|| path.to_string_lossy().to_string());
            out.push(json!({
                "name": value.get("name").and_then(|v| v.as_str()).unwrap_or(stem),
                "kind": value.get("kind").and_then(|v| v.as_str()).unwrap_or("imported"),
                "source": source,
                "cases": count,
            }));
        }
    }
    out.sort_by(|a, b| {
        a["name"]
            .as_str()
            .unwrap_or_default()
            .cmp(b["name"].as_str().unwrap_or_default())
    });
    Ok(out)
}

fn run_profile_benchmark(
    engine: &AgentEngine,
    requested_cases: usize,
    benchmark_seed: Option<u64>,
    suite_path: Option<&Path>,
    pack_name: Option<&str>,
    cwd: &Path,
    signing_key_env: &str,
) -> Result<serde_json::Value> {
    let (source_kind, cases, corpus_id) = if let Some(path) = suite_path {
        (
            "suite".to_string(),
            load_benchmark_cases(path)?,
            path.display().to_string(),
        )
    } else if let Some(pack_name) = pack_name {
        let (corpus_id, cases) = load_benchmark_pack_cases(cwd, pack_name)?;
        (format!("pack:{pack_name}"), cases, corpus_id)
    } else {
        (
            "builtin".to_string(),
            built_in_benchmark_cases(),
            "builtin".to_string(),
        )
    };

    if cases.is_empty() {
        return Err(anyhow!("benchmark suite is empty"));
    }

    let total = requested_cases.min(cases.len()).max(1);
    let seed =
        benchmark_seed.unwrap_or_else(|| derive_benchmark_seed(&corpus_id, requested_cases, total));
    let corpus_manifest =
        build_benchmark_manifest(&corpus_id, &source_kind, &cases, seed, signing_key_env);
    let mut selected_cases = select_benchmark_cases(cases, total, seed);
    let selected_case_ids = selected_cases
        .iter()
        .map(|case| case.case_id.clone())
        .collect::<Vec<_>>();
    let mut records = Vec::new();
    let mut latencies = Vec::new();
    let mut succeeded = 0usize;

    for case in selected_cases.drain(..) {
        let started = Instant::now();
        let result = engine.plan_only(&case.prompt);
        let elapsed_ms = started.elapsed().as_millis() as u64;
        latencies.push(elapsed_ms);
        match result {
            Ok(plan) => {
                let plan_text = serde_json::to_string(&plan)
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                let keyword_total = case.expected_keywords.len();
                let keyword_matches = case
                    .expected_keywords
                    .iter()
                    .filter(|kw| plan_text.contains(kw.as_str()))
                    .count();
                let min_steps = case.min_steps.unwrap_or(1);
                let min_verification_steps = case.min_verification_steps.unwrap_or(1);
                let min_steps_ok = plan.steps.len() >= min_steps;
                let min_verification_ok = plan.verification.len() >= min_verification_steps;
                let keywords_ok = keyword_total == 0 || keyword_matches == keyword_total;
                let case_ok = min_steps_ok && min_verification_ok && keywords_ok;
                if case_ok {
                    succeeded += 1;
                }
                records.push(json!({
                    "case_id": case.case_id,
                    "prompt_sha256": sha256_hex(case.prompt.as_bytes()),
                    "ok": case_ok,
                    "elapsed_ms": elapsed_ms,
                    "steps": plan.steps.len(),
                    "verification_steps": plan.verification.len(),
                    "min_steps_required": min_steps,
                    "min_verification_required": min_verification_steps,
                    "keyword_matches": keyword_matches,
                    "keyword_total": keyword_total,
                    "keywords_ok": keywords_ok,
                    "quality_ok": case_ok,
                }));
            }
            Err(err) => {
                records.push(json!({
                    "case_id": case.case_id,
                    "prompt_sha256": sha256_hex(case.prompt.as_bytes()),
                    "ok": false,
                    "elapsed_ms": elapsed_ms,
                    "error": err.to_string(),
                }));
            }
        }
    }

    let avg_ms = if latencies.is_empty() {
        0
    } else {
        latencies.iter().sum::<u64>() / (latencies.len() as u64)
    };
    latencies.sort_unstable();
    let p95_ms = if latencies.is_empty() {
        0
    } else {
        let idx = ((latencies.len() - 1) as f64 * 0.95).round() as usize;
        latencies[idx.min(latencies.len() - 1)]
    };
    let quality_passed = records
        .iter()
        .filter(|record| {
            if let Some(ok) = record.get("quality_ok").and_then(|v| v.as_bool()) {
                return ok;
            }
            record.get("ok").and_then(|v| v.as_bool()).unwrap_or(false)
        })
        .count();
    let quality_rate = if total == 0 {
        0.0
    } else {
        quality_passed as f64 / total as f64
    };
    let corpus_manifest_sha = corpus_manifest
        .get("manifest_sha256")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let execution_manifest = build_benchmark_execution_manifest(
        &corpus_manifest_sha,
        &selected_case_ids,
        seed,
        signing_key_env,
    );
    let scorecard = build_benchmark_scorecard(
        "deepseek-cli",
        &corpus_manifest_sha,
        &selected_case_ids,
        seed,
        succeeded as u64,
        total as u64,
        quality_rate,
        avg_ms,
        p95_ms,
        signing_key_env,
    );

    Ok(json!({
        "agent": "deepseek-cli",
        "generated_at": Utc::now().to_rfc3339(),
        "ok": succeeded == total,
        "suite": suite_path
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| source_kind.clone()),
        "pack": pack_name,
        "corpus_id": corpus_id,
        "seed": seed,
        "case_ids": selected_case_ids,
        "requested_cases": requested_cases,
        "executed_cases": total,
        "succeeded": succeeded,
        "failed": total - succeeded,
        "avg_latency_ms": avg_ms,
        "p95_latency_ms": p95_ms,
        "quality_rate": quality_rate,
        "corpus_manifest": corpus_manifest,
        "execution_manifest": execution_manifest,
        "scorecard": scorecard,
        "records": records,
    }))
}

fn derive_benchmark_seed(corpus_id: &str, requested_cases: usize, executed_cases: usize) -> u64 {
    let digest = sha256_hex(
        format!(
            "{}|requested:{}|executed:{}",
            corpus_id, requested_cases, executed_cases
        )
        .as_bytes(),
    );
    u64::from_str_radix(&digest[..16], 16).unwrap_or(0)
}

fn select_benchmark_cases(
    mut cases: Vec<BenchmarkCase>,
    total: usize,
    seed: u64,
) -> Vec<BenchmarkCase> {
    cases.sort_by(|a, b| {
        benchmark_case_rank(seed, a)
            .cmp(&benchmark_case_rank(seed, b))
            .then(a.case_id.cmp(&b.case_id))
    });
    cases.into_iter().take(total).collect()
}

fn benchmark_case_rank(seed: u64, case: &BenchmarkCase) -> String {
    sha256_hex(format!("{seed}|{}|{}", case.case_id, case.prompt).as_bytes())
}

fn build_benchmark_manifest(
    corpus_id: &str,
    source_kind: &str,
    cases: &[BenchmarkCase],
    seed: u64,
    signing_key_env: &str,
) -> serde_json::Value {
    let mut case_rows = cases
        .iter()
        .map(|case| {
            let mut keywords = case
                .expected_keywords
                .iter()
                .map(|kw| kw.trim().to_ascii_lowercase())
                .filter(|kw| !kw.is_empty())
                .collect::<Vec<_>>();
            keywords.sort();
            keywords.dedup();
            let constraints = json!({
                "expected_keywords": keywords,
                "min_steps": case.min_steps.unwrap_or(1),
                "min_verification_steps": case.min_verification_steps.unwrap_or(1),
            });
            json!({
                "case_id": case.case_id,
                "prompt_sha256": sha256_hex(case.prompt.as_bytes()),
                "constraints_sha256": hash_json_value(&constraints),
            })
        })
        .collect::<Vec<_>>();
    case_rows.sort_by(|a, b| {
        a.get("case_id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .cmp(
                b.get("case_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default(),
            )
    });
    let mut manifest = json!({
        "schema": "deepseek.benchmark.manifest.v1",
        "corpus_id": corpus_id,
        "source_kind": source_kind,
        "seed": seed,
        "case_count": case_rows.len(),
        "cases": case_rows,
    });
    let manifest_sha256 = hash_json_value(&manifest);
    let signature = sign_benchmark_hash(&manifest_sha256, signing_key_env);
    if let Some(object) = manifest.as_object_mut() {
        object.insert("manifest_sha256".to_string(), json!(manifest_sha256));
        object.insert("signature".to_string(), signature);
    }
    manifest
}

fn build_benchmark_execution_manifest(
    corpus_manifest_sha256: &str,
    case_ids: &[String],
    seed: u64,
    signing_key_env: &str,
) -> serde_json::Value {
    let mut manifest = json!({
        "schema": "deepseek.benchmark.execution.v1",
        "seed": seed,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "case_ids": case_ids,
        "case_count": case_ids.len(),
    });
    let manifest_sha256 = hash_json_value(&manifest);
    let signature = sign_benchmark_hash(&manifest_sha256, signing_key_env);
    if let Some(object) = manifest.as_object_mut() {
        object.insert("manifest_sha256".to_string(), json!(manifest_sha256));
        object.insert("signature".to_string(), signature);
    }
    manifest
}

#[allow(clippy::too_many_arguments)]
fn build_benchmark_scorecard(
    agent: &str,
    corpus_manifest_sha256: &str,
    case_ids: &[String],
    seed: u64,
    succeeded: u64,
    executed_cases: u64,
    quality_rate: f64,
    avg_latency_ms: u64,
    p95_latency_ms: u64,
    signing_key_env: &str,
) -> serde_json::Value {
    let mut scorecard = json!({
        "schema": "deepseek.benchmark.scorecard.v1",
        "agent": agent,
        "seed": seed,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "case_ids": case_ids,
        "executed_cases": executed_cases,
        "succeeded": succeeded,
        "success_rate": if executed_cases == 0 { 0.0 } else { succeeded as f64 / executed_cases as f64 },
        "quality_rate": quality_rate,
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
    });
    let scorecard_sha256 = hash_json_value(&scorecard);
    let signature = sign_benchmark_hash(&scorecard_sha256, signing_key_env);
    if let Some(object) = scorecard.as_object_mut() {
        object.insert("scorecard_sha256".to_string(), json!(scorecard_sha256));
        object.insert("signature".to_string(), signature);
    }
    scorecard
}

fn hash_json_value(value: &serde_json::Value) -> String {
    let canonical = canonicalize_json(value);
    let rendered = serde_json::to_string(&canonical).unwrap_or_default();
    sha256_hex(rendered.as_bytes())
}

fn canonicalize_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys = map.keys().cloned().collect::<Vec<_>>();
            keys.sort();
            let mut ordered = serde_json::Map::new();
            for key in keys {
                if let Some(item) = map.get(&key) {
                    ordered.insert(key, canonicalize_json(item));
                }
            }
            serde_json::Value::Object(ordered)
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(canonicalize_json).collect())
        }
        _ => value.clone(),
    }
}

fn sign_benchmark_hash(hash: &str, signing_key_env: &str) -> serde_json::Value {
    let key_env = signing_key_env.trim();
    if key_env.is_empty() {
        return json!({
            "algorithm": "none",
            "key_env": "",
            "present": false,
        });
    }
    let secret = std::env::var(key_env)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    if let Some(secret) = secret {
        let digest = sha256_hex(format!("{secret}:{hash}").as_bytes());
        return json!({
            "algorithm": "sha256-keyed",
            "key_env": key_env,
            "present": true,
            "digest": digest,
        });
    }
    json!({
        "algorithm": "none",
        "key_env": key_env,
        "present": false,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn benchmark_quality_rate(bench: &serde_json::Value) -> f64 {
    let records = bench
        .get("records")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if records.is_empty() {
        return 0.0;
    }
    let passed = records
        .iter()
        .filter(|record| {
            if let Some(ok) = record.get("quality_ok").and_then(|v| v.as_bool()) {
                return ok;
            }
            record.get("ok").and_then(|v| v.as_bool()).unwrap_or(false)
        })
        .count() as f64;
    passed / records.len() as f64
}

fn compare_benchmark_runs(
    current: &serde_json::Value,
    baseline: &serde_json::Value,
) -> Result<serde_json::Value> {
    let current_p95 = current
        .get("p95_latency_ms")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("current benchmark missing p95_latency_ms"))?;
    let baseline_p95 = baseline
        .get("p95_latency_ms")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("baseline benchmark missing p95_latency_ms"))?;
    let current_success = current
        .get("succeeded")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let current_total = current
        .get("executed_cases")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let baseline_success = baseline
        .get("succeeded")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let baseline_total = baseline
        .get("executed_cases")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let current_quality = benchmark_quality_rate(current);
    let baseline_quality = benchmark_quality_rate(baseline);
    Ok(json!({
        "baseline_p95_latency_ms": baseline_p95,
        "current_p95_latency_ms": current_p95,
        "p95_regression_ms": current_p95.saturating_sub(baseline_p95),
        "p95_improvement_ms": baseline_p95.saturating_sub(current_p95),
        "baseline_success_rate": if baseline_total == 0 { 0.0 } else { baseline_success as f64 / baseline_total as f64 },
        "current_success_rate": if current_total == 0 { 0.0 } else { current_success as f64 / current_total as f64 },
        "baseline_quality_rate": baseline_quality,
        "current_quality_rate": current_quality,
        "quality_delta": current_quality - baseline_quality,
    }))
}

#[derive(Debug, Clone)]
struct BenchmarkMetrics {
    agent: String,
    success_rate: f64,
    quality_rate: f64,
    p95_latency_ms: u64,
    executed_cases: u64,
    corpus_id: String,
    manifest_sha256: Option<String>,
    seed: Option<u64>,
    case_outcomes: HashMap<String, CaseOutcome>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseOutcome {
    ok: bool,
    quality_ok: bool,
    elapsed_ms: Option<u64>,
}

fn benchmark_metrics_from_value(
    value: &serde_json::Value,
    fallback_agent: &str,
) -> Result<BenchmarkMetrics> {
    let bench = value.get("benchmark").unwrap_or(value);
    let executed = bench
        .get("executed_cases")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("benchmark report missing executed_cases"))?;
    let succeeded = bench.get("succeeded").and_then(|v| v.as_u64()).unwrap_or(0);
    let p95_latency_ms = bench
        .get("p95_latency_ms")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("benchmark report missing p95_latency_ms"))?;
    let success_rate = if executed == 0 {
        0.0
    } else {
        succeeded as f64 / executed as f64
    };
    let quality_rate = benchmark_quality_rate(bench);
    let agent = value
        .get("agent")
        .and_then(|v| v.as_str())
        .or_else(|| bench.get("agent").and_then(|v| v.as_str()))
        .unwrap_or(fallback_agent)
        .to_string();
    let corpus_id = bench
        .get("corpus_id")
        .and_then(|v| v.as_str())
        .or_else(|| bench.get("suite").and_then(|v| v.as_str()))
        .unwrap_or("unknown")
        .to_string();
    let manifest_sha256 = bench
        .get("scorecard")
        .and_then(|v| v.get("corpus_manifest_sha256"))
        .and_then(|v| v.as_str())
        .or_else(|| {
            bench
                .get("corpus_manifest")
                .and_then(|v| v.get("manifest_sha256"))
                .and_then(|v| v.as_str())
        })
        .map(ToString::to_string);
    let seed = bench
        .get("scorecard")
        .and_then(|v| v.get("seed"))
        .and_then(|v| v.as_u64())
        .or_else(|| bench.get("seed").and_then(|v| v.as_u64()));
    let case_outcomes = benchmark_case_outcomes(bench);
    Ok(BenchmarkMetrics {
        agent,
        success_rate,
        quality_rate,
        p95_latency_ms,
        executed_cases: executed,
        corpus_id,
        manifest_sha256,
        seed,
        case_outcomes,
    })
}

fn benchmark_case_outcomes(bench: &serde_json::Value) -> HashMap<String, CaseOutcome> {
    let mut out = HashMap::new();
    for (idx, record) in bench
        .get("records")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .enumerate()
    {
        let case_id = record
            .get("case_id")
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("case-{}", idx + 1));
        let ok = record.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
        let quality_ok = record
            .get("quality_ok")
            .and_then(|v| v.as_bool())
            .unwrap_or(ok);
        let elapsed_ms = record.get("elapsed_ms").and_then(|v| v.as_u64());
        out.insert(
            case_id,
            CaseOutcome {
                ok,
                quality_ok,
                elapsed_ms,
            },
        );
    }
    out
}

fn compare_benchmark_with_peers(
    current_bench: &serde_json::Value,
    paths: &[String],
) -> Result<serde_json::Value> {
    let mut rows = Vec::new();
    rows.push(benchmark_metrics_from_value(
        &json!({"agent": "deepseek-cli", "benchmark": current_bench}),
        "deepseek-cli",
    )?);

    for path in paths {
        let raw = fs::read_to_string(path)?;
        let parsed: serde_json::Value = serde_json::from_str(&raw)?;
        let fallback_name = Path::new(path)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("peer");
        rows.push(benchmark_metrics_from_value(&parsed, fallback_name)?);
    }

    let mut corpus_warnings = Vec::new();
    let mut manifest_warnings = Vec::new();
    let mut seed_warnings = Vec::new();
    let canonical_corpus = rows
        .iter()
        .find(|row| row.agent == "deepseek-cli")
        .map(|row| row.corpus_id.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let canonical_manifest = rows
        .iter()
        .find(|row| row.agent == "deepseek-cli")
        .and_then(|row| row.manifest_sha256.clone());
    let canonical_seed = rows
        .iter()
        .find(|row| row.agent == "deepseek-cli")
        .and_then(|row| row.seed);
    for row in &rows {
        if row.corpus_id != canonical_corpus {
            corpus_warnings.push(format!(
                "agent {} corpus_id={} differs from deepseek-cli corpus_id={}",
                row.agent, row.corpus_id, canonical_corpus
            ));
        }
        if let Some(expected_manifest) = canonical_manifest.as_deref() {
            match row.manifest_sha256.as_deref() {
                Some(value) if value != expected_manifest => {
                    manifest_warnings.push(format!(
                        "agent {} manifest_sha256={} differs from deepseek-cli manifest_sha256={}",
                        row.agent, value, expected_manifest
                    ));
                }
                None => {
                    manifest_warnings.push(format!(
                        "agent {} missing manifest_sha256 for reproducible comparison",
                        row.agent
                    ));
                }
                _ => {}
            }
        } else if row.manifest_sha256.is_none() {
            manifest_warnings.push(format!(
                "agent {} missing manifest_sha256 for reproducible comparison",
                row.agent
            ));
        }
        if let Some(seed) = canonical_seed
            && row.seed.is_some_and(|value| value != seed)
        {
            seed_warnings.push(format!(
                "agent {} seed={} differs from deepseek-cli seed={}",
                row.agent,
                row.seed.unwrap_or_default(),
                seed
            ));
        }
    }

    let mut ranking = rows.clone();
    ranking.sort_by(|a, b| {
        b.quality_rate
            .partial_cmp(&a.quality_rate)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.success_rate
                    .partial_cmp(&a.success_rate)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(a.p95_latency_ms.cmp(&b.p95_latency_ms))
    });

    let current_rank = ranking
        .iter()
        .position(|row| row.agent == "deepseek-cli")
        .map(|idx| idx + 1)
        .unwrap_or(0);

    let mut case_ids = BTreeSet::new();
    for row in &rows {
        for case_id in row.case_outcomes.keys() {
            case_ids.insert(case_id.clone());
        }
    }
    let case_matrix = case_ids
        .into_iter()
        .map(|case_id| {
            let agents = rows
                .iter()
                .map(|row| {
                    let outcome = row.case_outcomes.get(&case_id).cloned();
                    json!({
                        "agent": row.agent.clone(),
                        "present": outcome.is_some(),
                        "ok": outcome.as_ref().map(|value| value.ok),
                        "quality_ok": outcome.as_ref().map(|value| value.quality_ok),
                        "elapsed_ms": outcome.and_then(|value| value.elapsed_ms),
                    })
                })
                .collect::<Vec<_>>();
            json!({
                "case_id": case_id,
                "agents": agents,
            })
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "current_rank": current_rank,
        "total_agents": ranking.len(),
        "corpus_id": canonical_corpus,
        "corpus_match_warnings": corpus_warnings,
        "ranking": ranking
            .into_iter()
            .map(|row| json!({
                "agent": row.agent,
                "quality_rate": row.quality_rate,
                "success_rate": row.success_rate,
                "p95_latency_ms": row.p95_latency_ms,
                "executed_cases": row.executed_cases,
                "corpus_id": row.corpus_id,
                "manifest_sha256": row.manifest_sha256,
                "seed": row.seed,
            }))
            .collect::<Vec<_>>(),
        "manifest_match_warnings": manifest_warnings,
        "seed_match_warnings": seed_warnings,
        "case_matrix": case_matrix,
    }))
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
                "base": cfg.llm.base_model,
                "max_think": cfg.llm.max_think_model,
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

fn run_benchmark(cwd: &Path, cmd: BenchmarkCmd, json_mode: bool) -> Result<()> {
    match cmd {
        BenchmarkCmd::ListPacks => {
            let packs = list_benchmark_packs(cwd)?;
            if json_mode {
                print_json(&json!(packs))?;
            } else if packs.is_empty() {
                println!("no benchmark packs found");
            } else {
                for pack in packs {
                    println!(
                        "{} ({}) cases={} source={}",
                        pack["name"].as_str().unwrap_or_default(),
                        pack["kind"].as_str().unwrap_or_default(),
                        pack["cases"].as_u64().unwrap_or(0),
                        pack["source"].as_str().unwrap_or_default(),
                    );
                }
            }
        }
        BenchmarkCmd::ShowPack(args) => {
            let pack = load_benchmark_pack(cwd, &args.name)?;
            if json_mode {
                print_json(&pack)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&pack)?);
            }
        }
        BenchmarkCmd::ImportPack(args) => {
            let source = Path::new(&args.source);
            let cases = load_benchmark_cases(source)?;
            let dir = benchmark_pack_dir(cwd);
            fs::create_dir_all(&dir)?;
            let destination = dir.join(format!("{}.json", sanitize_pack_name(&args.name)));
            let payload = json!({
                "name": args.name,
                "kind": "imported",
                "source": source.display().to_string(),
                "imported_at": Utc::now().to_rfc3339(),
                "cases": cases,
            });
            fs::write(&destination, serde_json::to_vec_pretty(&payload)?)?;
            if json_mode {
                print_json(&json!({
                    "imported": true,
                    "name": payload["name"],
                    "cases": payload["cases"].as_array().map(|rows| rows.len()).unwrap_or(0),
                    "path": destination,
                }))?;
            } else {
                println!(
                    "imported benchmark pack {} with {} cases",
                    payload["name"].as_str().unwrap_or_default(),
                    payload["cases"]
                        .as_array()
                        .map(|rows| rows.len())
                        .unwrap_or(0),
                );
            }
        }
    }
    Ok(())
}

fn run_permissions(cwd: &Path, cmd: PermissionsCmd, json_mode: bool) -> Result<()> {
    let payload = permissions_payload(cwd, cmd)?;
    if json_mode {
        print_json(&payload)?;
    } else {
        println!(
            "permissions: bash={} edits={} sandbox={} allowlist={}",
            payload["policy"]["approve_bash"]
                .as_str()
                .unwrap_or_default(),
            payload["policy"]["approve_edits"]
                .as_str()
                .unwrap_or_default(),
            payload["policy"]["sandbox_mode"]
                .as_str()
                .unwrap_or_default(),
            payload["policy"]["allowlist_entries"].as_u64().unwrap_or(0),
        );
        if payload["updated"].as_bool().unwrap_or(false) {
            println!(
                "updated project permissions at {}",
                AppConfig::config_path(cwd).display()
            );
        }
    }
    Ok(())
}

fn permissions_payload(cwd: &Path, cmd: PermissionsCmd) -> Result<serde_json::Value> {
    let mut cfg = AppConfig::ensure(cwd)?;
    let mut updated = false;
    match cmd {
        PermissionsCmd::Show => {}
        PermissionsCmd::Set(args) => {
            if let Some(mode) = args.approve_bash {
                let value = mode.as_str().to_string();
                if cfg.policy.approve_bash != value {
                    cfg.policy.approve_bash = value;
                    updated = true;
                }
            }
            if let Some(mode) = args.approve_edits {
                let value = mode.as_str().to_string();
                if cfg.policy.approve_edits != value {
                    cfg.policy.approve_edits = value;
                    updated = true;
                }
            }
            if let Some(mode) = args.sandbox_mode {
                let mode = mode.trim();
                if mode.is_empty() {
                    return Err(anyhow!("sandbox_mode cannot be empty"));
                }
                if cfg.policy.sandbox_mode != mode {
                    cfg.policy.sandbox_mode = mode.to_string();
                    updated = true;
                }
            }
            if args.clear_allowlist && !cfg.policy.allowlist.is_empty() {
                cfg.policy.allowlist.clear();
                updated = true;
            }
            for entry in args.allow {
                let trimmed = entry.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if !cfg
                    .policy
                    .allowlist
                    .iter()
                    .any(|existing| existing == trimmed)
                {
                    cfg.policy.allowlist.push(trimmed.to_string());
                    updated = true;
                }
            }
            if updated {
                cfg.save(cwd)?;
            }
        }
    }

    Ok(json!({
        "updated": updated,
        "policy": {
            "approve_bash": cfg.policy.approve_bash,
            "approve_edits": cfg.policy.approve_edits,
            "sandbox_mode": cfg.policy.sandbox_mode,
            "allowlist_entries": cfg.policy.allowlist.len(),
            "allowlist": cfg.policy.allowlist,
        }
    }))
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
            ensure_llm_ready(cwd, json_mode)?;
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

fn run_resume(cwd: &Path, args: RunArgs, json_mode: bool) -> Result<String> {
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

fn run_git(cwd: &Path, cmd: GitCmd, json_mode: bool) -> Result<()> {
    match cmd {
        GitCmd::Status => {
            let porcelain = run_process(cwd, "git", &["status", "--porcelain=2", "--branch"])?;
            let summary = parse_git_status_summary(&porcelain);
            let output = run_process(cwd, "git", &["status", "--short"])?;
            if json_mode {
                print_json(&json!({
                    "command":"git status --short",
                    "output": output,
                    "summary": summary
                }))?;
            } else {
                println!(
                    "branch={} ahead={} behind={} staged={} unstaged={} untracked={} conflicts={}",
                    summary.branch.as_deref().unwrap_or("detached"),
                    summary.ahead,
                    summary.behind,
                    summary.staged,
                    summary.unstaged,
                    summary.untracked,
                    summary.conflicts
                );
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
                let conflicts = output
                    .lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .map(ToString::to_string)
                    .collect::<Vec<_>>();
                let suggestions = conflicts
                    .iter()
                    .map(|path| {
                        json!({
                            "file": path,
                            "ours": format!("deepseek git resolve --strategy ours --file {path}"),
                            "theirs": format!("deepseek git resolve --strategy theirs --file {path}")
                        })
                    })
                    .collect::<Vec<_>>();
                if json_mode {
                    print_json(&json!({
                        "conflicts": conflicts,
                        "count": suggestions.len(),
                        "suggestions": suggestions
                    }))?;
                } else if suggestions.is_empty() {
                    println!("no merge conflicts");
                } else {
                    println!("{output}");
                    for item in suggestions {
                        println!(
                            "resolve {}: {} | {}",
                            item["file"].as_str().unwrap_or_default(),
                            item["ours"].as_str().unwrap_or_default(),
                            item["theirs"].as_str().unwrap_or_default()
                        );
                    }
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

#[derive(Debug, Default, Serialize)]
struct GitStatusSummary {
    branch: Option<String>,
    ahead: u64,
    behind: u64,
    staged: u64,
    unstaged: u64,
    untracked: u64,
    conflicts: u64,
}

fn parse_git_status_summary(porcelain: &str) -> GitStatusSummary {
    let mut summary = GitStatusSummary::default();
    for line in porcelain.lines() {
        if let Some(branch) = line.strip_prefix("# branch.head ") {
            if branch != "(detached)" {
                summary.branch = Some(branch.trim().to_string());
            }
            continue;
        }
        if let Some(ab) = line.strip_prefix("# branch.ab ") {
            let mut parts = ab.split_whitespace();
            if let Some(ahead) = parts.next().and_then(|part| part.strip_prefix('+')) {
                summary.ahead = ahead.parse::<u64>().unwrap_or(0);
            }
            if let Some(behind) = parts.next().and_then(|part| part.strip_prefix('-')) {
                summary.behind = behind.parse::<u64>().unwrap_or(0);
            }
            continue;
        }
        if line.starts_with("? ") {
            summary.untracked += 1;
            continue;
        }
        if line.starts_with("u ") {
            summary.conflicts += 1;
            continue;
        }
        if line.starts_with("1 ") || line.starts_with("2 ") {
            let xy = line.split_whitespace().nth(1).unwrap_or("..");
            let mut chars = xy.chars();
            let x = chars.next().unwrap_or('.');
            let y = chars.next().unwrap_or('.');
            if x != '.' && x != ' ' {
                summary.staged += 1;
            }
            if y != '.' && y != ' ' {
                summary.unstaged += 1;
            }
        }
    }
    summary
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
                ensure_llm_ready(cwd, json_mode)?;
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
            let validation = validate_replay_events(&events);
            if cfg.replay.strict_mode && !validation.passed {
                return Err(anyhow!(
                    "strict replay validation failed: {}",
                    serde_json::to_string(&validation)?
                ));
            }
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
                "validation": validation,
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

#[derive(Debug, Default, Serialize)]
struct ReplayValidation {
    passed: bool,
    monotonic_seq: bool,
    missing_tool_results: Vec<String>,
    orphan_tool_results: Vec<String>,
}

fn validate_replay_events(events: &[EventEnvelope]) -> ReplayValidation {
    use std::collections::{BTreeSet, HashSet};

    let mut proposed = HashSet::new();
    let mut approved = HashSet::new();
    let mut results = HashSet::new();
    let mut missing_tool_results = BTreeSet::new();
    let mut orphan_tool_results = BTreeSet::new();

    let mut monotonic_seq = true;
    let mut last_seq = 0_u64;
    for event in events {
        if event.seq_no < last_seq {
            monotonic_seq = false;
        }
        last_seq = event.seq_no;
        match &event.kind {
            EventKind::ToolProposedV1 { proposal } => {
                if proposal.approved {
                    proposed.insert(proposal.invocation_id);
                }
            }
            EventKind::ToolApprovedV1 { invocation_id } => {
                approved.insert(*invocation_id);
            }
            EventKind::ToolResultV1 { result } => {
                results.insert(result.invocation_id);
            }
            _ => {}
        }
    }

    for invocation_id in proposed.union(&approved) {
        if !results.contains(invocation_id) {
            missing_tool_results.insert(invocation_id.to_string());
        }
    }
    for invocation_id in &results {
        if !approved.contains(invocation_id) && !proposed.contains(invocation_id) {
            orphan_tool_results.insert(invocation_id.to_string());
        }
    }

    let missing_tool_results = missing_tool_results.into_iter().collect::<Vec<_>>();
    let orphan_tool_results = orphan_tool_results.into_iter().collect::<Vec<_>>();
    let passed = monotonic_seq && missing_tool_results.is_empty() && orphan_tool_results.is_empty();
    ReplayValidation {
        passed,
        monotonic_seq,
        missing_tool_results,
        orphan_tool_results,
    }
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
    let result = teleport_now(cwd, args)?;
    if let Some(imported) = result.imported {
        if json_mode {
            print_json(&json!({"imported": true, "path": imported}))?;
        } else {
            println!("imported teleport bundle from {}", imported);
        }
    } else {
        let bundle_id = result
            .bundle_id
            .ok_or_else(|| anyhow!("missing bundle id for teleport export"))?;
        let path = result
            .path
            .ok_or_else(|| anyhow!("missing output path for teleport export"))?;
        if json_mode {
            print_json(&json!({"bundle_id": bundle_id, "path": path}))?;
        } else {
            println!("teleport bundle created at {}", path);
        }
    }
    Ok(())
}

fn run_remote_env(cwd: &Path, cmd: RemoteEnvCmd, json_mode: bool) -> Result<()> {
    let payload = remote_env_now(cwd, cmd)?;
    if json_mode {
        if payload.get("profiles").is_some() {
            print_json(payload.get("profiles").unwrap_or(&serde_json::Value::Null))?;
        } else {
            print_json(&payload)?;
        }
        return Ok(());
    }

    if let Some(profiles) = payload.get("profiles").and_then(|v| v.as_array()) {
        if profiles.is_empty() {
            println!("no remote environment profiles configured");
        } else {
            for profile in profiles {
                println!(
                    "{} {} {}",
                    profile["profile_id"].as_str().unwrap_or_default(),
                    profile["name"].as_str().unwrap_or_default(),
                    profile["endpoint"].as_str().unwrap_or_default()
                );
            }
        }
    } else {
        println!("{}", serde_json::to_string_pretty(&payload)?);
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
