use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use clap_complete::Shell;
use deepseek_agent::AgentEngine;
use deepseek_core::AppConfig;
use deepseek_mcp::McpTransport;
use deepseek_memory::MemoryManager;
use deepseek_policy::load_managed_settings;
use serde_json::json;
use std::path::PathBuf;

// Enhanced context and error handling
use deepseek_errors::ErrorHandler;
mod context_enhanced;
use context_enhanced::errors as enhanced_errors;

mod commands;
mod context;
mod output;
mod util;

use commands::admin::{run_clean, run_config, run_doctor, run_index, run_permissions, run_plugins};
use commands::autopilot::run_autopilot_cmd;
use commands::background::run_background;
use commands::chat::{
    run_chat, run_continue_session, run_print_mode, run_resume, run_resume_specific,
};
use commands::compact::{run_compact, run_rewind};
use commands::diff::{run_apply, run_diff};
use commands::exec::run_exec;
use commands::fork::run_fork;
use commands::git::run_git;
use commands::leadership::run_leadership;
use commands::mcp::run_mcp;
use commands::memory::{run_export, run_memory};
use commands::profile::{run_benchmark, run_profile};
use commands::remote_env::run_remote_env;
use commands::replay::run_replay;
use commands::review::run_review;
use commands::search::run_search;
use commands::serve::{run_completions, run_native_host, run_serve};
use commands::skills::run_skills;
use commands::status::{run_context, run_status, run_usage};
use commands::tasks::run_tasks;
use commands::teleport::run_teleport;
use commands::update::run_update;
use commands::visual::run_visual;
use context::{apply_cli_flags, chat_options_from_cli, ensure_llm_ready, wire_subagent_worker};
use output::print_json;

#[derive(Parser)]
#[command(name = "deepseek")]
#[command(about = "DeepSeek CLI coding agent", long_about = None)]
struct Cli {
    #[arg(long, global = true)]
    json: bool,

    /// Non-interactive mode: run prompt and print result to stdout, then exit.
    /// Accepts prompt as positional arg or reads from stdin.
    #[arg(short = 'p', long = "print")]
    print_mode: bool,

    /// Resume last session and continue conversation.
    #[arg(long = "continue", short = 'c')]
    continue_session: bool,

    /// Resume a specific session by ID.
    #[arg(long = "resume", short = 'r')]
    resume_session: Option<String>,

    /// Fork an existing session: clone conversation + state into a new session.
    #[arg(long = "fork-session")]
    fork_session: Option<String>,

    /// Override the LLM model for this invocation.
    #[arg(long)]
    model: Option<String>,

    /// Output format for print mode: text (default), json, stream-json.
    #[arg(long, default_value = "text")]
    output_format: String,

    /// Maximum number of agent turns before stopping.
    #[arg(long)]
    max_turns: Option<u64>,

    /// Maximum cost in USD before stopping.
    #[arg(long)]
    max_budget_usd: Option<f64>,

    /// Start from a GitHub PR context (fetches diff as prompt context).
    #[arg(long)]
    from_pr: Option<u64>,

    /// Permission mode: ask (default), auto, or plan.
    #[arg(long = "permission-mode", global = true)]
    permission_mode: Option<String>,

    /// Skip all permission checks (dangerous — use only in trusted environments).
    /// Requires `--allow-dangerously-skip-permissions` to also be set.
    #[arg(long = "dangerously-skip-permissions", global = true)]
    dangerously_skip_permissions: bool,

    /// Confirm intent to skip all permission checks.
    /// Must be combined with `--dangerously-skip-permissions`.
    #[arg(long = "allow-dangerously-skip-permissions", global = true)]
    allow_dangerously_skip_permissions: bool,

    /// Only allow these tools (comma-separated function names, e.g. fs_read,fs_grep).
    #[arg(
        long = "allowed-tools",
        visible_alias = "allowedTools",
        global = true,
        value_delimiter = ','
    )]
    allowed_tools: Vec<String>,

    /// Disallow these tools (comma-separated function names, e.g. bash_run,fs_write).
    #[arg(
        long = "disallowed-tools",
        visible_alias = "disallowedTools",
        global = true,
        value_delimiter = ','
    )]
    disallowed_tools: Vec<String>,

    /// Replace the default system prompt entirely.
    #[arg(long = "system-prompt", global = true)]
    system_prompt: Option<String>,

    /// Append text to the default system prompt.
    #[arg(long = "append-system-prompt", global = true)]
    append_system_prompt: Option<String>,

    /// Additional directories to include in workspace context (repeatable).
    #[arg(long = "add-dir", global = true)]
    add_dir: Vec<PathBuf>,

    /// Enable verbose logging to stderr.
    #[arg(short = 'v', long = "verbose", global = true)]
    verbose: bool,

    /// Initialize a new DEEPSEEK.md in the current directory and exit.
    #[arg(long = "init")]
    init: bool,

    /// Non-interactive mode: auto-deny all approval prompts.
    #[arg(long = "no-input")]
    no_input: bool,

    /// Specify a named subagent for the session.
    #[arg(long = "agent")]
    agent: Option<String>,

    /// Define custom subagents dynamically (JSON array).
    #[arg(long = "agents")]
    agents: Option<String>,

    /// Enable IDE integration mode (compatibility flag).
    #[arg(long = "ide", global = true, default_value_t = false, action = clap::ArgAction::SetTrue)]
    ide: bool,

    /// Enable remote workflow mode (compatibility flag).
    #[arg(long = "remote", global = true, default_value_t = false, action = clap::ArgAction::SetTrue)]
    remote: bool,

    /// Enable teammate orchestration mode (compatibility flag).
    #[arg(long = "teammate-mode", global = true)]
    teammate_mode: Option<String>,

    /// Enable beta features by name.
    #[arg(long = "betas", global = true, value_delimiter = ',')]
    betas: Vec<String>,

    /// Run in maintenance mode.
    #[arg(long = "maintenance", global = true, default_value_t = false, action = clap::ArgAction::SetTrue)]
    maintenance: bool,

    /// Print effective settings source paths and exit.
    #[arg(long = "setting-sources", global = true, default_value_t = false, action = clap::ArgAction::SetTrue)]
    setting_sources: bool,

    /// Include partial message chunks in stream interfaces.
    #[arg(long = "include-partial-messages", global = true, default_value_t = false, action = clap::ArgAction::SetTrue)]
    include_partial_messages: bool,

    /// Flag form for teleport handoff.
    #[arg(long = "teleport", global = true, default_value_t = false, action = clap::ArgAction::SetTrue)]
    teleport_flag: bool,

    /// Enable Chrome browser integration.
    #[arg(long = "chrome")]
    chrome: bool,

    /// Disable Chrome browser integration.
    #[arg(long = "no-chrome")]
    no_chrome: bool,

    /// Enable debug mode with optional category filtering (comma-separated).
    #[arg(long = "debug")]
    debug: Option<Option<String>>,

    /// Disable all slash commands / skills.
    #[arg(long = "disable-slash-commands")]
    disable_slash_commands: bool,

    /// Fallback model when primary is overloaded.
    #[arg(long = "fallback-model")]
    fallback_model: Option<String>,

    /// Run initialization hooks and exit.
    #[arg(long = "init-only")]
    init_only: bool,

    /// Input format: text (default) or stream-json.
    #[arg(long = "input-format", default_value = "text")]
    input_format: String,

    /// JSON schema for validated structured output (print mode).
    #[arg(long = "json-schema")]
    json_schema: Option<String>,

    /// Path to MCP server configuration JSON file.
    #[arg(long = "mcp-config")]
    mcp_config: Option<PathBuf>,

    /// Disable session persistence (ephemeral session).
    #[arg(long = "no-session-persistence")]
    no_session_persistence: bool,

    /// MCP tool name for non-interactive permission handling.
    #[arg(long = "permission-prompt-tool")]
    permission_prompt_tool: Option<String>,

    /// Load plugins from directory (repeatable).
    #[arg(long = "plugin-dir")]
    plugin_dir: Vec<PathBuf>,

    /// Use a specific session UUID.
    #[arg(long = "session-id")]
    session_id: Option<String>,

    /// Path to settings JSON file or inline JSON string.
    #[arg(long = "settings")]
    settings: Option<String>,

    /// Only use MCP servers from --mcp-config (ignore project/user configs).
    #[arg(long = "strict-mcp-config")]
    strict_mcp_config: bool,

    /// Replace the system prompt from a file.
    #[arg(long = "system-prompt-file")]
    system_prompt_file: Option<PathBuf>,

    /// Append file contents to the system prompt.
    #[arg(long = "append-system-prompt-file")]
    append_system_prompt_file: Option<PathBuf>,

    /// Restrict built-in tools: "" = none, "default" = all, or comma-separated list.
    #[arg(long = "tools")]
    tools: Option<String>,

    /// Start in an isolated git worktree.
    #[arg(short = 'w', long = "worktree")]
    worktree: bool,

    /// Prompt for print mode (positional, used when -p is set).
    #[arg(trailing_var_arg = true)]
    prompt_args: Vec<String>,

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
    Visual {
        #[command(subcommand)]
        command: VisualCmd,
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
    Leadership(LeadershipArgs),
    Update(UpdateArgs),
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
    /// Code review: analyze diffs and provide structured feedback.
    Review(ReviewArgs),
    /// Execute a shell command under policy enforcement.
    Exec(ExecArgs),
    /// Manage the task queue.
    Tasks {
        #[command(subcommand)]
        command: TasksCmd,
    },
    /// Web search from CLI with provenance metadata.
    Search(SearchArgs),
    /// Fork a session into a new one.
    Fork(ForkArgs),
    /// Inspect context window token usage breakdown.
    Context(ContextArgs),
    /// Generate shell completion scripts.
    Completions(CompletionsArgs),
    /// Start JSON-RPC server for IDE integration.
    Serve(ServeArgs),
    /// Start Chrome native messaging host bridge.
    NativeHost(NativeHostArgs),
}

#[derive(Args)]
struct AskArgs {
    prompt: String,
    #[arg(
        long,
        default_value_t = true,
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    tools: bool,
}

#[derive(Args, Default)]
struct ChatArgs {
    #[arg(
        long,
        default_value_t = true,
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        default_missing_value = "true"
    )]
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
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    benchmark_compare_strict: bool,
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
struct ReviewArgs {
    /// Review unstaged changes (git diff).
    #[arg(long)]
    diff: bool,
    /// Review staged changes (git diff --staged).
    #[arg(long)]
    staged: bool,
    /// Review a specific PR by number (requires gh CLI).
    #[arg(long)]
    pr: Option<u64>,
    /// Review a specific file or path.
    #[arg(long)]
    path: Option<String>,
    /// Focus area for review (security, performance, correctness, style).
    #[arg(long)]
    focus: Option<String>,
    /// Enforce strict JSON findings schema parsing.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    strict: bool,
    /// Publish findings to GitHub PR comments (requires --pr and gh CLI).
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    publish: bool,
    /// Maximum number of inline comments to publish.
    #[arg(long, default_value_t = 20)]
    max_comments: usize,
    /// Dry-run publishing (build publish plan without posting comments).
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
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

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum DoctorModeArg {
    Auto,
    Runtime,
    Test,
    Performance,
}

#[derive(Args, Clone)]
struct DoctorArgs {
    /// Analyze debug input from a file path.
    #[arg(long)]
    analyze_file: Option<String>,
    /// Analyze debug input provided inline.
    #[arg(long)]
    analyze_text: Option<String>,
    /// Debug analysis mode (auto infers from input content).
    #[arg(long, value_enum, default_value_t = DoctorModeArg::Auto)]
    mode: DoctorModeArg,
}

impl Default for DoctorArgs {
    fn default() -> Self {
        Self {
            analyze_file: None,
            analyze_text: None,
            mode: DoctorModeArg::Auto,
        }
    }
}

#[derive(Args, Clone)]
struct LeadershipArgs {
    /// Audit lookback window (hours) for enterprise report metrics.
    #[arg(long, default_value_t = 24)]
    audit_window_hours: u64,
    /// Optional file path to write report JSON (pretty-printed).
    #[arg(long)]
    export: Option<String>,
    /// Exit with non-zero status when readiness checks fail.
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    strict: bool,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum UpdateChannelArg {
    Stable,
    Nightly,
}

#[derive(Args, Clone)]
struct UpdateArgs {
    /// Check update metadata only, do not perform update.
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    check: bool,
    /// Print what would run without executing update commands.
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    dry_run: bool,
    /// Update channel.
    #[arg(long, value_enum, default_value_t = UpdateChannelArg::Stable)]
    channel: UpdateChannelArg,
}

#[derive(Args)]
struct ExecArgs {
    /// Command to execute under policy enforcement.
    command: String,
    #[arg(long, default_value_t = 120)]
    timeout: u64,
}

#[derive(Subcommand)]
enum TasksCmd {
    /// List all tasks in the queue.
    List,
    /// Show details for a specific task.
    Show(TaskShowArgs),
    /// Cancel a task.
    Cancel(TaskCancelArgs),
}

#[derive(Args)]
struct TaskShowArgs {
    /// Task ID.
    id: String,
}

#[derive(Args)]
struct TaskCancelArgs {
    /// Task ID.
    id: String,
}

#[derive(Args)]
struct SearchArgs {
    /// Search query.
    query: String,
    /// Maximum number of results.
    #[arg(long, default_value_t = 10)]
    max_results: u64,
}

#[derive(Args)]
struct ForkArgs {
    /// Session ID to fork from.
    session_id: String,
}

#[derive(Args, Default)]
struct ContextArgs {}

#[derive(Args)]
struct CompletionsArgs {
    /// Shell to generate completions for (bash, zsh, fish, powershell, elvish).
    #[arg(long)]
    shell: Shell,
}

#[derive(Args)]
struct ServeArgs {
    /// Transport to use: stdio (default).
    #[arg(long, default_value = "stdio")]
    transport: String,
}

#[derive(Args)]
struct NativeHostArgs {
    /// Optional workspace root to serve instead of the current directory.
    #[arg(long)]
    workspace_root: Option<PathBuf>,
}

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
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    all: bool,
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    stage: bool,
    #[arg(
        long = "continue",
        default_value_t = false,
        action = clap::ArgAction::SetTrue
    )]
    continue_after: bool,
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
    List(ReplayListArgs),
}

#[derive(Args)]
struct ReplayRunArgs {
    #[arg(long)]
    session_id: String,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    deterministic: bool,
}

#[derive(Args)]
struct ReplayListArgs {
    #[arg(long)]
    session_id: Option<String>,
    #[arg(long, default_value_t = 20)]
    limit: usize,
}

#[derive(Subcommand)]
enum BackgroundCmd {
    List,
    Attach(BackgroundAttachArgs),
    Stop(BackgroundStopArgs),
    RunAgent(BackgroundRunAgentArgs),
    RunShell(BackgroundRunShellArgs),
}

#[derive(Subcommand)]
enum VisualCmd {
    List(VisualListArgs),
    Analyze(VisualAnalyzeArgs),
    Show(VisualShowArgs),
}

#[derive(Args, Clone)]
struct BackgroundAttachArgs {
    job_id: String,
    #[arg(long, default_value_t = 40)]
    tail_lines: usize,
}

#[derive(Args)]
struct BackgroundStopArgs {
    job_id: String,
}

#[derive(Args)]
struct BackgroundRunAgentArgs {
    #[arg(required = true, num_args = 1..)]
    prompt: Vec<String>,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    tools: bool,
}

#[derive(Args)]
struct BackgroundRunShellArgs {
    #[arg(required = true, num_args = 1.., trailing_var_arg = true, allow_hyphen_values = true)]
    command: Vec<String>,
}

#[derive(Args)]
struct VisualListArgs {
    #[arg(long, default_value_t = 25)]
    limit: usize,
}

#[derive(Args)]
struct VisualAnalyzeArgs {
    #[arg(long, default_value_t = 25)]
    limit: usize,
    #[arg(long, default_value_t = 128)]
    min_bytes: u64,
    #[arg(long, default_value_t = 1)]
    min_artifacts: usize,
    #[arg(long, default_value_t = 1)]
    min_image_artifacts: usize,
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    strict: bool,
    #[arg(long)]
    baseline: Option<String>,
    #[arg(long = "write-baseline")]
    write_baseline: Option<String>,
    #[arg(long, default_value_t = 0)]
    max_new_artifacts: usize,
    #[arg(long, default_value_t = 0)]
    max_missing_artifacts: usize,
    #[arg(long, default_value_t = 0)]
    max_changed_artifacts: usize,
    #[arg(long = "expect", alias = "expectations")]
    expectations: Option<String>,
}

#[derive(Args)]
struct VisualShowArgs {
    /// Artifact ID from `visual list` or an absolute/relative file path.
    target: String,
    /// Disable opening external viewer even when fallback mode is `open`.
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    no_open: bool,
}

#[derive(Args, Default)]
struct TeleportArgs {
    #[command(subcommand)]
    command: Option<TeleportCmd>,
    // Legacy export/import flags remain for backward compatibility.
    #[arg(long)]
    session_id: Option<String>,
    #[arg(long)]
    output: Option<String>,
    #[arg(long)]
    import: Option<String>,
}

#[derive(Subcommand)]
enum TeleportCmd {
    /// Export a session bundle (legacy mode).
    Export(TeleportExportArgs),
    /// Import a session bundle (legacy mode).
    Import(TeleportImportArgs),
    /// Create a secure one-time handoff link for desktop/web clients.
    Link(TeleportLinkArgs),
    /// Consume a secure handoff link token and import the bundle.
    Consume(TeleportConsumeArgs),
}

#[derive(Args, Default)]
struct TeleportExportArgs {
    #[arg(long)]
    session_id: Option<String>,
    #[arg(long)]
    output: Option<String>,
}

#[derive(Args, Default)]
struct TeleportImportArgs {
    input: String,
}

#[derive(Args, Default)]
struct TeleportLinkArgs {
    #[arg(long)]
    session_id: Option<String>,
    #[arg(long)]
    base_url: Option<String>,
    #[arg(long, default_value_t = 30)]
    ttl_minutes: u64,
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    open: bool,
}

#[derive(Args, Default)]
struct TeleportConsumeArgs {
    #[arg(long)]
    handoff_id: String,
    #[arg(long)]
    token: String,
}

#[derive(Subcommand)]
enum RemoteEnvCmd {
    List,
    Add(RemoteEnvAddArgs),
    Remove(RemoteEnvRemoveArgs),
    Check(RemoteEnvCheckArgs),
    Exec(RemoteEnvExecArgs),
    RunAgent(RemoteEnvRunAgentArgs),
    Logs(RemoteEnvLogsArgs),
}

#[derive(Args)]
struct RemoteEnvAddArgs {
    name: String,
    endpoint: String,
    #[arg(long, default_value = "token")]
    auth_mode: String,
    /// Optional SSH username override.
    #[arg(long)]
    ssh_user: Option<String>,
    /// Optional SSH port override.
    #[arg(long)]
    ssh_port: Option<u16>,
    /// Optional SSH private key path.
    #[arg(long)]
    ssh_key_path: Option<String>,
    /// Remote workspace root used by `remote-env run-agent`.
    #[arg(long)]
    workspace_root: Option<String>,
    /// Additional environment key-value pairs (`KEY=VALUE`), repeatable.
    #[arg(long = "env")]
    env: Vec<String>,
}

#[derive(Args)]
struct RemoteEnvRemoveArgs {
    profile_id: String,
}

#[derive(Args)]
struct RemoteEnvCheckArgs {
    profile_id: String,
}

#[derive(Args)]
struct RemoteEnvExecArgs {
    profile_id: String,
    #[arg(long = "cmd")]
    cmd: String,
    #[arg(long)]
    timeout_seconds: Option<u64>,
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    background: bool,
}

#[derive(Args)]
struct RemoteEnvRunAgentArgs {
    profile_id: String,
    #[arg(long)]
    prompt: String,
    #[arg(
        long,
        default_value_t = true,
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    tools: bool,
    #[arg(
        long,
        default_value_t = true,
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    max_think: bool,
    #[arg(long)]
    timeout_seconds: Option<u64>,
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    background: bool,
}

#[derive(Args)]
struct RemoteEnvLogsArgs {
    job_id: String,
    #[arg(long, default_value_t = 40)]
    tail_lines: usize,
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
    SyncPublic(BenchmarkSyncPublicArgs),
    PublishParity(BenchmarkPublishParityArgs),
    RunMatrix(BenchmarkRunMatrixArgs),
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

#[derive(Args)]
struct BenchmarkSyncPublicArgs {
    catalog: String,
    #[arg(long, value_delimiter = ',')]
    only: Vec<String>,
    #[arg(long)]
    prefix: Option<String>,
}

#[derive(Args)]
struct BenchmarkPublishParityArgs {
    #[arg(long)]
    matrix: Option<String>,
    #[arg(long = "output-dir")]
    output_dir: Option<String>,
    #[arg(long = "compare")]
    compare: Vec<String>,
    #[arg(long = "require-agent", value_delimiter = ',')]
    require_agent: Vec<String>,
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    strict: bool,
    #[arg(long, default_value = "DEEPSEEK_BENCHMARK_SIGNING_KEY")]
    signing_key_env: String,
}

#[derive(Args)]
struct BenchmarkRunMatrixArgs {
    matrix: String,
    #[arg(long)]
    output: Option<String>,
    #[arg(long = "compare")]
    compare: Vec<String>,
    #[arg(long = "report-output")]
    report_output: Option<String>,
    #[arg(long = "require-agent", value_delimiter = ',')]
    require_agent: Vec<String>,
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    strict: bool,
    #[arg(long, default_value = "DEEPSEEK_BENCHMARK_SIGNING_KEY")]
    signing_key_env: String,
}

#[derive(Subcommand)]
enum PermissionsCmd {
    Show,
    Set(PermissionsSetArgs),
    DryRun(PermissionsDryRunArgs),
}

#[derive(Args)]
struct PermissionsDryRunArgs {
    tool_name: String,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum PermissionModeArg {
    Ask,
    Always,
    Never,
}

impl PermissionModeArg {
    fn to_approval_mode(self) -> deepseek_core::ApprovalMode {
        match self {
            Self::Ask => deepseek_core::ApprovalMode::Ask,
            Self::Always => deepseek_core::ApprovalMode::Always,
            Self::Never => deepseek_core::ApprovalMode::Never,
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

/// Handle error with enhanced formatting
fn handle_error(error: anyhow::Error, json_mode: bool) -> ! {
    let handler = ErrorHandler::new().verbose(false).show_suggestions(true);
    let error_message = handler.handle(&error);

    if json_mode {
        // JSON output for machine consumption
        let json_error = json!({
            "error": error.to_string(),
            "message": error_message,
            "success": false
        });
        eprintln!("{}", serde_json::to_string_pretty(&json_error).unwrap());
    } else {
        // Human-readable output
        eprintln!("{}", error_message);
    }

    std::process::exit(1);
}

/// Validate mutually exclusive CLI flags.
fn validate_cli_flags(cli: &Cli) -> Result<()> {
    if !cli.allowed_tools.is_empty() && !cli.disallowed_tools.is_empty() {
        return Err(enhanced_errors::config_error(
            "--allowed-tools and --disallowed-tools are mutually exclusive",
        )
        .into_error());
    }
    if cli.system_prompt.is_some() && cli.append_system_prompt.is_some() {
        return Err(enhanced_errors::config_error(
            "--system-prompt and --append-system-prompt are mutually exclusive",
        )
        .into_error());
    }
    if cli.system_prompt.is_some() && cli.system_prompt_file.is_some() {
        return Err(enhanced_errors::config_error(
            "--system-prompt and --system-prompt-file are mutually exclusive",
        )
        .into_error());
    }
    if cli.chrome && cli.no_chrome {
        return Err(enhanced_errors::config_error(
            "--chrome and --no-chrome are mutually exclusive",
        )
        .into_error());
    }
    if cli.dangerously_skip_permissions && !cli.allow_dangerously_skip_permissions {
        return Err(enhanced_errors::config_error(
            "--dangerously-skip-permissions requires --allow-dangerously-skip-permissions to confirm intent"
        ).into_error());
    }
    if cli.allow_dangerously_skip_permissions && !cli.dangerously_skip_permissions {
        return Err(enhanced_errors::config_error(
            "--allow-dangerously-skip-permissions has no effect without --dangerously-skip-permissions"
        ).into_error());
    }
    // Check managed settings: enterprise may disable bypass mode.
    if cli.dangerously_skip_permissions
        && cli.allow_dangerously_skip_permissions
        && let Some(managed) = load_managed_settings()
        && managed.disable_bypass_permissions_mode
    {
        return Err(enhanced_errors::permission_error(
            "bypass permissions mode is disabled by managed settings",
        )
        .into_error());
    }
    Ok(())
}

fn main() {
    match run() {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            // Get json mode from args
            let cli = Cli::parse();
            handle_error(e, cli.json);
        }
    }
}

fn run() -> Result<()> {
    let mut cli = Cli::parse();
    validate_cli_flags(&cli)?;
    let cwd = std::env::current_dir()?;

    if cli.ide {
        // SAFETY: called before any threads are spawned in main()
        unsafe { std::env::set_var("DEEPSEEK_IDE_MODE", "1") };
    }
    if cli.remote {
        // SAFETY: called before any threads are spawned in main()
        unsafe { std::env::set_var("DEEPSEEK_REMOTE_MODE", "1") };
    }
    if let Some(mode) = cli.teammate_mode.as_deref() {
        // SAFETY: called before any threads are spawned in main()
        unsafe { std::env::set_var("DEEPSEEK_TEAMMATE_MODE", mode) };
    }
    if !cli.betas.is_empty() {
        // SAFETY: called before any threads are spawned in main()
        unsafe { std::env::set_var("DEEPSEEK_BETAS", cli.betas.join(",")) };
    }
    if cli.maintenance {
        // SAFETY: called before any threads are spawned in main()
        unsafe { std::env::set_var("DEEPSEEK_MAINTENANCE_MODE", "1") };
    }
    if cli.include_partial_messages {
        // SAFETY: called before any threads are spawned in main()
        unsafe { std::env::set_var("DEEPSEEK_INCLUDE_PARTIAL_MESSAGES", "1") };
    }

    if cli.setting_sources {
        let payload = json!({
            "schema": "deepseek.settings_sources.v1",
            "user": AppConfig::user_settings_path(),
            "project": AppConfig::project_settings_path(&cwd),
            "project_local": AppConfig::project_local_settings_path(&cwd),
            "legacy_toml": AppConfig::legacy_toml_path(&cwd),
            "managed": deepseek_policy::managed_settings_path(),
            "team_policy_env": std::env::var("DEEPSEEK_TEAM_POLICY_PATH").ok(),
            "managed_env": std::env::var("DEEPSEEK_MANAGED_SETTINGS_PATH").ok(),
        });
        if cli.json {
            print_json(&payload)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&payload)?);
        }
        return Ok(());
    }

    // Handle --init: create DEEPSEEK.md and exit
    if cli.init {
        let manager = MemoryManager::new(&cwd)?;
        let path = manager.ensure_initialized()?;
        println!("Initialized {}", path.display());
        return Ok(());
    }

    // Handle --init-only: run initialization hooks and exit
    if cli.init_only {
        let engine = AgentEngine::new(&cwd)?;
        drop(engine);
        if cli.json {
            print_json(&json!({"status": "initialized"}))?;
        } else {
            println!("Initialization complete.");
        }
        return Ok(());
    }

    // Handle --debug: enable debug logging
    if let Some(ref _categories) = cli.debug {
        // SAFETY: called before any threads are spawned in main()
        unsafe { std::env::set_var("DEEPSEEK_DEBUG", "1") };
    }

    // Handle --settings: load settings from path or inline JSON
    if let Some(ref settings_arg) = cli.settings {
        // SAFETY: called before any threads are spawned in main()
        unsafe { std::env::set_var("DEEPSEEK_SETTINGS", settings_arg) };
    }

    // Handle -p/--print mode: non-interactive single-shot execution
    if cli.print_mode {
        return run_print_mode(&cwd, &cli);
    }

    // Handle --continue: resume last session
    if cli.continue_session {
        return run_continue_session(&cwd, cli.json, cli.model.as_deref());
    }

    // Handle --resume SESSION_ID: resume specific session
    if let Some(ref session_id) = cli.resume_session {
        return run_resume_specific(&cwd, session_id, cli.json, cli.model.as_deref());
    }

    if cli.teleport_flag && cli.command.is_none() {
        return run_teleport(&cwd, TeleportArgs::default(), cli.json);
    }

    let command = cli
        .command
        .take()
        .unwrap_or(Commands::Chat(ChatArgs::default()));

    match command {
        Commands::Chat(args) => run_chat(&cwd, cli.json, args.tools, args.tui, Some(&cli)),
        Commands::Autopilot(args) => run_autopilot_cmd(&cwd, args, cli.json),
        Commands::Ask(args) => {
            ensure_llm_ready(&cwd, cli.json)?;
            let mut engine = AgentEngine::new(&cwd)?;
            apply_cli_flags(&mut engine, &cli);
            wire_subagent_worker(&engine, &cwd);
            let options = chat_options_from_cli(&cli, args.tools);
            let output = engine.chat_with_options(&args.prompt, options)?;
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
            #[allow(deprecated)]
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
        Commands::Visual { command } => run_visual(&cwd, command, cli.json),
        Commands::Teleport(args) => run_teleport(&cwd, args, cli.json),
        Commands::RemoteEnv { command } => run_remote_env(&cwd, command, cli.json),
        Commands::Status => run_status(&cwd, cli.json),
        Commands::Usage(args) => run_usage(&cwd, args, cli.json),
        Commands::Compact(args) => run_compact(&cwd, args, cli.json),
        Commands::Doctor(args) => run_doctor(&cwd, args, cli.json),
        Commands::Leadership(args) => run_leadership(&cwd, args, cli.json),
        Commands::Update(args) => run_update(&cwd, args, cli.json),
        Commands::Index { command } => run_index(&cwd, command, cli.json),
        Commands::Config { command } => run_config(&cwd, command, cli.json),
        Commands::Benchmark { command } => run_benchmark(cwd.as_path(), command, cli.json),
        Commands::Permissions { command } => run_permissions(&cwd, command, cli.json),
        Commands::Plugins { command } => run_plugins(&cwd, command, cli.json),
        Commands::Clean(args) => run_clean(&cwd, args, cli.json),
        Commands::Review(args) => run_review(&cwd, args, cli.json),
        Commands::Exec(args) => run_exec(&cwd, args, cli.json),
        Commands::Tasks { command } => run_tasks(&cwd, command, cli.json),
        Commands::Search(args) => run_search(&cwd, args, cli.json),
        Commands::Fork(args) => run_fork(&cwd, args, cli.json),
        Commands::Context(_) => run_context(&cwd, cli.json),
        Commands::Completions(args) => run_completions(args),
        Commands::Serve(args) => run_serve(args, cli.json),
        Commands::NativeHost(args) => run_native_host(&cwd, args),
    }
}

#[allow(dead_code)]
fn validate_json_schema(output: &str, schema_str: &str) -> Result<bool> {
    // Parse the schema
    let schema: serde_json::Value = serde_json::from_str(schema_str)
        .map_err(|e| anyhow::anyhow!("invalid JSON schema: {e}"))?;

    // Try to parse the output as JSON
    let value: serde_json::Value = match serde_json::from_str(output) {
        Ok(v) => v,
        Err(_) => return Ok(false),
    };

    // Basic type validation against schema
    if let Some(schema_type) = schema.get("type").and_then(|v| v.as_str()) {
        let type_ok = match schema_type {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "string" => value.is_string(),
            "number" | "integer" => value.is_number(),
            "boolean" => value.is_boolean(),
            "null" => value.is_null(),
            _ => true,
        };
        if !type_ok {
            return Ok(false);
        }
    }

    // Check required fields if schema is object type
    if let Some(required) = schema.get("required").and_then(|v| v.as_array())
        && let Some(obj) = value.as_object()
    {
        for req in required {
            if let Some(key) = req.as_str()
                && !obj.contains_key(key)
            {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, Commands};

    #[test]
    fn native_host_command_parses() {
        let cli = Cli::try_parse_from(["deepseek", "native-host"]).expect("parse native-host");
        assert!(matches!(cli.command, Some(Commands::NativeHost(_))));
    }

    #[test]
    fn json_schema_validates_object() {
        let schema = r#"{"type":"object","required":["name","age"]}"#;
        let valid = r#"{"name":"Alice","age":30}"#;
        let invalid = r#"{"name":"Bob"}"#;
        assert!(super::validate_json_schema(valid, schema).unwrap());
        assert!(!super::validate_json_schema(invalid, schema).unwrap());
    }

    #[test]
    fn json_schema_validates_type() {
        let schema = r#"{"type":"array"}"#;
        assert!(super::validate_json_schema("[1,2,3]", schema).unwrap());
        assert!(!super::validate_json_schema(r#"{"a":1}"#, schema).unwrap());
    }

    #[test]
    fn json_schema_non_json_returns_false() {
        let schema = r#"{"type":"object"}"#;
        assert!(!super::validate_json_schema("not json", schema).unwrap());
    }

    #[test]
    fn json_schema_invalid_schema_errors() {
        let result = super::validate_json_schema("{}", "not valid json");
        assert!(result.is_err());
    }
}
