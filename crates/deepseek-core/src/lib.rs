use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

pub type Result<T> = anyhow::Result<T>;

// DeepSeek V3.2 API model aliases.
pub const DEEPSEEK_V32_CHAT_MODEL: &str = "deepseek-chat";
pub const DEEPSEEK_V32_REASONER_MODEL: &str = "deepseek-reasoner";
pub const DEEPSEEK_PROFILE_V32: &str = "v3_2";
pub const DEEPSEEK_PROFILE_V32_SPECIALE: &str = "v3_2_speciale";
pub const DEEPSEEK_V32_SPECIALE_END_DATE: &str = "2025-12-15";

pub fn normalize_deepseek_model(model: &str) -> Option<&'static str> {
    let normalized = model.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "deepseek-chat" | "deepseek-v3.2" | "deepseek-v3.2-chat" | "v3.2" | "v3_2" => {
            Some(DEEPSEEK_V32_CHAT_MODEL)
        }
        "deepseek-reasoner"
        | "deepseek-v3.2-reasoner"
        | "reasoner"
        | "v3.2-reasoner"
        | "v3_2_reasoner" => Some(DEEPSEEK_V32_REASONER_MODEL),
        "deepseek-v3.2-speciale" | "deepseek-v3.2-special" | "v3.2-speciale" | "v3_2_speciale" => {
            Some(DEEPSEEK_V32_CHAT_MODEL)
        }
        _ => None,
    }
}

pub fn normalize_deepseek_profile(profile: &str) -> Option<&'static str> {
    let normalized = profile.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "" | "v3_2" | "v3.2" | "v32" | "deepseek-v3.2" => Some(DEEPSEEK_PROFILE_V32),
        "v3_2_speciale" | "v3.2-speciale" | "v32-speciale" | "deepseek-v3.2-speciale" => {
            Some(DEEPSEEK_PROFILE_V32_SPECIALE)
        }
        _ => None,
    }
}

pub fn runtime_dir(workspace: &Path) -> PathBuf {
    workspace.join(".deepseek")
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SessionState {
    Idle,
    Planning,
    ExecutingStep,
    AwaitingApproval,
    Verifying,
    Completed,
    Paused,
    Failed,
}

pub fn is_valid_session_state_transition(from: &SessionState, to: &SessionState) -> bool {
    if from == to {
        return true;
    }
    match from {
        SessionState::Idle => matches!(
            to,
            SessionState::Planning | SessionState::Paused | SessionState::Failed
        ),
        SessionState::Planning => matches!(
            to,
            SessionState::ExecutingStep
                | SessionState::Verifying
                | SessionState::Paused
                | SessionState::Failed
        ),
        SessionState::ExecutingStep => matches!(
            to,
            SessionState::AwaitingApproval
                | SessionState::Verifying
                | SessionState::Planning
                | SessionState::Completed
                | SessionState::Paused
                | SessionState::Failed
        ),
        SessionState::AwaitingApproval => matches!(
            to,
            SessionState::ExecutingStep
                | SessionState::Planning
                | SessionState::Paused
                | SessionState::Failed
        ),
        SessionState::Verifying => matches!(
            to,
            SessionState::Planning
                | SessionState::ExecutingStep
                | SessionState::Completed
                | SessionState::Paused
                | SessionState::Failed
        ),
        SessionState::Completed => matches!(to, SessionState::Idle | SessionState::Planning),
        SessionState::Paused => matches!(
            to,
            SessionState::Planning | SessionState::ExecutingStep | SessionState::Failed
        ),
        SessionState::Failed => matches!(to, SessionState::Planning | SessionState::Idle),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionBudgets {
    pub per_turn_seconds: u64,
    pub max_think_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub session_id: Uuid,
    pub workspace_root: String,
    pub baseline_commit: Option<String>,
    pub status: SessionState,
    pub budgets: SessionBudgets,
    pub active_plan_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub plan_id: Uuid,
    pub version: u32,
    pub goal: String,
    pub assumptions: Vec<String>,
    pub steps: Vec<PlanStep>,
    pub verification: Vec<String>,
    pub risk_notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub step_id: Uuid,
    pub title: String,
    pub intent: String,
    pub tools: Vec<String>,
    pub files: Vec<String>,
    pub done: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterSignals {
    pub prompt_complexity: f32,
    pub repo_breadth: f32,
    pub failure_streak: f32,
    pub verification_failures: f32,
    pub low_confidence: f32,
    pub ambiguity_flags: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterWeights {
    pub w1: f32,
    pub w2: f32,
    pub w3: f32,
    pub w4: f32,
    pub w5: f32,
    pub w6: f32,
}

impl Default for RouterWeights {
    fn default() -> Self {
        Self {
            w1: 0.2,
            w2: 0.15,
            w3: 0.2,
            w4: 0.15,
            w5: 0.2,
            w6: 0.1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmUnit {
    Planner,
    Executor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterDecision {
    pub decision_id: Uuid,
    pub reason_codes: Vec<String>,
    pub selected_model: String,
    pub confidence: f32,
    pub score: f32,
    pub escalated: bool,
    /// When true, thinking mode should be enabled on the chat model.
    /// This replaces routing to `deepseek-reasoner` — instead we use
    /// `deepseek-chat` with `thinking: {type: "enabled", budget_tokens: N}`.
    #[serde(default)]
    pub thinking_enabled: bool,
}

/// Type-safe tool name enum covering all built-in tools.
///
/// Maps between underscored API names (`fs_read`) and dotted internal names (`fs.read`).
/// Plugin and MCP tools are not represented here — use `from_api_name` which returns `None`
/// for unknown tool names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolName {
    FsRead,
    FsWrite,
    FsEdit,
    FsList,
    FsGlob,
    FsGrep,
    BashRun,
    MultiEdit,
    GitStatus,
    GitDiff,
    GitShow,
    WebFetch,
    WebSearch,
    NotebookRead,
    NotebookEdit,
    IndexQuery,
    PatchStage,
    PatchApply,
    DiagnosticsCheck,
    ChromeNavigate,
    ChromeClick,
    ChromeTypeText,
    ChromeScreenshot,
    ChromeReadConsole,
    ChromeEvaluate,
    UserQuestion,
    TaskCreate,
    TaskUpdate,
    TaskGet,
    TaskList,
    SpawnTask,
    TaskOutput,
    TaskStop,
    EnterPlanMode,
    ExitPlanMode,
    Skill,
    KillShell,
}

impl ToolName {
    /// Parse from underscored API name (e.g. `"fs_read"`). Returns `None` for
    /// unknown names (plugins, MCP tools).
    #[must_use]
    pub fn from_api_name(s: &str) -> Option<Self> {
        Some(match s {
            "fs_read" => Self::FsRead,
            "fs_write" => Self::FsWrite,
            "fs_edit" => Self::FsEdit,
            "fs_list" => Self::FsList,
            "fs_glob" => Self::FsGlob,
            "fs_grep" => Self::FsGrep,
            "bash_run" => Self::BashRun,
            "multi_edit" => Self::MultiEdit,
            "git_status" => Self::GitStatus,
            "git_diff" => Self::GitDiff,
            "git_show" => Self::GitShow,
            "web_fetch" => Self::WebFetch,
            "web_search" => Self::WebSearch,
            "notebook_read" => Self::NotebookRead,
            "notebook_edit" => Self::NotebookEdit,
            "index_query" => Self::IndexQuery,
            "patch_stage" => Self::PatchStage,
            "patch_apply" => Self::PatchApply,
            "diagnostics_check" => Self::DiagnosticsCheck,
            "chrome_navigate" => Self::ChromeNavigate,
            "chrome_click" => Self::ChromeClick,
            "chrome_type_text" => Self::ChromeTypeText,
            "chrome_screenshot" => Self::ChromeScreenshot,
            "chrome_read_console" => Self::ChromeReadConsole,
            "chrome_evaluate" => Self::ChromeEvaluate,
            "user_question" => Self::UserQuestion,
            "task_create" => Self::TaskCreate,
            "task_update" => Self::TaskUpdate,
            "task_get" => Self::TaskGet,
            "task_list" => Self::TaskList,
            "spawn_task" => Self::SpawnTask,
            "task_output" => Self::TaskOutput,
            "task_stop" => Self::TaskStop,
            "enter_plan_mode" => Self::EnterPlanMode,
            "exit_plan_mode" => Self::ExitPlanMode,
            "skill" => Self::Skill,
            "kill_shell" => Self::KillShell,
            _ => return None,
        })
    }

    /// Parse from dotted internal name (e.g. `"fs.read"`). Returns `None` for
    /// unknown names.
    #[must_use]
    pub fn from_internal_name(s: &str) -> Option<Self> {
        Some(match s {
            "fs.read" => Self::FsRead,
            "fs.write" => Self::FsWrite,
            "fs.edit" => Self::FsEdit,
            "fs.list" => Self::FsList,
            "fs.glob" => Self::FsGlob,
            "fs.grep" => Self::FsGrep,
            "bash.run" => Self::BashRun,
            "multi_edit" => Self::MultiEdit,
            "git.status" => Self::GitStatus,
            "git.diff" => Self::GitDiff,
            "git.show" => Self::GitShow,
            "web.fetch" => Self::WebFetch,
            "web.search" => Self::WebSearch,
            "notebook.read" => Self::NotebookRead,
            "notebook.edit" => Self::NotebookEdit,
            "index.query" => Self::IndexQuery,
            "patch.stage" => Self::PatchStage,
            "patch.apply" => Self::PatchApply,
            "diagnostics.check" => Self::DiagnosticsCheck,
            "chrome.navigate" => Self::ChromeNavigate,
            "chrome.click" => Self::ChromeClick,
            "chrome.type_text" => Self::ChromeTypeText,
            "chrome.screenshot" => Self::ChromeScreenshot,
            "chrome.read_console" => Self::ChromeReadConsole,
            "chrome.evaluate" => Self::ChromeEvaluate,
            "user_question" => Self::UserQuestion,
            "task_create" => Self::TaskCreate,
            "task_update" => Self::TaskUpdate,
            "task_get" => Self::TaskGet,
            "task_list" => Self::TaskList,
            "spawn_task" => Self::SpawnTask,
            "task_output" => Self::TaskOutput,
            "task_stop" => Self::TaskStop,
            "enter_plan_mode" => Self::EnterPlanMode,
            "exit_plan_mode" => Self::ExitPlanMode,
            "skill" => Self::Skill,
            "kill_shell" => Self::KillShell,
            _ => return None,
        })
    }

    /// Dotted internal name (e.g. `"fs.read"`, `"bash.run"`).
    #[must_use]
    pub fn as_internal(&self) -> &'static str {
        match self {
            Self::FsRead => "fs.read",
            Self::FsWrite => "fs.write",
            Self::FsEdit => "fs.edit",
            Self::FsList => "fs.list",
            Self::FsGlob => "fs.glob",
            Self::FsGrep => "fs.grep",
            Self::BashRun => "bash.run",
            Self::MultiEdit => "multi_edit",
            Self::GitStatus => "git.status",
            Self::GitDiff => "git.diff",
            Self::GitShow => "git.show",
            Self::WebFetch => "web.fetch",
            Self::WebSearch => "web.search",
            Self::NotebookRead => "notebook.read",
            Self::NotebookEdit => "notebook.edit",
            Self::IndexQuery => "index.query",
            Self::PatchStage => "patch.stage",
            Self::PatchApply => "patch.apply",
            Self::DiagnosticsCheck => "diagnostics.check",
            Self::ChromeNavigate => "chrome.navigate",
            Self::ChromeClick => "chrome.click",
            Self::ChromeTypeText => "chrome.type_text",
            Self::ChromeScreenshot => "chrome.screenshot",
            Self::ChromeReadConsole => "chrome.read_console",
            Self::ChromeEvaluate => "chrome.evaluate",
            Self::UserQuestion => "user_question",
            Self::TaskCreate => "task_create",
            Self::TaskUpdate => "task_update",
            Self::TaskGet => "task_get",
            Self::TaskList => "task_list",
            Self::SpawnTask => "spawn_task",
            Self::TaskOutput => "task_output",
            Self::TaskStop => "task_stop",
            Self::EnterPlanMode => "enter_plan_mode",
            Self::ExitPlanMode => "exit_plan_mode",
            Self::Skill => "skill",
            Self::KillShell => "kill_shell",
        }
    }

    /// Underscored API name (e.g. `"fs_read"`, `"bash_run"`).
    #[must_use]
    pub fn as_api_name(&self) -> &'static str {
        match self {
            Self::FsRead => "fs_read",
            Self::FsWrite => "fs_write",
            Self::FsEdit => "fs_edit",
            Self::FsList => "fs_list",
            Self::FsGlob => "fs_glob",
            Self::FsGrep => "fs_grep",
            Self::BashRun => "bash_run",
            Self::MultiEdit => "multi_edit",
            Self::GitStatus => "git_status",
            Self::GitDiff => "git_diff",
            Self::GitShow => "git_show",
            Self::WebFetch => "web_fetch",
            Self::WebSearch => "web_search",
            Self::NotebookRead => "notebook_read",
            Self::NotebookEdit => "notebook_edit",
            Self::IndexQuery => "index_query",
            Self::PatchStage => "patch_stage",
            Self::PatchApply => "patch_apply",
            Self::DiagnosticsCheck => "diagnostics_check",
            Self::ChromeNavigate => "chrome_navigate",
            Self::ChromeClick => "chrome_click",
            Self::ChromeTypeText => "chrome_type_text",
            Self::ChromeScreenshot => "chrome_screenshot",
            Self::ChromeReadConsole => "chrome_read_console",
            Self::ChromeEvaluate => "chrome_evaluate",
            Self::UserQuestion => "user_question",
            Self::TaskCreate => "task_create",
            Self::TaskUpdate => "task_update",
            Self::TaskGet => "task_get",
            Self::TaskList => "task_list",
            Self::SpawnTask => "spawn_task",
            Self::TaskOutput => "task_output",
            Self::TaskStop => "task_stop",
            Self::EnterPlanMode => "enter_plan_mode",
            Self::ExitPlanMode => "exit_plan_mode",
            Self::Skill => "skill",
            Self::KillShell => "kill_shell",
        }
    }

    /// Whether this tool is read-only (allowed in plan/explore mode).
    #[must_use]
    pub fn is_read_only(&self) -> bool {
        matches!(
            self,
            Self::FsRead
                | Self::FsList
                | Self::FsGlob
                | Self::FsGrep
                | Self::GitStatus
                | Self::GitDiff
                | Self::GitShow
                | Self::WebFetch
                | Self::WebSearch
                | Self::IndexQuery
                | Self::NotebookRead
                | Self::DiagnosticsCheck
                | Self::UserQuestion
                | Self::TaskCreate
                | Self::TaskUpdate
                | Self::TaskGet
                | Self::TaskList
                | Self::SpawnTask
                | Self::ExitPlanMode
        )
    }

    /// Whether this tool is handled by AgentEngine, not LocalToolHost.
    #[must_use]
    pub fn is_agent_level(&self) -> bool {
        matches!(
            self,
            Self::UserQuestion
                | Self::TaskCreate
                | Self::TaskUpdate
                | Self::TaskGet
                | Self::TaskList
                | Self::SpawnTask
                | Self::TaskOutput
                | Self::TaskStop
                | Self::EnterPlanMode
                | Self::ExitPlanMode
                | Self::Skill
                | Self::KillShell
        )
    }

    /// Whether this tool is blocked during review mode.
    #[must_use]
    pub fn is_review_blocked(&self) -> bool {
        matches!(
            self,
            Self::FsWrite
                | Self::FsEdit
                | Self::MultiEdit
                | Self::PatchStage
                | Self::PatchApply
                | Self::BashRun
                | Self::NotebookEdit
        )
    }

    /// All built-in tool name variants.
    pub const ALL: &'static [ToolName] = &[
        Self::FsRead,
        Self::FsWrite,
        Self::FsEdit,
        Self::FsList,
        Self::FsGlob,
        Self::FsGrep,
        Self::BashRun,
        Self::MultiEdit,
        Self::GitStatus,
        Self::GitDiff,
        Self::GitShow,
        Self::WebFetch,
        Self::WebSearch,
        Self::NotebookRead,
        Self::NotebookEdit,
        Self::IndexQuery,
        Self::PatchStage,
        Self::PatchApply,
        Self::DiagnosticsCheck,
        Self::ChromeNavigate,
        Self::ChromeClick,
        Self::ChromeTypeText,
        Self::ChromeScreenshot,
        Self::ChromeReadConsole,
        Self::ChromeEvaluate,
        Self::UserQuestion,
        Self::TaskCreate,
        Self::TaskUpdate,
        Self::TaskGet,
        Self::TaskList,
        Self::SpawnTask,
        Self::TaskOutput,
        Self::TaskStop,
        Self::EnterPlanMode,
        Self::ExitPlanMode,
        Self::Skill,
        Self::KillShell,
    ];
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub args: serde_json::Value,
    pub requires_approval: bool,
}

impl ToolCall {
    /// Parse the internal tool name into a typed `ToolName`.
    /// Returns `None` for plugin/MCP tools or unknown names.
    #[must_use]
    pub fn tool_name(&self) -> Option<ToolName> {
        ToolName::from_internal_name(&self.name)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolProposal {
    pub invocation_id: Uuid,
    pub call: ToolCall,
    pub approved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovedToolCall {
    pub invocation_id: Uuid,
    pub call: ToolCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub invocation_id: Uuid,
    pub success: bool,
    pub output: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventEnvelope {
    pub seq_no: u64,
    pub at: DateTime<Utc>,
    pub session_id: Uuid,
    pub kind: EventKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum EventKind {
    TurnAddedV1 {
        role: String,
        content: String,
    },
    /// Structured chat turn with full tool_call data for accurate session resume.
    ChatTurnV1 {
        message: ChatMessage,
    },
    SessionStateChangedV1 {
        from: SessionState,
        to: SessionState,
    },
    PlanCreatedV1 {
        plan: Plan,
    },
    PlanRevisedV1 {
        plan: Plan,
    },
    StepMarkedV1 {
        step_id: Uuid,
        done: bool,
        note: String,
    },
    RouterDecisionV1 {
        decision: RouterDecision,
    },
    RouterEscalationV1 {
        reason_codes: Vec<String>,
    },
    ToolProposedV1 {
        proposal: ToolProposal,
    },
    ToolApprovedV1 {
        invocation_id: Uuid,
    },
    ToolResultV1 {
        result: ToolResult,
    },
    PatchStagedV1 {
        patch_id: Uuid,
        base_sha256: String,
    },
    PatchAppliedV1 {
        patch_id: Uuid,
        applied: bool,
        conflicts: Vec<String>,
    },
    VerificationRunV1 {
        command: String,
        success: bool,
        output: String,
    },
    PluginInstalledV1 {
        plugin_id: String,
        version: String,
    },
    PluginRemovedV1 {
        plugin_id: String,
    },
    PluginEnabledV1 {
        plugin_id: String,
    },
    PluginDisabledV1 {
        plugin_id: String,
    },
    UsageUpdatedV1 {
        unit: LlmUnit,
        model: String,
        input_tokens: u64,
        output_tokens: u64,
    },
    ContextCompactedV1 {
        summary_id: Uuid,
        from_turn: u64,
        to_turn: u64,
        token_delta_estimate: i64,
        replay_pointer: String,
    },
    AutopilotRunStartedV1 {
        run_id: Uuid,
        prompt: String,
    },
    AutopilotRunHeartbeatV1 {
        run_id: Uuid,
        completed_iterations: u64,
        failed_iterations: u64,
        consecutive_failures: u64,
        last_error: Option<String>,
    },
    AutopilotRunStoppedV1 {
        run_id: Uuid,
        stop_reason: String,
        completed_iterations: u64,
        failed_iterations: u64,
    },
    PluginCatalogSyncedV1 {
        source: String,
        total: usize,
        verified_count: usize,
    },
    PluginVerifiedV1 {
        plugin_id: String,
        verified: bool,
        reason: String,
    },
    CheckpointCreatedV1 {
        checkpoint_id: Uuid,
        reason: String,
        files_count: u64,
        snapshot_path: String,
    },
    CheckpointRewoundV1 {
        checkpoint_id: Uuid,
        reason: String,
    },
    TranscriptExportedV1 {
        export_id: Uuid,
        format: String,
        output_path: String,
    },
    McpServerAddedV1 {
        server_id: String,
        transport: String,
        endpoint: String,
    },
    McpServerRemovedV1 {
        server_id: String,
    },
    McpToolDiscoveredV1 {
        server_id: String,
        tool_name: String,
    },
    SubagentSpawnedV1 {
        run_id: Uuid,
        name: String,
        goal: String,
    },
    SubagentCompletedV1 {
        run_id: Uuid,
        output: String,
    },
    SubagentFailedV1 {
        run_id: Uuid,
        error: String,
    },
    CostUpdatedV1 {
        input_tokens: u64,
        output_tokens: u64,
        estimated_cost_usd: f64,
    },
    EffortChangedV1 {
        level: String,
    },
    ProfileCapturedV1 {
        profile_id: Uuid,
        summary: String,
        elapsed_ms: u64,
    },
    MemorySyncedV1 {
        version_id: Uuid,
        path: String,
        note: String,
    },
    HookExecutedV1 {
        phase: String,
        hook_path: String,
        success: bool,
        timed_out: bool,
        exit_code: Option<i32>,
    },
    SessionForkedV1 {
        from_session_id: Uuid,
        to_session_id: Uuid,
    },
    PermissionModeChangedV1 {
        from: String,
        to: String,
    },
    WebSearchExecutedV1 {
        query: String,
        results_count: u64,
        cached: bool,
    },
    ReviewStartedV1 {
        review_id: Uuid,
        preset: String,
        target: String,
    },
    ReviewCompletedV1 {
        review_id: Uuid,
        findings_count: u64,
        critical_count: u64,
    },
    ReviewPublishedV1 {
        review_id: Uuid,
        pr_number: u64,
        comments_published: u64,
        dry_run: bool,
    },
    TaskCreatedV1 {
        task_id: Uuid,
        title: String,
        priority: u32,
    },
    TaskCompletedV1 {
        task_id: Uuid,
        outcome: String,
    },
    ArtifactBundledV1 {
        task_id: Uuid,
        artifact_path: String,
        files: Vec<String>,
    },
    TelemetryEventV1 {
        name: String,
        properties: serde_json::Value,
    },
    BackgroundJobStartedV1 {
        job_id: Uuid,
        kind: String,
        reference: String,
    },
    BackgroundJobResumedV1 {
        job_id: Uuid,
        reference: String,
    },
    BackgroundJobStoppedV1 {
        job_id: Uuid,
        reason: String,
    },
    SkillLoadedV1 {
        skill_id: String,
        source_path: String,
    },
    ReplayExecutedV1 {
        session_id: Uuid,
        deterministic: bool,
        events_replayed: u64,
    },
    PromptCacheHitV1 {
        cache_key: String,
        model: String,
    },
    OffPeakScheduledV1 {
        reason: String,
        resume_after: String,
    },
    VisualArtifactCapturedV1 {
        artifact_id: Uuid,
        path: String,
        mime: String,
    },
    RemoteEnvConfiguredV1 {
        profile_id: Uuid,
        name: String,
        endpoint: String,
    },
    RemoteEnvExecutionStartedV1 {
        execution_id: Uuid,
        profile_id: Uuid,
        mode: String,
        background: bool,
        reference: String,
    },
    RemoteEnvExecutionCompletedV1 {
        execution_id: Uuid,
        profile_id: Uuid,
        mode: String,
        success: bool,
        duration_ms: u64,
        background: bool,
        reference: String,
    },
    TeleportBundleCreatedV1 {
        bundle_id: Uuid,
        path: String,
    },
    TeleportHandoffLinkCreatedV1 {
        handoff_id: Uuid,
        session_id: Uuid,
        expires_at: String,
    },
    TeleportHandoffLinkConsumedV1 {
        handoff_id: Uuid,
        session_id: Uuid,
        success: bool,
        reason: String,
    },
    SessionStartedV1 {
        session_id: Uuid,
        workspace: String,
    },
    SessionResumedV1 {
        session_id: Uuid,
        events_replayed: u64,
    },
    ToolDeniedV1 {
        invocation_id: Uuid,
        tool_name: String,
        reason: String,
    },
    NotebookEditedV1 {
        path: String,
        operation: String,
        cell_index: u64,
        cell_type: Option<String>,
    },
    PdfTextExtractedV1 {
        path: String,
        pages: String,
        text_length: u64,
    },
    IdeSessionStartedV1 {
        transport: String,
        client_info: String,
    },
    TurnLimitExceededV1 {
        limit: u64,
        actual: u64,
    },
    BudgetExceededV1 {
        limit_usd: f64,
        actual_usd: f64,
    },
    TaskUpdatedV1 {
        task_id: String,
        status: String,
    },
    TaskDeletedV1 {
        task_id: String,
    },
    EnterPlanModeV1 {
        session_id: Uuid,
    },
    ExitPlanModeV1 {
        session_id: Uuid,
    },
    ProviderSelectedV1 {
        provider: String,
        model: String,
    },
}

impl EventKind {
    /// Logical category for this event kind.
    #[must_use]
    pub fn category(&self) -> &'static str {
        match self {
            // Session lifecycle
            Self::SessionStartedV1 { .. }
            | Self::SessionResumedV1 { .. }
            | Self::SessionStateChangedV1 { .. }
            | Self::SessionForkedV1 { .. } => "session",

            // Chat / transcript
            Self::TurnAddedV1 { .. }
            | Self::ChatTurnV1 { .. }
            | Self::ContextCompactedV1 { .. }
            | Self::EffortChangedV1 { .. }
            | Self::PermissionModeChangedV1 { .. }
            | Self::TurnLimitExceededV1 { .. }
            | Self::BudgetExceededV1 { .. } => "chat",

            // Tool invocations
            Self::ToolProposedV1 { .. }
            | Self::ToolApprovedV1 { .. }
            | Self::ToolResultV1 { .. }
            | Self::ToolDeniedV1 { .. } => "tool",

            // Plans
            Self::PlanCreatedV1 { .. }
            | Self::PlanRevisedV1 { .. }
            | Self::StepMarkedV1 { .. }
            | Self::EnterPlanModeV1 { .. }
            | Self::ExitPlanModeV1 { .. } => "plan",

            // Tasks
            Self::TaskCreatedV1 { .. }
            | Self::TaskCompletedV1 { .. }
            | Self::TaskUpdatedV1 { .. }
            | Self::TaskDeletedV1 { .. } => "task",

            // Routing / model selection
            Self::RouterDecisionV1 { .. }
            | Self::RouterEscalationV1 { .. }
            | Self::ProviderSelectedV1 { .. } => "router",

            // Patches (diff/apply)
            Self::PatchStagedV1 { .. } | Self::PatchAppliedV1 { .. } => "patch",

            // Plugins
            Self::PluginInstalledV1 { .. }
            | Self::PluginRemovedV1 { .. }
            | Self::PluginEnabledV1 { .. }
            | Self::PluginDisabledV1 { .. }
            | Self::PluginCatalogSyncedV1 { .. }
            | Self::PluginVerifiedV1 { .. } => "plugin",

            // Usage / cost tracking
            Self::UsageUpdatedV1 { .. }
            | Self::CostUpdatedV1 { .. }
            | Self::PromptCacheHitV1 { .. } => "usage",

            // Autopilot
            Self::AutopilotRunStartedV1 { .. }
            | Self::AutopilotRunHeartbeatV1 { .. }
            | Self::AutopilotRunStoppedV1 { .. } => "autopilot",

            // Subagent orchestration
            Self::SubagentSpawnedV1 { .. }
            | Self::SubagentCompletedV1 { .. }
            | Self::SubagentFailedV1 { .. } => "subagent",

            // Background jobs
            Self::BackgroundJobStartedV1 { .. }
            | Self::BackgroundJobResumedV1 { .. }
            | Self::BackgroundJobStoppedV1 { .. } => "background",

            // MCP
            Self::McpServerAddedV1 { .. }
            | Self::McpServerRemovedV1 { .. }
            | Self::McpToolDiscoveredV1 { .. } => "mcp",

            // Skills
            Self::SkillLoadedV1 { .. } => "skill",

            // Hooks
            Self::HookExecutedV1 { .. } => "hook",

            // Memory
            Self::MemorySyncedV1 { .. } => "memory",

            // Profile / benchmark
            Self::ProfileCapturedV1 { .. } => "profile",

            // Verification
            Self::VerificationRunV1 { .. } => "verification",

            // Checkpoint / rewind
            Self::CheckpointCreatedV1 { .. } | Self::CheckpointRewoundV1 { .. } => "checkpoint",

            // Export
            Self::TranscriptExportedV1 { .. } => "export",

            // Web search
            Self::WebSearchExecutedV1 { .. } => "web",

            // Review
            Self::ReviewStartedV1 { .. }
            | Self::ReviewCompletedV1 { .. }
            | Self::ReviewPublishedV1 { .. } => "review",

            // Artifact bundling
            Self::ArtifactBundledV1 { .. } => "artifact",

            // Telemetry
            Self::TelemetryEventV1 { .. } => "telemetry",

            // Replay
            Self::ReplayExecutedV1 { .. } => "replay",

            // Scheduling
            Self::OffPeakScheduledV1 { .. } => "scheduling",

            // Visual testing
            Self::VisualArtifactCapturedV1 { .. } => "visual",

            // Remote environments
            Self::RemoteEnvConfiguredV1 { .. }
            | Self::RemoteEnvExecutionStartedV1 { .. }
            | Self::RemoteEnvExecutionCompletedV1 { .. } => "remote_env",

            // Teleport
            Self::TeleportBundleCreatedV1 { .. }
            | Self::TeleportHandoffLinkCreatedV1 { .. }
            | Self::TeleportHandoffLinkConsumedV1 { .. } => "teleport",

            // Notebook
            Self::NotebookEditedV1 { .. } => "notebook",

            // PDF
            Self::PdfTextExtractedV1 { .. } => "pdf",

            // IDE
            Self::IdeSessionStartedV1 { .. } => "ide",
        }
    }

    /// Whether this event is a tool-related event (proposal, approval, result, denial).
    #[must_use]
    pub fn is_tool_event(&self) -> bool {
        self.category() == "tool"
    }

    /// Whether this event is a session lifecycle event.
    #[must_use]
    pub fn is_session_event(&self) -> bool {
        self.category() == "session"
    }
}

pub trait ModelRouter {
    fn select(&self, unit: LlmUnit, signals: RouterSignals) -> RouterDecision;
}

pub trait ToolHost {
    fn propose(&self, call: ToolCall) -> ToolProposal;
    fn execute(&self, approved: ApprovedToolCall) -> ToolResult;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageContent {
    pub mime: String,
    pub base64_data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    pub unit: LlmUnit,
    pub prompt: String,
    pub model: String,
    pub max_tokens: u32,
    #[serde(default)]
    pub non_urgent: bool,
    #[serde(default)]
    pub images: Vec<ImageContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

fn default_finish_reason() -> String {
    "stop".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub text: String,
    #[serde(default = "default_finish_reason")]
    pub finish_reason: String,
    #[serde(default)]
    pub reasoning_content: String,
    #[serde(default)]
    pub tool_calls: Vec<LlmToolCall>,
}

/// A single chunk emitted during streaming.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// A content text delta.
    ContentDelta(String),
    /// A reasoning/thinking text delta.
    ReasoningDelta(String),
    /// Architect phase started.
    ArchitectStarted { iteration: u64 },
    /// Architect phase completed.
    ArchitectCompleted {
        iteration: u64,
        files: u32,
        no_edit: bool,
    },
    /// Editor phase started.
    EditorStarted { iteration: u64, files: u32 },
    /// Editor phase completed.
    EditorCompleted { iteration: u64, status: String },
    /// Apply phase started.
    ApplyStarted { iteration: u64 },
    /// Apply phase completed.
    ApplyCompleted {
        iteration: u64,
        success: bool,
        summary: String,
    },
    /// Verify phase started.
    VerifyStarted {
        iteration: u64,
        commands: Vec<String>,
    },
    /// Verify phase completed.
    VerifyCompleted {
        iteration: u64,
        success: bool,
        summary: String,
    },
    /// A tool call has started execution.
    ToolCallStart {
        tool_name: String,
        args_summary: String,
    },
    /// A tool call has completed execution.
    ToolCallEnd {
        tool_name: String,
        duration_ms: u64,
        success: bool,
        summary: String,
    },
    /// Agent mode transition.
    ModeTransition {
        from: String,
        to: String,
        reason: String,
    },
    /// A subagent was spawned for a complex task lane.
    SubagentSpawned {
        run_id: String,
        name: String,
        goal: String,
    },
    /// A subagent completed successfully.
    SubagentCompleted {
        run_id: String,
        name: String,
        summary: String,
    },
    /// A subagent failed.
    SubagentFailed {
        run_id: String,
        name: String,
        error: String,
    },
    /// An image was read and should be displayed inline in the terminal.
    ImageData { data: Vec<u8>, label: String },
    /// Clear any previously streamed text — the response contains tool calls,
    /// so the interleaved text fragments should be discarded from the display.
    ClearStreamingText,
    /// Streaming is done; the final assembled response follows.
    Done,
}

/// Callback type for receiving streaming chunks.
/// Uses `Arc<dyn Fn>` so it can be cloned across multiple turns in a chat loop.
pub type StreamCallback = std::sync::Arc<dyn Fn(StreamChunk) + Send + Sync>;

/// A question posed by the agent to the user via the `user_question` tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserQuestion {
    pub question: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<String>,
}

/// Handler invoked when the agent asks the user a question.
/// Returns `Some(answer)` if the user provides an answer, or `None` on cancellation.
pub type UserQuestionHandler = std::sync::Arc<dyn Fn(UserQuestion) -> Option<String> + Send + Sync>;

// ── Chat-with-tools types (DeepSeek function calling) ──────────────────

/// Configuration for thinking mode on `deepseek-chat`.
///
/// When enabled, the model produces chain-of-thought reasoning before responding.
/// This replaces the old `deepseek-reasoner` model selection — `deepseek-chat`
/// with thinking enabled gives us reasoning + function calling simultaneously.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
}

impl ThinkingConfig {
    /// Enable thinking mode with a token budget for chain-of-thought.
    #[must_use]
    pub fn enabled(budget: u32) -> Self {
        Self {
            thinking_type: "enabled".to_string(),
            budget_tokens: Some(budget),
        }
    }

    /// Disable thinking mode.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            thinking_type: "disabled".to_string(),
            budget_tokens: None,
        }
    }
}

/// A message in a multi-turn conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum ChatMessage {
    #[serde(rename = "system")]
    System { content: String },
    #[serde(rename = "user")]
    User { content: String },
    #[serde(rename = "assistant")]
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        /// Chain-of-thought reasoning from thinking mode (DeepSeek V3.2).
        /// Present when the model uses thinking and returns reasoning alongside
        /// tool calls or content. Per V3.2 docs: **keep** reasoning_content
        /// within a tool loop (same user question) so the model retains its
        /// logical thread, but **strip** it from prior conversation turns
        /// (previous user questions) to save bandwidth.
        #[serde(skip_serializing_if = "Option::is_none", default)]
        reasoning_content: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        tool_calls: Vec<LlmToolCall>,
    },
    #[serde(rename = "tool")]
    Tool {
        tool_call_id: String,
        content: String,
    },
}

/// A tool (function) definition sent to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

/// The function schema within a tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Controls how the model picks tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// "none", "auto", or "required"
    Mode(String),
    /// Force a specific function.
    Function {
        #[serde(rename = "type")]
        choice_type: String,
        function: ToolChoiceFunction,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

impl ToolChoice {
    pub fn auto() -> Self {
        Self::Mode("auto".to_string())
    }
    pub fn none() -> Self {
        Self::Mode("none".to_string())
    }
    /// Force the model to return at least one tool call.
    pub fn required() -> Self {
        Self::Mode("required".to_string())
    }
}

/// Request for the chat-with-tools API.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    /// When set, enables thinking mode on `deepseek-chat`.
    /// The API requires `temperature` to be omitted when thinking is enabled.
    pub thinking: Option<ThinkingConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AppConfig {
    pub llm: LlmConfig,
    pub router: RouterConfig,
    pub agent_loop: AgentLoopConfig,
    pub policy: PolicyConfig,
    pub tools: ToolsConfig,
    pub plugins: PluginsConfig,
    pub skills: SkillsConfig,
    pub usage: UsageConfig,
    pub context: ContextConfig,
    pub autopilot: AutopilotConfig,
    pub scheduling: SchedulingConfig,
    pub replay: ReplayConfig,
    pub ui: UiConfig,
    pub experiments: ExperimentsConfig,
    pub telemetry: TelemetryConfig,
    pub index: IndexConfig,
    pub budgets: BudgetsConfig,
    pub theme: ThemeConfig,
    /// Hooks configuration (maps event names to hook definitions).
    /// Stored as raw JSON, parsed by deepseek-hooks at runtime.
    #[serde(default)]
    pub hooks: serde_json::Value,
    /// Directory for storing plan files (default: .deepseek/plans).
    #[serde(default = "default_plans_directory")]
    pub plans_directory: String,
    /// Output style: adjusts system prompt tone ("concise", "verbose", "normal").
    #[serde(default = "default_output_style")]
    pub output_style: String,
    /// Preferred response language (e.g. "en", "zh", "ja").
    #[serde(default)]
    pub language: String,
    /// Git commit/PR attribution text.
    #[serde(default = "default_attribution")]
    pub attribution: String,
    /// Restrict available models to this list. Empty = all models available.
    #[serde(default)]
    pub available_models: Vec<String>,
    /// Session cleanup period in days (default: 30).
    #[serde(default = "default_cleanup_period_days")]
    pub cleanup_period_days: u32,
    /// Custom status line script (shell command, stdout replaces status line).
    #[serde(default)]
    pub status_line: String,
    /// Custom `@` file autocomplete script.
    #[serde(default)]
    pub file_suggestion: String,
    /// Custom spinner action verbs.
    #[serde(default)]
    pub spinner_verbs: Vec<String>,
    /// Whether to respect .gitignore when listing/searching files (default: true).
    #[serde(default = "default_respect_gitignore")]
    pub respect_gitignore: bool,
}

fn default_plans_directory() -> String {
    ".deepseek/plans".to_string()
}
fn default_output_style() -> String {
    "normal".to_string()
}
fn default_attribution() -> String {
    "DeepSeek CLI".to_string()
}
fn default_cleanup_period_days() -> u32 {
    30
}
fn default_respect_gitignore() -> bool {
    true
}
fn default_agent_loop_max_iterations() -> u64 {
    6
}
fn default_agent_loop_parse_retries() -> u64 {
    2
}
fn default_agent_loop_max_files_per_iteration() -> u64 {
    12
}
fn default_agent_loop_max_file_bytes() -> u64 {
    200_000
}
fn default_agent_loop_max_diff_bytes() -> u64 {
    400_000
}
fn default_agent_loop_verify_timeout_seconds() -> u64 {
    60
}

impl AppConfig {
    pub fn user_settings_path() -> Option<PathBuf> {
        let home = std::env::var("HOME")
            .ok()
            .or_else(|| std::env::var("USERPROFILE").ok())?;
        Some(Path::new(&home).join(".deepseek/settings.json"))
    }

    pub fn project_settings_path(workspace: &Path) -> PathBuf {
        runtime_dir(workspace).join("settings.json")
    }

    pub fn project_local_settings_path(workspace: &Path) -> PathBuf {
        runtime_dir(workspace).join("settings.local.json")
    }

    pub fn keybindings_path() -> Option<PathBuf> {
        let home = std::env::var("HOME")
            .ok()
            .or_else(|| std::env::var("USERPROFILE").ok())?;
        Some(Path::new(&home).join(".deepseek/keybindings.json"))
    }

    pub fn config_path(workspace: &Path) -> PathBuf {
        Self::project_settings_path(workspace)
    }

    pub fn legacy_toml_path(workspace: &Path) -> PathBuf {
        runtime_dir(workspace).join("config.toml")
    }

    pub fn load(workspace: &Path) -> Result<Self> {
        let mut merged = serde_json::to_value(Self::default())?;

        let legacy = Self::legacy_toml_path(workspace);
        if legacy.exists() {
            let raw = fs::read_to_string(legacy)?;
            let legacy_cfg: AppConfig = toml::from_str(&raw)?;
            merge_json_value(&mut merged, &serde_json::to_value(legacy_cfg)?);
        }

        let mut paths = Vec::new();
        if let Some(user) = Self::user_settings_path() {
            paths.push(user);
        }
        paths.push(Self::project_settings_path(workspace));
        paths.push(Self::project_local_settings_path(workspace));

        for path in paths {
            if !path.exists() {
                continue;
            }
            let raw = fs::read_to_string(path)?;
            let value: serde_json::Value = serde_json::from_str(&raw)?;
            merge_json_value(&mut merged, &value);
        }

        Ok(serde_json::from_value(merged)?)
    }

    pub fn ensure(workspace: &Path) -> Result<Self> {
        let path = Self::project_settings_path(workspace);
        if path.exists()
            || Self::project_local_settings_path(workspace).exists()
            || Self::legacy_toml_path(workspace).exists()
            || Self::user_settings_path().is_some_and(|p| p.exists())
        {
            return Self::load(workspace);
        }
        fs::create_dir_all(
            path.parent()
                .ok_or_else(|| anyhow::anyhow!("invalid config path"))?,
        )?;
        let cfg = Self::default();
        cfg.save(workspace)?;
        Ok(cfg)
    }

    pub fn save(&self, workspace: &Path) -> Result<()> {
        let path = Self::project_settings_path(workspace);
        fs::create_dir_all(
            path.parent()
                .ok_or_else(|| anyhow::anyhow!("invalid config path"))?,
        )?;
        fs::write(path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
}

fn merge_json_value(base: &mut serde_json::Value, overlay: &serde_json::Value) {
    match (base, overlay) {
        (serde_json::Value::Object(base_obj), serde_json::Value::Object(overlay_obj)) => {
            for (key, overlay_value) in overlay_obj {
                if let Some(base_value) = base_obj.get_mut(key) {
                    merge_json_value(base_value, overlay_value);
                } else {
                    base_obj.insert(key.clone(), overlay_value.clone());
                }
            }
        }
        (base_slot, overlay_value) => {
            *base_slot = overlay_value.clone();
        }
    }
}

/// Prompt caching strategy for API requests.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CacheStrategy {
    /// Annotate system prompt + first user message (stable prefix).
    #[default]
    Auto,
    /// Annotate system prompt + first 3 messages (more aggressive prefix caching).
    Aggressive,
    /// No cache annotations.
    Off,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    pub base_model: String,
    pub max_think_model: String,
    pub provider: String,
    pub profile: String,
    pub context_window_tokens: u64,
    pub temperature: f32,
    pub endpoint: String,
    pub api_key: Option<String>,
    pub api_key_env: String,
    pub fast_mode: bool,
    pub language: String,
    pub prompt_cache_enabled: bool,
    pub cache_strategy: CacheStrategy,
    pub timeout_seconds: u64,
    pub max_retries: u8,
    pub retry_base_ms: u64,
    pub stream: bool,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_model: DEEPSEEK_V32_CHAT_MODEL.to_string(),
            max_think_model: DEEPSEEK_V32_REASONER_MODEL.to_string(),
            provider: "deepseek".to_string(),
            profile: DEEPSEEK_PROFILE_V32.to_string(),
            context_window_tokens: 128_000,
            temperature: 0.2,
            endpoint: "https://api.deepseek.com/chat/completions".to_string(),
            api_key: None,
            api_key_env: "DEEPSEEK_API_KEY".to_string(),
            fast_mode: false,
            language: "en".to_string(),
            prompt_cache_enabled: true,
            cache_strategy: CacheStrategy::Auto,
            timeout_seconds: 60,
            max_retries: 3,
            retry_base_ms: 400,
            stream: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RouterConfig {
    pub auto_max_think: bool,
    pub threshold_high: f32,
    pub w1: f32,
    pub w2: f32,
    pub w3: f32,
    pub w4: f32,
    pub w5: f32,
    pub w6: f32,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            auto_max_think: true,
            threshold_high: 0.72,
            w1: 0.2,
            w2: 0.15,
            w3: 0.2,
            w4: 0.15,
            w5: 0.2,
            w6: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentLoopConfig {
    #[serde(default = "default_agent_loop_max_iterations")]
    pub max_iterations: u64,
    #[serde(default = "default_agent_loop_parse_retries")]
    pub architect_parse_retries: u64,
    #[serde(default = "default_agent_loop_parse_retries")]
    pub editor_parse_retries: u64,
    #[serde(default = "default_agent_loop_max_files_per_iteration")]
    pub max_files_per_iteration: u64,
    #[serde(default = "default_agent_loop_max_file_bytes")]
    pub max_file_bytes: u64,
    #[serde(default = "default_agent_loop_max_diff_bytes")]
    pub max_diff_bytes: u64,
    #[serde(default = "default_agent_loop_verify_timeout_seconds")]
    pub verify_timeout_seconds: u64,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: default_agent_loop_max_iterations(),
            architect_parse_retries: default_agent_loop_parse_retries(),
            editor_parse_retries: default_agent_loop_parse_retries(),
            max_files_per_iteration: default_agent_loop_max_files_per_iteration(),
            max_file_bytes: default_agent_loop_max_file_bytes(),
            max_diff_bytes: default_agent_loop_max_diff_bytes(),
            verify_timeout_seconds: default_agent_loop_verify_timeout_seconds(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SandboxConfig {
    pub enabled: bool,
    /// When true and sandbox is enabled, auto-approve bash commands without prompting.
    pub auto_allow_bash_if_sandboxed: bool,
    pub network: SandboxNetworkConfig,
    pub excluded_commands: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SandboxNetworkConfig {
    pub allowed_domains: Vec<String>,
    pub block_all: bool,
    /// Allow binding to localhost ports (e.g., for dev servers).
    pub allow_local_binding: bool,
    /// Allow Unix domain sockets.
    pub allow_unix_sockets: bool,
}

/// Controls whether edits or bash commands require interactive approval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApprovalMode {
    Ask,
    Always,
    Never,
}

impl std::fmt::Display for ApprovalMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ask => write!(f, "ask"),
            Self::Always => write!(f, "always"),
            Self::Never => write!(f, "never"),
        }
    }
}

impl std::str::FromStr for ApprovalMode {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ask" => Ok(Self::Ask),
            "always" => Ok(Self::Always),
            "never" | "false" | "off" => Ok(Self::Never),
            other => Err(anyhow::anyhow!(
                "invalid approval mode '{}' (expected ask|always|never)",
                other
            )),
        }
    }
}

/// Review mode controls which tools are sent to the LLM.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReviewMode {
    /// All tools available (default behavior).
    #[default]
    Off,
    /// All tools sent but write tools require explicit user approval before execution.
    Suggest,
    /// Only read-only tools are sent to the LLM — write tools are omitted entirely.
    Strict,
}

impl ReviewMode {
    /// Returns true if this tool name is read-only and allowed in strict mode.
    pub fn is_read_only_tool(name: &str) -> bool {
        matches!(
            name,
            "fs_read"
                | "fs_glob"
                | "fs_grep"
                | "fs_list"
                | "git_status"
                | "git_diff"
                | "git_show"
                | "git_log"
                | "web_search"
                | "web_fetch"
                | "notebook_read"
                | "index_query"
                | "think_deeply"
                | "spawn_task"
                | "task_output"
                | "task_list"
                | "task_get"
                | "user_question"
        )
    }
}

/// Sandbox enforcement mode for tool execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SandboxMode {
    Allowlist,
    Isolated,
    Off,
    ReadOnly,
    WorkspaceWrite,
}

impl std::fmt::Display for SandboxMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Allowlist => write!(f, "allowlist"),
            Self::Isolated => write!(f, "isolated"),
            Self::Off => write!(f, "off"),
            Self::ReadOnly => write!(f, "read-only"),
            Self::WorkspaceWrite => write!(f, "workspace-write"),
        }
    }
}

impl std::str::FromStr for SandboxMode {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "allowlist" => Ok(Self::Allowlist),
            "isolated" | "container" | "os-sandbox" | "os_sandbox" => Ok(Self::Isolated),
            "off" => Ok(Self::Off),
            "read-only" | "readonly" => Ok(Self::ReadOnly),
            "workspace-write" | "workspace_write" => Ok(Self::WorkspaceWrite),
            other => Err(anyhow::anyhow!(
                "invalid sandbox mode '{}' (expected allowlist|isolated|off|read-only|workspace-write)",
                other
            )),
        }
    }
}

/// Permission mode for the overall policy engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionMode {
    Ask,
    Auto,
    Locked,
}

impl std::fmt::Display for PermissionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ask => write!(f, "ask"),
            Self::Auto => write!(f, "auto"),
            Self::Locked => write!(f, "locked"),
        }
    }
}

impl std::str::FromStr for PermissionMode {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ask" => Ok(Self::Ask),
            "auto" => Ok(Self::Auto),
            "locked" => Ok(Self::Locked),
            other => Err(anyhow::anyhow!(
                "invalid permission mode '{}' (expected ask|auto|locked)",
                other
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PolicyConfig {
    pub approve_edits: ApprovalMode,
    pub approve_bash: ApprovalMode,
    pub allowlist: Vec<String>,
    pub block_paths: Vec<String>,
    pub redact_patterns: Vec<String>,
    pub sandbox_mode: SandboxMode,
    pub sandbox_wrapper: Option<String>,
    /// Permission mode: "ask" (default), "auto", or "locked".
    pub permission_mode: PermissionMode,
    /// Optional lint command to run automatically after fs.edit (e.g., "cargo fmt --check").
    pub lint_after_edit: Option<String>,
    /// OS-level sandbox configuration.
    #[serde(default)]
    pub sandbox: SandboxConfig,
    /// Review mode: controls which tools are sent to the LLM.
    #[serde(default)]
    pub review_mode: ReviewMode,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            approve_edits: ApprovalMode::Ask,
            approve_bash: ApprovalMode::Ask,
            allowlist: vec![
                "rg".to_string(),
                "git status".to_string(),
                "git diff".to_string(),
                "git show".to_string(),
                "cargo test".to_string(),
                "cargo fmt --check".to_string(),
                "cargo clippy".to_string(),
            ],
            block_paths: vec![
                ".env".to_string(),
                ".ssh".to_string(),
                ".aws".to_string(),
                ".gnupg".to_string(),
                "**/id_*".to_string(),
                "**/secret".to_string(),
            ],
            redact_patterns: vec![
                "(?i)(api[_-]?key|token|secret|password)\\s*[:=]\\s*['\\\"]?[a-z0-9_\\-]{8,}['\\\"]?".to_string(),
                "\\b\\d{3}-\\d{2}-\\d{4}\\b".to_string(),
                "(?i)\\b(mrn|medical_record_number|patient_id)\\s*[:=]\\s*[a-z0-9\\-]{4,}\\b".to_string(),
            ],
            sandbox_mode: SandboxMode::Allowlist,
            sandbox_wrapper: None,
            permission_mode: PermissionMode::Ask,
            lint_after_edit: None,
            sandbox: SandboxConfig::default(),
            review_mode: ReviewMode::Off,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PluginsConfig {
    pub enabled: bool,
    pub search_paths: Vec<String>,
    pub enable_hooks: bool,
    pub catalog: PluginCatalogConfig,
}

impl Default for PluginsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            search_paths: vec![".deepseek/plugins".to_string(), ".plugins".to_string()],
            enable_hooks: false,
            catalog: PluginCatalogConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PluginCatalogConfig {
    pub enabled: bool,
    pub index_url: String,
    pub signature_key: Option<String>,
    pub refresh_hours: u64,
}

impl Default for PluginCatalogConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            index_url: ".deepseek/plugins/catalog.json".to_string(),
            signature_key: Some("deepseek-local-dev-key".to_string()),
            refresh_hours: 24,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ManagedSettings {
    pub disable_bypass_permissions_mode: bool,
    pub allow_managed_permission_rules_only: bool,
    pub forced_permission_mode: Option<String>,
    pub blocked_tools: Vec<String>,
}

pub fn managed_settings_path() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        Some(PathBuf::from(
            "/Library/Application Support/DeepSeekCode/managed-settings.json",
        ))
    }
    #[cfg(target_os = "linux")]
    {
        Some(PathBuf::from("/etc/deepseek-code/managed-settings.json"))
    }
    #[cfg(target_os = "windows")]
    {
        Some(PathBuf::from(
            "C:\\Program Files\\DeepSeekCode\\managed-settings.json",
        ))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

pub fn load_managed_settings() -> Option<ManagedSettings> {
    let path = managed_settings_path()?;
    if !path.exists() {
        return None;
    }
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UsageConfig {
    pub show_statusline: bool,
    pub cost_per_million_input: f64,
    pub cost_per_million_output: f64,
}

impl Default for UsageConfig {
    fn default() -> Self {
        Self {
            show_statusline: true,
            cost_per_million_input: 0.27,
            cost_per_million_output: 1.10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextConfig {
    pub auto_compact_threshold: f32,
    pub compact_preview: bool,
    /// Number of recent messages to preserve during compaction.
    /// Higher values retain more context at the cost of more tokens.
    #[serde(default)]
    pub compaction_tail_window: Option<usize>,
    /// Tokens reserved for tool definitions + system prompt overhead.
    /// Subtracted from the context window before applying `auto_compact_threshold`.
    #[serde(default)]
    pub reserved_overhead_tokens: u64,
    /// Tokens reserved for the model's response.
    /// Ensures enough space remains for output after filling conversation history.
    #[serde(default)]
    pub response_budget_tokens: u64,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            auto_compact_threshold: 0.86,
            compact_preview: true,
            compaction_tail_window: None,
            reserved_overhead_tokens: 4_000,
            response_budget_tokens: 8_192,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AutopilotConfig {
    pub default_max_consecutive_failures: u64,
    pub heartbeat_interval_seconds: u64,
    pub persist_checkpoints: bool,
}

impl Default for AutopilotConfig {
    fn default() -> Self {
        Self {
            default_max_consecutive_failures: 10,
            heartbeat_interval_seconds: 5,
            persist_checkpoints: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    pub enable_tui: bool,
    pub keybindings_path: String,
    pub reduced_motion: bool,
    pub statusline_mode: String,
    /// Image fallback mode when terminal protocol does not support inline rendering.
    /// Allowed values: "open", "path", "none".
    pub image_fallback: String,
    /// Optional base URL used when generating teleport handoff links.
    pub handoff_base_url: Option<String>,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            enable_tui: true,
            keybindings_path: "~/.deepseek/keybindings.json".to_string(),
            reduced_motion: false,
            statusline_mode: "minimal".to_string(),
            image_fallback: "open".to_string(),
            handoff_base_url: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ToolsConfig {
    pub chrome: ChromeToolsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChromeToolsConfig {
    /// Keep deterministic placeholder fallbacks when live Chrome websocket
    /// is unavailable. Defaults to false (strict-live behavior).
    pub allow_stub_fallback: bool,
}

impl Default for ChromeToolsConfig {
    fn default() -> Self {
        Self {
            allow_stub_fallback: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SkillsConfig {
    pub paths: Vec<String>,
    pub hot_reload: bool,
}

impl Default for SkillsConfig {
    fn default() -> Self {
        Self {
            paths: vec![
                ".deepseek/skills".to_string(),
                "~/.deepseek/skills".to_string(),
            ],
            hot_reload: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SchedulingConfig {
    pub off_peak: bool,
    pub off_peak_start_hour: u8,
    pub off_peak_end_hour: u8,
    pub defer_non_urgent: bool,
    pub max_defer_seconds: u64,
}

impl Default for SchedulingConfig {
    fn default() -> Self {
        Self {
            off_peak: false,
            off_peak_start_hour: 0,
            off_peak_end_hour: 6,
            defer_non_urgent: false,
            max_defer_seconds: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReplayConfig {
    pub strict_mode: bool,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self { strict_mode: true }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ExperimentsConfig {
    pub visual_verification: bool,
    pub wasm_hooks: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TelemetryConfig {
    pub enabled: bool,
    pub endpoint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexConfig {
    pub enabled: bool,
    pub engine: String,
    pub watch_files: bool,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            engine: "tantivy".to_string(),
            watch_files: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BudgetsConfig {
    pub max_turn_duration_secs: u64,
    pub max_reasoner_tokens_per_session: u64,
    pub max_turns: Option<u64>,
    pub max_budget_usd: Option<f64>,
}

impl Default for BudgetsConfig {
    fn default() -> Self {
        Self {
            max_turn_duration_secs: 300,
            max_reasoner_tokens_per_session: 1_000_000,
            max_turns: None,
            max_budget_usd: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ThemeConfig {
    pub primary: String,
    pub secondary: String,
    pub error: String,
}

impl Default for ThemeConfig {
    fn default() -> Self {
        Self {
            primary: "Cyan".to_string(),
            secondary: "Yellow".to_string(),
            error: "Red".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;

    fn model_alias_strategy() -> impl Strategy<Value = &'static str> {
        prop_oneof![
            Just("deepseek-chat"),
            Just("deepseek-v3.2"),
            Just("v3.2"),
            Just("v3_2"),
            Just("deepseek-reasoner"),
            Just("deepseek-v3.2-reasoner"),
            Just("reasoner"),
            Just("deepseek-v3.2-speciale"),
            Just("v3_2_speciale"),
        ]
    }

    fn session_state_strategy() -> impl Strategy<Value = SessionState> {
        prop_oneof![
            Just(SessionState::Idle),
            Just(SessionState::Planning),
            Just(SessionState::ExecutingStep),
            Just(SessionState::AwaitingApproval),
            Just(SessionState::Verifying),
            Just(SessionState::Completed),
            Just(SessionState::Paused),
            Just(SessionState::Failed),
        ]
    }

    proptest! {
        #[test]
        fn deepseek_model_normalization_is_case_and_whitespace_tolerant(
            alias in model_alias_strategy(),
            left_ws in 0usize..3,
            right_ws in 0usize..3,
            upper in any::<bool>(),
        ) {
            let source = if upper {
                alias.to_ascii_uppercase()
            } else {
                alias.to_string()
            };
            let candidate = format!("{}{}{}", " ".repeat(left_ws), source, " ".repeat(right_ws));
            prop_assert!(normalize_deepseek_model(&candidate).is_some());
        }

        #[test]
        fn merge_json_value_is_idempotent_for_flat_objects(
            base in prop::collection::btree_map("[a-z]{1,8}", any::<i64>(), 0..12),
            overlay in prop::collection::btree_map("[a-z]{1,8}", any::<i64>(), 0..12),
        ) {
            let mut base_value = json!(base);
            let overlay_value = json!(overlay);
            merge_json_value(&mut base_value, &overlay_value);
            let once = base_value.clone();
            merge_json_value(&mut base_value, &overlay_value);
            prop_assert_eq!(base_value, once);
        }

        #[test]
        fn completed_state_does_not_jump_directly_to_execution(
            to in session_state_strategy()
        ) {
            if matches!(to, SessionState::ExecutingStep | SessionState::AwaitingApproval | SessionState::Verifying) {
                prop_assert!(!is_valid_session_state_transition(&SessionState::Completed, &to));
            }
        }
    }

    #[test]
    fn session_state_transition_allows_expected_recovery_paths() {
        assert!(is_valid_session_state_transition(
            &SessionState::Failed,
            &SessionState::Planning
        ));
        assert!(is_valid_session_state_transition(
            &SessionState::Paused,
            &SessionState::ExecutingStep
        ));
        assert!(!is_valid_session_state_transition(
            &SessionState::Idle,
            &SessionState::Completed
        ));
    }

    #[test]
    fn new_event_types_round_trip_via_serde() {
        let events = vec![
            EventKind::SessionStartedV1 {
                session_id: Uuid::now_v7(),
                workspace: "/tmp/test".to_string(),
            },
            EventKind::SessionResumedV1 {
                session_id: Uuid::now_v7(),
                events_replayed: 42,
            },
            EventKind::ToolDeniedV1 {
                invocation_id: Uuid::now_v7(),
                tool_name: "bash.run".to_string(),
                reason: "locked mode".to_string(),
            },
        ];
        for event in events {
            let serialized = serde_json::to_string(&event).expect("serialize");
            let deserialized: EventKind = serde_json::from_str(&serialized).expect("deserialize");
            let re_serialized = serde_json::to_string(&deserialized).expect("re-serialize");
            assert_eq!(serialized, re_serialized);
        }
    }

    #[test]
    fn approval_mode_serde_roundtrip() {
        for (mode, expected) in [
            (ApprovalMode::Ask, "\"ask\""),
            (ApprovalMode::Always, "\"always\""),
            (ApprovalMode::Never, "\"never\""),
        ] {
            let serialized = serde_json::to_string(&mode).expect("serialize");
            assert_eq!(serialized, expected);
            let deserialized: ApprovalMode =
                serde_json::from_str(&serialized).expect("deserialize");
            assert_eq!(deserialized, mode);
        }
    }

    #[test]
    fn sandbox_mode_serde_roundtrip() {
        for (mode, expected) in [
            (SandboxMode::Allowlist, "\"allowlist\""),
            (SandboxMode::Isolated, "\"isolated\""),
            (SandboxMode::Off, "\"off\""),
            (SandboxMode::ReadOnly, "\"read-only\""),
            (SandboxMode::WorkspaceWrite, "\"workspace-write\""),
        ] {
            let serialized = serde_json::to_string(&mode).expect("serialize");
            assert_eq!(serialized, expected);
            let deserialized: SandboxMode = serde_json::from_str(&serialized).expect("deserialize");
            assert_eq!(deserialized, mode);
        }
    }

    #[test]
    fn permission_mode_serde_roundtrip() {
        for (mode, expected) in [
            (PermissionMode::Ask, "\"ask\""),
            (PermissionMode::Auto, "\"auto\""),
            (PermissionMode::Locked, "\"locked\""),
        ] {
            let serialized = serde_json::to_string(&mode).expect("serialize");
            assert_eq!(serialized, expected);
            let deserialized: PermissionMode =
                serde_json::from_str(&serialized).expect("deserialize");
            assert_eq!(deserialized, mode);
        }
    }

    #[test]
    fn tool_name_roundtrip_all_variants() {
        for &tool in ToolName::ALL {
            let api = tool.as_api_name();
            let internal = tool.as_internal();
            assert_eq!(
                ToolName::from_api_name(api),
                Some(tool),
                "from_api_name roundtrip failed for {api}"
            );
            assert_eq!(
                ToolName::from_internal_name(internal),
                Some(tool),
                "from_internal_name roundtrip failed for {internal}"
            );
        }
    }

    #[test]
    fn tool_name_is_read_only_matches_plan_mode() {
        let read_only: Vec<&str> = ToolName::ALL
            .iter()
            .filter(|t| t.is_read_only())
            .map(|t| t.as_api_name())
            .collect();
        // Plan mode tools must be a subset of read-only tools
        for name in &[
            "fs_read",
            "fs_list",
            "fs_glob",
            "fs_grep",
            "git_status",
            "git_diff",
            "git_show",
            "web_fetch",
            "web_search",
            "index_query",
            "notebook_read",
            "diagnostics_check",
        ] {
            assert!(read_only.contains(name), "{name} should be read-only");
        }
        // Write tools must not be read-only
        for name in &["fs_write", "fs_edit", "bash_run", "notebook_edit"] {
            assert!(!read_only.contains(name), "{name} should not be read-only");
        }
    }

    #[test]
    fn tool_name_unknown_returns_none() {
        assert_eq!(ToolName::from_api_name("nonexistent_tool"), None);
        assert_eq!(ToolName::from_api_name("plugin__foo__bar"), None);
        assert_eq!(ToolName::from_api_name("mcp__server_tool"), None);
        assert_eq!(ToolName::from_internal_name("unknown.tool"), None);
    }

    #[test]
    fn event_kind_all_variants_have_category() {
        // Verify a representative from each category returns a non-empty string
        let events = vec![
            EventKind::SessionStartedV1 {
                session_id: Uuid::nil(),
                workspace: String::new(),
            },
            EventKind::ChatTurnV1 {
                message: ChatMessage::User {
                    content: String::new(),
                },
            },
            EventKind::ToolProposedV1 {
                proposal: ToolProposal {
                    invocation_id: Uuid::nil(),
                    call: ToolCall {
                        name: "fs.read".to_string(),
                        args: serde_json::json!({}),
                        requires_approval: false,
                    },
                    approved: false,
                },
            },
            EventKind::TaskCreatedV1 {
                task_id: Uuid::nil(),
                title: String::new(),
                priority: 0,
            },
            EventKind::PluginEnabledV1 {
                plugin_id: String::new(),
            },
            EventKind::HookExecutedV1 {
                phase: String::new(),
                hook_path: String::new(),
                success: true,
                timed_out: false,
                exit_code: Some(0),
            },
        ];
        for event in &events {
            let cat = event.category();
            assert!(!cat.is_empty(), "category should not be empty");
        }
        // Verify specific categories
        assert!(events[0].is_session_event());
        assert!(!events[0].is_tool_event());
        assert!(events[2].is_tool_event());
        assert!(!events[2].is_session_event());
    }
}
