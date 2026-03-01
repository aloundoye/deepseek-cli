use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

pub type Result<T> = anyhow::Result<T>;

// DeepSeek API model constants.
pub const CODINGBUDDY_V32_CHAT_MODEL: &str = "deepseek-chat";
pub const CODINGBUDDY_V32_REASONER_MODEL: &str = "deepseek-reasoner";
pub const CODINGBUDDY_PROFILE_V32: &str = "v3_2";

/// Maximum output tokens for deepseek-chat (V3 non-thinking).
pub const CODINGBUDDY_CHAT_MAX_OUTPUT_TOKENS: u32 = 8192;
/// Maximum output tokens for deepseek-chat with thinking enabled.
pub const CODINGBUDDY_CHAT_THINKING_MAX_OUTPUT_TOKENS: u32 = 32_768;
/// Maximum output tokens for deepseek-reasoner (thinking/R1).
pub const CODINGBUDDY_REASONER_MAX_OUTPUT_TOKENS: u32 = 65536;

pub fn normalize_codingbuddy_model(model: &str) -> Option<&'static str> {
    let normalized = model.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "deepseek-chat" | "codingbuddy-v3.2" | "codingbuddy-v3.2-chat" | "v3.2" | "v3_2" => {
            Some(CODINGBUDDY_V32_CHAT_MODEL)
        }
        "deepseek-reasoner"
        | "codingbuddy-v3.2-reasoner"
        | "reasoner"
        | "v3.2-reasoner"
        | "v3_2_reasoner" => Some(CODINGBUDDY_V32_REASONER_MODEL),
        _ => None,
    }
}

/// Returns true if the model name refers to the deepseek-reasoner (or any alias).
pub fn is_reasoner_model(model: &str) -> bool {
    normalize_codingbuddy_model(model) == Some(CODINGBUDDY_V32_REASONER_MODEL)
}

pub fn normalize_codingbuddy_profile(profile: &str) -> Option<&'static str> {
    let normalized = profile.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "" | "v3_2" | "v3.2" | "v32" | "codingbuddy-v3.2" => Some(CODINGBUDDY_PROFILE_V32),
        _ => None,
    }
}

pub fn runtime_dir(workspace: &Path) -> PathBuf {
    workspace.join(".codingbuddy")
}

/// Actions available in the rewind picker menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RewindAction {
    RestoreCodeAndConversation,
    RestoreConversationOnly,
    RestoreCodeOnly,
    Summarize,
    Cancel,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RunState {
    Context,
    Planning,
    GatherEvidence,
    Subagents,
    Executing,
    Completed,
    Recover,
    Final,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    pub run_id: Uuid,
    pub session_id: Uuid,
    pub status: RunState,
    pub prompt: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmUnit {
    Model(String),
    Planner,
    Executor,
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

#[derive(Debug, Clone, Deserialize)]
struct RawEventEnvelope {
    seq_no: u64,
    at: DateTime<Utc>,
    session_id: Uuid,
    kind: serde_json::Value,
}

fn parse_event_kind_compat_value(kind_value: serde_json::Value) -> Result<EventKind> {
    if let Ok(kind) = serde_json::from_value::<EventKind>(kind_value.clone()) {
        return Ok(kind);
    }
    let kind_obj = kind_value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("invalid event kind shape"))?;
    let kind_type = kind_obj
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let payload = kind_obj
        .get("payload")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    // Handle well-known removed/renamed event types explicitly.
    match kind_type {
        "RouterDecisionV1" => {
            return Ok(EventKind::TelemetryEvent {
                name: "legacy.router_decision".to_string(),
                properties: payload,
            });
        }
        "RouterEscalationV1" => {
            return Ok(EventKind::TelemetryEvent {
                name: "legacy.router_escalation".to_string(),
                properties: payload,
            });
        }
        _ => {}
    }

    // For known event types whose schema evolved (e.g. added fields with
    // defaults), inject default values and retry deserialization.
    if let serde_json::Value::Object(mut payload_map) = payload.clone() {
        // UsageUpdatedV1: added cache_hit_tokens / cache_miss_tokens
        if kind_type == "UsageUpdatedV1" {
            payload_map
                .entry("cache_hit_tokens")
                .or_insert(serde_json::Value::Number(0.into()));
            payload_map
                .entry("cache_miss_tokens")
                .or_insert(serde_json::Value::Number(0.into()));
        }

        let patched = serde_json::json!({
            "type": kind_type,
            "payload": payload_map,
        });
        if let Ok(kind) = serde_json::from_value::<EventKind>(patched) {
            return Ok(kind);
        }
    }

    // Last resort: wrap in a telemetry event so sessions with unknown future
    // event types can still load rather than crashing.
    Ok(EventKind::TelemetryEvent {
        name: format!("unknown.{}", kind_type),
        properties: payload,
    })
}

/// Parse an `EventKind` payload with read compatibility for removed router events.
pub fn parse_event_kind_compat(raw: &str) -> Result<EventKind> {
    let value: serde_json::Value = serde_json::from_str(raw)?;
    parse_event_kind_compat_value(value)
}

/// Parse an `EventEnvelope` line with read compatibility for removed router events.
pub fn parse_event_envelope_compat(raw: &str) -> Result<EventEnvelope> {
    let envelope: RawEventEnvelope = serde_json::from_str(raw)?;
    Ok(EventEnvelope {
        seq_no: envelope.seq_no,
        at: envelope.at,
        session_id: envelope.session_id,
        kind: parse_event_kind_compat_value(envelope.kind)?,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum EventKind {
    #[serde(alias = "TurnAddedV1")]
    TurnAdded { role: String, content: String },
    /// Structured chat turn with full tool_call data for accurate session resume.
    #[serde(alias = "ChatTurnV1")]
    ChatTurn { message: ChatMessage },
    /// Reverts the session to a prior state by dropping N turns.
    #[serde(alias = "TurnRevertedV1")]
    TurnReverted { turns_dropped: u32 },
    #[serde(alias = "SessionStateChangedV1")]
    SessionStateChanged {
        from: SessionState,
        to: SessionState,
    },
    #[serde(alias = "PlanCreatedV1")]
    PlanCreated { plan: Plan },
    #[serde(alias = "PlanRevisedV1")]
    PlanRevised { plan: Plan },
    #[serde(alias = "RunStartedV1")]
    RunStarted { run_id: Uuid, prompt: String },
    #[serde(alias = "RunStateChangedV1")]
    RunStateChanged {
        run_id: Uuid,
        from: RunState,
        to: RunState,
    },
    #[serde(alias = "RunCompletedV1")]
    RunCompleted { run_id: Uuid, success: bool },
    #[serde(alias = "StepMarkedV1")]
    StepMarked {
        step_id: Uuid,
        done: bool,
        note: String,
    },
    #[serde(alias = "ToolProposedV1")]
    ToolProposed { proposal: ToolProposal },
    #[serde(alias = "ToolApprovedV1")]
    ToolApproved { invocation_id: Uuid },
    #[serde(alias = "ToolResultV1")]
    ToolResult { result: ToolResult },
    #[serde(alias = "PatchStagedV1")]
    PatchStaged { patch_id: Uuid, base_sha256: String },
    #[serde(alias = "PatchAppliedV1")]
    PatchApplied {
        patch_id: Uuid,
        applied: bool,
        conflicts: Vec<String>,
    },
    #[serde(alias = "VerificationRunV1")]
    VerificationRun {
        command: String,
        success: bool,
        output: String,
    },
    #[serde(alias = "CommitProposalV1")]
    CommitProposal {
        files: Vec<String>,
        touched_files: u64,
        loc_delta: u64,
        verify_commands: Vec<String>,
        verify_status: String,
        suggested_message: String,
    },
    #[serde(alias = "PluginInstalledV1")]
    PluginInstalled { plugin_id: String, version: String },
    #[serde(alias = "PluginRemovedV1")]
    PluginRemoved { plugin_id: String },
    #[serde(alias = "PluginEnabledV1")]
    PluginEnabled { plugin_id: String },
    #[serde(alias = "PluginDisabledV1")]
    PluginDisabled { plugin_id: String },
    #[serde(alias = "UsageUpdatedV1")]
    UsageUpdated {
        unit: LlmUnit,
        model: String,
        input_tokens: u64,
        cache_hit_tokens: u64,
        cache_miss_tokens: u64,
        output_tokens: u64,
    },
    #[serde(alias = "ContextCompactedV1")]
    ContextCompacted {
        summary_id: Uuid,
        from_turn: u64,
        to_turn: u64,
        token_delta_estimate: i64,
        replay_pointer: String,
    },
    #[serde(alias = "AutopilotRunStartedV1")]
    AutopilotRunStarted { run_id: Uuid, prompt: String },
    #[serde(alias = "AutopilotRunHeartbeatV1")]
    AutopilotRunHeartbeat {
        run_id: Uuid,
        completed_iterations: u64,
        failed_iterations: u64,
        consecutive_failures: u64,
        last_error: Option<String>,
    },
    #[serde(alias = "AutopilotRunStoppedV1")]
    AutopilotRunStopped {
        run_id: Uuid,
        stop_reason: String,
        completed_iterations: u64,
        failed_iterations: u64,
    },
    #[serde(alias = "PluginCatalogSyncedV1")]
    PluginCatalogSynced {
        source: String,
        total: usize,
        verified_count: usize,
    },
    #[serde(alias = "PluginVerifiedV1")]
    PluginVerified {
        plugin_id: String,
        verified: bool,
        reason: String,
    },
    #[serde(alias = "CheckpointCreatedV1")]
    CheckpointCreated {
        checkpoint_id: Uuid,
        reason: String,
        files_count: u64,
        snapshot_path: String,
    },
    #[serde(alias = "CheckpointRewoundV1")]
    CheckpointRewound { checkpoint_id: Uuid, reason: String },
    #[serde(alias = "TranscriptExportedV1")]
    TranscriptExported {
        export_id: Uuid,
        format: String,
        output_path: String,
    },
    #[serde(alias = "McpServerAddedV1")]
    McpServerAdded {
        server_id: String,
        transport: String,
        endpoint: String,
    },
    #[serde(alias = "McpServerRemovedV1")]
    McpServerRemoved { server_id: String },
    #[serde(alias = "McpToolDiscoveredV1")]
    McpToolDiscovered {
        server_id: String,
        tool_name: String,
    },
    #[serde(alias = "SubagentSpawnedV1")]
    SubagentSpawned {
        run_id: Uuid,
        name: String,
        goal: String,
    },
    #[serde(alias = "SubagentCompletedV1")]
    SubagentCompleted { run_id: Uuid, output: String },
    #[serde(alias = "SubagentFailedV1")]
    SubagentFailed { run_id: Uuid, error: String },
    #[serde(alias = "CostUpdatedV1")]
    CostUpdated {
        input_tokens: u64,
        output_tokens: u64,
        estimated_cost_usd: f64,
    },
    #[serde(alias = "EffortChangedV1")]
    EffortChanged { level: String },
    #[serde(alias = "ProfileCapturedV1")]
    ProfileCaptured {
        profile_id: Uuid,
        summary: String,
        elapsed_ms: u64,
    },
    #[serde(alias = "MemorySyncedV1")]
    MemorySynced {
        version_id: Uuid,
        path: String,
        note: String,
    },
    #[serde(alias = "HookExecutedV1")]
    HookExecuted {
        phase: String,
        hook_path: String,
        success: bool,
        timed_out: bool,
        exit_code: Option<i32>,
    },
    #[serde(alias = "SessionForkedV1")]
    SessionForked {
        from_session_id: Uuid,
        to_session_id: Uuid,
    },
    #[serde(alias = "PermissionModeChangedV1")]
    PermissionModeChanged { from: String, to: String },
    #[serde(alias = "WebSearchExecutedV1")]
    WebSearchExecuted {
        query: String,
        results_count: u64,
        cached: bool,
    },
    #[serde(alias = "ReviewStartedV1")]
    ReviewStarted {
        review_id: Uuid,
        preset: String,
        target: String,
    },
    #[serde(alias = "ReviewCompletedV1")]
    ReviewCompleted {
        review_id: Uuid,
        findings_count: u64,
        critical_count: u64,
    },
    #[serde(alias = "ReviewPublishedV1")]
    ReviewPublished {
        review_id: Uuid,
        pr_number: u64,
        comments_published: u64,
        dry_run: bool,
    },
    #[serde(alias = "TaskCreatedV1")]
    TaskCreated {
        task_id: Uuid,
        title: String,
        priority: u32,
    },
    #[serde(alias = "TaskCompletedV1")]
    TaskCompleted { task_id: Uuid, outcome: String },
    #[serde(alias = "ArtifactBundledV1")]
    ArtifactBundled {
        task_id: Uuid,
        artifact_path: String,
        files: Vec<String>,
    },
    #[serde(alias = "TelemetryEventV1")]
    TelemetryEvent {
        name: String,
        properties: serde_json::Value,
    },
    #[serde(alias = "BackgroundJobStartedV1")]
    BackgroundJobStarted {
        job_id: Uuid,
        kind: String,
        reference: String,
    },
    #[serde(alias = "BackgroundJobResumedV1")]
    BackgroundJobResumed { job_id: Uuid, reference: String },
    #[serde(alias = "BackgroundJobStoppedV1")]
    BackgroundJobStopped { job_id: Uuid, reason: String },
    #[serde(alias = "SkillLoadedV1")]
    SkillLoaded {
        skill_id: String,
        source_path: String,
    },
    #[serde(alias = "ReplayExecutedV1")]
    ReplayExecuted {
        session_id: Uuid,
        deterministic: bool,
        events_replayed: u64,
    },
    #[serde(alias = "PromptCacheHitV1")]
    PromptCacheHit { cache_key: String, model: String },
    #[serde(alias = "OffPeakScheduledV1")]
    OffPeakScheduled {
        reason: String,
        resume_after: String,
    },
    #[serde(alias = "VisualArtifactCapturedV1")]
    VisualArtifactCaptured {
        artifact_id: Uuid,
        path: String,
        mime: String,
    },
    #[serde(alias = "RemoteEnvConfiguredV1")]
    RemoteEnvConfigured {
        profile_id: Uuid,
        name: String,
        endpoint: String,
    },
    #[serde(alias = "RemoteEnvExecutionStartedV1")]
    RemoteEnvExecutionStarted {
        execution_id: Uuid,
        profile_id: Uuid,
        mode: String,
        background: bool,
        reference: String,
    },
    #[serde(alias = "RemoteEnvExecutionCompletedV1")]
    RemoteEnvExecutionCompleted {
        execution_id: Uuid,
        profile_id: Uuid,
        mode: String,
        success: bool,
        duration_ms: u64,
        background: bool,
        reference: String,
    },
    #[serde(alias = "TeleportBundleCreatedV1")]
    TeleportBundleCreated { bundle_id: Uuid, path: String },
    #[serde(alias = "TeleportHandoffLinkCreatedV1")]
    TeleportHandoffLinkCreated {
        handoff_id: Uuid,
        session_id: Uuid,
        expires_at: String,
    },
    #[serde(alias = "TeleportHandoffLinkConsumedV1")]
    TeleportHandoffLinkConsumed {
        handoff_id: Uuid,
        session_id: Uuid,
        success: bool,
        reason: String,
    },
    #[serde(alias = "SessionStartedV1")]
    SessionStarted { session_id: Uuid, workspace: String },
    #[serde(alias = "SessionResumedV1")]
    SessionResumed {
        session_id: Uuid,
        events_replayed: u64,
    },
    #[serde(alias = "ToolDeniedV1")]
    ToolDenied {
        invocation_id: Uuid,
        tool_name: String,
        reason: String,
    },
    #[serde(alias = "NotebookEditedV1")]
    NotebookEdited {
        path: String,
        operation: String,
        cell_index: u64,
        cell_type: Option<String>,
    },
    #[serde(alias = "PdfTextExtractedV1")]
    PdfTextExtracted {
        path: String,
        pages: String,
        text_length: u64,
    },
    #[serde(alias = "IdeSessionStartedV1")]
    IdeSessionStarted {
        transport: String,
        client_info: String,
    },
    #[serde(alias = "TurnLimitExceededV1")]
    TurnLimitExceeded { limit: u64, actual: u64 },
    #[serde(alias = "BudgetExceededV1")]
    BudgetExceeded { limit_usd: f64, actual_usd: f64 },
    #[serde(alias = "TaskUpdatedV1")]
    TaskUpdated { task_id: String, status: String },
    #[serde(alias = "TaskDeletedV1")]
    TaskDeleted { task_id: String },
    #[serde(alias = "EnterPlanModeV1")]
    EnterPlanMode { session_id: Uuid },
    #[serde(alias = "ExitPlanModeV1")]
    ExitPlanMode { session_id: Uuid },
    #[serde(alias = "ProviderSelectedV1")]
    ProviderSelected { provider: String, model: String },
    /// Vector/code index was built from scratch.
    #[serde(alias = "IndexBuildV1")]
    IndexBuild {
        chunks_indexed: u64,
        files_processed: u64,
        duration_ms: u64,
    },
    /// Vector/code index was incrementally updated.
    #[serde(alias = "IndexUpdateV1")]
    IndexUpdate {
        chunks_added: u64,
        chunks_removed: u64,
        files_changed: u64,
        duration_ms: u64,
    },
    /// A retrieval query was executed against the vector index.
    #[serde(alias = "IndexQueryV1")]
    IndexQueryEvent {
        query: String,
        results_count: u64,
        duration_ms: u64,
    },
    /// Retrieval context was injected before an LLM call.
    #[serde(alias = "RetrievalV1")]
    Retrieval {
        query: String,
        chunks_injected: u64,
        token_estimate: u64,
    },
    /// Sensitive content was redacted by the privacy router.
    #[serde(alias = "PrivacyRedactionV1")]
    PrivacyRedaction {
        path: String,
        patterns_matched: u64,
        policy: String,
    },
    /// Content was blocked from cloud by the privacy router.
    #[serde(alias = "PrivacyBlockV1")]
    PrivacyBlock { path: String, reason: String },
    /// Local autocomplete / ghost text was generated.
    #[serde(alias = "AutocompleteV1")]
    Autocomplete {
        model_id: String,
        tokens_generated: u64,
        latency_ms: u64,
        accepted: bool,
    },
}

impl EventKind {
    /// Logical category for this event kind.
    #[must_use]
    pub fn category(&self) -> &'static str {
        match self {
            // Session lifecycle
            Self::SessionStarted { .. }
            | Self::SessionResumed { .. }
            | Self::SessionStateChanged { .. }
            | Self::SessionForked { .. } => "session",

            // Run lifecycle
            Self::RunStarted { .. } | Self::RunStateChanged { .. } | Self::RunCompleted { .. } => {
                "run"
            }

            // Chat / transcript
            Self::TurnAdded { .. }
            | Self::ChatTurn { .. }
            | Self::TurnReverted { .. }
            | Self::ContextCompacted { .. }
            | Self::EffortChanged { .. }
            | Self::PermissionModeChanged { .. }
            | Self::TurnLimitExceeded { .. }
            | Self::BudgetExceeded { .. } => "chat",

            // Tool invocations
            Self::ToolProposed { .. }
            | Self::ToolApproved { .. }
            | Self::ToolResult { .. }
            | Self::ToolDenied { .. } => "tool",

            // Plans
            Self::PlanCreated { .. }
            | Self::PlanRevised { .. }
            | Self::StepMarked { .. }
            | Self::EnterPlanMode { .. }
            | Self::ExitPlanMode { .. } => "plan",

            // Tasks
            Self::TaskCreated { .. }
            | Self::TaskCompleted { .. }
            | Self::TaskUpdated { .. }
            | Self::TaskDeleted { .. } => "task",

            // Model/provider selection
            Self::ProviderSelected { .. } => "model",

            // Patches (diff/apply)
            Self::PatchStaged { .. } | Self::PatchApplied { .. } => "patch",

            // Plugins
            Self::PluginInstalled { .. }
            | Self::PluginRemoved { .. }
            | Self::PluginEnabled { .. }
            | Self::PluginDisabled { .. }
            | Self::PluginCatalogSynced { .. }
            | Self::PluginVerified { .. } => "plugin",

            // Usage / cost tracking
            Self::UsageUpdated { .. } | Self::CostUpdated { .. } | Self::PromptCacheHit { .. } => {
                "usage"
            }

            // Autopilot
            Self::AutopilotRunStarted { .. }
            | Self::AutopilotRunHeartbeat { .. }
            | Self::AutopilotRunStopped { .. } => "autopilot",

            // Subagent orchestration
            Self::SubagentSpawned { .. }
            | Self::SubagentCompleted { .. }
            | Self::SubagentFailed { .. } => "subagent",

            // Background jobs
            Self::BackgroundJobStarted { .. }
            | Self::BackgroundJobResumed { .. }
            | Self::BackgroundJobStopped { .. } => "background",

            // MCP
            Self::McpServerAdded { .. }
            | Self::McpServerRemoved { .. }
            | Self::McpToolDiscovered { .. } => "mcp",

            // Skills
            Self::SkillLoaded { .. } => "skill",

            // Hooks
            Self::HookExecuted { .. } => "hook",

            // Memory
            Self::MemorySynced { .. } => "memory",

            // Profile / benchmark
            Self::ProfileCaptured { .. } => "profile",

            // Verification
            Self::VerificationRun { .. } => "verification",

            // Git workflow guidance
            Self::CommitProposal { .. } => "git",

            // Checkpoint / rewind
            Self::CheckpointCreated { .. } | Self::CheckpointRewound { .. } => "checkpoint",

            // Export
            Self::TranscriptExported { .. } => "export",

            // Web search
            Self::WebSearchExecuted { .. } => "web",

            // Review
            Self::ReviewStarted { .. }
            | Self::ReviewCompleted { .. }
            | Self::ReviewPublished { .. } => "review",

            // Artifact bundling
            Self::ArtifactBundled { .. } => "artifact",

            // Telemetry
            Self::TelemetryEvent { .. } => "telemetry",

            // Replay
            Self::ReplayExecuted { .. } => "replay",

            // Scheduling
            Self::OffPeakScheduled { .. } => "scheduling",

            // Visual testing
            Self::VisualArtifactCaptured { .. } => "visual",

            // Remote environments
            Self::RemoteEnvConfigured { .. }
            | Self::RemoteEnvExecutionStarted { .. }
            | Self::RemoteEnvExecutionCompleted { .. } => "remote_env",

            // Teleport
            Self::TeleportBundleCreated { .. }
            | Self::TeleportHandoffLinkCreated { .. }
            | Self::TeleportHandoffLinkConsumed { .. } => "teleport",

            // Notebook
            Self::NotebookEdited { .. } => "notebook",

            // PDF
            Self::PdfTextExtracted { .. } => "pdf",

            // IDE
            Self::IdeSessionStarted { .. } => "ide",

            // Local ML
            Self::IndexBuild { .. }
            | Self::IndexUpdate { .. }
            | Self::IndexQueryEvent { .. }
            | Self::Retrieval { .. } => "index",

            Self::PrivacyRedaction { .. } | Self::PrivacyBlock { .. } => "privacy",

            Self::Autocomplete { .. } => "autocomplete",
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
    /// Token usage from the API response.
    #[serde(default)]
    pub usage: Option<TokenUsage>,
}

/// Token usage information from a DeepSeek API response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub prompt_cache_hit_tokens: u64,
    #[serde(default)]
    pub prompt_cache_miss_tokens: u64,
    /// Tokens used for chain-of-thought reasoning (thinking mode).
    #[serde(default)]
    pub reasoning_tokens: u64,
}

/// A single chunk emitted during streaming.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// A content text delta.
    ContentDelta(String),
    /// A reasoning/thinking text delta.
    ReasoningDelta(String),
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
    /// Watch mode auto-triggered because comment digest changed.
    WatchTriggered { digest: u64, comment_count: usize },
    /// A security warning detected in tool output (prompt injection, suspicious patterns).
    SecurityWarning { message: String },
    /// Clear any previously streamed text — the response contains tool calls,
    /// so the interleaved text fragments should be discarded from the display.
    ClearStreamingText,
    /// Streaming token usage update — emitted after each LLM call for cost/progress tracking.
    UsageUpdate {
        input_tokens: u64,
        output_tokens: u64,
        cache_hit_tokens: u64,
        estimated_cost_usd: f64,
    },
    /// A per-step snapshot was recorded for undo capability.
    SnapshotRecorded {
        snapshot_id: String,
        tool_name: String,
        files: Vec<String>,
    },
    /// Streaming is done; the final assembled response follows.
    /// An optional reason string explains *why* the agent stopped
    /// (e.g. "max iterations reached", "plan dedup", content filter).
    Done { reason: Option<String> },
}

/// Canonical stream-json event representation used across CLI/TUI/RPC adapters.
pub fn stream_chunk_to_event_json(chunk: &StreamChunk) -> serde_json::Value {
    match chunk {
        StreamChunk::ContentDelta(text) => serde_json::json!({
            "type": "content_delta",
            "text": text,
        }),
        StreamChunk::ReasoningDelta(text) => serde_json::json!({
            "type": "reasoning_delta",
            "text": text,
        }),
        StreamChunk::ToolCallStart {
            tool_name,
            args_summary,
        } => serde_json::json!({
            "type": "tool_start",
            "tool_name": tool_name,
            "args_summary": args_summary,
        }),
        StreamChunk::ToolCallEnd {
            tool_name,
            duration_ms,
            success,
            summary,
        } => serde_json::json!({
            "type": "tool_end",
            "tool_name": tool_name,
            "duration_ms": duration_ms,
            "success": success,
            "summary": summary,
        }),
        StreamChunk::ModeTransition { from, to, reason } => serde_json::json!({
            "type": "mode_transition",
            "from": from,
            "to": to,
            "reason": reason,
        }),
        StreamChunk::SubagentSpawned { run_id, name, goal } => serde_json::json!({
            "type": "subagent_spawned",
            "run_id": run_id,
            "name": name,
            "goal": goal,
        }),
        StreamChunk::SubagentCompleted {
            run_id,
            name,
            summary,
        } => serde_json::json!({
            "type": "subagent_completed",
            "run_id": run_id,
            "name": name,
            "summary": summary,
        }),
        StreamChunk::SubagentFailed {
            run_id,
            name,
            error,
        } => serde_json::json!({
            "type": "subagent_failed",
            "run_id": run_id,
            "name": name,
            "error": error,
        }),
        StreamChunk::ImageData { label, .. } => serde_json::json!({
            "type": "image",
            "label": label,
        }),
        StreamChunk::WatchTriggered {
            digest,
            comment_count,
        } => serde_json::json!({
            "type": "watch_triggered",
            "digest": digest,
            "comment_count": comment_count,
        }),
        StreamChunk::SecurityWarning { message } => serde_json::json!({
            "type": "security_warning",
            "message": message,
        }),
        StreamChunk::ClearStreamingText => serde_json::json!({
            "type": "clear_streaming_text",
        }),
        StreamChunk::UsageUpdate {
            input_tokens,
            output_tokens,
            cache_hit_tokens,
            estimated_cost_usd,
        } => serde_json::json!({
            "type": "usage_update",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_hit_tokens": cache_hit_tokens,
            "estimated_cost_usd": estimated_cost_usd,
        }),
        StreamChunk::SnapshotRecorded {
            snapshot_id,
            tool_name,
            files,
        } => serde_json::json!({
            "type": "snapshot_recorded",
            "snapshot_id": snapshot_id,
            "tool_name": tool_name,
            "files": files,
        }),
        StreamChunk::Done { reason } => {
            let mut obj = serde_json::json!({ "type": "done" });
            if let Some(r) = reason {
                obj["reason"] = serde_json::json!(r);
            }
            obj
        }
    }
}

/// Callback type for receiving streaming chunks.
/// Uses `Arc<dyn Fn>` so it can be cloned across multiple turns in a chat loop.
pub type StreamCallback = std::sync::Arc<dyn Fn(StreamChunk) + Send + Sync>;

/// A thread-safe cancellation token for aborting in-progress streaming requests.
/// Set by the UI layer (e.g. on Ctrl+C), checked by the LLM streaming loop.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl CancellationToken {
    /// Create a new token in the "not cancelled" state.
    pub fn new() -> Self {
        Self {
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Signal cancellation.
    pub fn cancel(&self) {
        self.cancelled
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Check whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Reset the token to "not cancelled" for reuse across turns.
    pub fn reset(&self) {
        self.cancelled
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

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

/// Strip `reasoning_content` from all assistant messages that belong to
/// **prior** user-question turns, while keeping it for the **current** turn's
/// tool-call loop.
///
/// DeepSeek API lifecycle rules:
/// - Within a single user question's tool-call loop: **keep** reasoning_content
///   so the model retains its logical thread.
/// - When the next user question begins: **clear** reasoning_content from
///   earlier turns to save bandwidth and avoid API 400 errors.
///
/// Strategy: walk backward from the end; every assistant message *after* the
/// last User message belongs to the current turn (keep reasoning). Everything
/// before that last User message is a prior turn (strip reasoning).
pub fn strip_prior_reasoning_content(messages: &mut [ChatMessage]) {
    // Find the index of the last User message.
    let last_user_idx = messages
        .iter()
        .rposition(|m| matches!(m, ChatMessage::User { .. }));

    let boundary = match last_user_idx {
        Some(idx) => idx,
        None => return, // no user messages → nothing to strip
    };

    // Strip reasoning_content from all assistant messages before the boundary.
    for msg in &mut messages[..boundary] {
        if let ChatMessage::Assistant {
            reasoning_content, ..
        } = msg
        {
            *reasoning_content = None;
        }
    }
}

/// Rough token estimate: ~4 chars per token for English (conservative).
/// DeepSeek docs say ~0.3 tokens/char; we use 0.25 as safety margin.
pub fn estimate_message_tokens(messages: &[ChatMessage]) -> u64 {
    let total_chars: u64 = messages
        .iter()
        .map(|m| match m {
            ChatMessage::System { content } => content.len() as u64,
            ChatMessage::User { content } => content.len() as u64,
            ChatMessage::Assistant {
                content,
                reasoning_content,
                tool_calls,
            } => {
                content.as_deref().map_or(0, |c| c.len() as u64)
                    + reasoning_content.as_deref().map_or(0, |r| r.len() as u64)
                    + tool_calls
                        .iter()
                        .map(|tc| tc.arguments.len() as u64)
                        .sum::<u64>()
            }
            ChatMessage::Tool { content, .. } => content.len() as u64,
        })
        .sum();
    total_chars / 4
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
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub strict: Option<bool>,
    pub parameters: serde_json::Value,
}

/// Controls how the model picks tools.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    /// Request log probabilities. Incompatible with thinking mode (API will error).
    pub logprobs: Option<bool>,
    /// Number of top log probabilities to return. Incompatible with thinking mode.
    pub top_logprobs: Option<u8>,
    /// When set, enables thinking mode on `deepseek-chat`.
    /// The API requires temperature/top_p/presence_penalty/frequency_penalty to be
    /// omitted and logprobs/top_logprobs to not be set when thinking is enabled.
    pub thinking: Option<ThinkingConfig>,
    /// Optional images to include with the user message (multimodal).
    pub images: Vec<ImageContent>,
    /// Optional response format, e.g json_object
    pub response_format: Option<serde_json::Value>,
}

/// Request for the Beta Fill-In-the-Middle (FIM) Completion API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FimRequest {
    pub model: String,
    pub prompt: String,
    pub suffix: Option<String>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AppConfig {
    pub llm: LlmConfig,
    pub agent_loop: AgentLoopConfig,
    pub git: GitWorkflowConfig,
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
    pub local_ml: LocalMlConfig,
    /// Hooks configuration (maps event names to hook definitions).
    /// Stored as raw JSON, parsed by codingbuddy-hooks at runtime.
    #[serde(default)]
    pub hooks: serde_json::Value,
    /// Directory for storing plan files (default: .codingbuddy/plans).
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
    ".codingbuddy/plans".to_string()
}
fn default_output_style() -> String {
    "normal".to_string()
}
fn default_attribution() -> String {
    "CodingBuddy".to_string()
}
fn default_cleanup_period_days() -> u32 {
    30
}
fn default_respect_gitignore() -> bool {
    true
}
fn default_tool_loop_max_turns() -> u64 {
    50
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
fn default_agent_loop_max_context_requests_per_iteration() -> u64 {
    3
}
fn default_agent_loop_max_context_range_lines() -> u64 {
    400
}
fn default_agent_loop_context_bootstrap_enabled() -> bool {
    true
}
fn default_agent_loop_context_bootstrap_max_tree_entries() -> u64 {
    120
}
fn default_agent_loop_context_bootstrap_max_readme_bytes() -> u64 {
    24_000
}
fn default_agent_loop_context_bootstrap_max_manifest_bytes() -> u64 {
    16_000
}
fn default_agent_loop_context_bootstrap_max_repo_map_lines() -> u64 {
    80
}
fn default_agent_loop_context_bootstrap_max_audit_findings() -> u64 {
    20
}
fn default_failure_classifier_repeat_threshold() -> u64 {
    2
}
fn default_failure_classifier_similarity_threshold() -> f32 {
    0.8
}
fn default_failure_classifier_fingerprint_lines() -> u64 {
    40
}
fn default_safety_gate_max_files_without_approval() -> u64 {
    8
}
fn default_safety_gate_max_loc_without_approval() -> u64 {
    600
}
fn default_team_auto_enabled() -> bool {
    true
}
fn default_team_complexity_threshold() -> u64 {
    60
}
fn default_team_max_lanes() -> u64 {
    4
}
fn default_team_max_concurrency() -> u64 {
    2
}
fn default_git_auto_commit_on_verify_pass() -> bool {
    false
}
fn default_git_commit_message_template() -> String {
    "codingbuddy: {goal}".to_string()
}
fn default_git_require_signing() -> bool {
    false
}
fn default_ui_thinking_visibility() -> String {
    "concise".to_string()
}
fn default_ui_phase_heartbeat_ms() -> u64 {
    5000
}
fn default_ui_mission_control_max_events() -> u64 {
    400
}

impl AppConfig {
    pub fn user_settings_path() -> Option<PathBuf> {
        let home = std::env::var("HOME")
            .ok()
            .or_else(|| std::env::var("USERPROFILE").ok())?;
        Some(Path::new(&home).join(".codingbuddy/settings.json"))
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
        Some(Path::new(&home).join(".codingbuddy/keybindings.json"))
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

/// Prompt caching strategy — deprecated, kept only for config deserialization compat.
///
/// DeepSeek uses automatic server-side prefix caching. Client-side `cache_control`
/// annotations cause 400 errors. All variants are now no-ops.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CacheStrategy {
    #[default]
    Auto,
    Aggressive,
    Off,
}

/// Configuration for a single provider endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub base_url: String,
    pub api_key_env: String,
    pub models: ProviderModels,
}

/// Models available for a given provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderModels {
    pub chat: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoner: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    pub base_model: String,
    pub max_think_model: String,
    pub provider: String,
    #[serde(default = "default_providers")]
    pub providers: std::collections::HashMap<String, ProviderConfig>,
    pub profile: String,
    pub context_window_tokens: u64,
    pub temperature: f32,
    #[serde(default = "default_base_url")]
    pub base_url: String,
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
    /// When true, add `/v1` prefix to API paths for OpenAI SDK compatibility.
    #[serde(default)]
    pub openai_compat_prefix: bool,
}

impl LlmConfig {
    /// Returns the active provider config. Falls back to constructing one from
    /// the top-level fields if the provider key is not in the map.
    pub fn active_provider(&self) -> ProviderConfig {
        if let Some(p) = self.providers.get(&self.provider) {
            return p.clone();
        }
        // Fallback: build from legacy top-level fields
        ProviderConfig {
            base_url: self.base_url.clone(),
            api_key_env: self.api_key_env.clone(),
            models: ProviderModels {
                chat: self.base_model.clone(),
                reasoner: Some(self.max_think_model.clone()),
            },
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_model: CODINGBUDDY_V32_CHAT_MODEL.to_string(),
            max_think_model: CODINGBUDDY_V32_REASONER_MODEL.to_string(),
            provider: "deepseek".to_string(),
            providers: default_providers(),
            profile: CODINGBUDDY_PROFILE_V32.to_string(),
            context_window_tokens: 128_000,
            temperature: 0.2,
            base_url: "https://api.deepseek.com".to_string(),
            endpoint: "https://api.deepseek.com/chat/completions".to_string(),
            api_key: None,
            api_key_env: "DEEPSEEK_API_KEY".to_string(),
            fast_mode: false,
            language: "en".to_string(),
            prompt_cache_enabled: true,
            cache_strategy: CacheStrategy::Auto,
            timeout_seconds: 600,
            max_retries: 3,
            retry_base_ms: 400,
            stream: true,
            openai_compat_prefix: false,
        }
    }
}

fn default_providers() -> std::collections::HashMap<String, ProviderConfig> {
    let mut map = std::collections::HashMap::new();
    map.insert(
        "deepseek".to_string(),
        ProviderConfig {
            base_url: "https://api.deepseek.com".to_string(),
            api_key_env: "DEEPSEEK_API_KEY".to_string(),
            models: ProviderModels {
                chat: CODINGBUDDY_V32_CHAT_MODEL.to_string(),
                reasoner: Some(CODINGBUDDY_V32_REASONER_MODEL.to_string()),
            },
        },
    );
    map
}

fn default_base_url() -> String {
    "https://api.deepseek.com".to_string()
}

fn default_agent_loop_max_editor_apply_retries() -> u64 {
    3
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
    #[serde(default = "default_agent_loop_max_editor_apply_retries")]
    pub max_editor_apply_retries: u64,
    #[serde(default = "default_agent_loop_max_files_per_iteration")]
    pub max_files_per_iteration: u64,
    #[serde(default = "default_agent_loop_max_file_bytes")]
    pub max_file_bytes: u64,
    #[serde(default = "default_agent_loop_max_diff_bytes")]
    pub max_diff_bytes: u64,
    #[serde(default = "default_agent_loop_verify_timeout_seconds")]
    pub verify_timeout_seconds: u64,
    #[serde(default = "default_agent_loop_max_context_requests_per_iteration")]
    pub max_context_requests_per_iteration: u64,
    #[serde(default = "default_agent_loop_max_context_range_lines")]
    pub max_context_range_lines: u64,
    #[serde(default = "default_agent_loop_context_bootstrap_enabled")]
    pub context_bootstrap_enabled: bool,
    #[serde(default = "default_agent_loop_context_bootstrap_max_tree_entries")]
    pub context_bootstrap_max_tree_entries: u64,
    #[serde(default = "default_agent_loop_context_bootstrap_max_readme_bytes")]
    pub context_bootstrap_max_readme_bytes: u64,
    #[serde(default = "default_agent_loop_context_bootstrap_max_manifest_bytes")]
    pub context_bootstrap_max_manifest_bytes: u64,
    #[serde(default = "default_agent_loop_context_bootstrap_max_repo_map_lines")]
    pub context_bootstrap_max_repo_map_lines: u64,
    #[serde(default = "default_agent_loop_context_bootstrap_max_audit_findings")]
    pub context_bootstrap_max_audit_findings: u64,
    #[serde(default)]
    pub failure_classifier: FailureClassifierConfig,
    #[serde(default)]
    pub safety_gate: SafetyGateConfig,
    #[serde(default)]
    pub apply_strategy: ApplyStrategy,
    #[serde(default)]
    pub team: TeamOrchestrationConfig,
    #[serde(default)]
    pub lint: LintConfig,
    /// Maximum turns (LLM calls) for the tool-use loop. Defaults to 50.
    #[serde(default = "default_tool_loop_max_turns")]
    pub tool_loop_max_turns: u64,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: default_agent_loop_max_iterations(),
            architect_parse_retries: default_agent_loop_parse_retries(),
            editor_parse_retries: default_agent_loop_parse_retries(),
            max_editor_apply_retries: default_agent_loop_max_editor_apply_retries(),
            max_files_per_iteration: default_agent_loop_max_files_per_iteration(),
            max_file_bytes: default_agent_loop_max_file_bytes(),
            max_diff_bytes: default_agent_loop_max_diff_bytes(),
            verify_timeout_seconds: default_agent_loop_verify_timeout_seconds(),
            max_context_requests_per_iteration:
                default_agent_loop_max_context_requests_per_iteration(),
            max_context_range_lines: default_agent_loop_max_context_range_lines(),
            context_bootstrap_enabled: default_agent_loop_context_bootstrap_enabled(),
            context_bootstrap_max_tree_entries:
                default_agent_loop_context_bootstrap_max_tree_entries(),
            context_bootstrap_max_readme_bytes:
                default_agent_loop_context_bootstrap_max_readme_bytes(),
            context_bootstrap_max_manifest_bytes:
                default_agent_loop_context_bootstrap_max_manifest_bytes(),
            context_bootstrap_max_repo_map_lines:
                default_agent_loop_context_bootstrap_max_repo_map_lines(),
            context_bootstrap_max_audit_findings:
                default_agent_loop_context_bootstrap_max_audit_findings(),
            failure_classifier: FailureClassifierConfig::default(),
            safety_gate: SafetyGateConfig::default(),
            apply_strategy: ApplyStrategy::default(),
            team: TeamOrchestrationConfig::default(),
            lint: LintConfig::default(),
            tool_loop_max_turns: default_tool_loop_max_turns(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TeamOrchestrationConfig {
    #[serde(default = "default_team_auto_enabled")]
    pub auto_enabled: bool,
    #[serde(default = "default_team_complexity_threshold")]
    pub complexity_threshold: u64,
    #[serde(default = "default_team_max_lanes")]
    pub max_lanes: u64,
    #[serde(default = "default_team_max_concurrency")]
    pub max_concurrency: u64,
}

impl Default for TeamOrchestrationConfig {
    fn default() -> Self {
        Self {
            auto_enabled: default_team_auto_enabled(),
            complexity_threshold: default_team_complexity_threshold(),
            max_lanes: default_team_max_lanes(),
            max_concurrency: default_team_max_concurrency(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FailureClassifierConfig {
    #[serde(default = "default_failure_classifier_repeat_threshold")]
    pub repeat_threshold: u64,
    #[serde(default = "default_failure_classifier_similarity_threshold")]
    pub similarity_threshold: f32,
    #[serde(default = "default_failure_classifier_fingerprint_lines")]
    pub fingerprint_lines: u64,
}

impl Default for FailureClassifierConfig {
    fn default() -> Self {
        Self {
            repeat_threshold: default_failure_classifier_repeat_threshold(),
            similarity_threshold: default_failure_classifier_similarity_threshold(),
            fingerprint_lines: default_failure_classifier_fingerprint_lines(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SafetyGateConfig {
    #[serde(default = "default_safety_gate_max_files_without_approval")]
    pub max_files_without_approval: u64,
    #[serde(default = "default_safety_gate_max_loc_without_approval")]
    pub max_loc_without_approval: u64,
}

impl Default for SafetyGateConfig {
    fn default() -> Self {
        Self {
            max_files_without_approval: default_safety_gate_max_files_without_approval(),
            max_loc_without_approval: default_safety_gate_max_loc_without_approval(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LintConfig {
    /// Whether to run lint automatically after apply, before verify.
    #[serde(default)]
    pub enabled: bool,
    /// Per-language lint commands. Key = language/glob, value = command.
    /// Example: `{"rust": "cargo clippy --fix --allow-dirty", "python": "ruff check --fix"}`
    #[serde(default)]
    pub commands: std::collections::BTreeMap<String, String>,
    /// Maximum lint-fix iterations before giving up.
    #[serde(default = "default_lint_max_iterations")]
    pub max_iterations: u64,
    /// Timeout in seconds for each lint command.
    #[serde(default = "default_lint_timeout_seconds")]
    pub timeout_seconds: u64,
}

fn default_lint_max_iterations() -> u64 {
    3
}
fn default_lint_timeout_seconds() -> u64 {
    30
}

impl Default for LintConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            commands: std::collections::BTreeMap::new(),
            max_iterations: default_lint_max_iterations(),
            timeout_seconds: default_lint_timeout_seconds(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApplyStrategy {
    #[default]
    Auto,
    ThreeWay,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GitWorkflowConfig {
    #[serde(default = "default_git_auto_commit_on_verify_pass")]
    pub auto_commit_on_verify_pass: bool,
    #[serde(default = "default_git_commit_message_template")]
    pub commit_message_template: String,
    #[serde(default = "default_git_require_signing")]
    pub require_signing: bool,
    #[serde(default)]
    pub allowed_branch_patterns: Vec<String>,
    #[serde(default)]
    pub commit_message_regex: Option<String>,
}

impl Default for GitWorkflowConfig {
    fn default() -> Self {
        Self {
            auto_commit_on_verify_pass: default_git_auto_commit_on_verify_pass(),
            commit_message_template: default_git_commit_message_template(),
            require_signing: default_git_require_signing(),
            allowed_branch_patterns: Vec::new(),
            commit_message_regex: None,
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
                | "extended_thinking"
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
            search_paths: vec![".codingbuddy/plugins".to_string(), ".plugins".to_string()],
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
            index_url: ".codingbuddy/plugins/catalog.json".to_string(),
            signature_key: Some("codingbuddy-local-dev-key".to_string()),
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
            "/Library/Application Support/CodingBuddyCode/managed-settings.json",
        ))
    }
    #[cfg(target_os = "linux")]
    {
        Some(PathBuf::from("/etc/codingbuddy-code/managed-settings.json"))
    }
    #[cfg(target_os = "windows")]
    {
        Some(PathBuf::from(
            "C:\\Program Files\\CodingBuddyCode\\managed-settings.json",
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
    /// Thinking visibility mode in TUI.
    /// Allowed values: "concise", "raw".
    #[serde(default = "default_ui_thinking_visibility")]
    pub thinking_visibility: String,
    /// Heartbeat interval for active phase progress in milliseconds.
    #[serde(default = "default_ui_phase_heartbeat_ms")]
    pub phase_heartbeat_ms: u64,
    /// Maximum retained mission-control timeline events.
    #[serde(default = "default_ui_mission_control_max_events")]
    pub mission_control_max_events: u64,
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
            keybindings_path: "~/.codingbuddy/keybindings.json".to_string(),
            reduced_motion: false,
            statusline_mode: "minimal".to_string(),
            thinking_visibility: default_ui_thinking_visibility(),
            phase_heartbeat_ms: default_ui_phase_heartbeat_ms(),
            mission_control_max_events: default_ui_mission_control_max_events(),
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChromeToolsConfig {
    /// Keep deterministic placeholder fallbacks when live Chrome websocket
    /// is unavailable. Defaults to false (strict-live behavior).
    pub allow_stub_fallback: bool,
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
                ".codingbuddy/skills".to_string(),
                "~/.codingbuddy/skills".to_string(),
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

/// Configuration for local ML capabilities (embeddings, autocomplete, privacy routing).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LocalMlConfig {
    /// Master switch for all local ML features.
    pub enabled: bool,
    /// Compute device: "cpu", "cuda", "metal".
    pub device: String,
    /// Directory for cached model files.
    pub cache_dir: String,
    /// Embeddings model configuration.
    pub embeddings: EmbeddingsModelConfig,
    /// Vector index configuration.
    pub index: VectorIndexConfig,
    /// Local autocomplete / ghost text configuration.
    pub autocomplete: AutocompleteLocalConfig,
    /// Privacy routing configuration.
    pub privacy: PrivacyLocalConfig,
}

impl Default for LocalMlConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            device: "cpu".to_string(),
            cache_dir: ".codingbuddy/models".to_string(),
            embeddings: EmbeddingsModelConfig::default(),
            index: VectorIndexConfig::default(),
            autocomplete: AutocompleteLocalConfig::default(),
            privacy: PrivacyLocalConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingsModelConfig {
    pub enabled: bool,
    pub model_id: String,
    pub normalize: bool,
    pub batch_size: usize,
}

impl Default for EmbeddingsModelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            normalize: true,
            batch_size: 32,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VectorIndexConfig {
    pub chunk_lines: usize,
    pub chunk_overlap: usize,
    /// Use hybrid retrieval (vector + BM25).
    pub hybrid: bool,
    /// Blend alpha: 0.0 = pure BM25, 1.0 = pure vector.
    pub blend_alpha: f32,
    /// Maximum chunks to return per query.
    pub max_results: usize,
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self {
            chunk_lines: 50,
            chunk_overlap: 10,
            hybrid: true,
            blend_alpha: 0.7,
            max_results: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AutocompleteLocalConfig {
    pub enabled: bool,
    pub model_id: String,
    pub debounce_ms: u64,
    pub timeout_ms: u64,
    pub max_tokens: u32,
}

impl Default for AutocompleteLocalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_id: "qwen2.5-coder-3b".to_string(),
            debounce_ms: 200,
            timeout_ms: 2000,
            max_tokens: 128,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PrivacyLocalConfig {
    pub enabled: bool,
    pub sensitive_globs: Vec<String>,
    pub sensitive_regex: Vec<String>,
    pub policy: String,
    pub store_raw_in_logs: bool,
}

impl Default for PrivacyLocalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sensitive_globs: vec![
                "**/.env".to_string(),
                "**/.env.*".to_string(),
                "**/*.pem".to_string(),
                "**/*.key".to_string(),
            ],
            sensitive_regex: Vec::new(),
            policy: "redact".to_string(),
            store_raw_in_logs: false,
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
            Just("codingbuddy-v3.2"),
            Just("v3.2"),
            Just("v3_2"),
            Just("deepseek-reasoner"),
            Just("codingbuddy-v3.2-reasoner"),
            Just("reasoner"),
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
        fn codingbuddy_model_normalization_is_case_and_whitespace_tolerant(
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
            prop_assert!(normalize_codingbuddy_model(&candidate).is_some());
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
            EventKind::SessionStarted {
                session_id: Uuid::now_v7(),
                workspace: "/tmp/test".to_string(),
            },
            EventKind::SessionResumed {
                session_id: Uuid::now_v7(),
                events_replayed: 42,
            },
            EventKind::ToolDenied {
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
            EventKind::SessionStarted {
                session_id: Uuid::nil(),
                workspace: String::new(),
            },
            EventKind::ChatTurn {
                message: ChatMessage::User {
                    content: String::new(),
                },
            },
            EventKind::ToolProposed {
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
            EventKind::TaskCreated {
                task_id: Uuid::nil(),
                title: String::new(),
                priority: 0,
            },
            EventKind::PluginEnabled {
                plugin_id: String::new(),
            },
            EventKind::HookExecuted {
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

    #[test]
    fn parse_event_kind_compat_maps_legacy_router_event() {
        let raw = r#"{"type":"RouterDecisionV1","payload":{"decision":"v3","score":0.91}}"#;
        let kind = parse_event_kind_compat(raw).expect("compat parse");
        match kind {
            EventKind::TelemetryEvent { name, properties } => {
                assert_eq!(name, "legacy.router_decision");
                assert_eq!(properties["decision"], "v3");
                assert_eq!(properties["score"], 0.91);
            }
            other => panic!("unexpected mapped kind: {:?}", other),
        }
    }

    #[test]
    fn parse_event_envelope_compat_maps_legacy_router_event() {
        let session_id = Uuid::now_v7();
        let raw = format!(
            r#"{{"seq_no":7,"at":"2026-02-23T00:00:00Z","session_id":"{}","kind":{{"type":"RouterEscalationV1","payload":{{"reason":"repeat failures"}}}}}}"#,
            session_id
        );
        let envelope = parse_event_envelope_compat(&raw).expect("compat envelope parse");
        match envelope.kind {
            EventKind::TelemetryEvent { name, properties } => {
                assert_eq!(name, "legacy.router_escalation");
                assert_eq!(properties["reason"], "repeat failures");
            }
            other => panic!("unexpected mapped kind: {:?}", other),
        }
    }

    #[test]
    fn strip_prior_reasoning_clears_old_turns_keeps_current() {
        let mut messages = vec![
            ChatMessage::System {
                content: "system".to_string(),
            },
            ChatMessage::User {
                content: "first question".to_string(),
            },
            ChatMessage::Assistant {
                content: Some("answer 1".to_string()),
                reasoning_content: Some("thinking about q1".to_string()),
                tool_calls: vec![],
            },
            // New user turn starts here
            ChatMessage::User {
                content: "second question".to_string(),
            },
            ChatMessage::Assistant {
                content: Some("tool call".to_string()),
                reasoning_content: Some("thinking about q2".to_string()),
                tool_calls: vec![],
            },
        ];

        strip_prior_reasoning_content(&mut messages);

        // Prior turn (index 2): reasoning_content should be cleared
        if let ChatMessage::Assistant {
            reasoning_content, ..
        } = &messages[2]
        {
            assert!(
                reasoning_content.is_none(),
                "prior turn reasoning should be stripped"
            );
        } else {
            panic!("expected assistant at index 2");
        }

        // Current turn (index 4, after last User): reasoning_content should be kept
        if let ChatMessage::Assistant {
            reasoning_content, ..
        } = &messages[4]
        {
            assert_eq!(
                reasoning_content.as_deref(),
                Some("thinking about q2"),
                "current turn reasoning should be kept"
            );
        } else {
            panic!("expected assistant at index 4");
        }
    }

    #[test]
    fn strip_prior_reasoning_no_user_messages_is_noop() {
        let mut messages = vec![
            ChatMessage::System {
                content: "system".to_string(),
            },
            ChatMessage::Assistant {
                content: Some("answer".to_string()),
                reasoning_content: Some("thinking".to_string()),
                tool_calls: vec![],
            },
        ];
        strip_prior_reasoning_content(&mut messages);
        // No user messages → nothing stripped
        if let ChatMessage::Assistant {
            reasoning_content, ..
        } = &messages[1]
        {
            assert_eq!(reasoning_content.as_deref(), Some("thinking"));
        }
    }

    #[test]
    fn strip_prior_reasoning_single_turn_preserves_all() {
        let mut messages = vec![
            ChatMessage::User {
                content: "question".to_string(),
            },
            ChatMessage::Assistant {
                content: Some("answer".to_string()),
                reasoning_content: Some("thinking".to_string()),
                tool_calls: vec![],
            },
        ];
        strip_prior_reasoning_content(&mut messages);
        // Only one user turn → assistant after it is "current", reasoning kept
        if let ChatMessage::Assistant {
            reasoning_content, ..
        } = &messages[1]
        {
            assert_eq!(reasoning_content.as_deref(), Some("thinking"));
        }
    }

    // ── P0 estimate_message_tokens tests ──────────────────────────────

    #[test]
    fn estimate_message_tokens_basic() {
        // 400 chars → ~100 tokens
        let messages = vec![ChatMessage::User {
            content: "a".repeat(400),
        }];
        assert_eq!(estimate_message_tokens(&messages), 100);
    }

    #[test]
    fn estimate_message_tokens_empty() {
        assert_eq!(estimate_message_tokens(&[]), 0);
    }

    #[test]
    fn estimate_message_tokens_multi_role() {
        let messages = vec![
            ChatMessage::System {
                content: "x".repeat(100),
            },
            ChatMessage::User {
                content: "y".repeat(200),
            },
            ChatMessage::Assistant {
                content: Some("z".repeat(100)),
                reasoning_content: Some("r".repeat(200)),
                tool_calls: vec![],
            },
        ];
        // (100 + 200 + 100 + 200) / 4 = 150
        assert_eq!(estimate_message_tokens(&messages), 150);
    }

    // ── P0 strip_reasoning_from_prior_turns tests ─────────────────────

    #[test]
    fn strip_reasoning_from_prior_turns() {
        let mut messages = vec![
            ChatMessage::User {
                content: "turn 1 question".to_string(),
            },
            ChatMessage::Assistant {
                content: Some("turn 1 answer".to_string()),
                reasoning_content: Some("turn 1 thinking".to_string()),
                tool_calls: vec![],
            },
            ChatMessage::User {
                content: "turn 2 question".to_string(),
            },
            ChatMessage::Assistant {
                content: Some("turn 2 answer".to_string()),
                reasoning_content: Some("turn 2 thinking".to_string()),
                tool_calls: vec![],
            },
        ];
        strip_prior_reasoning_content(&mut messages);
        // Turn 1 assistant reasoning cleared
        if let ChatMessage::Assistant {
            reasoning_content, ..
        } = &messages[1]
        {
            assert!(
                reasoning_content.is_none(),
                "turn 1 reasoning should be stripped"
            );
        }
        // Turn 2 assistant reasoning kept (current turn)
        if let ChatMessage::Assistant {
            reasoning_content, ..
        } = &messages[3]
        {
            assert_eq!(reasoning_content.as_deref(), Some("turn 2 thinking"));
        }
    }

    #[test]
    fn strip_reasoning_preserves_current_tool_loop() {
        let mut messages = vec![
            ChatMessage::User {
                content: "question".to_string(),
            },
            ChatMessage::Assistant {
                content: Some("thinking step 1".to_string()),
                reasoning_content: Some("reasoning 1".to_string()),
                tool_calls: vec![LlmToolCall {
                    id: "c1".to_string(),
                    name: "test".to_string(),
                    arguments: "{}".to_string(),
                }],
            },
            ChatMessage::Tool {
                tool_call_id: "c1".to_string(),
                content: "result".to_string(),
            },
            ChatMessage::Assistant {
                content: Some("thinking step 2".to_string()),
                reasoning_content: Some("reasoning 2".to_string()),
                tool_calls: vec![],
            },
        ];
        strip_prior_reasoning_content(&mut messages);
        // All assistant messages are after the last User message (idx 0),
        // so they're all in the "current" turn and reasoning is preserved.
        if let ChatMessage::Assistant {
            reasoning_content, ..
        } = &messages[1]
        {
            assert_eq!(reasoning_content.as_deref(), Some("reasoning 1"));
        }
        if let ChatMessage::Assistant {
            reasoning_content, ..
        } = &messages[3]
        {
            assert_eq!(reasoning_content.as_deref(), Some("reasoning 2"));
        }
    }

    #[test]
    fn local_ml_config_defaults_correct() {
        let cfg = LocalMlConfig::default();
        assert!(!cfg.enabled, "local_ml must be disabled by default");
        assert_eq!(cfg.device, "cpu");
        assert_eq!(cfg.cache_dir, ".codingbuddy/models");
        assert!(cfg.embeddings.enabled);
        assert!(!cfg.autocomplete.enabled);
        assert!(!cfg.privacy.enabled);
        assert_eq!(cfg.index.chunk_lines, 50);
        assert_eq!(cfg.index.chunk_overlap, 10);
        assert!(cfg.index.hybrid);
    }

    #[test]
    fn local_ml_config_from_json_parsed() {
        let json = serde_json::json!({
            "local_ml": {
                "enabled": true,
                "device": "metal",
                "cache_dir": "/tmp/models",
                "embeddings": {
                    "enabled": true,
                    "model_id": "jina-code-v2",
                    "normalize": false,
                    "batch_size": 64
                },
                "index": {
                    "chunk_lines": 100,
                    "chunk_overlap": 20,
                    "hybrid": false,
                    "blend_alpha": 0.5,
                    "max_results": 20
                },
                "autocomplete": {
                    "enabled": true,
                    "model_id": "custom-model",
                    "debounce_ms": 300,
                    "timeout_ms": 3000,
                    "max_tokens": 256
                },
                "privacy": {
                    "enabled": true,
                    "sensitive_globs": ["**/.secret"],
                    "sensitive_regex": ["SSN-\\d+"],
                    "policy": "block_cloud",
                    "store_raw_in_logs": true
                }
            }
        });
        let cfg: AppConfig = serde_json::from_value(json).expect("parse");
        assert!(cfg.local_ml.enabled);
        assert_eq!(cfg.local_ml.device, "metal");
        assert_eq!(cfg.local_ml.cache_dir, "/tmp/models");
        assert_eq!(cfg.local_ml.embeddings.model_id, "jina-code-v2");
        assert!(!cfg.local_ml.embeddings.normalize);
        assert_eq!(cfg.local_ml.embeddings.batch_size, 64);
        assert_eq!(cfg.local_ml.index.chunk_lines, 100);
        assert!(!cfg.local_ml.index.hybrid);
        assert!(cfg.local_ml.autocomplete.enabled);
        assert_eq!(cfg.local_ml.autocomplete.max_tokens, 256);
        assert!(cfg.local_ml.privacy.enabled);
        assert_eq!(cfg.local_ml.privacy.policy, "block_cloud");
    }

    #[test]
    fn local_ml_disabled_by_default_in_appconfig() {
        let cfg = AppConfig::default();
        assert!(!cfg.local_ml.enabled);
    }

    #[test]
    fn default_config_has_bootstrap_enabled() {
        let cfg = AppConfig::default();
        assert!(
            cfg.agent_loop.context_bootstrap_enabled,
            "context bootstrap should be enabled by default"
        );
    }

    #[test]
    fn index_build_event_roundtrip() {
        let event = EventKind::IndexBuild {
            chunks_indexed: 1500,
            files_processed: 120,
            duration_ms: 3456,
        };
        let serialized = serde_json::to_string(&event).expect("serialize");
        let deserialized: EventKind = serde_json::from_str(&serialized).expect("deserialize");
        let re_serialized = serde_json::to_string(&deserialized).expect("re-serialize");
        assert_eq!(serialized, re_serialized);
    }

    #[test]
    fn privacy_event_roundtrip() {
        let events = vec![
            EventKind::PrivacyRedaction {
                path: "/project/.env".to_string(),
                patterns_matched: 3,
                policy: "redact".to_string(),
            },
            EventKind::PrivacyBlock {
                path: "/project/secrets.yaml".to_string(),
                reason: "block_cloud policy".to_string(),
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
    fn autocomplete_event_roundtrip() {
        let event = EventKind::Autocomplete {
            model_id: "qwen2.5-coder-3b".to_string(),
            tokens_generated: 42,
            latency_ms: 150,
            accepted: true,
        };
        let serialized = serde_json::to_string(&event).expect("serialize");
        let deserialized: EventKind = serde_json::from_str(&serialized).expect("deserialize");
        let re_serialized = serde_json::to_string(&deserialized).expect("re-serialize");
        assert_eq!(serialized, re_serialized);
    }

    #[test]
    fn cancellation_token_default_not_cancelled() {
        let token = CancellationToken::default();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn cancellation_token_cancel_and_reset() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
        token.reset();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn cancellation_token_clone_shares_state() {
        let token = CancellationToken::new();
        let clone = token.clone();
        token.cancel();
        assert!(
            clone.is_cancelled(),
            "clone should see cancellation from original"
        );
    }

    #[test]
    fn usage_update_json_serialization() {
        let chunk = StreamChunk::UsageUpdate {
            input_tokens: 1000,
            output_tokens: 500,
            cache_hit_tokens: 200,
            estimated_cost_usd: 0.0042,
        };
        let json = stream_chunk_to_event_json(&chunk);
        assert_eq!(json["type"], "usage_update");
        assert_eq!(json["input_tokens"], 1000);
        assert_eq!(json["output_tokens"], 500);
        assert_eq!(json["cache_hit_tokens"], 200);
        assert!((json["estimated_cost_usd"].as_f64().unwrap() - 0.0042).abs() < 0.0001);
    }

    #[test]
    fn active_provider_returns_deepseek_by_default() {
        let cfg = LlmConfig::default();
        let provider = cfg.active_provider();
        assert_eq!(provider.base_url, "https://api.deepseek.com");
        assert_eq!(provider.api_key_env, "DEEPSEEK_API_KEY");
        assert_eq!(provider.models.chat, "deepseek-chat");
        assert_eq!(
            provider.models.reasoner.as_deref(),
            Some("deepseek-reasoner")
        );
    }

    #[test]
    fn active_provider_falls_back_to_legacy_fields() {
        let cfg = LlmConfig {
            provider: "custom".to_string(),
            providers: std::collections::HashMap::new(), // no entry for "custom"
            base_url: "http://my-llm:8000".to_string(),
            api_key_env: "MY_KEY".to_string(),
            base_model: "my-model".to_string(),
            max_think_model: "my-reasoner".to_string(),
            ..LlmConfig::default()
        };
        let provider = cfg.active_provider();
        assert_eq!(provider.base_url, "http://my-llm:8000");
        assert_eq!(provider.api_key_env, "MY_KEY");
        assert_eq!(provider.models.chat, "my-model");
        assert_eq!(provider.models.reasoner.as_deref(), Some("my-reasoner"));
    }

    #[test]
    fn active_provider_uses_providers_map_when_available() {
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "openai-compat".to_string(),
            ProviderConfig {
                base_url: "http://localhost:11434/v1".to_string(),
                api_key_env: "OLLAMA_KEY".to_string(),
                models: ProviderModels {
                    chat: "llama3".to_string(),
                    reasoner: None,
                },
            },
        );
        let cfg = LlmConfig {
            provider: "openai-compat".to_string(),
            providers,
            ..LlmConfig::default()
        };
        let provider = cfg.active_provider();
        assert_eq!(provider.base_url, "http://localhost:11434/v1");
        assert_eq!(provider.api_key_env, "OLLAMA_KEY");
        assert_eq!(provider.models.chat, "llama3");
        assert!(provider.models.reasoner.is_none());
    }

    #[test]
    fn default_providers_includes_deepseek() {
        let providers = default_providers();
        assert!(providers.contains_key("deepseek"));
        let ds = &providers["deepseek"];
        assert_eq!(ds.base_url, "https://api.deepseek.com");
        assert_eq!(ds.models.chat, "deepseek-chat");
    }
}
