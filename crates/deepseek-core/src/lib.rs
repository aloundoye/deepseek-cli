use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

pub type Result<T> = anyhow::Result<T>;

// DeepSeek V3.2 API model aliases.
pub const DEEPSEEK_V32_CHAT_MODEL: &str = "deepseek-chat";
pub const DEEPSEEK_V32_REASONER_MODEL: &str = "deepseek-reasoner";

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
pub struct Failure {
    pub summary: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutcome {
    pub step_id: Uuid,
    pub success: bool,
    pub notes: String,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanContext {
    pub session: Session,
    pub user_prompt: String,
    pub prior_failures: Vec<Failure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecContext {
    pub session: Session,
    pub plan: Plan,
    pub approved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub args: serde_json::Value,
    pub requires_approval: bool,
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
    TelemetryEventV1 {
        name: String,
        properties: serde_json::Value,
    },
}

pub trait Planner {
    fn create_plan(&self, ctx: PlanContext) -> Result<Plan>;
    fn revise_plan(&self, ctx: PlanContext, last_plan: &Plan, failure: Failure) -> Result<Plan>;
}

pub trait Executor {
    fn run_step(&self, ctx: ExecContext, step: &PlanStep) -> Result<StepOutcome>;
}

pub trait ModelRouter {
    fn select(&self, unit: LlmUnit, signals: RouterSignals) -> RouterDecision;
}

pub trait ToolHost {
    fn propose(&self, call: ToolCall) -> ToolProposal;
    fn execute(&self, approved: ApprovedToolCall) -> ToolResult;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    pub unit: LlmUnit,
    pub prompt: String,
    pub model: String,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub text: String,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AppConfig {
    pub llm: LlmConfig,
    pub router: RouterConfig,
    pub policy: PolicyConfig,
    pub plugins: PluginsConfig,
    pub usage: UsageConfig,
    pub context: ContextConfig,
    pub autopilot: AutopilotConfig,
    pub ui: UiConfig,
    pub telemetry: TelemetryConfig,
    pub index: IndexConfig,
}

impl AppConfig {
    pub fn config_path(workspace: &Path) -> PathBuf {
        runtime_dir(workspace).join("config.toml")
    }

    pub fn load(workspace: &Path) -> Result<Self> {
        let path = Self::config_path(workspace);
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = fs::read_to_string(path)?;
        Ok(toml::from_str(&raw)?)
    }

    pub fn ensure(workspace: &Path) -> Result<Self> {
        let path = Self::config_path(workspace);
        if path.exists() {
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
        let path = Self::config_path(workspace);
        fs::create_dir_all(
            path.parent()
                .ok_or_else(|| anyhow::anyhow!("invalid config path"))?,
        )?;
        fs::write(path, toml::to_string_pretty(self)?)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    pub base_model: String,
    pub max_think_model: String,
    pub temperature: f32,
    pub endpoint: String,
    pub api_key_env: String,
    pub timeout_seconds: u64,
    pub max_retries: u8,
    pub retry_base_ms: u64,
    pub stream: bool,
    pub offline_fallback: bool,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_model: DEEPSEEK_V32_CHAT_MODEL.to_string(),
            max_think_model: DEEPSEEK_V32_REASONER_MODEL.to_string(),
            temperature: 0.2,
            endpoint: "https://api.deepseek.com/chat/completions".to_string(),
            api_key_env: "DEEPSEEK_API_KEY".to_string(),
            timeout_seconds: 60,
            max_retries: 3,
            retry_base_ms: 400,
            stream: true,
            offline_fallback: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RouterConfig {
    pub auto_max_think: bool,
    pub threshold_high: f32,
    pub escalate_on_invalid_plan: bool,
    pub max_escalations_per_unit: u8,
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
            escalate_on_invalid_plan: true,
            max_escalations_per_unit: 1,
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
pub struct PolicyConfig {
    pub approve_edits: String,
    pub approve_bash: String,
    pub allowlist: Vec<String>,
    pub sandbox_mode: String,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            approve_edits: "ask".to_string(),
            approve_bash: "ask".to_string(),
            allowlist: vec![
                "rg".to_string(),
                "git status".to_string(),
                "git diff".to_string(),
                "git show".to_string(),
                "cargo test".to_string(),
                "cargo fmt --check".to_string(),
                "cargo clippy".to_string(),
            ],
            sandbox_mode: "allowlist".to_string(),
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
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            auto_compact_threshold: 0.86,
            compact_preview: true,
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
    pub reduced_motion: bool,
    pub statusline_mode: String,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            reduced_motion: false,
            statusline_mode: "minimal".to_string(),
        }
    }
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
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            engine: "tantivy".to_string(),
        }
    }
}
