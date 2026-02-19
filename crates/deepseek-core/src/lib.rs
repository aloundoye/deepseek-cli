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
    TeleportBundleCreatedV1 {
        bundle_id: Uuid,
        path: String,
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
    /// Streaming is done; the final assembled response follows.
    Done,
}

/// Callback type for receiving streaming chunks.
pub type StreamCallback = Box<dyn FnMut(StreamChunk) + Send>;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AppConfig {
    pub llm: LlmConfig,
    pub router: RouterConfig,
    pub policy: PolicyConfig,
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
    pub block_paths: Vec<String>,
    pub redact_patterns: Vec<String>,
    pub sandbox_mode: String,
    pub sandbox_wrapper: Option<String>,
    /// Permission mode: "ask" (default), "auto", or "locked".
    pub permission_mode: String,
    /// Optional lint command to run automatically after fs.edit (e.g., "cargo fmt --check").
    pub lint_after_edit: Option<String>,
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
            sandbox_mode: "allowlist".to_string(),
            sandbox_wrapper: None,
            permission_mode: "ask".to_string(),
            lint_after_edit: None,
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
    pub enable_tui: bool,
    pub keybindings_path: String,
    pub reduced_motion: bool,
    pub statusline_mode: String,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            enable_tui: true,
            keybindings_path: "~/.deepseek/keybindings.json".to_string(),
            reduced_motion: false,
            statusline_mode: "minimal".to_string(),
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
}
