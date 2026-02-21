//! Mode router: selects the execution mode for each agent turn.
//!
//! Three modes:
//! - **V3Autopilot**: `deepseek-chat` with thinking + tools (fast, default).
//! - **R1Supervise**: Existing `reasoner_directed` loop (R1 directives + V3 executor).
//! - **R1DriveTools**: R1 emits tool-intent JSON, orchestrator executes, R1 iterates.

use crate::observation::{ErrorClass, ObservationPack};
use serde::{Deserialize, Serialize};

/// The active execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentMode {
    /// V3 with thinking + tools in a single call. Fast default.
    V3Autopilot,
    /// R1 produces directives, V3 executes tool calls (existing reasoner_directed).
    R1Supervise,
    /// R1 drives tools step-by-step via JSON intents. Orchestrator executes.
    R1DriveTools,
}

impl std::fmt::Display for AgentMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::V3Autopilot => write!(f, "v3_autopilot"),
            Self::R1Supervise => write!(f, "r1_supervise"),
            Self::R1DriveTools => write!(f, "r1_drive_tools"),
        }
    }
}

/// Configuration for mode routing thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModeRouterConfig {
    /// Enable mode routing (false = always V3Autopilot).
    pub enabled: bool,
    /// Max consecutive failures on same step before escalating from V3 to R1.
    pub v3_max_step_failures: u32,
    /// Max files changed since last green verify before escalating.
    pub blast_radius_threshold: u32,
    /// Allow V3 one bounded recovery attempt for mechanical errors.
    pub v3_mechanical_recovery: bool,
    /// Max total R1 drive-tools steps per task.
    pub r1_max_steps: u32,
    /// Max retries when R1 output fails schema validation.
    pub r1_max_parse_retries: u32,
    /// Max context requests from V3 patch writer before giving up.
    pub v3_patch_max_context_requests: u32,
}

impl Default for ModeRouterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            v3_max_step_failures: 2,
            blast_radius_threshold: 5,
            v3_mechanical_recovery: true,
            r1_max_steps: 30,
            r1_max_parse_retries: 2,
            v3_patch_max_context_requests: 3,
        }
    }
}

impl ModeRouterConfig {
    /// Construct from the core `RouterConfig` fields.
    pub fn from_router_config(rc: &deepseek_core::RouterConfig) -> Self {
        Self {
            enabled: rc.mode_router_enabled,
            v3_max_step_failures: rc.v3_max_step_failures,
            blast_radius_threshold: rc.blast_radius_threshold,
            v3_mechanical_recovery: rc.v3_mechanical_recovery,
            r1_max_steps: rc.r1_max_steps,
            r1_max_parse_retries: rc.r1_max_parse_retries,
            v3_patch_max_context_requests: rc.v3_patch_max_context_requests,
        }
    }
}

/// Tracks per-step failure state for escalation decisions.
#[derive(Debug, Clone, Default)]
pub struct FailureTracker {
    /// Consecutive failures on the current logical step.
    pub consecutive_step_failures: u32,
    /// Whether V3 already used its mechanical recovery attempt.
    pub v3_recovery_used: bool,
    /// Files changed since last green verification.
    pub files_changed_since_verify: Vec<String>,
    /// Module/package names seen in recent errors.
    pub error_modules: Vec<String>,
    /// Total R1 drive-tools steps used.
    pub r1_steps_used: u32,
}

impl FailureTracker {
    /// Record a successful action (resets step failure counter).
    pub fn record_success(&mut self) {
        self.consecutive_step_failures = 0;
    }

    /// Record a failure.
    pub fn record_failure(&mut self) {
        self.consecutive_step_failures += 1;
    }

    /// Record a successful verification (resets blast radius tracking).
    pub fn record_verify_pass(&mut self) {
        self.files_changed_since_verify.clear();
        self.error_modules.clear();
        self.consecutive_step_failures = 0;
        self.v3_recovery_used = false;
    }

    /// Record a file being changed.
    pub fn record_file_change(&mut self, path: &str) {
        if !self.files_changed_since_verify.contains(&path.to_string()) {
            self.files_changed_since_verify.push(path.to_string());
        }
    }

    /// Record an error module.
    pub fn record_error_module(&mut self, module: &str) {
        if !self.error_modules.contains(&module.to_string()) {
            self.error_modules.push(module.to_string());
        }
    }
}

/// Reason for a mode transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EscalationReason {
    /// Same step failed too many times.
    RepeatedStepFailure,
    /// Errors are ambiguous or unclassifiable.
    AmbiguousError,
    /// Too many files changed without verification.
    BlastRadiusExceeded,
    /// Errors span multiple modules.
    CrossModuleFailure,
    /// Task is architecturally complex (from initial classification).
    ArchitecturalTask,
    /// R1 drive-tools budget exhausted, fall back to V3.
    R1BudgetExhausted,
}

impl std::fmt::Display for EscalationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RepeatedStepFailure => write!(f, "repeated_step_failure"),
            Self::AmbiguousError => write!(f, "ambiguous_error"),
            Self::BlastRadiusExceeded => write!(f, "blast_radius_exceeded"),
            Self::CrossModuleFailure => write!(f, "cross_module_failure"),
            Self::ArchitecturalTask => write!(f, "architectural_task"),
            Self::R1BudgetExhausted => write!(f, "r1_budget_exhausted"),
        }
    }
}

/// Result of a mode routing decision.
#[derive(Debug, Clone)]
pub struct ModeDecision {
    pub mode: AgentMode,
    pub reason: Option<EscalationReason>,
}

/// Decide the execution mode based on current state.
///
/// Returns V3Autopilot by default, escalates to R1DriveTools when:
/// - Same step fails >= v3_max_step_failures times
/// - Error is ambiguous
/// - Blast radius exceeds threshold
/// - Errors span multiple modules
pub fn decide_mode(
    config: &ModeRouterConfig,
    current_mode: AgentMode,
    tracker: &FailureTracker,
    observation: Option<&ObservationPack>,
) -> ModeDecision {
    if !config.enabled {
        return ModeDecision {
            mode: AgentMode::V3Autopilot,
            reason: None,
        };
    }

    // If already in R1DriveTools, check budget
    if current_mode == AgentMode::R1DriveTools {
        if tracker.r1_steps_used >= config.r1_max_steps {
            return ModeDecision {
                mode: AgentMode::V3Autopilot,
                reason: Some(EscalationReason::R1BudgetExhausted),
            };
        }
        // Stay in R1 mode until done/abort
        return ModeDecision {
            mode: AgentMode::R1DriveTools,
            reason: None,
        };
    }

    // Check escalation triggers from V3Autopilot or R1Supervise

    // 1. Repeated step failures
    if tracker.consecutive_step_failures >= config.v3_max_step_failures {
        // Allow one mechanical recovery attempt for V3
        if config.v3_mechanical_recovery
            && !tracker.v3_recovery_used
            && observation
                .map(|o| o.error_class.is_mechanical())
                .unwrap_or(false)
        {
            // Don't escalate yet — let V3 try recovery
            return ModeDecision {
                mode: AgentMode::V3Autopilot,
                reason: None,
            };
        }
        return ModeDecision {
            mode: AgentMode::R1DriveTools,
            reason: Some(EscalationReason::RepeatedStepFailure),
        };
    }

    // 2. Ambiguous errors
    if let Some(obs) = observation {
        if obs.error_class == ErrorClass::Ambiguous {
            return ModeDecision {
                mode: AgentMode::R1DriveTools,
                reason: Some(EscalationReason::AmbiguousError),
            };
        }
    }

    // 3. Blast radius
    if tracker.files_changed_since_verify.len() as u32 >= config.blast_radius_threshold {
        return ModeDecision {
            mode: AgentMode::R1DriveTools,
            reason: Some(EscalationReason::BlastRadiusExceeded),
        };
    }

    // 4. Cross-module failures
    if tracker.error_modules.len() >= 2 {
        return ModeDecision {
            mode: AgentMode::R1DriveTools,
            reason: Some(EscalationReason::CrossModuleFailure),
        };
    }

    // No escalation needed — stay in current mode
    ModeDecision {
        mode: current_mode,
        reason: None,
    }
}

/// Classify whether a task is "architectural" based on prompt keywords.
///
/// Returns true if the prompt suggests multi-file refactoring, API changes, etc.
pub fn is_architectural_task(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    let indicators = [
        "refactor",
        "restructure",
        "redesign",
        "migrate",
        "rewrite",
        "architecture",
        "api boundary",
        "split module",
        "merge module",
        "rename across",
        "move to new",
        "extract crate",
        "cross-cutting",
    ];
    indicators.iter().any(|kw| lower.contains(kw))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> ModeRouterConfig {
        ModeRouterConfig::default()
    }

    #[test]
    fn default_mode_is_v3_autopilot() {
        let config = default_config();
        let tracker = FailureTracker::default();
        let decision = decide_mode(&config, AgentMode::V3Autopilot, &tracker, None);
        assert_eq!(decision.mode, AgentMode::V3Autopilot);
        assert!(decision.reason.is_none());
    }

    #[test]
    fn disabled_config_always_returns_v3() {
        let config = ModeRouterConfig {
            enabled: false,
            ..default_config()
        };
        let mut tracker = FailureTracker::default();
        tracker.consecutive_step_failures = 100;
        let decision = decide_mode(&config, AgentMode::V3Autopilot, &tracker, None);
        assert_eq!(decision.mode, AgentMode::V3Autopilot);
    }

    #[test]
    fn escalates_on_repeated_failures() {
        let config = ModeRouterConfig {
            v3_max_step_failures: 2,
            v3_mechanical_recovery: false,
            ..default_config()
        };
        let mut tracker = FailureTracker::default();
        tracker.record_failure();
        tracker.record_failure();
        let decision = decide_mode(&config, AgentMode::V3Autopilot, &tracker, None);
        assert_eq!(decision.mode, AgentMode::R1DriveTools);
        assert_eq!(decision.reason, Some(EscalationReason::RepeatedStepFailure));
    }

    #[test]
    fn mechanical_recovery_defers_escalation() {
        let config = ModeRouterConfig {
            v3_max_step_failures: 2,
            v3_mechanical_recovery: true,
            ..default_config()
        };
        let mut tracker = FailureTracker::default();
        tracker.record_failure();
        tracker.record_failure();

        let obs = ObservationPack {
            step: 1,
            actions: vec![],
            stderr_summary: String::new(),
            error_class: ErrorClass::CompileError, // mechanical
            changed_files: vec![],
            diff_summary: String::new(),
            test_summary: None,
            repo: crate::observation::RepoFacts::default(),
            since_last_verify: String::new(),
        };
        let decision = decide_mode(&config, AgentMode::V3Autopilot, &tracker, Some(&obs));
        // Should NOT escalate yet — mechanical recovery allowed
        assert_eq!(decision.mode, AgentMode::V3Autopilot);

        // After recovery used, should escalate
        let mut tracker2 = tracker.clone();
        tracker2.v3_recovery_used = true;
        let decision2 = decide_mode(&config, AgentMode::V3Autopilot, &tracker2, Some(&obs));
        assert_eq!(decision2.mode, AgentMode::R1DriveTools);
    }

    #[test]
    fn escalates_on_ambiguous_error() {
        let config = default_config();
        let tracker = FailureTracker::default();
        let obs = ObservationPack {
            step: 1,
            actions: vec![],
            stderr_summary: String::new(),
            error_class: ErrorClass::Ambiguous,
            changed_files: vec![],
            diff_summary: String::new(),
            test_summary: None,
            repo: crate::observation::RepoFacts::default(),
            since_last_verify: String::new(),
        };
        let decision = decide_mode(&config, AgentMode::V3Autopilot, &tracker, Some(&obs));
        assert_eq!(decision.mode, AgentMode::R1DriveTools);
        assert_eq!(decision.reason, Some(EscalationReason::AmbiguousError));
    }

    #[test]
    fn escalates_on_blast_radius() {
        let config = ModeRouterConfig {
            blast_radius_threshold: 3,
            ..default_config()
        };
        let mut tracker = FailureTracker::default();
        tracker.record_file_change("a.rs");
        tracker.record_file_change("b.rs");
        tracker.record_file_change("c.rs");
        let decision = decide_mode(&config, AgentMode::V3Autopilot, &tracker, None);
        assert_eq!(decision.mode, AgentMode::R1DriveTools);
        assert_eq!(decision.reason, Some(EscalationReason::BlastRadiusExceeded));
    }

    #[test]
    fn escalates_on_cross_module() {
        let config = default_config();
        let mut tracker = FailureTracker::default();
        tracker.record_error_module("auth");
        tracker.record_error_module("config");
        let decision = decide_mode(&config, AgentMode::V3Autopilot, &tracker, None);
        assert_eq!(decision.mode, AgentMode::R1DriveTools);
        assert_eq!(decision.reason, Some(EscalationReason::CrossModuleFailure));
    }

    #[test]
    fn r1_budget_exhaustion_falls_back() {
        let config = ModeRouterConfig {
            r1_max_steps: 5,
            ..default_config()
        };
        let mut tracker = FailureTracker::default();
        tracker.r1_steps_used = 5;
        let decision = decide_mode(&config, AgentMode::R1DriveTools, &tracker, None);
        assert_eq!(decision.mode, AgentMode::V3Autopilot);
        assert_eq!(decision.reason, Some(EscalationReason::R1BudgetExhausted));
    }

    #[test]
    fn stays_in_r1_drive_within_budget() {
        let config = default_config();
        let mut tracker = FailureTracker::default();
        tracker.r1_steps_used = 3;
        let decision = decide_mode(&config, AgentMode::R1DriveTools, &tracker, None);
        assert_eq!(decision.mode, AgentMode::R1DriveTools);
    }

    #[test]
    fn verify_pass_resets_tracker() {
        let mut tracker = FailureTracker::default();
        tracker.consecutive_step_failures = 5;
        tracker.v3_recovery_used = true;
        tracker.record_file_change("a.rs");
        tracker.record_error_module("auth");
        tracker.record_verify_pass();
        assert_eq!(tracker.consecutive_step_failures, 0);
        assert!(!tracker.v3_recovery_used);
        assert!(tracker.files_changed_since_verify.is_empty());
        assert!(tracker.error_modules.is_empty());
    }

    #[test]
    fn architectural_task_detection() {
        assert!(is_architectural_task("refactor the authentication module"));
        assert!(is_architectural_task("Migrate from REST to GraphQL API"));
        assert!(is_architectural_task("RESTRUCTURE the codebase"));
        assert!(!is_architectural_task("fix the typo in readme"));
        assert!(!is_architectural_task("add a new test"));
    }

    #[test]
    fn duplicate_file_changes_not_counted() {
        let mut tracker = FailureTracker::default();
        tracker.record_file_change("a.rs");
        tracker.record_file_change("a.rs");
        tracker.record_file_change("a.rs");
        assert_eq!(tracker.files_changed_since_verify.len(), 1);
    }
}
