//! Mode router: selects the execution mode for each agent turn.
//!
//! Two modes:
//! - **V3Autopilot**: `deepseek-chat` with thinking + tools (fast, default).
//! - **R1DriveTools**: R1 emits tool-intent JSON, orchestrator executes, R1 iterates.
//!
//! Escalation triggers (checked by `decide_mode()` after each tool execution batch):
//! 1. **Doom-loop** (highest priority): same `ToolSignature` fails ≥ threshold (default 2).
//! 2. **Repeated step failures**: consecutive failures ≥ threshold (default 5).
//! 3. **Ambiguous errors**: `ErrorClass::Ambiguous`.
//! 4. **Blast radius**: ≥ threshold (default 10) files changed without verification.
//! 5. **Cross-module failures**: errors span 2+ modules.
//!
//! Special case: policy doom-loop breaker (in agent `lib.rs`, not here). When all
//! doom-loop signatures are `bash.run` policy errors, the agent injects tool guidance
//! and resets trackers instead of escalating — R1 faces the same policy restrictions.
//!
//! V3 also has lightweight R1 consultation (`consultation.rs`) as an alternative to
//! full escalation — R1 returns text-only advice while V3 keeps control and tools.
//!
//! Hysteresis: once in R1, stay until verify green, R1 `done`/`abort`, or budget exhausted.

use crate::observation::{ErrorClass, ObservationPack};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The active execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentMode {
    /// V3 with thinking + tools in a single call. Fast default.
    V3Autopilot,
    /// R1 drives tools step-by-step via JSON intents. Orchestrator executes.
    R1DriveTools,
}

impl std::fmt::Display for AgentMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::V3Autopilot => write!(f, "v3_autopilot"),
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
    /// Doom-loop threshold: if the same tool signature fails this many times, escalate.
    pub doom_loop_threshold: u32,
}

impl Default for ModeRouterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            // Raised from 2 → 5: V3 now has R1 consultation via think_deeply,
            // so it gets more attempts before full R1DriveTools escalation.
            v3_max_step_failures: 5,
            // Raised from 5 → 10: higher threshold for blast radius escalation.
            blast_radius_threshold: 10,
            v3_mechanical_recovery: true,
            r1_max_steps: 30,
            r1_max_parse_retries: 2,
            v3_patch_max_context_requests: 3,
            doom_loop_threshold: 2,
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
            doom_loop_threshold: 2,
        }
    }
}

/// Compact signature for doom-loop detection.
///
/// Tracks (tool_name, normalized_arg_key, exit_code_bucket) to detect
/// repeated identical failures — the hallmark of a doom-loop.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToolSignature {
    pub tool: String,
    pub arg_key: String,
    pub exit_bucket: i32,
}

impl ToolSignature {
    /// Build a signature from a tool call result.
    pub fn new(tool: &str, args: &serde_json::Value, exit_code: Option<i32>) -> Self {
        // Normalize args to a stable key: for file tools use file_path,
        // for bash use the command, for grep use pattern, otherwise hash.
        let arg_key = if let Some(fp) = args.get("file_path").and_then(|v| v.as_str()) {
            fp.to_string()
        } else if let Some(cmd) = args.get("cmd").and_then(|v| v.as_str()) {
            // Normalize whitespace for command comparison
            cmd.split_whitespace().collect::<Vec<_>>().join(" ")
        } else if let Some(pat) = args.get("pattern").and_then(|v| v.as_str()) {
            pat.to_string()
        } else {
            // Stable hash of full args
            let mut s = args.to_string();
            s.truncate(200);
            s
        };
        // Bucket exit codes: 0=success, 1=general error, 2+=other
        let exit_bucket = match exit_code {
            Some(0) => 0,
            Some(1) => 1,
            Some(c) if c > 1 => 2,
            Some(c) => c, // negative signals
            None => -1,
        };
        Self {
            tool: tool.to_string(),
            arg_key,
            exit_bucket,
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
    /// Doom-loop tracker: maps tool signatures to failure count.
    pub doom_loop_sigs: HashMap<ToolSignature, u32>,
    /// Whether verification has passed since last R1 escalation.
    /// Used for hysteresis: R1 stays active until this is true.
    pub verify_passed_since_r1: bool,
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

    /// Record a tool call result for doom-loop detection.
    /// Returns true if a doom-loop is detected (same signature failed >= threshold).
    pub fn record_tool_signature(&mut self, sig: ToolSignature, success: bool) -> bool {
        if success {
            // Remove from doom-loop tracking on success
            self.doom_loop_sigs.remove(&sig);
            false
        } else {
            let count = self.doom_loop_sigs.entry(sig).or_insert(0);
            *count += 1;
            // The caller checks against threshold
            false // just records; decide_mode checks
        }
    }

    /// Check if any tool signature has hit the doom-loop threshold.
    pub fn has_doom_loop(&self, threshold: u32) -> bool {
        self.doom_loop_sigs
            .values()
            .any(|&count| count >= threshold)
    }

    /// Record a successful verification (resets blast radius tracking).
    pub fn record_verify_pass(&mut self) {
        self.files_changed_since_verify.clear();
        self.error_modules.clear();
        self.consecutive_step_failures = 0;
        self.v3_recovery_used = false;
        self.doom_loop_sigs.clear();
        self.verify_passed_since_r1 = true;
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

    /// Mark that we entered R1 mode (reset hysteresis flag).
    pub fn entered_r1(&mut self) {
        self.verify_passed_since_r1 = false;
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
    /// Same tool+args+exit repeated N times — doom-loop detected.
    DoomLoop,
    /// R1 completed or verification green — handoff back to V3.
    R1Completed,
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
            Self::DoomLoop => write!(f, "doom_loop"),
            Self::R1Completed => write!(f, "r1_completed"),
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
/// Escalation policy (V3 → R1):
/// - Doom-loop: same tool signature fails N times
/// - Same step fails >= v3_max_step_failures times
/// - Error is ambiguous
/// - Blast radius exceeds threshold
/// - Errors span multiple modules
///
/// Hysteresis: once in R1, stay until:
/// - Budget exhausted → fall back to V3
/// - R1 returns done/abort (handled by caller)
/// - Verification passes (verify_passed_since_r1)
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

    // If already in R1DriveTools, apply hysteresis
    if current_mode == AgentMode::R1DriveTools {
        // Budget exhausted → forced return to V3
        if tracker.r1_steps_used >= config.r1_max_steps {
            return ModeDecision {
                mode: AgentMode::V3Autopilot,
                reason: Some(EscalationReason::R1BudgetExhausted),
            };
        }
        // Verification passed → R1 can hand back to V3
        if tracker.verify_passed_since_r1 {
            return ModeDecision {
                mode: AgentMode::V3Autopilot,
                reason: Some(EscalationReason::R1Completed),
            };
        }
        // Stay in R1 (hysteresis: don't ping-pong back to V3)
        return ModeDecision {
            mode: AgentMode::R1DriveTools,
            reason: None,
        };
    }

    // Check escalation triggers from V3Autopilot

    // 0. Doom-loop detection (highest priority)
    if tracker.has_doom_loop(config.doom_loop_threshold) {
        return ModeDecision {
            mode: AgentMode::R1DriveTools,
            reason: Some(EscalationReason::DoomLoop),
        };
    }

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
    if let Some(obs) = observation
        && obs.error_class == ErrorClass::Ambiguous
    {
        return ModeDecision {
            mode: AgentMode::R1DriveTools,
            reason: Some(EscalationReason::AmbiguousError),
        };
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
        let tracker = FailureTracker {
            consecutive_step_failures: 100,
            ..FailureTracker::default()
        };
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
        let tracker = FailureTracker {
            r1_steps_used: 5,
            ..FailureTracker::default()
        };
        let decision = decide_mode(&config, AgentMode::R1DriveTools, &tracker, None);
        assert_eq!(decision.mode, AgentMode::V3Autopilot);
        assert_eq!(decision.reason, Some(EscalationReason::R1BudgetExhausted));
    }

    #[test]
    fn stays_in_r1_drive_within_budget() {
        let config = default_config();
        let tracker = FailureTracker {
            r1_steps_used: 3,
            ..FailureTracker::default()
        };
        let decision = decide_mode(&config, AgentMode::R1DriveTools, &tracker, None);
        assert_eq!(decision.mode, AgentMode::R1DriveTools);
    }

    #[test]
    fn verify_pass_resets_tracker() {
        let mut tracker = FailureTracker {
            consecutive_step_failures: 5,
            v3_recovery_used: true,
            ..FailureTracker::default()
        };
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

    #[test]
    fn doom_loop_triggers_escalation() {
        let config = ModeRouterConfig {
            doom_loop_threshold: 2,
            ..default_config()
        };
        let mut tracker = FailureTracker::default();
        let sig = ToolSignature::new(
            "bash.run",
            &serde_json::json!({"cmd": "cargo test"}),
            Some(1),
        );
        tracker.record_tool_signature(sig.clone(), false);
        // First failure — not yet a doom-loop
        let d1 = decide_mode(&config, AgentMode::V3Autopilot, &tracker, None);
        assert_eq!(d1.mode, AgentMode::V3Autopilot);

        // Second failure with same signature — doom-loop!
        tracker.record_tool_signature(sig, false);
        let d2 = decide_mode(&config, AgentMode::V3Autopilot, &tracker, None);
        assert_eq!(d2.mode, AgentMode::R1DriveTools);
        assert_eq!(d2.reason, Some(EscalationReason::DoomLoop));
    }

    #[test]
    fn doom_loop_cleared_on_success() {
        let mut tracker = FailureTracker::default();
        let sig = ToolSignature::new(
            "bash.run",
            &serde_json::json!({"cmd": "cargo test"}),
            Some(1),
        );
        tracker.record_tool_signature(sig.clone(), false);
        assert_eq!(tracker.doom_loop_sigs.len(), 1);
        // Success clears doom-loop tracking for that signature
        tracker.record_tool_signature(sig, true);
        assert!(tracker.doom_loop_sigs.is_empty());
    }

    #[test]
    fn doom_loop_cleared_on_verify_pass() {
        let mut tracker = FailureTracker::default();
        let sig = ToolSignature::new(
            "bash.run",
            &serde_json::json!({"cmd": "cargo test"}),
            Some(1),
        );
        tracker.record_tool_signature(sig, false);
        assert!(!tracker.doom_loop_sigs.is_empty());
        tracker.record_verify_pass();
        assert!(tracker.doom_loop_sigs.is_empty());
    }

    #[test]
    fn different_signatures_tracked_independently() {
        let mut tracker = FailureTracker::default();
        let sig1 = ToolSignature::new(
            "bash.run",
            &serde_json::json!({"cmd": "cargo test"}),
            Some(1),
        );
        let sig2 = ToolSignature::new(
            "bash.run",
            &serde_json::json!({"cmd": "cargo check"}),
            Some(1),
        );
        tracker.record_tool_signature(sig1, false);
        tracker.record_tool_signature(sig2, false);
        assert!(!tracker.has_doom_loop(2));
        assert_eq!(tracker.doom_loop_sigs.len(), 2);
    }

    #[test]
    fn hysteresis_stays_in_r1_until_verify() {
        let config = default_config();
        let mut tracker = FailureTracker::default();
        tracker.entered_r1();
        tracker.r1_steps_used = 3;

        // In R1 mode, no verify yet — should stay
        let d1 = decide_mode(&config, AgentMode::R1DriveTools, &tracker, None);
        assert_eq!(d1.mode, AgentMode::R1DriveTools);
        assert!(d1.reason.is_none());

        // Verify passes — should allow return to V3
        tracker.record_verify_pass();
        let d2 = decide_mode(&config, AgentMode::R1DriveTools, &tracker, None);
        assert_eq!(d2.mode, AgentMode::V3Autopilot);
        assert_eq!(d2.reason, Some(EscalationReason::R1Completed));
    }

    #[test]
    fn hysteresis_r1_budget_overrides() {
        let config = ModeRouterConfig {
            r1_max_steps: 5,
            ..default_config()
        };
        let mut tracker = FailureTracker::default();
        tracker.entered_r1();
        tracker.r1_steps_used = 5;
        // Budget exhausted, even though verify hasn't passed
        let d = decide_mode(&config, AgentMode::R1DriveTools, &tracker, None);
        assert_eq!(d.mode, AgentMode::V3Autopilot);
        assert_eq!(d.reason, Some(EscalationReason::R1BudgetExhausted));
    }

    #[test]
    fn tool_signature_normalizes_commands() {
        let sig1 = ToolSignature::new(
            "bash.run",
            &serde_json::json!({"cmd": "cargo  test  --lib"}),
            Some(1),
        );
        let sig2 = ToolSignature::new(
            "bash.run",
            &serde_json::json!({"cmd": "cargo test --lib"}),
            Some(1),
        );
        assert_eq!(sig1, sig2, "whitespace normalization should match");
    }
}
