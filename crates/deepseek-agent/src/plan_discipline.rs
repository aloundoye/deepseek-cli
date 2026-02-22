//! Plan → Execute → Verify discipline layer.
//!
//! A thin wrapper around the chat-with-tools loop that:
//! - Triggers plan generation for complex prompts
//! - Tracks plan state and step progress
//! - Injects step context into LLM API calls
//! - Gates exit with verification checkpoints
//!
//! This does NOT restructure the existing loop — it adds a few variables
//! checked at strategic points.

use deepseek_core::{ChatMessage, Plan};
use std::collections::HashSet;
use std::path::Path;

/// Lifecycle status of a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlanStatus {
    /// No plan has been generated for this task.
    #[default]
    NotPlanned,
    /// Plan exists but execution has not started.
    Planned,
    /// Actively executing plan steps.
    Executing,
    /// All steps done, running verification.
    Verifying,
    /// Verification passed, task complete.
    Completed,
}

/// State tracker for the plan discipline loop.
#[derive(Debug, Clone, Default)]
pub struct PlanState {
    /// Current lifecycle status.
    pub status: PlanStatus,
    /// The active plan (if any).
    pub plan: Option<Plan>,
    /// Index of the current step (0-based).
    pub current_step: usize,
    /// Files touched by successful write/edit tools.
    pub files_touched: HashSet<String>,
    /// Verification commands derived from repo signals.
    pub verify_commands: Vec<String>,
    /// Number of consecutive verification failures.
    pub verify_failure_streak: u32,
    /// Fingerprint of last verification error (normalized hash).
    pub last_verify_fingerprint: Option<u64>,
}

impl PlanState {
    /// Create a new plan state with the given plan.
    pub fn with_plan(plan: Plan, verify_commands: Vec<String>) -> Self {
        Self {
            status: PlanStatus::Planned,
            plan: Some(plan),
            current_step: 0,
            files_touched: HashSet::new(),
            verify_commands,
            verify_failure_streak: 0,
            last_verify_fingerprint: None,
        }
    }

    /// Mark execution as started.
    pub fn start_execution(&mut self) {
        if self.status == PlanStatus::Planned {
            self.status = PlanStatus::Executing;
        }
    }

    /// Record a file being touched by a write/edit tool.
    pub fn record_file_touched(&mut self, path: &str) {
        self.files_touched.insert(path.to_string());
    }

    /// Check if the current step should be marked done based on files touched.
    ///
    /// A step advances only when a touched file is explicitly listed in the
    /// step's `files` list. Steps with no declared files are skipped
    /// automatically (they represent read-only or exploration steps).
    ///
    /// Returns true if a step was advanced.
    pub fn maybe_advance_step(&mut self) -> bool {
        let Some(ref mut plan) = self.plan else {
            return false;
        };

        let mut advanced = false;
        while self.current_step < plan.steps.len() {
            let step = &plan.steps[self.current_step];
            if step.done {
                self.current_step += 1;
                continue;
            }

            if step.files.is_empty() {
                // Steps with no declared files auto-advance (read-only/exploration)
                plan.steps[self.current_step].done = true;
                self.current_step += 1;
                advanced = true;
                continue;
            }

            // Advance only when at least one declared file has been touched
            let has_match = step.files.iter().any(|f| self.files_touched.contains(f));
            if has_match {
                plan.steps[self.current_step].done = true;
                self.current_step += 1;
                advanced = true;
            } else {
                break;
            }
        }
        advanced
    }

    /// Check if all steps are done.
    pub fn all_steps_done(&self) -> bool {
        self.plan
            .as_ref()
            .is_some_and(|p| p.steps.iter().all(|s| s.done))
    }

    /// Transition to verifying status. Only transitions if all steps are done.
    pub fn enter_verification(&mut self) {
        if self.status == PlanStatus::Executing && self.all_steps_done() {
            self.status = PlanStatus::Verifying;
        }
    }

    /// Check if verification should be treated as a no-op (no commands configured).
    pub fn verification_is_noop(&self) -> bool {
        self.verify_commands.is_empty()
    }

    /// Mark verification as passed.
    pub fn verification_passed(&mut self) {
        self.status = PlanStatus::Completed;
        self.verify_failure_streak = 0;
        self.last_verify_fingerprint = None;
    }

    /// Record a verification failure. Increments streak only if the error
    /// fingerprint is the same as last time (avoids counting different errors
    /// as "repeated").
    pub fn verification_failed(&mut self, error: &str) {
        let fp = fingerprint_error(error);
        if self.last_verify_fingerprint == Some(fp) {
            self.verify_failure_streak += 1;
        } else {
            // New/different error — reset streak
            self.verify_failure_streak = 1;
            self.last_verify_fingerprint = Some(fp);
        }
        // Return to executing to allow fixes
        self.status = PlanStatus::Executing;
    }

    /// Generate the current step context string for injection.
    pub fn current_step_context(&self) -> Option<String> {
        let plan = self.plan.as_ref()?;
        if self.current_step >= plan.steps.len() {
            return None;
        }
        let step = &plan.steps[self.current_step];
        let files_hint = if step.files.is_empty() {
            String::new()
        } else {
            format!(" Files: {}", step.files.join(", "))
        };
        Some(format!(
            "You are on step {} of {}: {}. Intent: {}.{}",
            self.current_step + 1,
            plan.steps.len(),
            step.title,
            step.intent,
            files_hint
        ))
    }

    /// Generate a compact summary for compaction survival.
    pub fn to_compact_summary(&self) -> String {
        let Some(ref plan) = self.plan else {
            return String::new();
        };
        let done_count = plan.steps.iter().filter(|s| s.done).count();
        let total = plan.steps.len();
        let status_str = match self.status {
            PlanStatus::NotPlanned => "not_planned",
            PlanStatus::Planned => "planned",
            PlanStatus::Executing => "executing",
            PlanStatus::Verifying => "verifying",
            PlanStatus::Completed => "completed",
        };
        let step_titles: Vec<_> = plan
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let marker = if s.done {
                    "[x]"
                } else if i == self.current_step {
                    "[>]"
                } else {
                    "[ ]"
                };
                format!("{marker} {}", s.title)
            })
            .collect();
        format!(
            "[Plan State]\nGoal: {}\nStatus: {status_str} ({done_count}/{total} steps done)\n\
             Steps:\n{}\nVerify: {}",
            plan.goal,
            step_titles.join("\n"),
            if self.verify_commands.is_empty() {
                "(none)".to_string()
            } else {
                self.verify_commands.join(" && ")
            }
        )
    }
}

// ── Error fingerprinting ──

/// Normalize and hash verification error output for repeat detection.
/// Strips ANSI escape codes, trims timestamps/temp paths, keeps last 2000 chars.
fn fingerprint_error(error: &str) -> u64 {
    use std::hash::{Hash, Hasher};

    let normalized = strip_ansi_codes(error);
    // Keep only last 2000 chars for comparison (errors diverge at the start)
    let tail = if normalized.len() > 2000 {
        &normalized[normalized.len() - 2000..]
    } else {
        &normalized
    };
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tail.hash(&mut hasher);
    hasher.finish()
}

/// Strip ANSI escape codes from a string.
fn strip_ansi_codes(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            // Skip ESC [ ... (letter)
            if chars.peek() == Some(&'[') {
                chars.next();
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else {
            result.push(ch);
        }
    }
    result
}

// ── Planning triggers ──

/// Signals that indicate planning should be triggered.
#[derive(Debug, Clone, Default)]
pub struct PlanningTriggers {
    /// Prompt contains keywords suggesting multi-step work.
    pub has_planning_keywords: bool,
    /// Estimated tool calls exceeds threshold.
    pub predicted_tool_calls_high: bool,
    /// Large codebase (>500 files).
    pub large_repo: bool,
    /// Previous failures in this session.
    pub has_previous_failures: bool,
    /// Prompt is flagged as complex by detect_signals().
    pub prompt_is_complex: bool,
}

impl PlanningTriggers {
    /// Check if planning should be activated.
    /// True if keywords are present OR 2+ signals fire.
    pub fn should_plan(&self) -> bool {
        let signal_count = [
            self.has_planning_keywords,
            self.predicted_tool_calls_high,
            self.large_repo,
            self.has_previous_failures,
            self.prompt_is_complex,
        ]
        .iter()
        .filter(|&&x| x)
        .count();
        self.has_planning_keywords || signal_count >= 2
    }
}

/// Detect planning triggers from prompt and context.
pub fn detect_planning_triggers(
    prompt: &str,
    codebase_file_count: usize,
    failure_streak: u32,
    prompt_is_complex: bool,
) -> PlanningTriggers {
    let prompt_lower = prompt.to_ascii_lowercase();
    let keywords = [
        "refactor",
        "implement",
        "migrate",
        "restructure",
        "redesign",
        "build a",
        "create a new",
        "add feature",
        "rewrite",
        "convert",
        "extract",
        "split",
        "merge",
    ];
    let has_planning_keywords = keywords.iter().any(|kw| prompt_lower.contains(kw));

    // Heuristic: long prompts with multiple file references suggest many tool calls
    let predicted_tool_calls_high = prompt.len() > 500
        || prompt_lower.matches("file").count() >= 3
        || prompt_lower.matches(".rs").count() >= 2
        || prompt_lower.matches(".ts").count() >= 2;

    PlanningTriggers {
        has_planning_keywords,
        predicted_tool_calls_high,
        large_repo: codebase_file_count > 500,
        has_previous_failures: failure_streak > 0,
        prompt_is_complex,
    }
}

// ── Verification command derivation ──

/// Derive verification commands from workspace signals.
///
/// Conservative by default: Rust projects get `cargo test` only.
/// Linting (`cargo clippy`) is intentionally omitted as a default since
/// it's slow and not always appropriate for every task.
pub fn derive_verify_commands(workspace: &Path) -> Vec<String> {
    let mut commands = Vec::new();

    // Rust/Cargo — just test by default
    if workspace.join("Cargo.toml").exists() {
        commands.push("cargo test --workspace --all-targets".to_string());
    }

    // Node/npm
    if workspace.join("package.json").exists() {
        if workspace.join("pnpm-lock.yaml").exists() {
            commands.push("pnpm test".to_string());
        } else if workspace.join("yarn.lock").exists() {
            commands.push("yarn test".to_string());
        } else {
            commands.push("npm test".to_string());
        }
    }

    // Python
    if workspace.join("pyproject.toml").exists() || workspace.join("setup.py").exists() {
        commands.push("pytest".to_string());
    }

    // Go
    if workspace.join("go.mod").exists() {
        commands.push("go test ./...".to_string());
    }

    // Makefile fallback
    if commands.is_empty() && workspace.join("Makefile").exists() {
        commands.push("make test".to_string());
    }

    // If no commands detected, verification is a no-op (empty vec).
    // The caller should treat empty verify_commands as automatic pass.
    commands
}

// ── Context injection helpers ──

/// Inject plan step context into messages before LLM API call.
/// Returns the context string that was injected (for logging).
pub fn inject_step_context(
    messages: &mut Vec<ChatMessage>,
    plan_state: &PlanState,
) -> Option<String> {
    if plan_state.status != PlanStatus::Executing {
        return None;
    }
    let context = plan_state.current_step_context()?;
    messages.push(ChatMessage::User {
        content: format!("<plan-context>{context}</plan-context>"),
    });
    Some(context)
}

/// Remove injected plan context from messages (for history cleanliness).
pub fn remove_step_context(messages: &mut Vec<ChatMessage>) {
    messages.retain(|msg| {
        if let ChatMessage::User { content } = msg {
            !content.starts_with("<plan-context>")
        } else {
            true
        }
    });
}

/// Inject verification failure feedback into messages.
pub fn inject_verification_feedback(
    messages: &mut Vec<ChatMessage>,
    error: &str,
    failure_count: u32,
) {
    let feedback = format!(
        "<verification-failure count=\"{failure_count}\">\n\
         Verification failed. Fix the following error and retry.\n\
         Error:\n{error}\n\
         </verification-failure>"
    );
    messages.push(ChatMessage::User { content: feedback });
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepseek_core::PlanStep;
    use uuid::Uuid;

    fn mock_plan() -> Plan {
        Plan {
            plan_id: Uuid::now_v7(),
            version: 1,
            goal: "Add feature X".to_string(),
            assumptions: vec!["Workspace writable".to_string()],
            steps: vec![
                PlanStep {
                    step_id: Uuid::now_v7(),
                    title: "Search existing code".to_string(),
                    intent: "search".to_string(),
                    tools: vec!["fs.grep".to_string()],
                    files: vec![], // no files = auto-advance
                    done: false,
                },
                PlanStep {
                    step_id: Uuid::now_v7(),
                    title: "Implement feature".to_string(),
                    intent: "edit".to_string(),
                    tools: vec!["fs.edit".to_string()],
                    files: vec!["src/lib.rs".to_string()],
                    done: false,
                },
            ],
            verification: vec!["cargo test".to_string()],
            risk_notes: vec![],
        }
    }

    #[test]
    fn plan_state_lifecycle() {
        let plan = mock_plan();
        let mut state = PlanState::with_plan(plan, vec!["cargo test".to_string()]);
        assert_eq!(state.status, PlanStatus::Planned);

        state.start_execution();
        assert_eq!(state.status, PlanStatus::Executing);
        assert_eq!(state.current_step, 0);

        // Step 0 has no declared files — auto-advances
        assert!(state.maybe_advance_step());
        assert_eq!(state.current_step, 1);

        // Step 1 requires src/lib.rs
        state.record_file_touched("other.rs");
        assert!(!state.maybe_advance_step()); // wrong file

        state.record_file_touched("src/lib.rs");
        assert!(state.maybe_advance_step());
        assert!(state.all_steps_done());

        state.enter_verification();
        assert_eq!(state.status, PlanStatus::Verifying);

        state.verification_passed();
        assert_eq!(state.status, PlanStatus::Completed);
    }

    #[test]
    fn step_advancement_requires_matching_file() {
        let plan = Plan {
            plan_id: Uuid::now_v7(),
            version: 1,
            goal: "test".to_string(),
            assumptions: vec![],
            steps: vec![PlanStep {
                step_id: Uuid::now_v7(),
                title: "Edit main".to_string(),
                intent: "edit".to_string(),
                tools: vec!["fs.edit".to_string()],
                files: vec!["src/main.rs".to_string()],
                done: false,
            }],
            verification: vec![],
            risk_notes: vec![],
        };
        let mut state = PlanState::with_plan(plan, vec![]);
        state.start_execution();

        // Touching unrelated file should NOT advance
        state.record_file_touched("README.md");
        assert!(!state.maybe_advance_step());
        assert_eq!(state.current_step, 0);

        // Touching the right file should advance
        state.record_file_touched("src/main.rs");
        assert!(state.maybe_advance_step());
        assert_eq!(state.current_step, 1);
        assert!(state.all_steps_done());
    }

    #[test]
    fn planning_triggers_keywords() {
        let triggers = detect_planning_triggers("refactor the auth module", 100, 0, false);
        assert!(triggers.has_planning_keywords);
        assert!(triggers.should_plan());

        let triggers2 = detect_planning_triggers("fix typo in readme", 100, 0, false);
        assert!(!triggers2.has_planning_keywords);
        assert!(!triggers2.should_plan());
    }

    #[test]
    fn planning_triggers_multiple_signals() {
        // Large repo + failure = 2 signals → should plan
        let triggers = detect_planning_triggers("update config", 1000, 2, false);
        assert!(triggers.should_plan());

        // Small repo, no failures, simple prompt → should not plan
        let triggers2 = detect_planning_triggers("update config", 50, 0, false);
        assert!(!triggers2.should_plan());
    }

    #[test]
    fn derive_verify_commands_rust() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("Cargo.toml"), "").unwrap();
        let commands = derive_verify_commands(temp.path());
        assert!(commands.iter().any(|c| c.contains("cargo test")));
        // clippy should NOT be included by default (conservative)
        assert!(!commands.iter().any(|c| c.contains("clippy")));
    }

    #[test]
    fn derive_verify_commands_empty_workspace() {
        let temp = tempfile::tempdir().unwrap();
        let commands = derive_verify_commands(temp.path());
        assert!(commands.is_empty());
    }

    #[test]
    fn empty_verify_commands_is_noop() {
        let plan = mock_plan();
        let state = PlanState::with_plan(plan, vec![]);
        assert!(state.verification_is_noop());

        let state2 = PlanState::with_plan(mock_plan(), vec!["cargo test".to_string()]);
        assert!(!state2.verification_is_noop());
    }

    #[test]
    fn compact_summary_format() {
        let plan = mock_plan();
        let state = PlanState::with_plan(plan, vec!["cargo test".to_string()]);
        let summary = state.to_compact_summary();
        assert!(summary.contains("Add feature X"));
        assert!(summary.contains("planned"));
        assert!(summary.contains("[>] Search existing code"));
        assert!(summary.contains("[ ] Implement feature"));
    }

    #[test]
    fn verification_failure_tracking_same_error() {
        let plan = mock_plan();
        let mut state = PlanState::with_plan(plan, vec!["cargo test".to_string()]);
        state.start_execution();

        // Mark all steps done to allow verification
        state.plan.as_mut().unwrap().steps[0].done = true;
        state.plan.as_mut().unwrap().steps[1].done = true;
        state.current_step = 2;

        state.enter_verification();
        assert_eq!(state.status, PlanStatus::Verifying);

        // Same error twice → streak increments
        state.verification_failed("error: test failed");
        assert_eq!(state.verify_failure_streak, 1);
        assert_eq!(state.status, PlanStatus::Executing);

        state.enter_verification();
        state.verification_failed("error: test failed"); // same error
        assert_eq!(state.verify_failure_streak, 2);
    }

    #[test]
    fn verification_failure_tracking_different_error() {
        let plan = mock_plan();
        let mut state = PlanState::with_plan(plan, vec!["cargo test".to_string()]);
        state.start_execution();
        state.plan.as_mut().unwrap().steps[0].done = true;
        state.plan.as_mut().unwrap().steps[1].done = true;
        state.current_step = 2;

        state.enter_verification();
        state.verification_failed("error: test A failed");
        assert_eq!(state.verify_failure_streak, 1);

        state.enter_verification();
        state.verification_failed("error: completely different B failed"); // different error
        assert_eq!(state.verify_failure_streak, 1); // reset to 1, not incremented
    }

    #[test]
    fn verification_passed_resets_streak() {
        let plan = mock_plan();
        let mut state = PlanState::with_plan(plan, vec!["cargo test".to_string()]);
        state.start_execution();
        state.plan.as_mut().unwrap().steps[0].done = true;
        state.plan.as_mut().unwrap().steps[1].done = true;
        state.current_step = 2;

        state.enter_verification();
        state.verification_failed("error");
        state.enter_verification();
        state.verification_passed();
        assert_eq!(state.verify_failure_streak, 0);
        assert_eq!(state.status, PlanStatus::Completed);
    }

    #[test]
    fn enter_verification_requires_all_steps_done() {
        let plan = mock_plan();
        let mut state = PlanState::with_plan(plan, vec![]);
        state.start_execution();

        // Not all steps done — should stay Executing
        state.enter_verification();
        assert_eq!(state.status, PlanStatus::Executing);
    }

    #[test]
    fn inject_remove_context_roundtrip() {
        let plan = mock_plan();
        let mut state = PlanState::with_plan(plan, vec![]);
        state.start_execution();

        let mut messages = vec![
            ChatMessage::System {
                content: "system".to_string(),
            },
            ChatMessage::User {
                content: "do something".to_string(),
            },
        ];

        let ctx = inject_step_context(&mut messages, &state);
        assert!(ctx.is_some());
        assert_eq!(messages.len(), 3);
        assert!(
            matches!(&messages[2], ChatMessage::User { content } if content.starts_with("<plan-context>"))
        );

        remove_step_context(&mut messages);
        assert_eq!(messages.len(), 2); // plan context removed
    }

    #[test]
    fn no_context_injection_when_not_executing() {
        let state = PlanState::default();
        let mut messages = vec![ChatMessage::User {
            content: "hi".to_string(),
        }];
        let ctx = inject_step_context(&mut messages, &state);
        assert!(ctx.is_none());
        assert_eq!(messages.len(), 1); // nothing added
    }

    #[test]
    fn current_step_context_format() {
        let plan = mock_plan();
        let mut state = PlanState::with_plan(plan, vec![]);
        // Auto-advance past step 0 (no files)
        state.start_execution();
        state.maybe_advance_step();
        // Now on step 1
        let ctx = state.current_step_context().unwrap();
        assert!(ctx.contains("step 2 of 2"));
        assert!(ctx.contains("Implement feature"));
    }

    #[test]
    fn fingerprint_strips_ansi() {
        let with_ansi = "\x1b[31merror\x1b[0m: test failed";
        let without_ansi = "error: test failed";
        assert_eq!(
            fingerprint_error(with_ansi),
            fingerprint_error(without_ansi)
        );
    }

    #[test]
    fn fingerprint_different_errors() {
        assert_ne!(
            fingerprint_error("error: cannot find module X"),
            fingerprint_error("error: type mismatch in Y")
        );
    }
}
