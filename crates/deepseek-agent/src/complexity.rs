/// Thinking budget control for DeepSeek.
///
/// DeepSeek expands to fill the budget you give it — unlike Claude which
/// self-regulates. So we use two budgets and switch based on evidence:
///
/// - `DEFAULT_THINK_BUDGET`: Moderate baseline for most tasks.
/// - `HARD_THINK_BUDGET`: Opened only when tool outputs prove the model needs it
///   (compile failures, test failures, repeated errors, multi-file refactors).
///
/// The tool loop escalates dynamically based on *observed* failure states,
/// not heuristic prompt classification.
/// Prompt complexity — used for planning injection, not budget selection.
///
/// Budget is controlled by `DEFAULT_THINK_BUDGET` vs `HARD_THINK_BUDGET`,
/// switched by evidence. Complexity only determines whether the system prompt
/// includes the full planning protocol or lightweight guidance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptComplexity {
    /// Quick fixes, renames, trivial single-location changes.
    Simple,
    /// Default. Single-feature work, debugging, moderate exploration.
    Medium,
    /// Refactors, migrations, multi-file coordination, architectural work.
    Complex,
}

// ── Thinking Budgets ──────────────────────────────────────────────────────────
//
// Two budgets, not a spectrum. The default is moderate so DeepSeek doesn't
// waste tokens on easy tasks. The hard budget opens when evidence demands it.

/// Moderate baseline. Enough for single-feature work, debugging, exploration.
/// DeepSeek will use most of this even for simple tasks, so don't over-allocate.
pub const DEFAULT_THINK_BUDGET: u32 = 8_192;

/// Hard budget. Opened on compile/test failures, multi-file refactors, or
/// repeated errors. Large enough for architectural reasoning.
pub const HARD_THINK_BUDGET: u32 = 32_768;

/// Absolute maximum (force_max_think). For explicit user override.
pub const MAX_THINK_BUDGET: u32 = 65_536;

/// Per-failure escalation step within the hard budget range.
pub const ESCALATION_STEP: u32 = 8_192;

/// Medium budget — for non-trivial single-feature work.
pub const MEDIUM_THINK_BUDGET: u32 = 16_384;

// Keep old names as aliases so callers don't break.
pub const SIMPLE_THINK_BUDGET: u32 = DEFAULT_THINK_BUDGET;
pub const COMPLEX_THINK_BUDGET: u32 = HARD_THINK_BUDGET;
pub const ESCALATION_BUDGET_DELTA: u32 = ESCALATION_STEP;

// ── Evidence-based escalation ────────────────────────────────────────────────

/// Signals from tool outputs that indicate the model needs more thinking budget.
///
/// These are checked by the tool loop after each turn. When any are present,
/// the budget escalates from DEFAULT to HARD (and further within HARD on
/// repeated failures).
#[derive(Debug, Clone, Default)]
pub struct EscalationSignals {
    /// Tool returned a compilation error (cargo, tsc, gcc, etc.)
    pub compile_error: bool,
    /// Tool returned test failures
    pub test_failure: bool,
    /// fs_edit patch was rejected (conflict, wrong context)
    pub patch_rejected: bool,
    /// Model searched for something and got zero results
    pub search_miss: bool,
    /// Number of consecutive turns where all tool calls failed
    pub consecutive_failure_turns: usize,
    /// Number of consecutive turns where tool calls succeeded (for de-escalation)
    pub consecutive_success_turns: usize,
}

impl EscalationSignals {
    /// Should we escalate from DEFAULT to HARD budget?
    pub fn should_escalate(&self) -> bool {
        self.compile_error
            || self.test_failure
            || self.patch_rejected
            || self.search_miss
            || self.consecutive_failure_turns >= 2
    }

    /// Compute the thinking budget based on observed evidence.
    pub fn budget(&self) -> u32 {
        if !self.should_escalate() {
            return DEFAULT_THINK_BUDGET;
        }

        // Within HARD, further escalate based on repeated failures
        let base = HARD_THINK_BUDGET;
        if self.consecutive_failure_turns > 2 {
            let extra = ESCALATION_STEP * ((self.consecutive_failure_turns - 2) as u32);
            (base + extra).min(MAX_THINK_BUDGET)
        } else {
            base
        }
    }

    /// Record a successful tool turn (resets consecutive failures).
    /// After 3 consecutive successes, de-escalates sticky flags — the issue was resolved.
    pub fn record_success(&mut self) {
        self.consecutive_failure_turns = 0;
        self.consecutive_success_turns += 1;
        if self.consecutive_success_turns >= 3 {
            // Issue was resolved — de-escalate
            self.compile_error = false;
            self.test_failure = false;
            self.patch_rejected = false;
            self.search_miss = false;
        }
    }

    /// Record a failed tool turn.
    pub fn record_failure(&mut self) {
        self.consecutive_failure_turns += 1;
        self.consecutive_success_turns = 0;
    }

    /// Detect escalation signals from tool output text.
    pub fn scan_tool_output(&mut self, tool_name: &str, output: &str) {
        let lower = output.to_ascii_lowercase();

        // Compile errors — require stronger signal to avoid false positives
        // from tool error messages like "no such file" or "permission denied".
        if lower.contains("error[e")           // Rust: error[E0308]
            || (lower.contains("error:")
                && !lower.contains("no such file")
                && !lower.contains("permission denied")
                && (lower.contains("cannot find") || lower.contains("expected")
                    || lower.contains("mismatched") || lower.contains("undeclared")))
            || lower.contains("compilation failed")
            || lower.contains("build failed")
            || lower.contains("syntaxerror")
            || lower.contains("typeerror")
        {
            self.compile_error = true;
        }

        // Test failures
        if lower.contains("test failed")
            || lower.contains("failures:")
            || lower.contains("assertion failed")
            || lower.contains("failed. 0 passed")
            || (lower.contains("failed") && lower.contains("passed"))
        {
            self.test_failure = true;
        }

        // Patch rejection
        if (tool_name == "fs_edit" || tool_name == "fs.edit")
            && (lower.contains("no match")
                || lower.contains("conflict")
                || lower.contains("not found")
                || lower.contains("failed to apply"))
        {
            self.patch_rejected = true;
        }

        // Search miss
        if (tool_name == "fs_grep"
            || tool_name == "fs.grep"
            || tool_name == "fs_glob"
            || tool_name == "fs.glob")
            && (lower.contains("no matches")
                || lower.contains("0 results")
                || output.trim().is_empty())
        {
            self.search_miss = true;
        }
    }
}

// ── Complexity Classification ────────────────────────────────────────────────
//
// Used ONLY for planning injection (which system prompt tier to use).
// NOT used for budget selection — that's evidence-driven.

/// Classify a user prompt for planning injection.
///
/// Simple decision tree. Default is Medium; only clear signals deviate.
pub fn classify_complexity(prompt: &str) -> PromptComplexity {
    let lower = prompt.to_ascii_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    let word_count = words.len();

    if is_trivial(&lower, &words, word_count) {
        return PromptComplexity::Simple;
    }

    if is_complex(&lower, &words, word_count) {
        return PromptComplexity::Complex;
    }

    PromptComplexity::Medium
}

fn is_trivial(lower: &str, words: &[&str], word_count: usize) -> bool {
    if word_count > 15 {
        return false;
    }

    const TRIVIAL: &[&str] = &[
        "fix typo",
        "fix spelling",
        "fix indent",
        "fix whitespace",
        "fix formatting",
        "format code",
        "rename",
        "add import",
        "remove import",
        "update version",
        "bump version",
        "remove unused",
        "delete line",
        "remove line",
        "add newline",
        "comment out",
        "uncomment",
        "add comma",
        "remove comma",
        "add semicolon",
        "toggle",
        "swap",
        "flip",
    ];

    if TRIVIAL.iter().any(|t| lower.contains(t)) {
        return true;
    }

    if matches!(words.first(), Some(&"just" | &"only" | &"simply")) {
        // Check if the rest of the prompt contains complexity keywords.
        // "just refactor everything" is NOT simple, even with a minimizer prefix.
        let has_complex_keyword = [
            "refactor",
            "rewrite",
            "migrate",
            "redesign",
            "overhaul",
            "restructure",
            "implement",
            "add",
            "create",
            "build",
        ]
        .iter()
        .any(|k| lower.contains(k));
        return !has_complex_keyword;
    }
    false
}

fn is_complex(lower: &str, words: &[&str], word_count: usize) -> bool {
    let has_arch = [
        "refactor",
        "redesign",
        "rewrite",
        "rearchitect",
        "migrate",
        "restructure",
        "overhaul",
    ]
    .iter()
    .any(|k| lower.contains(k));

    let has_scope = [
        "across multiple",
        "multiple files",
        "all files",
        "entire codebase",
        "whole project",
        "every file",
        "each module",
        "end-to-end",
        "throughout",
    ]
    .iter()
    .any(|k| lower.contains(k));

    let conjunctions = lower.matches(" and ").count()
        + lower.matches(" then ").count()
        + lower.matches(" also ").count();

    let list_items = lower
        .lines()
        .filter(|l| {
            let t = l.trim().as_bytes();
            (t.len() > 2 && t[0].is_ascii_digit() && t[1] == b'.')
                || l.trim().starts_with("- ")
                || l.trim().starts_with("* ")
        })
        .count();

    let file_refs = words
        .iter()
        .filter(|w| {
            let w = w.trim_matches(|c: char| c == ',' || c == ';' || c == ')' || c == '(');
            (w.contains('/') && w.len() > 3)
                || w.ends_with(".rs")
                || w.ends_with(".ts")
                || w.ends_with(".tsx")
                || w.ends_with(".js")
                || w.ends_with(".py")
                || w.ends_with(".go")
                || w.ends_with(".java")
        })
        .count();

    if has_arch && word_count > 5 {
        return true;
    }
    if has_scope && word_count > 15 {
        return true;
    }
    if conjunctions >= 2 {
        return true;
    }
    if list_items >= 3 {
        return true;
    }
    if file_refs >= 3 {
        return true;
    }

    false
}

/// Classify with conversation history (only upgrades, never downgrades).
pub fn classify_with_history(
    prompt: &str,
    prior_tool_calls: usize,
    had_errors: bool,
    consecutive_failures: usize,
) -> PromptComplexity {
    let base = classify_complexity(prompt);

    if consecutive_failures >= 3 {
        return PromptComplexity::Complex;
    }

    if had_errors && prior_tool_calls > 15 {
        return match base {
            PromptComplexity::Simple => PromptComplexity::Medium,
            _ => PromptComplexity::Complex,
        };
    }

    if prior_tool_calls > 25 && base == PromptComplexity::Simple {
        return PromptComplexity::Medium;
    }

    base
}

/// Thinking budget for a complexity tier.
///
/// Simple → 8K, Medium → 16K, Complex → 32K.
/// Further escalation is handled by `EscalationSignals` in the tool loop.
pub fn thinking_budget_for(complexity: PromptComplexity) -> u32 {
    match complexity {
        PromptComplexity::Simple => DEFAULT_THINK_BUDGET, // 8K
        PromptComplexity::Medium => MEDIUM_THINK_BUDGET,  // 16K
        PromptComplexity::Complex => HARD_THINK_BUDGET,   // 32K
    }
}

/// Escalated budget after consecutive failures (legacy API — use EscalationSignals instead).
pub fn escalated_budget(base_budget: u32, consecutive_failures: usize) -> u32 {
    let delta = ESCALATION_STEP * (consecutive_failures as u32);
    (base_budget + delta).min(MAX_THINK_BUDGET)
}

/// Score a prompt for team orchestration threshold.
pub fn score_prompt(prompt: &str) -> u64 {
    let lower = prompt.to_ascii_lowercase();
    let mut score = 0_u64;

    for keyword in [
        "frontend",
        "backend",
        "api",
        "database",
        "migration",
        "test",
        "ci",
        "docs",
        "security",
        "performance",
    ] {
        if lower.contains(keyword) {
            score = score.saturating_add(10);
        }
    }

    for keyword in [
        "across",
        "multiple files",
        "end-to-end",
        "refactor",
        "orchestrate",
        "workflow",
        "integration",
        "pipeline",
    ] {
        if lower.contains(keyword) {
            score = score.saturating_add(8);
        }
    }

    let tokenish = lower.split_whitespace().count() as u64;
    score.saturating_add((tokenish / 16).min(20))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Two-budget system ──

    #[test]
    fn default_budget_is_moderate() {
        const {
            assert!(
                DEFAULT_THINK_BUDGET <= 16_384,
                "default should be moderate, not huge"
            );
        }
        const {
            assert!(
                DEFAULT_THINK_BUDGET >= 4_096,
                "default should not starve reasoning"
            );
        }
    }

    #[test]
    fn hard_budget_is_generous() {
        const { assert!(HARD_THINK_BUDGET >= 4 * DEFAULT_THINK_BUDGET) }
        const { assert!(MAX_THINK_BUDGET >= HARD_THINK_BUDGET) }
    }

    #[test]
    fn thinking_budget_matches_complexity() {
        // 3-tier: Simple 8K, Medium 16K, Complex 32K
        assert_eq!(
            thinking_budget_for(PromptComplexity::Simple),
            DEFAULT_THINK_BUDGET
        );
        assert_eq!(
            thinking_budget_for(PromptComplexity::Medium),
            MEDIUM_THINK_BUDGET
        );
        assert_eq!(
            thinking_budget_for(PromptComplexity::Complex),
            HARD_THINK_BUDGET
        );
        // Verify actual values
        assert_eq!(DEFAULT_THINK_BUDGET, 8_192);
        assert_eq!(MEDIUM_THINK_BUDGET, 16_384);
        assert_eq!(HARD_THINK_BUDGET, 32_768);
    }

    // ── Evidence-driven escalation ──

    #[test]
    fn no_evidence_uses_default_budget() {
        let signals = EscalationSignals::default();
        assert_eq!(signals.budget(), DEFAULT_THINK_BUDGET);
        assert!(!signals.should_escalate());
    }

    #[test]
    fn compile_error_triggers_hard_budget() {
        let signals = EscalationSignals {
            compile_error: true,
            ..Default::default()
        };
        assert!(signals.should_escalate());
        assert_eq!(signals.budget(), HARD_THINK_BUDGET);
    }

    #[test]
    fn test_failure_triggers_hard_budget() {
        let signals = EscalationSignals {
            test_failure: true,
            ..Default::default()
        };
        assert!(signals.should_escalate());
        assert_eq!(signals.budget(), HARD_THINK_BUDGET);
    }

    #[test]
    fn patch_rejected_triggers_hard_budget() {
        let signals = EscalationSignals {
            patch_rejected: true,
            ..Default::default()
        };
        assert!(signals.should_escalate());
        assert_eq!(signals.budget(), HARD_THINK_BUDGET);
    }

    #[test]
    fn repeated_failures_escalate_further() {
        let signals = EscalationSignals {
            compile_error: true,
            consecutive_failure_turns: 4,
            ..Default::default()
        };
        let budget = signals.budget();
        assert!(
            budget > HARD_THINK_BUDGET,
            "repeated failures should escalate beyond hard"
        );
        assert!(budget <= MAX_THINK_BUDGET, "should be capped at max");
    }

    #[test]
    fn escalation_capped_at_max() {
        let signals = EscalationSignals {
            compile_error: true,
            consecutive_failure_turns: 100,
            ..Default::default()
        };
        assert_eq!(signals.budget(), MAX_THINK_BUDGET);
    }

    #[test]
    fn scan_detects_rust_compile_error() {
        let mut signals = EscalationSignals::default();
        signals.scan_tool_output(
            "bash_run",
            "error[E0308]: mismatched types\n  expected `u32`",
        );
        assert!(signals.compile_error);
    }

    #[test]
    fn scan_detects_test_failure() {
        let mut signals = EscalationSignals::default();
        signals.scan_tool_output("bash_run", "test result: FAILED. 2 passed; 1 failed;");
        assert!(signals.test_failure);
    }

    #[test]
    fn scan_detects_patch_rejection() {
        let mut signals = EscalationSignals::default();
        signals.scan_tool_output("fs_edit", "Error: no match found for the specified text");
        assert!(signals.patch_rejected);
    }

    #[test]
    fn success_resets_consecutive_failures() {
        let mut signals = EscalationSignals {
            consecutive_failure_turns: 3,
            ..Default::default()
        };
        signals.record_success();
        assert_eq!(signals.consecutive_failure_turns, 0);
    }

    // ── Classification (for planning, not budgets) ──

    #[test]
    fn classify_simple_prompts() {
        assert_eq!(
            classify_complexity("fix typo in readme"),
            PromptComplexity::Simple
        );
        assert_eq!(
            classify_complexity("rename variable foo"),
            PromptComplexity::Simple
        );
        assert_eq!(
            classify_complexity("add import for serde"),
            PromptComplexity::Simple
        );
        assert_eq!(
            classify_complexity("update version to 2.0"),
            PromptComplexity::Simple
        );
    }

    #[test]
    fn minimizers_signal_simplicity() {
        assert_eq!(
            classify_complexity("just fix the typo"),
            PromptComplexity::Simple
        );
        assert_eq!(
            classify_complexity("only rename the variable"),
            PromptComplexity::Simple
        );
        assert_eq!(
            classify_complexity("simply update the comment"),
            PromptComplexity::Simple
        );
    }

    #[test]
    fn just_refactor_is_not_simple() {
        // "just" with complexity keywords should not be treated as simple
        assert_ne!(
            classify_complexity("just refactor everything"),
            PromptComplexity::Simple
        );
        assert_ne!(
            classify_complexity("just rewrite the auth module"),
            PromptComplexity::Simple
        );
        assert_ne!(
            classify_complexity("simply migrate the database"),
            PromptComplexity::Simple
        );
        assert_ne!(
            classify_complexity("only implement the new feature"),
            PromptComplexity::Simple
        );
    }

    #[test]
    fn just_fix_typo_is_simple() {
        assert_eq!(
            classify_complexity("just fix the typo"),
            PromptComplexity::Simple
        );
        assert_eq!(
            classify_complexity("just remove the comma"),
            PromptComplexity::Simple
        );
    }

    #[test]
    fn classify_medium_prompts() {
        assert_eq!(
            classify_complexity("add a login button to the header"),
            PromptComplexity::Medium
        );
        assert_eq!(
            classify_complexity("implement user authentication"),
            PromptComplexity::Medium
        );
        assert_eq!(
            classify_complexity("what does this function do?"),
            PromptComplexity::Medium
        );
    }

    #[test]
    fn classify_complex_prompts() {
        assert_eq!(
            classify_complexity("refactor the auth module across multiple services"),
            PromptComplexity::Complex,
        );
        assert_eq!(
            classify_complexity("redesign the entire codebase to use async/await"),
            PromptComplexity::Complex,
        );
        assert_eq!(
            classify_complexity(
                "migrate the database schema and update all files that reference it"
            ),
            PromptComplexity::Complex,
        );
    }

    #[test]
    fn multi_step_tasks_are_complex() {
        assert_eq!(
            classify_complexity(
                "implement authentication and add rate limiting and update the database schema"
            ),
            PromptComplexity::Complex,
        );
        assert_eq!(
            classify_complexity(
                "1. Add the new endpoint\n2. Update the database\n3. Write tests\n4. Update docs"
            ),
            PromptComplexity::Complex,
        );
    }

    #[test]
    fn many_file_references_are_complex() {
        assert_eq!(
            classify_complexity(
                "Update src/auth.rs, src/middleware.rs, src/routes.rs, and tests/auth_test.rs to use the new token format"
            ),
            PromptComplexity::Complex,
        );
    }

    #[test]
    fn constraint_heavy_tasks_are_complex() {
        assert_eq!(
            classify_complexity(
                "refactor the module and ensure that thread safety is maintained while keeping backward compatible"
            ),
            PromptComplexity::Complex,
        );
    }

    #[test]
    fn error_context_with_code_increases_complexity() {
        let prompt = "This code doesn't compile:\n```rust\nfn main() { broken(); }\n```\nThe error says failed to compile. Fix it while keeping backward compatible.";
        let c = classify_complexity(prompt);
        assert!(
            c == PromptComplexity::Medium || c == PromptComplexity::Complex,
            "error context with code should be at least Medium, got {c:?}"
        );
    }

    #[test]
    fn pure_questions_stay_medium() {
        assert_eq!(
            classify_complexity("what does this function do?"),
            PromptComplexity::Medium
        );
        assert_eq!(
            classify_complexity("how does the auth flow work?"),
            PromptComplexity::Medium
        );
        assert_eq!(
            classify_complexity("explain the architecture"),
            PromptComplexity::Medium
        );
    }

    // ── History-aware ──

    #[test]
    fn history_upgrades_on_repeated_failures() {
        assert_eq!(
            classify_with_history("fix typo", 5, true, 3),
            PromptComplexity::Complex,
        );
    }

    #[test]
    fn history_upgrades_deep_session_with_errors() {
        assert_eq!(
            classify_with_history("add a feature", 20, true, 1),
            PromptComplexity::Complex
        );
        assert_eq!(
            classify_with_history("fix typo", 16, true, 0),
            PromptComplexity::Medium
        );
    }

    #[test]
    fn history_preserves_without_escalation() {
        assert_eq!(
            classify_with_history("fix typo", 0, false, 0),
            PromptComplexity::Simple
        );
    }

    // ── Legacy escalation API ──

    #[test]
    fn escalated_budget_grows_with_failures() {
        let base = DEFAULT_THINK_BUDGET;
        assert_eq!(escalated_budget(base, 0), base);
        assert_eq!(escalated_budget(base, 1), base + ESCALATION_STEP);
        assert_eq!(escalated_budget(base, 2), base + 2 * ESCALATION_STEP);
    }

    #[test]
    fn escalated_budget_capped_at_max() {
        assert_eq!(escalated_budget(HARD_THINK_BUDGET, 100), MAX_THINK_BUDGET);
    }

    // ── Team orchestration ──

    #[test]
    fn complexity_score_increases_for_cross_domain_tasks() {
        let simple = score_prompt("fix typo");
        let complex = score_prompt(
            "refactor backend api and frontend flows across multiple files with tests",
        );
        assert!(complex > simple);
    }

    // ── P9: Batch 3 new tests ──

    #[test]
    fn complex_gets_32k_initial_budget() {
        assert_eq!(
            thinking_budget_for(PromptComplexity::Complex),
            HARD_THINK_BUDGET
        );
        assert_eq!(HARD_THINK_BUDGET, 32_768);
    }

    #[test]
    fn escalation_de_escalates_after_3_successes() {
        let mut signals = EscalationSignals {
            compile_error: true,
            test_failure: true,
            ..Default::default()
        };
        assert!(signals.should_escalate());

        // 3 consecutive successes should de-escalate
        signals.record_success();
        signals.record_success();
        assert!(
            signals.should_escalate(),
            "should still be escalated after 2 successes"
        );
        signals.record_success();
        assert!(
            !signals.should_escalate(),
            "should de-escalate after 3 successes"
        );
        assert!(!signals.compile_error);
        assert!(!signals.test_failure);
        assert_eq!(signals.budget(), DEFAULT_THINK_BUDGET);
    }

    #[test]
    fn search_miss_triggers_escalation() {
        let signals = EscalationSignals {
            search_miss: true,
            ..Default::default()
        };
        assert!(signals.should_escalate());
        assert_eq!(signals.budget(), HARD_THINK_BUDGET);
    }

    #[test]
    fn scan_tool_output_no_false_positive_on_file_not_found() {
        // "no such file" errors from fs_read should NOT trigger compile_error
        let mut signals = EscalationSignals::default();
        signals.scan_tool_output(
            "fs_read",
            "error: no such file or directory: /tmp/missing.rs",
        );
        assert!(
            !signals.compile_error,
            "file-not-found should not flag compile error"
        );
    }

    #[test]
    fn failure_resets_success_counter() {
        let mut signals = EscalationSignals::default();
        signals.record_success();
        signals.record_success();
        assert_eq!(signals.consecutive_success_turns, 2);
        signals.record_failure();
        assert_eq!(signals.consecutive_success_turns, 0);
    }
}
