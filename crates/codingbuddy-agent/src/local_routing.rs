//! Local inference routing — decides when to use a local model vs the DeepSeek API.
//!
//! For Simple tasks that require no tool use, a small local model (e.g. DeepSeek-Coder-1.3B)
//! can answer faster and at zero API cost. This module provides the routing decision logic
//! and a facade that tries local inference first, falling back to API on failure.

use crate::complexity::PromptComplexity;

/// Criteria for routing a request to local inference.
#[derive(Debug, Clone)]
pub struct LocalRoutingDecision {
    /// Whether to attempt local inference.
    pub use_local: bool,
    /// Reason for the decision.
    pub reason: &'static str,
}

/// Maximum prompt length (chars) considered "locally solvable".
const MAX_LOCAL_PROMPT_CHARS: usize = 2000;

/// Maximum expected output tokens for local inference.
const MAX_LOCAL_OUTPUT_TOKENS: u32 = 512;

/// Keywords that indicate the task needs tool use (not locally solvable).
const TOOL_REQUIRING_KEYWORDS: &[&str] = &[
    "fix",
    "edit",
    "change",
    "modify",
    "update",
    "create",
    "write",
    "delete",
    "refactor",
    "implement",
    "add",
    "remove",
    "replace",
    "rename",
    "run",
    "test",
    "build",
    "compile",
    "deploy",
    "install",
    "commit",
    "push",
    "pull",
    "merge",
    "rebase",
    "debug",
    "profile",
    "benchmark",
];

/// Keywords that suggest a simple Q&A that can be answered locally.
const LOCAL_FRIENDLY_KEYWORDS: &[&str] = &[
    "what is",
    "what are",
    "how does",
    "explain",
    "describe",
    "summarize",
    "compare",
    "difference between",
    "meaning of",
    "definition of",
    "why",
    "when",
    "which",
    "tell me about",
];

/// Keywords that indicate the prompt is about the user's project/codebase.
/// These ALWAYS route to API because local models lack project context.
const PROJECT_CONTEXT_KEYWORDS: &[&str] = &[
    "this file",
    "this code",
    "this project",
    "this repo",
    "this crate",
    "this module",
    "this function",
    "the codebase",
    "our code",
    "my code",
    "my project",
    "the project",
    "in this",
    "in the repo",
    "here",
];

/// Decide whether a user prompt should be routed to local inference.
///
/// Returns `use_local=true` when ALL of these conditions hold:
/// 1. Complexity is `Simple` (no multi-step reasoning needed)
/// 2. Prompt is short enough for local context window
/// 3. Prompt does NOT contain keywords indicating tool use is needed
/// 4. Prompt DOES contain keywords suggesting a simple Q&A
pub fn should_use_local(
    prompt: &str,
    complexity: PromptComplexity,
    local_ml_enabled: bool,
) -> LocalRoutingDecision {
    if !local_ml_enabled {
        return LocalRoutingDecision {
            use_local: false,
            reason: "local ML not enabled",
        };
    }

    if complexity != PromptComplexity::Simple {
        return LocalRoutingDecision {
            use_local: false,
            reason: "complexity not Simple",
        };
    }

    if prompt.len() > MAX_LOCAL_PROMPT_CHARS {
        return LocalRoutingDecision {
            use_local: false,
            reason: "prompt too long for local model",
        };
    }

    let lower = prompt.to_ascii_lowercase();

    // Check for project-context keywords — always route to API
    if PROJECT_CONTEXT_KEYWORDS.iter().any(|kw| lower.contains(kw)) {
        return LocalRoutingDecision {
            use_local: false,
            reason: "prompt references project context — needs API with tools",
        };
    }

    // Check for tool-requiring keywords
    if TOOL_REQUIRING_KEYWORDS.iter().any(|kw| lower.contains(kw)) {
        return LocalRoutingDecision {
            use_local: false,
            reason: "prompt suggests tool use needed",
        };
    }

    // Check for locally-friendly keywords
    let is_friendly = LOCAL_FRIENDLY_KEYWORDS.iter().any(|kw| lower.contains(kw));

    if is_friendly {
        return LocalRoutingDecision {
            use_local: true,
            reason: "simple Q&A suitable for local model",
        };
    }

    // Default: don't route locally unless we're confident
    LocalRoutingDecision {
        use_local: false,
        reason: "no strong signal for local routing",
    }
}

/// Maximum output tokens for local inference.
pub fn local_max_tokens() -> u32 {
    MAX_LOCAL_OUTPUT_TOKENS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_qa_routes_locally() {
        let decision =
            should_use_local("What is a closure in Rust?", PromptComplexity::Simple, true);
        assert!(decision.use_local, "simple Q&A should route locally");
    }

    #[test]
    fn complex_task_stays_on_api() {
        let decision = should_use_local("What is a closure?", PromptComplexity::Complex, true);
        assert!(!decision.use_local, "complex task should stay on API");
    }

    #[test]
    fn tool_requiring_stays_on_api() {
        let decision = should_use_local("Fix the bug in main.rs", PromptComplexity::Simple, true);
        assert!(!decision.use_local, "fix/edit tasks should stay on API");
    }

    #[test]
    fn long_prompt_stays_on_api() {
        let long_prompt = format!("Explain this: {}", "x".repeat(3000));
        let decision = should_use_local(&long_prompt, PromptComplexity::Simple, true);
        assert!(!decision.use_local, "long prompt should stay on API");
    }

    #[test]
    fn local_ml_disabled_stays_on_api() {
        let decision = should_use_local("What is a closure?", PromptComplexity::Simple, false);
        assert!(!decision.use_local, "disabled local ML should stay on API");
    }

    #[test]
    fn ambiguous_prompt_stays_on_api() {
        let decision = should_use_local("hello world", PromptComplexity::Simple, true);
        assert!(!decision.use_local, "ambiguous prompt should stay on API");
    }

    // ── Project context routing ──

    #[test]
    fn project_context_routes_to_api() {
        let decision = should_use_local("What does this code do?", PromptComplexity::Simple, true);
        assert!(!decision.use_local, "project context should route to API");
        assert!(decision.reason.contains("project context"));
    }

    #[test]
    fn codebase_reference_routes_to_api() {
        let decision = should_use_local(
            "Explain the codebase architecture",
            PromptComplexity::Simple,
            true,
        );
        assert!(!decision.use_local, "'the codebase' should route to API");
    }

    #[test]
    fn this_file_reference_routes_to_api() {
        let decision = should_use_local("What is this file for?", PromptComplexity::Simple, true);
        assert!(!decision.use_local, "'this file' should route to API");
    }

    #[test]
    fn generic_question_without_project_context_can_route_locally() {
        let decision = should_use_local("What is a mutex in Rust?", PromptComplexity::Simple, true);
        assert!(
            decision.use_local,
            "generic Q&A without project context should route locally"
        );
    }
}
