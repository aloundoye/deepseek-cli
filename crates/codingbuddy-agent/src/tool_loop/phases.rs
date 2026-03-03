//! Phase-aware execution for complex tasks.
//!
//! Implements the Explore→Plan→Execute→Verify workflow with tool filtering
//! per phase. Simple/Medium tasks bypass phases entirely.

use codingbuddy_core::TaskPhase;

/// Read-only tools allowed during the Explore phase.
const EXPLORE_TOOLS: &[&str] = &[
    "fs_read",
    "fs_list",
    "fs_glob",
    "fs_grep",
    "git_status",
    "git_diff",
    "git_show",
    "index_query",
    "git_log",
];

/// Verify phase allows read-only tools plus bash_run for tests.
const VERIFY_TOOLS: &[&str] = &[
    "fs_read",
    "fs_list",
    "fs_glob",
    "fs_grep",
    "git_status",
    "git_diff",
    "git_show",
    "index_query",
    "git_log",
    "bash_run",
];

/// Check if a tool is allowed in the given phase.
/// Returns true if the tool should be included in the request.
/// Plan and Execute phases allow all tools.
pub fn is_tool_allowed_in_phase(tool_name: &str, phase: TaskPhase) -> bool {
    match phase {
        TaskPhase::Explore => {
            // MCP tools always pass through
            if tool_name.starts_with("mcp__") {
                return true;
            }
            EXPLORE_TOOLS.contains(&tool_name)
        }
        TaskPhase::Verify => {
            if tool_name.starts_with("mcp__") {
                return true;
            }
            VERIFY_TOOLS.contains(&tool_name)
        }
        TaskPhase::Plan | TaskPhase::Execute => true,
    }
}

/// Determine if a phase transition should occur based on the current state.
///
/// Returns `Some(new_phase)` if a transition should happen, `None` otherwise.
pub fn check_phase_transition(
    current_phase: TaskPhase,
    read_only_tool_calls: usize,
    has_text_response: bool,
    text_has_plan_keywords: bool,
    used_write_tool: bool,
    edit_tool_calls_since_execute: usize,
) -> Option<TaskPhase> {
    match current_phase {
        TaskPhase::Explore => {
            // Manual override: if model used a write tool, jump to Execute
            if used_write_tool {
                return Some(TaskPhase::Execute);
            }
            // After 3+ read-only tool calls, transition to Plan
            if read_only_tool_calls >= 3 && has_text_response {
                return Some(TaskPhase::Plan);
            }
            None
        }
        TaskPhase::Plan => {
            // After model produces text with plan keywords, transition to Execute
            if has_text_response && text_has_plan_keywords {
                return Some(TaskPhase::Execute);
            }
            None
        }
        TaskPhase::Execute => {
            // After text-only response (no tool calls) or N edit calls, transition to Verify
            if has_text_response || edit_tool_calls_since_execute >= 5 {
                return Some(TaskPhase::Verify);
            }
            None
        }
        TaskPhase::Verify => {
            // Verify is terminal — no further transitions
            None
        }
    }
}

/// Check if text contains plan-like keywords indicating the model is stating its plan.
pub fn text_has_plan_keywords(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    // Look for plan indicators
    let has_plan_marker = lower.contains("i will")
        || lower.contains("my plan")
        || lower.contains("steps:")
        || lower.contains("step 1")
        || lower.contains("approach:")
        || lower.contains("strategy:");

    // Also check for numbered lists (1. 2. 3.)
    let has_numbered_list = {
        let lines: Vec<&str> = text.lines().collect();
        let numbered_count = lines
            .iter()
            .filter(|l| {
                let trimmed = l.trim();
                trimmed.starts_with("1.") || trimmed.starts_with("2.") || trimmed.starts_with("3.")
            })
            .count();
        numbered_count >= 2
    };

    has_plan_marker || has_numbered_list
}

/// System message to inject when transitioning to Plan phase.
pub const PLAN_TRANSITION_MESSAGE: &str = "You have explored the codebase. Now state your plan: list the files to modify in order, \
     what changes each needs, and what risks to watch for. Be specific.";

/// System message to inject when transitioning to Verify phase.
pub const VERIFY_TRANSITION_MESSAGE: &str = "Implementation complete. Now verify your changes: run tests, check for compilation errors, \
     and confirm the changes work as intended.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explore_allows_read_only() {
        assert!(is_tool_allowed_in_phase("fs_read", TaskPhase::Explore));
        assert!(is_tool_allowed_in_phase("fs_grep", TaskPhase::Explore));
        assert!(is_tool_allowed_in_phase("git_status", TaskPhase::Explore));
    }

    #[test]
    fn explore_blocks_write_tools() {
        assert!(!is_tool_allowed_in_phase("fs_edit", TaskPhase::Explore));
        assert!(!is_tool_allowed_in_phase("fs_write", TaskPhase::Explore));
        assert!(!is_tool_allowed_in_phase("bash_run", TaskPhase::Explore));
    }

    #[test]
    fn explore_allows_mcp() {
        assert!(is_tool_allowed_in_phase(
            "mcp__server__tool",
            TaskPhase::Explore
        ));
    }

    #[test]
    fn execute_allows_all() {
        assert!(is_tool_allowed_in_phase("fs_edit", TaskPhase::Execute));
        assert!(is_tool_allowed_in_phase("bash_run", TaskPhase::Execute));
        assert!(is_tool_allowed_in_phase("fs_read", TaskPhase::Execute));
    }

    #[test]
    fn verify_allows_bash_run() {
        assert!(is_tool_allowed_in_phase("bash_run", TaskPhase::Verify));
        assert!(is_tool_allowed_in_phase("fs_read", TaskPhase::Verify));
        assert!(!is_tool_allowed_in_phase("fs_edit", TaskPhase::Verify));
    }

    #[test]
    fn explore_to_plan_after_reads() {
        let result = check_phase_transition(TaskPhase::Explore, 3, true, false, false, 0);
        assert_eq!(result, Some(TaskPhase::Plan));
    }

    #[test]
    fn explore_to_execute_on_write() {
        let result = check_phase_transition(TaskPhase::Explore, 1, false, false, true, 0);
        assert_eq!(result, Some(TaskPhase::Execute));
    }

    #[test]
    fn plan_to_execute_with_keywords() {
        let result = check_phase_transition(TaskPhase::Plan, 0, true, true, false, 0);
        assert_eq!(result, Some(TaskPhase::Execute));
    }

    #[test]
    fn plan_keywords_detection() {
        assert!(text_has_plan_keywords("I will modify the following files:"));
        assert!(text_has_plan_keywords("Steps:\n1. Read file\n2. Edit"));
        assert!(text_has_plan_keywords(
            "1. First thing\n2. Second thing\n3. Third"
        ));
        assert!(!text_has_plan_keywords("The code looks fine."));
    }
}
