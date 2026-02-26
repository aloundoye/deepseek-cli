//! System prompts for the tool-use agent loop.
//!
//! One stable system prompt with always-included planning/verification guidance.
//! DeepSeek follows stable system constraints well — a single consistent
//! instruction set is more robust than dynamically switching tiers.

/// System prompt that makes deepseek-chat behave as an intelligent coding agent.
///
/// Contains: tool policy, planning rubric, patch workflow, stop conditions.
/// This is always the same regardless of task complexity — the model decides
/// when to plan vs proceed based on the guidance.
pub const TOOL_USE_SYSTEM_PROMPT: &str = r#"You are DeepSeek, an expert software engineering assistant operating in a terminal.

## CRITICAL RULES
1. ALWAYS use tools to gather information. NEVER fabricate file contents, paths, or project structure.
2. Read files before editing them. Search before guessing paths.
3. Be concise: respond in 1-3 sentences unless showing code. No preamble. No lengthy plans.
4. Mimic existing code style. Never assume a library is available without checking.
5. Do not add comments, docstrings, or type annotations unless asked.
6. After making changes, verify with tests or relevant commands.

## OUTPUT RULES
- Minimize output tokens. Show results, not plans.
- When multiple independent lookups are needed, call multiple tools simultaneously.
- Respond based ONLY on tool results, never from memory.

Tool descriptions contain detailed usage instructions. Read them carefully.

## WORKING PROTOCOL
For trivial changes (rename, fix typo, add import): just do it. No planning needed.

For anything non-trivial, follow this workflow:
1. **Read before write**: Use `fs_read` on every file you plan to modify. Understand existing patterns.
2. **Search before assuming**: Use `fs_glob` and `fs_grep` to find files. Do NOT guess paths.
3. **Trace impacts**: If changing a type, function signature, or module interface, grep for all call sites first.
4. **Verify after changes**: Run the build command or test suite after modifications.
5. **One step at a time**: Make changes incrementally. Verify each file before moving to the next.

For multi-file refactors, migrations, or architectural changes:
- State your plan briefly BEFORE making changes (which files, in what order, what risks).
- Modify files in dependency order (dependencies first, dependents after).
- Run tests after each file change, not just at the end.

### ANTI-PATTERNS
- Do NOT edit a file you haven't read.
- Do NOT change a type/interface without grepping for all usages.
- Do NOT skip running tests after changes.
- Do NOT make changes beyond what was requested.
"#;

use crate::complexity::PromptComplexity;

/// Workspace context injected into the system prompt environment section.
pub struct WorkspaceContext {
    pub cwd: String,
    pub git_branch: Option<String>,
    pub os: String,
}

/// Build the complete system prompt for a tool-use session.
///
/// Layers:
/// 1. Base tool-use prompt (always includes planning/verification guidance)
/// 2. Environment context (cwd, git branch, OS)
/// 3. Project memory (DEEPSEEK.md equivalent)
/// 4. User system prompt override or append
pub fn build_tool_use_system_prompt(
    project_memory: Option<&str>,
    system_prompt_override: Option<&str>,
    system_prompt_append: Option<&str>,
    workspace_context: Option<&WorkspaceContext>,
) -> String {
    // If the user provides a complete override, use it directly
    if let Some(override_prompt) = system_prompt_override {
        let mut prompt = override_prompt.to_string();
        if let Some(ctx) = workspace_context {
            prompt.push_str(&format_environment_section(ctx));
        }
        if let Some(memory) = project_memory {
            prompt.push_str("\n\n# Project Instructions\n\n");
            prompt.push_str(memory);
        }
        return prompt;
    }

    let mut parts = vec![TOOL_USE_SYSTEM_PROMPT.to_string()];

    if let Some(ctx) = workspace_context {
        parts.push(format_environment_section(ctx));
    }

    if let Some(memory) = project_memory
        && !memory.is_empty()
    {
        parts.push(format!("\n# Project Instructions (DEEPSEEK.md)\n\n{memory}"));
    }

    if let Some(append) = system_prompt_append
        && !append.is_empty()
    {
        parts.push(format!("\n# Additional Instructions\n\n{append}"));
    }

    parts.join("\n")
}

/// Build system prompt with complexity-based additions.
///
/// The base prompt always includes the working protocol. For Complex tasks,
/// we add an explicit "PLAN FIRST" reminder. For Simple, nothing extra.
/// This is a lightweight nudge, not a different prompt.
pub fn build_tool_use_system_prompt_with_complexity(
    project_memory: Option<&str>,
    system_prompt_override: Option<&str>,
    system_prompt_append: Option<&str>,
    workspace_context: Option<&WorkspaceContext>,
    complexity: PromptComplexity,
) -> String {
    let base = build_tool_use_system_prompt(
        project_memory,
        system_prompt_override,
        system_prompt_append,
        workspace_context,
    );

    // Don't inject anything if user provided a full system prompt override.
    if system_prompt_override.is_some() {
        return base;
    }

    match complexity {
        PromptComplexity::Complex => format!("{base}{COMPLEX_REMINDER}"),
        _ => base,
    }
}

/// Brief reminder for complex tasks. The full protocol is already in the base
/// prompt — this just reinforces it for tasks where skipping planning is risky.
const COMPLEX_REMINDER: &str = "\n\n\
## IMPORTANT: This looks like a complex task.\n\
Follow the WORKING PROTOCOL above strictly. State your plan before making changes.\n\
Modify files in dependency order and verify after each change.\n";

/// Format the environment section for the system prompt.
fn format_environment_section(ctx: &WorkspaceContext) -> String {
    let mut section = String::from("\n# Environment\n\n");
    section.push_str(&format!("- Working directory: {}\n", ctx.cwd));
    if let Some(ref branch) = ctx.git_branch {
        section.push_str(&format!("- Git branch: {branch}\n"));
    }
    section.push_str(&format!("- OS: {}\n", ctx.os));
    section
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompt_includes_tool_guidance() {
        let prompt = build_tool_use_system_prompt(None, None, None, None);
        assert!(prompt.contains("ALWAYS use tools"));
        assert!(prompt.contains("Read files before editing"));
    }

    #[test]
    fn system_prompt_includes_anti_hallucination_rules() {
        let prompt = build_tool_use_system_prompt(None, None, None, None);
        assert!(prompt.contains("NEVER fabricate"));
        assert!(prompt.contains("tool results"));
    }

    #[test]
    fn system_prompt_always_includes_working_protocol() {
        let prompt = build_tool_use_system_prompt(None, None, None, None);
        assert!(prompt.contains("WORKING PROTOCOL"), "should always include protocol");
        assert!(prompt.contains("Read before write"), "should include read-first rule");
        assert!(prompt.contains("ANTI-PATTERNS"), "should include anti-patterns");
        assert!(prompt.contains("grep for all call sites"), "should include impact tracing");
    }

    #[test]
    fn system_prompt_includes_project_memory() {
        let prompt = build_tool_use_system_prompt(
            Some("Always use snake_case in Rust code."),
            None,
            None,
            None,
        );
        assert!(prompt.contains("Always use snake_case in Rust code."));
        assert!(prompt.contains("Project Instructions"));
    }

    #[test]
    fn system_prompt_respects_override() {
        let prompt = build_tool_use_system_prompt(
            Some("project memory"),
            Some("Custom system prompt"),
            None,
            None,
        );
        assert!(prompt.starts_with("Custom system prompt"));
        assert!(prompt.contains("project memory"));
        assert!(!prompt.contains("You are DeepSeek, an expert software"));
    }

    #[test]
    fn system_prompt_respects_append() {
        let prompt = build_tool_use_system_prompt(
            None,
            None,
            Some("Extra rule: always add tests."),
            None,
        );
        assert!(prompt.contains("You are DeepSeek, an expert software"));
        assert!(prompt.contains("Extra rule: always add tests."));
        assert!(prompt.contains("Additional Instructions"));
    }

    #[test]
    fn system_prompt_empty_memory_not_added() {
        let prompt = build_tool_use_system_prompt(Some(""), None, None, None);
        assert!(!prompt.contains("Project Instructions"));
    }

    #[test]
    fn system_prompt_includes_workspace_context() {
        let ctx = WorkspaceContext {
            cwd: "/home/user/project".to_string(),
            git_branch: Some("main".to_string()),
            os: "linux".to_string(),
        };
        let prompt = build_tool_use_system_prompt(None, None, None, Some(&ctx));
        assert!(prompt.contains("/home/user/project"));
        assert!(prompt.contains("Git branch: main"));
        assert!(prompt.contains("OS: linux"));
    }

    #[test]
    fn system_prompt_is_concise() {
        let line_count = TOOL_USE_SYSTEM_PROMPT.lines().count();
        assert!(
            line_count < 50,
            "system prompt should be concise (< 50 lines), got {line_count}"
        );
    }

    #[test]
    fn system_prompt_emphasizes_brevity() {
        let prompt = build_tool_use_system_prompt(None, None, None, None);
        assert!(prompt.contains("concise") || prompt.contains("Minimize"));
    }

    // ── Complexity injection ──

    #[test]
    fn complex_task_injects_reminder() {
        let prompt = build_tool_use_system_prompt_with_complexity(
            None, None, None, None,
            PromptComplexity::Complex,
        );
        assert!(prompt.contains("IMPORTANT"), "complex should get reminder");
        assert!(prompt.contains("WORKING PROTOCOL"), "should have protocol in base");
        assert!(prompt.contains("ANTI-PATTERNS"), "should have anti-patterns in base");
    }

    #[test]
    fn medium_task_uses_base_prompt_only() {
        let prompt = build_tool_use_system_prompt_with_complexity(
            None, None, None, None,
            PromptComplexity::Medium,
        );
        assert!(prompt.contains("WORKING PROTOCOL"), "medium gets base protocol");
        assert!(!prompt.contains("IMPORTANT: This looks like"), "medium should NOT get reminder");
    }

    #[test]
    fn simple_task_uses_base_prompt_only() {
        let prompt = build_tool_use_system_prompt_with_complexity(
            None, None, None, None,
            PromptComplexity::Simple,
        );
        assert!(prompt.contains("WORKING PROTOCOL"), "simple gets base protocol");
        assert!(!prompt.contains("IMPORTANT: This looks like"), "simple should NOT get reminder");
    }

    #[test]
    fn override_skips_complexity_injection() {
        let prompt = build_tool_use_system_prompt_with_complexity(
            None,
            Some("Custom prompt"),
            None,
            None,
            PromptComplexity::Complex,
        );
        assert!(!prompt.contains("IMPORTANT: This looks like"));
    }

    #[test]
    fn system_prompt_environment_section_no_branch() {
        let ctx = WorkspaceContext {
            cwd: "/tmp/test".to_string(),
            git_branch: None,
            os: "macos".to_string(),
        };
        let prompt = build_tool_use_system_prompt(None, None, None, Some(&ctx));
        assert!(prompt.contains("/tmp/test"));
        assert!(!prompt.contains("Git branch"));
        assert!(prompt.contains("OS: macos"));
    }
}
