//! System prompts for the tool-use agent loop.
//!
//! The `TOOL_USE_SYSTEM_PROMPT` is used when the agent runs in tool-use
//! mode (Code/Ask/Context).

/// System prompt that makes deepseek-chat behave as an intelligent coding agent
/// with tool guidance. This prompt is used for the tool-use conversation loop
/// where the model freely decides which tools to call.
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
"#;

/// Workspace context injected into the system prompt environment section.
pub struct WorkspaceContext {
    pub cwd: String,
    pub git_branch: Option<String>,
    pub os: String,
}

/// Build the complete system prompt for a tool-use session.
///
/// Layers:
/// 1. Base tool-use prompt
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
        assert!(
            prompt.contains("NEVER fabricate"),
            "prompt should contain anti-hallucination warning"
        );
        assert!(
            prompt.contains("tool results"),
            "prompt should emphasize responding based on tool results"
        );
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
        // Should NOT contain the default tool-use prompt
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

    // ── Batch 6: System prompt conciseness tests ──

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
        assert!(
            prompt.contains("concise") || prompt.contains("Minimize"),
            "prompt should emphasize brevity"
        );
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
