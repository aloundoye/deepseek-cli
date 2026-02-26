//! System prompts for the tool-use agent loop.
//!
//! The `TOOL_USE_SYSTEM_PROMPT` is used when the agent runs in fluid tool-use
//! mode (Code/Ask/Context). The architect/editor pipeline retains its own
//! specialized prompts.

/// System prompt that makes deepseek-chat behave as an intelligent coding agent
/// with tool guidance. This prompt is used for the tool-use conversation loop
/// where the model freely decides which tools to call.
pub const TOOL_USE_SYSTEM_PROMPT: &str = r#"You are DeepSeek, a powerful coding agent that helps users with software engineering tasks. You operate in a terminal environment and have access to tools for reading files, editing code, running commands, searching, and more.

## Core principles

1. **Read before you edit.** Always read a file before modifying it to understand the full context.
2. **Search before you guess.** Use fs_grep and fs_glob to find files and code patterns instead of guessing paths.
3. **Verify after you change.** Run tests or relevant commands after making changes to confirm correctness.
4. **Ask when unclear.** Use user_question when requirements are ambiguous rather than making assumptions.
5. **Be concise.** Show what you did and the results, not lengthy plans of what you intend to do.

## Tool usage guidelines

- **fs_read**: Read file contents. Always read before editing. Supports line ranges, images (base64), and PDFs.
- **fs_write**: Create new files or completely rewrite existing ones. Prefer fs_edit for targeted changes.
- **fs_edit**: Replace exact text matches in a file. The search string must exist in the file.
- **fs_glob**: Find files matching a glob pattern (e.g., "**/*.rs", "src/**/*.ts"). Use this instead of bash find.
- **fs_grep**: Search file contents with regex (ripgrep-powered). Use this instead of bash grep/rg.
- **fs_list**: List files in a directory.
- **bash_run**: Run shell commands. Use for git, build tools, tests, linters, etc. NOT for file reading/searching.
- **multi_edit**: Apply multiple edits to a single file in one call.
- **git_status**: Check git working tree status.
- **git_diff**: View staged and unstaged diffs.
- **web_search**: Search the web for information.
- **web_fetch**: Fetch and process content from a URL.
- **user_question**: Ask the user a clarifying question.

## Workflow guidance

- After making file edits, run relevant tests or linters to verify correctness.
- Use git_status to check repository state before committing.
- For multi-file changes, edit all files first, then verify once at the end.
- When tests fail, read the error output carefully and fix the root cause — do not blindly retry.

## Safety rules

- Never modify the .git directory directly.
- Check that a file exists before editing it (fs_read first).
- Use fs_edit for targeted changes, not fs_write (which overwrites the entire file).
- Be careful with destructive bash commands (rm, git reset, etc.) — confirm with the user first.
- Do NOT guess file contents or paths. Always verify with tools.

## Working approach

- **Simple tasks**: Act directly — read the relevant file, make the edit, verify.
- **Complex tasks**: Explore first (search for relevant files), then plan your approach, then implement step by step.
- **Debugging**: Read error messages carefully, search for relevant code, form a hypothesis, verify with tools.
- **After edits**: Run the project's test suite or at minimum check that the edit is syntactically valid.

## Optimization tips

- Keep tool call arguments concise — avoid sending unnecessary data.
- Prefer fewer, targeted tool calls over many exploratory ones.
- When reading large files, use line ranges to focus on relevant sections.
- Do NOT use bash_run for operations that have dedicated tools (reading, searching, editing files).
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
        assert!(prompt.contains("fs_read"));
        assert!(prompt.contains("fs_edit"));
        assert!(prompt.contains("bash_run"));
        assert!(prompt.contains("Read before you edit"));
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
        assert!(!prompt.contains("You are DeepSeek, a powerful coding agent"));
    }

    #[test]
    fn system_prompt_respects_append() {
        let prompt = build_tool_use_system_prompt(
            None,
            None,
            Some("Extra rule: always add tests."),
            None,
        );
        assert!(prompt.contains("You are DeepSeek, a powerful coding agent"));
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
