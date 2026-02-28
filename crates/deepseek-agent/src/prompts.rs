//! System prompts for the tool-use agent loop.
//!
//! Two model-tier prompts: CHAT (action-biased, compensates for weaker reasoning)
//! and REASONER (thinking-leveraging, grants more autonomy). Selected by model name.

/// Default/fallback system prompt used by both tiers. Kept as compatibility alias.
pub const TOOL_USE_SYSTEM_PROMPT: &str = CHAT_SYSTEM_PROMPT;

/// System prompt for `deepseek-chat` — action-biased, compensates for weak reasoning.
///
/// Key principles:
/// - "Just do it" bias: minimize planning, maximize action
/// - Aggressive anti-hallucination: never fabricate anything
/// - Explicit DO NOT list to constrain the weaker model
/// - Strict verification requirements after every change
pub const CHAT_SYSTEM_PROMPT: &str = r#"You are DeepSeek, an expert software engineering assistant operating in a terminal.

## PRIME DIRECTIVE
Do the work. Do not ask for permission. Do not explain what you will do. Just do it.

## CRITICAL RULES
1. ALWAYS use tools to gather information. NEVER fabricate file contents, paths, or project structure.
2. Read files before editing them. Search before guessing paths.
3. Be concise: respond in 1-3 sentences unless showing code. No preamble.
4. Mimic existing code style. Never assume a library is available without checking.
5. Do not add comments, docstrings, or type annotations unless asked.
6. After making changes, verify with tests or relevant commands.

## DO NOT
- Guess file paths — use `fs_glob` or `fs_list` to find them.
- Skip verification — run tests after every change.
- Ask the user to do things you can do with tools.
- Synthesize answers from memory — respond ONLY based on tool results.
- Make changes beyond what was requested.
- Edit a file you haven't read in this session.

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

/// System prompt for `deepseek-reasoner` — leverages native chain-of-thought.
///
/// Key principles:
/// - Grants autonomy: reasoner can self-plan via thinking
/// - Less prescriptive than chat (reasoner self-plans)
/// - Directs thinking to high-value tasks (planning, error analysis)
/// - Same core safety rules (read before edit, verify)
pub const REASONER_SYSTEM_PROMPT: &str = r#"You are DeepSeek, an expert software engineering assistant with extended thinking.

## CORE RULES
1. ALWAYS use tools to gather information. NEVER fabricate file contents, paths, or code.
2. Read files before editing. Search before guessing paths.
3. Be concise in your responses. Use thinking for internal planning.
4. After changes, verify with tests or the build command.

## THINKING STRATEGY
Use your thinking capability strategically:
- **Before complex edits**: Plan multi-file changes, trace impacts, verify understanding.
- **After errors**: Analyze root cause in thinking before retrying.
- **For architecture**: Think through dependency order and risks.
- Do NOT use thinking for trivial operations (reading files, running commands).

## WORKING PROTOCOL
- Read every file before editing it.
- If changing a type/interface, grep for all callers first.
- Modify files in dependency order. Verify after each change.
- State your plan in thinking, then execute. Show results, not plans.

## DO NOT
- Fabricate file paths or content.
- Skip verification after changes.
- Make changes beyond what was requested.
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
        parts.push(format!(
            "\n# Project Instructions (DEEPSEEK.md)\n\n{memory}"
        ));
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
/// we add a full planning protocol. For Medium, lightweight guidance.
/// For Simple, no extra injection.
pub fn build_tool_use_system_prompt_with_complexity(
    project_memory: Option<&str>,
    system_prompt_override: Option<&str>,
    system_prompt_append: Option<&str>,
    workspace_context: Option<&WorkspaceContext>,
    complexity: PromptComplexity,
    repo_map_summary: Option<&str>,
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
        PromptComplexity::Complex => {
            let mut prompt = format!("{base}{COMPLEX_REMINDER}");
            if let Some(repo_map) = repo_map_summary
                && !repo_map.is_empty()
            {
                prompt.push_str(&format!("\n## Project Files\n{repo_map}\n"));
            }
            prompt
        }
        PromptComplexity::Medium => format!("{base}{MEDIUM_GUIDANCE}"),
        PromptComplexity::Simple => base,
    }
}

/// Full planning protocol for Complex tasks. Provides step-by-step methodology
/// with explore→plan→execute phases and explicit anti-patterns.
const COMPLEX_REMINDER: &str = "\n\n\
## COMPLEX TASK — Mandatory Planning Protocol\n\n\
This task requires architectural thinking. Before making ANY changes:\n\n\
### Step 1: Explore\n\
- Read ALL files you plan to modify\n\
- `fs_grep` for every type, function, or interface you'll change to find ALL call sites\n\
- Identify the dependency order: which files depend on which\n\n\
### Step 2: Plan (state this explicitly)\n\
- List the files to modify in dependency order (change dependencies BEFORE dependents)\n\
- For each file: what changes, what could break, what to verify\n\
- Identify risks: shared state, concurrent access, type mismatches, missing imports\n\n\
### Step 3: Execute Incrementally\n\
- Modify ONE file at a time\n\
- After each file: run `bash_run` with the build/test command to verify\n\
- If a test fails: fix it BEFORE moving to the next file\n\
- If your plan was wrong: stop, re-read affected files, adjust plan\n\n\
### Anti-Patterns (NEVER do these)\n\
- Editing a file you haven't read in THIS session\n\
- Changing a function signature without grepping for all callers\n\
- Making all changes then testing at the end (test after EACH change)\n\
- Continuing after a test failure without fixing it first\n";

/// Lightweight guidance for Medium-complexity tasks. Not the full protocol,
/// but reminds the model to read-before-write and verify after changes.
const MEDIUM_GUIDANCE: &str = "\n\n\
## Task Guidance\n\
This is a multi-step task. Before making changes:\n\
1. Read the files you plan to modify.\n\
2. If changing an interface (function signature, type, struct field), grep for all usages first.\n\
3. After changes, run tests to verify.\n";

/// Additional system prompt section for deepseek-reasoner model.
/// The reasoner has native chain-of-thought, so we guide it to use thinking
/// for planning and verification rather than just acting.
pub const REASONER_GUIDANCE: &str = "\n\n\
## Model: DeepSeek-Reasoner\n\
You have extended thinking capabilities. Use them strategically:\n\
- **Before complex edits**: Think through the change, its impacts, and verify your understanding.\n\
- **After errors**: Use thinking to analyze why the error occurred before retrying.\n\
- **For multi-file changes**: Think through the dependency order and plan before acting.\n\
Do NOT use thinking for trivial operations (reading files, running commands).\n";

/// Additional prescriptive guidance for deepseek-chat when handling Complex tasks
/// without thinking mode. Since chat lacks native reasoning, we compensate with
/// explicit step-by-step instructions.
pub const CHAT_PRESCRIPTIVE_GUIDANCE: &str = "\n\n\
## Explicit Verification Protocol\n\
After EVERY file modification, you MUST:\n\
1. State what you changed and why (one sentence)\n\
2. Run the build/test command to verify\n\
3. If the test fails, re-read the error FULLY before making another edit\n\
\n\
Every 5 tool calls, ask yourself: have I verified all file paths exist? \
Am I working on the right files? Have I read the files I'm about to edit?\n";

/// Build system prompt with model-specific base prompt selection.
///
/// Selects `CHAT_SYSTEM_PROMPT` or `REASONER_SYSTEM_PROMPT` by model name,
/// then layers complexity and environment context on top.
pub fn build_model_aware_system_prompt(
    project_memory: Option<&str>,
    system_prompt_override: Option<&str>,
    system_prompt_append: Option<&str>,
    workspace_context: Option<&WorkspaceContext>,
    complexity: PromptComplexity,
    repo_map_summary: Option<&str>,
    model: &str,
) -> String {
    // If user provided a complete override, skip model selection
    if system_prompt_override.is_some() {
        return build_tool_use_system_prompt_with_complexity(
            project_memory,
            system_prompt_override,
            system_prompt_append,
            workspace_context,
            complexity,
            repo_map_summary,
        );
    }

    // Select base prompt by model tier
    let is_reasoner = model.contains("reasoner");
    let base_prompt = if is_reasoner {
        REASONER_SYSTEM_PROMPT
    } else {
        CHAT_SYSTEM_PROMPT
    };

    // Build with model-specific base
    let base = build_tool_use_system_prompt_with_base(
        base_prompt,
        project_memory,
        system_prompt_append,
        workspace_context,
        complexity,
        repo_map_summary,
    );

    // Add model-tier-specific guidance on top
    if is_reasoner {
        format!("{base}{REASONER_GUIDANCE}")
    } else if complexity == PromptComplexity::Complex {
        format!("{base}{CHAT_PRESCRIPTIVE_GUIDANCE}")
    } else {
        base
    }
}

/// Build system prompt from an explicit base prompt (used by model-aware builder).
fn build_tool_use_system_prompt_with_base(
    base_prompt: &str,
    project_memory: Option<&str>,
    system_prompt_append: Option<&str>,
    workspace_context: Option<&WorkspaceContext>,
    complexity: PromptComplexity,
    repo_map_summary: Option<&str>,
) -> String {
    let mut parts = vec![base_prompt.to_string()];

    if let Some(ctx) = workspace_context {
        parts.push(format_environment_section(ctx));
    }

    if let Some(memory) = project_memory
        && !memory.is_empty()
    {
        parts.push(format!(
            "\n# Project Instructions (DEEPSEEK.md)\n\n{memory}"
        ));
    }

    if let Some(append) = system_prompt_append
        && !append.is_empty()
    {
        parts.push(format!("\n# Additional Instructions\n\n{append}"));
    }

    let base = parts.join("\n");

    // Apply complexity injection (same logic as build_tool_use_system_prompt_with_complexity)
    match complexity {
        PromptComplexity::Complex => {
            let mut prompt = format!("{base}{COMPLEX_REMINDER}");
            if let Some(repo_map) = repo_map_summary
                && !repo_map.is_empty()
            {
                prompt.push_str(&format!("\n## Project Files\n{repo_map}\n"));
            }
            prompt
        }
        PromptComplexity::Medium => format!("{base}{MEDIUM_GUIDANCE}"),
        PromptComplexity::Simple => base,
    }
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
        assert!(prompt.contains("NEVER fabricate"));
        assert!(prompt.contains("tool results"));
    }

    #[test]
    fn system_prompt_always_includes_working_protocol() {
        let prompt = build_tool_use_system_prompt(None, None, None, None);
        assert!(
            prompt.contains("WORKING PROTOCOL"),
            "should always include protocol"
        );
        assert!(
            prompt.contains("Read before write"),
            "should include read-first rule"
        );
        assert!(
            prompt.contains("ANTI-PATTERNS"),
            "should include anti-patterns"
        );
        assert!(
            prompt.contains("grep for all call sites"),
            "should include impact tracing"
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
        assert!(!prompt.contains("You are DeepSeek, an expert software"));
    }

    #[test]
    fn system_prompt_respects_append() {
        let prompt =
            build_tool_use_system_prompt(None, None, Some("Extra rule: always add tests."), None);
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
        let chat_lines = CHAT_SYSTEM_PROMPT.lines().count();
        assert!(
            chat_lines < 55,
            "chat system prompt should be concise (< 55 lines), got {chat_lines}"
        );
        let reasoner_lines = REASONER_SYSTEM_PROMPT.lines().count();
        assert!(
            reasoner_lines < 30,
            "reasoner prompt should be concise (< 30 lines), got {reasoner_lines}"
        );
    }

    #[test]
    fn system_prompt_emphasizes_brevity() {
        let prompt = build_tool_use_system_prompt(None, None, None, None);
        assert!(prompt.contains("concise") || prompt.contains("Minimize"));
    }

    // ── Complexity injection ──

    #[test]
    fn complex_gets_full_planning_protocol() {
        let prompt = build_tool_use_system_prompt_with_complexity(
            None,
            None,
            None,
            None,
            PromptComplexity::Complex,
            None,
        );
        assert!(
            prompt.contains("COMPLEX TASK"),
            "complex should get planning protocol"
        );
        assert!(
            prompt.contains("Step 1: Explore"),
            "should include explore step"
        );
        assert!(prompt.contains("Step 2: Plan"), "should include plan step");
        assert!(
            prompt.contains("Step 3: Execute"),
            "should include execute step"
        );
        assert!(
            prompt.contains("Anti-Patterns"),
            "should include anti-patterns"
        );
        assert!(
            prompt.contains("WORKING PROTOCOL"),
            "should have protocol in base"
        );
    }

    #[test]
    fn medium_gets_lightweight_guidance() {
        let prompt = build_tool_use_system_prompt_with_complexity(
            None,
            None,
            None,
            None,
            PromptComplexity::Medium,
            None,
        );
        assert!(
            prompt.contains("Task Guidance"),
            "medium should get guidance"
        );
        assert!(
            prompt.contains("Read the files"),
            "should include read-before-write"
        );
        assert!(
            prompt.contains("grep for all usages"),
            "should include impact tracing"
        );
        assert!(
            !prompt.contains("COMPLEX TASK"),
            "medium should NOT get full protocol"
        );
    }

    #[test]
    fn simple_gets_no_injection() {
        let prompt = build_tool_use_system_prompt_with_complexity(
            None,
            None,
            None,
            None,
            PromptComplexity::Simple,
            None,
        );
        assert!(
            prompt.contains("WORKING PROTOCOL"),
            "simple gets base protocol"
        );
        assert!(
            !prompt.contains("COMPLEX TASK"),
            "simple should NOT get complex protocol"
        );
        assert!(
            !prompt.contains("Task Guidance"),
            "simple should NOT get medium guidance"
        );
    }

    #[test]
    fn complex_includes_repo_map() {
        let repo_map = "- src/lib.rs (2048 bytes) score=100\n- src/main.rs (512 bytes) score=50";
        let prompt = build_tool_use_system_prompt_with_complexity(
            None,
            None,
            None,
            None,
            PromptComplexity::Complex,
            Some(repo_map),
        );
        assert!(
            prompt.contains("Project Files"),
            "complex should include project files"
        );
        assert!(
            prompt.contains("src/lib.rs"),
            "should include repo map entries"
        );
        assert!(
            prompt.contains("src/main.rs"),
            "should include all repo map entries"
        );
    }

    #[test]
    fn override_skips_complexity_injection() {
        let prompt = build_tool_use_system_prompt_with_complexity(
            None,
            Some("Custom prompt"),
            None,
            None,
            PromptComplexity::Complex,
            Some("repo map content"),
        );
        assert!(
            !prompt.contains("COMPLEX TASK"),
            "override should skip complexity"
        );
        assert!(
            !prompt.contains("Project Files"),
            "override should skip repo map"
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

    // ── T3.2: Model-aware prompt tests ──

    #[test]
    fn reasoner_gets_thinking_guidance() {
        let prompt = build_model_aware_system_prompt(
            None,
            None,
            None,
            None,
            PromptComplexity::Complex,
            None,
            "deepseek-reasoner",
        );
        assert!(
            prompt.contains("Reasoner"),
            "reasoner should get thinking guidance"
        );
        assert!(
            prompt.contains("extended thinking"),
            "should mention thinking capability"
        );
    }

    #[test]
    fn chat_complex_gets_prescriptive_guidance() {
        let prompt = build_model_aware_system_prompt(
            None,
            None,
            None,
            None,
            PromptComplexity::Complex,
            None,
            "deepseek-chat",
        );
        assert!(
            prompt.contains("Verification Protocol"),
            "chat on complex should get prescriptive guidance"
        );
        assert!(
            prompt.contains("EVERY file modification"),
            "should emphasize verification"
        );
    }

    #[test]
    fn chat_simple_gets_no_model_guidance() {
        let prompt = build_model_aware_system_prompt(
            None,
            None,
            None,
            None,
            PromptComplexity::Simple,
            None,
            "deepseek-chat",
        );
        assert!(
            !prompt.contains("Verification Protocol"),
            "simple should not get prescriptive guidance"
        );
        assert!(
            !prompt.contains("Reasoner"),
            "chat should not get reasoner guidance"
        );
    }

    #[test]
    fn override_skips_model_guidance() {
        let prompt = build_model_aware_system_prompt(
            None,
            Some("Custom override"),
            None,
            None,
            PromptComplexity::Complex,
            None,
            "deepseek-reasoner",
        );
        assert!(
            !prompt.contains("Reasoner"),
            "override should skip model guidance"
        );
    }

    // ── T3.3: Dual-prompt model-tier tests ──

    #[test]
    fn chat_prompt_has_action_bias() {
        assert!(
            CHAT_SYSTEM_PROMPT.contains("PRIME DIRECTIVE"),
            "chat should have prime directive"
        );
        assert!(
            CHAT_SYSTEM_PROMPT.contains("Just do it"),
            "chat should have action bias"
        );
        assert!(
            CHAT_SYSTEM_PROMPT.contains("DO NOT"),
            "chat should have explicit DO NOT section"
        );
    }

    #[test]
    fn reasoner_prompt_has_thinking_strategy() {
        assert!(
            REASONER_SYSTEM_PROMPT.contains("THINKING STRATEGY"),
            "reasoner should have thinking strategy"
        );
        assert!(
            REASONER_SYSTEM_PROMPT.contains("extended thinking"),
            "reasoner should mention thinking capability"
        );
        assert!(
            !REASONER_SYSTEM_PROMPT.contains("PRIME DIRECTIVE"),
            "reasoner should NOT have chat's prime directive"
        );
    }

    #[test]
    fn both_prompts_share_core_rules() {
        assert!(CHAT_SYSTEM_PROMPT.contains("NEVER fabricate"));
        assert!(REASONER_SYSTEM_PROMPT.contains("NEVER fabricate"));
        assert!(CHAT_SYSTEM_PROMPT.contains("Read files before editing"));
        assert!(REASONER_SYSTEM_PROMPT.contains("Read"));
    }

    #[test]
    fn model_aware_selects_correct_base() {
        let chat = build_model_aware_system_prompt(
            None,
            None,
            None,
            None,
            PromptComplexity::Simple,
            None,
            "deepseek-chat",
        );
        let reasoner = build_model_aware_system_prompt(
            None,
            None,
            None,
            None,
            PromptComplexity::Simple,
            None,
            "deepseek-reasoner",
        );
        assert!(
            chat.contains("PRIME DIRECTIVE"),
            "chat should use CHAT_SYSTEM_PROMPT"
        );
        assert!(
            reasoner.contains("THINKING STRATEGY"),
            "reasoner should use REASONER_SYSTEM_PROMPT"
        );
        assert!(
            !chat.contains("THINKING STRATEGY"),
            "chat should NOT have reasoner content"
        );
        assert!(
            !reasoner.contains("PRIME DIRECTIVE"),
            "reasoner should NOT have chat content"
        );
    }

    #[test]
    fn model_aware_complexity_injection_works_on_both_tiers() {
        let chat_complex = build_model_aware_system_prompt(
            None,
            None,
            None,
            None,
            PromptComplexity::Complex,
            None,
            "deepseek-chat",
        );
        let reasoner_complex = build_model_aware_system_prompt(
            None,
            None,
            None,
            None,
            PromptComplexity::Complex,
            None,
            "deepseek-reasoner",
        );
        assert!(
            chat_complex.contains("COMPLEX TASK"),
            "chat complex should get planning"
        );
        assert!(
            reasoner_complex.contains("COMPLEX TASK"),
            "reasoner complex should get planning"
        );
    }

    #[test]
    fn model_aware_preserves_memory_and_context() {
        let ctx = WorkspaceContext {
            cwd: "/project".to_string(),
            git_branch: Some("main".to_string()),
            os: "linux".to_string(),
        };
        let prompt = build_model_aware_system_prompt(
            Some("Use tabs."),
            None,
            Some("Be brief."),
            Some(&ctx),
            PromptComplexity::Simple,
            None,
            "deepseek-chat",
        );
        assert!(prompt.contains("Use tabs."));
        assert!(prompt.contains("Be brief."));
        assert!(prompt.contains("/project"));
        assert!(prompt.contains("main"));
    }
}
