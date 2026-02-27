//! Tool tier classification for dynamic tool loading.
//!
//! Instead of sending all 30+ tool definitions to the LLM on every API call,
//! tools are classified into tiers:
//!
//! - **Core**: Always included (~12 tools). These are the fundamental tools
//!   that the model needs for most tasks.
//! - **Contextual**: Included when project or prompt signals match. For
//!   example, `git_*` tools are only included in git repositories.
//! - **Extended**: Available on-demand via the `tool_search` meta-tool.
//!   The model discovers these when it needs capabilities beyond the core set.

use deepseek_core::{FunctionDefinition, ToolDefinition};
use serde_json::json;

/// Classification of a tool into a loading tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolTier {
    /// Always included in every API call.
    Core,
    /// Included when contextually relevant.
    Contextual,
    /// Available on-demand via `tool_search`.
    Extended,
}

/// Classify a tool by its API name into a tier.
pub fn tool_tier(api_name: &str) -> ToolTier {
    match api_name {
        // Core: fundamental tools always available
        "fs_read" | "fs_write" | "fs_edit" | "fs_glob" | "fs_grep" | "fs_list" | "bash_run"
        | "multi_edit" => ToolTier::Core,

        // Core: agent-level tools always available
        "user_question" | "spawn_task" | "task_output" | "task_stop" | "kill_shell"
        | "extended_thinking" | "think_deeply" => ToolTier::Core,

        // Contextual: git tools (included in git repos)
        "git_status" | "git_diff" | "git_show" => ToolTier::Contextual,

        // Contextual: diagnostics (included when build system detected)
        "diagnostics_check" => ToolTier::Contextual,

        // Contextual: web tools (included when prompt mentions URLs/docs)
        "web_fetch" | "web_search" => ToolTier::Contextual,

        // Contextual: index (included for large codebases)
        "index_query" => ToolTier::Contextual,

        // Contextual: task management (included when multi-step work detected)
        "task_create" | "task_update" | "task_get" | "task_list" => ToolTier::Contextual,

        // Everything else: notebooks, chrome, patches, plan mode, skills
        _ => ToolTier::Extended,
    }
}

/// Signals used to decide which contextual tools to include.
#[derive(Debug, Clone, Default)]
pub struct ToolContextSignals {
    /// Whether the workspace is a git repository.
    pub is_git_repo: bool,
    /// Whether .ipynb files exist in the workspace.
    pub has_notebooks: bool,
    /// Whether the prompt mentions URLs, documentation, or web resources.
    pub prompt_mentions_web: bool,
    /// Whether the prompt mentions browser, chrome, or UI testing.
    pub prompt_mentions_chrome: bool,
    /// Approximate number of files in the codebase.
    pub codebase_file_count: usize,
    /// Whether the prompt suggests multi-step work (planning, tasks, etc.).
    pub prompt_is_complex: bool,
}

/// Determine which contextual tool API names to include based on signals.
pub fn contextual_tool_names(signals: &ToolContextSignals) -> Vec<&'static str> {
    let mut tools = Vec::new();
    if signals.is_git_repo {
        tools.extend_from_slice(&["git_status", "git_diff", "git_show"]);
    }
    // Diagnostics are useful in most projects
    tools.push("diagnostics_check");
    if signals.prompt_mentions_web {
        tools.extend_from_slice(&["web_fetch", "web_search"]);
    }
    if signals.codebase_file_count > 500 {
        tools.push("index_query");
    }
    if signals.prompt_is_complex {
        tools.extend_from_slice(&["task_create", "task_update", "task_get", "task_list"]);
    }
    tools
}

/// Detect context signals from the prompt text and workspace.
pub fn detect_signals(prompt: &str, workspace: &std::path::Path) -> ToolContextSignals {
    let prompt_lower = prompt.to_lowercase();
    ToolContextSignals {
        is_git_repo: workspace.join(".git").exists(),
        has_notebooks: std::fs::read_dir(workspace)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .any(|e| e.path().extension().is_some_and(|ext| ext == "ipynb"))
            })
            .unwrap_or(false),
        prompt_mentions_web: prompt_lower.contains("http://")
            || prompt_lower.contains("https://")
            || prompt_lower.contains("url")
            || prompt_lower.contains("fetch")
            || prompt_lower.contains("documentation")
            || prompt_lower.contains("web search"),
        prompt_mentions_chrome: prompt_lower.contains("chrome")
            || prompt_lower.contains("browser")
            || prompt_lower.contains("screenshot")
            || prompt_lower.contains("ui test"),
        codebase_file_count: estimate_file_count(workspace),
        prompt_is_complex: prompt.len() > 300
            || prompt_lower.contains("refactor")
            || prompt_lower.contains("migrate")
            || prompt_lower.contains("restructure")
            || prompt_lower.contains("implement")
            || prompt_lower.contains("build a")
            || prompt_lower.contains("create a"),
    }
}

/// Quick file count estimate (shallow scan, capped at 2000).
fn estimate_file_count(workspace: &std::path::Path) -> usize {
    let Ok(entries) = std::fs::read_dir(workspace) else {
        return 0;
    };
    let mut count = 0;
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            count += 1;
        } else if path.is_dir() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str != ".git" && name_str != "target" && name_str != "node_modules" {
                count += std::fs::read_dir(&path)
                    .map(|e| e.count().min(200))
                    .unwrap_or(0);
            }
        }
        if count > 2000 {
            break;
        }
    }
    count
}

/// Filter tool definitions to include only core tools + matching contextual tools.
/// Extended tools are excluded (discoverable via `tool_search`).
pub fn tiered_tool_definitions(
    all_defs: Vec<ToolDefinition>,
    signals: &ToolContextSignals,
) -> (Vec<ToolDefinition>, Vec<ToolDefinition>) {
    let contextual_names = contextual_tool_names(signals);
    let mut active = Vec::new();
    let mut extended = Vec::new();

    for def in all_defs {
        let tier = tool_tier(&def.function.name);
        match tier {
            ToolTier::Core => active.push(def),
            ToolTier::Contextual => {
                if contextual_names.contains(&def.function.name.as_str()) {
                    active.push(def);
                } else {
                    extended.push(def);
                }
            }
            ToolTier::Extended => {
                // Notebooks included if detected
                if (signals.has_notebooks
                    && (def.function.name == "notebook_read"
                        || def.function.name == "notebook_edit"))
                    || (signals.prompt_mentions_chrome && def.function.name.starts_with("chrome_"))
                {
                    active.push(def);
                } else {
                    extended.push(def);
                }
            }
        }
    }

    (active, extended)
}

/// Definition for the `tool_search` meta-tool that discovers extended tools.
pub fn tool_search_definition() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "tool_search".to_string(),
            description: "Search for additional tools not in your current set. Use when you \
                          need capabilities like: notebooks (notebook_read, notebook_edit), \
                          browser automation (chrome_navigate, chrome_click, chrome_screenshot), \
                          patch management (patch_stage, patch_apply), skills, or plan mode. \
                          Returns matching tool names and descriptions that you can then use \
                          directly."
                .to_string(),
            strict: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What capability you need (e.g., 'browser automation', 'jupyter notebook', 'unified diff')"
                    }
                },
                "required": ["query"]
            }),
        },
    }
}

/// Search extended tools by query string. Returns matching definitions.
pub fn search_extended_tools(query: &str, extended_defs: &[ToolDefinition]) -> Vec<ToolDefinition> {
    let query_lower = query.to_lowercase();
    let keywords: Vec<&str> = query_lower.split_whitespace().collect();

    extended_defs
        .iter()
        .filter(|def| {
            let name = def.function.name.to_lowercase();
            let desc = def.function.description.to_lowercase();
            keywords
                .iter()
                .any(|kw| name.contains(kw) || desc.contains(kw))
        })
        .cloned()
        .collect()
}

/// Format tool search results for the LLM.
pub fn format_tool_search_results(matches: &[ToolDefinition]) -> String {
    if matches.is_empty() {
        return "No matching tools found. Available extended tools cover: notebooks, browser \
                automation, patch management, plan mode, and skills."
            .to_string();
    }
    let mut lines = vec![format!("Found {} tools:", matches.len())];
    for def in matches {
        lines.push(format!(
            "- `{}`: {}",
            def.function.name, def.function.description
        ));
    }
    lines.push("\nYou can now call these tools directly.".to_string());
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_definitions;

    #[test]
    fn core_tools_are_classified() {
        assert_eq!(tool_tier("fs_read"), ToolTier::Core);
        assert_eq!(tool_tier("bash_run"), ToolTier::Core);
        assert_eq!(tool_tier("multi_edit"), ToolTier::Core);
        assert_eq!(tool_tier("user_question"), ToolTier::Core);
        assert_eq!(tool_tier("spawn_task"), ToolTier::Core);
        assert_eq!(tool_tier("extended_thinking"), ToolTier::Core);
        assert_eq!(tool_tier("think_deeply"), ToolTier::Core); // alias
    }

    #[test]
    fn extended_thinking_tool_exists() {
        let defs = tool_definitions();
        let et = defs.iter().find(|t| t.function.name == "extended_thinking");
        assert!(
            et.is_some(),
            "extended_thinking tool should exist in definitions"
        );
        assert!(
            et.unwrap()
                .function
                .description
                .contains("chain-of-thought"),
            "description should mention chain-of-thought reasoning"
        );
    }

    #[test]
    fn think_deeply_alias_accepted() {
        // Both names should resolve to Core tier
        assert_eq!(tool_tier("extended_thinking"), ToolTier::Core);
        assert_eq!(tool_tier("think_deeply"), ToolTier::Core);
    }

    #[test]
    fn contextual_tools_are_classified() {
        assert_eq!(tool_tier("git_status"), ToolTier::Contextual);
        assert_eq!(tool_tier("web_fetch"), ToolTier::Contextual);
        assert_eq!(tool_tier("index_query"), ToolTier::Contextual);
        assert_eq!(tool_tier("diagnostics_check"), ToolTier::Contextual);
    }

    #[test]
    fn extended_tools_are_classified() {
        assert_eq!(tool_tier("notebook_read"), ToolTier::Extended);
        assert_eq!(tool_tier("chrome_navigate"), ToolTier::Extended);
        assert_eq!(tool_tier("patch_stage"), ToolTier::Extended);
        assert_eq!(tool_tier("enter_plan_mode"), ToolTier::Extended);
    }

    #[test]
    fn tiered_filtering_reduces_tool_count() {
        let all = tool_definitions();
        let total = all.len();
        let signals = ToolContextSignals {
            is_git_repo: true,
            ..Default::default()
        };
        let (active, extended) = tiered_tool_definitions(all, &signals);
        // Active should be significantly less than total
        assert!(
            active.len() < total,
            "active={} total={}",
            active.len(),
            total
        );
        // Extended should have the rest
        assert_eq!(active.len() + extended.len(), total);
        // Core tools should always be present
        assert!(active.iter().any(|t| t.function.name == "fs_read"));
        assert!(active.iter().any(|t| t.function.name == "bash_run"));
    }

    #[test]
    fn contextual_git_tools_included_for_git_repos() {
        let all = tool_definitions();
        let signals = ToolContextSignals {
            is_git_repo: true,
            ..Default::default()
        };
        let (active, _) = tiered_tool_definitions(all, &signals);
        assert!(active.iter().any(|t| t.function.name == "git_status"));
        assert!(active.iter().any(|t| t.function.name == "git_diff"));
    }

    #[test]
    fn contextual_git_tools_excluded_for_non_git() {
        let all = tool_definitions();
        let signals = ToolContextSignals::default();
        let (active, extended) = tiered_tool_definitions(all, &signals);
        assert!(!active.iter().any(|t| t.function.name == "git_status"));
        assert!(extended.iter().any(|t| t.function.name == "git_status"));
    }

    #[test]
    fn tool_search_finds_matching_tools() {
        let all = tool_definitions();
        let signals = ToolContextSignals::default();
        let (_, extended) = tiered_tool_definitions(all, &signals);
        let results = search_extended_tools("notebook", &extended);
        assert!(
            results.iter().any(|t| t.function.name == "notebook_read"),
            "should find notebook_read"
        );
    }

    #[test]
    fn tool_search_no_match() {
        let all = tool_definitions();
        let signals = ToolContextSignals::default();
        let (_, extended) = tiered_tool_definitions(all, &signals);
        let results = search_extended_tools("nonexistent_capability_xyz", &extended);
        assert!(results.is_empty());
    }

    #[test]
    fn tool_search_definition_is_valid() {
        let def = tool_search_definition();
        assert_eq!(def.function.name, "tool_search");
        assert!(!def.function.description.is_empty());
    }
}
