use codingbuddy_core::{ModelCapabilities, PreferredEditTool, ToolDefinition, ToolName, ToolTier};
use std::collections::HashSet;

pub(crate) fn shape_tool_surface(
    active: Vec<ToolDefinition>,
    discoverable: Vec<ToolDefinition>,
    capabilities: ModelCapabilities,
) -> (Vec<ToolDefinition>, Vec<ToolDefinition>) {
    let mut active = dedupe_tools(active);
    let mut discoverable = dedupe_tools(discoverable);

    prefer_edit_tool(
        &mut active,
        &mut discoverable,
        capabilities.preferred_edit_tool,
    );
    dedupe_cross_lists(&active, &mut discoverable);
    cap_active_tools(&mut active, &mut discoverable, capabilities);
    dedupe_cross_lists(&active, &mut discoverable);

    (active, discoverable)
}

fn dedupe_tools(tools: Vec<ToolDefinition>) -> Vec<ToolDefinition> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::with_capacity(tools.len());
    for def in tools {
        if seen.insert(def.function.name.clone()) {
            deduped.push(def);
        }
    }
    deduped
}

fn dedupe_cross_lists(active: &[ToolDefinition], discoverable: &mut Vec<ToolDefinition>) {
    let active_names = active
        .iter()
        .map(|def| def.function.name.as_str())
        .collect::<HashSet<_>>();
    discoverable.retain(|def| !active_names.contains(def.function.name.as_str()));
}

fn prefer_edit_tool(
    active: &mut Vec<ToolDefinition>,
    discoverable: &mut Vec<ToolDefinition>,
    preferred: PreferredEditTool,
) {
    let preferred_name = preferred_edit_tool_name(preferred);
    let edit_candidates = ["fs_edit", "multi_edit", "patch_direct"];

    if !active.iter().any(|def| def.function.name == preferred_name)
        && let Some(idx) = discoverable
            .iter()
            .position(|def| def.function.name == preferred_name)
    {
        active.push(discoverable.remove(idx));
    }

    for alt in edit_candidates {
        if alt == preferred_name {
            continue;
        }
        if let Some(idx) = active.iter().position(|def| def.function.name == alt) {
            discoverable.push(active.remove(idx));
        }
    }
}

fn preferred_edit_tool_name(tool: PreferredEditTool) -> &'static str {
    match tool {
        PreferredEditTool::FsEdit => "fs_edit",
        PreferredEditTool::MultiEdit => "multi_edit",
        PreferredEditTool::PatchDirect => "patch_direct",
    }
}

fn cap_active_tools(
    active: &mut Vec<ToolDefinition>,
    discoverable: &mut Vec<ToolDefinition>,
    capabilities: ModelCapabilities,
) {
    let reserve_tool_search = usize::from(!discoverable.is_empty());
    let active_limit = capabilities
        .max_safe_tool_count
        .saturating_sub(reserve_tool_search)
        .max(1);
    if active.len() <= active_limit {
        return;
    }

    let mut ranked = active
        .iter()
        .enumerate()
        .map(|(idx, def)| {
            (
                idx,
                tool_priority(def.function.name.as_str(), &capabilities),
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_by_key(|(_, priority)| priority.clone());
    let keep_indices = ranked
        .into_iter()
        .take(active_limit)
        .map(|(idx, _)| idx)
        .collect::<HashSet<_>>();

    let mut kept = Vec::with_capacity(active_limit);
    let mut overflow = Vec::new();
    for (idx, def) in std::mem::take(active).into_iter().enumerate() {
        if keep_indices.contains(&idx) {
            kept.push(def);
        } else {
            overflow.push(def);
        }
    }

    *active = kept;
    discoverable.extend(overflow);
}

fn tool_priority(name: &str, capabilities: &ModelCapabilities) -> (u8, u8, String) {
    let preferred_edit = preferred_edit_tool_name(capabilities.preferred_edit_tool);
    let primary = match name {
        "fs_read" => 0,
        "fs_list" => 1,
        "fs_glob" => 2,
        "fs_grep" => 3,
        value if value == preferred_edit => 4,
        "bash_run" => 5,
        "diagnostics_check" => 6,
        "git_status" => 7,
        "git_diff" => 8,
        "git_show" => 9,
        "batch" => 10,
        "task_get" => 11,
        "task_list" => 12,
        "task_output" => 13,
        "todo_read" => 14,
        "todo_write" => 15,
        "spawn_task" => 16,
        "task_create" => 17,
        "task_update" => 18,
        "task_stop" => 19,
        "enter_plan_mode" => 20,
        "exit_plan_mode" => 21,
        "user_question" => 22,
        "extended_thinking" => 23,
        _ => 100,
    };

    let tier = ToolName::from_api_name(name)
        .map(|tool| match tool.metadata().tier {
            ToolTier::Core => 0,
            ToolTier::Contextual => 1,
            ToolTier::Extended => 2,
        })
        .unwrap_or(3);
    (primary, tier, name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use codingbuddy_core::{FunctionDefinition, ProviderKind, model_capabilities};
    use serde_json::json;

    fn make_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.to_string(),
                description: format!("tool {name}"),
                parameters: json!({"type":"object"}),
                strict: None,
            },
        }
    }

    #[test]
    fn openai_surface_prefers_patch_direct() {
        let active = vec![
            make_tool("fs_read"),
            make_tool("fs_edit"),
            make_tool("multi_edit"),
            make_tool("bash_run"),
        ];
        let discoverable = vec![make_tool("patch_direct"), make_tool("web_search")];

        let (active, discoverable) = shape_tool_surface(
            active,
            discoverable,
            model_capabilities(ProviderKind::OpenAiCompatible, "gpt-4o-mini"),
        );
        let active_names = active
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect::<HashSet<_>>();
        let discoverable_names = discoverable
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect::<HashSet<_>>();

        assert!(active_names.contains("patch_direct"));
        assert!(!active_names.contains("fs_edit"));
        assert!(!active_names.contains("multi_edit"));
        assert!(discoverable_names.contains("fs_edit"));
        assert!(discoverable_names.contains("multi_edit"));
    }

    #[test]
    fn ollama_qwen_surface_prefers_multi_edit() {
        let active = vec![
            make_tool("fs_read"),
            make_tool("fs_edit"),
            make_tool("patch_direct"),
            make_tool("bash_run"),
        ];
        let discoverable = vec![make_tool("multi_edit"), make_tool("web_search")];

        let (active, discoverable) = shape_tool_surface(
            active,
            discoverable,
            model_capabilities(ProviderKind::Ollama, "qwen2.5-coder:7b"),
        );

        let active_names = active
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect::<HashSet<_>>();
        let discoverable_names = discoverable
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect::<HashSet<_>>();

        assert!(active_names.contains("multi_edit"));
        assert!(!active_names.contains("fs_edit"));
        assert!(!active_names.contains("patch_direct"));
        assert!(discoverable_names.contains("fs_edit"));
        assert!(discoverable_names.contains("patch_direct"));
    }

    #[test]
    fn active_tool_count_respects_model_limit() {
        let active = vec![
            make_tool("fs_read"),
            make_tool("fs_list"),
            make_tool("fs_glob"),
            make_tool("fs_grep"),
            make_tool("git_status"),
            make_tool("git_diff"),
            make_tool("git_show"),
            make_tool("diagnostics_check"),
            make_tool("fs_edit"),
            make_tool("bash_run"),
            make_tool("task_get"),
            make_tool("task_list"),
            make_tool("task_output"),
            make_tool("spawn_task"),
        ];
        let discoverable = vec![make_tool("chrome_navigate")];
        let caps = ModelCapabilities {
            provider: ProviderKind::Ollama,
            family: codingbuddy_core::ModelFamily::Qwen,
            supports_tool_calling: true,
            supports_tool_choice: true,
            supports_parallel_tool_calls: false,
            supports_reasoning_mode: false,
            supports_thinking_config: false,
            supports_streaming_tool_deltas: true,
            supports_fim: false,
            max_safe_tool_count: 6,
            preferred_edit_tool: PreferredEditTool::FsEdit,
        };

        let (active, discoverable) = shape_tool_surface(active, discoverable, caps);
        assert!(active.len() <= 5, "reserve one slot for tool_search");
        assert!(!discoverable.is_empty());
        assert!(active.iter().any(|tool| tool.function.name == "fs_edit"));
    }
}
