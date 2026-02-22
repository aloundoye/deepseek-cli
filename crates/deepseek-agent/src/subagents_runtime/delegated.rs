use crate::planner::memory::plan_goal_pattern;
use crate::*;

pub(crate) fn subagent_readonly_fallback_call(
    task: &SubagentTask,
    root_goal: &str,
    targets: &[String],
) -> ToolCall {
    if let Some(path) = targets.first() {
        return ToolCall {
            name: "fs.read".to_string(),
            args: json!({"path": path}),
            requires_approval: false,
        };
    }
    let pattern = if task.goal.trim().is_empty() {
        plan_goal_pattern(root_goal)
    } else {
        plan_goal_pattern(&task.goal)
    };
    ToolCall {
        name: "fs.grep".to_string(),
        args: json!({
            "pattern": pattern,
            "glob": "**/*",
            "limit": 20,
            "respectGitignore": true
        }),
        requires_approval: false,
    }
}

pub(crate) fn subagent_delegated_calls(
    task: &SubagentTask,
    root_goal: &str,
    targets: &[String],
) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    if let Some(primary) = subagent_probe_call(task, root_goal) {
        calls.push(primary);
    }
    if matches!(task.role, SubagentRole::Task) {
        let note_path = format!(".deepseek/subagents/{}.md", task.run_id);
        let target_render = if targets.is_empty() {
            "none".to_string()
        } else {
            targets.join(", ")
        };
        calls.push(ToolCall {
            name: "fs.write".to_string(),
            args: json!({
                "path": note_path,
                "content": format!(
                    "subagent={}\nrole={:?}\nteam={}\nmain_goal={}\ntargets={}\n",
                    task.name, task.role, task.team, root_goal, target_render
                )
            }),
            requires_approval: true,
        });
    }
    calls.truncate(2);
    calls
}

pub(crate) fn is_parallel_safe_tool(name: &str) -> bool {
    matches!(
        name,
        "fs.list"
            | "fs.read"
            | "fs.grep"
            | "fs.glob"
            | "fs.search_rg"
            | "git.status"
            | "git.diff"
            | "git.show"
            | "index.query"
    )
}

pub(crate) fn subagent_retry_call(
    task: &SubagentTask,
    previous_call: &ToolCall,
    root_goal: &str,
    targets: &[String],
) -> Option<ToolCall> {
    if previous_call.name == "fs.write" || previous_call.name == "patch.apply" {
        return Some(subagent_readonly_fallback_call(task, root_goal, targets));
    }
    if previous_call.name == "bash.run" {
        return Some(ToolCall {
            name: "fs.grep".to_string(),
            args: json!({
                "pattern": plan_goal_pattern(root_goal),
                "glob": "**/*",
                "limit": 25,
                "respectGitignore": true
            }),
            requires_approval: false,
        });
    }
    None
}

pub(crate) fn subagent_probe_call(task: &SubagentTask, root_goal: &str) -> Option<ToolCall> {
    match task.role {
        SubagentRole::Explore => Some(ToolCall {
            name: "index.query".to_string(),
            args: json!({"q": root_goal, "top_k": 8}),
            requires_approval: false,
        }),
        SubagentRole::Plan => Some(ToolCall {
            name: "fs.grep".to_string(),
            args: json!({
                "pattern": plan_goal_pattern(root_goal),
                "glob": "**/*",
                "limit": 20,
                "respectGitignore": true
            }),
            requires_approval: false,
        }),
        SubagentRole::Task => Some(ToolCall {
            name: "git.status".to_string(),
            args: json!({}),
            requires_approval: false,
        }),
        SubagentRole::Custom(_) => None,
    }
}

pub(crate) fn should_parallel_execute_calls(proposals: &[deepseek_core::ToolProposal]) -> bool {
    proposals.len() > 1
        && proposals
            .iter()
            .all(|proposal| is_parallel_safe_tool(&proposal.call.name))
}

pub(crate) fn run_subagent_delegated_tools(
    tool_host: &LocalToolHost,
    task: &SubagentTask,
    root_goal: &str,
    targets: &[String],
) -> Option<String> {
    let calls = subagent_delegated_calls(task, root_goal, targets);
    if calls.is_empty() {
        return None;
    }
    let mut lines = Vec::new();
    for call in calls {
        let mut active_call = call;
        let mut attempt = 1usize;
        loop {
            let proposal = tool_host.propose(active_call.clone());
            if !proposal.approved {
                lines.push(format!(
                    "{} {:?} delegated {} blocked (attempt={})",
                    task.team, task.role, proposal.call.name, attempt
                ));
                if let Some(retry) = subagent_retry_call(task, &proposal.call, root_goal, targets) {
                    active_call = retry;
                    attempt += 1;
                    if attempt <= 2 {
                        continue;
                    }
                }
                break;
            }
            let result = tool_host.execute(ApprovedToolCall {
                invocation_id: proposal.invocation_id,
                call: proposal.call.clone(),
            });
            let output = truncate_probe_text(
                serde_json::to_string(&result.output)
                    .unwrap_or_else(|_| "<unserializable delegated output>".to_string()),
            );
            if result.success {
                lines.push(format!(
                    "{} {:?} delegated {} => {}",
                    task.team, task.role, proposal.call.name, output
                ));
                break;
            }
            lines.push(format!(
                "{} {:?} delegated {} failed (attempt={}) => {}",
                task.team, task.role, proposal.call.name, attempt, output
            ));
            if let Some(retry) = subagent_retry_call(task, &proposal.call, root_goal, targets) {
                active_call = retry;
                attempt += 1;
                if attempt <= 2 {
                    continue;
                }
            }
            break;
        }
    }
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

pub(crate) fn truncate_probe_text(text: String) -> String {
    const MAX_CHARS: usize = 480;
    if text.chars().count() <= MAX_CHARS {
        return text;
    }
    let head = text.chars().take(MAX_CHARS).collect::<String>();
    format!("{head}...")
}
