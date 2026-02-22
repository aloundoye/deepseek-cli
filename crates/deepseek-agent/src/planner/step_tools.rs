use crate::planner::memory::plan_goal_pattern;
use crate::*;

impl AgentEngine {
    pub(crate) fn calls_for_step(&self, step: &PlanStep, plan_goal: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();
        for tool in &step.tools {
            if let Some(call) = self.call_for_declared_tool(step, plan_goal, tool) {
                if calls
                    .iter()
                    .any(|existing: &ToolCall| existing.name == call.name)
                {
                    continue;
                }
                calls.push(call);
            }
            if calls.len() >= 3 {
                break;
            }
        }
        if calls.is_empty() {
            calls.push(self.call_for_step(step, plan_goal));
        }
        calls
    }

    pub(crate) fn call_for_declared_tool(
        &self,
        step: &PlanStep,
        plan_goal: &str,
        tool: &str,
    ) -> Option<ToolCall> {
        let (tool_name, suffix) = parse_declared_tool(tool);
        let suffix = suffix.as_deref();
        let primary_file = step.files.first().cloned();
        let search_pattern = plan_goal_pattern(plan_goal);
        match tool_name.as_str() {
            "index.query" => Some(ToolCall {
                name: "index.query".to_string(),
                args: json!({"q": plan_goal, "top_k": 10}),
                requires_approval: false,
            }),
            "fs.grep" => Some(ToolCall {
                name: "fs.grep".to_string(),
                args: json!({
                    "pattern": suffix.filter(|s| !s.is_empty()).unwrap_or(&search_pattern),
                    "glob": "**/*",
                    "limit": 50,
                    "respectGitignore": true
                }),
                requires_approval: false,
            }),
            "fs.read" => {
                let path = suffix
                    .filter(|s| !s.is_empty())
                    .map(ToString::to_string)
                    .or(primary_file)?;
                Some(ToolCall {
                    name: "fs.read".to_string(),
                    args: json!({"path": path}),
                    requires_approval: false,
                })
            }
            "fs.search_rg" => Some(ToolCall {
                name: "fs.search_rg".to_string(),
                args: json!({"query": plan_goal, "limit": 20}),
                requires_approval: false,
            }),
            "fs.list" => Some(ToolCall {
                name: "fs.list".to_string(),
                args: json!({"dir": "."}),
                requires_approval: false,
            }),
            "git.status" => Some(ToolCall {
                name: "git.status".to_string(),
                args: json!({}),
                requires_approval: false,
            }),
            "git.diff" => Some(ToolCall {
                name: "git.diff".to_string(),
                args: json!({}),
                requires_approval: false,
            }),
            "git.show" => Some(ToolCall {
                name: "git.show".to_string(),
                args: json!({"spec": suffix.filter(|s| !s.is_empty()).unwrap_or("HEAD")}),
                requires_approval: false,
            }),
            "bash.run" => Some(ToolCall {
                name: "bash.run".to_string(),
                args: json!({"cmd": suffix.filter(|s| !s.is_empty()).unwrap_or("cargo test --workspace")}),
                requires_approval: true,
            }),
            // Retain legacy behavior so existing patch/apply workflows continue to function.
            "patch.stage" => Some(ToolCall {
                name: "patch.stage".to_string(),
                args: json!({
                    "unified_diff": format!(
                            "diff --git a/.deepseek/notes.txt b/.deepseek/notes.txt\nnew file mode 100644\nindex 0000000..2b9d865\n--- /dev/null\n+++ b/.deepseek/notes.txt\n@@ -0,0 +1 @@\n+{}\n",
                        plan_goal.replace('\n', " ")
                    ),
                    "base": ""
                }),
                requires_approval: false,
            }),
            "fs.edit" => {
                let path = if let Some(path) = primary_file {
                    path
                } else if step.intent == "docs" {
                    "README.md".to_string()
                } else {
                    return None;
                };
                Some(ToolCall {
                    name: "fs.edit".to_string(),
                    args: json!({
                        "path": path,
                        "search": "## Verification",
                        "replace": "## Verification\n- Ensure DeepSeek API key is configured for strict-online mode.\n",
                        "all": false
                    }),
                    requires_approval: false,
                })
            }
            "fs.write" => {
                let path = suffix
                    .filter(|s| !s.is_empty())
                    .map(ToString::to_string)
                    .or(primary_file)
                    .unwrap_or_else(|| ".deepseek/notes.txt".to_string());
                Some(ToolCall {
                    name: "fs.write".to_string(),
                    args: json!({
                        "path": path,
                        "content": format!("Plan goal: {}\nStep: {}\nIntent: {}\n", plan_goal, step.title, step.intent)
                    }),
                    requires_approval: false,
                })
            }
            _ => None,
        }
    }

    pub(crate) fn call_for_step(&self, step: &PlanStep, plan_goal: &str) -> ToolCall {
        match step.intent.as_str() {
            "search" => ToolCall {
                name: "fs.search_rg".to_string(),
                args: json!({"query": plan_goal, "limit": 10}),
                requires_approval: false,
            },
            "git" => ToolCall {
                name: "git.status".to_string(),
                args: json!({}),
                requires_approval: false,
            },
            "edit" => ToolCall {
                name: "patch.stage".to_string(),
                args: json!({
                    "unified_diff": format!(
                        "diff --git a/.deepseek/notes.txt b/.deepseek/notes.txt\nnew file mode 100644\nindex 0000000..2b9d865\n--- /dev/null\n+++ b/.deepseek/notes.txt\n@@ -0,0 +1 @@\n+{}\n",
                        plan_goal.replace('\n', " ")
                    ),
                    "base": ""
                }),
                requires_approval: false,
            },
            "docs" => ToolCall {
                name: "fs.edit".to_string(),
                args: json!({
                    "path": "README.md",
                    "search": "## Verification",
                    "replace": "## Verification\n- Ensure DeepSeek API key is configured for strict-online mode.\n",
                    "all": false
                }),
                requires_approval: false,
            },
            "recover" => ToolCall {
                name: "fs.grep".to_string(),
                args: json!({"pattern":"error|failed|panic","glob":"**/*","limit":25}),
                requires_approval: false,
            },
            "verify" => ToolCall {
                name: "bash.run".to_string(),
                args: json!({"cmd": "cargo test --workspace"}),
                requires_approval: true,
            },
            _ => ToolCall {
                name: "fs.list".to_string(),
                args: json!({"dir": "."}),
                requires_approval: false,
            },
        }
    }
}
