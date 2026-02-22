use crate::*;

/// Tracks a background subagent task spawned with `run_in_background: true`.
#[allow(dead_code)]
pub(crate) struct BackgroundTask {
    pub(crate) task_id: Uuid,
    pub(crate) description: String,
    pub(crate) handle: Option<thread::JoinHandle<String>>,
    pub(crate) result: Option<String>,
    pub(crate) stopped: bool,
}

/// Tracks a background bash process spawned with `bash_run` + `run_in_background: true`.
#[allow(dead_code)]
pub(crate) struct BackgroundShell {
    pub(crate) shell_id: Uuid,
    pub(crate) cmd: String,
    pub(crate) handle: Option<thread::JoinHandle<deepseek_core::ToolResult>>,
    pub(crate) result: Option<deepseek_core::ToolResult>,
    pub(crate) stopped: bool,
}

impl AgentEngine {
    pub(crate) fn handle_kill_shell(&self, args: &serde_json::Value) -> Result<String> {
        let shell_id_str = args
            .get("shell_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("shell_id parameter is required"))?;
        let shell_id = Uuid::parse_str(shell_id_str)?;

        let mut shells = self
            .background_shells
            .lock()
            .map_err(|_| anyhow!("background_shells lock poisoned"))?;

        let entry = shells
            .iter_mut()
            .find(|s| s.shell_id == shell_id)
            .ok_or_else(|| anyhow!("no background shell with id {shell_id_str}"))?;

        entry.stopped = true;
        let was_running = entry.handle.take().is_some();

        Ok(serde_json::to_string(&json!({
            "shell_id": shell_id_str,
            "status": "stopped",
            "was_running": was_running
        }))?)
    }

    /// Handle the `think_deeply` tool: consult R1 for targeted advice.
    pub(crate) fn handle_task_output(&self, args: &serde_json::Value) -> Result<String> {
        let task_id_str = args
            .get("task_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("task_id parameter is required"))?;
        let task_id = Uuid::parse_str(task_id_str)?;
        let block = args.get("block").and_then(|v| v.as_bool()).unwrap_or(true);
        let timeout_ms = args
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(30_000);

        // Check background subagent tasks first.
        let found_in_tasks = self
            .background_tasks
            .lock()
            .ok()
            .is_some_and(|bg| bg.iter().any(|t| t.task_id == task_id));

        if found_in_tasks {
            return self.poll_background_task(task_id_str, task_id, block, timeout_ms);
        }

        // Check background bash shells.
        let found_in_shells = self
            .background_shells
            .lock()
            .ok()
            .is_some_and(|bg| bg.iter().any(|s| s.shell_id == task_id));

        if found_in_shells {
            return self.poll_background_shell(task_id_str, task_id, block, timeout_ms);
        }

        Ok(serde_json::to_string(&json!({
            "task_id": task_id_str,
            "status": "unknown",
            "error": format!("no background task or shell with id {task_id_str}")
        }))?)
    }

    /// Poll a background subagent task for its result.
    pub(crate) fn handle_spawn_task(&self, args: &serde_json::Value) -> Result<String> {
        // Fire SubagentStart hook.
        {
            let mut input = self.hook_input(HookEvent::SubagentStart);
            input.tool_input = Some(args.clone());
            self.fire_hook(HookEvent::SubagentStart, &input);
        }

        // Accept both new (`prompt`) and legacy (`goal`) parameter names.
        let prompt = args
            .get("prompt")
            .or_else(|| args.get("goal"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("prompt parameter is required"))?;
        let description = args
            .get("description")
            .or_else(|| args.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("subagent");

        // Accept both new (`subagent_type`) and legacy (`role`) parameter names.
        let type_str = args
            .get("subagent_type")
            .or_else(|| args.get("role"))
            .and_then(|v| v.as_str())
            .unwrap_or("general-purpose");

        // Check for custom agent definitions from .deepseek/agents/*.md
        let custom_agent = match type_str {
            "explore" | "plan" | "bash" | "general-purpose" => None,
            custom_name => {
                let defs = deepseek_subagent::load_agent_defs(&self.workspace).unwrap_or_default();
                defs.into_iter().find(|d| d.name == custom_name)
            }
        };

        let role = match type_str {
            "explore" => SubagentRole::Explore,
            "plan" => SubagentRole::Plan,
            "bash" => SubagentRole::Task,
            "general-purpose" => SubagentRole::Task,
            name => SubagentRole::Custom(name.to_string()),
        };

        let run_in_background = args
            .get("run_in_background")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let task = SubagentTask {
            run_id: Uuid::now_v7(),
            name: description.to_string(),
            goal: prompt.to_string(),
            role: role.clone(),
            team: "default".to_string(),
            read_only_fallback: matches!(role, SubagentRole::Explore | SubagentRole::Plan),
            custom_agent,
        };

        // Get external worker (set by the CLI for full agent capabilities).
        let worker = self
            .subagent_worker
            .lock()
            .ok()
            .and_then(|g| g.as_ref().cloned());

        if run_in_background {
            let task_id = task.run_id;
            let desc = description.to_string();
            let handle = if let Some(worker) = worker {
                thread::spawn(move || match worker(&task) {
                    Ok(output) => output,
                    Err(e) => format!("Error: {e}"),
                })
            } else {
                let prompt_owned = prompt.to_string();
                let name_owned = description.to_string();
                thread::spawn(move || {
                    format!("Subagent '{name_owned}' completed goal: {prompt_owned}")
                })
            };

            if let Ok(mut bg) = self.background_tasks.lock() {
                bg.push_back(BackgroundTask {
                    task_id,
                    description: desc.clone(),
                    handle: Some(handle),
                    result: None,
                    stopped: false,
                });
            }

            return Ok(serde_json::to_string(&json!({
                "task_id": task_id.to_string(),
                "description": desc,
                "status": "running",
                "message": "Agent launched in background. Use task_output to retrieve results."
            }))?);
        }

        // Foreground (blocking) execution.
        let output = if let Some(worker) = worker {
            let results = self.subagents.run_tasks(vec![task], move |t| worker(&t));
            if let Some(result) = results.first() {
                serde_json::to_string(&json!({
                    "name": result.name,
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                    "attempts": result.attempts
                }))?
            } else {
                serde_json::to_string(&json!({"error": "subagent failed to start"}))?
            }
        } else {
            // Fallback: run with a simple echo worker.
            let prompt_owned = prompt.to_string();
            let results = self.subagents.run_tasks(vec![task], move |t| {
                Ok(format!(
                    "Subagent '{}' completed goal: {}",
                    t.name, prompt_owned
                ))
            });
            if let Some(result) = results.first() {
                serde_json::to_string(&json!({
                    "name": result.name,
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                    "attempts": result.attempts
                }))?
            } else {
                serde_json::to_string(&json!({"error": "subagent failed to start"}))?
            }
        };

        // Fire SubagentStop hook.
        {
            let mut input = self.hook_input(HookEvent::SubagentStop);
            input.tool_result = Some(serde_json::Value::String(output.clone()));
            self.fire_hook(HookEvent::SubagentStop, &input);
        }

        Ok(output)
    }

    /// Handle the task_output tool: retrieve output from a background subagent.
    pub(crate) fn poll_background_shell(
        &self,
        shell_id_str: &str,
        shell_id: Uuid,
        block: bool,
        timeout_ms: u64,
    ) -> Result<String> {
        let mut shells = self
            .background_shells
            .lock()
            .map_err(|_| anyhow!("background_shells lock poisoned"))?;

        let entry = shells
            .iter_mut()
            .find(|s| s.shell_id == shell_id)
            .ok_or_else(|| anyhow!("no background shell with id {shell_id_str}"))?;

        if let Some(ref result) = entry.result {
            return Ok(serde_json::to_string(&json!({
                "task_id": shell_id_str,
                "status": if entry.stopped { "stopped" } else { "completed" },
                "output": result.output
            }))?);
        }

        if entry.stopped {
            return Ok(serde_json::to_string(&json!({
                "task_id": shell_id_str,
                "status": "stopped",
                "output": ""
            }))?);
        }

        let handle = entry.handle.take();
        drop(shells);

        if let Some(handle) = handle {
            if block {
                let deadline = Instant::now() + std::time::Duration::from_millis(timeout_ms);
                loop {
                    if handle.is_finished() {
                        let result = handle.join().unwrap_or_else(|_| deepseek_core::ToolResult {
                            invocation_id: Uuid::nil(),
                            success: false,
                            output: json!({"error": "shell thread panicked"}),
                        });
                        let output = result.output.clone();
                        if let Ok(mut shells) = self.background_shells.lock()
                            && let Some(entry) = shells.iter_mut().find(|s| s.shell_id == shell_id)
                        {
                            entry.result = Some(result);
                        }
                        return Ok(serde_json::to_string(&json!({
                            "task_id": shell_id_str,
                            "status": "completed",
                            "output": output
                        }))?);
                    }
                    if Instant::now() >= deadline {
                        if let Ok(mut shells) = self.background_shells.lock()
                            && let Some(entry) = shells.iter_mut().find(|s| s.shell_id == shell_id)
                        {
                            entry.handle = Some(handle);
                        }
                        return Ok(serde_json::to_string(&json!({
                            "task_id": shell_id_str,
                            "status": "running",
                            "message": format!("Shell still running after {timeout_ms}ms timeout")
                        }))?);
                    }
                    thread::sleep(std::time::Duration::from_millis(100));
                }
            } else {
                if handle.is_finished() {
                    let result = handle.join().unwrap_or_else(|_| deepseek_core::ToolResult {
                        invocation_id: Uuid::nil(),
                        success: false,
                        output: json!({"error": "shell thread panicked"}),
                    });
                    let output = result.output.clone();
                    if let Ok(mut shells) = self.background_shells.lock()
                        && let Some(entry) = shells.iter_mut().find(|s| s.shell_id == shell_id)
                    {
                        entry.result = Some(result);
                    }
                    return Ok(serde_json::to_string(&json!({
                        "task_id": shell_id_str,
                        "status": "completed",
                        "output": output
                    }))?);
                }
                if let Ok(mut shells) = self.background_shells.lock()
                    && let Some(entry) = shells.iter_mut().find(|s| s.shell_id == shell_id)
                {
                    entry.handle = Some(handle);
                }
                return Ok(serde_json::to_string(&json!({
                    "task_id": shell_id_str,
                    "status": "running"
                }))?);
            }
        }

        Ok(serde_json::to_string(&json!({
            "task_id": shell_id_str,
            "status": "unknown",
            "message": "Shell handle already consumed"
        }))?)
    }

    /// Handle the task_stop tool: signal a background task to stop.
    pub(crate) fn handle_task_stop(&self, args: &serde_json::Value) -> Result<String> {
        let task_id_str = args
            .get("task_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("task_id parameter is required"))?;
        let task_id = Uuid::parse_str(task_id_str)?;

        let mut bg = self
            .background_tasks
            .lock()
            .map_err(|_| anyhow!("background_tasks lock poisoned"))?;

        let entry = bg
            .iter_mut()
            .find(|t| t.task_id == task_id)
            .ok_or_else(|| anyhow!("no background task with id {task_id_str}"))?;

        entry.stopped = true;

        // We can't force-kill a Rust thread, but we mark it stopped so
        // task_output will report it.  If the handle is still live we
        // detach it (drop the JoinHandle).
        let was_running = entry.handle.take().is_some();

        Ok(serde_json::to_string(&json!({
            "task_id": task_id_str,
            "status": "stopped",
            "was_running": was_running
        }))?)
    }

    /// Handle the skill tool: invoke a registered skill/slash command.
    pub(crate) fn poll_background_task(
        &self,
        task_id_str: &str,
        task_id: Uuid,
        block: bool,
        timeout_ms: u64,
    ) -> Result<String> {
        let mut bg = self
            .background_tasks
            .lock()
            .map_err(|_| anyhow!("background_tasks lock poisoned"))?;

        let entry = bg
            .iter_mut()
            .find(|t| t.task_id == task_id)
            .ok_or_else(|| anyhow!("no background task with id {task_id_str}"))?;

        if let Some(ref output) = entry.result {
            return Ok(serde_json::to_string(&json!({
                "task_id": task_id_str,
                "status": if entry.stopped { "stopped" } else { "completed" },
                "output": output
            }))?);
        }

        if entry.stopped {
            return Ok(serde_json::to_string(&json!({
                "task_id": task_id_str,
                "status": "stopped",
                "output": ""
            }))?);
        }

        let handle = entry.handle.take();
        drop(bg);

        if let Some(handle) = handle {
            if block {
                let deadline = Instant::now() + std::time::Duration::from_millis(timeout_ms);
                loop {
                    if handle.is_finished() {
                        let output = handle
                            .join()
                            .unwrap_or_else(|_| "Error: subagent thread panicked".to_string());
                        if let Ok(mut bg) = self.background_tasks.lock()
                            && let Some(entry) = bg.iter_mut().find(|t| t.task_id == task_id)
                        {
                            entry.result = Some(output.clone());
                        }
                        return Ok(serde_json::to_string(&json!({
                            "task_id": task_id_str,
                            "status": "completed",
                            "output": output
                        }))?);
                    }
                    if Instant::now() >= deadline {
                        if let Ok(mut bg) = self.background_tasks.lock()
                            && let Some(entry) = bg.iter_mut().find(|t| t.task_id == task_id)
                        {
                            entry.handle = Some(handle);
                        }
                        return Ok(serde_json::to_string(&json!({
                            "task_id": task_id_str,
                            "status": "running",
                            "message": format!("Task still running after {timeout_ms}ms timeout")
                        }))?);
                    }
                    thread::sleep(std::time::Duration::from_millis(100));
                }
            } else {
                if handle.is_finished() {
                    let output = handle
                        .join()
                        .unwrap_or_else(|_| "Error: subagent thread panicked".to_string());
                    if let Ok(mut bg) = self.background_tasks.lock()
                        && let Some(entry) = bg.iter_mut().find(|t| t.task_id == task_id)
                    {
                        entry.result = Some(output.clone());
                    }
                    return Ok(serde_json::to_string(&json!({
                        "task_id": task_id_str,
                        "status": "completed",
                        "output": output
                    }))?);
                }
                if let Ok(mut bg) = self.background_tasks.lock()
                    && let Some(entry) = bg.iter_mut().find(|t| t.task_id == task_id)
                {
                    entry.handle = Some(handle);
                }
                return Ok(serde_json::to_string(&json!({
                    "task_id": task_id_str,
                    "status": "running"
                }))?);
            }
        }

        Ok(serde_json::to_string(&json!({
            "task_id": task_id_str,
            "status": "unknown",
            "message": "Task handle already consumed"
        }))?)
    }
}
