use crate::*;

impl AgentEngine {
    pub(crate) fn handle_task_list(&self, session: &Session) -> Result<String> {
        let tasks = self.store.list_tasks(Some(session.session_id))?;
        let items: Vec<serde_json::Value> = tasks
            .iter()
            .map(|t| {
                serde_json::json!({
                    "id": t.task_id.to_string(),
                    "subject": t.title,
                    "status": t.status,
                    "priority": t.priority,
                })
            })
            .collect();
        Ok(serde_json::to_string(
            &serde_json::json!({ "tasks": items }),
        )?)
    }

    /// Handle the spawn_task tool: spawn a subagent to work on a subtask.
    ///
    /// Supports Claude Code-compatible parameters:
    /// - `description`: short label (3-5 words)
    /// - `prompt`: the task for the agent
    /// - `subagent_type`: explore | plan | bash | general-purpose
    /// - `model`: optional model override
    /// - `max_turns`: optional turn limit
    /// - `run_in_background`: if true, returns immediately with a task_id
    /// - `resume`: optional previous agent ID to continue from
    pub(crate) fn handle_user_question(&self, args: &serde_json::Value) -> Result<String> {
        let question = args
            .get("question")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("question parameter is required"))?;
        let options: Vec<String> = args
            .get("options")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let user_q = deepseek_core::UserQuestion {
            question: question.to_string(),
            options: options.clone(),
        };

        // Try external handler first (TUI mode)
        if let Ok(guard) = self.user_question_handler.lock()
            && let Some(handler) = guard.as_ref()
        {
            if let Some(answer) = handler(user_q) {
                return Ok(serde_json::to_string(&serde_json::json!({
                    "answer": answer
                }))?);
            }
            return Ok(serde_json::to_string(
                &serde_json::json!({"answer": null, "cancelled": true}),
            )?);
        }

        // Fallback: blocking stdin
        let mut stdout = std::io::stdout();
        let stdin = std::io::stdin();
        if !stdin.is_terminal() || !stdout.is_terminal() {
            return Ok(serde_json::to_string(
                &serde_json::json!({"error": "user_question not available in non-interactive mode"}),
            )?);
        }

        writeln!(stdout, "\n{question}")?;
        if !options.is_empty() {
            for (i, opt) in options.iter().enumerate() {
                writeln!(stdout, "  {}: {opt}", i + 1)?;
            }
        }
        write!(stdout, "> ")?;
        stdout.flush()?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let answer = input.trim().to_string();
        Ok(serde_json::to_string(
            &serde_json::json!({"answer": answer}),
        )?)
    }

    /// Handle the task_create tool: create a task in the store.
    pub(crate) fn handle_task_create(
        &self,
        args: &serde_json::Value,
        session: &Session,
    ) -> Result<String> {
        let subject = args
            .get("subject")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("subject parameter is required"))?;
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let priority = args.get("priority").and_then(|v| v.as_u64()).unwrap_or(1) as u32;

        let task_id = Uuid::now_v7();
        let now = chrono::Utc::now().to_rfc3339();
        let title = if description.is_empty() {
            subject.to_string()
        } else {
            format!("{subject}: {description}")
        };

        let record = TaskQueueRecord {
            task_id,
            session_id: session.session_id,
            title,
            priority,
            status: "pending".to_string(),
            outcome: None,
            artifact_path: None,
            created_at: now.clone(),
            updated_at: now,
        };
        self.store.insert_task(&record)?;

        self.emit(
            session.session_id,
            EventKind::TaskCreatedV1 {
                task_id,
                title: subject.to_string(),
                priority,
            },
        )?;

        Ok(serde_json::to_string(&serde_json::json!({
            "task_id": task_id.to_string(),
            "status": "pending",
            "subject": subject
        }))?)
    }

    /// Handle the task_update tool: update a task's status.
    pub(crate) fn execute_agent_tool(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
        session: &Session,
    ) -> Result<String> {
        // Notify TUI of tool execution
        let arg_summary = summarize_tool_args(tool_name, args);
        if let Ok(mut cb_guard) = self.stream_callback.lock()
            && let Some(ref mut cb) = *cb_guard
        {
            cb(StreamChunk::ToolCallStart {
                tool_name: tool_name.to_string(),
                args_summary: arg_summary.clone(),
            });
        }

        let tool_start = Instant::now();
        let result = match tool_name {
            "user_question" => self.handle_user_question(args),
            "task_create" => self.handle_task_create(args, session),
            "task_update" => self.handle_task_update(args),
            "task_get" => self.handle_task_get(args),
            "task_list" => self.handle_task_list(session),
            "spawn_task" => self.handle_spawn_task(args),
            "task_output" => self.handle_task_output(args),
            "task_stop" => self.handle_task_stop(args),
            "skill" => self.handle_skill(args),
            "kill_shell" => self.handle_kill_shell(args),
            "think_deeply" => self.handle_think_deeply(args),
            // Plan mode tools are handled inline in chat loop, not here
            "enter_plan_mode" | "exit_plan_mode" => Ok("Plan mode transition handled.".to_string()),
            _ => Err(anyhow!("unknown agent tool: {tool_name}")),
        };
        let tool_elapsed = tool_start.elapsed();

        let (status_label, output) = match result {
            Ok(output) => ("ok", output),
            Err(e) => ("error", format!("Error: {e}")),
        };

        // Notify TUI of completion
        if let Ok(mut cb_guard) = self.stream_callback.lock()
            && let Some(ref mut cb) = *cb_guard
        {
            let preview_line = output
                .lines()
                .next()
                .unwrap_or("")
                .chars()
                .take(120)
                .collect::<String>();
            cb(StreamChunk::ToolCallEnd {
                tool_name: tool_name.to_string(),
                duration_ms: tool_elapsed.as_millis() as u64,
                success: status_label == "ok",
                summary: preview_line,
            });
        }

        Ok(output)
    }

    /// Handle the user_question tool: ask the user a question and return their answer.
    pub(crate) fn handle_task_update(&self, args: &serde_json::Value) -> Result<String> {
        let task_id_str = args
            .get("task_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("task_id parameter is required"))?;
        let task_id = Uuid::parse_str(task_id_str)?;
        let status = args
            .get("status")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("status parameter is required"))?;
        let outcome = args.get("outcome").and_then(|v| v.as_str());

        self.store.update_task_status(task_id, status, outcome)?;

        Ok(serde_json::to_string(&serde_json::json!({
            "task_id": task_id_str,
            "status": status,
            "updated": true
        }))?)
    }

    /// Handle the task_get tool: retrieve a task by ID.
    pub(crate) fn handle_skill(&self, args: &serde_json::Value) -> Result<String> {
        let skill_name = args
            .get("skill")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("skill parameter is required"))?;
        let skill_args = args.get("args").and_then(|v| v.as_str());

        let mgr = deepseek_skills::SkillManager::new(&self.workspace)?;
        let configured_paths = &self.cfg.skills.paths;

        match mgr.run(skill_name, skill_args, configured_paths) {
            Ok(output) => Ok(serde_json::to_string(&json!({
                "skill": skill_name,
                "rendered_prompt": output.rendered_prompt,
                "source_path": output.source_path.to_string_lossy(),
            }))?),
            Err(e) => Ok(serde_json::to_string(&json!({
                "error": format!("skill '{}' not found or failed: {}", skill_name, e)
            }))?),
        }
    }

    /// Handle the kill_shell tool: stop a background bash process.
    pub(crate) fn handle_think_deeply(&self, args: &serde_json::Value) -> Result<String> {
        let question = args
            .get("question")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let context = args
            .get("context")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let consultation_type = consultation::ConsultationType::from_str(
            args.get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("error_analysis"),
        );

        if question.is_empty() {
            return Err(anyhow!("question parameter is required"));
        }

        let cb = self.stream_callback.lock().ok().and_then(|g| g.clone());
        let result = consultation::consult_r1(
            self.llm.as_ref(),
            &self.cfg.llm.max_think_model,
            &consultation::ConsultationRequest {
                question,
                context,
                consultation_type,
            },
            cb.as_ref(),
        )?;

        Ok(format!(
            "## R1 Analysis\n\n{}\n\n---\n*Use this advice to guide your next actions. You have full tool access.*",
            result.advice
        ))
    }

    pub(crate) fn handle_task_get(&self, args: &serde_json::Value) -> Result<String> {
        let task_id_str = args
            .get("task_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("task_id parameter is required"))?;
        let task_id = Uuid::parse_str(task_id_str)?;
        let tasks = self.store.list_tasks(None)?;
        if let Some(task) = tasks.iter().find(|t| t.task_id == task_id) {
            Ok(serde_json::to_string(&serde_json::json!({
                "task_id": task.task_id.to_string(),
                "subject": task.title,
                "status": task.status,
                "priority": task.priority,
                "outcome": task.outcome,
                "created_at": task.created_at,
                "updated_at": task.updated_at,
            }))?)
        } else {
            Ok(serde_json::to_string(
                &serde_json::json!({"error": format!("task not found: {task_id_str}")}),
            )?)
        }
    }
}
