use super::*;
use codingbuddy_core::{ToolName, ToolTier};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

pub(super) fn handle_agent_level_tool(
    tool_loop: &mut ToolUseLoop<'_>,
    llm_call: &LlmToolCall,
) -> Result<Vec<ToolCallRecord>> {
    let start = Instant::now();
    let args: serde_json::Value = serde_json::from_str(&llm_call.arguments).unwrap_or_else(|e| {
        eprintln!(
            "[tool_loop] failed to parse agent tool arguments for '{}': {e}",
            llm_call.name
        );
        serde_json::json!({})
    });

    let args_summary = summarize_args(&args);

    tool_loop.emit(StreamChunk::ToolCallStart {
        tool_name: llm_call.name.clone(),
        args_summary: args_summary.clone(),
    });

    let result_content = match llm_call.name.as_str() {
        "extended_thinking" | "think_deeply" => {
            let question = args.get("question").and_then(|v| v.as_str()).unwrap_or("");
            let context = args.get("context").and_then(|v| v.as_str()).unwrap_or("");
            handle_extended_thinking(tool_loop, question, context)?
        }
        "tool_search" => handle_tool_search(tool_loop, &args),
        "user_question" => {
            let question = args
                .get("question")
                .and_then(|v| v.as_str())
                .unwrap_or("(no question)");
            let options: Vec<String> = args
                .get("options")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            if let Some(ref cb) = tool_loop.user_question_cb {
                let uq = UserQuestion {
                    question: question.to_string(),
                    options,
                };
                match cb(uq) {
                    Some(answer) => format!("User response: {answer}"),
                    None => format!(
                        "Question for user: {question}\n[User cancelled. Proceed with your best judgment.]"
                    ),
                }
            } else {
                format!(
                    "Question for user: {question}\n[User response not available in this context. Proceed with your best judgment or state your assumption.]"
                )
            }
        }
        "task_create" => handle_task_create(tool_loop, &args)?,
        "task_update" => handle_task_update(tool_loop, &args)?,
        "todo_read" => handle_todo_read(tool_loop)?,
        "todo_write" => handle_todo_write(tool_loop, &args)?,
        "task_get" => handle_task_get(tool_loop, &args)?,
        "task_list" => handle_task_list(tool_loop)?,
        "task_output" => handle_task_output(tool_loop, &args)?,
        "task_stop" => handle_task_stop(tool_loop, &args)?,
        "spawn_task" => handle_spawn_task(tool_loop, &args)?,
        "enter_plan_mode" => handle_enter_plan_mode(tool_loop)?,
        "exit_plan_mode" => handle_exit_plan_mode(tool_loop, &args)?,
        "skill" => handle_skill(tool_loop, &args)?,
        _ => {
            format!(
                "Agent-level tool '{}' is not yet available in tool-use mode. Try a different approach.",
                llm_call.name
            )
        }
    };

    let duration = start.elapsed().as_millis() as u64;
    tool_loop.emit(StreamChunk::ToolCallEnd {
        tool_name: llm_call.name.clone(),
        duration_ms: duration,
        success: true,
        summary: "ok".to_string(),
    });

    let content = tool_bridge::truncate_agent_output(&result_content);

    tool_loop.messages.push(ChatMessage::Tool {
        tool_call_id: llm_call.id.clone(),
        content,
    });

    Ok(vec![ToolCallRecord {
        tool_name: llm_call.name.clone(),
        tool_call_id: llm_call.id.clone(),
        args_summary,
        success: true,
        duration_ms: duration,
        args_json: args_json_for_record(&llm_call.name, &llm_call.arguments),
        result_preview: None,
    }])
}

pub(super) fn handle_tool_search(
    tool_loop: &mut ToolUseLoop<'_>,
    args: &serde_json::Value,
) -> String {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .unwrap_or("");
    if query.is_empty() {
        return "Error: tool_search requires a non-empty 'query' string.".to_string();
    }

    let mut matches = search_extended_tools(query, &tool_loop.discoverable_tools);
    matches.sort_by_key(|tool| tool_search_priority(&tool.function.name));

    let weak_model = is_weak_model_for_tool_search(&tool_loop.config.model);
    let keyword_count = query
        .split_whitespace()
        .filter(|segment| !segment.trim().is_empty())
        .count();
    let promotion_cap = if weak_model {
        if keyword_count <= 1 { 1 } else { 2 }
    } else {
        6
    };

    let mut newly_enabled = 0usize;
    let mut already_enabled = 0usize;
    let mut deferred = 0usize;
    for tool in &matches {
        if tool_loop
            .tools
            .iter()
            .any(|existing| existing.function.name == tool.function.name)
        {
            already_enabled += 1;
            continue;
        }
        if newly_enabled < promotion_cap {
            tool_loop.tools.push(tool.clone());
            newly_enabled += 1;
        } else {
            deferred += 1;
        }
    }

    let visible = matches
        .iter()
        .take(8)
        .cloned()
        .collect::<Vec<ToolDefinition>>();
    let mut result = format_tool_search_results(&visible);
    if matches.len() > visible.len() {
        let hidden = matches.len().saturating_sub(visible.len());
        result.push_str(&format!(
            "\n\n{hidden} additional match(es) hidden for brevity."
        ));
    }
    if newly_enabled > 0 {
        result.push_str(&format!(
            "\n\nEnabled {newly_enabled} matching tool(s) for subsequent turns{}.",
            if weak_model {
                " (weak-model guardrail: limited promotion set)"
            } else {
                ""
            }
        ));
    }
    if already_enabled > 0 {
        result.push_str(&format!(
            "\nAlready active in this session: {already_enabled} match(es)."
        ));
    }
    if deferred > 0 {
        result.push_str(&format!(
            "\nDeferred {deferred} match(es). Narrow your query to promote more tools."
        ));
    }
    result
}

fn is_weak_model_for_tool_search(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    !codingbuddy_core::is_reasoner_model(model)
        && (lower.contains("deepseek")
            || lower.contains("qwen")
            || lower.contains("phi")
            || lower.contains("tinyllama"))
}

fn tool_search_priority(name: &str) -> (u8, u8, String) {
    let primary = match name {
        "enter_plan_mode" | "exit_plan_mode" => 0,
        "task_output" | "task_get" | "task_list" => 1,
        "todo_read" | "todo_write" => 2,
        "spawn_task" => 3,
        "diagnostics_check" => 4,
        "patch_direct" | "multi_edit" | "fs_edit" => 5,
        "web_fetch" | "web_search" => 6,
        "chrome_navigate" | "chrome_click" | "chrome_screenshot" => 7,
        _ => 20,
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

pub(super) fn handle_extended_thinking(
    tool_loop: &ToolUseLoop<'_>,
    question: &str,
    context: &str,
) -> Result<String> {
    let prompt = format!(
        "Analyze this problem carefully:\n\nQuestion: {question}\n\nContext:\n{context}\n\nProvide a clear, actionable recommendation."
    );
    let request = ChatRequest {
        model: tool_loop.config.extended_thinking_model.clone(),
        messages: vec![
            ChatMessage::System {
                content: "You are a deep reasoning engine. Provide thorough analysis and clear recommendations.".to_string(),
            },
            ChatMessage::User { content: prompt },
        ],
        tools: tool_loop.tools.clone(),
        tool_choice: ToolChoice::auto(),
        max_tokens: codingbuddy_core::CODINGBUDDY_REASONER_MAX_OUTPUT_TOKENS,
        temperature: None,
        top_p: None,
        presence_penalty: None,
        frequency_penalty: None,
        logprobs: None,
        top_logprobs: None,
        thinking: None,
        images: vec![],
        response_format: None,
    };
    let response = tool_loop.llm.complete_chat(&request)?;
    Ok(response.text)
}

pub(super) fn handle_task_create(
    tool_loop: &ToolUseLoop<'_>,
    args: &serde_json::Value,
) -> Result<String> {
    let subject = args
        .get("subject")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow!("task_create requires a non-empty 'subject'"))?;
    let description = args
        .get("description")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToString::to_string);
    let priority = args.get("priority").and_then(|v| v.as_u64()).unwrap_or(1) as u32;

    let store = tool_loop.workspace_store()?;
    let session = tool_loop.current_session(&store)?;
    let now = chrono::Utc::now().to_rfc3339();
    let record = TaskQueueRecord {
        task_id: Uuid::now_v7(),
        session_id: session.session_id,
        title: subject.to_string(),
        description,
        priority,
        status: "pending".to_string(),
        outcome: None,
        artifact_path: None,
        created_at: now.clone(),
        updated_at: now,
    };
    store.insert_task(&record)?;

    Ok(serde_json::json!({
        "task_id": record.task_id,
        "subject": record.title,
        "description": record.description,
        "priority": record.priority,
        "status": record.status,
    })
    .to_string())
}

pub(super) fn handle_task_update(
    tool_loop: &ToolUseLoop<'_>,
    args: &serde_json::Value,
) -> Result<String> {
    let task_id = args
        .get("task_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("task_update requires 'task_id'"))?;
    let task_id = Uuid::parse_str(task_id)?;
    let status = args
        .get("status")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("task_update requires 'status'"))?;
    let outcome = args.get("outcome").and_then(|v| v.as_str());

    let store = tool_loop.workspace_store()?;
    store.update_task_status(task_id, status, outcome)?;
    tool_loop.emit_event_if_present(EventKind::TaskUpdated {
        task_id: task_id.to_string(),
        status: status.to_string(),
    });

    Ok(serde_json::json!({
        "task_id": task_id,
        "status": status,
        "outcome": outcome,
    })
    .to_string())
}

pub(super) fn canonical_todo_status(value: Option<&str>) -> &'static str {
    match value
        .unwrap_or("pending")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "in_progress" | "in-progress" | "active" | "working" | "current" => "in_progress",
        "completed" | "done" | "finished" => "completed",
        _ => "pending",
    }
}

pub(super) fn todo_summary_payload(todos: &[SessionTodoRecord]) -> serde_json::Value {
    let completed = todos
        .iter()
        .filter(|todo| todo.status.eq_ignore_ascii_case("completed"))
        .count();
    let in_progress = todos
        .iter()
        .filter(|todo| todo.status.eq_ignore_ascii_case("in_progress"))
        .count();
    let active = todos.len().saturating_sub(completed);
    let current = todos
        .iter()
        .find(|todo| todo.status.eq_ignore_ascii_case("in_progress"))
        .or_else(|| {
            todos
                .iter()
                .find(|todo| todo.status.eq_ignore_ascii_case("pending"))
        })
        .map(|todo| {
            serde_json::json!({
                "todo_id": todo.todo_id.to_string(),
                "content": todo.content,
                "status": todo.status,
                "position": todo.position,
            })
        });
    serde_json::json!({
        "total": todos.len(),
        "active": active,
        "completed": completed,
        "in_progress": in_progress,
        "current": current,
    })
}

pub(super) fn handle_todo_read(tool_loop: &ToolUseLoop<'_>) -> Result<String> {
    let store = tool_loop.workspace_store()?;
    let session = tool_loop.current_session(&store)?;
    let todos = store.list_session_todos(session.session_id)?;
    let summary = todo_summary_payload(&todos);
    Ok(serde_json::json!({
        "session_id": session.session_id.to_string(),
        "todos": todos,
        "summary": summary,
    })
    .to_string())
}

pub(super) fn handle_todo_write(
    tool_loop: &ToolUseLoop<'_>,
    args: &serde_json::Value,
) -> Result<String> {
    let items = args
        .get("items")
        .or_else(|| args.get("todos"))
        .and_then(|value| value.as_array())
        .ok_or_else(|| anyhow!("todo_write requires an 'items' array"))?;
    let store = tool_loop.workspace_store()?;
    let session = tool_loop.current_session(&store)?;
    let existing = store.list_session_todos(session.session_id)?;
    let existing_created_at = existing
        .iter()
        .map(|todo| (todo.todo_id, todo.created_at.clone()))
        .collect::<HashMap<_, _>>();
    let now = chrono::Utc::now().to_rfc3339();
    let mut seen_ids = HashSet::new();
    let mut todos = Vec::with_capacity(items.len());
    for (idx, item) in items.iter().enumerate() {
        let content = item
            .get("content")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("todo_write items[{idx}] requires non-empty 'content'"))?;
        let todo_id = item
            .get("id")
            .and_then(|value| value.as_str())
            .map(Uuid::parse_str)
            .transpose()?
            .unwrap_or_else(Uuid::now_v7);
        if !seen_ids.insert(todo_id) {
            return Err(anyhow!("todo_write items include duplicate id: {todo_id}"));
        }
        let status = canonical_todo_status(item.get("status").and_then(|value| value.as_str()));
        let created_at = existing_created_at
            .get(&todo_id)
            .cloned()
            .unwrap_or_else(|| now.clone());
        todos.push(SessionTodoRecord {
            todo_id,
            session_id: session.session_id,
            content: content.to_string(),
            status: status.to_string(),
            position: idx as u32,
            created_at,
            updated_at: now.clone(),
        });
    }
    let in_progress_count = todos
        .iter()
        .filter(|todo| todo.status.eq_ignore_ascii_case("in_progress"))
        .count();
    if in_progress_count > 1 {
        return Err(anyhow!(
            "todo_write allows at most one 'in_progress' item (got {in_progress_count})"
        ));
    }
    store.replace_session_todos(session.session_id, &todos)?;
    let summary = todo_summary_payload(&todos);
    Ok(serde_json::json!({
        "session_id": session.session_id.to_string(),
        "todos": todos,
        "summary": summary,
    })
    .to_string())
}

pub(super) fn handle_task_get(
    tool_loop: &ToolUseLoop<'_>,
    args: &serde_json::Value,
) -> Result<String> {
    let task_id = args
        .get("task_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("task_get requires 'task_id'"))?;
    let task_id = Uuid::parse_str(task_id)?;
    let store = tool_loop.workspace_store()?;
    let task = store
        .load_task(task_id)?
        .ok_or_else(|| anyhow!("task not found: {task_id}"))?;
    Ok(serde_json::to_string(&task)?)
}

pub(super) fn handle_task_list(tool_loop: &ToolUseLoop<'_>) -> Result<String> {
    let store = tool_loop.workspace_store()?;
    let session = tool_loop.current_session(&store)?;
    let tasks = store.list_tasks(Some(session.session_id))?;
    Ok(serde_json::json!({ "tasks": tasks }).to_string())
}

pub(super) fn handle_task_output(
    tool_loop: &ToolUseLoop<'_>,
    args: &serde_json::Value,
) -> Result<String> {
    let store = tool_loop.workspace_store()?;
    let task = if let Some(task_id) = args.get("task_id").and_then(|v| v.as_str()) {
        let task_id = Uuid::parse_str(task_id)?;
        Some(
            store
                .load_task(task_id)?
                .ok_or_else(|| anyhow!("task not found: {task_id}"))?,
        )
    } else {
        None
    };

    let run = if let Some(run_id) = args.get("run_id").and_then(|v| v.as_str()) {
        let run_id = Uuid::parse_str(run_id)?;
        Some(
            store
                .load_subagent_run(run_id)?
                .ok_or_else(|| anyhow!("subagent run not found: {run_id}"))?,
        )
    } else if let Some(task) = task.as_ref() {
        store.load_subagent_run_for_task(task.task_id)?
    } else {
        return Err(anyhow!("task_output requires 'task_id' or 'run_id'"));
    };

    let background_job = if let Some(run) = run.as_ref() {
        if let Some(job_id) = run.background_job_id {
            store.load_background_job(job_id)?
        } else {
            store.load_background_job_by_reference(&format!("subagent:{}", run.run_id))?
        }
    } else {
        None
    };

    let child_session = run
        .as_ref()
        .and_then(|record| record.child_session_id)
        .map(|session_id| store.load_session(session_id))
        .transpose()?
        .flatten();

    let (output_source, output_text) = if let Some(run) = run.as_ref() {
        if let Some(output) = run.output.as_deref() {
            (Some("subagent"), Some(output.to_string()))
        } else if let Some(error) = run.error.as_deref() {
            (Some("subagent_error"), Some(error.to_string()))
        } else if let Some(task) = task.as_ref() {
            (
                task.outcome.as_deref().map(|_| "task"),
                task.outcome.as_deref().map(str::to_string),
            )
        } else {
            (None, None)
        }
    } else if let Some(task) = task.as_ref() {
        (
            task.outcome.as_deref().map(|_| "task"),
            task.outcome.as_deref().map(str::to_string),
        )
    } else {
        (None, None)
    };

    let task_status = run
        .as_ref()
        .map(|record| record.status.as_str())
        .or_else(|| task.as_ref().map(|record| record.status.as_str()))
        .unwrap_or("unknown");
    let resume_session_id = child_session
        .as_ref()
        .map(|session| session.session_id.to_string());
    let next_action = if task_status.eq_ignore_ascii_case("running") {
        "monitor_or_resume"
    } else if task_status.eq_ignore_ascii_case("failed")
        || task_status.eq_ignore_ascii_case("error")
    {
        "inspect_failure"
    } else if resume_session_id.is_some() {
        "resume_child_session_or_integrate_output"
    } else {
        "integrate_output"
    };
    let summary = output_text
        .as_deref()
        .map(|text| {
            let trimmed = text.trim();
            let end = trimmed.floor_char_boundary(240.min(trimmed.len()));
            if trimmed.len() > end {
                format!("{}...", &trimmed[..end])
            } else {
                trimmed.to_string()
            }
        })
        .filter(|text| !text.is_empty());

    Ok(serde_json::json!({
        "task": task,
        "run": run,
        "background_job": background_job,
        "child_session": child_session,
        "output_source": output_source,
        "output_text": output_text,
        "handoff": {
            "schema": "deepseek.task_handoff.v1",
            "status": task_status,
            "summary": summary,
            "next_action": next_action,
            "resume_session_id": resume_session_id,
            "resume_command": resume_session_id.as_ref().map(|id| format!("codingbuddy --resume {id}")),
            "source": output_source,
        }
    })
    .to_string())
}

pub(super) fn handle_task_stop(
    tool_loop: &ToolUseLoop<'_>,
    args: &serde_json::Value,
) -> Result<String> {
    let store = tool_loop.workspace_store()?;
    let mut run = if let Some(run_id) = args.get("run_id").and_then(|v| v.as_str()) {
        let run_id = Uuid::parse_str(run_id)?;
        store
            .load_subagent_run(run_id)?
            .ok_or_else(|| anyhow!("subagent run not found: {run_id}"))?
    } else if let Some(task_id) = args.get("task_id").and_then(|v| v.as_str()) {
        let task_id = Uuid::parse_str(task_id)?;
        store
            .load_subagent_run_for_task(task_id)?
            .ok_or_else(|| anyhow!("task has no associated subagent run: {task_id}"))?
    } else {
        return Err(anyhow!("task_stop requires 'task_id' or 'run_id'"));
    };
    let task_id = run
        .task_id
        .ok_or_else(|| anyhow!("subagent run is missing task_id"))?;
    let session_id = run
        .session_id
        .or(tool_loop.config.session_id)
        .ok_or_else(|| anyhow!("subagent run is missing session_id"))?;

    let mut background_job = if let Some(job_id) = run.background_job_id {
        store.load_background_job(job_id)?
    } else {
        store.load_background_job_by_reference(&format!("subagent:{}", run.run_id))?
    }
    .ok_or_else(|| anyhow!("no background job found for task {task_id}"))?;

    let mut metadata = serde_json::from_str::<serde_json::Value>(&background_job.metadata_json)
        .unwrap_or_else(|_| serde_json::json!({}));
    let mut terminated_pid = false;
    let mut stop_file_written = false;

    if let Some(stop_file) = metadata.get("stop_file").and_then(|value| value.as_str()) {
        let stop_path = std::path::PathBuf::from(stop_file);
        if let Some(parent) = stop_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(
            &stop_path,
            format!("stop requested at {}\n", chrono::Utc::now().to_rfc3339()),
        );
        stop_file_written = true;
    }

    if let Some(pid) = metadata.get("pid").and_then(|value| value.as_u64()) {
        #[cfg(not(windows))]
        {
            let status = std::process::Command::new("kill")
                .args(["-TERM", &pid.to_string()])
                .status();
            terminated_pid = status.as_ref().is_ok_and(|status| status.success());
        }
        #[cfg(windows)]
        {
            let status = std::process::Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .status();
            terminated_pid = status.as_ref().is_ok_and(|status| status.success());
        }
    }

    if let Some(object) = metadata.as_object_mut() {
        object.insert("reason".to_string(), serde_json::json!("manual_stop"));
        object.insert(
            "terminated_pid".to_string(),
            serde_json::json!(terminated_pid),
        );
        object.insert(
            "stop_file_written".to_string(),
            serde_json::json!(stop_file_written),
        );
        object.insert(
            "stopped_at".to_string(),
            serde_json::json!(chrono::Utc::now().to_rfc3339()),
        );
    }

    background_job.status = "stopped".to_string();
    background_job.updated_at = chrono::Utc::now().to_rfc3339();
    background_job.metadata_json = metadata.to_string();
    store.upsert_background_job(&background_job)?;
    store.update_task_status(task_id, "cancelled", Some("stopped by user"))?;

    run.status = "stopped".to_string();
    run.error = Some("stopped by user".to_string());
    run.updated_at = chrono::Utc::now().to_rfc3339();
    store.upsert_subagent_run(&run)?;

    let event = |kind| {
        let seq_no = store.next_seq_no(session_id)?;
        store.append_event(&codingbuddy_core::EventEnvelope {
            seq_no,
            at: chrono::Utc::now(),
            session_id,
            kind,
        })
    };
    event(EventKind::BackgroundJobStopped {
        job_id: background_job.job_id,
        reason: "manual_stop".to_string(),
    })?;
    event(EventKind::TaskUpdated {
        task_id: task_id.to_string(),
        status: "cancelled".to_string(),
    })?;

    Ok(serde_json::json!({
        "task_id": task_id,
        "run_id": run.run_id,
        "background_job_id": background_job.job_id,
        "status": "cancelled",
        "terminated_pid": terminated_pid,
        "stop_file_written": stop_file_written,
    })
    .to_string())
}

pub(super) fn handle_spawn_task(
    tool_loop: &ToolUseLoop<'_>,
    args: &serde_json::Value,
) -> Result<String> {
    let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    let task_name = args
        .get("description")
        .or_else(|| args.get("task_name"))
        .and_then(|v| v.as_str())
        .unwrap_or("subtask");
    let subagent_type = args
        .get("subagent_type")
        .and_then(|v| v.as_str())
        .unwrap_or("general-purpose");
    let model_override = args.get("model").and_then(|v| v.as_str()).map(String::from);
    let max_turns = args
        .get("max_turns")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    let run_in_background = args
        .get("run_in_background")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if prompt.is_empty() {
        return Ok(
            "Error: spawn_task requires a 'prompt' argument describing the task.".to_string(),
        );
    }

    let store = tool_loop.workspace_store()?;
    let parent_session = tool_loop.current_session(&store)?;
    let child_session = store.fork_session(parent_session.session_id)?;
    store.save_session(&parent_session)?;
    let now = chrono::Utc::now().to_rfc3339();
    let run_id = Uuid::now_v7();
    let task_id = Uuid::now_v7();
    let role_label = match subagent_type {
        "explore" => "explore",
        "plan" => "plan",
        "bash" => "bash",
        _ => "general-purpose",
    };

    store.insert_task(&TaskQueueRecord {
        task_id,
        session_id: parent_session.session_id,
        title: task_name.to_string(),
        description: Some(format!("[{role_label}] {prompt}")),
        priority: 1,
        status: "in_progress".to_string(),
        outcome: None,
        artifact_path: Some(format!("session://{}", child_session.session_id)),
        created_at: now.clone(),
        updated_at: now.clone(),
    })?;
    store.upsert_subagent_run(&codingbuddy_store::SubagentRunRecord {
        run_id,
        session_id: Some(parent_session.session_id),
        task_id: Some(task_id),
        child_session_id: Some(child_session.session_id),
        background_job_id: None,
        name: task_name.to_string(),
        goal: prompt.to_string(),
        status: "running".to_string(),
        output: None,
        error: None,
        created_at: now.clone(),
        updated_at: now.clone(),
    })?;
    tool_loop.emit_event_if_present(EventKind::TaskUpdated {
        task_id: task_id.to_string(),
        status: "in_progress".to_string(),
    });

    tool_loop.emit(StreamChunk::SubagentSpawned {
        run_id: run_id.to_string(),
        name: task_name.to_string(),
        goal: prompt.to_string(),
    });
    tool_loop.emit_event_if_present(EventKind::SubagentSpawned {
        run_id,
        name: task_name.to_string(),
        goal: prompt.to_string(),
    });

    let request = SubagentRequest {
        run_id,
        task_id: Some(task_id),
        parent_session_id: Some(parent_session.session_id),
        child_session_id: Some(child_session.session_id),
        prompt: prompt.to_string(),
        task_name: task_name.to_string(),
        subagent_type: subagent_type.to_string(),
        model_override,
        max_turns,
        run_in_background,
    };

    if let Some(ref worker) = tool_loop.config.subagent_worker {
        if let Some(ref hooks) = tool_loop.hooks {
            let input = HookInput {
                event: HookEvent::SubagentStart.as_str().to_string(),
                tool_name: Some("spawn_task".to_string()),
                tool_input: Some(args.clone()),
                tool_result: None,
                prompt: Some(prompt.to_string()),
                session_type: None,
                workspace: tool_loop.workspace_str().to_string(),
            };
            ToolUseLoop::fire_hook_logged(hooks, HookEvent::SubagentStart, &input);
        }
        let result = match worker(request) {
            Ok(result) => {
                if run_in_background {
                    let payload = serde_json::from_str::<serde_json::Value>(&result)
                        .unwrap_or_else(|_| serde_json::json!({ "raw": result }));
                    let background_job_id = payload
                        .get("job_id")
                        .and_then(|value| value.as_str())
                        .and_then(|value| Uuid::parse_str(value).ok());
                    store.upsert_subagent_run(&codingbuddy_store::SubagentRunRecord {
                        run_id,
                        session_id: Some(parent_session.session_id),
                        task_id: Some(task_id),
                        child_session_id: Some(child_session.session_id),
                        background_job_id,
                        name: task_name.to_string(),
                        goal: prompt.to_string(),
                        status: "running".to_string(),
                        output: None,
                        error: None,
                        created_at: now.clone(),
                        updated_at: chrono::Utc::now().to_rfc3339(),
                    })?;
                    return Ok(serde_json::json!({
                        "task_id": task_id,
                        "run_id": run_id,
                        "child_session_id": child_session.session_id,
                        "background_job_id": background_job_id,
                        "status": "running",
                        "handoff": {
                            "schema": "deepseek.task_handoff.v1",
                            "status": "running",
                            "summary": "Subagent scheduled in background.",
                            "next_action": "monitor_or_resume",
                            "resume_session_id": child_session.session_id,
                            "resume_command": format!("codingbuddy --resume {}", child_session.session_id),
                        },
                        "todo_update_hint": {
                            "action": "set_in_progress",
                            "content": task_name,
                            "suggested_status": "in_progress",
                        }
                    })
                    .to_string());
                }
                store.update_task_status(task_id, "completed", Some(&result))?;
                store.upsert_subagent_run(&codingbuddy_store::SubagentRunRecord {
                    run_id,
                    session_id: Some(parent_session.session_id),
                    task_id: Some(task_id),
                    child_session_id: Some(child_session.session_id),
                    background_job_id: None,
                    name: task_name.to_string(),
                    goal: prompt.to_string(),
                    status: "completed".to_string(),
                    output: Some(result.clone()),
                    error: None,
                    created_at: now.clone(),
                    updated_at: chrono::Utc::now().to_rfc3339(),
                })?;
                tool_loop.emit_event_if_present(EventKind::TaskUpdated {
                    task_id: task_id.to_string(),
                    status: "completed".to_string(),
                });
                tool_loop.emit(StreamChunk::SubagentCompleted {
                    run_id: run_id.to_string(),
                    name: task_name.to_string(),
                    summary: result.clone(),
                });
                tool_loop.emit_event_if_present(EventKind::SubagentCompleted {
                    run_id,
                    output: result.clone(),
                });
                Ok(serde_json::json!({
                    "task_id": task_id,
                    "run_id": run_id,
                    "child_session_id": child_session.session_id,
                    "status": "completed",
                    "output": result,
                    "handoff": {
                        "schema": "deepseek.task_handoff.v1",
                        "status": "completed",
                        "summary": "Subagent finished. Integrate output and advance checklist.",
                        "next_action": "integrate_output",
                        "resume_session_id": child_session.session_id,
                        "resume_command": format!("codingbuddy --resume {}", child_session.session_id),
                    },
                    "todo_update_hint": {
                        "action": "mark_completed",
                        "content": task_name,
                        "suggested_status": "completed",
                    }
                })
                .to_string())
            }
            Err(e) => {
                store.update_task_status(task_id, "failed", Some(&e.to_string()))?;
                store.upsert_subagent_run(&codingbuddy_store::SubagentRunRecord {
                    run_id,
                    session_id: Some(parent_session.session_id),
                    task_id: Some(task_id),
                    child_session_id: Some(child_session.session_id),
                    background_job_id: None,
                    name: task_name.to_string(),
                    goal: prompt.to_string(),
                    status: "failed".to_string(),
                    output: None,
                    error: Some(e.to_string()),
                    created_at: now.clone(),
                    updated_at: chrono::Utc::now().to_rfc3339(),
                })?;
                tool_loop.emit_event_if_present(EventKind::TaskUpdated {
                    task_id: task_id.to_string(),
                    status: "failed".to_string(),
                });
                let message = format!(
                    "Subagent '{task_name}' failed: {e}. Try a different approach or handle the task directly. \
                     Update your todo checklist to reflect this blocker (status=pending or completed with note)."
                );
                tool_loop.emit(StreamChunk::SubagentFailed {
                    run_id: run_id.to_string(),
                    name: task_name.to_string(),
                    error: message.clone(),
                });
                tool_loop.emit_event_if_present(EventKind::SubagentFailed {
                    run_id,
                    error: e.to_string(),
                });
                Ok(serde_json::json!({
                    "task_id": task_id,
                    "run_id": run_id,
                    "child_session_id": child_session.session_id,
                    "status": "failed",
                    "error": message,
                    "handoff": {
                        "schema": "deepseek.task_handoff.v1",
                        "status": "failed",
                        "summary": "Subagent failed; inspect error and retry with narrowed scope.",
                        "next_action": "inspect_failure",
                        "resume_session_id": child_session.session_id,
                        "resume_command": format!("codingbuddy --resume {}", child_session.session_id),
                    }
                })
                .to_string())
            }
        };
        if let Some(ref hooks) = tool_loop.hooks {
            let input = HookInput {
                event: HookEvent::SubagentStop.as_str().to_string(),
                tool_name: Some("spawn_task".to_string()),
                tool_input: Some(args.clone()),
                tool_result: result
                    .as_ref()
                    .ok()
                    .map(|r| serde_json::Value::String(r.clone())),
                prompt: Some(prompt.to_string()),
                session_type: None,
                workspace: tool_loop.workspace_str().to_string(),
            };
            ToolUseLoop::fire_hook_logged(hooks, HookEvent::SubagentStop, &input);
        }
        result
    } else {
        store.update_task_status(
            task_id,
            "failed",
            Some("spawn_task is not available in this context"),
        )?;
        store.upsert_subagent_run(&codingbuddy_store::SubagentRunRecord {
            run_id,
            session_id: Some(parent_session.session_id),
            task_id: Some(task_id),
            child_session_id: Some(child_session.session_id),
            background_job_id: None,
            name: task_name.to_string(),
            goal: prompt.to_string(),
            status: "failed".to_string(),
            output: None,
            error: Some("spawn_task is not available in this context".to_string()),
            created_at: now,
            updated_at: chrono::Utc::now().to_rfc3339(),
        })?;
        tool_loop.emit_event_if_present(EventKind::TaskUpdated {
            task_id: task_id.to_string(),
            status: "failed".to_string(),
        });
        Ok(format!(
            "spawn_task is not available in this context. Handle the task directly instead.\n\
             Task: {task_name}\n\
             Prompt: {prompt}\n\
             Use the available tools to accomplish this yourself."
        ))
    }
}

pub(super) fn handle_enter_plan_mode(tool_loop: &mut ToolUseLoop<'_>) -> Result<String> {
    let store = tool_loop.workspace_store()?;
    let mut session = tool_loop.current_session(&store)?;
    let goal = tool_loop.latest_user_prompt();
    let plan_id = session.active_plan_id.unwrap_or_else(Uuid::now_v7);
    let plan = Plan {
        plan_id,
        version: 1,
        goal,
        assumptions: Vec::new(),
        steps: Vec::new(),
        verification: Vec::new(),
        risk_notes: Vec::new(),
    };

    session.status = SessionState::Planning;
    session.active_plan_id = Some(plan_id);
    store.save_session(&session)?;
    store.save_plan(session.session_id, &plan)?;
    tool_loop.apply_phase_transition(codingbuddy_core::TaskPhase::Plan);

    tool_loop.emit_event_if_present(EventKind::EnterPlanMode {
        session_id: session.session_id,
    });
    tool_loop.emit_event_if_present(EventKind::PlanCreated { plan });

    Ok(
        "Plan mode entered. Explore and write the implementation plan before executing changes."
            .to_string(),
    )
}

pub(super) fn handle_exit_plan_mode(
    tool_loop: &mut ToolUseLoop<'_>,
    _args: &serde_json::Value,
) -> Result<String> {
    let store = tool_loop.workspace_store()?;
    let mut session = tool_loop.current_session(&store)?;
    let plan_id = session.active_plan_id.unwrap_or_else(Uuid::now_v7);
    let previous_version = store
        .load_plan(plan_id)?
        .map(|plan| plan.version)
        .unwrap_or(0);
    let plan_text = tool_loop.latest_assistant_text();
    let goal = if let Some(plan) = store.load_plan(plan_id)? {
        plan.goal
    } else {
        tool_loop.latest_user_prompt()
    };
    let plan = tool_loop.build_plan_from_text(plan_id, previous_version + 1, goal, &plan_text);

    session.status = SessionState::AwaitingApproval;
    session.active_plan_id = Some(plan_id);
    store.save_session(&session)?;
    store.save_plan(session.session_id, &plan)?;
    tool_loop.phase = Some(codingbuddy_core::TaskPhase::Plan);
    tool_loop.phase_read_only_calls = 0;
    tool_loop.phase_edit_calls = 0;

    tool_loop.emit_event_if_present(EventKind::PlanRevised { plan });
    tool_loop.emit_event_if_present(EventKind::ExitPlanMode {
        session_id: session.session_id,
    });

    Ok("Plan saved. Await user approval before executing code changes.".to_string())
}

pub(super) fn handle_skill(
    tool_loop: &ToolUseLoop<'_>,
    args: &serde_json::Value,
) -> Result<String> {
    let skill_name = args.get("skill").and_then(|v| v.as_str()).unwrap_or("");
    let skill_args = args.get("args").and_then(|v| v.as_str());

    if skill_name.is_empty() {
        return Ok(
            "Error: skill tool requires a 'skill' argument with the skill name.".to_string(),
        );
    }

    let Some(ref runner) = tool_loop.config.skill_runner else {
        return Ok(format!(
            "Skill '{skill_name}' is not available in this context. \
             Skills require the CLI environment to be properly initialized."
        ));
    };

    let result = match runner(skill_name, skill_args) {
        Ok(Some(result)) => result,
        Ok(None) => {
            return Ok(format!(
                "Skill '{skill_name}' not found. Check available skills with /skills list."
            ));
        }
        Err(e) => {
            return Ok(format!("Failed to load skill '{skill_name}': {e}"));
        }
    };

    if result.disable_model_invocation {
        return Ok(format!(
            "Skill '{skill_name}' has disable-model-invocation set. \
             This skill can only be invoked directly by the user via /{skill_name}, \
             not programmatically by the model."
        ));
    }

    if result.forked {
        if let Some(ref worker) = tool_loop.config.subagent_worker {
            let request = SubagentRequest {
                run_id: Uuid::now_v7(),
                task_id: None,
                parent_session_id: tool_loop
                    .workspace_store()
                    .ok()
                    .and_then(|store| tool_loop.current_session(&store).ok())
                    .map(|session| session.session_id),
                child_session_id: None,
                prompt: result.rendered_prompt.clone(),
                task_name: format!("skill:{skill_name}"),
                subagent_type: "general-purpose".to_string(),
                model_override: None,
                max_turns: None,
                run_in_background: false,
            };
            return match worker(request) {
                Ok(output) => Ok(format!(
                    "Skill '{skill_name}' (forked) completed:\n{output}"
                )),
                Err(e) => Ok(format!(
                    "Skill '{skill_name}' (forked) failed: {e}. \
                     Try running the skill directly or handle the task yourself."
                )),
            };
        }
        return Ok(format!(
            "Skill '{skill_name}' requires forked execution but no subagent worker is available. \
             Returning the skill prompt for inline execution:\n\n{}",
            result.rendered_prompt
        ));
    }

    let mut response = format!(
        "<skill-execution name=\"{skill_name}\">\n{}\n</skill-execution>",
        result.rendered_prompt
    );

    if !result.allowed_tools.is_empty() || !result.disallowed_tools.is_empty() {
        response.push_str("\n\nNote: This skill has tool restrictions. ");
        if !result.allowed_tools.is_empty() {
            response.push_str(&format!(
                "Allowed tools: {}. ",
                result.allowed_tools.join(", ")
            ));
        }
        if !result.disallowed_tools.is_empty() {
            response.push_str(&format!(
                "Do NOT use: {}.",
                result.disallowed_tools.join(", ")
            ));
        }
    }

    Ok(response)
}
