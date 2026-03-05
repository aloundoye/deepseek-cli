use anyhow::{Result, anyhow};
use codingbuddy_store::{BackgroundJobRecord, SessionTodoRecord, Store, SubagentRunRecord};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::path::Path;
use uuid::Uuid;

use crate::TasksCmd;
use crate::commands::plan::{current_plan_payload, plan_state_label, workflow_phase_label};
use crate::output::*;

pub(crate) struct TasksSlashResponse {
    pub payload: Value,
    pub text: String,
    pub session_switch: Option<Uuid>,
}

fn scoped_session_id(store: &Store, session_override: Option<Uuid>) -> Result<Option<Uuid>> {
    if let Some(session_id) = session_override {
        store
            .load_session(session_id)?
            .ok_or_else(|| anyhow!("session not found: {session_id}"))?;
        return Ok(Some(session_id));
    }
    Ok(store
        .load_latest_session()?
        .map(|session| session.session_id))
}

fn session_id_from_artifact_path(path: Option<&str>) -> Option<Uuid> {
    let raw = path?.strip_prefix("session://")?;
    Uuid::parse_str(raw).ok()
}

fn background_reason(job: &BackgroundJobRecord) -> Option<String> {
    serde_json::from_str::<Value>(&job.metadata_json)
        .ok()
        .and_then(|value| {
            value
                .get("reason")
                .and_then(|raw| raw.as_str())
                .map(str::to_string)
        })
}

fn todos_summary_payload(todos: &[SessionTodoRecord]) -> Value {
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
            json!({
                "todo_id": todo.todo_id.to_string(),
                "content": todo.content,
                "status": todo.status,
                "position": todo.position,
            })
        });
    json!({
        "total": todos.len(),
        "active": active,
        "completed": completed,
        "in_progress": in_progress,
        "current": current,
    })
}

fn current_plan_step_payload(plan: Option<&Value>) -> Value {
    let Some(plan) = plan else {
        return Value::Null;
    };
    let Some(steps) = plan.get("steps").and_then(Value::as_array) else {
        return Value::Null;
    };
    for (index, step) in steps.iter().enumerate() {
        if !step.get("done").and_then(Value::as_bool).unwrap_or(false) {
            return json!({
                "index": index + 1,
                "step_id": step.get("step_id").and_then(Value::as_str),
                "title": step.get("title").and_then(Value::as_str).unwrap_or_default(),
                "intent": step.get("intent").and_then(Value::as_str).unwrap_or_default(),
            });
        }
    }
    Value::Null
}

fn task_output_text(
    task: &codingbuddy_store::TaskQueueRecord,
    run: Option<&SubagentRunRecord>,
    background_job: Option<&BackgroundJobRecord>,
) -> Option<(String, String)> {
    if let Some(run) = run {
        if let Some(output) = run.output.as_deref() {
            return Some(("subagent".to_string(), output.to_string()));
        }
        if let Some(error) = run.error.as_deref() {
            return Some(("subagent_error".to_string(), error.to_string()));
        }
    }
    if let Some(outcome) = task.outcome.as_deref() {
        return Some(("task".to_string(), outcome.to_string()));
    }
    let metadata = background_job
        .and_then(|job| serde_json::from_str::<Value>(&job.metadata_json).ok())
        .unwrap_or_else(|| json!({}));
    if let Some(result) = metadata.get("result").and_then(|raw| raw.as_str()) {
        return Some(("background".to_string(), result.to_string()));
    }
    if let Some(error) = metadata.get("error").and_then(|raw| raw.as_str()) {
        return Some(("background_error".to_string(), error.to_string()));
    }
    None
}

pub(crate) fn mission_control_payload(
    cwd: &Path,
    session_override: Option<Uuid>,
    limit: usize,
) -> Result<Value> {
    let store = Store::new(cwd)?;
    let session_id = scoped_session_id(&store, session_override)?;
    let session = session_id.and_then(|id| store.load_session(id).ok().flatten());
    let tasks = store.list_tasks(session_id)?;
    let todos = if let Some(session_id) = session_id {
        store.list_session_todos(session_id)?
    } else {
        Vec::new()
    };
    let subagents = store.list_subagent_runs(session_id, limit)?;
    let mut background_jobs = Vec::new();
    let mut seen_jobs = HashSet::new();
    for run in &subagents {
        let Some(job_id) = run.background_job_id else {
            continue;
        };
        if !seen_jobs.insert(job_id) {
            continue;
        }
        if let Some(job) = store.load_background_job(job_id)? {
            background_jobs.push(json!({
                "job_id": job.job_id.to_string(),
                "kind": job.kind,
                "reference": job.reference,
                "status": job.status,
                "reason": background_reason(&job),
                "updated_at": job.updated_at,
                "run_status": run.status,
                "task_id": run.task_id.map(|id| id.to_string()),
            }));
        }
    }

    let queued_tasks = tasks
        .iter()
        .filter(|task| task.status.eq_ignore_ascii_case("queued"))
        .count();
    let running_tasks = tasks
        .iter()
        .filter(|task| task.status.eq_ignore_ascii_case("running"))
        .count();
    let completed_tasks = tasks
        .iter()
        .filter(|task| task.status.eq_ignore_ascii_case("completed"))
        .count();
    let failed_tasks = tasks
        .iter()
        .filter(|task| task.status.eq_ignore_ascii_case("failed"))
        .count();
    let running_subagents = subagents
        .iter()
        .filter(|run| run.status.eq_ignore_ascii_case("running"))
        .count();
    let failed_subagents = subagents
        .iter()
        .filter(|run| run.status.eq_ignore_ascii_case("failed"))
        .count();
    let running_background_jobs = background_jobs
        .iter()
        .filter(|job| {
            job.get("status")
                .and_then(Value::as_str)
                .is_some_and(|status| status.eq_ignore_ascii_case("running"))
        })
        .count();
    let stopped_background_jobs = background_jobs
        .iter()
        .filter(|job| {
            job.get("status")
                .and_then(Value::as_str)
                .is_some_and(|status| status.eq_ignore_ascii_case("stopped"))
        })
        .count();
    let task_count = tasks.len();
    let todo_summary = todos_summary_payload(&todos);
    let subagent_count = subagents.len();
    let background_job_count = background_jobs.len();
    let workflow_phase = session
        .as_ref()
        .map(|record| workflow_phase_label(&record.status));
    let plan_state = session.as_ref().map(|_| plan_state_label(session.as_ref()));
    let active_plan_id = session
        .as_ref()
        .and_then(|record| record.active_plan_id.map(|id| id.to_string()));
    let active_plan = current_plan_payload(&store, session.as_ref())?;
    let current_step = current_plan_step_payload(active_plan.as_ref());

    Ok(json!({
        "schema": "deepseek.chat.mission_control.v1",
        "session_id": session_id.map(|id| id.to_string()),
        "workflow_phase": workflow_phase,
        "plan_state": plan_state,
        "active_plan_id": active_plan_id,
        "plan": active_plan.unwrap_or(Value::Null),
        "current_step": current_step,
        "todos": todos,
        "tasks": tasks,
        "subagents": subagents,
        "background_jobs": background_jobs,
        "summary": {
            "todo_count": todo_summary["total"].as_u64().unwrap_or(0),
            "active_todos": todo_summary["active"].as_u64().unwrap_or(0),
            "completed_todos": todo_summary["completed"].as_u64().unwrap_or(0),
            "in_progress_todos": todo_summary["in_progress"].as_u64().unwrap_or(0),
            "current_todo": todo_summary["current"].clone(),
            "task_count": task_count,
            "queued_tasks": queued_tasks,
            "running_tasks": running_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "subagent_count": subagent_count,
            "running_subagents": running_subagents,
            "failed_subagents": failed_subagents,
            "background_job_count": background_job_count,
            "running_background_jobs": running_background_jobs,
            "stopped_background_jobs": stopped_background_jobs,
        }
    }))
}

pub(crate) fn render_mission_control_lines(payload: &Value) -> Vec<String> {
    let todos = payload
        .get("todos")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let tasks = payload
        .get("tasks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let subagents = payload
        .get("subagents")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let background_jobs = payload
        .get("background_jobs")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let workflow_phase = payload
        .get("workflow_phase")
        .and_then(Value::as_str)
        .unwrap_or("idle");
    let plan_state = payload
        .get("plan_state")
        .and_then(Value::as_str)
        .unwrap_or("none");
    let plan = payload.get("plan").filter(|value| !value.is_null());
    let session_id = payload
        .get("session_id")
        .and_then(Value::as_str)
        .unwrap_or("pending");
    let current_step = payload.get("current_step").filter(|value| !value.is_null());
    let summary = payload.get("summary").cloned().unwrap_or_else(|| json!({}));
    let mut lines = vec![format!(
        "Mission Control (agent queue + session todos): session={} phase={} plan={} {} task(s), {} subagent run(s), {} background job(s)",
        session_id,
        workflow_phase,
        plan_state,
        tasks.len(),
        subagents.len(),
        background_jobs.len()
    )];
    lines.push(format!(
        "- Summary: tasks queued={} running={} completed={} failed={} | subagents running={} failed={} | background running={} stopped={}",
        summary["queued_tasks"].as_u64().unwrap_or(0),
        summary["running_tasks"].as_u64().unwrap_or(0),
        summary["completed_tasks"].as_u64().unwrap_or(0),
        summary["failed_tasks"].as_u64().unwrap_or(0),
        summary["running_subagents"].as_u64().unwrap_or(0),
        summary["failed_subagents"].as_u64().unwrap_or(0),
        summary["running_background_jobs"].as_u64().unwrap_or(0),
        summary["stopped_background_jobs"].as_u64().unwrap_or(0),
    ));
    lines.push(format!(
        "- Todos: active={} in_progress={} completed={}",
        summary["active_todos"].as_u64().unwrap_or(0),
        summary["in_progress_todos"].as_u64().unwrap_or(0),
        summary["completed_todos"].as_u64().unwrap_or(0),
    ));
    if let Some(current_todo) = summary
        .get("current_todo")
        .and_then(Value::as_object)
        .map(|todo| {
            format!(
                "{} [{}]",
                todo.get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                todo.get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("pending")
            )
        })
    {
        lines.push(format!("- Current todo: {current_todo}"));
    }
    if let Some(step) = current_step {
        lines.push(format!(
            "- Current step: {}",
            step.get("title")
                .and_then(Value::as_str)
                .unwrap_or_default()
        ));
    }
    if let Some(plan) = plan {
        lines.push(format!(
            "- Plan: state={} version={} steps={} goal={}",
            plan_state,
            plan["version"].as_u64().unwrap_or(0),
            plan["steps_count"].as_u64().unwrap_or(0),
            plan["goal_preview"].as_str().unwrap_or_default(),
        ));
        if plan_state == "awaiting_approval" {
            lines.push(
                "- Plan next: /plan show | /plan approve | /plan reject <feedback>".to_string(),
            );
        }
    }
    if tasks.is_empty() {
        lines.push("- Tasks: none".to_string());
    } else {
        lines.push("- Tasks:".to_string());
        for task in tasks.iter().take(10) {
            let title = task.get("title").and_then(Value::as_str).unwrap_or("task");
            let status = task
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let priority = task
                .get("priority")
                .and_then(Value::as_u64)
                .unwrap_or_default();
            let task_id = task
                .get("task_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            lines.push(format!(
                "  - {title} [{status}] priority={priority} {task_id}"
            ));
        }
    }
    if todos.is_empty() {
        lines.push("- Session todos: none".to_string());
    } else {
        lines.push("- Session todos:".to_string());
        for todo in todos.iter().take(12) {
            let content = todo
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("todo");
            let status = todo
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("pending");
            let todo_id = todo
                .get("todo_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            lines.push(format!("  - [{status}] {content} {todo_id}"));
        }
    }
    if subagents.is_empty() {
        lines.push("- Subagents: none".to_string());
    } else {
        lines.push("- Subagents:".to_string());
        for run in subagents.iter().take(10) {
            let name = run
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("subagent");
            let status = run
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let run_id = run
                .get("run_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            lines.push(format!("  - {name} [{status}] {run_id}"));
        }
    }
    if background_jobs.is_empty() {
        lines.push("- Background: none".to_string());
    } else {
        lines.push("- Background:".to_string());
        for job in background_jobs.iter().take(10) {
            let job_id = job
                .get("job_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let status = job
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let run_status = job
                .get("run_status")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let reason = job.get("reason").and_then(Value::as_str).unwrap_or("-");
            lines.push(format!(
                "  - {job_id} [{status}] run={run_status} reason={reason}"
            ));
        }
    }
    lines.push(
        "Use /todos for session checklist updates and /comment-todos for source TODO/FIXME scans."
            .to_string(),
    );
    lines.push("Use /tasks show <task_id> for one task, /tasks output <task_id> for full output, or /tasks resume <task_id> to switch into a child session.".to_string());
    lines
}

pub(crate) fn render_mission_control_payload(payload: &Value) -> String {
    render_mission_control_lines(payload).join("\n")
}

pub(crate) fn task_detail_payload(cwd: &Path, task_id: Uuid) -> Result<Value> {
    let store = Store::new(cwd)?;
    let task = store
        .load_task(task_id)?
        .ok_or_else(|| anyhow!("task not found: {task_id}"))?;
    let run = store.load_subagent_run_for_task(task_id)?;
    let background_job = run
        .as_ref()
        .and_then(|record| record.background_job_id)
        .map(|job_id| store.load_background_job(job_id))
        .transpose()?
        .flatten();
    let artifacts = store.list_artifacts_for_task(task_id)?;
    let resume_session_id = run
        .as_ref()
        .and_then(|record| record.child_session_id)
        .or_else(|| session_id_from_artifact_path(task.artifact_path.as_deref()));
    let (output_source, output_text) =
        task_output_text(&task, run.as_ref(), background_job.as_ref())
            .map(|(source, text)| (Some(source), Some(text)))
            .unwrap_or((None, None));

    Ok(json!({
        "schema": "deepseek.tasks.detail.v1",
        "task": task,
        "run": run,
        "background_job": background_job,
        "artifacts": artifacts,
        "resume_session_id": resume_session_id.map(|id| id.to_string()),
        "resume_command": resume_session_id.map(|id| format!("codingbuddy --resume {id}")),
        "output_source": output_source,
        "output_text": output_text,
    }))
}

pub(crate) fn render_task_detail_payload(payload: &Value) -> String {
    let task = payload.get("task").cloned().unwrap_or_else(|| json!({}));
    let run = payload.get("run").cloned().unwrap_or(Value::Null);
    let background_job = payload
        .get("background_job")
        .cloned()
        .unwrap_or(Value::Null);
    let artifacts = payload
        .get("artifacts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut lines = vec![format!(
        "Task {}",
        task.get("task_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
    )];
    lines.push(format!(
        "Title:    {}",
        task.get("title").and_then(Value::as_str).unwrap_or("task")
    ));
    if let Some(description) = task.get("description").and_then(Value::as_str) {
        lines.push(format!("Desc:     {description}"));
    }
    lines.push(format!(
        "Status:   {}",
        task.get("status")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
    ));
    lines.push(format!(
        "Priority: {}",
        task.get("priority")
            .and_then(Value::as_u64)
            .unwrap_or_default()
    ));
    lines.push(format!(
        "Session:  {}",
        task.get("session_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
    ));
    if let Some(outcome) = task.get("outcome").and_then(Value::as_str) {
        lines.push(format!("Outcome:  {outcome}"));
    }
    if let Some(path) = task.get("artifact_path").and_then(Value::as_str) {
        lines.push(format!("Artifact: {path}"));
    }
    if let Some(run_id) = run.get("run_id").and_then(Value::as_str) {
        lines.push(format!(
            "Run:      {} [{}]",
            run_id,
            run.get("status")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
        ));
        if let Some(goal) = run.get("goal").and_then(Value::as_str) {
            lines.push(format!("Goal:     {goal}"));
        }
        if let Some(child_session_id) = run.get("child_session_id").and_then(Value::as_str) {
            lines.push(format!("Child:    {child_session_id}"));
        }
        if let Some(error) = run.get("error").and_then(Value::as_str) {
            lines.push(format!("Error:    {error}"));
        }
    }
    if let Some(job_id) = background_job.get("job_id").and_then(Value::as_str) {
        lines.push(format!(
            "Bg Job:   {} [{}]",
            job_id,
            background_job
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
        ));
        if let Some(reason) = background_job
            .get("metadata_json")
            .and_then(Value::as_str)
            .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
            .and_then(|value| {
                value
                    .get("reason")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
        {
            lines.push(format!("Reason:   {reason}"));
        }
    }
    if !artifacts.is_empty() {
        lines.push("Artifacts:".to_string());
        for artifact in artifacts.iter().take(10) {
            let path = artifact
                .get("artifact_path")
                .and_then(Value::as_str)
                .unwrap_or_default();
            lines.push(format!("  - {path}"));
        }
    }
    if let Some(session_id) = payload.get("resume_session_id").and_then(Value::as_str) {
        lines.push(format!("Resume:   /resume {session_id}"));
    }
    if let Some(source) = payload.get("output_source").and_then(Value::as_str) {
        let output = payload
            .get("output_text")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let preview = if output.len() > 240 {
            format!("{}...", &output[..output.floor_char_boundary(240)])
        } else {
            output.to_string()
        };
        lines.push(format!("Output:   [{source}] {preview}"));
    }
    lines.push(format!(
        "Created:  {}",
        task.get("created_at")
            .and_then(Value::as_str)
            .unwrap_or_default()
    ));
    lines.push(format!(
        "Updated:  {}",
        task.get("updated_at")
            .and_then(Value::as_str)
            .unwrap_or_default()
    ));
    lines.join("\n")
}

pub(crate) fn task_output_payload(cwd: &Path, task_id: Uuid) -> Result<Value> {
    let payload = task_detail_payload(cwd, task_id)?;
    let output_text = payload
        .get("output_text")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("task has no recorded output: {task_id}"))?;
    Ok(json!({
        "schema": "deepseek.tasks.output.v1",
        "task_id": task_id.to_string(),
        "output_source": payload.get("output_source").and_then(Value::as_str).unwrap_or("task"),
        "output_text": output_text,
        "resume_session_id": payload.get("resume_session_id").and_then(Value::as_str),
    }))
}

pub(crate) fn render_task_output_payload(payload: &Value) -> String {
    let task_id = payload
        .get("task_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let source = payload
        .get("output_source")
        .and_then(Value::as_str)
        .unwrap_or("task");
    let output_text = payload
        .get("output_text")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut lines = vec![format!("Task Output: {task_id} [{source}]")];
    if let Some(session_id) = payload.get("resume_session_id").and_then(Value::as_str) {
        lines.push(format!("Resume: /resume {session_id}"));
    }
    lines.push(String::new());
    lines.push(output_text.to_string());
    lines.join("\n")
}

pub(crate) fn resumable_session_for_task(cwd: &Path, task_id: Uuid) -> Result<Option<Uuid>> {
    let payload = task_detail_payload(cwd, task_id)?;
    Ok(payload
        .get("resume_session_id")
        .and_then(Value::as_str)
        .and_then(|raw| Uuid::parse_str(raw).ok()))
}

pub(crate) fn handle_tasks_slash(
    cwd: &Path,
    args: &[String],
    session_override: Option<Uuid>,
) -> Result<TasksSlashResponse> {
    let subcommand = args
        .first()
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_else(|| "list".to_string());
    match subcommand.as_str() {
        "list" => {
            let payload = mission_control_payload(cwd, session_override, 20)?;
            let text = render_mission_control_payload(&payload);
            Ok(TasksSlashResponse {
                payload,
                text,
                session_switch: None,
            })
        }
        "show" => {
            let id = args
                .get(1)
                .ok_or_else(|| anyhow!("usage: /tasks show <task_id>"))?;
            let task_id = Uuid::parse_str(id)?;
            let payload = task_detail_payload(cwd, task_id)?;
            let text = render_task_detail_payload(&payload);
            Ok(TasksSlashResponse {
                payload,
                text,
                session_switch: None,
            })
        }
        "output" => {
            let id = args
                .get(1)
                .ok_or_else(|| anyhow!("usage: /tasks output <task_id>"))?;
            let task_id = Uuid::parse_str(id)?;
            let payload = task_output_payload(cwd, task_id)?;
            let text = render_task_output_payload(&payload);
            Ok(TasksSlashResponse {
                payload,
                text,
                session_switch: None,
            })
        }
        "resume" => {
            let id = args
                .get(1)
                .ok_or_else(|| anyhow!("usage: /tasks resume <task_id>"))?;
            let task_id = Uuid::parse_str(id)?;
            let session_id = resumable_session_for_task(cwd, task_id)?
                .ok_or_else(|| anyhow!("task has no resumable child session: {task_id}"))?;
            let payload = json!({
                "schema": "deepseek.tasks.resume.v1",
                "task_id": task_id.to_string(),
                "session_id": session_id.to_string(),
                "resume_command": format!("codingbuddy --resume {session_id}"),
                "message": format!("switched active chat session to child session {session_id} for task {task_id}"),
            });
            let text = payload
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("task session selected")
                .to_string();
            Ok(TasksSlashResponse {
                payload,
                text,
                session_switch: Some(session_id),
            })
        }
        "help" => {
            let payload = json!({
                "schema": "deepseek.tasks.help.v1",
                "usage": [
                    "/tasks",
                    "/tasks list",
                    "/tasks show <task_id>",
                    "/tasks output <task_id>",
                    "/tasks resume <task_id>",
                ],
                "note": "/todos inspects session-native agent checklist; /comment-todos scans source comments; /tasks inspects delegated work tracking."
            });
            let text = "Usage: /tasks [list|show <task_id>|output <task_id>|resume <task_id>]\n/todos inspects session-native agent checklist.\n/comment-todos scans source comments.\n/tasks inspects delegated work tracking.".to_string();
            Ok(TasksSlashResponse {
                payload,
                text,
                session_switch: None,
            })
        }
        _ => Err(anyhow!(
            "use /tasks [list|show <task_id>|output <task_id>|resume <task_id>]"
        )),
    }
}

pub(crate) fn run_tasks(cwd: &Path, command: TasksCmd, json_mode: bool) -> Result<()> {
    let store = Store::new(cwd)?;
    match command {
        TasksCmd::List => {
            let tasks = store.list_tasks(None)?;
            let session_id = store
                .load_latest_session()?
                .map(|session| session.session_id);
            let subagents = store.list_subagent_runs(session_id, 20)?;
            if json_mode {
                print_json(&json!({"tasks": tasks, "subagents": subagents}))?;
            } else if tasks.is_empty() {
                println!("No tasks in queue.");
                if !subagents.is_empty() {
                    println!("\nRecent subagents:");
                    for run in &subagents {
                        println!("- {} [{}] {}", run.name, run.status, run.run_id);
                    }
                }
            } else {
                println!("{:<36}  {:<10}  {:<4}  TITLE", "ID", "STATUS", "PRI");
                println!("{}", "-".repeat(80));
                for task in &tasks {
                    println!(
                        "{:<36}  {:<10}  {:<4}  {}",
                        task.task_id, task.status, task.priority, task.title
                    );
                }
                println!("\n{} task(s) total.", tasks.len());
                if !subagents.is_empty() {
                    println!("\nRecent subagents:");
                    for run in &subagents {
                        println!("- {} [{}] {}", run.name, run.status, run.run_id);
                    }
                }
            }
        }
        TasksCmd::Show(args) => {
            let task_id = Uuid::parse_str(&args.id)?;
            let payload = task_detail_payload(cwd, task_id)?;
            if json_mode {
                print_json(&payload)?;
            } else {
                println!("{}", render_task_detail_payload(&payload));
            }
        }
        TasksCmd::Output(args) => {
            let task_id = Uuid::parse_str(&args.id)?;
            let payload = task_output_payload(cwd, task_id)?;
            if json_mode {
                print_json(&payload)?;
            } else {
                println!("{}", render_task_output_payload(&payload));
            }
        }
        TasksCmd::Cancel(args) => {
            let task_id = Uuid::parse_str(&args.id)?;
            store.update_task_status(task_id, "cancelled", Some("cancelled by user"))?;
            if json_mode {
                print_json(&json!({"task_id": args.id, "status": "cancelled"}))?;
            } else {
                println!("Task {task_id} cancelled.");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use codingbuddy_core::{Plan, PlanStep, SessionState};
    use codingbuddy_store::{
        BackgroundJobRecord, SessionTodoRecord, SubagentRunRecord, TaskQueueRecord,
    };
    use tempfile::tempdir;

    #[test]
    fn mission_control_payload_includes_session_todos_and_current_step() -> Result<()> {
        let temp = tempdir()?;
        let store = Store::new(temp.path())?;
        let plan_id = Uuid::now_v7();
        let session = codingbuddy_core::Session {
            session_id: Uuid::now_v7(),
            workspace_root: temp.path().display().to_string(),
            baseline_commit: None,
            status: SessionState::ExecutingStep,
            budgets: codingbuddy_core::SessionBudgets {
                per_turn_seconds: 30,
                max_think_tokens: 4096,
            },
            active_plan_id: Some(plan_id),
        };
        store.save_session(&session)?;
        store.save_plan(
            session.session_id,
            &Plan {
                plan_id,
                version: 1,
                goal: "Stabilize workspace automation".to_string(),
                assumptions: vec![],
                steps: vec![
                    PlanStep {
                        step_id: Uuid::now_v7(),
                        title: "Audit current signals".to_string(),
                        intent: "Confirm failure modes".to_string(),
                        tools: vec![],
                        files: vec![],
                        done: true,
                    },
                    PlanStep {
                        step_id: Uuid::now_v7(),
                        title: "Patch windows-specific path handling".to_string(),
                        intent: "Fix platform parity".to_string(),
                        tools: vec![],
                        files: vec!["crates/codingbuddy-agent/src/apply.rs".to_string()],
                        done: false,
                    },
                ],
                verification: vec!["cargo test -p codingbuddy-agent --lib".to_string()],
                risk_notes: vec![],
            },
        )?;
        let now = Utc::now().to_rfc3339();
        store.replace_session_todos(
            session.session_id,
            &[
                SessionTodoRecord {
                    todo_id: Uuid::now_v7(),
                    session_id: session.session_id,
                    content: "Audit current signals".to_string(),
                    status: "completed".to_string(),
                    position: 0,
                    created_at: now.clone(),
                    updated_at: now.clone(),
                },
                SessionTodoRecord {
                    todo_id: Uuid::now_v7(),
                    session_id: session.session_id,
                    content: "Patch windows-specific path handling".to_string(),
                    status: "in_progress".to_string(),
                    position: 1,
                    created_at: now.clone(),
                    updated_at: now,
                },
            ],
        )?;

        let payload = mission_control_payload(temp.path(), Some(session.session_id), 20)?;
        assert_eq!(payload["summary"]["todo_count"].as_u64(), Some(2));
        assert_eq!(payload["summary"]["completed_todos"].as_u64(), Some(1));
        assert_eq!(payload["summary"]["in_progress_todos"].as_u64(), Some(1));
        assert_eq!(
            payload["summary"]["current_todo"]["content"].as_str(),
            Some("Patch windows-specific path handling")
        );
        assert_eq!(
            payload["current_step"]["title"].as_str(),
            Some("Patch windows-specific path handling")
        );
        assert_eq!(payload["todos"].as_array().map(Vec::len), Some(2));

        let rendered = render_mission_control_payload(&payload);
        assert!(rendered.contains("Mission Control (agent queue + session todos)"));
        assert!(rendered.contains("Current todo:"));
        assert!(rendered.contains("Current step: Patch windows-specific path handling"));
        assert!(rendered.contains("/comment-todos"));
        Ok(())
    }

    #[test]
    fn task_detail_payload_surfaces_resume_session_and_output() -> Result<()> {
        let temp = tempdir()?;
        let store = Store::new(temp.path())?;
        let session = codingbuddy_core::Session {
            session_id: Uuid::now_v7(),
            workspace_root: temp.path().display().to_string(),
            baseline_commit: None,
            status: SessionState::ExecutingStep,
            budgets: codingbuddy_core::SessionBudgets {
                per_turn_seconds: 30,
                max_think_tokens: 4096,
            },
            active_plan_id: None,
        };
        store.save_session(&session)?;
        let child_session = store.fork_session(session.session_id)?;
        let now = Utc::now().to_rfc3339();
        let task_id = Uuid::now_v7();
        let run_id = Uuid::now_v7();
        let job_id = Uuid::now_v7();
        store.insert_task(&TaskQueueRecord {
            task_id,
            session_id: session.session_id,
            title: "Implement task view".to_string(),
            description: Some("inspect linked output".to_string()),
            priority: 2,
            status: "completed".to_string(),
            outcome: None,
            artifact_path: Some(format!("session://{}", child_session.session_id)),
            created_at: now.clone(),
            updated_at: now.clone(),
        })?;
        store.upsert_subagent_run(&SubagentRunRecord {
            run_id,
            session_id: Some(session.session_id),
            task_id: Some(task_id),
            child_session_id: Some(child_session.session_id),
            background_job_id: Some(job_id),
            name: "plan".to_string(),
            goal: "inspect".to_string(),
            status: "completed".to_string(),
            output: Some("task finished".to_string()),
            error: None,
            created_at: now.clone(),
            updated_at: now.clone(),
        })?;
        store.upsert_background_job(&BackgroundJobRecord {
            job_id,
            kind: "subagent".to_string(),
            reference: format!("subagent:{run_id}"),
            status: "stopped".to_string(),
            metadata_json: json!({"reason":"completed"}).to_string(),
            started_at: now.clone(),
            updated_at: now,
        })?;

        let payload = task_detail_payload(temp.path(), task_id)?;
        let child_session_id = child_session.session_id.to_string();
        assert_eq!(
            payload["resume_session_id"].as_str(),
            Some(child_session_id.as_str())
        );
        assert_eq!(payload["output_source"].as_str(), Some("subagent"));
        assert_eq!(payload["output_text"].as_str(), Some("task finished"));
        Ok(())
    }

    #[test]
    fn handle_tasks_slash_resume_switches_to_child_session() -> Result<()> {
        let temp = tempdir()?;
        let store = Store::new(temp.path())?;
        let session = codingbuddy_core::Session {
            session_id: Uuid::now_v7(),
            workspace_root: temp.path().display().to_string(),
            baseline_commit: None,
            status: SessionState::ExecutingStep,
            budgets: codingbuddy_core::SessionBudgets {
                per_turn_seconds: 30,
                max_think_tokens: 4096,
            },
            active_plan_id: None,
        };
        store.save_session(&session)?;
        let child_session = store.fork_session(session.session_id)?;
        let now = Utc::now().to_rfc3339();
        let task_id = Uuid::now_v7();
        let run_id = Uuid::now_v7();
        store.insert_task(&TaskQueueRecord {
            task_id,
            session_id: session.session_id,
            title: "Resume subagent".to_string(),
            description: Some("switch into child session".to_string()),
            priority: 1,
            status: "completed".to_string(),
            outcome: Some("ready".to_string()),
            artifact_path: Some(format!("session://{}", child_session.session_id)),
            created_at: now.clone(),
            updated_at: now.clone(),
        })?;
        store.upsert_subagent_run(&SubagentRunRecord {
            run_id,
            session_id: Some(session.session_id),
            task_id: Some(task_id),
            child_session_id: Some(child_session.session_id),
            background_job_id: None,
            name: "resume".to_string(),
            goal: "switch sessions".to_string(),
            status: "completed".to_string(),
            output: Some("ready".to_string()),
            error: None,
            created_at: now.clone(),
            updated_at: now,
        })?;

        let response = handle_tasks_slash(
            temp.path(),
            &["resume".to_string(), task_id.to_string()],
            Some(session.session_id),
        )?;

        let child_session_id = child_session.session_id.to_string();
        assert_eq!(response.session_switch, Some(child_session.session_id));
        assert_eq!(
            response.payload["session_id"].as_str(),
            Some(child_session_id.as_str())
        );
        assert!(
            response.text.contains(&child_session_id),
            "resume text should reference the child session"
        );
        Ok(())
    }
}
