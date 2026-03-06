use super::*;

pub(super) fn agents_payload(cwd: &Path, limit: usize) -> Result<serde_json::Value> {
    let store = Store::new(cwd)?;
    let session_id = store
        .load_latest_session()?
        .map(|session| session.session_id);
    let runs = store.list_subagent_runs(session_id, limit)?;
    Ok(json!({
        "schema": "deepseek.chat.agents.v1",
        "session_id": session_id.map(|id| id.to_string()),
        "count": runs.len(),
        "agents": runs,
    }))
}

pub(super) fn render_agents_payload(payload: &serde_json::Value) -> String {
    let runs = payload
        .get("agents")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    if runs.is_empty() {
        return "No subagent runs recorded in this session.".to_string();
    }
    let parsed_runs = runs
        .into_iter()
        .filter_map(|row| serde_json::from_value::<SubagentRunRecord>(row).ok())
        .collect::<Vec<_>>();
    if parsed_runs.is_empty() {
        return "No subagent runs recorded in this session.".to_string();
    }
    let mut lines = vec![format!("Subagents ({} total):", parsed_runs.len())];
    for run in parsed_runs {
        let detail = run
            .output
            .as_deref()
            .or(run.error.as_deref())
            .unwrap_or_default()
            .replace('\n', " ");
        let detail = truncate_inline(&detail, 120);
        lines.push(format!(
            "- {} [{}] {} — {}",
            run.name, run.status, run.run_id, detail
        ));
    }
    lines.join("\n")
}

pub(super) fn render_todos_payload(payload: &serde_json::Value) -> String {
    let count = payload
        .get("count")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let session_id = payload
        .get("session_id")
        .and_then(|value| value.as_str())
        .unwrap_or("pending");
    let workflow_phase = payload
        .get("workflow_phase")
        .and_then(|value| value.as_str())
        .unwrap_or("idle");
    let plan_state = payload
        .get("plan_state")
        .and_then(|value| value.as_str())
        .unwrap_or("none");
    let summary = payload.get("summary").cloned().unwrap_or_else(|| json!({}));
    let mut lines = vec![
        format!(
            "Session todos: {count} item(s) — session={session_id} phase={workflow_phase} plan={plan_state}"
        ),
        format!(
            "Active={} In progress={} Completed={}",
            summary["active"].as_u64().unwrap_or(0),
            summary["in_progress"].as_u64().unwrap_or(0),
            summary["completed"].as_u64().unwrap_or(0),
        ),
    ];
    if let Some(current) = summary
        .get("current")
        .and_then(|value| value.as_object())
        .map(|obj| {
            format!(
                "{} [{}]",
                obj.get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default(),
                obj.get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("pending")
            )
        })
    {
        lines.push(format!("Current todo: {current}"));
    }
    if let Some(step) = payload
        .get("current_step")
        .and_then(|value| value.as_object())
    {
        lines.push(format!(
            "Current plan step: {}",
            step.get("title")
                .and_then(|value| value.as_str())
                .unwrap_or("none")
        ));
    }
    if let Some(rows) = payload.get("items").and_then(|value| value.as_array()) {
        for row in rows.iter().take(30) {
            lines.push(format!(
                "- [{}] {} ({})",
                row["status"].as_str().unwrap_or("pending"),
                row["content"].as_str().unwrap_or_default(),
                row["todo_id"].as_str().unwrap_or_default()
            ));
        }
    }
    lines.push("Use /comment-todos to scan TODO/FIXME comments in source files.".to_string());
    lines.join("\n")
}

pub(super) fn render_comment_todos_payload(payload: &serde_json::Value) -> String {
    let count = payload
        .get("count")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let mut lines = vec![
        format!("Workspace comment scan: {count} result(s)"),
        "This is source-comment scanning only. Use /todos for session-native agent checklist tracking."
            .to_string(),
    ];
    if let Some(rows) = payload.get("items").and_then(|value| value.as_array()) {
        for row in rows.iter().take(20) {
            lines.push(format!(
                "- {}:{} {}",
                row["path"].as_str().unwrap_or_default(),
                row["line"].as_u64().unwrap_or(0),
                row["text"].as_str().unwrap_or_default()
            ));
        }
    }
    lines.join("\n")
}

pub(super) fn session_focus_payload(cwd: &Path, session_id: Uuid) -> Result<serde_json::Value> {
    let store = Store::new(cwd)?;
    let session = store
        .load_session(session_id)?
        .ok_or_else(|| anyhow!("session not found: {session_id}"))?;
    let projection = store.rebuild_from_events(session.session_id)?;
    Ok(json!({
        "schema": "deepseek.chat.resume.v1",
        "session_id": session.session_id.to_string(),
        "state": format!("{:?}", session.status),
        "turns": projection.transcript.len(),
        "steps": projection.step_status.len(),
        "message": format!(
            "switched active chat session to {} ({} turns, state={:?})",
            session.session_id,
            projection.transcript.len(),
            session.status
        ),
    }))
}

pub(super) type SessionLifecycleNotice = chat_lifecycle::SessionLifecycleNotice;

pub(super) fn poll_session_lifecycle_notices(
    cwd: &Path,
    session_override: Option<Uuid>,
    watermarks: &mut HashMap<Uuid, u64>,
) -> Result<Vec<SessionLifecycleNotice>> {
    chat_lifecycle::poll_session_lifecycle_notices(cwd, session_override, watermarks)
}

pub(super) fn pr_comments_payload(
    cwd: &Path,
    pr_number: &str,
    output_path: Option<&str>,
) -> Result<serde_json::Value> {
    let gh_available = std::process::Command::new("gh")
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false);
    if !gh_available {
        return Err(anyhow!(
            "GitHub CLI ('gh') is required for /pr_comments. Install gh and authenticate first."
        ));
    }

    let output = std::process::Command::new("gh")
        .current_dir(cwd)
        .args([
            "pr",
            "view",
            pr_number,
            "--json",
            "number,title,url,author,comments",
        ])
        .output()?;
    if !output.status.success() {
        return Err(anyhow!(
            "failed to fetch PR comments: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let comments_count = parsed["comments"]
        .as_array()
        .map(|rows| rows.len())
        .unwrap_or(0);

    let mut saved_to = None;
    if let Some(path) = output_path {
        let destination = resolve_additional_dir(cwd, path);
        if let Some(parent) = destination.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&destination, serde_json::to_vec_pretty(&parsed)?)?;
        saved_to = Some(destination.to_string_lossy().to_string());
    }

    Ok(json!({
        "schema": "deepseek.pr_comments.v1",
        "ok": true,
        "pr": pr_number,
        "summary": format!("Fetched {} comment(s) for PR #{}", comments_count, pr_number),
        "saved_to": saved_to,
        "data": parsed,
    }))
}

pub(super) fn release_notes_payload(
    cwd: &Path,
    range: &str,
    output_path: Option<&str>,
) -> Result<serde_json::Value> {
    let output = std::process::Command::new("git")
        .current_dir(cwd)
        .args(["log", "--no-merges", "--pretty=format:%h %s", range])
        .output()?;
    if !output.status.success() {
        return Err(anyhow!(
            "failed to generate release notes: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    let mut saved_to = None;
    if let Some(path) = output_path {
        let destination = resolve_additional_dir(cwd, path);
        if let Some(parent) = destination.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let mut rendered = format!("# Release Notes ({range})\n\n");
        for line in &lines {
            rendered.push_str("- ");
            rendered.push_str(line);
            rendered.push('\n');
        }
        std::fs::write(&destination, rendered)?;
        saved_to = Some(destination.to_string_lossy().to_string());
    }

    Ok(json!({
        "schema": "deepseek.release_notes.v1",
        "ok": true,
        "range": range,
        "count": lines.len(),
        "lines": lines,
        "saved_to": saved_to,
    }))
}

pub(super) fn login_payload(cwd: &Path) -> Result<serde_json::Value> {
    let cfg = AppConfig::ensure(cwd)?;
    let env_key = if cfg.llm.api_key_env.trim().is_empty() {
        "DEEPSEEK_API_KEY".to_string()
    } else {
        cfg.llm.api_key_env.clone()
    };
    let token = std::env::var(&env_key).unwrap_or_default();
    if token.trim().is_empty() {
        return Err(anyhow!(
            "missing {}. export the key first, then run /login",
            env_key
        ));
    }

    let runtime_auth = runtime_dir(cwd).join("auth").join("session.json");
    if let Some(parent) = runtime_auth.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mask = format!(
        "***{}",
        token
            .chars()
            .rev()
            .take(4)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
    );
    let session_payload = json!({
        "provider": "deepseek",
        "api_key_env": env_key,
        "masked": mask,
        "created_at": Utc::now().to_rfc3339(),
    });
    std::fs::write(&runtime_auth, serde_json::to_vec_pretty(&session_payload)?)?;

    let local_path = AppConfig::project_local_settings_path(cwd);
    if let Some(parent) = local_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut root = if local_path.exists() {
        let raw = std::fs::read_to_string(&local_path)?;
        serde_json::from_str::<serde_json::Value>(&raw).unwrap_or_else(|_| json!({}))
    } else {
        json!({})
    };
    if !root.is_object() {
        root = json!({});
    }
    let map = root
        .as_object_mut()
        .ok_or_else(|| anyhow!("settings.local.json root must be an object"))?;
    let llm_value = map.entry("llm".to_string()).or_insert_with(|| json!({}));
    if !llm_value.is_object() {
        *llm_value = json!({});
    }
    if let Some(llm) = llm_value.as_object_mut() {
        llm.insert("api_key".to_string(), json!(token));
        llm.insert("api_key_env".to_string(), json!(env_key));
    }
    std::fs::write(&local_path, serde_json::to_vec_pretty(&root)?)?;

    Ok(json!({
        "schema": "deepseek.auth.v1",
        "logged_in": true,
        "session_path": runtime_auth.to_string_lossy().to_string(),
        "settings_path": local_path.to_string_lossy().to_string(),
        "message": "Login successful. Workspace auth session and settings.local.json updated.",
    }))
}

pub(super) fn logout_payload(cwd: &Path) -> Result<serde_json::Value> {
    let _cfg = AppConfig::ensure(cwd)?;
    let runtime_auth = runtime_dir(cwd).join("auth").join("session.json");
    let session_removed = if runtime_auth.exists() {
        std::fs::remove_file(&runtime_auth)?;
        true
    } else {
        false
    };

    let local_path = AppConfig::project_local_settings_path(cwd);
    let mut settings_updated = false;
    if local_path.exists() {
        let raw = std::fs::read_to_string(&local_path)?;
        let mut root =
            serde_json::from_str::<serde_json::Value>(&raw).unwrap_or_else(|_| json!({}));
        if let Some(llm) = root.get_mut("llm").and_then(|entry| entry.as_object_mut())
            && llm.remove("api_key").is_some()
        {
            settings_updated = true;
        }
        std::fs::write(&local_path, serde_json::to_vec_pretty(&root)?)?;
    }

    // NOTE: We no longer mutate the process environment (unsafe data race).
    // The ApiClient caches the key at construction, so clearing the env var
    // would not affect the running session anyway. The settings file update
    // above ensures the key is gone on next launch.

    Ok(json!({
        "schema": "deepseek.auth.v1",
        "logged_in": false,
        "session_removed": session_removed,
        "settings_updated": settings_updated,
        "message": "Logged out. Restart the session to complete logout.",
    }))
}

pub(super) fn desktop_payload(cwd: &Path, _args: &[String]) -> Result<serde_json::Value> {
    let session_id = Store::new(cwd)?
        .load_latest_session()?
        .map(|session| session.session_id.to_string());
    Ok(json!({
        "schema": "deepseek.desktop_handoff.v2",
        "session_id": session_id,
        "resume_command": session_id.map(|id| format!("deepseek --resume {id}")),
    }))
}

/// Generate 2-3 context-aware follow-up prompt suggestions from the assistant response.
pub(super) fn generate_prompt_suggestions(response: &str) -> Vec<String> {
    let lower = response.to_ascii_lowercase();
    let mut suggestions = Vec::new();

    // Detect edits → suggest test/review/commit
    if lower.contains("applied") || lower.contains("modified") || lower.contains("created") {
        suggestions.push("run tests".to_string());
        suggestions.push("/diff".to_string());
        if lower.contains("created") {
            suggestions.push("document this change".to_string());
        }
    }

    // Detect errors → suggest debug/fix
    if lower.contains("error") || lower.contains("failed") || lower.contains("panic") {
        suggestions.push("fix the error".to_string());
        suggestions.push("show the full stack trace".to_string());
    }

    // Detect test results → suggest coverage
    if lower.contains("test") && (lower.contains("passed") || lower.contains("ok")) {
        suggestions.push("check test coverage".to_string());
    }

    // Detect refactoring → suggest verification
    if lower.contains("refactor") || lower.contains("renamed") || lower.contains("moved") {
        suggestions.push("verify no regressions".to_string());
    }

    // Detect explanations → suggest deeper dives
    if lower.contains("because") || lower.contains("reason") || lower.contains("architecture") {
        suggestions.push("explain in more detail".to_string());
    }

    // Always cap at 3 suggestions
    suggestions.truncate(3);

    // Fallback if nothing triggered
    if suggestions.is_empty() {
        suggestions.push("/compact".to_string());
        suggestions.push("/cost".to_string());
    }

    suggestions
}
pub(super) fn todo_summary_payload(items: &[SessionTodoRecord]) -> serde_json::Value {
    let completed = items
        .iter()
        .filter(|item| item.status.eq_ignore_ascii_case("completed"))
        .count();
    let in_progress = items
        .iter()
        .filter(|item| item.status.eq_ignore_ascii_case("in_progress"))
        .count();
    let active = items.len().saturating_sub(completed);
    let current = items
        .iter()
        .find(|item| item.status.eq_ignore_ascii_case("in_progress"))
        .or_else(|| {
            items
                .iter()
                .find(|item| item.status.eq_ignore_ascii_case("pending"))
        })
        .map(|item| {
            json!({
                "todo_id": item.todo_id.to_string(),
                "content": item.content,
                "status": item.status,
                "position": item.position,
            })
        });
    json!({
        "total": items.len(),
        "active": active,
        "completed": completed,
        "in_progress": in_progress,
        "current": current,
    })
}

pub(super) fn current_plan_step_payload(
    plan_payload: &serde_json::Value,
) -> Option<serde_json::Value> {
    let steps = plan_payload.get("steps")?.as_array()?;
    for (index, step) in steps.iter().enumerate() {
        if !step
            .get("done")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            return Some(json!({
                "index": index + 1,
                "step_id": step.get("step_id").and_then(|value| value.as_str()),
                "title": step.get("title").and_then(|value| value.as_str()).unwrap_or_default(),
                "intent": step.get("intent").and_then(|value| value.as_str()).unwrap_or_default(),
            }));
        }
    }
    None
}

pub(super) fn todos_payload(
    cwd: &Path,
    session_override: Option<Uuid>,
    args: &[String],
) -> Result<serde_json::Value> {
    let mut max_results = 200usize;
    let mut query_parts = Vec::new();
    for arg in args {
        if query_parts.is_empty()
            && let Ok(parsed) = arg.parse::<usize>()
        {
            max_results = parsed.clamp(1, 2000);
            continue;
        }
        query_parts.push(arg.clone());
    }
    let query = (!query_parts.is_empty()).then(|| query_parts.join(" "));
    let query_lower = query.as_deref().map(|value| value.to_ascii_lowercase());

    let store = Store::new(cwd)?;
    let session = if let Some(session_id) = session_override {
        Some(
            store
                .load_session(session_id)?
                .ok_or_else(|| anyhow!("session not found: {session_id}"))?,
        )
    } else {
        store.load_latest_session()?
    };
    let Some(session) = session else {
        return Ok(json!({
            "schema": "deepseek.session_todos.v1",
            "session_id": serde_json::Value::Null,
            "workflow_phase": "idle",
            "plan_state": "none",
            "current_step": serde_json::Value::Null,
            "query": query,
            "count": 0,
            "summary": {
                "total": 0,
                "active": 0,
                "completed": 0,
                "in_progress": 0,
                "current": serde_json::Value::Null,
            },
            "items": [],
        }));
    };

    let all_items = store.list_session_todos(session.session_id)?;
    let mut filtered = Vec::new();
    for item in all_items.iter() {
        if let Some(filter) = query_lower.as_deref()
            && !item.content.to_ascii_lowercase().contains(filter)
        {
            continue;
        }
        filtered.push(item.clone());
        if filtered.len() >= max_results {
            break;
        }
    }
    let summary = todo_summary_payload(&all_items);
    let active_plan = current_plan_payload(&store, Some(&session))?;
    let current_step = active_plan
        .as_ref()
        .and_then(current_plan_step_payload)
        .unwrap_or(serde_json::Value::Null);

    Ok(json!({
        "schema": "deepseek.session_todos.v1",
        "session_id": session.session_id.to_string(),
        "workflow_phase": workflow_phase_label(&session.status),
        "plan_state": plan_state_label(Some(&session)),
        "current_step": current_step,
        "query": query,
        "count": filtered.len(),
        "summary": summary,
        "items": filtered,
    }))
}

pub(super) fn comment_todos_payload(cwd: &Path, args: &[String]) -> Result<serde_json::Value> {
    let mut max_results = 100usize;
    let mut query = None;
    if let Some(first) = args.first() {
        if let Ok(parsed) = first.parse::<usize>() {
            max_results = parsed.clamp(1, 2000);
            query = args.get(1).cloned();
        } else {
            query = Some(first.clone());
        }
    }
    let query_lower = query.as_deref().map(|value| value.to_ascii_lowercase());

    let output = std::process::Command::new("rg")
        .current_dir(cwd)
        .args([
            "--line-number",
            "--no-heading",
            "--hidden",
            "--glob",
            "!.git/*",
            "--glob",
            "!target/*",
            "--glob",
            "!node_modules/*",
            "TODO|FIXME",
            ".",
        ])
        .output();

    let mut items = Vec::new();
    if let Ok(out) = output {
        let stdout = String::from_utf8_lossy(&out.stdout);
        for line in stdout.lines() {
            let mut parts = line.splitn(3, ':');
            let path = parts.next().unwrap_or_default();
            let line_no = parts
                .next()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
            let text = parts.next().unwrap_or_default().trim().to_string();
            if let Some(filter) = query_lower.as_deref()
                && !text.to_ascii_lowercase().contains(filter)
            {
                continue;
            }
            items.push(json!({
                "path": path,
                "line": line_no,
                "text": text,
            }));
            if items.len() >= max_results {
                break;
            }
        }
    }

    Ok(json!({
        "schema": "deepseek.comment_todos.v1",
        "count": items.len(),
        "query": query,
        "items": items,
    }))
}

pub(super) fn chrome_payload(cwd: &Path, args: &[String]) -> Result<serde_json::Value> {
    let mut idx = 0usize;
    let subcommand = args
        .first()
        .cloned()
        .unwrap_or_else(|| "status".to_string())
        .to_ascii_lowercase();
    if !args.is_empty() {
        idx = 1;
    }

    let port = std::env::var("CODINGBUDDY_CHROME_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(9222);
    let cfg = AppConfig::load(cwd).unwrap_or_default();
    let mut session = ChromeSession::new(port)?;
    session.set_allow_stub_fallback(cfg.tools.chrome.allow_stub_fallback);
    let debug_url = session.debug_url().to_string();

    match subcommand.as_str() {
        "status" => {
            let status = session.connection_status()?;
            Ok(json!({
                "schema": "deepseek.chrome.v1",
                "action": "status",
                "port": port,
                "connected": status.connected,
                "debug_url": debug_url,
                "status": status,
            }))
        }
        "reconnect" => {
            let status = session.reconnect(true)?;
            Ok(json!({
                "schema": "deepseek.chrome.v1",
                "action": "reconnect",
                "port": port,
                "connected": status.connected,
                "debug_url": debug_url,
                "status": status,
            }))
        }
        "tabs" => match session.list_tabs() {
            Ok(tabs) => Ok(json!({
                "schema": "deepseek.chrome.v1",
                "action": "tabs",
                "port": port,
                "count": tabs.len(),
                "tabs": tabs,
            })),
            Err(err) => Ok(chrome_error_payload("tabs", port, &debug_url, &err)),
        },
        "tab" => {
            let Some(tab_action) = args.get(idx).map(|v| v.to_ascii_lowercase()) else {
                return Err(anyhow!("usage: /chrome tab [new <url>|focus <target_id>]"));
            };
            idx += 1;
            match tab_action.as_str() {
                "new" => {
                    let url = args.get(idx).map(String::as_str).unwrap_or("about:blank");
                    match session.create_tab(url) {
                        Ok(tab) => Ok(json!({
                            "schema": "deepseek.chrome.v1",
                            "action": "tab.new",
                            "port": port,
                            "ok": true,
                            "tab": tab,
                        })),
                        Err(err) => Ok(chrome_error_payload("tab.new", port, &debug_url, &err)),
                    }
                }
                "focus" | "activate" => {
                    let Some(target_id) = args.get(idx) else {
                        return Err(anyhow!("usage: /chrome tab focus <target_id>"));
                    };
                    match session.activate_tab(target_id) {
                        Ok(_) => Ok(json!({
                            "schema": "deepseek.chrome.v1",
                            "action": "tab.focus",
                            "port": port,
                            "ok": true,
                            "target_id": target_id,
                        })),
                        Err(err) => Ok(chrome_error_payload("tab.focus", port, &debug_url, &err)),
                    }
                }
                _ => Err(anyhow!("usage: /chrome tab [new <url>|focus <target_id>]")),
            }
        }
        "navigate" => {
            let Some(url) = args.get(idx) else {
                return Err(anyhow!("usage: /chrome navigate <url>"));
            };
            if let Err(err) = session.ensure_live_connection() {
                return Ok(chrome_error_payload("navigate", port, &debug_url, &err));
            }
            match session.navigate(url) {
                Ok(result) => Ok(json!({
                    "schema": "deepseek.chrome.v1",
                    "action": "navigate",
                    "port": port,
                    "url": url,
                    "ok": result.error.is_none(),
                    "error": result.error.map(|e| e.message),
                })),
                Err(err) => Ok(chrome_error_payload("navigate", port, &debug_url, &err)),
            }
        }
        "click" => {
            let Some(selector) = args.get(idx) else {
                return Err(anyhow!("usage: /chrome click <css-selector>"));
            };
            if let Err(err) = session.ensure_live_connection() {
                return Ok(chrome_error_payload("click", port, &debug_url, &err));
            }
            match session.click(selector) {
                Ok(result) => Ok(json!({
                    "schema": "deepseek.chrome.v1",
                    "action": "click",
                    "port": port,
                    "selector": selector,
                    "ok": result.error.is_none(),
                    "error": result.error.map(|e| e.message),
                })),
                Err(err) => Ok(chrome_error_payload("click", port, &debug_url, &err)),
            }
        }
        "type" => {
            let Some(selector) = args.get(idx) else {
                return Err(anyhow!("usage: /chrome type <css-selector> <text>"));
            };
            let Some(text) = args.get(idx + 1) else {
                return Err(anyhow!("usage: /chrome type <css-selector> <text>"));
            };
            if let Err(err) = session.ensure_live_connection() {
                return Ok(chrome_error_payload("type", port, &debug_url, &err));
            }
            match session.type_text(selector, text) {
                Ok(result) => Ok(json!({
                    "schema": "deepseek.chrome.v1",
                    "action": "type",
                    "port": port,
                    "selector": selector,
                    "text": text,
                    "ok": result.error.is_none(),
                    "error": result.error.map(|e| e.message),
                })),
                Err(err) => Ok(chrome_error_payload("type", port, &debug_url, &err)),
            }
        }
        "evaluate" => {
            let expression = args[idx..].join(" ");
            if expression.trim().is_empty() {
                return Err(anyhow!("usage: /chrome evaluate <javascript-expression>"));
            }
            if let Err(err) = session.ensure_live_connection() {
                return Ok(chrome_error_payload("evaluate", port, &debug_url, &err));
            }
            match session.evaluate(&expression) {
                Ok(value) => Ok(json!({
                    "schema": "deepseek.chrome.v1",
                    "action": "evaluate",
                    "port": port,
                    "expression": expression,
                    "ok": true,
                    "value": value,
                })),
                Err(err) => Ok(chrome_error_payload("evaluate", port, &debug_url, &err)),
            }
        }
        "record" => {
            let duration_seconds = args
                .get(idx)
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(3)
                .clamp(1, 60);
            let output_path = args
                .get(idx + 1)
                .cloned()
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    runtime_dir(cwd)
                        .join("chrome")
                        .join("recordings")
                        .join(format!("recording-{}.gif", Utc::now().timestamp()))
                });
            if let Err(err) = session.ensure_live_connection() {
                return Ok(chrome_error_payload("record", port, &debug_url, &err));
            }
            let frames_dir = runtime_dir(cwd)
                .join("chrome")
                .join("recordings")
                .join(format!("frames-{}", Utc::now().timestamp_millis()));
            std::fs::create_dir_all(&frames_dir)?;
            let start = std::time::Instant::now();
            let mut frame_count = 0usize;
            while start.elapsed().as_secs() < duration_seconds {
                let base64_png = session.screenshot(ScreenshotFormat::Png)?;
                let bytes = base64::engine::general_purpose::STANDARD.decode(base64_png)?;
                let frame_path = frames_dir.join(format!("frame-{:04}.png", frame_count));
                std::fs::write(&frame_path, bytes)?;
                frame_count += 1;
                std::thread::sleep(std::time::Duration::from_millis(250));
            }
            if frame_count == 0 {
                return Err(anyhow!("recording produced no frames"));
            }

            if let Some(parent) = output_path.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent)?;
            }

            let ffmpeg_result = std::process::Command::new("ffmpeg")
                .args([
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-framerate",
                    "4",
                    "-i",
                    "frame-%04d.png",
                    output_path.to_string_lossy().as_ref(),
                ])
                .current_dir(&frames_dir)
                .output();

            let mut export_mode = "frames_only".to_string();
            let mut export_error = None;
            if let Ok(output) = ffmpeg_result {
                if output.status.success() {
                    export_mode = "gif".to_string();
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                    if !stderr.is_empty() {
                        export_error = Some(stderr);
                    }
                }
            } else {
                export_error = Some("ffmpeg unavailable; kept PNG frame sequence".to_string());
            }

            Ok(json!({
                "schema": "deepseek.chrome.v1",
                "action": "record",
                "port": port,
                "ok": true,
                "duration_seconds": duration_seconds,
                "frame_count": frame_count,
                "frames_dir": frames_dir.to_string_lossy().to_string(),
                "output_path": if export_mode == "gif" {
                    output_path.to_string_lossy().to_string()
                } else {
                    String::new()
                },
                "export_mode": export_mode,
                "export_error": export_error,
            }))
        }
        "screenshot" => {
            let format = match args.get(idx).map(|v| v.to_ascii_lowercase()) {
                Some(value) if value == "jpeg" || value == "jpg" => ScreenshotFormat::Jpeg,
                Some(value) if value == "webp" => ScreenshotFormat::Webp,
                _ => ScreenshotFormat::Png,
            };
            let output_path = args
                .get(idx + 1)
                .cloned()
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    let ext = match format {
                        ScreenshotFormat::Png => "png",
                        ScreenshotFormat::Jpeg => "jpg",
                        ScreenshotFormat::Webp => "webp",
                    };
                    runtime_dir(cwd).join("chrome").join(format!(
                        "screenshot-{}.{}",
                        Utc::now().timestamp(),
                        ext
                    ))
                });
            if let Some(parent) = output_path.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent)?;
            }
            if let Err(err) = session.ensure_live_connection() {
                return Ok(chrome_error_payload("screenshot", port, &debug_url, &err));
            }
            match session.screenshot(format) {
                Ok(data) => {
                    let bytes = base64::engine::general_purpose::STANDARD.decode(data)?;
                    std::fs::write(&output_path, bytes)?;
                    Ok(json!({
                        "schema": "deepseek.chrome.v1",
                        "action": "screenshot",
                        "port": port,
                        "path": output_path.to_string_lossy().to_string(),
                        "ok": true,
                    }))
                }
                Err(err) => Ok(chrome_error_payload("screenshot", port, &debug_url, &err)),
            }
        }
        "console" => match session
            .ensure_live_connection()
            .and_then(|_| session.read_console())
        {
            Ok(entries) => Ok(json!({
                "schema": "deepseek.chrome.v1",
                "action": "console",
                "port": port,
                "count": entries.len(),
                "entries": entries,
            })),
            Err(err) => Ok(chrome_error_payload("console", port, &debug_url, &err)),
        },
        _ => Err(anyhow!(
            "usage: /chrome [status|reconnect|tabs|tab new <url>|tab focus <target_id>|navigate <url>|click <selector>|type <selector> <text>|evaluate <js>|record [seconds] [output.gif]|screenshot [png|jpeg|webp] [output]|console]"
        )),
    }
}

pub(super) fn chrome_error_payload(
    action: &str,
    port: u16,
    debug_url: &str,
    err: &anyhow::Error,
) -> serde_json::Value {
    let message = err.to_string();
    let lower = message.to_ascii_lowercase();
    let (kind, hints): (&str, Vec<&str>) =
        if lower.contains("endpoint_unreachable") || lower.contains("connection refused") {
            (
                "endpoint_unreachable",
                vec![
                    "Start Chrome with --remote-debugging-port=9222",
                    "Verify CODINGBUDDY_CHROME_PORT points to the active debugging port",
                ],
            )
        } else if lower.contains("no_page_targets") || lower.contains("no debuggable page target") {
            (
                "no_page_targets",
                vec![
                    "Run /chrome reconnect to create a recovery tab",
                    "Open at least one browser tab in the target Chrome profile",
                ],
            )
        } else if lower.contains("timed out") {
            (
                "endpoint_timeout",
                vec![
                    "Confirm local firewall/proxy rules are not blocking localhost debug traffic",
                    "Retry /chrome reconnect once the browser is responsive",
                ],
            )
        } else if lower.contains("cdp error") {
            (
                "cdp_command_failed",
                vec![
                    "Retry the command after /chrome reconnect",
                    "Use /chrome tabs to confirm the active page target",
                ],
            )
        } else {
            (
                "chrome_error",
                vec![
                    "Run /chrome status for live connection diagnostics",
                    "Run /chrome reconnect to recover stale sessions",
                ],
            )
        };

    json!({
        "schema": "deepseek.chrome.v1",
        "action": action,
        "port": port,
        "ok": false,
        "debug_url": debug_url,
        "error": {
            "kind": kind,
            "message": message,
            "hints": hints,
        }
    })
}

pub(super) fn parse_debug_analysis_args(_args: &[String]) -> Result<Option<DoctorArgs>> {
    Ok(None)
}
