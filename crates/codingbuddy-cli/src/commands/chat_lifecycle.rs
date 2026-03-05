use super::*;

#[derive(Debug, Clone)]
pub(super) struct SessionLifecycleNotice {
    pub(super) line: String,
    pub(super) is_error: bool,
}

pub(super) fn active_session_id_for_notices(
    cwd: &Path,
    session_override: Option<Uuid>,
) -> Result<Option<Uuid>> {
    if let Some(session_id) = session_override {
        return Ok(Some(session_id));
    }
    Ok(Store::new(cwd)?
        .load_latest_session()?
        .map(|session| session.session_id))
}

pub(super) fn latest_session_event_seq(cwd: &Path, session_id: Uuid) -> Result<u64> {
    Ok(read_session_events(cwd, session_id)?
        .last()
        .map(|event| event.seq_no)
        .unwrap_or(0))
}

pub(super) fn background_task_notice(
    store: &Store,
    task_id: &str,
    status: &str,
) -> Result<Option<SessionLifecycleNotice>> {
    let Ok(task_id) = Uuid::parse_str(task_id) else {
        return Ok(None);
    };
    let Some(run) = store.load_subagent_run_for_task(task_id)? else {
        return Ok(None);
    };
    if run.background_job_id.is_none() {
        return Ok(None);
    }
    let Some(task) = store.load_task(task_id)? else {
        return Ok(None);
    };
    let subject = if task.title.trim().is_empty() {
        run.name.clone()
    } else {
        task.title.clone()
    };
    let detail = task
        .outcome
        .as_deref()
        .or(run.output.as_deref())
        .or(run.error.as_deref())
        .unwrap_or_default();
    let detail = truncate_inline(&detail.replace('\n', " "), 120);
    let is_error = matches!(status, "failed" | "cancelled");
    let line = match status {
        "completed" if detail.is_empty() => format!("[background] task completed: {subject}"),
        "completed" => format!("[background] task completed: {subject} — {detail}"),
        "failed" if detail.is_empty() => format!("[background] task failed: {subject}"),
        "failed" => format!("[background] task failed: {subject} — {detail}"),
        "cancelled" if detail.is_empty() => format!("[background] task cancelled: {subject}"),
        "cancelled" => format!("[background] task cancelled: {subject} — {detail}"),
        _ => return Ok(None),
    };
    Ok(Some(SessionLifecycleNotice { line, is_error }))
}

pub(super) fn background_job_notice(
    store: &Store,
    job_id: Uuid,
    reason: &str,
) -> Result<Option<SessionLifecycleNotice>> {
    let Some(job) = store.load_background_job(job_id)? else {
        return Ok(None);
    };
    if job.kind == "subagent" {
        return Ok(None);
    }
    let reason_lower = reason.to_ascii_lowercase();
    let subject = if job.reference.trim().is_empty() {
        format!("{} {}", job.kind, job.job_id)
    } else {
        format!("{} {}", job.kind, job.reference)
    };
    let is_error = reason_lower.contains("fail");
    let line = match reason_lower.as_str() {
        "manual_stop" => format!("[background] stopped: {subject}"),
        "completed" => format!("[background] completed: {subject}"),
        _ if is_error => format!("[background] failed: {subject} — {reason}"),
        _ => format!("[background] update: {subject} — {reason}"),
    };
    Ok(Some(SessionLifecycleNotice { line, is_error }))
}

pub(super) fn plan_notices(store: &Store, session_id: Uuid) -> Result<Vec<SessionLifecycleNotice>> {
    let Some(session) = store.load_session(session_id)? else {
        return Ok(Vec::new());
    };
    let Some(payload) = current_plan_payload(store, Some(&session))? else {
        return Ok(Vec::new());
    };
    Ok(render_plan_notice_lines(&payload)
        .into_iter()
        .map(|line| SessionLifecycleNotice {
            line,
            is_error: false,
        })
        .collect())
}

pub(super) fn plan_state_transition_notice(
    from: &codingbuddy_core::SessionState,
    to: &codingbuddy_core::SessionState,
) -> Option<SessionLifecycleNotice> {
    let line = match (from, to) {
        (
            codingbuddy_core::SessionState::AwaitingApproval,
            codingbuddy_core::SessionState::ExecutingStep,
        ) => "[plan] approved; workflow moved to execute".to_string(),
        (
            codingbuddy_core::SessionState::AwaitingApproval,
            codingbuddy_core::SessionState::Planning,
        ) => "[plan] returned to drafting for revision".to_string(),
        _ => return None,
    };
    Some(SessionLifecycleNotice {
        line,
        is_error: false,
    })
}

pub(super) fn poll_session_lifecycle_notices(
    cwd: &Path,
    session_override: Option<Uuid>,
    watermarks: &mut HashMap<Uuid, u64>,
) -> Result<Vec<SessionLifecycleNotice>> {
    let Some(session_id) = active_session_id_for_notices(cwd, session_override)? else {
        return Ok(Vec::new());
    };
    let store = Store::new(cwd)?;
    let baseline = if let Some(seq) = watermarks.get(&session_id).copied() {
        seq
    } else {
        let latest = latest_session_event_seq(cwd, session_id)?;
        watermarks.insert(session_id, latest);
        if let Some(session) = store.load_session(session_id)?
            && matches!(
                session.status,
                codingbuddy_core::SessionState::AwaitingApproval
            )
            && session.active_plan_id.is_some()
        {
            return plan_notices(&store, session_id);
        }
        return Ok(Vec::new());
    };
    let mut next_seq = baseline;
    let mut notices = Vec::new();
    for event in read_session_events(cwd, session_id)? {
        if event.seq_no <= baseline {
            continue;
        }
        next_seq = next_seq.max(event.seq_no);
        let event_notices = match event.kind {
            EventKind::TaskUpdated { task_id, status } => {
                background_task_notice(&store, &task_id, &status)?
                    .into_iter()
                    .collect::<Vec<_>>()
            }
            EventKind::BackgroundJobStopped { job_id, reason } => {
                background_job_notice(&store, job_id, &reason)?
                    .into_iter()
                    .collect::<Vec<_>>()
            }
            EventKind::PlanCreated { .. } | EventKind::PlanRevised { .. } => {
                plan_notices(&store, session_id)?
            }
            EventKind::SessionStateChanged { from, to } => plan_state_transition_notice(&from, &to)
                .into_iter()
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        for notice in event_notices {
            notices.push(notice);
        }
    }
    watermarks.insert(session_id, next_seq);
    Ok(notices)
}
