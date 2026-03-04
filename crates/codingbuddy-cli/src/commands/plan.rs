use anyhow::{Result, anyhow};
use codingbuddy_core::{EventKind, Plan, Session, SessionState, is_valid_session_state_transition};
use codingbuddy_store::Store;
use serde_json::{Value, json};
use std::path::Path;
use uuid::Uuid;

use crate::context::append_control_event_for_session;

pub(crate) struct PlanSlashResponse {
    pub payload: Value,
    pub text: String,
    pub session_switch: Option<Uuid>,
}

pub(crate) fn workflow_phase_label(state: &SessionState) -> &'static str {
    match state {
        SessionState::Idle => "idle",
        SessionState::Planning => "plan",
        SessionState::ExecutingStep => "execute",
        SessionState::AwaitingApproval => "approval",
        SessionState::Verifying => "verify",
        SessionState::Completed => "completed",
        SessionState::Paused => "paused",
        SessionState::Failed => "failed",
    }
}

pub(crate) fn plan_state_label(session: Option<&Session>) -> &'static str {
    let Some(session) = session else {
        return "none";
    };
    if session.active_plan_id.is_none() {
        return "none";
    }
    match session.status {
        SessionState::Planning => "draft",
        SessionState::AwaitingApproval => "awaiting_approval",
        SessionState::ExecutingStep | SessionState::Verifying | SessionState::Completed => {
            "approved"
        }
        SessionState::Paused => "paused",
        SessionState::Failed => "failed",
        SessionState::Idle => "available",
    }
}

fn truncate_inline(text: &str, max_chars: usize) -> String {
    let trimmed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if trimmed.len() <= max_chars {
        trimmed
    } else {
        format!("{}...", &trimmed[..trimmed.floor_char_boundary(max_chars)])
    }
}

fn scoped_session(store: &Store, session_override: Option<Uuid>) -> Result<Option<Session>> {
    if let Some(session_id) = session_override {
        return store.load_session(session_id);
    }
    store.load_latest_session()
}

pub(crate) fn load_active_plan(store: &Store, session: Option<&Session>) -> Result<Option<Plan>> {
    let Some(session) = session else {
        return Ok(None);
    };
    let Some(plan_id) = session.active_plan_id else {
        return Ok(None);
    };
    store.load_plan(plan_id)
}

pub(crate) fn current_plan_payload(
    store: &Store,
    session: Option<&Session>,
) -> Result<Option<Value>> {
    let Some(session) = session else {
        return Ok(None);
    };
    let Some(plan) = load_active_plan(store, Some(session))? else {
        return Ok(None);
    };
    Ok(Some(plan_payload(session, &plan)))
}

pub(crate) fn plan_payload(session: &Session, plan: &Plan) -> Value {
    json!({
        "session_id": session.session_id.to_string(),
        "state": session.status,
        "workflow_phase": workflow_phase_label(&session.status),
        "plan_state": plan_state_label(Some(session)),
        "plan_id": plan.plan_id.to_string(),
        "version": plan.version,
        "goal": plan.goal,
        "goal_preview": truncate_inline(&plan.goal, 140),
        "steps_count": plan.steps.len(),
        "assumptions_count": plan.assumptions.len(),
        "verification_count": plan.verification.len(),
        "risk_count": plan.risk_notes.len(),
        "steps": plan.steps.iter().map(|step| {
            json!({
                "step_id": step.step_id.to_string(),
                "title": step.title,
                "intent": step.intent,
                "done": step.done,
                "files": step.files,
                "tools": step.tools,
            })
        }).collect::<Vec<_>>(),
        "assumptions": plan.assumptions,
        "verification": plan.verification,
        "risk_notes": plan.risk_notes,
        "next_actions": [
            "/plan approve",
            "/plan reject <feedback>"
        ],
    })
}

pub(crate) fn render_plan_payload(payload: &Value) -> String {
    let session_id = payload
        .get("session_id")
        .and_then(Value::as_str)
        .unwrap_or("pending");
    let workflow_phase = payload
        .get("workflow_phase")
        .and_then(Value::as_str)
        .unwrap_or("idle");
    let plan_state = payload
        .get("plan_state")
        .and_then(Value::as_str)
        .unwrap_or("none");
    let version = payload.get("version").and_then(Value::as_u64).unwrap_or(0);
    let steps = payload
        .get("steps")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let assumptions = payload
        .get("assumptions")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let verification = payload
        .get("verification")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let risks = payload
        .get("risk_notes")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut lines = vec![format!(
        "Plan review: session={} phase={} state={} version={} steps={}",
        session_id,
        workflow_phase,
        plan_state,
        version,
        steps.len()
    )];
    if let Some(goal) = payload.get("goal").and_then(Value::as_str)
        && !goal.trim().is_empty()
    {
        lines.push(format!("- Goal: {}", truncate_inline(goal, 180)));
    }
    if steps.is_empty() {
        lines.push("- Steps: none".to_string());
    } else {
        lines.push("- Steps:".to_string());
        for (idx, step) in steps.iter().enumerate() {
            let title = step
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("untitled");
            lines.push(format!("  {}. {}", idx + 1, title));
        }
    }
    if !assumptions.is_empty() {
        lines.push("- Assumptions:".to_string());
        for row in assumptions.iter().take(5) {
            if let Some(value) = row.as_str() {
                lines.push(format!("  - {}", truncate_inline(value, 160)));
            }
        }
    }
    if !verification.is_empty() {
        lines.push("- Verification:".to_string());
        for row in verification.iter().take(5) {
            if let Some(value) = row.as_str() {
                lines.push(format!("  - {}", truncate_inline(value, 160)));
            }
        }
    }
    if !risks.is_empty() {
        lines.push("- Risks:".to_string());
        for row in risks.iter().take(5) {
            if let Some(value) = row.as_str() {
                lines.push(format!("  - {}", truncate_inline(value, 160)));
            }
        }
    }
    if plan_state == "awaiting_approval" {
        lines.push("- Next: /plan approve | /plan reject <feedback>".to_string());
    }
    lines.join("\n")
}

pub(crate) fn render_plan_notice_lines(payload: &Value) -> Vec<String> {
    let plan_state = payload
        .get("plan_state")
        .and_then(Value::as_str)
        .unwrap_or("none");
    let version = payload.get("version").and_then(Value::as_u64).unwrap_or(0);
    let goal = payload
        .get("goal_preview")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let steps = payload
        .get("steps")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let verification = payload
        .get("verification")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let label = match plan_state {
        "draft" => "draft ready",
        "awaiting_approval" => "awaiting approval",
        "approved" => "approved",
        "paused" => "paused",
        "failed" => "failed",
        other => other,
    };

    let mut lines = vec![format!("[plan] {label} v{version}: {goal}")];
    if !steps.is_empty() {
        let preview = steps
            .iter()
            .take(3)
            .enumerate()
            .map(|(idx, step)| {
                let title = step
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or("untitled");
                format!("{}. {}", idx + 1, truncate_inline(title, 48))
            })
            .collect::<Vec<_>>()
            .join(" | ");
        let extra = steps.len().saturating_sub(3);
        let suffix = if extra > 0 {
            format!(" | +{} more", extra)
        } else {
            String::new()
        };
        lines.push(format!("[plan] steps: {preview}{suffix}"));
    }
    if let Some(first_verify) = verification.first().and_then(Value::as_str) {
        lines.push(format!(
            "[plan] verify: {}",
            truncate_inline(first_verify, 100)
        ));
    }
    if plan_state == "awaiting_approval" {
        lines.push("[plan] next: /plan show | /plan approve | /plan reject <feedback>".to_string());
    }
    lines
}

fn transition_session_state(cwd: &Path, session: &Session, to: SessionState) -> Result<()> {
    if !is_valid_session_state_transition(&session.status, &to) {
        return Err(anyhow!(
            "invalid plan state transition: {:?} -> {:?}",
            session.status,
            to
        ));
    }
    append_control_event_for_session(
        cwd,
        session.session_id,
        EventKind::SessionStateChanged {
            from: session.status.clone(),
            to,
        },
    )
}

fn require_active_plan(store: &Store, session: &Session) -> Result<Plan> {
    load_active_plan(store, Some(session))?.ok_or_else(|| {
        anyhow!(
            "no active plan for session {}. Ask the agent to enter plan mode first.",
            session.session_id
        )
    })
}

pub(crate) fn handle_plan_slash(
    cwd: &Path,
    args: &[String],
    session_override: Option<Uuid>,
) -> Result<PlanSlashResponse> {
    let store = Store::new(cwd)?;
    let session =
        scoped_session(&store, session_override)?.ok_or_else(|| anyhow!("no active session"))?;
    let subcommand = args
        .first()
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_else(|| "show".to_string());

    match subcommand.as_str() {
        "show" | "status" | "review" => {
            let payload = current_plan_payload(&store, Some(&session))?.ok_or_else(|| {
                anyhow!(
                    "no active plan for session {}. Ask the agent to create a plan first.",
                    session.session_id
                )
            })?;
            let text = render_plan_payload(&payload);
            Ok(PlanSlashResponse {
                payload,
                text,
                session_switch: Some(session.session_id),
            })
        }
        "approve" | "accept" | "apply" => {
            let plan = require_active_plan(&store, &session)?;
            match session.status {
                SessionState::AwaitingApproval => {
                    transition_session_state(cwd, &session, SessionState::ExecutingStep)?;
                }
                SessionState::ExecutingStep | SessionState::Verifying | SessionState::Completed => {
                    return Ok(PlanSlashResponse {
                        payload: json!({
                            "approved": true,
                            "session_id": session.session_id.to_string(),
                            "plan": plan_payload(&session, &plan),
                            "message": "plan already approved",
                        }),
                        text: "Plan is already approved. Continue with your next implementation instruction.".to_string(),
                        session_switch: Some(session.session_id),
                    });
                }
                SessionState::Planning => {
                    return Err(anyhow!(
                        "plan is still being drafted. Finish the draft and save it for approval first."
                    ));
                }
                _ => {
                    return Err(anyhow!(
                        "plan approval is only available when the session is awaiting approval"
                    ));
                }
            }
            let refreshed = store.load_session(session.session_id)?.unwrap_or(Session {
                status: SessionState::ExecutingStep,
                ..session.clone()
            });
            let payload = json!({
                "approved": true,
                "session_id": session.session_id.to_string(),
                "plan": plan_payload(&refreshed, &plan),
                "message": "plan approved; session moved to execute",
            });
            Ok(PlanSlashResponse {
                payload,
                text: format!(
                    "Plan approved for session {}. The workflow is now in execute. Send your next instruction to continue implementation.",
                    session.session_id
                ),
                session_switch: Some(session.session_id),
            })
        }
        "reject" | "revise" => {
            let plan = require_active_plan(&store, &session)?;
            match session.status {
                SessionState::AwaitingApproval => {
                    transition_session_state(cwd, &session, SessionState::Planning)?;
                }
                SessionState::Planning => {}
                _ => {
                    return Err(anyhow!(
                        "plan rejection is only available while drafting or awaiting approval"
                    ));
                }
            }
            let feedback = args.iter().skip(1).cloned().collect::<Vec<_>>().join(" ");
            let refreshed = store
                .load_session(session.session_id)?
                .unwrap_or(session.clone());
            let payload = json!({
                "approved": false,
                "feedback": if feedback.trim().is_empty() { Value::Null } else { Value::String(feedback.clone()) },
                "session_id": session.session_id.to_string(),
                "plan": plan_payload(&refreshed, &plan),
                "message": "plan returned to drafting",
            });
            let text = if feedback.trim().is_empty() {
                format!(
                    "Plan returned to drafting for session {}. Send feedback in your next message so the agent can revise it.",
                    session.session_id
                )
            } else {
                format!(
                    "Plan returned to drafting for session {}.\nInclude this feedback in your next message: {}",
                    session.session_id,
                    truncate_inline(&feedback, 180)
                )
            };
            Ok(PlanSlashResponse {
                payload,
                text,
                session_switch: Some(session.session_id),
            })
        }
        _ => Err(anyhow!(
            "unknown /plan subcommand: {subcommand}. Use /plan show|approve|reject <feedback>"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codingbuddy_core::{SessionBudgets, SessionState};
    use tempfile::tempdir;

    fn sample_plan(goal: &str) -> Plan {
        Plan {
            plan_id: Uuid::now_v7(),
            version: 2,
            goal: goal.to_string(),
            assumptions: vec!["Assume tests are already green.".to_string()],
            steps: vec![
                codingbuddy_core::PlanStep {
                    step_id: Uuid::now_v7(),
                    title: "Inspect auth flow".to_string(),
                    intent: "Read current implementation".to_string(),
                    tools: Vec::new(),
                    files: vec!["src/auth.rs".to_string()],
                    done: false,
                },
                codingbuddy_core::PlanStep {
                    step_id: Uuid::now_v7(),
                    title: "Patch login handler".to_string(),
                    intent: "Fix the regression".to_string(),
                    tools: Vec::new(),
                    files: vec!["src/login.rs".to_string()],
                    done: false,
                },
            ],
            verification: vec!["Run cargo test -p codingbuddy-cli".to_string()],
            risk_notes: vec!["Be careful with session persistence.".to_string()],
        }
    }

    fn sample_session(root: &Path, status: SessionState, plan_id: Option<Uuid>) -> Session {
        Session {
            session_id: Uuid::now_v7(),
            workspace_root: root.display().to_string(),
            baseline_commit: None,
            status,
            budgets: SessionBudgets {
                per_turn_seconds: 30,
                max_think_tokens: 4096,
            },
            active_plan_id: plan_id,
        }
    }

    #[test]
    fn plan_state_label_tracks_review_state() {
        let root = Path::new("/tmp");
        let plan_id = Uuid::now_v7();
        let draft = sample_session(root, SessionState::Planning, Some(plan_id));
        let review = sample_session(root, SessionState::AwaitingApproval, Some(plan_id));
        let execute = sample_session(root, SessionState::ExecutingStep, Some(plan_id));
        assert_eq!(plan_state_label(Some(&draft)), "draft");
        assert_eq!(plan_state_label(Some(&review)), "awaiting_approval");
        assert_eq!(plan_state_label(Some(&execute)), "approved");
        assert_eq!(plan_state_label(None), "none");
    }

    #[test]
    fn handle_plan_slash_approve_transitions_to_execute() -> Result<()> {
        let temp = tempdir()?;
        let store = Store::new(temp.path())?;
        let plan = sample_plan("Fix login");
        let session = sample_session(
            temp.path(),
            SessionState::AwaitingApproval,
            Some(plan.plan_id),
        );
        store.save_session(&session)?;
        store.save_plan(session.session_id, &plan)?;

        let response = handle_plan_slash(
            temp.path(),
            &["approve".to_string()],
            Some(session.session_id),
        )?;
        let updated = store
            .load_session(session.session_id)?
            .ok_or_else(|| anyhow!("missing session"))?;

        assert_eq!(updated.status, SessionState::ExecutingStep);
        assert!(response.text.contains("execute"));
        assert_eq!(response.session_switch, Some(session.session_id));
        Ok(())
    }

    #[test]
    fn handle_plan_slash_reject_transitions_back_to_planning() -> Result<()> {
        let temp = tempdir()?;
        let store = Store::new(temp.path())?;
        let plan = sample_plan("Fix login");
        let session = sample_session(
            temp.path(),
            SessionState::AwaitingApproval,
            Some(plan.plan_id),
        );
        store.save_session(&session)?;
        store.save_plan(session.session_id, &plan)?;

        let response = handle_plan_slash(
            temp.path(),
            &[
                "reject".to_string(),
                "Need".to_string(),
                "less".to_string(),
                "risk".to_string(),
            ],
            Some(session.session_id),
        )?;
        let updated = store
            .load_session(session.session_id)?
            .ok_or_else(|| anyhow!("missing session"))?;

        assert_eq!(updated.status, SessionState::Planning);
        assert!(response.text.contains("drafting"));
        Ok(())
    }

    #[test]
    fn render_plan_notice_lines_summarizes_review_state() {
        let root = Path::new("/tmp");
        let plan = sample_plan("Fix login");
        let session = sample_session(root, SessionState::AwaitingApproval, Some(plan.plan_id));
        let payload = plan_payload(&session, &plan);
        let lines = render_plan_notice_lines(&payload);
        assert!(!lines.is_empty());
        assert!(lines[0].contains("awaiting approval"));
        assert!(lines.iter().any(|line| line.contains("steps:")));
        assert!(
            lines
                .iter()
                .any(|line| line.contains("/plan approve") && line.contains("/plan reject"))
        );
    }
}
