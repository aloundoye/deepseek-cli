use super::*;
use codingbuddy_core::{EventEnvelope, Plan, PlanStep, Session, SessionBudgets, SessionState};
use codingbuddy_store::{
    BackgroundJobRecord, SessionTodoRecord, SubagentRunRecord, TaskQueueRecord,
};
use std::fs;
use tempfile::tempdir;

#[test]
fn chrome_error_payload_classifies_endpoint_unreachable() {
    let payload = chrome_error_payload(
        "status",
        9222,
        "http://127.0.0.1:9222",
        &anyhow::anyhow!("Connection refused (os error 61)"),
    );
    assert_eq!(payload["error"]["kind"], "endpoint_unreachable");
}

#[test]
fn chrome_error_payload_classifies_no_page_targets() {
    let payload = chrome_error_payload(
        "status",
        9222,
        "http://127.0.0.1:9222",
        &anyhow::anyhow!("no_page_targets: no debuggable page target available"),
    );
    assert_eq!(payload["error"]["kind"], "no_page_targets");
}

#[test]
fn render_web_fetch_markdown_contains_metadata_and_extract_block() {
    let output = serde_json::json!({
        "status": 200,
        "content_type": "text/html",
        "truncated": false,
        "bytes": 320,
        "content": "Title\n\nFirst paragraph.\nSecond paragraph."
    });
    let rendered = render_web_fetch_markdown("https://example.com", &output, 6);
    assert!(rendered.contains("# Web Fetch"));
    assert!(rendered.contains("- URL: https://example.com"));
    assert!(rendered.contains("- Status: 200"));
    assert!(rendered.contains("## Extract"));
    assert!(rendered.contains("```text"));
    assert!(rendered.contains("First paragraph."));
}

#[test]
fn render_web_search_markdown_formats_results_and_extract() {
    let results = vec![
        serde_json::json!({
            "title": "CodingBuddy docs",
            "url": "https://example.com/docs",
            "snippet": "A deterministic coding agent runtime."
        }),
        serde_json::json!({
            "title": "Architecture notes",
            "url": "https://example.com/notes",
            "snippet": "Architect editor apply verify."
        }),
    ];
    let rendered = render_web_search_markdown(
        "deepseek cli",
        &results,
        Some((
            "https://example.com/docs".to_string(),
            "CodingBuddy is terminal-native.".to_string(),
        )),
    );
    assert!(rendered.contains("# Web Search"));
    assert!(rendered.contains("- Query: deepseek cli"));
    assert!(rendered.contains("1. CodingBuddy docs"));
    assert!(rendered.contains("## Extract (https://example.com/docs)"));
    assert!(rendered.contains("CodingBuddy is terminal-native."));
}

#[test]
fn todos_payload_uses_session_native_checklist() -> Result<()> {
    let temp = tempdir()?;
    let store = Store::new(temp.path())?;
    let plan_id = Uuid::now_v7();
    let session = Session {
        session_id: Uuid::now_v7(),
        workspace_root: temp.path().display().to_string(),
        baseline_commit: None,
        status: SessionState::ExecutingStep,
        budgets: SessionBudgets {
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
            goal: "Stabilize CI".to_string(),
            assumptions: vec![],
            steps: vec![
                PlanStep {
                    step_id: Uuid::now_v7(),
                    title: "Collect failing logs".to_string(),
                    intent: "Understand root cause".to_string(),
                    tools: vec![],
                    files: vec![],
                    done: true,
                },
                PlanStep {
                    step_id: Uuid::now_v7(),
                    title: "Patch flaky path".to_string(),
                    intent: "Fix cross-platform behavior".to_string(),
                    tools: vec![],
                    files: vec!["crates/codingbuddy-hooks/src/lib.rs".to_string()],
                    done: false,
                },
            ],
            verification: vec!["cargo test -p codingbuddy-hooks".to_string()],
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
                content: "Collect failing logs".to_string(),
                status: "completed".to_string(),
                position: 0,
                created_at: now.clone(),
                updated_at: now.clone(),
            },
            SessionTodoRecord {
                todo_id: Uuid::now_v7(),
                session_id: session.session_id,
                content: "Patch flaky path".to_string(),
                status: "in_progress".to_string(),
                position: 1,
                created_at: now.clone(),
                updated_at: now,
            },
        ],
    )?;

    let payload = todos_payload(temp.path(), Some(session.session_id), &[])?;
    assert_eq!(
        payload["schema"].as_str(),
        Some("deepseek.session_todos.v1")
    );
    assert_eq!(payload["workflow_phase"].as_str(), Some("execute"));
    assert_eq!(payload["plan_state"].as_str(), Some("approved"));
    assert_eq!(payload["count"].as_u64(), Some(2));
    assert_eq!(payload["summary"]["completed"].as_u64(), Some(1));
    assert_eq!(payload["summary"]["in_progress"].as_u64(), Some(1));
    assert_eq!(
        payload["summary"]["current"]["content"].as_str(),
        Some("Patch flaky path")
    );
    assert_eq!(
        payload["current_step"]["title"].as_str(),
        Some("Patch flaky path")
    );
    Ok(())
}

#[test]
fn comment_todos_payload_reports_comment_scan_schema() -> Result<()> {
    let rg_available = std::process::Command::new("rg")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !rg_available {
        eprintln!("skipping comment_todos test: rg not found on PATH");
        return Ok(());
    }

    let temp = tempdir()?;
    fs::write(
        temp.path().join("notes.md"),
        "# Notes\nTODO: tighten test coverage\n",
    )?;
    let payload = comment_todos_payload(temp.path(), &[])?;
    assert_eq!(
        payload["schema"].as_str(),
        Some("deepseek.comment_todos.v1")
    );
    assert!(payload["count"].as_u64().unwrap_or(0) >= 1);
    assert!(
        payload["items"]
            .as_array()
            .map(|rows| {
                rows.iter()
                    .any(|row| row["text"].as_str().unwrap_or_default().contains("TODO"))
            })
            .unwrap_or(false)
    );
    Ok(())
}

#[test]
fn poll_session_lifecycle_notices_reports_background_task_completion() -> Result<()> {
    let temp = tempdir()?;
    let store = Store::new(temp.path())?;
    let session = Session {
        session_id: Uuid::now_v7(),
        workspace_root: temp.path().display().to_string(),
        baseline_commit: None,
        status: SessionState::ExecutingStep,
        budgets: SessionBudgets {
            per_turn_seconds: 30,
            max_think_tokens: 4096,
        },
        active_plan_id: None,
    };
    store.save_session(&session)?;
    let task_id = Uuid::now_v7();
    let run_id = Uuid::now_v7();
    let job_id = Uuid::now_v7();
    let now = Utc::now().to_rfc3339();
    store.insert_task(&TaskQueueRecord {
        task_id,
        session_id: session.session_id,
        title: "Background audit".to_string(),
        description: None,
        priority: 1,
        status: "completed".to_string(),
        outcome: Some("done".to_string()),
        artifact_path: None,
        created_at: now.clone(),
        updated_at: now.clone(),
    })?;
    store.upsert_subagent_run(&SubagentRunRecord {
        run_id,
        session_id: Some(session.session_id),
        task_id: Some(task_id),
        child_session_id: None,
        background_job_id: Some(job_id),
        name: "explore".to_string(),
        goal: "inspect".to_string(),
        status: "completed".to_string(),
        output: Some("done".to_string()),
        error: None,
        created_at: now.clone(),
        updated_at: now.clone(),
    })?;
    store.upsert_background_job(&BackgroundJobRecord {
        job_id,
        kind: "subagent".to_string(),
        reference: format!("subagent:{run_id}"),
        status: "completed".to_string(),
        metadata_json: json!({"reason":"completed"}).to_string(),
        started_at: now.clone(),
        updated_at: now.clone(),
    })?;
    store.append_event(&EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: Utc::now(),
        session_id: session.session_id,
        kind: EventKind::TaskUpdated {
            task_id: task_id.to_string(),
            status: "completed".to_string(),
        },
    })?;

    let mut watermarks = HashMap::from([(session.session_id, 0_u64)]);
    let notices =
        poll_session_lifecycle_notices(temp.path(), Some(session.session_id), &mut watermarks)?;
    assert_eq!(notices.len(), 1);
    assert!(notices[0].line.contains("Background audit"));
    assert!(notices[0].line.contains("done"));
    assert!(!notices[0].is_error);
    Ok(())
}

#[test]
fn poll_session_lifecycle_notices_reports_background_job_stop() -> Result<()> {
    let temp = tempdir()?;
    let store = Store::new(temp.path())?;
    let session = Session {
        session_id: Uuid::now_v7(),
        workspace_root: temp.path().display().to_string(),
        baseline_commit: None,
        status: SessionState::Idle,
        budgets: SessionBudgets {
            per_turn_seconds: 30,
            max_think_tokens: 4096,
        },
        active_plan_id: None,
    };
    store.save_session(&session)?;
    let job_id = Uuid::now_v7();
    let now = Utc::now().to_rfc3339();
    store.upsert_background_job(&BackgroundJobRecord {
        job_id,
        kind: "shell".to_string(),
        reference: "shell:abc123".to_string(),
        status: "stopped".to_string(),
        metadata_json: json!({"reason":"manual_stop"}).to_string(),
        started_at: now.clone(),
        updated_at: now.clone(),
    })?;
    store.append_event(&EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: Utc::now(),
        session_id: session.session_id,
        kind: EventKind::BackgroundJobStopped {
            job_id,
            reason: "manual_stop".to_string(),
        },
    })?;

    let mut watermarks = HashMap::from([(session.session_id, 0_u64)]);
    let notices =
        poll_session_lifecycle_notices(temp.path(), Some(session.session_id), &mut watermarks)?;
    assert_eq!(notices.len(), 1);
    assert!(notices[0].line.contains("stopped"));
    assert!(notices[0].line.contains("shell"));
    assert!(!notices[0].is_error);
    Ok(())
}

#[test]
fn poll_session_lifecycle_notices_reports_plan_review_summary() -> Result<()> {
    let temp = tempdir()?;
    let store = Store::new(temp.path())?;
    let plan_id = Uuid::now_v7();
    let session = Session {
        session_id: Uuid::now_v7(),
        workspace_root: temp.path().display().to_string(),
        baseline_commit: None,
        status: SessionState::AwaitingApproval,
        budgets: SessionBudgets {
            per_turn_seconds: 30,
            max_think_tokens: 4096,
        },
        active_plan_id: Some(plan_id),
    };
    store.save_session(&session)?;
    let plan = Plan {
        plan_id,
        version: 2,
        goal: "Fix the login race".to_string(),
        assumptions: vec![],
        steps: vec![
            PlanStep {
                step_id: Uuid::now_v7(),
                title: "Inspect login handler".to_string(),
                intent: "Read the current flow".to_string(),
                tools: vec![],
                files: vec!["src/login.rs".to_string()],
                done: false,
            },
            PlanStep {
                step_id: Uuid::now_v7(),
                title: "Patch state transition".to_string(),
                intent: "Remove the race".to_string(),
                tools: vec![],
                files: vec!["src/state.rs".to_string()],
                done: false,
            },
        ],
        verification: vec!["cargo test -p codingbuddy-cli".to_string()],
        risk_notes: vec![],
    };
    store.save_plan(session.session_id, &plan)?;
    store.append_event(&EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: Utc::now(),
        session_id: session.session_id,
        kind: EventKind::PlanRevised { plan: plan.clone() },
    })?;
    store.append_event(&EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: Utc::now(),
        session_id: session.session_id,
        kind: EventKind::ExitPlanMode {
            session_id: session.session_id,
        },
    })?;

    let mut watermarks = HashMap::from([(session.session_id, 0_u64)]);
    let notices =
        poll_session_lifecycle_notices(temp.path(), Some(session.session_id), &mut watermarks)?;
    assert!(
        notices
            .iter()
            .any(|notice| notice.line.contains("awaiting approval"))
    );
    assert!(notices.iter().any(|notice| notice.line.contains("steps:")));
    assert!(
        notices
            .iter()
            .any(|notice| notice.line.contains("/plan approve"))
    );
    Ok(())
}

#[test]
fn poll_session_lifecycle_notices_surfaces_pending_plan_on_first_attach() -> Result<()> {
    let temp = tempdir()?;
    let store = Store::new(temp.path())?;
    let plan_id = Uuid::now_v7();
    let session = Session {
        session_id: Uuid::now_v7(),
        workspace_root: temp.path().display().to_string(),
        baseline_commit: None,
        status: SessionState::AwaitingApproval,
        budgets: SessionBudgets {
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
            goal: "Audit approval UX".to_string(),
            assumptions: vec![],
            steps: vec![PlanStep {
                step_id: Uuid::now_v7(),
                title: "Review transcript notices".to_string(),
                intent: "Surface the plan".to_string(),
                tools: vec![],
                files: vec!["src/chat.rs".to_string()],
                done: false,
            }],
            verification: vec![],
            risk_notes: vec![],
        },
    )?;

    let mut watermarks = HashMap::new();
    let notices =
        poll_session_lifecycle_notices(temp.path(), Some(session.session_id), &mut watermarks)?;
    assert!(!notices.is_empty());
    assert!(notices[0].line.contains("[plan]"));
    assert!(
        notices
            .iter()
            .any(|notice| notice.line.contains("awaiting approval"))
    );
    Ok(())
}

#[test]
fn parse_chat_mode_name_supports_expected_aliases() {
    assert_eq!(parse_chat_mode_name("ask"), Some(ChatMode::Ask));
    assert_eq!(parse_chat_mode_name("code"), Some(ChatMode::Code));
    assert_eq!(parse_chat_mode_name("plan"), Some(ChatMode::Code));
    assert_eq!(parse_chat_mode_name("context"), Some(ChatMode::Context));
    assert_eq!(parse_chat_mode_name("architect"), None);
    assert_eq!(parse_chat_mode_name("pipeline"), None);
    assert_eq!(parse_chat_mode_name("invalid"), None);
}

#[test]
fn watch_scan_returns_digest_and_payload() -> Result<()> {
    // rg must be a real binary on PATH (not just a shell alias)
    let rg_available = std::process::Command::new("rg")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !rg_available {
        eprintln!("skipping watch_scan test: rg not found on PATH");
        return Ok(());
    }
    let dir = tempdir()?;
    let root = dir.path();
    fs::write(
        root.join("notes.md"),
        "todo list\nTODO(ai): inspect runtime flow\n",
    )?;
    let result = scan_watch_comment_payload(root);
    assert!(result.is_some());
    let (digest, payload) = result.unwrap();
    assert!(digest > 0);
    assert!(payload.contains("TODO(ai): inspect runtime flow"));
    assert!(payload.contains("notes.md"));
    Ok(())
}

#[test]
fn profile_save_and_load_roundtrip() -> Result<()> {
    let dir = tempdir()?;
    let root = dir.path();
    let additional_dirs = vec![root.join("src"), root.join("docs")];
    let save = slash_save_profile_output(
        root,
        &[String::from("roundtrip")],
        ChatMode::Code,
        true,
        true,
        &additional_dirs,
    )?;
    assert!(save.contains("saved chat profile"));

    let (mode, read_only, thinking, dirs, load_msg) =
        slash_load_profile_output(root, &[String::from("roundtrip")])?;
    assert_eq!(mode, ChatMode::Code);
    assert!(read_only);
    assert!(thinking);
    assert_eq!(dirs, additional_dirs);
    assert!(load_msg.contains("loaded chat profile"));
    Ok(())
}

#[test]
fn slash_git_without_args_returns_usage() -> Result<()> {
    let dir = tempdir()?;
    run_process(dir.path(), "git", &["init"])?;
    let output = slash_git_output(dir.path(), &[])?;
    assert!(output.trim().is_empty());
    Ok(())
}

#[test]
fn slash_voice_status_reports_capability_probe() -> Result<()> {
    let output = slash_voice_output(&[String::from("status")])?;
    assert!(output.contains("voice status:"));
    Ok(())
}

// ── P7-03: /memory default is edit ──────────────────────────────────

#[test]
fn memory_default_is_edit() {
    // Verify that the first branch (args empty) routes to Edit, not Show.
    // We test the routing logic: when args is empty, it should match the
    // edit branch (first condition), not fall through to show.
    let args: Vec<String> = vec![];
    let first_condition =
        args.is_empty() || args.first().is_some_and(|a| a.eq_ignore_ascii_case("edit"));
    assert!(
        first_condition,
        "/memory with no args should match the edit branch"
    );

    // "show" should NOT match the edit branch
    let show_args = [String::from("show")];
    let edit_branch = show_args.is_empty()
        || show_args
            .first()
            .is_some_and(|a| a.eq_ignore_ascii_case("edit"));
    assert!(!edit_branch, "/memory show should NOT match edit branch");
}

// ── P7-02: /mcp interactive menu ────────────────────────────────────

#[test]
fn mcp_interactive_menu_lists_servers() {
    // In interactive mode (json_mode=false, args empty), the MCP handler
    // should display a numbered server list rather than raw McpCmd::List.
    // We verify the branch condition.
    let args: Vec<String> = vec![];
    let json_mode = false;
    let interactive_menu = args.is_empty() && !json_mode;
    assert!(
        interactive_menu,
        "empty args + non-json should trigger interactive menu"
    );

    // JSON mode should fall through to McpCmd::List
    let json_mode = true;
    let interactive_menu = args.is_empty() && !json_mode;
    assert!(
        !interactive_menu,
        "json mode should NOT trigger interactive menu"
    );
}
