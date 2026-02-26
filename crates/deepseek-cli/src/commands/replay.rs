use anyhow::{Result, anyhow};
use chrono::Utc;
use deepseek_core::{AppConfig, EventEnvelope, EventKind};
use deepseek_store::{ReplayCassetteRecord, Store};
use serde::Serialize;
use serde_json::json;
use std::path::Path;
use uuid::Uuid;

use crate::ReplayCmd;
use crate::context::*;
use crate::output::*;

#[derive(Debug, Default, Serialize)]
pub(crate) struct ReplayValidation {
    pub(crate) passed: bool,
    pub(crate) monotonic_seq: bool,
    pub(crate) missing_tool_results: Vec<String>,
    pub(crate) orphan_tool_results: Vec<String>,
}

pub(crate) fn run_replay(cwd: &Path, cmd: ReplayCmd, json_mode: bool) -> Result<()> {
    match cmd {
        ReplayCmd::Run(args) => {
            let cfg = AppConfig::ensure(cwd)?;
            if cfg.replay.strict_mode && !args.deterministic {
                return Err(anyhow!(
                    "replay.strict_mode=true requires --deterministic=true"
                ));
            }
            let session_id = Uuid::parse_str(&args.session_id)?;
            let store = Store::new(cwd)?;
            let events = read_session_events(cwd, session_id)?;
            let validation = validate_replay_events(&events);
            if cfg.replay.strict_mode && !validation.passed {
                return Err(anyhow!(
                    "strict replay validation failed: {}",
                    serde_json::to_string(&validation)?
                ));
            }
            let projection = store.rebuild_from_events(session_id)?;
            let events_replayed = events.len() as u64;
            let tool_results_replayed = events
                .iter()
                .filter(|event| matches!(event.kind, EventKind::ToolResult { .. }))
                .count() as u64;
            let payload = json!({
                "session_id": session_id,
                "deterministic": args.deterministic,
                "strict_mode": cfg.replay.strict_mode,
                "events_replayed": events_replayed,
                "tool_results_replayed": tool_results_replayed,
                "turns": projection.transcript.len(),
                "steps": projection.step_status.len(),
                "validation": validation,
            });
            store.insert_replay_cassette(&ReplayCassetteRecord {
                cassette_id: Uuid::now_v7(),
                session_id,
                deterministic: args.deterministic,
                events_count: events_replayed,
                payload_json: payload.to_string(),
                created_at: Utc::now().to_rfc3339(),
            })?;
            append_control_event(
                cwd,
                EventKind::ReplayExecuted {
                    session_id,
                    deterministic: args.deterministic,
                    events_replayed,
                },
            )?;
            if json_mode {
                print_json(&payload)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&payload)?);
            }
        }
        ReplayCmd::List(args) => {
            let store = Store::new(cwd)?;
            let session_id = if let Some(raw) = args.session_id.as_deref() {
                Some(Uuid::parse_str(raw)?)
            } else {
                None
            };
            let rows = store.list_replay_cassettes(session_id, args.limit)?;
            if json_mode {
                print_json(&rows)?;
            } else if rows.is_empty() {
                println!("no replay cassettes found");
            } else {
                for row in rows {
                    println!(
                        "{} session={} deterministic={} events={} created_at={}",
                        row.cassette_id,
                        row.session_id,
                        row.deterministic,
                        row.events_count,
                        row.created_at
                    );
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_replay_events(events: &[EventEnvelope]) -> ReplayValidation {
    use std::collections::{BTreeSet, HashSet};

    let mut proposed = HashSet::new();
    let mut approved = HashSet::new();
    let mut results = HashSet::new();
    let mut missing_tool_results = BTreeSet::new();
    let mut orphan_tool_results = BTreeSet::new();

    let mut monotonic_seq = true;
    let mut last_seq = 0_u64;
    for event in events {
        if event.seq_no < last_seq {
            monotonic_seq = false;
        }
        last_seq = event.seq_no;
        match &event.kind {
            EventKind::ToolProposed { proposal } => {
                if proposal.approved {
                    proposed.insert(proposal.invocation_id);
                }
            }
            EventKind::ToolApproved { invocation_id } => {
                approved.insert(*invocation_id);
            }
            EventKind::ToolResult { result } => {
                results.insert(result.invocation_id);
            }
            _ => {}
        }
    }

    for invocation_id in proposed.union(&approved) {
        if !results.contains(invocation_id) {
            missing_tool_results.insert(invocation_id.to_string());
        }
    }
    for invocation_id in &results {
        if !approved.contains(invocation_id) && !proposed.contains(invocation_id) {
            orphan_tool_results.insert(invocation_id.to_string());
        }
    }

    let missing_tool_results = missing_tool_results.into_iter().collect::<Vec<_>>();
    let orphan_tool_results = orphan_tool_results.into_iter().collect::<Vec<_>>();
    let passed = monotonic_seq && missing_tool_results.is_empty() && orphan_tool_results.is_empty();
    ReplayValidation {
        passed,
        monotonic_seq,
        missing_tool_results,
        orphan_tool_results,
    }
}
