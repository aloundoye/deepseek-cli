use anyhow::{Result, anyhow};
use deepseek_core::{AppConfig, EventKind};
use deepseek_memory::MemoryManager;
use deepseek_store::Store;
use serde::Serialize;
use serde_json::json;
use std::fs;
use std::path::Path;
use uuid::Uuid;

use crate::context::*;
use crate::output::*;
use crate::util::*;
use crate::{CompactArgs, RewindArgs};

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CompactSummary {
    pub(crate) summary_id: Uuid,
    pub(crate) from_turn: u64,
    pub(crate) to_turn: u64,
    pub(crate) token_delta_estimate: i64,
}

pub(crate) fn compact_now(cwd: &Path, from_turn: Option<u64>) -> Result<CompactSummary> {
    let store = Store::new(cwd)?;
    let session = store
        .load_latest_session()?
        .ok_or_else(|| anyhow!("no session found to compact"))?;
    let projection = store.rebuild_from_events(session.session_id)?;
    if projection.transcript.is_empty() {
        return Err(anyhow!("no transcript to compact"));
    }
    let from_turn = from_turn.unwrap_or(1).max(1);
    let transcript_len = projection.transcript.len() as u64;
    if from_turn > transcript_len {
        return Err(anyhow!(
            "from_turn {} exceeds transcript length {}",
            from_turn,
            transcript_len
        ));
    }
    let selected = projection
        .transcript
        .iter()
        .skip((from_turn - 1) as usize)
        .cloned()
        .collect::<Vec<_>>();
    let summary_id = Uuid::now_v7();
    let full_text = selected.join("\n");
    let before_tokens = estimate_tokens(&full_text);
    let summary_lines = selected
        .iter()
        .take(12)
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.len() > 200 {
                format!("- {}...", &trimmed[..200])
            } else {
                format!("- {trimmed}")
            }
        })
        .collect::<Vec<_>>();
    let summary = format!(
        "Compaction summary {}\nfrom_turn: {}\nto_turn: {}\n\n{}",
        summary_id,
        from_turn,
        transcript_len,
        summary_lines.join("\n")
    );
    let token_delta_estimate = before_tokens as i64 - estimate_tokens(&summary) as i64;
    let replay_pointer = format!(".deepseek/compactions/{summary_id}.md");
    let summary_path = cwd.join(&replay_pointer);
    if let Some(parent) = summary_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(summary_path, summary)?;
    append_control_event(
        cwd,
        EventKind::ContextCompactedV1 {
            summary_id,
            from_turn,
            to_turn: transcript_len,
            token_delta_estimate,
            replay_pointer,
        },
    )?;
    Ok(CompactSummary {
        summary_id,
        from_turn,
        to_turn: transcript_len,
        token_delta_estimate,
    })
}

pub(crate) fn rewind_now(
    cwd: &Path,
    to_checkpoint: Option<String>,
) -> Result<deepseek_store::CheckpointRecord> {
    let memory = MemoryManager::new(cwd)?;
    let checkpoint_id = if let Some(value) = to_checkpoint.as_deref() {
        Uuid::parse_str(value)?
    } else {
        memory
            .list_checkpoints()?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no checkpoints available"))?
            .checkpoint_id
    };
    let checkpoint = memory.rewind_to_checkpoint(checkpoint_id)?;
    append_control_event(
        cwd,
        EventKind::CheckpointRewoundV1 {
            checkpoint_id: checkpoint.checkpoint_id,
            reason: checkpoint.reason.clone(),
        },
    )?;
    Ok(checkpoint)
}

pub(crate) fn run_rewind(cwd: &Path, args: RewindArgs, json_mode: bool) -> Result<()> {
    let memory = MemoryManager::new(cwd)?;
    let checkpoint_id = if let Some(value) = args.to_checkpoint.as_deref() {
        Uuid::parse_str(value)?
    } else {
        let checkpoints = memory.list_checkpoints()?;
        if checkpoints.is_empty() {
            let payload = json!({"rewound": false, "reason": "no_checkpoints"});
            if json_mode {
                print_json(&payload)?;
            } else {
                println!("no checkpoints available");
            }
            return Ok(());
        }
        checkpoints
            .into_iter()
            .next()
            .expect("non-empty checkpoints")
            .checkpoint_id
    };
    if !args.yes {
        return Err(anyhow!(
            "rewind requires --yes to confirm (target checkpoint: {})",
            checkpoint_id
        ));
    }
    let checkpoint = memory.rewind_to_checkpoint(checkpoint_id)?;
    append_control_event(
        cwd,
        EventKind::CheckpointRewoundV1 {
            checkpoint_id: checkpoint.checkpoint_id,
            reason: checkpoint.reason.clone(),
        },
    )?;
    let payload = json!({
        "checkpoint_id": checkpoint.checkpoint_id,
        "reason": checkpoint.reason,
        "snapshot_path": checkpoint.snapshot_path,
        "files_count": checkpoint.files_count,
        "created_at": checkpoint.created_at,
        "rewound": true,
    });
    if json_mode {
        print_json(&payload)?;
    } else {
        println!(
            "rewound to checkpoint {} (files={})",
            payload["checkpoint_id"].as_str().unwrap_or_default(),
            payload["files_count"].as_u64().unwrap_or(0)
        );
    }
    Ok(())
}

pub(crate) fn run_compact(cwd: &Path, args: CompactArgs, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;
    let store = Store::new(cwd)?;
    let session = store
        .load_latest_session()?
        .ok_or_else(|| anyhow!("no session found to compact"))?;
    let projection = store.rebuild_from_events(session.session_id)?;
    if projection.transcript.is_empty() {
        let payload = json!({"status":"no_op", "reason":"empty_transcript"});
        if json_mode {
            print_json(&payload)?;
        } else {
            println!("no transcript to compact");
        }
        return Ok(());
    }

    let from_turn = args.from_turn.unwrap_or(1).max(1);
    let transcript_len = projection.transcript.len() as u64;
    if from_turn > transcript_len {
        return Err(anyhow!(
            "--from-turn {} exceeds transcript length {}",
            from_turn,
            transcript_len
        ));
    }

    let selected = projection
        .transcript
        .iter()
        .skip((from_turn - 1) as usize)
        .cloned()
        .collect::<Vec<_>>();
    let summary_id = Uuid::now_v7();
    let full_text = selected.join("\n");
    let before_tokens = estimate_tokens(&full_text);
    let summary_lines = selected
        .iter()
        .take(12)
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.len() > 200 {
                format!("- {}...", &trimmed[..200])
            } else {
                format!("- {trimmed}")
            }
        })
        .collect::<Vec<_>>();
    let summary = format!(
        "Compaction summary {}\nfrom_turn: {}\nto_turn: {}\n\n{}",
        summary_id,
        from_turn,
        transcript_len,
        summary_lines.join("\n")
    );
    let after_tokens = estimate_tokens(&summary);
    let token_delta_estimate = before_tokens as i64 - after_tokens as i64;
    let replay_pointer = format!(".deepseek/compactions/{summary_id}.md");
    let payload = json!({
        "summary_id": summary_id,
        "from_turn": from_turn,
        "to_turn": transcript_len,
        "token_delta_estimate": token_delta_estimate,
        "replay_pointer": replay_pointer,
    });

    if cfg.context.compact_preview && !args.yes {
        if json_mode {
            print_json(&json!({
                "preview": true,
                "persisted": false,
                "summary": summary,
                "result": payload,
            }))?;
        } else {
            println!("compaction preview (not persisted):");
            println!("{summary}");
            println!("rerun with --yes to persist");
        }
        return Ok(());
    }

    let summary_path = cwd.join(&replay_pointer);
    if let Some(parent) = summary_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&summary_path, summary)?;
    append_control_event(
        cwd,
        EventKind::ContextCompactedV1 {
            summary_id,
            from_turn,
            to_turn: transcript_len,
            token_delta_estimate,
            replay_pointer: replay_pointer.clone(),
        },
    )?;

    if json_mode {
        print_json(&json!({
            "preview": false,
            "persisted": true,
            "result": payload,
        }))?;
    } else {
        println!(
            "compacted turns {}..{} summary_id={} token_delta_estimate={} replay_pointer={}",
            from_turn, transcript_len, summary_id, token_delta_estimate, replay_pointer
        );
    }

    Ok(())
}
