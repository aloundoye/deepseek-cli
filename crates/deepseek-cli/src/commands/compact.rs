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

pub(crate) fn compact_now(
    cwd: &Path,
    from_turn: Option<u64>,
    focus: Option<&str>,
) -> Result<CompactSummary> {
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

    // When a focus topic is provided, prioritize lines mentioning the topic
    let summary_lines = build_compact_summary_lines(&selected, focus);

    let focus_header = focus.map(|f| format!("\nfocus: {f}")).unwrap_or_default();
    let summary = format!(
        "Compaction summary {}\nfrom_turn: {}\nto_turn: {}{}\n\n{}",
        summary_id,
        from_turn,
        transcript_len,
        focus_header,
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
        EventKind::ContextCompacted {
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

/// Build summary lines for compaction. When a focus topic is provided,
/// relevant lines are placed first, and the budget is increased for them.
fn build_compact_summary_lines(selected: &[String], focus: Option<&str>) -> Vec<String> {
    let format_line = |line: &str| {
        let trimmed = line.trim();
        if trimmed.len() > 200 {
            format!("- {}...", &trimmed[..trimmed.floor_char_boundary(200)])
        } else {
            format!("- {trimmed}")
        }
    };

    if let Some(topic) = focus {
        let topic_lower = topic.to_ascii_lowercase();
        let (relevant, other): (Vec<_>, Vec<_>) = selected
            .iter()
            .partition(|line| line.to_ascii_lowercase().contains(&topic_lower));

        let mut lines = Vec::new();
        // Give focus lines more budget (up to 16), then fill with others (up to 8)
        for line in relevant.iter().take(16) {
            lines.push(format_line(line));
        }
        if !lines.is_empty() && !other.is_empty() {
            lines.push(format!("--- (non-{topic} context) ---"));
        }
        for line in other.iter().take(8) {
            lines.push(format_line(line));
        }
        lines
    } else {
        selected.iter().take(12).map(|l| format_line(l)).collect()
    }
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
        EventKind::CheckpointRewound {
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
        EventKind::CheckpointRewound {
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
    let summary_lines = build_compact_summary_lines(&selected, args.focus.as_deref());
    let focus_header = args
        .focus
        .as_deref()
        .map(|f| format!("\nfocus: {f}"))
        .unwrap_or_default();
    let summary = format!(
        "Compaction summary {}\nfrom_turn: {}\nto_turn: {}{}\n\n{}",
        summary_id,
        from_turn,
        transcript_len,
        focus_header,
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
        EventKind::ContextCompacted {
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

/// Summarize older conversation messages using a heuristic extraction approach.
/// Extracts key decisions, file paths, and task outcomes into a compressed summary
/// that's roughly <25% of the original token count.
#[allow(dead_code)]
pub(crate) fn summarize_conversation(transcript: &[String]) -> String {
    if transcript.is_empty() {
        return String::new();
    }

    let original_tokens = transcript.iter().map(|t| estimate_tokens(t)).sum::<u64>();
    let target_lines = (original_tokens / 40).clamp(5, 50) as usize; // ~10 tokens per summary line

    let mut summary_parts = Vec::new();
    summary_parts.push(format!(
        "Conversation summary ({} turns, ~{} tokens compressed):",
        transcript.len(),
        original_tokens
    ));

    // Extract key information from each turn
    for (i, turn) in transcript.iter().enumerate() {
        let trimmed = turn.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Take first meaningful line as a summary
        let first_line = trimmed
            .lines()
            .find(|l| !l.trim().is_empty() && l.trim().len() > 5)
            .unwrap_or(trimmed);
        let truncated = if first_line.len() > 150 {
            format!("{}...", &first_line[..first_line.floor_char_boundary(150)])
        } else {
            first_line.to_string()
        };
        summary_parts.push(format!("[turn {}] {}", i + 1, truncated));

        if summary_parts.len() >= target_lines {
            break;
        }
    }

    summary_parts.join("\n")
}

/// Check if the current context usage exceeds the auto-compact threshold.
#[allow(dead_code)]
pub(crate) fn should_auto_compact(cfg: &AppConfig, transcript: &[String]) -> bool {
    if transcript.len() < 4 {
        return false; // too few turns to compact
    }
    let total_chars: u64 = transcript.iter().map(|t| t.len() as u64).sum();
    let estimated_tokens = total_chars / 4;
    let context_window = cfg.llm.context_window_tokens;
    if context_window == 0 {
        return false;
    }
    let reserved = cfg.context.reserved_overhead_tokens + cfg.context.response_budget_tokens;
    let usable = context_window.saturating_sub(reserved);
    let threshold = cfg.context.auto_compact_threshold;
    let ratio = estimated_tokens as f64 / usable as f64;
    ratio >= threshold as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_compact_summary_lines_no_focus() {
        let lines: Vec<String> = (1..=20).map(|i| format!("turn {i} content")).collect();
        let result = build_compact_summary_lines(&lines, None);
        assert_eq!(result.len(), 12, "should take up to 12 lines without focus");
        assert!(result[0].contains("turn 1"));
    }

    #[test]
    fn build_compact_summary_lines_with_focus() {
        let lines = vec![
            "setup database".to_string(),
            "fix authentication bug".to_string(),
            "update database schema".to_string(),
            "refactor logging".to_string(),
            "add database migration".to_string(),
            "update readme".to_string(),
        ];
        let result = build_compact_summary_lines(&lines, Some("database"));
        // Relevant lines first: "setup database", "update database schema", "add database migration"
        assert!(result[0].contains("database"));
        assert!(result[1].contains("database"));
        assert!(result[2].contains("database"));
        // Then separator and non-relevant lines
        assert!(result.iter().any(|l| l.contains("non-database context")));
    }

    #[test]
    fn build_compact_summary_lines_focus_case_insensitive() {
        let lines = vec![
            "fix AUTH bug".to_string(),
            "update tests".to_string(),
            "auth module refactored".to_string(),
        ];
        let result = build_compact_summary_lines(&lines, Some("auth"));
        assert!(result[0].contains("AUTH"));
        assert!(result[1].contains("auth"));
    }

    #[test]
    fn build_compact_summary_lines_empty() {
        let lines: Vec<String> = vec![];
        let result = build_compact_summary_lines(&lines, Some("anything"));
        assert!(result.is_empty());
    }
}
