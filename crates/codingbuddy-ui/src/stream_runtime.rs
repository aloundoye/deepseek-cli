use crate::{ChatShell, TuiStreamEvent, display_image_inline, truncate_inline};
use std::io;
use std::sync::mpsc;

pub(crate) enum StreamEventResult {
    Continue,
    Done(String),
}

pub(crate) struct StreamRuntimeState<'a> {
    pub shell: &'a mut ChatShell,
    pub streaming_buffer: &'a mut String,
    pub active_phase: &'a mut Option<(u64, String)>,
    pub pending_approval: &'a mut Option<(String, String, mpsc::Sender<bool>)>,
    pub info_line: &'a mut String,
    pub is_processing: &'a mut bool,
    pub streaming_in_code_block: &'a mut bool,
    pub streaming_in_diff_block: &'a mut bool,
    pub streaming_code_block_lang: &'a mut String,
}

pub(crate) fn filter_stream_event(
    event: TuiStreamEvent,
    cancelled: &mut bool,
) -> Option<TuiStreamEvent> {
    if !*cancelled {
        return Some(event);
    }

    match event {
        TuiStreamEvent::Done(_) | TuiStreamEvent::Error(_) => {
            *cancelled = false;
        }
        TuiStreamEvent::ApprovalNeeded { response_tx, .. } => {
            let _ = response_tx.send(false);
        }
        _ => {}
    }
    None
}

pub(crate) fn handle_stream_event(
    event: TuiStreamEvent,
    state: &mut StreamRuntimeState<'_>,
) -> StreamEventResult {
    match event {
        TuiStreamEvent::ContentDelta(text) => {
            if state.shell.is_thinking {
                state.shell.is_thinking = false;
                state.shell.thinking_buffer.clear();
            }
            state.streaming_buffer.push_str(&text);
            state.shell.append_streaming(&text);
            StreamEventResult::Continue
        }
        TuiStreamEvent::ReasoningDelta(text) => {
            if state.shell.thinking_visibility == "raw" {
                if !state.shell.is_thinking {
                    state.shell.is_thinking = true;
                    state.shell.thinking_buffer.clear();
                }
                state.shell.thinking_buffer.push_str(&text);
            } else {
                state.shell.is_thinking = true;
                if state.shell.thinking_buffer.is_empty() {
                    let label = state
                        .active_phase
                        .as_ref()
                        .map(|(_, phase)| phase.as_str())
                        .unwrap_or("reasoning");
                    state.shell.thinking_buffer = format!("{label}: analyzing...");
                }
            }
            StreamEventResult::Continue
        }
        TuiStreamEvent::ToolActive(name) => {
            state.shell.is_thinking = false;
            state.shell.active_tool = Some(name);
            StreamEventResult::Continue
        }
        TuiStreamEvent::ToolCallStart {
            tool_name,
            args_summary,
        } => {
            state.shell.is_thinking = false;
            state.shell.push_tool_call(&tool_name, &args_summary);
            state.shell.active_tool = Some(tool_name);
            StreamEventResult::Continue
        }
        TuiStreamEvent::ToolCallEnd {
            tool_name,
            duration_ms,
            summary,
            success,
        } => {
            let marker = if success { "✓" } else { "✗" };
            state
                .shell
                .push_tool_result(&format!("{marker} {tool_name}"), duration_ms, &summary);
            state.shell.active_tool = None;
            StreamEventResult::Continue
        }
        TuiStreamEvent::ModeTransition { from, to, reason } => {
            state.shell.agent_mode = to.clone();
            let label = format!("mode transition {from} -> {to} ({reason})");
            state.shell.push_system(label);
            *state.info_line = format!("mode: {from} -> {to}");
            StreamEventResult::Continue
        }
        TuiStreamEvent::SubagentSpawned { run_id, name, goal } => {
            let goal_compact = truncate_inline(&goal.replace('\n', " "), 120);
            state
                .shell
                .push_mission_control(format!("spawned {name} ({run_id}) goal={goal_compact}"));
            state
                .shell
                .push_system(format!("[subagent] started {name}: {goal_compact}"));
            *state.info_line = format!("subagent started: {name}");
            StreamEventResult::Continue
        }
        TuiStreamEvent::SubagentCompleted {
            run_id,
            name,
            summary,
        } => {
            let summary_compact = truncate_inline(&summary.replace('\n', " "), 120);
            state.shell.push_mission_control(format!(
                "completed {name} ({run_id}) summary={summary_compact}"
            ));
            state
                .shell
                .push_system(format!("[subagent] completed {name}: {summary_compact}"));
            *state.info_line = format!("subagent completed: {name}");
            StreamEventResult::Continue
        }
        TuiStreamEvent::SubagentFailed {
            run_id,
            name,
            error,
        } => {
            let error_compact = truncate_inline(&error.replace('\n', " "), 120);
            state
                .shell
                .push_mission_control(format!("failed {name} ({run_id}) error={error_compact}"));
            state
                .shell
                .push_error(format!("[subagent] failed {name}: {error_compact}"));
            *state.info_line = format!("subagent failed: {name}");
            StreamEventResult::Continue
        }
        TuiStreamEvent::SystemNotice { line, error } => {
            if line.starts_with("[background]")
                || line.starts_with("[task]")
                || line.starts_with("[plan]")
            {
                state.shell.push_mission_control(line.clone());
            }
            if error {
                state.shell.push_error(line.clone());
            } else {
                state.shell.push_system(line.clone());
            }
            *state.info_line = truncate_inline(&line, 96);
            StreamEventResult::Continue
        }
        TuiStreamEvent::WatchTriggered { comment_count, .. } => {
            state.shell.push_system(format!(
                "[watch: {comment_count} comment(s) detected, auto-triggering]"
            ));
            *state.info_line = format!("watch: {comment_count} hints");
            StreamEventResult::Continue
        }
        TuiStreamEvent::ImageDisplay { data, label } => {
            if !display_image_inline(&data) {
                state
                    .shell
                    .push_system(format!("[image: {label} ({} bytes)]", data.len()));
            } else {
                state.shell.push_system(format!("[image: {label}]"));
            }
            StreamEventResult::Continue
        }
        TuiStreamEvent::ClearStreamingText => {
            state.streaming_buffer.clear();
            state.shell.clear_streaming_text();
            StreamEventResult::Continue
        }
        TuiStreamEvent::DiffApplied {
            path,
            hunks,
            added,
            removed,
        } => {
            state.shell.push_system(format!(
                "  \u{2502} {} \u{2014} {} hunk(s), +{} -{}",
                path, hunks, added, removed
            ));
            StreamEventResult::Continue
        }
        TuiStreamEvent::UsageSummary {
            input_tokens,
            output_tokens,
            cache_hit_tokens,
            cost_usd,
        } => {
            let cache_info = if cache_hit_tokens > 0 {
                format!(" (cache hit: {})", cache_hit_tokens)
            } else {
                String::new()
            };
            state.shell.push_system(format!(
                "\u{2500}\u{2500} tokens: {}in / {}out{} \u{2502} cost: ${:.4}",
                input_tokens, output_tokens, cache_info, cost_usd
            ));
            StreamEventResult::Continue
        }
        TuiStreamEvent::RoleHeader { role, model } => {
            state.shell.push_system(format!(
                "\u{2501}\u{2501} {} ({}) \u{2501}\u{2501}",
                role, model
            ));
            StreamEventResult::Continue
        }
        TuiStreamEvent::ApprovalNeeded {
            tool_name,
            args_summary,
            response_tx,
        } => {
            let compact_args = truncate_inline(&args_summary.replace('\n', " "), 96);
            *state.info_line = format!(
                "ACTION REQUIRED: `{tool_name}` {compact_args} [press Y to approve / any key denies]"
            );
            state.shell.push_system(format!(
                "ACTION REQUIRED: `{tool_name}` {compact_args} [press Y to approve / any key denies]"
            ));
            use std::io::Write as _;
            let _ = write!(io::stdout(), "\x07");
            let _ = io::stdout().flush();
            *state.pending_approval = Some((tool_name, args_summary, response_tx));
            StreamEventResult::Continue
        }
        TuiStreamEvent::Error(msg) => {
            state.streaming_buffer.clear();
            *state.streaming_in_code_block = false;
            *state.streaming_in_diff_block = false;
            state.streaming_code_block_lang.clear();
            *state.active_phase = None;
            state.shell.is_thinking = false;
            state.shell.thinking_buffer.clear();
            state.shell.push_error(&msg);
            *state.info_line = format!("error: {msg}");
            *state.is_processing = false;
            state.shell.active_tool = None;
            StreamEventResult::Continue
        }
        TuiStreamEvent::Done(output) => StreamEventResult::Done(output),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_stream_event_drops_approval_when_cancelled() {
        let (tx, rx) = mpsc::channel();
        let mut cancelled = true;
        let event = TuiStreamEvent::ApprovalNeeded {
            tool_name: "fs_edit".to_string(),
            args_summary: "file=main.rs".to_string(),
            response_tx: tx,
        };
        assert!(filter_stream_event(event, &mut cancelled).is_none());
        assert_eq!(rx.recv().ok(), Some(false));
        assert!(cancelled);
    }

    #[test]
    fn handle_stream_event_promotes_background_notice() {
        let mut shell = ChatShell::default();
        let mut streaming_buffer = String::new();
        let mut active_phase = None;
        let mut pending_approval = None;
        let mut info_line = String::new();
        let mut is_processing = true;
        let mut streaming_in_code_block = false;
        let mut streaming_in_diff_block = false;
        let mut streaming_code_block_lang = String::new();
        let mut state = StreamRuntimeState {
            shell: &mut shell,
            streaming_buffer: &mut streaming_buffer,
            active_phase: &mut active_phase,
            pending_approval: &mut pending_approval,
            info_line: &mut info_line,
            is_processing: &mut is_processing,
            streaming_in_code_block: &mut streaming_in_code_block,
            streaming_in_diff_block: &mut streaming_in_diff_block,
            streaming_code_block_lang: &mut streaming_code_block_lang,
        };

        let result = handle_stream_event(
            TuiStreamEvent::SystemNotice {
                line: "[background] task completed: audit".to_string(),
                error: false,
            },
            &mut state,
        );

        assert!(matches!(result, StreamEventResult::Continue));
        assert_eq!(state.shell.mission_control_lines.len(), 1);
        assert_eq!(state.shell.transcript.len(), 1);
        assert!(state.info_line.contains("task completed"));
    }
}
