use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget, Wrap};
use ratatui::{Terminal, TerminalOptions, Viewport};
use std::collections::VecDeque;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

mod image;
mod input;
mod keybindings;
mod panels;
mod rendering;
mod slash_commands;
mod state;
mod statusline;
mod stream_runtime;
mod theme;

pub use image::{ImageProtocol, detect_image_protocol, display_image_inline, render_inline_image};
pub use input::expand_at_mentions;
use input::*;
#[cfg(test)]
use keybindings::KeyBindingsFile;
pub use keybindings::{KeyBindings, load_keybindings};
use panels::{load_artifact_lines, render_mission_control_panel};
pub(crate) use rendering::truncate_inline;
use rendering::*;
pub use rendering::{PROMPT_SUGGESTIONS, TuiStreamEvent, render_prompt_suggestions};
use slash_commands::{
    SLASH_COMMAND_CATALOG, slash_command_suggestions, slash_suggestion_to_command,
};
pub use slash_commands::{SlashCommand, slash_command_catalog_entries};
pub use state::{
    AutocompleteState, ChatShell, GhostTextState, MessageKind, MlCompletionCallback,
    ModelPickerState, REWIND_ACTIONS, RewindPickerPhase, RewindPickerState, TranscriptEntry,
    format_relative_time,
};
use state::{MODEL_CHOICES, VimMode};
use statusline::render_statusline_spans;
pub use statusline::{UiStatus, render_statusline};
use stream_runtime::{
    StreamEventResult, StreamRuntimeState, filter_stream_event, handle_stream_event,
};
pub use theme::TuiTheme;
pub fn run_tui_shell<F>(status: UiStatus, mut on_submit: F) -> Result<()>
where
    F: FnMut(&str) -> Result<String>,
{
    let (tx, rx) = mpsc::channel();
    run_tui_shell_with_bindings(
        status,
        KeyBindings::default(),
        TuiTheme::default(),
        false,
        rx,
        move |prompt| {
            let result = on_submit(prompt);
            match result {
                Ok(output) => {
                    let _ = tx.send(TuiStreamEvent::Done(output));
                }
                Err(e) => {
                    let _ = tx.send(TuiStreamEvent::Error(e.to_string()));
                }
            }
        },
        || None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_tui_shell_with_bindings<F, S>(
    mut status: UiStatus,
    bindings: KeyBindings,
    _theme: TuiTheme,
    reduced_motion: bool,
    stream_rx: mpsc::Receiver<TuiStreamEvent>,
    mut on_submit: F,
    mut refresh_status: S,
    ml_completion: Option<MlCompletionCallback>,
) -> Result<()>
where
    F: FnMut(&str),
    S: FnMut() -> Option<UiStatus>,
{
    // Install a SIGINT handler that sets a flag instead of killing the process.
    let sigint_flag = Arc::new(AtomicBool::new(false));
    #[cfg(unix)]
    {
        let flag = Arc::clone(&sigint_flag);
        signal_hook::flag::register(signal_hook::consts::SIGINT, flag)?;
    }

    // Set a panic hook that restores the terminal before printing the panic.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = crossterm::execute!(io::stdout(), crossterm::cursor::Show);
        prev_hook(info);
    }));

    // Clear terminal and print welcome banner so it feels like a fresh session.
    {
        use std::io::Write;
        let mut out = io::stdout();
        // Clear screen + move cursor home.
        // Skip \x1b[3J (clear scrollback) — it's a non-standard xterm extension
        // that causes issues in VS Code's integrated terminal and other emulators.
        out.write_all(b"\x1b[2J\x1b[H")?;

        // ASCII art logo + info
        let version = env!("CARGO_PKG_VERSION");
        let model = &status.model;
        let cwd = &status.working_directory;

        // Shark logo in blue, info text in white/gray
        writeln!(out)?;
        writeln!(
            out,
            "\x1b[1;34m        ▄       \x1b[0m \x1b[1mCodingBuddy\x1b[0m v{version}"
        )?;
        writeln!(
            out,
            "\x1b[1;34m    ▗▄▀ ● ▀▀▀▀▄ \x1b[0m \x1b[36m{model}\x1b[0m"
        )?;
        writeln!(
            out,
            "\x1b[1;34m    ▝▀▄▄▄▄▄▄▀▀▘ \x1b[0m \x1b[90m{cwd}\x1b[0m"
        )?;
        writeln!(out, "\x1b[1;34m        ▀        \x1b[0m")?;
        writeln!(out)?;
        out.flush()?;
    }

    enable_raw_mode()?;
    let _guard = TerminalGuard;
    let stdout = io::stdout();
    let backend = CrosstermBackend::new(stdout);
    // Inline viewport: bottom rows managed by ratatui
    // (streaming partial line + separator + multi-line input + separator + status).
    // Everything above is native terminal scrollback.
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(INLINE_VIEWPORT_HEIGHT),
        },
    )?;

    let workspace_path = PathBuf::from(status.working_directory.clone());
    let ui_cfg = codingbuddy_core::AppConfig::load(&workspace_path)
        .unwrap_or_default()
        .ui;
    let thinking_visibility = if ui_cfg.thinking_visibility.trim().is_empty() {
        "concise".to_string()
    } else {
        ui_cfg.thinking_visibility.to_ascii_lowercase()
    };
    let phase_heartbeat_ms = ui_cfg.phase_heartbeat_ms.max(1000);
    let mission_control_max_events = ui_cfg.mission_control_max_events.max(50) as usize;

    let mut shell = ChatShell {
        agent_mode: "ToolUseLoop".to_string(),
        reduced_motion,
        mission_control_max_events,
        thinking_visibility: thinking_visibility.clone(),
        ..Default::default()
    };
    let mut input = String::new();
    let mut cursor_pos: usize = 0;
    let mut history_cursor: Option<usize> = None;
    let mut saved_input = String::new();
    let mut info_line = String::from(" Ctrl+C exit | Tab autocomplete | Native scroll & select");
    let mut history: VecDeque<String> = VecDeque::new();
    let mut last_escape_at: Option<Instant> = None;
    let mut cursor_visible;
    let mut tick_count: usize = 0;
    let mut vim_enabled = false;
    let mut vim_mode = VimMode::Insert;
    let mut vim_command_buffer = String::new();
    let mut vim_visual_anchor: Option<usize> = None;
    let mut vim_yank_buffer = String::new();
    let mut vim_pending_operator: Option<char> = None;
    let mut vim_pending_text_object: Option<(char, char)> = None;
    let mut vim_pending_g = false;
    let mut is_processing = false;
    let mut pending_approval: Option<(String, String, mpsc::Sender<bool>)> = None;
    let mut cancelled = false;
    // Track the last transcript length so we only print new entries via insert_before.
    let mut last_printed_idx: usize = 0;
    // Buffer for streaming content — complete lines get flushed to scrollback,
    // the partial (incomplete) last line is rendered in the viewport.
    let mut streaming_buffer = String::new();
    let mut streaming_in_code_block = false;
    let mut streaming_in_diff_block = false;
    let mut streaming_code_block_lang = String::new();
    let mut mission_control_visible = false;
    let mut artifacts_visible = false;
    let mut plan_collapsed = false;
    let mut reverse_search_active = false;
    let mut reverse_search_query = String::new();
    let mut reverse_search_index: Option<usize> = None;
    let mut active_phase: Option<(u64, String)> = None;
    let mut last_phase_event_at = Instant::now();
    let mut last_phase_heartbeat_at = Instant::now();
    let mut last_mission_refresh_at = Instant::now();
    let mut model_picker: Option<ModelPickerState> = None;
    let mut pending_images: Vec<PathBuf> = Vec::new();
    let mut autocomplete_dropdown: Option<AutocompleteState> = None;
    let mut ml_ghost: GhostTextState = GhostTextState::default();

    loop {
        // ML ghost text: check debounce and invoke completion callback
        if ml_ghost.should_request(input.len()) {
            if let Some(ref cb) = ml_completion {
                let suggestion = cb(&input);
                ml_ghost.set_suggestion(suggestion);
            } else {
                ml_ghost.set_suggestion(None);
            }
        }

        tick_count = tick_count.wrapping_add(1);
        shell.spinner_tick = tick_count;
        cursor_visible = tick_count % 16 < 8;
        let approval_alert_active = pending_approval.is_some();
        let approval_flash_on = tick_count % 12 < 6;
        if let Some((iteration, phase)) = active_phase.as_ref()
            && last_phase_event_at.elapsed() >= Duration::from_millis(phase_heartbeat_ms)
            && last_phase_heartbeat_at.elapsed() >= Duration::from_millis(phase_heartbeat_ms)
        {
            info_line = format!("iter {iteration} {phase} in progress...");
            shell.push_mission_control(format!(
                "iteration {iteration}: {phase} in progress (heartbeat)"
            ));
            last_phase_heartbeat_at = Instant::now();
        }
        if mission_control_visible
            && last_mission_refresh_at.elapsed() >= Duration::from_millis(1500)
        {
            if let Some(new_status) = refresh_status() {
                status = new_status;
            }
            last_mission_refresh_at = Instant::now();
        }

        // Print any new transcript entries above the inline viewport
        // so they go into native terminal scrollback.
        flush_transcript_above(&mut terminal, &shell, &mut last_printed_idx, plan_collapsed)?;
        // Flush complete lines from the streaming buffer into scrollback.
        flush_streaming_lines(
            &mut terminal,
            &mut streaming_buffer,
            &mut streaming_in_code_block,
            &mut streaming_in_diff_block,
            &mut streaming_code_block_lang,
            plan_collapsed,
        )?;

        terminal.draw(|frame| {
            let area = frame.area();
            if area.width == 0 || area.height < 3 {
                return;
            }
            // Inline viewport rows:
            //   Row 0..A: stream partial output
            //   Row A+1: separator ─────
            //   Row A+2..B: multi-line input prompt
            //   Row B+1: separator ─────
            //   Row B+2: status bar
            let width = area.width;
            let (prompt_str, prompt_style) = if vim_enabled {
                match vim_mode {
                    VimMode::Normal => (
                        ": ",
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    VimMode::Visual => (
                        ": ",
                        Style::default()
                            .fg(Color::Magenta)
                            .add_modifier(Modifier::BOLD),
                    ),
                    VimMode::Command => (
                        ": ",
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    VimMode::Insert => (
                        "> ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                }
            } else {
                (
                    "> ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )
            };
            let cursor_ch = if cursor_visible { "\u{2588}" } else { " " };
            let desired_input_rows = if pending_approval.is_some() || reverse_search_active {
                INPUT_MIN_HEIGHT as usize
            } else if vim_enabled && vim_mode == VimMode::Command {
                wrapped_text_rows(
                    &format!("{prompt_str}:{}{}", vim_command_buffer, cursor_ch),
                    width,
                )
            } else {
                let before = &input[..cursor_pos.min(input.len())];
                wrapped_text_rows(&format!("{prompt_str}{before}{cursor_ch}"), width)
            };
            let (stream_height, input_height) = compute_inline_heights(area.height, desired_input_rows);
            let stream_area = Rect::new(area.x, area.y, width, stream_height);
            let sep_area = Rect::new(area.x, area.y + stream_height, width, 1);
            let input_area = Rect::new(area.x, sep_area.y + 1, width, input_height);
            let sep2_y = input_area.y + input_area.height;
            let status_y = sep2_y + 1;

            // Row 0..A: current streaming partial line(s) (or blank when idle)
            if let Some((tool_name, args_summary, _)) = pending_approval.as_ref() {
                let compact_args = truncate_inline(&args_summary.replace('\n', " "), 72);
                let banner = format!(
                    " !!! APPROVAL REQUIRED !!! {tool_name} {compact_args} | PRESS Y TO APPROVE | ANY KEY DENIES "
                );
                let bg = if approval_flash_on {
                    Color::LightYellow
                } else {
                    Color::LightRed
                };
                frame.render_widget(
                    Paragraph::new(Line::from(vec![Span::styled(
                        banner,
                        Style::default()
                            .fg(Color::Black)
                            .bg(bg)
                            .add_modifier(Modifier::BOLD),
                    )]))
                    .wrap(Wrap { trim: false }),
                    stream_area,
                );
            } else if !streaming_buffer.is_empty() {
                let styled = if let Some(meta) = parse_stream_meta_entry(&streaming_buffer) {
                    style_transcript_line(&meta, false, false, "")
                } else {
                    let partial_entry = TranscriptEntry {
                        kind: MessageKind::Assistant,
                        text: streaming_buffer.clone(),
                    };
                    if streaming_in_code_block && streaming_in_diff_block {
                        render_diff_line(&partial_entry.text)
                    } else {
                        render_assistant_markdown(&partial_entry.text)
                    }
                };
                let scroll_y = scroll_to_keep_row_visible(
                    wrapped_text_rows(&streaming_buffer, stream_area.width).saturating_sub(1),
                    stream_area.height,
                );
                frame.render_widget(
                    Paragraph::new(styled)
                        .wrap(Wrap { trim: false })
                        .scroll((scroll_y, 0)),
                    stream_area,
                );
            } else if shell.is_thinking && !shell.thinking_buffer.is_empty() {
                // Show the tail of the thinking buffer in the streaming area,
                // styled as dimmed italic to distinguish from normal output.
                let display = shell.thinking_buffer.clone();
                let scroll_y = scroll_to_keep_row_visible(
                    wrapped_text_rows(&display, stream_area.width).saturating_sub(1),
                    stream_area.height,
                );
                frame.render_widget(
                    Paragraph::new(Line::from(vec![
                        Span::styled(
                            "  \u{25cb} ",
                            Style::default()
                                .fg(Color::Magenta)
                                .add_modifier(Modifier::ITALIC),
                        ),
                        Span::styled(
                            display,
                            Style::default()
                                .fg(Color::DarkGray)
                                .add_modifier(Modifier::ITALIC),
                        ),
                    ]))
                    .wrap(Wrap { trim: false })
                    .scroll((scroll_y, 0)),
                    stream_area,
                );
            } else if mission_control_visible {
                let panel = render_mission_control_panel(&status, &shell.mission_control_lines);
                frame.render_widget(
                    Paragraph::new(panel).wrap(Wrap { trim: false }),
                    stream_area,
                );
            } else {
                frame.render_widget(Paragraph::new(Span::raw("")), stream_area);
            }

            // Row A+1: thin separator line
            let sep_style = if approval_alert_active {
                Style::default()
                    .fg(if approval_flash_on {
                        Color::LightYellow
                    } else {
                        Color::LightRed
                    })
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            frame.render_widget(
                Paragraph::new(Span::styled(
                    if approval_alert_active {
                        "\u{2550}".repeat(width as usize)
                    } else {
                        "\u{2500}".repeat(width as usize)
                    },
                    sep_style,
                )),
                sep_area,
            );

            // Row A+2..B: input prompt
            if let Some((tool_name, args_summary, _)) = pending_approval.as_ref() {
                let compact_args = truncate_inline(&args_summary.replace('\n', " "), 56);
                let prompt = format!(
                    " ACTION NEEDED: `{tool_name}` {compact_args} | Press Y to approve, any other key to deny "
                );
                let bg = if approval_flash_on {
                    Color::LightYellow
                } else {
                    Color::LightRed
                };
                frame.render_widget(
                    Paragraph::new(Line::from(vec![Span::styled(
                        truncate_inline(&prompt, width.saturating_sub(1) as usize),
                        Style::default()
                            .fg(Color::Black)
                            .bg(bg)
                            .add_modifier(Modifier::BOLD),
                    )]))
                    .wrap(Wrap { trim: false }),
                    input_area,
                );
            } else if reverse_search_active {
                let match_preview = reverse_search_index
                    .and_then(|idx| history.get(idx))
                    .cloned()
                    .unwrap_or_else(|| "<no match>".to_string());
                let text = format!(
                    "(reverse-i-search)`{}': {}",
                    reverse_search_query,
                    truncate_inline(&match_preview, width.saturating_sub(28) as usize)
                );
                frame.render_widget(
                    Paragraph::new(Line::from(vec![Span::styled(
                        truncate_inline(&text, width.saturating_sub(1) as usize),
                        Style::default()
                            .fg(Color::LightYellow)
                            .add_modifier(Modifier::BOLD),
                    )]))
                    .wrap(Wrap { trim: false }),
                    input_area,
                );
            } else if vim_enabled && vim_mode == VimMode::Command {
                let body = format!(":{}{}", vim_command_buffer, cursor_ch);
                let cursor_anchor = format!("{prompt_str}:{}{}", vim_command_buffer, cursor_ch);
                let scroll_y = scroll_to_keep_row_visible(
                    wrapped_text_rows(&cursor_anchor, input_area.width).saturating_sub(1),
                    input_area.height,
                );
                frame.render_widget(
                    Paragraph::new(Line::from(vec![
                        Span::styled(prompt_str.to_string(), prompt_style),
                        Span::raw(body),
                    ]))
                    .wrap(Wrap { trim: false })
                    .scroll((scroll_y, 0)),
                    input_area,
                );
            } else {
                let before = &input[..cursor_pos.min(input.len())];
                let after = &input[cursor_pos.min(input.len())..];
                let cursor_anchor = format!("{prompt_str}{before}{cursor_ch}");
                let scroll_y = scroll_to_keep_row_visible(
                    wrapped_text_rows(&cursor_anchor, input_area.width).saturating_sub(1),
                    input_area.height,
                );
                let mut spans = vec![
                    Span::styled(prompt_str.to_string(), prompt_style),
                    Span::raw(before.to_string()),
                    Span::raw(cursor_ch.to_string()),
                    Span::raw(after.to_string()),
                ];
                // Ghost text priority: ML suggestion > history ghost > none
                if scroll_y == 0 && cursor_pos >= input.len() {
                    if let Some(ref ml_text) = ml_ghost.suggestion {
                        spans.push(Span::styled(
                            ml_text.clone(),
                            Style::default()
                                .fg(Color::DarkGray)
                                .add_modifier(Modifier::ITALIC),
                        ));
                    } else if let Some(ghost) = history_ghost_suffix(&history, &input) {
                        spans.push(Span::styled(
                            ghost,
                            Style::default()
                                .fg(Color::DarkGray)
                                .add_modifier(Modifier::ITALIC),
                        ));
                    }
                }
                frame.render_widget(
                    Paragraph::new(Line::from(spans))
                        .wrap(Wrap { trim: false })
                        .scroll((scroll_y, 0)),
                    input_area,
                );
            }
            // Row N+1: always-visible operator summary (phase/todo/subagents/capabilities/counters)
            let summary_area = Rect::new(area.x, sep2_y, width, 1);
            let operator_summary = operator_summary_line(&status);
            let operator_style = if status.failed_tasks > 0 || status.failed_subagents > 0 {
                Style::default()
                    .fg(Color::Red)
                    .add_modifier(Modifier::BOLD)
            } else if approval_alert_active {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            frame.render_widget(
                Paragraph::new(Line::from(vec![Span::styled(
                    format!(" {}", truncate_inline(&operator_summary, width.saturating_sub(1) as usize)),
                    operator_style,
                )])),
                summary_area,
            );

            // Row N+2: compact status bar (model/mode/usage + dynamic notices)
            let status_area = Rect::new(area.x, status_y, width, 1);
            let mut status_spans = render_statusline_spans(
                &status,
                if shell.is_thinking {
                    None
                } else {
                    shell.active_tool.as_deref()
                },
                shell.spinner_frame(),
                None,
                if vim_enabled {
                    Some(vim_mode.label())
                } else {
                    None
                },
                false,
                shell.is_thinking,
            );
            if mission_control_visible {
                status_spans.push(Span::raw(" "));
                status_spans.push(Span::styled(
                    " MISSION ".to_string(),
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::LightBlue)
                        .add_modifier(Modifier::BOLD),
                ));
            }
            if artifacts_visible {
                status_spans.push(Span::raw(" "));
                status_spans.push(Span::styled(
                    " ARTIFACTS ".to_string(),
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::LightGreen)
                        .add_modifier(Modifier::BOLD),
                ));
            }
            if plan_collapsed {
                status_spans.push(Span::raw(" "));
                status_spans.push(Span::styled(
                    " PLAN COLLAPSED ".to_string(),
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::LightMagenta)
                        .add_modifier(Modifier::BOLD),
                ));
            }
            if approval_alert_active {
                let approval_bg = if approval_flash_on {
                    Color::Yellow
                } else {
                    Color::Red
                };
                status_spans.push(Span::raw(" "));
                status_spans.push(Span::styled(
                    " APPROVAL REQUIRED ".to_string(),
                    Style::default()
                        .fg(Color::Black)
                        .bg(approval_bg)
                        .add_modifier(Modifier::BOLD),
                ));
            }
            status_spans.push(Span::styled(
                format!(" {}", info_line),
                if approval_alert_active {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::DarkGray)
                },
            ));
            frame.render_widget(Paragraph::new(Line::from(status_spans)), status_area);
        })?;

        // Drain streaming events from background agent thread.
        while let Ok(ev) = stream_rx.try_recv() {
            let Some(ev) = filter_stream_event(ev, &mut cancelled) else {
                continue;
            };
            let result = {
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
                handle_stream_event(ev, &mut state)
            };
            match result {
                StreamEventResult::Continue => {}
                StreamEventResult::Done(output) => {
                    // Flush thinking buffer to transcript as a collapsed summary.
                    if !shell.thinking_buffer.is_empty() {
                        let thought = std::mem::take(&mut shell.thinking_buffer);
                        // Show a single-line summary of the thinking in the transcript.
                        let summary = truncate_inline(&thought.replace('\n', " "), 120);
                        shell.push_thinking(summary);
                    }
                    shell.is_thinking = false;
                    // Flush any remaining partial streaming line to scrollback.
                    if !streaming_buffer.is_empty() {
                        let remaining = std::mem::take(&mut streaming_buffer);
                        let logical_lines = if streaming_in_code_block {
                            vec![remaining]
                        } else {
                            split_inline_markdown_blocks(&remaining)
                        };
                        let mut styled_lines: Vec<Line<'static>> =
                            Vec::with_capacity(logical_lines.len().max(1));
                        for text in logical_lines {
                            let entry = TranscriptEntry {
                                kind: MessageKind::Assistant,
                                text,
                            };
                            styled_lines.push(style_transcript_line(
                                &entry,
                                streaming_in_code_block,
                                streaming_in_diff_block,
                                &streaming_code_block_lang,
                            ));
                        }
                        let _ = insert_wrapped_lines_above(&mut terminal, &styled_lines);
                    }
                    shell.finalize_streaming(&output);
                    // Skip re-printing entries that were already streamed.
                    last_printed_idx = shell.transcript.len();
                    streaming_in_code_block = false;
                    streaming_in_diff_block = false;
                    streaming_code_block_lang.clear();
                    active_phase = None;
                    info_line = "ok".to_string();
                    is_processing = false;
                    shell.active_tool = None;
                    if let Some(new_status) = refresh_status() {
                        status = new_status;
                    }
                }
            }
        }

        // Handle pending approval prompts via keyboard
        if let Some((ref tool_name, ref args_summary, _)) = pending_approval {
            let compact_args = truncate_inline(&args_summary.replace('\n', " "), 96);
            info_line = format!(
                "ACTION REQUIRED: `{tool_name}` {compact_args} [press Y to approve / any key denies]"
            );
        }

        // Check for external SIGINT (e.g. `kill -INT`).
        if sigint_flag.swap(false, Ordering::Relaxed) {
            if is_processing {
                is_processing = false;
                cancelled = true;
                streaming_buffer.clear();
                shell.active_tool = None;
                active_phase = None;
                shell.is_thinking = false;
                shell.thinking_buffer.clear();
                if let Some((_, _, response_tx)) = pending_approval.take() {
                    let _ = response_tx.send(false);
                }
                info_line = "interrupted — press Ctrl+C again to exit".to_string();
            } else {
                break;
            }
        }

        if !event::poll(Duration::from_millis(33))? {
            continue;
        }
        let mut key = match event::read()? {
            Event::Resize(_, _) => {
                // Let the next draw pass recompute viewport partitions immediately.
                continue;
            }
            Event::Paste(pasted) => {
                input.insert_str(cursor_pos.min(input.len()), &pasted);
                cursor_pos = (cursor_pos + pasted.len()).min(input.len());
                info_line = "pasted input".to_string();
                continue;
            }
            Event::Key(key) => key,
            _ => continue,
        };
        // Only handle key press events (ignore release/repeat on platforms that send them)
        if key.kind != KeyEventKind::Press {
            continue;
        }

        // ── Model picker overlay ──────────────────────────────────────────
        if let Some(ref mut mp) = model_picker {
            match key.code {
                KeyCode::Up => {
                    mp.up();
                    mp.selected = mp.selected.min(MODEL_CHOICES.len().saturating_sub(1));
                    let (name, desc) = MODEL_CHOICES[mp.selected];
                    info_line = format!("Select model: > {name}  ({desc})");
                    continue;
                }
                KeyCode::Down => {
                    mp.down();
                    mp.selected = mp.selected.min(MODEL_CHOICES.len().saturating_sub(1));
                    let (name, desc) = MODEL_CHOICES[mp.selected];
                    info_line = format!("Select model: > {name}  ({desc})");
                    continue;
                }
                KeyCode::Enter => {
                    let chosen = mp.confirm();
                    model_picker = None;
                    // Send the model selection as a /model command
                    let model_cmd = format!("/model {chosen}");
                    on_submit(&model_cmd);
                    info_line = format!("Model: {chosen}");
                    continue;
                }
                KeyCode::Esc => {
                    model_picker = None;
                    info_line = "model selection cancelled".to_string();
                    continue;
                }
                KeyCode::Char('1') => {
                    model_picker = None;
                    let chosen = MODEL_CHOICES[0].0;
                    let model_cmd = format!("/model {chosen}");
                    on_submit(&model_cmd);
                    info_line = format!("Model: {chosen}");
                    continue;
                }
                KeyCode::Char('2') => {
                    model_picker = None;
                    let chosen = MODEL_CHOICES[1].0;
                    let model_cmd = format!("/model {chosen}");
                    on_submit(&model_cmd);
                    info_line = format!("Model: {chosen}");
                    continue;
                }
                _ => continue, // ignore other keys while picker is active
            }
        }

        // ── Autocomplete dropdown overlay ────────────────────────────────────
        if autocomplete_dropdown.is_some() {
            match key.code {
                KeyCode::Up | KeyCode::BackTab => {
                    autocomplete_dropdown.as_mut().unwrap().up();
                    let lines = autocomplete_dropdown.as_ref().unwrap().display_lines(6);
                    info_line = lines.join("  ");
                    continue;
                }
                KeyCode::Down | KeyCode::Tab => {
                    autocomplete_dropdown.as_mut().unwrap().down();
                    let lines = autocomplete_dropdown.as_ref().unwrap().display_lines(6);
                    info_line = lines.join("  ");
                    continue;
                }
                KeyCode::Enter | KeyCode::Right => {
                    // Accept selected suggestion
                    if let Some(ref ac) = autocomplete_dropdown
                        && let Some(value) = ac.selected_value()
                    {
                        let slash_mode = ac.trigger_pos == 0 && input.starts_with('/');
                        if slash_mode {
                            // Replace entire input with the slash command
                            let cmd = slash_suggestion_to_command(value);
                            input = format!("{cmd} ");
                            cursor_pos = input.len();
                        } else {
                            let trigger = ac.trigger_pos;
                            // Replace @prefix with @fullpath
                            input.truncate(trigger);
                            input.push('@');
                            input.push_str(value);
                            input.push(' ');
                            cursor_pos = input.len();
                        }
                    }
                    autocomplete_dropdown = None;
                    info_line = String::new();
                    continue;
                }
                KeyCode::Esc => {
                    autocomplete_dropdown = None;
                    info_line = "autocomplete dismissed".to_string();
                    continue;
                }
                KeyCode::Char(ch) => {
                    // Continue typing — update suggestions
                    input.insert(cursor_pos.min(input.len()), ch);
                    cursor_pos += 1;
                    if let Some(ref ac) = autocomplete_dropdown {
                        let slash_mode = ac.trigger_pos == 0 && input.starts_with('/');
                        let suggestions = if slash_mode {
                            let prefix = &input[1..cursor_pos];
                            slash_command_suggestions(prefix, 8)
                        } else {
                            let prefix = &input[ac.trigger_pos + 1..cursor_pos];
                            autocomplete_at_suggestions(prefix, &workspace_path)
                        };
                        if suggestions.is_empty() {
                            autocomplete_dropdown = None;
                            info_line = String::new();
                        } else {
                            autocomplete_dropdown =
                                Some(AutocompleteState::new(suggestions, ac.trigger_pos));
                            let lines = autocomplete_dropdown.as_ref().unwrap().display_lines(6);
                            info_line = lines.join("  ");
                        }
                    }
                    continue;
                }
                KeyCode::Backspace => {
                    if cursor_pos > 0 && cursor_pos <= input.len() {
                        input.remove(cursor_pos - 1);
                        cursor_pos -= 1;
                    }
                    if let Some(ref ac) = autocomplete_dropdown {
                        let slash_mode = ac.trigger_pos == 0 && input.starts_with('/');
                        if cursor_pos <= ac.trigger_pos {
                            // Deleted the trigger char itself (@ or /)
                            autocomplete_dropdown = None;
                            info_line = String::new();
                        } else {
                            let suggestions = if slash_mode {
                                let prefix = &input[1..cursor_pos];
                                slash_command_suggestions(prefix, 8)
                            } else {
                                let prefix = &input[ac.trigger_pos + 1..cursor_pos];
                                autocomplete_at_suggestions(prefix, &workspace_path)
                            };
                            if suggestions.is_empty() {
                                autocomplete_dropdown = None;
                                info_line = String::new();
                            } else {
                                autocomplete_dropdown =
                                    Some(AutocompleteState::new(suggestions, ac.trigger_pos));
                                let lines =
                                    autocomplete_dropdown.as_ref().unwrap().display_lines(6);
                                info_line = lines.join("  ");
                            }
                        }
                    }
                    continue;
                }
                _ => {
                    autocomplete_dropdown = None;
                    info_line = String::new();
                    // Fall through to normal key handling
                }
            }
        }

        if key == bindings.exit {
            if is_processing {
                is_processing = false;
                cancelled = true;
                streaming_buffer.clear();
                shell.active_tool = None;
                shell.is_thinking = false;
                shell.thinking_buffer.clear();
                if let Some((_, _, response_tx)) = pending_approval.take() {
                    let _ = response_tx.send(false);
                }
                info_line = "interrupted — press Ctrl+C again to exit".to_string();
                continue;
            }
            break;
        }

        // Handle pending approval y/N
        if pending_approval.is_some() {
            let approved = matches!(key.code, KeyCode::Char('y') | KeyCode::Char('Y'));
            if let Some((tool_name, _, response_tx)) = pending_approval.take() {
                let _ = response_tx.send(approved);
                info_line = if approved {
                    format!("approved `{tool_name}`")
                } else {
                    format!("denied `{tool_name}`")
                };
            }
            continue;
        }

        if reverse_search_active {
            match key.code {
                KeyCode::Esc => {
                    reverse_search_active = false;
                    reverse_search_query.clear();
                    reverse_search_index = None;
                    input = saved_input.clone();
                    cursor_pos = input.len();
                    info_line = "reverse search canceled".to_string();
                    continue;
                }
                KeyCode::Enter => {
                    reverse_search_active = false;
                    reverse_search_query.clear();
                    reverse_search_index = None;
                    info_line = "reverse search accepted".to_string();
                }
                KeyCode::Backspace => {
                    reverse_search_query.pop();
                    reverse_search_index =
                        history_reverse_search_index(&history, &reverse_search_query, None);
                    apply_reverse_search_result(
                        &history,
                        reverse_search_index,
                        &mut input,
                        &mut cursor_pos,
                    );
                    info_line = if reverse_search_index.is_some() {
                        format!("reverse search: `{}`", reverse_search_query)
                    } else {
                        format!("reverse search: no match for `{}`", reverse_search_query)
                    };
                    continue;
                }
                KeyCode::Char(ch) if key.modifiers == KeyModifiers::NONE => {
                    reverse_search_query.push(ch);
                    reverse_search_index =
                        history_reverse_search_index(&history, &reverse_search_query, None);
                    apply_reverse_search_result(
                        &history,
                        reverse_search_index,
                        &mut input,
                        &mut cursor_pos,
                    );
                    info_line = if reverse_search_index.is_some() {
                        format!("reverse search: `{}`", reverse_search_query)
                    } else {
                        format!("reverse search: no match for `{}`", reverse_search_query)
                    };
                    continue;
                }
                _ => {
                    reverse_search_active = false;
                    reverse_search_query.clear();
                    reverse_search_index = None;
                }
            }
        }

        if key == bindings.history_search {
            if history.is_empty() {
                info_line = "reverse search: history is empty".to_string();
                continue;
            }
            if !reverse_search_active {
                saved_input = input.clone();
                reverse_search_active = true;
                reverse_search_query.clear();
                reverse_search_index = history_reverse_search_index(&history, "", None);
            } else {
                let before = reverse_search_index;
                reverse_search_index =
                    history_reverse_search_index(&history, &reverse_search_query, before);
            }
            apply_reverse_search_result(
                &history,
                reverse_search_index,
                &mut input,
                &mut cursor_pos,
            );
            info_line = if reverse_search_index.is_some() {
                if reverse_search_query.is_empty() {
                    "reverse search: latest entry".to_string()
                } else {
                    format!("reverse search: `{}`", reverse_search_query)
                }
            } else {
                format!("reverse search: no match for `{}`", reverse_search_query)
            };
            continue;
        }

        let mut vim_quit_after_submit = false;
        if vim_enabled {
            let mut vim_consumed = false;
            match vim_mode {
                VimMode::Insert => {
                    if key.code == KeyCode::Esc {
                        vim_mode = VimMode::Normal;
                        vim_pending_operator = None;
                        vim_pending_text_object = None;
                        vim_pending_g = false;
                        info_line = "-- NORMAL --".to_string();
                        vim_consumed = true;
                    }
                }
                VimMode::Normal => {
                    if let Some((operator, scope)) = vim_pending_text_object.take() {
                        if let Some((start, end)) =
                            resolve_vim_text_object_bounds(&input, cursor_pos, scope, key.code)
                        {
                            if apply_vim_operator_range(
                                operator,
                                &mut input,
                                &mut cursor_pos,
                                &mut vim_yank_buffer,
                                &mut vim_mode,
                                start,
                                end,
                            ) {
                                info_line = match operator {
                                    'c' => "-- INSERT --".to_string(),
                                    'y' => format!("yanked {} chars", end.saturating_sub(start)),
                                    _ => format!(
                                        "applied {operator}{}{}",
                                        scope,
                                        display_key_code(key.code)
                                    ),
                                };
                            }
                        } else {
                            info_line = format!(
                                "unknown text object {}{}",
                                scope,
                                display_key_code(key.code)
                            );
                        }
                        vim_pending_operator = None;
                        vim_pending_g = false;
                    } else {
                        match key.code {
                            KeyCode::Char('i') => {
                                if let Some(op) = vim_pending_operator.take() {
                                    vim_pending_text_object = Some((op, 'i'));
                                    info_line = format!("{op}i");
                                } else {
                                    vim_mode = VimMode::Insert;
                                    vim_pending_operator = None;
                                    vim_pending_g = false;
                                    info_line = "-- INSERT --".to_string();
                                }
                            }
                            KeyCode::Char('a') => {
                                if let Some(op) = vim_pending_operator.take() {
                                    vim_pending_text_object = Some((op, 'a'));
                                    info_line = format!("{op}a");
                                } else {
                                    cursor_pos = (cursor_pos + 1).min(input.len());
                                    vim_mode = VimMode::Insert;
                                    vim_pending_operator = None;
                                    vim_pending_g = false;
                                    info_line = "-- INSERT --".to_string();
                                }
                            }
                            KeyCode::Char('v') => {
                                vim_mode = VimMode::Visual;
                                vim_visual_anchor = Some(cursor_pos);
                                vim_pending_operator = None;
                                vim_pending_text_object = None;
                                vim_pending_g = false;
                                info_line = "-- VISUAL --".to_string();
                            }
                            KeyCode::Char(':') => {
                                vim_mode = VimMode::Command;
                                vim_command_buffer.clear();
                                vim_pending_operator = None;
                                vim_pending_text_object = None;
                                vim_pending_g = false;
                                info_line = ":".to_string();
                            }
                            KeyCode::Char('h') | KeyCode::Left => {
                                cursor_pos = cursor_pos.saturating_sub(1);
                                vim_pending_operator = None;
                                vim_pending_text_object = None;
                                vim_pending_g = false;
                            }
                            KeyCode::Char('l') | KeyCode::Right => {
                                cursor_pos = (cursor_pos + 1).min(input.len());
                                vim_pending_operator = None;
                                vim_pending_text_object = None;
                                vim_pending_g = false;
                            }
                            KeyCode::Char('0') => {
                                cursor_pos = 0;
                                vim_pending_operator = None;
                                vim_pending_text_object = None;
                                vim_pending_g = false;
                            }
                            KeyCode::Char('$') => {
                                cursor_pos = input.len();
                                vim_pending_operator = None;
                                vim_pending_text_object = None;
                                vim_pending_g = false;
                            }
                            KeyCode::Char('g') => {
                                if vim_pending_g {
                                    cursor_pos = 0;
                                    vim_pending_g = false;
                                    vim_pending_operator = None;
                                    vim_pending_text_object = None;
                                    info_line = "gg".to_string();
                                } else {
                                    vim_pending_g = true;
                                    info_line = "g".to_string();
                                }
                            }
                            KeyCode::Char('G') => {
                                cursor_pos = input.len();
                                vim_pending_operator = None;
                                vim_pending_text_object = None;
                                vim_pending_g = false;
                                info_line = "G".to_string();
                            }
                            KeyCode::Char('w') => {
                                if let Some(op) = vim_pending_operator.take() {
                                    if let Some((start, end)) =
                                        word_text_object_bounds(&input, cursor_pos, false)
                                    {
                                        let _ = apply_vim_operator_range(
                                            op,
                                            &mut input,
                                            &mut cursor_pos,
                                            &mut vim_yank_buffer,
                                            &mut vim_mode,
                                            start,
                                            end,
                                        );
                                        info_line = if op == 'c' {
                                            "-- INSERT --".to_string()
                                        } else {
                                            format!("applied {op}w")
                                        };
                                    }
                                } else {
                                    cursor_pos = move_to_next_word_start(&input, cursor_pos);
                                }
                                vim_pending_text_object = None;
                                vim_pending_g = false;
                            }
                            KeyCode::Char('b') => {
                                cursor_pos = move_to_prev_word_start(&input, cursor_pos);
                                vim_pending_operator = None;
                                vim_pending_text_object = None;
                                vim_pending_g = false;
                            }
                            KeyCode::Char('e') => {
                                cursor_pos = move_to_word_end(&input, cursor_pos);
                                vim_pending_operator = None;
                                vim_pending_text_object = None;
                                vim_pending_g = false;
                            }
                            KeyCode::Char('x') => {
                                if cursor_pos < input.len() {
                                    input.remove(cursor_pos);
                                }
                                vim_pending_operator = None;
                                vim_pending_text_object = None;
                                vim_pending_g = false;
                            }
                            KeyCode::Char('p') => {
                                if !vim_yank_buffer.is_empty() {
                                    let insert_at = (cursor_pos + 1).min(input.len());
                                    input.insert_str(insert_at, &vim_yank_buffer);
                                    cursor_pos =
                                        (insert_at + vim_yank_buffer.len()).min(input.len());
                                }
                                vim_pending_operator = None;
                                vim_pending_text_object = None;
                                vim_pending_g = false;
                            }
                            KeyCode::Char('d') => {
                                if vim_pending_operator == Some('d') {
                                    input.clear();
                                    cursor_pos = 0;
                                    vim_pending_operator = None;
                                    vim_pending_text_object = None;
                                    vim_pending_g = false;
                                    info_line = "deleted line".to_string();
                                } else {
                                    vim_pending_operator = Some('d');
                                    vim_pending_text_object = None;
                                    vim_pending_g = false;
                                    info_line = "d".to_string();
                                }
                            }
                            KeyCode::Char('c') => {
                                if vim_pending_operator == Some('c') {
                                    input.clear();
                                    cursor_pos = 0;
                                    vim_mode = VimMode::Insert;
                                    vim_pending_operator = None;
                                    vim_pending_text_object = None;
                                    vim_pending_g = false;
                                    info_line = "-- INSERT --".to_string();
                                } else {
                                    vim_pending_operator = Some('c');
                                    vim_pending_text_object = None;
                                    vim_pending_g = false;
                                    info_line = "c".to_string();
                                }
                            }
                            KeyCode::Char('y') => {
                                if vim_pending_operator == Some('y') {
                                    vim_yank_buffer = input.clone();
                                    vim_pending_operator = None;
                                    vim_pending_text_object = None;
                                    vim_pending_g = false;
                                    info_line = "yanked line".to_string();
                                } else {
                                    vim_pending_operator = Some('y');
                                    vim_pending_text_object = None;
                                    vim_pending_g = false;
                                    info_line = "y".to_string();
                                }
                            }
                            KeyCode::Enter => {
                                key = bindings.submit;
                                vim_pending_operator = None;
                                vim_pending_text_object = None;
                                vim_pending_g = false;
                            }
                            KeyCode::Esc => {
                                vim_pending_operator = None;
                                vim_pending_text_object = None;
                                vim_pending_g = false;
                            }
                            _ => {
                                vim_pending_operator = None;
                                vim_pending_text_object = None;
                                vim_pending_g = false;
                            }
                        }
                    }
                    vim_consumed = true;
                }
                VimMode::Visual => {
                    vim_pending_operator = None;
                    vim_pending_text_object = None;
                    vim_pending_g = false;
                    if vim_visual_anchor.is_none() {
                        vim_visual_anchor = Some(cursor_pos);
                    }
                    match key.code {
                        KeyCode::Esc => {
                            vim_mode = VimMode::Normal;
                            vim_visual_anchor = None;
                            info_line = "-- NORMAL --".to_string();
                        }
                        KeyCode::Char('h') | KeyCode::Left => {
                            cursor_pos = cursor_pos.saturating_sub(1);
                        }
                        KeyCode::Char('l') | KeyCode::Right => {
                            cursor_pos = (cursor_pos + 1).min(input.len());
                        }
                        KeyCode::Char('0') => {
                            cursor_pos = 0;
                        }
                        KeyCode::Char('$') => {
                            cursor_pos = input.len();
                        }
                        KeyCode::Char('w') => {
                            cursor_pos = move_to_next_word_start(&input, cursor_pos);
                        }
                        KeyCode::Char('b') => {
                            cursor_pos = move_to_prev_word_start(&input, cursor_pos);
                        }
                        KeyCode::Char('e') => {
                            cursor_pos = move_to_word_end(&input, cursor_pos);
                        }
                        KeyCode::Char('y') => {
                            if let Some(anchor) = vim_visual_anchor {
                                let selection =
                                    extract_visual_selection(&input, anchor, cursor_pos);
                                if !selection.is_empty() {
                                    vim_yank_buffer = selection;
                                }
                            }
                            vim_mode = VimMode::Normal;
                            vim_visual_anchor = None;
                            info_line = "-- NORMAL --".to_string();
                        }
                        KeyCode::Char('d') | KeyCode::Char('c') => {
                            if let Some(anchor) = vim_visual_anchor {
                                let (start, end) = visual_bounds(input.len(), anchor, cursor_pos);
                                if end > start {
                                    input.replace_range(start..end, "");
                                    cursor_pos = start.min(input.len());
                                }
                            }
                            vim_visual_anchor = None;
                            if key.code == KeyCode::Char('c') {
                                vim_mode = VimMode::Insert;
                                info_line = "-- INSERT --".to_string();
                            } else {
                                vim_mode = VimMode::Normal;
                                info_line = "-- NORMAL --".to_string();
                            }
                        }
                        KeyCode::Char(':') => {
                            vim_mode = VimMode::Command;
                            vim_command_buffer.clear();
                            vim_visual_anchor = None;
                            vim_pending_operator = None;
                            vim_pending_text_object = None;
                            vim_pending_g = false;
                            info_line = ":".to_string();
                        }
                        _ => {}
                    }
                    vim_consumed = true;
                }
                VimMode::Command => {
                    vim_pending_operator = None;
                    vim_pending_text_object = None;
                    vim_pending_g = false;
                    match key.code {
                        KeyCode::Esc => {
                            vim_mode = VimMode::Normal;
                            vim_command_buffer.clear();
                            info_line = "-- NORMAL --".to_string();
                        }
                        KeyCode::Backspace => {
                            vim_command_buffer.pop();
                        }
                        KeyCode::Enter => {
                            let cmd = vim_command_buffer.trim().to_ascii_lowercase();
                            vim_command_buffer.clear();
                            match cmd.as_str() {
                                "" => {
                                    vim_mode = VimMode::Normal;
                                    info_line = "-- NORMAL --".to_string();
                                }
                                "q" => {
                                    vim_quit_after_submit = true;
                                    info_line = "quit requested".to_string();
                                }
                                "w" => {
                                    vim_mode = VimMode::Normal;
                                    key = bindings.submit;
                                }
                                "wq" | "x" => {
                                    vim_mode = VimMode::Normal;
                                    key = bindings.submit;
                                    vim_quit_after_submit = true;
                                }
                                other => {
                                    vim_mode = VimMode::Normal;
                                    info_line = format!("unknown :{other}");
                                }
                            }
                        }
                        KeyCode::Char(ch) => {
                            vim_command_buffer.push(ch);
                        }
                        _ => {}
                    }
                    vim_consumed = true;
                }
            }

            if vim_consumed {
                if vim_quit_after_submit && key != bindings.submit {
                    break;
                }
                if key != bindings.submit {
                    continue;
                }
            }
        }
        if key == bindings.cycle_permission_mode {
            let current = status.permission_mode.clone();
            let next = match current.as_str() {
                "ask" => "plan",
                "plan" => "auto",
                "auto" => "locked",
                _ => "ask",
            };
            status.permission_mode = next.to_string();
            info_line = format!("permission mode: {} -> {}", current, next);
            continue;
        }
        if key == bindings.toggle_mission_control {
            mission_control_visible = !mission_control_visible;
            info_line = if mission_control_visible {
                if let Some(new_status) = refresh_status() {
                    status = new_status;
                }
                last_mission_refresh_at = Instant::now();
                format!("mission control visible ({})", status.workflow_phase)
            } else {
                "mission control hidden".to_string()
            };
            continue;
        }
        if key == bindings.toggle_artifacts {
            artifacts_visible = !artifacts_visible;
            info_line = if artifacts_visible {
                let lines = load_artifact_lines(&workspace_path);
                for line in lines.into_iter().take(16) {
                    shell.push_system(format!("[artifact] {line}"));
                }
                "artifacts panel enabled".to_string()
            } else {
                "artifacts panel hidden".to_string()
            };
            continue;
        }
        if key == bindings.toggle_plan_collapse {
            plan_collapsed = !plan_collapsed;
            info_line = if plan_collapsed {
                "plan collapse enabled".to_string()
            } else {
                "plan collapse disabled".to_string()
            };
            continue;
        }
        if key == bindings.approve_plan {
            if status.plan_state != "awaiting_approval" {
                info_line = "no plan is awaiting approval".to_string();
                continue;
            }
            if is_processing {
                info_line = "plan approval is already in progress".to_string();
                continue;
            }
            let plan_cmd = "/plan approve".to_string();
            shell.push_user(&plan_cmd);
            is_processing = true;
            cancelled = false;
            active_phase = None;
            last_phase_event_at = Instant::now();
            last_phase_heartbeat_at = Instant::now();
            shell.active_tool = Some("plan review".to_string());
            info_line = "approving current plan".to_string();
            on_submit(&plan_cmd);
            continue;
        }
        if key == bindings.reject_plan {
            if status.plan_state != "awaiting_approval" {
                info_line = "no plan is awaiting approval".to_string();
                continue;
            }
            input = "/plan reject ".to_string();
            cursor_pos = input.len();
            info_line = "plan reject: add feedback and press Enter".to_string();
            continue;
        }
        if key == bindings.background {
            let queued = input.trim().to_string();
            if queued.is_empty() {
                info_line = "background: type a prompt or !<shell command> first".to_string();
                continue;
            }
            if queued.starts_with('/') {
                info_line =
                    "background hotkey supports prompts or !<shell command>, not slash commands"
                        .to_string();
                continue;
            }
            let background_cmd = if queued.starts_with('!') {
                let command = queued.trim_start_matches('!').trim().to_string();
                if command.is_empty() {
                    info_line = "background shell command is empty".to_string();
                    continue;
                }
                format!("/background run-shell {command}")
            } else {
                format!("/background run-agent {queued}")
            };
            shell.push_user(&background_cmd);
            input.clear();
            cursor_pos = 0;
            is_processing = true;
            cancelled = false;
            active_phase = None;
            last_phase_event_at = Instant::now();
            last_phase_heartbeat_at = Instant::now();
            shell.active_tool = Some("processing...".to_string());
            on_submit(&background_cmd);
            continue;
        }
        if key == bindings.paste_hint {
            if let Some(image_data) = try_clipboard_image() {
                let bytes = image_data.len();
                let path = std::env::temp_dir().join(format!(
                    "codingbuddy-paste-{}.png",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis())
                        .unwrap_or(0)
                ));
                if std::fs::write(&path, &image_data).is_ok() {
                    pending_images.push(path.clone());
                    info_line = format!(
                        "Image pasted ({bytes} bytes). Will be included as @{} in next message.",
                        path.display()
                    );
                } else {
                    info_line = "Failed to save clipboard image.".to_string();
                }
            } else {
                info_line =
                    "No image in clipboard. Use terminal bracketed paste for text.".to_string();
            }
            continue;
        }
        // Ctrl+D — exit session
        if key == bindings.exit_session {
            break;
        }
        // Ctrl+L — clear terminal screen
        if key == bindings.clear_screen {
            shell.transcript.clear();
            info_line = "screen cleared".to_string();
            continue;
        }
        // Ctrl+J — alternative newline (like Shift+Enter)
        if key == bindings.newline_alt {
            input.insert(cursor_pos.min(input.len()), '\n');
            cursor_pos = (cursor_pos + 1).min(input.len());
            continue;
        }
        // Alt+P — switch model
        if key == bindings.switch_model {
            info_line = "model switch: use /model <name>".to_string();
            input = "/model ".to_string();
            cursor_pos = input.len();
            continue;
        }
        // Alt+T — toggle extended thinking
        if key == bindings.toggle_thinking {
            info_line = "thinking toggle: use /effort <low|medium|high>".to_string();
            input = "/effort ".to_string();
            cursor_pos = input.len();
            continue;
        }
        // Ctrl+F — kill all background agents
        if key == bindings.kill_background {
            info_line = "kill background agents: send /background kill-all".to_string();
            input = "/background kill-all".to_string();
            cursor_pos = input.len();
            continue;
        }
        if key == bindings.open_editor {
            let editor = std::env::var("EDITOR")
                .or_else(|_| std::env::var("VISUAL"))
                .unwrap_or_else(|_| "vim".into());
            let tmp =
                std::env::temp_dir().join(format!("codingbuddy-edit-{}.md", std::process::id()));
            let _ = std::fs::write(&tmp, &input);
            crossterm::terminal::disable_raw_mode().ok();
            let _ = std::process::Command::new(&editor).arg(&tmp).status();
            crossterm::terminal::enable_raw_mode().ok();
            if let Ok(text) = std::fs::read_to_string(&tmp) {
                input = text.trim_end().to_string();
                cursor_pos = input.len();
            }
            let _ = std::fs::remove_file(&tmp);
            info_line = format!("editor closed ({editor})");
            continue;
        }
        if key == bindings.stop {
            if let Some(last) = last_escape_at
                && last.elapsed() <= Duration::from_millis(600)
            {
                info_line = "rewind menu: use /rewind".to_string();
                input = "/rewind".to_string();
                cursor_pos = input.len();
                last_escape_at = None;
                continue;
            }
            last_escape_at = Some(Instant::now());
            info_line = "stop requested (escape; press Esc again for rewind)".to_string();
            continue;
        }
        if key == bindings.rewind_menu {
            info_line = "rewind menu: use /rewind".to_string();
            input = "/rewind".to_string();
            cursor_pos = input.len();
            continue;
        }
        // Alt+Right: accept one word of ML ghost text
        if key.code == KeyCode::Right
            && key.modifiers == KeyModifiers::ALT
            && let Some(word) = ml_ghost.accept_word()
        {
            input.push_str(&word);
            cursor_pos = input.len();
            info_line = "accepted word".to_string();
            continue;
        }
        if key == bindings.autocomplete {
            // ML ghost text has highest priority
            if cursor_pos >= input.len()
                && !input.starts_with('/')
                && ml_ghost.suggestion.is_some()
                && let Some(text) = ml_ghost.accept_full()
            {
                input.push_str(&text);
                cursor_pos = input.len();
                info_line = "accepted ML suggestion".to_string();
                continue;
            }
            // Then history ghost
            if cursor_pos >= input.len()
                && !input.starts_with('/')
                && let Some(ghost) = history_ghost_suffix(&history, &input)
            {
                input.push_str(&ghost);
                cursor_pos = input.len();
                info_line = "accepted suggestion".to_string();
                continue;
            }
            if input.starts_with('/') {
                let prefix = input.trim_start_matches('/').to_ascii_lowercase();
                if let Some((name, _)) = SLASH_COMMAND_CATALOG
                    .iter()
                    .find(|(name, _)| name.starts_with(&prefix))
                {
                    input = format!("/{name}");
                    cursor_pos = input.len();
                }
            } else if let Some(completed) = autocomplete_at_mention(&input) {
                input = completed;
                cursor_pos = input.len();
            } else if let Some(completed) = autocomplete_path_input(&input) {
                input = completed;
                cursor_pos = input.len();
            }
            continue;
        }
        if key == bindings.history_prev {
            if !history.is_empty() {
                if history_cursor.is_none() {
                    saved_input = input.clone();
                    history_cursor = Some(history.len() - 1);
                } else if let Some(idx) = history_cursor
                    && idx > 0
                {
                    history_cursor = Some(idx - 1);
                }
                if let Some(idx) = history_cursor
                    && let Some(entry) = history.get(idx)
                {
                    input = entry.clone();
                    cursor_pos = input.len();
                }
            }
            continue;
        }
        if key.code == KeyCode::Down && key.modifiers == KeyModifiers::NONE {
            if let Some(idx) = history_cursor {
                if idx + 1 < history.len() {
                    history_cursor = Some(idx + 1);
                    if let Some(entry) = history.get(idx + 1) {
                        input = entry.clone();
                        cursor_pos = input.len();
                    }
                } else {
                    history_cursor = None;
                    input = saved_input.clone();
                    cursor_pos = input.len();
                }
            }
            continue;
        }
        if key == bindings.newline {
            input.insert(cursor_pos.min(input.len()), '\n');
            cursor_pos = (cursor_pos + 1).min(input.len());
            continue;
        }
        if key == bindings.submit {
            if is_processing {
                info_line = "already processing, please wait...".to_string();
                continue;
            }
            // Backslash at end of line = continuation (multiline)
            if input.ends_with('\\') {
                input.pop(); // Remove trailing backslash
                input.push('\n');
                cursor_pos = input.len();
                continue;
            }
            let prompt = input.trim().to_string();
            if prompt.is_empty() {
                continue;
            }
            if let Some(result) = parse_vim_slash_command(&prompt) {
                match result {
                    Ok(vim_cmd) => {
                        apply_vim_command(
                            vim_cmd,
                            &mut vim_enabled,
                            &mut vim_mode,
                            &mut vim_command_buffer,
                            &mut vim_visual_anchor,
                            &mut vim_pending_operator,
                            &mut vim_pending_text_object,
                            &mut vim_pending_g,
                        );
                        info_line = if vim_enabled {
                            format!("vim mode on ({})", vim_mode.label())
                        } else {
                            "vim mode off".to_string()
                        };
                    }
                    Err(msg) => {
                        info_line = msg.to_string();
                    }
                }
                input.clear();
                cursor_pos = 0;
                history_cursor = None;
                continue;
            }
            // ! prefix — direct shell execution, bypasses LLM
            if let Some(cmd) = prompt.strip_prefix('!') {
                let cmd = cmd.trim();
                if !cmd.is_empty() {
                    history.push_back(prompt.clone());
                    if history.len() > 100 {
                        let _ = history.pop_front();
                    }
                    shell.push_user(&prompt);
                    let output = execute_bang_command(cmd);
                    shell.push_system(&output);
                    input.clear();
                    cursor_pos = 0;
                    history_cursor = None;
                }
                continue;
            }
            // /model (no args) — open interactive model picker
            if prompt == "/model" {
                model_picker = Some(ModelPickerState::new());
                info_line =
                    "Select model: Up/Down to move, Enter to confirm, Esc to cancel".to_string();
                input.clear();
                cursor_pos = 0;
                continue;
            }
            history.push_back(prompt.clone());
            if history.len() > 100 {
                let _ = history.pop_front();
            }
            shell.push_user(&prompt);
            input.clear();
            cursor_pos = 0;
            history_cursor = None;
            is_processing = true;
            cancelled = false;
            active_phase = None;
            last_phase_event_at = Instant::now();
            last_phase_heartbeat_at = Instant::now();
            shell.active_tool = Some("processing...".to_string());
            // Include any pending pasted images as @file references
            let final_prompt = if pending_images.is_empty() {
                prompt
            } else {
                let mut parts = vec![prompt];
                for img_path in pending_images.drain(..) {
                    parts.push(format!("@{}", img_path.display()));
                }
                parts.join(" ")
            };
            on_submit(&final_prompt);
            if vim_quit_after_submit {
                break;
            }
            continue;
        }
        match key.code {
            KeyCode::Backspace => {
                if cursor_pos > 0 && cursor_pos <= input.len() {
                    input.remove(cursor_pos - 1);
                    cursor_pos -= 1;
                    ml_ghost.on_keystroke();
                }
            }
            KeyCode::Delete => {
                if cursor_pos < input.len() {
                    input.remove(cursor_pos);
                    ml_ghost.on_keystroke();
                }
            }
            KeyCode::Left => {
                cursor_pos = cursor_pos.saturating_sub(1);
            }
            KeyCode::Right => {
                if cursor_pos < input.len() {
                    cursor_pos += 1;
                }
            }
            KeyCode::Home => {
                cursor_pos = 0;
            }
            KeyCode::End => {
                cursor_pos = input.len();
            }
            KeyCode::Char(ch) => {
                input.insert(cursor_pos.min(input.len()), ch);
                cursor_pos += 1;
                ml_ghost.on_keystroke();
                if autocomplete_dropdown.is_none() {
                    if ch == '@' {
                        // Trigger @ file autocomplete dropdown
                        let trigger = cursor_pos - 1;
                        let suggestions = autocomplete_at_suggestions("", &workspace_path);
                        if !suggestions.is_empty() {
                            autocomplete_dropdown =
                                Some(AutocompleteState::new(suggestions, trigger));
                            let lines = autocomplete_dropdown.as_ref().unwrap().display_lines(6);
                            info_line = lines.join("  ");
                        }
                    } else if ch == '/' && cursor_pos == 1 && input == "/" {
                        // Trigger slash command autocomplete at start of input
                        let suggestions = slash_command_suggestions("", 8);
                        if !suggestions.is_empty() {
                            autocomplete_dropdown = Some(AutocompleteState::new(suggestions, 0));
                            let lines = autocomplete_dropdown.as_ref().unwrap().display_lines(6);
                            info_line = lines.join("  ");
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // TerminalGuard handles raw mode + alternate screen on drop.
    // Show cursor and clear inline viewport before the guard runs.
    terminal.show_cursor()?;
    drop(_guard);
    Ok(())
}

#[cfg(test)]
mod tests;
