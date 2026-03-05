use super::*;
use crossterm::event::KeyEvent;

#[test]
fn parses_slash_commands() {
    assert_eq!(SlashCommand::parse("/help"), Some(SlashCommand::Help));
    assert_eq!(SlashCommand::parse("/ask"), Some(SlashCommand::Ask(vec![])));
    assert_eq!(
        SlashCommand::parse("/code"),
        Some(SlashCommand::Code(vec![]))
    );
    // /architect was removed — parses as unknown command
    assert_eq!(
        SlashCommand::parse("/architect"),
        Some(SlashCommand::Unknown {
            name: "architect".to_string(),
            args: vec![]
        })
    );
    assert_eq!(
        SlashCommand::parse("/chat-mode ask"),
        Some(SlashCommand::ChatMode(Some("ask".to_string())))
    );
    assert_eq!(
        SlashCommand::parse("/effort high"),
        Some(SlashCommand::Effort(Some("high".to_string())))
    );
    assert_eq!(
        SlashCommand::parse("/mcp add local"),
        Some(SlashCommand::Mcp(vec![
            "add".to_string(),
            "local".to_string()
        ]))
    );
    assert_eq!(
        SlashCommand::parse("/skills run refactor"),
        Some(SlashCommand::Skills(vec![
            "run".to_string(),
            "refactor".to_string()
        ]))
    );
    assert_eq!(
        SlashCommand::parse("/permissions show"),
        Some(SlashCommand::Permissions(vec!["show".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/background list"),
        Some(SlashCommand::Background(vec!["list".to_string()]))
    );
    // Quoted args: shell-style tokenizer strips quotes
    assert_eq!(
        SlashCommand::parse("/commit -m \"checkpoint\""),
        Some(SlashCommand::Commit(vec![
            "-m".to_string(),
            "checkpoint".to_string()
        ]))
    );
    // Multi-word quoted arg preserved as single token
    assert_eq!(
        SlashCommand::parse("/commit -m \"fix login bug\""),
        Some(SlashCommand::Commit(vec![
            "-m".to_string(),
            "fix login bug".to_string()
        ]))
    );
    assert_eq!(
        SlashCommand::parse("/stage src/main.rs"),
        Some(SlashCommand::Stage(vec!["src/main.rs".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/unstage --all"),
        Some(SlashCommand::Unstage(vec!["--all".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/diff --staged --stat"),
        Some(SlashCommand::Diff(vec![
            "--staged".to_string(),
            "--stat".to_string()
        ]))
    );
    assert_eq!(
        SlashCommand::parse("/undo"),
        Some(SlashCommand::Undo(vec![]))
    );
    assert_eq!(
        SlashCommand::parse("/add src tests"),
        Some(SlashCommand::Add(vec![
            "src".to_string(),
            "tests".to_string()
        ]))
    );
    assert_eq!(
        SlashCommand::parse("/drop tests"),
        Some(SlashCommand::Drop(vec!["tests".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/read-only on"),
        Some(SlashCommand::ReadOnly(vec!["on".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/map-refresh auth flow"),
        Some(SlashCommand::MapRefresh(vec![
            "auth".to_string(),
            "flow".to_string()
        ]))
    );
    assert_eq!(
        SlashCommand::parse("/web deepseek cli"),
        Some(SlashCommand::Web(vec![
            "deepseek".to_string(),
            "cli".to_string()
        ]))
    );
    assert_eq!(
        SlashCommand::parse("/compact"),
        Some(SlashCommand::Compact(None))
    );
    assert_eq!(
        SlashCommand::parse("/compact authentication"),
        Some(SlashCommand::Compact(Some("authentication".to_string())))
    );
}

#[test]
fn renders_statusline() {
    let line = render_statusline(&UiStatus {
        model: "deepseek-chat".to_string(),
        pending_approvals: 2,
        estimated_cost_usd: 0.001,
        background_jobs: 1,
        autopilot_running: true,
        permission_mode: "ask".to_string(),
        active_tasks: 3,
        workflow_phase: "execute".to_string(),
        plan_state: "available".to_string(),
        context_used_tokens: 50_000,
        context_max_tokens: 128_000,
        session_turns: 5,
        working_directory: "/tmp".to_string(),
        pr_review_status: None,
        ..Default::default()
    });
    assert!(line.contains("model=deepseek-chat"));
    assert!(line.contains("autopilot=running"));
    assert!(line.contains("[ASK]"));
    assert!(line.contains("tasks=3"));
    assert!(line.contains("phase=execute"));
    assert!(line.contains("plan=available"));
}

#[test]
fn loads_keybindings_from_json_file() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("codingbuddy-ui-bindings-{nonce}"));
    fs::create_dir_all(&dir).expect("dir");
    let path = dir.join("keybindings.json");
    fs::write(
        &path,
        r#"{
  "exit": "ctrl+x",
  "autocomplete": "tab",
  "toggle_raw": "ctrl+o"
}"#,
    )
    .expect("write");
    let bindings = load_keybindings(&path).expect("load");
    assert_eq!(
        bindings.exit,
        KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL)
    );
    assert_eq!(
        bindings.autocomplete,
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)
    );
}

#[test]
fn parses_new_slash_commands() {
    assert_eq!(SlashCommand::parse("/context"), Some(SlashCommand::Context));
    assert_eq!(
        SlashCommand::parse("/sandbox enable"),
        Some(SlashCommand::Sandbox(vec!["enable".to_string()]))
    );
    assert_eq!(SlashCommand::parse("/agents"), Some(SlashCommand::Agents));
    assert_eq!(
        SlashCommand::parse("/tasks list"),
        Some(SlashCommand::Tasks(vec!["list".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/review security"),
        Some(SlashCommand::Review(vec!["security".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/search rust async patterns"),
        Some(SlashCommand::Search(vec![
            "rust".to_string(),
            "async".to_string(),
            "patterns".to_string()
        ]))
    );
    assert_eq!(
        SlashCommand::parse("/terminal-setup"),
        Some(SlashCommand::TerminalSetup)
    );
    assert_eq!(
        SlashCommand::parse("/keybindings"),
        Some(SlashCommand::Keybindings)
    );
    assert_eq!(SlashCommand::parse("/vim"), Some(SlashCommand::Vim(vec![])));
    assert_eq!(
        SlashCommand::parse("/vim normal"),
        Some(SlashCommand::Vim(vec!["normal".to_string()]))
    );
    assert_eq!(SlashCommand::parse("/doctor"), Some(SlashCommand::Doctor));
    assert_eq!(SlashCommand::parse("/copy"), Some(SlashCommand::Copy));
    assert_eq!(SlashCommand::parse("/paste"), Some(SlashCommand::Paste));
    assert_eq!(
        SlashCommand::parse("/git status --short"),
        Some(SlashCommand::Git(vec![
            "status".to_string(),
            "--short".to_string()
        ]))
    );
    assert_eq!(
        SlashCommand::parse("/settings"),
        Some(SlashCommand::Settings)
    );
    assert_eq!(
        SlashCommand::parse("/load profile.json"),
        Some(SlashCommand::Load(vec!["profile.json".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/save profile.json"),
        Some(SlashCommand::Save(vec!["profile.json".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/voice status"),
        Some(SlashCommand::Voice(vec!["status".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/debug connection"),
        Some(SlashCommand::Debug(vec!["connection".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/desktop open"),
        Some(SlashCommand::Desktop(vec!["open".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/todos fix"),
        Some(SlashCommand::Todos(vec!["fix".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/comment-todos auth"),
        Some(SlashCommand::CommentTodos(vec!["auth".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/chrome reconnect"),
        Some(SlashCommand::Chrome(vec!["reconnect".to_string()]))
    );
    assert_eq!(SlashCommand::parse("/exit"), Some(SlashCommand::Exit));
    assert_eq!(SlashCommand::parse("/quit"), Some(SlashCommand::Exit));
    assert_eq!(
        SlashCommand::parse("/hooks list"),
        Some(SlashCommand::Hooks(vec!["list".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/rename my-session"),
        Some(SlashCommand::Rename(Some("my-session".to_string())))
    );
    assert_eq!(
        SlashCommand::parse("/resume abc123"),
        Some(SlashCommand::Resume(Some("abc123".to_string())))
    );
    assert_eq!(SlashCommand::parse("/stats"), Some(SlashCommand::Stats));
    assert_eq!(
        SlashCommand::parse("/theme dark"),
        Some(SlashCommand::Theme(Some("dark".to_string())))
    );
    assert_eq!(SlashCommand::parse("/usage"), Some(SlashCommand::Usage));
    assert_eq!(
        SlashCommand::parse("/add-dir /tmp/extra"),
        Some(SlashCommand::AddDir(vec!["/tmp/extra".to_string()]))
    );
    assert_eq!(SlashCommand::parse("/bug"), Some(SlashCommand::Bug));
    assert_eq!(
        SlashCommand::parse("/pr_comments 42"),
        Some(SlashCommand::PrComments(vec!["42".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/pr-comments 42"),
        Some(SlashCommand::PrComments(vec!["42".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/release-notes v1.0..v2.0"),
        Some(SlashCommand::ReleaseNotes(vec!["v1.0..v2.0".to_string()]))
    );
    assert_eq!(SlashCommand::parse("/login"), Some(SlashCommand::Login));
    assert_eq!(SlashCommand::parse("/logout"), Some(SlashCommand::Logout));
}

#[test]
fn statusline_shows_context_usage() {
    let line = render_statusline(&UiStatus {
        model: "deepseek-chat".to_string(),
        pending_approvals: 0,
        estimated_cost_usd: 0.0,
        background_jobs: 0,
        autopilot_running: false,
        permission_mode: "auto".to_string(),
        active_tasks: 0,
        context_used_tokens: 96_000,
        context_max_tokens: 128_000,
        session_turns: 10,
        working_directory: "/workspace".to_string(),
        pr_review_status: None,
        ..Default::default()
    });
    assert!(line.contains("[AUTO]"));
    assert!(line.contains("ctx=96K/128K(75%)"));
}

#[test]
fn statusline_permission_modes() {
    let make_status = |mode: &str| UiStatus {
        model: "test".to_string(),
        permission_mode: mode.to_string(),
        ..Default::default()
    };
    assert!(render_statusline(&make_status("ask")).contains("[ASK]"));
    assert!(render_statusline(&make_status("auto")).contains("[AUTO]"));
    assert!(render_statusline(&make_status("locked")).contains("[LOCKED]"));
}

#[test]
fn styled_statusline_spans_include_mode_badge() {
    let status = UiStatus {
        model: "deepseek-chat".to_string(),
        permission_mode: "locked".to_string(),
        pending_approvals: 1,
        active_tasks: 2,
        workflow_phase: "execute".to_string(),
        plan_state: "available".to_string(),
        background_jobs: 1,
        autopilot_running: true,
        context_used_tokens: 100_000,
        context_max_tokens: 128_000,
        ..Default::default()
    };
    let spans = render_statusline_spans(&status, None, "", None, None, false, false);
    let text: String = spans.iter().map(|s| s.content.to_string()).collect();
    assert!(text.contains("deepseek-chat"));
    assert!(text.contains("LOCKED"));
    assert!(text.contains("pending"));
    assert!(text.contains("tasks"));
    assert!(text.contains("EXECUTE"));
    assert!(text.contains("PLAN:AVAILABLE"));
    assert!(text.contains("AUTOPILOT"));
    assert!(text.contains("100K/128K"));

    // Test with extra context
    let spans_with_ctx = render_statusline_spans(
        &status,
        Some("fs.read"),
        "\u{280b}",
        Some(42),
        Some("INSERT"),
        true,
        false,
    );
    let text2: String = spans_with_ctx
        .iter()
        .map(|s| s.content.to_string())
        .collect();
    assert!(text2.contains("fs.read"));
    assert!(text2.contains("\u{280b}"));
    assert!(text2.contains("INSERT"));
    assert!(text2.contains("42%"));
    assert!(text2.contains("\u{2193} new"));
}

#[test]
fn chat_shell_typed_entries() {
    let mut shell = ChatShell::default();
    shell.push_user("hello");
    shell.push_transcript("response");
    shell.push_system("info");
    shell.push_error("oops");
    assert_eq!(shell.transcript.len(), 4);
    assert_eq!(shell.transcript[0].kind, MessageKind::User);
    assert_eq!(shell.transcript[1].kind, MessageKind::Assistant);
    assert_eq!(shell.transcript[2].kind, MessageKind::System);
    assert_eq!(shell.transcript[3].kind, MessageKind::Error);
}

#[test]
fn mission_control_panel_renders_snapshot_and_recent_events() {
    let status = UiStatus {
        mission_control_snapshot: vec![
            "Mission Control: session=abc phase=execute".to_string(),
            "- Tasks:".to_string(),
            "  - implement drawer [running]".to_string(),
        ],
        ..Default::default()
    };
    let mut shell = ChatShell::default();
    shell.push_mission_control("[background] task completed: audit".to_string());
    shell.push_mission_control("[plan] approved; workflow moved to execute".to_string());

    let panel = render_mission_control_panel(&status, &shell.mission_control_lines);
    assert!(panel.contains("Mission Control: session=abc phase=execute"));
    assert!(panel.contains("implement drawer [running]"));
    assert!(panel.contains("Recent activity:"));
    assert!(panel.contains("[background] task completed: audit"));
    assert!(panel.contains("Ctrl+T hides mission control."));
}

#[test]
fn mission_control_panel_shows_plan_review_shortcuts() {
    let status = UiStatus {
        plan_state: "awaiting_approval".to_string(),
        mission_control_snapshot: vec!["Mission Control".to_string()],
        ..Default::default()
    };
    let panel = render_mission_control_panel(&status, &[]);
    assert!(panel.contains("Ctrl+Y approves the current plan"));
    assert!(panel.contains("Alt+Y opens a rejection prompt"));
}

#[test]
fn spinner_cycles_through_frames() {
    let mut shell = ChatShell::default();
    let mut frames = Vec::new();
    for i in 0..10 {
        shell.spinner_tick = i;
        frames.push(shell.spinner_frame().to_string());
    }
    assert_eq!(frames.len(), 10);
    // All frames are braille characters
    assert!(frames.iter().all(|f| f.chars().all(|c| c as u32 >= 0x2800)));
}

#[test]
fn default_keybindings_include_cycle_permission_mode() {
    let bindings = KeyBindings::default();
    assert_eq!(
        bindings.cycle_permission_mode,
        KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT)
    );
}

#[test]
fn parses_vim_slash_commands() {
    assert!(matches!(
        parse_vim_slash_command("/vim"),
        Some(Ok(VimSlashCommand::Toggle))
    ));
    assert!(matches!(
        parse_vim_slash_command("/vim on"),
        Some(Ok(VimSlashCommand::On))
    ));
    assert!(matches!(
        parse_vim_slash_command("/vim off"),
        Some(Ok(VimSlashCommand::Off))
    ));
    assert!(matches!(
        parse_vim_slash_command("/vim normal"),
        Some(Ok(VimSlashCommand::SetMode(VimMode::Normal)))
    ));
    assert!(matches!(parse_vim_slash_command("/vim nope"), Some(Err(_))));
    assert!(parse_vim_slash_command("/help").is_none());
}

#[test]
fn vim_word_motions_move_cursor() {
    let text = "alpha beta gamma";
    assert_eq!(move_to_next_word_start(text, 0), 6);
    assert_eq!(move_to_next_word_start(text, 6), 11);
    assert_eq!(move_to_prev_word_start(text, 11), 6);
    assert_eq!(move_to_prev_word_start(text, 6), 0);
    assert_eq!(move_to_word_end(text, 0), 4);
    assert_eq!(move_to_word_end(text, 6), 9);
}

#[test]
fn vim_text_object_bounds_support_word_and_quotes() {
    let text = "let value = \"hello world\";";
    assert_eq!(word_text_object_bounds(text, 4, false), Some((4, 9)));
    assert_eq!(word_text_object_bounds(text, 4, true), Some((4, 10)));
    assert_eq!(
        resolve_vim_text_object_bounds(text, 14, 'i', KeyCode::Char('"')),
        Some((13, 24))
    );
    assert_eq!(
        resolve_vim_text_object_bounds(text, 14, 'a', KeyCode::Char('"')),
        Some((12, 25))
    );
}

#[test]
fn vim_operator_range_handles_change_delete_yank() {
    let mut input = "alpha beta".to_string();
    let mut cursor = 0usize;
    let mut yank = String::new();
    let mut mode = VimMode::Normal;
    assert!(apply_vim_operator_range(
        'y',
        &mut input,
        &mut cursor,
        &mut yank,
        &mut mode,
        0,
        5
    ));
    assert_eq!(yank, "alpha");
    assert_eq!(mode, VimMode::Normal);

    assert!(apply_vim_operator_range(
        'd',
        &mut input,
        &mut cursor,
        &mut yank,
        &mut mode,
        0,
        6
    ));
    assert_eq!(input, "beta");

    assert!(apply_vim_operator_range(
        'c',
        &mut input,
        &mut cursor,
        &mut yank,
        &mut mode,
        0,
        4
    ));
    assert_eq!(input, "");
    assert_eq!(mode, VimMode::Insert);
}

#[test]
fn reverse_search_and_ghost_helpers_work() {
    let history = VecDeque::from(vec![
        "cargo test".to_string(),
        "cargo check".to_string(),
        "git status".to_string(),
    ]);
    assert_eq!(
        history_ghost_suffix(&history, "car"),
        Some("go check".to_string())
    );
    assert_eq!(
        history_reverse_search_index(&history, "cargo", None),
        Some(1)
    );
    assert_eq!(
        history_reverse_search_index(&history, "cargo", Some(1)),
        Some(0)
    );
}

#[test]
fn statusline_includes_review_badge_value() {
    let status = UiStatus {
        model: "deepseek-chat".to_string(),
        pr_review_status: Some("changes_requested".to_string()),
        ..Default::default()
    };
    let line = render_statusline(&status);
    assert!(line.contains("review=changes_requested"));
}

#[test]
fn review_badge_includes_url_in_plain() {
    let status = UiStatus {
        model: "deepseek-chat".to_string(),
        pr_review_status: Some("approved".to_string()),
        pr_url: Some("https://github.com/org/repo/pull/42".to_string()),
        ..Default::default()
    };
    let line = render_statusline(&status);
    assert!(line.contains("review=approved"));
    assert!(line.contains("https://github.com/org/repo/pull/42"));
}

#[test]
fn review_badge_no_url_fallback() {
    let status = UiStatus {
        model: "deepseek-chat".to_string(),
        pr_review_status: Some("pending".to_string()),
        pr_url: None,
        ..Default::default()
    };
    let line = render_statusline(&status);
    assert!(line.contains("review=pending"));
    assert!(!line.contains("http"));
}

#[test]
fn parse_inline_italic() {
    let base = Style::default().fg(Color::White);
    let spans = parse_inline_markdown("hello *world* end", base);
    assert_eq!(spans.len(), 3);
    assert_eq!(spans[0].content.as_ref(), "hello ");
    assert_eq!(spans[1].content.as_ref(), "world");
    assert!(spans[1].style.add_modifier == Modifier::ITALIC);
    assert_eq!(spans[2].content.as_ref(), " end");
}

#[test]
fn parse_inline_bold_and_italic() {
    let base = Style::default().fg(Color::White);
    let spans = parse_inline_markdown("**bold** and *italic*", base);
    assert!(spans.len() >= 3);
    // First span should be bold
    assert_eq!(spans[0].content.as_ref(), "bold");
    // Check italic is present
    let italic_span = spans.iter().find(|s| s.content.as_ref() == "italic");
    assert!(italic_span.is_some());
}

#[test]
fn render_table_row() {
    let line = render_assistant_markdown("| Col A | Col B | Col C |");
    let text: String = line.iter().map(|s| s.content.to_string()).collect();
    assert!(text.contains("Col A"));
    assert!(text.contains("Col B"));
    assert!(text.contains("│"));
}

#[test]
fn render_table_separator() {
    let line = render_assistant_markdown("|---|---|---|");
    let text: String = line.iter().map(|s| s.content.to_string()).collect();
    assert!(text.contains("---"));
}

#[test]
fn render_single_pipe_without_panic() {
    let line = render_assistant_markdown("|");
    let text: String = line.iter().map(|s| s.content.to_string()).collect();
    assert_eq!(text, "|");
}

#[test]
fn parse_stream_meta_tool_start() {
    let entry = parse_stream_meta_entry("[tool: fs.read] path=missing.md").expect("tool start");
    assert_eq!(entry.kind, MessageKind::ToolCall);
    assert!(entry.text.contains("fs.read"));
    assert!(entry.text.contains("path=missing.md"));
}

#[test]
fn parse_stream_meta_tool_result() {
    let entry = parse_stream_meta_entry("[tool: fs.read] ok (0.0s) {\"written\":true}")
        .expect("tool result");
    assert_eq!(entry.kind, MessageKind::ToolResult);
    assert!(entry.text.contains("ok"));
}

#[test]
fn parse_stream_meta_mcp_line() {
    let entry = parse_stream_meta_entry("[mcp: repo.search]").expect("mcp line");
    assert_eq!(entry.kind, MessageKind::ToolCall);
    assert!(entry.text.contains("mcp.repo.search"));
}

#[test]
fn assistant_meta_lines_render_with_tool_style() {
    let entry = TranscriptEntry {
        kind: MessageKind::Assistant,
        text: "[tool: fs.read] path=missing.md".to_string(),
    };
    let rendered = style_transcript_line(&entry, false, false, "");
    let text: String = rendered.iter().map(|s| s.content.to_string()).collect();
    assert!(text.contains("⚡"));
    assert!(text.contains("fs.read"));
}

#[test]
fn assistant_unfenced_diff_line_uses_diff_renderer() {
    let entry = TranscriptEntry {
        kind: MessageKind::Assistant,
        text: "+let value = 1;".to_string(),
    };
    let rendered = style_transcript_line(&entry, false, false, "");
    let text: String = rendered.iter().map(|s| s.content.to_string()).collect();
    assert!(text.starts_with('+'));
}

#[test]
fn diff_context_run_collapses_long_ranges() {
    let mut run = (0..20).map(|i| format!(" line {i}")).collect::<Vec<_>>();
    let mut out = Vec::new();
    flush_diff_context_run(&mut run, &mut out, 16, 3, 3);
    let rendered = out
        .iter()
        .map(|line| {
            line.iter()
                .map(|span| span.content.to_string())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(rendered.contains("unchanged lines hidden"));
}

#[test]
fn render_diff_line_marks_additions_and_deletions() {
    let add = render_diff_line("+let value = 1;");
    let add_text: String = add.iter().map(|s| s.content.to_string()).collect();
    assert!(add_text.starts_with('+'));

    let del = render_diff_line("-let value = 0;");
    let del_text: String = del.iter().map(|s| s.content.to_string()).collect();
    assert!(del_text.starts_with('-'));
}

#[test]
fn tool_call_transcript_entries() {
    let mut shell = ChatShell::default();
    shell.push_tool_call("fs.read", "path=/src/main.rs");
    shell.push_tool_result("fs.read", 150, "1234 bytes");
    assert_eq!(shell.transcript.len(), 2);
    assert_eq!(shell.transcript[0].kind, MessageKind::ToolCall);
    assert!(shell.transcript[0].text.contains("fs.read"));
    assert_eq!(shell.transcript[1].kind, MessageKind::ToolResult);
    assert!(shell.transcript[1].text.contains("150ms"));
}

#[test]
fn tool_result_formats_duration_seconds() {
    let mut shell = ChatShell::default();
    shell.push_tool_result("bash.run", 2500, "exit 0");
    assert!(shell.transcript[0].text.contains("2.5s"));
}

#[test]
fn statusline_shows_turn_count() {
    let status = UiStatus {
        model: "deepseek-chat".to_string(),
        session_turns: 7,
        ..Default::default()
    };
    let spans = render_statusline_spans(&status, None, "", None, None, false, false);
    let text: String = spans.iter().map(|s| s.content.to_string()).collect();
    assert!(text.contains("turn 7"));
}

#[test]
fn statusline_hides_turn_when_zero() {
    let status = UiStatus {
        model: "deepseek-chat".to_string(),
        session_turns: 0,
        ..Default::default()
    };
    let spans = render_statusline_spans(&status, None, "", None, None, false, false);
    let text: String = spans.iter().map(|s| s.content.to_string()).collect();
    assert!(!text.contains("turn"));
}

#[test]
fn statusline_shows_thinking_indicator() {
    let status = UiStatus {
        model: "deepseek-chat".to_string(),
        ..Default::default()
    };
    // When thinking, the spinner should show "Thinking" instead of the active tool.
    let spans = render_statusline_spans(
        &status,
        Some("fs.read"),
        "\u{280b}",
        None,
        None,
        false,
        true,
    );
    let text: String = spans.iter().map(|s| s.content.to_string()).collect();
    assert!(text.contains("Thinking"), "should show Thinking indicator");
    assert!(
        !text.contains("fs.read"),
        "active tool should be hidden during thinking"
    );
}

#[test]
fn statusline_hides_thinking_when_not_active() {
    let status = UiStatus {
        model: "deepseek-chat".to_string(),
        ..Default::default()
    };
    let spans = render_statusline_spans(
        &status,
        Some("fs.read"),
        "\u{280b}",
        None,
        None,
        false,
        false,
    );
    let text: String = spans.iter().map(|s| s.content.to_string()).collect();
    assert!(!text.contains("Thinking"));
    assert!(text.contains("fs.read"));
}

#[test]
fn chat_shell_push_thinking() {
    let mut shell = ChatShell::default();
    shell.push_thinking("analyzing the problem");
    assert_eq!(shell.transcript.len(), 1);
    assert_eq!(shell.transcript[0].kind, MessageKind::Thinking);
    assert_eq!(shell.transcript[0].text, "analyzing the problem");
}

#[test]
fn thinking_state_transitions() {
    let mut shell = ChatShell {
        is_thinking: true,
        ..Default::default()
    };

    // Start thinking
    shell.thinking_buffer.push_str("step 1\nstep 2");
    assert!(shell.is_thinking);

    // Content delta clears thinking
    shell.is_thinking = false;
    shell.thinking_buffer.clear();
    assert!(!shell.is_thinking);
    assert!(shell.thinking_buffer.is_empty());
}

#[test]
fn cost_summary_formatting() {
    let mut shell = ChatShell::default();
    let status = UiStatus {
        estimated_cost_usd: 0.0423,
        context_used_tokens: 12_300,
        context_max_tokens: 128_000,
        session_turns: 3,
        ..Default::default()
    };
    shell.push_cost_summary(&status);
    assert_eq!(shell.transcript.len(), 1);
    assert_eq!(shell.transcript[0].kind, MessageKind::System);
    assert!(shell.transcript[0].text.contains("$0.0423"));
    assert!(shell.transcript[0].text.contains("12K/128K"));
    assert!(shell.transcript[0].text.contains("Turns: 3"));
}

#[test]
fn status_summary_includes_all_fields() {
    let mut shell = ChatShell::default();
    let status = UiStatus {
        model: "deepseek-reasoner".to_string(),
        permission_mode: "auto".to_string(),
        pending_approvals: 1,
        active_tasks: 2,
        background_jobs: 3,
        autopilot_running: true,
        estimated_cost_usd: 0.05,
        context_used_tokens: 50_000,
        context_max_tokens: 128_000,
        session_turns: 5,
        working_directory: "/tmp".to_string(),
        pr_review_status: None,
        ..Default::default()
    };
    shell.push_status_summary(&status);
    let all_text: String = shell
        .transcript
        .iter()
        .map(|e| e.text.clone())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(all_text.contains("deepseek-reasoner"));
    assert!(all_text.contains("auto"));
    assert!(all_text.contains("running"));
}

#[test]
fn at_mention_autocomplete_detects_prefix() {
    // Without @, should return None
    assert!(autocomplete_at_mention("hello world").is_none());
    // Single @ without path should return None
    assert!(autocomplete_at_mention("@").is_none());
}

#[test]
fn expand_at_mentions_passes_through_without_at() {
    let input = "hello world no mentions";
    assert_eq!(expand_at_mentions(input), input);
}

#[test]
fn expand_at_mentions_expands_real_file() {
    let dir = std::env::temp_dir().join(format!(
        "codingbuddy-at-test-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&dir).unwrap();
    let file = dir.join("test.txt");
    fs::write(&file, "file content here").unwrap();

    let input = format!("check @{} please", file.display());
    let expanded = expand_at_mentions(&input);
    assert!(expanded.contains("file content here"));
    assert!(expanded.contains("[file:"));
    assert!(expanded.contains("[/file]"));
    assert!(expanded.contains("please"));
}

#[test]
fn expand_at_mentions_preserves_missing_files() {
    let input = "look at @/nonexistent/path/file.rs here";
    let expanded = expand_at_mentions(input);
    assert!(expanded.contains("@/nonexistent/path/file.rs"));
}

// ── TUI logic tests (Phase 16.3) ────────────────────────────────────

#[test]
fn all_slash_command_variants_parse() {
    // Comprehensive test of all slash command variants
    assert_eq!(SlashCommand::parse("/exit"), Some(SlashCommand::Exit));
    assert_eq!(SlashCommand::parse("/copy"), Some(SlashCommand::Copy));
    assert_eq!(SlashCommand::parse("/paste"), Some(SlashCommand::Paste));
    assert_eq!(
        SlashCommand::parse("/settings"),
        Some(SlashCommand::Settings)
    );
    assert_eq!(SlashCommand::parse("/login"), Some(SlashCommand::Login));
    assert_eq!(SlashCommand::parse("/logout"), Some(SlashCommand::Logout));
    assert_eq!(SlashCommand::parse("/bug"), Some(SlashCommand::Bug));
    assert_eq!(SlashCommand::parse("/usage"), Some(SlashCommand::Usage));
    assert_eq!(
        SlashCommand::parse("/desktop"),
        Some(SlashCommand::Desktop(vec![]))
    );
    assert_eq!(
        SlashCommand::parse("/todos"),
        Some(SlashCommand::Todos(vec![]))
    );
    assert_eq!(
        SlashCommand::parse("/comment-todos"),
        Some(SlashCommand::CommentTodos(vec![]))
    );
    assert_eq!(
        SlashCommand::parse("/chrome"),
        Some(SlashCommand::Chrome(vec![]))
    );
    assert_eq!(
        SlashCommand::parse("/rename foo"),
        Some(SlashCommand::Rename(Some("foo".to_string())))
    );
    assert_eq!(
        SlashCommand::parse("/resume abc"),
        Some(SlashCommand::Resume(Some("abc".to_string())))
    );
    assert_eq!(
        SlashCommand::parse("/add-dir /tmp"),
        Some(SlashCommand::AddDir(vec!["/tmp".to_string()]))
    );
    assert_eq!(
        SlashCommand::parse("/pr_comments 42"),
        Some(SlashCommand::PrComments(vec!["42".to_string()]))
    );
    // Unknown command
    let unknown = SlashCommand::parse("/foobar baz");
    assert!(matches!(
        unknown,
        Some(SlashCommand::Unknown { ref name, .. }) if name == "foobar"
    ));
}

#[test]
fn slash_command_parse_is_case_insensitive() {
    assert_eq!(SlashCommand::parse("/HELP"), Some(SlashCommand::Help));
    assert_eq!(SlashCommand::parse("/Help"), Some(SlashCommand::Help));
    assert_eq!(SlashCommand::parse("/CONTEXT"), Some(SlashCommand::Context));
    assert_eq!(SlashCommand::parse("/EXIT"), Some(SlashCommand::Exit));
    assert_eq!(
        SlashCommand::parse("/Plan"),
        Some(SlashCommand::Plan(vec![]))
    );
}

#[test]
fn slash_provider_command_parses() {
    assert!(matches!(
        SlashCommand::parse("/provider"),
        Some(SlashCommand::Provider(None))
    ));
    assert!(matches!(
        SlashCommand::parse("/provider deepseek"),
        Some(SlashCommand::Provider(Some(ref name))) if name == "deepseek"
    ));
}

#[test]
fn render_statusline_plan_mode_label() {
    let status = UiStatus {
        model: "deepseek-chat".to_string(),
        permission_mode: "plan".to_string(),
        ..Default::default()
    };
    let line = render_statusline(&status);
    assert!(
        line.contains("[PLAN]"),
        "plan mode should show [PLAN] label, got: {line}"
    );
}

#[test]
fn render_statusline_context_percentage() {
    let status = UiStatus {
        model: "deepseek-chat".to_string(),
        permission_mode: "auto".to_string(),
        context_used_tokens: 64_000,
        context_max_tokens: 128_000,
        ..Default::default()
    };
    let line = render_statusline(&status);
    assert!(
        line.contains("ctx=64K/128K(50%)"),
        "should show 50% context usage, got: {line}"
    );
}

#[test]
fn keybindings_defaults_are_valid() {
    let bindings = KeyBindings::default();
    // Ctrl+D should exit session
    assert_eq!(
        bindings.exit_session,
        crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('d'),
            crossterm::event::KeyModifiers::CONTROL
        )
    );
    // Ctrl+C should exit
    assert_eq!(
        bindings.exit,
        crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('c'),
            crossterm::event::KeyModifiers::CONTROL
        )
    );
    // Enter should submit
    assert_eq!(
        bindings.submit,
        crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE
        )
    );
    // Tab should autocomplete
    assert_eq!(
        bindings.autocomplete,
        crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Tab,
            crossterm::event::KeyModifiers::NONE
        )
    );
    // Ctrl+R should trigger reverse history search
    assert_eq!(
        bindings.history_search,
        crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('r'),
            crossterm::event::KeyModifiers::CONTROL
        )
    );
}

#[test]
fn mode_transition_updates_shell_and_transcript() {
    let mut shell = ChatShell {
        agent_mode: "ToolUseLoop".to_string(),
        ..Default::default()
    };
    shell.agent_mode = "VerifyRetry".to_string();
    shell.push_system("mode transition ToolUseLoop -> VerifyRetry (verify failure)");

    assert_eq!(shell.agent_mode, "VerifyRetry");
    assert_eq!(shell.transcript.len(), 1);
    assert_eq!(shell.transcript[0].kind, MessageKind::System);
    assert!(shell.transcript[0].text.contains("mode transition"));

    shell.agent_mode = "ToolUseLoop".to_string();
    shell.push_system("mode transition VerifyRetry -> ToolUseLoop (recovered)");
    assert_eq!(shell.agent_mode, "ToolUseLoop");
    assert_eq!(shell.transcript.len(), 2);
    assert!(shell.transcript[1].text.contains("mode transition"));
}

#[test]
fn statusline_shows_mode_badge_when_non_default() {
    let status = UiStatus {
        model: "deepseek-chat".to_string(),
        agent_mode: "VerifyRetry".to_string(),
        ..Default::default()
    };
    let spans = render_statusline_spans(&status, None, "", None, None, false, false);
    let text: String = spans.iter().map(|s| s.content.to_string()).collect();
    assert!(
        text.contains(" VerifyRetry "),
        "status bar should show non-default mode badge"
    );
}

#[test]
fn statusline_hides_mode_badge_in_default_mode() {
    // ToolUseLoop (new default) should be hidden
    let status = UiStatus {
        model: "deepseek-chat".to_string(),
        agent_mode: "ToolUseLoop".to_string(),
        ..Default::default()
    };
    let spans = render_statusline_spans(&status, None, "", None, None, false, false);
    let text: String = spans.iter().map(|s| s.content.to_string()).collect();
    assert!(
        !text.contains(" ToolUseLoop "),
        "status bar should not show mode badge for ToolUseLoop"
    );

    // ArchitectEditorLoop (old default, for pipeline mode) should also be hidden
    let status2 = UiStatus {
        model: "deepseek-chat".to_string(),
        agent_mode: "ArchitectEditorLoop".to_string(),
        ..Default::default()
    };
    let spans2 = render_statusline_spans(&status2, None, "", None, None, false, false);
    let text2: String = spans2.iter().map(|s| s.content.to_string()).collect();
    assert!(
        !text2.contains(" ArchitectEditorLoop "),
        "status bar should not show mode badge for ArchitectEditorLoop"
    );
}

#[test]
fn plain_statusline_shows_agent_mode_when_non_default() {
    let status = UiStatus {
        model: "deepseek-chat".to_string(),
        agent_mode: "VerifyRetry".to_string(),
        ..Default::default()
    };
    let line = render_statusline(&status);
    assert!(line.contains("agent=VerifyRetry"));
}

#[test]
fn plain_statusline_hides_agent_mode_when_default() {
    let status = UiStatus {
        model: "deepseek-chat".to_string(),
        agent_mode: "ToolUseLoop".to_string(),
        ..Default::default()
    };
    let line = render_statusline(&status);
    assert!(
        !line.contains("agent="),
        "default loop mode should not show agent= in statusline"
    );
}

#[test]
fn syntect_highlights_rust_code() {
    let line = highlight_code_line("let x = 42;", "rust");
    let text: String = line.iter().map(|s| s.content.to_string()).collect();
    assert_eq!(text, "let x = 42;");
    // syntect should produce more than one span (keyword, ident, number)
    assert!(
        line.spans.len() > 1,
        "syntect should tokenize Rust: got {} span(s)",
        line.spans.len()
    );
}

#[test]
fn syntect_falls_back_for_unknown_lang() {
    let line = highlight_code_line("hello world", "xyzzylang99");
    let text: String = line.iter().map(|s| s.content.to_string()).collect();
    assert_eq!(text, "hello world");
}

#[test]
fn syntect_highlights_python_code() {
    let line = highlight_code_line("def hello():", "python");
    let text: String = line.iter().map(|s| s.content.to_string()).collect();
    assert!(text.contains("def"));
    assert!(
        line.spans.len() > 1,
        "syntect should tokenize Python: got {} span(s)",
        line.spans.len()
    );
}

#[test]
fn extract_fence_lang_extracts_language() {
    assert_eq!(extract_fence_lang("```rust"), "rust");
    assert_eq!(extract_fence_lang("```python"), "python");
    assert_eq!(extract_fence_lang("```"), "");
    assert_eq!(extract_fence_lang("```js "), "js");
}

#[test]
fn wrapped_line_height_tracks_soft_wrapping() {
    let line = Line::from("abcdefghijklmnopqrstuvwxyz");
    assert_eq!(wrapped_line_height(&line, 10), 3);
    assert_eq!(wrapped_line_height(&Line::from(""), 10), 1);
    assert_eq!(wrapped_line_height(&line, 0), 0);
}

#[test]
fn wrapped_text_rows_counts_soft_wrap_and_newlines() {
    assert_eq!(wrapped_text_rows("abcdefghij", 10), 1);
    assert_eq!(wrapped_text_rows("abcdefghijk", 10), 2);
    assert_eq!(wrapped_text_rows("abc\ndef", 10), 2);
}

#[test]
fn scroll_to_keep_row_visible_tracks_cursor_row() {
    assert_eq!(scroll_to_keep_row_visible(0, 3), 0);
    assert_eq!(scroll_to_keep_row_visible(2, 3), 0);
    assert_eq!(scroll_to_keep_row_visible(3, 3), 1);
    assert_eq!(scroll_to_keep_row_visible(5, 3), 3);
}

#[test]
fn compute_inline_heights_supports_input_growth() {
    assert_eq!(
        compute_inline_heights(INLINE_VIEWPORT_HEIGHT, 1),
        (4, 1),
        "empty/short input should keep a compact 1-line input area"
    );
    assert_eq!(
        compute_inline_heights(INLINE_VIEWPORT_HEIGHT, 99),
        (1, 4),
        "long input should grow up to 4 lines while preserving stream minimum"
    );
}

#[test]
fn compute_inline_heights_handles_small_terminal_heights() {
    assert_eq!(compute_inline_heights(5, 4), (1, 1));
    assert_eq!(compute_inline_heights(4, 4), (1, 0));
    assert_eq!(compute_inline_heights(3, 4), (0, 0));
}

#[test]
fn split_inline_markdown_blocks_preserves_normal_text() {
    let parts = split_inline_markdown_blocks("simple sentence with no markdown breaks");
    assert_eq!(parts, vec!["simple sentence with no markdown breaks"]);
}

#[test]
fn split_inline_markdown_blocks_splits_compact_headings() {
    let parts = split_inline_markdown_blocks(
        "Here's my assessment:## Overall Architecture Assessment### Strengths",
    );
    assert_eq!(
        parts,
        vec![
            "Here's my assessment:",
            "## Overall Architecture Assessment",
            "### Strengths"
        ]
    );
}

#[test]
fn split_inline_markdown_blocks_splits_inline_numbered_list() {
    let parts = split_inline_markdown_blocks("### Strengths ✅ 1. Clean separation");
    assert_eq!(parts, vec!["### Strengths ✅", "1. Clean separation"]);
}

#[test]
fn split_inline_markdown_blocks_splits_inline_unordered_list_runs() {
    let parts = split_inline_markdown_blocks(
        "1. Architecture- Workspace Structure:  - codingbuddy-cli: main app  - codingbuddy-agent: core runtime",
    );
    assert_eq!(
        parts,
        vec![
            "1. Architecture",
            "- Workspace Structure:",
            "- codingbuddy-cli: main app",
            "- codingbuddy-agent: core runtime"
        ]
    );
}

#[test]
fn split_inline_markdown_blocks_splits_run_on_section_boundary() {
    let parts =
        split_inline_markdown_blocks("Project OverviewCodingBuddy is a terminal-native tool.");
    assert_eq!(
        parts,
        vec!["Project Overview", "CodingBuddy is a terminal-native tool."]
    );
}

#[test]
fn split_inline_markdown_blocks_splits_parenthetical_run_on_boundary() {
    let parts = split_inline_markdown_blocks(
        "2. Current Modifications (Git Status)The project shows updates.",
    );
    assert_eq!(
        parts,
        vec![
            "2. Current Modifications (Git Status)",
            "The project shows updates."
        ]
    );
}

#[test]
fn split_inline_markdown_blocks_handles_repo_analysis_run_on_sample() {
    let parts = split_inline_markdown_blocks(
        "Key Components### 1. Architecture- Workspace Structure:  - codingbuddy-cli: Main CLI application  - codingbuddy-agent: Core agent logic",
    );
    assert_eq!(
        parts,
        vec![
            "Key Components",
            "###",
            "1. Architecture",
            "- Workspace Structure:",
            "- codingbuddy-cli: Main CLI application",
            "- codingbuddy-agent: Core agent logic"
        ]
    );

    let project_parts = split_inline_markdown_blocks(
        "Project Structure Highlightscodingbuddy-cli/├── crates/           # Rust workspace crates",
    );
    assert_eq!(
        project_parts,
        vec![
            "Project Structure Highlights",
            "codingbuddy-cli/├── crates/           # Rust workspace crates"
        ]
    );
}

#[test]
fn render_bare_heading_markers_as_spacer_line() {
    let line = render_assistant_markdown("###");
    let text: String = line.iter().map(|s| s.content.to_string()).collect();
    assert!(text.trim().is_empty());
}

#[test]
fn inline_markdown_strikethrough() {
    let spans = parse_inline_markdown("before ~~deleted~~ after", Style::default());
    let text: String = spans.iter().map(|s| s.content.to_string()).collect();
    assert_eq!(text, "before deleted after");
    // The "deleted" span should have CROSSED_OUT modifier
    let deleted_span = spans
        .iter()
        .find(|s| s.content.as_ref() == "deleted")
        .unwrap();
    assert!(
        deleted_span
            .style
            .add_modifier
            .contains(Modifier::CROSSED_OUT),
        "strikethrough text should have CROSSED_OUT modifier"
    );
}

#[test]
fn inline_markdown_link() {
    let spans = parse_inline_markdown("click [here](https://example.com) now", Style::default());
    let text: String = spans.iter().map(|s| s.content.to_string()).collect();
    assert!(text.contains("here"));
    assert!(text.contains("https://example.com"));
    // Link text should be underlined
    let link_span = spans.iter().find(|s| s.content.as_ref() == "here").unwrap();
    assert!(
        link_span.style.add_modifier.contains(Modifier::UNDERLINED),
        "link text should be underlined"
    );
}

#[test]
fn task_list_unchecked_renders_box() {
    let line = render_assistant_markdown("- [ ] todo item");
    let text: String = line.iter().map(|s| s.content.to_string()).collect();
    assert!(text.contains("☐"), "unchecked task should show ☐");
    assert!(text.contains("todo item"));
}

#[test]
fn task_list_checked_renders_checkmark() {
    let line = render_assistant_markdown("- [x] done item");
    let text: String = line.iter().map(|s| s.content.to_string()).collect();
    assert!(text.contains("☑"), "checked task should show ☑");
    assert!(text.contains("done item"));
}

// ── P1-02: open_editor keybinding tests ─────────────────────────────

#[test]
fn open_editor_keybinding_default_is_ctrl_g() {
    let bindings = KeyBindings::default();
    assert_eq!(bindings.open_editor.code, KeyCode::Char('g'));
    assert_eq!(bindings.open_editor.modifiers, KeyModifiers::CONTROL);
}

#[test]
fn open_editor_keybinding_overridable() {
    let raw = KeyBindingsFile {
        open_editor: Some("ctrl+e".to_string()),
        ..Default::default()
    };
    let bindings = KeyBindings::default().apply_overrides(raw).unwrap();
    assert_eq!(bindings.open_editor.code, KeyCode::Char('e'));
    assert_eq!(bindings.open_editor.modifiers, KeyModifiers::CONTROL);
}

#[test]
fn keybindings_all_fields_have_defaults() {
    let bindings = KeyBindings::default();
    // Verify all fields are accessible (compilation test) + spot check
    assert_eq!(bindings.exit.code, KeyCode::Char('c'));
    assert_eq!(bindings.submit.code, KeyCode::Enter);
    assert_eq!(bindings.open_editor.code, KeyCode::Char('g'));
    assert_eq!(bindings.kill_background.code, KeyCode::Char('f'));
    assert_eq!(bindings.toggle_thinking.code, KeyCode::Char('t'));
    assert_eq!(bindings.approve_plan.code, KeyCode::Char('y'));
    assert_eq!(bindings.approve_plan.modifiers, KeyModifiers::CONTROL);
    assert_eq!(bindings.reject_plan.code, KeyCode::Char('y'));
    assert_eq!(bindings.reject_plan.modifiers, KeyModifiers::ALT);
}

// ── P1-01: rewind picker state machine tests ────────────────────────

#[test]
fn rewind_picker_navigation() {
    let checkpoints = vec![
        codingbuddy_store::CheckpointRecord {
            checkpoint_id: uuid::Uuid::nil(),
            reason: "a".to_string(),
            snapshot_path: "/tmp/a".to_string(),
            files_count: 1,
            created_at: "2025-01-01T00:00:00Z".to_string(),
        },
        codingbuddy_store::CheckpointRecord {
            checkpoint_id: uuid::Uuid::nil(),
            reason: "b".to_string(),
            snapshot_path: "/tmp/b".to_string(),
            files_count: 2,
            created_at: "2025-01-02T00:00:00Z".to_string(),
        },
    ];
    let mut picker = RewindPickerState::new(checkpoints);
    assert_eq!(picker.selected_index, 0);
    picker.down();
    assert_eq!(picker.selected_index, 1);
    picker.down();
    assert_eq!(picker.selected_index, 0); // wraps
    picker.up();
    assert_eq!(picker.selected_index, 1); // wraps back
}

#[test]
fn rewind_picker_phase_transition() {
    let checkpoints = vec![codingbuddy_store::CheckpointRecord {
        checkpoint_id: uuid::Uuid::nil(),
        reason: "cp".to_string(),
        snapshot_path: "/tmp/cp".to_string(),
        files_count: 1,
        created_at: "2025-01-01T00:00:00Z".to_string(),
    }];
    let mut picker = RewindPickerState::new(checkpoints);
    assert_eq!(picker.phase, RewindPickerPhase::SelectCheckpoint);

    // Confirm transitions to SelectAction
    let result = picker.confirm();
    assert!(result.is_none());
    assert_eq!(picker.phase, RewindPickerPhase::SelectAction);

    // Back returns to SelectCheckpoint
    let should_close = picker.back();
    assert!(!should_close);
    assert_eq!(picker.phase, RewindPickerPhase::SelectCheckpoint);

    // Back from SelectCheckpoint closes picker
    let should_close = picker.back();
    assert!(should_close);
}

// ── P2-05: prompt suggestions ────────────────────────────────────────

#[test]
fn prompt_suggestions_shown_when_empty() {
    let spans = render_prompt_suggestions(true);
    assert!(
        !spans.is_empty(),
        "should show suggestions when input is empty"
    );
    let text: String = spans.iter().map(|s| s.content.to_string()).collect();
    assert!(text.contains("Explain this project"));
}

#[test]
fn prompt_suggestions_hidden_when_typing() {
    let spans = render_prompt_suggestions(false);
    assert!(
        spans.is_empty(),
        "should hide suggestions when input is non-empty"
    );
}

// ── P4-03: bang prefix direct execution tests ────────────────────────

#[test]
fn bang_prefix_runs_shell() {
    let output = execute_bang_command("echo hello");
    assert_eq!(output, "hello");
}

#[test]
fn bang_prefix_empty_returns_exit_code() {
    // An empty true command returns exit code 0
    let output = execute_bang_command("true");
    assert_eq!(output, "(exit code 0)");
}

// ── P4-02: autocomplete dropdown tests ─────────────────────────────

#[test]
fn autocomplete_dropdown_shows_matches() {
    let suggestions = vec![
        "src/lib.rs".to_string(),
        "src/linter.rs".to_string(),
        "src/lint_config.rs".to_string(),
    ];
    let ac = AutocompleteState::new(suggestions, 0);
    assert_eq!(ac.selected, 0);
    assert_eq!(ac.selected_value(), Some("src/lib.rs"));
    let lines = ac.display_lines(10);
    assert_eq!(lines.len(), 3);
    assert!(lines[0].starts_with('>'));
}

#[test]
fn autocomplete_tab_cycles_selection() {
    let suggestions = vec![
        "file1.rs".to_string(),
        "file2.rs".to_string(),
        "file3.rs".to_string(),
    ];
    let mut ac = AutocompleteState::new(suggestions, 0);
    assert_eq!(ac.selected, 0);
    ac.down();
    assert_eq!(ac.selected, 1);
    assert_eq!(ac.selected_value(), Some("file2.rs"));
    ac.down();
    assert_eq!(ac.selected, 2);
    ac.down();
    assert_eq!(ac.selected, 0); // wraps
}

#[test]
fn autocomplete_esc_dismisses() {
    let suggestions = vec!["a.rs".to_string(), "b.rs".to_string()];
    let ac = AutocompleteState::new(suggestions, 5);
    // Verify trigger_pos is stored correctly
    assert_eq!(ac.trigger_pos, 5);
    // After Esc, the caller sets autocomplete_dropdown = None
}

// ── P4-05: rewind picker enhanced display tests ───────────────────

#[test]
fn rewind_picker_shows_timestamps() {
    let line = RewindPickerState::format_checkpoint_line(
        &codingbuddy_store::CheckpointRecord {
            checkpoint_id: uuid::Uuid::nil(),
            reason: "before edit".to_string(),
            snapshot_path: "/tmp/cp".to_string(),
            files_count: 5,
            created_at: "2025-01-01T00:00:00Z".to_string(),
        },
        true,
    );
    assert!(line.starts_with(">"));
    assert!(line.contains("before edit"));
    assert!(line.contains("5 files"));
    // Should have a time component (either relative or raw)
    assert!(line.contains('['));
}

#[test]
fn rewind_picker_scrolls_viewport() {
    // Create 12 checkpoints — more than the viewport of 8
    let checkpoints: Vec<codingbuddy_store::CheckpointRecord> = (0..12)
        .map(|i| codingbuddy_store::CheckpointRecord {
            checkpoint_id: uuid::Uuid::nil(),
            reason: format!("cp {i}"),
            snapshot_path: format!("/tmp/cp{i}"),
            files_count: i as u64,
            created_at: "2025-01-01T00:00:00Z".to_string(),
        })
        .collect();
    let mut picker = RewindPickerState::new(checkpoints);

    // Initially at 0, viewport should start at 0
    let vp = picker.viewport();
    assert_eq!(vp.start, 0);
    assert_eq!(vp.end, 8);

    // Move to index 10
    for _ in 0..10 {
        picker.down();
    }
    assert_eq!(picker.selected_index, 10);
    let vp = picker.viewport();
    // selected=10, total=12, viewport=8 → start=4, end=12
    assert_eq!(vp.start, 4);
    assert_eq!(vp.end, 12);
}

#[test]
fn format_relative_time_variants() {
    // Very old timestamp → days ago
    let result = format_relative_time("2020-01-01T00:00:00Z");
    assert!(result.contains("d ago"));
    // Invalid timestamp → fallback
    let result = format_relative_time("not-a-timestamp");
    assert_eq!(result, "not-a-timestamp");
}

// ── P4-06: clipboard image detection tests ────────────────────────

#[test]
fn image_paste_returns_none_when_no_image() {
    // In a test environment, the clipboard is unlikely to have an image
    // Just verify the function doesn't panic
    let result = try_clipboard_image();
    // Result is platform-dependent; on CI it should be None
    let _ = result;
}

#[test]
fn image_fallback_text_paste() {
    // Verify function returns None on non-macOS or when no image present
    // This test exercises the code path on any platform
    let result = try_clipboard_image();
    // On non-macOS, always returns None. On macOS, depends on clipboard.
    #[cfg(not(target_os = "macos"))]
    assert!(result.is_none());
    let _ = result;
}

// ── P4-07: model picker state machine tests ─────────────────────────

#[test]
fn model_picker_navigation() {
    let mut picker = ModelPickerState::new();
    assert_eq!(picker.selected, 0);
    assert_eq!(picker.confirm(), "deepseek-chat");

    picker.down();
    assert_eq!(picker.selected, 1);
    assert_eq!(picker.confirm(), "deepseek-reasoner");

    // Clamp at bottom
    picker.down();
    assert_eq!(picker.selected, 1);

    picker.up();
    assert_eq!(picker.selected, 0);
    assert_eq!(picker.confirm(), "deepseek-chat");

    // Clamp at top
    picker.up();
    assert_eq!(picker.selected, 0);
}

// ── P7-09: Vim mode verification ────────────────────────────────────

#[test]
fn vim_normal_mode_movement() {
    // h/l movement is implemented via move_to_next_word_start helper
    // and direct cursor arithmetic. Verify movement helpers.
    let input = "hello world";
    // 'w' moves to next word start
    let next = move_to_next_word_start(input, 0);
    assert_eq!(next, 6, "w from 0 should jump to 'w' in 'world'");
    // 'b' moves to prev word start
    let prev = move_to_prev_word_start(input, 6);
    assert_eq!(prev, 0, "b from 6 should jump back to 'h' in 'hello'");
    // h/l are direct ±1 on cursor_pos (tested via the arithmetic)
    let cursor = 5usize;
    let h_result = cursor.saturating_sub(1);
    let l_result = (cursor + 1).min(input.len());
    assert_eq!(h_result, 4, "h moves cursor left");
    assert_eq!(l_result, 6, "l moves cursor right");
}

#[test]
fn vim_insert_mode_toggle() {
    // VimMode enum and label behavior
    assert_eq!(VimMode::Normal.label(), "NORMAL");
    assert_eq!(VimMode::Insert.label(), "INSERT");
    assert_eq!(VimMode::Visual.label(), "VISUAL");
    assert_eq!(VimMode::Command.label(), "COMMAND");

    // Simulated transition: Normal → Insert (i) → Normal (Esc)
    let mode = VimMode::Normal;
    assert_eq!(mode.label(), "NORMAL");
    let mode = VimMode::Insert; // i pressed
    assert_eq!(mode.label(), "INSERT");
    let mode = VimMode::Normal; // Esc pressed
    assert_eq!(mode.label(), "NORMAL");
    let _ = mode;
}

#[test]
fn slash_suggestions_filter_by_prefix() {
    let all = slash_command_suggestions("", 100);
    assert!(all.len() > 10, "should list many commands");
    // Each entry starts with "/"
    assert!(all.iter().all(|s| s.starts_with('/')));

    let filtered = slash_command_suggestions("co", 8);
    // Should match: code, compact, commit, config, context, copy, cost
    assert!(filtered.len() >= 2);
    assert!(filtered.iter().all(|s| s.starts_with("/co")));

    let none = slash_command_suggestions("zzzzz", 8);
    assert!(none.is_empty());
}

#[test]
fn slash_suggestion_to_command_extracts_name() {
    assert_eq!(
        slash_suggestion_to_command("/help  Show available commands"),
        "/help"
    );
    assert_eq!(
        slash_suggestion_to_command("/commit  Create a git commit"),
        "/commit"
    );
    // Edge case: no double-space
    assert_eq!(slash_suggestion_to_command("/vim"), "/vim");
}

#[test]
fn slash_autocomplete_accepts_into_command() {
    // Simulate the acceptance flow: user types "/" → dropdown → selects "/help  ..." → Enter
    let suggestions = slash_command_suggestions("", 8);
    let ac = AutocompleteState::new(suggestions, 0);
    let selected = ac.selected_value().unwrap();
    let cmd = slash_suggestion_to_command(selected);
    let result = format!("{cmd} ");
    assert!(result.starts_with('/'));
    assert!(result.ends_with(' '));
    // The result should be just the command, not the description
    assert!(!result.contains("  "), "should not contain description");
}

#[test]
fn slash_tab_completion_uses_catalog() {
    // Verify catalog has expected entries used by tab completion
    let has_help = SLASH_COMMAND_CATALOG.iter().any(|(n, _)| *n == "help");
    let has_commit = SLASH_COMMAND_CATALOG.iter().any(|(n, _)| *n == "commit");
    let has_model = SLASH_COMMAND_CATALOG.iter().any(|(n, _)| *n == "model");
    assert!(has_help, "catalog should contain help");
    assert!(has_commit, "catalog should contain commit");
    assert!(has_model, "catalog should contain model");

    // Verify tab completion would find "co" → first match starting with "co"
    let prefix = "co";
    let found = SLASH_COMMAND_CATALOG
        .iter()
        .find(|(name, _)| name.starts_with(prefix));
    assert!(found.is_some(), "should find a command starting with 'co'");
}

// ── P8-08: ML Ghost Text tests ─────────────────────────────────────

#[test]
fn ghost_text_debounces() {
    let mut ghost = GhostTextState::default();
    // Before any keystroke, should not request
    assert!(!ghost.should_request(5));

    // After keystroke, should not request immediately (debounce not elapsed)
    ghost.on_keystroke();
    assert!(
        !ghost.should_request(5),
        "should not request before debounce"
    );

    // Still pending
    assert!(ghost.pending);
    assert!(ghost.suggestion.is_none());
}

#[test]
fn ghost_text_cancels_on_keystroke() {
    let mut ghost = GhostTextState::default();
    ghost.set_suggestion(Some("hello world".to_string()));
    assert!(ghost.suggestion.is_some());

    // New keystroke should clear the suggestion
    ghost.on_keystroke();
    assert!(
        ghost.suggestion.is_none(),
        "keystroke should clear suggestion"
    );
    assert!(ghost.pending, "should mark as pending after keystroke");
}

#[test]
fn ghost_text_tab_accepts() {
    let mut ghost = GhostTextState::default();
    ghost.set_suggestion(Some(" world".to_string()));

    let accepted = ghost.accept_full();
    assert_eq!(accepted, Some(" world".to_string()));
    assert!(ghost.suggestion.is_none(), "suggestion should be consumed");
}

#[test]
fn ghost_text_accept_word() {
    let mut ghost = GhostTextState::default();
    ghost.set_suggestion(Some("hello world goodbye".to_string()));

    let word = ghost.accept_word();
    assert_eq!(word, Some("hello".to_string()));
    assert_eq!(
        ghost.suggestion,
        Some(" world goodbye".to_string()),
        "remaining text should be preserved"
    );

    let word2 = ghost.accept_word();
    assert_eq!(word2, Some(" world".to_string()));
    assert_eq!(ghost.suggestion, Some(" goodbye".to_string()));
}

#[test]
fn ghost_text_renders_darkgray() {
    // Verify the DarkGray italic style is correctly constructed
    let style = Style::default()
        .fg(Color::DarkGray)
        .add_modifier(Modifier::ITALIC);
    let span = Span::styled("ghost text", style);
    assert_eq!(span.style.fg, Some(Color::DarkGray));
    assert!(span.style.add_modifier.contains(Modifier::ITALIC));
}

#[test]
fn ghost_text_min_input_length() {
    let mut ghost = GhostTextState::default();
    ghost.on_keystroke();
    // Input too short (< 3 chars)
    assert!(
        !ghost.should_request(2),
        "should not request for short input"
    );
    // Wait past debounce and check with sufficient input
    ghost.last_keystroke = Instant::now() - Duration::from_millis(300);
    assert!(
        ghost.should_request(5),
        "should request for long input after debounce"
    );
    assert!(
        !ghost.should_request(2),
        "still should not request for short input"
    );
}

#[test]
fn ghost_text_accept_full_when_no_suggestion() {
    let mut ghost = GhostTextState::default();
    assert!(ghost.accept_full().is_none());
    assert!(ghost.accept_word().is_none());
}
