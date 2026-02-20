use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};
use ratatui::{Terminal, TerminalOptions, Viewport};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

/// Events sent from the background agent thread to the TUI event loop.
pub enum TuiStreamEvent {
    /// Incremental content text from the LLM.
    ContentDelta(String),
    /// Incremental reasoning/thinking text from the LLM.
    ReasoningDelta(String),
    /// A tool is now actively executing.
    ToolActive(String),
    /// A tool call has started — pushed to transcript.
    ToolCallStart {
        tool_name: String,
        args_summary: String,
    },
    /// A tool call has completed — pushed to transcript with duration.
    ToolCallEnd {
        tool_name: String,
        duration_ms: u64,
        summary: String,
    },
    /// The agent needs user approval before proceeding.
    ApprovalNeeded {
        tool_name: String,
        args_summary: String,
        response_tx: mpsc::Sender<bool>,
    },
    /// An error occurred during agent execution.
    Error(String),
    /// Agent execution completed with the given output.
    Done(String),
}

/// RAII guard that restores the terminal on drop (including panics).
struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        // Show cursor in case it was hidden
        let _ = crossterm::execute!(io::stdout(), crossterm::cursor::Show);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SlashCommand {
    Help,
    Init,
    Clear,
    Compact,
    Memory(Vec<String>),
    Config,
    Model(Option<String>),
    Cost,
    Mcp(Vec<String>),
    Rewind(Vec<String>),
    Export(Vec<String>),
    Plan,
    Teleport(Vec<String>),
    RemoteEnv(Vec<String>),
    Status,
    Effort(Option<String>),
    Skills(Vec<String>),
    Permissions(Vec<String>),
    Background(Vec<String>),
    Visual(Vec<String>),
    Context,
    Sandbox(Vec<String>),
    Agents,
    Tasks(Vec<String>),
    Review(Vec<String>),
    Search(Vec<String>),
    Vim(Vec<String>),
    TerminalSetup,
    Keybindings,
    Doctor,
    Unknown { name: String, args: Vec<String> },
}

impl SlashCommand {
    pub fn parse(input: &str) -> Option<Self> {
        let line = input.trim();
        if !line.starts_with('/') {
            return None;
        }
        let mut parts = line[1..].split_whitespace();
        let name = parts.next()?.to_ascii_lowercase();
        let args = parts.map(ToString::to_string).collect::<Vec<_>>();

        let cmd = match name.as_str() {
            "help" => Self::Help,
            "init" => Self::Init,
            "clear" => Self::Clear,
            "compact" => Self::Compact,
            "memory" => Self::Memory(args),
            "config" => Self::Config,
            "model" => Self::Model(args.first().cloned()),
            "cost" => Self::Cost,
            "mcp" => Self::Mcp(args),
            "rewind" => Self::Rewind(args),
            "export" => Self::Export(args),
            "plan" => Self::Plan,
            "teleport" => Self::Teleport(args),
            "remote-env" => Self::RemoteEnv(args),
            "status" => Self::Status,
            "effort" => Self::Effort(args.first().cloned()),
            "skills" => Self::Skills(args),
            "permissions" => Self::Permissions(args),
            "background" => Self::Background(args),
            "visual" => Self::Visual(args),
            "context" => Self::Context,
            "sandbox" => Self::Sandbox(args),
            "agents" => Self::Agents,
            "tasks" => Self::Tasks(args),
            "review" => Self::Review(args),
            "search" => Self::Search(args),
            "vim" => Self::Vim(args),
            "terminal-setup" => Self::TerminalSetup,
            "keybindings" => Self::Keybindings,
            "doctor" => Self::Doctor,
            other => Self::Unknown {
                name: other.to_string(),
                args,
            },
        };
        Some(cmd)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UiStatus {
    pub model: String,
    pub pending_approvals: usize,
    pub estimated_cost_usd: f64,
    pub background_jobs: usize,
    pub autopilot_running: bool,
    pub permission_mode: String,
    pub active_tasks: usize,
    #[serde(default)]
    pub context_used_tokens: u64,
    #[serde(default = "default_context_max")]
    pub context_max_tokens: u64,
    #[serde(default)]
    pub session_turns: usize,
    #[serde(default)]
    pub working_directory: String,
}

fn default_context_max() -> u64 {
    128_000
}

pub fn render_statusline(status: &UiStatus) -> String {
    let mode_indicator = match status.permission_mode.as_str() {
        "auto" => "[AUTO]",
        "locked" => "[LOCKED]",
        _ => "[ASK]",
    };
    let tasks_part = if status.active_tasks > 0 {
        format!(" tasks={}", status.active_tasks)
    } else {
        String::new()
    };
    let ctx_pct = if status.context_max_tokens > 0 {
        (status.context_used_tokens as f64 / status.context_max_tokens as f64 * 100.0) as u64
    } else {
        0
    };
    let ctx_part = if status.context_max_tokens > 0 {
        format!(
            " ctx={}K/{}K({}%)",
            status.context_used_tokens / 1000,
            status.context_max_tokens / 1000,
            ctx_pct
        )
    } else {
        String::new()
    };
    format!(
        "model={} {} approvals={} jobs={}{} autopilot={}{} cost=${:.4}",
        status.model,
        mode_indicator,
        status.pending_approvals,
        status.background_jobs,
        tasks_part,
        if status.autopilot_running {
            "running"
        } else {
            "idle"
        },
        ctx_part,
        status.estimated_cost_usd,
    )
}

#[cfg(test)]
fn render_statusline_spans(
    status: &UiStatus,
    active_tool: Option<&str>,
    spinner_frame: &str,
    scroll_pct: Option<usize>,
    vim_mode_label: Option<&str>,
    has_new_content_below: bool,
) -> Vec<Span<'static>> {
    let mode_color = match status.permission_mode.as_str() {
        "auto" => Color::Green,
        "locked" => Color::Red,
        _ => Color::Yellow,
    };
    let mode_label = match status.permission_mode.as_str() {
        "auto" => " AUTO ",
        "locked" => " LOCKED ",
        _ => " ASK ",
    };

    let mut spans = Vec::new();

    // Active tool with spinner (left side)
    if let Some(tool) = active_tool {
        spans.push(Span::styled(
            format!(" {} {} ", spinner_frame, tool),
            Style::default().fg(Color::Yellow),
        ));
    }

    spans.push(Span::styled(
        format!(" {} ", status.model),
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    ));
    spans.push(Span::raw(" "));
    spans.push(Span::styled(
        mode_label.to_string(),
        Style::default()
            .fg(Color::Black)
            .bg(mode_color)
            .add_modifier(Modifier::BOLD),
    ));

    if status.pending_approvals > 0 {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            format!(" {} pending ", status.pending_approvals),
            Style::default()
                .fg(Color::Black)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
    }

    if status.active_tasks > 0 {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            format!(" {} tasks ", status.active_tasks),
            Style::default().fg(Color::Magenta),
        ));
    }

    if status.background_jobs > 0 {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            format!(" {} jobs ", status.background_jobs),
            Style::default().fg(Color::Blue),
        ));
    }

    if status.autopilot_running {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            " AUTOPILOT ".to_string(),
            Style::default()
                .fg(Color::Black)
                .bg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ));
    }

    if status.context_max_tokens > 0 {
        let pct =
            (status.context_used_tokens as f64 / status.context_max_tokens as f64 * 100.0) as u64;
        let ctx_color = if pct > 80 {
            Color::Red
        } else if pct > 60 {
            Color::Yellow
        } else {
            Color::Green
        };
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            format!(
                " {}K/{}K ",
                status.context_used_tokens / 1000,
                status.context_max_tokens / 1000
            ),
            Style::default().fg(ctx_color),
        ));
    }

    if status.session_turns > 0 {
        spans.push(Span::styled(
            format!(" turn {} ", status.session_turns),
            Style::default().fg(Color::DarkGray),
        ));
    }

    spans.push(Span::styled(
        format!(" ${:.4} ", status.estimated_cost_usd),
        Style::default().fg(Color::DarkGray),
    ));

    // Vim mode badge
    if let Some(mode) = vim_mode_label {
        spans.push(Span::styled(
            format!(" {} ", mode),
            Style::default()
                .fg(Color::Black)
                .bg(Color::Gray)
                .add_modifier(Modifier::BOLD),
        ));
    }

    // Scroll position
    if let Some(pct) = scroll_pct {
        spans.push(Span::styled(
            format!(" {}% ", pct),
            Style::default().fg(Color::DarkGray),
        ));
    }

    // New content below indicator
    if has_new_content_below {
        spans.push(Span::styled(
            " \u{2193} new ".to_string(),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
    }

    spans
}

/// Check if a word is a keyword in common programming languages.
fn syntax_keyword_color(word: &str) -> Option<Color> {
    match word {
        // Rust keywords
        "fn" | "let" | "mut" | "const" | "static" | "struct" | "enum" | "impl" | "trait"
        | "pub" | "mod" | "use" | "crate" | "super" | "self" | "Self" | "match" | "if" | "else"
        | "loop" | "while" | "for" | "in" | "break" | "continue" | "return" | "async" | "await"
        | "move" | "where" | "type" | "unsafe" | "dyn" | "ref" | "as" | "extern" => {
            Some(Color::Magenta)
        }
        // Python keywords
        "def" | "class" | "import" | "from" | "try" | "except" | "finally" | "with" | "yield"
        | "lambda" | "pass" | "raise" | "global" | "nonlocal" | "elif" | "del" | "assert"
        | "is" | "not" | "and" | "or" => Some(Color::Magenta),
        // JS/TS keywords
        "function" | "var" | "new" | "this" | "typeof" | "instanceof" | "throw" | "catch"
        | "switch" | "case" | "default" | "export" | "interface" | "extends" | "implements"
        | "abstract" | "override" => Some(Color::Magenta),
        // Common types
        "String" | "Vec" | "Option" | "Result" | "Box" | "Arc" | "Rc" | "HashMap" | "HashSet"
        | "bool" | "i8" | "i16" | "i32" | "i64" | "i128" | "u8" | "u16" | "u32" | "u64"
        | "u128" | "f32" | "f64" | "usize" | "isize" | "str" | "char" | "int" | "float"
        | "dict" | "list" | "tuple" | "set" | "bytes" | "number" | "string" | "boolean"
        | "void" | "any" | "never" => Some(Color::Yellow),
        // Literals
        "true" | "false" | "True" | "False" | "None" | "null" | "undefined" | "nil" | "Ok"
        | "Err" | "Some" => Some(Color::Cyan),
        _ => None,
    }
}

/// Apply syntax highlighting to a line of code inside a code block.
fn highlight_code_line(line: &str) -> Line<'static> {
    let mut spans = Vec::new();
    let mut chars = line.char_indices().peekable();
    let mut last = 0;

    while let Some(&(i, ch)) = chars.peek() {
        // String literals
        if ch == '"' || ch == '\'' {
            if i > last {
                spans.push(Span::styled(
                    line[last..i].to_string(),
                    Style::default().fg(Color::White),
                ));
            }
            let quote = ch;
            chars.next();
            let start = i;
            while let Some(&(j, c)) = chars.peek() {
                chars.next();
                if c == quote && (j == 0 || line.as_bytes().get(j - 1) != Some(&b'\\')) {
                    break;
                }
            }
            let end = chars.peek().map_or(line.len(), |&(j, _)| j);
            spans.push(Span::styled(
                line[start..end].to_string(),
                Style::default().fg(Color::Green),
            ));
            last = end;
            continue;
        }
        // Line comments
        if ch == '/' && line.get(i + 1..i + 2) == Some("/") {
            if i > last {
                spans.push(Span::styled(
                    line[last..i].to_string(),
                    Style::default().fg(Color::White),
                ));
            }
            spans.push(Span::styled(
                line[i..].to_string(),
                Style::default().fg(Color::DarkGray),
            ));
            last = line.len();
            break;
        }
        // Hash comments (Python, shell)
        if ch == '#' {
            if i > last {
                spans.push(Span::styled(
                    line[last..i].to_string(),
                    Style::default().fg(Color::White),
                ));
            }
            spans.push(Span::styled(
                line[i..].to_string(),
                Style::default().fg(Color::DarkGray),
            ));
            last = line.len();
            break;
        }
        // Numbers
        if ch.is_ascii_digit() && (i == 0 || !line.as_bytes()[i - 1].is_ascii_alphanumeric()) {
            if i > last {
                spans.push(Span::styled(
                    line[last..i].to_string(),
                    Style::default().fg(Color::White),
                ));
            }
            let start = i;
            while let Some(&(_, c)) = chars.peek() {
                if !c.is_ascii_digit() && c != '.' && c != 'x' && c != '_' {
                    break;
                }
                chars.next();
            }
            let end = chars.peek().map_or(line.len(), |&(j, _)| j);
            spans.push(Span::styled(
                line[start..end].to_string(),
                Style::default().fg(Color::LightCyan),
            ));
            last = end;
            continue;
        }
        // Identifiers / keywords
        if ch.is_ascii_alphabetic() || ch == '_' {
            if i > last {
                spans.push(Span::styled(
                    line[last..i].to_string(),
                    Style::default().fg(Color::White),
                ));
            }
            let start = i;
            while let Some(&(_, c)) = chars.peek() {
                if !c.is_ascii_alphanumeric() && c != '_' {
                    break;
                }
                chars.next();
            }
            let end = chars.peek().map_or(line.len(), |&(j, _)| j);
            let word = &line[start..end];
            if let Some(color) = syntax_keyword_color(word) {
                spans.push(Span::styled(
                    word.to_string(),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ));
            } else {
                spans.push(Span::styled(
                    word.to_string(),
                    Style::default().fg(Color::White),
                ));
            }
            last = end;
            continue;
        }
        chars.next();
    }
    if last < line.len() {
        spans.push(Span::styled(
            line[last..].to_string(),
            Style::default().fg(Color::White),
        ));
    }
    if spans.is_empty() {
        spans.push(Span::styled(
            line.to_string(),
            Style::default().fg(Color::White),
        ));
    }
    Line::from(spans)
}

/// Parse inline markdown (`**bold**`, `` `code` ``) and return styled spans.
fn parse_inline_markdown(text: &str, base_style: Style) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i: usize = 0;
    let mut seg_start: usize = 0;

    while i < len {
        // Only inspect ASCII marker bytes; multi-byte chars are never `*` or `` ` ``.
        if bytes[i] == b'`' {
            // Inline code: `...`
            if let Some(end) = text[i + 1..].find('`') {
                if i > seg_start {
                    spans.push(Span::styled(text[seg_start..i].to_string(), base_style));
                }
                spans.push(Span::styled(
                    text[i + 1..i + 1 + end].to_string(),
                    Style::default().fg(Color::Yellow),
                ));
                i = i + 1 + end + 1;
                seg_start = i;
                continue;
            }
        } else if bytes[i] == b'*' && i + 1 < len && bytes[i + 1] == b'*' {
            // Bold: **...**
            if let Some(end) = text[i + 2..].find("**") {
                if i > seg_start {
                    spans.push(Span::styled(text[seg_start..i].to_string(), base_style));
                }
                // Recursively parse nested inline code within bold.
                spans.extend(parse_inline_markdown(
                    &text[i + 2..i + 2 + end],
                    base_style.add_modifier(Modifier::BOLD),
                ));
                i = i + 2 + end + 2;
                seg_start = i;
                continue;
            }
        } else if bytes[i] == b'*' {
            // Italic: *...*
            if let Some(end) = text[i + 1..].find('*') {
                // Ensure it's not empty (**)
                if end > 0 {
                    if i > seg_start {
                        spans.push(Span::styled(text[seg_start..i].to_string(), base_style));
                    }
                    spans.extend(parse_inline_markdown(
                        &text[i + 1..i + 1 + end],
                        base_style.add_modifier(Modifier::ITALIC),
                    ));
                    i = i + 1 + end + 1;
                    seg_start = i;
                    continue;
                }
            }
        }
        // Advance by one character (handle multi-byte safely).
        i += text[i..].chars().next().map_or(1, |c| c.len_utf8());
    }

    if seg_start < len {
        spans.push(Span::styled(text[seg_start..].to_string(), base_style));
    }
    if spans.is_empty() {
        spans.push(Span::styled(text.to_string(), base_style));
    }
    spans
}

/// Render a single line of assistant markdown (headings, lists, blockquotes,
/// horizontal rules, inline formatting). Used by both first-line and
/// continuation-line renderers.
fn render_assistant_markdown(text: &str) -> Line<'static> {
    let body_style = Style::default().fg(Color::White);

    // Code fence
    if text.starts_with("```") {
        return Line::from(vec![Span::styled(
            text.to_string(),
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM),
        )]);
    }

    // Horizontal rule
    if text == "---" || text == "***" || text == "___" {
        return Line::from(vec![Span::styled(
            "  ─────────────────────────────────────────".to_string(),
            Style::default().fg(Color::DarkGray),
        )]);
    }

    // Headings — strip `#` prefix and render with level-appropriate styling.
    // Handles `# H1`, `## H2`, `### H3`, `#### H4+` (with or without space).
    if text.starts_with("# ") || (text.starts_with('#') && !text.starts_with("#!")) {
        let trimmed = text.trim_start_matches('#');
        let level = text.len() - trimmed.len();
        // Accept both "## Heading" and "##Heading" (no space)
        let heading_text = trimmed.strip_prefix(' ').unwrap_or(trimmed);
        if !heading_text.is_empty() {
            let heading_style = match level {
                1 => Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
                2 => Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
                3 => Style::default().fg(Color::Cyan),
                _ => Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            };
            let mut spans = vec![Span::styled("  ".to_string(), Style::default())];
            spans.extend(parse_inline_markdown(heading_text, heading_style));
            return Line::from(spans);
        }
    }

    // Blockquote
    if let Some(quote) = text.strip_prefix("> ") {
        let dim = Style::default().fg(Color::DarkGray);
        let mut spans = vec![Span::styled("  │ ".to_string(), dim)];
        spans.extend(parse_inline_markdown(quote, dim));
        return Line::from(spans);
    }

    // Unordered list
    if let Some(item) = text.strip_prefix("- ").or_else(|| text.strip_prefix("* ")) {
        let mut spans = vec![Span::styled(
            "  • ".to_string(),
            Style::default().fg(Color::Cyan),
        )];
        spans.extend(parse_inline_markdown(item, body_style));
        return Line::from(spans);
    }

    // Numbered list (e.g. "1. ", "12. ")
    if let Some(dot_pos) = text.find(". ")
        && text[..dot_pos].chars().all(|c| c.is_ascii_digit())
        && dot_pos <= 4
    {
        let num = &text[..dot_pos + 2]; // "1. "
        let item = &text[dot_pos + 2..];
        let mut spans = vec![Span::styled(
            format!("  {num}"),
            Style::default().fg(Color::Cyan),
        )];
        spans.extend(parse_inline_markdown(item, body_style));
        return Line::from(spans);
    }

    // Table rows: lines starting and ending with `|`
    if text.starts_with('|') && text.ends_with('|') {
        let trimmed = &text[1..text.len() - 1];
        // Separator row (e.g. |---|---|)
        if trimmed
            .chars()
            .all(|c| c == '-' || c == '|' || c == ':' || c == ' ')
        {
            return Line::from(vec![Span::styled(
                format!("  {text}"),
                Style::default().fg(Color::DarkGray),
            )]);
        }
        // Data/header row
        let cells: Vec<&str> = trimmed.split('|').map(|c| c.trim()).collect();
        let mut spans = vec![Span::styled(
            "  │".to_string(),
            Style::default().fg(Color::DarkGray),
        )];
        for (i, cell) in cells.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled(
                    "│".to_string(),
                    Style::default().fg(Color::DarkGray),
                ));
            }
            spans.push(Span::styled(
                format!(" {cell} "),
                Style::default().fg(Color::White),
            ));
        }
        spans.push(Span::styled(
            "│".to_string(),
            Style::default().fg(Color::DarkGray),
        ));
        return Line::from(spans);
    }

    // Default: indented body with inline markdown
    let mut spans = vec![Span::styled("  ".to_string(), Style::default())];
    spans.extend(parse_inline_markdown(text, body_style));
    Line::from(spans)
}

fn style_transcript_line(entry: &TranscriptEntry, in_code_block: bool) -> Line<'static> {
    let (prefix, prefix_style, body_style) = match entry.kind {
        MessageKind::User => (
            "❯ ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
            Style::default().fg(Color::White),
        ),
        MessageKind::Assistant => ("  ", Style::default(), Style::default().fg(Color::White)),
        MessageKind::System => (
            "⚙ ",
            Style::default().fg(Color::DarkGray),
            Style::default().fg(Color::DarkGray),
        ),
        MessageKind::ToolCall => (
            "⚡ ",
            Style::default().fg(Color::Yellow),
            Style::default().fg(Color::Yellow),
        ),
        MessageKind::ToolResult => (
            "  ↳ ",
            Style::default().fg(Color::Green),
            Style::default().fg(Color::DarkGray),
        ),
        MessageKind::Error => (
            "✗ ",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            Style::default().fg(Color::Red),
        ),
    };

    let text = &entry.text;

    // Apply syntax highlighting inside code blocks
    if in_code_block && entry.kind == MessageKind::Assistant && !text.starts_with("```") {
        return highlight_code_line(text);
    }

    // Full markdown rendering for assistant messages
    if entry.kind == MessageKind::Assistant {
        return render_assistant_markdown(text);
    }

    Line::from(vec![
        Span::styled(prefix.to_string(), prefix_style),
        Span::styled(text.clone(), body_style),
    ])
}

/// Style a continuation line (2nd+ line of a multi-line entry). Same body
/// styling as the entry kind but no prefix character, just indentation.
fn style_continuation_line(entry: &TranscriptEntry, in_code_block: bool) -> Line<'static> {
    let body_style = match entry.kind {
        MessageKind::User => Style::default().fg(Color::White),
        MessageKind::Assistant => Style::default().fg(Color::White),
        MessageKind::System => Style::default().fg(Color::DarkGray),
        MessageKind::ToolCall => Style::default().fg(Color::Yellow),
        MessageKind::ToolResult => Style::default().fg(Color::DarkGray),
        MessageKind::Error => Style::default().fg(Color::Red),
    };
    let text = &entry.text;
    if in_code_block && entry.kind == MessageKind::Assistant && !text.starts_with("```") {
        return highlight_code_line(text);
    }
    // Full markdown rendering for assistant continuation lines too
    if entry.kind == MessageKind::Assistant {
        return render_assistant_markdown(text);
    }
    // Indent continuation to align with body text after the prefix
    Line::from(vec![Span::styled(format!("  {text}"), body_style)])
}

/// Flush new transcript entries above the inline viewport into native terminal scrollback.
fn flush_transcript_above(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    shell: &ChatShell,
    last_printed_idx: &mut usize,
) -> Result<()> {
    if *last_printed_idx >= shell.transcript.len() {
        return Ok(());
    }
    let new_entries = &shell.transcript[*last_printed_idx..];
    // Build styled lines for the new entries
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut in_code_block = false;
    // Scan earlier entries to figure out if we're inside a code block
    for entry in &shell.transcript[..*last_printed_idx] {
        for sub in entry.text.split('\n') {
            if entry.kind == MessageKind::Assistant && sub.starts_with("```") {
                in_code_block = !in_code_block;
            }
        }
    }
    for entry in new_entries {
        for (i, sub_text) in entry.text.split('\n').enumerate() {
            let sub_entry = TranscriptEntry {
                kind: entry.kind,
                text: sub_text.to_string(),
            };
            if sub_entry.kind == MessageKind::Assistant && sub_entry.text.starts_with("```") {
                in_code_block = !in_code_block;
            }
            if i == 0 {
                lines.push(style_transcript_line(&sub_entry, in_code_block));
            } else {
                lines.push(style_continuation_line(&sub_entry, in_code_block));
            }
        }
    }
    *last_printed_idx = shell.transcript.len();
    if lines.is_empty() {
        return Ok(());
    }
    let height = lines.len() as u16;
    terminal.insert_before(height, |buf| {
        let area = buf.area;
        for (i, line) in lines.iter().enumerate() {
            if i as u16 >= area.height {
                break;
            }
            let line_area = Rect::new(area.x, area.y + i as u16, area.width, 1);
            Paragraph::new(line.clone()).render(line_area, buf);
        }
    })?;
    Ok(())
}

/// Flush complete lines from the streaming buffer into native scrollback.
/// Returns the remaining partial line (text after the last `\n`).
fn flush_streaming_lines(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    buffer: &mut String,
) -> Result<()> {
    // Find the last newline — everything before it consists of complete lines.
    let Some(last_nl) = buffer.rfind('\n') else {
        return Ok(());
    };
    let complete = buffer[..last_nl].to_string();
    let remaining = buffer[last_nl + 1..].to_string();
    *buffer = remaining;

    let mut in_code_block = false;
    let lines_text: Vec<&str> = complete.split('\n').collect();
    let mut styled_lines: Vec<Line<'static>> = Vec::with_capacity(lines_text.len());
    for line_text in &lines_text {
        let entry = TranscriptEntry {
            kind: MessageKind::Assistant,
            text: line_text.to_string(),
        };
        if entry.text.starts_with("```") {
            in_code_block = !in_code_block;
        }
        styled_lines.push(style_transcript_line(&entry, in_code_block));
    }
    if styled_lines.is_empty() {
        return Ok(());
    }
    let height = styled_lines.len() as u16;
    terminal.insert_before(height, |buf| {
        let area = buf.area;
        for (i, line) in styled_lines.iter().enumerate() {
            if i as u16 >= area.height {
                break;
            }
            let line_area = Rect::new(area.x, area.y + i as u16, area.width, 1);
            Paragraph::new(line.clone()).render(line_area, buf);
        }
    })?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageKind {
    User,
    Assistant,
    System,
    ToolCall,
    ToolResult,
    Error,
}

#[derive(Debug, Clone)]
pub struct TranscriptEntry {
    pub kind: MessageKind,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RightPane {
    Plan,
    Tools,
    MissionControl,
    Artifacts,
}

#[cfg(test)]
impl RightPane {
    fn cycle(self) -> Self {
        match self {
            Self::Plan => Self::Tools,
            Self::Tools => Self::MissionControl,
            Self::MissionControl => Self::Artifacts,
            Self::Artifacts => Self::Plan,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VimMode {
    Insert,
    Normal,
    Visual,
    Command,
}

impl VimMode {
    fn label(self) -> &'static str {
        match self {
            Self::Insert => "INSERT",
            Self::Normal => "NORMAL",
            Self::Visual => "VISUAL",
            Self::Command => "COMMAND",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChatShell {
    pub transcript: Vec<TranscriptEntry>,
    pub plan_lines: Vec<String>,
    pub tool_lines: Vec<String>,
    pub mission_control_lines: Vec<String>,
    pub artifact_lines: Vec<String>,
    pub active_tool: Option<String>,
    pub spinner_tick: usize,
}

impl ChatShell {
    pub fn push_transcript(&mut self, line: impl Into<String>) {
        self.transcript.push(TranscriptEntry {
            kind: MessageKind::Assistant,
            text: line.into(),
        });
    }

    pub fn push_user(&mut self, line: impl Into<String>) {
        self.transcript.push(TranscriptEntry {
            kind: MessageKind::User,
            text: line.into(),
        });
    }

    pub fn push_system(&mut self, line: impl Into<String>) {
        self.transcript.push(TranscriptEntry {
            kind: MessageKind::System,
            text: line.into(),
        });
    }

    pub fn push_error(&mut self, line: impl Into<String>) {
        self.transcript.push(TranscriptEntry {
            kind: MessageKind::Error,
            text: line.into(),
        });
    }

    pub fn push_tool_call(&mut self, tool_name: &str, args_summary: &str) {
        self.transcript.push(TranscriptEntry {
            kind: MessageKind::ToolCall,
            text: format!("{tool_name} {args_summary}"),
        });
    }

    pub fn push_tool_result(&mut self, tool_name: &str, duration_ms: u64, summary: &str) {
        let duration_str = if duration_ms >= 1000 {
            format!("{:.1}s", duration_ms as f64 / 1000.0)
        } else {
            format!("{duration_ms}ms")
        };
        self.transcript.push(TranscriptEntry {
            kind: MessageKind::ToolResult,
            text: format!("{tool_name} ({duration_str}) {summary}"),
        });
    }

    pub fn push_plan(&mut self, line: impl Into<String>) {
        self.plan_lines.push(line.into());
    }

    pub fn push_tool(&mut self, line: impl Into<String>) {
        self.tool_lines.push(line.into());
    }

    pub fn push_mission_control(&mut self, line: impl Into<String>) {
        self.mission_control_lines.push(line.into());
    }

    pub fn push_artifact(&mut self, line: impl Into<String>) {
        self.artifact_lines.push(line.into());
    }

    /// Append text to the current streaming assistant response.
    /// Creates a new assistant entry if none is in progress.
    pub fn append_streaming(&mut self, text: &str) {
        if let Some(last) = self.transcript.last_mut()
            && last.kind == MessageKind::Assistant
        {
            last.text.push_str(text);
            return;
        }
        self.transcript.push(TranscriptEntry {
            kind: MessageKind::Assistant,
            text: text.to_string(),
        });
    }

    /// Finalize the streaming response — ensure the complete output is in the transcript.
    /// If streaming deltas were received, replaces the partial entry with the final output.
    /// If no streaming happened, pushes the full output as new transcript entries.
    pub fn finalize_streaming(&mut self, final_output: &str) {
        if final_output.is_empty() {
            return;
        }
        // Remove any partial streaming assistant entry built by append_streaming.
        if self
            .transcript
            .last()
            .is_some_and(|e| e.kind == MessageKind::Assistant)
        {
            self.transcript.pop();
        }
        // Push the complete output, one entry per line for proper rendering.
        for line in final_output.lines() {
            self.transcript.push(TranscriptEntry {
                kind: MessageKind::Assistant,
                text: line.to_string(),
            });
        }
    }

    /// Format and push a `/cost` command response into the transcript.
    pub fn push_cost_summary(&mut self, status: &UiStatus) {
        let ctx_pct = if status.context_max_tokens > 0 {
            (status.context_used_tokens as f64 / status.context_max_tokens as f64 * 100.0) as u64
        } else {
            0
        };
        self.push_system(format!(
            "Cost: ${:.4} | Tokens: {}K/{}K ({}%) | Turns: {}",
            status.estimated_cost_usd,
            status.context_used_tokens / 1000,
            status.context_max_tokens / 1000,
            ctx_pct,
            status.session_turns,
        ));
    }

    /// Format and push a `/status` command response into the transcript.
    pub fn push_status_summary(&mut self, status: &UiStatus) {
        self.push_system(format!("Model: {}", status.model));
        self.push_system(format!("Permission mode: {}", status.permission_mode));
        self.push_system(format!("Pending approvals: {}", status.pending_approvals));
        self.push_system(format!("Active tasks: {}", status.active_tasks));
        self.push_system(format!("Background jobs: {}", status.background_jobs));
        self.push_system(format!(
            "Autopilot: {}",
            if status.autopilot_running {
                "running"
            } else {
                "idle"
            }
        ));
        self.push_cost_summary(status);
    }

    /// Format and push a `/model` command response into the transcript.
    pub fn push_model_info(&mut self, model: &str) {
        self.push_system(format!("Current model: {model}"));
    }

    fn spinner_frame(&self) -> &'static str {
        const FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        FRAMES[self.spinner_tick % FRAMES.len()]
    }
}

#[derive(Debug, Clone)]
pub struct KeyBindings {
    pub exit: KeyEvent,
    pub submit: KeyEvent,
    pub newline: KeyEvent,
    pub stop: KeyEvent,
    pub rewind_menu: KeyEvent,
    pub autocomplete: KeyEvent,
    pub background: KeyEvent,
    pub toggle_raw: KeyEvent,
    pub history_prev: KeyEvent,
    pub paste_hint: KeyEvent,
    pub toggle_mission_control: KeyEvent,
    pub toggle_artifacts: KeyEvent,
    pub toggle_plan_collapse: KeyEvent,
    pub cycle_permission_mode: KeyEvent,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct KeyBindingsFile {
    exit: Option<String>,
    submit: Option<String>,
    newline: Option<String>,
    stop: Option<String>,
    rewind_menu: Option<String>,
    autocomplete: Option<String>,
    background: Option<String>,
    toggle_raw: Option<String>,
    history_prev: Option<String>,
    paste_hint: Option<String>,
    toggle_mission_control: Option<String>,
    toggle_artifacts: Option<String>,
    toggle_plan_collapse: Option<String>,
    cycle_permission_mode: Option<String>,
}

impl Default for KeyBindings {
    fn default() -> Self {
        Self {
            exit: KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
            submit: KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            newline: KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT),
            stop: KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
            rewind_menu: KeyEvent::new(KeyCode::Esc, KeyModifiers::SHIFT),
            autocomplete: KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
            background: KeyEvent::new(KeyCode::Char('b'), KeyModifiers::CONTROL),
            toggle_raw: KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
            history_prev: KeyEvent::new(KeyCode::Up, KeyModifiers::NONE),
            paste_hint: KeyEvent::new(KeyCode::Char('v'), KeyModifiers::CONTROL),
            toggle_mission_control: KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL),
            toggle_artifacts: KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL),
            toggle_plan_collapse: KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL),
            cycle_permission_mode: KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT),
        }
    }
}

impl KeyBindings {
    fn apply_overrides(mut self, raw: KeyBindingsFile) -> Result<Self> {
        if let Some(value) = raw.exit {
            self.exit = parse_key_event(&value)?;
        }
        if let Some(value) = raw.submit {
            self.submit = parse_key_event(&value)?;
        }
        if let Some(value) = raw.newline {
            self.newline = parse_key_event(&value)?;
        }
        if let Some(value) = raw.stop {
            self.stop = parse_key_event(&value)?;
        }
        if let Some(value) = raw.rewind_menu {
            self.rewind_menu = parse_key_event(&value)?;
        }
        if let Some(value) = raw.autocomplete {
            self.autocomplete = parse_key_event(&value)?;
        }
        if let Some(value) = raw.background {
            self.background = parse_key_event(&value)?;
        }
        if let Some(value) = raw.toggle_raw {
            self.toggle_raw = parse_key_event(&value)?;
        }
        if let Some(value) = raw.history_prev {
            self.history_prev = parse_key_event(&value)?;
        }
        if let Some(value) = raw.paste_hint {
            self.paste_hint = parse_key_event(&value)?;
        }
        if let Some(value) = raw.toggle_mission_control {
            self.toggle_mission_control = parse_key_event(&value)?;
        }
        if let Some(value) = raw.toggle_artifacts {
            self.toggle_artifacts = parse_key_event(&value)?;
        }
        if let Some(value) = raw.toggle_plan_collapse {
            self.toggle_plan_collapse = parse_key_event(&value)?;
        }
        if let Some(value) = raw.cycle_permission_mode {
            self.cycle_permission_mode = parse_key_event(&value)?;
        }
        Ok(self)
    }
}

pub fn load_keybindings(path: &Path) -> Result<KeyBindings> {
    let raw = fs::read_to_string(path)?;
    let parsed: KeyBindingsFile = serde_json::from_str(&raw)?;
    KeyBindings::default().apply_overrides(parsed)
}

fn parse_theme_color(name: &str) -> Color {
    match name.to_ascii_lowercase().as_str() {
        "black" => Color::Black,
        "red" => Color::Red,
        "green" => Color::Green,
        "yellow" => Color::Yellow,
        "blue" => Color::Blue,
        "magenta" => Color::Magenta,
        "cyan" => Color::Cyan,
        "white" => Color::White,
        "gray" | "grey" => Color::Gray,
        "darkgray" | "darkgrey" => Color::DarkGray,
        "lightred" => Color::LightRed,
        "lightgreen" => Color::LightGreen,
        "lightyellow" => Color::LightYellow,
        "lightblue" => Color::LightBlue,
        "lightmagenta" => Color::LightMagenta,
        "lightcyan" => Color::LightCyan,
        _ => Color::Cyan,
    }
}

#[derive(Debug, Clone)]
pub struct TuiTheme {
    pub primary: Color,
    pub secondary: Color,
    pub error: Color,
}

impl Default for TuiTheme {
    fn default() -> Self {
        Self {
            primary: Color::Cyan,
            secondary: Color::Yellow,
            error: Color::Red,
        }
    }
}

impl TuiTheme {
    pub fn from_config(primary: &str, secondary: &str, error: &str) -> Self {
        Self {
            primary: parse_theme_color(primary),
            secondary: parse_theme_color(secondary),
            error: parse_theme_color(error),
        }
    }
}

pub fn load_artifact_lines(workspace: &Path) -> Vec<String> {
    let artifacts_dir = workspace.join(".deepseek").join("artifacts");
    let mut lines = Vec::new();
    if !artifacts_dir.exists() {
        lines.push("No artifacts found.".to_string());
        lines.push(format!("Directory: {}", artifacts_dir.display()));
        return lines;
    }
    let mut entries: Vec<_> = fs::read_dir(&artifacts_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    entries.sort_by_key(|e| e.file_name());
    if entries.is_empty() {
        lines.push("No task artifacts found.".to_string());
        return lines;
    }
    for entry in entries {
        let task_dir = entry.path();
        let task_id = entry.file_name().to_string_lossy().to_string();
        lines.push(format!("## Task: {task_id}"));
        for name in &["plan.md", "diff.patch", "verification.md"] {
            let file_path = task_dir.join(name);
            if file_path.exists() {
                let size = fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);
                lines.push(format!("  {name} ({size} bytes)"));
                // Show first few lines as preview
                if let Ok(content) = fs::read_to_string(&file_path) {
                    for (i, line) in content.lines().take(5).enumerate() {
                        lines.push(format!("    {}", line));
                        if i == 4 {
                            lines.push("    ...".to_string());
                        }
                    }
                }
            }
        }
        // Also show any other files in the task directory
        if let Ok(files) = fs::read_dir(&task_dir) {
            for file in files.filter_map(|f| f.ok()) {
                let fname = file.file_name().to_string_lossy().to_string();
                if !["plan.md", "diff.patch", "verification.md"].contains(&fname.as_str()) {
                    let size = fs::metadata(file.path()).map(|m| m.len()).unwrap_or(0);
                    lines.push(format!("  {fname} ({size} bytes)"));
                }
            }
        }
        lines.push(String::new());
    }
    lines
}

pub fn run_tui_shell<F>(status: UiStatus, mut on_submit: F) -> Result<()>
where
    F: FnMut(&str) -> Result<String>,
{
    let (tx, rx) = mpsc::channel();
    run_tui_shell_with_bindings(
        status,
        KeyBindings::default(),
        TuiTheme::default(),
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
    )
}

pub fn run_tui_shell_with_bindings<F, S>(
    mut status: UiStatus,
    bindings: KeyBindings,
    _theme: TuiTheme,
    stream_rx: mpsc::Receiver<TuiStreamEvent>,
    mut on_submit: F,
    mut refresh_status: S,
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
        // Clear screen + scrollback + move cursor home
        out.write_all(b"\x1b[2J\x1b[3J\x1b[H")?;

        // ASCII art logo + info, styled like Claude Code
        let version = env!("CARGO_PKG_VERSION");
        let model = &status.model;
        let cwd = &status.working_directory;

        // Shark logo in blue, info text in white/gray
        writeln!(out, "")?;
        writeln!(out, "\x1b[1;34m        ▄       \x1b[0m \x1b[1mDeepSeek CLI\x1b[0m v{version}")?;
        writeln!(out, "\x1b[1;34m    ▗▄▀ ● ▀▀▀▀▄ \x1b[0m \x1b[36m{model}\x1b[0m")?;
        writeln!(out, "\x1b[1;34m    ▝▀▄▄▄▄▄▄▀▀▘ \x1b[0m \x1b[90m{cwd}\x1b[0m")?;
        writeln!(out, "\x1b[1;34m        ▀        \x1b[0m")?;
        writeln!(out, "")?;
        out.flush()?;
    }

    enable_raw_mode()?;
    let _guard = TerminalGuard;
    let stdout = io::stdout();
    let backend = CrosstermBackend::new(stdout);
    // Inline viewport: only the bottom 5 lines are managed by ratatui
    // (streaming partial line + separator + input + separator + status).
    // Everything above is native terminal scrollback.
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(5),
        },
    )?;

    let mut shell = ChatShell::default();
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
    let mut is_processing = false;
    let mut pending_approval: Option<(String, String, mpsc::Sender<bool>)> = None;
    let mut cancelled = false;
    // Track the last transcript length so we only print new entries via insert_before.
    let mut last_printed_idx: usize = 0;
    // Buffer for streaming content — complete lines get flushed to scrollback,
    // the partial (incomplete) last line is rendered in the viewport.
    let mut streaming_buffer = String::new();

    loop {
        tick_count = tick_count.wrapping_add(1);
        shell.spinner_tick = tick_count;
        cursor_visible = tick_count % 16 < 8;

        // Print any new transcript entries above the inline viewport
        // so they go into native terminal scrollback.
        flush_transcript_above(&mut terminal, &shell, &mut last_printed_idx)?;
        // Flush complete lines from the streaming buffer into scrollback.
        flush_streaming_lines(&mut terminal, &mut streaming_buffer)?;

        terminal.draw(|frame| {
            let area = frame.area();
            // Inline viewport is 5 lines:
            //   Row 0: streaming partial line (current incomplete line being received)
            //   Row 1: separator ─────
            //   Row 2: input prompt "> ..."
            //   Row 3: separator ─────
            //   Row 4: status bar
            let width = area.width;

            // Row 0: current streaming partial line (or blank when idle)
            let stream_area = Rect::new(area.x, area.y, width, 1);
            if !streaming_buffer.is_empty() {
                let partial_entry = TranscriptEntry {
                    kind: MessageKind::Assistant,
                    text: streaming_buffer.clone(),
                };
                let styled = render_assistant_markdown(&partial_entry.text);
                frame.render_widget(Paragraph::new(styled), stream_area);
            } else {
                frame.render_widget(Paragraph::new(Span::raw("")), stream_area);
            }

            // Row 1: thin separator line
            let sep_area = Rect::new(area.x, area.y + 1, width, 1);
            frame.render_widget(
                Paragraph::new(Span::styled(
                    "\u{2500}".repeat(width as usize),
                    Style::default().fg(Color::DarkGray),
                )),
                sep_area,
            );

            // Row 2: input prompt
            let input_area = Rect::new(area.x, area.y + 2, width, 1);
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
            let input_text = if vim_enabled && vim_mode == VimMode::Command {
                let cursor_ch = if cursor_visible { "\u{2588}" } else { " " };
                format!(":{}{}", vim_command_buffer, cursor_ch)
            } else {
                let before = &input[..cursor_pos.min(input.len())];
                let after = &input[cursor_pos.min(input.len())..];
                let cursor_ch = if cursor_visible { "\u{2588}" } else { " " };
                format!("{}{}{}", before, cursor_ch, after)
            };
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(prompt_str.to_string(), prompt_style),
                    Span::raw(input_text),
                ])),
                input_area,
            );

            // Row 3: thin separator line below input
            let sep2_area = Rect::new(area.x, area.y + 3, width, 1);
            frame.render_widget(
                Paragraph::new(Span::styled(
                    "\u{2500}".repeat(width as usize),
                    Style::default().fg(Color::DarkGray),
                )),
                sep2_area,
            );

            // Row 4: status bar (compact inline — model, active tool spinner, info)
            let status_area = Rect::new(area.x, area.y + 4, width, 1);
            let mut status_spans: Vec<Span<'static>> = Vec::new();
            if let Some(ref tool) = shell.active_tool {
                status_spans.push(Span::styled(
                    format!(" {} {} ", shell.spinner_frame(), tool),
                    Style::default().fg(Color::Yellow),
                ));
                status_spans.push(Span::raw(" "));
            }
            status_spans.push(Span::styled(
                format!(" {} ", status.model),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ));
            let mode_color = match status.permission_mode.as_str() {
                "auto" => Color::Green,
                "locked" => Color::Red,
                _ => Color::Yellow,
            };
            let mode_label = match status.permission_mode.as_str() {
                "auto" => " AUTO ",
                "locked" => " LOCKED ",
                _ => " ASK ",
            };
            status_spans.push(Span::styled(
                mode_label.to_string(),
                Style::default()
                    .fg(Color::Black)
                    .bg(mode_color)
                    .add_modifier(Modifier::BOLD),
            ));
            if vim_enabled {
                status_spans.push(Span::styled(
                    format!(" {} ", vim_mode.label()),
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Gray)
                        .add_modifier(Modifier::BOLD),
                ));
            }
            status_spans.push(Span::styled(
                format!(" {}", info_line),
                Style::default().fg(Color::DarkGray),
            ));
            frame.render_widget(Paragraph::new(Line::from(status_spans)), status_area);
        })?;

        // Drain streaming events from background agent thread.
        while let Ok(ev) = stream_rx.try_recv() {
            if cancelled {
                match ev {
                    TuiStreamEvent::Done(_) | TuiStreamEvent::Error(_) => {
                        cancelled = false;
                    }
                    TuiStreamEvent::ApprovalNeeded { response_tx, .. } => {
                        let _ = response_tx.send(false);
                    }
                    _ => {}
                }
                continue;
            }
            match ev {
                TuiStreamEvent::ContentDelta(text) => {
                    // Buffer streaming text — complete lines will be flushed
                    // to scrollback on next loop iteration, partial line shown
                    // in the viewport.
                    streaming_buffer.push_str(&text);
                    shell.append_streaming(&text);
                }
                TuiStreamEvent::ReasoningDelta(text) => {
                    shell.push_tool(format!("[thinking] {text}"));
                }
                TuiStreamEvent::ToolActive(name) => {
                    shell.active_tool = Some(name);
                }
                TuiStreamEvent::ToolCallStart {
                    tool_name,
                    args_summary,
                } => {
                    shell.push_tool_call(&tool_name, &args_summary);
                    shell.active_tool = Some(tool_name);
                }
                TuiStreamEvent::ToolCallEnd {
                    tool_name,
                    duration_ms,
                    summary,
                } => {
                    shell.push_tool_result(&tool_name, duration_ms, &summary);
                    shell.active_tool = None;
                }
                TuiStreamEvent::ApprovalNeeded {
                    tool_name,
                    args_summary,
                    response_tx,
                } => {
                    pending_approval = Some((tool_name, args_summary, response_tx));
                }
                TuiStreamEvent::Error(msg) => {
                    streaming_buffer.clear();
                    shell.push_error(&msg);
                    info_line = format!("error: {msg}");
                    is_processing = false;
                    shell.active_tool = None;
                }
                TuiStreamEvent::Done(output) => {
                    // Flush any remaining partial streaming line to scrollback.
                    if !streaming_buffer.is_empty() {
                        let remaining = std::mem::take(&mut streaming_buffer);
                        let entry = TranscriptEntry {
                            kind: MessageKind::Assistant,
                            text: remaining,
                        };
                        let styled = render_assistant_markdown(&entry.text);
                        let _ = terminal.insert_before(1, |buf| {
                            let a = buf.area;
                            Paragraph::new(styled).render(a, buf);
                        });
                    }
                    shell.finalize_streaming(&output);
                    // Skip re-printing entries that were already streamed.
                    last_printed_idx = shell.transcript.len();
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
            info_line = format!("approve `{tool_name}` {args_summary}? [y/N]");
        }

        // Check for external SIGINT (e.g. `kill -INT`).
        if sigint_flag.swap(false, Ordering::Relaxed) {
            if is_processing {
                is_processing = false;
                cancelled = true;
                streaming_buffer.clear();
                shell.active_tool = None;
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
        let input_event = event::read()?;
        if let Event::Paste(pasted) = &input_event {
            input.insert_str(cursor_pos.min(input.len()), pasted);
            cursor_pos = (cursor_pos + pasted.len()).min(input.len());
            info_line = "pasted input".to_string();
            continue;
        }
        let Event::Key(mut key) = input_event else {
            continue;
        };
        // Only handle key press events (ignore release/repeat on platforms that send them)
        if key.kind != KeyEventKind::Press {
            continue;
        }

        if key == bindings.exit {
            if is_processing {
                is_processing = false;
                cancelled = true;
                streaming_buffer.clear();
                shell.active_tool = None;
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

        let mut vim_quit_after_submit = false;
        if vim_enabled {
            let mut vim_consumed = false;
            match vim_mode {
                VimMode::Insert => {
                    if key.code == KeyCode::Esc {
                        vim_mode = VimMode::Normal;
                        vim_pending_operator = None;
                        info_line = "-- NORMAL --".to_string();
                        vim_consumed = true;
                    }
                }
                VimMode::Normal => {
                    match key.code {
                        KeyCode::Char('i') => {
                            vim_mode = VimMode::Insert;
                            vim_pending_operator = None;
                            info_line = "-- INSERT --".to_string();
                        }
                        KeyCode::Char('a') => {
                            cursor_pos = (cursor_pos + 1).min(input.len());
                            vim_mode = VimMode::Insert;
                            vim_pending_operator = None;
                            info_line = "-- INSERT --".to_string();
                        }
                        KeyCode::Char('v') => {
                            vim_mode = VimMode::Visual;
                            vim_visual_anchor = Some(cursor_pos);
                            vim_pending_operator = None;
                            info_line = "-- VISUAL --".to_string();
                        }
                        KeyCode::Char(':') => {
                            vim_mode = VimMode::Command;
                            vim_command_buffer.clear();
                            vim_pending_operator = None;
                            info_line = ":".to_string();
                        }
                        KeyCode::Char('h') | KeyCode::Left => {
                            cursor_pos = cursor_pos.saturating_sub(1);
                            vim_pending_operator = None;
                        }
                        KeyCode::Char('l') | KeyCode::Right => {
                            cursor_pos = (cursor_pos + 1).min(input.len());
                            vim_pending_operator = None;
                        }
                        KeyCode::Char('0') => {
                            cursor_pos = 0;
                            vim_pending_operator = None;
                        }
                        KeyCode::Char('$') => {
                            cursor_pos = input.len();
                            vim_pending_operator = None;
                        }
                        KeyCode::Char('w') => {
                            cursor_pos = move_to_next_word_start(&input, cursor_pos);
                            vim_pending_operator = None;
                        }
                        KeyCode::Char('b') => {
                            cursor_pos = move_to_prev_word_start(&input, cursor_pos);
                            vim_pending_operator = None;
                        }
                        KeyCode::Char('e') => {
                            cursor_pos = move_to_word_end(&input, cursor_pos);
                            vim_pending_operator = None;
                        }
                        KeyCode::Char('x') => {
                            if cursor_pos < input.len() {
                                input.remove(cursor_pos);
                            }
                            vim_pending_operator = None;
                        }
                        KeyCode::Char('p') => {
                            if !vim_yank_buffer.is_empty() {
                                let insert_at = (cursor_pos + 1).min(input.len());
                                input.insert_str(insert_at, &vim_yank_buffer);
                                cursor_pos = (insert_at + vim_yank_buffer.len()).min(input.len());
                            }
                            vim_pending_operator = None;
                        }
                        KeyCode::Char('d') => {
                            if vim_pending_operator == Some('d') {
                                input.clear();
                                cursor_pos = 0;
                                vim_pending_operator = None;
                                info_line = "deleted line".to_string();
                            } else {
                                vim_pending_operator = Some('d');
                                info_line = "d".to_string();
                            }
                        }
                        KeyCode::Char('c') => {
                            if vim_pending_operator == Some('c') {
                                input.clear();
                                cursor_pos = 0;
                                vim_mode = VimMode::Insert;
                                vim_pending_operator = None;
                                info_line = "-- INSERT --".to_string();
                            } else {
                                vim_pending_operator = Some('c');
                                info_line = "c".to_string();
                            }
                        }
                        KeyCode::Char('y') => {
                            if vim_pending_operator == Some('y') {
                                vim_yank_buffer = input.clone();
                                vim_pending_operator = None;
                                info_line = "yanked line".to_string();
                            } else {
                                vim_pending_operator = Some('y');
                                info_line = "y".to_string();
                            }
                        }
                        KeyCode::Enter => {
                            key = bindings.submit;
                            vim_pending_operator = None;
                        }
                        KeyCode::Esc => {
                            vim_pending_operator = None;
                        }
                        _ => {
                            vim_pending_operator = None;
                        }
                    }
                    vim_consumed = true;
                }
                VimMode::Visual => {
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
                            info_line = ":".to_string();
                        }
                        _ => {}
                    }
                    vim_consumed = true;
                }
                VimMode::Command => {
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
                "ask" => "auto",
                "auto" => "locked",
                _ => "ask",
            };
            status.permission_mode = next.to_string();
            info_line = format!("permission mode: {} -> {}", current, next);
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
            shell.active_tool = Some("processing...".to_string());
            on_submit(&background_cmd);
            continue;
        }
        if key == bindings.paste_hint {
            info_line = "paste is supported via terminal bracketed paste".to_string();
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
        if key == bindings.autocomplete {
            if input.starts_with('/') {
                let prefix = input.trim_start_matches('/').to_ascii_lowercase();
                let commands = [
                    "help",
                    "init",
                    "clear",
                    "compact",
                    "memory",
                    "config",
                    "model",
                    "cost",
                    "mcp",
                    "rewind",
                    "export",
                    "plan",
                    "teleport",
                    "remote-env",
                    "status",
                    "effort",
                    "skills",
                    "permissions",
                    "background",
                    "visual",
                    "context",
                    "sandbox",
                    "agents",
                    "tasks",
                    "review",
                    "search",
                    "vim",
                    "terminal-setup",
                    "keybindings",
                    "doctor",
                ];
                if let Some(next) = commands.iter().find(|cmd| cmd.starts_with(&prefix)) {
                    input = format!("/{next}");
                    cursor_pos = input.len();
                }
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
            shell.active_tool = Some("processing...".to_string());
            on_submit(&prompt);
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
                }
            }
            KeyCode::Delete => {
                if cursor_pos < input.len() {
                    input.remove(cursor_pos);
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

fn parse_key_event(value: &str) -> Result<KeyEvent> {
    let mut modifiers = KeyModifiers::NONE;
    let mut key_code: Option<KeyCode> = None;
    for token in value
        .split('+')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let normalized = token.to_ascii_lowercase();
        match normalized.as_str() {
            "ctrl" | "control" => modifiers |= KeyModifiers::CONTROL,
            "shift" => modifiers |= KeyModifiers::SHIFT,
            "alt" | "option" => modifiers |= KeyModifiers::ALT,
            other => {
                key_code = Some(
                    parse_key_code(other)
                        .ok_or_else(|| anyhow::anyhow!("unsupported keybinding token: {token}"))?,
                );
            }
        }
    }
    let code = key_code.ok_or_else(|| anyhow::anyhow!("missing key code in keybinding"))?;
    Ok(KeyEvent::new(code, modifiers))
}

fn parse_key_code(value: &str) -> Option<KeyCode> {
    match value {
        "enter" => Some(KeyCode::Enter),
        "esc" | "escape" => Some(KeyCode::Esc),
        "tab" => Some(KeyCode::Tab),
        "up" => Some(KeyCode::Up),
        "down" => Some(KeyCode::Down),
        "left" => Some(KeyCode::Left),
        "right" => Some(KeyCode::Right),
        "backspace" => Some(KeyCode::Backspace),
        "space" => Some(KeyCode::Char(' ')),
        value if value.chars().count() == 1 => value.chars().next().map(KeyCode::Char),
        _ => None,
    }
}

fn autocomplete_path_input(input: &str) -> Option<String> {
    let split_at = input
        .char_indices()
        .rfind(|(_, ch)| ch.is_whitespace())
        .map(|(idx, _)| idx + 1)
        .unwrap_or(0);
    let token = input[split_at..].trim();
    if token.is_empty() {
        return None;
    }
    let completed = autocomplete_path_token(token)?;
    let mut out = String::with_capacity(input.len() + completed.len() + 2);
    out.push_str(&input[..split_at]);
    out.push_str(&completed);
    Some(out)
}

fn autocomplete_path_token(token: &str) -> Option<String> {
    let (display_token, lookup_token) = if token == "~" || token.starts_with("~/") {
        let home = std::env::var("HOME")
            .ok()
            .or_else(|| std::env::var("USERPROFILE").ok())?;
        let rest = token.trim_start_matches("~/");
        (token.to_string(), PathBuf::from(home).join(rest))
    } else {
        (token.to_string(), PathBuf::from(token))
    };

    let cwd = std::env::current_dir().ok()?;
    let absolute_lookup = if lookup_token.is_absolute() {
        lookup_token
    } else {
        cwd.join(lookup_token)
    };
    let parent = absolute_lookup.parent()?.to_path_buf();
    let prefix = absolute_lookup
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let mut matches = Vec::new();
    for entry in fs::read_dir(parent).ok()?.filter_map(|entry| entry.ok()) {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with(prefix) {
            continue;
        }
        let mut candidate = name.to_string();
        if entry.path().is_dir() {
            candidate.push('/');
        }
        matches.push(candidate);
    }
    matches.sort();
    matches.dedup();
    if matches.len() != 1 {
        return None;
    }
    let replacement = matches.into_iter().next()?;
    let cut = display_token.len().saturating_sub(prefix.len());
    let base = &display_token[..cut];
    Some(format!("{base}{replacement}"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VimSlashCommand {
    Toggle,
    On,
    Off,
    SetMode(VimMode),
}

fn parse_vim_slash_command(prompt: &str) -> Option<Result<VimSlashCommand, &'static str>> {
    let trimmed = prompt.trim();
    if !trimmed.starts_with("/vim") {
        return None;
    }
    let parts = trimmed.split_whitespace().collect::<Vec<_>>();
    if parts.len() == 1 {
        return Some(Ok(VimSlashCommand::Toggle));
    }
    let arg = parts
        .get(1)
        .copied()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let cmd = match arg.as_str() {
        "on" | "enable" => VimSlashCommand::On,
        "off" | "disable" => VimSlashCommand::Off,
        "normal" => VimSlashCommand::SetMode(VimMode::Normal),
        "insert" => VimSlashCommand::SetMode(VimMode::Insert),
        "visual" => VimSlashCommand::SetMode(VimMode::Visual),
        "command" => VimSlashCommand::SetMode(VimMode::Command),
        _ => {
            return Some(Err("usage: /vim [on|off|normal|insert|visual|command]"));
        }
    };
    Some(Ok(cmd))
}

fn apply_vim_command(
    command: VimSlashCommand,
    vim_enabled: &mut bool,
    vim_mode: &mut VimMode,
    vim_command_buffer: &mut String,
    vim_visual_anchor: &mut Option<usize>,
    vim_pending_operator: &mut Option<char>,
) {
    match command {
        VimSlashCommand::Toggle => {
            *vim_enabled = !*vim_enabled;
            *vim_mode = if *vim_enabled {
                VimMode::Normal
            } else {
                VimMode::Insert
            };
        }
        VimSlashCommand::On => {
            *vim_enabled = true;
            *vim_mode = VimMode::Normal;
        }
        VimSlashCommand::Off => {
            *vim_enabled = false;
            *vim_mode = VimMode::Insert;
        }
        VimSlashCommand::SetMode(mode) => {
            *vim_enabled = true;
            *vim_mode = mode;
        }
    }
    vim_command_buffer.clear();
    *vim_visual_anchor = None;
    *vim_pending_operator = None;
}

fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn move_to_next_word_start(input: &str, cursor_pos: usize) -> usize {
    let bytes = input.as_bytes();
    let mut pos = cursor_pos.min(bytes.len());
    if pos < bytes.len() && is_word_byte(bytes[pos]) {
        while pos < bytes.len() && is_word_byte(bytes[pos]) {
            pos += 1;
        }
    }
    while pos < bytes.len() && !is_word_byte(bytes[pos]) {
        pos += 1;
    }
    pos
}

fn move_to_prev_word_start(input: &str, cursor_pos: usize) -> usize {
    let bytes = input.as_bytes();
    if bytes.is_empty() {
        return 0;
    }
    let mut pos = cursor_pos.min(bytes.len());
    pos = pos.saturating_sub(1);
    while pos > 0 && !is_word_byte(bytes[pos]) {
        pos -= 1;
    }
    while pos > 0 && is_word_byte(bytes[pos - 1]) {
        pos -= 1;
    }
    pos
}

fn move_to_word_end(input: &str, cursor_pos: usize) -> usize {
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut pos = cursor_pos.min(len);
    if pos >= len {
        return len;
    }
    if !is_word_byte(bytes[pos]) {
        while pos < len && !is_word_byte(bytes[pos]) {
            pos += 1;
        }
        if pos >= len {
            return len;
        }
    }
    while pos + 1 < len && is_word_byte(bytes[pos + 1]) {
        pos += 1;
    }
    pos
}

fn visual_bounds(len: usize, anchor: usize, cursor_pos: usize) -> (usize, usize) {
    let start = anchor.min(cursor_pos).min(len);
    let mut end = anchor.max(cursor_pos).min(len);
    if end < len {
        end += 1;
    }
    (start, end)
}

fn extract_visual_selection(input: &str, anchor: usize, cursor_pos: usize) -> String {
    let (start, end) = visual_bounds(input.len(), anchor, cursor_pos);
    if start >= end {
        return String::new();
    }
    input[start..end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_slash_commands() {
        assert_eq!(SlashCommand::parse("/help"), Some(SlashCommand::Help));
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
            SlashCommand::parse("/remote-env list"),
            Some(SlashCommand::RemoteEnv(vec!["list".to_string()]))
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
        assert_eq!(
            SlashCommand::parse("/visual analyze --strict"),
            Some(SlashCommand::Visual(vec![
                "analyze".to_string(),
                "--strict".to_string()
            ]))
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
            context_used_tokens: 50_000,
            context_max_tokens: 128_000,
            session_turns: 5,
            working_directory: "/tmp".to_string(),
        });
        assert!(line.contains("model=deepseek-chat"));
        assert!(line.contains("autopilot=running"));
        assert!(line.contains("[ASK]"));
        assert!(line.contains("tasks=3"));
    }

    #[test]
    fn loads_keybindings_from_json_file() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("deepseek-ui-bindings-{nonce}"));
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
            background_jobs: 1,
            autopilot_running: true,
            context_used_tokens: 100_000,
            context_max_tokens: 128_000,
            ..Default::default()
        };
        let spans = render_statusline_spans(&status, None, "", None, None, false);
        let text: String = spans.iter().map(|s| s.content.to_string()).collect();
        assert!(text.contains("deepseek-chat"));
        assert!(text.contains("LOCKED"));
        assert!(text.contains("pending"));
        assert!(text.contains("tasks"));
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
    fn right_pane_cycles_through_all_variants() {
        let mut pane = RightPane::Plan;
        pane = pane.cycle();
        assert_eq!(pane, RightPane::Tools);
        pane = pane.cycle();
        assert_eq!(pane, RightPane::MissionControl);
        pane = pane.cycle();
        assert_eq!(pane, RightPane::Artifacts);
        pane = pane.cycle();
        assert_eq!(pane, RightPane::Plan);
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
        let spans = render_statusline_spans(&status, None, "", None, None, false);
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
        let spans = render_statusline_spans(&status, None, "", None, None, false);
        let text: String = spans.iter().map(|s| s.content.to_string()).collect();
        assert!(!text.contains("turn"));
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
}
