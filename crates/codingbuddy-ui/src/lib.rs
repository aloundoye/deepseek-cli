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
use syntect::highlighting::{Theme as SyntectTheme, ThemeSet};
use syntect::parsing::SyntaxSet;

mod keybindings;
mod panels;
mod slash_commands;
mod statusline;
mod stream_runtime;
mod theme;

#[cfg(test)]
use keybindings::KeyBindingsFile;
pub use keybindings::{KeyBindings, load_keybindings};
use panels::{load_artifact_lines, render_mission_control_panel};
use slash_commands::{
    SLASH_COMMAND_CATALOG, slash_command_suggestions, slash_suggestion_to_command,
};
pub use slash_commands::{SlashCommand, slash_command_catalog_entries};
use statusline::render_statusline_spans;
pub use statusline::{UiStatus, render_statusline};
use stream_runtime::{
    StreamEventResult, StreamRuntimeState, filter_stream_event, handle_stream_event,
};
pub use theme::TuiTheme;

/// Lazy-initialized syntect highlighting assets.
struct SyntectAssets {
    syntax_set: SyntaxSet,
    theme: SyntectTheme,
}

/// Returns a reference to the shared syntect assets (loaded once).
fn syntect_assets() -> &'static SyntectAssets {
    use std::sync::OnceLock;
    static ASSETS: OnceLock<SyntectAssets> = OnceLock::new();
    ASSETS.get_or_init(|| {
        let syntax_set = SyntaxSet::load_defaults_newlines();
        let theme_set = ThemeSet::load_defaults();
        let theme = theme_set
            .themes
            .get("base16-eighties.dark")
            .cloned()
            .unwrap_or_else(|| {
                theme_set
                    .themes
                    .values()
                    .next()
                    .cloned()
                    .expect("syntect ships with at least one theme")
            });
        SyntectAssets { syntax_set, theme }
    })
}

/// Map a syntect RGBA color to a ratatui terminal color.
fn syntect_color_to_ratatui(c: syntect::highlighting::Color) -> Color {
    Color::Rgb(c.r, c.g, c.b)
}

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
        /// Whether the tool call succeeded (true) or failed (false).
        success: bool,
    },
    /// The agent needs user approval before proceeding.
    ApprovalNeeded {
        tool_name: String,
        args_summary: String,
        response_tx: mpsc::Sender<bool>,
    },
    /// The agent switched execution modes.
    ModeTransition {
        from: String,
        to: String,
        reason: String,
    },
    /// A complex-task subagent was started.
    SubagentSpawned {
        run_id: String,
        name: String,
        goal: String,
    },
    /// A subagent completed.
    SubagentCompleted {
        run_id: String,
        name: String,
        summary: String,
    },
    /// A subagent failed.
    SubagentFailed {
        run_id: String,
        name: String,
        error: String,
    },
    /// A persisted system/background notice arrived outside the active turn.
    SystemNotice { line: String, error: bool },
    /// Watch mode auto-triggered because comment digest changed.
    WatchTriggered { digest: u64, comment_count: usize },
    /// Display an inline image in the terminal (raw bytes).
    ImageDisplay { data: Vec<u8>, label: String },
    /// Clear any previously streamed text — the response contained tool calls,
    /// so interleaved text fragments are noise and should be removed.
    ClearStreamingText,
    /// A diff was applied to a file — render inline in the TUI.
    DiffApplied {
        path: String,
        hunks: u32,
        added: u32,
        removed: u32,
    },
    /// End-of-turn usage summary for cost display.
    UsageSummary {
        input_tokens: u64,
        output_tokens: u64,
        cache_hit_tokens: u64,
        cost_usd: f64,
    },
    /// Role/phase header for clear visual boundaries.
    RoleHeader { role: String, model: String },
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

// ── Prompt Suggestions ──────────────────────────────────────────────────────

/// Example prompts shown as grayed-out placeholders when the input area is empty.
pub const PROMPT_SUGGESTIONS: &[&str] = &[
    "Explain this project's architecture",
    "Find and fix bugs in src/",
    "Add tests for the auth module",
    "Refactor the database layer",
    "Review my latest changes",
];

/// Render prompt suggestion spans for display in the input area.
/// Returns non-empty spans only when the input is empty.
pub fn render_prompt_suggestions(is_empty_input: bool) -> Vec<Span<'static>> {
    if !is_empty_input {
        return vec![];
    }
    let mut spans = Vec::new();
    for (i, suggestion) in PROMPT_SUGGESTIONS.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("  ", Style::default()));
        }
        spans.push(Span::styled(
            suggestion.to_string(),
            Style::default().fg(Color::DarkGray),
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

/// Extract the language token from a code fence line (e.g. ```` ```rust ```` → `"rust"`).
fn extract_fence_lang(fence: &str) -> &str {
    fence.trim().trim_start_matches('`').trim()
}

/// Apply syntax highlighting to a line of code inside a code block.
///
/// Uses `syntect` when a matching syntax definition exists for the given
/// `lang` (the language tag from the opening code fence).  Falls back to the
/// keyword-based highlighter for unknown languages.
fn highlight_code_line(line: &str, lang: &str) -> Line<'static> {
    let assets = syntect_assets();
    let syntax = if lang.is_empty() {
        None
    } else {
        assets
            .syntax_set
            .find_syntax_by_token(lang)
            .or_else(|| assets.syntax_set.find_syntax_by_extension(lang))
    };

    if let Some(syntax) = syntax {
        use syntect::easy::HighlightLines;
        let mut h = HighlightLines::new(syntax, &assets.theme);
        if let Ok(ranges) = h.highlight_line(line, &assets.syntax_set) {
            let spans: Vec<Span<'static>> = ranges
                .into_iter()
                .map(|(style, text)| {
                    let fg = syntect_color_to_ratatui(style.foreground);
                    let mut ratatui_style = Style::default().fg(fg);
                    if style
                        .font_style
                        .contains(syntect::highlighting::FontStyle::BOLD)
                    {
                        ratatui_style = ratatui_style.add_modifier(Modifier::BOLD);
                    }
                    if style
                        .font_style
                        .contains(syntect::highlighting::FontStyle::ITALIC)
                    {
                        ratatui_style = ratatui_style.add_modifier(Modifier::ITALIC);
                    }
                    Span::styled(text.to_string(), ratatui_style)
                })
                .collect();
            if !spans.is_empty() {
                return Line::from(spans);
            }
        }
    }

    // Fallback: keyword-based highlighting
    highlight_code_line_fallback(line)
}

/// Keyword-based code highlighting used when syntect has no matching syntax.
fn highlight_code_line_fallback(line: &str) -> Line<'static> {
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
        } else if bytes[i] == b'~' && i + 1 < len && bytes[i + 1] == b'~' {
            // Strikethrough: ~~...~~
            if let Some(end) = text[i + 2..].find("~~")
                && end > 0
            {
                if i > seg_start {
                    spans.push(Span::styled(text[seg_start..i].to_string(), base_style));
                }
                spans.extend(parse_inline_markdown(
                    &text[i + 2..i + 2 + end],
                    base_style
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::CROSSED_OUT),
                ));
                i = i + 2 + end + 2;
                seg_start = i;
                continue;
            }
        } else if bytes[i] == b'[' {
            // Markdown link: [text](url)
            if let Some(close_bracket) = text[i + 1..].find(']') {
                let after_bracket = i + 1 + close_bracket + 1;
                if after_bracket < len
                    && bytes[after_bracket] == b'('
                    && let Some(close_paren) = text[after_bracket + 1..].find(')')
                {
                    let link_text = &text[i + 1..i + 1 + close_bracket];
                    let link_url = &text[after_bracket + 1..after_bracket + 1 + close_paren];
                    if !link_text.is_empty() {
                        if i > seg_start {
                            spans.push(Span::styled(text[seg_start..i].to_string(), base_style));
                        }
                        spans.push(Span::styled(
                            link_text.to_string(),
                            base_style
                                .fg(Color::LightBlue)
                                .add_modifier(Modifier::UNDERLINED),
                        ));
                        spans.push(Span::styled(
                            format!(" ({link_url})"),
                            Style::default().fg(Color::DarkGray),
                        ));
                        i = after_bracket + 1 + close_paren + 1;
                        seg_start = i;
                        continue;
                    }
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
    if !text.is_empty() && text.chars().all(|c| c == '#') && text.len() <= 6 {
        return Line::from(vec![Span::raw("")]);
    }
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
            let mut spans = vec![];
            spans.extend(parse_inline_markdown(heading_text, heading_style));
            return Line::from(spans);
        }
    }

    // Blockquote
    if let Some(quote) = text.strip_prefix("> ") {
        let dim = Style::default().fg(Color::DarkGray);
        let mut spans = vec![Span::styled("│ ".to_string(), dim)];
        spans.extend(parse_inline_markdown(quote, dim));
        return Line::from(spans);
    }

    // Task list: - [ ] unchecked / - [x] checked
    if let Some(rest) = text
        .strip_prefix("- [ ] ")
        .or_else(|| text.strip_prefix("* [ ] "))
    {
        let mut spans = vec![Span::styled(
            "☐ ".to_string(),
            Style::default().fg(Color::DarkGray),
        )];
        spans.extend(parse_inline_markdown(rest, body_style));
        return Line::from(spans);
    }
    if let Some(rest) = text
        .strip_prefix("- [x] ")
        .or_else(|| text.strip_prefix("* [x] "))
        .or_else(|| text.strip_prefix("- [X] "))
        .or_else(|| text.strip_prefix("* [X] "))
    {
        let mut spans = vec![Span::styled(
            "☑ ".to_string(),
            Style::default().fg(Color::Green),
        )];
        spans.extend(parse_inline_markdown(
            rest,
            body_style
                .fg(Color::DarkGray)
                .add_modifier(Modifier::CROSSED_OUT),
        ));
        return Line::from(spans);
    }

    // Unordered list
    if let Some(item) = text.strip_prefix("- ").or_else(|| text.strip_prefix("* ")) {
        let mut spans = vec![Span::styled(
            "• ".to_string(),
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
            num.to_string(),
            Style::default().fg(Color::Cyan),
        )];
        spans.extend(parse_inline_markdown(item, body_style));
        return Line::from(spans);
    }

    // Table rows: lines starting and ending with `|`
    if text.len() >= 2 && text.starts_with('|') && text.ends_with('|') {
        let trimmed = &text[1..text.len() - 1];
        // Separator row (e.g. |---|---|)
        if trimmed
            .chars()
            .all(|c| c == '-' || c == '|' || c == ':' || c == ' ')
        {
            return Line::from(vec![Span::styled(
                text.to_string(),
                Style::default().fg(Color::DarkGray),
            )]);
        }
        // Data/header row
        let cells: Vec<&str> = trimmed.split('|').map(|c| c.trim()).collect();
        let mut spans = vec![Span::styled(
            "│".to_string(),
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

    // Default: plain body with inline markdown
    Line::from(parse_inline_markdown(text, body_style))
}

fn code_fence_starts_diff(line: &str) -> bool {
    let trimmed = line.trim();
    if !trimmed.starts_with("```") {
        return false;
    }
    let lang = trimmed.trim_start_matches('`').trim().to_ascii_lowercase();
    lang.starts_with("diff") || lang.starts_with("patch")
}

fn looks_like_unfenced_diff_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with("diff --git")
        || trimmed.starts_with("index ")
        || trimmed.starts_with("@@")
        || trimmed.starts_with("--- ")
        || trimmed.starts_with("+++ ")
    {
        return true;
    }

    let mut chars = trimmed.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    let second = chars.next();

    match first {
        '+' if !trimmed.starts_with("+++") => second.is_some_and(|c| !c.is_whitespace()),
        '-' if !trimmed.starts_with("---") => second.is_some_and(|c| !c.is_whitespace()),
        _ => false,
    }
}

fn render_diff_line(line: &str) -> Line<'static> {
    if line.starts_with("diff --git")
        || line.starts_with("index ")
        || line.starts_with("--- ")
        || line.starts_with("+++ ")
    {
        return Line::from(vec![Span::styled(
            line.to_string(),
            Style::default().fg(Color::DarkGray),
        )]);
    }
    if line.starts_with("@@") {
        return Line::from(vec![Span::styled(
            line.to_string(),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )]);
    }
    if let Some(rest) = line.strip_prefix('+') {
        return Line::from(vec![
            Span::styled(
                "+".to_string(),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(rest.to_string(), Style::default().fg(Color::Green)),
        ]);
    }
    if let Some(rest) = line.strip_prefix('-') {
        return Line::from(vec![
            Span::styled(
                "-".to_string(),
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::styled(rest.to_string(), Style::default().fg(Color::Red)),
        ]);
    }
    if let Some(rest) = line.strip_prefix(' ') {
        return Line::from(vec![
            Span::styled(" ".to_string(), Style::default().fg(Color::DarkGray)),
            Span::styled(rest.to_string(), Style::default().fg(Color::DarkGray)),
        ]);
    }
    Line::from(vec![Span::styled(
        line.to_string(),
        Style::default().fg(Color::DarkGray),
    )])
}

const DIFF_CONTEXT_COLLAPSE_THRESHOLD: usize = 12;
const DIFF_CONTEXT_KEEP_HEAD: usize = 2;
const DIFF_CONTEXT_KEEP_TAIL: usize = 2;
const INPUT_MIN_HEIGHT: u16 = 1;
const INPUT_MAX_HEIGHT: u16 = 4;
const STREAM_MIN_HEIGHT: u16 = 1;
const INLINE_VIEWPORT_HEIGHT: u16 = INPUT_MAX_HEIGHT + STREAM_MIN_HEIGHT + 3;

fn flush_diff_context_run(
    run: &mut Vec<String>,
    out: &mut Vec<Line<'static>>,
    collapse_threshold: usize,
    keep_head: usize,
    keep_tail: usize,
) {
    if run.is_empty() {
        return;
    }
    if run.len() > collapse_threshold {
        for line in run.iter().take(keep_head) {
            out.push(render_diff_line(line));
        }
        let hidden = run.len().saturating_sub(keep_head + keep_tail);
        if hidden > 0 {
            out.push(Line::from(vec![Span::styled(
                format!("… {hidden} unchanged lines hidden …"),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            )]));
        }
        for line in run.iter().skip(run.len().saturating_sub(keep_tail)) {
            out.push(render_diff_line(line));
        }
    } else {
        for line in run.iter() {
            out.push(render_diff_line(line));
        }
    }
    run.clear();
}

fn wrapped_line_height(line: &Line<'_>, width: u16) -> u16 {
    if width == 0 {
        return 0;
    }
    let width = width as usize;
    let content_width = line.width().max(1);
    let rows = (content_width.saturating_sub(1) / width) + 1;
    u16::try_from(rows).unwrap_or(u16::MAX)
}

fn wrapped_text_rows(text: &str, width: u16) -> usize {
    if width == 0 {
        return 0;
    }
    let width = width as usize;
    let mut rows = 0usize;
    for segment in text.split('\n') {
        let len = segment.chars().count();
        rows = rows.saturating_add((len.saturating_sub(1) / width) + 1);
    }
    rows.max(1)
}

fn scroll_to_keep_row_visible(row: usize, viewport_rows: u16) -> u16 {
    if viewport_rows == 0 {
        return 0;
    }
    row.saturating_sub(viewport_rows.saturating_sub(1) as usize)
        .min(u16::MAX as usize) as u16
}

fn compute_inline_heights(total_height: u16, desired_input_rows: usize) -> (u16, u16) {
    // Reserve 3 fixed rows: separator above input, separator below input, and status line.
    let dynamic_rows = total_height.saturating_sub(3);
    if dynamic_rows == 0 {
        return (0, 0);
    }
    if dynamic_rows == 1 {
        return (1, 0);
    }

    let desired_input =
        desired_input_rows.clamp(INPUT_MIN_HEIGHT as usize, INPUT_MAX_HEIGHT as usize) as u16;
    let max_input_by_space = dynamic_rows.saturating_sub(STREAM_MIN_HEIGHT);
    let input_height = desired_input.min(max_input_by_space).max(INPUT_MIN_HEIGHT);
    let stream_height = dynamic_rows
        .saturating_sub(input_height)
        .max(STREAM_MIN_HEIGHT);
    (stream_height, input_height)
}

pub(crate) fn truncate_inline(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let keep = max_chars.saturating_sub(3);
    let mut out = String::new();
    for ch in text.chars().take(keep) {
        out.push(ch);
    }
    out.push_str("...");
    out
}

fn operator_summary_line(status: &UiStatus) -> String {
    let phase = if status.workflow_phase.is_empty() {
        "idle"
    } else {
        status.workflow_phase.as_str()
    };
    let todo = if status.current_todo.trim().is_empty() {
        "none".to_string()
    } else {
        truncate_inline(&status.current_todo, 36)
    };
    let step = if status.current_step.trim().is_empty() {
        "none".to_string()
    } else {
        truncate_inline(&status.current_step, 28)
    };
    let blockers = format!(
        "failed(tasks={},subagents={})",
        status.failed_tasks, status.failed_subagents
    );
    let capabilities = if status.capability_summary.trim().is_empty() {
        "n/a".to_string()
    } else {
        truncate_inline(&status.capability_summary, 48)
    };
    format!(
        "phase={phase} step={step} todo={todo} subagents={}/{} bg={} {} caps={} compaction/replay={}/{}",
        status.running_subagents,
        status.failed_subagents,
        status.running_background_jobs.max(status.background_jobs),
        blockers,
        capabilities,
        status.compaction_count,
        status.replay_count
    )
}

fn compact_tool_meta_tail(tail: &str, max_chars: usize) -> String {
    let squashed = tail.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some(json_start) = squashed.find('{') {
        let head = squashed[..json_start].trim();
        let compact = if head.is_empty() {
            "{…}".to_string()
        } else {
            format!("{head} {{…}}")
        };
        return truncate_inline(&compact, max_chars);
    }
    truncate_inline(&squashed, max_chars)
}

fn parse_stream_meta_entry(text: &str) -> Option<TranscriptEntry> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Some(rest) = trimmed.strip_prefix("[tool: ")
        && let Some(end) = rest.find(']')
    {
        let tool_name = rest[..end].trim();
        let tail = rest[end + 1..].trim();
        if tool_name.is_empty() {
            return None;
        }
        let is_result = tail.starts_with("ok ")
            || tail.starts_with("error ")
            || tail.starts_with("denied ")
            || tail.starts_with("failed ")
            || tail == "ok"
            || tail == "error"
            || tail == "denied";
        let compact_tail = compact_tool_meta_tail(tail, 120);
        return Some(TranscriptEntry {
            kind: if is_result {
                MessageKind::ToolResult
            } else {
                MessageKind::ToolCall
            },
            text: if compact_tail.is_empty() {
                tool_name.to_string()
            } else {
                format!("{tool_name} {compact_tail}")
            },
        });
    }

    if let Some(rest) = trimmed.strip_prefix("[mcp: ")
        && let Some(end) = rest.find(']')
    {
        let tool_name = rest[..end].trim();
        if tool_name.is_empty() {
            return None;
        }
        return Some(TranscriptEntry {
            kind: MessageKind::ToolCall,
            text: format!("mcp.{tool_name}"),
        });
    }

    if let Some(rest) = trimmed.strip_prefix("[thinking] ") {
        return Some(TranscriptEntry {
            kind: MessageKind::System,
            text: format!("thinking: {rest}"),
        });
    }

    None
}

fn style_transcript_line(
    entry: &TranscriptEntry,
    in_code_block: bool,
    in_diff_block: bool,
    code_block_lang: &str,
) -> Line<'static> {
    if entry.kind == MessageKind::Assistant
        && let Some(meta_entry) = parse_stream_meta_entry(&entry.text)
    {
        return style_transcript_line(&meta_entry, false, false, "");
    }

    let (prefix, prefix_style, mut body_style) = match entry.kind {
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
            Style::default().fg(Color::Green),
        ),
        MessageKind::Error => (
            "✗ ",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            Style::default().fg(Color::Red),
        ),
        MessageKind::Thinking => (
            "  ◐ ",
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::ITALIC),
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        ),
    };

    let text = &entry.text;
    if entry.kind == MessageKind::ToolResult {
        let lower = text.to_ascii_lowercase();
        if lower.contains("error") || lower.contains("failed") || lower.contains("denied") {
            body_style = Style::default().fg(Color::Red);
        } else if lower.contains("ok") {
            body_style = Style::default().fg(Color::Green);
        } else {
            body_style = Style::default().fg(Color::DarkGray);
        }
    }

    // Apply syntax highlighting inside code blocks
    if in_code_block && entry.kind == MessageKind::Assistant && !text.starts_with("```") {
        if in_diff_block {
            return render_diff_line(text);
        }
        return highlight_code_line(text, code_block_lang);
    }

    if entry.kind == MessageKind::Assistant && !in_code_block && looks_like_unfenced_diff_line(text)
    {
        return render_diff_line(text);
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
fn style_continuation_line(
    entry: &TranscriptEntry,
    in_code_block: bool,
    in_diff_block: bool,
    code_block_lang: &str,
) -> Line<'static> {
    if entry.kind == MessageKind::Assistant
        && let Some(meta_entry) = parse_stream_meta_entry(&entry.text)
    {
        return style_transcript_line(&meta_entry, false, false, "");
    }

    let mut body_style = match entry.kind {
        MessageKind::User => Style::default().fg(Color::White),
        MessageKind::Assistant => Style::default().fg(Color::White),
        MessageKind::System => Style::default().fg(Color::DarkGray),
        MessageKind::ToolCall => Style::default().fg(Color::Yellow),
        MessageKind::ToolResult => Style::default().fg(Color::Green),
        MessageKind::Error => Style::default().fg(Color::Red),
        MessageKind::Thinking => Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC),
    };
    let text = &entry.text;
    if entry.kind == MessageKind::ToolResult {
        let lower = text.to_ascii_lowercase();
        if lower.contains("error") || lower.contains("failed") || lower.contains("denied") {
            body_style = Style::default().fg(Color::Red);
        } else if lower.contains("ok") {
            body_style = Style::default().fg(Color::Green);
        } else {
            body_style = Style::default().fg(Color::DarkGray);
        }
    }
    if in_code_block && entry.kind == MessageKind::Assistant && !text.starts_with("```") {
        if in_diff_block {
            return render_diff_line(text);
        }
        return highlight_code_line(text, code_block_lang);
    }
    if entry.kind == MessageKind::Assistant && !in_code_block && looks_like_unfenced_diff_line(text)
    {
        return render_diff_line(text);
    }
    // Full markdown rendering for assistant continuation lines too
    if entry.kind == MessageKind::Assistant {
        return render_assistant_markdown(text);
    }
    // Indent continuation to align with body text after the prefix
    Line::from(vec![Span::styled(format!("  {text}"), body_style)])
}

fn split_inline_markdown_blocks(text: &str) -> Vec<String> {
    if text.is_empty() || text.starts_with("```") {
        return vec![text.to_string()];
    }

    let mut breakpoints: Vec<usize> = Vec::new();
    let bytes = text.as_bytes();

    for (idx, ch) in text.char_indices() {
        if idx == 0 {
            continue;
        }

        // Split inline headings like "intro:## Heading" into two logical lines.
        if ch == '#' {
            if bytes.get(idx.saturating_sub(1)) == Some(&b'#') {
                continue;
            }
            let mut j = idx;
            while j < bytes.len() && bytes[j] == b'#' && (j - idx) < 6 {
                j += 1;
            }
            let run_len = j.saturating_sub(idx);
            if (2..=6).contains(&run_len) {
                let next = text[j..].chars().next();
                let heading_like =
                    next.is_none_or(|value| value.is_whitespace() || value.is_alphanumeric());
                if heading_like {
                    let prefix = text[..idx].trim_end();
                    let inline_code_ticks = text[..idx].chars().filter(|c| *c == '`').count();
                    if !prefix.is_empty() && inline_code_ticks % 2 == 0 {
                        breakpoints.push(idx);
                        continue;
                    }
                }
            }
        }

        // Split inline ordered list starts like "Strengths ✅ 1. Item".
        if ch.is_ascii_digit() {
            let mut j = idx;
            while j < bytes.len() && bytes[j].is_ascii_digit() && (j - idx) < 3 {
                j += 1;
            }
            if j > idx
                && bytes.get(j) == Some(&b'.')
                && bytes.get(j + 1) == Some(&b' ')
                && let Some(prev) = text[..idx].chars().next_back()
                && (prev.is_whitespace() || "✅✔☑☐•-".contains(prev))
            {
                let prefix = text[..idx].trim_end();
                if !prefix.is_empty() {
                    breakpoints.push(idx);
                }
            }
        }

        // Split inline unordered/task list starts:
        // "Architecture- Detail", "areas:- Item", or "...  - Item".
        if ch == '-' {
            let next_char = text[idx + 1..].chars().next();
            if next_char == Some(' ') && outside_inline_code(text, idx) {
                let prefix_raw = &text[..idx];
                let prefix = prefix_raw.trim_end();
                if !prefix.is_empty() {
                    let prev = prefix_raw.chars().next_back();
                    let looks_inline_list = prev.is_some_and(|p| !p.is_whitespace())
                        || prefix_raw.ends_with("  ")
                        || prefix_raw.ends_with(':')
                        || prefix_raw.ends_with(')');
                    if looks_inline_list {
                        breakpoints.push(idx);
                        continue;
                    }
                }
            }
        }

        // Split run-on section boundaries like:
        // "Project OverviewDeepSeek ...", "(Git Status)The ...".
        if ch.is_ascii_uppercase()
            && outside_inline_code(text, idx)
            && let Some(prev) = text[..idx].chars().next_back()
        {
            if prev == ')' {
                let prefix = text[..idx].trim_end();
                if !prefix.is_empty() {
                    breakpoints.push(idx);
                    continue;
                }
            } else if prev.is_ascii_lowercase() {
                let prefix = text[..idx].trim_end();
                if is_run_on_section_boundary(prefix) {
                    breakpoints.push(idx);
                    continue;
                }
            }
        }

        if text[idx..].starts_with("codingbuddy-")
            && outside_inline_code(text, idx)
            && let Some(prev) = text[..idx].chars().next_back()
            && prev.is_ascii_lowercase()
        {
            let prefix = text[..idx].trim_end();
            if is_run_on_section_boundary(prefix) {
                breakpoints.push(idx);
                continue;
            }
        }
    }

    breakpoints.sort_unstable();
    breakpoints.dedup();
    if breakpoints.is_empty() {
        return vec![text.to_string()];
    }

    let mut out = Vec::with_capacity(breakpoints.len() + 1);
    let mut start = 0usize;
    for bp in breakpoints {
        if bp <= start || bp >= text.len() {
            continue;
        }
        let segment = text[start..bp].trim_end();
        if !segment.is_empty() {
            out.push(segment.to_string());
        }
        start = bp;
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }

    if out.is_empty() {
        vec![text.to_string()]
    } else {
        out
    }
}

fn outside_inline_code(text: &str, idx: usize) -> bool {
    text[..idx].chars().filter(|c| *c == '`').count() % 2 == 0
}

fn is_run_on_section_boundary(prefix: &str) -> bool {
    let words = prefix
        .split_whitespace()
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>();
    if words.len() < 2 || words.len() > 8 {
        return false;
    }
    let last = words
        .last()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_ascii_alphanumeric())
                .to_ascii_lowercase()
        })
        .unwrap_or_default();
    matches!(
        last.as_str(),
        "overview"
            | "components"
            | "architecture"
            | "features"
            | "infrastructure"
            | "structure"
            | "stack"
            | "focus"
            | "highlights"
            | "summary"
            | "findings"
            | "modifications"
            | "indicators"
            | "scope"
    )
}

fn is_plan_list_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
        return true;
    }
    let mut saw_digit = false;
    for ch in trimmed.chars() {
        if ch.is_ascii_digit() {
            saw_digit = true;
            continue;
        }
        if saw_digit && ch == '.' {
            return true;
        }
        break;
    }
    false
}

fn insert_wrapped_lines_above(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    lines: &[Line<'static>],
) -> Result<()> {
    if lines.is_empty() {
        return Ok(());
    }
    let viewport_width = terminal.size()?.width.max(1);
    let height = lines
        .iter()
        .map(|line| wrapped_line_height(line, viewport_width) as u32)
        .sum::<u32>()
        .min(u16::MAX as u32) as u16;
    if height == 0 {
        return Ok(());
    }
    terminal.insert_before(height, |buf| {
        let area = buf.area;
        let bottom = area.y.saturating_add(area.height);
        let mut y = area.y;
        for line in lines {
            if y >= bottom {
                break;
            }
            let logical_height = wrapped_line_height(line, area.width).max(1);
            let remaining = bottom.saturating_sub(y);
            let render_height = logical_height.min(remaining);
            if render_height == 0 {
                break;
            }
            let line_area = Rect::new(area.x, y, area.width, render_height);
            Paragraph::new(line.clone())
                .wrap(Wrap { trim: false })
                .render(line_area, buf);
            y = y.saturating_add(logical_height);
        }
    })?;
    Ok(())
}

/// Flush new transcript entries above the inline viewport into native terminal scrollback.
fn flush_transcript_above(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    shell: &ChatShell,
    last_printed_idx: &mut usize,
    collapse_plan: bool,
) -> Result<()> {
    if *last_printed_idx >= shell.transcript.len() {
        return Ok(());
    }
    let new_entries = &shell.transcript[*last_printed_idx..];
    // Build styled lines for the new entries
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut in_code_block = false;
    let mut in_diff_block = false;
    let mut code_block_lang = String::new();
    let mut diff_context_run: Vec<String> = Vec::new();
    let mut collapsed_plan_lines = 0usize;
    // Scan earlier entries to figure out if we're inside a code block
    for entry in &shell.transcript[..*last_printed_idx] {
        for sub in entry.text.split('\n') {
            if entry.kind == MessageKind::Assistant && sub.starts_with("```") {
                if !in_code_block {
                    in_code_block = true;
                    in_diff_block = code_fence_starts_diff(sub);
                    code_block_lang = extract_fence_lang(sub).to_string();
                } else {
                    in_code_block = false;
                    in_diff_block = false;
                    code_block_lang.clear();
                }
            }
        }
    }
    for entry in new_entries {
        let mut rendered_first_line = false;
        for sub_text in entry.text.split('\n') {
            let logical_lines = if entry.kind == MessageKind::Assistant && !in_code_block {
                split_inline_markdown_blocks(sub_text)
            } else {
                vec![sub_text.to_string()]
            };
            for logical_text in logical_lines {
                let sub_entry = TranscriptEntry {
                    kind: entry.kind,
                    text: logical_text,
                };
                if collapse_plan
                    && sub_entry.kind == MessageKind::Assistant
                    && !in_code_block
                    && is_plan_list_line(&sub_entry.text)
                {
                    collapsed_plan_lines = collapsed_plan_lines.saturating_add(1);
                    continue;
                }
                let is_fence =
                    sub_entry.kind == MessageKind::Assistant && sub_entry.text.starts_with("```");
                if is_fence
                    || sub_entry.kind != MessageKind::Assistant
                    || !in_code_block
                    || !in_diff_block
                {
                    flush_diff_context_run(
                        &mut diff_context_run,
                        &mut lines,
                        DIFF_CONTEXT_COLLAPSE_THRESHOLD,
                        DIFF_CONTEXT_KEEP_HEAD,
                        DIFF_CONTEXT_KEEP_TAIL,
                    );
                }
                if sub_entry.kind == MessageKind::Assistant
                    && in_code_block
                    && in_diff_block
                    && sub_entry.text.starts_with(' ')
                    && !is_fence
                {
                    diff_context_run.push(sub_entry.text.clone());
                    continue;
                }
                if sub_entry.kind == MessageKind::Assistant && is_fence {
                    if !in_code_block {
                        in_code_block = true;
                        in_diff_block = code_fence_starts_diff(&sub_entry.text);
                        code_block_lang = extract_fence_lang(&sub_entry.text).to_string();
                    } else {
                        in_code_block = false;
                        in_diff_block = false;
                        code_block_lang.clear();
                    }
                }
                if !rendered_first_line {
                    lines.push(style_transcript_line(
                        &sub_entry,
                        in_code_block,
                        in_diff_block,
                        &code_block_lang,
                    ));
                    rendered_first_line = true;
                } else {
                    lines.push(style_continuation_line(
                        &sub_entry,
                        in_code_block,
                        in_diff_block,
                        &code_block_lang,
                    ));
                }
            }
        }
    }
    flush_diff_context_run(
        &mut diff_context_run,
        &mut lines,
        DIFF_CONTEXT_COLLAPSE_THRESHOLD,
        DIFF_CONTEXT_KEEP_HEAD,
        DIFF_CONTEXT_KEEP_TAIL,
    );
    if collapsed_plan_lines > 0 {
        lines.push(Line::from(vec![Span::styled(
            format!(
                "… {} plan lines hidden (Ctrl+P to expand)",
                collapsed_plan_lines
            ),
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        )]));
    }
    *last_printed_idx = shell.transcript.len();
    insert_wrapped_lines_above(terminal, &lines)
}

/// Flush complete lines from the streaming buffer into native scrollback.
/// Returns the remaining partial line (text after the last `\n`).
fn flush_streaming_lines(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    buffer: &mut String,
    in_code_block: &mut bool,
    in_diff_block: &mut bool,
    code_block_lang: &mut String,
    collapse_plan: bool,
) -> Result<()> {
    // Find the last newline — everything before it consists of complete lines.
    let Some(last_nl) = buffer.rfind('\n') else {
        return Ok(());
    };
    let complete = buffer[..last_nl].to_string();
    let remaining = buffer[last_nl + 1..].to_string();
    *buffer = remaining;

    let lines_text: Vec<&str> = complete.split('\n').collect();
    let mut styled_lines: Vec<Line<'static>> = Vec::with_capacity(lines_text.len());
    let mut diff_context_run: Vec<String> = Vec::new();
    let mut collapsed_plan_lines = 0usize;
    for line_text in &lines_text {
        let logical_lines = if !*in_code_block {
            split_inline_markdown_blocks(line_text)
        } else {
            vec![(*line_text).to_string()]
        };
        for logical_text in logical_lines {
            if collapse_plan && !*in_code_block && is_plan_list_line(&logical_text) {
                collapsed_plan_lines = collapsed_plan_lines.saturating_add(1);
                continue;
            }
            let is_fence = logical_text.starts_with("```");
            if is_fence || !*in_code_block || !*in_diff_block {
                flush_diff_context_run(
                    &mut diff_context_run,
                    &mut styled_lines,
                    DIFF_CONTEXT_COLLAPSE_THRESHOLD,
                    DIFF_CONTEXT_KEEP_HEAD,
                    DIFF_CONTEXT_KEEP_TAIL,
                );
            }
            if *in_code_block && *in_diff_block && logical_text.starts_with(' ') && !is_fence {
                diff_context_run.push(logical_text);
                continue;
            }
            if let Some(entry) = parse_stream_meta_entry(&logical_text) {
                flush_diff_context_run(
                    &mut diff_context_run,
                    &mut styled_lines,
                    DIFF_CONTEXT_COLLAPSE_THRESHOLD,
                    DIFF_CONTEXT_KEEP_HEAD,
                    DIFF_CONTEXT_KEEP_TAIL,
                );
                styled_lines.push(style_transcript_line(&entry, false, false, ""));
                continue;
            }
            let entry = TranscriptEntry {
                kind: MessageKind::Assistant,
                text: logical_text,
            };
            if entry.text.starts_with("```") {
                if !*in_code_block {
                    *in_code_block = true;
                    *in_diff_block = code_fence_starts_diff(&entry.text);
                    *code_block_lang = extract_fence_lang(&entry.text).to_string();
                } else {
                    *in_code_block = false;
                    *in_diff_block = false;
                    code_block_lang.clear();
                }
            }
            styled_lines.push(style_transcript_line(
                &entry,
                *in_code_block,
                *in_diff_block,
                code_block_lang,
            ));
        }
    }
    flush_diff_context_run(
        &mut diff_context_run,
        &mut styled_lines,
        DIFF_CONTEXT_COLLAPSE_THRESHOLD,
        DIFF_CONTEXT_KEEP_HEAD,
        DIFF_CONTEXT_KEEP_TAIL,
    );
    if collapsed_plan_lines > 0 {
        styled_lines.push(Line::from(vec![Span::styled(
            format!(
                "… {} streaming plan lines hidden (Ctrl+P to expand)",
                collapsed_plan_lines
            ),
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        )]));
    }
    insert_wrapped_lines_above(terminal, &styled_lines)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageKind {
    User,
    Assistant,
    System,
    ToolCall,
    ToolResult,
    Error,
    Thinking,
}

#[derive(Debug, Clone)]
pub struct TranscriptEntry {
    pub kind: MessageKind,
    pub text: String,
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
    /// Whether the model is currently in a thinking/reasoning phase.
    pub is_thinking: bool,
    /// Accumulated thinking text for the current reasoning block.
    pub thinking_buffer: String,
    /// Current agent execution mode label.
    pub agent_mode: String,
    /// When true, disable spinner animations (accessibility/reduced-motion).
    pub reduced_motion: bool,
    /// Maximum retained mission-control lines.
    pub mission_control_max_events: usize,
    /// Thinking visibility mode (`concise` or `raw`).
    pub thinking_visibility: String,
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

    pub fn push_thinking(&mut self, line: impl Into<String>) {
        self.transcript.push(TranscriptEntry {
            kind: MessageKind::Thinking,
            text: line.into(),
        });
    }

    pub fn push_tool(&mut self, line: impl Into<String>) {
        self.tool_lines.push(line.into());
    }

    pub fn push_mission_control(&mut self, line: impl Into<String>) {
        self.mission_control_lines.push(line.into());
        if self.mission_control_max_events > 0
            && self.mission_control_lines.len() > self.mission_control_max_events
        {
            let over = self
                .mission_control_lines
                .len()
                .saturating_sub(self.mission_control_max_events);
            if over > 0 {
                self.mission_control_lines.drain(0..over);
            }
        }
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

    /// Clear any partially streamed assistant text — used when the response turns out to
    /// contain tool calls, making the interleaved text fragments visual noise.
    pub fn clear_streaming_text(&mut self) {
        if let Some(last) = self.transcript.last_mut()
            && last.kind == MessageKind::Assistant
        {
            last.text.clear();
        }
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
        if !status.workflow_phase.is_empty() {
            self.push_system(format!("Workflow phase: {}", status.workflow_phase));
        }
        if status.plan_state != "none" {
            self.push_system(format!("Plan state: {}", status.plan_state));
        }
        if let Some(review) = status.pr_review_status.as_deref() {
            self.push_system(format!("PR review: {}", review));
        }
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
        if self.reduced_motion {
            return "●";
        }
        const FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        FRAMES[self.spinner_tick % FRAMES.len()]
    }
}

// ─── Rewind Picker ──────────────────────────────────────────────────────────

/// Human-readable labels and their corresponding [`RewindAction`] values.
pub const REWIND_ACTIONS: &[(&str, codingbuddy_core::RewindAction)] = &[
    (
        "Restore code & conversation",
        codingbuddy_core::RewindAction::RestoreCodeAndConversation,
    ),
    (
        "Restore conversation only",
        codingbuddy_core::RewindAction::RestoreConversationOnly,
    ),
    (
        "Restore code only",
        codingbuddy_core::RewindAction::RestoreCodeOnly,
    ),
    (
        "Summarize from here",
        codingbuddy_core::RewindAction::Summarize,
    ),
    ("Cancel", codingbuddy_core::RewindAction::Cancel),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RewindPickerPhase {
    SelectCheckpoint,
    SelectAction,
}

#[derive(Debug, Clone)]
pub struct RewindPickerState {
    pub checkpoints: Vec<codingbuddy_store::CheckpointRecord>,
    pub selected_index: usize,
    pub action_index: usize,
    pub phase: RewindPickerPhase,
}

impl RewindPickerState {
    pub fn new(checkpoints: Vec<codingbuddy_store::CheckpointRecord>) -> Self {
        Self {
            checkpoints,
            selected_index: 0,
            action_index: 0,
            phase: RewindPickerPhase::SelectCheckpoint,
        }
    }

    /// Move selection up.
    pub fn up(&mut self) {
        let limit = match self.phase {
            RewindPickerPhase::SelectCheckpoint => self.checkpoints.len(),
            RewindPickerPhase::SelectAction => REWIND_ACTIONS.len(),
        };
        let idx = match self.phase {
            RewindPickerPhase::SelectCheckpoint => &mut self.selected_index,
            RewindPickerPhase::SelectAction => &mut self.action_index,
        };
        if *idx > 0 {
            *idx -= 1;
        } else {
            *idx = limit.saturating_sub(1);
        }
    }

    /// Move selection down.
    pub fn down(&mut self) {
        let limit = match self.phase {
            RewindPickerPhase::SelectCheckpoint => self.checkpoints.len(),
            RewindPickerPhase::SelectAction => REWIND_ACTIONS.len(),
        };
        let idx = match self.phase {
            RewindPickerPhase::SelectCheckpoint => &mut self.selected_index,
            RewindPickerPhase::SelectAction => &mut self.action_index,
        };
        if *idx + 1 < limit {
            *idx += 1;
        } else {
            *idx = 0;
        }
    }

    /// Confirm current selection. Returns `Some(action)` when a final action is chosen.
    pub fn confirm(&mut self) -> Option<(usize, codingbuddy_core::RewindAction)> {
        match self.phase {
            RewindPickerPhase::SelectCheckpoint => {
                if self.checkpoints.is_empty() {
                    return Some((0, codingbuddy_core::RewindAction::Cancel));
                }
                self.phase = RewindPickerPhase::SelectAction;
                self.action_index = 0;
                None
            }
            RewindPickerPhase::SelectAction => {
                let (_, action) = REWIND_ACTIONS[self.action_index];
                Some((self.selected_index, action))
            }
        }
    }

    /// Go back one phase, or return true if we should close the picker.
    pub fn back(&mut self) -> bool {
        match self.phase {
            RewindPickerPhase::SelectAction => {
                self.phase = RewindPickerPhase::SelectCheckpoint;
                false
            }
            RewindPickerPhase::SelectCheckpoint => true,
        }
    }

    /// Format a checkpoint line for display with relative timestamp and file count.
    pub fn format_checkpoint_line(
        cp: &codingbuddy_store::CheckpointRecord,
        selected: bool,
    ) -> String {
        let marker = if selected { ">" } else { " " };
        let time = format_relative_time(&cp.created_at);
        format!(
            "{marker} [{time}] {reason} ({files} files)",
            reason = cp.reason,
            files = cp.files_count
        )
    }

    /// Return the visible viewport range for long checkpoint lists (max 8 visible).
    pub fn viewport(&self) -> std::ops::Range<usize> {
        const VIEWPORT_SIZE: usize = 8;
        let total = self.checkpoints.len();
        if total <= VIEWPORT_SIZE {
            return 0..total;
        }
        let half = VIEWPORT_SIZE / 2;
        let start = if self.selected_index <= half {
            0
        } else if self.selected_index + half >= total {
            total - VIEWPORT_SIZE
        } else {
            self.selected_index - half
        };
        start..(start + VIEWPORT_SIZE).min(total)
    }
}

/// Format an ISO 8601 timestamp as a relative time string (e.g., "2m ago", "1h ago").
pub fn format_relative_time(timestamp: &str) -> String {
    // Parse ISO 8601 timestamp
    if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(timestamp) {
        let now = chrono::Utc::now();
        let delta = now.signed_duration_since(ts);

        if delta.num_seconds() < 60 {
            return "just now".to_string();
        }
        if delta.num_minutes() < 60 {
            return format!("{}m ago", delta.num_minutes());
        }
        if delta.num_hours() < 24 {
            return format!("{}h ago", delta.num_hours());
        }
        return format!("{}d ago", delta.num_days());
    }
    // Fallback: return the raw timestamp truncated
    timestamp.chars().take(19).collect()
}

// ─── Model Picker ───────────────────────────────────────────────────────────

/// Available model choices for the interactive `/model` picker.
const MODEL_CHOICES: &[(&str, &str)] = &[
    ("deepseek-chat", "Thinking + tools (default)"),
    ("deepseek-reasoner", "Deep reasoning + tools"),
];

#[derive(Debug, Clone, Default)]
pub struct ModelPickerState {
    pub selected: usize,
}

impl ModelPickerState {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }
    pub fn down(&mut self) {
        self.selected = (self.selected + 1).min(MODEL_CHOICES.len() - 1);
    }
    pub fn confirm(&self) -> &'static str {
        MODEL_CHOICES[self.selected].0
    }
}

// ─── Autocomplete Dropdown ───────────────────────────────────────────────────

/// State for the `@` file autocomplete dropdown.
#[derive(Debug, Clone)]
pub struct AutocompleteState {
    pub suggestions: Vec<String>,
    pub selected: usize,
    pub trigger_pos: usize, // position of trigger char ('@' or '/') in input
}

impl AutocompleteState {
    pub fn new(suggestions: Vec<String>, trigger_pos: usize) -> Self {
        Self {
            suggestions,
            selected: 0,
            trigger_pos,
        }
    }

    pub fn up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        } else if !self.suggestions.is_empty() {
            self.selected = self.suggestions.len() - 1;
        }
    }

    pub fn down(&mut self) {
        if !self.suggestions.is_empty() {
            self.selected = (self.selected + 1) % self.suggestions.len();
        }
    }

    pub fn selected_value(&self) -> Option<&str> {
        self.suggestions.get(self.selected).map(|s| s.as_str())
    }

    /// Format the dropdown display as text lines for the info area.
    pub fn display_lines(&self, max_lines: usize) -> Vec<String> {
        let total = self.suggestions.len();
        if total == 0 {
            return vec!["no matches".to_string()];
        }
        let show = total.min(max_lines);
        // Center viewport on selected
        let half = show / 2;
        let start = if self.selected <= half {
            0
        } else if self.selected + half >= total {
            total.saturating_sub(show)
        } else {
            self.selected - half
        };
        let end = (start + show).min(total);

        (start..end)
            .map(|i| {
                let marker = if i == self.selected { ">" } else { " " };
                format!("{marker} {}", self.suggestions[i])
            })
            .collect()
    }
}

// ─── ML Ghost Text ──────────────────────────────────────────────────────────

/// State for ML-powered ghost text (autocomplete) suggestions.
///
/// Ghost text rendering priority: ML suggestion > history ghost > none.
/// Uses DarkGray italic styling to distinguish from user input.
#[derive(Debug, Clone)]
pub struct GhostTextState {
    /// The current ghost text suggestion, if any.
    pub suggestion: Option<String>,
    /// When the last keystroke occurred (for debounce).
    pub last_keystroke: Instant,
    /// Whether a completion request is pending (debounce not yet elapsed).
    pub pending: bool,
    /// Debounce duration (default 200ms).
    pub debounce_ms: u64,
    /// Minimum input length before triggering completions.
    pub min_input_len: usize,
}

impl Default for GhostTextState {
    fn default() -> Self {
        Self {
            suggestion: None,
            last_keystroke: Instant::now(),
            pending: false,
            debounce_ms: 200,
            min_input_len: 3,
        }
    }
}

impl GhostTextState {
    /// Record a keystroke — clears current suggestion and resets debounce.
    pub fn on_keystroke(&mut self) {
        self.suggestion = None;
        self.last_keystroke = Instant::now();
        self.pending = true;
    }

    /// Check if the debounce period has elapsed and a completion should be requested.
    pub fn should_request(&self, input_len: usize) -> bool {
        self.pending
            && input_len >= self.min_input_len
            && self.last_keystroke.elapsed() >= Duration::from_millis(self.debounce_ms)
    }

    /// Set the suggestion from a completion callback result.
    pub fn set_suggestion(&mut self, text: Option<String>) {
        self.suggestion = text;
        self.pending = false;
    }

    /// Accept the full ghost text suggestion, returning it.
    pub fn accept_full(&mut self) -> Option<String> {
        self.pending = false;
        self.suggestion.take()
    }

    /// Accept one word from the ghost text suggestion.
    ///
    /// Uses word boundary detection that treats alphanumeric and underscore
    /// characters as part of a word (matching identifier conventions).
    pub fn accept_word(&mut self) -> Option<String> {
        if let Some(ref text) = self.suggestion {
            let trimmed = text.trim_start();
            let leading_ws = text.len() - trimmed.len();
            // Find end of the word: alphanumeric or underscore characters
            let word_len = trimmed
                .find(|c: char| !c.is_alphanumeric() && c != '_')
                .unwrap_or(trimmed.len());
            // If we're at a non-word char, take at least one character
            let word_end = leading_ws + word_len.max(1).min(trimmed.len());
            if word_end == 0 {
                return self.accept_full();
            }
            let word = text[..word_end].to_string();
            let remaining = &text[word_end..];
            if remaining.is_empty() {
                self.suggestion = None;
            } else {
                self.suggestion = Some(remaining.to_string());
            }
            self.pending = false;
            Some(word)
        } else {
            None
        }
    }
}

/// Type alias for the ML completion callback.
///
/// Takes the current input text and returns an optional completion suggestion.
pub type MlCompletionCallback = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// Gather file suggestions for an `@` prefix query.
fn autocomplete_at_suggestions(prefix: &str, workspace: &Path) -> Vec<String> {
    // Try to glob for matching files
    let pattern = if prefix.is_empty() {
        // Show top-level files
        "*".to_string()
    } else {
        format!("{prefix}*")
    };

    let search_dir = workspace;
    let full_pattern = search_dir.join(&pattern);
    let matches: Vec<String> = glob::glob(&full_pattern.to_string_lossy())
        .ok()
        .map(|paths| {
            paths
                .filter_map(Result::ok)
                .filter_map(|p| {
                    p.strip_prefix(search_dir).ok().map(|rel| {
                        if p.is_dir() {
                            format!("{}/", rel.display())
                        } else {
                            rel.display().to_string()
                        }
                    })
                })
                .filter(|s| {
                    !s.starts_with('.')
                        && !s.starts_with("target/")
                        && !s.starts_with("node_modules/")
                })
                .take(20)
                .collect()
        })
        .unwrap_or_default();
    matches
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

/// Execute a `!` prefixed shell command directly, bypassing the LLM.
/// Returns the combined stdout/stderr output as a string.
fn execute_bang_command(cmd: &str) -> String {
    let shell = if cfg!(target_os = "windows") {
        "cmd"
    } else {
        "sh"
    };
    let flag = if cfg!(target_os = "windows") {
        "/C"
    } else {
        "-c"
    };
    match std::process::Command::new(shell).args([flag, cmd]).output() {
        Ok(output) => {
            let mut result = String::new();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stdout.is_empty() {
                result.push_str(&stdout);
            }
            if !stderr.is_empty() {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str(&stderr);
            }
            if result.is_empty() {
                format!("(exit code {})", output.status.code().unwrap_or(-1))
            } else {
                // Trim trailing newline for cleaner display
                result.truncate(result.trim_end().len());
                result
            }
        }
        Err(e) => format!("Error: {e}"),
    }
}

/// Try to read an image from the system clipboard.
/// Returns the raw PNG bytes if an image is available, `None` otherwise.
fn try_clipboard_image() -> Option<Vec<u8>> {
    #[cfg(target_os = "macos")]
    {
        // Use osascript to check if clipboard contains image data
        let check = std::process::Command::new("osascript")
            .args(["-e", "clipboard info for (the clipboard as «class PNGf»)"])
            .output()
            .ok()?;
        if !check.status.success() {
            return None;
        }

        // Write clipboard image to a temp file via osascript and read it back
        let tmp = std::env::temp_dir().join("codingbuddy-clipboard-check.png");
        // Write clipboard PNG data to temp file via osascript
        let output = std::process::Command::new("osascript")
            .args([
                "-e",
                &format!(
                    "set f to open for access POSIX file \"{}\" with write permission\n\
                     set eof of f to 0\n\
                     write (the clipboard as «class PNGf») to f\n\
                     close access f",
                    tmp.display()
                ),
            ])
            .output()
            .ok()?;
        if output.status.success() {
            let data = std::fs::read(&tmp).ok()?;
            let _ = std::fs::remove_file(&tmp);
            if data.len() > 8 {
                return Some(data);
            }
        }
        None
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Autocomplete `@path` file mentions. Finds the last `@` token in the input and
/// completes the path after it using the same logic as path autocomplete.
fn autocomplete_at_mention(input: &str) -> Option<String> {
    // Find the last token starting with '@'
    let split_at = input
        .char_indices()
        .rfind(|(_, ch)| ch.is_whitespace())
        .map(|(idx, _)| idx + 1)
        .unwrap_or(0);
    let token = &input[split_at..];
    if !token.starts_with('@') || token.len() < 2 {
        return None;
    }
    let path_part = &token[1..]; // strip the '@'
    let completed = autocomplete_path_token(path_part)?;
    let mut out = String::with_capacity(input.len() + completed.len() + 2);
    out.push_str(&input[..split_at]);
    out.push('@');
    out.push_str(&completed);
    Some(out)
}

/// Expand `@file` mentions in a prompt into inline file content references.
/// Returns the prompt with `@path` replaced by `[file: path]\n<content>\n[/file]`.
pub fn expand_at_mentions(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    for token in input.split_whitespace() {
        if !result.is_empty() {
            result.push(' ');
        }
        if token.starts_with('@') && token.len() > 1 {
            let raw_path = &token[1..];
            let path = if let Some(stripped) = raw_path.strip_prefix("~/") {
                let home = std::env::var("HOME")
                    .or_else(|_| std::env::var("USERPROFILE"))
                    .unwrap_or_default();
                PathBuf::from(home).join(stripped)
            } else {
                let p = PathBuf::from(raw_path);
                if p.is_absolute() {
                    p
                } else {
                    std::env::current_dir().unwrap_or_default().join(p)
                }
            };
            if path.is_file()
                && let Ok(content) = fs::read_to_string(&path)
            {
                result.push_str(&format!(
                    "[file: {}]\n{}\n[/file]",
                    path.display(),
                    content.trim()
                ));
                continue;
            }
        }
        result.push_str(token);
    }
    result
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

fn history_ghost_suffix(history: &VecDeque<String>, input: &str) -> Option<String> {
    if input.is_empty() {
        return None;
    }
    for entry in history.iter().rev() {
        if entry.starts_with(input) && entry.len() > input.len() {
            return Some(entry[input.len()..].to_string());
        }
    }
    None
}

fn history_reverse_search_index(
    history: &VecDeque<String>,
    query: &str,
    before: Option<usize>,
) -> Option<usize> {
    if history.is_empty() {
        return None;
    }
    let matches = |value: &String| query.is_empty() || value.contains(query);

    if let Some(current) = before {
        if current > 0 {
            for idx in (0..current).rev() {
                if matches(&history[idx]) {
                    return Some(idx);
                }
            }
        }
        for idx in ((current + 1).min(history.len())..history.len()).rev() {
            if matches(&history[idx]) {
                return Some(idx);
            }
        }
        return None;
    }

    (0..history.len()).rev().find(|&idx| matches(&history[idx]))
}

fn apply_reverse_search_result(
    history: &VecDeque<String>,
    index: Option<usize>,
    input: &mut String,
    cursor_pos: &mut usize,
) {
    if let Some(idx) = index
        && let Some(entry) = history.get(idx)
    {
        *input = entry.clone();
        *cursor_pos = input.len();
    }
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

#[allow(clippy::too_many_arguments)]
fn apply_vim_command(
    command: VimSlashCommand,
    vim_enabled: &mut bool,
    vim_mode: &mut VimMode,
    vim_command_buffer: &mut String,
    vim_visual_anchor: &mut Option<usize>,
    vim_pending_operator: &mut Option<char>,
    vim_pending_text_object: &mut Option<(char, char)>,
    vim_pending_g: &mut bool,
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
    *vim_pending_text_object = None;
    *vim_pending_g = false;
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

fn display_key_code(code: KeyCode) -> String {
    match code {
        KeyCode::Char(ch) => ch.to_string(),
        KeyCode::Left => "left".to_string(),
        KeyCode::Right => "right".to_string(),
        KeyCode::Up => "up".to_string(),
        KeyCode::Down => "down".to_string(),
        KeyCode::Esc => "esc".to_string(),
        KeyCode::Enter => "enter".to_string(),
        _ => "?".to_string(),
    }
}

fn apply_vim_operator_range(
    operator: char,
    input: &mut String,
    cursor_pos: &mut usize,
    vim_yank_buffer: &mut String,
    vim_mode: &mut VimMode,
    start: usize,
    end: usize,
) -> bool {
    if end <= start || start >= input.len() || end > input.len() {
        return false;
    }
    let selected = input[start..end].to_string();
    match operator {
        'y' => {
            *vim_yank_buffer = selected;
        }
        'd' => {
            input.replace_range(start..end, "");
            *cursor_pos = start.min(input.len());
        }
        'c' => {
            input.replace_range(start..end, "");
            *cursor_pos = start.min(input.len());
            *vim_mode = VimMode::Insert;
        }
        _ => return false,
    }
    true
}

fn word_text_object_bounds(input: &str, cursor_pos: usize, around: bool) -> Option<(usize, usize)> {
    let bytes = input.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let mut cursor = cursor_pos.min(bytes.len().saturating_sub(1));
    if !is_word_byte(bytes[cursor]) {
        if let Some(next) = (cursor..bytes.len()).find(|idx| is_word_byte(bytes[*idx])) {
            cursor = next;
        } else if let Some(prev) = (0..=cursor).rev().find(|idx| is_word_byte(bytes[*idx])) {
            cursor = prev;
        } else {
            return None;
        }
    }

    let mut start = cursor;
    while start > 0 && is_word_byte(bytes[start - 1]) {
        start -= 1;
    }
    let mut end = cursor + 1;
    while end < bytes.len() && is_word_byte(bytes[end]) {
        end += 1;
    }

    if around {
        if end < bytes.len() && bytes[end].is_ascii_whitespace() {
            while end < bytes.len() && bytes[end].is_ascii_whitespace() {
                end += 1;
            }
        } else {
            while start > 0 && bytes[start - 1].is_ascii_whitespace() {
                start -= 1;
            }
        }
    }
    Some((start, end))
}

fn delimiter_text_object_bounds(
    input: &str,
    cursor_pos: usize,
    open: u8,
    close: u8,
    around: bool,
) -> Option<(usize, usize)> {
    let bytes = input.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let cursor = cursor_pos.min(bytes.len().saturating_sub(1));
    let left = (0..=cursor).rev().find(|idx| bytes[*idx] == open)?;
    let right = ((left + 1)..bytes.len()).find(|idx| bytes[*idx] == close)?;
    if around {
        Some((left, right + 1))
    } else if right > left + 1 {
        Some((left + 1, right))
    } else {
        None
    }
}

fn resolve_vim_text_object_bounds(
    input: &str,
    cursor_pos: usize,
    scope: char,
    key: KeyCode,
) -> Option<(usize, usize)> {
    let around = scope == 'a';
    match key {
        KeyCode::Char('w') => word_text_object_bounds(input, cursor_pos, around),
        KeyCode::Char('"') => delimiter_text_object_bounds(input, cursor_pos, b'"', b'"', around),
        KeyCode::Char('\'') => {
            delimiter_text_object_bounds(input, cursor_pos, b'\'', b'\'', around)
        }
        KeyCode::Char('(') | KeyCode::Char(')') => {
            delimiter_text_object_bounds(input, cursor_pos, b'(', b')', around)
        }
        KeyCode::Char('[') | KeyCode::Char(']') => {
            delimiter_text_object_bounds(input, cursor_pos, b'[', b']', around)
        }
        KeyCode::Char('{') | KeyCode::Char('}') => {
            delimiter_text_object_bounds(input, cursor_pos, b'{', b'}', around)
        }
        _ => None,
    }
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
mod tests;

// ─── Terminal Image Rendering ───────────────────────────────────────────────

/// Terminal image protocol support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageProtocol {
    /// iTerm2 inline image protocol (also supported by WezTerm, mintty).
    Iterm2,
    /// Kitty graphics protocol.
    Kitty,
    /// No inline image support — use ASCII placeholder.
    None,
}

/// Detect which image protocol the current terminal supports.
pub fn detect_image_protocol() -> ImageProtocol {
    // iTerm2 and WezTerm set TERM_PROGRAM
    if let Ok(program) = std::env::var("TERM_PROGRAM") {
        let lower = program.to_ascii_lowercase();
        if lower.contains("iterm") || lower.contains("wezterm") || lower.contains("mintty") {
            return ImageProtocol::Iterm2;
        }
    }
    // Kitty sets TERM=xterm-kitty or TERM_PROGRAM=kitty
    if let Ok(term) = std::env::var("TERM")
        && term.contains("kitty")
    {
        return ImageProtocol::Kitty;
    }
    if let Ok(program) = std::env::var("TERM_PROGRAM")
        && program.to_ascii_lowercase().contains("kitty")
    {
        return ImageProtocol::Kitty;
    }
    // KITTY_WINDOW_ID is set inside Kitty
    if std::env::var("KITTY_WINDOW_ID").is_ok() {
        return ImageProtocol::Kitty;
    }
    ImageProtocol::None
}

/// Render an image inline in the terminal.
/// `data` is the raw image bytes. Returns the escape sequence to write.
pub fn render_inline_image(data: &[u8], protocol: ImageProtocol) -> Option<String> {
    use base64::Engine;
    let engine = base64::engine::general_purpose::STANDARD;

    match protocol {
        ImageProtocol::Iterm2 => {
            let b64 = engine.encode(data);
            // iTerm2 protocol: ESC ] 1337 ; File=[args] : <base64> BEL
            Some(format!(
                "\x1b]1337;File=inline=1;size={};preserveAspectRatio=1:{}\x07",
                data.len(),
                b64
            ))
        }
        ImageProtocol::Kitty => {
            let b64 = engine.encode(data);
            // Kitty protocol: send in chunks of 4096 bytes
            let mut output = String::new();
            let chunks: Vec<&str> = {
                let mut v = Vec::new();
                let mut i = 0;
                while i < b64.len() {
                    let end = (i + 4096).min(b64.len());
                    v.push(&b64[i..end]);
                    i = end;
                }
                v
            };
            for (idx, chunk) in chunks.iter().enumerate() {
                let more = if idx < chunks.len() - 1 { 1 } else { 0 };
                if idx == 0 {
                    // First chunk: action=transmit+display, format=100 (auto-detect)
                    output.push_str(&format!("\x1b_Ga=T,f=100,m={more};{chunk}\x1b\\"));
                } else {
                    // Continuation chunks
                    output.push_str(&format!("\x1b_Gm={more};{chunk}\x1b\\"));
                }
            }
            Some(output)
        }
        ImageProtocol::None => None,
    }
}

/// Print an image inline to stdout, if the terminal supports it.
/// Returns true if the image was displayed, false if no protocol is available.
pub fn display_image_inline(data: &[u8]) -> bool {
    let protocol = detect_image_protocol();
    if let Some(escape) = render_inline_image(data, protocol) {
        use std::io::Write;
        let mut out = io::stdout();
        let _ = out.write_all(escape.as_bytes());
        let _ = writeln!(out);
        let _ = out.flush();
        true
    } else {
        false
    }
}
