use super::state::{ChatShell, MessageKind, TranscriptEntry};
use super::*;
use syntect::highlighting::{Theme as SyntectTheme, ThemeSet};
use syntect::parsing::SyntaxSet;

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
pub(crate) struct TerminalGuard;

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
pub(crate) fn syntax_keyword_color(word: &str) -> Option<Color> {
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
pub(crate) fn extract_fence_lang(fence: &str) -> &str {
    fence.trim().trim_start_matches('`').trim()
}

/// Apply syntax highlighting to a line of code inside a code block.
///
/// Uses `syntect` when a matching syntax definition exists for the given
/// `lang` (the language tag from the opening code fence).  Falls back to the
/// keyword-based highlighter for unknown languages.
pub(crate) fn highlight_code_line(line: &str, lang: &str) -> Line<'static> {
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
pub(crate) fn highlight_code_line_fallback(line: &str) -> Line<'static> {
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
pub(crate) fn parse_inline_markdown(text: &str, base_style: Style) -> Vec<Span<'static>> {
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
pub(crate) fn render_assistant_markdown(text: &str) -> Line<'static> {
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

pub(crate) fn code_fence_starts_diff(line: &str) -> bool {
    let trimmed = line.trim();
    if !trimmed.starts_with("```") {
        return false;
    }
    let lang = trimmed.trim_start_matches('`').trim().to_ascii_lowercase();
    lang.starts_with("diff") || lang.starts_with("patch")
}

pub(crate) fn looks_like_unfenced_diff_line(line: &str) -> bool {
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

pub(crate) fn render_diff_line(line: &str) -> Line<'static> {
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

pub(crate) const DIFF_CONTEXT_COLLAPSE_THRESHOLD: usize = 12;
pub(crate) const DIFF_CONTEXT_KEEP_HEAD: usize = 2;
pub(crate) const DIFF_CONTEXT_KEEP_TAIL: usize = 2;
pub(crate) const INPUT_MIN_HEIGHT: u16 = 1;
pub(crate) const INPUT_MAX_HEIGHT: u16 = 4;
pub(crate) const STREAM_MIN_HEIGHT: u16 = 1;
pub(crate) const INLINE_VIEWPORT_HEIGHT: u16 = INPUT_MAX_HEIGHT + STREAM_MIN_HEIGHT + 3;

pub(crate) fn flush_diff_context_run(
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

pub(crate) fn wrapped_line_height(line: &Line<'_>, width: u16) -> u16 {
    if width == 0 {
        return 0;
    }
    let width = width as usize;
    let content_width = line.width().max(1);
    let rows = (content_width.saturating_sub(1) / width) + 1;
    u16::try_from(rows).unwrap_or(u16::MAX)
}

pub(crate) fn wrapped_text_rows(text: &str, width: u16) -> usize {
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

pub(crate) fn scroll_to_keep_row_visible(row: usize, viewport_rows: u16) -> u16 {
    if viewport_rows == 0 {
        return 0;
    }
    row.saturating_sub(viewport_rows.saturating_sub(1) as usize)
        .min(u16::MAX as usize) as u16
}

pub(crate) fn compute_inline_heights(total_height: u16, desired_input_rows: usize) -> (u16, u16) {
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

pub(crate) fn operator_summary_line(status: &UiStatus) -> String {
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
    let provider_diagnostics = if status.provider_diagnostics_summary.trim().is_empty() {
        String::new()
    } else {
        format!(
            " compat={}",
            truncate_inline(&status.provider_diagnostics_summary, 40)
        )
    };
    let runtime_diagnostics = if status.runtime_diagnostics_summary.trim().is_empty() {
        String::new()
    } else {
        format!(
            " runtime={}",
            truncate_inline(&status.runtime_diagnostics_summary, 40)
        )
    };
    format!(
        "phase={phase} step={step} todo={todo} subagents={}/{} bg={} {} caps={}{}{} compaction/replay={}/{}",
        status.running_subagents,
        status.failed_subagents,
        status.running_background_jobs.max(status.background_jobs),
        blockers,
        capabilities,
        provider_diagnostics,
        runtime_diagnostics,
        status.compaction_count,
        status.replay_count
    )
}

pub(crate) fn compact_tool_meta_tail(tail: &str, max_chars: usize) -> String {
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

pub(crate) fn parse_stream_meta_entry(text: &str) -> Option<TranscriptEntry> {
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

pub(crate) fn style_transcript_line(
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
pub(crate) fn style_continuation_line(
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

pub(crate) fn split_inline_markdown_blocks(text: &str) -> Vec<String> {
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

pub(crate) fn outside_inline_code(text: &str, idx: usize) -> bool {
    text[..idx].chars().filter(|c| *c == '`').count() % 2 == 0
}

pub(crate) fn is_run_on_section_boundary(prefix: &str) -> bool {
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

pub(crate) fn is_plan_list_line(line: &str) -> bool {
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

pub(crate) fn insert_wrapped_lines_above(
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
pub(crate) fn flush_transcript_above(
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
pub(crate) fn flush_streaming_lines(
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
