//! Streaming markdown-to-ANSI renderer for the non-interactive (readline) chat path.
//!
//! `StreamingMdRenderer` buffers streaming LLM tokens and renders complete lines
//! with ANSI formatting as they arrive, providing real-time formatted output that
//! matches the TUI's `render_assistant_markdown` quality.

use std::io::{self, Write};

// ── ANSI escape helpers ─────────────────────────────────────────────────

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const ITALIC: &str = "\x1b[3m";
const UNDERLINE: &str = "\x1b[4m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const WHITE: &str = "\x1b[37m";
const GRAY: &str = "\x1b[90m";
const RED: &str = "\x1b[31m";
const MAGENTA: &str = "\x1b[35m";
const BG_GRAY: &str = "\x1b[48;5;236m";

// ── StreamingMdRenderer ─────────────────────────────────────────────────

/// Incremental markdown-to-ANSI renderer.
///
/// Feed streaming tokens via [`push`]. Complete lines are rendered immediately
/// with ANSI formatting. Call [`flush_remaining`] after streaming ends to emit
/// any trailing partial line.
pub struct StreamingMdRenderer {
    buffer: String,
    in_code_block: bool,
    code_lang: String,
    line_count: usize,
}

impl StreamingMdRenderer {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            in_code_block: false,
            code_lang: String::new(),
            line_count: 0,
        }
    }

    /// Feed streaming text into the renderer. Complete lines are rendered
    /// immediately; partial lines are buffered.
    pub fn push(&mut self, text: &str) {
        self.buffer.push_str(text);
        self.flush_complete_lines();
    }

    /// Render any remaining buffered text (partial line at end of stream).
    pub fn flush_remaining(&mut self) {
        if !self.buffer.is_empty() {
            let line = std::mem::take(&mut self.buffer);
            self.render_line(&line);
        }
    }

    /// Flush and render all complete lines (text before the last `\n`).
    fn flush_complete_lines(&mut self) {
        while let Some(nl_pos) = self.buffer.find('\n') {
            let line = self.buffer[..nl_pos].to_string();
            self.buffer = self.buffer[nl_pos + 1..].to_string();
            self.render_line(&line);
        }
    }

    /// Render a single complete line with markdown-to-ANSI formatting.
    fn render_line(&mut self, line: &str) {
        let out = io::stdout();
        let mut w = out.lock();
        self.line_count += 1;

        // ── Code fence toggle ───────────────────────────────────────
        if line.starts_with("```") {
            if !self.in_code_block {
                self.in_code_block = true;
                self.code_lang = line.trim_start_matches('`').trim().to_string();
                let lang_label = if self.code_lang.is_empty() {
                    String::new()
                } else {
                    format!(" {}", self.code_lang)
                };
                let _ = writeln!(w, "  {DIM}{GRAY}┌──{lang_label}──{RESET}");
            } else {
                self.in_code_block = false;
                self.code_lang.clear();
                let _ = writeln!(w, "  {DIM}{GRAY}└────────{RESET}");
            }
            return;
        }

        // ── Inside code block — render with dim gray + line prefix ──
        if self.in_code_block {
            let _ = writeln!(w, "  {DIM}{GRAY}│{RESET} {DIM}{WHITE}{line}{RESET}");
            return;
        }

        // ── Horizontal rule ─────────────────────────────────────────
        if line == "---" || line == "***" || line == "___" {
            let _ = writeln!(
                w,
                "  {GRAY}─────────────────────────────────────────{RESET}"
            );
            return;
        }

        // ── Headings ────────────────────────────────────────────────
        if line.starts_with("# ") || (line.starts_with('#') && !line.starts_with("#!")) {
            let trimmed = line.trim_start_matches('#');
            let level = line.len() - trimmed.len();
            let heading = trimmed.strip_prefix(' ').unwrap_or(trimmed);
            if !heading.is_empty() {
                let styled = render_inline_markdown(heading);
                match level {
                    1 => {
                        let _ = writeln!(w);
                        let _ = writeln!(w, "  {BOLD}{CYAN}{UNDERLINE}{styled}{RESET}");
                        let _ = writeln!(w);
                    }
                    2 => {
                        let _ = writeln!(w);
                        let _ = writeln!(w, "  {BOLD}{CYAN}{styled}{RESET}");
                    }
                    3 => {
                        let _ = writeln!(w, "  {CYAN}{styled}{RESET}");
                    }
                    _ => {
                        let _ = writeln!(w, "  {BOLD}{WHITE}{styled}{RESET}");
                    }
                }
                return;
            }
        }

        // ── Blockquote ──────────────────────────────────────────────
        if let Some(quote) = line.strip_prefix("> ") {
            let styled = render_inline_markdown(quote);
            let _ = writeln!(w, "  {GRAY}│{RESET} {DIM}{styled}{RESET}");
            return;
        }

        // ── Task list ───────────────────────────────────────────────
        if let Some(rest) = line
            .strip_prefix("- [ ] ")
            .or_else(|| line.strip_prefix("* [ ] "))
        {
            let styled = render_inline_markdown(rest);
            let _ = writeln!(w, "  {GRAY}☐{RESET} {styled}");
            return;
        }
        if let Some(rest) = line
            .strip_prefix("- [x] ")
            .or_else(|| line.strip_prefix("* [x] "))
            .or_else(|| line.strip_prefix("- [X] "))
            .or_else(|| line.strip_prefix("* [X] "))
        {
            let styled = render_inline_markdown(rest);
            let _ = writeln!(w, "  {GREEN}☑{RESET} {DIM}{styled}{RESET}");
            return;
        }

        // ── Unordered list ──────────────────────────────────────────
        if let Some(item) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
            let styled = render_inline_markdown(item);
            let _ = writeln!(w, "  {CYAN}•{RESET} {styled}");
            return;
        }

        // ── Indented sub-list (2-4 spaces + bullet) ─────────────────
        let stripped = line.trim_start();
        let indent_len = line.len() - stripped.len();
        if indent_len >= 2 {
            if let Some(item) = stripped
                .strip_prefix("- ")
                .or_else(|| stripped.strip_prefix("* "))
            {
                let indent = " ".repeat(indent_len);
                let styled = render_inline_markdown(item);
                let _ = writeln!(w, "  {indent}{CYAN}◦{RESET} {styled}");
                return;
            }
        }

        // ── Numbered list ───────────────────────────────────────────
        if let Some(dot_pos) = line.find(". ") {
            if dot_pos <= 4 && line[..dot_pos].chars().all(|c| c.is_ascii_digit()) {
                let num = &line[..dot_pos + 2];
                let item = &line[dot_pos + 2..];
                let styled = render_inline_markdown(item);
                let _ = writeln!(w, "  {CYAN}{num}{RESET}{styled}");
                return;
            }
        }

        // ── Table rows ──────────────────────────────────────────────
        if line.len() >= 2 && line.starts_with('|') && line.ends_with('|') {
            let inner = &line[1..line.len() - 1];
            // Separator row
            if inner
                .chars()
                .all(|c| c == '-' || c == '|' || c == ':' || c == ' ')
            {
                let _ = writeln!(w, "  {GRAY}{line}{RESET}");
                return;
            }
            // Data row — highlight separators
            let cells: Vec<&str> = inner.split('|').collect();
            let _ = write!(w, "  {GRAY}│{RESET}");
            for cell in cells {
                let styled = render_inline_markdown(cell.trim());
                let _ = write!(w, " {styled} {GRAY}│{RESET}");
            }
            let _ = writeln!(w);
            return;
        }

        // ── Blank line ──────────────────────────────────────────────
        if line.trim().is_empty() {
            let _ = writeln!(w);
            return;
        }

        // ── Regular paragraph text with inline markdown ─────────────
        let styled = render_inline_markdown(line);
        let _ = writeln!(w, "  {styled}");
    }
}

// ── Inline markdown rendering ───────────────────────────────────────────

/// Render inline markdown elements (bold, italic, code, strikethrough) to ANSI.
fn render_inline_markdown(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 64);
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Bold+italic: ***text***
        if i + 2 < len && chars[i] == '*' && chars[i + 1] == '*' && chars[i + 2] == '*' {
            if let Some(end) = find_closing(&chars, i + 3, &['*', '*', '*']) {
                let inner: String = chars[i + 3..end].iter().collect();
                out.push_str(&format!("{BOLD}{ITALIC}{inner}{RESET}"));
                i = end + 3;
                continue;
            }
        }

        // Bold: **text**
        if i + 1 < len && chars[i] == '*' && chars[i + 1] == '*' {
            if let Some(end) = find_closing(&chars, i + 2, &['*', '*']) {
                let inner: String = chars[i + 2..end].iter().collect();
                out.push_str(&format!("{BOLD}{inner}{RESET}"));
                i = end + 2;
                continue;
            }
        }

        // Bold (underscores): __text__
        if i + 1 < len && chars[i] == '_' && chars[i + 1] == '_' {
            if let Some(end) = find_closing(&chars, i + 2, &['_', '_']) {
                let inner: String = chars[i + 2..end].iter().collect();
                out.push_str(&format!("{BOLD}{inner}{RESET}"));
                i = end + 2;
                continue;
            }
        }

        // Italic: *text* (single asterisk, not followed by another)
        if chars[i] == '*' && (i + 1 >= len || chars[i + 1] != '*') {
            if let Some(end) = find_closing(&chars, i + 1, &['*']) {
                let inner: String = chars[i + 1..end].iter().collect();
                out.push_str(&format!("{ITALIC}{inner}{RESET}"));
                i = end + 1;
                continue;
            }
        }

        // Italic (underscore): _text_
        if chars[i] == '_' && (i + 1 >= len || chars[i + 1] != '_') {
            if let Some(end) = find_closing(&chars, i + 1, &['_']) {
                // Avoid matching snake_case identifiers
                if end > i + 1 {
                    let inner: String = chars[i + 1..end].iter().collect();
                    if !inner.contains(' ') && inner.chars().any(|c| c == '_') {
                        // Likely snake_case — don't treat as italic
                        out.push(chars[i]);
                        i += 1;
                        continue;
                    }
                    out.push_str(&format!("{ITALIC}{inner}{RESET}"));
                    i = end + 1;
                    continue;
                }
            }
        }

        // Strikethrough: ~~text~~
        if i + 1 < len && chars[i] == '~' && chars[i + 1] == '~' {
            if let Some(end) = find_closing(&chars, i + 2, &['~', '~']) {
                let inner: String = chars[i + 2..end].iter().collect();
                out.push_str(&format!("\x1b[9m{inner}{RESET}"));
                i = end + 2;
                continue;
            }
        }

        // Inline code: `text`
        if chars[i] == '`' {
            if let Some(end) = find_single_closing(&chars, i + 1, '`') {
                let inner: String = chars[i + 1..end].iter().collect();
                out.push_str(&format!("{BG_GRAY}{YELLOW} {inner} {RESET}"));
                i = end + 1;
                continue;
            }
        }

        out.push(chars[i]);
        i += 1;
    }

    out
}

/// Find closing delimiter sequence starting at `start`.
fn find_closing(chars: &[char], start: usize, delim: &[char]) -> Option<usize> {
    let dlen = delim.len();
    if chars.len() < start + dlen {
        return None;
    }
    for i in start..=(chars.len() - dlen) {
        if chars[i..i + dlen] == *delim {
            return Some(i);
        }
    }
    None
}

/// Find single closing character.
fn find_single_closing(chars: &[char], start: usize, ch: char) -> Option<usize> {
    for i in start..chars.len() {
        if chars[i] == ch {
            return Some(i);
        }
    }
    None
}

// ── Role headers ────────────────────────────────────────────────────────

/// Print a styled role header before an assistant response.
pub fn print_role_header(role: &str, model: &str) {
    let out = io::stdout();
    let mut w = out.lock();
    let _ = writeln!(w);
    let label = if model.is_empty() {
        format!(" {role} ")
    } else {
        format!(" {role} ({model}) ")
    };
    let _ = writeln!(w, "{CYAN}{BOLD}╭─{label}─╮{RESET}");
}

/// Print a styled separator after an assistant response.
pub fn print_role_footer() {
    let out = io::stdout();
    let mut w = out.lock();
    let _ = writeln!(w, "{CYAN}╰──────────────────╯{RESET}");
    let _ = writeln!(w);
}

/// Print a styled phase indicator (architect/editor/verify/lint).
pub fn print_phase(icon: &str, label: &str, detail: &str) {
    let out = io::stdout();
    let mut w = out.lock();
    if detail.is_empty() {
        let _ = writeln!(w, "  {MAGENTA}{icon}{RESET}  {BOLD}{label}{RESET}");
    } else {
        let _ = writeln!(
            w,
            "  {MAGENTA}{icon}{RESET}  {BOLD}{label}{RESET} {GRAY}({detail}){RESET}"
        );
    }
}

/// Print a phase completion indicator.
pub fn print_phase_done(success: bool, label: &str, detail: &str) {
    let out = io::stdout();
    let mut w = out.lock();
    let (icon, color) = if success {
        ("✓", GREEN)
    } else {
        ("✗", RED)
    };
    if detail.is_empty() {
        let _ = writeln!(w, "  {color}{icon}{RESET}  {label}");
    } else {
        let _ = writeln!(w, "  {color}{icon}{RESET}  {label} {GRAY}({detail}){RESET}");
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_bold() {
        let result = render_inline_markdown("hello **world**");
        assert!(result.contains("\x1b[1m"));
        assert!(result.contains("world"));
    }

    #[test]
    fn inline_italic() {
        let result = render_inline_markdown("hello *world*");
        assert!(result.contains("\x1b[3m"));
        assert!(result.contains("world"));
    }

    #[test]
    fn inline_code() {
        let result = render_inline_markdown("use `cargo test`");
        assert!(result.contains("\x1b[33m")); // yellow
        assert!(result.contains("cargo test"));
    }

    #[test]
    fn inline_bold_italic() {
        let result = render_inline_markdown("***emphasis***");
        assert!(result.contains("\x1b[1m\x1b[3m")); // bold+italic
    }

    #[test]
    fn streaming_code_block() {
        let mut renderer = StreamingMdRenderer::new();
        renderer.push("hello\n```rust\nfn main() {}\n```\nbye\n");
        assert_eq!(renderer.line_count, 5);
        assert!(!renderer.in_code_block);
    }

    #[test]
    fn streaming_partial_lines() {
        let mut renderer = StreamingMdRenderer::new();
        renderer.push("hel");
        assert_eq!(renderer.line_count, 0);
        assert_eq!(renderer.buffer, "hel");
        renderer.push("lo\nworld\n");
        assert_eq!(renderer.line_count, 2);
        assert!(renderer.buffer.is_empty());
    }

    #[test]
    fn heading_levels() {
        let mut renderer = StreamingMdRenderer::new();
        renderer.push("# H1\n## H2\n### H3\n#### H4\n");
        assert_eq!(renderer.line_count, 4);
    }

    #[test]
    fn list_rendering() {
        let mut renderer = StreamingMdRenderer::new();
        renderer.push("- item one\n- item two\n1. numbered\n");
        assert_eq!(renderer.line_count, 3);
    }

    #[test]
    fn flush_remaining_partial() {
        let mut renderer = StreamingMdRenderer::new();
        renderer.push("partial line no newline");
        assert_eq!(renderer.line_count, 0);
        renderer.flush_remaining();
        assert_eq!(renderer.line_count, 1);
    }
}
