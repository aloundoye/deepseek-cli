use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Tabs, Wrap};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

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

fn render_statusline_spans(status: &UiStatus) -> Vec<Span<'static>> {
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

    let mut spans = vec![
        Span::styled(
            format!(" {} ", status.model),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(
            mode_label.to_string(),
            Style::default()
                .fg(Color::Black)
                .bg(mode_color)
                .add_modifier(Modifier::BOLD),
        ),
    ];

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

    spans.push(Span::styled(
        format!(" ${:.4} ", status.estimated_cost_usd),
        Style::default().fg(Color::DarkGray),
    ));

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

    // Simple markdown-like styling for assistant messages
    if entry.kind == MessageKind::Assistant {
        if text.starts_with("```") {
            return Line::from(vec![Span::styled(
                text.clone(),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::DIM),
            )]);
        }
        if text.starts_with("# ") || text.starts_with("## ") || text.starts_with("### ") {
            return Line::from(vec![Span::styled(
                text.clone(),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )]);
        }
        if text.starts_with("- ") || text.starts_with("* ") {
            return Line::from(vec![
                Span::styled("  • ".to_string(), Style::default().fg(Color::Cyan)),
                Span::styled(text[2..].to_string(), body_style),
            ]);
        }
    }

    Line::from(vec![
        Span::styled(prefix.to_string(), prefix_style),
        Span::styled(text.clone(), body_style),
    ])
}

fn render_context_gauge(status: &UiStatus, area: Rect, frame: &mut ratatui::Frame<'_>) {
    if status.context_max_tokens == 0 {
        return;
    }
    let ratio =
        (status.context_used_tokens as f64 / status.context_max_tokens as f64).clamp(0.0, 1.0);
    let color = if ratio > 0.8 {
        Color::Red
    } else if ratio > 0.6 {
        Color::Yellow
    } else {
        Color::Green
    };
    let gauge = Gauge::default()
        .block(Block::default())
        .gauge_style(Style::default().fg(color).bg(Color::DarkGray))
        .ratio(ratio)
        .label(format!(
            "Context: {}K / {}K ({:.0}%)",
            status.context_used_tokens / 1000,
            status.context_max_tokens / 1000,
            ratio * 100.0
        ));
    frame.render_widget(gauge, area);
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

impl RightPane {
    fn title(self) -> &'static str {
        match self {
            Self::Plan => "Plan",
            Self::Tools => "Tools",
            Self::MissionControl => "Mission Control",
            Self::Artifacts => "Artifacts",
        }
    }

    fn cycle(self) -> Self {
        match self {
            Self::Plan => Self::Tools,
            Self::Tools => Self::MissionControl,
            Self::MissionControl => Self::Artifacts,
            Self::Artifacts => Self::Plan,
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
    run_tui_shell_with_bindings(
        status,
        KeyBindings::default(),
        TuiTheme::default(),
        move |prompt| on_submit(prompt),
    )
}

pub fn run_tui_shell_with_bindings<F>(
    mut status: UiStatus,
    bindings: KeyBindings,
    theme: TuiTheme,
    mut on_submit: F,
) -> Result<()>
where
    F: FnMut(&str) -> Result<String>,
{
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut shell = ChatShell::default();
    let mut input = String::new();
    let mut info_line = String::from(
        "Ctrl+C exit | Tab autocomplete | Ctrl+O toggle pane | Ctrl+B background | Shift+Enter newline",
    );
    let mut right_pane = RightPane::Plan;
    let mut right_pane_collapsed = false;
    let mut history: VecDeque<String> = VecDeque::new();
    let mut last_escape_at: Option<Instant> = None;
    let mut cursor_visible;
    let mut tick_count: usize = 0;

    loop {
        tick_count = tick_count.wrapping_add(1);
        shell.spinner_tick = tick_count;
        cursor_visible = tick_count % 8 < 5;

        terminal.draw(|frame| {
            let area = frame.area();

            // Main vertical layout:
            //  [context gauge]  (1 line)
            //  [body area]      (fills)
            //  [tool output]    (20%)
            //  [input]          (3 lines)
            //  [status bar]     (1 line)
            //  [info/help]      (1 line)
            let vertical = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1),      // context gauge
                    Constraint::Min(8),         // body (transcript + side pane)
                    Constraint::Percentage(18), // tool output
                    Constraint::Length(3),      // input
                    Constraint::Length(1),      // status bar
                    Constraint::Length(1),      // info/help line
                ])
                .split(area);

            // Context usage gauge
            render_context_gauge(&status, vertical[0], frame);

            // Body: Transcript (left) + Right pane (collapsible)
            let body = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(if right_pane_collapsed {
                    [Constraint::Percentage(100), Constraint::Percentage(0)]
                } else {
                    [Constraint::Percentage(72), Constraint::Percentage(28)]
                })
                .split(vertical[1]);

            // Transcript with styled entries and syntax highlighting
            let visible_entries: Vec<&TranscriptEntry> = shell
                .transcript
                .iter()
                .rev()
                .take(500)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            let mut in_code_block = false;
            let transcript_lines: Vec<Line<'_>> = visible_entries
                .iter()
                .map(|entry| {
                    if entry.kind == MessageKind::Assistant && entry.text.starts_with("```") {
                        in_code_block = !in_code_block;
                    }
                    style_transcript_line(entry, in_code_block)
                })
                .collect();

            let transcript_title = if let Some(ref tool) = shell.active_tool {
                format!(" {} {} ", shell.spinner_frame(), tool)
            } else {
                " Transcript ".to_string()
            };

            frame.render_widget(
                Paragraph::new(transcript_lines)
                    .block(
                        Block::default()
                            .title(Span::styled(
                                transcript_title,
                                Style::default()
                                    .fg(theme.primary)
                                    .add_modifier(Modifier::BOLD),
                            ))
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(if shell.active_tool.is_some() {
                                Color::Yellow
                            } else {
                                theme.primary
                            })),
                    )
                    .wrap(Wrap { trim: false }),
                body[0],
            );

            // Right pane (Plan / Tools / Mission Control / Artifacts)
            let right_lines: Vec<Line<'_>> = match right_pane {
                RightPane::Plan => shell
                    .plan_lines
                    .iter()
                    .map(|l| {
                        if l.starts_with("##") {
                            Line::from(Span::styled(
                                l.clone(),
                                Style::default()
                                    .fg(Color::Cyan)
                                    .add_modifier(Modifier::BOLD),
                            ))
                        } else if l.starts_with("- [x]") {
                            Line::from(Span::styled(l.clone(), Style::default().fg(Color::Green)))
                        } else if l.starts_with("- [ ]") {
                            Line::from(Span::styled(
                                l.clone(),
                                Style::default().fg(Color::DarkGray),
                            ))
                        } else {
                            Line::from(l.as_str())
                        }
                    })
                    .collect(),
                RightPane::Tools => shell
                    .tool_lines
                    .iter()
                    .map(|l| {
                        Line::from(Span::styled(l.clone(), Style::default().fg(Color::Yellow)))
                    })
                    .collect(),
                RightPane::MissionControl => shell
                    .mission_control_lines
                    .iter()
                    .map(|l| Line::from(l.as_str()))
                    .collect(),
                RightPane::Artifacts => shell
                    .artifact_lines
                    .iter()
                    .map(|l| Line::from(l.as_str()))
                    .collect(),
            };

            // Pane tabs
            let pane_tabs = Tabs::new(vec![
                Span::raw("Plan"),
                Span::raw("Tools"),
                Span::raw("Mission"),
                Span::raw("Artifacts"),
            ])
            .select(match right_pane {
                RightPane::Plan => 0,
                RightPane::Tools => 1,
                RightPane::MissionControl => 2,
                RightPane::Artifacts => 3,
            })
            .style(Style::default().fg(Color::DarkGray))
            .highlight_style(
                Style::default()
                    .fg(theme.secondary)
                    .add_modifier(Modifier::BOLD),
            );

            let right_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(1), Constraint::Min(1)])
                .split(body[1]);

            frame.render_widget(pane_tabs, right_layout[0]);
            frame.render_widget(
                Paragraph::new(right_lines)
                    .block(
                        Block::default()
                            .title(right_pane.title())
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(Color::DarkGray)),
                    )
                    .wrap(Wrap { trim: false }),
                right_layout[1],
            );

            // Tool output area
            let tool_output_lines: Vec<Line<'_>> = shell
                .tool_lines
                .iter()
                .rev()
                .take(50)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .map(|l| {
                    if l.starts_with("error") || l.starts_with("Error") {
                        Line::from(Span::styled(l.clone(), Style::default().fg(Color::Red)))
                    } else if l.starts_with("ok") || l.starts_with("success") {
                        Line::from(Span::styled(l.clone(), Style::default().fg(Color::Green)))
                    } else {
                        Line::from(Span::styled(
                            l.clone(),
                            Style::default().fg(Color::DarkGray),
                        ))
                    }
                })
                .collect();

            frame.render_widget(
                Paragraph::new(tool_output_lines)
                    .block(
                        Block::default()
                            .title(Span::styled(
                                " Tool Output ",
                                Style::default().fg(Color::Yellow),
                            ))
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(Color::DarkGray)),
                    )
                    .wrap(Wrap { trim: false }),
                vertical[2],
            );

            // Input with blinking cursor
            let input_display = if cursor_visible {
                format!("{}█", input)
            } else {
                format!("{} ", input)
            };
            frame.render_widget(
                Paragraph::new(input_display).block(
                    Block::default()
                        .title(Span::styled(
                            " deepseek ",
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ))
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(theme.primary)),
                ),
                vertical[3],
            );

            // Status bar with styled spans
            frame.render_widget(
                Paragraph::new(Line::from(render_statusline_spans(&status))),
                vertical[4],
            );

            // Info/help line
            frame.render_widget(
                Paragraph::new(Span::styled(
                    info_line.clone(),
                    Style::default().fg(Color::DarkGray),
                )),
                vertical[5],
            );
        })?;

        if !event::poll(Duration::from_millis(120))? {
            continue;
        }
        let input_event = event::read()?;
        if let Event::Paste(pasted) = &input_event {
            input.push_str(pasted);
            info_line = "pasted input".to_string();
            continue;
        }
        let Event::Key(key) = input_event else {
            continue;
        };

        if key == bindings.exit {
            break;
        }
        if key == bindings.toggle_raw {
            right_pane = right_pane.cycle();
            right_pane_collapsed = false;
            // Load artifacts on-demand when switching to Artifacts pane
            if right_pane == RightPane::Artifacts && shell.artifact_lines.is_empty() {
                let cwd = std::env::current_dir().unwrap_or_default();
                shell.artifact_lines = load_artifact_lines(&cwd);
            }
            info_line = format!("pane: {}", right_pane.title());
            continue;
        }
        if key == bindings.toggle_mission_control {
            if right_pane == RightPane::MissionControl && !right_pane_collapsed {
                right_pane_collapsed = true;
            } else {
                right_pane = RightPane::MissionControl;
                right_pane_collapsed = false;
            }
            info_line = if right_pane_collapsed {
                "Mission Control: collapsed".to_string()
            } else {
                "Mission Control".to_string()
            };
            continue;
        }
        if key == bindings.toggle_artifacts {
            if right_pane == RightPane::Artifacts && !right_pane_collapsed {
                right_pane_collapsed = true;
            } else {
                right_pane = RightPane::Artifacts;
                right_pane_collapsed = false;
                if shell.artifact_lines.is_empty() {
                    let cwd = std::env::current_dir().unwrap_or_default();
                    shell.artifact_lines = load_artifact_lines(&cwd);
                }
            }
            info_line = if right_pane_collapsed {
                "Artifacts: collapsed".to_string()
            } else {
                "Artifacts".to_string()
            };
            continue;
        }
        if key == bindings.toggle_plan_collapse {
            if right_pane == RightPane::Plan {
                right_pane_collapsed = !right_pane_collapsed;
            } else {
                right_pane = RightPane::Plan;
                right_pane_collapsed = false;
            }
            info_line = if right_pane_collapsed {
                "Plan: collapsed".to_string()
            } else {
                "Plan".to_string()
            };
            continue;
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
            match on_submit(&background_cmd) {
                Ok(output) => {
                    shell.push_transcript(output);
                    info_line = "background job started".to_string();
                }
                Err(err) => {
                    let text = format!("error: {err}");
                    shell.push_tool(text.clone());
                    info_line = text;
                }
            }
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
                    "terminal-setup",
                    "keybindings",
                    "doctor",
                ];
                if let Some(next) = commands.iter().find(|cmd| cmd.starts_with(&prefix)) {
                    input = format!("/{next}");
                }
            } else if let Some(completed) = autocomplete_path_input(&input) {
                input = completed;
            }
            continue;
        }
        if key == bindings.history_prev {
            if let Some(last) = history.back() {
                input = last.clone();
            }
            continue;
        }
        if key == bindings.newline {
            input.push('\n');
            continue;
        }
        if key == bindings.submit {
            let prompt = input.trim().to_string();
            if prompt.is_empty() {
                continue;
            }
            history.push_back(prompt.clone());
            if history.len() > 100 {
                let _ = history.pop_front();
            }
            shell.push_user(&prompt);
            input.clear();
            match on_submit(&prompt) {
                Ok(output) => {
                    shell.push_transcript(output);
                    info_line = "ok".to_string();
                }
                Err(err) => {
                    let text = format!("error: {err}");
                    shell.push_tool(text.clone());
                    info_line = text;
                }
            }
            continue;
        }
        if let KeyCode::Backspace = key.code {
            let _ = input.pop();
            continue;
        }
        if let KeyCode::Char(ch) = key.code {
            input.push(ch);
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
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
        let spans = render_statusline_spans(&status);
        let text: String = spans.iter().map(|s| s.content.to_string()).collect();
        assert!(text.contains("deepseek-chat"));
        assert!(text.contains("LOCKED"));
        assert!(text.contains("pending"));
        assert!(text.contains("tasks"));
        assert!(text.contains("AUTOPILOT"));
        assert!(text.contains("100K/128K"));
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
}
