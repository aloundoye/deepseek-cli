use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io;
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
    Teleport,
    RemoteEnv,
    Status,
    Effort(Option<String>),
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
            "teleport" => Self::Teleport,
            "remote-env" => Self::RemoteEnv,
            "status" => Self::Status,
            "effort" => Self::Effort(args.first().cloned()),
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
}

pub fn render_statusline(status: &UiStatus) -> String {
    format!(
        "model={} approvals={} jobs={} autopilot={} cost=${:.6}",
        status.model,
        status.pending_approvals,
        status.background_jobs,
        if status.autopilot_running {
            "running"
        } else {
            "idle"
        },
        status.estimated_cost_usd,
    )
}

#[derive(Debug, Clone, Default)]
pub struct ChatShell {
    pub transcript: Vec<String>,
    pub plan_lines: Vec<String>,
    pub tool_lines: Vec<String>,
}

impl ChatShell {
    pub fn push_transcript(&mut self, line: impl Into<String>) {
        self.transcript.push(line.into());
    }

    pub fn push_plan(&mut self, line: impl Into<String>) {
        self.plan_lines.push(line.into());
    }

    pub fn push_tool(&mut self, line: impl Into<String>) {
        self.tool_lines.push(line.into());
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
        }
    }
}

pub fn run_tui_shell<F>(status: UiStatus, mut on_submit: F) -> Result<()>
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
    let mut info_line =
        String::from("Ctrl+C exit | Ctrl+O toggle raw | Ctrl+B background hint | Ctrl+V paste");
    let mut show_raw = false;
    let bindings = KeyBindings::default();
    let mut history: VecDeque<String> = VecDeque::new();
    let mut last_escape_at: Option<Instant> = None;

    loop {
        let status_line = render_statusline(&status);
        terminal.draw(|frame| {
            let area = frame.area();
            let vertical = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Percentage(65),
                    Constraint::Percentage(25),
                    Constraint::Length(3),
                    Constraint::Length(1),
                    Constraint::Length(1),
                ])
                .split(area);

            let body = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
                .split(vertical[0]);

            let transcript_lines = shell
                .transcript
                .iter()
                .rev()
                .take(200)
                .cloned()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .map(Line::from)
                .collect::<Vec<_>>();
            frame.render_widget(
                Paragraph::new(transcript_lines)
                    .block(Block::default().title("Transcript").borders(Borders::ALL))
                    .wrap(Wrap { trim: false }),
                body[0],
            );

            let right_lines = if show_raw {
                shell.tool_lines.clone()
            } else {
                shell.plan_lines.clone()
            };
            frame.render_widget(
                Paragraph::new(
                    right_lines
                        .into_iter()
                        .map(Line::from)
                        .collect::<Vec<Line>>(),
                )
                .block(
                    Block::default()
                        .title(if show_raw { "Tools" } else { "Plan" })
                        .borders(Borders::ALL),
                )
                .wrap(Wrap { trim: false }),
                body[1],
            );

            frame.render_widget(
                Paragraph::new(shell.tool_lines.join("\n"))
                    .block(Block::default().title("Tool Output").borders(Borders::ALL))
                    .wrap(Wrap { trim: false }),
                vertical[1],
            );

            frame.render_widget(
                Paragraph::new(input.clone()).block(
                    Block::default()
                        .title("Input")
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::Cyan)),
                ),
                vertical[2],
            );
            frame.render_widget(Paragraph::new(status_line), vertical[3]);
            frame.render_widget(Paragraph::new(info_line.clone()), vertical[4]);
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
            show_raw = !show_raw;
            info_line = format!("raw_mode={show_raw}");
            continue;
        }
        if key == bindings.background {
            info_line = "background hint: use /status then deepseek background list".to_string();
            shell.push_tool("background hotkey pressed");
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
                ];
                if let Some(next) = commands.iter().find(|cmd| cmd.starts_with(&prefix)) {
                    input = format!("/{next}");
                }
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
            shell.push_transcript(format!("> {prompt}"));
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
    }

    #[test]
    fn renders_statusline() {
        let line = render_statusline(&UiStatus {
            model: "deepseek-chat".to_string(),
            pending_approvals: 2,
            estimated_cost_usd: 0.001,
            background_jobs: 1,
            autopilot_running: true,
        });
        assert!(line.contains("model=deepseek-chat"));
        assert!(line.contains("autopilot=running"));
    }
}
