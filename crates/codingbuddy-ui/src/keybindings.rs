use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use serde::Deserialize;
use std::fs;
use std::path::Path;

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
    pub history_search: KeyEvent,
    pub paste_hint: KeyEvent,
    pub toggle_mission_control: KeyEvent,
    pub toggle_artifacts: KeyEvent,
    pub toggle_plan_collapse: KeyEvent,
    pub cycle_permission_mode: KeyEvent,
    pub exit_session: KeyEvent,
    pub clear_screen: KeyEvent,
    pub newline_alt: KeyEvent,
    pub switch_model: KeyEvent,
    pub toggle_thinking: KeyEvent,
    pub kill_background: KeyEvent,
    pub open_editor: KeyEvent,
    pub approve_plan: KeyEvent,
    pub reject_plan: KeyEvent,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub(crate) struct KeyBindingsFile {
    pub(crate) exit: Option<String>,
    pub(crate) submit: Option<String>,
    pub(crate) newline: Option<String>,
    pub(crate) stop: Option<String>,
    pub(crate) rewind_menu: Option<String>,
    pub(crate) autocomplete: Option<String>,
    pub(crate) background: Option<String>,
    pub(crate) toggle_raw: Option<String>,
    pub(crate) history_prev: Option<String>,
    pub(crate) history_search: Option<String>,
    pub(crate) paste_hint: Option<String>,
    pub(crate) toggle_mission_control: Option<String>,
    pub(crate) toggle_artifacts: Option<String>,
    pub(crate) toggle_plan_collapse: Option<String>,
    pub(crate) cycle_permission_mode: Option<String>,
    pub(crate) exit_session: Option<String>,
    pub(crate) clear_screen: Option<String>,
    pub(crate) newline_alt: Option<String>,
    pub(crate) switch_model: Option<String>,
    pub(crate) toggle_thinking: Option<String>,
    pub(crate) kill_background: Option<String>,
    pub(crate) open_editor: Option<String>,
    pub(crate) approve_plan: Option<String>,
    pub(crate) reject_plan: Option<String>,
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
            history_search: KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL),
            paste_hint: KeyEvent::new(KeyCode::Char('v'), KeyModifiers::CONTROL),
            toggle_mission_control: KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL),
            toggle_artifacts: KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL),
            toggle_plan_collapse: KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL),
            cycle_permission_mode: KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT),
            exit_session: KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL),
            clear_screen: KeyEvent::new(KeyCode::Char('l'), KeyModifiers::CONTROL),
            newline_alt: KeyEvent::new(KeyCode::Char('j'), KeyModifiers::CONTROL),
            switch_model: KeyEvent::new(KeyCode::Char('p'), KeyModifiers::ALT),
            toggle_thinking: KeyEvent::new(KeyCode::Char('t'), KeyModifiers::ALT),
            kill_background: KeyEvent::new(KeyCode::Char('f'), KeyModifiers::CONTROL),
            open_editor: KeyEvent::new(KeyCode::Char('g'), KeyModifiers::CONTROL),
            approve_plan: KeyEvent::new(KeyCode::Char('y'), KeyModifiers::CONTROL),
            reject_plan: KeyEvent::new(KeyCode::Char('y'), KeyModifiers::ALT),
        }
    }
}

impl KeyBindings {
    pub(crate) fn apply_overrides(mut self, raw: KeyBindingsFile) -> Result<Self> {
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
        if let Some(value) = raw.history_search {
            self.history_search = parse_key_event(&value)?;
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
        if let Some(value) = raw.exit_session {
            self.exit_session = parse_key_event(&value)?;
        }
        if let Some(value) = raw.clear_screen {
            self.clear_screen = parse_key_event(&value)?;
        }
        if let Some(value) = raw.newline_alt {
            self.newline_alt = parse_key_event(&value)?;
        }
        if let Some(value) = raw.switch_model {
            self.switch_model = parse_key_event(&value)?;
        }
        if let Some(value) = raw.toggle_thinking {
            self.toggle_thinking = parse_key_event(&value)?;
        }
        if let Some(value) = raw.kill_background {
            self.kill_background = parse_key_event(&value)?;
        }
        if let Some(value) = raw.open_editor {
            self.open_editor = parse_key_event(&value)?;
        }
        if let Some(value) = raw.approve_plan {
            self.approve_plan = parse_key_event(&value)?;
        }
        if let Some(value) = raw.reject_plan {
            self.reject_plan = parse_key_event(&value)?;
        }
        Ok(self)
    }
}

pub fn load_keybindings(path: &Path) -> Result<KeyBindings> {
    let raw = fs::read_to_string(path)?;
    let parsed: KeyBindingsFile = serde_json::from_str(&raw)?;
    KeyBindings::default().apply_overrides(parsed)
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
