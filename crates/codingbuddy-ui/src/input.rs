use super::state::VimMode;
use super::*;

pub(crate) fn autocomplete_at_suggestions(prefix: &str, workspace: &Path) -> Vec<String> {
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

/// Execute a `!` prefixed shell command directly, bypassing the LLM.
/// Returns the combined stdout/stderr output as a string.
pub(crate) fn execute_bang_command(cmd: &str) -> String {
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
pub(crate) fn try_clipboard_image() -> Option<Vec<u8>> {
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
pub(crate) fn autocomplete_at_mention(input: &str) -> Option<String> {
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

pub(crate) fn autocomplete_path_input(input: &str) -> Option<String> {
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

pub(crate) fn autocomplete_path_token(token: &str) -> Option<String> {
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

pub(crate) fn history_ghost_suffix(history: &VecDeque<String>, input: &str) -> Option<String> {
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

pub(crate) fn history_reverse_search_index(
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

pub(crate) fn apply_reverse_search_result(
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
pub(crate) enum VimSlashCommand {
    Toggle,
    On,
    Off,
    SetMode(VimMode),
}

pub(crate) fn parse_vim_slash_command(
    prompt: &str,
) -> Option<Result<VimSlashCommand, &'static str>> {
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
pub(crate) fn apply_vim_command(
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

pub(crate) fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

pub(crate) fn move_to_next_word_start(input: &str, cursor_pos: usize) -> usize {
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

pub(crate) fn move_to_prev_word_start(input: &str, cursor_pos: usize) -> usize {
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

pub(crate) fn move_to_word_end(input: &str, cursor_pos: usize) -> usize {
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

pub(crate) fn display_key_code(code: KeyCode) -> String {
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

pub(crate) fn apply_vim_operator_range(
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

pub(crate) fn word_text_object_bounds(
    input: &str,
    cursor_pos: usize,
    around: bool,
) -> Option<(usize, usize)> {
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

pub(crate) fn delimiter_text_object_bounds(
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

pub(crate) fn resolve_vim_text_object_bounds(
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

pub(crate) fn visual_bounds(len: usize, anchor: usize, cursor_pos: usize) -> (usize, usize) {
    let start = anchor.min(cursor_pos).min(len);
    let mut end = anchor.max(cursor_pos).min(len);
    if end < len {
        end += 1;
    }
    (start, end)
}

pub(crate) fn extract_visual_selection(input: &str, anchor: usize, cursor_pos: usize) -> String {
    let (start, end) = visual_bounds(input.len(), anchor, cursor_pos);
    if start >= end {
        return String::new();
    }
    input[start..end].to_string()
}
