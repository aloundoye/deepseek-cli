use anyhow::{Result, anyhow};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

pub(crate) fn copy_to_clipboard(text: &str) {
    #[cfg(target_os = "macos")]
    {
        let mut child = Command::new("pbcopy").stdin(Stdio::piped()).spawn().ok();
        if let Some(ref mut c) = child {
            if let Some(ref mut stdin) = c.stdin {
                let _ = stdin.write_all(text.as_bytes());
            }
            let _ = c.wait();
        }
    }
    #[cfg(target_os = "linux")]
    {
        let mut child = Command::new("xclip")
            .args(["-selection", "clipboard"])
            .stdin(Stdio::piped())
            .spawn()
            .ok();
        if let Some(ref mut c) = child {
            if let Some(ref mut stdin) = c.stdin {
                let _ = stdin.write_all(text.as_bytes());
            }
            let _ = c.wait();
        }
    }
    #[cfg(target_os = "windows")]
    {
        let mut child = Command::new("clip").stdin(Stdio::piped()).spawn().ok();
        if let Some(ref mut c) = child {
            if let Some(ref mut stdin) = c.stdin {
                let _ = stdin.write_all(text.as_bytes());
            }
            let _ = c.wait();
        }
    }
}

pub(crate) fn read_from_clipboard() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        return run_capture("pbpaste", &[]);
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(value) = run_capture("xclip", &["-selection", "clipboard", "-o"]) {
            return Some(value);
        }
        return run_capture("wl-paste", &[]);
    }
    #[cfg(target_os = "windows")]
    {
        let output = Command::new("powershell")
            .args(["-NoProfile", "-Command", "Get-Clipboard -Raw | Out-String"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        return Some(String::from_utf8_lossy(&output.stdout).trim().to_string());
    }
    #[allow(unreachable_code)]
    None
}

pub(crate) fn run_process(cwd: &Path, program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).current_dir(cwd).args(args).output()?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        return Err(anyhow!(
            "{} {:?} failed with status {}: {}{}",
            program,
            args,
            output.status,
            stdout,
            stderr
        ));
    }
    Ok(format!("{stdout}{stderr}").trim().to_string())
}

pub(crate) fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = std::env::var("HOME")
            .ok()
            .or_else(|| std::env::var("USERPROFILE").ok())
    {
        return Path::new(&home).join(rest).to_string_lossy().to_string();
    }
    path.to_string()
}

pub(crate) fn default_editor() -> &'static str {
    if cfg!(target_os = "windows") {
        "notepad"
    } else {
        "nano"
    }
}

pub(crate) fn estimate_tokens(text: &str) -> u64 {
    (text.chars().count() as u64).div_ceil(4)
}

pub(crate) fn estimate_rate_limit_events(cwd: &Path) -> u64 {
    let path = deepseek_core::runtime_dir(cwd).join("observe.log");
    let Ok(raw) = std::fs::read_to_string(path) else {
        return 0;
    };
    raw.lines()
        .filter(|line| {
            let lower = line.to_ascii_lowercase();
            lower.contains("429") || lower.contains("rate limit")
        })
        .count() as u64
}

pub(crate) fn run_capture(cmd: &str, args: &[&str]) -> Option<String> {
    Command::new(cmd).args(args).output().ok().and_then(|out| {
        if out.status.success() {
            Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
        } else {
            None
        }
    })
}

pub(crate) fn command_exists(name: &str) -> bool {
    if name.trim().is_empty() {
        return false;
    }
    let checker = if cfg!(target_os = "windows") {
        ("where", vec![name])
    } else {
        ("which", vec![name])
    };
    Command::new(checker.0)
        .args(checker.1)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[allow(dead_code)]
pub(crate) fn validate_json_schema(
    value: &serde_json::Value,
    expected_keys: &[&str],
    required_keys: &[&str],
) -> Result<()> {
    let obj = value
        .as_object()
        .ok_or_else(|| anyhow!("expected JSON object"))?;
    for key in required_keys {
        if !obj.contains_key(*key) {
            return Err(anyhow!("missing required key '{}'", key));
        }
    }
    let allowed: std::collections::HashSet<&str> = expected_keys.iter().copied().collect();
    for key in obj.keys() {
        if !allowed.contains(key.as_str()) {
            return Err(anyhow!("unexpected key '{}'", key));
        }
    }
    Ok(())
}
