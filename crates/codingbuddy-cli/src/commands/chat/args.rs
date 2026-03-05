use anyhow::{Result, anyhow};
use codingbuddy_agent::ChatMode;
use codingbuddy_core::AppConfig;

pub(crate) fn is_max_think_selection(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    lower.contains("reasoner") || lower.contains("max") || lower.contains("high")
}

/// Format `/provider` info as a plain-text string. Used by both TUI and non-TUI paths.
pub(crate) fn format_provider_info(cfg: &AppConfig, provider: Option<String>) -> String {
    let active = cfg.llm.active_provider();
    if let Some(name) = provider {
        if let Some(p) = cfg.llm.providers.get(&name) {
            format!(
                "Provider: {} ({}, model: {})\nTo switch permanently, run: codingbuddy setup",
                name, p.base_url, p.models.chat
            )
        } else {
            format!(
                "Unknown provider: {}. Available: {}",
                name,
                cfg.llm
                    .providers
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    } else {
        let mut lines = vec![format!(
            "Current provider: {} ({})",
            cfg.llm.provider, active.base_url
        )];
        for (name, p) in &cfg.llm.providers {
            let marker = if *name == cfg.llm.provider {
                " (active)"
            } else {
                ""
            };
            lines.push(format!(
                "  {} — {} (model: {}){}",
                name, p.base_url, p.models.chat, marker
            ));
        }
        lines.push("Switch permanently with: codingbuddy setup".to_string());
        lines.join("\n")
    }
}

pub(crate) fn truncate_inline(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        text.to_string()
    } else {
        format!("{}...", &text[..text.floor_char_boundary(max_chars)])
    }
}

pub(crate) fn parse_commit_message(args: &[String]) -> Result<Option<String>> {
    if args.is_empty() {
        return Ok(None);
    }
    if matches!(
        args.first().map(String::as_str),
        Some("-m") | Some("--message")
    ) {
        let message = args[1..].join(" ").trim().to_string();
        if message.is_empty() {
            return Err(anyhow!("commit message cannot be empty"));
        }
        return Ok(Some(message));
    }
    Ok(Some(args.join(" ")))
}

pub(crate) fn parse_stage_args(args: &[String]) -> (bool, Vec<String>) {
    let mut all = false;
    let mut files = Vec::new();
    for arg in args {
        if matches!(arg.as_str(), "--all" | "-A") {
            all = true;
        } else {
            files.push(arg.clone());
        }
    }
    (all, files)
}

pub(crate) fn parse_diff_args(args: &[String]) -> (bool, bool) {
    let staged = args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--staged" | "--cached" | "-s"));
    let stat = args.iter().any(|arg| arg == "--stat");
    (staged, stat)
}

pub(crate) fn parse_chat_mode_name(raw: &str) -> Option<ChatMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "ask" => Some(ChatMode::Ask),
        "code" | "plan" => Some(ChatMode::Code),
        "context" => Some(ChatMode::Context),
        _ => None,
    }
}

pub(crate) fn chat_mode_name(mode: ChatMode) -> &'static str {
    match mode {
        ChatMode::Ask => "ask",
        ChatMode::Code => "code",
        ChatMode::Context => "context",
    }
}
