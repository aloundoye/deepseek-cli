//! Context compaction: pruning old tool outputs and building compaction summaries.

use anyhow::{Result, anyhow};
use codingbuddy_core::{ChatMessage, ChatRequest, ToolChoice};
use codingbuddy_llm::LlmClient;

/// Age threshold for pruning old tool outputs (turn groups older than this
/// are eligible for truncation during the prune phase).
pub(crate) const PRUNE_AGE_TURNS: usize = 3;

/// Target token usage after compaction — we try to fit the compacted
/// conversation into this fraction of the context window.
pub const COMPACTION_TARGET_PCT: f64 = 0.80;

/// Template for LLM-based compaction. The LLM fills this in based on the conversation.
pub(crate) const COMPACTION_TEMPLATE: &str = "Summarize this conversation into the following sections. \
Be precise and factual — include file paths, function names, and error messages. \
Keep each section to 2-5 bullet points. Output ONLY the filled template:\n\n\
## Goal\n(What the user asked for)\n\n\
## Completed\n(What was done successfully — include file paths)\n\n\
## In Progress\n(What's partially done or pending)\n\n\
## Key Facts Established\n\
(Important facts the user stated or that were discovered during the conversation. \
Include: file paths discussed, decisions made, user preferences stated, corrections given. \
These facts must be preserved across compaction so the model does not lose context.)\n\n\
## Key Findings\n(Important discoveries, errors hit, architectural decisions)\n\n\
## Modified Files\n(List of files created, edited, or deleted)";

/// Extract a file path from a tool result (JSON or plain text).
pub(crate) fn extract_tool_path(content: &str) -> Option<String> {
    // Try parsing as JSON object with "path" or "file_path" key
    if content.starts_with('{')
        && let Ok(obj) = serde_json::from_str::<serde_json::Value>(content)
        && let Some(map) = obj.as_object()
    {
        if let Some(path) = map.get("path").and_then(|v| v.as_str()) {
            return Some(path.to_string());
        }
        if let Some(path) = map.get("file_path").and_then(|v| v.as_str()) {
            return Some(path.to_string());
        }
    }
    // Try as string containing a path-like pattern at the start
    if (content.starts_with('/') || content.starts_with("./"))
        && let Some(line) = content.lines().next()
    {
        let trimmed = line.trim();
        if trimmed.len() < 256 && !trimmed.contains(' ') {
            return Some(trimmed.to_string());
        }
    }
    None
}

/// Build a structured summary from messages being compacted.
///
/// Extracts key facts: files modified/read, errors encountered, key decisions,
/// and tool usage counts. This preserves important context that would otherwise
/// be lost during compaction.
pub(crate) fn build_compaction_summary(messages: &[ChatMessage]) -> String {
    let mut files_read: Vec<String> = Vec::new();
    let mut files_modified: Vec<String> = Vec::new();
    let mut tools_used: Vec<String> = Vec::new();
    let mut errors_hit: Vec<String> = Vec::new();
    let mut key_decisions: Vec<String> = Vec::new();

    for msg in messages {
        match msg {
            ChatMessage::Assistant {
                tool_calls,
                content,
                ..
            } => {
                for tc in tool_calls {
                    tools_used.push(tc.name.clone());
                    if let Ok(args) = serde_json::from_str::<serde_json::Value>(&tc.arguments)
                        && let Some(path) = args
                            .get("file_path")
                            .or(args.get("path"))
                            .and_then(|v| v.as_str())
                    {
                        if tc.name.contains("read")
                            || tc.name.contains("glob")
                            || tc.name.contains("grep")
                        {
                            files_read.push(path.to_string());
                        } else {
                            files_modified.push(path.to_string());
                        }
                    }
                }
                if let Some(text) = content
                    && text.len() > 50
                    && text.len() < 500
                {
                    key_decisions.push(truncate_line(text, 150));
                }
            }
            ChatMessage::Tool { content, .. } => {
                let lower = content.to_ascii_lowercase();
                if lower.contains("error")
                    || lower.contains("failed")
                    || lower.contains("not found")
                {
                    errors_hit.push(truncate_line(content, 100));
                }
            }
            _ => {}
        }
    }

    files_read.sort();
    files_read.dedup();
    files_modified.sort();
    files_modified.dedup();

    let mut summary = String::new();
    if !files_modified.is_empty() {
        summary.push_str(&format!("Files modified: {}\n", files_modified.join(", ")));
    }
    if !files_read.is_empty() {
        summary.push_str(&format!("Files read: {}\n", files_read.join(", ")));
    }
    if !errors_hit.is_empty() {
        summary.push_str(&format!("Errors encountered: {}\n", errors_hit.join("; ")));
    }
    if !key_decisions.is_empty() {
        summary.push_str("Key decisions:\n");
        for d in key_decisions.iter().take(5) {
            summary.push_str(&format!("- {d}\n"));
        }
    }
    let tool_counts = count_tool_usage(&tools_used);
    summary.push_str(&format!("Tools used: {tool_counts}\n"));
    summary
}

/// Build a structured compaction summary using an LLM call.
///
/// Sends the conversation to `deepseek-chat` with a structured template.
/// Falls back to code-based extraction on any error.
pub(crate) fn build_compaction_summary_with_llm(
    llm: &(dyn LlmClient + Send + Sync),
    messages: &[ChatMessage],
) -> Result<String> {
    // Build a condensed representation of the conversation for the LLM
    let mut conversation_text = String::new();
    for msg in messages {
        match msg {
            ChatMessage::User { content } => {
                conversation_text.push_str(&format!("USER: {}\n", truncate_line(content, 500)));
            }
            ChatMessage::Assistant {
                content,
                tool_calls,
                ..
            } => {
                if let Some(text) = content {
                    conversation_text
                        .push_str(&format!("ASSISTANT: {}\n", truncate_line(text, 500)));
                }
                for tc in tool_calls {
                    conversation_text.push_str(&format!(
                        "TOOL_CALL: {}({})\n",
                        tc.name,
                        truncate_line(&tc.arguments, 200)
                    ));
                }
            }
            ChatMessage::Tool {
                tool_call_id,
                content,
            } => {
                conversation_text.push_str(&format!(
                    "TOOL_RESULT[{tool_call_id}]: {}\n",
                    truncate_line(content, 300)
                ));
            }
            ChatMessage::System { content } => {
                conversation_text.push_str(&format!("SYSTEM: {}\n", truncate_line(content, 200)));
            }
        }
    }

    // Skip if conversation is too short to be worth an LLM call
    if conversation_text.len() < 200 {
        return Err(anyhow!("conversation too short for LLM compaction"));
    }

    let request = ChatRequest {
        model: codingbuddy_core::CODINGBUDDY_V32_CHAT_MODEL.to_string(),
        messages: vec![
            ChatMessage::System {
                content: COMPACTION_TEMPLATE.to_string(),
            },
            ChatMessage::User {
                content: conversation_text,
            },
        ],
        max_tokens: 2048,
        temperature: Some(0.0),
        top_p: None,
        presence_penalty: None,
        frequency_penalty: None,
        logprobs: None,
        top_logprobs: None,
        tools: vec![],
        tool_choice: ToolChoice::none(),
        thinking: None,
        images: vec![],
        response_format: None,
    };

    let response = llm.complete_chat(&request)?;
    let text = response.text.trim().to_string();

    // Validate the response has at least some structure
    if text.contains("## Goal") || text.contains("## Completed") || text.contains("## Modified") {
        Ok(text)
    } else if text.len() > 50 {
        // Acceptable even without perfect headers
        Ok(text)
    } else {
        Err(anyhow!("LLM compaction produced empty or invalid response"))
    }
}

/// Truncate a line to max_len chars, appending "..." if truncated.
/// Uses `floor_char_boundary` to avoid panicking on multi-byte UTF-8 characters.
pub(crate) fn truncate_line(text: &str, max_len: usize) -> String {
    let first_line = text.lines().next().unwrap_or(text);
    if first_line.len() <= max_len {
        first_line.to_string()
    } else {
        let safe_end = first_line.floor_char_boundary(max_len);
        format!("{}...", &first_line[..safe_end])
    }
}

/// Count tool usage into a human-readable summary (e.g. "fs_read×5, fs_edit×2").
pub(crate) fn count_tool_usage(tools: &[String]) -> String {
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for name in tools {
        *counts.entry(name.as_str()).or_default() += 1;
    }
    if counts.is_empty() {
        return "none".to_string();
    }
    counts
        .iter()
        .map(|(name, count)| format!("{name}×{count}"))
        .collect::<Vec<_>>()
        .join(", ")
}
