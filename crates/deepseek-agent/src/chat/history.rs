use deepseek_core::ChatMessage;
use deepseek_tools::map_tool_name;
use serde_json::json;
use std::collections::HashSet;

/// Estimate BPE token count using a character-class-aware heuristic.
///
/// BPE tokenizers (including DeepSeek's) split text at whitespace and
/// punctuation boundaries, then merge frequent byte-pairs into single tokens.
/// This heuristic models that behavior by:
///
/// 1. Splitting on whitespace (each whitespace run ≈ 1 token)
/// 2. Further splitting words at punctuation/case boundaries
/// 3. Estimating tokens per fragment based on character class:
///    - Common English words (all-alpha, ≤6 chars): 1 token
///    - Longer words / identifiers: ~1 token per 4 chars (BPE average)
///    - Digits/hex: ~1 token per 3 chars
///    - Punctuation/operators: 1-2 tokens per symbol cluster
///    - Non-ASCII / Unicode: ~1 token per 2-4 bytes
pub(crate) fn estimate_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    let mut tokens: u64 = 0;
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        let b = bytes[i];

        if b.is_ascii_whitespace() {
            // Whitespace runs count as ~1 token
            while i < len && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            tokens += 1;
            continue;
        }

        // Scan a "word" of contiguous non-whitespace
        let word_start = i;
        while i < len && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let word = &text[word_start..i];
        let word_len = word.len();

        // Classify the word
        let all_alpha = word.bytes().all(|b| b.is_ascii_alphabetic());
        let all_digit = word.bytes().all(|b| b.is_ascii_digit());
        let all_hex = word.bytes().all(|b| b.is_ascii_hexdigit());
        let all_ascii = word.is_ascii();
        let has_punctuation = word.bytes().any(|b| b.is_ascii_punctuation());

        if all_alpha {
            // Pure alphabetic words: common English words ≤6 chars ≈ 1 token
            tokens += match word_len {
                1..=6 => 1,
                7..=12 => 2,
                _ => (word_len as u64).div_ceil(4),
            };
        } else if all_digit || all_hex {
            // Numbers / hex: ~3 chars per token
            tokens += (word_len as u64).div_ceil(3).max(1);
        } else if all_ascii && has_punctuation {
            // Mixed ASCII with punctuation (paths, URLs, operators)
            // Count punctuation chars as ~1 token each, alpha runs as words
            let punct_count = word.bytes().filter(|b| b.is_ascii_punctuation()).count();
            let alpha_len = word_len - punct_count;
            tokens += punct_count as u64
                + (alpha_len as u64)
                    .div_ceil(4)
                    .max(if alpha_len > 0 { 1 } else { 0 });
        } else if !all_ascii {
            // Non-ASCII (Unicode, CJK, etc.): ~2-4 bytes per token
            // CJK characters are typically 1 token each, others vary
            let char_count = word.chars().count();
            tokens += ((char_count as u64) * 3).div_ceil(4).max(1);
        } else {
            // Fallback: ~4 chars per token
            tokens += (word_len as u64).div_ceil(4).max(1);
        }
    }

    tokens.max(1)
}

pub(crate) fn estimate_messages_tokens(messages: &[ChatMessage]) -> u64 {
    // Each message has ~4 tokens of framing overhead (role, delimiters).
    const MSG_OVERHEAD: u64 = 4;
    messages
        .iter()
        .map(|m| {
            MSG_OVERHEAD
                + match m {
                    ChatMessage::System { content } | ChatMessage::User { content } => {
                        estimate_tokens(content)
                    }
                    ChatMessage::Assistant {
                        content,
                        tool_calls,
                        ..
                    } => {
                        let c = content.as_deref().map(estimate_tokens).unwrap_or(0);
                        let t: u64 = tool_calls
                            .iter()
                            .map(|tc| estimate_tokens(&tc.name) + estimate_tokens(&tc.arguments))
                            .sum();
                        c + t
                    }
                    ChatMessage::Tool { content, .. } => estimate_tokens(content),
                }
        })
        .sum()
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ChatHistoryRepairStats {
    pub(crate) dropped_tool_messages: usize,
    pub(crate) stripped_tool_calls: usize,
}

impl ChatHistoryRepairStats {
    pub(crate) fn changed(self) -> bool {
        self.dropped_tool_messages > 0 || self.stripped_tool_calls > 0
    }
}

pub(crate) fn strip_unresolved_tool_calls(
    messages: &mut [ChatMessage],
    assistant_index: Option<usize>,
    unresolved_ids: &HashSet<String>,
) -> usize {
    let Some(idx) = assistant_index else {
        return 0;
    };
    let Some(ChatMessage::Assistant { tool_calls, .. }) = messages.get_mut(idx) else {
        return 0;
    };
    let before = tool_calls.len();
    tool_calls.retain(|tc| !unresolved_ids.contains(&tc.id));
    before.saturating_sub(tool_calls.len())
}

/// Normalize history so each `tool` role message directly corresponds to a
/// preceding assistant message's `tool_calls` entry.
pub(crate) fn sanitize_chat_history_for_tool_calls(
    messages: &mut Vec<ChatMessage>,
) -> ChatHistoryRepairStats {
    let mut stats = ChatHistoryRepairStats::default();
    let mut normalized = Vec::with_capacity(messages.len());
    let mut pending_tool_ids: HashSet<String> = HashSet::new();
    let mut pending_assistant_index: Option<usize> = None;

    for msg in messages.drain(..) {
        match msg {
            ChatMessage::Assistant {
                content,
                reasoning_content,
                tool_calls,
            } => {
                if !pending_tool_ids.is_empty() {
                    stats.stripped_tool_calls += strip_unresolved_tool_calls(
                        &mut normalized,
                        pending_assistant_index,
                        &pending_tool_ids,
                    );
                    pending_tool_ids.clear();
                    pending_assistant_index = None;
                }

                if tool_calls.is_empty() {
                    normalized.push(ChatMessage::Assistant {
                        content,
                        reasoning_content,
                        tool_calls,
                    });
                } else {
                    pending_tool_ids.extend(tool_calls.iter().map(|tc| tc.id.clone()));
                    pending_assistant_index = Some(normalized.len());
                    normalized.push(ChatMessage::Assistant {
                        content,
                        reasoning_content,
                        tool_calls,
                    });
                }
            }
            ChatMessage::Tool {
                tool_call_id,
                content,
            } => {
                if pending_tool_ids.remove(&tool_call_id) {
                    normalized.push(ChatMessage::Tool {
                        tool_call_id,
                        content,
                    });
                    if pending_tool_ids.is_empty() {
                        pending_assistant_index = None;
                    }
                } else {
                    stats.dropped_tool_messages += 1;
                }
            }
            other => {
                if !pending_tool_ids.is_empty() {
                    stats.stripped_tool_calls += strip_unresolved_tool_calls(
                        &mut normalized,
                        pending_assistant_index,
                        &pending_tool_ids,
                    );
                    pending_tool_ids.clear();
                    pending_assistant_index = None;
                }
                normalized.push(other);
            }
        }
    }

    if !pending_tool_ids.is_empty() {
        stats.stripped_tool_calls += strip_unresolved_tool_calls(
            &mut normalized,
            pending_assistant_index,
            &pending_tool_ids,
        );
    }

    *messages = normalized;
    stats
}

/// Compute a compaction tail start index that preserves assistant/tool pairing.
pub(crate) fn compaction_tail_start(messages: &[ChatMessage], desired_tail: usize) -> usize {
    if messages.len() <= 1 {
        return 0;
    }
    let mut start = messages.len().saturating_sub(desired_tail).max(1);
    while start > 1 && matches!(messages.get(start), Some(ChatMessage::Tool { .. })) {
        start -= 1;
    }
    start
}

/// Build a guidance message for the model when a doom-loop is detected.
/// Returns per-tool hints so the model knows how to break out of the loop.
pub(crate) fn doom_loop_guidance(failing_tools: &[String]) -> String {
    let mut parts = Vec::new();
    for tool in failing_tools {
        let hint = match tool.as_str() {
            "bash.run" => {
                "bash.run: Shell metacharacters (&&, ||, ;, backticks, $()) are forbidden. \
                 A single pipeline (|) is allowed only when each segment is allowlisted. \
                 Only allowlisted commands are permitted. \
                 Do NOT retry bash commands for file exploration. Instead use: \
                 fs_glob to find files by pattern, fs_grep to search file contents, \
                 fs_read to read files. These built-in tools have no shell restrictions."
            }
            "fs.write" | "fs.edit" | "multi_edit" => {
                "fs.write/fs.edit: The file write was rejected. Check that the target path is \
                 within the workspace and not in a protected directory. Verify the file exists \
                 before editing. Do NOT retry the same write — fix the path or content first."
            }
            "chrome.navigate" | "chrome.click" | "chrome.screenshot" => {
                "chrome.*: The browser tool failed. The Chrome connection may be unavailable. \
                 Do NOT retry browser tools repeatedly. Use web_fetch for HTTP requests, \
                 or inform the user that browser automation is not available."
            }
            "web_fetch" | "web_search" => {
                "web_fetch/web_search: The web request failed. The URL may be unreachable \
                 or the network may be unavailable. Do NOT retry with the same URL. \
                 Try a different URL or inform the user of the failure."
            }
            _ => {
                "A tool is failing repeatedly. Do NOT retry the same call. \
                 Try an alternative approach or inform the user of the limitation."
            }
        };
        parts.push(format!("IMPORTANT: {hint}"));
    }
    parts.join("\n\n")
}

/// Compress repeated tool call patterns in the message history.
///
/// When the same tool is called 3+ times consecutively (looking at the tail of
/// the messages list), the older tool results are replaced with a compact
/// summary to reduce context token usage. The most recent result is always kept
/// in full so the model has the latest data.
pub(crate) fn compress_repeated_tool_results(messages: &mut [ChatMessage]) {
    // Walk backwards from the end to find consecutive Tool messages that share
    // the same tool_call_id prefix pattern (same tool name).  We look for runs
    // of (Assistant{tool_calls} + Tool*) pairs where the assistant called the
    // same single tool each time.

    // Collect the tail sequence of Tool messages and extract tool names from
    // the preceding Assistant message's tool_calls.
    let len = messages.len();
    if len < 6 {
        return; // Need at least 3 rounds of (assistant + tool) = 6 messages
    }

    // Walk backward to find consecutive (Assistant with single tool_call, Tool) pairs
    // that all call the same tool name.
    let mut run_end = len; // exclusive end index
    let mut tool_name: Option<String> = None;
    let mut count = 0_usize;

    let mut idx = len;
    while idx >= 2 {
        idx -= 1;
        // Check if this is a Tool message
        if !matches!(&messages[idx], ChatMessage::Tool { .. }) {
            break;
        }
        // The message before should be an Assistant with tool_calls
        if idx == 0 {
            break;
        }
        if let ChatMessage::Assistant {
            tool_calls: tcs, ..
        } = &messages[idx - 1]
        {
            if tcs.len() != 1 {
                break; // Only compress single-tool-call rounds
            }
            let name = map_tool_name(&tcs[0].name);
            match &tool_name {
                None => {
                    tool_name = Some(name.to_string());
                    count = 1;
                    run_end = idx + 1; // include this Tool message
                }
                Some(prev) if prev.as_str() == name => {
                    count += 1;
                }
                _ => break, // Different tool, stop
            }
            idx -= 1; // skip the Assistant message
        } else {
            break;
        }
    }

    if count < 3 {
        return; // Not enough repetitions to compress
    }

    // The run covers messages from idx+1 to run_end (exclusive).
    // Keep the last (assistant + tool) pair intact, compress the rest.
    let run_start = idx + 1;
    let pairs_to_compress = count - 1; // keep last pair
    let msgs_to_compress = pairs_to_compress * 2; // each pair is 2 messages

    if msgs_to_compress == 0 {
        return;
    }

    let tn = tool_name.unwrap_or_default();
    let summary = format!(
        "[{} prior calls to {} compressed — showing latest result only]",
        pairs_to_compress, tn
    );

    // Replace the compressed region with a single User note.
    // We must keep the message structure valid: the API expects each
    // tool_call_id in an Assistant message to have a matching Tool response.
    // So instead of removing them outright, replace each compressed Tool
    // message content with a one-line summary, and clear the assistant text.
    for i in 0..msgs_to_compress {
        let msg_idx = run_start + i;
        if msg_idx >= run_end {
            break;
        }
        match &mut messages[msg_idx] {
            ChatMessage::Tool { content, .. } => {
                if i == 0 {
                    *content = summary.clone();
                } else {
                    *content = format!("[compressed — call {} of {}]", (i / 2) + 1, count);
                }
            }
            ChatMessage::Assistant { content, .. } => {
                *content = None; // strip any intermediate text
            }
            _ => {}
        }
    }
}

pub(crate) fn summarize_chat_messages(messages: &[ChatMessage]) -> String {
    let mut entries: Vec<String> = Vec::new();
    for msg in messages {
        match msg {
            ChatMessage::System { .. } => {}
            ChatMessage::User { content } => {
                let truncated = if content.len() > 200 {
                    format!("{}...", &content[..content.floor_char_boundary(200)])
                } else {
                    content.clone()
                };
                entries.push(format!("- User: {truncated}"));
            }
            ChatMessage::Assistant {
                content,
                tool_calls,
                ..
            } => {
                if let Some(text) = content {
                    let truncated = if text.len() > 200 {
                        format!("{}...", &text[..text.floor_char_boundary(200)])
                    } else {
                        text.clone()
                    };
                    entries.push(format!("- Assistant: {truncated}"));
                }
                for tc in tool_calls {
                    let summary = crate::summarize_tool_args(
                        map_tool_name(&tc.name),
                        &serde_json::from_str(&tc.arguments).unwrap_or(json!({})),
                    );
                    entries.push(format!("- Tool call: {}({})", tc.name, summary));
                }
            }
            ChatMessage::Tool { content, .. } => {
                let truncated = if content.len() > 100 {
                    format!("{}...", &content[..content.floor_char_boundary(100)])
                } else {
                    content.clone()
                };
                entries.push(format!("- Tool result: {truncated}"));
            }
        }
    }
    if entries.len() > 30 {
        let head = &entries[..15];
        let tail = &entries[entries.len() - 15..];
        format!(
            "{}\n... ({} entries omitted) ...\n{}",
            head.join("\n"),
            entries.len() - 30,
            tail.join("\n")
        )
    } else {
        entries.join("\n")
    }
}
