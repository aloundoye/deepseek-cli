use anyhow::{Result, anyhow};
use codingbuddy_core::{ChatMessage, ChatRequest, LlmResponse, ModelCapabilities};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

pub(crate) fn preflight_chat_messages(
    req: &ChatRequest,
    capabilities: &ModelCapabilities,
) -> Result<Vec<Value>> {
    if !req.images.is_empty() && !capabilities.supports_image_input {
        return Err(anyhow!(
            "model '{}' on provider '{}' does not support image input; choose a vision-capable model or remove images",
            req.model,
            capabilities.provider.as_key()
        ));
    }

    let mut messages: Vec<Value> = Vec::new();
    let mut tool_call_id_map: HashMap<String, String> = HashMap::new();
    let mut declared_tool_call_ids: HashSet<String> = HashSet::new();
    let mut tool_call_seq: usize = 1;

    for message in &req.messages {
        match message {
            ChatMessage::System { content } => {
                if capabilities.strict_empty_content_filtering && content.trim().is_empty() {
                    continue;
                }
                messages.push(json!({"role": "system", "content": content}));
            }
            ChatMessage::User { content } => {
                if capabilities.strict_empty_content_filtering && content.trim().is_empty() {
                    continue;
                }
                messages.push(json!({"role": "user", "content": content}));
            }
            ChatMessage::Assistant {
                content,
                reasoning_content,
                tool_calls,
            } => {
                let mut msg = json!({"role": "assistant"});
                let mut has_visible_content = false;

                if let Some(c) = content
                    && (!capabilities.strict_empty_content_filtering || !c.trim().is_empty())
                {
                    msg["content"] = json!(c);
                    has_visible_content = true;
                }

                if let Some(rc) = reasoning_content
                    && (!capabilities.strict_empty_content_filtering || !rc.trim().is_empty())
                {
                    msg["reasoning_content"] = json!(rc);
                    has_visible_content = true;
                }

                let mut normalized_tool_calls: Vec<Value> = Vec::new();
                for tool_call in tool_calls {
                    if capabilities.strict_empty_content_filtering
                        && tool_call.name.trim().is_empty()
                    {
                        tool_call_seq = tool_call_seq.saturating_add(1);
                        continue;
                    }
                    let normalized_id = normalize_tool_call_id(
                        &tool_call.id,
                        tool_call_seq,
                        capabilities.normalize_tool_call_ids,
                    );
                    tool_call_seq = tool_call_seq.saturating_add(1);

                    if !tool_call.id.trim().is_empty() {
                        tool_call_id_map.insert(tool_call.id.clone(), normalized_id.clone());
                    }
                    declared_tool_call_ids.insert(normalized_id.clone());

                    normalized_tool_calls.push(json!({
                        "id": normalized_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        }
                    }));
                }

                if !normalized_tool_calls.is_empty() {
                    msg["tool_calls"] = json!(normalized_tool_calls);
                    has_visible_content = true;
                }

                if !capabilities.strict_empty_content_filtering || has_visible_content {
                    messages.push(msg);
                }
            }
            ChatMessage::Tool {
                tool_call_id,
                content,
            } => {
                let normalized_tool_call_id = tool_call_id_map
                    .get(tool_call_id)
                    .cloned()
                    .unwrap_or_else(|| {
                        normalize_tool_call_id(
                            tool_call_id,
                            tool_call_seq,
                            capabilities.normalize_tool_call_ids,
                        )
                    });
                if !tool_call_id_map.contains_key(tool_call_id) {
                    tool_call_seq = tool_call_seq.saturating_add(1);
                }

                if capabilities.strict_empty_content_filtering {
                    if content.trim().is_empty() || normalized_tool_call_id.trim().is_empty() {
                        continue;
                    }
                    if !declared_tool_call_ids.contains(&normalized_tool_call_id) {
                        // Drop dangling tool messages that do not map to assistant tool calls.
                        continue;
                    }
                }

                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": normalized_tool_call_id,
                    "content": content,
                }));
            }
        }
    }

    if !req.images.is_empty() {
        attach_images_to_last_user_message(
            &mut messages,
            &req.images,
            capabilities.strict_empty_content_filtering,
        );
    }

    Ok(messages)
}

pub(crate) fn postprocess_chat_response(
    mut response: LlmResponse,
    capabilities: &ModelCapabilities,
) -> LlmResponse {
    if response.text.trim().is_empty() && !response.reasoning_content.trim().is_empty() {
        response.text = response.reasoning_content.clone();
    }

    if capabilities.normalize_tool_call_ids {
        for (idx, tool_call) in response.tool_calls.iter_mut().enumerate() {
            tool_call.id = normalize_tool_call_id(&tool_call.id, idx + 1, true);
        }
    }

    if let Some(usage) = response.usage.as_mut() {
        if usage.prompt_cache_hit_tokens > usage.prompt_tokens {
            usage.prompt_cache_hit_tokens = usage.prompt_tokens;
        }
        let remaining_prompt_tokens = usage
            .prompt_tokens
            .saturating_sub(usage.prompt_cache_hit_tokens);
        if usage.prompt_cache_miss_tokens > remaining_prompt_tokens {
            usage.prompt_cache_miss_tokens = remaining_prompt_tokens;
        }
    }

    response
}

fn attach_images_to_last_user_message(
    messages: &mut Vec<Value>,
    images: &[codingbuddy_core::ImageContent],
    strict_filtering: bool,
) {
    if let Some(last_user) = messages.iter_mut().rev().find(|msg| msg["role"] == "user") {
        let text = extract_user_text(last_user.get("content"));
        let mut parts: Vec<Value> = Vec::new();
        if !strict_filtering || !text.trim().is_empty() {
            parts.push(json!({"type": "text", "text": text}));
        }
        for image in images {
            parts.push(json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{};base64,{}", image.mime, image.base64_data),
                }
            }));
        }
        last_user["content"] = json!(parts);
    } else {
        let mut parts: Vec<Value> = Vec::new();
        if !strict_filtering {
            parts.push(json!({"type": "text", "text": ""}));
        }
        for image in images {
            parts.push(json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{};base64,{}", image.mime, image.base64_data),
                }
            }));
        }
        messages.push(json!({"role": "user", "content": parts}));
    }
}

fn extract_user_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .find_map(|part| {
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    return part
                        .get("text")
                        .and_then(Value::as_str)
                        .map(ToString::to_string);
                }
                None
            })
            .unwrap_or_default(),
        _ => String::new(),
    }
}

fn normalize_tool_call_id(raw: &str, fallback_idx: usize, normalize: bool) -> String {
    let trimmed = raw.trim();
    if !normalize {
        if trimmed.is_empty() {
            return format!("tool_call_{}", fallback_idx);
        }
        return trimmed.to_string();
    }

    let mut sanitized = String::with_capacity(trimmed.len());
    for ch in trimmed.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            sanitized.push(ch);
        } else if !sanitized.ends_with('_') {
            sanitized.push('_');
        }
    }

    let mut normalized = sanitized.trim_matches('_').to_string();
    if normalized.is_empty() {
        normalized = format!("tool_call_{}", fallback_idx);
    }

    if normalized
        .chars()
        .next()
        .is_none_or(|ch| !ch.is_ascii_alphabetic())
    {
        normalized = format!("call_{}", normalized);
    }

    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    use codingbuddy_core::{
        ChatMessage, ChatRequest, LlmToolCall, ProviderKind, ToolChoice, model_capabilities,
    };

    fn req_for(provider: ProviderKind, model: &str) -> (ChatRequest, ModelCapabilities) {
        let req = ChatRequest {
            model: model.to_string(),
            messages: vec![ChatMessage::User {
                content: "hello".to_string(),
            }],
            tools: vec![],
            tool_choice: ToolChoice::none(),
            max_tokens: 128,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            logprobs: None,
            top_logprobs: None,
            thinking: None,
            images: vec![],
            response_format: None,
        };
        (req, model_capabilities(provider, model))
    }

    #[test]
    fn strict_filtering_drops_empty_messages_for_ollama() {
        let (mut req, caps) = req_for(ProviderKind::Ollama, "qwen2.5-coder:7b");
        req.messages = vec![
            ChatMessage::System {
                content: "   ".to_string(),
            },
            ChatMessage::User {
                content: "hello".to_string(),
            },
            ChatMessage::Assistant {
                content: Some(" ".to_string()),
                reasoning_content: None,
                tool_calls: vec![],
            },
        ];

        let messages = preflight_chat_messages(&req, &caps).expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
    }

    #[test]
    fn deepseek_keeps_empty_assistant_message_for_non_strict_provider() {
        let (mut req, caps) = req_for(ProviderKind::Deepseek, "deepseek-chat");
        req.messages = vec![
            ChatMessage::User {
                content: "hello".to_string(),
            },
            ChatMessage::Assistant {
                content: Some(" ".to_string()),
                reasoning_content: None,
                tool_calls: vec![],
            },
        ];

        let messages = preflight_chat_messages(&req, &caps).expect("messages");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["role"], "assistant");
    }

    #[test]
    fn ollama_normalizes_tool_call_ids_and_tool_links() {
        let (mut req, caps) = req_for(ProviderKind::Ollama, "qwen2.5-coder:7b");
        req.messages = vec![
            ChatMessage::User {
                content: "run tools".to_string(),
            },
            ChatMessage::Assistant {
                content: None,
                reasoning_content: None,
                tool_calls: vec![LlmToolCall {
                    id: " 123 bad id ".to_string(),
                    name: "fs_read".to_string(),
                    arguments: "{}".to_string(),
                }],
            },
            ChatMessage::Tool {
                tool_call_id: " 123 bad id ".to_string(),
                content: "ok".to_string(),
            },
        ];

        let messages = preflight_chat_messages(&req, &caps).expect("messages");
        let tool_call_id = messages[1]["tool_calls"][0]["id"]
            .as_str()
            .unwrap_or_default();
        assert_eq!(tool_call_id, "call_123_bad_id");
        assert_eq!(messages[2]["tool_call_id"], "call_123_bad_id");
    }

    #[test]
    fn ollama_rejects_images_for_non_vision_model() {
        let (mut req, caps) = req_for(ProviderKind::Ollama, "qwen2.5-coder:7b");
        req.images = vec![codingbuddy_core::ImageContent {
            mime: "image/png".to_string(),
            base64_data: "AAAA".to_string(),
        }];

        let err = preflight_chat_messages(&req, &caps).expect_err("should reject");
        assert!(
            err.to_string().contains("does not support image input"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn vision_model_allows_images_for_ollama() {
        let (mut req, caps) = req_for(ProviderKind::Ollama, "llava:13b");
        req.images = vec![codingbuddy_core::ImageContent {
            mime: "image/png".to_string(),
            base64_data: "AAAA".to_string(),
        }];

        let messages = preflight_chat_messages(&req, &caps).expect("should allow images");
        let content = messages[0]["content"]
            .as_array()
            .expect("multipart content");
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
    }

    #[test]
    fn response_postprocess_falls_back_to_reasoning_and_normalizes_usage() {
        let (_, caps) = req_for(ProviderKind::Ollama, "qwen2.5-coder:7b");
        let response = LlmResponse {
            text: String::new(),
            finish_reason: "stop".to_string(),
            reasoning_content: "thoughts".to_string(),
            tool_calls: vec![LlmToolCall {
                id: "%bad".to_string(),
                name: "fs_read".to_string(),
                arguments: "{}".to_string(),
            }],
            usage: Some(codingbuddy_core::TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                prompt_cache_hit_tokens: 20,
                prompt_cache_miss_tokens: 100,
                reasoning_tokens: 1,
            }),
        };

        let normalized = postprocess_chat_response(response, &caps);
        assert_eq!(normalized.text, "thoughts");
        assert_eq!(normalized.tool_calls[0].id, "bad");
        assert_eq!(
            normalized
                .usage
                .as_ref()
                .expect("usage")
                .prompt_cache_hit_tokens,
            10
        );
        assert_eq!(
            normalized
                .usage
                .as_ref()
                .expect("usage")
                .prompt_cache_miss_tokens,
            0
        );
    }

    #[test]
    fn dangling_tool_message_is_dropped_for_strict_provider() {
        let (mut req, caps) = req_for(ProviderKind::OpenAiCompatible, "gpt-4o-mini");
        req.messages = vec![
            ChatMessage::User {
                content: "hello".to_string(),
            },
            ChatMessage::Tool {
                tool_call_id: "missing".to_string(),
                content: "result".to_string(),
            },
        ];

        let messages = preflight_chat_messages(&req, &caps).expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
    }
}
