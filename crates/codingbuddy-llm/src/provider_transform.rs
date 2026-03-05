use anyhow::{Result, anyhow};
use codingbuddy_core::{
    ChatMessage, ChatRequest, LlmResponse, ModelCapabilities, ModelFamily, ProviderKind,
    ToolDefinition, repair_tool_api_name,
};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

pub(crate) struct PreparedToolPayload {
    pub tools: Vec<Value>,
    pub shim_only: bool,
}

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
                    && provider_supports_reasoning_content(capabilities.provider)
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
                    let normalized_id = normalize_tool_call_id_for_provider(
                        &tool_call.id,
                        tool_call_seq,
                        capabilities,
                    );
                    tool_call_seq = tool_call_seq.saturating_add(1);

                    if !tool_call.id.trim().is_empty() {
                        tool_call_id_map.insert(tool_call.id.clone(), normalized_id.clone());
                    }
                    declared_tool_call_ids.insert(normalized_id.clone());
                    let tool_name = repair_tool_api_name(&tool_call.name, &req.tools)
                        .unwrap_or_else(|| tool_call.name.clone());

                    normalized_tool_calls.push(json!({
                        "id": normalized_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_call.arguments,
                        }
                    }));
                }

                if !normalized_tool_calls.is_empty() {
                    msg["tool_calls"] = json!(normalized_tool_calls);
                    has_visible_content = true;
                    if capabilities.strict_empty_content_filtering && msg.get("content").is_none() {
                        // Some strict OpenAI-compatible gateways require explicit
                        // assistant content=null when tool_calls are present.
                        msg["content"] = Value::Null;
                    }
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
                        normalize_tool_call_id_for_provider(
                            tool_call_id,
                            tool_call_seq,
                            capabilities,
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
    repair_message_sequence_for_provider(&mut messages, capabilities);

    Ok(messages)
}

pub(crate) fn postprocess_chat_response(
    mut response: LlmResponse,
    capabilities: &ModelCapabilities,
    available_tools: &[ToolDefinition],
) -> LlmResponse {
    if response.text.trim().is_empty() && !response.reasoning_content.trim().is_empty() {
        response.text = response.reasoning_content.clone();
    }

    for (idx, tool_call) in response.tool_calls.iter_mut().enumerate() {
        if let Some(repaired_name) = repair_tool_api_name(&tool_call.name, available_tools) {
            tool_call.name = repaired_name;
        }

        if capabilities.normalize_tool_call_ids {
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

pub(crate) fn prepare_chat_tools(
    req: &ChatRequest,
    capabilities: &ModelCapabilities,
    provider_base_url: &str,
    endpoint: &str,
) -> Result<Option<PreparedToolPayload>> {
    let mut tools = serialize_tool_definitions(&req.tools)?;
    sanitize_tool_definitions_for_provider(&mut tools, capabilities);

    let shim_only = tools.is_empty()
        && requires_placeholder_tool(req, capabilities, provider_base_url, endpoint);
    if shim_only {
        tools.push(placeholder_tool_definition());
    }

    if tools.is_empty() {
        Ok(None)
    } else {
        Ok(Some(PreparedToolPayload { tools, shim_only }))
    }
}

pub(crate) fn apply_chat_payload_compatibility(
    payload: &mut Value,
    req: &ChatRequest,
    capabilities: &ModelCapabilities,
) {
    if capabilities.provider == ProviderKind::OpenAiCompatible
        && prefers_max_completion_tokens(&req.model)
        && let Some(max_tokens) = payload.get("max_tokens").cloned()
    {
        payload["max_completion_tokens"] = max_tokens;
        if let Some(obj) = payload.as_object_mut() {
            obj.remove("max_tokens");
        }
    }

    if capabilities.provider == ProviderKind::OpenAiCompatible
        && req
            .thinking
            .as_ref()
            .is_some_and(|thinking| thinking.thinking_type == "enabled")
    {
        let effort = thinking_budget_to_reasoning_effort(
            req.thinking.as_ref().and_then(|t| t.budget_tokens),
        );
        payload["reasoning_effort"] = json!(effort);
        if let Some(obj) = payload.as_object_mut() {
            obj.remove("thinking");
        }
    }

    if capabilities.provider == ProviderKind::OpenAiCompatible
        && capabilities.family == ModelFamily::Gemini
        && payload.get("max_output_tokens").is_none()
        && let Some(max_tokens) = payload.get("max_tokens").cloned()
    {
        // Gemini-compatible OpenAI gateways often accept max_output_tokens.
        payload["max_output_tokens"] = max_tokens;
    }

    if capabilities.provider == ProviderKind::OpenAiCompatible
        && payload.get("reasoning_effort").is_some()
        && prefers_max_completion_tokens(&req.model)
        && let Some(obj) = payload.as_object_mut()
    {
        // Strict OpenAI-compatible reasoning gateways commonly reject sampling
        // controls when reasoning_effort is active.
        for key in [
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
            "top_logprobs",
        ] {
            obj.remove(key);
        }
    }

    if capabilities.provider == ProviderKind::OpenAiCompatible
        && capabilities.family == ModelFamily::Gemini
        && payload.get("tool_choice").and_then(Value::as_str) == Some("required")
    {
        // Gemini gateways are inconsistent with required tool_choice.
        payload["tool_choice"] = json!("auto");
    }

    if capabilities.provider == ProviderKind::Ollama
        && payload.get("tool_choice").and_then(Value::as_str) == Some("required")
    {
        // Ollama accepts tool_choice but "required" is inconsistently supported
        // across model families and fronting gateways.
        payload["tool_choice"] = json!("auto");
    }

    if capabilities.provider == ProviderKind::Ollama
        && let Some(max_tokens) = payload.get("max_tokens").cloned()
    {
        // Ollama native/runtime adapters typically use options.num_predict.
        if !payload.get("options").is_some_and(Value::is_object) {
            payload["options"] = json!({});
        }
        payload["options"]["num_predict"] = max_tokens;
    }
}

fn serialize_tool_definitions(tools: &[ToolDefinition]) -> Result<Vec<Value>> {
    tools
        .iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn sanitize_tool_definitions_for_provider(tools: &mut [Value], capabilities: &ModelCapabilities) {
    if capabilities.provider == ProviderKind::OpenAiCompatible
        && capabilities.family == ModelFamily::Gemini
    {
        for tool in tools {
            if let Some(schema) = tool
                .get_mut("function")
                .and_then(Value::as_object_mut)
                .and_then(|function| function.get_mut("parameters"))
            {
                sanitize_gemini_schema(schema);
            }
        }
    }
}

fn requires_placeholder_tool(
    req: &ChatRequest,
    capabilities: &ModelCapabilities,
    provider_base_url: &str,
    endpoint: &str,
) -> bool {
    if capabilities.provider != ProviderKind::OpenAiCompatible {
        return false;
    }
    let lower_base = provider_base_url.to_ascii_lowercase();
    let lower_endpoint = endpoint.to_ascii_lowercase();
    let looks_like_litellm = lower_base.contains("litellm") || lower_endpoint.contains("litellm");
    if !looks_like_litellm {
        return false;
    }
    history_uses_tool_protocol(req)
}

fn history_uses_tool_protocol(req: &ChatRequest) -> bool {
    req.messages.iter().any(|message| match message {
        ChatMessage::Assistant { tool_calls, .. } => !tool_calls.is_empty(),
        ChatMessage::Tool { .. } => true,
        _ => false,
    })
}

fn placeholder_tool_definition() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "_noop",
            "description": "Placeholder tool for proxy compatibility when tool history exists but no active tools are needed.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    })
}

fn sanitize_gemini_schema(schema: &mut Value) {
    match schema {
        Value::Array(items) => {
            for item in items {
                sanitize_gemini_schema(item);
            }
        }
        Value::Object(map) => {
            let keys = map.keys().cloned().collect::<Vec<_>>();
            for key in keys {
                if key == "enum" {
                    if let Some(enum_values) = map.get_mut(&key).and_then(Value::as_array_mut) {
                        for value in enum_values {
                            if !value.is_string() {
                                let normalized = match value {
                                    Value::Null => "null".to_string(),
                                    Value::Bool(b) => b.to_string(),
                                    Value::Number(n) => n.to_string(),
                                    Value::String(s) => s.clone(),
                                    Value::Array(_) | Value::Object(_) => value.to_string(),
                                };
                                *value = Value::String(normalized);
                            }
                        }
                        if map
                            .get("type")
                            .and_then(Value::as_str)
                            .is_some_and(|ty| ty == "integer" || ty == "number")
                        {
                            map.insert("type".to_string(), Value::String("string".to_string()));
                        }
                    }
                    continue;
                }
                if let Some(value) = map.get_mut(&key) {
                    sanitize_gemini_schema(value);
                }
            }

            let has_combiner = has_schema_combiner(map);
            if map.get("type").and_then(Value::as_str) == Some("object") {
                let property_names = map
                    .get("properties")
                    .and_then(Value::as_object)
                    .map(|properties| properties.keys().cloned().collect::<HashSet<String>>());
                if let (Some(property_names), Some(required)) = (
                    property_names,
                    map.get_mut("required").and_then(Value::as_array_mut),
                ) {
                    required.retain(|field| {
                        field
                            .as_str()
                            .is_some_and(|name| property_names.contains(name))
                    });
                }
            }

            if map.get("type").and_then(Value::as_str) == Some("array") && !has_combiner {
                if !map.contains_key("items") || map.get("items").is_some_and(Value::is_null) {
                    map.insert("items".to_string(), json!({}));
                }
                if let Some(items) = map.get_mut("items")
                    && items.as_object().is_some_and(|obj| !has_schema_intent(obj))
                {
                    *items = json!({"type": "string"});
                }
            }

            if map.contains_key("type")
                && map
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|ty| ty != "object")
                && !has_combiner
            {
                map.remove("properties");
                map.remove("required");
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

fn has_schema_combiner(map: &serde_json::Map<String, Value>) -> bool {
    ["anyOf", "oneOf", "allOf"]
        .iter()
        .any(|key| map.get(*key).is_some_and(Value::is_array))
}

fn has_schema_intent(map: &serde_json::Map<String, Value>) -> bool {
    if has_schema_combiner(map) {
        return true;
    }
    [
        "type",
        "properties",
        "items",
        "prefixItems",
        "enum",
        "const",
        "$ref",
        "additionalProperties",
        "patternProperties",
        "required",
        "not",
        "if",
        "then",
        "else",
    ]
    .iter()
    .any(|key| map.contains_key(*key))
}

fn provider_supports_reasoning_content(provider: ProviderKind) -> bool {
    provider == ProviderKind::Deepseek
}

fn prefers_max_completion_tokens(model: &str) -> bool {
    let lower = model.trim().to_ascii_lowercase();
    lower.starts_with("o1")
        || lower.starts_with("o3")
        || lower.starts_with("o4")
        || lower.contains("reasoning")
}

fn thinking_budget_to_reasoning_effort(budget_tokens: Option<u32>) -> &'static str {
    match budget_tokens.unwrap_or(4096) {
        0..=2048 => "low",
        2049..=8192 => "medium",
        _ => "high",
    }
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

fn normalize_tool_call_id_for_provider(
    raw: &str,
    fallback_idx: usize,
    capabilities: &ModelCapabilities,
) -> String {
    if requires_mistral_tool_id_compat(capabilities) {
        return normalize_tool_call_id_mistral(raw, fallback_idx);
    }
    normalize_tool_call_id(raw, fallback_idx, capabilities.normalize_tool_call_ids)
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

fn normalize_tool_call_id_mistral(raw: &str, fallback_idx: usize) -> String {
    let mut normalized = raw
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>();
    if normalized.is_empty() {
        normalized = format!("t{:08}", fallback_idx % 100_000_000);
    }
    if normalized.len() > 9 {
        normalized.truncate(9);
    }
    while normalized.len() < 9 {
        normalized.push('0');
    }
    normalized
}

fn requires_mistral_tool_id_compat(capabilities: &ModelCapabilities) -> bool {
    matches!(
        capabilities.provider,
        ProviderKind::OpenAiCompatible | ProviderKind::Ollama
    ) && capabilities.family == ModelFamily::Mistral
}

fn repair_message_sequence_for_provider(
    messages: &mut Vec<Value>,
    capabilities: &ModelCapabilities,
) {
    if !requires_mistral_tool_id_compat(capabilities) {
        return;
    }
    let mut repaired: Vec<Value> = Vec::with_capacity(messages.len());
    for idx in 0..messages.len() {
        repaired.push(messages[idx].clone());
        let current_role = messages[idx].get("role").and_then(Value::as_str);
        let next_role = messages
            .get(idx + 1)
            .and_then(|next| next.get("role"))
            .and_then(Value::as_str);
        if current_role == Some("tool") && next_role == Some("user") {
            repaired.push(json!({
                "role": "assistant",
                "content": "Done.",
            }));
        }
    }
    *messages = repaired;
}

#[cfg(test)]
mod tests {
    use super::*;
    use codingbuddy_core::{
        ChatMessage, ChatRequest, FunctionDefinition, LlmToolCall, ProviderKind, ToolChoice,
        ToolDefinition, model_capabilities,
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

    fn tool_def(name: &str) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.to_string(),
                description: format!("Tool {name}"),
                strict: None,
                parameters: json!({}),
            },
        }
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

        let normalized = postprocess_chat_response(response, &caps, &[]);
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
    fn response_postprocess_normalizes_built_in_tool_names_to_api_format() {
        let (_, caps) = req_for(ProviderKind::OpenAiCompatible, "gpt-4o-mini");
        let response = LlmResponse {
            text: String::new(),
            finish_reason: "tool_calls".to_string(),
            reasoning_content: String::new(),
            tool_calls: vec![LlmToolCall {
                id: "call_1".to_string(),
                name: "FS.Read".to_string(),
                arguments: "{}".to_string(),
            }],
            usage: None,
        };

        let normalized = postprocess_chat_response(response, &caps, &[]);
        assert_eq!(normalized.tool_calls[0].name, "fs_read");
    }

    #[test]
    fn response_postprocess_repairs_active_plugin_tool_names() {
        let (mut req, caps) = req_for(ProviderKind::OpenAiCompatible, "gpt-4o-mini");
        req.tools = vec![tool_def("plugin__phase_c_plugin__review")];
        let response = LlmResponse {
            text: String::new(),
            finish_reason: "tool_calls".to_string(),
            reasoning_content: String::new(),
            tool_calls: vec![LlmToolCall {
                id: "call_1".to_string(),
                name: "plugin__phase-c-plugin__review".to_string(),
                arguments: "{}".to_string(),
            }],
            usage: None,
        };

        let normalized = postprocess_chat_response(response, &caps, &req.tools);
        assert_eq!(
            normalized.tool_calls[0].name,
            "plugin__phase_c_plugin__review"
        );
    }

    #[test]
    fn preflight_normalizes_historical_tool_call_names() {
        let (mut req, caps) = req_for(ProviderKind::OpenAiCompatible, "gpt-4o-mini");
        req.messages = vec![
            ChatMessage::User {
                content: "run tool".to_string(),
            },
            ChatMessage::Assistant {
                content: None,
                reasoning_content: None,
                tool_calls: vec![LlmToolCall {
                    id: "call_1".to_string(),
                    name: "fs.read".to_string(),
                    arguments: "{}".to_string(),
                }],
            },
        ];

        let messages = preflight_chat_messages(&req, &caps).expect("messages");
        assert_eq!(messages[1]["tool_calls"][0]["function"]["name"], "fs_read");
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

    #[test]
    fn openai_compatible_drops_reasoning_content_from_assistant_messages() {
        let (mut req, caps) = req_for(ProviderKind::OpenAiCompatible, "gpt-4o-mini");
        req.messages = vec![
            ChatMessage::User {
                content: "hello".to_string(),
            },
            ChatMessage::Assistant {
                content: Some("answer".to_string()),
                reasoning_content: Some("hidden reasoning".to_string()),
                tool_calls: vec![],
            },
        ];

        let messages = preflight_chat_messages(&req, &caps).expect("messages");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], "answer");
        assert!(
            messages[1].get("reasoning_content").is_none(),
            "non-deepseek providers should not receive reasoning_content field"
        );
    }

    #[test]
    fn strict_provider_sets_null_content_for_tool_calls() {
        let (mut req, caps) = req_for(ProviderKind::OpenAiCompatible, "gpt-4o-mini");
        req.messages = vec![
            ChatMessage::User {
                content: "run tool".to_string(),
            },
            ChatMessage::Assistant {
                content: None,
                reasoning_content: None,
                tool_calls: vec![LlmToolCall {
                    id: "call_1".to_string(),
                    name: "fs_read".to_string(),
                    arguments: "{}".to_string(),
                }],
            },
        ];

        let messages = preflight_chat_messages(&req, &caps).expect("messages");
        assert_eq!(messages[1]["content"], Value::Null);
        assert!(messages[1].get("tool_calls").is_some());
    }

    #[test]
    fn payload_compat_remaps_max_tokens_for_openai_reasoning_models() {
        let (req, caps) = req_for(ProviderKind::OpenAiCompatible, "o3-mini");
        let mut payload = json!({
            "model": "o3-mini",
            "messages": [],
            "max_tokens": 2048
        });

        apply_chat_payload_compatibility(&mut payload, &req, &caps);
        assert_eq!(payload["max_completion_tokens"], 2048);
        assert!(payload.get("max_tokens").is_none());
    }

    #[test]
    fn payload_compat_maps_thinking_to_reasoning_effort_for_openai_compat() {
        let (mut req, caps) = req_for(ProviderKind::OpenAiCompatible, "gpt-4o-mini");
        req.thinking = Some(codingbuddy_core::ThinkingConfig::enabled(16_384));
        let mut payload = json!({
            "model": "gpt-4o-mini",
            "messages": [],
            "max_tokens": 1024
        });

        apply_chat_payload_compatibility(&mut payload, &req, &caps);
        assert_eq!(payload["reasoning_effort"], "high");
    }

    #[test]
    fn payload_compat_downgrades_required_tool_choice_for_ollama() {
        let (req, caps) = req_for(ProviderKind::Ollama, "qwen2.5-coder:7b");
        let mut payload = json!({
            "model": "qwen2.5-coder:7b",
            "messages": [],
            "tool_choice": "required"
        });

        apply_chat_payload_compatibility(&mut payload, &req, &caps);
        assert_eq!(payload["tool_choice"], "auto");
    }

    #[test]
    fn mistral_repair_inserts_assistant_between_tool_and_user() {
        let (mut req, caps) = req_for(ProviderKind::OpenAiCompatible, "mistral-large");
        req.messages = vec![
            ChatMessage::User {
                content: "run tool".to_string(),
            },
            ChatMessage::Assistant {
                content: None,
                reasoning_content: None,
                tool_calls: vec![LlmToolCall {
                    id: "bad id".to_string(),
                    name: "fs_read".to_string(),
                    arguments: "{}".to_string(),
                }],
            },
            ChatMessage::Tool {
                tool_call_id: "bad id".to_string(),
                content: "ok".to_string(),
            },
            ChatMessage::User {
                content: "continue".to_string(),
            },
        ];

        let messages = preflight_chat_messages(&req, &caps).expect("messages");
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[3]["role"], "assistant");
        assert_eq!(messages[3]["content"], "Done.");
        assert_eq!(messages[4]["role"], "user");
    }

    #[test]
    fn mistral_tool_ids_are_strictly_normalized() {
        let (mut req, caps) = req_for(ProviderKind::OpenAiCompatible, "mistral-large");
        req.messages = vec![
            ChatMessage::User {
                content: "run tools".to_string(),
            },
            ChatMessage::Assistant {
                content: None,
                reasoning_content: None,
                tool_calls: vec![LlmToolCall {
                    id: "bad-id!?".to_string(),
                    name: "fs_read".to_string(),
                    arguments: "{}".to_string(),
                }],
            },
            ChatMessage::Tool {
                tool_call_id: "bad-id!?".to_string(),
                content: "ok".to_string(),
            },
        ];

        let messages = preflight_chat_messages(&req, &caps).expect("messages");
        let call_id = messages[1]["tool_calls"][0]["id"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        assert_eq!(call_id.len(), 9);
        assert!(call_id.chars().all(|ch| ch.is_ascii_alphanumeric()));
        assert_eq!(messages[2]["tool_call_id"], call_id);
    }

    #[test]
    fn payload_compat_adds_gemini_max_output_tokens_alias() {
        let (req, caps) = req_for(ProviderKind::OpenAiCompatible, "gemini-2.0-flash");
        let mut payload = json!({
            "model": "gemini-2.0-flash",
            "messages": [],
            "max_tokens": 1024
        });

        apply_chat_payload_compatibility(&mut payload, &req, &caps);
        assert_eq!(payload["max_output_tokens"], 1024);
        assert_eq!(payload["max_tokens"], 1024);
    }

    #[test]
    fn payload_compat_strips_sampling_fields_for_reasoning_effort_models() {
        let (mut req, caps) = req_for(ProviderKind::OpenAiCompatible, "o3-mini");
        req.thinking = Some(codingbuddy_core::ThinkingConfig::enabled(16_384));
        let mut payload = json!({
            "model": "o3-mini",
            "messages": [],
            "max_tokens": 1024,
            "temperature": 0.5,
            "top_p": 0.9,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.3,
            "logprobs": true,
            "top_logprobs": 3
        });

        apply_chat_payload_compatibility(&mut payload, &req, &caps);
        assert_eq!(payload["reasoning_effort"], "high");
        assert!(payload.get("temperature").is_none());
        assert!(payload.get("top_p").is_none());
        assert!(payload.get("presence_penalty").is_none());
        assert!(payload.get("frequency_penalty").is_none());
        assert!(payload.get("logprobs").is_none());
        assert!(payload.get("top_logprobs").is_none());
    }

    #[test]
    fn payload_compat_downgrades_required_tool_choice_for_gemini_gateways() {
        let (req, caps) = req_for(ProviderKind::OpenAiCompatible, "gemini-2.0-flash");
        let mut payload = json!({
            "model": "gemini-2.0-flash",
            "messages": [],
            "tool_choice": "required"
        });

        apply_chat_payload_compatibility(&mut payload, &req, &caps);
        assert_eq!(payload["tool_choice"], "auto");
    }

    #[test]
    fn payload_compat_maps_ollama_max_tokens_to_num_predict_option() {
        let (req, caps) = req_for(ProviderKind::Ollama, "qwen2.5-coder:7b");
        let mut payload = json!({
            "model": "qwen2.5-coder:7b",
            "messages": [],
            "max_tokens": 2048
        });

        apply_chat_payload_compatibility(&mut payload, &req, &caps);
        assert_eq!(payload["options"]["num_predict"], 2048);
    }

    #[test]
    fn prepare_chat_tools_adds_placeholder_tool_for_litellm_tool_history() {
        let (mut req, caps) = req_for(ProviderKind::OpenAiCompatible, "gpt-4o-mini");
        req.messages = vec![
            ChatMessage::User {
                content: "run tools".to_string(),
            },
            ChatMessage::Assistant {
                content: None,
                reasoning_content: None,
                tool_calls: vec![LlmToolCall {
                    id: "call_1".to_string(),
                    name: "fs_read".to_string(),
                    arguments: "{}".to_string(),
                }],
            },
            ChatMessage::Tool {
                tool_call_id: "call_1".to_string(),
                content: "ok".to_string(),
            },
        ];

        let prepared = prepare_chat_tools(
            &req,
            &caps,
            "https://litellm.internal",
            "https://litellm.internal/v1/chat/completions",
        )
        .expect("prepared")
        .expect("placeholder");

        assert!(prepared.shim_only);
        assert_eq!(prepared.tools.len(), 1);
        assert_eq!(prepared.tools[0]["function"]["name"], "_noop");
    }

    #[test]
    fn prepare_chat_tools_skips_placeholder_without_tool_history() {
        let (req, caps) = req_for(ProviderKind::OpenAiCompatible, "gpt-4o-mini");

        let prepared = prepare_chat_tools(
            &req,
            &caps,
            "https://litellm.internal",
            "https://litellm.internal/v1/chat/completions",
        )
        .expect("prepared");

        assert!(prepared.is_none());
    }

    #[test]
    fn prepare_chat_tools_sanitizes_gemini_tool_schema() {
        let (mut req, caps) = req_for(ProviderKind::OpenAiCompatible, "gemini-2.0-flash");
        req.tools = vec![ToolDefinition {
            tool_type: "function".to_string(),
            function: codingbuddy_core::FunctionDefinition {
                name: "pick_mode".to_string(),
                description: "Pick a mode".to_string(),
                strict: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "integer",
                            "enum": [1, 2]
                        },
                        "items": {
                            "type": "array",
                            "items": {}
                        },
                        "status": {
                            "type": "string",
                            "properties": {
                                "unused": { "type": "string" }
                            },
                            "required": ["unused"]
                        }
                    },
                    "required": ["mode", "missing"]
                }),
            },
        }];

        let prepared = prepare_chat_tools(
            &req,
            &caps,
            "https://api.openai.com",
            "https://api.openai.com/v1/chat/completions",
        )
        .expect("prepared")
        .expect("tools");

        assert!(!prepared.shim_only);
        let schema = &prepared.tools[0]["function"]["parameters"];
        assert_eq!(schema["required"], json!(["mode"]));
        assert_eq!(schema["properties"]["mode"]["type"], "string");
        assert_eq!(schema["properties"]["mode"]["enum"], json!(["1", "2"]));
        assert_eq!(schema["properties"]["items"]["items"]["type"], "string");
        assert!(
            schema["properties"]["status"].get("properties").is_none(),
            "non-object schema members should not retain object-only keys"
        );
        assert!(
            schema["properties"]["status"].get("required").is_none(),
            "non-object schema members should not retain required"
        );
    }
}
