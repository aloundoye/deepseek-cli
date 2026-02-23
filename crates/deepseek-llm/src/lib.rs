use anyhow::{Result, anyhow};
use chrono::{DateTime, NaiveDateTime, Utc};
use deepseek_core::{
    CacheStrategy, ChatMessage, ChatRequest, DEEPSEEK_PROFILE_V32_SPECIALE, LlmConfig, LlmRequest,
    LlmResponse, LlmToolCall, StreamCallback, StreamChunk, normalize_deepseek_model,
    normalize_deepseek_profile,
};
use reqwest::StatusCode;
use reqwest::blocking::Client;
use reqwest::header::RETRY_AFTER;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::io::BufRead;
use std::thread;
use std::time::Duration;

/// Base delay for network/transport error retries (1s, 2s, 4s exponential backoff).
const NETWORK_RETRY_BASE_MS: u64 = 1000;

pub trait LlmClient {
    fn complete(&self, req: &LlmRequest) -> Result<LlmResponse>;

    /// Streaming variant that invokes `cb` for each token chunk as it arrives.
    /// Returns the fully assembled `LlmResponse` once the stream ends.
    fn complete_streaming(&self, req: &LlmRequest, cb: StreamCallback) -> Result<LlmResponse>;

    /// Chat completion with tool definitions (function calling).
    /// Sends a multi-turn conversation with tool schemas and returns the response.
    fn complete_chat(&self, req: &ChatRequest) -> Result<LlmResponse>;

    /// Streaming chat completion with tool definitions.
    fn complete_chat_streaming(&self, req: &ChatRequest, cb: StreamCallback)
    -> Result<LlmResponse>;
}

#[derive(Debug, Clone)]
pub struct DeepSeekClient {
    cfg: LlmConfig,
    client: Client,
}

impl DeepSeekClient {
    pub fn new(cfg: LlmConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_seconds))
            .build()?;
        Ok(Self { cfg, client })
    }

    fn complete_inner(&self, req: &LlmRequest, api_key: &str) -> Result<LlmResponse> {
        let mut server_cache_enabled = self.cfg.prompt_cache_enabled;
        let mut payload = self.build_payload(req);

        let mut last_err: Option<anyhow::Error> = None;
        let mut attempt: u8 = 0;
        while attempt <= self.cfg.max_retries {
            let response = self
                .client
                .post(&self.cfg.endpoint)
                .bearer_auth(api_key)
                .json(&payload)
                .send();

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    let retry_after = parse_retry_after_seconds(resp.headers().get(RETRY_AFTER));
                    let body = resp.text()?;
                    if status.is_success() {
                        let parsed = if self.cfg.stream {
                            parse_streaming_payload(&body)?
                        } else {
                            parse_non_streaming_payload(&body)?
                        };
                        return Ok(parsed);
                    }

                    if status == StatusCode::BAD_REQUEST
                        && server_cache_enabled
                        && payload_rejects_cache_control(&body)
                    {
                        server_cache_enabled = false;
                        payload = self.build_payload_with_server_cache(req, server_cache_enabled);
                        continue;
                    }

                    last_err = Some(format_api_error(
                        status,
                        &body,
                        attempt,
                        self.cfg.max_retries,
                    ));
                    if should_retry_status(status) && attempt < self.cfg.max_retries {
                        thread::sleep(retry_delay_ms(self.cfg.retry_base_ms, attempt, retry_after));
                        attempt = attempt.saturating_add(1);
                        continue;
                    }
                    break;
                }
                Err(e) => {
                    last_err = Some(format_transport_error(&e));
                    if should_retry_transport_error(&e) && attempt < self.cfg.max_retries {
                        thread::sleep(retry_delay_ms(NETWORK_RETRY_BASE_MS, attempt, None));
                        attempt = attempt.saturating_add(1);
                        continue;
                    }
                    break;
                }
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow!("deepseek request failed without detailed error")))
    }

    fn build_payload(&self, req: &LlmRequest) -> Value {
        self.build_payload_with_server_cache(req, self.cfg.prompt_cache_enabled)
    }

    fn build_payload_with_server_cache(
        &self,
        req: &LlmRequest,
        enable_server_cache: bool,
    ) -> Value {
        let fast_mode = self.cfg.fast_mode;
        let max_tokens = if fast_mode {
            req.max_tokens.min(2048)
        } else {
            req.max_tokens
        };
        let temperature = if fast_mode {
            self.cfg.temperature.min(0.2)
        } else {
            self.cfg.temperature
        };
        let mut messages = Vec::new();
        if !self.cfg.language.trim().is_empty() && !self.cfg.language.eq_ignore_ascii_case("en") {
            messages.push(json!({
                "role": "system",
                "content": format!(
                    "Respond in {} unless the user explicitly asks for another language.",
                    self.cfg.language
                )
            }));
        }
        if req.images.is_empty() {
            messages.push(json!({"role": "user", "content": req.prompt}));
        } else {
            let mut parts = vec![json!({"type": "text", "text": req.prompt})];
            for img in &req.images {
                parts.push(json!({
                    "type": "image_url",
                    "image_url": {"url": format!("data:{};base64,{}", img.mime, img.base64_data)}
                }));
            }
            messages.push(json!({"role": "user", "content": parts}));
        }
        if enable_server_cache {
            let strategy = &self.cfg.cache_strategy;
            if *strategy != CacheStrategy::Off {
                apply_cache_annotations(&mut messages, strategy);
            }
        }

        let model = normalize_deepseek_model(&req.model).unwrap_or(req.model.as_str());

        json!({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": self.cfg.stream,
            "max_tokens": max_tokens
        })
    }

    fn build_chat_payload(&self, req: &ChatRequest) -> Value {
        self.build_chat_payload_with_cache(req, self.cfg.prompt_cache_enabled)
    }

    fn build_chat_payload_with_cache(&self, req: &ChatRequest, enable_cache: bool) -> Value {
        let thinking_enabled = req
            .thinking
            .as_ref()
            .is_some_and(|t| t.thinking_type == "enabled");

        let mut messages: Vec<Value> = req
            .messages
            .iter()
            .map(|m| match m {
                ChatMessage::System { content } => json!({"role": "system", "content": content}),
                ChatMessage::User { content } => json!({"role": "user", "content": content}),
                ChatMessage::Assistant {
                    content,
                    reasoning_content,
                    tool_calls,
                } => {
                    let mut msg = json!({"role": "assistant"});
                    if let Some(c) = content {
                        msg["content"] = json!(c);
                    }
                    if let Some(rc) = reasoning_content {
                        msg["reasoning_content"] = json!(rc);
                    }
                    if !tool_calls.is_empty() {
                        let tc: Vec<Value> = tool_calls
                            .iter()
                            .map(|tc| {
                                json!({
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": tc.arguments
                                    }
                                })
                            })
                            .collect();
                        msg["tool_calls"] = json!(tc);
                    }
                    msg
                }
                ChatMessage::Tool {
                    tool_call_id,
                    content,
                } => json!({"role": "tool", "tool_call_id": tool_call_id, "content": content}),
            })
            .collect();

        if enable_cache {
            let strategy = &self.cfg.cache_strategy;
            if *strategy != CacheStrategy::Off {
                apply_cache_annotations(&mut messages, strategy);
            }
        }

        let mut payload = json!({
            "model": normalize_deepseek_model(&req.model).unwrap_or(&req.model),
            "messages": messages,
            "max_tokens": req.max_tokens,
            "stream": false
        });
        // DeepSeek API requires temperature to be omitted when thinking is enabled.
        if !thinking_enabled && let Some(temp) = req.temperature {
            payload["temperature"] = json!(temp);
        }
        if let Some(ref thinking) = req.thinking {
            payload["thinking"] = serde_json::to_value(thinking).unwrap_or(json!(null));
        }
        if !req.tools.is_empty() {
            payload["tools"] = serde_json::to_value(&req.tools).unwrap_or(json!([]));
            payload["tool_choice"] =
                serde_json::to_value(&req.tool_choice).unwrap_or(json!("auto"));
        }
        payload
    }

    fn complete_chat_inner(&self, req: &ChatRequest, api_key: &str) -> Result<LlmResponse> {
        let mut server_cache_enabled = self.cfg.prompt_cache_enabled;
        let mut payload = self.build_chat_payload(req);

        let mut last_err: Option<anyhow::Error> = None;
        let mut attempt: u8 = 0;
        while attempt <= self.cfg.max_retries {
            let response = self
                .client
                .post(&self.cfg.endpoint)
                .bearer_auth(api_key)
                .json(&payload)
                .send();

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    let retry_after = parse_retry_after_seconds(resp.headers().get(RETRY_AFTER));
                    let body = resp.text()?;
                    if status.is_success() {
                        return parse_non_streaming_payload(&body);
                    }
                    // Cache control rejection fallback
                    if status == StatusCode::BAD_REQUEST
                        && server_cache_enabled
                        && payload_rejects_cache_control(&body)
                    {
                        server_cache_enabled = false;
                        payload = self.build_chat_payload_with_cache(req, false);
                        continue;
                    }
                    last_err = Some(format_api_error(
                        status,
                        &body,
                        attempt,
                        self.cfg.max_retries,
                    ));
                    if should_retry_status(status) && attempt < self.cfg.max_retries {
                        thread::sleep(retry_delay_ms(self.cfg.retry_base_ms, attempt, retry_after));
                        attempt = attempt.saturating_add(1);
                        continue;
                    }
                    break;
                }
                Err(e) => {
                    last_err = Some(format_transport_error(&e));
                    if should_retry_transport_error(&e) && attempt < self.cfg.max_retries {
                        thread::sleep(retry_delay_ms(NETWORK_RETRY_BASE_MS, attempt, None));
                        attempt = attempt.saturating_add(1);
                        continue;
                    }
                    break;
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow!("chat request failed")))
    }

    fn complete_chat_streaming_inner(
        &self,
        req: &ChatRequest,
        api_key: &str,
        cb: StreamCallback,
    ) -> Result<LlmResponse> {
        let mut server_cache_enabled = self.cfg.prompt_cache_enabled;
        let mut payload = self.build_chat_payload(req);
        payload["stream"] = json!(true);

        let mut last_err: Option<anyhow::Error> = None;
        let mut attempt: u8 = 0;
        while attempt <= self.cfg.max_retries {
            let response = self
                .client
                .post(&self.cfg.endpoint)
                .bearer_auth(api_key)
                .json(&payload)
                .send();

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    let retry_after = parse_retry_after_seconds(resp.headers().get(RETRY_AFTER));

                    if status.is_success() {
                        let mut content_out = String::new();
                        let mut reasoning_out = String::new();
                        let mut finish_reason: Option<String> = None;
                        let mut tool_call_parts: BTreeMap<u64, StreamToolCall> = BTreeMap::new();
                        let completed_tool_calls: Vec<LlmToolCall> = Vec::new();
                        // Track whether we're inside a DSML markup block or a
                        // plain-text tool call so we can buffer instead of emitting
                        // to the TUI.
                        let mut dsml_buffering = false;
                        // Suppress text display once structured tool_calls
                        // deltas arrive — the text fragments between tool calls
                        // are noise that confuses the TUI display.
                        let mut has_structured_tool_calls = false;

                        let reader = std::io::BufReader::new(resp);
                        for line_result in reader.lines() {
                            let line = match line_result {
                                Ok(l) => l,
                                Err(e) => {
                                    last_err = Some(anyhow!("stream read error: {e}"));
                                    break;
                                }
                            };
                            let trimmed = line.trim();
                            if !trimmed.starts_with("data:") {
                                continue;
                            }
                            let chunk = trimmed.trim_start_matches("data:").trim();
                            if chunk == "[DONE]" {
                                cb(StreamChunk::Done);
                                break;
                            }
                            let value: Value = match serde_json::from_str(chunk) {
                                Ok(v) => v,
                                Err(_) => continue,
                            };
                            let choice = value
                                .get("choices")
                                .and_then(|v| v.as_array())
                                .and_then(|arr| arr.first());
                            let Some(choice) = choice else {
                                continue;
                            };
                            if let Some(reason) =
                                choice.get("finish_reason").and_then(|v| v.as_str())
                            {
                                finish_reason = Some(reason.to_string());
                            }
                            if let Some(delta) = choice.get("delta") {
                                if let Some(content) = delta.get("content").and_then(|v| v.as_str())
                                {
                                    content_out.push_str(content);
                                    // Suppress DSML markup from the streaming
                                    // display — buffer it instead of emitting.
                                    if !dsml_buffering && contains_dsml_markup(content) {
                                        dsml_buffering = true;
                                    }
                                    if !dsml_buffering && !has_structured_tool_calls {
                                        // Check if accumulated content started
                                        // a DSML block we missed on earlier chunks.
                                        if contains_dsml_markup(&content_out) {
                                            dsml_buffering = true;
                                        } else {
                                            cb(StreamChunk::ContentDelta(content.to_string()));
                                        }
                                    }
                                }
                                if let Some(reasoning) =
                                    delta.get("reasoning_content").and_then(|v| v.as_str())
                                {
                                    reasoning_out.push_str(reasoning);
                                    cb(StreamChunk::ReasoningDelta(reasoning.to_string()));
                                }
                                if let Some(tool_calls) =
                                    delta.get("tool_calls").and_then(|v| v.as_array())
                                {
                                    has_structured_tool_calls = true;
                                    merge_stream_tool_calls(tool_calls, &mut tool_call_parts);
                                }
                            }
                        }

                        if let Some(err) = last_err.take() {
                            return Err(err);
                        }

                        // If content text was streamed to the TUI but we also
                        // have structured tool calls, tell the TUI to clear the
                        // noise text fragments that appeared before/between
                        // tool calls.
                        if has_structured_tool_calls && !content_out.is_empty() {
                            cb(StreamChunk::ClearStreamingText);
                        }

                        let mut tool_calls: Vec<LlmToolCall> = tool_call_parts
                            .into_iter()
                            .filter_map(|(index, value)| {
                                if value.name.trim().is_empty() {
                                    return None;
                                }
                                Some(LlmToolCall {
                                    id: value
                                        .id
                                        .unwrap_or_else(|| format!("tool_call_{}", index + 1)),
                                    name: value.name,
                                    arguments: value.arguments,
                                })
                            })
                            .collect();
                        if !completed_tool_calls.is_empty() {
                            tool_calls.extend(completed_tool_calls);
                        }

                        let text = if !content_out.is_empty() {
                            content_out
                        } else if !reasoning_out.is_empty() {
                            reasoning_out.clone()
                        } else {
                            String::new()
                        };

                        return Ok(LlmResponse {
                            text,
                            finish_reason: finish_reason.unwrap_or_else(|| "stop".to_string()),
                            reasoning_content: reasoning_out,
                            tool_calls,
                        });
                    }

                    let body = resp.text().unwrap_or_default();
                    // Cache control rejection fallback
                    if status == StatusCode::BAD_REQUEST
                        && server_cache_enabled
                        && payload_rejects_cache_control(&body)
                    {
                        server_cache_enabled = false;
                        payload = self.build_chat_payload_with_cache(req, false);
                        payload["stream"] = json!(true);
                        continue;
                    }
                    last_err = Some(format_api_error(
                        status,
                        &body,
                        attempt,
                        self.cfg.max_retries,
                    ));
                    if should_retry_status(status) && attempt < self.cfg.max_retries {
                        thread::sleep(retry_delay_ms(self.cfg.retry_base_ms, attempt, retry_after));
                        attempt = attempt.saturating_add(1);
                        continue;
                    }
                    break;
                }
                Err(e) => {
                    last_err = Some(format_transport_error(&e));
                    if should_retry_transport_error(&e) && attempt < self.cfg.max_retries {
                        thread::sleep(retry_delay_ms(NETWORK_RETRY_BASE_MS, attempt, None));
                        attempt = attempt.saturating_add(1);
                        continue;
                    }
                    break;
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow!("chat streaming request failed")))
    }

    fn resolve_api_key(&self) -> Option<String> {
        std::env::var(&self.cfg.api_key_env)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .or_else(|| {
                self.cfg
                    .api_key
                    .as_ref()
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty())
            })
    }

    fn resolve_request_model(&self, requested: &str, profile: &str) -> Result<String> {
        let normalized = normalize_deepseek_model(requested).ok_or_else(|| {
            anyhow!(
                "unsupported model '{}' (supported aliases: deepseek-chat, deepseek-reasoner, deepseek-v3.2, deepseek-v3.2-speciale)",
                requested
            )
        })?;
        if is_speciale_alias(requested) && profile != DEEPSEEK_PROFILE_V32_SPECIALE {
            return Err(anyhow!(
                "model '{}' requires llm.profile='v3_2_speciale'",
                requested
            ));
        }
        Ok(normalized.to_string())
    }

    /// Streaming variant: reads the SSE response line-by-line, invoking `cb`
    /// for each content/reasoning delta, then returns the assembled response.
    fn complete_streaming_inner(
        &self,
        req: &LlmRequest,
        api_key: &str,
        cb: StreamCallback,
    ) -> Result<LlmResponse> {
        let mut server_cache_enabled = self.cfg.prompt_cache_enabled;
        let mut payload = self.build_payload(req);
        // Force streaming on for the HTTP request
        payload["stream"] = json!(true);

        let mut last_err: Option<anyhow::Error> = None;
        let mut attempt: u8 = 0;
        while attempt <= self.cfg.max_retries {
            let response = self
                .client
                .post(&self.cfg.endpoint)
                .bearer_auth(api_key)
                .json(&payload)
                .send();

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    let retry_after = parse_retry_after_seconds(resp.headers().get(RETRY_AFTER));

                    if status.is_success() {
                        // Read SSE line-by-line, invoking callback for each delta
                        let mut content_out = String::new();
                        let mut reasoning_out = String::new();
                        let mut finish_reason: Option<String> = None;
                        let mut tool_call_parts: BTreeMap<u64, StreamToolCall> = BTreeMap::new();
                        let mut completed_tool_calls = Vec::new();

                        let reader = std::io::BufReader::new(resp);
                        for line_result in reader.lines() {
                            let line = match line_result {
                                Ok(l) => l,
                                Err(e) => {
                                    last_err = Some(anyhow!("stream read error: {e}"));
                                    break;
                                }
                            };
                            let trimmed = line.trim();
                            if !trimmed.starts_with("data:") {
                                continue;
                            }
                            let chunk = trimmed.trim_start_matches("data:").trim();
                            if chunk == "[DONE]" {
                                cb(StreamChunk::Done);
                                break;
                            }
                            let value: Value = match serde_json::from_str(chunk) {
                                Ok(v) => v,
                                Err(_) => continue,
                            };
                            let choice = value
                                .get("choices")
                                .and_then(|v| v.as_array())
                                .and_then(|arr| arr.first());
                            let Some(choice) = choice else {
                                continue;
                            };
                            if let Some(reason) =
                                choice.get("finish_reason").and_then(|v| v.as_str())
                            {
                                finish_reason = Some(reason.to_string());
                            }
                            if let Some(delta) = choice.get("delta") {
                                if let Some(content) = delta.get("content").and_then(|v| v.as_str())
                                {
                                    content_out.push_str(content);
                                    cb(StreamChunk::ContentDelta(content.to_string()));
                                }
                                if let Some(reasoning) =
                                    delta.get("reasoning_content").and_then(|v| v.as_str())
                                {
                                    reasoning_out.push_str(reasoning);
                                    cb(StreamChunk::ReasoningDelta(reasoning.to_string()));
                                }
                                if let Some(tool_calls) =
                                    delta.get("tool_calls").and_then(|v| v.as_array())
                                {
                                    merge_stream_tool_calls(tool_calls, &mut tool_call_parts);
                                }
                            }
                            if let Some(message) = choice.get("message") {
                                if let Some(content) =
                                    message.get("content").and_then(|v| v.as_str())
                                {
                                    content_out.push_str(content);
                                    cb(StreamChunk::ContentDelta(content.to_string()));
                                }
                                if let Some(reasoning) =
                                    message.get("reasoning_content").and_then(|v| v.as_str())
                                {
                                    reasoning_out.push_str(reasoning);
                                    cb(StreamChunk::ReasoningDelta(reasoning.to_string()));
                                }
                                if let Some(tool_calls) = message.get("tool_calls") {
                                    completed_tool_calls.extend(parse_tool_calls_array(tool_calls));
                                }
                            }
                        }

                        // If stream read failed, propagate the error
                        if let Some(err) = last_err.take() {
                            return Err(err);
                        }

                        let mut tool_calls: Vec<LlmToolCall> = tool_call_parts
                            .into_iter()
                            .filter_map(|(index, value)| {
                                if value.name.trim().is_empty() {
                                    return None;
                                }
                                Some(LlmToolCall {
                                    id: value
                                        .id
                                        .unwrap_or_else(|| format!("tool_call_{}", index + 1)),
                                    name: value.name,
                                    arguments: value.arguments,
                                })
                            })
                            .collect();
                        if !completed_tool_calls.is_empty() {
                            tool_calls.extend(completed_tool_calls);
                        }

                        let text = if !content_out.is_empty() {
                            content_out
                        } else {
                            reasoning_out.clone()
                        };
                        return Ok(LlmResponse {
                            text,
                            finish_reason: finish_reason.unwrap_or_else(|| "stop".to_string()),
                            reasoning_content: reasoning_out,
                            tool_calls,
                        });
                    }

                    let body = resp.text().unwrap_or_default();
                    if status == StatusCode::BAD_REQUEST
                        && server_cache_enabled
                        && payload_rejects_cache_control(&body)
                    {
                        server_cache_enabled = false;
                        payload = self.build_payload_with_server_cache(req, server_cache_enabled);
                        payload["stream"] = json!(true);
                        continue;
                    }
                    last_err = Some(format_api_error(
                        status,
                        &body,
                        attempt,
                        self.cfg.max_retries,
                    ));
                    if should_retry_status(status) && attempt < self.cfg.max_retries {
                        thread::sleep(retry_delay_ms(self.cfg.retry_base_ms, attempt, retry_after));
                        attempt = attempt.saturating_add(1);
                        continue;
                    }
                    break;
                }
                Err(e) => {
                    last_err = Some(format_transport_error(&e));
                    if should_retry_transport_error(&e) && attempt < self.cfg.max_retries {
                        thread::sleep(retry_delay_ms(NETWORK_RETRY_BASE_MS, attempt, None));
                        attempt = attempt.saturating_add(1);
                        continue;
                    }
                    break;
                }
            }
        }

        Err(last_err
            .unwrap_or_else(|| anyhow!("deepseek streaming request failed without detailed error")))
    }
}

fn annotate_cache_control(message: &mut Value) {
    if let Some(obj) = message.as_object_mut() {
        obj.insert("cache_control".to_string(), json!({ "type": "ephemeral" }));
    }
}

/// Apply cache annotations to a messages array based on the cache strategy.
/// Annotates the stable prefix (system prompt + early messages) for cache hits.
fn apply_cache_annotations(messages: &mut [Value], strategy: &CacheStrategy) {
    match strategy {
        CacheStrategy::Off => {}
        CacheStrategy::Auto => {
            // Annotate system prompt (index 0) + first user message (index 1)
            if let Some(msg) = messages.first_mut() {
                annotate_cache_control(msg);
            }
            if messages.len() > 1 {
                annotate_cache_control(&mut messages[1]);
            }
        }
        CacheStrategy::Aggressive => {
            // Annotate the first 3 messages (system + early conversation prefix)
            let count = messages.len().min(3);
            for msg in messages[..count].iter_mut() {
                annotate_cache_control(msg);
            }
        }
    }
}

fn payload_rejects_cache_control(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("cache_control")
        || (lower.contains("unknown")
            && (lower.contains("field") || lower.contains("property"))
            && lower.contains("cache"))
        || (lower.contains("extra")
            && lower.contains("field")
            && lower.contains("cache")
            && lower.contains("permitted"))
}

impl LlmClient for DeepSeekClient {
    fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let provider = self.cfg.provider.to_ascii_lowercase();
        if provider != "deepseek" {
            return Err(anyhow!(
                "unsupported llm.provider='{}' (only 'deepseek' is supported)",
                self.cfg.provider
            ));
        }
        let key = self
            .resolve_api_key()
            .ok_or_else(|| anyhow!("{} not set and llm.api_key is empty", self.cfg.api_key_env))?;

        let profile = normalize_deepseek_profile(&self.cfg.profile).ok_or_else(|| {
            anyhow!(
                "unsupported llm.profile='{}' (supported: v3_2, v3_2_speciale)",
                self.cfg.profile
            )
        })?;
        let mut normalized_req = req.clone();
        normalized_req.model = self.resolve_request_model(&req.model, profile)?;
        self.complete_inner(&normalized_req, &key)
    }

    fn complete_streaming(&self, req: &LlmRequest, cb: StreamCallback) -> Result<LlmResponse> {
        let provider = self.cfg.provider.to_ascii_lowercase();
        if provider != "deepseek" {
            return Err(anyhow!(
                "unsupported llm.provider='{}' (only 'deepseek' is supported)",
                self.cfg.provider
            ));
        }
        let key = self
            .resolve_api_key()
            .ok_or_else(|| anyhow!("{} not set and llm.api_key is empty", self.cfg.api_key_env))?;

        let profile = normalize_deepseek_profile(&self.cfg.profile).ok_or_else(|| {
            anyhow!(
                "unsupported llm.profile='{}' (supported: v3_2, v3_2_speciale)",
                self.cfg.profile
            )
        })?;
        let mut normalized_req = req.clone();
        normalized_req.model = self.resolve_request_model(&req.model, profile)?;
        self.complete_streaming_inner(&normalized_req, &key, cb)
    }

    fn complete_chat(&self, req: &ChatRequest) -> Result<LlmResponse> {
        let provider = self.cfg.provider.to_ascii_lowercase();
        if provider != "deepseek" {
            return Err(anyhow!(
                "unsupported llm.provider='{}' (only 'deepseek' is supported)",
                self.cfg.provider
            ));
        }
        let key = self
            .resolve_api_key()
            .ok_or_else(|| anyhow!("{} not set and llm.api_key is empty", self.cfg.api_key_env))?;
        self.complete_chat_inner(req, &key)
    }

    fn complete_chat_streaming(
        &self,
        req: &ChatRequest,
        cb: StreamCallback,
    ) -> Result<LlmResponse> {
        let provider = self.cfg.provider.to_ascii_lowercase();
        if provider != "deepseek" {
            return Err(anyhow!(
                "unsupported llm.provider='{}' (only 'deepseek' is supported)",
                self.cfg.provider
            ));
        }
        let key = self
            .resolve_api_key()
            .ok_or_else(|| anyhow!("{} not set and llm.api_key is empty", self.cfg.api_key_env))?;
        self.complete_chat_streaming_inner(req, &key, cb)
    }
}

/// Produce a user-friendly error from a DeepSeek API HTTP response.
fn format_api_error(status: StatusCode, body: &str, attempt: u8, max_retries: u8) -> anyhow::Error {
    // Try to extract the error message from JSON body
    let detail = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message").or(Some(e)))
                .and_then(|m| m.as_str().map(ToString::to_string))
        })
        .unwrap_or_else(|| body.chars().take(200).collect());

    match status {
        StatusCode::UNAUTHORIZED => anyhow!(
            "Invalid or missing API key (HTTP 401).\n\
             Set DEEPSEEK_API_KEY environment variable or configure llm.api_key in settings.\n\
             Get an API key at https://platform.deepseek.com"
        ),
        StatusCode::TOO_MANY_REQUESTS => anyhow!(
            "Rate limited (HTTP 429). Exhausted {}/{} retries. Try again shortly or reduce request frequency. Detail: {}",
            attempt + 1,
            max_retries + 1,
            detail
        ),
        StatusCode::PAYMENT_REQUIRED => anyhow!(
            "Insufficient balance (HTTP 402). Top up your DeepSeek account at https://platform.deepseek.com"
        ),
        StatusCode::INTERNAL_SERVER_ERROR | StatusCode::SERVICE_UNAVAILABLE => anyhow!(
            "DeepSeek server error (HTTP {}). Exhausted {}/{} retries. The service may be temporarily unavailable. Detail: {}",
            status.as_u16(),
            attempt + 1,
            max_retries + 1,
            detail
        ),
        _ => anyhow!("DeepSeek API error (HTTP {}): {}", status.as_u16(), detail),
    }
}

/// Produce a user-friendly error from a transport/network failure.
fn format_transport_error(err: &reqwest::Error) -> anyhow::Error {
    let inner_msg = err
        .source()
        .map(|e| e.to_string())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let is_dns = inner_msg.contains("dns")
        || inner_msg.contains("resolve")
        || inner_msg.contains("name or service not known")
        || inner_msg.contains("no such host")
        || inner_msg.contains("getaddrinfo");

    if err.is_timeout() {
        anyhow!(
            "Request timed out. The DeepSeek API did not respond in time.\n\
             Retrying with exponential backoff. If this persists, try increasing \
             llm.timeout_seconds in your config."
        )
    } else if is_dns {
        anyhow!(
            "DNS resolution failed. Could not resolve the DeepSeek API hostname.\n\
             Check your internet connection and DNS settings. \
             Retrying with exponential backoff."
        )
    } else if err.is_connect() {
        anyhow!(
            "Connection refused. Could not reach the DeepSeek API at the configured endpoint.\n\
             Check your network connection and firewall settings. \
             Retrying with exponential backoff."
        )
    } else {
        anyhow!("Network error: {err}. Retrying with exponential backoff if retries remain.")
    }
}

fn should_retry_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::TOO_MANY_REQUESTS
            | StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::SERVICE_UNAVAILABLE
    )
}

fn should_retry_transport_error(err: &reqwest::Error) -> bool {
    err.is_timeout() || err.is_connect() || err.is_request()
}

fn is_speciale_alias(model: &str) -> bool {
    matches!(
        model.trim().to_ascii_lowercase().as_str(),
        "deepseek-v3.2-speciale" | "deepseek-v3.2-special" | "v3.2-speciale" | "v3_2_speciale"
    )
}

fn parse_retry_after_seconds(header: Option<&reqwest::header::HeaderValue>) -> Option<u64> {
    let value = header?.to_str().ok()?.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(seconds);
    }
    parse_retry_after_http_date(value)
}

fn parse_retry_after_http_date(value: &str) -> Option<u64> {
    let retry_at = DateTime::parse_from_rfc2822(value)
        .map(|dt| dt.with_timezone(&Utc))
        .or_else(|_| {
            NaiveDateTime::parse_from_str(value, "%a, %d %b %Y %H:%M:%S GMT")
                .map(|naive| DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc))
        })
        .ok()?;
    let now = Utc::now();
    let delta = retry_at.signed_duration_since(now).num_seconds();
    Some(delta.max(0) as u64)
}

fn retry_delay_ms(base_ms: u64, attempt: u8, retry_after_seconds: Option<u64>) -> Duration {
    if let Some(seconds) = retry_after_seconds {
        return Duration::from_millis(seconds.saturating_mul(1000));
    }
    let exponent = u32::from(attempt);
    let exponential = base_ms.saturating_mul(2_u64.saturating_pow(exponent));
    Duration::from_millis(exponential.max(base_ms.max(100)))
}

fn parse_non_streaming_payload(body: &str) -> Result<LlmResponse> {
    let value: Value = serde_json::from_str(body)?;
    let choice = value
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first());
    let Some(choice) = choice else {
        return Err(anyhow!(
            "unexpected non-streaming payload: missing choices[0]"
        ));
    };
    let finish_reason = choice
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("stop")
        .to_string();
    let message = choice.get("message").cloned().unwrap_or_else(|| json!({}));
    let content = message
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let reasoning_content = message
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let tool_calls = message
        .get("tool_calls")
        .map(parse_tool_calls_array)
        .unwrap_or_default();
    if content.is_empty() && reasoning_content.is_empty() && tool_calls.is_empty() {
        return Err(anyhow!(
            "unexpected non-streaming payload: missing message.content/reasoning_content/tool_calls"
        ));
    }
    let text = if content.is_empty() {
        reasoning_content.clone()
    } else {
        content
    };
    Ok(LlmResponse {
        text,
        finish_reason,
        reasoning_content,
        tool_calls,
    })
}

fn parse_streaming_payload(body: &str) -> Result<LlmResponse> {
    let mut content_out = String::new();
    let mut reasoning_out = String::new();
    let mut finish_reason: Option<String> = None;
    let mut tool_call_parts: BTreeMap<u64, StreamToolCall> = BTreeMap::new();
    let mut completed_tool_calls = Vec::new();
    let mut parsed_any = false;
    for line in body.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("data:") {
            continue;
        }
        let chunk = trimmed.trim_start_matches("data:").trim();
        if chunk == "[DONE]" {
            break;
        }
        let value: Value = serde_json::from_str(chunk)?;
        let choice = value
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first());
        let Some(choice) = choice else {
            continue;
        };
        if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
            finish_reason = Some(reason.to_string());
            parsed_any = true;
        }
        if let Some(delta) = choice.get("delta") {
            if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                content_out.push_str(content);
                parsed_any = true;
            }
            if let Some(reasoning) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                reasoning_out.push_str(reasoning);
                parsed_any = true;
            }
            if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                merge_stream_tool_calls(tool_calls, &mut tool_call_parts);
                parsed_any = true;
            }
        }
        if let Some(message) = choice.get("message") {
            if let Some(content) = message.get("content").and_then(|v| v.as_str()) {
                content_out.push_str(content);
                parsed_any = true;
            }
            if let Some(reasoning) = message.get("reasoning_content").and_then(|v| v.as_str()) {
                reasoning_out.push_str(reasoning);
                parsed_any = true;
            }
            if let Some(tool_calls) = message.get("tool_calls") {
                completed_tool_calls.extend(parse_tool_calls_array(tool_calls));
                parsed_any = true;
            }
        }
    }

    let mut tool_calls = tool_call_parts
        .into_iter()
        .filter_map(|(index, value)| {
            if value.name.trim().is_empty() {
                return None;
            }
            Some(LlmToolCall {
                id: value
                    .id
                    .unwrap_or_else(|| format!("tool_call_{}", index + 1)),
                name: value.name,
                arguments: value.arguments,
            })
        })
        .collect::<Vec<_>>();
    if !completed_tool_calls.is_empty() {
        tool_calls.extend(completed_tool_calls);
    }

    if parsed_any {
        let text = if !content_out.is_empty() {
            content_out
        } else {
            reasoning_out.clone()
        };
        Ok(LlmResponse {
            text,
            finish_reason: finish_reason.unwrap_or_else(|| "stop".to_string()),
            reasoning_content: reasoning_out,
            tool_calls,
        })
    } else {
        parse_non_streaming_payload(body)
    }
}

#[derive(Default)]
struct StreamToolCall {
    id: Option<String>,
    name: String,
    arguments: String,
}

fn merge_stream_tool_calls(chunks: &[Value], out: &mut BTreeMap<u64, StreamToolCall>) {
    for (idx, item) in chunks.iter().enumerate() {
        let index = item
            .get("index")
            .and_then(|v| v.as_u64())
            .unwrap_or(idx as u64);
        let entry = out.entry(index).or_default();
        if let Some(id) = item.get("id").and_then(|v| v.as_str())
            && !id.trim().is_empty()
        {
            entry.id = Some(id.to_string());
        }
        if let Some(function) = item.get("function") {
            if let Some(name) = function.get("name").and_then(|v| v.as_str())
                && !name.trim().is_empty()
            {
                entry.name = name.to_string();
            }
            if let Some(arguments) = function.get("arguments").and_then(|v| v.as_str()) {
                entry.arguments.push_str(arguments);
            }
        }
    }
}

fn parse_tool_calls_array(value: &Value) -> Vec<LlmToolCall> {
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    items
        .iter()
        .enumerate()
        .filter_map(|(idx, item)| {
            let name = item
                .get("function")
                .and_then(|v| v.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            if name.trim().is_empty() {
                return None;
            }
            let arguments = item
                .get("function")
                .and_then(|v| v.get("arguments"))
                .and_then(|v| v.as_str())
                .map(ToString::to_string)
                .unwrap_or_else(|| {
                    item.get("function")
                        .and_then(|v| v.get("arguments"))
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "{}".to_string())
                });
            let id = item
                .get("id")
                .and_then(|v| v.as_str())
                .filter(|id| !id.trim().is_empty())
                .map(ToString::to_string)
                .unwrap_or_else(|| format!("tool_call_{}", idx + 1));
            Some(LlmToolCall {
                id,
                name,
                arguments,
            })
        })
        .collect()
}

/// Marker substrings for DSML-style raw markup in streamed content. We suppress
/// these fragments from text streaming output instead of attempting tool rescue.
const DSML_MARKERS: &[&str] = &[
    "\u{ff5c}DSML\u{ff5c}",                           // ｜DSML｜
    "\u{ff5c}tool\u{2581}calls\u{2581}begin\u{ff5c}", // ｜tool▁calls▁begin｜
];

/// Returns `true` when a stream chunk appears to contain raw DSML markup.
fn contains_dsml_markup(content: &str) -> bool {
    DSML_MARKERS.iter().any(|marker| content.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, mpsc};
    use std::thread;
    use std::time::Duration as StdDuration;

    #[test]
    fn parses_non_streaming() {
        let body = r#"{"choices":[{"message":{"content":"hello"}}]}"#;
        let got = parse_non_streaming_payload(body).expect("parse");
        assert_eq!(got.text, "hello");
        assert_eq!(got.finish_reason, "stop");
    }

    #[test]
    fn parses_streaming_sse_lines() {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\ndata: [DONE]";
        let got = parse_streaming_payload(body).expect("stream parse");
        assert_eq!(got.text, "hello");
    }

    #[test]
    fn parses_streaming_reasoning_content() {
        let body = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"step1\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"reasoning_content\":\"step2\"}}]}\n\ndata: [DONE]";
        let got = parse_streaming_payload(body).expect("stream parse");
        assert_eq!(got.text, "step1step2");
        assert_eq!(got.reasoning_content, "step1step2");
    }

    #[test]
    fn parses_non_streaming_tool_calls() {
        let body = r#"{
          "choices": [
            {
              "finish_reason": "tool_calls",
              "message": {
                "content": "",
                "tool_calls": [
                  {
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "fs.read", "arguments": "{\"path\":\"README.md\"}" }
                  }
                ]
              }
            }
          ]
        }"#;
        let got = parse_non_streaming_payload(body).expect("parse");
        assert_eq!(got.finish_reason, "tool_calls");
        assert_eq!(got.tool_calls.len(), 1);
        assert_eq!(got.tool_calls[0].name, "fs.read");
    }

    #[test]
    fn parses_streaming_tool_call_fragments() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"fs.read\",\"arguments\":\"{\\\"path\\\":\\\"REA\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"DME.md\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n"
        );
        let got = parse_streaming_payload(body).expect("stream parse");
        assert_eq!(got.finish_reason, "tool_calls");
        assert_eq!(got.tool_calls.len(), 1);
        assert_eq!(got.tool_calls[0].name, "fs.read");
        assert_eq!(got.tool_calls[0].arguments, "{\"path\":\"README.md\"}");
    }

    #[test]
    fn fast_mode_caps_max_tokens_in_payload() {
        let cfg = LlmConfig {
            fast_mode: true,
            ..LlmConfig::default()
        };
        let client = DeepSeekClient::new(cfg).expect("client");
        let payload = client.build_payload(&LlmRequest {
            unit: deepseek_core::LlmUnit::Planner,
            prompt: "hello".to_string(),
            model: "deepseek-chat".to_string(),
            max_tokens: 16_000,
            non_urgent: false,
            images: vec![],
        });
        assert_eq!(payload["max_tokens"], 2048);
    }

    #[test]
    fn model_alias_is_normalized_in_payload() {
        let client = DeepSeekClient::new(LlmConfig::default()).expect("client");
        let payload = client.build_payload(&LlmRequest {
            unit: deepseek_core::LlmUnit::Planner,
            prompt: "hello".to_string(),
            model: "deepseek-v3.2".to_string(),
            max_tokens: 256,
            non_urgent: false,
            images: vec![],
        });
        assert_eq!(payload["model"], "deepseek-chat");
    }

    #[test]
    fn truly_unsupported_provider_is_rejected() {
        let cfg = LlmConfig {
            provider: "invalid_provider_xyz".to_string(),
            api_key: Some("test-key".to_string()),
            ..LlmConfig::default()
        };
        let client = DeepSeekClient::new(cfg).expect("client");
        let err = client
            .complete(&LlmRequest {
                unit: deepseek_core::LlmUnit::Planner,
                prompt: "hello".to_string(),
                model: "any".to_string(),
                max_tokens: 128,
                non_urgent: false,
                images: vec![],
            })
            .expect_err("truly unsupported provider should fail");
        assert!(err.to_string().contains("unsupported llm.provider"));
    }

    #[test]
    fn unsupported_profile_is_rejected() {
        let cfg = LlmConfig {
            profile: "unknown".to_string(),
            api_key: Some("test-key".to_string()),
            ..LlmConfig::default()
        };
        let client = DeepSeekClient::new(cfg).expect("client");
        let err = client
            .complete(&LlmRequest {
                unit: deepseek_core::LlmUnit::Planner,
                prompt: "hello".to_string(),
                model: "deepseek-chat".to_string(),
                max_tokens: 128,
                non_urgent: false,
                images: vec![],
            })
            .expect_err("unsupported profile should fail");
        assert!(err.to_string().contains("unsupported llm.profile"));
    }

    #[test]
    fn unsupported_model_is_rejected_before_network_call() {
        let cfg = LlmConfig {
            api_key: Some("test-key".to_string()),
            ..LlmConfig::default()
        };
        let client = DeepSeekClient::new(cfg).expect("client");
        let err = client
            .complete(&LlmRequest {
                unit: deepseek_core::LlmUnit::Planner,
                prompt: "hello".to_string(),
                model: "not-a-deepseek-model".to_string(),
                max_tokens: 128,
                non_urgent: false,
                images: vec![],
            })
            .expect_err("unsupported model should fail");
        assert!(err.to_string().contains("unsupported model"));
    }

    #[test]
    fn speciale_model_requires_speciale_profile() {
        let cfg = LlmConfig {
            api_key: Some("test-key".to_string()),
            ..LlmConfig::default()
        };
        let client = DeepSeekClient::new(cfg).expect("client");
        let err = client
            .complete(&LlmRequest {
                unit: deepseek_core::LlmUnit::Planner,
                prompt: "hello".to_string(),
                model: "deepseek-v3.2-speciale".to_string(),
                max_tokens: 128,
                non_urgent: false,
                images: vec![],
            })
            .expect_err("speciale model should require speciale profile");
        assert!(
            err.to_string()
                .contains("requires llm.profile='v3_2_speciale'")
        );
    }

    #[test]
    fn non_deepseek_provider_is_rejected() {
        for provider in &["openai", "anthropic", "custom", "local", "ollama"] {
            let cfg = LlmConfig {
                provider: provider.to_string(),
                api_key: Some("test-key".to_string()),
                ..LlmConfig::default()
            };
            let client = DeepSeekClient::new(cfg).expect("client");
            let err = client
                .complete(&LlmRequest {
                    unit: deepseek_core::LlmUnit::Planner,
                    prompt: "hello".to_string(),
                    model: "deepseek-chat".to_string(),
                    max_tokens: 128,
                    non_urgent: false,
                    images: vec![],
                })
                .expect_err("non-deepseek provider should be rejected");
            assert!(
                err.to_string().contains("only 'deepseek' is supported"),
                "provider '{provider}' should be rejected but got: {err}"
            );
        }
    }

    #[test]
    fn missing_api_key_is_rejected() {
        let cfg = LlmConfig {
            api_key: None,
            api_key_env: "DEEPSEEK_NONEXISTENT_KEY_FOR_TEST".to_string(),
            ..LlmConfig::default()
        };
        let client = DeepSeekClient::new(cfg).expect("client");
        let err = client
            .complete(&LlmRequest {
                unit: deepseek_core::LlmUnit::Planner,
                prompt: "hello".to_string(),
                model: "deepseek-chat".to_string(),
                max_tokens: 128,
                non_urgent: false,
                images: vec![],
            })
            .expect_err("missing API key should fail");
        assert!(err.to_string().contains("not set and llm.api_key is empty"));
    }

    #[test]
    fn adds_language_system_instruction_when_not_english() {
        let cfg = LlmConfig {
            language: "es".to_string(),
            ..LlmConfig::default()
        };
        let client = DeepSeekClient::new(cfg).expect("client");
        let payload = client.build_payload(&LlmRequest {
            unit: deepseek_core::LlmUnit::Planner,
            prompt: "hola".to_string(),
            model: "deepseek-chat".to_string(),
            max_tokens: 128,
            non_urgent: false,
            images: vec![],
        });
        let messages = payload["messages"].as_array().expect("messages");
        assert_eq!(messages[0]["role"], "system");
        assert!(
            messages[0]["content"]
                .as_str()
                .unwrap_or_default()
                .contains("Respond in es")
        );
        assert!(messages[0].get("cache_control").is_some());
    }

    #[test]
    fn prompt_cache_annotations_can_be_disabled_for_server_payload() {
        let cfg = LlmConfig {
            prompt_cache_enabled: false,
            ..LlmConfig::default()
        };
        let client = DeepSeekClient::new(cfg).expect("client");
        let payload = client.build_payload(&LlmRequest {
            unit: deepseek_core::LlmUnit::Planner,
            prompt: "hello".to_string(),
            model: "deepseek-chat".to_string(),
            max_tokens: 128,
            non_urgent: false,
            images: vec![],
        });
        let messages = payload["messages"].as_array().expect("messages");
        assert!(
            messages
                .iter()
                .all(|msg| msg.get("cache_control").is_none())
        );
    }

    #[test]
    fn bad_request_cache_control_fallback_disables_server_cache_annotations() {
        let server = start_mock_retry_server(vec![
            MockHttpResponse {
                status: 400,
                body: r#"{"error":"unknown field cache_control"}"#.to_string(),
                retry_after: None,
            },
            MockHttpResponse {
                status: 200,
                body: r#"{"choices":[{"message":{"content":"ok-after-cache-fallback"}}]}"#
                    .to_string(),
                retry_after: None,
            },
        ]);

        let cfg = LlmConfig {
            endpoint: server.endpoint.clone(),
            stream: false,
            api_key_env: "DEEPSEEK_API_KEY_CACHE_FALLBACK_TEST".to_string(),
            max_retries: 0,
            ..LlmConfig::default()
        };
        let client = DeepSeekClient::new(cfg).expect("client");
        // SAFETY: test-only process-level env mutation.
        unsafe {
            std::env::set_var("DEEPSEEK_API_KEY_CACHE_FALLBACK_TEST", "test-key");
        }

        let out = client
            .complete(&LlmRequest {
                unit: deepseek_core::LlmUnit::Planner,
                prompt: "fallback".to_string(),
                model: "deepseek-chat".to_string(),
                max_tokens: 64,
                non_urgent: false,
                images: vec![],
            })
            .expect("response should succeed after fallback");
        assert_eq!(out.text, "ok-after-cache-fallback");
        assert_eq!(server.request_count(), 2);

        // SAFETY: test-only process-level env mutation.
        unsafe {
            std::env::remove_var("DEEPSEEK_API_KEY_CACHE_FALLBACK_TEST");
        }
    }

    #[test]
    fn chat_payload_includes_cache_annotations_by_default() {
        let cfg = LlmConfig::default();
        let client = DeepSeekClient::new(cfg).expect("client");
        let req = ChatRequest {
            model: "deepseek-chat".to_string(),
            messages: vec![
                ChatMessage::System {
                    content: "system prompt".to_string(),
                },
                ChatMessage::User {
                    content: "hello".to_string(),
                },
            ],
            tools: vec![],
            tool_choice: deepseek_core::ToolChoice::none(),
            max_tokens: 64,
            temperature: Some(0.2),
            thinking: None,
        };
        let payload = client.build_chat_payload(&req);
        let messages = payload["messages"].as_array().expect("messages");
        assert_eq!(messages.len(), 2);
        // Auto strategy: system prompt + first user message annotated
        assert!(messages[0].get("cache_control").is_some());
        assert!(messages[1].get("cache_control").is_some());
    }

    #[test]
    fn chat_payload_off_strategy_has_no_cache_annotations() {
        let cfg = LlmConfig {
            cache_strategy: CacheStrategy::Off,
            ..LlmConfig::default()
        };
        let client = DeepSeekClient::new(cfg).expect("client");
        let req = ChatRequest {
            model: "deepseek-chat".to_string(),
            messages: vec![
                ChatMessage::System {
                    content: "system prompt".to_string(),
                },
                ChatMessage::User {
                    content: "hello".to_string(),
                },
            ],
            tools: vec![],
            tool_choice: deepseek_core::ToolChoice::none(),
            max_tokens: 64,
            temperature: Some(0.2),
            thinking: None,
        };
        let payload = client.build_chat_payload(&req);
        let messages = payload["messages"].as_array().expect("messages");
        assert!(messages.iter().all(|m| m.get("cache_control").is_none()));
    }

    #[test]
    fn chat_payload_aggressive_strategy_annotates_first_three() {
        let cfg = LlmConfig {
            cache_strategy: CacheStrategy::Aggressive,
            ..LlmConfig::default()
        };
        let client = DeepSeekClient::new(cfg).expect("client");
        let req = ChatRequest {
            model: "deepseek-chat".to_string(),
            messages: vec![
                ChatMessage::System {
                    content: "system prompt".to_string(),
                },
                ChatMessage::User {
                    content: "hello".to_string(),
                },
                ChatMessage::Assistant {
                    content: Some("hi there".to_string()),
                    reasoning_content: None,
                    tool_calls: vec![],
                },
                ChatMessage::User {
                    content: "follow up".to_string(),
                },
            ],
            tools: vec![],
            tool_choice: deepseek_core::ToolChoice::none(),
            max_tokens: 64,
            temperature: Some(0.2),
            thinking: None,
        };
        let payload = client.build_chat_payload(&req);
        let messages = payload["messages"].as_array().expect("messages");
        assert_eq!(messages.len(), 4);
        // Aggressive: first 3 messages annotated, 4th is not
        assert!(messages[0].get("cache_control").is_some());
        assert!(messages[1].get("cache_control").is_some());
        assert!(messages[2].get("cache_control").is_some());
        assert!(messages[3].get("cache_control").is_none());
    }

    #[test]
    fn resolve_api_key_uses_config_fallback() {
        let cfg = LlmConfig {
            api_key_env: "DEEPSEEK_API_KEY_TEST_FALLBACK".to_string(),
            api_key: Some("local-key".to_string()),
            ..LlmConfig::default()
        };
        let client = DeepSeekClient::new(cfg).expect("client");
        // SAFETY: test-only process-level env mutation.
        unsafe {
            std::env::remove_var("DEEPSEEK_API_KEY_TEST_FALLBACK");
        }
        let resolved = client.resolve_api_key().expect("fallback key");
        assert_eq!(resolved, "local-key");
    }

    #[test]
    fn retry_status_classification_matches_deepseek_guidance() {
        assert!(should_retry_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(should_retry_status(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(should_retry_status(StatusCode::SERVICE_UNAVAILABLE));
        assert!(!should_retry_status(StatusCode::UNAUTHORIZED));
        assert!(!should_retry_status(StatusCode::BAD_REQUEST));
    }

    #[test]
    fn format_api_error_401_includes_setup_instructions() {
        let err = format_api_error(
            StatusCode::UNAUTHORIZED,
            r#"{"error":"invalid_api_key"}"#,
            0,
            3,
        );
        let msg = err.to_string();
        assert!(
            msg.contains("Invalid or missing API key"),
            "should mention invalid/missing key: {msg}"
        );
        assert!(
            msg.contains("DEEPSEEK_API_KEY"),
            "should mention env var: {msg}"
        );
        assert!(
            msg.contains("llm.api_key"),
            "should mention config field: {msg}"
        );
        assert!(
            msg.contains("https://platform.deepseek.com"),
            "should include signup URL: {msg}"
        );
    }

    #[test]
    fn format_api_error_401_does_not_retry() {
        // 401 should not be retryable
        assert!(!should_retry_status(StatusCode::UNAUTHORIZED));
    }

    #[test]
    fn network_retry_base_uses_one_second_delays() {
        // Verify the constant matches the spec: 1s, 2s, 4s
        assert_eq!(NETWORK_RETRY_BASE_MS, 1000);
        let d0 = retry_delay_ms(NETWORK_RETRY_BASE_MS, 0, None);
        let d1 = retry_delay_ms(NETWORK_RETRY_BASE_MS, 1, None);
        let d2 = retry_delay_ms(NETWORK_RETRY_BASE_MS, 2, None);
        assert_eq!(d0, Duration::from_millis(1000));
        assert_eq!(d1, Duration::from_millis(2000));
        assert_eq!(d2, Duration::from_millis(4000));
    }

    #[test]
    fn retry_after_parses_seconds_and_http_date() {
        let seconds_header = reqwest::header::HeaderValue::from_static("7");
        assert_eq!(parse_retry_after_seconds(Some(&seconds_header)), Some(7));

        let future = Utc::now() + chrono::Duration::seconds(5);
        let http_date = future.format("%a, %d %b %Y %H:%M:%S GMT").to_string();
        let date_header = reqwest::header::HeaderValue::from_str(&http_date).expect("header");
        let parsed = parse_retry_after_seconds(Some(&date_header)).expect("parsed");
        assert!(parsed <= 10);
    }

    #[test]
    fn complete_retries_transient_status_then_succeeds() {
        let server = start_mock_retry_server(vec![
            MockHttpResponse {
                status: 503,
                body: r#"{"error":"temporarily_unavailable"}"#.to_string(),
                retry_after: Some("0".to_string()),
            },
            MockHttpResponse {
                status: 200,
                body: r#"{"choices":[{"message":{"content":"ok-after-retry"}}]}"#.to_string(),
                retry_after: None,
            },
        ]);

        let cfg = LlmConfig {
            endpoint: server.endpoint.clone(),
            stream: false,
            api_key_env: "DEEPSEEK_API_KEY_RETRY_TEST".to_string(),
            max_retries: 3,
            retry_base_ms: 1,
            ..LlmConfig::default()
        };
        let client = DeepSeekClient::new(cfg).expect("client");
        // SAFETY: test-only process-level env mutation.
        unsafe {
            std::env::set_var("DEEPSEEK_API_KEY_RETRY_TEST", "test-key");
        }

        let out = client
            .complete(&LlmRequest {
                unit: deepseek_core::LlmUnit::Planner,
                prompt: "retry test".to_string(),
                model: "deepseek-chat".to_string(),
                max_tokens: 64,
                non_urgent: false,
                images: vec![],
            })
            .expect("response should eventually succeed");
        assert_eq!(out.text, "ok-after-retry");
        assert!(server.request_count() >= 2);

        // SAFETY: test-only process-level env mutation.
        unsafe {
            std::env::remove_var("DEEPSEEK_API_KEY_RETRY_TEST");
        }
    }

    #[test]
    fn complete_stops_after_bounded_retries() {
        let server = start_mock_retry_server(vec![MockHttpResponse {
            status: 429,
            body: r#"{"error":"rate_limited"}"#.to_string(),
            retry_after: Some("0".to_string()),
        }]);

        let cfg = LlmConfig {
            endpoint: server.endpoint.clone(),
            stream: false,
            api_key_env: "DEEPSEEK_API_KEY_RETRY_LIMIT_TEST".to_string(),
            max_retries: 2,
            retry_base_ms: 1,
            ..LlmConfig::default()
        };
        let client = DeepSeekClient::new(cfg).expect("client");
        // SAFETY: test-only process-level env mutation.
        unsafe {
            std::env::set_var("DEEPSEEK_API_KEY_RETRY_LIMIT_TEST", "test-key");
        }

        let err = client
            .complete(&LlmRequest {
                unit: deepseek_core::LlmUnit::Planner,
                prompt: "retry limit test".to_string(),
                model: "deepseek-chat".to_string(),
                max_tokens: 64,
                non_urgent: false,
                images: vec![],
            })
            .expect_err("request should fail after retries are exhausted");
        assert!(err.to_string().contains("Rate limited (HTTP 429)"));
        assert_eq!(server.request_count(), 3);

        // SAFETY: test-only process-level env mutation.
        unsafe {
            std::env::remove_var("DEEPSEEK_API_KEY_RETRY_LIMIT_TEST");
        }
    }

    #[test]
    fn complete_returns_clear_instructions_on_401() {
        let server = start_mock_retry_server(vec![MockHttpResponse {
            status: 401,
            body: r#"{"error":{"message":"invalid_api_key"}}"#.to_string(),
            retry_after: None,
        }]);

        let cfg = LlmConfig {
            endpoint: server.endpoint.clone(),
            stream: false,
            api_key_env: "DEEPSEEK_API_KEY_401_TEST".to_string(),
            max_retries: 2,
            retry_base_ms: 1,
            ..LlmConfig::default()
        };
        let client = DeepSeekClient::new(cfg).expect("client");
        // SAFETY: test-only process-level env mutation.
        unsafe {
            std::env::set_var("DEEPSEEK_API_KEY_401_TEST", "bad-key");
        }

        let err = client
            .complete(&LlmRequest {
                unit: deepseek_core::LlmUnit::Planner,
                prompt: "hello".to_string(),
                model: "deepseek-chat".to_string(),
                max_tokens: 64,
                non_urgent: false,
                images: vec![],
            })
            .expect_err("401 should fail without retrying");

        let msg = err.to_string();
        assert!(
            msg.contains("Invalid or missing API key"),
            "401 error should include clear message: {msg}"
        );
        assert!(
            msg.contains("DEEPSEEK_API_KEY"),
            "401 error should mention env var: {msg}"
        );
        assert!(
            msg.contains("https://platform.deepseek.com"),
            "401 error should include platform URL: {msg}"
        );
        // 401 is non-retryable, so only 1 request
        assert_eq!(server.request_count(), 1);

        // SAFETY: test-only process-level env mutation.
        unsafe {
            std::env::remove_var("DEEPSEEK_API_KEY_401_TEST");
        }
    }

    #[derive(Clone)]
    struct MockHttpResponse {
        status: u16,
        body: String,
        retry_after: Option<String>,
    }

    struct RetryMockServer {
        endpoint: String,
        request_count: Arc<AtomicUsize>,
        stop_tx: Option<mpsc::Sender<()>>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl RetryMockServer {
        fn request_count(&self) -> usize {
            self.request_count.load(Ordering::SeqCst)
        }
    }

    impl Drop for RetryMockServer {
        fn drop(&mut self) {
            if let Some(tx) = self.stop_tx.take() {
                let _ = tx.send(());
            }
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn start_mock_retry_server(responses: Vec<MockHttpResponse>) -> RetryMockServer {
        let scripted = if responses.is_empty() {
            vec![MockHttpResponse {
                status: 500,
                body: r#"{"error":"empty_script"}"#.to_string(),
                retry_after: None,
            }]
        } else {
            responses
        };
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind retry mock server");
        listener
            .set_nonblocking(true)
            .expect("set nonblocking listener");
        let addr = listener.local_addr().expect("addr");
        let request_count = Arc::new(AtomicUsize::new(0));
        let request_count_thread = Arc::clone(&request_count);
        let (tx, rx) = mpsc::channel::<()>();
        let handle = thread::spawn(move || {
            loop {
                if rx.try_recv().is_ok() {
                    break;
                }
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let _ = consume_http_request(&mut stream);
                        let idx = request_count_thread.fetch_add(1, Ordering::SeqCst);
                        let selected = scripted
                            .get(idx)
                            .cloned()
                            .or_else(|| scripted.last().cloned())
                            .expect("scripted response");
                        let status_text = match selected.status {
                            200 => "OK",
                            429 => "Too Many Requests",
                            500 => "Internal Server Error",
                            503 => "Service Unavailable",
                            _ => "Error",
                        };
                        let mut headers = format!(
                            "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
                            selected.status,
                            status_text,
                            selected.body.len()
                        );
                        if let Some(retry_after) = selected.retry_after {
                            headers.push_str(&format!("Retry-After: {retry_after}\r\n"));
                        }
                        headers.push_str("\r\n");
                        let response = format!("{headers}{}", selected.body);
                        let _ = stream.write_all(response.as_bytes());
                        let _ = stream.flush();
                    }
                    Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(StdDuration::from_millis(2));
                    }
                    Err(_) => break,
                }
            }
        });
        RetryMockServer {
            endpoint: format!("http://{addr}/chat/completions"),
            request_count,
            stop_tx: Some(tx),
            handle: Some(handle),
        }
    }

    fn consume_http_request(stream: &mut std::net::TcpStream) -> std::io::Result<()> {
        let mut buffer = Vec::new();
        let mut chunk = [0_u8; 1024];
        let mut header_end = None;
        while header_end.is_none() {
            let read = stream.read(&mut chunk)?;
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&chunk[..read]);
            header_end = find_subsequence(&buffer, b"\r\n\r\n").map(|idx| idx + 4);
            if buffer.len() > 1_048_576 {
                break;
            }
        }
        let header_len = header_end.unwrap_or(buffer.len());
        let content_length = parse_content_length(&buffer[..header_len]);
        let mut body = if header_len <= buffer.len() {
            buffer[header_len..].to_vec()
        } else {
            Vec::new()
        };
        while body.len() < content_length {
            let read = stream.read(&mut chunk)?;
            if read == 0 {
                break;
            }
            body.extend_from_slice(&chunk[..read]);
        }
        Ok(())
    }

    fn parse_content_length(headers: &[u8]) -> usize {
        let raw = String::from_utf8_lossy(headers);
        for line in raw.lines() {
            let mut parts = line.splitn(2, ':');
            let key = parts.next().unwrap_or_default().trim();
            if key.eq_ignore_ascii_case("content-length")
                && let Some(value) = parts.next()
                && let Ok(parsed) = value.trim().parse::<usize>()
            {
                return parsed;
            }
        }
        0
    }

    fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() || haystack.len() < needle.len() {
            return None;
        }
        haystack
            .windows(needle.len())
            .position(|window| window == needle)
    }

    #[test]
    fn complete_streaming_invokes_callback_per_chunk() {
        let sse_body = "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\ndata: [DONE]\n";
        let server = start_mock_retry_server(vec![MockHttpResponse {
            status: 200,
            body: sse_body.to_string(),
            retry_after: None,
        }]);

        let cfg = LlmConfig {
            endpoint: server.endpoint.clone(),
            stream: true,
            api_key_env: "DEEPSEEK_API_KEY_STREAM_TEST".to_string(),
            max_retries: 0,
            ..LlmConfig::default()
        };
        let client = DeepSeekClient::new(cfg).expect("client");
        // SAFETY: test-only process-level env mutation.
        unsafe {
            std::env::set_var("DEEPSEEK_API_KEY_STREAM_TEST", "test-key");
        }

        let chunks = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let chunks_clone = Arc::clone(&chunks);
        let cb: StreamCallback = Arc::new(move |chunk| match chunk {
            StreamChunk::ContentDelta(text) => {
                chunks_clone.lock().expect("test lock").push(text);
            }
            StreamChunk::Done => {
                chunks_clone
                    .lock()
                    .expect("test lock")
                    .push("[DONE]".to_string());
            }
            _ => {}
        });

        let resp = client
            .complete_streaming(
                &LlmRequest {
                    unit: deepseek_core::LlmUnit::Planner,
                    prompt: "hello".to_string(),
                    model: "deepseek-chat".to_string(),
                    max_tokens: 128,
                    non_urgent: false,
                    images: vec![],
                },
                cb,
            )
            .expect("streaming response");

        assert_eq!(resp.text, "hello");
        let collected = chunks.lock().expect("test lock");
        assert_eq!(collected.len(), 3); // "hel", "lo", "[DONE]"
        assert_eq!(collected[0], "hel");
        assert_eq!(collected[1], "lo");
        assert_eq!(collected[2], "[DONE]");

        // SAFETY: test-only process-level env mutation.
        unsafe {
            std::env::remove_var("DEEPSEEK_API_KEY_STREAM_TEST");
        }
    }
}
