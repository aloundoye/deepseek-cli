use anyhow::{Result, anyhow};
use deepseek_core::{LlmConfig, LlmRequest, LlmResponse};
use reqwest::StatusCode;
use reqwest::blocking::Client;
use serde_json::{Value, json};
use std::thread;
use std::time::Duration;

pub trait LlmClient {
    fn complete(&self, req: &LlmRequest) -> Result<LlmResponse>;
}

#[derive(Debug, Clone)]
pub struct OfflineDeepSeek;

impl LlmClient for OfflineDeepSeek {
    fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let text = if req.prompt.to_lowercase().contains("plan") {
            "Generated plan: discover files, propose edits, verify with tests.".to_string()
        } else {
            format!("Offline response via {}: {}", req.model, req.prompt)
        };
        Ok(LlmResponse {
            text,
            finish_reason: "stop".to_string(),
        })
    }
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
        let payload = json!({
            "model": req.model,
            "messages": [
                {"role": "user", "content": req.prompt}
            ],
            "temperature": self.cfg.temperature,
            "stream": self.cfg.stream,
            "max_tokens": req.max_tokens,
        });

        let mut last_err: Option<anyhow::Error> = None;

        for attempt in 0..=self.cfg.max_retries {
            let response = self
                .client
                .post(&self.cfg.endpoint)
                .bearer_auth(api_key)
                .json(&payload)
                .send();

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp.text()?;
                    if status.is_success() {
                        let text = if self.cfg.stream {
                            parse_streaming_payload(&body)?
                        } else {
                            parse_non_streaming_payload(&body)?
                        };
                        return Ok(LlmResponse {
                            text,
                            finish_reason: "stop".to_string(),
                        });
                    }

                    let retriable =
                        status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
                    last_err = Some(anyhow!("deepseek API error {}: {}", status, body));
                    if retriable && attempt < self.cfg.max_retries {
                        let backoff = self
                            .cfg
                            .retry_base_ms
                            .saturating_mul(2_u64.pow(attempt as u32));
                        thread::sleep(Duration::from_millis(backoff));
                        continue;
                    }
                    break;
                }
                Err(e) => {
                    last_err = Some(anyhow!("deepseek request failed: {e}"));
                    if attempt < self.cfg.max_retries {
                        let backoff = self
                            .cfg
                            .retry_base_ms
                            .saturating_mul(2_u64.pow(attempt as u32));
                        thread::sleep(Duration::from_millis(backoff));
                        continue;
                    }
                    break;
                }
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow!("deepseek request failed without detailed error")))
    }
}

impl LlmClient for DeepSeekClient {
    fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let key = std::env::var(&self.cfg.api_key_env).ok();
        if let Some(key) = key {
            return self.complete_inner(req, &key);
        }
        if self.cfg.offline_fallback {
            return OfflineDeepSeek.complete(req);
        }
        Err(anyhow!(
            "{} not set and offline_fallback=false",
            self.cfg.api_key_env
        ))
    }
}

fn parse_non_streaming_payload(body: &str) -> Result<String> {
    let value: Value = serde_json::from_str(body)?;
    if let Some(content) = value
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|v| v.get("message"))
        .and_then(|v| v.get("content"))
        .and_then(|v| v.as_str())
    {
        return Ok(content.to_string());
    }
    if let Some(reasoning) = value
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|v| v.get("message"))
        .and_then(|v| v.get("reasoning_content"))
        .and_then(|v| v.as_str())
    {
        return Ok(reasoning.to_string());
    }
    Err(anyhow!(
        "unexpected non-streaming payload: missing choices[0].message.content/reasoning_content"
    ))
}

fn parse_streaming_payload(body: &str) -> Result<String> {
    let mut content_out = String::new();
    let mut reasoning_out = String::new();
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
        if let Some(content) = value
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.get("delta"))
            .and_then(|v| v.get("content"))
            .and_then(|v| v.as_str())
        {
            content_out.push_str(content);
            parsed_any = true;
            continue;
        }
        if let Some(reasoning) = value
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.get("delta"))
            .and_then(|v| v.get("reasoning_content"))
            .and_then(|v| v.as_str())
        {
            reasoning_out.push_str(reasoning);
            parsed_any = true;
            continue;
        }
        if let Some(content) = value
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.get("message"))
            .and_then(|v| v.get("content"))
            .and_then(|v| v.as_str())
        {
            content_out.push_str(content);
            parsed_any = true;
            continue;
        }
        if let Some(reasoning) = value
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.get("message"))
            .and_then(|v| v.get("reasoning_content"))
            .and_then(|v| v.as_str())
        {
            reasoning_out.push_str(reasoning);
            parsed_any = true;
        }
    }

    if parsed_any {
        if !content_out.is_empty() {
            Ok(content_out)
        } else {
            Ok(reasoning_out)
        }
    } else {
        parse_non_streaming_payload(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_non_streaming() {
        let body = r#"{"choices":[{"message":{"content":"hello"}}]}"#;
        let got = parse_non_streaming_payload(body).expect("parse");
        assert_eq!(got, "hello");
    }

    #[test]
    fn parses_streaming_sse_lines() {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\ndata: [DONE]";
        let got = parse_streaming_payload(body).expect("stream parse");
        assert_eq!(got, "hello");
    }

    #[test]
    fn parses_streaming_reasoning_content() {
        let body = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"step1\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"reasoning_content\":\"step2\"}}]}\n\ndata: [DONE]";
        let got = parse_streaming_payload(body).expect("stream parse");
        assert_eq!(got, "step1step2");
    }
}
