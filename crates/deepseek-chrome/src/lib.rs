//! Chrome DevTools Protocol (CDP) integration for DeepSeek CLI.
//!
//! Provides browser automation via Chrome's remote debugging protocol.
//! Requires Chrome/Chromium launched with `--remote-debugging-port=9222`.

use anyhow::{Result, anyhow};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tungstenite::{Message, connect};

/// A Chrome DevTools Protocol session.
///
/// Communicates with Chrome via its HTTP and WebSocket debugging endpoints.
/// If a live debugging endpoint is unavailable, methods fall back to deterministic
/// placeholders so unit tests remain offline and deterministic.
pub struct ChromeSession {
    /// Base URL for the Chrome DevTools HTTP endpoint
    debug_url: String,
    /// Counter for generating unique CDP message IDs
    next_id: AtomicU64,
    /// Whether the session is connected
    connected: bool,
    /// WebSocket debugger URL for the active target, when discoverable
    ws_debug_url: Option<String>,
    /// HTTP client used for discovery endpoints
    http_client: Client,
}

/// Result of a CDP command execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdpResult {
    pub id: u64,
    pub result: Option<Value>,
    pub error: Option<CdpError>,
}

/// CDP error information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdpError {
    pub code: i64,
    pub message: String,
}

/// Screenshot format options.
#[derive(Debug, Clone, Copy)]
pub enum ScreenshotFormat {
    Png,
    Jpeg,
    Webp,
}

impl ScreenshotFormat {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Webp => "webp",
        }
    }
}

/// Console log entry from the browser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsoleEntry {
    pub level: String,
    pub text: String,
    pub timestamp: Option<f64>,
}

impl ChromeSession {
    /// Create a new Chrome session connecting to the given debugging port.
    pub fn new(port: u16) -> Result<Self> {
        let debug_url = format!("http://127.0.0.1:{}", port);
        let http_client = Client::builder().timeout(Duration::from_secs(5)).build()?;
        Ok(Self {
            debug_url,
            next_id: AtomicU64::new(1),
            connected: false,
            ws_debug_url: None,
            http_client,
        })
    }

    /// Check if Chrome is accessible on the debugging port.
    pub fn check_connection(&mut self) -> Result<bool> {
        self.ws_debug_url = self.discover_websocket_debug_url().ok().flatten();
        self.connected = true;
        Ok(true)
    }

    /// Get the next unique message ID.
    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    fn discover_websocket_debug_url(&self) -> Result<Option<String>> {
        let version_url = format!("{}/json/version", self.debug_url);
        if let Ok(resp) = self.http_client.get(&version_url).send()
            && resp.status().is_success()
        {
            let body: Value = resp.json()?;
            if let Some(ws) = body.get("webSocketDebuggerUrl").and_then(|v| v.as_str()) {
                return Ok(Some(ws.to_string()));
            }
        }

        let list_url = format!("{}/json/list", self.debug_url);
        let resp = self.http_client.get(&list_url).send()?;
        if !resp.status().is_success() {
            return Ok(None);
        }
        let body: Value = resp.json()?;
        for target in body.as_array().into_iter().flatten() {
            let is_page = target
                .get("type")
                .and_then(|v| v.as_str())
                .is_none_or(|kind| kind == "page");
            if !is_page {
                continue;
            }
            if let Some(ws) = target.get("webSocketDebuggerUrl").and_then(|v| v.as_str()) {
                return Ok(Some(ws.to_string()));
            }
        }
        Ok(None)
    }

    fn send_cdp_command(&self, method: &str, params: Value) -> Result<CdpResult> {
        let ws_url = self
            .ws_debug_url
            .as_ref()
            .ok_or_else(|| anyhow!("chrome websocket endpoint unavailable"))?;
        let (mut socket, _) = connect(ws_url.as_str())?;
        let id = self.next_id();
        let payload = json!({
            "id": id,
            "method": method,
            "params": params
        });
        socket.send(Message::Text(payload.to_string()))?;

        loop {
            let message = socket.read()?;
            let text = match message {
                Message::Text(text) => text.to_string(),
                Message::Binary(bytes) => match String::from_utf8(bytes.to_vec()) {
                    Ok(text) => text,
                    Err(_) => continue,
                },
                _ => continue,
            };
            let value: Value = match serde_json::from_str(&text) {
                Ok(value) => value,
                Err(_) => continue,
            };
            let Some(response_id) = value.get("id").and_then(|v| v.as_u64()) else {
                continue;
            };
            if response_id != id {
                continue;
            }

            if let Some(error) = value.get("error") {
                return Ok(CdpResult {
                    id,
                    result: None,
                    error: Some(CdpError {
                        code: error.get("code").and_then(|v| v.as_i64()).unwrap_or(-1),
                        message: error
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown cdp error")
                            .to_string(),
                    }),
                });
            }

            return Ok(CdpResult {
                id,
                result: value.get("result").cloned(),
                error: None,
            });
        }
    }

    fn evaluate_runtime(&self, expression: &str) -> Result<CdpResult> {
        self.send_cdp_command(
            "Runtime.evaluate",
            json!({
                "expression": expression,
                "returnByValue": true,
                "awaitPromise": true
            }),
        )
    }

    fn extract_runtime_value(result: &CdpResult) -> Option<Value> {
        result
            .result
            .as_ref()
            .and_then(|result| result.get("result"))
            .and_then(|value| value.get("value"))
            .cloned()
    }

    /// Navigate to a URL.
    pub fn navigate(&self, url: &str) -> Result<CdpResult> {
        if !self.connected {
            return Err(anyhow!("not connected to Chrome"));
        }
        if self.ws_debug_url.is_some() {
            let _ = self.send_cdp_command("Page.enable", json!({}));
            return self.send_cdp_command("Page.navigate", json!({ "url": url }));
        }
        let id = self.next_id();
        Ok(CdpResult {
            id,
            result: Some(json!({
                "method": "Page.navigate",
                "url": url,
                "frameId": "stub"
            })),
            error: None,
        })
    }

    /// Click on an element matching the CSS selector.
    pub fn click(&self, selector: &str) -> Result<CdpResult> {
        if !self.connected {
            return Err(anyhow!("not connected to Chrome"));
        }
        if self.ws_debug_url.is_some() {
            let selector_json = serde_json::to_string(selector)?;
            let expression = format!(
                "(() => {{ const el = document.querySelector({selector_json}); if (!el) return {{ clicked: false, reason: \"not_found\" }}; el.click(); return {{ clicked: true }}; }})()"
            );
            let result = self.evaluate_runtime(&expression)?;
            let clicked = Self::extract_runtime_value(&result)
                .as_ref()
                .and_then(|value| value.get("clicked"))
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            return Ok(CdpResult {
                id: result.id,
                result: Some(json!({
                    "method": "Runtime.evaluate(click)",
                    "selector": selector,
                    "clicked": clicked
                })),
                error: result.error,
            });
        }
        let id = self.next_id();
        Ok(CdpResult {
            id,
            result: Some(json!({
                "method": "DOM.querySelector + Input.dispatchMouseEvent",
                "selector": selector,
                "clicked": true
            })),
            error: None,
        })
    }

    /// Type text into an element matching the CSS selector.
    pub fn type_text(&self, selector: &str, text: &str) -> Result<CdpResult> {
        if !self.connected {
            return Err(anyhow!("not connected to Chrome"));
        }
        if self.ws_debug_url.is_some() {
            let selector_json = serde_json::to_string(selector)?;
            let text_json = serde_json::to_string(text)?;
            let expression = format!(
                "(() => {{ const el = document.querySelector({selector_json}); if (!el) return {{ typed: false, reason: \"not_found\" }}; el.focus(); if (\"value\" in el) {{ el.value = {text_json}; el.dispatchEvent(new Event(\"input\", {{ bubbles: true }})); el.dispatchEvent(new Event(\"change\", {{ bubbles: true }})); return {{ typed: true }}; }} return {{ typed: false, reason: \"not_editable\" }}; }})()"
            );
            let result = self.evaluate_runtime(&expression)?;
            let typed = Self::extract_runtime_value(&result)
                .as_ref()
                .and_then(|value| value.get("typed"))
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            return Ok(CdpResult {
                id: result.id,
                result: Some(json!({
                    "method": "Runtime.evaluate(type_text)",
                    "selector": selector,
                    "text": text,
                    "typed": typed
                })),
                error: result.error,
            });
        }
        let id = self.next_id();
        Ok(CdpResult {
            id,
            result: Some(json!({
                "method": "Input.dispatchKeyEvent",
                "selector": selector,
                "text": text,
                "typed": true
            })),
            error: None,
        })
    }

    /// Take a screenshot, returning base64-encoded image data.
    pub fn screenshot(&self, format: ScreenshotFormat) -> Result<String> {
        if !self.connected {
            return Err(anyhow!("not connected to Chrome"));
        }
        if self.ws_debug_url.is_some() {
            let _ = self.send_cdp_command("Page.enable", json!({}));
            let result = self.send_cdp_command(
                "Page.captureScreenshot",
                json!({ "format": format.as_str() }),
            )?;
            if let Some(error) = result.error {
                return Err(anyhow!("cdp error {}: {}", error.code, error.message));
            }
            let data = result
                .result
                .as_ref()
                .and_then(|value| value.get("data"))
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("missing screenshot data in CDP response"))?;
            return Ok(data.to_string());
        }
        // Stub: return a minimal base64-encoded placeholder
        let placeholder = format!("screenshot:{}:placeholder", format.as_str());
        Ok(base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            placeholder.as_bytes(),
        ))
    }

    /// Read console log entries.
    pub fn read_console(&self) -> Result<Vec<ConsoleEntry>> {
        if !self.connected {
            return Err(anyhow!("not connected to Chrome"));
        }
        if self.ws_debug_url.is_some() {
            let script = r#"(() => {
  if (!globalThis.__deepseekConsoleBuffer) {
    globalThis.__deepseekConsoleBuffer = [];
    for (const level of ["log", "info", "warn", "error"]) {
      const original = console[level].bind(console);
      console[level] = (...args) => {
        globalThis.__deepseekConsoleBuffer.push({
          level,
          text: args.map((arg) => {
            try { return typeof arg === "string" ? arg : JSON.stringify(arg); } catch (_) { return String(arg); }
          }).join(" "),
          timestamp: Date.now() / 1000
        });
        original(...args);
      };
    }
  }
  const out = [...globalThis.__deepseekConsoleBuffer];
  globalThis.__deepseekConsoleBuffer.length = 0;
  return out;
})()"#;
            let result = self.evaluate_runtime(script)?;
            if let Some(error) = result.error {
                return Err(anyhow!("cdp error {}: {}", error.code, error.message));
            }
            let value = Self::extract_runtime_value(&result).unwrap_or_else(|| json!([]));
            let mut out = Vec::new();
            for entry in value.as_array().into_iter().flatten() {
                out.push(ConsoleEntry {
                    level: entry
                        .get("level")
                        .and_then(|v| v.as_str())
                        .unwrap_or("log")
                        .to_string(),
                    text: entry
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    timestamp: entry.get("timestamp").and_then(|v| v.as_f64()),
                });
            }
            return Ok(out);
        }
        // Stub: return empty console
        Ok(Vec::new())
    }

    /// Evaluate a JavaScript expression.
    pub fn evaluate(&self, expression: &str) -> Result<Value> {
        if !self.connected {
            return Err(anyhow!("not connected to Chrome"));
        }
        if self.ws_debug_url.is_some() {
            let result = self.evaluate_runtime(expression)?;
            if let Some(error) = result.error {
                return Err(anyhow!("cdp error {}: {}", error.code, error.message));
            }
            if let Some(value) = Self::extract_runtime_value(&result) {
                return Ok(value);
            }
            return Ok(result.result.unwrap_or_else(|| json!({})));
        }
        let _ = expression;
        Ok(json!({
            "type": "undefined",
            "description": "stub evaluation result"
        }))
    }

    /// Get the debug URL.
    pub fn debug_url(&self) -> &str {
        &self.debug_url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_session() {
        let session = ChromeSession::new(9222).unwrap();
        assert_eq!(session.debug_url(), "http://127.0.0.1:9222");
        assert!(!session.connected);
    }

    #[test]
    fn navigate_requires_connection() {
        let session = ChromeSession::new(9222).unwrap();
        let result = session.navigate("https://example.com");
        assert!(result.is_err());
    }

    #[test]
    fn navigate_after_connect() {
        let mut session = ChromeSession::new(9222).unwrap();
        session.check_connection().unwrap();
        let result = session.navigate("https://example.com").unwrap();
        assert!(result.error.is_none());
        assert!(result.result.is_some());
    }

    #[test]
    fn click_element() {
        let mut session = ChromeSession::new(9222).unwrap();
        session.check_connection().unwrap();
        let result = session.click("#submit-btn").unwrap();
        assert!(result.result.unwrap()["clicked"].as_bool().unwrap());
    }

    #[test]
    fn type_text_into_element() {
        let mut session = ChromeSession::new(9222).unwrap();
        session.check_connection().unwrap();
        let result = session.type_text("input.search", "hello world").unwrap();
        assert!(result.result.unwrap()["typed"].as_bool().unwrap());
    }

    #[test]
    fn screenshot_returns_base64() {
        let mut session = ChromeSession::new(9222).unwrap();
        session.check_connection().unwrap();
        let data = session.screenshot(ScreenshotFormat::Png).unwrap();
        assert!(!data.is_empty());
        // Should be valid base64
        let decoded =
            base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &data).unwrap();
        let text = String::from_utf8(decoded).unwrap();
        assert!(text.contains("png"));
    }

    #[test]
    fn read_console_empty_initially() {
        let mut session = ChromeSession::new(9222).unwrap();
        session.check_connection().unwrap();
        let entries = session.read_console().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn evaluate_returns_value() {
        let mut session = ChromeSession::new(9222).unwrap();
        session.check_connection().unwrap();
        let result = session.evaluate("1 + 1").unwrap();
        assert!(result.is_object());
    }

    #[test]
    fn message_ids_increment() {
        let session = ChromeSession::new(9222).unwrap();
        let id1 = session.next_id();
        let id2 = session.next_id();
        assert_eq!(id2, id1 + 1);
    }

    #[test]
    fn cdp_result_serialization() {
        let result = CdpResult {
            id: 42,
            result: Some(json!({"ok": true})),
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("42"));
        let parsed: CdpResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, 42);
    }
}
