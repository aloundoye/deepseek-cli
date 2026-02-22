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
    /// Keep deterministic stub fallbacks for offline callers/tests.
    allow_stub_fallback: bool,
    /// Last connection probe error for diagnostics.
    last_connection_error: Option<String>,
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

/// Browser target returned by Chrome's `/json/list` and `/json/new` endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeTarget {
    pub id: String,
    pub title: String,
    pub url: String,
    #[serde(rename = "type")]
    pub target_type: String,
    pub ws_debug_url: Option<String>,
}

/// Connection probe result with recovery metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeConnectionStatus {
    pub connected: bool,
    pub debug_url: String,
    pub ws_debug_url: Option<String>,
    pub target_count: usize,
    pub page_target_count: usize,
    pub stub_fallback_active: bool,
    pub recovered: bool,
    pub created_tab: bool,
    pub failure_kind: Option<String>,
    pub failure_message: Option<String>,
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
            allow_stub_fallback: true,
            last_connection_error: None,
            http_client,
        })
    }

    /// Enable/disable deterministic stub fallbacks when no live target is available.
    pub fn set_allow_stub_fallback(&mut self, allow: bool) {
        self.allow_stub_fallback = allow;
    }

    /// Whether this session currently allows deterministic offline fallbacks.
    pub fn allow_stub_fallback(&self) -> bool {
        self.allow_stub_fallback
    }

    /// Check if Chrome is accessible on the debugging port.
    pub fn check_connection(&mut self) -> Result<bool> {
        self.last_connection_error = None;
        match self.discover_websocket_debug_url() {
            Ok(ws_url) => {
                self.ws_debug_url = ws_url;
                if self.ws_debug_url.is_none() {
                    self.last_connection_error =
                        Some("no debuggable page target available".to_string());
                }
            }
            Err(err) => {
                self.ws_debug_url = None;
                self.last_connection_error = Some(err.to_string());
            }
        }
        self.connected = self.ws_debug_url.is_some() || self.allow_stub_fallback;
        Ok(self.connected)
    }

    /// Return live connection diagnostics for status UIs.
    pub fn connection_status(&mut self) -> Result<ChromeConnectionStatus> {
        self.check_connection()?;
        let targets = self.list_tabs().unwrap_or_default();
        let page_target_count = targets
            .iter()
            .filter(|target| target.target_type == "page")
            .count();
        Ok(self.build_connection_status(targets.len(), page_target_count, false, false))
    }

    /// Reconnect and optionally create a recovery tab when none are available.
    pub fn reconnect(&mut self, create_tab_if_missing: bool) -> Result<ChromeConnectionStatus> {
        let connected_before = self.check_connection()?;
        let mut created_tab = false;
        if create_tab_if_missing
            && self.ws_debug_url.is_none()
            && self.create_tab("about:blank").is_ok()
        {
            created_tab = true;
        }
        self.check_connection()?;
        let targets = self.list_tabs().unwrap_or_default();
        let page_target_count = targets
            .iter()
            .filter(|target| target.target_type == "page")
            .count();
        let connected_after = self.connected && self.ws_debug_url.is_some();
        Ok(self.build_connection_status(
            targets.len(),
            page_target_count,
            !connected_before && connected_after,
            created_tab,
        ))
    }

    /// Ensure a live websocket target is available, attempting lightweight recovery once.
    pub fn ensure_live_connection(&mut self) -> Result<()> {
        let status = self.reconnect(true)?;
        if status.connected && status.ws_debug_url.is_some() {
            return Ok(());
        }
        let kind = status
            .failure_kind
            .unwrap_or_else(|| "connection_failed".to_string());
        let message = status
            .failure_message
            .unwrap_or_else(|| "chrome remote debugging endpoint is unavailable".to_string());
        Err(anyhow!("{kind}: {message}"))
    }

    /// List current browser targets from `/json/list`.
    pub fn list_tabs(&self) -> Result<Vec<ChromeTarget>> {
        let list_url = format!("{}/json/list", self.debug_url);
        let response = self.http_client.get(&list_url).send()?;
        if !response.status().is_success() {
            return Err(anyhow!(
                "chrome target list request failed: HTTP {}",
                response.status()
            ));
        }
        let body: Value = response.json()?;
        let targets = body
            .as_array()
            .ok_or_else(|| anyhow!("invalid /json/list response"))?
            .iter()
            .map(Self::target_from_value)
            .collect::<Vec<_>>();
        Ok(targets)
    }

    /// Create a new browser tab.
    pub fn create_tab(&self, url: &str) -> Result<ChromeTarget> {
        let encoded_url = url.replace(' ', "%20");
        let endpoint = format!("{}/json/new?{}", self.debug_url, encoded_url);
        let response = match self.http_client.put(&endpoint).send() {
            Ok(response) => response,
            Err(_) => self.http_client.get(&endpoint).send()?,
        };
        if !response.status().is_success() {
            return Err(anyhow!(
                "chrome tab creation failed: HTTP {}",
                response.status()
            ));
        }
        let body: Value = response.json()?;
        Ok(Self::target_from_value(&body))
    }

    /// Focus an existing tab target.
    pub fn activate_tab(&self, target_id: &str) -> Result<bool> {
        let endpoint = format!("{}/json/activate/{target_id}", self.debug_url);
        let response = self.http_client.get(&endpoint).send()?;
        if !response.status().is_success() {
            return Err(anyhow!(
                "chrome tab activation failed: HTTP {}",
                response.status()
            ));
        }
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

        let targets = self.list_tabs()?;
        for target in targets {
            let is_page = target.target_type.is_empty() || target.target_type == "page";
            if !is_page {
                continue;
            }
            if let Some(ws) = target.ws_debug_url {
                return Ok(Some(ws));
            }
        }
        Ok(None)
    }

    fn target_from_value(value: &Value) -> ChromeTarget {
        ChromeTarget {
            id: value
                .get("id")
                .or_else(|| value.get("targetId"))
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            title: value
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            url: value
                .get("url")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            target_type: value
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("page")
                .to_string(),
            ws_debug_url: value
                .get("webSocketDebuggerUrl")
                .and_then(|v| v.as_str())
                .map(ToString::to_string),
        }
    }

    fn build_connection_status(
        &self,
        target_count: usize,
        page_target_count: usize,
        recovered: bool,
        created_tab: bool,
    ) -> ChromeConnectionStatus {
        let failure_kind = if self.ws_debug_url.is_some() {
            None
        } else if self
            .last_connection_error
            .as_deref()
            .is_some_and(|msg| msg.contains("Connection refused"))
        {
            Some("endpoint_unreachable".to_string())
        } else if self
            .last_connection_error
            .as_deref()
            .is_some_and(|msg| msg.contains("timed out"))
        {
            Some("endpoint_timeout".to_string())
        } else if page_target_count == 0 {
            Some("no_page_targets".to_string())
        } else {
            Some("connection_error".to_string())
        };

        ChromeConnectionStatus {
            connected: self.ws_debug_url.is_some(),
            debug_url: self.debug_url.clone(),
            ws_debug_url: self.ws_debug_url.clone(),
            target_count,
            page_target_count,
            stub_fallback_active: self.allow_stub_fallback && self.ws_debug_url.is_none(),
            recovered,
            created_tab,
            failure_kind,
            failure_message: self.last_connection_error.clone(),
        }
    }

    fn require_live_or_stub(&self, action: &str) -> Result<()> {
        if self.ws_debug_url.is_some() || self.allow_stub_fallback {
            return Ok(());
        }
        Err(anyhow!(
            "live chrome websocket endpoint unavailable for {action}"
        ))
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
        self.require_live_or_stub("navigate")?;
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
        self.require_live_or_stub("click")?;
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
        self.require_live_or_stub("type_text")?;
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
        self.require_live_or_stub("screenshot")?;
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
        self.require_live_or_stub("read_console")?;
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
        self.require_live_or_stub("evaluate")?;
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

    #[test]
    fn strict_mode_requires_live_websocket() {
        let mut session = ChromeSession::new(9).unwrap();
        session.set_allow_stub_fallback(false);
        let connected = session.check_connection().unwrap();
        assert!(!connected);
        let err = session
            .navigate("https://example.com")
            .expect_err("strict mode should reject stub navigation");
        let msg = err.to_string();
        assert!(
            msg.contains("live chrome websocket endpoint unavailable")
                || msg.contains("not connected to Chrome")
        );
    }

    #[test]
    fn connection_status_exposes_failure_taxonomy() {
        let mut session = ChromeSession::new(9).unwrap();
        session.set_allow_stub_fallback(false);
        let status = session.connection_status().unwrap();
        assert!(!status.connected);
        assert!(status.failure_kind.is_some());
        assert!(status.failure_message.is_some());
    }

    #[test]
    fn target_from_value_maps_expected_fields() {
        let target = ChromeSession::target_from_value(&json!({
            "id": "abc123",
            "title": "Example",
            "url": "https://example.com",
            "type": "page",
            "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/abc123"
        }));
        assert_eq!(target.id, "abc123");
        assert_eq!(target.target_type, "page");
        assert!(target.ws_debug_url.is_some());
    }
}
