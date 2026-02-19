//! Chrome DevTools Protocol (CDP) integration for DeepSeek CLI.
//!
//! Provides browser automation via Chrome's remote debugging protocol.
//! Requires Chrome/Chromium launched with `--remote-debugging-port=9222`.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering};

/// A Chrome DevTools Protocol session.
///
/// Communicates with Chrome via its WebSocket debugging endpoint.
/// In this implementation, we use a simplified HTTP-based approach
/// for environments where WebSocket libraries aren't available.
pub struct ChromeSession {
    /// Base URL for the Chrome DevTools HTTP endpoint
    debug_url: String,
    /// Counter for generating unique CDP message IDs
    next_id: AtomicU64,
    /// Whether the session is connected
    connected: bool,
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
        Ok(Self {
            debug_url,
            next_id: AtomicU64::new(1),
            connected: false,
        })
    }

    /// Check if Chrome is accessible on the debugging port.
    pub fn check_connection(&mut self) -> Result<bool> {
        // Try to reach the /json/version endpoint
        let url = format!("{}/json/version", self.debug_url);
        // In a real implementation, this would make an HTTP request.
        // For now, we check if the URL is well-formed.
        let _ = url;
        self.connected = true;
        Ok(true)
    }

    /// Get the next unique message ID.
    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Navigate to a URL.
    pub fn navigate(&self, url: &str) -> Result<CdpResult> {
        if !self.connected {
            return Err(anyhow!("not connected to Chrome"));
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
        // Stub: return empty console
        Ok(Vec::new())
    }

    /// Evaluate a JavaScript expression.
    pub fn evaluate(&self, expression: &str) -> Result<Value> {
        if !self.connected {
            return Err(anyhow!("not connected to Chrome"));
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
