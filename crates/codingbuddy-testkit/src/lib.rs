use anyhow::Result;
use codingbuddy_agent::{AgentEngine, ChatOptions};
use codingbuddy_core::{
    ChatMessage, ChatRequest, LlmRequest, LlmResponse, LlmToolCall, StreamCallback, TokenUsage,
};
use codingbuddy_llm::LlmClient;
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{Mutex, mpsc};
use std::thread;
use std::time::Duration;

// ── Scripted LLM for tool-use loop testing ───────────────────────────────

/// A scripted LLM client that pops pre-defined responses from a queue.
///
/// Optionally captures all `ChatRequest` messages for post-test inspection.
/// This is the canonical mock — integration tests should use this instead of
/// defining their own.
pub struct ScriptedLlm {
    responses: Mutex<VecDeque<LlmResponse>>,
    captured: Mutex<Vec<Vec<ChatMessage>>>,
}

impl ScriptedLlm {
    /// Create a new scripted LLM with the given response queue.
    pub fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::from(responses)),
            captured: Mutex::new(Vec::new()),
        }
    }

    /// Return all message histories captured from `complete_chat` calls.
    pub fn captured_messages(&self) -> Vec<Vec<ChatMessage>> {
        self.captured.lock().expect("captured lock").clone()
    }
}

impl LlmClient for ScriptedLlm {
    fn complete(&self, _req: &LlmRequest) -> Result<LlmResponse> {
        Err(anyhow::anyhow!("complete() not used in scripted tests"))
    }
    fn complete_streaming(&self, _req: &LlmRequest, _cb: StreamCallback) -> Result<LlmResponse> {
        Err(anyhow::anyhow!(
            "complete_streaming() not used in scripted tests"
        ))
    }
    fn complete_chat(&self, req: &ChatRequest) -> Result<LlmResponse> {
        self.captured
            .lock()
            .expect("captured lock")
            .push(req.messages.clone());
        self.responses
            .lock()
            .expect("responses lock")
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("scripted llm exhausted"))
    }
    fn complete_chat_streaming(
        &self,
        req: &ChatRequest,
        _cb: StreamCallback,
    ) -> Result<LlmResponse> {
        self.complete_chat(req)
    }
    fn complete_fim(&self, _req: &codingbuddy_core::FimRequest) -> Result<LlmResponse> {
        Err(anyhow::anyhow!("complete_fim() not used in scripted tests"))
    }
    fn complete_fim_streaming(
        &self,
        _req: &codingbuddy_core::FimRequest,
        _cb: StreamCallback,
    ) -> Result<LlmResponse> {
        Err(anyhow::anyhow!(
            "complete_fim_streaming() not used in scripted tests"
        ))
    }
}

/// Build a text-only LLM response (no tool calls).
pub fn scripted_text_response(text: &str) -> LlmResponse {
    LlmResponse {
        text: text.to_string(),
        finish_reason: "stop".to_string(),
        reasoning_content: String::new(),
        tool_calls: vec![],
        usage: Some(TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            ..Default::default()
        }),
    }
}

/// Build an LLM response containing tool calls.
pub fn scripted_tool_response(calls: Vec<LlmToolCall>) -> LlmResponse {
    LlmResponse {
        text: String::new(),
        finish_reason: "tool_calls".to_string(),
        reasoning_content: String::new(),
        tool_calls: calls,
        usage: Some(TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            ..Default::default()
        }),
    }
}

// ── Scenario-based mock LLM server ──────────────────────────────────────

/// A scripted response the mock LLM server should return.
#[derive(Debug, Clone)]
pub enum Scenario {
    /// Return a plain text response.
    TextResponse(String),
    /// Return a single tool call.
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
    /// Return multiple tool calls in one response.
    MultiToolCall(Vec<ToolCallSpec>),
    /// Return a response with reasoning_content and text (simulates deepseek-reasoner).
    ReasonerResponse {
        reasoning_content: String,
        text: String,
    },
    /// Return an HTTP error status code.
    HttpError(u16),
}

/// A single tool call within a `MultiToolCall` scenario.
#[derive(Debug, Clone)]
pub struct ToolCallSpec {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// A mock LLM HTTP server that returns scripted `Scenario` responses.
///
/// Each incoming HTTP request pops one scenario from the queue.
/// If the queue is empty, returns a default text response.
pub struct MockLlmServer {
    pub endpoint: String,
    scenario_tx: mpsc::Sender<Scenario>,
    stop_tx: Option<mpsc::Sender<()>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl MockLlmServer {
    /// Push a single scenario to the response queue.
    pub fn push(&self, scenario: Scenario) {
        self.scenario_tx
            .send(scenario)
            .expect("mock llm scenario queue receiver dropped");
    }

    /// Push multiple scenarios to the response queue.
    pub fn push_many(&self, scenarios: impl IntoIterator<Item = Scenario>) {
        for s in scenarios {
            let _ = self.scenario_tx.send(s);
        }
    }
}

impl Drop for MockLlmServer {
    fn drop(&mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Start a mock LLM server on a random local port.
///
/// The server listens for HTTP requests matching the CodingBuddy
/// `/chat/completions` endpoint format and returns scripted responses.
pub fn start_mock_llm_server() -> MockLlmServer {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock llm");
    listener
        .set_nonblocking(true)
        .expect("set nonblocking listener");
    let addr = listener.local_addr().expect("mock addr");
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let (scenario_tx, scenario_rx) = mpsc::channel::<Scenario>();

    let handle = thread::spawn(move || {
        loop {
            if stop_rx.try_recv().is_ok() {
                break;
            }
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let _ = stream.set_nonblocking(false);
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
                    let scenario = scenario_rx.try_recv().ok();
                    let _ = handle_mock_llm_connection(&mut stream, scenario.as_ref());
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5));
                }
                Err(_) => break,
            }
        }
    });

    MockLlmServer {
        endpoint: format!("http://{addr}/chat/completions"),
        scenario_tx,
        stop_tx: Some(stop_tx),
        handle: Some(handle),
    }
}

/// Write a `.codingbuddy/settings.local.json` pointing at the mock LLM endpoint.
pub fn configure_runtime_for_mock_llm(workspace: &Path, endpoint: &str) {
    let runtime = workspace.join(".codingbuddy");
    std::fs::create_dir_all(&runtime).expect("runtime");
    std::fs::write(
        runtime.join("settings.local.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "llm": {
                "provider": "deepseek",
                "endpoint": endpoint,
                "api_key_env": "DEEPSEEK_API_KEY"
            }
        }))
        .expect("serialize settings"),
    )
    .expect("write settings");
}

/// Create a temporary workspace directory with a pre-configured mock LLM server.
///
/// Returns `(TempDir, MockLlmServer)`. The `TempDir` is cleaned up on drop.
pub fn temp_workspace_with_mock() -> (tempfile::TempDir, MockLlmServer) {
    let dir = tempfile::tempdir().expect("create temp workspace");
    let mock = start_mock_llm_server();
    configure_runtime_for_mock_llm(dir.path(), &mock.endpoint);
    // SAFETY: process-local env setup in test before engine creation.
    unsafe {
        std::env::set_var("DEEPSEEK_API_KEY", "test-api-key");
    }
    (dir, mock)
}

// ── Public test helpers ─────────────────────────────────────────────────

/// Return a minimal valid `Session` for test use across crates.
pub fn fake_session() -> codingbuddy_core::Session {
    codingbuddy_core::Session {
        session_id: uuid::Uuid::now_v7(),
        workspace_root: "/tmp/test-workspace".to_string(),
        baseline_commit: None,
        status: codingbuddy_core::SessionState::Idle,
        budgets: codingbuddy_core::SessionBudgets {
            per_turn_seconds: 30,
            max_think_tokens: 1000,
        },
        active_plan_id: None,
    }
}

pub fn run_replay_smoke(workspace: &Path) -> Result<String> {
    let engine = AgentEngine::new(workspace)?;
    engine.chat_with_options(
        "replay test",
        ChatOptions {
            tools: false,
            ..Default::default()
        },
    )
}

// ── HTTP mock internals ─────────────────────────────────────────────────

fn handle_mock_llm_connection(
    stream: &mut TcpStream,
    scenario: Option<&Scenario>,
) -> std::io::Result<()> {
    // Read the full HTTP request (headers + body)
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 1024];
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

    // Build the response based on the scenario
    match scenario {
        Some(Scenario::HttpError(code)) => {
            let status_text = match code {
                400 => "Bad Request",
                429 => "Too Many Requests",
                500 => "Internal Server Error",
                _ => "Error",
            };
            let response = format!(
                "HTTP/1.1 {code} {status_text}\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
            );
            stream.write_all(response.as_bytes())?;
            stream.flush()?;
            return Ok(());
        }
        Some(Scenario::ToolCall {
            id,
            name,
            arguments,
        }) => {
            let payload = serde_json::json!({
                "choices": [{
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": null,
                        "tool_calls": [{
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments
                            }
                        }]
                    }
                }]
            })
            .to_string();
            write_http_response(stream, 200, &payload)?;
        }
        Some(Scenario::MultiToolCall(specs)) => {
            let tool_calls: Vec<serde_json::Value> = specs
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "id": s.id,
                        "type": "function",
                        "function": {
                            "name": s.name,
                            "arguments": s.arguments
                        }
                    })
                })
                .collect();
            let payload = serde_json::json!({
                "choices": [{
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": null,
                        "tool_calls": tool_calls
                    }
                }]
            })
            .to_string();
            write_http_response(stream, 200, &payload)?;
        }
        Some(Scenario::ReasonerResponse {
            reasoning_content,
            text,
        }) => {
            let payload = serde_json::json!({
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "reasoning_content": reasoning_content,
                        "content": text
                    }
                }]
            })
            .to_string();
            write_http_response(stream, 200, &payload)?;
        }
        Some(Scenario::TextResponse(text)) => {
            let payload = serde_json::json!({
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "content": text
                    }
                }]
            })
            .to_string();
            write_http_response(stream, 200, &payload)?;
        }
        None => {
            // Default: extract prompt and return text response
            let prompt =
                extract_prompt_from_request_body(&body).unwrap_or_else(|| "test".to_string());
            let content = if prompt.to_ascii_lowercase().contains("plan") {
                "Generated plan: discover files, propose edits, verify with tests.".to_string()
            } else {
                format!("Mock response: {prompt}")
            };
            let payload = serde_json::json!({
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "content": content
                    }
                }]
            })
            .to_string();
            write_http_response(stream, 200, &payload)?;
        }
    }
    Ok(())
}

fn write_http_response(stream: &mut TcpStream, status: u16, payload: &str) -> std::io::Result<()> {
    let status_text = match status {
        200 => "OK",
        _ => "Error",
    };
    let response = format!(
        "HTTP/1.1 {status} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()?;
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

fn extract_prompt_from_request_body(body: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    value
        .get("messages")
        .and_then(|v| v.as_array())
        .and_then(|rows| rows.last())
        .and_then(|row| row.get("content"))
        .and_then(|v| v.as_str())
        .map(ToString::to_string)
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn replay_smoke() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let workspace = std::env::temp_dir().join(format!("codingbuddy-testkit-replay-{suffix}"));
        fs::create_dir_all(&workspace).expect("workspace");
        let mock = start_mock_llm_server();
        configure_runtime_for_mock_llm(&workspace, &mock.endpoint);
        // SAFETY: process-local env setup in test before engine creation.
        unsafe {
            std::env::set_var("DEEPSEEK_API_KEY", "test-api-key");
        }
        let result = run_replay_smoke(&workspace);
        assert!(result.is_ok(), "replay smoke failed: {:?}", result.err());
    }

    #[test]
    fn scenario_text_response() {
        let mock = start_mock_llm_server();
        mock.push(Scenario::TextResponse("hello world".into()));
        // Verify we can connect and get a response
        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(&mock.endpoint)
            .json(&serde_json::json!({
                "messages": [{"role": "user", "content": "test"}]
            }))
            .send()
            .expect("request");
        let body: serde_json::Value = resp.json().expect("json");
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert_eq!(content, "hello world");
    }

    #[test]
    fn scenario_tool_call_response() {
        let mock = start_mock_llm_server();
        mock.push(Scenario::ToolCall {
            id: "call_1".into(),
            name: "fs_read".into(),
            arguments: r#"{"path":"README.md"}"#.into(),
        });
        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(&mock.endpoint)
            .json(&serde_json::json!({
                "messages": [{"role": "user", "content": "read readme"}]
            }))
            .send()
            .expect("request");
        let body: serde_json::Value = resp.json().expect("json");
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        let tc = &body["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["function"]["name"], "fs_read");
    }

    #[test]
    fn scenario_queue_pops_in_order() {
        let mock = start_mock_llm_server();
        mock.push(Scenario::TextResponse("first".into()));
        mock.push(Scenario::TextResponse("second".into()));

        let client = reqwest::blocking::Client::new();
        let req = serde_json::json!({"messages": [{"role": "user", "content": "test"}]});

        let r1: serde_json::Value = client
            .post(&mock.endpoint)
            .json(&req)
            .send()
            .expect("r1")
            .json()
            .expect("j1");
        assert_eq!(r1["choices"][0]["message"]["content"], "first");

        let r2: serde_json::Value = client
            .post(&mock.endpoint)
            .json(&req)
            .send()
            .expect("r2")
            .json()
            .expect("j2");
        assert_eq!(r2["choices"][0]["message"]["content"], "second");
    }

    #[test]
    fn temp_workspace_creates_settings() {
        let (dir, _mock) = temp_workspace_with_mock();
        let settings = dir.path().join(".codingbuddy/settings.local.json");
        assert!(settings.exists(), "settings.local.json should be created");
    }

    #[test]
    fn replay_deterministic_output() {
        let (dir, _mock) = temp_workspace_with_mock();
        let engine = AgentEngine::new(dir.path()).expect("engine");
        let r1 = engine.chat_with_options(
            "determinism test",
            ChatOptions {
                tools: false,
                ..Default::default()
            },
        );
        assert!(r1.is_ok(), "first chat failed: {:?}", r1.err());
        let r2 = engine.chat_with_options(
            "determinism test",
            ChatOptions {
                tools: false,
                ..Default::default()
            },
        );
        assert!(r2.is_ok(), "second chat failed: {:?}", r2.err());
        assert_eq!(r1.unwrap_or_default(), r2.unwrap_or_default());
    }

    #[test]
    fn replay_with_tool_calls_journals_events() {
        let (dir, _mock) = temp_workspace_with_mock();
        let engine = AgentEngine::new(dir.path()).expect("engine");
        let r = engine.chat_with_options(
            "journal test",
            ChatOptions {
                tools: false,
                ..Default::default()
            },
        );
        assert!(r.is_ok(), "chat failed: {:?}", r.err());
        assert!(r.unwrap_or_default().contains("Mock response"));
        // The non-edit analysis path no longer requires event journaling.
        let events_path = dir.path().join(".codingbuddy/events.jsonl");
        if events_path.exists() {
            let contents = fs::read_to_string(&events_path).expect("read events");
            assert!(!contents.trim().is_empty());
        }
    }
}
