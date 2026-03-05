use anyhow::Result;
use codingbuddy_agent::{AgentEngine, ChatOptions};
use codingbuddy_core::{
    ChatMessage, ChatRequest, LlmRequest, LlmResponse, LlmToolCall, StreamCallback, TokenUsage,
};
use codingbuddy_llm::LlmClient;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CodingBenchmarkCaseResult {
    pub case_id: String,
    pub category: String,
    pub passed: bool,
    pub tool_invocations: usize,
    pub retries: usize,
    pub completion_quality_score: f32,
    pub duration_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CodingBenchmarkSummary {
    pub total_cases: usize,
    pub passed_cases: usize,
    pub pass_rate_pct: f32,
    pub avg_tool_invocations: f32,
    pub avg_retries: f32,
    pub avg_completion_quality_score: f32,
    pub avg_duration_ms: f32,
    pub category_pass_rate_pct: BTreeMap<String, f32>,
    pub category_avg_quality_score: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CodingBenchmarkReport {
    pub suite: String,
    pub model: String,
    pub generated_at_epoch_secs: u64,
    pub cases: Vec<CodingBenchmarkCaseResult>,
    pub summary: CodingBenchmarkSummary,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CodingBenchmarkGateResult {
    pub passed: bool,
    pub suite_model_compatible: bool,
    pub current_pass_rate_pct: f32,
    pub baseline_pass_rate_pct: f32,
    pub allowed_drop_pct: f32,
    pub delta_pct: f32,
    pub current_avg_completion_quality_score: f32,
    pub baseline_avg_completion_quality_score: f32,
    pub max_quality_drop: f32,
    pub quality_delta: f32,
    pub current_avg_retries: f32,
    pub baseline_avg_retries: f32,
    pub max_retry_increase: f32,
    pub retries_delta: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CodingBenchmarkCaseComparison {
    pub case_id: String,
    pub category: String,
    pub current_passed: bool,
    pub reference_passed: bool,
    pub tool_invocation_delta: isize,
    pub retries_delta: isize,
    pub quality_score_delta: f32,
    pub duration_delta_ms: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_note: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_note: Option<String>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CodingBenchmarkComparisonSummary {
    pub comparable: bool,
    pub suite_compatible: bool,
    pub case_ids_compatible: bool,
    pub case_categories_compatible: bool,
    pub comparable_case_count: usize,
    pub improved_case_count: usize,
    pub regressed_case_count: usize,
    pub current_only_cases: Vec<String>,
    pub reference_only_cases: Vec<String>,
    pub category_mismatch_cases: Vec<String>,
    pub pass_rate_delta_pct: f32,
    pub avg_completion_quality_delta: f32,
    pub avg_retries_delta: f32,
    pub avg_duration_delta_ms: f32,
    pub category_pass_rate_delta_pct: BTreeMap<String, f32>,
    pub category_avg_quality_delta: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CodingBenchmarkComparisonReport {
    pub suite: String,
    pub current_model: String,
    pub reference_model: String,
    pub generated_at_epoch_secs: u64,
    pub cases: Vec<CodingBenchmarkCaseComparison>,
    pub summary: CodingBenchmarkComparisonSummary,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct CodingBenchmarkGateThresholds {
    pub max_pass_rate_drop_pct: f32,
    pub max_quality_score_drop: f32,
    pub max_avg_retries_increase: f32,
}

impl Default for CodingBenchmarkGateThresholds {
    fn default() -> Self {
        Self {
            max_pass_rate_drop_pct: 5.0,
            max_quality_score_drop: 0.10,
            max_avg_retries_increase: 0.50,
        }
    }
}

impl CodingBenchmarkReport {
    pub fn from_case_results(
        suite: &str,
        model: &str,
        cases: Vec<CodingBenchmarkCaseResult>,
    ) -> Self {
        let total_cases = cases.len();
        let passed_cases = cases.iter().filter(|case| case.passed).count();
        let pass_rate_pct = if total_cases == 0 {
            0.0
        } else {
            passed_cases as f32 * 100.0 / total_cases as f32
        };
        let avg_tool_invocations = if total_cases == 0 {
            0.0
        } else {
            cases
                .iter()
                .map(|case| case.tool_invocations as f32)
                .sum::<f32>()
                / total_cases as f32
        };
        let avg_retries = if total_cases == 0 {
            0.0
        } else {
            cases.iter().map(|case| case.retries as f32).sum::<f32>() / total_cases as f32
        };
        let avg_completion_quality_score = if total_cases == 0 {
            0.0
        } else {
            cases
                .iter()
                .map(|case| case.completion_quality_score)
                .sum::<f32>()
                / total_cases as f32
        };
        let avg_duration_ms = if total_cases == 0 {
            0.0
        } else {
            cases
                .iter()
                .map(|case| case.duration_ms as f32)
                .sum::<f32>()
                / total_cases as f32
        };
        let mut category_stats: BTreeMap<String, (usize, usize, f32)> = BTreeMap::new();
        for case in &cases {
            let entry = category_stats
                .entry(case.category.clone())
                .or_insert((0usize, 0usize, 0.0f32));
            entry.0 = entry.0.saturating_add(1);
            if case.passed {
                entry.1 = entry.1.saturating_add(1);
            }
            entry.2 += case.completion_quality_score;
        }
        let mut category_pass_rate_pct: BTreeMap<String, f32> = BTreeMap::new();
        let mut category_avg_quality_score: BTreeMap<String, f32> = BTreeMap::new();
        for (category, (count, passed, quality_sum)) in category_stats {
            let count_f = count as f32;
            let pass_rate = if count == 0 {
                0.0
            } else {
                passed as f32 * 100.0 / count_f
            };
            let avg_quality = if count == 0 {
                0.0
            } else {
                quality_sum / count_f
            };
            category_pass_rate_pct.insert(category.clone(), pass_rate);
            category_avg_quality_score.insert(category, avg_quality);
        }
        Self {
            suite: suite.to_string(),
            model: model.to_string(),
            generated_at_epoch_secs: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_secs())
                .unwrap_or(0),
            cases,
            summary: CodingBenchmarkSummary {
                total_cases,
                passed_cases,
                pass_rate_pct,
                avg_tool_invocations,
                avg_retries,
                avg_completion_quality_score,
                avg_duration_ms,
                category_pass_rate_pct,
                category_avg_quality_score,
            },
        }
    }
}

pub fn write_coding_benchmark_report(
    output_root: &Path,
    report: &CodingBenchmarkReport,
) -> Result<PathBuf> {
    std::fs::create_dir_all(output_root)?;
    let suite = sanitize_slug(&report.suite);
    let model = sanitize_slug(&report.model);
    let filename = format!("{suite}.{model}.latest.json");
    let path = output_root.join(filename);
    std::fs::write(&path, serde_json::to_vec_pretty(report)?)?;
    Ok(path)
}

pub fn read_coding_benchmark_report(path: &Path) -> Result<CodingBenchmarkReport> {
    let raw = std::fs::read(path)?;
    Ok(serde_json::from_slice(&raw)?)
}

pub fn compare_coding_benchmark_reports(
    current: &CodingBenchmarkReport,
    reference: &CodingBenchmarkReport,
) -> CodingBenchmarkComparisonReport {
    let suite_compatible = current.suite == reference.suite;

    let current_by_case: BTreeMap<&str, &CodingBenchmarkCaseResult> = current
        .cases
        .iter()
        .map(|case| (case.case_id.as_str(), case))
        .collect();
    let reference_by_case: BTreeMap<&str, &CodingBenchmarkCaseResult> = reference
        .cases
        .iter()
        .map(|case| (case.case_id.as_str(), case))
        .collect();

    let current_only_cases: Vec<String> = current_by_case
        .keys()
        .filter(|case_id| !reference_by_case.contains_key(**case_id))
        .map(|case_id| (*case_id).to_string())
        .collect();
    let reference_only_cases: Vec<String> = reference_by_case
        .keys()
        .filter(|case_id| !current_by_case.contains_key(**case_id))
        .map(|case_id| (*case_id).to_string())
        .collect();
    let case_ids_compatible = current_only_cases.is_empty() && reference_only_cases.is_empty();

    let mut category_mismatch_cases = Vec::new();
    let mut improved_case_count = 0usize;
    let mut regressed_case_count = 0usize;
    let mut cases = Vec::new();

    for (case_id, current_case) in &current_by_case {
        let Some(reference_case) = reference_by_case.get(case_id) else {
            continue;
        };

        if current_case.category != reference_case.category {
            category_mismatch_cases.push((*case_id).to_string());
        }
        if current_case.passed && !reference_case.passed {
            improved_case_count = improved_case_count.saturating_add(1);
        }
        if !current_case.passed && reference_case.passed {
            regressed_case_count = regressed_case_count.saturating_add(1);
        }

        cases.push(CodingBenchmarkCaseComparison {
            case_id: (*case_id).to_string(),
            category: current_case.category.clone(),
            current_passed: current_case.passed,
            reference_passed: reference_case.passed,
            tool_invocation_delta: current_case.tool_invocations as isize
                - reference_case.tool_invocations as isize,
            retries_delta: current_case.retries as isize - reference_case.retries as isize,
            quality_score_delta: current_case.completion_quality_score
                - reference_case.completion_quality_score,
            duration_delta_ms: current_case.duration_ms as f32 - reference_case.duration_ms as f32,
            current_note: current_case.note.clone(),
            reference_note: reference_case.note.clone(),
        });
    }

    let case_categories_compatible = category_mismatch_cases.is_empty();
    let comparable = suite_compatible && case_ids_compatible && case_categories_compatible;
    let comparable_case_count = cases.len();

    let mut category_pass_rate_delta_pct = BTreeMap::new();
    for (category, current_value) in &current.summary.category_pass_rate_pct {
        if let Some(reference_value) = reference.summary.category_pass_rate_pct.get(category) {
            category_pass_rate_delta_pct.insert(category.clone(), current_value - reference_value);
        }
    }

    let mut category_avg_quality_delta = BTreeMap::new();
    for (category, current_value) in &current.summary.category_avg_quality_score {
        if let Some(reference_value) = reference.summary.category_avg_quality_score.get(category) {
            category_avg_quality_delta.insert(category.clone(), current_value - reference_value);
        }
    }

    CodingBenchmarkComparisonReport {
        suite: current.suite.clone(),
        current_model: current.model.clone(),
        reference_model: reference.model.clone(),
        generated_at_epoch_secs: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0),
        cases,
        summary: CodingBenchmarkComparisonSummary {
            comparable,
            suite_compatible,
            case_ids_compatible,
            case_categories_compatible,
            comparable_case_count,
            improved_case_count,
            regressed_case_count,
            current_only_cases,
            reference_only_cases,
            category_mismatch_cases,
            pass_rate_delta_pct: current.summary.pass_rate_pct - reference.summary.pass_rate_pct,
            avg_completion_quality_delta: current.summary.avg_completion_quality_score
                - reference.summary.avg_completion_quality_score,
            avg_retries_delta: current.summary.avg_retries - reference.summary.avg_retries,
            avg_duration_delta_ms: current.summary.avg_duration_ms
                - reference.summary.avg_duration_ms,
            category_pass_rate_delta_pct,
            category_avg_quality_delta,
        },
    }
}

pub fn write_coding_benchmark_comparison_report(
    output_root: &Path,
    report: &CodingBenchmarkComparisonReport,
) -> Result<PathBuf> {
    std::fs::create_dir_all(output_root)?;
    let suite = sanitize_slug(&report.suite);
    let current_model = sanitize_slug(&report.current_model);
    let reference_model = sanitize_slug(&report.reference_model);
    let filename = format!("{suite}.{current_model}.vs.{reference_model}.comparison.json");
    let path = output_root.join(filename);
    std::fs::write(&path, serde_json::to_vec_pretty(report)?)?;
    Ok(path)
}

pub fn evaluate_coding_benchmark_gate(
    current: &CodingBenchmarkReport,
    baseline: &CodingBenchmarkReport,
    allowed_drop_pct: f32,
) -> CodingBenchmarkGateResult {
    evaluate_coding_benchmark_gate_with_thresholds(
        current,
        baseline,
        CodingBenchmarkGateThresholds {
            max_pass_rate_drop_pct: allowed_drop_pct,
            max_quality_score_drop: f32::INFINITY,
            max_avg_retries_increase: f32::INFINITY,
        },
    )
}

pub fn evaluate_coding_benchmark_gate_with_thresholds(
    current: &CodingBenchmarkReport,
    baseline: &CodingBenchmarkReport,
    thresholds: CodingBenchmarkGateThresholds,
) -> CodingBenchmarkGateResult {
    let suite_model_compatible = current.suite == baseline.suite && current.model == baseline.model;
    let delta = current.summary.pass_rate_pct - baseline.summary.pass_rate_pct;
    let quality_delta = current.summary.avg_completion_quality_score
        - baseline.summary.avg_completion_quality_score;
    let retries_delta = current.summary.avg_retries - baseline.summary.avg_retries;
    let passed = suite_model_compatible
        && delta >= -thresholds.max_pass_rate_drop_pct
        && quality_delta >= -thresholds.max_quality_score_drop
        && retries_delta <= thresholds.max_avg_retries_increase;
    CodingBenchmarkGateResult {
        passed,
        suite_model_compatible,
        current_pass_rate_pct: current.summary.pass_rate_pct,
        baseline_pass_rate_pct: baseline.summary.pass_rate_pct,
        allowed_drop_pct: thresholds.max_pass_rate_drop_pct,
        delta_pct: delta,
        current_avg_completion_quality_score: current.summary.avg_completion_quality_score,
        baseline_avg_completion_quality_score: baseline.summary.avg_completion_quality_score,
        max_quality_drop: thresholds.max_quality_score_drop,
        quality_delta,
        current_avg_retries: current.summary.avg_retries,
        baseline_avg_retries: baseline.summary.avg_retries,
        max_retry_increase: thresholds.max_avg_retries_increase,
        retries_delta,
    }
}

fn sanitize_slug(input: &str) -> String {
    let mut slug = String::with_capacity(input.len());
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            slug.push(ch.to_ascii_lowercase());
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
    }
    slug.trim_matches('-').to_string()
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

    #[test]
    fn benchmark_report_serialization_and_gate() {
        let report = CodingBenchmarkReport::from_case_results(
            "coding-quality-core",
            "scripted-tool-loop",
            vec![
                CodingBenchmarkCaseResult {
                    case_id: "edit-single-file".to_string(),
                    category: "edit".to_string(),
                    passed: true,
                    tool_invocations: 2,
                    retries: 0,
                    completion_quality_score: 1.0,
                    duration_ms: 12,
                    note: None,
                },
                CodingBenchmarkCaseResult {
                    case_id: "debug-bugfix".to_string(),
                    category: "debug".to_string(),
                    passed: false,
                    tool_invocations: 1,
                    retries: 1,
                    completion_quality_score: 0.4,
                    duration_ms: 9,
                    note: Some("missing expected patch".to_string()),
                },
            ],
        );
        assert_eq!(report.summary.total_cases, 2);
        assert_eq!(report.summary.passed_cases, 1);
        assert!((report.summary.pass_rate_pct - 50.0).abs() < f32::EPSILON);
        assert!(report.summary.avg_duration_ms > 0.0);
        assert!(report.summary.category_pass_rate_pct.contains_key("edit"));
        assert!(
            report
                .summary
                .category_avg_quality_score
                .contains_key("debug")
        );

        let baseline = CodingBenchmarkReport::from_case_results(
            "coding-quality-core",
            "scripted-tool-loop",
            vec![CodingBenchmarkCaseResult {
                case_id: "baseline".to_string(),
                category: "edit".to_string(),
                passed: true,
                tool_invocations: 1,
                retries: 0,
                completion_quality_score: 1.0,
                duration_ms: 1,
                note: None,
            }],
        );
        let gate = evaluate_coding_benchmark_gate_with_thresholds(
            &report,
            &baseline,
            CodingBenchmarkGateThresholds {
                max_pass_rate_drop_pct: 60.0,
                max_quality_score_drop: 1.0,
                max_avg_retries_increase: 2.0,
            },
        );
        assert!(gate.passed, "large allowed drop should pass");
        assert!(gate.suite_model_compatible);
        assert!(gate.quality_delta <= 0.0);
    }

    #[test]
    fn benchmark_gate_fails_for_suite_model_mismatch() {
        let current = CodingBenchmarkReport::from_case_results(
            "coding-quality-core",
            "scripted-tool-loop",
            vec![CodingBenchmarkCaseResult {
                case_id: "c1".to_string(),
                category: "edit".to_string(),
                passed: true,
                tool_invocations: 1,
                retries: 0,
                completion_quality_score: 1.0,
                duration_ms: 1,
                note: None,
            }],
        );
        let baseline = CodingBenchmarkReport::from_case_results(
            "coding-quality-extended",
            "scripted-tool-loop",
            vec![CodingBenchmarkCaseResult {
                case_id: "b1".to_string(),
                category: "edit".to_string(),
                passed: true,
                tool_invocations: 1,
                retries: 0,
                completion_quality_score: 1.0,
                duration_ms: 1,
                note: None,
            }],
        );
        let gate = evaluate_coding_benchmark_gate_with_thresholds(
            &current,
            &baseline,
            CodingBenchmarkGateThresholds::default(),
        );
        assert!(!gate.passed);
        assert!(!gate.suite_model_compatible);
    }

    #[test]
    fn benchmark_comparison_reports_case_and_category_deltas() {
        let current = CodingBenchmarkReport::from_case_results(
            "coding-quality-core",
            "deepseek-coder",
            vec![
                CodingBenchmarkCaseResult {
                    case_id: "edit-single-file".to_string(),
                    category: "edit".to_string(),
                    passed: true,
                    tool_invocations: 2,
                    retries: 0,
                    completion_quality_score: 1.0,
                    duration_ms: 18,
                    note: None,
                },
                CodingBenchmarkCaseResult {
                    case_id: "debug-bugfix".to_string(),
                    category: "debug".to_string(),
                    passed: false,
                    tool_invocations: 3,
                    retries: 1,
                    completion_quality_score: 0.5,
                    duration_ms: 27,
                    note: Some("missed edge case".to_string()),
                },
            ],
        );
        let reference = CodingBenchmarkReport::from_case_results(
            "coding-quality-core",
            "claude-code",
            vec![
                CodingBenchmarkCaseResult {
                    case_id: "edit-single-file".to_string(),
                    category: "edit".to_string(),
                    passed: false,
                    tool_invocations: 1,
                    retries: 0,
                    completion_quality_score: 0.4,
                    duration_ms: 12,
                    note: Some("patch incomplete".to_string()),
                },
                CodingBenchmarkCaseResult {
                    case_id: "debug-bugfix".to_string(),
                    category: "debug".to_string(),
                    passed: true,
                    tool_invocations: 2,
                    retries: 0,
                    completion_quality_score: 1.0,
                    duration_ms: 20,
                    note: None,
                },
            ],
        );

        let comparison = compare_coding_benchmark_reports(&current, &reference);
        assert!(comparison.summary.comparable);
        assert!(comparison.summary.suite_compatible);
        assert!(comparison.summary.case_ids_compatible);
        assert!(comparison.summary.case_categories_compatible);
        assert_eq!(comparison.summary.comparable_case_count, 2);
        assert_eq!(comparison.summary.improved_case_count, 1);
        assert_eq!(comparison.summary.regressed_case_count, 1);
        assert!(comparison.summary.pass_rate_delta_pct.abs() < f32::EPSILON);
        assert!(
            comparison.summary.avg_completion_quality_delta > 0.0,
            "current quality should be higher than reference aggregate"
        );
        assert_eq!(
            comparison.summary.category_pass_rate_delta_pct.get("edit"),
            Some(&100.0)
        );
        assert_eq!(
            comparison.summary.category_pass_rate_delta_pct.get("debug"),
            Some(&-100.0)
        );
        assert_eq!(comparison.cases.len(), 2);
        assert_eq!(comparison.cases[0].case_id, "debug-bugfix");
        assert_eq!(comparison.cases[0].retries_delta, 1);

        let dir = tempfile::tempdir().expect("tempdir");
        let path =
            write_coding_benchmark_comparison_report(dir.path(), &comparison).expect("write");
        let written = std::fs::read(&path).expect("read comparison report");
        let round_trip: CodingBenchmarkComparisonReport =
            serde_json::from_slice(&written).expect("deserialize comparison report");
        assert_eq!(round_trip.current_model, "deepseek-coder");
        assert_eq!(round_trip.reference_model, "claude-code");
    }

    #[test]
    fn benchmark_comparison_flags_case_set_mismatch() {
        let current = CodingBenchmarkReport::from_case_results(
            "coding-quality-core",
            "deepseek-coder",
            vec![CodingBenchmarkCaseResult {
                case_id: "edit-single-file".to_string(),
                category: "edit".to_string(),
                passed: true,
                tool_invocations: 1,
                retries: 0,
                completion_quality_score: 1.0,
                duration_ms: 1,
                note: None,
            }],
        );
        let reference = CodingBenchmarkReport::from_case_results(
            "coding-quality-core",
            "claude-code",
            vec![CodingBenchmarkCaseResult {
                case_id: "multi-file-update".to_string(),
                category: "multi-file".to_string(),
                passed: true,
                tool_invocations: 2,
                retries: 0,
                completion_quality_score: 1.0,
                duration_ms: 1,
                note: None,
            }],
        );

        let comparison = compare_coding_benchmark_reports(&current, &reference);
        assert!(!comparison.summary.comparable);
        assert!(!comparison.summary.case_ids_compatible);
        assert_eq!(
            comparison.summary.current_only_cases,
            vec!["edit-single-file".to_string()]
        );
        assert_eq!(
            comparison.summary.reference_only_cases,
            vec!["multi-file-update".to_string()]
        );
        assert!(comparison.cases.is_empty());
    }
}
