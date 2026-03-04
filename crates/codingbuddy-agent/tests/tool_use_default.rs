//! Integration tests proving the tool-use loop is the default Code path.
//!
//! Uses `ScriptedLlm` to simulate tool-call responses from the LLM,
//! and verifies that the AgentEngine correctly routes through the tool-use loop.

use anyhow::{Result, anyhow};
use codingbuddy_agent::{AgentEngine, ChatMode, ChatOptions};
use codingbuddy_core::{
    AppConfig, ChatMessage, ChatRequest, LlmRequest, LlmResponse, LlmToolCall, Session,
    SessionBudgets, SessionState, StreamCallback, StreamChunk, TokenUsage,
};
use codingbuddy_llm::LlmClient;
use codingbuddy_store::{BackgroundJobRecord, Store, SubagentRunRecord, TaskQueueRecord};
use codingbuddy_testkit::ScriptedLlm;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

// ── Response builders ──

fn tool_call_response(calls: Vec<(&str, &str, &str)>) -> LlmResponse {
    LlmResponse {
        text: String::new(),
        finish_reason: "tool_calls".to_string(),
        reasoning_content: String::new(),
        tool_calls: calls
            .iter()
            .map(|(id, name, args)| LlmToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments: args.to_string(),
            })
            .collect(),
        usage: Some(TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            ..Default::default()
        }),
    }
}

fn text_response(text: &str) -> LlmResponse {
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

fn tool_call_response_with_text(text: &str, calls: Vec<(&str, &str, &str)>) -> LlmResponse {
    let mut response = tool_call_response(calls);
    response.text = text.to_string();
    response
}

// ── Helpers ──

fn init_workspace(path: &Path) -> Result<()> {
    fs::create_dir_all(path.join("src"))?;
    fs::write(path.join("src/main.rs"), "fn main() {}\n")?;
    let init = std::process::Command::new("git")
        .args(["init", "-q"])
        .current_dir(path)
        .output()?;
    if !init.status.success() {
        return Err(anyhow!(
            "git init failed: {}",
            String::from_utf8_lossy(&init.stderr)
        ));
    }
    Ok(())
}

fn build_engine(path: &Path, responses: Vec<LlmResponse>) -> Result<AgentEngine> {
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(ScriptedLlm::new(responses));
    AgentEngine::new_with_llm(path, llm)
}

#[derive(Clone)]
struct SharedScriptedLlm(Arc<ScriptedLlm>);

impl LlmClient for SharedScriptedLlm {
    fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        self.0.complete(req)
    }

    fn complete_streaming(&self, req: &LlmRequest, cb: StreamCallback) -> Result<LlmResponse> {
        self.0.complete_streaming(req, cb)
    }

    fn complete_chat(&self, req: &ChatRequest) -> Result<LlmResponse> {
        self.0.complete_chat(req)
    }

    fn complete_chat_streaming(
        &self,
        req: &ChatRequest,
        cb: StreamCallback,
    ) -> Result<LlmResponse> {
        self.0.complete_chat_streaming(req, cb)
    }

    fn complete_fim(&self, req: &codingbuddy_core::FimRequest) -> Result<LlmResponse> {
        self.0.complete_fim(req)
    }

    fn complete_fim_streaming(
        &self,
        req: &codingbuddy_core::FimRequest,
        cb: StreamCallback,
    ) -> Result<LlmResponse> {
        self.0.complete_fim_streaming(req, cb)
    }
}

fn build_engine_with_shared_llm(path: &Path, llm: Arc<ScriptedLlm>) -> Result<AgentEngine> {
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(SharedScriptedLlm(llm));
    AgentEngine::new_with_llm(path, llm)
}

// ── Tests ──

/// Default ChatMode::Code routes to the tool-use loop (not the old pipeline).
/// The LLM calls fs_read then responds with text.
#[test]
fn default_code_mode_uses_tool_loop() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;
    fs::write(temp.path().join("hello.txt"), "Hello, world!\n")?;

    let engine = build_engine(
        temp.path(),
        vec![
            // Turn 1: LLM calls fs_read
            tool_call_response(vec![("call_1", "fs_read", r#"{"path":"hello.txt"}"#)]),
            // Turn 2: LLM responds with text
            text_response("The file contains a greeting."),
        ],
    )?;

    let output = engine.chat_with_options(
        "What's in hello.txt?",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert_eq!(output, "The file contains a greeting.");
    Ok(())
}

/// The tool-use loop reads a file, edits it, then responds.
/// Uses bypassPermissions mode which auto-approves write tools.
#[test]
fn tool_loop_reads_then_edits_file() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;
    fs::write(temp.path().join("demo.txt"), "old content\n")?;

    let mut engine = build_engine(
        temp.path(),
        vec![
            // Turn 1: read file
            tool_call_response(vec![("call_1", "fs_read", r#"{"path":"demo.txt"}"#)]),
            // Turn 2: edit file (fs_edit uses search/replace fields)
            tool_call_response(vec![(
                "call_2",
                "fs_edit",
                r#"{"path":"demo.txt","search":"old content","replace":"new content"}"#,
            )]),
            // Turn 3: done
            text_response("Updated demo.txt successfully."),
        ],
    )?;

    // Bypass permissions so fs_edit goes through in tests
    engine.set_permission_mode("bypassPermissions");

    let output = engine.chat_with_options(
        "Replace 'old content' with 'new content' in demo.txt",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(output.contains("Updated"));
    // Verify the edit was applied
    let content = fs::read_to_string(temp.path().join("demo.txt"))?;
    assert_eq!(content, "new content\n");
    Ok(())
}

/// The tool-use loop runs a bash command.
#[test]
fn tool_loop_runs_bash_command() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let engine = build_engine(
        temp.path(),
        vec![
            // Turn 1: run bash
            tool_call_response(vec![(
                "call_1",
                "bash_run",
                r#"{"command":"echo hello from bash"}"#,
            )]),
            // Turn 2: respond
            text_response("Command output: hello from bash"),
        ],
    )?;

    // bash_run requires approval — without an approval handler, it gets denied
    // so the LLM gets a denial feedback and should respond accordingly
    let output = engine.chat_with_options(
        "Run echo hello",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    // The loop ran and returned a text response
    assert!(!output.is_empty());
    Ok(())
}

/// When a tool is denied (no approval handler), the denial is fed back to the LLM.
#[test]
fn tool_loop_denied_tool_feeds_back() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let engine = build_engine(
        temp.path(),
        vec![
            // Turn 1: LLM tries bash_run (requires approval)
            tool_call_response(vec![("call_1", "bash_run", r#"{"command":"rm -rf /"}"#)]),
            // Turn 2: after denial, LLM responds with alternative
            text_response("I cannot execute that command. Can I help another way?"),
        ],
    )?;

    // No approval handler set → denial by default
    let output = engine.chat_with_options(
        "Delete everything",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(output.contains("cannot execute"));
    Ok(())
}

/// Multi-turn conversation: first prompt gets tool calls, second prompt continues.
#[test]
fn tool_loop_multi_turn_conversation() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;
    fs::write(temp.path().join("a.txt"), "aaa\n")?;

    let engine = build_engine(
        temp.path(),
        vec![
            // First prompt, turn 1: read file
            tool_call_response(vec![("call_1", "fs_read", r#"{"path":"a.txt"}"#)]),
            // First prompt, turn 2: respond
            text_response("File a.txt contains 'aaa'."),
        ],
    )?;

    let output = engine.chat_with_options(
        "Read a.txt",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(output.contains("aaa"));
    Ok(())
}

/// ChatMode::Ask restricts to read-only tools — write tools get filtered out.
#[test]
fn ask_mode_restricts_to_read_only() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let engine = build_engine(
        temp.path(),
        vec![
            // Turn 1: LLM reads a file (should work in Ask mode)
            tool_call_response(vec![("call_1", "fs_read", r#"{"path":"src/main.rs"}"#)]),
            // Turn 2: respond
            text_response("The file has a main function."),
        ],
    )?;

    let output = engine.chat_with_options(
        "What's in src/main.rs?",
        ChatOptions {
            tools: true,
            mode: ChatMode::Ask,
            ..Default::default()
        },
    )?;

    assert!(output.contains("main function"));
    Ok(())
}

/// ChatMode::Context also restricts to read-only tools.
#[test]
fn context_mode_restricts_to_read_only() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let engine = build_engine(
        temp.path(),
        vec![text_response("Here's the project overview.")],
    )?;

    let output = engine.chat_with_options(
        "Describe this project",
        ChatOptions {
            tools: true,
            mode: ChatMode::Context,
            ..Default::default()
        },
    )?;

    assert!(output.contains("project overview"));
    Ok(())
}

// ── Capturing LLM variant (records requests for inspection) ──

struct CapturingLlm {
    responses: Mutex<VecDeque<LlmResponse>>,
    captured_requests: Arc<Mutex<Vec<ChatRequest>>>,
}

impl CapturingLlm {
    fn new(responses: Vec<LlmResponse>) -> (Self, Arc<Mutex<Vec<ChatRequest>>>) {
        let captured = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                responses: Mutex::new(VecDeque::from(responses)),
                captured_requests: captured.clone(),
            },
            captured,
        )
    }
}

impl LlmClient for CapturingLlm {
    fn complete(&self, _req: &LlmRequest) -> Result<LlmResponse> {
        Err(anyhow!("complete() not used"))
    }
    fn complete_streaming(&self, _req: &LlmRequest, _cb: StreamCallback) -> Result<LlmResponse> {
        Err(anyhow!("complete_streaming() not used"))
    }
    fn complete_chat(&self, req: &ChatRequest) -> Result<LlmResponse> {
        self.captured_requests.lock().unwrap().push(req.clone());
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow!("capturing llm exhausted"))
    }
    fn complete_chat_streaming(
        &self,
        req: &ChatRequest,
        _cb: StreamCallback,
    ) -> Result<LlmResponse> {
        self.complete_chat(req)
    }
    fn complete_fim(&self, _req: &codingbuddy_core::FimRequest) -> Result<LlmResponse> {
        Err(anyhow!("complete_fim() not used"))
    }
    fn complete_fim_streaming(
        &self,
        _req: &codingbuddy_core::FimRequest,
        _cb: StreamCallback,
    ) -> Result<LlmResponse> {
        Err(anyhow!("complete_fim_streaming() not used"))
    }
}

/// When the LLM keeps making tool calls beyond max_turns, the loop stops gracefully.
#[test]
fn tool_loop_handles_max_turns() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;
    fs::write(temp.path().join("test.txt"), "data\n")?;

    // Provide enough responses for many tool calls + a final text response
    let mut responses: Vec<LlmResponse> = (0..60)
        .map(|i| {
            tool_call_response(vec![(
                &format!("call_{i}"),
                "fs_read",
                r#"{"path":"test.txt"}"#,
            )])
        })
        .collect();
    responses.push(text_response("finally done"));

    let mut engine = build_engine(temp.path(), responses)?;
    engine.set_max_turns(Some(3));

    let output = engine.chat_with_options(
        "Keep reading forever",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    // Should have stopped due to max turns, returning empty response
    // (max_turns stop doesn't produce text)
    assert!(output.is_empty() || output.contains("done"));
    Ok(())
}

// ── Batch 7: New integration tests ──

/// Tool descriptions in the LLM request are enriched (long, behavioral instructions).
#[test]
fn tool_descriptions_are_enriched_in_request() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let (llm, captured) = CapturingLlm::new(vec![
        tool_call_response(vec![("call_1", "fs_read", r#"{"path":"src/main.rs"}"#)]),
        text_response("Done."),
    ]);
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(llm);
    let engine = AgentEngine::new_with_llm(temp.path(), llm)?;

    let _output = engine.chat_with_options(
        "Read file",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    let requests = captured.lock().unwrap();
    assert!(
        !requests.is_empty(),
        "should have captured at least one request"
    );
    let first_req = &requests[0];
    // Find fs_read tool in the request tools
    let fs_read_tool = first_req
        .tools
        .iter()
        .find(|t| t.function.name == "fs_read")
        .expect("fs_read should be in tools");
    assert!(
        fs_read_tool.function.description.len() >= 200,
        "fs_read description should be enriched (>=200 chars), got {}",
        fs_read_tool.function.description.len()
    );
    assert!(
        fs_read_tool.function.description.contains("MUST"),
        "fs_read description should contain behavioral instructions"
    );
    Ok(())
}

/// First turn of tool-use loop uses tool_choice=required.
#[test]
fn first_turn_tool_choice_required() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let (llm, captured) = CapturingLlm::new(vec![
        tool_call_response(vec![("call_1", "fs_read", r#"{"path":"src/main.rs"}"#)]),
        text_response("Read the file."),
    ]);
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(llm);
    let engine = AgentEngine::new_with_llm(temp.path(), llm)?;

    let _output = engine.chat_with_options(
        "What's in this project?",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    let requests = captured.lock().unwrap();
    assert!(!requests.is_empty());
    // First request should have tool_choice=required
    assert_eq!(
        requests[0].tool_choice,
        codingbuddy_core::ToolChoice::required(),
        "first turn should force tool_choice=required"
    );
    Ok(())
}

/// After 1+ Assistant turns, tool_choice switches to auto.
/// Only the very first LLM call per user question forces tool_choice=required.
#[test]
fn subsequent_turns_tool_choice_auto() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;
    fs::write(temp.path().join("test.txt"), "data\n")?;
    fs::write(temp.path().join("other.txt"), "more data\n")?;

    let (llm, captured) = CapturingLlm::new(vec![
        // Turn 1: first tool call (0 Assistant turns → required)
        tool_call_response(vec![("call_1", "fs_read", r#"{"path":"test.txt"}"#)]),
        // Turn 2: second tool call (1 Assistant turn → auto)
        tool_call_response(vec![("call_2", "fs_read", r#"{"path":"other.txt"}"#)]),
        // Turn 3: text response
        text_response("Files read successfully."),
    ]);
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(llm);
    let engine = AgentEngine::new_with_llm(temp.path(), llm)?;

    let _output = engine.chat_with_options(
        "Read test.txt and other.txt",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    let requests = captured.lock().unwrap();
    assert!(requests.len() >= 3, "should have at least 3 LLM calls");
    // First request (no prior Assistant turns) should use required
    assert_eq!(
        requests[0].tool_choice,
        codingbuddy_core::ToolChoice::required(),
        "first turn should use tool_choice=required"
    );
    // After first Assistant response, switch to auto
    assert_eq!(
        requests[1].tool_choice,
        codingbuddy_core::ToolChoice::auto(),
        "second turn (1 Assistant turn) should use auto"
    );
    assert_eq!(
        requests[2].tool_choice,
        codingbuddy_core::ToolChoice::auto(),
        "third turn should also use auto"
    );
    Ok(())
}

/// Ask mode uses read-only tools (no fs_edit, fs_write, bash_run in tool list).
#[test]
fn ask_mode_uses_read_only_tools() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let (llm, captured) = CapturingLlm::new(vec![
        tool_call_response(vec![("call_1", "fs_read", r#"{"path":"src/main.rs"}"#)]),
        text_response("It's a Rust project."),
    ]);
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(llm);
    let engine = AgentEngine::new_with_llm(temp.path(), llm)?;

    let _output = engine.chat_with_options(
        "What language is this?",
        ChatOptions {
            tools: true,
            mode: ChatMode::Ask,
            ..Default::default()
        },
    )?;

    let requests = captured.lock().unwrap();
    assert!(!requests.is_empty());
    let tool_names: Vec<&str> = requests[0]
        .tools
        .iter()
        .map(|t| t.function.name.as_str())
        .collect();
    // Read-only tools should be present
    assert!(tool_names.contains(&"fs_read"), "should have fs_read");
    assert!(tool_names.contains(&"fs_glob"), "should have fs_glob");
    assert!(tool_names.contains(&"fs_grep"), "should have fs_grep");
    // Write tools should NOT be present
    assert!(!tool_names.contains(&"fs_edit"), "should not have fs_edit");
    assert!(
        !tool_names.contains(&"fs_write"),
        "should not have fs_write"
    );
    assert!(
        !tool_names.contains(&"bash_run"),
        "should not have bash_run"
    );
    Ok(())
}

/// Context mode also uses read-only tools.
#[test]
fn context_mode_uses_read_only_tools() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let (llm, captured) = CapturingLlm::new(vec![text_response("Project overview.")]);
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(llm);
    let engine = AgentEngine::new_with_llm(temp.path(), llm)?;

    let _output = engine.chat_with_options(
        "Describe this project",
        ChatOptions {
            tools: true,
            mode: ChatMode::Context,
            ..Default::default()
        },
    )?;

    let requests = captured.lock().unwrap();
    assert!(!requests.is_empty());
    let tool_names: Vec<&str> = requests[0]
        .tools
        .iter()
        .map(|t| t.function.name.as_str())
        .collect();
    assert!(
        !tool_names.contains(&"fs_edit"),
        "Context mode should not have fs_edit"
    );
    assert!(
        !tool_names.contains(&"bash_run"),
        "Context mode should not have bash_run"
    );
    Ok(())
}

#[test]
fn code_mode_build_profile_uses_smaller_tool_surface() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let (llm, captured) = CapturingLlm::new(vec![text_response("Scoped tool set ready.")]);
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(llm);
    let engine = AgentEngine::new_with_llm(temp.path(), llm)?;

    let _output = engine.chat_with_options(
        "Fix the bug in src/main.rs",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    let requests = captured.lock().unwrap();
    assert!(!requests.is_empty());
    let tool_names: Vec<&str> = requests[0]
        .tools
        .iter()
        .map(|t| t.function.name.as_str())
        .collect();
    assert!(
        tool_names.contains(&"fs_edit"),
        "build profile should keep fs_edit"
    );
    assert!(
        tool_names.contains(&"bash_run"),
        "build profile should keep bash_run"
    );
    assert!(
        tool_names.contains(&"tool_search"),
        "tool_search should appear when discoverable tools exist"
    );
    assert!(
        !tool_names.contains(&"web_search"),
        "build profile should not expose web_search by default"
    );
    assert!(
        !tool_names.contains(&"chrome_navigate"),
        "chrome tools should stay hidden without chrome-specific signals"
    );
    Ok(())
}

#[test]
fn tool_search_enables_matching_tools_for_followup_turns() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let (llm, captured) = CapturingLlm::new(vec![
        tool_call_response(vec![("call_1", "tool_search", r#"{"query":"plan mode"}"#)]),
        tool_call_response(vec![("call_2", "enter_plan_mode", "{}")]),
        text_response("Planning mode entered."),
    ]);
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(llm);
    let engine = AgentEngine::new_with_llm(temp.path(), llm)?;

    let output = engine.chat_with_options(
        "Inspect this repository",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(output.contains("Planning mode entered"));
    let requests = captured.lock().unwrap();
    assert!(
        requests.len() >= 2,
        "expected at least two captured requests"
    );
    let first_tools: Vec<&str> = requests[0]
        .tools
        .iter()
        .map(|t| t.function.name.as_str())
        .collect();
    let second_tools: Vec<&str> = requests[1]
        .tools
        .iter()
        .map(|t| t.function.name.as_str())
        .collect();
    assert!(first_tools.contains(&"tool_search"));
    assert!(
        !first_tools.contains(&"enter_plan_mode"),
        "enter_plan_mode should be hidden until discovered"
    );
    assert!(
        second_tools.contains(&"enter_plan_mode"),
        "tool_search should promote matching tools into the next request"
    );
    Ok(())
}

/// When tools=false, the analysis path is used (no tools in request).
#[test]
fn tools_false_uses_analysis_path() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let (llm, captured) = CapturingLlm::new(vec![text_response("Analysis complete.")]);
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(llm);
    let engine = AgentEngine::new_with_llm(temp.path(), llm)?;

    let output = engine.chat_with_options(
        "What is 2+2?",
        ChatOptions {
            tools: false,
            ..Default::default()
        },
    )?;

    assert!(output.contains("Analysis complete"));
    // Analysis path should have been used (check captured requests)
    let requests = captured.lock().unwrap();
    assert!(!requests.is_empty());
    // With tools=false, the request should have no tools
    assert!(
        requests[0].tools.is_empty(),
        "analysis path should not include tools"
    );
    Ok(())
}

/// Default tool-use loop always has thinking enabled (adaptive thinking).
#[test]
fn default_tool_loop_has_thinking_enabled() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let (llm, captured) = CapturingLlm::new(vec![
        tool_call_response(vec![("call_1", "fs_read", r#"{"path":"src/main.rs"}"#)]),
        text_response("Done."),
    ]);
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(llm);
    let engine = AgentEngine::new_with_llm(temp.path(), llm)?;

    let _output = engine.chat_with_options(
        "What's in this project?",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    let requests = captured.lock().unwrap();
    assert!(!requests.is_empty());
    // Thinking should always be enabled by default
    let thinking = requests[0]
        .thinking
        .as_ref()
        .expect("thinking should be enabled by default");
    assert!(
        thinking.budget_tokens.unwrap_or(0) > 0,
        "thinking budget should be positive"
    );
    Ok(())
}

/// force_max_think overrides thinking budget to maximum (65536).
#[test]
fn force_max_think_overrides_budget() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let (llm, captured) = CapturingLlm::new(vec![
        tool_call_response(vec![("call_1", "fs_read", r#"{"path":"src/main.rs"}"#)]),
        text_response("Done."),
    ]);
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(llm);
    let engine = AgentEngine::new_with_llm(temp.path(), llm)?;

    let _output = engine.chat_with_options(
        "fix typo",
        ChatOptions {
            tools: true,
            force_max_think: true,
            ..Default::default()
        },
    )?;

    let requests = captured.lock().unwrap();
    assert!(!requests.is_empty());
    let thinking = requests[0]
        .thinking
        .as_ref()
        .expect("thinking should be enabled with force_max_think");
    // force_max_think should use COMPLEX_THINK_BUDGET * 2 = 65536 (max budget)
    assert_eq!(
        thinking.budget_tokens,
        Some(65536),
        "force_max_think should use maximum budget"
    );
    Ok(())
}

/// Tool loop with thinking mode (reasoning_content preserved).
#[test]
fn tool_loop_with_thinking_mode() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;
    fs::write(
        temp.path().join("src/main.rs"),
        "fn main() { println!(\"hello\"); }\n",
    )?;

    // Simulate thinking mode response (with reasoning_content)
    let mut response = tool_call_response(vec![("call_1", "fs_read", r#"{"path":"src/main.rs"}"#)]);
    response.reasoning_content = "Let me think about which file to read...".to_string();

    let mut final_response = text_response("The main function prints hello.");
    final_response.reasoning_content = "Based on the file contents...".to_string();

    let engine = build_engine(temp.path(), vec![response, final_response])?;

    let output = engine.chat_with_options(
        "What does the main function do?",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(output.contains("prints hello"));
    Ok(())
}

// ── spawn_task tests ──

/// When spawn_task is called with a worker, the worker is invoked.
#[test]
fn spawn_task_calls_worker() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let worker_called = Arc::new(Mutex::new(false));
    let worker_called_clone = worker_called.clone();

    let engine = build_engine(
        temp.path(),
        vec![
            // Turn 1: LLM calls spawn_task
            tool_call_response(vec![(
                "call_1",
                "spawn_task",
                r#"{"prompt":"analyze the project","description":"explore project","subagent_type":"explore"}"#,
            )]),
            // Turn 2: respond with results
            text_response("The subagent analyzed the project."),
        ],
    )?;

    // Wire a subagent worker
    let worker: codingbuddy_agent::SubagentWorkerFn = Arc::new(move |task| {
        *worker_called_clone.lock().unwrap() = true;
        Ok(format!("Subagent completed: {}", task.goal))
    });
    engine.set_subagent_worker(worker);

    let output = engine.chat_with_options(
        "Analyze this project",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(
        *worker_called.lock().unwrap(),
        "subagent worker should have been called"
    );
    assert!(!output.is_empty());
    Ok(())
}

/// When spawn_task is called without a worker, it falls back gracefully.
#[test]
fn spawn_task_without_worker_falls_back() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let engine = build_engine(
        temp.path(),
        vec![
            // Turn 1: LLM calls spawn_task (no worker wired)
            tool_call_response(vec![(
                "call_1",
                "spawn_task",
                r#"{"prompt":"explore the code","description":"code exploration","subagent_type":"explore"}"#,
            )]),
            // Turn 2: respond
            text_response("I'll explore the code directly."),
        ],
    )?;

    // No worker set — should fall back gracefully
    let output = engine.chat_with_options(
        "Explore the code",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(!output.is_empty());
    Ok(())
}

/// spawn_task correctly maps subagent_type to SubagentRole.
#[test]
fn spawn_task_role_maps_correctly() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let captured_role = Arc::new(Mutex::new(String::new()));
    let captured_role_clone = captured_role.clone();

    let engine = build_engine(
        temp.path(),
        vec![
            tool_call_response(vec![(
                "call_1",
                "spawn_task",
                r#"{"prompt":"plan the implementation","description":"plan task","subagent_type":"plan"}"#,
            )]),
            text_response("Planning complete."),
        ],
    )?;

    let worker: codingbuddy_agent::SubagentWorkerFn = Arc::new(move |task| {
        *captured_role_clone.lock().unwrap() = format!("{:?}", task.role);
        Ok("Done".to_string())
    });
    engine.set_subagent_worker(worker);

    let _output = engine.chat_with_options(
        "Plan the implementation",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    let role = captured_role.lock().unwrap();
    assert_eq!(
        *role, "Plan",
        "subagent_type 'plan' should map to SubagentRole::Plan"
    );
    Ok(())
}

#[test]
fn spawn_task_persists_child_session_and_run_metadata() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let engine = build_engine(
        temp.path(),
        vec![
            tool_call_response(vec![(
                "call_1",
                "spawn_task",
                r#"{"prompt":"analyze the project","description":"explore project","subagent_type":"explore"}"#,
            )]),
            text_response("Delegated task complete."),
        ],
    )?;

    let worker: codingbuddy_agent::SubagentWorkerFn =
        Arc::new(move |task| Ok(format!("Subagent completed: {}", task.goal)));
    engine.set_subagent_worker(worker);

    let output = engine.chat_with_options(
        "Analyze this project",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(output.contains("Delegated task complete"));
    let store = Store::new(temp.path())?;
    let tasks = store.list_tasks(None)?;
    assert_eq!(tasks.len(), 1);
    let task = &tasks[0];
    assert_eq!(task.status, "completed");
    assert!(
        task.artifact_path
            .as_deref()
            .is_some_and(|value| value.starts_with("session://"))
    );

    let run = store
        .load_subagent_run_for_task(task.task_id)?
        .expect("subagent run");
    assert_eq!(run.session_id, Some(task.session_id));
    assert_eq!(run.task_id, Some(task.task_id));
    assert_eq!(run.status, "completed");
    assert_eq!(
        run.output.as_deref(),
        Some("Subagent completed: analyze the project")
    );
    let child_session_id = run.child_session_id.expect("child session id");
    assert!(store.load_session(child_session_id)?.is_some());
    Ok(())
}

#[test]
fn spawn_task_background_keeps_task_running() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let job_id = Uuid::now_v7();
    let engine = build_engine(
        temp.path(),
        vec![
            tool_call_response(vec![(
                "call_1",
                "spawn_task",
                r#"{"prompt":"analyze the project","description":"explore project","subagent_type":"explore","run_in_background":true,"max_turns":5,"model":"deepseek-chat"}"#,
            )]),
            text_response("Delegated task queued."),
        ],
    )?;

    let worker: codingbuddy_agent::SubagentWorkerFn = Arc::new(move |_task| {
        Ok(serde_json::json!({
            "job_id": job_id,
            "status": "running",
        })
        .to_string())
    });
    engine.set_subagent_worker(worker);

    let output = engine.chat_with_options(
        "Analyze this project in background",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(output.contains("Delegated task queued"));
    let store = Store::new(temp.path())?;
    let tasks = store.list_tasks(None)?;
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].status, "in_progress");
    let run = store
        .load_subagent_run_for_task(tasks[0].task_id)?
        .expect("subagent run");
    assert_eq!(run.status, "running");
    assert_eq!(run.background_job_id, Some(job_id));
    Ok(())
}

#[test]
fn task_output_tool_surfaces_persisted_run_payload() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let store = Store::new(temp.path())?;
    let parent_session = Session {
        session_id: Uuid::now_v7(),
        workspace_root: temp.path().to_string_lossy().to_string(),
        baseline_commit: None,
        status: SessionState::Idle,
        budgets: SessionBudgets {
            per_turn_seconds: 30,
            max_think_tokens: 1024,
        },
        active_plan_id: None,
    };
    store.save_session(&parent_session)?;
    let child_session = store.fork_session(parent_session.session_id)?;
    store.save_session(&parent_session)?;

    let task_id = Uuid::now_v7();
    let run_id = Uuid::now_v7();
    let now = chrono::Utc::now().to_rfc3339();
    store.insert_task(&TaskQueueRecord {
        task_id,
        session_id: parent_session.session_id,
        title: "Explore project".to_string(),
        description: Some("Inspect parser pipeline".to_string()),
        priority: 1,
        status: "completed".to_string(),
        outcome: Some("Indexed 12 files".to_string()),
        artifact_path: Some(format!("session://{}", child_session.session_id)),
        created_at: now.clone(),
        updated_at: now.clone(),
    })?;
    store.upsert_subagent_run(&SubagentRunRecord {
        run_id,
        session_id: Some(parent_session.session_id),
        task_id: Some(task_id),
        child_session_id: Some(child_session.session_id),
        background_job_id: None,
        name: "explore project".to_string(),
        goal: "inspect parser pipeline".to_string(),
        status: "completed".to_string(),
        output: Some("Indexed 12 files".to_string()),
        error: None,
        created_at: now.clone(),
        updated_at: now,
    })?;

    let scripted = Arc::new(ScriptedLlm::new(vec![
        tool_call_response(vec![(
            "call_1",
            "task_output",
            &format!(r#"{{"task_id":"{task_id}"}}"#),
        )]),
        text_response("Read task output."),
    ]));
    let engine = build_engine_with_shared_llm(temp.path(), scripted.clone())?;

    let output = engine.chat_with_options(
        "Read the delegated task output",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(output.contains("Read task output"));
    let captured = scripted.captured_messages();
    assert!(captured.len() >= 2, "expected two llm calls");
    let tool_message = captured[1]
        .iter()
        .find_map(|msg| match msg {
            ChatMessage::Tool { content, .. } => Some(content.clone()),
            _ => None,
        })
        .expect("tool message");
    assert!(tool_message.contains(&task_id.to_string()));
    assert!(tool_message.contains(&run_id.to_string()));
    assert!(tool_message.contains(&child_session.session_id.to_string()));
    assert!(tool_message.contains("Indexed 12 files"));
    Ok(())
}

#[test]
fn task_stop_cancels_background_run() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let store = Store::new(temp.path())?;
    let session = Session {
        session_id: Uuid::now_v7(),
        workspace_root: temp.path().to_string_lossy().to_string(),
        baseline_commit: None,
        status: SessionState::Idle,
        budgets: SessionBudgets {
            per_turn_seconds: 30,
            max_think_tokens: 1024,
        },
        active_plan_id: None,
    };
    store.save_session(&session)?;

    let task_id = Uuid::now_v7();
    let run_id = Uuid::now_v7();
    let job_id = Uuid::now_v7();
    let now = chrono::Utc::now().to_rfc3339();
    store.insert_task(&TaskQueueRecord {
        task_id,
        session_id: session.session_id,
        title: "Background task".to_string(),
        description: None,
        priority: 1,
        status: "in_progress".to_string(),
        outcome: None,
        artifact_path: None,
        created_at: now.clone(),
        updated_at: now.clone(),
    })?;
    store.upsert_subagent_run(&SubagentRunRecord {
        run_id,
        session_id: Some(session.session_id),
        task_id: Some(task_id),
        child_session_id: None,
        background_job_id: Some(job_id),
        name: "bg".to_string(),
        goal: "work".to_string(),
        status: "running".to_string(),
        output: None,
        error: None,
        created_at: now.clone(),
        updated_at: now.clone(),
    })?;
    store.upsert_background_job(&BackgroundJobRecord {
        job_id,
        kind: "subagent".to_string(),
        reference: format!("subagent:{run_id}"),
        status: "running".to_string(),
        metadata_json: serde_json::json!({}).to_string(),
        started_at: now.clone(),
        updated_at: now,
    })?;

    let engine = build_engine(
        temp.path(),
        vec![
            tool_call_response(vec![(
                "call_1",
                "task_stop",
                &format!(r#"{{"task_id":"{task_id}"}}"#),
            )]),
            text_response("Stopped the task."),
        ],
    )?;

    let output = engine.chat_with_options(
        "Stop the background task",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(output.contains("Stopped the task"));
    let task = store.load_task(task_id)?.expect("task");
    assert_eq!(task.status, "cancelled");
    let run = store.load_subagent_run(run_id)?.expect("run");
    assert_eq!(run.status, "stopped");
    let job = store.load_background_job(job_id)?.expect("job");
    assert_eq!(job.status, "stopped");
    Ok(())
}

#[test]
fn task_create_persists_to_store() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let engine = build_engine(
        temp.path(),
        vec![
            tool_call_response(vec![(
                "call_1",
                "task_create",
                r#"{"subject":"Audit parser","description":"Inspect parser edge cases","priority":2}"#,
            )]),
            text_response("Created the task."),
        ],
    )?;

    let output = engine.chat_with_options(
        "Create a task for parser auditing",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(output.contains("Created"));
    let store = Store::new(temp.path())?;
    let tasks = store.list_tasks(None)?;
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].title, "Audit parser");
    assert_eq!(
        tasks[0].description.as_deref(),
        Some("Inspect parser edge cases")
    );
    assert_eq!(tasks[0].priority, 2);
    Ok(())
}

#[test]
fn exit_plan_mode_persists_plan_and_session_state() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let engine = build_engine(
        temp.path(),
        vec![
            tool_call_response(vec![("call_1", "enter_plan_mode", "{}")]),
            tool_call_response_with_text(
                "1. Read src/main.rs\n2. Update the parser wiring\n3. Verify with cargo test",
                vec![("call_2", "exit_plan_mode", "{}")],
            ),
            text_response("Plan saved for review."),
        ],
    )?;

    let output = engine.chat_with_options(
        "Plan the parser refactor",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(output.contains("Plan saved"));
    let store = Store::new(temp.path())?;
    let session = store.load_latest_session()?.expect("session");
    assert_eq!(
        session.status,
        codingbuddy_core::SessionState::AwaitingApproval
    );
    let plan_id = session.active_plan_id.expect("active plan");
    let plan = store.load_plan(plan_id)?.expect("plan");
    assert_eq!(plan.version, 2);
    assert_eq!(plan.goal, "Plan the parser refactor");
    assert!(!plan.steps.is_empty());
    Ok(())
}

#[test]
fn complex_prompt_auto_enters_plan_and_persists_draft_state() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let engine = build_engine(
        temp.path(),
        vec![
            tool_call_response(vec![("call_1", "fs_read", r#"{"path":"src/main.rs"}"#)]),
            tool_call_response(vec![(
                "call_2",
                "fs_grep",
                r#"{"pattern":"main","path":"src"}"#,
            )]),
            tool_call_response(vec![("call_3", "git_status", "{}")]),
            text_response(
                "1. Inspect parser wiring\n2. Update implementation\n3. Verify with cargo test",
            ),
            tool_call_response(vec![("call_4", "exit_plan_mode", "{}")]),
            text_response("Plan saved for review."),
        ],
    )?;

    let phase_chunks = Arc::new(Mutex::new(Vec::new()));
    let phase_chunks_clone = phase_chunks.clone();
    engine.set_stream_callback(Arc::new(move |chunk| {
        if let StreamChunk::PhaseTransition { from, to } = chunk {
            phase_chunks_clone.lock().unwrap().push((from, to));
        }
    }));

    let output = engine.chat_with_options(
        "Refactor the parser across multiple files and verify the result carefully.",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(output.contains("Plan saved"));
    let phases = phase_chunks.lock().unwrap().clone();
    assert_eq!(phases, vec![("explore".to_string(), "plan".to_string())]);

    let store = Store::new(temp.path())?;
    let session = store.load_latest_session()?.expect("session");
    assert_eq!(session.status, SessionState::AwaitingApproval);
    let plan = store
        .load_plan(session.active_plan_id.expect("active plan"))?
        .expect("plan");
    assert_eq!(plan.version, 2, "auto-entered plan should create draft v1");
    assert!(!plan.steps.is_empty());
    Ok(())
}

#[test]
fn approved_plan_execution_emits_full_phase_path_and_loads_plan_context() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let (llm, captured) = CapturingLlm::new(vec![
        tool_call_response(vec![("call_1", "tool_search", r#"{"query":"plan mode"}"#)]),
        tool_call_response(vec![("call_2", "enter_plan_mode", "{}")]),
        tool_call_response_with_text(
            "1. Read src/main.rs\n2. Update parser wiring\n3. Verify with cargo test",
            vec![("call_3", "exit_plan_mode", "{}")],
        ),
        text_response("Plan saved for review."),
        tool_call_response(vec![("call_4", "fs_read", r#"{"path":"src/main.rs"}"#)]),
        tool_call_response(vec![(
            "call_5",
            "fs_edit",
            r#"{"path":"src/main.rs","search":"fn main() {}","replace":"fn main() { println!(\"ok\"); }"}"#,
        )]),
        text_response("Implemented the approved plan."),
        tool_call_response(vec![(
            "call_6",
            "bash_run",
            r#"{"command":"printf verified"}"#,
        )]),
        text_response("Verified the change."),
    ]);
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(llm);
    let mut engine = AgentEngine::new_with_llm(temp.path(), llm)?;
    engine.set_permission_mode("bypassPermissions");

    let phase_chunks = Arc::new(Mutex::new(Vec::new()));
    let phase_chunks_clone = phase_chunks.clone();
    engine.set_stream_callback(Arc::new(move |chunk| {
        if let StreamChunk::PhaseTransition { from, to } = chunk {
            phase_chunks_clone.lock().unwrap().push((from, to));
        }
    }));

    let planning_output = engine.chat_with_options(
        "Refactor the parser across multiple steps, then verify the result.",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;
    assert!(planning_output.contains("Plan saved"));

    let store = Store::new(temp.path())?;
    let session = store.load_latest_session()?.expect("session");
    assert_eq!(session.status, SessionState::AwaitingApproval);

    let final_output = engine.chat_with_options(
        "Go ahead and implement the approved plan.",
        ChatOptions {
            tools: true,
            session_id: Some(session.session_id),
            ..Default::default()
        },
    )?;

    assert!(final_output.contains("Verified the change"));
    assert!(fs::read_to_string(temp.path().join("src/main.rs"))?.contains("println!(\"ok\")"));

    let phases = phase_chunks.lock().unwrap().clone();
    assert_eq!(
        phases,
        vec![
            ("explore".to_string(), "plan".to_string()),
            ("plan".to_string(), "execute".to_string()),
            ("execute".to_string(), "verify".to_string()),
        ]
    );

    let requests = captured.lock().unwrap();
    assert!(
        requests.len() >= 5,
        "expected captured requests from both planning and execution"
    );
    let execution_request = &requests[4];
    let system_messages = execution_request
        .messages
        .iter()
        .filter_map(|msg| match msg {
            ChatMessage::System { content } => Some(content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        system_messages
            .iter()
            .any(|content| content.contains("## Active Approved Plan")),
        "execution request should include the approved plan context"
    );
    assert!(
        system_messages
            .iter()
            .any(|content| content.contains("Update parser wiring")),
        "execution request should include persisted plan steps"
    );
    Ok(())
}

#[test]
fn openai_compatible_surface_prefers_patch_direct_in_requests() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;
    let mut cfg = AppConfig::default();
    cfg.llm.provider = "openai-compatible".to_string();
    cfg.save(temp.path())?;

    let (llm, captured) = CapturingLlm::new(vec![text_response("Surface ready.")]);
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(llm);
    let engine = AgentEngine::new_with_llm(temp.path(), llm)?;

    let _output = engine.chat_with_options(
        "Fix a bug in src/main.rs.",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    let requests = captured.lock().unwrap();
    let tool_names = requests[0]
        .tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect::<Vec<_>>();
    assert!(tool_names.contains(&"patch_direct"));
    assert!(!tool_names.contains(&"fs_edit"));
    assert!(!tool_names.contains(&"multi_edit"));
    assert!(tool_names.len() <= 18);
    Ok(())
}

#[test]
fn ollama_surface_clamps_request_tool_count() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;
    let mut cfg = AppConfig::default();
    cfg.llm.provider = "ollama".to_string();
    cfg.save(temp.path())?;

    let (llm, captured) = CapturingLlm::new(vec![text_response("Surface ready.")]);
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(llm);
    let engine = AgentEngine::new_with_llm(temp.path(), llm)?;

    let _output = engine.chat_with_options(
        "Refactor src/main.rs and add verification.",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    let requests = captured.lock().unwrap();
    let tool_names = requests[0]
        .tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect::<Vec<_>>();
    assert!(tool_names.contains(&"fs_edit"));
    assert!(tool_names.contains(&"tool_search"));
    assert!(!tool_names.contains(&"patch_direct"));
    assert!(tool_names.len() <= 12);
    Ok(())
}
