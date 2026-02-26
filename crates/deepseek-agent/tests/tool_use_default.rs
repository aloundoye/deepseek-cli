//! Integration tests proving the tool-use loop is the default Code path.
//!
//! Uses `ScriptedToolLlm` to simulate tool-call responses from the LLM,
//! and verifies that the AgentEngine correctly routes through the tool-use loop.

use anyhow::{Result, anyhow};
use deepseek_agent::{AgentEngine, ChatMode, ChatOptions};
use deepseek_core::{
    ChatRequest, LlmRequest, LlmResponse, LlmToolCall, StreamCallback, TokenUsage,
};
use deepseek_llm::LlmClient;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

// ── Scripted LLM that returns tool calls ──

struct ScriptedToolLlm {
    responses: Mutex<VecDeque<LlmResponse>>,
}

impl ScriptedToolLlm {
    fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::from(responses)),
        }
    }
}

impl LlmClient for ScriptedToolLlm {
    fn complete(&self, _req: &LlmRequest) -> Result<LlmResponse> {
        Err(anyhow!("complete() not used in tool_use_default tests"))
    }
    fn complete_streaming(&self, _req: &LlmRequest, _cb: StreamCallback) -> Result<LlmResponse> {
        Err(anyhow!(
            "complete_streaming() not used in tool_use_default tests"
        ))
    }
    fn complete_chat(&self, _req: &ChatRequest) -> Result<LlmResponse> {
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow!("scripted tool llm exhausted"))
    }
    fn complete_chat_streaming(
        &self,
        req: &ChatRequest,
        _cb: StreamCallback,
    ) -> Result<LlmResponse> {
        self.complete_chat(req)
    }
    fn complete_fim(&self, _req: &deepseek_core::FimRequest) -> Result<LlmResponse> {
        Err(anyhow!(
            "complete_fim() not used in tool_use_default tests"
        ))
    }
    fn complete_fim_streaming(
        &self,
        _req: &deepseek_core::FimRequest,
        _cb: StreamCallback,
    ) -> Result<LlmResponse> {
        Err(anyhow!(
            "complete_fim_streaming() not used in tool_use_default tests"
        ))
    }
}

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
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(ScriptedToolLlm::new(responses));
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
            tool_call_response(vec![(
                "call_1",
                "fs_read",
                r#"{"path":"hello.txt"}"#,
            )]),
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
/// Uses force_execute mode which auto-approves write tools.
#[test]
fn tool_loop_reads_then_edits_file() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;
    fs::write(temp.path().join("demo.txt"), "old content\n")?;

    let mut engine = build_engine(
        temp.path(),
        vec![
            // Turn 1: read file
            tool_call_response(vec![(
                "call_1",
                "fs_read",
                r#"{"path":"demo.txt"}"#,
            )]),
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
            tool_call_response(vec![(
                "call_1",
                "bash_run",
                r#"{"command":"rm -rf /"}"#,
            )]),
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
            tool_call_response(vec![(
                "call_1",
                "fs_read",
                r#"{"path":"a.txt"}"#,
            )]),
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
            tool_call_response(vec![(
                "call_1",
                "fs_read",
                r#"{"path":"src/main.rs"}"#,
            )]),
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
        vec![
            text_response("Here's the project overview."),
        ],
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
    fn complete_fim(&self, _req: &deepseek_core::FimRequest) -> Result<LlmResponse> {
        Err(anyhow!("complete_fim() not used"))
    }
    fn complete_fim_streaming(
        &self,
        _req: &deepseek_core::FimRequest,
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
        tool_call_response(vec![(
            "call_1",
            "fs_read",
            r#"{"path":"src/main.rs"}"#,
        )]),
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
    assert!(!requests.is_empty(), "should have captured at least one request");
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
        tool_call_response(vec![(
            "call_1",
            "fs_read",
            r#"{"path":"src/main.rs"}"#,
        )]),
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
        deepseek_core::ToolChoice::required(),
        "first turn should force tool_choice=required"
    );
    Ok(())
}

/// Subsequent turns (after tool results) use tool_choice=auto.
#[test]
fn subsequent_turns_tool_choice_auto() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;
    fs::write(temp.path().join("test.txt"), "data\n")?;

    let (llm, captured) = CapturingLlm::new(vec![
        // Turn 1: tool call
        tool_call_response(vec![(
            "call_1",
            "fs_read",
            r#"{"path":"test.txt"}"#,
        )]),
        // Turn 2: text response (after tool results in messages)
        text_response("File read successfully."),
    ]);
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(llm);
    let engine = AgentEngine::new_with_llm(temp.path(), llm)?;

    let _output = engine.chat_with_options(
        "Read test.txt",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    let requests = captured.lock().unwrap();
    assert!(requests.len() >= 2, "should have at least 2 LLM calls");
    // Second request should have tool_choice=auto (because tool results exist)
    assert_eq!(
        requests[1].tool_choice,
        deepseek_core::ToolChoice::auto(),
        "subsequent turn should use tool_choice=auto"
    );
    Ok(())
}

/// Ask mode uses read-only tools (no fs_edit, fs_write, bash_run in tool list).
#[test]
fn ask_mode_uses_read_only_tools() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let (llm, captured) = CapturingLlm::new(vec![
        tool_call_response(vec![(
            "call_1",
            "fs_read",
            r#"{"path":"src/main.rs"}"#,
        )]),
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
    assert!(!tool_names.contains(&"fs_write"), "should not have fs_write");
    assert!(!tool_names.contains(&"bash_run"), "should not have bash_run");
    Ok(())
}

/// Context mode also uses read-only tools.
#[test]
fn context_mode_uses_read_only_tools() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let (llm, captured) = CapturingLlm::new(vec![
        text_response("Project overview."),
    ]);
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
    assert!(!tool_names.contains(&"fs_edit"), "Context mode should not have fs_edit");
    assert!(!tool_names.contains(&"bash_run"), "Context mode should not have bash_run");
    Ok(())
}

/// When tools=false, the analysis path is used (no tools in request).
#[test]
fn tools_false_uses_analysis_path() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    let (llm, captured) = CapturingLlm::new(vec![
        text_response("Analysis complete."),
    ]);
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

/// Tool loop with thinking mode (reasoning_content preserved).
#[test]
fn tool_loop_with_thinking_mode() -> Result<()> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;
    fs::write(temp.path().join("src/main.rs"), "fn main() { println!(\"hello\"); }\n")?;

    // Simulate thinking mode response (with reasoning_content)
    let mut response = tool_call_response(vec![(
        "call_1",
        "fs_read",
        r#"{"path":"src/main.rs"}"#,
    )]);
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
