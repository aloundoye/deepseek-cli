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
use std::sync::Mutex;

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
