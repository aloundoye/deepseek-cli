use anyhow::{Result, anyhow};
use deepseek_agent::{AgentEngine, ChatOptions};
use deepseek_core::{ChatRequest, LlmRequest, LlmResponse, LlmToolCall, StreamCallback};
use deepseek_llm::LlmClient;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;

struct ScriptedLlm {
    responses: Mutex<VecDeque<LlmResponse>>,
}

impl ScriptedLlm {
    fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
        }
    }

    fn next_response(&self) -> anyhow::Result<LlmResponse> {
        let mut guard = self
            .responses
            .lock()
            .map_err(|_| anyhow!("scripted llm mutex poisoned"))?;
        guard
            .pop_front()
            .ok_or_else(|| anyhow!("scripted llm exhausted"))
    }
}

impl LlmClient for ScriptedLlm {
    fn complete(&self, _req: &LlmRequest) -> anyhow::Result<LlmResponse> {
        Err(anyhow!("complete() not used in team orchestration tests"))
    }

    fn complete_streaming(
        &self,
        _req: &LlmRequest,
        _cb: StreamCallback,
    ) -> anyhow::Result<LlmResponse> {
        Err(anyhow!(
            "complete_streaming() not used in team orchestration tests"
        ))
    }

    fn complete_chat(&self, _req: &ChatRequest) -> anyhow::Result<LlmResponse> {
        self.next_response()
    }

    fn complete_chat_streaming(
        &self,
        _req: &ChatRequest,
        _cb: StreamCallback,
    ) -> anyhow::Result<LlmResponse> {
        Err(anyhow!(
            "complete_chat_streaming() not used in team orchestration tests"
        ))
    }

    fn complete_fim(&self, _req: &deepseek_core::FimRequest) -> anyhow::Result<LlmResponse> {
        Err(anyhow!(
            "complete_fim() not used in team orchestration tests"
        ))
    }

    fn complete_fim_streaming(
        &self,
        _req: &deepseek_core::FimRequest,
        _cb: StreamCallback,
    ) -> anyhow::Result<LlmResponse> {
        Err(anyhow!(
            "complete_fim_streaming() not used in team orchestration tests"
        ))
    }
}

fn git(workspace: &Path, args: &[&str]) -> Result<()> {
    let output = Command::new("git")
        .args(args)
        .current_dir(workspace)
        .output()?;
    if output.status.success() {
        return Ok(());
    }
    Err(anyhow!(
        "git {:?} failed: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    ))
}

fn init_git(workspace: &Path) -> Result<()> {
    git(workspace, &["init"])?;
    git(
        workspace,
        &["config", "user.email", "deepseek@example.test"],
    )?;
    git(workspace, &["config", "user.name", "DeepSeek Test"])?;
    Ok(())
}

fn commit_all(workspace: &Path, message: &str) -> Result<()> {
    git(workspace, &["add", "."])?;
    git(workspace, &["commit", "-m", message])?;
    Ok(())
}

/// Team orchestration falls back to standard tool-use loop when only one lane
/// is detected (prompt doesn't mention multiple domains).
#[test]
fn teammate_mode_single_lane_falls_back_to_tool_use_loop() -> Result<()> {
    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    fs::write(dir.path().join("demo.txt"), "old\n")?;
    commit_all(dir.path(), "seed")?;

    // The LLM returns a file planning response, then a tool-free text response.
    // Since the prompt doesn't span multiple domains, team orchestration
    // detects a single lane and falls back to the standard tool-use loop.
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(ScriptedLlm::new(vec![
        // Planning call (returns file list for lane classification)
        LlmResponse {
            text: "demo.txt".to_string(),
            finish_reason: "stop".to_string(),
            reasoning_content: String::new(),
            tool_calls: Vec::<LlmToolCall>::new(),
            usage: None,
        },
        // Tool-use loop: model responds with text (no tool calls) → loop exits
        LlmResponse {
            text: "Updated demo.txt successfully.".to_string(),
            finish_reason: "stop".to_string(),
            reasoning_content: String::new(),
            tool_calls: Vec::<LlmToolCall>::new(),
            usage: None,
        },
    ]));
    let mut engine = AgentEngine::new_with_llm(dir.path(), llm)?;
    engine.set_permission_mode("bypassPermissions");

    let output = engine.chat_with_options(
        "perform a quick core patch",
        ChatOptions {
            tools: true,
            teammate_mode: Some("on".to_string()),
            ..Default::default()
        },
    )?;

    assert!(
        output.contains("Updated") || output.contains("demo"),
        "expected tool-use loop fallback response, got: {output}"
    );
    Ok(())
}
