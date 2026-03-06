use anyhow::{Result, anyhow};
use codingbuddy_agent::{AgentEngine, ChatOptions};
use codingbuddy_core::{LlmResponse, LlmToolCall};
use codingbuddy_llm::LlmClient;
use codingbuddy_testkit::ScriptedLlm;
use std::fs;
use std::path::Path;
use std::process::Command;

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
            compatibility: None,
        },
        // Tool-use loop: model responds with text (no tool calls) → loop exits
        LlmResponse {
            text: "Task completed successfully.".to_string(),
            finish_reason: "stop".to_string(),
            reasoning_content: String::new(),
            tool_calls: Vec::<LlmToolCall>::new(),
            usage: None,
            compatibility: None,
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
        output.contains("Task completed") || output.contains("successfully"),
        "expected tool-use loop fallback response, got: {output}"
    );
    Ok(())
}
