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
    responses: Mutex<VecDeque<String>>,
}

impl ScriptedLlm {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
        }
    }

    fn next_response(&self) -> anyhow::Result<String> {
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
        Err(anyhow!(
            "complete() not used in architect/editor loop tests"
        ))
    }

    fn complete_streaming(
        &self,
        _req: &LlmRequest,
        _cb: StreamCallback,
    ) -> anyhow::Result<LlmResponse> {
        Err(anyhow!(
            "complete_streaming() not used in architect/editor loop tests"
        ))
    }

    fn complete_chat(&self, _req: &ChatRequest) -> anyhow::Result<LlmResponse> {
        Ok(LlmResponse {
            text: self.next_response()?,
            finish_reason: "stop".to_string(),
            reasoning_content: String::new(),
            tool_calls: Vec::<LlmToolCall>::new(),
        })
    }

    fn complete_chat_streaming(
        &self,
        _req: &ChatRequest,
        _cb: StreamCallback,
    ) -> anyhow::Result<LlmResponse> {
        Err(anyhow!(
            "complete_chat_streaming() not used in architect/editor loop tests"
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

fn generate_diff(workspace: &Path, updates: &[(&str, &str)]) -> Result<String> {
    let mut originals = Vec::new();
    for (path, new_content) in updates {
        let full = workspace.join(path);
        let original = fs::read_to_string(&full)?;
        originals.push((full.clone(), original));
        fs::write(&full, new_content)?;
    }

    let output = Command::new("git")
        .arg("diff")
        .current_dir(workspace)
        .output()?;

    for (path, original) in originals {
        fs::write(path, original)?;
    }

    if !output.status.success() {
        return Err(anyhow!(
            "git diff failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn build_engine(workspace: &Path, llm_responses: Vec<String>) -> Result<AgentEngine> {
    let llm = Box::new(ScriptedLlm::new(llm_responses));
    let mut engine = AgentEngine::new_with_llm(workspace, llm)?;
    engine.set_permission_mode("bypassPermissions");
    Ok(engine)
}

#[test]
fn single_file_edit_succeeds() -> Result<()> {
    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    fs::write(dir.path().join("demo.txt"), "old\n")?;
    commit_all(dir.path(), "seed")?;

    let good_diff = generate_diff(dir.path(), &[("demo.txt", "new\n")])?;
    let plan = "ARCHITECT_PLAN_V1\nPLAN|Update demo file\nFILE|demo.txt|Replace old with new\nVERIFY|git status --short\nACCEPT|demo updated\nARCHITECT_PLAN_END\n".to_string();
    let engine = build_engine(dir.path(), vec![plan, good_diff])?;

    let output = engine.chat_with_options(
        "Update demo.txt",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(output.contains("Implemented"));
    assert_eq!(fs::read_to_string(dir.path().join("demo.txt"))?, "new\n");
    Ok(())
}

#[test]
fn multi_file_edit_succeeds() -> Result<()> {
    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    fs::write(dir.path().join("a.txt"), "a0\n")?;
    fs::write(dir.path().join("b.txt"), "b0\n")?;
    commit_all(dir.path(), "seed")?;

    let good_diff = generate_diff(dir.path(), &[("a.txt", "a1\n"), ("b.txt", "b1\n")])?;
    let plan = "ARCHITECT_PLAN_V1\nPLAN|Update two files\nFILE|a.txt|Edit a\nFILE|b.txt|Edit b\nVERIFY|git status --short\nACCEPT|both updated\nARCHITECT_PLAN_END\n".to_string();
    let engine = build_engine(dir.path(), vec![plan, good_diff])?;

    let _ = engine.chat_with_options(
        "Update both files",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert_eq!(fs::read_to_string(dir.path().join("a.txt"))?, "a1\n");
    assert_eq!(fs::read_to_string(dir.path().join("b.txt"))?, "b1\n");
    Ok(())
}

#[test]
fn patch_fails_to_apply_then_recovers() -> Result<()> {
    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    fs::write(dir.path().join("demo.txt"), "old\n")?;
    commit_all(dir.path(), "seed")?;

    let good_diff = generate_diff(dir.path(), &[("demo.txt", "new\n")])?;
    let bad_diff = "diff --git a/demo.txt b/demo.txt\n--- a/demo.txt\n+++ b/demo.txt\n@@ -1 +1 @@\n-missing\n+new\n".to_string();
    let plan_one = "ARCHITECT_PLAN_V1\nPLAN|Update demo\nFILE|demo.txt|change value\nVERIFY|git status --short\nACCEPT|done\nARCHITECT_PLAN_END\n".to_string();
    let plan_two = "ARCHITECT_PLAN_V1\nPLAN|Retry with correct context\nFILE|demo.txt|fix patch context\nVERIFY|git status --short\nACCEPT|done\nARCHITECT_PLAN_END\n".to_string();

    let engine = build_engine(dir.path(), vec![plan_one, bad_diff, plan_two, good_diff])?;

    let _ = engine.chat_with_options(
        "Fix demo",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert_eq!(fs::read_to_string(dir.path().join("demo.txt"))?, "new\n");
    Ok(())
}

#[test]
fn tests_fail_then_recovers() -> Result<()> {
    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    fs::write(dir.path().join("demo.txt"), "old\n")?;
    commit_all(dir.path(), "seed")?;

    let first_diff = generate_diff(dir.path(), &[("demo.txt", "interim\n")])?;
    let second_diff =
        "diff --git a/demo.txt b/demo.txt\n--- a/demo.txt\n+++ b/demo.txt\n@@ -1 +1 @@\n-interim\n+final\n"
            .to_string();
    let plan_one = "ARCHITECT_PLAN_V1\nPLAN|First attempt\nFILE|demo.txt|set interim value\nVERIFY|cargo test -q\nACCEPT|passes\nARCHITECT_PLAN_END\n".to_string();
    let plan_two = "ARCHITECT_PLAN_V1\nPLAN|Second attempt\nFILE|demo.txt|set final value\nVERIFY|git status --short\nACCEPT|passes\nARCHITECT_PLAN_END\n".to_string();

    let engine = build_engine(
        dir.path(),
        vec![plan_one, first_diff, plan_two, second_diff],
    )?;

    let _ = engine.chat_with_options(
        "Fix demo and verify",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert_eq!(fs::read_to_string(dir.path().join("demo.txt"))?, "final\n");
    Ok(())
}
