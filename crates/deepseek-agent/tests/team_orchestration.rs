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
            "complete_chat_streaming() not used in team orchestration tests"
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

#[test]
fn teammate_mode_without_lane_match_uses_core_loop_deterministically() -> Result<()> {
    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    fs::write(dir.path().join("demo.txt"), "old\n")?;
    commit_all(dir.path(), "seed")?;

    let plan = "ARCHITECT_PLAN_V1\nPLAN|Update demo\nFILE|demo.txt|replace value\nVERIFY|git status --short\nACCEPT|demo updated\nARCHITECT_PLAN_END\n".to_string();
    let diff = generate_diff(dir.path(), &[("demo.txt", "new\n")])?;
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(ScriptedLlm::new(vec![
        plan.clone(),
        diff.clone(),
        plan,
        diff,
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

    assert!(output.contains("Implemented"));
    assert_eq!(fs::read_to_string(dir.path().join("demo.txt"))?, "new\n");
    Ok(())
}
