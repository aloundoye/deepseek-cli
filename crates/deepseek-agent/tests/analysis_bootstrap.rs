use anyhow::{Result, anyhow};
use deepseek_agent::{AgentEngine, ChatMode, ChatOptions};
use deepseek_core::{
    ChatMessage, ChatRequest, LlmRequest, LlmResponse, LlmToolCall, StreamCallback,
};
use deepseek_llm::LlmClient;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct CaptureState {
    responses: VecDeque<String>,
    chat_requests: Vec<ChatRequest>,
}

struct CapturingLlm {
    state: Arc<Mutex<CaptureState>>,
}

impl CapturingLlm {
    fn new(state: Arc<Mutex<CaptureState>>) -> Self {
        Self { state }
    }

    fn next_response(&self) -> anyhow::Result<String> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| anyhow!("responses mutex poisoned"))?;
        guard
            .responses
            .pop_front()
            .ok_or_else(|| anyhow!("scripted response queue exhausted"))
    }
}

impl LlmClient for CapturingLlm {
    fn complete(&self, _req: &LlmRequest) -> anyhow::Result<LlmResponse> {
        Err(anyhow!("complete() not used in analysis bootstrap tests"))
    }

    fn complete_streaming(
        &self,
        _req: &LlmRequest,
        _cb: StreamCallback,
    ) -> anyhow::Result<LlmResponse> {
        Err(anyhow!(
            "complete_streaming() not used in analysis bootstrap tests"
        ))
    }

    fn complete_chat(&self, req: &ChatRequest) -> anyhow::Result<LlmResponse> {
        if let Ok(mut guard) = self.state.lock() {
            guard.chat_requests.push(req.clone());
        }
        Ok(LlmResponse {
            text: self.next_response()?,
            finish_reason: "stop".to_string(),
            reasoning_content: String::new(),
            tool_calls: Vec::<LlmToolCall>::new(),
            usage: None,
        })
    }

    fn complete_chat_streaming(
        &self,
        _req: &ChatRequest,
        _cb: StreamCallback,
    ) -> anyhow::Result<LlmResponse> {
        Err(anyhow!(
            "complete_chat_streaming() not used in analysis bootstrap tests"
        ))
    }

    fn complete_fim(&self, _req: &deepseek_core::FimRequest) -> anyhow::Result<LlmResponse> {
        Err(anyhow!("complete_fim() not used in analysis bootstrap tests"))
    }

    fn complete_fim_streaming(
        &self,
        _req: &deepseek_core::FimRequest,
        _cb: StreamCallback,
    ) -> anyhow::Result<LlmResponse> {
        Err(anyhow!("complete_fim_streaming() not used in analysis bootstrap tests"))
    }
}

fn build_workspace(path: &Path) -> Result<()> {
    fs::create_dir_all(path.join("src"))?;
    fs::create_dir_all(path.join("tests"))?;
    fs::write(
        path.join("README.md"),
        "# Demo\n\nTODO: tighten architecture checks\n",
    )?;
    fs::write(
        path.join("Cargo.toml"),
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\n\n[dependencies]\nanyhow = \"1\"\nserde = \"1\"\n",
    )?;
    fs::write(
        path.join("src/lib.rs"),
        "// FIXME: improve\npub fn demo() {}\n",
    )?;
    fs::write(path.join("tests/smoke_test.rs"), "#[test]\nfn smoke() {}\n")?;
    let init = std::process::Command::new("git")
        .arg("init")
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

fn first_user_message(request: &ChatRequest) -> String {
    request
        .messages
        .iter()
        .find_map(|msg| match msg {
            ChatMessage::User { content } => Some(content.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

#[test]
fn repoish_prompt_injects_auto_context_bootstrap_packet() -> Result<()> {
    let temp = tempfile::tempdir()?;
    build_workspace(temp.path())?;

    let state = Arc::new(Mutex::new(CaptureState {
        responses: VecDeque::from(vec![
            "Initial Analysis\nRepo looks healthy.\nKey Findings\n- one\nFollow-ups\n- Any priority area?"
                .to_string(),
        ]),
        chat_requests: Vec::new(),
    }));
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(CapturingLlm::new(state.clone()));
    let engine = AgentEngine::new_with_llm(temp.path(), llm)?;

    let output = engine.chat_with_options(
        "analyze this project",
        ChatOptions {
            tools: false,
            mode: ChatMode::Ask,
            ..Default::default()
        },
    )?;

    assert!(output.contains("Initial Analysis"));

    let guard = state.lock().map_err(|_| anyhow!("state poisoned"))?;
    assert_eq!(guard.chat_requests.len(), 1);
    let user = first_user_message(&guard.chat_requests[0]);
    assert!(user.contains("AUTO_CONTEXT_BOOTSTRAP_V1"));
    assert!(user.contains("ROOT_TREE_SNAPSHOT"));
    assert!(user.contains("README_EXCERPT"));
    Ok(())
}

#[test]
fn vague_codebase_prompt_runs_audit_and_repairs_followup_budget() -> Result<()> {
    let temp = tempfile::tempdir()?;
    build_workspace(temp.path())?;

    let state = Arc::new(Mutex::new(CaptureState {
        responses: VecDeque::from(vec![
            "Initial Analysis\nA\nKey Findings\nB\nFollow-ups\n- q1\n- q2\n- q3".to_string(),
            "Initial Analysis\nRepo baseline checked.\nKey Findings\n- TODO/FIXME entries exist\nFollow-ups\n- Should I prioritize dependency hygiene or test hardening first?".to_string(),
        ]),
        chat_requests: Vec::new(),
    }));

    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(CapturingLlm::new(state.clone()));
    let engine = AgentEngine::new_with_llm(temp.path(), llm)?;

    let output = engine.chat_with_options(
        "check the codebase",
        ChatOptions {
            tools: false,
            mode: ChatMode::Ask,
            ..Default::default()
        },
    )?;

    assert!(output.contains("Initial Analysis"));
    assert!(output.contains("Follow-ups"));

    let guard = state.lock().map_err(|_| anyhow!("state poisoned"))?;
    assert_eq!(guard.chat_requests.len(), 2, "expected one repair retry");

    let first_user = first_user_message(&guard.chat_requests[0]);
    assert!(first_user.contains("BASELINE_AUDIT"));
    assert!(first_user.contains("TODO_FIXME"));
    assert!(first_user.contains("DEPENDENCIES"));

    let second = &guard.chat_requests[1];
    let repair_user = second
        .messages
        .iter()
        .filter_map(|message| match message {
            ChatMessage::User { content } => Some(content.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(repair_user.contains("Repair required"));
    assert!(repair_user.contains("exactly 1 focused follow-up"));
    Ok(())
}

#[test]
fn repoish_prompt_outside_repo_returns_explicit_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    fs::write(temp.path().join("notes.txt"), "scratch")?;

    let state = Arc::new(Mutex::new(CaptureState {
        responses: VecDeque::from(vec!["unused".to_string()]),
        chat_requests: Vec::new(),
    }));
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(CapturingLlm::new(state.clone()));
    let engine = AgentEngine::new_with_llm(temp.path(), llm)?;

    let err = engine
        .chat_with_options(
            "analyze this project",
            ChatOptions {
                tools: false,
                mode: ChatMode::Ask,
                ..Default::default()
            },
        )
        .expect_err("expected explicit no-repo error");
    assert_eq!(
        err.to_string(),
        "No repository detected. Run from project root or pass --repo <path>."
    );

    let guard = state.lock().map_err(|_| anyhow!("state poisoned"))?;
    assert!(
        guard.chat_requests.is_empty(),
        "analysis should fail before LLM call when no repo is detected"
    );
    Ok(())
}
