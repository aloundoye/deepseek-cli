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
            usage: None,
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
    let retry_diff =
        "diff --git a/demo.txt b/demo.txt\n--- a/demo.txt\n+++ b/demo.txt\n@@ -1 +1 @@\n-interim\n+interim_retry\n"
            .to_string();
    let final_diff =
        "diff --git a/demo.txt b/demo.txt\n--- a/demo.txt\n+++ b/demo.txt\n@@ -1 +1 @@\n-interim\n+final\n"
            .to_string();
    let plan_one = "ARCHITECT_PLAN_V1\nPLAN|First attempt\nFILE|demo.txt|set interim value\nVERIFY|cargo test -q\nACCEPT|passes\nARCHITECT_PLAN_END\n".to_string();
    let plan_two = "ARCHITECT_PLAN_V1\nPLAN|Second attempt\nFILE|demo.txt|set final value\nVERIFY|git status --short\nACCEPT|passes\nARCHITECT_PLAN_END\n".to_string();

    let engine = build_engine(
        dir.path(),
        vec![plan_one, first_diff, retry_diff, plan_two, final_diff],
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

/// Golden cassette #5: Editor requests NEED_CONTEXT, context is merged,
/// and the subsequent editor call emits a successful diff.
#[test]
fn need_context_round_trip_succeeds() -> Result<()> {
    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    fs::write(dir.path().join("app.txt"), "line_one\nline_two\nline_three\n")?;
    commit_all(dir.path(), "seed")?;

    let good_diff = generate_diff(dir.path(), &[("app.txt", "line_one\nline_two_edited\nline_three\n")])?;

    // Architect declares the file
    let plan = "ARCHITECT_PLAN_V1\nPLAN|Edit app.txt middle line\nFILE|app.txt|Update line two\nVERIFY|git status --short\nACCEPT|line two changed\nARCHITECT_PLAN_END\n".to_string();

    // Editor first responds with NEED_CONTEXT for the whole file
    let need_context = "NEED_CONTEXT|app.txt:1-3".to_string();

    // After context is provided, editor responds with a valid diff
    let engine = build_engine(dir.path(), vec![plan, need_context, good_diff])?;

    let output = engine.chat_with_options(
        "Update middle line of app.txt",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(output.contains("Implemented"));
    let content = fs::read_to_string(dir.path().join("app.txt"))?;
    assert!(content.contains("line_two_edited"));
    Ok(())
}

/// Golden cassette #6: Architect returns NO_EDIT, loop exits early without
/// entering editor/apply/verify. Stream events are captured to verify the
/// early-exit path.
#[test]
fn no_edit_response_exits_early() -> Result<()> {
    use deepseek_core::StreamChunk;
    use std::sync::Arc;

    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    fs::write(dir.path().join("demo.txt"), "content\n")?;
    commit_all(dir.path(), "seed")?;

    let plan = "ARCHITECT_PLAN_V1\nPLAN|Explain the situation\nNO_EDIT|true|Code is already correct, no changes needed\nARCHITECT_PLAN_END\n".to_string();
    let llm = Box::new(ScriptedLlm::new(vec![plan]));
    let mut engine = AgentEngine::new_with_llm(dir.path(), llm)?;
    engine.set_permission_mode("bypassPermissions");

    let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    engine.set_stream_callback(Arc::new(move |chunk: StreamChunk| {
        let label = match &chunk {
            StreamChunk::ArchitectStarted { .. } => "ArchitectStarted".to_string(),
            StreamChunk::ArchitectCompleted { no_edit, .. } => {
                format!("ArchitectCompleted(no_edit={})", no_edit)
            }
            StreamChunk::EditorStarted { .. } => "EditorStarted".to_string(),
            StreamChunk::Done => "Done".to_string(),
            _ => return,
        };
        if let Ok(mut guard) = events_clone.lock() {
            guard.push(label);
        }
    }));

    let output = engine.chat_with_options(
        "Check if demo.txt needs changes",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(output.contains("already correct"));
    // File unchanged
    assert_eq!(
        fs::read_to_string(dir.path().join("demo.txt"))?,
        "content\n"
    );

    let captured = events.lock().unwrap();
    assert!(captured.contains(&"ArchitectStarted".to_string()));
    assert!(captured.contains(&"ArchitectCompleted(no_edit=true)".to_string()));
    assert!(captured.contains(&"Done".to_string()));
    // Editor should never have started
    assert!(
        !captured.contains(&"EditorStarted".to_string()),
        "editor should not start on NO_EDIT path"
    );
    Ok(())
}

/// Golden cassette #7: Lint auto-fix loop runs between Apply and Verify.
/// When lint is enabled and configured, the LintStarted/LintCompleted events
/// are emitted during the pipeline.
#[test]
fn lint_loop_runs_when_configured() -> Result<()> {
    use deepseek_core::StreamChunk;
    use std::sync::Arc;

    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    fs::write(dir.path().join("demo.rs"), "fn main() {}\n")?;
    commit_all(dir.path(), "seed")?;

    let good_diff = generate_diff(dir.path(), &[("demo.rs", "fn main() { println!(\"hello\"); }\n")])?;
    let plan = "ARCHITECT_PLAN_V1\nPLAN|Update demo.rs\nFILE|demo.rs|Add hello print\nVERIFY|git status --short\nACCEPT|updated\nARCHITECT_PLAN_END\n".to_string();

    let llm = Box::new(ScriptedLlm::new(vec![plan, good_diff]));
    let mut engine = AgentEngine::new_with_llm(dir.path(), llm)?;
    engine.set_permission_mode("bypassPermissions");
    // Enable lint with a command that will pass (git status always succeeds)
    engine.set_lint_command("rust", "git status --short");

    let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    engine.set_stream_callback(Arc::new(move |chunk: StreamChunk| {
        let label = match &chunk {
            StreamChunk::LintStarted { iteration, .. } => {
                format!("LintStarted(iter={})", iteration)
            }
            StreamChunk::LintCompleted {
                iteration, success, ..
            } => format!("LintCompleted(iter={},success={})", iteration, success),
            StreamChunk::ArchitectStarted { .. } => "ArchitectStarted".to_string(),
            StreamChunk::Done => "Done".to_string(),
            _ => return,
        };
        if let Ok(mut guard) = events_clone.lock() {
            guard.push(label);
        }
    }));

    let output = engine.chat_with_options(
        "Add hello to demo.rs",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(output.contains("Implemented"));
    let captured = events.lock().unwrap();
    assert!(
        captured.contains(&"LintStarted(iter=1)".to_string()),
        "lint should have started; events: {:?}",
        *captured
    );
    assert!(
        captured.contains(&"LintCompleted(iter=1,success=true)".to_string()),
        "lint should have completed successfully; events: {:?}",
        *captured
    );
    Ok(())
}

/// Golden cassette #8: Lint is skipped when not enabled.
#[test]
fn lint_skipped_when_disabled() -> Result<()> {
    use deepseek_core::StreamChunk;
    use std::sync::Arc;

    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    fs::write(dir.path().join("demo.txt"), "old\n")?;
    commit_all(dir.path(), "seed")?;

    let good_diff = generate_diff(dir.path(), &[("demo.txt", "new\n")])?;
    let plan = "ARCHITECT_PLAN_V1\nPLAN|Update demo file\nFILE|demo.txt|Replace old with new\nVERIFY|git status --short\nACCEPT|demo updated\nARCHITECT_PLAN_END\n".to_string();

    let llm = Box::new(ScriptedLlm::new(vec![plan, good_diff]));
    let mut engine = AgentEngine::new_with_llm(dir.path(), llm)?;
    engine.set_permission_mode("bypassPermissions");
    // Do NOT enable lint

    let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    engine.set_stream_callback(Arc::new(move |chunk: StreamChunk| {
        let label = match &chunk {
            StreamChunk::LintStarted { .. } => "LintStarted".to_string(),
            StreamChunk::LintCompleted { .. } => "LintCompleted".to_string(),
            _ => return,
        };
        if let Ok(mut guard) = events_clone.lock() {
            guard.push(label);
        }
    }));

    let _ = engine.chat_with_options(
        "Update demo.txt",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    let captured = events.lock().unwrap();
    assert!(
        !captured.contains(&"LintStarted".to_string()),
        "lint should NOT have started when disabled; events: {:?}",
        *captured
    );
    assert!(
        !captured.contains(&"LintCompleted".to_string()),
        "lint should NOT have completed when disabled; events: {:?}",
        *captured
    );
    Ok(())
}

/// Golden cassette #9: Commit proposal asks user and commits when accepted.
#[test]
fn commit_proposal_accepted() -> Result<()> {
    use deepseek_core::{StreamChunk, UserQuestion};
    use std::sync::Arc;

    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    fs::write(dir.path().join("demo.txt"), "old\n")?;
    commit_all(dir.path(), "seed")?;

    let good_diff = generate_diff(dir.path(), &[("demo.txt", "new\n")])?;
    let plan = "ARCHITECT_PLAN_V1\nPLAN|Update demo file\nFILE|demo.txt|Replace old with new\nVERIFY|git status --short\nACCEPT|demo updated\nARCHITECT_PLAN_END\n".to_string();
    let llm = Box::new(ScriptedLlm::new(vec![plan, good_diff]));
    let mut engine = AgentEngine::new_with_llm(dir.path(), llm)?;
    engine.set_permission_mode("bypassPermissions");

    // Set user question handler that always accepts
    engine.set_user_question_handler(Arc::new(|_q: UserQuestion| Some("yes".to_string())));

    let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    engine.set_stream_callback(Arc::new(move |chunk: StreamChunk| {
        let label = match &chunk {
            StreamChunk::CommitCompleted { sha, message } => {
                format!("CommitCompleted(sha={},msg={})", sha, message)
            }
            StreamChunk::CommitSkipped => "CommitSkipped".to_string(),
            StreamChunk::CommitProposal { .. } => "CommitProposal".to_string(),
            _ => return,
        };
        if let Ok(mut guard) = events_clone.lock() {
            guard.push(label);
        }
    }));

    let output = engine.chat_with_options(
        "Update demo.txt",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    assert!(output.contains("Implemented"));
    let captured = events.lock().unwrap();
    assert!(
        captured.iter().any(|e| e.starts_with("CommitCompleted")),
        "commit should have completed; events: {:?}",
        *captured
    );
    assert!(
        !captured.contains(&"CommitSkipped".to_string()),
        "commit should not have been skipped; events: {:?}",
        *captured
    );

    // Verify git log shows the commit
    let log = Command::new("git")
        .args(["log", "--oneline", "-1"])
        .current_dir(dir.path())
        .output()?;
    let log_str = String::from_utf8_lossy(&log.stdout);
    assert!(
        log_str.contains("deepseek:"),
        "commit message should contain template; got: {}",
        log_str
    );
    Ok(())
}

/// Golden cassette #10: Commit proposal skipped when user declines.
#[test]
fn commit_proposal_declined() -> Result<()> {
    use deepseek_core::{StreamChunk, UserQuestion};
    use std::sync::Arc;

    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    fs::write(dir.path().join("demo.txt"), "old\n")?;
    commit_all(dir.path(), "seed")?;

    let good_diff = generate_diff(dir.path(), &[("demo.txt", "new\n")])?;
    let plan = "ARCHITECT_PLAN_V1\nPLAN|Update demo file\nFILE|demo.txt|Replace old with new\nVERIFY|git status --short\nACCEPT|demo updated\nARCHITECT_PLAN_END\n".to_string();
    let llm = Box::new(ScriptedLlm::new(vec![plan, good_diff]));
    let mut engine = AgentEngine::new_with_llm(dir.path(), llm)?;
    engine.set_permission_mode("bypassPermissions");

    // Set user question handler that always declines
    engine.set_user_question_handler(Arc::new(|_q: UserQuestion| Some("no".to_string())));

    let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    engine.set_stream_callback(Arc::new(move |chunk: StreamChunk| {
        let label = match &chunk {
            StreamChunk::CommitCompleted { .. } => "CommitCompleted".to_string(),
            StreamChunk::CommitSkipped => "CommitSkipped".to_string(),
            _ => return,
        };
        if let Ok(mut guard) = events_clone.lock() {
            guard.push(label);
        }
    }));

    let _ = engine.chat_with_options(
        "Update demo.txt",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    let captured = events.lock().unwrap();
    assert!(
        captured.contains(&"CommitSkipped".to_string()),
        "commit should have been skipped; events: {:?}",
        *captured
    );
    assert!(
        !captured.contains(&"CommitCompleted".to_string()),
        "commit should NOT have completed; events: {:?}",
        *captured
    );

    // File should still be modified (not committed)
    assert_eq!(fs::read_to_string(dir.path().join("demo.txt"))?, "new\n");
    Ok(())
}

/// Golden cassette #11: Commit proposal with custom message.
#[test]
fn commit_proposal_custom_message() -> Result<()> {
    use deepseek_core::{StreamChunk, UserQuestion};
    use std::sync::Arc;

    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    fs::write(dir.path().join("demo.txt"), "old\n")?;
    commit_all(dir.path(), "seed")?;

    let good_diff = generate_diff(dir.path(), &[("demo.txt", "new\n")])?;
    let plan = "ARCHITECT_PLAN_V1\nPLAN|Update demo file\nFILE|demo.txt|Replace old with new\nVERIFY|git status --short\nACCEPT|demo updated\nARCHITECT_PLAN_END\n".to_string();
    let llm = Box::new(ScriptedLlm::new(vec![plan, good_diff]));
    let mut engine = AgentEngine::new_with_llm(dir.path(), llm)?;
    engine.set_permission_mode("bypassPermissions");

    // User provides a custom commit message
    engine.set_user_question_handler(Arc::new(|_q: UserQuestion| {
        Some("fix: update demo file content".to_string())
    }));

    let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    engine.set_stream_callback(Arc::new(move |chunk: StreamChunk| {
        let label = match &chunk {
            StreamChunk::CommitCompleted { message, .. } => {
                format!("CommitCompleted(msg={})", message)
            }
            _ => return,
        };
        if let Ok(mut guard) = events_clone.lock() {
            guard.push(label);
        }
    }));

    let _ = engine.chat_with_options(
        "Update demo.txt",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    let captured = events.lock().unwrap();
    assert!(
        captured.iter().any(|e| e.contains("fix: update demo file content")),
        "commit should use custom message; events: {:?}",
        *captured
    );

    // Verify git log shows custom message
    let log = Command::new("git")
        .args(["log", "--oneline", "-1"])
        .current_dir(dir.path())
        .output()?;
    let log_str = String::from_utf8_lossy(&log.stdout);
    assert!(
        log_str.contains("fix: update demo file content"),
        "git log should contain custom message; got: {}",
        log_str
    );
    Ok(())
}
