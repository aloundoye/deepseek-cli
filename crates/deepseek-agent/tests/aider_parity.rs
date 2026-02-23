//! Aider behavioral parity tests.
//!
//! These scenarios verify that the agent pipeline produces results comparable
//! to Aider for common coding workflows: rename, multi-file refactor, test
//! generation, bug fix, and analysis-only queries.

use anyhow::{Result, anyhow};
use deepseek_agent::{AgentEngine, ChatMode, ChatOptions};
use deepseek_core::{ChatRequest, LlmRequest, LlmResponse, LlmToolCall, StreamCallback};
use deepseek_llm::LlmClient;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;

// ── ScriptedLlm mock ──────────────────────────────────────────────────

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
        Err(anyhow!("complete() not used in parity tests"))
    }

    fn complete_streaming(
        &self,
        _req: &LlmRequest,
        _cb: StreamCallback,
    ) -> anyhow::Result<LlmResponse> {
        Err(anyhow!("complete_streaming() not used in parity tests"))
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
            "complete_chat_streaming() not used in parity tests"
        ))
    }
}

// ── Test helpers ───────────────────────────────────────────────────────

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

// ── Scenario 1: Simple function rename ────────────────────────────────

#[test]
fn parity_simple_function_rename() -> Result<()> {
    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    fs::write(dir.path().join("math.py"), "def add(a, b):\n    return a + b\n")?;
    commit_all(dir.path(), "initial")?;

    let diff = generate_diff(
        dir.path(),
        &[("math.py", "def sum_numbers(a, b):\n    return a + b\n")],
    )?;
    let plan = "ARCHITECT_PLAN_V1\nPLAN|Rename add to sum_numbers\nFILE|math.py|Rename function add → sum_numbers\nVERIFY|git status --short\nACCEPT|function renamed\nARCHITECT_PLAN_END\n".to_string();
    let engine = build_engine(dir.path(), vec![plan, diff])?;

    let output = engine.chat_with_options(
        "Rename the function 'add' to 'sum_numbers' in math.py",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    let content = fs::read_to_string(dir.path().join("math.py"))?;
    assert!(
        content.contains("sum_numbers"),
        "function should be renamed; got: {content}"
    );
    assert!(
        !content.contains("def add("),
        "old function name should be gone; got: {content}"
    );
    assert!(
        output.contains("Implemented"),
        "output should indicate completion"
    );
    Ok(())
}

// ── Scenario 2: Multi-file refactor ───────────────────────────────────

#[test]
fn parity_multi_file_refactor() -> Result<()> {
    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    fs::write(
        dir.path().join("utils.py"),
        "def helper():\n    return 42\n",
    )?;
    fs::write(
        dir.path().join("main.py"),
        "from utils import helper\n\nresult = helper()\n",
    )?;
    commit_all(dir.path(), "initial")?;

    let diff = generate_diff(
        dir.path(),
        &[
            (
                "utils.py",
                "def compute_answer():\n    return 42\n",
            ),
            (
                "main.py",
                "from utils import compute_answer\n\nresult = compute_answer()\n",
            ),
        ],
    )?;
    let plan = "ARCHITECT_PLAN_V1\nPLAN|Rename helper to compute_answer across files\nFILE|utils.py|Rename function\nFILE|main.py|Update import and call\nVERIFY|git status --short\nACCEPT|refactored\nARCHITECT_PLAN_END\n".to_string();
    let engine = build_engine(dir.path(), vec![plan, diff])?;

    engine.chat_with_options(
        "Rename helper() to compute_answer() in utils.py and update main.py",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    let utils = fs::read_to_string(dir.path().join("utils.py"))?;
    let main = fs::read_to_string(dir.path().join("main.py"))?;
    assert!(
        utils.contains("compute_answer"),
        "utils.py should have new name; got: {utils}"
    );
    assert!(
        main.contains("compute_answer"),
        "main.py should reference new name; got: {main}"
    );
    Ok(())
}

// ── Scenario 3: Test generation ───────────────────────────────────────

#[test]
fn parity_test_generation() -> Result<()> {
    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    fs::write(
        dir.path().join("calculator.py"),
        "def multiply(a, b):\n    return a * b\n",
    )?;
    commit_all(dir.path(), "initial")?;

    // The diff creates a new test file
    fs::write(dir.path().join("test_calculator.py"), "")?;
    commit_all(dir.path(), "add empty test file")?;

    let diff = generate_diff(
        dir.path(),
        &[(
            "test_calculator.py",
            "from calculator import multiply\n\ndef test_multiply():\n    assert multiply(3, 4) == 12\n    assert multiply(0, 5) == 0\n    assert multiply(-1, 3) == -3\n",
        )],
    )?;
    let plan = "ARCHITECT_PLAN_V1\nPLAN|Generate tests for calculator\nFILE|test_calculator.py|Write unit tests for multiply\nVERIFY|git status --short\nACCEPT|tests written\nARCHITECT_PLAN_END\n".to_string();
    let engine = build_engine(dir.path(), vec![plan, diff])?;

    engine.chat_with_options(
        "Write unit tests for the multiply function in calculator.py",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    let tests = fs::read_to_string(dir.path().join("test_calculator.py"))?;
    assert!(
        tests.contains("test_multiply"),
        "test file should contain test function; got: {tests}"
    );
    assert!(
        tests.contains("assert"),
        "test file should contain assertions; got: {tests}"
    );
    Ok(())
}

// ── Scenario 4: Bug fix with context ──────────────────────────────────

#[test]
fn parity_bug_fix_with_context() -> Result<()> {
    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    // Off-by-one bug: uses < instead of <=
    fs::write(
        dir.path().join("search.py"),
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    )?;
    commit_all(dir.path(), "initial")?;

    let diff = generate_diff(
        dir.path(),
        &[(
            "search.py",
            "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
        )],
    )?;
    let plan = "ARCHITECT_PLAN_V1\nPLAN|Fix off-by-one in binary search\nFILE|search.py|Change < to <= in while condition\nVERIFY|git status --short\nACCEPT|bug fixed\nARCHITECT_PLAN_END\n".to_string();
    let engine = build_engine(dir.path(), vec![plan, diff])?;

    engine.chat_with_options(
        "Fix the off-by-one bug in binary_search — it should use <= not < in the while condition",
        ChatOptions {
            tools: true,
            ..Default::default()
        },
    )?;

    let content = fs::read_to_string(dir.path().join("search.py"))?;
    assert!(
        content.contains("while lo <= hi"),
        "should fix the <= condition; got: {content}"
    );
    Ok(())
}

// ── Scenario 5: No-edit query (analysis-only path) ────────────────────

#[test]
fn parity_no_edit_query_analysis_only() -> Result<()> {
    let dir = tempfile::tempdir()?;
    init_git(dir.path())?;
    fs::write(
        dir.path().join("app.py"),
        "def main():\n    print('hello world')\n\nif __name__ == '__main__':\n    main()\n",
    )?;
    commit_all(dir.path(), "initial")?;

    // Record the file content before the query
    let before = fs::read_to_string(dir.path().join("app.py"))?;

    // The LLM returns an analysis response (no plan, no diff)
    let analysis = "## Initial Analysis\nThis is a simple Python entry-point script.\n\n## Key Findings\n- Single function `main()` that prints \"hello world\".\n- Standard `if __name__` guard.\n\n## Follow-ups\n- What specific behavior would you like to change?\n".to_string();

    let llm = Box::new(ScriptedLlm::new(vec![analysis]));
    let mut engine = AgentEngine::new_with_llm(dir.path(), llm)?;
    engine.set_permission_mode("bypassPermissions");

    let output = engine.analyze_with_options(
        "What does app.py do?",
        ChatOptions {
            mode: ChatMode::Ask,
            ..Default::default()
        },
    )?;

    // Verify no files were changed
    let after = fs::read_to_string(dir.path().join("app.py"))?;
    assert_eq!(
        before, after,
        "analysis-only query should not modify files"
    );

    // Verify we got analysis output
    assert!(
        output.contains("Initial Analysis") || output.contains("hello world"),
        "should return analysis output; got: {output}"
    );
    Ok(())
}
