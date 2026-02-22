//! ObservationPack: canonical context structure sent to R1 between tool steps.
//!
//! Keeps R1 informed about what happened without flooding its context window.

use serde::{Deserialize, Serialize};

/// Maximum lines of stderr to include in observations.
const STDERR_HEAD_LINES: usize = 15;
const STDERR_TAIL_LINES: usize = 10;
/// Maximum number of recent actions to include.
const MAX_RECENT_ACTIONS: usize = 8;
/// Maximum diff hunk lines to include.
const MAX_DIFF_LINES: usize = 40;

/// A single observed tool action and its result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRecord {
    pub tool: String,
    pub args_summary: String,
    pub success: bool,
    pub exit_code: Option<i32>,
    /// Short extract of output (first few lines).
    pub output_head: String,
    /// File references discovered in output (file:line patterns).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub refs: Vec<String>,
}

/// Classified error type for routing decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorClass {
    /// Compile / build error with clear file:line.
    CompileError,
    /// Test failure with identifiable test name.
    TestFailure,
    /// Missing file or dependency.
    MissingDependency,
    /// Lint / format violation.
    LintError,
    /// Runtime error (panic, segfault, timeout).
    RuntimeError,
    /// Permission denied or sandbox violation.
    PermissionDenied,
    /// Ambiguous or unclassifiable error.
    Ambiguous,
    /// No error — success.
    None,
}

impl ErrorClass {
    /// Whether this error is "mechanical" (a V3 recovery attempt may fix it).
    pub fn is_mechanical(&self) -> bool {
        matches!(
            self,
            Self::CompileError | Self::MissingDependency | Self::LintError
        )
    }
}

/// Test result summary.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TestSummary {
    pub total: u32,
    pub passed: u32,
    pub failed: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failed_tests: Vec<FailedTest>,
}

/// A single failed test with location info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedTest {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_line: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub stack_head: String,
}

/// Repo-level facts that don't change between steps.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RepoFacts {
    pub language: String,
    pub build_system: String,
    pub workspace_root: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relevant_paths: Vec<String>,
}

/// The full observation pack sent to R1 at each step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationPack {
    /// Step counter (1-indexed).
    pub step: u32,
    /// Recent actions taken (most recent last).
    pub actions: Vec<ActionRecord>,
    /// Stderr summary from the last failing action.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub stderr_summary: String,
    /// Classified error from the last failure.
    pub error_class: ErrorClass,
    /// Files changed since last successful verify.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub changed_files: Vec<String>,
    /// Compact diff summary (hunk headers + changed lines, truncated).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub diff_summary: String,
    /// Test results from last verification run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test_summary: Option<TestSummary>,
    /// Repo-level facts.
    pub repo: RepoFacts,
    /// What changed since the last green verification.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub since_last_verify: String,
}

/// Builder for constructing an ObservationPack incrementally.
pub struct ObservationPackBuilder {
    step: u32,
    actions: Vec<ActionRecord>,
    last_stderr: String,
    changed_files: Vec<String>,
    diff_summary: String,
    test_summary: Option<TestSummary>,
    repo: RepoFacts,
    since_last_verify: String,
}

impl ObservationPackBuilder {
    pub fn new(step: u32, repo: RepoFacts) -> Self {
        Self {
            step,
            actions: Vec::new(),
            last_stderr: String::new(),
            changed_files: Vec::new(),
            diff_summary: String::new(),
            test_summary: None,
            repo,
            since_last_verify: String::new(),
        }
    }

    /// Record a tool action result.
    pub fn add_action(&mut self, record: ActionRecord) -> &mut Self {
        if self.actions.len() >= MAX_RECENT_ACTIONS {
            self.actions.remove(0);
        }
        self.actions.push(record);
        self
    }

    /// Set stderr from the last failing action.
    pub fn set_stderr(&mut self, stderr: &str) -> &mut Self {
        self.last_stderr = truncate_stderr(stderr);
        self
    }

    /// Set the list of files changed since last verify.
    pub fn set_changed_files(&mut self, files: Vec<String>) -> &mut Self {
        self.changed_files = files;
        self
    }

    /// Set the diff summary.
    pub fn set_diff_summary(&mut self, diff: &str) -> &mut Self {
        self.diff_summary = truncate_diff(diff);
        self
    }

    /// Set test results.
    pub fn set_test_summary(&mut self, summary: TestSummary) -> &mut Self {
        self.test_summary = Some(summary);
        self
    }

    /// Set description of changes since last verify.
    pub fn set_since_last_verify(&mut self, desc: &str) -> &mut Self {
        self.since_last_verify = desc.to_string();
        self
    }

    /// Build the final ObservationPack.
    pub fn build(&self) -> ObservationPack {
        let error_class = classify_last_error(&self.actions, &self.last_stderr);
        ObservationPack {
            step: self.step,
            actions: self.actions.clone(),
            stderr_summary: self.last_stderr.clone(),
            error_class,
            changed_files: self.changed_files.clone(),
            diff_summary: self.diff_summary.clone(),
            test_summary: self.test_summary.clone(),
            repo: self.repo.clone(),
            since_last_verify: self.since_last_verify.clone(),
        }
    }
}

// ── Error classification ────────────────────────────────────────────────

/// Classify the error from the most recent failed action + stderr.
pub fn classify_last_error(actions: &[ActionRecord], stderr: &str) -> ErrorClass {
    // Find last failed action
    let last_fail = actions.iter().rev().find(|a| !a.success);
    if last_fail.is_none() && stderr.is_empty() {
        return ErrorClass::None;
    }

    let combined = if let Some(fail) = last_fail {
        format!("{}\n{}\n{}", fail.output_head, stderr, fail.tool)
    } else {
        stderr.to_string()
    };
    let lower = combined.to_ascii_lowercase();

    // Permission / sandbox
    if lower.contains("permission denied")
        || lower.contains("sandbox")
        || lower.contains("not allowed")
    {
        return ErrorClass::PermissionDenied;
    }

    // Compile errors
    if lower.contains("error[e")
        || lower.contains("error: ")
        || lower.contains("syntaxerror")
        || lower.contains("compileerror")
        || lower.contains("cannot find")
        || lower.contains("undefined reference")
        || lower.contains("no such module")
    {
        return ErrorClass::CompileError;
    }

    // Test failures
    if lower.contains("test result: failed")
        || lower.contains("failures:")
        || lower.contains("assert")
        || lower.contains("test failed")
        || lower.contains("pytest")
        || lower.contains("jest")
    {
        return ErrorClass::TestFailure;
    }

    // Missing dependency
    if lower.contains("not found")
        || lower.contains("no such file")
        || lower.contains("module not found")
        || lower.contains("package")
            && (lower.contains("missing") || lower.contains("not installed"))
    {
        return ErrorClass::MissingDependency;
    }

    // Lint errors
    if lower.contains("clippy")
        || lower.contains("lint")
        || lower.contains("fmt")
        || lower.contains("formatting")
        || lower.contains("eslint")
        || lower.contains("ruff")
    {
        return ErrorClass::LintError;
    }

    // Runtime errors
    if lower.contains("panic")
        || lower.contains("segfault")
        || lower.contains("timeout")
        || lower.contains("killed")
        || lower.contains("signal")
    {
        return ErrorClass::RuntimeError;
    }

    ErrorClass::Ambiguous
}

/// Extract file:line references from output text.
pub fn extract_file_refs(text: &str) -> Vec<String> {
    let mut refs = Vec::new();
    for line in text.lines() {
        // Rust-style: --> src/lib.rs:42:10
        if let Some(arrow_pos) = line.find("--> ") {
            let rest = &line[arrow_pos + 4..];
            if let Some(ref_str) = rest.split_whitespace().next()
                && ref_str.contains(':')
                && !ref_str.starts_with("http")
            {
                refs.push(ref_str.to_string());
            }
        }
        // Generic: path.rs:42 or path.rs:42:10
        for word in line.split_whitespace() {
            let cleaned = word.trim_matches(|c: char| c == ',' || c == ')' || c == '(');
            if let Some((path, rest)) = cleaned.split_once(':')
                && (path.ends_with(".rs")
                    || path.ends_with(".py")
                    || path.ends_with(".ts")
                    || path.ends_with(".js")
                    || path.ends_with(".go"))
                && rest.chars().next().is_some_and(|c| c.is_ascii_digit())
                && !refs.contains(&cleaned.to_string())
            {
                refs.push(cleaned.to_string());
            }
        }
    }
    refs.truncate(10);
    refs
}

/// Build a compact args summary for an action record.
pub fn summarize_args(tool: &str, args: &serde_json::Value) -> String {
    match tool {
        "fs.read" | "fs.write" | "fs.edit" => args
            .get("file_path")
            .or_else(|| args.get("path"))
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string(),
        "bash.run" => {
            let cmd = args.get("cmd").and_then(|v| v.as_str()).unwrap_or("?");
            if cmd.len() > 80 {
                format!("{}...", &cmd[..cmd.floor_char_boundary(80)])
            } else {
                cmd.to_string()
            }
        }
        "fs.grep" => {
            let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("?");
            let glob = args.get("glob").and_then(|v| v.as_str()).unwrap_or("*");
            format!("{pattern} in {glob}")
        }
        "fs.glob" => args
            .get("pattern")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string(),
        _ => {
            let s = serde_json::to_string(args).unwrap_or_default();
            if s.len() > 80 {
                format!("{}...", &s[..s.floor_char_boundary(80)])
            } else {
                s
            }
        }
    }
}

// ── Serialization for R1 context ────────────────────────────────────────

impl ObservationPack {
    /// Serialize to compact text for inclusion in R1 system/user prompt.
    pub fn to_r1_context(&self) -> String {
        let mut out = String::with_capacity(2048);
        out.push_str(&format!("## Observation (step {})\n\n", self.step));

        // Recent actions
        if !self.actions.is_empty() {
            out.push_str("### Recent actions\n");
            for a in &self.actions {
                let status = if a.success { "OK" } else { "FAIL" };
                out.push_str(&format!(
                    "- [{status}] {tool}({args})",
                    tool = a.tool,
                    args = a.args_summary
                ));
                if let Some(code) = a.exit_code {
                    out.push_str(&format!(" exit={code}"));
                }
                out.push('\n');
                if !a.output_head.is_empty() {
                    for line in a.output_head.lines().take(3) {
                        out.push_str(&format!("  > {line}\n"));
                    }
                }
                if !a.refs.is_empty() {
                    out.push_str(&format!("  refs: {}\n", a.refs.join(", ")));
                }
            }
            out.push('\n');
        }

        // Error info
        if self.error_class != ErrorClass::None {
            out.push_str(&format!(
                "### Error classification: {:?}\n",
                self.error_class
            ));
            if !self.stderr_summary.is_empty() {
                out.push_str("```\n");
                out.push_str(&self.stderr_summary);
                out.push_str("\n```\n\n");
            }
        }

        // Changed files
        if !self.changed_files.is_empty() {
            out.push_str(&format!(
                "### Changed files ({})\n",
                self.changed_files.len()
            ));
            for f in &self.changed_files {
                out.push_str(&format!("- {f}\n"));
            }
            out.push('\n');
        }

        // Diff summary
        if !self.diff_summary.is_empty() {
            out.push_str("### Diff summary\n```\n");
            out.push_str(&self.diff_summary);
            out.push_str("\n```\n\n");
        }

        // Test summary
        if let Some(ref ts) = self.test_summary {
            out.push_str(&format!(
                "### Tests: {}/{} passed, {} failed\n",
                ts.passed, ts.total, ts.failed
            ));
            for ft in &ts.failed_tests {
                out.push_str(&format!("- FAIL: {}", ft.name));
                if let Some(ref fl) = ft.file_line {
                    out.push_str(&format!(" at {fl}"));
                }
                out.push('\n');
                if !ft.stack_head.is_empty() {
                    out.push_str(&format!("  {}\n", ft.stack_head));
                }
            }
            out.push('\n');
        }

        // Since last verify
        if !self.since_last_verify.is_empty() {
            out.push_str(&format!(
                "### Since last verify\n{}\n\n",
                self.since_last_verify
            ));
        }

        // Repo facts
        out.push_str(&format!(
            "### Repo: {} ({})\n",
            self.repo.build_system, self.repo.language
        ));

        out
    }
}

// ── Internal helpers ────────────────────────────────────────────────────

/// Truncate stderr to head + tail lines.
fn truncate_stderr(stderr: &str) -> String {
    let lines: Vec<&str> = stderr.lines().collect();
    if lines.len() <= STDERR_HEAD_LINES + STDERR_TAIL_LINES {
        return stderr.to_string();
    }
    let mut out = lines[..STDERR_HEAD_LINES].join("\n");
    out.push_str(&format!(
        "\n... ({} lines omitted) ...\n",
        lines.len() - STDERR_HEAD_LINES - STDERR_TAIL_LINES
    ));
    out.push_str(&lines[lines.len() - STDERR_TAIL_LINES..].join("\n"));
    out
}

/// Truncate diff to max lines, keeping hunk headers.
fn truncate_diff(diff: &str) -> String {
    let lines: Vec<&str> = diff.lines().collect();
    if lines.len() <= MAX_DIFF_LINES {
        return diff.to_string();
    }
    let mut out: Vec<&str> = Vec::with_capacity(MAX_DIFF_LINES + 2);
    let mut count = 0;
    for line in &lines {
        if count >= MAX_DIFF_LINES {
            out.push("... (diff truncated)");
            break;
        }
        // Always include file headers and hunk headers
        if line.starts_with("---")
            || line.starts_with("+++")
            || line.starts_with("@@")
            || line.starts_with("diff ")
        {
            out.push(line);
        } else {
            out.push(line);
            count += 1;
        }
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_classification_compile() {
        let actions = vec![ActionRecord {
            tool: "bash.run".into(),
            args_summary: "cargo build".into(),
            success: false,
            exit_code: Some(101),
            output_head: "error[E0308]: mismatched types\n  --> src/lib.rs:42:10".into(),
            refs: vec!["src/lib.rs:42:10".into()],
        }];
        assert_eq!(classify_last_error(&actions, ""), ErrorClass::CompileError);
    }

    #[test]
    fn error_classification_test_failure() {
        let actions = vec![ActionRecord {
            tool: "bash.run".into(),
            args_summary: "cargo test".into(),
            success: false,
            exit_code: Some(101),
            output_head: "test result: FAILED. 10 passed; 2 failed".into(),
            refs: vec![],
        }];
        assert_eq!(classify_last_error(&actions, ""), ErrorClass::TestFailure);
    }

    #[test]
    fn error_classification_none_on_success() {
        let actions = vec![ActionRecord {
            tool: "bash.run".into(),
            args_summary: "cargo test".into(),
            success: true,
            exit_code: Some(0),
            output_head: "test result: ok".into(),
            refs: vec![],
        }];
        assert_eq!(classify_last_error(&actions, ""), ErrorClass::None);
    }

    #[test]
    fn error_classification_ambiguous() {
        let actions = vec![ActionRecord {
            tool: "bash.run".into(),
            args_summary: "some-cmd".into(),
            success: false,
            exit_code: Some(1),
            output_head: "something went wrong".into(),
            refs: vec![],
        }];
        assert_eq!(classify_last_error(&actions, ""), ErrorClass::Ambiguous);
    }

    #[test]
    fn mechanical_error_check() {
        assert!(ErrorClass::CompileError.is_mechanical());
        assert!(ErrorClass::LintError.is_mechanical());
        assert!(ErrorClass::MissingDependency.is_mechanical());
        assert!(!ErrorClass::TestFailure.is_mechanical());
        assert!(!ErrorClass::Ambiguous.is_mechanical());
        assert!(!ErrorClass::RuntimeError.is_mechanical());
    }

    #[test]
    fn extract_file_refs_from_rust_output() {
        let output =
            "error[E0308]: mismatched types\n  --> src/lib.rs:42:10\n  --> src/main.rs:5:3";
        let refs = extract_file_refs(output);
        assert_eq!(refs, vec!["src/lib.rs:42:10", "src/main.rs:5:3"]);
    }

    #[test]
    fn observation_pack_serialization() {
        let repo = RepoFacts {
            language: "rust".into(),
            build_system: "cargo".into(),
            workspace_root: "/workspace".into(),
            relevant_paths: vec![],
        };
        let mut builder = ObservationPackBuilder::new(1, repo);
        builder.add_action(ActionRecord {
            tool: "fs.read".into(),
            args_summary: "src/lib.rs".into(),
            success: true,
            exit_code: None,
            output_head: "fn main() {}".into(),
            refs: vec![],
        });
        let pack = builder.build();
        let text = pack.to_r1_context();
        assert!(text.contains("step 1"));
        assert!(text.contains("fs.read"));
        assert!(text.contains("OK"));
    }

    #[test]
    fn stderr_truncation() {
        let long_stderr: String = (0..100)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let truncated = truncate_stderr(&long_stderr);
        assert!(truncated.contains("line 0"));
        assert!(truncated.contains("lines omitted"));
        assert!(truncated.contains("line 99"));
    }

    #[test]
    fn builder_limits_actions() {
        let repo = RepoFacts::default();
        let mut builder = ObservationPackBuilder::new(1, repo);
        for i in 0..20 {
            builder.add_action(ActionRecord {
                tool: format!("tool_{i}"),
                args_summary: String::new(),
                success: true,
                exit_code: None,
                output_head: String::new(),
                refs: vec![],
            });
        }
        let pack = builder.build();
        assert_eq!(pack.actions.len(), MAX_RECENT_ACTIONS);
        assert_eq!(pack.actions.last().unwrap().tool, "tool_19");
    }
}
