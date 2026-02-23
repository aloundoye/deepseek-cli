use deepseek_core::{ApprovedToolCall, LintConfig, ToolCall, ToolHost};
use deepseek_tools::LocalToolHost;
use serde_json::json;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct LintResult {
    pub success: bool,
    pub fixed: u32,
    pub remaining: u32,
    pub summary: String,
}

/// Derive lint commands for the given changed files based on the lint config.
/// Returns a vec of commands to execute. If no commands match, returns empty.
pub fn derive_lint_commands(config: &LintConfig, changed_files: &[String]) -> Vec<String> {
    if !config.enabled || config.commands.is_empty() {
        return Vec::new();
    }

    let mut commands = Vec::new();
    let mut matched_keys = std::collections::HashSet::new();

    for file in changed_files {
        for (lang_key, cmd) in &config.commands {
            if matched_keys.contains(lang_key.as_str()) {
                continue;
            }
            if file_matches_language(file, lang_key) {
                commands.push(cmd.clone());
                matched_keys.insert(lang_key.as_str());
            }
        }
    }

    commands
}

/// Check if a file path matches a language key.
/// Supports language names ("rust", "python", "javascript", "typescript", "go", "java")
/// and glob-like extension patterns ("*.rs", "*.py").
fn file_matches_language(file: &str, lang_key: &str) -> bool {
    let ext = Path::new(file)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match lang_key {
        "rust" => ext == "rs",
        "python" => ext == "py" || ext == "pyi",
        "javascript" | "js" => ext == "js" || ext == "jsx" || ext == "mjs" || ext == "cjs",
        "typescript" | "ts" => ext == "ts" || ext == "tsx",
        "go" => ext == "go",
        "java" => ext == "java",
        "c" => ext == "c" || ext == "h",
        "cpp" | "c++" => ext == "cpp" || ext == "hpp" || ext == "cc" || ext == "cxx",
        pattern if pattern.starts_with("*.") => {
            let target_ext = &pattern[2..];
            ext == target_ext
        }
        _ => false,
    }
}

/// Callback type for lint-step approval gating.
type ApprovalCallback<'a> = Option<&'a mut dyn FnMut(&ToolCall) -> anyhow::Result<bool>>;

/// Run the configured lint commands and return a unified result.
pub fn run_lint(
    tool_host: &LocalToolHost,
    commands: &[String],
    timeout_seconds: u64,
    mut approval: ApprovalCallback<'_>,
) -> LintResult {
    if commands.is_empty() {
        return LintResult {
            success: true,
            fixed: 0,
            remaining: 0,
            summary: "no lint commands configured".to_string(),
        };
    }

    let mut failures = Vec::new();
    let mut total_remaining: u32 = 0;

    for command in commands {
        let call = ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd": command, "timeout": timeout_seconds}),
            requires_approval: false,
        };
        let proposal = tool_host.propose(call.clone());

        if !proposal.approved {
            let approved_by_user = if let Some(ref mut cb) = approval {
                cb(&proposal.call).unwrap_or(false)
            } else {
                false
            };
            if !approved_by_user {
                failures.push(format!("lint command denied by policy: `{}`", command));
                continue;
            }
        }

        let result = tool_host.execute(ApprovedToolCall {
            invocation_id: proposal.invocation_id,
            call: proposal.call,
        });

        let output = extract_output(&result.output);
        if command_passed(&result.output, result.success) {
            continue;
        }

        let error_count = count_lint_errors(&output);
        total_remaining += error_count;
        failures.push(format!(
            "`{}` found {} issue(s):\n{}",
            command,
            error_count,
            truncate(&output, 2500)
        ));
    }

    if failures.is_empty() {
        LintResult {
            success: true,
            fixed: 0,
            remaining: 0,
            summary: "all lint checks passed".to_string(),
        }
    } else {
        LintResult {
            success: false,
            fixed: 0,
            remaining: total_remaining,
            summary: failures.join("\n\n"),
        }
    }
}

/// Count lint errors from output. Heuristic: count non-empty lines that look
/// like error/warning markers or file:line patterns.
fn count_lint_errors(output: &str) -> u32 {
    let mut count: u32 = 0;
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Common lint output patterns: "file:line:col: error/warning"
        if trimmed.contains("error") || trimmed.contains("warning") || trimmed.contains("Error") {
            if trimmed.contains(':') {
                count += 1;
            }
        }
    }
    count.max(1) // at least 1 if the command failed
}

fn command_passed(output: &serde_json::Value, tool_success: bool) -> bool {
    if !tool_success {
        return false;
    }
    if output
        .get("timed_out")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return false;
    }
    if let Some(ok) = output.get("success").and_then(|v| v.as_bool()) {
        return ok;
    }
    if let Some(status) = output.get("status").and_then(|v| v.as_i64()) {
        return status == 0;
    }
    if let Some(code) = output.get("exit_code").and_then(|v| v.as_i64()) {
        return code == 0;
    }
    true
}

fn extract_output(value: &serde_json::Value) -> String {
    if let Some(text) = value.as_str() {
        return text.to_string();
    }
    let stderr = value
        .get("stderr")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let stdout = value
        .get("stdout")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let text = format!(
        "{}{}{}",
        stdout,
        if !stdout.is_empty() && !stderr.is_empty() {
            "\n"
        } else {
            ""
        },
        stderr
    );
    if text.trim().is_empty() {
        value.to_string()
    } else {
        text
    }
}

fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    format!("{}...(truncated)", &text[..text.floor_char_boundary(max)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeMap;

    #[test]
    fn file_matches_rust() {
        assert!(file_matches_language("src/main.rs", "rust"));
        assert!(!file_matches_language("src/main.py", "rust"));
    }

    #[test]
    fn file_matches_python() {
        assert!(file_matches_language("app.py", "python"));
        assert!(file_matches_language("types.pyi", "python"));
        assert!(!file_matches_language("app.rs", "python"));
    }

    #[test]
    fn file_matches_javascript() {
        assert!(file_matches_language("index.js", "javascript"));
        assert!(file_matches_language("comp.jsx", "javascript"));
        assert!(file_matches_language("lib.mjs", "js"));
        assert!(!file_matches_language("index.ts", "javascript"));
    }

    #[test]
    fn file_matches_typescript() {
        assert!(file_matches_language("app.ts", "typescript"));
        assert!(file_matches_language("comp.tsx", "ts"));
        assert!(!file_matches_language("app.js", "typescript"));
    }

    #[test]
    fn file_matches_go() {
        assert!(file_matches_language("main.go", "go"));
        assert!(!file_matches_language("main.rs", "go"));
    }

    #[test]
    fn file_matches_java() {
        assert!(file_matches_language("App.java", "java"));
        assert!(!file_matches_language("app.kt", "java"));
    }

    #[test]
    fn file_matches_glob_pattern() {
        assert!(file_matches_language("style.css", "*.css"));
        assert!(!file_matches_language("style.scss", "*.css"));
    }

    #[test]
    fn file_matches_cpp() {
        assert!(file_matches_language("main.cpp", "cpp"));
        assert!(file_matches_language("lib.hpp", "c++"));
        assert!(file_matches_language("impl.cc", "cpp"));
    }

    #[test]
    fn derive_commands_empty_when_disabled() {
        let config = LintConfig::default(); // enabled=false
        let cmds = derive_lint_commands(&config, &["src/main.rs".to_string()]);
        assert!(cmds.is_empty());
    }

    #[test]
    fn derive_commands_matches_rust() {
        let mut commands = BTreeMap::new();
        commands.insert("rust".to_string(), "cargo clippy --fix".to_string());
        let config = LintConfig {
            enabled: true,
            commands,
            max_iterations: 3,
            timeout_seconds: 30,
        };
        let cmds = derive_lint_commands(&config, &["src/main.rs".to_string()]);
        assert_eq!(cmds, vec!["cargo clippy --fix"]);
    }

    #[test]
    fn derive_commands_deduplicates_per_language() {
        let mut commands = BTreeMap::new();
        commands.insert("rust".to_string(), "cargo clippy --fix".to_string());
        let config = LintConfig {
            enabled: true,
            commands,
            max_iterations: 3,
            timeout_seconds: 30,
        };
        let files = vec!["src/main.rs".to_string(), "src/lib.rs".to_string()];
        let cmds = derive_lint_commands(&config, &files);
        assert_eq!(cmds.len(), 1, "should deduplicate same language");
    }

    #[test]
    fn derive_commands_multiple_languages() {
        let mut commands = BTreeMap::new();
        commands.insert("rust".to_string(), "cargo clippy".to_string());
        commands.insert("python".to_string(), "ruff check --fix".to_string());
        let config = LintConfig {
            enabled: true,
            commands,
            max_iterations: 3,
            timeout_seconds: 30,
        };
        let files = vec!["src/main.rs".to_string(), "scripts/run.py".to_string()];
        let cmds = derive_lint_commands(&config, &files);
        assert_eq!(cmds.len(), 2);
    }

    #[test]
    fn derive_commands_no_matching_files() {
        let mut commands = BTreeMap::new();
        commands.insert("rust".to_string(), "cargo clippy".to_string());
        let config = LintConfig {
            enabled: true,
            commands,
            max_iterations: 3,
            timeout_seconds: 30,
        };
        let cmds = derive_lint_commands(&config, &["readme.md".to_string()]);
        assert!(cmds.is_empty());
    }

    #[test]
    fn count_lint_errors_basic() {
        let output = "src/main.rs:10:5: error[E0001]: something\nsrc/main.rs:20:3: warning: unused\n";
        assert_eq!(count_lint_errors(output), 2);
    }

    #[test]
    fn count_lint_errors_min_one() {
        let output = "lint failed with exit code 1\n";
        // No file:line pattern but has non-empty output, returns at least 1
        assert_eq!(count_lint_errors(output), 1);
    }

    #[test]
    fn count_lint_errors_empty() {
        assert_eq!(count_lint_errors(""), 1); // min 1
    }

    #[test]
    fn command_passed_checks_status() {
        assert!(command_passed(&json!({"status": 0}), true));
        assert!(!command_passed(&json!({"status": 1}), true));
        assert!(!command_passed(&json!({"status": 0}), false));
    }

    #[test]
    fn command_passed_timeout() {
        assert!(!command_passed(
            &json!({"timed_out": true, "status": 0}),
            true
        ));
    }

    #[test]
    fn empty_commands_returns_success() {
        let config = LintConfig::default();
        let cmds = derive_lint_commands(&config, &[]);
        assert!(cmds.is_empty());
    }

    #[test]
    fn truncate_preserves_short() {
        assert_eq!(truncate("hello", 100), "hello");
    }

    #[test]
    fn truncate_clips_long() {
        let long = "a".repeat(100);
        let result = truncate(&long, 50);
        assert!(result.len() < 100);
        assert!(result.contains("(truncated)"));
    }
}
