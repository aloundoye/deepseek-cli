use deepseek_core::{ApprovedToolCall, ToolCall, ToolHost};
use deepseek_tools::LocalToolHost;
use serde_json::json;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct VerifyResult {
    pub success: bool,
    pub summary: String,
}

/// Callback type for verify-step approval gating.
type ApprovalCallback<'a> = Option<&'a mut dyn FnMut(&ToolCall) -> anyhow::Result<bool>>;

pub fn derive_verify_commands(workspace: &Path) -> Vec<String> {
    if workspace.join("Cargo.toml").exists() {
        return vec!["cargo test -q".to_string()];
    }
    if workspace.join("package.json").exists() {
        return vec!["npm test --silent".to_string()];
    }
    if workspace.join("pyproject.toml").exists() || workspace.join("setup.py").exists() {
        return vec!["pytest -q".to_string()];
    }
    if workspace.join("go.mod").exists() {
        return vec!["go test ./...".to_string()];
    }
    vec!["git status --short".to_string()]
}

pub fn run_verify(
    workspace: &Path,
    tool_host: &LocalToolHost,
    commands: &[String],
    timeout_seconds: u64,
    mut approval: ApprovalCallback<'_>,
) -> VerifyResult {
    let mut failures = Vec::new();

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
                let line = format!("verification command denied by policy: `{}`", command);
                failures.push(line.clone());
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
        } else {
            let mut failure_message = format!(
                "`{}` failed:\n{}",
                command,
                truncate(&output, 2500)
            );
            
            // FailureContextPack: Append auto-extracted files matching paths from the trace
            let context_pack = build_failure_context_pack(workspace, &output);
            if !context_pack.is_empty() {
                failure_message.push_str("\n\n=== Failure Context Pack (Referenced Files) ===\n");
                failure_message.push_str(&context_pack);
                failure_message.push_str("===============================================\n");
            }

            failures.push(failure_message);
        }
    }

    if failures.is_empty() {
        VerifyResult {
            success: true,
            summary: "verification passed".to_string(),
        }
    } else {
        VerifyResult {
            success: false,
            summary: failures.join("\n\n"),
        }
    }
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

fn build_failure_context_pack(workspace: &Path, output: &str) -> String {
    use std::collections::HashSet;
    use std::fs;

    let mut found_files = HashSet::new();
    let mut pack = String::new();

    // Tokenize roughly to try finding workspace-relative files
    for token in output.split(|c: char| c.is_whitespace() || c == '"' || c == '\'' || c == '=' || c == '[' || c == ']') {
        // Strip common trailing punctuation (e.g., from Python stack trace: file.py:20)
        let clean_token = token
            .trim_end_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '/' && c != '.')
            .trim_start_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '/' && c != '.');
            
        // Look for typical file names and line numbers
        let path_str = if let Some(idx) = clean_token.rfind(':') {
            &clean_token[..idx]
        } else {
            clean_token
        };

        if path_str.is_empty() || path_str.len() > 255 {
            continue;
        }

        // Ignore things that are obviously not files (numbers, short words)
        if !path_str.contains('.') && !path_str.contains('/') {
            continue;
        }

        let full_path = workspace.join(path_str);
        if full_path.is_file() && !found_files.contains(path_str) {
            found_files.insert(path_str.to_string());
            if let Ok(content) = fs::read_to_string(&full_path) {
                let content_str = if content.len() > 8000 {
                    format!("{}...(truncated)", &content[..8000])
                } else {
                    content
                };
                pack.push_str(&format!("\n--- Context Auto-Attached: {} ---\n{}\n", path_str, content_str));
            }
        }
    }
    pack
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn derive_commands_fallback() {
        let temp = tempfile::tempdir().expect("tempdir");
        let cmds = derive_verify_commands(temp.path());
        assert!(!cmds.is_empty());
        assert!(cmds[0].contains("git status"));
    }

    #[test]
    fn derive_commands_cargo() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("Cargo.toml"), "[package]\nname=\"x\"").unwrap();
        let cmds = derive_verify_commands(temp.path());
        assert_eq!(cmds, vec!["cargo test -q"]);
    }

    #[test]
    fn derive_commands_npm() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("package.json"), "{}").unwrap();
        let cmds = derive_verify_commands(temp.path());
        assert_eq!(cmds, vec!["npm test --silent"]);
    }

    #[test]
    fn derive_commands_python_pyproject() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("pyproject.toml"), "[project]").unwrap();
        let cmds = derive_verify_commands(temp.path());
        assert_eq!(cmds, vec!["pytest -q"]);
    }

    #[test]
    fn derive_commands_python_setup() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("setup.py"), "# setup").unwrap();
        let cmds = derive_verify_commands(temp.path());
        assert_eq!(cmds, vec!["pytest -q"]);
    }

    #[test]
    fn derive_commands_go() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("go.mod"), "module x").unwrap();
        let cmds = derive_verify_commands(temp.path());
        assert_eq!(cmds, vec!["go test ./..."]);
    }

    #[test]
    fn derive_commands_priority_cargo_over_npm() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("Cargo.toml"), "").unwrap();
        std::fs::write(temp.path().join("package.json"), "").unwrap();
        let cmds = derive_verify_commands(temp.path());
        assert_eq!(cmds, vec!["cargo test -q"]);
    }

    #[test]
    fn command_passed_checks_status() {
        assert!(command_passed(&json!({"status": 0}), true));
        assert!(!command_passed(&json!({"status": 1}), true));
        assert!(!command_passed(
            &json!({"timed_out": true, "status": 0}),
            true
        ));
        assert!(!command_passed(&json!({"status": 0}), false));
    }

    #[test]
    fn command_passed_checks_exit_code() {
        assert!(command_passed(&json!({"exit_code": 0}), true));
        assert!(!command_passed(&json!({"exit_code": 1}), true));
    }

    #[test]
    fn command_passed_checks_success_bool() {
        assert!(command_passed(&json!({"success": true}), true));
        assert!(!command_passed(&json!({"success": false}), true));
    }

    #[test]
    fn command_passed_defaults_to_true_on_empty_output() {
        assert!(command_passed(&json!({}), true));
    }

    #[test]
    fn command_passed_tool_failure_overrides() {
        assert!(!command_passed(&json!({"success": true}), false));
    }

    #[test]
    fn extract_output_from_string() {
        let val = json!("hello world");
        assert_eq!(extract_output(&val), "hello world");
    }

    #[test]
    fn extract_output_from_stdout_stderr() {
        let val = json!({"stdout": "out", "stderr": "err"});
        let output = extract_output(&val);
        assert!(output.contains("out"));
        assert!(output.contains("err"));
    }

    #[test]
    fn extract_output_fallback_to_json_string() {
        let val = json!({"code": 42});
        let output = extract_output(&val);
        assert!(output.contains("42"));
    }

    #[test]
    fn truncate_short_text_unchanged() {
        assert_eq!(truncate("hello", 100), "hello");
    }

    #[test]
    fn truncate_long_text_clips() {
        let long = "a".repeat(100);
        let result = truncate(&long, 50);
        assert!(result.len() < 100);
        assert!(result.contains("(truncated)"));
    }
}
