//! Pre-execution validation for tool call arguments.
//!
//! Catches common malformed arguments BEFORE executing the tool, giving the
//! model a clear error message to self-correct without wasting an execution
//! cycle.

use serde_json::Value;

/// Validate tool arguments before execution.
///
/// Returns `Ok(())` if valid, `Err(message)` with a human-readable error
/// if the arguments are malformed. The error message is designed to be
/// directly useful to the LLM for self-correction.
pub fn validate_tool_args(tool_name: &str, args: &Value) -> Result<(), String> {
    match tool_name {
        "fs.read" => {
            require_path(args)?;
            validate_line_range(args)?;
            Ok(())
        }
        "fs.edit" => {
            require_path(args)?;
            require_string(
                args,
                "search",
                "search string is required (the exact text to find)",
            )?;
            // "replace" can be an empty string (deletion), but must exist
            if args.get("replace").is_none() && args.get("new_string").is_none() {
                return Err(
                    "'replace' field is required (use empty string \"\" for deletion)".to_string(),
                );
            }
            Ok(())
        }
        "fs.write" => {
            require_path(args)?;
            if args.get("content").is_none() {
                return Err("'content' field is required".to_string());
            }
            Ok(())
        }
        "fs.grep" => {
            require_string(args, "pattern", "regex pattern is required")?;
            Ok(())
        }
        "fs.glob" => {
            require_string(args, "pattern", "glob pattern is required")?;
            Ok(())
        }
        "bash.run" => {
            require_string(args, "cmd", "command string is required")?;
            if let Some(timeout) = args.get("timeout").and_then(|v| v.as_i64()) {
                if timeout <= 0 {
                    return Err("timeout must be a positive number of seconds".to_string());
                }
                if timeout > 600 {
                    return Err(
                        "timeout cannot exceed 600 seconds — break the task into smaller steps"
                            .to_string(),
                    );
                }
            }
            Ok(())
        }
        "multi_edit" => {
            if let Some(edits) = args.get("edits") {
                if !edits.is_array() {
                    return Err("'edits' must be an array of edit objects".to_string());
                }
                if let Some(arr) = edits.as_array() {
                    if arr.is_empty() {
                        return Err("'edits' array must not be empty".to_string());
                    }
                    for (i, edit) in arr.iter().enumerate() {
                        if edit
                            .get("path")
                            .or_else(|| edit.get("file_path"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .is_empty()
                        {
                            return Err(format!("edits[{i}]: 'path' is required"));
                        }
                    }
                }
            } else {
                return Err("'edits' field is required".to_string());
            }
            Ok(())
        }
        _ => Ok(()), // Unknown tools pass through without validation
    }
}

fn require_string(args: &Value, field: &str, msg: &str) -> Result<(), String> {
    match args.get(field) {
        Some(Value::String(s)) if !s.is_empty() => Ok(()),
        Some(Value::String(_)) => Err(format!("{field}: {msg} (got empty string)")),
        Some(v) => Err(format!(
            "{field}: expected string, got {}",
            v.to_string().chars().take(50).collect::<String>()
        )),
        None => Err(format!("{field}: {msg}")),
    }
}

/// Check that at least one of the given field names is a non-empty string.
/// Models may use either `path` or `file_path` depending on the schema version.
fn require_path(args: &Value) -> Result<(), String> {
    for field in &["path", "file_path"] {
        if let Some(Value::String(s)) = args.get(*field) {
            if !s.is_empty() {
                return Ok(());
            }
        }
    }
    Err("'path' (or 'file_path') is required".to_string())
}

fn validate_line_range(args: &Value) -> Result<(), String> {
    if let Some(start) = args.get("start_line").and_then(|v| v.as_i64()) {
        if start < 1 {
            return Err("start_line must be >= 1 (lines are 1-based)".to_string());
        }
    }
    if let Some(end) = args.get("end_line").and_then(|v| v.as_i64()) {
        if end < 1 {
            return Err("end_line must be >= 1 (lines are 1-based)".to_string());
        }
        if let Some(start) = args.get("start_line").and_then(|v| v.as_i64()) {
            if end < start {
                return Err(format!("end_line ({end}) must be >= start_line ({start})"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn fs_read_valid() {
        assert!(validate_tool_args("fs.read", &json!({"path": "src/main.rs"})).is_ok());
    }

    #[test]
    fn fs_read_missing_path() {
        let err = validate_tool_args("fs.read", &json!({})).unwrap_err();
        assert!(err.contains("path"), "error should mention path: {err}");
    }

    #[test]
    fn fs_read_empty_path() {
        let err = validate_tool_args("fs.read", &json!({"path": ""})).unwrap_err();
        assert!(err.contains("path"), "{err}");
    }

    #[test]
    fn fs_read_valid_file_path_alias() {
        assert!(validate_tool_args("fs.read", &json!({"file_path": "src/main.rs"})).is_ok());
    }

    #[test]
    fn fs_read_invalid_line_range() {
        let err = validate_tool_args(
            "fs.read",
            &json!({"path": "f.rs", "start_line": 10, "end_line": 5}),
        )
        .unwrap_err();
        assert!(err.contains("end_line"), "{err}");
    }

    #[test]
    fn fs_edit_valid() {
        assert!(
            validate_tool_args(
                "fs.edit",
                &json!({"path": "f.rs", "search": "old", "replace": "new"})
            )
            .is_ok()
        );
    }

    #[test]
    fn fs_edit_missing_search() {
        let err =
            validate_tool_args("fs.edit", &json!({"path": "f.rs", "replace": "new"})).unwrap_err();
        assert!(err.contains("search"), "{err}");
    }

    #[test]
    fn fs_edit_missing_replace() {
        let err =
            validate_tool_args("fs.edit", &json!({"path": "f.rs", "search": "old"})).unwrap_err();
        assert!(err.contains("replace"), "{err}");
    }

    #[test]
    fn fs_edit_empty_replace_is_ok() {
        // Empty replace = deletion, which is valid
        assert!(
            validate_tool_args(
                "fs.edit",
                &json!({"path": "f.rs", "search": "old", "replace": ""})
            )
            .is_ok()
        );
    }

    #[test]
    fn bash_run_valid() {
        assert!(validate_tool_args("bash.run", &json!({"cmd": "ls"})).is_ok());
    }

    #[test]
    fn bash_run_missing_cmd() {
        let err = validate_tool_args("bash.run", &json!({})).unwrap_err();
        assert!(err.contains("cmd"), "{err}");
    }

    #[test]
    fn bash_run_timeout_too_high() {
        let err =
            validate_tool_args("bash.run", &json!({"cmd": "ls", "timeout": 9999})).unwrap_err();
        assert!(err.contains("600"), "{err}");
    }

    #[test]
    fn bash_run_negative_timeout() {
        let err = validate_tool_args("bash.run", &json!({"cmd": "ls", "timeout": -1})).unwrap_err();
        assert!(err.contains("positive"), "{err}");
    }

    #[test]
    fn unknown_tool_passes() {
        assert!(validate_tool_args("some.unknown.tool", &json!({})).is_ok());
    }

    #[test]
    fn multi_edit_valid() {
        assert!(
            validate_tool_args(
                "multi_edit",
                &json!({"edits": [{"path": "f.rs", "search": "a", "replace": "b"}]})
            )
            .is_ok()
        );
    }

    #[test]
    fn multi_edit_empty_edits() {
        let err = validate_tool_args("multi_edit", &json!({"edits": []})).unwrap_err();
        assert!(err.contains("empty"), "{err}");
    }

    #[test]
    fn multi_edit_missing_path() {
        let err = validate_tool_args(
            "multi_edit",
            &json!({"edits": [{"search": "a", "replace": "b"}]}),
        )
        .unwrap_err();
        assert!(err.contains("path"), "{err}");
    }
}
