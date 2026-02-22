//! Pre-execution validation for tool call arguments.
//!
//! Catches common malformed arguments BEFORE executing the tool, giving the
//! model a clear error message to self-correct without wasting an execution
//! cycle.

use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

/// Normalize tool arguments into the canonical shapes expected by tool host
/// implementations.
///
/// This makes the execution layer resilient to common schema variants
/// (`file_path` vs `path`, `command` vs `cmd`, `timeout_ms` vs `timeout`,
/// legacy `multi_edit.edits` vs canonical `multi_edit.files`).
pub fn normalize_tool_args(tool_name: &str, args: &mut Value) {
    let Some(obj) = args.as_object_mut() else {
        return;
    };

    match tool_name {
        "fs.read" | "fs.write" | "fs.edit" | "notebook.read" | "notebook.edit"
        | "diagnostics.check" => {
            move_alias(obj, "file_path", "path");
        }
        "fs.list" => {
            move_alias(obj, "path", "dir");
        }
        "bash.run" => {
            move_alias(obj, "command", "cmd");
            if !obj.contains_key("timeout")
                && let Some(ms) = obj.get("timeout_ms").and_then(|v| v.as_u64())
            {
                // Tool host expects timeout in seconds.
                let secs = ((ms.saturating_add(999)) / 1000).clamp(1, 600);
                obj.insert("timeout".to_string(), Value::from(secs));
            }
            obj.remove("timeout_ms");
        }
        "index.query" => {
            move_alias(obj, "query", "q");
        }
        "multi_edit" => {
            normalize_multi_edit(obj);
        }
        _ => {}
    }
}

/// Normalize tool arguments and map absolute in-workspace paths to workspace-relative.
///
/// This preserves strict policy checks (absolute paths outside workspace are still
/// rejected) while keeping compatibility with model outputs that provide absolute
/// file paths under the current workspace.
pub fn normalize_tool_args_with_workspace(tool_name: &str, args: &mut Value, workspace: &Path) {
    normalize_tool_args(tool_name, args);
    relativize_workspace_paths(tool_name, args, workspace);
}

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
            if let Some(files) = args.get("files") {
                if !files.is_array() {
                    return Err("'files' must be an array".to_string());
                }
                if let Some(arr) = files.as_array() {
                    if arr.is_empty() {
                        return Err("'files' array must not be empty".to_string());
                    }
                    for (i, file) in arr.iter().enumerate() {
                        let path = file
                            .get("path")
                            .or_else(|| file.get("file_path"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if path.is_empty() {
                            return Err(format!("files[{i}]: 'path' is required"));
                        }
                        let edits =
                            file.get("edits")
                                .and_then(|v| v.as_array())
                                .ok_or_else(|| {
                                    format!("files[{i}]: 'edits' must be a non-empty array")
                                })?;
                        if edits.is_empty() {
                            return Err(format!("files[{i}]: 'edits' must not be empty"));
                        }
                    }
                }
            } else if let Some(edits) = args.get("edits") {
                // Backward compatibility for legacy shape.
                if !edits.is_array() {
                    return Err("'edits' must be an array of edit objects".to_string());
                }
                if let Some(arr) = edits.as_array()
                    && arr.is_empty()
                {
                    return Err("'edits' array must not be empty".to_string());
                }
            } else {
                return Err("'files' (or legacy 'edits') field is required".to_string());
            }
            Ok(())
        }
        _ => Ok(()), // Unknown tools pass through without validation
    }
}

fn move_alias(obj: &mut Map<String, Value>, from: &str, to: &str) {
    if !obj.contains_key(to)
        && let Some(v) = obj.get(from).cloned()
    {
        obj.insert(to.to_string(), v);
    }
}

fn relativize_workspace_paths(tool_name: &str, args: &mut Value, workspace: &Path) {
    let Some(obj) = args.as_object_mut() else {
        return;
    };

    let workspace_roots = collect_workspace_roots(workspace);
    if workspace_roots.is_empty() {
        return;
    }

    match tool_name {
        "fs.read" | "fs.write" | "fs.edit" | "notebook.read" | "notebook.edit"
        | "diagnostics.check" => {
            relativize_path_field(obj, "path", &workspace_roots);
        }
        "fs.list" => {
            relativize_path_field(obj, "dir", &workspace_roots);
        }
        "fs.glob" => {
            relativize_path_field(obj, "base", &workspace_roots);
        }
        "multi_edit" => {
            if let Some(files) = obj.get_mut("files").and_then(|v| v.as_array_mut()) {
                for file in files {
                    if let Some(file_obj) = file.as_object_mut() {
                        relativize_path_field(file_obj, "path", &workspace_roots);
                    }
                }
            }
            relativize_path_field(obj, "path", &workspace_roots);
        }
        _ => {}
    }
}

fn collect_workspace_roots(workspace: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if workspace.is_absolute() {
        roots.push(workspace.to_path_buf());
    }
    if let Ok(canonical) = std::fs::canonicalize(workspace)
        && !roots.contains(&canonical)
    {
        roots.push(canonical);
    }
    roots
}

fn relativize_path_field(obj: &mut Map<String, Value>, field: &str, workspace_roots: &[PathBuf]) {
    let original = obj
        .get(field)
        .and_then(|v| v.as_str())
        .map(ToString::to_string);
    let Some(original) = original else {
        return;
    };

    if let Some(relative) = absolute_to_workspace_relative(&original, workspace_roots) {
        obj.insert(field.to_string(), Value::String(relative));
    }
}

fn absolute_to_workspace_relative(path: &str, workspace_roots: &[PathBuf]) -> Option<String> {
    let candidate = Path::new(path);
    if !candidate.is_absolute() {
        return None;
    }

    for root in workspace_roots {
        if let Ok(rel) = candidate.strip_prefix(root) {
            return Some(normalize_relative_path(rel));
        }
    }

    if let Ok(canonical_candidate) = std::fs::canonicalize(candidate) {
        for root in workspace_roots {
            if let Ok(rel) = canonical_candidate.strip_prefix(root) {
                return Some(normalize_relative_path(rel));
            }
        }
    }

    None
}

fn normalize_relative_path(path: &Path) -> String {
    let normalized = path.to_string_lossy().replace('\\', "/");
    if normalized.is_empty() {
        ".".to_string()
    } else {
        normalized
    }
}

fn normalize_multi_edit(obj: &mut Map<String, Value>) {
    // Canonical shape already provided: normalize nested aliases only.
    if let Some(files) = obj.get_mut("files").and_then(|v| v.as_array_mut()) {
        for file in files {
            let Some(file_obj) = file.as_object_mut() else {
                continue;
            };
            move_alias(file_obj, "file_path", "path");
            if let Some(edits) = file_obj.get_mut("edits").and_then(|v| v.as_array_mut()) {
                for edit in edits {
                    let Some(edit_obj) = edit.as_object_mut() else {
                        continue;
                    };
                    normalize_edit_fields(edit_obj);
                }
            }
        }
        return;
    }

    let parent_path = obj
        .get("path")
        .or_else(|| obj.get("file_path"))
        .and_then(|v| v.as_str())
        .map(ToString::to_string);

    let edits_value = match obj.remove("edits") {
        Some(v) => v,
        None => return,
    };
    let Some(edits_arr) = edits_value.as_array() else {
        obj.insert("edits".to_string(), edits_value);
        return;
    };

    let mut grouped: std::collections::BTreeMap<String, Vec<Value>> =
        std::collections::BTreeMap::new();

    for raw_edit in edits_arr {
        let Some(raw_obj) = raw_edit.as_object() else {
            continue;
        };
        let mut edit_obj = raw_obj.clone();
        normalize_edit_fields(&mut edit_obj);

        let path = edit_obj
            .get("path")
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
            .or_else(|| parent_path.clone());
        let Some(path) = path else {
            continue;
        };

        edit_obj.remove("path");
        grouped
            .entry(path)
            .or_default()
            .push(Value::Object(edit_obj));
    }

    if grouped.is_empty() {
        obj.insert("edits".to_string(), edits_value);
        return;
    }

    let mut files = Vec::new();
    for (path, edits) in grouped {
        let mut file_obj = Map::new();
        file_obj.insert("path".to_string(), Value::String(path));
        file_obj.insert("edits".to_string(), Value::Array(edits));
        files.push(Value::Object(file_obj));
    }
    obj.insert("files".to_string(), Value::Array(files));
}

fn normalize_edit_fields(edit_obj: &mut Map<String, Value>) {
    move_alias(edit_obj, "file_path", "path");
    move_alias(edit_obj, "old_string", "search");
    move_alias(edit_obj, "new_string", "replace");
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
        if let Some(Value::String(s)) = args.get(*field)
            && !s.is_empty()
        {
            return Ok(());
        }
    }
    Err("'path' (or 'file_path') is required".to_string())
}

fn validate_line_range(args: &Value) -> Result<(), String> {
    if let Some(start) = args.get("start_line").and_then(|v| v.as_i64())
        && start < 1
    {
        return Err("start_line must be >= 1 (lines are 1-based)".to_string());
    }
    if let Some(end) = args.get("end_line").and_then(|v| v.as_i64()) {
        if end < 1 {
            return Err("end_line must be >= 1 (lines are 1-based)".to_string());
        }
        if let Some(start) = args.get("start_line").and_then(|v| v.as_i64())
            && end < start
        {
            return Err(format!("end_line ({end}) must be >= start_line ({start})"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_workspace(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("{prefix}-{nanos}"));
        fs::create_dir_all(&dir).expect("create temp workspace");
        dir
    }

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
                &json!({"files": [{"path": "f.rs", "edits": [{"search": "a", "replace": "b"}]}]})
            )
            .is_ok()
        );
    }

    #[test]
    fn multi_edit_empty_edits() {
        let err = validate_tool_args("multi_edit", &json!({"files": []})).unwrap_err();
        assert!(err.contains("empty") || err.contains("required"), "{err}");
    }

    #[test]
    fn multi_edit_missing_path() {
        let err = validate_tool_args(
            "multi_edit",
            &json!({"files": [{"edits": [{"search":"a","replace":"b"}]}]}),
        )
        .unwrap_err();
        assert!(err.contains("path"), "{err}");
    }

    #[test]
    fn normalize_bash_run_aliases() {
        let mut args = json!({"command": "cargo test", "timeout_ms": 3100});
        normalize_tool_args("bash.run", &mut args);
        assert_eq!(args["cmd"], "cargo test");
        assert_eq!(args["timeout"], 4);
        assert!(args.get("timeout_ms").is_none());
    }

    #[test]
    fn normalize_multi_edit_legacy_grouping() {
        let mut args = json!({
            "edits": [
                {"file_path": "a.rs", "old_string": "A", "new_string": "B"},
                {"path": "a.rs", "search": "C", "replace": "D"},
                {"path": "b.rs", "old_string": "X", "new_string": "Y"}
            ]
        });
        normalize_tool_args("multi_edit", &mut args);
        let files = args.get("files").and_then(|v| v.as_array()).unwrap();
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|f| f["path"] == "a.rs"));
        assert!(files.iter().any(|f| f["path"] == "b.rs"));
    }

    #[test]
    fn normalize_multi_edit_with_parent_path() {
        let mut args = json!({
            "path": "src/lib.rs",
            "edits": [
                {"old_string": "foo", "new_string": "bar"}
            ]
        });
        normalize_tool_args("multi_edit", &mut args);
        let files = args.get("files").and_then(|v| v.as_array()).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0]["path"], "src/lib.rs");
    }

    #[test]
    fn normalize_with_workspace_relativizes_legacy_absolute_file_path() {
        let dir = make_temp_workspace("deepseek-tools-validation-path");
        let target = dir.join("src/lib.rs");
        fs::create_dir_all(target.parent().expect("parent")).expect("mkdir");
        fs::write(&target, "fn main() {}").expect("write");

        let mut args = json!({"file_path": target.to_string_lossy().to_string()});
        normalize_tool_args_with_workspace("fs.read", &mut args, &dir);
        assert_eq!(args["path"], "src/lib.rs");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn normalize_with_workspace_relativizes_nonexistent_absolute_path_under_workspace() {
        let dir = make_temp_workspace("deepseek-tools-validation-missing");
        let target = dir.join("missing.rs");
        let mut args = json!({"file_path": target.to_string_lossy().to_string()});
        normalize_tool_args_with_workspace("fs.read", &mut args, &dir);
        assert_eq!(args["path"], "missing.rs");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn normalize_with_workspace_relativizes_multi_edit_files() {
        let dir = make_temp_workspace("deepseek-tools-validation-multi");
        let file_a = dir.join("a.rs");
        let mut args = json!({
            "files": [{
                "path": file_a.to_string_lossy().to_string(),
                "edits": [{"search": "a", "replace": "b"}]
            }]
        });

        normalize_tool_args_with_workspace("multi_edit", &mut args, &dir);
        let files = args.get("files").and_then(|v| v.as_array()).unwrap();
        assert_eq!(files[0]["path"], "a.rs");
        let _ = fs::remove_dir_all(&dir);
    }
}
