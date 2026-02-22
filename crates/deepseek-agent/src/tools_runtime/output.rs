use crate::mode_router::FailureTracker;
use crate::observation::{ActionRecord, extract_file_refs, summarize_args};

pub(crate) fn summarize_tool_args(tool_name: &str, args: &serde_json::Value) -> String {
    match tool_name {
        "fs.read" | "fs.write" | "fs.edit" => args
            .get("path")
            .and_then(|v| v.as_str())
            .map(|p| format!("path={p}"))
            .unwrap_or_default(),
        "fs.glob" | "fs.grep" => args
            .get("pattern")
            .and_then(|v| v.as_str())
            .map(|p| format!("pattern={p}"))
            .unwrap_or_default(),
        "bash.run" => args
            .get("cmd")
            .and_then(|v| v.as_str())
            .map(|c| {
                if c.len() > 80 {
                    format!("cmd={}...", &c[..80])
                } else {
                    format!("cmd={c}")
                }
            })
            .unwrap_or_default(),
        "multi_edit" => args
            .get("files")
            .and_then(|v| v.as_array())
            .map(|a| format!("{} files", a.len()))
            .unwrap_or_default(),
        "web.fetch" => args
            .get("url")
            .and_then(|v| v.as_str())
            .map(|u| {
                if u.len() > 80 {
                    format!("url={}...", &u[..80])
                } else {
                    format!("url={u}")
                }
            })
            .unwrap_or_default(),
        "web.search" => args
            .get("query")
            .and_then(|v| v.as_str())
            .map(|q| format!("query={q}"))
            .unwrap_or_default(),
        "git.show" => args
            .get("spec")
            .and_then(|v| v.as_str())
            .map(|s| format!("spec={s}"))
            .unwrap_or_default(),
        "index.query" => args
            .get("q")
            .and_then(|v| v.as_str())
            .map(|q| format!("q={q}"))
            .unwrap_or_default(),
        "patch.stage" => args
            .get("unified_diff")
            .and_then(|v| v.as_str())
            .map(|d| {
                if d.len() > 100 {
                    format!("diff={}...", &d[..100])
                } else {
                    format!("diff={d}")
                }
            })
            .unwrap_or_default(),
        "patch.apply" => args
            .get("patch_id")
            .and_then(|v| v.as_str())
            .map(|id| format!("patch_id={id}"))
            .unwrap_or_default(),
        "diagnostics.check" => args
            .get("path")
            .and_then(|v| v.as_str())
            .map(|p| format!("path={p}"))
            .unwrap_or_else(|| "project".to_string()),
        "notebook.read" | "notebook.edit" => args
            .get("path")
            .and_then(|v| v.as_str())
            .map(|p| format!("path={p}"))
            .unwrap_or_default(),
        _ => String::new(),
    }
}

pub(crate) fn build_observation_action(
    tool_name: &str,
    args: &serde_json::Value,
    success: bool,
    exit_code: Option<i32>,
    output: &str,
) -> ActionRecord {
    let output_head = if output.len() > 500 {
        let cut = output.floor_char_boundary(500);
        format!("{}...", &output[..cut])
    } else {
        output.to_string()
    };

    ActionRecord {
        tool: tool_name.to_string(),
        args_summary: summarize_args(tool_name, args),
        success,
        exit_code,
        output_head,
        refs: extract_file_refs(output),
    }
}

pub(crate) fn record_error_modules_from_output(tracker: &mut FailureTracker, output: &str) {
    for file_ref in extract_file_refs(output) {
        if let Some(module) = extract_module_name_from_ref(&file_ref) {
            tracker.record_error_module(&module);
        }
    }
}

pub(crate) fn extract_module_name_from_ref(file_ref: &str) -> Option<String> {
    let path = file_ref.split(':').next()?;
    let parts: Vec<&str> = path.split('/').collect();
    for part in parts.iter().rev().skip(1) {
        if !matches!(*part, "src" | "lib" | "tests" | "test" | "." | "..") {
            return Some((*part).to_string());
        }
    }
    let filename = parts.last()?;
    filename.split('.').next().map(|s| s.to_string())
}

pub(crate) fn truncate_tool_output(tool_name: &str, output: &str, max_bytes: usize) -> String {
    if output.len() <= max_bytes {
        return output.to_string();
    }
    let lines: Vec<&str> = output.lines().collect();

    // Tool-specific truncation strategies
    match tool_name {
        "bash.run" => {
            // For bash: try to separate stderr from stdout, always keep stderr,
            // then keep last 200 lines of stdout.
            // Since output is a JSON value, parse it to extract stderr if possible.
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(output) {
                let stderr = val.get("stderr").and_then(|v| v.as_str()).unwrap_or("");
                let stdout = val.get("stdout").and_then(|v| v.as_str()).unwrap_or("");
                let stdout_lines: Vec<&str> = stdout.lines().collect();
                let kept_stdout = if stdout_lines.len() > 200 {
                    format!(
                        "... ({} lines omitted) ...\n{}",
                        stdout_lines.len() - 200,
                        stdout_lines[stdout_lines.len() - 200..].join("\n")
                    )
                } else {
                    stdout.to_string()
                };
                return serde_json::to_string(&serde_json::json!({
                    "stdout": kept_stdout,
                    "stderr": stderr,
                    "truncated": true,
                    "original_stdout_lines": stdout_lines.len()
                }))
                .unwrap_or_else(|_| truncate_generic(&lines, max_bytes, output.len()));
            }
            truncate_generic(&lines, max_bytes, output.len())
        }
        "fs.read" => {
            // For fs.read: show line count + head/tail
            if lines.len() > 200 {
                let head_count = 80;
                let tail_count = 80;
                let head: Vec<&str> = lines[..head_count].to_vec();
                let tail: Vec<&str> = lines[lines.len() - tail_count..].to_vec();
                format!(
                    "{}\n\n... ({} total lines, {} lines omitted) ...\n\n{}",
                    head.join("\n"),
                    lines.len(),
                    lines.len() - head_count - tail_count,
                    tail.join("\n")
                )
            } else {
                truncate_generic(&lines, max_bytes, output.len())
            }
        }
        _ => truncate_generic(&lines, max_bytes, output.len()),
    }
}

pub(crate) fn truncate_generic(lines: &[&str], max_bytes: usize, total_bytes: usize) -> String {
    if lines.len() > 200 {
        let head: Vec<&str> = lines[..100].to_vec();
        let tail: Vec<&str> = lines[lines.len() - 100..].to_vec();
        format!(
            "{}\n... ({} lines omitted) ...\n{}",
            head.join("\n"),
            lines.len() - 200,
            tail.join("\n")
        )
    } else {
        // Byte-truncate at a safe char boundary
        let joined = lines.join("\n");
        let truncated = &joined[..joined.floor_char_boundary(max_bytes)];
        format!("{}... (truncated, {} bytes total)", truncated, total_bytes)
    }
}
