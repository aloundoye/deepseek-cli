//! Small utility functions used across the tool loop.

/// Check if a tool API name is read-only.
pub(crate) fn is_read_only_api_name(name: &str) -> bool {
    matches!(
        name,
        "fs_read"
            | "fs_glob"
            | "fs_grep"
            | "fs_list"
            | "git_status"
            | "git_diff"
            | "git_show"
            | "web_search"
            | "web_fetch"
            | "notebook_read"
            | "index_query"
            | "extended_thinking"
            | "think_deeply"
            | "user_question"
            | "spawn_task"
            | "task_output"
            | "task_list"
            | "task_get"
            | "diagnostics_check"
    )
}

/// Produce a short summary of tool arguments for display.
pub(crate) fn summarize_args(args: &serde_json::Value) -> String {
    let mut parts = Vec::new();
    if let Some(obj) = args.as_object() {
        for (key, val) in obj {
            let short = match val {
                serde_json::Value::String(s) => {
                    if s.len() > 60 {
                        let safe_end = s.floor_char_boundary(57);
                        format!("{key}=\"{}...\"", &s[..safe_end])
                    } else {
                        format!("{key}=\"{s}\"")
                    }
                }
                serde_json::Value::Number(n) => format!("{key}={n}"),
                serde_json::Value::Bool(b) => format!("{key}={b}"),
                _ => format!("{key}=..."),
            };
            parts.push(short);
        }
    }
    if parts.is_empty() {
        return "()".to_string();
    }
    parts.join(", ")
}
