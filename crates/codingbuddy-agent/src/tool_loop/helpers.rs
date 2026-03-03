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

/// Strip fenced code blocks from text so that word counting applies only to prose.
///
/// Removes everything between ``` markers (inclusive), leaving surrounding text intact.
pub(crate) fn strip_code_blocks(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut inside_code = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            inside_code = !inside_code;
            continue;
        }
        if !inside_code {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_code_blocks_removes_fenced_code() {
        let text = "Hello world\n```rust\nfn main() {}\n```\nGoodbye world\n";
        let prose = strip_code_blocks(text);
        assert!(prose.contains("Hello world"));
        assert!(prose.contains("Goodbye world"));
        assert!(!prose.contains("fn main"));
    }

    #[test]
    fn strip_code_blocks_preserves_prose_only() {
        let text = "Just some prose with no code blocks at all.";
        let prose = strip_code_blocks(text);
        assert!(prose.contains("Just some prose"));
    }

    #[test]
    fn strip_code_blocks_handles_multiple_blocks() {
        let text = "intro\n```\ncode1\n```\nmiddle\n```python\ncode2\n```\nend\n";
        let prose = strip_code_blocks(text);
        assert!(prose.contains("intro"));
        assert!(prose.contains("middle"));
        assert!(prose.contains("end"));
        assert!(!prose.contains("code1"));
        assert!(!prose.contains("code2"));
    }

    #[test]
    fn verbosity_nudge_strips_code_blocks() {
        // Simulate: verbose prose (~500 words) + a small code block.
        // Old logic would see "```" and zero the count. New logic strips blocks first.
        let prose = "word ".repeat(500);
        let text = format!("{prose}\n```rust\nfn main() {{}}\n```\n");
        let stripped = strip_code_blocks(&text);
        let word_count = stripped.split_whitespace().count();
        assert!(
            word_count > 400,
            "expected >400 prose words, got {word_count}"
        );
    }
}
