//! Anti-hallucination mechanisms: file reference validation, shell command
//! detection, numeric consistency checking, and user directive extraction.

use std::collections::HashSet;

use super::types::ToolCallRecord;

// ── Constants ──

/// Character threshold for the hallucination nudge. Responses shorter than
/// this are allowed through without nudging (the model might be legitimately
/// giving a concise answer).
pub(crate) const HALLUCINATION_NUDGE_THRESHOLD: usize = 150;

/// Maximum number of hallucination nudges before letting the response through.
/// After this many nudges without the model using tools, it may be a legitimate
/// text-only response.
pub(crate) const MAX_NUDGE_ATTEMPTS: usize = 3;

/// The nudge message injected when hallucination is detected.
pub(crate) const HALLUCINATION_NUDGE: &str = "STOP. You are answering without using tools. \
     You MUST use a tool to verify your answer. Call fs_read, fs_glob, or fs_grep \
     to check the actual code before responding. Do NOT guess file contents or project structure.";

/// Known file extensions for path detection.
const FILE_EXTENSIONS: &[&str] = &[
    ".rs", ".ts", ".tsx", ".js", ".jsx", ".py", ".go", ".java", ".cpp", ".c", ".h", ".hpp", ".rb",
    ".swift", ".kt", ".sh", ".sql", ".json", ".toml", ".yaml", ".yml", ".md", ".html", ".css",
    ".scss", ".vue", ".svelte", ".zig", ".ex", ".exs", ".lock", ".cfg", ".ini", ".env", ".txt",
    ".xml", ".proto", ".tf", ".lua", ".php", ".cs", ".r", ".jl", ".dart", ".wasm",
];

/// Common dot-prefixed words that are NOT file paths (module access, version numbers, URLs).
const DOT_ACCESS_PREFIXES: &[&str] = &[
    "self.", "req.", "resp.", "cfg.", "fmt.", "io.", "os.", "fs.", "http.", "https.", "std.",
    "env.", "log.", "err.", "ctx.", "v0.", "v1.", "v2.", "v3.", "e.g.", "i.e.",
];

/// Tool names that represent read/search operations for path extraction.
const READ_TOOL_NAMES: &[&str] = &[
    "fs_read", "fs.read", "fs_glob", "fs.glob", "fs_grep", "fs.grep", "fs_list", "fs.list",
];

// ── File reference validation ──

/// Represents an accessed path with its type (file or directory).
#[derive(Debug)]
struct AccessedPath {
    path: String,
    is_directory: bool,
}

/// Extract file paths that the model actually accessed via tool calls.
/// Parses `args_json` for `"path"` or `"file_path"` keys.
/// For directory-listing tools (fs_glob, fs_list), also records the path as a directory.
fn extract_accessed_paths(tool_calls_made: &[ToolCallRecord]) -> Vec<AccessedPath> {
    let mut accessed = Vec::new();
    let mut seen = HashSet::new();

    let dir_tools = ["fs_glob", "fs.glob", "fs_list", "fs.list"];

    for tc in tool_calls_made {
        if !READ_TOOL_NAMES.contains(&tc.tool_name.as_str()) {
            continue;
        }

        let is_dir_tool = dir_tools.contains(&tc.tool_name.as_str());
        let prev_len = accessed.len();

        // Try to extract path from args_json
        if let Some(ref args_json) = tc.args_json
            && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(args_json)
        {
            for key in &["path", "file_path"] {
                if let Some(path_val) = parsed.get(key).and_then(|v| v.as_str()) {
                    let normalized = path_val.trim_start_matches("./").to_string();
                    if seen.insert(normalized.clone()) {
                        accessed.push(AccessedPath {
                            path: normalized,
                            is_directory: is_dir_tool,
                        });
                    }
                }
            }
        }

        // Fallback: try to extract path from args_summary if args_json didn't yield results
        let extracted_from_json = accessed.len() > prev_len;
        if !extracted_from_json {
            for segment in tc.args_summary.split('"') {
                let trimmed = segment.trim();
                if trimmed.contains('/') || FILE_EXTENSIONS.iter().any(|ext| trimmed.ends_with(ext))
                {
                    let normalized = trimmed.trim_start_matches("./").to_string();
                    if seen.insert(normalized.clone()) {
                        accessed.push(AccessedPath {
                            path: normalized,
                            is_directory: is_dir_tool,
                        });
                    }
                }
            }
        }
    }

    accessed
}

/// Check whether a mentioned path is covered by any accessed path.
///
/// Exact match: the mentioned path equals an accessed path (or vice versa).
/// Directory coverage: if a directory was listed (fs_glob/fs_list), any path
/// under that directory is considered covered.
fn path_is_covered(mentioned: &str, accessed_paths: &[AccessedPath]) -> bool {
    let normalized = mentioned.trim_start_matches("./");
    for ap in accessed_paths {
        // Exact match (either way)
        if ap.path == normalized {
            return true;
        }
        // Directory coverage: if accessed path is a directory, check prefix
        if ap.is_directory {
            if ap.path.ends_with('/') {
                if normalized.starts_with(&ap.path) {
                    return true;
                }
            } else if normalized.starts_with(&ap.path)
                && normalized.as_bytes().get(ap.path.len()) == Some(&b'/')
            {
                return true;
            }
        }
    }
    false
}

/// Detect when the model mentions file paths without having called fs_read/fs_glob first.
/// Uses path-specific verification: reading file A does NOT verify claims about file B.
/// Returns true if the text references paths that haven't been verified by tool calls.
pub(crate) fn has_unverified_file_references(
    text: &str,
    tool_calls_made: &[ToolCallRecord],
) -> bool {
    use std::sync::LazyLock;
    // Match file-like references: word chars, dots, slashes, hyphens with a 1-6 char extension.
    // Also match quoted paths like `"src/main.rs"` and backtick paths like `src/main.rs`.
    static FILE_REF_PATTERN: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"[\w./\-]+\.\w{1,6}\b").unwrap());

    let mentioned_paths: Vec<&str> = FILE_REF_PATTERN
        .find_iter(text)
        .map(|m| m.as_str())
        .filter(|p| {
            // Must look like a real file path, not a dot-access pattern
            let has_known_ext = FILE_EXTENSIONS.iter().any(|ext| p.ends_with(ext));
            let has_path_sep = p.contains('/');
            (has_known_ext || has_path_sep)
                && !DOT_ACCESS_PREFIXES.iter().any(|prefix| p.starts_with(prefix))
                && !p.contains('(')
                // Exclude version-like patterns (1.0, 2.34, etc.)
                && !p.chars().next().is_some_and(|c| c.is_ascii_digit())
        })
        .collect();

    if mentioned_paths.is_empty() {
        return false;
    }

    // If no read tools were used at all, any file reference is unverified
    let has_read_tools = tool_calls_made
        .iter()
        .any(|tc| READ_TOOL_NAMES.contains(&tc.tool_name.as_str()));
    if !has_read_tools {
        return true;
    }

    // Path-specific verification: build set of actually-accessed paths and check
    // each mentioned path against it
    let accessed_paths = extract_accessed_paths(tool_calls_made);

    // If we could not extract any specific paths from tool calls (e.g. args_json
    // was None for all), fall back to the blanket "read tool was used" heuristic
    // to avoid false positives.
    if accessed_paths.is_empty() {
        return false;
    }

    // Check if ANY mentioned path is NOT covered by accessed paths
    mentioned_paths
        .iter()
        .any(|mentioned| !path_is_covered(mentioned, &accessed_paths))
}

// ── Shell command detection ──

/// Detect when the model outputs shell commands as text instead of using tools.
/// Catches patterns like `cat file.rs`, `grep pattern`, `find . -name ...` etc.
/// that appear inside code blocks or as bare commands.
pub(crate) fn contains_shell_command_pattern(text: &str) -> bool {
    use std::sync::LazyLock;
    static SHELL_PATTERNS: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(
            r"(?m)(?:^```(?:bash|sh|shell|zsh)?\s*\n\s*(?:cat|head|tail|grep|find|ls|sed|awk)\s+|^\s*\$\s+(?:cat|head|tail|grep|find|ls|sed|awk)\s+)"
        ).unwrap()
    });
    SHELL_PATTERNS.is_match(text)
}

// ── Numeric consistency checking ──

/// Check the response text for numeric claims that contradict tool call results.
///
/// Scans for patterns like "6 crates", "22 files", etc. and cross-references
/// against tool result previews. Returns a correction message if a discrepancy
/// is found, or `None` if everything is consistent or there is insufficient
/// evidence to judge.
pub(crate) fn check_response_consistency(
    response_text: &str,
    tool_calls_made: &[ToolCallRecord],
) -> Option<String> {
    use std::sync::LazyLock;

    // Match patterns like "6 crates", "22 files", "3 errors"
    static NUMERIC_CLAIM: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(
            r"(\d+)\s+(crate|file|member|module|package|function|test|error|warning)s?",
        )
        .unwrap()
    });

    // Early return: skip string building if no numeric claims in the response
    if !NUMERIC_CLAIM.is_match(response_text) {
        return None;
    }

    // Collect all tool result text for evidence searching
    let mut result_text = String::new();
    for tc in tool_calls_made {
        if let Some(ref preview) = tc.result_preview {
            result_text.push_str(preview);
            result_text.push('\n');
        }
    }

    if result_text.is_empty() {
        return None;
    }

    let result_lower = result_text.to_ascii_lowercase();

    for cap in NUMERIC_CLAIM.captures_iter(response_text) {
        let claimed: usize = match cap[1].parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        let noun = &cap[2];

        // Count occurrences of the noun in tool results as evidence
        let actual = result_lower.matches(noun).count();

        // Only flag if there is meaningful evidence (actual > 0) and a real
        // discrepancy (difference > 1 to avoid off-by-one noise)
        if actual > 0 && claimed.abs_diff(actual) > 1 {
            return Some(format!(
                "Your response claims {} {}s, but tool results show {}. Please verify and correct.",
                claimed, noun, actual
            ));
        }
    }

    None
}

// ── User directive extraction ──

/// Extract explicit user directives from text.
///
/// Looks for sentences containing directive keywords like "always", "never",
/// "don't", "prefer", "make sure", etc. These represent user preferences that
/// must survive compaction so DeepSeek doesn't forget them mid-conversation.
///
/// Returns deduplicated directive sentences (max 10 to bound memory).
pub(crate) fn extract_user_directives(text: &str) -> Vec<String> {
    /// Keywords that signal a user directive (case-insensitive matching).
    const DIRECTIVE_KEYWORDS: &[&str] = &[
        "always ",
        "never ",
        "don't ",
        "do not ",
        "make sure ",
        "must ",
        "prefer ",
        "use ",
        "avoid ",
        "remember ",
        "important:",
        "rule:",
        "convention:",
    ];

    let mut directives = Vec::new();
    for sentence in split_sentences(text) {
        let lower = sentence.to_ascii_lowercase();
        // Skip very short or very long sentences (likely not directives)
        if sentence.len() < 10 || sentence.len() > 300 {
            continue;
        }
        // Skip sentences that are questions
        if sentence.trim_end().ends_with('?') {
            continue;
        }
        if DIRECTIVE_KEYWORDS.iter().any(|kw| lower.contains(kw)) {
            directives.push(sentence.trim().to_string());
        }
    }

    directives.truncate(10);
    directives
}

/// Split text into rough sentences by `. `, `! `, `\n`, etc.
pub(crate) fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    for i in 0..bytes.len() {
        let is_boundary = matches!(bytes[i], b'\n')
            || (bytes[i] == b'.'
                && i + 1 < bytes.len()
                && bytes[i + 1] == b' '
                && i > 0
                && bytes[i - 1].is_ascii_alphanumeric());
        if is_boundary {
            let end = if bytes[i] == b'.' { i + 1 } else { i };
            let seg = text[start..end].trim();
            if !seg.is_empty() {
                sentences.push(seg);
            }
            start = i + 1;
        }
    }
    // Remainder
    let seg = text[start..].trim();
    if !seg.is_empty() {
        sentences.push(seg);
    }
    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unverified_refs_path_specific() {
        // Tool read "src/main.rs" but text mentions "src/lib.rs" — should flag
        let tool_calls = vec![ToolCallRecord {
            tool_name: "fs_read".to_string(),
            tool_call_id: "c1".to_string(),
            args_summary: "path=\"src/main.rs\"".to_string(),
            success: true,
            duration_ms: 10,
            args_json: Some(r#"{"path":"src/main.rs"}"#.to_string()),
            result_preview: None,
        }];
        assert!(
            has_unverified_file_references("Check src/lib.rs for the issue", &tool_calls),
            "mentioning src/lib.rs when only src/main.rs was read should flag"
        );
    }

    #[test]
    fn test_verified_refs_exact_path_match() {
        // Tool read "src/main.rs" and text mentions "src/main.rs" — should not flag
        let tool_calls = vec![ToolCallRecord {
            tool_name: "fs_read".to_string(),
            tool_call_id: "c1".to_string(),
            args_summary: "path=\"src/main.rs\"".to_string(),
            success: true,
            duration_ms: 10,
            args_json: Some(r#"{"path":"src/main.rs"}"#.to_string()),
            result_preview: None,
        }];
        assert!(
            !has_unverified_file_references(
                "The file src/main.rs contains the entry point",
                &tool_calls
            ),
            "mentioning src/main.rs when it was read should not flag"
        );
    }

    #[test]
    fn test_unverified_refs_parent_dir_coverage() {
        // Reading a file in "src/" should cover references to files in "src/"
        // via the parent directory heuristic
        let tool_calls = vec![ToolCallRecord {
            tool_name: "fs_glob".to_string(),
            tool_call_id: "c1".to_string(),
            args_summary: "path=\"src\"".to_string(),
            success: true,
            duration_ms: 10,
            args_json: Some(r#"{"path":"src"}"#.to_string()),
            result_preview: None,
        }];
        assert!(
            !has_unverified_file_references("Found src/main.rs in the directory", &tool_calls),
            "mentioning src/main.rs when src/ was listed should not flag"
        );
    }

    #[test]
    fn test_unverified_refs_no_tool_calls() {
        // No tool calls at all — any file reference is unverified
        assert!(
            has_unverified_file_references("Check src/main.rs for bugs", &[]),
            "file ref with no tool calls should flag"
        );
    }

    #[test]
    fn test_consistency_check_catches_wrong_count() {
        // Response says "6 crates" but tool results have many crate mentions
        let tool_calls = vec![ToolCallRecord {
            tool_name: "fs_read".to_string(),
            tool_call_id: "c1".to_string(),
            args_summary: String::new(),
            success: true,
            duration_ms: 10,
            args_json: None,
            result_preview: Some(
                "crate codingbuddy-cli\ncrate codingbuddy-agent\ncrate codingbuddy-core\n\
                 crate codingbuddy-llm\ncrate codingbuddy-tools\ncrate codingbuddy-policy\n\
                 crate codingbuddy-hooks\ncrate codingbuddy-store\ncrate codingbuddy-memory\n\
                 crate codingbuddy-index\ncrate codingbuddy-mcp\ncrate codingbuddy-ui"
                    .to_string(),
            ),
        }];
        let result = check_response_consistency("This workspace has 6 crates total.", &tool_calls);
        assert!(
            result.is_some(),
            "should detect discrepancy between claimed 6 and actual 12 crates"
        );
        let msg = result.unwrap();
        assert!(msg.contains("6"), "correction should mention claimed count");
        assert!(msg.contains("crate"), "correction should mention noun");
    }

    #[test]
    fn test_consistency_check_no_false_positive() {
        // Response says "3 errors" but no tool results mention errors — no evidence to contradict
        let tool_calls = vec![ToolCallRecord {
            tool_name: "fs_read".to_string(),
            tool_call_id: "c1".to_string(),
            args_summary: String::new(),
            success: true,
            duration_ms: 10,
            args_json: None,
            result_preview: Some("fn main() { println!(\"hello\"); }".to_string()),
        }];
        let result = check_response_consistency("I found 3 errors in the code.", &tool_calls);
        assert!(
            result.is_none(),
            "should not flag when tool results have no conflicting evidence"
        );
    }

    #[test]
    fn test_consistency_check_no_tool_results() {
        // No tool result previews — should return None (insufficient evidence)
        let tool_calls = vec![ToolCallRecord {
            tool_name: "fs_read".to_string(),
            tool_call_id: "c1".to_string(),
            args_summary: String::new(),
            success: true,
            duration_ms: 10,
            args_json: None,
            result_preview: None,
        }];
        let result = check_response_consistency("This workspace has 22 crates.", &tool_calls);
        assert!(
            result.is_none(),
            "should not flag when there are no tool result previews"
        );
    }

    #[test]
    fn test_consistency_check_close_count_not_flagged() {
        // Claimed 3 files, tool shows 2 — difference of 1, should NOT flag (off-by-one tolerance)
        let tool_calls = vec![ToolCallRecord {
            tool_name: "fs_glob".to_string(),
            tool_call_id: "c1".to_string(),
            args_summary: String::new(),
            success: true,
            duration_ms: 10,
            args_json: None,
            result_preview: Some("file: main.rs\nfile: lib.rs".to_string()),
        }];
        let result = check_response_consistency("Found 3 files in src/.", &tool_calls);
        assert!(
            result.is_none(),
            "off-by-one difference should not trigger correction"
        );
    }
}
