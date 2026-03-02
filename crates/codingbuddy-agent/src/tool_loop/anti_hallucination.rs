//! Anti-hallucination mechanisms: file reference validation, shell command
//! detection, and user directive extraction.

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

// ── File reference validation ──

/// Detect when the model mentions file paths without having called fs_read/fs_glob first.
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

    // Check if any read/glob/grep tool was called (the model at least tried to look)
    let used_read_tools = tool_calls_made.iter().any(|tc| {
        tc.tool_name == "fs_read"
            || tc.tool_name == "fs.read"
            || tc.tool_name == "fs_glob"
            || tc.tool_name == "fs.glob"
            || tc.tool_name == "fs_grep"
            || tc.tool_name == "fs.grep"
            || tc.tool_name == "fs_list"
            || tc.tool_name == "fs.list"
    });

    // If any path is mentioned but no read/search tools were used, flag it.
    // Even a single unverified file reference (e.g. "cat audit.md") should trigger.
    !used_read_tools && !mentioned_paths.is_empty()
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
