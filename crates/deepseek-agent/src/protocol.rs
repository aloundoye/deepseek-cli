//! Strict protocol types and validators for R1 ↔ orchestrator communication.
//!
//! R1 cannot use function calling, so it emits structured JSON as plain text.
//! This module extracts, validates, and deserializes those JSON envelopes.

use serde::{Deserialize, Serialize};

// ── R1 protocol types ───────────────────────────────────────────────────

/// Top-level envelope emitted by R1 as plain text JSON.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum R1Response {
    ToolIntent(ToolIntent),
    DelegatePatch(DelegatePatch),
    Done(DoneResponse),
    Abort(AbortResponse),
}

/// R1 requests the orchestrator to execute a single tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolIntent {
    pub step_id: String,
    pub tool: String,
    pub args: serde_json::Value,
    pub why: String,
    #[serde(default)]
    pub expected: String,
    #[serde(default)]
    pub verify_after: bool,
}

/// R1 delegates a coding task to V3 as a patch writer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DelegatePatch {
    pub step_id: String,
    pub task: String,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub acceptance: Vec<String>,
    #[serde(default)]
    pub inputs: DelegatePatchInputs,
    #[serde(default = "default_true")]
    pub verify_after: bool,
}

fn default_true() -> bool {
    true
}

/// Context references for the V3 patch writer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct DelegatePatchInputs {
    #[serde(default)]
    pub relevant_files: Vec<String>,
    #[serde(default)]
    pub context_refs: Vec<String>,
}

/// R1 signals task completion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DoneResponse {
    pub summary: String,
    #[serde(default)]
    pub verification: String,
}

/// R1 signals it cannot complete the task.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AbortResponse {
    pub reason: String,
}

// ── V3 patch-only response types ────────────────────────────────────────

/// V3's response when operating in patch-only mode.
#[derive(Debug, Clone, PartialEq)]
pub enum V3PatchResponse {
    /// A unified diff to apply.
    UnifiedDiff(String),
    /// V3 needs more context before producing a patch.
    NeedMoreContext(NeedMoreContext),
}

/// V3 requests additional file context.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeedMoreContext {
    #[serde(rename = "type")]
    pub response_type: String,
    pub missing: Vec<String>,
}

// ── Known R1 tool names ─────────────────────────────────────────────────

/// Allowed tool names in R1 tool_intent. Maps R1 names to internal names.
pub fn r1_tool_to_internal(r1_name: &str) -> Option<&'static str> {
    Some(match r1_name {
        "read_file" | "fs_read" => "fs.read",
        "write_file" | "fs_write" => "fs.write",
        "edit_file" | "fs_edit" => "fs.edit",
        "list_dir" | "fs_list" => "fs.list",
        "glob" | "fs_glob" => "fs.glob",
        "ripgrep" | "grep" | "fs_grep" => "fs.grep",
        "run_cmd" | "bash_run" | "bash" => "bash.run",
        "git_status" => "git.status",
        "git_diff" => "git.diff",
        "git_show" => "git.show",
        "apply_patch" | "patch_apply" => "patch.apply",
        "multi_edit" => "multi_edit",
        "diagnostics_check" | "diagnostics" => "diagnostics.check",
        "index_query" | "search" => "index.query",
        _ => return None,
    })
}

// ── JSON extraction from freeform text ──────────────────────────────────

/// Extract the first valid JSON object from freeform text.
///
/// R1 may wrap its JSON in markdown code fences, add commentary before/after,
/// or emit it inline. This function robustly finds the first `{...}` block
/// that parses as valid JSON.
pub fn extract_json_object(text: &str) -> Option<&str> {
    // Strip common markdown fences first
    let cleaned = strip_code_fences(text);

    // Find the first `{` and try progressively larger substrings
    let bytes = cleaned.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            // Try to find matching closing brace with nesting
            if let Some(end) = find_matching_brace(cleaned, i) {
                let candidate = &cleaned[i..=end];
                // Validate it's actual JSON
                if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                    return Some(candidate);
                }
            }
        }
        i += 1;
    }
    None
}

/// Strip markdown code fences (```json ... ``` or ``` ... ```)
fn strip_code_fences(text: &str) -> &str {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix("```json") {
        if let Some(inner) = rest.strip_suffix("```") {
            return inner.trim();
        }
    }
    if let Some(rest) = trimmed.strip_prefix("```") {
        if let Some(inner) = rest.strip_suffix("```") {
            return inner.trim();
        }
    }
    trimmed
}

/// Find the index of the closing `}` that matches the `{` at `start`.
fn find_matching_brace(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if escape_next {
            escape_next = false;
            continue;
        }
        if b == b'\\' && in_string {
            escape_next = true;
            continue;
        }
        if b == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

// ── R1 response parsing ─────────────────────────────────────────────────

/// Parse and validate an R1 response from freeform text.
///
/// Returns the parsed response and the raw JSON string that was extracted.
pub fn parse_r1_response(text: &str) -> Result<(R1Response, String), R1ParseError> {
    let json_str = extract_json_object(text).ok_or(R1ParseError::NoJsonFound)?;

    // First check that "type" field exists
    let raw: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| R1ParseError::InvalidJson(e.to_string()))?;

    let type_field = raw
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or(R1ParseError::MissingTypeField)?;

    // Validate against known types
    match type_field {
        "tool_intent" | "delegate_patch" | "done" | "abort" => {}
        other => return Err(R1ParseError::UnknownType(other.to_string())),
    }

    // Deserialize with full schema validation
    let response: R1Response =
        serde_json::from_str(json_str).map_err(|e| R1ParseError::SchemaViolation(e.to_string()))?;

    // Additional semantic validation
    validate_r1_response(&response)?;

    Ok((response, json_str.to_string()))
}

/// Semantic validation beyond JSON schema.
fn validate_r1_response(response: &R1Response) -> Result<(), R1ParseError> {
    match response {
        R1Response::ToolIntent(ti) => {
            if ti.step_id.is_empty() {
                return Err(R1ParseError::SchemaViolation(
                    "tool_intent.step_id is empty".into(),
                ));
            }
            if ti.tool.is_empty() {
                return Err(R1ParseError::SchemaViolation(
                    "tool_intent.tool is empty".into(),
                ));
            }
            // Validate tool name is known
            if r1_tool_to_internal(&ti.tool).is_none() {
                return Err(R1ParseError::UnknownTool(ti.tool.clone()));
            }
        }
        R1Response::DelegatePatch(dp) => {
            if dp.step_id.is_empty() {
                return Err(R1ParseError::SchemaViolation(
                    "delegate_patch.step_id is empty".into(),
                ));
            }
            if dp.task.is_empty() {
                return Err(R1ParseError::SchemaViolation(
                    "delegate_patch.task is empty".into(),
                ));
            }
        }
        R1Response::Done(d) => {
            if d.summary.is_empty() {
                return Err(R1ParseError::SchemaViolation(
                    "done.summary is empty".into(),
                ));
            }
        }
        R1Response::Abort(a) => {
            if a.reason.is_empty() {
                return Err(R1ParseError::SchemaViolation(
                    "abort.reason is empty".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Errors from R1 response parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum R1ParseError {
    /// No JSON object found in text.
    NoJsonFound,
    /// JSON found but invalid syntax.
    InvalidJson(String),
    /// Missing required "type" field.
    MissingTypeField,
    /// Unknown "type" value.
    UnknownType(String),
    /// JSON doesn't match expected schema.
    SchemaViolation(String),
    /// Tool name not in allowed set.
    UnknownTool(String),
}

impl std::fmt::Display for R1ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoJsonFound => write!(f, "no JSON object found in R1 output"),
            Self::InvalidJson(e) => write!(f, "invalid JSON: {e}"),
            Self::MissingTypeField => write!(f, "missing 'type' field in R1 JSON"),
            Self::UnknownType(t) => write!(f, "unknown R1 response type: {t}"),
            Self::SchemaViolation(e) => write!(f, "schema violation: {e}"),
            Self::UnknownTool(t) => write!(f, "unknown tool in tool_intent: {t}"),
        }
    }
}

// ── V3 patch response parsing ───────────────────────────────────────────

/// Parse V3 patch-only response.
///
/// V3 should return either a unified diff or `{"type":"need_more_context",...}`.
pub fn parse_v3_patch_response(text: &str) -> V3PatchResponse {
    // Check for need_more_context JSON first
    if let Some(json_str) = extract_json_object(text) {
        if let Ok(nmc) = serde_json::from_str::<NeedMoreContext>(json_str) {
            if nmc.response_type == "need_more_context" {
                return V3PatchResponse::NeedMoreContext(nmc);
            }
        }
    }

    // Otherwise treat entire output as unified diff
    let diff = extract_unified_diff(text);
    V3PatchResponse::UnifiedDiff(diff)
}

/// Extract unified diff content, stripping markdown fences if present.
fn extract_unified_diff(text: &str) -> String {
    let trimmed = text.trim();

    // Try to extract from ```diff ... ``` fences
    if let Some(rest) = trimmed.strip_prefix("```diff") {
        if let Some(inner) = rest.strip_suffix("```") {
            return inner.trim().to_string();
        }
    }
    if let Some(rest) = trimmed.strip_prefix("```") {
        if let Some(inner) = rest.strip_suffix("```") {
            // Only use if it looks like a diff
            let inner = inner.trim();
            if inner.starts_with("---") || inner.starts_with("diff ") {
                return inner.to_string();
            }
        }
    }

    // Return as-is if it looks like a diff
    if trimmed.starts_with("---")
        || trimmed.starts_with("diff ")
        || trimmed.contains("\n--- ")
        || trimmed.contains("\n+++ ")
    {
        return trimmed.to_string();
    }

    // Last resort: return the raw text
    trimmed.to_string()
}

/// Build the retry prompt when R1 output fails validation.
pub fn r1_retry_prompt(error: &R1ParseError) -> String {
    format!(
        "Your previous response could not be parsed: {error}\n\n\
         You MUST respond with EXACTLY ONE valid JSON object matching one of these types:\n\
         - {{\"type\":\"tool_intent\", \"step_id\":\"S#\", \"tool\":\"...\", \"args\":{{...}}, \"why\":\"...\"}}\n\
         - {{\"type\":\"delegate_patch\", \"step_id\":\"S#\", \"task\":\"...\", \"constraints\":[...], \"acceptance\":[...], \
           \"inputs\":{{\"relevant_files\":[...]}}}}\n\
         - {{\"type\":\"done\", \"summary\":\"...\"}}\n\
         - {{\"type\":\"abort\", \"reason\":\"...\"}}\n\n\
         Emit the JSON object now with no other text."
    )
}

/// Build the retry prompt when V3 patch output is invalid.
pub fn v3_patch_retry_prompt() -> &'static str {
    "Your previous response was not a valid unified diff.\n\n\
     You MUST respond with ONLY a unified diff (no explanation, no JSON, no markdown).\n\
     Format:\n\
     --- a/path/to/file\n\
     +++ b/path/to/file\n\
     @@ -start,count +start,count @@\n\
     -removed line\n\
     +added line\n\
      context line\n\n\
     OR if you need more context:\n\
     {\"type\":\"need_more_context\", \"missing\":[\"path/to/file:10-50\", ...]}\n\n\
     Emit the diff or need_more_context JSON now."
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_from_clean_text() {
        let text = r#"{"type":"done","summary":"All tests pass."}"#;
        let json = extract_json_object(text).unwrap();
        assert_eq!(json, text);
    }

    #[test]
    fn extract_json_from_markdown_fence() {
        let text =
            "Here is my response:\n```json\n{\"type\":\"done\",\"summary\":\"ok\"}\n```\nEnd.";
        let json = extract_json_object(text).unwrap();
        assert!(json.contains("\"type\":\"done\""));
    }

    #[test]
    fn extract_json_with_surrounding_text() {
        let text = "I'll read the file first.\n{\"type\":\"tool_intent\",\"step_id\":\"S1\",\"tool\":\"read_file\",\"args\":{\"file_path\":\"src/main.rs\"},\"why\":\"inspect\"}\nThat should help.";
        let json = extract_json_object(text).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(parsed["type"], "tool_intent");
        assert_eq!(parsed["tool"], "read_file");
    }

    #[test]
    fn extract_json_with_nested_objects() {
        let text = r#"{"type":"tool_intent","step_id":"S1","tool":"ripgrep","args":{"pattern":"fn main","glob":"**/*.rs"},"why":"find entry point"}"#;
        let json = extract_json_object(text).unwrap();
        let parsed: R1Response = serde_json::from_str(json).unwrap();
        assert!(matches!(parsed, R1Response::ToolIntent(_)));
    }

    #[test]
    fn extract_json_returns_none_for_no_json() {
        assert!(extract_json_object("just plain text").is_none());
        assert!(extract_json_object("").is_none());
        assert!(extract_json_object("{unclosed").is_none());
    }

    #[test]
    fn extract_json_handles_strings_with_braces() {
        let text = r#"{"type":"done","summary":"Fixed {formatting} issue in {file}"}"#;
        let json = extract_json_object(text).unwrap();
        let parsed: R1Response = serde_json::from_str(json).unwrap();
        if let R1Response::Done(d) = parsed {
            assert!(d.summary.contains("{formatting}"));
        } else {
            panic!("expected Done");
        }
    }

    #[test]
    fn parse_tool_intent_valid() {
        let text = r#"{"type":"tool_intent","step_id":"S1","tool":"read_file","args":{"file_path":"src/lib.rs"},"why":"read the module","expected":"source code"}"#;
        let (resp, _raw) = parse_r1_response(text).unwrap();
        if let R1Response::ToolIntent(ti) = resp {
            assert_eq!(ti.step_id, "S1");
            assert_eq!(ti.tool, "read_file");
            assert_eq!(ti.args["file_path"], "src/lib.rs");
            assert!(!ti.verify_after);
        } else {
            panic!("expected ToolIntent");
        }
    }

    #[test]
    fn parse_delegate_patch_valid() {
        let text = r#"
        {
            "type": "delegate_patch",
            "step_id": "S3",
            "task": "Add error handling to parse_config",
            "constraints": ["touch only src/config.rs", "no API changes"],
            "acceptance": ["cargo test -p config", "cargo clippy"],
            "inputs": {
                "relevant_files": ["src/config.rs", "src/error.rs"],
                "context_refs": []
            },
            "verify_after": true
        }
        "#;
        let (resp, _raw) = parse_r1_response(text).unwrap();
        if let R1Response::DelegatePatch(dp) = resp {
            assert_eq!(dp.step_id, "S3");
            assert_eq!(dp.constraints.len(), 2);
            assert_eq!(dp.acceptance.len(), 2);
            assert!(dp.verify_after);
        } else {
            panic!("expected DelegatePatch");
        }
    }

    #[test]
    fn parse_done_valid() {
        let text = r#"{"type":"done","summary":"Refactored auth module.","verification":"cargo test passes"}"#;
        let (resp, _) = parse_r1_response(text).unwrap();
        assert!(matches!(resp, R1Response::Done(DoneResponse { .. })));
    }

    #[test]
    fn parse_abort_valid() {
        let text =
            r#"{"type":"abort","reason":"Cannot modify generated code without schema updates."}"#;
        let (resp, _) = parse_r1_response(text).unwrap();
        assert!(matches!(resp, R1Response::Abort(AbortResponse { .. })));
    }

    #[test]
    fn parse_rejects_unknown_type() {
        let text = r#"{"type":"explode","data":"boom"}"#;
        let err = parse_r1_response(text).unwrap_err();
        assert!(matches!(err, R1ParseError::UnknownType(_)));
    }

    #[test]
    fn parse_rejects_missing_type() {
        let text = r#"{"tool":"read_file","args":{}}"#;
        let err = parse_r1_response(text).unwrap_err();
        assert!(matches!(err, R1ParseError::MissingTypeField));
    }

    #[test]
    fn parse_rejects_unknown_tool() {
        let text =
            r#"{"type":"tool_intent","step_id":"S1","tool":"hack_server","args":{},"why":"test"}"#;
        let err = parse_r1_response(text).unwrap_err();
        assert!(matches!(err, R1ParseError::UnknownTool(_)));
    }

    #[test]
    fn parse_rejects_empty_step_id() {
        let text =
            r#"{"type":"tool_intent","step_id":"","tool":"read_file","args":{},"why":"test"}"#;
        let err = parse_r1_response(text).unwrap_err();
        assert!(matches!(err, R1ParseError::SchemaViolation(_)));
    }

    #[test]
    fn parse_rejects_empty_summary() {
        let text = r#"{"type":"done","summary":""}"#;
        let err = parse_r1_response(text).unwrap_err();
        assert!(matches!(err, R1ParseError::SchemaViolation(_)));
    }

    #[test]
    fn r1_tool_mapping_covers_common_names() {
        assert_eq!(r1_tool_to_internal("read_file"), Some("fs.read"));
        assert_eq!(r1_tool_to_internal("ripgrep"), Some("fs.grep"));
        assert_eq!(r1_tool_to_internal("run_cmd"), Some("bash.run"));
        assert_eq!(r1_tool_to_internal("apply_patch"), Some("patch.apply"));
        assert_eq!(r1_tool_to_internal("glob"), Some("fs.glob"));
        assert_eq!(r1_tool_to_internal("search"), Some("index.query"));
        assert_eq!(r1_tool_to_internal("nonexistent"), None);
    }

    #[test]
    fn v3_patch_response_parses_diff() {
        let diff = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1,3 +1,4 @@\n fn main() {\n+    println!(\"hello\");\n }\n";
        let resp = parse_v3_patch_response(diff);
        assert!(matches!(resp, V3PatchResponse::UnifiedDiff(_)));
    }

    #[test]
    fn v3_patch_response_parses_fenced_diff() {
        let text = "```diff\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-old\n+new\n```";
        let resp = parse_v3_patch_response(text);
        if let V3PatchResponse::UnifiedDiff(d) = resp {
            assert!(d.starts_with("--- a/src/lib.rs"));
        } else {
            panic!("expected UnifiedDiff");
        }
    }

    #[test]
    fn v3_patch_response_parses_need_more_context() {
        let text =
            r#"{"type":"need_more_context","missing":["src/auth.rs:10-50","src/config.rs"]}"#;
        let resp = parse_v3_patch_response(text);
        if let V3PatchResponse::NeedMoreContext(nmc) = resp {
            assert_eq!(nmc.missing.len(), 2);
        } else {
            panic!("expected NeedMoreContext");
        }
    }

    #[test]
    fn retry_prompt_contains_error_info() {
        let prompt = r1_retry_prompt(&R1ParseError::NoJsonFound);
        assert!(prompt.contains("no JSON object found"));
        assert!(prompt.contains("tool_intent"));
        assert!(prompt.contains("delegate_patch"));
    }

    #[test]
    fn parse_from_chatty_r1_output() {
        let text = "Let me analyze the situation.\n\nThe error is in the config parser. I need to read the file first.\n\n```json\n{\"type\":\"tool_intent\",\"step_id\":\"S2\",\"tool\":\"read_file\",\"args\":{\"file_path\":\"src/config.rs\"},\"why\":\"inspect config parser\",\"expected\":\"source with parse function\"}\n```\n\nThis will help me understand the issue.";
        let (resp, _) = parse_r1_response(text).unwrap();
        assert!(matches!(resp, R1Response::ToolIntent(_)));
    }
}
