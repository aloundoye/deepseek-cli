//! Bridge between LLM tool call responses and the internal tool execution system.
//!
//! Converts `LlmToolCall` (API-facing, underscored names like `fs_read`) to/from
//! `ToolCall` (internal, dotted names like `fs.read`) and formats `ToolResult`
//! back into `ChatMessage::Tool` for feeding to the next LLM turn.

use deepseek_core::{ChatMessage, LlmToolCall, ToolCall, ToolName, ToolResult};
use deepseek_policy::output_scanner::{InjectionWarning, OutputScanner};

/// Maximum characters for tool output before truncation.
pub const MAX_TOOL_OUTPUT_CHARS: usize = 25_000;

/// Maximum characters for MCP tool output before truncation (~25K tokens).
pub const MCP_MAX_OUTPUT_CHARS: usize = 100_000;

/// Convert an `LlmToolCall` from the API into an internal `ToolCall`.
///
/// Translates the API name (`fs_read`) to the internal dotted name (`fs.read`),
/// parses the arguments JSON string, and determines whether approval is required
/// based on the tool type (read-only tools don't need approval).
pub fn llm_tool_call_to_internal(call: &LlmToolCall) -> ToolCall {
    // Translate API name to internal name. For unknown tools (MCP, plugins),
    // use the API name as-is since they bypass the ToolName enum.
    let internal_name = ToolName::from_api_name(&call.name)
        .map(|tn| tn.as_internal().to_string())
        .unwrap_or_else(|| call.name.clone());

    // Parse arguments from JSON string. If parsing fails, pass empty object.
    let args: serde_json::Value =
        serde_json::from_str(&call.arguments).unwrap_or_else(|_| serde_json::json!({}));

    // Determine approval requirement: read-only tools don't need it.
    let requires_approval = ToolName::from_api_name(&call.name)
        .map(|tn| !tn.is_read_only())
        .unwrap_or(true); // Unknown tools (MCP, plugins) require approval by default

    ToolCall {
        name: internal_name,
        args,
        requires_approval,
    }
}

/// Convert a `ToolResult` into a `ChatMessage::Tool` for inclusion in the
/// conversation history sent to the LLM.
///
/// Truncates large outputs to `MAX_TOOL_OUTPUT_CHARS` (or `MCP_MAX_OUTPUT_CHARS`
/// for MCP tools) to prevent context overflow. When a scanner is provided,
/// secrets are redacted and injection warnings are returned.
pub fn tool_result_to_message(
    tool_call_id: &str,
    tool_name: &str,
    result: &ToolResult,
    scanner: Option<&OutputScanner>,
) -> (ChatMessage, Vec<InjectionWarning>) {
    let raw = format_tool_output(&result.output, result.success);

    // Use higher limit for MCP tools
    let max_chars = if tool_name.starts_with("mcp__") {
        MCP_MAX_OUTPUT_CHARS
    } else {
        MAX_TOOL_OUTPUT_CHARS
    };
    let mut content = truncate_output(&raw, max_chars);

    // Apply security scanning if available
    let warnings = if let Some(s) = scanner {
        let scan = s.scan(&content);
        content = scan.redacted_output;
        scan.injection_warnings
    } else {
        Vec::new()
    };

    let msg = ChatMessage::Tool {
        tool_call_id: tool_call_id.to_string(),
        content,
    };
    (msg, warnings)
}

/// Format a tool error as a `ChatMessage::Tool` with error content.
pub fn tool_error_to_message(tool_call_id: &str, error: &str) -> ChatMessage {
    ChatMessage::Tool {
        tool_call_id: tool_call_id.to_string(),
        content: format!("Error: {error}"),
    }
}

/// Whether a tool name (API-format) is a write/destructive tool that warrants
/// creating a checkpoint before execution.
pub fn is_write_tool(api_name: &str) -> bool {
    matches!(
        api_name,
        "fs_edit" | "fs_write" | "bash_run" | "multi_edit" | "patch_apply" | "notebook_edit"
    )
}

/// Extract the target file path(s) from a write tool's arguments JSON string.
/// Returns file paths that the tool will modify, for targeted checkpointing.
pub fn extract_modified_paths(api_name: &str, args_json: &str) -> Vec<std::path::PathBuf> {
    let Ok(args) = serde_json::from_str::<serde_json::Value>(args_json) else {
        return Vec::new();
    };
    let mut paths = Vec::new();
    match api_name {
        "fs_edit" | "fs_write" | "notebook_edit" => {
            // These tools have a "path" or "file_path" argument
            for key in &["path", "file_path"] {
                if let Some(p) = args.get(key).and_then(|v| v.as_str()) {
                    paths.push(std::path::PathBuf::from(p));
                }
            }
        }
        "multi_edit" => {
            // multi_edit has an array of edits, each with a "path"
            if let Some(edits) = args.get("edits").and_then(|v| v.as_array()) {
                for edit in edits {
                    if let Some(p) = edit.get("path").and_then(|v| v.as_str()) {
                        paths.push(std::path::PathBuf::from(p));
                    }
                }
            }
        }
        "patch_apply" => {
            if let Some(p) = args.get("path").and_then(|v| v.as_str()) {
                paths.push(std::path::PathBuf::from(p));
            }
        }
        // bash_run — can't reliably determine modified files
        _ => {}
    }
    paths
}

/// Whether a tool name (API-format) is agent-level, meaning it should be
/// handled by the AgentEngine itself rather than dispatched to LocalToolHost.
pub fn is_agent_level_tool(api_name: &str) -> bool {
    ToolName::from_api_name(api_name)
        .map(|tn| tn.is_agent_level())
        .unwrap_or(false)
}

/// Format tool output for the LLM. Extracts text from JSON structures.
fn format_tool_output(output: &serde_json::Value, success: bool) -> String {
    if !success {
        if let Some(err) = output.get("error").and_then(|v| v.as_str()) {
            return format!("Error: {err}");
        }
        return format!("Error: {output}");
    }

    // Try to extract meaningful text from common output shapes
    if let Some(s) = output.as_str() {
        return s.to_string();
    }
    if let Some(content) = output.get("content").and_then(|v| v.as_str()) {
        return content.to_string();
    }
    if let Some(output_str) = output.get("output").and_then(|v| v.as_str()) {
        return output_str.to_string();
    }

    // Fallback: pretty-print the JSON
    serde_json::to_string_pretty(output).unwrap_or_else(|_| output.to_string())
}

/// Truncate output to max chars, appending a notice if truncated.
fn truncate_output(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    // Find a safe UTF-8 boundary
    let boundary = text
        .char_indices()
        .take_while(|(i, _)| *i < max_chars.saturating_sub(80))
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    let truncated = &text[..boundary];
    format!(
        "{truncated}\n\n[Output truncated: showing {boundary}/{} chars. Use more specific queries to reduce output size.]",
        text.len()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_fs_read_tool_call() {
        let llm_call = LlmToolCall {
            id: "call_1".to_string(),
            name: "fs_read".to_string(),
            arguments: r#"{"path":"src/lib.rs"}"#.to_string(),
        };
        let tc = llm_tool_call_to_internal(&llm_call);
        assert_eq!(tc.name, "fs.read");
        assert_eq!(tc.args["path"], "src/lib.rs");
        assert!(!tc.requires_approval, "fs.read is read-only");
    }

    #[test]
    fn convert_bash_run_requires_approval() {
        let llm_call = LlmToolCall {
            id: "call_2".to_string(),
            name: "bash_run".to_string(),
            arguments: r#"{"command":"rm -rf /"}"#.to_string(),
        };
        let tc = llm_tool_call_to_internal(&llm_call);
        assert_eq!(tc.name, "bash.run");
        assert!(tc.requires_approval, "bash.run needs approval");
    }

    #[test]
    fn convert_unknown_mcp_tool() {
        let llm_call = LlmToolCall {
            id: "call_3".to_string(),
            name: "mcp__github__search".to_string(),
            arguments: r#"{"query":"test"}"#.to_string(),
        };
        let tc = llm_tool_call_to_internal(&llm_call);
        assert_eq!(tc.name, "mcp__github__search", "MCP tools keep API name");
        assert!(tc.requires_approval, "unknown tools require approval");
    }

    #[test]
    fn convert_tool_result_to_message_success() {
        let result = ToolResult {
            invocation_id: uuid::Uuid::nil(),
            success: true,
            output: serde_json::json!({"content": "file contents here"}),
        };
        let (msg, warnings) = tool_result_to_message("call_1", "fs_read", &result, None);
        assert!(warnings.is_empty());
        match msg {
            ChatMessage::Tool {
                tool_call_id,
                content,
            } => {
                assert_eq!(tool_call_id, "call_1");
                assert_eq!(content, "file contents here");
            }
            _ => panic!("expected Tool message"),
        }
    }

    #[test]
    fn convert_tool_result_to_message_error() {
        let result = ToolResult {
            invocation_id: uuid::Uuid::nil(),
            success: false,
            output: serde_json::json!({"error": "file not found"}),
        };
        let (msg, _) = tool_result_to_message("call_2", "fs_read", &result, None);
        match msg {
            ChatMessage::Tool { content, .. } => {
                assert!(content.contains("Error: file not found"));
            }
            _ => panic!("expected Tool message"),
        }
    }

    #[test]
    fn truncate_large_tool_output() {
        let big = "x".repeat(30_000);
        let result = ToolResult {
            invocation_id: uuid::Uuid::nil(),
            success: true,
            output: serde_json::json!(big),
        };
        let (msg, _) = tool_result_to_message("call_4", "fs_read", &result, None);
        match msg {
            ChatMessage::Tool { content, .. } => {
                assert!(content.len() < 26_000);
                assert!(content.contains("[Output truncated"));
            }
            _ => panic!("expected Tool message"),
        }
    }

    #[test]
    fn mcp_output_truncated_at_limit() {
        let big = "x".repeat(120_000);
        let result = ToolResult {
            invocation_id: uuid::Uuid::nil(),
            success: true,
            output: serde_json::json!(big),
        };
        let (msg, _) = tool_result_to_message("call_mcp", "mcp__github__search", &result, None);
        match msg {
            ChatMessage::Tool { content, .. } => {
                assert!(
                    content.len() < 101_000,
                    "MCP output should truncate around 100K"
                );
                assert!(content.contains("[Output truncated"));
            }
            _ => panic!("expected Tool message"),
        }
    }

    #[test]
    fn agent_level_tools_identified() {
        assert!(is_agent_level_tool("user_question"));
        assert!(is_agent_level_tool("spawn_task"));
        assert!(is_agent_level_tool("skill"));
        assert!(is_agent_level_tool("enter_plan_mode"));
        assert!(!is_agent_level_tool("fs_read"));
        assert!(!is_agent_level_tool("bash_run"));
        assert!(!is_agent_level_tool("mcp__something"));
    }

    #[test]
    fn tool_error_to_message_formats() {
        let msg = tool_error_to_message("call_5", "permission denied");
        match msg {
            ChatMessage::Tool { content, .. } => {
                assert_eq!(content, "Error: permission denied");
            }
            _ => panic!("expected Tool message"),
        }
    }

    #[test]
    fn is_write_tool_identifies_correctly() {
        assert!(super::is_write_tool("fs_edit"));
        assert!(super::is_write_tool("fs_write"));
        assert!(super::is_write_tool("bash_run"));
        assert!(super::is_write_tool("multi_edit"));
        assert!(super::is_write_tool("patch_apply"));
        assert!(super::is_write_tool("notebook_edit"));
        assert!(!super::is_write_tool("fs_read"));
        assert!(!super::is_write_tool("fs_glob"));
        assert!(!super::is_write_tool("git_status"));
        assert!(!super::is_write_tool("user_question"));
    }

    #[test]
    fn invalid_json_arguments_handled() {
        let llm_call = LlmToolCall {
            id: "call_6".to_string(),
            name: "fs_read".to_string(),
            arguments: "not valid json".to_string(),
        };
        let tc = llm_tool_call_to_internal(&llm_call);
        assert_eq!(tc.args, serde_json::json!({}));
    }

    // ── P4-04: extract_modified_paths tests ─────────────────────────────

    #[test]
    fn extract_paths_from_fs_edit() {
        let paths = extract_modified_paths("fs_edit", r#"{"file_path": "/tmp/foo.rs"}"#);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], std::path::PathBuf::from("/tmp/foo.rs"));
    }

    #[test]
    fn extract_paths_from_multi_edit() {
        let paths = extract_modified_paths(
            "multi_edit",
            r#"{"edits": [{"path": "a.rs"}, {"path": "b.rs"}]}"#,
        );
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn extract_paths_bash_returns_empty() {
        let paths = extract_modified_paths("bash_run", r#"{"command": "rm -rf /"}"#);
        assert!(paths.is_empty());
    }

    #[test]
    fn extract_paths_invalid_json() {
        let paths = extract_modified_paths("fs_edit", "not json");
        assert!(paths.is_empty());
    }
}
