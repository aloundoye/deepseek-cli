use anyhow::{Result, anyhow};
use serde_json::json;

/// Convert an MCP tool to an OpenAI-compatible ToolDefinition.
/// Uses `mcp__<server_id>__<tool_name>` naming convention.
pub(crate) fn mcp_tool_to_definition(
    tool: &deepseek_mcp::McpTool,
) -> deepseek_core::ToolDefinition {
    let api_name = format!("mcp__{}_{}", tool.server_id, tool.name)
        .replace('-', "_")
        .replace('.', "_");
    deepseek_core::ToolDefinition {
        tool_type: "function".to_string(),
        function: deepseek_core::FunctionDefinition {
            name: api_name,
            description: format!(
                "[MCP: {}] {}",
                tool.server_id,
                if tool.description.is_empty() {
                    &tool.name
                } else {
                    &tool.description
                }
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "arguments": {
                        "type": "object",
                        "description": "Arguments to pass to the MCP tool"
                    }
                },
                "required": []
            }),
        },
    }
}

/// Parse an MCP tool name (`mcp__server__tool`) into `(server_id, tool_name)`.
pub(crate) fn parse_mcp_tool_name(name: &str) -> Option<(String, String)> {
    let stripped = name.strip_prefix("mcp__")?;
    let underscore_pos = stripped.find('_')?;
    let server_id = stripped[..underscore_pos].replace('_', "-");
    let tool_name = stripped[underscore_pos + 1..].replace('_', ".");
    Some((server_id, tool_name))
}

/// Extract the text result from an MCP JSON-RPC `tools/call` response.
pub(crate) fn extract_mcp_call_result(response: &serde_json::Value) -> Result<String> {
    // Check for JSON-RPC error
    if let Some(error) = response.get("error") {
        let msg = error
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error");
        return Err(anyhow!("MCP error: {msg}"));
    }

    // Extract result content
    let result = response.get("result").unwrap_or(response);

    // MCP tools/call returns {content: [{type: "text", text: "..."}]}
    if let Some(content) = result.get("content").and_then(|v| v.as_array()) {
        let texts: Vec<&str> = content
            .iter()
            .filter_map(|item| {
                if item.get("type").and_then(|v| v.as_str()) == Some("text") {
                    item.get("text").and_then(|v| v.as_str())
                } else {
                    None
                }
            })
            .collect();
        if !texts.is_empty() {
            return Ok(texts.join("\n"));
        }
    }

    // Fallback: stringify the result
    Ok(serde_json::to_string_pretty(result).unwrap_or_else(|_| "{}".to_string()))
}

/// Execute a tool call on an MCP stdio server.
pub(crate) fn execute_mcp_stdio_tool(
    server: &deepseek_mcp::McpServer,
    tool_name: &str,
    args: &serde_json::Value,
) -> Result<String> {
    let call_params = json!({
        "name": tool_name,
        "arguments": args.get("arguments").unwrap_or(args),
    });
    let response = deepseek_mcp::execute_mcp_stdio_request(
        server,
        "tools/call",
        call_params,
        2,
        std::time::Duration::from_secs(6),
    )?;
    extract_mcp_call_result(&response)
}

/// Build ToolDefinition for the `mcp_search` meta-tool (lazy loading).
pub(crate) fn mcp_search_tool_definition() -> deepseek_core::ToolDefinition {
    deepseek_core::ToolDefinition {
        tool_type: "function".to_string(),
        function: deepseek_core::FunctionDefinition {
            name: "mcp_search".to_string(),
            description: "Search for available MCP tools by description or name. Use this when you need a capability from an external MCP server but don't know which tool to use.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find matching MCP tools"
                    }
                },
                "required": ["query"]
            }),
        },
    }
}
