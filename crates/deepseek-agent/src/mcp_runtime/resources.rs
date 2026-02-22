use anyhow::{Context as _, Result, anyhow};
use serde_json::json;
use std::io::{BufRead, Write};
use std::thread;

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
    let command = server
        .command
        .as_deref()
        .ok_or_else(|| anyhow!("stdio MCP server has no command"))?;
    let mut child = std::process::Command::new(command)
        .args(&server.args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("failed to start MCP server: {command}"))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("failed to open MCP stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("failed to open MCP stdout"))?;

    let (tx, rx) = std::sync::mpsc::channel::<serde_json::Value>();
    let reader_handle = thread::spawn(move || {
        let mut reader = std::io::BufReader::new(stdout);
        while let Ok(Some(value)) = read_mcp_frame(&mut reader) {
            if tx.send(value).is_err() {
                break;
            }
        }
    });

    // Initialize
    let init_params = json!({
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": { "name": "deepseek-cli", "version": "0.2.0" }
    });
    write_mcp_request(&mut stdin, 1, "initialize", init_params)?;

    // Call the tool
    let call_params = json!({
        "name": tool_name,
        "arguments": args.get("arguments").unwrap_or(args),
    });
    write_mcp_request(&mut stdin, 2, "tools/call", call_params)?;
    let _ = stdin.flush();
    drop(stdin);

    // Read responses — look for the tools/call result (id=2)
    let mut result = None;
    for _ in 0..40 {
        match rx.recv_timeout(std::time::Duration::from_millis(150)) {
            Ok(message) => {
                if message.get("id").and_then(|v| v.as_u64()) == Some(2) {
                    result = Some(message);
                    break;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    let _ = child.kill();
    let _ = child.wait();
    let _ = reader_handle.join();

    match result {
        Some(resp) => extract_mcp_call_result(&resp),
        None => Err(anyhow!("MCP tool call timed out")),
    }
}

/// Write a JSON-RPC request with Content-Length framing (MCP protocol).
pub(crate) fn write_mcp_request(
    writer: &mut dyn std::io::Write,
    id: u64,
    method: &str,
    params: serde_json::Value,
) -> Result<()> {
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params
    }))?;
    write!(writer, "Content-Length: {}\r\n\r\n", body.len())?;
    writer.write_all(&body)?;
    writer.flush()?;
    Ok(())
}

/// Read a Content-Length-framed JSON-RPC message.
pub(crate) fn read_mcp_frame<R: BufRead>(reader: &mut R) -> Result<Option<serde_json::Value>> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            return Ok(None);
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some((header, value)) = trimmed.split_once(':')
            && header.eq_ignore_ascii_case("Content-Length")
        {
            content_length = Some(value.trim().parse::<usize>()?);
        }
    }

    let length = content_length.ok_or_else(|| anyhow!("missing Content-Length header"))?;
    let mut buf = vec![0_u8; length];
    std::io::Read::read_exact(reader, &mut buf)?;
    Ok(Some(serde_json::from_slice(&buf)?))
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
