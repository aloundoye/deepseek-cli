use crate::*;

impl AgentEngine {
    pub(crate) fn is_mcp_tool(name: &str) -> bool {
        name.starts_with("mcp__")
    }

    /// Execute an MCP tool call by routing to the appropriate server via JSON-RPC.
    pub(crate) fn execute_mcp_tool(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Result<String> {
        let (server_id, mcp_tool_name) = parse_mcp_tool_name(tool_name)
            .ok_or_else(|| anyhow!("invalid MCP tool name: {tool_name}"))?;

        let Some(ref mcp) = self.mcp else {
            return Err(anyhow!("MCP manager not available"));
        };

        let server = mcp
            .get_server(&server_id)?
            .ok_or_else(|| anyhow!("MCP server not found: {server_id}"))?;

        match server.transport {
            deepseek_mcp::McpTransport::Http => {
                let url = server
                    .url
                    .as_deref()
                    .ok_or_else(|| anyhow!("HTTP MCP server has no URL"))?;
                let client = reqwest::blocking::Client::builder()
                    .timeout(std::time::Duration::from_secs(30))
                    .build()?;

                // Add OAuth token if available
                let mut request = client.post(url).json(&serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": mcp_tool_name,
                        "arguments": args,
                    }
                }));
                if let Ok(Some(token)) = deepseek_mcp::load_mcp_token(&server_id) {
                    request = request.bearer_auth(token.access_token);
                }

                let response: serde_json::Value = request.send()?.error_for_status()?.json()?;
                extract_mcp_call_result(&response)
            }
            deepseek_mcp::McpTransport::Stdio => {
                execute_mcp_stdio_tool(&server, &mcp_tool_name, args)
            }
        }
    }

    /// Resolve `@server:uri` references in a prompt string into inline content.
    pub(crate) fn resolve_mcp_resources(&self, prompt: &str) -> String {
        let Some(ref mcp) = self.mcp else {
            return prompt.to_string();
        };
        let session_id = self
            .store
            .load_latest_session()
            .ok()
            .flatten()
            .map(|session| session.session_id);
        let mut result = String::with_capacity(prompt.len());
        for token in prompt.split_whitespace() {
            if !result.is_empty() {
                result.push(' ');
            }
            // Match @server:protocol://path pattern
            if token.starts_with('@')
                && token.len() > 2
                && token[1..].contains(':')
                && !token[1..].starts_with('/')
            {
                match mcp.resolve_resource(token) {
                    Ok(content) => {
                        if let Some(session_id) = session_id {
                            let _ = self.emit(
                                session_id,
                                EventKind::TelemetryEventV1 {
                                    name: "kpi.mcp.resource_resolve".to_string(),
                                    properties: json!({
                                        "status": "success",
                                        "reference": token,
                                    }),
                                },
                            );
                        }
                        result.push_str(&format!(
                            "[resource: {}]\n{}\n[/resource]",
                            &token[1..],
                            content.trim()
                        ));
                    }
                    Err(err) => {
                        if let Some(session_id) = session_id {
                            let _ = self.emit(
                                session_id,
                                EventKind::TelemetryEventV1 {
                                    name: "kpi.mcp.resource_resolve".to_string(),
                                    properties: json!({
                                        "status": "failure",
                                        "reference": token,
                                        "error": err.to_string(),
                                    }),
                                },
                            );
                        }
                        let mut reason = err.to_string().replace('\n', " ");
                        if reason.len() > 160 {
                            reason.truncate(160);
                            reason.push_str("...");
                        }
                        result.push_str(&format!("[resource-unavailable: {} ({})]", token, reason));
                    }
                }
                continue;
            }
            result.push_str(token);
        }
        result
    }
}
