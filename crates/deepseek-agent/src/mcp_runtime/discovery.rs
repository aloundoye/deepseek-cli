use crate::*;

impl AgentEngine {
    pub(crate) fn discover_mcp_tool_definitions(&self) -> Vec<deepseek_core::ToolDefinition> {
        let Some(ref mcp) = self.mcp else {
            return vec![];
        };
        let tools = match mcp.discover_tools() {
            Ok(t) => t,
            Err(e) => {
                self.observer
                    .verbose_log(&format!("MCP tool discovery failed: {e}"));
                return vec![];
            }
        };
        // Filter tools by managed settings MCP server allow/deny lists
        let managed = deepseek_policy::load_managed_settings();
        let filtered: Vec<McpTool> = if let Some(ref managed) = managed {
            tools
                .into_iter()
                .filter(|t| deepseek_policy::is_mcp_server_allowed(&t.server_id, managed))
                .collect()
        } else {
            tools
        };
        if let Ok(mut cache) = self.mcp_tools.lock() {
            *cache = filtered.clone();
        }
        filtered
            .into_iter()
            .map(|t| mcp_tool_to_definition(&t))
            .collect()
    }

    /// Check if a tool name is an MCP tool (starts with `mcp__`).
    pub(crate) fn search_mcp_tools(&self, query: &str) -> Vec<McpTool> {
        let cache = self.mcp_tools.lock().ok();
        let tools = match cache {
            Some(ref guard) => guard.as_slice(),
            None => return vec![],
        };
        let query_lower = query.to_ascii_lowercase();
        tools
            .iter()
            .filter(|t| {
                t.name.to_ascii_lowercase().contains(&query_lower)
                    || t.description.to_ascii_lowercase().contains(&query_lower)
            })
            .cloned()
            .collect()
    }
}
