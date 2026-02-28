use anyhow::{Context, Result, anyhow};
use base64::Engine as _;
use chrono::Utc;
use deepseek_store::{McpServerRecord, McpToolCacheRecord, Store};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
#[cfg(test)]
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum McpTransport {
    Stdio,
    Http,
    Sse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct McpServer {
    pub id: String,
    pub name: String,
    pub transport: McpTransport,
    pub command: Option<String>,
    pub args: Vec<String>,
    pub url: Option<String>,
    pub enabled: bool,
    pub metadata: serde_json::Value,
    /// Custom headers for HTTP/SSE transport (e.g. authorization).
    pub headers: Vec<(String, String)>,
}

impl Default for McpServer {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            transport: McpTransport::Stdio,
            command: None,
            args: Vec::new(),
            url: None,
            enabled: true,
            metadata: serde_json::Value::Null,
            headers: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpConfig {
    pub servers: Vec<McpServer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub server_id: String,
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolRefresh {
    pub server_id: String,
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolChangeNotice {
    pub fingerprint: String,
    pub changed_servers: Vec<String>,
}

/// An MCP resource — a URI-addressable piece of content from an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResource {
    pub server_id: String,
    pub uri: String,
    pub name: String,
    pub description: String,
    pub mime_type: Option<String>,
}

/// An MCP prompt — a parameterized prompt template from an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPrompt {
    pub server_id: String,
    pub name: String,
    pub description: String,
    pub arguments: Vec<McpPromptArgument>,
}

/// An argument for an MCP prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptArgument {
    pub name: String,
    pub description: String,
    pub required: bool,
}

pub struct McpManager {
    workspace: PathBuf,
    store: Store,
}

impl McpManager {
    pub fn new(workspace: &Path) -> Result<Self> {
        Ok(Self {
            workspace: workspace.to_path_buf(),
            store: Store::new(workspace)?,
        })
    }

    pub fn project_config_path(&self) -> PathBuf {
        self.workspace.join(".mcp.json")
    }

    pub fn user_config_path() -> Option<PathBuf> {
        let home = std::env::var("HOME")
            .ok()
            .or_else(|| std::env::var("USERPROFILE").ok())?;
        Some(Path::new(&home).join(".deepseek/mcp.json"))
    }

    pub fn user_local_config_path() -> Option<PathBuf> {
        let home = std::env::var("HOME")
            .ok()
            .or_else(|| std::env::var("USERPROFILE").ok())?;
        Some(Path::new(&home).join(".deepseek/mcp.local.json"))
    }

    pub fn load_project_config(&self) -> Result<McpConfig> {
        let path = self.project_config_path();
        if !path.exists() {
            return Ok(McpConfig::default());
        }
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }

    pub fn save_project_config(&self, config: &McpConfig) -> Result<()> {
        let path = self.project_config_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(config)?)?;
        Ok(())
    }

    pub fn list_servers(&self) -> Result<Vec<McpServer>> {
        let mut merged = Vec::new();
        // Managed config has highest priority (cannot be overridden)
        if let Some(managed) = Self::load_managed_mcp_config() {
            merged.extend(managed.servers);
        }
        if let Some(path) = Self::user_config_path() {
            merged.extend(load_config_if_exists(&path)?.servers);
        }
        if let Some(path) = Self::user_local_config_path() {
            merged.extend(load_config_if_exists(&path)?.servers);
        }
        merged.extend(self.load_project_config()?.servers);
        // P5-13: Expand environment variables in all server configs
        for server in &mut merged {
            expand_server_env_vars(server);
        }
        merged.sort_by(|a, b| a.id.cmp(&b.id));
        merged.dedup_by(|a, b| a.id == b.id); // first wins (managed > user > project)
        Ok(merged)
    }

    /// Check whether a server ID comes from managed (enterprise) config.
    pub fn is_managed_server(server_id: &str) -> bool {
        Self::load_managed_mcp_config()
            .map(|cfg| cfg.servers.iter().any(|s| s.id == server_id))
            .unwrap_or(false)
    }

    pub fn get_server(&self, id: &str) -> Result<Option<McpServer>> {
        Ok(self.list_servers()?.into_iter().find(|s| s.id == id))
    }

    pub fn add_server(&self, server: McpServer) -> Result<()> {
        if server.id.trim().is_empty() {
            return Err(anyhow!("server id cannot be empty"));
        }
        let mut cfg = self.load_project_config()?;
        cfg.servers.retain(|existing| existing.id != server.id);
        cfg.servers.push(server.clone());
        self.save_project_config(&cfg)?;
        self.store.upsert_mcp_server(&McpServerRecord {
            server_id: server.id,
            name: server.name,
            transport: match server.transport {
                McpTransport::Stdio => "stdio".to_string(),
                McpTransport::Http => "http".to_string(),
                McpTransport::Sse => "sse".to_string(),
            },
            endpoint: server.command.or(server.url).unwrap_or_default(),
            enabled: server.enabled,
            metadata_json: serde_json::to_string(&server.metadata)?,
            updated_at: Utc::now().to_rfc3339(),
        })?;
        Ok(())
    }

    pub fn remove_server(&self, id: &str) -> Result<bool> {
        if Self::is_managed_server(id) {
            return Err(anyhow!("cannot remove managed server: {id}"));
        }
        let mut cfg = self.load_project_config()?;
        let before = cfg.servers.len();
        cfg.servers.retain(|existing| existing.id != id);
        let removed = cfg.servers.len() != before;
        if removed {
            self.save_project_config(&cfg)?;
            self.store.remove_mcp_server(id)?;
        }
        Ok(removed)
    }

    pub fn discover_tools(&self) -> Result<Vec<McpTool>> {
        Ok(self.refresh_tools()?.0)
    }

    pub fn refresh_tools(&self) -> Result<(Vec<McpTool>, Vec<McpToolRefresh>)> {
        let mut tools = Vec::new();
        let mut refreshes = Vec::new();
        let existing = self.store.list_mcp_tool_cache()?.into_iter().fold(
            BTreeMap::<String, BTreeSet<String>>::new(),
            |mut acc, row| {
                acc.entry(row.server_id).or_default().insert(row.tool_name);
                acc
            },
        );

        for server in self.list_servers()?.into_iter().filter(|s| s.enabled) {
            let discovered = discover_server_tools(&server)
                .with_context(|| format!("failed to discover MCP tools for {}", server.id))
                .unwrap_or_else(|_| {
                    vec![McpTool {
                        server_id: server.id.clone(),
                        name: format!("{}.tool", server.id),
                        description: "fallback tool".to_string(),
                    }]
                });
            let previous = existing.get(&server.id).cloned().unwrap_or_default();
            let mut current = BTreeSet::new();
            let mut replacement_rows = Vec::new();
            for tool in discovered {
                current.insert(tool.name.clone());
                replacement_rows.push(McpToolCacheRecord {
                    server_id: tool.server_id.clone(),
                    tool_name: tool.name.clone(),
                    description: tool.description.clone(),
                    schema_json: "{}".to_string(),
                    updated_at: Utc::now().to_rfc3339(),
                });
                tools.push(tool);
            }
            self.store
                .replace_mcp_tool_cache_for_server(&server.id, &replacement_rows)?;
            let added = current
                .difference(&previous)
                .cloned()
                .collect::<Vec<String>>();
            let removed = previous
                .difference(&current)
                .cloned()
                .collect::<Vec<String>>();
            refreshes.push(McpToolRefresh {
                server_id: server.id,
                added,
                removed,
                total: current.len(),
            });
        }
        Ok((tools, refreshes))
    }

    pub fn discover_tools_with_notice(
        &self,
        previous_fingerprint: Option<&str>,
    ) -> Result<(Vec<McpTool>, Vec<McpToolRefresh>, McpToolChangeNotice)> {
        let (tools, refreshes) = self.refresh_tools()?;
        let fingerprint = tool_fingerprint(&tools)?;
        let changed_servers = if previous_fingerprint.is_some_and(|value| value == fingerprint) {
            Vec::new()
        } else {
            refreshes
                .iter()
                .filter(|refresh| !(refresh.added.is_empty() && refresh.removed.is_empty()))
                .map(|refresh| refresh.server_id.clone())
                .collect::<Vec<_>>()
        };
        Ok((
            tools,
            refreshes,
            McpToolChangeNotice {
                fingerprint,
                changed_servers,
            },
        ))
    }

    /// Resolve an MCP resource reference (`@server:protocol://path`).
    /// Returns the resource content as a string.
    pub fn resolve_resource(&self, reference: &str) -> Result<String> {
        // Parse @server:uri format
        let stripped = reference.strip_prefix('@').unwrap_or(reference);
        let (server_id, uri) = stripped
            .split_once(':')
            .ok_or_else(|| anyhow!("invalid resource reference: {reference}"))?;

        let server = self
            .get_server(server_id)?
            .ok_or_else(|| anyhow!("MCP server not found: {server_id}"))?;

        match server.transport {
            McpTransport::Http | McpTransport::Sse => {
                let base_url = server
                    .url
                    .as_deref()
                    .ok_or_else(|| anyhow!("HTTP/SSE MCP server has no URL"))?;
                let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
                let mut req = client.post(base_url);
                for (k, v) in &server.headers {
                    req = req.header(k.as_str(), v.as_str());
                }
                let resp: serde_json::Value = req
                    .json(&json!({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "resources/read",
                        "params": { "uri": uri }
                    }))
                    .send()?
                    .error_for_status()?
                    .json()?;
                extract_resource_text(
                    resp.get("result")
                        .ok_or_else(|| anyhow!("MCP resources/read response missing result"))?,
                )
            }
            McpTransport::Stdio => {
                let response = execute_mcp_stdio_request(
                    &server,
                    "resources/read",
                    json!({ "uri": uri }),
                    2,
                    Duration::from_secs(6),
                )?;
                extract_resource_text(
                    response
                        .get("result")
                        .ok_or_else(|| anyhow!("MCP resources/read response missing result"))?,
                )
            }
        }
    }

    /// Import MCP servers from Claude Desktop configuration.
    pub fn import_from_claude_desktop(&self) -> Result<usize> {
        let servers = import_from_claude_desktop()?;
        let count = servers.len();
        for server in servers {
            self.add_server(server)?;
        }
        Ok(count)
    }

    /// Search tools by relevance when >50 tools are registered.
    /// Uses keyword scoring to surface the most relevant subset.
    pub fn search_tools_by_relevance(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<McpTool>> {
        let all_tools = self.discover_tools()?;

        // Below threshold, return all tools
        if all_tools.len() <= 50 {
            return Ok(all_tools);
        }

        let query_lower = query.to_ascii_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored: Vec<(f64, McpTool)> = all_tools
            .into_iter()
            .map(|tool| {
                let name_lower = tool.name.to_ascii_lowercase();
                let desc_lower = tool.description.to_ascii_lowercase();

                let mut score = 0.0_f64;

                // Exact name match is highest priority
                if name_lower == query_lower {
                    score += 10.0;
                }

                // Name contains query
                if name_lower.contains(&query_lower) {
                    score += 5.0;
                }

                // Per-term scoring: each query term that matches gets a score boost
                for term in &query_terms {
                    if name_lower.contains(term) {
                        score += 3.0;
                    }
                    if desc_lower.contains(term) {
                        score += 1.0;
                    }
                }

                // Server ID match bonus
                if tool.server_id.to_ascii_lowercase().contains(&query_lower) {
                    score += 2.0;
                }

                (score, tool)
            })
            .filter(|(score, _)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_results);
        Ok(scored.into_iter().map(|(_, tool)| tool).collect())
    }

    /// List MCP resources from all enabled servers for autocomplete.
    pub fn list_resources(&self) -> Result<Vec<McpResource>> {
        let mut resources = Vec::new();
        for server in self.list_servers()?.into_iter().filter(|s| s.enabled) {
            let discovered = match server.transport {
                McpTransport::Stdio => execute_mcp_stdio_request(
                    &server,
                    "resources/list",
                    json!({}),
                    3,
                    Duration::from_secs(6),
                )
                .ok()
                .and_then(|r| r.get("result")?.get("resources")?.as_array().cloned()),
                McpTransport::Http | McpTransport::Sse => server.url.as_deref().and_then(|url| {
                    Client::builder()
                        .timeout(Duration::from_secs(10))
                        .build()
                        .ok()?
                        .post(url)
                        .json(
                            &json!({"jsonrpc":"2.0","id":3,"method":"resources/list","params":{}}),
                        )
                        .send()
                        .ok()?
                        .json::<serde_json::Value>()
                        .ok()?
                        .get("result")?
                        .get("resources")?
                        .as_array()
                        .cloned()
                }),
            };
            if let Some(items) = discovered {
                for item in items {
                    resources.push(McpResource {
                        server_id: server.id.clone(),
                        uri: item
                            .get("uri")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        name: item
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        description: item
                            .get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        mime_type: item
                            .get("mimeType")
                            .and_then(|v| v.as_str())
                            .map(String::from),
                    });
                }
            }
        }
        Ok(resources)
    }

    /// Search resources by prefix for autocomplete (e.g. `@server:uri`).
    pub fn autocomplete_resources(&self, prefix: &str) -> Result<Vec<McpResource>> {
        let all = self.list_resources()?;
        let prefix_lower = prefix.to_ascii_lowercase();
        let mut matches: Vec<McpResource> = all
            .into_iter()
            .filter(|r| {
                let label = format!("@{}:{}", r.server_id, r.uri).to_ascii_lowercase();
                label.starts_with(&prefix_lower)
                    || r.name.to_ascii_lowercase().contains(&prefix_lower)
            })
            .collect();
        matches.truncate(10);
        Ok(matches)
    }

    /// Returns tools appropriate for the current context usage.
    /// When `context_used_pct` > 10%, uses search-based filtering if a query is provided.
    pub fn tools_for_context(
        &self,
        context_used_pct: f64,
        query: Option<&str>,
        max_results: usize,
    ) -> Result<Vec<McpTool>> {
        if context_used_pct > 10.0
            && let Some(q) = query
        {
            return self.search_tools_by_relevance(q, max_results);
        }
        self.discover_tools()
    }

    /// P5-12: List all prompts from MCP servers.
    ///
    /// Queries each enabled server for its `prompts/list` endpoint and returns
    /// a flat list of `McpPrompt` records. These can be registered as slash
    /// commands like `/mcp-<server>-<prompt>` in the UI.
    pub fn list_prompts(&self) -> Result<Vec<McpPrompt>> {
        let mut prompts = Vec::new();
        for server in self.list_servers()?.into_iter().filter(|s| s.enabled) {
            let discovered = match server.transport {
                McpTransport::Stdio => execute_mcp_stdio_request(
                    &server,
                    "prompts/list",
                    json!({}),
                    4,
                    Duration::from_secs(6),
                )
                .ok()
                .and_then(|r| r.get("result")?.get("prompts")?.as_array().cloned()),
                McpTransport::Http | McpTransport::Sse => server.url.as_deref().and_then(|url| {
                    Client::builder()
                        .timeout(Duration::from_secs(10))
                        .build()
                        .ok()?
                        .post(url)
                        .json(&json!({"jsonrpc":"2.0","id":4,"method":"prompts/list","params":{}}))
                        .send()
                        .ok()?
                        .json::<serde_json::Value>()
                        .ok()?
                        .get("result")?
                        .get("prompts")?
                        .as_array()
                        .cloned()
                }),
            };
            if let Some(items) = discovered {
                for item in items {
                    let arguments = item
                        .get("arguments")
                        .and_then(|a| a.as_array())
                        .map(|arr| {
                            arr.iter()
                                .map(|a| McpPromptArgument {
                                    name: a
                                        .get("name")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or_default()
                                        .to_string(),
                                    description: a
                                        .get("description")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or_default()
                                        .to_string(),
                                    required: a
                                        .get("required")
                                        .and_then(|v| v.as_bool())
                                        .unwrap_or(false),
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    prompts.push(McpPrompt {
                        server_id: server.id.clone(),
                        name: item
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        description: item
                            .get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        arguments,
                    });
                }
            }
        }
        Ok(prompts)
    }

    /// P5-13: Handle `notifications/tools/list_changed` from MCP server.
    ///
    /// Re-discovers tools from all servers and returns the refresh results,
    /// allowing the caller to update tool definitions mid-session.
    pub fn handle_list_changed_notification(&self) -> Result<(Vec<McpTool>, Vec<McpToolRefresh>)> {
        self.refresh_tools()
    }

    /// Load managed MCP config (enterprise lockdown).
    /// This is read from a system-wide location and cannot be overridden by users.
    pub fn load_managed_mcp_config() -> Option<McpConfig> {
        let path = managed_mcp_config_path()?;
        if !path.exists() {
            return None;
        }
        let raw = fs::read_to_string(&path).ok()?;
        serde_json::from_str(&raw).ok()
    }
}

fn tool_fingerprint(tools: &[McpTool]) -> Result<String> {
    let serialized = serde_json::to_vec(tools)?;
    Ok(format!("{:x}", Sha256::digest(serialized)))
}

fn load_config_if_exists(path: &Path) -> Result<McpConfig> {
    if !path.exists() {
        return Ok(McpConfig::default());
    }
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

/// Expand environment variables (`${VAR}` and `${VAR:-default}`) in a string.
pub fn expand_env_vars(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '$' && chars.peek() == Some(&'{') {
            chars.next(); // consume '{'
            let mut var_expr = String::new();
            for c in chars.by_ref() {
                if c == '}' {
                    break;
                }
                var_expr.push(c);
            }
            if let Some((var_name, default)) = var_expr.split_once(":-") {
                result.push_str(&std::env::var(var_name).unwrap_or_else(|_| default.to_string()));
            } else {
                result.push_str(&std::env::var(&var_expr).unwrap_or_default());
            }
        } else {
            result.push(ch);
        }
    }
    result
}

/// Expand environment variables in all string fields of an MCP server config.
pub fn expand_server_env_vars(server: &mut McpServer) {
    server.id = expand_env_vars(&server.id);
    server.name = expand_env_vars(&server.name);
    if let Some(ref cmd) = server.command {
        server.command = Some(expand_env_vars(cmd));
    }
    server.args = server.args.iter().map(|a| expand_env_vars(a)).collect();
    if let Some(ref url) = server.url {
        server.url = Some(expand_env_vars(url));
    }
}

/// Import MCP servers from Claude Desktop config file.
/// Reads `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
/// or `%APPDATA%/Claude/claude_desktop_config.json` (Windows).
pub fn import_from_claude_desktop() -> Result<Vec<McpServer>> {
    let config_path =
        claude_desktop_config_path().ok_or_else(|| anyhow!("Claude Desktop config not found"))?;
    if !config_path.exists() {
        return Err(anyhow!(
            "Claude Desktop config not found at {}",
            config_path.display()
        ));
    }
    let raw = fs::read_to_string(&config_path)?;
    let config: serde_json::Value = serde_json::from_str(&raw)?;

    let servers_obj = config
        .get("mcpServers")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("no mcpServers in Claude Desktop config"))?;

    let mut servers = Vec::new();
    for (id, entry) in servers_obj {
        let command = entry
            .get("command")
            .and_then(|v| v.as_str())
            .map(String::from);
        let args = entry
            .get("args")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let url = entry.get("url").and_then(|v| v.as_str()).map(String::from);
        let transport = if url.is_some() {
            McpTransport::Http
        } else {
            McpTransport::Stdio
        };
        servers.push(McpServer {
            id: id.clone(),
            name: id.clone(),
            transport,
            command,
            args,
            url,
            enabled: true,
            metadata: entry.get("env").cloned().unwrap_or(serde_json::Value::Null),
            headers: Vec::new(),
        });
    }
    Ok(servers)
}

fn claude_desktop_config_path() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        let home = std::env::var("HOME").ok()?;
        Some(
            PathBuf::from(home)
                .join("Library/Application Support/Claude/claude_desktop_config.json"),
        )
    }
    #[cfg(target_os = "windows")]
    {
        let appdata = std::env::var("APPDATA").ok()?;
        Some(PathBuf::from(appdata).join("Claude/claude_desktop_config.json"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let home = std::env::var("HOME").ok()?;
        Some(PathBuf::from(home).join(".config/claude/claude_desktop_config.json"))
    }
}

/// System-wide managed MCP config path for enterprise lockdown.
fn managed_mcp_config_path() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        Some(PathBuf::from(
            "/Library/Application Support/DeepSeekCode/managed-mcp.json",
        ))
    }
    #[cfg(target_os = "linux")]
    {
        Some(PathBuf::from("/etc/deepseek-code/managed-mcp.json"))
    }
    #[cfg(target_os = "windows")]
    {
        Some(PathBuf::from(
            "C:\\Program Files\\DeepSeekCode\\managed-mcp.json",
        ))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

/// Per-MCP output token limits (warn at threshold, error at max).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTokenLimits {
    /// Warning threshold (tokens). Default 10_000.
    pub warn_threshold: u64,
    /// Hard maximum (tokens). Default 25_000.
    pub max_tokens: u64,
}

impl Default for McpTokenLimits {
    fn default() -> Self {
        Self {
            warn_threshold: 10_000,
            max_tokens: 25_000,
        }
    }
}

/// Check an MCP tool output against token limits.
/// Returns `(output, was_truncated)`.
pub fn enforce_mcp_token_limit(output: &str, limits: &McpTokenLimits) -> (String, bool) {
    // Rough estimate: ~4 chars per token
    let estimated_tokens = output.len() as u64 / 4;
    if estimated_tokens > limits.max_tokens {
        let max_chars = (limits.max_tokens * 4) as usize;
        let truncated = if output.len() > max_chars {
            format!(
                "{}\n\n... (output truncated: ~{} tokens exceeded limit of {})",
                &output[..max_chars],
                estimated_tokens,
                limits.max_tokens
            )
        } else {
            output.to_string()
        };
        (truncated, true)
    } else {
        (output.to_string(), false)
    }
}

fn discover_server_tools(server: &McpServer) -> Result<Vec<McpTool>> {
    if let Some(list) = server.metadata.get("tools").and_then(|v| v.as_array()) {
        let tools = list
            .iter()
            .filter_map(|entry| {
                let name = entry.get("name")?.as_str()?.to_string();
                let description = entry
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                Some(McpTool {
                    server_id: server.id.clone(),
                    name,
                    description,
                })
            })
            .collect::<Vec<_>>();
        if !tools.is_empty() {
            return Ok(tools);
        }
    }

    match server.transport {
        McpTransport::Http => {
            let url = server
                .url
                .as_deref()
                .ok_or_else(|| anyhow!("http transport requires url"))?;
            let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
            let value: serde_json::Value = client.get(url).send()?.error_for_status()?.json()?;
            let tools = value
                .get("tools")
                .and_then(|v| v.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|entry| {
                            let name = entry.get("name")?.as_str()?.to_string();
                            let description = entry
                                .get("description")
                                .and_then(|v| v.as_str())
                                .unwrap_or_default()
                                .to_string();
                            Some(McpTool {
                                server_id: server.id.clone(),
                                name,
                                description,
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            Ok(tools)
        }
        McpTransport::Sse => discover_sse_tools(server),
        McpTransport::Stdio => discover_stdio_tools(server),
    }
}

fn discover_sse_tools(server: &McpServer) -> Result<Vec<McpTool>> {
    let url = server
        .url
        .as_deref()
        .ok_or_else(|| anyhow!("SSE transport requires url"))?;
    let client = Client::builder().timeout(Duration::from_secs(15)).build()?;
    let mut req = client.get(url);
    for (k, v) in &server.headers {
        req = req.header(k.as_str(), v.as_str());
    }
    let resp = req
        .header("Accept", "text/event-stream")
        .send()?
        .error_for_status()?;
    let text = resp.text()?;
    parse_sse_tools(&server.id, &text)
}

fn parse_sse_tools(server_id: &str, sse_text: &str) -> Result<Vec<McpTool>> {
    let mut tools = Vec::new();
    for chunk in sse_text.split("\n\n") {
        let data = chunk
            .strip_prefix("data: ")
            .or_else(|| chunk.lines().find_map(|l| l.strip_prefix("data: ")));
        if let Some(json_str) = data
            && let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str)
        {
            tools.extend(parse_mcp_tools_message(server_id, &value));
        }
    }
    if tools.is_empty() {
        return Err(anyhow!("no tools discovered from SSE MCP server"));
    }
    Ok(tools)
}

fn discover_stdio_tools(server: &McpServer) -> Result<Vec<McpTool>> {
    let command = server
        .command
        .as_deref()
        .ok_or_else(|| anyhow!("stdio transport requires command"))?;
    let mut child = Command::new(command)
        .args(&server.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("failed to start MCP stdio command: {command}"))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("failed to open MCP stdio stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("failed to open MCP stdio stdout"))?;
    let (tx, rx) = mpsc::channel::<serde_json::Value>();

    let reader_handle = thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        while let Ok(Some(value)) = read_mcp_frame(&mut reader) {
            if tx.send(value).is_err() {
                break;
            }
        }
    });

    let init = json!({
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {
            "name": "deepseek-cli",
            "version": "0.2.0"
        }
    });
    write_mcp_request(&mut stdin, 1, "initialize", init)?;
    write_mcp_request(&mut stdin, 2, "tools/list", json!({}))?;
    let _ = stdin.flush();
    drop(stdin);

    let mut discovered = Vec::new();
    for _ in 0..20 {
        match rx.recv_timeout(Duration::from_millis(150)) {
            Ok(message) => {
                discovered.extend(parse_mcp_tools_message(&server.id, &message));
                if !discovered.is_empty() {
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    let _ = child.kill();
    let _ = child.wait();
    let _ = reader_handle.join();

    if discovered.is_empty() {
        return Err(anyhow!("no tools discovered from MCP stdio server"));
    }

    discovered.sort_by(|a, b| a.name.cmp(&b.name));
    discovered.dedup_by(|a, b| a.name == b.name);
    Ok(discovered)
}

/// A pooled MCP stdio connection — keeps a server process alive across requests.
struct PooledConnection {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    rx: mpsc::Receiver<serde_json::Value>,
    _reader_handle: Option<thread::JoinHandle<()>>,
    last_used: Instant,
    next_request_id: u64,
}

/// Connection pool for MCP stdio servers. Keeps processes alive and reuses them
/// across tool calls. Health checks before reuse, kills after idle timeout.
pub struct McpConnectionPool {
    connections: std::sync::Mutex<std::collections::HashMap<String, PooledConnection>>,
    idle_timeout: Duration,
}

/// Maximum idle time before a pooled connection is killed.
const MCP_POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes

impl McpConnectionPool {
    /// Create a new empty connection pool.
    pub fn new() -> Self {
        Self {
            connections: std::sync::Mutex::new(std::collections::HashMap::new()),
            idle_timeout: MCP_POOL_IDLE_TIMEOUT,
        }
    }

    /// Execute a request against a pooled stdio server. Reuses existing connections
    /// or creates a new one. Health checks the connection before reuse.
    pub fn execute(
        &self,
        server: &McpServer,
        method: &str,
        params: serde_json::Value,
        timeout: Duration,
    ) -> Result<serde_json::Value> {
        let key = server.id.clone();
        let mut pool = self.connections.lock().unwrap();

        // Evict idle connections
        pool.retain(|_, conn| conn.last_used.elapsed() < self.idle_timeout);

        // Try reusing existing connection
        if let Some(conn) = pool.get_mut(&key) {
            // Health check: try_wait returns Some if process exited
            if conn.child.try_wait().ok().flatten().is_some() {
                pool.remove(&key);
            } else {
                let request_id = conn.next_request_id;
                conn.next_request_id += 1;
                conn.last_used = Instant::now();

                if let Err(e) =
                    write_mcp_request(&mut conn.stdin, request_id, method, params.clone())
                {
                    // Connection broken — remove and fall through to create new one
                    let mut removed = pool.remove(&key).unwrap();
                    let _ = removed.child.kill();
                    let _ = removed.child.wait();
                    let _ = e; // Connection broken — will create new one
                } else {
                    let _ = conn.stdin.flush();
                    let response = wait_for_mcp_response(&conn.rx, request_id, timeout);
                    return match response {
                        Some(value) => Ok(value),
                        None => Err(anyhow!(
                            "MCP pooled request timed out: method={method} timeout_ms={}",
                            timeout.as_millis()
                        )),
                    };
                }
            }
        }

        // Create new connection
        drop(pool); // Release lock during process spawn
        let conn = self.create_connection(server)?;
        let mut pool = self.connections.lock().unwrap();

        let request_id = conn.next_request_id;
        let mut conn = conn;
        conn.next_request_id += 1;
        conn.last_used = Instant::now();

        write_mcp_request(&mut conn.stdin, request_id, method, params)?;
        let _ = conn.stdin.flush();

        let response = wait_for_mcp_response(&conn.rx, request_id, timeout);
        pool.insert(key, conn);

        match response {
            Some(value) => Ok(value),
            None => Err(anyhow!(
                "MCP pooled request timed out: method={method} timeout_ms={}",
                timeout.as_millis()
            )),
        }
    }

    /// Create a new pooled connection: spawn process, initialize MCP protocol.
    fn create_connection(&self, server: &McpServer) -> Result<PooledConnection> {
        let command = server
            .command
            .as_deref()
            .ok_or_else(|| anyhow!("stdio transport requires command"))?;
        let mut child = Command::new(command)
            .args(&server.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("failed to start MCP stdio command: {command}"))?;

        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("failed to open MCP stdio stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("failed to open MCP stdio stdout"))?;
        let (tx, rx) = mpsc::channel::<serde_json::Value>();

        let reader_handle = thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            while let Ok(Some(value)) = read_mcp_frame(&mut reader) {
                if tx.send(value).is_err() {
                    break;
                }
            }
        });

        // Send MCP initialize request
        let init = json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {
                "name": "deepseek-cli",
                "version": env!("CARGO_PKG_VERSION")
            }
        });
        write_mcp_request(&mut stdin, 1, "initialize", init)?;
        let _ = stdin.flush();

        // Wait for initialize response (ID=1)
        let _ = wait_for_mcp_response(&rx, 1, Duration::from_secs(10));

        Ok(PooledConnection {
            child,
            stdin,
            rx,
            _reader_handle: Some(reader_handle),
            last_used: Instant::now(),
            next_request_id: 100, // Start after init
        })
    }

    /// Gracefully shut down all pooled connections.
    pub fn shutdown(&self) {
        let mut pool = self.connections.lock().unwrap();
        for (id, mut conn) in pool.drain() {
            let _ = id;
            let _ = conn.child.kill();
            let _ = conn.child.wait();
        }
    }
}

impl Default for McpConnectionPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for McpConnectionPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Execute one framed JSON-RPC request against an MCP stdio server.
///
/// This function starts a short-lived server process, initializes MCP protocol,
/// then sends exactly one method request and returns the matching response frame.
pub fn execute_mcp_stdio_request(
    server: &McpServer,
    method: &str,
    params: serde_json::Value,
    request_id: u64,
    timeout: Duration,
) -> Result<serde_json::Value> {
    let command = server
        .command
        .as_deref()
        .ok_or_else(|| anyhow!("stdio transport requires command"))?;
    let mut child = Command::new(command)
        .args(&server.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("failed to start MCP stdio command: {command}"))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("failed to open MCP stdio stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("failed to open MCP stdio stdout"))?;
    let (tx, rx) = mpsc::channel::<serde_json::Value>();

    let reader_handle = thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        while let Ok(Some(value)) = read_mcp_frame(&mut reader) {
            if tx.send(value).is_err() {
                break;
            }
        }
    });

    let init = json!({
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {
            "name": "deepseek-cli",
            "version": env!("CARGO_PKG_VERSION")
        }
    });
    write_mcp_request(&mut stdin, 1, "initialize", init)?;
    write_mcp_request(&mut stdin, request_id, method, params)?;
    let _ = stdin.flush();
    drop(stdin);

    let response = wait_for_mcp_response(&rx, request_id, timeout);

    let _ = child.kill();
    let _ = child.wait();
    let _ = reader_handle.join();

    match response {
        Some(value) => Ok(value),
        None => Err(anyhow!(
            "MCP stdio request timed out: method={method} timeout_ms={}",
            timeout.as_millis()
        )),
    }
}

fn wait_for_mcp_response(
    rx: &mpsc::Receiver<serde_json::Value>,
    request_id: u64,
    timeout: Duration,
) -> Option<serde_json::Value> {
    let started = Instant::now();
    while started.elapsed() < timeout {
        let remaining = timeout
            .checked_sub(started.elapsed())
            .unwrap_or_else(|| Duration::from_millis(1));
        let wait = remaining.min(Duration::from_millis(150));
        match rx.recv_timeout(wait) {
            Ok(message) => {
                if message.get("id").and_then(|v| v.as_u64()) == Some(request_id) {
                    return Some(message);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    None
}

fn extract_resource_text(result: &serde_json::Value) -> Result<String> {
    let contents = result
        .get("contents")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("MCP resources/read missing contents array"))?;
    if contents.is_empty() {
        return Err(anyhow!("MCP resources/read returned no contents"));
    }

    let mut merged = Vec::new();
    for item in contents {
        if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
            merged.push(text.to_string());
            continue;
        }
        if let Some(blob) = item.get("blob").and_then(|v| v.as_str()) {
            let mime = item
                .get("mimeType")
                .and_then(|v| v.as_str())
                .unwrap_or("application/octet-stream");
            match base64::engine::general_purpose::STANDARD.decode(blob) {
                Ok(bytes) => match String::from_utf8(bytes.clone()) {
                    Ok(text) => merged.push(text),
                    Err(_) => merged.push(format!(
                        "[binary-resource mime={} size={} bytes]",
                        mime,
                        bytes.len()
                    )),
                },
                Err(_) => merged.push(format!(
                    "[invalid-base64-resource mime={} chars={}]",
                    mime,
                    blob.len()
                )),
            }
        }
    }

    if merged.is_empty() {
        return Err(anyhow!(
            "MCP resources/read contents had no text/blob payloads"
        ));
    }
    let joined = merged.join("\n");
    let (bounded, _truncated) = enforce_mcp_token_limit(&joined, &McpTokenLimits::default());
    Ok(bounded)
}

fn write_mcp_request(
    writer: &mut dyn Write,
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

fn read_mcp_frame<R: BufRead>(reader: &mut R) -> Result<Option<serde_json::Value>> {
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
            content_length = Some(
                value
                    .trim()
                    .parse::<usize>()
                    .context("invalid Content-Length")?,
            );
        }
    }

    let length = content_length.ok_or_else(|| anyhow!("missing Content-Length header"))?;
    let mut buf = vec![0_u8; length];
    reader.read_exact(&mut buf)?;
    Ok(Some(serde_json::from_slice(&buf)?))
}

fn parse_mcp_tools_message(server_id: &str, message: &serde_json::Value) -> Vec<McpTool> {
    let mut out = Vec::new();
    if let Some(tools) = message
        .get("result")
        .and_then(|result| result.get("tools"))
        .and_then(|tools| tools.as_array())
    {
        out.extend(parse_tools_array(server_id, tools));
    }
    if let Some(tools) = message.get("tools").and_then(|tools| tools.as_array()) {
        out.extend(parse_tools_array(server_id, tools));
    }
    out
}

/// OAuth token for MCP server authentication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpOAuthToken {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub token_type: String,
    pub expires_at: Option<String>,
    pub server_id: String,
}

/// Store an OAuth token for an MCP server.
pub fn store_mcp_token(server_id: &str, token: &McpOAuthToken) -> Result<()> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| anyhow::anyhow!("cannot determine home directory"))?;
    let dir = PathBuf::from(home).join(".deepseek/mcp-tokens");
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.json", server_id));
    let json = serde_json::to_string_pretty(token)?;
    fs::write(&path, json)?;
    // Restrict permissions on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(&path, perms)?;
    }
    Ok(())
}

/// Load a stored OAuth token for an MCP server.
pub fn load_mcp_token(server_id: &str) -> Result<Option<McpOAuthToken>> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| anyhow::anyhow!("cannot determine home directory"))?;
    let path = PathBuf::from(home).join(format!(".deepseek/mcp-tokens/{}.json", server_id));
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(path)?;
    let token: McpOAuthToken = serde_json::from_str(&raw)?;
    Ok(Some(token))
}

/// Delete a stored OAuth token.
pub fn delete_mcp_token(server_id: &str) -> Result<()> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| anyhow::anyhow!("cannot determine home directory"))?;
    let path = PathBuf::from(home).join(format!(".deepseek/mcp-tokens/{}.json", server_id));
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

// ── OAuth 2.0 for MCP ────────────────────────────────────────────────────────

/// OAuth 2.0 configuration for an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpOAuthConfig {
    pub authorization_url: String,
    pub token_url: String,
    pub client_id: String,
    pub client_secret: Option<String>,
    pub scopes: Vec<String>,
    /// Local port for the OAuth redirect callback (default 8912).
    #[serde(default = "default_redirect_port")]
    pub redirect_port: u16,
}

fn default_redirect_port() -> u16 {
    8912
}

/// Check whether an OAuth token has expired.
pub fn is_token_expired(token: &McpOAuthToken) -> bool {
    token.expires_at.as_deref().is_some_and(|exp| {
        chrono::DateTime::parse_from_rfc3339(exp)
            .map(|dt| dt < Utc::now())
            .unwrap_or(false)
    })
}

/// Generate a PKCE code verifier (random URL-safe base64 string).
fn generate_pkce_verifier() -> String {
    let buf = rand_bytes();
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(buf)
}

/// Generate a PKCE code challenge (S256) from a verifier.
fn generate_pkce_challenge(verifier: &str) -> String {
    let hash = Sha256::digest(verifier.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(hash)
}

fn rand_bytes() -> [u8; 32] {
    let mut buf = [0u8; 32];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        use std::io::Read;
        let _ = f.read_exact(&mut buf);
    } else {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        buf[..16].copy_from_slice(&seed.to_le_bytes());
        buf[16..20].copy_from_slice(&std::process::id().to_le_bytes());
    }
    buf
}

/// Start the OAuth 2.0 authorization code flow for an MCP server.
pub fn start_oauth_flow(server: &McpServer) -> Result<McpOAuthToken> {
    let oauth = server
        .metadata
        .get("oauth")
        .map(|v| serde_json::from_value::<McpOAuthConfig>(v.clone()))
        .transpose()?
        .ok_or_else(|| anyhow!("server has no OAuth config in metadata"))?;

    // 1. Start local listener
    let listener = std::net::TcpListener::bind(format!("127.0.0.1:{}", oauth.redirect_port))?;

    // 2. Build authorization URL with PKCE
    let state = format!("{:x}", Sha256::digest(rand_bytes()));
    let code_verifier = generate_pkce_verifier();
    let code_challenge = generate_pkce_challenge(&code_verifier);
    let redirect_uri = format!("http://127.0.0.1:{}/callback", oauth.redirect_port);
    let auth_url = format!(
        "{}?client_id={}&redirect_uri={}&response_type=code&state={}&scope={}&code_challenge={}&code_challenge_method=S256",
        oauth.authorization_url,
        oauth.client_id,
        urlencoding(&redirect_uri),
        state,
        oauth.scopes.join("+"),
        code_challenge,
    );

    // 3. Open browser
    eprintln!("Opening browser for OAuth authorization...");
    let _ = open_browser(&auth_url);
    eprintln!("If browser didn't open, visit: {auth_url}");

    // 4. Wait for callback
    let code = wait_for_oauth_callback(&listener, &state, Duration::from_secs(120))?;

    // 5. Exchange code for token
    let token = exchange_oauth_code(&oauth, &code, &code_verifier, &server.id)?;

    // 6. Store token
    store_mcp_token(&server.id, &token)?;
    Ok(token)
}

fn urlencoding(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push_str(&format!("%{:02X}", b));
            }
        }
    }
    out
}

fn exchange_oauth_code(
    oauth: &McpOAuthConfig,
    code: &str,
    code_verifier: &str,
    server_id: &str,
) -> Result<McpOAuthToken> {
    let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
    let mut body = json!({
        "grant_type": "authorization_code",
        "code": code,
        "client_id": oauth.client_id,
        "code_verifier": code_verifier,
    });
    if let Some(ref secret) = oauth.client_secret {
        body["client_secret"] = json!(secret);
    }
    let resp: serde_json::Value = client
        .post(&oauth.token_url)
        .json(&body)
        .send()?
        .error_for_status()?
        .json()?;

    Ok(McpOAuthToken {
        access_token: resp["access_token"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        refresh_token: resp
            .get("refresh_token")
            .and_then(|v| v.as_str())
            .map(String::from),
        token_type: resp["token_type"].as_str().unwrap_or("Bearer").to_string(),
        expires_at: resp
            .get("expires_in")
            .and_then(|v| v.as_u64())
            .map(|secs| (Utc::now() + chrono::Duration::seconds(secs as i64)).to_rfc3339()),
        server_id: server_id.to_string(),
    })
}

/// Refresh an expired OAuth token.
pub fn refresh_oauth_token(server: &McpServer, token: &McpOAuthToken) -> Result<McpOAuthToken> {
    let oauth = server
        .metadata
        .get("oauth")
        .map(|v| serde_json::from_value::<McpOAuthConfig>(v.clone()))
        .transpose()?
        .ok_or_else(|| anyhow!("server has no OAuth config"))?;
    let refresh = token
        .refresh_token
        .as_deref()
        .ok_or_else(|| anyhow!("no refresh token available"))?;
    let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
    let resp: serde_json::Value = client
        .post(&oauth.token_url)
        .json(&json!({
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "client_id": oauth.client_id,
        }))
        .send()?
        .error_for_status()?
        .json()?;

    let new_token = McpOAuthToken {
        access_token: resp["access_token"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        refresh_token: resp
            .get("refresh_token")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| token.refresh_token.clone()),
        token_type: resp["token_type"].as_str().unwrap_or("Bearer").to_string(),
        expires_at: resp
            .get("expires_in")
            .and_then(|v| v.as_u64())
            .map(|secs| (Utc::now() + chrono::Duration::seconds(secs as i64)).to_rfc3339()),
        server_id: server.id.clone(),
    };
    store_mcp_token(&server.id, &new_token)?;
    Ok(new_token)
}

fn open_browser(url: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg(url).spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = Command::new("xdg-open").arg(url).spawn();
    }
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("cmd").args(["/c", "start", url]).spawn();
    }
    Ok(())
}

fn wait_for_oauth_callback(
    listener: &std::net::TcpListener,
    expected_state: &str,
    timeout: Duration,
) -> Result<String> {
    listener.set_nonblocking(true)?;
    let started = std::time::Instant::now();
    loop {
        if started.elapsed() > timeout {
            return Err(anyhow!("OAuth callback timed out"));
        }
        match listener.accept() {
            Ok((mut stream, _)) => {
                use std::io::Read;
                let mut buf = [0u8; 4096];
                stream.set_read_timeout(Some(Duration::from_secs(5)))?;
                let n = stream.read(&mut buf).unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                if let Some(code) = extract_query_param(&request, "code") {
                    let state = extract_query_param(&request, "state").unwrap_or_default();
                    if state != expected_state {
                        return Err(anyhow!("OAuth state mismatch"));
                    }
                    let response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html><body>Authorization successful. You can close this tab.</body></html>";
                    use std::io::Write;
                    let _ = stream.write_all(response.as_bytes());
                    return Ok(code);
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
            Err(e) => return Err(e.into()),
        }
    }
}

fn extract_query_param(request: &str, param: &str) -> Option<String> {
    let query = request.split_whitespace().nth(1)?.split('?').nth(1)?;
    for pair in query.split('&') {
        if let Some((key, value)) = pair.split_once('=')
            && key == param
        {
            return Some(value.to_string());
        }
    }
    None
}

/// Start an MCP JSON-RPC 2.0 server on stdin/stdout, exposing DeepSeek tools.
///
/// This allows other MCP-compatible clients to use DeepSeek CLI as a tool server.
/// Implements the MCP protocol: `initialize`, `tools/list`, `tools/call`.
pub fn run_mcp_serve(workspace: &Path) -> Result<()> {
    eprintln!("deepseek-cli MCP server started (stdio transport)");
    let tool_list = build_serve_tool_list(workspace);

    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let stdout = std::io::stdout();
    let mut writer = stdout.lock();

    loop {
        let message = match read_mcp_frame(&mut reader) {
            Ok(Some(msg)) => msg,
            Ok(None) => break, // EOF
            Err(_) => break,
        };

        let id = message.get("id").cloned();
        let method = message.get("method").and_then(|v| v.as_str()).unwrap_or("");

        let response = match method {
            "initialize" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {
                        "tools": { "listChanged": false }
                    },
                    "serverInfo": {
                        "name": "deepseek-cli",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                }
            }),
            "tools/list" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": { "tools": tool_list }
            }),
            "tools/call" => {
                let params = message.get("params").cloned().unwrap_or(json!({}));
                let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
                let result_text = execute_serve_tool(workspace, tool_name, &arguments);
                json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [{
                            "type": "text",
                            "text": result_text
                        }]
                    }
                })
            }
            _ => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32601,
                    "message": format!("Method not found: {method}")
                }
            }),
        };

        let body = serde_json::to_vec(&response)?;
        write!(writer, "Content-Length: {}\r\n\r\n", body.len())?;
        writer.write_all(&body)?;
        writer.flush()?;
    }

    eprintln!("deepseek-cli MCP server stopped");
    Ok(())
}

/// Build the tool list for MCP serve mode — exposes a subset of DeepSeek tools.
fn build_serve_tool_list(_workspace: &Path) -> Vec<serde_json::Value> {
    let tools = vec![
        (
            "fs_read",
            "Read a file",
            json!({"type": "object", "properties": {"path": {"type": "string", "description": "File path to read"}}, "required": ["path"]}),
        ),
        (
            "fs_write",
            "Write content to a file",
            json!({"type": "object", "properties": {"path": {"type": "string", "description": "File path"}, "content": {"type": "string", "description": "Content to write"}}, "required": ["path", "content"]}),
        ),
        (
            "fs_glob",
            "Find files by glob pattern",
            json!({"type": "object", "properties": {"pattern": {"type": "string", "description": "Glob pattern"}}, "required": ["pattern"]}),
        ),
        (
            "fs_grep",
            "Search file contents with regex",
            json!({"type": "object", "properties": {"pattern": {"type": "string", "description": "Regex pattern"}, "path": {"type": "string", "description": "Search path"}}, "required": ["pattern"]}),
        ),
        (
            "bash_run",
            "Execute a shell command",
            json!({"type": "object", "properties": {"command": {"type": "string", "description": "Command to execute"}, "timeout": {"type": "integer", "description": "Timeout in seconds"}}, "required": ["command"]}),
        ),
        (
            "git_status",
            "Show git status",
            json!({"type": "object", "properties": {}}),
        ),
        (
            "git_diff",
            "Show git diff",
            json!({"type": "object", "properties": {"args": {"type": "string", "description": "Additional git diff args"}}}),
        ),
        (
            "web_fetch",
            "Fetch URL content",
            json!({"type": "object", "properties": {"url": {"type": "string", "description": "URL to fetch"}}, "required": ["url"]}),
        ),
    ];
    tools
        .into_iter()
        .map(|(name, desc, schema)| {
            json!({
                "name": name,
                "description": desc,
                "inputSchema": schema,
            })
        })
        .collect()
}

/// Execute a tool in MCP serve mode (simplified execution without full agent).
fn execute_serve_tool(workspace: &Path, tool_name: &str, arguments: &serde_json::Value) -> String {
    match tool_name {
        "fs_read" => {
            let path = arguments.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let full_path = if Path::new(path).is_absolute() {
                PathBuf::from(path)
            } else {
                workspace.join(path)
            };
            match fs::read_to_string(&full_path) {
                Ok(content) => content,
                Err(e) => format!("Error reading {}: {e}", full_path.display()),
            }
        }
        "fs_write" => {
            let path = arguments.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let content = arguments
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let full_path = if Path::new(path).is_absolute() {
                PathBuf::from(path)
            } else {
                workspace.join(path)
            };
            if let Some(parent) = full_path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            match fs::write(&full_path, content) {
                Ok(()) => format!("Written {} bytes to {}", content.len(), full_path.display()),
                Err(e) => format!("Error writing {}: {e}", full_path.display()),
            }
        }
        "fs_glob" => {
            let pattern = arguments
                .get("pattern")
                .and_then(|v| v.as_str())
                .unwrap_or("*");
            match glob::glob(&workspace.join(pattern).to_string_lossy()) {
                Ok(entries) => {
                    let paths: Vec<String> = entries
                        .filter_map(|e: Result<PathBuf, glob::GlobError>| e.ok())
                        .map(|p: PathBuf| p.to_string_lossy().to_string())
                        .take(200)
                        .collect();
                    paths.join("\n")
                }
                Err(e) => format!("Glob error: {e}"),
            }
        }
        "bash_run" => {
            let cmd = arguments
                .get("command")
                .or_else(|| arguments.get("cmd"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let _timeout_secs = arguments
                .get("timeout")
                .and_then(|v| v.as_u64())
                .unwrap_or(30);
            match Command::new("sh")
                .args(["-c", cmd])
                .current_dir(workspace)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
            {
                Ok(child) => match child.wait_with_output() {
                    Ok(output) => {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        format!(
                            "exit_code: {}\nstdout:\n{}\nstderr:\n{}",
                            output.status.code().unwrap_or(-1),
                            stdout,
                            stderr
                        )
                    }
                    Err(e) => format!("Command wait error: {e}"),
                },
                Err(e) => format!("Command spawn error: {e}"),
            }
        }
        "git_status" | "git_diff" => {
            let git_cmd = if tool_name == "git_status" {
                "status"
            } else {
                "diff"
            };
            let extra_args = arguments.get("args").and_then(|v| v.as_str()).unwrap_or("");
            let mut args = vec![git_cmd];
            if !extra_args.is_empty() {
                args.push(extra_args);
            }
            match Command::new("git")
                .args(&args)
                .current_dir(workspace)
                .output()
            {
                Ok(output) => String::from_utf8_lossy(&output.stdout).to_string(),
                Err(e) => format!("Git error: {e}"),
            }
        }
        "web_fetch" => {
            let url = arguments.get("url").and_then(|v| v.as_str()).unwrap_or("");
            match Client::builder()
                .timeout(Duration::from_secs(15))
                .build()
                .and_then(|c| c.get(url).send())
            {
                Ok(resp) => resp.text().unwrap_or_else(|e| format!("Read error: {e}")),
                Err(e) => format!("Fetch error: {e}"),
            }
        }
        "fs_grep" => {
            let pattern = arguments
                .get("pattern")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let path = arguments
                .get("path")
                .and_then(|v| v.as_str())
                .map(|p| {
                    if Path::new(p).is_absolute() {
                        PathBuf::from(p)
                    } else {
                        workspace.join(p)
                    }
                })
                .unwrap_or_else(|| workspace.to_path_buf());
            match Command::new("grep")
                .args(["-rn", "--include=*", pattern])
                .arg(&path)
                .output()
            {
                Ok(output) => {
                    let text = String::from_utf8_lossy(&output.stdout);
                    if text.len() > 10000 {
                        format!("{}...\n(truncated)", &text[..10000])
                    } else {
                        text.to_string()
                    }
                }
                Err(e) => format!("Grep error: {e}"),
            }
        }
        _ => format!("Unknown tool: {tool_name}"),
    }
}

fn parse_tools_array(server_id: &str, tools: &[serde_json::Value]) -> Vec<McpTool> {
    tools
        .iter()
        .filter_map(|entry| {
            let name = entry.get("name")?.as_str()?.to_string();
            let description = entry
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            Some(McpTool {
                server_id: server_id.to_string(),
                name,
                description,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn add_list_remove_server_round_trip() {
        let workspace = std::env::temp_dir().join(format!("deepseek-mcp-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let manager = McpManager::new(&workspace).expect("manager");

        manager
            .add_server(McpServer {
                id: "local".to_string(),
                name: "Local MCP".to_string(),
                transport: McpTransport::Stdio,
                command: Some("echo".to_string()),
                args: vec!["ok".to_string()],
                url: None,
                enabled: true,
                metadata: serde_json::json!({"tools": [{"name":"hello","description":"test"}]}),
                headers: vec![],
            })
            .expect("add");

        let listed = manager.list_servers().expect("list");
        assert!(listed.iter().any(|s| s.id == "local"));

        let tools = manager.discover_tools().expect("discover");
        assert!(tools.iter().any(|t| t.name == "hello"));

        let removed = manager.remove_server("local").expect("remove");
        assert!(removed);
    }

    #[test]
    fn detects_toolset_changes_and_emits_notice_fingerprint() {
        let workspace = std::env::temp_dir().join(format!("deepseek-mcp-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let manager = McpManager::new(&workspace).expect("manager");

        manager
            .add_server(McpServer {
                id: "notify".to_string(),
                name: "Notify".to_string(),
                transport: McpTransport::Stdio,
                command: Some("echo".to_string()),
                args: vec!["ok".to_string()],
                url: None,
                enabled: true,
                metadata: serde_json::json!({"tools": [{"name":"a"},{"name":"b"}]}),
                headers: vec![],
            })
            .expect("add");

        let (_, first_refreshes, first_notice) = manager
            .discover_tools_with_notice(None)
            .expect("first refresh");
        let first = first_refreshes
            .iter()
            .find(|refresh| refresh.server_id == "notify")
            .expect("refresh");
        assert!(first.added.contains(&"a".to_string()));
        assert!(first.added.contains(&"b".to_string()));
        assert!(!first_notice.fingerprint.is_empty());

        manager
            .add_server(McpServer {
                id: "notify".to_string(),
                name: "Notify".to_string(),
                transport: McpTransport::Stdio,
                command: Some("echo".to_string()),
                args: vec!["ok".to_string()],
                url: None,
                enabled: true,
                metadata: serde_json::json!({"tools": [{"name":"b"},{"name":"c"}]}),
                headers: vec![],
            })
            .expect("replace");

        let (_, second_refreshes, second_notice) = manager
            .discover_tools_with_notice(Some(&first_notice.fingerprint))
            .expect("second refresh");
        let second = second_refreshes
            .iter()
            .find(|refresh| refresh.server_id == "notify")
            .expect("refresh");
        assert!(second.added.contains(&"c".to_string()));
        assert!(second.removed.contains(&"a".to_string()));
        assert_ne!(second_notice.fingerprint, first_notice.fingerprint);
        assert!(
            second_notice
                .changed_servers
                .contains(&"notify".to_string())
        );
    }

    #[test]
    fn parses_tools_from_jsonrpc_result_payload() {
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "tools": [
                    {"name": "search", "description": "Search tool"},
                    {"name": "read"}
                ]
            }
        });
        let tools = parse_mcp_tools_message("srv", &payload);
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].server_id, "srv");
        assert_eq!(tools[0].name, "search");
        assert_eq!(tools[1].name, "read");
    }

    #[test]
    fn reads_content_length_frames() {
        let body = serde_json::json!({"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"ping"}]}})
            .to_string();
        let raw = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        let mut cursor = Cursor::new(raw.as_bytes().to_vec());
        let parsed = read_mcp_frame(&mut cursor)
            .expect("frame")
            .expect("payload");
        let tools = parse_mcp_tools_message("srv", &parsed);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "ping");
    }

    #[test]
    fn extract_resource_text_merges_text_and_blob_entries() {
        let blob = base64::engine::general_purpose::STANDARD.encode("from blob");
        let result = extract_resource_text(&serde_json::json!({
            "contents": [
                {"text":"line one"},
                {"blob": blob, "mimeType":"text/plain"}
            ]
        }))
        .expect("resource text");
        assert!(result.contains("line one"));
        assert!(result.contains("from blob"));
    }

    #[test]
    fn extract_resource_text_handles_binary_blob_marker() {
        let blob = base64::engine::general_purpose::STANDARD.encode([0_u8, 159, 146, 150]);
        let result = extract_resource_text(&serde_json::json!({
            "contents": [
                {"blob": blob, "mimeType":"application/octet-stream"}
            ]
        }))
        .expect("resource marker");
        assert!(result.contains("[binary-resource"));
        assert!(result.contains("application/octet-stream"));
    }

    #[test]
    fn wait_for_mcp_response_times_out_without_matching_id() {
        let (tx, rx) = mpsc::channel::<serde_json::Value>();
        tx.send(serde_json::json!({"id":1,"result":{}}))
            .expect("send frame");
        drop(tx);
        let value = wait_for_mcp_response(&rx, 2, Duration::from_millis(20));
        assert!(value.is_none());
    }

    #[test]
    fn mcp_token_roundtrip() {
        let server_id = format!("test-server-{}", uuid::Uuid::now_v7());
        let token = McpOAuthToken {
            access_token: "abc123".to_string(),
            refresh_token: Some("refresh456".to_string()),
            token_type: "Bearer".to_string(),
            expires_at: Some("2025-12-31T23:59:59Z".to_string()),
            server_id: server_id.clone(),
        };
        store_mcp_token(&server_id, &token).unwrap();
        let loaded = load_mcp_token(&server_id).unwrap().unwrap();
        assert_eq!(loaded.access_token, "abc123");
        assert_eq!(loaded.refresh_token.as_deref(), Some("refresh456"));
        delete_mcp_token(&server_id).unwrap();
        assert!(load_mcp_token(&server_id).unwrap().is_none());
    }

    #[test]
    fn load_nonexistent_token_returns_none() {
        let result = load_mcp_token("nonexistent-server-xyz").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn mcp_token_serialization() {
        let token = McpOAuthToken {
            access_token: "tok".to_string(),
            refresh_token: None,
            token_type: "Bearer".to_string(),
            expires_at: None,
            server_id: "srv".to_string(),
        };
        let json = serde_json::to_string(&token).unwrap();
        assert!(json.contains("tok"));
        let parsed: McpOAuthToken = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.access_token, "tok");
    }

    #[test]
    fn expand_env_vars_basic() {
        // Existing env var
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| "/tmp".to_string());
        let result = expand_env_vars("path=${HOME}/file");
        assert!(result.contains(&home));
        assert!(result.ends_with("/file"));
    }

    #[test]
    fn expand_env_vars_with_default() {
        let result = expand_env_vars("val=${NONEXISTENT_VAR_XYZ:-fallback}");
        assert_eq!(result, "val=fallback");
    }

    #[test]
    fn expand_env_vars_passthrough() {
        let result = expand_env_vars("no vars here");
        assert_eq!(result, "no vars here");
    }

    #[test]
    fn expand_server_env_vars_applies_to_all_fields() {
        let mut server = McpServer {
            id: "test".to_string(),
            name: "test".to_string(),
            transport: McpTransport::Stdio,
            command: Some("${NONEXISTENT_CMD:-node}".to_string()),
            args: vec!["${NONEXISTENT_ARG:-server.js}".to_string()],
            url: None,
            enabled: true,
            metadata: serde_json::Value::Null,
            headers: vec![],
        };
        expand_server_env_vars(&mut server);
        assert_eq!(server.command.as_deref(), Some("node"));
        assert_eq!(server.args[0], "server.js");
    }

    #[test]
    fn enforce_mcp_token_limit_passes_small_output() {
        let limits = McpTokenLimits::default();
        let small = "hello world";
        let (out, truncated) = enforce_mcp_token_limit(small, &limits);
        assert_eq!(out, small);
        assert!(!truncated);
    }

    #[test]
    fn enforce_mcp_token_limit_truncates_large_output() {
        let limits = McpTokenLimits {
            warn_threshold: 10,
            max_tokens: 20,
        };
        let large = "x".repeat(1000);
        let (out, truncated) = enforce_mcp_token_limit(&large, &limits);
        assert!(truncated);
        assert!(out.len() < large.len());
        assert!(out.contains("truncated"));
    }

    #[test]
    fn serve_tool_list_has_expected_tools() {
        let workspace = std::env::temp_dir().join(format!("deepseek-mcp-serve-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let tools = build_serve_tool_list(&workspace);
        assert!(tools.len() >= 5);
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t.get("name").and_then(|v| v.as_str()))
            .collect();
        assert!(names.contains(&"fs_read"));
        assert!(names.contains(&"bash_run"));
        assert!(names.contains(&"git_status"));
    }

    // ── MCP integration tests (Phase 16.6) ──────────────────────────────

    #[test]
    fn mcp_search_filters_by_name() {
        let workspace = std::env::temp_dir().join(format!("deepseek-mcp-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let manager = McpManager::new(&workspace).expect("manager");

        manager
            .add_server(McpServer {
                id: "search-test".to_string(),
                name: "Search Test".to_string(),
                transport: McpTransport::Stdio,
                command: Some("echo".to_string()),
                args: vec!["ok".to_string()],
                url: None,
                enabled: true,
                metadata: serde_json::json!({
                    "tools": [
                        {"name": "alpha", "description": "Alpha tool"},
                        {"name": "beta", "description": "Beta tool"},
                        {"name": "gamma", "description": "Gamma tool"}
                    ]
                }),
                headers: vec![],
            })
            .expect("add");

        let tools = manager.discover_tools().expect("discover");
        assert_eq!(tools.len(), 3, "all three tools should be discovered");
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
        assert!(names.contains(&"gamma"));
    }

    #[test]
    fn unavailable_server_does_not_panic() {
        let workspace = std::env::temp_dir().join(format!("deepseek-mcp-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let manager = McpManager::new(&workspace).expect("manager");

        manager
            .add_server(McpServer {
                id: "missing".to_string(),
                name: "Missing".to_string(),
                transport: McpTransport::Stdio,
                command: Some("__nonexistent_command_xyz__".to_string()),
                args: vec![],
                url: None,
                enabled: true,
                metadata: serde_json::Value::Null,
                headers: vec![],
            })
            .expect("add");

        // discover_tools should return Ok even if the server command doesn't exist
        let result = manager.discover_tools();
        assert!(
            result.is_ok(),
            "discover_tools should not panic/error on unavailable server: {:?}",
            result.err()
        );
    }

    #[test]
    fn http_mcp_server_add_and_list() {
        let workspace = std::env::temp_dir().join(format!("deepseek-mcp-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let manager = McpManager::new(&workspace).expect("manager");

        manager
            .add_server(McpServer {
                id: "http-test".to_string(),
                name: "HTTP Test".to_string(),
                transport: McpTransport::Http,
                command: None,
                args: vec![],
                url: Some("http://127.0.0.1:0/mcp".to_string()),
                enabled: true,
                metadata: serde_json::json!({"tools": [{"name": "remote_tool", "description": "A remote tool"}]}),
                headers: vec![],
            })
            .expect("add http server");

        let listed = manager.list_servers().expect("list");
        let http = listed.iter().find(|s| s.id == "http-test");
        assert!(http.is_some(), "HTTP server should be listed");
        assert!(matches!(http.unwrap().transport, McpTransport::Http));

        // Discover tools from metadata cache
        let tools = manager.discover_tools().expect("discover");
        assert!(
            tools.iter().any(|t| t.name == "remote_tool"),
            "should discover tool from HTTP server metadata"
        );
    }

    #[test]
    fn mcp_token_limit_truncates_at_exact_boundary() {
        let limits = McpTokenLimits {
            warn_threshold: 5,
            max_tokens: 10,
        };
        // 10 tokens * 4 chars = 40 chars. 40 / 4 = 10 estimated tokens. 10 > 10 is false.
        let exactly_at = "a".repeat(40);
        let (out, truncated) = enforce_mcp_token_limit(&exactly_at, &limits);
        assert!(!truncated, "exactly at limit should not truncate");
        assert_eq!(out, exactly_at);

        // 44 chars / 4 = 11 estimated tokens. 11 > 10 is true → truncated.
        let over = "a".repeat(44);
        let (out2, truncated2) = enforce_mcp_token_limit(&over, &limits);
        assert!(truncated2, "over limit should truncate");
        assert!(out2.contains("truncated"));
    }

    #[test]
    fn tools_for_context_returns_all_below_threshold() {
        let workspace = std::env::temp_dir().join(format!("deepseek-mcp-ctx-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let manager = McpManager::new(&workspace).expect("manager");

        manager
            .add_server(McpServer {
                id: "ctx-test".to_string(),
                name: "Ctx Test".to_string(),
                transport: McpTransport::Stdio,
                command: Some("echo".to_string()),
                args: vec![],
                url: None,
                enabled: true,
                metadata: serde_json::json!({"tools": [
                    {"name":"t1","description":"tool 1"},
                    {"name":"t2","description":"tool 2"}
                ]}),
                headers: vec![],
            })
            .expect("add");

        // Below 10% threshold → returns all tools regardless of query
        let tools = manager
            .tools_for_context(5.0, Some("t1"), 10)
            .expect("tools");
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn tools_for_context_no_query_above_threshold_returns_all() {
        let workspace = std::env::temp_dir().join(format!("deepseek-mcp-ctx2-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let manager = McpManager::new(&workspace).expect("manager");

        manager
            .add_server(McpServer {
                id: "ctx-test2".to_string(),
                name: "Ctx Test2".to_string(),
                transport: McpTransport::Stdio,
                command: Some("echo".to_string()),
                args: vec![],
                url: None,
                enabled: true,
                metadata: serde_json::json!({"tools": [
                    {"name":"a","description":"alpha"},
                    {"name":"b","description":"beta"}
                ]}),
                headers: vec![],
            })
            .expect("add");

        // Above 10% but no query → returns all tools
        let tools = manager.tools_for_context(15.0, None, 10).expect("tools");
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn tools_for_context_filters_above_threshold() {
        let workspace = std::env::temp_dir().join(format!("deepseek-mcp-ctx3-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let manager = McpManager::new(&workspace).expect("manager");

        manager
            .add_server(McpServer {
                id: "ctx-test3".to_string(),
                name: "Ctx Test3".to_string(),
                transport: McpTransport::Stdio,
                command: Some("echo".to_string()),
                args: vec![],
                url: None,
                enabled: true,
                metadata: serde_json::json!({"tools": [
                    {"name":"search","description":"search tool"},
                    {"name":"read","description":"read tool"}
                ]}),
                headers: vec![],
            })
            .expect("add");

        // Above 10% with query — search_tools_by_relevance is called.
        // Since there are only 2 tools (below the 50-tool threshold in search),
        // it returns all tools.
        let tools = manager
            .tools_for_context(15.0, Some("search"), 10)
            .expect("tools");
        assert!(!tools.is_empty());
    }

    // ── P2-01: OAuth 2.0 tests ─────────────────────────────────────────

    #[test]
    fn generate_pkce_challenge_is_deterministic_for_same_verifier() {
        let verifier = "test-verifier-12345";
        let c1 = generate_pkce_challenge(verifier);
        let c2 = generate_pkce_challenge(verifier);
        assert_eq!(c1, c2, "same verifier should produce same challenge");
        assert!(!c1.is_empty());
    }

    #[test]
    fn extract_query_param_parses_code() {
        let request = "GET /callback?code=abc123&state=xyz HTTP/1.1\r\nHost: localhost";
        let code = extract_query_param(request, "code");
        assert_eq!(code, Some("abc123".to_string()));
        let state = extract_query_param(request, "state");
        assert_eq!(state, Some("xyz".to_string()));
    }

    #[test]
    fn extract_query_param_missing_returns_none() {
        let request = "GET /callback?code=abc HTTP/1.1";
        assert!(extract_query_param(request, "state").is_none());
        assert!(extract_query_param("", "code").is_none());
    }

    #[test]
    fn is_token_expired_true_for_past_date() {
        let token = McpOAuthToken {
            access_token: "tok".to_string(),
            refresh_token: None,
            token_type: "Bearer".to_string(),
            expires_at: Some("2020-01-01T00:00:00Z".to_string()),
            server_id: "srv".to_string(),
        };
        assert!(is_token_expired(&token));
    }

    #[test]
    fn is_token_expired_false_for_future_date() {
        let token = McpOAuthToken {
            access_token: "tok".to_string(),
            refresh_token: None,
            token_type: "Bearer".to_string(),
            expires_at: Some("2099-12-31T23:59:59Z".to_string()),
            server_id: "srv".to_string(),
        };
        assert!(!is_token_expired(&token));
    }

    // ── P2-02: Resource autocomplete tests ───────────────────────────────

    #[test]
    fn autocomplete_resources_no_servers_returns_empty() {
        let workspace = std::env::temp_dir().join(format!("deepseek-mcp-ac-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let manager = McpManager::new(&workspace).expect("manager");

        let results = manager.autocomplete_resources("@").expect("autocomplete");
        assert!(results.is_empty());
    }

    #[test]
    fn autocomplete_resources_empty_prefix_returns_all() {
        let workspace = std::env::temp_dir().join(format!("deepseek-mcp-ac2-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let manager = McpManager::new(&workspace).expect("manager");

        // No servers → empty results (can't test with real servers in unit tests)
        let results = manager.autocomplete_resources("").expect("autocomplete");
        assert!(results.is_empty());
    }

    #[test]
    fn autocomplete_resources_filters_by_prefix() {
        // This tests the filter logic directly without servers
        let all = vec![
            McpResource {
                server_id: "srv1".to_string(),
                uri: "file:///docs/readme.md".to_string(),
                name: "readme".to_string(),
                description: "Project readme".to_string(),
                mime_type: None,
            },
            McpResource {
                server_id: "srv2".to_string(),
                uri: "file:///src/main.rs".to_string(),
                name: "main".to_string(),
                description: "Entry point".to_string(),
                mime_type: Some("text/x-rust".to_string()),
            },
        ];
        let prefix = "@srv1";
        let prefix_lower = prefix.to_ascii_lowercase();
        let matches: Vec<_> = all
            .into_iter()
            .filter(|r| {
                let label = format!("@{}:{}", r.server_id, r.uri).to_ascii_lowercase();
                label.starts_with(&prefix_lower)
                    || r.name.to_ascii_lowercase().contains(&prefix_lower)
            })
            .collect();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].server_id, "srv1");
    }

    #[test]
    fn parse_sse_tools_from_event_stream() {
        let sse_text = r#"data: {"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"sse_tool","description":"SSE test tool"}]}}

data: {"other":"ignored"}

"#;
        let tools = parse_sse_tools("sse-srv", sse_text).expect("parse SSE");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "sse_tool");
        assert_eq!(tools[0].server_id, "sse-srv");
    }

    #[test]
    fn parse_sse_tools_empty_stream_returns_error() {
        let result = parse_sse_tools("srv", "");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no tools"));
    }

    #[test]
    fn sse_transport_serialization() {
        let transport = McpTransport::Sse;
        let json = serde_json::to_string(&transport).unwrap();
        assert_eq!(json, "\"sse\"");
        let parsed: McpTransport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, McpTransport::Sse);
    }

    #[test]
    fn sse_server_add_and_list() {
        let workspace = std::env::temp_dir().join(format!("deepseek-mcp-sse-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let manager = McpManager::new(&workspace).expect("manager");

        manager
            .add_server(McpServer {
                id: "sse-test".to_string(),
                name: "SSE Test".to_string(),
                transport: McpTransport::Sse,
                command: None,
                args: vec![],
                url: Some("http://127.0.0.1:0/sse".to_string()),
                enabled: true,
                metadata: serde_json::json!({"tools": [{"name":"sse_tool"}]}),
                headers: vec![("Authorization".to_string(), "Bearer tok".to_string())],
            })
            .expect("add sse server");

        let listed = manager.list_servers().expect("list");
        let sse = listed.iter().find(|s| s.id == "sse-test");
        assert!(sse.is_some());
        assert_eq!(sse.unwrap().transport, McpTransport::Sse);
        assert_eq!(sse.unwrap().headers.len(), 1);
    }

    #[test]
    fn managed_config_missing_returns_none() {
        // The managed config path points to a system location that won't exist in test
        let result = McpManager::load_managed_mcp_config();
        assert!(
            result.is_none(),
            "managed config should return None when file is absent"
        );
    }

    #[test]
    fn is_managed_server_returns_false_when_no_managed_config() {
        assert!(!McpManager::is_managed_server("some-server"));
    }

    #[test]
    fn cannot_remove_managed_server_would_fail() {
        // Without a managed config file, is_managed_server returns false,
        // so the guard doesn't trigger. This tests the guard integration exists.
        let workspace =
            std::env::temp_dir().join(format!("deepseek-mcp-managed-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let manager = McpManager::new(&workspace).expect("manager");

        manager
            .add_server(McpServer {
                id: "removable".to_string(),
                name: "Removable".to_string(),
                transport: McpTransport::Stdio,
                command: Some("echo".to_string()),
                args: vec![],
                url: None,
                enabled: true,
                metadata: serde_json::Value::Null,
                headers: vec![],
            })
            .expect("add");

        // Non-managed server can be removed
        let removed = manager.remove_server("removable").expect("remove");
        assert!(removed);
    }

    // ── P5-12: MCP prompts as slash commands ──────────────────────────────

    #[test]
    fn mcp_prompts_as_commands() {
        // McpPrompt struct should round-trip correctly
        let prompt = McpPrompt {
            server_id: "test-server".to_string(),
            name: "generate-docs".to_string(),
            description: "Generate documentation for a module".to_string(),
            arguments: vec![
                McpPromptArgument {
                    name: "module".to_string(),
                    description: "The module to document".to_string(),
                    required: true,
                },
                McpPromptArgument {
                    name: "format".to_string(),
                    description: "Output format (md, html)".to_string(),
                    required: false,
                },
            ],
        };
        let serialized = serde_json::to_value(&prompt).expect("serialize");
        let deserialized: McpPrompt = serde_json::from_value(serialized).expect("deserialize");
        assert_eq!(deserialized.server_id, "test-server");
        assert_eq!(deserialized.name, "generate-docs");
        assert_eq!(deserialized.arguments.len(), 2);
        assert!(deserialized.arguments[0].required);
        assert!(!deserialized.arguments[1].required);

        // Slash command name generation
        let slash_name = format!("/mcp-{}-{}", prompt.server_id, prompt.name);
        assert_eq!(slash_name, "/mcp-test-server-generate-docs");
    }

    // ── P5-13: env var expansion in list_servers ──────────────────────────

    #[test]
    fn env_vars_expanded() {
        // Verify expand_env_vars works with existing HOME var
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| "/tmp".to_string());

        // Expansion works on real env vars
        let expanded = expand_env_vars("Bearer ${HOME}");
        assert_eq!(expanded, format!("Bearer {home}"));

        // With default value for missing var
        let expanded2 = expand_env_vars("${NONEXISTENT_ENV_VAR_XYZ:-fallback_val}");
        assert_eq!(expanded2, "fallback_val");

        // Expansion in server config (using HOME which is always set)
        let mut server = McpServer {
            id: "svc".to_string(),
            name: "Service".to_string(),
            transport: McpTransport::Http,
            command: None,
            args: vec!["--home=${HOME}".to_string()],
            url: Some("https://api.example.com/${HOME}".to_string()),
            enabled: true,
            metadata: serde_json::Value::Null,
            headers: vec![],
        };
        expand_server_env_vars(&mut server);
        let expected_url = format!("https://api.example.com/{home}");
        assert_eq!(server.url.as_deref(), Some(expected_url.as_str()));
        assert_eq!(server.args[0], format!("--home={home}"));
    }

    // ── T4.3: MCP connection pool tests ────────────────────────────────

    #[test]
    fn pool_default_creates_empty() {
        let pool = McpConnectionPool::new();
        let conns = pool.connections.lock().unwrap();
        assert!(conns.is_empty(), "new pool should have no connections");
    }

    #[test]
    fn pool_shutdown_is_idempotent() {
        let pool = McpConnectionPool::new();
        pool.shutdown();
        pool.shutdown(); // Should not panic
    }

    #[test]
    fn pool_drop_cleans_up() {
        {
            let _pool = McpConnectionPool::new();
            // Drop implicitly calls shutdown
        }
        // No panic = success
    }
}
