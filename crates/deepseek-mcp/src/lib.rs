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
        if let Some(path) = Self::user_config_path() {
            merged.extend(load_config_if_exists(&path)?.servers);
        }
        if let Some(path) = Self::user_local_config_path() {
            merged.extend(load_config_if_exists(&path)?.servers);
        }
        merged.extend(self.load_project_config()?.servers);
        merged.sort_by(|a, b| a.id.cmp(&b.id));
        merged.dedup_by(|a, b| a.id == b.id);
        Ok(merged)
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
            },
            endpoint: server.command.or(server.url).unwrap_or_default(),
            enabled: server.enabled,
            metadata_json: serde_json::to_string(&server.metadata)?,
            updated_at: Utc::now().to_rfc3339(),
        })?;
        Ok(())
    }

    pub fn remove_server(&self, id: &str) -> Result<bool> {
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
            McpTransport::Http => {
                let base_url = server
                    .url
                    .as_deref()
                    .ok_or_else(|| anyhow!("HTTP MCP server has no URL"))?;
                let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
                let resp: serde_json::Value = client
                    .post(base_url)
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
        McpTransport::Stdio => discover_stdio_tools(server),
    }
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
}
