use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use deepseek_store::{McpServerRecord, McpToolCacheRecord, Store};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
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
        McpTransport::Stdio => {
            let command = server
                .command
                .as_deref()
                .unwrap_or("mcp-server")
                .to_string();
            Ok(vec![McpTool {
                server_id: server.id.clone(),
                name: format!("{}.stdio", command.replace(' ', "_")),
                description: "stdio MCP placeholder tool".to_string(),
            }])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
