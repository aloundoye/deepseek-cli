use anyhow::{Context, Result, anyhow};
use deepseek_core::{PluginCatalogConfig, runtime_dir};
use deepseek_store::{PluginCatalogEntryRecord, PluginStateRecord, Store};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use walkdir::WalkDir;

const PRIMARY_MANIFEST_DIR: &str = ".deepseek-plugin";
const LEGACY_MANIFEST_DIR: &str = ".claude-plugin";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    pub id: String,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub description: String,
    /// Optional LSP server configurations bundled with the plugin.
    #[serde(default)]
    pub lsp: Vec<PluginLspConfig>,
}

/// LSP server configuration bundled with a plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginLspConfig {
    /// Language identifier (e.g., "rust", "python", "typescript").
    pub language_id: String,
    /// Command to start the LSP server.
    pub command: String,
    /// Arguments to pass to the LSP server command.
    #[serde(default)]
    pub args: Vec<String>,
    /// File extensions this LSP server handles.
    #[serde(default)]
    pub extensions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInfo {
    pub manifest: PluginManifest,
    pub root: PathBuf,
    pub enabled: bool,
    pub commands: Vec<PathBuf>,
    pub agents: Vec<PathBuf>,
    pub skills: Vec<PathBuf>,
    pub hooks: Vec<PathBuf>,
    /// Discovered LSP configs (from manifest + .lsp.json files).
    #[serde(default)]
    pub lsp_configs: Vec<PluginLspConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginCommandPrompt {
    pub plugin_id: String,
    pub command_name: String,
    pub source_path: PathBuf,
    pub prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogPlugin {
    pub plugin_id: String,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub description: String,
    pub source: String,
    #[serde(default)]
    pub signature: Option<String>,
    #[serde(default)]
    pub verified: bool,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginVerifyResult {
    pub plugin_id: String,
    pub verified: bool,
    pub reason: String,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CatalogIndexShape {
    #[serde(default)]
    plugins: Vec<CatalogPluginShape>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CatalogPluginShape {
    #[serde(alias = "id")]
    plugin_id: String,
    name: String,
    version: String,
    #[serde(default)]
    description: String,
    source: String,
    #[serde(default)]
    signature: Option<String>,
    #[serde(default)]
    metadata: serde_json::Value,
}

pub struct PluginManager {
    workspace: PathBuf,
    install_root: PathBuf,
    store: Store,
}

impl PluginManager {
    pub fn new(workspace: &Path) -> Result<Self> {
        let install_root = runtime_dir(workspace).join("plugins");
        fs::create_dir_all(&install_root)?;
        Ok(Self {
            workspace: workspace.to_path_buf(),
            install_root,
            store: Store::new(workspace)?,
        })
    }

    pub fn list(&self) -> Result<Vec<PluginInfo>> {
        let states = self.store.list_plugin_states()?;
        let mut out = Vec::new();
        for state in states {
            if let Ok(mut info) = self.inspect(&state.plugin_id) {
                info.enabled = state.enabled;
                out.push(info);
            }
        }
        Ok(out)
    }

    pub fn discover(&self, search_paths: &[String]) -> Result<Vec<PluginInfo>> {
        let mut found = Vec::new();
        for sp in search_paths {
            let candidate = self.workspace.join(sp);
            if !candidate.exists() {
                continue;
            }
            if is_plugin_root(&candidate) {
                found.push(load_plugin_info(&candidate, true)?);
                continue;
            }

            for entry in fs::read_dir(candidate)? {
                let path = entry?.path();
                if path.is_dir() && is_plugin_root(&path) {
                    found.push(load_plugin_info(&path, true)?);
                }
            }
        }
        Ok(found)
    }

    pub fn install(&self, source: &Path) -> Result<PluginInfo> {
        if !is_plugin_root(source) {
            return Err(anyhow!(
                "invalid plugin root: missing .deepseek-plugin/plugin.json"
            ));
        }
        let manifest = load_manifest(source)?;
        let destination = self.install_root.join(&manifest.id);
        if destination.exists() {
            fs::remove_dir_all(&destination)?;
        }
        copy_dir(source, &destination)?;

        let info = load_plugin_info(&destination, true)?;
        self.store.set_plugin_state(&PluginStateRecord {
            plugin_id: info.manifest.id.clone(),
            name: info.manifest.name.clone(),
            version: info.manifest.version.clone(),
            path: destination.to_string_lossy().to_string(),
            enabled: true,
            manifest_json: serde_json::to_string(&info.manifest)?,
        })?;
        Ok(info)
    }

    pub fn remove(&self, plugin_id: &str) -> Result<()> {
        let destination = self.install_root.join(plugin_id);
        if destination.exists() {
            fs::remove_dir_all(destination)?;
        }
        self.store.remove_plugin_state(plugin_id)?;
        Ok(())
    }

    pub fn enable(&self, plugin_id: &str) -> Result<()> {
        let info = self.inspect(plugin_id)?;
        let manifest_json = serde_json::to_string(&info.manifest)?;
        self.store.set_plugin_state(&PluginStateRecord {
            plugin_id: info.manifest.id,
            name: info.manifest.name,
            version: info.manifest.version,
            path: info.root.to_string_lossy().to_string(),
            enabled: true,
            manifest_json,
        })
    }

    pub fn disable(&self, plugin_id: &str) -> Result<()> {
        let info = self.inspect(plugin_id)?;
        let manifest_json = serde_json::to_string(&info.manifest)?;
        self.store.set_plugin_state(&PluginStateRecord {
            plugin_id: info.manifest.id,
            name: info.manifest.name,
            version: info.manifest.version,
            path: info.root.to_string_lossy().to_string(),
            enabled: false,
            manifest_json,
        })
    }

    pub fn inspect(&self, plugin_id: &str) -> Result<PluginInfo> {
        let path = self.install_root.join(plugin_id);
        if !path.exists() {
            return Err(anyhow!("plugin not installed: {plugin_id}"));
        }
        let states = self.store.list_plugin_states()?;
        let enabled = states
            .iter()
            .find(|s| s.plugin_id == plugin_id)
            .map(|s| s.enabled)
            .unwrap_or(true);
        load_plugin_info(&path, enabled)
    }

    pub fn render_command_prompt(
        &self,
        plugin_id: &str,
        command_name: &str,
        input: Option<&str>,
    ) -> Result<PluginCommandPrompt> {
        let info = self.inspect(plugin_id)?;
        if !info.enabled {
            return Err(anyhow!("plugin is disabled: {plugin_id}"));
        }

        let source_path = info
            .commands
            .iter()
            .find(|path| command_matches(path, &info.root, command_name))
            .cloned()
            .ok_or_else(|| anyhow!("command not found: {plugin_id}:{command_name}"))?;
        let template = fs::read_to_string(&source_path).with_context(|| {
            format!(
                "failed to read plugin command template {}",
                source_path.display()
            )
        })?;
        let prompt = render_command_template(&template, plugin_id, command_name, input);

        Ok(PluginCommandPrompt {
            plugin_id: plugin_id.to_string(),
            command_name: command_name.to_string(),
            source_path,
            prompt,
        })
    }

    pub fn hook_paths_for(&self, hook_name: &str) -> Result<Vec<PathBuf>> {
        let plugins = self.list()?;
        let mut out = Vec::new();
        let needle = hook_name.to_ascii_lowercase();
        for plugin in plugins.into_iter().filter(|p| p.enabled) {
            for hook in plugin.hooks {
                let file = hook.file_name().and_then(OsStr::to_str).unwrap_or_default();
                if file.to_ascii_lowercase().starts_with(&needle) {
                    out.push(hook);
                }
            }
        }
        Ok(out)
    }

    pub fn sync_catalog(&self, cfg: &PluginCatalogConfig) -> Result<Vec<CatalogPlugin>> {
        if !cfg.enabled {
            return Ok(Vec::new());
        }
        let source = cfg.index_url.trim();
        if source.is_empty() {
            return Ok(self.catalog_from_cache());
        }

        let raw = if source.starts_with("http://") || source.starts_with("https://") {
            Client::builder()
                .timeout(Duration::from_secs(10))
                .build()?
                .get(source)
                .send()
                .with_context(|| format!("failed to fetch plugin catalog from {source}"))?
                .error_for_status()?
                .text()?
        } else {
            let path = Path::new(source);
            let resolved = if path.is_absolute() {
                path.to_path_buf()
            } else {
                self.workspace.join(path)
            };
            fs::read_to_string(&resolved).with_context(|| {
                format!("failed to read plugin catalog at {}", resolved.display())
            })?
        };

        let parsed: CatalogIndexShape = serde_json::from_str(&raw)
            .with_context(|| format!("invalid plugin catalog JSON in {}", cfg.index_url))?;
        let mut out = Vec::new();
        let mut cache_records = Vec::new();
        for p in parsed.plugins {
            let verified = verify_catalog_signature(
                cfg.signature_key.as_deref(),
                &p.plugin_id,
                &p.version,
                &p.source,
                p.signature.as_deref(),
            );
            out.push(CatalogPlugin {
                plugin_id: p.plugin_id.clone(),
                name: p.name.clone(),
                version: p.version.clone(),
                description: p.description.clone(),
                source: p.source.clone(),
                signature: p.signature.clone(),
                verified,
                metadata: p.metadata.clone(),
            });
            cache_records.push(PluginCatalogEntryRecord {
                plugin_id: p.plugin_id,
                name: p.name,
                version: p.version,
                description: p.description,
                source: p.source,
                signature: p.signature,
                verified,
                metadata_json: serde_json::to_string(&p.metadata)?,
                updated_at: String::new(),
            });
        }

        self.store
            .set_plugin_catalog_entries(source, &cache_records)?;
        Ok(out)
    }

    pub fn search_catalog(
        &self,
        query: &str,
        cfg: &PluginCatalogConfig,
    ) -> Result<Vec<CatalogPlugin>> {
        let query = query.trim().to_ascii_lowercase();
        let catalog = self
            .sync_catalog(cfg)
            .or_else(|_| Ok::<_, anyhow::Error>(self.catalog_from_cache()))?;
        if query.is_empty() {
            return Ok(catalog);
        }
        Ok(catalog
            .into_iter()
            .filter(|p| {
                p.plugin_id.to_ascii_lowercase().contains(&query)
                    || p.name.to_ascii_lowercase().contains(&query)
                    || p.description.to_ascii_lowercase().contains(&query)
            })
            .collect())
    }

    pub fn verify_catalog_plugin(
        &self,
        plugin_id: &str,
        cfg: &PluginCatalogConfig,
    ) -> Result<PluginVerifyResult> {
        let plugin_id = plugin_id.trim();
        if plugin_id.is_empty() {
            return Err(anyhow!("plugin_id cannot be empty"));
        }
        let catalog = self
            .sync_catalog(cfg)
            .or_else(|_| Ok::<_, anyhow::Error>(self.catalog_from_cache()))?;
        let plugin = catalog
            .into_iter()
            .find(|p| p.plugin_id == plugin_id)
            .ok_or_else(|| anyhow!("plugin not found in catalog: {plugin_id}"))?;

        let reason = if plugin.verified {
            "signature_valid".to_string()
        } else if cfg.signature_key.is_none() {
            "signature_key_missing".to_string()
        } else {
            "signature_invalid_or_missing".to_string()
        };
        Ok(PluginVerifyResult {
            plugin_id: plugin.plugin_id,
            verified: plugin.verified,
            reason,
            source: plugin.source,
        })
    }

    fn catalog_from_cache(&self) -> Vec<CatalogPlugin> {
        self.store
            .list_plugin_catalog_entries()
            .unwrap_or_default()
            .into_iter()
            .map(|p| CatalogPlugin {
                plugin_id: p.plugin_id,
                name: p.name,
                version: p.version,
                description: p.description,
                source: p.source,
                signature: p.signature,
                verified: p.verified,
                metadata: serde_json::from_str(&p.metadata_json).unwrap_or(serde_json::Value::Null),
            })
            .collect()
    }
}

fn verify_catalog_signature(
    signature_key: Option<&str>,
    plugin_id: &str,
    version: &str,
    source: &str,
    provided_signature: Option<&str>,
) -> bool {
    let Some(key) = signature_key else {
        return false;
    };
    let Some(provided) = provided_signature else {
        return false;
    };
    let expected = catalog_signature(key, plugin_id, version, source);
    expected.eq_ignore_ascii_case(provided)
}

fn catalog_signature(key: &str, plugin_id: &str, version: &str, source: &str) -> String {
    let mut hasher = sha2::Sha256::new();
    hasher.update(key.as_bytes());
    hasher.update([0_u8]);
    hasher.update(plugin_id.as_bytes());
    hasher.update([0_u8]);
    hasher.update(version.as_bytes());
    hasher.update([0_u8]);
    hasher.update(source.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn is_plugin_root(path: &Path) -> bool {
    manifest_file(path).is_some()
}

fn load_plugin_info(path: &Path, enabled: bool) -> Result<PluginInfo> {
    let manifest = load_manifest(path)?;

    // Collect LSP configs from manifest + .lsp.json files
    let mut lsp_configs = manifest.lsp.clone();
    let lsp_json_path = path.join(".lsp.json");
    if lsp_json_path.exists()
        && let Ok(raw) = fs::read_to_string(&lsp_json_path)
    {
        if let Ok(mut file_configs) = serde_json::from_str::<Vec<PluginLspConfig>>(&raw) {
            lsp_configs.append(&mut file_configs);
        } else if let Ok(single) = serde_json::from_str::<PluginLspConfig>(&raw) {
            lsp_configs.push(single);
        }
    }

    Ok(PluginInfo {
        manifest,
        root: path.to_path_buf(),
        enabled,
        commands: collect_files(path.join("commands"), Some("md"))?,
        agents: collect_files(path.join("agents"), Some("md"))?,
        skills: collect_files(path.join("skills"), Some("md"))?,
        hooks: collect_files(path.join("hooks"), None)?,
        lsp_configs,
    })
}

fn load_manifest(path: &Path) -> Result<PluginManifest> {
    let manifest_path = manifest_file(path)
        .ok_or_else(|| anyhow!("failed to locate plugin manifest at {}", path.display()))?;
    let raw = fs::read_to_string(&manifest_path)
        .with_context(|| format!("failed to read plugin manifest at {}", path.display()))?;
    let mut parsed: serde_json::Value = serde_json::from_str(&raw)?;
    if parsed.get("id").is_none()
        && let Some(name) = parsed.get("name").and_then(|v| v.as_str())
    {
        parsed["id"] = serde_json::Value::String(name.to_string());
    }
    Ok(serde_json::from_value(parsed)?)
}

fn collect_files(root: PathBuf, extension: Option<&str>) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in WalkDir::new(&root).into_iter().filter_map(Result::ok) {
        if !entry.path().is_file() {
            continue;
        }
        if let Some(ext) = extension
            && entry.path().extension().and_then(OsStr::to_str) != Some(ext)
        {
            continue;
        }
        files.push(entry.path().to_path_buf());
    }
    files.sort();
    Ok(files)
}

fn command_matches(path: &Path, plugin_root: &Path, requested: &str) -> bool {
    let requested = requested.trim().trim_end_matches(".md");
    if requested.is_empty() {
        return false;
    }
    let requested_normalized = requested.replace('\\', "/").to_ascii_lowercase();
    let file_name = path
        .file_name()
        .and_then(OsStr::to_str)
        .map(|s| s.trim_end_matches(".md").to_ascii_lowercase());
    if file_name.as_deref() == Some(requested_normalized.as_str()) {
        return true;
    }
    let stem = path
        .file_stem()
        .and_then(OsStr::to_str)
        .map(|s| s.to_ascii_lowercase());
    if stem.as_deref() == Some(requested_normalized.as_str()) {
        return true;
    }

    let commands_root = plugin_root.join("commands");
    if let Ok(rel) = path.strip_prefix(&commands_root) {
        let rel_no_ext = rel.with_extension("");
        let rel_normalized = rel_no_ext
            .to_string_lossy()
            .replace('\\', "/")
            .to_ascii_lowercase();
        return rel_normalized == requested_normalized;
    }
    false
}

fn render_command_template(
    template: &str,
    plugin_id: &str,
    command_name: &str,
    input: Option<&str>,
) -> String {
    let mut rendered = template
        .replace("{{plugin_id}}", plugin_id)
        .replace("{{command_name}}", command_name)
        .replace("{{input}}", input.unwrap_or_default());
    if !template.contains("{{input}}")
        && let Some(text) = input.map(str::trim)
        && !text.is_empty()
    {
        rendered.push_str("\n\nUser input:\n");
        rendered.push_str(text);
    }
    rendered
}

fn manifest_file(path: &Path) -> Option<PathBuf> {
    let primary = path.join(PRIMARY_MANIFEST_DIR).join("plugin.json");
    if primary.exists() {
        return Some(primary);
    }
    let legacy = path.join(LEGACY_MANIFEST_DIR).join("plugin.json");
    if legacy.exists() {
        return Some(legacy);
    }
    None
}

fn copy_dir(source: &Path, dest: &Path) -> Result<()> {
    fs::create_dir_all(dest)?;
    for entry in WalkDir::new(source).into_iter().filter_map(Result::ok) {
        let from = entry.path();
        let rel = from.strip_prefix(source)?;
        let to = dest.join(rel);
        if entry.file_type().is_dir() {
            fs::create_dir_all(&to)?;
        } else {
            if let Some(parent) = to.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(from, to)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_plugin_layout() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-plugin-test-{}", uuid::Uuid::now_v7()));
        fs::create_dir_all(workspace.join("plugins/demo/.deepseek-plugin")).expect("create dirs");
        fs::write(
            workspace.join("plugins/demo/.deepseek-plugin/plugin.json"),
            r#"{"id":"demo","name":"Demo","version":"0.1.0"}"#,
        )
        .expect("manifest");
        fs::create_dir_all(workspace.join("plugins/demo/commands")).expect("commands");
        fs::write(workspace.join("plugins/demo/commands/x.md"), "# cmd").expect("cmd");

        let manager = PluginManager::new(&workspace).expect("manager");
        let found = manager
            .discover(&["plugins".to_string()])
            .expect("discover plugins");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].manifest.id, "demo");
    }

    #[test]
    fn renders_plugin_command_prompt_with_input_substitution() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-plugin-test-{}", uuid::Uuid::now_v7()));
        let plugin = workspace.join("plugin-src");
        fs::create_dir_all(plugin.join(".deepseek-plugin")).expect("create dirs");
        fs::create_dir_all(plugin.join("commands")).expect("commands dir");
        fs::write(
            plugin.join(".deepseek-plugin/plugin.json"),
            r#"{"id":"demo","name":"Demo","version":"0.1.0"}"#,
        )
        .expect("manifest");
        fs::write(
            plugin.join("commands/review.md"),
            "Review task: {{input}}\nplugin={{plugin_id}} cmd={{command_name}}",
        )
        .expect("command template");

        let manager = PluginManager::new(&workspace).expect("manager");
        manager.install(&plugin).expect("install");
        let rendered = manager
            .render_command_prompt("demo", "review", Some("check migrations"))
            .expect("render");
        assert!(rendered.prompt.contains("check migrations"));
        assert!(rendered.prompt.contains("plugin=demo"));
        assert!(rendered.prompt.contains("cmd=review"));
    }
}
