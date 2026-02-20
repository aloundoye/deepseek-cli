mod plugins;
mod shell;

use anyhow::{Result, anyhow};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use chrono::Utc;
use deepseek_chrome::{ChromeSession, ScreenshotFormat};
use deepseek_core::{
    AppConfig, ApprovedToolCall, EventEnvelope, EventKind, FunctionDefinition, ToolCall,
    ToolDefinition, ToolHost, ToolProposal, ToolResult,
};
use deepseek_diff::PatchStore;
use deepseek_hooks::{HookContext, HookRuntime};
use deepseek_index::IndexService;
use deepseek_memory::MemoryManager;
use deepseek_policy::PolicyEngine;
use deepseek_store::Store;
use ignore::WalkBuilder;
pub use plugins::{
    CatalogPlugin, PluginCommandPrompt, PluginInfo, PluginManager, PluginVerifyResult,
};
use serde_json::json;
use sha2::Digest;
pub use shell::{PlatformShellRunner, ShellRunResult, ShellRunner};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

const DEFAULT_TIMEOUT_SECONDS: u64 = 120;
const READ_MAX_BYTES_DEFAULT: usize = 1_000_000;

/// Tools that are forbidden during review mode (read-only pipeline).
const REVIEW_BLOCKED_TOOLS: &[&str] = &[
    "fs.write",
    "fs.edit",
    "multi_edit",
    "patch.stage",
    "patch.apply",
    "bash.run",
    "notebook.edit",
];

pub struct LocalToolHost {
    workspace: PathBuf,
    policy: PolicyEngine,
    sandbox_mode: String,
    sandbox_wrapper: Option<String>,
    patches: PatchStore,
    index: IndexService,
    store: Store,
    runner: Arc<dyn ShellRunner + Send + Sync>,
    plugins: Option<PluginManager>,
    hooks_enabled: bool,
    visual_verification_enabled: bool,
    lint_after_edit: Option<String>,
    review_mode: bool,
}

impl LocalToolHost {
    pub fn new(workspace: &Path, policy: PolicyEngine) -> Result<Self> {
        Self::with_runner(workspace, policy, Arc::new(PlatformShellRunner))
    }

    pub fn with_runner(
        workspace: &Path,
        policy: PolicyEngine,
        runner: Arc<dyn ShellRunner + Send + Sync>,
    ) -> Result<Self> {
        let cfg = AppConfig::load(workspace).unwrap_or_default();
        let sandbox_mode = policy.sandbox_mode().to_ascii_lowercase();
        let sandbox_wrapper = cfg
            .policy
            .sandbox_wrapper
            .as_ref()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let lint_after_edit = cfg
            .policy
            .lint_after_edit
            .as_ref()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        Ok(Self {
            workspace: workspace.to_path_buf(),
            patches: PatchStore::new(workspace)?,
            index: IndexService::new(workspace)?,
            store: Store::new(workspace)?,
            policy,
            sandbox_mode,
            sandbox_wrapper,
            runner,
            plugins: PluginManager::new(workspace).ok(),
            hooks_enabled: cfg.plugins.enabled && cfg.plugins.enable_hooks,
            visual_verification_enabled: cfg.experiments.visual_verification,
            lint_after_edit,
            review_mode: false,
        })
    }

    /// Enable review mode (read-only pipeline).
    pub fn set_review_mode(&mut self, enabled: bool) {
        self.review_mode = enabled;
    }

    pub fn is_review_mode(&self) -> bool {
        self.review_mode
    }

    fn chrome_port_from_call(call: &ToolCall) -> Result<u16> {
        let port = call
            .args
            .get("port")
            .and_then(|v| v.as_u64())
            .unwrap_or(9222);
        u16::try_from(port).map_err(|_| anyhow!("invalid chrome debug port: {port}"))
    }

    fn chrome_session_from_call(call: &ToolCall) -> Result<ChromeSession> {
        let port = Self::chrome_port_from_call(call)?;
        let mut session = ChromeSession::new(port)?;
        session.check_connection()?;
        Ok(session)
    }

    fn run_tool(&self, call: &ToolCall) -> Result<serde_json::Value> {
        // Enforce review mode: block all non-read tools
        if self.review_mode && REVIEW_BLOCKED_TOOLS.contains(&call.name.as_str()) {
            return Err(anyhow!(
                "tool '{}' is blocked during review mode (read-only pipeline)",
                call.name
            ));
        }
        match call.name.as_str() {
            "fs.list" => {
                let dir = call.args.get("dir").and_then(|v| v.as_str()).unwrap_or(".");
                self.policy.check_path(dir)?;
                let path = self.workspace.join(dir);
                let mut out = Vec::new();
                for entry in fs::read_dir(path)? {
                    let e = entry?;
                    out.push(e.file_name().to_string_lossy().to_string());
                }
                Ok(json!({"entries": out}))
            }
            "fs.read" => {
                let path = call
                    .args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("path missing"))?;
                self.policy.check_path(path)?;
                let full = self.workspace.join(path);
                let bytes = fs::read(&full)?;
                let sha = format!("{:x}", sha2::Sha256::digest(&bytes));
                let mime = guess_mime(&full);
                let max_bytes = call
                    .args
                    .get("max_bytes")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(READ_MAX_BYTES_DEFAULT);
                let start_line = call
                    .args
                    .get("start_line")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
                let end_line = call
                    .args
                    .get("end_line")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);

                let binary = is_binary(&bytes);
                if binary {
                    if self.visual_verification_enabled
                        && (mime.starts_with("image/") || mime == "application/pdf")
                    {
                        self.emit_visual_artifact_event(path, mime);
                    }
                    // Return base64 for images so multimodal models can process them
                    if mime.starts_with("image/") {
                        let encoded = BASE64.encode(&bytes);
                        return Ok(json!({
                            "path": path,
                            "mime": mime,
                            "binary": true,
                            "size_bytes": bytes.len(),
                            "sha256": sha,
                            "base64": encoded
                        }));
                    }
                    // PDF text extraction
                    if mime == "application/pdf" {
                        let pages_arg = call.args.get("pages").and_then(|v| v.as_str());
                        match extract_pdf_text(&full, pages_arg) {
                            Ok(text) => {
                                return Ok(json!({
                                    "path": path,
                                    "mime": mime,
                                    "binary": true,
                                    "size_bytes": bytes.len(),
                                    "sha256": sha,
                                    "content": text,
                                    "pages": pages_arg.unwrap_or("all")
                                }));
                            }
                            Err(_) => {
                                // Fall through to generic binary return if extraction fails
                            }
                        }
                    }
                    return Ok(json!({
                        "path": path,
                        "mime": mime,
                        "binary": true,
                        "size_bytes": bytes.len(),
                        "sha256": sha
                    }));
                }

                let truncated = if bytes.len() > max_bytes {
                    // Find the last valid UTF-8 char boundary to avoid splitting
                    // a multi-byte character, which would cause from_utf8 to fail.
                    let mut end = max_bytes;
                    while end > 0 && std::str::from_utf8(&bytes[..end]).is_err() {
                        end -= 1;
                    }
                    bytes[..end].to_vec()
                } else {
                    bytes.clone()
                };
                let content = String::from_utf8(truncated)?;
                let lines = collect_lines(&content, start_line, end_line);
                Ok(json!({
                    "path": path,
                    "mime": mime,
                    "binary": false,
                    "size_bytes": bytes.len(),
                    "truncated": bytes.len() > max_bytes,
                    "sha256": sha,
                    "content": content,
                    "lines": lines
                }))
            }
            "fs.glob" => {
                let pattern = call
                    .args
                    .get("pattern")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("pattern missing"))?;
                let limit = call
                    .args
                    .get("limit")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(200) as usize;
                let respect_gitignore = call
                    .args
                    .get("respectGitignore")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let base = call
                    .args
                    .get("base")
                    .and_then(|v| v.as_str())
                    .unwrap_or(".");
                self.policy.check_path(base)?;
                let base_path = self.workspace.join(base);
                let mut matches = Vec::new();
                let compiled = glob::Pattern::new(pattern)
                    .map_err(|err| anyhow!("invalid glob pattern '{pattern}': {err}"))?;
                for path in walk_paths(&base_path, &self.workspace, respect_gitignore) {
                    let rel_path = match path.strip_prefix(&self.workspace) {
                        Ok(rel) => rel,
                        Err(_) => continue,
                    };
                    if should_skip_rel_path(rel_path) {
                        continue;
                    }
                    let rel = normalize_rel_path(rel_path);
                    if compiled.matches(&rel) {
                        matches.push(json!({
                            "path": rel,
                            "is_dir": path.is_dir()
                        }));
                        if matches.len() >= limit {
                            break;
                        }
                    }
                }
                Ok(json!({
                    "pattern": pattern,
                    "matches": matches
                }))
            }
            "fs.grep" => {
                let pattern = call
                    .args
                    .get("pattern")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("pattern missing"))?;
                let limit = call
                    .args
                    .get("limit")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(200) as usize;
                let glob_pattern = call
                    .args
                    .get("glob")
                    .and_then(|v| v.as_str())
                    .unwrap_or("**/*");
                let respect_gitignore = call
                    .args
                    .get("respectGitignore")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let case_sensitive = call
                    .args
                    .get("case_sensitive")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let compiled_glob = glob::Pattern::new(glob_pattern)
                    .map_err(|err| anyhow!("invalid glob pattern '{glob_pattern}': {err}"))?;
                let regex = regex::RegexBuilder::new(pattern)
                    .case_insensitive(!case_sensitive)
                    .build()?;
                let mut matches = Vec::new();
                for path in walk_paths(&self.workspace, &self.workspace, respect_gitignore) {
                    if !path.is_file() {
                        continue;
                    }
                    let rel_path = match path.strip_prefix(&self.workspace) {
                        Ok(rel) => rel,
                        Err(_) => continue,
                    };
                    if should_skip_rel_path(rel_path) {
                        continue;
                    }
                    let rel = normalize_rel_path(rel_path);
                    if !compiled_glob.matches(&rel) {
                        continue;
                    }
                    let bytes = match fs::read(&path) {
                        Ok(bytes) => bytes,
                        Err(_) => continue,
                    };
                    if is_binary(&bytes) {
                        continue;
                    }
                    let content = match String::from_utf8(bytes) {
                        Ok(content) => content,
                        Err(_) => continue,
                    };
                    for (idx, line) in content.lines().enumerate() {
                        if regex.is_match(line) {
                            matches.push(json!({
                                "path": rel,
                                "line": idx + 1,
                                "text": line
                            }));
                            if matches.len() >= limit {
                                return Ok(json!({
                                    "pattern": pattern,
                                    "glob": glob_pattern,
                                    "matches": matches
                                }));
                            }
                        }
                    }
                }
                Ok(json!({
                    "pattern": pattern,
                    "glob": glob_pattern,
                    "matches": matches
                }))
            }
            "fs.edit" => {
                let path = call
                    .args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("path missing"))?;
                self.policy.check_path(path)?;
                let full = self.workspace.join(path);
                let before = fs::read_to_string(&full)?;
                let mut after = before.clone();
                let mut replacements = 0usize;

                if let Some(edits) = call.args.get("edits").and_then(|v| v.as_array()) {
                    for edit in edits {
                        replacements += apply_single_edit(&mut after, edit)?;
                    }
                } else {
                    replacements += apply_single_edit(&mut after, &call.args)?;
                }

                if after == before {
                    return Ok(json!({
                        "path": path,
                        "edited": false,
                        "replacements": 0
                    }));
                }

                let diff = generate_unified_diff(path, &before, &after);
                let checkpoint = self.create_checkpoint("fs_edit")?;
                fs::write(&full, &after)?;
                let before_sha = format!("{:x}", sha2::Sha256::digest(before.as_bytes()));
                let after_sha = format!("{:x}", sha2::Sha256::digest(after.as_bytes()));

                // Auto-lint after edit (from Aider: lint-fix loop)
                let lint_output = if let Some(ref lint_cmd) = self.lint_after_edit {
                    match self
                        .runner
                        .run(lint_cmd, &self.workspace, Duration::from_secs(30))
                    {
                        Ok(result) => {
                            if result.status.unwrap_or(1) != 0 {
                                Some(json!({
                                    "lint_command": lint_cmd,
                                    "lint_passed": false,
                                    "lint_stdout": result.stdout,
                                    "lint_stderr": result.stderr,
                                }))
                            } else {
                                Some(json!({
                                    "lint_command": lint_cmd,
                                    "lint_passed": true,
                                }))
                            }
                        }
                        Err(_) => None,
                    }
                } else {
                    None
                };

                let mut result = json!({
                    "path": path,
                    "edited": true,
                    "replacements": replacements,
                    "diff": diff,
                    "before_sha256": before_sha,
                    "after_sha256": after_sha,
                    "checkpoint_id": checkpoint.map(|id| id.to_string())
                });
                if let Some(lint) = lint_output {
                    result["lint"] = lint;
                }
                Ok(result)
            }
            "fs.search_rg" => {
                let q = call
                    .args
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("query missing"))?;
                let limit = call
                    .args
                    .get("limit")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(20) as usize;
                let mut matches = Vec::new();
                for path in walk_paths(&self.workspace, &self.workspace, true) {
                    if !path.is_file() {
                        continue;
                    }
                    let rel_path = path.strip_prefix(&self.workspace)?;
                    if should_skip_rel_path(rel_path) {
                        continue;
                    }
                    let rel = rel_path.to_string_lossy().to_string();
                    if let Ok(content) = fs::read_to_string(&path) {
                        for (idx, line) in content.lines().enumerate() {
                            if line.contains(q) {
                                matches.push(json!({"path": rel, "line": idx + 1, "text": line}));
                                if matches.len() >= limit {
                                    return Ok(json!({"matches": matches}));
                                }
                            }
                        }
                    }
                }
                Ok(json!({"matches": matches}))
            }
            "git.status" => self.run_cmd("git status --short", DEFAULT_TIMEOUT_SECONDS),
            "git.diff" => self.run_cmd("git diff", DEFAULT_TIMEOUT_SECONDS),
            "git.show" => {
                let spec = call
                    .args
                    .get("spec")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("spec missing"))?;
                // Reject specs containing shell metacharacters to prevent injection.
                if spec.contains([
                    ';', '|', '&', '$', '`', '(', ')', '{', '}', '<', '>', '\'', '"', '\\', '\n',
                ]) {
                    return Err(anyhow!("git.show: spec contains forbidden characters"));
                }
                self.run_cmd(&format!("git show {spec}"), DEFAULT_TIMEOUT_SECONDS)
            }
            "index.query" => {
                let q = call
                    .args
                    .get("q")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("q missing"))?;
                let top_k = call
                    .args
                    .get("top_k")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10) as usize;
                Ok(serde_json::to_value(self.index.query(q, top_k)?)?)
            }
            "patch.stage" => {
                let diff = call
                    .args
                    .get("unified_diff")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("unified_diff missing"))?;
                let base = call
                    .args
                    .get("base")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .as_bytes()
                    .to_vec();
                let patch = self.patches.stage(diff, &base)?;
                Ok(
                    json!({"patch_id": patch.patch_id.to_string(), "base_sha256": patch.base_sha256}),
                )
            }
            "patch.apply" => {
                let id = call
                    .args
                    .get("patch_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("patch_id missing"))?;
                let patch_id = Uuid::parse_str(id)?;
                let (applied, conflicts) = self.patches.apply(&self.workspace, patch_id)?;
                Ok(json!({"patch_id": id, "applied": applied, "conflicts": conflicts}))
            }
            "fs.write" => {
                let path = call
                    .args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("path missing"))?;
                self.policy.check_path(path)?;
                let content = call
                    .args
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("content missing"))?;
                let full = self.workspace.join(path);
                if let Some(parent) = full.parent() {
                    fs::create_dir_all(parent)?;
                }
                let checkpoint = self.create_checkpoint("fs_write")?;
                fs::write(full, content)?;
                Ok(json!({
                    "written": true,
                    "checkpoint_id": checkpoint.map(|id| id.to_string())
                }))
            }
            "bash.run" => {
                let cmd = call
                    .args
                    .get("cmd")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("cmd missing"))?;
                self.enforce_sandbox_mode(cmd)?;
                self.policy.check_command(cmd)?;
                let timeout = call
                    .args
                    .get("timeout")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(DEFAULT_TIMEOUT_SECONDS);
                self.run_bash_cmd(cmd, timeout)
            }
            "web.fetch" => {
                let url = call
                    .args
                    .get("url")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("url missing"))?;
                if !url.starts_with("http://") && !url.starts_with("https://") {
                    return Err(anyhow!("url must start with http:// or https://"));
                }
                let max_bytes = call
                    .args
                    .get("max_bytes")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(500_000) as usize;
                let timeout = call
                    .args
                    .get("timeout")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(30);
                let client = reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(timeout))
                    .user_agent("deepseek-cli/0.1")
                    .build()?;
                let resp = client.get(url).send()?;
                let status = resp.status().as_u16();
                let content_type = resp
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("unknown")
                    .to_string();
                let body = resp.text()?;
                let truncated = body.len() > max_bytes;
                let text = if truncated {
                    &body[..body.floor_char_boundary(max_bytes)]
                } else {
                    &body
                };
                // Strip HTML tags for readable text extraction
                let content = if content_type.contains("html") {
                    strip_html_tags(text)
                } else {
                    text.to_string()
                };
                Ok(json!({
                    "url": url,
                    "status": status,
                    "content_type": content_type,
                    "content": content,
                    "truncated": truncated,
                    "bytes": body.len()
                }))
            }
            "web.search" => {
                let query = call
                    .args
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("query missing"))?;
                let max_results = call
                    .args
                    .get("max_results")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10) as usize;

                // Check cache first
                let query_hash = format!("{:x}", sha2::Sha256::digest(query.as_bytes()));
                if let Ok(Some(cached)) = self.store.get_web_search_cache(&query_hash) {
                    return Ok(json!({
                        "query": query,
                        "results": serde_json::from_str::<serde_json::Value>(&cached.results_json).unwrap_or(json!([])),
                        "cached": true,
                        "results_count": cached.results_count,
                        "provenance": {
                            "source": "cache",
                            "cached_at": cached.cached_at,
                        }
                    }));
                }

                // Perform web search via HTML scraping of a search engine
                let search_url = format!(
                    "https://html.duckduckgo.com/html/?q={}",
                    url_encode_query(query)
                );
                let client = reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(15))
                    .user_agent("deepseek-cli/0.1")
                    .build()?;
                let resp = client.get(&search_url).send()?;
                let body = resp.text()?;

                // Parse search results from HTML
                let results = parse_search_results(&body, max_results);
                let results_json = serde_json::to_string(&results)?;
                let results_count = results.len() as u64;

                // Cache the results
                let _ = self
                    .store
                    .set_web_search_cache(&deepseek_store::WebSearchCacheRecord {
                        query_hash: query_hash.clone(),
                        query: query.to_string(),
                        results_json: results_json.clone(),
                        results_count,
                        cached_at: chrono::Utc::now().to_rfc3339(),
                        ttl_seconds: 900, // 15 minutes
                    });

                // Emit event
                let seq = self.store.next_seq_no(uuid::Uuid::nil()).unwrap_or(1);
                let _ = self.store.append_event(&deepseek_core::EventEnvelope {
                    seq_no: seq,
                    at: chrono::Utc::now(),
                    session_id: uuid::Uuid::nil(),
                    kind: deepseek_core::EventKind::WebSearchExecutedV1 {
                        query: query.to_string(),
                        results_count,
                        cached: false,
                    },
                });

                Ok(json!({
                    "query": query,
                    "results": results,
                    "cached": false,
                    "results_count": results_count,
                    "provenance": {
                        "source": "duckduckgo",
                        "searched_at": chrono::Utc::now().to_rfc3339(),
                    }
                }))
            }
            "chrome.navigate" => {
                let url = call
                    .args
                    .get("url")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("url missing"))?;
                if !url.starts_with("http://")
                    && !url.starts_with("https://")
                    && !url.starts_with("about:")
                {
                    return Err(anyhow!("url must start with http://, https://, or about:"));
                }
                let session = Self::chrome_session_from_call(call)?;
                let result = session.navigate(url)?;
                Ok(json!({
                    "url": url,
                    "cdp": result,
                    "ok": result.error.is_none()
                }))
            }
            "chrome.click" => {
                let selector = call
                    .args
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("selector missing"))?;
                let session = Self::chrome_session_from_call(call)?;
                let result = session.click(selector)?;
                Ok(json!({
                    "selector": selector,
                    "cdp": result,
                    "ok": result.error.is_none()
                }))
            }
            "chrome.type_text" => {
                let selector = call
                    .args
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("selector missing"))?;
                let text = call
                    .args
                    .get("text")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("text missing"))?;
                let session = Self::chrome_session_from_call(call)?;
                let result = session.type_text(selector, text)?;
                Ok(json!({
                    "selector": selector,
                    "text": text,
                    "cdp": result,
                    "ok": result.error.is_none()
                }))
            }
            "chrome.screenshot" => {
                let format = match call
                    .args
                    .get("format")
                    .and_then(|v| v.as_str())
                    .unwrap_or("png")
                    .to_ascii_lowercase()
                    .as_str()
                {
                    "png" => ScreenshotFormat::Png,
                    "jpeg" | "jpg" => ScreenshotFormat::Jpeg,
                    "webp" => ScreenshotFormat::Webp,
                    other => return Err(anyhow!("unsupported screenshot format: {other}")),
                };
                let session = Self::chrome_session_from_call(call)?;
                let image_base64 = session.screenshot(format)?;
                Ok(json!({
                    "format": call.args.get("format").and_then(|v| v.as_str()).unwrap_or("png"),
                    "base64": image_base64
                }))
            }
            "chrome.read_console" => {
                let session = Self::chrome_session_from_call(call)?;
                let entries = session.read_console()?;
                Ok(json!({
                    "entries": entries,
                    "count": entries.len()
                }))
            }
            "chrome.evaluate" => {
                let expression = call
                    .args
                    .get("expression")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("expression missing"))?;
                let session = Self::chrome_session_from_call(call)?;
                let value = session.evaluate(expression)?;
                Ok(json!({
                    "expression": expression,
                    "value": value
                }))
            }
            "notebook.read" => {
                let path = call
                    .args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("path missing"))?;
                self.policy.check_path(path)?;
                let full = self.workspace.join(path);
                let content = fs::read_to_string(&full)?;
                let nb: serde_json::Value = serde_json::from_str(&content)
                    .map_err(|e| anyhow!("invalid notebook JSON: {e}"))?;
                let cells = nb
                    .get("cells")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| anyhow!("notebook has no cells array"))?;
                let cell_summaries: Vec<serde_json::Value> = cells
                    .iter()
                    .enumerate()
                    .map(|(i, cell)| {
                        let cell_type = cell
                            .get("cell_type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        let source = cell
                            .get("source")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|s| s.as_str())
                                    .collect::<Vec<_>>()
                                    .join("")
                            })
                            .or_else(|| {
                                cell.get("source")
                                    .and_then(|v| v.as_str())
                                    .map(String::from)
                            })
                            .unwrap_or_default();
                        let preview: String = source.chars().take(200).collect();
                        json!({
                            "index": i,
                            "cell_type": cell_type,
                            "source_preview": preview,
                            "source_length": source.len()
                        })
                    })
                    .collect();
                Ok(json!({
                    "path": path,
                    "cells_count": cells.len(),
                    "cells": cell_summaries
                }))
            }
            "notebook.edit" => {
                let path = call
                    .args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("path missing"))?;
                self.policy.check_path(path)?;
                let full = self.workspace.join(path);
                let content = fs::read_to_string(&full)?;
                let mut nb: serde_json::Value = serde_json::from_str(&content)
                    .map_err(|e| anyhow!("invalid notebook JSON: {e}"))?;
                let operation = call
                    .args
                    .get("operation")
                    .and_then(|v| v.as_str())
                    .unwrap_or("replace");
                let cell_index =
                    call.args
                        .get("cell_index")
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| anyhow!("cell_index missing"))? as usize;
                let cells = nb
                    .get_mut("cells")
                    .and_then(|v| v.as_array_mut())
                    .ok_or_else(|| anyhow!("notebook has no cells array"))?;
                let checkpoint = self.create_checkpoint("notebook_edit")?;
                match operation {
                    "replace" => {
                        if cell_index >= cells.len() {
                            return Err(anyhow!(
                                "cell_index {cell_index} out of range ({} cells)",
                                cells.len()
                            ));
                        }
                        let new_source = call
                            .args
                            .get("new_source")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| anyhow!("new_source missing for replace"))?;
                        let source_lines: Vec<serde_json::Value> = new_source
                            .lines()
                            .map(|l| json!(format!("{l}\n")))
                            .collect();
                        cells[cell_index]["source"] = json!(source_lines);
                        if let Some(ct) = call.args.get("cell_type").and_then(|v| v.as_str()) {
                            cells[cell_index]["cell_type"] = json!(ct);
                        }
                    }
                    "insert" => {
                        if cell_index > cells.len() {
                            return Err(anyhow!(
                                "cell_index {cell_index} out of range for insert ({} cells)",
                                cells.len()
                            ));
                        }
                        let new_source = call
                            .args
                            .get("new_source")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let cell_type = call
                            .args
                            .get("cell_type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("code");
                        let source_lines: Vec<serde_json::Value> = new_source
                            .lines()
                            .map(|l| json!(format!("{l}\n")))
                            .collect();
                        let new_cell = json!({
                            "cell_type": cell_type,
                            "source": source_lines,
                            "metadata": {},
                            "outputs": [],
                            "execution_count": null
                        });
                        cells.insert(cell_index, new_cell);
                    }
                    "delete" => {
                        if cell_index >= cells.len() {
                            return Err(anyhow!(
                                "cell_index {cell_index} out of range ({} cells)",
                                cells.len()
                            ));
                        }
                        cells.remove(cell_index);
                    }
                    _ => return Err(anyhow!("unknown notebook operation: {operation}")),
                }
                let updated = serde_json::to_string_pretty(&nb)?;
                fs::write(&full, &updated)?;
                Ok(json!({
                    "path": path,
                    "operation": operation,
                    "cell_index": cell_index,
                    "cells_count": nb.get("cells").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0),
                    "checkpoint_id": checkpoint.map(|id| id.to_string())
                }))
            }
            "multi_edit" => {
                let files = call
                    .args
                    .get("files")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| anyhow!("files array missing"))?;
                let checkpoint = self.create_checkpoint("multi_edit")?;
                let mut results = Vec::new();
                let mut total_replacements = 0usize;
                let mut all_succeeded = true;

                for file_entry in files {
                    let path = file_entry
                        .get("path")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow!("path missing in file entry"))?;
                    self.policy.check_path(path)?;
                    let full = self.workspace.join(path);
                    let before = fs::read_to_string(&full)?;
                    let mut after = before.clone();
                    let mut file_replacements = 0usize;

                    if let Some(edits) = file_entry.get("edits").and_then(|v| v.as_array()) {
                        for edit in edits {
                            match apply_single_edit(&mut after, edit) {
                                Ok(count) => file_replacements += count,
                                Err(e) => {
                                    all_succeeded = false;
                                    results.push(json!({
                                        "path": path,
                                        "edited": false,
                                        "error": e.to_string()
                                    }));
                                    continue;
                                }
                            }
                        }
                    }

                    if after == before {
                        results.push(json!({"path": path, "edited": false, "replacements": 0}));
                        continue;
                    }

                    let diff = generate_unified_diff(path, &before, &after);
                    fs::write(&full, &after)?;
                    let before_sha = format!("{:x}", sha2::Sha256::digest(before.as_bytes()));
                    let after_sha = format!("{:x}", sha2::Sha256::digest(after.as_bytes()));
                    total_replacements += file_replacements;

                    results.push(json!({
                        "path": path,
                        "edited": true,
                        "replacements": file_replacements,
                        "diff": diff,
                        "before_sha256": before_sha,
                        "after_sha256": after_sha,
                    }));
                }

                // Run lint once after all edits
                let lint_output = if let Some(ref lint_cmd) = self.lint_after_edit {
                    match self
                        .runner
                        .run(lint_cmd, &self.workspace, Duration::from_secs(30))
                    {
                        Ok(result) => {
                            if result.status.unwrap_or(1) != 0 {
                                Some(json!({
                                    "lint_command": lint_cmd,
                                    "lint_passed": false,
                                    "lint_stdout": result.stdout,
                                    "lint_stderr": result.stderr,
                                }))
                            } else {
                                Some(json!({
                                    "lint_command": lint_cmd,
                                    "lint_passed": true,
                                }))
                            }
                        }
                        Err(_) => None,
                    }
                } else {
                    None
                };

                let mut result = json!({
                    "results": results,
                    "total_files": files.len(),
                    "total_replacements": total_replacements,
                    "all_succeeded": all_succeeded,
                    "checkpoint_id": checkpoint.map(|id| id.to_string()),
                });
                if let Some(lint) = lint_output {
                    result["lint"] = lint;
                }
                Ok(result)
            }
            "diagnostics.check" => {
                let target = call.args.get("path").and_then(|v| v.as_str());
                let (cmd, source) = detect_diagnostics_command(&self.workspace, target)?;
                let result = self
                    .runner
                    .run(&cmd, &self.workspace, Duration::from_secs(60))?;
                let diagnostics = parse_diagnostics(&result.stdout, &result.stderr, source);
                Ok(json!({
                    "diagnostics": diagnostics,
                    "command": cmd,
                    "success": result.status.unwrap_or(1) == 0,
                    "source": source,
                }))
            }
            _ => Err(anyhow!("unknown tool: {}", call.name)),
        }
    }

    fn run_cmd(&self, cmd: &str, timeout_secs: u64) -> Result<serde_json::Value> {
        let result = self
            .runner
            .run(cmd, &self.workspace, Duration::from_secs(timeout_secs))?;
        Ok(json!({
            "status": result.status,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
        }))
    }

    fn run_bash_cmd(&self, cmd: &str, timeout_secs: u64) -> Result<serde_json::Value> {
        if is_isolated_sandbox_mode(&self.sandbox_mode) {
            return self.run_cmd_in_isolated_sandbox(cmd, timeout_secs);
        }
        // Apply OS-level sandbox if enabled
        let sandbox_cfg = AppConfig::load(&self.workspace)
            .unwrap_or_default()
            .policy
            .sandbox;
        let sandboxed_cmd = sandbox_wrap_command(&self.workspace, cmd, &sandbox_cfg);
        self.run_cmd(&sandboxed_cmd, timeout_secs)
    }

    fn run_cmd_in_isolated_sandbox(
        &self,
        cmd: &str,
        timeout_secs: u64,
    ) -> Result<serde_json::Value> {
        let workspace =
            std::fs::canonicalize(&self.workspace).unwrap_or_else(|_| self.workspace.clone());
        if let Some(template) = self.sandbox_wrapper.as_deref() {
            let wrapped = render_wrapper_template(template, &workspace, cmd)?;
            return self.run_cmd(&wrapped, timeout_secs);
        }
        if let Ok(template) = std::env::var("DEEPSEEK_SANDBOX_WRAPPER")
            && !template.trim().is_empty()
        {
            let wrapped = render_wrapper_template(template.trim(), &workspace, cmd)?;
            return self.run_cmd(&wrapped, timeout_secs);
        }
        if let Some(wrapped) = auto_isolated_wrapper_command(&workspace, cmd) {
            return self.run_cmd(&wrapped, timeout_secs);
        }
        Err(anyhow!(
            "sandbox_mode={} requires an OS-level wrapper (set policy.sandbox_wrapper or DEEPSEEK_SANDBOX_WRAPPER) or install bwrap/firejail/sandbox-exec",
            self.sandbox_mode
        ))
    }

    fn enforce_sandbox_mode(&self, cmd: &str) -> Result<()> {
        match self.sandbox_mode.as_str() {
            "read-only" | "readonly" => {
                if command_has_mutating_intent(cmd) {
                    return Err(anyhow!(
                        "sandbox_mode=read-only blocked mutating command: {}",
                        cmd
                    ));
                }
                if command_has_network_egress_intent(cmd) {
                    return Err(anyhow!(
                        "sandbox_mode=read-only blocked network command: {}",
                        cmd
                    ));
                }
            }
            "workspace-write" | "workspace_write" => {
                if command_references_outside_workspace(cmd, &self.workspace) {
                    return Err(anyhow!(
                        "sandbox_mode=workspace-write blocked path outside workspace: {}",
                        cmd
                    ));
                }
                if command_has_network_egress_intent(cmd) {
                    return Err(anyhow!(
                        "sandbox_mode=workspace-write blocked network command: {}",
                        cmd
                    ));
                }
            }
            "isolated" | "container" | "os-sandbox" | "os_sandbox" => {
                // Defer strict containment to the configured OS-level wrapper.
            }
            _ => {}
        }
        Ok(())
    }
}

impl ToolHost for LocalToolHost {
    fn propose(&self, call: ToolCall) -> ToolProposal {
        ToolProposal {
            invocation_id: Uuid::now_v7(),
            approved: !self.policy.requires_approval(&call),
            call,
        }
    }

    fn execute(&self, approved: ApprovedToolCall) -> ToolResult {
        let call = approved.call;
        self.execute_hooks("pretooluse", Some(&call), None);
        let (success, output) = match self.run_tool(&call) {
            Ok(output) => (true, output),
            Err(err) => (false, json!({"error": err.to_string()})),
        };
        if success {
            self.execute_hooks("posttooluse", Some(&call), Some(&output));
        } else {
            self.execute_hooks("posttooluse_failure", Some(&call), Some(&output));
        }
        ToolResult {
            invocation_id: approved.invocation_id,
            success,
            output,
        }
    }
}

/// Returns tool definitions for the DeepSeek API function calling interface.
/// Parameter names MUST match what `run_tool()` reads from `call.args`.
pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "fs_read".to_string(),
                description: "Read the contents of a file. Returns file content with line numbers. For images, returns base64. For PDFs, extracts text.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file to read"
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "1-based line number to start reading from. Optional."
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "1-based line number to stop reading at. Optional."
                        },
                        "max_bytes": {
                            "type": "integer",
                            "description": "Maximum bytes to read. Defaults to 1MB."
                        },
                        "pages": {
                            "type": "string",
                            "description": "Page range for PDF files (e.g. '1-5', '3', '10-20'). Only applicable to PDF files."
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "fs_write".to_string(),
                description: "Write content to a file. Creates the file and parent directories if they don't exist. Prefer fs_edit for modifying existing files.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "The full content to write to the file"
                        }
                    },
                    "required": ["path", "content"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "fs_edit".to_string(),
                description: "Edit a file by replacing an exact string match. The 'search' string must appear in the file. By default replaces all occurrences; set 'all' to false for first-only.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file to edit"
                        },
                        "search": {
                            "type": "string",
                            "description": "The exact text to find in the file"
                        },
                        "replace": {
                            "type": "string",
                            "description": "The text to replace the search string with"
                        },
                        "all": {
                            "type": "boolean",
                            "description": "Replace all occurrences (true, default) or just first (false)"
                        }
                    },
                    "required": ["path", "search", "replace"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "fs_list".to_string(),
                description: "List files and directories in the given directory.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "dir": {
                            "type": "string",
                            "description": "Directory path to list. Defaults to '.' (workspace root)."
                        }
                    },
                    "required": []
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "fs_glob".to_string(),
                description: "Find files matching a glob pattern. Returns matching file paths.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern to match (e.g. '**/*.rs', 'src/**/*.ts')"
                        },
                        "base": {
                            "type": "string",
                            "description": "Base directory to search in. Defaults to '.'."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return. Defaults to 200."
                        },
                        "respectGitignore": {
                            "type": "boolean",
                            "description": "Whether to respect .gitignore rules. Defaults to true."
                        }
                    },
                    "required": ["pattern"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "fs_grep".to_string(),
                description: "Search file contents using a regex pattern. Returns matching lines with file paths and line numbers.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for"
                        },
                        "glob": {
                            "type": "string",
                            "description": "Glob pattern to filter which files to search (e.g. '**/*.rs'). Defaults to '**/*'."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum matches to return. Defaults to 200."
                        },
                        "respectGitignore": {
                            "type": "boolean",
                            "description": "Whether to respect .gitignore rules. Defaults to true."
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "Whether the search is case-sensitive. Defaults to true."
                        }
                    },
                    "required": ["pattern"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "bash_run".to_string(),
                description: "Execute a shell command and return stdout/stderr. Use for git, build, test, and other terminal commands.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "string",
                            "description": "The shell command to execute"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds. Defaults to 120."
                        }
                    },
                    "required": ["cmd"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "multi_edit".to_string(),
                description: "Apply edits across multiple files with checkpoint support. Each file entry contains a path and edits array with search/replace pairs.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": { "type": "string", "description": "Relative file path" },
                                    "edits": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "search": { "type": "string", "description": "Text to find" },
                                                "replace": { "type": "string", "description": "Replacement text" }
                                            },
                                            "required": ["search", "replace"]
                                        }
                                    }
                                },
                                "required": ["path", "edits"]
                            },
                            "description": "Array of files with their edit operations"
                        }
                    },
                    "required": ["files"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "git_status".to_string(),
                description: "Show the working tree status (git status --short).".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "git_diff".to_string(),
                description: "Show changes in the working directory (git diff).".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "web_fetch".to_string(),
                description: "Fetch content from a URL and return it as text. HTML is stripped to plain text.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch (must start with http:// or https://)"
                        },
                        "max_bytes": {
                            "type": "integer",
                            "description": "Maximum bytes to retrieve. Defaults to 500000 (500KB)."
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Request timeout in seconds. Defaults to 30."
                        }
                    },
                    "required": ["url"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "web_search".to_string(),
                description: "Search the web and return results with titles, URLs, and snippets.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results to return. Defaults to 10."
                        }
                    },
                    "required": ["query"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "notebook_read".to_string(),
                description: "Read a Jupyter notebook (.ipynb), returning cell summaries with type and source preview.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the notebook file"
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "notebook_edit".to_string(),
                description: "Edit a cell in a Jupyter notebook. Supports replace, insert, and delete operations.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the notebook file"
                        },
                        "cell_index": {
                            "type": "integer",
                            "description": "0-based cell index to edit"
                        },
                        "new_source": {
                            "type": "string",
                            "description": "New content for the cell"
                        },
                        "cell_type": {
                            "type": "string",
                            "enum": ["code", "markdown"],
                            "description": "Cell type. Optional for replace, required for insert."
                        },
                        "operation": {
                            "type": "string",
                            "enum": ["replace", "insert", "delete"],
                            "description": "Edit operation. Defaults to 'replace'."
                        }
                    },
                    "required": ["path", "cell_index", "new_source"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "git_show".to_string(),
                description: "Show a git object (commit, tag, tree, or blob). Use to inspect specific commits, files at a revision, or diff between commits.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "spec": {
                            "type": "string",
                            "description": "Git revision spec (e.g. 'HEAD', 'abc123', 'HEAD:src/main.rs', 'main..feature')"
                        }
                    },
                    "required": ["spec"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "index_query".to_string(),
                description: "Full-text search the code index. Returns matching file paths and snippets ranked by relevance.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "q": {
                            "type": "string",
                            "description": "Search query string"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum results to return. Defaults to 10."
                        }
                    },
                    "required": ["q"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "patch_stage".to_string(),
                description: "Stage a unified diff as a patch for later application. Returns a patch_id for use with patch_apply.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "unified_diff": {
                            "type": "string",
                            "description": "The unified diff content to stage"
                        },
                        "base": {
                            "type": "string",
                            "description": "Base file content for SHA verification. Optional."
                        }
                    },
                    "required": ["unified_diff"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "patch_apply".to_string(),
                description: "Apply a previously staged patch by its ID. Returns whether the patch was applied successfully and any conflicts.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "patch_id": {
                            "type": "string",
                            "description": "UUID of the staged patch to apply"
                        }
                    },
                    "required": ["patch_id"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "diagnostics_check".to_string(),
                description: "Run language-specific diagnostics (cargo check, tsc, ruff, etc.) on the project or a specific path. Auto-detects the appropriate checker.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Optional path to check. If omitted, checks the entire project."
                        }
                    },
                    "required": []
                }),
            },
        },
        // ── Chrome browser automation tools ─────────────────────────────────
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "chrome_navigate".to_string(),
                description: "Navigate a Chrome browser to a URL. Requires Chrome to be running with remote debugging enabled (--remote-debugging-port=9222).".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to navigate to (must start with http://, https://, or about:)"
                        },
                        "port": {
                            "type": "integer",
                            "description": "Chrome debug port. Defaults to 9222."
                        }
                    },
                    "required": ["url"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "chrome_click".to_string(),
                description: "Click an element on the page by CSS selector.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "selector": {
                            "type": "string",
                            "description": "CSS selector of the element to click"
                        },
                        "port": {
                            "type": "integer",
                            "description": "Chrome debug port. Defaults to 9222."
                        }
                    },
                    "required": ["selector"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "chrome_type_text".to_string(),
                description: "Type text into an input element selected by CSS selector.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "selector": {
                            "type": "string",
                            "description": "CSS selector of the input element"
                        },
                        "text": {
                            "type": "string",
                            "description": "The text to type into the element"
                        },
                        "port": {
                            "type": "integer",
                            "description": "Chrome debug port. Defaults to 9222."
                        }
                    },
                    "required": ["selector", "text"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "chrome_screenshot".to_string(),
                description: "Capture a screenshot of the current page. Returns the image as base64.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["png", "jpeg", "webp"],
                            "description": "Image format. Defaults to 'png'."
                        },
                        "port": {
                            "type": "integer",
                            "description": "Chrome debug port. Defaults to 9222."
                        }
                    },
                    "required": []
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "chrome_read_console".to_string(),
                description: "Read the browser's console log entries.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "port": {
                            "type": "integer",
                            "description": "Chrome debug port. Defaults to 9222."
                        }
                    },
                    "required": []
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "chrome_evaluate".to_string(),
                description: "Evaluate a JavaScript expression in the browser and return the result.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "JavaScript expression to evaluate"
                        },
                        "port": {
                            "type": "integer",
                            "description": "Chrome debug port. Defaults to 9222."
                        }
                    },
                    "required": ["expression"]
                }),
            },
        },
        // ── Agent-level tools (handled by AgentEngine, not LocalToolHost) ───
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "user_question".to_string(),
                description: "Ask the user a question and wait for their response. Use this when you need clarification, a decision, or user input to proceed.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask the user"
                        },
                        "options": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Optional list of suggested answer choices"
                        }
                    },
                    "required": ["question"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "task_create".to_string(),
                description: "Create a task to track progress on the current work. Returns the task ID.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "Brief title for the task"
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description of what needs to be done"
                        },
                        "priority": {
                            "type": "integer",
                            "description": "Priority level (0=low, 1=normal, 2=high). Defaults to 1."
                        }
                    },
                    "required": ["subject"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "task_update".to_string(),
                description: "Update a task's status. Use to mark tasks as in_progress, completed, or failed.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "UUID of the task to update"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "failed"],
                            "description": "New status for the task"
                        },
                        "outcome": {
                            "type": "string",
                            "description": "Optional outcome description (typically set when completing or failing)"
                        }
                    },
                    "required": ["task_id", "status"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "spawn_task".to_string(),
                description: "Spawn an autonomous subagent to work on a subtask in parallel. The subagent runs independently with its own context and returns results when complete. Use for tasks that can be done independently (research, exploration, isolated code changes).".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Short name for the subagent (e.g. 'explore-auth', 'fix-tests')"
                        },
                        "goal": {
                            "type": "string",
                            "description": "Detailed description of what the subagent should accomplish"
                        },
                        "role": {
                            "type": "string",
                            "enum": ["explore", "plan", "task"],
                            "description": "Subagent role: 'explore' for research/information gathering, 'plan' for breaking down tasks, 'task' for executing specific work"
                        }
                    },
                    "required": ["name", "goal", "role"]
                }),
            },
        },
    ]
}

/// Filter tool definitions by allowed/disallowed lists.
///
/// - If `allowed` is `Some`, only tools whose `function.name` is in the list are kept.
/// - If `disallowed` is `Some`, tools whose `function.name` is in the list are removed.
/// - `allowed` and `disallowed` should not both be `Some` (caller must validate).
pub fn filter_tool_definitions(
    tools: Vec<ToolDefinition>,
    allowed: Option<&[String]>,
    disallowed: Option<&[String]>,
) -> Vec<ToolDefinition> {
    let tools = if let Some(allow_list) = allowed {
        tools
            .into_iter()
            .filter(|t| allow_list.iter().any(|a| a == &t.function.name))
            .collect()
    } else {
        tools
    };
    if let Some(deny_list) = disallowed {
        tools
            .into_iter()
            .filter(|t| !deny_list.iter().any(|d| d == &t.function.name))
            .collect()
    } else {
        tools
    }
}

/// Map tool definition function names (underscored) to internal tool names (dotted).
pub fn map_tool_name(function_name: &str) -> &str {
    match function_name {
        "fs_read" => "fs.read",
        "fs_write" => "fs.write",
        "fs_edit" => "fs.edit",
        "fs_list" => "fs.list",
        "fs_glob" => "fs.glob",
        "fs_grep" => "fs.grep",
        "bash_run" => "bash.run",
        "multi_edit" => "multi_edit",
        "git_status" => "git.status",
        "git_diff" => "git.diff",
        "git_show" => "git.show",
        "web_fetch" => "web.fetch",
        "web_search" => "web.search",
        "notebook_read" => "notebook.read",
        "notebook_edit" => "notebook.edit",
        "index_query" => "index.query",
        "patch_stage" => "patch.stage",
        "patch_apply" => "patch.apply",
        "diagnostics_check" => "diagnostics.check",
        "chrome_navigate" => "chrome.navigate",
        "chrome_click" => "chrome.click",
        "chrome_type_text" => "chrome.type_text",
        "chrome_screenshot" => "chrome.screenshot",
        "chrome_read_console" => "chrome.read_console",
        "chrome_evaluate" => "chrome.evaluate",
        "user_question" => "user_question",
        "task_create" => "task_create",
        "task_update" => "task_update",
        "spawn_task" => "spawn_task",
        other => other,
    }
}

/// Return a user-friendly hint for a tool error, or `None` if no specific hint applies.
pub fn tool_error_hint(tool_name: &str, error_msg: &str) -> Option<String> {
    let lower = error_msg.to_ascii_lowercase();
    match tool_name {
        "fs.edit" | "multi_edit" => {
            if lower.contains("search pattern not found") {
                Some("Hint: the old_string was not found in the file. Try reading the file first with fs.read to verify the exact content.".to_string())
            } else if lower.contains("line range out of bounds") {
                Some("Hint: the line range exceeds the file length. Read the file first to check how many lines it has.".to_string())
            } else {
                None
            }
        }
        "fs.read" => {
            if lower.contains("no such file") || lower.contains("not found") {
                Some("Hint: file does not exist. Use fs.glob to search for the correct path.".to_string())
            } else if lower.contains("permission denied") {
                Some("Hint: permission denied. The file may be outside the allowed workspace.".to_string())
            } else {
                None
            }
        }
        "fs.write" => {
            if lower.contains("permission denied") {
                Some("Hint: permission denied. Check that the directory exists and is writable.".to_string())
            } else {
                None
            }
        }
        "bash.run" => {
            if lower.contains("timed out") || lower.contains("timeout") {
                Some("Hint: command timed out. Try a shorter operation or increase the timeout.".to_string())
            } else if lower.contains("not found") || lower.contains("command not found") {
                Some("Hint: command not found. Check that the program is installed and in PATH.".to_string())
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Tools that are handled by AgentEngine directly, not by LocalToolHost.
pub const AGENT_LEVEL_TOOLS: &[&str] =
    &["user_question", "task_create", "task_update", "spawn_task"];

fn should_skip_rel_path(path: &Path) -> bool {
    path.components().any(|c| {
        c.as_os_str() == ".git" || c.as_os_str() == ".deepseek" || c.as_os_str() == "target"
    })
}

fn walk_paths(root: &Path, workspace: &Path, respect_gitignore: bool) -> Vec<PathBuf> {
    let mut builder = WalkBuilder::new(root);
    builder.hidden(false);
    builder.follow_links(false);
    builder.parents(respect_gitignore);
    builder.git_ignore(respect_gitignore);
    builder.git_global(respect_gitignore);
    builder.git_exclude(respect_gitignore);
    builder.require_git(false);

    let mut paths = Vec::new();
    for entry in builder.build() {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        let Ok(rel) = path.strip_prefix(workspace) else {
            continue;
        };
        if should_skip_rel_path(rel) {
            continue;
        }
        paths.push(path.to_path_buf());
    }
    paths
}

fn normalize_rel_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn is_isolated_sandbox_mode(mode: &str) -> bool {
    matches!(mode, "isolated" | "container" | "os-sandbox" | "os_sandbox")
}

fn render_wrapper_template(template: &str, workspace: &Path, cmd: &str) -> Result<String> {
    if !template.contains("{cmd}") {
        return Err(anyhow!(
            "sandbox wrapper template must include {{cmd}} placeholder"
        ));
    }
    let workspace_q = shell_quote(workspace.to_string_lossy().as_ref());
    let cmd_q = shell_quote(cmd);
    Ok(template
        .replace("{workspace}", &workspace_q)
        .replace("{cmd}", &cmd_q))
}

fn auto_isolated_wrapper_command(workspace: &Path, cmd: &str) -> Option<String> {
    let workspace_q = shell_quote(workspace.to_string_lossy().as_ref());
    let cmd_q = shell_quote(cmd);

    if cfg!(target_os = "linux") {
        if command_in_path("bwrap") {
            return Some(format!(
                "bwrap --die-with-parent --new-session --unshare-all --proc /proc --dev /dev --ro-bind /usr /usr --ro-bind /bin /bin --ro-bind /lib /lib --ro-bind /lib64 /lib64 --ro-bind /etc /etc --tmpfs /tmp --bind {workspace_q} {workspace_q} --chdir {workspace_q} /bin/sh -lc {cmd_q}"
            ));
        }
        if command_in_path("firejail") {
            return Some(format!(
                "firejail --quiet --net=none --private={workspace_q} sh -lc {cmd_q}"
            ));
        }
    }
    if cfg!(target_os = "macos") && command_in_path("sandbox-exec") {
        let workspace_profile = workspace
            .to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        let profile = format!(
            "(version 1) \
             (deny default) \
             (allow process*) \
             (allow file-read* (subpath \"/usr\")) \
             (allow file-read* (subpath \"/bin\")) \
             (allow file-read* (subpath \"/System\")) \
             (allow file-read* (subpath \"/Library\")) \
             (allow file-read* (subpath \"/private/tmp\")) \
             (allow file-write* (subpath \"/private/tmp\")) \
             (allow file-read* file-write* (subpath \"{workspace_profile}\"))"
        );
        return Some(format!(
            "sandbox-exec -p {} /bin/sh -lc {}",
            shell_quote(&profile),
            cmd_q
        ));
    }
    None
}

fn command_in_path(command: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    let separators = if cfg!(target_os = "windows") {
        vec![".exe", ".cmd", ".bat", ""]
    } else {
        vec![""]
    };
    std::env::split_paths(&paths).any(|dir| {
        separators
            .iter()
            .map(|suffix| dir.join(format!("{command}{suffix}")))
            .any(|candidate| candidate.exists() && candidate.is_file())
    })
}

fn shell_quote(value: &str) -> String {
    let escaped = value.replace('\'', "'\"'\"'");
    format!("'{escaped}'")
}

fn shell_tokens(cmd: &str) -> Vec<String> {
    cmd.split_whitespace()
        .map(|token| {
            token
                .trim_matches(|ch| matches!(ch, '"' | '\'' | '`' | ',' | '(' | ')' | '[' | ']'))
                .to_ascii_lowercase()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

fn command_has_mutating_intent(cmd: &str) -> bool {
    let tokens = shell_tokens(cmd);
    if tokens.is_empty() {
        return false;
    }
    let mut lowered = cmd.to_ascii_lowercase();
    lowered.retain(|ch| ch != '\n' && ch != '\r');
    if lowered.contains(">>")
        || lowered.contains(" >")
        || lowered.contains("1>")
        || lowered.contains("2>")
    {
        return true;
    }

    let command = tokens[0].as_str();
    if matches!(
        command,
        "rm" | "rmdir"
            | "del"
            | "rd"
            | "mv"
            | "cp"
            | "mkdir"
            | "touch"
            | "chmod"
            | "chown"
            | "chgrp"
            | "truncate"
            | "dd"
            | "mkfs"
            | "format"
            | "patch"
            | "tee"
    ) {
        return true;
    }

    if command == "git"
        && tokens.get(1).is_some_and(|sub| {
            matches!(
                sub.as_str(),
                "add"
                    | "commit"
                    | "push"
                    | "merge"
                    | "rebase"
                    | "reset"
                    | "checkout"
                    | "restore"
                    | "clean"
                    | "cherry-pick"
            )
        })
    {
        return true;
    }
    if command == "cargo"
        && tokens.get(1).is_some_and(|sub| {
            matches!(
                sub.as_str(),
                "add" | "remove" | "update" | "install" | "publish" | "fix"
            )
        })
    {
        return true;
    }
    if matches!(command, "npm" | "pnpm" | "yarn")
        && tokens
            .get(1)
            .is_some_and(|sub| matches!(sub.as_str(), "install" | "update" | "uninstall" | "add"))
    {
        return true;
    }
    if matches!(command, "pip" | "pip3")
        && tokens
            .get(1)
            .is_some_and(|sub| matches!(sub.as_str(), "install" | "uninstall"))
    {
        return true;
    }
    if command == "sed" && tokens.iter().any(|token| token == "-i") {
        return true;
    }
    if command == "perl" && tokens.iter().any(|token| token.starts_with("-i")) {
        return true;
    }

    false
}

fn command_has_network_egress_intent(cmd: &str) -> bool {
    let tokens = shell_tokens(cmd);
    if tokens.is_empty() {
        return false;
    }
    let command = tokens[0].as_str();
    if matches!(
        command,
        "curl"
            | "wget"
            | "scp"
            | "sftp"
            | "ssh"
            | "nc"
            | "ncat"
            | "telnet"
            | "ftp"
            | "rsync"
            | "http"
    ) {
        return true;
    }
    if command == "git"
        && tokens.get(1).is_some_and(|sub| {
            matches!(
                sub.as_str(),
                "push" | "fetch" | "pull" | "clone" | "ls-remote"
            )
        })
    {
        return true;
    }
    false
}

fn token_to_absolute_path(token: &str) -> Option<PathBuf> {
    let token = token.trim();
    if token.is_empty() || token.starts_with('-') {
        return None;
    }
    if token.starts_with("~/") {
        let home = std::env::var("HOME")
            .ok()
            .or_else(|| std::env::var("USERPROFILE").ok())?;
        return Some(PathBuf::from(home).join(token.trim_start_matches("~/")));
    }
    let candidate = PathBuf::from(token);
    if candidate.is_absolute() {
        return Some(candidate);
    }
    None
}

fn command_references_outside_workspace(cmd: &str, workspace: &Path) -> bool {
    let workspace_root =
        std::fs::canonicalize(workspace).unwrap_or_else(|_| workspace.to_path_buf());
    for raw in cmd.split_whitespace() {
        let token =
            raw.trim_matches(|ch| matches!(ch, '"' | '\'' | '`' | ',' | '(' | ')' | '[' | ']'));
        if token.is_empty() || token.starts_with('-') {
            continue;
        }
        if token == ".."
            || token.starts_with("../")
            || token.contains("/../")
            || token.ends_with("/..")
        {
            return true;
        }
        if let Some(path) = token_to_absolute_path(token)
            && !path.starts_with(&workspace_root)
        {
            return true;
        }
    }
    false
}

fn is_binary(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    if bytes.contains(&0) {
        return true;
    }
    let sample = bytes.iter().take(8192);
    let non_text = sample
        .filter(|b| !(b.is_ascii() || **b == b'\n' || **b == b'\r' || **b == b'\t'))
        .count();
    non_text > 64
}

fn guess_mime(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "md" => "text/markdown",
        "txt" | "log" | "rs" | "toml" | "json" | "yaml" | "yml" | "js" | "ts" | "tsx" | "jsx"
        | "py" | "go" | "java" | "c" | "h" | "cpp" | "hpp" | "cs" | "sh" | "ps1" => "text/plain",
        "pdf" => "application/pdf",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        "ipynb" => "application/x-ipynb+json",
        _ => "application/octet-stream",
    }
}

fn collect_lines(
    content: &str,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> Vec<serde_json::Value> {
    let start = start_line.unwrap_or(1).max(1);
    let end = end_line.unwrap_or(usize::MAX).max(start);
    content
        .lines()
        .enumerate()
        .filter_map(|(idx, text)| {
            let line = idx + 1;
            if line < start || line > end {
                return None;
            }
            Some(json!({
                "line": line,
                "text": text
            }))
        })
        .collect()
}

fn extract_pdf_text(path: &Path, pages: Option<&str>) -> Result<String> {
    let full_text =
        pdf_extract::extract_text(path).map_err(|e| anyhow!("PDF extraction failed: {e}"))?;
    match pages {
        None => Ok(full_text),
        Some(range) => {
            let (start, end) = parse_page_range(range)?;
            let page_texts: Vec<&str> = full_text.split('\x0C').collect();
            let total = page_texts.len();
            if start > total {
                return Err(anyhow!("page {start} out of range ({total} pages)"));
            }
            let end_idx = end.min(total);
            Ok(page_texts[(start - 1)..end_idx].join("\n--- page break ---\n"))
        }
    }
}

fn parse_page_range(range: &str) -> Result<(usize, usize)> {
    if let Some((s, e)) = range.split_once('-') {
        let start: usize = s
            .trim()
            .parse()
            .map_err(|_| anyhow!("invalid range start"))?;
        let end: usize = e.trim().parse().map_err(|_| anyhow!("invalid range end"))?;
        if start == 0 || end < start {
            return Err(anyhow!("invalid page range"));
        }
        Ok((start, end))
    } else {
        let p: usize = range
            .trim()
            .parse()
            .map_err(|_| anyhow!("invalid page number"))?;
        if p == 0 {
            return Err(anyhow!("page must be >= 1"));
        }
        Ok((p, p))
    }
}

fn strip_html_tags(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let lower = html.to_ascii_lowercase();
    let chars: Vec<char> = html.chars().collect();
    let lower_chars: Vec<char> = lower.chars().collect();

    let mut i = 0;
    while i < chars.len() {
        if !in_tag && chars[i] == '<' {
            in_tag = true;
            // Check for script/style open/close
            let remaining: String = lower_chars[i..].iter().take(20).collect();
            if remaining.starts_with("<script") {
                in_script = true;
            } else if remaining.starts_with("</script") {
                in_script = false;
            } else if remaining.starts_with("<style") {
                in_style = true;
            } else if remaining.starts_with("</style") {
                in_style = false;
            }
        } else if in_tag && chars[i] == '>' {
            in_tag = false;
        } else if !in_tag && !in_script && !in_style {
            out.push(chars[i]);
        }
        i += 1;
    }
    // Collapse multiple blank lines
    let mut result = String::with_capacity(out.len());
    let mut blank_count = 0;
    for line in out.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            blank_count += 1;
            if blank_count <= 1 {
                result.push('\n');
            }
        } else {
            blank_count = 0;
            result.push_str(trimmed);
            result.push('\n');
        }
    }
    result.trim().to_string()
}

/// Parse search results from DuckDuckGo HTML response.
fn parse_search_results(html: &str, max_results: usize) -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    // Simple parsing: look for result links and snippets in DuckDuckGo HTML
    // DuckDuckGo HTML uses class="result__a" for links and "result__snippet" for snippets
    let lower = html.to_ascii_lowercase();
    let mut pos = 0;
    while results.len() < max_results {
        // Find next result link
        let link_marker = "class=\"result__a\"";
        let link_pos = match lower[pos..].find(link_marker) {
            Some(p) => pos + p,
            None => break,
        };
        // Extract href
        let href_start = match html[..link_pos].rfind("href=\"") {
            Some(p) => p + 6,
            None => {
                pos = link_pos + link_marker.len();
                continue;
            }
        };
        let href_end = match html[href_start..].find('"') {
            Some(p) => href_start + p,
            None => {
                pos = link_pos + link_marker.len();
                continue;
            }
        };
        let url = html[href_start..href_end].to_string();
        // Extract title (text within the <a> tag)
        let tag_close = match html[link_pos..].find('>') {
            Some(p) => link_pos + p + 1,
            None => {
                pos = link_pos + link_marker.len();
                continue;
            }
        };
        let tag_end = match html[tag_close..].find("</a>") {
            Some(p) => tag_close + p,
            None => {
                pos = link_pos + link_marker.len();
                continue;
            }
        };
        let title = strip_html_tags(&html[tag_close..tag_end]);
        // Extract snippet
        let snippet_marker = "class=\"result__snippet\"";
        let snippet = if let Some(sp) = lower[tag_end..].find(snippet_marker) {
            let snippet_pos = tag_end + sp;
            let snippet_start = match html[snippet_pos..].find('>') {
                Some(p) => snippet_pos + p + 1,
                None => tag_end,
            };
            let snippet_end = match html[snippet_start..].find("</") {
                Some(p) => snippet_start + p,
                None => snippet_start,
            };
            strip_html_tags(&html[snippet_start..snippet_end])
        } else {
            String::new()
        };
        if !url.is_empty() && !title.is_empty() {
            results.push(json!({
                "title": title.trim(),
                "url": url,
                "snippet": snippet.trim(),
            }));
        }
        pos = tag_end;
    }
    results
}

/// URL-encode a string (percent encoding). Simple implementation for search queries.
fn url_encode_query(input: &str) -> String {
    let mut encoded = String::with_capacity(input.len() * 3);
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            b' ' => encoded.push('+'),
            _ => {
                encoded.push('%');
                encoded.push_str(&format!("{byte:02X}"));
            }
        }
    }
    encoded
}

fn generate_unified_diff(path: &str, before: &str, after: &str) -> String {
    let old_lines: Vec<&str> = before.lines().collect();
    let new_lines: Vec<&str> = after.lines().collect();
    let n = old_lines.len();
    let m = new_lines.len();

    // LCS dynamic programming table
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            if old_lines[i - 1] == new_lines[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    // Backtrack to produce edit operations: Equal / Delete / Insert
    #[derive(Clone, Copy, PartialEq)]
    enum Op {
        Equal,
        Delete,
        Insert,
    }
    let mut ops: Vec<(Op, usize)> = Vec::new(); // (op, line index in old or new)
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && old_lines[i - 1] == new_lines[j - 1] {
            ops.push((Op::Equal, i - 1));
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || dp[i][j - 1] >= dp[i - 1][j]) {
            ops.push((Op::Insert, j - 1));
            j -= 1;
        } else {
            ops.push((Op::Delete, i - 1));
            i -= 1;
        }
    }
    ops.reverse();

    // Group into hunks with 3 lines of context
    let context = 3usize;
    let mut hunks: Vec<(usize, usize)> = Vec::new(); // (start, end) indices into ops
    let mut hunk_start: Option<usize> = None;
    let mut last_change: Option<usize> = None;

    for (idx, &(op, _)) in ops.iter().enumerate() {
        if op != Op::Equal {
            if let Some(prev) = last_change {
                if idx.saturating_sub(prev) > context * 2 {
                    // Close previous hunk
                    let end = (prev + context).min(ops.len() - 1);
                    hunks.push((hunk_start.unwrap(), end));
                    hunk_start = Some(idx.saturating_sub(context));
                }
            } else {
                hunk_start = Some(idx.saturating_sub(context));
            }
            last_change = Some(idx);
        }
    }
    if let (Some(start), Some(prev)) = (hunk_start, last_change) {
        let end = (prev + context).min(ops.len() - 1);
        hunks.push((start, end));
    }

    let mut out = format!("--- a/{path}\n+++ b/{path}\n");
    for (hunk_start, hunk_end) in &hunks {
        // Count old/new line numbers at hunk boundaries
        let mut old_start = 1usize;
        let mut new_start = 1usize;
        for &(op, _) in ops.iter().take(*hunk_start) {
            match op {
                Op::Equal | Op::Delete => old_start += 1,
                _ => {}
            }
            match op {
                Op::Equal | Op::Insert => new_start += 1,
                _ => {}
            }
        }
        let mut old_count = 0usize;
        let mut new_count = 0usize;
        for &(op, _) in ops.iter().take(hunk_end + 1).skip(*hunk_start) {
            match op {
                Op::Equal | Op::Delete => old_count += 1,
                _ => {}
            }
            match op {
                Op::Equal | Op::Insert => new_count += 1,
                _ => {}
            }
        }
        out.push_str(&format!(
            "@@ -{},{} +{},{} @@\n",
            old_start, old_count, new_start, new_count
        ));
        for &(op, line_idx) in ops.iter().take(hunk_end + 1).skip(*hunk_start) {
            match op {
                Op::Equal => out.push_str(&format!(" {}\n", old_lines[line_idx])),
                Op::Delete => out.push_str(&format!("-{}\n", old_lines[line_idx])),
                Op::Insert => out.push_str(&format!("+{}\n", new_lines[line_idx])),
            }
        }
    }
    out
}

fn detect_diagnostics_command(
    workspace: &Path,
    target: Option<&str>,
) -> Result<(String, &'static str)> {
    if workspace.join("Cargo.toml").exists() {
        let cmd = match target {
            Some(_) => "cargo check --message-format=json 2>&1".to_string(),
            None => "cargo check --message-format=json 2>&1".to_string(),
        };
        return Ok((cmd, "rustc"));
    }
    if workspace.join("tsconfig.json").exists() {
        return Ok(("npx tsc --noEmit --pretty false 2>&1".to_string(), "tsc"));
    }
    if workspace.join("pyproject.toml").exists() || workspace.join("setup.py").exists() {
        let cmd = match target {
            Some(path) => format!("ruff check {path} --output-format json 2>&1"),
            None => "ruff check . --output-format json 2>&1".to_string(),
        };
        return Ok((cmd, "ruff"));
    }
    Err(anyhow!("no supported language project detected"))
}

fn parse_diagnostics(stdout: &str, stderr: &str, source: &str) -> Vec<serde_json::Value> {
    match source {
        "rustc" => parse_cargo_check_json(stdout),
        "tsc" => parse_tsc_output(stderr),
        "ruff" => parse_ruff_json(stdout),
        _ => vec![],
    }
}

fn parse_cargo_check_json(output: &str) -> Vec<serde_json::Value> {
    let mut diagnostics = Vec::new();
    for line in output.lines() {
        let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if entry.get("reason").and_then(|v| v.as_str()) != Some("compiler-message") {
            continue;
        }
        let Some(message) = entry.get("message") else {
            continue;
        };
        let level = message
            .get("level")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let text = message
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let spans = message.get("spans").and_then(|v| v.as_array());
        let (file, line_num, col) = if let Some(spans) = spans
            && let Some(span) = spans.first()
        {
            (
                span.get("file_name").and_then(|v| v.as_str()).unwrap_or(""),
                span.get("line_start").and_then(|v| v.as_u64()).unwrap_or(0),
                span.get("column_start")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
            )
        } else {
            ("", 0, 0)
        };
        diagnostics.push(json!({
            "level": level,
            "message": text,
            "file": file,
            "line": line_num,
            "column": col,
        }));
    }
    diagnostics
}

fn parse_tsc_output(output: &str) -> Vec<serde_json::Value> {
    let mut diagnostics = Vec::new();
    let re = regex::Regex::new(r"^(.+)\((\d+),(\d+)\):\s+error\s+(TS\d+):\s+(.+)$").ok();
    let Some(re) = re else {
        return diagnostics;
    };
    for line in output.lines() {
        if let Some(caps) = re.captures(line) {
            diagnostics.push(json!({
                "level": "error",
                "message": caps.get(5).map(|m| m.as_str()).unwrap_or(""),
                "file": caps.get(1).map(|m| m.as_str()).unwrap_or(""),
                "line": caps.get(2).map(|m| m.as_str()).unwrap_or("0").parse::<u64>().unwrap_or(0),
                "column": caps.get(3).map(|m| m.as_str()).unwrap_or("0").parse::<u64>().unwrap_or(0),
                "code": caps.get(4).map(|m| m.as_str()).unwrap_or(""),
            }));
        }
    }
    diagnostics
}

fn parse_ruff_json(output: &str) -> Vec<serde_json::Value> {
    let mut diagnostics = Vec::new();
    let Ok(entries) = serde_json::from_str::<Vec<serde_json::Value>>(output) else {
        return diagnostics;
    };
    for entry in entries {
        let file = entry.get("filename").and_then(|v| v.as_str()).unwrap_or("");
        let code = entry.get("code").and_then(|v| v.as_str()).unwrap_or("");
        let message = entry.get("message").and_then(|v| v.as_str()).unwrap_or("");
        let location = entry.get("location");
        let line_num = location
            .and_then(|l| l.get("row"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let col = location
            .and_then(|l| l.get("column"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        diagnostics.push(json!({
            "level": "warning",
            "message": message,
            "file": file,
            "line": line_num,
            "column": col,
            "code": code,
        }));
    }
    diagnostics
}

fn apply_single_edit(content: &mut String, edit: &serde_json::Value) -> Result<usize> {
    if let (Some(search), Some(replace)) = (
        edit.get("search").and_then(|v| v.as_str()),
        edit.get("replace").and_then(|v| v.as_str()),
    ) {
        let replace_all = edit.get("all").and_then(|v| v.as_bool()).unwrap_or(true);
        if replace_all {
            let count = content.matches(search).count();
            if count == 0 {
                return Err(anyhow!("search pattern not found: {search}"));
            }
            *content = content.replace(search, replace);
            return Ok(count);
        }
        if let Some(pos) = content.find(search) {
            content.replace_range(pos..pos + search.len(), replace);
            return Ok(1);
        }
        return Err(anyhow!("search pattern not found: {search}"));
    }

    if let (Some(start_line), Some(end_line), Some(replacement)) = (
        edit.get("start_line").and_then(|v| v.as_u64()),
        edit.get("end_line").and_then(|v| v.as_u64()),
        edit.get("replacement").and_then(|v| v.as_str()),
    ) {
        let start = start_line as usize;
        let end = end_line as usize;
        if start == 0 || end < start {
            return Err(anyhow!(
                "invalid line range: start_line={start_line} end_line={end_line}"
            ));
        }

        let had_trailing_newline = content.ends_with('\n');
        let mut lines = content.lines().map(ToString::to_string).collect::<Vec<_>>();
        if end > lines.len() {
            return Err(anyhow!(
                "line range out of bounds: end_line={end_line} file_lines={}",
                lines.len()
            ));
        }
        let replacement_lines = replacement
            .split('\n')
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        lines.splice((start - 1)..end, replacement_lines);
        *content = lines.join("\n");
        if had_trailing_newline {
            content.push('\n');
        }
        return Ok(1);
    }

    Err(anyhow!(
        "edit requires either search+replace or start_line+end_line+replacement"
    ))
}

/// Generate a macOS Seatbelt sandbox profile.
#[allow(dead_code)]
fn build_seatbelt_profile(workspace: &Path, config: &deepseek_core::SandboxConfig) -> String {
    let workspace_str = workspace.to_string_lossy();
    let mut profile = String::from("(version 1)\n(deny default)\n");
    profile.push_str("(allow process*)\n");
    profile.push_str("(allow file-read* (subpath \"/usr\") (subpath \"/lib\") (subpath \"/bin\") (subpath \"/System\"))\n");
    profile.push_str(&format!(
        "(allow file-read* file-write* (subpath \"{}\"))\n",
        workspace_str
    ));
    // Allow /tmp access
    profile
        .push_str("(allow file-read* file-write* (subpath \"/tmp\") (subpath \"/private/tmp\"))\n");
    // Network access
    if config.network.block_all {
        profile.push_str("(deny network*)\n");
    } else if !config.network.allowed_domains.is_empty() {
        // Allow network but note: Seatbelt doesn't support per-domain filtering natively.
        // Allow all network when specific domains are requested (filtering done at app level).
        profile.push_str("(allow network*)\n");
    } else {
        profile.push_str("(allow network*)\n");
    }
    profile
}

/// Wrap a command with macOS Seatbelt sandbox.
#[allow(dead_code)]
fn seatbelt_wrap(cmd: &str, profile: &str) -> String {
    // Escape single quotes in profile
    let escaped_profile = profile.replace('\'', "'\\''");
    format!("sandbox-exec -p '{}' -- {}", escaped_profile, cmd)
}

/// Build a Linux bubblewrap (bwrap) sandboxed command.
#[allow(dead_code)]
fn build_bwrap_command(
    workspace: &Path,
    cmd: &str,
    config: &deepseek_core::SandboxConfig,
) -> String {
    let workspace_str = workspace.to_string_lossy();
    let mut parts = vec![
        "bwrap".to_string(),
        "--die-with-parent".to_string(),
        "--new-session".to_string(),
    ];
    // Read-only system paths
    for sys_path in &["/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc"] {
        parts.push("--ro-bind".to_string());
        parts.push(sys_path.to_string());
        parts.push(sys_path.to_string());
    }
    // Read-write workspace
    parts.push("--bind".to_string());
    parts.push(workspace_str.to_string());
    parts.push(workspace_str.to_string());
    // Tmp
    parts.push("--tmpfs".to_string());
    parts.push("/tmp".to_string());
    // Network
    if config.network.block_all {
        parts.push("--unshare-net".to_string());
    }
    // Proc and dev
    parts.push("--proc".to_string());
    parts.push("/proc".to_string());
    parts.push("--dev".to_string());
    parts.push("/dev".to_string());
    parts.push("--".to_string());
    parts.push(cmd.to_string());
    parts.join(" ")
}

/// Wrap a command with the appropriate OS-level sandbox if enabled.
#[allow(clippy::needless_return)]
fn sandbox_wrap_command(
    workspace: &Path,
    cmd: &str,
    config: &deepseek_core::SandboxConfig,
) -> String {
    if !config.enabled {
        return cmd.to_string();
    }
    // Check if command is excluded
    for excluded in &config.excluded_commands {
        if cmd.starts_with(excluded) || cmd.contains(excluded) {
            return cmd.to_string();
        }
    }
    #[cfg(target_os = "macos")]
    {
        let profile = build_seatbelt_profile(workspace, config);
        return seatbelt_wrap(cmd, &profile);
    }
    #[cfg(target_os = "linux")]
    {
        return build_bwrap_command(workspace, cmd, config);
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = workspace;
        cmd.to_string()
    }
}

impl LocalToolHost {
    /// Fire "stop" hooks when the agent run completes.
    pub fn fire_stop_hooks(&self) {
        self.execute_hooks("stop", None, None);
    }

    /// Fire hooks for a specific lifecycle phase (sessionstart, notification, etc.).
    pub fn fire_session_hooks(&self, phase: &str) {
        self.execute_hooks(phase, None, None);
    }

    fn execute_hooks(
        &self,
        phase: &str,
        call: Option<&ToolCall>,
        result: Option<&serde_json::Value>,
    ) {
        if !self.hooks_enabled {
            return;
        }
        let Some(manager) = &self.plugins else {
            return;
        };
        let Ok(hooks) = manager.hook_paths_for(phase) else {
            return;
        };
        let context = HookContext {
            phase: phase.to_string(),
            workspace: self.workspace.clone(),
            tool_name: call.map(|call| call.name.clone()),
            tool_args_json: call.map(|call| call.args.to_string()),
            tool_result_json: result.map(ToString::to_string),
        };
        match HookRuntime::run(&hooks, &context, Duration::from_secs(30)) {
            Ok(runs) => {
                for run in runs {
                    self.emit_hook_event(
                        phase,
                        &run.path,
                        run.success,
                        run.timed_out,
                        run.exit_code,
                    );
                }
            }
            Err(_) => {
                for hook in hooks {
                    self.emit_hook_event(phase, &hook, false, false, None);
                }
            }
        }
    }

    fn emit_hook_event(
        &self,
        phase: &str,
        hook_path: &Path,
        success: bool,
        timed_out: bool,
        exit_code: Option<i32>,
    ) {
        let Ok(Some(session)) = self.store.load_latest_session() else {
            return;
        };
        let Ok(seq_no) = self.store.next_seq_no(session.session_id) else {
            return;
        };
        let event = EventEnvelope {
            seq_no,
            at: Utc::now(),
            session_id: session.session_id,
            kind: EventKind::HookExecutedV1 {
                phase: phase.to_string(),
                hook_path: hook_path.to_string_lossy().to_string(),
                success,
                timed_out,
                exit_code,
            },
        };
        let _ = self.store.append_event(&event);
    }

    fn emit_visual_artifact_event(&self, path: &str, mime: &str) {
        let Ok(Some(session)) = self.store.load_latest_session() else {
            return;
        };
        let Ok(seq_no) = self.store.next_seq_no(session.session_id) else {
            return;
        };
        let event = EventEnvelope {
            seq_no,
            at: Utc::now(),
            session_id: session.session_id,
            kind: EventKind::VisualArtifactCapturedV1 {
                artifact_id: Uuid::now_v7(),
                path: path.to_string(),
                mime: mime.to_string(),
            },
        };
        let _ = self.store.append_event(&event);
    }

    fn create_checkpoint(&self, reason: &str) -> Result<Option<Uuid>> {
        let manager = MemoryManager::new(&self.workspace)?;
        let checkpoint = manager.create_checkpoint(reason)?;
        let Ok(Some(session)) = self.store.load_latest_session() else {
            return Ok(Some(checkpoint.checkpoint_id));
        };
        let event = EventEnvelope {
            seq_no: self.store.next_seq_no(session.session_id)?,
            at: Utc::now(),
            session_id: session.session_id,
            kind: EventKind::CheckpointCreatedV1 {
                checkpoint_id: checkpoint.checkpoint_id,
                reason: checkpoint.reason,
                files_count: checkpoint.files_count,
                snapshot_path: checkpoint.snapshot_path,
            },
        };
        self.store.append_event(&event)?;
        self.fire_session_hooks("checkpointcreated");
        Ok(Some(checkpoint.checkpoint_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepseek_core::{
        AppConfig, ApprovedToolCall, Session, SessionBudgets, SessionState, ToolCall, ToolHost,
        runtime_dir,
    };
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    fn temp_host() -> (PathBuf, LocalToolHost) {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-tools-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let host = LocalToolHost::new(&workspace, PolicyEngine::default()).expect("tool host");
        (workspace, host)
    }

    #[derive(Clone, Default)]
    struct RecordingRunner {
        commands: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingRunner {
        fn captured(&self) -> Vec<String> {
            self.commands.lock().expect("commands").clone()
        }
    }

    impl ShellRunner for RecordingRunner {
        fn run(&self, cmd: &str, _cwd: &Path, _timeout: Duration) -> Result<ShellRunResult> {
            self.commands
                .lock()
                .expect("commands")
                .push(cmd.to_string());
            Ok(ShellRunResult {
                status: Some(0),
                stdout: "ok".to_string(),
                stderr: String::new(),
                timed_out: false,
            })
        }
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn hooks_receive_phase_and_tool_context() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-tools-hook-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");

        let mut cfg = AppConfig::default();
        cfg.plugins.enable_hooks = true;
        cfg.save(&workspace).expect("save config");

        let plugin_src = workspace.join("plugin-src");
        fs::create_dir_all(plugin_src.join(".deepseek-plugin")).expect("plugin dir");
        fs::create_dir_all(plugin_src.join("hooks")).expect("hooks dir");
        fs::write(
            plugin_src.join(".deepseek-plugin/plugin.json"),
            r#"{"id":"hookdemo","name":"Hook Demo","version":"0.1.0"}"#,
        )
        .expect("manifest");
        fs::write(
            plugin_src.join("hooks/pretooluse.sh"),
            "#!/bin/sh\nprintf \"%s|%s\" \"$DEEPSEEK_HOOK_PHASE\" \"$DEEPSEEK_TOOL_NAME\" > \"$DEEPSEEK_WORKSPACE/hook.out\"\n",
        )
        .expect("hook script");

        let manager = PluginManager::new(&workspace).expect("plugin manager");
        manager.install(&plugin_src).expect("install plugin");

        let host = LocalToolHost::new(&workspace, PolicyEngine::default()).expect("tool host");
        let result = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "fs.list".to_string(),
                args: json!({"dir":"."}),
                requires_approval: false,
            },
        });
        assert!(result.success);

        let hook_out = fs::read_to_string(workspace.join("hook.out")).expect("hook output");
        assert!(hook_out.contains("pretooluse|fs.list"));
    }

    #[test]
    fn fs_read_supports_line_ranges_and_mime_metadata() {
        let (workspace, host) = temp_host();
        fs::write(workspace.join("note.txt"), "a\nb\nc\n").expect("seed");

        let result = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "fs.read".to_string(),
                args: json!({"path":"note.txt","start_line":2,"end_line":3}),
                requires_approval: false,
            },
        });
        assert!(result.success);
        assert_eq!(result.output["mime"], "text/plain");
        assert_eq!(result.output["binary"], false);
        let lines = result.output["lines"].as_array().expect("lines");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["line"], 2);
        assert_eq!(lines[0]["text"], "b");
    }

    #[test]
    fn fs_glob_grep_and_edit_work() {
        let (workspace, host) = temp_host();
        fs::create_dir_all(workspace.join("src")).expect("src");
        fs::write(workspace.join("src/main.rs"), "fn old_name() {}\n").expect("seed");
        fs::write(workspace.join("src/lib.rs"), "pub fn helper() {}\n").expect("seed");

        let globbed = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "fs.glob".to_string(),
                args: json!({"pattern":"src/*.rs"}),
                requires_approval: false,
            },
        });
        assert!(globbed.success);
        assert!(
            globbed.output["matches"]
                .as_array()
                .is_some_and(|items| items.len() >= 2)
        );

        let grepped = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "fs.grep".to_string(),
                args: json!({"pattern":"old_name","glob":"src/*.rs"}),
                requires_approval: false,
            },
        });
        assert!(grepped.success);
        assert_eq!(
            grepped.output["matches"].as_array().expect("matches").len(),
            1
        );

        let edited = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "fs.edit".to_string(),
                args: json!({"path":"src/main.rs","search":"old_name","replace":"new_name","all":false}),
                requires_approval: false,
            },
        });
        assert!(edited.success);
        assert_eq!(edited.output["edited"], true);
        let content = fs::read_to_string(workspace.join("src/main.rs")).expect("updated");
        assert!(content.contains("new_name"));
    }

    #[test]
    fn fs_edit_includes_unified_diff_in_result() {
        let (workspace, host) = temp_host();
        fs::write(workspace.join("demo.rs"), "fn old() {}\n").expect("seed");

        let result = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "fs.edit".to_string(),
                args: json!({"path":"demo.rs","search":"old","replace":"new","all":false}),
                requires_approval: false,
            },
        });
        assert!(result.success);
        assert_eq!(result.output["edited"], true);
        let diff = result.output["diff"].as_str().expect("diff field");
        assert!(diff.contains("--- a/demo.rs"));
        assert!(diff.contains("+++ b/demo.rs"));
        assert!(diff.contains("-fn old() {}"));
        assert!(diff.contains("+fn new() {}"));
    }

    #[test]
    fn fs_glob_respects_gitignore_rules() {
        let (workspace, host) = temp_host();
        fs::create_dir_all(workspace.join("ignored")).expect("ignored dir");
        fs::create_dir_all(workspace.join("src")).expect("src");
        fs::write(workspace.join(".gitignore"), "ignored/\n").expect("gitignore");
        fs::write(workspace.join("ignored/secret.txt"), "secret\n").expect("secret");
        fs::write(workspace.join("src/main.rs"), "fn main() {}\n").expect("main");

        let result = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "fs.glob".to_string(),
                args: json!({"pattern":"**/*","respectGitignore":true}),
                requires_approval: false,
            },
        });
        assert!(result.success);
        let paths = result
            .output
            .get("matches")
            .and_then(|items| items.as_array())
            .expect("matches")
            .iter()
            .filter_map(|item| item.get("path").and_then(|value| value.as_str()))
            .collect::<Vec<_>>();
        assert!(paths.iter().any(|path| path.ends_with("src/main.rs")));
        assert!(
            !paths
                .iter()
                .any(|path| path.ends_with("ignored/secret.txt"))
        );
    }

    #[test]
    fn fs_read_emits_visual_artifact_event_when_enabled() {
        let workspace = std::env::temp_dir().join(format!("deepseek-tools-vis-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");

        let mut cfg = AppConfig::default();
        cfg.experiments.visual_verification = true;
        cfg.save(&workspace).expect("save config");

        let store = Store::new(&workspace).expect("store");
        store
            .save_session(&Session {
                session_id: Uuid::now_v7(),
                workspace_root: workspace.to_string_lossy().to_string(),
                baseline_commit: None,
                status: SessionState::Idle,
                budgets: SessionBudgets {
                    per_turn_seconds: 30,
                    max_think_tokens: 1024,
                },
                active_plan_id: None,
            })
            .expect("session");

        fs::write(workspace.join("image.png"), [0x89, b'P', b'N', b'G', 0, 1]).expect("image");
        let host = LocalToolHost::new(&workspace, PolicyEngine::default()).expect("host");
        let result = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "fs.read".to_string(),
                args: json!({"path":"image.png"}),
                requires_approval: false,
            },
        });
        assert!(result.success);
        assert_eq!(result.output["binary"], true);

        let events_path = runtime_dir(&workspace).join("events.jsonl");
        let events = fs::read_to_string(events_path).expect("events");
        assert!(events.contains("VisualArtifactCapturedV1"));
    }

    #[test]
    fn read_only_sandbox_blocks_mutating_bash_commands() {
        let workspace = std::env::temp_dir().join(format!("deepseek-tools-ro-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let mut cfg = AppConfig::default();
        cfg.policy.allowlist = vec!["touch *".to_string()];
        cfg.policy.sandbox_mode = "read-only".to_string();
        cfg.save(&workspace).expect("save config");
        let policy = PolicyEngine::from_app_config(&cfg.policy);
        let runner = RecordingRunner::default();
        let host = LocalToolHost::with_runner(&workspace, policy, Arc::new(runner.clone()))
            .expect("tool host");
        let result = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "bash.run".to_string(),
                args: json!({"cmd":"touch note.txt"}),
                requires_approval: false,
            },
        });
        assert!(!result.success);
        assert!(
            result.output["error"]
                .as_str()
                .unwrap_or_default()
                .contains("sandbox_mode=read-only")
        );
        assert!(runner.captured().is_empty());
    }

    #[test]
    fn workspace_write_sandbox_blocks_absolute_outside_paths() {
        let workspace = std::env::temp_dir().join(format!("deepseek-tools-ww-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let outside = std::env::temp_dir().join(format!("deepseek-outside-{}.txt", Uuid::now_v7()));
        fs::write(&outside, "outside").expect("outside file");

        let mut cfg = AppConfig::default();
        cfg.policy.allowlist = vec!["cat *".to_string()];
        cfg.policy.sandbox_mode = "workspace-write".to_string();
        cfg.save(&workspace).expect("save config");
        let policy = PolicyEngine::from_app_config(&cfg.policy);
        let runner = RecordingRunner::default();
        let host = LocalToolHost::with_runner(&workspace, policy, Arc::new(runner.clone()))
            .expect("tool host");
        let result = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "bash.run".to_string(),
                args: json!({"cmd": format!("cat {}", outside.display())}),
                requires_approval: false,
            },
        });
        assert!(!result.success);
        assert!(
            result.output["error"]
                .as_str()
                .unwrap_or_default()
                .contains("sandbox_mode=workspace-write")
        );
        assert!(runner.captured().is_empty());
    }

    #[test]
    fn workspace_write_sandbox_allows_workspace_relative_paths() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-tools-ww-allow-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        fs::write(workspace.join("note.txt"), "hello").expect("note");

        let mut cfg = AppConfig::default();
        cfg.policy.allowlist = vec!["cat *".to_string()];
        cfg.policy.sandbox_mode = "workspace-write".to_string();
        cfg.save(&workspace).expect("save config");
        let policy = PolicyEngine::from_app_config(&cfg.policy);
        let runner = RecordingRunner::default();
        let host = LocalToolHost::with_runner(&workspace, policy, Arc::new(runner.clone()))
            .expect("tool host");
        let result = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "bash.run".to_string(),
                args: json!({"cmd":"cat note.txt"}),
                requires_approval: false,
            },
        });
        assert!(result.success);
        assert_eq!(runner.captured(), vec!["cat note.txt".to_string()]);
    }

    #[test]
    fn read_only_sandbox_blocks_network_commands() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-tools-ro-net-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let mut cfg = AppConfig::default();
        cfg.policy.allowlist = vec!["curl *".to_string()];
        cfg.policy.sandbox_mode = "read-only".to_string();
        cfg.save(&workspace).expect("save config");
        let policy = PolicyEngine::from_app_config(&cfg.policy);
        let runner = RecordingRunner::default();
        let host = LocalToolHost::with_runner(&workspace, policy, Arc::new(runner.clone()))
            .expect("tool host");
        let result = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "bash.run".to_string(),
                args: json!({"cmd":"curl https://example.com"}),
                requires_approval: false,
            },
        });
        assert!(!result.success);
        assert!(
            result.output["error"]
                .as_str()
                .unwrap_or_default()
                .contains("blocked network command")
        );
        assert!(runner.captured().is_empty());
    }

    #[test]
    fn isolated_sandbox_uses_configured_wrapper_template() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-tools-iso-wrap-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        fs::write(workspace.join("note.txt"), "hello").expect("note");

        let mut cfg = AppConfig::default();
        cfg.policy.allowlist = vec!["cat *".to_string()];
        cfg.policy.sandbox_mode = "isolated".to_string();
        cfg.policy.sandbox_wrapper =
            Some("sandboxctl --workspace {workspace} --cmd {cmd}".to_string());
        cfg.save(&workspace).expect("save config");
        let policy = PolicyEngine::from_app_config(&cfg.policy);
        let runner = RecordingRunner::default();
        let host = LocalToolHost::with_runner(&workspace, policy, Arc::new(runner.clone()))
            .expect("tool host");
        let result = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "bash.run".to_string(),
                args: json!({"cmd":"cat note.txt"}),
                requires_approval: false,
            },
        });
        assert!(result.success);
        let normalized_workspace =
            std::fs::canonicalize(&workspace).unwrap_or_else(|_| workspace.clone());
        let expected = render_wrapper_template(
            "sandboxctl --workspace {workspace} --cmd {cmd}",
            &normalized_workspace,
            "cat note.txt",
        )
        .expect("render");
        assert_eq!(runner.captured(), vec![expected]);
    }

    #[test]
    fn isolated_sandbox_requires_cmd_placeholder_in_wrapper_template() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-tools-iso-bad-wrap-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");

        let mut cfg = AppConfig::default();
        cfg.policy.allowlist = vec!["cat *".to_string()];
        cfg.policy.sandbox_mode = "isolated".to_string();
        cfg.policy.sandbox_wrapper = Some("sandboxctl --workspace {workspace}".to_string());
        cfg.save(&workspace).expect("save config");
        let policy = PolicyEngine::from_app_config(&cfg.policy);
        let runner = RecordingRunner::default();
        let host = LocalToolHost::with_runner(&workspace, policy, Arc::new(runner.clone()))
            .expect("tool host");
        let result = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "bash.run".to_string(),
                args: json!({"cmd":"cat note.txt"}),
                requires_approval: false,
            },
        });
        assert!(!result.success);
        assert!(
            result.output["error"]
                .as_str()
                .unwrap_or_default()
                .contains("must include {cmd}")
        );
        assert!(runner.captured().is_empty());
    }

    #[test]
    fn multi_edit_modifies_multiple_files() {
        let (workspace, host) = temp_host();
        fs::write(workspace.join("a.txt"), "hello world\n").expect("seed a");
        fs::write(workspace.join("b.txt"), "foo bar\n").expect("seed b");

        let result = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "multi_edit".to_string(),
                args: json!({
                    "files": [
                        {"path": "a.txt", "edits": [{"search": "hello", "replace": "hi"}]},
                        {"path": "b.txt", "edits": [{"search": "foo", "replace": "baz"}]}
                    ]
                }),
                requires_approval: false,
            },
        });
        assert!(result.success);
        assert_eq!(result.output["total_files"], 2);
        assert!(result.output["total_replacements"].as_u64().unwrap() >= 2);
        assert_eq!(
            fs::read_to_string(workspace.join("a.txt")).expect("a"),
            "hi world\n"
        );
        assert_eq!(
            fs::read_to_string(workspace.join("b.txt")).expect("b"),
            "baz bar\n"
        );
    }

    #[test]
    fn multi_edit_returns_diffs_and_shas() {
        let (workspace, host) = temp_host();
        fs::write(workspace.join("c.txt"), "old value\n").expect("seed c");

        let result = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "multi_edit".to_string(),
                args: json!({
                    "files": [
                        {"path": "c.txt", "edits": [{"search": "old", "replace": "new"}]}
                    ]
                }),
                requires_approval: false,
            },
        });
        assert!(result.success);
        let results = result.output["results"].as_array().expect("results");
        assert_eq!(results.len(), 1);
        let entry = &results[0];
        assert_eq!(entry["edited"], true);
        assert!(entry["diff"].as_str().unwrap().contains("--- a/c.txt"));
        assert!(entry["before_sha256"].as_str().is_some());
        assert!(entry["after_sha256"].as_str().is_some());
    }

    #[test]
    fn multi_edit_blocked_in_review_mode() {
        let (workspace, mut host) = temp_host();
        host.set_review_mode(true);
        fs::write(workspace.join("d.txt"), "data\n").expect("seed d");

        let result = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "multi_edit".to_string(),
                args: json!({
                    "files": [
                        {"path": "d.txt", "edits": [{"search": "data", "replace": "new"}]}
                    ]
                }),
                requires_approval: false,
            },
        });
        assert!(!result.success);
        assert!(
            result.output["error"]
                .as_str()
                .unwrap_or_default()
                .contains("review mode")
        );
    }

    #[test]
    fn multi_edit_skips_unmodified_files() {
        let (workspace, host) = temp_host();
        fs::write(workspace.join("e.txt"), "keep me\n").expect("seed e");
        fs::write(workspace.join("f.txt"), "change me\n").expect("seed f");

        let result = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "multi_edit".to_string(),
                args: json!({
                    "files": [
                        {"path": "e.txt", "edits": [{"search": "missing", "replace": "gone"}]},
                        {"path": "f.txt", "edits": [{"search": "change", "replace": "changed"}]}
                    ]
                }),
                requires_approval: false,
            },
        });
        assert!(result.success);
        let results = result.output["results"].as_array().expect("results");
        // e.txt had an error (search not found), f.txt was edited
        let f_entry = results
            .iter()
            .find(|r| r["path"] == "f.txt")
            .expect("f.txt");
        assert_eq!(f_entry["edited"], true);
        assert_eq!(
            fs::read_to_string(workspace.join("e.txt")).expect("e"),
            "keep me\n"
        );
    }

    #[test]
    fn diagnostics_check_detects_rust_project() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-tools-diag-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        fs::write(workspace.join("Cargo.toml"), "[package]\nname = \"test\"\n").expect("cargo");

        let (cmd, source) = detect_diagnostics_command(&workspace, None).expect("detect");
        assert_eq!(source, "rustc");
        assert!(cmd.contains("cargo check"));
    }

    #[test]
    fn diagnostics_check_is_read_only() {
        let policy = PolicyEngine::new(deepseek_policy::PolicyConfig {
            permission_mode: deepseek_policy::PermissionMode::Plan,
            ..deepseek_policy::PolicyConfig::default()
        });
        let call = ToolCall {
            name: "diagnostics.check".to_string(),
            args: json!({}),
            requires_approval: false,
        };
        assert!(!policy.requires_approval(&call));
    }

    #[test]
    fn parse_cargo_check_json_extracts_errors() {
        let output = r#"{"reason":"compiler-message","message":{"level":"error","message":"unused variable","spans":[{"file_name":"src/main.rs","line_start":10,"column_start":5}]}}"#;
        let diagnostics = parse_cargo_check_json(output);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0]["level"], "error");
        assert_eq!(diagnostics[0]["file"], "src/main.rs");
        assert_eq!(diagnostics[0]["line"], 10);
    }

    #[test]
    fn parse_tsc_output_extracts_errors() {
        let output = "src/app.ts(42,13): error TS2304: Cannot find name 'foo'.";
        let diagnostics = parse_tsc_output(output);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0]["level"], "error");
        assert_eq!(diagnostics[0]["file"], "src/app.ts");
        assert_eq!(diagnostics[0]["line"], 42);
        assert_eq!(diagnostics[0]["column"], 13);
        assert_eq!(diagnostics[0]["code"], "TS2304");
    }

    #[test]
    fn seatbelt_profile_allows_workspace() {
        let workspace = std::path::Path::new("/tmp/test-project");
        let config = deepseek_core::SandboxConfig {
            enabled: true,
            network: deepseek_core::SandboxNetworkConfig {
                allowed_domains: vec![],
                block_all: false,
            },
            excluded_commands: vec![],
        };
        let profile = super::build_seatbelt_profile(workspace, &config);
        assert!(profile.contains("/tmp/test-project"));
        assert!(profile.contains("(deny default)"));
        assert!(profile.contains("(allow process*)"));
    }

    #[test]
    fn seatbelt_profile_blocks_network() {
        let workspace = std::path::Path::new("/tmp/test");
        let config = deepseek_core::SandboxConfig {
            enabled: true,
            network: deepseek_core::SandboxNetworkConfig {
                allowed_domains: vec![],
                block_all: true,
            },
            excluded_commands: vec![],
        };
        let profile = super::build_seatbelt_profile(workspace, &config);
        assert!(profile.contains("(deny network*)"));
    }

    #[test]
    fn bwrap_command_includes_workspace() {
        let workspace = std::path::Path::new("/home/user/project");
        let config = deepseek_core::SandboxConfig {
            enabled: true,
            network: deepseek_core::SandboxNetworkConfig {
                allowed_domains: vec![],
                block_all: false,
            },
            excluded_commands: vec![],
        };
        let result = super::build_bwrap_command(workspace, "ls -la", &config);
        assert!(result.contains("bwrap"));
        assert!(result.contains("/home/user/project"));
        assert!(result.contains("ls -la"));
    }

    #[test]
    fn bwrap_unshares_network_when_blocked() {
        let workspace = std::path::Path::new("/tmp/test");
        let config = deepseek_core::SandboxConfig {
            enabled: true,
            network: deepseek_core::SandboxNetworkConfig {
                allowed_domains: vec![],
                block_all: true,
            },
            excluded_commands: vec![],
        };
        let result = super::build_bwrap_command(workspace, "echo hi", &config);
        assert!(result.contains("--unshare-net"));
    }

    #[test]
    fn sandbox_disabled_passes_through() {
        let workspace = std::path::Path::new("/tmp/test");
        let config = deepseek_core::SandboxConfig {
            enabled: false,
            ..Default::default()
        };
        let result = super::sandbox_wrap_command(workspace, "echo hello", &config);
        assert_eq!(result, "echo hello");
    }

    #[test]
    fn sandbox_excluded_command_passes_through() {
        let workspace = std::path::Path::new("/tmp/test");
        let config = deepseek_core::SandboxConfig {
            enabled: true,
            network: deepseek_core::SandboxNetworkConfig::default(),
            excluded_commands: vec!["cargo".to_string()],
        };
        let result = super::sandbox_wrap_command(workspace, "cargo test", &config);
        assert_eq!(result, "cargo test");
    }

    #[test]
    fn seatbelt_wrap_formats_correctly() {
        let profile = "(version 1)\n(deny default)";
        let result = super::seatbelt_wrap("echo test", profile);
        assert!(result.starts_with("sandbox-exec"));
        assert!(result.contains("echo test"));
    }

    #[test]
    fn bwrap_includes_proc_and_dev() {
        let workspace = std::path::Path::new("/tmp/test");
        let config = deepseek_core::SandboxConfig::default();
        let result = super::build_bwrap_command(workspace, "ls", &config);
        assert!(result.contains("--proc"));
        assert!(result.contains("--dev"));
    }

    #[test]
    fn chrome_tool_calls_return_structured_payloads() {
        let (_workspace, host) = temp_host();
        let navigate = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "chrome.navigate".to_string(),
                args: json!({"url":"https://example.com"}),
                requires_approval: false,
            },
        });
        assert!(navigate.success);
        assert_eq!(navigate.output["ok"], true);
        assert_eq!(navigate.output["url"], "https://example.com");

        let evaluate = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "chrome.evaluate".to_string(),
                args: json!({"expression":"1 + 1"}),
                requires_approval: false,
            },
        });
        assert!(evaluate.success);
        assert!(evaluate.output["value"].is_object() || evaluate.output["value"].is_number());

        let screenshot = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "chrome.screenshot".to_string(),
                args: json!({"format":"png"}),
                requires_approval: false,
            },
        });
        assert!(screenshot.success);
        assert!(
            screenshot.output["base64"]
                .as_str()
                .is_some_and(|data| !data.is_empty())
        );
    }

    #[test]
    fn tool_definitions_include_all_tools() {
        let defs = tool_definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.function.name.as_str()).collect();

        // Core tools
        for expected in [
            "fs_read",
            "fs_write",
            "fs_edit",
            "fs_list",
            "fs_glob",
            "fs_grep",
            "bash_run",
            "multi_edit",
            "git_status",
            "git_diff",
            "git_show",
            "web_fetch",
            "web_search",
            "notebook_read",
            "notebook_edit",
            "index_query",
            "patch_stage",
            "patch_apply",
            "diagnostics_check",
        ] {
            assert!(
                names.contains(&expected),
                "missing tool definition: {expected}"
            );
        }
        // Chrome tools
        for expected in [
            "chrome_navigate",
            "chrome_click",
            "chrome_type_text",
            "chrome_screenshot",
            "chrome_read_console",
            "chrome_evaluate",
        ] {
            assert!(
                names.contains(&expected),
                "missing chrome tool definition: {expected}"
            );
        }
        // Agent-level tools
        for expected in ["user_question", "task_create", "task_update", "spawn_task"] {
            assert!(
                names.contains(&expected),
                "missing agent tool definition: {expected}"
            );
        }
    }

    #[test]
    fn map_tool_name_covers_all_definitions() {
        let defs = tool_definitions();
        for def in &defs {
            let fn_name = &def.function.name;
            let mapped = map_tool_name(fn_name);
            // Every underscored name should map to something
            assert!(
                !mapped.is_empty(),
                "map_tool_name returned empty for {fn_name}"
            );
        }
        // Verify chrome mappings specifically
        assert_eq!(map_tool_name("chrome_navigate"), "chrome.navigate");
        assert_eq!(map_tool_name("chrome_click"), "chrome.click");
        assert_eq!(map_tool_name("chrome_type_text"), "chrome.type_text");
        assert_eq!(map_tool_name("chrome_screenshot"), "chrome.screenshot");
        assert_eq!(map_tool_name("chrome_read_console"), "chrome.read_console");
        assert_eq!(map_tool_name("chrome_evaluate"), "chrome.evaluate");
        // Agent-level tool mappings
        assert_eq!(map_tool_name("user_question"), "user_question");
        assert_eq!(map_tool_name("task_create"), "task_create");
        assert_eq!(map_tool_name("task_update"), "task_update");
        assert_eq!(map_tool_name("spawn_task"), "spawn_task");
    }

    #[test]
    fn agent_level_tools_constant_matches_definitions() {
        let defs = tool_definitions();
        let def_names: Vec<&str> = defs.iter().map(|d| d.function.name.as_str()).collect();
        for tool_name in AGENT_LEVEL_TOOLS {
            assert!(
                def_names.contains(tool_name),
                "AGENT_LEVEL_TOOLS contains '{tool_name}' but no definition exists"
            );
        }
    }

    #[test]
    fn filter_tool_definitions_allowed() {
        let defs = tool_definitions();
        let allowed = vec!["fs_read".to_string(), "fs_write".to_string()];
        let filtered = filter_tool_definitions(defs, Some(&allowed), None);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().any(|t| t.function.name == "fs_read"));
        assert!(filtered.iter().any(|t| t.function.name == "fs_write"));
    }

    #[test]
    fn filter_tool_definitions_disallowed() {
        let defs = tool_definitions();
        let original_count = defs.len();
        let disallowed = vec!["bash_run".to_string()];
        let filtered = filter_tool_definitions(defs, None, Some(&disallowed));
        assert_eq!(filtered.len(), original_count - 1);
        assert!(!filtered.iter().any(|t| t.function.name == "bash_run"));
    }

    #[test]
    fn filter_tool_definitions_no_filters() {
        let defs = tool_definitions();
        let original_count = defs.len();
        let filtered = filter_tool_definitions(defs, None, None);
        assert_eq!(filtered.len(), original_count);
    }
}
