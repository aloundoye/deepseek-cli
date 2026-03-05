mod fuzzy_edit;
mod plugins;
mod shell;
pub mod tool_tiers;
pub mod validation;

pub use codingbuddy_core::ToolTier;
pub use tool_tiers::{
    ToolContextSignals, detect_signals, format_tool_search_results, search_extended_tools,
    tiered_tool_definitions, tool_search_definition, tool_tier,
};
pub use validation::{normalize_tool_args, normalize_tool_args_with_workspace, validate_tool_args};

use anyhow::{Result, anyhow};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use chrono::Utc;
use codingbuddy_chrome::{ChromeSession, ScreenshotFormat};
use codingbuddy_core::{
    AppConfig, ApprovedToolCall, EventEnvelope, EventKind, FunctionDefinition, ToolCall,
    ToolDefinition, ToolHost, ToolProposal, ToolResult,
};
use codingbuddy_diff::PatchStore;
use codingbuddy_hooks::{HookContext, HookRuntime};
use codingbuddy_index::IndexService;
use codingbuddy_memory::MemoryManager;
use codingbuddy_policy::PolicyEngine;
use codingbuddy_store::Store;
use ignore::WalkBuilder;
pub use plugins::{
    CatalogPlugin, PluginCommandPrompt, PluginInfo, PluginManager, PluginVerifyResult,
    plugin_tool_definitions,
};
use plugins::{plugin_command_lookup_name, plugin_tool_api_name};
use serde_json::json;
use sha2::Digest;
pub use shell::{PlatformShellRunner, ShellRunResult, ShellRunner};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Duration;
use uuid::Uuid;

use fuzzy_edit::apply_single_edit;
#[cfg(test)]
use fuzzy_edit::{
    fuzzy_block_anchor, fuzzy_context_aware, fuzzy_escape_normalized, fuzzy_indentation_flexible,
    fuzzy_line_trimmed, fuzzy_trimmed_boundary, fuzzy_whitespace_normalized,
};

const DEFAULT_TIMEOUT_SECONDS: u64 = 120;
const READ_MAX_BYTES_DEFAULT: usize = 1_000_000;

/// Check whether a tool (by internal name) is blocked in review mode.
fn is_review_blocked(tool_name: &str) -> bool {
    codingbuddy_core::ToolName::from_internal_name(tool_name).is_some_and(|t| t.is_review_blocked())
}

/// Callback type for executing MCP tool calls.
/// Takes (server_id, tool_name, arguments) and returns the tool result.
pub type McpExecutor =
    Arc<dyn Fn(&str, &str, &serde_json::Value) -> Result<serde_json::Value> + Send + Sync>;

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
    chrome_allow_stub_fallback: bool,
    lint_after_edit: Option<String>,
    diagnostics_after_edit: bool,
    review_mode: bool,
    mcp_executor: Option<McpExecutor>,
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
            chrome_allow_stub_fallback: cfg.tools.chrome.allow_stub_fallback,
            lint_after_edit,
            diagnostics_after_edit: cfg.tools.diagnostics_after_edit,
            review_mode: false,
            mcp_executor: None,
        })
    }

    /// Set the MCP executor callback for handling `mcp__*` tool calls.
    pub fn set_mcp_executor(&mut self, executor: McpExecutor) {
        self.mcp_executor = Some(executor);
    }

    pub fn index(&self) -> &codingbuddy_index::IndexService {
        &self.index
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

    fn chrome_session_from_call(&self, call: &ToolCall) -> Result<ChromeSession> {
        let port = Self::chrome_port_from_call(call)?;
        let mut session = ChromeSession::new(port)?;
        session.set_allow_stub_fallback(self.chrome_allow_stub_fallback);
        session.check_connection()?;
        Ok(session)
    }

    fn run_tool(&self, call: &ToolCall) -> Result<serde_json::Value> {
        // Enforce review mode: block all non-read tools
        if self.review_mode && is_review_blocked(&call.name) {
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

                // Auto-lint after edit (lint-fix loop)
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
                // Auto-diagnostics after edit
                if self.diagnostics_after_edit {
                    maybe_run_auto_diagnostics(
                        self.runner.as_ref(),
                        &self.workspace,
                        Some(path),
                        &mut result,
                    );
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
                let scope = call.args.get("scope").and_then(|v| v.as_str());
                Ok(serde_json::to_value(self.index.query(q, top_k, scope)?)?)
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
            "patch.direct" => {
                // Single-step: stage + apply in one call. Ideal for DeepSeek-reasoner
                // which naturally produces unified diffs when thinking.
                let diff = call
                    .args
                    .get("unified_diff")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("unified_diff missing"))?;
                let patch = self.patches.stage(diff, &[])?;
                let (applied, conflicts) = self.patches.apply(&self.workspace, patch.patch_id)?;
                let mut result = if applied {
                    json!({
                        "applied": true,
                        "patch_id": patch.patch_id.to_string(),
                        "files": patch.target_files,
                    })
                } else {
                    json!({
                        "applied": false,
                        "patch_id": patch.patch_id.to_string(),
                        "conflicts": conflicts,
                        "files": patch.target_files,
                    })
                };
                // Auto-diagnostics after patch application
                if applied && self.diagnostics_after_edit {
                    maybe_run_auto_diagnostics(
                        self.runner.as_ref(),
                        &self.workspace,
                        None,
                        &mut result,
                    );
                }
                Ok(result)
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
                // Accept both "cmd" (canonical) and "command" (model hallucination)
                let cmd = call
                    .args
                    .get("cmd")
                    .or_else(|| call.args.get("command"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("cmd missing — expected 'cmd' parameter"))?;
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
                    .user_agent("codingbuddy-cli/0.2")
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
                    .user_agent("codingbuddy-cli/0.2")
                    .build()?;
                let resp = client.get(&search_url).send()?;
                let body = resp.text()?;

                // Parse search results from HTML
                let results = parse_search_results(&body, max_results);
                let results_json = serde_json::to_string(&results)?;
                let results_count = results.len() as u64;

                // Cache the results
                if let Err(e) =
                    self.store
                        .set_web_search_cache(&codingbuddy_store::WebSearchCacheRecord {
                            query_hash: query_hash.clone(),
                            query: query.to_string(),
                            results_json: results_json.clone(),
                            results_count,
                            cached_at: chrono::Utc::now().to_rfc3339(),
                            ttl_seconds: 900, // 15 minutes
                        })
                {
                    eprintln!("[deepseek WARN] web_search: failed to cache results: {e}");
                }

                // Emit event
                let seq = self.store.next_seq_no(uuid::Uuid::nil()).unwrap_or(1);
                if let Err(e) = self.store.append_event(&codingbuddy_core::EventEnvelope {
                    seq_no: seq,
                    at: chrono::Utc::now(),
                    session_id: uuid::Uuid::nil(),
                    kind: codingbuddy_core::EventKind::WebSearchExecuted {
                        query: query.to_string(),
                        results_count,
                        cached: false,
                    },
                }) {
                    eprintln!("[deepseek WARN] web_search: failed to emit event: {e}");
                }

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
                let session = self.chrome_session_from_call(call)?;
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
                let session = self.chrome_session_from_call(call)?;
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
                let session = self.chrome_session_from_call(call)?;
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
                let session = self.chrome_session_from_call(call)?;
                let image_base64 = session.screenshot(format)?;
                Ok(json!({
                    "format": call.args.get("format").and_then(|v| v.as_str()).unwrap_or("png"),
                    "base64": image_base64
                }))
            }
            "chrome.read_console" => {
                let session = self.chrome_session_from_call(call)?;
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
                let session = self.chrome_session_from_call(call)?;
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
                // Auto-diagnostics after multi_edit
                if self.diagnostics_after_edit {
                    maybe_run_auto_diagnostics(
                        self.runner.as_ref(),
                        &self.workspace,
                        None,
                        &mut result,
                    );
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
            name if name.starts_with("plugin__") => self.run_plugin_tool(name, &call.args),
            name if name.starts_with("mcp__") => {
                let rest = &name[5..]; // skip "mcp__"
                let (server_id, tool_name) = rest
                    .split_once("__")
                    .ok_or_else(|| anyhow!("invalid MCP tool name: {name}"))?;
                let args = call.args.get("arguments").cloned().unwrap_or(json!({}));
                match &self.mcp_executor {
                    Some(executor) => executor(server_id, tool_name, &args),
                    None => Err(anyhow!(
                        "MCP tool '{name}' called but no MCP executor configured"
                    )),
                }
            }
            "batch" => {
                let tool_calls = call
                    .args
                    .get("tool_calls")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| anyhow!("tool_calls array missing"))?;
                if tool_calls.len() > 25 {
                    return Err(anyhow!(
                        "batch limited to 25 tool calls, got {}",
                        tool_calls.len()
                    ));
                }

                // Only allow read-only tools in batch (no writes, no nested batch)
                let read_only_tools = [
                    "fs.read",
                    "fs.list",
                    "fs.glob",
                    "fs.grep",
                    "fs.search_rg",
                    "git.status",
                    "git.diff",
                    "git.show",
                    "web.fetch",
                    "web.search",
                    "notebook.read",
                    "index.query",
                    "diagnostics.check",
                ];
                for tc in tool_calls {
                    let tool_name = tc.get("tool").and_then(|v| v.as_str()).unwrap_or("");
                    if tool_name == "batch" {
                        return Err(anyhow!("cannot nest batch inside batch"));
                    }
                    if !read_only_tools.contains(&tool_name) {
                        return Err(anyhow!(
                            "batch only allows read-only tools, got '{tool_name}'. \
                             Use individual tool calls for write operations."
                        ));
                    }
                }

                // Execute each sub-call sequentially (reusing existing handlers)
                let mut results = Vec::with_capacity(tool_calls.len());
                let mut successes = 0usize;
                for tc in tool_calls {
                    let tool_name = tc.get("tool").and_then(|v| v.as_str()).unwrap_or("unknown");
                    let params = tc.get("parameters").cloned().unwrap_or(json!({}));
                    let sub_call = ToolCall {
                        name: tool_name.to_string(),
                        args: params,
                        requires_approval: false,
                    };
                    let approved = ApprovedToolCall {
                        invocation_id: Uuid::now_v7(),
                        call: sub_call,
                    };
                    let sub_result = self.execute(approved);
                    if sub_result.success {
                        successes += 1;
                    }
                    results.push(json!({
                        "tool": tool_name,
                        "success": sub_result.success,
                        "output": sub_result.output,
                    }));
                }
                let total = results.len();
                Ok(json!({
                    "results": results,
                    "total": total,
                    "succeeded": successes,
                    "failed": total - successes,
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
        if is_isolated_sandbox_mode(&self.sandbox_mode) && detect_container_environment().is_none()
        {
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
        if let Ok(template) = std::env::var("CODINGBUDDY_SANDBOX_WRAPPER")
            && !template.trim().is_empty()
        {
            let wrapped = render_wrapper_template(template.trim(), &workspace, cmd)?;
            return self.run_cmd(&wrapped, timeout_secs);
        }
        if let Some(wrapped) = auto_isolated_wrapper_command(&workspace, cmd) {
            return self.run_cmd(&wrapped, timeout_secs);
        }
        #[cfg(target_os = "windows")]
        {
            // Windows doesn't have a built-in wrapper path today. Fall back to direct
            // execution with strict logical isolation checks.
            if command_references_outside_workspace(cmd, &workspace) {
                return Err(anyhow!(
                    "sandbox_mode={} blocked path outside workspace: {}",
                    self.sandbox_mode,
                    cmd
                ));
            }
            if command_has_network_egress_intent(cmd) {
                return Err(anyhow!(
                    "sandbox_mode={} blocked network command: {}",
                    self.sandbox_mode,
                    cmd
                ));
            }
            self.run_cmd(cmd, timeout_secs)
        }
        #[cfg(not(target_os = "windows"))]
        Err(anyhow!(
            "sandbox_mode={} requires an OS-level wrapper (set policy.sandbox_wrapper or CODINGBUDDY_SANDBOX_WRAPPER) or install bwrap/firejail/sandbox-exec",
            self.sandbox_mode
        ))
    }

    fn run_plugin_tool(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        let manager = self
            .plugins
            .as_ref()
            .ok_or_else(|| anyhow!("plugin runtime unavailable"))?;
        let (plugin_id, command_name) = self.resolve_plugin_command(tool_name)?;

        let input = args.get("arguments").or_else(|| args.get("input"));
        let input = match input {
            Some(serde_json::Value::String(text)) => Some(text.clone()),
            Some(value) if !value.is_null() => Some(serde_json::to_string(value)?),
            _ => None,
        };

        let rendered =
            manager.render_command_prompt(&plugin_id, &command_name, input.as_deref())?;
        Ok(json!({
            "plugin_id": rendered.plugin_id,
            "command_name": rendered.command_name,
            "source_path": rendered.source_path,
            "prompt": rendered.prompt,
        }))
    }

    fn resolve_plugin_command(&self, tool_name: &str) -> Result<(String, String)> {
        let manager = self
            .plugins
            .as_ref()
            .ok_or_else(|| anyhow!("plugin runtime unavailable"))?;
        let plugins = manager.list()?;
        for plugin in plugins.into_iter().filter(|p| p.enabled) {
            for cmd_path in &plugin.commands {
                let command_name = plugin_command_lookup_name(&plugin.root, cmd_path);
                let candidate = plugin_tool_api_name(&plugin.manifest.id, &command_name);
                if candidate == tool_name {
                    return Ok((plugin.manifest.id.clone(), command_name));
                }
            }
        }
        Err(anyhow!("unknown plugin tool: {tool_name}"))
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
            Err(err) => {
                let message = err.to_string();
                let payload = if call.name.starts_with("chrome.") {
                    let (kind, hints) = classify_chrome_error(&message);
                    json!({
                        "error": message,
                        "error_kind": kind,
                        "hints": hints,
                    })
                } else {
                    json!({"error": message})
                };
                (false, payload)
            }
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

fn classify_chrome_error(message: &str) -> (&'static str, Vec<&'static str>) {
    let lower = message.to_ascii_lowercase();
    if lower.contains("connection refused")
        || lower.contains("endpoint_unreachable")
        || lower.contains("debugging endpoint is unavailable")
    {
        (
            "endpoint_unreachable",
            vec![
                "Start Chrome with --remote-debugging-port=9222",
                "Verify CODINGBUDDY_CHROME_PORT points to the active debugging port",
            ],
        )
    } else if lower.contains("no_page_targets") || lower.contains("no debuggable page target") {
        (
            "no_page_targets",
            vec![
                "Open at least one tab in the target Chrome profile",
                "Run /chrome reconnect to create a recovery tab",
            ],
        )
    } else if lower.contains("timed out") || lower.contains("endpoint_timeout") {
        (
            "endpoint_timeout",
            vec![
                "Retry /chrome reconnect once the browser is responsive",
                "Confirm no firewall/proxy is blocking localhost CDP traffic",
            ],
        )
    } else {
        (
            "chrome_error",
            vec![
                "Run /chrome status for live connection diagnostics",
                "Run /chrome reconnect to recover stale sessions",
            ],
        )
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
                description: "Reads a file from the filesystem and returns its contents with line numbers.\n\n\
## CRITICAL RULES\n\
- You MUST call fs_read BEFORE making any claims about a file's contents. NEVER guess or fabricate what a file contains.\n\
- You MUST call fs_read BEFORE calling fs_edit on any file. The edit tool requires an exact string match, so you need to see the current content first.\n\
- DO NOT use bash_run with cat, head, tail, or sed to read files — use this tool instead. It provides structured output with line numbers and metadata.\n\n\
## Usage\n\
- By default reads the entire file (up to max_bytes, default 1MB).\n\
- For large files, use start_line and end_line to read specific sections. This is especially useful when you already know which lines to inspect.\n\
- Returns line-numbered content in the format `  N→content` for easy reference.\n\
- For binary files (images), returns base64-encoded content with MIME type metadata.\n\
- For PDF files, extracts text content. Use the `pages` parameter (e.g. '1-5') for large PDFs.\n\n\
## When to use\n\
- Before editing any file (ALWAYS)\n\
- To verify file contents before making claims about them\n\
- To understand existing code before suggesting modifications\n\
- To check current state after making edits\n\n\
## When NOT to use\n\
- To search for content across many files — use fs_grep instead\n\
- To find files by name pattern — use fs_glob instead\n\
- To list directory contents — use fs_list instead".to_string(),
            strict: None,
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
                description: "Creates or completely overwrites a file with the specified content. Creates parent directories automatically if they don't exist.\n\n\
## CRITICAL RULES\n\
- ALWAYS prefer fs_edit over fs_write for modifying existing files. fs_write replaces the ENTIRE file content, which is dangerous for partial changes.\n\
- If the file already exists, you MUST have read it with fs_read first to understand what will be overwritten.\n\
- NEVER write files that contain secrets, credentials, API keys, or sensitive data.\n\n\
## When to use\n\
- Creating new files that don't exist yet\n\
- Complete rewrites where fs_edit would require too many individual replacements\n\
- Writing generated content (configs, boilerplate, test fixtures)\n\n\
## When NOT to use\n\
- Making targeted changes to existing files — use fs_edit instead\n\
- Making multiple small edits across a file — use fs_edit or multi_edit instead".to_string(),
            strict: None,
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
                description: "Performs exact string replacement in a file. The search string must match EXACTLY — including whitespace, indentation, and line endings.\n\n\
## CRITICAL RULES\n\
- You MUST call fs_read on the file BEFORE using fs_edit. The search string must be copied exactly from the file's current content. If you guess the content, the edit will fail.\n\
- The search string must be unique enough to match only the intended location. If the search matches multiple places, all occurrences are replaced by default (set 'all': false for first-only).\n\
- Preserve the exact indentation (tabs/spaces) as shown in the fs_read output. The line number prefix (e.g. '  42→') is NOT part of the file content — do not include it in the search string.\n\n\
## When to use\n\
- Making targeted changes to existing files (the preferred editing approach)\n\
- Renaming variables, functions, or identifiers\n\
- Updating specific code blocks, imports, or configuration values\n\
- Any modification where you want to change only part of a file\n\n\
## When NOT to use\n\
- Creating new files — use fs_write instead\n\
- Complete file rewrites — use fs_write instead\n\
- Editing multiple files at once — use multi_edit instead\n\n\
## Tips\n\
- If the search string is not unique, include more surrounding context (extra lines above/below) to make it unique.\n\
- After editing, consider reading the file again to verify the change was applied correctly.".to_string(),
            strict: None,
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
                description: "Lists files and directories in a single directory level. Returns names, types (file/dir), and sizes.\n\n\
## When to use\n\
- To see what's in a directory before navigating deeper\n\
- To discover project structure at the top level\n\
- Quick check of a specific directory's immediate contents\n\n\
## When NOT to use\n\
- For recursive file searching — use fs_glob with a pattern like '**/*.rs' instead\n\
- For searching file contents — use fs_grep instead\n\
- To read a specific file — use fs_read instead".to_string(),
            strict: None,
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
                description: "Fast file pattern matching that finds files by name/path pattern. Returns matching file paths sorted by modification time.\n\n\
## CRITICAL RULES\n\
- ALWAYS use fs_glob for finding files by name. NEVER use bash_run with find or ls for file discovery.\n\
- Use this tool when you need to locate files before reading or editing them.\n\n\
## Pattern examples\n\
- '**/*.rs' — all Rust files recursively\n\
- 'src/**/*.ts' — TypeScript files under src/\n\
- '*.json' — JSON files in root only\n\
- '**/test_*.py' — Python test files anywhere\n\
- 'Cargo.toml' — exact file name anywhere (if base is '.')\n\n\
## When to use\n\
- Finding files by extension or name pattern\n\
- Discovering project structure (e.g. all config files, all test files)\n\
- Locating a file when you know part of its name but not the full path\n\n\
## When NOT to use\n\
- Searching file CONTENTS — use fs_grep instead (fs_glob only matches file paths)\n\
- Listing a single directory — use fs_list for simpler directory listing\n\
- Reading file contents — use fs_read after finding the path".to_string(),
            strict: None,
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
                description: "Searches file contents using regex patterns. Returns matching lines with file paths and line numbers. Built on ripgrep for fast, accurate results.\n\n\
## CRITICAL RULES\n\
- ALWAYS use fs_grep for searching file contents. NEVER use bash_run with grep, rg, ag, or ack — this tool is faster and provides structured output.\n\
- Supports full regex syntax (e.g. 'log.*Error', 'function\\s+\\w+', 'impl\\s+Display').\n\n\
## When to use\n\
- Finding where a function, class, variable, or string is defined or used\n\
- Searching for error messages, log patterns, or specific code constructs\n\
- Finding all usages of an API or import across the codebase\n\
- Locating TODO/FIXME/HACK comments\n\n\
## When NOT to use\n\
- Finding files by name pattern — use fs_glob instead (fs_grep searches contents, not names)\n\
- Reading a specific file — use fs_read instead\n\
- Listing directory contents — use fs_list instead\n\n\
## Tips\n\
- Use the 'glob' parameter to narrow the search to specific file types (e.g. '**/*.rs' for Rust files only)\n\
- Use case_sensitive: false for case-insensitive searches\n\
- Results include file path, line number, and matching line content".to_string(),
            strict: None,
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
                description: "Executes a shell command in the workspace directory and returns stdout/stderr.\n\n\
## CRITICAL RULES\n\
- DO NOT use bash_run for operations that have dedicated tools:\n\
  - Reading files: use fs_read (NOT cat, head, tail, less)\n\
  - Searching file contents: use fs_grep (NOT grep, rg, ag, ack)\n\
  - Finding files: use fs_glob (NOT find, ls -R, fd)\n\
  - Editing files: use fs_edit or multi_edit (NOT sed, awk, perl -i)\n\
  - Writing files: use fs_write (NOT echo >, cat <<EOF, tee)\n\
  - Git status: use git_status (NOT git status)\n\
  - Git diff: use git_diff (NOT git diff)\n\
Using dedicated tools provides structured output, better error handling, and clearer audit trail.\n\n\
## USE bash_run for\n\
- Building projects: cargo build, npm run build, make\n\
- Running tests: cargo test, pytest, npm test\n\
- Package management: cargo add, npm install, pip install\n\
- Git operations beyond status/diff: git commit, git push, git log, git branch\n\
- System commands: docker, curl (for APIs), env checks, process management\n\
- Language-specific tools: rustfmt, eslint --fix, black\n\
- Any command that doesn't have a dedicated tool equivalent\n\n\
## Tips\n\
- Always provide a 'description' so the user understands what the command does\n\
- Commands time out after 120 seconds by default. Set a higher timeout for long builds.\n\
- Quote file paths with spaces using double quotes\n\
- Prefer absolute paths to avoid directory confusion".to_string(),
            strict: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds. Defaults to 120."
                        },
                        "description": {
                            "type": "string",
                            "description": "Short description of what this command does"
                        }
                    },
                    "required": ["command"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "multi_edit".to_string(),
                description: "Apply multiple search/replace edits across one or more files in a single atomic operation. Each file entry contains a path and an array of edits.\n\n\
## CRITICAL RULES\n\
- You MUST have read each file with fs_read before editing it. Search strings must exactly match current file content.\n\
- Same rules as fs_edit apply to each individual edit: exact string match required, preserve indentation.\n\n\
## When to use\n\
- Making related changes across multiple files (e.g. renaming a function and updating all call sites)\n\
- Applying several edits to the same file in one operation\n\
- Refactoring that touches many files simultaneously\n\n\
## When NOT to use\n\
- Single edit to a single file — use fs_edit instead (simpler)\n\
- Creating new files — use fs_write instead\n\
- Complete file rewrites — use fs_write instead".to_string(),
            strict: None,
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
                description: "Shows the working tree status (staged, unstaged, untracked files) in short format.\n\n\
## When to use\n\
- Before committing: check which files are staged, modified, or untracked\n\
- After making changes: verify that only the intended files were modified\n\
- Before creating a PR: ensure no unintended files are included\n\
- To check for merge conflicts or unresolved state\n\n\
## When NOT to use\n\
- To see file contents — use fs_read instead\n\
- To see actual diff of changes — use git_diff instead\n\
- To see commit history — use bash_run with 'git log'\n\n\
## Common mistakes\n\
- Do NOT run bash_run with 'git status' — use this tool, it's faster and returns structured output".to_string(),
            strict: None,
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
                description: "Shows the unified diff of all unstaged changes in the working directory.\n\n\
## When to use\n\
- Before committing: review exactly what changed to write an accurate commit message\n\
- After editing files: verify your changes are correct and complete\n\
- To understand what modifications exist before deciding next steps\n\
- To compare current working tree against the last commit\n\n\
## When NOT to use\n\
- To see staged changes — use bash_run with 'git diff --cached' instead\n\
- To compare branches — use bash_run with 'git diff branch1..branch2'\n\
- To see which files changed (without content) — use git_status instead\n\
- To see a specific file's full content — use fs_read instead\n\n\
## Common mistakes\n\
- Do NOT run bash_run with 'git diff' — use this tool, it returns structured output\n\
- This only shows UNSTAGED changes. If you just staged files, the diff will be empty".to_string(),
            strict: None,
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
                description: "Fetches content from a URL and returns it as text. HTML is automatically stripped to plain text.\n\n\
## When to use\n\
- Fetching documentation, API references, or web pages referenced by the user\n\
- Downloading configuration or data files from URLs\n\
- Checking API endpoints or service responses\n\n\
## Limitations\n\
- Will fail for authenticated/private URLs (Google Docs, Confluence, Jira, private GitHub repos)\n\
- HTTP URLs are automatically upgraded to HTTPS\n\
- Large pages may be truncated at max_bytes (default 500KB)".to_string(),
            strict: None,
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
                description: "Searches the web and returns results with titles, URLs, and snippets. Use for finding documentation, looking up APIs, researching error messages, or getting up-to-date information beyond the model's training data.".to_string(),
            strict: None,
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
                description: "Read a Jupyter notebook (.ipynb file), returning all cells with their type (code/markdown), source content, and output summaries. Use this to understand notebook structure before making edits with notebook_edit.".to_string(),
            strict: None,
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
                description: "Edit a cell in a Jupyter notebook (.ipynb file). Supports replace (overwrite cell content), insert (add new cell), and delete operations. You MUST read the notebook with notebook_read first to know the cell indices and current content.".to_string(),
            strict: None,
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
                description: "Show a git object (commit, tag, tree, or blob). Use to inspect commit details, view files at specific revisions (e.g. 'HEAD:src/main.rs'), or compare changes between commits (e.g. 'main..feature'). For current working directory changes, use git_diff instead.".to_string(),
            strict: None,
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
                description: "Full-text semantic search across the code index. Returns matching file paths and snippets ranked by relevance. Use for conceptual searches (e.g. 'authentication middleware') when you don't know the exact string. For exact pattern matching, use fs_grep instead.".to_string(),
            strict: None,
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
                description: "Stage a unified diff as a patch for later application. Returns a patch_id for use with patch_apply. Use this for complex multi-file changes where you want to prepare a patch first and apply it atomically. For simple edits, prefer fs_edit or multi_edit instead.".to_string(),
            strict: None,
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
                description: "Apply a previously staged patch by its patch_id (from patch_stage). Returns success/failure and any conflicts. Always stage with patch_stage first before applying.".to_string(),
            strict: None,
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
                name: "patch_direct".to_string(),
                description: "Apply a unified diff directly to the workspace in one step. \
This is the preferred tool for applying code changes when you have a unified diff. \
Unlike patch_stage + patch_apply (which require two calls), this combines both steps. \
\n\nThe diff must be in standard unified diff format:\n\
```\n\
--- a/path/to/file.rs\n\
+++ b/path/to/file.rs\n\
@@ -10,3 +10,4 @@\n\
 unchanged line\n\
-old line\n\
+new line\n\
+added line\n\
 unchanged line\n\
```\n\n\
Returns the list of affected files and whether the patch applied cleanly. \
If there are conflicts, they are reported in the response. \
For simple single-site edits, prefer fs_edit. Use this tool for multi-hunk or multi-file changes \
where a unified diff is more natural.".to_string(),
            strict: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "unified_diff": {
                            "type": "string",
                            "description": "The unified diff to apply. Must use standard unified diff format with --- a/ and +++ b/ headers."
                        }
                    },
                    "required": ["unified_diff"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "diagnostics_check".to_string(),
                description: "Run language-specific diagnostics (cargo check, tsc, ruff, etc.) on the project or a specific path. Auto-detects the appropriate checker based on project files. Use this after making changes to verify they compile and pass basic checks. Prefer this over running build commands manually with bash_run when you just need to check for errors.".to_string(),
            strict: None,
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
        // ── Batch tool ────────────────────────────────────────────────────
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "batch".to_string(),
                description: "Execute multiple read-only tool calls in a single request. \
Use this when you need to perform several independent read operations (reading multiple files, \
searching in parallel, checking git status and diff together). \
\n\nRules:\n\
- Only read-only tools allowed: fs_read, fs_list, fs_glob, fs_grep, git_status, git_diff, \
git_show, web_fetch, web_search, notebook_read, index_query, diagnostics_check\n\
- Cannot nest batch inside batch\n\
- Cannot batch write tools (fs_edit, fs_write, bash_run) — use individual calls for those\n\
- Maximum 25 tool calls per batch\n\
\nReturns an array of results with per-tool success/error status.".to_string(),
            strict: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "tool_calls": {
                            "type": "array",
                            "description": "Array of tool calls to execute",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tool": {
                                        "type": "string",
                                        "description": "Name of the tool to call (e.g. 'fs.read', 'fs.grep')"
                                    },
                                    "parameters": {
                                        "type": "object",
                                        "description": "Parameters to pass to the tool"
                                    }
                                },
                                "required": ["tool", "parameters"]
                            },
                            "minItems": 1,
                            "maxItems": 25
                        }
                    },
                    "required": ["tool_calls"]
                }),
            },
        },
        // ── Chrome browser automation tools ─────────────────────────────────
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "chrome_navigate".to_string(),
                description: "Navigate a Chrome browser to a URL. Requires Chrome to be running with remote debugging enabled (--remote-debugging-port=9222). Use for web development testing, taking screenshots of web pages, or automating browser interactions.".to_string(),
            strict: None,
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
                description: "Click an element on the current Chrome page by CSS selector. Use after chrome_navigate to interact with page elements. Requires Chrome remote debugging.".to_string(),
            strict: None,
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
                description: "Type text into an input element on the current Chrome page, selected by CSS selector. Use for filling forms, search boxes, or any text input. Requires Chrome remote debugging.".to_string(),
            strict: None,
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
                description: "Capture a screenshot of the current Chrome page. Returns the image as base64. Use for visual verification of web changes, debugging UI issues, or documenting current state. Requires Chrome remote debugging.".to_string(),
            strict: None,
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
                description: "Read the browser's console log entries. Use to check for JavaScript errors, warnings, or debug output after navigating to a page. Requires Chrome remote debugging.".to_string(),
            strict: None,
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
                description: "Evaluate a JavaScript expression in the Chrome browser and return the result. Use for querying DOM state, checking variable values, or running custom JavaScript. Requires Chrome remote debugging.".to_string(),
            strict: None,
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
                description: "Ask the user a question and wait for their response. Use when requirements are genuinely ambiguous or you need a decision to proceed.\n\n\
## CRITICAL RULES\n\
- Do NOT ask to confirm actions the user already explicitly requested. If they said 'fix the bug', just fix it.\n\
- Do NOT ask permission before using tools. Just use them.\n\
- DO ask when: there are multiple valid approaches and user preference matters, requirements are unclear, you need information that isn't in the codebase.\n\n\
## Tips\n\
- Provide 'options' when the choices are clear-cut, so the user can pick quickly\n\
- Keep questions concise and specific — ask one thing at a time".to_string(),
            strict: None,
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
                description: "Create a task to track progress on the current work. Returns the task ID.\n\n\
## When to use\n\
- Complex multi-step work: break the work into trackable tasks before starting\n\
- User provides multiple items: capture each as a separate task\n\
- Non-trivial implementations: create tasks so the user can see progress\n\n\
## When NOT to use\n\
- Single trivial tasks (e.g. fixing a typo, answering a question)\n\
- Tasks that can be completed in one step without tracking\n\n\
## Tips\n\
- Use imperative form for subject ('Fix auth bug', not 'Fixing auth bug')\n\
- Include enough detail in description for another agent to complete the task\n\
- After creating tasks, use task_update to set status as you work".to_string(),
            strict: None,
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
                description: "Update a task's status or details.\n\n\
## When to use\n\
- Mark a task as in_progress BEFORE starting work on it\n\
- Mark a task as completed AFTER fully finishing the work\n\
- Mark a task as failed if you encounter unresolvable blockers\n\
- Update the outcome field with a summary when completing or failing\n\n\
## When NOT to use\n\
- Do NOT mark a task completed if tests are failing or implementation is partial\n\
- Do NOT update tasks that don't exist — use task_list to check first\n\n\
## Status workflow\n\
pending → in_progress → completed (or failed)\n\
Always set in_progress before starting, completed only when fully done".to_string(),
            strict: None,
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
                name: "todo_read".to_string(),
                description: "Read the session-native working checklist for this conversation.\n\n\
## When to use\n\
- At the start of complex work to see the current checklist\n\
- Before updating todos so you keep existing IDs and ordering stable\n\
- After subagent work to decide which items to mark complete\n\n\
## Returns\n\
- Current todo items with id/content/status\n\
- Summary counts (active/completed/in_progress)\n\
- Current executing item (if any)\n\n\
## Notes\n\
- This checklist is session-local and separate from task queue delegation.\n\
- Use task_* tools for durable delegated/background units; use todo_* for the live checklist.".to_string(),
                strict: None,
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
                name: "todo_write".to_string(),
                description: "Replace the session-native working checklist with an updated ordered list.\n\n\
## CRITICAL RULES\n\
- Send the FULL desired checklist each time; omitted items are removed.\n\
- Keep exactly one in_progress item while actively executing.\n\
- Mark items completed as soon as work is verified.\n\
- Preserve existing ids when possible (read first with todo_read).\n\n\
## Status values\n\
- pending\n\
- in_progress\n\
- completed\n\
\n\
## Notes\n\
- This checklist is separate from task_* delegated work records.\n\
- For complex tasks, update this after each meaningful step.".to_string(),
                strict: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "description": "Complete ordered todo checklist for this session.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "string",
                                        "description": "Existing todo UUID (optional; preserve if known)."
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Todo item description."
                                    },
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "in_progress", "completed"],
                                        "description": "Todo status."
                                    }
                                },
                                "required": ["content"]
                            }
                        }
                    },
                    "required": ["items"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "spawn_task".to_string(),
                description: "Launch a specialized sub-agent to handle complex, multi-step tasks autonomously. Each agent type has specific capabilities.\n\n\
## Agent types\n\
- 'explore': Fast codebase search/read — use for finding files, searching code, answering questions about the codebase\n\
- 'plan': Design implementation approaches — returns step-by-step plans with architectural considerations\n\
- 'bash': Command execution — for build, test, deploy tasks that need many sequential commands\n\
- 'general-purpose': Full capabilities — for complex multi-step tasks requiring all tools\n\n\
## Tips\n\
- Launch multiple agents concurrently when tasks are independent (use separate calls)\n\
- Provide clear, detailed prompts so the agent can work autonomously".to_string(),
            strict: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "A short (3-5 word) description of the task"
                        },
                        "prompt": {
                            "type": "string",
                            "description": "The task for the agent to perform"
                        },
                        "subagent_type": {
                            "type": "string",
                            "enum": ["explore", "plan", "bash", "general-purpose"],
                            "description": "The type of specialized agent: 'explore' for codebase search/read, 'plan' for designing approaches, 'bash' for command execution, 'general-purpose' for complex multi-step tasks"
                        },
                        "model": {
                            "type": "string",
                            "description": "Optional model override for this delegated task"
                        },
                        "max_turns": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Optional maximum turns for the delegated task"
                        },
                        "run_in_background": {
                            "type": "boolean",
                            "description": "When true, run the delegated task as a detached background job and return immediately with tracking IDs"
                        }
                    },
                    "required": ["description", "prompt", "subagent_type"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "task_output".to_string(),
                description: "Read the latest persisted output for a delegated task or subagent run. Use this after spawn_task when you need the current status, the child session ID, or the final summary.".to_string(),
                strict: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Task UUID returned by task_create or spawn_task"
                        },
                        "run_id": {
                            "type": "string",
                            "description": "Subagent run UUID returned by spawn_task"
                        }
                    },
                    "required": []
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "task_stop".to_string(),
                description: "Stop a running delegated background task. Use this when a spawned task is hung, no longer needed, or should be cancelled before completion.".to_string(),
                strict: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Task UUID returned by task_create or spawn_task"
                        },
                        "run_id": {
                            "type": "string",
                            "description": "Subagent run UUID returned by spawn_task"
                        }
                    },
                    "required": []
                }),
            },
        },
        // ── Plan mode tools (handled by AgentEngine) ──────────────────────
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "enter_plan_mode".to_string(),
                description: "Enter plan mode to design an implementation approach before writing code. In plan mode, you can only use read-only tools (Read, Glob, Grep, search, git status/diff). Use this proactively when the task requires planning: new features, multiple valid approaches, multi-file changes, or unclear requirements. You will explore the codebase, design a plan, and present it for user approval before executing.".to_string(),
            strict: None,
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
                name: "exit_plan_mode".to_string(),
                description: "Exit plan mode after writing your plan. This signals that you are done planning and ready for the user to review and approve. The user will see the plan and decide whether to let you proceed with execution. Only use this after you have completed your plan.".to_string(),
            strict: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "allowedPrompts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tool": { "type": "string", "description": "The tool this permission applies to (e.g. 'bash_run')" },
                                    "prompt": { "type": "string", "description": "Description of the action (e.g. 'run tests', 'install dependencies')" }
                                },
                                "required": ["tool", "prompt"]
                            },
                            "description": "Prompt-based permissions needed to implement the plan"
                        }
                    },
                    "required": []
                }),
            },
        },
        // ── Background task management tools (handled by AgentEngine) ─────
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "task_get".to_string(),
                description: "Retrieve full details of a task by its ID.\n\n\
## When to use\n\
- Before starting work on a task: read the full description and requirements\n\
- To check task dependencies (blockedBy) before claiming it\n\
- To understand the full context of what a task requires\n\n\
## Returns\n\
- subject, description, status (pending/in_progress/completed/failed)\n\
- blocks: tasks waiting on this one\n\
- blockedBy: tasks that must complete first\n\n\
## Tips\n\
- Use task_list for a summary of all tasks; use task_get for one task's full details\n\
- Always verify blockedBy is empty before starting work on a task".to_string(),
            strict: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The ID of the task to retrieve"
                        }
                    },
                    "required": ["task_id"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "task_list".to_string(),
                description: "List all tasks in the task list with summary info.\n\n\
## When to use\n\
- To see what tasks are available to work on (status=pending, not blocked)\n\
- To check overall progress on the project\n\
- After completing a task, to find the next one to work on\n\
- To verify task dependencies and identify blocked work\n\n\
## Returns\n\
For each task: id, subject, status, owner, blockedBy list\n\n\
## Tips\n\
- Prefer working on tasks in ID order (lowest first) — earlier tasks set up context\n\
- Use task_get with a specific ID to see full description and requirements\n\
- Tasks with non-empty blockedBy cannot be started until dependencies resolve".to_string(),
            strict: None,
                parameters: json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            },
        },
        // ── Skill tool (LLM invokes slash commands) ──────────────────────
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "skill".to_string(),
                description: "Execute a skill (slash command) within the current conversation. Use this when the user asks you to perform tasks that match available skills, or when they reference a slash command like '/commit' or '/review-pr'.".to_string(),
            strict: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "skill": {
                            "type": "string",
                            "description": "The skill name to invoke (e.g. 'commit', 'review-pr', 'pdf')"
                        },
                        "args": {
                            "type": "string",
                            "description": "Optional arguments for the skill"
                        }
                    },
                    "required": ["skill"]
                }),
            },
        },
        // ── extended_thinking: R1 consultation for complex subproblems ──
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "extended_thinking".to_string(),
                description: "Escalate to deep reasoning model for problems requiring extensive \
                              chain-of-thought beyond the main model's thinking budget. Use when \
                              you encounter: repeated failures on the same approach, architectural \
                              decisions with multiple valid options, complex error analysis needing \
                              root cause identification, or task decomposition for multi-step \
                              changes. Returns strategic advice — you keep control and execute \
                              the recommended approach."
                    .to_string(),
                strict: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The specific question to reason about"
                        },
                        "context": {
                            "type": "string",
                            "description": "Relevant context: error messages, file contents, constraints, what you have tried so far"
                        },
                        "type": {
                            "type": "string",
                            "enum": ["error_analysis", "architecture_advice", "plan_review", "task_decomposition"],
                            "description": "Type of reasoning needed"
                        }
                    },
                    "required": ["question", "context", "type"]
                }),
            },
        },
    ]
}

/// Tools allowed in plan mode.
///
/// Plan mode blocks file edits and shell execution, but still allows planning
/// metadata tools such as task creation and plan completion.
pub static PLAN_MODE_TOOLS: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    tool_definitions()
        .into_iter()
        .filter_map(|tool| codingbuddy_core::ToolName::from_api_name(&tool.function.name))
        .filter(|tool| tool.is_allowed_in_phase(codingbuddy_core::TaskPhase::Plan))
        .map(|tool| tool.as_api_name())
        .collect()
});

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
///
/// Delegates to [`codingbuddy_core::ToolName`] for known tools. Unknown names
/// (plugins, MCP tools) pass through unchanged.
pub fn map_tool_name(function_name: &str) -> &str {
    match codingbuddy_core::ToolName::from_api_name(function_name) {
        Some(t) => t.as_internal(),
        None => function_name,
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
                Some(
                    "Hint: file does not exist. Use fs.glob to search for the correct path."
                        .to_string(),
                )
            } else if lower.contains("permission denied") {
                Some(
                    "Hint: permission denied. The file may be outside the allowed workspace."
                        .to_string(),
                )
            } else {
                None
            }
        }
        "fs.write" => {
            if lower.contains("permission denied") {
                Some(
                    "Hint: permission denied. Check that the directory exists and is writable."
                        .to_string(),
                )
            } else {
                None
            }
        }
        "bash.run" => {
            if lower.contains("timed out") || lower.contains("timeout") {
                Some(
                    "Hint: command timed out. Try a shorter operation or increase the timeout."
                        .to_string(),
                )
            } else if lower.contains("not found") || lower.contains("command not found") {
                Some(
                    "Hint: command not found. Check that the program is installed and in PATH."
                        .to_string(),
                )
            } else if lower.contains("forbidden shell metacharacters") {
                Some(
                    "Hint: bash.run blocks shell metacharacters (;, &&, ||, backticks, $()). \
                     A single pipeline (|) is allowed only when each command segment is allowlisted. \
                     Do NOT retry with similar commands. Instead use the built-in tools: \
                     fs.grep for searching file contents, fs.glob for finding files by pattern, \
                     fs.read for reading files. These tools do not have shell restrictions."
                        .to_string(),
                )
            } else if lower.contains("not allowlisted") {
                Some(
                    "Hint: this command is not in the allowed command list. \
                     Do NOT retry other shell commands — most are restricted. Instead use built-in tools: \
                     fs.glob to list/find files, fs.grep to search content, fs.read to read files, \
                     git_status/git_diff/git_show for git operations. \
                     Only allowlisted commands (e.g. cargo, git, rg) can be run via bash.run."
                        .to_string(),
                )
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Validate tool call arguments against the tool's JSON schema.
///
/// Returns `Ok(())` if the arguments are valid, or `Err(message)` with a
/// structured description of which fields are wrong, so the model can
/// self-correct instead of getting cryptic runtime errors.
///
/// This complements `validation::validate_tool_args` which does deeper
/// semantic checks (required fields, range checks). This function validates
/// against the formal JSON schema from tool definitions.
pub fn validate_tool_args_schema(
    tool_name: &str,
    args: &serde_json::Value,
    tools: &[ToolDefinition],
) -> Result<(), String> {
    // Find the tool definition by API name
    let tool_def = tools.iter().find(|t| t.function.name == tool_name);
    let Some(tool_def) = tool_def else {
        return Ok(()); // Unknown tool — skip validation (MCP, plugin)
    };

    let schema = &tool_def.function.parameters;
    if schema.is_null() || (schema.is_object() && schema.as_object().unwrap().is_empty()) {
        return Ok(()); // No schema defined
    }

    // Compile and validate
    let validator = match jsonschema::validator_for(schema) {
        Ok(v) => v,
        Err(_) => return Ok(()), // Invalid schema — skip validation
    };

    let errors: Vec<String> = validator
        .iter_errors(args)
        .map(|e| {
            let path = e.instance_path.to_string();
            if path.is_empty() {
                e.to_string()
            } else {
                format!("{}: {}", path, e)
            }
        })
        .collect();

    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Invalid arguments for tool '{}': {}",
            tool_name,
            errors.join("; ")
        ))
    }
}

/// Tools that are handled by AgentEngine directly, not by LocalToolHost.
pub static AGENT_LEVEL_TOOLS: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    tool_definitions()
        .into_iter()
        .filter_map(|tool| codingbuddy_core::ToolName::from_api_name(&tool.function.name))
        .filter(|tool| tool.is_agent_level())
        .map(|tool| tool.as_api_name())
        .collect()
});

fn should_skip_rel_path(path: &Path) -> bool {
    path.components().any(|c| {
        c.as_os_str() == ".git" || c.as_os_str() == ".codingbuddy" || c.as_os_str() == "target"
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
    builder.add_custom_ignore_filename(".codingbuddyignore");

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

/// Detect if we're already running inside a container environment.
///
/// When running inside Docker/Podman/LXC, adding seatbelt/bwrap sandboxing
/// is redundant, may fail, and adds overhead. This function checks several
/// indicators and returns the container type if detected.
pub fn detect_container_environment() -> Option<&'static str> {
    // Explicit env var override
    if std::env::var("CODINGBUDDY_CONTAINER_MODE").is_ok() {
        return Some("explicit");
    }
    // Docker
    if Path::new("/.dockerenv").exists() {
        return Some("docker");
    }
    // Podman
    if Path::new("/run/.containerenv").exists() {
        return Some("podman");
    }
    // Check cgroup for container runtime indicators
    if let Ok(cgroup) = std::fs::read_to_string("/proc/1/cgroup") {
        let lower = cgroup.to_ascii_lowercase();
        if lower.contains("docker") || lower.contains("containerd") || lower.contains("lxc") {
            return Some("cgroup");
        }
    }
    None
}

fn render_wrapper_template(template: &str, workspace: &Path, cmd: &str) -> Result<String> {
    if !template.contains("{cmd}") {
        return Err(anyhow!(
            "sandbox wrapper template must include {{cmd}} placeholder"
        ));
    }
    let workspace_q = platform_shell_quote(workspace.to_string_lossy().as_ref());
    let cmd_q = platform_shell_quote(cmd);
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

#[cfg(target_os = "windows")]
fn platform_shell_quote(value: &str) -> String {
    windows_shell_quote(value)
}

#[cfg(not(target_os = "windows"))]
fn platform_shell_quote(value: &str) -> String {
    shell_quote(value)
}

#[cfg(target_os = "windows")]
fn windows_shell_quote(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    for ch in value.chars() {
        match ch {
            // Keep arguments stable when the wrapper is executed via cmd /C.
            '"' => escaped.push_str("\\\""),
            '%' => escaped.push_str("%%"),
            '^' | '&' | '|' | '<' | '>' => {
                escaped.push('^');
                escaped.push(ch);
            }
            _ => escaped.push(ch),
        }
    }
    format!("\"{escaped}\"")
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
    #[cfg(target_os = "windows")]
    {
        if token.starts_with("~\\") {
            let home = std::env::var("USERPROFILE")
                .ok()
                .or_else(|| std::env::var("HOME").ok())?;
            return Some(PathBuf::from(home).join(token.trim_start_matches("~\\")));
        }
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
            || token.starts_with("..\\")
            || token.contains("\\..\\")
            || token.ends_with("\\..")
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

/// Extract readable text from HTML, stripping tags, scripts, and styles.
/// Uses the `scraper` crate for proper DOM-based extraction.
fn strip_html_tags(html: &str) -> String {
    use scraper::{Html, Selector};

    let document = Html::parse_document(html);

    let body_sel = Selector::parse("body").unwrap_or_else(|_| Selector::parse("*").unwrap());

    let mut out = String::with_capacity(html.len() / 2);

    if let Some(element) = document.select(&body_sel).next() {
        for text in element.text() {
            // Skip text that is inside script/style by checking ancestors
            // (scraper's .text() already skips script/style content in well-formed HTML)
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                out.push_str(trimmed);
                out.push('\n');
            }
        }
    }

    // If no <body> found, fall back to full document text
    if out.is_empty() {
        // No <body> found — extract text from root element
        // (.text() already skips script/style content in well-formed HTML)
        for text_node in document.root_element().text() {
            let trimmed = text_node.trim();
            if !trimmed.is_empty() {
                out.push_str(trimmed);
                out.push('\n');
            }
        }
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

/// Parse search results from DuckDuckGo HTML response using CSS selectors.
/// Falls back to legacy string matching if selector-based parsing fails.
fn parse_search_results(html: &str, max_results: usize) -> Vec<serde_json::Value> {
    let results = parse_search_results_css(html, max_results);
    if results.is_empty() {
        // Fallback to legacy string-matching parser
        parse_search_results_legacy(html, max_results)
    } else {
        results
    }
}

/// CSS selector-based DuckDuckGo HTML parser (primary).
fn parse_search_results_css(html: &str, max_results: usize) -> Vec<serde_json::Value> {
    use scraper::{Html, Selector};

    let document = Html::parse_document(html);
    let mut results = Vec::new();

    // DuckDuckGo result containers
    let result_sel = match Selector::parse(".result, .web-result") {
        Ok(s) => s,
        Err(_) => return results,
    };
    let link_sel = Selector::parse(".result__a, .result-title-a, a.result__a")
        .unwrap_or_else(|_| Selector::parse("a").unwrap());
    let snippet_sel = Selector::parse(".result__snippet, .result-snippet")
        .unwrap_or_else(|_| Selector::parse(".snippet").unwrap());

    for element in document.select(&result_sel) {
        if results.len() >= max_results {
            break;
        }

        // Extract link and title
        if let Some(link_el) = element.select(&link_sel).next() {
            let url = link_el.value().attr("href").unwrap_or_default().to_string();
            let title: String = link_el.text().collect::<Vec<_>>().join(" ");
            let title = title.trim().to_string();

            // Extract snippet
            let snippet = if let Some(snippet_el) = element.select(&snippet_sel).next() {
                snippet_el
                    .text()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .trim()
                    .to_string()
            } else {
                String::new()
            };

            if !url.is_empty() && !title.is_empty() {
                results.push(json!({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }));
            }
        }
    }
    results
}

/// Legacy string-matching parser (fallback).
fn parse_search_results_legacy(html: &str, max_results: usize) -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    let lower = html.to_ascii_lowercase();
    let mut pos = 0;
    while results.len() < max_results {
        let link_marker = "class=\"result__a\"";
        let link_pos = match lower[pos..].find(link_marker) {
            Some(p) => pos + p,
            None => break,
        };
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

/// Run auto-diagnostics after an edit and inject results into the tool result JSON.
fn maybe_run_auto_diagnostics(
    runner: &dyn ShellRunner,
    workspace: &Path,
    target: Option<&str>,
    result: &mut serde_json::Value,
) {
    let Ok((cmd, source)) = detect_diagnostics_command(workspace, target) else {
        return;
    };
    let Ok(diag_result) = runner.run(&cmd, workspace, Duration::from_secs(30)) else {
        return;
    };
    let diagnostics = parse_diagnostics(&diag_result.stdout, &diag_result.stderr, source);
    if diagnostics.is_empty() {
        return;
    }
    result["diagnostics"] = json!(diagnostics);
    let msg: Vec<String> = diagnostics
        .iter()
        .map(|d| {
            let file = d.get("file").and_then(|v| v.as_str()).unwrap_or("?");
            let line = d.get("line").and_then(|v| v.as_u64()).unwrap_or(0);
            let text = d.get("message").and_then(|v| v.as_str()).unwrap_or("?");
            format!("  {file}:{line}: {text}")
        })
        .take(10)
        .collect();
    result["diagnostics_message"] = json!(format!(
        "Errors detected after edit — please fix:\n{}",
        msg.join("\n")
    ));
}

fn detect_diagnostics_command(
    workspace: &Path,
    target: Option<&str>,
) -> Result<(String, &'static str)> {
    if workspace.join("Cargo.toml").exists() {
        // cargo check always checks the whole workspace; target is unused for Rust.
        return Ok((
            "cargo check --message-format=json 2>&1".to_string(),
            "rustc",
        ));
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

fn build_seatbelt_profile(workspace: &Path, config: &codingbuddy_core::SandboxConfig) -> String {
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
        // Even when blocking network, allow local binding if configured
        if config.network.allow_local_binding {
            profile.push_str("(allow network-bind (local ip \"localhost:*\"))\n");
        }
        if config.network.allow_unix_sockets {
            profile.push_str("(allow network* (local unix-socket))\n");
        }
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
    config: &codingbuddy_core::SandboxConfig,
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
        if config.network.allow_local_binding || config.network.allow_unix_sockets {
            // When local binding or unix sockets are allowed, we can't fully unshare
            // network. Instead we rely on application-level filtering.
            // bwrap doesn't support fine-grained socket filtering natively.
        } else {
            parts.push("--unshare-net".to_string());
        }
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
    config: &codingbuddy_core::SandboxConfig,
) -> String {
    if !config.enabled {
        return cmd.to_string();
    }
    // Skip sandbox wrapping if already inside a container
    if detect_container_environment().is_some() {
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
        match HookRuntime::run_legacy(&hooks, &context, Duration::from_secs(30)) {
            Ok(runs) => {
                for run in runs {
                    self.emit_hook_event(
                        phase,
                        Path::new(&run.handler_description),
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
            kind: EventKind::HookExecuted {
                phase: phase.to_string(),
                hook_path: hook_path.to_string_lossy().to_string(),
                success,
                timed_out,
                exit_code,
            },
        };
        if let Err(e) = self.store.append_event(&event) {
            eprintln!("[deepseek WARN] hook: failed to emit event: {e}");
        }
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
            kind: EventKind::VisualArtifactCaptured {
                artifact_id: Uuid::now_v7(),
                path: path.to_string(),
                mime: mime.to_string(),
            },
        };
        if let Err(e) = self.store.append_event(&event) {
            eprintln!("[deepseek WARN] visual_artifact: failed to emit event: {e}");
        }
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
            kind: EventKind::CheckpointCreated {
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
mod tests;
