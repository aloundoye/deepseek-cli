use super::utils::*;
use super::*;

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
        if command_guard::is_isolated_sandbox_mode(&self.sandbox_mode)
            && detect_container_environment().is_none()
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
            let wrapped = command_guard::render_wrapper_template(template, &workspace, cmd)?;
            return self.run_cmd(&wrapped, timeout_secs);
        }
        if let Ok(template) = std::env::var("CODINGBUDDY_SANDBOX_WRAPPER")
            && !template.trim().is_empty()
        {
            let wrapped = command_guard::render_wrapper_template(template.trim(), &workspace, cmd)?;
            return self.run_cmd(&wrapped, timeout_secs);
        }
        if let Some(wrapped) = command_guard::auto_isolated_wrapper_command(&workspace, cmd) {
            return self.run_cmd(&wrapped, timeout_secs);
        }
        #[cfg(target_os = "windows")]
        {
            // Windows doesn't have a built-in wrapper path today. Fall back to direct
            // execution with strict logical isolation checks.
            if command_guard::command_references_outside_workspace(cmd, &workspace) {
                return Err(anyhow!(
                    "sandbox_mode={} blocked path outside workspace: {}",
                    self.sandbox_mode,
                    cmd
                ));
            }
            if command_guard::command_has_network_egress_intent(cmd) {
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
                if command_guard::command_has_mutating_intent(cmd) {
                    return Err(anyhow!(
                        "sandbox_mode=read-only blocked mutating command: {}",
                        cmd
                    ));
                }
                if command_guard::command_has_network_egress_intent(cmd) {
                    return Err(anyhow!(
                        "sandbox_mode=read-only blocked network command: {}",
                        cmd
                    ));
                }
            }
            "workspace-write" | "workspace_write" => {
                if command_guard::command_references_outside_workspace(cmd, &self.workspace) {
                    return Err(anyhow!(
                        "sandbox_mode=workspace-write blocked path outside workspace: {}",
                        cmd
                    ));
                }
                if command_guard::command_has_network_egress_intent(cmd) {
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
