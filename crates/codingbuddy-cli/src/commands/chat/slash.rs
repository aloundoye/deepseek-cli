use super::*;

pub(super) fn slash_help_payload() -> serde_json::Value {
    let commands = slash_command_catalog_entries()
        .iter()
        .map(|(name, description)| {
            json!({
                "name": format!("/{name}"),
                "description": description,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "schema": "deepseek.chat.slash_help.v1",
        "commands": commands,
        "tips": [
            "Type / then press Tab to browse command suggestions.",
            "Use /status for current execution state and /tasks mission for queue/subagent details.",
            "Use /todos for session-native checklist and /comment-todos for source comments."
        ]
    })
}

pub(super) fn render_slash_help(payload: &serde_json::Value) -> String {
    let mut lines = vec![
        "Slash commands (Tab after / for autocomplete):".to_string(),
        "Status workflow: /status | /tasks mission | /todos | /plan".to_string(),
        "Execution: /run | /test | /lint | /background".to_string(),
        String::new(),
    ];
    if let Some(commands) = payload
        .get("commands")
        .and_then(serde_json::Value::as_array)
    {
        for command in commands {
            let name = command.get("name").and_then(serde_json::Value::as_str);
            let description = command
                .get("description")
                .and_then(serde_json::Value::as_str);
            if let (Some(name), Some(description)) = (name, description) {
                lines.push(format!("- {name}: {description}"));
            }
        }
    }
    lines.join("\n")
}

pub(super) fn resolve_autocomplete_model_id(model_id_cfg: &str) -> Option<String> {
    if model_id_cfg != "auto" {
        return Some(model_id_cfg.to_string());
    }

    let hw = codingbuddy_local_ml::hardware::detect_hardware();
    let selected = codingbuddy_local_ml::model_registry::recommend_completion_model(
        hw.available_for_models_mb,
    );
    match selected {
        Some(entry) => {
            eprintln!(
                "[codingbuddy] auto-selected model: {} ({:.1}B params, needs {} MB)",
                entry.display_name, entry.params_b, entry.estimated_vram_mb
            );
            Some(entry.model_id.to_string())
        }
        None => {
            eprintln!(
                "[codingbuddy] insufficient RAM ({} MB available) for local autocomplete",
                hw.available_for_models_mb
            );
            None
        }
    }
}

pub(super) fn load_autocomplete_backend(
    model_id: &str,
    cache_dir: &str,
    device_str: &str,
) -> Result<Arc<dyn codingbuddy_local_ml::LocalGenBackend>> {
    #[cfg(feature = "local-ml")]
    {
        let (entry, gguf_filename) =
            match codingbuddy_local_ml::model_registry::find_completion_model(model_id) {
                Some(e) => match e.gguf_filename {
                    Some(name) => (e, name),
                    None => anyhow::bail!(
                        "model '{model_id}' has no GGUF file configured for local completion"
                    ),
                },
                None => anyhow::bail!("model '{model_id}' not found in local registry"),
            };

        let mut mgr = codingbuddy_local_ml::ModelManager::new(PathBuf::from(cache_dir));
        let files = entry.download_files();
        let model_path =
            mgr.ensure_model_with_progress(model_id, entry.hf_repo, &files, |current, total| {
                if current < total {
                    eprintln!(
                        "[codingbuddy] downloading model file {}/{}",
                        current + 1,
                        total
                    );
                }
            })?;

        let hw = codingbuddy_local_ml::hardware::detect_hardware();
        if let Err(msg) = codingbuddy_local_ml::model_registry::check_model_fits(
            model_id,
            hw.available_for_models_mb,
        ) {
            anyhow::bail!(msg);
        }

        let (device, detected) = codingbuddy_local_ml::resolve_device(device_str);
        eprintln!("[codingbuddy] loading {} on {detected}", entry.display_name);
        let backend = codingbuddy_local_ml::CandleCompletion::load(
            &model_path.join(gguf_filename),
            &model_path.join("tokenizer.json"),
            &device,
        )?;
        Ok(Arc::new(backend))
    }

    #[cfg(not(feature = "local-ml"))]
    {
        let _ = (model_id, cache_dir, device_str);
        Ok(Arc::new(codingbuddy_local_ml::MockGenerator::new(
            String::new(),
        )))
    }
}

pub(super) fn resolve_chat_profile_path(cwd: &Path, args: &[String]) -> PathBuf {
    if let Some(first) = args.first() {
        resolve_additional_dir(cwd, first)
    } else {
        runtime_dir(cwd).join("chat_profile.json")
    }
}

pub(super) fn slash_save_profile_output(
    cwd: &Path,
    args: &[String],
    mode: ChatMode,
    read_only: bool,
    thinking_enabled: bool,
    additional_dirs: &[PathBuf],
) -> Result<String> {
    let path = resolve_chat_profile_path(cwd, args);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let payload = json!({
        "schema": "deepseek.chat_profile.v1",
        "mode": chat_mode_name(mode),
        "read_only": read_only,
        "thinking_enabled": thinking_enabled,
        "additional_dirs": additional_dirs.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>(),
    });
    std::fs::write(&path, serde_json::to_vec_pretty(&payload)?)?;
    Ok(format!("saved chat profile: {}", path.display()))
}

pub(super) fn slash_load_profile_output(
    cwd: &Path,
    args: &[String],
) -> Result<(ChatMode, bool, bool, Vec<PathBuf>, String)> {
    let path = resolve_chat_profile_path(cwd, args);
    let raw = std::fs::read_to_string(&path)?;
    let payload: serde_json::Value = serde_json::from_str(&raw)?;
    let mode = payload
        .get("mode")
        .and_then(|v| v.as_str())
        .and_then(parse_chat_mode_name)
        .unwrap_or(ChatMode::Code);
    let read_only = payload
        .get("read_only")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let thinking_enabled = payload
        .get("thinking_enabled")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let additional_dirs = payload
        .get("additional_dirs")
        .and_then(|v| v.as_array())
        .map(|rows| {
            rows.iter()
                .filter_map(|row| row.as_str())
                .map(PathBuf::from)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let output = format!(
        "loaded chat profile: {} mode={} read_only={} thinking={}",
        path.display(),
        chat_mode_name(mode),
        read_only,
        if thinking_enabled { "enabled" } else { "auto" }
    );
    Ok((mode, read_only, thinking_enabled, additional_dirs, output))
}

pub(super) fn slash_git_output(cwd: &Path, args: &[String]) -> Result<String> {
    if args.is_empty() {
        return run_process(cwd, "git", &["status", "--short"]);
    }
    let command = args.iter().map(String::as_str).collect::<Vec<_>>();
    run_process(cwd, "git", &command)
}

pub(super) fn slash_voice_output(args: &[String]) -> Result<String> {
    if args.first().is_some_and(|v| v == "status") {
        return Ok(format!(
            "voice status: ffmpeg={} sox={} arecord={}",
            command_exists("ffmpeg"),
            command_exists("sox"),
            command_exists("arecord")
        ));
    }
    Ok(
        "voice scaffold: not configured for capture in this build. Use text input or `/voice status` for capability checks."
            .to_string(),
    )
}

pub(super) fn scan_watch_comment_payload(cwd: &Path) -> Option<(u64, String)> {
    let output = run_process(
        cwd,
        "rg",
        &[
            "-n",
            "--hidden",
            "--glob",
            "!.git/**",
            "--glob",
            "!target/**",
            "--glob",
            "!.codingbuddy/**",
            "--ignore-file",
            ".codingbuddyignore",
            "TODO\\(ai\\)|FIXME\\(ai\\)|AI:",
            ".",
        ],
    )
    .ok()?;
    let trimmed = output.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut hasher = DefaultHasher::new();
    trimmed.hash(&mut hasher);
    let digest = hasher.finish();
    Some((digest, trimmed.to_string()))
}

pub(super) fn slash_stage_output(cwd: &Path, args: &[String]) -> Result<String> {
    let (all, files) = parse_stage_args(args);
    let summary = git_stage(cwd, all, &files)?;
    Ok(format!(
        "staged changes: staged={} unstaged={} untracked={}",
        summary.staged, summary.unstaged, summary.untracked
    ))
}

pub(super) fn slash_unstage_output(cwd: &Path, args: &[String]) -> Result<String> {
    let (all, files) = parse_stage_args(args);
    let summary = git_unstage(cwd, all, &files)?;
    Ok(format!(
        "unstaged changes: staged={} unstaged={} untracked={}",
        summary.staged, summary.unstaged, summary.untracked
    ))
}

pub(super) fn slash_diff_output(cwd: &Path, args: &[String]) -> Result<String> {
    let (staged, stat) = parse_diff_args(args);
    let output = git_diff_output(cwd, staged, stat)?;
    if output.trim().is_empty() {
        Ok("(no diff)".to_string())
    } else {
        Ok(output)
    }
}

pub(super) fn slash_commit_output(cwd: &Path, cfg: &AppConfig, args: &[String]) -> Result<String> {
    let summary = git_status_summary(cwd)?;
    if summary.staged == 0 {
        return Err(anyhow!("no staged changes to commit. Run /stage first."));
    }
    if let Some(message) = parse_commit_message(args)? {
        let output = git_commit_staged(cwd, cfg, &message)?;
        return Ok(if output.trim().is_empty() {
            format!("committed staged changes: {message}")
        } else {
            output
        });
    }
    git_commit_interactive(cwd, cfg)?;
    Ok("committed staged changes".to_string())
}

pub(super) fn slash_add_dirs(
    cwd: &Path,
    dirs: &mut Vec<PathBuf>,
    args: &[String],
) -> Result<String> {
    if args.is_empty() {
        return Err(anyhow!("usage: /add <path> [path ...]"));
    }
    let mut added = Vec::new();
    for raw in args {
        let path = resolve_additional_dir(cwd, raw);
        if path.exists() && path.is_dir() && !dirs.contains(&path) {
            dirs.push(path.clone());
            added.push(path);
        }
    }
    if added.is_empty() {
        return Ok("no new directories added".to_string());
    }
    Ok(format!(
        "added: {}",
        added
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

pub(super) fn slash_drop_dirs(
    cwd: &Path,
    dirs: &mut Vec<PathBuf>,
    args: &[String],
) -> Result<String> {
    if args.is_empty() {
        return Err(anyhow!("usage: /drop <path> [path ...]"));
    }
    let mut removed = Vec::new();
    for raw in args {
        let target = resolve_additional_dir(cwd, raw);
        if let Some(pos) = dirs.iter().position(|dir| dir == &target) {
            removed.push(dirs.remove(pos));
        }
    }
    if removed.is_empty() {
        return Ok("no matching directories were active".to_string());
    }
    Ok(format!(
        "dropped: {}",
        removed
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

pub(super) fn local_tool_output(cwd: &Path, call: ToolCall) -> Result<serde_json::Value> {
    let cfg = AppConfig::load(cwd).unwrap_or_default();
    let policy = PolicyEngine::from_app_config(&cfg.policy);
    let tool_host = LocalToolHost::new(cwd, policy)?;
    let proposal = tool_host.propose(call);
    let result = tool_host.execute(ApprovedToolCall {
        invocation_id: proposal.invocation_id,
        call: proposal.call,
    });
    if !result.success {
        return Err(anyhow!(
            "{}",
            result
                .output
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("tool execution failed")
        ));
    }
    Ok(result.output)
}

pub(super) fn slash_run_output(cwd: &Path, args: &[String]) -> Result<String> {
    if args.is_empty() {
        return Err(anyhow!("usage: /run <command>"));
    }
    let command = args.join(" ");
    let output = local_tool_output(
        cwd,
        ToolCall {
            name: "bash.run".to_string(),
            args: json!({
                "cmd": command,
                "timeout": 120,
            }),
            requires_approval: false,
        },
    )?;
    let stdout = output
        .get("stdout")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let stderr = output
        .get("stderr")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let mut out = String::new();
    if !stdout.trim().is_empty() {
        out.push_str(stdout.trim());
    }
    if !stderr.trim().is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(stderr.trim());
    }
    if out.is_empty() {
        Ok("command completed with no output".to_string())
    } else {
        Ok(out)
    }
}

pub(super) fn inferred_test_command(cwd: &Path) -> Option<String> {
    if cwd.join("Cargo.toml").exists() {
        return Some("cargo test -q".to_string());
    }
    if cwd.join("package.json").exists() {
        return Some("npm test -- --runInBand".to_string());
    }
    if cwd.join("pyproject.toml").exists() || cwd.join("requirements.txt").exists() {
        return Some("pytest -q".to_string());
    }
    None
}

pub(super) fn inferred_lint_command(cwd: &Path) -> Option<String> {
    if cwd.join("Cargo.toml").exists() {
        return Some("cargo clippy --all-targets -- -D warnings".to_string());
    }
    if cwd.join("package.json").exists() {
        return Some("npm run lint".to_string());
    }
    if cwd.join("pyproject.toml").exists() {
        return Some("ruff check .".to_string());
    }
    None
}

pub(super) fn slash_test_output(cwd: &Path, args: &[String]) -> Result<String> {
    let command = if args.is_empty() {
        inferred_test_command(cwd).ok_or_else(|| {
            anyhow!("could not infer a test command for this repo; pass one: /test <command>")
        })?
    } else {
        args.join(" ")
    };
    slash_run_output(cwd, &[command])
}

pub(super) fn slash_lint_output(
    cwd: &Path,
    args: &[String],
    cfg: Option<&AppConfig>,
) -> Result<String> {
    if !args.is_empty() {
        return slash_run_output(cwd, &[args.join(" ")]);
    }

    // Try configured lint commands from AppConfig.agent_loop.lint
    if let Some(cfg) = cfg {
        let lint_cfg = &cfg.agent_loop.lint;
        if lint_cfg.enabled && !lint_cfg.commands.is_empty() {
            let changed = changed_file_list(cwd);
            // Derive matching lint commands for the changed file extensions
            let mut commands = Vec::new();
            for file in &changed {
                let ext = file.rsplit('.').next().unwrap_or("");
                let lang = match ext {
                    "rs" => "rust",
                    "py" => "python",
                    "js" | "jsx" => "javascript",
                    "ts" | "tsx" => "typescript",
                    "go" => "go",
                    "java" => "java",
                    _ => ext,
                };
                if let Some(cmd) = lint_cfg.commands.get(lang)
                    && !commands.contains(cmd)
                {
                    commands.push(cmd.clone());
                }
            }
            if !commands.is_empty() {
                let mut results = Vec::new();
                for cmd in &commands {
                    let output = slash_run_output(cwd, std::slice::from_ref(cmd))?;
                    results.push(format!("$ {cmd}\n{output}"));
                }
                return Ok(results.join("\n\n"));
            }
        }
    }

    // Fallback to inferred command
    let command = inferred_lint_command(cwd).ok_or_else(|| {
        anyhow!("could not infer a lint command for this repo; pass one: /lint <command>")
    })?;
    slash_run_output(cwd, &[command])
}

/// List changed file paths (staged + unstaged) for lint command derivation.
pub(super) fn changed_file_list(cwd: &Path) -> Vec<String> {
    let Ok(output) = run_process(cwd, "git", &["status", "--porcelain"]) else {
        return Vec::new();
    };
    output
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.len() < 3 {
                return None;
            }
            Some(trimmed[3..].trim().to_string())
        })
        .collect()
}

pub(super) fn slash_tokens_output(cwd: &Path, cfg: &AppConfig) -> Result<String> {
    let context_window = cfg.llm.context_window_tokens;

    // System prompt estimate (model instructions, tool definitions, etc.)
    let system_prompt_est: u64 = 800;

    // Memory (CODINGBUDDY.md)
    let memory_tokens = {
        let md_path = cwd.join("CODINGBUDDY.md");
        if md_path.exists() {
            let content = std::fs::read_to_string(&md_path).unwrap_or_default();
            estimate_tokens(&content)
        } else {
            0
        }
    };

    // Conversation history from store
    let conversation_tokens: u64 = {
        let store = codingbuddy_store::Store::new(cwd).ok();
        store
            .and_then(|s| {
                let session = s.load_latest_session().ok()??;
                let projection = s.rebuild_from_events(session.session_id).ok()?;
                let chars: u64 = projection
                    .transcript
                    .iter()
                    .map(|t: &String| t.len() as u64)
                    .sum();
                Some(chars / 4)
            })
            .unwrap_or(0)
    };

    // Repo map estimate
    let repo_map_tokens: u64 = 400; // typical bootstrap overhead

    let total_used = system_prompt_est + memory_tokens + conversation_tokens + repo_map_tokens;
    let reserved = cfg.context.reserved_overhead_tokens + cfg.context.response_budget_tokens;
    let available = context_window
        .saturating_sub(total_used)
        .saturating_sub(reserved);
    let utilization = if context_window > 0 {
        (total_used as f64 / context_window as f64) * 100.0
    } else {
        0.0
    };

    // Cost estimate (based on tokens used so far)
    let input_cost = (total_used as f64 / 1_000_000.0) * cfg.usage.cost_per_million_input;

    Ok(format!(
        "token breakdown:\n  system_prompt:  ~{system_prompt_est}\n  memory:         ~{memory_tokens}\n  conversation:   ~{conversation_tokens}\n  repo_map:       ~{repo_map_tokens}\n  ────────────────────\n  total_used:     ~{total_used}\n  reserved:       ~{reserved} (overhead={} response={})\n  available:      ~{available}\n  context_window: {context_window}\n  utilization:    {utilization:.1}%\n  est_input_cost: ${input_cost:.4}",
        cfg.context.reserved_overhead_tokens, cfg.context.response_budget_tokens,
    ))
}

pub(super) fn slash_web_output(cwd: &Path, args: &[String]) -> Result<String> {
    if args.is_empty() {
        return Err(anyhow!("usage: /web <query|url>"));
    }

    let first = args.first().map(String::as_str).unwrap_or_default();
    if first.starts_with("http://") || first.starts_with("https://") {
        let output = local_tool_output(
            cwd,
            ToolCall {
                name: "web.fetch".to_string(),
                args: json!({
                    "url": first,
                    "max_bytes": 120000,
                    "timeout": 20,
                }),
                requires_approval: false,
            },
        )?;
        return Ok(render_web_fetch_markdown(first, &output, 140));
    }

    let query = args.join(" ");
    let output = local_tool_output(
        cwd,
        ToolCall {
            name: "web.search".to_string(),
            args: json!({
                "query": query,
                "max_results": 5,
            }),
            requires_approval: false,
        },
    )?;
    let results = output
        .get("results")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut top_extract = None;
    for row in results.iter().take(3) {
        let url = row.get("url").and_then(|v| v.as_str()).unwrap_or_default();
        if !url.starts_with("http://") && !url.starts_with("https://") {
            continue;
        }
        if let Ok(fetch) = local_tool_output(
            cwd,
            ToolCall {
                name: "web.fetch".to_string(),
                args: json!({
                    "url": url,
                    "max_bytes": 64000,
                    "timeout": 18,
                }),
                requires_approval: false,
            },
        ) {
            let preview = fetch_preview_text(&fetch, 28);
            if !preview.is_empty() {
                top_extract = Some((url.to_string(), preview));
                break;
            }
        }
    }

    Ok(render_web_search_markdown(&query, &results, top_extract))
}

pub(super) fn slash_map_output(
    cwd: &Path,
    args: &[String],
    additional_dirs: &[PathBuf],
) -> Result<String> {
    let query = if args.is_empty() {
        "project map".to_string()
    } else {
        args.join(" ")
    };
    let mut manager = ContextManager::new(cwd)?;
    manager.analyze_workspace()?;
    let suggestions = manager.suggest_relevant_files(&query, 20);
    let mut lines = vec![format!(
        "repo map: indexed_files={} query=\"{}\"",
        manager.file_count(),
        query
    )];
    if suggestions.is_empty() {
        lines.push("(no scored files)".to_string());
    } else {
        for item in suggestions {
            lines.push(format!(
                "- {} score={:.2} {}",
                item.path.strip_prefix(cwd).unwrap_or(&item.path).display(),
                item.score,
                truncate_inline(&item.reasons.join(" | "), 120)
            ));
        }
    }
    if !additional_dirs.is_empty() {
        lines.push("additional dirs:".to_string());
        for dir in additional_dirs {
            lines.push(format!("- {}", dir.display()));
        }
    }
    Ok(lines.join("\n"))
}
