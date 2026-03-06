use super::*;

pub(crate) struct ChatTuiArgs<'a> {
    pub(crate) cwd: &'a Path,
    pub(crate) allow_tools: bool,
    pub(crate) cfg: &'a AppConfig,
    pub(crate) initial_force_max_think: bool,
    pub(crate) teammate_mode: Option<String>,
    pub(crate) repo_root_override: Option<PathBuf>,
    pub(crate) debug_context: bool,
    pub(crate) detect_urls: bool,
    pub(crate) watch_files_enabled: bool,
    pub(crate) initial_session_id: Option<Uuid>,
}

pub(crate) fn run_chat_tui(args: ChatTuiArgs<'_>) -> Result<()> {
    let ChatTuiArgs {
        cwd,
        allow_tools,
        cfg,
        initial_force_max_think,
        teammate_mode,
        repo_root_override,
        debug_context,
        detect_urls,
        watch_files_enabled,
        initial_session_id,
    } = args;
    let debug = std::env::var("CODINGBUDDY_DEBUG").is_ok();
    if debug {
        eprintln!("[debug] creating agent engine...");
    }
    let engine = Arc::new(AgentEngine::new(cwd)?);
    wire_subagent_worker(&engine, cwd);
    if debug {
        eprintln!("[debug] agent engine ready");
    }
    let force_max_think = Arc::new(AtomicBool::new(initial_force_max_think));
    let additional_dirs = Arc::new(std::sync::Mutex::new(Vec::<PathBuf>::new()));
    let read_only_mode = Arc::new(AtomicBool::new(false));
    let active_chat_mode = Arc::new(std::sync::Mutex::new(ChatMode::Code));
    let last_watch_digest = Arc::new(std::sync::Mutex::new(None::<u64>));
    let active_session_id = Arc::new(std::sync::Mutex::new(initial_session_id));
    let pending_images = Arc::new(std::sync::Mutex::new(
        Vec::<codingbuddy_core::ImageContent>::new(),
    ));

    // Create the channel for TUI stream events.
    let (tx, rx) = mpsc::channel::<TuiStreamEvent>();

    // Set approval handler that routes through the TUI channel.
    {
        let approval_tx = tx.clone();
        engine.set_approval_handler(Box::new(move |call| {
            let (resp_tx, resp_rx) = mpsc::channel();
            let compact_args = serde_json::to_string(&call.args)
                .unwrap_or_else(|_| "<unserializable>".to_string());
            let _ = approval_tx.send(TuiStreamEvent::ApprovalNeeded {
                tool_name: call.name.clone(),
                args_summary: compact_args,
                response_tx: resp_tx,
            });
            // Block agent thread waiting for TUI user response.
            resp_rx
                .recv()
                .map_err(|e| anyhow!("approval channel closed: {e}"))
        }));
    }

    let status = current_ui_status(
        cwd,
        cfg,
        force_max_think.load(Ordering::Relaxed),
        initial_session_id,
    )?;
    let bindings = load_tui_keybindings(cwd, cfg);
    let theme = TuiTheme::from_config(&cfg.theme.primary, &cfg.theme.secondary, &cfg.theme.error);
    let fmt_refresh = Arc::clone(&force_max_think);
    let additional_dirs_for_closure = Arc::clone(&additional_dirs);
    let read_only_for_closure = Arc::clone(&read_only_mode);
    let active_mode_for_closure = Arc::clone(&active_chat_mode);
    let watch_digest_for_closure = Arc::clone(&last_watch_digest);
    let active_session_for_closure = Arc::clone(&active_session_id);

    {
        let tx_notices = tx.clone();
        let cwd_for_notices = cwd.to_path_buf();
        let active_session_for_notices = Arc::clone(&active_session_id);
        thread::spawn(move || {
            let mut watermarks = HashMap::<Uuid, u64>::new();
            loop {
                let session_override = active_session_for_notices
                    .lock()
                    .map(|guard| *guard)
                    .unwrap_or(None);
                if let Ok(notices) = poll_session_lifecycle_notices(
                    &cwd_for_notices,
                    session_override,
                    &mut watermarks,
                ) {
                    for notice in notices {
                        if tx_notices
                            .send(TuiStreamEvent::SystemNotice {
                                line: notice.line,
                                error: notice.is_error,
                            })
                            .is_err()
                        {
                            return;
                        }
                    }
                }
                thread::sleep(Duration::from_millis(400));
            }
        });
    }

    // Build ML completion callback for ghost text (if local_ml + autocomplete enabled).
    // Runners are loaded lazily and reused via a lifecycle manager with queueing/eviction.
    let ml_completion_cb: Option<codingbuddy_ui::MlCompletionCallback> = if cfg.local_ml.enabled
        && cfg.local_ml.autocomplete.enabled
    {
        if let Some(resolved_model_id) =
            resolve_autocomplete_model_id(&cfg.local_ml.autocomplete.model_id)
        {
            let cache_dir = cfg.local_ml.cache_dir.clone();
            let device_str = cfg.local_ml.device.clone();
            let runtime_manager =
                codingbuddy_local_ml::ModelManager::new(PathBuf::from(&cache_dir));
            let scheduler = Arc::new(codingbuddy_local_ml::LocalRunnerLifecycleManager::new(
                runtime_manager,
                Arc::new({
                    let cache_dir = cache_dir.clone();
                    let device_str = device_str.clone();
                    move |model_id: &str| -> Result<Arc<dyn codingbuddy_local_ml::LocalGenBackend>> {
                        load_autocomplete_backend(model_id, &cache_dir, &device_str).or_else(
                            |error| {
                                eprintln!(
                                    "[codingbuddy] local completion backend load failed for '{model_id}' ({error}), using mock backend"
                                );
                                Ok(Arc::new(codingbuddy_local_ml::MockGenerator::new(
                                    String::new(),
                                ))
                                    as Arc<dyn codingbuddy_local_ml::LocalGenBackend>)
                            },
                        )
                    }
                }),
            ));

            // Prewarm the chosen autocomplete model in the background.
            {
                let scheduler = Arc::clone(&scheduler);
                let model_id = resolved_model_id.clone();
                std::thread::spawn(move || {
                    if let Err(error) = scheduler.prewarm(&model_id) {
                        eprintln!(
                            "[codingbuddy] failed to prewarm local autocomplete model '{model_id}': {error}"
                        );
                    }
                });
            }

            let max_tokens = cfg.local_ml.autocomplete.max_tokens;
            let timeout_ms = cfg.local_ml.autocomplete.timeout_ms;
            Some(Arc::new(move |input: &str| -> Option<String> {
                let opts = codingbuddy_local_ml::completion::GenOpts {
                    max_tokens,
                    timeout_ms,
                    ..Default::default()
                };
                scheduler
                    .generate(&resolved_model_id, input, &opts)
                    .ok()
                    .filter(|s| !s.is_empty())
            }))
        } else {
            None
        }
    } else {
        None
    };

    if debug {
        eprintln!("[debug] starting TUI shell...");
    }
    run_tui_shell_with_bindings(
        status,
        bindings,
        theme,
        cfg.ui.reduced_motion,
        rx,
        |prompt| {
            // Handle slash commands synchronously, sending result via channel.
            if let Some(cmd) = SlashCommand::parse(prompt) {
                let result: Result<String> = (|| {
                    let out = match cmd {
                        SlashCommand::Help => render_slash_help(&slash_help_payload()),
                        SlashCommand::Ask(_) => {
                            if let Ok(mut guard) = active_mode_for_closure.lock() {
                                *guard = ChatMode::Ask;
                            }
                            "chat mode set to ask".to_string()
                        }
                        SlashCommand::Code(_) => {
                            if let Ok(mut guard) = active_mode_for_closure.lock() {
                                *guard = ChatMode::Code;
                            }
                            "chat mode set to code".to_string()
                        }
                        SlashCommand::ChatMode(mode) => {
                            if let Some(raw_mode) = mode {
                                if let Some(parsed) = parse_chat_mode_name(&raw_mode) {
                                    if let Ok(mut guard) = active_mode_for_closure.lock() {
                                        *guard = parsed;
                                    }
                                    format!("chat mode set to {}", chat_mode_name(parsed))
                                } else {
                                    format!("unsupported chat mode: {raw_mode} (ask|code|context)")
                                }
                            } else if let Ok(guard) = active_mode_for_closure.lock() {
                                format!("current chat mode: {}", chat_mode_name(*guard))
                            } else {
                                "current chat mode: code".to_string()
                            }
                        }
                        SlashCommand::Init => {
                            let manager = MemoryManager::new(cwd)?;
                            let path = manager.ensure_initialized()?;
                            format!("initialized memory at {}", path.display())
                        }
                        SlashCommand::Clear => "cleared".to_string(),
                        SlashCommand::Compact(focus) => {
                            let summary = compact_now(cwd, None, focus.as_deref())?;
                            format!(
                                "compacted turns {}..{} summary_id={} token_delta={}",
                                summary.from_turn,
                                summary.to_turn,
                                summary.summary_id,
                                summary.token_delta_estimate
                            )
                        }
                        SlashCommand::Memory(args) => {
                            if args.is_empty() || args[0].eq_ignore_ascii_case("show") {
                                // Non-interactive: default to show (no editor available)
                                MemoryManager::new(cwd)?.read_memory()?
                            } else if args[0].eq_ignore_ascii_case("edit") {
                                let manager = MemoryManager::new(cwd)?;
                                let path = manager.ensure_initialized()?;
                                let checkpoint = manager.create_checkpoint("memory_edit")?;
                                append_control_event(
                                    cwd,
                                    EventKind::CheckpointCreated {
                                        checkpoint_id: checkpoint.checkpoint_id,
                                        reason: checkpoint.reason.clone(),
                                        files_count: checkpoint.files_count,
                                        snapshot_path: checkpoint.snapshot_path.clone(),
                                    },
                                )?;
                                let editor = std::env::var("EDITOR")
                                    .unwrap_or_else(|_| default_editor().to_string());
                                let status =
                                    std::process::Command::new(editor).arg(&path).status()?;
                                if !status.success() {
                                    return Err(anyhow!("editor exited with non-zero status"));
                                }
                                let version_id = manager.sync_memory_version("edit")?;
                                append_control_event(
                                    cwd,
                                    EventKind::MemorySynced {
                                        version_id,
                                        path: path.to_string_lossy().to_string(),
                                        note: "edit".to_string(),
                                    },
                                )?;
                                format!("memory edited at {}", path.display())
                            } else if args[0].eq_ignore_ascii_case("sync") {
                                let note = args
                                    .get(1)
                                    .cloned()
                                    .unwrap_or_else(|| "tui-sync".to_string());
                                let manager = MemoryManager::new(cwd)?;
                                let version_id = manager.sync_memory_version(&note)?;
                                append_control_event(
                                    cwd,
                                    EventKind::MemorySynced {
                                        version_id,
                                        path: manager.memory_path().to_string_lossy().to_string(),
                                        note,
                                    },
                                )?;
                                format!("memory synced: {version_id}")
                            } else {
                                "unknown /memory subcommand".to_string()
                            }
                        }
                        SlashCommand::Config => format!(
                            "config file: {}",
                            AppConfig::project_settings_path(cwd).display()
                        ),
                        SlashCommand::Model(model) => {
                            if let Some(model) = model {
                                force_max_think
                                    .store(is_max_think_selection(&model), Ordering::Relaxed);
                            }
                            if force_max_think.load(Ordering::Relaxed) {
                                format!(
                                    "model mode: thinking-enabled ({})",
                                    cfg.llm.active_reasoner_model()
                                )
                            } else {
                                format!(
                                    "model mode: auto ({} thinking=on-demand)",
                                    cfg.llm.active_base_model()
                                )
                            }
                        }
                        SlashCommand::Provider(provider) => format_provider_info(cfg, provider),
                        SlashCommand::Cost => {
                            let store = Store::new(cwd)?;
                            let usage = store.usage_summary(None, Some(24))?;
                            format!(
                                "24h usage input={} output={}",
                                usage.input_tokens, usage.output_tokens
                            )
                        }
                        SlashCommand::Tokens => slash_tokens_output(cwd, cfg)?,
                        SlashCommand::Mcp(args) => {
                            let manager = McpManager::new(cwd)?;
                            if args.is_empty() || args[0].eq_ignore_ascii_case("list") {
                                let servers = manager.list_servers()?;
                                format!("mcp servers: {}", servers.len())
                            } else if args[0].eq_ignore_ascii_case("get") && args.len() > 1 {
                                match manager.get_server(&args[1])? {
                                    Some(server) => format!(
                                        "mcp {} transport={:?} enabled={}",
                                        server.id, server.transport, server.enabled
                                    ),
                                    None => format!("mcp server not found: {}", args[1]),
                                }
                            } else if args[0].eq_ignore_ascii_case("remove") && args.len() > 1 {
                                let removed = manager.remove_server(&args[1])?;
                                format!("mcp remove {} -> {}", args[1], removed)
                            } else {
                                "use /mcp list|get <id>|remove <id>".to_string()
                            }
                        }
                        SlashCommand::Rewind(args) => {
                            if args.is_empty() {
                                // Show rewind picker (checkpoint list + action menu).
                                let mem = MemoryManager::new(cwd)?;
                                let checkpoints = mem.list_checkpoints().unwrap_or_default();
                                if checkpoints.is_empty() {
                                    "no checkpoints available".to_string()
                                } else {
                                    let picker =
                                        codingbuddy_ui::RewindPickerState::new(checkpoints);
                                    // Display as numbered list for user to choose
                                    let mut lines: Vec<String> =
                                        vec!["Rewind checkpoints:".to_string()];
                                    for (i, cp) in picker.checkpoints.iter().enumerate() {
                                        lines.push(format!(
                                            "  {}. [{}] {} ({} files)",
                                            i + 1,
                                            cp.created_at,
                                            cp.reason,
                                            cp.files_count
                                        ));
                                    }
                                    lines.push(String::new());
                                    lines.push(
                                        "Use /rewind <number> to rewind to a checkpoint."
                                            .to_string(),
                                    );
                                    lines.join("\n")
                                }
                            } else {
                                let to_checkpoint = args.first().cloned();
                                let checkpoint = rewind_now(cwd, to_checkpoint)?;
                                format!("rewound to checkpoint {}", checkpoint.checkpoint_id)
                            }
                        }
                        SlashCommand::Export(_) => {
                            let record = MemoryManager::new(cwd)?.export_transcript(
                                ExportFormat::Json,
                                None,
                                None,
                            )?;
                            format!("exported transcript {}", record.output_path)
                        }
                        SlashCommand::Plan(args) => {
                            if args.is_empty() {
                                let store = Store::new(cwd)?;
                                let session_override = active_session_for_closure
                                    .lock()
                                    .map_err(|_| anyhow!("failed to access active session state"))?
                                    .to_owned();
                                let session = if let Some(session_id) = session_override {
                                    store.load_session(session_id)?
                                } else {
                                    store.load_latest_session()?
                                };
                                if current_plan_payload(&store, session.as_ref())?.is_some() {
                                    let response = handle_plan_slash(
                                        cwd,
                                        &["show".to_string()],
                                        session_override,
                                    )?;
                                    if let Some(session_id) = response.session_switch
                                        && let Ok(mut guard) = active_session_for_closure.lock()
                                    {
                                        *guard = Some(session_id);
                                    }
                                    response.text
                                } else {
                                    force_max_think.store(true, Ordering::Relaxed);
                                    "plan mode enabled (thinking enabled). Use /plan show|approve|reject <feedback> for the persisted review flow.".to_string()
                                }
                            } else {
                                let session_override = active_session_for_closure
                                    .lock()
                                    .map_err(|_| anyhow!("failed to access active session state"))?
                                    .to_owned();
                                let response = handle_plan_slash(cwd, &args, session_override)?;
                                if let Some(session_id) = response.session_switch
                                    && let Ok(mut guard) = active_session_for_closure.lock()
                                {
                                    *guard = Some(session_id);
                                }
                                response.text
                            }
                        }
                        SlashCommand::Add(args) => {
                            let mut guard = additional_dirs_for_closure
                                .lock()
                                .map_err(|_| anyhow!("failed to access additional dir state"))?;
                            slash_add_dirs(cwd, &mut guard, &args)?
                        }
                        SlashCommand::Drop(args) => {
                            let mut guard = additional_dirs_for_closure
                                .lock()
                                .map_err(|_| anyhow!("failed to access additional dir state"))?;
                            slash_drop_dirs(cwd, &mut guard, &args)?
                        }
                        SlashCommand::ReadOnly(args) => {
                            let action = args
                                .first()
                                .map(|v| v.to_ascii_lowercase())
                                .unwrap_or_else(|| "toggle".to_string());
                            match action.as_str() {
                                "on" | "true" | "1" => {
                                    read_only_for_closure.store(true, Ordering::Relaxed);
                                }
                                "off" | "false" | "0" => {
                                    read_only_for_closure.store(false, Ordering::Relaxed);
                                }
                                "status" => {}
                                _ => {
                                    let next = !read_only_for_closure.load(Ordering::Relaxed);
                                    read_only_for_closure.store(next, Ordering::Relaxed);
                                }
                            }
                            format!(
                                "read-only mode: {}",
                                if read_only_for_closure.load(Ordering::Relaxed) {
                                    "enabled"
                                } else {
                                    "disabled"
                                }
                            )
                        }
                        SlashCommand::Map(args) => {
                            let guard = additional_dirs_for_closure
                                .lock()
                                .map_err(|_| anyhow!("failed to access additional dir state"))?;
                            slash_map_output(cwd, &args, &guard)?
                        }
                        SlashCommand::MapRefresh(args) => {
                            codingbuddy_agent::clear_tag_cache();
                            let guard = additional_dirs_for_closure
                                .lock()
                                .map_err(|_| anyhow!("failed to access additional dir state"))?;
                            format!("(cache cleared)\n{}", slash_map_output(cwd, &args, &guard)?)
                        }
                        SlashCommand::Run(args) => slash_run_output(cwd, &args)?,
                        SlashCommand::Test(args) => slash_test_output(cwd, &args)?,
                        SlashCommand::Lint(args) => slash_lint_output(cwd, &args, Some(cfg))?,
                        SlashCommand::Web(args) => slash_web_output(cwd, &args)?,
                        SlashCommand::Diff(args) => slash_diff_output(cwd, &args)?,
                        SlashCommand::Stage(args) => slash_stage_output(cwd, &args)?,
                        SlashCommand::Unstage(args) => slash_unstage_output(cwd, &args)?,
                        SlashCommand::Commit(args) => {
                            if parse_commit_message(&args)?.is_none() {
                                "commit requires a message in TUI mode: /commit -m \"your message\""
                                    .to_string()
                            } else {
                                slash_commit_output(cwd, cfg, &args)?
                            }
                        }
                        SlashCommand::Undo(_args) => {
                            let checkpoint_msg =
                                match crate::commands::compact::rewind_now(cwd, None) {
                                    Ok(checkpoint) => format!(
                                        "rewound to checkpoint {} (files={})",
                                        checkpoint.checkpoint_id, checkpoint.files_count
                                    ),
                                    Err(e) => format!("no files rewound ({e})"),
                                };
                            crate::context::append_control_event(
                                cwd,
                                codingbuddy_core::EventKind::TurnReverted { turns_dropped: 1 },
                            )?;
                            format!("Reverted 1 conversation turn. {checkpoint_msg}")
                        }
                        SlashCommand::Status => {
                            let session_override = active_session_for_closure
                                .lock()
                                .map(|guard| *guard)
                                .unwrap_or(None);
                            let status = current_ui_status(
                                cwd,
                                cfg,
                                force_max_think.load(Ordering::Relaxed),
                                session_override,
                            )?;
                            render_statusline(&status)
                        }
                        SlashCommand::Effort(level) => {
                            let level = level.unwrap_or_else(|| "medium".to_string());
                            let normalized = level.to_ascii_lowercase();
                            force_max_think.store(
                                matches!(normalized.as_str(), "high" | "max"),
                                Ordering::Relaxed,
                            );
                            format!(
                                "effort={} thinking={}",
                                normalized,
                                if force_max_think.load(Ordering::Relaxed) {
                                    "enabled"
                                } else {
                                    "auto"
                                }
                            )
                        }
                        SlashCommand::Skills(args) => {
                            let manager = SkillManager::new(cwd)?;
                            let paths = cfg
                                .skills
                                .paths
                                .iter()
                                .map(|path| expand_tilde(path))
                                .collect::<Vec<_>>();
                            if args.is_empty() || args[0].eq_ignore_ascii_case("list") {
                                let skills = manager.list(&paths)?;
                                if skills.is_empty() {
                                    "no skills found".to_string()
                                } else {
                                    skills
                                        .into_iter()
                                        .map(|skill| format!("{} - {}", skill.id, skill.summary))
                                        .collect::<Vec<_>>()
                                        .join("\n")
                                }
                            } else if args[0].eq_ignore_ascii_case("reload") {
                                let loaded = manager.reload(&paths)?;
                                format!("reloaded {} skills", loaded.len())
                            } else if args[0].eq_ignore_ascii_case("run") && args.len() > 1 {
                                let input = if args.len() > 2 {
                                    Some(args[2..].join(" "))
                                } else {
                                    None
                                };
                                let rendered = manager.run(&args[1], input.as_deref(), &paths)?;
                                format!("{}\n{}", rendered.skill_id, rendered.rendered_prompt)
                            } else {
                                "use /skills list|reload|run <id> [input]".to_string()
                            }
                        }
                        SlashCommand::Permissions(args) => match parse_permissions_cmd(args) {
                            Ok(cmd) => {
                                serde_json::to_string_pretty(&permissions_payload(cwd, cmd)?)?
                            }
                            Err(err) => format!("permissions parse error: {err}"),
                        },
                        SlashCommand::Background(args) => match parse_background_cmd(args) {
                            Ok(cmd) => {
                                serde_json::to_string_pretty(&background_payload(cwd, cmd)?)?
                            }
                            Err(err) => format!("background parse error: {err}"),
                        },
                        SlashCommand::Git(args) => slash_git_output(cwd, &args)?,
                        SlashCommand::Settings => format!(
                            "config file: {}",
                            AppConfig::project_settings_path(cwd).display()
                        ),
                        SlashCommand::Load(args) => {
                            let (mode, read_only, thinking, dirs, output) =
                                slash_load_profile_output(cwd, &args)?;
                            if let Ok(mut guard) = active_mode_for_closure.lock() {
                                *guard = mode;
                            }
                            read_only_for_closure.store(read_only, Ordering::Relaxed);
                            force_max_think.store(thinking, Ordering::Relaxed);
                            if let Ok(mut guard) = additional_dirs_for_closure.lock() {
                                *guard = dirs;
                            }
                            output
                        }
                        SlashCommand::Save(args) => {
                            let mode = active_mode_for_closure
                                .lock()
                                .map(|g| *g)
                                .unwrap_or(ChatMode::Code);
                            let dirs = additional_dirs_for_closure
                                .lock()
                                .map(|d| d.clone())
                                .unwrap_or_default();
                            slash_save_profile_output(
                                cwd,
                                &args,
                                mode,
                                read_only_for_closure.load(Ordering::Relaxed),
                                force_max_think.load(Ordering::Relaxed),
                                &dirs,
                            )?
                        }
                        SlashCommand::Context => {
                            let ctx_cfg = AppConfig::load(cwd).unwrap_or_default();
                            let ctx_store = Store::new(cwd)?;
                            let context_window = ctx_cfg.llm.context_window_tokens;
                            let compact_threshold = ctx_cfg.context.auto_compact_threshold;
                            let session = ctx_store.load_latest_session()?;
                            let (session_tokens, compactions) = if let Some(ref s) = session {
                                let usage = ctx_store.usage_summary(Some(s.session_id), None)?;
                                let compactions =
                                    ctx_store.list_context_compactions(Some(s.session_id))?;
                                (usage.input_tokens + usage.output_tokens, compactions.len())
                            } else {
                                (0, 0)
                            };
                            let memory_tokens = {
                                let mem = codingbuddy_memory::MemoryManager::new(cwd).ok();
                                let text = mem
                                    .and_then(|m| m.read_combined_memory().ok())
                                    .unwrap_or_default();
                                (text.len() as u64) / 4
                            };
                            let system_prompt_tokens: u64 =
                                800 + ctx_cfg.policy.allowlist.len() as u64 * 40 + 400;
                            let conversation_tokens =
                                session_tokens.saturating_sub(system_prompt_tokens + memory_tokens);
                            let utilization = if context_window > 0 {
                                (session_tokens as f64 / context_window as f64) * 100.0
                            } else {
                                0.0
                            };
                            let mut out = format!(
                                "Context Window Inspector\n========================\nWindow size:       {} tokens\nCompact threshold: {:.0}%\nSession tokens:    {}\nUtilization:       {:.1}%\nCompactions:       {}\n\nBreakdown:\n  System prompt:        ~{} tokens\n  Memory (CODINGBUDDY.md): ~{} tokens\n  Conversation:         ~{} tokens",
                                context_window,
                                compact_threshold * 100.0,
                                session_tokens,
                                utilization,
                                compactions,
                                system_prompt_tokens,
                                memory_tokens,
                                conversation_tokens,
                            );
                            if utilization > (compact_threshold as f64 * 100.0) {
                                out.push_str("\n\nContext is above compact threshold. Use /compact to free space.");
                            }
                            out
                        }
                        SlashCommand::Sandbox(_) => format!(
                            "Sandbox mode: {}",
                            AppConfig::load(cwd).unwrap_or_default().policy.sandbox_mode
                        ),
                        SlashCommand::Agents => {
                            let payload = agents_payload(cwd, 20)?;
                            render_agents_payload(&payload)
                        }
                        SlashCommand::Tasks(args) => {
                            let session_override = active_session_for_closure
                                .lock()
                                .map(|guard| *guard)
                                .unwrap_or(None);
                            let response = handle_tasks_slash(cwd, &args, session_override)?;
                            if let Some(session_id) = response.session_switch
                                && let Ok(mut guard) = active_session_for_closure.lock()
                            {
                                *guard = Some(session_id);
                            }
                            response.text
                        }
                        SlashCommand::Review(_) => {
                            "Use 'deepseek review' subcommand for code review.".to_string()
                        }
                        SlashCommand::Search(args) => {
                            let query = args.join(" ");
                            if query.is_empty() {
                                "Usage: /search <query>".to_string()
                            } else {
                                format!("Search '{}': use 'deepseek search' subcommand.", query)
                            }
                        }
                        SlashCommand::Vim(args) => {
                            if args.is_empty() {
                                "vim mode toggled in the TUI input layer".to_string()
                            } else {
                                format!("vim mode command received: {}", args.join(" "))
                            }
                        }
                        SlashCommand::TerminalSetup => {
                            "Use /terminal-setup in interactive mode.".to_string()
                        }
                        SlashCommand::Keybindings => {
                            let path = AppConfig::keybindings_path().unwrap_or_default();
                            format!("Keybindings: {}", path.display())
                        }
                        SlashCommand::Doctor => serde_json::to_string_pretty(&doctor_payload(
                            cwd,
                            &DoctorArgs::default(),
                        )?)?,
                        SlashCommand::Copy => "Copied last response to clipboard.".to_string(),
                        SlashCommand::Paste => {
                            if let Some(img_bytes) = read_image_from_clipboard() {
                                use base64::Engine;
                                let b64 =
                                    base64::engine::general_purpose::STANDARD.encode(&img_bytes);
                                if let Ok(mut imgs) = pending_images.lock() {
                                    imgs.push(codingbuddy_core::ImageContent {
                                        mime: "image/png".to_string(),
                                        base64_data: b64,
                                    });
                                }
                                format!(
                                    "Image pasted ({} bytes). It will be included in your next prompt.",
                                    img_bytes.len()
                                )
                            } else {
                                let pasted = read_from_clipboard().unwrap_or_default();
                                if pasted.is_empty() {
                                    "clipboard is empty or unavailable".to_string()
                                } else {
                                    pasted
                                }
                            }
                        }
                        SlashCommand::Voice(args) => slash_voice_output(&args)?,
                        SlashCommand::Debug(args) => match parse_debug_analysis_args(&args)? {
                            Some(doctor_args) => {
                                serde_json::to_string_pretty(&doctor_payload(cwd, &doctor_args)?)?
                            }
                            None => format!(
                                "Debug: {}",
                                if args.is_empty() {
                                    "general".to_string()
                                } else {
                                    args.join(" ")
                                }
                            ),
                        },
                        SlashCommand::Exit => "Exiting...".to_string(),
                        SlashCommand::Hooks(_) => {
                            let hooks = &cfg.hooks;
                            serde_json::to_string_pretty(hooks)
                                .unwrap_or_else(|_| "no hooks configured".to_string())
                        }
                        SlashCommand::Rename(name) => {
                            if let Some(n) = name {
                                format!("Session renamed to: {n}")
                            } else {
                                "Usage: /rename <name>".to_string()
                            }
                        }
                        SlashCommand::Resume(id) => {
                            if let Some(id) = id {
                                let session_id = Uuid::parse_str(&id)
                                    .map_err(|_| anyhow!("invalid session ID: {id}"))?;
                                let payload = session_focus_payload(cwd, session_id)?;
                                if let Ok(mut guard) = active_session_for_closure.lock() {
                                    *guard = Some(session_id);
                                }
                                payload["message"]
                                    .as_str()
                                    .unwrap_or("active chat session updated")
                                    .to_string()
                            } else {
                                let current = active_session_for_closure
                                    .lock()
                                    .map(|guard| *guard)
                                    .unwrap_or(None);
                                current
                                    .map(|session_id| format!("current chat session: {session_id}"))
                                    .unwrap_or_else(|| "Usage: /resume <session-id>".to_string())
                            }
                        }
                        SlashCommand::Stats => {
                            let store = Store::new(cwd)?;
                            let usage = store.usage_summary(None, Some(24))?;
                            format!(
                                "24h: input={} output={} records={}",
                                usage.input_tokens, usage.output_tokens, usage.records
                            )
                        }
                        SlashCommand::Statusline(_) => {
                            let session_override = active_session_for_closure
                                .lock()
                                .map(|guard| *guard)
                                .unwrap_or(None);
                            let status = current_ui_status(
                                cwd,
                                cfg,
                                force_max_think.load(Ordering::Relaxed),
                                session_override,
                            )?;
                            format!(
                                "{}\nstatusline shortcut is deprecated; use /status for state and settings.json for formatting.",
                                render_statusline(&status)
                            )
                        }
                        SlashCommand::Theme(t) => {
                            if let Some(t) = t {
                                format!("Theme: {t}")
                            } else {
                                "Available: default, dark, light".to_string()
                            }
                        }
                        SlashCommand::Usage => {
                            let store = Store::new(cwd)?;
                            let usage = store.usage_summary(None, None)?;
                            format!(
                                "Usage: input={} output={}",
                                usage.input_tokens, usage.output_tokens
                            )
                        }
                        SlashCommand::AddDir(args) => {
                            let mut guard = additional_dirs_for_closure
                                .lock()
                                .map_err(|_| anyhow!("failed to access additional dir state"))?;
                            slash_add_dirs(cwd, &mut guard, &args)?
                        }
                        SlashCommand::Bug => format!(
                            "Report bugs at https://github.com/anthropics/codingbuddy-cli/issues\nLogs: {}",
                            codingbuddy_core::runtime_dir(cwd).join("logs").display()
                        ),
                        SlashCommand::PrComments(args) => {
                            if let Some(pr) = args.first() {
                                let payload =
                                    pr_comments_payload(cwd, pr, args.get(1).map(|s| s.as_str()))?;
                                serde_json::to_string_pretty(&payload)?
                            } else {
                                "Usage: /pr_comments <number> [output.json]".to_string()
                            }
                        }
                        SlashCommand::ReleaseNotes(args) => {
                            let range = args.first().map(|s| s.as_str()).unwrap_or("HEAD~10..HEAD");
                            let payload =
                                release_notes_payload(cwd, range, args.get(1).map(|s| s.as_str()))?;
                            serde_json::to_string_pretty(&payload)?
                        }
                        SlashCommand::Login => {
                            let payload = login_payload(cwd)?;
                            serde_json::to_string_pretty(&payload)?
                        }
                        SlashCommand::Logout => {
                            let payload = logout_payload(cwd)?;
                            serde_json::to_string_pretty(&payload)?
                        }
                        SlashCommand::Desktop(args) => {
                            let payload = desktop_payload(cwd, &args)?;
                            serde_json::to_string_pretty(&payload)?
                        }
                        SlashCommand::Todos(args) => {
                            let session_override = active_session_for_closure
                                .lock()
                                .map(|guard| *guard)
                                .unwrap_or(None);
                            let payload = todos_payload(cwd, session_override, &args)?;
                            render_todos_payload(&payload)
                        }
                        SlashCommand::CommentTodos(args) => {
                            let payload = comment_todos_payload(cwd, &args)?;
                            render_comment_todos_payload(&payload)
                        }
                        SlashCommand::Chrome(args) => {
                            let payload = chrome_payload(cwd, &args)?;
                            serde_json::to_string_pretty(&payload)?
                        }
                        SlashCommand::Unknown { name, args } => {
                            let custom_cmds = codingbuddy_skills::load_custom_commands(cwd);
                            if let Some(cmd) = custom_cmds.iter().find(|c| c.name == name) {
                                codingbuddy_skills::render_custom_command(
                                    cmd,
                                    &args.join(" "),
                                    cwd,
                                    &uuid::Uuid::now_v7().to_string(),
                                )
                            } else {
                                format!("unknown slash command: /{name}")
                            }
                        }
                    };
                    Ok(out)
                })();
                match result {
                    Ok(output) => {
                        let _ = tx.send(TuiStreamEvent::Done(output));
                    }
                    Err(e) => {
                        let _ = tx.send(TuiStreamEvent::Error(e.to_string()));
                    }
                }
                return;
            }

            // Agent prompt — expand @file mentions and set stream callback.
            let engine_clone = Arc::clone(&engine);
            let mut prompt = codingbuddy_ui::expand_at_mentions(prompt);
            if watch_files_enabled
                && let Some((digest, hints)) = scan_watch_comment_payload(cwd)
                && let Ok(mut guard) = watch_digest_for_closure.lock()
                && *guard != Some(digest)
            {
                *guard = Some(digest);
                prompt.push_str("\n\nAUTO_WATCH_CONTEXT\nDetected comment hints in workspace:\n");
                prompt.push_str(&hints);
                prompt.push_str("\nAUTO_WATCH_CONTEXT_END");
            }
            let max_think = force_max_think.load(Ordering::Relaxed);
            let tx_stream = tx.clone();
            let tx_done = tx.clone();
            let read_only_for_turn = read_only_for_closure.load(Ordering::Relaxed);
            let mode_for_turn = active_mode_for_closure
                .lock()
                .map(|mode| *mode)
                .unwrap_or(ChatMode::Code);
            let prompt_additional_dirs = additional_dirs_for_closure
                .lock()
                .map(|dirs| dirs.clone())
                .unwrap_or_default();
            let images_for_turn = pending_images
                .lock()
                .map(|mut imgs| std::mem::take(&mut *imgs))
                .unwrap_or_default();
            let teammate_mode_for_turn = teammate_mode.clone();
            let session_id_for_turn = active_session_for_closure
                .lock()
                .map(|guard| *guard)
                .unwrap_or(None);

            engine.set_stream_callback(std::sync::Arc::new(move |chunk| match chunk {
                StreamChunk::ContentDelta(s) => {
                    let _ = tx_stream.send(TuiStreamEvent::ContentDelta(s));
                }
                StreamChunk::ReasoningDelta(s) => {
                    let _ = tx_stream.send(TuiStreamEvent::ReasoningDelta(s));
                }
                StreamChunk::ToolCallStart {
                    tool_name,
                    args_summary,
                } => {
                    let _ = tx_stream.send(TuiStreamEvent::ToolCallStart {
                        tool_name,
                        args_summary,
                    });
                }
                StreamChunk::ToolCallEnd {
                    tool_name,
                    duration_ms,
                    success,
                    summary,
                } => {
                    let _ = tx_stream.send(TuiStreamEvent::ToolCallEnd {
                        tool_name,
                        duration_ms,
                        summary,
                        success,
                    });
                }
                StreamChunk::ModeTransition { from, to, reason } => {
                    let _ = tx_stream.send(TuiStreamEvent::ModeTransition { from, to, reason });
                }
                StreamChunk::SubagentSpawned { run_id, name, goal } => {
                    let _ = tx_stream.send(TuiStreamEvent::SubagentSpawned { run_id, name, goal });
                }
                StreamChunk::SubagentCompleted {
                    run_id,
                    name,
                    summary,
                } => {
                    let _ = tx_stream.send(TuiStreamEvent::SubagentCompleted {
                        run_id,
                        name,
                        summary,
                    });
                }
                StreamChunk::SubagentFailed {
                    run_id,
                    name,
                    error,
                } => {
                    let _ = tx_stream.send(TuiStreamEvent::SubagentFailed {
                        run_id,
                        name,
                        error,
                    });
                }
                StreamChunk::ImageData { data, label } => {
                    let _ = tx_stream.send(TuiStreamEvent::ImageDisplay { data, label });
                }
                StreamChunk::WatchTriggered {
                    digest,
                    comment_count,
                } => {
                    let _ = tx_stream.send(TuiStreamEvent::WatchTriggered {
                        digest,
                        comment_count,
                    });
                }
                StreamChunk::SecurityWarning { message } => {
                    let _ = tx_stream.send(TuiStreamEvent::Error(format!("⚠ Security: {message}")));
                }
                StreamChunk::Error { message, .. } => {
                    let _ = tx_stream.send(TuiStreamEvent::Error(message));
                }
                StreamChunk::PhaseTransition { from, to } => {
                    let _ = tx_stream.send(TuiStreamEvent::ModeTransition {
                        from,
                        to,
                        reason: "phase".to_string(),
                    });
                }
                StreamChunk::ModelChanged { model } => {
                    let _ = tx_stream.send(TuiStreamEvent::ModeTransition {
                        from: String::new(),
                        to: model,
                        reason: "model_switch".to_string(),
                    });
                }
                StreamChunk::UsageUpdate { .. } => {}
                StreamChunk::ClearStreamingText => {
                    let _ = tx_stream.send(TuiStreamEvent::ClearStreamingText);
                }
                StreamChunk::SnapshotRecorded { .. } => {}
                StreamChunk::Done { .. } => {}
            }));

            let prompt_repo_root_override = repo_root_override.clone();
            thread::spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    engine_clone.chat_with_options(
                        &prompt,
                        ChatOptions {
                            tools: allow_tools && !read_only_for_turn,
                            force_max_think: max_think,
                            additional_dirs: prompt_additional_dirs,
                            repo_root_override: prompt_repo_root_override.clone(),
                            debug_context,
                            mode: mode_for_turn,
                            teammate_mode: teammate_mode_for_turn.clone(),
                            detect_urls,
                            watch_files: watch_files_enabled,
                            images: images_for_turn,
                            session_id: session_id_for_turn,
                            ..Default::default()
                        },
                    )
                }));
                match result {
                    Ok(Ok(output)) => {
                        let _ = tx_done.send(TuiStreamEvent::Done(output));
                    }
                    Ok(Err(e)) => {
                        let _ = tx_done.send(TuiStreamEvent::Error(e.to_string()));
                    }
                    Err(panic_payload) => {
                        let msg = panic_payload
                            .downcast_ref::<String>()
                            .map(|s| s.as_str())
                            .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                            .unwrap_or("unknown panic");
                        let _ = tx_done.send(TuiStreamEvent::Error(format!(
                            "agent thread panicked: {msg}"
                        )));
                    }
                }
            });
        },
        move || {
            let session_override = active_session_id.lock().map(|guard| *guard).unwrap_or(None);
            current_ui_status(
                cwd,
                cfg,
                fmt_refresh.load(Ordering::Relaxed),
                session_override,
            )
            .ok()
        },
        ml_completion_cb,
    )
}
