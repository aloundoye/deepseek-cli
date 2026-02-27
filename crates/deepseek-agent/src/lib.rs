mod analysis;
pub mod apply;
pub mod complexity;
mod gather_context;
mod intent;
pub mod prompts;
mod repo_map;
mod shared;
mod team;
pub mod tool_bridge;
pub mod tool_loop;
mod verify;
pub mod watch;

pub use repo_map::clear_tag_cache;

use anyhow::{Result, anyhow};
use chrono::Utc;
use deepseek_core::{
    AppConfig, ChatMessage, ChatRequest, EventEnvelope, EventKind, StreamChunk, ToolCall,
    ToolChoice, UserQuestionHandler,
};
use deepseek_hooks::{HookRuntime, HooksConfig};
use deepseek_llm::{DeepSeekClient, LlmClient};
use deepseek_mcp::McpManager;
use deepseek_observe::Observer;
use deepseek_policy::PolicyEngine;
use deepseek_store::Store;
use deepseek_subagent::SubagentTask;
use deepseek_tools::{LocalToolHost, tool_definitions};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Handler for spawn_task subagent workers. Retained for API compatibility,
/// but the core loop does not dispatch subagents in this architecture.
type SubagentWorkerFn = Arc<dyn Fn(&SubagentTask) -> Result<String> + Send + Sync>;
type ApprovalHandler = Box<dyn FnMut(&ToolCall) -> Result<bool> + Send>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatMode {
    Ask,          // Read-only, tool-use loop (read_only=true)
    #[default]
    Code,         // Default: tool-use loop (full capability)
    Context,      // Read-only, tool-use loop (read_only=true)
}

#[derive(Debug, Default, Clone)]
pub struct ChatOptions {
    pub tools: bool,
    /// When true in analysis path, uses max_think model.
    pub force_max_think: bool,
    /// Replace default analysis system prompt.
    pub system_prompt_override: Option<String>,
    /// Append to default analysis system prompt.
    pub system_prompt_append: Option<String>,
    /// Additional directories for repository context lookup.
    pub additional_dirs: Vec<PathBuf>,
    /// Optional repository root override for context bootstrap/analysis.
    pub repo_root_override: Option<PathBuf>,
    /// Log expanded context blocks to stderr before sending to LLM.
    pub debug_context: bool,
    pub mode: ChatMode,
    pub teammate_mode: Option<String>,
    pub disable_team_orchestration: bool,
    /// When true, attempts to parse and fetch web URLs from prompt.
    pub detect_urls: bool,
    /// Watch mode hint (CLI handles filesystem watch orchestration).
    pub watch_files: bool,
    /// Images to include with the next LLM request (multimodal paste).
    pub images: Vec<deepseek_core::ImageContent>,
    /// Full context history (transcript) of the active session.
    pub chat_history: Vec<ChatMessage>,
}

pub struct AgentEngine {
    pub(crate) workspace: PathBuf,
    store: Store,
    pub(crate) llm: Box<dyn LlmClient + Send + Sync>,
    observer: Observer,
    pub(crate) tool_host: Arc<LocalToolHost>,
    pub(crate) policy: PolicyEngine,
    pub(crate) cfg: AppConfig,
    stream_callback: Mutex<Option<deepseek_core::StreamCallback>>,
    max_turns: Option<u64>,
    max_budget_usd: Option<f64>,
    approval_handler: Mutex<Option<ApprovalHandler>>,
    user_question_handler: Mutex<Option<UserQuestionHandler>>,
    subagent_worker: Mutex<Option<SubagentWorkerFn>>,
    pub(crate) hooks: HookRuntime,
    #[allow(dead_code)]
    mcp: Option<McpManager>,
}

impl AgentEngine {
    pub fn new(workspace: &Path) -> Result<Self> {
        let cfg = AppConfig::ensure(workspace)?;
        let llm = Box::new(DeepSeekClient::new(cfg.llm.clone())?);
        Self::new_with_components(workspace, cfg, llm)
    }

    pub fn new_with_llm(workspace: &Path, llm: Box<dyn LlmClient + Send + Sync>) -> Result<Self> {
        let cfg = AppConfig::ensure(workspace)?;
        Self::new_with_components(workspace, cfg, llm)
    }

    fn new_with_components(
        workspace: &Path,
        mut cfg: AppConfig,
        llm: Box<dyn LlmClient + Send + Sync>,
    ) -> Result<Self> {
        // Apply managed settings overrides (enterprise/team constraints)
        if let Some(managed) = deepseek_policy::load_managed_settings() {
            managed.apply_to_config(&mut cfg);
        }

        let store = Store::new(workspace)?;
        let observer = Observer::new(workspace, &cfg.telemetry)?;
        let mut policy = PolicyEngine::from_app_config(&cfg.policy);

        // Load persistent bash approvals from store.
        let project_hash = format!("{:x}", {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            workspace.to_string_lossy().as_bytes().hash(&mut h);
            h.finish()
        });
        if let Ok(approvals) = store.list_persistent_approvals(&project_hash) {
            let bash_approvals: Vec<String> = approvals
                .iter()
                .filter(|a| a.tool_name == "bash.run")
                .map(|a| a.command_pattern.clone())
                .collect();
            if !bash_approvals.is_empty() {
                policy.set_persistent_bash_approvals(bash_approvals);
            }
        }

        let tool_host = Arc::new(LocalToolHost::new(workspace, policy.clone())?);

        // Best-effort checkpoint cleanup on session start.
        if cfg.cleanup_period_days > 0 {
            if let Ok(mem) = deepseek_memory::MemoryManager::new(workspace) {
                let _ = mem.cleanup_old_checkpoints(cfg.cleanup_period_days);
            }
        }

        let hooks_config: HooksConfig =
            serde_json::from_value(cfg.hooks.clone()).unwrap_or_default();
        let hooks = HookRuntime::new(workspace, hooks_config);

        Ok(Self {
            workspace: workspace.to_path_buf(),
            store,
            llm,
            observer,
            tool_host,
            policy,
            cfg,
            stream_callback: Mutex::new(None),
            max_turns: None,
            max_budget_usd: None,
            approval_handler: Mutex::new(None),
            user_question_handler: Mutex::new(None),
            subagent_worker: Mutex::new(None),
            hooks,
            mcp: McpManager::new(workspace).ok(),
        })
    }

    pub fn set_max_turns(&mut self, max: Option<u64>) {
        self.max_turns = max;
    }

    pub fn set_max_budget_usd(&mut self, max: Option<f64>) {
        self.max_budget_usd = max;
    }

    /// Enable lint auto-fix loop with a single command (convenience for --lint-cmd).
    pub fn set_lint_command(&mut self, lang: &str, cmd: &str) {
        self.cfg.agent_loop.lint.enabled = true;
        self.cfg
            .agent_loop
            .lint
            .commands
            .insert(lang.to_string(), cmd.to_string());
    }

    /// Enable lint auto-fix loop (convenience for --auto-lint).
    pub fn enable_lint(&mut self) {
        self.cfg.agent_loop.lint.enabled = true;
    }

    pub fn validate_api_key(&self) -> Result<()> {
        let test_request = ChatRequest {
            model: self.cfg.llm.base_model.clone(),
            messages: vec![ChatMessage::User {
                content: "hi".to_string(),
            }],
            tools: vec![],
            tool_choice: ToolChoice::none(),
            max_tokens: 1,
            temperature: Some(0.0),
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            logprobs: None,
            top_logprobs: None,
            thinking: None,
            images: vec![],
            response_format: None,
        };

        match self.llm.complete_chat(&test_request) {
            Ok(_) => Ok(()),
            Err(e) => {
                let err_str = e.to_string().to_ascii_lowercase();
                if err_str.contains("401")
                    || err_str.contains("unauthorized")
                    || err_str.contains("invalid api key")
                    || err_str.contains("authentication")
                {
                    Err(anyhow!(
                        "Invalid or missing API key.\n\nSet your DeepSeek API key using one of:\n• export DEEPSEEK_API_KEY=your-key-here\n• Add \"api_key\" to ~/.deepseek/settings.json under the \"llm\" section\n\nGet an API key at: https://platform.deepseek.com/api_keys"
                    ))
                } else {
                    self.observer.warn_log(&format!(
                        "API key validation got non-auth error (startup continues): {e}"
                    ));
                    Ok(())
                }
            }
        }
    }

    pub fn set_permission_mode(&mut self, mode: &str) {
        self.policy = PolicyEngine::from_mode(mode);
        match LocalToolHost::new(&self.workspace, self.policy.clone()) {
            Ok(host) => {
                self.tool_host = Arc::new(host);
            }
            Err(e) => {
                self.observer.warn_log(&format!(
                    "failed to rebuild tool host for permission mode `{mode}`: {e}"
                ));
            }
        }
    }

    pub fn set_verbose(&mut self, verbose: bool) {
        self.observer.set_verbose(verbose);
    }

    pub fn set_approval_handler(&self, handler: ApprovalHandler) {
        if let Ok(mut guard) = self.approval_handler.lock() {
            *guard = Some(handler);
        }
    }

    pub fn set_stream_callback(&self, cb: deepseek_core::StreamCallback) {
        if let Ok(mut guard) = self.stream_callback.lock() {
            *guard = Some(cb);
        }
    }

    pub fn set_user_question_handler(&self, handler: UserQuestionHandler) {
        if let Ok(mut guard) = self.user_question_handler.lock() {
            *guard = Some(handler);
        }
    }

    pub fn set_subagent_worker(&self, worker: SubagentWorkerFn) {
        if let Ok(mut guard) = self.subagent_worker.lock() {
            *guard = Some(worker);
        }
    }

    pub fn subagent_worker(&self) -> Option<SubagentWorkerFn> {
        self.subagent_worker.lock().ok().and_then(|g| g.clone())
    }

    /// Build a subagent worker closure that delegates spawn_task to a new engine instance.
    ///
    /// If the user has provided an explicit subagent worker (via `set_subagent_worker`),
    /// that takes priority. Otherwise we build a default worker that creates a new
    /// `AgentEngine` and runs `chat_with_options()` with the subtask prompt.
    fn build_subagent_worker(&self) -> Option<tool_loop::SubagentWorker> {
        // Check if user provided an explicit worker
        if let Some(worker) = self.subagent_worker() {
            // Adapt the SubagentWorkerFn to the SubagentWorker signature
            return Some(Arc::new(move |req: tool_loop::SubagentRequest| {
                let role = match req.subagent_type.as_str() {
                    "explore" => deepseek_subagent::SubagentRole::Explore,
                    "plan" => deepseek_subagent::SubagentRole::Plan,
                    _ => deepseek_subagent::SubagentRole::Task,
                };
                let task = SubagentTask {
                    run_id: uuid::Uuid::new_v4(),
                    name: req.task_name.clone(),
                    goal: req.prompt.clone(),
                    role,
                    team: String::new(),
                    read_only_fallback: false,
                    custom_agent: None,
                };
                worker(&task)
            }));
        }
        None
    }

    /// Build a skill runner callback that wraps SkillManager.
    fn build_skill_runner(&self) -> Option<tool_loop::SkillRunner> {
        let workspace = self.workspace.clone();
        let skill_paths: Vec<String> = self
            .cfg
            .skills
            .paths
            .iter()
            .cloned()
            .collect();

        Some(Arc::new(move |skill_name: &str, args: Option<&str>| {
            let manager = deepseek_skills::SkillManager::new(&workspace)?;
            let output = manager.run(skill_name, args, &skill_paths);
            match output {
                Ok(run_output) => Ok(Some(tool_loop::SkillInvocationResult {
                    rendered_prompt: run_output.rendered_prompt,
                    forked: run_output.forked,
                    allowed_tools: run_output.allowed_tools,
                    disallowed_tools: run_output.disallowed_tools,
                    disable_model_invocation: run_output.disable_model_invocation,
                })),
                Err(e) => {
                    let msg = e.to_string();
                    if msg.contains("not found") {
                        Ok(None)
                    } else {
                        Err(e)
                    }
                }
            }
        }))
    }

    /// Mutable access to the configuration for test overrides.
    pub fn cfg_mut(&mut self) -> &mut deepseek_core::AppConfig {
        &mut self.cfg
    }

    /// Fire PreToolUse hooks. Returns the aggregate result.
    /// Callers should check `result.blocked` to decide whether to skip tool execution.
    pub fn fire_pre_tool_use(&self, tool_name: &str, tool_input: &serde_json::Value) -> deepseek_hooks::HookResult {
        let input = deepseek_hooks::HookInput {
            event: deepseek_hooks::HookEvent::PreToolUse.as_str().to_string(),
            tool_name: Some(tool_name.to_string()),
            tool_input: Some(tool_input.clone()),
            tool_result: None,
            prompt: None,
            session_type: None,
            workspace: self.workspace.display().to_string(),
        };
        self.hooks.fire(deepseek_hooks::HookEvent::PreToolUse, &input)
    }

    /// Fire Stop hooks at end of agent execution.
    pub fn fire_stop(&self) {
        let input = deepseek_hooks::HookInput {
            event: deepseek_hooks::HookEvent::Stop.as_str().to_string(),
            tool_name: None,
            tool_input: None,
            tool_result: None,
            prompt: None,
            session_type: None,
            workspace: self.workspace.display().to_string(),
        };
        let _ = self.hooks.fire(deepseek_hooks::HookEvent::Stop, &input);
    }

    pub(crate) fn stream(&self, chunk: StreamChunk) {
        if let Ok(guard) = self.stream_callback.lock()
            && let Some(cb) = guard.as_ref()
        {
            cb(chunk);
        }
    }

    pub fn chat(&self, prompt: &str) -> Result<String> {
        self.chat_with_options(
            prompt,
            ChatOptions {
                tools: true,
                ..Default::default()
            },
        )
    }

    pub fn analyze_with_options(&self, prompt: &str, options: ChatOptions) -> Result<String> {
        // Grab a clone of the stream callback (if any) to pass to analyze().
        // complete_chat_streaming will fire ContentDelta per-token in real-time.
        let stream_cb = self
            .stream_callback
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().cloned());

        let response = analysis::analyze(
            self.llm.as_ref(),
            &self.cfg,
            &self.workspace,
            prompt,
            &options,
            stream_cb.clone(),
        )?;

        // If we streamed token-by-token, complete_chat_streaming already emitted Done.
        // Only emit bulk ContentDelta + Done for the non-streaming fallback path.
        if stream_cb.is_none() {
            self.stream(StreamChunk::ContentDelta(response.clone()));
            self.stream(StreamChunk::Done { reason: None });
        }
        Ok(response)
    }

    pub fn context_debug_preview(&self, prompt: &str, options: &ChatOptions) -> String {
        let task_intent = intent::classify_intent(&intent::IntentInput {
            prompt,
            mode: options.mode,
            tools: options.tools,
        });
        let intent_label = match task_intent {
            intent::TaskIntent::InspectRepo => "InspectRepo",
            intent::TaskIntent::EditCode => "EditCode",
        };
        let bootstrap = gather_context::gather_for_prompt(
            &self.workspace,
            &self.cfg,
            prompt,
            options.mode,
            &options.additional_dirs,
            options.repo_root_override.as_deref(),
        );
        bootstrap.debug_digest(intent_label, options.mode)
    }

    /// Run the tool-use conversation loop.
    ///
    /// The model freely decides which tools to call at each step, enabling
    /// fluid think→act→observe behavior. Bootstrap context (project structure,
    /// repo map, manifests) is injected on the first turn so the model starts
    /// with project awareness instead of blind.
    fn run_tool_use_loop(&self, prompt: &str, options: &ChatOptions) -> Result<String> {
        let project_memory = deepseek_memory::MemoryManager::new(&self.workspace)
            .ok()
            .and_then(|mm| mm.read_combined_memory().ok());

        // Build workspace context for environment section in system prompt
        let git_branch = std::process::Command::new("git")
            .args(["branch", "--show-current"])
            .current_dir(&self.workspace)
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string());
        let ws_context = prompts::WorkspaceContext {
            cwd: self.workspace.display().to_string(),
            git_branch,
            os: std::env::consts::OS.to_string(),
        };

        // Classify complexity before building system prompt (planning injection depends on it).
        let complexity = complexity::classify_complexity(prompt);

        // Build repo map for Complex tasks so the model knows the project structure
        let repo_map_summary = if complexity == complexity::PromptComplexity::Complex {
            Some(shared::build_repo_map(
                &self.workspace,
                prompt,
                40, // top 40 files
                &options.additional_dirs,
            ))
        } else {
            None
        };

        let system_prompt = prompts::build_tool_use_system_prompt_with_complexity(
            project_memory.as_deref(),
            options.system_prompt_override.as_deref(),
            options.system_prompt_append.as_deref(),
            Some(&ws_context),
            complexity,
            repo_map_summary.as_deref(),
        );

        let read_only = matches!(options.mode, ChatMode::Ask | ChatMode::Context);

        // Complexity-driven initial budget, with evidence-driven escalation in the loop.
        // force_max_think is an explicit user override to maximum.
        let think_budget = if options.force_max_think {
            complexity::MAX_THINK_BUDGET      // 64K
        } else {
            complexity::thinking_budget_for(complexity) // Simple 8K / Medium 16K / Complex 32K
        };

        // Gather lightweight bootstrap context so the model starts with project awareness.
        // Budget: ~15% of context window to leave room for conversation.
        let initial_context = self.build_bootstrap_context(
            prompt,
            options,
            self.cfg.llm.context_window_tokens,
        );

        let config = tool_loop::ToolLoopConfig {
            model: self.cfg.llm.base_model.clone(), // Always deepseek-chat
            max_tokens: deepseek_core::DEEPSEEK_CHAT_THINKING_MAX_OUTPUT_TOKENS,
            temperature: None, // Incompatible with thinking mode
            context_window_tokens: self.cfg.llm.context_window_tokens,
            max_turns: self
                .max_turns
                .unwrap_or(self.cfg.agent_loop.tool_loop_max_turns)
                as usize,
            read_only,
            thinking: Some(deepseek_core::ThinkingConfig::enabled(think_budget)),
            extended_thinking_model: self.cfg.llm.max_think_model.clone(),
            complexity,
            subagent_worker: self.build_subagent_worker(),
            skill_runner: self.build_skill_runner(),
            workspace: Some(self.workspace.clone()),
            retriever: build_retriever_callback(&self.workspace, &self.cfg),
            privacy_router: build_privacy_router(&self.cfg),
            images: options.images.clone(),
            initial_context,
        };

        let mut loop_ = tool_loop::ToolUseLoop::new(
            self.llm.as_ref(),
            self.tool_host.clone(),
            config,
            system_prompt,
            tool_definitions(),
        );

        // Wire stream callback
        if let Ok(guard) = self.stream_callback.lock()
            && let Some(ref cb) = *guard
        {
            loop_.set_stream_callback(cb.clone());
        }

        // Wire approval handler (bridge FnMut → Fn via Mutex)
        if let Ok(mut guard) = self.approval_handler.lock()
            && let Some(handler) = guard.take()
        {
            let handler = Arc::new(Mutex::new(handler));
            loop_.set_approval_callback(Arc::new(move |call| {
                let mut h = handler.lock().map_err(|_| anyhow!("approval handler mutex poisoned"))?;
                h(call)
            }));
        }

        // Wire user question handler
        if let Ok(mut guard) = self.user_question_handler.lock()
            && let Some(handler) = guard.take()
        {
            loop_.set_user_question_callback(handler);
        }

        // Wire hooks — create a new runtime with the same config for the loop
        {
            let hooks_config: deepseek_hooks::HooksConfig =
                serde_json::from_value(self.cfg.hooks.clone()).unwrap_or_default();
            let hooks = deepseek_hooks::HookRuntime::new(&self.workspace, hooks_config);
            loop_.set_hooks(hooks);
        }

        // Wire checkpoint callback — prefers git shadow commits, falls back to file snapshots
        let ws = self.workspace.clone();
        loop_.set_checkpoint_callback(Arc::new(move |reason, modified_files| {
            if let Ok(mm) = deepseek_memory::MemoryManager::new(&ws) {
                // Try shadow commit first (lightweight git-based checkpoint)
                match mm.create_shadow_commit(reason) {
                    Ok(sc) if sc.git_backed => {
                        // Shadow commit succeeded — no file copy needed
                        return Ok(());
                    }
                    _ => {}
                }
                // Fall back to file-based checkpoint
                if modified_files.is_empty() {
                    mm.create_checkpoint(reason)?;
                } else {
                    mm.create_checkpoint_for_files(reason, modified_files)?;
                }
            }
            Ok(())
        }));

        if !options.chat_history.is_empty() {
            loop_ = loop_.with_history(options.chat_history.clone());
        }

        let result = loop_.run(prompt)?;

        if !result.tool_calls_made.is_empty() {
            self.observer.verbose_log(&format!(
                "tool-use loop: {} turns, {} tool calls, {} prompt tokens",
                result.turns,
                result.tool_calls_made.len(),
                result.usage.prompt_tokens,
            ));
        }

        Ok(result.response)
    }

    /// Build bootstrap context messages from gather_context.
    ///
    /// Returns a vec of System messages containing project structure, repo map, etc.
    /// Budgeted to ~15% of context window to leave room for conversation.
    fn build_bootstrap_context(
        &self,
        prompt: &str,
        options: &ChatOptions,
        context_window_tokens: u64,
    ) -> Vec<ChatMessage> {
        let bootstrap = gather_context::gather_for_prompt(
            &self.workspace,
            &self.cfg,
            prompt,
            options.mode,
            &options.additional_dirs,
            options.repo_root_override.as_deref(),
        );

        if !bootstrap.enabled || bootstrap.packet.is_empty() {
            return vec![];
        }

        // Enrich with dependency-analysis hub files
        let mut packet = bootstrap.packet;
        if let Ok(mut ctx_mgr) = deepseek_context::ContextManager::new(&self.workspace) {
            if ctx_mgr.analyze_workspace().is_ok() {
                let suggestions = ctx_mgr.suggest_relevant_files(prompt, 10);
                if !suggestions.is_empty() {
                    packet.push_str("\nKey files (by dependency centrality):\n");
                    for s in &suggestions {
                        let reasons = s.reasons.join(", ");
                        packet.push_str(&format!(
                            "  - {} (score: {:.1}, {})\n",
                            s.path.display(),
                            s.score,
                            reasons
                        ));
                    }
                }
            }
        }

        // Budget: ~15% of context window (128K × 0.15 ≈ 19K tokens)
        let budget_tokens = context_window_tokens / 7;
        let truncated = truncate_to_token_budget(&packet, budget_tokens);

        vec![ChatMessage::System {
            content: format!(
                "PROJECT_CONTEXT (auto-gathered, do NOT repeat this to the user):\n{truncated}"
            ),
        }]
    }

    pub fn chat_with_options(&self, prompt: &str, mut options: ChatOptions) -> Result<String> {
        let prompt_enriched = enrich_prompt_with_urls(prompt, options.detect_urls);

        // Load chat history from the store if possible
        if let Ok(Some(session)) = self.store.load_latest_session() {
            if let Ok(projection) = self.store.rebuild_from_events(session.session_id) {
                options.chat_history = projection.chat_messages;
            }
        }

        // Record User Turn
        self.append_event_best_effort(EventKind::ChatTurn {
            message: ChatMessage::User {
                content: prompt_enriched.clone(),
            },
        });

        let result = if !options.tools {
            // No tools requested → use the analysis path (read-only Q&A)
            self.analyze_with_options(&prompt_enriched, options)
        } else if should_run_team_orchestration(self, &prompt_enriched, &options) {
            team::run(self, &prompt_enriched, &options)
        } else {
            self.run_tool_use_loop(&prompt_enriched, &options)
        };

        if let Ok(text) = &result {
            // Record Assistant Turn
            self.append_event_best_effort(EventKind::ChatTurn {
                message: ChatMessage::Assistant {
                    content: Some(text.clone()),
                    reasoning_content: None,
                    tool_calls: vec![],
                },
            });
        }

        // Fire Stop hooks for post-processing (logging, notifications, etc.)
        self.fire_stop();

        result
    }

    pub(crate) fn append_event_best_effort(&self, kind: EventKind) {
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
            kind,
        };
        let _ = self.store.append_event(&event);
    }

    #[allow(dead_code)]
    pub(crate) fn record_usage(&self, model: &str, usage: &deepseek_core::TokenUsage) {
        self.append_event_best_effort(EventKind::UsageUpdated {
            unit: deepseek_core::LlmUnit::Model(model.to_string()),
            model: model.to_string(),
            input_tokens: usage.prompt_tokens,
            cache_hit_tokens: usage.prompt_cache_hit_tokens,
            cache_miss_tokens: usage.prompt_cache_miss_tokens,
            output_tokens: usage.completion_tokens,
        });
    }

    pub(crate) fn request_approval(&self, call: &ToolCall) -> Result<bool> {
        if let Ok(mut guard) = self.approval_handler.lock()
            && let Some(handler) = guard.as_mut()
        {
            return handler(call);
        }
        Ok(false)
    }

    #[deprecated(note = "resume returns a session summary only")]
    pub fn resume(&self) -> Result<String> {
        let session = self
            .store
            .load_latest_session()?
            .ok_or_else(|| anyhow!("no prior session to resume"))?;
        let projection = self.store.rebuild_from_events(session.session_id)?;
        Ok(format!(
            "resumed session={} turns={} steps={}",
            session.session_id,
            projection.transcript.len(),
            projection.step_status.len()
        ))
    }
}

/// Truncate text to fit a token budget (chars/4 heuristic).
/// Finds a clean break at a newline near the limit.
fn truncate_to_token_budget(text: &str, max_tokens: u64) -> String {
    let max_chars = (max_tokens * 4) as usize;
    if text.len() <= max_chars {
        return text.to_string();
    }
    let truncated = &text[..max_chars.min(text.len())];
    if let Some(last_newline) = truncated.rfind('\n') {
        let remaining = text.len() - last_newline;
        format!(
            "{}...\n[truncated — {} more chars]",
            &text[..last_newline],
            remaining
        )
    } else {
        format!("{}...\n[truncated]", truncated)
    }
}

/// Build a retriever callback from config. Returns None if local_ml is disabled.
///
/// Uses MockEmbeddings (deterministic SHA-256 hashing) by default for consistent
/// file-level retrieval. When compiled with `--features local-ml`, tries Candle
/// embeddings (jina-code-v2) first for true semantic search.
fn build_retriever_callback(
    workspace: &std::path::Path,
    cfg: &deepseek_core::AppConfig,
) -> Option<std::sync::Arc<dyn Fn(&str, usize) -> anyhow::Result<Vec<tool_loop::RetrievalContext>> + Send + Sync>> {
    if !cfg.local_ml.enabled {
        return None;
    }

    // Build embeddings backend (MockEmbeddings provides deterministic SHA-256 hashing;
    // for true semantic search, compile deepseek-local-ml with --features local-ml).
    let embeddings: std::sync::Arc<dyn deepseek_local_ml::EmbeddingsBackend> =
        std::sync::Arc::new(deepseek_local_ml::MockEmbeddings::new(384));

    // Build vector index path
    let index_path = workspace.join(".deepseek").join("vector_index");
    let chunk_config = deepseek_local_ml::ChunkConfig {
        chunk_lines: cfg.local_ml.index.chunk_lines as usize,
        ..Default::default()
    };

    // Build hybrid retriever
    let retriever = match deepseek_local_ml::HybridRetriever::new(
        &index_path,
        embeddings,
        None, // tantivy index — optional
        0.7,  // blend_alpha: favor vector search
        chunk_config,
    ) {
        Ok(r) => std::sync::Arc::new(std::sync::Mutex::new(r)),
        Err(e) => {
            eprintln!("[deepseek] retriever init failed ({e}), retrieval disabled");
            return None;
        }
    };

    // Lazy index on first retrieval call
    let workspace_path = workspace.to_path_buf();
    let indexed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    Some(std::sync::Arc::new(move |query: &str, max_results: usize| {
        let mut retriever = retriever.lock().map_err(|e| anyhow!("retriever lock: {e}"))?;

        // Lazy index on first call
        if !indexed.load(std::sync::atomic::Ordering::Relaxed) {
            if let Err(e) = retriever.build_index(&workspace_path) {
                eprintln!("[deepseek] index build failed: {e}");
            }
            indexed.store(true, std::sync::atomic::Ordering::Relaxed);
        }

        let filter = deepseek_local_ml::SearchFilter {
            max_results,
            ..Default::default()
        };
        let results = retriever.search(query, max_results, &filter)?;
        Ok(results
            .into_iter()
            .map(|r| tool_loop::RetrievalContext {
                file_path: r.chunk.file_path.to_string_lossy().to_string(),
                start_line: r.chunk.start_line,
                end_line: r.chunk.end_line,
                content: r.chunk.content,
                score: r.hybrid_score,
            })
            .collect())
    }))
}

/// Build a privacy router from config. Returns None if privacy is disabled.
fn build_privacy_router(
    cfg: &deepseek_core::AppConfig,
) -> Option<std::sync::Arc<deepseek_local_ml::PrivacyRouter>> {
    if !cfg.local_ml.enabled || !cfg.local_ml.privacy.enabled {
        return None;
    }
    let privacy_config = deepseek_local_ml::PrivacyConfig {
        enabled: true,
        sensitive_globs: cfg.local_ml.privacy.sensitive_globs.clone(),
        sensitive_regex: cfg.local_ml.privacy.sensitive_regex.clone(),
        policy: match cfg.local_ml.privacy.policy.as_str() {
            "block_cloud" => deepseek_local_ml::PrivacyPolicy::BlockCloud,
            "local_only_summary" => deepseek_local_ml::PrivacyPolicy::LocalOnlySummary,
            _ => deepseek_local_ml::PrivacyPolicy::Redact,
        },
        store_raw_in_logs: cfg.local_ml.privacy.store_raw_in_logs,
    };
    deepseek_local_ml::PrivacyRouter::new(privacy_config)
        .ok()
        .map(std::sync::Arc::new)
}

fn should_run_team_orchestration(
    engine: &AgentEngine,
    prompt: &str,
    options: &ChatOptions,
) -> bool {
    if options.disable_team_orchestration {
        return false;
    }

    let value = options
        .teammate_mode
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .or_else(|| {
            std::env::var("DEEPSEEK_TEAMMATE_MODE")
                .ok()
                .map(|v| v.trim().to_ascii_lowercase())
        });

    match value.as_deref() {
        Some("") | Some("0") | Some("false") | Some("off") | Some("none") => false,
        Some(_) => true,
        None => {
            if !engine.cfg.agent_loop.team.auto_enabled {
                return false;
            }
            crate::complexity::score_prompt(prompt)
                >= engine.cfg.agent_loop.team.complexity_threshold
        }
    }
}

fn enrich_prompt_with_urls(prompt: &str, enabled: bool) -> String {
    if !enabled {
        return prompt.to_string();
    }
    let urls = extract_urls(prompt, 3);
    if urls.is_empty() {
        return prompt.to_string();
    }
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .user_agent("deepseek-cli/0.2")
        .build()
    {
        Ok(client) => client,
        Err(_) => return prompt.to_string(),
    };

    let mut rows = Vec::new();
    for url in urls {
        let Ok(resp) = client.get(&url).send() else {
            continue;
        };
        let status = resp.status().as_u16();
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("unknown")
            .to_string();
        let Ok(body) = resp.text() else {
            continue;
        };
        let max_bytes = 24_000usize;
        let truncated = body.len() > max_bytes;
        let clipped = if truncated {
            body[..body.floor_char_boundary(max_bytes)].to_string()
        } else {
            body
        };
        let extracted = if content_type.contains("html") {
            naive_strip_html(&clipped)
        } else {
            clipped
        };
        if extracted.trim().is_empty() {
            continue;
        }
        rows.push(format!(
            "URL|{}\nSTATUS|{}\nCONTENT_TYPE|{}\nTRUNCATED|{}\nEXTRACT|\n{}",
            url,
            status,
            content_type,
            truncated,
            truncate_text(&extracted, 8_000)
        ));
    }

    if rows.is_empty() {
        return prompt.to_string();
    }

    format!(
        "{prompt}\n\nAUTO_URL_CONTEXT\n{}\nAUTO_URL_CONTEXT_END",
        rows.join("\n---\n")
    )
}

fn extract_urls(prompt: &str, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    for token in prompt.split_whitespace() {
        let trimmed = token
            .trim_matches(|c: char| {
                c == ','
                    || c == ';'
                    || c == ')'
                    || c == '('
                    || c == ']'
                    || c == '['
                    || c == '"'
                    || c == '\''
            })
            .trim_end_matches('.');
        if (trimmed.starts_with("http://") || trimmed.starts_with("https://"))
            && !out.iter().any(|row| row == trimmed)
        {
            out.push(trimmed.to_string());
            if out.len() >= limit {
                break;
            }
        }
    }
    out
}

fn naive_strip_html(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut in_tag = false;
    for ch in input.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                out.push('\n');
            }
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    format!("{}...", &text[..text.floor_char_boundary(max_chars)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_urls_handles_punctuation_and_dedup() {
        let prompt = "check https://example.com, then (https://example.com) and https://a.dev.";
        let urls = extract_urls(prompt, 5);
        assert_eq!(urls, vec!["https://example.com", "https://a.dev"]);
    }

    #[test]
    fn html_strip_removes_tags() {
        let html = "<html><body><h1>Title</h1><p>Hello world</p></body></html>";
        let stripped = naive_strip_html(html);
        assert!(stripped.contains("Title"));
        assert!(stripped.contains("Hello world"));
        assert!(!stripped.contains("<h1>"));
    }

    #[test]
    fn truncate_to_token_budget_short_text_unchanged() {
        let text = "hello world\nsecond line";
        let result = truncate_to_token_budget(text, 1000);
        assert_eq!(result, text);
    }

    #[test]
    fn truncate_to_token_budget_long_text_truncated() {
        let text = "line1\nline2\nline3\nline4\nline5\nline6";
        // 4 tokens = 16 chars, which should cut mid-text
        let result = truncate_to_token_budget(text, 4);
        assert!(result.contains("truncated"));
        assert!(result.len() < text.len() + 40); // truncated + marker
    }

    #[test]
    fn shadow_commit_used_in_checkpoint() {
        // Verify that MemoryManager::create_shadow_commit exists and returns
        // the expected ShadowCommit struct. On non-git directories, it should
        // still return a result (with git_backed=false), allowing fallback.
        let workspace = std::env::temp_dir().join(format!("shadow-test-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(&workspace).expect("workspace dir");
        std::fs::write(workspace.join("test.txt"), "hello").expect("write file");

        let mm = deepseek_memory::MemoryManager::new(&workspace).expect("memory manager");
        let result = mm.create_shadow_commit("test checkpoint");
        // In a non-git directory, shadow commit should not be git-backed
        // (it falls back to file-based internally, or returns an error)
        match result {
            Ok(sc) => {
                assert!(!sc.git_backed, "non-git dir should not produce git-backed shadow commit");
            }
            Err(_) => {
                // Also acceptable: shadow commit fails in non-git dir
                // and the checkpoint callback falls back to file-based
            }
        }

        let _ = std::fs::remove_dir_all(&workspace);
    }

    // ── Batch 7: Context Manager Wiring ──

    #[test]
    fn context_manager_handles_missing_repo() {
        // ContextManager should gracefully handle a non-existent or empty workspace
        let result = deepseek_context::ContextManager::new("/tmp/nonexistent_deepseek_test_dir");
        // Should succeed (creates empty graph) — not panic
        assert!(result.is_ok(), "ContextManager::new should not fail on missing dir");
    }

    #[test]
    fn context_manager_suggest_files_on_real_workspace() {
        // Create a temp workspace with a few files that have import relationships
        let workspace = std::env::temp_dir().join("deepseek_test_ctx_mgr");
        let _ = std::fs::remove_dir_all(&workspace);
        std::fs::create_dir_all(workspace.join("src")).expect("create src dir");
        std::fs::write(
            workspace.join("src/lib.rs"),
            "pub mod utils;\npub mod handler;\n",
        )
        .expect("write lib.rs");
        std::fs::write(
            workspace.join("src/utils.rs"),
            "pub fn helper() {}\n",
        )
        .expect("write utils.rs");
        std::fs::write(
            workspace.join("src/handler.rs"),
            "use crate::utils;\npub fn handle() { utils::helper(); }\n",
        )
        .expect("write handler.rs");

        let mut ctx_mgr =
            deepseek_context::ContextManager::new(&workspace).expect("context manager");
        ctx_mgr.analyze_workspace().expect("analyze");

        let suggestions = ctx_mgr.suggest_relevant_files("handler", 5);
        // handler.rs should score high because its name matches the query
        let has_handler = suggestions
            .iter()
            .any(|s| s.path.to_string_lossy().contains("handler"));
        assert!(has_handler, "handler.rs should appear in suggestions for query 'handler'");

        let _ = std::fs::remove_dir_all(&workspace);
    }
}
