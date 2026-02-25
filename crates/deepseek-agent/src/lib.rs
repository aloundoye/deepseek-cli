mod analysis;
pub mod apply;
mod architect;
mod complexity;
mod editor;
mod gather_context;
mod intent;
pub mod linter;
mod r#loop;
pub mod run_engine;
mod repo_map_v2;
mod team;
mod verify;
pub mod watch;

pub use repo_map_v2::clear_tag_cache;

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
use deepseek_tools::LocalToolHost;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Handler for spawn_task subagent workers. Retained for API compatibility,
/// but the core loop does not dispatch subagents in this architecture.
type SubagentWorkerFn = Arc<dyn Fn(&SubagentTask) -> Result<String> + Send + Sync>;
type ApprovalHandler = Box<dyn FnMut(&ToolCall) -> Result<bool> + Send>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatMode {
    Ask,
    #[default]
    Code,
    Architect,
    Context,
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
    /// When true, agents apply file changes directly without asking for confirmation.
    pub force_execute: bool,
    pub force_plan_only: bool,
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
        cfg: AppConfig,
        llm: Box<dyn LlmClient + Send + Sync>,
    ) -> Result<Self> {
        let store = Store::new(workspace)?;
        let observer = Observer::new(workspace, &cfg.telemetry)?;
        let policy = PolicyEngine::from_app_config(&cfg.policy);
        let tool_host = Arc::new(LocalToolHost::new(workspace, policy.clone())?);

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
            force_execute: options.force_execute,
            force_plan_only: options.force_plan_only,
        });
        let intent_label = match task_intent {
            intent::TaskIntent::InspectRepo => "InspectRepo",
            intent::TaskIntent::EditCode => "EditCode",
            intent::TaskIntent::ArchitectOnly => "ArchitectOnly",
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

    pub fn plan_only(&self, prompt: &str, options: &ChatOptions) -> Result<deepseek_core::Plan> {
        let feedback = architect::ArchitectFeedback::default();
        let input = architect::ArchitectInput {
            user_prompt: prompt,
            iteration: 1,
            feedback: &feedback,
            max_files: self.cfg.agent_loop.max_files_per_iteration as usize,
            additional_dirs: &[],
            debug_context: false,
            chat_history: &options.chat_history,
        };
        let plan = architect::run_architect(
            self,
            &self.workspace,
            &input,
            self.cfg.agent_loop.architect_parse_retries as usize,
        )?;
        Ok(deepseek_core::Plan {
            plan_id: uuid::Uuid::now_v7(),
            version: 1,
            goal: prompt.to_string(),
            assumptions: vec![],
            steps: plan
                .steps
                .iter()
                .enumerate()
                .map(|(idx, step)| deepseek_core::PlanStep {
                    step_id: uuid::Uuid::now_v7(),
                    title: format!("Step {}", idx + 1),
                    intent: step.clone(),
                    tools: vec![],
                    files: plan.files.iter().map(|f| f.path.clone()).collect(),
                    done: false,
                })
                .collect(),
            verification: plan.verify_commands.clone(),
            risk_notes: plan.acceptance.clone(),
        })
    }

    pub fn chat_with_options(&self, prompt: &str, mut options: ChatOptions) -> Result<String> {
        let prompt_enriched = enrich_prompt_with_urls(prompt, options.detect_urls);
        let task_intent = intent::classify_intent(&intent::IntentInput {
            prompt: &prompt_enriched,
            mode: options.mode,
            tools: options.tools,
            force_execute: options.force_execute,
            force_plan_only: options.force_plan_only,
        });

        // Load chat history from the store if possible
        if let Ok(Some(session)) = self.store.load_latest_session() {
            if let Ok(projection) = self.store.rebuild_from_events(session.session_id) {
                options.chat_history = projection.chat_messages;
            }
        }

        // Record User Turn
        self.append_event_best_effort(EventKind::ChatTurnV1 {
            message: ChatMessage::User {
                content: prompt_enriched.clone(),
            },
        });

        let result = if task_intent == intent::TaskIntent::ArchitectOnly {
            run_engine::run(self, &prompt_enriched, &options)
        } else if task_intent == intent::TaskIntent::InspectRepo {
            self.analyze_with_options(&prompt_enriched, options)
        } else if should_run_team_orchestration(self, &prompt_enriched, &options) {
            team::run(self, &prompt_enriched, &options)
        } else {
            run_engine::run(self, &prompt_enriched, &options)
        };

        if let Ok(text) = &result {
            // Record Assistant Turn
            self.append_event_best_effort(EventKind::ChatTurnV1 {
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

    /// Alternative agent loop with integrated lint auto-fix, failure classification,
    /// and commit proposals. Uses a monolithic loop (architect → editor → lint → verify)
    /// instead of the state-machine `RunEngine` approach.
    pub fn chat_loop(&self, prompt: &str, options: ChatOptions) -> Result<String> {
        let prompt_enriched = enrich_prompt_with_urls(prompt, options.detect_urls);

        // Load chat history from the store if possible
        let mut options = options;
        if let Ok(Some(session)) = self.store.load_latest_session() {
            if let Ok(projection) = self.store.rebuild_from_events(session.session_id) {
                options.chat_history = projection.chat_messages;
            }
        }

        // Record User Turn
        self.append_event_best_effort(EventKind::ChatTurnV1 {
            message: ChatMessage::User {
                content: prompt_enriched.clone(),
            },
        });

        let result = r#loop::run(self, &prompt_enriched, &options);

        if let Ok(text) = &result {
            self.append_event_best_effort(EventKind::ChatTurnV1 {
                message: ChatMessage::Assistant {
                    content: Some(text.clone()),
                    reasoning_content: None,
                    tool_calls: Vec::new(),
                },
            });
        }

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

    pub(crate) fn record_usage(&self, model: &str, usage: &deepseek_core::TokenUsage) {
        self.append_event_best_effort(EventKind::UsageUpdatedV1 {
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
        "{prompt}\n\nAUTO_URL_CONTEXT_V1\n{}\nAUTO_URL_CONTEXT_END",
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
}
