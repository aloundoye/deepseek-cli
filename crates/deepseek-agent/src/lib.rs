mod analysis;
mod apply;
mod architect;
mod editor;
mod r#loop;
mod verify;

use anyhow::{Result, anyhow};
use deepseek_core::{
    AppConfig, ChatMessage, ChatRequest, StreamChunk, ToolCall, ToolChoice, UserQuestionHandler,
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

/// Handler for spawn_task subagent workers. Retained for API compatibility,
/// but the core loop does not dispatch subagents in this architecture.
type SubagentWorkerFn = Arc<dyn Fn(&SubagentTask) -> Result<String> + Send + Sync>;
type ApprovalHandler = Box<dyn FnMut(&ToolCall) -> Result<bool> + Send>;

#[derive(Debug, Clone, Default)]
pub struct ChatOptions {
    /// When false, run analysis-only response path (no edit/apply/verify loop).
    pub tools: bool,
    /// When true in analysis path, uses max_think model.
    pub force_max_think: bool,
    /// Replace default analysis system prompt.
    pub system_prompt_override: Option<String>,
    /// Append to default analysis system prompt.
    pub system_prompt_append: Option<String>,
    /// Additional directories for repository context lookup.
    pub additional_dirs: Vec<PathBuf>,
}

pub struct AgentEngine {
    pub(crate) workspace: PathBuf,
    store: Store,
    pub(crate) llm: Box<dyn LlmClient + Send + Sync>,
    observer: Observer,
    pub(crate) tool_host: Arc<LocalToolHost>,
    policy: PolicyEngine,
    pub(crate) cfg: AppConfig,
    stream_callback: Mutex<Option<deepseek_core::StreamCallback>>,
    max_turns: Option<u64>,
    max_budget_usd: Option<f64>,
    approval_handler: Mutex<Option<ApprovalHandler>>,
    user_question_handler: Mutex<Option<UserQuestionHandler>>,
    subagent_worker: Mutex<Option<SubagentWorkerFn>>,
    #[allow(dead_code)]
    hooks: HookRuntime,
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
            thinking: None,
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
        let response = analysis::analyze(self.llm.as_ref(), &self.cfg, prompt, &options)?;
        self.stream(StreamChunk::ContentDelta(response.clone()));
        self.stream(StreamChunk::Done);
        Ok(response)
    }

    pub fn plan_only(&self, prompt: &str) -> Result<deepseek_core::Plan> {
        let feedback = architect::ArchitectFeedback::default();
        let input = architect::ArchitectInput {
            user_prompt: prompt,
            iteration: 1,
            feedback: &feedback,
            max_files: self.cfg.agent_loop.max_files_per_iteration as usize,
            additional_dirs: &[],
        };
        let plan = architect::run_architect(
            self.llm.as_ref(),
            &self.cfg,
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

    pub fn chat_with_options(&self, prompt: &str, options: ChatOptions) -> Result<String> {
        if !options.tools {
            return self.analyze_with_options(prompt, options);
        }

        r#loop::run(self, prompt, &options)
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
