//! Public types, configuration, and callback signatures for the tool-use loop.

use anyhow::Result;
use codingbuddy_core::{ChatMessage, EventKind, TokenUsage, UserQuestion};
use std::path::PathBuf;
use std::sync::Arc;

/// Default maximum turns (LLM calls) before stopping the loop.
pub const DEFAULT_MAX_TURNS: usize = 50;

/// Record of a single tool call made during the loop.
#[derive(Debug, Clone)]
pub struct ToolCallRecord {
    pub tool_name: String,
    pub tool_call_id: String,
    pub args_summary: String,
    pub success: bool,
    pub duration_ms: u64,
}

/// Result of running the tool-use loop.
#[derive(Debug, Clone)]
pub struct ToolLoopResult {
    /// Final text response from the LLM.
    pub response: String,
    /// All tool calls made during the loop.
    pub tool_calls_made: Vec<ToolCallRecord>,
    /// Why the loop stopped.
    pub finish_reason: String,
    /// Aggregated token usage across all LLM calls.
    pub usage: TokenUsage,
    /// Number of LLM calls made.
    pub turns: usize,
    /// Full conversation messages (for continuing the conversation).
    pub messages: Vec<ChatMessage>,
}

/// Callback for requesting tool approval from the user.
/// Returns `true` if approved, `false` if denied.
pub type ApprovalCallback = Arc<dyn Fn(&codingbuddy_core::ToolCall) -> Result<bool> + Send + Sync>;

/// Callback for asking the user a question during tool execution.
pub type UserQuestionCallback = Arc<dyn Fn(UserQuestion) -> Option<String> + Send + Sync>;

/// Callback for event logging (tool proposed/result events).
pub type EventCallback = Arc<dyn Fn(EventKind) + Send + Sync>;

/// Callback for creating a checkpoint before destructive tool calls.
/// The second argument contains the files about to be modified (if known).
pub type CheckpointCallback = Arc<dyn Fn(&str, &[PathBuf]) -> Result<()> + Send + Sync>;

/// Callback for executing a subagent task (spawn_task tool).
pub type SubagentWorker = Arc<dyn Fn(SubagentRequest) -> Result<String> + Send + Sync>;

/// Result of invoking a skill (returned by SkillRunner callback).
#[derive(Debug, Clone)]
pub struct SkillInvocationResult {
    /// The rendered skill prompt.
    pub rendered_prompt: String,
    /// Whether the skill should run in an isolated context.
    pub forked: bool,
    /// Tools allowed (empty = all).
    pub allowed_tools: Vec<String>,
    /// Tools disallowed.
    pub disallowed_tools: Vec<String>,
    /// Whether model auto-invocation is disabled.
    pub disable_model_invocation: bool,
}

/// Callback for looking up and running a skill by name.
/// Returns `None` if the skill is not found.
pub type SkillRunner =
    Arc<dyn Fn(&str, Option<&str>) -> Result<Option<SkillInvocationResult>> + Send + Sync>;

/// Callback for injecting relevant code context before LLM calls.
/// Takes `(query, max_results)` and returns matching code chunks.
pub type RetrieverCallback =
    Arc<dyn Fn(&str, usize) -> Result<Vec<RetrievalContext>> + Send + Sync>;

/// Request to spawn a subagent task.
#[derive(Debug, Clone)]
pub struct SubagentRequest {
    pub prompt: String,
    pub task_name: String,
    pub subagent_type: String,
    pub model_override: Option<String>,
    pub max_turns: Option<usize>,
    pub run_in_background: bool,
}

/// Configuration for the tool-use loop.
pub struct ToolLoopConfig {
    pub model: String,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub context_window_tokens: u64,
    pub max_turns: usize,
    /// When true, use read-only tools only (Ask/Context mode).
    pub read_only: bool,
    /// Thinking configuration — enables chain-of-thought reasoning for the main model.
    pub thinking: Option<codingbuddy_core::ThinkingConfig>,
    /// Model name used by `extended_thinking` agent-level tool.
    pub extended_thinking_model: String,
    /// Detected complexity of the user prompt.
    pub complexity: crate::complexity::PromptComplexity,
    /// Optional worker for executing spawn_task subagents.
    pub subagent_worker: Option<SubagentWorker>,
    /// Optional callback for looking up and running skills (slash commands).
    pub skill_runner: Option<SkillRunner>,
    /// Workspace root path (for subagent spawning).
    pub workspace: Option<PathBuf>,
    /// Optional retriever callback for injecting relevant code context before LLM calls.
    /// Takes (query, max_results) and returns matching code chunks.
    pub retriever: Option<RetrieverCallback>,
    /// Optional privacy router for scanning tool outputs before appending to messages.
    pub privacy_router: Option<Arc<codingbuddy_local_ml::PrivacyRouter>>,
    /// Images to include with the LLM request (multimodal).
    pub images: Vec<codingbuddy_core::ImageContent>,
    /// Initial context messages injected after the system prompt but before the
    /// user message. Used for bootstrap context (project structure, repo map, etc.).
    pub initial_context: Vec<ChatMessage>,
    /// Active agent profile name for logging (e.g. "build", "explore", "plan").
    pub profile_name: Option<String>,
}

impl Default for ToolLoopConfig {
    fn default() -> Self {
        Self {
            model: codingbuddy_core::CODINGBUDDY_V32_CHAT_MODEL.to_string(),
            max_tokens: codingbuddy_core::CODINGBUDDY_CHAT_THINKING_MAX_OUTPUT_TOKENS,
            temperature: None,
            context_window_tokens: 128_000,
            max_turns: DEFAULT_MAX_TURNS,
            read_only: false,
            thinking: Some(codingbuddy_core::ThinkingConfig::enabled(
                crate::complexity::MEDIUM_THINK_BUDGET,
            )),
            extended_thinking_model: codingbuddy_core::CODINGBUDDY_V32_REASONER_MODEL.to_string(),
            complexity: crate::complexity::PromptComplexity::Medium,
            subagent_worker: None,
            skill_runner: None,
            workspace: None,
            retriever: None,
            privacy_router: None,
            images: vec![],
            initial_context: vec![],
            profile_name: None,
        }
    }
}

/// A piece of retrieved code context from the workspace index.
#[derive(Debug, Clone)]
pub struct RetrievalContext {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub score: f32,
}
