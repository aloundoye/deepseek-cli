//! Core tool-use conversation loop.
//!
//! Implements the fluid think→act→observe loop where the LLM decides which
//! tools to call at each turn. This replaces the rigid Architect→Editor pipeline
//! for default chat mode.
//!
//! The loop continues until:
//! - The LLM responds with text only (no tool calls) — `finish_reason == "stop"`
//! - Maximum turns reached
//! - Context window overflow after compaction
//! - Unrecoverable error

use anyhow::{Result, anyhow};
use deepseek_core::{
    ApprovedToolCall, ChatMessage, ChatRequest, EventKind, LlmToolCall, StreamCallback,
    StreamChunk, ToolChoice, ToolDefinition, ToolHost, TokenUsage, UserQuestion,
    estimate_message_tokens, strip_prior_reasoning_content,
};
use deepseek_hooks::{HookEvent, HookInput, HookRuntime};
use deepseek_llm::LlmClient;
use std::sync::Arc;
use std::time::Instant;

use crate::tool_bridge;

/// Default maximum turns (LLM calls) before stopping the loop.
pub const DEFAULT_MAX_TURNS: usize = 50;

/// Context window usage percentage that triggers compaction.
pub const COMPACTION_THRESHOLD_PCT: f64 = 0.95;

/// Target context usage after compaction.
pub const COMPACTION_TARGET_PCT: f64 = 0.80;

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
pub type ApprovalCallback = Arc<dyn Fn(&deepseek_core::ToolCall) -> Result<bool> + Send + Sync>;

/// Callback for asking the user a question during tool execution.
pub type UserQuestionCallback = Arc<dyn Fn(UserQuestion) -> Option<String> + Send + Sync>;

/// Callback for event logging (tool proposed/result events).
pub type EventCallback = Arc<dyn Fn(EventKind) + Send + Sync>;

/// Callback for creating a checkpoint before destructive tool calls.
pub type CheckpointCallback = Arc<dyn Fn(&str) -> Result<()> + Send + Sync>;

/// Configuration for the tool-use loop.
pub struct ToolLoopConfig {
    pub model: String,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub context_window_tokens: u64,
    pub max_turns: usize,
    /// When true, use read-only tools only (Ask/Context mode).
    pub read_only: bool,
}

impl Default for ToolLoopConfig {
    fn default() -> Self {
        Self {
            model: "deepseek-chat".to_string(),
            max_tokens: 8192,
            temperature: None,
            context_window_tokens: 128_000,
            max_turns: DEFAULT_MAX_TURNS,
            read_only: false,
        }
    }
}

/// The core tool-use conversation loop.
///
/// Implements the think→act→observe pattern where the LLM freely decides
/// which tools to call at each step.
pub struct ToolUseLoop<'a> {
    llm: &'a (dyn LlmClient + Send + Sync),
    tool_host: Arc<dyn ToolHost + Send + Sync>,
    config: ToolLoopConfig,
    messages: Vec<ChatMessage>,
    tools: Vec<ToolDefinition>,
    stream_cb: Option<StreamCallback>,
    approval_cb: Option<ApprovalCallback>,
    user_question_cb: Option<UserQuestionCallback>,
    hooks: Option<HookRuntime>,
    event_cb: Option<EventCallback>,
    checkpoint_cb: Option<CheckpointCallback>,
}

impl<'a> ToolUseLoop<'a> {
    /// Create a new tool-use loop.
    pub fn new(
        llm: &'a (dyn LlmClient + Send + Sync),
        tool_host: Arc<dyn ToolHost + Send + Sync>,
        config: ToolLoopConfig,
        system_prompt: String,
        tools: Vec<ToolDefinition>,
    ) -> Self {
        let messages = vec![ChatMessage::System {
            content: system_prompt,
        }];

        Self {
            llm,
            tool_host,
            config,
            messages,
            tools,
            stream_cb: None,
            approval_cb: None,
            user_question_cb: None,
            hooks: None,
            event_cb: None,
            checkpoint_cb: None,
        }
    }

    /// Set the stream callback for real-time UI updates.
    pub fn set_stream_callback(&mut self, cb: StreamCallback) {
        self.stream_cb = Some(cb);
    }

    /// Set the approval callback for tool permission prompts.
    pub fn set_approval_callback(&mut self, cb: ApprovalCallback) {
        self.approval_cb = Some(cb);
    }

    /// Set the user question callback for interactive prompts.
    pub fn set_user_question_callback(&mut self, cb: UserQuestionCallback) {
        self.user_question_cb = Some(cb);
    }

    /// Set the hook runtime for pre/post tool-use hooks.
    pub fn set_hooks(&mut self, hooks: HookRuntime) {
        self.hooks = Some(hooks);
    }

    /// Set the event callback for logging tool proposed/result events.
    pub fn set_event_callback(&mut self, cb: EventCallback) {
        self.event_cb = Some(cb);
    }

    /// Set the checkpoint callback invoked before destructive tool calls.
    pub fn set_checkpoint_callback(&mut self, cb: CheckpointCallback) {
        self.checkpoint_cb = Some(cb);
    }

    /// Initialize from existing conversation history (for multi-turn).
    pub fn with_history(mut self, history: Vec<ChatMessage>) -> Self {
        // Insert history after system message
        for msg in history {
            self.messages.push(msg);
        }
        self
    }

    /// Run the loop with a user message.
    pub fn run(&mut self, user_message: &str) -> Result<ToolLoopResult> {
        self.messages.push(ChatMessage::User {
            content: user_message.to_string(),
        });

        self.execute_loop()
    }

    /// Continue the conversation with additional user input (multi-turn).
    pub fn continue_with(&mut self, user_message: &str) -> Result<ToolLoopResult> {
        // Strip reasoning from prior turns before adding new user message
        strip_prior_reasoning_content(&mut self.messages);

        self.messages.push(ChatMessage::User {
            content: user_message.to_string(),
        });

        self.execute_loop()
    }

    /// The main loop: call LLM, execute tools, feed results back, repeat.
    fn execute_loop(&mut self) -> Result<ToolLoopResult> {
        let mut tool_calls_made = Vec::new();
        let mut total_usage = TokenUsage::default();
        let mut turns: usize = 0;

        loop {
            // Check turn limit
            if turns >= self.config.max_turns {
                self.emit(StreamChunk::Done {
                    reason: Some("max turns reached".to_string()),
                });
                return Ok(ToolLoopResult {
                    response: String::new(),
                    tool_calls_made,
                    finish_reason: "max_turns".to_string(),
                    usage: total_usage,
                    turns,
                    messages: self.messages.clone(),
                });
            }

            // Check context window usage
            let estimated_tokens = estimate_message_tokens(&self.messages);
            let threshold = (self.config.context_window_tokens as f64 * COMPACTION_THRESHOLD_PCT) as u64;
            if estimated_tokens > threshold {
                // Try to compact
                let target = (self.config.context_window_tokens as f64 * COMPACTION_TARGET_PCT) as u64;
                let compacted = self.compact_messages(target);
                if !compacted {
                    self.emit(StreamChunk::Done {
                        reason: Some("context window full".to_string()),
                    });
                    return Ok(ToolLoopResult {
                        response: "Context window is full. Please start a new conversation or use /compact.".to_string(),
                        tool_calls_made,
                        finish_reason: "context_overflow".to_string(),
                        usage: total_usage,
                        turns,
                        messages: self.messages.clone(),
                    });
                }
            }

            // Build and send the LLM request
            turns += 1;
            let request = self.build_request();
            let response = if let Some(ref cb) = self.stream_cb {
                self.llm.complete_chat_streaming(&request, cb.clone())?
            } else {
                self.llm.complete_chat(&request)?
            };

            // Accumulate usage
            if let Some(ref usage) = response.usage {
                total_usage.prompt_tokens += usage.prompt_tokens;
                total_usage.completion_tokens += usage.completion_tokens;
                total_usage.prompt_cache_hit_tokens += usage.prompt_cache_hit_tokens;
                total_usage.prompt_cache_miss_tokens += usage.prompt_cache_miss_tokens;
                total_usage.reasoning_tokens += usage.reasoning_tokens;
            }

            // Handle content filter before anything else
            if response.finish_reason == "content_filter" {
                return Err(anyhow!("Response blocked by content filter"));
            }

            // No tool calls → return the text response
            if response.tool_calls.is_empty() {
                let text = response.text.clone();

                // Append assistant message to history
                self.messages.push(ChatMessage::Assistant {
                    content: Some(text.clone()),
                    reasoning_content: if response.reasoning_content.is_empty() {
                        None
                    } else {
                        Some(response.reasoning_content.clone())
                    },
                    tool_calls: vec![],
                });

                self.emit(StreamChunk::Done { reason: None });
                return Ok(ToolLoopResult {
                    response: text,
                    tool_calls_made,
                    finish_reason: response.finish_reason.clone(),
                    usage: total_usage,
                    turns,
                    messages: self.messages.clone(),
                });
            }

            // Tool calls present — execute them
            // First, append the assistant message with tool_calls
            self.messages.push(ChatMessage::Assistant {
                content: if response.text.is_empty() {
                    None
                } else {
                    Some(response.text.clone())
                },
                reasoning_content: if response.reasoning_content.is_empty() {
                    None
                } else {
                    Some(response.reasoning_content.clone())
                },
                tool_calls: response.tool_calls.clone(),
            });

            // Execute each tool call and collect results
            for llm_call in &response.tool_calls {
                let records = self.execute_tool_call(llm_call)?;
                tool_calls_made.extend(records);
            }

            // Strip reasoning from prior turns for DeepSeek API compliance
            strip_prior_reasoning_content(&mut self.messages);
        }
    }

    /// Execute a single tool call, handling approval flow, hooks, events, and checkpoints.
    fn execute_tool_call(
        &mut self,
        llm_call: &LlmToolCall,
    ) -> Result<Vec<ToolCallRecord>> {
        let mut records = Vec::new();
        let start = Instant::now();

        // Check if this is an agent-level tool that needs special handling
        if tool_bridge::is_agent_level_tool(&llm_call.name) {
            return self.handle_agent_level_tool(llm_call);
        }

        // Convert LLM call to internal format
        let tool_call = tool_bridge::llm_tool_call_to_internal(llm_call);

        let args_summary = summarize_args(&tool_call.args);
        self.emit(StreamChunk::ToolCallStart {
            tool_name: llm_call.name.clone(),
            args_summary: args_summary.clone(),
        });

        // Fire PreToolUse hook
        if let Some(ref hooks) = self.hooks {
            let input = HookInput {
                event: HookEvent::PreToolUse.as_str().to_string(),
                tool_name: Some(llm_call.name.clone()),
                tool_input: Some(tool_call.args.clone()),
                tool_result: None,
                prompt: None,
                session_type: None,
                workspace: String::new(),
            };
            let hook_result = hooks.fire(HookEvent::PreToolUse, &input);
            if hook_result.blocked {
                let duration = start.elapsed().as_millis() as u64;
                self.emit(StreamChunk::ToolCallEnd {
                    tool_name: llm_call.name.clone(),
                    duration_ms: duration,
                    success: false,
                    summary: "blocked by hook".to_string(),
                });
                self.messages.push(tool_bridge::tool_error_to_message(
                    &llm_call.id,
                    "Tool call blocked by pre-tool-use hook. Try a different approach.",
                ));
                records.push(ToolCallRecord {
                    tool_name: llm_call.name.clone(),
                    tool_call_id: llm_call.id.clone(),
                    args_summary,
                    success: false,
                    duration_ms: duration,
                });
                return Ok(records);
            }
        }

        // Log ToolProposedV1 event
        if let Some(ref cb) = self.event_cb {
            cb(EventKind::ToolProposedV1 {
                proposal: deepseek_core::ToolProposal {
                    invocation_id: uuid::Uuid::nil(),
                    call: tool_call.clone(),
                    approved: false,
                },
            });
        }

        // Propose the tool call (checks policy)
        let proposal = self.tool_host.propose(tool_call);

        if !proposal.approved {
            // Needs user approval
            let approved = if let Some(ref cb) = self.approval_cb {
                cb(&proposal.call)?
            } else {
                // No approval handler — deny by default in non-interactive mode
                false
            };

            if !approved {
                let duration = start.elapsed().as_millis() as u64;
                self.emit(StreamChunk::ToolCallEnd {
                    tool_name: llm_call.name.clone(),
                    duration_ms: duration,
                    success: false,
                    summary: "denied by user".to_string(),
                });
                // Feed denial back to LLM
                self.messages.push(tool_bridge::tool_error_to_message(
                    &llm_call.id,
                    "Tool call denied by user. Try a different approach or ask the user for guidance.",
                ));
                records.push(ToolCallRecord {
                    tool_name: llm_call.name.clone(),
                    tool_call_id: llm_call.id.clone(),
                    args_summary,
                    success: false,
                    duration_ms: duration,
                });
                return Ok(records);
            }
        }

        // Create checkpoint before destructive tool calls
        if tool_bridge::is_write_tool(&llm_call.name)
            && let Some(ref cp) = self.checkpoint_cb
        {
            let _ = cp("before tool execution");
        }

        // Execute the approved tool
        let approved_call = ApprovedToolCall {
            invocation_id: proposal.invocation_id,
            call: proposal.call,
        };
        let result = self.tool_host.execute(approved_call);

        let duration = start.elapsed().as_millis() as u64;
        let success = result.success;

        // Fire PostToolUse hook
        if let Some(ref hooks) = self.hooks {
            let input = HookInput {
                event: HookEvent::PostToolUse.as_str().to_string(),
                tool_name: Some(llm_call.name.clone()),
                tool_input: Some(serde_json::json!({})),
                tool_result: Some(result.output.clone()),
                prompt: None,
                session_type: None,
                workspace: String::new(),
            };
            let _ = hooks.fire(HookEvent::PostToolUse, &input);
        }

        // Log ToolResultV1 event
        if let Some(ref cb) = self.event_cb {
            cb(EventKind::ToolResultV1 {
                result: deepseek_core::ToolResult {
                    invocation_id: result.invocation_id,
                    success: result.success,
                    output: result.output.clone(),
                },
            });
        }

        self.emit(StreamChunk::ToolCallEnd {
            tool_name: llm_call.name.clone(),
            duration_ms: duration,
            success,
            summary: if success {
                "ok".to_string()
            } else {
                "error".to_string()
            },
        });

        // Convert result to ChatMessage::Tool and append
        let msg = tool_bridge::tool_result_to_message(&llm_call.id, &result);
        self.messages.push(msg);

        records.push(ToolCallRecord {
            tool_name: llm_call.name.clone(),
            tool_call_id: llm_call.id.clone(),
            args_summary,
            success,
            duration_ms: duration,
        });

        Ok(records)
    }

    /// Handle agent-level tools (user_question, spawn_task, think_deeply, etc.)
    fn handle_agent_level_tool(
        &mut self,
        llm_call: &LlmToolCall,
    ) -> Result<Vec<ToolCallRecord>> {
        let start = Instant::now();
        let args: serde_json::Value = serde_json::from_str(&llm_call.arguments)
            .unwrap_or(serde_json::json!({}));

        let args_summary = summarize_args(&args);

        self.emit(StreamChunk::ToolCallStart {
            tool_name: llm_call.name.clone(),
            args_summary: args_summary.clone(),
        });

        let result_content = match llm_call.name.as_str() {
            "think_deeply" => {
                // Call the reasoner model for deep analysis
                let question = args.get("question").and_then(|v| v.as_str()).unwrap_or("");
                let context = args.get("context").and_then(|v| v.as_str()).unwrap_or("");
                self.handle_think_deeply(question, context)?
            }
            "user_question" => {
                let question = args.get("question").and_then(|v| v.as_str()).unwrap_or("(no question)");
                let options: Vec<String> = args.get("options")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                if let Some(ref cb) = self.user_question_cb {
                    let uq = UserQuestion { question: question.to_string(), options };
                    match cb(uq) {
                        Some(answer) => format!("User response: {answer}"),
                        None => format!("Question for user: {question}\n[User cancelled. Proceed with your best judgment.]"),
                    }
                } else {
                    format!("Question for user: {question}\n[User response not available in this context. Proceed with your best judgment or state your assumption.]")
                }
            }
            _ => {
                // Other agent-level tools (spawn_task, skill, etc.) — not yet wired
                format!("Agent-level tool '{}' is not yet available in tool-use mode. Try a different approach.", llm_call.name)
            }
        };

        let duration = start.elapsed().as_millis() as u64;
        self.emit(StreamChunk::ToolCallEnd {
            tool_name: llm_call.name.clone(),
            duration_ms: duration,
            success: true,
            summary: "ok".to_string(),
        });

        self.messages.push(ChatMessage::Tool {
            tool_call_id: llm_call.id.clone(),
            content: result_content,
        });

        Ok(vec![ToolCallRecord {
            tool_name: llm_call.name.clone(),
            tool_call_id: llm_call.id.clone(),
            args_summary,
            success: true,
            duration_ms: duration,
        }])
    }

    /// Handle the think_deeply tool by calling the reasoner model.
    fn handle_think_deeply(&self, question: &str, context: &str) -> Result<String> {
        let prompt = format!(
            "Analyze this problem carefully:\n\nQuestion: {question}\n\nContext:\n{context}\n\nProvide a clear, actionable recommendation."
        );
        let request = ChatRequest {
            model: "deepseek-reasoner".to_string(),
            messages: vec![
                ChatMessage::System {
                    content: "You are a deep reasoning engine. Provide thorough analysis and clear recommendations.".to_string(),
                },
                ChatMessage::User { content: prompt },
            ],
            tools: vec![],
            tool_choice: ToolChoice::none(),
            max_tokens: 4096,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            logprobs: None,
            top_logprobs: None,
            thinking: None,
            images: vec![],
            response_format: None,
        };
        let response = self.llm.complete_chat(&request)?;
        Ok(response.text)
    }

    /// Build a ChatRequest from current state.
    fn build_request(&self) -> ChatRequest {
        let tools = if self.config.read_only {
            self.tools
                .iter()
                .filter(|t| is_read_only_api_name(&t.function.name))
                .cloned()
                .collect()
        } else {
            self.tools.clone()
        };

        ChatRequest {
            model: self.config.model.clone(),
            messages: self.messages.clone(),
            tools,
            tool_choice: ToolChoice::auto(),
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            logprobs: None,
            top_logprobs: None,
            thinking: None,
            images: vec![],
            response_format: None,
        }
    }

    /// Simple message compaction: remove middle messages keeping system + recent.
    /// Returns true if compaction happened, false if nothing could be compacted.
    fn compact_messages(&mut self, _target_tokens: u64) -> bool {
        // Keep system message (index 0) and last 6 messages
        if self.messages.len() <= 7 {
            return false; // Nothing to compact
        }

        let system = self.messages[0].clone();
        let keep_recent = 6;
        let recent_start = self.messages.len() - keep_recent;
        let recent: Vec<ChatMessage> = self.messages[recent_start..].to_vec();

        // Summarize the middle section into a single system-like message
        let middle_count = recent_start - 1; // everything between system and recent
        let summary_msg = ChatMessage::User {
            content: format!(
                "[Conversation compacted: {middle_count} earlier messages were summarized. \
                 Continue from the recent context below.]"
            ),
        };

        self.messages = vec![system, summary_msg];
        self.messages.extend(recent);
        true
    }

    /// Emit a stream chunk to the callback.
    fn emit(&self, chunk: StreamChunk) {
        if let Some(ref cb) = self.stream_cb {
            cb(chunk);
        }
    }
}

/// Check if a tool API name is read-only.
fn is_read_only_api_name(name: &str) -> bool {
    matches!(
        name,
        "fs_read"
            | "fs_glob"
            | "fs_grep"
            | "fs_list"
            | "git_status"
            | "git_diff"
            | "git_show"
            | "web_search"
            | "web_fetch"
            | "notebook_read"
            | "index_query"
            | "think_deeply"
            | "user_question"
            | "spawn_task"
            | "task_output"
            | "task_list"
            | "task_get"
            | "diagnostics_check"
    )
}

/// Produce a short summary of tool arguments for display.
fn summarize_args(args: &serde_json::Value) -> String {
    let mut parts = Vec::new();
    if let Some(obj) = args.as_object() {
        for (key, val) in obj {
            let short = match val {
                serde_json::Value::String(s) => {
                    if s.len() > 60 {
                        format!("{key}=\"{}...\"", &s[..57])
                    } else {
                        format!("{key}=\"{s}\"")
                    }
                }
                serde_json::Value::Number(n) => format!("{key}={n}"),
                serde_json::Value::Bool(b) => format!("{key}={b}"),
                _ => format!("{key}=..."),
            };
            parts.push(short);
        }
    }
    if parts.is_empty() {
        return "()".to_string();
    }
    parts.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepseek_core::{LlmResponse, ToolCall, ToolProposal, ToolResult};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    // ── Scripted LLM mock ──

    struct ScriptedLlm {
        responses: Mutex<VecDeque<LlmResponse>>,
    }

    impl ScriptedLlm {
        fn new(responses: Vec<LlmResponse>) -> Self {
            Self {
                responses: Mutex::new(VecDeque::from(responses)),
            }
        }
    }

    impl LlmClient for ScriptedLlm {
        fn complete(&self, _req: &deepseek_core::LlmRequest) -> Result<LlmResponse> {
            unimplemented!()
        }
        fn complete_streaming(
            &self,
            _req: &deepseek_core::LlmRequest,
            _cb: StreamCallback,
        ) -> Result<LlmResponse> {
            unimplemented!()
        }
        fn complete_chat(&self, _req: &ChatRequest) -> Result<LlmResponse> {
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| anyhow!("no more scripted responses"))
        }
        fn complete_chat_streaming(
            &self,
            req: &ChatRequest,
            _cb: StreamCallback,
        ) -> Result<LlmResponse> {
            self.complete_chat(req)
        }
        fn complete_fim(&self, _req: &deepseek_core::FimRequest) -> Result<LlmResponse> {
            unimplemented!()
        }
        fn complete_fim_streaming(
            &self,
            _req: &deepseek_core::FimRequest,
            _cb: StreamCallback,
        ) -> Result<LlmResponse> {
            unimplemented!()
        }
    }

    // ── Scripted tool host mock ──

    struct MockToolHost {
        results: Mutex<VecDeque<ToolResult>>,
        auto_approve: bool,
    }

    impl MockToolHost {
        fn new(results: Vec<ToolResult>, auto_approve: bool) -> Self {
            Self {
                results: Mutex::new(VecDeque::from(results)),
                auto_approve,
            }
        }
    }

    impl ToolHost for MockToolHost {
        fn propose(&self, call: ToolCall) -> ToolProposal {
            ToolProposal {
                invocation_id: uuid::Uuid::nil(),
                call,
                approved: self.auto_approve,
            }
        }
        fn execute(&self, _approved: ApprovedToolCall) -> ToolResult {
            self.results
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(ToolResult {
                    invocation_id: uuid::Uuid::nil(),
                    success: false,
                    output: serde_json::json!({"error": "no mock result"}),
                })
        }
    }

    fn make_text_response(text: &str) -> LlmResponse {
        LlmResponse {
            text: text.to_string(),
            finish_reason: "stop".to_string(),
            reasoning_content: String::new(),
            tool_calls: vec![],
            usage: Some(TokenUsage {
                prompt_tokens: 100,
                completion_tokens: 50,
                ..Default::default()
            }),
        }
    }

    fn make_tool_response(tool_calls: Vec<LlmToolCall>) -> LlmResponse {
        LlmResponse {
            text: String::new(),
            finish_reason: "tool_calls".to_string(),
            reasoning_content: String::new(),
            tool_calls,
            usage: Some(TokenUsage {
                prompt_tokens: 100,
                completion_tokens: 50,
                ..Default::default()
            }),
        }
    }

    fn default_tools() -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            tool_type: "function".to_string(),
            function: deepseek_core::FunctionDefinition {
                name: "fs_read".to_string(),
                description: "Read a file".to_string(),
                strict: None,
                parameters: serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
            },
        }]
    }

    // ── Tests ──

    #[test]
    fn simple_text_response_no_tools() {
        let llm = ScriptedLlm::new(vec![make_text_response("Hello, world!")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            ToolLoopConfig::default(),
            "You are helpful.".to_string(),
            default_tools(),
        );

        let result = loop_.run("Hi").unwrap();
        assert_eq!(result.response, "Hello, world!");
        assert_eq!(result.turns, 1);
        assert!(result.tool_calls_made.is_empty());
        assert_eq!(result.finish_reason, "stop");
    }

    #[test]
    fn single_tool_call_and_response() {
        let llm = ScriptedLlm::new(vec![
            // Turn 1: LLM calls fs_read
            make_tool_response(vec![LlmToolCall {
                id: "call_1".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path":"src/lib.rs"}"#.to_string(),
            }]),
            // Turn 2: LLM responds with text
            make_text_response("The file contains a module definition."),
        ]);

        let tool_host = Arc::new(MockToolHost::new(
            vec![ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: true,
                output: serde_json::json!({"content": "mod tests;"}),
            }],
            true,
        ));

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            ToolLoopConfig::default(),
            "You are helpful.".to_string(),
            default_tools(),
        );

        let result = loop_.run("What's in src/lib.rs?").unwrap();
        assert_eq!(result.response, "The file contains a module definition.");
        assert_eq!(result.turns, 2);
        assert_eq!(result.tool_calls_made.len(), 1);
        assert_eq!(result.tool_calls_made[0].tool_name, "fs_read");
        assert!(result.tool_calls_made[0].success);
    }

    #[test]
    fn multi_tool_calls_parallel() {
        let llm = ScriptedLlm::new(vec![
            // Turn 1: LLM calls two tools in parallel
            make_tool_response(vec![
                LlmToolCall {
                    id: "call_1".to_string(),
                    name: "fs_read".to_string(),
                    arguments: r#"{"path":"a.rs"}"#.to_string(),
                },
                LlmToolCall {
                    id: "call_2".to_string(),
                    name: "fs_read".to_string(),
                    arguments: r#"{"path":"b.rs"}"#.to_string(),
                },
            ]),
            // Turn 2: text response
            make_text_response("Both files processed."),
        ]);

        let tool_host = Arc::new(MockToolHost::new(
            vec![
                ToolResult {
                    invocation_id: uuid::Uuid::nil(),
                    success: true,
                    output: serde_json::json!("file a"),
                },
                ToolResult {
                    invocation_id: uuid::Uuid::nil(),
                    success: true,
                    output: serde_json::json!("file b"),
                },
            ],
            true,
        ));

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            ToolLoopConfig::default(),
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("Read both files").unwrap();
        assert_eq!(result.turns, 2);
        assert_eq!(result.tool_calls_made.len(), 2);
    }

    #[test]
    fn tool_chain_multiple_turns() {
        let llm = ScriptedLlm::new(vec![
            // Turn 1: grep
            make_tool_response(vec![LlmToolCall {
                id: "c1".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path":"x.rs"}"#.to_string(),
            }]),
            // Turn 2: read the found file
            make_tool_response(vec![LlmToolCall {
                id: "c2".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path":"y.rs"}"#.to_string(),
            }]),
            // Turn 3: final answer
            make_text_response("Found the implementation in y.rs."),
        ]);

        let tool_host = Arc::new(MockToolHost::new(
            vec![
                ToolResult {
                    invocation_id: uuid::Uuid::nil(),
                    success: true,
                    output: serde_json::json!("result 1"),
                },
                ToolResult {
                    invocation_id: uuid::Uuid::nil(),
                    success: true,
                    output: serde_json::json!("result 2"),
                },
            ],
            true,
        ));

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            ToolLoopConfig::default(),
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("Find auth code").unwrap();
        assert_eq!(result.turns, 3);
        assert_eq!(result.tool_calls_made.len(), 2);
        assert_eq!(result.response, "Found the implementation in y.rs.");
    }

    #[test]
    fn max_turns_limit_stops_loop() {
        // LLM always returns tool calls, never stops
        let infinite_tool_calls: Vec<LlmResponse> = (0..5)
            .map(|i| {
                make_tool_response(vec![LlmToolCall {
                    id: format!("c{i}"),
                    name: "fs_read".to_string(),
                    arguments: r#"{"path":"test.rs"}"#.to_string(),
                }])
            })
            .collect();

        let llm = ScriptedLlm::new(infinite_tool_calls);
        let tool_host = Arc::new(MockToolHost::new(
            (0..5)
                .map(|_| ToolResult {
                    invocation_id: uuid::Uuid::nil(),
                    success: true,
                    output: serde_json::json!("ok"),
                })
                .collect(),
            true,
        ));

        let config = ToolLoopConfig {
            max_turns: 3,
            ..Default::default()
        };

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("loop forever").unwrap();
        assert_eq!(result.turns, 3);
        assert_eq!(result.finish_reason, "max_turns");
    }

    #[test]
    fn tool_call_denied_by_policy() {
        let llm = ScriptedLlm::new(vec![
            // Turn 1: LLM calls a tool
            make_tool_response(vec![LlmToolCall {
                id: "c1".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path":"secret.txt"}"#.to_string(),
            }]),
            // Turn 2: After denial, LLM responds differently
            make_text_response("I cannot access that file."),
        ]);

        // Tool host does NOT auto-approve
        let tool_host = Arc::new(MockToolHost::new(vec![], false));

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            ToolLoopConfig::default(),
            "system".to_string(),
            default_tools(),
        );

        // No approval callback → denied by default
        let result = loop_.run("Read secret file").unwrap();
        assert_eq!(result.turns, 2);
        assert_eq!(result.tool_calls_made.len(), 1);
        assert!(!result.tool_calls_made[0].success);
        assert_eq!(result.response, "I cannot access that file.");
    }

    #[test]
    fn content_filter_returns_error() {
        let llm = ScriptedLlm::new(vec![LlmResponse {
            text: String::new(),
            finish_reason: "content_filter".to_string(),
            reasoning_content: String::new(),
            tool_calls: vec![],
            usage: None,
        }]);

        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            ToolLoopConfig::default(),
            "system".to_string(),
            default_tools(),
        );

        let err = loop_.run("bad prompt").unwrap_err();
        assert!(err.to_string().contains("content filter"));
    }

    #[test]
    fn usage_accumulated_across_turns() {
        let llm = ScriptedLlm::new(vec![
            make_tool_response(vec![LlmToolCall {
                id: "c1".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path":"x"}"#.to_string(),
            }]),
            make_text_response("done"),
        ]);

        let tool_host = Arc::new(MockToolHost::new(
            vec![ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: true,
                output: serde_json::json!("ok"),
            }],
            true,
        ));

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            ToolLoopConfig::default(),
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("test").unwrap();
        // Each response has 100 prompt + 50 completion, 2 turns
        assert_eq!(result.usage.prompt_tokens, 200);
        assert_eq!(result.usage.completion_tokens, 100);
    }

    #[test]
    fn summarize_args_formats_correctly() {
        let args = serde_json::json!({"path": "src/lib.rs", "start_line": 10});
        let summary = summarize_args(&args);
        assert!(summary.contains("path=\"src/lib.rs\""));
        assert!(summary.contains("start_line=10"));
    }

    #[test]
    fn continue_with_appends_to_history() {
        let llm = ScriptedLlm::new(vec![
            make_text_response("First answer"),
            make_text_response("Second answer"),
        ]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            ToolLoopConfig::default(),
            "system".to_string(),
            default_tools(),
        );

        let r1 = loop_.run("first question").unwrap();
        assert_eq!(r1.response, "First answer");

        let r2 = loop_.continue_with("follow up").unwrap();
        assert_eq!(r2.response, "Second answer");
        // Messages should contain: system, user1, assistant1, user2, assistant2
        assert_eq!(r2.messages.len(), 5);
    }
}
