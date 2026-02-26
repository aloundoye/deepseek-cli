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
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::tool_bridge;

/// Default maximum turns (LLM calls) before stopping the loop.
pub const DEFAULT_MAX_TURNS: usize = 50;

/// Context window usage percentage that triggers compaction.
pub const COMPACTION_THRESHOLD_PCT: f64 = 0.95;

/// Target context usage after compaction.
pub const COMPACTION_TARGET_PCT: f64 = 0.80;

/// If the model's first response (with no tool calls) exceeds this many
/// characters, it's likely hallucinating a long answer instead of using tools.
/// We inject a nudge to get it back on track.
const HALLUCINATION_NUDGE_THRESHOLD: usize = 300;

/// Every N tool calls, inject a brief system reminder to keep the model on track.
const MID_CONVERSATION_REMINDER_INTERVAL: usize = 10;

/// Brief reminder injected every `MID_CONVERSATION_REMINDER_INTERVAL` tool calls.
const MID_CONVERSATION_REMINDER: &str = "Reminder: Verify changes with tests. Be concise. Use tools — do not guess.";

/// Message injected when the model tries to answer a question about the codebase
/// without using any tools first.
const HALLUCINATION_NUDGE: &str = "STOP. You are answering without using tools. \
Your response likely contains fabricated information. You MUST use tools first. \
Start with fs_list or fs_glob to explore the project, then fs_read for specific files. \
Do NOT guess or synthesize answers from memory.";

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
pub type SkillRunner = Arc<dyn Fn(&str, Option<&str>) -> Result<Option<SkillInvocationResult>> + Send + Sync>;

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
    pub thinking: Option<deepseek_core::ThinkingConfig>,
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
}

impl Default for ToolLoopConfig {
    fn default() -> Self {
        Self {
            model: deepseek_core::DEEPSEEK_V32_CHAT_MODEL.to_string(),
            max_tokens: deepseek_core::DEEPSEEK_CHAT_THINKING_MAX_OUTPUT_TOKENS,
            temperature: None,
            context_window_tokens: 128_000,
            max_turns: DEFAULT_MAX_TURNS,
            read_only: false,
            thinking: Some(deepseek_core::ThinkingConfig::enabled(
                crate::complexity::MEDIUM_THINK_BUDGET,
            )),
            extended_thinking_model: deepseek_core::DEEPSEEK_V32_REASONER_MODEL.to_string(),
            complexity: crate::complexity::PromptComplexity::Medium,
            subagent_worker: None,
            skill_runner: None,
            workspace: None,
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
    /// Evidence-driven escalation signals from tool outputs.
    /// When the model hits compile errors, test failures, or patch rejections,
    /// the thinking budget automatically escalates.
    escalation: crate::complexity::EscalationSignals,
    /// Pre-compiled output scanner for injection/secret detection.
    output_scanner: deepseek_policy::output_scanner::OutputScanner,
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
            escalation: crate::complexity::EscalationSignals::default(),
            output_scanner: deepseek_policy::output_scanner::OutputScanner::new(),
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

            // No tool calls → return the text response (or nudge to use tools)
            if response.tool_calls.is_empty() {
                let text = response.text.clone();

                // Anti-hallucination guard: if this is the first turn and the
                // response is long (likely a fabricated analysis), nudge the
                // model to use tools instead of guessing.
                let should_nudge = turns == 1
                    && tool_calls_made.is_empty()
                    && !self.config.read_only
                    && (text.len() > HALLUCINATION_NUDGE_THRESHOLD
                        || has_unverified_file_references(&text, &tool_calls_made));

                if should_nudge {
                    // Don't emit this attempt — inject a nudge and retry
                    self.messages.push(ChatMessage::Assistant {
                        content: Some(text),
                        reasoning_content: None,
                        tool_calls: vec![],
                    });
                    self.messages.push(ChatMessage::User {
                        content: HALLUCINATION_NUDGE.to_string(),
                    });
                    continue;
                }

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

            // Execute each tool call and collect results.
            // Scan tool outputs for evidence-driven budget escalation.
            let mut batch_had_failure = false;
            let mut batch_had_success = false;
            for llm_call in &response.tool_calls {
                let records = self.execute_tool_call(llm_call)?;
                for r in &records {
                    if r.success {
                        batch_had_success = true;
                    } else {
                        batch_had_failure = true;
                    }
                }
                tool_calls_made.extend(records);
            }

            // Update escalation signals based on batch outcome
            if batch_had_success {
                self.escalation.record_success();
            } else if batch_had_failure {
                self.escalation.record_failure();
            }

            // Mid-conversation reminder: every N tool calls, inject a brief
            // system-like nudge to keep the model focused.
            if !tool_calls_made.is_empty()
                && tool_calls_made.len() % MID_CONVERSATION_REMINDER_INTERVAL == 0
            {
                self.messages.push(ChatMessage::User {
                    content: MID_CONVERSATION_REMINDER.to_string(),
                });
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

        // Log ToolProposed event
        if let Some(ref cb) = self.event_cb {
            cb(EventKind::ToolProposed {
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
            let modified = tool_bridge::extract_modified_paths(&llm_call.name, &llm_call.arguments);
            let _ = cp("before tool execution", &modified);
        }

        // Execute the approved tool
        let approved_call = ApprovedToolCall {
            invocation_id: proposal.invocation_id,
            call: proposal.call,
        };
        let result = self.tool_host.execute(approved_call);

        let duration = start.elapsed().as_millis() as u64;
        let success = result.success;

        // Scan tool output for evidence-driven budget escalation signals
        // (compile errors, test failures, patch rejections, search misses).
        if let Some(output_str) = result.output.as_str() {
            self.escalation.scan_tool_output(&llm_call.name, output_str);
        } else if let Ok(output_text) = serde_json::to_string(&result.output) {
            self.escalation.scan_tool_output(&llm_call.name, &output_text);
        }

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

        // Log ToolResult event
        if let Some(ref cb) = self.event_cb {
            cb(EventKind::ToolResult {
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

        // Convert result to ChatMessage::Tool and append, scanning for security issues
        let (msg, injection_warnings) = tool_bridge::tool_result_to_message(
            &llm_call.id,
            &llm_call.name,
            &result,
            Some(&self.output_scanner),
        );
        self.messages.push(msg);

        // Emit security warnings to user
        for warning in &injection_warnings {
            self.emit(deepseek_core::StreamChunk::SecurityWarning {
                message: format!(
                    "{:?} — {} (matched: {})",
                    warning.severity, warning.pattern_name, warning.matched_text
                ),
            });
        }

        records.push(ToolCallRecord {
            tool_name: llm_call.name.clone(),
            tool_call_id: llm_call.id.clone(),
            args_summary,
            success,
            duration_ms: duration,
        });

        Ok(records)
    }

    /// Handle agent-level tools (user_question, spawn_task, extended_thinking, etc.)
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
            "extended_thinking" | "think_deeply" => {
                // Call the reasoner model for deep analysis
                let question = args.get("question").and_then(|v| v.as_str()).unwrap_or("");
                let context = args.get("context").and_then(|v| v.as_str()).unwrap_or("");
                self.handle_extended_thinking(question, context)?
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
            "spawn_task" => {
                self.handle_spawn_task(&args)?
            }
            "skill" => {
                self.handle_skill(&args)?
            }
            _ => {
                // Other agent-level tools (task_*, etc.) — not yet wired
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

    /// Handle the extended_thinking tool by calling the reasoner model.
    fn handle_extended_thinking(&self, question: &str, context: &str) -> Result<String> {
        let prompt = format!(
            "Analyze this problem carefully:\n\nQuestion: {question}\n\nContext:\n{context}\n\nProvide a clear, actionable recommendation."
        );
        let request = ChatRequest {
            model: self.config.extended_thinking_model.clone(),
            messages: vec![
                ChatMessage::System {
                    content: "You are a deep reasoning engine. Provide thorough analysis and clear recommendations.".to_string(),
                },
                ChatMessage::User { content: prompt },
            ],
            tools: vec![],
            tool_choice: ToolChoice::none(),
            max_tokens: deepseek_core::DEEPSEEK_REASONER_MAX_OUTPUT_TOKENS,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            logprobs: None,
            top_logprobs: None,
            thinking: Some(deepseek_core::ThinkingConfig::enabled(16_384)),
            images: vec![],
            response_format: None,
        };
        let response = self.llm.complete_chat(&request)?;
        Ok(response.text)
    }

    /// Handle the spawn_task tool by delegating to the subagent worker.
    fn handle_spawn_task(&self, args: &serde_json::Value) -> Result<String> {
        let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
        let task_name = args.get("description")
            .or_else(|| args.get("task_name"))
            .and_then(|v| v.as_str())
            .unwrap_or("subtask");
        let subagent_type = args.get("subagent_type")
            .and_then(|v| v.as_str())
            .unwrap_or("general-purpose");
        let model_override = args.get("model").and_then(|v| v.as_str()).map(String::from);
        let max_turns = args.get("max_turns").and_then(|v| v.as_u64()).map(|n| n as usize);
        let run_in_background = args.get("run_in_background").and_then(|v| v.as_bool()).unwrap_or(false);

        if prompt.is_empty() {
            return Ok("Error: spawn_task requires a 'prompt' argument describing the task.".to_string());
        }

        let request = SubagentRequest {
            prompt: prompt.to_string(),
            task_name: task_name.to_string(),
            subagent_type: subagent_type.to_string(),
            model_override,
            max_turns,
            run_in_background,
        };

        if let Some(ref worker) = self.config.subagent_worker {
            // P5-14: Fire SubagentStart hook
            if let Some(ref hooks) = self.hooks {
                let input = HookInput {
                    event: HookEvent::SubagentStart.as_str().to_string(),
                    tool_name: Some("spawn_task".to_string()),
                    tool_input: Some(args.clone()),
                    tool_result: None,
                    prompt: Some(prompt.to_string()),
                    session_type: None,
                    workspace: self.config.workspace.as_deref().unwrap_or(std::path::Path::new(".")).display().to_string(),
                };
                let _ = hooks.fire(HookEvent::SubagentStart, &input);
            }
            let result = match worker(request) {
                Ok(result) => Ok(result),
                Err(e) => Ok(format!("Subagent '{task_name}' failed: {e}. Try a different approach or handle the task directly.")),
            };
            // P5-14: Fire SubagentStop hook
            if let Some(ref hooks) = self.hooks {
                let input = HookInput {
                    event: HookEvent::SubagentStop.as_str().to_string(),
                    tool_name: Some("spawn_task".to_string()),
                    tool_input: Some(args.clone()),
                    tool_result: result.as_ref().ok().map(|r| serde_json::Value::String(r.clone())),
                    prompt: Some(prompt.to_string()),
                    session_type: None,
                    workspace: self.config.workspace.as_deref().unwrap_or(std::path::Path::new(".")).display().to_string(),
                };
                let _ = hooks.fire(HookEvent::SubagentStop, &input);
            }
            result
        } else {
            // No worker wired — run inline by providing guidance
            Ok(format!(
                "spawn_task is not available in this context. Handle the task directly instead.\n\
                 Task: {task_name}\n\
                 Prompt: {prompt}\n\
                 Use the available tools to accomplish this yourself."
            ))
        }
    }

    /// Handle the skill tool invocation.
    ///
    /// Looks up the skill, checks `disable_model_invocation`, and either:
    /// - For `context: fork`: delegates to subagent_worker for isolated execution
    /// - For `context: normal`: returns the rendered prompt for inline injection
    fn handle_skill(&self, args: &serde_json::Value) -> Result<String> {
        let skill_name = args
            .get("skill")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let skill_args = args.get("args").and_then(|v| v.as_str());

        if skill_name.is_empty() {
            return Ok(
                "Error: skill tool requires a 'skill' argument with the skill name.".to_string(),
            );
        }

        let Some(ref runner) = self.config.skill_runner else {
            return Ok(format!(
                "Skill '{skill_name}' is not available in this context. \
                 Skills require the CLI environment to be properly initialized."
            ));
        };

        let result = match runner(skill_name, skill_args) {
            Ok(Some(result)) => result,
            Ok(None) => {
                return Ok(format!(
                    "Skill '{skill_name}' not found. Check available skills with /skills list."
                ));
            }
            Err(e) => {
                return Ok(format!("Failed to load skill '{skill_name}': {e}"));
            }
        };

        // P5-08: Respect disable-model-invocation flag
        if result.disable_model_invocation {
            return Ok(format!(
                "Skill '{skill_name}' has disable-model-invocation set. \
                 This skill can only be invoked directly by the user via /{skill_name}, \
                 not programmatically by the model."
            ));
        }

        // P5-06: Forked execution — delegate to subagent for isolated context
        if result.forked {
            if let Some(ref worker) = self.config.subagent_worker {
                let request = SubagentRequest {
                    prompt: result.rendered_prompt.clone(),
                    task_name: format!("skill:{skill_name}"),
                    subagent_type: "general-purpose".to_string(),
                    model_override: None,
                    max_turns: None,
                    run_in_background: false,
                };
                return match worker(request) {
                    Ok(output) => Ok(format!(
                        "Skill '{skill_name}' (forked) completed:\n{output}"
                    )),
                    Err(e) => Ok(format!(
                        "Skill '{skill_name}' (forked) failed: {e}. \
                         Try running the skill directly or handle the task yourself."
                    )),
                };
            }
            // No subagent worker — fall through to inline with a note
            return Ok(format!(
                "Skill '{skill_name}' requires forked execution but no subagent worker is available. \
                 Returning the skill prompt for inline execution:\n\n{}",
                result.rendered_prompt
            ));
        }

        // P5-07: Normal execution — return rendered prompt with tool restriction metadata
        // The tool restrictions (allowed_tools, disallowed_tools) are advisory here;
        // they would be enforced if this skill were run in a forked ToolUseLoop.
        let mut response = format!(
            "<skill-execution name=\"{skill_name}\">\n{}\n</skill-execution>",
            result.rendered_prompt
        );

        if !result.allowed_tools.is_empty() || !result.disallowed_tools.is_empty() {
            response.push_str("\n\nNote: This skill has tool restrictions. ");
            if !result.allowed_tools.is_empty() {
                response.push_str(&format!(
                    "Allowed tools: {}. ",
                    result.allowed_tools.join(", ")
                ));
            }
            if !result.disallowed_tools.is_empty() {
                response.push_str(&format!(
                    "Do NOT use: {}.",
                    result.disallowed_tools.join(", ")
                ));
            }
        }

        Ok(response)
    }

    /// Build a ChatRequest from current state.
    ///
    /// Applies dynamic thinking budget escalation when consecutive failures
    /// are detected — giving the model more reasoning power to self-correct.
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

        // Force tool use on first turn (no tool results yet) so the model
        // explores the codebase instead of fabricating an answer.
        let has_tool_results = self
            .messages
            .iter()
            .any(|m| matches!(m, ChatMessage::Tool { .. }));
        let tool_choice = if !has_tool_results && !self.config.read_only {
            ToolChoice::required()
        } else {
            ToolChoice::auto()
        };

        // Evidence-driven thinking budget: escalate when tool outputs show
        // compile errors, test failures, or repeated problems.
        let thinking = self.config.thinking.as_ref().map(|base| {
            if self.escalation.should_escalate() {
                deepseek_core::ThinkingConfig::enabled(self.escalation.budget())
            } else {
                base.clone()
            }
        });

        ChatRequest {
            model: self.config.model.clone(),
            messages: self.messages.clone(),
            tools,
            tool_choice,
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            logprobs: None,
            top_logprobs: None,
            thinking,
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
            | "extended_thinking"
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

/// Detect when the model mentions file paths without having called fs_read/fs_glob first.
/// Returns true if the text references paths that haven't been verified by tool calls.
fn has_unverified_file_references(text: &str, tool_calls_made: &[ToolCallRecord]) -> bool {
    // Extract path-like patterns from text (e.g. src/main.rs, ./config.json, crates/foo/bar.rs)
    let path_pattern = regex::Regex::new(r"\b[\w./\-]+\.\w{1,6}\b").unwrap_or_else(|_| return regex::Regex::new(r"^$").unwrap());
    let mentioned_paths: Vec<&str> = path_pattern
        .find_iter(text)
        .map(|m| m.as_str())
        .filter(|p| {
            // Filter to things that look like real file paths
            p.contains('/') || p.ends_with(".rs") || p.ends_with(".ts") || p.ends_with(".js")
                || p.ends_with(".py") || p.ends_with(".json") || p.ends_with(".toml")
                || p.ends_with(".yaml") || p.ends_with(".yml") || p.ends_with(".md")
        })
        .collect();

    if mentioned_paths.is_empty() {
        return false;
    }

    // Check if any read/glob/grep tool was called (the model at least tried to look)
    let used_read_tools = tool_calls_made.iter().any(|tc| {
        tc.tool_name == "fs_read" || tc.tool_name == "fs.read"
            || tc.tool_name == "fs_glob" || tc.tool_name == "fs.glob"
            || tc.tool_name == "fs_grep" || tc.tool_name == "fs.grep"
            || tc.tool_name == "fs_list" || tc.tool_name == "fs.list"
    });

    // If paths are mentioned but no read/search tools were used, flag it
    !used_read_tools && mentioned_paths.len() >= 2
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

    #[test]
    fn hallucination_nudge_triggers_on_long_first_response() {
        // First response: long text with no tools (hallucination)
        // Second response: shorter text (after nudge, model cooperates)
        let long_hallucination = "a".repeat(HALLUCINATION_NUDGE_THRESHOLD + 100);
        let llm = ScriptedLlm::new(vec![
            make_text_response(&long_hallucination),
            make_text_response("Let me check with tools."), // after nudge
        ]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            ToolLoopConfig::default(),
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("analyze this project").unwrap();
        // Should get the second (post-nudge) response, not the hallucinated one
        assert_eq!(result.response, "Let me check with tools.");
        // Should have used 2 turns
        assert_eq!(result.turns, 2);
        // Messages should contain the nudge
        let has_nudge = result.messages.iter().any(|m| {
            if let ChatMessage::User { content } = m {
                content.contains("STOP. You are answering without using tools")
            } else {
                false
            }
        });
        assert!(has_nudge, "should contain hallucination nudge message");
    }

    #[test]
    fn hallucination_nudge_skips_short_responses() {
        // Short response should NOT trigger nudge
        let llm = ScriptedLlm::new(vec![make_text_response("Done.")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            ToolLoopConfig::default(),
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("hello").unwrap();
        assert_eq!(result.response, "Done.");
        assert_eq!(result.turns, 1); // no nudge, single turn
    }

    // ── Batch 6: Anti-hallucination hardening tests ──

    #[test]
    fn first_turn_uses_required_tool_choice() {
        // Build a loop with no tool results in messages (first turn)
        let llm = ScriptedLlm::new(vec![
            // Force tool_choice=required will cause the model to call a tool
            make_tool_response(vec![LlmToolCall {
                id: "call_1".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path":"Cargo.toml"}"#.to_string(),
            }]),
            make_text_response("Read the file."),
        ]);
        let tool_host = Arc::new(MockToolHost::new(
            vec![ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: true,
                output: serde_json::json!({"content": "[package]\nname = \"test\""}),
            }],
            true,
        ));

        let config = ToolLoopConfig::default();
        let loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        // The first build_request should use tool_choice=required
        let request = loop_.build_request();
        assert_eq!(
            request.tool_choice,
            ToolChoice::required(),
            "first turn should force tool_choice=required"
        );
    }

    #[test]
    fn subsequent_turns_use_auto_tool_choice() {
        let llm = ScriptedLlm::new(vec![]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));
        let config = ToolLoopConfig::default();
        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        // Add a tool result message to simulate a later turn
        loop_.messages.push(ChatMessage::Tool {
            tool_call_id: "call_1".to_string(),
            content: "file content".to_string(),
        });

        let request = loop_.build_request();
        assert_eq!(
            request.tool_choice,
            ToolChoice::auto(),
            "subsequent turns should use tool_choice=auto"
        );
    }

    #[test]
    fn hallucination_nudge_triggers_at_300_chars() {
        // Response of exactly 301 chars should trigger nudge
        let text_301 = "x".repeat(301);
        let llm = ScriptedLlm::new(vec![
            make_text_response(&text_301),
            make_text_response("OK"), // after nudge
        ]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            ToolLoopConfig::default(),
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("describe the project").unwrap();
        assert_eq!(result.response, "OK");
        assert_eq!(result.turns, 2, "nudge should have triggered an extra turn");

        // Response of exactly 300 chars should NOT trigger nudge
        let text_300 = "y".repeat(300);
        let llm2 = ScriptedLlm::new(vec![make_text_response(&text_300)]);
        let tool_host2 = Arc::new(MockToolHost::new(vec![], true));

        let mut loop2 = ToolUseLoop::new(
            &llm2,
            tool_host2,
            ToolLoopConfig::default(),
            "system".to_string(),
            default_tools(),
        );

        let result2 = loop2.run("describe the project").unwrap();
        assert_eq!(result2.turns, 1, "300 chars should not trigger nudge");
    }

    #[test]
    fn unverified_file_references_detected() {
        // Text mentioning file paths without any tool calls
        assert!(has_unverified_file_references(
            "The project has src/main.rs and src/lib.rs with key functions.",
            &[]
        ));

        // After reading files, should not flag
        let read_calls = vec![ToolCallRecord {
            tool_name: "fs_read".to_string(),
            tool_call_id: "c1".to_string(),
            args_summary: "path=\"src/main.rs\"".to_string(),
            success: true,
            duration_ms: 10,
        }];
        assert!(!has_unverified_file_references(
            "The project has src/main.rs and src/lib.rs with key functions.",
            &read_calls
        ));

        // Short text with no path-like patterns should not flag
        assert!(!has_unverified_file_references("Hello, world!", &[]));
    }

    #[test]
    fn evidence_driven_escalation_on_tool_failures() {
        // Simulate: tool outputs contain compile error → model retries → responds.
        // Escalation signals should detect the error and switch to hard budget.
        let llm = ScriptedLlm::new(vec![
            make_tool_response(vec![LlmToolCall {
                id: "c1".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path":"missing.rs"}"#.to_string(),
            }]),
            make_tool_response(vec![LlmToolCall {
                id: "c2".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path":"also_missing.rs"}"#.to_string(),
            }]),
            make_text_response("Got it."),
        ]);

        let tool_host = Arc::new(MockToolHost::new(
            vec![
                ToolResult {
                    invocation_id: uuid::Uuid::nil(),
                    success: false,
                    output: serde_json::json!("error[E0308]: mismatched types"),
                },
                ToolResult {
                    invocation_id: uuid::Uuid::nil(),
                    success: false,
                    output: serde_json::json!("test result: FAILED. 0 passed; 1 failed;"),
                },
            ],
            true,
        ));

        let config = ToolLoopConfig {
            thinking: Some(deepseek_core::ThinkingConfig::enabled(
                crate::complexity::DEFAULT_THINK_BUDGET,
            )),
            ..Default::default()
        };

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "You are helpful.".to_string(),
            default_tools(),
        );

        let result = loop_.run("Read those files").unwrap();
        assert_eq!(result.response, "Got it.");
        assert_eq!(result.turns, 3);
        // Evidence-driven: escalation detected from tool output content
        assert!(loop_.escalation.compile_error, "should detect compile error");
        assert!(loop_.escalation.test_failure, "should detect test failure");
        assert!(loop_.escalation.should_escalate(), "should be escalated");
        assert_eq!(loop_.escalation.consecutive_failure_turns, 2);
    }
}
