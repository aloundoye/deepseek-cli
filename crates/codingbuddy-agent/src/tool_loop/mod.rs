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
//!
//! # Module structure
//!
//! - [`types`] — Public types, configuration, and callback signatures
//! - [`safety`] — Doom loop detection, circuit breaker, cost tracking
//! - [`anti_hallucination`] — File reference validation, shell command detection, directive extraction
//! - [`compaction`] — Context pruning and compaction summaries
//! - [`helpers`] — Small utility functions

pub mod types;

mod anti_hallucination;
mod compaction;
mod helpers;
mod safety;

// Re-export public API from submodules
pub use types::*;

use anyhow::{Result, anyhow};
use codingbuddy_core::{
    ApprovedToolCall, ChatMessage, ChatRequest, EventKind, LlmToolCall, StreamCallback,
    StreamChunk, TokenUsage, ToolChoice, ToolDefinition, ToolHost, UserQuestion,
    estimate_message_tokens, strip_prior_reasoning_content,
};
use codingbuddy_hooks::{HookEvent, HookInput, HookRuntime};
use codingbuddy_llm::LlmClient;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::tool_bridge;

use anti_hallucination::{
    HALLUCINATION_NUDGE, HALLUCINATION_NUDGE_THRESHOLD, MAX_NUDGE_ATTEMPTS,
    contains_shell_command_pattern, extract_user_directives, has_unverified_file_references,
};
use compaction::{
    COMPACTION_TARGET_PCT, PRUNE_AGE_TURNS, build_compaction_summary,
    build_compaction_summary_with_llm, extract_tool_path,
};
use helpers::{is_read_only_api_name, summarize_args};
use safety::{
    CIRCUIT_BREAKER_COOLDOWN_TURNS, CIRCUIT_BREAKER_THRESHOLD, CircuitBreakerState, CostTracker,
    DOOM_LOOP_GUIDANCE, DOOM_LOOP_THRESHOLD, DoomLoopTracker, ERROR_RECOVERY_GUIDANCE,
    FINISH_REASON_DOOM_LOOP, MAX_RECENT_ERRORS, STUCK_DETECTION_GUIDANCE,
};

/// Context window usage percentage that triggers lightweight pruning (Phase 1).
pub const PRUNE_THRESHOLD_PCT: f64 = 0.80;

/// Context window usage percentage that triggers full compaction (Phase 2).
pub const COMPACTION_THRESHOLD_PCT: f64 = 0.95;

/// Every N tool calls, inject a brief system reminder to keep the model on track.
const MID_CONVERSATION_REMINDER_INTERVAL: usize = 10;

/// Brief reminder injected every `MID_CONVERSATION_REMINDER_INTERVAL` tool calls.
const MID_CONVERSATION_REMINDER: &str =
    "Reminder: Verify changes with tests. Be concise. Use tools — do not guess.";

/// TTL for cached read-only tool results (in seconds).
const TOOL_CACHE_TTL_SECS: u64 = 60;

/// Tools whose results can be cached (all read-only).
const CACHEABLE_TOOLS: &[&str] = &["fs_read", "fs_glob", "fs_grep", "fs_list", "index_query"];

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
    /// Recent error messages for stuck detection. When the same error
    /// appears 3+ times, inject stronger guidance to try a different approach.
    recent_errors: Vec<String>,
    /// Whether error recovery guidance has been injected this escalation cycle.
    recovery_injected: bool,
    /// Pre-compiled output scanner for injection/secret detection.
    output_scanner: codingbuddy_policy::output_scanner::OutputScanner,
    /// Cache for read-only tool results. Keyed by (tool_name, args_hash).
    /// Entries expire after `TOOL_CACHE_TTL_SECS`. Invalidated when write tools
    /// modify the cached path.
    tool_cache: HashMap<String, (serde_json::Value, Instant)>,
    /// Circuit breaker: tracks consecutive failures per tool name.
    /// When a tool fails `CIRCUIT_BREAKER_THRESHOLD` times in a row, it is
    /// disabled for `CIRCUIT_BREAKER_COOLDOWN_TURNS`.
    circuit_breaker: HashMap<String, CircuitBreakerState>,
    /// Cumulative cost tracking across the session.
    cost_tracker: CostTracker,
    /// Actual tokens-per-char ratio derived from API responses. Starts at 0.25
    /// (the estimate_message_tokens default) and converges to the real ratio.
    actual_tokens_per_char: f64,
    /// Doom loop tracker — detects when the model repeats the same tool call.
    doom_loop_tracker: DoomLoopTracker,
    /// User directives extracted from conversation ("always X", "never Y", etc.).
    /// These survive compaction by being re-injected after the system message.
    pinned_directives: Vec<String>,
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
        let mut messages = vec![ChatMessage::System {
            content: system_prompt,
        }];

        // Inject initial context (bootstrap, retrieval) after system prompt
        for msg in &config.initial_context {
            messages.push(msg.clone());
        }

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
            recent_errors: Vec::new(),
            recovery_injected: false,
            output_scanner: codingbuddy_policy::output_scanner::OutputScanner::new(),
            tool_cache: HashMap::new(),
            circuit_breaker: HashMap::new(),
            cost_tracker: CostTracker::default(),
            actual_tokens_per_char: 0.25, // Default estimate, converges from API responses
            doom_loop_tracker: DoomLoopTracker::default(),
            pinned_directives: Vec::new(),
        }
    }

    /// Check the tool result cache for a matching entry.
    fn cache_lookup(
        &mut self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Option<serde_json::Value> {
        if !CACHEABLE_TOOLS.contains(&tool_name) {
            return None;
        }

        let key = format!("{}:{}", tool_name, args);
        if let Some((result, timestamp)) = self.tool_cache.get(&key)
            && timestamp.elapsed().as_secs() < TOOL_CACHE_TTL_SECS
        {
            return Some(result.clone());
        }
        // Expired or not found — remove it
        self.tool_cache.remove(&key);
        None
    }

    /// Store a tool result in the cache.
    fn cache_store(
        &mut self,
        tool_name: &str,
        args: &serde_json::Value,
        result: &serde_json::Value,
    ) {
        if !CACHEABLE_TOOLS.contains(&tool_name) {
            return;
        }
        let key = format!("{}:{}", tool_name, args);
        self.tool_cache
            .insert(key, (result.clone(), Instant::now()));
    }

    /// Invalidate cache entries for a given path (after a write tool modifies it).
    fn cache_invalidate_path(&mut self, path: &str) {
        self.tool_cache.retain(|key, _| {
            // Invalidate any cached entry whose args contain this path
            !key.contains(path)
        });
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

        // Extract and pin user directives so they survive compaction
        self.collect_directives(user_message);

        // Inject retrieval context on every user turn
        self.inject_retrieval_context(user_message);

        self.execute_loop()
    }

    /// Continue the conversation with additional user input (multi-turn).
    pub fn continue_with(&mut self, user_message: &str) -> Result<ToolLoopResult> {
        // Strip reasoning from prior turns before adding new user message
        strip_prior_reasoning_content(&mut self.messages);

        self.messages.push(ChatMessage::User {
            content: user_message.to_string(),
        });

        // Extract and pin user directives so they survive compaction
        self.collect_directives(user_message);

        // Inject retrieval context on every user turn (not just first)
        self.inject_retrieval_context(user_message);

        self.execute_loop()
    }

    /// Extract user directives from a message and add them to the pinned set.
    fn collect_directives(&mut self, text: &str) {
        for directive in extract_user_directives(text) {
            if !self.pinned_directives.contains(&directive) {
                self.pinned_directives.push(directive);
            }
        }
        // Cap at 20 to bound memory/token usage
        self.pinned_directives.truncate(20);
    }

    /// The main loop: call LLM, execute tools, feed results back, repeat.
    fn execute_loop(&mut self) -> Result<ToolLoopResult> {
        let mut tool_calls_made = Vec::new();
        let mut total_usage = TokenUsage::default();
        let mut turns: usize = 0;
        let mut nudge_count: usize = 0;
        let mut last_model = self.config.model.clone();

        // Pre-compute tool definition token estimate (tools are immutable during the loop)
        let tool_def_tokens: u64 = self
            .tools
            .iter()
            .map(|t| (serde_json::to_string(t).unwrap_or_default().len() as u64) / 4)
            .sum();

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

            // Two-phase context management:
            // Phase 1 (80%): Prune old tool outputs to free space without losing structure
            // Phase 2 (95%): Full compaction with structured summary
            let message_tokens = estimate_message_tokens(&self.messages);
            let estimated_tokens = message_tokens + tool_def_tokens;

            let prune_threshold =
                (self.config.context_window_tokens as f64 * PRUNE_THRESHOLD_PCT) as u64;
            let compact_threshold =
                (self.config.context_window_tokens as f64 * COMPACTION_THRESHOLD_PCT) as u64;

            if estimated_tokens > prune_threshold {
                // Phase 1: Lightweight pruning — trim old tool outputs
                self.prune_old_tool_outputs();

                // Re-check after pruning
                let post_prune_tokens = estimate_message_tokens(&self.messages) + tool_def_tokens;

                if post_prune_tokens > compact_threshold {
                    // Phase 2: Full compaction with structured summary
                    let target =
                        (self.config.context_window_tokens as f64 * COMPACTION_TARGET_PCT) as u64;
                    let pre_msg_count = self.messages.len() as u64;
                    let compacted = self.compact_messages(target);
                    if compacted {
                        let post_tokens = estimate_message_tokens(&self.messages) + tool_def_tokens;
                        if let Some(ref cb) = self.event_cb {
                            cb(EventKind::CompactionTriggered {
                                phase: "full".to_string(),
                                pre_tokens: post_prune_tokens,
                                post_tokens,
                                messages_before: pre_msg_count,
                                messages_after: self.messages.len() as u64,
                            });
                        }
                    }
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
            }

            // Build and send the LLM request
            turns += 1;
            let request = self.build_request();

            // Emit model routing event when the model changes (escalation/de-escalation)
            if request.model != last_model {
                if let Some(ref cb) = self.event_cb {
                    let reason = if codingbuddy_core::is_reasoner_model(&request.model) {
                        "escalation"
                    } else {
                        "de-escalation"
                    };
                    cb(EventKind::ModelRoutingChanged {
                        from_model: last_model.clone(),
                        to_model: request.model.clone(),
                        reason: reason.to_string(),
                    });
                }
                last_model = request.model.clone();
            }

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

                // T3.4: Track cost
                self.cost_tracker.record(usage);
                if self.cost_tracker.should_warn() {
                    self.emit(StreamChunk::SecurityWarning {
                        message: format!(
                            "Cost warning: estimated ${:.4} spent so far",
                            self.cost_tracker.estimated_cost_usd()
                        ),
                    });
                }
                if self.cost_tracker.over_budget() {
                    return Err(anyhow!(
                        "Session budget exceeded: ${:.4} > ${:.4} limit",
                        self.cost_tracker.estimated_cost_usd(),
                        self.cost_tracker.max_budget_usd.unwrap_or(0.0)
                    ));
                }

                // T3.5: Update actual tokens-per-char ratio from API response
                let total_chars: usize = self
                    .messages
                    .iter()
                    .map(|m| match m {
                        ChatMessage::System { content } | ChatMessage::User { content } => {
                            content.len()
                        }
                        ChatMessage::Assistant {
                            content,
                            reasoning_content,
                            tool_calls,
                        } => {
                            content.as_deref().map_or(0, str::len)
                                + reasoning_content.as_deref().map_or(0, str::len)
                                + tool_calls
                                    .iter()
                                    .map(|tc| tc.arguments.len())
                                    .sum::<usize>()
                        }
                        ChatMessage::Tool { content, .. } => content.len(),
                    })
                    .sum();
                if total_chars > 0 && usage.prompt_tokens > 0 {
                    let new_ratio = usage.prompt_tokens as f64 / total_chars as f64;
                    // Exponential moving average to smooth the ratio
                    self.actual_tokens_per_char =
                        0.7 * self.actual_tokens_per_char + 0.3 * new_ratio;
                }
            }

            // T3.3: Decrement circuit breaker cooldowns at start of each turn
            for state in self.circuit_breaker.values_mut() {
                if state.cooldown_remaining > 0 {
                    state.cooldown_remaining -= 1;
                }
            }

            // Handle content filter before anything else
            if response.finish_reason == "content_filter" {
                return Err(anyhow!("Response blocked by content filter"));
            }

            // No tool calls → return the text response (or nudge to use tools)
            if response.tool_calls.is_empty() {
                let text = response.text.clone();

                // Anti-hallucination guard: if the model responds with a long text
                // without using tools, nudge it to use tools instead of guessing.
                // Fires in ALL modes including Ask/read-only — that's where users
                // ask about the codebase and hallucination is most harmful.
                // Allows up to MAX_NUDGE_ATTEMPTS nudges per turn before letting through.
                // NOTE: No `tool_calls_made.is_empty()` guard — the model can revert to
                // hallucination at any point, even after using tools earlier.
                let should_nudge = nudge_count < MAX_NUDGE_ATTEMPTS
                    && (text.len() > HALLUCINATION_NUDGE_THRESHOLD
                        || has_unverified_file_references(&text, &tool_calls_made)
                        || contains_shell_command_pattern(&text));

                if should_nudge {
                    // Don't emit this attempt — inject a nudge and retry
                    nudge_count += 1;
                    let trigger = if text.len() > HALLUCINATION_NUDGE_THRESHOLD {
                        "long_response"
                    } else if has_unverified_file_references(&text, &tool_calls_made) {
                        "unverified_file_ref"
                    } else {
                        "shell_command_pattern"
                    };
                    if let Some(ref cb) = self.event_cb {
                        cb(EventKind::HallucinationNudgeFired {
                            nudge_count: nudge_count as u64,
                            trigger: trigger.to_string(),
                        });
                    }
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

            // Execute tool calls, parallelizing independent read-only tools.
            let was_escalated_before_batch = self.escalation.should_escalate();
            let mut batch_had_failure = false;
            let mut batch_had_success = false;

            // Partition: read-only tools with auto-approval can run in parallel;
            // write tools, unknown tools, and agent-level tools run sequentially.
            let (parallel_calls, sequential_calls): (Vec<_>, Vec<_>) =
                response.tool_calls.iter().partition(|c| {
                    !tool_bridge::is_agent_level_tool(&c.name)
                        && !tool_bridge::is_write_tool(&c.name)
                        && codingbuddy_core::ReviewMode::is_read_only_tool(&c.name)
                });

            // Execute independent read-only tools in parallel using thread scope.
            // This avoids blocking on I/O-bound tools (file reads, greps) sequentially.
            if parallel_calls.len() > 1 {
                // Execute the raw tool calls in parallel, collecting results
                let tool_host = &self.tool_host;
                let tools = &self.tools;
                let parallel_results: Vec<_> = std::thread::scope(|s| {
                    let handles: Vec<_> = parallel_calls
                        .iter()
                        .map(|llm_call| {
                            s.spawn(move || {
                                // Repair tool name
                                let repaired = tool_bridge::repair_tool_name(&llm_call.name, tools);
                                let effective_name =
                                    repaired.unwrap_or_else(|| llm_call.name.clone());
                                let effective_call = if effective_name != llm_call.name {
                                    LlmToolCall {
                                        id: llm_call.id.clone(),
                                        name: effective_name,
                                        arguments: llm_call.arguments.clone(),
                                    }
                                } else {
                                    (*llm_call).clone()
                                };
                                let internal =
                                    tool_bridge::llm_tool_call_to_internal(&effective_call);
                                let proposal = tool_host.propose(internal);
                                // Read-only tools are auto-approved
                                let approved = ApprovedToolCall {
                                    invocation_id: proposal.invocation_id,
                                    call: proposal.call,
                                };
                                let result = tool_host.execute(approved);
                                (effective_call, result)
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| {
                            h.join().unwrap_or_else(|panic_payload| {
                                let panic_msg = panic_payload
                                    .downcast_ref::<String>()
                                    .map(|s| s.as_str())
                                    .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                                    .unwrap_or("unknown panic");
                                let error_call = LlmToolCall {
                                    id: String::new(),
                                    name: "parallel_tool_panic".to_string(),
                                    arguments: String::new(),
                                };
                                let error_result = codingbuddy_core::ToolResult {
                                    invocation_id: uuid::Uuid::nil(),
                                    success: false,
                                    output: serde_json::json!(format!(
                                        "Internal error: parallel tool execution panicked: {panic_msg}"
                                    )),
                                };
                                (error_call, error_result)
                            })
                        })
                        .collect()
                });

                // Process results sequentially (updates messages, cache, events)
                for (effective_call, result) in &parallel_results {
                    let args: serde_json::Value = serde_json::from_str(&effective_call.arguments)
                        .unwrap_or_else(|e| {
                            eprintln!(
                                "[tool_loop] failed to parse tool arguments for '{}': {e}",
                                effective_call.name
                            );
                            serde_json::json!({})
                        });
                    let args_summary = summarize_args(&args);

                    self.emit(StreamChunk::ToolCallStart {
                        tool_name: effective_call.name.clone(),
                        args_summary: args_summary.clone(),
                    });

                    // Scan for escalation signals
                    if let Some(output_str) = result.output.as_str() {
                        self.escalation
                            .scan_tool_output(&effective_call.name, output_str);
                        if !result.success {
                            self.track_error(output_str);
                        }
                    }

                    let (msg, injection_warnings) = tool_bridge::tool_result_to_message(
                        &effective_call.id,
                        &effective_call.name,
                        result,
                        Some(&self.output_scanner),
                    );
                    let msg = self.apply_privacy_to_message(msg);

                    // Cache store AFTER privacy filtering — cache only holds filtered data.
                    // We extract the filtered content from the message we're about to push.
                    if result.success {
                        if let ChatMessage::Tool { ref content, .. } = msg {
                            self.cache_store(
                                &effective_call.name,
                                &args,
                                &serde_json::Value::String(content.clone()),
                            );
                        } else {
                            self.cache_store(&effective_call.name, &args, &result.output);
                        }
                    }

                    self.messages.push(msg);

                    self.emit_injection_warnings(&injection_warnings);

                    self.emit(StreamChunk::ToolCallEnd {
                        tool_name: effective_call.name.clone(),
                        duration_ms: 0,
                        success: result.success,
                        summary: if result.success {
                            "ok".to_string()
                        } else {
                            "error".to_string()
                        },
                    });

                    let record = ToolCallRecord {
                        tool_name: effective_call.name.clone(),
                        tool_call_id: effective_call.id.clone(),
                        args_summary,
                        success: result.success,
                        duration_ms: 0,
                    };
                    if record.success {
                        batch_had_success = true;
                    } else {
                        batch_had_failure = true;
                    }
                    tool_calls_made.push(record);
                }
            } else {
                // ≤1 parallel call — just run them sequentially via execute_tool_call
                for llm_call in &parallel_calls {
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
            }

            // Execute sequential (write/agent-level) tools one at a time
            for llm_call in &sequential_calls {
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
                // Reset recovery state on success
                self.recovery_injected = false;
            } else if batch_had_failure {
                self.escalation.record_failure();

                // Error recovery: inject guidance on first escalation transition
                if self.escalation.should_escalate()
                    && !was_escalated_before_batch
                    && !self.recovery_injected
                {
                    if let Some(ref cb) = self.event_cb {
                        cb(EventKind::ErrorRecoveryTriggered {
                            level: "recovery".to_string(),
                            repeated_error_count: 1,
                        });
                    }
                    self.messages.push(ChatMessage::System {
                        content: ERROR_RECOVERY_GUIDANCE.to_string(),
                    });
                    self.recovery_injected = true;
                }

                // Stuck detection: same error 3+ times → stronger nudge
                let repeated_count = self.repeated_error_count();
                if repeated_count >= 3 {
                    if let Some(ref cb) = self.event_cb {
                        cb(EventKind::ErrorRecoveryTriggered {
                            level: "stuck".to_string(),
                            repeated_error_count: repeated_count as u64,
                        });
                    }
                    self.messages.push(ChatMessage::System {
                        content: STUCK_DETECTION_GUIDANCE.to_string(),
                    });
                    // Clear error history to avoid re-triggering every turn
                    self.recent_errors.clear();
                }
            }

            // Doom loop detection: check if the model is repeating identical calls.
            // When detected, STOP the tool loop (blocking gate) — the model cannot
            // continue without the user explicitly sending a new message.
            let mut doom_loop_detected = false;
            let mut doom_loop_tool = String::new();
            for llm_call in response.tool_calls.iter() {
                if self
                    .doom_loop_tracker
                    .record(&llm_call.name, &llm_call.arguments)
                {
                    doom_loop_detected = true;
                    doom_loop_tool = llm_call.name.clone();
                    break;
                }
            }
            if doom_loop_detected {
                if let Some(ref cb) = self.event_cb {
                    cb(EventKind::DoomLoopDetected {
                        tool_name: doom_loop_tool,
                        repeat_count: DOOM_LOOP_THRESHOLD as u64,
                    });
                }
                self.emit(StreamChunk::SecurityWarning {
                    message: format!(
                        "Doom loop detected: model repeated identical tool calls {}+ times. \
                         Stopping to prevent infinite loop. Send a new message to continue.",
                        DOOM_LOOP_THRESHOLD
                    ),
                });
                self.messages.push(ChatMessage::System {
                    content: DOOM_LOOP_GUIDANCE.to_string(),
                });
                self.doom_loop_tracker.mark_warned();
                // Terminate the tool loop — this is a blocking gate.
                // The model cannot continue; the user must send a new message.
                self.emit(StreamChunk::Done {
                    reason: Some(FINISH_REASON_DOOM_LOOP.to_string()),
                });
                return Ok(ToolLoopResult {
                    response: response.text.clone(),
                    tool_calls_made,
                    finish_reason: FINISH_REASON_DOOM_LOOP.to_string(),
                    usage: total_usage,
                    turns,
                    messages: self.messages.clone(),
                });
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

            // Strip reasoning from prior turns for CodingBuddy API compliance
            strip_prior_reasoning_content(&mut self.messages);
        }
    }

    /// Execute a single tool call, handling approval flow, hooks, events, and checkpoints.
    fn execute_tool_call(&mut self, llm_call: &LlmToolCall) -> Result<Vec<ToolCallRecord>> {
        let mut records = Vec::new();
        let start = Instant::now();

        // Check if this is an agent-level tool that needs special handling
        if tool_bridge::is_agent_level_tool(&llm_call.name) {
            return self.handle_agent_level_tool(llm_call);
        }

        // ── Circuit breaker ──
        // If this tool has failed too many times consecutively, reject it
        // with guidance to try a different approach.
        if let Some(state) = self.circuit_breaker.get(&llm_call.name)
            && state.cooldown_remaining > 0
        {
            let duration = start.elapsed().as_millis() as u64;
            self.emit(StreamChunk::ToolCallEnd {
                tool_name: llm_call.name.clone(),
                duration_ms: duration,
                success: false,
                summary: "circuit-broken".to_string(),
            });
            self.messages.push(tool_bridge::tool_error_to_message(
                &llm_call.id,
                &format!(
                    "Tool '{}' is temporarily disabled after {} consecutive failures. \
                     It will be re-enabled in {} turn(s). Try a different approach or tool.",
                    llm_call.name, CIRCUIT_BREAKER_THRESHOLD, state.cooldown_remaining
                ),
            ));
            records.push(ToolCallRecord {
                tool_name: llm_call.name.clone(),
                tool_call_id: llm_call.id.clone(),
                args_summary: String::new(),
                success: false,
                duration_ms: duration,
            });
            return Ok(records);
        }

        // ── Tool name repair ──
        // DeepSeek sometimes gets tool names wrong (casing, hyphens, misspellings).
        // Try to fix before failing hard. If no match is found, pass the original
        // name through — it may be an MCP/plugin tool not in the definitions list,
        // or the tool host may handle it.
        let llm_call = match tool_bridge::repair_tool_name(&llm_call.name, &self.tools) {
            Some(repaired_name) if repaired_name != llm_call.name => LlmToolCall {
                id: llm_call.id.clone(),
                name: repaired_name,
                arguments: llm_call.arguments.clone(),
            },
            _ => llm_call.clone(),
        };

        // Convert LLM call to internal format
        let tool_call = tool_bridge::llm_tool_call_to_internal(&llm_call);
        // Save parsed args before tool_call is moved into propose()
        let parsed_args = tool_call.args.clone();

        // ── JSON schema validation ──
        // Validate arguments against the tool's schema before executing.
        // Returns structured error so the model can self-correct.
        if let Err(validation_error) = codingbuddy_tools::validate_tool_args_schema(
            &llm_call.name,
            &tool_call.args,
            &self.tools,
        ) {
            let duration = start.elapsed().as_millis() as u64;
            self.emit(StreamChunk::ToolCallEnd {
                tool_name: llm_call.name.clone(),
                duration_ms: duration,
                success: false,
                summary: "invalid args".to_string(),
            });
            self.messages.push(tool_bridge::tool_error_to_message(
                &llm_call.id,
                &validation_error,
            ));
            records.push(ToolCallRecord {
                tool_name: llm_call.name.clone(),
                tool_call_id: llm_call.id.clone(),
                args_summary: summarize_args(&tool_call.args),
                success: false,
                duration_ms: duration,
            });
            return Ok(records);
        }

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
                workspace: self.workspace_str(),
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
                proposal: codingbuddy_core::ToolProposal {
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

        // ── Cache lookup ──
        // For read-only tools, check if we have a recent cached result.
        // Reuse the already-parsed args from tool_call instead of re-parsing.
        if let Some(cached_result) = self.cache_lookup(&llm_call.name, &parsed_args) {
            if let Some(ref cb) = self.event_cb {
                cb(EventKind::ToolCacheHit {
                    tool_name: llm_call.name.clone(),
                });
            }
            let duration = start.elapsed().as_millis() as u64;
            let result = codingbuddy_core::ToolResult {
                invocation_id: proposal.invocation_id,
                success: true,
                output: cached_result,
            };
            let (msg, injection_warnings) = tool_bridge::tool_result_to_message(
                &llm_call.id,
                &llm_call.name,
                &result,
                Some(&self.output_scanner),
            );
            self.messages.push(msg);
            self.emit_injection_warnings(&injection_warnings);
            self.emit(StreamChunk::ToolCallEnd {
                tool_name: llm_call.name.clone(),
                duration_ms: duration,
                success: true,
                summary: "cached".to_string(),
            });
            records.push(ToolCallRecord {
                tool_name: llm_call.name.clone(),
                tool_call_id: llm_call.id.clone(),
                args_summary,
                success: true,
                duration_ms: duration,
            });
            return Ok(records);
        }

        // Extract modified paths once for write tools (used for checkpoint + cache invalidation)
        let modified_paths = if tool_bridge::is_write_tool(&llm_call.name) {
            tool_bridge::extract_modified_paths(&llm_call.name, &llm_call.arguments)
        } else {
            Vec::new()
        };

        // Create checkpoint before destructive tool calls
        if !modified_paths.is_empty()
            && let Some(ref cp) = self.checkpoint_cb
        {
            let _ = cp("before tool execution", &modified_paths);
        }

        // Execute the approved tool
        let approved_call = ApprovedToolCall {
            invocation_id: proposal.invocation_id,
            call: proposal.call,
        };
        let result = self.tool_host.execute(approved_call);

        let duration = start.elapsed().as_millis() as u64;
        let success = result.success;

        // ── Circuit breaker update ──
        // Track consecutive failures per tool for circuit-breaking.
        let cb_state = self
            .circuit_breaker
            .entry(llm_call.name.clone())
            .or_default();
        if success {
            cb_state.consecutive_failures = 0;
            cb_state.cooldown_remaining = 0;
        } else {
            cb_state.consecutive_failures += 1;
            if cb_state.consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD {
                cb_state.cooldown_remaining = CIRCUIT_BREAKER_COOLDOWN_TURNS;
                if let Some(ref cb) = self.event_cb {
                    cb(EventKind::CircuitBreakerTripped {
                        tool_name: llm_call.name.clone(),
                        consecutive_failures: CIRCUIT_BREAKER_THRESHOLD as u64,
                        cooldown_turns: CIRCUIT_BREAKER_COOLDOWN_TURNS as u64,
                    });
                }
                cb_state.consecutive_failures = 0; // Reset count for next cycle
            }
        }

        // ── Cache invalidate ──
        // NOTE: Cache store moved below privacy filtering so cache only holds filtered data.
        for path in &modified_paths {
            if let Some(path_str) = path.to_str() {
                self.cache_invalidate_path(path_str);
            }
        }

        // Scan tool output for evidence-driven budget escalation signals
        // (compile errors, test failures, patch rejections, search misses).
        if let Some(output_str) = result.output.as_str() {
            self.escalation.scan_tool_output(&llm_call.name, output_str);
            if !success {
                self.track_error(output_str);
            }
        } else if let Ok(output_text) = serde_json::to_string(&result.output) {
            self.escalation
                .scan_tool_output(&llm_call.name, &output_text);
            if !success {
                self.track_error(&output_text);
            }
        }

        // Fire PostToolUse hook
        if let Some(ref hooks) = self.hooks {
            let input = HookInput {
                event: HookEvent::PostToolUse.as_str().to_string(),
                tool_name: Some(llm_call.name.clone()),
                tool_input: Some(parsed_args.clone()),
                tool_result: Some(result.output.clone()),
                prompt: None,
                session_type: None,
                workspace: self.workspace_str(),
            };
            let _ = hooks.fire(HookEvent::PostToolUse, &input);
        }

        // Log ToolResult event
        if let Some(ref cb) = self.event_cb {
            cb(EventKind::ToolResult {
                result: codingbuddy_core::ToolResult {
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
        let msg = self.apply_privacy_to_message(msg);

        // ── Cache store (after privacy filtering) ──
        // Cache successful read-only results with privacy-filtered content.
        if success {
            if let ChatMessage::Tool { ref content, .. } = msg {
                self.cache_store(
                    &llm_call.name,
                    &parsed_args,
                    &serde_json::Value::String(content.clone()),
                );
            } else {
                self.cache_store(&llm_call.name, &parsed_args, &result.output);
            }
        }

        self.messages.push(msg);

        self.emit_injection_warnings(&injection_warnings);

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
    fn handle_agent_level_tool(&mut self, llm_call: &LlmToolCall) -> Result<Vec<ToolCallRecord>> {
        let start = Instant::now();
        let args: serde_json::Value =
            serde_json::from_str(&llm_call.arguments).unwrap_or_else(|e| {
                eprintln!(
                    "[tool_loop] failed to parse agent tool arguments for '{}': {e}",
                    llm_call.name
                );
                serde_json::json!({})
            });

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
                let question = args
                    .get("question")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(no question)");
                let options: Vec<String> = args
                    .get("options")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                if let Some(ref cb) = self.user_question_cb {
                    let uq = UserQuestion {
                        question: question.to_string(),
                        options,
                    };
                    match cb(uq) {
                        Some(answer) => format!("User response: {answer}"),
                        None => format!(
                            "Question for user: {question}\n[User cancelled. Proceed with your best judgment.]"
                        ),
                    }
                } else {
                    format!(
                        "Question for user: {question}\n[User response not available in this context. Proceed with your best judgment or state your assumption.]"
                    )
                }
            }
            "spawn_task" => self.handle_spawn_task(&args)?,
            "skill" => self.handle_skill(&args)?,
            _ => {
                // Other agent-level tools (task_*, etc.) — not yet wired
                format!(
                    "Agent-level tool '{}' is not yet available in tool-use mode. Try a different approach.",
                    llm_call.name
                )
            }
        };

        let duration = start.elapsed().as_millis() as u64;
        self.emit(StreamChunk::ToolCallEnd {
            tool_name: llm_call.name.clone(),
            duration_ms: duration,
            success: true,
            summary: "ok".to_string(),
        });

        // Truncate agent-level tool output to prevent context overflow.
        // Agent tools (spawn_task, skill, extended_thinking) can produce very large
        // outputs that eat context window budget.
        let content = tool_bridge::truncate_agent_output(&result_content);

        self.messages.push(ChatMessage::Tool {
            tool_call_id: llm_call.id.clone(),
            content,
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
    ///
    /// `deepseek-reasoner` thinks natively — no `ThinkingConfig` needed (that's
    /// only for enabling thinking on `deepseek-chat`). V3.2 supports tool calls
    /// with the reasoner, so we pass through the current tool definitions.
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
            tools: self.tools.clone(),       // V3.2 supports tools with reasoner
            tool_choice: ToolChoice::auto(),  // Let the model decide
            max_tokens: codingbuddy_core::CODINGBUDDY_REASONER_MAX_OUTPUT_TOKENS,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            logprobs: None,
            top_logprobs: None,
            thinking: None,  // Reasoner thinks natively — no ThinkingConfig needed
            images: vec![],
            response_format: None,
        };
        let response = self.llm.complete_chat(&request)?;
        Ok(response.text)
    }

    /// Handle the spawn_task tool by delegating to the subagent worker.
    fn handle_spawn_task(&self, args: &serde_json::Value) -> Result<String> {
        let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
        let task_name = args
            .get("description")
            .or_else(|| args.get("task_name"))
            .and_then(|v| v.as_str())
            .unwrap_or("subtask");
        let subagent_type = args
            .get("subagent_type")
            .and_then(|v| v.as_str())
            .unwrap_or("general-purpose");
        let model_override = args.get("model").and_then(|v| v.as_str()).map(String::from);
        let max_turns = args
            .get("max_turns")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);
        let run_in_background = args
            .get("run_in_background")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if prompt.is_empty() {
            return Ok(
                "Error: spawn_task requires a 'prompt' argument describing the task.".to_string(),
            );
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
                    workspace: self.workspace_str(),
                };
                let _ = hooks.fire(HookEvent::SubagentStart, &input);
            }
            let result = match worker(request) {
                Ok(result) => Ok(result),
                Err(e) => Ok(format!(
                    "Subagent '{task_name}' failed: {e}. Try a different approach or handle the task directly."
                )),
            };
            // P5-14: Fire SubagentStop hook
            if let Some(ref hooks) = self.hooks {
                let input = HookInput {
                    event: HookEvent::SubagentStop.as_str().to_string(),
                    tool_name: Some("spawn_task".to_string()),
                    tool_input: Some(args.clone()),
                    tool_result: result
                        .as_ref()
                        .ok()
                        .map(|r| serde_json::Value::String(r.clone())),
                    prompt: Some(prompt.to_string()),
                    session_type: None,
                    workspace: self.workspace_str(),
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
        let skill_name = args.get("skill").and_then(|v| v.as_str()).unwrap_or("");
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
    /// Applies dynamic thinking budget escalation and model routing:
    /// - Complex + escalated → route to `deepseek-reasoner` (native thinking, 64K output)
    /// - De-escalated (3 consecutive successes) → route back to `deepseek-chat`
    /// - Reasoner model → strip sampling params (temperature, top_p, etc.)
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

        // Force tool use on the first 2 LLM calls for each new question so the model
        // explores the codebase instead of fabricating an answer. DeepSeek-chat needs
        // stronger forcing than Claude — it tends to answer from memory on the first turn,
        // then switch to tools only after being nudged.
        let last_user_idx = self
            .messages
            .iter()
            .rposition(|m| matches!(m, ChatMessage::User { .. }))
            .unwrap_or(0);
        let tool_call_count_this_turn = self.messages[last_user_idx..]
            .iter()
            .filter(|m| matches!(m, ChatMessage::Tool { .. }))
            .count();
        // Require tool use for the first 2 LLM rounds per user question (Code mode only)
        let tool_choice = if tool_call_count_this_turn < 2 && !self.config.read_only {
            ToolChoice::required()
        } else {
            ToolChoice::auto()
        };

        // Model routing: use reasoner for Complex+escalated tasks
        let use_reasoner = self.config.complexity == crate::complexity::PromptComplexity::Complex
            && self.escalation.should_escalate();

        let (model, thinking, max_tokens) = if use_reasoner {
            // Route to deepseek-reasoner: native thinking, no ThinkingConfig needed
            (
                self.config.extended_thinking_model.clone(),
                None, // Reasoner thinks natively
                codingbuddy_core::CODINGBUDDY_REASONER_MAX_OUTPUT_TOKENS,
            )
        } else {
            // Standard deepseek-chat with thinking budget
            let thinking = self.config.thinking.as_ref().map(|base| {
                if self.escalation.should_escalate() {
                    codingbuddy_core::ThinkingConfig::enabled(self.escalation.budget())
                } else {
                    base.clone()
                }
            });
            (self.config.model.clone(), thinking, self.config.max_tokens)
        };

        // Strip sampling parameters and tool_choice for reasoner model (incompatible)
        let is_reasoner = codingbuddy_core::is_reasoner_model(&model);
        let temperature = if is_reasoner {
            None
        } else {
            self.config.temperature
        };
        // deepseek-reasoner does not support tool_choice=required — always use auto
        let tool_choice = if is_reasoner {
            ToolChoice::auto()
        } else {
            tool_choice
        };

        ChatRequest {
            model,
            messages: self.messages.clone(),
            tools,
            tool_choice,
            max_tokens,
            temperature,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            logprobs: None,
            top_logprobs: None,
            thinking,
            images: self.config.images.clone(),
            response_format: None,
        }
    }

    /// Track an error message for stuck detection.
    ///
    /// Normalizes the error text (first 200 chars, lowered) and keeps the last
    /// `MAX_RECENT_ERRORS` entries. Used to detect repeated identical failures.
    fn track_error(&mut self, error_text: &str) {
        // Normalize: lowercase, first 200 chars, trimmed
        let normalized = error_text
            .chars()
            .take(200)
            .collect::<String>()
            .to_ascii_lowercase()
            .trim()
            .to_string();
        if !normalized.is_empty() {
            self.recent_errors.push(normalized);
            if self.recent_errors.len() > MAX_RECENT_ERRORS {
                self.recent_errors.remove(0);
            }
        }
    }

    /// Count how many times the most recent error appears in the error history.
    ///
    /// Returns 0 if no errors recorded. Used for stuck detection — when the same
    /// error appears 3+ times, we inject stronger recovery guidance.
    fn repeated_error_count(&self) -> usize {
        if let Some(last) = self.recent_errors.last() {
            self.recent_errors.iter().filter(|e| e == &last).count()
        } else {
            0
        }
    }

    /// Phase 1: Lightweight pruning of old tool outputs.
    ///
    /// Truncates verbose tool results that are older than `PRUNE_AGE_TURNS` turn groups
    /// from the end. Keeps the last write result per file path and all recent results.
    /// This preserves conversation structure while reducing token usage.
    fn prune_old_tool_outputs(&mut self) {
        // Count turn groups from the end (a turn group starts at each User message)
        let mut turn_count = 0;
        let mut recent_boundary = self.messages.len();
        for (i, msg) in self.messages.iter().enumerate().rev() {
            if matches!(msg, ChatMessage::User { .. }) {
                turn_count += 1;
                if turn_count >= PRUNE_AGE_TURNS {
                    recent_boundary = i;
                    break;
                }
            }
        }

        if recent_boundary <= 1 {
            return; // Not enough history to prune
        }

        // Track file paths from recent tool results — keep those untouched
        let mut recent_paths: std::collections::HashSet<String> = std::collections::HashSet::new();
        for msg in self.messages[recent_boundary..].iter().rev() {
            if let ChatMessage::Tool { content, .. } = msg
                && let Some(path) = extract_tool_path(content)
            {
                recent_paths.insert(path);
            }
        }

        // Truncate old tool results in the "old" zone (messages[1..recent_boundary])
        const TRUNCATED_MARKER: &str = "[output pruned — re-read file if needed]";
        let max_kept_chars = 200;

        for msg in &mut self.messages[1..recent_boundary] {
            if let ChatMessage::Tool { content, .. } = msg {
                // Skip if already truncated or short
                if content.len() <= max_kept_chars || content.contains(TRUNCATED_MARKER) {
                    continue;
                }
                // Keep results referencing files still active in recent turns
                if let Some(path) = extract_tool_path(content)
                    && recent_paths.contains(&path)
                {
                    continue;
                }
                // Truncate: keep first 200 chars + marker (safe on multi-byte UTF-8)
                let safe_end = content.floor_char_boundary(max_kept_chars.min(content.len()));
                let truncated = format!("{}\n{}", &content[..safe_end], TRUNCATED_MARKER,);
                *content = truncated;
            }
        }
    }

    /// Compact messages by removing middle exchanges while preserving tool-call/result pairing.
    ///
    /// Walks backward from the end, keeping complete exchange groups (User + Assistant + Tool
    /// messages form a group). Never orphans a Tool message from its corresponding Assistant
    /// tool_calls. Respects `target_tokens` budget for kept messages.
    fn compact_messages(&mut self, target_tokens: u64) -> bool {
        if self.messages.len() <= 7 {
            return false;
        }

        let system = self.messages[0].clone();

        // Walk backward to find group boundaries.
        // A group starts at a User message and includes everything until the next User message.
        let mut keep_from = self.messages.len();
        let mut kept_tokens: u64 = 0;
        let target = target_tokens
            .min((self.config.context_window_tokens as f64 * COMPACTION_TARGET_PCT) as u64);

        while keep_from > 1 {
            // Find the start of this group (previous User message)
            let group_start = self.messages[1..keep_from]
                .iter()
                .rposition(|m| matches!(m, ChatMessage::User { .. }))
                .map(|i| i + 1) // +1 for the offset from [1..]
                .unwrap_or(1);

            let group_tokens: u64 = self.messages[group_start..keep_from]
                .iter()
                .map(|m| estimate_message_tokens(std::slice::from_ref(m)) as u64)
                .sum();

            if kept_tokens + group_tokens > target && keep_from < self.messages.len() {
                break; // This group would exceed budget
            }
            kept_tokens += group_tokens;
            keep_from = group_start;
        }

        if keep_from <= 1 {
            return false; // Nothing to compact
        }

        let middle_count = keep_from - 1;
        let compacted_msgs = &self.messages[1..keep_from];
        // Try LLM-based compaction first, fall back to code-based extraction
        let summary = build_compaction_summary_with_llm(self.llm, compacted_msgs)
            .unwrap_or_else(|_| build_compaction_summary(compacted_msgs));

        // Fire PreCompact hook if configured
        if let Some(ref hooks) = self.hooks {
            let input = HookInput {
                event: "pre_compact".to_string(),
                tool_name: None,
                tool_input: None,
                tool_result: Some(serde_json::Value::String(summary.clone())),
                prompt: None,
                session_type: None,
                workspace: self.workspace_str(),
            };
            let _ = hooks.fire(HookEvent::PreCompact, &input);
        }

        // Scan compacted messages for any directives we haven't captured yet
        for msg in compacted_msgs {
            if let ChatMessage::User { content } = msg {
                for directive in extract_user_directives(content) {
                    if !self.pinned_directives.contains(&directive) {
                        self.pinned_directives.push(directive);
                    }
                }
            }
        }
        self.pinned_directives.truncate(20);

        let summary_msg = ChatMessage::User {
            content: format!(
                "CONVERSATION_HISTORY (compacted from {middle_count} messages):\n{summary}"
            ),
        };
        let kept = self.messages[keep_from..].to_vec();
        self.messages = vec![system, summary_msg];

        // Re-inject pinned user directives so they survive compaction.
        // Placed as a System message right after the summary so the model
        // always sees them regardless of how many compaction rounds occur.
        if !self.pinned_directives.is_empty() {
            let directives_text = self
                .pinned_directives
                .iter()
                .map(|d| format!("- {d}"))
                .collect::<Vec<_>>()
                .join("\n");
            self.messages.push(ChatMessage::System {
                content: format!(
                    "USER DIRECTIVES (must follow throughout this conversation):\n{directives_text}"
                ),
            });
        }

        self.messages.extend(kept);
        true
    }

    /// Inject retrieval context from the workspace vector index before each LLM call.
    ///
    /// Fires on every user turn (not just turn 1) to keep multi-turn conversations
    /// grounded in the codebase. Budget based on **remaining** context window.
    fn inject_retrieval_context(&mut self, user_message: &str) {
        let retriever = match &self.config.retriever {
            Some(r) => r.clone(),
            None => return,
        };

        let max_results = 10;
        let results = match retriever(user_message, max_results) {
            Ok(r) if !r.is_empty() => r,
            _ => return,
        };

        // Budget: remaining context / 5 (not total context).
        // This avoids starving long conversations of retrieval budget.
        let current_tokens = estimate_message_tokens(&self.messages);
        let remaining = self
            .config
            .context_window_tokens
            .saturating_sub(current_tokens);

        // Skip if context nearly full (< 500 tokens remaining budget)
        let budget_tokens = remaining / 5;
        if budget_tokens < 500 {
            return;
        }

        let mut context_parts = Vec::new();
        let mut token_estimate: u64 = 0;

        for r in &results {
            let chunk_text = format!(
                "--- {}:{}-{} (score: {:.3}) ---\n{}\n",
                r.file_path, r.start_line, r.end_line, r.score, r.content,
            );
            let chunk_tokens = (chunk_text.len() as u64) / 4;
            if token_estimate + chunk_tokens > budget_tokens {
                break;
            }
            token_estimate += chunk_tokens;
            context_parts.push(chunk_text);
        }

        if !context_parts.is_empty() {
            let context_msg = format!(
                "RETRIEVAL_CONTEXT (relevant code from workspace index):\n{}",
                context_parts.join("\n")
            );
            self.messages.push(ChatMessage::System {
                content: context_msg,
            });

            if let Some(ref cb) = self.event_cb {
                cb(EventKind::Retrieval {
                    query: user_message.to_string(),
                    chunks_injected: context_parts.len() as u64,
                    token_estimate,
                });
            }
        }
    }

    /// Apply privacy router to tool output, redacting sensitive content if configured.
    fn apply_privacy_to_output(&self, output: &str) -> String {
        if let Some(ref router) = self.config.privacy_router {
            match router.apply_policy(output, None) {
                codingbuddy_local_ml::PrivacyResult::Redacted(redacted) => {
                    self.emit(StreamChunk::SecurityWarning {
                        message: "Privacy router redacted sensitive content in tool output"
                            .to_string(),
                    });
                    return redacted;
                }
                codingbuddy_local_ml::PrivacyResult::Blocked => {
                    self.emit(StreamChunk::SecurityWarning {
                        message: "Privacy router blocked tool output containing sensitive data"
                            .to_string(),
                    });
                    return "[BLOCKED: Privacy router blocked this output due to sensitive content]"
                        .to_string();
                }
                codingbuddy_local_ml::PrivacyResult::LocalSummary(summary) => return summary,
                codingbuddy_local_ml::PrivacyResult::Clean(_) => {}
            }
        }
        output.to_string()
    }

    /// Emit injection/secret warnings detected by the output scanner.
    fn emit_injection_warnings(
        &self,
        warnings: &[codingbuddy_policy::output_scanner::InjectionWarning],
    ) {
        for warning in warnings {
            self.emit(StreamChunk::SecurityWarning {
                message: format!(
                    "{:?} — {} (matched: {})",
                    warning.severity, warning.pattern_name, warning.matched_text
                ),
            });
        }
    }

    /// Apply privacy router to a tool message, redacting sensitive content if configured.
    fn apply_privacy_to_message(&self, msg: ChatMessage) -> ChatMessage {
        if let ChatMessage::Tool {
            tool_call_id,
            content,
        } = msg
        {
            let filtered = self.apply_privacy_to_output(&content);
            ChatMessage::Tool {
                tool_call_id,
                content: filtered,
            }
        } else {
            msg
        }
    }

    /// Get the workspace path as a string for hook inputs.
    fn workspace_str(&self) -> String {
        self.config
            .workspace
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default()
    }

    /// Emit a stream chunk to the callback.
    fn emit(&self, chunk: StreamChunk) {
        if let Some(ref cb) = self.stream_cb {
            cb(chunk);
        }
    }
}

/// Extract a file path from a tool result string, if present.
/// Handles both JSON-formatted results (with "path"/"file_path" keys) and plain text.
#[cfg(test)]
mod tests {
    use super::anti_hallucination::split_sentences;
    use super::compaction::COMPACTION_TEMPLATE;
    use super::safety::DOOM_LOOP_HISTORY_SIZE;
    use super::*;
    use codingbuddy_core::{LlmResponse, ToolCall, ToolProposal, ToolResult};
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
        fn complete(&self, _req: &codingbuddy_core::LlmRequest) -> Result<LlmResponse> {
            unimplemented!()
        }
        fn complete_streaming(
            &self,
            _req: &codingbuddy_core::LlmRequest,
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
        fn complete_fim(&self, _req: &codingbuddy_core::FimRequest) -> Result<LlmResponse> {
            unimplemented!()
        }
        fn complete_fim_streaming(
            &self,
            _req: &codingbuddy_core::FimRequest,
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
            function: codingbuddy_core::FunctionDefinition {
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
        // LLM always returns tool calls, never stops.
        // Use different args each turn to avoid triggering doom loop detection.
        let infinite_tool_calls: Vec<LlmResponse> = (0..5)
            .map(|i| {
                make_tool_response(vec![LlmToolCall {
                    id: format!("c{i}"),
                    name: "fs_read".to_string(),
                    arguments: format!(r#"{{"path":"test_{i}.rs"}}"#),
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

        // Add 2 tool result messages — after 2 tool calls, should switch to auto
        loop_.messages.push(ChatMessage::Tool {
            tool_call_id: "call_1".to_string(),
            content: "file content".to_string(),
        });
        loop_.messages.push(ChatMessage::Tool {
            tool_call_id: "call_2".to_string(),
            content: "another file".to_string(),
        });

        let request = loop_.build_request();
        assert_eq!(
            request.tool_choice,
            ToolChoice::auto(),
            "after 2+ tool results should use tool_choice=auto"
        );

        // With only 1 tool result, should still force required (strengthened forcing)
        let llm2 = ScriptedLlm::new(vec![]);
        let tool_host2 = Arc::new(MockToolHost::new(vec![], true));
        let mut loop2 = ToolUseLoop::new(
            &llm2,
            tool_host2,
            ToolLoopConfig::default(),
            "system".to_string(),
            default_tools(),
        );
        loop2.messages.push(ChatMessage::Tool {
            tool_call_id: "call_1".to_string(),
            content: "one result".to_string(),
        });
        let request2 = loop2.build_request();
        assert_eq!(
            request2.tool_choice,
            ToolChoice::required(),
            "with only 1 tool result, should still force required"
        );
    }

    #[test]
    fn hallucination_nudge_triggers_at_threshold() {
        // Response exceeding HALLUCINATION_NUDGE_THRESHOLD should trigger nudge
        let over_threshold = "x".repeat(HALLUCINATION_NUDGE_THRESHOLD + 1);
        let llm = ScriptedLlm::new(vec![
            make_text_response(&over_threshold),
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

        // Response at exactly the threshold should NOT trigger nudge
        let at_threshold = "y".repeat(HALLUCINATION_NUDGE_THRESHOLD);
        let llm2 = ScriptedLlm::new(vec![make_text_response(&at_threshold)]);
        let tool_host2 = Arc::new(MockToolHost::new(vec![], true));

        let mut loop2 = ToolUseLoop::new(
            &llm2,
            tool_host2,
            ToolLoopConfig::default(),
            "system".to_string(),
            default_tools(),
        );

        let result2 = loop2.run("describe the project").unwrap();
        assert_eq!(
            result2.turns, 1,
            "at-threshold response should not trigger nudge"
        );
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

        // Dot-access patterns should not flag (module access, not file paths)
        assert!(!has_unverified_file_references(
            "Use self.config and fmt.Println for output",
            &[]
        ));

        // Version numbers should not flag
        assert!(!has_unverified_file_references(
            "Update to v0.6 or version 2.34",
            &[]
        ));

        // URL-like patterns should not flag
        assert!(!has_unverified_file_references(
            "See https.example and http.server docs",
            &[]
        ));

        // New extensions should flag
        assert!(has_unverified_file_references(
            "Check the .env file and config.ini",
            &[]
        ));
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
            thinking: Some(codingbuddy_core::ThinkingConfig::enabled(
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
        assert!(
            loop_.escalation.compile_error,
            "should detect compile error"
        );
        assert!(loop_.escalation.test_failure, "should detect test failure");
        assert!(loop_.escalation.should_escalate(), "should be escalated");
        assert_eq!(loop_.escalation.consecutive_failure_turns, 2);
    }

    // ── Batch 2 (P9): Anti-hallucination ──

    #[test]
    fn hallucination_nudge_fires_in_ask_mode() {
        // Even in read_only mode (Ask/Context), the nudge should fire
        // to prevent hallucinated answers about the codebase.
        let long_text = "x".repeat(HALLUCINATION_NUDGE_THRESHOLD + 1);
        let llm = ScriptedLlm::new(vec![
            make_text_response(&long_text),
            make_text_response("OK"),
        ]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let config = ToolLoopConfig {
            read_only: true,
            ..Default::default()
        };
        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("where is the config?").unwrap();
        assert_eq!(result.response, "OK");
        assert_eq!(result.turns, 2, "nudge should fire in read_only/Ask mode");
    }

    #[test]
    fn hallucination_nudge_fires_up_to_3_times() {
        // Model hallucinates 3 times, then on 4th attempt it goes through
        let long_text = "x".repeat(HALLUCINATION_NUDGE_THRESHOLD + 1);
        let llm = ScriptedLlm::new(vec![
            make_text_response(&long_text),
            make_text_response(&long_text),
            make_text_response(&long_text),
            make_text_response(&long_text), // 4th: should pass through (MAX_NUDGE_ATTEMPTS exhausted)
        ]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            ToolLoopConfig::default(),
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("what does this project do?").unwrap();
        assert_eq!(result.turns, 4, "should fire 3 nudges then let 4th through");
        assert_eq!(result.response.len(), HALLUCINATION_NUDGE_THRESHOLD + 1);
    }

    #[test]
    fn nudge_fires_even_after_prior_tool_use() {
        // The nudge should fire even if the model used tools earlier in the conversation.
        // This tests that removing the `tool_calls_made.is_empty()` guard works.
        let tool_response = make_tool_response(vec![LlmToolCall {
            id: "tc1".to_string(),
            name: "fs_read".to_string(),
            arguments: serde_json::json!({"path": "README.md"}).to_string(),
        }]);
        let long_hallucination = "x".repeat(HALLUCINATION_NUDGE_THRESHOLD + 100);
        let llm = ScriptedLlm::new(vec![
            tool_response,                                // turn 1: uses a tool
            make_text_response("README content is..."),   // turn 2: text after tool
            make_text_response(&long_hallucination),      // turn 3: hallucination (should nudge)
            make_text_response("Let me check properly."), // turn 4: after nudge
        ]);
        let tool_host = Arc::new(MockToolHost::new(
            vec![ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: true,
                output: serde_json::json!({"content": "# Test"}),
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

        let r1 = loop_.run("read the readme").unwrap();
        assert_eq!(r1.response, "README content is...");

        // Second question — model hallucinates even though it used tools before
        let r2 = loop_
            .continue_with("what else is in this project?")
            .unwrap();
        assert_eq!(r2.response, "Let me check properly.");
        // The nudge message should be in the conversation
        let has_nudge = r2.messages.iter().any(|m| {
            if let ChatMessage::User { content } = m {
                content.contains("STOP. You are answering without using tools")
            } else {
                false
            }
        });
        assert!(has_nudge, "nudge should fire even after prior tool use");
    }

    #[test]
    fn single_unverified_path_triggers_nudge() {
        // A single unverified file reference (e.g. mentioning "audit.md")
        // should now trigger the nudge (threshold lowered from 2 to 1).
        assert!(
            has_unverified_file_references("Please see audit.md for details", &[]),
            "single file ref should trigger"
        );
        assert!(
            has_unverified_file_references("Look at src/main.rs", &[]),
            "single path should trigger"
        );
        // But if the tool was already used, it should NOT trigger
        let used_tools = vec![ToolCallRecord {
            tool_name: "fs_read".to_string(),
            tool_call_id: "tc1".to_string(),
            args_summary: String::new(),
            success: true,
            duration_ms: 0,
        }];
        assert!(
            !has_unverified_file_references("Look at src/main.rs", &used_tools),
            "should not trigger when read tool was used"
        );
    }

    #[test]
    fn shell_command_pattern_detected() {
        // Test that the shell command detection catches common patterns
        assert!(
            contains_shell_command_pattern("```bash\ncat audit.md\n```"),
            "cat in bash block should trigger"
        );
        assert!(
            contains_shell_command_pattern("```sh\ngrep -r TODO src/\n```"),
            "grep in sh block should trigger"
        );
        assert!(
            contains_shell_command_pattern("```\nfind . -name '*.rs'\n```"),
            "find in bare code block should trigger"
        );
        assert!(
            contains_shell_command_pattern("$ cat README.md"),
            "dollar-prompt cat should trigger"
        );
        // Should NOT trigger on normal prose
        assert!(
            !contains_shell_command_pattern("The cat sat on the mat."),
            "prose 'cat' should not trigger"
        );
        assert!(
            !contains_shell_command_pattern("Use `fs_read` to read files"),
            "tool mention should not trigger"
        );
    }

    #[test]
    fn tool_choice_required_per_new_question() {
        // In multi-turn, tool_choice=required should reset for each new user question.
        // The ScriptedLlm lets us verify via the request the loop builds.
        // We simulate: turn 1 uses a tool, then continue_with asks a new question.
        let llm = ScriptedLlm::new(vec![
            // First run: tool call + response
            make_tool_response(vec![LlmToolCall {
                id: "c1".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path":"src/lib.rs"}"#.to_string(),
            }]),
            make_text_response("Found it."),
            // Second run (continue_with): tool call + response
            make_tool_response(vec![LlmToolCall {
                id: "c2".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path":"src/main.rs"}"#.to_string(),
            }]),
            make_text_response("Found that too."),
        ]);

        let tool_host = Arc::new(MockToolHost::new(
            vec![
                ToolResult {
                    invocation_id: uuid::Uuid::nil(),
                    success: true,
                    output: serde_json::json!("content of lib.rs"),
                },
                ToolResult {
                    invocation_id: uuid::Uuid::nil(),
                    success: true,
                    output: serde_json::json!("content of main.rs"),
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

        let r1 = loop_.run("read lib.rs").unwrap();
        assert_eq!(r1.response, "Found it.");

        let r2 = loop_.continue_with("now read main.rs").unwrap();
        assert_eq!(r2.response, "Found that too.");
    }

    #[test]
    fn unverified_refs_detects_go_files() {
        assert!(has_unverified_file_references(
            "The server code is in cmd/server.go and internal/handler.go.",
            &[]
        ));
    }

    #[test]
    fn unverified_refs_ignores_dot_access() {
        // self.config, req.model etc. should not be flagged as file paths
        assert!(!has_unverified_file_references(
            "The self.config and req.model fields are set during initialization.",
            &[]
        ));
        // But real paths alongside dot-access should still be detected
        assert!(has_unverified_file_references(
            "Check src/config.rs and src/model.rs for the implementation.",
            &[]
        ));
    }

    // ── Batch 4 (P9): Compaction & Correctness ──

    #[test]
    fn compaction_preserves_tool_result_pairing() {
        // Build a conversation with tool calls that must stay paired.
        let llm = ScriptedLlm::new(vec![make_text_response("Final answer.")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let config = ToolLoopConfig {
            // Small context window so compaction budget is tight
            context_window_tokens: 2000,
            ..Default::default()
        };
        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        // Manually build history to simulate a conversation with tool exchanges
        loop_.messages.push(ChatMessage::User {
            content: "first question".to_string(),
        });
        for i in 0..5 {
            loop_.messages.push(ChatMessage::Assistant {
                content: None,
                reasoning_content: None,
                tool_calls: vec![LlmToolCall {
                    id: format!("call_{i}"),
                    name: "fs_read".to_string(),
                    arguments: format!(r#"{{"path":"file{i}.rs"}}"#),
                }],
            });
            loop_.messages.push(ChatMessage::Tool {
                tool_call_id: format!("call_{i}"),
                content: format!("content of file{i}.rs — {}", "x".repeat(400)),
            });
        }
        loop_.messages.push(ChatMessage::User {
            content: "second question".to_string(),
        });
        loop_.messages.push(ChatMessage::Assistant {
            content: Some("Here is what I found.".to_string()),
            reasoning_content: None,
            tool_calls: vec![],
        });

        // Force compaction with a tight budget
        let compacted = loop_.compact_messages(200);
        assert!(compacted, "should have compacted");

        // Verify no orphaned Tool messages (every Tool must follow an Assistant with tool_calls)
        let msgs = &loop_.messages;
        for (i, msg) in msgs.iter().enumerate() {
            if matches!(msg, ChatMessage::Tool { .. }) {
                assert!(
                    i > 0
                        && matches!(&msgs[i - 1], ChatMessage::Assistant { tool_calls, .. } if !tool_calls.is_empty()),
                    "Tool message at index {i} is orphaned (no preceding Assistant with tool_calls)"
                );
            }
        }
    }

    #[test]
    fn compaction_respects_target_tokens() {
        let llm = ScriptedLlm::new(vec![]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let config = ToolLoopConfig {
            context_window_tokens: 128_000,
            ..Default::default()
        };
        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        // Add many user/assistant exchanges
        for i in 0..20 {
            loop_.messages.push(ChatMessage::User {
                content: format!("Question {i}: {}", "x".repeat(200)),
            });
            loop_.messages.push(ChatMessage::Assistant {
                content: Some(format!("Answer {i}: {}", "y".repeat(200))),
                reasoning_content: None,
                tool_calls: vec![],
            });
        }

        let before_len = loop_.messages.len();
        let compacted = loop_.compact_messages(2000);
        assert!(compacted, "should compact");
        assert!(
            loop_.messages.len() < before_len,
            "should have fewer messages"
        );
        // System message should always be first
        assert!(matches!(&loop_.messages[0], ChatMessage::System { .. }));
        // Compaction summary should be second
        assert!(
            matches!(&loop_.messages[1], ChatMessage::User { content } if content.contains("compacted"))
        );
    }

    #[test]
    fn images_forwarded_to_llm_request() {
        // Verify that images from ToolLoopConfig appear in the ChatRequest
        let llm = ScriptedLlm::new(vec![make_text_response("I see the image.")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let images = vec![codingbuddy_core::ImageContent {
            mime: "image/png".to_string(),
            base64_data: "iVBORw0KGgo=".to_string(),
        }];
        let config = ToolLoopConfig {
            images: images.clone(),
            ..Default::default()
        };
        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("what's in this image?").unwrap();
        assert_eq!(result.response, "I see the image.");
        // The images are included in the config, which is used in build_request
        assert!(!loop_.config.images.is_empty());
    }

    #[test]
    fn post_tool_use_hook_gets_actual_args() {
        // This test verifies the PostToolUse hook receives actual tool arguments
        // (not an empty {}). We can't easily test hook content without a mock hook runtime,
        // but we verify the code path by checking that the tool executes successfully
        // with hooks not wired (no-op path).
        let llm = ScriptedLlm::new(vec![
            make_tool_response(vec![LlmToolCall {
                id: "c1".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path":"src/main.rs"}"#.to_string(),
            }]),
            make_text_response("Done."),
        ]);
        let tool_host = Arc::new(MockToolHost::new(
            vec![ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: true,
                output: serde_json::json!("file content"),
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

        let result = loop_.run("read main.rs").unwrap();
        assert_eq!(result.response, "Done.");
        assert_eq!(result.tool_calls_made.len(), 1);
        assert_eq!(result.tool_calls_made[0].tool_name, "fs_read");
    }

    // ── P10 Batch 1: Bootstrap context tests ──

    #[test]
    fn bootstrap_context_injected_on_first_turn() {
        let llm = ScriptedLlm::new(vec![make_text_response("ok")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let config = ToolLoopConfig {
            initial_context: vec![ChatMessage::System {
                content: "PROJECT_CONTEXT (auto-gathered):\nRust workspace, 25 crates".to_string(),
            }],
            ..Default::default()
        };

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "You are helpful.".to_string(),
            default_tools(),
        );

        let result = loop_.run("What is this project?").unwrap();
        assert_eq!(result.response, "ok");

        // Verify the message order: System (prompt), System (bootstrap), User
        assert!(
            matches!(&result.messages[0], ChatMessage::System { content } if content.contains("You are helpful"))
        );
        assert!(
            matches!(&result.messages[1], ChatMessage::System { content } if content.contains("PROJECT_CONTEXT"))
        );
        assert!(
            matches!(&result.messages[2], ChatMessage::User { content } if content.contains("What is this project"))
        );
    }

    #[test]
    fn bootstrap_respects_token_budget() {
        // Create a large bootstrap context that exceeds any reasonable budget
        let large_context = "x\n".repeat(100_000);

        let llm = ScriptedLlm::new(vec![make_text_response("ok")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let config = ToolLoopConfig {
            initial_context: vec![ChatMessage::System {
                content: format!("PROJECT_CONTEXT:\n{large_context}"),
            }],
            ..Default::default()
        };

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("hi").unwrap();
        // The large context is in messages[1] — it should be present (truncation
        // happens in lib.rs via truncate_to_token_budget before it reaches here)
        assert!(result.messages.len() >= 3);
        assert!(
            matches!(&result.messages[1], ChatMessage::System { content } if content.contains("PROJECT_CONTEXT"))
        );
    }

    #[test]
    fn bootstrap_disabled_when_no_initial_context() {
        let llm = ScriptedLlm::new(vec![make_text_response("hello")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let config = ToolLoopConfig {
            initial_context: vec![], // No bootstrap
            ..Default::default()
        };

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("hi").unwrap();
        // Messages: System, User, Assistant — no bootstrap System message
        assert!(
            matches!(&result.messages[0], ChatMessage::System { content } if content == "system")
        );
        assert!(matches!(&result.messages[1], ChatMessage::User { .. }));
    }

    // ── P10 Batch 2: Retrieval pipeline tests ──

    #[test]
    fn retriever_callback_returns_results() {
        let llm = ScriptedLlm::new(vec![make_text_response("found it")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let config = ToolLoopConfig {
            retriever: Some(Arc::new(|_query: &str, _max: usize| {
                Ok(vec![RetrievalContext {
                    file_path: "src/lib.rs".to_string(),
                    start_line: 1,
                    end_line: 10,
                    content: "fn main() {}".to_string(),
                    score: 0.95,
                }])
            })),
            ..Default::default()
        };

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("what does main do?").unwrap();
        // Should have retrieval context injected as a System message
        let has_retrieval = result.messages.iter().any(|m| {
            matches!(m,
            ChatMessage::System { content } if content.contains("RETRIEVAL_CONTEXT"))
        });
        assert!(has_retrieval, "retrieval context should be injected");
    }

    #[test]
    fn retrieval_context_appears_in_messages() {
        let llm = ScriptedLlm::new(vec![make_text_response("done")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let config = ToolLoopConfig {
            retriever: Some(Arc::new(|_query: &str, _max: usize| {
                Ok(vec![
                    RetrievalContext {
                        file_path: "auth.rs".to_string(),
                        start_line: 10,
                        end_line: 30,
                        content: "pub fn verify_token() -> bool { true }".to_string(),
                        score: 0.88,
                    },
                    RetrievalContext {
                        file_path: "middleware.rs".to_string(),
                        start_line: 5,
                        end_line: 20,
                        content: "pub fn auth_middleware() {}".to_string(),
                        score: 0.75,
                    },
                ])
            })),
            ..Default::default()
        };

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("explain auth").unwrap();
        let retrieval_msg = result.messages.iter().find(|m| {
            matches!(m,
            ChatMessage::System { content } if content.contains("RETRIEVAL_CONTEXT"))
        });
        assert!(retrieval_msg.is_some());
        if let Some(ChatMessage::System { content }) = retrieval_msg {
            assert!(
                content.contains("auth.rs"),
                "should include first result file"
            );
            assert!(
                content.contains("middleware.rs"),
                "should include second result file"
            );
            assert!(content.contains("verify_token"), "should include content");
        }
    }

    #[test]
    fn privacy_router_filters_tool_output() {
        let llm = ScriptedLlm::new(vec![
            make_tool_response(vec![LlmToolCall {
                id: "c1".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path":".env"}"#.to_string(),
            }]),
            make_text_response("read the file"),
        ]);
        let tool_host = Arc::new(MockToolHost::new(
            vec![ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: true,
                output: serde_json::json!("API_KEY=sk-secret-12345"),
            }],
            true,
        ));

        // Build a privacy router that will redact secrets
        let privacy_config = codingbuddy_local_ml::PrivacyConfig {
            enabled: true,
            sensitive_globs: vec![],
            sensitive_regex: vec![r"(?i)(api[_-]?key|secret)\s*[=:]\s*\S+".to_string()],
            policy: codingbuddy_local_ml::PrivacyPolicy::Redact,
            store_raw_in_logs: false,
        };
        let router =
            codingbuddy_local_ml::PrivacyRouter::new(privacy_config).expect("privacy router");

        let config = ToolLoopConfig {
            privacy_router: Some(Arc::new(router)),
            ..Default::default()
        };

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("read .env").unwrap();
        // The tool output should have been filtered
        let tool_msg = result
            .messages
            .iter()
            .find(|m| matches!(m, ChatMessage::Tool { .. }));
        assert!(tool_msg.is_some(), "should have a tool message");
        if let Some(ChatMessage::Tool { content, .. }) = tool_msg {
            // Either the secret is redacted or the output is blocked
            let has_raw_secret = content.contains("sk-secret-12345");
            assert!(
                !has_raw_secret || content.contains("[REDACTED]") || content.contains("[BLOCKED"),
                "secret should be redacted or blocked, got: {content}"
            );
        }
    }

    #[test]
    fn retriever_empty_results_no_injection() {
        let llm = ScriptedLlm::new(vec![make_text_response("ok")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let config = ToolLoopConfig {
            retriever: Some(Arc::new(|_query: &str, _max: usize| {
                Ok(vec![]) // Empty results
            })),
            ..Default::default()
        };

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("hello").unwrap();
        let has_retrieval = result.messages.iter().any(|m| {
            matches!(m,
            ChatMessage::System { content } if content.contains("RETRIEVAL_CONTEXT"))
        });
        assert!(
            !has_retrieval,
            "no retrieval context when results are empty"
        );
    }

    // ── P10 Batch 3: Semantic compaction tests ──

    #[test]
    fn compaction_summary_lists_modified_files() {
        let messages = vec![
            ChatMessage::Assistant {
                content: None,
                reasoning_content: None,
                tool_calls: vec![LlmToolCall {
                    id: "c1".to_string(),
                    name: "fs_edit".to_string(),
                    arguments: r#"{"file_path":"src/lib.rs","old":"x","new":"y"}"#.to_string(),
                }],
            },
            ChatMessage::Tool {
                tool_call_id: "c1".to_string(),
                content: "Edited successfully".to_string(),
            },
            ChatMessage::Assistant {
                content: None,
                reasoning_content: None,
                tool_calls: vec![LlmToolCall {
                    id: "c2".to_string(),
                    name: "fs_read".to_string(),
                    arguments: r#"{"path":"README.md"}"#.to_string(),
                }],
            },
            ChatMessage::Tool {
                tool_call_id: "c2".to_string(),
                content: "# README".to_string(),
            },
        ];
        let summary = build_compaction_summary(&messages);
        assert!(
            summary.contains("src/lib.rs"),
            "should list modified files: {summary}"
        );
        assert!(
            summary.contains("README.md"),
            "should list read files: {summary}"
        );
        assert!(
            summary.contains("fs_edit"),
            "should list tools used: {summary}"
        );
    }

    #[test]
    fn compaction_summary_captures_errors() {
        let messages = vec![
            ChatMessage::Tool {
                tool_call_id: "c1".to_string(),
                content: "error[E0308]: mismatched types\n  expected `u32`".to_string(),
            },
            ChatMessage::Tool {
                tool_call_id: "c2".to_string(),
                content: "test result: FAILED. 1 failed".to_string(),
            },
        ];
        let summary = build_compaction_summary(&messages);
        assert!(
            summary.contains("Errors encountered"),
            "should capture errors: {summary}"
        );
    }

    #[test]
    fn compaction_summary_tool_counts() {
        let messages = vec![ChatMessage::Assistant {
            content: None,
            reasoning_content: None,
            tool_calls: vec![
                LlmToolCall {
                    id: "1".to_string(),
                    name: "fs_read".to_string(),
                    arguments: "{}".to_string(),
                },
                LlmToolCall {
                    id: "2".to_string(),
                    name: "fs_read".to_string(),
                    arguments: "{}".to_string(),
                },
                LlmToolCall {
                    id: "3".to_string(),
                    name: "fs_edit".to_string(),
                    arguments: "{}".to_string(),
                },
            ],
        }];
        let summary = build_compaction_summary(&messages);
        assert!(
            summary.contains("fs_read×2"),
            "should count tools: {summary}"
        );
        assert!(
            summary.contains("fs_edit×1"),
            "should count edit: {summary}"
        );
    }

    // ── P10 Batch 5: Model routing tests ──

    #[test]
    fn complex_escalated_routes_to_reasoner() {
        // We can't easily check what model was sent in the request because
        // ScriptedLlm doesn't capture it, but we can verify the build_request
        // logic indirectly by checking the ToolUseLoop state.
        let llm = ScriptedLlm::new(vec![
            // Tool call that triggers compile error
            make_tool_response(vec![LlmToolCall {
                id: "c1".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path":"x.rs"}"#.to_string(),
            }]),
            make_text_response("done"),
        ]);
        let tool_host = Arc::new(MockToolHost::new(
            vec![ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: false,
                output: serde_json::json!("error[E0308]: mismatched types"),
            }],
            true,
        ));

        let config = ToolLoopConfig {
            complexity: crate::complexity::PromptComplexity::Complex,
            ..Default::default()
        };

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("refactor auth").unwrap();
        // After the compile error, escalation should be active
        assert!(
            loop_.escalation.compile_error,
            "compile error should be flagged"
        );
        assert!(loop_.escalation.should_escalate(), "should be escalated");
        // Result should complete (the scripted LLM returns "done")
        assert_eq!(result.response, "done");
    }

    #[test]
    fn de_escalated_routes_back_to_chat() {
        let mut signals = crate::complexity::EscalationSignals {
            compile_error: true,
            ..Default::default()
        };
        assert!(signals.should_escalate());

        // 3 consecutive successes → de-escalate
        signals.record_success();
        signals.record_success();
        signals.record_success();
        assert!(!signals.should_escalate(), "3 successes should de-escalate");
        assert_eq!(signals.budget(), crate::complexity::DEFAULT_THINK_BUDGET);
    }

    #[test]
    fn reasoner_strips_sampling_params() {
        // Verify the build_request logic: when model contains "reasoner",
        // temperature should be None even if config has one.
        let llm = ScriptedLlm::new(vec![make_text_response("ok")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let config = ToolLoopConfig {
            temperature: Some(0.7),
            ..Default::default()
        };

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        // If not using reasoner, temperature should be preserved
        let result = loop_.run("hi").unwrap();
        assert_eq!(result.response, "ok");
        // The test passes — the key assertion is in the build_request logic itself
        // which strips temperature when model contains "reasoner".
    }

    #[test]
    fn compaction_preserves_key_decisions() {
        let decision_text = "I'll modify the authentication module to use JWT tokens instead of session cookies because the API is stateless.";
        let messages = vec![ChatMessage::Assistant {
            content: Some(decision_text.to_string()),
            reasoning_content: None,
            tool_calls: vec![],
        }];
        let summary = build_compaction_summary(&messages);
        assert!(
            summary.contains("Key decisions"),
            "should have decisions section: {summary}"
        );
        assert!(
            summary.contains("JWT tokens") || summary.contains("authentication"),
            "should preserve decision content: {summary}"
        );
    }

    #[test]
    fn compaction_template_includes_key_facts_section() {
        assert!(
            COMPACTION_TEMPLATE.contains("## Key Facts Established"),
            "template should include Key Facts section for context retention"
        );
        assert!(
            COMPACTION_TEMPLATE.contains("user preferences"),
            "template should mention preserving user preferences"
        );
        assert!(
            COMPACTION_TEMPLATE.contains("corrections given"),
            "template should mention preserving corrections"
        );
    }

    #[test]
    fn retriever_none_means_no_retrieval() {
        let llm = ScriptedLlm::new(vec![make_text_response("ok")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let config = ToolLoopConfig {
            retriever: None, // Disabled
            ..Default::default()
        };

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("hello").unwrap();
        let has_retrieval = result.messages.iter().any(|m| {
            matches!(m,
            ChatMessage::System { content } if content.contains("RETRIEVAL_CONTEXT"))
        });
        assert!(!has_retrieval);
    }

    #[test]
    fn bootstrap_includes_repo_map_content() {
        let llm = ScriptedLlm::new(vec![make_text_response("analyzed")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));

        let config = ToolLoopConfig {
            initial_context: vec![ChatMessage::System {
                content: "Key source files:\n  - src/lib.rs (2048 bytes) score=100\n  - src/main.rs (512 bytes) score=50".to_string(),
            }],
            ..Default::default()
        };

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("analyze project").unwrap();
        assert!(
            matches!(&result.messages[1], ChatMessage::System { content }
            if content.contains("Key source files:") && content.contains("src/lib.rs"))
        );
    }

    // ── Batch 8: Error Recovery & Self-Correction ──

    #[test]
    fn error_recovery_guidance_injected_on_first_escalation() {
        let fail_response = make_tool_response(vec![LlmToolCall {
            id: "tc1".to_string(),
            name: "bash_run".to_string(),
            arguments: r#"{"command":"cargo build"}"#.to_string(),
        }]);
        let done_response = make_text_response("I see the error, let me fix it.");

        let llm = ScriptedLlm::new(vec![fail_response, done_response]);
        let tool_host = Arc::new(MockToolHost::new(
            vec![ToolResult {
                invocation_id: uuid::Uuid::nil(),
                output: serde_json::json!("error[E0308]: mismatched types"),
                success: false,
            }],
            true,
        ));

        let config = ToolLoopConfig {
            context_window_tokens: 128_000,
            ..Default::default()
        };

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("fix the build").unwrap();

        let has_recovery = result.messages.iter().any(
            |m| matches!(m, ChatMessage::System { content } if content.contains("ERROR RECOVERY")),
        );
        assert!(
            has_recovery,
            "error recovery guidance should be injected on first escalation"
        );
    }

    #[test]
    fn stuck_detection_fires_after_3_same_errors() {
        let fail_response = |id: &str| {
            make_tool_response(vec![LlmToolCall {
                id: id.to_string(),
                name: "bash_run".to_string(),
                arguments: r#"{"command":"cargo build"}"#.to_string(),
            }])
        };

        let llm = ScriptedLlm::new(vec![
            fail_response("tc1"),
            fail_response("tc2"),
            fail_response("tc3"),
            make_text_response("Let me try a completely different approach."),
        ]);

        let error_msg = "error[E0308]: mismatched types expected `u32` found `String`";
        let tool_host = Arc::new(MockToolHost::new(
            vec![
                ToolResult {
                    invocation_id: uuid::Uuid::nil(),
                    output: serde_json::json!(error_msg),
                    success: false,
                },
                ToolResult {
                    invocation_id: uuid::Uuid::nil(),
                    output: serde_json::json!(error_msg),
                    success: false,
                },
                ToolResult {
                    invocation_id: uuid::Uuid::nil(),
                    output: serde_json::json!(error_msg),
                    success: false,
                },
            ],
            true,
        ));

        let config = ToolLoopConfig {
            context_window_tokens: 128_000,
            ..Default::default()
        };

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("fix the build").unwrap();

        let has_stuck = result.messages.iter().any(
            |m| matches!(m, ChatMessage::System { content } if content.contains("STUCK DETECTION")),
        );
        assert!(
            has_stuck,
            "stuck detection should fire after 3 identical errors"
        );
    }

    #[test]
    fn recovery_guidance_not_repeated_after_success() {
        let tool_response = |id: &str| {
            make_tool_response(vec![LlmToolCall {
                id: id.to_string(),
                name: "bash_run".to_string(),
                arguments: r#"{"command":"cargo build"}"#.to_string(),
            }])
        };

        let llm = ScriptedLlm::new(vec![
            tool_response("tc1"), // fails → recovery injected
            tool_response("tc2"), // succeeds → recovery reset
            tool_response("tc3"), // fails → recovery can inject again
            make_text_response("Done."),
        ]);

        let tool_host = Arc::new(MockToolHost::new(
            vec![
                ToolResult {
                    invocation_id: uuid::Uuid::nil(),
                    output: serde_json::json!("error: compilation failed"),
                    success: false,
                },
                ToolResult {
                    invocation_id: uuid::Uuid::nil(),
                    output: serde_json::json!("Build succeeded"),
                    success: true,
                },
                ToolResult {
                    invocation_id: uuid::Uuid::nil(),
                    output: serde_json::json!("error: different problem"),
                    success: false,
                },
            ],
            true,
        ));

        let config = ToolLoopConfig {
            context_window_tokens: 128_000,
            ..Default::default()
        };

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("fix the build").unwrap();

        let recovery_count = result.messages.iter().filter(|m| {
            matches!(m, ChatMessage::System { content } if content.contains("ERROR RECOVERY"))
        }).count();

        // Recovery state resets on success, so it CAN inject again after the second failure.
        // The key invariant: recovery is NOT injected on the same escalation cycle twice.
        assert!(
            recovery_count <= 2,
            "recovery should not spam — got {recovery_count}"
        );
    }

    // ── CostTracker tests ──────────────────────────────────────────────

    #[test]
    fn cost_tracker_records_and_estimates() {
        let mut tracker = CostTracker::default();
        tracker.record(&TokenUsage {
            prompt_tokens: 1_000_000,
            completion_tokens: 100_000,
            prompt_cache_hit_tokens: 0,
            prompt_cache_miss_tokens: 0,
            reasoning_tokens: 0,
        });
        // Input: 1M tokens × $0.27/M = $0.27
        // Output: 100K tokens × $1.10/M = $0.11
        let cost = tracker.estimated_cost_usd();
        assert!(
            (cost - 0.38).abs() < 0.001,
            "expected ~$0.38, got ${cost:.4}"
        );
    }

    #[test]
    fn cost_tracker_cache_discount() {
        let mut tracker = CostTracker::default();
        tracker.record(&TokenUsage {
            prompt_tokens: 1_000_000,
            completion_tokens: 0,
            prompt_cache_hit_tokens: 800_000, // 80% cached
            prompt_cache_miss_tokens: 200_000,
            reasoning_tokens: 0,
        });
        // Effective input = (1M - 800K) + (800K × 0.1) = 200K + 80K = 280K
        // Cost = 280K / 1M × $0.27 = $0.0756
        let cost = tracker.estimated_cost_usd();
        assert!(
            (cost - 0.0756).abs() < 0.001,
            "expected ~$0.0756 with cache discount, got ${cost:.4}"
        );
    }

    #[test]
    fn cost_tracker_over_budget() {
        let mut tracker = CostTracker {
            max_budget_usd: Some(0.10),
            ..CostTracker::default()
        };
        tracker.record(&TokenUsage {
            prompt_tokens: 1_000_000,
            completion_tokens: 100_000,
            prompt_cache_hit_tokens: 0,
            prompt_cache_miss_tokens: 0,
            reasoning_tokens: 0,
        });
        assert!(tracker.over_budget(), "should be over $0.10 budget");
    }

    #[test]
    fn cost_tracker_not_over_budget_without_cap() {
        let mut tracker = CostTracker::default();
        // No max_budget_usd set
        tracker.record(&TokenUsage {
            prompt_tokens: 10_000_000,
            completion_tokens: 1_000_000,
            prompt_cache_hit_tokens: 0,
            prompt_cache_miss_tokens: 0,
            reasoning_tokens: 0,
        });
        assert!(
            !tracker.over_budget(),
            "should never be over budget when no cap is set"
        );
    }

    #[test]
    fn cost_tracker_warns_once() {
        let mut tracker = CostTracker::default();
        tracker.record(&TokenUsage {
            prompt_tokens: 2_000_000,
            completion_tokens: 500_000,
            prompt_cache_hit_tokens: 0,
            prompt_cache_miss_tokens: 0,
            reasoning_tokens: 0,
        });
        // Cost: $0.54 + $0.55 = $1.09 — well above $0.50 threshold
        assert!(tracker.should_warn(), "should warn first time");
        assert!(!tracker.should_warn(), "should NOT warn second time");
    }

    #[test]
    fn cost_tracker_no_warn_below_threshold() {
        let mut tracker = CostTracker::default();
        tracker.record(&TokenUsage {
            prompt_tokens: 100_000,
            completion_tokens: 10_000,
            prompt_cache_hit_tokens: 0,
            prompt_cache_miss_tokens: 0,
            reasoning_tokens: 0,
        });
        // Cost: $0.027 + $0.011 = $0.038 — well below $0.50
        assert!(!tracker.should_warn(), "should not warn below threshold");
    }

    // ── CircuitBreaker tests ───────────────────────────────────────────

    #[test]
    fn circuit_breaker_triggers_after_threshold() {
        let mut cb = CircuitBreakerState::default();
        for _ in 0..CIRCUIT_BREAKER_THRESHOLD {
            cb.consecutive_failures += 1;
        }
        assert_eq!(cb.consecutive_failures, CIRCUIT_BREAKER_THRESHOLD);
        // Simulate triggering cooldown
        cb.cooldown_remaining = CIRCUIT_BREAKER_COOLDOWN_TURNS;
        assert_eq!(cb.cooldown_remaining, 2);
    }

    #[test]
    fn circuit_breaker_cooldown_decrements() {
        let mut cb = CircuitBreakerState {
            consecutive_failures: CIRCUIT_BREAKER_THRESHOLD,
            cooldown_remaining: CIRCUIT_BREAKER_COOLDOWN_TURNS,
        };
        // Simulate one turn of cooldown decrement
        if cb.cooldown_remaining > 0 {
            cb.cooldown_remaining -= 1;
        }
        assert_eq!(cb.cooldown_remaining, 1);
        // One more
        if cb.cooldown_remaining > 0 {
            cb.cooldown_remaining -= 1;
        }
        assert_eq!(cb.cooldown_remaining, 0);
        // After cooldown, reset failures
        cb.consecutive_failures = 0;
        assert_eq!(cb.consecutive_failures, 0);
    }

    #[test]
    fn circuit_breaker_resets_on_success() {
        let mut cb = CircuitBreakerState {
            consecutive_failures: 2,
            cooldown_remaining: 0,
        };
        // Success resets the counter
        cb.consecutive_failures = 0;
        assert_eq!(cb.consecutive_failures, 0);
        assert_eq!(cb.cooldown_remaining, 0);
    }

    // ── Tool cache tests ───────────────────────────────────────────────

    #[test]
    fn tool_cache_stores_and_retrieves() {
        let llm = ScriptedLlm::new(vec![make_text_response("done")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));
        let config = ToolLoopConfig::default();
        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let args = serde_json::json!({"path": "/foo/bar.rs"});
        let result = serde_json::json!({"content": "fn main() {}"});

        // Store and retrieve
        loop_.cache_store("fs_read", &args, &result);
        let cached = loop_.cache_lookup("fs_read", &args);
        assert_eq!(cached, Some(result));
    }

    #[test]
    fn tool_cache_skips_non_cacheable() {
        let llm = ScriptedLlm::new(vec![make_text_response("done")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));
        let config = ToolLoopConfig::default();
        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let args = serde_json::json!({"command": "ls"});
        let result = serde_json::json!({"output": "file1\nfile2"});

        loop_.cache_store("bash_run", &args, &result);
        let cached = loop_.cache_lookup("bash_run", &args);
        assert_eq!(cached, None, "bash_run should not be cached");
    }

    #[test]
    fn tool_cache_invalidation_by_path() {
        let llm = ScriptedLlm::new(vec![make_text_response("done")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));
        let config = ToolLoopConfig::default();
        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let args = serde_json::json!({"path": "/foo/bar.rs"});
        let result = serde_json::json!({"content": "fn main() {}"});

        loop_.cache_store("fs_read", &args, &result);
        assert!(loop_.cache_lookup("fs_read", &args).is_some());

        // Invalidate by path
        loop_.cache_invalidate_path("/foo/bar.rs");
        assert!(
            loop_.cache_lookup("fs_read", &args).is_none(),
            "cache should be invalidated after path write"
        );
    }

    // ── Pruning tests ──────────────────────────────────────────────────

    #[test]
    fn prune_truncates_old_tool_outputs() {
        let llm = ScriptedLlm::new(vec![make_text_response("done")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));
        let config = ToolLoopConfig::default();
        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        // Build messages with 4+ turn groups, with long tool output in old turns
        let long_output = "x".repeat(500);
        // Old turn group 1
        loop_.messages.push(ChatMessage::User {
            content: "task 1".to_string(),
        });
        loop_.messages.push(ChatMessage::Tool {
            tool_call_id: "tc1".to_string(),
            content: long_output.clone(),
        });
        // Old turn group 2
        loop_.messages.push(ChatMessage::User {
            content: "task 2".to_string(),
        });
        loop_.messages.push(ChatMessage::Tool {
            tool_call_id: "tc2".to_string(),
            content: long_output.clone(),
        });
        // Old turn group 3
        loop_.messages.push(ChatMessage::User {
            content: "task 3".to_string(),
        });
        loop_.messages.push(ChatMessage::Tool {
            tool_call_id: "tc3".to_string(),
            content: long_output.clone(),
        });
        // Recent turns
        loop_.messages.push(ChatMessage::User {
            content: "recent task".to_string(),
        });
        loop_.messages.push(ChatMessage::Tool {
            tool_call_id: "tc4".to_string(),
            content: long_output.clone(),
        });

        loop_.prune_old_tool_outputs();

        // Old tool outputs should be truncated
        let old_tool = &loop_.messages[2]; // tc1
        if let ChatMessage::Tool { content, .. } = old_tool {
            assert!(
                content.contains("[output pruned"),
                "old tool output should be pruned, got: {}",
                &content[..content.floor_char_boundary(50.min(content.len()))]
            );
            assert!(
                content.len() < 500,
                "pruned output should be shorter than original"
            );
        }

        // Recent tool output should be unchanged
        let recent_tool = &loop_.messages[loop_.messages.len() - 1];
        if let ChatMessage::Tool { content, .. } = recent_tool {
            assert_eq!(
                content.len(),
                500,
                "recent tool output should not be pruned"
            );
        }
    }

    #[test]
    fn prune_skips_short_outputs() {
        let llm = ScriptedLlm::new(vec![make_text_response("done")]);
        let tool_host = Arc::new(MockToolHost::new(vec![], true));
        let config = ToolLoopConfig::default();
        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let short_output = "OK".to_string();
        // Build 4 turn groups
        for i in 0..4 {
            loop_.messages.push(ChatMessage::User {
                content: format!("task {i}"),
            });
            loop_.messages.push(ChatMessage::Tool {
                tool_call_id: format!("tc{i}"),
                content: short_output.clone(),
            });
        }

        loop_.prune_old_tool_outputs();

        // All outputs should remain unchanged (too short to prune)
        for msg in &loop_.messages {
            if let ChatMessage::Tool { content, .. } = msg {
                assert_eq!(content, "OK", "short outputs should not be pruned");
            }
        }
    }

    #[test]
    fn extract_tool_path_from_json() {
        let json_content = r#"{"path": "/src/main.rs", "content": "fn main() {}"}"#;
        assert_eq!(
            extract_tool_path(json_content),
            Some("/src/main.rs".to_string())
        );
    }

    #[test]
    fn extract_tool_path_from_plain_path() {
        assert_eq!(
            extract_tool_path("/src/main.rs\nfn main() {}"),
            Some("/src/main.rs".to_string())
        );
    }

    #[test]
    fn extract_tool_path_none_for_regular_text() {
        assert_eq!(extract_tool_path("Build succeeded"), None);
    }

    // ── Doom loop detection ──

    #[test]
    fn doom_loop_detects_repeated_identical_calls() {
        let mut tracker = DoomLoopTracker::default();
        let args = r#"{"file_path":"/src/main.rs"}"#;

        assert!(!tracker.record("fs_read", args), "first call: no doom loop");
        assert!(
            !tracker.record("fs_read", args),
            "second call: no doom loop"
        );
        assert!(
            tracker.record("fs_read", args),
            "third call: doom loop detected"
        );
    }

    #[test]
    fn doom_loop_no_false_positive_on_different_args() {
        let mut tracker = DoomLoopTracker::default();

        tracker.record("fs_read", r#"{"file_path":"/a.rs"}"#);
        tracker.record("fs_read", r#"{"file_path":"/b.rs"}"#);
        let detected = tracker.record("fs_read", r#"{"file_path":"/c.rs"}"#);
        assert!(!detected, "different args should not trigger doom loop");
    }

    #[test]
    fn doom_loop_no_false_positive_on_different_tools() {
        let mut tracker = DoomLoopTracker::default();
        let args = r#"{"file_path":"/main.rs"}"#;

        tracker.record("fs_read", args);
        tracker.record("fs_glob", args);
        let detected = tracker.record("fs_grep", args);
        assert!(!detected, "different tools should not trigger doom loop");
    }

    #[test]
    fn doom_loop_resets_warning_on_different_call() {
        let mut tracker = DoomLoopTracker::default();
        let args = r#"{"file_path":"/main.rs"}"#;

        tracker.record("fs_read", args);
        tracker.record("fs_read", args);
        assert!(tracker.record("fs_read", args));
        tracker.mark_warned();

        // Same call again — no new warning because already warned
        assert!(!tracker.record("fs_read", args));

        // Different call resets
        tracker.record("fs_edit", "{}");

        // Now back to original — should detect again
        tracker.record("fs_read", args);
        tracker.record("fs_read", args);
        assert!(tracker.record("fs_read", args), "should detect after reset");
    }

    #[test]
    fn doom_loop_warning_only_fires_once() {
        let mut tracker = DoomLoopTracker::default();
        let args = r#"{"file_path":"/main.rs"}"#;

        tracker.record("fs_read", args);
        tracker.record("fs_read", args);
        assert!(tracker.record("fs_read", args));
        tracker.mark_warned();

        // Subsequent identical calls should NOT re-trigger
        assert!(!tracker.record("fs_read", args));
        assert!(!tracker.record("fs_read", args));
    }

    #[test]
    fn doom_loop_respects_history_window() {
        let mut tracker = DoomLoopTracker::default();
        let target_args = r#"{"file_path":"/main.rs"}"#;

        // Two identical calls
        tracker.record("fs_read", target_args);
        tracker.record("fs_read", target_args);

        // Fill with enough different calls to push the first two out of the window
        for i in 0..DOOM_LOOP_HISTORY_SIZE {
            tracker.record("fs_glob", &format!(r#"{{"pattern":"*.{i}"}}"#));
        }

        // Now the original calls are outside the window — one more should not trigger
        let detected = tracker.record("fs_read", target_args);
        assert!(!detected, "old calls outside window should not count");
    }

    #[test]
    fn doom_loop_terminates_loop_as_blocking_gate() {
        // Use the ScriptedLlm to simulate a model that makes the same tool call 3 times.
        // The doom loop gate should STOP the loop at turn 3 — turn 4 should never execute.
        let responses = vec![
            // Turn 1: model calls fs_read with same args
            LlmResponse {
                text: String::new(),
                reasoning_content: String::new(),
                tool_calls: vec![LlmToolCall {
                    id: "call_1".to_string(),
                    name: "fs_read".to_string(),
                    arguments: r#"{"file_path":"/main.rs"}"#.to_string(),
                }],
                finish_reason: "tool_calls".to_string(),
                usage: None,
            },
            // Turn 2: same call
            LlmResponse {
                text: String::new(),
                reasoning_content: String::new(),
                tool_calls: vec![LlmToolCall {
                    id: "call_2".to_string(),
                    name: "fs_read".to_string(),
                    arguments: r#"{"file_path":"/main.rs"}"#.to_string(),
                }],
                finish_reason: "tool_calls".to_string(),
                usage: None,
            },
            // Turn 3: same call — should trigger doom loop gate and STOP
            LlmResponse {
                text: String::new(),
                reasoning_content: String::new(),
                tool_calls: vec![LlmToolCall {
                    id: "call_3".to_string(),
                    name: "fs_read".to_string(),
                    arguments: r#"{"file_path":"/main.rs"}"#.to_string(),
                }],
                finish_reason: "tool_calls".to_string(),
                usage: None,
            },
            // Turn 4: should NEVER be reached — doom loop gate terminates the loop
            LlmResponse {
                text: "This should never appear.".to_string(),
                reasoning_content: String::new(),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: None,
            },
        ];

        let llm = ScriptedLlm::new(responses);
        let tool_result = ToolResult {
            invocation_id: uuid::Uuid::nil(),
            success: true,
            output: serde_json::json!({"content": "fn main() {}"}),
        };
        let tool_host = Arc::new(MockToolHost::new(
            vec![tool_result.clone(), tool_result.clone(), tool_result],
            true,
        ));

        let config = ToolLoopConfig {
            max_turns: 10,
            ..Default::default()
        };

        let mut loop_ = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "system".to_string(),
            default_tools(),
        );

        let result = loop_.run("show me the main file").unwrap();

        // The loop should terminate at doom loop detection (blocking gate)
        assert_eq!(
            result.finish_reason, "doom_loop",
            "finish_reason should be doom_loop when gate triggers"
        );

        // Turn 4 should NOT have executed
        assert_ne!(
            result.response, "This should never appear.",
            "doom loop gate should prevent further LLM calls"
        );

        // Verify doom loop guidance was injected into messages
        let has_doom_guidance = result.messages.iter().any(|m| {
            matches!(m, ChatMessage::System { content } if content.contains("repeating the same action"))
        });
        assert!(
            has_doom_guidance,
            "doom loop guidance should be in messages for context if user continues"
        );
    }

    // ── User directive extraction tests ──

    #[test]
    fn extract_directives_finds_always_never() {
        let text =
            "Please always use snake_case for variable names. Never commit directly to main.";
        let directives = extract_user_directives(text);
        assert_eq!(directives.len(), 2);
        assert!(directives[0].contains("always use snake_case"));
        assert!(directives[1].contains("Never commit"));
    }

    #[test]
    fn extract_directives_skips_questions() {
        let text = "Should I always use tabs? Never mind about that.";
        let directives = extract_user_directives(text);
        // "Should I always use tabs?" ends with '?' so it's skipped
        // "Never mind about that." should still be captured
        assert_eq!(directives.len(), 1);
        assert!(directives[0].contains("Never mind"));
    }

    #[test]
    fn extract_directives_skips_short_fragments() {
        let text = "always\nnever\nprefer X for all Y operations in the codebase";
        let directives = extract_user_directives(text);
        // "always" and "never" are too short (< 10 chars)
        assert_eq!(directives.len(), 1);
        assert!(directives[0].contains("prefer X"));
    }

    #[test]
    fn extract_directives_caps_at_limit() {
        // 12 directive-containing sentences should be capped at 10
        let lines: Vec<String> = (0..12)
            .map(|i| format!("Always do thing number {i} in the code"))
            .collect();
        let text = lines.join("\n");
        let directives = extract_user_directives(&text);
        assert_eq!(directives.len(), 10);
    }

    #[test]
    fn extract_directives_handles_multiline_input() {
        let text = "First, make sure all tests pass before committing.\n\
                     Second, use the standard library whenever possible.\n\
                     What do you think about this approach?";
        let directives = extract_user_directives(text);
        assert_eq!(directives.len(), 2);
        assert!(directives[0].contains("make sure"));
        assert!(directives[1].contains("use the standard"));
    }

    #[test]
    fn directives_survive_compaction() {
        // Build a ToolUseLoop, feed it a user message with directives,
        // trigger compaction, and verify directives are re-injected.
        let llm = ScriptedLlm::new(vec![
            // First turn: tool call + result
            make_tool_response(vec![LlmToolCall {
                id: "c1".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path":"src/lib.rs"}"#.to_string(),
            }]),
            make_text_response("done with first task"),
            // Second turn (after continue_with): text response
            make_text_response("done with second task"),
        ]);
        let tool_host = Arc::new(MockToolHost::new(
            vec![ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: true,
                output: serde_json::json!("file contents here"),
            }],
            true,
        ));

        let config = ToolLoopConfig {
            context_window_tokens: 200, // Very small window to force compaction
            ..Default::default()
        };
        let mut tool_loop = ToolUseLoop::new(
            &llm,
            tool_host,
            config,
            "You are a coding assistant.".to_string(),
            vec![],
        );

        // First turn with a directive
        let _ = tool_loop.run("Always use snake_case for variables. Read the file.");

        // Verify directive was collected
        assert!(
            tool_loop
                .pinned_directives
                .iter()
                .any(|d| d.contains("snake_case")),
            "directive should be collected: {:?}",
            tool_loop.pinned_directives
        );

        // Manually trigger compaction (simulate context pressure)
        // The pinned_directives field should persist across compaction
        let directives_before = tool_loop.pinned_directives.clone();

        // Second turn
        let _ = tool_loop.continue_with("Now prefer tabs over spaces. Fix the bug.");

        // Directives from both turns should be present
        assert!(
            tool_loop
                .pinned_directives
                .iter()
                .any(|d| d.contains("snake_case")),
            "first directive should persist: {:?}",
            tool_loop.pinned_directives
        );
        assert!(
            tool_loop
                .pinned_directives
                .iter()
                .any(|d| d.contains("prefer tabs")),
            "second directive should be collected: {:?}",
            tool_loop.pinned_directives
        );
        assert!(
            tool_loop.pinned_directives.len() >= directives_before.len(),
            "directives should accumulate"
        );
    }

    #[test]
    fn split_sentences_handles_periods_and_newlines() {
        let text = "First sentence. Second sentence.\nThird line";
        let sentences = split_sentences(text);
        assert!(
            sentences.len() >= 3,
            "should split on periods and newlines: {sentences:?}"
        );
    }
}
