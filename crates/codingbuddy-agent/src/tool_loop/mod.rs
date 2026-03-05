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
//! - [`agent_tools`] — Agent-level tool handlers (tasks/todos/spawn/skills/plan mode)
//! - [`anti_hallucination`] — File reference validation, shell command detection, directive extraction
//! - [`compaction`] — Context pruning and compaction summaries
//! - [`helpers`] — Small utility functions

pub mod types;

mod agent_tools;
mod anti_hallucination;
mod compaction;
mod event_emission;
mod helpers;
pub mod phases;
mod safety;

// Re-export public API from submodules
pub use types::*;

use anyhow::{Result, anyhow};
use codingbuddy_core::{
    ApprovedToolCall, ChatMessage, ChatRequest, EventKind, LlmToolCall, Plan, PlanStep, Session,
    SessionState, StreamCallback, StreamChunk, TokenUsage, ToolCall, ToolChoice, ToolDefinition,
    ToolHost, UserQuestion, estimate_message_tokens, strip_prior_reasoning_content,
};
use codingbuddy_hooks::{HookEvent, HookInput, HookRuntime};
use codingbuddy_llm::{LlmClient, max_output_tokens_for_model};
use codingbuddy_store::{SessionTodoRecord, Store, TaskQueueRecord};
use codingbuddy_tools::{format_tool_search_results, search_extended_tools};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

use crate::tool_bridge;

use anti_hallucination::{
    HALLUCINATION_NUDGE, HALLUCINATION_NUDGE_THRESHOLD, MAX_NUDGE_ATTEMPTS,
    check_response_consistency, contains_shell_command_pattern, extract_user_directives,
    has_unverified_file_references,
};
use compaction::{
    COMPACTION_TARGET_PCT, PRUNE_AGE_TURNS, build_compaction_summary,
    build_compaction_summary_with_llm, extract_tool_path, truncate_line,
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

/// Minimum usable input budget during compaction to avoid zero-token edge cases.
const MIN_COMPACTION_INPUT_BUDGET_TOKENS: u64 = 1024;

/// Extra fixed overhead beyond tool definition size (system + control messages).
const COMPACTION_EXTRA_OVERHEAD_TOKENS: u64 = 512;

/// Every N tool calls, inject a brief system reminder to keep the model on track.
const MID_CONVERSATION_REMINDER_INTERVAL: usize = 10;

/// Return `Some(args_json)` only for read tools (needed for anti-hallucination
/// path extraction). Write/agent tools don't need the raw JSON stored.
fn args_json_for_record(tool_name: &str, raw_args: &str) -> Option<String> {
    if anti_hallucination::READ_TOOL_NAMES.contains(&tool_name) {
        Some(raw_args.to_string())
    } else {
        None
    }
}

/// Brief reminder injected every `MID_CONVERSATION_REMINDER_INTERVAL` tool calls.
const MID_CONVERSATION_REMINDER: &str =
    "Reminder: Verify changes with tests. Be concise. Use tools — do not guess.";

/// Runtime checklist policy for complex work.
const COMPLEX_TODO_POLICY: &str = "For complex tasks, maintain a live checklist with todo_read/todo_write. \
Initialize it before edits, keep exactly one in_progress item, and update it after each meaningful step.";

/// TTL for cached read-only tool results (in seconds).
const TOOL_CACHE_TTL_SECS: u64 = 60;

/// Tools whose results can be cached (all read-only).
const CACHEABLE_TOOLS: &[&str] = &["fs_read", "fs_glob", "fs_grep", "fs_list", "index_query"];

/// Maximum number of entries in the tool result cache.
/// When exceeded, the oldest entry (by insertion order approximation) is evicted.
const MAX_CACHE_ENTRIES: usize = 256;

/// Cached tool result entry.
struct CacheEntry {
    result: serde_json::Value,
    timestamp: Instant,
    /// Unhashed "tool_name:args" string for path-based invalidation.
    raw_key: String,
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
    discoverable_tools: Vec<ToolDefinition>,
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
    recent_errors: VecDeque<String>,
    /// Whether error recovery guidance has been injected this escalation cycle.
    recovery_injected: bool,
    /// Pre-compiled output scanner for injection/secret detection.
    output_scanner: codingbuddy_policy::output_scanner::OutputScanner,
    /// Cache for read-only tool results. Keyed by SHA-256 hash of (tool_name, args).
    /// Entries expire after `TOOL_CACHE_TTL_SECS`. Invalidated when write tools
    /// modify the cached path. Bounded at `MAX_CACHE_ENTRIES`.
    tool_cache: HashMap<String, CacheEntry>,
    /// Circuit breaker: tracks consecutive failures per tool name.
    /// When a tool fails `CIRCUIT_BREAKER_THRESHOLD` times in a row, it is
    /// disabled for `CIRCUIT_BREAKER_COOLDOWN_TURNS`.
    circuit_breaker: HashMap<String, CircuitBreakerState>,
    /// Cumulative cost tracking across the session.
    cost_tracker: CostTracker,
    /// Doom loop tracker — detects when the model repeats the same tool call.
    doom_loop_tracker: DoomLoopTracker,
    /// User directives extracted from conversation ("always X", "never Y", etc.).
    /// These survive compaction by being re-injected after the system message.
    pinned_directives: Vec<String>,
    /// Pre-computed workspace path string for hook inputs (avoids repeated allocation).
    workspace_path_str: String,
    /// Whether `cleanup_old_tool_outputs` has run this session (at most once).
    tool_output_cleanup_done: bool,
    /// When true, complex-task execution has progressed but the checklist has not
    /// been refreshed via `todo_read`/`todo_write` yet.
    pending_todo_sync: bool,
    /// Current execution phase (None for Simple/Medium tasks).
    phase: Option<codingbuddy_core::TaskPhase>,
    /// Count of read-only tool calls since entering current phase.
    phase_read_only_calls: usize,
    /// Count of edit/write tool calls since entering Execute phase.
    phase_edit_calls: usize,
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
        if config.complexity == crate::complexity::PromptComplexity::Complex {
            messages.push(ChatMessage::System {
                content: COMPLEX_TODO_POLICY.to_string(),
            });
        }

        let workspace_path_str = config
            .workspace
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();

        // Phase loop: only for Complex tasks
        let initial_phase = config.initial_phase.or_else(|| {
            if config.complexity == crate::complexity::PromptComplexity::Complex {
                Some(codingbuddy_core::TaskPhase::Explore)
            } else {
                None
            }
        });

        Self {
            llm,
            tool_host,
            config,
            messages,
            tools,
            discoverable_tools: Vec::new(),
            stream_cb: None,
            approval_cb: None,
            user_question_cb: None,
            hooks: None,
            event_cb: None,
            checkpoint_cb: None,
            escalation: crate::complexity::EscalationSignals::default(),
            recent_errors: VecDeque::new(),
            recovery_injected: false,
            output_scanner: codingbuddy_policy::output_scanner::OutputScanner::new(),
            tool_cache: HashMap::new(),
            circuit_breaker: HashMap::new(),
            cost_tracker: CostTracker::default(),
            doom_loop_tracker: DoomLoopTracker::default(),
            pinned_directives: Vec::new(),
            workspace_path_str,
            tool_output_cleanup_done: false,
            pending_todo_sync: false,
            phase: initial_phase,
            phase_read_only_calls: 0,
            phase_edit_calls: 0,
        }
    }

    /// Install tools that can be revealed on demand via `tool_search`.
    pub fn set_discoverable_tools(&mut self, tools: Vec<ToolDefinition>) {
        self.discoverable_tools = tools;
    }

    /// Build the raw cache input string and its SHA-256 hash key.
    /// Returns `(hash_key, raw_string)` from a single `format!` call.
    fn cache_key_with_raw(tool_name: &str, args: &serde_json::Value) -> (String, String) {
        use sha2::{Digest, Sha256};
        let raw = format!("{}:{}", tool_name, args);
        let hash = Sha256::digest(raw.as_bytes());
        let bytes: [u8; 8] = hash[..8].try_into().unwrap();
        (format!("{:016x}", u64::from_be_bytes(bytes)), raw)
    }

    fn workspace_store(&self) -> Result<Store> {
        let workspace = self
            .config
            .workspace
            .as_ref()
            .ok_or_else(|| anyhow!("workspace is not available in this context"))?;
        Store::new(workspace)
    }

    fn current_session(&self, store: &Store) -> Result<Session> {
        if let Some(session_id) = self.config.session_id {
            return store
                .load_session(session_id)?
                .ok_or_else(|| anyhow!("no active session record found for {session_id}"));
        }
        store
            .load_latest_session()?
            .ok_or_else(|| anyhow!("no active session record found"))
    }

    fn latest_user_prompt(&self) -> String {
        self.messages
            .iter()
            .rev()
            .find_map(|msg| match msg {
                ChatMessage::User { content } => Some(content.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    fn latest_assistant_text(&self) -> String {
        self.messages
            .iter()
            .rev()
            .find_map(|msg| match msg {
                ChatMessage::Assistant { content, .. } => content.clone(),
                _ => None,
            })
            .unwrap_or_default()
    }

    fn continuation_anchor_message(&self) -> Option<String> {
        if self.config.complexity != crate::complexity::PromptComplexity::Complex {
            return None;
        }

        let store = self.workspace_store().ok()?;
        let session = self.current_session(&store).ok()?;
        let mut lines = Vec::new();

        lines.push(format!("- session_id: {}", session.session_id));
        if let Some(phase) = self.phase {
            lines.push(format!("- workflow_phase: {}", phase.as_str()));
        }

        if let Some(plan_id) = session.active_plan_id
            && let Ok(Some(plan)) = store.load_plan(plan_id)
        {
            let done_steps = plan.steps.iter().filter(|step| step.done).count();
            let total_steps = plan.steps.len();
            lines.push(format!("- plan_progress: {done_steps}/{total_steps}"));
            if let Some(step) = plan.steps.iter().find(|step| !step.done) {
                lines.push(format!(
                    "- current_step: {}",
                    truncate_line(&step.title, 120)
                ));
            }
        }

        if let Ok(todos) = store.list_session_todos(session.session_id) {
            let in_progress = todos
                .iter()
                .filter(|todo| todo.status.eq_ignore_ascii_case("in_progress"))
                .count();
            let completed = todos
                .iter()
                .filter(|todo| todo.status.eq_ignore_ascii_case("completed"))
                .count();
            lines.push(format!(
                "- todos: total={} completed={} in_progress={}",
                todos.len(),
                completed,
                in_progress
            ));
            if let Some(current) = todos
                .iter()
                .find(|todo| todo.status.eq_ignore_ascii_case("in_progress"))
                .or_else(|| {
                    todos
                        .iter()
                        .find(|todo| todo.status.eq_ignore_ascii_case("pending"))
                })
            {
                lines.push(format!(
                    "- current_todo: {} [{}]",
                    truncate_line(&current.content, 120),
                    current.status
                ));
            }
        }

        if lines.len() <= 1 {
            return None;
        }

        Some(format!(
            "CONTINUATION_CONTEXT (stay aligned with current work state):\n{}",
            lines.join("\n")
        ))
    }

    fn should_track_checklist_discipline(&self) -> bool {
        self.config.complexity == crate::complexity::PromptComplexity::Complex
    }

    fn maybe_inject_complex_checklist_nudge(&mut self, batch_calls: &[LlmToolCall]) {
        if !self.should_track_checklist_discipline() || batch_calls.is_empty() {
            return;
        }

        let touched_todo = batch_calls
            .iter()
            .any(|call| matches!(call.name.as_str(), "todo_read" | "todo_write"));
        if touched_todo {
            self.pending_todo_sync = false;
            return;
        }

        let progressed_work = batch_calls.iter().any(|call| {
            tool_bridge::is_write_tool(&call.name)
                || matches!(
                    call.name.as_str(),
                    "spawn_task" | "task_create" | "task_update" | "task_stop"
                )
        });
        if !progressed_work || self.pending_todo_sync {
            return;
        }

        let mut reminder = "Checklist discipline (complex workflow): refresh session todos now with todo_read/todo_write before continuing. Keep exactly one in_progress item.".to_string();
        if let Ok(store) = self.workspace_store()
            && let Ok(session) = self.current_session(&store)
            && let Ok(todos) = store.list_session_todos(session.session_id)
        {
            let in_progress = todos
                .iter()
                .filter(|todo| todo.status.eq_ignore_ascii_case("in_progress"))
                .count();
            if todos.is_empty() {
                reminder = "Checklist discipline (complex workflow): initialize session todos now via todo_write before more edits or delegation.".to_string();
            } else if in_progress != 1 {
                reminder = format!(
                    "Checklist discipline (complex workflow): expected exactly one in_progress todo, found {in_progress}. Normalize with todo_write before proceeding."
                );
            }
        }

        self.messages
            .push(ChatMessage::System { content: reminder });
        self.pending_todo_sync = true;
    }

    fn next_request_route(&self) -> (String, Option<codingbuddy_core::ThinkingConfig>, u32) {
        // Model routing: use reasoner for Complex+escalated tasks
        let use_reasoner = self.config.complexity == crate::complexity::PromptComplexity::Complex
            && self.escalation.should_escalate();

        if use_reasoner {
            (
                self.config.extended_thinking_model.clone(),
                None, // Reasoner thinks natively
                codingbuddy_core::CODINGBUDDY_REASONER_MAX_OUTPUT_TOKENS,
            )
        } else {
            let thinking = self.config.thinking.as_ref().map(|base| {
                if self.escalation.should_escalate() {
                    codingbuddy_core::ThinkingConfig::enabled(self.escalation.budget())
                } else {
                    base.clone()
                }
            });
            (self.config.model.clone(), thinking, self.config.max_tokens)
        }
    }

    fn compaction_budget_for_next_turn(&self, tool_def_tokens: u64) -> (u64, u64, u64) {
        let (route_model, route_thinking, route_max_tokens) = self.next_request_route();
        let thinking_enabled = route_thinking
            .as_ref()
            .is_some_and(|cfg| cfg.thinking_type.eq_ignore_ascii_case("enabled"));
        let model_max_output =
            max_output_tokens_for_model(self.config.provider_kind, &route_model, thinking_enabled);
        let reserved_output_tokens = u64::from(route_max_tokens.min(model_max_output))
            .max(self.config.response_budget_tokens);
        let reserved_overhead_tokens = self
            .config
            .reserved_overhead_tokens
            .max(tool_def_tokens.saturating_add(COMPACTION_EXTRA_OVERHEAD_TOKENS));
        let usable_input_budget = self
            .config
            .context_window_tokens
            .saturating_sub(reserved_overhead_tokens)
            .saturating_sub(reserved_output_tokens)
            .max(MIN_COMPACTION_INPUT_BUDGET_TOKENS);
        let prune_threshold = ((usable_input_budget as f64) * PRUNE_THRESHOLD_PCT) as u64;
        let compact_threshold = ((usable_input_budget as f64) * COMPACTION_THRESHOLD_PCT) as u64;
        let compact_target = ((usable_input_budget as f64) * COMPACTION_TARGET_PCT) as u64;
        (
            prune_threshold.max(1),
            compact_threshold.max(1),
            compact_target.max(1),
        )
    }

    fn is_active_task_status(status: &str) -> bool {
        !matches!(
            status.trim().to_ascii_lowercase().as_str(),
            "completed" | "failed" | "cancelled"
        )
    }

    fn compaction_work_state_message(&self) -> Option<String> {
        let store = self.workspace_store().ok()?;
        let session = self.current_session(&store).ok()?;
        let mut lines = Vec::new();

        if let Some(phase) = self.phase {
            lines.push(format!("- phase: {}", phase.as_str()));
        }

        let session_state = serde_json::to_string(&session.status)
            .ok()
            .map(|value| value.trim_matches('"').to_string())
            .unwrap_or_else(|| format!("{:?}", session.status));
        lines.push(format!("- session_state: {session_state}"));

        if let Some(plan_id) = session.active_plan_id
            && let Ok(Some(plan)) = store.load_plan(plan_id)
        {
            let done_steps = plan.steps.iter().filter(|step| step.done).count();
            let total_steps = plan.steps.len();
            lines.push(format!(
                "- active_plan_goal: {}",
                truncate_line(&plan.goal, 160)
            ));
            lines.push(format!(
                "- active_plan_progress: {done_steps}/{total_steps} step(s) done"
            ));
            if let Some(next_step) = plan.steps.iter().find(|step| !step.done) {
                lines.push(format!(
                    "- active_plan_next_step: {}",
                    truncate_line(&next_step.title, 120)
                ));
            }
        }

        if let Ok(tasks) = store.list_tasks(Some(session.session_id)) {
            let active_tasks = tasks
                .into_iter()
                .filter(|task| Self::is_active_task_status(&task.status))
                .take(5)
                .collect::<Vec<_>>();
            if !active_tasks.is_empty() {
                lines.push("- active_tasks:".to_string());
                for task in active_tasks {
                    let mut task_line =
                        format!("  - [{}] {}", task.status, truncate_line(&task.title, 80));
                    if let Some(artifact_path) = task
                        .artifact_path
                        .as_ref()
                        .filter(|value| !value.trim().is_empty())
                    {
                        task_line
                            .push_str(&format!(" artifact={}", truncate_line(artifact_path, 80)));
                    }
                    if let Ok(Some(run)) = store.load_subagent_run_for_task(task.task_id)
                        && let Some(child_session_id) = run.child_session_id
                    {
                        task_line.push_str(&format!(" child_session={child_session_id}"));
                    }
                    lines.push(task_line);
                }
            }
        }

        let has_plan = lines.iter().any(|line| line.starts_with("- active_plan_"));
        let has_tasks = lines.iter().any(|line| line.starts_with("- active_tasks:"));
        if !has_plan && !has_tasks && self.phase.is_none() {
            return None;
        }

        Some(format!(
            "ACTIVE_WORK_STATE (preserve after compaction):\n{}",
            lines.join("\n")
        ))
    }

    fn is_compaction_control_prompt(content: &str) -> bool {
        let trimmed = content.trim();
        let lower = trimmed.to_ascii_lowercase();
        if lower.starts_with("conversation_history (compacted") {
            return true;
        }
        if lower.starts_with("context was compacted.") {
            return true;
        }
        if lower.starts_with("continue with your next steps if any remain") {
            return true;
        }
        if lower.starts_with("your response is too verbose") {
            return true;
        }
        trimmed == HALLUCINATION_NUDGE
    }

    fn replayable_user_prompt(messages: &[ChatMessage]) -> Option<String> {
        messages.iter().rev().find_map(|msg| match msg {
            ChatMessage::User { content } => {
                let trimmed = content.trim();
                if trimmed.is_empty() || Self::is_compaction_control_prompt(trimmed) {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            }
            _ => None,
        })
    }

    fn append_compaction_followup(
        &self,
        new_messages: &mut Vec<ChatMessage>,
        compacted_msgs: &[ChatMessage],
        compacted_count: usize,
    ) {
        if compacted_count < 5 {
            return;
        }
        if new_messages
            .last()
            .is_some_and(|msg| matches!(msg, ChatMessage::User { .. }))
        {
            return;
        }

        let content = if let Some(replay) = Self::replayable_user_prompt(compacted_msgs) {
            format!(
                "Context was compacted. Resume the current task using this prior user request as the anchor:\n\n{}\n\nContinue with the next concrete step. If blocked, ask one focused clarification question.",
                truncate_line(&replay, 700)
            )
        } else {
            "Context was compacted. Continue with your next steps if any remain, or summarize what was accomplished.".to_string()
        };
        new_messages.push(ChatMessage::User { content });
    }

    fn persist_phase_state_best_effort(&self, new_phase: codingbuddy_core::TaskPhase) {
        let Some(session_id) = self.config.session_id else {
            return;
        };
        if self.config.workspace.is_none() {
            return;
        }

        let Ok(store) = self.workspace_store() else {
            return;
        };
        let Ok(Some(mut session)) = store.load_session(session_id) else {
            return;
        };

        let target_status = match new_phase {
            codingbuddy_core::TaskPhase::Explore => return,
            codingbuddy_core::TaskPhase::Plan => SessionState::Planning,
            codingbuddy_core::TaskPhase::Execute => SessionState::ExecutingStep,
            codingbuddy_core::TaskPhase::Verify => SessionState::Verifying,
        };

        if new_phase == codingbuddy_core::TaskPhase::Plan && session.active_plan_id.is_none() {
            let plan_id = Uuid::now_v7();
            let plan = Plan {
                plan_id,
                version: 1,
                goal: self.latest_user_prompt(),
                assumptions: Vec::new(),
                steps: Vec::new(),
                verification: Vec::new(),
                risk_notes: Vec::new(),
            };
            if store.save_plan(session.session_id, &plan).is_ok() {
                session.active_plan_id = Some(plan_id);
                self.emit_event_if_present(EventKind::PlanCreated { plan });
            }
        }

        if session.status == target_status {
            if session.active_plan_id.is_some() {
                let _ = store.save_session(&session);
            }
            return;
        }

        let previous = session.status.clone();
        session.status = target_status.clone();
        if store.save_session(&session).is_ok() {
            self.emit_event_if_present(EventKind::SessionStateChanged {
                from: previous,
                to: target_status,
            });
        }
    }

    fn emit_event_if_present(&self, kind: EventKind) {
        event_emission::emit_event_if_present(self, kind);
    }

    fn build_plan_from_text(&self, plan_id: Uuid, version: u32, goal: String, text: &str) -> Plan {
        let mut assumptions = Vec::new();
        let mut verification = Vec::new();
        let mut risk_notes = Vec::new();
        let mut steps = Vec::new();

        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let lower = trimmed.to_ascii_lowercase();
            if lower.contains("assum") {
                assumptions.push(trimmed.to_string());
                continue;
            }
            if lower.contains("verify") || lower.contains("test") || lower.contains("check") {
                verification.push(trimmed.to_string());
            }
            if lower.contains("risk") || lower.contains("watch") || lower.contains("careful") {
                risk_notes.push(trimmed.to_string());
            }

            let numbered = trimmed
                .strip_prefix("1. ")
                .or_else(|| trimmed.strip_prefix("2. "))
                .or_else(|| trimmed.strip_prefix("3. "))
                .or_else(|| trimmed.strip_prefix("4. "))
                .or_else(|| trimmed.strip_prefix("5. "));
            let bulleted = trimmed
                .strip_prefix("- ")
                .or_else(|| trimmed.strip_prefix("* "));
            if let Some(step_text) = numbered.or(bulleted) {
                let title = step_text.trim().to_string();
                if !title.is_empty() {
                    steps.push(PlanStep {
                        step_id: Uuid::now_v7(),
                        title: title.clone(),
                        intent: title,
                        tools: Vec::new(),
                        files: Vec::new(),
                        done: false,
                    });
                }
            }
        }

        if steps.is_empty() {
            let fallback = text
                .lines()
                .find_map(|line| {
                    let trimmed = line.trim();
                    (!trimmed.is_empty()).then(|| trimmed.to_string())
                })
                .unwrap_or_else(|| "Finalize the approved implementation plan".to_string());
            steps.push(PlanStep {
                step_id: Uuid::now_v7(),
                title: fallback.clone(),
                intent: fallback,
                tools: Vec::new(),
                files: Vec::new(),
                done: false,
            });
        }

        Plan {
            plan_id,
            version,
            goal,
            assumptions,
            steps,
            verification,
            risk_notes,
        }
    }

    /// Build a bounded cache key by hashing the tool name + args with SHA-256.
    fn cache_key(tool_name: &str, args: &serde_json::Value) -> String {
        Self::cache_key_with_raw(tool_name, args).0
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

        let key = Self::cache_key(tool_name, args);
        if let Some(entry) = self.tool_cache.get(&key)
            && entry.timestamp.elapsed().as_secs() < TOOL_CACHE_TTL_SECS
        {
            return Some(entry.result.clone());
        }
        // Expired or not found — remove it
        self.tool_cache.remove(&key);
        None
    }

    /// Store a tool result in the cache.
    /// Enforces `MAX_CACHE_ENTRIES` by evicting the oldest entry when full.
    fn cache_store(
        &mut self,
        tool_name: &str,
        args: &serde_json::Value,
        result: &serde_json::Value,
    ) {
        if !CACHEABLE_TOOLS.contains(&tool_name) {
            return;
        }
        let (key, raw) = Self::cache_key_with_raw(tool_name, args);
        self.tool_cache.insert(
            key,
            CacheEntry {
                result: result.clone(),
                timestamp: Instant::now(),
                raw_key: raw,
            },
        );

        // Evict oldest entry if cache exceeds maximum size (single insert → at most 1 eviction)
        if self.tool_cache.len() > MAX_CACHE_ENTRIES
            && let Some(oldest_key) = self
                .tool_cache
                .iter()
                .min_by_key(|(_, entry)| entry.timestamp)
                .map(|(k, _)| k.clone())
        {
            self.tool_cache.remove(&oldest_key);
        }
    }

    /// Invalidate cache entries after a write tool modifies a path.
    ///
    /// Glob, list, and index queries can match any path, so we evict ALL of those
    /// on any write. Only `fs_read` and `fs_grep` entries use path-containment checks.
    fn cache_invalidate_path(&mut self, path: &str) {
        let quoted = format!("\"{path}\"");
        let parent_quoted = path
            .rsplit_once('/')
            .map(|(parent, _)| format!("\"{parent}\""));

        self.tool_cache.retain(|_key, entry| {
            // Always evict glob, list, and index queries — they could match any path.
            // raw_key format is "tool_name:{args}", so starts_with avoids false matches
            // from arguments that happen to contain these tool names.
            if entry.raw_key.starts_with("fs_glob:")
                || entry.raw_key.starts_with("fs_list:")
                || entry.raw_key.starts_with("index_query:")
            {
                return false;
            }
            // For path-specific tools (fs_read, fs_grep, etc.), check containment.
            if entry.raw_key.contains(&quoted) {
                return false;
            }
            if let Some(ref pq) = parent_quoted
                && entry.raw_key.contains(pq.as_str())
            {
                return false;
            }
            true
        });
    }

    /// Clean up old tool output files (>7 days). Runs at most once per session.
    fn cleanup_old_tool_outputs(&mut self) {
        if self.tool_output_cleanup_done {
            return;
        }
        self.tool_output_cleanup_done = true;
        let output_dir = std::path::Path::new(&self.workspace_path_str)
            .join(".codingbuddy")
            .join("tool_outputs");
        let Ok(entries) = std::fs::read_dir(&output_dir) else {
            return;
        };
        let cutoff = std::time::SystemTime::now() - std::time::Duration::from_secs(7 * 24 * 3600);
        for entry in entries.flatten() {
            if let Ok(metadata) = entry.metadata() {
                let modified = metadata
                    .modified()
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                if modified < cutoff {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }

    /// Persist a large tool output to disk and return a hint for the truncated message.
    /// Writes to `.codingbuddy/tool_outputs/<hash>.txt` so the model can re-read it.
    const LARGE_OUTPUT_THRESHOLD: usize = 50_000; // 50KB
    fn persist_large_output(&mut self, tool_name: &str, raw_output: &str) -> Option<String> {
        if raw_output.len() < Self::LARGE_OUTPUT_THRESHOLD {
            return None;
        }
        // Opportunistic cleanup of old files
        self.cleanup_old_tool_outputs();
        let output_dir = std::path::Path::new(&self.workspace_path_str)
            .join(".codingbuddy")
            .join("tool_outputs");
        if std::fs::create_dir_all(&output_dir).is_err() {
            return None;
        }
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        raw_output.hash(&mut hasher);
        let hash = hasher.finish();
        let filename = format!("{tool_name}_{hash:016x}.txt");
        let path = output_dir.join(&filename);
        if std::fs::write(&path, raw_output).is_ok() {
            let line_count = raw_output.lines().count();
            Some(format!(
                "\n[Full output ({line_count} lines, {} bytes) saved to \
                 .codingbuddy/tool_outputs/{filename}. \
                 Use fs_read with start_line/end_line to view specific sections, \
                 or fs_grep to search within it.]",
                raw_output.len()
            ))
        } else {
            None
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

    /// Fire a hook event and log any failures.
    fn fire_hook_logged(hooks: &HookRuntime, event: HookEvent, input: &HookInput) {
        let result = hooks.fire(event, input);
        for run in &result.runs {
            if !run.success {
                eprintln!(
                    "[hooks] {:?} hook failed: {}",
                    event, run.handler_description
                );
            }
        }
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
        self.add_user_turn(user_message, false)
    }

    /// Continue the conversation with additional user input (multi-turn).
    pub fn continue_with(&mut self, user_message: &str) -> Result<ToolLoopResult> {
        self.add_user_turn(user_message, true)
    }

    /// Shared logic for adding a user turn and executing the loop.
    fn add_user_turn(
        &mut self,
        user_message: &str,
        is_continuation: bool,
    ) -> Result<ToolLoopResult> {
        if is_continuation {
            strip_prior_reasoning_content(&mut self.messages);
            if let Some(anchor) = self.continuation_anchor_message() {
                self.messages.push(ChatMessage::System { content: anchor });
            }
        }

        // Reset per-question state so previous errors don't leak across turns
        self.recovery_injected = false;
        self.pending_todo_sync = false;

        self.messages.push(ChatMessage::User {
            content: user_message.to_string(),
        });

        self.collect_directives(user_message);
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

        let mut verbosity_nudge_count: usize = 0;

        loop {
            // Check turn limit — on final turn, ask for a text-only summary.
            if turns >= self.config.max_turns {
                // Inject a final system message asking for text-only summary
                self.messages.push(ChatMessage::System {
                    content: "MAXIMUM STEPS REACHED. Tools are now disabled. Respond with text only: \
                        summarize what you accomplished, what remains incomplete, and any recommendations."
                        .to_string(),
                });
                // Make one final LLM call with no tools to get a summary
                let summary_request = ChatRequest {
                    messages: self.messages.clone(),
                    model: self.config.model.clone(),
                    tools: Vec::new(),
                    tool_choice: ToolChoice::none(),
                    max_tokens: 2048,
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
                let summary_text = match self.llm.complete_chat(&summary_request) {
                    Ok(resp) => {
                        if let Some(u) = &resp.usage {
                            total_usage.prompt_tokens += u.prompt_tokens;
                            total_usage.completion_tokens += u.completion_tokens;
                        }
                        resp.text
                    }
                    Err(e) => format!("(Summary generation failed: {e})"),
                };
                if !summary_text.is_empty() {
                    self.emit(StreamChunk::ContentDelta(summary_text.clone()));
                }
                self.emit(StreamChunk::Done {
                    reason: Some("max turns reached".to_string()),
                });
                return Ok(ToolLoopResult {
                    response: summary_text,
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
            let tool_def_tokens: u64 = self
                .tools
                .iter()
                .map(|t| (serde_json::to_string(t).unwrap_or_default().len() as u64) / 4)
                .sum();
            let message_tokens = estimate_message_tokens(&self.messages);
            let estimated_tokens = message_tokens + tool_def_tokens;
            let (prune_threshold, compact_threshold, compact_target) =
                self.compaction_budget_for_next_turn(tool_def_tokens);

            if estimated_tokens > prune_threshold {
                // Phase 1: Lightweight pruning — trim old tool outputs
                self.prune_old_tool_outputs();

                // Re-check after pruning
                let post_prune_tokens = estimate_message_tokens(&self.messages) + tool_def_tokens;

                if post_prune_tokens > compact_threshold {
                    // Phase 2: Full compaction with structured summary
                    let pre_msg_count = self.messages.len() as u64;
                    let compacted = self.compact_messages(compact_target);
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
            // Images only need to be sent once — clear after first turn to save tokens.
            if turns == 1 && !self.config.images.is_empty() {
                self.config.images.clear();
            }

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
            }

            // Evict expired cache entries to prevent unbounded memory growth
            self.tool_cache
                .retain(|_, entry| entry.timestamp.elapsed().as_secs() < TOOL_CACHE_TTL_SECS);

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
                let (has_unverified_refs, has_shell_cmd) = if nudge_count < MAX_NUDGE_ATTEMPTS {
                    (
                        has_unverified_file_references(&text, &tool_calls_made),
                        contains_shell_command_pattern(&text),
                    )
                } else {
                    (false, false)
                };
                let should_nudge = nudge_count < MAX_NUDGE_ATTEMPTS
                    && (text.len() > HALLUCINATION_NUDGE_THRESHOLD
                        || has_unverified_refs
                        || has_shell_cmd);

                if should_nudge {
                    // Don't emit this attempt — inject a nudge and retry
                    nudge_count += 1;
                    let trigger = if text.len() > HALLUCINATION_NUDGE_THRESHOLD {
                        "long_response"
                    } else if has_unverified_refs {
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

                // Numeric consistency check: if the response contains numeric
                // claims that contradict tool results, log a warning event.
                if let Some(correction) = check_response_consistency(&text, &tool_calls_made) {
                    self.emit(StreamChunk::SecurityWarning {
                        message: correction,
                    });
                }

                // Verbosity enforcement — strip code blocks before counting prose words.
                // Nudge up to 2 times per turn, then let it through.
                // Fast path: skip the strip_code_blocks allocation if total word count
                // (including code) is under the threshold — most responses are short.
                if verbosity_nudge_count < 2 && text.split_whitespace().count() > 400 {
                    let prose = helpers::strip_code_blocks(&text);
                    let word_count = prose.split_whitespace().count();
                    if word_count > 400 {
                        verbosity_nudge_count += 1;
                        self.messages.push(ChatMessage::Assistant {
                            content: Some(text),
                            reasoning_content: None,
                            tool_calls: vec![],
                        });
                        self.messages.push(ChatMessage::User {
                            content: format!(
                                "Your response is too verbose ({word_count} words of prose, excluding code). \
                                 Keep responses under 200 words unless showing code. Rewrite concisely."
                            ),
                        });
                        continue;
                    }
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

                if let Some(current_phase) = self.phase {
                    let should_allow_text_transition =
                        current_phase != codingbuddy_core::TaskPhase::Plan;
                    let text_has_plan_keywords =
                        should_allow_text_transition && phases::text_has_plan_keywords(&text);
                    if should_allow_text_transition
                        && let Some(new_phase) = phases::check_phase_transition(
                            current_phase,
                            self.phase_read_only_calls,
                            true,
                            text_has_plan_keywords,
                            false,
                            self.phase_edit_calls,
                        )
                    {
                        self.apply_phase_transition(new_phase);
                        strip_prior_reasoning_content(&mut self.messages);
                        continue;
                    }
                }

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
                // Pre-validate, check circuit breaker, and look up cache BEFORE entering
                // thread scope (these require &mut self which is not Sync).
                let mut pre_validated: Vec<(
                    LlmToolCall,               // repaired call
                    Option<serde_json::Value>, // cached result (Some = skip execution)
                    Option<String>,            // validation/breaker error (Some = skip execution)
                )> = Vec::with_capacity(parallel_calls.len());

                for llm_call in &parallel_calls {
                    let repaired = tool_bridge::repair_tool_name(&llm_call.name, &self.tools);
                    let effective_name = repaired.unwrap_or_else(|| llm_call.name.clone());
                    let effective_call = if effective_name != llm_call.name {
                        LlmToolCall {
                            id: llm_call.id.clone(),
                            name: effective_name,
                            arguments: llm_call.arguments.clone(),
                        }
                    } else {
                        (*llm_call).clone()
                    };
                    let parsed_args: serde_json::Value =
                        serde_json::from_str(&effective_call.arguments).unwrap_or_default();

                    // Schema validation
                    if let Err(e) = codingbuddy_tools::validate_tool_args_schema(
                        &effective_call.name,
                        &parsed_args,
                        &self.tools,
                    ) {
                        pre_validated.push((
                            effective_call,
                            None,
                            Some(format!("Validation error: {e}")),
                        ));
                        continue;
                    }

                    // Circuit breaker check
                    if let Some(state) = self.circuit_breaker.get(&effective_call.name)
                        && state.consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD
                        && state.cooldown_remaining > 0
                    {
                        let err = format!(
                            "Tool '{}' is temporarily disabled due to repeated failures. Try a different approach.",
                            effective_call.name
                        );
                        pre_validated.push((effective_call, None, Some(err)));
                        continue;
                    }

                    // Cache lookup
                    let cached = self.cache_lookup(&effective_call.name, &parsed_args);
                    pre_validated.push((effective_call, cached, None));
                }

                // Execute in parallel — only calls that passed pre-validation and aren't cached.
                // Threads return (index, result, elapsed_ms) to avoid cloning effective_call.
                let tool_host = &self.tool_host;
                let parallel_results: Vec<_> = std::thread::scope(|s| {
                    // Collect (idx, handle) pairs so idx is available even if the thread panics
                    let indexed_handles: Vec<_> = pre_validated
                        .iter()
                        .enumerate()
                        .map(|(idx, (effective_call, cached, error))| {
                            let handle = s.spawn(move || {
                                // Return pre-computed error
                                if let Some(err_msg) = error {
                                    let error_result = codingbuddy_core::ToolResult {
                                        invocation_id: uuid::Uuid::new_v4(),
                                        success: false,
                                        output: serde_json::json!(err_msg),
                                    };
                                    return (idx, error_result, 0u64);
                                }

                                // Return cached result
                                if let Some(cached_val) = cached {
                                    let result = codingbuddy_core::ToolResult {
                                        invocation_id: uuid::Uuid::new_v4(),
                                        success: true,
                                        output: cached_val.clone(),
                                    };
                                    return (idx, result, 0u64);
                                }

                                // Execute the tool
                                let start = Instant::now();
                                let internal =
                                    tool_bridge::llm_tool_call_to_internal(effective_call);
                                let proposal = tool_host.propose(internal);
                                let approved = ApprovedToolCall {
                                    invocation_id: proposal.invocation_id,
                                    call: proposal.call,
                                };
                                let result = tool_host.execute(approved);
                                let elapsed = start.elapsed().as_millis() as u64;
                                (idx, result, elapsed)
                            });
                            (idx, handle)
                        })
                        .collect();
                    indexed_handles
                        .into_iter()
                        .map(|(original_idx, h)| {
                            h.join().unwrap_or_else(|panic_payload| {
                                let panic_msg = panic_payload
                                    .downcast_ref::<String>()
                                    .map(|s| s.as_str())
                                    .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                                    .unwrap_or("unknown panic");
                                let error_result = codingbuddy_core::ToolResult {
                                    invocation_id: uuid::Uuid::new_v4(),
                                    success: false,
                                    output: serde_json::json!(format!(
                                        "Internal error: parallel tool execution panicked: {panic_msg}"
                                    )),
                                };
                                // Use the original index so the caller can look up the tool_call_id
                                (original_idx, error_result, 0u64)
                            })
                        })
                        .collect()
                });

                // Process results sequentially (updates messages, cache, events).
                // Look up effective_call from pre_validated by index.
                for (idx, result, par_duration) in &parallel_results {
                    let effective_call = if *idx < pre_validated.len() {
                        &pre_validated[*idx].0
                    } else {
                        // Should not happen now that we preserve original_idx,
                        // but keep as safety fallback
                        self.messages.push(ChatMessage::Tool {
                            tool_call_id: String::new(),
                            content: result.output.to_string(),
                        });
                        continue;
                    };
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
                        duration_ms: *par_duration,
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
                        duration_ms: *par_duration,
                        args_json: args_json_for_record(
                            &effective_call.name,
                            &effective_call.arguments,
                        ),
                        result_preview: None,
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

            self.maybe_inject_complex_checklist_nudge(&response.tool_calls);

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
                        tool_name: doom_loop_tool.clone(),
                        repeat_count: DOOM_LOOP_THRESHOLD as u64,
                    });
                }
                self.emit(StreamChunk::SecurityWarning {
                    message: format!(
                        "Doom loop detected: model repeated identical tool calls {}+ times. \
                         The tool '{doom_loop_tool}' has been called with the same arguments repeatedly.",
                        DOOM_LOOP_THRESHOLD
                    ),
                });

                // Try to get user decision via approval callback
                let user_wants_continue = if let Some(ref cb) = self.approval_cb {
                    // Present as a "doom loop" approval — user sees the warning and decides
                    let dummy_call = ToolCall {
                        name: format!("__doom_loop_continue__{doom_loop_tool}"),
                        args: serde_json::json!({
                            "reason": "doom_loop_detected",
                            "repeated_tool": doom_loop_tool,
                            "message": "The model is repeating the same action. Continue, abort, or redirect?"
                        }),
                        requires_approval: true,
                    };
                    cb(&dummy_call).unwrap_or(false)
                } else {
                    false
                };

                if user_wants_continue {
                    // User chose to continue — reset doom loop tracker and proceed
                    self.doom_loop_tracker.reset();
                    self.messages.push(ChatMessage::System {
                        content: DOOM_LOOP_GUIDANCE.to_string(),
                    });
                } else {
                    // User chose to abort (or no approval callback) — stop the loop
                    self.messages.push(ChatMessage::System {
                        content: DOOM_LOOP_GUIDANCE.to_string(),
                    });
                    self.doom_loop_tracker.mark_warned();
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
            }

            // Phase transition logic for Complex tasks
            if let Some(current_phase) = self.phase {
                // Count read-only vs write tool calls in this batch
                let prev_edit_calls = self.phase_edit_calls;
                for tc in &response.tool_calls {
                    if helpers::is_read_only_api_name(&tc.name) {
                        self.phase_read_only_calls += 1;
                    } else {
                        self.phase_edit_calls += 1;
                    }
                }
                let used_write = self.phase_edit_calls > prev_edit_calls;
                if let Some(new_phase) = phases::check_phase_transition(
                    current_phase,
                    self.phase_read_only_calls,
                    false, // no text response (we had tool calls)
                    false,
                    used_write,
                    self.phase_edit_calls,
                ) {
                    self.apply_phase_transition(new_phase);
                }
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
            && state.consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD
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
                args_json: None,
                result_preview: None,
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
                args_json: args_json_for_record(&llm_call.name, &llm_call.arguments),
                result_preview: None,
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
                workspace: self.workspace_str().to_string(),
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
                    args_json: args_json_for_record(&llm_call.name, &llm_call.arguments),
                    result_preview: None,
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
                    args_json: args_json_for_record(&llm_call.name, &llm_call.arguments),
                    result_preview: None,
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
                args_json: args_json_for_record(&llm_call.name, &llm_call.arguments),
                result_preview: None,
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
                workspace: self.workspace_str().to_string(),
            };
            Self::fire_hook_logged(hooks, HookEvent::PostToolUse, &input);
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

        // Persist large outputs to disk so the model can re-read the full content.
        let file_hint = result
            .output
            .as_str()
            .and_then(|s| self.persist_large_output(&llm_call.name, s));

        // Convert result to ChatMessage::Tool and append, scanning for security issues
        let (msg, injection_warnings) = tool_bridge::tool_result_to_message(
            &llm_call.id,
            &llm_call.name,
            &result,
            Some(&self.output_scanner),
        );
        // Append file persistence hint to the truncated message content.
        let msg = if let Some(hint) = file_hint {
            match msg {
                ChatMessage::Tool {
                    tool_call_id,
                    content,
                } => ChatMessage::Tool {
                    tool_call_id,
                    content: format!("{content}{hint}"),
                },
                other => other,
            }
        } else {
            msg
        };
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

        // Build result preview for consistency checking (first ~200 chars)
        let result_preview = result.output.as_str().map(|s| {
            let end = s.floor_char_boundary(200.min(s.len()));
            s[..end].to_string()
        });

        records.push(ToolCallRecord {
            tool_name: llm_call.name.clone(),
            tool_call_id: llm_call.id.clone(),
            args_summary,
            success,
            duration_ms: duration,
            args_json: args_json_for_record(&llm_call.name, &llm_call.arguments),
            result_preview,
        });

        Ok(records)
    }

    /// Handle agent-level tools (user_question, spawn_task, extended_thinking, etc.)
    fn handle_agent_level_tool(&mut self, llm_call: &LlmToolCall) -> Result<Vec<ToolCallRecord>> {
        agent_tools::handle_agent_level_tool(self, llm_call)
    }

    fn apply_phase_transition(&mut self, new_phase: codingbuddy_core::TaskPhase) {
        if self.phase == Some(new_phase) {
            self.phase_read_only_calls = 0;
            self.phase_edit_calls = 0;
            return;
        }

        self.persist_phase_state_best_effort(new_phase);

        if let Some(current_phase) = self.phase {
            self.emit(StreamChunk::PhaseTransition {
                from: current_phase.as_str().to_string(),
                to: new_phase.as_str().to_string(),
            });
        }

        match new_phase {
            codingbuddy_core::TaskPhase::Plan => {
                self.messages.push(ChatMessage::System {
                    content: phases::PLAN_TRANSITION_MESSAGE.to_string(),
                });
            }
            codingbuddy_core::TaskPhase::Verify => {
                self.messages.push(ChatMessage::System {
                    content: phases::VERIFY_TRANSITION_MESSAGE.to_string(),
                });
            }
            _ => {}
        }

        self.phase = Some(new_phase);
        self.phase_read_only_calls = 0;
        self.phase_edit_calls = 0;
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

        // Phase-based tool filtering for Complex tasks
        let tools = if let Some(phase) = self.phase {
            tools
                .into_iter()
                .filter(|t| phases::is_tool_allowed_in_phase(&t.function.name, phase))
                .collect()
        } else {
            tools
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
        let llm_turns_this_question = self.messages[last_user_idx..]
            .iter()
            .filter(|m| matches!(m, ChatMessage::Assistant { .. }))
            .count();
        // Require tool use for the first LLM round per user question (Code mode only)
        let tool_choice = if llm_turns_this_question < 1 && !self.config.read_only {
            ToolChoice::required()
        } else {
            ToolChoice::auto()
        };

        let (model, thinking, max_tokens) = self.next_request_route();

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
            self.recent_errors.push_back(normalized);
            if self.recent_errors.len() > MAX_RECENT_ERRORS {
                self.recent_errors.pop_front();
            }
        }
    }

    /// Count how many times the most recent error appears in the error history.
    ///
    /// Returns 0 if no errors recorded. Used for stuck detection — when the same
    /// error appears 3+ times, we inject stronger recovery guidance.
    fn repeated_error_count(&self) -> usize {
        if let Some(last) = self.recent_errors.back() {
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
        let target = target_tokens;

        while keep_from > 1 {
            // Find the start of this group (previous User message)
            let group_start = self.messages[1..keep_from]
                .iter()
                .rposition(|m| matches!(m, ChatMessage::User { .. }))
                .map(|i| i + 1) // +1 for the offset from [1..]
                .unwrap_or(1);

            let group_tokens: u64 = self.messages[group_start..keep_from]
                .iter()
                .map(|m| estimate_message_tokens(std::slice::from_ref(m)))
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
        // Try LLM-based compaction first, fall back to code-based extraction on any error.
        let summary = match build_compaction_summary_with_llm(self.llm, compacted_msgs) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "[compact] LLM compaction failed: {e:#}; falling back to code-based extraction"
                );
                build_compaction_summary(compacted_msgs)
            }
        };

        // P2.7: Validate summary before applying compaction.
        // Reject empty or trivially short summaries that would lose context.
        if summary.trim().is_empty() || summary.trim().len() < 50 {
            eprintln!(
                "[compact] rejecting compaction: summary too short ({} chars)",
                summary.trim().len()
            );
            return false;
        }

        // Fire PreCompact hook if configured
        if let Some(ref hooks) = self.hooks {
            let input = HookInput {
                event: "pre_compact".to_string(),
                tool_name: None,
                tool_input: None,
                tool_result: Some(serde_json::Value::String(summary.clone())),
                prompt: None,
                session_type: None,
                workspace: self.workspace_str().to_string(),
            };
            Self::fire_hook_logged(hooks, HookEvent::PreCompact, &input);
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

        // Build the new message list in a local vec, validate, then swap.
        // This avoids cloning the entire (potentially large) messages vec for rollback.
        let summary_msg = ChatMessage::User {
            content: format!(
                "CONVERSATION_HISTORY (compacted from {middle_count} messages):\n{summary}"
            ),
        };

        let mut new_messages = vec![system, summary_msg];

        if let Some(work_state_message) = self.compaction_work_state_message() {
            new_messages.push(ChatMessage::System {
                content: work_state_message,
            });
        }

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
            new_messages.push(ChatMessage::System {
                content: format!(
                    "USER DIRECTIVES (must follow throughout this conversation):\n{directives_text}"
                ),
            });
        }

        new_messages.extend(self.messages[keep_from..].to_vec());
        self.append_compaction_followup(&mut new_messages, compacted_msgs, middle_count);

        // P2.7: Post-compaction validation — ensure the result is structurally sound.
        // Must have at least 2 messages and start with a System message.
        if new_messages.len() < 2 || !matches!(&new_messages[0], ChatMessage::System { .. }) {
            eprintln!(
                "[compact] post-compaction validation failed: {} messages, discarding",
                new_messages.len()
            );
            return false;
        }

        self.messages = new_messages;

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
        event_emission::emit_injection_warnings(self, warnings);
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

    /// Get the workspace path as a string for hook inputs (pre-computed in `new()`).
    fn workspace_str(&self) -> &str {
        &self.workspace_path_str
    }

    /// Emit a stream chunk to the callback.
    fn emit(&self, chunk: StreamChunk) {
        event_emission::emit(self, chunk);
    }
}

#[cfg(test)]
mod tests;
