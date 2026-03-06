use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageKind {
    User,
    Assistant,
    System,
    ToolCall,
    ToolResult,
    Error,
    Thinking,
}

#[derive(Debug, Clone)]
pub struct TranscriptEntry {
    pub kind: MessageKind,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VimMode {
    Insert,
    Normal,
    Visual,
    Command,
}

impl VimMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Insert => "INSERT",
            Self::Normal => "NORMAL",
            Self::Visual => "VISUAL",
            Self::Command => "COMMAND",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChatShell {
    pub transcript: Vec<TranscriptEntry>,
    pub plan_lines: Vec<String>,
    pub tool_lines: Vec<String>,
    pub mission_control_lines: Vec<String>,
    pub artifact_lines: Vec<String>,
    pub active_tool: Option<String>,
    pub spinner_tick: usize,
    /// Whether the model is currently in a thinking/reasoning phase.
    pub is_thinking: bool,
    /// Accumulated thinking text for the current reasoning block.
    pub thinking_buffer: String,
    /// Current agent execution mode label.
    pub agent_mode: String,
    /// When true, disable spinner animations (accessibility/reduced-motion).
    pub reduced_motion: bool,
    /// Maximum retained mission-control lines.
    pub mission_control_max_events: usize,
    /// Thinking visibility mode (`concise` or `raw`).
    pub thinking_visibility: String,
}

impl ChatShell {
    pub fn push_transcript(&mut self, line: impl Into<String>) {
        self.transcript.push(TranscriptEntry {
            kind: MessageKind::Assistant,
            text: line.into(),
        });
    }

    pub fn push_user(&mut self, line: impl Into<String>) {
        self.transcript.push(TranscriptEntry {
            kind: MessageKind::User,
            text: line.into(),
        });
    }

    pub fn push_system(&mut self, line: impl Into<String>) {
        self.transcript.push(TranscriptEntry {
            kind: MessageKind::System,
            text: line.into(),
        });
    }

    pub fn push_error(&mut self, line: impl Into<String>) {
        self.transcript.push(TranscriptEntry {
            kind: MessageKind::Error,
            text: line.into(),
        });
    }

    pub fn push_tool_call(&mut self, tool_name: &str, args_summary: &str) {
        self.transcript.push(TranscriptEntry {
            kind: MessageKind::ToolCall,
            text: format!("{tool_name} {args_summary}"),
        });
    }

    pub fn push_tool_result(&mut self, tool_name: &str, duration_ms: u64, summary: &str) {
        let duration_str = if duration_ms >= 1000 {
            format!("{:.1}s", duration_ms as f64 / 1000.0)
        } else {
            format!("{duration_ms}ms")
        };
        self.transcript.push(TranscriptEntry {
            kind: MessageKind::ToolResult,
            text: format!("{tool_name} ({duration_str}) {summary}"),
        });
    }

    pub fn push_plan(&mut self, line: impl Into<String>) {
        self.plan_lines.push(line.into());
    }

    pub fn push_thinking(&mut self, line: impl Into<String>) {
        self.transcript.push(TranscriptEntry {
            kind: MessageKind::Thinking,
            text: line.into(),
        });
    }

    pub fn push_tool(&mut self, line: impl Into<String>) {
        self.tool_lines.push(line.into());
    }

    pub fn push_mission_control(&mut self, line: impl Into<String>) {
        self.mission_control_lines.push(line.into());
        if self.mission_control_max_events > 0
            && self.mission_control_lines.len() > self.mission_control_max_events
        {
            let over = self
                .mission_control_lines
                .len()
                .saturating_sub(self.mission_control_max_events);
            if over > 0 {
                self.mission_control_lines.drain(0..over);
            }
        }
    }

    pub fn push_artifact(&mut self, line: impl Into<String>) {
        self.artifact_lines.push(line.into());
    }

    /// Append text to the current streaming assistant response.
    /// Creates a new assistant entry if none is in progress.
    pub fn append_streaming(&mut self, text: &str) {
        if let Some(last) = self.transcript.last_mut()
            && last.kind == MessageKind::Assistant
        {
            last.text.push_str(text);
            return;
        }
        self.transcript.push(TranscriptEntry {
            kind: MessageKind::Assistant,
            text: text.to_string(),
        });
    }

    /// Clear any partially streamed assistant text — used when the response turns out to
    /// contain tool calls, making the interleaved text fragments visual noise.
    pub fn clear_streaming_text(&mut self) {
        if let Some(last) = self.transcript.last_mut()
            && last.kind == MessageKind::Assistant
        {
            last.text.clear();
        }
    }

    /// Finalize the streaming response — ensure the complete output is in the transcript.
    /// If streaming deltas were received, replaces the partial entry with the final output.
    /// If no streaming happened, pushes the full output as new transcript entries.
    pub fn finalize_streaming(&mut self, final_output: &str) {
        if final_output.is_empty() {
            return;
        }
        // Remove any partial streaming assistant entry built by append_streaming.
        if self
            .transcript
            .last()
            .is_some_and(|e| e.kind == MessageKind::Assistant)
        {
            self.transcript.pop();
        }
        // Push the complete output, one entry per line for proper rendering.
        for line in final_output.lines() {
            self.transcript.push(TranscriptEntry {
                kind: MessageKind::Assistant,
                text: line.to_string(),
            });
        }
    }

    /// Format and push a `/cost` command response into the transcript.
    pub fn push_cost_summary(&mut self, status: &UiStatus) {
        let ctx_pct = if status.context_max_tokens > 0 {
            (status.context_used_tokens as f64 / status.context_max_tokens as f64 * 100.0) as u64
        } else {
            0
        };
        self.push_system(format!(
            "Cost: ${:.4} | Tokens: {}K/{}K ({}%) | Turns: {}",
            status.estimated_cost_usd,
            status.context_used_tokens / 1000,
            status.context_max_tokens / 1000,
            ctx_pct,
            status.session_turns,
        ));
    }

    /// Format and push a `/status` command response into the transcript.
    pub fn push_status_summary(&mut self, status: &UiStatus) {
        self.push_system(format!("Model: {}", status.model));
        self.push_system(format!("Permission mode: {}", status.permission_mode));
        if !status.workflow_phase.is_empty() {
            self.push_system(format!("Workflow phase: {}", status.workflow_phase));
        }
        if status.plan_state != "none" {
            self.push_system(format!("Plan state: {}", status.plan_state));
        }
        if let Some(review) = status.pr_review_status.as_deref() {
            self.push_system(format!("PR review: {}", review));
        }
        self.push_system(format!("Pending approvals: {}", status.pending_approvals));
        self.push_system(format!("Active tasks: {}", status.active_tasks));
        self.push_system(format!("Background jobs: {}", status.background_jobs));
        if !status.provider_diagnostics_summary.is_empty() {
            self.push_system(format!(
                "Provider compatibility: {}",
                status.provider_diagnostics_summary
            ));
        }
        if !status.runtime_diagnostics_summary.is_empty() {
            self.push_system(format!(
                "Runtime diagnostics: {}",
                status.runtime_diagnostics_summary
            ));
        }
        self.push_system(format!(
            "Autopilot: {}",
            if status.autopilot_running {
                "running"
            } else {
                "idle"
            }
        ));
        self.push_cost_summary(status);
    }

    /// Format and push a `/model` command response into the transcript.
    pub fn push_model_info(&mut self, model: &str) {
        self.push_system(format!("Current model: {model}"));
    }

    pub(crate) fn spinner_frame(&self) -> &'static str {
        if self.reduced_motion {
            return "●";
        }
        const FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        FRAMES[self.spinner_tick % FRAMES.len()]
    }
}

// ─── Rewind Picker ──────────────────────────────────────────────────────────

/// Human-readable labels and their corresponding [`RewindAction`] values.
pub const REWIND_ACTIONS: &[(&str, codingbuddy_core::RewindAction)] = &[
    (
        "Restore code & conversation",
        codingbuddy_core::RewindAction::RestoreCodeAndConversation,
    ),
    (
        "Restore conversation only",
        codingbuddy_core::RewindAction::RestoreConversationOnly,
    ),
    (
        "Restore code only",
        codingbuddy_core::RewindAction::RestoreCodeOnly,
    ),
    (
        "Summarize from here",
        codingbuddy_core::RewindAction::Summarize,
    ),
    ("Cancel", codingbuddy_core::RewindAction::Cancel),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RewindPickerPhase {
    SelectCheckpoint,
    SelectAction,
}

#[derive(Debug, Clone)]
pub struct RewindPickerState {
    pub checkpoints: Vec<codingbuddy_store::CheckpointRecord>,
    pub selected_index: usize,
    pub action_index: usize,
    pub phase: RewindPickerPhase,
}

impl RewindPickerState {
    pub fn new(checkpoints: Vec<codingbuddy_store::CheckpointRecord>) -> Self {
        Self {
            checkpoints,
            selected_index: 0,
            action_index: 0,
            phase: RewindPickerPhase::SelectCheckpoint,
        }
    }

    /// Move selection up.
    pub fn up(&mut self) {
        let limit = match self.phase {
            RewindPickerPhase::SelectCheckpoint => self.checkpoints.len(),
            RewindPickerPhase::SelectAction => REWIND_ACTIONS.len(),
        };
        let idx = match self.phase {
            RewindPickerPhase::SelectCheckpoint => &mut self.selected_index,
            RewindPickerPhase::SelectAction => &mut self.action_index,
        };
        if *idx > 0 {
            *idx -= 1;
        } else {
            *idx = limit.saturating_sub(1);
        }
    }

    /// Move selection down.
    pub fn down(&mut self) {
        let limit = match self.phase {
            RewindPickerPhase::SelectCheckpoint => self.checkpoints.len(),
            RewindPickerPhase::SelectAction => REWIND_ACTIONS.len(),
        };
        let idx = match self.phase {
            RewindPickerPhase::SelectCheckpoint => &mut self.selected_index,
            RewindPickerPhase::SelectAction => &mut self.action_index,
        };
        if *idx + 1 < limit {
            *idx += 1;
        } else {
            *idx = 0;
        }
    }

    /// Confirm current selection. Returns `Some(action)` when a final action is chosen.
    pub fn confirm(&mut self) -> Option<(usize, codingbuddy_core::RewindAction)> {
        match self.phase {
            RewindPickerPhase::SelectCheckpoint => {
                if self.checkpoints.is_empty() {
                    return Some((0, codingbuddy_core::RewindAction::Cancel));
                }
                self.phase = RewindPickerPhase::SelectAction;
                self.action_index = 0;
                None
            }
            RewindPickerPhase::SelectAction => {
                let (_, action) = REWIND_ACTIONS[self.action_index];
                Some((self.selected_index, action))
            }
        }
    }

    /// Go back one phase, or return true if we should close the picker.
    pub fn back(&mut self) -> bool {
        match self.phase {
            RewindPickerPhase::SelectAction => {
                self.phase = RewindPickerPhase::SelectCheckpoint;
                false
            }
            RewindPickerPhase::SelectCheckpoint => true,
        }
    }

    /// Format a checkpoint line for display with relative timestamp and file count.
    pub fn format_checkpoint_line(
        cp: &codingbuddy_store::CheckpointRecord,
        selected: bool,
    ) -> String {
        let marker = if selected { ">" } else { " " };
        let time = format_relative_time(&cp.created_at);
        format!(
            "{marker} [{time}] {reason} ({files} files)",
            reason = cp.reason,
            files = cp.files_count
        )
    }

    /// Return the visible viewport range for long checkpoint lists (max 8 visible).
    pub fn viewport(&self) -> std::ops::Range<usize> {
        const VIEWPORT_SIZE: usize = 8;
        let total = self.checkpoints.len();
        if total <= VIEWPORT_SIZE {
            return 0..total;
        }
        let half = VIEWPORT_SIZE / 2;
        let start = if self.selected_index <= half {
            0
        } else if self.selected_index + half >= total {
            total - VIEWPORT_SIZE
        } else {
            self.selected_index - half
        };
        start..(start + VIEWPORT_SIZE).min(total)
    }
}

/// Format an ISO 8601 timestamp as a relative time string (e.g., "2m ago", "1h ago").
pub fn format_relative_time(timestamp: &str) -> String {
    // Parse ISO 8601 timestamp
    if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(timestamp) {
        let now = chrono::Utc::now();
        let delta = now.signed_duration_since(ts);

        if delta.num_seconds() < 60 {
            return "just now".to_string();
        }
        if delta.num_minutes() < 60 {
            return format!("{}m ago", delta.num_minutes());
        }
        if delta.num_hours() < 24 {
            return format!("{}h ago", delta.num_hours());
        }
        return format!("{}d ago", delta.num_days());
    }
    // Fallback: return the raw timestamp truncated
    timestamp.chars().take(19).collect()
}

// ─── Model Picker ───────────────────────────────────────────────────────────

/// Available model choices for the interactive `/model` picker.
pub(crate) const MODEL_CHOICES: &[(&str, &str)] = &[
    ("deepseek-chat", "Thinking + tools (default)"),
    ("deepseek-reasoner", "Deep reasoning + tools"),
];

#[derive(Debug, Clone, Default)]
pub struct ModelPickerState {
    pub selected: usize,
}

impl ModelPickerState {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }
    pub fn down(&mut self) {
        self.selected = (self.selected + 1).min(MODEL_CHOICES.len() - 1);
    }
    pub fn confirm(&self) -> &'static str {
        MODEL_CHOICES[self.selected].0
    }
}

// ─── Autocomplete Dropdown ───────────────────────────────────────────────────

/// State for the `@` file autocomplete dropdown.
#[derive(Debug, Clone)]
pub struct AutocompleteState {
    pub suggestions: Vec<String>,
    pub selected: usize,
    pub trigger_pos: usize, // position of trigger char ('@' or '/') in input
}

impl AutocompleteState {
    pub fn new(suggestions: Vec<String>, trigger_pos: usize) -> Self {
        Self {
            suggestions,
            selected: 0,
            trigger_pos,
        }
    }

    pub fn up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        } else if !self.suggestions.is_empty() {
            self.selected = self.suggestions.len() - 1;
        }
    }

    pub fn down(&mut self) {
        if !self.suggestions.is_empty() {
            self.selected = (self.selected + 1) % self.suggestions.len();
        }
    }

    pub fn selected_value(&self) -> Option<&str> {
        self.suggestions.get(self.selected).map(|s| s.as_str())
    }

    /// Format the dropdown display as text lines for the info area.
    pub fn display_lines(&self, max_lines: usize) -> Vec<String> {
        let total = self.suggestions.len();
        if total == 0 {
            return vec!["no matches".to_string()];
        }
        let show = total.min(max_lines);
        // Center viewport on selected
        let half = show / 2;
        let start = if self.selected <= half {
            0
        } else if self.selected + half >= total {
            total.saturating_sub(show)
        } else {
            self.selected - half
        };
        let end = (start + show).min(total);

        (start..end)
            .map(|i| {
                let marker = if i == self.selected { ">" } else { " " };
                format!("{marker} {}", self.suggestions[i])
            })
            .collect()
    }
}

// ─── ML Ghost Text ──────────────────────────────────────────────────────────

/// State for ML-powered ghost text (autocomplete) suggestions.
///
/// Ghost text rendering priority: ML suggestion > history ghost > none.
/// Uses DarkGray italic styling to distinguish from user input.
#[derive(Debug, Clone)]
pub struct GhostTextState {
    /// The current ghost text suggestion, if any.
    pub suggestion: Option<String>,
    /// When the last keystroke occurred (for debounce).
    pub last_keystroke: Instant,
    /// Whether a completion request is pending (debounce not yet elapsed).
    pub pending: bool,
    /// Debounce duration (default 200ms).
    pub debounce_ms: u64,
    /// Minimum input length before triggering completions.
    pub min_input_len: usize,
}

impl Default for GhostTextState {
    fn default() -> Self {
        Self {
            suggestion: None,
            last_keystroke: Instant::now(),
            pending: false,
            debounce_ms: 200,
            min_input_len: 3,
        }
    }
}

impl GhostTextState {
    /// Record a keystroke — clears current suggestion and resets debounce.
    pub fn on_keystroke(&mut self) {
        self.suggestion = None;
        self.last_keystroke = Instant::now();
        self.pending = true;
    }

    /// Check if the debounce period has elapsed and a completion should be requested.
    pub fn should_request(&self, input_len: usize) -> bool {
        self.pending
            && input_len >= self.min_input_len
            && self.last_keystroke.elapsed() >= Duration::from_millis(self.debounce_ms)
    }

    /// Set the suggestion from a completion callback result.
    pub fn set_suggestion(&mut self, text: Option<String>) {
        self.suggestion = text;
        self.pending = false;
    }

    /// Accept the full ghost text suggestion, returning it.
    pub fn accept_full(&mut self) -> Option<String> {
        self.pending = false;
        self.suggestion.take()
    }

    /// Accept one word from the ghost text suggestion.
    ///
    /// Uses word boundary detection that treats alphanumeric and underscore
    /// characters as part of a word (matching identifier conventions).
    pub fn accept_word(&mut self) -> Option<String> {
        if let Some(ref text) = self.suggestion {
            let trimmed = text.trim_start();
            let leading_ws = text.len() - trimmed.len();
            // Find end of the word: alphanumeric or underscore characters
            let word_len = trimmed
                .find(|c: char| !c.is_alphanumeric() && c != '_')
                .unwrap_or(trimmed.len());
            // If we're at a non-word char, take at least one character
            let word_end = leading_ws + word_len.max(1).min(trimmed.len());
            if word_end == 0 {
                return self.accept_full();
            }
            let word = text[..word_end].to_string();
            let remaining = &text[word_end..];
            if remaining.is_empty() {
                self.suggestion = None;
            } else {
                self.suggestion = Some(remaining.to_string());
            }
            self.pending = false;
            Some(word)
        } else {
            None
        }
    }
}

/// Type alias for the ML completion callback.
///
/// Takes the current input text and returns an optional completion suggestion.
pub type MlCompletionCallback = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;
