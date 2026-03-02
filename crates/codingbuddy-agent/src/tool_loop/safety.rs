//! Safety mechanisms: doom loop detection, circuit breaker, and cost tracking.

use codingbuddy_core::TokenUsage;
use std::collections::VecDeque;

// ── Doom loop detection ──

/// Threshold: same tool+args hash this many times in a row → doom loop detected.
pub(crate) const DOOM_LOOP_THRESHOLD: usize = 3;

/// Rolling window size for doom loop detection.
pub(crate) const DOOM_LOOP_HISTORY_SIZE: usize = 10;

/// Finish reason returned when the doom loop gate terminates the loop.
pub(crate) const FINISH_REASON_DOOM_LOOP: &str = "doom_loop";

/// Guidance injected when a doom loop is detected.
pub(crate) const DOOM_LOOP_GUIDANCE: &str = "STOP — you are repeating the same action without progress. \
     Try a DIFFERENT approach: use a different tool, different arguments, \
     or ask the user for clarification. Do NOT repeat the same call again.";

/// Tracks repeated identical tool calls to detect doom loops — the model
/// repeating the same action without progress. Uses a rolling window of
/// (tool_name, args_hash) tuples.
#[derive(Debug, Clone)]
pub(crate) struct DoomLoopTracker {
    /// Rolling window of recent (tool_name, args_hash) pairs.
    pub(crate) recent_calls: VecDeque<(String, u64)>,
    /// Whether doom loop guidance has been injected (reset when model uses a different call).
    pub(crate) warning_injected: bool,
}

impl Default for DoomLoopTracker {
    fn default() -> Self {
        Self {
            recent_calls: VecDeque::with_capacity(DOOM_LOOP_HISTORY_SIZE),
            warning_injected: false,
        }
    }
}

impl DoomLoopTracker {
    /// Record a tool call. Returns `true` if a doom loop is detected and
    /// guidance should be injected (fires once per loop cycle).
    ///
    /// Hashes the raw args string directly (no JSON round-trip) for efficiency.
    pub(crate) fn record(&mut self, tool_name: &str, raw_args: &str) -> bool {
        use std::hash::{Hash, Hasher};

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        raw_args.hash(&mut hasher);
        let args_hash = hasher.finish();

        let entry = (tool_name.to_string(), args_hash);

        // Check if this is a different call than the last one — reset warning
        if let Some(last) = self.recent_calls.back()
            && *last != entry
        {
            self.warning_injected = false;
        }

        // Add to rolling window
        self.recent_calls.push_back(entry.clone());
        if self.recent_calls.len() > DOOM_LOOP_HISTORY_SIZE {
            self.recent_calls.pop_front();
        }

        // Count occurrences of this exact call in recent history
        let count = self.recent_calls.iter().filter(|c| **c == entry).count();

        count >= DOOM_LOOP_THRESHOLD && !self.warning_injected
    }

    /// Mark that guidance has been injected for the current doom loop.
    pub(crate) fn mark_warned(&mut self) {
        self.warning_injected = true;
    }
}

// ── Circuit breaker ──

/// Number of consecutive failures before a tool is temporarily disabled.
pub(crate) const CIRCUIT_BREAKER_THRESHOLD: usize = 3;

/// Number of turns a disabled tool stays in cooldown before re-enabling.
pub(crate) const CIRCUIT_BREAKER_COOLDOWN_TURNS: usize = 2;

/// Tracks consecutive failures for a single tool.
#[derive(Debug, Clone, Default)]
pub(crate) struct CircuitBreakerState {
    pub(crate) consecutive_failures: usize,
    /// Turns remaining in cooldown (0 = active).
    pub(crate) cooldown_remaining: usize,
}

// ── Cost tracking ──

/// Default cost warning threshold in USD. One-shot warning at this level.
pub(crate) const DEFAULT_COST_WARNING_USD: f64 = 0.50;

/// Tracks cumulative token usage and estimated cost across a session.
#[derive(Debug)]
pub struct CostTracker {
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cache_hit_tokens: u64,
    pub total_reasoning_tokens: u64,
    /// Per-million-token pricing for input.
    pub cost_per_million_input: f64,
    /// Per-million-token pricing for output.
    pub cost_per_million_output: f64,
    /// Discount factor for cache-hit tokens (e.g. 0.1 means 90% discount).
    pub cache_discount: f64,
    /// Optional hard budget cap. If set, loop stops when exceeded.
    pub max_budget_usd: Option<f64>,
    /// Whether the user has already been warned about cost threshold.
    pub(crate) warned: bool,
}

impl Default for CostTracker {
    fn default() -> Self {
        Self {
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_cache_hit_tokens: 0,
            total_reasoning_tokens: 0,
            cost_per_million_input: 0.27,
            cost_per_million_output: 1.10,
            cache_discount: 0.1,
            max_budget_usd: None,
            warned: false,
        }
    }
}

impl CostTracker {
    /// Record token usage from an API response.
    pub fn record(&mut self, usage: &TokenUsage) {
        self.total_input_tokens += usage.prompt_tokens;
        self.total_output_tokens += usage.completion_tokens;
        self.total_cache_hit_tokens += usage.prompt_cache_hit_tokens;
        self.total_reasoning_tokens += usage.reasoning_tokens;
    }

    /// Compute estimated cumulative cost in USD.
    pub fn estimated_cost_usd(&self) -> f64 {
        let input_tokens = self.total_input_tokens as f64;
        let cache_tokens = self.total_cache_hit_tokens as f64;
        let output_tokens = self.total_output_tokens as f64;

        // Cache-hit tokens are charged at a discount
        let effective_input = (input_tokens - cache_tokens) + (cache_tokens * self.cache_discount);
        let input_cost = effective_input / 1_000_000.0 * self.cost_per_million_input;
        let output_cost = output_tokens / 1_000_000.0 * self.cost_per_million_output;

        input_cost + output_cost
    }

    /// Whether the cost has exceeded the hard budget cap.
    pub fn over_budget(&self) -> bool {
        self.max_budget_usd
            .is_some_and(|cap| self.estimated_cost_usd() > cap)
    }

    /// Whether the cost has passed the warning threshold but not yet warned.
    pub fn should_warn(&mut self) -> bool {
        if self.warned {
            return false;
        }
        if self.estimated_cost_usd() > DEFAULT_COST_WARNING_USD {
            self.warned = true;
            return true;
        }
        false
    }
}

// ── Error recovery guidance ──

/// Guidance injected when the first evidence-driven escalation fires.
pub(crate) const ERROR_RECOVERY_GUIDANCE: &str = "ERROR RECOVERY: Your previous approach failed. Before retrying:\n\
     1. Re-read the relevant file(s) to see the actual current state\n\
     2. Check for typos in file paths, function names, and variable names\n\
     3. Consider a fundamentally different approach if the same strategy has failed";

/// Stronger guidance injected when the same error appears 3+ times.
pub(crate) const STUCK_DETECTION_GUIDANCE: &str = "STUCK DETECTION: You have hit the same error 3+ times. You MUST try a \
     completely different approach. Consider:\n\
     - Reading more context from adjacent files\n\
     - Checking the project's test suite or build system\n\
     - Simplifying your change to the smallest possible edit\n\
     - Asking the user for clarification";

/// Maximum number of recent errors to track for stuck detection.
pub(crate) const MAX_RECENT_ERRORS: usize = 10;
