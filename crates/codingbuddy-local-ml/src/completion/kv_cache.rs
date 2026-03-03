//! KV cache for prefix sharing in autoregressive generation.
//!
//! When the user types incrementally (e.g., "fn foo(" → "fn foo(x:"), the
//! prefix tokens don't change. Caching their key/value projections avoids
//! recomputing them, reducing latency by 2-10x for ghost text.

/// Token-level KV cache for autoregressive models.
///
/// Stores key/value tensors from previous forward passes so that shared
/// prefixes don't need to be recomputed.
pub trait KvCache: Send + Sync {
    /// Number of cached token positions.
    fn prefix_len(&self) -> usize;

    /// Invalidate all cached positions from `position` onward.
    fn invalidate_from(&mut self, position: usize);

    /// Clear the entire cache.
    fn clear(&mut self);
}

/// Simple in-memory KV cache that tracks the length of the cached prefix.
///
/// The actual tensor storage is managed by the model backend (candle's
/// `ModelWeights` has internal KV state). This struct tracks the logical
/// prefix length so we know where the cached prefix ends and where new
/// computation must begin.
pub struct PrefixTracker {
    /// The token IDs of the cached prefix.
    cached_tokens: Vec<u32>,
}

impl PrefixTracker {
    pub fn new() -> Self {
        Self {
            cached_tokens: Vec::new(),
        }
    }

    /// Compute how many tokens of `new_tokens` match the cached prefix.
    ///
    /// Returns the length of the shared prefix. The caller should:
    /// - Skip forward computation for positions 0..shared_len
    /// - Invalidate the model's KV cache from `shared_len` onward
    /// - Run forward on `new_tokens[shared_len..]`
    pub fn compute_prefix_match(&self, new_tokens: &[u32]) -> usize {
        self.cached_tokens
            .iter()
            .zip(new_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Update the cached prefix after a successful forward pass.
    pub fn update(&mut self, tokens: &[u32]) {
        self.cached_tokens.clear();
        self.cached_tokens.extend_from_slice(tokens);
    }
}

impl Default for PrefixTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl KvCache for PrefixTracker {
    fn prefix_len(&self) -> usize {
        self.cached_tokens.len()
    }

    fn invalidate_from(&mut self, position: usize) {
        self.cached_tokens.truncate(position);
    }

    fn clear(&mut self) {
        self.cached_tokens.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_cache_matches_nothing() {
        let tracker = PrefixTracker::new();
        assert_eq!(tracker.prefix_len(), 0);
        assert_eq!(tracker.compute_prefix_match(&[1, 2, 3]), 0);
    }

    #[test]
    fn full_prefix_match() {
        let mut tracker = PrefixTracker::new();
        tracker.update(&[10, 20, 30]);
        assert_eq!(tracker.prefix_len(), 3);
        assert_eq!(tracker.compute_prefix_match(&[10, 20, 30, 40]), 3);
    }

    #[test]
    fn partial_prefix_match() {
        let mut tracker = PrefixTracker::new();
        tracker.update(&[10, 20, 30]);
        // First two tokens match, third diverges
        assert_eq!(tracker.compute_prefix_match(&[10, 20, 99]), 2);
    }

    #[test]
    fn no_prefix_match() {
        let mut tracker = PrefixTracker::new();
        tracker.update(&[10, 20, 30]);
        assert_eq!(tracker.compute_prefix_match(&[99, 88, 77]), 0);
    }

    #[test]
    fn invalidate_from_truncates() {
        let mut tracker = PrefixTracker::new();
        tracker.update(&[1, 2, 3, 4, 5]);
        tracker.invalidate_from(3);
        assert_eq!(tracker.prefix_len(), 3);
        assert_eq!(tracker.compute_prefix_match(&[1, 2, 3, 99]), 3);
    }

    #[test]
    fn clear_resets() {
        let mut tracker = PrefixTracker::new();
        tracker.update(&[1, 2, 3]);
        tracker.clear();
        assert_eq!(tracker.prefix_len(), 0);
    }

    #[test]
    fn update_replaces_previous() {
        let mut tracker = PrefixTracker::new();
        tracker.update(&[1, 2, 3]);
        tracker.update(&[4, 5]);
        assert_eq!(tracker.prefix_len(), 2);
        assert_eq!(tracker.compute_prefix_match(&[4, 5, 6]), 2);
        assert_eq!(tracker.compute_prefix_match(&[1, 2, 3]), 0);
    }
}
