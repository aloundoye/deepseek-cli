//! Speculative decoding — accelerates local generation using a draft model.
//!
//! A small "draft" model proposes N tokens, then the larger "target" model
//! verifies them in a single forward pass. Accepted tokens skip the expensive
//! per-token autoregressive loop. Typical speedup: 2-3x when the draft model
//! has high acceptance rate (e.g., for common code patterns).
//!
//! This module provides the verification algorithm. The actual model loading
//! is handled by the `CandleCompletion` backend with an optional draft model.

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens to draft speculatively before verification.
    pub draft_tokens: usize,
    /// Maximum probability ratio threshold for acceptance (higher = more lenient).
    /// Tokens where P(target)/P(draft) >= threshold are accepted.
    pub acceptance_threshold: f64,
    /// Model ID for the draft model (must be GGUF-compatible).
    pub draft_model_id: String,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_tokens: 5,
            acceptance_threshold: 0.5,
            draft_model_id: "tinyllama-1.1b-chat".to_string(),
        }
    }
}

/// Result of verifying a draft sequence against the target model.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Number of draft tokens accepted.
    pub accepted: usize,
    /// Total draft tokens proposed.
    pub proposed: usize,
    /// The next token to generate (from the target model at the first rejection point).
    pub correction_token: Option<u32>,
}

/// Verify a draft token sequence against target model logits.
///
/// For each drafted token, compares draft probability vs target probability.
/// Accepts greedily from left until a rejection occurs.
///
/// # Arguments
/// - `draft_tokens`: Token IDs proposed by the draft model
/// - `draft_probs`: Log-probabilities from draft model for each token
/// - `target_probs`: Log-probabilities from target model for each token
/// - `threshold`: Minimum acceptance ratio
///
/// Returns a `VerificationResult` indicating how many tokens were accepted.
pub fn verify_draft(
    draft_tokens: &[u32],
    draft_probs: &[f64],
    target_probs: &[f64],
    threshold: f64,
) -> VerificationResult {
    assert_eq!(draft_tokens.len(), draft_probs.len());
    assert_eq!(draft_tokens.len(), target_probs.len());

    let mut accepted = 0;

    for i in 0..draft_tokens.len() {
        // Convert log-probs to probs for ratio computation
        let p_draft = draft_probs[i].exp();
        let p_target = target_probs[i].exp();

        // Accept if target agrees with draft (high probability ratio)
        if p_draft > 0.0 && (p_target / p_draft) >= threshold {
            accepted += 1;
        } else {
            // First rejection — stop accepting
            break;
        }
    }

    VerificationResult {
        accepted,
        proposed: draft_tokens.len(),
        correction_token: None, // Filled by caller from target model
    }
}

/// Compute the acceptance rate for monitoring/logging.
pub fn acceptance_rate(result: &VerificationResult) -> f64 {
    if result.proposed == 0 {
        return 0.0;
    }
    result.accepted as f64 / result.proposed as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_tokens_accepted_when_models_agree() {
        // Draft and target produce identical probabilities
        let draft_probs = vec![(-0.1_f64).exp().ln(); 5]; // same log-probs
        let target_probs = draft_probs.clone();
        let tokens = vec![1, 2, 3, 4, 5];

        let result = verify_draft(&tokens, &draft_probs, &target_probs, 0.5);
        assert_eq!(result.accepted, 5);
        assert_eq!(result.proposed, 5);
    }

    #[test]
    fn no_tokens_accepted_when_target_disagrees() {
        // Target has much lower probability
        let draft_probs = vec![(-0.1_f64).ln(); 3]; // high prob
        let target_probs = vec![(-5.0_f64).ln(); 3]; // very low prob
        let tokens = vec![1, 2, 3];

        let result = verify_draft(&tokens, &draft_probs, &target_probs, 0.5);
        assert_eq!(result.accepted, 0);
    }

    #[test]
    fn partial_acceptance() {
        let tokens = vec![1, 2, 3, 4];
        // First two agree, third disagrees
        let draft_probs = vec![-0.1, -0.1, -0.1, -0.1]; // log-probs
        let target_probs = vec![-0.1, -0.1, -10.0, -0.1]; // third token rejected

        let result = verify_draft(&tokens, &draft_probs, &target_probs, 0.5);
        assert_eq!(result.accepted, 2);
        assert_eq!(result.proposed, 4);
    }

    #[test]
    fn acceptance_rate_computation() {
        let result = VerificationResult {
            accepted: 3,
            proposed: 5,
            correction_token: None,
        };
        assert!((acceptance_rate(&result) - 0.6).abs() < 0.001);
    }

    #[test]
    fn acceptance_rate_zero_proposed() {
        let result = VerificationResult {
            accepted: 0,
            proposed: 0,
            correction_token: None,
        };
        assert_eq!(acceptance_rate(&result), 0.0);
    }

    #[test]
    fn default_config_reasonable() {
        let cfg = SpeculativeConfig::default();
        assert_eq!(cfg.draft_tokens, 5);
        assert!(cfg.acceptance_threshold > 0.0 && cfg.acceptance_threshold < 1.0);
    }
}
