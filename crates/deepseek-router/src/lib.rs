use deepseek_core::{
    DEEPSEEK_V32_CHAT_MODEL, DEEPSEEK_V32_REASONER_MODEL, LlmConfig, LlmUnit, ModelRouter,
    RouterConfig as AppRouterConfig, RouterDecision, RouterSignals, RouterWeights,
};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct RouterConfig {
    pub base_model: String,
    pub max_think_model: String,
    pub threshold_high: f32,
    pub max_escalations_per_unit: u8,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            base_model: DEEPSEEK_V32_CHAT_MODEL.to_string(),
            max_think_model: DEEPSEEK_V32_REASONER_MODEL.to_string(),
            threshold_high: 0.72,
            max_escalations_per_unit: 1,
        }
    }
}

pub struct WeightedRouter {
    pub cfg: RouterConfig,
    pub weights: RouterWeights,
}

impl WeightedRouter {
    pub fn new(cfg: RouterConfig) -> Self {
        Self {
            cfg,
            weights: RouterWeights::default(),
        }
    }

    pub fn from_app_config(router: &AppRouterConfig, llm: &LlmConfig) -> Self {
        Self {
            cfg: RouterConfig {
                base_model: llm.base_model.clone(),
                max_think_model: llm.max_think_model.clone(),
                threshold_high: router.threshold_high,
                max_escalations_per_unit: router.max_escalations_per_unit,
            },
            weights: RouterWeights {
                w1: router.w1,
                w2: router.w2,
                w3: router.w3,
                w4: router.w4,
                w5: router.w5,
                w6: router.w6,
            },
        }
    }

    #[must_use]
    pub fn score(&self, s: &RouterSignals) -> f32 {
        self.weights.w1 * s.prompt_complexity
            + self.weights.w2 * s.repo_breadth
            + self.weights.w3 * s.failure_streak
            + self.weights.w4 * s.verification_failures
            + self.weights.w5 * s.low_confidence
            + self.weights.w6 * s.ambiguity_flags
    }

    #[must_use]
    pub fn should_escalate_retry(&self, unit: &LlmUnit, invalid_output: bool, retries: u8) -> bool {
        if retries >= self.cfg.max_escalations_per_unit || !invalid_output {
            return false;
        }
        match unit {
            LlmUnit::Planner => true,
            LlmUnit::Executor => true,
        }
    }
}

impl ModelRouter for WeightedRouter {
    fn select(&self, unit: LlmUnit, signals: RouterSignals) -> RouterDecision {
        let score = self.score(&signals);
        let score_high = score >= self.cfg.threshold_high;
        // Planner bias should only kick in for broad, uncertain planning states.
        // This keeps default hybrid routing chat-first for routine prompts.
        let planner_bias = matches!(unit, LlmUnit::Planner)
            && signals.repo_breadth >= 0.85
            && (signals.low_confidence >= 0.55
                || signals.failure_streak >= 0.45
                || signals.verification_failures >= 0.35
                || signals.ambiguity_flags >= 0.65);
        let high = score_high || planner_bias;
        let mut reason_codes = Vec::new();

        if score_high {
            reason_codes.push("threshold_high".to_string());
        }
        if planner_bias {
            reason_codes.push("planner_repo_breadth_bias".to_string());
        }
        if signals.failure_streak > 0.6 {
            reason_codes.push("failure_streak".to_string());
        }

        RouterDecision {
            decision_id: Uuid::now_v7(),
            reason_codes,
            selected_model: if high {
                self.cfg.max_think_model.clone()
            } else {
                self.cfg.base_model.clone()
            },
            confidence: (1.0 - (score - self.cfg.threshold_high).abs()).clamp(0.1, 1.0),
            score,
            escalated: high,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escalates_on_high_score() {
        let router = WeightedRouter::new(RouterConfig::default());
        let decision = router.select(
            LlmUnit::Planner,
            RouterSignals {
                prompt_complexity: 1.0,
                repo_breadth: 1.0,
                failure_streak: 1.0,
                verification_failures: 1.0,
                low_confidence: 1.0,
                ambiguity_flags: 1.0,
            },
        );
        assert_eq!(decision.selected_model, "deepseek-reasoner");
    }

    #[test]
    fn low_score_selects_base_model() {
        let router = WeightedRouter::new(RouterConfig::default());
        let decision = router.select(
            LlmUnit::Executor,
            RouterSignals {
                prompt_complexity: 0.0,
                repo_breadth: 0.0,
                failure_streak: 0.0,
                verification_failures: 0.0,
                low_confidence: 0.0,
                ambiguity_flags: 0.0,
            },
        );
        assert_eq!(decision.selected_model, "deepseek-chat");
        assert!(!decision.escalated);
    }

    #[test]
    fn threshold_boundary_exactly_at_072() {
        let mut router = WeightedRouter::new(RouterConfig::default());
        // Set all weights to 1.0 and all signals to produce exactly 0.72
        router.weights = RouterWeights {
            w1: 1.0,
            w2: 0.0,
            w3: 0.0,
            w4: 0.0,
            w5: 0.0,
            w6: 0.0,
        };
        let decision = router.select(
            LlmUnit::Executor,
            RouterSignals {
                prompt_complexity: 0.72,
                repo_breadth: 0.0,
                failure_streak: 0.0,
                verification_failures: 0.0,
                low_confidence: 0.0,
                ambiguity_flags: 0.0,
            },
        );
        assert!(
            decision.escalated,
            "score exactly at threshold should escalate"
        );
        assert_eq!(decision.selected_model, "deepseek-reasoner");
    }

    #[test]
    fn threshold_boundary_just_below_072() {
        let mut router = WeightedRouter::new(RouterConfig::default());
        router.weights = RouterWeights {
            w1: 1.0,
            w2: 0.0,
            w3: 0.0,
            w4: 0.0,
            w5: 0.0,
            w6: 0.0,
        };
        let decision = router.select(
            LlmUnit::Executor,
            RouterSignals {
                prompt_complexity: 0.71,
                repo_breadth: 0.0,
                failure_streak: 0.0,
                verification_failures: 0.0,
                low_confidence: 0.0,
                ambiguity_flags: 0.0,
            },
        );
        assert!(
            !decision.escalated,
            "score below threshold should not escalate"
        );
        assert_eq!(decision.selected_model, "deepseek-chat");
    }

    #[test]
    fn planner_bias_overrides_low_score() {
        let router = WeightedRouter::new(RouterConfig::default());
        let decision = router.select(
            LlmUnit::Planner,
            RouterSignals {
                prompt_complexity: 0.0,
                repo_breadth: 0.9, // broad scope
                failure_streak: 0.0,
                verification_failures: 0.0,
                low_confidence: 0.8, // uncertain prompt requires stronger reasoning
                ambiguity_flags: 0.0,
            },
        );
        assert!(
            decision.escalated,
            "planner bias should escalate even with low score"
        );
        assert_eq!(decision.selected_model, "deepseek-reasoner");
        assert!(
            decision
                .reason_codes
                .iter()
                .any(|code| code == "planner_repo_breadth_bias")
        );
        assert!(
            !decision
                .reason_codes
                .iter()
                .any(|code| code == "threshold_high"),
            "planner bias escalation should not report threshold_high unless score crossed threshold"
        );
    }

    #[test]
    fn planner_bias_does_not_trigger_on_routine_scope() {
        let router = WeightedRouter::new(RouterConfig::default());
        let decision = router.select(
            LlmUnit::Planner,
            RouterSignals {
                prompt_complexity: 0.1,
                repo_breadth: 0.6,
                failure_streak: 0.0,
                verification_failures: 0.0,
                low_confidence: 0.2,
                ambiguity_flags: 0.1,
            },
        );
        assert!(!decision.escalated);
        assert_eq!(decision.selected_model, "deepseek-chat");
    }

    #[test]
    fn should_escalate_retry_respects_max() {
        let router = WeightedRouter::new(RouterConfig {
            max_escalations_per_unit: 1,
            ..RouterConfig::default()
        });
        assert!(router.should_escalate_retry(&LlmUnit::Planner, true, 0));
        assert!(!router.should_escalate_retry(&LlmUnit::Planner, true, 1));
        assert!(!router.should_escalate_retry(&LlmUnit::Planner, false, 0));
    }

    #[test]
    fn custom_weights_change_score() {
        let mut router = WeightedRouter::new(RouterConfig::default());
        let signals = RouterSignals {
            prompt_complexity: 0.5,
            repo_breadth: 0.5,
            failure_streak: 0.0,
            verification_failures: 0.0,
            low_confidence: 0.0,
            ambiguity_flags: 0.0,
        };
        let default_score = router.score(&signals);

        router.weights = RouterWeights {
            w1: 2.0,
            w2: 0.0,
            w3: 0.0,
            w4: 0.0,
            w5: 0.0,
            w6: 0.0,
        };
        let custom_score = router.score(&signals);
        assert!((custom_score - 1.0).abs() < f32::EPSILON);
        assert!((custom_score - default_score).abs() > f32::EPSILON);
    }
}
