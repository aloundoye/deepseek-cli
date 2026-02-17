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

    pub fn score(&self, s: &RouterSignals) -> f32 {
        self.weights.w1 * s.prompt_complexity
            + self.weights.w2 * s.repo_breadth
            + self.weights.w3 * s.failure_streak
            + self.weights.w4 * s.verification_failures
            + self.weights.w5 * s.low_confidence
            + self.weights.w6 * s.ambiguity_flags
    }

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
        let planner_bias = matches!(unit, LlmUnit::Planner) && signals.repo_breadth > 0.5;
        let high = score >= self.cfg.threshold_high || planner_bias;
        let mut reason_codes = Vec::new();

        if high {
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
}
