use crate::*;

impl AgentEngine {
    pub(crate) fn generate_plan_for_chat(
        &self,
        prompt: &str,
        messages: &[ChatMessage],
    ) -> Option<Plan> {
        let planning_prompt = format!(
            "Before executing, create a step-by-step plan for the following task.\n\n\
             Task: {prompt}\n\n\
             Return a JSON object with these fields:\n\
             - \"goal\": string (1-sentence summary)\n\
             - \"assumptions\": [string] (what you're assuming)\n\
             - \"steps\": [{{\"title\": string, \"intent\": string, \"tools\": [string], \"files\": [string]}}]\n\
             - \"verification\": [string] (shell commands to verify success)\n\
             - \"risk_notes\": [string]\n\n\
             Keep it concise: 3-8 steps max. Only include files you plan to modify.\n\
             Return ONLY the JSON object, no markdown fences."
        );

        // Build minimal messages: system + planning prompt
        let plan_messages = vec![
            messages.first().cloned().unwrap_or(ChatMessage::System {
                content: "You are a coding assistant creating a task plan.".to_string(),
            }),
            ChatMessage::User {
                content: planning_prompt,
            },
        ];

        let request = ChatRequest {
            model: self.cfg.llm.base_model.clone(),
            messages: plan_messages,
            tools: vec![],
            tool_choice: ToolChoice::none(),
            max_tokens: 2048,
            temperature: Some(0.3),
            thinking: None,
        };

        match self.llm.complete_chat(&request) {
            Ok(response) => {
                let text = if !response.text.is_empty() {
                    &response.text
                } else {
                    &response.reasoning_content
                };
                parse_plan_from_llm(text, prompt)
            }
            Err(e) => {
                self.observer
                    .verbose_log(&format!("plan_discipline: LLM plan generation error: {e}"));
                None
            }
        }
    }

    /// Run verification commands and return true if all pass.
    pub(crate) fn revise_plan_with_llm(
        &self,
        session_id: Uuid,
        user_prompt: &str,
        current_plan: &Plan,
        failure_streak: u32,
        failure_detail: &str,
        non_urgent: bool,
    ) -> Result<Plan> {
        let revision_prompt =
            build_plan_revision_prompt(user_prompt, current_plan, failure_streak, failure_detail);
        let mut decision = self.router.select(
            LlmUnit::Planner,
            RouterSignals {
                prompt_complexity: (user_prompt.len() as f32 / 500.0).min(1.0),
                repo_breadth: 0.7,
                failure_streak: (failure_streak as f32 / 3.0).min(1.0),
                verification_failures: 0.0,
                low_confidence: 0.6,
                ambiguity_flags: 0.4,
            },
        );
        let revision_retry_index = failure_streak.saturating_sub(1).min(u32::from(u8::MAX)) as u8;
        if self.cfg.router.auto_max_think
            && self
                .router
                .should_escalate_retry(&LlmUnit::Planner, true, revision_retry_index)
            && !decision
                .selected_model
                .eq_ignore_ascii_case(&self.cfg.llm.max_think_model)
        {
            decision.selected_model = self.cfg.llm.max_think_model.clone();
            decision.escalated = true;
            if !decision
                .reason_codes
                .iter()
                .any(|code| code == "revision_failure_escalation")
            {
                decision
                    .reason_codes
                    .push("revision_failure_escalation".to_string());
            }
        }
        self.emit(
            session_id,
            EventKind::RouterDecisionV1 {
                decision: decision.clone(),
            },
        )?;
        self.observer.record_router_decision(&decision)?;
        if decision.escalated {
            self.emit(
                session_id,
                EventKind::RouterEscalationV1 {
                    reason_codes: decision.reason_codes.clone(),
                },
            )?;
        }

        let response = self.complete_with_cache(
            session_id,
            &LlmRequest {
                unit: LlmUnit::Planner,
                prompt: revision_prompt.clone(),
                model: decision.selected_model.clone(),
                max_tokens: 4096,
                non_urgent,
                images: vec![],
            },
        )?;
        self.emit(
            session_id,
            EventKind::UsageUpdatedV1 {
                unit: LlmUnit::Planner,
                model: decision.selected_model.clone(),
                input_tokens: estimate_tokens(&revision_prompt),
                output_tokens: estimate_tokens(&response.text),
            },
        )?;
        self.emit_cost_event(
            session_id,
            estimate_tokens(&revision_prompt),
            estimate_tokens(&response.text),
        )?;

        let mut revised = parse_plan_from_llm(&response.text, user_prompt)
            .ok_or_else(|| anyhow!("llm revision response did not contain a valid plan"))?;
        revised.version = current_plan.version + 1;
        Ok(revised)
    }
}

pub(crate) fn build_planner_prompt(task: &str) -> String {
    format!(
        "Return only JSON with keys: goal, assumptions, steps, verification, risk_notes. \
         Each step must include: title, intent, tools, files. User task: {task}"
    )
}

pub(crate) fn build_plan_revision_prompt(
    user_prompt: &str,
    current_plan: &Plan,
    failure_streak: u32,
    failure_detail: &str,
) -> String {
    let plan_json = serde_json::to_string_pretty(current_plan)
        .unwrap_or_else(|_| "{\"error\":\"failed to serialize plan\"}".to_string());
    format!(
        "The current execution plan failed and needs revision.\n\
Return ONLY JSON with keys: goal, assumptions, steps, verification, risk_notes.\n\
Each step must include: title, intent, tools, files.\n\
Keep successful structure where possible and focus on fixing the failure.\n\n\
User goal:\n{user_prompt}\n\n\
Failure streak: {failure_streak}\n\
Latest failure:\n{failure_detail}\n\n\
Current plan:\n{plan_json}"
    )
}
