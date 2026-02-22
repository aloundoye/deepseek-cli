use crate::planner::generation::build_planner_prompt;
use crate::subagents_runtime::delegated::should_parallel_execute_calls;
use crate::subagents_runtime::memory::{
    augment_goal_with_subagent_notes, summarize_subagent_notes,
};
use crate::*;

impl AgentEngine {
    pub fn ensure_session(&self) -> Result<Session> {
        if let Some(existing) = self.store.load_latest_session()? {
            return Ok(existing);
        }
        let session = Session {
            session_id: Uuid::now_v7(),
            workspace_root: self.workspace.to_string_lossy().to_string(),
            baseline_commit: None,
            status: SessionState::Idle,
            budgets: SessionBudgets {
                per_turn_seconds: 120,
                max_think_tokens: 8192,
            },
            active_plan_id: None,
        };
        self.store.save_session(&session)?;
        Ok(session)
    }

    pub fn run_once_with_mode_and_priority(
        &self,
        prompt: &str,
        allow_tools: bool,
        force_max_think: bool,
        non_urgent: bool,
    ) -> Result<String> {
        let mut state = self.prepare_run_once_state(prompt, force_max_think, non_urgent)?;
        let execution = self.execute_run_once_plan_steps(
            state.session.session_id,
            prompt,
            &mut state.plan,
            &state.runtime_goal,
            allow_tools,
            non_urgent,
        )?;
        let verification_failures = self.run_once_verification(
            &mut state.session,
            &state.plan,
            allow_tools,
            execution.execution_failed,
        )?;
        self.emit_run_once_verification_router(
            state.session.session_id,
            prompt,
            execution.failure_streak,
            verification_failures,
        )?;
        self.emit_run_once_telemetry(
            state.session.session_id,
            &state.plan,
            execution.failure_streak,
            verification_failures,
        )?;
        let run_succeeded = self.finalize_run_once_session(
            &mut state.session,
            prompt,
            &state.plan,
            execution.failure_streak,
            verification_failures,
            execution.execution_failed,
        )?;
        self.tool_host.fire_stop_hooks();
        self.build_run_once_summary(
            prompt,
            &state,
            &execution,
            verification_failures,
            run_succeeded,
        )
    }

    fn prepare_run_once_state(
        &self,
        prompt: &str,
        force_max_think: bool,
        non_urgent: bool,
    ) -> Result<RunOnceState> {
        let mut session = self.ensure_session()?;
        self.tool_host.fire_session_hooks("sessionstart");
        self.emit(
            session.session_id,
            EventKind::TurnAddedV1 {
                role: "user".to_string(),
                content: prompt.to_string(),
            },
        )?;
        let plan = self.plan_only_with_mode(prompt, force_max_think, non_urgent)?;
        session = self.ensure_session()?;
        self.transition(&mut session, SessionState::ExecutingStep)?;
        let subagent_notes = self.run_subagents(session.session_id, &plan, None)?;
        let runtime_goal = augment_goal_with_subagent_notes(&plan.goal, &subagent_notes);
        if !subagent_notes.is_empty() {
            self.emit(
                session.session_id,
                EventKind::TurnAddedV1 {
                    role: "assistant".to_string(),
                    content: format!("[subagents]\n{}", summarize_subagent_notes(&subagent_notes)),
                },
            )?;
            self.tool_host.fire_session_hooks("notification");
        }
        Ok(RunOnceState {
            session,
            plan,
            runtime_goal,
            subagent_notes,
        })
    }

    fn execute_run_once_plan_steps(
        &self,
        session_id: Uuid,
        prompt: &str,
        plan: &mut Plan,
        runtime_goal: &str,
        allow_tools: bool,
        non_urgent: bool,
    ) -> Result<RunOnceExecution> {
        let mut failure_streak = 0_u32;
        let mut execution_failed = false;
        let mut step_cursor = 0usize;
        let mut revision_budget = self.cfg.router.max_escalations_per_unit.max(1) as usize;
        let mut turn_count: u64 = 0;

        while step_cursor < plan.steps.len() {
            turn_count += 1;
            if self.run_once_limits_exceeded(session_id, turn_count)? {
                execution_failed = true;
                break;
            }

            let step = plan.steps[step_cursor].clone();
            let outcome =
                self.execute_run_once_step(session_id, &step, runtime_goal, allow_tools)?;
            plan.steps[step_cursor].done = outcome.success;
            self.emit(
                session_id,
                EventKind::StepMarkedV1 {
                    step_id: step.step_id,
                    done: outcome.success,
                    note: outcome.notes.clone(),
                },
            )?;

            if outcome.success {
                step_cursor += 1;
                continue;
            }

            failure_streak += 1;
            if revision_budget == 0 {
                execution_failed = true;
                break;
            }
            revision_budget = revision_budget.saturating_sub(1);
            let revised = self.revise_run_once_plan(
                session_id,
                prompt,
                plan,
                failure_streak,
                &outcome.notes,
                non_urgent,
            )?;
            self.emit(
                session_id,
                EventKind::PlanRevisedV1 {
                    plan: revised.clone(),
                },
            )?;
            self.tool_host.fire_session_hooks("planrevised");
            *plan = revised;
            step_cursor = 0;
        }

        Ok(RunOnceExecution {
            failure_streak,
            execution_failed,
        })
    }

    fn run_once_limits_exceeded(&self, session_id: Uuid, turn_count: u64) -> Result<bool> {
        if let Some(max) = self.max_turns
            && turn_count > max
        {
            self.emit(
                session_id,
                EventKind::TurnLimitExceededV1 {
                    limit: max,
                    actual: turn_count,
                },
            )?;
            self.tool_host.fire_session_hooks("budgetexceeded");
            return Ok(true);
        }
        if let Some(max_usd) = self.max_budget_usd {
            let accumulated_cost = self.store.total_session_cost(session_id).unwrap_or(0.0);
            if accumulated_cost >= max_usd {
                self.emit(
                    session_id,
                    EventKind::BudgetExceededV1 {
                        limit_usd: max_usd,
                        actual_usd: accumulated_cost,
                    },
                )?;
                self.tool_host.fire_session_hooks("budgetexceeded");
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn execute_run_once_step(
        &self,
        session_id: Uuid,
        step: &PlanStep,
        runtime_goal: &str,
        allow_tools: bool,
    ) -> Result<RunOnceStepOutcome> {
        let calls = self.calls_for_step(step, runtime_goal);
        let mut notes = Vec::new();
        let mut step_success = true;
        let mut proposals = Vec::new();
        for call in calls {
            let proposal = self.tool_host.propose(call);
            self.emit(
                session_id,
                EventKind::ToolProposedV1 {
                    proposal: proposal.clone(),
                },
            )?;
            proposals.push(proposal);
        }
        if proposals.is_empty() {
            step_success = false;
            notes.push("no executable tools for step".to_string());
        } else {
            for proposal in &proposals {
                if !(proposal.approved
                    || allow_tools
                    || self.request_tool_approval(&proposal.call).unwrap_or(false))
                {
                    step_success = false;
                    notes.push(format!("approval required for {}", proposal.call.name));
                    break;
                }
            }
        }
        if step_success {
            for proposal in &proposals {
                self.emit(
                    session_id,
                    EventKind::ToolApprovedV1 {
                        invocation_id: proposal.invocation_id,
                    },
                )?;
            }
            let proposal_outcome = self.execute_run_once_proposals(session_id, &proposals)?;
            step_success = proposal_outcome.success;
            notes.extend(proposal_outcome.notes);
        }
        Ok(RunOnceStepOutcome {
            success: step_success,
            notes: notes.join("\n"),
        })
    }

    fn execute_run_once_proposals(
        &self,
        session_id: Uuid,
        proposals: &[deepseek_core::ToolProposal],
    ) -> Result<RunOnceProposalOutcome> {
        if should_parallel_execute_calls(proposals) {
            self.execute_run_once_parallel_proposals(session_id, proposals)
        } else {
            self.execute_run_once_serial_proposals(session_id, proposals)
        }
    }

    fn execute_run_once_parallel_proposals(
        &self,
        session_id: Uuid,
        proposals: &[deepseek_core::ToolProposal],
    ) -> Result<RunOnceProposalOutcome> {
        let mut handles = Vec::new();
        for proposal in proposals {
            let tool_host = Arc::clone(&self.tool_host);
            let invocation_id = proposal.invocation_id;
            let call = proposal.call.clone();
            handles.push(thread::spawn(move || {
                let result = tool_host.execute(ApprovedToolCall {
                    invocation_id,
                    call: call.clone(),
                });
                (call.name, result)
            }));
        }
        let mut results = Vec::new();
        for handle in handles {
            let joined = handle
                .join()
                .map_err(|_| anyhow!("parallel tool execution thread panicked"))?;
            results.push(joined);
        }
        let mut notes = Vec::new();
        let mut success = true;
        for (call_name, result) in results {
            self.emit(
                session_id,
                EventKind::ToolResultV1 {
                    result: result.clone(),
                },
            )?;
            self.emit_patch_events_if_any(session_id, &call_name, &result)?;
            notes.push(format!("{call_name} => {}", result.output));
            if !result.success {
                success = false;
                break;
            }
        }
        Ok(RunOnceProposalOutcome { success, notes })
    }

    fn execute_run_once_serial_proposals(
        &self,
        session_id: Uuid,
        proposals: &[deepseek_core::ToolProposal],
    ) -> Result<RunOnceProposalOutcome> {
        let mut notes = Vec::new();
        let mut success = true;
        for proposal in proposals {
            let result = self.tool_host.execute(ApprovedToolCall {
                invocation_id: proposal.invocation_id,
                call: proposal.call.clone(),
            });
            self.emit(
                session_id,
                EventKind::ToolResultV1 {
                    result: result.clone(),
                },
            )?;
            self.emit_patch_events_if_any(session_id, &proposal.call.name, &result)?;
            notes.push(format!("{} => {}", proposal.call.name, result.output));
            if !result.success {
                success = false;
                break;
            }
        }
        Ok(RunOnceProposalOutcome { success, notes })
    }

    fn revise_run_once_plan(
        &self,
        session_id: Uuid,
        prompt: &str,
        plan: &Plan,
        failure_streak: u32,
        failure_detail: &str,
        non_urgent: bool,
    ) -> Result<Plan> {
        self.revise_plan_with_llm(
            session_id,
            prompt,
            plan,
            failure_streak,
            failure_detail,
            non_urgent,
        )
        .or_else(|_: anyhow::Error| -> Result<Plan> {
            let mut fallback = plan.clone();
            fallback.version += 1;
            fallback.steps.retain(|s| !s.done);
            fallback
                .risk_notes
                .push(format!("revision due to failure: {}", failure_detail));
            Ok(fallback)
        })
    }

    fn run_once_verification(
        &self,
        session: &mut Session,
        plan: &Plan,
        allow_tools: bool,
        execution_failed: bool,
    ) -> Result<u32> {
        if execution_failed {
            self.transition(session, SessionState::Failed)?;
            return Ok(0);
        }
        self.transition(session, SessionState::Verifying)?;
        let mut verification_failures = 0_u32;
        self.tool_host.fire_session_hooks("verificationstarted");
        for cmd in &plan.verification {
            let proposal = self.tool_host.propose(ToolCall {
                name: "bash.run".to_string(),
                args: json!({"cmd": cmd}),
                requires_approval: true,
            });
            let (ok, output) = if proposal.approved
                || allow_tools
                || self.request_tool_approval(&proposal.call).unwrap_or(false)
            {
                let result = self.tool_host.execute(ApprovedToolCall {
                    invocation_id: proposal.invocation_id,
                    call: proposal.call,
                });
                (result.success, result.output.to_string())
            } else {
                (false, "approval required".to_string())
            };

            self.emit(
                session.session_id,
                EventKind::VerificationRunV1 {
                    command: cmd.clone(),
                    success: ok,
                    output,
                },
            )?;
            if !ok {
                verification_failures += 1;
            }
        }
        self.tool_host.fire_session_hooks("verificationcompleted");
        Ok(verification_failures)
    }

    fn emit_run_once_verification_router(
        &self,
        session_id: Uuid,
        prompt: &str,
        failure_streak: u32,
        verification_failures: u32,
    ) -> Result<()> {
        if verification_failures == 0 {
            return Ok(());
        }
        let decision = self.router.select(
            LlmUnit::Planner,
            RouterSignals {
                prompt_complexity: (prompt.len() as f32 / 500.0).min(1.0),
                repo_breadth: 0.6,
                failure_streak: (failure_streak as f32 / 3.0).min(1.0),
                verification_failures: (verification_failures as f32 / 3.0).min(1.0),
                low_confidence: 0.5,
                ambiguity_flags: 0.3,
            },
        );
        self.emit(
            session_id,
            EventKind::RouterDecisionV1 {
                decision: decision.clone(),
            },
        )?;
        if decision.escalated {
            self.emit(
                session_id,
                EventKind::RouterEscalationV1 {
                    reason_codes: vec!["verification_failures".to_string()],
                },
            )?;
        }
        Ok(())
    }

    fn emit_run_once_telemetry(
        &self,
        session_id: Uuid,
        plan: &Plan,
        failure_streak: u32,
        verification_failures: u32,
    ) -> Result<()> {
        if !self.cfg.telemetry.enabled {
            return Ok(());
        }
        self.emit(
            session_id,
            EventKind::TelemetryEventV1 {
                name: "run_once".to_string(),
                properties: json!({
                    "failure_streak": failure_streak,
                    "verification_failures": verification_failures,
                    "plan_steps": plan.steps.len(),
                }),
            },
        )
    }

    fn finalize_run_once_session(
        &self,
        session: &mut Session,
        prompt: &str,
        plan: &Plan,
        failure_streak: u32,
        verification_failures: u32,
        execution_failed: bool,
    ) -> Result<bool> {
        let run_succeeded = !(execution_failed || verification_failures > 0);
        if run_succeeded {
            self.transition(session, SessionState::Completed)?;
            self.remember_successful_strategy(prompt, plan)?;
        } else {
            self.transition(session, SessionState::Failed)?;
            self.remember_failed_strategy(prompt, plan, failure_streak, verification_failures)?;
        }
        self.remember_objective_outcome(
            prompt,
            plan,
            failure_streak,
            verification_failures,
            run_succeeded,
        )?;
        Ok(run_succeeded)
    }

    fn build_run_once_summary(
        &self,
        prompt: &str,
        state: &RunOnceState,
        execution: &RunOnceExecution,
        verification_failures: u32,
        run_succeeded: bool,
    ) -> Result<String> {
        let projection = self.store.rebuild_from_events(state.session.session_id)?;
        let summary = format!(
            "session={} steps={} failures={} subagents={} router_models={:?} base_model={} max_model={}",
            state.session.session_id,
            projection.step_status.len(),
            execution.failure_streak + verification_failures,
            state.subagent_notes.len(),
            projection.router_models,
            self.cfg.llm.base_model,
            self.cfg.llm.max_think_model
        );
        if let Ok(manager) = MemoryManager::new(&self.workspace) {
            let mut patterns = Vec::new();
            patterns.push(format!(
                "steps={} verification_failures={} execution_failed={}",
                state.plan.steps.len(),
                verification_failures,
                execution.execution_failed
            ));
            if verification_failures > 0 {
                patterns.push("verification failures require focused plan revision".to_string());
            }
            if let Err(e) = manager.append_auto_memory_observation(AutoMemoryObservation {
                objective: prompt.to_string(),
                summary: summary.clone(),
                success: run_succeeded,
                patterns,
                recorded_at: None,
            }) {
                self.observer
                    .warn_log(&format!("memory: failed to persist observation: {e}"));
            }
        }
        Ok(summary)
    }

    /// Chat-with-tools loop (convenience wrapper with tools enabled).
    pub fn run_once_with_mode(
        &self,
        prompt: &str,
        allow_tools: bool,
        force_max_think: bool,
    ) -> Result<String> {
        self.run_once_with_mode_and_priority(prompt, allow_tools, force_max_think, false)
    }

    #[deprecated(note = "use chat() or chat_with_options() instead")]
    #[allow(deprecated)]
    pub fn run_once(&self, prompt: &str, allow_tools: bool) -> Result<String> {
        self.run_once_with_mode_and_priority(prompt, allow_tools, false, false)
    }

    #[deprecated(note = "use chat() or chat_with_options() instead")]
    #[allow(deprecated)]
    pub fn plan_only(&self, prompt: &str) -> Result<Plan> {
        self.plan_only_with_mode(prompt, false, false)
    }

    pub(crate) fn plan_only_with_mode(
        &self,
        prompt: &str,
        force_max_think: bool,
        non_urgent: bool,
    ) -> Result<Plan> {
        let mut session = self.ensure_session()?;
        self.transition(&mut session, SessionState::Planning)?;
        let planner_request = self.build_planner_request(session.session_id, prompt)?;
        let decision = self.select_planner_decision(prompt, force_max_think);
        self.emit(
            session.session_id,
            EventKind::RouterDecisionV1 {
                decision: decision.clone(),
            },
        )?;
        self.observer.record_router_decision(&decision)?;

        let (mut plan, mut planner_escalations) = self.generate_initial_plan_with_retry(
            session.session_id,
            prompt,
            non_urgent,
            &planner_request,
            &decision,
            session.budgets.max_think_tokens,
        )?;
        let objective_outcomes = self
            .load_matching_objective_outcomes(prompt, 6)
            .unwrap_or_default();
        let quality_retry_budget = self.cfg.router.max_escalations_per_unit as usize;
        let (updated_plan, quality_attempt, quality_repairs, escalations_after_quality) = self
            .apply_plan_quality_repairs(
                session.session_id,
                prompt,
                non_urgent,
                &decision,
                plan,
                planner_escalations,
                quality_retry_budget,
                &objective_outcomes,
                session.budgets.max_think_tokens,
            )?;
        plan = updated_plan;
        planner_escalations = escalations_after_quality;
        let verification_feedback = self
            .store
            .list_recent_verification_runs(session.session_id, 12)?
            .into_iter()
            .filter(|run| !run.success)
            .collect::<Vec<_>>();
        let (plan, feedback_attempt, feedback_repairs, _planner_escalations) = self
            .apply_plan_feedback_repairs(
                session.session_id,
                prompt,
                non_urgent,
                &decision,
                plan,
                planner_escalations,
                quality_retry_budget,
                &verification_feedback,
                session.budgets.max_think_tokens,
            )?;
        let mut plan = if let Some(plan) = plan {
            plan
        } else {
            default_planner_fallback_plan(prompt)
        };
        if !quality_repairs.is_empty() {
            plan.risk_notes.push(format!(
                "plan_quality_repairs={} issues={}",
                quality_attempt,
                quality_repairs.join(" | ")
            ));
        }
        if !feedback_repairs.is_empty() {
            plan.risk_notes.push(format!(
                "verification_feedback_repairs={} issues={}",
                feedback_attempt,
                feedback_repairs.join(" | ")
            ));
        }

        self.persist_created_plan(&mut session, &plan)?;
        Ok(plan)
    }

    fn build_planner_request(&self, session_id: Uuid, prompt: &str) -> Result<String> {
        let context_augmented_prompt = self.augment_prompt_context(session_id, prompt)?;
        let reference_expanded_prompt =
            expand_prompt_references(&self.workspace, &context_augmented_prompt, true)
                .unwrap_or_else(|_| context_augmented_prompt.clone());
        let redacted_prompt = self.policy.redact(&reference_expanded_prompt);
        Ok(build_planner_prompt(&redacted_prompt))
    }

    fn select_planner_decision(
        &self,
        prompt: &str,
        force_max_think: bool,
    ) -> deepseek_core::RouterDecision {
        let mut decision = self.router.select(
            LlmUnit::Planner,
            RouterSignals {
                prompt_complexity: (prompt.len() as f32 / 500.0).min(1.0),
                repo_breadth: 0.5,
                failure_streak: 0.0,
                verification_failures: 0.0,
                low_confidence: 0.2,
                ambiguity_flags: if prompt.contains('?') { 0.5 } else { 0.1 },
            },
        );
        if force_max_think {
            if !decision
                .reason_codes
                .iter()
                .any(|r| r == "autopilot_force_max_think")
            {
                decision
                    .reason_codes
                    .push("autopilot_force_max_think".to_string());
            }
            decision.selected_model = self.cfg.llm.max_think_model.clone();
            decision.escalated = true;
            decision.confidence = decision.confidence.max(0.95);
            decision.score = decision.score.max(self.cfg.router.threshold_high);
        } else if self.cfg.router.auto_max_think
            && !decision
                .selected_model
                .eq_ignore_ascii_case(&self.cfg.llm.max_think_model)
        {
            if !decision
                .reason_codes
                .iter()
                .any(|r| r == "planner_default_reasoner")
            {
                decision
                    .reason_codes
                    .push("planner_default_reasoner".to_string());
            }
            decision.selected_model = self.cfg.llm.max_think_model.clone();
            decision.escalated = true;
            decision.confidence = decision.confidence.max(0.9);
            decision.score = decision.score.max(self.cfg.router.threshold_high);
        }
        decision
    }

    fn generate_initial_plan_with_retry(
        &self,
        session_id: Uuid,
        prompt: &str,
        non_urgent: bool,
        planner_request: &str,
        decision: &deepseek_core::RouterDecision,
        max_tokens: u32,
    ) -> Result<(Option<Plan>, u8)> {
        let llm_response = self.complete_with_cache(
            session_id,
            &LlmRequest {
                unit: LlmUnit::Planner,
                prompt: planner_request.to_string(),
                model: decision.selected_model.clone(),
                max_tokens,
                non_urgent,
                images: vec![],
            },
        )?;
        self.emit(
            session_id,
            EventKind::UsageUpdatedV1 {
                unit: LlmUnit::Planner,
                model: decision.selected_model.clone(),
                input_tokens: estimate_tokens(planner_request),
                output_tokens: estimate_tokens(&llm_response.text),
            },
        )?;
        self.emit_cost_event(
            session_id,
            estimate_tokens(planner_request),
            estimate_tokens(&llm_response.text),
        )?;
        let mut plan = parse_plan_from_llm(&llm_response.text, prompt);
        let mut planner_escalations: u8 = u8::from(
            decision
                .selected_model
                .eq_ignore_ascii_case(&self.cfg.llm.max_think_model),
        );
        if plan.is_none()
            && self.cfg.router.auto_max_think
            && self.cfg.router.escalate_on_invalid_plan
            && self
                .router
                .should_escalate_retry(&LlmUnit::Planner, true, planner_escalations)
        {
            self.emit(
                session_id,
                EventKind::RouterEscalationV1 {
                    reason_codes: vec!["invalid_or_empty_plan".to_string()],
                },
            )?;
            let retry = self.complete_with_cache(
                session_id,
                &LlmRequest {
                    unit: LlmUnit::Planner,
                    prompt: planner_request.to_string(),
                    model: self.cfg.llm.max_think_model.clone(),
                    max_tokens,
                    non_urgent,
                    images: vec![],
                },
            )?;
            self.emit(
                session_id,
                EventKind::UsageUpdatedV1 {
                    unit: LlmUnit::Planner,
                    model: self.cfg.llm.max_think_model.clone(),
                    input_tokens: estimate_tokens(prompt),
                    output_tokens: estimate_tokens(&retry.text),
                },
            )?;
            self.emit_cost_event(
                session_id,
                estimate_tokens(prompt),
                estimate_tokens(&retry.text),
            )?;
            planner_escalations = planner_escalations.saturating_add(1);
            plan = parse_plan_from_llm(&retry.text, prompt);
        }
        Ok((plan, planner_escalations))
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_plan_quality_repairs(
        &self,
        session_id: Uuid,
        prompt: &str,
        non_urgent: bool,
        decision: &deepseek_core::RouterDecision,
        mut plan: Option<Plan>,
        mut planner_escalations: u8,
        quality_retry_budget: usize,
        objective_outcomes: &[ObjectiveOutcomeEntry],
        max_tokens: u32,
    ) -> Result<(Option<Plan>, usize, Vec<String>, u8)> {
        let mut quality_attempt = 0usize;
        let mut quality_repairs = Vec::new();
        while let Some(candidate) = plan.clone() {
            let report = combine_plan_quality_reports(
                assess_plan_quality(&candidate, prompt),
                assess_plan_long_horizon_quality(&candidate, prompt, objective_outcomes),
            );
            if report.acceptable {
                break;
            }
            quality_repairs = report.issues.clone();
            if quality_attempt >= quality_retry_budget {
                plan = None;
                break;
            }
            quality_attempt += 1;
            let use_max_think = self.cfg.router.auto_max_think
                && self
                    .router
                    .should_escalate_retry(&LlmUnit::Planner, true, planner_escalations);
            if use_max_think {
                self.emit(
                    session_id,
                    EventKind::RouterEscalationV1 {
                        reason_codes: vec![
                            "plan_quality_retry".to_string(),
                            format!("quality_score_{:.2}", report.score),
                        ],
                    },
                )?;
            }
            let repair_model = if use_max_think {
                planner_escalations = planner_escalations.saturating_add(1);
                self.cfg.llm.max_think_model.clone()
            } else {
                decision.selected_model.clone()
            };
            let repair_prompt = build_plan_quality_repair_prompt(prompt, &candidate, &report);
            let repaired = self.complete_with_cache(
                session_id,
                &LlmRequest {
                    unit: LlmUnit::Planner,
                    prompt: repair_prompt.clone(),
                    model: repair_model.clone(),
                    max_tokens,
                    non_urgent,
                    images: vec![],
                },
            )?;
            self.emit(
                session_id,
                EventKind::UsageUpdatedV1 {
                    unit: LlmUnit::Planner,
                    model: repair_model.clone(),
                    input_tokens: estimate_tokens(&repair_prompt),
                    output_tokens: estimate_tokens(&repaired.text),
                },
            )?;
            self.emit_cost_event(
                session_id,
                estimate_tokens(&repair_prompt),
                estimate_tokens(&repaired.text),
            )?;
            plan = parse_plan_from_llm(&repaired.text, prompt);
        }
        Ok((plan, quality_attempt, quality_repairs, planner_escalations))
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_plan_feedback_repairs(
        &self,
        session_id: Uuid,
        prompt: &str,
        non_urgent: bool,
        decision: &deepseek_core::RouterDecision,
        mut plan: Option<Plan>,
        mut planner_escalations: u8,
        quality_retry_budget: usize,
        verification_feedback: &[VerificationRunRecord],
        max_tokens: u32,
    ) -> Result<(Option<Plan>, usize, Vec<String>, u8)> {
        let mut feedback_attempt = 0usize;
        let mut feedback_repairs = Vec::new();
        while let Some(candidate) = plan.clone() {
            let report = assess_plan_feedback_alignment(&candidate, verification_feedback);
            if report.acceptable {
                break;
            }
            feedback_repairs = report.issues.clone();
            if feedback_attempt >= quality_retry_budget {
                plan = None;
                break;
            }
            feedback_attempt += 1;
            let use_max_think = self.cfg.router.auto_max_think
                && self
                    .router
                    .should_escalate_retry(&LlmUnit::Planner, true, planner_escalations);
            if use_max_think {
                self.emit(
                    session_id,
                    EventKind::RouterEscalationV1 {
                        reason_codes: vec![
                            "plan_feedback_retry".to_string(),
                            format!("feedback_score_{:.2}", report.score),
                        ],
                    },
                )?;
            }
            let repair_model = if use_max_think {
                planner_escalations = planner_escalations.saturating_add(1);
                self.cfg.llm.max_think_model.clone()
            } else {
                decision.selected_model.clone()
            };
            let repair_prompt = build_verification_feedback_repair_prompt(
                prompt,
                &candidate,
                &report,
                verification_feedback,
            );
            let repaired = self.complete_with_cache(
                session_id,
                &LlmRequest {
                    unit: LlmUnit::Planner,
                    prompt: repair_prompt.clone(),
                    model: repair_model.clone(),
                    max_tokens,
                    non_urgent,
                    images: vec![],
                },
            )?;
            self.emit(
                session_id,
                EventKind::UsageUpdatedV1 {
                    unit: LlmUnit::Planner,
                    model: repair_model.clone(),
                    input_tokens: estimate_tokens(&repair_prompt),
                    output_tokens: estimate_tokens(&repaired.text),
                },
            )?;
            self.emit_cost_event(
                session_id,
                estimate_tokens(&repair_prompt),
                estimate_tokens(&repaired.text),
            )?;
            plan = parse_plan_from_llm(&repaired.text, prompt);
        }
        Ok((
            plan,
            feedback_attempt,
            feedback_repairs,
            planner_escalations,
        ))
    }

    fn persist_created_plan(&self, session: &mut Session, plan: &Plan) -> Result<()> {
        session.active_plan_id = Some(plan.plan_id);
        self.store.save_session(session)?;
        self.store.save_plan(session.session_id, plan)?;
        self.emit(
            session.session_id,
            EventKind::PlanCreatedV1 { plan: plan.clone() },
        )?;
        self.tool_host.fire_session_hooks("plancreated");
        Ok(())
    }

    #[deprecated(note = "use chat() or chat_with_options() instead")]
    #[allow(deprecated)]
    pub fn resume(&self) -> Result<String> {
        let session = self
            .store
            .load_latest_session()?
            .ok_or_else(|| anyhow!("no session exists"))?;
        let projection = self.store.rebuild_from_events(session.session_id)?;
        Ok(format!(
            "resumed session={} state={:?} turns={} steps={}",
            session.session_id,
            projection.state,
            projection.transcript.len(),
            projection.step_status.len()
        ))
    }
}

#[derive(Debug)]
struct RunOnceState {
    session: Session,
    plan: Plan,
    runtime_goal: String,
    subagent_notes: Vec<String>,
}

#[derive(Debug)]
struct RunOnceExecution {
    failure_streak: u32,
    execution_failed: bool,
}

#[derive(Debug)]
struct RunOnceStepOutcome {
    success: bool,
    notes: String,
}

#[derive(Debug)]
struct RunOnceProposalOutcome {
    success: bool,
    notes: Vec<String>,
}

fn default_planner_fallback_plan(prompt: &str) -> Plan {
    Plan {
        plan_id: Uuid::now_v7(),
        version: 1,
        goal: prompt.to_string(),
        assumptions: vec!["Workspace is writable".to_string()],
        steps: vec![PlanStep {
            step_id: Uuid::now_v7(),
            title: "Analyze scope".to_string(),
            intent: "search".to_string(),
            tools: vec!["index.query".to_string(), "fs.grep".to_string()],
            files: vec![],
            done: false,
        }],
        verification: vec!["cargo test --workspace".to_string()],
        risk_notes: vec![],
    }
}
