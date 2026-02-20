use anyhow::{Result, anyhow};
use chrono::{Timelike, Utc};
use deepseek_core::{
    AppConfig, ApprovedToolCall, ChatMessage, ChatRequest, EventEnvelope, EventKind, ExecContext,
    Executor, Failure, LlmRequest, LlmUnit, ModelRouter, Plan, PlanContext, PlanStep, Planner,
    RouterSignals, Session, SessionBudgets, SessionState, StepOutcome, StreamChunk, ToolCall,
    ToolChoice, ToolHost, is_valid_session_state_transition,
};
use deepseek_llm::{DeepSeekClient, LlmClient};
use deepseek_memory::{AutoMemoryObservation, MemoryManager};
use deepseek_observe::Observer;
use deepseek_policy::PolicyEngine;
use deepseek_router::WeightedRouter;
use deepseek_store::{ProviderMetricRecord, Store, SubagentRunRecord, VerificationRunRecord};
use deepseek_subagent::{SubagentManager, SubagentRole, SubagentTask};
use deepseek_tools::{LocalToolHost, map_tool_name, tool_definitions};
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use uuid::Uuid;

pub struct SchemaPlanner;

impl Planner for SchemaPlanner {
    fn create_plan(&self, ctx: PlanContext) -> Result<Plan> {
        let prompt_lower = ctx.user_prompt.to_ascii_lowercase();
        let mut steps = vec![PlanStep {
            step_id: Uuid::now_v7(),
            title: "Analyze scope and locate relevant modules".to_string(),
            intent: "search".to_string(),
            tools: vec![
                "index.query".to_string(),
                "fs.grep".to_string(),
                "fs.read".to_string(),
            ],
            files: vec![],
            done: false,
        }];

        if prompt_lower.contains("git")
            || prompt_lower.contains("branch")
            || prompt_lower.contains("commit")
            || prompt_lower.contains("pr")
        {
            steps.push(PlanStep {
                step_id: Uuid::now_v7(),
                title: "Assess repository status and history".to_string(),
                intent: "git".to_string(),
                tools: vec!["git.status".to_string(), "git.diff".to_string()],
                files: vec![],
                done: false,
            });
        }

        if prompt_lower.contains("refactor")
            || prompt_lower.contains("implement")
            || prompt_lower.contains("fix")
            || prompt_lower.contains("change")
            || prompt_lower.contains("update")
            || prompt_lower.contains("patch")
        {
            steps.push(PlanStep {
                step_id: Uuid::now_v7(),
                title: "Implement code and config updates".to_string(),
                intent: "edit".to_string(),
                tools: vec![
                    "fs.edit".to_string(),
                    "patch.stage".to_string(),
                    "patch.apply".to_string(),
                ],
                files: vec![],
                done: false,
            });
        }

        if prompt_lower.contains("docs")
            || prompt_lower.contains("readme")
            || prompt_lower.contains("guide")
        {
            steps.push(PlanStep {
                step_id: Uuid::now_v7(),
                title: "Update docs".to_string(),
                intent: "docs".to_string(),
                tools: vec!["fs.edit".to_string(), "patch.stage".to_string()],
                files: vec!["README.md".to_string()],
                done: false,
            });
        }

        steps.push(PlanStep {
            step_id: Uuid::now_v7(),
            title: "Run verification and collect outcomes".to_string(),
            intent: "verify".to_string(),
            tools: vec!["bash.run".to_string()],
            files: vec![],
            done: false,
        });

        Ok(Plan {
            plan_id: Uuid::now_v7(),
            version: 1,
            goal: ctx.user_prompt,
            assumptions: vec![
                "Workspace is writable".to_string(),
                "DeepSeek API key is configured".to_string(),
            ],
            steps,
            verification: vec![
                "cargo fmt --all -- --check".to_string(),
                "cargo clippy --workspace --all-targets -- -D warnings".to_string(),
                "cargo test --workspace".to_string(),
            ],
            risk_notes: vec!["May require approval for patch apply and bash.run".to_string()],
        })
    }

    fn revise_plan(&self, _ctx: PlanContext, last_plan: &Plan, failure: Failure) -> Result<Plan> {
        let mut revised = last_plan.clone();
        revised.version += 1;
        revised.steps.retain(|step| !step.done);
        revised.steps.push(PlanStep {
            step_id: Uuid::now_v7(),
            title: "Recovery: inspect failure and apply targeted fix".to_string(),
            intent: "recover".to_string(),
            tools: vec![
                "fs.grep".to_string(),
                "fs.read".to_string(),
                "fs.edit".to_string(),
            ],
            files: vec![],
            done: false,
        });
        revised.risk_notes.push(format!(
            "revision due to failure: {} ({})",
            failure.summary, failure.detail
        ));
        Ok(revised)
    }
}

pub struct SimpleExecutor {
    tool_host: Arc<LocalToolHost>,
}

impl SimpleExecutor {
    pub fn new(tool_host: Arc<LocalToolHost>) -> Self {
        Self { tool_host }
    }
}

impl Executor for SimpleExecutor {
    fn run_step(&self, ctx: ExecContext, step: &PlanStep) -> Result<StepOutcome> {
        let call = match step.intent.as_str() {
            "search" => ToolCall {
                name: "fs.search_rg".to_string(),
                args: json!({"query": ctx.plan.goal, "limit": 10}),
                requires_approval: false,
            },
            "verify" => ToolCall {
                name: "bash.run".to_string(),
                args: json!({"cmd": "cargo test --workspace"}),
                requires_approval: true,
            },
            _ => ToolCall {
                name: "fs.list".to_string(),
                args: json!({"dir": "."}),
                requires_approval: false,
            },
        };

        let proposal = self.tool_host.propose(call);
        if !proposal.approved && !ctx.approved {
            return Ok(StepOutcome {
                step_id: step.step_id,
                success: false,
                notes: "approval required".to_string(),
            });
        }

        let result = self.tool_host.execute(ApprovedToolCall {
            invocation_id: proposal.invocation_id,
            call: proposal.call,
        });

        Ok(StepOutcome {
            step_id: step.step_id,
            success: result.success,
            notes: result.output.to_string(),
        })
    }
}

type ApprovalHandler = Box<dyn FnMut(&ToolCall) -> Result<bool> + Send>;

/// Options for `chat_with_options()`.
#[derive(Debug, Clone, Default)]
pub struct ChatOptions {
    /// Whether to include tool definitions and allow tool execution.
    pub tools: bool,
}

pub struct AgentEngine {
    workspace: PathBuf,
    store: Store,
    planner: SchemaPlanner,
    router: WeightedRouter,
    llm: Box<dyn LlmClient + Send + Sync>,
    observer: Observer,
    tool_host: Arc<LocalToolHost>,
    policy: PolicyEngine,
    cfg: AppConfig,
    subagents: SubagentManager,
    /// Optional callback invoked for each streaming token chunk.
    stream_callback: Mutex<Option<deepseek_core::StreamCallback>>,
    /// Maximum number of agent turns before stopping (CLI override).
    max_turns: Option<u64>,
    /// Maximum cost in USD before stopping (CLI override).
    max_budget_usd: Option<f64>,
    /// Optional external approval handler (e.g. crossterm-based for TUI mode).
    approval_handler: Mutex<Option<ApprovalHandler>>,
}

impl AgentEngine {
    pub fn new(workspace: &Path) -> Result<Self> {
        let cfg = AppConfig::ensure(workspace)?;
        let store = Store::new(workspace)?;
        let observer = Observer::new(workspace, &cfg.telemetry)?;
        let policy = PolicyEngine::from_app_config(&cfg.policy);
        let tool_host = Arc::new(LocalToolHost::new(workspace, policy.clone())?);
        let router = WeightedRouter::from_app_config(&cfg.router, &cfg.llm);
        let llm = Box::new(DeepSeekClient::new(cfg.llm.clone())?);

        let max_turns = cfg.budgets.max_turns;
        let max_budget_usd = cfg.budgets.max_budget_usd;
        Ok(Self {
            workspace: workspace.to_path_buf(),
            store,
            planner: SchemaPlanner,
            router,
            llm,
            observer,
            tool_host,
            policy,
            cfg,
            subagents: SubagentManager::default(),
            stream_callback: Mutex::new(None),
            max_turns,
            max_budget_usd,
            approval_handler: Mutex::new(None),
        })
    }

    /// Override max turns limit (from CLI flag).
    pub fn set_max_turns(&mut self, max: Option<u64>) {
        self.max_turns = max;
    }

    /// Override max budget USD limit (from CLI flag).
    pub fn set_max_budget_usd(&mut self, max: Option<f64>) {
        self.max_budget_usd = max;
    }

    /// Set an external approval handler (e.g. for TUI/raw-mode compatible input).
    pub fn set_approval_handler(&self, handler: ApprovalHandler) {
        if let Ok(mut guard) = self.approval_handler.lock() {
            *guard = Some(handler);
        }
    }

    /// Set a streaming callback that will be invoked for each token chunk
    /// during LLM completions.
    pub fn set_stream_callback(&self, cb: deepseek_core::StreamCallback) {
        if let Ok(mut guard) = self.stream_callback.lock() {
            *guard = Some(cb);
        }
    }

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

    pub fn plan_only(&self, prompt: &str) -> Result<Plan> {
        self.plan_only_with_mode(prompt, false, false)
    }

    fn plan_only_with_mode(
        &self,
        prompt: &str,
        force_max_think: bool,
        non_urgent: bool,
    ) -> Result<Plan> {
        let mut session = self.ensure_session()?;
        self.transition(&mut session, SessionState::Planning)?;
        let context_augmented_prompt = self.augment_prompt_context(session.session_id, prompt)?;
        let reference_expanded_prompt =
            expand_prompt_references(&self.workspace, &context_augmented_prompt, true)
                .unwrap_or_else(|_| context_augmented_prompt.clone());
        let redacted_prompt = self.policy.redact(&reference_expanded_prompt);
        let planner_request = build_planner_prompt(&redacted_prompt);

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
        }
        self.emit(
            session.session_id,
            EventKind::RouterDecisionV1 {
                decision: decision.clone(),
            },
        )?;
        self.observer.record_router_decision(&decision)?;

        let llm_response = self.complete_with_cache(
            session.session_id,
            &LlmRequest {
                unit: LlmUnit::Planner,
                prompt: planner_request.clone(),
                model: decision.selected_model.clone(),
                max_tokens: session.budgets.max_think_tokens,
                non_urgent,
                images: vec![],
            },
        )?;
        self.emit(
            session.session_id,
            EventKind::UsageUpdatedV1 {
                unit: LlmUnit::Planner,
                model: decision.selected_model.clone(),
                input_tokens: estimate_tokens(&planner_request),
                output_tokens: estimate_tokens(&llm_response.text),
            },
        )?;
        self.emit_cost_event(
            session.session_id,
            estimate_tokens(&planner_request),
            estimate_tokens(&llm_response.text),
        )?;
        let mut plan = parse_plan_from_llm(&llm_response.text, prompt);
        if plan.is_none()
            && self.cfg.router.auto_max_think
            && self
                .router
                .should_escalate_retry(&LlmUnit::Planner, true, 0)
        {
            self.emit(
                session.session_id,
                EventKind::RouterEscalationV1 {
                    reason_codes: vec!["invalid_or_empty_plan".to_string()],
                },
            )?;
            let retry = self.complete_with_cache(
                session.session_id,
                &LlmRequest {
                    unit: LlmUnit::Planner,
                    prompt: planner_request,
                    model: self.cfg.llm.max_think_model.clone(),
                    max_tokens: session.budgets.max_think_tokens,
                    non_urgent,
                    images: vec![],
                },
            )?;
            self.emit(
                session.session_id,
                EventKind::UsageUpdatedV1 {
                    unit: LlmUnit::Planner,
                    model: self.cfg.llm.max_think_model.clone(),
                    input_tokens: estimate_tokens(prompt),
                    output_tokens: estimate_tokens(&retry.text),
                },
            )?;
            self.emit_cost_event(
                session.session_id,
                estimate_tokens(prompt),
                estimate_tokens(&retry.text),
            )?;
            plan = parse_plan_from_llm(&retry.text, prompt);
        }
        let objective_outcomes = self
            .load_matching_objective_outcomes(prompt, 6)
            .unwrap_or_default();
        let quality_retry_budget = self.cfg.router.max_escalations_per_unit.max(1) as usize;
        let mut quality_attempt = 0usize;
        let mut quality_repairs = Vec::new();
        while let Some(candidate) = plan.clone() {
            let report = combine_plan_quality_reports(
                assess_plan_quality(&candidate, prompt),
                assess_plan_long_horizon_quality(&candidate, prompt, &objective_outcomes),
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
            self.emit(
                session.session_id,
                EventKind::RouterEscalationV1 {
                    reason_codes: vec![
                        "plan_quality_retry".to_string(),
                        format!("quality_score_{:.2}", report.score),
                    ],
                },
            )?;
            let repair_prompt = build_plan_quality_repair_prompt(prompt, &candidate, &report);
            let repaired = self.complete_with_cache(
                session.session_id,
                &LlmRequest {
                    unit: LlmUnit::Planner,
                    prompt: repair_prompt.clone(),
                    model: self.cfg.llm.max_think_model.clone(),
                    max_tokens: session.budgets.max_think_tokens,
                    non_urgent,
                    images: vec![],
                },
            )?;
            self.emit(
                session.session_id,
                EventKind::UsageUpdatedV1 {
                    unit: LlmUnit::Planner,
                    model: self.cfg.llm.max_think_model.clone(),
                    input_tokens: estimate_tokens(&repair_prompt),
                    output_tokens: estimate_tokens(&repaired.text),
                },
            )?;
            self.emit_cost_event(
                session.session_id,
                estimate_tokens(&repair_prompt),
                estimate_tokens(&repaired.text),
            )?;
            plan = parse_plan_from_llm(&repaired.text, prompt);
        }
        let verification_feedback = self
            .store
            .list_recent_verification_runs(session.session_id, 12)?
            .into_iter()
            .filter(|run| !run.success)
            .collect::<Vec<_>>();
        let mut feedback_attempt = 0usize;
        let mut feedback_repairs = Vec::new();
        while let Some(candidate) = plan.clone() {
            let report = assess_plan_feedback_alignment(&candidate, &verification_feedback);
            if report.acceptable {
                break;
            }
            feedback_repairs = report.issues.clone();
            if feedback_attempt >= quality_retry_budget {
                plan = None;
                break;
            }
            feedback_attempt += 1;
            self.emit(
                session.session_id,
                EventKind::RouterEscalationV1 {
                    reason_codes: vec![
                        "plan_feedback_retry".to_string(),
                        format!("feedback_score_{:.2}", report.score),
                    ],
                },
            )?;
            let repair_prompt = build_verification_feedback_repair_prompt(
                prompt,
                &candidate,
                &report,
                &verification_feedback,
            );
            let repaired = self.complete_with_cache(
                session.session_id,
                &LlmRequest {
                    unit: LlmUnit::Planner,
                    prompt: repair_prompt.clone(),
                    model: self.cfg.llm.max_think_model.clone(),
                    max_tokens: session.budgets.max_think_tokens,
                    non_urgent,
                    images: vec![],
                },
            )?;
            self.emit(
                session.session_id,
                EventKind::UsageUpdatedV1 {
                    unit: LlmUnit::Planner,
                    model: self.cfg.llm.max_think_model.clone(),
                    input_tokens: estimate_tokens(&repair_prompt),
                    output_tokens: estimate_tokens(&repaired.text),
                },
            )?;
            self.emit_cost_event(
                session.session_id,
                estimate_tokens(&repair_prompt),
                estimate_tokens(&repaired.text),
            )?;
            plan = parse_plan_from_llm(&repaired.text, prompt);
        }
        let mut plan = if let Some(plan) = plan {
            plan
        } else {
            self.planner.create_plan(PlanContext {
                session: session.clone(),
                user_prompt: prompt.to_string(),
                prior_failures: vec![],
            })?
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

        session.active_plan_id = Some(plan.plan_id);
        self.store.save_session(&session)?;
        self.store.save_plan(session.session_id, &plan)?;
        self.emit(
            session.session_id,
            EventKind::PlanCreatedV1 { plan: plan.clone() },
        )?;
        self.tool_host.fire_session_hooks("plancreated");

        Ok(plan)
    }

    #[deprecated(note = "use chat() or chat_with_options() instead")]
    #[allow(deprecated)]
    pub fn run_once(&self, prompt: &str, allow_tools: bool) -> Result<String> {
        self.run_once_with_mode_and_priority(prompt, allow_tools, false, false)
    }

    #[deprecated(note = "use chat() or chat_with_options() instead")]
    #[allow(deprecated)]
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
    pub fn run_once_with_mode_and_priority(
        &self,
        prompt: &str,
        allow_tools: bool,
        force_max_think: bool,
        non_urgent: bool,
    ) -> Result<String> {
        let mut session = self.ensure_session()?;

        // Fire "sessionstart" hooks (spec 2.9 extended hooks)
        self.tool_host.fire_session_hooks("sessionstart");

        self.emit(
            session.session_id,
            EventKind::TurnAddedV1 {
                role: "user".to_string(),
                content: prompt.to_string(),
            },
        )?;

        let mut plan = self.plan_only_with_mode(prompt, force_max_think, non_urgent)?;
        session = self.ensure_session()?;
        self.transition(&mut session, SessionState::ExecutingStep)?;
        let subagent_notes = self.run_subagents(session.session_id, &plan)?;
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
        let mut failure_streak = 0_u32;
        let mut execution_failed = false;
        let mut step_cursor = 0usize;
        let mut revision_budget = self.cfg.router.max_escalations_per_unit.max(1) as usize;
        let mut turn_count: u64 = 0;

        while step_cursor < plan.steps.len() {
            // Turn limit enforcement
            turn_count += 1;
            if let Some(max) = self.max_turns
                && turn_count > max
            {
                self.emit(
                    session.session_id,
                    EventKind::TurnLimitExceededV1 {
                        limit: max,
                        actual: turn_count,
                    },
                )?;
                self.tool_host.fire_session_hooks("budgetexceeded");
                execution_failed = true;
                break;
            }
            // Budget limit enforcement
            if let Some(max_usd) = self.max_budget_usd {
                let accumulated_cost = self
                    .store
                    .total_session_cost(session.session_id)
                    .unwrap_or(0.0);
                if accumulated_cost >= max_usd {
                    self.emit(
                        session.session_id,
                        EventKind::BudgetExceededV1 {
                            limit_usd: max_usd,
                            actual_usd: accumulated_cost,
                        },
                    )?;
                    self.tool_host.fire_session_hooks("budgetexceeded");
                    execution_failed = true;
                    break;
                }
            }
            let step = plan.steps[step_cursor].clone();
            let calls = self.calls_for_step(&step, &runtime_goal);
            let mut notes = Vec::new();
            let mut step_success = true;
            let mut proposals = Vec::new();
            for call in calls {
                let proposal = self.tool_host.propose(call);
                self.emit(
                    session.session_id,
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
                        session.session_id,
                        EventKind::ToolApprovedV1 {
                            invocation_id: proposal.invocation_id,
                        },
                    )?;
                }

                if should_parallel_execute_calls(&proposals) {
                    let mut handles = Vec::new();
                    for proposal in &proposals {
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
                    for (call_name, result) in results {
                        self.emit(
                            session.session_id,
                            EventKind::ToolResultV1 {
                                result: result.clone(),
                            },
                        )?;
                        self.emit_patch_events_if_any(session.session_id, &call_name, &result)?;
                        notes.push(format!("{call_name} => {}", result.output));
                        if !result.success {
                            step_success = false;
                            break;
                        }
                    }
                } else {
                    for proposal in &proposals {
                        let result = self.tool_host.execute(ApprovedToolCall {
                            invocation_id: proposal.invocation_id,
                            call: proposal.call.clone(),
                        });
                        self.emit(
                            session.session_id,
                            EventKind::ToolResultV1 {
                                result: result.clone(),
                            },
                        )?;
                        self.emit_patch_events_if_any(
                            session.session_id,
                            &proposal.call.name,
                            &result,
                        )?;
                        notes.push(format!("{} => {}", proposal.call.name, result.output));
                        if !result.success {
                            step_success = false;
                            break;
                        }
                    }
                }
            }

            let outcome = StepOutcome {
                step_id: step.step_id,
                success: step_success,
                notes: notes.join("\n"),
            };

            plan.steps[step_cursor].done = outcome.success;
            self.emit(
                session.session_id,
                EventKind::StepMarkedV1 {
                    step_id: step.step_id,
                    done: outcome.success,
                    note: outcome.notes.clone(),
                },
            )?;

            if !outcome.success {
                failure_streak += 1;
                if revision_budget == 0 {
                    execution_failed = true;
                    break;
                }
                revision_budget = revision_budget.saturating_sub(1);
                let failure_detail = outcome.notes.clone();
                let revised = self
                    .revise_plan_with_llm(
                        session.session_id,
                        prompt,
                        &plan,
                        failure_streak,
                        &failure_detail,
                        non_urgent,
                    )
                    .or_else(|_| {
                        self.planner.revise_plan(
                            PlanContext {
                                session: session.clone(),
                                user_prompt: prompt.to_string(),
                                prior_failures: vec![],
                            },
                            &plan,
                            Failure {
                                summary: "step failed".to_string(),
                                detail: failure_detail.clone(),
                            },
                        )
                    })?;
                self.emit(
                    session.session_id,
                    EventKind::PlanRevisedV1 {
                        plan: revised.clone(),
                    },
                )?;
                self.tool_host.fire_session_hooks("planrevised");
                plan = revised;
                step_cursor = 0;
                continue;
            }
            step_cursor += 1;
        }

        if execution_failed {
            self.transition(&mut session, SessionState::Failed)?;
        } else {
            self.transition(&mut session, SessionState::Verifying)?;
        }
        let mut verification_failures = 0_u32;
        if !execution_failed {
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
        }

        if verification_failures > 0 {
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
                session.session_id,
                EventKind::RouterDecisionV1 {
                    decision: decision.clone(),
                },
            )?;
            if decision.escalated {
                self.emit(
                    session.session_id,
                    EventKind::RouterEscalationV1 {
                        reason_codes: vec!["verification_failures".to_string()],
                    },
                )?;
            }
        }

        if self.cfg.telemetry.enabled {
            self.emit(
                session.session_id,
                EventKind::TelemetryEventV1 {
                    name: "run_once".to_string(),
                    properties: json!({
                        "failure_streak": failure_streak,
                        "verification_failures": verification_failures,
                        "plan_steps": plan.steps.len(),
                    }),
                },
            )?;
        }

        let run_succeeded = !(execution_failed || verification_failures > 0);
        if !run_succeeded {
            self.transition(&mut session, SessionState::Failed)?;
            self.remember_failed_strategy(prompt, &plan, failure_streak, verification_failures)?;
        } else {
            self.transition(&mut session, SessionState::Completed)?;
            self.remember_successful_strategy(prompt, &plan)?;
        }
        self.remember_objective_outcome(
            prompt,
            &plan,
            failure_streak,
            verification_failures,
            run_succeeded,
        )?;

        // Fire "stop" hooks at agent completion (spec 2.9)
        self.tool_host.fire_stop_hooks();

        let projection = self.store.rebuild_from_events(session.session_id)?;
        let summary = format!(
            "session={} steps={} failures={} subagents={} router_models={:?} base_model={} max_model={}",
            session.session_id,
            projection.step_status.len(),
            failure_streak + verification_failures,
            subagent_notes.len(),
            projection.router_models,
            self.cfg.llm.base_model,
            self.cfg.llm.max_think_model
        );
        if let Ok(manager) = MemoryManager::new(&self.workspace) {
            let mut patterns = Vec::new();
            patterns.push(format!(
                "steps={} verification_failures={} execution_failed={}",
                plan.steps.len(),
                verification_failures,
                execution_failed
            ));
            if verification_failures > 0 {
                patterns.push("verification failures require focused plan revision".to_string());
            }
            let _ = manager.append_auto_memory_observation(AutoMemoryObservation {
                objective: prompt.to_string(),
                summary: summary.clone(),
                success: run_succeeded,
                patterns,
                recorded_at: None,
            });
        }
        Ok(summary)
    }

    /// Chat-with-tools loop (convenience wrapper with tools enabled).
    pub fn chat(&self, prompt: &str) -> Result<String> {
        self.chat_with_options(prompt, ChatOptions { tools: true })
    }

    /// Chat loop with configurable options. When `options.tools` is false, no tool
    /// definitions are sent and the model produces a single text response.
    pub fn chat_with_options(&self, prompt: &str, options: ChatOptions) -> Result<String> {
        let mut session = self.ensure_session()?;

        self.tool_host.fire_session_hooks("sessionstart");
        self.transition(&mut session, SessionState::Planning)?;

        self.emit(
            session.session_id,
            EventKind::TurnAddedV1 {
                role: "user".to_string(),
                content: prompt.to_string(),
            },
        )?;

        // Build system prompt with workspace context
        let system_prompt = self.build_chat_system_prompt(prompt)?;
        let tools = if options.tools {
            tool_definitions()
        } else {
            vec![]
        };

        // Initialize conversation with system + user message
        let mut messages: Vec<ChatMessage> = vec![ChatMessage::System {
            content: system_prompt,
        }];

        // Load prior conversation turns if resuming an existing session
        let projection = self.store.rebuild_from_events(session.session_id)?;
        if projection.transcript.len() > 1 {
            // Session has prior turns — restore them as messages.
            // Tool results are included as user context summaries since we
            // cannot reconstruct exact tool_call IDs for the API format.
            let mut pending_tool_summaries: Vec<String> = Vec::new();
            for entry in &projection.transcript {
                if let Some(content) = entry.strip_prefix("user: ") {
                    // Flush any pending tool summaries before the next user message
                    if !pending_tool_summaries.is_empty() {
                        messages.push(ChatMessage::User {
                            content: format!(
                                "[Prior tool results]\n{}",
                                pending_tool_summaries.join("\n")
                            ),
                        });
                        pending_tool_summaries.clear();
                    }
                    messages.push(ChatMessage::User {
                        content: content.to_string(),
                    });
                } else if let Some(content) = entry.strip_prefix("assistant: ") {
                    // Flush any pending tool summaries before the next assistant message
                    if !pending_tool_summaries.is_empty() {
                        messages.push(ChatMessage::User {
                            content: format!(
                                "[Prior tool results]\n{}",
                                pending_tool_summaries.join("\n")
                            ),
                        });
                        pending_tool_summaries.clear();
                    }
                    messages.push(ChatMessage::Assistant {
                        content: Some(content.to_string()),
                        tool_calls: vec![],
                    });
                } else if let Some(content) = entry.strip_prefix("tool: ") {
                    pending_tool_summaries.push(content.to_string());
                }
            }
            // Flush remaining tool summaries
            if !pending_tool_summaries.is_empty() {
                messages.push(ChatMessage::User {
                    content: format!(
                        "[Prior tool results]\n{}",
                        pending_tool_summaries.join("\n")
                    ),
                });
            }
        }

        // Add the current user message
        messages.push(ChatMessage::User {
            content: prompt.to_string(),
        });

        let max_turns = self.max_turns.unwrap_or(200);
        let mut turn_count: u64 = 0;
        let mut failure_streak: u32 = 0;
        let mut empty_response_count: u32 = 0;
        self.transition(&mut session, SessionState::ExecutingStep)?;

        loop {
            turn_count += 1;
            if turn_count > max_turns {
                let _ = self.transition(&mut session, SessionState::Failed);
                if let Ok(manager) = MemoryManager::new(&self.workspace) {
                    let _ = manager.append_auto_memory_observation(AutoMemoryObservation {
                        objective: prompt.to_string(),
                        summary: format!(
                            "chat failed: max turn limit ({}) reached, failure_streak={}",
                            max_turns, failure_streak
                        ),
                        success: false,
                        patterns: vec![
                            format!("turns={max_turns} failure_streak={failure_streak}"),
                            "max turn limit suggests task is too complex for single session"
                                .to_string(),
                        ],
                        recorded_at: None,
                    });
                }
                return Err(anyhow!(
                    "Reached maximum turn limit ({}). The task may be too complex for a single session.",
                    max_turns
                ));
            }

            // Budget check
            if let Some(max_usd) = self.max_budget_usd {
                let cost = self
                    .store
                    .total_session_cost(session.session_id)
                    .unwrap_or(0.0);
                if cost >= max_usd {
                    let _ = self.transition(&mut session, SessionState::Failed);
                    if let Ok(manager) = MemoryManager::new(&self.workspace) {
                        let _ = manager.append_auto_memory_observation(AutoMemoryObservation {
                            objective: prompt.to_string(),
                            summary: format!(
                                "chat failed: budget limit (${:.2}/${:.2}) at turn {}, failure_streak={}",
                                cost, max_usd, turn_count, failure_streak
                            ),
                            success: false,
                            patterns: vec![format!(
                                "turns={turn_count} failure_streak={failure_streak} cost=${cost:.2}"
                            )],
                            recorded_at: None,
                        });
                    }
                    return Err(anyhow!(
                        "Budget limit reached (${:.2} / ${:.2})",
                        cost,
                        max_usd
                    ));
                }
            }

            // Context window compaction
            let token_count = estimate_messages_tokens(&messages);
            let threshold = (self.cfg.llm.context_window_tokens as f64
                * self.cfg.context.auto_compact_threshold.clamp(0.1, 1.0) as f64)
                as u64;
            if token_count > threshold && messages.len() > 4 {
                // Keep system prompt (index 0) + last 6 messages
                let keep_tail = 6.min(messages.len() - 1);
                let compacted_range = &messages[1..messages.len() - keep_tail];
                let summary = summarize_chat_messages(compacted_range);
                let from_turn = 1u64;
                let to_turn = compacted_range.len() as u64;
                let summary_id = Uuid::now_v7();

                let mut new_messages = vec![messages[0].clone()];
                new_messages.push(ChatMessage::User {
                    content: format!("[Context compacted — prior conversation summary]\n{summary}"),
                });
                new_messages.extend_from_slice(&messages[messages.len() - keep_tail..]);
                let new_token_count = estimate_messages_tokens(&new_messages);

                self.emit(
                    session.session_id,
                    EventKind::ContextCompactedV1 {
                        summary_id,
                        from_turn,
                        to_turn,
                        token_delta_estimate: token_count as i64 - new_token_count as i64,
                        replay_pointer: String::new(),
                    },
                )?;

                messages = new_messages;
            }

            // Route model selection based on complexity signals
            let verification_failure_count = self
                .store
                .list_recent_verification_runs(session.session_id, 10)
                .unwrap_or_default()
                .iter()
                .filter(|r| !r.success)
                .count() as f32;
            let conversation_depth = (messages.len() as f32 / 20.0).min(1.0);
            let ambiguity = if prompt.contains('?')
                || prompt.contains(" or ")
                || prompt.contains("which ")
                || prompt.contains("should ")
            {
                0.5
            } else {
                0.0
            };
            let signals = RouterSignals {
                prompt_complexity: (prompt.len() as f32 / 500.0).min(1.0),
                repo_breadth: conversation_depth,
                failure_streak: (failure_streak as f32 / 3.0).min(1.0),
                verification_failures: (verification_failure_count / 3.0).min(1.0),
                low_confidence: if empty_response_count > 0 { 0.6 } else { 0.2 },
                ambiguity_flags: ambiguity,
            };
            let decision = self.router.select(LlmUnit::Executor, signals);
            self.emit(
                session.session_id,
                EventKind::RouterDecisionV1 {
                    decision: decision.clone(),
                },
            )?;

            let request = ChatRequest {
                model: decision.selected_model.clone(),
                messages: messages.clone(),
                tools: tools.clone(),
                tool_choice: if options.tools {
                    ToolChoice::auto()
                } else {
                    ToolChoice::none()
                },
                max_tokens: session.budgets.max_think_tokens.max(4096),
                temperature: Some(0.0),
            };

            // Call the LLM with streaming (clone the Arc so it persists across turns)
            let response = {
                let cb = self.stream_callback.lock().ok().and_then(|g| g.clone());
                if let Some(cb) = cb {
                    self.llm.complete_chat_streaming(&request, cb)?
                } else {
                    self.llm.complete_chat(&request)?
                }
            };

            // Escalation retry: if base model returned empty response and we weren't
            // already using the reasoner, retry with the max_think model once.
            let response =
                if response.text.is_empty() && response.tool_calls.is_empty() && turn_count <= 1 {
                    let max_model = &self.cfg.llm.max_think_model;
                    if decision.selected_model != *max_model {
                        self.emit(
                            session.session_id,
                            EventKind::RouterEscalationV1 {
                                reason_codes: vec!["empty_response_escalation".to_string()],
                            },
                        )?;
                        let escalated_request = ChatRequest {
                            model: max_model.clone(),
                            ..request
                        };
                        let cb = self.stream_callback.lock().ok().and_then(|g| g.clone());
                        if let Some(cb) = cb {
                            self.llm.complete_chat_streaming(&escalated_request, cb)?
                        } else {
                            self.llm.complete_chat(&escalated_request)?
                        }
                    } else {
                        response
                    }
                } else {
                    response
                };

            // Record usage metrics
            let input_tok = estimate_tokens(
                &messages
                    .iter()
                    .map(|m| match m {
                        ChatMessage::System { content } | ChatMessage::User { content } => {
                            content.as_str()
                        }
                        ChatMessage::Assistant { content, .. } => content.as_deref().unwrap_or(""),
                        ChatMessage::Tool { content, .. } => content.as_str(),
                    })
                    .collect::<Vec<_>>()
                    .join(""),
            );
            let output_tok =
                estimate_tokens(&response.text) + estimate_tokens(&response.reasoning_content);
            self.emit(
                session.session_id,
                EventKind::UsageUpdatedV1 {
                    unit: LlmUnit::Executor,
                    model: decision.selected_model.clone(),
                    input_tokens: input_tok,
                    output_tokens: output_tok,
                },
            )?;
            self.emit_cost_event(session.session_id, input_tok, output_tok)?;

            // If the model returned text content with no tool calls, we're done
            if response.tool_calls.is_empty() {
                let answer = if !response.text.is_empty() {
                    response.text.clone()
                } else if !response.reasoning_content.is_empty() {
                    response.reasoning_content.clone()
                } else {
                    "(No response generated)".to_string()
                };

                self.emit(
                    session.session_id,
                    EventKind::TurnAddedV1 {
                        role: "assistant".to_string(),
                        content: answer.clone(),
                    },
                )?;

                self.tool_host.fire_stop_hooks();
                let _ = self.transition(&mut session, SessionState::Completed);

                // Persist memory observation for future sessions
                if let Ok(manager) = MemoryManager::new(&self.workspace) {
                    let mut patterns = vec![format!(
                        "turns={turn_count} failure_streak={failure_streak}"
                    )];
                    if failure_streak > 0 {
                        patterns.push(
                            "some tool failures occurred but task completed successfully"
                                .to_string(),
                        );
                    }
                    let _ = manager.append_auto_memory_observation(AutoMemoryObservation {
                        objective: prompt.to_string(),
                        summary: format!(
                            "chat completed in {} turns, failure_streak={}",
                            turn_count, failure_streak
                        ),
                        success: true,
                        patterns,
                        recorded_at: None,
                    });
                }

                return Ok(answer);
            }

            // The model wants to call tools — add the assistant message with tool_calls
            if response.text.is_empty() {
                empty_response_count += 1;
            }
            messages.push(ChatMessage::Assistant {
                content: if response.text.is_empty() {
                    None
                } else {
                    Some(response.text.clone())
                },
                tool_calls: response.tool_calls.clone(),
            });

            if !response.text.is_empty() {
                self.emit(
                    session.session_id,
                    EventKind::TurnAddedV1 {
                        role: "assistant".to_string(),
                        content: response.text.clone(),
                    },
                )?;
            }

            // Execute each tool call and collect results
            for tc in &response.tool_calls {
                let internal_name = map_tool_name(&tc.name);
                let args: serde_json::Value =
                    serde_json::from_str(&tc.arguments).unwrap_or_else(|_| json!({}));

                let tool_call = ToolCall {
                    name: internal_name.to_string(),
                    args: args.clone(),
                    requires_approval: false,
                };

                // Emit tool proposal event
                let proposal = self.tool_host.propose(tool_call.clone());
                self.emit(
                    session.session_id,
                    EventKind::ToolProposedV1 {
                        proposal: proposal.clone(),
                    },
                )?;

                // Check approval
                let approved = proposal.approved
                    || self.request_tool_approval(&proposal.call).unwrap_or(false);

                let tool_arg_summary = summarize_tool_args(internal_name, &args);
                let tool_result = if approved {
                    self.emit(
                        session.session_id,
                        EventKind::ToolApprovedV1 {
                            invocation_id: proposal.invocation_id,
                        },
                    )?;

                    // Notify TUI that a tool is being executed
                    if let Ok(mut cb_guard) = self.stream_callback.lock()
                        && let Some(ref mut cb) = *cb_guard
                    {
                        let detail = if tool_arg_summary.is_empty() {
                            String::new()
                        } else {
                            format!(" {tool_arg_summary}")
                        };
                        cb(StreamChunk::ContentDelta(format!(
                            "\n[tool: {}]{detail}\n",
                            internal_name
                        )));
                    }

                    let tool_start = Instant::now();
                    let result = self.tool_host.execute(ApprovedToolCall {
                        invocation_id: proposal.invocation_id,
                        call: proposal.call,
                    });
                    let tool_elapsed = tool_start.elapsed();

                    self.emit(
                        session.session_id,
                        EventKind::ToolResultV1 {
                            result: result.clone(),
                        },
                    )?;

                    // Notify TUI of completion with result preview
                    if let Ok(mut cb_guard) = self.stream_callback.lock()
                        && let Some(ref mut cb) = *cb_guard
                    {
                        let status_label = if result.success { "ok" } else { "error" };
                        let output_str = result.output.to_string();
                        let preview = if output_str.len() > 200 {
                            format!("{}...", &output_str[..output_str.floor_char_boundary(200)])
                        } else {
                            output_str
                        };
                        let preview_line = preview
                            .lines()
                            .next()
                            .unwrap_or("")
                            .chars()
                            .take(120)
                            .collect::<String>();
                        cb(StreamChunk::ContentDelta(format!(
                            "[tool: {}] {status_label} ({:.1}s) {preview_line}\n",
                            internal_name,
                            tool_elapsed.as_secs_f64()
                        )));
                    }

                    if result.success {
                        failure_streak = 0;
                    } else {
                        failure_streak += 1;
                    }

                    truncate_tool_output(&result.output.to_string(), 30000)
                } else {
                    failure_streak += 1;
                    // Notify TUI of denial
                    if let Ok(mut cb_guard) = self.stream_callback.lock()
                        && let Some(ref mut cb) = *cb_guard
                    {
                        cb(StreamChunk::ContentDelta(format!(
                            "\n[tool: {}] denied (requires approval)\n",
                            internal_name
                        )));
                    }
                    format!(
                        "Tool '{}' requires approval. Please grant permission to proceed.",
                        internal_name
                    )
                };

                // Add tool result message to conversation
                messages.push(ChatMessage::Tool {
                    tool_call_id: tc.id.clone(),
                    content: tool_result.clone(),
                });

                // Emit tool interaction as transcript entry for session resume
                let status = if approved { "ok" } else { "denied" };
                let result_preview = if tool_result.len() > 200 {
                    format!("{}...", &tool_result[..200])
                } else {
                    tool_result
                };
                self.emit(
                    session.session_id,
                    EventKind::TurnAddedV1 {
                        role: "tool".to_string(),
                        content: format!("[{}] {}: {}", internal_name, status, result_preview),
                    },
                )?;
            }
        }
    }

    /// Build a system prompt for chat-with-tools mode.
    fn build_chat_system_prompt(&self, _user_prompt: &str) -> Result<String> {
        let workspace = self.workspace.to_string_lossy();
        let now = Utc::now();

        // Detect project type
        let mut project_markers = Vec::new();
        for (file, lang) in &[
            ("Cargo.toml", "Rust"),
            ("package.json", "JavaScript/TypeScript"),
            ("pyproject.toml", "Python"),
            ("go.mod", "Go"),
            ("pom.xml", "Java/Maven"),
            ("build.gradle", "Java/Gradle"),
            ("Gemfile", "Ruby"),
            ("composer.json", "PHP"),
            ("CMakeLists.txt", "C/C++"),
            ("Makefile", "Make"),
        ] {
            if self.workspace.join(file).exists() {
                project_markers.push(*lang);
            }
        }
        let project_info = if project_markers.is_empty() {
            String::new()
        } else {
            format!("Project type: {}\n", project_markers.join(", "))
        };

        let shell = std::env::var("SHELL").unwrap_or_else(|_| "unknown".to_string());
        let permission_mode = self.policy.permission_mode();
        let base_model = &self.cfg.llm.base_model;
        let max_model = &self.cfg.llm.max_think_model;

        // Try to get git branch and short status
        let git_info = std::process::Command::new("git")
            .args(["branch", "--show-current"])
            .current_dir(&self.workspace)
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
                } else {
                    None
                }
            })
            .map(|branch| format!("Git branch: {branch}\n"))
            .unwrap_or_default();

        let mut parts = vec![format!(
            "You are DeepSeek, an AI coding assistant powering the `deepseek` CLI. \
             You help users with software engineering tasks including writing code, \
             debugging, refactoring, testing, and explaining codebases.\n\n\
             # Environment\n\
             Working directory: {workspace}\n\
             Platform: {} ({})\n\
             Shell: {shell}\n\
             Date: {}\n\
             {project_info}\
             {git_info}\
             Models: {base_model} (fast) / {max_model} (reasoning)\n\
             Permission mode: {permission_mode:?}\n\n\
             # Permission Mode\n\
             Current mode is **{permission_mode:?}**.\n\
             - Ask: tool calls that modify files or run commands require user approval\n\
             - Auto: tool calls matching the allowlist are auto-approved\n\
             - Locked: all non-read operations are denied — read-only session\n\
             Adjust your approach accordingly. In Locked mode, only use read tools.\n\n\
             # Tool Usage Guidelines\n\
             - **fs_read**: Always read a file before editing it. Use `start_line`/`end_line` for large files. Supports images and PDFs (use `pages` for PDFs).\n\
             - **fs_edit**: Use exact `search` strings to make precise edits. The search string must appear verbatim in the file. Set `all: false` for first-occurrence-only.\n\
             - **fs_write**: Only for creating new files. Prefer `fs_edit` to modify existing files.\n\
             - **fs_glob**: Find files by pattern (e.g. `**/*.rs`). Use before grep to scope searches.\n\
             - **fs_grep**: Search file contents with regex. Use `case_sensitive: false` for case-insensitive. Use `glob` to filter file types.\n\
             - **bash_run**: Execute shell commands (git, build, test, etc.). Commands have a timeout (default 120s).\n\
             - **multi_edit**: Batch edits across multiple files in one call. Each entry has a path and search/replace pairs.\n\
             - **git_status / git_diff / git_show**: Inspect repository state, diffs, and specific commits.\n\
             - **web_fetch**: Retrieve URL content as text. Use for documentation lookup.\n\
             - **web_search**: Search the web and return structured results.\n\
             - **index_query**: Full-text code search across the indexed codebase.\n\
             - **diagnostics_check**: Run language-specific diagnostics (cargo check, tsc, ruff, etc.).\n\
             - **patch_stage / patch_apply**: Stage and apply unified diffs with SHA verification.\n\
             - **notebook_read / notebook_edit**: Read and modify Jupyter notebooks.\n\n\
             # Safety Rules\n\
             - Always read a file before editing it — understand existing code first\n\
             - Never delete files, force-push, or run destructive commands without explicit user approval\n\
             - Stay within the working directory unless told otherwise\n\
             - Do not modify files outside the project without asking\n\
             - Do not introduce security vulnerabilities (command injection, XSS, SQL injection)\n\
             - Do not commit files containing secrets (.env, credentials, API keys)\n\n\
             # Git Protocol\n\
             - Create new commits — never amend unless explicitly asked\n\
             - Never push without user confirmation\n\
             - Use descriptive commit messages that explain the \"why\"\n\
             - Stage specific files by name, not `git add -A` or `git add .`\n\
             - Never use --force, --no-verify, or destructive git commands without asking\n\n\
             # Error Recovery\n\
             - If a tool call fails, read the error message carefully\n\
             - Try a different approach rather than repeating the same call\n\
             - Re-read the file after an edit failure to check current state\n\
             - If blocked, explain the issue to the user rather than brute-forcing\n\n\
             # Style\n\
             - Be concise and focused — avoid over-engineering\n\
             - Use markdown formatting in responses\n\
             - Reference files with path:line_number format\n\
             - When done, briefly explain what you changed and why",
            std::env::consts::OS,
            std::env::consts::ARCH,
            now.format("%Y-%m-%d"),
        )];

        // Add DEEPSEEK.md / memory if available
        if let Ok(manager) = MemoryManager::new(&self.workspace)
            && let Ok(memory) = manager.read_combined_memory()
            && !memory.trim().is_empty()
        {
            parts.push(format!("\n\n[Project Memory]\n{memory}"));
        }

        // Add recent verification feedback
        let session = self.ensure_session()?;
        let verification_feedback = self
            .store
            .list_recent_verification_runs(session.session_id, 5)
            .unwrap_or_default()
            .into_iter()
            .filter(|run| !run.success)
            .take(3)
            .collect::<Vec<_>>();
        if !verification_feedback.is_empty() {
            parts.push(format!(
                "\n\n[Recent Verification Failures]\n{}",
                format_verification_feedback(&verification_feedback)
            ));
        }

        Ok(parts.join(""))
    }

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

    fn run_subagents(&self, session_id: Uuid, plan: &Plan) -> Result<Vec<String>> {
        let max_tasks = self
            .subagents
            .max_concurrency
            .saturating_mul(3)
            .max(self.subagents.max_concurrency);
        let lane_plan = plan_subagent_execution_lanes(&plan.steps, max_tasks);
        let mut tasks = Vec::new();
        let mut task_targets = HashMap::new();
        let mut task_specialization = HashMap::new();
        let mut task_phase = HashMap::new();
        let mut task_dependencies = HashMap::new();
        let mut task_lane = HashMap::new();
        for lane in lane_plan {
            let goal = if lane.dependencies.is_empty() {
                format!(
                    "{} [phase={} lane={}]",
                    lane.intent,
                    lane.phase + 1,
                    lane.ownership_lane
                )
            } else {
                format!(
                    "{} [phase={} lane={} deps={}]",
                    lane.intent,
                    lane.phase + 1,
                    lane.ownership_lane,
                    lane.dependencies.join("|")
                )
            };
            let task = SubagentTask {
                run_id: Uuid::now_v7(),
                name: lane.title.clone(),
                goal,
                role: lane.role.clone(),
                team: lane.team.clone(),
                read_only_fallback: false,
            };
            let specialization_hint =
                self.load_subagent_specialization_hint(&lane.role, &lane.domain)?;
            task_targets.insert(task.run_id, lane.targets);
            task_specialization.insert(task.run_id, (lane.domain, specialization_hint));
            task_phase.insert(task.run_id, lane.phase);
            task_dependencies.insert(task.run_id, lane.dependencies);
            task_lane.insert(task.run_id, lane.ownership_lane);
            tasks.push(task);
        }
        if tasks.is_empty() {
            return Ok(Vec::new());
        }

        let now = Utc::now().to_rfc3339();
        let task_by_id = tasks
            .iter()
            .map(|task| {
                let targets = task_targets.get(&task.run_id).cloned().unwrap_or_default();
                let (domain, specialization_hint) = task_specialization
                    .get(&task.run_id)
                    .cloned()
                    .unwrap_or_else(|| ("general".to_string(), None));
                let phase = task_phase.get(&task.run_id).copied().unwrap_or(0);
                let dependencies = task_dependencies
                    .get(&task.run_id)
                    .cloned()
                    .unwrap_or_default();
                let ownership_lane = task_lane
                    .get(&task.run_id)
                    .cloned()
                    .unwrap_or_else(|| "execution:unscoped".to_string());
                (
                    task.run_id,
                    SubagentTaskMeta {
                        name: task.name.clone(),
                        goal: task.goal.clone(),
                        created_at: now.clone(),
                        targets,
                        domain,
                        specialization_hint,
                        phase,
                        dependencies,
                        ownership_lane,
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        for task in &tasks {
            self.emit(
                session_id,
                EventKind::SubagentSpawnedV1 {
                    run_id: task.run_id,
                    name: task.name.clone(),
                    goal: task.goal.clone(),
                },
            )?;
            self.tool_host.fire_session_hooks("subagentspawned");
            self.store.upsert_subagent_run(&SubagentRunRecord {
                run_id: task.run_id,
                name: task.name.clone(),
                goal: task.goal.clone(),
                status: "running".to_string(),
                output: None,
                error: None,
                created_at: now.clone(),
                updated_at: now.clone(),
            })?;
        }

        let llm_cfg = self.cfg.llm.clone();
        let base_model = self.cfg.llm.base_model.clone();
        let max_think_model = self.cfg.llm.max_think_model.clone();
        let root_goal = plan.goal.clone();
        let tool_host = Arc::clone(&self.tool_host);
        let task_targets_ref = Arc::new(task_targets.clone());
        let task_meta_ref = Arc::new(task_by_id.clone());
        let verification = if plan.verification.is_empty() {
            "none".to_string()
        } else {
            plan.verification.join(" ; ")
        };
        let mut tasks_by_phase: BTreeMap<usize, Vec<SubagentTask>> = BTreeMap::new();
        for task in tasks {
            let phase = task_by_id
                .get(&task.run_id)
                .map(|meta| meta.phase)
                .unwrap_or(0);
            tasks_by_phase.entry(phase).or_default().push(task);
        }
        for phase_tasks in tasks_by_phase.values_mut() {
            phase_tasks.sort_by(|a, b| {
                a.team
                    .cmp(&b.team)
                    .then(a.name.cmp(&b.name))
                    .then(a.run_id.cmp(&b.run_id))
            });
        }

        let mut results = Vec::new();
        for phase_tasks in tasks_by_phase.into_values() {
            let llm_cfg = llm_cfg.clone();
            let base_model = base_model.clone();
            let max_think_model = max_think_model.clone();
            let root_goal = root_goal.clone();
            let verification = verification.clone();
            let tool_host = Arc::clone(&tool_host);
            let task_targets_ref = Arc::clone(&task_targets_ref);
            let task_meta_ref = Arc::clone(&task_meta_ref);
            let mut phase_results = self.subagents.run_tasks(phase_tasks, move |task| {
                let meta = task_meta_ref
                    .get(&task.run_id)
                    .cloned()
                    .unwrap_or_else(|| SubagentTaskMeta {
                        name: task.name.clone(),
                        goal: task.goal.clone(),
                        created_at: Utc::now().to_rfc3339(),
                        targets: Vec::new(),
                        domain: "general".to_string(),
                        specialization_hint: None,
                        phase: 0,
                        dependencies: Vec::new(),
                        ownership_lane: "execution:unscoped".to_string(),
                    });
                let request = subagent_request_for_task(
                    &task,
                    &root_goal,
                    &verification,
                    &base_model,
                    &max_think_model,
                    &meta.domain,
                    meta.specialization_hint.as_deref(),
                );
                let client = DeepSeekClient::new(llm_cfg.clone())?;
                let targets = task_targets_ref
                    .get(&task.run_id)
                    .cloned()
                    .unwrap_or_default();
                let delegated =
                    run_subagent_delegated_tools(&tool_host, &task, &root_goal, &targets);
                match client.complete(&request) {
                    Ok(resp) => {
                        if let Some(delegated) = delegated {
                            Ok(format!("{}\n[delegated_tools]\n{}", resp.text, delegated))
                        } else {
                            Ok(resp.text)
                        }
                    }
                    Err(err) => Ok(format!(
                        "fallback subagent '{}' role={:?} team={} analyzed intent '{}' (llm_error={}){}",
                        task.name,
                        task.role,
                        task.team,
                        task.goal,
                        err,
                        delegated
                            .map(|summary| format!("\n[delegated_tools]\n{summary}"))
                            .unwrap_or_default()
                    )),
                }
            });
            results.append(&mut phase_results);
        }
        let merged_summary = self.subagents.merge_results(&results);
        let arbitration = summarize_subagent_merge_arbitration(&results, &task_targets);
        let mut notes = Vec::new();
        let lane_notes = summarize_subagent_execution_lanes(&task_by_id);
        if !lane_notes.is_empty() {
            notes.push(lane_notes.join("\n"));
        }
        for result in results {
            let updated_at = Utc::now().to_rfc3339();
            let meta =
                task_by_id
                    .get(&result.run_id)
                    .cloned()
                    .unwrap_or_else(|| SubagentTaskMeta {
                        name: "subagent".to_string(),
                        goal: String::new(),
                        created_at: updated_at.clone(),
                        targets: Vec::new(),
                        domain: "general".to_string(),
                        specialization_hint: None,
                        phase: 0,
                        dependencies: Vec::new(),
                        ownership_lane: "execution:unscoped".to_string(),
                    });
            let mut details = vec![
                format!("phase={}", meta.phase + 1),
                format!("lane={}", meta.ownership_lane),
            ];
            if !meta.targets.is_empty() {
                details.push(format!("targets={}", meta.targets.join(",")));
            }
            if !meta.dependencies.is_empty() {
                details.push(format!("deps={}", meta.dependencies.join("|")));
            }
            let persisted_goal = format!("{} [{}]", meta.goal, details.join(" "));
            if result.success {
                self.emit(
                    session_id,
                    EventKind::SubagentCompletedV1 {
                        run_id: result.run_id,
                        output: result.output.clone(),
                    },
                )?;
                self.tool_host.fire_session_hooks("subagentcompleted");
                self.store.upsert_subagent_run(&SubagentRunRecord {
                    run_id: result.run_id,
                    name: meta.name,
                    goal: persisted_goal,
                    status: "completed".to_string(),
                    output: Some(result.output.clone()),
                    error: None,
                    created_at: meta.created_at,
                    updated_at,
                })?;
                self.remember_subagent_specialization(
                    &result.role,
                    &meta.domain,
                    true,
                    result.attempts,
                    &result.output,
                )?;
            } else {
                let error = result
                    .error
                    .unwrap_or_else(|| "unknown subagent error".to_string());
                let error_for_memory = error.clone();
                let persisted_goal = if meta.targets.is_empty() {
                    meta.goal.clone()
                } else {
                    format!("{} [targets={}]", meta.goal, meta.targets.join(","))
                };
                self.emit(
                    session_id,
                    EventKind::SubagentFailedV1 {
                        run_id: result.run_id,
                        error: error.clone(),
                    },
                )?;
                self.tool_host.fire_session_hooks("subagentcompleted");
                self.store.upsert_subagent_run(&SubagentRunRecord {
                    run_id: result.run_id,
                    name: meta.name,
                    goal: persisted_goal,
                    status: "failed".to_string(),
                    output: None,
                    error: Some(error),
                    created_at: meta.created_at,
                    updated_at,
                })?;
                self.remember_subagent_specialization(
                    &result.role,
                    &meta.domain,
                    false,
                    result.attempts,
                    &error_for_memory,
                )?;
            }
        }
        if !merged_summary.is_empty() {
            notes.push(merged_summary);
        }
        if !arbitration.is_empty() {
            notes.push(arbitration.join("\n"));
        }
        Ok(notes)
    }

    fn revise_plan_with_llm(
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
        if self.cfg.router.auto_max_think
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

    fn complete_with_cache(
        &self,
        session_id: Uuid,
        req: &LlmRequest,
    ) -> Result<deepseek_core::LlmResponse> {
        if self.cfg.scheduling.off_peak {
            let hour = Utc::now().hour() as u8;
            let start = self.cfg.scheduling.off_peak_start_hour;
            let end = self.cfg.scheduling.off_peak_end_hour;
            let in_window = in_off_peak_window(hour, start, end);
            if !in_window {
                let resume_after = next_off_peak_start(start);
                let mut reason = "outside_off_peak_window".to_string();
                if req.non_urgent && self.cfg.scheduling.defer_non_urgent {
                    let delay = seconds_until_off_peak_start(hour, start);
                    let capped_delay = if self.cfg.scheduling.max_defer_seconds == 0 {
                        delay
                    } else {
                        delay.min(self.cfg.scheduling.max_defer_seconds)
                    };
                    if capped_delay > 0 {
                        reason = format!("outside_off_peak_window_deferred_{}s", capped_delay);
                        thread::sleep(std::time::Duration::from_secs(capped_delay));
                    }
                }
                self.emit(
                    session_id,
                    EventKind::OffPeakScheduledV1 {
                        reason,
                        resume_after,
                    },
                )?;
            }
        }

        let cache_key = prompt_cache_key(&req.model, &req.prompt);
        if self.cfg.llm.prompt_cache_enabled
            && let Some(cached) = self.read_prompt_cache(&cache_key)?
        {
            self.store.insert_provider_metric(&ProviderMetricRecord {
                provider: "deepseek".to_string(),
                model: req.model.clone(),
                cache_key: Some(cache_key.clone()),
                cache_hit: true,
                latency_ms: 0,
                recorded_at: Utc::now().to_rfc3339(),
            })?;
            self.emit(
                session_id,
                EventKind::PromptCacheHitV1 {
                    cache_key,
                    model: req.model.clone(),
                },
            )?;
            return Ok(cached);
        }

        let started = Instant::now();
        let response = {
            let cb = self.stream_callback.lock().ok().and_then(|g| g.clone());
            if let Some(cb) = cb {
                self.llm.complete_streaming(req, cb)
            } else {
                self.llm.complete(req)
            }
        }?;
        let latency_ms = started.elapsed().as_millis() as u64;
        self.store.insert_provider_metric(&ProviderMetricRecord {
            provider: "deepseek".to_string(),
            model: req.model.clone(),
            cache_key: Some(cache_key.clone()),
            cache_hit: false,
            latency_ms,
            recorded_at: Utc::now().to_rfc3339(),
        })?;

        if self.cfg.llm.prompt_cache_enabled {
            self.write_prompt_cache(&cache_key, &response)?;
        }

        Ok(response)
    }

    fn augment_prompt_context(&self, session_id: Uuid, prompt: &str) -> Result<String> {
        let projection = self.store.rebuild_from_events(session_id)?;
        let transcript = projection.transcript;
        let transcript_text = transcript.join("\n");
        let transcript_tokens = estimate_tokens(&transcript_text);
        let threshold = (self.cfg.llm.context_window_tokens as f32
            * self.cfg.context.auto_compact_threshold.clamp(0.1, 1.0))
            as u64;

        let mut blocks = Vec::new();
        if transcript_tokens >= threshold && !transcript.is_empty() {
            if transcript.len() >= 20 && transcript.len().is_multiple_of(10) {
                let summary = summarize_transcript(&transcript, 30);
                let summary_id = Uuid::now_v7();
                let replay_pointer = format!(".deepseek/compactions/{summary_id}.md");
                let summary_path =
                    deepseek_core::runtime_dir(&self.workspace).join(&replay_pointer);
                if let Some(parent) = summary_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&summary_path, &summary)?;
                self.emit(
                    session_id,
                    EventKind::ContextCompactedV1 {
                        summary_id,
                        from_turn: 1,
                        to_turn: transcript.len() as u64,
                        token_delta_estimate: transcript_tokens as i64
                            - estimate_tokens(&summary) as i64,
                        replay_pointer,
                    },
                )?;
                self.tool_host.fire_session_hooks("contextcompacted");
                blocks.push(format!("[auto_compaction]\n{summary}"));
            }
        } else if !transcript.is_empty() {
            let recent = transcript
                .iter()
                .rev()
                .take(16)
                .cloned()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n");
            blocks.push(format!("[recent_transcript]\n{recent}"));
        }

        if let Ok(memory) = MemoryManager::new(&self.workspace)?.read_combined_memory()
            && !memory.trim().is_empty()
        {
            blocks.push(format!("[memory]\n{memory}"));
        }
        let strategy_entries = self.load_matching_strategies(prompt, 4).unwrap_or_default();
        if !strategy_entries.is_empty() {
            blocks.push(format!(
                "[strategy_memory]\n{}",
                format_strategy_entries(&strategy_entries)
            ));
        }
        let objective_entries = self
            .load_matching_objective_outcomes(prompt, 4)
            .unwrap_or_default();
        if !objective_entries.is_empty() {
            blocks.push(format!(
                "[objective_outcomes]\n{}",
                format_objective_outcomes(&objective_entries)
            ));
        }
        let verification_feedback = self
            .store
            .list_recent_verification_runs(session_id, 10)
            .unwrap_or_default()
            .into_iter()
            .filter(|run| !run.success)
            .take(6)
            .collect::<Vec<_>>();
        if !verification_feedback.is_empty() {
            blocks.push(format!(
                "[verification_feedback]\n{}",
                format_verification_feedback(&verification_feedback)
            ));
        }

        if blocks.is_empty() {
            Ok(prompt.to_string())
        } else {
            Ok(format!("{prompt}\n\n{}", blocks.join("\n\n")))
        }
    }

    fn prompt_cache_dir(&self) -> PathBuf {
        deepseek_core::runtime_dir(&self.workspace).join("prompt-cache")
    }

    fn read_prompt_cache(&self, cache_key: &str) -> Result<Option<deepseek_core::LlmResponse>> {
        let path = self.prompt_cache_dir().join(format!("{cache_key}.json"));
        if !path.exists() {
            return Ok(None);
        }
        let raw = fs::read_to_string(path)?;
        Ok(Some(serde_json::from_str(&raw)?))
    }

    fn write_prompt_cache(
        &self,
        cache_key: &str,
        response: &deepseek_core::LlmResponse,
    ) -> Result<()> {
        let dir = self.prompt_cache_dir();
        fs::create_dir_all(&dir)?;
        fs::write(
            dir.join(format!("{cache_key}.json")),
            serde_json::to_vec(response)?,
        )?;
        Ok(())
    }

    fn planner_strategy_memory_path(&self) -> PathBuf {
        deepseek_core::runtime_dir(&self.workspace).join("planner-strategies.json")
    }

    fn read_planner_strategy_memory(&self) -> Result<PlannerStrategyMemory> {
        let path = self.planner_strategy_memory_path();
        if !path.exists() {
            return Ok(PlannerStrategyMemory::default());
        }
        let raw = fs::read_to_string(path)?;
        let mut memory = serde_json::from_str(&raw).unwrap_or_default();
        normalize_strategy_memory(&mut memory);
        Ok(memory)
    }

    fn write_planner_strategy_memory(&self, memory: &PlannerStrategyMemory) -> Result<()> {
        let path = self.planner_strategy_memory_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(memory)?)?;
        Ok(())
    }

    fn load_matching_strategies(
        &self,
        prompt: &str,
        limit: usize,
    ) -> Result<Vec<PlannerStrategyEntry>> {
        let key = plan_goal_pattern(prompt);
        let key_terms = key
            .split('|')
            .map(str::trim)
            .filter(|term| !term.is_empty())
            .collect::<Vec<_>>();
        let memory = self.read_planner_strategy_memory()?;
        let mut matches = memory
            .entries
            .into_iter()
            .filter(|entry| {
                if entry.key == key {
                    return true;
                }
                key_terms
                    .iter()
                    .any(|term| entry.key.contains(term) || entry.goal_excerpt.contains(term))
            })
            .collect::<Vec<_>>();
        matches.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(b.success_count.cmp(&a.success_count))
                .then(a.failure_count.cmp(&b.failure_count))
                .then(b.updated_at.cmp(&a.updated_at))
        });
        matches.truncate(limit.max(1));
        Ok(matches)
    }

    fn remember_successful_strategy(&self, prompt: &str, plan: &Plan) -> Result<()> {
        let mut memory = self.read_planner_strategy_memory()?;
        let key = plan_goal_pattern(prompt);
        let strategy_summary = summarize_strategy(plan);
        let verification = plan
            .verification
            .iter()
            .map(|cmd| cmd.trim().to_string())
            .filter(|cmd| !cmd.is_empty())
            .take(6)
            .collect::<Vec<_>>();
        if verification.is_empty() {
            return Ok(());
        }
        let now = Utc::now().to_rfc3339();
        if let Some(entry) = memory.entries.iter_mut().find(|entry| entry.key == key) {
            entry.goal_excerpt = truncate_strategy_prompt(prompt);
            entry.strategy_summary = strategy_summary;
            entry.verification = verification;
            entry.success_count = entry.success_count.saturating_add(1);
            entry.score = compute_strategy_score(entry.success_count, entry.failure_count);
            entry.last_outcome = "success".to_string();
            entry.updated_at = now;
        } else {
            memory.entries.push(PlannerStrategyEntry {
                key,
                goal_excerpt: truncate_strategy_prompt(prompt),
                strategy_summary,
                verification,
                success_count: 1,
                failure_count: 0,
                score: compute_strategy_score(1, 0),
                last_outcome: "success".to_string(),
                updated_at: now,
            });
        }
        sort_and_prune_strategy_entries(&mut memory.entries);
        self.write_planner_strategy_memory(&memory)
    }

    fn remember_failed_strategy(
        &self,
        prompt: &str,
        plan: &Plan,
        failure_streak: u32,
        verification_failures: u32,
    ) -> Result<()> {
        let mut memory = self.read_planner_strategy_memory()?;
        let key = plan_goal_pattern(prompt);
        let mut strategy_summary = summarize_strategy(plan);
        strategy_summary.push_str(&format!(
            " | failures=execution:{} verification:{}",
            failure_streak, verification_failures
        ));
        let verification = plan
            .verification
            .iter()
            .map(|cmd| cmd.trim().to_string())
            .filter(|cmd| !cmd.is_empty())
            .take(6)
            .collect::<Vec<_>>();
        let now = Utc::now().to_rfc3339();
        if let Some(entry) = memory.entries.iter_mut().find(|entry| entry.key == key) {
            entry.goal_excerpt = truncate_strategy_prompt(prompt);
            entry.strategy_summary = strategy_summary;
            if !verification.is_empty() {
                entry.verification = verification;
            }
            entry.failure_count = entry.failure_count.saturating_add(1);
            entry.score = compute_strategy_score(entry.success_count, entry.failure_count);
            entry.last_outcome = "failure".to_string();
            entry.updated_at = now;
        } else {
            memory.entries.push(PlannerStrategyEntry {
                key,
                goal_excerpt: truncate_strategy_prompt(prompt),
                strategy_summary,
                verification,
                success_count: 0,
                failure_count: 1,
                score: compute_strategy_score(0, 1),
                last_outcome: "failure".to_string(),
                updated_at: now,
            });
        }
        sort_and_prune_strategy_entries(&mut memory.entries);
        self.write_planner_strategy_memory(&memory)
    }

    fn objective_outcome_memory_path(&self) -> PathBuf {
        deepseek_core::runtime_dir(&self.workspace).join("objective-outcomes.json")
    }

    fn read_objective_outcome_memory(&self) -> Result<ObjectiveOutcomeMemory> {
        let path = self.objective_outcome_memory_path();
        if !path.exists() {
            return Ok(ObjectiveOutcomeMemory::default());
        }
        let raw = fs::read_to_string(path)?;
        let mut memory = serde_json::from_str(&raw).unwrap_or_default();
        normalize_objective_outcome_memory(&mut memory);
        Ok(memory)
    }

    fn write_objective_outcome_memory(&self, memory: &ObjectiveOutcomeMemory) -> Result<()> {
        let path = self.objective_outcome_memory_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(memory)?)?;
        Ok(())
    }

    fn load_matching_objective_outcomes(
        &self,
        prompt: &str,
        limit: usize,
    ) -> Result<Vec<ObjectiveOutcomeEntry>> {
        let key = plan_goal_pattern(prompt);
        let key_terms = key
            .split('|')
            .map(str::trim)
            .filter(|term| !term.is_empty())
            .collect::<Vec<_>>();
        let memory = self.read_objective_outcome_memory()?;
        let mut matches = memory
            .entries
            .into_iter()
            .filter(|entry| {
                if entry.key == key {
                    return true;
                }
                key_terms
                    .iter()
                    .any(|term| entry.key.contains(term) || entry.goal_excerpt.contains(term))
            })
            .collect::<Vec<_>>();
        matches.sort_by(|a, b| {
            b.confidence
                .total_cmp(&a.confidence)
                .then(b.success_count.cmp(&a.success_count))
                .then(a.failure_count.cmp(&b.failure_count))
                .then(b.updated_at.cmp(&a.updated_at))
        });
        matches.truncate(limit.max(1));
        Ok(matches)
    }

    fn remember_objective_outcome(
        &self,
        prompt: &str,
        plan: &Plan,
        failure_streak: u32,
        verification_failures: u32,
        success: bool,
    ) -> Result<()> {
        let mut memory = self.read_objective_outcome_memory()?;
        let key = plan_goal_pattern(prompt);
        let goal_excerpt = truncate_strategy_prompt(prompt);
        let now = Utc::now().to_rfc3339();
        let observed_step_count = plan.steps.len() as f32;
        let observed_failure_count = failure_streak as f32 + verification_failures as f32;
        let execution_failures = failure_streak.saturating_sub(verification_failures) as u64;
        let verification_failures = verification_failures as u64;
        let failure_summary = if success {
            "none".to_string()
        } else if failure_streak > 0 && verification_failures > 0 {
            format!(
                "execution_failures={} verification_failures={}",
                execution_failures, verification_failures
            )
        } else if verification_failures > 0 {
            format!("verification_failures={verification_failures}")
        } else {
            format!("execution_failures={execution_failures}")
        };

        if let Some(entry) = memory.entries.iter_mut().find(|entry| entry.key == key) {
            let prior_observations = entry.success_count.saturating_add(entry.failure_count);
            let next_observations = prior_observations.saturating_add(1);
            entry.goal_excerpt = goal_excerpt;
            entry.avg_step_count = rolling_average(
                entry.avg_step_count,
                prior_observations,
                observed_step_count,
                next_observations,
            );
            entry.avg_failure_count = rolling_average(
                entry.avg_failure_count,
                prior_observations,
                observed_failure_count,
                next_observations,
            );
            if success {
                entry.success_count = entry.success_count.saturating_add(1);
                entry.last_outcome = "success".to_string();
            } else {
                entry.failure_count = entry.failure_count.saturating_add(1);
                entry.execution_failure_count = entry
                    .execution_failure_count
                    .saturating_add(execution_failures);
                entry.verification_failure_count = entry
                    .verification_failure_count
                    .saturating_add(verification_failures);
                entry.last_outcome = "failure".to_string();
            }
            entry.last_failure_summary = failure_summary;
            entry.next_focus = objective_next_focus(entry);
            entry.confidence = compute_objective_confidence(entry);
            entry.updated_at = now;
        } else {
            let mut entry = ObjectiveOutcomeEntry {
                key,
                goal_excerpt,
                success_count: if success { 1 } else { 0 },
                failure_count: if success { 0 } else { 1 },
                execution_failure_count: if success { 0 } else { execution_failures },
                verification_failure_count: if success { 0 } else { verification_failures },
                avg_step_count: observed_step_count,
                avg_failure_count: observed_failure_count,
                confidence: 0.5,
                last_outcome: if success {
                    "success".to_string()
                } else {
                    "failure".to_string()
                },
                last_failure_summary: failure_summary,
                next_focus: String::new(),
                updated_at: now,
            };
            entry.next_focus = objective_next_focus(&entry);
            entry.confidence = compute_objective_confidence(&entry);
            memory.entries.push(entry);
        }
        sort_and_prune_objective_entries(&mut memory.entries);
        self.write_objective_outcome_memory(&memory)
    }

    fn subagent_specialization_memory_path(&self) -> PathBuf {
        deepseek_core::runtime_dir(&self.workspace).join("subagent-specializations.json")
    }

    fn read_subagent_specialization_memory(&self) -> Result<SubagentSpecializationMemory> {
        let path = self.subagent_specialization_memory_path();
        if !path.exists() {
            return Ok(SubagentSpecializationMemory::default());
        }
        let raw = fs::read_to_string(path)?;
        let mut memory = serde_json::from_str(&raw).unwrap_or_default();
        normalize_subagent_specialization_memory(&mut memory);
        Ok(memory)
    }

    fn write_subagent_specialization_memory(
        &self,
        memory: &SubagentSpecializationMemory,
    ) -> Result<()> {
        let path = self.subagent_specialization_memory_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(memory)?)?;
        Ok(())
    }

    fn load_subagent_specialization_hint(
        &self,
        role: &SubagentRole,
        domain: &str,
    ) -> Result<Option<String>> {
        let key = subagent_specialization_key(role, domain);
        let memory = self.read_subagent_specialization_memory()?;
        let best = memory.entries.into_iter().find(|entry| entry.key == key);
        Ok(best.map(|entry| format_subagent_specialization_hint(&entry)))
    }

    fn remember_subagent_specialization(
        &self,
        role: &SubagentRole,
        domain: &str,
        success: bool,
        attempts: u8,
        summary: &str,
    ) -> Result<()> {
        let key = subagent_specialization_key(role, domain);
        let role_name = format!("{role:?}");
        let now = Utc::now().to_rfc3339();
        let mut memory = self.read_subagent_specialization_memory()?;
        if let Some(entry) = memory.entries.iter_mut().find(|entry| entry.key == key) {
            let prior_observations = entry.success_count.saturating_add(entry.failure_count);
            let next_observations = prior_observations.saturating_add(1);
            entry.avg_attempts = rolling_average(
                entry.avg_attempts,
                prior_observations,
                attempts as f32,
                next_observations,
            );
            if success {
                entry.success_count = entry.success_count.saturating_add(1);
                entry.last_outcome = "success".to_string();
            } else {
                entry.failure_count = entry.failure_count.saturating_add(1);
                entry.last_outcome = "failure".to_string();
            }
            entry.last_summary = truncate_probe_text(summary.to_string());
            entry.next_guidance = subagent_specialization_guidance(entry);
            entry.confidence = compute_subagent_specialization_confidence(entry);
            entry.updated_at = now;
        } else {
            let mut entry = SubagentSpecializationEntry {
                key,
                role: role_name,
                domain: domain.to_string(),
                success_count: if success { 1 } else { 0 },
                failure_count: if success { 0 } else { 1 },
                avg_attempts: attempts as f32,
                confidence: 0.5,
                last_outcome: if success {
                    "success".to_string()
                } else {
                    "failure".to_string()
                },
                last_summary: truncate_probe_text(summary.to_string()),
                next_guidance: String::new(),
                updated_at: now,
            };
            entry.next_guidance = subagent_specialization_guidance(&entry);
            entry.confidence = compute_subagent_specialization_confidence(&entry);
            memory.entries.push(entry);
        }
        sort_and_prune_subagent_specialization_entries(&mut memory.entries);
        self.write_subagent_specialization_memory(&memory)
    }

    fn emit_cost_event(
        &self,
        session_id: Uuid,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Result<f64> {
        let estimated_cost_usd = (input_tokens as f64 / 1_000_000.0)
            * self.cfg.usage.cost_per_million_input
            + (output_tokens as f64 / 1_000_000.0) * self.cfg.usage.cost_per_million_output;
        self.emit(
            session_id,
            EventKind::CostUpdatedV1 {
                input_tokens,
                output_tokens,
                estimated_cost_usd,
            },
        )?;
        Ok(estimated_cost_usd)
    }

    fn transition(&self, session: &mut Session, to: SessionState) -> Result<()> {
        let from = session.status.clone();
        if !is_valid_session_state_transition(&from, &to) {
            return Err(anyhow!(
                "invalid session state transition: {:?} -> {:?}",
                from,
                to
            ));
        }
        session.status = to.clone();
        self.store.save_session(session)?;
        self.emit(
            session.session_id,
            EventKind::SessionStateChangedV1 { from, to },
        )
    }

    fn emit(&self, session_id: Uuid, kind: EventKind) -> Result<()> {
        let event = EventEnvelope {
            seq_no: self.store.next_seq_no(session_id)?,
            at: Utc::now(),
            session_id,
            kind,
        };
        self.store.append_event(&event)?;
        self.observer.record_event(&event)?;
        Ok(())
    }

    fn calls_for_step(&self, step: &PlanStep, plan_goal: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();
        for tool in &step.tools {
            if let Some(call) = self.call_for_declared_tool(step, plan_goal, tool) {
                if calls
                    .iter()
                    .any(|existing: &ToolCall| existing.name == call.name)
                {
                    continue;
                }
                calls.push(call);
            }
            if calls.len() >= 3 {
                break;
            }
        }
        if calls.is_empty() {
            calls.push(self.call_for_step(step, plan_goal));
        }
        calls
    }

    fn call_for_declared_tool(
        &self,
        step: &PlanStep,
        plan_goal: &str,
        tool: &str,
    ) -> Option<ToolCall> {
        let (tool_name, suffix) = parse_declared_tool(tool);
        let suffix = suffix.as_deref();
        let primary_file = step.files.first().cloned();
        let search_pattern = plan_goal_pattern(plan_goal);
        match tool_name.as_str() {
            "index.query" => Some(ToolCall {
                name: "index.query".to_string(),
                args: json!({"q": plan_goal, "top_k": 10}),
                requires_approval: false,
            }),
            "fs.grep" => Some(ToolCall {
                name: "fs.grep".to_string(),
                args: json!({
                    "pattern": suffix.filter(|s| !s.is_empty()).unwrap_or(&search_pattern),
                    "glob": "**/*",
                    "limit": 50,
                    "respectGitignore": true
                }),
                requires_approval: false,
            }),
            "fs.read" => {
                let path = suffix
                    .filter(|s| !s.is_empty())
                    .map(ToString::to_string)
                    .or(primary_file)?;
                Some(ToolCall {
                    name: "fs.read".to_string(),
                    args: json!({"path": path}),
                    requires_approval: false,
                })
            }
            "fs.search_rg" => Some(ToolCall {
                name: "fs.search_rg".to_string(),
                args: json!({"query": plan_goal, "limit": 20}),
                requires_approval: false,
            }),
            "fs.list" => Some(ToolCall {
                name: "fs.list".to_string(),
                args: json!({"dir": "."}),
                requires_approval: false,
            }),
            "git.status" => Some(ToolCall {
                name: "git.status".to_string(),
                args: json!({}),
                requires_approval: false,
            }),
            "git.diff" => Some(ToolCall {
                name: "git.diff".to_string(),
                args: json!({}),
                requires_approval: false,
            }),
            "git.show" => Some(ToolCall {
                name: "git.show".to_string(),
                args: json!({"spec": suffix.filter(|s| !s.is_empty()).unwrap_or("HEAD")}),
                requires_approval: false,
            }),
            "bash.run" => Some(ToolCall {
                name: "bash.run".to_string(),
                args: json!({"cmd": suffix.filter(|s| !s.is_empty()).unwrap_or("cargo test --workspace")}),
                requires_approval: true,
            }),
            // Retain legacy behavior so existing patch/apply workflows continue to function.
            "patch.stage" => Some(ToolCall {
                name: "patch.stage".to_string(),
                args: json!({
                    "unified_diff": format!(
                            "diff --git a/.deepseek/notes.txt b/.deepseek/notes.txt\nnew file mode 100644\nindex 0000000..2b9d865\n--- /dev/null\n+++ b/.deepseek/notes.txt\n@@ -0,0 +1 @@\n+{}\n",
                        plan_goal.replace('\n', " ")
                    ),
                    "base": ""
                }),
                requires_approval: false,
            }),
            "fs.edit" => {
                let path = if let Some(path) = primary_file {
                    path
                } else if step.intent == "docs" {
                    "README.md".to_string()
                } else {
                    return None;
                };
                Some(ToolCall {
                    name: "fs.edit".to_string(),
                    args: json!({
                        "path": path,
                        "search": "## Verification",
                        "replace": "## Verification\n- Ensure DeepSeek API key is configured for strict-online mode.\n",
                        "all": false
                    }),
                    requires_approval: false,
                })
            }
            "fs.write" => {
                let path = suffix
                    .filter(|s| !s.is_empty())
                    .map(ToString::to_string)
                    .or(primary_file)
                    .unwrap_or_else(|| ".deepseek/notes.txt".to_string());
                Some(ToolCall {
                    name: "fs.write".to_string(),
                    args: json!({
                        "path": path,
                        "content": format!("Plan goal: {}\nStep: {}\nIntent: {}\n", plan_goal, step.title, step.intent)
                    }),
                    requires_approval: false,
                })
            }
            _ => None,
        }
    }

    fn call_for_step(&self, step: &PlanStep, plan_goal: &str) -> ToolCall {
        match step.intent.as_str() {
            "search" => ToolCall {
                name: "fs.search_rg".to_string(),
                args: json!({"query": plan_goal, "limit": 10}),
                requires_approval: false,
            },
            "git" => ToolCall {
                name: "git.status".to_string(),
                args: json!({}),
                requires_approval: false,
            },
            "edit" => ToolCall {
                name: "patch.stage".to_string(),
                args: json!({
                    "unified_diff": format!(
                        "diff --git a/.deepseek/notes.txt b/.deepseek/notes.txt\nnew file mode 100644\nindex 0000000..2b9d865\n--- /dev/null\n+++ b/.deepseek/notes.txt\n@@ -0,0 +1 @@\n+{}\n",
                        plan_goal.replace('\n', " ")
                    ),
                    "base": ""
                }),
                requires_approval: false,
            },
            "docs" => ToolCall {
                name: "fs.edit".to_string(),
                args: json!({
                    "path": "README.md",
                    "search": "## Verification",
                    "replace": "## Verification\n- Ensure DeepSeek API key is configured for strict-online mode.\n",
                    "all": false
                }),
                requires_approval: false,
            },
            "recover" => ToolCall {
                name: "fs.grep".to_string(),
                args: json!({"pattern":"error|failed|panic","glob":"**/*","limit":25}),
                requires_approval: false,
            },
            "verify" => ToolCall {
                name: "bash.run".to_string(),
                args: json!({"cmd": "cargo test --workspace"}),
                requires_approval: true,
            },
            _ => ToolCall {
                name: "fs.list".to_string(),
                args: json!({"dir": "."}),
                requires_approval: false,
            },
        }
    }

    fn request_tool_approval(&self, call: &ToolCall) -> Result<bool> {
        // Try external approval handler first (TUI / raw-mode compatible).
        if let Ok(mut guard) = self.approval_handler.lock()
            && let Some(handler) = guard.as_mut()
        {
            return handler(call);
        }

        // Fallback: blocking stdin for non-TUI mode.
        let mut stdout = std::io::stdout();
        let stdin = std::io::stdin();
        if !stdin.is_terminal() || !stdout.is_terminal() {
            return Ok(false);
        }

        let compact_args = serde_json::to_string(&call.args)
            .unwrap_or_else(|_| "<unserializable args>".to_string());
        writeln!(
            stdout,
            "approval required for tool `{}` with args {}",
            call.name, compact_args
        )?;
        write!(stdout, "approve this call? [y/N]: ")?;
        stdout.flush()?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let normalized = input.trim().to_ascii_lowercase();
        Ok(matches!(normalized.as_str(), "y" | "yes"))
    }

    fn emit_patch_events_if_any(
        &self,
        session_id: Uuid,
        call_name: &str,
        result: &deepseek_core::ToolResult,
    ) -> Result<()> {
        if call_name == "patch.stage"
            && result.success
            && let Some(id) = result.output.get("patch_id").and_then(|v| v.as_str())
        {
            let patch_id = Uuid::parse_str(id)?;
            let base_sha256 = result
                .output
                .get("base_sha256")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            self.emit(
                session_id,
                EventKind::PatchStagedV1 {
                    patch_id,
                    base_sha256,
                },
            )?;
        }
        if call_name == "patch.apply" && result.success {
            let patch_id = result
                .output
                .get("patch_id")
                .and_then(|v| v.as_str())
                .and_then(|s| Uuid::parse_str(s).ok())
                .unwrap_or_else(Uuid::now_v7);
            let applied = result
                .output
                .get("applied")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let conflicts = result
                .output
                .get("conflicts")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(ToString::to_string))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            self.emit(
                session_id,
                EventKind::PatchAppliedV1 {
                    patch_id,
                    applied,
                    conflicts,
                },
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct PlanLlmShape {
    #[serde(default)]
    goal: Option<String>,
    #[serde(default)]
    assumptions: Vec<String>,
    steps: Vec<PlanLlmStep>,
    #[serde(default)]
    verification: Vec<String>,
    #[serde(default)]
    risk_notes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct PlanLlmStep {
    title: String,
    intent: String,
    #[serde(default)]
    tools: Vec<String>,
    #[serde(default)]
    files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PlannerStrategyMemory {
    entries: Vec<PlannerStrategyEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PlannerStrategyEntry {
    key: String,
    goal_excerpt: String,
    strategy_summary: String,
    verification: Vec<String>,
    success_count: u64,
    #[serde(default)]
    failure_count: u64,
    #[serde(default = "default_strategy_score")]
    score: f32,
    #[serde(default)]
    last_outcome: String,
    updated_at: String,
}

fn default_strategy_score() -> f32 {
    0.5
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ObjectiveOutcomeMemory {
    entries: Vec<ObjectiveOutcomeEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ObjectiveOutcomeEntry {
    key: String,
    goal_excerpt: String,
    success_count: u64,
    #[serde(default)]
    failure_count: u64,
    #[serde(default)]
    execution_failure_count: u64,
    #[serde(default)]
    verification_failure_count: u64,
    #[serde(default)]
    avg_step_count: f32,
    #[serde(default)]
    avg_failure_count: f32,
    #[serde(default = "default_objective_confidence")]
    confidence: f32,
    #[serde(default)]
    last_outcome: String,
    #[serde(default)]
    last_failure_summary: String,
    #[serde(default)]
    next_focus: String,
    updated_at: String,
}

fn default_objective_confidence() -> f32 {
    0.5
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SubagentSpecializationMemory {
    entries: Vec<SubagentSpecializationEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SubagentSpecializationEntry {
    key: String,
    role: String,
    domain: String,
    success_count: u64,
    #[serde(default)]
    failure_count: u64,
    #[serde(default)]
    avg_attempts: f32,
    #[serde(default = "default_specialization_confidence")]
    confidence: f32,
    #[serde(default)]
    last_outcome: String,
    #[serde(default)]
    last_summary: String,
    #[serde(default)]
    next_guidance: String,
    updated_at: String,
}

fn default_specialization_confidence() -> f32 {
    0.5
}

fn build_planner_prompt(task: &str) -> String {
    format!(
        "Return only JSON with keys: goal, assumptions, steps, verification, risk_notes. \
         Each step must include: title, intent, tools, files. User task: {task}"
    )
}

fn build_plan_revision_prompt(
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

#[derive(Debug, Clone)]
struct PlanQualityReport {
    acceptable: bool,
    score: f32,
    issues: Vec<String>,
}

fn assess_plan_quality(plan: &Plan, user_prompt: &str) -> PlanQualityReport {
    let mut issues = Vec::new();
    let mut penalty = 0.0_f32;
    let prompt_lower = user_prompt.to_ascii_lowercase();

    let prompt_words = user_prompt.split_whitespace().count();
    let min_steps = if prompt_words >= 18 || user_prompt.len() >= 120 {
        3
    } else {
        2
    };
    if plan.steps.len() < min_steps {
        issues.push(format!(
            "plan has {} steps; expected at least {min_steps}",
            plan.steps.len()
        ));
        penalty += 0.25;
    }
    if plan.verification.is_empty() {
        issues.push("verification is empty".to_string());
        penalty += 0.35;
    }

    let steps_without_tools = plan
        .steps
        .iter()
        .filter(|step| step.tools.is_empty())
        .count();
    if steps_without_tools > 0 {
        issues.push(format!("{steps_without_tools} step(s) missing tools"));
        penalty += 0.20;
    }

    let mut unique_tools = HashSet::new();
    for step in &plan.steps {
        for tool in &step.tools {
            unique_tools.insert(normalize_declared_tool_name(
                tool.split_once(':').map_or(tool.as_str(), |(name, _)| name),
            ));
        }
    }
    if unique_tools.len() < 2 {
        issues.push("tool diversity is low (fewer than 2 unique tools)".to_string());
        penalty += 0.10;
    }

    let mut titles = HashSet::new();
    let mut duplicate_titles = 0usize;
    for step in &plan.steps {
        let lowered = step.title.trim().to_ascii_lowercase();
        if !lowered.is_empty() && !titles.insert(lowered) {
            duplicate_titles += 1;
        }
    }
    if duplicate_titles > 0 {
        issues.push(format!("{duplicate_titles} duplicate step title(s)"));
        penalty += 0.10;
    }

    if prompt_lower.contains("implement")
        || prompt_lower.contains("fix")
        || prompt_lower.contains("refactor")
        || prompt_lower.contains("change")
    {
        let has_edit = unique_tools.iter().any(|tool| {
            matches!(
                tool.as_str(),
                "fs.edit" | "fs.write" | "patch.stage" | "patch.apply"
            )
        });
        if !has_edit {
            issues
                .push("implementation intent detected but no edit/patch tool in plan".to_string());
            penalty += 0.20;
        }
    }

    let has_verification_tool = unique_tools.contains("bash.run")
        || plan
            .verification
            .iter()
            .any(|cmd| !cmd.trim().is_empty() && !cmd.trim().starts_with('#'));
    if !has_verification_tool {
        issues.push("verification commands or verification tool are missing".to_string());
        penalty += 0.20;
    }

    let score = (1.0 - penalty).clamp(0.0, 1.0);
    let acceptable = score >= 0.65
        && plan.steps.len() >= 2
        && !plan.verification.is_empty()
        && steps_without_tools == 0;
    PlanQualityReport {
        acceptable,
        score,
        issues,
    }
}

fn combine_plan_quality_reports(
    primary: PlanQualityReport,
    secondary: PlanQualityReport,
) -> PlanQualityReport {
    let mut issues = primary.issues;
    issues.extend(secondary.issues);
    PlanQualityReport {
        acceptable: primary.acceptable && secondary.acceptable,
        score: ((primary.score + secondary.score) / 2.0).clamp(0.0, 1.0),
        issues,
    }
}

fn assess_plan_long_horizon_quality(
    plan: &Plan,
    user_prompt: &str,
    objective_outcomes: &[ObjectiveOutcomeEntry],
) -> PlanQualityReport {
    let mut issues = Vec::new();
    let mut penalty = 0.0_f32;
    let prompt_lower = user_prompt.to_ascii_lowercase();
    let long_horizon_prompt = user_prompt.len() >= 170
        || prompt_lower.contains("end-to-end")
        || prompt_lower.contains("cross")
        || prompt_lower.contains("multi")
        || prompt_lower.contains("migration")
        || prompt_lower.contains("large")
        || prompt_lower.contains("long");
    let risk_heavy_objective = objective_outcomes
        .iter()
        .take(4)
        .any(|entry| entry.avg_failure_count >= 1.0 || entry.confidence < 0.45);

    if !long_horizon_prompt && !risk_heavy_objective {
        return PlanQualityReport {
            acceptable: true,
            score: 1.0,
            issues,
        };
    }

    let min_steps = if risk_heavy_objective { 4 } else { 3 };
    if plan.steps.len() < min_steps {
        issues.push(format!(
            "long-horizon objective requires at least {min_steps} decomposed steps"
        ));
        penalty += 0.25;
    }

    let has_phase_structure = plan.steps.iter().any(|step| {
        let title = step.title.to_ascii_lowercase();
        title.contains("phase")
            || title.contains("milestone")
            || title.contains("step 1")
            || title.contains("checkpoint")
            || title.contains("rollout")
    });
    if !has_phase_structure {
        issues.push("plan lacks explicit milestone/checkpoint decomposition".to_string());
        penalty += 0.20;
    }

    let has_checkpoint_guard = plan.steps.iter().any(|step| {
        let title = step.title.to_ascii_lowercase();
        title.contains("checkpoint")
            || title.contains("rollback")
            || title.contains("recovery")
            || title.contains("rewind")
    }) || plan
        .risk_notes
        .iter()
        .any(|note| note.to_ascii_lowercase().contains("rollback"));
    if !has_checkpoint_guard {
        issues.push("plan missing checkpoint/rollback guard for replanning safety".to_string());
        penalty += 0.25;
    }

    let has_replan_path = plan.steps.iter().any(|step| {
        let title = step.title.to_ascii_lowercase();
        title.contains("recover") || title.contains("fallback") || title.contains("triage")
    });
    if risk_heavy_objective && !has_replan_path {
        issues.push("historically risky objective lacks explicit recovery/replan path".to_string());
        penalty += 0.20;
    }

    let score = (1.0 - penalty).clamp(0.0, 1.0);
    PlanQualityReport {
        acceptable: score >= 0.70,
        score,
        issues,
    }
}

fn build_plan_quality_repair_prompt(
    user_prompt: &str,
    current_plan: &Plan,
    report: &PlanQualityReport,
) -> String {
    let plan_json = serde_json::to_string_pretty(current_plan)
        .unwrap_or_else(|_| "{\"error\":\"failed to serialize plan\"}".to_string());
    let issues = if report.issues.is_empty() {
        "- no issues captured".to_string()
    } else {
        report
            .issues
            .iter()
            .map(|issue| format!("- {issue}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "Improve the following plan quality and return ONLY JSON with keys: goal, assumptions, steps, verification, risk_notes.\n\
Each step must include: title, intent, tools, files.\n\
Keep the original goal and preserve useful progress, but resolve all quality issues.\n\n\
User goal:\n{user_prompt}\n\n\
Quality score: {:.2}\n\
Quality issues:\n{}\n\n\
Current draft plan:\n{}",
        report.score, issues, plan_json
    )
}

fn assess_plan_feedback_alignment(
    plan: &Plan,
    feedback: &[VerificationRunRecord],
) -> PlanQualityReport {
    if feedback.is_empty() {
        return PlanQualityReport {
            acceptable: true,
            score: 1.0,
            issues: Vec::new(),
        };
    }
    let mut issues = Vec::new();
    let mut missing = 0usize;
    let verification_text = plan
        .verification
        .iter()
        .chain(plan.steps.iter().map(|step| &step.title))
        .chain(plan.steps.iter().flat_map(|step| step.tools.iter()))
        .map(|item| item.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(" ");

    for run in feedback.iter().take(6) {
        let markers = verification_feedback_markers(&run.command);
        if markers.is_empty() {
            continue;
        }
        let covered = markers
            .iter()
            .any(|marker| verification_text.contains(marker));
        if !covered {
            missing += 1;
            issues.push(format!(
                "plan does not address previously failing command context: {}",
                run.command
            ));
        }
    }
    if issues.is_empty() {
        return PlanQualityReport {
            acceptable: true,
            score: 1.0,
            issues,
        };
    }
    let total = feedback.iter().take(6).count().max(1) as f32;
    let penalty = (missing as f32 / total) * 0.8;
    let score = (1.0 - penalty).clamp(0.0, 1.0);
    PlanQualityReport {
        acceptable: score >= 0.70 && missing == 0,
        score,
        issues,
    }
}

fn build_verification_feedback_repair_prompt(
    user_prompt: &str,
    current_plan: &Plan,
    report: &PlanQualityReport,
    feedback: &[VerificationRunRecord],
) -> String {
    let plan_json = serde_json::to_string_pretty(current_plan)
        .unwrap_or_else(|_| "{\"error\":\"failed to serialize plan\"}".to_string());
    let issues = if report.issues.is_empty() {
        "- no issues captured".to_string()
    } else {
        report
            .issues
            .iter()
            .map(|issue| format!("- {issue}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "Revise the plan and return ONLY JSON with keys: goal, assumptions, steps, verification, risk_notes.\n\
Each step must include: title, intent, tools, files.\n\
Incorporate verification feedback from previous failures.\n\n\
User goal:\n{user_prompt}\n\n\
Feedback alignment score: {:.2}\n\
Issues:\n{}\n\n\
Previous verification failures:\n{}\n\n\
Current draft plan:\n{}",
        report.score,
        issues,
        format_verification_feedback(feedback),
        plan_json
    )
}

fn verification_feedback_markers(command: &str) -> Vec<String> {
    command
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == ':'))
        .map(|token| token.trim().to_ascii_lowercase())
        .filter(|token| token.len() >= 3 && token != "and" && token != "the")
        .take(6)
        .collect::<Vec<_>>()
}

fn format_verification_feedback(feedback: &[VerificationRunRecord]) -> String {
    feedback
        .iter()
        .take(8)
        .map(|run| {
            let output = run.output.trim();
            let compact = if output.chars().count() > 120 {
                let head = output.chars().take(120).collect::<String>();
                format!("{head}...")
            } else {
                output.to_string()
            };
            format!("- [{}] {} => {}", run.run_at, run.command, compact)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn summarize_strategy(plan: &Plan) -> String {
    let mut segments = Vec::new();
    let step_titles = plan
        .steps
        .iter()
        .take(4)
        .map(|step| step.title.trim())
        .filter(|title| !title.is_empty())
        .collect::<Vec<_>>();
    if !step_titles.is_empty() {
        segments.push(format!("steps={}", step_titles.join(" -> ")));
    }
    let tools = plan
        .steps
        .iter()
        .flat_map(|step| step.tools.iter())
        .map(|tool| {
            normalize_declared_tool_name(tool.split_once(':').map_or(tool, |(name, _)| name))
        })
        .collect::<HashSet<_>>();
    if !tools.is_empty() {
        let mut sorted = tools.into_iter().collect::<Vec<_>>();
        sorted.sort();
        segments.push(format!("tools={}", sorted.join(",")));
    }
    if plan.verification.is_empty() {
        segments.push("verification=none".to_string());
    } else {
        segments.push(format!(
            "verification={}",
            plan.verification
                .iter()
                .take(3)
                .map(|cmd| cmd.trim())
                .filter(|cmd| !cmd.is_empty())
                .collect::<Vec<_>>()
                .join(" ; ")
        ));
    }
    segments.join(" | ")
}

fn truncate_strategy_prompt(prompt: &str) -> String {
    let trimmed = prompt.trim();
    if trimmed.chars().count() <= 220 {
        return trimmed.to_string();
    }
    let head = trimmed.chars().take(220).collect::<String>();
    format!("{head}...")
}

fn format_strategy_entries(entries: &[PlannerStrategyEntry]) -> String {
    entries
        .iter()
        .take(6)
        .map(|entry| {
            format!(
                "- key={} score={:.3} success_count={} failure_count={} last_outcome={} goal=\"{}\" strategy=\"{}\" verification={}",
                entry.key,
                entry.score,
                entry.success_count,
                entry.failure_count,
                if entry.last_outcome.is_empty() {
                    "unknown"
                } else {
                    entry.last_outcome.as_str()
                },
                entry.goal_excerpt,
                entry.strategy_summary,
                entry
                    .verification
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" ; ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn compute_strategy_score(success_count: u64, failure_count: u64) -> f32 {
    let observations = success_count.saturating_add(failure_count) as f32;
    let posterior_mean = (success_count as f32 + 1.0) / (observations + 2.0);
    let confidence = (observations / 10.0).clamp(0.0, 1.0);
    (0.5 * (1.0 - confidence) + posterior_mean * confidence).clamp(0.0, 1.0)
}

fn normalize_strategy_memory(memory: &mut PlannerStrategyMemory) {
    for entry in &mut memory.entries {
        if !entry.score.is_finite() || entry.score <= 0.0 {
            entry.score = compute_strategy_score(entry.success_count, entry.failure_count);
        }
        if entry.last_outcome.trim().is_empty() {
            entry.last_outcome = if entry.success_count >= entry.failure_count {
                "success".to_string()
            } else {
                "failure".to_string()
            };
        }
    }
    sort_and_prune_strategy_entries(&mut memory.entries);
}

fn sort_and_prune_strategy_entries(entries: &mut Vec<PlannerStrategyEntry>) {
    entries.retain(|entry| {
        let observations = entry.success_count.saturating_add(entry.failure_count);
        !(observations >= 3
            && entry.failure_count > entry.success_count.saturating_add(1)
            && entry.score < 0.35)
    });
    entries.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(b.success_count.cmp(&a.success_count))
            .then(a.failure_count.cmp(&b.failure_count))
            .then(b.updated_at.cmp(&a.updated_at))
    });
    entries.truncate(64);
}

fn rolling_average(previous: f32, prior_count: u64, observed: f32, next_count: u64) -> f32 {
    if prior_count == 0 || next_count == 0 {
        return observed;
    }
    ((previous * prior_count as f32) + observed) / next_count as f32
}

fn compute_objective_confidence(entry: &ObjectiveOutcomeEntry) -> f32 {
    let observations = entry.success_count.saturating_add(entry.failure_count);
    if observations == 0 {
        return 0.5;
    }
    let success_rate = entry.success_count as f32 / observations as f32;
    let verification_penalty = if observations == 0 {
        0.0
    } else {
        (entry.verification_failure_count as f32 / observations as f32).min(1.0) * 0.25
    };
    let failure_penalty = (entry.avg_failure_count / 3.0).min(1.0) * 0.20;
    let sample_confidence = (observations as f32 / 10.0).min(1.0);
    let posterior = (entry.success_count as f32 + 1.0) / (observations as f32 + 2.0);
    let blended = ((1.0 - sample_confidence) * 0.5) + (sample_confidence * posterior);
    (blended + success_rate * 0.15 - verification_penalty - failure_penalty).clamp(0.0, 1.0)
}

fn objective_next_focus(entry: &ObjectiveOutcomeEntry) -> String {
    let verification_heavy = entry.verification_failure_count > entry.execution_failure_count;
    if entry.last_outcome.eq_ignore_ascii_case("success") {
        return "preserve successful decomposition and keep verification breadth".to_string();
    }
    if verification_heavy {
        return "expand verification coverage and map prior failing checks into plan steps"
            .to_string();
    }
    if entry.execution_failure_count > 0 {
        return "reduce plan branching and add explicit recovery checkpoints for execution failures"
            .to_string();
    }
    "stabilize plan ordering and retain explicit validation gates".to_string()
}

fn format_objective_outcomes(entries: &[ObjectiveOutcomeEntry]) -> String {
    entries
        .iter()
        .take(6)
        .map(|entry| {
            let observations = entry.success_count.saturating_add(entry.failure_count);
            let success_rate = if observations == 0 {
                0.0
            } else {
                entry.success_count as f32 / observations as f32
            };
            format!(
                "- key={} confidence={:.3} success_rate={:.3} avg_steps={:.2} avg_failures={:.2} last_outcome={} focus=\"{}\" last_failure=\"{}\"",
                entry.key,
                entry.confidence,
                success_rate,
                entry.avg_step_count,
                entry.avg_failure_count,
                if entry.last_outcome.is_empty() {
                    "unknown"
                } else {
                    entry.last_outcome.as_str()
                },
                if entry.next_focus.is_empty() {
                    "none"
                } else {
                    entry.next_focus.as_str()
                },
                if entry.last_failure_summary.is_empty() {
                    "none"
                } else {
                    entry.last_failure_summary.as_str()
                }
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize_objective_outcome_memory(memory: &mut ObjectiveOutcomeMemory) {
    for entry in &mut memory.entries {
        if !entry.confidence.is_finite() || entry.confidence <= 0.0 {
            entry.confidence = compute_objective_confidence(entry);
        }
        if entry.last_outcome.trim().is_empty() {
            entry.last_outcome = if entry.success_count >= entry.failure_count {
                "success".to_string()
            } else {
                "failure".to_string()
            };
        }
        if entry.next_focus.trim().is_empty() {
            entry.next_focus = objective_next_focus(entry);
        }
        if entry.avg_step_count <= 0.0 {
            entry.avg_step_count = 1.0;
        }
    }
    sort_and_prune_objective_entries(&mut memory.entries);
}

fn sort_and_prune_objective_entries(entries: &mut Vec<ObjectiveOutcomeEntry>) {
    entries.retain(|entry| {
        let observations = entry.success_count.saturating_add(entry.failure_count);
        !(observations >= 4
            && entry.failure_count > entry.success_count.saturating_add(2)
            && entry.confidence < 0.30)
    });
    entries.sort_by(|a, b| {
        b.confidence
            .total_cmp(&a.confidence)
            .then(b.success_count.cmp(&a.success_count))
            .then(a.failure_count.cmp(&b.failure_count))
            .then(b.updated_at.cmp(&a.updated_at))
    });
    entries.truncate(96);
}

fn parse_plan_from_llm(text: &str, fallback_goal: &str) -> Option<Plan> {
    let snippet = extract_json_snippet(text)?;
    let parsed: PlanLlmShape = serde_json::from_str(snippet).ok()?;
    let mut steps = Vec::new();
    for step in parsed.steps.into_iter().take(16) {
        let title = step.title.trim();
        if title.is_empty() {
            continue;
        }
        let inferred_intent = infer_intent(&step.intent, &step.tools, title);
        let mut tools = step
            .tools
            .into_iter()
            .map(|tool| tool.trim().to_string())
            .filter(|tool| !tool.is_empty())
            .collect::<Vec<_>>();
        if tools.is_empty() {
            tools = default_tools_for_intent(&inferred_intent);
        }
        if tools.is_empty() {
            continue;
        }
        let mut files = step
            .files
            .into_iter()
            .map(|file| file.trim().to_string())
            .filter(|file| !file.is_empty())
            .collect::<Vec<_>>();
        files.sort();
        files.dedup();
        steps.push(PlanStep {
            step_id: Uuid::now_v7(),
            title: title.to_string(),
            intent: inferred_intent,
            tools,
            files,
            done: false,
        });
    }
    if steps.is_empty() {
        return None;
    }

    Some(Plan {
        plan_id: Uuid::now_v7(),
        version: 1,
        goal: parsed
            .goal
            .map(|goal| goal.trim().to_string())
            .filter(|goal| !goal.is_empty())
            .unwrap_or_else(|| fallback_goal.to_string()),
        assumptions: parsed.assumptions,
        steps,
        verification: if parsed.verification.is_empty() {
            vec![
                "cargo fmt --all -- --check".to_string(),
                "cargo test --workspace".to_string(),
            ]
        } else {
            parsed.verification
        },
        risk_notes: parsed.risk_notes,
    })
}

fn extract_json_snippet(text: &str) -> Option<&str> {
    if let Some(start) = text.find("```json") {
        let rest = &text[start + "```json".len()..];
        if let Some(end) = rest.find("```") {
            return Some(rest[..end].trim());
        }
    }
    if let Some(start) = text.find('{')
        && let Some(end) = text.rfind('}')
        && end > start
    {
        return Some(text[start..=end].trim());
    }
    None
}

fn infer_intent(raw_intent: &str, tools: &[String], title: &str) -> String {
    let intent = raw_intent.trim().to_ascii_lowercase();
    if !intent.is_empty() {
        return intent;
    }
    let title_lc = title.to_ascii_lowercase();
    if title_lc.contains("verify") || title_lc.contains("test") {
        return "verify".to_string();
    }
    if title_lc.contains("doc") || title_lc.contains("readme") {
        return "docs".to_string();
    }
    if title_lc.contains("git") || title_lc.contains("branch") || title_lc.contains("commit") {
        return "git".to_string();
    }
    if title_lc.contains("search") || title_lc.contains("find") || title_lc.contains("analy") {
        return "search".to_string();
    }
    if title_lc.contains("edit")
        || title_lc.contains("implement")
        || title_lc.contains("fix")
        || title_lc.contains("refactor")
    {
        return "edit".to_string();
    }
    if let Some(tool) = tools.first() {
        let base = tool.split_once(':').map_or(tool.as_str(), |(name, _)| name);
        if base.starts_with("git.") {
            return "git".to_string();
        }
        if base == "bash.run" {
            return "verify".to_string();
        }
    }
    "task".to_string()
}

fn default_tools_for_intent(intent: &str) -> Vec<String> {
    match intent {
        "search" => vec![
            "index.query".to_string(),
            "fs.grep".to_string(),
            "fs.read".to_string(),
        ],
        "git" => vec!["git.status".to_string(), "git.diff".to_string()],
        "edit" => vec!["fs.edit".to_string(), "patch.stage".to_string()],
        "docs" => vec!["fs.edit".to_string()],
        "verify" => vec!["bash.run".to_string()],
        "recover" => vec!["fs.grep".to_string(), "fs.read".to_string()],
        _ => vec!["fs.list".to_string()],
    }
}

fn parse_declared_tool(raw: &str) -> (String, Option<String>) {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return ("".to_string(), None);
    }

    let (name, arg) = if let Some((name, rest)) = trimmed.split_once(':') {
        (name.trim(), Some(rest.trim().to_string()))
    } else if trimmed.ends_with(')') {
        if let Some(open_idx) = trimmed.find('(') {
            let name = trimmed[..open_idx].trim();
            let inner = trimmed[(open_idx + 1)..(trimmed.len() - 1)].trim();
            (name, Some(inner.to_string()))
        } else {
            (trimmed, None)
        }
    } else {
        (trimmed, None)
    };

    let normalized = normalize_declared_tool_name(name);
    (normalized, arg.filter(|s| !s.is_empty()))
}

fn normalize_declared_tool_name(name: &str) -> String {
    match name.trim().to_ascii_lowercase().as_str() {
        "bash" | "shell" | "shell.run" | "run" => "bash.run".to_string(),
        "grep" | "search" => "fs.grep".to_string(),
        "read" | "read_file" | "fs.read_file" => "fs.read".to_string(),
        "write" | "write_file" | "fs.write_file" => "fs.write".to_string(),
        "edit" | "modify" => "fs.edit".to_string(),
        "list" => "fs.list".to_string(),
        "git_status" => "git.status".to_string(),
        "git_diff" => "git.diff".to_string(),
        "git_show" => "git.show".to_string(),
        other => other.to_string(),
    }
}

fn should_parallel_execute_calls(proposals: &[deepseek_core::ToolProposal]) -> bool {
    proposals.len() > 1
        && proposals
            .iter()
            .all(|proposal| is_parallel_safe_tool(&proposal.call.name))
}

fn is_parallel_safe_tool(name: &str) -> bool {
    matches!(
        name,
        "fs.list"
            | "fs.read"
            | "fs.grep"
            | "fs.glob"
            | "fs.search_rg"
            | "git.status"
            | "git.diff"
            | "git.show"
            | "index.query"
    )
}

#[derive(Debug, Clone)]
struct PromptReference {
    raw: String,
    path: String,
    start_line: Option<usize>,
    end_line: Option<usize>,
    force_dir: bool,
}

fn expand_prompt_references(
    workspace: &Path,
    prompt: &str,
    respect_gitignore: bool,
) -> Result<String> {
    let refs = extract_prompt_references(prompt);
    if refs.is_empty() {
        return Ok(prompt.to_string());
    }

    let mut extra = String::new();
    extra.push_str("\n\n[Resolved references]\n");
    for reference in refs {
        let full = workspace.join(&reference.path);
        if !full.exists() {
            extra.push_str(&format!(
                "- {} -> missing ({})\n",
                reference.raw, reference.path
            ));
            continue;
        }

        if reference.force_dir || full.is_dir() {
            let mut shown = 0usize;
            extra.push_str(&format!(
                "- {} -> directory {}\n",
                reference.raw, reference.path
            ));
            for rel in walk_workspace_files(workspace, &full, respect_gitignore, 50) {
                extra.push_str(&format!("  - {rel}\n"));
                shown += 1;
            }
            if shown >= 50 {
                extra.push_str("  - ... (truncated)\n");
            }
            continue;
        }

        let bytes = fs::read(&full)?;
        if is_binary(&bytes) {
            extra.push_str(&format!(
                "- {} -> file {} (binary, {} bytes)\n",
                reference.raw,
                reference.path,
                bytes.len()
            ));
            continue;
        }

        let text = String::from_utf8(bytes)?;
        let rendered = render_referenced_lines(&text, reference.start_line, reference.end_line);
        extra.push_str(&format!(
            "```text\n# {}\n{}\n```\n",
            reference.path, rendered
        ));
    }

    Ok(format!("{prompt}{extra}"))
}

fn extract_prompt_references(prompt: &str) -> Vec<PromptReference> {
    prompt
        .split_whitespace()
        .filter_map(parse_prompt_reference)
        .collect()
}

fn parse_prompt_reference(token: &str) -> Option<PromptReference> {
    let trimmed = token.trim();
    if !trimmed.starts_with('@') {
        return None;
    }
    let body = trimmed
        .trim_start_matches('@')
        .trim_end_matches(|ch: char| {
            ch == ',' || ch == '.' || ch == ';' || ch == ')' || ch == ']' || ch == '>'
        });
    if body.is_empty() {
        return None;
    }

    let mut path = body.to_string();
    let mut start_line = None;
    let mut end_line = None;
    let mut force_dir = false;

    if let Some(rest) = body.strip_prefix("dir:") {
        path = rest.to_string();
        force_dir = true;
    } else if let Some(rest) = body.strip_prefix("file:") {
        path = rest.to_string();
    }
    if let Some(idx) = path.rfind(':') {
        let (candidate_path, range) = path.split_at(idx);
        let range = &range[1..];
        if let Some((start, end)) = parse_line_range(range) {
            path = candidate_path.to_string();
            start_line = Some(start);
            end_line = Some(end);
        }
    }

    Some(PromptReference {
        raw: trimmed.to_string(),
        path,
        start_line,
        end_line,
        force_dir,
    })
}

fn parse_line_range(range: &str) -> Option<(usize, usize)> {
    if range.is_empty() {
        return None;
    }
    if let Some((start, end)) = range.split_once('-') {
        let start = start.parse::<usize>().ok()?;
        let end = end.parse::<usize>().ok()?;
        if start == 0 || end < start {
            return None;
        }
        return Some((start, end));
    }
    let line = range.parse::<usize>().ok()?;
    if line == 0 {
        return None;
    }
    Some((line, line))
}

fn render_referenced_lines(
    content: &str,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> String {
    let start = start_line.unwrap_or(1).max(1);
    let end = end_line.unwrap_or(start + 200).max(start);
    let mut out = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        let line_no = idx + 1;
        if line_no < start || line_no > end {
            continue;
        }
        out.push(format!("{line_no:>5}: {line}"));
        if out.len() >= 200 {
            out.push("... (truncated)".to_string());
            break;
        }
    }
    out.join("\n")
}

fn is_binary(bytes: &[u8]) -> bool {
    bytes.contains(&0)
}

fn has_ignored_component(path: &Path) -> bool {
    path.components().any(|component| {
        let value = component.as_os_str();
        value == ".git" || value == ".deepseek" || value == "target"
    })
}

fn walk_workspace_files(
    workspace: &Path,
    root: &Path,
    respect_gitignore: bool,
    limit: usize,
) -> Vec<String> {
    let mut builder = WalkBuilder::new(root);
    builder.hidden(false);
    builder.follow_links(false);
    builder.parents(respect_gitignore);
    builder.git_ignore(respect_gitignore);
    builder.git_global(respect_gitignore);
    builder.git_exclude(respect_gitignore);
    builder.require_git(false);

    let mut out = Vec::new();
    for entry in builder.build() {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Ok(rel) = path.strip_prefix(workspace) else {
            continue;
        };
        if has_ignored_component(rel) {
            continue;
        }
        out.push(rel.to_string_lossy().replace('\\', "/"));
        if out.len() >= limit {
            break;
        }
    }
    out
}

fn prompt_cache_key(model: &str, prompt: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"deepseek:");
    hasher.update(model.as_bytes());
    hasher.update(b":");
    hasher.update(prompt.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn next_off_peak_start(start_hour: u8) -> String {
    let now = Utc::now();
    let current = now.hour() as u8;
    let mut day_delta = 0_i64;
    if current >= start_hour {
        day_delta = 1;
    }
    let date = now.date_naive() + chrono::Duration::days(day_delta);
    if let Some(dt) = date.and_hms_opt(start_hour as u32, 0, 0) {
        return chrono::DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc).to_rfc3339();
    }
    now.to_rfc3339()
}

fn in_off_peak_window(hour: u8, start: u8, end: u8) -> bool {
    if start <= end {
        hour >= start && hour < end
    } else {
        hour >= start || hour < end
    }
}

fn seconds_until_off_peak_start(current_hour: u8, start_hour: u8) -> u64 {
    let now_minutes = (current_hour as u64) * 60;
    let start_minutes = (start_hour as u64) * 60;
    let minutes_until = if now_minutes <= start_minutes {
        start_minutes - now_minutes
    } else {
        (24 * 60 - now_minutes) + start_minutes
    };
    minutes_until * 60
}

fn estimate_tokens(text: &str) -> u64 {
    // Rough token estimate for local accounting and status reporting.
    (text.chars().count() as u64).div_ceil(4)
}

fn estimate_messages_tokens(messages: &[ChatMessage]) -> u64 {
    messages
        .iter()
        .map(|m| match m {
            ChatMessage::System { content } | ChatMessage::User { content } => {
                estimate_tokens(content)
            }
            ChatMessage::Assistant {
                content,
                tool_calls,
            } => {
                let c = content.as_deref().map(estimate_tokens).unwrap_or(0);
                let t: u64 = tool_calls
                    .iter()
                    .map(|tc| estimate_tokens(&tc.name) + estimate_tokens(&tc.arguments))
                    .sum();
                c + t
            }
            ChatMessage::Tool { content, .. } => estimate_tokens(content),
        })
        .sum()
}

fn summarize_tool_args(tool_name: &str, args: &serde_json::Value) -> String {
    match tool_name {
        "fs.read" | "fs.write" | "fs.edit" => args
            .get("path")
            .and_then(|v| v.as_str())
            .map(|p| format!("path={p}"))
            .unwrap_or_default(),
        "fs.glob" | "fs.grep" => args
            .get("pattern")
            .and_then(|v| v.as_str())
            .map(|p| format!("pattern={p}"))
            .unwrap_or_default(),
        "bash.run" => args
            .get("cmd")
            .and_then(|v| v.as_str())
            .map(|c| {
                if c.len() > 80 {
                    format!("cmd={}...", &c[..80])
                } else {
                    format!("cmd={c}")
                }
            })
            .unwrap_or_default(),
        "multi_edit" => args
            .get("files")
            .and_then(|v| v.as_array())
            .map(|a| format!("{} files", a.len()))
            .unwrap_or_default(),
        "web.fetch" => args
            .get("url")
            .and_then(|v| v.as_str())
            .map(|u| {
                if u.len() > 80 {
                    format!("url={}...", &u[..80])
                } else {
                    format!("url={u}")
                }
            })
            .unwrap_or_default(),
        "web.search" => args
            .get("query")
            .and_then(|v| v.as_str())
            .map(|q| format!("query={q}"))
            .unwrap_or_default(),
        "git.show" => args
            .get("spec")
            .and_then(|v| v.as_str())
            .map(|s| format!("spec={s}"))
            .unwrap_or_default(),
        "index.query" => args
            .get("q")
            .and_then(|v| v.as_str())
            .map(|q| format!("q={q}"))
            .unwrap_or_default(),
        "patch.stage" => args
            .get("unified_diff")
            .and_then(|v| v.as_str())
            .map(|d| {
                if d.len() > 100 {
                    format!("diff={}...", &d[..100])
                } else {
                    format!("diff={d}")
                }
            })
            .unwrap_or_default(),
        "patch.apply" => args
            .get("patch_id")
            .and_then(|v| v.as_str())
            .map(|id| format!("patch_id={id}"))
            .unwrap_or_default(),
        "diagnostics.check" => args
            .get("path")
            .and_then(|v| v.as_str())
            .map(|p| format!("path={p}"))
            .unwrap_or_else(|| "project".to_string()),
        "notebook.read" | "notebook.edit" => args
            .get("path")
            .and_then(|v| v.as_str())
            .map(|p| format!("path={p}"))
            .unwrap_or_default(),
        _ => String::new(),
    }
}

fn truncate_tool_output(output: &str, max_bytes: usize) -> String {
    if output.len() <= max_bytes {
        return output.to_string();
    }
    let lines: Vec<&str> = output.lines().collect();
    if lines.len() > 200 {
        let head: Vec<&str> = lines[..100].to_vec();
        let tail: Vec<&str> = lines[lines.len() - 100..].to_vec();
        format!(
            "{}\n... ({} lines omitted) ...\n{}",
            head.join("\n"),
            lines.len() - 200,
            tail.join("\n")
        )
    } else {
        // Byte-truncate at a safe char boundary
        let truncated = &output[..output.floor_char_boundary(max_bytes)];
        format!("{}... (truncated, {} bytes total)", truncated, output.len())
    }
}

fn summarize_chat_messages(messages: &[ChatMessage]) -> String {
    let mut entries: Vec<String> = Vec::new();
    for msg in messages {
        match msg {
            ChatMessage::System { .. } => {}
            ChatMessage::User { content } => {
                let truncated = if content.len() > 200 {
                    format!("{}...", &content[..content.floor_char_boundary(200)])
                } else {
                    content.clone()
                };
                entries.push(format!("- User: {truncated}"));
            }
            ChatMessage::Assistant {
                content,
                tool_calls,
            } => {
                if let Some(text) = content {
                    let truncated = if text.len() > 200 {
                        format!("{}...", &text[..text.floor_char_boundary(200)])
                    } else {
                        text.clone()
                    };
                    entries.push(format!("- Assistant: {truncated}"));
                }
                for tc in tool_calls {
                    let summary = summarize_tool_args(
                        map_tool_name(&tc.name),
                        &serde_json::from_str(&tc.arguments).unwrap_or(json!({})),
                    );
                    entries.push(format!("- Tool call: {}({})", tc.name, summary));
                }
            }
            ChatMessage::Tool { content, .. } => {
                let truncated = if content.len() > 100 {
                    format!("{}...", &content[..content.floor_char_boundary(100)])
                } else {
                    content.clone()
                };
                entries.push(format!("- Tool result: {truncated}"));
            }
        }
    }
    if entries.len() > 30 {
        let head = &entries[..15];
        let tail = &entries[entries.len() - 15..];
        format!(
            "{}\n... ({} entries omitted) ...\n{}",
            head.join("\n"),
            entries.len() - 30,
            tail.join("\n")
        )
    } else {
        entries.join("\n")
    }
}

fn subagent_request_for_task(
    task: &SubagentTask,
    root_goal: &str,
    verification: &str,
    base_model: &str,
    max_think_model: &str,
    domain: &str,
    specialization_hint: Option<&str>,
) -> LlmRequest {
    let (unit, model, max_tokens) = match task.role {
        SubagentRole::Plan => (LlmUnit::Planner, max_think_model.to_string(), 3072),
        SubagentRole::Explore => (LlmUnit::Planner, base_model.to_string(), 2048),
        SubagentRole::Task => (LlmUnit::Executor, base_model.to_string(), 2048),
        SubagentRole::Custom(_) => (LlmUnit::Executor, base_model.to_string(), 2048),
    };
    let prompt = format!(
        "You are a coding subagent.\n\
role={:?}\n\
team={}\n\
name={}\n\
task_goal={}\n\
main_goal={}\n\
verification_targets={}\n\
domain={}\n\
specialization_hint={}\n\
Return concise actionable findings only (max 6 bullet points).",
        task.role,
        task.team,
        task.name,
        task.goal,
        root_goal,
        verification,
        domain,
        specialization_hint.unwrap_or("none"),
    );
    LlmRequest {
        unit,
        prompt,
        model,
        max_tokens,
        non_urgent: false,
        images: vec![],
    }
}

#[derive(Debug, Clone)]
struct SubagentTaskMeta {
    name: String,
    goal: String,
    created_at: String,
    targets: Vec<String>,
    domain: String,
    specialization_hint: Option<String>,
    phase: usize,
    dependencies: Vec<String>,
    ownership_lane: String,
}

#[derive(Debug, Clone)]
struct SubagentExecutionLane {
    title: String,
    intent: String,
    role: SubagentRole,
    team: String,
    targets: Vec<String>,
    domain: String,
    phase: usize,
    dependencies: Vec<String>,
    ownership_lane: String,
}

fn subagent_role_for_step(step: &PlanStep) -> SubagentRole {
    match step.intent.as_str() {
        "search" => SubagentRole::Explore,
        "plan" | "recover" => SubagentRole::Plan,
        _ => SubagentRole::Task,
    }
}

fn subagent_team_for_role(role: &SubagentRole) -> &'static str {
    match role {
        SubagentRole::Explore => "explore",
        SubagentRole::Plan => "planning",
        SubagentRole::Task => "execution",
        SubagentRole::Custom(_) => "custom",
    }
}

fn plan_subagent_execution_lanes(
    steps: &[PlanStep],
    max_tasks: usize,
) -> Vec<SubagentExecutionLane> {
    let capped = max_tasks.max(1);
    let mut lanes = Vec::new();
    let mut target_last_phase: HashMap<String, usize> = HashMap::new();
    let mut target_owner: HashMap<String, String> = HashMap::new();

    for step in steps.iter().take(capped) {
        let role = subagent_role_for_step(step);
        let team = subagent_team_for_role(&role).to_string();
        let targets = subagent_targets_for_step(step);
        let domain = subagent_domain_for_step(step, &targets);
        let mut phase = 0usize;
        let mut dependencies = Vec::new();

        for target in &targets {
            for (known_target, previous_phase) in &target_last_phase {
                if !target_patterns_overlap(target, known_target) {
                    continue;
                }
                phase = phase.max(previous_phase.saturating_add(1));
                dependencies.push(format!("{known_target}@phase{}", previous_phase + 1));
                if let Some(owner) = target_owner.get(known_target)
                    && owner != &team
                {
                    dependencies.push(format!("{known_target}@owner={owner}"));
                }
            }
        }
        if targets.is_empty()
            && matches!(role, SubagentRole::Task)
            && let Some(previous_phase) = target_last_phase.values().copied().max()
        {
            phase = phase.max(previous_phase.saturating_add(1));
            dependencies.push(format!("unscoped@phase{}", previous_phase + 1));
        }
        dependencies.sort();
        dependencies.dedup();

        for target in &targets {
            target_last_phase.insert(target.clone(), phase);
            target_owner
                .entry(target.clone())
                .or_insert_with(|| team.clone());
        }

        let ownership_lane = if targets.is_empty() {
            format!("{team}:unscoped")
        } else {
            format!("{team}:{}", targets.join(","))
        };
        lanes.push(SubagentExecutionLane {
            title: step.title.clone(),
            intent: step.intent.clone(),
            role,
            team,
            targets,
            domain,
            phase,
            dependencies,
            ownership_lane,
        });
    }

    lanes.sort_by(|a, b| {
        a.phase
            .cmp(&b.phase)
            .then(a.team.cmp(&b.team))
            .then(a.title.cmp(&b.title))
    });
    lanes
}

fn target_patterns_overlap(a: &str, b: &str) -> bool {
    let normalize = |value: &str| value.trim().trim_end_matches('/').to_ascii_lowercase();
    let a = normalize(a);
    let b = normalize(b);
    if a.is_empty() || b.is_empty() {
        return false;
    }
    if a == "." || b == "." {
        return true;
    }
    if a == b {
        return true;
    }
    if a.starts_with(&(b.clone() + "/")) || b.starts_with(&(a.clone() + "/")) {
        return true;
    }
    let wildcard_prefix = |value: &str| {
        value
            .split('*')
            .next()
            .unwrap_or("")
            .trim_end_matches('/')
            .to_string()
    };
    if a.contains('*') {
        let prefix = wildcard_prefix(&a);
        if !prefix.is_empty() && (b == prefix || b.starts_with(&(prefix.clone() + "/"))) {
            return true;
        }
    }
    if b.contains('*') {
        let prefix = wildcard_prefix(&b);
        if !prefix.is_empty() && (a == prefix || a.starts_with(&(prefix.clone() + "/"))) {
            return true;
        }
    }
    false
}

fn summarize_subagent_execution_lanes(
    meta_by_run: &HashMap<Uuid, SubagentTaskMeta>,
) -> Vec<String> {
    let mut phases: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for meta in meta_by_run.values() {
        let dependencies = if meta.dependencies.is_empty() {
            "none".to_string()
        } else {
            meta.dependencies.join("|")
        };
        phases.entry(meta.phase).or_default().push(format!(
            "{} lane={} deps={}",
            meta.name, meta.ownership_lane, dependencies
        ));
    }
    phases
        .into_iter()
        .map(|(phase, mut rows)| {
            rows.sort();
            format!("subagent_phase {}: {}", phase + 1, rows.join(" ; "))
        })
        .collect()
}

fn subagent_targets_for_step(step: &PlanStep) -> Vec<String> {
    if !step.files.is_empty() {
        let mut files = step
            .files
            .iter()
            .map(|file| file.trim().to_string())
            .filter(|file| !file.is_empty())
            .collect::<Vec<_>>();
        files.sort();
        files.dedup();
        return files;
    }
    if step.intent.eq_ignore_ascii_case("docs") {
        return vec!["README.md".to_string()];
    }
    Vec::new()
}

fn subagent_domain_for_step(step: &PlanStep, targets: &[String]) -> String {
    let intent = step.intent.to_ascii_lowercase();
    if intent.contains("git") {
        return "version-control".to_string();
    }
    if intent.contains("verify") {
        return "verification".to_string();
    }
    if intent.contains("docs") {
        return "documentation".to_string();
    }
    if intent.contains("search") {
        return "code-discovery".to_string();
    }
    if let Some(domain) = targets.iter().find_map(|target| {
        let lower = target.to_ascii_lowercase();
        if lower.ends_with(".rs") {
            Some("rust-code")
        } else if lower.ends_with(".ts") || lower.ends_with(".tsx") {
            Some("typescript-code")
        } else if lower.ends_with(".js") || lower.ends_with(".jsx") {
            Some("javascript-code")
        } else if lower.ends_with(".py") {
            Some("python-code")
        } else if lower.ends_with(".md") {
            Some("documentation")
        } else if lower.ends_with(".json") || lower.ends_with(".toml") || lower.ends_with(".yaml") {
            Some("configuration")
        } else {
            None
        }
    }) {
        return domain.to_string();
    }
    "general".to_string()
}

fn subagent_specialization_key(role: &SubagentRole, domain: &str) -> String {
    format!("role={role:?}|domain={domain}")
}

fn subagent_specialization_guidance(entry: &SubagentSpecializationEntry) -> String {
    if entry.last_outcome.eq_ignore_ascii_case("success") {
        return "reuse successful decomposition pattern and keep concise evidence".to_string();
    }
    if entry.failure_count > entry.success_count {
        return "reduce branching; gather stronger evidence before proposing edits".to_string();
    }
    "maintain deterministic ordering and verification-first summaries".to_string()
}

fn compute_subagent_specialization_confidence(entry: &SubagentSpecializationEntry) -> f32 {
    let observations = entry.success_count.saturating_add(entry.failure_count) as f32;
    if observations <= f32::EPSILON {
        return 0.5;
    }
    let posterior = (entry.success_count as f32 + 1.0) / (observations + 2.0);
    let attempts_penalty = ((entry.avg_attempts - 1.0).max(0.0) / 3.0).min(1.0) * 0.20;
    (posterior - attempts_penalty).clamp(0.0, 1.0)
}

fn format_subagent_specialization_hint(entry: &SubagentSpecializationEntry) -> String {
    format!(
        "confidence={:.3}; successes={}; failures={}; avg_attempts={:.2}; next_guidance={}; last_summary={}",
        entry.confidence,
        entry.success_count,
        entry.failure_count,
        entry.avg_attempts,
        entry.next_guidance,
        if entry.last_summary.is_empty() {
            "none"
        } else {
            entry.last_summary.as_str()
        }
    )
}

fn normalize_subagent_specialization_memory(memory: &mut SubagentSpecializationMemory) {
    for entry in &mut memory.entries {
        if !entry.confidence.is_finite() || entry.confidence <= 0.0 {
            entry.confidence = compute_subagent_specialization_confidence(entry);
        }
        if entry.next_guidance.trim().is_empty() {
            entry.next_guidance = subagent_specialization_guidance(entry);
        }
        if entry.role.trim().is_empty() {
            entry.role = "Task".to_string();
        }
        if entry.domain.trim().is_empty() {
            entry.domain = "general".to_string();
        }
        if entry.avg_attempts <= 0.0 {
            entry.avg_attempts = 1.0;
        }
    }
    sort_and_prune_subagent_specialization_entries(&mut memory.entries);
}

fn sort_and_prune_subagent_specialization_entries(entries: &mut Vec<SubagentSpecializationEntry>) {
    entries.retain(|entry| {
        let observations = entry.success_count.saturating_add(entry.failure_count);
        !(observations >= 5
            && entry.failure_count > entry.success_count.saturating_add(2)
            && entry.confidence < 0.30)
    });
    entries.sort_by(|a, b| {
        b.confidence
            .total_cmp(&a.confidence)
            .then(b.success_count.cmp(&a.success_count))
            .then(a.failure_count.cmp(&b.failure_count))
            .then(b.updated_at.cmp(&a.updated_at))
    });
    entries.truncate(128);
}

fn run_subagent_delegated_tools(
    tool_host: &LocalToolHost,
    task: &SubagentTask,
    root_goal: &str,
    targets: &[String],
) -> Option<String> {
    let calls = subagent_delegated_calls(task, root_goal, targets);
    if calls.is_empty() {
        return None;
    }
    let mut lines = Vec::new();
    for call in calls {
        let mut active_call = call;
        let mut attempt = 1usize;
        loop {
            let proposal = tool_host.propose(active_call.clone());
            if !proposal.approved {
                lines.push(format!(
                    "{} {:?} delegated {} blocked (attempt={})",
                    task.team, task.role, proposal.call.name, attempt
                ));
                if let Some(retry) = subagent_retry_call(task, &proposal.call, root_goal, targets) {
                    active_call = retry;
                    attempt += 1;
                    if attempt <= 2 {
                        continue;
                    }
                }
                break;
            }
            let result = tool_host.execute(ApprovedToolCall {
                invocation_id: proposal.invocation_id,
                call: proposal.call.clone(),
            });
            let output = truncate_probe_text(
                serde_json::to_string(&result.output)
                    .unwrap_or_else(|_| "<unserializable delegated output>".to_string()),
            );
            if result.success {
                lines.push(format!(
                    "{} {:?} delegated {} => {}",
                    task.team, task.role, proposal.call.name, output
                ));
                break;
            }
            lines.push(format!(
                "{} {:?} delegated {} failed (attempt={}) => {}",
                task.team, task.role, proposal.call.name, attempt, output
            ));
            if let Some(retry) = subagent_retry_call(task, &proposal.call, root_goal, targets) {
                active_call = retry;
                attempt += 1;
                if attempt <= 2 {
                    continue;
                }
            }
            break;
        }
    }
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

fn subagent_delegated_calls(
    task: &SubagentTask,
    root_goal: &str,
    targets: &[String],
) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    if let Some(primary) = subagent_probe_call(task, root_goal) {
        calls.push(primary);
    }
    if matches!(task.role, SubagentRole::Task) {
        let note_path = format!(".deepseek/subagents/{}.md", task.run_id);
        let target_render = if targets.is_empty() {
            "none".to_string()
        } else {
            targets.join(", ")
        };
        calls.push(ToolCall {
            name: "fs.write".to_string(),
            args: json!({
                "path": note_path,
                "content": format!(
                    "subagent={}\nrole={:?}\nteam={}\nmain_goal={}\ntargets={}\n",
                    task.name, task.role, task.team, root_goal, target_render
                )
            }),
            requires_approval: true,
        });
    }
    calls.truncate(2);
    calls
}

fn subagent_retry_call(
    task: &SubagentTask,
    previous_call: &ToolCall,
    root_goal: &str,
    targets: &[String],
) -> Option<ToolCall> {
    if previous_call.name == "fs.write" || previous_call.name == "patch.apply" {
        return Some(subagent_readonly_fallback_call(task, root_goal, targets));
    }
    if previous_call.name == "bash.run" {
        return Some(ToolCall {
            name: "fs.grep".to_string(),
            args: json!({
                "pattern": plan_goal_pattern(root_goal),
                "glob": "**/*",
                "limit": 25,
                "respectGitignore": true
            }),
            requires_approval: false,
        });
    }
    None
}

fn subagent_readonly_fallback_call(
    task: &SubagentTask,
    root_goal: &str,
    targets: &[String],
) -> ToolCall {
    if let Some(path) = targets.first() {
        return ToolCall {
            name: "fs.read".to_string(),
            args: json!({"path": path}),
            requires_approval: false,
        };
    }
    let pattern = if task.goal.trim().is_empty() {
        plan_goal_pattern(root_goal)
    } else {
        plan_goal_pattern(&task.goal)
    };
    ToolCall {
        name: "fs.grep".to_string(),
        args: json!({
            "pattern": pattern,
            "glob": "**/*",
            "limit": 20,
            "respectGitignore": true
        }),
        requires_approval: false,
    }
}

fn summarize_subagent_merge_arbitration(
    results: &[deepseek_subagent::SubagentResult],
    targets_by_run: &HashMap<Uuid, Vec<String>>,
) -> Vec<String> {
    let mut by_target: HashMap<String, Vec<&deepseek_subagent::SubagentResult>> = HashMap::new();
    for result in results {
        let targets = targets_by_run
            .get(&result.run_id)
            .cloned()
            .unwrap_or_default();
        for target in targets {
            if target.trim().is_empty() {
                continue;
            }
            by_target.entry(target).or_default().push(result);
        }
    }
    let mut notes = Vec::new();
    let mut targets = by_target.keys().cloned().collect::<Vec<_>>();
    targets.sort();
    for target in targets {
        let Some(candidates) = by_target.get(&target) else {
            continue;
        };
        if candidates.len() <= 1 {
            continue;
        }
        let mut ordered = candidates.clone();
        ordered.sort_by(|a, b| {
            subagent_arbitration_score(b, &target)
                .total_cmp(&subagent_arbitration_score(a, &target))
                .then(
                    subagent_arbitration_priority(&a.role)
                        .cmp(&subagent_arbitration_priority(&b.role)),
                )
                .then(a.attempts.cmp(&b.attempts))
                .then(a.name.cmp(&b.name))
                .then(a.run_id.cmp(&b.run_id))
        });
        let winner = ordered[0];
        let contenders = ordered
            .iter()
            .map(|candidate| {
                format!(
                    "{}::{:?}({}) score={:.3}",
                    candidate.team,
                    candidate.role,
                    candidate.name,
                    subagent_arbitration_score(candidate, &target)
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let winner_score = subagent_arbitration_score(winner, &target);
        notes.push(format!(
            "merge_arbitration target={} contenders=[{}] winner={}::{:?}({}) winner_score={:.3} rationale={}",
            target,
            contenders,
            winner.team,
            winner.role,
            winner.name,
            winner_score,
            subagent_arbitration_rationale(winner, &target)
        ));
    }
    notes
}

fn subagent_arbitration_priority(role: &SubagentRole) -> u8 {
    match role {
        SubagentRole::Task => 0,
        SubagentRole::Plan => 1,
        SubagentRole::Explore => 2,
        SubagentRole::Custom(_) => 3,
    }
}

fn subagent_arbitration_score(result: &deepseek_subagent::SubagentResult, target: &str) -> f32 {
    let mut score = if result.success { 0.7 } else { 0.15 };
    score += match result.role {
        SubagentRole::Task => 0.20,
        SubagentRole::Plan => 0.14,
        SubagentRole::Explore => 0.08,
        SubagentRole::Custom(_) => 0.15,
    };
    let output = result.output.to_ascii_lowercase();
    let target_lc = target.to_ascii_lowercase();
    if !target_lc.is_empty() && output.contains(&target_lc) {
        score += 0.12;
    }
    if output.contains("verify") || output.contains("test") {
        score += 0.05;
    }
    if output.contains("blocked") || output.contains("failed") {
        score -= 0.10;
    }
    score -= (result.attempts.saturating_sub(1) as f32) * 0.04;
    score.clamp(0.0, 1.0)
}

fn subagent_arbitration_rationale(
    result: &deepseek_subagent::SubagentResult,
    target: &str,
) -> String {
    let mut signals = Vec::new();
    if result.success {
        signals.push("success");
    } else {
        signals.push("failed");
    }
    if result.attempts <= 1 {
        signals.push("single-attempt");
    } else {
        signals.push("retried");
    }
    let output = result.output.to_ascii_lowercase();
    if output.contains(&target.to_ascii_lowercase()) {
        signals.push("mentions-target");
    }
    if output.contains("verify") || output.contains("test") {
        signals.push("has-verification-signal");
    }
    format!("{} role={:?}", signals.join("+"), result.role)
}

fn subagent_probe_call(task: &SubagentTask, root_goal: &str) -> Option<ToolCall> {
    match task.role {
        SubagentRole::Explore => Some(ToolCall {
            name: "index.query".to_string(),
            args: json!({"q": root_goal, "top_k": 8}),
            requires_approval: false,
        }),
        SubagentRole::Plan => Some(ToolCall {
            name: "fs.grep".to_string(),
            args: json!({
                "pattern": plan_goal_pattern(root_goal),
                "glob": "**/*",
                "limit": 20,
                "respectGitignore": true
            }),
            requires_approval: false,
        }),
        SubagentRole::Task => Some(ToolCall {
            name: "git.status".to_string(),
            args: json!({}),
            requires_approval: false,
        }),
        SubagentRole::Custom(_) => None,
    }
}

fn truncate_probe_text(text: String) -> String {
    const MAX_CHARS: usize = 480;
    if text.chars().count() <= MAX_CHARS {
        return text;
    }
    let head = text.chars().take(MAX_CHARS).collect::<String>();
    format!("{head}...")
}

fn plan_goal_pattern(goal: &str) -> String {
    let mut terms = Vec::new();
    let mut current = String::new();
    for ch in goal.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            current.push(ch.to_ascii_lowercase());
            continue;
        }
        if current.len() >= 4 {
            terms.push(current.clone());
        }
        current.clear();
    }
    if current.len() >= 4 {
        terms.push(current);
    }
    terms.sort();
    terms.dedup();
    if terms.is_empty() {
        return "TODO|FIXME|panic|error".to_string();
    }
    terms.into_iter().take(4).collect::<Vec<_>>().join("|")
}

fn summarize_transcript(transcript: &[String], max_lines: usize) -> String {
    let mut lines = transcript
        .iter()
        .rev()
        .take(max_lines)
        .cloned()
        .collect::<Vec<_>>();
    lines.reverse();
    let rendered = lines
        .into_iter()
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.len() > 240 {
                format!("- {}...", &trimmed[..240])
            } else {
                format!("- {trimmed}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("Auto-compacted transcript summary:\n{rendered}")
}

fn summarize_subagent_notes(notes: &[String]) -> String {
    notes
        .iter()
        .flat_map(|note| note.lines())
        .take(12)
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.len() > 240 {
                format!("- {}...", &trimmed[..240])
            } else {
                format!("- {trimmed}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn augment_goal_with_subagent_notes(goal: &str, notes: &[String]) -> String {
    if notes.is_empty() {
        return goal.to_string();
    }
    let joined = notes
        .iter()
        .flat_map(|note| note.lines())
        .take(6)
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" | ");
    if joined.is_empty() {
        goal.to_string()
    } else {
        format!("{goal} [subagent_findings: {joined}]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_prompt_references_with_optional_line_ranges() {
        let refs = extract_prompt_references("inspect @src/main.rs:10-20 and @README.md");
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].path, "src/main.rs");
        assert_eq!(refs[0].start_line, Some(10));
        assert_eq!(refs[0].end_line, Some(20));
        assert_eq!(refs[1].path, "README.md");
        assert_eq!(refs[1].start_line, None);
    }

    #[test]
    fn expands_file_reference_context() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-agent-ref-test-{}", Uuid::now_v7()));
        fs::create_dir_all(workspace.join("src")).expect("workspace");
        fs::write(
            workspace.join("src/lib.rs"),
            "fn alpha() {}\nfn beta() {}\n",
        )
        .expect("seed");

        let expanded =
            expand_prompt_references(&workspace, "review @src/lib.rs:2-2", true).expect("expand");
        assert!(expanded.contains("[Resolved references]"));
        assert!(expanded.contains("2: fn beta() {}"));
    }

    #[test]
    fn parses_file_and_dir_reference_prefixes() {
        let refs = extract_prompt_references("look at @file:src/main.rs:4 and @dir:crates");
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].path, "src/main.rs");
        assert_eq!(refs[0].start_line, Some(4));
        assert_eq!(refs[1].path, "crates");
        assert!(refs[1].force_dir);
    }

    #[test]
    fn expands_dir_reference_with_gitignore_respect() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-agent-dir-ref-test-{}", Uuid::now_v7()));
        fs::create_dir_all(workspace.join("src")).expect("workspace");
        fs::create_dir_all(workspace.join("target")).expect("target");
        fs::write(workspace.join("src/lib.rs"), "fn alpha() {}\n").expect("seed");
        fs::write(workspace.join("target/build.log"), "ignore me\n").expect("seed");

        let expanded = expand_prompt_references(&workspace, "scan @dir:.", true).expect("expand");
        assert!(expanded.contains("src/lib.rs"));
        assert!(!expanded.contains("target/build.log"));
    }

    #[test]
    fn prompt_cache_key_is_stable() {
        let a = prompt_cache_key("deepseek-chat", "hello");
        let b = prompt_cache_key("deepseek-chat", "hello");
        let c = prompt_cache_key("deepseek-chat", "hello!");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn computes_off_peak_window_and_delay() {
        assert!(in_off_peak_window(1, 0, 6));
        assert!(!in_off_peak_window(12, 0, 6));
        assert_eq!(seconds_until_off_peak_start(12, 0), 43_200);
        assert_eq!(seconds_until_off_peak_start(2, 3), 3_600);
    }

    #[test]
    fn summarizes_transcript_with_line_cap() {
        let transcript = (0..50)
            .map(|idx| format!("user: line {idx}"))
            .collect::<Vec<_>>();
        let summary = summarize_transcript(&transcript, 5);
        assert!(summary.contains("line 45"));
        assert!(!summary.contains("line 1"));
    }

    #[test]
    fn plan_goal_pattern_uses_meaningful_terms() {
        let pattern = plan_goal_pattern("Refactor planner execution for git status and retries");
        assert!(pattern.contains("refactor"));
        assert!(pattern.contains("planner"));
    }

    #[test]
    fn declared_tool_mapping_prefers_step_files_for_reads() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-agent-tool-map-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let engine = AgentEngine::new(&workspace).expect("engine");
        let step = PlanStep {
            step_id: Uuid::now_v7(),
            title: "Read target file".to_string(),
            intent: "search".to_string(),
            tools: vec!["fs.read".to_string()],
            files: vec!["src/main.rs".to_string()],
            done: false,
        };
        let plan = Plan {
            plan_id: Uuid::now_v7(),
            version: 1,
            goal: "inspect".to_string(),
            assumptions: vec![],
            steps: vec![step.clone()],
            verification: vec![],
            risk_notes: vec![],
        };
        let calls = engine.calls_for_step(&step, &plan.goal);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "fs.read");
        assert_eq!(calls[0].args["path"], "src/main.rs");
    }

    #[test]
    fn declared_tools_generate_multiple_calls() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-agent-multicall-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let engine = AgentEngine::new(&workspace).expect("engine");
        let step = PlanStep {
            step_id: Uuid::now_v7(),
            title: "Explore".to_string(),
            intent: "search".to_string(),
            tools: vec![
                "index.query".to_string(),
                "fs.grep".to_string(),
                "git.status".to_string(),
            ],
            files: vec![],
            done: false,
        };
        let plan = Plan {
            plan_id: Uuid::now_v7(),
            version: 1,
            goal: "router thresholds".to_string(),
            assumptions: vec![],
            steps: vec![step.clone()],
            verification: vec![],
            risk_notes: vec![],
        };
        let calls = engine.calls_for_step(&step, &plan.goal);
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].name, "index.query");
        assert_eq!(calls[1].name, "fs.grep");
        assert_eq!(calls[2].name, "git.status");
    }

    #[test]
    fn declared_tools_support_suffix_syntax() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-agent-suffix-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let engine = AgentEngine::new(&workspace).expect("engine");
        let step = PlanStep {
            step_id: Uuid::now_v7(),
            title: "Run precise checks".to_string(),
            intent: "verify".to_string(),
            tools: vec![
                "bash.run:cargo test -p deepseek-agent".to_string(),
                "git.show:HEAD~1".to_string(),
                "fs.grep:router|planner".to_string(),
            ],
            files: vec![],
            done: false,
        };
        let plan = Plan {
            plan_id: Uuid::now_v7(),
            version: 1,
            goal: "verify runtime".to_string(),
            assumptions: vec![],
            steps: vec![step.clone()],
            verification: vec![],
            risk_notes: vec![],
        };
        let calls = engine.calls_for_step(&step, &plan.goal);
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].name, "bash.run");
        assert_eq!(calls[0].args["cmd"], "cargo test -p deepseek-agent");
        assert_eq!(calls[1].name, "git.show");
        assert_eq!(calls[1].args["spec"], "HEAD~1");
        assert_eq!(calls[2].name, "fs.grep");
        assert_eq!(calls[2].args["pattern"], "router|planner");
    }

    #[test]
    fn subagent_request_uses_reasoner_for_plan_role() {
        let task = SubagentTask {
            run_id: Uuid::now_v7(),
            name: "planner".to_string(),
            goal: "decompose".to_string(),
            role: SubagentRole::Plan,
            team: "planning".to_string(),
            read_only_fallback: false,
        };
        let req = subagent_request_for_task(
            &task,
            "improve runtime",
            "cargo test --workspace",
            "deepseek-chat",
            "deepseek-reasoner",
            "rust-code",
            Some("reuse prior decomposition strategy"),
        );
        assert!(matches!(req.unit, LlmUnit::Planner));
        assert_eq!(req.model, "deepseek-reasoner");
        assert!(req.prompt.contains("main_goal=improve runtime"));
        assert!(req.prompt.contains("domain=rust-code"));
    }

    #[test]
    fn parse_plan_infers_intent_and_default_tools() {
        let text = r#"
```json
{
  "goal": "stabilize planner",
  "steps": [
    { "title": "Search for failures", "intent": "", "tools": [], "files": [] },
    { "title": "Run verification", "intent": "", "tools": ["bash.run:cargo test -p deepseek-agent"], "files": [] }
  ],
  "verification": []
}
```
"#;
        let plan = parse_plan_from_llm(text, "fallback").expect("plan");
        assert_eq!(plan.goal, "stabilize planner");
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[0].intent, "search");
        assert_eq!(
            plan.steps[0].tools,
            vec!["index.query", "fs.grep", "fs.read"]
        );
        assert_eq!(
            plan.steps[1].tools,
            vec!["bash.run:cargo test -p deepseek-agent"]
        );
        assert!(!plan.verification.is_empty());
    }

    #[test]
    fn parse_plan_discards_empty_steps() {
        let text = r#"
{"goal":"x","steps":[{"title":"   ","intent":"","tools":[],"files":[]}]}
"#;
        assert!(parse_plan_from_llm(text, "fallback").is_none());
    }

    #[test]
    fn summarize_subagent_notes_limits_lines() {
        let notes = vec![
            "one\ntwo\nthree".to_string(),
            "four\nfive\nsix\nseven\neight\nnine\nten\neleven\ntwelve\nthirteen".to_string(),
        ];
        let summary = summarize_subagent_notes(&notes);
        let lines = summary.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 12);
        assert!(summary.contains("one"));
        assert!(!summary.contains("thirteen"));
    }

    #[test]
    fn parse_declared_tool_supports_aliases_and_paren_args() {
        let (name, arg) = parse_declared_tool("bash(cargo test --workspace)");
        assert_eq!(name, "bash.run");
        assert_eq!(arg.as_deref(), Some("cargo test --workspace"));

        let (name, arg) = parse_declared_tool("read:src/main.rs");
        assert_eq!(name, "fs.read");
        assert_eq!(arg.as_deref(), Some("src/main.rs"));

        let (name, arg) = parse_declared_tool("git_show(HEAD~2)");
        assert_eq!(name, "git.show");
        assert_eq!(arg.as_deref(), Some("HEAD~2"));
    }

    #[test]
    fn plan_revision_prompt_contains_context() {
        let plan = Plan {
            plan_id: Uuid::now_v7(),
            version: 2,
            goal: "Improve runtime".to_string(),
            assumptions: vec!["workspace writable".to_string()],
            steps: vec![PlanStep {
                step_id: Uuid::now_v7(),
                title: "Inspect".to_string(),
                intent: "search".to_string(),
                tools: vec!["index.query".to_string()],
                files: vec!["src/lib.rs".to_string()],
                done: false,
            }],
            verification: vec!["cargo test -p deepseek-agent".to_string()],
            risk_notes: vec![],
        };
        let prompt = build_plan_revision_prompt("Fix planner", &plan, 2, "approval required");
        assert!(prompt.contains("Fix planner"));
        assert!(prompt.contains("approval required"));
        assert!(prompt.contains("\"version\": 2"));
    }

    #[test]
    fn plan_quality_detects_missing_depth_and_verification() {
        let plan = Plan {
            plan_id: Uuid::now_v7(),
            version: 1,
            goal: "implement robust retry handling".to_string(),
            assumptions: vec![],
            steps: vec![PlanStep {
                step_id: Uuid::now_v7(),
                title: "Inspect retry code".to_string(),
                intent: "search".to_string(),
                tools: vec!["fs.read".to_string()],
                files: vec!["src/retry.rs".to_string()],
                done: false,
            }],
            verification: Vec::new(),
            risk_notes: vec![],
        };
        let report = assess_plan_quality(
            &plan,
            "Implement retry handling and add verification commands for reliability.",
        );
        assert!(!report.acceptable);
        assert!(report.score < 0.65);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.contains("verification"))
        );
        assert!(report.issues.iter().any(|issue| issue.contains("steps")));
    }

    #[test]
    fn plan_quality_repair_prompt_includes_reported_issues() {
        let plan = Plan {
            plan_id: Uuid::now_v7(),
            version: 1,
            goal: "stabilize planner".to_string(),
            assumptions: vec![],
            steps: vec![PlanStep {
                step_id: Uuid::now_v7(),
                title: "Do task".to_string(),
                intent: "task".to_string(),
                tools: vec!["fs.list".to_string()],
                files: vec![],
                done: false,
            }],
            verification: vec![],
            risk_notes: vec![],
        };
        let report = PlanQualityReport {
            acceptable: false,
            score: 0.42,
            issues: vec!["verification is empty".to_string()],
        };
        let prompt = build_plan_quality_repair_prompt("Fix planner reliability", &plan, &report);
        assert!(prompt.contains("Fix planner reliability"));
        assert!(prompt.contains("verification is empty"));
        assert!(prompt.contains("Quality score: 0.42"));
    }

    #[test]
    fn feedback_alignment_flags_missing_failed_command_coverage() {
        let plan = Plan {
            plan_id: Uuid::now_v7(),
            version: 1,
            goal: "stabilize".to_string(),
            assumptions: vec![],
            steps: vec![PlanStep {
                step_id: Uuid::now_v7(),
                title: "Inspect logs".to_string(),
                intent: "search".to_string(),
                tools: vec!["fs.grep".to_string()],
                files: vec![],
                done: false,
            }],
            verification: vec!["cargo fmt --all -- --check".to_string()],
            risk_notes: vec![],
        };
        let feedback = vec![VerificationRunRecord {
            command: "pytest -k router_smoke".to_string(),
            success: false,
            output: "test failure".to_string(),
            run_at: Utc::now().to_rfc3339(),
        }];
        let report = assess_plan_feedback_alignment(&plan, &feedback);
        assert!(!report.acceptable);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.contains("failing command"))
        );
    }

    #[test]
    fn delegated_calls_for_task_include_bounded_write_step() {
        let task = SubagentTask {
            run_id: Uuid::now_v7(),
            name: "apply-fix".to_string(),
            goal: "edit".to_string(),
            role: SubagentRole::Task,
            team: "execution".to_string(),
            read_only_fallback: false,
        };
        let calls = subagent_delegated_calls(
            &task,
            "fix runtime",
            &["src/lib.rs".to_string(), "src/main.rs".to_string()],
        );
        assert!(!calls.is_empty());
        assert!(calls.len() <= 2);
        assert!(calls.iter().any(|call| call.name == "fs.write"));
    }

    #[test]
    fn merge_arbitration_reports_conflicting_targets() {
        let shared = "src/lib.rs".to_string();
        let a = deepseek_subagent::SubagentResult {
            run_id: Uuid::now_v7(),
            name: "planner".to_string(),
            role: SubagentRole::Plan,
            team: "planning".to_string(),
            attempts: 1,
            success: true,
            output: "a".to_string(),
            error: None,
            used_read_only_fallback: false,
        };
        let b = deepseek_subagent::SubagentResult {
            run_id: Uuid::now_v7(),
            name: "executor".to_string(),
            role: SubagentRole::Task,
            team: "execution".to_string(),
            attempts: 1,
            success: true,
            output: "b".to_string(),
            error: None,
            used_read_only_fallback: false,
        };
        let mut targets = HashMap::new();
        targets.insert(a.run_id, vec![shared.clone()]);
        targets.insert(b.run_id, vec![shared.clone()]);
        let notes = summarize_subagent_merge_arbitration(&[a, b], &targets);
        assert_eq!(notes.len(), 1);
        assert!(notes[0].contains("merge_arbitration"));
        assert!(notes[0].contains(&shared));
    }

    #[test]
    fn augment_goal_uses_subagent_notes() {
        let goal = "stabilize runtime";
        let notes = vec!["first finding".to_string(), "second finding".to_string()];
        let augmented = augment_goal_with_subagent_notes(goal, &notes);
        assert!(augmented.contains("stabilize runtime"));
        assert!(augmented.contains("subagent_findings"));
        assert!(augmented.contains("first finding"));
    }

    #[test]
    fn parallel_execution_only_for_read_only_calls() {
        use deepseek_core::{ToolCall, ToolProposal};
        let safe = vec![
            ToolProposal {
                invocation_id: Uuid::now_v7(),
                call: ToolCall {
                    name: "fs.grep".to_string(),
                    args: json!({}),
                    requires_approval: false,
                },
                approved: true,
            },
            ToolProposal {
                invocation_id: Uuid::now_v7(),
                call: ToolCall {
                    name: "git.status".to_string(),
                    args: json!({}),
                    requires_approval: false,
                },
                approved: true,
            },
        ];
        assert!(should_parallel_execute_calls(&safe));

        let mixed = vec![
            ToolProposal {
                invocation_id: Uuid::now_v7(),
                call: ToolCall {
                    name: "fs.grep".to_string(),
                    args: json!({}),
                    requires_approval: false,
                },
                approved: true,
            },
            ToolProposal {
                invocation_id: Uuid::now_v7(),
                call: ToolCall {
                    name: "fs.edit".to_string(),
                    args: json!({}),
                    requires_approval: false,
                },
                approved: true,
            },
        ];
        assert!(!should_parallel_execute_calls(&mixed));
    }

    #[test]
    fn subagent_probe_call_maps_roles_to_read_only_tools() {
        let run_id = Uuid::now_v7();
        let explore = SubagentTask {
            run_id,
            name: "scan".to_string(),
            goal: "search".to_string(),
            role: SubagentRole::Explore,
            team: "explore".to_string(),
            read_only_fallback: false,
        };
        let plan = SubagentTask {
            role: SubagentRole::Plan,
            ..explore.clone()
        };
        let task = SubagentTask {
            role: SubagentRole::Task,
            ..explore
        };
        let explore_probe = subagent_probe_call(&plan, "planner").expect("plan probe");
        assert_eq!(explore_probe.name, "fs.grep");
        assert_eq!(explore_probe.args["pattern"], "planner");
        assert_eq!(explore_probe.args["glob"], "**/*");
        assert_eq!(explore_probe.args["limit"], 20);
        let explore_role_probe = subagent_probe_call(
            &SubagentTask {
                role: SubagentRole::Explore,
                ..plan.clone()
            },
            "planner",
        )
        .expect("explore probe");
        assert_eq!(explore_role_probe.name, "index.query");
        assert_eq!(explore_role_probe.args["q"], "planner");
        assert_eq!(
            subagent_probe_call(&task, "planner")
                .map(|call| call.name)
                .as_deref(),
            Some("git.status")
        );
    }

    #[test]
    fn strategy_scores_penalize_failures_and_prune_chronic_entries() {
        assert!(compute_strategy_score(6, 1) > compute_strategy_score(1, 6));

        let mut entries = vec![
            PlannerStrategyEntry {
                key: "stable".to_string(),
                goal_excerpt: "keep stable behavior".to_string(),
                strategy_summary: "good".to_string(),
                verification: vec!["cargo test".to_string()],
                success_count: 5,
                failure_count: 1,
                score: compute_strategy_score(5, 1),
                last_outcome: "success".to_string(),
                updated_at: Utc::now().to_rfc3339(),
            },
            PlannerStrategyEntry {
                key: "noisy".to_string(),
                goal_excerpt: "broken path".to_string(),
                strategy_summary: "bad".to_string(),
                verification: vec!["cargo test".to_string()],
                success_count: 0,
                failure_count: 5,
                score: compute_strategy_score(0, 5),
                last_outcome: "failure".to_string(),
                updated_at: Utc::now().to_rfc3339(),
            },
        ];
        sort_and_prune_strategy_entries(&mut entries);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].key, "stable");
    }

    #[test]
    fn subagent_retry_downgrades_blocked_write_to_read_only_fallback() {
        let task = SubagentTask {
            run_id: Uuid::now_v7(),
            name: "apply-fix".to_string(),
            goal: "edit".to_string(),
            role: SubagentRole::Task,
            team: "execution".to_string(),
            read_only_fallback: false,
        };
        let write_call = deepseek_core::ToolCall {
            name: "fs.write".to_string(),
            args: json!({"path": ".deepseek/subagents/test.md", "content": "x"}),
            requires_approval: true,
        };
        let fallback = subagent_retry_call(
            &task,
            &write_call,
            "stabilize runtime",
            &["src/lib.rs".to_string()],
        )
        .expect("fallback");
        assert_eq!(fallback.name, "fs.read");
        assert_eq!(fallback.args["path"], "src/lib.rs");
    }

    #[test]
    fn arbitration_score_prefers_successful_target_specific_results() {
        let target = "src/lib.rs";
        let weak = deepseek_subagent::SubagentResult {
            run_id: Uuid::now_v7(),
            name: "explore".to_string(),
            role: SubagentRole::Explore,
            team: "explore".to_string(),
            attempts: 2,
            success: false,
            output: "blocked".to_string(),
            error: Some("approval".to_string()),
            used_read_only_fallback: false,
        };
        let strong = deepseek_subagent::SubagentResult {
            run_id: Uuid::now_v7(),
            name: "executor".to_string(),
            role: SubagentRole::Task,
            team: "execution".to_string(),
            attempts: 1,
            success: true,
            output: "updated src/lib.rs and added verification test".to_string(),
            error: None,
            used_read_only_fallback: false,
        };
        assert!(
            subagent_arbitration_score(&strong, target) > subagent_arbitration_score(&weak, target)
        );
    }

    #[test]
    fn objective_confidence_tracks_success_and_failures() {
        let high = ObjectiveOutcomeEntry {
            key: "router|retry".to_string(),
            goal_excerpt: "stabilize router retries".to_string(),
            success_count: 7,
            failure_count: 1,
            execution_failure_count: 1,
            verification_failure_count: 0,
            avg_step_count: 4.0,
            avg_failure_count: 0.2,
            confidence: 0.0,
            last_outcome: "success".to_string(),
            last_failure_summary: "none".to_string(),
            next_focus: String::new(),
            updated_at: Utc::now().to_rfc3339(),
        };
        let low = ObjectiveOutcomeEntry {
            key: "router|retry".to_string(),
            goal_excerpt: "stabilize router retries".to_string(),
            success_count: 1,
            failure_count: 6,
            execution_failure_count: 4,
            verification_failure_count: 3,
            avg_step_count: 2.0,
            avg_failure_count: 2.2,
            confidence: 0.0,
            last_outcome: "failure".to_string(),
            last_failure_summary: "verification".to_string(),
            next_focus: String::new(),
            updated_at: Utc::now().to_rfc3339(),
        };
        assert!(compute_objective_confidence(&high) > compute_objective_confidence(&low));
    }

    #[test]
    fn objective_entries_prune_chronic_low_confidence_items() {
        let mut entries = vec![
            ObjectiveOutcomeEntry {
                key: "stable".to_string(),
                goal_excerpt: "stable objective".to_string(),
                success_count: 5,
                failure_count: 1,
                execution_failure_count: 1,
                verification_failure_count: 0,
                avg_step_count: 4.0,
                avg_failure_count: 0.5,
                confidence: 0.9,
                last_outcome: "success".to_string(),
                last_failure_summary: "none".to_string(),
                next_focus: "keep".to_string(),
                updated_at: Utc::now().to_rfc3339(),
            },
            ObjectiveOutcomeEntry {
                key: "noisy".to_string(),
                goal_excerpt: "noisy objective".to_string(),
                success_count: 0,
                failure_count: 6,
                execution_failure_count: 4,
                verification_failure_count: 4,
                avg_step_count: 2.0,
                avg_failure_count: 2.8,
                confidence: 0.1,
                last_outcome: "failure".to_string(),
                last_failure_summary: "many".to_string(),
                next_focus: "repair".to_string(),
                updated_at: Utc::now().to_rfc3339(),
            },
        ];
        sort_and_prune_objective_entries(&mut entries);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].key, "stable");
    }

    #[test]
    fn objective_outcome_format_includes_focus_and_failure_summary() {
        let entry = ObjectiveOutcomeEntry {
            key: "router".to_string(),
            goal_excerpt: "router objective".to_string(),
            success_count: 2,
            failure_count: 1,
            execution_failure_count: 1,
            verification_failure_count: 0,
            avg_step_count: 3.0,
            avg_failure_count: 1.0,
            confidence: 0.7,
            last_outcome: "failure".to_string(),
            last_failure_summary: "execution_failures=1".to_string(),
            next_focus: "add checkpoints".to_string(),
            updated_at: Utc::now().to_rfc3339(),
        };
        let rendered = format_objective_outcomes(&[entry]);
        assert!(rendered.contains("focus=\"add checkpoints\""));
        assert!(rendered.contains("last_failure=\"execution_failures=1\""));
    }

    #[test]
    fn long_horizon_quality_requires_checkpoint_guards_for_risky_objectives() {
        let plan = Plan {
            plan_id: Uuid::now_v7(),
            version: 1,
            goal: "migrate service end-to-end".to_string(),
            assumptions: vec![],
            steps: vec![
                PlanStep {
                    step_id: Uuid::now_v7(),
                    title: "Phase 1 discover code".to_string(),
                    intent: "search".to_string(),
                    tools: vec!["fs.grep".to_string()],
                    files: vec![],
                    done: false,
                },
                PlanStep {
                    step_id: Uuid::now_v7(),
                    title: "Phase 2 apply edits".to_string(),
                    intent: "edit".to_string(),
                    tools: vec!["fs.edit".to_string()],
                    files: vec![],
                    done: false,
                },
                PlanStep {
                    step_id: Uuid::now_v7(),
                    title: "Phase 3 verify".to_string(),
                    intent: "verify".to_string(),
                    tools: vec!["bash.run".to_string()],
                    files: vec![],
                    done: false,
                },
            ],
            verification: vec!["cargo test --workspace".to_string()],
            risk_notes: vec![],
        };
        let objective = ObjectiveOutcomeEntry {
            key: "migrate|service".to_string(),
            goal_excerpt: "migrate service".to_string(),
            success_count: 1,
            failure_count: 4,
            execution_failure_count: 2,
            verification_failure_count: 2,
            avg_step_count: 3.0,
            avg_failure_count: 1.6,
            confidence: 0.35,
            last_outcome: "failure".to_string(),
            last_failure_summary: "verification_failures=2".to_string(),
            next_focus: String::new(),
            updated_at: Utc::now().to_rfc3339(),
        };
        let report = assess_plan_long_horizon_quality(
            &plan,
            "Plan a large end-to-end migration across services",
            &[objective],
        );
        assert!(!report.acceptable);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.contains("checkpoint/rollback"))
        );
    }

    #[test]
    fn subagent_domain_detection_prefers_file_types_and_intent() {
        let step = PlanStep {
            step_id: Uuid::now_v7(),
            title: "Update docs".to_string(),
            intent: "docs".to_string(),
            tools: vec!["fs.edit".to_string()],
            files: vec!["README.md".to_string()],
            done: false,
        };
        let targets = subagent_targets_for_step(&step);
        assert_eq!(subagent_domain_for_step(&step, &targets), "documentation");

        let rust_step = PlanStep {
            step_id: Uuid::now_v7(),
            title: "Edit engine".to_string(),
            intent: "task".to_string(),
            tools: vec!["fs.edit".to_string()],
            files: vec!["src/lib.rs".to_string()],
            done: false,
        };
        let rust_targets = subagent_targets_for_step(&rust_step);
        assert_eq!(
            subagent_domain_for_step(&rust_step, &rust_targets),
            "rust-code"
        );
    }

    #[test]
    fn specialization_confidence_penalizes_failures_and_retries() {
        let good = SubagentSpecializationEntry {
            key: "role=Task|domain=rust-code".to_string(),
            role: "Task".to_string(),
            domain: "rust-code".to_string(),
            success_count: 6,
            failure_count: 1,
            avg_attempts: 1.1,
            confidence: 0.0,
            last_outcome: "success".to_string(),
            last_summary: "ok".to_string(),
            next_guidance: String::new(),
            updated_at: Utc::now().to_rfc3339(),
        };
        let poor = SubagentSpecializationEntry {
            key: "role=Task|domain=rust-code".to_string(),
            role: "Task".to_string(),
            domain: "rust-code".to_string(),
            success_count: 1,
            failure_count: 6,
            avg_attempts: 2.8,
            confidence: 0.0,
            last_outcome: "failure".to_string(),
            last_summary: "blocked".to_string(),
            next_guidance: String::new(),
            updated_at: Utc::now().to_rfc3339(),
        };
        assert!(
            compute_subagent_specialization_confidence(&good)
                > compute_subagent_specialization_confidence(&poor)
        );
    }

    #[test]
    fn subagent_lane_planner_serializes_shared_target_dependencies() {
        let steps = vec![
            PlanStep {
                step_id: Uuid::now_v7(),
                title: "Plan api changes".to_string(),
                intent: "plan".to_string(),
                tools: vec!["fs.read".to_string()],
                files: vec!["src/api.rs".to_string()],
                done: false,
            },
            PlanStep {
                step_id: Uuid::now_v7(),
                title: "Apply api edits".to_string(),
                intent: "task".to_string(),
                tools: vec!["fs.edit".to_string()],
                files: vec!["src/api.rs".to_string()],
                done: false,
            },
        ];
        let lanes = plan_subagent_execution_lanes(&steps, 8);
        assert_eq!(lanes.len(), 2);
        assert_eq!(lanes[0].phase, 0);
        assert_eq!(lanes[1].phase, 1);
        assert!(
            lanes[1]
                .dependencies
                .iter()
                .any(|dep| dep.contains("src/api.rs@phase1"))
        );
    }

    #[test]
    fn subagent_lane_planner_serializes_overlapping_directory_targets() {
        let steps = vec![
            PlanStep {
                step_id: Uuid::now_v7(),
                title: "Task touching src tree".to_string(),
                intent: "task".to_string(),
                tools: vec!["fs.edit".to_string()],
                files: vec!["src".to_string()],
                done: false,
            },
            PlanStep {
                step_id: Uuid::now_v7(),
                title: "Task touching file in src".to_string(),
                intent: "task".to_string(),
                tools: vec!["fs.edit".to_string()],
                files: vec!["src/main.rs".to_string()],
                done: false,
            },
        ];
        let lanes = plan_subagent_execution_lanes(&steps, 8);
        assert_eq!(lanes.len(), 2);
        assert_eq!(lanes[0].phase, 0);
        assert_eq!(lanes[1].phase, 1);
        assert!(
            lanes[1]
                .dependencies
                .iter()
                .any(|dep| dep.contains("src@phase1"))
        );
    }

    #[test]
    fn subagent_lane_planner_serializes_unscoped_task_after_targeted_tasks() {
        let steps = vec![
            PlanStep {
                step_id: Uuid::now_v7(),
                title: "Edit scoped file".to_string(),
                intent: "task".to_string(),
                tools: vec!["fs.edit".to_string()],
                files: vec!["src/lib.rs".to_string()],
                done: false,
            },
            PlanStep {
                step_id: Uuid::now_v7(),
                title: "Follow-up generic task".to_string(),
                intent: "task".to_string(),
                tools: vec!["bash.run".to_string()],
                files: vec![],
                done: false,
            },
        ];
        let lanes = plan_subagent_execution_lanes(&steps, 8);
        assert_eq!(lanes.len(), 2);
        assert_eq!(lanes[0].phase, 0);
        assert_eq!(lanes[1].phase, 1);
        assert!(
            lanes[1]
                .dependencies
                .iter()
                .any(|dep| dep.contains("unscoped@phase1"))
        );
    }

    #[test]
    fn subagent_lane_summary_includes_phase_and_lane_metadata() {
        let run_id = Uuid::now_v7();
        let mut map = HashMap::new();
        map.insert(
            run_id,
            SubagentTaskMeta {
                name: "Apply api edits".to_string(),
                goal: "task".to_string(),
                created_at: Utc::now().to_rfc3339(),
                targets: vec!["src/api.rs".to_string()],
                domain: "rust-code".to_string(),
                specialization_hint: None,
                phase: 1,
                dependencies: vec!["src/api.rs@phase1".to_string()],
                ownership_lane: "execution:src/api.rs".to_string(),
            },
        );
        let summary = summarize_subagent_execution_lanes(&map);
        assert_eq!(summary.len(), 1);
        assert!(summary[0].contains("subagent_phase 2"));
        assert!(summary[0].contains("lane=execution:src/api.rs"));
    }

    #[test]
    fn target_patterns_overlap_detects_prefix_and_wildcards() {
        assert!(target_patterns_overlap("src", "src/main.rs"));
        assert!(target_patterns_overlap("src/*.rs", "src/lib.rs"));
        assert!(target_patterns_overlap("src/lib.rs", "src/lib.rs"));
        assert!(!target_patterns_overlap("docs", "src/lib.rs"));
    }
}
