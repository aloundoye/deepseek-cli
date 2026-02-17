use anyhow::{Result, anyhow};
use chrono::Utc;
use deepseek_core::{
    AppConfig, ApprovedToolCall, EventEnvelope, EventKind, ExecContext, Executor, Failure,
    LlmRequest, LlmUnit, ModelRouter, Plan, PlanContext, PlanStep, Planner, RouterSignals, Session,
    SessionBudgets, SessionState, StepOutcome, ToolCall, ToolHost,
};
use deepseek_llm::{DeepSeekClient, LlmClient};
use deepseek_observe::Observer;
use deepseek_policy::PolicyEngine;
use deepseek_router::WeightedRouter;
use deepseek_store::{Store, SubagentRunRecord};
use deepseek_subagent::{SubagentManager, SubagentTask};
use deepseek_tools::LocalToolHost;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

pub struct SimplePlanner;

impl Planner for SimplePlanner {
    fn create_plan(&self, ctx: PlanContext) -> Result<Plan> {
        let mut steps = vec![
            PlanStep {
                step_id: Uuid::now_v7(),
                title: "Locate relevant modules".to_string(),
                intent: "search".to_string(),
                tools: vec![
                    "index.query".to_string(),
                    "fs.search_rg".to_string(),
                    "fs.read".to_string(),
                ],
                files: vec![],
                done: false,
            },
            PlanStep {
                step_id: Uuid::now_v7(),
                title: "Implement changes".to_string(),
                intent: "edit".to_string(),
                tools: vec!["patch.stage".to_string(), "patch.apply".to_string()],
                files: vec![],
                done: false,
            },
            PlanStep {
                step_id: Uuid::now_v7(),
                title: "Run verification".to_string(),
                intent: "verify".to_string(),
                tools: vec!["bash.run".to_string()],
                files: vec![],
                done: false,
            },
        ];

        if ctx.user_prompt.to_lowercase().contains("docs") {
            steps.push(PlanStep {
                step_id: Uuid::now_v7(),
                title: "Update docs".to_string(),
                intent: "docs".to_string(),
                tools: vec!["patch.stage".to_string(), "patch.apply".to_string()],
                files: vec!["README.md".to_string()],
                done: false,
            });
        }

        Ok(Plan {
            plan_id: Uuid::now_v7(),
            version: 1,
            goal: ctx.user_prompt,
            assumptions: vec![
                "Workspace is writable".to_string(),
                "Offline LLM mode is acceptable by default".to_string(),
            ],
            steps,
            verification: vec![
                "cargo fmt --all -- --check".to_string(),
                "cargo test --workspace".to_string(),
            ],
            risk_notes: vec!["May require approval for patch apply and bash.run".to_string()],
        })
    }

    fn revise_plan(&self, _ctx: PlanContext, last_plan: &Plan, failure: Failure) -> Result<Plan> {
        let mut revised = last_plan.clone();
        revised.version += 1;
        revised.steps.push(PlanStep {
            step_id: Uuid::now_v7(),
            title: "Recovery: resolve failure".to_string(),
            intent: "recover".to_string(),
            tools: vec!["fs.search_rg".to_string(), "fs.read".to_string()],
            files: vec![],
            done: false,
        });
        revised
            .risk_notes
            .push(format!("revision due to failure: {}", failure.summary));
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

pub struct AgentEngine {
    workspace: PathBuf,
    store: Store,
    planner: SimplePlanner,
    router: WeightedRouter,
    llm: Box<dyn LlmClient + Send + Sync>,
    observer: Observer,
    tool_host: Arc<LocalToolHost>,
    policy: PolicyEngine,
    cfg: AppConfig,
    subagents: SubagentManager,
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

        Ok(Self {
            workspace: workspace.to_path_buf(),
            store,
            planner: SimplePlanner,
            router,
            llm,
            observer,
            tool_host,
            policy,
            cfg,
            subagents: SubagentManager::default(),
        })
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
        self.plan_only_with_mode(prompt, false)
    }

    fn plan_only_with_mode(&self, prompt: &str, force_max_think: bool) -> Result<Plan> {
        let mut session = self.ensure_session()?;
        self.transition(&mut session, SessionState::Planning)?;
        let redacted_prompt = self.policy.redact(prompt);
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

        let llm_response = self.llm.complete(&LlmRequest {
            unit: LlmUnit::Planner,
            prompt: planner_request.clone(),
            model: decision.selected_model.clone(),
            max_tokens: session.budgets.max_think_tokens,
        })?;
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
            let retry = self.llm.complete(&LlmRequest {
                unit: LlmUnit::Planner,
                prompt: planner_request,
                model: self.cfg.llm.max_think_model.clone(),
                max_tokens: session.budgets.max_think_tokens,
            })?;
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
        let plan = if let Some(plan) = plan {
            plan
        } else {
            self.planner.create_plan(PlanContext {
                session: session.clone(),
                user_prompt: prompt.to_string(),
                prior_failures: vec![],
            })?
        };

        session.active_plan_id = Some(plan.plan_id);
        self.store.save_session(&session)?;
        self.store.save_plan(session.session_id, &plan)?;
        self.emit(
            session.session_id,
            EventKind::PlanCreatedV1 { plan: plan.clone() },
        )?;

        Ok(plan)
    }

    pub fn run_once(&self, prompt: &str, allow_tools: bool) -> Result<String> {
        self.run_once_with_mode(prompt, allow_tools, false)
    }

    pub fn run_once_with_mode(
        &self,
        prompt: &str,
        allow_tools: bool,
        force_max_think: bool,
    ) -> Result<String> {
        let mut session = self.ensure_session()?;
        self.emit(
            session.session_id,
            EventKind::TurnAddedV1 {
                role: "user".to_string(),
                content: prompt.to_string(),
            },
        )?;

        let mut plan = self.plan_only_with_mode(prompt, force_max_think)?;
        session = self.ensure_session()?;
        self.transition(&mut session, SessionState::ExecutingStep)?;
        let _subagent_notes = self.run_subagents(session.session_id, &plan)?;
        let mut failure_streak = 0_u32;

        for idx in 0..plan.steps.len() {
            let step = plan.steps[idx].clone();
            let call = self.call_for_step(&step, &plan);
            let proposal = self.tool_host.propose(call);
            self.emit(
                session.session_id,
                EventKind::ToolProposedV1 {
                    proposal: proposal.clone(),
                },
            )?;

            let outcome = if proposal.approved || allow_tools {
                self.emit(
                    session.session_id,
                    EventKind::ToolApprovedV1 {
                        invocation_id: proposal.invocation_id,
                    },
                )?;
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
                self.emit_patch_events_if_any(session.session_id, &proposal.call.name, &result)?;
                StepOutcome {
                    step_id: step.step_id,
                    success: result.success,
                    notes: result.output.to_string(),
                }
            } else {
                StepOutcome {
                    step_id: step.step_id,
                    success: false,
                    notes: "approval required".to_string(),
                }
            };

            plan.steps[idx].done = outcome.success;
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
                let revised = self.planner.revise_plan(
                    PlanContext {
                        session: session.clone(),
                        user_prompt: prompt.to_string(),
                        prior_failures: vec![],
                    },
                    &plan,
                    Failure {
                        summary: "step failed".to_string(),
                        detail: outcome.notes,
                    },
                )?;
                self.emit(
                    session.session_id,
                    EventKind::PlanRevisedV1 {
                        plan: revised.clone(),
                    },
                )?;
                plan = revised;
                break;
            }
        }

        self.transition(&mut session, SessionState::Verifying)?;
        let mut verification_failures = 0_u32;
        for cmd in &plan.verification {
            let proposal = self.tool_host.propose(ToolCall {
                name: "bash.run".to_string(),
                args: json!({"cmd": cmd}),
                requires_approval: true,
            });
            let (ok, output) = if proposal.approved || allow_tools {
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

        self.transition(&mut session, SessionState::Completed)?;

        let projection = self.store.rebuild_from_events(session.session_id)?;
        Ok(format!(
            "session={} steps={} router_models={:?} base_model={} max_model={}",
            session.session_id,
            projection.step_status.len(),
            projection.router_models,
            self.cfg.llm.base_model,
            self.cfg.llm.max_think_model
        ))
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
        let tasks = plan
            .steps
            .iter()
            .take(self.subagents.max_concurrency)
            .map(|step| SubagentTask {
                run_id: Uuid::now_v7(),
                name: step.title.clone(),
                goal: step.intent.clone(),
            })
            .collect::<Vec<_>>();
        if tasks.is_empty() {
            return Ok(Vec::new());
        }

        let now = Utc::now().to_rfc3339();
        let task_by_id = tasks
            .iter()
            .map(|task| {
                (
                    task.run_id,
                    (task.name.clone(), task.goal.clone(), now.clone()),
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

        let results = self.subagents.run_tasks(tasks, |task| {
            Ok(format!(
                "subagent '{}' analyzed intent '{}'",
                task.name, task.goal
            ))
        });
        let mut notes = Vec::new();
        for result in results {
            let updated_at = Utc::now().to_rfc3339();
            let (name, goal, created_at) = task_by_id
                .get(&result.run_id)
                .cloned()
                .unwrap_or_else(|| ("subagent".to_string(), String::new(), updated_at.clone()));
            if result.success {
                self.emit(
                    session_id,
                    EventKind::SubagentCompletedV1 {
                        run_id: result.run_id,
                        output: result.output.clone(),
                    },
                )?;
                self.store.upsert_subagent_run(&SubagentRunRecord {
                    run_id: result.run_id,
                    name,
                    goal,
                    status: "completed".to_string(),
                    output: Some(result.output.clone()),
                    error: None,
                    created_at,
                    updated_at,
                })?;
                notes.push(result.output);
            } else {
                let error = result
                    .error
                    .unwrap_or_else(|| "unknown subagent error".to_string());
                self.emit(
                    session_id,
                    EventKind::SubagentFailedV1 {
                        run_id: result.run_id,
                        error: error.clone(),
                    },
                )?;
                self.store.upsert_subagent_run(&SubagentRunRecord {
                    run_id: result.run_id,
                    name,
                    goal,
                    status: "failed".to_string(),
                    output: None,
                    error: Some(error),
                    created_at,
                    updated_at,
                })?;
            }
        }
        Ok(notes)
    }

    fn emit_cost_event(
        &self,
        session_id: Uuid,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Result<()> {
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
        )
    }

    fn transition(&self, session: &mut Session, to: SessionState) -> Result<()> {
        let from = session.status.clone();
        if !is_valid_transition(&from, &to) {
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

    fn call_for_step(&self, step: &PlanStep, plan: &Plan) -> ToolCall {
        match step.intent.as_str() {
            "search" => ToolCall {
                name: "fs.search_rg".to_string(),
                args: json!({"query": plan.goal, "limit": 10}),
                requires_approval: false,
            },
            "edit" => ToolCall {
                name: "patch.stage".to_string(),
                args: json!({
                    "unified_diff": format!(
                        "diff --git a/.deepseek/notes.txt b/.deepseek/notes.txt\nnew file mode 100644\nindex 0000000..2b9d865\n--- /dev/null\n+++ b/.deepseek/notes.txt\n@@ -0,0 +1 @@\n+{}\n",
                        plan.goal.replace('\n', " ")
                    ),
                    "base": ""
                }),
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

fn build_planner_prompt(task: &str) -> String {
    format!(
        "Return only JSON with keys: goal, assumptions, steps, verification, risk_notes. \
         Each step must include: title, intent, tools, files. User task: {task}"
    )
}

fn parse_plan_from_llm(text: &str, fallback_goal: &str) -> Option<Plan> {
    let snippet = extract_json_snippet(text)?;
    let parsed: PlanLlmShape = serde_json::from_str(snippet).ok()?;
    if parsed.steps.is_empty() {
        return None;
    }
    let steps = parsed
        .steps
        .into_iter()
        .map(|s| PlanStep {
            step_id: Uuid::now_v7(),
            title: s.title,
            intent: s.intent,
            tools: s.tools,
            files: s.files,
            done: false,
        })
        .collect::<Vec<_>>();

    Some(Plan {
        plan_id: Uuid::now_v7(),
        version: 1,
        goal: parsed.goal.unwrap_or_else(|| fallback_goal.to_string()),
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

fn estimate_tokens(text: &str) -> u64 {
    // Rough token estimate for local accounting and status reporting.
    (text.chars().count() as u64).div_ceil(4)
}

fn is_valid_transition(from: &SessionState, to: &SessionState) -> bool {
    use SessionState::*;
    if from == to {
        return true;
    }
    matches!(
        (from, to),
        (Idle, Planning)
            | (Completed, Planning)
            | (Failed, Planning)
            | (Planning, ExecutingStep)
            | (Planning, Failed)
            | (ExecutingStep, AwaitingApproval)
            | (ExecutingStep, Planning)
            | (ExecutingStep, Verifying)
            | (ExecutingStep, Failed)
            | (AwaitingApproval, ExecutingStep)
            | (AwaitingApproval, Failed)
            | (Verifying, ExecutingStep)
            | (Verifying, Completed)
            | (Verifying, Failed)
            | (_, Paused)
            | (Paused, Planning)
            | (Paused, ExecutingStep)
            | (Paused, Completed)
            | (Paused, Failed)
    )
}
