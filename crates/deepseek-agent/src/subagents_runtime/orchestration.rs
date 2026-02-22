use crate::subagents_runtime::arbitration::summarize_subagent_merge_arbitration;
use crate::subagents_runtime::delegated::run_subagent_delegated_tools;
use crate::subagents_runtime::lanes::{
    plan_subagent_execution_lanes, summarize_subagent_execution_lanes,
};
use crate::*;

impl AgentEngine {
    pub(crate) fn run_subagents(
        &self,
        session_id: Uuid,
        plan: &Plan,
        task_budget: Option<usize>,
    ) -> Result<Vec<String>> {
        let default_budget = self
            .subagents
            .max_concurrency
            .saturating_mul(3)
            .max(self.subagents.max_concurrency);
        let max_tasks = task_budget.unwrap_or(default_budget).max(1);
        let prepared = self.prepare_subagent_tasks(plan, max_tasks)?;
        if prepared.tasks.is_empty() {
            return Ok(Vec::new());
        }

        let now = Utc::now().to_rfc3339();
        self.emit_subagent_spawn_events(session_id, &prepared.tasks, &now)?;
        let results = self.execute_subagent_tasks(plan, &prepared);
        self.persist_subagent_results(session_id, results, &prepared)
    }

    fn prepare_subagent_tasks(
        &self,
        plan: &Plan,
        max_tasks: usize,
    ) -> Result<PreparedSubagentTasks> {
        let lane_plan = plan_subagent_execution_lanes(&plan.steps, max_tasks);
        let mut tasks = Vec::new();
        let mut task_targets = HashMap::new();
        let mut task_by_id = HashMap::new();
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
                custom_agent: None,
            };
            let specialization_hint =
                self.load_subagent_specialization_hint(&lane.role, &lane.domain)?;
            task_by_id.insert(
                task.run_id,
                SubagentTaskMeta {
                    name: task.name.clone(),
                    goal: task.goal.clone(),
                    created_at: String::new(),
                    targets: lane.targets.clone(),
                    domain: lane.domain,
                    specialization_hint,
                    phase: lane.phase,
                    dependencies: lane.dependencies,
                    ownership_lane: lane.ownership_lane,
                },
            );
            task_targets.insert(task.run_id, lane.targets);
            tasks.push(task);
        }
        Ok(PreparedSubagentTasks {
            tasks,
            task_targets,
            task_by_id,
        })
    }

    fn emit_subagent_spawn_events(
        &self,
        session_id: Uuid,
        tasks: &[SubagentTask],
        now: &str,
    ) -> Result<()> {
        for task in tasks {
            self.emit(
                session_id,
                EventKind::SubagentSpawnedV1 {
                    run_id: task.run_id,
                    name: task.name.clone(),
                    goal: task.goal.clone(),
                },
            )?;
            if let Ok(cb_guard) = self.stream_callback.lock()
                && let Some(ref cb) = *cb_guard
            {
                cb(StreamChunk::SubagentSpawned {
                    run_id: task.run_id.to_string(),
                    name: task.name.clone(),
                    goal: task.goal.clone(),
                });
            }
            self.tool_host.fire_session_hooks("subagentspawned");
            self.store.upsert_subagent_run(&SubagentRunRecord {
                run_id: task.run_id,
                name: task.name.clone(),
                goal: task.goal.clone(),
                status: "running".to_string(),
                output: None,
                error: None,
                created_at: now.to_string(),
                updated_at: now.to_string(),
            })?;
        }
        Ok(())
    }

    fn execute_subagent_tasks(
        &self,
        plan: &Plan,
        prepared: &PreparedSubagentTasks,
    ) -> Vec<deepseek_subagent::SubagentResult> {
        let llm_cfg = self.cfg.llm.clone();
        let base_model = self.cfg.llm.base_model.clone();
        let max_think_model = self.cfg.llm.max_think_model.clone();
        let root_goal = plan.goal.clone();
        let tool_host = Arc::clone(&self.tool_host);
        let task_targets_ref = Arc::new(prepared.task_targets.clone());
        let task_meta_ref = Arc::new(prepared.task_by_id.clone());
        let verification = if plan.verification.is_empty() {
            "none".to_string()
        } else {
            plan.verification.join(" ; ")
        };
        let mut tasks_by_phase: BTreeMap<usize, Vec<SubagentTask>> = BTreeMap::new();
        for task in prepared.tasks.clone() {
            let phase = prepared
                .task_by_id
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
                    .unwrap_or_else(|| default_subagent_task_meta(&task));
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
        results
    }

    fn persist_subagent_results(
        &self,
        session_id: Uuid,
        results: Vec<deepseek_subagent::SubagentResult>,
        prepared: &PreparedSubagentTasks,
    ) -> Result<Vec<String>> {
        let merged_summary = self.subagents.merge_results(&results);
        let arbitration = summarize_subagent_merge_arbitration(&results, &prepared.task_targets);
        let mut notes = Vec::new();
        let lane_notes = summarize_subagent_execution_lanes(&prepared.task_by_id);
        if !lane_notes.is_empty() {
            notes.push(lane_notes.join("\n"));
        }
        for result in results {
            let updated_at = Utc::now().to_rfc3339();
            let mut meta = prepared
                .task_by_id
                .get(&result.run_id)
                .cloned()
                .unwrap_or_else(|| default_subagent_task_meta_from_time(&updated_at));
            if meta.created_at.is_empty() {
                meta.created_at = updated_at.clone();
            }
            if result.success {
                self.persist_subagent_success(session_id, &result, &meta, &updated_at)?;
            } else {
                self.persist_subagent_failure(session_id, result, &meta, &updated_at)?;
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

    fn persist_subagent_success(
        &self,
        session_id: Uuid,
        result: &deepseek_subagent::SubagentResult,
        meta: &SubagentTaskMeta,
        updated_at: &str,
    ) -> Result<()> {
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
        let output_summary = if result.output.len() > 240 {
            format!(
                "{}...",
                &result.output[..result.output.floor_char_boundary(240)]
            )
        } else {
            result.output.clone()
        };
        self.emit(
            session_id,
            EventKind::SubagentCompletedV1 {
                run_id: result.run_id,
                output: result.output.clone(),
            },
        )?;
        if let Ok(cb_guard) = self.stream_callback.lock()
            && let Some(ref cb) = *cb_guard
        {
            cb(StreamChunk::SubagentCompleted {
                run_id: result.run_id.to_string(),
                name: meta.name.clone(),
                summary: output_summary,
            });
        }
        self.tool_host.fire_session_hooks("subagentcompleted");
        self.store.upsert_subagent_run(&SubagentRunRecord {
            run_id: result.run_id,
            name: meta.name.clone(),
            goal: persisted_goal,
            status: "completed".to_string(),
            output: Some(result.output.clone()),
            error: None,
            created_at: meta.created_at.clone(),
            updated_at: updated_at.to_string(),
        })?;
        self.remember_subagent_specialization(
            &result.role,
            &meta.domain,
            true,
            result.attempts,
            &result.output,
        )?;
        Ok(())
    }

    fn persist_subagent_failure(
        &self,
        session_id: Uuid,
        result: deepseek_subagent::SubagentResult,
        meta: &SubagentTaskMeta,
        updated_at: &str,
    ) -> Result<()> {
        let error = result
            .error
            .clone()
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
        if let Ok(cb_guard) = self.stream_callback.lock()
            && let Some(ref cb) = *cb_guard
        {
            cb(StreamChunk::SubagentFailed {
                run_id: result.run_id.to_string(),
                name: meta.name.clone(),
                error: error.clone(),
            });
        }
        self.tool_host.fire_session_hooks("subagentcompleted");
        self.store.upsert_subagent_run(&SubagentRunRecord {
            run_id: result.run_id,
            name: meta.name.clone(),
            goal: persisted_goal,
            status: "failed".to_string(),
            output: None,
            error: Some(error),
            created_at: meta.created_at.clone(),
            updated_at: updated_at.to_string(),
        })?;
        self.remember_subagent_specialization(
            &result.role,
            &meta.domain,
            false,
            result.attempts,
            &error_for_memory,
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct PreparedSubagentTasks {
    tasks: Vec<SubagentTask>,
    task_targets: HashMap<Uuid, Vec<String>>,
    task_by_id: HashMap<Uuid, SubagentTaskMeta>,
}

fn default_subagent_task_meta(task: &SubagentTask) -> SubagentTaskMeta {
    SubagentTaskMeta {
        name: task.name.clone(),
        goal: task.goal.clone(),
        created_at: Utc::now().to_rfc3339(),
        targets: Vec::new(),
        domain: "general".to_string(),
        specialization_hint: None,
        phase: 0,
        dependencies: Vec::new(),
        ownership_lane: "execution:unscoped".to_string(),
    }
}

fn default_subagent_task_meta_from_time(now: &str) -> SubagentTaskMeta {
    SubagentTaskMeta {
        name: "subagent".to_string(),
        goal: String::new(),
        created_at: now.to_string(),
        targets: Vec::new(),
        domain: "general".to_string(),
        specialization_hint: None,
        phase: 0,
        dependencies: Vec::new(),
        ownership_lane: "execution:unscoped".to_string(),
    }
}

pub(crate) fn subagent_request_for_task(
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
pub(crate) struct SubagentTaskMeta {
    pub(crate) name: String,
    pub(crate) goal: String,
    pub(crate) created_at: String,
    pub(crate) targets: Vec<String>,
    pub(crate) domain: String,
    pub(crate) specialization_hint: Option<String>,
    pub(crate) phase: usize,
    pub(crate) dependencies: Vec<String>,
    pub(crate) ownership_lane: String,
}

#[derive(Debug, Clone)]
pub(crate) struct SubagentExecutionLane {
    pub(crate) title: String,
    pub(crate) intent: String,
    pub(crate) role: SubagentRole,
    pub(crate) team: String,
    pub(crate) targets: Vec<String>,
    pub(crate) domain: String,
    pub(crate) phase: usize,
    pub(crate) dependencies: Vec<String>,
    pub(crate) ownership_lane: String,
}
