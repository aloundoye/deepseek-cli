use crate::planner::memory::rolling_average;
use crate::runtime::prompt::ChatSubagentSpawnDecision;
use crate::subagents_runtime::delegated::truncate_probe_text;
use crate::subagents_runtime::lanes::{subagent_domain_for_step, subagent_targets_for_step};
use crate::*;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct SubagentSpecializationMemory {
    pub(crate) entries: Vec<SubagentSpecializationEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SubagentSpecializationEntry {
    pub(crate) key: String,
    pub(crate) role: String,
    pub(crate) domain: String,
    pub(crate) success_count: u64,
    #[serde(default)]
    pub(crate) failure_count: u64,
    #[serde(default)]
    pub(crate) avg_attempts: f32,
    #[serde(default = "default_specialization_confidence")]
    pub(crate) confidence: f32,
    #[serde(default)]
    pub(crate) last_outcome: String,
    #[serde(default)]
    pub(crate) last_summary: String,
    #[serde(default)]
    pub(crate) next_guidance: String,
    pub(crate) updated_at: String,
}

impl AgentEngine {
    pub(crate) fn subagent_specialization_memory_path(&self) -> PathBuf {
        deepseek_core::runtime_dir(&self.workspace).join("subagent-specializations.json")
    }

    pub(crate) fn write_subagent_specialization_memory(
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

    pub(crate) fn load_subagent_specialization_hint(
        &self,
        role: &SubagentRole,
        domain: &str,
    ) -> Result<Option<String>> {
        let key = subagent_specialization_key(role, domain);
        let memory = self.read_subagent_specialization_memory()?;
        let best = memory.entries.into_iter().find(|entry| entry.key == key);
        Ok(best.map(|entry| format_subagent_specialization_hint(&entry)))
    }

    pub(crate) fn remember_subagent_specialization(
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

    pub(crate) fn read_subagent_specialization_memory(
        &self,
    ) -> Result<SubagentSpecializationMemory> {
        let path = self.subagent_specialization_memory_path();
        if !path.exists() {
            return Ok(SubagentSpecializationMemory::default());
        }
        let raw = fs::read_to_string(path)?;
        let mut memory = serde_json::from_str(&raw).unwrap_or_default();
        normalize_subagent_specialization_memory(&mut memory);
        Ok(memory)
    }
}

pub(crate) fn summarize_subagent_notes(notes: &[String]) -> String {
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

pub(crate) fn sort_and_prune_subagent_specialization_entries(
    entries: &mut Vec<SubagentSpecializationEntry>,
) {
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

pub(crate) fn decide_chat_subagent_spawn(
    options: &ChatOptions,
    signals: &deepseek_tools::ToolContextSignals,
    plan: &Plan,
    max_concurrency: usize,
) -> ChatSubagentSpawnDecision {
    let step_count = plan.steps.len();
    if step_count == 0 {
        return ChatSubagentSpawnDecision {
            should_spawn: false,
            blocked_by_tools: false,
            score: 0.0,
            task_budget: 1,
        };
    }
    let target_count = plan
        .steps
        .iter()
        .flat_map(|step| {
            if step.files.is_empty() {
                subagent_targets_for_step(step)
            } else {
                step.files.clone()
            }
        })
        .map(|target| target.trim().to_string())
        .filter(|target| !target.is_empty())
        .collect::<HashSet<_>>()
        .len();
    let write_steps = plan
        .steps
        .iter()
        .filter(|step| {
            let intent = step.intent.to_ascii_lowercase();
            intent.contains("edit")
                || intent.contains("write")
                || intent.contains("patch")
                || step.tools.iter().any(|tool| {
                    tool.contains("fs.write")
                        || tool.contains("fs.edit")
                        || tool.contains("multi_edit")
                        || tool.contains("patch.apply")
                })
        })
        .count();
    let write_ratio = if step_count > 0 {
        write_steps as f32 / step_count as f32
    } else {
        0.0
    };
    let verification_breadth = (plan.verification.len() as f32 / 4.0).clamp(0.0, 1.0);
    let domain_count = plan
        .steps
        .iter()
        .map(|step| {
            let targets = subagent_targets_for_step(step);
            subagent_domain_for_step(step, &targets)
        })
        .collect::<HashSet<_>>()
        .len();
    let step_factor = (step_count as f32 / 6.0).clamp(0.0, 1.0);
    let target_factor = (target_count as f32 / 5.0).clamp(0.0, 1.0);
    let repo_factor = (signals.codebase_file_count as f32 / 400.0).clamp(0.0, 1.0);
    let cross_domain = if domain_count >= 2 { 1.0 } else { 0.0 };
    let prompt_complex = if signals.prompt_is_complex { 1.0 } else { 0.0 };
    let score = step_factor * 0.28
        + target_factor * 0.22
        + cross_domain * 0.15
        + write_ratio * 0.12
        + verification_breadth * 0.10
        + prompt_complex * 0.08
        + repo_factor * 0.05;
    let threshold = 0.27;
    let scope_large_enough = step_count >= 2 || target_count >= 2;
    let should_spawn = options.tools && scope_large_enough && score >= threshold;
    let blocked_by_tools = !options.tools && scope_large_enough && score >= threshold;
    let max_lanes = plan
        .steps
        .iter()
        .flat_map(|step| step.files.iter().map(|f| f.as_str()))
        .collect::<HashSet<_>>()
        .len();
    let max_budget = max_concurrency
        .saturating_mul(3)
        .max(max_concurrency)
        .max(2);
    let lane_hint = (((score * max_budget as f32).ceil() as usize).max(2))
        .min(max_lanes.max(2))
        .min(max_budget);
    ChatSubagentSpawnDecision {
        should_spawn,
        blocked_by_tools,
        score: score.clamp(0.0, 1.0),
        task_budget: lane_hint,
    }
}

pub(crate) fn subagent_specialization_guidance(entry: &SubagentSpecializationEntry) -> String {
    if entry.last_outcome.eq_ignore_ascii_case("success") {
        return "reuse successful decomposition pattern and keep concise evidence".to_string();
    }
    if entry.failure_count > entry.success_count {
        return "reduce branching; gather stronger evidence before proposing edits".to_string();
    }
    "maintain deterministic ordering and verification-first summaries".to_string()
}

pub(crate) fn default_specialization_confidence() -> f32 {
    0.5
}

pub(crate) fn normalize_subagent_specialization_memory(memory: &mut SubagentSpecializationMemory) {
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

pub(crate) fn compute_subagent_specialization_confidence(
    entry: &SubagentSpecializationEntry,
) -> f32 {
    let observations = entry.success_count.saturating_add(entry.failure_count) as f32;
    if observations <= f32::EPSILON {
        return 0.5;
    }
    let posterior = (entry.success_count as f32 + 1.0) / (observations + 2.0);
    let attempts_penalty = ((entry.avg_attempts - 1.0).max(0.0) / 3.0).min(1.0) * 0.20;
    (posterior - attempts_penalty).clamp(0.0, 1.0)
}

pub(crate) fn format_subagent_specialization_hint(entry: &SubagentSpecializationEntry) -> String {
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

pub(crate) fn subagent_specialization_key(role: &SubagentRole, domain: &str) -> String {
    format!("role={role:?}|domain={domain}")
}

pub(crate) fn augment_goal_with_subagent_notes(goal: &str, notes: &[String]) -> String {
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
