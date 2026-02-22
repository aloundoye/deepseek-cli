use crate::*;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct PlannerStrategyMemory {
    pub(crate) entries: Vec<PlannerStrategyEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PlannerStrategyEntry {
    pub(crate) key: String,
    pub(crate) goal_excerpt: String,
    pub(crate) strategy_summary: String,
    pub(crate) verification: Vec<String>,
    pub(crate) success_count: u64,
    #[serde(default)]
    pub(crate) failure_count: u64,
    #[serde(default = "default_strategy_score")]
    pub(crate) score: f32,
    #[serde(default)]
    pub(crate) last_outcome: String,
    pub(crate) updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct ObjectiveOutcomeMemory {
    pub(crate) entries: Vec<ObjectiveOutcomeEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ObjectiveOutcomeEntry {
    pub(crate) key: String,
    pub(crate) goal_excerpt: String,
    pub(crate) success_count: u64,
    #[serde(default)]
    pub(crate) failure_count: u64,
    #[serde(default)]
    pub(crate) execution_failure_count: u64,
    #[serde(default)]
    pub(crate) verification_failure_count: u64,
    #[serde(default)]
    pub(crate) avg_step_count: f32,
    #[serde(default)]
    pub(crate) avg_failure_count: f32,
    #[serde(default = "default_objective_confidence")]
    pub(crate) confidence: f32,
    #[serde(default)]
    pub(crate) last_outcome: String,
    #[serde(default)]
    pub(crate) last_failure_summary: String,
    #[serde(default)]
    pub(crate) next_focus: String,
    pub(crate) updated_at: String,
}

impl AgentEngine {
    pub(crate) fn remember_objective_outcome(
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

    pub(crate) fn read_objective_outcome_memory(&self) -> Result<ObjectiveOutcomeMemory> {
        let path = self.objective_outcome_memory_path();
        if !path.exists() {
            return Ok(ObjectiveOutcomeMemory::default());
        }
        let raw = fs::read_to_string(path)?;
        let mut memory = serde_json::from_str(&raw).unwrap_or_default();
        normalize_objective_outcome_memory(&mut memory);
        Ok(memory)
    }

    pub(crate) fn write_planner_strategy_memory(
        &self,
        memory: &PlannerStrategyMemory,
    ) -> Result<()> {
        let path = self.planner_strategy_memory_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(memory)?)?;
        Ok(())
    }

    pub(crate) fn load_matching_objective_outcomes(
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

    pub(crate) fn objective_outcome_memory_path(&self) -> PathBuf {
        deepseek_core::runtime_dir(&self.workspace).join("objective-outcomes.json")
    }

    pub(crate) fn write_objective_outcome_memory(
        &self,
        memory: &ObjectiveOutcomeMemory,
    ) -> Result<()> {
        let path = self.objective_outcome_memory_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(memory)?)?;
        Ok(())
    }

    pub(crate) fn remember_failed_strategy(
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

    pub(crate) fn load_matching_strategies(
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

    pub(crate) fn planner_strategy_memory_path(&self) -> PathBuf {
        deepseek_core::runtime_dir(&self.workspace).join("planner-strategies.json")
    }

    pub(crate) fn read_planner_strategy_memory(&self) -> Result<PlannerStrategyMemory> {
        let path = self.planner_strategy_memory_path();
        if !path.exists() {
            return Ok(PlannerStrategyMemory::default());
        }
        let raw = fs::read_to_string(path)?;
        let mut memory = serde_json::from_str(&raw).unwrap_or_default();
        normalize_strategy_memory(&mut memory);
        Ok(memory)
    }

    pub(crate) fn remember_successful_strategy(&self, prompt: &str, plan: &Plan) -> Result<()> {
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
}

pub(crate) fn plan_goal_pattern(goal: &str) -> String {
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

pub(crate) fn sort_and_prune_strategy_entries(entries: &mut Vec<PlannerStrategyEntry>) {
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

pub(crate) fn format_strategy_entries(entries: &[PlannerStrategyEntry]) -> String {
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

pub(crate) fn format_objective_outcomes(entries: &[ObjectiveOutcomeEntry]) -> String {
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

pub(crate) fn normalize_objective_outcome_memory(memory: &mut ObjectiveOutcomeMemory) {
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

pub(crate) fn compute_strategy_score(success_count: u64, failure_count: u64) -> f32 {
    let observations = success_count.saturating_add(failure_count) as f32;
    let posterior_mean = (success_count as f32 + 1.0) / (observations + 2.0);
    let confidence = (observations / 10.0).clamp(0.0, 1.0);
    (0.5 * (1.0 - confidence) + posterior_mean * confidence).clamp(0.0, 1.0)
}

pub(crate) fn summarize_strategy(plan: &Plan) -> String {
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

pub(crate) fn default_objective_confidence() -> f32 {
    0.5
}

pub(crate) fn default_strategy_score() -> f32 {
    0.5
}

pub(crate) fn objective_next_focus(entry: &ObjectiveOutcomeEntry) -> String {
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

pub(crate) fn normalize_strategy_memory(memory: &mut PlannerStrategyMemory) {
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

pub(crate) fn compute_objective_confidence(entry: &ObjectiveOutcomeEntry) -> f32 {
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

pub(crate) fn rolling_average(
    previous: f32,
    prior_count: u64,
    observed: f32,
    next_count: u64,
) -> f32 {
    if prior_count == 0 || next_count == 0 {
        return observed;
    }
    ((previous * prior_count as f32) + observed) / next_count as f32
}

pub(crate) fn truncate_strategy_prompt(prompt: &str) -> String {
    let trimmed = prompt.trim();
    if trimmed.chars().count() <= 220 {
        return trimmed.to_string();
    }
    let head = trimmed.chars().take(220).collect::<String>();
    format!("{head}...")
}

pub(crate) fn sort_and_prune_objective_entries(entries: &mut Vec<ObjectiveOutcomeEntry>) {
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
