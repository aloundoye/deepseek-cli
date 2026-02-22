use crate::subagents_runtime::orchestration::{SubagentExecutionLane, SubagentTaskMeta};
use crate::*;

pub(crate) fn target_patterns_overlap(a: &str, b: &str) -> bool {
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

pub(crate) fn subagent_domain_for_step(step: &PlanStep, targets: &[String]) -> String {
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

pub(crate) fn plan_subagent_execution_lanes(
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

pub(crate) fn subagent_targets_for_step(step: &PlanStep) -> Vec<String> {
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

pub(crate) fn summarize_subagent_execution_lanes(
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

pub(crate) fn subagent_team_for_role(role: &SubagentRole) -> &'static str {
    match role {
        SubagentRole::Explore => "explore",
        SubagentRole::Plan => "planning",
        SubagentRole::Task => "execution",
        SubagentRole::Custom(_) => "custom",
    }
}

pub(crate) fn subagent_role_for_step(step: &PlanStep) -> SubagentRole {
    match step.intent.as_str() {
        "search" => SubagentRole::Explore,
        "plan" | "recover" => SubagentRole::Plan,
        _ => SubagentRole::Task,
    }
}
