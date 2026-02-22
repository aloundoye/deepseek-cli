use crate::*;

pub(crate) fn subagent_arbitration_priority(role: &SubagentRole) -> u8 {
    match role {
        SubagentRole::Task => 0,
        SubagentRole::Plan => 1,
        SubagentRole::Explore => 2,
        SubagentRole::Custom(_) => 3,
    }
}

pub(crate) fn summarize_subagent_merge_arbitration(
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

pub(crate) fn subagent_arbitration_rationale(
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

pub(crate) fn subagent_arbitration_score(
    result: &deepseek_subagent::SubagentResult,
    target: &str,
) -> f32 {
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
