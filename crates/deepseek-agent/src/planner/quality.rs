use crate::planner::parsing::normalize_declared_tool_name;
use deepseek_core::Plan;
use deepseek_store::VerificationRunRecord;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub(crate) struct PlanQualityReport {
    pub(crate) acceptable: bool,
    pub(crate) score: f32,
    pub(crate) issues: Vec<String>,
}

pub(crate) fn assess_plan_quality(plan: &Plan, user_prompt: &str) -> PlanQualityReport {
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

pub(crate) fn combine_plan_quality_reports(
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

pub(crate) fn assess_plan_long_horizon_quality(
    plan: &Plan,
    user_prompt: &str,
    objective_outcomes: &[crate::planner::memory::ObjectiveOutcomeEntry],
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

pub(crate) fn build_plan_quality_repair_prompt(
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

pub(crate) fn assess_plan_feedback_alignment(
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

pub(crate) fn build_verification_feedback_repair_prompt(
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

pub(crate) fn verification_feedback_markers(command: &str) -> Vec<String> {
    command
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == ':'))
        .map(|token| token.trim().to_ascii_lowercase())
        .filter(|token| token.len() >= 3 && token != "and" && token != "the")
        .take(6)
        .collect::<Vec<_>>()
}

pub(crate) fn format_verification_feedback(feedback: &[VerificationRunRecord]) -> String {
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
