use anyhow::{Result, anyhow};
use chrono::{Duration, Utc};
use deepseek_core::{AppConfig, EventEnvelope, EventKind, runtime_dir};
use deepseek_policy::{
    ManagedSettings, load_managed_settings, managed_settings_path, team_policy_locks,
};
use deepseek_store::Store;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::LeadershipArgs;
use crate::output::print_json;

const MAX_SCAN_FILES: usize = 5_000;
const MAX_EVIDENCE: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct LeadershipReport {
    pub(crate) enterprise: EnterpriseReport,
    pub(crate) ecosystem: EcosystemReport,
    pub(crate) deployment: DeploymentReport,
    pub(crate) readiness: LeadershipReadiness,
    pub(crate) warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct LeadershipReadiness {
    pub(crate) ok: bool,
    pub(crate) score: u8,
    pub(crate) risks: Vec<String>,
    pub(crate) gaps: Vec<String>,
    pub(crate) recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct EnterpriseReport {
    pub(crate) sso: SsoStatus,
    pub(crate) managed_settings: ManagedSettingsStatus,
    pub(crate) team_policy: TeamPolicyStatus,
    pub(crate) audit: AuditSummary,
    pub(crate) admin: EnterpriseAdminSummary,
    pub(crate) recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct SsoStatus {
    pub(crate) configured: bool,
    pub(crate) providers: Vec<String>,
    pub(crate) evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct ManagedSettingsStatus {
    pub(crate) path: Option<String>,
    pub(crate) exists: bool,
    pub(crate) loaded: bool,
    pub(crate) forced_permission_mode: Option<String>,
    pub(crate) allow_managed_permission_rules_only: bool,
    pub(crate) allow_managed_hooks_only: bool,
    pub(crate) disable_bypass_permissions_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct TeamPolicyStatus {
    pub(crate) active: bool,
    pub(crate) path: Option<String>,
    pub(crate) permission_fields_locked: bool,
    pub(crate) approve_edits_locked: bool,
    pub(crate) approve_bash_locked: bool,
    pub(crate) allowlist_locked: bool,
    pub(crate) sandbox_mode_locked: bool,
    pub(crate) permission_mode_locked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct AuditSummary {
    pub(crate) events_path: String,
    pub(crate) events_total: u64,
    pub(crate) events_in_window: u64,
    pub(crate) window_hours: u64,
    pub(crate) parse_errors: u64,
    pub(crate) failed_verifications: u64,
    pub(crate) pending_approvals: u64,
    pub(crate) top_event_types: Vec<EventCount>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct EventCount {
    pub(crate) event_type: String,
    pub(crate) count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct EnterpriseAdminSummary {
    pub(crate) usage_records: u64,
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) estimated_cost_usd: f64,
    pub(crate) pending_tasks: u64,
    pub(crate) running_background_jobs: u64,
    pub(crate) pending_approvals: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct EcosystemReport {
    pub(crate) integrations: Vec<IntegrationSignal>,
    pub(crate) coverage: EcosystemCoverage,
    pub(crate) recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct EcosystemCoverage {
    pub(crate) scm: bool,
    pub(crate) work_tracking: bool,
    pub(crate) collaboration: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct IntegrationSignal {
    pub(crate) name: String,
    pub(crate) category: String,
    pub(crate) confidence: f32,
    pub(crate) evidence: Vec<String>,
    pub(crate) recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct DeploymentReport {
    pub(crate) coverage: DeploymentCoverage,
    pub(crate) signals: Vec<DeploymentSignal>,
    pub(crate) risks: Vec<String>,
    pub(crate) recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct DeploymentCoverage {
    pub(crate) ci: bool,
    pub(crate) container: bool,
    pub(crate) kubernetes: bool,
    pub(crate) cloud: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct DeploymentSignal {
    pub(crate) area: String,
    pub(crate) confidence: f32,
    pub(crate) evidence: Vec<String>,
    pub(crate) recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
struct WorkspaceFile {
    abs: PathBuf,
    rel: String,
    rel_lower: String,
}

#[derive(Default)]
struct SignalAccumulator {
    category: String,
    confidence: f32,
    evidence: BTreeSet<String>,
    recommendations: BTreeSet<String>,
}

pub(crate) fn run_leadership(cwd: &Path, args: LeadershipArgs, json_mode: bool) -> Result<()> {
    let payload = build_leadership_report(cwd, args.audit_window_hours)?;

    if let Some(path) = args.export.as_deref() {
        let output = PathBuf::from(path);
        if let Some(parent) = output.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(&output, serde_json::to_vec_pretty(&payload)?)?;
    }

    if args.strict && !payload.readiness.ok {
        return Err(anyhow!(
            "leadership checks failed: {}",
            payload.readiness.risks.join(" | ")
        ));
    }

    if json_mode {
        print_json(&payload)?;
    } else {
        println!(
            "leadership readiness: score={} ok={}",
            payload.readiness.score, payload.readiness.ok
        );
        println!(
            "enterprise: sso={} team_policy={} managed_settings={} audit_events={}",
            payload.enterprise.sso.configured,
            payload.enterprise.team_policy.active,
            payload.enterprise.managed_settings.loaded,
            payload.enterprise.audit.events_total
        );
        println!(
            "ecosystem: scm={} work_tracking={} collaboration={} integrations={}",
            payload.ecosystem.coverage.scm,
            payload.ecosystem.coverage.work_tracking,
            payload.ecosystem.coverage.collaboration,
            payload.ecosystem.integrations.len()
        );
        println!(
            "deployment: ci={} container={} kubernetes={} cloud={} risks={}",
            payload.deployment.coverage.ci,
            payload.deployment.coverage.container,
            payload.deployment.coverage.kubernetes,
            payload.deployment.coverage.cloud,
            payload.deployment.risks.len()
        );
        if !payload.readiness.risks.is_empty() {
            println!("risks:");
            for risk in &payload.readiness.risks {
                println!("- {risk}");
            }
        }
        if !payload.readiness.recommendations.is_empty() {
            println!("recommendations:");
            for recommendation in &payload.readiness.recommendations {
                println!("- {recommendation}");
            }
        }
        if let Some(path) = args.export {
            println!("report written: {}", PathBuf::from(path).display());
        }
    }

    Ok(())
}

pub(crate) fn build_leadership_report(
    cwd: &Path,
    audit_window_hours: u64,
) -> Result<LeadershipReport> {
    let files = scan_workspace_files(cwd, MAX_SCAN_FILES);
    let mut warnings = Vec::new();

    let enterprise = match analyze_enterprise(cwd, audit_window_hours) {
        Ok(report) => report,
        Err(err) => {
            warnings.push(format!("enterprise analysis failed: {err}"));
            EnterpriseReport::default()
        }
    };
    let ecosystem = match analyze_ecosystem(cwd, &files) {
        Ok(report) => report,
        Err(err) => {
            warnings.push(format!("ecosystem analysis failed: {err}"));
            EcosystemReport::default()
        }
    };
    let deployment = match analyze_deployment(cwd, &files) {
        Ok(report) => report,
        Err(err) => {
            warnings.push(format!("deployment analysis failed: {err}"));
            DeploymentReport::default()
        }
    };

    let readiness = compute_readiness(&enterprise, &ecosystem, &deployment, &warnings);

    Ok(LeadershipReport {
        enterprise,
        ecosystem,
        deployment,
        readiness,
        warnings,
    })
}

fn analyze_enterprise(cwd: &Path, audit_window_hours: u64) -> Result<EnterpriseReport> {
    let cfg = AppConfig::ensure(cwd)?;
    let store = Store::new(cwd)?;

    let sso = detect_sso(cwd);
    let team_policy = build_team_policy_status();
    let managed_settings = build_managed_settings_status();
    let audit = collect_audit_summary(cwd, audit_window_hours);

    let usage = store.usage_summary(None, None)?;
    let input_cost = (usage.input_tokens as f64 / 1_000_000.0) * cfg.usage.cost_per_million_input;
    let output_cost =
        (usage.output_tokens as f64 / 1_000_000.0) * cfg.usage.cost_per_million_output;

    let pending_tasks = store
        .list_tasks(None)?
        .into_iter()
        .filter(|task| {
            !task.status.eq_ignore_ascii_case("completed")
                && !task.status.eq_ignore_ascii_case("cancelled")
        })
        .count() as u64;

    let running_background_jobs = store
        .list_background_jobs()?
        .into_iter()
        .filter(|job| job.status.eq_ignore_ascii_case("running"))
        .count() as u64;

    let pending_approvals = store
        .load_latest_session()?
        .map(|session| {
            store
                .rebuild_from_events(session.session_id)
                .map(|projection| {
                    projection
                        .tool_invocations
                        .len()
                        .saturating_sub(projection.approved_invocations.len())
                        as u64
                })
        })
        .transpose()?
        .unwrap_or(0);

    let admin = EnterpriseAdminSummary {
        usage_records: usage.records,
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        estimated_cost_usd: input_cost + output_cost,
        pending_tasks,
        running_background_jobs,
        pending_approvals,
    };

    let mut recommendations = Vec::new();
    if !sso.configured {
        recommendations.push(
            "Configure OAuth2/SSO provider metadata for enterprise authentication.".to_string(),
        );
    }
    if !team_policy.active {
        recommendations.push(
            "Enable team policy locks to enforce permission and sandbox standards.".to_string(),
        );
    }
    if !managed_settings.loaded {
        recommendations.push(
            "Roll out managed settings for immutable enterprise-wide policy controls.".to_string(),
        );
    }
    if audit.events_total == 0 {
        recommendations.push(
            "Generate and retain audit events (events.jsonl) for compliance reporting.".to_string(),
        );
    }
    if audit.failed_verifications > 0 {
        recommendations.push(
            "Address failing verification runs to improve release readiness and compliance evidence."
                .to_string(),
        );
    }

    Ok(EnterpriseReport {
        sso,
        managed_settings,
        team_policy,
        audit,
        admin,
        recommendations,
    })
}

fn analyze_ecosystem(cwd: &Path, files: &[WorkspaceFile]) -> Result<EcosystemReport> {
    let mut signals: HashMap<String, SignalAccumulator> = HashMap::new();

    let github_workflows = matching_paths(
        files,
        |rel| {
            rel.starts_with(".github/workflows/")
                && (rel.ends_with(".yml") || rel.ends_with(".yaml"))
        },
        MAX_EVIDENCE,
    );
    if !github_workflows.is_empty() {
        add_signal(
            &mut signals,
            "GitHub",
            "scm",
            0.95,
            github_workflows
                .into_iter()
                .map(|path| format!("file:{path}")),
            [
                "Automate PR review summaries and CI status feedback in pull request workflows.",
                "Add required checks and CODEOWNERS review gates for critical paths.",
            ],
        );
    }

    let gitlab_files = matching_paths(
        files,
        |rel| rel == ".gitlab-ci.yml" || rel.starts_with(".gitlab/"),
        MAX_EVIDENCE,
    );
    if !gitlab_files.is_empty() {
        add_signal(
            &mut signals,
            "GitLab",
            "scm",
            0.93,
            gitlab_files.into_iter().map(|path| format!("file:{path}")),
            [
                "Add MR quality gates and release pipeline approvals for production branches.",
                "Use review apps for merge request validation before deploy stages.",
            ],
        );
    }

    for (env_key, name, recommendation) in [
        (
            "JIRA_API_TOKEN",
            "Jira",
            "Connect issue transitions to merge/deploy events for end-to-end traceability.",
        ),
        (
            "JIRA_BASE_URL",
            "Jira",
            "Store Jira project mappings for automated ticket updates.",
        ),
        (
            "LINEAR_API_KEY",
            "Linear",
            "Map PR and release events to Linear issue states.",
        ),
        (
            "ASANA_ACCESS_TOKEN",
            "Asana",
            "Automate task updates on CI and deployment milestones.",
        ),
    ] {
        if env_var_set(env_key) {
            add_signal(
                &mut signals,
                name,
                "work_tracking",
                0.9,
                [format!("env:{env_key}")],
                [recommendation],
            );
        }
    }

    let tracker_files = matching_paths(
        files,
        |rel| {
            rel.ends_with("jira.json")
                || rel.ends_with("linear.json")
                || rel.ends_with("asana.json")
                || rel.contains("/integrations/jira")
                || rel.contains("/integrations/linear")
                || rel.contains("/integrations/asana")
        },
        MAX_EVIDENCE,
    );
    for file in tracker_files {
        let lower = file.to_ascii_lowercase();
        if lower.contains("jira") {
            add_signal(
                &mut signals,
                "Jira",
                "work_tracking",
                0.82,
                [format!("file:{file}")],
                ["Synchronize ticket state changes with PR/MR workflow events."],
            );
        } else if lower.contains("linear") {
            add_signal(
                &mut signals,
                "Linear",
                "work_tracking",
                0.82,
                [format!("file:{file}")],
                ["Link CI failures back to affected Linear issues automatically."],
            );
        } else if lower.contains("asana") {
            add_signal(
                &mut signals,
                "Asana",
                "work_tracking",
                0.82,
                [format!("file:{file}")],
                ["Attach deployment metadata to Asana tasks for release audits."],
            );
        }
    }

    for (env_key, name, recommendation) in [
        (
            "SLACK_BOT_TOKEN",
            "Slack",
            "Publish build/test/deploy notifications with direct links to incidents and fixes.",
        ),
        (
            "SLACK_WEBHOOK_URL",
            "Slack",
            "Route high-severity deployment alerts to dedicated on-call channels.",
        ),
        (
            "TEAMS_WEBHOOK_URL",
            "Microsoft Teams",
            "Send release and rollback notifications to the engineering operations team.",
        ),
    ] {
        if env_var_set(env_key) {
            add_signal(
                &mut signals,
                name,
                "collaboration",
                0.89,
                [format!("env:{env_key}")],
                [recommendation],
            );
        }
    }

    let collaboration_files = matching_paths(
        files,
        |rel| {
            rel.starts_with(".slack/")
                || rel.contains("/integrations/slack")
                || rel.contains("/integrations/teams")
                || rel.ends_with("teams.json")
        },
        MAX_EVIDENCE,
    );
    for file in collaboration_files {
        if file.to_ascii_lowercase().contains("teams") {
            add_signal(
                &mut signals,
                "Microsoft Teams",
                "collaboration",
                0.79,
                [format!("file:{file}")],
                ["Document escalation channels for release incidents in Teams."],
            );
        } else {
            add_signal(
                &mut signals,
                "Slack",
                "collaboration",
                0.79,
                [format!("file:{file}")],
                ["Standardize Slack channels for CI and deployment events."],
            );
        }
    }

    let mut integrations = signals
        .into_iter()
        .map(|(name, acc)| IntegrationSignal {
            name,
            category: acc.category,
            confidence: acc.confidence,
            evidence: acc.evidence.into_iter().collect(),
            recommendations: acc.recommendations.into_iter().collect(),
        })
        .collect::<Vec<_>>();
    integrations.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.name.cmp(&b.name))
    });

    let coverage = EcosystemCoverage {
        scm: integrations.iter().any(|signal| signal.category == "scm"),
        work_tracking: integrations
            .iter()
            .any(|signal| signal.category == "work_tracking"),
        collaboration: integrations
            .iter()
            .any(|signal| signal.category == "collaboration"),
    };

    let mut recommendations = Vec::new();
    if !coverage.scm {
        recommendations.push(
            "Set up GitHub/GitLab integration for PR/MR-aware automation and review flows."
                .to_string(),
        );
    }
    if !coverage.work_tracking {
        recommendations.push(
            "Connect Jira, Linear, or Asana to keep ticket lifecycle aligned with code changes."
                .to_string(),
        );
    }
    if !coverage.collaboration {
        recommendations.push(
            "Configure Slack or Teams notifications for CI failures and deployment events."
                .to_string(),
        );
    }

    for signal in &integrations {
        for recommendation in &signal.recommendations {
            if !recommendations.contains(recommendation) {
                recommendations.push(recommendation.clone());
            }
        }
    }

    let _ = cwd;
    Ok(EcosystemReport {
        integrations,
        coverage,
        recommendations,
    })
}

fn analyze_deployment(cwd: &Path, files: &[WorkspaceFile]) -> Result<DeploymentReport> {
    let ci_files = matching_workspace_files(
        files,
        |rel| {
            rel == ".gitlab-ci.yml"
                || rel == "jenkinsfile"
                || rel == "azure-pipelines.yml"
                || rel == ".circleci/config.yml"
                || (rel.starts_with(".github/workflows/")
                    && (rel.ends_with(".yml") || rel.ends_with(".yaml")))
        },
        MAX_EVIDENCE,
    );

    let container_files = matching_workspace_files(
        files,
        |rel| {
            rel == "dockerfile"
                || rel.ends_with("/dockerfile")
                || rel.contains("/dockerfile.")
                || rel.starts_with("dockerfile.")
                || rel.ends_with("docker-compose.yml")
                || rel.ends_with("docker-compose.yaml")
        },
        MAX_EVIDENCE,
    );

    let kubernetes_files = matching_workspace_files(
        files,
        |rel| {
            rel.starts_with("k8s/")
                || rel.starts_with("kubernetes/")
                || rel.ends_with("chart.yaml")
                || rel.contains("/helm/")
        },
        MAX_EVIDENCE,
    );

    let cloud_files = matching_workspace_files(
        files,
        |rel| {
            rel.ends_with(".tf")
                || rel.ends_with(".tfvars")
                || rel.contains("cloudformation")
                || rel == "cdk.json"
                || rel.contains("pulumi")
        },
        MAX_EVIDENCE,
    );

    let coverage = DeploymentCoverage {
        ci: !ci_files.is_empty(),
        container: !container_files.is_empty(),
        kubernetes: !kubernetes_files.is_empty(),
        cloud: !cloud_files.is_empty(),
    };

    let mut signals = Vec::new();
    if !ci_files.is_empty() {
        signals.push(DeploymentSignal {
            area: "ci_cd".to_string(),
            confidence: 0.95,
            evidence: ci_files.iter().map(|file| file.rel.clone()).collect(),
            recommendations: vec![
                "Enforce branch protections and required CI checks before merges.".to_string(),
                "Add build, test, and security scan stages with explicit failure thresholds."
                    .to_string(),
            ],
        });
    }
    if !container_files.is_empty() {
        signals.push(DeploymentSignal {
            area: "containerization".to_string(),
            confidence: 0.92,
            evidence: container_files
                .iter()
                .map(|file| file.rel.clone())
                .collect(),
            recommendations: vec![
                "Pin container base image tags and run builds with non-root users.".to_string(),
            ],
        });
    }
    if !kubernetes_files.is_empty() {
        signals.push(DeploymentSignal {
            area: "kubernetes".to_string(),
            confidence: 0.9,
            evidence: kubernetes_files
                .iter()
                .map(|file| file.rel.clone())
                .collect(),
            recommendations: vec![
                "Add readiness/liveness probes and explicit resource requests/limits.".to_string(),
            ],
        });
    }
    if !cloud_files.is_empty() {
        signals.push(DeploymentSignal {
            area: "cloud_iac".to_string(),
            confidence: 0.88,
            evidence: cloud_files.iter().map(|file| file.rel.clone()).collect(),
            recommendations: vec![
                "Validate IaC plans in CI and require approval gates for production changes."
                    .to_string(),
            ],
        });
    }

    let mut risks = Vec::new();
    if !coverage.ci {
        risks.push("No CI/CD pipeline configuration detected.".to_string());
    }
    if !coverage.container && !coverage.kubernetes {
        risks.push(
            "No container or Kubernetes deployment configuration detected for production delivery."
                .to_string(),
        );
    }

    let dockerfiles = container_files
        .iter()
        .filter(|file| {
            file.rel_lower == "dockerfile"
                || file.rel_lower.ends_with("/dockerfile")
                || file.rel_lower.contains("/dockerfile.")
                || file.rel_lower.starts_with("dockerfile.")
        })
        .collect::<Vec<_>>();
    if !dockerfiles.is_empty() {
        let mut latest_tag_hits = Vec::new();
        let mut missing_user_hits = Vec::new();
        for file in dockerfiles {
            if let Ok(raw) = fs::read_to_string(&file.abs) {
                let lower = raw.to_ascii_lowercase();
                if lower.lines().any(|line| {
                    let trimmed = line.trim();
                    trimmed.starts_with("from ") && trimmed.contains(":latest")
                }) {
                    latest_tag_hits.push(file.rel.clone());
                }
                if !lower
                    .lines()
                    .any(|line| line.trim_start().starts_with("user "))
                {
                    missing_user_hits.push(file.rel.clone());
                }
            }
        }
        if !latest_tag_hits.is_empty() {
            risks.push(format!(
                "Container images pinned to ':latest' were found in: {}",
                latest_tag_hits.join(", ")
            ));
        }
        if !missing_user_hits.is_empty() {
            risks.push(format!(
                "Dockerfiles without explicit USER instruction found in: {}",
                missing_user_hits.join(", ")
            ));
        }
    }

    if coverage.kubernetes {
        let has_probe = kubernetes_files.iter().any(|file| {
            fs::read_to_string(&file.abs)
                .map(|raw| {
                    let lower = raw.to_ascii_lowercase();
                    lower.contains("livenessprobe") || lower.contains("readinessprobe")
                })
                .unwrap_or(false)
        });
        if !has_probe {
            risks.push(
                "Kubernetes manifests detected without readiness/liveness probes.".to_string(),
            );
        }
    }

    if coverage.ci {
        let has_test_step = ci_files.iter().any(|file| {
            fs::read_to_string(&file.abs)
                .map(|raw| {
                    let lower = raw.to_ascii_lowercase();
                    lower.contains("test")
                        || lower.contains("cargo test")
                        || lower.contains("pytest")
                })
                .unwrap_or(false)
        });
        if !has_test_step {
            risks.push(
                "CI configuration detected but no explicit test stage was found.".to_string(),
            );
        }
    }

    let mut recommendations = Vec::new();
    if !coverage.ci {
        recommendations.push(
            "Add a CI pipeline (GitHub Actions/GitLab CI/Jenkins) with build + test + lint gates."
                .to_string(),
        );
    }
    if !coverage.container {
        recommendations.push(
            "Add Dockerfile and reproducible container build stages for release artifacts."
                .to_string(),
        );
    }
    if !coverage.kubernetes {
        recommendations.push(
            "Add Kubernetes manifests or Helm charts for predictable runtime orchestration."
                .to_string(),
        );
    }
    if !coverage.cloud {
        recommendations.push(
            "Add Terraform/CloudFormation/Pulumi configuration for auditable cloud provisioning."
                .to_string(),
        );
    }
    if risks.iter().any(|risk| risk.contains(":latest")) {
        recommendations
            .push("Pin Docker base images to immutable versions or digests.".to_string());
    }
    if risks
        .iter()
        .any(|risk| risk.contains("without explicit USER"))
    {
        recommendations
            .push("Run containers with non-root users and least privilege by default.".to_string());
    }
    if risks
        .iter()
        .any(|risk| risk.contains("without readiness/liveness"))
    {
        recommendations.push(
            "Add readiness/liveness probes and resource limits to Kubernetes workloads."
                .to_string(),
        );
    }

    let _ = cwd;
    Ok(DeploymentReport {
        coverage,
        signals,
        risks,
        recommendations,
    })
}

fn compute_readiness(
    enterprise: &EnterpriseReport,
    ecosystem: &EcosystemReport,
    deployment: &DeploymentReport,
    warnings: &[String],
) -> LeadershipReadiness {
    let mut score = 100i32;
    let mut gaps = Vec::new();
    let mut risks = Vec::new();

    if !enterprise.sso.configured {
        score -= 15;
        gaps.push("sso_not_configured".to_string());
        risks.push("No SSO/OAuth configuration signals detected.".to_string());
    }
    if !enterprise.team_policy.active {
        score -= 10;
        gaps.push("team_policy_not_active".to_string());
        risks.push("Team policy locks are not active.".to_string());
    }
    if !enterprise.managed_settings.loaded {
        score -= 10;
        gaps.push("managed_settings_not_loaded".to_string());
    }
    if enterprise.audit.events_total == 0 {
        score -= 10;
        gaps.push("audit_log_empty".to_string());
        risks.push("Audit trail has no recorded events yet.".to_string());
    }

    if !ecosystem.coverage.scm {
        score -= 20;
        gaps.push("scm_integration_missing".to_string());
    }
    if !ecosystem.coverage.work_tracking {
        score -= 10;
        gaps.push("work_tracking_integration_missing".to_string());
    }
    if !ecosystem.coverage.collaboration {
        score -= 10;
        gaps.push("collaboration_integration_missing".to_string());
    }

    if !deployment.coverage.ci {
        score -= 20;
        gaps.push("ci_cd_missing".to_string());
    }
    if !deployment.coverage.container {
        score -= 10;
        gaps.push("container_config_missing".to_string());
    }
    if !deployment.coverage.cloud {
        score -= 5;
        gaps.push("cloud_iac_missing".to_string());
    }

    risks.extend(deployment.risks.clone());
    if enterprise.audit.failed_verifications > 0 {
        risks.push(format!(
            "{} verification run(s) failed; release controls may be unstable.",
            enterprise.audit.failed_verifications
        ));
    }
    risks.extend(warnings.iter().cloned());
    risks.sort();
    risks.dedup();

    let mut recommendations = dedupe_strings([
        enterprise.recommendations.clone(),
        ecosystem.recommendations.clone(),
        deployment.recommendations.clone(),
    ]);

    if recommendations.is_empty() {
        recommendations.push(
            "Maintain current controls and keep integrations/deployment checks in CI for regression prevention."
                .to_string(),
        );
    }

    let score = score.clamp(0, 100) as u8;
    let ok = score >= 70 && deployment.coverage.ci && warnings.is_empty();

    LeadershipReadiness {
        ok,
        score,
        risks,
        gaps,
        recommendations,
    }
}

fn detect_sso(cwd: &Path) -> SsoStatus {
    let mut providers = BTreeSet::new();
    let mut evidence = BTreeSet::new();

    for (env_key, provider) in [
        ("AUTH0_DOMAIN", "Auth0"),
        ("OKTA_DOMAIN", "Okta"),
        ("AZURE_TENANT_ID", "Azure AD"),
        ("GOOGLE_CLIENT_ID", "Google Workspace"),
        ("OIDC_ISSUER_URL", "OIDC"),
        ("SAML_SSO_URL", "SAML"),
    ] {
        if env_var_set(env_key) {
            providers.insert(provider.to_string());
            evidence.insert(format!("env:{env_key}"));
        }
    }

    for relative in [
        ".deepseek/sso.json",
        ".deepseek/auth.json",
        "auth.config.json",
        "oauth.json",
    ] {
        let path = cwd.join(relative);
        if !path.exists() {
            continue;
        }
        evidence.insert(format!("file:{relative}"));
        if let Ok(raw) = fs::read_to_string(path) {
            let lower = raw.to_ascii_lowercase();
            for (token, provider) in [
                ("auth0", "Auth0"),
                ("okta", "Okta"),
                ("azure", "Azure AD"),
                ("google", "Google Workspace"),
                ("oidc", "OIDC"),
                ("saml", "SAML"),
            ] {
                if lower.contains(token) {
                    providers.insert(provider.to_string());
                }
            }
        }
    }

    SsoStatus {
        configured: !providers.is_empty() || !evidence.is_empty(),
        providers: providers.into_iter().collect(),
        evidence: evidence.into_iter().collect(),
    }
}

fn build_team_policy_status() -> TeamPolicyStatus {
    if let Some(locks) = team_policy_locks() {
        let permission_fields_locked = locks.has_permission_locks();
        return TeamPolicyStatus {
            active: true,
            path: Some(locks.path),
            permission_fields_locked,
            approve_edits_locked: locks.approve_edits_locked,
            approve_bash_locked: locks.approve_bash_locked,
            allowlist_locked: locks.allowlist_locked,
            sandbox_mode_locked: locks.sandbox_mode_locked,
            permission_mode_locked: locks.permission_mode_locked,
        };
    }

    TeamPolicyStatus::default()
}

fn collect_audit_summary(cwd: &Path, audit_window_hours: u64) -> AuditSummary {
    let events_path = runtime_dir(cwd).join("events.jsonl");
    let mut summary = AuditSummary {
        events_path: events_path.display().to_string(),
        events_total: 0,
        events_in_window: 0,
        window_hours: audit_window_hours,
        parse_errors: 0,
        failed_verifications: 0,
        pending_approvals: 0,
        top_event_types: Vec::new(),
    };

    if !events_path.exists() {
        return summary;
    }

    let file = match fs::File::open(&events_path) {
        Ok(file) => file,
        Err(_) => {
            summary.parse_errors = summary.parse_errors.saturating_add(1);
            return summary;
        }
    };

    let safe_hours = audit_window_hours.min(24 * 365 * 2) as i64;
    let since = Utc::now() - Duration::hours(safe_hours);
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut proposed_tools = 0u64;
    let mut approved_tools = 0u64;

    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = match line {
            Ok(value) => value,
            Err(_) => {
                summary.parse_errors = summary.parse_errors.saturating_add(1);
                continue;
            }
        };
        let envelope: EventEnvelope = match deepseek_core::parse_event_envelope_compat(&line) {
            Ok(value) => value,
            Err(_) => {
                summary.parse_errors = summary.parse_errors.saturating_add(1);
                continue;
            }
        };
        summary.events_total = summary.events_total.saturating_add(1);
        if envelope.at >= since {
            summary.events_in_window = summary.events_in_window.saturating_add(1);
        }

        let event_type = event_type_name(&envelope.kind);
        *counts.entry(event_type).or_insert(0) += 1;

        match envelope.kind {
            EventKind::ToolProposedV1 { .. } => {
                proposed_tools = proposed_tools.saturating_add(1);
            }
            EventKind::ToolApprovedV1 { .. } => {
                approved_tools = approved_tools.saturating_add(1);
            }
            EventKind::VerificationRunV1 { success, .. } => {
                if !success {
                    summary.failed_verifications = summary.failed_verifications.saturating_add(1);
                }
            }
            _ => {}
        }
    }

    summary.pending_approvals = proposed_tools.saturating_sub(approved_tools);

    let mut event_counts = counts
        .into_iter()
        .map(|(event_type, count)| EventCount { event_type, count })
        .collect::<Vec<_>>();
    event_counts.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| a.event_type.cmp(&b.event_type))
    });
    event_counts.truncate(MAX_EVIDENCE);
    summary.top_event_types = event_counts;

    summary
}

fn event_type_name(kind: &EventKind) -> String {
    serde_json::to_value(kind)
        .ok()
        .and_then(|value| {
            value
                .get("type")
                .and_then(|entry| entry.as_str())
                .map(|entry| entry.to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn scan_workspace_files(root: &Path, max_files: usize) -> Vec<WorkspaceFile> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        if out.len() >= max_files {
            break;
        }
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => continue,
            };

            if file_type.is_dir() {
                if should_skip_dir(&path) {
                    continue;
                }
                stack.push(path);
                continue;
            }

            if !file_type.is_file() {
                continue;
            }

            let rel = match path.strip_prefix(root) {
                Ok(rel) => rel,
                Err(_) => continue,
            };
            let rel_render = rel.to_string_lossy().replace('\\', "/");
            let rel_lower = rel_render.to_ascii_lowercase();
            out.push(WorkspaceFile {
                abs: path,
                rel: rel_render,
                rel_lower,
            });

            if out.len() >= max_files {
                break;
            }
        }
    }

    out.sort_by(|a, b| a.rel_lower.cmp(&b.rel_lower));
    out
}

fn should_skip_dir(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    matches!(
        name,
        ".git"
            | "target"
            | "node_modules"
            | ".deepseek"
            | ".idea"
            | ".vscode"
            | "dist"
            | "build"
            | "tmp"
    )
}

fn env_var_set(key: &str) -> bool {
    std::env::var(key)
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
}

fn add_signal<E, R, ES, RS>(
    signals: &mut HashMap<String, SignalAccumulator>,
    name: &str,
    category: &str,
    confidence: f32,
    evidence: E,
    recommendations: R,
) where
    E: IntoIterator<Item = ES>,
    ES: Into<String>,
    R: IntoIterator<Item = RS>,
    RS: AsRef<str>,
{
    let entry = signals.entry(name.to_string()).or_default();
    if entry.category.is_empty() {
        entry.category = category.to_string();
    }
    entry.confidence = entry.confidence.max(confidence);
    for value in evidence {
        entry.evidence.insert(value.into());
    }
    for recommendation in recommendations {
        entry
            .recommendations
            .insert(recommendation.as_ref().to_string());
    }
}

fn matching_paths<F>(files: &[WorkspaceFile], mut predicate: F, max: usize) -> Vec<String>
where
    F: FnMut(&str) -> bool,
{
    let mut out = Vec::new();
    for file in files {
        if predicate(&file.rel_lower) {
            out.push(file.rel.clone());
            if out.len() >= max {
                break;
            }
        }
    }
    out
}

fn matching_workspace_files<F>(
    files: &[WorkspaceFile],
    mut predicate: F,
    max: usize,
) -> Vec<WorkspaceFile>
where
    F: FnMut(&str) -> bool,
{
    let mut out = Vec::new();
    for file in files {
        if predicate(&file.rel_lower) {
            out.push(file.clone());
            if out.len() >= max {
                break;
            }
        }
    }
    out
}

fn dedupe_strings<const N: usize>(groups: [Vec<String>; N]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for group in groups {
        for value in group {
            if seen.insert(value.clone()) {
                out.push(value);
            }
        }
    }
    out
}

fn build_managed_settings_status_from(
    managed: Option<&ManagedSettings>,
    path: Option<&Path>,
) -> ManagedSettingsStatus {
    ManagedSettingsStatus {
        path: path.map(|value| value.to_string_lossy().to_string()),
        exists: path.is_some_and(|value| value.exists()),
        loaded: managed.is_some(),
        forced_permission_mode: managed.and_then(|value| value.permission_mode.clone()),
        allow_managed_permission_rules_only: managed
            .map(|value| value.allow_managed_permission_rules_only)
            .unwrap_or(false),
        allow_managed_hooks_only: managed
            .map(|value| value.allow_managed_hooks_only)
            .unwrap_or(false),
        disable_bypass_permissions_mode: managed
            .map(|value| value.disable_bypass_permissions_mode)
            .unwrap_or(false),
    }
}

fn build_managed_settings_status() -> ManagedSettingsStatus {
    let path = managed_settings_path();
    let managed = load_managed_settings();
    build_managed_settings_status_from(managed.as_ref(), path.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn scan_workspace_files_skips_target_directory() {
        let dir = TempDir::new().expect("temp dir");
        fs::create_dir_all(dir.path().join("target/sub")).expect("target dir");
        fs::create_dir_all(dir.path().join("src")).expect("src dir");
        fs::write(dir.path().join("target/sub/ignored.txt"), "x").expect("ignored file");
        fs::write(dir.path().join("src/main.rs"), "fn main() {}\n").expect("main file");

        let files = scan_workspace_files(dir.path(), 100);
        assert!(files.iter().any(|entry| entry.rel == "src/main.rs"));
        assert!(!files.iter().any(|entry| entry.rel.contains("ignored.txt")));
    }

    #[test]
    fn managed_settings_status_handles_absent_values() {
        let status = build_managed_settings_status_from(None, None);
        assert!(!status.loaded);
        assert!(!status.exists);
    }
}
