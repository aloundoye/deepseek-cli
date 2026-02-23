use anyhow::{Context as _, Result, anyhow};
use chrono::Utc;
use deepseek_agent::{AgentEngine, ChatOptions};
use deepseek_core::EventKind;
use deepseek_store::{ReviewRunRecord, Store};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::process::Command;
use uuid::Uuid;

use crate::ReviewArgs;
use crate::context::*;
use crate::output::*;
use crate::util::*;

const REVIEW_SCHEMA: &str = "deepseek.review.findings.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReviewFinding {
    id: String,
    severity: String,
    #[serde(default)]
    file: Option<String>,
    #[serde(default)]
    line: Option<u64>,
    title: String,
    body: String,
    #[serde(default)]
    suggestion: String,
    #[serde(default)]
    confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReviewFindingsPayload {
    schema: String,
    findings: Vec<ReviewFinding>,
    #[serde(default)]
    summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PublishSummary {
    schema: String,
    pr_number: u64,
    dry_run: bool,
    inline_comments: usize,
    summary_comment: bool,
    actions: Vec<String>,
}

pub(crate) fn run_review(cwd: &Path, args: ReviewArgs, json_mode: bool) -> Result<()> {
    ensure_llm_ready(cwd, json_mode)?;
    if args.publish && !args.strict {
        return Err(anyhow!(
            "--publish requires strict findings mode; use --strict=true (default)"
        ));
    }

    let diff_content = load_review_target(cwd, &args)?;
    if diff_content.trim().is_empty() {
        if json_mode {
            print_json(
                &json!({"schema": REVIEW_SCHEMA, "findings": [], "summary": "no changes to review"}),
            )?;
        } else {
            println!("no changes to review");
        }
        return Ok(());
    }

    let focus = args
        .focus
        .as_deref()
        .unwrap_or("correctness, security, performance, style");
    let target = review_target_label(&args);
    let review_id = Uuid::now_v7();

    append_control_event(
        cwd,
        EventKind::ReviewStartedV1 {
            review_id,
            preset: focus.to_string(),
            target: target.clone(),
        },
    )?;

    let prompt = build_strict_review_prompt(focus, &diff_content);
    let engine = AgentEngine::new(cwd)?;
    let output = engine.analyze_with_options(
        &prompt,
        ChatOptions {
            tools: false,
            ..Default::default()
        },
    )?;

    let mut publish_summary = None;
    let payload = if args.strict {
        let payload = match parse_review_findings_payload(&output) {
            Ok(payload) => {
                let _ = append_control_event(
                    cwd,
                    EventKind::TelemetryEventV1 {
                        name: "kpi.review.strict_parse".to_string(),
                        properties: json!({"success": true}),
                    },
                );
                payload
            }
            Err(err) => {
                let _ = append_control_event(
                    cwd,
                    EventKind::TelemetryEventV1 {
                        name: "kpi.review.strict_parse".to_string(),
                        properties: json!({
                            "success": false,
                            "error": err.to_string(),
                        }),
                    },
                );
                return Err(err);
            }
        };
        validate_review_findings(&payload)?;

        let findings_count = payload.findings.len() as u64;
        let critical_count = payload
            .findings
            .iter()
            .filter(|finding| finding.severity == "critical")
            .count() as u64;

        let store = Store::new(cwd)?;
        let session = ensure_session_record(cwd, &store)?;
        store.insert_review_run(&ReviewRunRecord {
            review_id,
            session_id: session.session_id,
            preset: focus.to_string(),
            target: target.clone(),
            findings_json: serde_json::to_string(&payload)?,
            findings_count,
            critical_count,
            created_at: Utc::now().to_rfc3339(),
        })?;

        append_control_event(
            cwd,
            EventKind::ReviewCompletedV1 {
                review_id,
                findings_count,
                critical_count,
            },
        )?;

        if args.publish {
            let pr_number = args
                .pr
                .ok_or_else(|| anyhow!("--publish requires --pr <number>"))?;
            let summary = match publish_findings_with_gh(
                cwd,
                pr_number,
                &payload.findings,
                args.max_comments.max(1),
                args.dry_run,
            ) {
                Ok(summary) => {
                    let _ = append_control_event(
                        cwd,
                        EventKind::TelemetryEventV1 {
                            name: "kpi.review.publish".to_string(),
                            properties: json!({
                                "success": true,
                                "dry_run": args.dry_run,
                                "pr": pr_number,
                                "inline_comments": summary.inline_comments,
                                "summary_comment": summary.summary_comment,
                            }),
                        },
                    );
                    summary
                }
                Err(err) => {
                    let _ = append_control_event(
                        cwd,
                        EventKind::TelemetryEventV1 {
                            name: "kpi.review.publish".to_string(),
                            properties: json!({
                                "success": false,
                                "dry_run": args.dry_run,
                                "pr": pr_number,
                                "error": err.to_string(),
                            }),
                        },
                    );
                    return Err(err);
                }
            };
            append_control_event(
                cwd,
                EventKind::ReviewPublishedV1 {
                    review_id,
                    pr_number,
                    comments_published: summary.inline_comments as u64
                        + u64::from(summary.summary_comment),
                    dry_run: summary.dry_run,
                },
            )?;
            publish_summary = Some(summary);
        }

        json!({
            "schema": REVIEW_SCHEMA,
            "review_id": review_id,
            "target": target,
            "focus": focus,
            "findings_count": findings_count,
            "critical_count": critical_count,
            "findings": payload.findings,
            "summary": payload.summary,
            "publish": publish_summary,
        })
    } else {
        json!({
            "schema": "deepseek.review.freeform.v1",
            "review_id": review_id,
            "target": target,
            "focus": focus,
            "review": output,
            "publish": null
        })
    };

    if json_mode {
        print_json(&payload)?;
    } else {
        render_review_text(&payload);
    }
    Ok(())
}

fn load_review_target(cwd: &Path, args: &ReviewArgs) -> Result<String> {
    let diff_content = if let Some(pr_number) = args.pr {
        let output = Command::new("gh")
            .args(["pr", "diff", &pr_number.to_string()])
            .current_dir(cwd)
            .output()
            .map_err(|_| anyhow!("gh CLI not found; install it for PR review support"))?;
        if !output.status.success() {
            return Err(anyhow!(
                "gh pr diff failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        String::from_utf8_lossy(&output.stdout).to_string()
    } else if let Some(ref path) = args.path {
        fs::read_to_string(cwd.join(path)).map_err(|e| anyhow!("cannot read {path}: {e}"))?
    } else if args.staged {
        run_capture("git", &["diff", "--staged"]).unwrap_or_default()
    } else {
        run_capture("git", &["diff"]).unwrap_or_default()
    };
    Ok(diff_content)
}

fn review_target_label(args: &ReviewArgs) -> String {
    if let Some(pr) = args.pr {
        format!("pr:{pr}")
    } else if let Some(path) = args.path.as_deref() {
        format!("path:{path}")
    } else if args.staged {
        "git:staged".to_string()
    } else {
        "git:diff".to_string()
    }
}

fn build_strict_review_prompt(focus: &str, diff_content: &str) -> String {
    format!(
        "You are a principal code reviewer. Analyze the diff and return ONLY strict JSON.\n\
         Focus areas: {focus}\n\n\
         Output schema:\n\
         {{\n\
           \"schema\": \"{REVIEW_SCHEMA}\",\n\
           \"summary\": \"short overall assessment\",\n\
           \"findings\": [\n\
             {{\n\
               \"id\": \"stable-id\",\n\
               \"severity\": \"critical|warning|suggestion\",\n\
               \"file\": \"path/or/null\",\n\
               \"line\": 123,\n\
               \"title\": \"short title\",\n\
               \"body\": \"one-paragraph explanation\",\n\
               \"suggestion\": \"concrete fix\",\n\
               \"confidence\": 0.0\n\
             }}\n\
           ]\n\
         }}\n\
         Rules:\n\
         - Return valid JSON only, no markdown fences.\n\
         - `confidence` must be between 0 and 1.\n\
         - Use `file`+`line` when location is known.\n\
         - If no issues, return findings as [] with a summary.\n\n\
         Diff:\n\
         ```diff\n{diff_content}\n```"
    )
}

fn parse_review_findings_payload(raw: &str) -> Result<ReviewFindingsPayload> {
    let mut candidates = vec![raw.trim().to_string()];
    if let Some(snippet) = extract_json_snippet(raw) {
        candidates.push(snippet);
    }
    let mut last_error: Option<anyhow::Error> = None;

    for candidate in candidates {
        if candidate.is_empty() {
            continue;
        }
        let value = match serde_json::from_str::<serde_json::Value>(&candidate) {
            Ok(value) => value,
            Err(err) => {
                last_error = Some(anyhow!("invalid JSON: {err}"));
                continue;
            }
        };
        let Some(schema) = value.get("schema").and_then(|v| v.as_str()) else {
            last_error = Some(anyhow!("review output missing required 'schema' field"));
            continue;
        };
        let Some(findings_value) = value.get("findings") else {
            last_error = Some(anyhow!("review output missing required 'findings' field"));
            continue;
        };
        let findings = match serde_json::from_value::<Vec<ReviewFinding>>(findings_value.clone()) {
            Ok(findings) => findings,
            Err(err) => {
                last_error = Some(anyhow!(
                    "review output contains invalid findings array: {err}"
                ));
                continue;
            }
        };
        let summary = value
            .get("summary")
            .and_then(|v| v.as_str())
            .map(ToString::to_string);
        return Ok(ReviewFindingsPayload {
            schema: schema.to_string(),
            findings,
            summary,
        });
    }

    if let Some(err) = last_error {
        return Err(err.context(format!(
            "strict review parsing failed: model output is not valid {} JSON",
            REVIEW_SCHEMA
        )));
    }

    Err(anyhow!(
        "strict review parsing failed: model output is not valid {} JSON",
        REVIEW_SCHEMA
    ))
}

fn extract_json_snippet(raw: &str) -> Option<String> {
    let first_obj = raw.find('{');
    let last_obj = raw.rfind('}');
    if let (Some(start), Some(end)) = (first_obj, last_obj)
        && start < end
    {
        return Some(raw[start..=end].to_string());
    }
    let first_arr = raw.find('[');
    let last_arr = raw.rfind(']');
    if let (Some(start), Some(end)) = (first_arr, last_arr)
        && start < end
    {
        return Some(raw[start..=end].to_string());
    }
    None
}

fn validate_review_findings(payload: &ReviewFindingsPayload) -> Result<()> {
    if payload.schema != REVIEW_SCHEMA {
        return Err(anyhow!(
            "invalid review schema '{}'; expected '{}'",
            payload.schema,
            REVIEW_SCHEMA
        ));
    }
    let allowed: HashSet<&str> = ["critical", "warning", "suggestion"].into_iter().collect();
    for finding in &payload.findings {
        if finding.id.trim().is_empty() {
            return Err(anyhow!("finding id is required"));
        }
        if !allowed.contains(finding.severity.as_str()) {
            return Err(anyhow!(
                "invalid finding severity '{}'; allowed: critical|warning|suggestion",
                finding.severity
            ));
        }
        if finding.title.trim().is_empty() {
            return Err(anyhow!("finding title is required"));
        }
        if finding.body.trim().is_empty() {
            return Err(anyhow!("finding body is required"));
        }
        if !(0.0..=1.0).contains(&finding.confidence) {
            return Err(anyhow!(
                "finding confidence out of range [0,1]: {}",
                finding.confidence
            ));
        }
        if let Some(line) = finding.line
            && line == 0
        {
            return Err(anyhow!("finding line must be >= 1"));
        }
    }
    Ok(())
}

fn publish_findings_with_gh(
    cwd: &Path,
    pr_number: u64,
    findings: &[ReviewFinding],
    max_comments: usize,
    dry_run: bool,
) -> Result<PublishSummary> {
    let gh_available = command_exists("gh");
    if !gh_available {
        return Err(anyhow!(
            "GitHub CLI ('gh') is required for --publish. Install gh and authenticate first."
        ));
    }

    let repo = run_process(
        cwd,
        "gh",
        &[
            "repo",
            "view",
            "--json",
            "nameWithOwner",
            "-q",
            ".nameWithOwner",
        ],
    )
    .context("failed to resolve GitHub repository")?;
    let head_oid = run_process(
        cwd,
        "gh",
        &[
            "pr",
            "view",
            &pr_number.to_string(),
            "--json",
            "headRefOid",
            "-q",
            ".headRefOid",
        ],
    )
    .context("failed to resolve PR head commit")?;

    let mut actions = Vec::new();
    let mut inline_comments = 0usize;
    let mut overflow = Vec::new();

    for finding in findings {
        if inline_comments >= max_comments {
            overflow.push(finding);
            continue;
        }
        let (Some(file), Some(line)) = (finding.file.as_deref(), finding.line) else {
            overflow.push(finding);
            continue;
        };
        let body = format!(
            "[{}] {}\n\n{}\n\nSuggestion: {}\nConfidence: {:.2}",
            finding.severity.to_ascii_uppercase(),
            finding.title,
            finding.body,
            if finding.suggestion.trim().is_empty() {
                "n/a"
            } else {
                &finding.suggestion
            },
            finding.confidence
        );
        actions.push(format!("inline:{}:{}#{}", file, line, finding.id));
        if !dry_run {
            let endpoint = format!("repos/{repo}/pulls/{pr_number}/comments");
            let status = Command::new("gh")
                .current_dir(cwd)
                .args([
                    "api",
                    "-X",
                    "POST",
                    &endpoint,
                    "-f",
                    &format!("body={body}"),
                    "-f",
                    &format!("commit_id={head_oid}"),
                    "-f",
                    &format!("path={file}"),
                    "-F",
                    &format!("line={line}"),
                    "-f",
                    "side=RIGHT",
                ])
                .output()
                .context("failed to run gh api for inline review comment")?;
            if !status.status.success() {
                overflow.push(finding);
                actions.push(format!(
                    "inline-fallback:{}",
                    String::from_utf8_lossy(&status.stderr).trim()
                ));
                continue;
            }
        }
        inline_comments += 1;
    }

    let mut summary_lines = Vec::new();
    for finding in overflow {
        summary_lines.push(format!(
            "- [{}] {}{}{}",
            finding.severity.to_ascii_uppercase(),
            finding.title,
            finding
                .file
                .as_ref()
                .map(|f| format!(" ({f}"))
                .unwrap_or_default(),
            finding
                .line
                .map(|l| format!(":{l})"))
                .unwrap_or_else(|| if finding.file.is_some() {
                    ")".to_string()
                } else {
                    String::new()
                })
        ));
    }
    let summary_comment = !summary_lines.is_empty();
    if summary_comment {
        let body = format!(
            "DeepSeek review summary for PR #{pr_number}:\n\n{}",
            summary_lines.join("\n")
        );
        actions.push("summary-comment".to_string());
        if !dry_run {
            let status = Command::new("gh")
                .current_dir(cwd)
                .args(["pr", "comment", &pr_number.to_string(), "--body", &body])
                .output()
                .context("failed to run gh pr comment")?;
            if !status.status.success() {
                return Err(anyhow!(
                    "failed to publish summary review comment: {}",
                    String::from_utf8_lossy(&status.stderr).trim()
                ));
            }
        }
    }

    Ok(PublishSummary {
        schema: "deepseek.review.publish.v1".to_string(),
        pr_number,
        dry_run,
        inline_comments,
        summary_comment,
        actions,
    })
}

fn render_review_text(payload: &serde_json::Value) {
    if payload["schema"] == REVIEW_SCHEMA {
        let findings = payload["findings"].as_array().cloned().unwrap_or_default();
        let critical_count = payload["critical_count"].as_u64().unwrap_or(0);
        println!(
            "review findings: {} (critical: {})",
            findings.len(),
            critical_count
        );
        for finding in findings {
            let sev = finding["severity"].as_str().unwrap_or("warning");
            let title = finding["title"].as_str().unwrap_or_default();
            let file = finding["file"].as_str().unwrap_or_default();
            let line = finding["line"].as_u64().unwrap_or(0);
            if !file.is_empty() && line > 0 {
                println!("- [{}] {} ({}:{})", sev, title, file, line);
            } else {
                println!("- [{}] {}", sev, title);
            }
        }
        if let Some(summary) = payload["summary"].as_str()
            && !summary.is_empty()
        {
            println!("\nsummary: {summary}");
        }
        if !payload["publish"].is_null() {
            println!(
                "\npublish: {}",
                serde_json::to_string_pretty(&payload["publish"]).unwrap_or_default()
            );
        }
    } else if let Some(text) = payload["review"].as_str() {
        println!("{text}");
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(payload).unwrap_or_default()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_review_payload_accepts_schema_object() {
        let raw = r#"{"schema":"deepseek.review.findings.v1","summary":"ok","findings":[{"id":"f1","severity":"warning","file":"src/lib.rs","line":10,"title":"t","body":"b","suggestion":"s","confidence":0.8}]}"#;
        let payload = parse_review_findings_payload(raw).expect("payload");
        assert_eq!(payload.findings.len(), 1);
        assert_eq!(payload.findings[0].id, "f1");
    }

    #[test]
    fn parse_review_payload_requires_schema() {
        let raw = r#"{"findings":[]}"#;
        let err = parse_review_findings_payload(raw).expect_err("missing schema");
        assert!(err.to_string().contains("strict review parsing failed"));
    }

    #[test]
    fn parse_review_payload_rejects_invalid_severity() {
        let payload = ReviewFindingsPayload {
            schema: REVIEW_SCHEMA.to_string(),
            summary: None,
            findings: vec![ReviewFinding {
                id: "x".to_string(),
                severity: "info".to_string(),
                file: Some("a.rs".to_string()),
                line: Some(1),
                title: "title".to_string(),
                body: "body".to_string(),
                suggestion: String::new(),
                confidence: 0.5,
            }],
        };
        let err = validate_review_findings(&payload).expect_err("invalid severity");
        assert!(err.to_string().contains("invalid finding severity"));
    }

    #[test]
    fn extract_json_snippet_grabs_object_from_markdown() {
        let raw = "```json\n{\"schema\":\"deepseek.review.findings.v1\",\"findings\":[]}\n```";
        let snippet = extract_json_snippet(raw).expect("snippet");
        assert!(snippet.starts_with('{'));
        assert!(snippet.ends_with('}'));
    }
}
