//! CLI commands for privacy scanning and policy management.

use anyhow::Result;
use deepseek_core::AppConfig;
use deepseek_local_ml::{PrivacyConfig, PrivacyPolicy, PrivacyRouter};
use serde_json::json;
use std::path::Path;

use crate::output::print_json;

/// Privacy subcommands.
#[derive(clap::Subcommand)]
pub(crate) enum PrivacyCmd {
    /// Scan the workspace for sensitive files and content.
    Scan,
    /// Show or test the privacy policy.
    Policy {
        #[command(subcommand)]
        action: PolicyAction,
    },
}

#[derive(clap::Subcommand)]
pub(crate) enum PolicyAction {
    /// Show the current privacy policy configuration.
    Show,
    /// Test privacy scanning on a specific file.
    Test {
        /// Path to test (relative to workspace).
        path: String,
    },
}

pub(crate) fn run_privacy(cwd: &Path, cmd: PrivacyCmd, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::load(cwd).unwrap_or_default();

    match cmd {
        PrivacyCmd::Scan => run_scan(cwd, &cfg, json_mode),
        PrivacyCmd::Policy { action } => match action {
            PolicyAction::Show => run_policy_show(&cfg, json_mode),
            PolicyAction::Test { path } => run_policy_test(cwd, &cfg, &path, json_mode),
        },
    }
}

fn build_router(cfg: &AppConfig) -> Result<PrivacyRouter> {
    let privacy_cfg = PrivacyConfig {
        enabled: cfg.local_ml.privacy.enabled,
        sensitive_globs: cfg.local_ml.privacy.sensitive_globs.clone(),
        sensitive_regex: cfg.local_ml.privacy.sensitive_regex.clone(),
        policy: match cfg.local_ml.privacy.policy.as_str() {
            "block" | "block_cloud" => PrivacyPolicy::BlockCloud,
            "local_summary" => PrivacyPolicy::LocalOnlySummary,
            _ => PrivacyPolicy::Redact,
        },
        store_raw_in_logs: cfg.local_ml.privacy.store_raw_in_logs,
    };
    PrivacyRouter::new(privacy_cfg)
}

fn run_scan(cwd: &Path, cfg: &AppConfig, json_mode: bool) -> Result<()> {
    let router = build_router(cfg)?;
    let mut findings = Vec::new();

    // Walk the workspace looking for sensitive files and content
    let walker = ignore::WalkBuilder::new(cwd)
        .hidden(true)
        .git_ignore(true)
        .build();

    for entry in walker.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let rel = path.strip_prefix(cwd).unwrap_or(path);
        let rel_str = rel.to_string_lossy().to_string();

        // Check path sensitivity
        if router.is_sensitive_path(&rel_str) {
            findings.push(json!({
                "file": rel_str,
                "reason": "sensitive_path",
                "details": "path matches sensitive glob pattern",
            }));
            continue;
        }

        // Check content for smaller text files
        if let Ok(meta) = path.metadata()
            && meta.len() < 1_048_576
            && let Ok(content) = std::fs::read_to_string(path)
        {
            let matches = router.scan_content(&content);
            if !matches.is_empty() {
                for m in &matches {
                    findings.push(json!({
                        "file": rel_str,
                        "reason": "sensitive_content",
                        "pattern": m.pattern,
                        "line": m.line_number,
                        "preview": m.redacted_preview,
                    }));
                }
            }
        }
    }

    if json_mode {
        print_json(&json!({
            "scan_results": findings,
            "total_findings": findings.len(),
        }))?;
    } else if findings.is_empty() {
        println!("No sensitive content found.");
    } else {
        println!("Found {} sensitive items:", findings.len());
        for f in &findings {
            let file = f["file"].as_str().unwrap_or("?");
            let reason = f["reason"].as_str().unwrap_or("?");
            if let Some(line) = f["line"].as_u64() {
                println!("  {} (line {}) — {}", file, line, reason);
            } else {
                println!("  {} — {}", file, reason);
            }
        }
    }
    Ok(())
}

fn run_policy_show(cfg: &AppConfig, json_mode: bool) -> Result<()> {
    let policy_info = json!({
        "enabled": cfg.local_ml.privacy.enabled,
        "policy": cfg.local_ml.privacy.policy,
        "sensitive_globs": cfg.local_ml.privacy.sensitive_globs,
        "sensitive_regex": cfg.local_ml.privacy.sensitive_regex,
        "store_raw_in_logs": cfg.local_ml.privacy.store_raw_in_logs,
    });

    if json_mode {
        print_json(&policy_info)?;
    } else {
        println!("Privacy policy:");
        println!("  enabled: {}", cfg.local_ml.privacy.enabled);
        println!("  policy: {}", cfg.local_ml.privacy.policy);
        println!(
            "  sensitive globs: {:?}",
            cfg.local_ml.privacy.sensitive_globs
        );
        println!(
            "  sensitive regex: {:?}",
            cfg.local_ml.privacy.sensitive_regex
        );
        println!(
            "  store raw in logs: {}",
            cfg.local_ml.privacy.store_raw_in_logs
        );
    }
    Ok(())
}

fn run_policy_test(cwd: &Path, cfg: &AppConfig, path: &str, json_mode: bool) -> Result<()> {
    let router = build_router(cfg)?;
    let full_path = cwd.join(path);

    let mut result = json!({
        "path": path,
        "path_sensitive": router.is_sensitive_path(path),
    });

    if full_path.is_file() {
        if let Ok(content) = std::fs::read_to_string(&full_path) {
            let matches = router.scan_content(&content);
            let policy_result = router.apply_policy(&content, Some(path));
            result["content_matches"] = json!(matches.len());
            result["policy_action"] = json!(match policy_result {
                deepseek_local_ml::PrivacyResult::Clean(_) => "clean",
                deepseek_local_ml::PrivacyResult::Redacted(_) => "redacted",
                deepseek_local_ml::PrivacyResult::Blocked => "blocked",
                deepseek_local_ml::PrivacyResult::LocalSummary(_) => "local_summary",
            });
        } else {
            result["error"] = json!("could not read file");
        }
    } else {
        result["error"] = json!("file not found");
    }

    if json_mode {
        print_json(&result)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&result)?);
    }
    Ok(())
}
