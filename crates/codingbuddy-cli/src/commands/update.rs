use anyhow::Result;
use serde_json::json;
use std::path::Path;
use std::process::Command;

use crate::output::print_json;
use crate::{UpdateArgs, UpdateChannelArg};

pub(crate) fn run_update(cwd: &Path, args: UpdateArgs, json_mode: bool) -> Result<()> {
    let current_version = env!("CARGO_PKG_VERSION").to_string();

    let git_repo = Command::new("git")
        .current_dir(cwd)
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
        .ok()
        .is_some_and(|out| out.status.success());

    let branch = if git_repo {
        capture(cwd, "git", &["branch", "--show-current"])
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .or_else(|| Some("main".to_string()))
    } else {
        None
    };

    let mut updated = false;
    let mut steps = Vec::new();
    let mut warnings = Vec::new();

    if git_repo {
        if args.check {
            steps.push("check_only".to_string());
        } else if args.dry_run {
            steps.push("dry_run_git_pull_ff_only".to_string());
        } else {
            let pull = Command::new("git")
                .current_dir(cwd)
                .args(["pull", "--ff-only"])
                .output();
            match pull {
                Ok(out) if out.status.success() => {
                    updated = true;
                    steps.push("git_pull_ff_only".to_string());
                }
                Ok(out) => {
                    warnings.push(format!(
                        "git pull --ff-only failed: {}",
                        String::from_utf8_lossy(&out.stderr).trim()
                    ));
                    steps.push("git_pull_failed".to_string());
                }
                Err(err) => {
                    warnings.push(format!("failed to execute git pull: {err}"));
                    steps.push("git_pull_failed".to_string());
                }
            }
        }
    } else if !args.check {
        steps.push("manual_update_required".to_string());
        warnings.push(
            "not running from a git checkout; run your package-manager update flow".to_string(),
        );
    }

    let channel = match args.channel {
        UpdateChannelArg::Stable => "stable",
        UpdateChannelArg::Nightly => "nightly",
    };

    let payload = json!({
        "schema": "deepseek.update.v1",
        "current_version": current_version,
        "channel": channel,
        "source": if git_repo { "git" } else { "packaged" },
        "branch": branch,
        "check_only": args.check,
        "dry_run": args.dry_run,
        "updated": updated,
        "steps": steps,
        "warnings": warnings,
        "suggested_commands": if git_repo {
            vec!["git pull --ff-only"]
        } else {
            vec!["cargo install --path . --force --locked"]
        }
    });

    if json_mode {
        print_json(&payload)?;
    } else {
        println!(
            "update: version={} source={} channel={} updated={}",
            payload["current_version"].as_str().unwrap_or_default(),
            payload["source"].as_str().unwrap_or_default(),
            payload["channel"].as_str().unwrap_or_default(),
            payload["updated"].as_bool().unwrap_or(false)
        );
        if let Some(warnings) = payload["warnings"].as_array()
            && !warnings.is_empty()
        {
            println!("warnings:");
            for warning in warnings {
                if let Some(text) = warning.as_str() {
                    println!("- {text}");
                }
            }
        }
        if let Some(commands) = payload["suggested_commands"].as_array()
            && !commands.is_empty()
        {
            println!("suggested:");
            for command in commands {
                if let Some(text) = command.as_str() {
                    println!("- {text}");
                }
            }
        }
    }

    Ok(())
}

fn capture(cwd: &Path, program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program)
        .current_dir(cwd)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let rendered = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if rendered.is_empty() {
        return None;
    }
    Some(rendered)
}
