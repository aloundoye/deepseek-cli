use anyhow::{Result, anyhow};
use serde::Serialize;
use serde_json::json;
use std::path::Path;

use crate::GitCmd;
use crate::output::*;
use crate::util::*;

#[derive(Debug, Default, Serialize)]
pub(crate) struct GitStatusSummary {
    pub(crate) branch: Option<String>,
    pub(crate) ahead: u64,
    pub(crate) behind: u64,
    pub(crate) staged: u64,
    pub(crate) unstaged: u64,
    pub(crate) untracked: u64,
    pub(crate) conflicts: u64,
}

pub(crate) fn run_git(cwd: &Path, cmd: GitCmd, json_mode: bool) -> Result<()> {
    match cmd {
        GitCmd::Status => {
            let porcelain = run_process(cwd, "git", &["status", "--porcelain=2", "--branch"])?;
            let summary = parse_git_status_summary(&porcelain);
            let output = run_process(cwd, "git", &["status", "--short"])?;
            if json_mode {
                print_json(&json!({
                    "command":"git status --short",
                    "output": output,
                    "summary": summary
                }))?;
            } else {
                println!(
                    "branch={} ahead={} behind={} staged={} unstaged={} untracked={} conflicts={}",
                    summary.branch.as_deref().unwrap_or("detached"),
                    summary.ahead,
                    summary.behind,
                    summary.staged,
                    summary.unstaged,
                    summary.untracked,
                    summary.conflicts
                );
                println!("{output}");
            }
        }
        GitCmd::History(args) => {
            let output = run_process(
                cwd,
                "git",
                &["log", "--oneline", "-n", &args.limit.to_string()],
            )?;
            if json_mode {
                print_json(&json!({"limit": args.limit, "output": output}))?;
            } else {
                println!("{output}");
            }
        }
        GitCmd::Branch => {
            let output = run_process(cwd, "git", &["branch", "--all", "--verbose"])?;
            if json_mode {
                print_json(&json!({"output": output}))?;
            } else {
                println!("{output}");
            }
        }
        GitCmd::Checkout(args) => {
            let output = run_process(cwd, "git", &["checkout", &args.target])?;
            if json_mode {
                print_json(&json!({"target": args.target, "output": output}))?;
            } else {
                println!("{output}");
            }
        }
        GitCmd::Commit(args) => {
            let mut logs = Vec::new();
            if args.all {
                logs.push(run_process(cwd, "git", &["add", "-A"])?);
            }
            logs.push(run_process(cwd, "git", &["commit", "-m", &args.message])?);
            if json_mode {
                print_json(&json!({"message": args.message, "output": logs}))?;
            } else {
                println!("{}", logs.join("\n"));
            }
        }
        GitCmd::Pr(args) => {
            let gh_available = command_exists("gh");
            if !gh_available || args.dry_run {
                let payload = json!({
                    "available": gh_available,
                    "dry_run": args.dry_run,
                    "suggested_command": format!(
                        "gh pr create{}{}{}{}",
                        args.title.as_deref().map(|title| format!(" --title \"{title}\"")).unwrap_or_default(),
                        args.body.as_deref().map(|body| format!(" --body \"{body}\"")).unwrap_or_default(),
                        args.base.as_deref().map(|base| format!(" --base {base}")).unwrap_or_default(),
                        args.head.as_deref().map(|head| format!(" --head {head}")).unwrap_or_default(),
                    )
                });
                if json_mode {
                    print_json(&payload)?;
                } else {
                    println!("{}", serde_json::to_string_pretty(&payload)?);
                }
            } else {
                let mut cmd = std::process::Command::new("gh");
                cmd.current_dir(cwd).arg("pr").arg("create");
                if let Some(title) = args.title {
                    cmd.arg("--title").arg(title);
                }
                if let Some(body) = args.body {
                    cmd.arg("--body").arg(body);
                }
                if let Some(base) = args.base {
                    cmd.arg("--base").arg(base);
                }
                if let Some(head) = args.head {
                    cmd.arg("--head").arg(head);
                }
                let output = cmd.output()?;
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                if !output.status.success() {
                    return Err(anyhow!(
                        "gh pr create failed: {}\n{}",
                        output.status,
                        stderr
                    ));
                }
                if json_mode {
                    print_json(&json!({"stdout": stdout, "stderr": stderr}))?;
                } else {
                    println!("{stdout}");
                }
            }
        }
        GitCmd::Resolve(args) => {
            let strategy = args.strategy.to_ascii_lowercase();
            if strategy == "list" {
                let output = run_process(cwd, "git", &["diff", "--name-only", "--diff-filter=U"])?;
                let conflicts = parse_conflict_files(&output);
                let suggestions = conflicts
                    .iter()
                    .map(|path| {
                        json!({
                            "file": path,
                            "ours": format!("deepseek git resolve --strategy ours --file {path}"),
                            "theirs": format!("deepseek git resolve --strategy theirs --file {path}"),
                            "ours_all": "deepseek git resolve --strategy ours --all --stage",
                            "theirs_all": "deepseek git resolve --strategy theirs --all --stage"
                        })
                    })
                    .collect::<Vec<_>>();
                if json_mode {
                    print_json(&json!({
                        "conflicts": conflicts,
                        "count": suggestions.len(),
                        "suggestions": suggestions
                    }))?;
                } else if suggestions.is_empty() {
                    println!("no merge conflicts");
                } else {
                    println!("{output}");
                    for item in suggestions {
                        println!(
                            "resolve {}: {} | {}",
                            item["file"].as_str().unwrap_or_default(),
                            item["ours"].as_str().unwrap_or_default(),
                            item["theirs"].as_str().unwrap_or_default()
                        );
                    }
                }
            } else {
                if strategy != "ours" && strategy != "theirs" {
                    return Err(anyhow!("unsupported strategy '{}'", strategy));
                }
                let files =
                    if args.all {
                        let output =
                            run_process(cwd, "git", &["diff", "--name-only", "--diff-filter=U"])?;
                        parse_conflict_files(&output)
                    } else {
                        vec![args.file.ok_or_else(|| {
                            anyhow!("--file is required for strategy '{}'", strategy)
                        })?]
                    };
                if files.is_empty() {
                    return Err(anyhow!("no unresolved conflicts found"));
                }

                let mut outputs = Vec::new();
                for file in &files {
                    let output = run_process(
                        cwd,
                        "git",
                        &["checkout", &format!("--{strategy}"), "--", file],
                    )?;
                    outputs.push(json!({"file": file, "output": output}));
                }

                if args.stage {
                    for file in &files {
                        run_process(cwd, "git", &["add", "--", file])?;
                    }
                }

                let continued = if args.continue_after {
                    Some(run_git_continue(cwd)?)
                } else {
                    None
                };
                if json_mode {
                    print_json(&json!({
                        "strategy": strategy,
                        "resolved_files": files,
                        "count": outputs.len(),
                        "stage": args.stage,
                        "continued": continued,
                        "outputs": outputs
                    }))?;
                } else {
                    println!(
                        "resolved {} conflict file(s) with strategy={} stage={}",
                        outputs.len(),
                        strategy,
                        args.stage
                    );
                    if let Some(continued) = &continued {
                        println!(
                            "continued {}: {}",
                            continued["action"].as_str().unwrap_or_default(),
                            continued["output"].as_str().unwrap_or_default()
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn parse_git_status_summary(porcelain: &str) -> GitStatusSummary {
    let mut summary = GitStatusSummary::default();
    for line in porcelain.lines() {
        if let Some(branch) = line.strip_prefix("# branch.head ") {
            if branch != "(detached)" {
                summary.branch = Some(branch.trim().to_string());
            }
            continue;
        }
        if let Some(ab) = line.strip_prefix("# branch.ab ") {
            let mut parts = ab.split_whitespace();
            if let Some(ahead) = parts.next().and_then(|part| part.strip_prefix('+')) {
                summary.ahead = ahead.parse::<u64>().unwrap_or(0);
            }
            if let Some(behind) = parts.next().and_then(|part| part.strip_prefix('-')) {
                summary.behind = behind.parse::<u64>().unwrap_or(0);
            }
            continue;
        }
        if line.starts_with("? ") {
            summary.untracked += 1;
            continue;
        }
        if line.starts_with("u ") {
            summary.conflicts += 1;
            continue;
        }
        if line.starts_with("1 ") || line.starts_with("2 ") {
            let xy = line.split_whitespace().nth(1).unwrap_or("..");
            let mut chars = xy.chars();
            let x = chars.next().unwrap_or('.');
            let y = chars.next().unwrap_or('.');
            if x != '.' && x != ' ' {
                summary.staged += 1;
            }
            if y != '.' && y != ' ' {
                summary.unstaged += 1;
            }
        }
    }
    summary
}

pub(crate) fn parse_conflict_files(output: &str) -> Vec<String> {
    output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>()
}

pub(crate) fn run_git_continue(cwd: &Path) -> Result<serde_json::Value> {
    let candidates = [
        ("merge", vec!["merge", "--continue"]),
        ("rebase", vec!["rebase", "--continue"]),
        ("cherry-pick", vec!["cherry-pick", "--continue"]),
    ];
    let mut errors = Vec::new();
    for (action, args) in candidates {
        match run_process(cwd, "git", &args) {
            Ok(output) => {
                return Ok(json!({
                    "action": action,
                    "output": output
                }));
            }
            Err(err) => {
                errors.push(format!("{action}: {err}"));
            }
        }
    }
    Err(anyhow!(
        "no continuation command succeeded: {}",
        errors.join(" | ")
    ))
}
