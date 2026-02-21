use anyhow::{Result, anyhow};
use deepseek_agent::{AgentEngine, ChatOptions};
use serde_json::json;
use std::fs;
use std::path::Path;
use std::process::Command;

use crate::ReviewArgs;
use crate::context::*;
use crate::output::*;
use crate::util::*;

pub(crate) fn run_review(cwd: &Path, args: ReviewArgs, json_mode: bool) -> Result<()> {
    ensure_llm_ready(cwd, json_mode)?;

    let diff_content = if let Some(pr_number) = args.pr {
        // Get PR diff via gh CLI
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
        // Review a specific file
        fs::read_to_string(cwd.join(path)).map_err(|e| anyhow!("cannot read {path}: {e}"))?
    } else if args.staged {
        run_capture("git", &["diff", "--staged"]).unwrap_or_default()
    } else {
        // Default: unstaged diff (or --diff flag)
        run_capture("git", &["diff"]).unwrap_or_default()
    };

    if diff_content.trim().is_empty() {
        if json_mode {
            print_json(&json!({"review": "no changes to review"}))?;
        } else {
            println!("no changes to review");
        }
        return Ok(());
    }

    let focus = args
        .focus
        .as_deref()
        .unwrap_or("correctness, security, performance, style");
    let review_prompt = format!(
        "You are a senior code reviewer. Analyze the following diff and provide structured feedback.\n\
         Focus areas: {focus}\n\n\
         For each issue found, provide:\n\
         - **severity**: critical / warning / suggestion\n\
         - **file**: the affected file\n\
         - **line**: approximate line number\n\
         - **issue**: concise description\n\
         - **suggestion**: how to fix it\n\n\
         If the code looks good, say so.\n\n\
         ```diff\n{diff_content}\n```"
    );

    let engine = AgentEngine::new(cwd)?;
    let output = engine.chat_with_options(
        &review_prompt,
        ChatOptions {
            tools: false,
            ..Default::default()
        },
    )?;

    if json_mode {
        print_json(&json!({
            "review": output,
            "diff_lines": diff_content.lines().count(),
            "focus": focus,
        }))?;
    } else {
        println!("{output}");
    }
    Ok(())
}
