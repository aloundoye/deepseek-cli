use crate::repo_map_v2;
use anyhow::{Result, anyhow};
use deepseek_core::{AppConfig, ChatMessage, ChatRequest, ToolChoice};
use deepseek_llm::LlmClient;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct ArchitectFileIntent {
    pub path: String,
    pub intent: String,
}

#[derive(Debug, Clone)]
pub struct ArchitectPlan {
    pub steps: Vec<String>,
    pub files: Vec<ArchitectFileIntent>,
    pub verify_commands: Vec<String>,
    pub acceptance: Vec<String>,
    pub no_edit_reason: Option<String>,
    pub raw: String,
}

#[derive(Debug, Clone, Default)]
pub struct ArchitectFeedback {
    pub verify_feedback: Option<String>,
    pub apply_feedback: Option<String>,
    pub last_diff_summary: Option<String>,
}

pub struct ArchitectInput<'a> {
    pub user_prompt: &'a str,
    pub iteration: u64,
    pub feedback: &'a ArchitectFeedback,
    pub max_files: usize,
    pub additional_dirs: &'a [PathBuf],
    pub debug_context: bool,
}

const ARCHITECT_SYSTEM_PROMPT: &str = r#"You are Architect (reasoning-only).

Return ONLY this line-oriented contract and nothing else:

ARCHITECT_PLAN_V1
PLAN|<step text>
FILE|<path>|<intent>
VERIFY|<command>
ACCEPT|<criterion>
NO_EDIT|true|<reason>   # optional
ARCHITECT_PLAN_END

Rules:
- Never emit unified diff, JSON, XML, markdown fences, or tool calls.
- FILE paths must be workspace-relative.
- Be concrete and deterministic.
"#;

pub fn run_architect(
    llm: &(dyn LlmClient + Send + Sync),
    cfg: &AppConfig,
    workspace: &Path,
    input: &ArchitectInput<'_>,
    retries: usize,
) -> Result<ArchitectPlan> {
    let repo_map = build_repo_map(
        workspace,
        input.user_prompt,
        input.max_files,
        input.additional_dirs,
    );
    if input.debug_context {
        let lines = repo_map.lines().count();
        let est_tokens = repo_map.len() / 4;
        eprintln!(
            "[debug-context] intent=EditCode phase=Architect mode=Code repo_root={} repo_map_lines={} repo_map_est_tokens={} iteration={}",
            workspace.display(),
            lines,
            est_tokens,
            input.iteration
        );
    }

    let mut messages = vec![
        ChatMessage::System {
            content: ARCHITECT_SYSTEM_PROMPT.to_string(),
        },
        ChatMessage::User {
            content: build_architect_user_prompt(input, &repo_map),
        },
    ];

    for attempt in 0..=retries {
        let req = ChatRequest {
            model: cfg.llm.max_think_model.clone(),
            messages: messages.clone(),
            tools: vec![],
            tool_choice: ToolChoice::none(),
            max_tokens: 4096,
            temperature: None,
            thinking: Some(deepseek_core::ThinkingConfig::enabled(16_384)),
        };

        let response = llm.complete_chat(&req)?;
        match parse_architect_plan(&response.text) {
            Ok(mut plan) => {
                plan.raw = response.text;
                return Ok(plan);
            }
            Err(err) => {
                if attempt == retries {
                    return Err(anyhow!(
                        "{err}; raw_response={}",
                        response.text.replace('\n', "\\n")
                    ));
                }
                messages.push(ChatMessage::Assistant {
                    content: Some(response.text),
                    reasoning_content: None,
                    tool_calls: vec![],
                });
                messages.push(ChatMessage::User {
                    content: repair_prompt(),
                });
            }
        }
    }

    Err(anyhow!("architect failed to produce a valid plan"))
}

fn build_architect_user_prompt(input: &ArchitectInput<'_>, repo_map: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Iteration: {}\n\nUser request:\n{}\n\n",
        input.iteration, input.user_prompt
    ));
    if !repo_map.is_empty() {
        out.push_str("Repository map:\n");
        out.push_str(repo_map);
        out.push_str("\n\n");
    }
    if let Some(ref apply) = input.feedback.apply_feedback {
        out.push_str("Last apply failure:\n");
        out.push_str(apply);
        out.push_str("\n\n");
    }
    if let Some(ref verify) = input.feedback.verify_feedback {
        out.push_str("Last verification failure:\n");
        out.push_str(verify);
        out.push_str("\n\n");
    }
    if let Some(ref diff) = input.feedback.last_diff_summary {
        out.push_str("Last diff summary:\n");
        out.push_str(diff);
        out.push_str("\n\n");
    }
    out.push_str("Return ARCHITECT_PLAN_V1 now.");
    out
}

fn repair_prompt() -> String {
    "Your previous response violated the strict format. Return ONLY ARCHITECT_PLAN_V1 lines."
        .to_string()
}

pub fn parse_architect_plan(text: &str) -> Result<ArchitectPlan> {
    let mut lines = text.lines().map(str::trim).filter(|line| !line.is_empty());
    let start = lines.next().unwrap_or_default();
    if start != "ARCHITECT_PLAN_V1" {
        return Err(anyhow!("architect output missing ARCHITECT_PLAN_V1"));
    }

    let mut steps = Vec::new();
    let mut files = Vec::new();
    let mut verify_commands = Vec::new();
    let mut acceptance = Vec::new();
    let mut no_edit_reason = None;
    let mut saw_end = false;

    for line in lines {
        if line == "ARCHITECT_PLAN_END" {
            saw_end = true;
            break;
        }
        if let Some(value) = line.strip_prefix("PLAN|") {
            if !value.trim().is_empty() {
                steps.push(value.trim().to_string());
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("FILE|") {
            let mut parts = rest.splitn(2, '|');
            let path = parts.next().unwrap_or_default().trim();
            let intent = parts.next().unwrap_or_default().trim();
            if path.is_empty() || intent.is_empty() {
                return Err(anyhow!("invalid FILE line: {line}"));
            }
            if Path::new(path).is_absolute() {
                return Err(anyhow!("architect declared absolute path: {path}"));
            }
            files.push(ArchitectFileIntent {
                path: path.to_string(),
                intent: intent.to_string(),
            });
            continue;
        }
        if let Some(value) = line.strip_prefix("VERIFY|") {
            if !value.trim().is_empty() {
                verify_commands.push(value.trim().to_string());
            }
            continue;
        }
        if let Some(value) = line.strip_prefix("ACCEPT|") {
            if !value.trim().is_empty() {
                acceptance.push(value.trim().to_string());
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("NO_EDIT|") {
            let mut parts = rest.splitn(2, '|');
            let flag = parts.next().unwrap_or_default().trim();
            let reason = parts.next().unwrap_or_default().trim();
            if flag.eq_ignore_ascii_case("true") {
                let message = if reason.is_empty() {
                    "No file edits required".to_string()
                } else {
                    reason.to_string()
                };
                no_edit_reason = Some(message);
            }
            continue;
        }
        return Err(anyhow!("unknown architect line: {line}"));
    }

    if !saw_end {
        return Err(anyhow!("architect output missing ARCHITECT_PLAN_END"));
    }

    let mut seen = HashSet::new();
    files.retain(|entry| seen.insert(entry.path.clone()));

    Ok(ArchitectPlan {
        steps,
        files,
        verify_commands,
        acceptance,
        no_edit_reason,
        raw: String::new(),
    })
}

pub fn build_repo_map(
    workspace: &Path,
    user_prompt: &str,
    max_files: usize,
    additional_dirs: &[PathBuf],
) -> String {
    let v2_rows =
        repo_map_v2::build_repo_map_v2(workspace, user_prompt, max_files, additional_dirs);
    if !v2_rows.is_empty() {
        return repo_map_v2::render_repo_map(&v2_rows);
    }

    let mut files = tracked_files(workspace);
    for dir in additional_dirs {
        let joined = if dir.is_absolute() {
            dir.clone()
        } else {
            workspace.join(dir)
        };
        files.extend(list_files(&joined, Some(workspace)));
    }

    files.sort();
    files.dedup();

    let changed = changed_files(workspace);
    let tokens = prompt_tokens(user_prompt);

    let mut scored: Vec<(i32, String)> = files
        .into_iter()
        .map(|path| {
            let mut score = 0;
            if changed.contains(&path) {
                score += 100;
            }
            let lower = path.to_ascii_lowercase();
            for token in &tokens {
                if lower.contains(token) {
                    score += 10;
                }
            }
            if lower.ends_with("readme.md") || lower.ends_with("specs.md") {
                score += 5;
            }
            (score, path)
        })
        .collect();

    scored.sort_by(|a, b| b.cmp(a));

    let mut lines = Vec::new();
    let mut count = 0usize;
    for (score, path) in scored {
        if count >= max_files.max(1) {
            break;
        }
        if score <= 0 && count >= max_files / 2 {
            break;
        }
        let size = workspace
            .join(&path)
            .metadata()
            .ok()
            .map(|m| m.len())
            .unwrap_or(0);
        lines.push(format!("- {path} ({size} bytes) score={score}"));
        count += 1;
    }

    lines.join("\n")
}

fn tracked_files(workspace: &Path) -> Vec<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(workspace)
        .arg("ls-files")
        .output();
    if let Ok(out) = output
        && out.status.success()
    {
        return String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToString::to_string)
            .collect();
    }
    list_files(workspace, None)
}

fn list_files(root: &Path, workspace: Option<&Path>) -> Vec<String> {
    let mut out = Vec::new();
    let walker = ignore::WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .git_exclude(true)
        .build();
    for entry in walker.flatten() {
        if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.into_path();
        let rel = if let Some(base) = workspace {
            path.strip_prefix(base).ok().map(|p| p.to_path_buf())
        } else {
            path.strip_prefix(root).ok().map(|p| p.to_path_buf())
        };
        if let Some(rel) = rel {
            out.push(rel.to_string_lossy().to_string());
        }
    }
    out
}

fn changed_files(workspace: &Path) -> HashSet<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(workspace)
        .args(["status", "--porcelain"])
        .output();
    let mut set = HashSet::new();
    if let Ok(out) = output
        && out.status.success()
    {
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            let trimmed = line.trim();
            if trimmed.len() >= 3 {
                set.insert(trimmed[3..].trim().to_string());
            }
        }
    }
    set
}

fn prompt_tokens(prompt: &str) -> HashSet<String> {
    prompt
        .split(|c: char| !c.is_ascii_alphanumeric())
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| s.len() >= 3)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_architect_ok() {
        let input = "ARCHITECT_PLAN_V1\nPLAN|Do thing\nFILE|src/lib.rs|Update fn\nVERIFY|cargo test -q\nACCEPT|Tests pass\nARCHITECT_PLAN_END\n";
        let plan = parse_architect_plan(input).expect("parse");
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.files.len(), 1);
        assert_eq!(plan.verify_commands.len(), 1);
        assert_eq!(plan.acceptance.len(), 1);
    }

    #[test]
    fn parse_architect_requires_markers() {
        let err = parse_architect_plan("PLAN|x").unwrap_err();
        assert!(err.to_string().contains("ARCHITECT_PLAN_V1"));
    }
}
