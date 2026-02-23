use crate::apply::{diff_stats, ensure_repo_relative_path, extract_target_files};
use crate::r#loop;
use crate::verify::{derive_verify_commands, run_verify};
use crate::{AgentEngine, ChatMode, ChatOptions};
use anyhow::{Context, Result, anyhow};
use deepseek_core::{ApplyStrategy, EventKind, StreamChunk, ToolCall, runtime_dir};
use deepseek_diff::{GitApplyStrategy, PatchStore};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::thread;
use uuid::Uuid;

#[derive(Debug, Clone)]
struct LaneSpec {
    id: String,
    name: String,
    goal: String,
    files: Vec<String>,
}

#[derive(Debug, Clone)]
struct LaneResult {
    lane: LaneSpec,
    run_id: Uuid,
    patch: String,
}

pub fn run(engine: &AgentEngine, prompt: &str, options: &ChatOptions) -> Result<String> {
    if !is_git_workspace(&engine.workspace) {
        let mut fallback = options.clone();
        fallback.disable_team_orchestration = true;
        return r#loop::run(engine, prompt, &fallback);
    }

    let max_lanes = engine.cfg.agent_loop.team.max_lanes.max(1) as usize;
    let max_concurrency = engine.cfg.agent_loop.team.max_concurrency.max(1) as usize;
    let mut lanes = plan_lanes(engine, prompt, max_lanes);
    if lanes.len() <= 1 {
        let mut fallback = options.clone();
        fallback.disable_team_orchestration = true;
        return r#loop::run(engine, prompt, &fallback);
    }
    lanes.truncate(max_lanes);
    lanes.sort_by(|a, b| a.id.cmp(&b.id));

    let base_commit = git_stdout(&engine.workspace, &["rev-parse", "HEAD"])
        .context("failed to read base commit for team lanes")?;
    let run_group = Uuid::now_v7();
    let lanes_root = runtime_dir(&engine.workspace)
        .join("team_lanes")
        .join(run_group.to_string());
    fs::create_dir_all(&lanes_root)?;

    let mut results = Vec::new();
    for chunk in lanes.chunks(max_concurrency) {
        let mut handles = Vec::new();
        for lane in chunk.iter().cloned() {
            let run_id = Uuid::now_v7();
            emit_spawned(engine, run_id, &lane);

            let workspace = engine.workspace.clone();
            let base_commit = base_commit.clone();
            let lanes_root = lanes_root.clone();
            let prompt = prompt.to_string();
            let mut lane_options = options.clone();
            lane_options.mode = ChatMode::Code;
            lane_options.force_execute = true;
            lane_options.force_plan_only = false;
            lane_options.disable_team_orchestration = true;

            handles.push(thread::spawn(move || {
                execute_lane(
                    workspace,
                    lanes_root,
                    &base_commit,
                    &prompt,
                    lane_options,
                    lane,
                    run_id,
                )
            }));
        }

        for handle in handles {
            let lane_result = handle
                .join()
                .map_err(|_| anyhow!("team lane worker panicked"))?;
            match lane_result {
                Ok(result) => {
                    emit_completed(engine, result.run_id, &result.lane, &result.patch);
                    results.push(result);
                }
                Err((lane, run_id, err)) => {
                    emit_failed(engine, run_id, &lane, &err.to_string());
                    return Err(err);
                }
            }
        }
    }

    results.sort_by(|a, b| a.lane.id.cmp(&b.lane.id));
    for lane in &results {
        if lane.patch.trim().is_empty() {
            continue;
        }
        validate_lane_patch_artifact(&lane.patch)
            .with_context(|| format!("invalid lane patch artifact for {}", lane.lane.id))?;
        if !require_patch_approval(engine, &lane.patch, &lane.lane)? {
            return Err(anyhow!("patch approval denied for lane {}", lane.lane.id));
        }
        let (applied, conflicts) = apply_lane_patch(
            &engine.workspace,
            &lane.patch,
            engine.cfg.agent_loop.apply_strategy.clone(),
        )?;
        if !applied {
            let conflict_text = if conflicts.is_empty() {
                "patch apply failed".to_string()
            } else {
                conflicts.join("\n")
            };
            let mut recovery_options = options.clone();
            recovery_options.disable_team_orchestration = true;
            recovery_options.force_execute = true;
            let recovery_prompt = format!(
                "Lane merge conflict while applying {} ({})\n\nConflict details:\n{}\n\nOriginal request:\n{}\n\nResolve with a single unified patch and re-verify.",
                lane.lane.id, lane.lane.name, conflict_text, prompt
            );
            return r#loop::run(engine, &recovery_prompt, &recovery_options);
        }
    }

    let verify_commands = derive_verify_commands(&engine.workspace);
    let verify = run_verify(
        engine.tool_host.as_ref(),
        &verify_commands,
        engine.cfg.agent_loop.verify_timeout_seconds,
        None,
    );
    if !verify.success {
        let mut recovery_options = options.clone();
        recovery_options.disable_team_orchestration = true;
        recovery_options.force_execute = true;
        let recovery_prompt = format!(
            "Global verify failed after deterministic lane merge.\n\nVerify summary:\n{}\n\nOriginal request:\n{}\n\nProduce a corrective unified diff and pass verification.",
            verify.summary, prompt
        );
        return r#loop::run(engine, &recovery_prompt, &recovery_options);
    }

    let non_empty = results
        .iter()
        .filter(|lane| !lane.patch.trim().is_empty())
        .count();
    let mut response = format!(
        "Applied {} team lane patch artifact(s) in deterministic order and passed global verification.",
        non_empty
    );
    if !verify_commands.is_empty() {
        response.push_str("\n\nVerification:\n");
        for command in verify_commands {
            response.push_str(&format!("- `{}`\n", command));
        }
        response = response.trim_end().to_string();
    }
    engine.stream(StreamChunk::ContentDelta(response.clone()));
    engine.stream(StreamChunk::Done);
    Ok(response)
}

fn execute_lane(
    workspace: std::path::PathBuf,
    lanes_root: std::path::PathBuf,
    base_commit: &str,
    prompt: &str,
    lane_options: ChatOptions,
    lane: LaneSpec,
    run_id: Uuid,
) -> std::result::Result<LaneResult, (LaneSpec, Uuid, anyhow::Error)> {
    let lane_root = lanes_root.join(format!("{}-{}", lane.id, sanitize_name(&lane.name)));
    if lane_root.exists() {
        let _ = fs::remove_dir_all(&lane_root);
    }
    if let Err(err) = add_worktree(&workspace, &lane_root, base_commit) {
        return Err((lane, run_id, err));
    }

    let lane_prompt = format!(
        "{}\n\n[Team lane {}: {}]\nFocus strictly on this lane goal:\n{}\n\nTarget files:\n{}",
        prompt,
        lane.id,
        lane.name,
        lane.goal,
        if lane.files.is_empty() {
            "- (none listed)".to_string()
        } else {
            lane.files
                .iter()
                .take(12)
                .map(|file| format!("- {file}"))
                .collect::<Vec<_>>()
                .join("\n")
        }
    );

    let lane_result = (|| -> Result<LaneResult> {
        let lane_engine = AgentEngine::new(&lane_root)?;
        lane_engine.set_approval_handler(Box::new(|_call| Ok(true)));
        let _ = lane_engine.chat_with_options(&lane_prompt, lane_options)?;
        let patch = git_stdout(&lane_root, &["diff", "--binary"])
            .context("failed to collect lane patch artifact")?;
        Ok(LaneResult {
            lane: lane.clone(),
            run_id,
            patch,
        })
    })();

    let _ = remove_worktree(&workspace, &lane_root);
    match lane_result {
        Ok(result) => Ok(result),
        Err(err) => Err((lane, run_id, err)),
    }
}

fn require_patch_approval(engine: &AgentEngine, diff: &str, lane: &LaneSpec) -> Result<bool> {
    let stats = diff_stats(diff);
    let gate = &engine.cfg.agent_loop.safety_gate;
    let files_over = stats.touched_files as u64 > gate.max_files_without_approval;
    let loc_over = stats.loc_delta as u64 > gate.max_loc_without_approval;
    if !files_over && !loc_over {
        return Ok(true);
    }

    let call = ToolCall {
        name: "patch.apply".to_string(),
        args: json!({
            "lane_id": lane.id,
            "lane_name": lane.name,
            "touched_files": stats.touched_files,
            "loc_delta": stats.loc_delta,
            "max_files_without_approval": gate.max_files_without_approval,
            "max_loc_without_approval": gate.max_loc_without_approval,
        }),
        requires_approval: true,
    };
    engine.request_approval(&call)
}

fn apply_lane_patch(
    workspace: &Path,
    patch: &str,
    strategy: ApplyStrategy,
) -> Result<(bool, Vec<String>)> {
    let store = PatchStore::new(workspace)?;
    let staged = store.stage(patch, &[])?;
    let strategy = match strategy {
        ApplyStrategy::Auto => GitApplyStrategy::Auto,
        ApplyStrategy::ThreeWay => GitApplyStrategy::ThreeWay,
    };
    store.apply_with_strategy(workspace, staged.patch_id, strategy)
}

fn emit_spawned(engine: &AgentEngine, run_id: Uuid, lane: &LaneSpec) {
    let lane_name = format!("{}:{}", lane.id, lane.name);
    let lane_goal = format!("[lane={}] {}", lane.id, lane.goal);
    engine.stream(StreamChunk::SubagentSpawned {
        run_id: run_id.to_string(),
        name: lane_name.clone(),
        goal: lane_goal.clone(),
    });
    engine.append_event_best_effort(EventKind::SubagentSpawnedV1 {
        run_id,
        name: lane_name,
        goal: lane_goal,
    });
}

fn emit_completed(engine: &AgentEngine, run_id: Uuid, lane: &LaneSpec, patch: &str) {
    let lane_name = format!("{}:{}", lane.id, lane.name);
    let summary = if patch.trim().is_empty() {
        format!("[lane={}] no patch generated", lane.id)
    } else {
        let stats = diff_stats(patch);
        format!(
            "[lane={}] patch files={} loc={}",
            lane.id, stats.touched_files, stats.loc_delta
        )
    };
    engine.stream(StreamChunk::SubagentCompleted {
        run_id: run_id.to_string(),
        name: lane_name.clone(),
        summary: summary.clone(),
    });
    engine.append_event_best_effort(EventKind::SubagentCompletedV1 {
        run_id,
        output: summary,
    });
}

fn emit_failed(engine: &AgentEngine, run_id: Uuid, lane: &LaneSpec, error: &str) {
    let lane_name = format!("{}:{}", lane.id, lane.name);
    let lane_error = format!("[lane={}] {error}", lane.id);
    engine.stream(StreamChunk::SubagentFailed {
        run_id: run_id.to_string(),
        name: lane_name.clone(),
        error: lane_error.clone(),
    });
    engine.append_event_best_effort(EventKind::SubagentFailedV1 {
        run_id,
        error: lane_error,
    });
}

fn validate_lane_patch_artifact(patch: &str) -> Result<()> {
    if patch.trim().is_empty() {
        return Ok(());
    }
    let files = extract_target_files(patch);
    if files.is_empty() {
        return Err(anyhow!("lane patch artifact contains no target files"));
    }
    for file in files {
        ensure_repo_relative_path(&file).map_err(|err| anyhow!(err.reason))?;
        if file == ".git" || file.starts_with(".git/") {
            return Err(anyhow!(".git mutation forbidden in lane patch: {file}"));
        }
    }
    Ok(())
}

fn plan_lanes(engine: &AgentEngine, prompt: &str, max_lanes: usize) -> Vec<LaneSpec> {
    let mut grouped: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    if let Ok(plan) = engine.plan_only(prompt) {
        for step in &plan.steps {
            for path in &step.files {
                let lane = classify_lane_for_path(path);
                grouped.entry(lane).or_default().insert(path.clone());
            }
        }
    }

    if grouped.is_empty() {
        grouped = keyword_lane_groups(prompt);
    }

    if grouped.is_empty() {
        grouped
            .entry("core".to_string())
            .or_default()
            .insert("core".to_string());
    }

    let lane_order = ["frontend", "backend", "testing", "docs", "infra", "core"];
    let mut lanes = Vec::new();
    for (idx, lane_name) in lane_order.iter().enumerate() {
        if let Some(paths) = grouped.get(*lane_name) {
            let files = paths.iter().cloned().collect::<Vec<_>>();
            lanes.push(LaneSpec {
                id: format!("{:02}", idx + 1),
                name: lane_name.to_string(),
                goal: lane_goal(lane_name, &files),
                files,
            });
        }
    }

    for (name, paths) in grouped {
        if lanes.iter().any(|lane| lane.name == name) {
            continue;
        }
        let next = lanes.len() + 1;
        lanes.push(LaneSpec {
            id: format!("{next:02}"),
            name: name.clone(),
            goal: lane_goal(&name, &paths.iter().cloned().collect::<Vec<_>>()),
            files: paths.iter().cloned().collect(),
        });
    }

    lanes.sort_by(|a, b| a.id.cmp(&b.id));
    lanes.truncate(max_lanes.max(1));
    lanes
}

fn lane_goal(name: &str, files: &[String]) -> String {
    let scope = files.iter().take(6).cloned().collect::<Vec<_>>().join(", ");
    let prefix = match name {
        "frontend" => "UI and UX implementation updates",
        "backend" => "API, server, and data-flow updates",
        "testing" => "Verification, tests, and lint hardening",
        "docs" => "Documentation and operator guidance updates",
        "infra" => "Build, CI, and environment orchestration updates",
        _ => "Core implementation updates",
    };
    if scope.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}; focus files: {scope}")
    }
}

fn classify_lane_for_path(path: &str) -> String {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".md") || lower.contains("/docs/") || lower.contains("readme") {
        return "docs".to_string();
    }
    if lower.contains("test")
        || lower.ends_with("_test.rs")
        || lower.ends_with(".spec.ts")
        || lower.ends_with(".test.ts")
    {
        return "testing".to_string();
    }
    if lower.ends_with(".tsx")
        || lower.ends_with(".jsx")
        || lower.ends_with(".css")
        || lower.ends_with(".html")
        || lower.contains("/ui/")
        || lower.contains("/frontend/")
    {
        return "frontend".to_string();
    }
    if lower.contains("/.github/")
        || lower.contains("docker")
        || lower.ends_with(".yml")
        || lower.ends_with(".yaml")
    {
        return "infra".to_string();
    }
    "backend".to_string()
}

fn contains_any(input: &str, keywords: &[&str]) -> bool {
    keywords.iter().any(|keyword| input.contains(keyword))
}

fn keyword_lane_groups(prompt: &str) -> BTreeMap<String, BTreeSet<String>> {
    let lower = prompt.to_ascii_lowercase();
    let mut grouped: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    if contains_any(
        &lower,
        &["frontend", "ui", "react", "vue", "css", "html", "web"],
    ) {
        grouped
            .entry("frontend".to_string())
            .or_default()
            .insert("ui".to_string());
    }
    if contains_any(
        &lower,
        &[
            "backend", "api", "server", "database", "db", "sql", "endpoint",
        ],
    ) {
        grouped
            .entry("backend".to_string())
            .or_default()
            .insert("backend".to_string());
    }
    if contains_any(&lower, &["test", "testing", "qa", "verify", "lint", "ci"]) {
        grouped
            .entry("testing".to_string())
            .or_default()
            .insert("tests".to_string());
    }
    if contains_any(&lower, &["docs", "readme", "spec", "documentation"]) {
        grouped
            .entry("docs".to_string())
            .or_default()
            .insert("docs".to_string());
    }
    grouped
}

fn is_git_workspace(workspace: &Path) -> bool {
    Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .current_dir(workspace)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn add_worktree(workspace: &Path, lane_path: &Path, base_commit: &str) -> Result<()> {
    let lane_path_str = lane_path.to_string_lossy().to_string();
    let output = Command::new("git")
        .arg("-C")
        .arg(workspace)
        .args(["worktree", "add", "--detach"])
        .arg(&lane_path_str)
        .arg(base_commit)
        .output()
        .context("failed to spawn git worktree add")?;
    if output.status.success() {
        return Ok(());
    }
    Err(anyhow!(
        "git worktree add failed: {}",
        String::from_utf8_lossy(&output.stderr)
    ))
}

fn remove_worktree(workspace: &Path, lane_path: &Path) -> Result<()> {
    let lane_path_str = lane_path.to_string_lossy().to_string();
    let output = Command::new("git")
        .arg("-C")
        .arg(workspace)
        .args(["worktree", "remove", "--force"])
        .arg(&lane_path_str)
        .output()
        .context("failed to spawn git worktree remove")?;
    if output.status.success() {
        return Ok(());
    }
    Err(anyhow!(
        "git worktree remove failed: {}",
        String::from_utf8_lossy(&output.stderr)
    ))
}

fn git_stdout(workspace: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(workspace)
        .output()
        .context("failed to spawn git command")?;
    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).trim().to_string());
    }
    Err(anyhow!(
        "git {:?} failed: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    ))
}

fn sanitize_name(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::process::Command;

    fn git(workspace: &Path, args: &[&str]) -> Result<()> {
        let output = Command::new("git")
            .args(args)
            .current_dir(workspace)
            .output()?;
        if output.status.success() {
            return Ok(());
        }
        Err(anyhow!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        ))
    }

    fn init_git(workspace: &Path) -> Result<()> {
        git(workspace, &["init"])?;
        git(
            workspace,
            &["config", "user.email", "deepseek@example.test"],
        )?;
        git(workspace, &["config", "user.name", "DeepSeek Test"])?;
        Ok(())
    }

    fn commit_all(workspace: &Path, message: &str) -> Result<()> {
        git(workspace, &["add", "."])?;
        git(workspace, &["commit", "-m", message])?;
        Ok(())
    }

    #[test]
    fn lane_planning_is_deterministic() {
        let a = keyword_lane_groups("Refactor frontend and backend and add tests");
        let b = keyword_lane_groups("Refactor frontend and backend and add tests");
        assert_eq!(a, b);
    }

    #[test]
    fn lane_planning_falls_back_to_core() {
        let grouped = keyword_lane_groups("minor tweak");
        assert!(grouped.is_empty());
    }

    #[test]
    fn lane_patch_validation_rejects_git_mutation() {
        let patch = "--- a/.git/config\n+++ b/.git/config\n@@ -1 +1 @@\n-a\n+b\n";
        let err = validate_lane_patch_artifact(patch).expect_err("must reject .git patch");
        assert!(err.to_string().contains(".git mutation forbidden"));
    }

    #[test]
    fn lane_patch_validation_accepts_repo_relative_targets() {
        let patch = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-old\n+new\n";
        validate_lane_patch_artifact(patch).expect("repo-relative patch should pass");
    }

    #[test]
    fn apply_lane_patch_reports_conflict_on_context_mismatch() {
        let temp = tempfile::tempdir().expect("tempdir");
        init_git(temp.path()).expect("init git");
        fs::write(temp.path().join("demo.txt"), "old\n").expect("seed file");
        commit_all(temp.path(), "seed").expect("seed commit");

        let patch = "diff --git a/demo.txt b/demo.txt\n--- a/demo.txt\n+++ b/demo.txt\n@@ -1 +1 @@\n-missing\n+new\n";
        let (applied, _conflicts) =
            apply_lane_patch(temp.path(), patch, ApplyStrategy::Auto).expect("apply lane patch");
        assert!(!applied, "mismatched context should not apply");
    }
}
