use crate::apply::{diff_stats, ensure_repo_relative_path, extract_target_files};
use crate::r#loop;
use crate::verify::{derive_verify_commands, run_verify};
use crate::{AgentEngine, ChatMode, ChatOptions};
use anyhow::{Context, Result, anyhow};
use deepseek_core::{ApplyStrategy, EventKind, StreamChunk, ToolCall, runtime_dir};
use deepseek_diff::{GitApplyStrategy, PatchStore};
use serde_json::json;
use std::fs;
use std::path::Path;
use std::process::Command;
use uuid::Uuid;

#[derive(Debug, Clone)]
struct LaneSpec {
    id: String,
    name: String,
    goal: String,
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

    let mut lanes = plan_lanes(prompt);
    if lanes.len() <= 1 {
        let mut fallback = options.clone();
        fallback.disable_team_orchestration = true;
        return r#loop::run(engine, prompt, &fallback);
    }
    lanes.sort_by(|a, b| a.id.cmp(&b.id));

    let base_commit = git_stdout(&engine.workspace, &["rev-parse", "HEAD"])
        .context("failed to read base commit for team lanes")?;
    let run_group = Uuid::now_v7();
    let lanes_root = runtime_dir(&engine.workspace)
        .join("team_lanes")
        .join(run_group.to_string());
    fs::create_dir_all(&lanes_root)?;

    let mut results = Vec::new();
    for lane in lanes {
        let run_id = Uuid::now_v7();
        emit_spawned(engine, run_id, &lane);

        let lane_root = lanes_root.join(format!("{}-{}", lane.id, sanitize_name(&lane.name)));
        if lane_root.exists() {
            let _ = fs::remove_dir_all(&lane_root);
        }
        if let Err(err) = add_worktree(&engine.workspace, &lane_root, &base_commit) {
            emit_failed(engine, run_id, &lane, &err.to_string());
            return Err(err);
        }

        let mut lane_options = options.clone();
        lane_options.mode = ChatMode::Code;
        lane_options.force_execute = true;
        lane_options.force_plan_only = false;
        lane_options.disable_team_orchestration = true;

        let lane_prompt = format!(
            "{}\n\n[Team lane {}: {}]\nFocus strictly on this lane goal:\n{}",
            prompt, lane.id, lane.name, lane.goal
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

        let _ = remove_worktree(&engine.workspace, &lane_root);
        match lane_result {
            Ok(result) => {
                emit_completed(engine, result.run_id, &result.lane, &result.patch);
                results.push(result);
            }
            Err(err) => {
                emit_failed(engine, run_id, &lane, &err.to_string());
                return Err(err);
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

fn plan_lanes(prompt: &str) -> Vec<LaneSpec> {
    let lower = prompt.to_ascii_lowercase();
    let mut lanes = Vec::new();
    maybe_add_lane(
        &mut lanes,
        &lower,
        "01",
        "frontend",
        "UI, UX, and client-side behavior",
        &["frontend", "ui", "react", "vue", "css", "html", "web"],
    );
    maybe_add_lane(
        &mut lanes,
        &lower,
        "02",
        "backend",
        "API, server logic, and data flows",
        &[
            "backend", "api", "server", "database", "db", "sql", "endpoint",
        ],
    );
    maybe_add_lane(
        &mut lanes,
        &lower,
        "03",
        "testing",
        "Tests, linting, and verification hardening",
        &["test", "testing", "qa", "verify", "lint", "ci"],
    );
    maybe_add_lane(
        &mut lanes,
        &lower,
        "04",
        "docs",
        "Documentation and developer guidance",
        &["docs", "readme", "spec", "documentation"],
    );
    if lanes.is_empty() {
        lanes.push(LaneSpec {
            id: "01".to_string(),
            name: "core".to_string(),
            goal: "Primary implementation lane".to_string(),
        });
    }
    lanes
}

fn maybe_add_lane(
    lanes: &mut Vec<LaneSpec>,
    prompt_lower: &str,
    id: &str,
    name: &str,
    goal: &str,
    keywords: &[&str],
) {
    if keywords
        .iter()
        .any(|keyword| prompt_lower.contains(keyword))
    {
        lanes.push(LaneSpec {
            id: id.to_string(),
            name: name.to_string(),
            goal: goal.to_string(),
        });
    }
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
        let a = plan_lanes("Refactor frontend and backend and add tests");
        let b = plan_lanes("Refactor frontend and backend and add tests");
        assert_eq!(a.len(), b.len());
        assert_eq!(
            a.iter().map(|lane| lane.id.clone()).collect::<Vec<_>>(),
            b.iter().map(|lane| lane.id.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn lane_planning_falls_back_to_core() {
        let lanes = plan_lanes("minor tweak");
        assert_eq!(lanes.len(), 1);
        assert_eq!(lanes[0].name, "core");
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
