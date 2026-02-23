use crate::apply::{apply_unified_diff, diff_stats, ensure_repo_relative_path, hash_text};
use crate::architect::{ArchitectFeedback, ArchitectInput, ArchitectPlan, run_architect};
use crate::editor::{EditorFileContext, EditorInput, EditorResponse, FileRequest, run_editor};
use crate::verify::{derive_verify_commands, run_verify};
use crate::{AgentEngine, ChatOptions};
use anyhow::{Result, anyhow};
use deepseek_core::{FailureClassifierConfig, StreamChunk, ToolCall};
use deepseek_memory::MemoryManager;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailureClass {
    PatchMismatch,
    MechanicalVerifyFailure,
    RepeatedVerifyFailure,
    DesignMismatch,
}

impl FailureClass {
    fn as_str(self) -> &'static str {
        match self {
            Self::PatchMismatch => "PatchMismatch",
            Self::MechanicalVerifyFailure => "MechanicalVerifyFailure",
            Self::RepeatedVerifyFailure => "RepeatedVerifyFailure",
            Self::DesignMismatch => "DesignMismatch",
        }
    }
}

#[derive(Default)]
struct FailureTracker {
    last_fingerprint: Option<String>,
    repeat_count: u64,
    last_error_set: Option<HashSet<String>>,
}

pub fn run(engine: &AgentEngine, prompt: &str, options: &ChatOptions) -> Result<String> {
    let mut feedback = ArchitectFeedback::default();
    let cfg = &engine.cfg.agent_loop;
    let mut failure_tracker = FailureTracker::default();

    for iteration in 1..=cfg.max_iterations.max(1) {
        engine.stream(StreamChunk::ArchitectStarted { iteration });

        let architect_input = ArchitectInput {
            user_prompt: prompt,
            iteration,
            feedback: &feedback,
            max_files: cfg.max_files_per_iteration as usize,
            additional_dirs: &options.additional_dirs,
        };
        let plan = run_architect(
            engine.llm.as_ref(),
            &engine.cfg,
            &engine.workspace,
            &architect_input,
            cfg.architect_parse_retries as usize,
        )?;

        engine.stream(StreamChunk::ArchitectCompleted {
            iteration,
            files: plan.files.len() as u32,
            no_edit: plan.no_edit_reason.is_some(),
        });

        if let Some(reason) = plan.no_edit_reason.clone() {
            let response = format_no_edit_response(&plan, &reason);
            engine.stream(StreamChunk::ContentDelta(response.clone()));
            engine.stream(StreamChunk::Done);
            return Ok(response);
        }

        if plan.files.is_empty() {
            let response = format_plan_only_response(&plan);
            engine.stream(StreamChunk::ContentDelta(response.clone()));
            engine.stream(StreamChunk::Done);
            return Ok(response);
        }

        let mut file_context = load_architect_files(
            &engine.workspace,
            &plan,
            cfg.max_files_per_iteration as usize,
            cfg.max_file_bytes as usize,
        )?;
        let mut context_requests = 0u64;
        let mut editor_retry_used = false;

        'editor_cycle: loop {
            engine.stream(StreamChunk::EditorStarted {
                iteration,
                files: file_context.len() as u32,
            });

            let editor_input = EditorInput {
                user_prompt: prompt,
                iteration,
                plan: &plan,
                files: &file_context,
                verify_feedback: feedback.verify_feedback.as_deref(),
                apply_feedback: feedback.apply_feedback.as_deref(),
                max_diff_bytes: cfg.max_diff_bytes as usize,
            };

            let response = run_editor(
                engine.llm.as_ref(),
                &engine.cfg,
                &editor_input,
                cfg.editor_parse_retries as usize,
            )?;

            let diff = match response {
                EditorResponse::Diff(diff) => {
                    engine.stream(StreamChunk::EditorCompleted {
                        iteration,
                        status: "diff".to_string(),
                    });
                    diff
                }
                EditorResponse::NeedContext(requests) => {
                    engine.stream(StreamChunk::EditorCompleted {
                        iteration,
                        status: "need_context".to_string(),
                    });

                    context_requests = context_requests.saturating_add(1);
                    if context_requests > cfg.max_context_requests_per_iteration {
                        feedback.apply_feedback = Some(format!(
                            "classification={}\neditor exceeded max context requests for iteration ({})",
                            FailureClass::PatchMismatch.as_str(),
                            cfg.max_context_requests_per_iteration
                        ));
                        feedback.verify_feedback = None;
                        break 'editor_cycle;
                    }

                    if let Err(err) = merge_requested_files(
                        &engine.workspace,
                        &plan,
                        &mut file_context,
                        &requests,
                        cfg.max_file_bytes as usize,
                        cfg.max_context_range_lines as usize,
                    ) {
                        feedback.apply_feedback = Some(format!(
                            "classification={}\n{}",
                            FailureClass::PatchMismatch.as_str(),
                            err
                        ));
                        feedback.verify_feedback = None;
                        break 'editor_cycle;
                    }
                    continue 'editor_cycle;
                }
            };

            let patch_approved = require_patch_approval(engine, &diff)?;
            if !patch_approved {
                feedback.apply_feedback = Some(format!(
                    "classification={}\npatch exceeds safety gate and approval was denied",
                    FailureClass::PatchMismatch.as_str()
                ));
                feedback.verify_feedback = None;
                break 'editor_cycle;
            }

            checkpoint_best_effort(&engine.workspace, "agent_pre_apply");
            engine.stream(StreamChunk::ApplyStarted { iteration });
            let allowed_files: HashSet<String> =
                plan.files.iter().map(|f| f.path.clone()).collect();
            let expected_hashes = build_expected_hashes(&file_context);
            let apply_result = apply_unified_diff(
                &engine.workspace,
                &diff,
                &allowed_files,
                &expected_hashes,
                cfg.apply_strategy.clone(),
            );

            match apply_result {
                Ok(success) => {
                    checkpoint_best_effort(&engine.workspace, "agent_post_apply");
                    let summary = format!(
                        "patch={} files={}",
                        success.patch_id,
                        success.changed_files.join(",")
                    );
                    engine.stream(StreamChunk::ApplyCompleted {
                        iteration,
                        success: true,
                        summary: summary.clone(),
                    });
                    feedback.apply_feedback = None;
                    feedback.last_diff_summary = Some(summary);

                    let verify_commands = if plan.verify_commands.is_empty() {
                        derive_verify_commands(&engine.workspace)
                    } else {
                        plan.verify_commands.clone()
                    };

                    engine.stream(StreamChunk::VerifyStarted {
                        iteration,
                        commands: verify_commands.clone(),
                    });

                    let verify_result = if let Ok(mut approval_guard) =
                        engine.approval_handler.lock()
                    {
                        let callback = approval_guard.as_mut().map(|cb| {
                            cb as &mut dyn FnMut(&deepseek_core::ToolCall) -> anyhow::Result<bool>
                        });
                        run_verify(
                            engine.tool_host.as_ref(),
                            &verify_commands,
                            cfg.verify_timeout_seconds,
                            callback,
                        )
                    } else {
                        run_verify(
                            engine.tool_host.as_ref(),
                            &verify_commands,
                            cfg.verify_timeout_seconds,
                            None,
                        )
                    };

                    if verify_result.success {
                        engine.stream(StreamChunk::VerifyCompleted {
                            iteration,
                            success: true,
                            summary: verify_result.summary.clone(),
                        });
                        let response = format_success_response(&plan, &verify_commands);
                        engine.stream(StreamChunk::ContentDelta(response.clone()));
                        engine.stream(StreamChunk::Done);
                        return Ok(response);
                    }

                    let (class, fingerprint, similarity, old_count, new_count) =
                        classify_verify_failure(
                            &verify_result.summary,
                            &mut failure_tracker,
                            &cfg.failure_classifier,
                            editor_retry_used,
                        );
                    let summary = format!(
                        "classification={}\nfingerprint={}\nsimilarity={:.3}\nerror_count_old={}\nerror_count_new={}\n\n{}",
                        class.as_str(),
                        fingerprint,
                        similarity,
                        old_count,
                        new_count,
                        verify_result.summary
                    );
                    engine.stream(StreamChunk::VerifyCompleted {
                        iteration,
                        success: false,
                        summary: summary.clone(),
                    });

                    if class == FailureClass::MechanicalVerifyFailure && !editor_retry_used {
                        editor_retry_used = true;
                        feedback.verify_feedback = Some(summary);
                        feedback.apply_feedback = None;
                        continue 'editor_cycle;
                    }

                    feedback.verify_feedback = Some(summary);
                    feedback.apply_feedback = None;
                    break 'editor_cycle;
                }
                Err(error) => {
                    engine.stream(StreamChunk::ApplyCompleted {
                        iteration,
                        success: false,
                        summary: error.to_feedback(),
                    });
                    feedback.apply_feedback = Some(error.to_feedback());
                    feedback.verify_feedback = None;
                    break 'editor_cycle;
                }
            }
        }
    }

    Err(anyhow!(
        "max iterations ({}) reached without passing verification; last_apply={:?}; last_verify={:?}",
        cfg.max_iterations,
        feedback.apply_feedback,
        feedback.verify_feedback
    ))
}

fn require_patch_approval(engine: &AgentEngine, diff: &str) -> Result<bool> {
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
            "touched_files": stats.touched_files,
            "loc_delta": stats.loc_delta,
            "max_files_without_approval": gate.max_files_without_approval,
            "max_loc_without_approval": gate.max_loc_without_approval,
        }),
        requires_approval: true,
    };

    if let Ok(mut guard) = engine.approval_handler.lock()
        && let Some(handler) = guard.as_mut()
    {
        return handler(&call);
    }
    Ok(false)
}

fn build_expected_hashes(file_context: &[EditorFileContext]) -> HashMap<String, String> {
    file_context
        .iter()
        .filter_map(|file| {
            file.base_hash
                .as_ref()
                .map(|hash| (file.path.clone(), hash.clone()))
        })
        .collect()
}

fn classify_verify_failure(
    summary: &str,
    tracker: &mut FailureTracker,
    cfg: &FailureClassifierConfig,
    editor_retry_used: bool,
) -> (FailureClass, String, f32, usize, usize) {
    let error_set = normalize_error_set(summary, cfg.fingerprint_lines as usize);
    let mut sorted = error_set.iter().cloned().collect::<Vec<_>>();
    sorted.sort();
    let fingerprint = hash_text(&sorted.join("\n"));
    let same_fingerprint = tracker
        .last_fingerprint
        .as_ref()
        .is_some_and(|last| last == &fingerprint);
    let old_set = tracker.last_error_set.clone().unwrap_or_default();

    if same_fingerprint {
        tracker.repeat_count = tracker.repeat_count.saturating_add(1);
    } else {
        tracker.repeat_count = 1;
    }

    let mut class = FailureClass::MechanicalVerifyFailure;
    let mut similarity = 0.0_f32;
    let repeat_threshold = cfg.repeat_threshold.max(1);
    if tracker.repeat_count >= repeat_threshold || (editor_retry_used && same_fingerprint) {
        if old_set.is_empty() {
            class = FailureClass::RepeatedVerifyFailure;
        } else {
            similarity = jaccard_similarity(&old_set, &error_set);
            let materially_reduced =
                error_set.len() + 1 <= old_set.len() || similarity < cfg.similarity_threshold;
            class = if materially_reduced {
                FailureClass::RepeatedVerifyFailure
            } else {
                FailureClass::DesignMismatch
            };
        }
    }

    tracker.last_fingerprint = Some(fingerprint.clone());
    tracker.last_error_set = Some(error_set.clone());
    (
        class,
        fingerprint,
        similarity,
        old_set.len(),
        error_set.len(),
    )
}

fn normalize_error_set(summary: &str, max_lines: usize) -> HashSet<String> {
    let mut out = HashSet::new();
    for line in summary.lines().take(max_lines.max(1)) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();
        let normalized = lower
            .chars()
            .map(|ch| if ch.is_ascii_digit() { '#' } else { ch })
            .collect::<String>();
        let normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
        if !normalized.is_empty() {
            out.insert(normalized);
        }
    }
    out
}

fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union <= f32::EPSILON {
        0.0
    } else {
        intersection / union
    }
}

fn checkpoint_best_effort(workspace: &Path, reason: &str) {
    if let Ok(manager) = MemoryManager::new(workspace) {
        let _ = manager.create_checkpoint(reason);
    }
}

fn load_architect_files(
    workspace: &Path,
    plan: &ArchitectPlan,
    max_files: usize,
    max_file_bytes: usize,
) -> Result<Vec<EditorFileContext>> {
    let mut files = Vec::new();
    for file in plan.files.iter().take(max_files.max(1)) {
        files.push(read_file_context(
            workspace,
            &file.path,
            None,
            max_file_bytes,
        )?);
    }
    Ok(files)
}

fn merge_requested_files(
    workspace: &Path,
    plan: &ArchitectPlan,
    current: &mut Vec<EditorFileContext>,
    requests: &[FileRequest],
    max_file_bytes: usize,
    max_context_range_lines: usize,
) -> Result<()> {
    let allowed: HashSet<&str> = plan.files.iter().map(|f| f.path.as_str()).collect();
    let mut by_path: HashMap<String, usize> = current
        .iter()
        .enumerate()
        .map(|(idx, f)| (f.path.clone(), idx))
        .collect();

    for req in requests {
        if !allowed.contains(req.path.as_str()) {
            return Err(anyhow!(
                "requested context for undeclared file: {}",
                req.path
            ));
        }
        ensure_repo_relative_path(&req.path).map_err(|e| anyhow!(e.reason))?;
        if let Some((start, end)) = req.range {
            let lines = end.saturating_sub(start).saturating_add(1);
            if lines > max_context_range_lines {
                return Err(anyhow!(
                    "requested context range too large for {}: {} lines > {}",
                    req.path,
                    lines,
                    max_context_range_lines
                ));
            }
        }

        let ctx = read_file_context(workspace, &req.path, req.range, max_file_bytes)?;
        if let Some(idx) = by_path.get(&req.path).copied() {
            current[idx] = ctx;
        } else {
            by_path.insert(req.path.clone(), current.len());
            current.push(ctx);
        }
    }
    Ok(())
}

fn read_file_context(
    workspace: &Path,
    rel_path: &str,
    range: Option<(usize, usize)>,
    max_file_bytes: usize,
) -> Result<EditorFileContext> {
    ensure_repo_relative_path(rel_path).map_err(|e| anyhow!(e.reason))?;
    let full_path = workspace.join(rel_path);
    if !full_path.exists() {
        return Ok(EditorFileContext {
            path: rel_path.to_string(),
            content: String::new(),
            partial: false,
            base_hash: None,
        });
    }

    let raw = fs::read_to_string(&full_path)?;
    let base_hash = Some(hash_text(&raw));
    let content = if let Some((start, end)) = range {
        slice_lines(&raw, start, end)
    } else {
        raw
    };

    if content.len() > max_file_bytes {
        Ok(EditorFileContext {
            path: rel_path.to_string(),
            content: content[..content.floor_char_boundary(max_file_bytes)].to_string(),
            partial: true,
            base_hash,
        })
    } else {
        Ok(EditorFileContext {
            path: rel_path.to_string(),
            content,
            partial: range.is_some(),
            base_hash,
        })
    }
}

fn slice_lines(content: &str, start: usize, end: usize) -> String {
    let start = start.max(1);
    let end = end.max(start);
    content
        .lines()
        .enumerate()
        .filter(|(idx, _)| {
            let line = idx + 1;
            line >= start && line <= end
        })
        .map(|(_, line)| line)
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_success_response(plan: &ArchitectPlan, verify_commands: &[String]) -> String {
    let mut out = String::new();
    out.push_str("Implemented the architect plan and verified locally.\n");
    if !plan.steps.is_empty() {
        out.push_str("\nPlan executed:\n");
        for step in &plan.steps {
            out.push_str(&format!("- {}\n", step));
        }
    }
    if !verify_commands.is_empty() {
        out.push_str("\nVerification:\n");
        for command in verify_commands {
            out.push_str(&format!("- `{}`\n", command));
        }
    }
    out.trim_end().to_string()
}

fn format_plan_only_response(plan: &ArchitectPlan) -> String {
    let mut out = String::new();
    out.push_str("No file edits were scheduled by architect.\n");
    if !plan.steps.is_empty() {
        out.push_str("\nPlan:\n");
        for step in &plan.steps {
            out.push_str(&format!("- {}\n", step));
        }
    }
    out.trim_end().to_string()
}

fn format_no_edit_response(plan: &ArchitectPlan, reason: &str) -> String {
    let mut out = String::new();
    out.push_str(reason);
    if !plan.steps.is_empty() {
        out.push_str("\n\nRecommended steps:\n");
        for step in &plan.steps {
            out.push_str(&format!("- {}\n", step));
        }
    }
    out.trim_end().to_string()
}
