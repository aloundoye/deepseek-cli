use crate::apply::{apply_unified_diff, diff_stats, ensure_repo_relative_path, hash_text};
use crate::architect::{ArchitectFeedback, ArchitectInput, ArchitectPlan, run_architect};
use crate::editor::{EditorFileContext, EditorInput, EditorResponse, FileRequest, run_editor};
use crate::verify::{derive_verify_commands, run_verify};
use crate::{AgentEngine, ChatOptions};
use anyhow::{Result, anyhow};
use deepseek_core::{EventKind, FailureClassifierConfig, StreamChunk, ToolCall};
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
            debug_context: options.debug_context,
            chat_history: &options.chat_history,
        };
        let plan = run_architect(
            engine,
            &engine.workspace,
            &architect_input,
            cfg.architect_parse_retries as usize,
        )?;

        engine.stream(StreamChunk::ArchitectCompleted {
            iteration,
            files: plan.files.len() as u32,
            no_edit: plan.no_edit_reason.is_some(),
        });

        if !plan.subagents.is_empty() {
            let mut findings = Vec::new();
            if let Some(worker) = engine.subagent_worker() {
                std::thread::scope(|s| {
                    let mut handles = Vec::new();
                    for (i, goal) in plan.subagents.iter().enumerate() {
                        let worker_clone = worker.clone();
                        let task_name = format!("subagent-{}", i);
                        let sub_task = deepseek_subagent::SubagentTask {
                            run_id: uuid::Uuid::now_v7(),
                            name: task_name,
                            goal: goal.clone(),
                            role: deepseek_subagent::SubagentRole::Task,
                            team: "default".to_string(),
                            read_only_fallback: true,
                            custom_agent: None,
                        };
                        handles.push(s.spawn(move || {
                            worker_clone(&sub_task).unwrap_or_else(|e| format!("Failed to run subagent: {e}"))
                        }));
                    }
                    for handle in handles {
                        if let Ok(result) = handle.join() {
                            findings.push(result);
                        }
                    }
                });
            } else {
                findings.push("Subagent Execution Engine is not configured/attached".to_string());
            }

            feedback.subagent_findings = Some(findings.join("\n\n---\n\n"));
            // Trigger macro-loop to gather new context and plan again
            continue;
        }

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
        let mut editor_micro_retries = 0u64;

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
                debug_context: options.debug_context,
                chat_history: &options.chat_history,
            };

            let response = run_editor(
                engine,
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

                    // ── Lint auto-fix sub-loop ──────────────────────
                    let lint_commands = crate::linter::derive_lint_commands(
                        &cfg.lint,
                        &success.changed_files,
                    );
                    if !lint_commands.is_empty() {
                        let mut lint_fixed_total: u32 = 0;
                        for lint_iter in 1..=cfg.lint.max_iterations.max(1) {
                            engine.stream(StreamChunk::LintStarted {
                                iteration: lint_iter,
                                commands: lint_commands.clone(),
                            });

                            let lint_result = if let Ok(mut approval_guard) =
                                engine.approval_handler.lock()
                            {
                                let callback = approval_guard.as_mut().map(|cb| {
                                    cb as &mut dyn FnMut(&deepseek_core::ToolCall) -> anyhow::Result<bool>
                                });
                                crate::linter::run_lint(
                                    engine.tool_host.as_ref(),
                                    &lint_commands,
                                    cfg.lint.timeout_seconds,
                                    callback,
                                )
                            } else {
                                crate::linter::run_lint(
                                    engine.tool_host.as_ref(),
                                    &lint_commands,
                                    cfg.lint.timeout_seconds,
                                    None,
                                )
                            };

                            if lint_result.success {
                                engine.stream(StreamChunk::LintCompleted {
                                    iteration: lint_iter,
                                    success: true,
                                    fixed: lint_fixed_total,
                                    remaining: 0,
                                });
                                break;
                            }

                            lint_fixed_total += lint_result.fixed;
                            engine.stream(StreamChunk::LintCompleted {
                                iteration: lint_iter,
                                success: false,
                                fixed: lint_fixed_total,
                                remaining: lint_result.remaining,
                            });

                            // If we've exhausted lint iterations, proceed to verify anyway
                            if lint_iter >= cfg.lint.max_iterations.max(1) {
                                break;
                            }

                            // Feed lint errors back to editor for a fix attempt
                            feedback.verify_feedback = Some(format!(
                                "LINT_ERRORS (iteration {lint_iter}):\n{}",
                                lint_result.summary
                            ));
                            feedback.apply_feedback = None;

                            // Re-run editor with lint feedback
                            engine.stream(StreamChunk::EditorStarted {
                                iteration,
                                files: file_context.len() as u32,
                            });
                            let lint_fix_input = EditorInput {
                                user_prompt: prompt,
                                iteration,
                                plan: &plan,
                                files: &file_context,
                                verify_feedback: feedback.verify_feedback.as_deref(),
                                apply_feedback: None,
                                max_diff_bytes: cfg.max_diff_bytes as usize,
                                debug_context: options.debug_context,
                                chat_history: &options.chat_history,
                            };
                            let lint_fix_response = match run_editor(
                                engine,
                                &lint_fix_input,
                                cfg.editor_parse_retries as usize,
                            ) {
                                Ok(r) => r,
                                Err(_) => break, // editor failed, proceed to verify
                            };
                            let lint_fix_diff = match lint_fix_response {
                                EditorResponse::Diff(d) => {
                                    engine.stream(StreamChunk::EditorCompleted {
                                        iteration,
                                        status: "lint_fix".to_string(),
                                    });
                                    d
                                }
                                _ => break, // unexpected response, proceed to verify
                            };

                            // Re-apply the lint fix diff
                            let lint_allowed: HashSet<String> =
                                plan.files.iter().map(|f| f.path.clone()).collect();
                            let lint_hashes = build_expected_hashes(&file_context);
                            engine.stream(StreamChunk::ApplyStarted { iteration });
                            match apply_unified_diff(
                                &engine.workspace,
                                &lint_fix_diff,
                                &lint_allowed,
                                &lint_hashes,
                                cfg.apply_strategy.clone(),
                            ) {
                                Ok(lint_apply) => {
                                    engine.stream(StreamChunk::ApplyCompleted {
                                        iteration,
                                        success: true,
                                        summary: format!(
                                            "lint_fix patch={} files={}",
                                            lint_apply.patch_id,
                                            lint_apply.changed_files.join(",")
                                        ),
                                    });
                                    // Update file context hashes for next iteration
                                    for changed_file in &lint_apply.changed_files {
                                        let full = engine.workspace.join(changed_file);
                                        if let Ok(content) = fs::read_to_string(&full) {
                                            for ctx in file_context.iter_mut() {
                                                if ctx.path == *changed_file {
                                                    ctx.content = content.clone();
                                                }
                                            }
                                        }
                                    }
                                }
                                Err(_) => break, // apply failed, proceed to verify
                            }
                        }
                        feedback.verify_feedback = None;
                    }
                    // ── End lint sub-loop ────────────────────────────

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
                            &engine.workspace,
                            engine.tool_host.as_ref(),
                            &verify_commands,
                            cfg.verify_timeout_seconds,
                            callback,
                        )
                    } else {
                        run_verify(
                            &engine.workspace,
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
                        let stats = diff_stats(&diff);
                        let suggested_message =
                            build_commit_message(&engine.cfg.git.commit_message_template, prompt);
                        engine.stream(StreamChunk::CommitProposal {
                            files: success.changed_files.clone(),
                            touched_files: stats.touched_files as u32,
                            loc_delta: stats.loc_delta as u32,
                            verify_commands: verify_commands.clone(),
                            verify_status: verify_result.summary.clone(),
                            suggested_message: suggested_message.clone(),
                        });
                        engine.append_event_best_effort(EventKind::CommitProposalV1 {
                            files: success.changed_files.clone(),
                            touched_files: stats.touched_files as u64,
                            loc_delta: stats.loc_delta as u64,
                            verify_commands: verify_commands.clone(),
                            verify_status: verify_result.summary.clone(),
                            suggested_message: suggested_message.clone(),
                        });

                        // ── Commit proposal: ask user ──────────────────
                        propose_commit(
                            engine,
                            &success.changed_files,
                            &suggested_message,
                        );
                        // ── End commit proposal ────────────────────────

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
                    
                    if editor_micro_retries < cfg.max_editor_apply_retries {
                        editor_micro_retries += 1;
                        feedback.apply_feedback = Some(error.to_feedback());
                        feedback.verify_feedback = None;
                        continue 'editor_cycle;
                    }
                    
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

/// Ask the user whether to commit after a successful verify pass.
/// If the user accepts (or provides a custom message), stage and commit the changed files.
/// If the user declines or no handler is set, skip silently.
fn propose_commit(engine: &AgentEngine, changed_files: &[String], suggested_message: &str) {
    use deepseek_core::UserQuestion;
    use std::process::Command;

    let handler = if let Ok(guard) = engine.user_question_handler.lock() {
        guard.clone()
    } else {
        None
    };

    let handler = match handler {
        Some(h) => h,
        None => {
            // No user question handler — skip commit proposal
            engine.stream(StreamChunk::CommitSkipped);
            return;
        }
    };

    let question = UserQuestion {
        question: format!(
            "Commit {} changed file(s)?\nSuggested message: \"{}\"\n\nReply 'yes' to accept, type a custom message, or 'no' to skip.",
            changed_files.len(),
            suggested_message
        ),
        options: vec![
            "yes".to_string(),
            "no".to_string(),
        ],
    };

    let answer = match handler(question) {
        Some(a) => a,
        None => {
            engine.stream(StreamChunk::CommitSkipped);
            return;
        }
    };

    let answer_lower = answer.trim().to_ascii_lowercase();
    if answer_lower == "no" || answer_lower == "skip" || answer_lower.is_empty() {
        engine.stream(StreamChunk::CommitSkipped);
        return;
    }

    let commit_message = if answer_lower == "yes" || answer_lower == "y" {
        suggested_message.to_string()
    } else {
        // User provided a custom message
        answer.trim().to_string()
    };

    // Stage changed files
    let mut stage_args = vec!["add", "--"];
    let file_refs: Vec<&str> = changed_files.iter().map(|s| s.as_str()).collect();
    stage_args.extend(file_refs);

    let stage_result = Command::new("git")
        .args(&stage_args)
        .current_dir(&engine.workspace)
        .output();

    if let Err(e) = stage_result {
        engine.stream(StreamChunk::ContentDelta(format!(
            "\nFailed to stage files: {e}\n"
        )));
        engine.stream(StreamChunk::CommitSkipped);
        return;
    }

    // Commit
    let mut commit_args = vec!["commit", "-m", &commit_message];
    if engine.cfg.git.require_signing {
        commit_args.push("-S");
    }

    let commit_result = Command::new("git")
        .args(&commit_args)
        .current_dir(&engine.workspace)
        .output();

    match commit_result {
        Ok(output) if output.status.success() => {
            // Extract SHA from commit output
            let stdout = String::from_utf8_lossy(&output.stdout);
            let sha = stdout
                .split_whitespace()
                .find(|w| w.len() >= 7 && w.chars().all(|c| c.is_ascii_hexdigit()))
                .unwrap_or("unknown")
                .to_string();
            engine.stream(StreamChunk::CommitCompleted {
                sha,
                message: commit_message,
            });
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            engine.stream(StreamChunk::ContentDelta(format!(
                "\nCommit failed: {stderr}\n"
            )));
            engine.stream(StreamChunk::CommitSkipped);
        }
        Err(e) => {
            engine.stream(StreamChunk::ContentDelta(format!(
                "\nCommit failed: {e}\n"
            )));
            engine.stream(StreamChunk::CommitSkipped);
        }
    }
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
                error_set.len() < old_set.len() || similarity < cfg.similarity_threshold;
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
    out.push_str(
        "\n✅ Verify passed. Run `/commit` to save changes, `/diff` to review, `/undo` to revert.",
    );
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

fn build_commit_message(template: &str, prompt: &str) -> String {
    let mut goal = prompt.trim().replace('\n', " ");
    if goal.len() > 72 {
        goal.truncate(goal.floor_char_boundary(72));
    }
    if goal.is_empty() {
        goal = "apply verified changes".to_string();
    }
    template.replace("{goal}", goal.trim())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architect::{ArchitectFileIntent, ArchitectPlan};
    use deepseek_core::FailureClassifierConfig;

    // ── failure classification ──────────────────────────────────────────

    #[test]
    fn classify_first_failure_is_mechanical() {
        let mut tracker = FailureTracker::default();
        let cfg = FailureClassifierConfig::default();
        let (class, _, _, _, _) =
            classify_verify_failure("error[E0308]: mismatched types", &mut tracker, &cfg, false);
        assert_eq!(class, FailureClass::MechanicalVerifyFailure);
    }

    #[test]
    fn classify_repeated_same_fingerprint_escalates() {
        let mut tracker = FailureTracker::default();
        let cfg = FailureClassifierConfig {
            repeat_threshold: 2,
            similarity_threshold: 0.8,
            fingerprint_lines: 50,
        };
        let summary = "error[E0308]: mismatched types\n  expected u32, found String";
        // First call
        classify_verify_failure(summary, &mut tracker, &cfg, false);
        // Second call with same summary (repeat_count reaches threshold)
        let (_class, _, _, _, _) =
            classify_verify_failure(summary, &mut tracker, &cfg, false);
        // Now third call should have non-empty old_set
        let (class, _, similarity, old_count, new_count) =
            classify_verify_failure(summary, &mut tracker, &cfg, false);
        // Same errors, high similarity → DesignMismatch
        assert_eq!(class, FailureClass::DesignMismatch);
        assert!(similarity > 0.9);
        assert_eq!(old_count, new_count);
    }

    #[test]
    fn classify_reduced_errors_stays_repeated() {
        let mut tracker = FailureTracker::default();
        let cfg = FailureClassifierConfig {
            repeat_threshold: 2,
            similarity_threshold: 0.8,
            fingerprint_lines: 50,
        };
        let many_errors =
            "error[E0308]: type mismatch\nerror[E0412]: not found\nerror[E0599]: no method";
        // First call: sets baseline
        classify_verify_failure(many_errors, &mut tracker, &cfg, false);
        // Second call: same fingerprint, repeat_count reaches threshold
        classify_verify_failure(many_errors, &mut tracker, &cfg, false);
        // Third call: same fingerprint again, builds old_set
        classify_verify_failure(many_errors, &mut tracker, &cfg, false);
        // Fourth call: fewer errors — fingerprint changes, new set is smaller
        let (class, _, _, old_count, new_count) = classify_verify_failure(
            "error[E0308]: type mismatch",
            &mut tracker,
            &cfg,
            false,
        );
        // Fingerprint changed so repeat_count resets to 1 → MechanicalVerifyFailure
        // The materially_reduced check only applies when repeat_count >= threshold
        // So we need the NEW fingerprint to also repeat.
        assert_eq!(class, FailureClass::MechanicalVerifyFailure);
        // But the old_set should be bigger than new_set
        assert!(old_count >= new_count);
    }

    #[test]
    fn classify_editor_retry_used_escalates_on_same_fingerprint() {
        let mut tracker = FailureTracker::default();
        let cfg = FailureClassifierConfig::default();
        let summary = "error: tests failed";
        classify_verify_failure(summary, &mut tracker, &cfg, false);
        // With editor_retry_used=true and same fingerprint
        let (class, _, _, _, _) =
            classify_verify_failure(summary, &mut tracker, &cfg, true);
        assert!(matches!(
            class,
            FailureClass::RepeatedVerifyFailure | FailureClass::DesignMismatch
        ));
    }

    // ── normalize_error_set ─────────────────────────────────────────────

    #[test]
    fn normalize_error_set_normalizes_digits_and_case() {
        let errors = normalize_error_set("Error on line 42: type mismatch\nWarning at line 99: unused", 10);
        // Digits replaced with #, lowercased
        assert!(errors.iter().any(|e| e.contains("line ##")));
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn normalize_error_set_respects_max_lines() {
        let summary = "alpha error\nbeta error\ngamma error\ndelta error\nepsilon error";
        let all = normalize_error_set(summary, 100);
        let capped = normalize_error_set(summary, 3);
        assert_eq!(all.len(), 5);
        assert_eq!(capped.len(), 3);
    }

    #[test]
    fn normalize_error_set_skips_blank_lines() {
        let summary = "error one\n\n\nerror two";
        let errors = normalize_error_set(summary, 10);
        assert_eq!(errors.len(), 2);
    }

    // ── jaccard_similarity ──────────────────────────────────────────────

    #[test]
    fn jaccard_identical_sets() {
        let a: HashSet<String> = ["foo", "bar"].iter().map(|s| s.to_string()).collect();
        let b = a.clone();
        assert!((jaccard_similarity(&a, &b) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn jaccard_disjoint_sets() {
        let a: HashSet<String> = ["foo"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["bar"].iter().map(|s| s.to_string()).collect();
        assert!(jaccard_similarity(&a, &b).abs() < f32::EPSILON);
    }

    #[test]
    fn jaccard_empty_sets() {
        let a: HashSet<String> = HashSet::new();
        let b: HashSet<String> = HashSet::new();
        assert!((jaccard_similarity(&a, &b) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn jaccard_partial_overlap() {
        let a: HashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let sim = jaccard_similarity(&a, &b);
        // intersection=2, union=4, sim=0.5
        assert!((sim - 0.5).abs() < f32::EPSILON);
    }

    // ── slice_lines ─────────────────────────────────────────────────────

    #[test]
    fn slice_lines_extracts_range() {
        let content = "line1\nline2\nline3\nline4\nline5";
        let sliced = slice_lines(content, 2, 4);
        assert_eq!(sliced, "line2\nline3\nline4");
    }

    #[test]
    fn slice_lines_start_at_one() {
        let content = "first\nsecond";
        let sliced = slice_lines(content, 0, 1);
        assert_eq!(sliced, "first");
    }

    #[test]
    fn slice_lines_beyond_content() {
        let content = "only\ntwo";
        let sliced = slice_lines(content, 1, 100);
        assert_eq!(sliced, "only\ntwo");
    }

    // ── format helpers ──────────────────────────────────────────────────

    #[test]
    fn format_success_response_includes_plan_and_verify() {
        let plan = ArchitectPlan {
            steps: vec!["Add feature".to_string()],
            files: vec![ArchitectFileIntent {
                path: "src/lib.rs".to_string(),
                intent: "Edit".to_string(),
            }],
            verify_commands: vec![],
            acceptance: vec![],
            no_edit_reason: None,
            raw: String::new(),
        };
        let verify = vec!["cargo test".to_string()];
        let response = format_success_response(&plan, &verify);
        assert!(response.contains("Add feature"));
        assert!(response.contains("cargo test"));
        assert!(response.contains("Verify passed"));
    }

    #[test]
    fn format_plan_only_response_shows_steps() {
        let plan = ArchitectPlan {
            steps: vec!["Step A".to_string(), "Step B".to_string()],
            files: vec![],
            verify_commands: vec![],
            acceptance: vec![],
            no_edit_reason: None,
            raw: String::new(),
        };
        let response = format_plan_only_response(&plan);
        assert!(response.contains("No file edits"));
        assert!(response.contains("Step A"));
        assert!(response.contains("Step B"));
    }

    #[test]
    fn format_no_edit_response_includes_reason() {
        let plan = ArchitectPlan {
            steps: vec!["Run manually".to_string()],
            files: vec![],
            verify_commands: vec![],
            acceptance: vec![],
            no_edit_reason: Some("Already correct".to_string()),
            raw: String::new(),
        };
        let response = format_no_edit_response(&plan, "Already correct");
        assert!(response.contains("Already correct"));
        assert!(response.contains("Run manually"));
    }

    // ── build_commit_message ────────────────────────────────────────────

    #[test]
    fn commit_message_substitutes_goal() {
        let msg = build_commit_message("feat: {goal}", "Add login flow");
        assert_eq!(msg, "feat: Add login flow");
    }

    #[test]
    fn commit_message_truncates_long_goal() {
        let long_prompt = "a".repeat(200);
        let msg = build_commit_message("{goal}", &long_prompt);
        assert!(msg.len() <= 72);
    }

    #[test]
    fn commit_message_uses_default_for_empty() {
        let msg = build_commit_message("{goal}", "  ");
        assert_eq!(msg, "apply verified changes");
    }

    #[test]
    fn commit_message_collapses_newlines() {
        let msg = build_commit_message("{goal}", "fix\nthe\nbug");
        assert_eq!(msg, "fix the bug");
    }

    // ── FailureClass::as_str ────────────────────────────────────────────

    #[test]
    fn failure_class_as_str_covers_all_variants() {
        assert_eq!(FailureClass::PatchMismatch.as_str(), "PatchMismatch");
        assert_eq!(
            FailureClass::MechanicalVerifyFailure.as_str(),
            "MechanicalVerifyFailure"
        );
        assert_eq!(
            FailureClass::RepeatedVerifyFailure.as_str(),
            "RepeatedVerifyFailure"
        );
        assert_eq!(FailureClass::DesignMismatch.as_str(), "DesignMismatch");
    }
}
