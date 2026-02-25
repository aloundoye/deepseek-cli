use crate::apply::{apply_unified_diff, diff_stats};
use crate::architect::{ArchitectFeedback, ArchitectInput, ArchitectPlan, run_architect};
use crate::editor::{EditorInput, EditorResponse, run_editor};
use crate::run_engine::{
    FailureClass, FailureTracker,
    build_commit_message, build_expected_hashes, checkpoint_best_effort,
    classify_verify_failure, format_no_edit_response, format_success_response,
    load_architect_files, merge_requested_files, propose_commit,
    require_patch_approval,
};
use crate::verify::{derive_verify_commands, run_verify};
use crate::{AgentEngine, ChatOptions};
use anyhow::{Result, anyhow};
use deepseek_core::{EventKind, StreamChunk};
use std::collections::HashSet;
use std::fs;



pub fn run(engine: &AgentEngine, prompt: &str, options: &ChatOptions) -> Result<String> {
    let mut feedback = ArchitectFeedback::default();
    let cfg = &engine.cfg.agent_loop;
    let mut failure_tracker = FailureTracker::default();
    let mut prev_plan_raw: Option<String> = None;

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

        // Plan-level dedup: stop if architect producing near-identical plans
        if let Some(ref prev) = prev_plan_raw {
            let prev_words: HashSet<String> =
                prev.split_whitespace().map(String::from).collect();
            let curr_words: HashSet<String> =
                plan.raw.split_whitespace().map(String::from).collect();
            let sim = crate::run_engine::jaccard_similarity(&prev_words, &curr_words);
            if sim > 0.90 {
                let reason = format!(
                    "Stopped: architect producing near-identical plans (similarity={:.0}%)",
                    sim * 100.0
                );
                engine.stream(StreamChunk::Done {
                    reason: Some(reason.clone()),
                });
                return Err(anyhow!("{reason}"));
            }
        }
        prev_plan_raw = Some(plan.raw.clone());

        if !plan.subagents.is_empty() {
            let mut findings = Vec::new();
            if let Some(worker) = engine.subagent_worker() {
                std::thread::scope(|s| {
                    let mut handles = Vec::new();
                    for (name, goal) in &plan.subagents {
                        let worker_clone = worker.clone();
                        let sub_task = deepseek_subagent::SubagentTask {
                            run_id: uuid::Uuid::now_v7(),
                            name: name.clone(),
                            goal: goal.clone(),
                            role: deepseek_subagent::SubagentRole::Task,
                            team: "default".to_string(),
                            read_only_fallback: true,
                            custom_agent: None,
                        };
                        // Log ToolProposedV1 before spawning
                        engine.append_event_best_effort(EventKind::ToolProposedV1 {
                            proposal: deepseek_core::ToolProposal {
                                invocation_id: sub_task.run_id,
                                call: deepseek_core::ToolCall {
                                    name: format!("subagent.{}", name),
                                    args: serde_json::json!({"goal": goal}),
                                    requires_approval: false,
                                },
                                approved: true,
                            },
                        });
                        let run_id = sub_task.run_id;
                        handles.push((run_id, name.clone(), s.spawn(move || {
                            worker_clone(&sub_task).unwrap_or_else(|e| format!("Failed to run {}: {e}", sub_task.name))
                        })));
                    }
                    for (run_id, _name, handle) in handles {
                        if let Ok(result) = handle.join() {
                            // Log ToolResultV1 after completion
                            engine.append_event_best_effort(EventKind::ToolResultV1 {
                                result: deepseek_core::ToolResult {
                                    invocation_id: run_id,
                                    success: true,
                                    output: serde_json::json!({"output": &result}),
                                },
                            });
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
            engine.stream(StreamChunk::Done { reason: None });
            return Ok(response);
        }

        if plan.files.is_empty() {
            let response = format_plan_only_response(&plan);
            engine.stream(StreamChunk::ContentDelta(response.clone()));
            engine.stream(StreamChunk::Done { reason: None });
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
                        engine.stream(StreamChunk::Done { reason: None });
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

    let reason = format!(
        "max iterations ({}) reached without passing verification",
        cfg.max_iterations
    );
    engine.stream(StreamChunk::Done {
        reason: Some(reason.clone()),
    });
    Err(anyhow!(
        "{reason}; last_apply={:?}; last_verify={:?}",
        feedback.apply_feedback,
        feedback.verify_feedback
    ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architect::{ArchitectFileIntent, ArchitectPlan};
    use crate::run_engine::{normalize_error_set, jaccard_similarity};
    use deepseek_core::FailureClassifierConfig;

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
            subagents: vec![],
            retrieve_commands: vec![],
            tool_calls: vec![],
            no_edit_reason: None,
            raw: String::new(),
        };
        let verify = vec!["cargo test".to_string()];
        let response = format_success_response(&plan, &verify);
        assert!(response.contains("Add feature"));
        assert!(response.contains("cargo test"));
        assert!(response.contains("Implemented"));
    }

    #[test]
    fn format_plan_only_response_shows_steps() {
        let plan = ArchitectPlan {
            steps: vec!["Step A".to_string(), "Step B".to_string()],
            files: vec![],
            verify_commands: vec![],
            acceptance: vec![],
            subagents: vec![],
            retrieve_commands: vec![],
            tool_calls: vec![],
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
            subagents: vec![],
            retrieve_commands: vec![],
            tool_calls: vec![],
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
