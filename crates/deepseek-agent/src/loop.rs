use crate::apply::{ApplyFailure, apply_unified_diff};
use crate::architect::{ArchitectFeedback, ArchitectInput, ArchitectPlan, run_architect};
use crate::editor::{EditorFileContext, EditorInput, EditorResponse, FileRequest, run_editor};
use crate::verify::{derive_verify_commands, run_verify};
use crate::{AgentEngine, ChatOptions};
use anyhow::{Result, anyhow};
use deepseek_core::StreamChunk;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

pub fn run(engine: &AgentEngine, prompt: &str, options: &ChatOptions) -> Result<String> {
    let mut feedback = ArchitectFeedback::default();
    let cfg = &engine.cfg.agent_loop;

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

        let diff = loop {
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

            match response {
                EditorResponse::Diff(diff) => {
                    engine.stream(StreamChunk::EditorCompleted {
                        iteration,
                        status: "diff".to_string(),
                    });
                    break diff;
                }
                EditorResponse::NeedFile(requests) => {
                    engine.stream(StreamChunk::EditorCompleted {
                        iteration,
                        status: "need_file".to_string(),
                    });
                    merge_requested_files(
                        &engine.workspace,
                        &plan,
                        &mut file_context,
                        &requests,
                        cfg.max_file_bytes as usize,
                    )?;
                }
            }
        };

        engine.stream(StreamChunk::ApplyStarted { iteration });
        let allowed_files: HashSet<String> = plan.files.iter().map(|f| f.path.clone()).collect();
        let apply_result = apply_unified_diff(&engine.workspace, &diff, &allowed_files);
        match apply_result {
            Ok(success) => {
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

                let verify_result = if let Ok(mut approval_guard) = engine.approval_handler.lock() {
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

                engine.stream(StreamChunk::VerifyCompleted {
                    iteration,
                    success: verify_result.success,
                    summary: verify_result.summary.clone(),
                });

                if verify_result.success {
                    let response = format_success_response(&plan, &verify_commands);
                    engine.stream(StreamChunk::ContentDelta(response.clone()));
                    engine.stream(StreamChunk::Done);
                    return Ok(response);
                }

                feedback.verify_feedback = Some(verify_result.summary);
            }
            Err(error) => {
                engine.stream(StreamChunk::ApplyCompleted {
                    iteration,
                    success: false,
                    summary: error.to_feedback(),
                });
                feedback.apply_feedback = Some(error.to_feedback());
                feedback.verify_feedback = None;
                continue;
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
) -> Result<()> {
    let allowed: HashSet<&str> = plan.files.iter().map(|f| f.path.as_str()).collect();
    let mut by_path: HashMap<String, usize> = current
        .iter()
        .enumerate()
        .map(|(idx, f)| (f.path.clone(), idx))
        .collect();

    for req in requests {
        if !allowed.contains(req.path.as_str()) {
            continue;
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
    let full_path = workspace.join(rel_path);
    if !full_path.exists() {
        return Ok(EditorFileContext {
            path: rel_path.to_string(),
            content: String::new(),
            partial: false,
        });
    }

    let raw = fs::read_to_string(&full_path)?;
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
        })
    } else {
        Ok(EditorFileContext {
            path: rel_path.to_string(),
            content,
            partial: range.is_some(),
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

#[allow(dead_code)]
fn _apply_failure_to_feedback(error: &ApplyFailure) -> String {
    error.to_feedback()
}
