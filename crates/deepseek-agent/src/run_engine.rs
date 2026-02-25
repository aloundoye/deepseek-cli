use crate::apply::{apply_unified_diff, diff_stats, ensure_repo_relative_path, hash_text};
use crate::architect::{ArchitectFeedback, ArchitectInput, ArchitectPlan, run_architect};
use crate::editor::{EditorFileContext, EditorInput, EditorResponse, FileRequest, run_editor};
use crate::verify::{derive_verify_commands, run_verify};
use crate::{AgentEngine, ChatOptions};
use anyhow::{Result, anyhow};
use deepseek_core::{EventKind, RunRecord, RunState, StreamChunk, ToolCall, FailureClassifierConfig};
use deepseek_memory::MemoryManager;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use chrono::Utc;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FailureClass {
    PatchMismatch,
    MechanicalVerifyFailure,
    RepeatedVerifyFailure,
    DesignMismatch,
}

impl FailureClass {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::PatchMismatch => "PatchMismatch",
            Self::MechanicalVerifyFailure => "MechanicalVerifyFailure",
            Self::RepeatedVerifyFailure => "RepeatedVerifyFailure",
            Self::DesignMismatch => "DesignMismatch",
        }
    }
}

#[derive(Default)]
pub(crate) struct FailureTracker {
    pub(crate) last_fingerprint: Option<String>,
    pub(crate) repeat_count: u64,
    pub(crate) last_error_set: Option<HashSet<String>>,
}

pub struct RunEngine<'a> {
    pub engine: &'a AgentEngine,
    pub run: RunRecord,
    pub options: ChatOptions,
    pub iteration: u64,

    pub feedback: ArchitectFeedback,
    pub plan: Option<ArchitectPlan>,
    pub file_context: Vec<EditorFileContext>,
    pub current_diff: Option<String>,
    pub context_requests: u64,
    pub editor_retry_used: bool,
    pub editor_micro_retries: u64,
    pub verify_commands: Vec<String>,
    failure_tracker: FailureTracker,
}

impl<'a> RunEngine<'a> {
    pub fn new(engine: &'a AgentEngine, prompt: &str, options: ChatOptions) -> Result<Self> {
        let session_id = engine.store.load_latest_session()?
            .map(|s| s.session_id)
            .unwrap_or_else(|| Uuid::now_v7());
            
        let run = RunRecord {
            run_id: Uuid::now_v7(),
            session_id,
            status: RunState::Context,
            prompt: prompt.to_string(),
            created_at: Utc::now().to_rfc3339(),
            updated_at: Utc::now().to_rfc3339(),
        };
        
        engine.store.save_run(&run)?;
        engine.append_event_best_effort(EventKind::RunStartedV1 {
            run_id: run.run_id,
            prompt: prompt.to_string(),
        });
        
        Ok(Self {
            engine,
            run,
            options,
            iteration: 1,
            feedback: ArchitectFeedback::default(),
            plan: None,
            file_context: Vec::new(),
            current_diff: None,
            context_requests: 0,
            editor_retry_used: false,
            editor_micro_retries: 0,
            verify_commands: Vec::new(),
            failure_tracker: FailureTracker::default(),
        })
    }

    pub fn advance(&mut self) -> Result<Option<String>> {
        // Evaluate the state machine until Final
        loop {
            let next_state = match self.run.status {
                RunState::Context => self.step_context()?,
                RunState::Architect => self.step_architect()?,
                RunState::GatherEvidence => self.step_gather_evidence()?,
                RunState::Subagents => self.step_subagents()?,
                RunState::Editor => self.step_editor()?,
                RunState::Apply => self.step_apply()?,
                RunState::Verify => self.step_verify()?,
                RunState::Recover => self.step_recover()?,
                RunState::Final => break,
            };
            
            if next_state != self.run.status {
                self.transition_to(next_state)?;
            }
        }
        
        // Return placeholder success message for now
        self.engine.append_event_best_effort(EventKind::RunCompletedV1 {
            run_id: self.run.run_id,
            success: true,
        });
        
        Ok(Some("Run completed".to_string()))
    }
    
    fn transition_to(&mut self, next: RunState) -> Result<()> {
        let from = self.run.status.clone();
        self.run.status = next.clone();
        self.run.updated_at = Utc::now().to_rfc3339();
        self.engine.store.save_run(&self.run)?;
        self.engine.append_event_best_effort(EventKind::RunStateChangedV1 {
            run_id: self.run.run_id,
            from,
            to: next,
        });
        Ok(())
    }

    fn step_context(&mut self) -> Result<RunState> {
        // Collect context or simply transition to Architect
        Ok(RunState::Architect)
    }

    fn step_architect(&mut self) -> Result<RunState> {
        // Reset per-iteration state trackers
        self.editor_micro_retries = 0;
        self.editor_retry_used = false;
        self.context_requests = 0;
        
        self.engine.stream(StreamChunk::ArchitectStarted { iteration: self.iteration });
        let cfg = &self.engine.cfg.agent_loop;

        let architect_input = ArchitectInput {
            user_prompt: &self.run.prompt,
            iteration: self.iteration,
            feedback: &self.feedback,
            max_files: cfg.max_files_per_iteration as usize,
            additional_dirs: &self.options.additional_dirs,
            debug_context: self.options.debug_context,
            chat_history: &self.options.chat_history,
        };

        let plan = run_architect(
            self.engine,
            &self.engine.workspace,
            &architect_input,
            cfg.architect_parse_retries as usize,
        )?;

        self.engine.stream(StreamChunk::ArchitectCompleted {
            iteration: self.iteration,
            files: plan.files.len() as u32,
            no_edit: plan.no_edit_reason.is_some(),
        });

        if !plan.subagents.is_empty() {
            let mut findings = Vec::new();
            if let Some(worker) = self.engine.subagent_worker() {
                std::thread::scope(|s| {
                    let mut handles = Vec::new();
                    for (name, goal) in &plan.subagents {
                        let worker_clone = worker.clone();
                        let _task_name = format!("subagent-{}", name);
                        let sub_task = deepseek_subagent::SubagentTask {
                            run_id: uuid::Uuid::now_v7(),
                            name: name.clone(),
                            goal: goal.clone(),
                            role: deepseek_subagent::SubagentRole::Task,
                            team: "default".to_string(),
                            read_only_fallback: true,
                            custom_agent: None,
                        };
                        handles.push(s.spawn(move || {
                            worker_clone(&sub_task).unwrap_or_else(|e| format!("Failed to run {name}: {e}"))
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

            self.feedback.subagent_findings = Some(findings.join("\n\n---\n\n"));
            self.iteration += 1;
            return Ok(RunState::Architect);
        }

        if !plan.retrieve_commands.is_empty() {
            let mut findings = Vec::new();
            for (query, scope) in &plan.retrieve_commands {
                match self.engine.tool_host.index().query(query, 10, scope.as_deref()) {
                    Ok(resp) => {
                        let mut block = format!("### Semantic Search: '{query}'");
                        if let Some(s) = scope {
                            block.push_str(&format!(" (scope: {s})"));
                        }
                        block.push_str(&format!("\nStatus: {}\n", resp.freshness));
                        if resp.results.is_empty() {
                            block.push_str("No matches found.\n");
                        } else {
                            for r in resp.results.iter().take(10) {
                                block.push_str(&format!("  - {}:{}: {}\n", r.path, r.line, r.excerpt.trim()));
                            }
                        }
                        findings.push(block);
                    }
                    Err(e) => {
                        findings.push(format!("### Semantic Search: '{query}'\nError: {e}"));
                    }
                }
            }
            self.feedback.retrieval_findings = Some(findings.join("\n\n"));
            self.iteration += 1;
            return Ok(RunState::Architect);
        }

        if !plan.tool_calls.is_empty() {
            let mut findings = Vec::new();
            std::thread::scope(|s| {
                let mut handles = Vec::new();
                for (name, args) in &plan.tool_calls {
                    let is_safe = matches!(
                        name.as_str(),
                        "fs_read" | "fs_list" | "fs_grep" | "fs_glob" | "index_query"
                    ) || name.starts_with("mcp_");
                    
                    if !is_safe {
                        findings.push(format!("### Tool: {name}\nError: Tool not allowed in Architect plan parallel execution (must be a read-only tool)."));
                        continue;
                    }
                    
                    let host = self.engine.tool_host.clone();
                    let name_clone = name.clone();
                    let args_clone = args.clone();
                    
                    handles.push(s.spawn(move || {
                        let parsed_args = serde_json::from_str(&args_clone).unwrap_or_else(|_| serde_json::json!({}));
                        let call = deepseek_core::ToolCall {
                            name: name_clone.clone(),
                            args: parsed_args,
                            requires_approval: false,
                        };
                        
                        use deepseek_core::ToolHost;
                        let approved = deepseek_core::ApprovedToolCall {
                            invocation_id: uuid::Uuid::now_v7(),
                            call,
                        };
                        let result = host.execute(approved);
                        
                        let mut block = format!("### Tool: {name_clone}");
                        if !args_clone.is_empty() {
                            block.push_str(&format!("\nArgs: {args_clone}"));
                        }
                        
                        let out_str = if result.output.is_string() {
                            result.output.as_str().unwrap().to_string()
                        } else {
                            serde_json::to_string_pretty(&result.output).unwrap_or_default()
                        };

                        if result.success {
                            block.push_str(&format!("\nResult:\n{out_str}"));
                        } else {
                            block.push_str(&format!("\nError:\n{out_str}"));
                        }
                        
                        (name_clone, block)
                    }));
                }
                
                let mut results: Vec<(String, String)> = handles.into_iter().filter_map(|h| h.join().ok()).collect();
                results.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                
                for (_, block) in results {
                    findings.push(block);
                }
            });
            
            self.feedback.tool_findings = Some(findings.join("\n\n---\n\n"));
            self.iteration += 1;
            return Ok(RunState::Architect);
        }

        if plan.no_edit_reason.is_some() || plan.files.is_empty() || self.options.mode == crate::ChatMode::Architect || self.options.force_plan_only {
            // Nothing to edit, complete run
            self.plan = Some(plan);
            return Ok(RunState::Final);
        }

        self.file_context = load_architect_files(
            &self.engine.workspace,
            &plan,
            cfg.max_files_per_iteration as usize,
            cfg.max_file_bytes as usize,
        )?;
        
        self.plan = Some(plan);
        self.context_requests = 0;
        self.editor_retry_used = false;
        self.editor_micro_retries = 0;

        Ok(RunState::Editor)
    }

    fn step_gather_evidence(&mut self) -> Result<RunState> {
        Ok(RunState::Subagents)
    }
    
    fn step_subagents(&mut self) -> Result<RunState> {
        Ok(RunState::Editor)
    }

    fn step_editor(&mut self) -> Result<RunState> {
        let cfg = &self.engine.cfg.agent_loop;
        let plan = self.plan.as_ref().unwrap();

        self.engine.stream(StreamChunk::EditorStarted {
            iteration: self.iteration,
            files: self.file_context.len() as u32,
        });

        let editor_input = EditorInput {
            user_prompt: &self.run.prompt,
            iteration: self.iteration,
            plan,
            files: &self.file_context,
            verify_feedback: self.feedback.verify_feedback.as_deref(),
            apply_feedback: self.feedback.apply_feedback.as_deref(),
            max_diff_bytes: cfg.max_diff_bytes as usize,
            debug_context: self.options.debug_context,
            chat_history: &self.options.chat_history,
        };

        let response = run_editor(
            self.engine,
            &editor_input,
            cfg.editor_parse_retries as usize,
        )?;

        match response {
            EditorResponse::Diff(diff) => {
                self.engine.stream(StreamChunk::EditorCompleted {
                    iteration: self.iteration,
                    status: "diff".to_string(),
                });
                self.current_diff = Some(diff);
                Ok(RunState::Apply)
            }
            EditorResponse::NeedContext(requests) => {
                self.engine.stream(StreamChunk::EditorCompleted {
                    iteration: self.iteration,
                    status: "need_context".to_string(),
                });

                self.context_requests = self.context_requests.saturating_add(1);
                if self.context_requests > cfg.max_context_requests_per_iteration as u64 {
                    self.feedback.apply_feedback = Some(format!(
                        "classification={}\neditor exceeded max context requests for iteration ({})",
                        FailureClass::PatchMismatch.as_str(),
                        cfg.max_context_requests_per_iteration
                    ));
                    self.feedback.verify_feedback = None;
                    // Fail back to architect if we can't complete the edit
                    self.iteration += 1;
                    if self.iteration > cfg.max_iterations.max(1) as u64 {
                        return Err(anyhow!(
                            "max iterations ({}) reached without passing verification; last_apply={:?}; last_verify={:?}",
                            cfg.max_iterations,
                            self.feedback.apply_feedback,
                            self.feedback.verify_feedback
                        ));
                    }
                    return Ok(RunState::Architect);
                }

                if let Err(err) = merge_requested_files(
                    &self.engine.workspace,
                    plan,
                    &mut self.file_context,
                    &requests,
                    cfg.max_file_bytes as usize,
                    cfg.max_context_range_lines as usize,
                ) {
                    self.feedback.apply_feedback = Some(format!(
                        "classification={}\n{}",
                        FailureClass::PatchMismatch.as_str(),
                        err
                    ));
                    self.feedback.verify_feedback = None;
                    self.iteration += 1;
                    if self.iteration > cfg.max_iterations.max(1) as u64 {
                        return Err(anyhow!(
                            "max iterations ({}) reached without passing verification; last_apply={:?}; last_verify={:?}",
                            cfg.max_iterations,
                            self.feedback.apply_feedback,
                            self.feedback.verify_feedback
                        ));
                    }
                    return Ok(RunState::Architect);
                }
                // Try editor again
                Ok(RunState::Editor)
            }
        }
    }

    fn step_apply(&mut self) -> Result<RunState> {
        let diff = self.current_diff.as_ref().unwrap();
        let plan = self.plan.as_ref().unwrap();
        let cfg = &self.engine.cfg.agent_loop;

        let patch_approved = require_patch_approval(self.engine, diff)?;
        if !patch_approved {
            self.feedback.apply_feedback = Some(format!(
                "classification={}\npatch exceeds safety gate and approval was denied",
                FailureClass::PatchMismatch.as_str()
            ));
            self.feedback.verify_feedback = None;
            self.iteration += 1;
            if self.iteration > cfg.max_iterations.max(1) as u64 {
                return Err(anyhow!(
                    "max iterations ({}) reached without passing verification; last_apply={:?}; last_verify={:?}",
                    cfg.max_iterations,
                    self.feedback.apply_feedback,
                    self.feedback.verify_feedback
                ));
            }

            return Ok(RunState::Architect);
        }

        checkpoint_best_effort(&self.engine.workspace, "agent_pre_apply");
        self.engine.stream(StreamChunk::ApplyStarted { iteration: self.iteration });
        
        let allowed_files: HashSet<String> = plan.files.iter().map(|f| f.path.clone()).collect();
        let expected_hashes = build_expected_hashes(&self.file_context);
        let apply_result = apply_unified_diff(
            &self.engine.workspace,
            diff,
            &allowed_files,
            &expected_hashes,
            cfg.apply_strategy.clone(),
        );

        match apply_result {
            Ok(success) => {

                checkpoint_best_effort(&self.engine.workspace, "agent_post_apply");
                let summary = format!(
                    "patch={} files={}",
                    success.patch_id,
                    success.changed_files.join(",")
                );
                self.engine.stream(StreamChunk::ApplyCompleted {
                    iteration: self.iteration,
                    success: true,
                    summary: summary.clone(),
                });
                self.feedback.apply_feedback = None;
                self.feedback.last_diff_summary = Some(summary);
                
                // ── Lint auto-fix sub-loop ──────────────────────
                let lint_commands = crate::linter::derive_lint_commands(
                    &cfg.lint,
                    &success.changed_files,
                );
                if !lint_commands.is_empty() {
                    let lint_iter = 1; // Simplify to single lint execution per iteration in the state machine
                    self.engine.stream(StreamChunk::LintStarted {
                        iteration: lint_iter,
                        commands: lint_commands.clone(),
                    });

                    let lint_result = if let Ok(mut approval_guard) =
                        self.engine.approval_handler.lock()
                    {
                        let callback = approval_guard.as_mut().map(|cb| {
                            cb as &mut dyn FnMut(&deepseek_core::ToolCall) -> anyhow::Result<bool>
                        });
                        crate::linter::run_lint(
                            self.engine.tool_host.as_ref(),
                            &lint_commands,
                            cfg.lint.timeout_seconds,
                            callback,
                        )
                    } else {
                        crate::linter::run_lint(
                            self.engine.tool_host.as_ref(),
                            &lint_commands,
                            cfg.lint.timeout_seconds,
                            None,
                        )
                    };

                    if lint_result.success {
                        self.engine.stream(StreamChunk::LintCompleted {
                            iteration: lint_iter,
                            success: true,
                            fixed: lint_result.fixed,
                            remaining: 0,
                        });
                    } else {
                        self.engine.stream(StreamChunk::LintCompleted {
                            iteration: lint_iter,
                            success: false,
                            fixed: lint_result.fixed,
                            remaining: lint_result.remaining,
                        });
                        
                        // Transition to editor for lint fixes
                        self.feedback.verify_feedback = Some(format!(
                            "LINT_ERRORS (iteration {}):\n{}",
                            lint_iter, lint_result.summary
                        ));
                        self.feedback.apply_feedback = None;
                        return Ok(RunState::Editor);
                    }
                }
                // ── End lint sub-loop ────────────────────────────

                Ok(RunState::Verify)
            }
            Err(error) => {

                self.engine.stream(StreamChunk::ApplyCompleted {
                    iteration: self.iteration,
                    success: false,
                    summary: error.to_feedback(),
                });
                
                if self.editor_micro_retries >= cfg.max_editor_apply_retries as u64 {
                    self.feedback.apply_feedback = Some(error.to_feedback());
                    self.feedback.verify_feedback = None;
                    self.iteration += 1;
                    if self.iteration > cfg.max_iterations.max(1) as u64 {
                        return Err(anyhow!(
                            "max iterations ({}) reached without passing verification; last_apply={:?}; last_verify={:?}",
                            cfg.max_iterations,
                            self.feedback.apply_feedback,
                            self.feedback.verify_feedback
                        ));
                    }

                    return Ok(RunState::Architect);
                }
                
                self.editor_micro_retries += 1;
                self.feedback.apply_feedback = Some(error.to_feedback());
                self.feedback.verify_feedback = None;

                Ok(RunState::Editor)
            }
        }
    }

    fn step_verify(&mut self) -> Result<RunState> {
        let plan = self.plan.as_ref().unwrap();
        let cfg = &self.engine.cfg.agent_loop;

        let verify_commands = if plan.verify_commands.is_empty() {
            derive_verify_commands(&self.engine.workspace)
        } else {
            plan.verify_commands.clone()
        };

        self.engine.stream(StreamChunk::VerifyStarted {
            iteration: self.iteration,
            commands: verify_commands.clone(),
        });


        let verify_result = if let Ok(mut approval_guard) = self.engine.approval_handler.lock() {
            let callback = approval_guard.as_mut().map(|cb| {
                cb as &mut dyn FnMut(&deepseek_core::ToolCall) -> anyhow::Result<bool>
            });
            run_verify(
                &self.engine.workspace,
                self.engine.tool_host.as_ref(),
                &verify_commands,
                cfg.verify_timeout_seconds,
                callback,
            )
        } else {
            run_verify(
                &self.engine.workspace,
                self.engine.tool_host.as_ref(),
                &verify_commands,
                cfg.verify_timeout_seconds,
                None,
            )
        };

        if verify_result.success {
            self.engine.stream(StreamChunk::VerifyCompleted {
                iteration: self.iteration,
                success: true,
                summary: verify_result.summary.clone(),
            });

            let changed_files: Vec<String> = plan
                .files
                .iter()
                .map(|f| f.path.clone())
                .collect();
            let suggested_message =
                build_commit_message(&self.engine.cfg.git.commit_message_template, &self.run.prompt);
            propose_commit(&self.engine, &changed_files, &suggested_message);

            // Reached success
            return Ok(RunState::Final);
        }

        let (class, fingerprint, similarity, old_count, new_count) =
            classify_verify_failure(
                &verify_result.summary,
                &mut self.failure_tracker,
                &cfg.failure_classifier,
                self.editor_retry_used,
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
        
        self.engine.stream(StreamChunk::VerifyCompleted {
            iteration: self.iteration,
            success: false,
            summary: summary.clone(),
        });

        if class == FailureClass::MechanicalVerifyFailure && !self.editor_retry_used {
            self.editor_retry_used = true;
            self.feedback.verify_feedback = Some(summary);
            self.feedback.apply_feedback = None;
            return Ok(RunState::Editor);
        }

        self.feedback.verify_feedback = Some(summary);
        self.feedback.apply_feedback = None;
        self.iteration += 1;
        
        if self.iteration > cfg.max_iterations.max(1) as u64 {
            return Err(anyhow!(
                "max iterations ({}) reached without passing verification; last_apply={:?}; last_verify={:?}",
                cfg.max_iterations,
                self.feedback.apply_feedback,
                self.feedback.verify_feedback
            ));
        }
        
        Ok(RunState::Architect)
    }

    fn step_recover(&mut self) -> Result<RunState> {
        Ok(RunState::Final)
    }
}

pub fn run(engine: &AgentEngine, prompt: &str, options: &ChatOptions) -> Result<String> {
    let mut run_engine = RunEngine::new(engine, prompt, options.clone())?;
    let response_str = run_engine.advance()?;

    // Final output generation
    fn format_plan_only_response(plan: &ArchitectPlan) -> String {
        let mut out = String::from("🛠️ **Plan Mode** (No file changes executed)\n\n");
        if !plan.steps.is_empty() {
            for (i, step) in plan.steps.iter().enumerate() {
                out.push_str(&format!("{}. {}\n", i + 1, step));
            }
        }
        if !plan.files.is_empty() {
            out.push_str("\n**Affected Files:**\n");
            for file in &plan.files {
                out.push_str(&format!("- `{}`: {}\n", file.path, file.intent));
            }
        }
        if !plan.verify_commands.is_empty() {
            out.push_str("\n**Verification Run:**\n");
            for cmd in &plan.verify_commands {
                out.push_str(&format!("- `{}`\n", cmd));
            }
        }
        out
    }

    let out = if let Some(plan) = &run_engine.plan {
        if plan.no_edit_reason.is_some() {
            format_no_edit_response(plan, plan.no_edit_reason.as_ref().unwrap())
        } else if plan.files.is_empty() || options.mode == crate::ChatMode::Architect || options.force_plan_only {
            format_plan_only_response(plan)
        } else {
            // Success response
            let verify_commands = if plan.verify_commands.is_empty() {
                derive_verify_commands(&engine.workspace)
            } else {
                plan.verify_commands.clone()
            };
            format_success_response(plan, &verify_commands)
        }
    } else {
        response_str.unwrap_or_else(|| "Completed".to_string())
    };
    
    engine.stream(StreamChunk::Done);
    Ok(out)
}

pub(crate) fn require_patch_approval(engine: &AgentEngine, diff: &str) -> Result<bool> {
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

    match engine.policy.dry_run(&call) {
        deepseek_policy::PermissionDryRunResult::Denied(msg) => {
            Err(anyhow::anyhow!("Policy explicitly denied patch apply: {msg}"))
        }
        deepseek_policy::PermissionDryRunResult::AutoApproved | deepseek_policy::PermissionDryRunResult::Allowed => Ok(true),
        deepseek_policy::PermissionDryRunResult::NeedsApproval => {
            if let Ok(mut guard) = engine.approval_handler.lock()
                && let Some(handler) = guard.as_mut()
            {
                handler(&call)
            } else {
                Ok(false)
            }
        }
    }
}

pub(crate) fn build_commit_message(template: &str, prompt: &str) -> String {
    let mut goal = prompt.trim().replace('\n', " ");
    if goal.len() > 72 {
        goal.truncate(goal.floor_char_boundary(72));
    }
    if goal.is_empty() {
        goal = "apply verified changes".to_string();
    }
    template.replace("{goal}", goal.trim())
}

/// Ask the user whether to commit after a successful verify pass.
/// If the user accepts (or provides a custom message), stage and commit the changed files.
/// If the user declines or no handler is set, skip silently.
pub(crate) fn propose_commit(engine: &AgentEngine, changed_files: &[String], suggested_message: &str) {
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

pub(crate) fn build_expected_hashes(file_context: &[EditorFileContext]) -> HashMap<String, String> {
    file_context
        .iter()
        .filter_map(|file| {
            file.base_hash
                .as_ref()
                .map(|hash| (file.path.clone(), hash.clone()))
        })
        .collect()
}

pub(crate) fn classify_verify_failure(
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

pub(crate) fn normalize_error_set(summary: &str, max_lines: usize) -> HashSet<String> {
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

pub(crate) fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
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

pub(crate) fn checkpoint_best_effort(workspace: &Path, reason: &str) {
    if let Ok(manager) = MemoryManager::new(workspace) {
        // Prefer git shadow commit; fall back to file-based checkpoint
        if manager.create_shadow_commit(reason).is_err() {
            let _ = manager.create_checkpoint(reason);
        }
    }
}

pub(crate) fn load_architect_files(
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

pub(crate) fn merge_requested_files(
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
                    "requested context range {}-{} exceeds max lines {} for {}",
                    start,
                    end,
                    max_context_range_lines,
                    req.path,
                ));
            }
        }

        if let Some(&idx) = by_path.get(&req.path) {
            let merged =
                merge_file_context(workspace, &current[idx], req.range, max_file_bytes)?;
            current[idx] = merged;
        } else {
            let new_ctx = read_file_context(workspace, &req.path, req.range, max_file_bytes)?;
            current.push(new_ctx);
            by_path.insert(req.path.clone(), current.len() - 1);
        }
    }
    Ok(())
}

pub(crate) fn read_file_context(
    workspace: &Path,
    rel_path: &str,
    range: Option<(usize, usize)>,
    max_file_bytes: usize,
) -> Result<EditorFileContext> {
    let target = workspace.join(rel_path);
    if !target.exists() {
        return Ok(EditorFileContext {
            path: rel_path.to_string(),
            content: String::new(),
            partial: false,
            base_hash: None,
        });
    }

    let full_content = fs::read_to_string(&target)?;
    let base_hash = hash_text(&full_content);

    if let Some((start_line, end_line)) = range {
        let lines: Vec<&str> = full_content.lines().collect();
        let start_idx = start_line.saturating_sub(1).min(lines.len());
        let end_idx = end_line.min(lines.len());
        let mut extracted = String::new();
        for (i, line) in lines.iter().enumerate().take(end_idx).skip(start_idx) {
            extracted.push_str(&format!("{:>4} | {}\n", i + 1, line));
        }
        return Ok(EditorFileContext {
            path: rel_path.to_string(),
            content: extracted,
            partial: true,
            base_hash: Some(base_hash),
        });
    }

    if full_content.len() > max_file_bytes {
        // Truncate to save tokens, mark as partial
        let mut truncated = full_content[..max_file_bytes].to_string();
        truncated.push_str("\n... [TRUNCATED] ...");
        return Ok(EditorFileContext {
            path: rel_path.to_string(),
            content: truncated,
            partial: true,
            base_hash: Some(base_hash),
        });
    }

    Ok(EditorFileContext {
        path: rel_path.to_string(),
        content: full_content,
        partial: false,
        base_hash: Some(base_hash),
    })
}

fn merge_file_context(
    workspace: &Path,
    current: &EditorFileContext,
    new_range: Option<(usize, usize)>,
    max_file_bytes: usize,
) -> Result<EditorFileContext> {
    if !current.partial {
        return Ok(current.clone());
    }
    // Simplistic merge: just re-read the file with the new range, or full file if no range
    read_file_context(workspace, &current.path, new_range, max_file_bytes)
}

pub(crate) fn format_no_edit_response(plan: &ArchitectPlan, reason: &str) -> String {
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

pub(crate) fn format_success_response(plan: &ArchitectPlan, verify_commands: &[String]) -> String {
    let mut out = String::from("✅ **Implemented & Verified** — Patch Applied\n\n**Steps Completed:**\n");
    for (i, step) in plan.steps.iter().enumerate() {
        out.push_str(&format!("{}. {}\n", i + 1, step));
    }
    if !verify_commands.is_empty() {
        out.push_str("\n**Verification Run:**\n");
        for cmd in verify_commands {
            out.push_str(&format!("- `{}`\n", cmd));
        }
    }
    out
}
