{
        let mut session = self.ensure_session()?;

        // Fire SessionStart hook (legacy + config-based).
        self.tool_host.fire_session_hooks("sessionstart");
        {
            let mut input = self.hook_input(HookEvent::SessionStart);
            input.session_type = Some("startup".to_string());
            self.fire_hook(HookEvent::SessionStart, &input);
        }
        self.transition(&mut session, SessionState::Planning)?;

        // Resolve @server:uri MCP resource references in the prompt
        let prompt = self.resolve_mcp_resources(prompt);
        let prompt = prompt.as_str();

        // Fire UserPromptSubmit hook (can block the prompt).
        {
            let mut input = self.hook_input(HookEvent::UserPromptSubmit);
            input.prompt = Some(prompt.to_string());
            let hr = self.fire_hook(HookEvent::UserPromptSubmit, &input);
            if hr.blocked {
                let reason = hr
                    .block_reason
                    .unwrap_or_else(|| "Blocked by hook".to_string());
                return Ok(format!("[hook blocked prompt] {reason}"));
            }
        }

        self.emit(
            session.session_id,
            EventKind::TurnAddedV1 {
                role: "user".to_string(),
                content: prompt.to_string(),
            },
        )?;
        self.emit(
            session.session_id,
            EventKind::ChatTurnV1 {
                message: ChatMessage::User {
                    content: prompt.to_string(),
                },
            },
        )?;

        // Build system prompt with workspace context
        let mut plan_mode_active =
            self.policy.permission_mode() == deepseek_policy::PermissionMode::Plan;
        let system_prompt = if let Some(ref override_prompt) = options.system_prompt_override {
            override_prompt.clone()
        } else {
            let base = self.build_chat_system_prompt(prompt)?;
            if let Some(ref append) = options.system_prompt_append {
                format!("{base}\n\n{append}")
            } else {
                base
            }
        };
        // Build tiered tool set: core + contextual + tool_search meta-tool.
        // Extended tools are available on-demand via tool_search.
        let (all_tools, extended_tool_defs) = if options.tools {
            let full_defs = tool_definitions();
            let signals = deepseek_tools::detect_signals(prompt, &self.workspace);
            let (mut active, extended) =
                deepseek_tools::tiered_tool_definitions(full_defs, &signals);

            // Add tool_search meta-tool for discovering extended tools
            if !extended.is_empty() {
                active.push(deepseek_tools::tool_search_definition());
            }

            self.observer.verbose_log(&format!(
                "Tool tiers: {} active, {} extended (discoverable via tool_search)",
                active.len(),
                extended.len()
            ));

            // Merge plugin tool definitions
            let plugin_defs = deepseek_tools::plugin_tool_definitions(&self.workspace);
            if !plugin_defs.is_empty() {
                self.observer
                    .verbose_log(&format!("Plugins: {} tools merged", plugin_defs.len()));
                active.extend(plugin_defs);
            }
            // Discover and merge MCP tools
            let mcp_defs = self.discover_mcp_tool_definitions();
            let mcp_tool_count = mcp_defs.len();
            if mcp_tool_count > 0 {
                // If MCP tools exceed threshold, use lazy loading via mcp_search
                let mcp_token_estimate = mcp_tool_count as u64 * 50; // ~50 tokens per def
                let context_threshold = self.cfg.llm.context_window_tokens / 10; // 10% of context
                if mcp_token_estimate > context_threshold {
                    // Too many MCP tools — add mcp_search instead for on-demand discovery
                    active.push(mcp_search_tool_definition());
                    self.observer.verbose_log(&format!(
                        "MCP: {} tools exceed context threshold, using mcp_search lazy loading",
                        mcp_tool_count
                    ));
                } else {
                    active.extend(mcp_defs);
                    self.observer
                        .verbose_log(&format!("MCP: {} tools merged", mcp_tool_count));
                }
            }
            let filtered = filter_tool_definitions(
                active,
                options.allowed_tools.as_deref(),
                options.disallowed_tools.as_deref(),
            );
            (filtered, extended)
        } else {
            (vec![], vec![])
        };
        // In plan mode, restrict to read-only tools
        let mut tools = if plan_mode_active {
            all_tools
                .iter()
                .filter(|t| PLAN_MODE_TOOLS.contains(&t.function.name.as_str()))
                .cloned()
                .collect::<Vec<_>>()
        } else {
            all_tools.clone()
        };

        // In strict review mode, filter to read-only tools only
        if self.cfg.policy.review_mode == deepseek_core::ReviewMode::Strict {
            let before = tools.len();
            tools.retain(|t| deepseek_core::ReviewMode::is_read_only_tool(&t.function.name));
            if tools.len() < before {
                self.observer.verbose_log(&format!(
                    "review_mode=strict: filtered {} tools → {} read-only tools",
                    before,
                    tools.len()
                ));
            }
        }

        self.observer
            .verbose_log(&format!("chat: {} tool definitions loaded", tools.len()));

        // Initialize conversation with system + user message
        let mut messages: Vec<ChatMessage> = vec![ChatMessage::System {
            content: system_prompt,
        }];

        // Load prior conversation turns if resuming an existing session
        let projection = self.store.rebuild_from_events(session.session_id)?;
        if !projection.chat_messages.is_empty() {
            // Prefer structured ChatTurnV1 messages (preserves tool_call IDs)
            messages.extend(projection.chat_messages.iter().cloned());
        } else if projection.transcript.len() > 1 {
            // Fallback: legacy string-based transcript (older sessions)
            let mut pending_tool_summaries: Vec<String> = Vec::new();
            for entry in &projection.transcript {
                if let Some(content) = entry.strip_prefix("user: ") {
                    if !pending_tool_summaries.is_empty() {
                        messages.push(ChatMessage::User {
                            content: format!(
                                "[Prior tool results]\n{}",
                                pending_tool_summaries.join("\n")
                            ),
                        });
                        pending_tool_summaries.clear();
                    }
                    messages.push(ChatMessage::User {
                        content: content.to_string(),
                    });
                } else if let Some(content) = entry.strip_prefix("assistant: ") {
                    if !pending_tool_summaries.is_empty() {
                        messages.push(ChatMessage::User {
                            content: format!(
                                "[Prior tool results]\n{}",
                                pending_tool_summaries.join("\n")
                            ),
                        });
                        pending_tool_summaries.clear();
                    }
                    messages.push(ChatMessage::Assistant {
                        content: Some(content.to_string()),
                        reasoning_content: None,
                        tool_calls: vec![],
                    });
                } else if let Some(content) = entry.strip_prefix("tool: ") {
                    pending_tool_summaries.push(content.to_string());
                }
            }
            if !pending_tool_summaries.is_empty() {
                messages.push(ChatMessage::User {
                    content: format!(
                        "[Prior tool results]\n{}",
                        pending_tool_summaries.join("\n")
                    ),
                });
            }
        }

        // Add the current user message
        messages.push(ChatMessage::User {
            content: prompt.to_string(),
        });

        let max_turns = self.max_turns.unwrap_or(200);
        let mut turn_count: u64 = 0;
        let mut failure_streak: u32 = 0;
        let mut empty_response_count: u32 = 0;
        let mut tool_choice_retried = false;
        let mode_router_config = ModeRouterConfig::from_router_config(&self.cfg.router);
        let mut failure_tracker = FailureTracker::default();
        let mut current_mode = AgentMode::V3Autopilot;
        let mut budget_warned = false;

        // ── Plan discipline: detect triggers and generate plan if needed ──
        let context_signals = deepseek_tools::detect_signals(prompt, &self.workspace);
        let planning_triggers = detect_planning_triggers(
            prompt,
            context_signals.codebase_file_count,
            failure_streak,
            context_signals.prompt_is_complex,
        );
        let mut plan_state = PlanState::default();
        if planning_triggers.should_plan() {
            let derived_verify = derive_verify_commands(&self.workspace);
            if let Some(plan) = self.generate_plan_for_chat(prompt, &messages) {
                let verify = if plan.verification.is_empty() {
                    derived_verify
                } else {
                    plan.verification.clone()
                };
                self.observer.verbose_log(&format!(
                    "plan_discipline: generated {}-step plan for: {}",
                    plan.steps.len(),
                    plan.goal
                ));
                plan_state = PlanState::with_plan(plan, verify);
                plan_state.start_execution();
            } else {
                self.observer
                    .verbose_log("plan_discipline: plan generation failed, continuing without");
            }
        }

        self.transition(&mut session, SessionState::ExecutingStep)?;

        if let Some(plan) = plan_state.plan.as_ref() {
            let spawn_decision = decide_chat_subagent_spawn(
                &options,
                &context_signals,
                plan,
                self.subagents.max_concurrency,
            );
            if spawn_decision.should_spawn {
                match self.run_subagents(session.session_id, plan, Some(spawn_decision.task_budget))
                {
                    Ok(notes) if !notes.is_empty() => {
                        let summary = summarize_subagent_notes(&notes);
                        self.emit(
                            session.session_id,
                            EventKind::TurnAddedV1 {
                                role: "assistant".to_string(),
                                content: format!("[subagents]\n{summary}"),
                            },
                        )?;
                        self.tool_host.fire_session_hooks("notification");
                        messages.push(ChatMessage::User {
                            content: format!(
                                "<subagent-findings>\n{summary}\nUse these findings as additional context while executing the task.\n</subagent-findings>"
                            ),
                        });
                    }
                    Ok(_) => {}
                    Err(e) => {
                        self.observer
                            .verbose_log(&format!("chat subagent orchestration failed: {e}"));
                    }
                }
            } else if spawn_decision.blocked_by_tools {
                let notice = format!(
                    "[subagents] complex task detected (score {:.2}) but tools are disabled (--tools=false), so subagent orchestration is skipped.",
                    spawn_decision.score
                );
                self.emit(
                    session.session_id,
                    EventKind::TurnAddedV1 {
                        role: "assistant".to_string(),
                        content: notice.clone(),
                    },
                )?;
                if let Ok(cb_guard) = self.stream_callback.lock()
                    && let Some(ref cb) = *cb_guard
                {
                    cb(StreamChunk::ContentDelta(format!("{notice}\n")));
                }
            }
        }

        loop {
            turn_count += 1;
            if turn_count > max_turns {
                if let Err(e) = self.transition(&mut session, SessionState::Failed) {
                    self.observer
                        .warn_log(&format!("session: failed to transition to Failed: {e}"));
                }
                if let Ok(manager) = MemoryManager::new(&self.workspace)
                    && let Err(e) = manager.append_auto_memory_observation(AutoMemoryObservation {
                        objective: prompt.to_string(),
                        summary: format!(
                            "chat failed: max turn limit ({}) reached, failure_streak={}",
                            max_turns, failure_streak
                        ),
                        success: false,
                        patterns: vec![
                            format!("turns={max_turns} failure_streak={failure_streak}"),
                            "max turn limit suggests task is too complex for single session"
                                .to_string(),
                        ],
                        recorded_at: None,
                    })
                {
                    self.observer
                        .warn_log(&format!("memory: failed to persist observation: {e}"));
                }
                return Err(anyhow!(
                    "Reached maximum turn limit ({max_turns}). Use --max-turns to increase the limit, or break the task into smaller pieces."
                ));
            }

            // Budget check
            if let Some(max_usd) = self.max_budget_usd {
                let cost = self
                    .store
                    .total_session_cost(session.session_id)
                    .unwrap_or(0.0);
                if cost >= max_usd {
                    if let Err(e) = self.transition(&mut session, SessionState::Failed) {
                        self.observer
                            .warn_log(&format!("session: failed to transition to Failed: {e}"));
                    }
                    if let Ok(manager) = MemoryManager::new(&self.workspace)
                        && let Err(e) = manager.append_auto_memory_observation(AutoMemoryObservation {
                            objective: prompt.to_string(),
                            summary: format!(
                                "chat failed: budget limit (${:.2}/${:.2}) at turn {}, failure_streak={}",
                                cost, max_usd, turn_count, failure_streak
                            ),
                            success: false,
                            patterns: vec![format!(
                                "turns={turn_count} failure_streak={failure_streak} cost=${cost:.2}"
                            )],
                            recorded_at: None,
                        })
                    {
                        self.observer
                            .warn_log(&format!("memory: failed to persist observation: {e}"));
                    }
                    return Err(anyhow!(
                        "Budget limit reached (${:.2}/${:.2}). Use --max-budget-usd to increase the limit.",
                        cost,
                        max_usd
                    ));
                }
                // Warn at 80% budget usage (once per session)
                if !budget_warned && cost >= max_usd * 0.8 {
                    budget_warned = true;
                    let remaining = max_usd - cost;
                    if let Ok(cb_guard) = self.stream_callback.lock()
                        && let Some(ref cb) = *cb_guard
                    {
                        cb(StreamChunk::ContentDelta(format!(
                            "\n⚠ Budget warning: ${:.2}/${:.2} used ({:.0}%). ${:.2} remaining.\n",
                            cost,
                            max_usd,
                            cost / max_usd * 100.0,
                            remaining
                        )));
                    }
                    self.observer.verbose_log(&format!(
                        "budget warning: ${:.2}/{:.2} ({:.0}%)",
                        cost,
                        max_usd,
                        cost / max_usd * 100.0
                    ));
                }
            }

            // Ensure chat history respects assistant(tool_calls)->tool pairing rules.
            let repair_stats = sanitize_chat_history_for_tool_calls(&mut messages);
            if repair_stats.changed() {
                self.observer.verbose_log(&format!(
                    "normalized chat history: dropped {} orphan tool msgs, stripped {} unresolved tool calls",
                    repair_stats.dropped_tool_messages, repair_stats.stripped_tool_calls
                ));
            }

            // Context window compaction — budget-aware threshold
            let token_count = estimate_messages_tokens(&messages);
            let effective_window = self
                .cfg
                .llm
                .context_window_tokens
                .saturating_sub(self.cfg.context.reserved_overhead_tokens)
                .saturating_sub(self.cfg.context.response_budget_tokens);
            let threshold = (effective_window as f64
                * self.cfg.context.auto_compact_threshold.clamp(0.1, 1.0) as f64)
                as u64;
            if token_count > threshold && messages.len() > 4 {
                // Fire PreCompact hook — hooks can inject additional context.
                let pre_compact_input = self.hook_input(HookEvent::PreCompact);
                let hr = self.fire_hook(HookEvent::PreCompact, &pre_compact_input);
                Self::inject_hook_context(&mut messages, &hr);

                // Keep system prompt (index 0) plus a tail window, but never
                // start the tail on a Tool message (that would orphan it).
                let desired_tail = self
                    .cfg
                    .context
                    .compaction_tail_window
                    .unwrap_or(12)
                    .min(messages.len() - 1);
                let tail_start = compaction_tail_start(&messages, desired_tail);
                let compacted_range = &messages[1..tail_start];
                if !compacted_range.is_empty() {
                    let summary = self.llm_compact_summary(compacted_range);
                    let from_turn = 1u64;
                    let to_turn = compacted_range.len() as u64;
                    let summary_id = Uuid::now_v7();

                    // ── Plan discipline: pin plan summary into compaction ──
                    let plan_summary_addendum = if plan_state.plan.is_some() {
                        format!("\n\n{}", plan_state.to_compact_summary())
                    } else {
                        String::new()
                    };

                    let mut new_messages = vec![messages[0].clone()];
                    new_messages.push(ChatMessage::User {
                        content: format!(
                            "[Context compacted — prior conversation summary]\n{summary}{plan_summary_addendum}"
                        ),
                    });
                    new_messages.extend_from_slice(&messages[tail_start..]);
                    let new_token_count = estimate_messages_tokens(&new_messages);

                    self.emit(
                        session.session_id,
                        EventKind::ContextCompactedV1 {
                            summary_id,
                            from_turn,
                            to_turn,
                            token_delta_estimate: token_count as i64 - new_token_count as i64,
                            replay_pointer: String::new(),
                        },
                    )?;

                    messages = new_messages;
                    let post_compact_repairs = sanitize_chat_history_for_tool_calls(&mut messages);
                    if post_compact_repairs.changed() {
                        self.observer.verbose_log(&format!(
                            "post-compact history normalization: dropped {} orphan tool msgs, stripped {} unresolved tool calls",
                            post_compact_repairs.dropped_tool_messages,
                            post_compact_repairs.stripped_tool_calls
                        ));
                    }
                }
            }

            // Route model selection based on complexity signals
            let verification_failure_count = self
                .store
                .list_recent_verification_runs(session.session_id, 10)
                .unwrap_or_default()
                .iter()
                .filter(|r| !r.success)
                .count() as f32;
            let conversation_depth = (messages.len() as f32 / 20.0).min(1.0);
            let ambiguity = if prompt.contains('?')
                || prompt.contains(" or ")
                || prompt.contains("which ")
                || prompt.contains("should ")
            {
                0.5
            } else {
                0.0
            };
            let signals = RouterSignals {
                prompt_complexity: (prompt.len() as f32 / 500.0).min(1.0),
                repo_breadth: conversation_depth,
                failure_streak: (failure_streak as f32 / 3.0).min(1.0),
                verification_failures: (verification_failure_count / 3.0).min(1.0),
                low_confidence: if empty_response_count > 0 { 0.6 } else { 0.2 },
                ambiguity_flags: ambiguity,
            };
            let mut decision = self.router.select(LlmUnit::Executor, signals);
            if options.force_max_think && !decision.thinking_enabled {
                if !decision
                    .reason_codes
                    .iter()
                    .any(|code| code == "user_force_max_think")
                {
                    decision
                        .reason_codes
                        .push("user_force_max_think".to_string());
                }
                decision.thinking_enabled = true;
                decision.escalated = true;
                decision.confidence = decision.confidence.max(0.95);
                decision.score = decision.score.max(self.cfg.router.threshold_high);
            }
            self.emit(
                session.session_id,
                EventKind::RouterDecisionV1 {
                    decision: decision.clone(),
                },
            )?;

            let request_model = decision.selected_model.clone();

            // Build thinking config when the router (or user) requests thinking mode.
            // DeepSeek V3.2 supports thinking + tools in a single call (unified mode).
            // When unified mode is disabled, thinking is only used without tools
            // (the old DSML leak workaround).
            let thinking = if decision.thinking_enabled
                && (!options.tools || self.cfg.router.unified_thinking_tools)
            {
                Some(ThinkingConfig::enabled(
                    session.budgets.max_think_tokens.max(4096),
                ))
            } else {
                None
            };

            // Strip reasoning_content from assistant messages that belong to
            // *prior* conversation turns (before the current user question).
            // Per DeepSeek V3.2 docs: keep reasoning_content within the current
            // tool loop so the model retains its logical thread, but clear it
            // from earlier turns to save bandwidth.
            //
            // Find the index of the last User message — everything before it
            // belongs to prior turns and should have reasoning_content stripped.
            let last_user_idx = messages
                .iter()
                .rposition(|m| matches!(m, ChatMessage::User { .. }))
                .unwrap_or(0);
            for msg in &mut messages[..last_user_idx] {
                if let ChatMessage::Assistant {
                    reasoning_content, ..
                } = msg
                {
                    *reasoning_content = None;
                }
            }

            // ── Plan discipline: inject current step context ──
            let injected_plan_ctx = inject_step_context(&mut messages, &plan_state);

            let request = ChatRequest {
                model: request_model.clone(),
                messages: messages.clone(),
                tools: tools.clone(),
                tool_choice: if options.tools {
                    ToolChoice::auto()
                } else {
                    ToolChoice::none()
                },
                max_tokens: session.budgets.max_think_tokens.max(4096),
                // DeepSeek API requires temperature to be omitted when thinking is enabled.
                temperature: if thinking.is_some() { None } else { Some(0.0) },
                thinking: thinking.clone(),
            };

            // Remove injected plan context immediately so it doesn't persist in history
            if injected_plan_ctx.is_some() {
                remove_step_context(&mut messages);
            }

            self.observer.verbose_log(&format!(
                "turn {turn_count}: calling LLM model={} messages={} tools={}",
                request.model,
                request.messages.len(),
                request.tools.len()
            ));

            // Call the LLM with streaming (clone the Arc so it persists across turns)
            let response_model = request_model.clone();
            let response = {
                let cb = self.stream_callback.lock().ok().and_then(|g| g.clone());
                if let Some(cb) = cb {
                    self.llm.complete_chat_streaming(&request, cb)?
                } else {
                    self.llm.complete_chat(&request)?
                }
            };

            // Escalation retry: if base model returned empty response and thinking
            // wasn't already enabled, retry with thinking mode on.
            let response = if response.text.is_empty()
                && response.tool_calls.is_empty()
                && turn_count <= 1
                && self.cfg.router.auto_max_think
                && !decision.thinking_enabled
            {
                self.emit(
                    session.session_id,
                    EventKind::RouterEscalationV1 {
                        reason_codes: vec!["empty_response_escalation".to_string()],
                    },
                )?;
                let escalated_request = ChatRequest {
                    thinking: Some(ThinkingConfig::enabled(
                        session.budgets.max_think_tokens.max(4096),
                    )),
                    temperature: None,
                    ..request
                };
                let cb = self.stream_callback.lock().ok().and_then(|g| g.clone());
                if let Some(cb) = cb {
                    self.llm.complete_chat_streaming(&escalated_request, cb)?
                } else {
                    self.llm.complete_chat(&escalated_request)?
                }
            } else {
                response
            };

            // Tool-call-as-text rescue retry: if response.tool_calls is empty
            // but the text contains known tool API names, retry once with
            // tool_choice="required" to force the model to use structured
            // function calling.
            let response = if options.tools
                && response.tool_calls.is_empty()
                && !tool_choice_retried
                && content_contains_tool_call(&response.text)
            {
                tool_choice_retried = true;
                self.observer
                    .verbose_log("tool-call-as-text detected: retrying with tool_choice=required");
                let retry_request = ChatRequest {
                    model: request_model.clone(),
                    messages: messages.clone(),
                    tools: tools.clone(),
                    tool_choice: ToolChoice::required(),
                    max_tokens: session.budgets.max_think_tokens.max(4096),
                    temperature: if thinking.is_some() { None } else { Some(0.0) },
                    thinking: thinking.clone(),
                };
                let cb = self.stream_callback.lock().ok().and_then(|g| g.clone());
                if let Some(cb) = cb {
                    self.llm.complete_chat_streaming(&retry_request, cb)?
                } else {
                    self.llm.complete_chat(&retry_request)?
                }
            } else {
                // Reset flag on successful structured responses so a later turn
                // can retry again if needed.
                if !response.tool_calls.is_empty() {
                    tool_choice_retried = false;
                }
                response
            };

            // Record usage metrics
            let input_tok = estimate_tokens(
                &messages
                    .iter()
                    .map(|m| match m {
                        ChatMessage::System { content } | ChatMessage::User { content } => {
                            content.as_str()
                        }
                        ChatMessage::Assistant { content, .. } => content.as_deref().unwrap_or(""),
                        ChatMessage::Tool { content, .. } => content.as_str(),
                    })
                    .collect::<Vec<_>>()
                    .join(""),
            );
            let output_tok =
                estimate_tokens(&response.text) + estimate_tokens(&response.reasoning_content);
            self.emit(
                session.session_id,
                EventKind::UsageUpdatedV1 {
                    unit: LlmUnit::Executor,
                    model: response_model,
                    input_tokens: input_tok,
                    output_tokens: output_tok,
                },
            )?;
            self.emit_cost_event(session.session_id, input_tok, output_tok)?;

            // If the model returned text content with no tool calls, we're done
            if response.tool_calls.is_empty() {
                let changed_files_before_completion =
                    failure_tracker.files_changed_since_verify.clone();
                let mut completion_verified = false;
                // ── Plan discipline: verification gate ──
                // Before returning, check if we need to run verification.
                if plan_state.status == PlanStatus::Executing {
                    plan_state.enter_verification();
                }
                if plan_state.status == PlanStatus::Verifying {
                    if plan_state.verification_is_noop() {
                        // No verify commands — treat as pass
                        plan_state.verification_passed();
                        failure_tracker.record_verify_pass();
                        completion_verified = true;
                    } else {
                        match self.run_verification_with_output(
                            &plan_state.verify_commands,
                            &mut failure_tracker,
                            session.session_id,
                        ) {
                            Ok(()) => {
                                self.observer
                                    .verbose_log("plan_discipline: verification passed");
                                plan_state.verification_passed();
                                failure_tracker.record_verify_pass();
                                completion_verified = true;
                            }
                            Err(error_msg) => {
                                plan_state.verification_failed(&error_msg);
                                self.observer.verbose_log(&format!(
                                    "plan_discipline: verification failed (streak={})",
                                    plan_state.verify_failure_streak
                                ));

                                // Consult R1 after 2+ repeated failures on the same error
                                if plan_state.verify_failure_streak >= 2 {
                                    let consultation = crate::consultation::ConsultationRequest {
                                        question: "Verification keeps failing with the same error. \
                                                   Analyze the error and suggest a targeted fix strategy."
                                            .to_string(),
                                        context: error_msg.clone(),
                                        consultation_type:
                                            crate::consultation::ConsultationType::ErrorAnalysis,
                                    };
                                    if let Ok(advice) = crate::consultation::consult_r1(
                                        self.llm.as_ref(),
                                        &self.cfg.llm.max_think_model,
                                        &consultation,
                                        None,
                                    ) {
                                        messages.push(ChatMessage::User {
                                            content: format!(
                                                "<r1-advice type=\"verification_fix\">\n{}\n</r1-advice>",
                                                advice.advice
                                            ),
                                        });
                                    }
                                }

                                // Inject failure feedback and re-enter loop
                                inject_verification_feedback(
                                    &mut messages,
                                    &error_msg,
                                    plan_state.verify_failure_streak,
                                );

                                if plan_state.verify_failure_streak >= 4 {
                                    // Give up after 4 repeated failures — fall through to return
                                    self.observer.verbose_log(
                                        "plan_discipline: giving up after 4 repeated verify failures",
                                    );
                                } else {
                                    // Re-enter loop for fixes
                                    continue;
                                }
                            }
                        }
                    }
                }
                if !completion_verified && !changed_files_before_completion.is_empty() {
                    let fallback_verify = derive_verify_commands(&self.workspace);
                    if fallback_verify.is_empty() {
                        self.observer.verbose_log(
                            "completion_gate: no verification command derived; treating as pass",
                        );
                        failure_tracker.record_verify_pass();
                    } else {
                        match self.run_verification_with_output(
                            &fallback_verify,
                            &mut failure_tracker,
                            session.session_id,
                        ) {
                            Ok(()) => {
                                self.observer
                                    .verbose_log("completion_gate: verification passed");
                                failure_tracker.record_verify_pass();
                            }
                            Err(error_msg) => {
                                self.observer
                                    .verbose_log("completion_gate: verification failed");
                                let consultation = crate::consultation::ConsultationRequest {
                                    question: "Final completion verification failed. Suggest a focused fix strategy before final answer."
                                        .to_string(),
                                    context: error_msg.clone(),
                                    consultation_type:
                                        crate::consultation::ConsultationType::ErrorAnalysis,
                                };
                                if let Ok(advice) = crate::consultation::consult_r1(
                                    self.llm.as_ref(),
                                    &self.cfg.llm.max_think_model,
                                    &consultation,
                                    None,
                                ) {
                                    messages.push(ChatMessage::User {
                                        content: format!(
                                            "<r1-advice type=\"completion_verify_fix\">\n{}\n</r1-advice>",
                                            advice.advice
                                        ),
                                    });
                                }
                                inject_verification_feedback(&mut messages, &error_msg, 1);
                                continue;
                            }
                        }
                    }
                }

                // The answer shown to the user may include reasoning_content as
                // fallback, but the message persisted for API history must NOT
                // include reasoning_content (it wastes context tokens on resume).
                let mut answer = if !response.text.is_empty() {
                    response.text.clone()
                } else if !response.reasoning_content.is_empty() {
                    response.reasoning_content.clone()
                } else {
                    "(No response generated)".to_string()
                };
                let mut advisory_added = false;
                if changed_files_before_completion.len() >= 3
                    && let Some(advice) = self.consult_r1_final_review(
                        prompt,
                        &answer,
                        &changed_files_before_completion,
                    )
                {
                    answer.push_str("\n\n[Copilot Review - Advisory]\n");
                    answer.push_str(&advice);
                    advisory_added = true;
                }

                // For chat history / session resume, only store the actual text
                // content — strip reasoning_content. The reasoning was already
                // displayed to the user via streaming.
                let history_content = if !response.text.is_empty() || advisory_added {
                    Some(answer.clone())
                } else {
                    // reasoning_content was used as the answer for display but
                    // we store a compact placeholder in the API message history.
                    Some("[reasoning-only response]".to_string())
                };

                self.emit(
                    session.session_id,
                    EventKind::TurnAddedV1 {
                        role: "assistant".to_string(),
                        content: answer.clone(),
                    },
                )?;
                self.emit(
                    session.session_id,
                    EventKind::ChatTurnV1 {
                        message: ChatMessage::Assistant {
                            content: history_content,
                            reasoning_content: None,
                            tool_calls: vec![],
                        },
                    },
                )?;

                // Fire Stop hook (can block — prompts agent to continue).
                self.tool_host.fire_stop_hooks();
                {
                    let stop_input = self.hook_input(HookEvent::Stop);
                    let hr = self.fire_hook(HookEvent::Stop, &stop_input);
                    if hr.blocked {
                        // Stop hook blocked — inject reason into conversation and continue loop.
                        let reason = hr
                            .block_reason
                            .unwrap_or_else(|| "Hook requested continuation".to_string());
                        messages.push(ChatMessage::User {
                            content: format!(
                                "<stop-hook-feedback>\n{reason}\nPlease continue.\n</stop-hook-feedback>"
                            ),
                        });
                        continue;
                    }
                }
                if let Err(e) = self.transition(&mut session, SessionState::Completed) {
                    self.observer
                        .warn_log(&format!("session: failed to transition to Completed: {e}"));
                }

                // Persist memory observation for future sessions
                if let Ok(manager) = MemoryManager::new(&self.workspace) {
                    let mut patterns = vec![format!(
                        "turns={turn_count} failure_streak={failure_streak}"
                    )];
                    if failure_streak > 0 {
                        patterns.push(
                            "some tool failures occurred but task completed successfully"
                                .to_string(),
                        );
                    }
                    if let Err(e) = manager.append_auto_memory_observation(AutoMemoryObservation {
                        objective: prompt.to_string(),
                        summary: format!(
                            "chat completed in {} turns, failure_streak={}",
                            turn_count, failure_streak
                        ),
                        success: true,
                        patterns,
                        recorded_at: None,
                    }) {
                        self.observer
                            .warn_log(&format!("memory: failed to persist observation: {e}"));
                    }
                }

                // Fire SessionEnd hook.
                {
                    let mut input = self.hook_input(HookEvent::SessionEnd);
                    input.session_type = Some("exit".to_string());
                    self.fire_hook(HookEvent::SessionEnd, &input);
                }

                return Ok(answer);
            }

            // The model wants to call tools — add the assistant message with tool_calls
            if response.text.is_empty() {
                empty_response_count += 1;
            }
            // Include reasoning_content in the in-memory history for the current
            // turn (will be stripped before the next API call above).
            let rc = if response.reasoning_content.is_empty() {
                None
            } else {
                Some(response.reasoning_content.clone())
            };
            messages.push(ChatMessage::Assistant {
                content: if response.text.is_empty() {
                    None
                } else {
                    Some(response.text.clone())
                },
                reasoning_content: rc,
                tool_calls: response.tool_calls.clone(),
            });

            if !response.text.is_empty() {
                self.emit(
                    session.session_id,
                    EventKind::TurnAddedV1 {
                        role: "assistant".to_string(),
                        content: response.text.clone(),
                    },
                )?;
            }
            // Persist full assistant message with tool_calls for resume.
            // Strip reasoning_content from persisted events — no need to store it.
            self.emit(
                session.session_id,
                EventKind::ChatTurnV1 {
                    message: ChatMessage::Assistant {
                        content: if response.text.is_empty() {
                            None
                        } else {
                            Some(response.text.clone())
                        },
                        reasoning_content: None,
                        tool_calls: response.tool_calls.clone(),
                    },
                },
            )?;

            // Execute each tool call and collect results
            let mode_obs_step = u32::try_from(turn_count).unwrap_or(u32::MAX);
            let mode_obs_repo = self.build_repo_facts();
            let mut mode_observation_action: Option<ActionRecord> = None;
            let mut mode_observation_stderr: Option<String> = None;
            for tc in &response.tool_calls {
                let internal_name = map_tool_name(&tc.name);
                let mut args: serde_json::Value =
                    serde_json::from_str(&tc.arguments).unwrap_or_else(|_| json!({}));
                normalize_tool_args_with_workspace(internal_name, &mut args, &self.workspace);

                // ── Pre-execution argument validation ──
                if let Err(validation_error) = validate_tool_args(internal_name, &args) {
                    let hint = format!(
                        "Invalid arguments for `{internal_name}`: {validation_error}\n\
                         Fix the arguments and try again."
                    );
                    self.observer.verbose_log(&format!(
                        "validation rejected {}: {validation_error}",
                        internal_name
                    ));
                    messages.push(ChatMessage::Tool {
                        tool_call_id: tc.id.clone(),
                        content: hint,
                    });
                    failure_streak += 1;
                    failure_tracker.record_failure();
                    let action = build_observation_action(
                        internal_name,
                        &args,
                        false,
                        None,
                        "validation_error",
                    );
                    mode_observation_action = Some(action);
                    mode_observation_stderr = Some(validation_error.clone());
                    continue;
                }

                // Safety net: block disallowed tools at execution time
                if let Some(ref deny_list) = options.disallowed_tools
                    && deny_list.iter().any(|d| d == &tc.name)
                {
                    self.observer
                        .verbose_log(&format!("blocked disallowed tool: {}", tc.name));
                    messages.push(ChatMessage::Tool {
                        tool_call_id: tc.id.clone(),
                        content: format!(
                            "Error: tool '{}' is not allowed in this session",
                            tc.name
                        ),
                    });
                    continue;
                }

                // ── MCP tool calls (routed to MCP servers, with permission check) ──
                if Self::is_mcp_tool(&tc.name) {
                    // Build a ToolCall for the permission engine
                    let mcp_tool_call = ToolCall {
                        name: tc.name.clone(),
                        args: args.clone(),
                        requires_approval: true,
                    };
                    if self.policy.requires_approval(&mcp_tool_call)
                        && !self.request_tool_approval(&mcp_tool_call).unwrap_or(false)
                    {
                        messages.push(ChatMessage::Tool {
                            tool_call_id: tc.id.clone(),
                            content: format!(
                                "Error: MCP tool '{}' was denied by permission policy",
                                tc.name
                            ),
                        });
                        continue;
                    }

                    if let Ok(cb) = self.stream_callback.lock()
                        && let Some(ref cb) = *cb
                    {
                        cb(StreamChunk::ToolCallStart {
                            tool_name: tc.name.clone(),
                            args_summary: summarize_tool_args(&tc.name, &args),
                        });
                    }
                    let tool_start = Instant::now();
                    let result_text = match self.execute_mcp_tool(&tc.name, &args) {
                        Ok(output) => output,
                        Err(e) => format!("MCP tool error: {e}"),
                    };
                    let elapsed = tool_start.elapsed();
                    let mcp_success = !result_text.starts_with("MCP tool error:");
                    if let Ok(cb) = self.stream_callback.lock()
                        && let Some(ref cb) = *cb
                    {
                        cb(StreamChunk::ToolCallEnd {
                            tool_name: tc.name.clone(),
                            duration_ms: elapsed.as_millis() as u64,
                            success: mcp_success,
                            summary: if mcp_success {
                                "done".to_string()
                            } else {
                                "error".to_string()
                            },
                        });
                    }
                    let result_text = truncate_tool_output(&tc.name, &result_text, 30000);
                    messages.push(ChatMessage::Tool {
                        tool_call_id: tc.id.clone(),
                        content: result_text,
                    });
                    let mcp_action = build_observation_action(
                        internal_name,
                        &args,
                        mcp_success,
                        None,
                        "mcp_tool",
                    );
                    if !mcp_success {
                        failure_streak += 1;
                        failure_tracker.record_failure();
                        mode_observation_stderr = Some("mcp tool failure".to_string());
                    } else if mode_observation_action.is_none() {
                        mode_observation_action = Some(mcp_action.clone());
                    }
                    if !mcp_success {
                        mode_observation_action = Some(mcp_action);
                    }
                    continue;
                }

                // ── mcp_search meta-tool (lazy loading) ──
                if internal_name == "mcp_search" {
                    let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                    let matches = self.search_mcp_tools(query);
                    let result = if matches.is_empty() {
                        format!("No MCP tools found matching '{query}'")
                    } else {
                        let tool_list: Vec<String> = matches
                            .iter()
                            .map(|t| {
                                format!("- mcp__{}_{}: {}", t.server_id, t.name, t.description)
                            })
                            .collect();
                        format!(
                            "Found {} MCP tools matching '{query}':\n{}",
                            matches.len(),
                            tool_list.join("\n")
                        )
                    };
                    messages.push(ChatMessage::Tool {
                        tool_call_id: tc.id.clone(),
                        content: result,
                    });
                    continue;
                }

                // ── tool_search meta-tool (discover extended tools) ──
                if internal_name == "tool_search" {
                    let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                    let matches = deepseek_tools::search_extended_tools(query, &extended_tool_defs);
                    // Dynamically add discovered tools to the active set for remaining turns
                    for matched in &matches {
                        if !tools
                            .iter()
                            .any(|t| t.function.name == matched.function.name)
                        {
                            tools.push(matched.clone());
                        }
                    }
                    let result = deepseek_tools::format_tool_search_results(&matches);
                    messages.push(ChatMessage::Tool {
                        tool_call_id: tc.id.clone(),
                        content: result,
                    });
                    continue;
                }

                // ── Agent-level tools (handled here, not by LocalToolHost) ──
                if AGENT_LEVEL_TOOLS.contains(&internal_name) {
                    // Handle plan mode transitions inline (needs access to loop state)
                    if internal_name == "enter_plan_mode" && !plan_mode_active {
                        plan_mode_active = true;
                        tools = all_tools
                            .iter()
                            .filter(|t| PLAN_MODE_TOOLS.contains(&t.function.name.as_str()))
                            .cloned()
                            .collect();
                        self.emit(
                            session.session_id,
                            EventKind::EnterPlanModeV1 {
                                session_id: session.session_id,
                            },
                        )?;
                        if let Ok(cb) = self.stream_callback.lock()
                            && let Some(ref cb) = *cb
                        {
                            cb(StreamChunk::ContentDelta(
                                "\n[plan mode] Entered plan mode — read-only tools only. Explore the codebase and design your approach.\n".to_string(),
                            ));
                        }
                        messages.push(ChatMessage::Tool {
                            tool_call_id: tc.id.clone(),
                            content: "Plan mode activated. You now have read-only tools only (Read, Glob, Grep, search, git status/diff). Explore the codebase, design your implementation plan, then call exit_plan_mode when ready for user approval.".to_string(),
                        });
                        continue;
                    } else if internal_name == "exit_plan_mode" && plan_mode_active {
                        plan_mode_active = false;
                        tools = all_tools.clone();
                        self.emit(
                            session.session_id,
                            EventKind::ExitPlanModeV1 {
                                session_id: session.session_id,
                            },
                        )?;
                        if let Ok(cb) = self.stream_callback.lock()
                            && let Some(ref cb) = *cb
                        {
                            cb(StreamChunk::ContentDelta(
                                "\n[plan mode] Exited plan mode — all tools now available.\n"
                                    .to_string(),
                            ));
                        }
                        messages.push(ChatMessage::Tool {
                            tool_call_id: tc.id.clone(),
                            content: "Plan mode deactivated. All tools are now available for execution. Proceed with implementing your plan.".to_string(),
                        });
                        continue;
                    }

                    let tool_result = self.execute_agent_tool(internal_name, &args, &session)?;
                    let tool_result = truncate_tool_output(internal_name, &tool_result, 30000);
                    messages.push(ChatMessage::Tool {
                        tool_call_id: tc.id.clone(),
                        content: tool_result.clone(),
                    });
                    self.emit(
                        session.session_id,
                        EventKind::TurnAddedV1 {
                            role: "tool".to_string(),
                            content: format!(
                                "[{}] ok: {}",
                                internal_name,
                                if tool_result.len() > 200 {
                                    format!("{}...", &tool_result[..200])
                                } else {
                                    tool_result
                                }
                            ),
                        },
                    )?;
                    continue;
                }

                // ── PreToolUse hook (can block/modify tool input) ──
                {
                    let mut input = self.hook_input(HookEvent::PreToolUse);
                    input.tool_name = Some(tc.name.clone());
                    input.tool_input = Some(args.clone());
                    let hr = self.fire_hook(HookEvent::PreToolUse, &input);
                    if hr.blocked {
                        let reason = hr.block_reason.as_deref().unwrap_or("Blocked by hook");
                        messages.push(ChatMessage::Tool {
                            tool_call_id: tc.id.clone(),
                            content: format!("Error: tool blocked by hook — {reason}"),
                        });
                        Self::inject_hook_context(&mut messages, &hr);
                        continue;
                    }
                    // Apply updated input if the hook modified tool parameters.
                    if let Some(ref updated) = hr.updated_input {
                        args = updated.clone();
                        normalize_tool_args_with_workspace(
                            internal_name,
                            &mut args,
                            &self.workspace,
                        );
                        if let Err(validation_error) = validate_tool_args(internal_name, &args) {
                            let hint = format!(
                                "Invalid arguments for `{internal_name}` after hook update: {validation_error}\n\
                                 Fix the arguments and try again."
                            );
                            messages.push(ChatMessage::Tool {
                                tool_call_id: tc.id.clone(),
                                content: hint,
                            });
                            failure_streak += 1;
                            failure_tracker.record_failure();
                            let action = build_observation_action(
                                internal_name,
                                &args,
                                false,
                                None,
                                "validation_error_after_hook",
                            );
                            mode_observation_action = Some(action);
                            mode_observation_stderr = Some(validation_error.clone());
                            continue;
                        }
                    }
                    Self::inject_hook_context(&mut messages, &hr);
                }

                let tool_call = ToolCall {
                    name: internal_name.to_string(),
                    args: args.clone(),
                    requires_approval: false,
                };

                // Emit tool proposal event
                let proposal = self.tool_host.propose(tool_call.clone());
                self.emit(
                    session.session_id,
                    EventKind::ToolProposedV1 {
                        proposal: proposal.clone(),
                    },
                )?;

                // Check approval
                let approved = proposal.approved
                    || self.request_tool_approval(&proposal.call).unwrap_or(false);

                let tool_arg_summary = summarize_tool_args(internal_name, &args);
                let tool_result = if approved {
                    self.emit(
                        session.session_id,
                        EventKind::ToolApprovedV1 {
                            invocation_id: proposal.invocation_id,
                        },
                    )?;

                    // Notify TUI that a tool is being executed
                    if let Ok(mut cb_guard) = self.stream_callback.lock()
                        && let Some(ref mut cb) = *cb_guard
                    {
                        cb(StreamChunk::ToolCallStart {
                            tool_name: internal_name.to_string(),
                            args_summary: tool_arg_summary.clone(),
                        });
                    }

                    // Intercept bash.run with run_in_background: true
                    if internal_name == "bash.run"
                        && args
                            .get("run_in_background")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false)
                    {
                        let shell_id = Uuid::now_v7();
                        let cmd_str = args
                            .get("cmd")
                            .or_else(|| args.get("command"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let host = Arc::clone(&self.tool_host);
                        let approved_call = ApprovedToolCall {
                            invocation_id: proposal.invocation_id,
                            call: proposal.call,
                        };
                        let handle = thread::spawn(move || host.execute(approved_call));

                        if let Ok(mut shells) = self.background_shells.lock() {
                            shells.push_back(BackgroundShell {
                                shell_id,
                                cmd: cmd_str.clone(),
                                handle: Some(handle),
                                result: None,
                                stopped: false,
                            });
                        }

                        messages.push(ChatMessage::Tool {
                            tool_call_id: tc.id.clone(),
                            content: serde_json::to_string(&json!({
                                "shell_id": shell_id.to_string(),
                                "status": "running",
                                "cmd": cmd_str,
                                "message": "Command started in background. Use task_output with this shell_id to check results, or kill_shell to stop it."
                            }))?,
                        });
                        self.emit(
                            session.session_id,
                            EventKind::BackgroundJobStartedV1 {
                                job_id: shell_id,
                                kind: "bash".to_string(),
                                reference: cmd_str,
                            },
                        )?;
                        continue;
                    }

                    // Checkpoint: snapshot files before modification
                    self.maybe_checkpoint(session.session_id, internal_name.as_ref(), &args);

                    let tool_start = Instant::now();
                    let result = self.tool_host.execute(ApprovedToolCall {
                        invocation_id: proposal.invocation_id,
                        call: proposal.call,
                    });
                    let tool_elapsed = tool_start.elapsed();

                    self.emit(
                        session.session_id,
                        EventKind::ToolResultV1 {
                            result: result.clone(),
                        },
                    )?;
                    if (internal_name.starts_with("chrome.")
                        || internal_name.starts_with("chrome_"))
                        && !result.success
                    {
                        let error_kind = result
                            .output
                            .get("error_kind")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        let live_connect_failure = matches!(
                            error_kind,
                            "chrome_unavailable" | "chrome_session_not_connected"
                        );
                        self.emit(
                            session.session_id,
                            EventKind::TelemetryEventV1 {
                                name: "kpi.chrome.tool_failure".to_string(),
                                properties: json!({
                                    "tool": internal_name,
                                    "error_kind": error_kind,
                                    "live_connect_failure": live_connect_failure,
                                }),
                            },
                        )?;
                    }

                    // Notify TUI of completion with result preview
                    if let Ok(mut cb_guard) = self.stream_callback.lock()
                        && let Some(ref mut cb) = *cb_guard
                    {
                        let output_str = result.output.to_string();
                        let preview = if output_str.len() > 200 {
                            format!("{}...", &output_str[..output_str.floor_char_boundary(200)])
                        } else {
                            output_str
                        };
                        let preview_line = preview
                            .lines()
                            .next()
                            .unwrap_or("")
                            .chars()
                            .take(120)
                            .collect::<String>();
                        cb(StreamChunk::ToolCallEnd {
                            tool_name: internal_name.to_string(),
                            duration_ms: tool_elapsed.as_millis() as u64,
                            success: result.success,
                            summary: preview_line,
                        });
                        // Emit inline image if tool result contains image data
                        if (internal_name == "fs.read" || internal_name == "fs_read")
                            && result
                                .output
                                .get("mime")
                                .and_then(|v| v.as_str())
                                .is_some_and(|m| m.starts_with("image/"))
                            && let Some(b64) = result.output.get("base64").and_then(|v| v.as_str())
                        {
                            use base64::Engine;
                            let engine = base64::engine::general_purpose::STANDARD;
                            if let Ok(data) = engine.decode(b64) {
                                let label = result
                                    .output
                                    .get("path")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("image")
                                    .to_string();
                                cb(StreamChunk::ImageData { data, label });
                            }
                        }
                    }

                    // Doom-loop tracking: record tool signature
                    let exit_code = result
                        .output
                        .get("exit_code")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32);
                    let output_for_observation = result.output.to_string();
                    let action = build_observation_action(
                        internal_name,
                        &args,
                        result.success,
                        exit_code,
                        &output_for_observation,
                    );
                    if !result.success {
                        mode_observation_action = Some(action.clone());
                        mode_observation_stderr = Some(output_for_observation.clone());
                        record_error_modules_from_output(
                            &mut failure_tracker,
                            &output_for_observation,
                        );
                    } else if mode_observation_action.is_none() {
                        mode_observation_action = Some(action.clone());
                    }
                    let sig = ToolSignature::new(internal_name, &args, exit_code);
                    failure_tracker.record_tool_signature(sig, result.success);

                    if result.success {
                        failure_streak = 0;
                        failure_tracker.record_success();
                        // Track file changes for blast radius + plan discipline
                        if matches!(
                            deepseek_core::ToolName::from_internal_name(internal_name),
                            Some(
                                deepseek_core::ToolName::FsWrite
                                    | deepseek_core::ToolName::FsEdit
                                    | deepseek_core::ToolName::MultiEdit
                                    | deepseek_core::ToolName::PatchApply
                            )
                        ) {
                            if let Some(path) = args
                                .get("path")
                                .or_else(|| args.get("file_path"))
                                .and_then(|v| v.as_str())
                            {
                                failure_tracker.record_file_change(path);
                                plan_state.record_file_touched(path);
                            }
                            // multi_edit: track each file in the canonical files array.
                            if internal_name == "multi_edit"
                                && let Some(files) = args.get("files").and_then(|v| v.as_array())
                            {
                                for file in files {
                                    if let Some(p) = file
                                        .get("path")
                                        .or_else(|| file.get("file_path"))
                                        .and_then(|v| v.as_str())
                                    {
                                        plan_state.record_file_touched(p);
                                        failure_tracker.record_file_change(p);
                                    }
                                }
                            }
                            plan_state.maybe_advance_step();
                        }
                    } else {
                        failure_streak += 1;
                        failure_tracker.record_failure();
                    }

                    // ── PostToolUse / PostToolUseFailure hook ──
                    {
                        let hook_event = if result.success {
                            HookEvent::PostToolUse
                        } else {
                            HookEvent::PostToolUseFailure
                        };
                        let mut post_input = self.hook_input(hook_event);
                        post_input.tool_name = Some(tc.name.clone());
                        post_input.tool_input = Some(args.clone());
                        post_input.tool_result =
                            Some(serde_json::Value::String(result.output.to_string()));
                        let hr = self.fire_hook(hook_event, &post_input);
                        Self::inject_hook_context(&mut messages, &hr);
                    }

                    let mut output =
                        truncate_tool_output(internal_name, &result.output.to_string(), 30000);
                    // Append actionable hint for failed tools
                    if !result.success
                        && let Some(hint) = tool_error_hint(internal_name, &output)
                    {
                        output.push_str("\n\n");
                        output.push_str(&hint);
                    }
                    output
                } else {
                    failure_streak += 1;
                    failure_tracker.record_failure();
                    // Track denied tools in doom-loop detection so repeated
                    // denials don't cause infinite retries.
                    let sig = ToolSignature::new(internal_name, &args, Some(-2));
                    failure_tracker.record_tool_signature(sig, false);
                    // Notify TUI of denial
                    if let Ok(mut cb_guard) = self.stream_callback.lock()
                        && let Some(ref mut cb) = *cb_guard
                    {
                        cb(StreamChunk::ToolCallEnd {
                            tool_name: internal_name.to_string(),
                            duration_ms: 0,
                            success: false,
                            summary: "denied (requires approval)".to_string(),
                        });
                    }
                    format!(
                        "Tool '{}' was denied by the user. Do NOT retry this tool with the same arguments. Try a different approach or ask the user for guidance.",
                        internal_name
                    )
                };
                if !approved {
                    let action = build_observation_action(
                        internal_name,
                        &args,
                        false,
                        Some(-2),
                        "tool denied by user",
                    );
                    mode_observation_action = Some(action);
                    mode_observation_stderr = Some("tool denied by user".to_string());
                }

                // Add tool result message to conversation
                let tool_msg = ChatMessage::Tool {
                    tool_call_id: tc.id.clone(),
                    content: tool_result.clone(),
                };
                messages.push(tool_msg.clone());

                // Emit tool interaction as transcript entry for session resume
                let status = if approved { "ok" } else { "denied" };
                let result_preview = if tool_result.len() > 200 {
                    format!("{}...", &tool_result[..200])
                } else {
                    tool_result
                };
                self.emit(
                    session.session_id,
                    EventKind::TurnAddedV1 {
                        role: "tool".to_string(),
                        content: format!("[{}] {}: {}", internal_name, status, result_preview),
                    },
                )?;
                self.emit(
                    session.session_id,
                    EventKind::ChatTurnV1 { message: tool_msg },
                )?;
            }
            let mode_observation: Option<ObservationPack> = mode_observation_action.map(|action| {
                let mut builder = ObservationPackBuilder::new(mode_obs_step, mode_obs_repo.clone());
                builder.add_action(action);
                if let Some(stderr) = mode_observation_stderr.as_deref() {
                    builder.set_stderr(stderr);
                }
                builder.set_changed_files(failure_tracker.files_changed_since_verify.clone());
                builder.build()
            });

            // ── Compress repeated tool call patterns ──
            // If the same tool was called 3+ times in a row (across the most
            // recent assistant+tool message pairs), replace the older results
            // with a compact summary to save context tokens.
            compress_repeated_tool_results(&mut messages);

            // ── Doom-loop breaker ──
            // If the doom-loop tracker shows repeated failures on the same tool(s),
            // inject per-tool guidance and reset the tracker instead of escalating
            // to R1 (which faces the same restrictions).
            if failure_tracker.has_doom_loop(mode_router_config.doom_loop_threshold) {
                let failing_tools: Vec<String> = failure_tracker
                    .doom_loop_sigs
                    .keys()
                    .map(|sig| sig.tool.clone())
                    .collect();
                let guidance = doom_loop_guidance(&failing_tools);
                if !guidance.is_empty() {
                    self.observer.verbose_log(&format!(
                        "doom-loop detected for tools {:?}: injecting guidance",
                        failing_tools
                    ));
                    messages.push(ChatMessage::User { content: guidance });
                    // Reset doom-loop state so we don't immediately escalate to R1
                    failure_tracker.doom_loop_sigs.clear();
                    failure_tracker.consecutive_step_failures = 0;
                    failure_streak = 0;
                }
            }

            // ── Mode router: check for R1DriveTools escalation ──
            let mode_decision = decide_mode(
                &mode_router_config,
                current_mode,
                &failure_tracker,
                mode_observation.as_ref(),
            );
            let mut next_mode = mode_decision.mode;
            let mut next_reason = mode_decision.reason.clone();
            let allow_r1_drive_tools =
                mode_router_config.r1_drive_auto_escalation || options.allow_r1_drive_tools;
            if next_mode == AgentMode::R1DriveTools
                && current_mode != AgentMode::R1DriveTools
                && !allow_r1_drive_tools
            {
                let suppressed_reason = next_reason
                    .as_ref()
                    .map(std::string::ToString::to_string)
                    .unwrap_or_else(|| "unspecified".to_string());
                self.observer.verbose_log(&format!(
                    "turn {turn_count}: suppressing R1DriveTools escalation ({suppressed_reason}); break-glass disabled"
                ));
                self.emit(
                    session.session_id,
                    EventKind::RouterEscalationV1 {
                        reason_codes: vec![
                            "r1_drive_breakglass_disabled".to_string(),
                            format!("suppressed_{suppressed_reason}"),
                        ],
                    },
                )?;
                if let Some(advice) = self.consult_r1_on_suppressed_escalation(
                    &suppressed_reason,
                    prompt,
                    mode_observation.as_ref(),
                ) {
                    messages.push(ChatMessage::User {
                        content: format!(
                            "<r1-checkpoint reason=\"{suppressed_reason}\">\n{advice}\n</r1-checkpoint>"
                        ),
                    });
                }
                next_mode = current_mode;
                next_reason = None;
            }
            if next_mode != current_mode {
                if let Some(ref reason) = next_reason {
                    self.observer.verbose_log(&format!(
                        "turn {turn_count}: mode transition {} -> {} (reason: {reason})",
                        current_mode, next_mode
                    ));
                    self.emit(
                        session.session_id,
                        EventKind::RouterEscalationV1 {
                            reason_codes: vec![reason.to_string()],
                        },
                    )?;
                    // Notify TUI of mode transition
                    if let Ok(mut cb_guard) = self.stream_callback.lock()
                        && let Some(ref mut cb) = *cb_guard
                    {
                        cb(StreamChunk::ModeTransition {
                            from: current_mode.to_string(),
                            to: next_mode.to_string(),
                            reason: reason.to_string(),
                        });
                    }
                }
                current_mode = next_mode;
            }

            // ── R1DriveTools dispatch ──
            // When mode router escalates to R1, hand control to the R1 drive-tools
            // loop. R1 drives tool execution via JSON intents until it returns
            // done/abort/delegate_patch or its budget is exhausted.
            if current_mode == AgentMode::R1DriveTools {
                failure_tracker.entered_r1();

                // Build context for R1 from recent conversation
                let initial_context = self.build_r1_initial_context(&messages, prompt);
                let repo_facts = self.build_repo_facts();

                let r1_config = R1DriveConfig::from_mode_router_config(
                    &mode_router_config,
                    &self.cfg.llm.max_think_model,
                );

                let stream_cb: Option<deepseek_core::StreamCallback> = if let Ok(cb_guard) =
                    self.stream_callback.lock()
                    && cb_guard.is_some()
                {
                    // Create a proxy callback. We can't borrow the mutex across
                    // the r1_drive_loop call, so we just pass None for now and
                    // let r1_drive use its own verbose logging.
                    None
                } else {
                    None
                };

                let tool_host_dyn: Arc<dyn deepseek_core::ToolHost + Send + Sync> =
                    self.tool_host.clone();

                let outcome = r1_drive_loop(
                    &r1_config,
                    self.llm.as_ref(),
                    &tool_host_dyn,
                    &self.observer,
                    &self.workspace,
                    &repo_facts,
                    &mut failure_tracker,
                    prompt,
                    &initial_context,
                    stream_cb.as_ref(),
                );

                match outcome {
                    R1DriveOutcome::Done(done) => {
                        // R1 completed the task. Return answer.
                        self.observer
                            .verbose_log(&format!("R1 drive-tools completed: {}", &done.summary));
                        if let Ok(mut cb_guard) = self.stream_callback.lock()
                            && let Some(ref mut cb) = *cb_guard
                        {
                            cb(StreamChunk::ContentDelta(format!(
                                "\n--- R1 completed: {} ---\n",
                                &done.summary
                            )));
                        }
                        // Emit final turn
                        let answer = if done.verification.is_empty() {
                            done.summary.clone()
                        } else {
                            format!("{}\n\nVerification: {}", done.summary, done.verification)
                        };
                        messages.push(ChatMessage::Assistant {
                            content: Some(answer.clone()),
                            reasoning_content: None,
                            tool_calls: vec![],
                        });
                        self.emit(
                            session.session_id,
                            EventKind::TurnAddedV1 {
                                role: "assistant".to_string(),
                                content: answer.clone(),
                            },
                        )?;
                        if let Err(e) = self.transition(&mut session, SessionState::Completed) {
                            self.observer.warn_log(&format!(
                                "session: failed to transition to Completed: {e}"
                            ));
                        }
                        return Ok(answer);
                    }

                    R1DriveOutcome::DelegatePatch(dp) => {
                        // R1 wants V3 to write a patch. Call V3 in patch-only mode.
                        self.observer
                            .verbose_log(&format!("R1 delegating patch: {}", &dp.task));
                        if let Ok(mut cb_guard) = self.stream_callback.lock()
                            && let Some(ref mut cb) = *cb_guard
                        {
                            cb(StreamChunk::ContentDelta(format!(
                                "\n--- V3 writing patch: {} ---\n",
                                &dp.task
                            )));
                        }

                        let workspace = self.workspace.clone();
                        let file_reader: FileReader = Box::new(move |path: &str| {
                            let full = workspace.join(path);
                            std::fs::read_to_string(&full).ok()
                        });
                        let repo_facts_patch = self.build_repo_facts();
                        let patch_config = V3PatchConfig {
                            model: self.cfg.llm.base_model.clone(),
                            max_tokens: 8192,
                            enable_thinking: self.cfg.router.unified_thinking_tools,
                            max_think_tokens: session.budgets.max_think_tokens.max(4096),
                            max_context_requests: mode_router_config.v3_patch_max_context_requests,
                            max_retries: 2,
                        };

                        let patch_outcome = v3_patch_write(
                            &patch_config,
                            self.llm.as_ref(),
                            &self.observer,
                            &dp,
                            &repo_facts_patch,
                            &file_reader,
                            stream_cb.as_ref(),
                        );

                        match patch_outcome {
                            V3PatchOutcome::Diff(diff_text) => {
                                // Apply the patch via tool host
                                self.observer.verbose_log(&format!(
                                    "V3 produced diff ({} bytes), applying...",
                                    diff_text.len()
                                ));
                                let patch_call = deepseek_core::ToolCall {
                                    name: "patch.apply".to_string(),
                                    args: serde_json::json!({"patch": diff_text}),
                                    requires_approval: true,
                                };
                                let proposal = self.tool_host.propose(patch_call);
                                if proposal.approved {
                                    let patch_result = self.tool_host.execute(ApprovedToolCall {
                                        invocation_id: proposal.invocation_id,
                                        call: proposal.call,
                                    });
                                    if patch_result.success {
                                        self.observer.verbose_log("Patch applied successfully");
                                        // Run verification if acceptance criteria given
                                        if !dp.acceptance.is_empty() {
                                            let verify_ok = self.run_verification(
                                                &dp.acceptance,
                                                &mut failure_tracker,
                                                session.session_id,
                                            );
                                            if verify_ok {
                                                failure_tracker.record_verify_pass();
                                            }
                                        } else {
                                            failure_tracker.record_verify_pass();
                                        }
                                    } else {
                                        self.observer.verbose_log("Patch apply failed");
                                        failure_tracker.record_failure();
                                    }
                                } else {
                                    self.observer.verbose_log("Patch apply denied by policy");
                                    failure_tracker.record_failure();
                                }
                            }
                            V3PatchOutcome::Failed { reason } => {
                                self.observer
                                    .verbose_log(&format!("V3 patch failed: {reason}"));
                                failure_tracker.record_failure();
                            }
                        }

                        // After patch handling, re-check mode. If verify passed,
                        // hysteresis will allow return to V3. Otherwise stay in R1.
                        let post_patch =
                            decide_mode(&mode_router_config, current_mode, &failure_tracker, None);
                        current_mode = post_patch.mode;
                        // Continue the main loop (next iteration will either
                        // run V3 or re-enter R1 based on mode)
                        continue;
                    }

                    R1DriveOutcome::Abort(reason) => {
                        self.observer.verbose_log(&format!("R1 aborted: {reason}"));
                        if let Ok(mut cb_guard) = self.stream_callback.lock()
                            && let Some(ref mut cb) = *cb_guard
                        {
                            cb(StreamChunk::ContentDelta(format!(
                                "\n--- R1 aborted: {reason} ---\n"
                            )));
                        }
                        // Fall back to V3 and inject context about the abort
                        current_mode = AgentMode::V3Autopilot;
                        messages.push(ChatMessage::User {
                            content: format!(
                                "[System] R1 analysis aborted: {reason}. \
                                 Continue with V3 to complete the task."
                            ),
                        });
                        continue;
                    }

                    R1DriveOutcome::BudgetExhausted { steps_used } => {
                        self.observer
                            .verbose_log(&format!("R1 budget exhausted after {steps_used} steps"));
                        current_mode = AgentMode::V3Autopilot;
                        messages.push(ChatMessage::User {
                            content: format!(
                                "[System] R1 budget exhausted ({steps_used} steps). \
                                 Continuing with V3 to complete remaining work."
                            ),
                        });
                        continue;
                    }

                    R1DriveOutcome::ParseFailure { last_error } => {
                        self.observer
                            .verbose_log(&format!("R1 parse failure: {last_error}"));
                        current_mode = AgentMode::V3Autopilot;
                        messages.push(ChatMessage::User {
                            content: format!(
                                "[System] R1 protocol error: {last_error}. \
                                 Continuing with V3."
                            ),
                        });
                        continue;
                    }
                }
            }
        }

}
