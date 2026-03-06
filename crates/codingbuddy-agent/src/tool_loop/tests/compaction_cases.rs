use super::*;

// ── Batch 4 (P9): Compaction & Correctness ──

#[test]
fn compaction_preserves_tool_result_pairing() {
    // Build a conversation with tool calls that must stay paired.
    let llm = ScriptedLlm::new(vec![make_text_response("Final answer.")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let config = ToolLoopConfig {
        // Small context window so compaction budget is tight
        context_window_tokens: 2000,
        ..Default::default()
    };
    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    // Manually build history to simulate a conversation with tool exchanges
    loop_.messages.push(ChatMessage::User {
        content: "first question".to_string(),
    });
    for i in 0..5 {
        loop_.messages.push(ChatMessage::Assistant {
            content: None,
            reasoning_content: None,
            tool_calls: vec![LlmToolCall {
                id: format!("call_{i}"),
                name: "fs_read".to_string(),
                arguments: format!(r#"{{"path":"file{i}.rs"}}"#),
            }],
        });
        loop_.messages.push(ChatMessage::Tool {
            tool_call_id: format!("call_{i}"),
            content: format!("content of file{i}.rs — {}", "x".repeat(400)),
        });
    }
    loop_.messages.push(ChatMessage::User {
        content: "second question".to_string(),
    });
    loop_.messages.push(ChatMessage::Assistant {
        content: Some("Here is what I found.".to_string()),
        reasoning_content: None,
        tool_calls: vec![],
    });

    // Force compaction with a tight budget
    let compacted = loop_.compact_messages(200);
    assert!(compacted, "should have compacted");

    // Verify no orphaned Tool messages (every Tool must follow an Assistant with tool_calls)
    let msgs = &loop_.messages;
    for (i, msg) in msgs.iter().enumerate() {
        if matches!(msg, ChatMessage::Tool { .. }) {
            assert!(
                i > 0
                    && matches!(&msgs[i - 1], ChatMessage::Assistant { tool_calls, .. } if !tool_calls.is_empty()),
                "Tool message at index {i} is orphaned (no preceding Assistant with tool_calls)"
            );
        }
    }
}

#[test]
fn compaction_respects_target_tokens() {
    let llm = ScriptedLlm::new(vec![]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let config = ToolLoopConfig {
        context_window_tokens: 128_000,
        ..Default::default()
    };
    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    // Add many user/assistant exchanges
    for i in 0..20 {
        loop_.messages.push(ChatMessage::User {
            content: format!("Question {i}: {}", "x".repeat(200)),
        });
        loop_.messages.push(ChatMessage::Assistant {
            content: Some(format!("Answer {i}: {}", "y".repeat(200))),
            reasoning_content: None,
            tool_calls: vec![],
        });
    }

    let before_len = loop_.messages.len();
    let compacted = loop_.compact_messages(2000);
    assert!(compacted, "should compact");
    assert!(
        loop_.messages.len() < before_len,
        "should have fewer messages"
    );
    // System message should always be first
    assert!(matches!(&loop_.messages[0], ChatMessage::System { .. }));
    // Compaction summary should be second
    assert!(
        matches!(&loop_.messages[1], ChatMessage::User { content } if content.contains("compacted"))
    );
}

#[test]
fn compaction_budget_is_provider_and_model_aware() {
    let llm = ScriptedLlm::new(vec![]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let mut deepseek_loop = ToolUseLoop::new(
        &llm,
        tool_host.clone(),
        ToolLoopConfig::default(),
        "system".to_string(),
        default_tools(),
    );
    deepseek_loop.config.provider_kind = codingbuddy_core::ProviderKind::Deepseek;
    deepseek_loop.config.model = "deepseek-chat".to_string();
    deepseek_loop.config.context_window_tokens = 128_000;
    deepseek_loop.config.max_tokens = 32_768;
    deepseek_loop.config.response_budget_tokens = 8_192;
    deepseek_loop.config.reserved_overhead_tokens = 4_000;

    let mut openai_loop = ToolUseLoop::new(
        &llm,
        tool_host,
        ToolLoopConfig::default(),
        "system".to_string(),
        default_tools(),
    );
    openai_loop.config.provider_kind = codingbuddy_core::ProviderKind::OpenAiCompatible;
    openai_loop.config.model = "gpt-4o-mini".to_string();
    openai_loop.config.context_window_tokens = 128_000;
    openai_loop.config.max_tokens = 32_768;
    openai_loop.config.response_budget_tokens = 8_192;
    openai_loop.config.reserved_overhead_tokens = 4_000;

    let (deep_prune, _, _) = deepseek_loop.compaction_budget_for_next_turn(2_000);
    let (openai_prune, _, _) = openai_loop.compaction_budget_for_next_turn(2_000);
    assert!(
        deep_prune < openai_prune,
        "deepseek thinking budget should reserve more output than openai-compatible chat"
    );
}

#[test]
fn compaction_appends_at_most_one_followup_user_prompt() {
    let llm = ScriptedLlm::new(vec![make_text_response(
        "## Goal\nPreserve context\n## Completed\nCompaction summary retained with files and findings.",
    )]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));
    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        ToolLoopConfig::default(),
        "system".to_string(),
        default_tools(),
    );

    loop_.messages.push(ChatMessage::User {
        content: "Please inspect auth, implement the fix, and run tests.".to_string(),
    });
    for i in 0..4 {
        loop_.messages.push(ChatMessage::Assistant {
            content: Some(format!("assistant-{i}: {}", "x".repeat(220))),
            reasoning_content: None,
            tool_calls: vec![],
        });
    }
    loop_.messages.push(ChatMessage::User {
        content: "Keep going with implementation details.".to_string(),
    });
    for i in 4..10 {
        loop_.messages.push(ChatMessage::Assistant {
            content: Some(format!("assistant-{i}: {}", "x".repeat(220))),
            reasoning_content: None,
            tool_calls: vec![],
        });
    }

    let compacted = loop_.compact_messages(240);
    assert!(compacted);

    let followup_count = loop_
            .messages
            .iter()
            .filter(|msg| matches!(msg, ChatMessage::User { content } if content.starts_with("Context was compacted.")))
            .count();
    assert_eq!(
        followup_count, 1,
        "compaction should emit exactly one followup user prompt"
    );
}

#[test]
fn compaction_followup_replays_last_actionable_user_prompt() {
    let llm = ScriptedLlm::new(vec![make_text_response(
        "## Goal\nContinue task\n## Completed\nHistory compacted with actionable replay anchor.",
    )]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));
    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        ToolLoopConfig::default(),
        "system".to_string(),
        default_tools(),
    );

    loop_.messages.push(ChatMessage::User {
        content: "Implement authentication middleware and run the integration tests.".to_string(),
    });
    for i in 0..4 {
        loop_.messages.push(ChatMessage::Assistant {
            content: Some(format!("step-{i}: {}", "y".repeat(180))),
            reasoning_content: None,
            tool_calls: vec![],
        });
    }
    loop_.messages.push(ChatMessage::User {
        content: "Proceed with the remaining execution steps.".to_string(),
    });
    for i in 4..12 {
        loop_.messages.push(ChatMessage::Assistant {
            content: Some(format!("step-{i}: {}", "y".repeat(180))),
            reasoning_content: None,
            tool_calls: vec![],
        });
    }

    let compacted = loop_.compact_messages(220);
    assert!(compacted);

    let replay = loop_
        .messages
        .iter()
        .rev()
        .find_map(|msg| match msg {
            ChatMessage::User { content } if content.starts_with("Context was compacted.") => {
                Some(content.clone())
            }
            _ => None,
        })
        .expect("expected followup replay user message");
    assert!(
        replay.contains("Implement authentication middleware"),
        "followup should replay the last actionable user request"
    );
}

#[test]
fn compaction_injects_active_work_state_snapshot() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Store::new(temp.path()).expect("store");

    let session_id = Uuid::now_v7();
    let plan_id = Uuid::now_v7();
    let session = Session {
        session_id,
        workspace_root: temp.path().display().to_string(),
        baseline_commit: None,
        status: SessionState::ExecutingStep,
        budgets: SessionBudgets {
            per_turn_seconds: 180,
            max_think_tokens: 32_768,
        },
        active_plan_id: Some(plan_id),
    };
    store.save_session(&session).expect("save session");
    store
        .save_plan(
            session_id,
            &Plan {
                plan_id,
                version: 1,
                goal: "Harden authentication middleware and validate with tests".to_string(),
                assumptions: vec![],
                steps: vec![
                    PlanStep {
                        step_id: Uuid::now_v7(),
                        title: "Inspect middleware entrypoints".to_string(),
                        intent: "Find all auth checks".to_string(),
                        tools: vec!["fs_read".to_string()],
                        files: vec!["src/auth/middleware.rs".to_string()],
                        done: true,
                    },
                    PlanStep {
                        step_id: Uuid::now_v7(),
                        title: "Implement missing guard".to_string(),
                        intent: "Enforce token validation".to_string(),
                        tools: vec!["fs_edit".to_string()],
                        files: vec!["src/auth/middleware.rs".to_string()],
                        done: false,
                    },
                ],
                verification: vec!["cargo test".to_string()],
                risk_notes: vec![],
            },
        )
        .expect("save plan");
    store
        .insert_task(&TaskQueueRecord {
            task_id: Uuid::now_v7(),
            session_id,
            title: "Patch auth middleware".to_string(),
            description: Some("Ensure missing token guard is enforced".to_string()),
            priority: 1,
            status: "in_progress".to_string(),
            outcome: None,
            artifact_path: Some("session://child-auth".to_string()),
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        })
        .expect("insert task");

    let llm = ScriptedLlm::new(vec![make_text_response(
        "## Goal\nKeep active plan/task state\n## Completed\nCompaction retained active work snapshot.",
    )]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));
    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        ToolLoopConfig {
            workspace: Some(temp.path().to_path_buf()),
            session_id: Some(session_id),
            initial_phase: Some(TaskPhase::Execute),
            ..Default::default()
        },
        "system".to_string(),
        default_tools(),
    );

    loop_.messages.push(ChatMessage::User {
        content: "Continue implementing the auth fix".to_string(),
    });
    for i in 0..4 {
        loop_.messages.push(ChatMessage::Assistant {
            content: Some(format!("work-{i}: {}", "z".repeat(200))),
            reasoning_content: None,
            tool_calls: vec![],
        });
    }
    loop_.messages.push(ChatMessage::User {
        content: "Carry on with the next execution step.".to_string(),
    });
    for i in 4..12 {
        loop_.messages.push(ChatMessage::Assistant {
            content: Some(format!("work-{i}: {}", "z".repeat(200))),
            reasoning_content: None,
            tool_calls: vec![],
        });
    }

    let compacted = loop_.compact_messages(240);
    assert!(compacted);

    let snapshot = loop_
        .messages
        .iter()
        .find_map(|msg| match msg {
            ChatMessage::System { content } if content.starts_with("ACTIVE_WORK_STATE") => {
                Some(content.clone())
            }
            _ => None,
        })
        .expect("expected active work snapshot message");
    assert!(snapshot.contains("Harden authentication middleware"));
    assert!(snapshot.contains("Patch auth middleware"));
}
