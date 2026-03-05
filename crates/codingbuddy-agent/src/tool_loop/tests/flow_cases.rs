use super::*;

#[test]
fn simple_text_response_no_tools() {
    let llm = ScriptedLlm::new(vec![make_text_response("Hello, world!")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        ToolLoopConfig::default(),
        "You are helpful.".to_string(),
        default_tools(),
    );

    let result = loop_.run("Hi").unwrap();
    assert_eq!(result.response, "Hello, world!");
    assert_eq!(result.turns, 1);
    assert!(result.tool_calls_made.is_empty());
    assert_eq!(result.finish_reason, "stop");
}

#[test]
fn single_tool_call_and_response() {
    let llm = ScriptedLlm::new(vec![
        // Turn 1: LLM calls fs_read
        make_tool_response(vec![LlmToolCall {
            id: "call_1".to_string(),
            name: "fs_read".to_string(),
            arguments: r#"{"path":"src/lib.rs"}"#.to_string(),
        }]),
        // Turn 2: LLM responds with text
        make_text_response("The file contains a module definition."),
    ]);

    let tool_host = Arc::new(MockToolHost::new(
        vec![ToolResult {
            invocation_id: uuid::Uuid::nil(),
            success: true,
            output: serde_json::json!({"content": "mod tests;"}),
        }],
        true,
    ));

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        ToolLoopConfig::default(),
        "You are helpful.".to_string(),
        default_tools(),
    );

    let result = loop_.run("What's in src/lib.rs?").unwrap();
    assert_eq!(result.response, "The file contains a module definition.");
    assert_eq!(result.turns, 2);
    assert_eq!(result.tool_calls_made.len(), 1);
    assert_eq!(result.tool_calls_made[0].tool_name, "fs_read");
    assert!(result.tool_calls_made[0].success);
}

#[test]
fn todo_tools_persist_session_checklist() {
    let workspace =
        std::env::temp_dir().join(format!("codingbuddy-tool-loop-test-{}", Uuid::now_v7()));
    std::fs::create_dir_all(&workspace).expect("workspace");
    let store = Store::new(&workspace).expect("store");
    let session = Session {
        session_id: Uuid::now_v7(),
        workspace_root: workspace.display().to_string(),
        baseline_commit: None,
        status: SessionState::Idle,
        budgets: SessionBudgets {
            per_turn_seconds: 30,
            max_think_tokens: 4096,
        },
        active_plan_id: None,
    };
    store.save_session(&session).expect("save session");

    let llm = ScriptedLlm::new(vec![
            make_tool_response(vec![LlmToolCall {
                id: "todo_write_1".to_string(),
                name: "todo_write".to_string(),
                arguments: r#"{"items":[{"content":"Investigate failing CI","status":"in_progress"},{"content":"Patch Windows build","status":"pending"}]}"#.to_string(),
            }]),
            make_tool_response(vec![LlmToolCall {
                id: "todo_read_1".to_string(),
                name: "todo_read".to_string(),
                arguments: "{}".to_string(),
            }]),
            make_text_response("Checklist initialized."),
        ]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));
    let config = ToolLoopConfig {
        workspace: Some(workspace.clone()),
        session_id: Some(session.session_id),
        ..Default::default()
    };
    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );
    let result = loop_.run("start complex work").expect("run");
    assert_eq!(result.response, "Checklist initialized.");
    assert_eq!(result.tool_calls_made.len(), 2);
    assert_eq!(result.tool_calls_made[0].tool_name, "todo_write");
    assert_eq!(result.tool_calls_made[1].tool_name, "todo_read");

    let persisted = store
        .list_session_todos(session.session_id)
        .expect("list todos");
    assert_eq!(persisted.len(), 2);
    assert_eq!(persisted[0].status, "in_progress");
    assert_eq!(persisted[1].status, "pending");
}

#[test]
fn todo_write_rejects_multiple_in_progress_items() {
    let workspace =
        std::env::temp_dir().join(format!("codingbuddy-tool-loop-test-{}", Uuid::now_v7()));
    std::fs::create_dir_all(&workspace).expect("workspace");
    let store = Store::new(&workspace).expect("store");
    let session = Session {
        session_id: Uuid::now_v7(),
        workspace_root: workspace.display().to_string(),
        baseline_commit: None,
        status: SessionState::Idle,
        budgets: SessionBudgets {
            per_turn_seconds: 30,
            max_think_tokens: 4096,
        },
        active_plan_id: None,
    };
    store.save_session(&session).expect("save session");

    let llm = ScriptedLlm::new(vec![make_tool_response(vec![LlmToolCall {
            id: "todo_write_1".to_string(),
            name: "todo_write".to_string(),
            arguments: r#"{"items":[{"content":"A","status":"in_progress"},{"content":"B","status":"in_progress"}]}"#.to_string(),
        }])]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));
    let config = ToolLoopConfig {
        workspace: Some(workspace),
        session_id: Some(session.session_id),
        max_turns: 1,
        ..Default::default()
    };
    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let err = loop_.run("set checklist").expect_err("run should fail");
    assert!(
        err.to_string()
            .contains("todo_write allows at most one 'in_progress'")
    );
}

#[test]
fn multi_tool_calls_parallel() {
    let llm = ScriptedLlm::new(vec![
        // Turn 1: LLM calls two tools in parallel
        make_tool_response(vec![
            LlmToolCall {
                id: "call_1".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path":"a.rs"}"#.to_string(),
            },
            LlmToolCall {
                id: "call_2".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path":"b.rs"}"#.to_string(),
            },
        ]),
        // Turn 2: text response
        make_text_response("Both files processed."),
    ]);

    let tool_host = Arc::new(MockToolHost::new(
        vec![
            ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: true,
                output: serde_json::json!("file a"),
            },
            ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: true,
                output: serde_json::json!("file b"),
            },
        ],
        true,
    ));

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        ToolLoopConfig::default(),
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("Read both files").unwrap();
    assert_eq!(result.turns, 2);
    assert_eq!(result.tool_calls_made.len(), 2);
}

#[test]
fn tool_search_caps_promotions_for_weak_models() {
    let llm = ScriptedLlm::new(vec![]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));
    let config = ToolLoopConfig {
        model: "deepseek-chat".to_string(),
        ..Default::default()
    };
    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: codingbuddy_core::FunctionDefinition {
            name: "fs_read".to_string(),
            description: "read".to_string(),
            strict: None,
            parameters: serde_json::json!({"type":"object"}),
        },
    }];
    let mut loop_ = ToolUseLoop::new(&llm, tool_host, config, "system".to_string(), tools);
    loop_.set_discoverable_tools(vec![
        ToolDefinition {
            tool_type: "function".to_string(),
            function: codingbuddy_core::FunctionDefinition {
                name: "enter_plan_mode".to_string(),
                description: "plan mode".to_string(),
                strict: None,
                parameters: serde_json::json!({"type":"object"}),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: codingbuddy_core::FunctionDefinition {
                name: "exit_plan_mode".to_string(),
                description: "leave plan mode".to_string(),
                strict: None,
                parameters: serde_json::json!({"type":"object"}),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: codingbuddy_core::FunctionDefinition {
                name: "web_search".to_string(),
                description: "web search mode".to_string(),
                strict: None,
                parameters: serde_json::json!({"type":"object"}),
            },
        },
    ]);

    let baseline = loop_.tools.len();
    let output =
        super::agent_tools::handle_tool_search(&mut loop_, &serde_json::json!({"query":"mode"}));
    let promoted = loop_.tools.len().saturating_sub(baseline);

    assert!(
        promoted <= 1,
        "single-keyword weak-model search should be capped"
    );
    assert!(
        output.contains("weak-model guardrail"),
        "result should explain capped promotion behavior"
    );
}

#[test]
fn tool_chain_multiple_turns() {
    let llm = ScriptedLlm::new(vec![
        // Turn 1: grep
        make_tool_response(vec![LlmToolCall {
            id: "c1".to_string(),
            name: "fs_read".to_string(),
            arguments: r#"{"path":"x.rs"}"#.to_string(),
        }]),
        // Turn 2: read the found file
        make_tool_response(vec![LlmToolCall {
            id: "c2".to_string(),
            name: "fs_read".to_string(),
            arguments: r#"{"path":"y.rs"}"#.to_string(),
        }]),
        // Turn 3: final answer
        make_text_response("Found the implementation in y.rs."),
    ]);

    let tool_host = Arc::new(MockToolHost::new(
        vec![
            ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: true,
                output: serde_json::json!("result 1"),
            },
            ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: true,
                output: serde_json::json!("result 2"),
            },
        ],
        true,
    ));

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        ToolLoopConfig::default(),
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("Find auth code").unwrap();
    assert_eq!(result.turns, 3);
    assert_eq!(result.tool_calls_made.len(), 2);
    assert_eq!(result.response, "Found the implementation in y.rs.");
}

#[test]
fn max_turns_limit_stops_loop() {
    // LLM always returns tool calls, never stops.
    // Use different args each turn to avoid triggering doom loop detection.
    let infinite_tool_calls: Vec<LlmResponse> = (0..5)
        .map(|i| {
            make_tool_response(vec![LlmToolCall {
                id: format!("c{i}"),
                name: "fs_read".to_string(),
                arguments: format!(r#"{{"path":"test_{i}.rs"}}"#),
            }])
        })
        .collect();

    let llm = ScriptedLlm::new(infinite_tool_calls);
    let tool_host = Arc::new(MockToolHost::new(
        (0..5)
            .map(|_| ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: true,
                output: serde_json::json!("ok"),
            })
            .collect(),
        true,
    ));

    let config = ToolLoopConfig {
        max_turns: 3,
        ..Default::default()
    };

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("loop forever").unwrap();
    assert_eq!(result.turns, 3);
    assert_eq!(result.finish_reason, "max_turns");
}

#[test]
fn tool_call_denied_by_policy() {
    let llm = ScriptedLlm::new(vec![
        // Turn 1: LLM calls a tool
        make_tool_response(vec![LlmToolCall {
            id: "c1".to_string(),
            name: "fs_read".to_string(),
            arguments: r#"{"path":"secret.txt"}"#.to_string(),
        }]),
        // Turn 2: After denial, LLM responds differently
        make_text_response("I cannot access that file."),
    ]);

    // Tool host does NOT auto-approve
    let tool_host = Arc::new(MockToolHost::new(vec![], false));

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        ToolLoopConfig::default(),
        "system".to_string(),
        default_tools(),
    );

    // No approval callback → denied by default
    let result = loop_.run("Read secret file").unwrap();
    assert_eq!(result.turns, 2);
    assert_eq!(result.tool_calls_made.len(), 1);
    assert!(!result.tool_calls_made[0].success);
    assert_eq!(result.response, "I cannot access that file.");
}

#[test]
fn content_filter_returns_error() {
    let llm = ScriptedLlm::new(vec![LlmResponse {
        text: String::new(),
        finish_reason: "content_filter".to_string(),
        reasoning_content: String::new(),
        tool_calls: vec![],
        usage: None,
    }]);

    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        ToolLoopConfig::default(),
        "system".to_string(),
        default_tools(),
    );

    let err = loop_.run("bad prompt").unwrap_err();
    assert!(err.to_string().contains("content filter"));
}

#[test]
fn usage_accumulated_across_turns() {
    let llm = ScriptedLlm::new(vec![
        make_tool_response(vec![LlmToolCall {
            id: "c1".to_string(),
            name: "fs_read".to_string(),
            arguments: r#"{"path":"x"}"#.to_string(),
        }]),
        make_text_response("done"),
    ]);

    let tool_host = Arc::new(MockToolHost::new(
        vec![ToolResult {
            invocation_id: uuid::Uuid::nil(),
            success: true,
            output: serde_json::json!("ok"),
        }],
        true,
    ));

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        ToolLoopConfig::default(),
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("test").unwrap();
    // Each response has 100 prompt + 50 completion, 2 turns
    assert_eq!(result.usage.prompt_tokens, 200);
    assert_eq!(result.usage.completion_tokens, 100);
}

#[test]
fn summarize_args_formats_correctly() {
    let args = serde_json::json!({"path": "src/lib.rs", "start_line": 10});
    let summary = summarize_args(&args);
    assert!(summary.contains("path=\"src/lib.rs\""));
    assert!(summary.contains("start_line=10"));
}

#[test]
fn continue_with_appends_to_history() {
    let llm = ScriptedLlm::new(vec![
        make_text_response("First answer"),
        make_text_response("Second answer"),
    ]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        ToolLoopConfig::default(),
        "system".to_string(),
        default_tools(),
    );

    let r1 = loop_.run("first question").unwrap();
    assert_eq!(r1.response, "First answer");

    let r2 = loop_.continue_with("follow up").unwrap();
    assert_eq!(r2.response, "Second answer");
    // Messages should contain: system, user1, assistant1, user2, assistant2
    assert_eq!(r2.messages.len(), 5);
}

#[test]
fn complex_continuation_injects_anchor_context() {
    let workspace =
        std::env::temp_dir().join(format!("codingbuddy-tool-loop-test-{}", Uuid::now_v7()));
    std::fs::create_dir_all(&workspace).expect("workspace");
    let store = Store::new(&workspace).expect("store");
    let session = Session {
        session_id: Uuid::now_v7(),
        workspace_root: workspace.display().to_string(),
        baseline_commit: None,
        status: SessionState::ExecutingStep,
        budgets: SessionBudgets {
            per_turn_seconds: 30,
            max_think_tokens: 4096,
        },
        active_plan_id: None,
    };
    store.save_session(&session).expect("save session");

    let llm = ScriptedLlm::new(vec![
        make_text_response("First answer"),
        make_text_response("Second answer"),
    ]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));
    let config = ToolLoopConfig {
        complexity: crate::complexity::PromptComplexity::Complex,
        workspace: Some(workspace),
        session_id: Some(session.session_id),
        initial_phase: Some(TaskPhase::Verify),
        ..Default::default()
    };
    let mut loop_ = ToolUseLoop::new(&llm, tool_host, config, "system".to_string(), vec![]);

    let _ = loop_.run("first question").expect("first run");
    let second = loop_.continue_with("follow up").expect("second run");

    let has_anchor = second.messages.iter().any(|msg| {
        matches!(
            msg,
            ChatMessage::System { content } if content.contains("CONTINUATION_CONTEXT")
        )
    });
    assert!(
        has_anchor,
        "complex continuation should inject anchor context"
    );
}

#[test]
fn complex_write_without_todo_refresh_injects_checklist_nudge() {
    let workspace =
        std::env::temp_dir().join(format!("codingbuddy-tool-loop-test-{}", Uuid::now_v7()));
    std::fs::create_dir_all(&workspace).expect("workspace");
    let store = Store::new(&workspace).expect("store");
    let session = Session {
        session_id: Uuid::now_v7(),
        workspace_root: workspace.display().to_string(),
        baseline_commit: None,
        status: SessionState::ExecutingStep,
        budgets: SessionBudgets {
            per_turn_seconds: 30,
            max_think_tokens: 4096,
        },
        active_plan_id: None,
    };
    store.save_session(&session).expect("save session");
    store
        .replace_session_todos(
            session.session_id,
            &[SessionTodoRecord {
                todo_id: Uuid::now_v7(),
                session_id: session.session_id,
                content: "Refactor parser".to_string(),
                status: "in_progress".to_string(),
                position: 0,
                created_at: chrono::Utc::now().to_rfc3339(),
                updated_at: chrono::Utc::now().to_rfc3339(),
            }],
        )
        .expect("seed todos");

    let llm = ScriptedLlm::new(vec![
            make_tool_response(vec![LlmToolCall {
                id: "edit_1".to_string(),
                name: "fs_edit".to_string(),
                arguments: r#"{"path":"src/main.rs","search":"fn main() {}","replace":"fn main() { println!(\"ok\"); }"}"#.to_string(),
            }]),
            make_text_response("done"),
        ]);
    let tool_host = Arc::new(MockToolHost::new(
        vec![ToolResult {
            invocation_id: uuid::Uuid::nil(),
            success: true,
            output: serde_json::json!("patched"),
        }],
        true,
    ));
    let tools = vec![
        ToolDefinition {
            tool_type: "function".to_string(),
            function: codingbuddy_core::FunctionDefinition {
                name: "fs_edit".to_string(),
                description: "edit".to_string(),
                strict: None,
                parameters: serde_json::json!({
                    "type":"object",
                    "properties": {
                        "path":{"type":"string"},
                        "search":{"type":"string"},
                        "replace":{"type":"string"}
                    },
                    "required":["path","search","replace"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: codingbuddy_core::FunctionDefinition {
                name: "todo_write".to_string(),
                description: "todo write".to_string(),
                strict: None,
                parameters: serde_json::json!({"type":"object"}),
            },
        },
    ];
    let config = ToolLoopConfig {
        complexity: crate::complexity::PromptComplexity::Complex,
        workspace: Some(workspace),
        session_id: Some(session.session_id),
        initial_phase: Some(TaskPhase::Verify),
        ..Default::default()
    };
    let mut loop_ = ToolUseLoop::new(&llm, tool_host, config, "system".to_string(), tools);

    let result = loop_.run("apply the patch").expect("run");
    let has_nudge = result.messages.iter().any(|msg| {
        matches!(
            msg,
            ChatMessage::System { content }
                if content.contains("Checklist discipline (complex workflow)")
        )
    });
    assert!(has_nudge, "missing checklist refresh should inject a nudge");
}
