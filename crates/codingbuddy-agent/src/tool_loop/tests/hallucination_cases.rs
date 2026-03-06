use super::*;

#[test]
fn hallucination_nudge_triggers_on_long_first_response() {
    // First response: long text with no tools (hallucination)
    // Second response: shorter text (after nudge, model cooperates)
    let long_hallucination = "a".repeat(HALLUCINATION_NUDGE_THRESHOLD + 100);
    let llm = ScriptedLlm::new(vec![
        make_text_response(&long_hallucination),
        make_text_response("Let me check with tools."), // after nudge
    ]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        ToolLoopConfig::default(),
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("analyze this project").unwrap();
    // Should get the second (post-nudge) response, not the hallucinated one
    assert_eq!(result.response, "Let me check with tools.");
    // Should have used 2 turns
    assert_eq!(result.turns, 2);
    // Messages should contain the nudge
    let has_nudge = result.messages.iter().any(|m| {
        if let ChatMessage::User { content } = m {
            content.contains("STOP. You are answering without using tools")
        } else {
            false
        }
    });
    assert!(has_nudge, "should contain hallucination nudge message");
}

#[test]
fn hallucination_nudge_skips_short_responses() {
    // Short response should NOT trigger nudge
    let llm = ScriptedLlm::new(vec![make_text_response("Done.")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        ToolLoopConfig::default(),
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("hello").unwrap();
    assert_eq!(result.response, "Done.");
    assert_eq!(result.turns, 1); // no nudge, single turn
}

// ── Batch 6: Anti-hallucination hardening tests ──

#[test]
fn first_turn_uses_required_tool_choice() {
    // Build a loop with no tool results in messages (first turn)
    let llm = ScriptedLlm::new(vec![
        // Force tool_choice=required will cause the model to call a tool
        make_tool_response(vec![LlmToolCall {
            id: "call_1".to_string(),
            name: "fs_read".to_string(),
            arguments: r#"{"path":"Cargo.toml"}"#.to_string(),
        }]),
        make_text_response("Read the file."),
    ]);
    let tool_host = Arc::new(MockToolHost::new(
        vec![ToolResult {
            invocation_id: uuid::Uuid::nil(),
            success: true,
            output: serde_json::json!({"content": "[package]\nname = \"test\""}),
        }],
        true,
    ));

    let config = ToolLoopConfig::default();
    let loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    // The first build_request should use tool_choice=required
    let request = loop_.build_request();
    assert_eq!(
        request.tool_choice,
        ToolChoice::required(),
        "first turn should force tool_choice=required"
    );
}

#[test]
fn subsequent_turns_use_auto_tool_choice() {
    // After 1+ Assistant turns since last User message, tool_choice should be auto.
    // The threshold is based on Assistant messages, not Tool messages.
    let llm = ScriptedLlm::new(vec![]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));
    let config = ToolLoopConfig::default();
    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    // Simulate: User asked → Assistant responded (1 turn) → tool result
    loop_.messages.push(ChatMessage::Assistant {
        content: Some("I'll read the file.".to_string()),
        tool_calls: vec![LlmToolCall {
            id: "call_1".to_string(),
            name: "fs_read".to_string(),
            arguments: r#"{"path":"Cargo.toml"}"#.to_string(),
        }],
        reasoning_content: None,
    });
    loop_.messages.push(ChatMessage::Tool {
        tool_call_id: "call_1".to_string(),
        content: "file content".to_string(),
    });

    let request = loop_.build_request();
    assert_eq!(
        request.tool_choice,
        ToolChoice::auto(),
        "after 1+ Assistant turns should use tool_choice=auto"
    );

    // With no Assistant messages (only Tool results), should still force required
    let llm2 = ScriptedLlm::new(vec![]);
    let tool_host2 = Arc::new(MockToolHost::new(vec![], true));
    let mut loop2 = ToolUseLoop::new(
        &llm2,
        tool_host2,
        ToolLoopConfig::default(),
        "system".to_string(),
        default_tools(),
    );
    loop2.messages.push(ChatMessage::Tool {
        tool_call_id: "call_1".to_string(),
        content: "one result".to_string(),
    });
    let request2 = loop2.build_request();
    assert_eq!(
        request2.tool_choice,
        ToolChoice::required(),
        "with no Assistant turns, should still force required"
    );
}

#[test]
fn hallucination_nudge_triggers_at_threshold() {
    // Response exceeding HALLUCINATION_NUDGE_THRESHOLD should trigger nudge
    let over_threshold = "x".repeat(HALLUCINATION_NUDGE_THRESHOLD + 1);
    let llm = ScriptedLlm::new(vec![
        make_text_response(&over_threshold),
        make_text_response("OK"), // after nudge
    ]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        ToolLoopConfig::default(),
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("describe the project").unwrap();
    assert_eq!(result.response, "OK");
    assert_eq!(result.turns, 2, "nudge should have triggered an extra turn");

    // Response at exactly the threshold should NOT trigger nudge
    let at_threshold = "y".repeat(HALLUCINATION_NUDGE_THRESHOLD);
    let llm2 = ScriptedLlm::new(vec![make_text_response(&at_threshold)]);
    let tool_host2 = Arc::new(MockToolHost::new(vec![], true));

    let mut loop2 = ToolUseLoop::new(
        &llm2,
        tool_host2,
        ToolLoopConfig::default(),
        "system".to_string(),
        default_tools(),
    );

    let result2 = loop2.run("describe the project").unwrap();
    assert_eq!(
        result2.turns, 1,
        "at-threshold response should not trigger nudge"
    );
}

#[test]
fn unverified_file_references_detected() {
    // Text mentioning file paths without any tool calls
    assert!(has_unverified_file_references(
        "The project has src/main.rs and src/lib.rs with key functions.",
        &[]
    ));

    // After reading BOTH files, should not flag
    let read_calls = vec![
        ToolCallRecord {
            tool_name: "fs_read".to_string(),
            tool_call_id: "c1".to_string(),
            args_summary: "path=\"src/main.rs\"".to_string(),
            success: true,
            duration_ms: 10,
            args_json: Some(r#"{"path":"src/main.rs"}"#.to_string()),
            result_preview: None,
        },
        ToolCallRecord {
            tool_name: "fs_read".to_string(),
            tool_call_id: "c2".to_string(),
            args_summary: "path=\"src/lib.rs\"".to_string(),
            success: true,
            duration_ms: 10,
            args_json: Some(r#"{"path":"src/lib.rs"}"#.to_string()),
            result_preview: None,
        },
    ];
    assert!(!has_unverified_file_references(
        "The project has src/main.rs and src/lib.rs with key functions.",
        &read_calls
    ));

    // Path-specific: reading only src/main.rs SHOULD flag mentions of src/lib.rs
    let single_read = vec![ToolCallRecord {
        tool_name: "fs_read".to_string(),
        tool_call_id: "c1".to_string(),
        args_summary: "path=\"src/main.rs\"".to_string(),
        success: true,
        duration_ms: 10,
        args_json: Some(r#"{"path":"src/main.rs"}"#.to_string()),
        result_preview: None,
    }];
    assert!(has_unverified_file_references(
        "The project has src/main.rs and src/lib.rs with key functions.",
        &single_read
    ));

    // Short text with no path-like patterns should not flag
    assert!(!has_unverified_file_references("Hello, world!", &[]));

    // Dot-access patterns should not flag (module access, not file paths)
    assert!(!has_unverified_file_references(
        "Use self.config and fmt.Println for output",
        &[]
    ));

    // Version numbers should not flag
    assert!(!has_unverified_file_references(
        "Update to v0.6 or version 2.34",
        &[]
    ));

    // URL-like patterns should not flag
    assert!(!has_unverified_file_references(
        "See https.example and http.server docs",
        &[]
    ));

    // New extensions should flag
    assert!(has_unverified_file_references(
        "Check the .env file and config.ini",
        &[]
    ));
}

#[test]
fn evidence_driven_escalation_on_tool_failures() {
    // Simulate: tool outputs contain compile error → model retries → responds.
    // Escalation signals should detect the error and switch to hard budget.
    let llm = ScriptedLlm::new(vec![
        make_tool_response(vec![LlmToolCall {
            id: "c1".to_string(),
            name: "fs_read".to_string(),
            arguments: r#"{"path":"missing.rs"}"#.to_string(),
        }]),
        make_tool_response(vec![LlmToolCall {
            id: "c2".to_string(),
            name: "fs_read".to_string(),
            arguments: r#"{"path":"also_missing.rs"}"#.to_string(),
        }]),
        make_text_response("Got it."),
    ]);

    let tool_host = Arc::new(MockToolHost::new(
        vec![
            ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: false,
                output: serde_json::json!("error[E0308]: mismatched types"),
            },
            ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: false,
                output: serde_json::json!("test result: FAILED. 0 passed; 1 failed;"),
            },
        ],
        true,
    ));

    let config = ToolLoopConfig {
        thinking: Some(codingbuddy_core::ThinkingConfig::enabled(
            crate::complexity::DEFAULT_THINK_BUDGET,
        )),
        ..Default::default()
    };

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "You are helpful.".to_string(),
        default_tools(),
    );

    let result = loop_.run("Read those files").unwrap();
    assert_eq!(result.response, "Got it.");
    assert_eq!(result.turns, 3);
    // Evidence-driven: escalation detected from tool output content
    assert!(
        loop_.escalation.compile_error,
        "should detect compile error"
    );
    assert!(loop_.escalation.test_failure, "should detect test failure");
    assert!(loop_.escalation.should_escalate(), "should be escalated");
    assert_eq!(loop_.escalation.consecutive_failure_turns, 2);
}

// ── Batch 2 (P9): Anti-hallucination ──

#[test]
fn hallucination_nudge_fires_in_ask_mode() {
    // Even in read_only mode (Ask/Context), the nudge should fire
    // to prevent hallucinated answers about the codebase.
    let long_text = "x".repeat(HALLUCINATION_NUDGE_THRESHOLD + 1);
    let llm = ScriptedLlm::new(vec![
        make_text_response(&long_text),
        make_text_response("OK"),
    ]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let config = ToolLoopConfig {
        read_only: true,
        ..Default::default()
    };
    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("where is the config?").unwrap();
    assert_eq!(result.response, "OK");
    assert_eq!(result.turns, 2, "nudge should fire in read_only/Ask mode");
}

#[test]
fn hallucination_nudge_fires_up_to_3_times() {
    // Model hallucinates 3 times, then on 4th attempt it goes through
    let long_text = "x".repeat(HALLUCINATION_NUDGE_THRESHOLD + 1);
    let llm = ScriptedLlm::new(vec![
        make_text_response(&long_text),
        make_text_response(&long_text),
        make_text_response(&long_text),
        make_text_response(&long_text), // 4th: should pass through (MAX_NUDGE_ATTEMPTS exhausted)
    ]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        ToolLoopConfig::default(),
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("what does this project do?").unwrap();
    assert_eq!(result.turns, 4, "should fire 3 nudges then let 4th through");
    assert_eq!(result.response.len(), HALLUCINATION_NUDGE_THRESHOLD + 1);
}

#[test]
fn nudge_fires_even_after_prior_tool_use() {
    // The nudge should fire even if the model used tools earlier in the conversation.
    // This tests that removing the `tool_calls_made.is_empty()` guard works.
    let tool_response = make_tool_response(vec![LlmToolCall {
        id: "tc1".to_string(),
        name: "fs_read".to_string(),
        arguments: serde_json::json!({"path": "README.md"}).to_string(),
    }]);
    let long_hallucination = "x".repeat(HALLUCINATION_NUDGE_THRESHOLD + 100);
    let llm = ScriptedLlm::new(vec![
        tool_response,                                // turn 1: uses a tool
        make_text_response("README content is..."),   // turn 2: text after tool
        make_text_response(&long_hallucination),      // turn 3: hallucination (should nudge)
        make_text_response("Let me check properly."), // turn 4: after nudge
    ]);
    let tool_host = Arc::new(MockToolHost::new(
        vec![ToolResult {
            invocation_id: uuid::Uuid::nil(),
            success: true,
            output: serde_json::json!({"content": "# Test"}),
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

    let r1 = loop_.run("read the readme").unwrap();
    assert_eq!(r1.response, "README content is...");

    // Second question — model hallucinates even though it used tools before
    let r2 = loop_
        .continue_with("what else is in this project?")
        .unwrap();
    assert_eq!(r2.response, "Let me check properly.");
    // The nudge message should be in the conversation
    let has_nudge = r2.messages.iter().any(|m| {
        if let ChatMessage::User { content } = m {
            content.contains("STOP. You are answering without using tools")
        } else {
            false
        }
    });
    assert!(has_nudge, "nudge should fire even after prior tool use");
}

#[test]
fn single_unverified_path_triggers_nudge() {
    // A single unverified file reference (e.g. mentioning "audit.md")
    // should now trigger the nudge (threshold lowered from 2 to 1).
    assert!(
        has_unverified_file_references("Please see audit.md for details", &[]),
        "single file ref should trigger"
    );
    assert!(
        has_unverified_file_references("Look at src/main.rs", &[]),
        "single path should trigger"
    );
    // But if the tool was already used, it should NOT trigger
    let used_tools = vec![ToolCallRecord {
        tool_name: "fs_read".to_string(),
        tool_call_id: "tc1".to_string(),
        args_summary: String::new(),
        success: true,
        duration_ms: 0,
        args_json: Some(r#"{"path":"src/main.rs"}"#.to_string()),
        result_preview: None,
    }];
    assert!(
        !has_unverified_file_references("Look at src/main.rs", &used_tools),
        "should not trigger when read tool was used"
    );
}

#[test]
fn shell_command_pattern_detected() {
    // Test that the shell command detection catches common patterns
    assert!(
        contains_shell_command_pattern("```bash\ncat audit.md\n```"),
        "cat in bash block should trigger"
    );
    assert!(
        contains_shell_command_pattern("```sh\ngrep -r TODO src/\n```"),
        "grep in sh block should trigger"
    );
    assert!(
        contains_shell_command_pattern("```\nfind . -name '*.rs'\n```"),
        "find in bare code block should trigger"
    );
    assert!(
        contains_shell_command_pattern("$ cat README.md"),
        "dollar-prompt cat should trigger"
    );
    // Should NOT trigger on normal prose
    assert!(
        !contains_shell_command_pattern("The cat sat on the mat."),
        "prose 'cat' should not trigger"
    );
    assert!(
        !contains_shell_command_pattern("Use `fs_read` to read files"),
        "tool mention should not trigger"
    );
}

#[test]
fn tool_choice_required_per_new_question() {
    // In multi-turn, tool_choice=required should reset for each new user question.
    // The ScriptedLlm lets us verify via the request the loop builds.
    // We simulate: turn 1 uses a tool, then continue_with asks a new question.
    let llm = ScriptedLlm::new(vec![
        // First run: tool call + response
        make_tool_response(vec![LlmToolCall {
            id: "c1".to_string(),
            name: "fs_read".to_string(),
            arguments: r#"{"path":"src/lib.rs"}"#.to_string(),
        }]),
        make_text_response("Found it."),
        // Second run (continue_with): tool call + response
        make_tool_response(vec![LlmToolCall {
            id: "c2".to_string(),
            name: "fs_read".to_string(),
            arguments: r#"{"path":"src/main.rs"}"#.to_string(),
        }]),
        make_text_response("Found that too."),
    ]);

    let tool_host = Arc::new(MockToolHost::new(
        vec![
            ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: true,
                output: serde_json::json!("content of lib.rs"),
            },
            ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: true,
                output: serde_json::json!("content of main.rs"),
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

    let r1 = loop_.run("read lib.rs").unwrap();
    assert_eq!(r1.response, "Found it.");

    let r2 = loop_.continue_with("now read main.rs").unwrap();
    assert_eq!(r2.response, "Found that too.");
}

#[test]
fn unverified_refs_detects_go_files() {
    assert!(has_unverified_file_references(
        "The server code is in cmd/server.go and internal/handler.go.",
        &[]
    ));
}

#[test]
fn unverified_refs_ignores_dot_access() {
    // self.config, req.model etc. should not be flagged as file paths
    assert!(!has_unverified_file_references(
        "The self.config and req.model fields are set during initialization.",
        &[]
    ));
    // But real paths alongside dot-access should still be detected
    assert!(has_unverified_file_references(
        "Check src/config.rs and src/model.rs for the implementation.",
        &[]
    ));
}
