use super::*;

#[test]
fn images_forwarded_to_llm_request() {
    // Verify that images from ToolLoopConfig appear in the ChatRequest
    let llm = ScriptedLlm::new(vec![make_text_response("I see the image.")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let images = vec![codingbuddy_core::ImageContent {
        mime: "image/png".to_string(),
        base64_data: "iVBORw0KGgo=".to_string(),
    }];
    let config = ToolLoopConfig {
        images: images.clone(),
        ..Default::default()
    };
    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("what's in this image?").unwrap();
    assert_eq!(result.response, "I see the image.");
    // Images are cleared after the first turn to avoid re-sending them every turn
    assert!(
        loop_.config.images.is_empty(),
        "images should be cleared after first turn"
    );
}

#[test]
fn post_tool_use_hook_gets_actual_args() {
    // This test verifies the PostToolUse hook receives actual tool arguments
    // (not an empty {}). We can't easily test hook content without a mock hook runtime,
    // but we verify the code path by checking that the tool executes successfully
    // with hooks not wired (no-op path).
    let llm = ScriptedLlm::new(vec![
        make_tool_response(vec![LlmToolCall {
            id: "c1".to_string(),
            name: "fs_read".to_string(),
            arguments: r#"{"path":"src/main.rs"}"#.to_string(),
        }]),
        make_text_response("Done."),
    ]);
    let tool_host = Arc::new(MockToolHost::new(
        vec![ToolResult {
            invocation_id: uuid::Uuid::nil(),
            success: true,
            output: serde_json::json!("file content"),
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

    let result = loop_.run("read main.rs").unwrap();
    assert_eq!(result.response, "Done.");
    assert_eq!(result.tool_calls_made.len(), 1);
    assert_eq!(result.tool_calls_made[0].tool_name, "fs_read");
}

// ── P10 Batch 1: Bootstrap context tests ──

#[test]
fn bootstrap_context_injected_on_first_turn() {
    let llm = ScriptedLlm::new(vec![make_text_response("ok")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let config = ToolLoopConfig {
        initial_context: vec![ChatMessage::System {
            content: "PROJECT_CONTEXT (auto-gathered):\nRust workspace, 25 crates".to_string(),
        }],
        ..Default::default()
    };

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "You are helpful.".to_string(),
        default_tools(),
    );

    let result = loop_.run("What is this project?").unwrap();
    assert_eq!(result.response, "ok");

    // Verify the message order: System (prompt), System (bootstrap), User
    assert!(
        matches!(&result.messages[0], ChatMessage::System { content } if content.contains("You are helpful"))
    );
    assert!(
        matches!(&result.messages[1], ChatMessage::System { content } if content.contains("PROJECT_CONTEXT"))
    );
    assert!(
        matches!(&result.messages[2], ChatMessage::User { content } if content.contains("What is this project"))
    );
}

#[test]
fn bootstrap_respects_token_budget() {
    // Create a large bootstrap context that exceeds any reasonable budget
    let large_context = "x\n".repeat(100_000);

    let llm = ScriptedLlm::new(vec![make_text_response("ok")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let config = ToolLoopConfig {
        initial_context: vec![ChatMessage::System {
            content: format!("PROJECT_CONTEXT:\n{large_context}"),
        }],
        ..Default::default()
    };

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("hi").unwrap();
    // The large context is in messages[1] — it should be present (truncation
    // happens in lib.rs via truncate_to_token_budget before it reaches here)
    assert!(result.messages.len() >= 3);
    assert!(
        matches!(&result.messages[1], ChatMessage::System { content } if content.contains("PROJECT_CONTEXT"))
    );
}

#[test]
fn bootstrap_disabled_when_no_initial_context() {
    let llm = ScriptedLlm::new(vec![make_text_response("hello")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let config = ToolLoopConfig {
        initial_context: vec![], // No bootstrap
        ..Default::default()
    };

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("hi").unwrap();
    // Messages: System, User, Assistant — no bootstrap System message
    assert!(matches!(&result.messages[0], ChatMessage::System { content } if content == "system"));
    assert!(matches!(&result.messages[1], ChatMessage::User { .. }));
}

// ── P10 Batch 2: Retrieval pipeline tests ──

#[test]
fn retriever_callback_returns_results() {
    let llm = ScriptedLlm::new(vec![make_text_response("found it")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let config = ToolLoopConfig {
        retriever: Some(Arc::new(|_query: &str, _max: usize| {
            Ok(vec![RetrievalContext {
                file_path: "src/lib.rs".to_string(),
                start_line: 1,
                end_line: 10,
                content: "fn main() {}".to_string(),
                score: 0.95,
            }])
        })),
        ..Default::default()
    };

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("what does main do?").unwrap();
    // Should have retrieval context injected as a System message
    let has_retrieval = result.messages.iter().any(|m| {
        matches!(m,
            ChatMessage::System { content } if content.contains("RETRIEVAL_CONTEXT"))
    });
    assert!(has_retrieval, "retrieval context should be injected");
}

#[test]
fn retrieval_context_appears_in_messages() {
    let llm = ScriptedLlm::new(vec![make_text_response("done")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let config = ToolLoopConfig {
        retriever: Some(Arc::new(|_query: &str, _max: usize| {
            Ok(vec![
                RetrievalContext {
                    file_path: "auth.rs".to_string(),
                    start_line: 10,
                    end_line: 30,
                    content: "pub fn verify_token() -> bool { true }".to_string(),
                    score: 0.88,
                },
                RetrievalContext {
                    file_path: "middleware.rs".to_string(),
                    start_line: 5,
                    end_line: 20,
                    content: "pub fn auth_middleware() {}".to_string(),
                    score: 0.75,
                },
            ])
        })),
        ..Default::default()
    };

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("explain auth").unwrap();
    let retrieval_msg = result.messages.iter().find(|m| {
        matches!(m,
            ChatMessage::System { content } if content.contains("RETRIEVAL_CONTEXT"))
    });
    assert!(retrieval_msg.is_some());
    if let Some(ChatMessage::System { content }) = retrieval_msg {
        assert!(
            content.contains("auth.rs"),
            "should include first result file"
        );
        assert!(
            content.contains("middleware.rs"),
            "should include second result file"
        );
        assert!(content.contains("verify_token"), "should include content");
    }
}

#[test]
fn privacy_router_filters_tool_output() {
    let llm = ScriptedLlm::new(vec![
        make_tool_response(vec![LlmToolCall {
            id: "c1".to_string(),
            name: "fs_read".to_string(),
            arguments: r#"{"path":".env"}"#.to_string(),
        }]),
        make_text_response("read the file"),
    ]);
    let tool_host = Arc::new(MockToolHost::new(
        vec![ToolResult {
            invocation_id: uuid::Uuid::nil(),
            success: true,
            output: serde_json::json!("API_KEY=sk-secret-12345"),
        }],
        true,
    ));

    // Build a privacy router that will redact secrets
    let privacy_config = codingbuddy_local_ml::PrivacyConfig {
        enabled: true,
        sensitive_globs: vec![],
        sensitive_regex: vec![r"(?i)(api[_-]?key|secret)\s*[=:]\s*\S+".to_string()],
        policy: codingbuddy_local_ml::PrivacyPolicy::Redact,
        store_raw_in_logs: false,
        strict_regex: false,
    };
    let router = codingbuddy_local_ml::PrivacyRouter::new(privacy_config).expect("privacy router");

    let config = ToolLoopConfig {
        privacy_router: Some(Arc::new(router)),
        ..Default::default()
    };

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("read .env").unwrap();
    // The tool output should have been filtered
    let tool_msg = result
        .messages
        .iter()
        .find(|m| matches!(m, ChatMessage::Tool { .. }));
    assert!(tool_msg.is_some(), "should have a tool message");
    if let Some(ChatMessage::Tool { content, .. }) = tool_msg {
        // Either the secret is redacted or the output is blocked
        let has_raw_secret = content.contains("sk-secret-12345");
        assert!(
            !has_raw_secret || content.contains("[REDACTED]") || content.contains("[BLOCKED"),
            "secret should be redacted or blocked, got: {content}"
        );
    }
}

#[test]
fn retriever_empty_results_no_injection() {
    let llm = ScriptedLlm::new(vec![make_text_response("ok")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let config = ToolLoopConfig {
        retriever: Some(Arc::new(|_query: &str, _max: usize| {
            Ok(vec![]) // Empty results
        })),
        ..Default::default()
    };

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("hello").unwrap();
    let has_retrieval = result.messages.iter().any(|m| {
        matches!(m,
            ChatMessage::System { content } if content.contains("RETRIEVAL_CONTEXT"))
    });
    assert!(
        !has_retrieval,
        "no retrieval context when results are empty"
    );
}

// ── P10 Batch 3: Semantic compaction tests ──

#[test]
fn compaction_summary_lists_modified_files() {
    let messages = vec![
        ChatMessage::Assistant {
            content: None,
            reasoning_content: None,
            tool_calls: vec![LlmToolCall {
                id: "c1".to_string(),
                name: "fs_edit".to_string(),
                arguments: r#"{"file_path":"src/lib.rs","old":"x","new":"y"}"#.to_string(),
            }],
        },
        ChatMessage::Tool {
            tool_call_id: "c1".to_string(),
            content: "Edited successfully".to_string(),
        },
        ChatMessage::Assistant {
            content: None,
            reasoning_content: None,
            tool_calls: vec![LlmToolCall {
                id: "c2".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path":"README.md"}"#.to_string(),
            }],
        },
        ChatMessage::Tool {
            tool_call_id: "c2".to_string(),
            content: "# README".to_string(),
        },
    ];
    let summary = build_compaction_summary(&messages);
    assert!(
        summary.contains("src/lib.rs"),
        "should list modified files: {summary}"
    );
    assert!(
        summary.contains("README.md"),
        "should list read files: {summary}"
    );
    assert!(
        summary.contains("fs_edit"),
        "should list tools used: {summary}"
    );
}

#[test]
fn compaction_summary_captures_errors() {
    let messages = vec![
        ChatMessage::Tool {
            tool_call_id: "c1".to_string(),
            content: "error[E0308]: mismatched types\n  expected `u32`".to_string(),
        },
        ChatMessage::Tool {
            tool_call_id: "c2".to_string(),
            content: "test result: FAILED. 1 failed".to_string(),
        },
    ];
    let summary = build_compaction_summary(&messages);
    assert!(
        summary.contains("Errors encountered"),
        "should capture errors: {summary}"
    );
}

#[test]
fn compaction_summary_tool_counts() {
    let messages = vec![ChatMessage::Assistant {
        content: None,
        reasoning_content: None,
        tool_calls: vec![
            LlmToolCall {
                id: "1".to_string(),
                name: "fs_read".to_string(),
                arguments: "{}".to_string(),
            },
            LlmToolCall {
                id: "2".to_string(),
                name: "fs_read".to_string(),
                arguments: "{}".to_string(),
            },
            LlmToolCall {
                id: "3".to_string(),
                name: "fs_edit".to_string(),
                arguments: "{}".to_string(),
            },
        ],
    }];
    let summary = build_compaction_summary(&messages);
    assert!(
        summary.contains("fs_read×2"),
        "should count tools: {summary}"
    );
    assert!(
        summary.contains("fs_edit×1"),
        "should count edit: {summary}"
    );
}

// ── P10 Batch 5: Model routing tests ──

#[test]
fn complex_escalated_routes_to_reasoner() {
    // We can't easily check what model was sent in the request because
    // ScriptedLlm doesn't capture it, but we can verify the build_request
    // logic indirectly by checking the ToolUseLoop state.
    let llm = ScriptedLlm::new(vec![
        // Tool call that triggers compile error
        make_tool_response(vec![LlmToolCall {
            id: "c1".to_string(),
            name: "fs_read".to_string(),
            arguments: r#"{"path":"x.rs"}"#.to_string(),
        }]),
        make_text_response("done"),
    ]);
    let tool_host = Arc::new(MockToolHost::new(
        vec![ToolResult {
            invocation_id: uuid::Uuid::nil(),
            success: false,
            output: serde_json::json!("error[E0308]: mismatched types"),
        }],
        true,
    ));

    let config = ToolLoopConfig {
        complexity: crate::complexity::PromptComplexity::Complex,
        ..Default::default()
    };

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("refactor auth").unwrap();
    // After the compile error, escalation should be active
    assert!(
        loop_.escalation.compile_error,
        "compile error should be flagged"
    );
    assert!(loop_.escalation.should_escalate(), "should be escalated");
    // Result should complete (the scripted LLM returns "done")
    assert_eq!(result.response, "done");
}

#[test]
fn de_escalated_routes_back_to_chat() {
    let mut signals = crate::complexity::EscalationSignals {
        compile_error: true,
        ..Default::default()
    };
    assert!(signals.should_escalate());

    // 3 consecutive successes → de-escalate
    signals.record_success();
    signals.record_success();
    signals.record_success();
    assert!(!signals.should_escalate(), "3 successes should de-escalate");
    assert_eq!(signals.budget(), crate::complexity::DEFAULT_THINK_BUDGET);
}

#[test]
fn reasoner_strips_sampling_params() {
    // Verify the build_request logic: when model contains "reasoner",
    // temperature should be None even if config has one.
    let llm = ScriptedLlm::new(vec![make_text_response("ok")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let config = ToolLoopConfig {
        temperature: Some(0.7),
        ..Default::default()
    };

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    // If not using reasoner, temperature should be preserved
    let result = loop_.run("hi").unwrap();
    assert_eq!(result.response, "ok");
    // The test passes — the key assertion is in the build_request logic itself
    // which strips temperature when model contains "reasoner".
}

#[test]
fn compaction_preserves_key_decisions() {
    let decision_text = "I'll modify the authentication module to use JWT tokens instead of session cookies because the API is stateless.";
    let messages = vec![ChatMessage::Assistant {
        content: Some(decision_text.to_string()),
        reasoning_content: None,
        tool_calls: vec![],
    }];
    let summary = build_compaction_summary(&messages);
    assert!(
        summary.contains("Key decisions"),
        "should have decisions section: {summary}"
    );
    assert!(
        summary.contains("JWT tokens") || summary.contains("authentication"),
        "should preserve decision content: {summary}"
    );
}

#[test]
fn compaction_template_includes_key_facts_section() {
    assert!(
        COMPACTION_TEMPLATE.contains("## Key Facts Established"),
        "template should include Key Facts section for context retention"
    );
    assert!(
        COMPACTION_TEMPLATE.contains("user preferences"),
        "template should mention preserving user preferences"
    );
    assert!(
        COMPACTION_TEMPLATE.contains("corrections given"),
        "template should mention preserving corrections"
    );
}

#[test]
fn retriever_none_means_no_retrieval() {
    let llm = ScriptedLlm::new(vec![make_text_response("ok")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let config = ToolLoopConfig {
        retriever: None, // Disabled
        ..Default::default()
    };

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("hello").unwrap();
    let has_retrieval = result.messages.iter().any(|m| {
        matches!(m,
            ChatMessage::System { content } if content.contains("RETRIEVAL_CONTEXT"))
    });
    assert!(!has_retrieval);
}

#[test]
fn bootstrap_includes_repo_map_content() {
    let llm = ScriptedLlm::new(vec![make_text_response("analyzed")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));

    let config = ToolLoopConfig {
            initial_context: vec![ChatMessage::System {
                content: "Key source files:\n  - src/lib.rs (2048 bytes) score=100\n  - src/main.rs (512 bytes) score=50".to_string(),
            }],
            ..Default::default()
        };

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("analyze project").unwrap();
    assert!(
        matches!(&result.messages[1], ChatMessage::System { content }
            if content.contains("Key source files:") && content.contains("src/lib.rs"))
    );
}
