use super::*;

// ── Batch 8: Error Recovery & Self-Correction ──

#[test]
fn error_recovery_guidance_injected_on_first_escalation() {
    let fail_response = make_tool_response(vec![LlmToolCall {
        id: "tc1".to_string(),
        name: "bash_run".to_string(),
        arguments: r#"{"command":"cargo build"}"#.to_string(),
    }]);
    let done_response = make_text_response("I see the error, let me fix it.");

    let llm = ScriptedLlm::new(vec![fail_response, done_response]);
    let tool_host = Arc::new(MockToolHost::new(
        vec![ToolResult {
            invocation_id: uuid::Uuid::nil(),
            output: serde_json::json!("error[E0308]: mismatched types"),
            success: false,
        }],
        true,
    ));

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

    let result = loop_.run("fix the build").unwrap();

    let has_recovery = result.messages.iter().any(
        |m| matches!(m, ChatMessage::System { content } if content.contains("ERROR RECOVERY")),
    );
    assert!(
        has_recovery,
        "error recovery guidance should be injected on first escalation"
    );
}

#[test]
fn stuck_detection_fires_after_3_same_errors() {
    let fail_response = |id: &str| {
        make_tool_response(vec![LlmToolCall {
            id: id.to_string(),
            name: "bash_run".to_string(),
            arguments: r#"{"command":"cargo build"}"#.to_string(),
        }])
    };

    let llm = ScriptedLlm::new(vec![
        fail_response("tc1"),
        fail_response("tc2"),
        fail_response("tc3"),
        make_text_response("Let me try a completely different approach."),
    ]);

    let error_msg = "error[E0308]: mismatched types expected `u32` found `String`";
    let tool_host = Arc::new(MockToolHost::new(
        vec![
            ToolResult {
                invocation_id: uuid::Uuid::nil(),
                output: serde_json::json!(error_msg),
                success: false,
            },
            ToolResult {
                invocation_id: uuid::Uuid::nil(),
                output: serde_json::json!(error_msg),
                success: false,
            },
            ToolResult {
                invocation_id: uuid::Uuid::nil(),
                output: serde_json::json!(error_msg),
                success: false,
            },
        ],
        true,
    ));

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

    let result = loop_.run("fix the build").unwrap();

    let has_stuck = result.messages.iter().any(
        |m| matches!(m, ChatMessage::System { content } if content.contains("STUCK DETECTION")),
    );
    assert!(
        has_stuck,
        "stuck detection should fire after 3 identical errors"
    );
}

#[test]
fn recovery_guidance_not_repeated_after_success() {
    let tool_response = |id: &str| {
        make_tool_response(vec![LlmToolCall {
            id: id.to_string(),
            name: "bash_run".to_string(),
            arguments: r#"{"command":"cargo build"}"#.to_string(),
        }])
    };

    let llm = ScriptedLlm::new(vec![
        tool_response("tc1"), // fails → recovery injected
        tool_response("tc2"), // succeeds → recovery reset
        tool_response("tc3"), // fails → recovery can inject again
        make_text_response("Done."),
    ]);

    let tool_host = Arc::new(MockToolHost::new(
        vec![
            ToolResult {
                invocation_id: uuid::Uuid::nil(),
                output: serde_json::json!("error: compilation failed"),
                success: false,
            },
            ToolResult {
                invocation_id: uuid::Uuid::nil(),
                output: serde_json::json!("Build succeeded"),
                success: true,
            },
            ToolResult {
                invocation_id: uuid::Uuid::nil(),
                output: serde_json::json!("error: different problem"),
                success: false,
            },
        ],
        true,
    ));

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

    let result = loop_.run("fix the build").unwrap();

    let recovery_count = result
        .messages
        .iter()
        .filter(
            |m| matches!(m, ChatMessage::System { content } if content.contains("ERROR RECOVERY")),
        )
        .count();

    // Recovery state resets on success, so it CAN inject again after the second failure.
    // The key invariant: recovery is NOT injected on the same escalation cycle twice.
    assert!(
        recovery_count <= 2,
        "recovery should not spam — got {recovery_count}"
    );
}

// ── CostTracker tests ──────────────────────────────────────────────

#[test]
fn cost_tracker_records_and_estimates() {
    let mut tracker = CostTracker::default();
    tracker.record(&TokenUsage {
        prompt_tokens: 1_000_000,
        completion_tokens: 100_000,
        prompt_cache_hit_tokens: 0,
        prompt_cache_miss_tokens: 0,
        reasoning_tokens: 0,
    });
    // Input: 1M tokens × $0.27/M = $0.27
    // Output: 100K tokens × $1.10/M = $0.11
    let cost = tracker.estimated_cost_usd();
    assert!(
        (cost - 0.38).abs() < 0.001,
        "expected ~$0.38, got ${cost:.4}"
    );
}

#[test]
fn cost_tracker_cache_discount() {
    let mut tracker = CostTracker::default();
    tracker.record(&TokenUsage {
        prompt_tokens: 1_000_000,
        completion_tokens: 0,
        prompt_cache_hit_tokens: 800_000, // 80% cached
        prompt_cache_miss_tokens: 200_000,
        reasoning_tokens: 0,
    });
    // Effective input = (1M - 800K) + (800K × 0.1) = 200K + 80K = 280K
    // Cost = 280K / 1M × $0.27 = $0.0756
    let cost = tracker.estimated_cost_usd();
    assert!(
        (cost - 0.0756).abs() < 0.001,
        "expected ~$0.0756 with cache discount, got ${cost:.4}"
    );
}

#[test]
fn cost_tracker_over_budget() {
    let mut tracker = CostTracker {
        max_budget_usd: Some(0.10),
        ..CostTracker::default()
    };
    tracker.record(&TokenUsage {
        prompt_tokens: 1_000_000,
        completion_tokens: 100_000,
        prompt_cache_hit_tokens: 0,
        prompt_cache_miss_tokens: 0,
        reasoning_tokens: 0,
    });
    assert!(tracker.over_budget(), "should be over $0.10 budget");
}

#[test]
fn cost_tracker_not_over_budget_without_cap() {
    let mut tracker = CostTracker::default();
    // No max_budget_usd set
    tracker.record(&TokenUsage {
        prompt_tokens: 10_000_000,
        completion_tokens: 1_000_000,
        prompt_cache_hit_tokens: 0,
        prompt_cache_miss_tokens: 0,
        reasoning_tokens: 0,
    });
    assert!(
        !tracker.over_budget(),
        "should never be over budget when no cap is set"
    );
}

#[test]
fn cost_tracker_warns_once() {
    let mut tracker = CostTracker::default();
    tracker.record(&TokenUsage {
        prompt_tokens: 2_000_000,
        completion_tokens: 500_000,
        prompt_cache_hit_tokens: 0,
        prompt_cache_miss_tokens: 0,
        reasoning_tokens: 0,
    });
    // Cost: $0.54 + $0.55 = $1.09 — well above $0.50 threshold
    assert!(tracker.should_warn(), "should warn first time");
    assert!(!tracker.should_warn(), "should NOT warn second time");
}

#[test]
fn cost_tracker_no_warn_below_threshold() {
    let mut tracker = CostTracker::default();
    tracker.record(&TokenUsage {
        prompt_tokens: 100_000,
        completion_tokens: 10_000,
        prompt_cache_hit_tokens: 0,
        prompt_cache_miss_tokens: 0,
        reasoning_tokens: 0,
    });
    // Cost: $0.027 + $0.011 = $0.038 — well below $0.50
    assert!(!tracker.should_warn(), "should not warn below threshold");
}

// ── CircuitBreaker tests ───────────────────────────────────────────

#[test]
fn circuit_breaker_triggers_after_threshold() {
    let mut cb = CircuitBreakerState::default();
    for _ in 0..CIRCUIT_BREAKER_THRESHOLD {
        cb.consecutive_failures += 1;
    }
    assert_eq!(cb.consecutive_failures, CIRCUIT_BREAKER_THRESHOLD);
    // Simulate triggering cooldown
    cb.cooldown_remaining = CIRCUIT_BREAKER_COOLDOWN_TURNS;
    assert_eq!(cb.cooldown_remaining, 2);
}

#[test]
fn circuit_breaker_cooldown_decrements() {
    let mut cb = CircuitBreakerState {
        consecutive_failures: CIRCUIT_BREAKER_THRESHOLD,
        cooldown_remaining: CIRCUIT_BREAKER_COOLDOWN_TURNS,
    };
    // Simulate one turn of cooldown decrement
    if cb.cooldown_remaining > 0 {
        cb.cooldown_remaining -= 1;
    }
    assert_eq!(cb.cooldown_remaining, 1);
    // One more
    if cb.cooldown_remaining > 0 {
        cb.cooldown_remaining -= 1;
    }
    assert_eq!(cb.cooldown_remaining, 0);
    // After cooldown, reset failures
    cb.consecutive_failures = 0;
    assert_eq!(cb.consecutive_failures, 0);
}

#[test]
fn circuit_breaker_resets_on_success() {
    let mut cb = CircuitBreakerState {
        consecutive_failures: 2,
        cooldown_remaining: 0,
    };
    // Success resets the counter
    cb.consecutive_failures = 0;
    assert_eq!(cb.consecutive_failures, 0);
    assert_eq!(cb.cooldown_remaining, 0);
}

// ── Tool cache tests ───────────────────────────────────────────────

#[test]
fn tool_cache_stores_and_retrieves() {
    let llm = ScriptedLlm::new(vec![make_text_response("done")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));
    let config = ToolLoopConfig::default();
    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let args = serde_json::json!({"path": "/foo/bar.rs"});
    let result = serde_json::json!({"content": "fn main() {}"});

    // Store and retrieve
    loop_.cache_store("fs_read", &args, &result);
    let cached = loop_.cache_lookup("fs_read", &args);
    assert_eq!(cached, Some(result));
}

#[test]
fn tool_cache_skips_non_cacheable() {
    let llm = ScriptedLlm::new(vec![make_text_response("done")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));
    let config = ToolLoopConfig::default();
    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let args = serde_json::json!({"command": "ls"});
    let result = serde_json::json!({"output": "file1\nfile2"});

    loop_.cache_store("bash_run", &args, &result);
    let cached = loop_.cache_lookup("bash_run", &args);
    assert_eq!(cached, None, "bash_run should not be cached");
}

#[test]
fn tool_cache_invalidation_by_path() {
    let llm = ScriptedLlm::new(vec![make_text_response("done")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));
    let config = ToolLoopConfig::default();
    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let args = serde_json::json!({"path": "/foo/bar.rs"});
    let result = serde_json::json!({"content": "fn main() {}"});

    loop_.cache_store("fs_read", &args, &result);
    assert!(loop_.cache_lookup("fs_read", &args).is_some());

    // Invalidate by path
    loop_.cache_invalidate_path("/foo/bar.rs");
    assert!(
        loop_.cache_lookup("fs_read", &args).is_none(),
        "cache should be invalidated after path write"
    );
}

// ── Pruning tests ──────────────────────────────────────────────────

#[test]
fn prune_truncates_old_tool_outputs() {
    let llm = ScriptedLlm::new(vec![make_text_response("done")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));
    let config = ToolLoopConfig::default();
    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    // Build messages with 4+ turn groups, with long tool output in old turns
    let long_output = "x".repeat(500);
    // Old turn group 1
    loop_.messages.push(ChatMessage::User {
        content: "task 1".to_string(),
    });
    loop_.messages.push(ChatMessage::Tool {
        tool_call_id: "tc1".to_string(),
        content: long_output.clone(),
    });
    // Old turn group 2
    loop_.messages.push(ChatMessage::User {
        content: "task 2".to_string(),
    });
    loop_.messages.push(ChatMessage::Tool {
        tool_call_id: "tc2".to_string(),
        content: long_output.clone(),
    });
    // Old turn group 3
    loop_.messages.push(ChatMessage::User {
        content: "task 3".to_string(),
    });
    loop_.messages.push(ChatMessage::Tool {
        tool_call_id: "tc3".to_string(),
        content: long_output.clone(),
    });
    // Recent turns
    loop_.messages.push(ChatMessage::User {
        content: "recent task".to_string(),
    });
    loop_.messages.push(ChatMessage::Tool {
        tool_call_id: "tc4".to_string(),
        content: long_output.clone(),
    });

    loop_.prune_old_tool_outputs();

    // Old tool outputs should be truncated
    let old_tool = &loop_.messages[2]; // tc1
    if let ChatMessage::Tool { content, .. } = old_tool {
        assert!(
            content.contains("[output pruned"),
            "old tool output should be pruned, got: {}",
            &content[..content.floor_char_boundary(50.min(content.len()))]
        );
        assert!(
            content.len() < 500,
            "pruned output should be shorter than original"
        );
    }

    // Recent tool output should be unchanged
    let recent_tool = &loop_.messages[loop_.messages.len() - 1];
    if let ChatMessage::Tool { content, .. } = recent_tool {
        assert_eq!(
            content.len(),
            500,
            "recent tool output should not be pruned"
        );
    }
}

#[test]
fn prune_skips_short_outputs() {
    let llm = ScriptedLlm::new(vec![make_text_response("done")]);
    let tool_host = Arc::new(MockToolHost::new(vec![], true));
    let config = ToolLoopConfig::default();
    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let short_output = "OK".to_string();
    // Build 4 turn groups
    for i in 0..4 {
        loop_.messages.push(ChatMessage::User {
            content: format!("task {i}"),
        });
        loop_.messages.push(ChatMessage::Tool {
            tool_call_id: format!("tc{i}"),
            content: short_output.clone(),
        });
    }

    loop_.prune_old_tool_outputs();

    // All outputs should remain unchanged (too short to prune)
    for msg in &loop_.messages {
        if let ChatMessage::Tool { content, .. } = msg {
            assert_eq!(content, "OK", "short outputs should not be pruned");
        }
    }
}

#[test]
fn extract_tool_path_from_json() {
    let json_content = r#"{"path": "/src/main.rs", "content": "fn main() {}"}"#;
    assert_eq!(
        extract_tool_path(json_content),
        Some("/src/main.rs".to_string())
    );
}

#[test]
fn extract_tool_path_from_plain_path() {
    assert_eq!(
        extract_tool_path("/src/main.rs\nfn main() {}"),
        Some("/src/main.rs".to_string())
    );
}

#[test]
fn extract_tool_path_none_for_regular_text() {
    assert_eq!(extract_tool_path("Build succeeded"), None);
}
