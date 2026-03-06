use super::*;

// ── Doom loop detection ──

#[test]
fn doom_loop_detects_repeated_identical_calls() {
    let mut tracker = DoomLoopTracker::default();
    let args = r#"{"file_path":"/src/main.rs"}"#;

    assert!(!tracker.record("fs_read", args), "first call: no doom loop");
    assert!(
        !tracker.record("fs_read", args),
        "second call: no doom loop"
    );
    assert!(
        tracker.record("fs_read", args),
        "third call: doom loop detected"
    );
}

#[test]
fn doom_loop_no_false_positive_on_different_args() {
    let mut tracker = DoomLoopTracker::default();

    tracker.record("fs_read", r#"{"file_path":"/a.rs"}"#);
    tracker.record("fs_read", r#"{"file_path":"/b.rs"}"#);
    let detected = tracker.record("fs_read", r#"{"file_path":"/c.rs"}"#);
    assert!(!detected, "different args should not trigger doom loop");
}

#[test]
fn doom_loop_no_false_positive_on_different_tools() {
    let mut tracker = DoomLoopTracker::default();
    let args = r#"{"file_path":"/main.rs"}"#;

    tracker.record("fs_read", args);
    tracker.record("fs_glob", args);
    let detected = tracker.record("fs_grep", args);
    assert!(!detected, "different tools should not trigger doom loop");
}

#[test]
fn doom_loop_resets_warning_on_different_call() {
    let mut tracker = DoomLoopTracker::default();
    let args = r#"{"file_path":"/main.rs"}"#;

    tracker.record("fs_read", args);
    tracker.record("fs_read", args);
    assert!(tracker.record("fs_read", args));
    tracker.mark_warned();

    // Same call again — no new warning because already warned
    assert!(!tracker.record("fs_read", args));

    // Different call resets
    tracker.record("fs_edit", "{}");

    // Now back to original — should detect again
    tracker.record("fs_read", args);
    tracker.record("fs_read", args);
    assert!(tracker.record("fs_read", args), "should detect after reset");
}

#[test]
fn doom_loop_warning_only_fires_once() {
    let mut tracker = DoomLoopTracker::default();
    let args = r#"{"file_path":"/main.rs"}"#;

    tracker.record("fs_read", args);
    tracker.record("fs_read", args);
    assert!(tracker.record("fs_read", args));
    tracker.mark_warned();

    // Subsequent identical calls should NOT re-trigger
    assert!(!tracker.record("fs_read", args));
    assert!(!tracker.record("fs_read", args));
}

#[test]
fn doom_loop_respects_history_window() {
    let mut tracker = DoomLoopTracker::default();
    let target_args = r#"{"file_path":"/main.rs"}"#;

    // Two identical calls
    tracker.record("fs_read", target_args);
    tracker.record("fs_read", target_args);

    // Fill with enough different calls to push the first two out of the window
    for i in 0..DOOM_LOOP_HISTORY_SIZE {
        tracker.record("fs_glob", &format!(r#"{{"pattern":"*.{i}"}}"#));
    }

    // Now the original calls are outside the window — one more should not trigger
    let detected = tracker.record("fs_read", target_args);
    assert!(!detected, "old calls outside window should not count");
}

#[test]
fn doom_loop_terminates_loop_as_blocking_gate() {
    // Use the ScriptedLlm to simulate a model that makes the same tool call 3 times.
    // The doom loop gate should STOP the loop at turn 3 — turn 4 should never execute.
    let responses = vec![
        // Turn 1: model calls fs_read with same args
        LlmResponse {
            text: String::new(),
            reasoning_content: String::new(),
            tool_calls: vec![LlmToolCall {
                id: "call_1".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"file_path":"/main.rs"}"#.to_string(),
            }],
            finish_reason: "tool_calls".to_string(),
            usage: None,
            compatibility: None,
        },
        // Turn 2: same call
        LlmResponse {
            text: String::new(),
            reasoning_content: String::new(),
            tool_calls: vec![LlmToolCall {
                id: "call_2".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"file_path":"/main.rs"}"#.to_string(),
            }],
            finish_reason: "tool_calls".to_string(),
            usage: None,
            compatibility: None,
        },
        // Turn 3: same call — should trigger doom loop gate and STOP
        LlmResponse {
            text: String::new(),
            reasoning_content: String::new(),
            tool_calls: vec![LlmToolCall {
                id: "call_3".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"file_path":"/main.rs"}"#.to_string(),
            }],
            finish_reason: "tool_calls".to_string(),
            usage: None,
            compatibility: None,
        },
        // Turn 4: should NEVER be reached — doom loop gate terminates the loop
        LlmResponse {
            text: "This should never appear.".to_string(),
            reasoning_content: String::new(),
            tool_calls: vec![],
            finish_reason: "stop".to_string(),
            usage: None,
            compatibility: None,
        },
    ];

    let llm = ScriptedLlm::new(responses);
    let tool_result = ToolResult {
        invocation_id: uuid::Uuid::nil(),
        success: true,
        output: serde_json::json!({"content": "fn main() {}"}),
    };
    let tool_host = Arc::new(MockToolHost::new(
        vec![tool_result.clone(), tool_result.clone(), tool_result],
        true,
    ));

    let config = ToolLoopConfig {
        max_turns: 10,
        ..Default::default()
    };

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        default_tools(),
    );

    let result = loop_.run("show me the main file").unwrap();

    // The loop should terminate at doom loop detection (blocking gate)
    assert_eq!(
        result.finish_reason, "doom_loop",
        "finish_reason should be doom_loop when gate triggers"
    );

    // Turn 4 should NOT have executed
    assert_ne!(
        result.response, "This should never appear.",
        "doom loop gate should prevent further LLM calls"
    );

    // Verify doom loop guidance was injected into messages
    let has_doom_guidance = result.messages.iter().any(|m| {
            matches!(m, ChatMessage::System { content } if content.contains("repeating the same action"))
        });
    assert!(
        has_doom_guidance,
        "doom loop guidance should be in messages for context if user continues"
    );
}

#[test]
fn doom_loop_reset_clears_all_state() {
    let mut tracker = DoomLoopTracker::default();
    let args = r#"{"file_path":"/main.rs"}"#;

    // Build up to doom loop detection
    tracker.record("fs_read", args);
    tracker.record("fs_read", args);
    assert!(tracker.record("fs_read", args), "doom loop detected");
    tracker.mark_warned();

    // Reset clears everything
    tracker.reset();
    assert!(
        tracker.recent_calls.is_empty(),
        "history should be empty after reset"
    );
    assert!(
        !tracker.warning_injected,
        "warning flag should be cleared after reset"
    );

    // Same calls after reset need full threshold again
    assert!(
        !tracker.record("fs_read", args),
        "1st call after reset: no doom loop"
    );
    assert!(
        !tracker.record("fs_read", args),
        "2nd call after reset: no doom loop"
    );
    assert!(
        tracker.record("fs_read", args),
        "3rd call after reset: doom loop re-detected"
    );
}
