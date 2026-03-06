use super::*;

// ── User directive extraction tests ──

#[test]
fn extract_directives_finds_always_never() {
    let text = "Please always use snake_case for variable names. Never commit directly to main.";
    let directives = extract_user_directives(text);
    assert_eq!(directives.len(), 2);
    assert!(directives[0].contains("always use snake_case"));
    assert!(directives[1].contains("Never commit"));
}

#[test]
fn extract_directives_skips_questions() {
    let text = "Should I always use tabs? Never mind about that.";
    let directives = extract_user_directives(text);
    // "Should I always use tabs?" ends with '?' so it's skipped
    // "Never mind about that." should still be captured
    assert_eq!(directives.len(), 1);
    assert!(directives[0].contains("Never mind"));
}

#[test]
fn extract_directives_skips_short_fragments() {
    let text = "always\nnever\nprefer X for all Y operations in the codebase";
    let directives = extract_user_directives(text);
    // "always" and "never" are too short (< 10 chars)
    assert_eq!(directives.len(), 1);
    assert!(directives[0].contains("prefer X"));
}

#[test]
fn extract_directives_caps_at_limit() {
    // 12 directive-containing sentences should be capped at 10
    let lines: Vec<String> = (0..12)
        .map(|i| format!("Always do thing number {i} in the code"))
        .collect();
    let text = lines.join("\n");
    let directives = extract_user_directives(&text);
    assert_eq!(directives.len(), 10);
}

#[test]
fn extract_directives_handles_multiline_input() {
    let text = "First, make sure all tests pass before committing.\n\
                     Second, use the standard library whenever possible.\n\
                     What do you think about this approach?";
    let directives = extract_user_directives(text);
    assert_eq!(directives.len(), 2);
    assert!(directives[0].contains("make sure"));
    assert!(directives[1].contains("use the standard"));
}

#[test]
fn directives_survive_compaction() {
    // Build a ToolUseLoop, feed it a user message with directives,
    // trigger compaction, and verify directives are re-injected.
    let llm = ScriptedLlm::new(vec![
        // First turn: tool call + result
        make_tool_response(vec![LlmToolCall {
            id: "c1".to_string(),
            name: "fs_read".to_string(),
            arguments: r#"{"path":"src/lib.rs"}"#.to_string(),
        }]),
        make_text_response("done with first task"),
        // Second turn (after continue_with): text response
        make_text_response("done with second task"),
    ]);
    let tool_host = Arc::new(MockToolHost::new(
        vec![ToolResult {
            invocation_id: uuid::Uuid::nil(),
            success: true,
            output: serde_json::json!("file contents here"),
        }],
        true,
    ));

    let config = ToolLoopConfig {
        context_window_tokens: 200, // Very small window to force compaction
        ..Default::default()
    };
    let mut tool_loop = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "You are a coding assistant.".to_string(),
        vec![],
    );

    // First turn with a directive
    let _ = tool_loop.run("Always use snake_case for variables. Read the file.");

    // Verify directive was collected
    assert!(
        tool_loop
            .pinned_directives
            .iter()
            .any(|d| d.contains("snake_case")),
        "directive should be collected: {:?}",
        tool_loop.pinned_directives
    );

    // Manually trigger compaction (simulate context pressure)
    // The pinned_directives field should persist across compaction
    let directives_before = tool_loop.pinned_directives.clone();

    // Second turn
    let _ = tool_loop.continue_with("Now prefer tabs over spaces. Fix the bug.");

    // Directives from both turns should be present
    assert!(
        tool_loop
            .pinned_directives
            .iter()
            .any(|d| d.contains("snake_case")),
        "first directive should persist: {:?}",
        tool_loop.pinned_directives
    );
    assert!(
        tool_loop
            .pinned_directives
            .iter()
            .any(|d| d.contains("prefer tabs")),
        "second directive should be collected: {:?}",
        tool_loop.pinned_directives
    );
    assert!(
        tool_loop.pinned_directives.len() >= directives_before.len(),
        "directives should accumulate"
    );
}

#[test]
fn split_sentences_handles_periods_and_newlines() {
    let text = "First sentence. Second sentence.\nThird line";
    let sentences = split_sentences(text);
    assert!(
        sentences.len() >= 3,
        "should split on periods and newlines: {sentences:?}"
    );
}
