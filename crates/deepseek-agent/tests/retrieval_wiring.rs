//! Tests for retrieval pipeline wiring (P8 Batch 4).

use anyhow::{Result, anyhow};
use deepseek_agent::tool_loop::{RetrievalContext, RetrieverCallback, ToolLoopConfig, ToolUseLoop};
use deepseek_core::{
    ChatMessage, ChatRequest, LlmRequest, LlmResponse, LlmToolCall, StreamCallback, StreamChunk,
    TokenUsage,
};
use deepseek_llm::LlmClient;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

// ── Scripted LLM ──

struct ScriptedLlm {
    responses: Mutex<VecDeque<LlmResponse>>,
    requests: Mutex<Vec<Vec<ChatMessage>>>,
}

impl ScriptedLlm {
    fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::from(responses)),
            requests: Mutex::new(Vec::new()),
        }
    }
}

impl LlmClient for ScriptedLlm {
    fn complete(&self, _req: &LlmRequest) -> Result<LlmResponse> {
        Err(anyhow!("not used"))
    }
    fn complete_streaming(&self, _req: &LlmRequest, _cb: StreamCallback) -> Result<LlmResponse> {
        Err(anyhow!("not used"))
    }
    fn complete_chat(&self, req: &ChatRequest) -> Result<LlmResponse> {
        self.requests.lock().unwrap().push(req.messages.clone());
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow!("exhausted"))
    }
    fn complete_chat_streaming(
        &self,
        req: &ChatRequest,
        _cb: StreamCallback,
    ) -> Result<LlmResponse> {
        self.complete_chat(req)
    }
    fn complete_fim(&self, _req: &deepseek_core::FimRequest) -> Result<LlmResponse> {
        Err(anyhow!("not used"))
    }
    fn complete_fim_streaming(
        &self,
        _req: &deepseek_core::FimRequest,
        _cb: StreamCallback,
    ) -> Result<LlmResponse> {
        Err(anyhow!("not used"))
    }
}

fn text_response(text: &str) -> LlmResponse {
    LlmResponse {
        text: text.to_string(),
        finish_reason: "stop".to_string(),
        reasoning_content: String::new(),
        tool_calls: vec![],
        usage: Some(TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            ..Default::default()
        }),
    }
}

/// A mock ToolHost that accepts everything.
struct NoopToolHost;

impl deepseek_core::ToolHost for NoopToolHost {
    fn propose(&self, call: deepseek_core::ToolCall) -> deepseek_core::ToolProposal {
        deepseek_core::ToolProposal {
            invocation_id: uuid::Uuid::now_v7(),
            call,
            approved: true,
        }
    }
    fn execute(&self, _approved: deepseek_core::ApprovedToolCall) -> deepseek_core::ToolResult {
        deepseek_core::ToolResult {
            invocation_id: uuid::Uuid::now_v7(),
            success: true,
            output: serde_json::json!({"result": "ok"}),
        }
    }
}

#[test]
fn retrieval_context_added_before_llm() {
    let llm = ScriptedLlm::new(vec![text_response("Done")]);
    let tool_host: Arc<dyn deepseek_core::ToolHost + Send + Sync> = Arc::new(NoopToolHost);

    let retriever: RetrieverCallback = Arc::new(|_query, _k| {
        Ok(vec![RetrievalContext {
            file_path: "src/main.rs".to_string(),
            start_line: 1,
            end_line: 10,
            content: "fn main() { println!(\"hello\"); }".to_string(),
            score: 0.95,
        }])
    });

    let config = ToolLoopConfig {
        retriever: Some(retriever),
        ..Default::default()
    };

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "You are a helpful assistant.".to_string(),
        vec![],
    );

    let result = loop_.run("How does main work?").unwrap();
    assert_eq!(result.response, "Done");

    // Check that the retrieval context was injected as a System message
    let requests = llm.requests.lock().unwrap();
    assert!(!requests.is_empty());
    let first_request_messages = &requests[0];

    let has_retrieval_context = first_request_messages.iter().any(|m| {
        if let ChatMessage::System { content } = m {
            content.contains("RETRIEVAL_CONTEXT")
        } else {
            false
        }
    });
    assert!(
        has_retrieval_context,
        "retrieval context should be injected before LLM call"
    );
}

#[test]
fn retrieval_respects_budget() {
    let llm = ScriptedLlm::new(vec![text_response("Done")]);
    let tool_host: Arc<dyn deepseek_core::ToolHost + Send + Sync> = Arc::new(NoopToolHost);

    // Create a retriever that returns lots of chunks
    let retriever: RetrieverCallback = Arc::new(|_query, _k| {
        let mut results = Vec::new();
        for i in 0..100 {
            results.push(RetrievalContext {
                file_path: format!("src/file_{}.rs", i),
                start_line: 1,
                end_line: 100,
                content: "x".repeat(10_000), // 10K chars each, ~2500 tokens
                score: 0.9 - i as f32 * 0.01,
            });
        }
        Ok(results)
    });

    let config = ToolLoopConfig {
        context_window_tokens: 10_000, // Small context window
        retriever: Some(retriever),
        ..Default::default()
    };

    let mut loop_ = ToolUseLoop::new(&llm, tool_host, config, "system".to_string(), vec![]);

    let result = loop_.run("search").unwrap();
    assert_eq!(result.response, "Done");

    // Verify context was truncated: budget is 10_000/5 = 2000 tokens
    // Each chunk is ~2500 tokens, so at most 0-1 chunks should be injected
    let requests = llm.requests.lock().unwrap();
    let first_request = &requests[0];
    let retrieval_msgs: Vec<_> = first_request
        .iter()
        .filter(|m| matches!(m, ChatMessage::System { content } if content.contains("RETRIEVAL_CONTEXT")))
        .collect();
    // Should have at most one retrieval context message
    assert!(retrieval_msgs.len() <= 1);
}

#[test]
fn index_query_tool_available() {
    // The index_query tool should be in the tool definitions
    let tools = deepseek_tools::tool_definitions();
    let has_index_query = tools.iter().any(|t| t.function.name == "index_query");
    assert!(
        has_index_query,
        "tool_definitions() should include index_query"
    );
}

#[test]
fn privacy_router_redacts_in_tool_loop() {
    let llm = ScriptedLlm::new(vec![
        // First response: call a tool
        LlmResponse {
            text: String::new(),
            finish_reason: "tool_calls".to_string(),
            reasoning_content: String::new(),
            tool_calls: vec![LlmToolCall {
                id: "call_1".to_string(),
                name: "fs_read".to_string(),
                arguments: r#"{"path": "secret.txt"}"#.to_string(),
            }],
            usage: Some(TokenUsage::default()),
        },
        // Second response: text reply
        text_response("Here is the info"),
    ]);

    let warnings: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let warnings_clone = warnings.clone();

    // Create a privacy router with a custom regex that the output_scanner doesn't have
    let privacy_config = deepseek_local_ml::PrivacyConfig {
        enabled: true,
        sensitive_regex: vec![r"INTERNAL_SECRET_\d+".to_string()],
        policy: deepseek_local_ml::PrivacyPolicy::Redact,
        ..Default::default()
    };
    let privacy_router = Arc::new(deepseek_local_ml::PrivacyRouter::new(privacy_config).unwrap());

    // Tool host that returns content with a custom secret pattern
    struct SensitiveToolHost;
    impl deepseek_core::ToolHost for SensitiveToolHost {
        fn propose(&self, call: deepseek_core::ToolCall) -> deepseek_core::ToolProposal {
            deepseek_core::ToolProposal {
                invocation_id: uuid::Uuid::now_v7(),
                call,
                approved: true,
            }
        }
        fn execute(&self, _approved: deepseek_core::ApprovedToolCall) -> deepseek_core::ToolResult {
            deepseek_core::ToolResult {
                invocation_id: uuid::Uuid::now_v7(),
                success: true,
                output: serde_json::json!({
                    "content": "Config value: INTERNAL_SECRET_42 is set\nOther line is fine"
                }),
            }
        }
    }

    let tool_host: Arc<dyn deepseek_core::ToolHost + Send + Sync> = Arc::new(SensitiveToolHost);

    let config = ToolLoopConfig {
        privacy_router: Some(privacy_router),
        ..Default::default()
    };

    let mut loop_ = ToolUseLoop::new(
        &llm,
        tool_host,
        config,
        "system".to_string(),
        deepseek_tools::tool_definitions(),
    );

    loop_.set_stream_callback(Arc::new(move |chunk| {
        if let StreamChunk::SecurityWarning { message } = chunk {
            warnings_clone.lock().unwrap().push(message);
        }
    }));

    let result = loop_.run("Read secret.txt");
    assert!(result.is_ok());

    // Check that privacy warning was emitted
    let w = warnings.lock().unwrap();
    let has_privacy_warning = w.iter().any(|msg| msg.contains("Privacy router"));
    assert!(
        has_privacy_warning,
        "privacy router should emit a security warning when redacting"
    );
}
