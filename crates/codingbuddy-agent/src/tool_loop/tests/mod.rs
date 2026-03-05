use super::anti_hallucination::split_sentences;
use super::compaction::COMPACTION_TEMPLATE;
use super::safety::DOOM_LOOP_HISTORY_SIZE;
use super::*;
use codingbuddy_core::{
    LlmResponse, Plan, PlanStep, Session, SessionBudgets, SessionState, TaskPhase, ToolCall,
    ToolProposal, ToolResult,
};
use codingbuddy_store::{SessionTodoRecord, Store, TaskQueueRecord};
use std::collections::VecDeque;
use std::sync::Mutex;

mod compaction_cases;
mod context_cases;
mod directive_cases;
mod doom_loop_cases;
mod flow_cases;
mod hallucination_cases;
mod quality_cases;
mod resilience_cases;

// ── Scripted LLM mock ──

struct ScriptedLlm {
    responses: Mutex<VecDeque<LlmResponse>>,
}

impl ScriptedLlm {
    fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::from(responses)),
        }
    }
}

impl LlmClient for ScriptedLlm {
    fn complete(&self, _req: &codingbuddy_core::LlmRequest) -> Result<LlmResponse> {
        unimplemented!()
    }
    fn complete_streaming(
        &self,
        _req: &codingbuddy_core::LlmRequest,
        _cb: StreamCallback,
    ) -> Result<LlmResponse> {
        unimplemented!()
    }
    fn complete_chat(&self, _req: &ChatRequest) -> Result<LlmResponse> {
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow!("no more scripted responses"))
    }
    fn complete_chat_streaming(
        &self,
        req: &ChatRequest,
        _cb: StreamCallback,
    ) -> Result<LlmResponse> {
        self.complete_chat(req)
    }
    fn complete_fim(&self, _req: &codingbuddy_core::FimRequest) -> Result<LlmResponse> {
        unimplemented!()
    }
    fn complete_fim_streaming(
        &self,
        _req: &codingbuddy_core::FimRequest,
        _cb: StreamCallback,
    ) -> Result<LlmResponse> {
        unimplemented!()
    }
}

// ── Scripted tool host mock ──

struct MockToolHost {
    results: Mutex<VecDeque<ToolResult>>,
    auto_approve: bool,
    execute_count: std::sync::atomic::AtomicUsize,
}

impl MockToolHost {
    fn new(results: Vec<ToolResult>, auto_approve: bool) -> Self {
        Self {
            results: Mutex::new(VecDeque::from(results)),
            auto_approve,
            execute_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    #[cfg(not(target_os = "windows"))]
    fn executed_count(&self) -> usize {
        self.execute_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl ToolHost for MockToolHost {
    fn propose(&self, call: ToolCall) -> ToolProposal {
        ToolProposal {
            invocation_id: uuid::Uuid::nil(),
            call,
            approved: self.auto_approve,
        }
    }
    fn execute(&self, _approved: ApprovedToolCall) -> ToolResult {
        self.execute_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.results
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(ToolResult {
                invocation_id: uuid::Uuid::nil(),
                success: false,
                output: serde_json::json!({"error": "no mock result"}),
            })
    }
}

fn make_text_response(text: &str) -> LlmResponse {
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

fn make_tool_response(tool_calls: Vec<LlmToolCall>) -> LlmResponse {
    LlmResponse {
        text: String::new(),
        finish_reason: "tool_calls".to_string(),
        reasoning_content: String::new(),
        tool_calls,
        usage: Some(TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            ..Default::default()
        }),
    }
}

fn default_tools() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: codingbuddy_core::FunctionDefinition {
            name: "fs_read".to_string(),
            description: "Read a file".to_string(),
            strict: None,
            parameters: serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
        },
    }]
}
