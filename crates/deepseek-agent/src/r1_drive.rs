//! R1 Drive-Tools loop: R1 emits tool-intent JSON, orchestrator executes.
//!
//! This is the core of the `R1DriveTools` mode. R1 (deepseek-reasoner) receives
//! an ObservationPack and returns structured JSON instructions. The orchestrator
//! validates, executes, and feeds results back until R1 signals completion.
//!
//! R1 cannot use function calling (the API doesn't support it for deepseek-reasoner),
//! so all tool invocation goes through plain-text JSON envelopes parsed by
//! `protocol::parse_r1_response()`.

use crate::mode_router::{FailureTracker, ModeRouterConfig};
use crate::observation::{
    ActionRecord, ObservationPackBuilder, RepoFacts, extract_file_refs, summarize_args,
};
use crate::protocol::{
    DelegatePatch, DoneResponse, R1ParseError, R1Response, ToolIntent, parse_r1_response,
    r1_retry_prompt, r1_tool_to_internal,
};
use deepseek_core::{
    ApprovedToolCall, ChatMessage, ChatRequest, StreamChunk, ToolCall, ToolChoice, ToolHost,
    ToolResult,
};
use deepseek_llm::LlmClient;
use deepseek_observe::Observer;
use std::sync::Arc;

/// Outcome of an R1 drive-tools session.
#[derive(Debug, Clone)]
pub enum R1DriveOutcome {
    /// R1 wants V3 to write a patch.
    DelegatePatch(DelegatePatch),
    /// R1 declares the task complete.
    Done(DoneResponse),
    /// R1 aborted with a reason.
    Abort(String),
    /// Budget exhausted (max steps reached).
    BudgetExhausted { steps_used: u32 },
    /// Parse failures exceeded retry limit.
    ParseFailure { last_error: String },
}

/// System prompt template for R1 in drive-tools mode.
const R1_SYSTEM_PROMPT: &str = r#"You are a senior software architect directing a coding agent. You analyze observations and issue precise tool commands.

## Rules
1. Respond with EXACTLY ONE JSON object per turn. No other text.
2. Valid response types:
   - {"type":"tool_intent", "step_id":"S<N>", "tool":"<tool_name>", "args":{...}, "why":"<rationale>"}
   - {"type":"delegate_patch", "step_id":"S<N>", "task":"<description>", "constraints":[...], "acceptance":[...], "inputs":{"relevant_files":[...]}}
   - {"type":"done", "summary":"<what was accomplished>"}
   - {"type":"abort", "reason":"<why task cannot be completed>"}
3. Available tools: read_file, write_file, edit_file, glob, ripgrep, run_cmd, git_status, git_diff, list_dir, apply_patch, multi_edit, diagnostics_check, index_query
4. Work incrementally: read before writing, verify after changing.
5. When you have enough information to write code changes, use "delegate_patch" to hand off to the code writer.
6. Use "done" when the task is fully complete and verified.
7. Use "abort" only when the task is fundamentally impossible.

## Tool argument formats
- read_file: {"file_path": "path/to/file"}
- write_file: {"file_path": "path/to/file", "content": "..."}
- edit_file: {"file_path": "path/to/file", "old_string": "...", "new_string": "..."}
- glob: {"pattern": "**/*.rs"}
- ripgrep: {"pattern": "regex", "glob": "*.rs"}
- run_cmd: {"command": "cargo test", "timeout_ms": 30000}
- git_status: {}
- git_diff: {"ref": "HEAD"}
- list_dir: {"path": "src/"}
- apply_patch: {"patch": "unified diff text"}
- multi_edit: {"edits": [{"file_path": "...", "old_string": "...", "new_string": "..."}]}
- diagnostics_check: {"path": "src/"}
- index_query: {"query": "search terms"}
"#;

/// Configuration for an R1 drive session.
#[derive(Debug, Clone)]
pub struct R1DriveConfig {
    /// Model name for R1 calls (typically "deepseek-reasoner").
    pub model: String,
    /// Max tokens for R1 responses.
    pub max_tokens: u32,
    /// Max steps before budget exhaustion.
    pub max_steps: u32,
    /// Max parse retries per step.
    pub max_parse_retries: u32,
}

impl R1DriveConfig {
    pub fn from_mode_router_config(config: &ModeRouterConfig, model: &str) -> Self {
        Self {
            model: model.to_string(),
            max_tokens: 8192,
            max_steps: config.r1_max_steps,
            max_parse_retries: config.r1_max_parse_retries,
        }
    }
}

/// Run the R1 drive-tools loop.
///
/// This function:
/// 1. Sends R1 an observation pack as context
/// 2. Parses R1's JSON response
/// 3. Executes tool_intent via the tool host
/// 4. Builds a new observation pack with results
/// 5. Repeats until R1 returns delegate_patch/done/abort or budget runs out
#[allow(clippy::too_many_arguments)]
pub fn r1_drive_loop(
    config: &R1DriveConfig,
    llm: &(dyn LlmClient + Send + Sync),
    tool_host: &Arc<dyn ToolHost + Send + Sync>,
    observer: &Observer,
    repo: &RepoFacts,
    tracker: &mut FailureTracker,
    task_description: &str,
    initial_context: &str,
    stream_callback: Option<&deepseek_core::StreamCallback>,
) -> R1DriveOutcome {
    let mut step: u32 = 1;
    let mut messages: Vec<ChatMessage> = Vec::new();
    let mut obs_builder = ObservationPackBuilder::new(step, repo.clone());

    // Build initial system prompt
    let system_prompt = format!(
        "{R1_SYSTEM_PROMPT}\n\n## Current task\n{task_description}\n\n## Initial context\n{initial_context}"
    );

    messages.push(ChatMessage::System {
        content: system_prompt,
    });

    // Initial user message with task
    messages.push(ChatMessage::User {
        content: format!(
            "Begin working on this task. Analyze the situation and issue your first command.\n\n\
             ## Task\n{task_description}"
        ),
    });

    loop {
        // Budget check
        if tracker.r1_steps_used >= config.max_steps {
            observer.verbose_log(&format!(
                "r1_drive: budget exhausted at step {step} ({} steps used)",
                tracker.r1_steps_used
            ));
            return R1DriveOutcome::BudgetExhausted {
                steps_used: tracker.r1_steps_used,
            };
        }

        // Call R1
        let request = ChatRequest {
            model: config.model.clone(),
            messages: messages.clone(),
            tools: vec![],
            tool_choice: ToolChoice::none(),
            max_tokens: config.max_tokens,
            temperature: None,
            thinking: None, // R1 has built-in CoT
        };

        observer.verbose_log(&format!(
            "r1_drive: step {step} calling R1 model={} messages={}",
            config.model,
            messages.len()
        ));

        // Notify stream callback
        if let Some(cb) = stream_callback {
            cb(StreamChunk::ContentDelta(format!(
                "\n[r1_drive step {step}] thinking...\n"
            )));
        }

        let response = match llm.complete_chat(&request) {
            Ok(r) => r,
            Err(e) => {
                observer.verbose_log(&format!("r1_drive: R1 API error at step {step}: {e}"));
                return R1DriveOutcome::Abort(format!("R1 API error: {e}"));
            }
        };

        tracker.r1_steps_used += 1;

        let r1_text = if !response.text.is_empty() {
            &response.text
        } else if !response.reasoning_content.is_empty() {
            // R1 may put the JSON in reasoning_content
            &response.reasoning_content
        } else {
            observer.verbose_log(&format!("r1_drive: empty R1 response at step {step}"));
            return R1DriveOutcome::Abort("R1 returned empty response".to_string());
        };

        // Parse R1 response with retries
        let parsed = match parse_with_retries(
            r1_text,
            config,
            llm,
            &mut messages,
            observer,
            step,
            stream_callback,
            tracker,
        ) {
            Ok(resp) => resp,
            Err(last_error) => {
                return R1DriveOutcome::ParseFailure {
                    last_error: last_error.to_string(),
                };
            }
        };

        // Add R1's response to message history
        messages.push(ChatMessage::Assistant {
            content: Some(r1_text.to_string()),
            reasoning_content: Some(response.reasoning_content.clone()),
            tool_calls: vec![],
        });

        match parsed {
            R1Response::ToolIntent(intent) => {
                observer.verbose_log(&format!(
                    "r1_drive: step {step} tool_intent tool={} step_id={}",
                    intent.tool, intent.step_id
                ));

                if let Some(cb) = stream_callback {
                    cb(StreamChunk::ContentDelta(format!(
                        "[r1_drive step {step}] executing {}({})\n",
                        intent.tool,
                        summarize_intent_args(&intent)
                    )));
                }

                // Execute the tool
                let result = execute_r1_tool_intent(&intent, tool_host, observer);

                // Build action record
                let output_str = result.output.to_string();
                let refs = extract_file_refs(&output_str);
                let action = ActionRecord {
                    tool: r1_tool_to_internal(&intent.tool)
                        .unwrap_or(&intent.tool)
                        .to_string(),
                    args_summary: summarize_args(
                        r1_tool_to_internal(&intent.tool).unwrap_or(&intent.tool),
                        &intent.args,
                    ),
                    success: result.success,
                    exit_code: result
                        .output
                        .get("exit_code")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32),
                    output_head: truncate_output(&output_str, 500),
                    refs,
                };

                // Track file changes
                if result.success {
                    tracker.record_success();
                    if is_write_tool(&intent.tool)
                        && let Some(path) = intent.args.get("file_path").and_then(|v| v.as_str())
                    {
                        tracker.record_file_change(path);
                    }
                } else {
                    tracker.record_failure();
                    // Track error modules from file refs
                    for r in &action.refs {
                        if let Some(module) = extract_module_name(r) {
                            tracker.record_error_module(&module);
                        }
                    }
                }

                obs_builder.add_action(action);
                if !result.success {
                    obs_builder.set_stderr(&output_str);
                }
                obs_builder.set_changed_files(tracker.files_changed_since_verify.clone());

                // Build observation pack and send back to R1
                let obs = obs_builder.build();
                let obs_context = obs.to_r1_context();

                messages.push(ChatMessage::User {
                    content: format!(
                        "Tool executed. Here is the observation:\n\n{obs_context}\n\n\
                         Issue your next command."
                    ),
                });

                // Prepare next step
                step += 1;
                obs_builder = ObservationPackBuilder::new(step, repo.clone());

                // If intent requested verification, note it
                if intent.verify_after
                    && let Some(cb) = stream_callback
                {
                    cb(StreamChunk::ContentDelta(format!(
                        "[r1_drive step {step}] verification requested\n"
                    )));
                }
            }

            R1Response::DelegatePatch(dp) => {
                observer.verbose_log(&format!(
                    "r1_drive: step {step} delegate_patch step_id={} task={}",
                    dp.step_id,
                    truncate_output(&dp.task, 80)
                ));
                if let Some(cb) = stream_callback {
                    cb(StreamChunk::ContentDelta(format!(
                        "[r1_drive step {step}] delegating patch: {}\n",
                        truncate_output(&dp.task, 80)
                    )));
                }
                return R1DriveOutcome::DelegatePatch(dp);
            }

            R1Response::Done(done) => {
                observer.verbose_log(&format!(
                    "r1_drive: step {step} done: {}",
                    truncate_output(&done.summary, 80)
                ));
                if let Some(cb) = stream_callback {
                    cb(StreamChunk::ContentDelta(format!(
                        "[r1_drive step {step}] task complete: {}\n",
                        truncate_output(&done.summary, 100)
                    )));
                }
                return R1DriveOutcome::Done(done);
            }

            R1Response::Abort(abort) => {
                observer.verbose_log(&format!("r1_drive: step {step} abort: {}", abort.reason));
                if let Some(cb) = stream_callback {
                    cb(StreamChunk::ContentDelta(format!(
                        "[r1_drive step {step}] aborted: {}\n",
                        abort.reason
                    )));
                }
                return R1DriveOutcome::Abort(abort.reason);
            }
        }
    }
}

/// Parse R1 response with retries on parse failure.
#[allow(clippy::too_many_arguments)]
fn parse_with_retries(
    initial_text: &str,
    config: &R1DriveConfig,
    llm: &(dyn LlmClient + Send + Sync),
    messages: &mut Vec<ChatMessage>,
    observer: &Observer,
    step: u32,
    stream_callback: Option<&deepseek_core::StreamCallback>,
    tracker: &mut FailureTracker,
) -> Result<R1Response, R1ParseError> {
    // First attempt
    match parse_r1_response(initial_text) {
        Ok((resp, _raw)) => Ok(resp),
        Err(e) if config.max_parse_retries == 0 => Err(e),
        Err(first_error) => {
            observer.verbose_log(&format!(
                "r1_drive: step {step} parse error (attempt 1): {first_error}"
            ));

            // Add R1's malformed response + retry prompt
            messages.push(ChatMessage::Assistant {
                content: Some(initial_text.to_string()),
                reasoning_content: None,
                tool_calls: vec![],
            });

            let mut last_error = first_error;
            for retry in 1..=config.max_parse_retries {
                let retry_prompt = r1_retry_prompt(&last_error);
                messages.push(ChatMessage::User {
                    content: retry_prompt,
                });

                if let Some(cb) = stream_callback {
                    cb(StreamChunk::ContentDelta(format!(
                        "[r1_drive step {step}] parse retry {retry}/{}\n",
                        config.max_parse_retries
                    )));
                }

                let retry_request = ChatRequest {
                    model: config.model.clone(),
                    messages: messages.clone(),
                    tools: vec![],
                    tool_choice: ToolChoice::none(),
                    max_tokens: config.max_tokens,
                    temperature: None,
                    thinking: None,
                };

                tracker.r1_steps_used += 1;

                match llm.complete_chat(&retry_request) {
                    Ok(resp) => {
                        let text = if !resp.text.is_empty() {
                            resp.text
                        } else {
                            resp.reasoning_content
                        };

                        match parse_r1_response(&text) {
                            Ok((parsed, _raw)) => {
                                // Remove the malformed assistant message we added
                                // (the successful response will be added by the caller)
                                observer.verbose_log(&format!(
                                    "r1_drive: step {step} parse retry {retry} succeeded"
                                ));
                                return Ok(parsed);
                            }
                            Err(e) => {
                                observer.verbose_log(&format!(
                                    "r1_drive: step {step} parse retry {retry} failed: {e}"
                                ));
                                messages.push(ChatMessage::Assistant {
                                    content: Some(text),
                                    reasoning_content: None,
                                    tool_calls: vec![],
                                });
                                last_error = e;
                            }
                        }
                    }
                    Err(e) => {
                        observer.verbose_log(&format!(
                            "r1_drive: step {step} retry {retry} API error: {e}"
                        ));
                        last_error = R1ParseError::InvalidJson(format!("API error: {e}"));
                    }
                }
            }

            Err(last_error)
        }
    }
}

/// Execute a single R1 tool_intent via the tool host.
fn execute_r1_tool_intent(
    intent: &ToolIntent,
    tool_host: &Arc<dyn ToolHost + Send + Sync>,
    observer: &Observer,
) -> ToolResult {
    let internal_name = r1_tool_to_internal(&intent.tool)
        .unwrap_or("unknown")
        .to_string();

    let tool_call = ToolCall {
        name: internal_name.clone(),
        args: intent.args.clone(),
        requires_approval: false,
    };

    let proposal = tool_host.propose(tool_call);

    if !proposal.approved {
        observer.verbose_log(&format!(
            "r1_drive: tool {} denied by policy",
            internal_name
        ));
        return ToolResult {
            invocation_id: proposal.invocation_id,
            success: false,
            output: serde_json::json!({
                "error": format!("Tool '{}' denied by policy. Choose a different approach.", internal_name)
            }),
        };
    }

    let result = tool_host.execute(ApprovedToolCall {
        invocation_id: proposal.invocation_id,
        call: proposal.call,
    });

    observer.verbose_log(&format!(
        "r1_drive: tool {} result success={}",
        internal_name, result.success
    ));

    result
}

/// Summarize tool intent args for display.
fn summarize_intent_args(intent: &ToolIntent) -> String {
    summarize_args(
        r1_tool_to_internal(&intent.tool).unwrap_or(&intent.tool),
        &intent.args,
    )
}

/// Check if a tool name represents a write operation.
fn is_write_tool(tool: &str) -> bool {
    matches!(
        tool,
        "write_file"
            | "fs_write"
            | "edit_file"
            | "fs_edit"
            | "apply_patch"
            | "patch_apply"
            | "multi_edit"
    )
}

/// Extract a module name from a file reference (e.g. "src/auth/mod.rs:42" → "auth").
fn extract_module_name(file_ref: &str) -> Option<String> {
    let path = file_ref.split(':').next()?;
    let parts: Vec<&str> = path.split('/').collect();
    // Look for a meaningful directory name (skip "src", "lib", "tests")
    for part in parts.iter().rev().skip(1) {
        // skip the filename
        if !matches!(*part, "src" | "lib" | "tests" | "test" | "." | "..") {
            return Some(part.to_string());
        }
    }
    // Fall back to filename without extension
    let filename = parts.last()?;
    filename.split('.').next().map(|s| s.to_string())
}

/// Truncate output string to max length.
fn truncate_output(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let boundary = s.floor_char_boundary(max);
        format!("{}...", &s[..boundary])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_module_from_rust_ref() {
        assert_eq!(
            extract_module_name("src/auth/login.rs:42"),
            Some("auth".to_string())
        );
        assert_eq!(
            extract_module_name("src/config.rs:10"),
            Some("config".to_string())
        );
        assert_eq!(
            extract_module_name("crates/deepseek-agent/src/mode_router.rs:5"),
            Some("deepseek-agent".to_string())
        );
    }

    #[test]
    fn extract_module_from_deep_path() {
        assert_eq!(
            extract_module_name("src/services/auth/handler.rs:15"),
            Some("auth".to_string())
        );
    }

    #[test]
    fn is_write_tool_check() {
        assert!(is_write_tool("write_file"));
        assert!(is_write_tool("fs_edit"));
        assert!(is_write_tool("apply_patch"));
        assert!(is_write_tool("multi_edit"));
        assert!(!is_write_tool("read_file"));
        assert!(!is_write_tool("ripgrep"));
        assert!(!is_write_tool("run_cmd"));
    }

    #[test]
    fn truncate_output_short() {
        assert_eq!(truncate_output("hello", 10), "hello");
    }

    #[test]
    fn truncate_output_long() {
        let long = "a".repeat(200);
        let truncated = truncate_output(&long, 50);
        assert!(truncated.len() <= 54); // 50 + "..."
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn summarize_intent_args_read() {
        let intent = ToolIntent {
            step_id: "S1".into(),
            tool: "read_file".into(),
            args: serde_json::json!({"file_path": "src/lib.rs"}),
            why: "inspect".into(),
            expected: String::new(),
            verify_after: false,
        };
        let summary = summarize_intent_args(&intent);
        assert_eq!(summary, "src/lib.rs");
    }

    #[test]
    fn summarize_intent_args_grep() {
        let intent = ToolIntent {
            step_id: "S2".into(),
            tool: "ripgrep".into(),
            args: serde_json::json!({"pattern": "fn main", "glob": "*.rs"}),
            why: "find entry".into(),
            expected: String::new(),
            verify_after: false,
        };
        let summary = summarize_intent_args(&intent);
        assert_eq!(summary, "fn main in *.rs");
    }
}
