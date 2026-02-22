//! V3 Patch-only mode: V3 produces unified diffs as a dedicated code writer.
//!
//! When R1 issues a `delegate_patch` intent, V3 (deepseek-chat) is called
//! with a focused prompt that instructs it to produce only a unified diff.
//! This module handles:
//! - Building the patch prompt from R1's delegation context
//! - Parsing V3's diff or need_more_context response
//! - Retrying on malformed output
//! - Reading additional context files when requested

use crate::observation::RepoFacts;
use crate::protocol::{
    DelegatePatch, V3PatchResponse, parse_v3_patch_response, v3_patch_retry_prompt,
};
use deepseek_core::{ChatMessage, ChatRequest, StreamChunk, ThinkingConfig, ToolChoice};
use deepseek_llm::LlmClient;
use deepseek_observe::Observer;

/// Outcome of a V3 patch call.
#[derive(Debug, Clone)]
pub enum V3PatchOutcome {
    /// V3 produced a unified diff.
    Diff(String),
    /// V3 could not produce a diff after retries.
    Failed { reason: String },
}

/// V3 patch writer system prompt.
const V3_PATCH_SYSTEM_PROMPT: &str = r#"You are a precise code writer. Your ONLY job is to produce a unified diff that implements the requested change.

## Rules
1. Output ONLY a unified diff. No explanation, no commentary, no markdown fences.
2. Format:
   --- a/path/to/file
   +++ b/path/to/file
   @@ -start,count +start,count @@
   -removed line
   +added line
    context line
3. Include 3 lines of context around each change for accurate patch application.
4. If you need to see more files before writing the patch, respond with:
   {"type":"need_more_context","missing":["path/to/file:start-end", ...]}
5. Produce the MINIMAL change that satisfies the task and constraints.
6. Do NOT modify files outside those specified in constraints/inputs.
7. Preserve existing formatting, indentation, and style.
"#;

/// Configuration for V3 patch calls.
#[derive(Debug, Clone)]
pub struct V3PatchConfig {
    /// Model name for V3 calls (typically "deepseek-chat").
    pub model: String,
    /// Max tokens for V3 patch responses.
    pub max_tokens: u32,
    /// Whether to enable thinking mode for the patch call.
    pub enable_thinking: bool,
    /// Max thinking tokens if thinking is enabled.
    pub max_think_tokens: u32,
    /// Max context requests before giving up.
    pub max_context_requests: u32,
    /// Max retries for malformed diff output.
    pub max_retries: u32,
}

/// Callback for reading file contents when V3 requests more context.
pub type FileReader = Box<dyn Fn(&str) -> Option<String> + Send>;

/// Run the V3 patch writer.
///
/// Takes a delegate_patch instruction from R1 and produces a unified diff.
/// Handles need_more_context requests by reading files and retrying.
pub fn v3_patch_write(
    config: &V3PatchConfig,
    llm: &(dyn LlmClient + Send + Sync),
    observer: &Observer,
    delegate: &DelegatePatch,
    repo: &RepoFacts,
    file_reader: &FileReader,
    stream_callback: Option<&deepseek_core::StreamCallback>,
) -> V3PatchOutcome {
    let mut messages: Vec<ChatMessage> = Vec::new();

    // Build system prompt
    messages.push(ChatMessage::System {
        content: V3_PATCH_SYSTEM_PROMPT.to_string(),
    });

    // Build the task prompt from delegate_patch
    let mut task_prompt = format!("## Task\n{}\n", delegate.task);

    if !delegate.constraints.is_empty() {
        task_prompt.push_str("\n## Constraints\n");
        for c in &delegate.constraints {
            task_prompt.push_str(&format!("- {c}\n"));
        }
    }

    if !delegate.acceptance.is_empty() {
        task_prompt.push_str("\n## Acceptance criteria\n");
        for a in &delegate.acceptance {
            task_prompt.push_str(&format!("- {a}\n"));
        }
    }

    // Include file context
    if !delegate.inputs.relevant_files.is_empty() {
        task_prompt.push_str("\n## Relevant files\n");
        for file_path in &delegate.inputs.relevant_files {
            if let Some(content) = file_reader(file_path) {
                task_prompt.push_str(&format!("\n### {file_path}\n```\n{content}\n```\n"));
            } else {
                task_prompt.push_str(&format!("- {file_path} (could not read)\n"));
            }
        }
    }

    if !delegate.inputs.context_refs.is_empty() {
        task_prompt.push_str("\n## Additional context\n");
        for ctx in &delegate.inputs.context_refs {
            task_prompt.push_str(&format!("- {ctx}\n"));
        }
    }

    task_prompt.push_str(&format!(
        "\n## Repository\n- Language: {}\n- Build system: {}\n",
        repo.language, repo.build_system
    ));

    task_prompt.push_str("\nProduce the unified diff now.");

    messages.push(ChatMessage::User {
        content: task_prompt,
    });

    let mut context_requests: u32 = 0;
    let mut retries: u32 = 0;

    loop {
        let thinking = if config.enable_thinking {
            Some(ThinkingConfig::enabled(config.max_think_tokens.max(4096)))
        } else {
            None
        };

        let request = ChatRequest {
            model: config.model.clone(),
            messages: messages.clone(),
            tools: vec![],
            tool_choice: ToolChoice::none(),
            max_tokens: config.max_tokens,
            temperature: if config.enable_thinking {
                None
            } else {
                Some(0.0)
            },
            thinking,
        };

        observer.verbose_log(&format!(
            "v3_patch: calling V3 model={} messages={} thinking={}",
            config.model,
            messages.len(),
            config.enable_thinking
        ));

        if let Some(cb) = stream_callback {
            cb(StreamChunk::ContentDelta(
                "[v3_patch] generating diff...\n".to_string(),
            ));
        }

        let response = match llm.complete_chat(&request) {
            Ok(r) => r,
            Err(e) => {
                observer.verbose_log(&format!("v3_patch: API error: {e}"));
                return V3PatchOutcome::Failed {
                    reason: format!("V3 API error: {e}"),
                };
            }
        };

        let response_text = &response.text;
        if response_text.trim().is_empty() {
            observer.verbose_log("v3_patch: empty V3 response");
            retries += 1;
            if retries > config.max_retries {
                return V3PatchOutcome::Failed {
                    reason: "V3 returned empty response after retries".to_string(),
                };
            }
            messages.push(ChatMessage::Assistant {
                content: Some(String::new()),
                reasoning_content: None,
                tool_calls: vec![],
            });
            messages.push(ChatMessage::User {
                content: v3_patch_retry_prompt().to_string(),
            });
            continue;
        }

        let parsed = parse_v3_patch_response(response_text);

        match parsed {
            V3PatchResponse::UnifiedDiff(diff) => {
                // Validate the diff looks reasonable
                if diff.contains("---") && diff.contains("+++") && diff.contains("@@") {
                    observer
                        .verbose_log(&format!("v3_patch: got valid diff ({} bytes)", diff.len()));
                    if let Some(cb) = stream_callback {
                        cb(StreamChunk::ContentDelta(format!(
                            "[v3_patch] diff produced ({} bytes)\n",
                            diff.len()
                        )));
                    }
                    return V3PatchOutcome::Diff(diff);
                }

                // Diff doesn't look valid — retry
                retries += 1;
                if retries > config.max_retries {
                    // Return it anyway — the caller can try to apply it
                    observer.verbose_log("v3_patch: diff doesn't look valid but retries exhausted");
                    return V3PatchOutcome::Diff(diff);
                }

                observer.verbose_log(&format!(
                    "v3_patch: diff doesn't contain required markers, retry {retries}"
                ));
                messages.push(ChatMessage::Assistant {
                    content: Some(response_text.clone()),
                    reasoning_content: None,
                    tool_calls: vec![],
                });
                messages.push(ChatMessage::User {
                    content: v3_patch_retry_prompt().to_string(),
                });
            }

            V3PatchResponse::NeedMoreContext(nmc) => {
                context_requests += 1;
                if context_requests > config.max_context_requests {
                    observer.verbose_log("v3_patch: max context requests exceeded");
                    return V3PatchOutcome::Failed {
                        reason: format!(
                            "V3 requested context {} times (max {})",
                            context_requests, config.max_context_requests
                        ),
                    };
                }

                observer.verbose_log(&format!(
                    "v3_patch: context request #{context_requests}: {:?}",
                    nmc.missing
                ));
                if let Some(cb) = stream_callback {
                    cb(StreamChunk::ContentDelta(format!(
                        "[v3_patch] reading {} additional file(s)...\n",
                        nmc.missing.len()
                    )));
                }

                // Add V3's response to history
                messages.push(ChatMessage::Assistant {
                    content: Some(response_text.clone()),
                    reasoning_content: None,
                    tool_calls: vec![],
                });

                // Read requested files and provide them
                let mut context_content =
                    String::from("Here is the additional context you requested:\n\n");
                for file_spec in &nmc.missing {
                    let (path, range) = parse_file_spec(file_spec);
                    if let Some(content) = file_reader(path) {
                        let content = if let Some((start, end)) = range {
                            extract_line_range(&content, start, end)
                        } else {
                            content
                        };
                        context_content
                            .push_str(&format!("### {file_spec}\n```\n{content}\n```\n\n"));
                    } else {
                        context_content.push_str(&format!("### {file_spec}\n(file not found)\n\n"));
                    }
                }
                context_content.push_str("Now produce the unified diff.");

                messages.push(ChatMessage::User {
                    content: context_content,
                });
            }
        }
    }
}

/// Parse a file spec like "src/lib.rs:10-50" into (path, optional (start, end)).
fn parse_file_spec(spec: &str) -> (&str, Option<(usize, usize)>) {
    if let Some((path, range)) = spec.rsplit_once(':') {
        if let Some((start, end)) = range.split_once('-') {
            if let (Ok(s), Ok(e)) = (start.parse::<usize>(), end.parse::<usize>()) {
                return (path, Some((s, e)));
            }
        }
    }
    (spec, None)
}

/// Extract lines from content by 1-indexed range.
fn extract_line_range(content: &str, start: usize, end: usize) -> String {
    content
        .lines()
        .enumerate()
        .filter(|(i, _)| {
            let line_num = i + 1;
            line_num >= start && line_num <= end
        })
        .map(|(_, line)| line)
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_file_spec_with_range() {
        let (path, range) = parse_file_spec("src/lib.rs:10-50");
        assert_eq!(path, "src/lib.rs");
        assert_eq!(range, Some((10, 50)));
    }

    #[test]
    fn parse_file_spec_without_range() {
        let (path, range) = parse_file_spec("src/lib.rs");
        assert_eq!(path, "src/lib.rs");
        assert_eq!(range, None);
    }

    #[test]
    fn parse_file_spec_with_line_only() {
        // "src/lib.rs:42" — no dash, so no range
        let (path, range) = parse_file_spec("src/lib.rs:42");
        assert_eq!(path, "src/lib.rs:42");
        assert_eq!(range, None);
    }

    #[test]
    fn extract_line_range_basic() {
        let content = "line1\nline2\nline3\nline4\nline5";
        let extracted = extract_line_range(content, 2, 4);
        assert_eq!(extracted, "line2\nline3\nline4");
    }

    #[test]
    fn extract_line_range_out_of_bounds() {
        let content = "line1\nline2\nline3";
        let extracted = extract_line_range(content, 2, 10);
        assert_eq!(extracted, "line2\nline3");
    }

    #[test]
    fn v3_patch_system_prompt_contains_rules() {
        assert!(V3_PATCH_SYSTEM_PROMPT.contains("unified diff"));
        assert!(V3_PATCH_SYSTEM_PROMPT.contains("need_more_context"));
        assert!(V3_PATCH_SYSTEM_PROMPT.contains("MINIMAL"));
    }
}
