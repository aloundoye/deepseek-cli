use crate::architect::ArchitectPlan;
use anyhow::{Result, anyhow};
use deepseek_core::{ChatMessage, ChatRequest, ToolChoice};

#[derive(Debug, Clone)]
pub struct EditorFileContext {
    pub path: String,
    pub content: String,
    pub partial: bool,
    pub base_hash: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FileRequest {
    pub path: String,
    pub range: Option<(usize, usize)>,
}

#[derive(Debug, Clone)]
pub enum EditorResponse {
    Diff(String),
    NeedContext(Vec<FileRequest>),
}

pub struct EditorInput<'a> {
    pub user_prompt: &'a str,
    pub iteration: u64,
    pub plan: &'a ArchitectPlan,
    pub files: &'a [EditorFileContext],
    pub verify_feedback: Option<&'a str>,
    pub apply_feedback: Option<&'a str>,
    pub max_diff_bytes: usize,
    pub debug_context: bool,
    pub chat_history: &'a [ChatMessage],
}

const EDITOR_SYSTEM_PROMPT: &str = r#"You are Editor (code writer).

Return ONLY one of:
1) A unified diff payload.
2) NEED_CONTEXT|path[:start-end] lines when required context is missing.

Rules:
- Never output commentary, markdown, JSON, or tool calls.
- Use standard unified diff headers (--- / +++ / @@).
- Modify only files explicitly listed by Architect.
"#;

pub fn run_editor(
    engine: &crate::AgentEngine,
    input: &EditorInput<'_>,
    retries: usize,
) -> Result<EditorResponse> {
    if input.debug_context {
        let total_bytes = input
            .files
            .iter()
            .map(|file| file.content.len())
            .sum::<usize>();
        let partial_files = input.files.iter().filter(|file| file.partial).count();
        eprintln!(
            "[debug-context] intent=EditCode phase=Editor mode=Code iteration={} files={} total_bytes={} partial_files={} verify_feedback={} apply_feedback={}",
            input.iteration,
            input.files.len(),
            total_bytes,
            partial_files,
            input.verify_feedback.is_some(),
            input.apply_feedback.is_some()
        );
    }
    let mut messages = vec![ChatMessage::System {
        content: EDITOR_SYSTEM_PROMPT.to_string(),
    }];
    messages.extend(input.chat_history.iter().cloned());
    messages.push(ChatMessage::User {
        content: build_editor_prompt(input),
    });

    for attempt in 0..=retries {
        deepseek_core::strip_prior_reasoning_content(&mut messages);
        let req = ChatRequest {
            model: engine.cfg.llm.base_model.clone(),
            messages: messages.clone(),
            tools: vec![],
            tool_choice: ToolChoice::none(),
            max_tokens: 8192,
            temperature: Some(0.0),
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            logprobs: None,
            top_logprobs: None,
            thinking: None,
            images: vec![],
            response_format: None,
        };

        let response = engine.llm.complete_chat(&req)?;
        if let Some(usage) = &response.usage {
            engine.record_usage(&req.model, usage);
        }
        match parse_editor_response(&response.text, input.max_diff_bytes) {
            Ok(editor) => return Ok(editor),
            Err(err) => {
                if attempt == retries {
                    return Err(err);
                }
                messages.push(ChatMessage::Assistant {
                    content: Some(response.text),
                    reasoning_content: None,
                    tool_calls: vec![],
                });
                messages.push(ChatMessage::User {
                    content: "Output invalid. Return ONLY unified diff OR NEED_CONTEXT lines."
                        .to_string(),
                });
            }
        }
    }

    Err(anyhow!("editor failed to return valid diff"))
}

pub fn parse_editor_response(text: &str, max_diff_bytes: usize) -> Result<EditorResponse> {
    let normalized = text.trim();
    if normalized.is_empty() {
        return Err(anyhow!("empty editor response"));
    }

    if normalized
        .lines()
        .all(|line| line.trim().starts_with("NEED_CONTEXT|"))
    {
        let mut requests = Vec::new();
        for line in normalized.lines() {
            let body = line
                .trim()
                .strip_prefix("NEED_CONTEXT|")
                .ok_or_else(|| anyhow!("invalid NEED_CONTEXT line"))?;
            requests.push(parse_file_request(body)?);
        }
        if requests.is_empty() {
            return Err(anyhow!("editor returned empty NEED_CONTEXT request"));
        }
        return Ok(EditorResponse::NeedContext(requests));
    }

    let mut diff = strip_fences(text);
    if !diff.ends_with('\n') {
        diff.push('\n');
    }
    if diff.len() > max_diff_bytes {
        return Err(anyhow!(
            "editor diff exceeds max size ({} > {})",
            diff.len(),
            max_diff_bytes
        ));
    }
    if !diff.contains("--- ") || !diff.contains("+++ ") || !diff.contains("@@") {
        return Err(anyhow!("editor output is not a valid unified diff"));
    }

    Ok(EditorResponse::Diff(diff))
}

fn parse_file_request(body: &str) -> Result<FileRequest> {
    let body = body.trim();
    if body.is_empty() {
        return Err(anyhow!("empty NEED_CONTEXT path"));
    }
    if let Some((path, range)) = body.rsplit_once(':')
        && let Some((start, end)) = range.split_once('-')
        && let (Ok(start), Ok(end)) = (start.trim().parse::<usize>(), end.trim().parse::<usize>())
    {
        return Ok(FileRequest {
            path: path.trim().to_string(),
            range: Some((start, end)),
        });
    }

    Ok(FileRequest {
        path: body.to_string(),
        range: None,
    })
}

fn strip_fences(text: &str) -> String {
    if !text.trim_start().starts_with("```") {
        return text.to_string();
    }
    let mut lines = text.trim_start().lines();
    let _ = lines.next();
    let content: Vec<&str> = lines
        .take_while(|line| !line.trim().starts_with("```"))
        .collect();
    if content.is_empty() {
        text.to_string()
    } else {
        let mut joined = content.join("\n");
        joined.push('\n');
        joined
    }
}

fn build_editor_prompt(input: &EditorInput<'_>) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Iteration: {}\n\nUser request:\n{}\n\n",
        input.iteration, input.user_prompt
    ));
    out.push_str("Architect steps:\n");
    for step in &input.plan.steps {
        out.push_str("- ");
        out.push_str(step);
        out.push('\n');
    }
    out.push_str("\nArchitect file intents:\n");
    for file in &input.plan.files {
        out.push_str(&format!("- {} :: {}\n", file.path, file.intent));
    }

    if let Some(apply) = input.apply_feedback {
        out.push_str("\nLast apply failure:\n");
        out.push_str(apply);
        out.push('\n');
    }
    if let Some(verify) = input.verify_feedback {
        out.push_str("\nLast verify failure:\n");
        out.push_str(verify);
        out.push('\n');
    }

    out.push_str("\nFile contents (truth source):\n");
    for file in input.files {
        out.push_str(&format!("\n### {}\n", file.path));
        if file.partial {
            out.push_str("[partial context]\n");
        }
        out.push_str("```");
        out.push('\n');
        out.push_str(&file.content);
        if !file.content.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("```\n");
    }

    out.push_str("\nNow return unified diff OR NEED_CONTEXT lines.");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_need_file() {
        let parsed = parse_editor_response("NEED_CONTEXT|src/lib.rs:1-40", 1024).expect("parse");
        match parsed {
            EditorResponse::NeedContext(reqs) => {
                assert_eq!(reqs.len(), 1);
                assert_eq!(reqs[0].path, "src/lib.rs");
                assert_eq!(reqs[0].range, Some((1, 40)));
            }
            _ => panic!("expected NeedFile"),
        }
    }

    #[test]
    fn parse_diff() {
        let diff = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-old\n+new\n";
        let parsed = parse_editor_response(diff, 4096).expect("parse");
        match parsed {
            EditorResponse::Diff(value) => assert!(value.contains("@@")),
            _ => panic!("expected diff"),
        }
    }

    #[test]
    fn parse_need_context_without_range() {
        let parsed =
            parse_editor_response("NEED_CONTEXT|src/main.rs", 1024).expect("parse");
        match parsed {
            EditorResponse::NeedContext(reqs) => {
                assert_eq!(reqs.len(), 1);
                assert_eq!(reqs[0].path, "src/main.rs");
                assert_eq!(reqs[0].range, None);
            }
            _ => panic!("expected NeedContext"),
        }
    }

    #[test]
    fn parse_need_context_multi_file() {
        let input = "NEED_CONTEXT|src/lib.rs:1-50\nNEED_CONTEXT|src/util.rs\nNEED_CONTEXT|tests/main.rs:10-20";
        let parsed = parse_editor_response(input, 1024).expect("parse");
        match parsed {
            EditorResponse::NeedContext(reqs) => {
                assert_eq!(reqs.len(), 3);
                assert_eq!(reqs[0].path, "src/lib.rs");
                assert_eq!(reqs[0].range, Some((1, 50)));
                assert_eq!(reqs[1].path, "src/util.rs");
                assert_eq!(reqs[1].range, None);
                assert_eq!(reqs[2].path, "tests/main.rs");
                assert_eq!(reqs[2].range, Some((10, 20)));
            }
            _ => panic!("expected NeedContext"),
        }
    }

    #[test]
    fn parse_empty_response_rejected() {
        let err = parse_editor_response("", 4096).unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn parse_whitespace_only_rejected() {
        let err = parse_editor_response("   \n  \n  ", 4096).unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn parse_diff_rejects_missing_markers() {
        let err = parse_editor_response("just some plain text output", 4096).unwrap_err();
        assert!(err.to_string().contains("not a valid unified diff"));
    }

    #[test]
    fn parse_diff_rejects_oversized() {
        let diff = format!(
            "--- a/big.rs\n+++ b/big.rs\n@@ -1 +1 @@\n-old\n+{}\n",
            "x".repeat(5000)
        );
        let err = parse_editor_response(&diff, 1024).unwrap_err();
        assert!(err.to_string().contains("exceeds max size"));
    }

    #[test]
    fn strip_fences_removes_markdown() {
        let input = "```diff\n--- a/f.rs\n+++ b/f.rs\n@@ -1 +1 @@\n-a\n+b\n```\n";
        let stripped = strip_fences(input);
        assert!(stripped.starts_with("--- a/f.rs"));
        assert!(!stripped.contains("```"));
    }

    #[test]
    fn strip_fences_noop_for_plain_diff() {
        let input = "--- a/f.rs\n+++ b/f.rs\n@@ -1 +1 @@\n-a\n+b\n";
        let stripped = strip_fences(input);
        assert_eq!(stripped, input);
    }

    #[test]
    fn parse_diff_with_fences_accepted() {
        let input = "```diff\n--- a/f.rs\n+++ b/f.rs\n@@ -1 +1 @@\n-a\n+b\n```\n";
        let parsed = parse_editor_response(input, 4096).expect("parse");
        match parsed {
            EditorResponse::Diff(value) => {
                assert!(value.contains("@@"));
                assert!(!value.contains("```"));
            }
            _ => panic!("expected Diff"),
        }
    }

    #[test]
    fn parse_diff_appends_trailing_newline() {
        let diff = "--- a/f.rs\n+++ b/f.rs\n@@ -1 +1 @@\n-old\n+new";
        let parsed = parse_editor_response(diff, 4096).expect("parse");
        match parsed {
            EditorResponse::Diff(value) => assert!(value.ends_with('\n')),
            _ => panic!("expected Diff"),
        }
    }

    #[test]
    fn parse_file_request_with_range() {
        let req = parse_file_request("src/lib.rs:10-50").expect("parse");
        assert_eq!(req.path, "src/lib.rs");
        assert_eq!(req.range, Some((10, 50)));
    }

    #[test]
    fn parse_file_request_without_range() {
        let req = parse_file_request("src/lib.rs").expect("parse");
        assert_eq!(req.path, "src/lib.rs");
        assert_eq!(req.range, None);
    }

    #[test]
    fn parse_file_request_empty_rejected() {
        let err = parse_file_request("").unwrap_err();
        assert!(err.to_string().contains("empty"));
    }
}
