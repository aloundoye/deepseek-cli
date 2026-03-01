use anyhow::{Result, anyhow};
use codingbuddy_core::{AppConfig, ChatMessage, ChatRequest, StreamCallback, ToolChoice};
use codingbuddy_llm::LlmClient;
use std::path::Path;

use crate::gather_context;
use crate::{ChatMode, ChatOptions};

pub fn analyze(
    llm: &(dyn LlmClient + Send + Sync),
    cfg: &AppConfig,
    workspace: &Path,
    prompt: &str,
    options: &ChatOptions,
    stream_cb: Option<StreamCallback>,
) -> Result<String> {
    let workspace = options.repo_root_override.as_deref().unwrap_or(workspace);
    let bootstrap = gather_context::gather_for_prompt(
        workspace,
        cfg,
        prompt,
        options.mode,
        &options.additional_dirs,
        options.repo_root_override.as_deref(),
    );

    if bootstrap.repoish && bootstrap.repo_root.is_none() {
        return Err(anyhow!(
            "{}",
            bootstrap
                .unavailable_reason
                .as_deref()
                .unwrap_or("No repository detected. Run from project root or pass --repo <path>.")
        ));
    }

    if gather_context::debug_context_enabled(options.debug_context) {
        eprintln!("{}", bootstrap.debug_digest("InspectRepo", options.mode));
    }

    let system_prompt =
        build_system_prompt(options, bootstrap.repoish, bootstrap.vague_codebase_prompt);
    let user_prompt = build_user_prompt(prompt, &bootstrap);

    let mut messages = vec![ChatMessage::System {
        content: system_prompt,
    }];
    messages.extend(options.chat_history.iter().cloned());
    messages.push(ChatMessage::User {
        content: user_prompt,
    });

    let (model, is_reasoner) = if options.force_max_think {
        let m = cfg.llm.max_think_model.clone();
        let r = codingbuddy_core::is_reasoner_model(&m);
        (m, r)
    } else {
        (cfg.llm.base_model.clone(), false)
    };

    let enforce_profile_shape = matches!(options.mode, ChatMode::Ask | ChatMode::Context)
        && bootstrap.enabled
        && bootstrap.repoish;
    let max_repairs = if enforce_profile_shape { 1 } else { 0 };

    for attempt in 0..=max_repairs {
        codingbuddy_core::strip_prior_reasoning_content(&mut messages);
        let request = ChatRequest {
            model: model.clone(),
            messages: messages.clone(),
            tools: vec![],
            tool_choice: ToolChoice::none(),
            max_tokens: if is_reasoner {
                codingbuddy_core::CODINGBUDDY_REASONER_MAX_OUTPUT_TOKENS
            } else {
                8192
            },
            // Reasoner rejects temperature; deepseek-chat needs 0.0 for determinism
            temperature: if options.force_max_think { None } else { Some(0.0) },
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            logprobs: None,
            top_logprobs: None,
            // Reasoner thinks natively — ThinkingConfig is only for deepseek-chat
            thinking: if options.force_max_think && !is_reasoner {
                Some(codingbuddy_core::ThinkingConfig::enabled(16_384))
            } else {
                None
            },
            images: options.images.clone(),
            response_format: None,
        };

        // First attempt: use streaming so the renderer can display tokens in real-time.
        // Repair attempts fall back to blocking since the first-pass stream already completed.
        let response = if attempt == 0 {
            if let Some(ref cb) = stream_cb {
                llm.complete_chat_streaming(&request, cb.clone())?
            } else {
                llm.complete_chat(&request)?
            }
        } else {
            // Repair pass: stream already done, use blocking to avoid double output
            llm.complete_chat(&request)?
        };
        let violation = if enforce_profile_shape {
            validate_analysis_shape(&response.text, bootstrap.vague_codebase_prompt)
        } else {
            None
        };

        if let Some(reason) = violation
            && attempt < max_repairs
        {
            messages.push(ChatMessage::Assistant {
                content: Some(response.text),
                reasoning_content: None,
                tool_calls: vec![],
            });
            messages.push(ChatMessage::User {
                content: repair_prompt(bootstrap.vague_codebase_prompt, &reason),
            });
            continue;
        }

        return Ok(response.text);
    }

    Ok(String::new())
}

fn build_system_prompt(
    options: &ChatOptions,
    repoish_bootstrap: bool,
    vague_codebase_prompt: bool,
) -> String {
    if let Some(ref override_prompt) = options.system_prompt_override {
        return override_prompt.clone();
    }

    let mut prompt = if matches!(options.mode, ChatMode::Ask | ChatMode::Context)
        && repoish_bootstrap
    {
        let role = if matches!(options.mode, ChatMode::Context) {
            "You are a codebase analysis specialist. Focus exclusively on understanding architecture, dependencies, patterns, and structure. Do not suggest code changes or generate code."
        } else {
            "You are a precise software engineering assistant."
        };
        let mut base = vec![
            role,
            "Assume you are operating inside the current repository unless explicitly told otherwise.",
            "Use the project context below to inform your analysis. Respond immediately with findings.",
            "Output sections in this exact order: Initial Analysis, Key Findings, Follow-ups.",
            "Follow-ups must be targeted and capped to at most 2 questions.",
            "Do not ask broad requests for more context before giving initial analysis.",
        ];
        if vague_codebase_prompt {
            base.push(
                "For vague codebase audit prompts, summarize baseline audit findings first and ask exactly 1 focused follow-up tied to the highest-risk area.",
            );
        }
        base.join("\n")
    } else {
        "You are a precise software engineering assistant.".to_string()
    };

    if let Some(ref append) = options.system_prompt_append {
        prompt.push('\n');
        prompt.push_str(append);
    }

    prompt
}

fn build_user_prompt(prompt: &str, bootstrap: &gather_context::AutoContextBootstrap) -> String {
    if !bootstrap.enabled || !bootstrap.repoish || bootstrap.packet.trim().is_empty() {
        return prompt.to_string();
    }

    format!("{prompt}\n\n---\n{}", bootstrap.packet.trim_end())
}

fn validate_analysis_shape(text: &str, strict_single_followup: bool) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    if !(lower.contains("initial analysis") && lower.contains("key findings")) {
        return Some("missing required sections".to_string());
    }

    let followups = extract_followups(text);
    if strict_single_followup {
        if followups.len() != 1 {
            return Some(format!(
                "expected exactly 1 focused follow-up, got {}",
                followups.len()
            ));
        }
    } else if followups.len() > 2 {
        return Some(format!(
            "follow-up budget exceeded; max 2, got {}",
            followups.len()
        ));
    }

    if lower.contains("need more details") || lower.contains("need more information") {
        return Some("response asked generic clarification before analysis".to_string());
    }

    None
}

fn extract_followups(text: &str) -> Vec<String> {
    let mut in_followups = false;
    let mut followups = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        if lower.starts_with("follow-ups") || lower.starts_with("followups") {
            in_followups = true;
            continue;
        }
        if in_followups
            && (lower.starts_with("initial analysis") || lower.starts_with("key findings"))
        {
            break;
        }

        if !in_followups || trimmed.is_empty() {
            continue;
        }

        let looks_like_followup = trimmed.starts_with('-')
            || trimmed.starts_with('*')
            || trimmed.chars().next().is_some_and(|ch| ch.is_ascii_digit())
            || trimmed.ends_with('?');
        if looks_like_followup {
            followups.push(trimmed.to_string());
        }
    }

    followups
}

fn repair_prompt(strict_single_followup: bool, reason: &str) -> String {
    if strict_single_followup {
        format!(
            "Repair required: {}. Re-answer using exactly these sections and order:\nInitial Analysis\nKey Findings\nFollow-ups\nInclude exactly 1 focused follow-up question in Follow-ups.",
            reason
        )
    } else {
        format!(
            "Repair required: {}. Re-answer using exactly these sections and order:\nInitial Analysis\nKey Findings\nFollow-ups\nInclude no more than 2 focused follow-up questions.",
            reason
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_followups_detects_bullets() {
        let text = "Initial Analysis\nA\nKey Findings\nB\nFollow-ups\n- one?\n- two?\n";
        let followups = extract_followups(text);
        assert_eq!(followups.len(), 2);
    }

    #[test]
    fn validator_enforces_limits() {
        let bad = "Initial Analysis\nA\nKey Findings\nB\nFollow-ups\n- q1\n- q2\n- q3\n";
        assert!(validate_analysis_shape(bad, false).is_some());

        let ok = "Initial Analysis\nA\nKey Findings\nB\nFollow-ups\n- q1\n";
        assert!(validate_analysis_shape(ok, true).is_none());
    }

    #[test]
    fn context_mode_system_prompt_excludes_code_gen() {
        let options = ChatOptions {
            mode: ChatMode::Context,
            ..Default::default()
        };
        let prompt = build_system_prompt(&options, true, false);
        assert!(
            prompt.contains("codebase analysis specialist"),
            "Context mode should use analysis-specialist role"
        );
        assert!(
            prompt.contains("Do not suggest code changes"),
            "Context mode should prohibit code generation"
        );
        assert!(
            !prompt.contains("precise software engineering assistant"),
            "Context mode should not use generic assistant role"
        );
    }

    #[test]
    fn ask_mode_system_prompt_uses_generic_role() {
        let options = ChatOptions {
            mode: ChatMode::Ask,
            ..Default::default()
        };
        let prompt = build_system_prompt(&options, true, false);
        assert!(
            prompt.contains("precise software engineering assistant"),
            "Ask mode should use generic assistant role"
        );
        assert!(
            !prompt.contains("codebase analysis specialist"),
            "Ask mode should not use context-specialist role"
        );
    }
}
