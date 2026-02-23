use anyhow::Result;
use deepseek_core::{AppConfig, ChatMessage, ChatRequest, ToolChoice};
use deepseek_llm::LlmClient;
use std::path::Path;

use crate::gather_context;
use crate::{ChatMode, ChatOptions};

pub fn analyze(
    llm: &(dyn LlmClient + Send + Sync),
    cfg: &AppConfig,
    workspace: &Path,
    prompt: &str,
    options: &ChatOptions,
) -> Result<String> {
    let bootstrap = gather_context::gather_for_prompt(
        workspace,
        cfg,
        prompt,
        options.mode,
        &options.additional_dirs,
    );

    let system_prompt =
        build_system_prompt(options, bootstrap.repoish, bootstrap.vague_codebase_prompt);
    let user_prompt = build_user_prompt(prompt, &bootstrap);

    let mut messages = vec![
        ChatMessage::System {
            content: system_prompt,
        },
        ChatMessage::User {
            content: user_prompt,
        },
    ];

    let model = if options.force_max_think {
        cfg.llm.max_think_model.clone()
    } else {
        cfg.llm.base_model.clone()
    };

    let enforce_profile_shape = matches!(options.mode, ChatMode::Ask | ChatMode::Context)
        && bootstrap.enabled
        && bootstrap.repoish;
    let max_repairs = if enforce_profile_shape { 1 } else { 0 };

    for attempt in 0..=max_repairs {
        let request = ChatRequest {
            model: model.clone(),
            messages: messages.clone(),
            tools: vec![],
            tool_choice: ToolChoice::none(),
            max_tokens: 8192,
            temperature: if options.force_max_think {
                None
            } else {
                Some(0.0)
            },
            thinking: if options.force_max_think {
                Some(deepseek_core::ThinkingConfig::enabled(16_384))
            } else {
                None
            },
        };

        let response = llm.complete_chat(&request)?;
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
        let mut base = vec![
            "You are a precise software engineering assistant.",
            "Assume you are operating inside the current repository unless explicitly told otherwise.",
            "Use AUTO_CONTEXT_BOOTSTRAP_V1 as primary context and provide analysis immediately.",
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

    format!("USER_REQUEST:\n{prompt}\n\n{}", bootstrap.packet.trim_end())
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
}
