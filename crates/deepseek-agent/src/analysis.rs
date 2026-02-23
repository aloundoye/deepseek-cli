use anyhow::Result;
use deepseek_core::{AppConfig, ChatMessage, ChatRequest, ToolChoice};
use deepseek_llm::LlmClient;

use crate::ChatOptions;

pub fn analyze(
    llm: &(dyn LlmClient + Send + Sync),
    cfg: &AppConfig,
    prompt: &str,
    options: &ChatOptions,
) -> Result<String> {
    let mut messages = Vec::new();

    if let Some(ref override_prompt) = options.system_prompt_override {
        messages.push(ChatMessage::System {
            content: override_prompt.clone(),
        });
    } else if let Some(ref append) = options.system_prompt_append {
        messages.push(ChatMessage::System {
            content: format!(
                "You are a precise software engineering assistant.\n{}",
                append
            ),
        });
    }

    messages.push(ChatMessage::User {
        content: prompt.to_string(),
    });

    let model = if options.force_max_think {
        cfg.llm.max_think_model.clone()
    } else {
        cfg.llm.base_model.clone()
    };

    let request = ChatRequest {
        model,
        messages,
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
    Ok(response.text)
}
