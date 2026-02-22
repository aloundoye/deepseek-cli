//! R1 Consultation: V3 asks R1 for targeted advice on specific subproblems.
//!
//! Unlike R1DriveTools (binary escalation), consultations are lightweight:
//! - V3 keeps control and tools throughout
//! - R1 returns text advice (no JSON intents, no tool execution)
//! - Consultation is scoped to a specific question
//! - R1's advice is injected into V3's conversation as a tool result

use anyhow::Result;
use deepseek_core::{ChatMessage, ChatRequest, StreamCallback, StreamChunk, ToolChoice};
use deepseek_llm::LlmClient;

/// A consultation request from V3 to R1.
#[derive(Debug, Clone)]
pub struct ConsultationRequest {
    /// The specific question V3 wants R1 to reason about.
    pub question: String,
    /// Relevant context (file contents, error messages, etc.).
    pub context: String,
    /// What kind of advice is needed.
    pub consultation_type: ConsultationType,
}

/// The type of reasoning assistance needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsultationType {
    /// Analyze an error and suggest a fix strategy.
    ErrorAnalysis,
    /// Design an approach for a complex change.
    ArchitectureAdvice,
    /// Review a plan before execution.
    PlanReview,
    /// Decompose a complex task into ordered steps.
    TaskDecomposition,
}

impl ConsultationType {
    fn label(self) -> &'static str {
        match self {
            Self::ErrorAnalysis => "Error Analysis",
            Self::ArchitectureAdvice => "Architecture Decision",
            Self::PlanReview => "Plan Review",
            Self::TaskDecomposition => "Task Decomposition",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "error_analysis" => Self::ErrorAnalysis,
            "architecture_advice" => Self::ArchitectureAdvice,
            "plan_review" => Self::PlanReview,
            "task_decomposition" => Self::TaskDecomposition,
            _ => Self::ErrorAnalysis,
        }
    }
}

/// Result of an R1 consultation.
#[derive(Debug, Clone)]
pub struct ConsultationResult {
    /// The actionable advice from R1.
    pub advice: String,
    /// R1's reasoning chain (may be empty for some models).
    pub reasoning: String,
}

/// System prompt for R1 consultations.
///
/// Key design: R1 returns plain text analysis, NOT JSON intents or tool
/// commands. This keeps the consultation lightweight and avoids the
/// fragile JSON parsing that plagues R1DriveTools.
const R1_CONSULTATION_PROMPT: &str = "\
You are a senior software architect providing focused, expert advice.

A coding agent is asking for your analysis on a specific subproblem. Your job \
is to provide clear, actionable guidance.

## Response Format

Structure your response as:

**ANALYSIS**: What is happening and why. Identify root causes, not just symptoms.

**RECOMMENDATION**: Concrete steps the agent should take. Be specific about \
which files to modify, what patterns to use, and what order to do things in.

**RISKS**: What could go wrong with the recommended approach. Mention edge \
cases, backwards compatibility concerns, or performance implications.

## Rules

1. Be concise and actionable. The agent will use your advice to guide tool calls.
2. Do NOT output JSON objects, tool commands, or structured intents.
3. Do NOT write implementation code — describe the strategy.
4. Focus on the specific question asked, not general best practices.
5. If the question involves multiple files, mention each file path explicitly.
";

/// Consult R1 for advice on a specific subproblem.
///
/// This is a single-shot API call — no tool loop, no JSON parsing.
/// R1 reasons about the question and returns text advice.
pub fn consult_r1(
    llm: &dyn LlmClient,
    model: &str,
    request: &ConsultationRequest,
    stream_callback: Option<&StreamCallback>,
) -> Result<ConsultationResult> {
    let type_label = request.consultation_type.label();

    // Notify TUI that consultation is starting
    if let Some(cb) = stream_callback {
        cb(StreamChunk::ToolCallStart {
            tool_name: "think_deeply".to_string(),
            args_summary: type_label.to_string(),
        });
    }

    let user_content = format!(
        "## Consultation Type: {type_label}\n\n\
         ## Question\n{}\n\n\
         ## Context\n{}",
        request.question, request.context
    );

    let messages = vec![
        ChatMessage::System {
            content: R1_CONSULTATION_PROMPT.to_string(),
        },
        ChatMessage::User {
            content: user_content,
        },
    ];

    let chat_request = ChatRequest {
        model: model.to_string(),
        messages,
        tools: vec![],
        tool_choice: ToolChoice::none(),
        max_tokens: 4096,
        temperature: None, // R1 manages its own temperature
        thinking: None,    // R1 has built-in chain-of-thought
    };

    let response = llm.complete_chat(&chat_request)?;

    // R1 may return advice in `text` or `reasoning_content` (or both).
    let advice = if !response.text.is_empty() {
        response.text.clone()
    } else if !response.reasoning_content.is_empty() {
        // If R1 only populated reasoning_content, use that as the advice.
        response.reasoning_content.clone()
    } else {
        "R1 returned no advice for this question.".to_string()
    };

    // Notify TUI that consultation is complete
    if let Some(cb) = stream_callback {
        cb(StreamChunk::ToolCallEnd {
            tool_name: "think_deeply".to_string(),
            duration_ms: 0, // caller can track timing
            success: !advice.is_empty(),
            summary: format!("{} chars", advice.len()),
        });
    }

    Ok(ConsultationResult {
        advice,
        reasoning: response.reasoning_content,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consultation_type_from_str_known() {
        assert_eq!(
            ConsultationType::from_str("error_analysis"),
            ConsultationType::ErrorAnalysis
        );
        assert_eq!(
            ConsultationType::from_str("architecture_advice"),
            ConsultationType::ArchitectureAdvice
        );
        assert_eq!(
            ConsultationType::from_str("plan_review"),
            ConsultationType::PlanReview
        );
        assert_eq!(
            ConsultationType::from_str("task_decomposition"),
            ConsultationType::TaskDecomposition
        );
    }

    #[test]
    fn consultation_type_from_str_unknown_defaults() {
        assert_eq!(
            ConsultationType::from_str("unknown"),
            ConsultationType::ErrorAnalysis
        );
    }

    #[test]
    fn consultation_type_labels() {
        assert_eq!(ConsultationType::ErrorAnalysis.label(), "Error Analysis");
        assert_eq!(
            ConsultationType::ArchitectureAdvice.label(),
            "Architecture Decision"
        );
    }
}
