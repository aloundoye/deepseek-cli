use crate::subagents_runtime::memory::{decide_chat_subagent_spawn, summarize_subagent_notes};
use crate::*;

impl AgentEngine {
    pub(crate) fn consult_r1_on_suppressed_escalation(
        &self,
        reason: &str,
        prompt: &str,
        observation: Option<&ObservationPack>,
    ) -> Option<String> {
        let consultation_type = match reason {
            "ambiguous_error" | "repeated_step_failure" | "doom_loop" => {
                crate::consultation::ConsultationType::ErrorAnalysis
            }
            "cross_module_failure" | "blast_radius_exceeded" | "architectural_task" => {
                crate::consultation::ConsultationType::ArchitectureAdvice
            }
            _ => crate::consultation::ConsultationType::TaskDecomposition,
        };
        let context = observation
            .map(ObservationPack::to_r1_context)
            .filter(|text| !text.trim().is_empty())
            .unwrap_or_else(|| format!("Prompt: {prompt}"));
        let request = crate::consultation::ConsultationRequest {
            question: format!(
                "Escalation to R1DriveTools was suppressed ({reason}) because break-glass mode is disabled. \
                 Provide the next concrete steps for the V3 executor loop."
            ),
            context,
            consultation_type,
        };
        let stream_callback = self.stream_callback.lock().ok().and_then(|g| g.clone());
        match crate::consultation::consult_r1(
            self.llm.as_ref(),
            &self.cfg.llm.max_think_model,
            &request,
            stream_callback.as_ref(),
        ) {
            Ok(result) => Some(result.advice),
            Err(error) => {
                self.observer.verbose_log(&format!(
                    "r1 checkpoint consultation failed after suppressed escalation: {error}"
                ));
                None
            }
        }
    }

    pub(crate) fn build_r1_initial_context(
        &self,
        messages: &[ChatMessage],
        prompt: &str,
    ) -> String {
        let mut ctx = String::with_capacity(4096);
        ctx.push_str(&format!("## User request\n{prompt}\n\n"));

        // Include recent tool results (last 5 turns of relevant context)
        ctx.push_str("## Recent context\n");
        let recent: Vec<_> = messages
            .iter()
            .rev()
            .take(10)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        for msg in recent {
            match msg {
                ChatMessage::Tool { content, .. } => {
                    let truncated = if content.len() > 500 {
                        format!("{}...", &content[..content.floor_char_boundary(500)])
                    } else {
                        content.clone()
                    };
                    ctx.push_str(&format!("Tool result: {truncated}\n"));
                }
                ChatMessage::Assistant {
                    content: Some(text),
                    ..
                } if !text.is_empty() => {
                    let truncated = if text.len() > 300 {
                        format!("{}...", &text[..text.floor_char_boundary(300)])
                    } else {
                        text.clone()
                    };
                    ctx.push_str(&format!("Assistant: {truncated}\n"));
                }
                _ => {}
            }
        }
        ctx
    }

    /// Build RepoFacts from the current workspace.
    pub fn chat(&self, prompt: &str) -> Result<String> {
        self.chat_with_options(
            prompt,
            ChatOptions {
                tools: true,
                ..Default::default()
            },
        )
    }

    /// Chat loop with configurable options. When `options.tools` is false, no tool
    /// definitions are sent and the model produces a single text response.
    pub(crate) fn build_repo_facts(&self) -> RepoFacts {
        let workspace = &self.workspace;
        let language = if workspace.join("Cargo.toml").exists() {
            "rust"
        } else if workspace.join("package.json").exists() {
            "javascript"
        } else if workspace.join("pyproject.toml").exists() || workspace.join("setup.py").exists() {
            "python"
        } else if workspace.join("go.mod").exists() {
            "go"
        } else {
            "unknown"
        };
        let build_system = if workspace.join("Cargo.toml").exists() {
            "cargo"
        } else if workspace.join("package.json").exists() {
            "npm"
        } else if workspace.join("Makefile").exists() {
            "make"
        } else {
            "unknown"
        };
        RepoFacts {
            language: language.to_string(),
            build_system: build_system.to_string(),
            workspace_root: workspace.to_string_lossy().to_string(),
            relevant_paths: vec![],
        }
    }

    /// Generate a lightweight plan for the chat discipline loop.
    ///
    /// Calls V3 with no tools to produce a JSON plan, then parses it using
    /// the existing `parse_plan_from_llm` infrastructure. Returns `None` on
    /// failure (graceful degradation — the loop continues without a plan).
    pub(crate) fn consult_r1_final_review(
        &self,
        prompt: &str,
        candidate_answer: &str,
        changed_files: &[String],
    ) -> Option<String> {
        let context = format!(
            "Prompt:\n{prompt}\n\nChanged files:\n{}\n\nCandidate final answer:\n{candidate_answer}",
            changed_files.join("\n")
        );
        let request = crate::consultation::ConsultationRequest {
            question: "Provide a short advisory review of this completion. Focus on risks, missing verification, and likely regressions."
                .to_string(),
            context,
            consultation_type: crate::consultation::ConsultationType::PlanReview,
        };
        match crate::consultation::consult_r1(
            self.llm.as_ref(),
            &self.cfg.llm.max_think_model,
            &request,
            None,
        ) {
            Ok(result) => Some(result.advice),
            Err(error) => {
                self.observer
                    .verbose_log(&format!("r1 final advisory review skipped: {error}"));
                None
            }
        }
    }

    /// Build a system prompt for chat-with-tools mode.
    /// Build the static portion of the system prompt at construction time.
    /// This includes project markers, platform info, tool guidelines, safety rules, etc.
    /// Dynamic parts (date, git branch, memory, verification feedback) are appended per turn.
    pub(crate) fn llm_compact_summary(&self, compacted_range: &[ChatMessage]) -> String {
        // Build a compact representation of the conversation for the summarizer
        let transcript = summarize_chat_messages(compacted_range);

        let summarize_request = ChatRequest {
            model: self.cfg.llm.base_model.clone(),
            messages: vec![
                ChatMessage::System {
                    content: "You are a conversation summarizer. Produce a concise summary of the \
                              following conversation transcript. Preserve:\n\
                              1. The user's original request and intent\n\
                              2. Key decisions made and approaches chosen\n\
                              3. Files read, created, or modified (with paths)\n\
                              4. Current task state and progress\n\
                              5. Any errors, blockers, or important context\n\n\
                              Output ONLY the summary, no preamble. Keep it under 500 words."
                        .to_string(),
                },
                ChatMessage::User {
                    content: format!("Summarize this conversation transcript:\n\n{transcript}"),
                },
            ],
            tools: vec![],
            tool_choice: ToolChoice::none(),
            max_tokens: 2048,
            temperature: Some(0.0),
            thinking: None,
        };

        match self.llm.complete_chat(&summarize_request) {
            Ok(response) => {
                let summary = response.text.trim().to_string();
                if summary.is_empty() {
                    self.observer
                        .verbose_log("llm compaction returned empty, falling back to truncation");
                    transcript
                } else {
                    self.observer.verbose_log(&format!(
                        "llm compaction: {} chars → {} chars",
                        transcript.len(),
                        summary.len()
                    ));
                    summary
                }
            }
            Err(e) => {
                self.observer.warn_log(&format!(
                    "llm compaction failed, using truncation fallback: {e}"
                ));
                transcript
            }
        }
    }

    pub fn chat_with_options(&self, prompt: &str, options: ChatOptions) -> Result<String> {
        include!("orchestrator/chat_with_options_flow.rs")
    }
}
