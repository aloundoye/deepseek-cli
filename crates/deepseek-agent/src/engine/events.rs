use crate::*;

impl AgentEngine {
    pub(crate) fn emit(&self, session_id: Uuid, kind: EventKind) -> Result<()> {
        let event = EventEnvelope {
            seq_no: self.store.next_seq_no(session_id)?,
            at: Utc::now(),
            session_id,
            kind,
        };
        self.store.append_event(&event)?;
        self.observer.record_event(&event)?;
        Ok(())
    }

    pub(crate) fn inject_hook_context(messages: &mut Vec<ChatMessage>, hook_result: &HookResult) {
        for ctx in &hook_result.additional_context {
            if !ctx.is_empty() {
                messages.push(ChatMessage::User {
                    content: format!("<hook-context>\n{ctx}\n</hook-context>"),
                });
            }
        }
    }

    // ── Checkpoint helpers ──────────────────────────────────────────────

    /// Returns true if this tool modifies files and should trigger a checkpoint.
    pub(crate) fn fire_hook(&self, event: HookEvent, input: &HookInput) -> HookResult {
        let result = self.hooks.fire(event, input);
        if !result.runs.is_empty() {
            self.observer.verbose_log(&format!(
                "hook {}: {} handler(s) fired, blocked={}",
                event.as_str(),
                result.runs.len(),
                result.blocked
            ));
        }
        result
    }

    /// Inject any additional context from hooks into the message list.
    pub(crate) fn hook_input(&self, event: HookEvent) -> HookInput {
        HookInput {
            event: event.as_str().to_string(),
            tool_name: None,
            tool_input: None,
            tool_result: None,
            prompt: None,
            session_type: None,
            workspace: self.workspace.to_string_lossy().to_string(),
        }
    }

    /// Fire a hook event and return the result.
    /// Logs hook runs via the observer.
    pub(crate) fn emit_cost_event(
        &self,
        session_id: Uuid,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Result<f64> {
        let estimated_cost_usd = (input_tokens as f64 / 1_000_000.0)
            * self.cfg.usage.cost_per_million_input
            + (output_tokens as f64 / 1_000_000.0) * self.cfg.usage.cost_per_million_output;
        self.emit(
            session_id,
            EventKind::CostUpdatedV1 {
                input_tokens,
                output_tokens,
                estimated_cost_usd,
            },
        )?;
        Ok(estimated_cost_usd)
    }

    pub(crate) fn transition(&self, session: &mut Session, to: SessionState) -> Result<()> {
        let from = session.status.clone();
        if !is_valid_session_state_transition(&from, &to) {
            return Err(anyhow!(
                "invalid session state transition: {:?} -> {:?}",
                from,
                to
            ));
        }
        session.status = to.clone();
        self.store.save_session(session)?;
        self.emit(
            session.session_id,
            EventKind::SessionStateChangedV1 { from, to },
        )
    }
}
