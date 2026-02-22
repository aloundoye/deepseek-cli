use crate::*;

impl AgentEngine {
    pub(crate) fn request_tool_approval(&self, call: &ToolCall) -> Result<bool> {
        // Try external approval handler first (TUI / raw-mode compatible).
        if let Ok(mut guard) = self.approval_handler.lock()
            && let Some(handler) = guard.as_mut()
        {
            return handler(call);
        }

        // Fallback: blocking stdin for non-TUI mode.
        let mut stdout = std::io::stdout();
        let stdin = std::io::stdin();
        if !stdin.is_terminal() || !stdout.is_terminal() {
            return Ok(false);
        }

        let compact_args = serde_json::to_string(&call.args)
            .unwrap_or_else(|_| "<unserializable args>".to_string());
        writeln!(
            stdout,
            "approval required for tool `{}` with args {}",
            call.name, compact_args
        )?;
        write!(stdout, "approve this call? [y/N]: ")?;
        stdout.flush()?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let normalized = input.trim().to_ascii_lowercase();
        Ok(matches!(normalized.as_str(), "y" | "yes"))
    }

    // ── MCP tool integration ─────────────────────────────────────────────
}
