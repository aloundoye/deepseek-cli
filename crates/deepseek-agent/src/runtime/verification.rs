use crate::*;

impl AgentEngine {
    pub(crate) fn run_verification_with_output(
        &self,
        commands: &[String],
        tracker: &mut FailureTracker,
        session_id: uuid::Uuid,
    ) -> std::result::Result<(), String> {
        let mut errors = Vec::new();
        for cmd in commands {
            self.observer
                .verbose_log(&format!("verify: running `{cmd}`"));
            let tool_call = deepseek_core::ToolCall {
                name: "bash.run".to_string(),
                args: serde_json::json!({"cmd": cmd, "timeout": 60}),
                requires_approval: false,
            };
            let proposal = self.tool_host.propose(tool_call);
            if !proposal.approved {
                self.observer
                    .verbose_log(&format!("verify: `{cmd}` denied by policy"));
                errors.push(format!("Command `{cmd}` denied by policy"));
                continue;
            }
            let result = self.tool_host.execute(ApprovedToolCall {
                invocation_id: proposal.invocation_id,
                call: proposal.call,
            });
            if result.success {
                self.observer
                    .verbose_log(&format!("verify: `{cmd}` passed"));
            } else {
                self.observer
                    .verbose_log(&format!("verify: `{cmd}` FAILED"));
                tracker.record_failure();
                // Extract error output from ToolResult
                let output_text = result
                    .output
                    .as_str()
                    .map(|s| s.to_string())
                    .or_else(|| {
                        result
                            .output
                            .get("stderr")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })
                    .or_else(|| {
                        result
                            .output
                            .get("stdout")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })
                    .unwrap_or_else(|| result.output.to_string());
                // Cap at 2000 chars to avoid flooding context
                let truncated = if output_text.len() > 2000 {
                    format!(
                        "{}...(truncated)",
                        &output_text[..output_text.floor_char_boundary(2000)]
                    )
                } else {
                    output_text
                };
                errors.push(format!("Command `{cmd}` failed:\n{truncated}"));
            }
            if let Err(e) = self.emit(
                session_id,
                EventKind::ToolResultV1 {
                    result: result.clone(),
                },
            ) {
                self.observer
                    .warn_log(&format!("event: failed to emit ToolResultV1: {e}"));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("\n\n"))
        }
    }

    pub(crate) fn run_verification(
        &self,
        commands: &[String],
        tracker: &mut FailureTracker,
        session_id: uuid::Uuid,
    ) -> bool {
        let mut all_passed = true;
        for cmd in commands {
            self.observer
                .verbose_log(&format!("verify: running `{cmd}`"));
            let tool_call = deepseek_core::ToolCall {
                name: "bash.run".to_string(),
                args: serde_json::json!({"cmd": cmd, "timeout": 60}),
                requires_approval: false,
            };
            let proposal = self.tool_host.propose(tool_call);
            if !proposal.approved {
                self.observer
                    .verbose_log(&format!("verify: `{cmd}` denied by policy"));
                all_passed = false;
                continue;
            }
            let result = self.tool_host.execute(ApprovedToolCall {
                invocation_id: proposal.invocation_id,
                call: proposal.call,
            });
            if result.success {
                self.observer
                    .verbose_log(&format!("verify: `{cmd}` passed"));
            } else {
                self.observer
                    .verbose_log(&format!("verify: `{cmd}` FAILED"));
                tracker.record_failure();
                all_passed = false;
            }
            if let Err(e) = self.emit(
                session_id,
                EventKind::ToolResultV1 {
                    result: result.clone(),
                },
            ) {
                self.observer
                    .warn_log(&format!("event: failed to emit ToolResultV1: {e}"));
            }
        }
        all_passed
    }
}
