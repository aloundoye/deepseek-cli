use super::*;

pub(super) fn emit_event_if_present(tool_loop: &ToolUseLoop<'_>, kind: EventKind) {
    if let Some(ref cb) = tool_loop.event_cb {
        cb(kind);
    }
}

pub(super) fn emit_injection_warnings(
    tool_loop: &ToolUseLoop<'_>,
    warnings: &[codingbuddy_policy::output_scanner::InjectionWarning],
) {
    for warning in warnings {
        emit(
            tool_loop,
            StreamChunk::SecurityWarning {
                message: format!(
                    "{:?} — {} (matched: {})",
                    warning.severity, warning.pattern_name, warning.matched_text
                ),
            },
        );
    }
}

pub(super) fn emit(tool_loop: &ToolUseLoop<'_>, chunk: StreamChunk) {
    if let Some(ref cb) = tool_loop.stream_cb {
        cb(chunk);
    }
}
