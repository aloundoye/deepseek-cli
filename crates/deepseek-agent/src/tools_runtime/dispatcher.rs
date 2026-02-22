#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolDispatchOutcome {
    Continue,
    Completed,
}
