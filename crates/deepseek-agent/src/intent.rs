use crate::ChatMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskIntent {
    InspectRepo,
    EditCode,
    ArchitectOnly,
}

#[derive(Debug, Clone)]
pub struct IntentInput<'a> {
    pub prompt: &'a str,
    pub mode: ChatMode,
    pub tools: bool,
    pub force_execute: bool,
    pub force_plan_only: bool,
}

pub fn classify_intent(input: &IntentInput<'_>) -> TaskIntent {
    if input.force_plan_only || (input.mode == ChatMode::Architect && !input.force_execute) {
        return TaskIntent::ArchitectOnly;
    }

    if !input.tools {
        return TaskIntent::InspectRepo;
    }

    if input.force_execute || input.mode == ChatMode::Code {
        return TaskIntent::EditCode;
    }

    if matches!(input.mode, ChatMode::Ask | ChatMode::Context) {
        return TaskIntent::InspectRepo;
    }

    let lower = input.prompt.to_ascii_lowercase();
    if contains_edit_intent(&lower) {
        return TaskIntent::EditCode;
    }
    if contains_repo_inspect_intent(&lower) {
        return TaskIntent::InspectRepo;
    }

    TaskIntent::InspectRepo
}

fn contains_edit_intent(lower: &str) -> bool {
    [
        "add ",
        "implement",
        "refactor",
        "fix ",
        "patch ",
        "update ",
        "rename ",
        "remove ",
        "create file",
        "edit ",
        "failing test",
        "write code",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn contains_repo_inspect_intent(lower: &str) -> bool {
    [
        "analyze",
        "analyse",
        "audit",
        "overview",
        "architecture",
        "repo",
        "repository",
        "project",
        "codebase",
        "structure",
        "dependencies",
        "tests",
        "security",
        "quality",
        "readme",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_architect_only() {
        let intent = classify_intent(&IntentInput {
            prompt: "plan changes",
            mode: ChatMode::Architect,
            tools: true,
            force_execute: false,
            force_plan_only: false,
        });
        assert_eq!(intent, TaskIntent::ArchitectOnly);
    }

    #[test]
    fn classifies_repo_inspect() {
        let intent = classify_intent(&IntentInput {
            prompt: "analyze this project",
            mode: ChatMode::Ask,
            tools: true,
            force_execute: false,
            force_plan_only: false,
        });
        assert_eq!(intent, TaskIntent::InspectRepo);
    }

    #[test]
    fn classifies_edit() {
        let intent = classify_intent(&IntentInput {
            prompt: "fix failing test in parser",
            mode: ChatMode::Code,
            tools: true,
            force_execute: false,
            force_plan_only: false,
        });
        assert_eq!(intent, TaskIntent::EditCode);
    }
}
