use crate::ChatMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskIntent {
    InspectRepo,
    EditCode,
}

#[derive(Debug, Clone)]
pub struct IntentInput<'a> {
    pub prompt: &'a str,
    pub mode: ChatMode,
    pub tools: bool,
}

pub fn classify_intent(input: &IntentInput<'_>) -> TaskIntent {
    if !input.tools {
        return TaskIntent::InspectRepo;
    }

    if matches!(input.mode, ChatMode::Code) {
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
    fn classifies_repo_inspect() {
        let intent = classify_intent(&IntentInput {
            prompt: "analyze this project",
            mode: ChatMode::Ask,
            tools: true,
        });
        assert_eq!(intent, TaskIntent::InspectRepo);
    }

    #[test]
    fn classifies_edit() {
        let intent = classify_intent(&IntentInput {
            prompt: "fix failing test in parser",
            mode: ChatMode::Code,
            tools: true,
        });
        assert_eq!(intent, TaskIntent::EditCode);
    }

    #[test]
    fn code_mode_defaults_to_edit() {
        let intent = classify_intent(&IntentInput {
            prompt: "anything",
            mode: ChatMode::Code,
            tools: true,
        });
        assert_eq!(intent, TaskIntent::EditCode);
    }
}
