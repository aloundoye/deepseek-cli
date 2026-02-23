pub fn score_prompt(prompt: &str) -> u64 {
    let lower = prompt.to_ascii_lowercase();
    let mut score = 0_u64;

    // Multi-domain indicators.
    for keyword in [
        "frontend",
        "backend",
        "api",
        "database",
        "migration",
        "test",
        "ci",
        "docs",
        "security",
        "performance",
    ] {
        if lower.contains(keyword) {
            score = score.saturating_add(10);
        }
    }

    // Multi-file / coordination cues.
    for keyword in [
        "across",
        "multiple files",
        "end-to-end",
        "refactor",
        "orchestrate",
        "workflow",
        "integration",
        "pipeline",
    ] {
        if lower.contains(keyword) {
            score = score.saturating_add(8);
        }
    }

    // Prompt length is a weak complexity signal.
    let tokenish = lower.split_whitespace().count() as u64;
    score.saturating_add((tokenish / 16).min(20))
}

#[cfg(test)]
mod tests {
    use super::score_prompt;

    #[test]
    fn complexity_score_increases_for_cross_domain_tasks() {
        let simple = score_prompt("fix typo");
        let complex = score_prompt(
            "refactor backend api and frontend flows across multiple files with tests",
        );
        assert!(complex > simple);
    }
}
