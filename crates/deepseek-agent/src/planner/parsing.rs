use deepseek_core::{Plan, PlanStep};
use serde::Deserialize;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub(crate) struct PlanLlmShape {
    #[serde(default)]
    pub(crate) goal: Option<String>,
    #[serde(default)]
    pub(crate) assumptions: Vec<String>,
    pub(crate) steps: Vec<PlanLlmStep>,
    #[serde(default)]
    pub(crate) verification: Vec<String>,
    #[serde(default)]
    pub(crate) risk_notes: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PlanLlmStep {
    pub(crate) title: String,
    pub(crate) intent: String,
    #[serde(default)]
    pub(crate) tools: Vec<String>,
    #[serde(default)]
    pub(crate) files: Vec<String>,
}

pub(crate) fn parse_plan_from_llm(text: &str, fallback_goal: &str) -> Option<Plan> {
    let snippet = extract_json_snippet(text)?;
    let parsed: PlanLlmShape = serde_json::from_str(snippet).ok()?;
    let mut steps = Vec::new();
    for step in parsed.steps.into_iter().take(16) {
        let title = step.title.trim();
        if title.is_empty() {
            continue;
        }
        let inferred_intent = infer_intent(&step.intent, &step.tools, title);
        let mut tools = step
            .tools
            .into_iter()
            .map(|tool| tool.trim().to_string())
            .filter(|tool| !tool.is_empty())
            .collect::<Vec<_>>();
        if tools.is_empty() {
            tools = default_tools_for_intent(&inferred_intent);
        }
        if tools.is_empty() {
            continue;
        }
        let mut files = step
            .files
            .into_iter()
            .map(|file| file.trim().to_string())
            .filter(|file| !file.is_empty())
            .collect::<Vec<_>>();
        files.sort();
        files.dedup();
        steps.push(PlanStep {
            step_id: Uuid::now_v7(),
            title: title.to_string(),
            intent: inferred_intent,
            tools,
            files,
            done: false,
        });
    }
    if steps.is_empty() {
        return None;
    }

    Some(Plan {
        plan_id: Uuid::now_v7(),
        version: 1,
        goal: parsed
            .goal
            .map(|goal| goal.trim().to_string())
            .filter(|goal| !goal.is_empty())
            .unwrap_or_else(|| fallback_goal.to_string()),
        assumptions: parsed.assumptions,
        steps,
        verification: if parsed.verification.is_empty() {
            vec![
                "cargo fmt --all -- --check".to_string(),
                "cargo test --workspace".to_string(),
            ]
        } else {
            parsed.verification
        },
        risk_notes: parsed.risk_notes,
    })
}

pub(crate) fn extract_json_snippet(text: &str) -> Option<&str> {
    if let Some(start) = text.find("```json") {
        let rest = &text[start + "```json".len()..];
        if let Some(end) = rest.find("```") {
            return Some(rest[..end].trim());
        }
    }
    if let Some(start) = text.find('{')
        && let Some(end) = text.rfind('}')
        && end > start
    {
        return Some(text[start..=end].trim());
    }
    None
}

pub(crate) fn infer_intent(raw_intent: &str, tools: &[String], title: &str) -> String {
    let intent = raw_intent.trim().to_ascii_lowercase();
    if !intent.is_empty() {
        return intent;
    }
    let title_lc = title.to_ascii_lowercase();
    if title_lc.contains("verify") || title_lc.contains("test") {
        return "verify".to_string();
    }
    if title_lc.contains("doc") || title_lc.contains("readme") {
        return "docs".to_string();
    }
    if title_lc.contains("git") || title_lc.contains("branch") || title_lc.contains("commit") {
        return "git".to_string();
    }
    if title_lc.contains("search") || title_lc.contains("find") || title_lc.contains("analy") {
        return "search".to_string();
    }
    if title_lc.contains("edit")
        || title_lc.contains("implement")
        || title_lc.contains("fix")
        || title_lc.contains("refactor")
    {
        return "edit".to_string();
    }
    if let Some(tool) = tools.first() {
        let base = tool.split_once(':').map_or(tool.as_str(), |(name, _)| name);
        if base.starts_with("git.") {
            return "git".to_string();
        }
        if base == "bash.run" {
            return "verify".to_string();
        }
    }
    "task".to_string()
}

pub(crate) fn default_tools_for_intent(intent: &str) -> Vec<String> {
    match intent {
        "search" => vec![
            "index.query".to_string(),
            "fs.grep".to_string(),
            "fs.read".to_string(),
        ],
        "git" => vec!["git.status".to_string(), "git.diff".to_string()],
        "edit" => vec!["fs.edit".to_string(), "patch.stage".to_string()],
        "docs" => vec!["fs.edit".to_string()],
        "verify" => vec!["bash.run".to_string()],
        "recover" => vec!["fs.grep".to_string(), "fs.read".to_string()],
        _ => vec!["fs.list".to_string()],
    }
}

pub(crate) fn parse_declared_tool(raw: &str) -> (String, Option<String>) {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return ("".to_string(), None);
    }

    let (name, arg) = if let Some((name, rest)) = trimmed.split_once(':') {
        (name.trim(), Some(rest.trim().to_string()))
    } else if trimmed.ends_with(')') {
        if let Some(open_idx) = trimmed.find('(') {
            let name = trimmed[..open_idx].trim();
            let inner = trimmed[(open_idx + 1)..(trimmed.len() - 1)].trim();
            (name, Some(inner.to_string()))
        } else {
            (trimmed, None)
        }
    } else {
        (trimmed, None)
    };

    let normalized = normalize_declared_tool_name(name);
    (normalized, arg.filter(|s| !s.is_empty()))
}

pub(crate) fn normalize_declared_tool_name(name: &str) -> String {
    match name.trim().to_ascii_lowercase().as_str() {
        "bash" | "shell" | "shell.run" | "run" => "bash.run".to_string(),
        "grep" | "search" => "fs.grep".to_string(),
        "read" | "read_file" | "fs.read_file" => "fs.read".to_string(),
        "write" | "write_file" | "fs.write_file" => "fs.write".to_string(),
        "edit" | "modify" => "fs.edit".to_string(),
        "list" => "fs.list".to_string(),
        "git_status" => "git.status".to_string(),
        "git_diff" => "git.diff".to_string(),
        "git_show" => "git.show".to_string(),
        other => other.to_string(),
    }
}
