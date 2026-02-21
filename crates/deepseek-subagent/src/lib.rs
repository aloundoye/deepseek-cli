use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::thread;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum SubagentRole {
    Explore,
    Plan,
    Task,
    Custom(String),
}

impl SubagentRole {
    fn rank(&self) -> u8 {
        match self {
            Self::Explore => 0,
            Self::Plan => 1,
            Self::Task => 2,
            Self::Custom(_) => 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentTask {
    pub run_id: Uuid,
    pub name: String,
    pub goal: String,
    pub role: SubagentRole,
    pub team: String,
    /// When true, the subagent should only use read-only operations.
    /// Set automatically on retry when a permission denial is detected.
    #[serde(default)]
    pub read_only_fallback: bool,
    /// Optional custom agent definition (from .deepseek/agents/*.md).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_agent: Option<CustomAgentDef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentResult {
    pub run_id: Uuid,
    pub name: String,
    pub role: SubagentRole,
    pub team: String,
    pub attempts: u8,
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    /// True if the result was produced under read-only fallback constraints.
    #[serde(default)]
    pub used_read_only_fallback: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomAgentDef {
    pub name: String,
    pub description: String,
    pub prompt: String,
    pub tools: Vec<String>,
    #[serde(default)]
    pub disallowed_tools: Vec<String>,
    pub model: Option<String>,
    pub max_turns: Option<u64>,
}

/// Load custom agent definitions from .deepseek/agents/*.md and ~/.deepseek/agents/*.md
pub fn load_agent_defs(workspace: &std::path::Path) -> Result<Vec<CustomAgentDef>> {
    let mut defs = Vec::new();
    let project_dir = workspace.join(".deepseek/agents");
    if project_dir.is_dir() {
        load_agent_defs_from_dir(&project_dir, &mut defs)?;
    }
    if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        let global_dir = std::path::PathBuf::from(home).join(".deepseek/agents");
        if global_dir.is_dir() {
            load_agent_defs_from_dir(&global_dir, &mut defs)?;
        }
    }
    Ok(defs)
}

fn load_agent_defs_from_dir(dir: &std::path::Path, out: &mut Vec<CustomAgentDef>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let raw = std::fs::read_to_string(&path)?;
        if let Some(def) = parse_agent_def(&raw, &path) {
            out.push(def);
        }
    }
    Ok(())
}

fn parse_agent_def(raw: &str, path: &std::path::Path) -> Option<CustomAgentDef> {
    let trimmed = raw.trim();
    if !trimmed.starts_with("---") {
        return None;
    }
    let end_idx = trimmed[3..].find("---")?;
    let frontmatter = &trimmed[3..3 + end_idx];
    let body = trimmed[3 + end_idx + 3..].trim().to_string();

    let mut name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("custom")
        .to_string();
    let mut description = String::new();
    let mut tools = Vec::new();
    let mut disallowed_tools = Vec::new();
    let mut model = None;
    let mut max_turns = None;

    for line in frontmatter.lines() {
        let line = line.trim();
        if let Some(value) = line.strip_prefix("name:") {
            name = value
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();
        } else if let Some(value) = line.strip_prefix("description:") {
            description = value
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();
        } else if let Some(value) = line.strip_prefix("model:") {
            model = Some(
                value
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_string(),
            );
        } else if let Some(value) = line.strip_prefix("max_turns:") {
            max_turns = value.trim().parse().ok();
        } else if let Some(value) = line.strip_prefix("tools:") {
            tools = value
                .trim()
                .trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
                .filter(|s| !s.is_empty())
                .collect();
        } else if let Some(value) = line.strip_prefix("disallowed_tools:") {
            disallowed_tools = value
                .trim()
                .trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
    }

    Some(CustomAgentDef {
        name,
        description,
        prompt: body,
        tools,
        disallowed_tools,
        model,
        max_turns,
    })
}

#[derive(Debug, Clone)]
pub struct SubagentManager {
    pub max_concurrency: usize,
    pub max_retries_per_task: usize,
}

impl Default for SubagentManager {
    fn default() -> Self {
        Self {
            max_concurrency: 7,
            max_retries_per_task: 1,
        }
    }
}

impl SubagentManager {
    pub fn new(max_concurrency: usize) -> Self {
        Self {
            max_concurrency: max_concurrency.max(1),
            ..Self::default()
        }
    }

    pub fn run_tasks<F>(&self, tasks: Vec<SubagentTask>, worker: F) -> Vec<SubagentResult>
    where
        F: Fn(SubagentTask) -> Result<String> + Send + Sync + 'static,
    {
        let worker = Arc::new(worker);
        let mut pending = tasks;
        pending.sort_by_key(|task| task.run_id);
        let mut out = Vec::new();

        while !pending.is_empty() {
            let chunk_len = pending.len().min(self.max_concurrency);
            let chunk = pending.drain(0..chunk_len).collect::<Vec<_>>();
            let mut handles = Vec::new();
            for task in chunk {
                let worker = Arc::clone(&worker);
                let retries = self.max_retries_per_task;
                handles.push(thread::spawn(move || {
                    let run_id = task.run_id;
                    let mut attempts = 0usize;
                    let mut current_task = task.clone();
                    loop {
                        attempts += 1;
                        match worker(current_task.clone()) {
                            Ok(output) => {
                                return SubagentResult {
                                    run_id,
                                    name: task.name.clone(),
                                    role: task.role.clone(),
                                    team: task.team.clone(),
                                    attempts: attempts.min(u8::MAX as usize) as u8,
                                    success: true,
                                    output,
                                    error: None,
                                    used_read_only_fallback: current_task.read_only_fallback,
                                };
                            }
                            Err(err) if attempts <= retries => {
                                let err_msg = err.to_string().to_ascii_lowercase();
                                // Detect permission denial errors and fall back to read-only
                                if err_msg.contains("permission denied")
                                    || err_msg.contains("approval denied")
                                    || err_msg.contains("locked mode")
                                    || err_msg.contains("policy blocked")
                                    || err_msg.contains("not allowed")
                                {
                                    current_task.read_only_fallback = true;
                                }
                                continue;
                            }
                            Err(err) => {
                                return SubagentResult {
                                    run_id,
                                    name: task.name.clone(),
                                    role: task.role.clone(),
                                    team: task.team.clone(),
                                    attempts: attempts.min(u8::MAX as usize) as u8,
                                    success: false,
                                    output: String::new(),
                                    error: Some(err.to_string()),
                                    used_read_only_fallback: current_task.read_only_fallback,
                                };
                            }
                        }
                    }
                }));
            }
            for handle in handles {
                if let Ok(result) = handle.join() {
                    out.push(result);
                }
            }
        }

        out.sort_by(|a, b| {
            a.role
                .rank()
                .cmp(&b.role.rank())
                .then(a.team.cmp(&b.team))
                .then(a.name.cmp(&b.name))
                .then(a.run_id.cmp(&b.run_id))
        });
        out
    }

    pub fn merge_results(&self, results: &[SubagentResult]) -> String {
        let mut lines = Vec::new();
        for result in results {
            if result.success {
                lines.push(format!(
                    "[{}::{:?}] {} (attempts={}): {}",
                    result.team, result.role, result.name, result.attempts, result.output
                ));
            } else {
                lines.push(format!(
                    "[{}::{:?}] {} failed (attempts={}): {}",
                    result.team,
                    result.role,
                    result.name,
                    result.attempts,
                    result.error.as_deref().unwrap_or("unknown error")
                ));
            }
        }
        lines.join("\n")
    }
}

/// Teammate execution mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TeammateMode {
    /// Run teammates in the same process using thread pool.
    InProcess,
    /// Run teammates in tmux windows.
    Tmux,
    /// Automatically choose based on environment.
    #[default]
    Auto,
}

impl TeammateMode {
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "in-process" | "in_process" | "inprocess" => Self::InProcess,
            "tmux" => Self::Tmux,
            _ => Self::Auto,
        }
    }
}

/// A message between team members.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamMessage {
    pub from: String,
    pub to: String,
    pub content: String,
    pub timestamp: String,
}

/// Shared state for a team of agents.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TeamState {
    pub messages: Vec<TeamMessage>,
    pub completed_tasks: Vec<String>,
}

impl TeamState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn send_message(&mut self, from: &str, to: &str, content: &str) {
        self.messages.push(TeamMessage {
            from: from.to_string(),
            to: to.to_string(),
            content: content.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        });
    }

    pub fn messages_for(&self, recipient: &str) -> Vec<&TeamMessage> {
        self.messages
            .iter()
            .filter(|m| m.to == recipient || m.to == "*")
            .collect()
    }

    pub fn mark_completed(&mut self, task_name: &str) {
        self.completed_tasks.push(task_name.to_string());
    }
}

/// Coordinator for a team of agents working together.
pub struct TeamCoordinator {
    pub manager: SubagentManager,
    pub mode: TeammateMode,
    pub shared_state: std::sync::Arc<std::sync::Mutex<TeamState>>,
}

impl TeamCoordinator {
    pub fn new(mode: TeammateMode, max_concurrency: usize) -> Self {
        Self {
            manager: SubagentManager::new(max_concurrency),
            mode,
            shared_state: std::sync::Arc::new(std::sync::Mutex::new(TeamState::new())),
        }
    }

    /// Break down a goal into subtasks and distribute to teammates.
    pub fn distribute_tasks<F>(&self, tasks: Vec<SubagentTask>, worker: F) -> Vec<SubagentResult>
    where
        F: Fn(SubagentTask) -> Result<String> + Send + Sync + 'static,
    {
        let shared = self.shared_state.clone();
        self.manager.run_tasks(tasks, move |task| {
            let result = worker(task.clone())?;
            if let Ok(mut state) = shared.lock() {
                state.mark_completed(&task.name);
            }
            Ok(result)
        })
    }

    /// Get the current team state.
    pub fn state(&self) -> TeamState {
        self.shared_state
            .lock()
            .expect("subagent shared_state lock")
            .clone()
    }

    /// Send a message from one teammate to another.
    pub fn send_message(&self, from: &str, to: &str, content: &str) {
        if let Ok(mut state) = self.shared_state.lock() {
            state.send_message(from, to, content);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn runs_tasks_in_bounded_batches() {
        let manager = SubagentManager::new(2);
        let tasks = (0..5)
            .map(|i| SubagentTask {
                run_id: Uuid::now_v7(),
                name: format!("task-{i}"),
                goal: "analyze".to_string(),
                role: SubagentRole::Task,
                team: "default".to_string(),
                read_only_fallback: false,
                custom_agent: None,
            })
            .collect::<Vec<_>>();

        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let in_flight_w = Arc::clone(&in_flight);
        let max_seen_w = Arc::clone(&max_seen);

        let results = manager.run_tasks(tasks, move |_task| {
            let now = in_flight_w.fetch_add(1, Ordering::SeqCst) + 1;
            max_seen_w.fetch_max(now, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(20));
            in_flight_w.fetch_sub(1, Ordering::SeqCst);
            Ok("done".to_string())
        });

        assert_eq!(results.len(), 5);
        assert!(max_seen.load(Ordering::SeqCst) <= 2);
        assert!(results.iter().all(|result| result.success));
    }

    #[test]
    fn retries_failed_subagents_within_budget() {
        let mut manager = SubagentManager::new(1);
        manager.max_retries_per_task = 2;
        let task = SubagentTask {
            run_id: Uuid::now_v7(),
            name: "retry-me".to_string(),
            goal: "recover".to_string(),
            role: SubagentRole::Task,
            team: "execution".to_string(),
            read_only_fallback: false,
            custom_agent: None,
        };
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_w = Arc::clone(&attempts);
        let results = manager.run_tasks(vec![task], move |_task| {
            let count = attempts_w.fetch_add(1, Ordering::SeqCst);
            if count < 1 {
                anyhow::bail!("transient error")
            }
            Ok("recovered".to_string())
        });
        assert_eq!(results.len(), 1);
        assert!(results[0].success);
        assert_eq!(results[0].attempts, 2);
    }

    #[test]
    fn merged_output_is_deterministic() {
        let manager = SubagentManager::new(3);
        let mut tasks = Vec::new();
        for (team, role, name) in [
            ("execution", SubagentRole::Task, "apply"),
            ("planning", SubagentRole::Plan, "split"),
            ("explore", SubagentRole::Explore, "scan"),
        ] {
            tasks.push(SubagentTask {
                run_id: Uuid::now_v7(),
                name: name.to_string(),
                goal: "x".to_string(),
                role,
                team: team.to_string(),
                read_only_fallback: false,
                custom_agent: None,
            });
        }
        let results = manager.run_tasks(tasks, |task| Ok(format!("ok:{}", task.name)));
        let merged = manager.merge_results(&results);
        let lines = merged.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("explore"));
        assert!(lines[1].contains("planning"));
        assert!(lines[2].contains("execution"));
    }

    #[test]
    fn approval_denied_triggers_read_only_fallback() {
        let mut manager = SubagentManager::new(1);
        manager.max_retries_per_task = 2;
        let task = SubagentTask {
            run_id: Uuid::now_v7(),
            name: "edit-file".to_string(),
            goal: "fix bug".to_string(),
            role: SubagentRole::Task,
            team: "execution".to_string(),
            read_only_fallback: false,
            custom_agent: None,
        };
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_w = Arc::clone(&attempts);
        let results = manager.run_tasks(vec![task], move |task| {
            let count = attempts_w.fetch_add(1, Ordering::SeqCst);
            if count == 0 {
                anyhow::bail!("permission denied: fs.write blocked by policy")
            }
            // On retry, the task should have read_only_fallback set
            assert!(
                task.read_only_fallback,
                "expected read_only_fallback on retry"
            );
            Ok("read-only result".to_string())
        });
        assert_eq!(results.len(), 1);
        assert!(results[0].success);
        assert!(results[0].used_read_only_fallback);
        assert_eq!(results[0].attempts, 2);
    }

    #[test]
    fn teammate_mode_parse() {
        assert_eq!(TeammateMode::parse("in-process"), TeammateMode::InProcess);
        assert_eq!(TeammateMode::parse("tmux"), TeammateMode::Tmux);
        assert_eq!(TeammateMode::parse("auto"), TeammateMode::Auto);
        assert_eq!(TeammateMode::parse("unknown"), TeammateMode::Auto);
    }

    #[test]
    fn team_state_messaging() {
        let mut state = TeamState::new();
        state.send_message("lead", "worker-1", "analyze file.rs");
        state.send_message("lead", "worker-2", "review tests");
        state.send_message("worker-1", "lead", "done with analysis");

        let for_worker1 = state.messages_for("worker-1");
        assert_eq!(for_worker1.len(), 1);
        assert_eq!(for_worker1[0].content, "analyze file.rs");

        let for_lead = state.messages_for("lead");
        assert_eq!(for_lead.len(), 1);
    }

    #[test]
    fn team_state_broadcast() {
        let mut state = TeamState::new();
        state.send_message("lead", "*", "everyone stop");
        let for_anyone = state.messages_for("worker-1");
        assert_eq!(for_anyone.len(), 1);
    }

    #[test]
    fn team_coordinator_distributes_tasks() {
        let coordinator = TeamCoordinator::new(TeammateMode::InProcess, 2);
        let tasks = (0..3)
            .map(|i| SubagentTask {
                run_id: Uuid::now_v7(),
                name: format!("task-{i}"),
                goal: "do work".to_string(),
                role: SubagentRole::Task,
                team: "alpha".to_string(),
                read_only_fallback: false,
                custom_agent: None,
            })
            .collect();

        let results =
            coordinator.distribute_tasks(tasks, |task| Ok(format!("completed:{}", task.name)));
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.success));
        let state = coordinator.state();
        assert_eq!(state.completed_tasks.len(), 3);
    }

    #[test]
    fn team_coordinator_sends_messages() {
        let coordinator = TeamCoordinator::new(TeammateMode::Auto, 1);
        coordinator.send_message("lead", "worker", "hello");
        let state = coordinator.state();
        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0].from, "lead");
    }
}
