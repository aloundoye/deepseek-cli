use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
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
    /// Optional custom agent definition (from .codingbuddy/agents/*.md).
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

/// Load custom agent definitions from .codingbuddy/agents/*.md and ~/.codingbuddy/agents/*.md
pub fn load_agent_defs(workspace: &Path) -> Result<Vec<CustomAgentDef>> {
    let mut defs = Vec::new();
    let project_dir = workspace.join(".codingbuddy/agents");
    if project_dir.is_dir() {
        load_agent_defs_from_dir(&project_dir, &mut defs)?;
    }
    if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        let global_dir = PathBuf::from(home).join(".codingbuddy/agents");
        if global_dir.is_dir() {
            load_agent_defs_from_dir(&global_dir, &mut defs)?;
        }
    }
    Ok(defs)
}

fn load_agent_defs_from_dir(dir: &Path, out: &mut Vec<CustomAgentDef>) -> Result<()> {
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

fn parse_agent_def(raw: &str, path: &Path) -> Option<CustomAgentDef> {
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

// ── P5-01: Background Task Registry ──

/// Status of a background subagent task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackgroundTaskStatus {
    Running,
    Completed,
    Failed,
}

/// A background task entry in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundTaskEntry {
    pub id: Uuid,
    pub prompt: String,
    pub status: BackgroundTaskStatus,
    pub result: Option<String>,
    pub error: Option<String>,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub finished_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Registry for background subagent tasks.
///
/// Manages lifecycle of tasks that run asynchronously while the main
/// agent loop continues. Results can be retrieved later.
#[derive(Debug, Clone, Default)]
pub struct BackgroundTaskRegistry {
    tasks: Arc<Mutex<HashMap<Uuid, BackgroundTaskEntry>>>,
}

impl BackgroundTaskRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Submit a task to run in the background.
    ///
    /// Returns the task ID immediately. The task runs in a separate thread.
    pub fn submit<F>(&self, prompt: String, worker: F) -> Uuid
    where
        F: FnOnce() -> Result<String> + Send + 'static,
    {
        let id = Uuid::now_v7();
        let entry = BackgroundTaskEntry {
            id,
            prompt: prompt.clone(),
            status: BackgroundTaskStatus::Running,
            result: None,
            error: None,
            started_at: chrono::Utc::now(),
            finished_at: None,
        };

        {
            let mut tasks = self.tasks.lock().expect("background registry lock");
            tasks.insert(id, entry);
        }

        let tasks_ref = self.tasks.clone();
        thread::spawn(move || {
            let (status, result, error) = match worker() {
                Ok(output) => (BackgroundTaskStatus::Completed, Some(output), None),
                Err(e) => (BackgroundTaskStatus::Failed, None, Some(e.to_string())),
            };
            let finished_at = chrono::Utc::now();
            if let Ok(mut tasks) = tasks_ref.lock()
                && let Some(entry) = tasks.get_mut(&id)
            {
                entry.status = status;
                entry.result = result;
                entry.error = error;
                entry.finished_at = Some(finished_at);
            }
        });

        id
    }

    /// Get the status of a background task.
    pub fn get(&self, id: Uuid) -> Option<BackgroundTaskEntry> {
        self.tasks.lock().ok()?.get(&id).cloned()
    }

    /// List all background tasks.
    pub fn list(&self) -> Vec<BackgroundTaskEntry> {
        self.tasks
            .lock()
            .map(|tasks| tasks.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Get running task count.
    pub fn running_count(&self) -> usize {
        self.tasks
            .lock()
            .map(|tasks| {
                tasks
                    .values()
                    .filter(|t| t.status == BackgroundTaskStatus::Running)
                    .count()
            })
            .unwrap_or(0)
    }
}

// ── P5-02: Worktree Isolation ──

/// Manages git worktree creation and cleanup for isolated subagent execution.
pub struct WorktreeIsolation {
    workspace: PathBuf,
    worktree_path: PathBuf,
    created: bool,
}

impl WorktreeIsolation {
    /// Create a new isolated worktree for a subagent.
    pub fn create(workspace: &Path, name: &str) -> Result<Self> {
        let runtime_dir = workspace.join(".codingbuddy");
        let worktrees_dir = runtime_dir.join("worktrees");
        std::fs::create_dir_all(&worktrees_dir)?;

        let worktree_path = worktrees_dir.join(name);
        if worktree_path.exists() {
            let _ = std::fs::remove_dir_all(&worktree_path);
        }

        let base_commit = git_stdout(workspace, &["rev-parse", "HEAD"])
            .context("failed to get HEAD commit for worktree")?;

        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(workspace)
            .args(["worktree", "add", "--detach"])
            .arg(&worktree_path)
            .arg(&base_commit)
            .output()
            .context("failed to spawn git worktree add")?;

        if !output.status.success() {
            return Err(anyhow!(
                "git worktree add failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        Ok(Self {
            workspace: workspace.to_path_buf(),
            worktree_path,
            created: true,
        })
    }

    /// Get the path to the worktree.
    pub fn path(&self) -> &Path {
        &self.worktree_path
    }

    /// Collect the diff of changes made in the worktree.
    pub fn collect_diff(&self) -> Result<String> {
        git_stdout(&self.worktree_path, &["diff", "--binary", "HEAD"])
    }

    /// Clean up the worktree.
    pub fn cleanup(&mut self) -> Result<()> {
        if !self.created {
            return Ok(());
        }
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(&self.workspace)
            .args(["worktree", "remove", "--force"])
            .arg(&self.worktree_path)
            .output()
            .context("failed to spawn git worktree remove")?;

        if output.status.success() {
            self.created = false;
            Ok(())
        } else {
            Err(anyhow!(
                "git worktree remove failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }
}

impl Drop for WorktreeIsolation {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

fn git_stdout(workspace: &Path, args: &[&str]) -> Result<String> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(workspace)
        .output()
        .context("failed to spawn git command")?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err(anyhow!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

// ── P5-03: Persistent Memory ──

/// Get the memory path for a named subagent.
///
/// Memory is stored per-project, per-agent:
/// `~/.codingbuddy/projects/<hash>/agents/<name>/MEMORY.md`
pub fn agent_memory_path(workspace: &Path, agent_name: &str) -> PathBuf {
    let hash = project_hash(workspace);
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".codingbuddy/projects")
        .join(&hash)
        .join("agents")
        .join(agent_name)
        .join("MEMORY.md")
}

/// Load agent memory (first 200 lines) for inclusion in system prompt.
pub fn load_agent_memory(workspace: &Path, agent_name: &str) -> Option<String> {
    let path = agent_memory_path(workspace, agent_name);
    let content = std::fs::read_to_string(&path).ok()?;
    let lines: Vec<&str> = content.lines().take(200).collect();
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

fn project_hash(workspace: &Path) -> String {
    let canonical = workspace
        .canonicalize()
        .unwrap_or_else(|_| workspace.to_path_buf());
    let path_str = canonical.to_string_lossy();
    // Simple hash for directory identification
    let mut hash: u64 = 0;
    for byte in path_str.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
    }
    format!("{hash:016x}")
}

// ── P5-04: Resumable Subagents ──

/// Path to a subagent's transcript file.
pub fn transcript_path(workspace: &Path, agent_id: Uuid) -> PathBuf {
    let hash = project_hash(workspace);
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".codingbuddy/projects")
        .join(&hash)
        .join("subagents")
        .join(format!("{agent_id}.jsonl"))
}

/// Append a message to a subagent's transcript.
pub fn append_transcript(workspace: &Path, agent_id: Uuid, line: &str) -> Result<()> {
    let path = transcript_path(workspace, agent_id);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    writeln!(file, "{line}")?;
    Ok(())
}

/// Load a subagent's transcript lines.
pub fn load_transcript(workspace: &Path, agent_id: Uuid) -> Result<Vec<String>> {
    let path = transcript_path(workspace, agent_id);
    let content = std::fs::read_to_string(&path).context("failed to load subagent transcript")?;
    Ok(content.lines().map(String::from).collect())
}

// ── P5-05: Subagent Configuration ──

/// Configuration for subagent execution (auto-compaction, max turns, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentConfig {
    /// Maximum turns for the subagent's tool-use loop.
    pub max_turns: usize,
    /// Whether the subagent should compact its context when approaching limits.
    pub compaction_enabled: bool,
    /// Model override for the subagent.
    pub model: Option<String>,
    /// Tool restrictions — if non-empty, only these tools are available.
    pub allowed_tools: Vec<String>,
    /// Tools to exclude from the subagent's toolset.
    pub disallowed_tools: Vec<String>,
}

impl Default for SubagentConfig {
    fn default() -> Self {
        Self {
            max_turns: 30,
            compaction_enabled: true,
            model: None,
            allowed_tools: Vec::new(),
            disallowed_tools: Vec::new(),
        }
    }
}

impl SubagentConfig {
    /// Build from a CustomAgentDef, falling back to defaults.
    pub fn from_agent_def(def: &CustomAgentDef) -> Self {
        Self {
            max_turns: def.max_turns.map(|n| n as usize).unwrap_or(30),
            compaction_enabled: true,
            model: def.model.clone(),
            allowed_tools: def.tools.clone(),
            disallowed_tools: def.disallowed_tools.clone(),
        }
    }
}

// ── SubagentManager ──

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
    pub shared_state: Arc<Mutex<TeamState>>,
}

impl TeamCoordinator {
    pub fn new(mode: TeammateMode, max_concurrency: usize) -> Self {
        Self {
            manager: SubagentManager::new(max_concurrency),
            mode,
            shared_state: Arc::new(Mutex::new(TeamState::new())),
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

    // ── P5-01: Background Task Registry Tests ──

    #[test]
    fn background_task_executes_async() {
        let registry = BackgroundTaskRegistry::new();
        let id = registry.submit("test task".to_string(), || {
            thread::sleep(Duration::from_millis(50));
            Ok("background result".to_string())
        });

        // Task should be running immediately
        let entry = registry.get(id).expect("task should exist");
        assert_eq!(entry.status, BackgroundTaskStatus::Running);

        // Wait for completion
        thread::sleep(Duration::from_millis(200));

        let entry = registry.get(id).expect("task should exist");
        assert_eq!(entry.status, BackgroundTaskStatus::Completed);
        assert_eq!(entry.result.as_deref(), Some("background result"));
    }

    #[test]
    fn background_results_retrievable() {
        let registry = BackgroundTaskRegistry::new();
        let id1 = registry.submit("task 1".to_string(), || Ok("result 1".to_string()));
        let id2 = registry.submit("task 2".to_string(), || Ok("result 2".to_string()));

        thread::sleep(Duration::from_millis(100));

        let all = registry.list();
        assert_eq!(all.len(), 2);

        let entry1 = registry.get(id1).unwrap();
        assert_eq!(entry1.result.as_deref(), Some("result 1"));
        let entry2 = registry.get(id2).unwrap();
        assert_eq!(entry2.result.as_deref(), Some("result 2"));
    }

    // ── P5-02: Worktree Isolation Tests ──

    #[test]
    fn worktree_creates_and_cleans() {
        let temp = tempfile::tempdir().unwrap();
        // Initialize a git repo
        let init = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(temp.path())
            .output()
            .unwrap();
        assert!(init.status.success(), "git init failed");

        // Need at least one commit for worktree
        std::fs::write(temp.path().join("test.txt"), "hello\n").unwrap();
        let _ = std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(temp.path())
            .output()
            .unwrap();
        let _ = std::process::Command::new("git")
            .args(["commit", "-m", "initial", "--allow-empty"])
            .current_dir(temp.path())
            .output()
            .unwrap();

        let mut wt = WorktreeIsolation::create(temp.path(), "test-agent").unwrap();
        assert!(wt.path().exists(), "worktree should exist");

        wt.cleanup().unwrap();
        assert!(!wt.path().exists(), "worktree should be cleaned up");
    }

    #[test]
    fn worktree_diff_collected() {
        let temp = tempfile::tempdir().unwrap();
        let init = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(temp.path())
            .output()
            .unwrap();
        assert!(init.status.success());

        std::fs::write(temp.path().join("file.txt"), "original\n").unwrap();
        let _ = std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(temp.path())
            .output();
        let _ = std::process::Command::new("git")
            .args(["commit", "-m", "initial"])
            .current_dir(temp.path())
            .output();

        let mut wt = WorktreeIsolation::create(temp.path(), "diff-test").unwrap();

        // Make a change in the worktree
        std::fs::write(wt.path().join("file.txt"), "modified\n").unwrap();
        let diff = wt.collect_diff().unwrap();
        assert!(
            diff.contains("modified") || diff.contains("original"),
            "diff should contain changes"
        );

        wt.cleanup().unwrap();
    }

    // ── P5-04: Transcript Tests ──

    #[test]
    fn transcript_persists() {
        let temp = tempfile::tempdir().unwrap();
        let agent_id = Uuid::now_v7();

        append_transcript(
            temp.path(),
            agent_id,
            r#"{"role":"user","content":"hello"}"#,
        )
        .unwrap();
        append_transcript(
            temp.path(),
            agent_id,
            r#"{"role":"assistant","content":"hi"}"#,
        )
        .unwrap();

        let lines = load_transcript(temp.path(), agent_id).unwrap();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("user"));
        assert!(lines[1].contains("assistant"));
    }

    #[test]
    fn resume_continues_conversation() {
        let temp = tempfile::tempdir().unwrap();
        let agent_id = Uuid::now_v7();

        // Write initial transcript
        append_transcript(temp.path(), agent_id, r#"{"turn":1}"#).unwrap();
        append_transcript(temp.path(), agent_id, r#"{"turn":2}"#).unwrap();

        // "Resume" by loading and appending
        let existing = load_transcript(temp.path(), agent_id).unwrap();
        assert_eq!(existing.len(), 2);

        append_transcript(temp.path(), agent_id, r#"{"turn":3}"#).unwrap();

        let updated = load_transcript(temp.path(), agent_id).unwrap();
        assert_eq!(updated.len(), 3);
    }

    // ── P5-05: SubagentConfig Tests ──

    #[test]
    fn subagent_inherits_compaction() {
        let config = SubagentConfig::default();
        assert!(
            config.compaction_enabled,
            "compaction should be enabled by default"
        );
        assert_eq!(config.max_turns, 30);
    }

    #[test]
    fn custom_agent_max_turns_respected() {
        let def = CustomAgentDef {
            name: "test".to_string(),
            description: "test agent".to_string(),
            prompt: "do stuff".to_string(),
            tools: vec!["fs_read".to_string()],
            disallowed_tools: vec!["bash_run".to_string()],
            model: Some("deepseek-reasoner".to_string()),
            max_turns: Some(10),
        };
        let config = SubagentConfig::from_agent_def(&def);
        assert_eq!(config.max_turns, 10);
        assert_eq!(config.model.as_deref(), Some("deepseek-reasoner"));
        assert_eq!(config.allowed_tools, vec!["fs_read"]);
        assert_eq!(config.disallowed_tools, vec!["bash_run"]);
    }

    // ── P5-17: Agent model override and tool restrictions ────────────────

    #[test]
    fn agent_model_override() {
        let def = CustomAgentDef {
            name: "code-reviewer".to_string(),
            description: "Reviews code quality".to_string(),
            prompt: "Review code carefully".to_string(),
            tools: vec![],
            disallowed_tools: vec![],
            model: Some("deepseek-chat".to_string()),
            max_turns: None,
        };
        let config = SubagentConfig::from_agent_def(&def);
        assert_eq!(config.model.as_deref(), Some("deepseek-chat"));
        assert_eq!(config.max_turns, 30); // default fallback
        assert!(config.compaction_enabled);
    }

    #[test]
    fn agent_tool_restriction() {
        let def = CustomAgentDef {
            name: "safe-agent".to_string(),
            description: "Read-only exploration".to_string(),
            prompt: "Explore only".to_string(),
            tools: vec![
                "fs_read".to_string(),
                "fs_glob".to_string(),
                "fs_grep".to_string(),
            ],
            disallowed_tools: vec![
                "bash_run".to_string(),
                "fs_edit".to_string(),
                "fs_write".to_string(),
            ],
            model: None,
            max_turns: Some(5),
        };
        let config = SubagentConfig::from_agent_def(&def);
        assert_eq!(config.allowed_tools.len(), 3);
        assert_eq!(config.disallowed_tools.len(), 3);
        assert_eq!(config.max_turns, 5);
        assert!(config.model.is_none());
    }

    // ── P5-16: Agent defs list/load ──────────────────────────────────────

    #[test]
    fn agents_list_shows_defs() {
        let workspace = std::env::temp_dir().join(format!(
            "codingbuddy-agents-list-test-{}",
            uuid::Uuid::now_v7()
        ));
        let agents_dir = workspace.join(".codingbuddy/agents");
        std::fs::create_dir_all(&agents_dir).expect("agents dir");

        std::fs::write(
            agents_dir.join("explorer.md"),
            "---\nname: explorer\ndescription: Explores codebase\ntools: [fs_read, fs_glob]\nmax_turns: 10\n---\nExplore the codebase.",
        )
        .expect("write agent def");

        std::fs::write(
            agents_dir.join("reviewer.md"),
            "---\nname: reviewer\ndescription: Reviews PRs\nmodel: deepseek-reasoner\n---\nReview pull requests carefully.",
        )
        .expect("write agent def 2");

        let defs = load_agent_defs(&workspace).expect("load");
        assert_eq!(defs.len(), 2);

        let explorer = defs.iter().find(|d| d.name == "explorer").unwrap();
        assert_eq!(explorer.description, "Explores codebase");
        assert_eq!(explorer.tools, vec!["fs_read", "fs_glob"]);
        assert_eq!(explorer.max_turns, Some(10));

        let reviewer = defs.iter().find(|d| d.name == "reviewer").unwrap();
        assert_eq!(reviewer.model.as_deref(), Some("deepseek-reasoner"));
    }
}
