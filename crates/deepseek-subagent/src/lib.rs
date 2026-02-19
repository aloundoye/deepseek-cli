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
}

impl SubagentRole {
    fn rank(&self) -> u8 {
        match self {
            Self::Explore => 0,
            Self::Plan => 1,
            Self::Task => 2,
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
}
