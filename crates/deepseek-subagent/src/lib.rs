use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::thread;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentTask {
    pub run_id: Uuid,
    pub name: String,
    pub goal: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentResult {
    pub run_id: Uuid,
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SubagentManager {
    pub max_concurrency: usize,
}

impl Default for SubagentManager {
    fn default() -> Self {
        Self { max_concurrency: 7 }
    }
}

impl SubagentManager {
    pub fn new(max_concurrency: usize) -> Self {
        Self {
            max_concurrency: max_concurrency.max(1),
        }
    }

    pub fn run_tasks<F>(&self, tasks: Vec<SubagentTask>, worker: F) -> Vec<SubagentResult>
    where
        F: Fn(SubagentTask) -> Result<String> + Send + Sync + 'static,
    {
        let worker = Arc::new(worker);
        let mut pending = tasks;
        let mut out = Vec::new();

        while !pending.is_empty() {
            let chunk_len = pending.len().min(self.max_concurrency);
            let chunk = pending.drain(0..chunk_len).collect::<Vec<_>>();
            let mut handles = Vec::new();
            for task in chunk {
                let worker = Arc::clone(&worker);
                handles.push(thread::spawn(move || {
                    let run_id = task.run_id;
                    match worker(task) {
                        Ok(output) => SubagentResult {
                            run_id,
                            success: true,
                            output,
                            error: None,
                        },
                        Err(err) => SubagentResult {
                            run_id,
                            success: false,
                            output: String::new(),
                            error: Some(err.to_string()),
                        },
                    }
                }));
            }
            for handle in handles {
                if let Ok(result) = handle.join() {
                    out.push(result);
                }
            }
        }

        out
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
    }
}
