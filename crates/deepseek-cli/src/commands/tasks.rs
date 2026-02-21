use anyhow::{Result, anyhow};
use deepseek_store::Store;
use serde_json::json;
use std::path::Path;
use uuid::Uuid;

use crate::TasksCmd;
use crate::output::*;

pub(crate) fn run_tasks(cwd: &Path, command: TasksCmd, json_mode: bool) -> Result<()> {
    let store = Store::new(cwd)?;
    match command {
        TasksCmd::List => {
            let tasks = store.list_tasks(None)?;
            if json_mode {
                print_json(&json!({"tasks": tasks}))?;
            } else if tasks.is_empty() {
                println!("No tasks in queue.");
            } else {
                println!("{:<36}  {:<10}  {:<4}  TITLE", "ID", "STATUS", "PRI");
                println!("{}", "-".repeat(80));
                for task in &tasks {
                    println!(
                        "{:<36}  {:<10}  {:<4}  {}",
                        task.task_id, task.status, task.priority, task.title
                    );
                }
                println!("\n{} task(s) total.", tasks.len());
            }
        }
        TasksCmd::Show(args) => {
            let task_id = Uuid::parse_str(&args.id)?;
            let tasks = store.list_tasks(None)?;
            let task = tasks
                .iter()
                .find(|t| t.task_id == task_id)
                .ok_or_else(|| anyhow!("task not found: {}", args.id))?;
            if json_mode {
                print_json(&serde_json::to_value(task)?)?;
            } else {
                println!("Task:     {}", task.task_id);
                println!("Title:    {}", task.title);
                println!("Status:   {}", task.status);
                println!("Priority: {}", task.priority);
                if let Some(outcome) = &task.outcome {
                    println!("Outcome:  {outcome}");
                }
                if let Some(path) = &task.artifact_path {
                    println!("Artifacts: {path}");
                }
                println!("Created:  {}", task.created_at);
                println!("Updated:  {}", task.updated_at);
            }
        }
        TasksCmd::Cancel(args) => {
            let task_id = Uuid::parse_str(&args.id)?;
            store.update_task_status(task_id, "cancelled", Some("cancelled by user"))?;
            if json_mode {
                print_json(&json!({"task_id": args.id, "status": "cancelled"}))?;
            } else {
                println!("Task {task_id} cancelled.");
            }
        }
    }
    Ok(())
}
