use anyhow::Result;
use clap::Subcommand;
use deepseek_store::Store;
use std::path::Path;
use uuid::Uuid;

#[derive(Subcommand)]
pub enum SessionCmd {
    /// List all sessions
    List,
    /// Show details of a specific session
    Show {
        /// Session ID to show (defaults to latest)
        id: Option<String>,
    },
    /// List all runs for a specific session
    Runs {
        /// Session ID to list runs for (defaults to latest)
        id: Option<String>,
    },
    /// Show a specific run
    Run {
        /// Run ID to show
        run_id: String,
    },
}

pub fn run_session_cmd(workspace: &Path, command: SessionCmd, _json: bool) -> Result<()> {
    let store = Store::new(workspace)?;
    
    match command {
        SessionCmd::List => {
            let sessions = store.list_sessions()?;
            if sessions.is_empty() {
                println!("No sessions found.");
                return Ok(());
            }
            
            println!("{:<40} | {:<20} | {}", "SESSION ID", "STATUS", "WORKSPACE");
            println!("{:-<40}-+-{:-<20}-+-{:-<40}", "", "", "");
            for s in sessions {
                println!(
                    "{:<40} | {:<20} | {}",
                    s.session_id.to_string(),
                    format!("{:?}", s.status),
                    s.workspace_root
                );
            }
        }
        SessionCmd::Show { id } => {
            let session = match id {
                Some(sid) => store.load_session(Uuid::parse_str(&sid)?)?,
                None => store.load_latest_session()?,
            };
            
            match session {
                Some(s) => {
                    println!("Session ID: {}", s.session_id);
                    println!("Status: {:?}", s.status);
                    println!("Workspace: {}", s.workspace_root);
                    if let Some(c) = s.baseline_commit {
                        println!("Baseline Commit: {}", c);
                    }
                    if let Some(p) = s.active_plan_id {
                        println!("Active Plan ID: {}", p);
                    }
                }
                None => println!("Session not found."),
            }
        }
        SessionCmd::Runs { id } => {
            let session = match id {
                Some(sid) => store.load_session(Uuid::parse_str(&sid)?)?,
                None => store.load_latest_session()?,
            };
            
            match session {
                Some(s) => {
                    let runs = store.list_runs(s.session_id)?;
                    if runs.is_empty() {
                        println!("No runs found for session {}.", s.session_id);
                        return Ok(());
                    }
                    
                    println!("{:<40} | {:<20} | {:<30} | {}", "RUN ID", "STATUS", "CREATED AT", "PROMPT");
                    println!("{:-<40}-+-{:-<20}-+-{:-<30}-+-{:-<40}", "", "", "", "");
                    for r in runs {
                        let short_prompt = if r.prompt.len() > 37 {
                            format!("{}...", &r.prompt[0..37]).replace("\n", " ")
                        } else {
                            r.prompt.replace("\n", " ")
                        };
                        println!(
                            "{:<40} | {:<20} | {:<30} | {}",
                            r.run_id.to_string(),
                            format!("{:?}", r.status),
                            r.created_at,
                            short_prompt
                        );
                    }
                }
                None => println!("Session not found."),
            }
        }
        SessionCmd::Run { run_id } => {
            let run = store.load_run(Uuid::parse_str(&run_id)?)?;
            match run {
                Some(r) => {
                    println!("Run ID:     {}", r.run_id);
                    println!("Session ID: {}", r.session_id);
                    println!("Status:     {:?}", r.status);
                    println!("Created At: {}", r.created_at);
                    println!("Updated At: {}", r.updated_at);
                    println!("Prompt:\n{}", r.prompt);
                }
                None => println!("Run not found."),
            }
        }
    }
    
    Ok(())
}
