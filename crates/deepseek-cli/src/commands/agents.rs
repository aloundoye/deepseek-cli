use anyhow::Result;
use deepseek_subagent::{CustomAgentDef, SubagentConfig, load_agent_defs};
use serde_json::json;
use std::path::Path;

use crate::AgentsCmd;
use crate::output::*;

pub(crate) fn run_agents(cwd: &Path, cmd: AgentsCmd, json_mode: bool) -> Result<()> {
    match cmd {
        AgentsCmd::List => {
            let defs = load_agent_defs(cwd)?;
            if json_mode {
                print_json(&defs)?;
            } else if defs.is_empty() {
                println!("No agent definitions found.");
                println!(
                    "Create .md files in .deepseek/agents/ or ~/.deepseek/agents/ with YAML frontmatter."
                );
            } else {
                println!("{:<20} {:<40} MODEL", "NAME", "DESCRIPTION");
                println!("{}", "-".repeat(70));
                for def in &defs {
                    println!(
                        "{:<20} {:<40} {}",
                        def.name,
                        truncate_str(&def.description, 38),
                        def.model.as_deref().unwrap_or("(default)"),
                    );
                }
                println!("\n{} agent(s) loaded", defs.len());
            }
        }
        AgentsCmd::Show { name } => {
            let defs = load_agent_defs(cwd)?;
            let def = defs
                .iter()
                .find(|d| d.name == name)
                .ok_or_else(|| anyhow::anyhow!("agent '{}' not found", name))?;
            if json_mode {
                let config = SubagentConfig::from_agent_def(def);
                print_json(&json!({
                    "definition": def,
                    "config": config,
                }))?;
            } else {
                print_agent_detail(def);
            }
        }
        AgentsCmd::Create { name } => {
            let agents_dir = cwd.join(".deepseek/agents");
            std::fs::create_dir_all(&agents_dir)?;
            let path = agents_dir.join(format!("{name}.md"));
            if path.exists() {
                anyhow::bail!("agent definition already exists: {}", path.display());
            }
            let template = format!(
                r#"---
name: {name}
description: Custom agent for specialized tasks
model: deepseek-chat
max_turns: 30
tools: [fs_read, fs_glob, fs_grep, bash_run]
disallowed_tools: []
---
# {name}

You are a specialized agent. Your task is to help with specific workloads.

## Instructions

- Use the available tools to accomplish your goals
- Be thorough and verify your work
- Report findings clearly
"#
            );
            std::fs::write(&path, template)?;
            if json_mode {
                print_json(&json!({
                    "created": path.display().to_string(),
                    "name": name,
                }))?;
            } else {
                println!("Created agent definition: {}", path.display());
                println!("Edit the file to customize the agent's behavior.");
            }
        }
    }
    Ok(())
}

fn print_agent_detail(def: &CustomAgentDef) {
    let config = SubagentConfig::from_agent_def(def);
    println!("Agent: {}", def.name);
    println!("Description: {}", def.description);
    println!("Model: {}", def.model.as_deref().unwrap_or("(default)"));
    println!("Max turns: {}", config.max_turns);
    if !def.tools.is_empty() {
        println!("Allowed tools: {}", def.tools.join(", "));
    }
    if !def.disallowed_tools.is_empty() {
        println!("Disallowed tools: {}", def.disallowed_tools.join(", "));
    }
    println!("\n--- Prompt ---\n{}", def.prompt);
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}
