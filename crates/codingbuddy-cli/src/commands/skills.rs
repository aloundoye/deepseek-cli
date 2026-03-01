use anyhow::Result;
use chrono::Utc;
use codingbuddy_agent::{AgentEngine, ChatOptions};
use codingbuddy_core::{AppConfig, EventKind};
use codingbuddy_skills::SkillManager;
use codingbuddy_store::{SkillRegistryRecord, Store};
use serde_json::json;
use std::path::Path;

use crate::SkillsCmd;
use crate::context::*;
use crate::output::*;
use crate::util::*;

pub(crate) fn run_skills(cwd: &Path, cmd: SkillsCmd, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;
    let manager = SkillManager::new(cwd)?;
    let paths = cfg
        .skills
        .paths
        .iter()
        .map(|path| expand_tilde(path))
        .collect::<Vec<_>>();
    let store = Store::new(cwd)?;
    match cmd {
        SkillsCmd::List => {
            let skills = manager.list(&paths)?;
            if json_mode {
                print_json(&skills)?;
            } else if skills.is_empty() {
                println!("no skills found");
            } else {
                for skill in skills {
                    println!("{} {} ({})", skill.id, skill.name, skill.path.display());
                }
            }
        }
        SkillsCmd::Install(args) => {
            let installed = manager.install(Path::new(&args.source))?;
            store.set_skill_registry(&SkillRegistryRecord {
                skill_id: installed.id.clone(),
                name: installed.name.clone(),
                path: installed.path.to_string_lossy().to_string(),
                enabled: true,
                metadata_json: serde_json::json!({"summary": installed.summary}).to_string(),
                updated_at: Utc::now().to_rfc3339(),
            })?;
            append_control_event(
                cwd,
                EventKind::SkillLoaded {
                    skill_id: installed.id.clone(),
                    source_path: installed.path.to_string_lossy().to_string(),
                },
            )?;
            if json_mode {
                print_json(&installed)?;
            } else {
                println!(
                    "installed skill {} ({})",
                    installed.id,
                    installed.path.display()
                );
            }
        }
        SkillsCmd::Remove(args) => {
            manager.remove(&args.skill_id)?;
            store.remove_skill_registry(&args.skill_id)?;
            if json_mode {
                print_json(&json!({"removed": args.skill_id}))?;
            } else {
                println!("removed skill {}", args.skill_id);
            }
        }
        SkillsCmd::Run(args) => {
            let run = manager.run(&args.skill_id, args.input.as_deref(), &paths)?;
            if args.execute {
                ensure_llm_ready(cwd, json_mode)?;
                let output = AgentEngine::new(cwd)?.chat_with_options(
                    &run.rendered_prompt,
                    ChatOptions {
                        tools: false,
                        ..Default::default()
                    },
                )?;
                if json_mode {
                    print_json(&json!({"skill": run, "output": output}))?;
                } else {
                    println!("{output}");
                }
            } else if json_mode {
                print_json(&run)?;
            } else {
                println!("{}", run.rendered_prompt);
            }
        }
        SkillsCmd::Reload => {
            let loaded = manager.reload(&paths)?;
            for skill in &loaded {
                store.set_skill_registry(&SkillRegistryRecord {
                    skill_id: skill.id.clone(),
                    name: skill.name.clone(),
                    path: skill.path.to_string_lossy().to_string(),
                    enabled: true,
                    metadata_json: serde_json::json!({"summary": skill.summary}).to_string(),
                    updated_at: Utc::now().to_rfc3339(),
                })?;
            }
            if json_mode {
                print_json(&json!({"reloaded": loaded.len(), "skills": loaded}))?;
            } else {
                println!("reloaded {} skills", loaded.len());
            }
        }
    }
    Ok(())
}
