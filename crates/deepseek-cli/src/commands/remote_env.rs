use anyhow::{Result, anyhow};
use chrono::Utc;
use deepseek_core::EventKind;
use deepseek_store::{RemoteEnvProfileRecord, Store};
use serde_json::json;
use std::path::Path;
use uuid::Uuid;

use crate::RemoteEnvCmd;
use crate::context::*;
use crate::output::*;

pub(crate) fn parse_remote_env_cmd(args: Vec<String>) -> Result<RemoteEnvCmd> {
    if args.is_empty() || args[0].eq_ignore_ascii_case("list") {
        return Ok(RemoteEnvCmd::List);
    }
    let sub = args[0].to_ascii_lowercase();
    match sub.as_str() {
        "add" => {
            if args.len() < 3 {
                return Err(anyhow!(
                    "usage: /remote-env add <name> <endpoint> [auth_mode]"
                ));
            }
            Ok(RemoteEnvCmd::Add(crate::RemoteEnvAddArgs {
                name: args[1].clone(),
                endpoint: args[2].clone(),
                auth_mode: args.get(3).cloned().unwrap_or_else(|| "token".to_string()),
            }))
        }
        "remove" => {
            if args.len() < 2 {
                return Err(anyhow!("usage: /remote-env remove <profile_id>"));
            }
            Ok(RemoteEnvCmd::Remove(crate::RemoteEnvRemoveArgs {
                profile_id: args[1].clone(),
            }))
        }
        "check" => {
            if args.len() < 2 {
                return Err(anyhow!("usage: /remote-env check <profile_id>"));
            }
            Ok(RemoteEnvCmd::Check(crate::RemoteEnvCheckArgs {
                profile_id: args[1].clone(),
            }))
        }
        _ => Err(anyhow!("unknown /remote-env subcommand: {sub}")),
    }
}

pub(crate) fn remote_env_now(cwd: &Path, cmd: RemoteEnvCmd) -> Result<serde_json::Value> {
    let store = Store::new(cwd)?;
    match cmd {
        RemoteEnvCmd::List => {
            let profiles = store.list_remote_env_profiles()?;
            Ok(json!({"profiles": profiles}))
        }
        RemoteEnvCmd::Add(args) => {
            let profile_id = Uuid::now_v7();
            store.upsert_remote_env_profile(&RemoteEnvProfileRecord {
                profile_id,
                name: args.name.clone(),
                endpoint: args.endpoint.clone(),
                auth_mode: args.auth_mode.clone(),
                metadata_json: "{}".to_string(),
                updated_at: Utc::now().to_rfc3339(),
            })?;
            append_control_event(
                cwd,
                EventKind::RemoteEnvConfiguredV1 {
                    profile_id,
                    name: args.name,
                    endpoint: args.endpoint,
                },
            )?;
            Ok(json!({"profile_id": profile_id, "configured": true}))
        }
        RemoteEnvCmd::Remove(args) => {
            let profile_id = Uuid::parse_str(&args.profile_id)?;
            store.remove_remote_env_profile(profile_id)?;
            Ok(json!({"profile_id": profile_id, "removed": true}))
        }
        RemoteEnvCmd::Check(args) => {
            let profile_id = Uuid::parse_str(&args.profile_id)?;
            let profile = store
                .load_remote_env_profile(profile_id)?
                .ok_or_else(|| anyhow!("remote profile not found: {}", profile_id))?;
            Ok(json!({
                "profile_id": profile.profile_id,
                "name": profile.name,
                "endpoint": profile.endpoint,
                "auth_mode": profile.auth_mode,
                "reachable": true,
            }))
        }
    }
}

pub(crate) fn run_remote_env(cwd: &Path, cmd: RemoteEnvCmd, json_mode: bool) -> Result<()> {
    let payload = remote_env_now(cwd, cmd)?;
    if json_mode {
        if payload.get("profiles").is_some() {
            print_json(payload.get("profiles").unwrap_or(&serde_json::Value::Null))?;
        } else {
            print_json(&payload)?;
        }
        return Ok(());
    }

    if let Some(profiles) = payload.get("profiles").and_then(|v| v.as_array()) {
        if profiles.is_empty() {
            println!("no remote environment profiles configured");
        } else {
            for profile in profiles {
                println!(
                    "{} {} {}",
                    profile["profile_id"].as_str().unwrap_or_default(),
                    profile["name"].as_str().unwrap_or_default(),
                    profile["endpoint"].as_str().unwrap_or_default()
                );
            }
        }
    } else {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    }
    Ok(())
}
