use anyhow::{Result, anyhow};
use chrono::Utc;
use deepseek_core::{EventKind, runtime_dir};
use deepseek_store::Store;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

use crate::TeleportArgs;
use crate::context::*;
use crate::output::*;

#[derive(Default)]
pub(crate) struct TeleportExecution {
    pub(crate) bundle_id: Option<Uuid>,
    pub(crate) path: Option<String>,
    pub(crate) imported: Option<String>,
}

pub(crate) fn parse_teleport_args(args: Vec<String>) -> Result<TeleportArgs> {
    if args.is_empty() {
        return Ok(TeleportArgs::default());
    }
    if args.len() >= 2 && args[0].eq_ignore_ascii_case("import") {
        return Ok(TeleportArgs {
            session_id: None,
            output: None,
            import: Some(args[1].clone()),
        });
    }
    let mut parsed = TeleportArgs::default();
    let mut idx = 0usize;
    while idx < args.len() {
        let token = &args[idx];
        if token.eq_ignore_ascii_case("session") && idx + 1 < args.len() {
            parsed.session_id = Some(args[idx + 1].clone());
            idx += 2;
            continue;
        }
        if token.eq_ignore_ascii_case("output") && idx + 1 < args.len() {
            parsed.output = Some(args[idx + 1].clone());
            idx += 2;
            continue;
        }
        if token.eq_ignore_ascii_case("import") && idx + 1 < args.len() {
            parsed.import = Some(args[idx + 1].clone());
            idx += 2;
            continue;
        }
        if parsed.output.is_none() {
            parsed.output = Some(token.clone());
        }
        idx += 1;
    }
    Ok(parsed)
}

pub(crate) fn teleport_now(cwd: &Path, args: TeleportArgs) -> Result<TeleportExecution> {
    if let Some(import_path) = args.import {
        let raw = fs::read_to_string(&import_path)?;
        let _: serde_json::Value = serde_json::from_str(&raw)?;
        return Ok(TeleportExecution {
            imported: Some(import_path),
            ..TeleportExecution::default()
        });
    }

    let store = Store::new(cwd)?;
    let session_id = if let Some(session_id) = args.session_id {
        Uuid::parse_str(&session_id)?
    } else {
        ensure_session_record(cwd, &store)?.session_id
    };
    let projection = store.rebuild_from_events(session_id)?;
    let bundle_id = Uuid::now_v7();
    let output_path = args.output.map(PathBuf::from).unwrap_or_else(|| {
        runtime_dir(cwd)
            .join("teleport")
            .join(format!("{bundle_id}.json"))
    });
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let payload = json!({
        "bundle_id": bundle_id,
        "session_id": session_id,
        "created_at": Utc::now().to_rfc3339(),
        "turns": projection.transcript,
        "steps": projection.step_status,
    });
    fs::write(&output_path, serde_json::to_vec_pretty(&payload)?)?;
    append_control_event(
        cwd,
        EventKind::TeleportBundleCreatedV1 {
            bundle_id,
            path: output_path.to_string_lossy().to_string(),
        },
    )?;
    Ok(TeleportExecution {
        bundle_id: Some(bundle_id),
        path: Some(output_path.to_string_lossy().to_string()),
        imported: None,
    })
}

pub(crate) fn run_teleport(cwd: &Path, args: TeleportArgs, json_mode: bool) -> Result<()> {
    let result = teleport_now(cwd, args)?;
    if let Some(imported) = result.imported {
        if json_mode {
            print_json(&json!({"imported": true, "path": imported}))?;
        } else {
            println!("imported teleport bundle from {}", imported);
        }
    } else {
        let bundle_id = result
            .bundle_id
            .ok_or_else(|| anyhow!("missing bundle id for teleport export"))?;
        let path = result
            .path
            .ok_or_else(|| anyhow!("missing output path for teleport export"))?;
        if json_mode {
            print_json(&json!({"bundle_id": bundle_id, "path": path}))?;
        } else {
            println!("teleport bundle created at {}", path);
        }
    }
    Ok(())
}
