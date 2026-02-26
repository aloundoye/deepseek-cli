use anyhow::{Result, anyhow};
use deepseek_core::EventKind;
use deepseek_diff::PatchStore;
use deepseek_memory::MemoryManager;
use serde_json::json;
use std::path::Path;
use uuid::Uuid;

use crate::ApplyArgs;
use crate::context::*;
use crate::output::*;

pub(crate) fn run_diff(cwd: &Path, json_mode: bool) -> Result<()> {
    let patches = PatchStore::new(cwd)?.list()?;
    if json_mode {
        print_json(&patches)?;
        return Ok(());
    }
    if patches.is_empty() {
        println!("No staged patches.");
        return Ok(());
    }
    for p in patches {
        println!(
            "patch_id={} applied={} created_at={}",
            p.patch_id, p.applied, p.created_at
        );
        println!("{}", p.unified_diff);
    }
    Ok(())
}

pub(crate) fn run_apply(cwd: &Path, args: ApplyArgs, json_mode: bool) -> Result<()> {
    let store = PatchStore::new(cwd)?;
    let patches = store.list()?;
    let patch = if let Some(id) = args.patch_id {
        let uid = Uuid::parse_str(&id)?;
        patches
            .into_iter()
            .find(|p| p.patch_id == uid)
            .ok_or_else(|| anyhow!("patch_id not found"))?
    } else {
        patches
            .into_iter()
            .last()
            .ok_or_else(|| anyhow!("no staged patch found"))?
    };

    if !args.yes {
        return Err(anyhow!("approval required: pass --yes to apply"));
    }

    let checkpoint = MemoryManager::new(cwd)?.create_checkpoint("patch_apply")?;
    append_control_event(
        cwd,
        EventKind::CheckpointCreated {
            checkpoint_id: checkpoint.checkpoint_id,
            reason: checkpoint.reason.clone(),
            files_count: checkpoint.files_count,
            snapshot_path: checkpoint.snapshot_path.clone(),
        },
    )?;

    let (applied, conflicts) = store.apply(cwd, patch.patch_id)?;
    if json_mode {
        print_json(&json!({
            "patch_id": patch.patch_id,
            "applied": applied,
            "conflicts": conflicts,
            "checkpoint_id": checkpoint.checkpoint_id
        }))?;
        return Ok(());
    }
    if applied {
        println!("Applied patch {}", patch.patch_id);
    } else {
        println!("Failed to apply patch {}", patch.patch_id);
        for c in conflicts {
            println!("conflict: {c}");
        }
    }
    Ok(())
}
