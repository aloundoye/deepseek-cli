use anyhow::{Result, anyhow};
use deepseek_core::EventKind;
use deepseek_memory::{ExportFormat, MemoryManager};
use deepseek_store::Store;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::process::Command;
use uuid::Uuid;

use crate::context::*;
use crate::output::*;
use crate::util::*;
use crate::{ExportArgs, MemoryCmd};

pub(crate) fn run_export(cwd: &Path, args: ExportArgs, json_mode: bool) -> Result<()> {
    let format = ExportFormat::parse(&args.format)
        .ok_or_else(|| anyhow!("unsupported format '{}'; expected json|md", args.format))?;
    let explicit_session = args.session.as_deref().map(Uuid::parse_str).transpose()?;
    let session = if explicit_session.is_none() {
        let store = Store::new(cwd)?;
        Some(ensure_session_record(cwd, &store)?.session_id)
    } else {
        explicit_session
    };
    let output = args.output.as_deref().map(PathBuf::from);
    let memory = MemoryManager::new(cwd)?;
    let record = memory.export_transcript(format, output.as_deref(), session)?;
    append_control_event(
        cwd,
        EventKind::TranscriptExported {
            export_id: record.export_id,
            format: record.format.clone(),
            output_path: record.output_path.clone(),
        },
    )?;
    if json_mode {
        print_json(&record)?;
    } else {
        println!(
            "exported transcript {} ({}) to {}",
            record.export_id, record.format, record.output_path
        );
    }
    Ok(())
}

pub(crate) fn run_memory(cwd: &Path, cmd: MemoryCmd, json_mode: bool) -> Result<()> {
    let manager = MemoryManager::new(cwd)?;
    match cmd {
        MemoryCmd::Show(_) => {
            let path = manager.ensure_initialized()?;
            let content = manager.read_memory()?;
            if json_mode {
                print_json(&json!({
                    "path": path,
                    "content": content,
                }))?;
            } else {
                println!("{}", content);
            }
        }
        MemoryCmd::Edit(_) => {
            let path = manager.ensure_initialized()?;
            let checkpoint = manager.create_checkpoint("memory_edit")?;
            append_control_event(
                cwd,
                EventKind::CheckpointCreated {
                    checkpoint_id: checkpoint.checkpoint_id,
                    reason: checkpoint.reason.clone(),
                    files_count: checkpoint.files_count,
                    snapshot_path: checkpoint.snapshot_path.clone(),
                },
            )?;
            let editor = std::env::var("EDITOR").unwrap_or_else(|_| default_editor().to_string());
            let status = Command::new(editor).arg(&path).status()?;
            if !status.success() {
                return Err(anyhow!("editor exited with non-zero status"));
            }
            let version_id = manager.sync_memory_version("edit")?;
            append_control_event(
                cwd,
                EventKind::MemorySynced {
                    version_id,
                    path: path.to_string_lossy().to_string(),
                    note: "edit".to_string(),
                },
            )?;
            if json_mode {
                print_json(&json!({
                    "edited": true,
                    "path": path,
                    "version_id": version_id,
                    "checkpoint_id": checkpoint.checkpoint_id
                }))?;
            } else {
                println!("updated {}", path.display());
            }
        }
        MemoryCmd::Sync(args) => {
            let path = manager.ensure_initialized()?;
            let note = args.note.unwrap_or_else(|| "sync".to_string());
            let version_id = manager.sync_memory_version(&note)?;
            append_control_event(
                cwd,
                EventKind::MemorySynced {
                    version_id,
                    path: path.to_string_lossy().to_string(),
                    note: note.clone(),
                },
            )?;
            if json_mode {
                print_json(&json!({
                    "synced": true,
                    "path": path,
                    "version_id": version_id,
                    "note": note,
                }))?;
            } else {
                println!("memory synced version={} note={}", version_id, note);
            }
        }
    }
    Ok(())
}
