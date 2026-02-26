use anyhow::{Context, Result, anyhow};
use chrono::{Duration as ChronoDuration, Utc};
use deepseek_core::{
    AppConfig, ChatMessage, EventEnvelope, EventKind, Session, SessionBudgets, SessionState,
    runtime_dir,
};
use deepseek_store::Store;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use uuid::Uuid;

use crate::context::*;
use crate::output::*;
use crate::{
    TeleportArgs, TeleportCmd, TeleportConsumeArgs, TeleportExportArgs, TeleportImportArgs,
    TeleportLinkArgs,
};

#[derive(Default)]
pub(crate) struct TeleportExecution {
    pub(crate) mode: String,
    pub(crate) bundle_id: Option<Uuid>,
    pub(crate) path: Option<String>,
    pub(crate) imported: Option<String>,
    pub(crate) handoff_id: Option<Uuid>,
    pub(crate) link_url: Option<String>,
    pub(crate) token: Option<String>,
    pub(crate) session_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HandoffDescriptor {
    schema: String,
    handoff_id: Uuid,
    bundle_path: String,
    token_hash: String,
    created_at: String,
    expires_at: String,
    used_at: Option<String>,
    session_id: Uuid,
}

pub(crate) fn parse_teleport_args(args: Vec<String>) -> Result<TeleportArgs> {
    if args.is_empty() {
        return Ok(TeleportArgs::default());
    }

    let first = args[0].to_ascii_lowercase();
    if first == "link" {
        let mut parsed = TeleportLinkArgs {
            ttl_minutes: 30,
            ..Default::default()
        };
        let mut idx = 1usize;
        while idx < args.len() {
            let token = &args[idx];
            match token.as_str() {
                "--session-id" | "session" if idx + 1 < args.len() => {
                    parsed.session_id = Some(args[idx + 1].clone());
                    idx += 2;
                }
                "--base-url" | "base-url" if idx + 1 < args.len() => {
                    parsed.base_url = Some(args[idx + 1].clone());
                    idx += 2;
                }
                "--ttl-minutes" | "ttl-minutes" if idx + 1 < args.len() => {
                    if let Ok(value) = args[idx + 1].parse::<u64>() {
                        parsed.ttl_minutes = value;
                    }
                    idx += 2;
                }
                "--open" | "open" => {
                    parsed.open = true;
                    idx += 1;
                }
                other if parsed.base_url.is_none() && other.starts_with("http") => {
                    parsed.base_url = Some(other.to_string());
                    idx += 1;
                }
                _ => {
                    idx += 1;
                }
            }
        }
        return Ok(TeleportArgs {
            command: Some(TeleportCmd::Link(parsed)),
            ..TeleportArgs::default()
        });
    }

    if first == "consume" {
        let mut parsed = TeleportConsumeArgs::default();
        let mut idx = 1usize;
        while idx < args.len() {
            let token = &args[idx];
            match token.as_str() {
                "--handoff-id" | "handoff-id" if idx + 1 < args.len() => {
                    parsed.handoff_id = args[idx + 1].clone();
                    idx += 2;
                }
                "--token" | "token" if idx + 1 < args.len() => {
                    parsed.token = args[idx + 1].clone();
                    idx += 2;
                }
                _ if parsed.handoff_id.is_empty() => {
                    parsed.handoff_id = token.clone();
                    idx += 1;
                }
                _ if parsed.token.is_empty() => {
                    parsed.token = token.clone();
                    idx += 1;
                }
                _ => {
                    idx += 1;
                }
            }
        }
        return Ok(TeleportArgs {
            command: Some(TeleportCmd::Consume(parsed)),
            ..TeleportArgs::default()
        });
    }

    if first == "export" {
        let mut export = TeleportExportArgs::default();
        let mut idx = 1usize;
        while idx < args.len() {
            let token = &args[idx];
            if (token == "--session-id" || token == "session") && idx + 1 < args.len() {
                export.session_id = Some(args[idx + 1].clone());
                idx += 2;
                continue;
            }
            if (token == "--output" || token == "output") && idx + 1 < args.len() {
                export.output = Some(args[idx + 1].clone());
                idx += 2;
                continue;
            }
            if export.output.is_none() {
                export.output = Some(token.clone());
            }
            idx += 1;
        }
        return Ok(TeleportArgs {
            command: Some(TeleportCmd::Export(export)),
            ..TeleportArgs::default()
        });
    }

    if first == "import" {
        let input = args
            .get(1)
            .cloned()
            .ok_or_else(|| anyhow!("usage: /teleport import <path>"))?;
        return Ok(TeleportArgs {
            command: Some(TeleportCmd::Import(TeleportImportArgs { input })),
            ..TeleportArgs::default()
        });
    }

    // Legacy fallback syntax.
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
    run_teleport_internal(cwd, args)
}

pub(crate) fn run_teleport(cwd: &Path, args: TeleportArgs, json_mode: bool) -> Result<()> {
    let result = run_teleport_internal(cwd, args)?;
    if json_mode {
        print_json(&teleport_result_payload(&result))?;
    } else {
        render_teleport_result_text(&result);
    }
    Ok(())
}

fn run_teleport_internal(cwd: &Path, args: TeleportArgs) -> Result<TeleportExecution> {
    match args.command {
        Some(TeleportCmd::Export(export)) => export_bundle(cwd, export),
        Some(TeleportCmd::Import(import)) => import_bundle(cwd, &import.input),
        Some(TeleportCmd::Link(link)) => create_handoff_link(cwd, link),
        Some(TeleportCmd::Consume(consume)) => consume_handoff_link(cwd, consume),
        None => {
            if let Some(import_path) = args.import {
                import_bundle(cwd, &import_path)
            } else {
                export_bundle(
                    cwd,
                    TeleportExportArgs {
                        session_id: args.session_id,
                        output: args.output,
                    },
                )
            }
        }
    }
}

fn export_bundle(cwd: &Path, args: TeleportExportArgs) -> Result<TeleportExecution> {
    let store = Store::new(cwd)?;
    let session = if let Some(session_id) = args.session_id {
        let parsed = Uuid::parse_str(&session_id)?;
        store
            .load_session(parsed)?
            .ok_or_else(|| anyhow!("session not found: {parsed}"))?
    } else {
        ensure_session_record(cwd, &store)?
    };

    let projection = store.rebuild_from_events(session.session_id)?;
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
        "schema": "deepseek.desktop_handoff.v2",
        "bundle_id": bundle_id,
        "session_id": session.session_id,
        "workspace_root": session.workspace_root,
        "created_at": Utc::now().to_rfc3339(),
        "chat_messages": projection.chat_messages,
        "transcript": projection.transcript,
        "step_status": projection.step_status,
    });
    fs::write(&output_path, serde_json::to_vec_pretty(&payload)?)?;

    append_control_event(
        cwd,
        EventKind::TeleportBundleCreated {
            bundle_id,
            path: output_path.to_string_lossy().to_string(),
        },
    )?;

    Ok(TeleportExecution {
        mode: "export".to_string(),
        bundle_id: Some(bundle_id),
        path: Some(output_path.to_string_lossy().to_string()),
        session_id: Some(session.session_id),
        ..TeleportExecution::default()
    })
}

fn import_bundle(cwd: &Path, input: &str) -> Result<TeleportExecution> {
    let raw = fs::read_to_string(input)
        .with_context(|| format!("failed to read teleport bundle: {}", input))?;
    let payload: serde_json::Value = serde_json::from_str(&raw)?;

    let store = Store::new(cwd)?;
    let session_id = Uuid::now_v7();
    let workspace_root = payload
        .get("workspace_root")
        .and_then(|v| v.as_str())
        .map(ToString::to_string)
        .unwrap_or_else(|| cwd.to_string_lossy().to_string());
    let session = Session {
        session_id,
        workspace_root,
        baseline_commit: None,
        status: SessionState::Idle,
        budgets: SessionBudgets {
            per_turn_seconds: 300,
            max_think_tokens: 8192,
        },
        active_plan_id: None,
    };
    store.save_session(&session)?;

    let mut replayed = 0u64;
    if let Some(messages_value) = payload.get("chat_messages") {
        if let Ok(messages) = serde_json::from_value::<Vec<ChatMessage>>(messages_value.clone()) {
            for message in messages {
                store.append_event(&EventEnvelope {
                    seq_no: store.next_seq_no(session_id)?,
                    at: Utc::now(),
                    session_id,
                    kind: EventKind::ChatTurn { message },
                })?;
                replayed = replayed.saturating_add(1);
            }
        }
    } else if let Some(turns) = payload.get("turns").and_then(|value| value.as_array()) {
        for turn in turns {
            if let Some(content) = turn.as_str() {
                store.append_event(&EventEnvelope {
                    seq_no: store.next_seq_no(session_id)?,
                    at: Utc::now(),
                    session_id,
                    kind: EventKind::TurnAdded {
                        role: "imported".to_string(),
                        content: content.to_string(),
                    },
                })?;
                replayed = replayed.saturating_add(1);
            }
        }
    }

    store.append_event(&EventEnvelope {
        seq_no: store.next_seq_no(session_id)?,
        at: Utc::now(),
        session_id,
        kind: EventKind::SessionResumed {
            session_id,
            events_replayed: replayed,
        },
    })?;

    Ok(TeleportExecution {
        mode: "import".to_string(),
        imported: Some(input.to_string()),
        session_id: Some(session_id),
        ..TeleportExecution::default()
    })
}

fn create_handoff_link(cwd: &Path, args: TeleportLinkArgs) -> Result<TeleportExecution> {
    let ttl_minutes = args.ttl_minutes.clamp(1, 24 * 60);
    let handoff_id = Uuid::now_v7();
    let bundle_path = runtime_dir(cwd)
        .join("teleport")
        .join("handoff_bundles")
        .join(format!("{handoff_id}.json"));

    let export = export_bundle(
        cwd,
        TeleportExportArgs {
            session_id: args.session_id,
            output: Some(bundle_path.to_string_lossy().to_string()),
        },
    )?;
    let session_id = export
        .session_id
        .ok_or_else(|| anyhow!("missing session id for handoff export"))?;

    let token = format!("{}{}", Uuid::now_v7().simple(), Uuid::now_v7().simple());
    let created_at = Utc::now();
    let expires_at = created_at + ChronoDuration::minutes(ttl_minutes as i64);

    let descriptor = HandoffDescriptor {
        schema: "deepseek.handoff_link.v1".to_string(),
        handoff_id,
        bundle_path: bundle_path.to_string_lossy().to_string(),
        token_hash: sha256_hex(&token),
        created_at: created_at.to_rfc3339(),
        expires_at: expires_at.to_rfc3339(),
        used_at: None,
        session_id,
    };
    write_handoff_descriptor(cwd, &descriptor)?;

    append_control_event(
        cwd,
        EventKind::TeleportHandoffLinkCreated {
            handoff_id,
            session_id,
            expires_at: descriptor.expires_at.clone(),
        },
    )?;
    append_control_event(
        cwd,
        EventKind::TelemetryEvent {
            name: "kpi.teleport.link_created".to_string(),
            properties: json!({
                "handoff_id": handoff_id,
                "ttl_minutes": ttl_minutes,
            }),
        },
    )?;

    let cfg = AppConfig::load(cwd).unwrap_or_default();
    let base_url = args
        .base_url
        .or(cfg.ui.handoff_base_url)
        .unwrap_or_else(|| "https://app.deepseek.com/handoff".to_string());
    let sep = if base_url.contains('?') { '&' } else { '?' };
    let link_url = format!("{base_url}{sep}handoff_id={handoff_id}&token={token}");

    if args.open {
        let _ = open_external_target(&link_url);
    }

    Ok(TeleportExecution {
        mode: "link".to_string(),
        bundle_id: export.bundle_id,
        path: Some(bundle_path.to_string_lossy().to_string()),
        handoff_id: Some(handoff_id),
        link_url: Some(link_url),
        token: Some(token),
        session_id: Some(session_id),
        ..TeleportExecution::default()
    })
}

fn consume_handoff_link(cwd: &Path, args: TeleportConsumeArgs) -> Result<TeleportExecution> {
    let handoff_id = Uuid::parse_str(&args.handoff_id)
        .with_context(|| format!("invalid handoff id: {}", args.handoff_id))?;
    let mut descriptor = read_handoff_descriptor(cwd, handoff_id)?;

    if descriptor.used_at.is_some() {
        record_handoff_consume(cwd, &descriptor, false, "already_used")?;
        return Err(anyhow!("handoff token already used"));
    }
    let now = Utc::now();
    let expires = chrono::DateTime::parse_from_rfc3339(&descriptor.expires_at)
        .map(|v| v.with_timezone(&Utc))
        .context("invalid handoff descriptor expiry")?;
    if now > expires {
        record_handoff_consume(cwd, &descriptor, false, "expired")?;
        return Err(anyhow!(
            "handoff token expired at {}",
            descriptor.expires_at
        ));
    }
    if sha256_hex(&args.token) != descriptor.token_hash {
        record_handoff_consume(cwd, &descriptor, false, "token_mismatch")?;
        return Err(anyhow!("invalid handoff token"));
    }

    let imported = import_bundle(cwd, &descriptor.bundle_path)?;
    descriptor.used_at = Some(Utc::now().to_rfc3339());
    write_handoff_descriptor(cwd, &descriptor)?;
    record_handoff_consume(cwd, &descriptor, true, "consumed")?;

    Ok(TeleportExecution {
        mode: "consume".to_string(),
        handoff_id: Some(handoff_id),
        imported: Some(descriptor.bundle_path),
        session_id: imported.session_id,
        ..TeleportExecution::default()
    })
}

fn record_handoff_consume(
    cwd: &Path,
    descriptor: &HandoffDescriptor,
    success: bool,
    reason: &str,
) -> Result<()> {
    append_control_event(
        cwd,
        EventKind::TeleportHandoffLinkConsumed {
            handoff_id: descriptor.handoff_id,
            session_id: descriptor.session_id,
            success,
            reason: reason.to_string(),
        },
    )?;
    append_control_event(
        cwd,
        EventKind::TelemetryEvent {
            name: "kpi.teleport.link_consume".to_string(),
            properties: json!({
                "handoff_id": descriptor.handoff_id,
                "success": success,
                "reason": reason,
            }),
        },
    )?;
    Ok(())
}

fn handoff_directory(cwd: &Path) -> PathBuf {
    runtime_dir(cwd).join("teleport").join("handoffs")
}

fn descriptor_path(cwd: &Path, handoff_id: Uuid) -> PathBuf {
    handoff_directory(cwd).join(format!("{handoff_id}.json"))
}

fn write_handoff_descriptor(cwd: &Path, descriptor: &HandoffDescriptor) -> Result<()> {
    let path = descriptor_path(cwd, descriptor.handoff_id);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(descriptor)?)?;
    Ok(())
}

fn read_handoff_descriptor(cwd: &Path, handoff_id: Uuid) -> Result<HandoffDescriptor> {
    let path = descriptor_path(cwd, handoff_id);
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("handoff descriptor not found: {}", path.display()))?;
    let descriptor = serde_json::from_str::<HandoffDescriptor>(&raw)?;
    Ok(descriptor)
}

fn sha256_hex(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn open_external_target(target: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let status = Command::new("open").arg(target).status()?;
        if !status.success() {
            return Err(anyhow!("open failed for {}", target));
        }
        return Ok(());
    }
    #[cfg(target_os = "linux")]
    {
        let status = Command::new("xdg-open").arg(target).status()?;
        if !status.success() {
            return Err(anyhow!("xdg-open failed for {}", target));
        }
        return Ok(());
    }
    #[cfg(target_os = "windows")]
    {
        let status = Command::new("cmd")
            .args(["/C", "start", "", target])
            .status()?;
        if !status.success() {
            return Err(anyhow!("start failed for {}", target));
        }
        return Ok(());
    }
    #[allow(unreachable_code)]
    Err(anyhow!("external open is unsupported on this platform"))
}

fn teleport_result_payload(result: &TeleportExecution) -> serde_json::Value {
    json!({
        "schema": if result.mode == "link" || result.mode == "consume" {
            "deepseek.handoff_link.v1"
        } else {
            "deepseek.desktop_handoff.v2"
        },
        "mode": result.mode,
        "bundle_id": result.bundle_id,
        "path": result.path,
        "imported": result.imported,
        "handoff_id": result.handoff_id,
        "link_url": result.link_url,
        "token": result.token,
        "session_id": result.session_id,
    })
}

fn render_teleport_result_text(result: &TeleportExecution) {
    match result.mode.as_str() {
        "export" => {
            println!(
                "teleport bundle created at {}",
                result.path.as_deref().unwrap_or_default()
            );
        }
        "import" => {
            println!(
                "imported teleport bundle {} into session {}",
                result.imported.as_deref().unwrap_or_default(),
                result
                    .session_id
                    .map(|id| id.to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            );
        }
        "link" => {
            println!(
                "handoff link created: {}",
                result.link_url.as_deref().unwrap_or_default()
            );
            println!("handoff id: {}", result.handoff_id.unwrap_or_default());
        }
        "consume" => {
            println!(
                "handoff consumed: {}",
                result.handoff_id.unwrap_or_default()
            );
            if let Some(session_id) = result.session_id {
                println!("session resumed: {}", session_id);
            }
        }
        _ => {
            println!(
                "{}",
                serde_json::to_string_pretty(&teleport_result_payload(result)).unwrap_or_default()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_teleport_link_args_from_slash() {
        let args = parse_teleport_args(vec![
            "link".to_string(),
            "--ttl-minutes".to_string(),
            "45".to_string(),
        ])
        .expect("parse");
        match args.command {
            Some(TeleportCmd::Link(link)) => assert_eq!(link.ttl_minutes, 45),
            _ => panic!("expected link command"),
        }
    }

    #[test]
    fn parse_teleport_consume_args_from_slash() {
        let args = parse_teleport_args(vec![
            "consume".to_string(),
            "123e4567-e89b-12d3-a456-426614174000".to_string(),
            "tok_abc".to_string(),
        ])
        .expect("parse");
        match args.command {
            Some(TeleportCmd::Consume(consume)) => {
                assert_eq!(consume.handoff_id, "123e4567-e89b-12d3-a456-426614174000");
                assert_eq!(consume.token, "tok_abc");
            }
            _ => panic!("expected consume command"),
        }
    }

    #[test]
    fn sha256_hex_is_stable() {
        assert_eq!(sha256_hex("abc"), sha256_hex("abc"));
    }
}
