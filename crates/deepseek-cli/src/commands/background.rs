use anyhow::{Result, anyhow};
use chrono::Utc;
use deepseek_core::{AppConfig, EventEnvelope, EventKind, runtime_dir};
use deepseek_store::{BackgroundJobRecord, Store};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use uuid::Uuid;

use crate::context::*;
use crate::output::*;
use crate::{
    BackgroundAttachArgs, BackgroundCmd, BackgroundRunAgentArgs, BackgroundRunShellArgs,
    BackgroundStopArgs,
};

pub(crate) fn parse_background_cmd(args: Vec<String>) -> Result<BackgroundCmd> {
    if args.is_empty() || args[0].eq_ignore_ascii_case("list") {
        return Ok(BackgroundCmd::List);
    }
    let first = args[0].to_ascii_lowercase();
    match first.as_str() {
        "attach" => {
            let job_id = args
                .get(1)
                .cloned()
                .ok_or_else(|| anyhow!("usage: /background attach <job_id>"))?;
            Ok(BackgroundCmd::Attach(BackgroundAttachArgs {
                job_id,
                tail_lines: 40,
            }))
        }
        "stop" => {
            let job_id = args
                .get(1)
                .cloned()
                .ok_or_else(|| anyhow!("usage: /background stop <job_id>"))?;
            Ok(BackgroundCmd::Stop(BackgroundStopArgs { job_id }))
        }
        "run-agent" | "agent" => {
            let prompt = args[1..].to_vec();
            if prompt.is_empty() {
                return Err(anyhow!("usage: /background run-agent <prompt>"));
            }
            Ok(BackgroundCmd::RunAgent(BackgroundRunAgentArgs {
                prompt,
                tools: true,
            }))
        }
        "run-shell" | "shell" => {
            let command = args[1..].to_vec();
            if command.is_empty() {
                return Err(anyhow!("usage: /background run-shell <command>"));
            }
            Ok(BackgroundCmd::RunShell(BackgroundRunShellArgs { command }))
        }
        _ => Err(anyhow!(
            "use /background list|attach|stop|run-agent|run-shell"
        )),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub(crate) fn run_background(cwd: &Path, cmd: BackgroundCmd, json_mode: bool) -> Result<()> {
    let payload = background_payload(cwd, cmd)?;
    if json_mode {
        print_json(&payload)?;
        return Ok(());
    }

    if let Some(rows) = payload.as_array() {
        if rows.is_empty() {
            println!("no background jobs");
            return Ok(());
        }
        for row in rows {
            println!(
                "{} {} {} {}",
                row["job_id"].as_str().unwrap_or_default(),
                row["kind"].as_str().unwrap_or_default(),
                row["status"].as_str().unwrap_or_default(),
                row["reference"].as_str().unwrap_or_default(),
            );
        }
        return Ok(());
    }

    if payload
        .get("stopped")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
    {
        println!(
            "stopped background job {}",
            payload["job_id"].as_str().unwrap_or_default(),
        );
        return Ok(());
    }

    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(())
}

pub(crate) fn background_payload(cwd: &Path, cmd: BackgroundCmd) -> Result<serde_json::Value> {
    let store = Store::new(cwd)?;
    match cmd {
        BackgroundCmd::List => Ok(serde_json::to_value(store.list_background_jobs()?)?),
        BackgroundCmd::Attach(args) => {
            let job_id = Uuid::parse_str(&args.job_id)?;
            let job = store
                .load_background_job(job_id)?
                .ok_or_else(|| anyhow!("background job not found: {}", args.job_id))?;
            append_control_event(
                cwd,
                EventKind::BackgroundJobResumed {
                    job_id,
                    reference: job.reference.clone(),
                },
            )?;
            let metadata =
                serde_json::from_str::<serde_json::Value>(&job.metadata_json).unwrap_or_default();
            let stdout_tail = metadata
                .get("stdout_log")
                .and_then(|v| v.as_str())
                .and_then(|path| tail_file_lines(Path::new(path), args.tail_lines));
            let stderr_tail = metadata
                .get("stderr_log")
                .and_then(|v| v.as_str())
                .and_then(|path| tail_file_lines(Path::new(path), args.tail_lines));
            Ok(json!({
                "job_id": job_id,
                "kind": job.kind,
                "status": job.status,
                "reference": job.reference,
                "metadata": metadata,
                "log_tail": {
                    "stdout": stdout_tail,
                    "stderr": stderr_tail
                }
            }))
        }
        BackgroundCmd::Stop(args) => {
            let job_id = Uuid::parse_str(&args.job_id)?;
            let mut job = store
                .load_background_job(job_id)?
                .ok_or_else(|| anyhow!("background job not found: {}", args.job_id))?;
            let metadata =
                serde_json::from_str::<serde_json::Value>(&job.metadata_json).unwrap_or_default();
            if job.kind == "autopilot"
                && let Ok(run_id) = Uuid::parse_str(&job.reference)
                && let Some(run) = store.load_autopilot_run(run_id)?
            {
                let stop_path = if run.stop_file.trim().is_empty() {
                    runtime_dir(cwd).join("autopilot.stop")
                } else {
                    PathBuf::from(run.stop_file)
                };
                if let Some(parent) = stop_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(
                    &stop_path,
                    format!("stop requested at {}\n", Utc::now().to_rfc3339()),
                )?;
            }
            let mut terminated_pid = false;
            if let Some(pid) = metadata.get("pid").and_then(|value| value.as_u64()) {
                match terminate_background_pid(pid as u32) {
                    Ok(_) => terminated_pid = true,
                    Err(err) => {
                        eprintln!("warning: failed to terminate background pid {pid}: {err}");
                    }
                }
            }
            job.status = "stopped".to_string();
            job.updated_at = Utc::now().to_rfc3339();
            job.metadata_json = serde_json::json!({
                "reason":"manual_stop",
                "terminated_pid": terminated_pid,
                "previous": metadata,
            })
            .to_string();
            store.upsert_background_job(&job)?;
            append_control_event(
                cwd,
                EventKind::BackgroundJobStopped {
                    job_id,
                    reason: "manual_stop".to_string(),
                },
            )?;
            Ok(json!({
                "job_id": job_id,
                "stopped": true,
                "terminated_pid": terminated_pid
            }))
        }
        BackgroundCmd::RunAgent(args) => {
            let prompt = args.prompt.join(" ").trim().to_string();
            if prompt.is_empty() {
                return Err(anyhow!("background run-agent prompt is empty"));
            }
            let cfg = AppConfig::ensure(cwd)?;
            ensure_llm_ready_with_cfg(Some(cwd), &cfg, true)?;
            let exe = std::env::current_exe()?;
            let mut command = Command::new(exe);
            command.arg("ask").arg(&prompt);
            if args.tools {
                command.arg("--tools");
            }
            spawn_background_process(
                cwd,
                "agent",
                format!("ask:{}", &sha256_hex(prompt.as_bytes())[..12]),
                json!({
                    "prompt": prompt,
                    "tools": args.tools,
                    "command": "deepseek ask",
                }),
                command,
            )
        }
        BackgroundCmd::RunShell(args) => {
            let command_line = args.command.join(" ").trim().to_string();
            if command_line.is_empty() {
                return Err(anyhow!("background run-shell command is empty"));
            }
            let (shell_label, command) = build_background_shell_command(&command_line);
            spawn_background_process(
                cwd,
                "shell",
                format!("shell:{}", &sha256_hex(command_line.as_bytes())[..12]),
                json!({
                    "command_line": command_line,
                    "shell": shell_label,
                }),
                command,
            )
        }
    }
}

pub(crate) fn build_background_shell_command(command_line: &str) -> (String, Command) {
    #[cfg(windows)]
    {
        let shell = std::env::var("COMSPEC").unwrap_or_else(|_| "cmd".to_string());
        let mut command = Command::new(&shell);
        command.arg("/C").arg(command_line);
        (shell, command)
    }
    #[cfg(not(windows))]
    {
        let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
        let mut command = Command::new(&shell);
        command.arg("-lc").arg(command_line);
        (shell, command)
    }
}

pub(crate) fn spawn_background_process(
    cwd: &Path,
    kind: &str,
    reference: String,
    metadata: serde_json::Value,
    mut command: Command,
) -> Result<serde_json::Value> {
    let store = Store::new(cwd)?;
    let session = ensure_session_record(cwd, &store)?;
    let job_id = Uuid::now_v7();
    let started_at = Utc::now().to_rfc3339();
    let log_dir = runtime_dir(cwd).join("background").join(kind);
    fs::create_dir_all(&log_dir)?;
    let stdout_log = log_dir.join(format!("{job_id}.stdout.log"));
    let stderr_log = log_dir.join(format!("{job_id}.stderr.log"));
    let stdout_file = File::create(&stdout_log)?;
    let stderr_file = File::create(&stderr_log)?;
    command
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file));
    let child = command.spawn()?;
    let pid = child.id();

    let mut metadata_map = metadata.as_object().cloned().unwrap_or_default();
    metadata_map.insert("pid".to_string(), json!(pid));
    metadata_map.insert(
        "stdout_log".to_string(),
        json!(stdout_log.to_string_lossy().to_string()),
    );
    metadata_map.insert(
        "stderr_log".to_string(),
        json!(stderr_log.to_string_lossy().to_string()),
    );
    metadata_map.insert("started_at".to_string(), json!(started_at.clone()));
    let metadata_value = serde_json::Value::Object(metadata_map);

    let record = BackgroundJobRecord {
        job_id,
        kind: kind.to_string(),
        reference: reference.clone(),
        status: "running".to_string(),
        metadata_json: metadata_value.to_string(),
        started_at: started_at.clone(),
        updated_at: started_at.clone(),
    };
    store.upsert_background_job(&record)?;
    store.append_event(&EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: Utc::now(),
        session_id: session.session_id,
        kind: EventKind::BackgroundJobStarted {
            job_id,
            kind: kind.to_string(),
            reference: reference.clone(),
        },
    })?;
    // Replay projection currently stores '{}' for BackgroundJobStarted metadata.
    // Re-apply the richer metadata payload after emitting the canonical event.
    store.upsert_background_job(&record)?;

    Ok(json!({
        "job_id": job_id,
        "kind": kind,
        "status": "running",
        "reference": reference,
        "pid": pid,
        "stdout_log": stdout_log,
        "stderr_log": stderr_log,
        "metadata": metadata_value,
    }))
}

pub(crate) fn tail_file_lines(path: &Path, max_lines: usize) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let text = String::from_utf8_lossy(&bytes);
    let lines = text.lines().collect::<Vec<_>>();
    if lines.is_empty() {
        return Some(String::new());
    }
    let keep = max_lines.max(1);
    let start = lines.len().saturating_sub(keep);
    Some(lines[start..].join("\n"))
}

pub(crate) fn terminate_background_pid(pid: u32) -> Result<()> {
    #[cfg(windows)]
    {
        let status = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .status()?;
        if !status.success() {
            return Err(anyhow!("taskkill failed for pid {}", pid));
        }
    }
    #[cfg(not(windows))]
    {
        let status = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status()?;
        if !status.success() {
            return Err(anyhow!("kill -TERM failed for pid {}", pid));
        }
    }
    Ok(())
}
