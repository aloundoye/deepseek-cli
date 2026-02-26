use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use deepseek_core::EventKind;
use deepseek_store::{RemoteEnvProfileRecord, Store};
use reqwest::Url;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::{Command, Output};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::RemoteEnvCmd;
use crate::context::*;
use crate::output::*;
use crate::{RemoteEnvAddArgs, RemoteEnvExecArgs, RemoteEnvLogsArgs, RemoteEnvRunAgentArgs};

use super::background::{spawn_background_process, tail_file_lines};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RemoteSshMetadata {
    #[serde(default)]
    ssh_user: Option<String>,
    #[serde(default)]
    ssh_port: Option<u16>,
    #[serde(default)]
    ssh_key_path: Option<String>,
    #[serde(default)]
    workspace_root: Option<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedSshEndpoint {
    host: String,
    user: Option<String>,
    port: Option<u16>,
    workspace_root: Option<String>,
}

struct HealthCheckResult {
    reachable: bool,
    latency_ms: Option<u128>,
    status_code: Option<u16>,
    checked_target: String,
    error: Option<String>,
}

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
            Ok(RemoteEnvCmd::Add(RemoteEnvAddArgs {
                name: args[1].clone(),
                endpoint: args[2].clone(),
                auth_mode: args.get(3).cloned().unwrap_or_else(|| "token".to_string()),
                ssh_user: None,
                ssh_port: None,
                ssh_key_path: None,
                workspace_root: None,
                env: Vec::new(),
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
        "exec" => {
            if args.len() < 3 {
                return Err(anyhow!(
                    "usage: /remote-env exec <profile_id> <command...> [--background] [--timeout-seconds N]"
                ));
            }
            let profile_id = args[1].clone();
            let mut background = false;
            let mut timeout_seconds = None;
            let mut cmd_parts = Vec::new();
            let mut idx = 2usize;
            while idx < args.len() {
                let token = &args[idx];
                if token == "--background" {
                    background = true;
                    idx += 1;
                    continue;
                }
                if token == "--timeout-seconds" && idx + 1 < args.len() {
                    timeout_seconds = args[idx + 1].parse::<u64>().ok();
                    idx += 2;
                    continue;
                }
                cmd_parts.push(token.clone());
                idx += 1;
            }
            let cmd = cmd_parts.join(" ").trim().to_string();
            if cmd.is_empty() {
                return Err(anyhow!("usage: /remote-env exec <profile_id> <command...>"));
            }
            Ok(RemoteEnvCmd::Exec(RemoteEnvExecArgs {
                profile_id,
                cmd,
                timeout_seconds,
                background,
            }))
        }
        "run-agent" | "agent" => {
            if args.len() < 3 {
                return Err(anyhow!(
                    "usage: /remote-env run-agent <profile_id> <prompt...> [--tools=true|false] [--max-think=true|false] [--background] [--timeout-seconds N]"
                ));
            }
            let profile_id = args[1].clone();
            let mut tools = true;
            let mut max_think = true;
            let mut background = false;
            let mut timeout_seconds = None;
            let mut prompt_parts = Vec::new();
            let mut idx = 2usize;
            while idx < args.len() {
                let token = &args[idx];
                if token == "--background" {
                    background = true;
                    idx += 1;
                    continue;
                }
                if token == "--tools=false" {
                    tools = false;
                    idx += 1;
                    continue;
                }
                if token == "--tools=true" {
                    tools = true;
                    idx += 1;
                    continue;
                }
                if token == "--max-think=false" {
                    max_think = false;
                    idx += 1;
                    continue;
                }
                if token == "--max-think=true" {
                    max_think = true;
                    idx += 1;
                    continue;
                }
                if token == "--timeout-seconds" && idx + 1 < args.len() {
                    timeout_seconds = args[idx + 1].parse::<u64>().ok();
                    idx += 2;
                    continue;
                }
                prompt_parts.push(token.clone());
                idx += 1;
            }
            let prompt = prompt_parts.join(" ").trim().to_string();
            if prompt.is_empty() {
                return Err(anyhow!(
                    "usage: /remote-env run-agent <profile_id> <prompt...>"
                ));
            }
            Ok(RemoteEnvCmd::RunAgent(RemoteEnvRunAgentArgs {
                profile_id,
                prompt,
                tools,
                max_think,
                timeout_seconds,
                background,
            }))
        }
        "logs" => {
            if args.len() < 2 {
                return Err(anyhow!("usage: /remote-env logs <job_id> [tail_lines]"));
            }
            let tail_lines = args
                .get(2)
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(40);
            Ok(RemoteEnvCmd::Logs(RemoteEnvLogsArgs {
                job_id: args[1].clone(),
                tail_lines,
            }))
        }
        _ => Err(anyhow!(
            "use /remote-env list|add|remove|check|exec|run-agent|logs"
        )),
    }
}

pub(crate) fn remote_env_now(cwd: &Path, cmd: RemoteEnvCmd) -> Result<serde_json::Value> {
    let store = Store::new(cwd)?;
    match cmd {
        RemoteEnvCmd::List => {
            let profiles = store.list_remote_env_profiles()?;
            let rows = profiles
                .into_iter()
                .map(|profile| {
                    let metadata = load_remote_metadata(&profile);
                    json!({
                        "profile_id": profile.profile_id,
                        "name": profile.name,
                        "endpoint": profile.endpoint,
                        "auth_mode": profile.auth_mode,
                        "updated_at": profile.updated_at,
                        "metadata": metadata,
                    })
                })
                .collect::<Vec<_>>();
            Ok(json!({"profiles": rows}))
        }
        RemoteEnvCmd::Add(args) => add_remote_profile(cwd, &store, args),
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
            let metadata = load_remote_metadata(&profile);
            let health = if endpoint_is_http(&profile.endpoint) {
                check_remote_http_endpoint(&profile.endpoint)
            } else {
                check_remote_ssh_endpoint(&profile, &metadata)
            };
            Ok(json!({
                "profile_id": profile.profile_id,
                "name": profile.name,
                "endpoint": profile.endpoint,
                "auth_mode": profile.auth_mode,
                "metadata": metadata,
                "reachable": health.reachable,
                "latency_ms": health.latency_ms,
                "status_code": health.status_code,
                "checked_target": health.checked_target,
                "error": health.error,
            }))
        }
        RemoteEnvCmd::Exec(args) => remote_exec(cwd, &store, args),
        RemoteEnvCmd::RunAgent(args) => remote_run_agent(cwd, &store, args),
        RemoteEnvCmd::Logs(args) => remote_logs(cwd, &store, args),
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

fn add_remote_profile(
    cwd: &Path,
    store: &Store,
    args: RemoteEnvAddArgs,
) -> Result<serde_json::Value> {
    let profile_id = Uuid::now_v7();
    let mut parsed_endpoint = None;
    if !endpoint_is_http(&args.endpoint) {
        parsed_endpoint = Some(parse_ssh_endpoint(&args.endpoint)?);
    }

    let metadata = RemoteSshMetadata {
        ssh_user: args.ssh_user.or_else(|| {
            parsed_endpoint
                .as_ref()
                .and_then(|parsed| parsed.user.clone())
        }),
        ssh_port: args
            .ssh_port
            .or_else(|| parsed_endpoint.as_ref().and_then(|parsed| parsed.port)),
        ssh_key_path: args.ssh_key_path,
        workspace_root: args.workspace_root.or_else(|| {
            parsed_endpoint
                .as_ref()
                .and_then(|parsed| parsed.workspace_root.clone())
        }),
        env: parse_env_pairs(&args.env)?,
    };

    store.upsert_remote_env_profile(&RemoteEnvProfileRecord {
        profile_id,
        name: args.name.clone(),
        endpoint: args.endpoint.clone(),
        auth_mode: args.auth_mode.clone(),
        metadata_json: serde_json::to_string(&metadata)?,
        updated_at: Utc::now().to_rfc3339(),
    })?;
    append_control_event(
        cwd,
        EventKind::RemoteEnvConfigured {
            profile_id,
            name: args.name,
            endpoint: args.endpoint,
        },
    )?;
    Ok(json!({
        "profile_id": profile_id,
        "configured": true,
        "metadata": metadata,
    }))
}

fn remote_exec(cwd: &Path, store: &Store, args: RemoteEnvExecArgs) -> Result<serde_json::Value> {
    let profile_id = Uuid::parse_str(&args.profile_id)?;
    let profile = store
        .load_remote_env_profile(profile_id)?
        .ok_or_else(|| anyhow!("remote profile not found: {}", profile_id))?;
    let metadata = load_remote_metadata(&profile);

    let remote_command = wrap_remote_command(&metadata, &args.cmd);
    let timeout_seconds = args.timeout_seconds.unwrap_or(120).clamp(1, 86_400);
    execute_remote_command(
        cwd,
        &profile,
        &metadata,
        "exec",
        &remote_command,
        timeout_seconds,
        args.background,
    )
}

fn remote_run_agent(
    cwd: &Path,
    store: &Store,
    args: RemoteEnvRunAgentArgs,
) -> Result<serde_json::Value> {
    let profile_id = Uuid::parse_str(&args.profile_id)?;
    let profile = store
        .load_remote_env_profile(profile_id)?
        .ok_or_else(|| anyhow!("remote profile not found: {}", profile_id))?;
    let metadata = load_remote_metadata(&profile);

    let mut cmd = format!(
        "deepseek ask {} --tools={}",
        shell_quote(&args.prompt),
        if args.tools { "true" } else { "false" }
    );
    if args.max_think {
        cmd.push_str(" --model deepseek-reasoner");
    }
    let remote_command = wrap_remote_command(&metadata, &cmd);
    let timeout_seconds = args.timeout_seconds.unwrap_or(900).clamp(1, 86_400);
    execute_remote_command(
        cwd,
        &profile,
        &metadata,
        "run-agent",
        &remote_command,
        timeout_seconds,
        args.background,
    )
}

fn remote_logs(cwd: &Path, store: &Store, args: RemoteEnvLogsArgs) -> Result<serde_json::Value> {
    let job_id = Uuid::parse_str(&args.job_id)?;
    let job = store
        .load_background_job(job_id)?
        .ok_or_else(|| anyhow!("background job not found: {}", args.job_id))?;

    let metadata =
        serde_json::from_str::<serde_json::Value>(&job.metadata_json).unwrap_or_default();
    let stdout_tail = metadata
        .get("stdout_log")
        .and_then(|v| v.as_str())
        .and_then(|path| tail_file_lines(Path::new(path), args.tail_lines.max(1)));
    let stderr_tail = metadata
        .get("stderr_log")
        .and_then(|v| v.as_str())
        .and_then(|path| tail_file_lines(Path::new(path), args.tail_lines.max(1)));

    append_control_event(
        cwd,
        EventKind::TelemetryEvent {
            name: "kpi.remote.logs_tail".to_string(),
            properties: json!({
                "job_id": job_id,
                "tail_lines": args.tail_lines,
                "kind": job.kind,
            }),
        },
    )?;

    Ok(json!({
        "job_id": job_id,
        "kind": job.kind,
        "status": job.status,
        "reference": job.reference,
        "metadata": metadata,
        "log_tail": {
            "stdout": stdout_tail,
            "stderr": stderr_tail,
        }
    }))
}

fn execute_remote_command(
    cwd: &Path,
    profile: &RemoteEnvProfileRecord,
    metadata: &RemoteSshMetadata,
    mode: &str,
    remote_command: &str,
    timeout_seconds: u64,
    background: bool,
) -> Result<serde_json::Value> {
    let execution_id = Uuid::now_v7();
    let start_instant = Instant::now();

    let command = build_ssh_command(profile, metadata, remote_command, timeout_seconds)?;
    let command_hash = short_sha256(remote_command);

    if background {
        let payload = spawn_background_process(
            cwd,
            "remote",
            format!("remote-{mode}:{command_hash}"),
            json!({
                "mode": mode,
                "profile_id": profile.profile_id,
                "endpoint": profile.endpoint,
                "remote_command": remote_command,
                "timeout_seconds": timeout_seconds,
            }),
            command,
        )?;
        let reference = payload
            .get("job_id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        append_control_event(
            cwd,
            EventKind::RemoteEnvExecutionStarted {
                execution_id,
                profile_id: profile.profile_id,
                mode: mode.to_string(),
                background: true,
                reference,
            },
        )?;
        append_control_event(
            cwd,
            EventKind::TelemetryEvent {
                name: "kpi.remote.exec_started".to_string(),
                properties: json!({
                    "mode": mode,
                    "background": true,
                    "profile_id": profile.profile_id,
                }),
            },
        )?;
        return Ok(json!({
            "schema": "deepseek.remote.exec.v1",
            "execution_id": execution_id,
            "background": true,
            "status": "running",
            "mode": mode,
            "profile_id": profile.profile_id,
            "payload": payload,
        }));
    }

    append_control_event(
        cwd,
        EventKind::RemoteEnvExecutionStarted {
            execution_id,
            profile_id: profile.profile_id,
            mode: mode.to_string(),
            background: false,
            reference: command_hash,
        },
    )?;

    let (output, duration_ms) = run_command_with_timeout(command, timeout_seconds)?;
    let success = output.status.success();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    append_control_event(
        cwd,
        EventKind::RemoteEnvExecutionCompleted {
            execution_id,
            profile_id: profile.profile_id,
            mode: mode.to_string(),
            success,
            duration_ms,
            background: false,
            reference: short_sha256(remote_command),
        },
    )?;
    append_control_event(
        cwd,
        EventKind::TelemetryEvent {
            name: "kpi.remote.exec_finished".to_string(),
            properties: json!({
                "mode": mode,
                "success": success,
                "duration_ms": duration_ms,
                "profile_id": profile.profile_id,
            }),
        },
    )?;

    Ok(json!({
        "schema": "deepseek.remote.exec.v1",
        "execution_id": execution_id,
        "background": false,
        "mode": mode,
        "profile_id": profile.profile_id,
        "endpoint": profile.endpoint,
        "success": success,
        "duration_ms": duration_ms,
        "exit_code": output.status.code(),
        "stdout": stdout,
        "stderr": stderr,
        "timeout_seconds": timeout_seconds,
        "remote_command": remote_command,
        "wall_time_ms": start_instant.elapsed().as_millis() as u64,
    }))
}

fn build_ssh_command(
    profile: &RemoteEnvProfileRecord,
    metadata: &RemoteSshMetadata,
    remote_command: &str,
    connect_timeout_seconds: u64,
) -> Result<Command> {
    let parsed = parse_ssh_endpoint(&profile.endpoint)?;
    let user = metadata.ssh_user.clone().or(parsed.user);
    let port = metadata.ssh_port.or(parsed.port);
    let target = if let Some(user) = user {
        format!("{user}@{}", parsed.host)
    } else {
        parsed.host
    };

    let mut command = Command::new("ssh");
    command.arg("-o").arg("BatchMode=yes");
    command.arg("-o").arg("StrictHostKeyChecking=accept-new");
    command
        .arg("-o")
        .arg(format!("ConnectTimeout={connect_timeout_seconds}"));
    if let Some(port) = port {
        command.arg("-p").arg(port.to_string());
    }
    if let Some(key_path) = metadata.ssh_key_path.as_deref()
        && !key_path.trim().is_empty()
    {
        command.arg("-i").arg(key_path);
    }
    command.arg(target);
    command.arg(remote_command);
    Ok(command)
}

fn load_remote_metadata(profile: &RemoteEnvProfileRecord) -> RemoteSshMetadata {
    serde_json::from_str(&profile.metadata_json).unwrap_or_default()
}

fn wrap_remote_command(metadata: &RemoteSshMetadata, command: &str) -> String {
    let mut rendered = String::new();
    for (key, value) in &metadata.env {
        rendered.push_str(key);
        rendered.push('=');
        rendered.push_str(&shell_quote(value));
        rendered.push(' ');
    }
    if let Some(workspace_root) = metadata.workspace_root.as_deref()
        && !workspace_root.trim().is_empty()
    {
        rendered.push_str("cd ");
        rendered.push_str(&shell_quote(workspace_root));
        rendered.push_str(" && ");
    }
    rendered.push_str(command);
    rendered
}

fn parse_env_pairs(raw: &[String]) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for pair in raw {
        let (key, value) = pair
            .split_once('=')
            .ok_or_else(|| anyhow!("invalid env entry '{}', expected KEY=VALUE", pair))?;
        let key = key.trim();
        if key.is_empty() {
            return Err(anyhow!("invalid env entry '{}': key is empty", pair));
        }
        if !key
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
            || key.chars().next().is_some_and(|ch| ch.is_ascii_digit())
        {
            return Err(anyhow!(
                "invalid env key '{}': use [A-Za-z_][A-Za-z0-9_]*",
                key
            ));
        }
        out.insert(key.to_string(), value.to_string());
    }
    Ok(out)
}

fn parse_ssh_endpoint(endpoint: &str) -> Result<ParsedSshEndpoint> {
    let endpoint = endpoint.trim();
    if endpoint.is_empty() {
        return Err(anyhow!("remote endpoint is empty"));
    }

    if endpoint.starts_with("ssh://") {
        let parsed = Url::parse(endpoint).context("invalid ssh endpoint URL")?;
        let host = parsed
            .host_str()
            .ok_or_else(|| anyhow!("ssh endpoint missing host"))?
            .to_string();
        let user = if parsed.username().is_empty() {
            None
        } else {
            Some(parsed.username().to_string())
        };
        let port = parsed.port();
        let workspace_root = {
            let path = parsed.path().trim();
            if path.is_empty() || path == "/" {
                None
            } else {
                Some(path.to_string())
            }
        };
        return Ok(ParsedSshEndpoint {
            host,
            user,
            port,
            workspace_root,
        });
    }

    let (host_target, workspace_root) = if let Some((left, right)) = endpoint.split_once('/') {
        let workspace_root = right.trim_matches('/');
        (
            left.to_string(),
            if workspace_root.is_empty() {
                None
            } else {
                Some(format!("/{workspace_root}"))
            },
        )
    } else {
        (endpoint.to_string(), None)
    };

    let (user, host_port) = if let Some((left, right)) = host_target.rsplit_once('@') {
        if left.trim().is_empty() {
            return Err(anyhow!("ssh endpoint user segment is empty"));
        }
        (Some(left.trim().to_string()), right.trim().to_string())
    } else {
        (None, host_target.trim().to_string())
    };

    if host_port.is_empty() {
        return Err(anyhow!("ssh endpoint missing host"));
    }

    let (host, port) = if let Some((host, port_raw)) = host_port.rsplit_once(':') {
        if port_raw.chars().all(|ch| ch.is_ascii_digit()) {
            let port = port_raw
                .parse::<u16>()
                .with_context(|| format!("invalid ssh port '{}': out of range", port_raw))?;
            (host.to_string(), Some(port))
        } else {
            (host_port, None)
        }
    } else {
        (host_port, None)
    };

    if host.trim().is_empty() {
        return Err(anyhow!("ssh endpoint missing host"));
    }

    Ok(ParsedSshEndpoint {
        host: host.trim().to_string(),
        user,
        port,
        workspace_root,
    })
}

fn endpoint_is_http(endpoint: &str) -> bool {
    endpoint.starts_with("http://") || endpoint.starts_with("https://")
}

fn check_remote_http_endpoint(endpoint: &str) -> HealthCheckResult {
    let parsed = match Url::parse(endpoint) {
        Ok(url) => url,
        Err(err) => {
            return HealthCheckResult {
                reachable: false,
                latency_ms: None,
                status_code: None,
                checked_target: endpoint.to_string(),
                error: Some(format!("invalid endpoint URL: {err}")),
            };
        }
    };

    let client = match Client::builder()
        .timeout(Duration::from_secs(4))
        .redirect(reqwest::redirect::Policy::limited(2))
        .build()
    {
        Ok(client) => client,
        Err(err) => {
            return HealthCheckResult {
                reachable: false,
                latency_ms: None,
                status_code: None,
                checked_target: endpoint.to_string(),
                error: Some(format!("failed to initialize HTTP client: {err}")),
            };
        }
    };

    let mut candidates = vec![parsed.to_string()];
    if parsed.path() == "/" {
        for suffix in ["/health", "/status", "/ping"] {
            if let Ok(candidate) = parsed.join(suffix) {
                candidates.push(candidate.to_string());
            }
        }
    }

    for candidate in candidates {
        let started = Instant::now();
        let response = client
            .head(&candidate)
            .send()
            .or_else(|_| client.get(&candidate).send());
        match response {
            Ok(resp) => {
                let status = resp.status();
                return HealthCheckResult {
                    reachable: status.is_success() || status.is_redirection(),
                    latency_ms: Some(started.elapsed().as_millis()),
                    status_code: Some(status.as_u16()),
                    checked_target: candidate,
                    error: None,
                };
            }
            Err(err) => {
                if candidate == endpoint {
                    continue;
                }
                return HealthCheckResult {
                    reachable: false,
                    latency_ms: Some(started.elapsed().as_millis()),
                    status_code: None,
                    checked_target: candidate,
                    error: Some(err.to_string()),
                };
            }
        }
    }

    HealthCheckResult {
        reachable: false,
        latency_ms: None,
        status_code: None,
        checked_target: endpoint.to_string(),
        error: Some("all endpoint probes failed".to_string()),
    }
}

fn check_remote_ssh_endpoint(
    profile: &RemoteEnvProfileRecord,
    metadata: &RemoteSshMetadata,
) -> HealthCheckResult {
    let started = Instant::now();
    let command = match build_ssh_command(profile, metadata, "echo deepseek-remote-ok", 4) {
        Ok(command) => command,
        Err(err) => {
            return HealthCheckResult {
                reachable: false,
                latency_ms: None,
                status_code: None,
                checked_target: profile.endpoint.clone(),
                error: Some(err.to_string()),
            };
        }
    };

    match run_command_with_timeout(command, 8) {
        Ok((output, _duration_ms)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let ok = output.status.success() && stdout.contains("deepseek-remote-ok");
            HealthCheckResult {
                reachable: ok,
                latency_ms: Some(started.elapsed().as_millis()),
                status_code: output.status.code().map(|c| c as u16),
                checked_target: profile.endpoint.clone(),
                error: if ok {
                    None
                } else if stderr.trim().is_empty() {
                    Some("ssh check command failed".to_string())
                } else {
                    Some(stderr.trim().to_string())
                },
            }
        }
        Err(err) => HealthCheckResult {
            reachable: false,
            latency_ms: Some(started.elapsed().as_millis()),
            status_code: None,
            checked_target: profile.endpoint.clone(),
            error: Some(err.to_string()),
        },
    }
}

fn run_command_with_timeout(mut command: Command, timeout_seconds: u64) -> Result<(Output, u64)> {
    let timeout = Duration::from_secs(timeout_seconds.max(1));
    let started = Instant::now();

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let result = command.output();
        let _ = tx.send(result);
    });

    match rx.recv_timeout(timeout) {
        Ok(result) => {
            let output = result?;
            Ok((output, started.elapsed().as_millis() as u64))
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            Err(anyhow!("remote command timed out after {timeout_seconds}s"))
        }
        Err(err) => Err(anyhow!("remote command execution failed: {err}")),
    }
}

fn short_sha256(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    format!("{:x}", digest)[..12].to_string()
}

fn shell_quote(input: &str) -> String {
    if input.is_empty() {
        return "''".to_string();
    }
    let escaped = input.replace('\'', "'\"'\"'");
    format!("'{escaped}'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ssh_endpoint_from_url() {
        let parsed = parse_ssh_endpoint("ssh://dev@example.com:2222/work/repo").expect("parse");
        assert_eq!(parsed.host, "example.com");
        assert_eq!(parsed.user.as_deref(), Some("dev"));
        assert_eq!(parsed.port, Some(2222));
        assert_eq!(parsed.workspace_root.as_deref(), Some("/work/repo"));
    }

    #[test]
    fn parse_ssh_endpoint_from_host_only() {
        let parsed = parse_ssh_endpoint("example.com").expect("parse");
        assert_eq!(parsed.host, "example.com");
        assert_eq!(parsed.user, None);
        assert_eq!(parsed.port, None);
        assert_eq!(parsed.workspace_root, None);
    }

    #[test]
    fn parse_ssh_endpoint_from_user_host_port() {
        let parsed = parse_ssh_endpoint("ops@example.com:2200").expect("parse");
        assert_eq!(parsed.host, "example.com");
        assert_eq!(parsed.user.as_deref(), Some("ops"));
        assert_eq!(parsed.port, Some(2200));
    }

    #[test]
    fn parse_env_pairs_rejects_bad_key() {
        let err = parse_env_pairs(&["1BAD=value".to_string()]).expect_err("invalid key");
        assert!(err.to_string().contains("invalid env key"));
    }

    #[test]
    fn shell_quote_handles_single_quotes() {
        assert_eq!(shell_quote("a'b"), "'a'\"'\"'b'");
    }
}
