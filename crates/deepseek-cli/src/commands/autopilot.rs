use anyhow::{Result, anyhow};
use chrono::Utc;
use deepseek_agent::AgentEngine;
use deepseek_core::{AppConfig, EventEnvelope, EventKind, runtime_dir};
use deepseek_store::{AutopilotRunRecord, BackgroundJobRecord, Store};
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::context::*;
use crate::output::*;
use crate::{AutopilotArgs, AutopilotCmd};

pub(crate) struct AutopilotStartArgs {
    pub prompt: String,
    pub tools: bool,
    pub max_think: bool,
    pub continue_on_error: bool,
    pub max_iterations: Option<u64>,
    pub duration_seconds: Option<u64>,
    pub hours: Option<f64>,
    pub forever: bool,
    pub sleep_seconds: u64,
    pub retry_delay_seconds: u64,
    pub stop_file: Option<String>,
    pub pause_file: Option<String>,
    pub heartbeat_file: Option<String>,
    pub max_consecutive_failures: u64,
}

pub(crate) fn run_autopilot_cmd(cwd: &Path, args: AutopilotArgs, json_mode: bool) -> Result<()> {
    match args.command {
        Some(AutopilotCmd::Status(status)) => run_autopilot_status(cwd, status, json_mode),
        Some(AutopilotCmd::Pause(pause)) => run_autopilot_pause(cwd, pause, json_mode),
        Some(AutopilotCmd::Stop(stop)) => run_autopilot_stop(cwd, stop, json_mode),
        Some(AutopilotCmd::Resume(resume)) => run_autopilot_resume(cwd, resume, json_mode),
        None => {
            let prompt = args.prompt.ok_or_else(|| {
                anyhow!("missing autopilot prompt; use `deepseek autopilot \"<prompt>\"`")
            })?;
            run_autopilot(
                cwd,
                AutopilotStartArgs {
                    prompt,
                    tools: args.tools,
                    max_think: args.max_think,
                    continue_on_error: args.continue_on_error,
                    max_iterations: args.max_iterations,
                    duration_seconds: args.duration_seconds,
                    hours: args.hours,
                    forever: args.forever,
                    sleep_seconds: args.sleep_seconds,
                    retry_delay_seconds: args.retry_delay_seconds,
                    stop_file: args.stop_file,
                    pause_file: args.pause_file,
                    heartbeat_file: args.heartbeat_file,
                    max_consecutive_failures: args.max_consecutive_failures,
                },
                json_mode,
            )
        }
    }
}

pub(crate) fn run_autopilot(cwd: &Path, args: AutopilotStartArgs, json_mode: bool) -> Result<()> {
    if let Some(max_iterations) = args.max_iterations
        && max_iterations == 0
    {
        return Err(anyhow!("--max-iterations must be greater than 0"));
    }
    if args.forever && (args.duration_seconds.is_some() || args.hours.is_some()) {
        return Err(anyhow!(
            "--forever cannot be combined with --duration-seconds or --hours"
        ));
    }
    if args.max_consecutive_failures == 0 {
        return Err(anyhow!("--max-consecutive-failures must be greater than 0"));
    }
    ensure_llm_ready(cwd, json_mode)?;

    let engine = AgentEngine::new(cwd)?;
    let store = Store::new(cwd)?;
    let session = ensure_session_record(cwd, &store)?;
    let run_id = Uuid::now_v7();
    let started = Instant::now();
    let deadline = autopilot_deadline(&args)?;
    let stop_file = args
        .stop_file
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| runtime_dir(cwd).join("autopilot.stop"));
    let pause_file = args
        .pause_file
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| autopilot_pause_path(&stop_file));
    let heartbeat_file = args
        .heartbeat_file
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| runtime_dir(cwd).join("autopilot.heartbeat.json"));

    let started_at = Utc::now().to_rfc3339();
    store.upsert_autopilot_run(&AutopilotRunRecord {
        run_id,
        session_id: session.session_id,
        prompt: args.prompt.clone(),
        status: "running".to_string(),
        stop_reason: None,
        completed_iterations: 0,
        failed_iterations: 0,
        consecutive_failures: 0,
        last_error: None,
        stop_file: stop_file.to_string_lossy().to_string(),
        heartbeat_file: heartbeat_file.to_string_lossy().to_string(),
        tools: args.tools,
        max_think: args.max_think,
        started_at: started_at.clone(),
        updated_at: started_at.clone(),
    })?;
    store.upsert_background_job(&BackgroundJobRecord {
        job_id: run_id,
        kind: "autopilot".to_string(),
        reference: run_id.to_string(),
        status: "running".to_string(),
        metadata_json: serde_json::json!({
            "prompt": args.prompt.clone(),
            "tools": args.tools,
            "max_think": args.max_think
        })
        .to_string(),
        started_at: started_at.clone(),
        updated_at: started_at.clone(),
    })?;
    store.append_event(&EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: Utc::now(),
        session_id: session.session_id,
        kind: EventKind::AutopilotRunStarted {
            run_id,
            prompt: args.prompt.clone(),
        },
    })?;
    store.append_event(&EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: Utc::now(),
        session_id: session.session_id,
        kind: EventKind::BackgroundJobStarted {
            job_id: run_id,
            kind: "autopilot".to_string(),
            reference: run_id.to_string(),
        },
    })?;

    if !json_mode {
        let runtime = if args.forever {
            "indefinite".to_string()
        } else if let Some(hours) = args.hours {
            format!("{hours:.2} hours")
        } else if let Some(seconds) = args.duration_seconds {
            format!("{seconds} seconds")
        } else {
            "7200 seconds (default)".to_string()
        };
        println!(
            "autopilot started: tools={} max_think={} runtime={} max_iterations={:?} stop_file={} pause_file={} heartbeat_file={}",
            args.tools,
            args.max_think,
            runtime,
            args.max_iterations,
            stop_file.display(),
            pause_file.display(),
            heartbeat_file.display(),
        );
    }

    let mut completed_iterations = 0_u64;
    let mut failed_iterations = 0_u64;
    let mut consecutive_failures = 0_u64;
    let mut last_error: Option<String> = None;
    let mut paused_state = false;

    write_autopilot_heartbeat(
        &heartbeat_file,
        &json!({
            "run_id": run_id,
            "status": "started",
            "at": Utc::now().to_rfc3339(),
            "completed_iterations": completed_iterations,
            "failed_iterations": failed_iterations,
            "consecutive_failures": consecutive_failures,
            "stop_file": stop_file,
            "pause_file": pause_file,
        }),
    )?;

    store.append_event(&EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: Utc::now(),
        session_id: session.session_id,
        kind: EventKind::AutopilotRunHeartbeat {
            run_id,
            completed_iterations,
            failed_iterations,
            consecutive_failures,
            last_error: last_error.clone(),
        },
    })?;

    let stop_reason = loop {
        if let Some(max_iterations) = args.max_iterations
            && completed_iterations + failed_iterations >= max_iterations
        {
            break "max_iterations_reached".to_string();
        }
        if let Some(deadline) = deadline
            && Instant::now() >= deadline
        {
            break "duration_elapsed".to_string();
        }
        if stop_file.exists() {
            break "stop_file_detected".to_string();
        }
        if pause_file.exists() {
            if !paused_state {
                paused_state = true;
                store.upsert_autopilot_run(&AutopilotRunRecord {
                    run_id,
                    session_id: session.session_id,
                    prompt: args.prompt.clone(),
                    status: "paused".to_string(),
                    stop_reason: None,
                    completed_iterations,
                    failed_iterations,
                    consecutive_failures,
                    last_error: last_error.clone(),
                    stop_file: stop_file.to_string_lossy().to_string(),
                    heartbeat_file: heartbeat_file.to_string_lossy().to_string(),
                    tools: args.tools,
                    max_think: args.max_think,
                    started_at: started_at.clone(),
                    updated_at: Utc::now().to_rfc3339(),
                })?;
                store.upsert_background_job(&BackgroundJobRecord {
                    job_id: run_id,
                    kind: "autopilot".to_string(),
                    reference: run_id.to_string(),
                    status: "paused".to_string(),
                    metadata_json: serde_json::json!({
                        "completed_iterations": completed_iterations,
                        "failed_iterations": failed_iterations,
                        "last_error": last_error.clone(),
                        "pause_file": pause_file,
                    })
                    .to_string(),
                    started_at: started_at.clone(),
                    updated_at: Utc::now().to_rfc3339(),
                })?;
                if !json_mode {
                    println!(
                        "autopilot paused; remove {} to continue",
                        pause_file.display()
                    );
                }
            }
            write_autopilot_heartbeat(
                &heartbeat_file,
                &json!({
                    "run_id": run_id,
                    "status": "paused",
                    "at": Utc::now().to_rfc3339(),
                    "completed_iterations": completed_iterations,
                    "failed_iterations": failed_iterations,
                    "consecutive_failures": consecutive_failures,
                    "last_error": last_error,
                    "pause_file": pause_file,
                }),
            )?;
            store.append_event(&EventEnvelope {
                seq_no: store.next_seq_no(session.session_id)?,
                at: Utc::now(),
                session_id: session.session_id,
                kind: EventKind::AutopilotRunHeartbeat {
                    run_id,
                    completed_iterations,
                    failed_iterations,
                    consecutive_failures,
                    last_error: last_error.clone(),
                },
            })?;
            thread::sleep(Duration::from_secs(1));
            continue;
        }
        if paused_state {
            paused_state = false;
            store.upsert_autopilot_run(&AutopilotRunRecord {
                run_id,
                session_id: session.session_id,
                prompt: args.prompt.clone(),
                status: "running".to_string(),
                stop_reason: None,
                completed_iterations,
                failed_iterations,
                consecutive_failures,
                last_error: last_error.clone(),
                stop_file: stop_file.to_string_lossy().to_string(),
                heartbeat_file: heartbeat_file.to_string_lossy().to_string(),
                tools: args.tools,
                max_think: args.max_think,
                started_at: started_at.clone(),
                updated_at: Utc::now().to_rfc3339(),
            })?;
            store.upsert_background_job(&BackgroundJobRecord {
                job_id: run_id,
                kind: "autopilot".to_string(),
                reference: run_id.to_string(),
                status: "running".to_string(),
                metadata_json: serde_json::json!({
                    "completed_iterations": completed_iterations,
                    "failed_iterations": failed_iterations,
                    "last_error": last_error.clone(),
                })
                .to_string(),
                started_at: started_at.clone(),
                updated_at: Utc::now().to_rfc3339(),
            })?;
            if !json_mode {
                println!("autopilot resumed");
            }
        }

        let iteration_no = completed_iterations + failed_iterations + 1;
        if !json_mode {
            println!("autopilot iteration {iteration_no}");
        }
        let iteration_prompt = build_autopilot_iteration_prompt(
            &args.prompt,
            iteration_no,
            consecutive_failures,
            last_error.as_deref(),
        );

        match engine.chat_with_options(
            &iteration_prompt,
            deepseek_agent::ChatOptions {
                tools: args.tools,
                ..Default::default()
            },
        ) {
            Ok(output) => {
                completed_iterations += 1;
                consecutive_failures = 0;
                if !json_mode {
                    println!("{output}");
                }
                if args.sleep_seconds > 0 {
                    thread::sleep(Duration::from_secs(args.sleep_seconds));
                }
            }
            Err(err) => {
                failed_iterations += 1;
                consecutive_failures += 1;
                let err_text = err.to_string();
                last_error = Some(err_text.clone());
                if !json_mode {
                    println!("autopilot iteration failed: {err_text}");
                }
                if !args.continue_on_error {
                    break "stopped_on_error".to_string();
                }
                if consecutive_failures >= args.max_consecutive_failures {
                    break "max_consecutive_failures_reached".to_string();
                }
                if args.retry_delay_seconds > 0 {
                    thread::sleep(Duration::from_secs(args.retry_delay_seconds));
                }
            }
        }
        write_autopilot_heartbeat(
            &heartbeat_file,
            &json!({
                "run_id": run_id,
                "status": "running",
                "at": Utc::now().to_rfc3339(),
                "completed_iterations": completed_iterations,
                "failed_iterations": failed_iterations,
                "consecutive_failures": consecutive_failures,
                "last_error": last_error,
            }),
        )?;
        store.upsert_autopilot_run(&AutopilotRunRecord {
            run_id,
            session_id: session.session_id,
            prompt: args.prompt.clone(),
            status: "running".to_string(),
            stop_reason: None,
            completed_iterations,
            failed_iterations,
            consecutive_failures,
            last_error: last_error.clone(),
            stop_file: stop_file.to_string_lossy().to_string(),
            heartbeat_file: heartbeat_file.to_string_lossy().to_string(),
            tools: args.tools,
            max_think: args.max_think,
            started_at: started_at.clone(),
            updated_at: Utc::now().to_rfc3339(),
        })?;
        store.append_event(&EventEnvelope {
            seq_no: store.next_seq_no(session.session_id)?,
            at: Utc::now(),
            session_id: session.session_id,
            kind: EventKind::AutopilotRunHeartbeat {
                run_id,
                completed_iterations,
                failed_iterations,
                consecutive_failures,
                last_error: last_error.clone(),
            },
        })?;
    };

    let summary = json!({
        "run_id": run_id,
        "stop_reason": stop_reason,
        "elapsed_seconds": started.elapsed().as_secs(),
        "completed_iterations": completed_iterations,
        "failed_iterations": failed_iterations,
        "consecutive_failures": consecutive_failures,
        "tools": args.tools,
        "max_think": args.max_think,
        "continue_on_error": args.continue_on_error,
        "stop_file": stop_file,
        "pause_file": pause_file,
        "heartbeat_file": heartbeat_file,
        "last_error": last_error,
    });
    write_autopilot_heartbeat(
        &heartbeat_file,
        &json!({
            "run_id": run_id,
            "status": "stopped",
            "at": Utc::now().to_rfc3339(),
            "summary": summary,
        }),
    )?;
    store.upsert_autopilot_run(&AutopilotRunRecord {
        run_id,
        session_id: session.session_id,
        prompt: args.prompt,
        status: "stopped".to_string(),
        stop_reason: Some(stop_reason.clone()),
        completed_iterations,
        failed_iterations,
        consecutive_failures,
        last_error: last_error.clone(),
        stop_file: stop_file.to_string_lossy().to_string(),
        heartbeat_file: heartbeat_file.to_string_lossy().to_string(),
        tools: args.tools,
        max_think: args.max_think,
        started_at: started_at.clone(),
        updated_at: Utc::now().to_rfc3339(),
    })?;
    store.upsert_background_job(&BackgroundJobRecord {
        job_id: run_id,
        kind: "autopilot".to_string(),
        reference: run_id.to_string(),
        status: "stopped".to_string(),
        metadata_json: serde_json::json!({
            "stop_reason": stop_reason,
            "completed_iterations": completed_iterations,
            "failed_iterations": failed_iterations,
            "last_error": last_error.clone(),
        })
        .to_string(),
        started_at: started_at.clone(),
        updated_at: Utc::now().to_rfc3339(),
    })?;
    store.append_event(&EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: Utc::now(),
        session_id: session.session_id,
        kind: EventKind::AutopilotRunStopped {
            run_id,
            stop_reason: stop_reason.clone(),
            completed_iterations,
            failed_iterations,
        },
    })?;
    store.append_event(&EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: Utc::now(),
        session_id: session.session_id,
        kind: EventKind::BackgroundJobStopped {
            job_id: run_id,
            reason: stop_reason.clone(),
        },
    })?;

    if json_mode {
        print_json(&summary)?;
    } else {
        println!(
            "autopilot stopped: {} (completed={} failed={} elapsed={}s)",
            summary["stop_reason"].as_str().unwrap_or_default(),
            completed_iterations,
            failed_iterations,
            summary["elapsed_seconds"].as_u64().unwrap_or(0)
        );
    }

    let stop_reason = summary["stop_reason"].as_str().unwrap_or_default();
    if stop_reason == "stopped_on_error" || stop_reason == "max_consecutive_failures_reached" {
        return Err(anyhow!(
            "autopilot stopped on error: {}",
            last_error.unwrap_or_else(|| "unknown error".to_string())
        ));
    }

    Ok(())
}

pub(crate) fn run_autopilot_status(
    cwd: &Path,
    args: crate::AutopilotStatusArgs,
    json_mode: bool,
) -> Result<()> {
    let store = Store::new(cwd)?;
    let Some(run) = find_autopilot_run(&store, args.run_id.as_deref())? else {
        let payload = json!({
            "status": "none",
            "run_id": null,
            "session_id": null,
            "completed_iterations": 0,
            "failed_iterations": 0,
            "consecutive_failures": 0,
        });
        if json_mode {
            print_json(&payload)?;
        } else {
            println!("no autopilot runs found");
        }
        return Ok(());
    };
    if args.follow {
        let max_samples = args.samples.unwrap_or(10).max(1);
        let interval = Duration::from_secs(args.interval_seconds.max(1));
        let mut samples = Vec::new();
        let mut current = run;
        for idx in 0..max_samples {
            let mut snapshot = autopilot_status_payload(&current);
            if let Some(obj) = snapshot.as_object_mut() {
                obj.insert("sample_index".to_string(), json!(idx + 1));
                obj.insert("sampled_at".to_string(), json!(Utc::now().to_rfc3339()));
            }
            samples.push(snapshot.clone());
            if !current.status.eq_ignore_ascii_case("running") {
                break;
            }
            if idx + 1 >= max_samples {
                break;
            }
            thread::sleep(interval);
            current = store
                .load_autopilot_run(current.run_id)?
                .unwrap_or_else(|| current.clone());
        }
        let payload = json!({
            "run_id": current.run_id,
            "follow": true,
            "interval_seconds": args.interval_seconds.max(1),
            "samples_collected": samples.len(),
            "samples": samples,
        });
        if json_mode {
            print_json(&payload)?;
        } else {
            for sample in payload["samples"].as_array().into_iter().flatten() {
                println!(
                    "sample#{} at={} status={} completed={} failed={} paused={}",
                    sample["sample_index"].as_u64().unwrap_or(0),
                    sample["sampled_at"].as_str().unwrap_or_default(),
                    sample["status"].as_str().unwrap_or_default(),
                    sample["completed_iterations"].as_u64().unwrap_or(0),
                    sample["failed_iterations"].as_u64().unwrap_or(0),
                    sample["paused"].as_bool().unwrap_or(false),
                );
            }
        }
    } else {
        let payload = autopilot_status_payload(&run);
        if json_mode {
            print_json(&payload)?;
        } else {
            println!(
                "run={} status={} completed={} failed={} consecutive_failures={}",
                run.run_id,
                run.status,
                run.completed_iterations,
                run.failed_iterations,
                run.consecutive_failures
            );
            println!("paused={}", payload["paused"].as_bool().unwrap_or(false));
            if let Some(reason) = run.stop_reason {
                println!("stop_reason={reason}");
            }
            if let Some(err) = run.last_error {
                println!("last_error={err}");
            }
            println!("stop_file={}", run.stop_file);
            println!(
                "pause_file={}",
                payload["pause_file"].as_str().unwrap_or_default()
            );
            println!("heartbeat_file={}", run.heartbeat_file);
        }
    }
    Ok(())
}

pub(crate) fn autopilot_status_payload(run: &AutopilotRunRecord) -> serde_json::Value {
    let heartbeat = if run.heartbeat_file.is_empty() {
        None
    } else {
        fs::read_to_string(&run.heartbeat_file)
            .ok()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
    };
    let pause_file = autopilot_pause_path(&PathBuf::from(&run.stop_file));
    let paused = pause_file.exists();
    json!({
        "run_id": run.run_id,
        "session_id": run.session_id,
        "status": run.status,
        "paused": paused,
        "stop_reason": run.stop_reason,
        "completed_iterations": run.completed_iterations,
        "failed_iterations": run.failed_iterations,
        "consecutive_failures": run.consecutive_failures,
        "last_error": run.last_error,
        "stop_file": run.stop_file,
        "pause_file": pause_file,
        "heartbeat_file": run.heartbeat_file,
        "tools": run.tools,
        "max_think": run.max_think,
        "heartbeat": heartbeat,
    })
}

pub(crate) fn run_autopilot_pause(
    cwd: &Path,
    args: crate::AutopilotPauseArgs,
    json_mode: bool,
) -> Result<()> {
    let store = Store::new(cwd)?;
    let run = find_autopilot_run(&store, args.run_id.as_deref())?
        .ok_or_else(|| anyhow!("no autopilot runs found"))?;
    let pause_path = autopilot_pause_path(&PathBuf::from(&run.stop_file));
    if let Some(parent) = pause_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &pause_path,
        format!("pause requested at {}\n", Utc::now().to_rfc3339()),
    )?;
    if json_mode {
        print_json(&json!({
            "run_id": run.run_id,
            "pause_requested": true,
            "pause_file": pause_path,
        }))?;
    } else {
        println!(
            "pause requested for run {} via {}",
            run.run_id,
            pause_path.display()
        );
    }
    Ok(())
}

pub(crate) fn run_autopilot_stop(
    cwd: &Path,
    args: crate::AutopilotStopArgs,
    json_mode: bool,
) -> Result<()> {
    let store = Store::new(cwd)?;
    let run = find_autopilot_run(&store, args.run_id.as_deref())?
        .ok_or_else(|| anyhow!("no autopilot runs found"))?;
    let stop_path = if run.stop_file.trim().is_empty() {
        runtime_dir(cwd).join("autopilot.stop")
    } else {
        PathBuf::from(run.stop_file.clone())
    };
    if let Some(parent) = stop_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &stop_path,
        format!("stop requested at {}\n", Utc::now().to_rfc3339()),
    )?;
    let pause_path = autopilot_pause_path(&stop_path);
    if pause_path.exists() {
        let _ = fs::remove_file(&pause_path);
    }
    if json_mode {
        print_json(&json!({
            "run_id": run.run_id,
            "stop_requested": true,
            "stop_file": stop_path,
            "pause_file": pause_path,
        }))?;
    } else {
        println!(
            "stop requested for run {} via {}",
            run.run_id,
            stop_path.display()
        );
    }
    Ok(())
}

pub(crate) fn run_autopilot_resume(
    cwd: &Path,
    args: crate::AutopilotResumeArgs,
    json_mode: bool,
) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;
    let store = Store::new(cwd)?;
    let run = find_autopilot_run(&store, args.run_id.as_deref())?
        .ok_or_else(|| anyhow!("no autopilot runs found"))?;
    let pause_path = autopilot_pause_path(&PathBuf::from(&run.stop_file));
    if pause_path.exists() {
        fs::remove_file(&pause_path)?;
        if run.status == "running" || run.status == "paused" {
            if json_mode {
                print_json(&json!({
                    "run_id": run.run_id,
                    "resumed_live": true,
                    "pause_file": pause_path,
                }))?;
            } else {
                println!("live resume requested for run {}", run.run_id);
            }
            return Ok(());
        }
    }
    if run.status == "running" {
        return Err(anyhow!(
            "autopilot run is already marked as running (no pause file present)"
        ));
    }

    run_autopilot(
        cwd,
        AutopilotStartArgs {
            prompt: run.prompt.clone(),
            tools: run.tools,
            max_think: run.max_think,
            continue_on_error: true,
            max_iterations: None,
            duration_seconds: None,
            hours: None,
            forever: false,
            sleep_seconds: 0,
            retry_delay_seconds: 2,
            stop_file: if run.stop_file.trim().is_empty() {
                None
            } else {
                Some(run.stop_file.clone())
            },
            pause_file: None,
            heartbeat_file: if run.heartbeat_file.trim().is_empty() {
                None
            } else {
                Some(run.heartbeat_file.clone())
            },
            max_consecutive_failures: cfg.autopilot.default_max_consecutive_failures.max(1),
        },
        json_mode,
    )
}

pub(crate) fn find_autopilot_run(
    store: &Store,
    run_id: Option<&str>,
) -> Result<Option<AutopilotRunRecord>> {
    if let Some(run_id) = run_id {
        let uid = Uuid::parse_str(run_id)?;
        return store.load_autopilot_run(uid);
    }
    store.load_latest_autopilot_run()
}

pub(crate) fn autopilot_pause_path(stop_file: &Path) -> PathBuf {
    if let Some(ext) = stop_file.extension()
        && ext == "stop"
    {
        return stop_file.with_extension("pause");
    }
    let file_name = stop_file
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| format!("{name}.pause"))
        .unwrap_or_else(|| "autopilot.pause".to_string());
    stop_file.with_file_name(file_name)
}

pub(crate) fn autopilot_deadline(args: &AutopilotStartArgs) -> Result<Option<Instant>> {
    if args.forever {
        return Ok(None);
    }

    let seconds = if let Some(seconds) = args.duration_seconds {
        seconds
    } else if let Some(hours) = args.hours {
        if !(hours.is_finite() && hours > 0.0) {
            return Err(anyhow!("--hours must be a positive finite value"));
        }
        (hours * 3600.0).round() as u64
    } else {
        2 * 3600
    };

    Ok(Some(Instant::now() + Duration::from_secs(seconds.max(1))))
}

pub(crate) fn build_autopilot_iteration_prompt(
    prompt: &str,
    iteration: u64,
    consecutive_failures: u64,
    last_error: Option<&str>,
) -> String {
    if consecutive_failures == 0 {
        return prompt.to_string();
    }
    let context = last_error
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("unknown error");
    format!(
        "{prompt}\n\n[autopilot_recovery]\niteration={iteration}\nconsecutive_failures={consecutive_failures}\nlast_error={context}\npriority=recover_and_continue"
    )
}

pub(crate) fn write_autopilot_heartbeat(path: &Path, payload: &serde_json::Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(payload)?)?;
    Ok(())
}
