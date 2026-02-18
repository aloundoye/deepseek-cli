use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::TempDir;

#[test]
fn plan_command_emits_json_shape() {
    let workspace = TempDir::new().expect("workspace");
    let out = run_json(
        workspace.path(),
        &["--json", "plan", "Refactor planner state"],
    );
    assert!(out.get("plan_id").is_some());
    assert!(out["steps"].as_array().is_some_and(|s| !s.is_empty()));
    assert!(
        out["verification"]
            .as_array()
            .is_some_and(|v| !v.is_empty())
    );
}

#[test]
fn plugins_commands_round_trip_json() {
    let workspace = TempDir::new().expect("workspace");
    let plugin_root = workspace.path().join("demo-plugin");
    fs::create_dir_all(plugin_root.join(".deepseek-plugin")).expect("plugin manifest dir");
    fs::create_dir_all(plugin_root.join("commands")).expect("plugin commands dir");
    fs::write(
        plugin_root.join(".deepseek-plugin/plugin.json"),
        r#"{"id":"demo","name":"Demo","version":"0.1.0"}"#,
    )
    .expect("plugin manifest");
    fs::write(plugin_root.join("commands/build.md"), "# build").expect("plugin command");

    let install = run_json(
        workspace.path(),
        &[
            "--json",
            "plugins",
            "install",
            plugin_root.to_string_lossy().as_ref(),
        ],
    );
    assert_eq!(install["manifest"]["id"], "demo");

    let listed = run_json(workspace.path(), &["--json", "plugins", "list"]);
    assert!(listed.as_array().is_some_and(|l| !l.is_empty()));

    let disabled = run_json(workspace.path(), &["--json", "plugins", "disable", "demo"]);
    assert_eq!(disabled["disabled"], "demo");

    let inspected = run_json(workspace.path(), &["--json", "plugins", "inspect", "demo"]);
    assert_eq!(inspected["manifest"]["version"], "0.1.0");
    assert_eq!(inspected["enabled"], false);

    let removed = run_json(workspace.path(), &["--json", "plugins", "remove", "demo"]);
    assert_eq!(removed["removed"], "demo");
}

#[test]
fn plugin_run_executes_command_prompt() {
    let workspace = TempDir::new().expect("workspace");
    let plugin_root = workspace.path().join("demo-plugin");
    fs::create_dir_all(plugin_root.join(".deepseek-plugin")).expect("plugin manifest dir");
    fs::create_dir_all(plugin_root.join("commands")).expect("plugin commands dir");
    fs::write(
        plugin_root.join(".deepseek-plugin/plugin.json"),
        r#"{"id":"demo","name":"Demo","version":"0.1.0"}"#,
    )
    .expect("plugin manifest");
    fs::write(
        plugin_root.join("commands/review.md"),
        "Please review: {{input}}",
    )
    .expect("plugin command");

    let _ = run_json(
        workspace.path(),
        &[
            "--json",
            "plugins",
            "install",
            plugin_root.to_string_lossy().as_ref(),
        ],
    );

    let out = run_json(
        workspace.path(),
        &[
            "--json",
            "plugins",
            "run",
            "demo",
            "review",
            "--input",
            "router thresholds",
            "--tools",
            "false",
            "--max-think",
            "true",
        ],
    );
    assert_eq!(out["plugin_id"], "demo");
    assert_eq!(out["command_name"], "review");
    assert_eq!(out["tools"], false);
    assert!(out["output"].as_str().is_some_and(|s| !s.is_empty()));
}

#[test]
fn autopilot_runs_bounded_iteration_in_json_mode() {
    let workspace = TempDir::new().expect("workspace");
    let out = run_json(
        workspace.path(),
        &[
            "--json",
            "autopilot",
            "Stabilize project",
            "--max-iterations",
            "1",
            "--duration-seconds",
            "30",
            "--tools",
            "false",
            "--max-think",
            "true",
        ],
    );

    assert_eq!(out["stop_reason"], "max_iterations_reached");
    assert_eq!(out["completed_iterations"], 1);
    assert_eq!(out["failed_iterations"], 0);
}

#[test]
fn autopilot_respects_stop_file_and_writes_heartbeat() {
    let workspace = TempDir::new().expect("workspace");
    let stop_file = workspace.path().join(".deepseek/autopilot.stop");
    let heartbeat_file = workspace.path().join(".deepseek/autopilot.heartbeat.json");
    fs::create_dir_all(
        stop_file
            .parent()
            .expect("stop file must have parent directory"),
    )
    .expect("create stop dir");
    fs::write(&stop_file, "stop").expect("write stop file");

    let out = run_json(
        workspace.path(),
        &[
            "--json",
            "autopilot",
            "Stabilize project",
            "--duration-seconds",
            "30",
            "--max-iterations",
            "5",
            "--stop-file",
            stop_file.to_string_lossy().as_ref(),
            "--heartbeat-file",
            heartbeat_file.to_string_lossy().as_ref(),
        ],
    );

    assert_eq!(out["stop_reason"], "stop_file_detected");
    assert_eq!(out["completed_iterations"], 0);

    let heartbeat: Value = serde_json::from_slice(
        &fs::read(heartbeat_file).expect("heartbeat file should be written"),
    )
    .expect("heartbeat json");
    assert_eq!(heartbeat["status"], "stopped");
}

#[test]
fn status_usage_compact_and_doctor_emit_json() {
    let workspace = TempDir::new().expect("workspace");
    let _ = run_json(
        workspace.path(),
        &["--json", "ask", "Create baseline session for status checks"],
    );

    let status = run_json(workspace.path(), &["--json", "status"]);
    assert!(status["session_id"].as_str().is_some());
    assert!(status["model"]["base"].as_str().is_some());

    let usage = run_json(workspace.path(), &["--json", "usage", "--session"]);
    assert!(usage["records"].as_u64().unwrap_or(0) >= 1);

    let compact = run_json(workspace.path(), &["--json", "compact", "--yes"]);
    assert_eq!(compact["persisted"], true);
    assert!(compact["result"]["summary_id"].as_str().is_some());

    let doctor = run_json(workspace.path(), &["--json", "doctor"]);
    assert!(doctor["os"].as_str().is_some());
    assert!(doctor["checks"]["cargo"].as_bool().is_some());
}

#[test]
fn autopilot_status_stop_resume_commands_work() {
    let workspace = TempDir::new().expect("workspace");
    let run = run_json(
        workspace.path(),
        &[
            "--json",
            "autopilot",
            "Control command smoke test",
            "--max-iterations",
            "1",
            "--duration-seconds",
            "30",
            "--tools",
            "false",
        ],
    );
    let run_id = run["run_id"].as_str().expect("run id").to_string();

    let status = run_json(
        workspace.path(),
        &["--json", "autopilot", "status", "--run-id", &run_id],
    );
    assert_eq!(status["run_id"], run_id);

    let stop = run_json(
        workspace.path(),
        &["--json", "autopilot", "stop", "--run-id", &run_id],
    );
    assert_eq!(stop["stop_requested"], true);

    let resumed = run_json(
        workspace.path(),
        &["--json", "autopilot", "resume", "--run-id", &run_id],
    );
    assert!(resumed["run_id"].as_str().is_some());
    assert!(resumed["stop_reason"].as_str().is_some());
}

#[test]
fn plugins_catalog_search_and_verify_emit_json() {
    let workspace = TempDir::new().expect("workspace");
    let catalog_root = workspace.path().join(".deepseek/plugins");
    fs::create_dir_all(&catalog_root).expect("catalog dir");
    fs::write(
        catalog_root.join("catalog.json"),
        r#"{
  "plugins": [
    {
      "id": "demo-plugin",
      "name": "Demo Plugin",
      "version": "0.1.0",
      "description": "catalog fixture",
      "source": "https://example.com/demo"
    }
  ]
}"#,
    )
    .expect("catalog file");

    let catalog = run_json(workspace.path(), &["--json", "plugins", "catalog"]);
    assert_eq!(catalog.as_array().map(|a| a.len()).unwrap_or_default(), 1);

    let search = run_json(
        workspace.path(),
        &["--json", "plugins", "search", "demo-plugin"],
    );
    assert_eq!(search.as_array().map(|a| a.len()).unwrap_or_default(), 1);

    let verify = run_json(
        workspace.path(),
        &["--json", "plugins", "verify", "demo-plugin"],
    );
    assert_eq!(verify["plugin_id"], "demo-plugin");
    assert_eq!(verify["verified"], false);
}

#[test]
fn mcp_memory_export_and_profile_emit_json() {
    let workspace = TempDir::new().expect("workspace");

    let memory = run_json(workspace.path(), &["--json", "memory", "show"]);
    assert!(
        memory["path"]
            .as_str()
            .is_some_and(|path| path.ends_with("DEEPSEEK.md"))
    );

    let sync = run_json(
        workspace.path(),
        &["--json", "memory", "sync", "--note", "test-sync"],
    );
    assert_eq!(sync["synced"], true);
    assert_eq!(sync["note"], "test-sync");

    let added = run_json(
        workspace.path(),
        &[
            "--json",
            "mcp",
            "add",
            "local",
            "--name",
            "Local MCP",
            "--transport",
            "stdio",
            "--command",
            "echo",
            "--arg",
            "ok",
            "--metadata",
            r#"{"tools":[{"name":"ping","description":"health"}]}"#,
        ],
    );
    assert_eq!(added["added"]["id"], "local");

    let listed = run_json(workspace.path(), &["--json", "mcp", "list"]);
    assert!(listed.as_array().is_some_and(|servers| !servers.is_empty()));

    let got = run_json(workspace.path(), &["--json", "mcp", "get", "local"]);
    assert_eq!(got["id"], "local");

    let _ = run_json(
        workspace.path(),
        &["--json", "ask", "Generate transcript for export command"],
    );
    let exported = run_json(workspace.path(), &["--json", "export", "--format", "md"]);
    assert_eq!(exported["format"], "md");
    assert!(exported["output_path"].as_str().is_some());

    let profile = run_json(workspace.path(), &["--json", "profile"]);
    assert!(profile["profile_id"].as_str().is_some());
    assert!(profile["elapsed_ms"].as_u64().is_some());

    let removed = run_json(workspace.path(), &["--json", "mcp", "remove", "local"]);
    assert_eq!(removed["removed"], true);
}

#[test]
fn rewind_uses_checkpoint_id_from_apply() {
    let workspace = TempDir::new().expect("workspace");
    let _ = run_json(
        workspace.path(),
        &["--json", "ask", "Create patch so apply can checkpoint"],
    );

    let applied = run_json(workspace.path(), &["--json", "apply", "--yes"]);
    let checkpoint_id = applied["checkpoint_id"]
        .as_str()
        .expect("checkpoint id from apply")
        .to_string();

    fs::write(workspace.path().join("scratch.txt"), "modified").expect("write");

    let rewound = run_json(
        workspace.path(),
        &[
            "--json",
            "rewind",
            "--to-checkpoint",
            &checkpoint_id,
            "--yes",
        ],
    );
    assert_eq!(rewound["rewound"], true);
    assert_eq!(rewound["checkpoint_id"], checkpoint_id);
}

#[test]
fn git_skills_replay_background_teleport_and_remote_env_emit_json() {
    if StdCommand::new("git").arg("--version").output().is_err() {
        return;
    }

    let workspace = TempDir::new().expect("workspace");
    let init_status = StdCommand::new("git")
        .current_dir(workspace.path())
        .arg("init")
        .status()
        .expect("git init");
    assert!(init_status.success());

    let git_status = run_json(workspace.path(), &["--json", "git", "status"]);
    assert!(git_status["output"].as_str().is_some());

    let skill_src = workspace.path().join("demo-skill");
    fs::create_dir_all(&skill_src).expect("skill source");
    fs::write(
        skill_src.join("SKILL.md"),
        "# Demo Skill\n\nAnalyze {{input}} in {{workspace}}.",
    )
    .expect("skill");

    let skill_install = run_json(
        workspace.path(),
        &[
            "--json",
            "skills",
            "install",
            skill_src.to_string_lossy().as_ref(),
        ],
    );
    assert_eq!(skill_install["id"], "demo-skill");

    let skill_run = run_json(
        workspace.path(),
        &[
            "--json",
            "skills",
            "run",
            "demo-skill",
            "--input",
            "routing",
        ],
    );
    assert_eq!(skill_run["skill_id"], "demo-skill");

    let _ = run_json(
        workspace.path(),
        &[
            "--json",
            "autopilot",
            "create replay fixture",
            "--max-iterations",
            "1",
            "--duration-seconds",
            "30",
            "--tools",
            "false",
        ],
    );
    let jobs = run_json(workspace.path(), &["--json", "background", "list"]);
    let first_job = jobs
        .as_array()
        .and_then(|rows| rows.first())
        .expect("background job");
    let job_id = first_job["job_id"].as_str().expect("job id");

    let attached = run_json(
        workspace.path(),
        &["--json", "background", "attach", job_id],
    );
    assert_eq!(attached["job_id"], job_id);

    let stopped = run_json(workspace.path(), &["--json", "background", "stop", job_id]);
    assert_eq!(stopped["stopped"], true);

    let status = run_json(workspace.path(), &["--json", "status"]);
    let session_id = status["session_id"].as_str().expect("session id");
    let replay = run_json(
        workspace.path(),
        &[
            "--json",
            "replay",
            "run",
            "--session-id",
            session_id,
            "--deterministic",
            "true",
        ],
    );
    assert_eq!(replay["deterministic"], true);
    assert!(replay["tool_results_replayed"].as_u64().is_some());

    let teleport = run_json(workspace.path(), &["--json", "teleport"]);
    assert!(teleport["bundle_id"].as_str().is_some());
    let teleport_path = teleport["path"].as_str().expect("teleport path");
    let imported = run_json(
        workspace.path(),
        &["--json", "teleport", "--import", teleport_path],
    );
    assert_eq!(imported["imported"], true);

    let added = run_json(
        workspace.path(),
        &[
            "--json",
            "remote-env",
            "add",
            "devbox",
            "https://example.invalid",
        ],
    );
    let profile_id = added["profile_id"].as_str().expect("profile id");

    let checked = run_json(
        workspace.path(),
        &["--json", "remote-env", "check", profile_id],
    );
    assert_eq!(checked["profile_id"], profile_id);

    let removed = run_json(
        workspace.path(),
        &["--json", "remote-env", "remove", profile_id],
    );
    assert_eq!(removed["removed"], true);
}

fn run_json(workspace: &Path, args: &[&str]) -> Value {
    let runtime = workspace.join(".deepseek");
    fs::create_dir_all(&runtime).expect("runtime dir");
    fs::write(
        runtime.join("settings.local.json"),
        r#"{"llm":{"offline_fallback":true}}"#,
    )
    .expect("settings override");
    let output = Command::new(assert_cmd::cargo::cargo_bin!("deepseek"))
        .current_dir(workspace)
        .args(args)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    serde_json::from_slice(&output).expect("json output")
}
