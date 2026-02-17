use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
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

fn run_json(workspace: &Path, args: &[&str]) -> Value {
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
