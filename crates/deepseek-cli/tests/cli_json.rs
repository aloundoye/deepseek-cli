use assert_cmd::Command;
use deepseek_store::{Store, VisualArtifactRecord};
use serde_json::Value;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::Command as StdCommand;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use uuid::Uuid;

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
fn ask_json_fails_fast_without_api_key_when_api_key_missing() {
    let workspace = TempDir::new().expect("workspace");
    let output = Command::new(assert_cmd::cargo::cargo_bin!("deepseek"))
        .current_dir(workspace.path())
        .env_remove("DEEPSEEK_API_KEY")
        .args(["--json", "ask", "hello"])
        .assert()
        .failure()
        .get_output()
        .stderr
        .clone();
    let stderr = String::from_utf8_lossy(&output);
    assert!(stderr.contains("DEEPSEEK_API_KEY"));
}

#[test]
fn ask_json_uses_workspace_local_api_key_when_env_missing() {
    let workspace = TempDir::new().expect("workspace");
    let mock = start_mock_llm_server();
    configure_runtime_for_mock_llm(workspace.path(), &mock.endpoint);
    fs::write(
        workspace.path().join(".deepseek/settings.local.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "llm": {
                "provider": "deepseek",
                "profile": "v3_2",
                "endpoint": mock.endpoint,
                "api_key_env": "DEEPSEEK_API_KEY",
                "api_key": "workspace-local-key"
            }
        }))
        .expect("serialize settings"),
    )
    .expect("settings");

    let output = Command::new(assert_cmd::cargo::cargo_bin!("deepseek"))
        .current_dir(workspace.path())
        .env_remove("DEEPSEEK_API_KEY")
        .args(["--json", "ask", "hello"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let payload: Value = serde_json::from_slice(&output).expect("json payload");
    assert!(
        payload["output"]
            .as_str()
            .is_some_and(|value| !value.is_empty())
    );
}

#[test]
fn ask_json_rejects_invalid_deepseek_profile() {
    let workspace = TempDir::new().expect("workspace");
    let mock = start_mock_llm_server();
    configure_runtime_for_mock_llm(workspace.path(), &mock.endpoint);
    fs::write(
        workspace.path().join(".deepseek/settings.local.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "llm": {
                "provider": "deepseek",
                "profile": "invalid_profile",
                "endpoint": mock.endpoint,
                "api_key_env": "DEEPSEEK_API_KEY"
            }
        }))
        .expect("serialize settings"),
    )
    .expect("settings");

    let output = Command::new(assert_cmd::cargo::cargo_bin!("deepseek"))
        .current_dir(workspace.path())
        .env("DEEPSEEK_API_KEY", "test-api-key")
        .args(["--json", "ask", "hello"])
        .assert()
        .failure()
        .get_output()
        .stderr
        .clone();
    let stderr = String::from_utf8_lossy(&output);
    assert!(stderr.contains("llm.profile"));
}

#[test]
fn ask_json_requires_speciale_profile_when_speciale_model_is_selected() {
    let workspace = TempDir::new().expect("workspace");
    let mock = start_mock_llm_server();
    configure_runtime_for_mock_llm(workspace.path(), &mock.endpoint);
    fs::write(
        workspace.path().join(".deepseek/settings.local.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "llm": {
                "provider": "deepseek",
                "profile": "v3_2",
                "base_model": "deepseek-v3.2-speciale",
                "endpoint": mock.endpoint,
                "api_key_env": "DEEPSEEK_API_KEY"
            }
        }))
        .expect("serialize settings"),
    )
    .expect("settings");

    let output = Command::new(assert_cmd::cargo::cargo_bin!("deepseek"))
        .current_dir(workspace.path())
        .env("DEEPSEEK_API_KEY", "test-api-key")
        .args(["--json", "ask", "hello"])
        .assert()
        .failure()
        .get_output()
        .stderr
        .clone();
    let stderr = String::from_utf8_lossy(&output);
    assert!(stderr.contains("requires llm.profile='v3_2_speciale'"));
}

#[test]
fn config_show_redacts_api_key_in_json_and_text_modes() {
    let workspace = TempDir::new().expect("workspace");
    let runtime = workspace.path().join(".deepseek");
    fs::create_dir_all(&runtime).expect("runtime dir");
    fs::write(
        runtime.join("settings.local.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "llm": {
                "api_key": "dsk-secret-value"
            }
        }))
        .expect("serialize settings"),
    )
    .expect("settings file");

    let json_out = Command::new(assert_cmd::cargo::cargo_bin!("deepseek"))
        .current_dir(workspace.path())
        .args(["--json", "config", "show"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let payload: Value = serde_json::from_slice(&json_out).expect("json");
    assert_eq!(payload["llm"]["api_key"], "***REDACTED***");

    let text_out = Command::new(assert_cmd::cargo::cargo_bin!("deepseek"))
        .current_dir(workspace.path())
        .args(["config", "show"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8_lossy(&text_out);
    assert!(stdout.contains("***REDACTED***"));
    assert!(!stdout.contains("dsk-secret-value"));
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
    assert!(
        doctor["leadership"]["readiness"]["score"]
            .as_u64()
            .is_some()
    );
    assert!(
        doctor["leadership"]["enterprise"]["audit"]["events_total"]
            .as_u64()
            .is_some()
    );
    assert!(
        doctor["leadership"]["ecosystem"]["coverage"]["scm"]
            .as_bool()
            .is_some()
    );
    assert!(
        doctor["leadership"]["deployment"]["coverage"]["ci"]
            .as_bool()
            .is_some()
    );
}

#[test]
fn parity_flags_update_and_teleport_emit_json() {
    let workspace = TempDir::new().expect("workspace");

    let sources = run_json(workspace.path(), &["--json", "--setting-sources"]);
    assert_eq!(sources["schema"], "deepseek.settings_sources.v1");
    assert!(sources["user"].as_str().is_some());
    assert!(sources["project"].as_str().is_some());
    assert!(sources["project_local"].as_str().is_some());
    assert!(sources["legacy_toml"].as_str().is_some());

    let status = run_json(
        workspace.path(),
        &[
            "--json",
            "--ide",
            "--remote",
            "--teammate-mode",
            "pair",
            "--betas",
            "ui,tools",
            "--maintenance",
            "--include-partial-messages",
            "--allowedTools",
            "fs_read",
            "status",
        ],
    );
    assert!(status["state"].as_str().is_some());

    let status2 = run_json(
        workspace.path(),
        &["--json", "--disallowedTools", "bash_run", "status"],
    );
    assert!(status2["state"].as_str().is_some());

    let update = run_json(
        workspace.path(),
        &["--json", "update", "--check", "--channel", "stable"],
    );
    assert_eq!(update["schema"], "deepseek.update.v1");
    assert_eq!(update["check_only"], true);
    assert_eq!(update["channel"], "stable");

    let teleport = run_json(workspace.path(), &["--json", "--teleport"]);
    assert!(teleport["bundle_id"].as_str().is_some());
}

#[test]
fn remote_env_check_performs_real_health_probe() {
    let workspace = TempDir::new().expect("workspace");
    let mock = start_mock_llm_server();

    let added = run_json(
        workspace.path(),
        &[
            "--json",
            "remote-env",
            "add",
            "local",
            mock.endpoint.as_str(),
        ],
    );
    let profile_id = added["profile_id"]
        .as_str()
        .expect("profile id")
        .to_string();

    let checked = run_json(
        workspace.path(),
        &["--json", "remote-env", "check", profile_id.as_str()],
    );
    assert_eq!(checked["profile_id"].as_str(), Some(profile_id.as_str()));
    assert_eq!(checked["reachable"], true);
    assert!(checked["latency_ms"].as_u64().is_some());
    assert!(checked["status_code"].as_u64().is_some());
    assert!(checked["checked_url"].as_str().is_some());
}

#[test]
fn leadership_report_detects_phase3_signals() {
    let workspace = TempDir::new().expect("workspace");

    fs::create_dir_all(workspace.path().join(".github/workflows")).expect("workflows dir");
    fs::write(
        workspace.path().join(".github/workflows/ci.yml"),
        "name: ci\non: [push]\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - run: cargo test\n",
    )
    .expect("ci workflow");

    fs::write(
        workspace.path().join("Dockerfile"),
        "FROM rust:latest\nWORKDIR /app\nCOPY . .\nRUN cargo build --release\n",
    )
    .expect("dockerfile");

    fs::create_dir_all(workspace.path().join("k8s")).expect("k8s dir");
    fs::write(
        workspace.path().join("k8s/deployment.yaml"),
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: app\nspec:\n  template:\n    spec:\n      containers:\n      - name: app\n        image: example/app:1\n",
    )
    .expect("k8s manifest");

    fs::write(
        workspace.path().join("main.tf"),
        "terraform { required_version = \">= 1.5.0\" }\n",
    )
    .expect("terraform");

    fs::create_dir_all(workspace.path().join(".deepseek")).expect("runtime dir");
    fs::write(
        workspace.path().join(".deepseek/sso.json"),
        "{\"provider\":\"okta\",\"oauth2\":true}\n",
    )
    .expect("sso config");

    let _ = run_json(
        workspace.path(),
        &["--json", "ask", "seed enterprise audit events"],
    );

    let payload = run_json_with_env(
        workspace.path(),
        &["--json", "leadership", "--audit-window-hours", "72"],
        &[("JIRA_API_TOKEN", "token"), ("SLACK_BOT_TOKEN", "token")],
    );
    assert_eq!(payload["enterprise"]["sso"]["configured"], true);
    assert_eq!(payload["ecosystem"]["coverage"]["scm"], true);
    assert_eq!(payload["ecosystem"]["coverage"]["work_tracking"], true);
    assert_eq!(payload["ecosystem"]["coverage"]["collaboration"], true);
    assert_eq!(payload["deployment"]["coverage"]["ci"], true);
    assert_eq!(payload["deployment"]["coverage"]["container"], true);
    assert_eq!(payload["deployment"]["coverage"]["kubernetes"], true);
    assert_eq!(payload["deployment"]["coverage"]["cloud"], true);
    assert!(payload["readiness"]["score"].as_u64().unwrap_or(0) >= 1);
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
    assert!(status["pause_file"].as_str().is_some());

    let follow = run_json(
        workspace.path(),
        &[
            "--json",
            "autopilot",
            "status",
            "--run-id",
            &run_id,
            "--follow",
            "--samples",
            "2",
            "--interval-seconds",
            "1",
        ],
    );
    assert_eq!(follow["follow"], true);
    assert!(follow["samples_collected"].as_u64().unwrap_or(0) >= 1);
    assert!(
        follow["samples"]
            .as_array()
            .is_some_and(|rows| !rows.is_empty())
    );

    let stop = run_json(
        workspace.path(),
        &["--json", "autopilot", "stop", "--run-id", &run_id],
    );
    assert_eq!(stop["stop_requested"], true);

    let pause = run_json(
        workspace.path(),
        &["--json", "autopilot", "pause", "--run-id", &run_id],
    );
    assert_eq!(pause["pause_requested"], true);

    let resumed = run_json(
        workspace.path(),
        &["--json", "autopilot", "resume", "--run-id", &run_id],
    );
    assert!(resumed["run_id"].as_str().is_some());
    if resumed["resumed_live"].is_null() {
        assert!(resumed["stop_reason"].as_str().is_some());
    } else {
        assert_eq!(resumed["resumed_live"], true);
    }
}

#[test]
fn permissions_show_set_and_status_emit_json() {
    let workspace = TempDir::new().expect("workspace");
    let shown = run_json(workspace.path(), &["--json", "permissions", "show"]);
    assert!(shown["policy"]["approve_bash"].as_str().is_some());
    assert!(shown["policy"]["allowlist"].is_array());

    let updated = run_json(
        workspace.path(),
        &[
            "--json",
            "permissions",
            "set",
            "--approve-bash",
            "always",
            "--approve-edits",
            "never",
            "--sandbox-mode",
            "workspace-write",
            "--clear-allowlist",
            "--allow",
            "npm *",
        ],
    );
    assert_eq!(updated["updated"], true);
    assert_eq!(updated["policy"]["approve_bash"], "always");
    assert_eq!(updated["policy"]["approve_edits"], "never");
    assert_eq!(updated["policy"]["sandbox_mode"], "workspace-write");
    assert!(
        updated["policy"]["allowlist"]
            .as_array()
            .is_some_and(|items| items.iter().any(|entry| entry == "npm *"))
    );

    let status = run_json(workspace.path(), &["--json", "status"]);
    assert_eq!(status["permissions"]["approve_bash"], "always");
    assert_eq!(status["permissions"]["approve_edits"], "never");
    assert_eq!(status["permissions"]["sandbox_mode"], "workspace-write");
}

#[test]
fn permissions_set_respects_team_policy_locks() {
    let workspace = TempDir::new().expect("workspace");
    let team_policy_path = workspace.path().join(".deepseek/team-policy.json");
    fs::create_dir_all(
        team_policy_path
            .parent()
            .expect("team policy parent directory"),
    )
    .expect("team policy dir");
    fs::write(
        &team_policy_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "approve_bash": "never",
            "allowlist": ["git status"],
            "sandbox_mode": "read-only"
        }))
        .expect("serialize team policy"),
    )
    .expect("write team policy");

    let shown = run_json_with_env(
        workspace.path(),
        &["--json", "permissions", "show"],
        &[(
            "DEEPSEEK_TEAM_POLICY_PATH",
            team_policy_path.to_string_lossy().as_ref(),
        )],
    );
    assert_eq!(shown["team_policy"]["active"], true);
    assert_eq!(shown["team_policy"]["approve_bash_locked"], true);
    assert_eq!(shown["team_policy"]["allowlist_locked"], true);
    assert_eq!(shown["team_policy"]["sandbox_mode_locked"], true);

    let output = Command::new(assert_cmd::cargo::cargo_bin!("deepseek"))
        .current_dir(workspace.path())
        .env(
            "DEEPSEEK_TEAM_POLICY_PATH",
            team_policy_path.to_string_lossy().as_ref(),
        )
        .args([
            "--json",
            "permissions",
            "set",
            "--approve-bash",
            "always",
            "--allow",
            "npm *",
        ])
        .assert()
        .failure()
        .get_output()
        .stderr
        .clone();
    let stderr = String::from_utf8_lossy(&output);
    assert!(stderr.contains("locks permissions fields"));
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

    let profile_benchmark = run_json(
        workspace.path(),
        &[
            "--json",
            "profile",
            "--benchmark",
            "--benchmark-cases",
            "2",
            "--benchmark-min-success-rate",
            "0.0",
            "--benchmark-max-p95-ms",
            "60000",
            "--benchmark-output",
            workspace
                .path()
                .join(".deepseek/benchmark.json")
                .to_string_lossy()
                .as_ref(),
        ],
    );
    assert_eq!(profile_benchmark["benchmark"]["executed_cases"], 2);
    assert!(profile_benchmark["benchmark"]["records"].is_array());
    assert!(workspace.path().join(".deepseek/benchmark.json").exists());

    fs::write(
        workspace.path().join(".deepseek/baseline-benchmark.json"),
        serde_json::to_vec_pretty(&profile_benchmark["benchmark"]).expect("serialize baseline"),
    )
    .expect("baseline");
    let compared = run_json(
        workspace.path(),
        &[
            "--json",
            "profile",
            "--benchmark",
            "--benchmark-cases",
            "2",
            "--benchmark-min-quality-rate",
            "0.0",
            "--benchmark-baseline",
            workspace
                .path()
                .join(".deepseek/baseline-benchmark.json")
                .to_string_lossy()
                .as_ref(),
            "--benchmark-max-regression-ms",
            "60000",
        ],
    );
    assert!(compared["benchmark_comparison"].is_object());
    assert!(compared["benchmark_comparison"]["p95_regression_ms"].is_u64());

    fs::write(
        workspace.path().join(".deepseek/peer-codex.json"),
        r#"{
  "agent": "codex",
  "benchmark": {
    "executed_cases": 2,
    "succeeded": 2,
    "p95_latency_ms": 1200,
    "records": [
      {"ok": true, "quality_ok": true},
      {"ok": true, "quality_ok": true}
    ]
  }
}"#,
    )
    .expect("peer report");
    let peer_compare = run_json(
        workspace.path(),
        &[
            "--json",
            "profile",
            "--benchmark",
            "--benchmark-cases",
            "2",
            "--benchmark-compare",
            workspace
                .path()
                .join(".deepseek/peer-codex.json")
                .to_string_lossy()
                .as_ref(),
            "--benchmark-min-quality-rate",
            "0.0",
        ],
    );
    assert!(peer_compare["benchmark_peer_comparison"]["ranking"].is_array());
    assert!(
        peer_compare["benchmark_peer_comparison"]["ranking"]
            .as_array()
            .is_some_and(|rows| rows.len() >= 2)
    );
    assert!(
        peer_compare["benchmark_peer_comparison"]["case_matrix"]
            .as_array()
            .is_some_and(|rows| !rows.is_empty())
    );

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

    // Stage a patch directly so `apply --yes` can create a checkpoint.
    // The chat-with-tools path doesn't auto-stage patches like the legacy
    // plan-and-execute path did.
    let store = deepseek_diff::PatchStore::new(workspace.path()).expect("patch store");
    let diff = "diff --git a/.deepseek/notes.txt b/.deepseek/notes.txt\nnew file mode 100644\nindex 0000000..2b9d865\n--- /dev/null\n+++ b/.deepseek/notes.txt\n@@ -0,0 +1 @@\n+rewind checkpoint test\n";
    let _ = store.stage(diff, &[]).expect("stage patch");

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
    let initial_branch = String::from_utf8_lossy(
        &StdCommand::new("git")
            .current_dir(workspace.path())
            .args(["branch", "--show-current"])
            .output()
            .expect("git branch --show-current")
            .stdout,
    )
    .trim()
    .to_string();
    assert!(!initial_branch.is_empty());

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
    let replay_list = run_json(
        workspace.path(),
        &[
            "--json",
            "replay",
            "list",
            "--session-id",
            session_id,
            "--limit",
            "5",
        ],
    );
    assert!(
        replay_list
            .as_array()
            .is_some_and(|rows| rows.iter().any(|row| row["session_id"] == session_id))
    );

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

#[test]
fn git_resolve_all_and_stage_clears_conflicts() {
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
    let initial_branch = String::from_utf8_lossy(
        &StdCommand::new("git")
            .current_dir(workspace.path())
            .args(["branch", "--show-current"])
            .output()
            .expect("git branch --show-current")
            .stdout,
    )
    .trim()
    .to_string();
    assert!(!initial_branch.is_empty());
    assert!(
        StdCommand::new("git")
            .current_dir(workspace.path())
            .args(["config", "user.email", "test@example.com"])
            .status()
            .expect("git config email")
            .success()
    );
    assert!(
        StdCommand::new("git")
            .current_dir(workspace.path())
            .args(["config", "user.name", "Test User"])
            .status()
            .expect("git config name")
            .success()
    );

    let conflicted = workspace.path().join("conflicted.txt");
    fs::write(&conflicted, "base\n").expect("write base");
    assert!(
        StdCommand::new("git")
            .current_dir(workspace.path())
            .args(["add", "conflicted.txt"])
            .status()
            .expect("git add")
            .success()
    );
    assert!(
        StdCommand::new("git")
            .current_dir(workspace.path())
            .args(["commit", "-m", "base"])
            .status()
            .expect("git commit base")
            .success()
    );

    assert!(
        StdCommand::new("git")
            .current_dir(workspace.path())
            .args(["checkout", "-b", "feature"])
            .status()
            .expect("git checkout feature")
            .success()
    );
    fs::write(&conflicted, "feature\n").expect("write feature");
    assert!(
        StdCommand::new("git")
            .current_dir(workspace.path())
            .args(["commit", "-am", "feature"])
            .status()
            .expect("git commit feature")
            .success()
    );

    assert!(
        StdCommand::new("git")
            .current_dir(workspace.path())
            .args(["checkout", &initial_branch])
            .status()
            .expect("git checkout initial branch")
            .success()
    );
    fs::write(&conflicted, "main\n").expect("write main");
    assert!(
        StdCommand::new("git")
            .current_dir(workspace.path())
            .args(["commit", "-am", "main"])
            .status()
            .expect("git commit main")
            .success()
    );

    let merge_status = StdCommand::new("git")
        .current_dir(workspace.path())
        .args(["merge", "feature"])
        .status()
        .expect("git merge");
    assert!(!merge_status.success());

    let resolved = run_json(
        workspace.path(),
        &[
            "--json",
            "git",
            "resolve",
            "--strategy",
            "ours",
            "--all",
            "--stage",
        ],
    );
    assert!(resolved["count"].as_u64().unwrap_or(0) >= 1);
    assert_eq!(resolved["stage"], true);
    assert!(
        resolved["resolved_files"]
            .as_array()
            .is_some_and(|rows| rows.iter().any(|row| row == "conflicted.txt"))
    );

    let remaining = StdCommand::new("git")
        .current_dir(workspace.path())
        .args(["diff", "--name-only", "--diff-filter=U"])
        .output()
        .expect("git diff conflicts");
    assert!(remaining.status.success());
    let remaining_stdout = String::from_utf8_lossy(&remaining.stdout);
    assert!(remaining_stdout.trim().is_empty());
}

#[test]
fn background_run_shell_attach_tail_and_stop_emit_json() {
    let workspace = TempDir::new().expect("workspace");
    let started = run_json(
        workspace.path(),
        &[
            "--json",
            "background",
            "run-shell",
            "echo",
            "background-shell-ok",
        ],
    );
    let job_id = started["job_id"].as_str().expect("job id");
    let stdout_log = started["stdout_log"].as_str().expect("stdout log");
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        let text = fs::read_to_string(stdout_log).unwrap_or_default();
        if text.contains("background-shell-ok") {
            break;
        }
        thread::sleep(Duration::from_millis(50));
    }

    let attached = run_json(
        workspace.path(),
        &[
            "--json",
            "background",
            "attach",
            job_id,
            "--tail-lines",
            "20",
        ],
    );
    assert_eq!(attached["job_id"], job_id);
    assert!(
        attached["log_tail"]["stdout"]
            .as_str()
            .unwrap_or_default()
            .contains("background-shell-ok")
    );

    let stopped = run_json(workspace.path(), &["--json", "background", "stop", job_id]);
    assert_eq!(stopped["stopped"], true);
}

#[test]
fn visual_list_and_analyze_emit_json() {
    let workspace = TempDir::new().expect("workspace");
    let image_rel = "ui/screen.png";
    let image_path = workspace.path().join(image_rel);
    fs::create_dir_all(
        image_path
            .parent()
            .expect("image file should have a parent directory"),
    )
    .expect("image dir");
    fs::write(
        &image_path,
        vec![0x89, b'P', b'N', b'G', 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    .expect("image");

    let store = Store::new(workspace.path()).expect("store");
    store
        .insert_visual_artifact(&VisualArtifactRecord {
            artifact_id: Uuid::now_v7(),
            path: image_rel.to_string(),
            mime: "image/png".to_string(),
            metadata_json: "{}".to_string(),
            created_at: "2026-02-18T00:00:00Z".to_string(),
        })
        .expect("insert visual artifact");

    let listed = run_json(
        workspace.path(),
        &["--json", "visual", "list", "--limit", "10"],
    );
    assert_eq!(listed.as_array().map(|rows| rows.len()), Some(1));
    assert_eq!(listed[0]["path"], image_rel);
    assert_eq!(listed[0]["exists"], true);

    let analyzed = run_json(
        workspace.path(),
        &[
            "--json",
            "visual",
            "analyze",
            "--limit",
            "10",
            "--min-bytes",
            "8",
            "--min-artifacts",
            "1",
            "--min-image-artifacts",
            "1",
            "--strict",
        ],
    );
    assert_eq!(analyzed["ok"], true);
    assert_eq!(analyzed["summary"]["image_like_artifacts"], 1);
}

#[test]
fn visual_analyze_strict_fails_without_artifacts() {
    let workspace = TempDir::new().expect("workspace");
    let output = Command::new(assert_cmd::cargo::cargo_bin!("deepseek"))
        .current_dir(workspace.path())
        .args([
            "--json",
            "visual",
            "analyze",
            "--strict",
            "--min-artifacts",
            "1",
            "--min-image-artifacts",
            "1",
        ])
        .assert()
        .failure()
        .get_output()
        .stderr
        .clone();
    let stderr = String::from_utf8_lossy(&output);
    assert!(stderr.contains("visual analysis failed"));
}

#[test]
fn visual_analyze_baseline_detects_changes_in_strict_mode() {
    let workspace = TempDir::new().expect("workspace");
    let image_rel = "ui/screen.png";
    let image_path = workspace.path().join(image_rel);
    fs::create_dir_all(
        image_path
            .parent()
            .expect("image file should have a parent directory"),
    )
    .expect("image dir");
    fs::write(&image_path, vec![0x89, b'P', b'N', b'G', 1, 2, 3, 4]).expect("image");

    let store = Store::new(workspace.path()).expect("store");
    store
        .insert_visual_artifact(&VisualArtifactRecord {
            artifact_id: Uuid::now_v7(),
            path: image_rel.to_string(),
            mime: "image/png".to_string(),
            metadata_json: "{}".to_string(),
            created_at: "2026-02-18T00:00:00Z".to_string(),
        })
        .expect("insert visual artifact");

    let baseline_path = workspace.path().join(".deepseek/visual-baseline.json");
    let seeded = run_json(
        workspace.path(),
        &[
            "--json",
            "visual",
            "analyze",
            "--min-bytes",
            "1",
            "--write-baseline",
            baseline_path.to_string_lossy().as_ref(),
        ],
    );
    assert_eq!(seeded["ok"], true);
    assert!(baseline_path.exists());

    fs::write(&image_path, vec![0x89, b'P', b'N', b'G', 9, 9, 9, 9, 9]).expect("mutate image");
    let output = Command::new(assert_cmd::cargo::cargo_bin!("deepseek"))
        .current_dir(workspace.path())
        .args([
            "--json",
            "visual",
            "analyze",
            "--baseline",
            baseline_path.to_string_lossy().as_ref(),
            "--strict",
        ])
        .assert()
        .failure()
        .get_output()
        .stderr
        .clone();
    let stderr = String::from_utf8_lossy(&output);
    assert!(stderr.contains("changed_artifacts"));
}

#[test]
fn visual_analyze_semantic_expectations_pass_with_dimensions() {
    let workspace = TempDir::new().expect("workspace");
    let image_rel = "ui/login-screen.png";
    let image_path = workspace.path().join(image_rel);
    fs::create_dir_all(
        image_path
            .parent()
            .expect("image file should have a parent directory"),
    )
    .expect("image dir");
    write_fake_png_with_dimensions(&image_path, 1280, 720).expect("image");

    let store = Store::new(workspace.path()).expect("store");
    store
        .insert_visual_artifact(&VisualArtifactRecord {
            artifact_id: Uuid::now_v7(),
            path: image_rel.to_string(),
            mime: "image/png".to_string(),
            metadata_json: "{}".to_string(),
            created_at: "2026-02-18T00:00:00Z".to_string(),
        })
        .expect("insert visual artifact");

    let expect_path = workspace.path().join(".deepseek/visual-expect.json");
    fs::create_dir_all(
        expect_path
            .parent()
            .expect("expectation file should have a parent directory"),
    )
    .expect("expect dir");
    fs::write(
        &expect_path,
        r#"{
  "schema": "deepseek.visual.expectation.v1",
  "rules": [
    {
      "name": "login-screen",
      "path_glob": "ui/login-*.png",
      "min_count": 1,
      "mime": "image/png",
      "min_width": 1200,
      "min_height": 700,
      "max_width": 1600,
      "max_height": 900,
      "required_path_substrings": ["login"]
    }
  ]
}"#,
    )
    .expect("expect file");

    let analyzed = run_json(
        workspace.path(),
        &[
            "--json",
            "visual",
            "analyze",
            "--min-bytes",
            "1",
            "--expect",
            expect_path.to_string_lossy().as_ref(),
            "--strict",
        ],
    );
    assert_eq!(analyzed["ok"], true);
    assert_eq!(analyzed["semantic"]["rules_total"], 1);
    assert_eq!(analyzed["semantic"]["rules_passed"], 1);
}

#[test]
fn visual_analyze_semantic_expectations_fail_in_strict_mode() {
    let workspace = TempDir::new().expect("workspace");
    let image_rel = "ui/dashboard.png";
    let image_path = workspace.path().join(image_rel);
    fs::create_dir_all(
        image_path
            .parent()
            .expect("image file should have a parent directory"),
    )
    .expect("image dir");
    write_fake_png_with_dimensions(&image_path, 1280, 720).expect("image");

    let store = Store::new(workspace.path()).expect("store");
    store
        .insert_visual_artifact(&VisualArtifactRecord {
            artifact_id: Uuid::now_v7(),
            path: image_rel.to_string(),
            mime: "image/png".to_string(),
            metadata_json: "{}".to_string(),
            created_at: "2026-02-18T00:00:00Z".to_string(),
        })
        .expect("insert visual artifact");

    let expect_path = workspace.path().join(".deepseek/visual-expect-fail.json");
    fs::create_dir_all(
        expect_path
            .parent()
            .expect("expectation file should have a parent directory"),
    )
    .expect("expect dir");
    fs::write(
        &expect_path,
        r#"{
  "rules": [
    {
      "name": "dashboard-too-wide",
      "path_glob": "ui/*.png",
      "min_count": 1,
      "max_width": 1000
    }
  ]
}"#,
    )
    .expect("expect file");

    let output = Command::new(assert_cmd::cargo::cargo_bin!("deepseek"))
        .current_dir(workspace.path())
        .args([
            "--json",
            "visual",
            "analyze",
            "--min-bytes",
            "1",
            "--expect",
            expect_path.to_string_lossy().as_ref(),
            "--strict",
        ])
        .assert()
        .failure()
        .get_output()
        .stderr
        .clone();
    let stderr = String::from_utf8_lossy(&output);
    assert!(stderr.contains("semantic_expectations_failed"));
}

#[test]
fn profile_benchmark_accepts_external_suite_file() {
    let workspace = TempDir::new().expect("workspace");
    let suite_path = workspace.path().join(".deepseek/benchmark-suite.json");
    fs::create_dir_all(
        suite_path
            .parent()
            .expect("benchmark suite file should have a parent"),
    )
    .expect("suite dir");
    fs::write(
        &suite_path,
        r#"{
  "cases": [
    {
      "case_id": "suite-docs",
      "prompt": "Plan docs updates with verification checks.",
      "expected_keywords": ["verify"],
      "min_steps": 1,
      "min_verification_steps": 1
    },
    {
      "case_id": "suite-git",
      "prompt": "Plan git cleanup and branch verification.",
      "expected_keywords": ["git"],
      "min_steps": 1,
      "min_verification_steps": 1
    }
  ]
}"#,
    )
    .expect("suite");

    let payload = run_json(
        workspace.path(),
        &[
            "--json",
            "profile",
            "--benchmark",
            "--benchmark-suite",
            suite_path.to_string_lossy().as_ref(),
            "--benchmark-cases",
            "2",
            "--benchmark-min-success-rate",
            "0.0",
        ],
    );

    assert_eq!(
        payload["benchmark"]["suite"],
        suite_path.to_string_lossy().to_string()
    );
    assert_eq!(payload["benchmark"]["executed_cases"], 2);
    assert_eq!(
        payload["benchmark"]["records"]
            .as_array()
            .map(|rows| rows.len()),
        Some(2)
    );
}

#[test]
fn benchmark_pack_import_list_show_and_profile_work() {
    let workspace = TempDir::new().expect("workspace");
    let source = workspace.path().join("pack-source.json");
    fs::write(
        &source,
        r#"{
  "cases": [
    {
      "case_id": "pack-case-1",
      "prompt": "Plan benchmark pack import verification.",
      "expected_keywords": ["verify"],
      "min_steps": 1,
      "min_verification_steps": 1
    }
  ]
}"#,
    )
    .expect("pack source");

    let imported = run_json(
        workspace.path(),
        &[
            "--json",
            "benchmark",
            "import-pack",
            "public-pack",
            source.to_string_lossy().as_ref(),
        ],
    );
    assert_eq!(imported["imported"], true);
    assert_eq!(imported["name"], "public-pack");

    let listed = run_json(workspace.path(), &["--json", "benchmark", "list-packs"]);
    assert!(
        listed
            .as_array()
            .is_some_and(|rows| rows.iter().any(|row| row["name"] == "public-pack"))
    );

    let shown = run_json(
        workspace.path(),
        &["--json", "benchmark", "show-pack", "public-pack"],
    );
    assert_eq!(shown["name"], "public-pack");
    assert_eq!(shown["cases"].as_array().map(|rows| rows.len()), Some(1));

    let from_pack = run_json(
        workspace.path(),
        &[
            "--json",
            "profile",
            "--benchmark",
            "--benchmark-pack",
            "public-pack",
            "--benchmark-cases",
            "1",
            "--benchmark-min-success-rate",
            "0.0",
        ],
    );
    assert_eq!(from_pack["benchmark"]["pack"], "public-pack");
    assert_eq!(from_pack["benchmark"]["executed_cases"], 1);
}

#[test]
fn benchmark_sync_public_catalog_imports_relative_sources() {
    let workspace = TempDir::new().expect("workspace");
    let catalog_dir = workspace.path().join(".deepseek/public-catalog");
    fs::create_dir_all(&catalog_dir).expect("catalog dir");

    let suite_a = catalog_dir.join("suite-a.json");
    fs::write(
        &suite_a,
        r#"{
  "cases": [
    {
      "case_id": "public-a",
      "prompt": "Plan public corpus A validation.",
      "expected_keywords": ["verify"],
      "min_steps": 1,
      "min_verification_steps": 1
    }
  ]
}"#,
    )
    .expect("suite a");
    let suite_b = catalog_dir.join("suite-b.jsonl");
    fs::write(
        &suite_b,
        r#"{"case_id":"public-b","prompt":"Plan public corpus B validation.","expected_keywords":["verify"],"min_steps":1,"min_verification_steps":1}"#,
    )
    .expect("suite b");

    let catalog = catalog_dir.join("catalog.json");
    fs::write(
        &catalog,
        r#"{
  "schema": "deepseek.benchmark.catalog.v1",
  "packs": [
    { "name": "public-a", "source": "suite-a.json", "kind": "public" },
    { "name": "public-b", "source": "suite-b.jsonl", "kind": "public" }
  ]
}"#,
    )
    .expect("catalog");

    let synced = run_json(
        workspace.path(),
        &[
            "--json",
            "benchmark",
            "sync-public",
            catalog.to_string_lossy().as_ref(),
            "--prefix",
            "ext-",
        ],
    );
    assert_eq!(synced["count"], 2);
    assert!(
        synced["imported"]
            .as_array()
            .is_some_and(|rows| rows.iter().any(|row| row["name"] == "ext-public-a"))
    );

    let listed = run_json(workspace.path(), &["--json", "benchmark", "list-packs"]);
    assert!(
        listed
            .as_array()
            .is_some_and(|rows| rows.iter().any(|row| row["name"] == "ext-public-b"))
    );
}

#[test]
fn benchmark_publish_parity_writes_latest_and_history() {
    let workspace = TempDir::new().expect("workspace");
    let matrix_path = workspace.path().join(".deepseek/publish-matrix.json");
    fs::create_dir_all(
        matrix_path
            .parent()
            .expect("matrix should have parent directory"),
    )
    .expect("matrix dir");
    fs::write(
        &matrix_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "name": "publish-smoke",
            "runs": [
                {"id": "smoke", "pack": "smoke", "cases": 1, "seed": 11}
            ]
        }))
        .expect("matrix serialize"),
    )
    .expect("matrix write");

    let output_dir = workspace.path().join(".deepseek/reports/parity");
    let payload = run_json(
        workspace.path(),
        &[
            "--json",
            "benchmark",
            "publish-parity",
            "--matrix",
            matrix_path.to_string_lossy().as_ref(),
            "--output-dir",
            output_dir.to_string_lossy().as_ref(),
        ],
    );
    assert_eq!(payload["published"], true);
    assert_eq!(
        payload["matrix_payload"]["schema"],
        "deepseek.benchmark.matrix.v1"
    );
    assert!(output_dir.join("latest.json").exists());
    assert!(output_dir.join("latest.md").exists());
    assert!(output_dir.join("history.jsonl").exists());
}

#[test]
fn benchmark_builtin_parity_pack_is_available() {
    let workspace = TempDir::new().expect("workspace");
    let listed = run_json(workspace.path(), &["--json", "benchmark", "list-packs"]);
    assert!(
        listed
            .as_array()
            .is_some_and(|rows| rows.iter().any(|row| row["name"] == "parity"))
    );
    let shown = run_json(
        workspace.path(),
        &["--json", "benchmark", "show-pack", "parity"],
    );
    assert_eq!(shown["name"], "parity");
    assert!(
        shown["cases"]
            .as_array()
            .is_some_and(|rows| rows.len() >= 12)
    );
}

#[test]
fn benchmark_scorecard_is_seeded_and_signed_when_key_present() {
    let workspace = TempDir::new().expect("workspace");
    let args = [
        "--json",
        "profile",
        "--benchmark",
        "--benchmark-cases",
        "2",
        "--benchmark-seed",
        "77",
        "--benchmark-min-success-rate",
        "0.0",
    ];

    let first = run_json_with_env(
        workspace.path(),
        &args,
        &[("DEEPSEEK_BENCHMARK_SIGNING_KEY", "test-signing-key")],
    );
    let benchmark = &first["benchmark"];
    assert_eq!(benchmark["seed"], 77);
    let manifest_sha = benchmark["corpus_manifest"]["manifest_sha256"]
        .as_str()
        .expect("manifest hash");
    assert_eq!(
        benchmark["scorecard"]["corpus_manifest_sha256"],
        manifest_sha
    );
    assert_eq!(benchmark["corpus_manifest"]["signature"]["present"], true);
    assert_eq!(
        benchmark["execution_manifest"]["signature"]["present"],
        true
    );
    assert_eq!(benchmark["scorecard"]["signature"]["present"], true);

    let second = run_json_with_env(
        workspace.path(),
        &args,
        &[("DEEPSEEK_BENCHMARK_SIGNING_KEY", "test-signing-key")],
    );
    assert_eq!(
        first["benchmark"]["case_ids"],
        second["benchmark"]["case_ids"]
    );
    assert_eq!(
        first["benchmark"]["execution_manifest"]["manifest_sha256"],
        second["benchmark"]["execution_manifest"]["manifest_sha256"]
    );
}

#[test]
fn benchmark_run_matrix_executes_multiple_runs_and_peer_compare() {
    let workspace = TempDir::new().expect("workspace");
    let suite_path = workspace.path().join(".deepseek/matrix-suite.json");
    fs::create_dir_all(
        suite_path
            .parent()
            .expect("suite path should have parent directory"),
    )
    .expect("suite dir");
    fs::write(
        &suite_path,
        r#"{
  "cases": [
    {
      "case_id": "matrix-suite-case",
      "prompt": "Plan matrix suite verification checks.",
      "expected_keywords": ["verify"],
      "min_steps": 1,
      "min_verification_steps": 1
    }
  ]
}"#,
    )
    .expect("suite");

    let matrix_path = workspace.path().join(".deepseek/benchmark-matrix.json");
    fs::write(
        &matrix_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "name": "parity-matrix",
            "runs": [
                {"id": "pack-smoke", "pack": "smoke", "cases": 1, "seed": 11},
                {"id": "suite-local", "suite": suite_path.to_string_lossy(), "cases": 1, "seed": 12}
            ]
        }))
        .expect("serialize matrix"),
    )
    .expect("matrix");

    let peer_path = workspace.path().join(".deepseek/peer-matrix.json");
    fs::write(
        &peer_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "agent": "codex",
            "summary": {
                "total_runs": 2,
                "total_cases": 2,
                "weighted_success_rate": 1.0,
                "weighted_quality_rate": 1.0,
                "worst_p95_latency_ms": 1200,
                "manifest_coverage": 1.0
            }
        }))
        .expect("serialize peer"),
    )
    .expect("peer");

    let output_path = workspace.path().join(".deepseek/matrix-output.json");
    let report_path = workspace.path().join(".deepseek/matrix-report.md");
    let payload = run_json_with_env(
        workspace.path(),
        &[
            "--json",
            "benchmark",
            "run-matrix",
            matrix_path.to_string_lossy().as_ref(),
            "--compare",
            peer_path.to_string_lossy().as_ref(),
            "--require-agent",
            "deepseek-cli,codex",
            "--report-output",
            report_path.to_string_lossy().as_ref(),
            "--output",
            output_path.to_string_lossy().as_ref(),
        ],
        &[("DEEPSEEK_BENCHMARK_SIGNING_KEY", "matrix-signing-key")],
    );

    assert_eq!(payload["schema"], "deepseek.benchmark.matrix.v1");
    assert_eq!(payload["summary"]["total_runs"], 2);
    assert_eq!(payload["summary"]["total_cases"], 2);
    assert!(payload["runs"].as_array().is_some_and(|runs| {
        runs.iter()
            .all(|run| run["benchmark"]["scorecard"].is_object())
    }));
    assert!(
        payload["peer_comparison"]["ranking"]
            .as_array()
            .is_some_and(|rows| rows.len() >= 2)
    );
    assert!(
        payload["report_markdown"]
            .as_str()
            .unwrap_or_default()
            .contains("Peer Ranking")
    );
    assert!(output_path.exists());
    assert!(report_path.exists());
    let report_text = fs::read_to_string(report_path).expect("report");
    assert!(report_text.contains("Benchmark Matrix Report"));
}

#[test]
fn profile_benchmark_compare_strict_fails_on_manifest_mismatch() {
    let workspace = TempDir::new().expect("workspace");
    fs::create_dir_all(workspace.path().join(".deepseek")).expect("runtime dir");
    fs::write(
        workspace.path().join(".deepseek/peer-no-manifest.json"),
        r#"{
  "agent": "codex",
  "benchmark": {
    "executed_cases": 1,
    "succeeded": 1,
    "p95_latency_ms": 900,
    "records": [{"case_id":"x","ok":true,"quality_ok":true}]
  }
}"#,
    )
    .expect("peer report");

    let mock = start_mock_llm_server();
    configure_runtime_for_mock_llm(workspace.path(), &mock.endpoint);

    let output = Command::new(assert_cmd::cargo::cargo_bin!("deepseek"))
        .current_dir(workspace.path())
        .env("DEEPSEEK_API_KEY", "test-api-key")
        .args([
            "--json",
            "profile",
            "--benchmark",
            "--benchmark-cases",
            "1",
            "--benchmark-compare",
            workspace
                .path()
                .join(".deepseek/peer-no-manifest.json")
                .to_string_lossy()
                .as_ref(),
            "--benchmark-compare-strict",
            "--benchmark-min-success-rate",
            "0.0",
        ])
        .assert()
        .failure()
        .get_output()
        .stderr
        .clone();
    let stderr = String::from_utf8_lossy(&output);
    assert!(stderr.contains("strict mode failed"));
}

#[test]
fn benchmark_run_matrix_strict_fails_on_mixed_corpus_compatibility() {
    let workspace = TempDir::new().expect("workspace");
    let suite_path = workspace.path().join(".deepseek/strict-suite.json");
    fs::create_dir_all(
        suite_path
            .parent()
            .expect("suite path should have parent directory"),
    )
    .expect("suite dir");
    fs::write(
        &suite_path,
        r#"{
  "cases": [
    {
      "case_id": "strict-suite-case",
      "prompt": "Plan strict matrix compatibility checks.",
      "expected_keywords": ["verify"],
      "min_steps": 1,
      "min_verification_steps": 1
    }
  ]
}"#,
    )
    .expect("suite");

    let matrix_path = workspace.path().join(".deepseek/strict-matrix.json");
    fs::write(
        &matrix_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "name": "strict-matrix",
            "runs": [
                {"id": "builtin-core", "pack": "core", "cases": 1, "seed": 2},
                {"id": "local-suite", "suite": suite_path.to_string_lossy(), "cases": 1, "seed": 3}
            ]
        }))
        .expect("serialize matrix"),
    )
    .expect("matrix");

    let mock = start_mock_llm_server();
    configure_runtime_for_mock_llm(workspace.path(), &mock.endpoint);

    let output = Command::new(assert_cmd::cargo::cargo_bin!("deepseek"))
        .current_dir(workspace.path())
        .env("DEEPSEEK_API_KEY", "test-api-key")
        .args([
            "--json",
            "benchmark",
            "run-matrix",
            matrix_path.to_string_lossy().as_ref(),
            "--strict",
        ])
        .assert()
        .failure()
        .get_output()
        .stderr
        .clone();
    let stderr = String::from_utf8_lossy(&output);
    assert!(stderr.contains("strict mode failed"));
}

#[test]
fn benchmark_run_matrix_fails_when_required_agent_missing() {
    let workspace = TempDir::new().expect("workspace");
    let matrix_path = workspace
        .path()
        .join(".deepseek/benchmark-matrix-required.json");
    fs::create_dir_all(
        matrix_path
            .parent()
            .expect("matrix file should have a parent directory"),
    )
    .expect("matrix dir");
    fs::write(
        &matrix_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "name": "required-agents",
            "runs": [
                {"id": "core", "pack": "core", "cases": 1, "seed": 1}
            ]
        }))
        .expect("serialize matrix"),
    )
    .expect("write matrix");

    let mock = start_mock_llm_server();
    configure_runtime_for_mock_llm(workspace.path(), &mock.endpoint);

    let output = Command::new(assert_cmd::cargo::cargo_bin!("deepseek"))
        .current_dir(workspace.path())
        .env("DEEPSEEK_API_KEY", "test-api-key")
        .args([
            "--json",
            "benchmark",
            "run-matrix",
            matrix_path.to_string_lossy().as_ref(),
            "--require-agent",
            "claude",
        ])
        .assert()
        .failure()
        .get_output()
        .stderr
        .clone();
    let stderr = String::from_utf8_lossy(&output);
    assert!(stderr.contains("missing required agents"));
}

fn run_json(workspace: &Path, args: &[&str]) -> Value {
    run_json_with_env(workspace, args, &[])
}

fn run_json_with_env(workspace: &Path, args: &[&str], envs: &[(&str, &str)]) -> Value {
    let mock = start_mock_llm_server();
    configure_runtime_for_mock_llm(workspace, &mock.endpoint);
    let mut command = Command::new(assert_cmd::cargo::cargo_bin!("deepseek"));
    command
        .current_dir(workspace)
        .env("DEEPSEEK_API_KEY", "test-api-key")
        .args(args);
    for (key, value) in envs {
        command.env(key, value);
    }
    let output = command.assert().success().get_output().stdout.clone();
    serde_json::from_slice(&output).expect("json output")
}

fn write_fake_png_with_dimensions(
    path: &Path,
    width: u32,
    height: u32,
) -> Result<(), std::io::Error> {
    let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    bytes.extend_from_slice(&[0, 0, 0, 13, b'I', b'H', b'D', b'R']);
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&[8, 2, 0, 0, 0, 0, 0, 0, 0]);
    fs::write(path, bytes)
}

struct MockLlmServer {
    endpoint: String,
    stop_tx: Option<mpsc::Sender<()>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Drop for MockLlmServer {
    fn drop(&mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn start_mock_llm_server() -> MockLlmServer {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock llm server");
    listener
        .set_nonblocking(true)
        .expect("set nonblocking listener");
    let addr = listener.local_addr().expect("mock addr");
    let (tx, rx) = mpsc::channel::<()>();
    let handle = thread::spawn(move || {
        loop {
            if rx.try_recv().is_ok() {
                break;
            }
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let _ = handle_mock_llm_connection(&mut stream);
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5));
                }
                Err(_) => break,
            }
        }
    });
    MockLlmServer {
        endpoint: format!("http://{addr}/chat/completions"),
        stop_tx: Some(tx),
        handle: Some(handle),
    }
}

fn configure_runtime_for_mock_llm(workspace: &Path, endpoint: &str) {
    let runtime = workspace.join(".deepseek");
    fs::create_dir_all(&runtime).expect("runtime dir");
    fs::write(
        runtime.join("settings.local.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "llm": {
                "provider": "deepseek",
                "endpoint": endpoint,
                "api_key_env": "DEEPSEEK_API_KEY"
            }
        }))
        .expect("serialize settings"),
    )
    .expect("settings override");
}

fn handle_mock_llm_connection(stream: &mut TcpStream) -> std::io::Result<()> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 1024];
    let mut header_end = None;
    while header_end.is_none() {
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        header_end = find_subsequence(&buffer, b"\r\n\r\n").map(|idx| idx + 4);
        if buffer.len() > 1_048_576 {
            break;
        }
    }

    let header_len = header_end.unwrap_or(buffer.len());
    let content_length = parse_content_length(&buffer[..header_len]);
    let mut body = if header_len <= buffer.len() {
        buffer[header_len..].to_vec()
    } else {
        Vec::new()
    };
    while body.len() < content_length {
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }

    let prompt =
        extract_prompt_from_request_body(&body).unwrap_or_else(|| "mock prompt".to_string());
    let content = if prompt.to_ascii_lowercase().contains("plan") {
        "Generated plan: discover files, propose edits, verify with tests.".to_string()
    } else {
        format!("Mock response: {prompt}")
    };
    let payload = serde_json::json!({
        "choices": [
            {
                "message": {
                    "content": content
                }
            }
        ]
    })
    .to_string();
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        payload.len(),
        payload
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn parse_content_length(headers: &[u8]) -> usize {
    let raw = String::from_utf8_lossy(headers);
    for line in raw.lines() {
        let mut parts = line.splitn(2, ':');
        let key = parts.next().unwrap_or_default().trim();
        if key.eq_ignore_ascii_case("content-length")
            && let Some(value) = parts.next()
            && let Ok(parsed) = value.trim().parse::<usize>()
        {
            return parsed;
        }
    }
    0
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn extract_prompt_from_request_body(body: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    value
        .get("messages")
        .and_then(|v| v.as_array())
        .and_then(|rows| rows.last())
        .and_then(|row| row.get("content"))
        .and_then(|v| v.as_str())
        .map(ToString::to_string)
}
