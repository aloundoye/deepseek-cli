use assert_cmd::Command;
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

#[test]
fn plan_command_emits_json_shape() {
    let workspace = TempDir::new().expect("workspace");
    let out = run_json(
        workspace.path(),
        &["--json", "plan", "Refactor planner state"],
    );
    // The plan command now routes through chat_with_options and wraps
    // the LLM response text in a {"plan": "..."} envelope.
    assert!(
        out.get("plan").is_some(),
        "plan command JSON should contain 'plan' key, got: {out}"
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
        &[
            "--json",
            "autopilot",
            "Create baseline session for status checks",
            "--max-iterations",
            "1",
            "--duration-seconds",
            "30",
            "--tools",
            "false",
        ],
    );

    let status = run_json(workspace.path(), &["--json", "status"]);
    assert!(status.get("session_id").is_some());
    assert!(status["model"]["base"].as_str().is_some());

    let usage = run_json(workspace.path(), &["--json", "usage", "--session"]);
    assert!(usage["records"].as_u64().is_some());

    let compact = run_json(workspace.path(), &["--json", "compact", "--yes"]);
    if compact["persisted"].as_bool().unwrap_or(false) {
        assert!(compact["result"]["summary_id"].as_str().is_some());
    } else {
        assert_eq!(compact["status"], "no_op");
    }

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
fn mcp_memory_and_export_emit_json() {
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
    let content = format!("Mock response: {prompt}");
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
