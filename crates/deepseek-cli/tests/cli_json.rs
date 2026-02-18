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
fn ask_json_fails_fast_without_api_key_when_offline_disabled() {
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

fn run_json(workspace: &Path, args: &[&str]) -> Value {
    run_json_with_env(workspace, args, &[])
}

fn run_json_with_env(workspace: &Path, args: &[&str], envs: &[(&str, &str)]) -> Value {
    let runtime = workspace.join(".deepseek");
    fs::create_dir_all(&runtime).expect("runtime dir");
    fs::write(
        runtime.join("settings.local.json"),
        r#"{"llm":{"offline_fallback":true}}"#,
    )
    .expect("settings override");
    let mut command = Command::new(assert_cmd::cargo::cargo_bin!("deepseek"));
    command.current_dir(workspace).args(args);
    for (key, value) in envs {
        command.env(key, value);
    }
    let output = command.assert().success().get_output().stdout.clone();
    serde_json::from_slice(&output).expect("json output")
}
