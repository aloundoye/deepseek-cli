use super::*;
use codingbuddy_core::{
    AppConfig, ApprovedToolCall, Session, SessionBudgets, SessionState, ToolCall, ToolHost,
    runtime_dir,
};
use serde_json::json;
use std::sync::{Arc, Mutex};

fn temp_host() -> (PathBuf, LocalToolHost) {
    let workspace = std::env::temp_dir().join(format!("codingbuddy-tools-test-{}", Uuid::now_v7()));
    fs::create_dir_all(&workspace).expect("workspace");
    let host = LocalToolHost::new(&workspace, PolicyEngine::default()).expect("tool host");
    (workspace, host)
}

#[derive(Clone, Default)]
struct RecordingRunner {
    commands: Arc<Mutex<Vec<String>>>,
}

impl RecordingRunner {
    fn captured(&self) -> Vec<String> {
        self.commands.lock().expect("commands").clone()
    }
}

impl ShellRunner for RecordingRunner {
    fn run(&self, cmd: &str, _cwd: &Path, _timeout: Duration) -> Result<ShellRunResult> {
        self.commands
            .lock()
            .expect("commands")
            .push(cmd.to_string());
        Ok(ShellRunResult {
            status: Some(0),
            stdout: "ok".to_string(),
            stderr: String::new(),
            timed_out: false,
        })
    }
}

#[cfg(not(target_os = "windows"))]
#[test]
fn hooks_receive_phase_and_tool_context() {
    let workspace =
        std::env::temp_dir().join(format!("codingbuddy-tools-hook-test-{}", Uuid::now_v7()));
    fs::create_dir_all(&workspace).expect("workspace");

    let mut cfg = AppConfig::default();
    cfg.plugins.enable_hooks = true;
    cfg.save(&workspace).expect("save config");

    let plugin_src = workspace.join("plugin-src");
    fs::create_dir_all(plugin_src.join(".codingbuddy-plugin")).expect("plugin dir");
    fs::create_dir_all(plugin_src.join("hooks")).expect("hooks dir");
    fs::write(
        plugin_src.join(".codingbuddy-plugin/plugin.json"),
        r#"{"id":"hookdemo","name":"Hook Demo","version":"0.1.0"}"#,
    )
    .expect("manifest");
    fs::write(
            plugin_src.join("hooks/pretooluse.sh"),
            "#!/bin/sh\nprintf \"%s|%s\" \"$CODINGBUDDY_HOOK_PHASE\" \"$CODINGBUDDY_TOOL_NAME\" > \"$CODINGBUDDY_WORKSPACE/hook.out\"\n",
        )
        .expect("hook script");

    let manager = PluginManager::new(&workspace).expect("plugin manager");
    manager.install(&plugin_src).expect("install plugin");

    let host = LocalToolHost::new(&workspace, PolicyEngine::default()).expect("tool host");
    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "fs.list".to_string(),
            args: json!({"dir":"."}),
            requires_approval: false,
        },
    });
    assert!(result.success);

    let hook_out = fs::read_to_string(workspace.join("hook.out")).expect("hook output");
    assert!(hook_out.contains("pretooluse|fs.list"));
}

#[test]
fn plugin_tool_definitions_include_installed_commands() {
    let workspace = std::env::temp_dir().join(format!(
        "codingbuddy-tools-plugin-defs-test-{}",
        Uuid::now_v7()
    ));
    fs::create_dir_all(&workspace).expect("workspace");

    let plugin_src = workspace.join("plugin-src");
    fs::create_dir_all(plugin_src.join(".codingbuddy-plugin")).expect("plugin dir");
    fs::create_dir_all(plugin_src.join("commands/security")).expect("commands dir");
    fs::write(
        plugin_src.join(".codingbuddy-plugin/plugin.json"),
        r#"{"id":"phase-c-plugin","name":"Phase C Plugin","version":"0.1.0"}"#,
    )
    .expect("manifest");
    fs::write(plugin_src.join("commands/security/review.md"), "# review").expect("command");

    let manager = PluginManager::new(&workspace).expect("plugin manager");
    manager.install(&plugin_src).expect("install plugin");

    let defs = plugin_tool_definitions(&workspace);
    assert!(
        defs.iter()
            .any(|def| def.function.name == "plugin__phase_c_plugin__security_review"),
        "expected nested plugin command in generated tool definitions"
    );
}

#[test]
fn plugin_tool_renders_prompt() {
    let workspace = std::env::temp_dir().join(format!(
        "codingbuddy-tools-plugin-runtime-test-{}",
        Uuid::now_v7()
    ));
    fs::create_dir_all(&workspace).expect("workspace");

    let plugin_src = workspace.join("plugin-src");
    fs::create_dir_all(plugin_src.join(".codingbuddy-plugin")).expect("plugin dir");
    fs::create_dir_all(plugin_src.join("commands")).expect("commands dir");
    fs::write(
        plugin_src.join(".codingbuddy-plugin/plugin.json"),
        r#"{"id":"phase-c-plugin","name":"Phase C Plugin","version":"0.1.0"}"#,
    )
    .expect("manifest");
    fs::write(
        plugin_src.join("commands/review.md"),
        "Plugin {{plugin_id}} / {{command_name}}\nInput={{input}}",
    )
    .expect("command");

    let manager = PluginManager::new(&workspace).expect("plugin manager");
    manager.install(&plugin_src).expect("install plugin");

    let host = LocalToolHost::new(&workspace, PolicyEngine::default()).expect("tool host");
    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "plugin__phase_c_plugin__review".to_string(),
            args: json!({"arguments":"focus auth module"}),
            requires_approval: false,
        },
    });
    assert!(result.success, "plugin tool execution should succeed");
    assert_eq!(result.output["plugin_id"], "phase-c-plugin");
    assert_eq!(result.output["command_name"], "review");
    assert!(
        result.output["prompt"]
            .as_str()
            .is_some_and(|p| p.contains("focus auth module")),
        "rendered prompt should include provided input"
    );
}

#[test]
fn fs_read_supports_line_ranges_and_mime_metadata() {
    let (workspace, host) = temp_host();
    fs::write(workspace.join("note.txt"), "a\nb\nc\n").expect("seed");

    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "fs.read".to_string(),
            args: json!({"path":"note.txt","start_line":2,"end_line":3}),
            requires_approval: false,
        },
    });
    assert!(result.success);
    assert_eq!(result.output["mime"], "text/plain");
    assert_eq!(result.output["binary"], false);
    let lines = result.output["lines"].as_array().expect("lines");
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["line"], 2);
    assert_eq!(lines[0]["text"], "b");
}

#[test]
fn fs_glob_grep_and_edit_work() {
    let (workspace, host) = temp_host();
    fs::create_dir_all(workspace.join("src")).expect("src");
    fs::write(workspace.join("src/main.rs"), "fn old_name() {}\n").expect("seed");
    fs::write(workspace.join("src/lib.rs"), "pub fn helper() {}\n").expect("seed");

    let globbed = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "fs.glob".to_string(),
            args: json!({"pattern":"src/*.rs"}),
            requires_approval: false,
        },
    });
    assert!(globbed.success);
    assert!(
        globbed.output["matches"]
            .as_array()
            .is_some_and(|items| items.len() >= 2)
    );

    let grepped = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "fs.grep".to_string(),
            args: json!({"pattern":"old_name","glob":"src/*.rs"}),
            requires_approval: false,
        },
    });
    assert!(grepped.success);
    assert_eq!(
        grepped.output["matches"].as_array().expect("matches").len(),
        1
    );

    let edited = host.execute(ApprovedToolCall {
            invocation_id: Uuid::now_v7(),
            call: ToolCall {
                name: "fs.edit".to_string(),
                args: json!({"path":"src/main.rs","search":"old_name","replace":"new_name","all":false}),
                requires_approval: false,
            },
        });
    assert!(edited.success);
    assert_eq!(edited.output["edited"], true);
    let content = fs::read_to_string(workspace.join("src/main.rs")).expect("updated");
    assert!(content.contains("new_name"));
}

#[test]
fn fs_edit_includes_unified_diff_in_result() {
    let (workspace, host) = temp_host();
    fs::write(workspace.join("demo.rs"), "fn old() {}\n").expect("seed");

    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "fs.edit".to_string(),
            args: json!({"path":"demo.rs","search":"old","replace":"new","all":false}),
            requires_approval: false,
        },
    });
    assert!(result.success);
    assert_eq!(result.output["edited"], true);
    let diff = result.output["diff"].as_str().expect("diff field");
    assert!(diff.contains("--- a/demo.rs"));
    assert!(diff.contains("+++ b/demo.rs"));
    assert!(diff.contains("-fn old() {}"));
    assert!(diff.contains("+fn new() {}"));
}

#[test]
fn fs_glob_respects_gitignore_rules() {
    let (workspace, host) = temp_host();
    fs::create_dir_all(workspace.join("ignored")).expect("ignored dir");
    fs::create_dir_all(workspace.join("src")).expect("src");
    fs::write(workspace.join(".gitignore"), "ignored/\n").expect("gitignore");
    fs::write(workspace.join("ignored/secret.txt"), "secret\n").expect("secret");
    fs::write(workspace.join("src/main.rs"), "fn main() {}\n").expect("main");

    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "fs.glob".to_string(),
            args: json!({"pattern":"**/*","respectGitignore":true}),
            requires_approval: false,
        },
    });
    assert!(result.success);
    let paths = result
        .output
        .get("matches")
        .and_then(|items| items.as_array())
        .expect("matches")
        .iter()
        .filter_map(|item| item.get("path").and_then(|value| value.as_str()))
        .collect::<Vec<_>>();
    assert!(paths.iter().any(|path| path.ends_with("src/main.rs")));
    assert!(
        !paths
            .iter()
            .any(|path| path.ends_with("ignored/secret.txt"))
    );
}

#[test]
fn fs_read_emits_visual_artifact_event_when_enabled() {
    let workspace = std::env::temp_dir().join(format!("codingbuddy-tools-vis-{}", Uuid::now_v7()));
    fs::create_dir_all(&workspace).expect("workspace");

    let mut cfg = AppConfig::default();
    cfg.experiments.visual_verification = true;
    cfg.save(&workspace).expect("save config");

    let store = Store::new(&workspace).expect("store");
    store
        .save_session(&Session {
            session_id: Uuid::now_v7(),
            workspace_root: workspace.to_string_lossy().to_string(),
            baseline_commit: None,
            status: SessionState::Idle,
            budgets: SessionBudgets {
                per_turn_seconds: 30,
                max_think_tokens: 1024,
            },
            active_plan_id: None,
        })
        .expect("session");

    fs::write(workspace.join("image.png"), [0x89, b'P', b'N', b'G', 0, 1]).expect("image");
    let host = LocalToolHost::new(&workspace, PolicyEngine::default()).expect("host");
    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "fs.read".to_string(),
            args: json!({"path":"image.png"}),
            requires_approval: false,
        },
    });
    assert!(result.success);
    assert_eq!(result.output["binary"], true);

    let events_path = runtime_dir(&workspace).join("events.jsonl");
    let events = fs::read_to_string(events_path).expect("events");
    assert!(events.contains("VisualArtifactCaptured"));
}

#[test]
fn read_only_sandbox_blocks_mutating_bash_commands() {
    let workspace = std::env::temp_dir().join(format!("codingbuddy-tools-ro-{}", Uuid::now_v7()));
    fs::create_dir_all(&workspace).expect("workspace");
    let mut cfg = AppConfig::default();
    cfg.policy.allowlist = vec!["touch *".to_string()];
    cfg.policy.sandbox_mode = codingbuddy_core::SandboxMode::ReadOnly;
    cfg.save(&workspace).expect("save config");
    let policy = PolicyEngine::from_app_config(&cfg.policy);
    let runner = RecordingRunner::default();
    let host = LocalToolHost::with_runner(&workspace, policy, Arc::new(runner.clone()))
        .expect("tool host");
    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd":"touch note.txt"}),
            requires_approval: false,
        },
    });
    assert!(!result.success);
    assert!(
        result.output["error"]
            .as_str()
            .unwrap_or_default()
            .contains("sandbox_mode=read-only")
    );
    assert!(runner.captured().is_empty());
}

#[test]
fn workspace_write_sandbox_blocks_absolute_outside_paths() {
    let workspace = std::env::temp_dir().join(format!("codingbuddy-tools-ww-{}", Uuid::now_v7()));
    fs::create_dir_all(&workspace).expect("workspace");
    let outside = std::env::temp_dir().join(format!("codingbuddy-outside-{}.txt", Uuid::now_v7()));
    fs::write(&outside, "outside").expect("outside file");

    let mut cfg = AppConfig::default();
    cfg.policy.allowlist = vec!["cat *".to_string()];
    cfg.policy.sandbox_mode = codingbuddy_core::SandboxMode::WorkspaceWrite;
    cfg.save(&workspace).expect("save config");
    let policy = PolicyEngine::from_app_config(&cfg.policy);
    let runner = RecordingRunner::default();
    let host = LocalToolHost::with_runner(&workspace, policy, Arc::new(runner.clone()))
        .expect("tool host");
    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd": format!("cat {}", outside.display())}),
            requires_approval: false,
        },
    });
    assert!(!result.success);
    assert!(
        result.output["error"]
            .as_str()
            .unwrap_or_default()
            .contains("sandbox_mode=workspace-write")
    );
    assert!(runner.captured().is_empty());
}

#[test]
fn workspace_write_sandbox_blocks_windows_parent_path_tokens() {
    let workspace =
        std::env::temp_dir().join(format!("codingbuddy-tools-ww-win-path-{}", Uuid::now_v7()));
    fs::create_dir_all(&workspace).expect("workspace");

    let mut cfg = AppConfig::default();
    cfg.policy.allowlist = vec!["type *".to_string()];
    cfg.policy.sandbox_mode = codingbuddy_core::SandboxMode::WorkspaceWrite;
    cfg.save(&workspace).expect("save config");
    let policy = PolicyEngine::from_app_config(&cfg.policy);
    let runner = RecordingRunner::default();
    let host = LocalToolHost::with_runner(&workspace, policy, Arc::new(runner.clone()))
        .expect("tool host");
    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd":"type ..\\secret.txt"}),
            requires_approval: false,
        },
    });
    assert!(!result.success);
    assert!(
        result.output["error"]
            .as_str()
            .unwrap_or_default()
            .contains("sandbox_mode=workspace-write")
    );
    assert!(runner.captured().is_empty());
}

#[test]
fn workspace_write_sandbox_allows_workspace_relative_paths() {
    let workspace =
        std::env::temp_dir().join(format!("codingbuddy-tools-ww-allow-{}", Uuid::now_v7()));
    fs::create_dir_all(&workspace).expect("workspace");
    fs::write(workspace.join("note.txt"), "hello").expect("note");

    let mut cfg = AppConfig::default();
    cfg.policy.allowlist = vec!["cat *".to_string()];
    cfg.policy.sandbox_mode = codingbuddy_core::SandboxMode::WorkspaceWrite;
    cfg.save(&workspace).expect("save config");
    let policy = PolicyEngine::from_app_config(&cfg.policy);
    let runner = RecordingRunner::default();
    let host = LocalToolHost::with_runner(&workspace, policy, Arc::new(runner.clone()))
        .expect("tool host");
    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd":"cat note.txt"}),
            requires_approval: false,
        },
    });
    assert!(result.success);
    assert_eq!(runner.captured(), vec!["cat note.txt".to_string()]);
}

#[test]
fn read_only_sandbox_blocks_network_commands() {
    let workspace =
        std::env::temp_dir().join(format!("codingbuddy-tools-ro-net-{}", Uuid::now_v7()));
    fs::create_dir_all(&workspace).expect("workspace");
    let mut cfg = AppConfig::default();
    cfg.policy.allowlist = vec!["curl *".to_string()];
    cfg.policy.sandbox_mode = codingbuddy_core::SandboxMode::ReadOnly;
    cfg.save(&workspace).expect("save config");
    let policy = PolicyEngine::from_app_config(&cfg.policy);
    let runner = RecordingRunner::default();
    let host = LocalToolHost::with_runner(&workspace, policy, Arc::new(runner.clone()))
        .expect("tool host");
    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd":"curl https://example.com"}),
            requires_approval: false,
        },
    });
    assert!(!result.success);
    assert!(
        result.output["error"]
            .as_str()
            .unwrap_or_default()
            .contains("blocked network command")
    );
    assert!(runner.captured().is_empty());
}

#[test]
fn isolated_sandbox_uses_configured_wrapper_template() {
    let workspace =
        std::env::temp_dir().join(format!("codingbuddy-tools-iso-wrap-{}", Uuid::now_v7()));
    fs::create_dir_all(&workspace).expect("workspace");
    fs::write(workspace.join("note.txt"), "hello").expect("note");

    let mut cfg = AppConfig::default();
    cfg.policy.allowlist = vec!["cat *".to_string()];
    cfg.policy.sandbox_mode = codingbuddy_core::SandboxMode::Isolated;
    cfg.policy.sandbox_wrapper = Some("sandboxctl --workspace {workspace} --cmd {cmd}".to_string());
    cfg.save(&workspace).expect("save config");
    let policy = PolicyEngine::from_app_config(&cfg.policy);
    let runner = RecordingRunner::default();
    let host = LocalToolHost::with_runner(&workspace, policy, Arc::new(runner.clone()))
        .expect("tool host");
    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd":"cat note.txt"}),
            requires_approval: false,
        },
    });
    assert!(result.success);
    let normalized_workspace =
        std::fs::canonicalize(&workspace).unwrap_or_else(|_| workspace.clone());
    let expected = render_wrapper_template(
        "sandboxctl --workspace {workspace} --cmd {cmd}",
        &normalized_workspace,
        "cat note.txt",
    )
    .expect("render");
    assert_eq!(runner.captured(), vec![expected]);
}

#[cfg(target_os = "windows")]
#[test]
fn isolated_sandbox_windows_falls_back_with_logical_checks() {
    let workspace =
        std::env::temp_dir().join(format!("codingbuddy-tools-iso-win-{}", Uuid::now_v7()));
    fs::create_dir_all(&workspace).expect("workspace");
    fs::write(workspace.join("note.txt"), "hello").expect("note");

    let mut cfg = AppConfig::default();
    cfg.policy.allowlist = vec![
        "type *".to_string(),
        "curl *".to_string(),
        "cat *".to_string(),
    ];
    cfg.policy.sandbox_mode = codingbuddy_core::SandboxMode::Isolated;
    cfg.save(&workspace).expect("save config");
    let policy = PolicyEngine::from_app_config(&cfg.policy);
    let runner = RecordingRunner::default();
    let host = LocalToolHost::with_runner(&workspace, policy, Arc::new(runner.clone()))
        .expect("tool host");

    let allowed = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd":"type note.txt"}),
            requires_approval: false,
        },
    });
    assert!(allowed.success);

    let blocked_path = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd":"type ..\\secret.txt"}),
            requires_approval: false,
        },
    });
    assert!(!blocked_path.success);
    assert!(
        blocked_path.output["error"]
            .as_str()
            .unwrap_or_default()
            .contains("blocked path outside workspace")
    );

    let blocked_network = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd":"curl https://example.com"}),
            requires_approval: false,
        },
    });
    assert!(!blocked_network.success);
    assert!(
        blocked_network.output["error"]
            .as_str()
            .unwrap_or_default()
            .contains("blocked network command")
    );
    assert_eq!(runner.captured(), vec!["type note.txt".to_string()]);
}

#[test]
fn isolated_sandbox_requires_cmd_placeholder_in_wrapper_template() {
    let workspace =
        std::env::temp_dir().join(format!("codingbuddy-tools-iso-bad-wrap-{}", Uuid::now_v7()));
    fs::create_dir_all(&workspace).expect("workspace");

    let mut cfg = AppConfig::default();
    cfg.policy.allowlist = vec!["cat *".to_string()];
    cfg.policy.sandbox_mode = codingbuddy_core::SandboxMode::Isolated;
    cfg.policy.sandbox_wrapper = Some("sandboxctl --workspace {workspace}".to_string());
    cfg.save(&workspace).expect("save config");
    let policy = PolicyEngine::from_app_config(&cfg.policy);
    let runner = RecordingRunner::default();
    let host = LocalToolHost::with_runner(&workspace, policy, Arc::new(runner.clone()))
        .expect("tool host");
    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd":"cat note.txt"}),
            requires_approval: false,
        },
    });
    assert!(!result.success);
    assert!(
        result.output["error"]
            .as_str()
            .unwrap_or_default()
            .contains("must include {cmd}")
    );
    assert!(runner.captured().is_empty());
}

#[test]
fn multi_edit_modifies_multiple_files() {
    let (workspace, host) = temp_host();
    fs::write(workspace.join("a.txt"), "hello world\n").expect("seed a");
    fs::write(workspace.join("b.txt"), "foo bar\n").expect("seed b");

    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "multi_edit".to_string(),
            args: json!({
                "files": [
                    {"path": "a.txt", "edits": [{"search": "hello", "replace": "hi"}]},
                    {"path": "b.txt", "edits": [{"search": "foo", "replace": "baz"}]}
                ]
            }),
            requires_approval: false,
        },
    });
    assert!(result.success);
    assert_eq!(result.output["total_files"], 2);
    assert!(result.output["total_replacements"].as_u64().unwrap() >= 2);
    assert_eq!(
        fs::read_to_string(workspace.join("a.txt")).expect("a"),
        "hi world\n"
    );
    assert_eq!(
        fs::read_to_string(workspace.join("b.txt")).expect("b"),
        "baz bar\n"
    );
}

#[test]
fn multi_edit_returns_diffs_and_shas() {
    let (workspace, host) = temp_host();
    fs::write(workspace.join("c.txt"), "old value\n").expect("seed c");

    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "multi_edit".to_string(),
            args: json!({
                "files": [
                    {"path": "c.txt", "edits": [{"search": "old", "replace": "new"}]}
                ]
            }),
            requires_approval: false,
        },
    });
    assert!(result.success);
    let results = result.output["results"].as_array().expect("results");
    assert_eq!(results.len(), 1);
    let entry = &results[0];
    assert_eq!(entry["edited"], true);
    assert!(entry["diff"].as_str().unwrap().contains("--- a/c.txt"));
    assert!(entry["before_sha256"].as_str().is_some());
    assert!(entry["after_sha256"].as_str().is_some());
}

#[test]
fn multi_edit_blocked_in_review_mode() {
    let (workspace, mut host) = temp_host();
    host.set_review_mode(true);
    fs::write(workspace.join("d.txt"), "data\n").expect("seed d");

    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "multi_edit".to_string(),
            args: json!({
                "files": [
                    {"path": "d.txt", "edits": [{"search": "data", "replace": "new"}]}
                ]
            }),
            requires_approval: false,
        },
    });
    assert!(!result.success);
    assert!(
        result.output["error"]
            .as_str()
            .unwrap_or_default()
            .contains("review mode")
    );
}

#[test]
fn multi_edit_skips_unmodified_files() {
    let (workspace, host) = temp_host();
    fs::write(workspace.join("e.txt"), "keep me\n").expect("seed e");
    fs::write(workspace.join("f.txt"), "change me\n").expect("seed f");

    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "multi_edit".to_string(),
            args: json!({
                "files": [
                    {"path": "e.txt", "edits": [{"search": "missing", "replace": "gone"}]},
                    {"path": "f.txt", "edits": [{"search": "change", "replace": "changed"}]}
                ]
            }),
            requires_approval: false,
        },
    });
    assert!(result.success);
    let results = result.output["results"].as_array().expect("results");
    // e.txt had an error (search not found), f.txt was edited
    let f_entry = results
        .iter()
        .find(|r| r["path"] == "f.txt")
        .expect("f.txt");
    assert_eq!(f_entry["edited"], true);
    assert_eq!(
        fs::read_to_string(workspace.join("e.txt")).expect("e"),
        "keep me\n"
    );
}

#[test]
fn diagnostics_check_detects_rust_project() {
    let workspace = std::env::temp_dir().join(format!("codingbuddy-tools-diag-{}", Uuid::now_v7()));
    fs::create_dir_all(&workspace).expect("workspace");
    fs::write(workspace.join("Cargo.toml"), "[package]\nname = \"test\"\n").expect("cargo");

    let (cmd, source) = detect_diagnostics_command(&workspace, None).expect("detect");
    assert_eq!(source, "rustc");
    assert!(cmd.contains("cargo check"));
}

#[test]
fn diagnostics_check_is_read_only() {
    let policy = PolicyEngine::new(codingbuddy_policy::PolicyConfig {
        permission_mode: codingbuddy_policy::PermissionMode::Plan,
        ..codingbuddy_policy::PolicyConfig::default()
    });
    let call = ToolCall {
        name: "diagnostics.check".to_string(),
        args: json!({}),
        requires_approval: false,
    };
    assert!(!policy.requires_approval(&call));
}

#[test]
fn parse_cargo_check_json_extracts_errors() {
    let output = r#"{"reason":"compiler-message","message":{"level":"error","message":"unused variable","spans":[{"file_name":"src/main.rs","line_start":10,"column_start":5}]}}"#;
    let diagnostics = parse_cargo_check_json(output);
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0]["level"], "error");
    assert_eq!(diagnostics[0]["file"], "src/main.rs");
    assert_eq!(diagnostics[0]["line"], 10);
}

#[test]
fn parse_tsc_output_extracts_errors() {
    let output = "src/app.ts(42,13): error TS2304: Cannot find name 'foo'.";
    let diagnostics = parse_tsc_output(output);
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0]["level"], "error");
    assert_eq!(diagnostics[0]["file"], "src/app.ts");
    assert_eq!(diagnostics[0]["line"], 42);
    assert_eq!(diagnostics[0]["column"], 13);
    assert_eq!(diagnostics[0]["code"], "TS2304");
}

#[test]
fn seatbelt_profile_allows_workspace() {
    let workspace = std::path::Path::new("/tmp/test-project");
    let config = codingbuddy_core::SandboxConfig {
        enabled: true,
        ..Default::default()
    };
    let profile = super::build_seatbelt_profile(workspace, &config);
    assert!(profile.contains("/tmp/test-project"));
    assert!(profile.contains("(deny default)"));
    assert!(profile.contains("(allow process*)"));
}

#[test]
fn seatbelt_profile_blocks_network() {
    let workspace = std::path::Path::new("/tmp/test");
    let config = codingbuddy_core::SandboxConfig {
        enabled: true,
        network: codingbuddy_core::SandboxNetworkConfig {
            block_all: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let profile = super::build_seatbelt_profile(workspace, &config);
    assert!(profile.contains("(deny network*)"));
}

#[test]
fn bwrap_command_includes_workspace() {
    let workspace = std::path::Path::new("/home/user/project");
    let config = codingbuddy_core::SandboxConfig {
        enabled: true,
        ..Default::default()
    };
    let result = super::build_bwrap_command(workspace, "ls -la", &config);
    assert!(result.contains("bwrap"));
    assert!(result.contains("/home/user/project"));
    assert!(result.contains("ls -la"));
}

#[test]
fn bwrap_unshares_network_when_blocked() {
    let workspace = std::path::Path::new("/tmp/test");
    let config = codingbuddy_core::SandboxConfig {
        enabled: true,
        network: codingbuddy_core::SandboxNetworkConfig {
            block_all: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = super::build_bwrap_command(workspace, "echo hi", &config);
    assert!(result.contains("--unshare-net"));
}

#[test]
fn sandbox_disabled_passes_through() {
    let workspace = std::path::Path::new("/tmp/test");
    let config = codingbuddy_core::SandboxConfig {
        enabled: false,
        ..Default::default()
    };
    let result = super::sandbox_wrap_command(workspace, "echo hello", &config);
    assert_eq!(result, "echo hello");
}

#[test]
fn sandbox_excluded_command_passes_through() {
    let workspace = std::path::Path::new("/tmp/test");
    let config = codingbuddy_core::SandboxConfig {
        enabled: true,
        excluded_commands: vec!["cargo".to_string()],
        ..Default::default()
    };
    let result = super::sandbox_wrap_command(workspace, "cargo test", &config);
    assert_eq!(result, "cargo test");
}

#[test]
fn seatbelt_wrap_formats_correctly() {
    let profile = "(version 1)\n(deny default)";
    let result = super::seatbelt_wrap("echo test", profile);
    assert!(result.starts_with("sandbox-exec"));
    assert!(result.contains("echo test"));
}

#[test]
fn bwrap_includes_proc_and_dev() {
    let workspace = std::path::Path::new("/tmp/test");
    let config = codingbuddy_core::SandboxConfig::default();
    let result = super::build_bwrap_command(workspace, "ls", &config);
    assert!(result.contains("--proc"));
    assert!(result.contains("--dev"));
}

#[test]
fn seatbelt_allows_local_binding_when_configured() {
    let workspace = std::path::Path::new("/tmp/test");
    let config = codingbuddy_core::SandboxConfig {
        enabled: true,
        network: codingbuddy_core::SandboxNetworkConfig {
            block_all: true,
            allow_local_binding: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let profile = super::build_seatbelt_profile(workspace, &config);
    assert!(profile.contains("(deny network*)"));
    assert!(profile.contains("localhost"));
}

#[test]
fn bwrap_preserves_net_when_local_binding_allowed() {
    let workspace = std::path::Path::new("/tmp/test");
    let config = codingbuddy_core::SandboxConfig {
        enabled: true,
        network: codingbuddy_core::SandboxNetworkConfig {
            block_all: true,
            allow_local_binding: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = super::build_bwrap_command(workspace, "node server.js", &config);
    // When local binding is allowed, we can't fully unshare network
    assert!(!result.contains("--unshare-net"));
}

#[test]
fn chrome_tool_calls_fail_without_live_endpoint_by_default() {
    let (_workspace, host) = temp_host();
    let navigate = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "chrome.navigate".to_string(),
            args: json!({"url":"https://example.com"}),
            requires_approval: false,
        },
    });
    assert!(!navigate.success);
    assert!(navigate.output["error"].as_str().is_some());
    assert!(navigate.output["error_kind"].as_str().is_some());
    assert!(navigate.output["hints"].is_array());
}

#[test]
fn chrome_tool_calls_allow_stub_with_config_opt_in() {
    let workspace =
        std::env::temp_dir().join(format!("codingbuddy-tools-chrome-test-{}", Uuid::now_v7()));
    fs::create_dir_all(&workspace).expect("workspace");
    let mut cfg = AppConfig::default();
    cfg.tools.chrome.allow_stub_fallback = true;
    cfg.save(&workspace).expect("save config");
    let host = LocalToolHost::new(&workspace, PolicyEngine::default()).expect("tool host");

    let navigate = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "chrome.navigate".to_string(),
            args: json!({"url":"https://example.com"}),
            requires_approval: false,
        },
    });
    assert!(navigate.success);
    assert_eq!(navigate.output["ok"], true);
    assert_eq!(navigate.output["url"], "https://example.com");

    let evaluate = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "chrome.evaluate".to_string(),
            args: json!({"expression":"1 + 1"}),
            requires_approval: false,
        },
    });
    assert!(evaluate.success);
    assert!(evaluate.output["value"].is_object() || evaluate.output["value"].is_number());

    let screenshot = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "chrome.screenshot".to_string(),
            args: json!({"format":"png"}),
            requires_approval: false,
        },
    });
    assert!(screenshot.success);
    assert!(
        screenshot.output["base64"]
            .as_str()
            .is_some_and(|data| !data.is_empty())
    );
}

#[test]
fn tool_definitions_include_all_tools() {
    let defs = tool_definitions();
    let names: Vec<&str> = defs.iter().map(|d| d.function.name.as_str()).collect();

    // Core tools
    for expected in [
        "fs_read",
        "fs_write",
        "fs_edit",
        "fs_list",
        "fs_glob",
        "fs_grep",
        "bash_run",
        "multi_edit",
        "git_status",
        "git_diff",
        "git_show",
        "web_fetch",
        "web_search",
        "notebook_read",
        "notebook_edit",
        "index_query",
        "patch_stage",
        "patch_apply",
        "patch_direct",
        "diagnostics_check",
        "batch",
    ] {
        assert!(
            names.contains(&expected),
            "missing tool definition: {expected}"
        );
    }
    // Chrome tools
    for expected in [
        "chrome_navigate",
        "chrome_click",
        "chrome_type_text",
        "chrome_screenshot",
        "chrome_read_console",
        "chrome_evaluate",
    ] {
        assert!(
            names.contains(&expected),
            "missing chrome tool definition: {expected}"
        );
    }
    // Agent-level tools
    for expected in [
        "user_question",
        "task_create",
        "task_update",
        "todo_read",
        "todo_write",
        "task_output",
        "task_stop",
        "spawn_task",
        "enter_plan_mode",
        "exit_plan_mode",
        "skill",
        "extended_thinking",
    ] {
        assert!(
            names.contains(&expected),
            "missing agent tool definition: {expected}"
        );
    }
}

#[test]
fn map_tool_name_covers_all_definitions() {
    let defs = tool_definitions();
    for def in &defs {
        let fn_name = &def.function.name;
        let mapped = map_tool_name(fn_name);
        // Every underscored name should map to something
        assert!(
            !mapped.is_empty(),
            "map_tool_name returned empty for {fn_name}"
        );
    }
    // Verify chrome mappings specifically
    assert_eq!(map_tool_name("chrome_navigate"), "chrome.navigate");
    assert_eq!(map_tool_name("chrome_click"), "chrome.click");
    assert_eq!(map_tool_name("chrome_type_text"), "chrome.type_text");
    assert_eq!(map_tool_name("chrome_screenshot"), "chrome.screenshot");
    assert_eq!(map_tool_name("chrome_read_console"), "chrome.read_console");
    assert_eq!(map_tool_name("chrome_evaluate"), "chrome.evaluate");
    // Agent-level tool mappings
    assert_eq!(map_tool_name("user_question"), "user_question");
    assert_eq!(map_tool_name("task_create"), "task_create");
    assert_eq!(map_tool_name("task_update"), "task_update");
    assert_eq!(map_tool_name("todo_read"), "todo_read");
    assert_eq!(map_tool_name("todo_write"), "todo_write");
    assert_eq!(map_tool_name("task_output"), "task_output");
    assert_eq!(map_tool_name("task_stop"), "task_stop");
    assert_eq!(map_tool_name("spawn_task"), "spawn_task");
    assert_eq!(map_tool_name("extended_thinking"), "extended_thinking");
}

#[test]
fn agent_level_tools_constant_matches_definitions() {
    let defs = tool_definitions();
    let def_names: Vec<&str> = defs.iter().map(|d| d.function.name.as_str()).collect();
    for tool_name in AGENT_LEVEL_TOOLS.iter() {
        assert!(
            def_names.contains(tool_name),
            "AGENT_LEVEL_TOOLS contains '{tool_name}' but no definition exists"
        );
    }
}

#[test]
fn filter_tool_definitions_allowed() {
    let defs = tool_definitions();
    let allowed = vec!["fs_read".to_string(), "fs_write".to_string()];
    let filtered = filter_tool_definitions(defs, Some(&allowed), None);
    assert_eq!(filtered.len(), 2);
    assert!(filtered.iter().any(|t| t.function.name == "fs_read"));
    assert!(filtered.iter().any(|t| t.function.name == "fs_write"));
}

#[test]
fn filter_tool_definitions_disallowed() {
    let defs = tool_definitions();
    let original_count = defs.len();
    let disallowed = vec!["bash_run".to_string()];
    let filtered = filter_tool_definitions(defs, None, Some(&disallowed));
    assert_eq!(filtered.len(), original_count - 1);
    assert!(!filtered.iter().any(|t| t.function.name == "bash_run"));
}

#[test]
fn filter_tool_definitions_no_filters() {
    let defs = tool_definitions();
    let original_count = defs.len();
    let filtered = filter_tool_definitions(defs, None, None);
    assert_eq!(filtered.len(), original_count);
}

// ── Parameter mapping tests (Phase 16.2) ────────────────────────────

#[test]
fn tool_definitions_required_fields_exist_in_properties() {
    let defs = tool_definitions();
    for def in &defs {
        let params = &def.function.parameters;
        let required = params
            .get("required")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let properties = params.get("properties").and_then(|v| v.as_object());
        for req in &required {
            let field_name = req.as_str().unwrap_or("(non-string)");
            if let Some(props) = properties {
                assert!(
                    props.contains_key(field_name),
                    "tool '{}': required field '{}' is missing from properties",
                    def.function.name,
                    field_name
                );
            }
        }
    }
}

#[test]
fn plan_mode_tools_all_exist_in_definitions() {
    let defs = tool_definitions();
    let def_names: Vec<&str> = defs.iter().map(|d| d.function.name.as_str()).collect();
    for tool_name in PLAN_MODE_TOOLS.iter() {
        assert!(
            def_names.contains(tool_name),
            "PLAN_MODE_TOOLS contains '{}' which is not in tool_definitions()",
            tool_name
        );
    }
}

#[test]
fn agent_level_tools_excluded_from_plan_mode() {
    for tool_name in AGENT_LEVEL_TOOLS.iter() {
        // Agent-level tools that ARE in plan mode are: user_question, task_*, todo_*,
        // task_output, spawn_task, exit_plan_mode, extended_thinking
        // The write-oriented agent-level tools should not be in plan mode:
        // enter_plan_mode makes no sense inside plan mode, kill_shell is mutating, skill is execution
        let expected_in_plan = matches!(
            *tool_name,
            "user_question"
                | "task_create"
                | "task_update"
                | "todo_read"
                | "todo_write"
                | "task_get"
                | "task_list"
                | "task_output"
                | "task_stop"
                | "spawn_task"
                | "extended_thinking"
                | "exit_plan_mode"
        );
        let in_plan = PLAN_MODE_TOOLS.contains(tool_name);
        if !expected_in_plan {
            assert!(
                !in_plan,
                "agent-level tool '{}' should not be in PLAN_MODE_TOOLS",
                tool_name
            );
        }
    }
}

#[test]
fn fuzz_random_json_args_do_not_panic() {
    let (workspace, host) = temp_host();
    let fuzz_args = vec![
        serde_json::json!(null),
        serde_json::json!({}),
        serde_json::json!({"wrong_field": 42}),
        serde_json::json!({"path": 123}),
        serde_json::json!([]),
    ];
    let tool_names = ["fs.read", "fs.glob", "fs.grep"];
    for tool_name in tool_names {
        for args in &fuzz_args {
            let approved = ApprovedToolCall {
                invocation_id: Uuid::now_v7(),
                call: ToolCall {
                    name: tool_name.to_string(),
                    args: args.clone(),
                    requires_approval: false,
                },
            };
            // Should not panic — Ok or Err are both fine
            let _result = host.execute(approved);
        }
    }
    let _ = fs::remove_dir_all(&workspace);
}

// ── Batch 5: Tool description enrichment tests ──────────────────────

#[test]
fn tool_descriptions_are_sufficiently_detailed() {
    let tools = tool_definitions();
    let core_tools = [
        "fs_read",
        "fs_write",
        "fs_edit",
        "fs_list",
        "fs_glob",
        "fs_grep",
        "bash_run",
        "multi_edit",
    ];
    for name in &core_tools {
        let tool = tools
            .iter()
            .find(|t| t.function.name == *name)
            .unwrap_or_else(|| panic!("tool {name} not found"));
        assert!(
            tool.function.description.len() >= 200,
            "tool {name} description is only {} chars (expected ≥200): {}",
            tool.function.description.len(),
            &tool.function.description[..tool.function.description.len().min(80)]
        );
    }
}

#[test]
fn fs_read_mentions_read_before_edit() {
    let tools = tool_definitions();
    let fs_read = tools
        .iter()
        .find(|t| t.function.name == "fs_read")
        .expect("fs_read tool");
    let desc = &fs_read.function.description;
    assert!(
        desc.contains("fs_edit"),
        "fs_read description should cross-reference fs_edit"
    );
    assert!(
        desc.contains("MUST") && desc.contains("BEFORE"),
        "fs_read description should emphasize reading before editing"
    );
}

#[test]
fn bash_run_cross_references_dedicated_tools() {
    let tools = tool_definitions();
    let bash = tools
        .iter()
        .find(|t| t.function.name == "bash_run")
        .expect("bash_run tool");
    let desc = &bash.function.description;
    assert!(
        desc.contains("fs_read"),
        "bash_run should cross-reference fs_read"
    );
    assert!(
        desc.contains("fs_grep"),
        "bash_run should cross-reference fs_grep"
    );
    assert!(
        desc.contains("fs_glob"),
        "bash_run should cross-reference fs_glob"
    );
    assert!(
        desc.contains("fs_edit"),
        "bash_run should cross-reference fs_edit"
    );
    assert!(
        desc.contains("DO NOT"),
        "bash_run should say DO NOT use for dedicated tool operations"
    );
}

// ── P6-06: Container sandbox detection tests ────────────────────────

#[test]
fn container_env_var_detected_and_skips_sandbox() {
    // SAFETY: tests in this crate are run with --test-threads=1 for env var safety
    unsafe { std::env::set_var("CODINGBUDDY_CONTAINER_MODE", "1") };

    // Part 1: detect_container_environment returns Some("explicit")
    let result = detect_container_environment();
    assert_eq!(result, Some("explicit"));

    // Part 2: sandbox_wrap_command returns raw command when container detected
    let config = codingbuddy_core::SandboxConfig {
        enabled: true,
        ..Default::default()
    };
    let wrapped = sandbox_wrap_command(Path::new("/workspace"), "ls -la", &config);
    assert_eq!(
        wrapped, "ls -la",
        "should return raw command when in container"
    );

    unsafe { std::env::remove_var("CODINGBUDDY_CONTAINER_MODE") };
}

#[test]
fn container_detection_returns_correct_type() {
    // Verify the function returns known type strings.
    // On dev machines this returns None (no container), in CI it might return something.
    let result = detect_container_environment();
    if let Some(container_type) = result {
        // Must be one of the known container type labels
        assert!(
            ["explicit", "docker", "podman", "cgroup"].contains(&container_type),
            "unexpected container type: {container_type}"
        );
    }
    // If None, that's also valid (running on bare metal)
}

// ── T2.1: JSON schema validation tests ──────────────────────────────

#[test]
fn schema_validation_passes_valid_args() {
    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: codingbuddy_core::FunctionDefinition {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            strict: None,
            parameters: json!({
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": { "type": "string" }
                }
            }),
        },
    }];
    let args = json!({"path": "src/main.rs"});
    assert!(validate_tool_args_schema("test_tool", &args, &tools).is_ok());
}

#[test]
fn schema_validation_rejects_missing_required() {
    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: codingbuddy_core::FunctionDefinition {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            strict: None,
            parameters: json!({
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": { "type": "string" }
                }
            }),
        },
    }];
    let args = json!({});
    let result = validate_tool_args_schema("test_tool", &args, &tools);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("path"));
}

#[test]
fn schema_validation_rejects_wrong_type() {
    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: codingbuddy_core::FunctionDefinition {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            strict: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "count": { "type": "integer" }
                }
            }),
        },
    }];
    let args = json!({"count": "not a number"});
    let result = validate_tool_args_schema("test_tool", &args, &tools);
    assert!(result.is_err());
}

#[test]
fn schema_validation_skips_unknown_tools() {
    let tools = vec![];
    // Unknown tool should not fail validation
    let args = json!({"anything": "goes"});
    assert!(validate_tool_args_schema("mcp__unknown__tool", &args, &tools).is_ok());
}

#[test]
fn mcp_tool_routing_splits_name_correctly() {
    let (workspace, mut host) = temp_host();
    let captured_server = Arc::new(Mutex::new(String::new()));
    let captured_tool = Arc::new(Mutex::new(String::new()));
    let captured_args = Arc::new(Mutex::new(json!(null)));
    let cs = captured_server.clone();
    let ct = captured_tool.clone();
    let ca = captured_args.clone();

    host.set_mcp_executor(Arc::new(move |server_id, tool_name, args| {
        *cs.lock().unwrap() = server_id.to_string();
        *ct.lock().unwrap() = tool_name.to_string();
        *ca.lock().unwrap() = args.clone();
        Ok(json!({"result": "ok"}))
    }));

    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "mcp__github__search".to_string(),
            args: json!({"arguments": {"query": "test"}}),
            requires_approval: false,
        },
    });
    assert!(result.success, "MCP tool call should succeed");
    assert_eq!(*captured_server.lock().unwrap(), "github");
    assert_eq!(*captured_tool.lock().unwrap(), "search");
    assert_eq!(*captured_args.lock().unwrap(), json!({"query": "test"}));
    let _ = fs::remove_dir_all(&workspace);
}

#[test]
fn mcp_tool_without_executor_returns_error() {
    let (workspace, host) = temp_host();
    // No MCP executor set — should fail
    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "mcp__github__search".to_string(),
            args: json!({"arguments": {"query": "test"}}),
            requires_approval: false,
        },
    });
    assert!(!result.success, "should fail without executor");
    let err = result.output["error"].as_str().unwrap_or("");
    assert!(
        err.contains("no MCP executor"),
        "error should mention missing executor: {err}"
    );
    let _ = fs::remove_dir_all(&workspace);
}

#[test]
fn mcp_tool_invalid_name_returns_error() {
    let (workspace, mut host) = temp_host();
    host.set_mcp_executor(Arc::new(|_, _, _| Ok(json!({"ok": true}))));

    // "mcp__noseparator" has no second "__" — should fail to split
    let result = host.execute(ApprovedToolCall {
        invocation_id: Uuid::now_v7(),
        call: ToolCall {
            name: "mcp__noseparator".to_string(),
            args: json!({}),
            requires_approval: false,
        },
    });
    assert!(!result.success, "should fail with invalid MCP name");
    let err = result.output["error"].as_str().unwrap_or("");
    assert!(
        err.contains("invalid MCP tool name"),
        "error should mention invalid name: {err}"
    );
    let _ = fs::remove_dir_all(&workspace);
}

// ── Fuzzy edit matching tests ────────────────────────────────────────

#[test]
fn fuzzy_line_trimmed_matches_with_different_indentation() {
    let content = "fn main() {\n    let x = 1;\n    let y = 2;\n}\n";
    let search = "  let x = 1;\n  let y = 2;";
    let result = fuzzy_line_trimmed(content, search);
    assert!(result.is_some(), "should match with different indentation");
    let m = result.unwrap();
    assert_eq!(m.strategy, "line_trimmed");
    assert_eq!(&content[m.start..m.end], "    let x = 1;\n    let y = 2;");
}

#[test]
fn fuzzy_line_trimmed_rejects_ambiguous() {
    let content = "let x = 1;\nlet y = 2;\nlet x = 1;\nlet y = 2;\n";
    let search = "let x = 1;\nlet y = 2;";
    let result = fuzzy_line_trimmed(content, search);
    assert!(result.is_none(), "should reject ambiguous matches");
}

#[test]
fn fuzzy_block_anchor_matches() {
    let content = "fn foo() {\n    // comment\n    let x = 1;\n    return x;\n}\n";
    // Search with slightly different middle line
    let search = "fn foo() {\n    // different comment\n    let x = 1;\n    return x;\n}";
    let result = fuzzy_block_anchor(content, search);
    assert!(result.is_some(), "should match with fuzzy middle");
    assert_eq!(result.unwrap().strategy, "block_anchor");
}

#[test]
fn fuzzy_whitespace_normalized_matches() {
    let content = "let   x   =   1;";
    let search = "let x = 1;";
    let result = fuzzy_whitespace_normalized(content, search);
    assert!(result.is_some(), "should match with normalized whitespace");
    let m = result.unwrap();
    assert_eq!(m.strategy, "whitespace_normalized");
}

#[test]
fn fuzzy_indentation_flexible_matches() {
    let content = "    fn foo() {\n        let x = 1;\n    }\n";
    let search = "fn foo() {\n    let x = 1;\n}";
    let result = fuzzy_indentation_flexible(content, search);
    assert!(result.is_some(), "should match with flexible indentation");
    assert_eq!(result.unwrap().strategy, "indentation_flexible");
}

#[test]
fn fuzzy_escape_normalized_matches() {
    let content = "hello\nworld";
    let search = "hello\\nworld";
    let result = fuzzy_escape_normalized(content, search);
    assert!(result.is_some(), "should match after unescape");
    assert_eq!(result.unwrap().strategy, "escape_normalized");
}

#[test]
fn fuzzy_trimmed_boundary_matches() {
    let content = "fn foo() {\n    let x = 1;\n}\n";
    let search = "\nfn foo() {\n    let x = 1;\n}\n\n";
    let result = fuzzy_trimmed_boundary(content, search);
    assert!(result.is_some(), "should match after trimming blank lines");
    assert_eq!(result.unwrap().strategy, "trimmed_boundary");
}

#[test]
fn fuzzy_context_aware_matches_with_50pct_middle() {
    let content = "fn foo() {\n    let a = 1;\n    let b = 2;\n    let c = 3;\n    let d = 4;\n}\n";
    // First and last lines match exactly, 2 of 4 middle lines match (50%)
    let search = "fn foo() {\n    let a = 1;\n    let DIFFERENT = 99;\n    let c = 3;\n    let ALSO_DIFFERENT = 99;\n}";
    let result = fuzzy_context_aware(content, search);
    assert!(result.is_some(), "should match with ≥50% middle line match");
    assert_eq!(result.unwrap().strategy, "context_aware");
}

#[test]
fn apply_single_edit_uses_fuzzy_fallback() {
    let mut content = "    fn main() {\n        println!(\"hello\");\n    }\n".to_string();
    let edit = json!({
        "search": "fn main() {\n    println!(\"hello\");\n}",
        "replace": "fn main() {\n    println!(\"world\");\n}",
        "all": false
    });
    let result = apply_single_edit(&mut content, &edit);
    assert!(result.is_ok(), "fuzzy fallback should succeed: {result:?}");
    assert!(content.contains("world"), "replacement should be applied");
}

#[test]
fn apply_single_edit_exact_match_preferred_over_fuzzy() {
    let mut content = "let x = 1;\nlet y = 2;\n".to_string();
    let edit = json!({
        "search": "let x = 1;",
        "replace": "let x = 42;",
        "all": false
    });
    let result = apply_single_edit(&mut content, &edit);
    assert!(result.is_ok());
    assert_eq!(content, "let x = 42;\nlet y = 2;\n");
}
