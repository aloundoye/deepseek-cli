mod plugins;
mod shell;

use anyhow::{Result, anyhow};
use chrono::Utc;
use deepseek_core::{
    AppConfig, ApprovedToolCall, EventEnvelope, EventKind, ToolCall, ToolHost, ToolProposal,
    ToolResult,
};
use deepseek_diff::PatchStore;
use deepseek_index::IndexService;
use deepseek_policy::PolicyEngine;
use deepseek_store::Store;
pub use plugins::{
    CatalogPlugin, PluginCommandPrompt, PluginInfo, PluginManager, PluginVerifyResult,
};
use serde_json::json;
use sha2::Digest;
pub use shell::{PlatformShellRunner, ShellRunResult, ShellRunner};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;
use wait_timeout::ChildExt;
use walkdir::WalkDir;

const DEFAULT_TIMEOUT_SECONDS: u64 = 120;

pub struct LocalToolHost {
    workspace: PathBuf,
    policy: PolicyEngine,
    patches: PatchStore,
    index: IndexService,
    store: Store,
    runner: Arc<dyn ShellRunner + Send + Sync>,
    plugins: Option<PluginManager>,
    hooks_enabled: bool,
}

impl LocalToolHost {
    pub fn new(workspace: &Path, policy: PolicyEngine) -> Result<Self> {
        Self::with_runner(workspace, policy, Arc::new(PlatformShellRunner))
    }

    pub fn with_runner(
        workspace: &Path,
        policy: PolicyEngine,
        runner: Arc<dyn ShellRunner + Send + Sync>,
    ) -> Result<Self> {
        Ok(Self {
            workspace: workspace.to_path_buf(),
            patches: PatchStore::new(workspace)?,
            index: IndexService::new(workspace)?,
            store: Store::new(workspace)?,
            policy,
            runner,
            plugins: PluginManager::new(workspace).ok(),
            hooks_enabled: AppConfig::load(workspace)
                .map(|cfg| cfg.plugins.enabled && cfg.plugins.enable_hooks)
                .unwrap_or(false),
        })
    }

    fn run_tool(&self, call: &ToolCall) -> Result<serde_json::Value> {
        match call.name.as_str() {
            "fs.list" => {
                let dir = call.args.get("dir").and_then(|v| v.as_str()).unwrap_or(".");
                self.policy.check_path(dir)?;
                let path = self.workspace.join(dir);
                let mut out = Vec::new();
                for entry in fs::read_dir(path)? {
                    let e = entry?;
                    out.push(e.file_name().to_string_lossy().to_string());
                }
                Ok(json!({"entries": out}))
            }
            "fs.read" => {
                let path = call
                    .args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("path missing"))?;
                self.policy.check_path(path)?;
                let full = self.workspace.join(path);
                let content = fs::read_to_string(&full)?;
                let sha = format!("{:x}", sha2::Sha256::digest(content.as_bytes()));
                Ok(json!({"content": content, "sha256": sha}))
            }
            "fs.search_rg" => {
                let q = call
                    .args
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("query missing"))?;
                let limit = call
                    .args
                    .get("limit")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(20) as usize;
                let mut matches = Vec::new();
                for entry in WalkDir::new(&self.workspace)
                    .into_iter()
                    .filter_map(Result::ok)
                {
                    if !entry.path().is_file() {
                        continue;
                    }
                    let rel_path = entry.path().strip_prefix(&self.workspace)?;
                    if has_ignored_component(rel_path) {
                        continue;
                    }
                    let rel = rel_path.to_string_lossy().to_string();
                    if let Ok(content) = fs::read_to_string(entry.path()) {
                        for (idx, line) in content.lines().enumerate() {
                            if line.contains(q) {
                                matches.push(json!({"path": rel, "line": idx + 1, "text": line}));
                                if matches.len() >= limit {
                                    return Ok(json!({"matches": matches}));
                                }
                            }
                        }
                    }
                }
                Ok(json!({"matches": matches}))
            }
            "git.status" => self.run_cmd("git status --short", DEFAULT_TIMEOUT_SECONDS),
            "git.diff" => self.run_cmd("git diff", DEFAULT_TIMEOUT_SECONDS),
            "git.show" => {
                let spec = call
                    .args
                    .get("spec")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("spec missing"))?;
                self.run_cmd(&format!("git show {spec}"), DEFAULT_TIMEOUT_SECONDS)
            }
            "index.query" => {
                let q = call
                    .args
                    .get("q")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("q missing"))?;
                let top_k = call
                    .args
                    .get("top_k")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10) as usize;
                Ok(serde_json::to_value(self.index.query(q, top_k)?)?)
            }
            "patch.stage" => {
                let diff = call
                    .args
                    .get("unified_diff")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("unified_diff missing"))?;
                let base = call
                    .args
                    .get("base")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .as_bytes()
                    .to_vec();
                let patch = self.patches.stage(diff, &base)?;
                Ok(
                    json!({"patch_id": patch.patch_id.to_string(), "base_sha256": patch.base_sha256}),
                )
            }
            "patch.apply" => {
                let id = call
                    .args
                    .get("patch_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("patch_id missing"))?;
                let patch_id = Uuid::parse_str(id)?;
                let (applied, conflicts) = self.patches.apply(&self.workspace, patch_id)?;
                Ok(json!({"patch_id": id, "applied": applied, "conflicts": conflicts}))
            }
            "fs.write" => {
                let path = call
                    .args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("path missing"))?;
                self.policy.check_path(path)?;
                let content = call
                    .args
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("content missing"))?;
                fs::write(self.workspace.join(path), content)?;
                Ok(json!({"written": true}))
            }
            "bash.run" => {
                let cmd = call
                    .args
                    .get("cmd")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("cmd missing"))?;
                self.policy.check_command(cmd)?;
                let timeout = call
                    .args
                    .get("timeout")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(DEFAULT_TIMEOUT_SECONDS);
                self.run_cmd(cmd, timeout)
            }
            _ => Err(anyhow!("unknown tool: {}", call.name)),
        }
    }

    fn run_cmd(&self, cmd: &str, timeout_secs: u64) -> Result<serde_json::Value> {
        let result = self
            .runner
            .run(cmd, &self.workspace, Duration::from_secs(timeout_secs))?;
        Ok(json!({
            "status": result.status,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
        }))
    }
}

impl ToolHost for LocalToolHost {
    fn propose(&self, call: ToolCall) -> ToolProposal {
        ToolProposal {
            invocation_id: Uuid::now_v7(),
            approved: !self.policy.requires_approval(&call),
            call,
        }
    }

    fn execute(&self, approved: ApprovedToolCall) -> ToolResult {
        let call = approved.call;
        self.execute_hooks("pretooluse", Some(&call), None);
        let (success, output) = match self.run_tool(&call) {
            Ok(output) => (true, output),
            Err(err) => (false, json!({"error": err.to_string()})),
        };
        self.execute_hooks("posttooluse", Some(&call), Some(&output));
        ToolResult {
            invocation_id: approved.invocation_id,
            success,
            output,
        }
    }
}

fn has_ignored_component(path: &Path) -> bool {
    path.components().any(|c| {
        c.as_os_str() == ".git" || c.as_os_str() == ".deepseek" || c.as_os_str() == "target"
    })
}

impl LocalToolHost {
    fn execute_hooks(
        &self,
        phase: &str,
        call: Option<&ToolCall>,
        result: Option<&serde_json::Value>,
    ) {
        if !self.hooks_enabled {
            return;
        }
        let Some(manager) = &self.plugins else {
            return;
        };
        let Ok(hooks) = manager.hook_paths_for(phase) else {
            return;
        };
        for hook in hooks {
            let run_outcome = self.run_hook_script(&hook, phase, call, result);
            match run_outcome {
                Ok(outcome) => {
                    self.emit_hook_event(
                        phase,
                        &hook,
                        outcome.success,
                        outcome.timed_out,
                        outcome.exit_code,
                    );
                }
                Err(_) => {
                    self.emit_hook_event(phase, &hook, false, false, None);
                }
            }
        }
    }

    fn run_hook_script(
        &self,
        path: &Path,
        phase: &str,
        call: Option<&ToolCall>,
        result: Option<&serde_json::Value>,
    ) -> Result<HookRunOutcome> {
        let mut command = hook_command(path);
        command.current_dir(&self.workspace);
        command.stdin(Stdio::null());
        command.stdout(Stdio::null());
        command.stderr(Stdio::null());
        command.env("DEEPSEEK_HOOK_PHASE", phase);
        command.env(
            "DEEPSEEK_WORKSPACE",
            self.workspace.to_string_lossy().to_string(),
        );
        if let Some(call) = call {
            command.env("DEEPSEEK_TOOL_NAME", &call.name);
            command.env("DEEPSEEK_TOOL_ARGS_JSON", call.args.to_string());
        }
        if let Some(result) = result {
            command.env("DEEPSEEK_TOOL_RESULT_JSON", result.to_string());
        }

        let mut child = command.spawn()?;
        if child.wait_timeout(Duration::from_secs(30))?.is_none() {
            let _ = child.kill();
            let output = child.wait_with_output()?;
            return Ok(HookRunOutcome {
                success: false,
                timed_out: true,
                exit_code: output.status.code(),
            });
        }
        let output = child.wait_with_output()?;
        Ok(HookRunOutcome {
            success: output.status.success(),
            timed_out: false,
            exit_code: output.status.code(),
        })
    }

    fn emit_hook_event(
        &self,
        phase: &str,
        hook_path: &Path,
        success: bool,
        timed_out: bool,
        exit_code: Option<i32>,
    ) {
        let Ok(Some(session)) = self.store.load_latest_session() else {
            return;
        };
        let Ok(seq_no) = self.store.next_seq_no(session.session_id) else {
            return;
        };
        let event = EventEnvelope {
            seq_no,
            at: Utc::now(),
            session_id: session.session_id,
            kind: EventKind::HookExecutedV1 {
                phase: phase.to_string(),
                hook_path: hook_path.to_string_lossy().to_string(),
                success,
                timed_out,
                exit_code,
            },
        };
        let _ = self.store.append_event(&event);
    }
}

#[derive(Debug, Clone)]
struct HookRunOutcome {
    success: bool,
    timed_out: bool,
    exit_code: Option<i32>,
}

fn hook_command(path: &Path) -> Command {
    let ext = path.extension().and_then(OsStr::to_str).unwrap_or_default();
    if ext.eq_ignore_ascii_case("ps1") {
        let mut cmd = if cfg!(target_os = "windows") {
            Command::new("powershell")
        } else {
            Command::new("pwsh")
        };
        cmd.arg("-ExecutionPolicy")
            .arg("Bypass")
            .arg("-File")
            .arg(path);
        return cmd;
    }
    if ext.eq_ignore_ascii_case("sh") {
        let mut cmd = Command::new("sh");
        cmd.arg(path);
        return cmd;
    }
    if ext.eq_ignore_ascii_case("py") {
        let mut cmd = Command::new("python");
        cmd.arg(path);
        return cmd;
    }

    Command::new(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepseek_core::{AppConfig, ApprovedToolCall, ToolCall, ToolHost};
    use serde_json::json;

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn hooks_receive_phase_and_tool_context() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-tools-hook-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");

        let mut cfg = AppConfig::default();
        cfg.plugins.enable_hooks = true;
        cfg.save(&workspace).expect("save config");

        let plugin_src = workspace.join("plugin-src");
        fs::create_dir_all(plugin_src.join(".deepseek-plugin")).expect("plugin dir");
        fs::create_dir_all(plugin_src.join("hooks")).expect("hooks dir");
        fs::write(
            plugin_src.join(".deepseek-plugin/plugin.json"),
            r#"{"id":"hookdemo","name":"Hook Demo","version":"0.1.0"}"#,
        )
        .expect("manifest");
        fs::write(
            plugin_src.join("hooks/pretooluse.sh"),
            "#!/bin/sh\nprintf \"%s|%s\" \"$DEEPSEEK_HOOK_PHASE\" \"$DEEPSEEK_TOOL_NAME\" > \"$DEEPSEEK_WORKSPACE/hook.out\"\n",
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
}
