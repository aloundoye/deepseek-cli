use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use wait_timeout::ChildExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookContext {
    pub phase: String,
    pub workspace: PathBuf,
    pub tool_name: Option<String>,
    pub tool_args_json: Option<String>,
    pub tool_result_json: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookRun {
    pub path: PathBuf,
    pub success: bool,
    pub timed_out: bool,
    pub exit_code: Option<i32>,
}

pub struct HookRuntime;

impl HookRuntime {
    pub fn run(paths: &[PathBuf], ctx: &HookContext, timeout: Duration) -> Result<Vec<HookRun>> {
        let mut out = Vec::new();
        for path in paths {
            let mut cmd = build_command(path);
            cmd.current_dir(&ctx.workspace);
            cmd.stdin(Stdio::null());
            cmd.stdout(Stdio::null());
            cmd.stderr(Stdio::null());
            cmd.env("DEEPSEEK_HOOK_PHASE", &ctx.phase);
            cmd.env(
                "DEEPSEEK_WORKSPACE",
                ctx.workspace.to_string_lossy().to_string(),
            );
            if let Some(tool_name) = &ctx.tool_name {
                cmd.env("DEEPSEEK_TOOL_NAME", tool_name);
            }
            if let Some(args) = &ctx.tool_args_json {
                cmd.env("DEEPSEEK_TOOL_ARGS_JSON", args);
            }
            if let Some(result) = &ctx.tool_result_json {
                cmd.env("DEEPSEEK_TOOL_RESULT_JSON", result);
            }

            let mut child = cmd.spawn()?;
            let (timed_out, status) = match child.wait_timeout(timeout)? {
                Some(status) => (false, status),
                None => {
                    let _ = child.kill();
                    (true, child.wait()?)
                }
            };
            out.push(HookRun {
                path: path.clone(),
                success: status.success() && !timed_out,
                timed_out,
                exit_code: status.code(),
            });
        }
        Ok(out)
    }
}

fn build_command(path: &Path) -> Command {
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
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
    use std::fs;

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn executes_hook_scripts() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-hooks-test-{}", uuid::Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let hook = workspace.join("hook.sh");
        fs::write(&hook, "#!/bin/sh\nexit 0\n").expect("hook");

        let runs = HookRuntime::run(
            &[hook],
            &HookContext {
                phase: "pretooluse".to_string(),
                workspace,
                tool_name: Some("fs.list".to_string()),
                tool_args_json: Some("{}".to_string()),
                tool_result_json: None,
            },
            Duration::from_secs(2),
        )
        .expect("run");
        assert_eq!(runs.len(), 1);
        assert!(runs[0].success);
    }
}
