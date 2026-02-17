use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use wait_timeout::ChildExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellRunResult {
    pub status: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
}

pub trait ShellRunner {
    fn run(&self, cmd: &str, cwd: &Path, timeout: Duration) -> Result<ShellRunResult>;
}

#[derive(Debug, Default)]
pub struct PlatformShellRunner;

impl ShellRunner for PlatformShellRunner {
    fn run(&self, cmd: &str, cwd: &Path, timeout: Duration) -> Result<ShellRunResult> {
        let mut child = spawn_command(cmd, cwd)?;

        let status = child.wait_timeout(timeout)?;
        if status.is_none() {
            child.kill()?;
            let output = child.wait_with_output()?;
            return Ok(ShellRunResult {
                status: output.status.code(),
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                timed_out: true,
            });
        }

        let output = child.wait_with_output()?;
        Ok(ShellRunResult {
            status: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            timed_out: false,
        })
    }
}

fn spawn_command(cmd: &str, cwd: &Path) -> Result<Child> {
    let cwd = if cwd.exists() {
        std::fs::canonicalize(cwd).unwrap_or_else(|_| cwd.to_path_buf())
    } else {
        cwd.to_path_buf()
    };
    let mut errors = Vec::new();
    for mut command in candidate_commands(cmd) {
        command.current_dir(&cwd);
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());
        command.stdin(Stdio::null());
        let program = command.get_program().to_string_lossy().to_string();
        match command.spawn() {
            Ok(child) => return Ok(child),
            Err(err) => errors.push(format!("{program}: {err}")),
        }
    }
    Err(anyhow!(
        "failed to spawn command '{cmd}' in '{}': {}",
        cwd.display(),
        errors.join(" | ")
    ))
}

#[cfg(target_os = "windows")]
fn candidate_commands(cmd: &str) -> Vec<Command> {
    let mut commands = Vec::new();
    let mut cmd_shell = Command::new("cmd");
    cmd_shell.arg("/C").arg(cmd);
    commands.push(cmd_shell);

    let mut ps_shell = Command::new("powershell");
    ps_shell
        .arg("-NoLogo")
        .arg("-NoProfile")
        .arg("-Command")
        .arg(cmd);
    commands.push(ps_shell);

    commands
}

#[cfg(not(target_os = "windows"))]
fn candidate_commands(cmd: &str) -> Vec<Command> {
    let mut commands = Vec::new();
    let mut sh_shell = Command::new("sh");
    sh_shell.arg("-lc").arg(cmd);
    commands.push(sh_shell);

    let mut bash_shell = Command::new("bash");
    bash_shell.arg("-lc").arg(cmd);
    commands.push(bash_shell);

    commands
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_runner_executes_command() {
        let runner = PlatformShellRunner;
        let out = runner
            .run("echo deepseek", Path::new("."), Duration::from_secs(2))
            .expect("run command");
        assert!(!out.timed_out);
        assert!(out.stdout.to_lowercase().contains("deepseek"));
    }
}
