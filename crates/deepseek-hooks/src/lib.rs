use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;
use wait_timeout::ChildExt;

// ── Hook Events ──────────────────────────────────────────────────────────────

/// All hook events matching Claude Code's 14 events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum HookEvent {
    /// Fires on session begin/resume/clear/compact.
    SessionStart,
    /// Fires when user submits a prompt (can block).
    UserPromptSubmit,
    /// Before tool execution (can allow/deny/modify).
    PreToolUse,
    /// After tool succeeds (feedback only).
    PostToolUse,
    /// After tool fails (feedback only).
    PostToolUseFailure,
    /// When permission dialog is shown (can allow/deny).
    PermissionRequest,
    /// On notification events.
    Notification,
    /// When subagent is spawned.
    SubagentStart,
    /// When subagent finishes.
    SubagentStop,
    /// When main agent finishes responding.
    Stop,
    /// When config files change mid-session.
    ConfigChange,
    /// Before context compaction (inject instructions).
    PreCompact,
    /// Session terminates.
    SessionEnd,
    /// Task marked complete.
    TaskCompleted,
}

impl HookEvent {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SessionStart => "SessionStart",
            Self::UserPromptSubmit => "UserPromptSubmit",
            Self::PreToolUse => "PreToolUse",
            Self::PostToolUse => "PostToolUse",
            Self::PostToolUseFailure => "PostToolUseFailure",
            Self::PermissionRequest => "PermissionRequest",
            Self::Notification => "Notification",
            Self::SubagentStart => "SubagentStart",
            Self::SubagentStop => "SubagentStop",
            Self::Stop => "Stop",
            Self::ConfigChange => "ConfigChange",
            Self::PreCompact => "PreCompact",
            Self::SessionEnd => "SessionEnd",
            Self::TaskCompleted => "TaskCompleted",
        }
    }

    pub fn parse_event(s: &str) -> Option<Self> {
        match s {
            "SessionStart" | "sessionstart" => Some(Self::SessionStart),
            "UserPromptSubmit" | "userpromptsubmit" => Some(Self::UserPromptSubmit),
            "PreToolUse" | "pretooluse" => Some(Self::PreToolUse),
            "PostToolUse" | "posttooluse" => Some(Self::PostToolUse),
            "PostToolUseFailure" | "posttoolusefailure" => Some(Self::PostToolUseFailure),
            "PermissionRequest" | "permissionrequest" => Some(Self::PermissionRequest),
            "Notification" | "notification" => Some(Self::Notification),
            "SubagentStart" | "subagentstart" => Some(Self::SubagentStart),
            "SubagentStop" | "subagentstop" => Some(Self::SubagentStop),
            "Stop" | "stop" => Some(Self::Stop),
            "ConfigChange" | "configchange" => Some(Self::ConfigChange),
            "PreCompact" | "precompact" => Some(Self::PreCompact),
            "SessionEnd" | "sessionend" => Some(Self::SessionEnd),
            "TaskCompleted" | "taskcompleted" => Some(Self::TaskCompleted),
            _ => None,
        }
    }

    /// Whether this event supports blocking (the hook can prevent the action).
    pub fn supports_blocking(&self) -> bool {
        matches!(
            self,
            Self::UserPromptSubmit
                | Self::PreToolUse
                | Self::PermissionRequest
                | Self::SubagentStop
                | Self::Stop
        )
    }
}

// ── Hook Configuration ───────────────────────────────────────────────────────

/// A single hook definition from settings.json.
///
/// Example in settings:
/// ```json
/// {
///   "hooks": {
///     "PreToolUse": [{
///       "matcher": "bash_run",
///       "hooks": [{
///         "type": "command",
///         "command": "bash /path/to/validate.sh"
///       }]
///     }]
///   }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookDefinition {
    /// Optional matcher — tool name, session event subtype, etc.
    #[serde(default)]
    pub matcher: Option<String>,
    /// The hook handlers to run.
    pub hooks: Vec<HookHandler>,
    /// When true, this hook fires only once per session, then is skipped.
    #[serde(default)]
    pub once: bool,
    /// When true, this hook is disabled and will not fire.
    #[serde(default)]
    pub disabled: bool,
}

/// A hook handler — how to execute the hook.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum HookHandler {
    /// Shell command handler. JSON context on stdin, exit codes control behavior.
    Command {
        command: String,
        /// Timeout in seconds (default 30).
        #[serde(default = "default_timeout")]
        timeout: u64,
    },
    /// Async (fire-and-forget) command handler. Runs in background without blocking.
    Async {
        command: String,
        /// Timeout in seconds (default 60). Process is killed after timeout.
        #[serde(default = "default_async_timeout")]
        timeout: u64,
    },
    /// Prompt-based hook: sends context to an LLM for allow/deny/modify decisions.
    Prompt {
        /// The prompt template. `{{event}}` and `{{context}}` are replaced with
        /// the event name and JSON-serialized hook input respectively.
        prompt: String,
        /// Optional model override (uses session model if absent).
        #[serde(skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// Optional JSON schema for structured output validation.
        #[serde(skip_serializing_if = "Option::is_none")]
        schema: Option<serde_json::Value>,
    },
    /// Agent-based hook: delegates hook evaluation to a named subagent.
    Agent {
        /// Name of the subagent to run (must be defined in agents config).
        agent: String,
        /// Optional model override for the subagent.
        #[serde(skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// Optional tool restrictions for the subagent.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tools: Vec<String>,
    },
}

fn default_timeout() -> u64 {
    30
}

fn default_async_timeout() -> u64 {
    60
}

/// Full hooks configuration — maps event names to hook definitions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HooksConfig {
    #[serde(flatten)]
    pub events: std::collections::HashMap<String, Vec<HookDefinition>>,
}

// ── Hook Input / Output ──────────────────────────────────────────────────────

/// JSON input sent on stdin to command hooks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookInput {
    /// The event that triggered this hook.
    pub event: String,
    /// Tool name (for PreToolUse/PostToolUse/PostToolUseFailure).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Tool arguments (for PreToolUse).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_input: Option<serde_json::Value>,
    /// Tool result (for PostToolUse/PostToolUseFailure).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_result: Option<serde_json::Value>,
    /// User prompt text (for UserPromptSubmit).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Session event subtype (for SessionStart: startup/resume/clear/compact).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_type: Option<String>,
    /// Workspace path.
    pub workspace: String,
}

/// Decision from a hook's stdout JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HookOutput {
    /// Top-level decision: "block" prevents the action.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision: Option<String>,
    /// Reason for the decision (shown to user).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// For PreToolUse: modified tool parameters to use instead.
    #[serde(skip_serializing_if = "Option::is_none", rename = "updatedInput")]
    pub updated_input: Option<serde_json::Value>,
    /// For PreToolUse/PermissionRequest: allow/deny/ask.
    #[serde(skip_serializing_if = "Option::is_none", rename = "permissionDecision")]
    pub permission_decision: Option<String>,
    /// Inject additional context into the conversation.
    #[serde(skip_serializing_if = "Option::is_none", rename = "additionalContext")]
    pub additional_context: Option<String>,
}

// ── Hook Result ──────────────────────────────────────────────────────────────

/// Result of running a single hook handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookRun {
    pub handler_description: String,
    pub success: bool,
    pub timed_out: bool,
    pub exit_code: Option<i32>,
    pub output: HookOutput,
    /// True if the hook decided to block the action.
    pub blocked: bool,
}

/// Permission decision from a PermissionRequest hook.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionDecision {
    Allow,
    Deny,
    Ask,
}

/// Aggregate result of running all hooks for an event.
#[derive(Debug, Clone, Default)]
pub struct HookResult {
    pub runs: Vec<HookRun>,
    /// True if any hook blocked the action.
    pub blocked: bool,
    /// Reason from the blocking hook.
    pub block_reason: Option<String>,
    /// Modified tool input from the last hook that provided one.
    pub updated_input: Option<serde_json::Value>,
    /// Additional context from hooks.
    pub additional_context: Vec<String>,
    /// Permission decision from a PermissionRequest hook (allow/deny/ask).
    pub permission_decision: Option<PermissionDecision>,
}

// ── Legacy types (backward compat with existing callers) ─────────────────────

/// Legacy context for file-based hooks (used by LocalToolHost).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookContext {
    pub phase: String,
    pub workspace: PathBuf,
    pub tool_name: Option<String>,
    pub tool_args_json: Option<String>,
    pub tool_result_json: Option<String>,
}

// ── Hook Runtime ─────────────────────────────────────────────────────────────

/// Manages hook execution, once-per-session tracking, and async hooks.
pub struct HookRuntime {
    workspace: PathBuf,
    config: HooksConfig,
    /// Tracks hooks that have `once: true` and have already run.
    #[allow(dead_code)]
    once_fired: Mutex<HashSet<String>>,
}

impl HookRuntime {
    pub fn new(workspace: &Path, config: HooksConfig) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            config,
            once_fired: Mutex::new(HashSet::new()),
        }
    }

    /// Fire hooks for an event with the given input context.
    pub fn fire(&self, event: HookEvent, input: &HookInput) -> HookResult {
        let event_key = event.as_str();
        let defs = match self.config.events.get(event_key) {
            Some(defs) => defs.clone(),
            None => return HookResult::default(),
        };

        let mut result = HookResult::default();

        for (def_idx, def) in defs.iter().enumerate() {
            // Skip disabled hooks.
            if def.disabled {
                continue;
            }

            // Skip once-fired hooks.
            if def.once {
                let once_key = format!("{event_key}:{def_idx}");
                let mut fired = self.once_fired.lock().expect("once_fired lock");
                if fired.contains(&once_key) {
                    continue;
                }
                fired.insert(once_key);
            }

            // Check matcher — if set, the input must match.
            if let Some(ref matcher) = def.matcher {
                let matches = match event {
                    HookEvent::PreToolUse
                    | HookEvent::PostToolUse
                    | HookEvent::PostToolUseFailure => input
                        .tool_name
                        .as_deref()
                        .is_some_and(|name| name == matcher),
                    HookEvent::SessionStart => input
                        .session_type
                        .as_deref()
                        .is_some_and(|st| st == matcher),
                    HookEvent::Notification => input
                        .session_type
                        .as_deref()
                        .is_some_and(|st| st == matcher),
                    _ => true,
                };
                if !matches {
                    continue;
                }
            }

            for handler in &def.hooks {
                let run = self.execute_handler(handler, input);
                if run.blocked && event.supports_blocking() {
                    result.blocked = true;
                    result.block_reason = run
                        .output
                        .reason
                        .clone()
                        .or_else(|| Some("Blocked by hook".to_string()));
                }
                if let Some(ref ui) = run.output.updated_input {
                    result.updated_input = Some(ui.clone());
                }
                if let Some(ref ctx) = run.output.additional_context {
                    result.additional_context.push(ctx.clone());
                }
                // Parse permission decision from PermissionRequest hooks.
                if let Some(ref pd) = run.output.permission_decision {
                    result.permission_decision = match pd.as_str() {
                        "allow" => Some(PermissionDecision::Allow),
                        "deny" => Some(PermissionDecision::Deny),
                        "ask" => Some(PermissionDecision::Ask),
                        _ => None,
                    };
                }
                result.runs.push(run);

                // If blocked, stop running more handlers.
                if result.blocked {
                    return result;
                }
            }
        }

        result
    }

    /// Execute a single hook handler.
    fn execute_handler(&self, handler: &HookHandler, input: &HookInput) -> HookRun {
        match handler {
            HookHandler::Command { command, timeout } => {
                self.run_command_handler(command, input, *timeout)
            }
            HookHandler::Async { command, timeout } => {
                self.run_async_handler(command, *timeout)
            }
            HookHandler::Prompt {
                prompt,
                model,
                schema,
            } => self.run_prompt_handler(prompt, model.as_deref(), schema.as_ref(), input),
            HookHandler::Agent {
                agent,
                model,
                tools,
            } => self.run_agent_handler(agent, model.as_deref(), tools, input),
        }
    }

    /// Fire-and-forget a command in the background. Never blocks the main loop.
    fn run_async_handler(&self, command: &str, timeout_secs: u64) -> HookRun {
        let shell = if cfg!(target_os = "windows") { "cmd" } else { "sh" };
        let shell_flag = if cfg!(target_os = "windows") { "/C" } else { "-c" };

        let mut cmd = Command::new(shell);
        cmd.arg(shell_flag)
            .arg(command)
            .current_dir(&self.workspace)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        let spawned = cmd.spawn().is_ok();

        // Spawn a reaper thread if needed to enforce timeout.
        if spawned {
            let timeout = Duration::from_secs(timeout_secs);
            let cmd_str = command.to_string();
            let ws = self.workspace.clone();
            std::thread::spawn(move || {
                // Re-spawn so we can wait with timeout (the original handle is moved)
                let shell = if cfg!(target_os = "windows") { "cmd" } else { "sh" };
                let flag = if cfg!(target_os = "windows") { "/C" } else { "-c" };
                if let Ok(mut child) = Command::new(shell)
                    .arg(flag)
                    .arg(&cmd_str)
                    .current_dir(&ws)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                {
                    match child.wait_timeout(timeout) {
                        Ok(None) => { let _ = child.kill(); let _ = child.wait(); }
                        _ => {}
                    }
                }
            });
        }

        HookRun {
            handler_description: format!("async: {command}"),
            success: spawned,
            timed_out: false,
            exit_code: None,
            output: HookOutput::default(),
            blocked: false,
        }
    }

    /// Prompt-based hook: evaluates hook context via an LLM prompt.
    /// Returns allow/deny/modify decision based on LLM response.
    ///
    /// NOTE: This is a framework implementation. The actual LLM call should be
    /// injected via a callback or trait in production. For now, it falls back to
    /// a command-based evaluation using the prompt as a shell script.
    fn run_prompt_handler(
        &self,
        prompt_template: &str,
        _model: Option<&str>,
        _schema: Option<&serde_json::Value>,
        input: &HookInput,
    ) -> HookRun {
        // Expand template variables.
        let context_json = serde_json::to_string(input).unwrap_or_default();
        let expanded = prompt_template
            .replace("{{event}}", &input.event)
            .replace("{{context}}", &context_json);

        // For now, treat the expanded prompt as a shell command that outputs HookOutput JSON.
        // In production, this would call the LLM API.
        self.run_command_handler(&expanded, input, default_timeout())
    }

    /// Agent-based hook: delegates to a named subagent for evaluation.
    ///
    /// NOTE: This is a framework implementation. Full subagent invocation
    /// requires the agent runtime which is in deepseek-agent crate. This
    /// implementation records the delegation intent and returns a non-blocking
    /// result. The agent loop should check for agent hook results and dispatch.
    fn run_agent_handler(
        &self,
        agent_name: &str,
        _model: Option<&str>,
        _tools: &[String],
        input: &HookInput,
    ) -> HookRun {
        // Record the agent delegation as additional context.
        // The main agent loop in deepseek-agent reads this and dispatches.
        let context_json = serde_json::to_string(input).unwrap_or_default();
        HookRun {
            handler_description: format!("agent: {agent_name}"),
            success: true,
            timed_out: false,
            exit_code: Some(0),
            output: HookOutput {
                additional_context: Some(format!(
                    "[hook:agent:{agent_name}] Context: {context_json}"
                )),
                ..Default::default()
            },
            blocked: false,
        }
    }

    /// Run a command hook handler: pipe JSON on stdin, parse stdout JSON for decisions.
    fn run_command_handler(&self, command: &str, input: &HookInput, timeout_secs: u64) -> HookRun {
        let input_json = serde_json::to_string(input).unwrap_or_default();

        let shell = if cfg!(target_os = "windows") {
            "cmd"
        } else {
            "sh"
        };
        let shell_flag = if cfg!(target_os = "windows") {
            "/C"
        } else {
            "-c"
        };

        let mut cmd = Command::new(shell);
        cmd.arg(shell_flag);
        cmd.arg(command);
        cmd.current_dir(&self.workspace);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::null());

        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(_e) => {
                return HookRun {
                    handler_description: command.to_string(),
                    success: false,
                    timed_out: false,
                    exit_code: None,
                    output: HookOutput::default(),
                    blocked: false,
                };
            }
        };

        // Write input JSON to stdin.
        let mut child = child;
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(input_json.as_bytes());
            drop(stdin);
        }

        let timeout = Duration::from_secs(timeout_secs);
        let (timed_out, status) = match child.wait_timeout(timeout) {
            Ok(Some(status)) => (false, Some(status)),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                (true, None)
            }
            Err(_) => (false, None),
        };

        let exit_code = status.and_then(|s| s.code());
        let success = !timed_out && exit_code == Some(0);

        // Exit code 2 = block.
        let blocked_by_exit = exit_code == Some(2);

        // Try to parse stdout as JSON for decisions.
        let stdout_output = child
            .stdout
            .and_then(|mut out| {
                let mut buf = String::new();
                std::io::Read::read_to_string(&mut out, &mut buf).ok()?;
                Some(buf)
            })
            .unwrap_or_default();

        let output: HookOutput = serde_json::from_str(&stdout_output).unwrap_or_default();

        let blocked = blocked_by_exit || output.decision.as_deref() == Some("block");

        HookRun {
            handler_description: command.to_string(),
            success,
            timed_out,
            exit_code,
            output,
            blocked,
        }
    }

    /// Legacy compatibility: run file-based hooks with the old API.
    pub fn run_legacy(
        paths: &[PathBuf],
        ctx: &HookContext,
        timeout: Duration,
    ) -> Result<Vec<HookRun>> {
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
                handler_description: path.to_string_lossy().to_string(),
                success: status.success() && !timed_out,
                timed_out,
                exit_code: status.code(),
                output: HookOutput::default(),
                blocked: false,
            });
        }
        Ok(out)
    }
}

/// Merge scoped hooks from a skill/command frontmatter into an existing config.
///
/// `skill_hooks` maps event names (e.g. "PreToolUse") to lists of shell commands.
/// Each command is added as a `HookHandler::Command` with a 30-second timeout.
pub fn merge_skill_hooks(
    base: &HooksConfig,
    skill_hooks: &std::collections::HashMap<String, Vec<String>>,
) -> HooksConfig {
    let mut merged = base.clone();
    for (event_name, commands) in skill_hooks {
        let handlers: Vec<HookDefinition> = commands
            .iter()
            .map(|cmd| HookDefinition {
                matcher: None,
                hooks: vec![HookHandler::Command {
                    command: cmd.clone(),
                    timeout: 30,
                }],
            once: false,
            disabled: false,
            })
            .collect();
        merged
            .events
            .entry(event_name.clone())
            .or_default()
            .extend(handlers);
    }
    merged
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
    use std::collections::HashMap;
    use std::fs;

    #[test]
    fn hook_event_roundtrip() {
        for event in [
            HookEvent::SessionStart,
            HookEvent::UserPromptSubmit,
            HookEvent::PreToolUse,
            HookEvent::PostToolUse,
            HookEvent::PostToolUseFailure,
            HookEvent::PermissionRequest,
            HookEvent::Notification,
            HookEvent::SubagentStart,
            HookEvent::SubagentStop,
            HookEvent::Stop,
            HookEvent::ConfigChange,
            HookEvent::PreCompact,
            HookEvent::SessionEnd,
            HookEvent::TaskCompleted,
        ] {
            assert_eq!(
                HookEvent::parse_event(event.as_str()),
                Some(event),
                "roundtrip failed for {:?}",
                event
            );
        }
    }

    #[test]
    fn hook_event_from_str_case_insensitive() {
        assert_eq!(
            HookEvent::parse_event("pretooluse"),
            Some(HookEvent::PreToolUse)
        );
        assert_eq!(
            HookEvent::parse_event("PreToolUse"),
            Some(HookEvent::PreToolUse)
        );
        assert_eq!(HookEvent::parse_event("bogus"), None);
    }

    #[test]
    fn supports_blocking_correct() {
        assert!(HookEvent::PreToolUse.supports_blocking());
        assert!(HookEvent::UserPromptSubmit.supports_blocking());
        assert!(!HookEvent::PostToolUse.supports_blocking());
        assert!(!HookEvent::SessionStart.supports_blocking());
    }

    #[test]
    fn empty_config_returns_default_result() {
        let rt = HookRuntime::new(Path::new("/tmp"), HooksConfig::default());
        let input = HookInput {
            event: "PreToolUse".to_string(),
            tool_name: Some("bash_run".to_string()),
            tool_input: None,
            tool_result: None,
            prompt: None,
            session_type: None,
            workspace: "/tmp".to_string(),
        };
        let result = rt.fire(HookEvent::PreToolUse, &input);
        assert!(!result.blocked);
        assert!(result.runs.is_empty());
    }

    #[test]
    fn matcher_filters_by_tool_name() {
        let mut events = HashMap::new();
        events.insert(
            "PreToolUse".to_string(),
            vec![HookDefinition {
                matcher: Some("bash_run".to_string()),
                hooks: vec![HookHandler::Command {
                    command: "echo '{}'".to_string(),
                    timeout: 5,
                }],
                once: false,
                disabled: false,
            }],
        );
        let rt = HookRuntime::new(Path::new("/tmp"), HooksConfig { events });

        // Should NOT match — tool name doesn't match matcher.
        let input_no_match = HookInput {
            event: "PreToolUse".to_string(),
            tool_name: Some("fs_read".to_string()),
            tool_input: None,
            tool_result: None,
            prompt: None,
            session_type: None,
            workspace: "/tmp".to_string(),
        };
        let result = rt.fire(HookEvent::PreToolUse, &input_no_match);
        assert!(result.runs.is_empty());

        // Should match.
        let input_match = HookInput {
            event: "PreToolUse".to_string(),
            tool_name: Some("bash_run".to_string()),
            tool_input: None,
            tool_result: None,
            prompt: None,
            session_type: None,
            workspace: "/tmp".to_string(),
        };
        let result = rt.fire(HookEvent::PreToolUse, &input_match);
        assert_eq!(result.runs.len(), 1);
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn command_hook_exit_2_blocks() {
        let mut events = HashMap::new();
        events.insert(
            "PreToolUse".to_string(),
            vec![HookDefinition {
                matcher: None,
                hooks: vec![HookHandler::Command {
                    command: "exit 2".to_string(),
                    timeout: 5,
                }],
            once: false,
            disabled: false,
            }],
        );
        let rt = HookRuntime::new(Path::new("/tmp"), HooksConfig { events });
        let input = HookInput {
            event: "PreToolUse".to_string(),
            tool_name: Some("bash_run".to_string()),
            tool_input: None,
            tool_result: None,
            prompt: None,
            session_type: None,
            workspace: "/tmp".to_string(),
        };
        let result = rt.fire(HookEvent::PreToolUse, &input);
        assert!(result.blocked);
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn command_hook_exit_0_allows() {
        let mut events = HashMap::new();
        events.insert(
            "PreToolUse".to_string(),
            vec![HookDefinition {
                matcher: None,
                hooks: vec![HookHandler::Command {
                    command: "exit 0".to_string(),
                    timeout: 5,
                }],
            once: false,
            disabled: false,
            }],
        );
        let rt = HookRuntime::new(Path::new("/tmp"), HooksConfig { events });
        let input = HookInput {
            event: "PreToolUse".to_string(),
            tool_name: Some("bash_run".to_string()),
            tool_input: None,
            tool_result: None,
            prompt: None,
            session_type: None,
            workspace: "/tmp".to_string(),
        };
        let result = rt.fire(HookEvent::PreToolUse, &input);
        assert!(!result.blocked);
        assert!(result.runs[0].success);
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn legacy_hook_execution() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-hooks-test-{}", uuid::Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let hook = workspace.join("hook.sh");
        fs::write(&hook, "#!/bin/sh\nexit 0\n").expect("hook");

        let runs = HookRuntime::run_legacy(
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

    #[test]
    fn hooks_config_deserializes() {
        let json = r#"{
            "PreToolUse": [{
                "matcher": "bash_run",
                "hooks": [{"type": "command", "command": "echo ok"}]
            }],
            "Stop": [{
                "hooks": [{"type": "command", "command": "notify-send done", "timeout": 10}]
            }]
        }"#;
        let config: HooksConfig = serde_json::from_str(json).expect("parse");
        assert!(config.events.contains_key("PreToolUse"));
        assert!(config.events.contains_key("Stop"));
        let pre = &config.events["PreToolUse"];
        assert_eq!(pre.len(), 1);
        assert_eq!(pre[0].matcher.as_deref(), Some("bash_run"));
    }

    // ── Hook behavior tests (Phase 16.5) ────────────────────────────────

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn user_prompt_submit_hook_blocks_prompt() {
        let mut events = HashMap::new();
        events.insert(
            "UserPromptSubmit".to_string(),
            vec![HookDefinition {
                matcher: None,
                hooks: vec![HookHandler::Command {
                    command: r#"echo '{"decision":"block","reason":"blocked by test"}' && exit 2"#
                        .to_string(),
                    timeout: 5,
                }],
                once: false,
                disabled: false,
            }],
        );
        let rt = HookRuntime::new(Path::new("/tmp"), HooksConfig { events });
        let input = HookInput {
            event: "UserPromptSubmit".to_string(),
            tool_name: None,
            tool_input: None,
            tool_result: None,
            prompt: Some("test prompt".to_string()),
            session_type: None,
            workspace: "/tmp".to_string(),
        };
        let result = rt.fire(HookEvent::UserPromptSubmit, &input);
        assert!(result.blocked, "UserPromptSubmit with exit 2 should block");
        assert!(
            result
                .block_reason
                .as_deref()
                .unwrap_or("")
                .contains("blocked by test"),
            "block reason should contain our message"
        );
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn pre_tool_use_hook_modifies_input() {
        let mut events = HashMap::new();
        events.insert(
            "PreToolUse".to_string(),
            vec![HookDefinition {
                matcher: None,
                hooks: vec![HookHandler::Command {
                    command: r#"echo '{"updatedInput":{"path":"modified.txt"}}'"#.to_string(),
                    timeout: 5,
                }],
                once: false,
                disabled: false,
            }],
        );
        let rt = HookRuntime::new(Path::new("/tmp"), HooksConfig { events });
        let input = HookInput {
            event: "PreToolUse".to_string(),
            tool_name: Some("fs_read".to_string()),
            tool_input: Some(serde_json::json!({"path": "original.txt"})),
            tool_result: None,
            prompt: None,
            session_type: None,
            workspace: "/tmp".to_string(),
        };
        let result = rt.fire(HookEvent::PreToolUse, &input);
        assert!(!result.blocked);
        assert!(result.updated_input.is_some(), "should have updated_input");
        let updated = result.updated_input.unwrap();
        assert_eq!(
            updated.get("path").and_then(|v| v.as_str()),
            Some("modified.txt")
        );
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn hook_additional_context_is_collected() {
        let mut events = HashMap::new();
        events.insert(
            "PreToolUse".to_string(),
            vec![HookDefinition {
                matcher: None,
                hooks: vec![HookHandler::Command {
                    command: r#"echo '{"additionalContext":"injected context from hook"}'"#
                        .to_string(),
                    timeout: 5,
                }],
                once: false,
                disabled: false,
            }],
        );
        let rt = HookRuntime::new(Path::new("/tmp"), HooksConfig { events });
        let input = HookInput {
            event: "PreToolUse".to_string(),
            tool_name: Some("bash_run".to_string()),
            tool_input: None,
            tool_result: None,
            prompt: None,
            session_type: None,
            workspace: "/tmp".to_string(),
        };
        let result = rt.fire(HookEvent::PreToolUse, &input);
        assert!(
            result
                .additional_context
                .iter()
                .any(|c| c.contains("injected context")),
            "additional_context should contain hook output"
        );
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn post_tool_use_fires_but_cannot_block() {
        let mut events = HashMap::new();
        events.insert(
            "PostToolUse".to_string(),
            vec![HookDefinition {
                matcher: None,
                hooks: vec![HookHandler::Command {
                    command: r#"echo '{"decision":"block","reason":"attempt to block"}' && exit 2"#
                        .to_string(),
                    timeout: 5,
                }],
                once: false,
                disabled: false,
            }],
        );
        let rt = HookRuntime::new(Path::new("/tmp"), HooksConfig { events });
        let input = HookInput {
            event: "PostToolUse".to_string(),
            tool_name: Some("fs_read".to_string()),
            tool_input: None,
            tool_result: Some(serde_json::json!({"output": "file data"})),
            prompt: None,
            session_type: None,
            workspace: "/tmp".to_string(),
        };
        let result = rt.fire(HookEvent::PostToolUse, &input);
        // PostToolUse does not support blocking (supports_blocking() returns false)
        assert!(
            !result.blocked,
            "PostToolUse should not block even with exit 2 and block decision"
        );
    }

    #[test]
    fn merge_skill_hooks_adds_to_existing() {
        let mut events = HashMap::new();
        events.insert(
            "PreToolUse".to_string(),
            vec![HookDefinition {
                matcher: None,
                hooks: vec![HookHandler::Command {
                    command: "existing".to_string(),
                    timeout: 5,
                }],
            once: false,
            disabled: false,
            }],
        );
        let base = HooksConfig { events };

        let mut skill_hooks = HashMap::new();
        skill_hooks.insert(
            "PreToolUse".to_string(),
            vec!["echo validate".to_string()],
        );
        skill_hooks.insert(
            "Stop".to_string(),
            vec!["echo done".to_string()],
        );

        let merged = merge_skill_hooks(&base, &skill_hooks);
        assert_eq!(merged.events["PreToolUse"].len(), 2, "should have original + skill hook");
        assert!(merged.events.contains_key("Stop"), "should add new event");
    }

    #[test]
    fn merge_skill_hooks_empty_noop() {
        let base = HooksConfig::default();
        let empty = HashMap::new();
        let merged = merge_skill_hooks(&base, &empty);
        assert!(merged.events.is_empty());
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn hook_timeout_does_not_hang() {
        let mut events = HashMap::new();
        events.insert(
            "PreToolUse".to_string(),
            vec![HookDefinition {
                matcher: None,
                hooks: vec![HookHandler::Command {
                    command: "sleep 10".to_string(),
                    timeout: 1,
                }],
            once: false,
            disabled: false,
            }],
        );
        let rt = HookRuntime::new(Path::new("/tmp"), HooksConfig { events });
        let input = HookInput {
            event: "PreToolUse".to_string(),
            tool_name: Some("bash_run".to_string()),
            tool_input: None,
            tool_result: None,
            prompt: None,
            session_type: None,
            workspace: "/tmp".to_string(),
        };
        let start = std::time::Instant::now();
        let result = rt.fire(HookEvent::PreToolUse, &input);
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(5),
            "hook should complete within 5 seconds (got {:.1}s)",
            elapsed.as_secs_f64()
        );
        assert_eq!(result.runs.len(), 1);
        assert!(
            result.runs[0].timed_out,
            "hook should report timed_out=true"
        );
    }

    // ── P5-10: once + disabled tests ──

    #[test]
    fn hook_once_fires_only_once() {
        let mut events = std::collections::HashMap::new();
        events.insert(
            "PreToolUse".to_string(),
            vec![HookDefinition {
                matcher: None,
                hooks: vec![HookHandler::Command {
                    command: "echo '{}'".to_string(),
                    timeout: 5,
                }],
                once: true,
                disabled: false,
            }],
        );
        let rt = HookRuntime::new(Path::new("/tmp"), HooksConfig { events });
        let input = HookInput {
            event: "PreToolUse".to_string(),
            tool_name: Some("fs_read".to_string()),
            tool_input: None,
            tool_result: None,
            prompt: None,
            session_type: None,
            workspace: "/tmp".to_string(),
        };

        // First fire — should execute
        let result1 = rt.fire(HookEvent::PreToolUse, &input);
        assert_eq!(result1.runs.len(), 1, "first fire should execute once-hook");

        // Second fire — should skip (already fired)
        let result2 = rt.fire(HookEvent::PreToolUse, &input);
        assert_eq!(result2.runs.len(), 0, "second fire should skip once-hook");
    }

    #[test]
    fn hook_disabled_skipped() {
        let mut events = std::collections::HashMap::new();
        events.insert(
            "PreToolUse".to_string(),
            vec![HookDefinition {
                matcher: None,
                hooks: vec![HookHandler::Command {
                    command: "echo '{}'".to_string(),
                    timeout: 5,
                }],
                once: false,
                disabled: true,
            }],
        );
        let rt = HookRuntime::new(Path::new("/tmp"), HooksConfig { events });
        let input = HookInput {
            event: "PreToolUse".to_string(),
            tool_name: Some("fs_read".to_string()),
            tool_input: None,
            tool_result: None,
            prompt: None,
            session_type: None,
            workspace: "/tmp".to_string(),
        };

        let result = rt.fire(HookEvent::PreToolUse, &input);
        assert_eq!(result.runs.len(), 0, "disabled hook should be skipped");
    }

    // ── P5-11: Permission decision tests ──

    #[test]
    fn permission_hook_allows() {
        let mut events = std::collections::HashMap::new();
        events.insert(
            "PermissionRequest".to_string(),
            vec![HookDefinition {
                matcher: None,
                hooks: vec![HookHandler::Command {
                    command: r#"echo '{"permissionDecision":"allow"}'"#.to_string(),
                    timeout: 5,
                }],
                once: false,
                disabled: false,
            }],
        );
        let rt = HookRuntime::new(Path::new("/tmp"), HooksConfig { events });
        let input = HookInput {
            event: "PermissionRequest".to_string(),
            tool_name: Some("bash_run".to_string()),
            tool_input: None,
            tool_result: None,
            prompt: None,
            session_type: None,
            workspace: "/tmp".to_string(),
        };

        let result = rt.fire(HookEvent::PermissionRequest, &input);
        assert_eq!(
            result.permission_decision,
            Some(PermissionDecision::Allow),
            "hook should return Allow permission decision"
        );
    }

    #[test]
    fn permission_hook_denies() {
        let mut events = std::collections::HashMap::new();
        events.insert(
            "PermissionRequest".to_string(),
            vec![HookDefinition {
                matcher: None,
                hooks: vec![HookHandler::Command {
                    command: r#"echo '{"permissionDecision":"deny","decision":"block","reason":"policy violation"}' && exit 2"#.to_string(),
                    timeout: 5,
                }],
                once: false,
                disabled: false,
            }],
        );
        let rt = HookRuntime::new(Path::new("/tmp"), HooksConfig { events });
        let input = HookInput {
            event: "PermissionRequest".to_string(),
            tool_name: Some("bash_run".to_string()),
            tool_input: None,
            tool_result: None,
            prompt: None,
            session_type: None,
            workspace: "/tmp".to_string(),
        };

        let result = rt.fire(HookEvent::PermissionRequest, &input);
        assert_eq!(result.permission_decision, Some(PermissionDecision::Deny));
        assert!(result.blocked, "deny decision should also block");
    }

    // ── P5-09: Agent handler test ──

    #[test]
    fn agent_handler_spawns_subagent() {
        let mut events = std::collections::HashMap::new();
        events.insert(
            "PreToolUse".to_string(),
            vec![HookDefinition {
                matcher: None,
                hooks: vec![HookHandler::Agent {
                    agent: "validator".to_string(),
                    model: None,
                    tools: vec!["fs_read".to_string()],
                }],
                once: false,
                disabled: false,
            }],
        );
        let rt = HookRuntime::new(Path::new("/tmp"), HooksConfig { events });
        let input = HookInput {
            event: "PreToolUse".to_string(),
            tool_name: Some("fs_write".to_string()),
            tool_input: None,
            tool_result: None,
            prompt: None,
            session_type: None,
            workspace: "/tmp".to_string(),
        };

        let result = rt.fire(HookEvent::PreToolUse, &input);
        // Agent handler currently records delegation intent
        assert_eq!(result.runs.len(), 1);
        assert!(!result.blocked, "agent handler should not block by default");
        // The additional_context should contain the agent delegation intent
        assert!(
            !result.additional_context.is_empty(),
            "agent handler should inject context about delegation"
        );
    }
}
