use deepseek_core::ToolCall;
use glob::Pattern;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Component, Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionMode {
    Ask,
    Auto,
    Plan,
    /// Auto-accepts file edits; still prompts for bash commands.
    AcceptEdits,
    /// Auto-denies everything unless pre-approved via allow rules.
    DontAsk,
    Locked,
    /// Skip ALL permission checks — requires both `--dangerously-skip-permissions`
    /// and `--allow-dangerously-skip-permissions` CLI flags.
    BypassPermissions,
}

impl PermissionMode {
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => PermissionMode::Auto,
            "plan" => PermissionMode::Plan,
            "acceptedits" | "accept-edits" | "accept_edits" => PermissionMode::AcceptEdits,
            "dontask" | "dont-ask" | "dont_ask" => PermissionMode::DontAsk,
            "locked" => PermissionMode::Locked,
            "bypasspermissions" | "bypass-permissions" | "bypass_permissions" | "bypass" => {
                PermissionMode::BypassPermissions
            }
            _ => PermissionMode::Ask,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            PermissionMode::Ask => "ask",
            PermissionMode::Auto => "auto",
            PermissionMode::Plan => "plan",
            PermissionMode::AcceptEdits => "acceptEdits",
            PermissionMode::DontAsk => "dontAsk",
            PermissionMode::Locked => "locked",
            PermissionMode::BypassPermissions => "bypassPermissions",
        }
    }

    /// Cycle to the next mode: ask → auto → acceptEdits → plan → dontAsk → locked → ask.
    /// BypassPermissions is NOT included in the cycle — it must be set explicitly.
    pub fn cycle(&self) -> Self {
        match self {
            PermissionMode::Ask => PermissionMode::Auto,
            PermissionMode::Auto => PermissionMode::AcceptEdits,
            PermissionMode::AcceptEdits => PermissionMode::Plan,
            PermissionMode::Plan => PermissionMode::DontAsk,
            PermissionMode::DontAsk => PermissionMode::Locked,
            PermissionMode::Locked => PermissionMode::Ask,
            PermissionMode::BypassPermissions => PermissionMode::Ask,
        }
    }
}

impl std::fmt::Display for PermissionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A permission rule in `Tool(specifier)` format.
/// Evaluation order: deny > ask > allow (first match wins).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionRule {
    /// The rule string, e.g. "Bash(npm run *)", "Edit(src/**/*.rs)", "WebFetch(domain:example.com)"
    pub rule: String,
    /// The decision: "allow", "deny", or "ask".
    pub decision: String,
}

impl PermissionRule {
    /// Parse a rule and check if it matches a tool call.
    /// Returns `Some(decision)` if matched, `None` otherwise.
    pub fn matches(&self, call: &ToolCall) -> Option<&str> {
        let (tool_prefix, specifier) = parse_rule_syntax(&self.rule)?;

        // MCP tool rules: Mcp(server_id) or Mcp(server_id__tool_name)
        if tool_prefix.eq_ignore_ascii_case("mcp") {
            if !call.name.starts_with("mcp__") {
                return None;
            }
            // specifier is either "server_id" (matches all tools) or "server_id__tool_name"
            let call_rest = &call.name["mcp__".len()..];
            if specifier == "*" {
                return Some(&self.decision);
            }
            if specifier.contains("__") {
                // Exact server+tool match
                if call_rest == specifier {
                    return Some(&self.decision);
                }
            } else {
                // Server-only match: "server_id" matches "mcp__server_id__*"
                if call_rest.starts_with(&specifier)
                    && call_rest[specifier.len()..].starts_with("__")
                {
                    return Some(&self.decision);
                }
            }
            return None;
        }

        let tool_name = match tool_prefix.as_str() {
            "Bash" | "bash" => "bash.run",
            "Read" | "read" => "fs.read",
            "Edit" | "edit" => "fs.edit",
            "Write" | "write" => "fs.write",
            "WebFetch" | "webfetch" | "web_fetch" => "web.fetch",
            "Task" | "task" => "spawn_task",
            "Glob" | "glob" => "fs.glob",
            "Grep" | "grep" => "fs.grep",
            _ => return None,
        };

        if call.name != tool_name {
            return None;
        }

        // Check specifier match.
        match tool_name {
            "bash.run" => {
                let cmd = call.args.get("cmd").and_then(|v| v.as_str()).unwrap_or("");
                if glob_command_matches(&specifier, cmd) {
                    Some(&self.decision)
                } else {
                    None
                }
            }
            "fs.read" | "fs.edit" | "fs.write" | "fs.glob" | "fs.grep" => {
                let path = call
                    .args
                    .get("path")
                    .or_else(|| call.args.get("file_path"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if let Ok(pattern) = Pattern::new(&specifier)
                    && (pattern.matches(path)
                        || pattern.matches_path(Path::new(path)))
                {
                    return Some(&self.decision);
                }
                None
            }
            "web.fetch" => {
                let url = call
                    .args
                    .get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if let Some(domain) = specifier.strip_prefix("domain:") {
                    if url.contains(domain) {
                        return Some(&self.decision);
                    }
                } else if url.contains(&specifier) {
                    return Some(&self.decision);
                }
                None
            }
            "spawn_task" => {
                let agent = call
                    .args
                    .get("subagent_type")
                    .or_else(|| call.args.get("role"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if agent == specifier || specifier == "*" {
                    Some(&self.decision)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Parse "Tool(specifier)" syntax. Returns (tool, specifier).
fn parse_rule_syntax(rule: &str) -> Option<(String, String)> {
    let rule = rule.trim();
    let paren_pos = rule.find('(')?;
    if !rule.ends_with(')') {
        return None;
    }
    let tool = rule[..paren_pos].trim().to_string();
    let specifier = rule[paren_pos + 1..rule.len() - 1].trim().to_string();
    if tool.is_empty() || specifier.is_empty() {
        return None;
    }
    Some((tool, specifier))
}

/// Check if a glob-like command pattern matches a command string.
fn glob_command_matches(pattern: &str, cmd: &str) -> bool {
    // Simple glob: "npm run *" matches "npm run test", "npm run build", etc.
    let pattern_tokens: Vec<&str> = pattern.split_whitespace().collect();
    let cmd_tokens: Vec<&str> = cmd.split_whitespace().collect();
    if pattern_tokens.is_empty() {
        return false;
    }
    for (i, pt) in pattern_tokens.iter().enumerate() {
        if *pt == "*" {
            return true; // wildcard matches rest
        }
        if i >= cmd_tokens.len() {
            return false;
        }
        if let Some(prefix) = pt.strip_suffix('*') {
            if !cmd_tokens[i].starts_with(prefix) {
                return false;
            }
        } else if !pt.eq_ignore_ascii_case(cmd_tokens[i]) {
            return false;
        }
    }
    cmd_tokens.len() >= pattern_tokens.len()
}

/// Evaluate permission rules against a tool call.
/// Returns: "allow", "deny", "ask", or None if no rule matched.
/// Evaluation order: deny > ask > allow (strongest match wins).
pub fn evaluate_permission_rules(rules: &[PermissionRule], call: &ToolCall) -> Option<String> {
    let mut result: Option<&str> = None;
    for rule in rules {
        if let Some(decision) = rule.matches(call) {
            match decision {
                "deny" => return Some("deny".to_string()), // deny wins immediately
                "ask" => {
                    if result.is_none() || result == Some("allow") {
                        result = Some("ask");
                    }
                }
                "allow" => {
                    if result.is_none() {
                        result = Some("allow");
                    }
                }
                _ => {}
            }
        }
    }
    result.map(|s| s.to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    pub approve_edits: bool,
    pub approve_bash: bool,
    pub allowlist: Vec<String>,
    pub denied_secret_paths: Vec<String>,
    pub denied_command_prefixes: Vec<String>,
    pub redact_patterns: Vec<String>,
    pub sandbox_mode: String,
    pub sandbox_wrapper: Option<String>,
    pub permission_mode: PermissionMode,
    /// Granular permission rules in Tool(specifier) format.
    #[serde(default)]
    pub permission_rules: Vec<PermissionRule>,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            approve_edits: true,
            approve_bash: true,
            allowlist: vec![
                "rg".to_string(),
                "git status".to_string(),
                "git diff".to_string(),
                "git show".to_string(),
                "cargo test".to_string(),
                "cargo fmt --check".to_string(),
                "cargo clippy".to_string(),
            ],
            denied_secret_paths: vec![
                ".env".to_string(),
                ".ssh".to_string(),
                ".aws".to_string(),
                ".gnupg".to_string(),
                "**/id_*".to_string(),
                "**/secret".to_string(),
            ],
            denied_command_prefixes: vec![
                "rm".to_string(),
                "rmdir".to_string(),
                "del".to_string(),
                "rd".to_string(),
                "mkfs".to_string(),
                "dd".to_string(),
                "format".to_string(),
                "shutdown".to_string(),
                "reboot".to_string(),
                "poweroff".to_string(),
            ],
            redact_patterns: vec![
                "(?i)(api[_-]?key|token|secret|password)\\s*[:=]\\s*['\"]?[a-z0-9_\\-]{8,}['\"]?"
                    .to_string(),
                "\\b\\d{3}-\\d{2}-\\d{4}\\b".to_string(),
                "(?i)\\b(mrn|medical_record_number|patient_id)\\s*[:=]\\s*[a-z0-9\\-]{4,}\\b"
                    .to_string(),
            ],
            sandbox_mode: "allowlist".to_string(),
            sandbox_wrapper: None,
            permission_mode: PermissionMode::Ask,
            permission_rules: vec![],
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum PolicyError {
    #[error("path traversal denied")]
    PathTraversal,
    #[error("secret path denied")]
    SecretPath,
    #[error("command is not allowlisted")]
    CommandNotAllowed,
    #[error("command contains forbidden shell metacharacters")]
    CommandInjection,
    #[error("command prefix is blocked by policy")]
    DangerousCommand,
}

#[derive(Debug, Clone)]
pub struct PolicyEngine {
    cfg: PolicyConfig,
    secret_regexes: Vec<Regex>,
    permission_mode: PermissionMode,
    /// When sandbox is enabled with auto_allow_bash_if_sandboxed, auto-approve bash.
    sandbox_auto_allow_bash: bool,
}

impl PolicyEngine {
    pub fn new(cfg: PolicyConfig) -> Self {
        let mut secret_regexes = cfg
            .redact_patterns
            .iter()
            .filter_map(|pattern| Regex::new(pattern).ok())
            .collect::<Vec<_>>();
        if secret_regexes.is_empty() {
            secret_regexes = PolicyConfig::default()
                .redact_patterns
                .into_iter()
                .filter_map(|pattern| Regex::new(&pattern).ok())
                .collect::<Vec<_>>();
        }
        let permission_mode = cfg.permission_mode;
        Self {
            cfg,
            secret_regexes,
            permission_mode,
            sandbox_auto_allow_bash: false,
        }
    }

    /// Create a PolicyEngine with default settings but a specific permission mode string.
    pub fn from_mode(mode: &str) -> Self {
        let cfg = PolicyConfig {
            permission_mode: PermissionMode::from_str_lossy(mode),
            ..PolicyConfig::default()
        };
        Self::new(cfg)
    }

    pub fn from_app_config(cfg: &deepseek_core::PolicyConfig) -> Self {
        let defaults = PolicyConfig::default();
        let mut mapped = PolicyConfig {
            approve_edits: parse_approval_mode(&cfg.approve_edits),
            approve_bash: parse_approval_mode(&cfg.approve_bash),
            allowlist: cfg.allowlist.clone(),
            denied_secret_paths: if cfg.block_paths.is_empty() {
                defaults.denied_secret_paths.clone()
            } else {
                cfg.block_paths.clone()
            },
            denied_command_prefixes: defaults.denied_command_prefixes,
            redact_patterns: if cfg.redact_patterns.is_empty() {
                defaults.redact_patterns
            } else {
                cfg.redact_patterns.clone()
            },
            sandbox_mode: if cfg.sandbox_mode.trim().is_empty() {
                defaults.sandbox_mode
            } else {
                cfg.sandbox_mode.trim().to_ascii_lowercase()
            },
            sandbox_wrapper: cfg
                .sandbox_wrapper
                .as_ref()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty()),
            permission_mode: PermissionMode::from_str_lossy(&cfg.permission_mode),
            permission_rules: vec![],
        };
        if let Some(team_policy) = load_team_policy_override() {
            mapped = apply_team_policy_override(mapped, &team_policy);
        }
        let sandbox_auto_allow_bash =
            cfg.sandbox.enabled && cfg.sandbox.auto_allow_bash_if_sandboxed;
        let mut engine = Self::new(mapped);
        engine.sandbox_auto_allow_bash = sandbox_auto_allow_bash;
        // Apply managed settings (enterprise overrides).
        if let Some(managed) = load_managed_settings() {
            apply_managed_settings(&mut engine, &managed);
        }
        engine
    }

    pub fn check_path(&self, path: &str) -> Result<(), PolicyError> {
        let candidate = Path::new(path);
        if candidate.is_absolute() {
            return Err(PolicyError::PathTraversal);
        }
        if candidate
            .components()
            .any(|component| matches!(component, Component::ParentDir))
        {
            return Err(PolicyError::PathTraversal);
        }
        let lowered = path.to_ascii_lowercase();
        if self.path_is_blocked(&lowered) {
            return Err(PolicyError::SecretPath);
        }
        Ok(())
    }

    pub fn check_command(&self, cmd: &str) -> Result<(), PolicyError> {
        if contains_forbidden_shell_tokens(cmd) {
            return Err(PolicyError::CommandInjection);
        }
        let cmd_tokens: Vec<&str> = cmd.split_whitespace().collect();
        if cmd_tokens.is_empty() {
            return Err(PolicyError::CommandNotAllowed);
        }
        let command_name = cmd_tokens[0].to_ascii_lowercase();
        if self
            .cfg
            .denied_command_prefixes
            .iter()
            .any(|prefix| prefix.eq_ignore_ascii_case(&command_name))
        {
            return Err(PolicyError::DangerousCommand);
        }
        for allowed in &self.cfg.allowlist {
            if allow_pattern_matches(allowed, &cmd_tokens) {
                return Ok(());
            }
        }
        Err(PolicyError::CommandNotAllowed)
    }

    pub fn redact(&self, text: &str) -> String {
        self.secret_regexes
            .iter()
            .fold(text.to_string(), |acc, regex| {
                regex.replace_all(&acc, "[REDACTED]").to_string()
            })
    }

    pub fn requires_approval(&self, call: &ToolCall) -> bool {
        // Check granular permission rules first (they take priority).
        if !self.cfg.permission_rules.is_empty()
            && let Some(decision) = evaluate_permission_rules(&self.cfg.permission_rules, call)
        {
            return match decision.as_str() {
                "allow" => false,
                "deny" | "ask" => true,
                _ => true,
            };
        }

        // Sandbox auto-approve: if sandbox is enabled with auto_allow_bash_if_sandboxed,
        // bash commands don't need approval since they'll be sandboxed at execution time.
        if self.sandbox_auto_allow_bash && call.name == "bash.run" {
            return false;
        }

        // MCP tools are non-read-only and require approval under most modes.
        let is_mcp = call.name.starts_with("mcp__");

        match self.permission_mode {
            PermissionMode::BypassPermissions => false,
            PermissionMode::Locked => {
                // In locked mode, all non-read tools require approval (and will be denied).
                if is_mcp {
                    return true;
                }
                !is_read_only_tool(&call.name)
            }
            PermissionMode::DontAsk => {
                // DontAsk: auto-denies all non-read tools unless allowlisted.
                if is_read_only_tool(&call.name) {
                    return false;
                }
                if call.name == "bash.run"
                    && let Some(cmd) = call.args.get("cmd").and_then(|v| v.as_str())
                    && self.check_command(cmd).is_ok()
                {
                    return false; // allowlisted command passes
                }
                true // everything else is denied
            }
            PermissionMode::Plan => {
                // In plan mode, reads are allowed; writes need approval.
                !is_read_only_tool(&call.name)
            }
            PermissionMode::AcceptEdits => {
                // AcceptEdits: file edits auto-approve; bash/MCP still needs approval.
                if is_mcp {
                    return true;
                }
                if is_read_only_tool(&call.name) || is_edit_tool(&call.name) {
                    false
                } else if call.name == "bash.run" {
                    // Bash still requires approval (unless allowlisted).
                    if let Some(cmd) = call.args.get("cmd").and_then(|v| v.as_str())
                        && self.check_command(cmd).is_ok()
                    {
                        return false;
                    }
                    true
                } else {
                    false
                }
            }
            PermissionMode::Auto => {
                // In auto mode, allowlisted tools auto-approve; others still need approval.
                if call.name == "bash.run" {
                    if let Some(cmd) = call.args.get("cmd").and_then(|v| v.as_str())
                        && self.check_command(cmd).is_ok()
                    {
                        return false; // allowlisted command, auto-approve
                    }
                    true
                } else if is_read_only_tool(&call.name) {
                    false
                } else {
                    // For edits/writes: auto mode doesn't prompt
                    false
                }
            }
            PermissionMode::Ask => {
                (call.name == "patch.apply" && self.cfg.approve_edits)
                    || (call.name == "bash.run" && self.cfg.approve_bash)
                    || call.requires_approval
            }
        }
    }

    /// Check if a tool call would be blocked in locked mode.
    pub fn is_blocked_in_locked_mode(&self, call: &ToolCall) -> bool {
        !is_read_only_tool(&call.name)
    }

    /// Dry-run evaluator: returns what would happen to a tool call under the current mode.
    pub fn dry_run(&self, call: &ToolCall) -> PermissionDryRunResult {
        match self.permission_mode {
            PermissionMode::BypassPermissions => PermissionDryRunResult::AutoApproved,
            PermissionMode::Locked => {
                if is_read_only_tool(&call.name) {
                    PermissionDryRunResult::Allowed
                } else {
                    PermissionDryRunResult::Denied(
                        "locked mode blocks all non-read operations".to_string(),
                    )
                }
            }
            PermissionMode::DontAsk => {
                if is_read_only_tool(&call.name) {
                    PermissionDryRunResult::Allowed
                } else if call.name == "bash.run" {
                    if let Some(cmd) = call.args.get("cmd").and_then(|v| v.as_str())
                        && self.check_command(cmd).is_ok()
                    {
                        return PermissionDryRunResult::AutoApproved;
                    }
                    PermissionDryRunResult::Denied(
                        "dontAsk mode denies non-allowlisted operations".to_string(),
                    )
                } else {
                    PermissionDryRunResult::Denied(
                        "dontAsk mode denies non-allowlisted operations".to_string(),
                    )
                }
            }
            PermissionMode::Plan => {
                if is_read_only_tool(&call.name) {
                    PermissionDryRunResult::Allowed
                } else {
                    PermissionDryRunResult::NeedsApproval
                }
            }
            PermissionMode::AcceptEdits => {
                if is_read_only_tool(&call.name) || is_edit_tool(&call.name) {
                    PermissionDryRunResult::AutoApproved
                } else if call.name == "bash.run" {
                    if let Some(cmd) = call.args.get("cmd").and_then(|v| v.as_str())
                        && self.check_command(cmd).is_ok()
                    {
                        return PermissionDryRunResult::AutoApproved;
                    }
                    PermissionDryRunResult::NeedsApproval
                } else {
                    PermissionDryRunResult::AutoApproved
                }
            }
            PermissionMode::Auto => {
                if is_read_only_tool(&call.name) {
                    PermissionDryRunResult::Allowed
                } else if call.name == "bash.run" {
                    if let Some(cmd) = call.args.get("cmd").and_then(|v| v.as_str())
                        && self.check_command(cmd).is_ok()
                    {
                        return PermissionDryRunResult::AutoApproved;
                    }
                    PermissionDryRunResult::NeedsApproval
                } else {
                    PermissionDryRunResult::AutoApproved
                }
            }
            PermissionMode::Ask => {
                if self.requires_approval(call) {
                    PermissionDryRunResult::NeedsApproval
                } else {
                    PermissionDryRunResult::Allowed
                }
            }
        }
    }

    pub fn permission_mode(&self) -> PermissionMode {
        self.permission_mode
    }

    pub fn set_permission_mode(&mut self, mode: PermissionMode) {
        self.permission_mode = mode;
    }

    pub fn sandbox_mode(&self) -> &str {
        &self.cfg.sandbox_mode
    }
}

/// Result of a dry-run evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionDryRunResult {
    /// Tool call is allowed (read-only or no gate).
    Allowed,
    /// Tool call would be auto-approved (auto mode).
    AutoApproved,
    /// Tool call needs user approval.
    NeedsApproval,
    /// Tool call would be denied.
    Denied(String),
}

/// Returns true if the tool modifies files (write/edit/patch).
fn is_edit_tool(name: &str) -> bool {
    matches!(
        name,
        "fs.write"
            | "fs.edit"
            | "multi_edit"
            | "patch.stage"
            | "patch.apply"
            | "notebook.edit"
    )
}

/// Returns true if the tool is read-only (never modifies state).
fn is_read_only_tool(name: &str) -> bool {
    matches!(
        name,
        "fs.read"
            | "fs.list"
            | "fs.glob"
            | "fs.grep"
            | "fs.search_rg"
            | "index.query"
            | "git.status"
            | "git.diff"
            | "git.show"
            | "git.log"
            | "web.fetch"
            | "web.search"
            | "notebook.read"
            | "diagnostics.check"
    )
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self::new(PolicyConfig::default())
    }
}

fn parse_approval_mode(mode: &str) -> bool {
    !matches!(mode, "never" | "false" | "off")
}

fn is_glob_pattern(value: &str) -> bool {
    value.contains('*') || value.contains('?') || value.contains('[')
}

impl PolicyEngine {
    fn path_is_blocked(&self, normalized_path: &str) -> bool {
        if self
            .cfg
            .denied_secret_paths
            .iter()
            .filter(|rule| is_glob_pattern(rule))
            .any(|rule| glob_rule_matches(rule, normalized_path))
        {
            return true;
        }
        self.cfg
            .denied_secret_paths
            .iter()
            .filter(|rule| !is_glob_pattern(rule))
            .any(|needle| normalized_path.contains(&needle.to_ascii_lowercase()))
    }
}

fn glob_rule_matches(rule: &str, normalized_path: &str) -> bool {
    if let Ok(pattern) = Pattern::new(rule)
        && (pattern.matches(normalized_path) || pattern.matches_path(Path::new(normalized_path)))
    {
        return true;
    }
    let relaxed = rule
        .replace("**/", "")
        .replace(['*', '?'], "")
        .to_ascii_lowercase();
    !relaxed.is_empty() && normalized_path.contains(&relaxed)
}

fn allow_pattern_matches(pattern: &str, cmd_tokens: &[&str]) -> bool {
    let allowed_tokens: Vec<&str> = pattern.split_whitespace().collect();
    if allowed_tokens.is_empty() {
        return false;
    }
    if cmd_tokens.len() < allowed_tokens.len() {
        return false;
    }

    for (idx, token) in allowed_tokens.iter().enumerate() {
        if *token == "*" {
            return true;
        }
        let cmd_token = cmd_tokens[idx];
        if let Some(prefix) = token.strip_suffix('*') {
            if !cmd_token.starts_with(prefix) {
                return false;
            }
            continue;
        }
        if !token.eq_ignore_ascii_case(cmd_token) {
            return false;
        }
    }
    true
}

fn contains_forbidden_shell_tokens(cmd: &str) -> bool {
    let forbidden = ["\n", "\r", ";", "&&", "||", "|", "`", "$("];
    forbidden.iter().any(|needle| cmd.contains(needle))
}

#[derive(Debug, Clone, Deserialize)]
struct TeamPolicyFile {
    #[serde(default)]
    approve_edits: Option<String>,
    #[serde(default)]
    approve_bash: Option<String>,
    #[serde(default)]
    allowlist: Vec<String>,
    #[serde(default)]
    deny_commands: Vec<String>,
    #[serde(default)]
    block_paths: Vec<String>,
    #[serde(default)]
    redact_patterns: Vec<String>,
    #[serde(default)]
    sandbox_mode: Option<String>,
    #[serde(default)]
    sandbox_wrapper: Option<String>,
    #[serde(default)]
    permission_mode: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct TeamPolicyLocks {
    pub path: String,
    pub approve_edits_locked: bool,
    pub approve_bash_locked: bool,
    pub allowlist_locked: bool,
    pub sandbox_mode_locked: bool,
    pub permission_mode_locked: bool,
}

impl TeamPolicyLocks {
    pub fn has_permission_locks(&self) -> bool {
        self.approve_edits_locked
            || self.approve_bash_locked
            || self.allowlist_locked
            || self.sandbox_mode_locked
            || self.permission_mode_locked
    }
}

pub fn team_policy_locks() -> Option<TeamPolicyLocks> {
    let (path, team) = load_team_policy_override_with_path()?;
    Some(TeamPolicyLocks {
        path: path.to_string_lossy().to_string(),
        approve_edits_locked: team.approve_edits.is_some(),
        approve_bash_locked: team.approve_bash.is_some(),
        allowlist_locked: !team.allowlist.is_empty(),
        sandbox_mode_locked: team
            .sandbox_mode
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty()),
        permission_mode_locked: team
            .permission_mode
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty()),
    })
}

fn load_team_policy_override() -> Option<TeamPolicyFile> {
    load_team_policy_override_with_path().map(|(_, team)| team)
}

fn resolve_team_policy_path() -> Option<std::path::PathBuf> {
    std::env::var("DEEPSEEK_TEAM_POLICY_PATH")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|home| std::path::PathBuf::from(home).join(".deepseek/team-policy.json"))
        })
}

fn load_team_policy_override_with_path() -> Option<(std::path::PathBuf, TeamPolicyFile)> {
    let path = resolve_team_policy_path()?;
    if !path.exists() {
        return None;
    }
    let raw = fs::read_to_string(&path).ok()?;
    let team = serde_json::from_str(&raw).ok()?;
    Some((path, team))
}

fn apply_team_policy_override(mut base: PolicyConfig, team: &TeamPolicyFile) -> PolicyConfig {
    if let Some(mode) = team.approve_edits.as_deref() {
        base.approve_edits = parse_approval_mode(mode);
    }
    if let Some(mode) = team.approve_bash.as_deref() {
        base.approve_bash = parse_approval_mode(mode);
    }
    if !team.allowlist.is_empty() {
        // Team-managed allowlist wins over local config.
        base.allowlist = team.allowlist.clone();
    }
    if !team.deny_commands.is_empty() {
        let mut denied = base.denied_command_prefixes;
        denied.extend(team.deny_commands.iter().cloned());
        denied.sort();
        denied.dedup();
        base.denied_command_prefixes = denied;
    }
    if !team.block_paths.is_empty() {
        let mut blocked = base.denied_secret_paths;
        blocked.extend(team.block_paths.iter().cloned());
        blocked.sort();
        blocked.dedup();
        base.denied_secret_paths = blocked;
    }
    if !team.redact_patterns.is_empty() {
        let mut patterns = base.redact_patterns;
        patterns.extend(team.redact_patterns.iter().cloned());
        patterns.sort();
        patterns.dedup();
        base.redact_patterns = patterns;
    }
    if let Some(mode) = team.sandbox_mode.as_deref() {
        let normalized = mode.trim().to_ascii_lowercase();
        if !normalized.is_empty() {
            base.sandbox_mode = normalized;
        }
    }
    if let Some(wrapper) = team.sandbox_wrapper.as_deref() {
        let normalized = wrapper.trim().to_string();
        base.sandbox_wrapper = if normalized.is_empty() {
            None
        } else {
            Some(normalized)
        };
    }
    if let Some(mode) = team.permission_mode.as_deref() {
        base.permission_mode = PermissionMode::from_str_lossy(mode);
    }
    base
}

// ──────────────────────────────────────────────────────────────────────────────
// Managed Settings (enterprise/team system-wide controls)
// ──────────────────────────────────────────────────────────────────────────────

/// System-wide managed settings for enterprise/team deployments.
/// Loaded from platform-specific paths and enforced as immutable overrides.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ManagedSettings {
    /// Prevent use of bypassPermissions mode entirely.
    pub disable_bypass_permissions_mode: bool,
    /// Only allow permission rules defined in managed settings.
    pub allow_managed_permission_rules_only: bool,
    /// Only allow hooks defined in managed settings.
    pub allow_managed_hooks_only: bool,
    /// Allowlist of MCP server IDs (empty = all allowed).
    pub allowed_mcp_servers: Vec<String>,
    /// Denylist of MCP server IDs (checked after allowlist).
    pub denied_mcp_servers: Vec<String>,
    /// Managed permission rules (added to or replace user rules).
    #[serde(default)]
    pub permission_rules: Vec<PermissionRule>,
    /// Force a specific permission mode.
    pub permission_mode: Option<String>,
}

/// Platform-specific path for managed settings.
pub fn managed_settings_path() -> Option<std::path::PathBuf> {
    // Check environment override first
    if let Ok(path) = std::env::var("DEEPSEEK_MANAGED_SETTINGS_PATH") {
        let path = path.trim().to_string();
        if !path.is_empty() {
            return Some(std::path::PathBuf::from(path));
        }
    }

    #[cfg(target_os = "macos")]
    {
        let path = std::path::PathBuf::from(
            "/Library/Application Support/DeepSeekCLI/managed-settings.json",
        );
        return Some(path);
    }

    #[cfg(target_os = "linux")]
    {
        let path = std::path::PathBuf::from("/etc/deepseek-cli/managed-settings.json");
        return Some(path);
    }

    #[cfg(target_os = "windows")]
    {
        if let Ok(program_data) = std::env::var("ProgramData") {
            return Some(
                std::path::PathBuf::from(program_data)
                    .join("DeepSeekCLI")
                    .join("managed-settings.json"),
            );
        }
        return Some(std::path::PathBuf::from(
            "C:\\ProgramData\\DeepSeekCLI\\managed-settings.json",
        ));
    }

    #[allow(unreachable_code)]
    None
}

/// Load managed settings from the platform-specific path.
pub fn load_managed_settings() -> Option<ManagedSettings> {
    let path = managed_settings_path()?;
    if !path.exists() {
        return None;
    }
    let raw = fs::read_to_string(&path).ok()?;
    serde_json::from_str(&raw).ok()
}

/// Check if an MCP server is allowed by managed settings.
pub fn is_mcp_server_allowed(server_id: &str, managed: &ManagedSettings) -> bool {
    // If denied list contains this server, block it.
    if managed
        .denied_mcp_servers
        .iter()
        .any(|s| s == server_id || s == "*")
    {
        return false;
    }
    // If allowed list is non-empty, server must be in it.
    if !managed.allowed_mcp_servers.is_empty() {
        return managed
            .allowed_mcp_servers
            .iter()
            .any(|s| s == server_id || s == "*");
    }
    true // default: allow
}

/// Apply managed settings enforcement to a PolicyEngine.
/// This is called after normal construction to apply enterprise overrides.
pub fn apply_managed_settings(engine: &mut PolicyEngine, managed: &ManagedSettings) {
    // Force permission mode if set.
    if let Some(ref mode) = managed.permission_mode {
        engine.permission_mode = PermissionMode::from_str_lossy(mode);
    }
    // Prevent bypass mode if disabled by managed settings.
    if managed.disable_bypass_permissions_mode
        && engine.permission_mode == PermissionMode::BypassPermissions
    {
        engine.permission_mode = PermissionMode::Ask;
    }
    // Apply managed permission rules.
    if managed.allow_managed_permission_rules_only {
        engine.cfg.permission_rules = managed.permission_rules.clone();
    } else {
        // Merge: managed rules take priority (prepended).
        let mut merged = managed.permission_rules.clone();
        merged.append(&mut engine.cfg.permission_rules);
        engine.cfg.permission_rules = merged;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;
    use std::sync::{Mutex, OnceLock};

    fn team_policy_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn denies_path_traversal_and_secret_dirs() {
        let policy = PolicyEngine::default();
        assert!(matches!(
            policy.check_path("../outside"),
            Err(PolicyError::PathTraversal)
        ));
        assert!(matches!(
            policy.check_path(".ssh/id_rsa"),
            Err(PolicyError::SecretPath)
        ));
    }

    #[test]
    fn allowlist_checks_command_prefix_tokens() {
        let policy = PolicyEngine::default();
        assert!(policy.check_command("cargo test --workspace").is_ok());
        assert!(
            policy
                .check_command("cargo test --workspace --all-targets")
                .is_ok()
        );
        assert!(matches!(
            policy.check_command("rm -rf /"),
            Err(PolicyError::DangerousCommand)
        ));
    }

    #[test]
    fn wildcard_allowlist_supports_prefix_forms() {
        let cfg = PolicyConfig {
            allowlist: vec!["npm *".to_string(), "python3*".to_string()],
            ..PolicyConfig::default()
        };
        let policy = PolicyEngine::new(cfg);
        assert!(policy.check_command("npm test").is_ok());
        assert!(policy.check_command("python3.12 -V").is_ok());
    }

    #[test]
    fn command_injection_tokens_are_blocked() {
        let policy = PolicyEngine::default();
        assert!(matches!(
            policy.check_command("git status && rm -rf /"),
            Err(PolicyError::CommandInjection)
        ));
        assert!(matches!(
            policy.check_command("cargo test; echo hacked"),
            Err(PolicyError::CommandInjection)
        ));
    }

    #[test]
    fn redacts_common_secret_patterns() {
        let policy = PolicyEngine::default();
        let out = policy.redact("api_key=abcd1234 token: xyz password = secret ssn=123-45-6789");
        assert!(out.contains("[REDACTED]"));
        assert!(!out.contains("123-45-6789"));
    }

    #[test]
    fn blocks_globbed_secret_paths() {
        let policy = PolicyEngine::default();
        assert!(matches!(
            policy.check_path("src/keys/id_rsa"),
            Err(PolicyError::SecretPath)
        ));
    }

    #[test]
    fn approval_gate_respects_tool_type_and_config() {
        let policy = PolicyEngine::default();
        let bash = ToolCall {
            name: "bash.run".to_string(),
            args: json!({}),
            requires_approval: false,
        };
        let read = ToolCall {
            name: "fs.read".to_string(),
            args: json!({}),
            requires_approval: false,
        };
        assert!(policy.requires_approval(&bash));
        assert!(!policy.requires_approval(&read));
    }

    #[test]
    fn team_policy_override_replaces_allowlist_and_forces_modes() {
        let base = PolicyConfig {
            approve_edits: false,
            approve_bash: false,
            allowlist: vec!["git status".to_string()],
            denied_secret_paths: vec![".env".to_string()],
            denied_command_prefixes: vec!["rm".to_string()],
            redact_patterns: vec!["token".to_string()],
            sandbox_mode: "allowlist".to_string(),
            sandbox_wrapper: None,
            permission_mode: PermissionMode::Ask,
            permission_rules: vec![],
        };
        let team = TeamPolicyFile {
            approve_edits: Some("ask".to_string()),
            approve_bash: Some("always".to_string()),
            allowlist: vec!["npm *".to_string()],
            deny_commands: vec!["curl".to_string()],
            block_paths: vec!["**/secrets".to_string()],
            redact_patterns: vec!["password".to_string()],
            sandbox_mode: Some("workspace-write".to_string()),
            sandbox_wrapper: Some("bwrap --cmd {cmd}".to_string()),
            permission_mode: None,
        };
        let merged = apply_team_policy_override(base, &team);
        assert!(merged.approve_edits);
        assert!(merged.approve_bash);
        assert_eq!(merged.allowlist, vec!["npm *"]);
        assert_eq!(merged.sandbox_mode, "workspace-write");
        assert_eq!(
            merged.sandbox_wrapper,
            Some("bwrap --cmd {cmd}".to_string())
        );
        assert!(
            merged
                .denied_command_prefixes
                .iter()
                .any(|rule| rule == "curl")
        );
        assert!(
            merged
                .denied_secret_paths
                .iter()
                .any(|rule| rule == "**/secrets")
        );
        assert!(
            merged
                .redact_patterns
                .iter()
                .any(|pattern| pattern == "password")
        );
    }

    #[test]
    fn sandbox_mode_maps_from_app_config() {
        let _guard = team_policy_env_lock().lock().expect("env lock");
        // SAFETY: test-only process-level env mutation.
        unsafe {
            std::env::remove_var("DEEPSEEK_TEAM_POLICY_PATH");
        }
        let cfg = deepseek_core::PolicyConfig {
            sandbox_mode: "read-only".to_string(),
            ..deepseek_core::PolicyConfig::default()
        };
        let policy = PolicyEngine::from_app_config(&cfg);
        assert_eq!(policy.sandbox_mode(), "read-only");
    }

    #[test]
    fn team_policy_locks_reflect_locked_fields() {
        let _guard = team_policy_env_lock().lock().expect("env lock");
        let dir = std::env::temp_dir().join("deepseek-policy-locks");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("team-policy.json");
        fs::write(
            &path,
            serde_json::to_vec_pretty(&json!({
                "approve_bash": "never",
                "allowlist": ["git status"],
                "sandbox_mode": "workspace-write"
            }))
            .expect("serialize"),
        )
        .expect("write team policy");

        // SAFETY: test-only process-level env mutation.
        unsafe {
            std::env::set_var("DEEPSEEK_TEAM_POLICY_PATH", &path);
        }
        let locks = team_policy_locks().expect("locks");
        assert!(!locks.approve_edits_locked);
        assert!(locks.approve_bash_locked);
        assert!(locks.allowlist_locked);
        assert!(locks.sandbox_mode_locked);
        assert!(locks.has_permission_locks());
        // SAFETY: test-only process-level env mutation.
        unsafe {
            std::env::remove_var("DEEPSEEK_TEAM_POLICY_PATH");
        }
    }

    #[test]
    fn locked_mode_blocks_all_non_read_operations() {
        let cfg = PolicyConfig {
            permission_mode: PermissionMode::Locked,
            ..PolicyConfig::default()
        };
        let policy = PolicyEngine::new(cfg);
        let write_call = ToolCall {
            name: "fs.write".to_string(),
            args: json!({"path": "test.txt"}),
            requires_approval: false,
        };
        let read_call = ToolCall {
            name: "fs.read".to_string(),
            args: json!({"path": "test.txt"}),
            requires_approval: false,
        };
        assert!(policy.requires_approval(&write_call));
        assert!(!policy.requires_approval(&read_call));
        assert_eq!(
            policy.dry_run(&write_call),
            PermissionDryRunResult::Denied(
                "locked mode blocks all non-read operations".to_string()
            )
        );
        assert_eq!(policy.dry_run(&read_call), PermissionDryRunResult::Allowed);
    }

    #[test]
    fn auto_mode_auto_approves_allowlisted_bash() {
        let cfg = PolicyConfig {
            permission_mode: PermissionMode::Auto,
            ..PolicyConfig::default()
        };
        let policy = PolicyEngine::new(cfg);
        let allowed_bash = ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd": "cargo test --workspace"}),
            requires_approval: false,
        };
        let blocked_bash = ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd": "unknown_command foo"}),
            requires_approval: false,
        };
        assert!(!policy.requires_approval(&allowed_bash));
        assert!(policy.requires_approval(&blocked_bash));
        assert_eq!(
            policy.dry_run(&allowed_bash),
            PermissionDryRunResult::AutoApproved
        );
        assert_eq!(
            policy.dry_run(&blocked_bash),
            PermissionDryRunResult::NeedsApproval
        );
    }

    #[test]
    fn permission_mode_cycles_correctly() {
        assert_eq!(PermissionMode::Ask.cycle(), PermissionMode::Auto);
        assert_eq!(PermissionMode::Auto.cycle(), PermissionMode::AcceptEdits);
        assert_eq!(PermissionMode::AcceptEdits.cycle(), PermissionMode::Plan);
        assert_eq!(PermissionMode::Plan.cycle(), PermissionMode::DontAsk);
        assert_eq!(PermissionMode::DontAsk.cycle(), PermissionMode::Locked);
        assert_eq!(PermissionMode::Locked.cycle(), PermissionMode::Ask);
    }

    #[test]
    fn permission_mode_from_str_lossy_handles_variants() {
        assert_eq!(PermissionMode::from_str_lossy("ask"), PermissionMode::Ask);
        assert_eq!(PermissionMode::from_str_lossy("auto"), PermissionMode::Auto);
        assert_eq!(PermissionMode::from_str_lossy("plan"), PermissionMode::Plan);
        assert_eq!(
            PermissionMode::from_str_lossy("locked"),
            PermissionMode::Locked
        );
        assert_eq!(
            PermissionMode::from_str_lossy("LOCKED"),
            PermissionMode::Locked
        );
        assert_eq!(
            PermissionMode::from_str_lossy("  Auto "),
            PermissionMode::Auto
        );
        assert_eq!(
            PermissionMode::from_str_lossy("  Plan "),
            PermissionMode::Plan
        );
        assert_eq!(
            PermissionMode::from_str_lossy("invalid"),
            PermissionMode::Ask
        );
    }

    #[test]
    fn plan_mode_requires_approval_for_writes() {
        let cfg = PolicyConfig {
            permission_mode: PermissionMode::Plan,
            ..PolicyConfig::default()
        };
        let policy = PolicyEngine::new(cfg);
        let write_call = ToolCall {
            name: "fs.write".to_string(),
            args: json!({"path": "test.txt"}),
            requires_approval: false,
        };
        let edit_call = ToolCall {
            name: "fs.edit".to_string(),
            args: json!({"path": "test.txt"}),
            requires_approval: false,
        };
        let bash_call = ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd": "cargo test"}),
            requires_approval: false,
        };
        assert!(policy.requires_approval(&write_call));
        assert!(policy.requires_approval(&edit_call));
        assert!(policy.requires_approval(&bash_call));
    }

    #[test]
    fn plan_mode_allows_reads() {
        let cfg = PolicyConfig {
            permission_mode: PermissionMode::Plan,
            ..PolicyConfig::default()
        };
        let policy = PolicyEngine::new(cfg);
        let read_call = ToolCall {
            name: "fs.read".to_string(),
            args: json!({"path": "test.txt"}),
            requires_approval: false,
        };
        let grep_call = ToolCall {
            name: "fs.grep".to_string(),
            args: json!({"pattern": "test"}),
            requires_approval: false,
        };
        let notebook_read = ToolCall {
            name: "notebook.read".to_string(),
            args: json!({"path": "test.ipynb"}),
            requires_approval: false,
        };
        let diagnostics_call = ToolCall {
            name: "diagnostics.check".to_string(),
            args: json!({}),
            requires_approval: false,
        };
        assert!(!policy.requires_approval(&read_call));
        assert!(!policy.requires_approval(&grep_call));
        assert!(!policy.requires_approval(&notebook_read));
        assert!(!policy.requires_approval(&diagnostics_call));
    }

    #[test]
    fn plan_mode_dry_run_needs_approval() {
        let cfg = PolicyConfig {
            permission_mode: PermissionMode::Plan,
            ..PolicyConfig::default()
        };
        let policy = PolicyEngine::new(cfg);
        let write_call = ToolCall {
            name: "fs.write".to_string(),
            args: json!({"path": "test.txt"}),
            requires_approval: false,
        };
        let read_call = ToolCall {
            name: "fs.read".to_string(),
            args: json!({"path": "test.txt"}),
            requires_approval: false,
        };
        assert_eq!(
            policy.dry_run(&write_call),
            PermissionDryRunResult::NeedsApproval
        );
        assert_eq!(policy.dry_run(&read_call), PermissionDryRunResult::Allowed);
    }

    proptest! {
        #[test]
        fn parent_dir_paths_are_always_rejected(
            head in "[a-z]{1,8}",
            tail in "[a-z]{1,8}",
        ) {
            let policy = PolicyEngine::default();
            let candidate = format!("{head}/../{tail}");
            prop_assert!(matches!(
                policy.check_path(&candidate),
                Err(PolicyError::PathTraversal)
            ));
        }

        #[test]
        fn commands_with_shell_injection_tokens_are_rejected(
            left in "[a-zA-Z0-9 _\\-]{0,24}",
            right in "[a-zA-Z0-9 _\\-]{0,24}",
            token in prop::sample::select(vec![";", "&&", "||", "|", "`", "$("]),
        ) {
            let policy = PolicyEngine::default();
            let cmd = format!("{left}{token}{right}");
            prop_assert!(matches!(
                policy.check_command(&cmd),
                Err(PolicyError::CommandInjection)
            ));
        }

        #[test]
        fn wildcard_allowlist_accepts_npm_subcommands(
            subcommand in "[a-z]{1,10}",
            arg in "[a-z0-9\\-]{0,10}",
        ) {
            let cfg = PolicyConfig {
                allowlist: vec!["npm *".to_string()],
                ..PolicyConfig::default()
            };
            let policy = PolicyEngine::new(cfg);
            let cmd = if arg.is_empty() {
                format!("npm {subcommand}")
            } else {
                format!("npm {subcommand} {arg}")
            };
            prop_assert!(policy.check_command(&cmd).is_ok());
        }
    }

    #[test]
    fn accept_edits_mode_auto_approves_edits_but_prompts_bash() {
        let cfg = PolicyConfig {
            permission_mode: PermissionMode::AcceptEdits,
            ..PolicyConfig::default()
        };
        let policy = PolicyEngine::new(cfg);
        let edit_call = ToolCall {
            name: "fs.edit".to_string(),
            args: json!({"path": "test.txt"}),
            requires_approval: false,
        };
        let write_call = ToolCall {
            name: "fs.write".to_string(),
            args: json!({"path": "test.txt"}),
            requires_approval: false,
        };
        let bash_call = ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd": "unknown_command"}),
            requires_approval: false,
        };
        assert!(!policy.requires_approval(&edit_call));
        assert!(!policy.requires_approval(&write_call));
        assert!(policy.requires_approval(&bash_call));
        assert_eq!(
            policy.dry_run(&edit_call),
            PermissionDryRunResult::AutoApproved
        );
        assert_eq!(
            policy.dry_run(&bash_call),
            PermissionDryRunResult::NeedsApproval
        );
    }

    #[test]
    fn dont_ask_mode_denies_non_allowlisted() {
        let cfg = PolicyConfig {
            permission_mode: PermissionMode::DontAsk,
            ..PolicyConfig::default()
        };
        let policy = PolicyEngine::new(cfg);
        let edit_call = ToolCall {
            name: "fs.edit".to_string(),
            args: json!({"path": "test.txt"}),
            requires_approval: false,
        };
        let read_call = ToolCall {
            name: "fs.read".to_string(),
            args: json!({"path": "test.txt"}),
            requires_approval: false,
        };
        let allowlisted_bash = ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd": "cargo test --workspace"}),
            requires_approval: false,
        };
        let random_bash = ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd": "unknown_command"}),
            requires_approval: false,
        };
        assert!(policy.requires_approval(&edit_call));
        assert!(!policy.requires_approval(&read_call));
        assert!(!policy.requires_approval(&allowlisted_bash));
        assert!(policy.requires_approval(&random_bash));
        assert_eq!(
            policy.dry_run(&edit_call),
            PermissionDryRunResult::Denied("dontAsk mode denies non-allowlisted operations".to_string())
        );
    }

    #[test]
    fn permission_rule_matches_bash_pattern() {
        let rule = PermissionRule {
            rule: "Bash(npm run *)".to_string(),
            decision: "allow".to_string(),
        };
        let matching = ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd": "npm run test"}),
            requires_approval: false,
        };
        let non_matching = ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd": "cargo test"}),
            requires_approval: false,
        };
        assert_eq!(rule.matches(&matching), Some("allow"));
        assert_eq!(rule.matches(&non_matching), None);
    }

    #[test]
    fn permission_rule_matches_edit_glob() {
        let rule = PermissionRule {
            rule: "Edit(src/**/*.rs)".to_string(),
            decision: "allow".to_string(),
        };
        let matching = ToolCall {
            name: "fs.edit".to_string(),
            args: json!({"path": "src/main.rs"}),
            requires_approval: false,
        };
        let non_matching = ToolCall {
            name: "fs.edit".to_string(),
            args: json!({"path": "tests/test.js"}),
            requires_approval: false,
        };
        assert_eq!(rule.matches(&matching), Some("allow"));
        assert_eq!(rule.matches(&non_matching), None);
    }

    #[test]
    fn permission_rule_matches_webfetch_domain() {
        let rule = PermissionRule {
            rule: "WebFetch(domain:example.com)".to_string(),
            decision: "allow".to_string(),
        };
        let matching = ToolCall {
            name: "web.fetch".to_string(),
            args: json!({"url": "https://example.com/api"}),
            requires_approval: false,
        };
        let non_matching = ToolCall {
            name: "web.fetch".to_string(),
            args: json!({"url": "https://evil.com/api"}),
            requires_approval: false,
        };
        assert_eq!(rule.matches(&matching), Some("allow"));
        assert_eq!(rule.matches(&non_matching), None);
    }

    #[test]
    fn evaluate_rules_deny_wins_over_allow() {
        let rules = vec![
            PermissionRule {
                rule: "Bash(npm *)".to_string(),
                decision: "allow".to_string(),
            },
            PermissionRule {
                rule: "Bash(npm run deploy)".to_string(),
                decision: "deny".to_string(),
            },
        ];
        let deploy = ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd": "npm run deploy"}),
            requires_approval: false,
        };
        let test = ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd": "npm test"}),
            requires_approval: false,
        };
        assert_eq!(evaluate_permission_rules(&rules, &deploy), Some("deny".to_string()));
        assert_eq!(evaluate_permission_rules(&rules, &test), Some("allow".to_string()));
    }

    #[test]
    fn permission_mode_from_str_handles_new_modes() {
        assert_eq!(
            PermissionMode::from_str_lossy("acceptEdits"),
            PermissionMode::AcceptEdits
        );
        assert_eq!(
            PermissionMode::from_str_lossy("accept-edits"),
            PermissionMode::AcceptEdits
        );
        assert_eq!(
            PermissionMode::from_str_lossy("dontAsk"),
            PermissionMode::DontAsk
        );
        assert_eq!(
            PermissionMode::from_str_lossy("dont-ask"),
            PermissionMode::DontAsk
        );
    }

    #[test]
    fn team_policy_locks_permission_mode() {
        let base = PolicyConfig {
            permission_mode: PermissionMode::Ask,
            ..PolicyConfig::default()
        };
        let team = TeamPolicyFile {
            approve_edits: None,
            approve_bash: None,
            allowlist: vec![],
            deny_commands: vec![],
            block_paths: vec![],
            redact_patterns: vec![],
            sandbox_mode: None,
            sandbox_wrapper: None,
            permission_mode: Some("locked".to_string()),
        };
        let merged = apply_team_policy_override(base, &team);
        assert_eq!(merged.permission_mode, PermissionMode::Locked);
    }

    #[test]
    fn team_policy_permission_mode_locked_flag() {
        let _guard = team_policy_env_lock().lock().expect("env lock");
        let dir = std::env::temp_dir().join("deepseek-policy-pm-lock");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("team-policy.json");
        fs::write(
            &path,
            serde_json::to_vec_pretty(&json!({
                "permission_mode": "locked"
            }))
            .expect("serialize"),
        )
        .expect("write team policy");

        // SAFETY: test-only process-level env mutation.
        unsafe {
            std::env::set_var("DEEPSEEK_TEAM_POLICY_PATH", &path);
        }
        let locks = team_policy_locks().expect("locks");
        assert!(locks.permission_mode_locked);
        assert!(locks.has_permission_locks());
        // SAFETY: test-only process-level env mutation.
        unsafe {
            std::env::remove_var("DEEPSEEK_TEAM_POLICY_PATH");
        }
    }

    #[test]
    fn bypass_permissions_mode_never_requires_approval() {
        let cfg = PolicyConfig {
            permission_mode: PermissionMode::BypassPermissions,
            ..PolicyConfig::default()
        };
        let policy = PolicyEngine::new(cfg);
        let bash_call = ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd": "rm -rf /"}),
            requires_approval: true,
        };
        let write_call = ToolCall {
            name: "fs.write".to_string(),
            args: json!({"path": "test.txt"}),
            requires_approval: true,
        };
        let read_call = ToolCall {
            name: "fs.read".to_string(),
            args: json!({"path": "test.txt"}),
            requires_approval: false,
        };
        assert!(!policy.requires_approval(&bash_call));
        assert!(!policy.requires_approval(&write_call));
        assert!(!policy.requires_approval(&read_call));
        assert_eq!(
            policy.dry_run(&bash_call),
            PermissionDryRunResult::AutoApproved
        );
        assert_eq!(
            policy.dry_run(&write_call),
            PermissionDryRunResult::AutoApproved
        );
    }

    #[test]
    fn bypass_permissions_from_str_lossy() {
        assert_eq!(
            PermissionMode::from_str_lossy("bypassPermissions"),
            PermissionMode::BypassPermissions
        );
        assert_eq!(
            PermissionMode::from_str_lossy("bypass"),
            PermissionMode::BypassPermissions
        );
        assert_eq!(
            PermissionMode::from_str_lossy("bypass-permissions"),
            PermissionMode::BypassPermissions
        );
    }

    #[test]
    fn bypass_permissions_not_in_cycle() {
        // BypassPermissions cycles back to Ask (it's excluded from normal rotation)
        assert_eq!(
            PermissionMode::BypassPermissions.cycle(),
            PermissionMode::Ask
        );
    }

    #[test]
    fn mcp_permission_rule_matches_server_wildcard() {
        let rule = PermissionRule {
            rule: "Mcp(myserver)".to_string(),
            decision: "allow".to_string(),
        };
        let matching = ToolCall {
            name: "mcp__myserver__read_file".to_string(),
            args: json!({}),
            requires_approval: true,
        };
        let non_matching = ToolCall {
            name: "mcp__otherserver__read_file".to_string(),
            args: json!({}),
            requires_approval: true,
        };
        assert_eq!(rule.matches(&matching), Some("allow"));
        assert_eq!(rule.matches(&non_matching), None);
    }

    #[test]
    fn mcp_permission_rule_matches_specific_tool() {
        let rule = PermissionRule {
            rule: "Mcp(myserver__read_file)".to_string(),
            decision: "deny".to_string(),
        };
        let matching = ToolCall {
            name: "mcp__myserver__read_file".to_string(),
            args: json!({}),
            requires_approval: true,
        };
        let non_matching = ToolCall {
            name: "mcp__myserver__write_file".to_string(),
            args: json!({}),
            requires_approval: true,
        };
        assert_eq!(rule.matches(&matching), Some("deny"));
        assert_eq!(rule.matches(&non_matching), None);
    }

    #[test]
    fn mcp_permission_rule_wildcard_all() {
        let rule = PermissionRule {
            rule: "Mcp(*)".to_string(),
            decision: "allow".to_string(),
        };
        let call = ToolCall {
            name: "mcp__anyserver__anytool".to_string(),
            args: json!({}),
            requires_approval: true,
        };
        assert_eq!(rule.matches(&call), Some("allow"));
    }

    #[test]
    fn mcp_permission_rule_ignores_non_mcp_tools() {
        let rule = PermissionRule {
            rule: "Mcp(myserver)".to_string(),
            decision: "allow".to_string(),
        };
        let call = ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd": "ls"}),
            requires_approval: false,
        };
        assert_eq!(rule.matches(&call), None);
    }

    #[test]
    fn mcp_tools_require_approval_in_ask_mode() {
        let policy = PolicyEngine::default();
        let mcp_call = ToolCall {
            name: "mcp__server__tool".to_string(),
            args: json!({}),
            requires_approval: true,
        };
        assert!(policy.requires_approval(&mcp_call));
    }

    #[test]
    fn mcp_tools_allowed_by_permission_rules() {
        let cfg = PolicyConfig {
            permission_rules: vec![PermissionRule {
                rule: "Mcp(myserver)".to_string(),
                decision: "allow".to_string(),
            }],
            ..PolicyConfig::default()
        };
        let policy = PolicyEngine::new(cfg);
        let allowed = ToolCall {
            name: "mcp__myserver__read_file".to_string(),
            args: json!({}),
            requires_approval: true,
        };
        let not_allowed = ToolCall {
            name: "mcp__otherserver__read_file".to_string(),
            args: json!({}),
            requires_approval: true,
        };
        assert!(!policy.requires_approval(&allowed));
        assert!(policy.requires_approval(&not_allowed));
    }

    #[test]
    fn sandbox_auto_allow_bash_skips_approval() {
        let mut policy = PolicyEngine::new(PolicyConfig {
            permission_mode: PermissionMode::Ask,
            approve_bash: true, // normally requires approval
            ..PolicyConfig::default()
        });
        let bash_call = ToolCall {
            name: "bash.run".to_string(),
            args: json!({"cmd": "unknown_command"}),
            requires_approval: false,
        };
        // Without sandbox auto-allow, approval is required
        assert!(policy.requires_approval(&bash_call));
        // With sandbox auto-allow, bash is auto-approved
        policy.sandbox_auto_allow_bash = true;
        assert!(!policy.requires_approval(&bash_call));
    }

    #[test]
    fn managed_settings_disable_bypass_mode() {
        let mut engine = PolicyEngine::new(PolicyConfig {
            permission_mode: PermissionMode::BypassPermissions,
            ..PolicyConfig::default()
        });
        assert_eq!(engine.permission_mode(), PermissionMode::BypassPermissions);

        let managed = ManagedSettings {
            disable_bypass_permissions_mode: true,
            ..Default::default()
        };
        apply_managed_settings(&mut engine, &managed);
        // Bypass should have been downgraded to Ask
        assert_eq!(engine.permission_mode(), PermissionMode::Ask);
    }

    #[test]
    fn managed_settings_force_permission_mode() {
        let mut engine = PolicyEngine::new(PolicyConfig {
            permission_mode: PermissionMode::Auto,
            ..PolicyConfig::default()
        });
        let managed = ManagedSettings {
            permission_mode: Some("locked".to_string()),
            ..Default::default()
        };
        apply_managed_settings(&mut engine, &managed);
        assert_eq!(engine.permission_mode(), PermissionMode::Locked);
    }

    #[test]
    fn managed_settings_replace_permission_rules() {
        let mut engine = PolicyEngine::new(PolicyConfig {
            permission_rules: vec![PermissionRule {
                rule: "Bash(npm *)".to_string(),
                decision: "allow".to_string(),
            }],
            ..PolicyConfig::default()
        });
        let managed = ManagedSettings {
            allow_managed_permission_rules_only: true,
            permission_rules: vec![PermissionRule {
                rule: "Bash(cargo *)".to_string(),
                decision: "allow".to_string(),
            }],
            ..Default::default()
        };
        apply_managed_settings(&mut engine, &managed);
        // User rules replaced by managed rules
        assert_eq!(engine.cfg.permission_rules.len(), 1);
        assert!(engine.cfg.permission_rules[0].rule.contains("cargo"));
    }

    #[test]
    fn mcp_server_allowed_by_managed_settings() {
        let managed = ManagedSettings {
            allowed_mcp_servers: vec!["server_a".to_string(), "server_b".to_string()],
            denied_mcp_servers: vec![],
            ..Default::default()
        };
        assert!(is_mcp_server_allowed("server_a", &managed));
        assert!(is_mcp_server_allowed("server_b", &managed));
        assert!(!is_mcp_server_allowed("server_c", &managed));
    }

    #[test]
    fn mcp_server_denied_by_managed_settings() {
        let managed = ManagedSettings {
            allowed_mcp_servers: vec![],
            denied_mcp_servers: vec!["evil_server".to_string()],
            ..Default::default()
        };
        assert!(is_mcp_server_allowed("good_server", &managed));
        assert!(!is_mcp_server_allowed("evil_server", &managed));
    }

    #[test]
    fn managed_settings_path_returns_some() {
        // On any platform, this should return a path.
        let path = managed_settings_path();
        assert!(path.is_some());
    }
}
