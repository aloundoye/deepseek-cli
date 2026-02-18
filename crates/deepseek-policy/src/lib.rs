use deepseek_core::ToolCall;
use glob::Pattern;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Component, Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    pub approve_edits: bool,
    pub approve_bash: bool,
    pub allowlist: Vec<String>,
    pub denied_secret_paths: Vec<String>,
    pub denied_command_prefixes: Vec<String>,
    pub redact_patterns: Vec<String>,
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
        Self {
            cfg,
            secret_regexes,
        }
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
        };
        if let Some(team_policy) = load_team_policy_override() {
            mapped = apply_team_policy_override(mapped, &team_policy);
        }
        Self::new(mapped)
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
        (call.name == "patch.apply" && self.cfg.approve_edits)
            || (call.name == "bash.run" && self.cfg.approve_bash)
            || call.requires_approval
    }
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
}

fn load_team_policy_override() -> Option<TeamPolicyFile> {
    let path = std::env::var("DEEPSEEK_TEAM_POLICY_PATH")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|home| std::path::PathBuf::from(home).join(".deepseek/team-policy.json"))
        })?;
    if !path.exists() {
        return None;
    }
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
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
    base
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
        };
        let team = TeamPolicyFile {
            approve_edits: Some("ask".to_string()),
            approve_bash: Some("always".to_string()),
            allowlist: vec!["npm *".to_string()],
            deny_commands: vec!["curl".to_string()],
            block_paths: vec!["**/secrets".to_string()],
            redact_patterns: vec!["password".to_string()],
        };
        let merged = apply_team_policy_override(base, &team);
        assert!(merged.approve_edits);
        assert!(merged.approve_bash);
        assert_eq!(merged.allowlist, vec!["npm *"]);
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
}
