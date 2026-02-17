use deepseek_core::ToolCall;
use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    pub approve_edits: bool,
    pub approve_bash: bool,
    pub allowlist: Vec<String>,
    pub denied_secret_paths: Vec<String>,
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
            denied_secret_paths: vec![".ssh".to_string(), ".aws".to_string(), ".gnupg".to_string()],
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
}

#[derive(Debug, Clone)]
pub struct PolicyEngine {
    cfg: PolicyConfig,
    secret_regex: Regex,
}

impl PolicyEngine {
    pub fn new(cfg: PolicyConfig) -> Self {
        Self {
            cfg,
            secret_regex: Regex::new(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s]+")
                .expect("valid regex"),
        }
    }

    pub fn from_app_config(cfg: &deepseek_core::PolicyConfig) -> Self {
        let mapped = PolicyConfig {
            approve_edits: parse_approval_mode(&cfg.approve_edits),
            approve_bash: parse_approval_mode(&cfg.approve_bash),
            allowlist: cfg.allowlist.clone(),
            denied_secret_paths: vec![".ssh".to_string(), ".aws".to_string(), ".gnupg".to_string()],
        };
        Self::new(mapped)
    }

    pub fn check_path(&self, path: &str) -> Result<(), PolicyError> {
        if path.contains("..") {
            return Err(PolicyError::PathTraversal);
        }
        if self
            .cfg
            .denied_secret_paths
            .iter()
            .any(|needle| path.contains(needle))
        {
            return Err(PolicyError::SecretPath);
        }
        Ok(())
    }

    pub fn check_command(&self, cmd: &str) -> Result<(), PolicyError> {
        let cmd_tokens: Vec<&str> = cmd.split_whitespace().collect();
        if cmd_tokens.is_empty() {
            return Err(PolicyError::CommandNotAllowed);
        }
        for allowed in &self.cfg.allowlist {
            let allowed_tokens: Vec<&str> = allowed.split_whitespace().collect();
            if allowed_tokens.is_empty() {
                continue;
            }
            if cmd_tokens.len() >= allowed_tokens.len()
                && cmd_tokens[..allowed_tokens.len()] == allowed_tokens[..]
            {
                return Ok(());
            }
        }
        Err(PolicyError::CommandNotAllowed)
    }

    pub fn redact(&self, text: &str) -> String {
        self.secret_regex
            .replace_all(text, "$1=REDACTED")
            .to_string()
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
            Err(PolicyError::CommandNotAllowed)
        ));
    }

    #[test]
    fn redacts_common_secret_patterns() {
        let policy = PolicyEngine::default();
        let out = policy.redact("api_key=abcd1234 token: xyz password = secret");
        assert!(out.contains("api_key=REDACTED"));
        assert!(out.contains("token=REDACTED"));
        assert!(out.contains("password=REDACTED"));
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
}
