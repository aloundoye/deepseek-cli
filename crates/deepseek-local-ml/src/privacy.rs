use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};

/// Policy action when sensitive content is detected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyPolicy {
    /// Block content from being sent to cloud APIs entirely.
    BlockCloud,
    /// Redact sensitive parts before sending to cloud.
    Redact,
    /// Generate a local-only summary instead of sending raw content.
    LocalOnlySummary,
}

impl Default for PrivacyPolicy {
    fn default() -> Self {
        Self::Redact
    }
}

/// Configuration for privacy routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    pub enabled: bool,
    pub sensitive_globs: Vec<String>,
    pub sensitive_regex: Vec<String>,
    pub policy: PrivacyPolicy,
    pub store_raw_in_logs: bool,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sensitive_globs: vec![
                "**/.env".to_string(),
                "**/.env.*".to_string(),
                "**/*.pem".to_string(),
                "**/*.key".to_string(),
                "**/id_rsa*".to_string(),
                "**/id_ed25519*".to_string(),
                "**/credentials.json".to_string(),
                "**/.aws/credentials".to_string(),
            ],
            sensitive_regex: Vec::new(),
            policy: PrivacyPolicy::Redact,
            store_raw_in_logs: false,
        }
    }
}

/// A match found during privacy scanning.
#[derive(Debug, Clone)]
pub struct SensitiveMatch {
    pub pattern: String,
    pub line_number: usize,
    pub redacted_preview: String,
}

/// Result of applying the privacy policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrivacyResult {
    /// Content is clean, no sensitive data found.
    Clean(String),
    /// Content was redacted (sensitive parts replaced).
    Redacted(String),
    /// Content was blocked from being sent to cloud.
    Blocked,
    /// A local summary was generated instead.
    LocalSummary(String),
}

/// Routes content through privacy checks based on configuration.
///
/// Combines three layers of sensitivity detection:
/// 1. Path-based: glob patterns match sensitive file paths (.env, .pem, etc.)
/// 2. Content-based: regex patterns match secrets in file content (API keys, tokens, etc.)
/// 3. Builtin patterns: common secret formats (SK-*, AKIA*, ghp_*, PEM blocks, connection strings)
pub struct PrivacyRouter {
    config: PrivacyConfig,
    glob_patterns: Vec<glob::Pattern>,
    regex_patterns: Vec<Regex>,
    builtin_patterns: Vec<(String, Regex)>,
}

impl PrivacyRouter {
    pub fn new(config: PrivacyConfig) -> Result<Self> {
        let glob_patterns: Vec<glob::Pattern> = config
            .sensitive_globs
            .iter()
            .filter_map(|g| glob::Pattern::new(g).ok())
            .collect();

        let regex_patterns: Vec<Regex> = config
            .sensitive_regex
            .iter()
            .filter_map(|r| Regex::new(r).ok())
            .collect();

        let builtin_patterns = build_builtin_secret_patterns();

        Ok(Self {
            config,
            glob_patterns,
            regex_patterns,
            builtin_patterns,
        })
    }

    /// Check if a file path is sensitive based on glob patterns.
    pub fn is_sensitive_path(&self, path: &str) -> bool {
        self.glob_patterns.iter().any(|g| g.matches(path))
    }

    /// Scan content for sensitive patterns. Returns all matches found.
    pub fn scan_content(&self, content: &str) -> Vec<SensitiveMatch> {
        let mut matches = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            // Check builtin patterns
            for (name, regex) in &self.builtin_patterns {
                if regex.is_match(line) {
                    matches.push(SensitiveMatch {
                        pattern: name.clone(),
                        line_number: line_num + 1,
                        redacted_preview: redact_line(line, regex),
                    });
                }
            }

            // Check user-configured regex patterns
            for (idx, regex) in self.regex_patterns.iter().enumerate() {
                if regex.is_match(line) {
                    matches.push(SensitiveMatch {
                        pattern: format!("user_regex_{}", idx),
                        line_number: line_num + 1,
                        redacted_preview: redact_line(line, regex),
                    });
                }
            }
        }

        matches
    }

    /// Apply the configured privacy policy to content.
    pub fn apply_policy(&self, content: &str, path: Option<&str>) -> PrivacyResult {
        if !self.config.enabled {
            return PrivacyResult::Clean(content.to_string());
        }

        // Check path-based sensitivity
        let path_sensitive = path.is_some_and(|p| self.is_sensitive_path(p));

        // Check content-based sensitivity
        let matches = self.scan_content(content);
        let content_sensitive = !matches.is_empty();

        if !path_sensitive && !content_sensitive {
            return PrivacyResult::Clean(content.to_string());
        }

        match self.config.policy {
            PrivacyPolicy::BlockCloud => PrivacyResult::Blocked,
            PrivacyPolicy::Redact => {
                let redacted = self.redact(content);
                PrivacyResult::Redacted(redacted)
            }
            PrivacyPolicy::LocalOnlySummary => {
                let summary = format!(
                    "[REDACTED: Sensitive content detected ({} patterns matched). \
                     Path sensitive: {}. Use local model for this content.]",
                    matches.len(),
                    path_sensitive
                );
                PrivacyResult::LocalSummary(summary)
            }
        }
    }

    /// Redact all sensitive patterns in content.
    pub fn redact(&self, content: &str) -> String {
        let mut result = content.to_string();

        for (name, regex) in &self.builtin_patterns {
            result = regex
                .replace_all(&result, format!("[REDACTED:{}]", name).as_str())
                .to_string();
        }

        for (idx, regex) in self.regex_patterns.iter().enumerate() {
            result = regex
                .replace_all(&result, format!("[REDACTED:user_regex_{}]", idx).as_str())
                .to_string();
        }

        result
    }

    /// Whether to redact content in session logs.
    pub fn should_redact_logs(&self) -> bool {
        self.config.enabled && !self.config.store_raw_in_logs
    }
}

fn redact_line(line: &str, regex: &Regex) -> String {
    regex.replace_all(line, "[REDACTED]").to_string()
}

/// Build builtin secret detection patterns, reusing patterns from output_scanner.
fn build_builtin_secret_patterns() -> Vec<(String, Regex)> {
    let patterns = vec![
        ("api_key_sk", r"sk-[a-zA-Z0-9]{20,}"),
        ("aws_key", r"AKIA[0-9A-Z]{16}"),
        ("github_token", r"ghp_[a-zA-Z0-9]{36}"),
        ("gitlab_token", r"glpat-[a-zA-Z0-9\-]{20,}"),
        ("private_key", r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----"),
        (
            "postgres_conn",
            r#"postgres(ql)?://[^\s'"]+:[^\s'"]+@[^\s'"]+"#,
        ),
        ("mysql_conn", r#"mysql://[^\s'"]+:[^\s'"]+@[^\s'"]+"#),
        (
            "mongodb_conn",
            r#"mongodb(\+srv)?://[^\s'"]+:[^\s'"]+@[^\s'"]+"#,
        ),
        ("redis_conn", r#"redis://[^\s'"]+:[^\s'"]+@[^\s'"]+"#),
        (
            "generic_secret_assign",
            r#"(?i)(api[_-]?key|secret[_-]?key|auth[_-]?token|password)\s*[:=]\s*['"]?[a-zA-Z0-9_\-/+=]{16,}['"]?"#,
        ),
    ];

    patterns
        .into_iter()
        .filter_map(|(name, pattern)| Regex::new(pattern).ok().map(|r| (name.to_string(), r)))
        .collect()
}
