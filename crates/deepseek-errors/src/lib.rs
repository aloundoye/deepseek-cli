//! Enhanced error handling and user guidance for DeepSeek CLI
//!
//! This module provides improved error messages, user guidance, and
//! error recovery suggestions to make the CLI more user-friendly.

use anyhow::Error;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Enhanced error with user-friendly message and recovery suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedError {
    pub title: String,
    pub message: String,
    pub suggestions: Vec<String>,
    pub error_type: ErrorType,
    pub context: Option<String>,
}

/// Types of errors for better categorization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ErrorType {
    /// Configuration errors (missing API key, invalid settings)
    Configuration,
    /// Network errors (timeout, connection issues)
    Network,
    /// Permission errors (file access, tool restrictions)
    Permission,
    /// Runtime errors (tool failures, parsing errors)
    Runtime,
    /// Validation errors (invalid input, constraints)
    Validation,
    /// Resource errors (memory, disk space)
    Resource,
    /// Unknown or uncategorized errors
    Unknown,
}

impl EnhancedError {
    /// Create a new enhanced error
    pub fn new(
        title: impl Into<String>,
        message: impl Into<String>,
        error_type: ErrorType,
    ) -> Self {
        Self {
            title: title.into(),
            message: message.into(),
            suggestions: Vec::new(),
            error_type,
            context: None,
        }
    }

    /// Add a recovery suggestion
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    /// Add multiple recovery suggestions
    pub fn with_suggestions(mut self, suggestions: Vec<String>) -> Self {
        self.suggestions.extend(suggestions);
        self
    }

    /// Add context information
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Convert to anyhow::Error
    pub fn into_error(self) -> Error {
        Error::new(self)
    }

    /// Format error for display
    pub fn format(&self, verbose: bool) -> String {
        let mut output = String::new();

        // Title
        output.push_str(&format!("{}: {}\n", self.error_type.emoji(), self.title));

        // Message
        output.push_str(&format!("  {}\n", self.message));

        // Context (if verbose)
        if verbose {
            if let Some(context) = &self.context {
                output.push_str(&format!("\n  Context: {}\n", context));
            }
        }

        // Suggestions
        if !self.suggestions.is_empty() {
            output.push_str("\n  Suggestions:\n");
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                output.push_str(&format!("    {}. {}\n", i + 1, suggestion));
            }
        }

        output
    }
}

impl ErrorType {
    /// Get emoji for error type
    pub fn emoji(&self) -> &'static str {
        match self {
            ErrorType::Configuration => "🔧",
            ErrorType::Network => "🌐",
            ErrorType::Permission => "🔒",
            ErrorType::Runtime => "⚡",
            ErrorType::Validation => "📋",
            ErrorType::Resource => "💾",
            ErrorType::Unknown => "❓",
        }
    }

    /// Get color code for error type
    pub fn color(&self) -> &'static str {
        match self {
            ErrorType::Configuration => "yellow",
            ErrorType::Network => "blue",
            ErrorType::Permission => "red",
            ErrorType::Runtime => "magenta",
            ErrorType::Validation => "cyan",
            ErrorType::Resource => "yellow",
            ErrorType::Unknown => "gray",
        }
    }
}

impl fmt::Display for EnhancedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format(false))
    }
}

impl std::error::Error for EnhancedError {}

/// Error handler for providing user-friendly error messages
pub struct ErrorHandler {
    verbose: bool,
    show_suggestions: bool,
}

impl ErrorHandler {
    /// Create a new error handler
    pub fn new() -> Self {
        Self {
            verbose: false,
            show_suggestions: true,
        }
    }

    /// Set verbose mode
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set whether to show suggestions
    pub fn show_suggestions(mut self, show: bool) -> Self {
        self.show_suggestions = show;
        self
    }

    /// Handle an error and print user-friendly message
    pub fn handle(&self, error: &Error) -> String {
        // Try to extract enhanced error
        if let Some(enhanced) = error.downcast_ref::<EnhancedError>() {
            return enhanced.format(self.verbose);
        }

        // Convert generic error to enhanced error
        let error_str = error.to_string();
        let enhanced = self.classify_error(&error_str);
        enhanced.format(self.verbose)
    }

    /// Classify error based on message patterns
    fn classify_error(&self, error_message: &str) -> EnhancedError {
        let lower_error = error_message.to_lowercase();

        // Configuration errors
        if lower_error.contains("api key") || lower_error.contains("configuration") {
            return EnhancedError::new(
                "Configuration Error",
                error_message,
                ErrorType::Configuration,
            )
            .with_suggestions(vec![
                "Check your .deepseek/settings.json file".to_string(),
                "Set the DEEPSEEK_API_KEY environment variable".to_string(),
                "Run `deepseek --init` to initialize configuration".to_string(),
            ]);
        }

        // Network errors
        if lower_error.contains("network")
            || lower_error.contains("timeout")
            || lower_error.contains("connection")
        {
            return EnhancedError::new("Network Error", error_message, ErrorType::Network)
                .with_suggestions(vec![
                    "Check your internet connection".to_string(),
                    "Verify the API endpoint is accessible".to_string(),
                    "Try again in a few moments".to_string(),
                ]);
        }

        // Permission errors
        if lower_error.contains("permission")
            || lower_error.contains("access")
            || lower_error.contains("denied")
        {
            return EnhancedError::new("Permission Error", error_message, ErrorType::Permission)
                .with_suggestions(vec![
                    "Check file permissions".to_string(),
                    "Run with appropriate user privileges".to_string(),
                    "Use --permission-mode flag to adjust permissions".to_string(),
                ]);
        }

        // Default unknown error
        EnhancedError::new("Error", error_message, ErrorType::Unknown)
            .with_suggestion("Check the documentation or report this issue".to_string())
    }
}

/// Common error constructors for frequently encountered errors
pub mod errors {
    use super::*;

    /// Missing API key error
    pub fn missing_api_key() -> EnhancedError {
        EnhancedError::new(
            "Missing API Key",
            "DeepSeek API key is required to use the service.",
            ErrorType::Configuration,
        )
        .with_suggestions(vec![
            "Set DEEPSEEK_API_KEY environment variable".to_string(),
            "Add 'api_key' to .deepseek/settings.json".to_string(),
            "Run `deepseek --init` to set up configuration".to_string(),
        ])
    }

    /// Network timeout error
    pub fn network_timeout() -> EnhancedError {
        EnhancedError::new(
            "Network Timeout",
            "The request timed out while connecting to the API.",
            ErrorType::Network,
        )
        .with_suggestions(vec![
            "Check your internet connection".to_string(),
            "Increase timeout settings if available".to_string(),
            "Try again later".to_string(),
        ])
    }

    /// File not found error
    pub fn file_not_found(path: &str) -> EnhancedError {
        EnhancedError::new(
            "File Not Found",
            &format!("The file '{}' does not exist.", path),
            ErrorType::Validation,
        )
        .with_suggestions(vec![
            "Check the file path".to_string(),
            "Verify the file exists".to_string(),
            "Use relative or absolute path".to_string(),
        ])
    }

    /// Invalid configuration error
    pub fn invalid_configuration(field: &str, value: &str) -> EnhancedError {
        EnhancedError::new(
            "Invalid Configuration",
            &format!("Invalid value '{}' for field '{}'.", value, field),
            ErrorType::Configuration,
        )
        .with_suggestions(vec![
            "Check .deepseek/settings.json".to_string(),
            "Refer to configuration documentation".to_string(),
            "Remove invalid settings and retry".to_string(),
        ])
    }

    /// Tool permission denied error
    pub fn tool_permission_denied(tool: &str) -> EnhancedError {
        EnhancedError::new(
            "Tool Permission Denied",
            &format!("Permission denied for tool '{}'.", tool),
            ErrorType::Permission,
        )
        .with_suggestions(vec![
            "Use --allowed-tools flag to allow this tool".to_string(),
            "Adjust permission mode with --permission-mode".to_string(),
            "Run with --dangerously-skip-permissions (not recommended)".to_string(),
        ])
    }
}

/// User guidance system for providing helpful tips
pub struct UserGuidance {
    tips: Vec<GuidanceTip>,
    last_shown: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidanceTip {
    pub id: String,
    pub title: String,
    pub message: String,
    pub context: Vec<String>, // When to show this tip
    pub frequency: u32,       // How often to show (1 = always, 10 = rarely)
}

impl UserGuidance {
    /// Create new user guidance system
    pub fn new() -> Self {
        let tips = Self::default_tips();
        Self {
            tips,
            last_shown: 0,
        }
    }

    /// Get a relevant tip based on context
    pub fn get_tip(&mut self, context: &str, user_level: UserLevel) -> Option<&GuidanceTip> {
        let lower_context = context.to_lowercase();

        // Find tips relevant to context
        let relevant_tips: Vec<&GuidanceTip> = self
            .tips
            .iter()
            .filter(|tip| {
                // Check if tip matches context
                tip.context.iter().any(|c| lower_context.contains(c)) ||
                // Or if it's a general tip for this user level
                tip.context.is_empty()
            })
            .filter(|tip| {
                // Filter by user level (simplified)
                match user_level {
                    UserLevel::Beginner => true,
                    UserLevel::Intermediate => tip.frequency <= 5,
                    UserLevel::Advanced => tip.frequency <= 2,
                }
            })
            .collect();

        if relevant_tips.is_empty() {
            return None;
        }

        // Simple round-robin selection
        let tip = relevant_tips[self.last_shown % relevant_tips.len()];
        self.last_shown += 1;

        Some(tip)
    }

    /// Add a custom tip
    pub fn add_tip(&mut self, tip: GuidanceTip) {
        self.tips.push(tip);
    }

    /// Default tips for new users
    fn default_tips() -> Vec<GuidanceTip> {
        vec![
            GuidanceTip {
                id: "use_at_mentions".to_string(),
                title: "Use @file references".to_string(),
                message: "You can reference files using @path syntax, e.g., @src/main.rs:10-20".to_string(),
                context: vec!["file".to_string(), "reference".to_string(), "context".to_string()],
                frequency: 3,
            },
            GuidanceTip {
                id: "context_management".to_string(),
                title: "Manage context window".to_string(),
                message: "Use --max-tokens to control context size, or let the system automatically optimize it.".to_string(),
                context: vec!["context".to_string(), "token".to_string(), "limit".to_string()],
                frequency: 4,
            },
            GuidanceTip {
                id: "permission_modes".to_string(),
                title: "Permission modes".to_string(),
                message: "Use --permission-mode ask/auto/locked to control tool permissions.".to_string(),
                context: vec!["permission".to_string(), "tool".to_string(), "approve".to_string()],
                frequency: 2,
            },
            GuidanceTip {
                id: "chrome_integration".to_string(),
                title: "Chrome integration".to_string(),
                message: "Use --chrome flag to enable browser automation for web development tasks.".to_string(),
                context: vec!["web".to_string(), "browser".to_string(), "chrome".to_string()],
                frequency: 5,
            },
            GuidanceTip {
                id: "json_output".to_string(),
                title: "JSON output mode".to_string(),
                message: "Use --json flag for machine-readable output in scripts and automation.".to_string(),
                context: vec!["json".to_string(), "output".to_string(), "script".to_string()],
                frequency: 3,
            },
        ]
    }
}

/// User experience level for personalized guidance
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UserLevel {
    Beginner,
    Intermediate,
    Advanced,
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    #[test]
    fn test_enhanced_error_formatting() {
        let error = EnhancedError::new("Test Error", "Something went wrong", ErrorType::Runtime)
            .with_suggestion("Try again")
            .with_suggestion("Check documentation");

        let formatted = error.format(false);
        assert!(formatted.contains("Test Error"));
        assert!(formatted.contains("Something went wrong"));
        assert!(formatted.contains("Suggestions:"));
    }

    #[test]
    fn test_error_handler() {
        let handler = ErrorHandler::new();
        let error = anyhow!("API key is missing");
        let output = handler.handle(&error);

        assert!(output.contains("Configuration Error"));
        assert!(output.contains("API key"));
    }

    #[test]
    fn test_into_error_preserves_enhanced_type() {
        let handler = ErrorHandler::new();
        let error = EnhancedError::new("Permission Error", "Denied", ErrorType::Permission)
            .with_suggestion("Retry with approval")
            .into_error();
        let output = handler.handle(&error);
        assert!(output.contains("Permission Error"));
        assert!(output.contains("Denied"));
    }

    #[test]
    fn test_user_guidance() {
        let mut guidance = UserGuidance::new();
        let tip = guidance.get_tip("I need to reference a file", UserLevel::Beginner);

        assert!(tip.is_some());
        let tip = tip.unwrap();
        assert!(tip.title.contains("@file"));
    }
}
