//! Enhanced context management integration for DeepSeek CLI
//!
//! This module integrates the deepseek-context and deepseek-errors crates
//! to provide smarter context management and better error handling.

use anyhow::Result;
use deepseek_context::{ContextSelector, FileSuggestion};
use deepseek_errors::{EnhancedError, ErrorHandler, ErrorType, UserGuidance, UserLevel};
use std::path::{Path, PathBuf};

/// Enhanced context manager wrapper
pub struct EnhancedContext {
    selector: Option<ContextSelector>,
    error_handler: ErrorHandler,
    user_guidance: UserGuidance,
    workspace_root: PathBuf,
}

impl EnhancedContext {
    /// Create a new enhanced context manager
    pub fn new(workspace_root: impl AsRef<Path>) -> Result<Self> {
        let workspace_root = workspace_root.as_ref().to_path_buf();

        // Initialize context selector (may fail if workspace is empty)
        let selector = match ContextSelector::new(&workspace_root) {
            Ok(selector) => Some(selector),
            Err(e) => {
                // Log but don't fail - context management is optional
                eprintln!("Note: Context analysis unavailable: {}", e);
                None
            }
        };

        Ok(Self {
            selector,
            error_handler: ErrorHandler::new().verbose(false).show_suggestions(true),
            user_guidance: UserGuidance::new(),
            workspace_root,
        })
    }

    /// Get file suggestions based on query
    pub fn suggest_files(&mut self, query: &str, limit: usize) -> Vec<FileSuggestion> {
        if let Some(selector) = &mut self.selector {
            let mut suggestions = selector.update_context(query);
            suggestions.truncate(limit);
            suggestions
        } else {
            Vec::new()
        }
    }

    /// Track a recently edited file
    pub fn track_recent_file(&mut self, path: PathBuf) {
        if let Some(selector) = &mut self.selector {
            selector.track_recent_file(path);
        }
    }

    /// Handle an error with enhanced formatting
    pub fn handle_error(&self, error: &anyhow::Error) -> String {
        self.error_handler.handle(error)
    }

    /// Create an enhanced error
    pub fn create_error(
        &self,
        title: impl Into<String>,
        message: impl Into<String>,
        error_type: ErrorType,
    ) -> EnhancedError {
        EnhancedError::new(title, message, error_type)
    }

    /// Get user guidance tip based on context
    pub fn get_guidance_tip(&mut self, context: &str) -> Option<String> {
        self.user_guidance
            .get_tip(context, UserLevel::Beginner)
            .map(|tip| format!("💡 {}: {}", tip.title, tip.message))
    }

    /// Compress context to fit within token limit
    pub fn compress_context(&self, context: &str, max_tokens: usize) -> String {
        if let Some(selector) = &self.selector {
            selector.manager().compress_context(context, max_tokens)
        } else {
            // Fallback simple compression
            let lines: Vec<&str> = context.lines().collect();
            if lines.len() * 10 > max_tokens {
                // Rough estimate: 10 lines per token
                let keep = max_tokens / 10;
                let compressed: Vec<&str> = lines
                    .iter()
                    .take(keep / 2)
                    .chain(&["// ... context truncated ..."])
                    .chain(lines.iter().rev().take(keep / 2).rev())
                    .copied()
                    .collect();
                compressed.join("\n")
            } else {
                context.to_string()
            }
        }
    }

    /// Get related files for a specific file
    pub fn get_related_files(&self, file_path: &Path, depth: usize) -> Vec<PathBuf> {
        if let Some(selector) = &self.selector {
            selector.manager().get_related_files(file_path, depth)
        } else {
            Vec::new()
        }
    }

    /// Analyze workspace and build dependency graph
    pub fn analyze_workspace(&mut self) -> Result<()> {
        if let Some(selector) = &mut self.selector {
            selector.analyze_workspace()?;
        }
        Ok(())
    }
}

/// Initialize enhanced context for a workspace
pub fn init_enhanced_context(workspace_root: impl AsRef<Path>) -> Result<EnhancedContext> {
    EnhancedContext::new(workspace_root)
}

/// Common error constructors using deepseek-errors
pub mod errors {
    use super::*;

    /// Missing API key error
    pub fn missing_api_key() -> EnhancedError {
        deepseek_errors::errors::missing_api_key()
    }

    /// Network timeout error
    pub fn network_timeout() -> EnhancedError {
        deepseek_errors::errors::network_timeout()
    }

    /// File not found error
    pub fn file_not_found(path: &str) -> EnhancedError {
        deepseek_errors::errors::file_not_found(path)
    }

    /// Invalid configuration error
    pub fn invalid_configuration(field: &str, value: &str) -> EnhancedError {
        deepseek_errors::errors::invalid_configuration(field, value)
    }

    /// Tool permission denied error
    pub fn tool_permission_denied(tool: &str) -> EnhancedError {
        deepseek_errors::errors::tool_permission_denied(tool)
    }

    /// Create a configuration error
    pub fn config_error(message: impl Into<String>) -> EnhancedError {
        EnhancedError::new("Configuration Error", message, ErrorType::Configuration)
    }

    /// Create a network error
    pub fn network_error(message: impl Into<String>) -> EnhancedError {
        EnhancedError::new("Network Error", message, ErrorType::Network)
    }

    /// Create a permission error
    pub fn permission_error(message: impl Into<String>) -> EnhancedError {
        EnhancedError::new("Permission Error", message, ErrorType::Permission)
    }

    /// Create a runtime error
    pub fn runtime_error(message: impl Into<String>) -> EnhancedError {
        EnhancedError::new("Runtime Error", message, ErrorType::Runtime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_enhanced_context_creation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let context = EnhancedContext::new(temp_dir.path())?;

        // Should create without error even for empty workspace
        assert!(
            context
                .handle_error(&anyhow!("test error"))
                .contains("test error")
        );

        Ok(())
    }

    #[test]
    fn test_error_helpers() -> Result<()> {
        let missing_api = errors::missing_api_key();
        assert!(missing_api.title.contains("Missing API Key"));

        let file_not_found = errors::file_not_found("test.rs");
        assert!(file_not_found.message.contains("test.rs"));

        Ok(())
    }

    #[test]
    fn test_context_compression() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let context = EnhancedContext::new(temp_dir.path())?;

        let long_context = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10";
        let compressed = context.compress_context(long_context, 50);

        // Should compress when needed
        assert!(compressed.len() <= long_context.len());

        Ok(())
    }
}
