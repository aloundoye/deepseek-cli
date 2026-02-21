//! Enhanced error constructors for DeepSeek CLI.

use deepseek_errors::{EnhancedError, ErrorType};

/// Common error constructors using deepseek-errors.
pub mod errors {
    use super::*;

    /// Create a configuration error.
    pub fn config_error(message: impl Into<String>) -> EnhancedError {
        EnhancedError::new("Configuration Error", message, ErrorType::Configuration)
    }

    /// Create a permission error.
    pub fn permission_error(message: impl Into<String>) -> EnhancedError {
        EnhancedError::new("Permission Error", message, ErrorType::Permission)
    }
}
