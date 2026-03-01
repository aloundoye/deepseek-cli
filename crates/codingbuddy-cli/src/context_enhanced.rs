//! Enhanced error constructors for CodingBuddy.

use codingbuddy_errors::{EnhancedError, ErrorType};

/// Common error constructors using codingbuddy-errors.
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
