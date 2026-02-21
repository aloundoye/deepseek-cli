# DeepSeek Errors

Enhanced error handling and user guidance for DeepSeek CLI. This crate provides user-friendly error messages, recovery suggestions, and contextual guidance to improve the user experience.

## Features

- **Enhanced Error Messages**: Clear, actionable error messages with recovery suggestions
- **Error Classification**: Automatic categorization of errors (Configuration, Network, Permission, etc.)
- **User Guidance System**: Contextual tips and suggestions based on user actions
- **Colorful Output**: Emoji and color-coded error messages for better readability
- **Recovery Suggestions**: Actionable steps to resolve common issues

## Usage

### Basic Error Handling

```rust
use deepseek_errors::{EnhancedError, ErrorType, ErrorHandler};

// Create an enhanced error
let error = EnhancedError::new(
    "Missing API Key",
    "DeepSeek API key is required to use the service.",
    ErrorType::Configuration,
)
.with_suggestions(vec![
    "Set DEEPSEEK_API_KEY environment variable".to_string(),
    "Add 'api_key' to .deepseek/settings.json".to_string(),
    "Run `deepseek --init` to set up configuration".to_string(),
]);

// Format error for display
println!("{}", error.format(true));

// Convert to anyhow::Error
let anyhow_error = error.into_error();
```

### Using Error Handler

```rust
use deepseek_errors::ErrorHandler;
use anyhow::anyhow;

let handler = ErrorHandler::new()
    .verbose(true)
    .show_suggestions(true);

let error = anyhow!("API key is missing or invalid");
let user_friendly_message = handler.handle(&error);
println!("{}", user_friendly_message);
```

### Predefined Common Errors

```rust
use deepseek_errors::errors;

// Common error constructors
let missing_api_key = errors::missing_api_key();
let network_timeout = errors::network_timeout();
let file_not_found = errors::file_not_found("/path/to/file");
let invalid_config = errors::invalid_configuration("llm.provider", "invalid");
let tool_denied = errors::tool_permission_denied("bash_run");
```

### User Guidance System

```rust
use deepseek_errors::{UserGuidance, UserLevel};

let mut guidance = UserGuidance::new();

// Get contextual tip
if let Some(tip) = guidance.get_tip("I need to reference a file", UserLevel::Beginner) {
    println!("Tip: {} - {}", tip.title, tip.message);
}

// Add custom tip
guidance.add_tip(GuidanceTip {
    id: "custom_tip".to_string(),
    title: "Custom Tip".to_string(),
    message: "This is a custom guidance tip.".to_string(),
    context: vec!["custom".to_string(), "context".to_string()],
    frequency: 1,
});
```

## Error Types

| Type | Emoji | Description | Common Causes |
|------|-------|-------------|---------------|
| Configuration | 🔧 | Issues with settings and configuration | Missing API key, invalid settings |
| Network | 🌐 | Connectivity issues | Timeout, connection refused |
| Permission | 🔒 | Access control issues | File permissions, tool restrictions |
| Runtime | ⚡ | Execution errors | Tool failures, parsing errors |
| Validation | 📋 | Input validation errors | Invalid arguments, constraints |
| Resource | 💾 | Resource limitations | Memory, disk space, rate limits |
| Unknown | ❓ | Unclassified errors | Generic errors |

## Integration with DeepSeek CLI

This crate is designed to integrate with the DeepSeek CLI to provide:

1. **Better Error Messages**: Replace generic errors with actionable messages
2. **Recovery Guidance**: Suggest next steps when errors occur
3. **Contextual Help**: Provide tips based on user actions and context
4. **User Onboarding**: Guide new users through common workflows

## Example Output

```
🔧 Configuration Error: Missing API Key
  DeepSeek API key is required to use the service.

  Suggestions:
    1. Set DEEPSEEK_API_KEY environment variable
    2. Add 'api_key' to .deepseek/settings.json
    3. Run `deepseek --init` to set up configuration
```

## Testing

Run tests with:
```bash
cargo test
```

## License

Same as the main DeepSeek CLI project.