# DeepSeek Context

Smart context management for DeepSeek CLI. This crate provides intelligent context management features including automatic relevant file detection, dependency analysis, and context compression.

## Features

- **Automatic Relevant File Detection**: Analyzes imports and dependencies to suggest relevant files
- **Dependency Graph Analysis**: Builds a graph of file relationships to understand project structure
- **Context Compression**: Intelligently compresses context to fit within token limits
- **Recent File Tracking**: Tracks recently edited files for better suggestions
- **Multi-language Support**: Supports Rust, JavaScript/TypeScript, Python, Java, and Go

## Usage

```rust
use deepseek_context::{ContextManager, ContextSelector};

// Create a context manager for your workspace
let mut manager = ContextManager::new("/path/to/workspace")?;
manager.analyze_workspace()?;

// Track recently edited files
manager.track_recent_file("/path/to/workspace/src/main.rs".into());

// Get file suggestions based on query
let suggestions = manager.suggest_relevant_files("utils", 10);
for suggestion in suggestions {
    println!("{} (score: {})", suggestion.path.display(), suggestion.score);
    for reason in &suggestion.reasons {
        println!("  - {}", reason);
    }
}

// Get files related to a specific file
let related = manager.get_related_files("/path/to/workspace/src/main.rs", 2);

// Compress context to fit token limit
let compressed = manager.compress_context(&long_context, 1000);

// Use context selector for automatic @file suggestions
let mut selector = ContextSelector::new("/path/to/workspace")?;
let suggestions = selector.update_context("How do I use the utils module?");
```

## Architecture

### ContextManager
The main struct that manages file relationships and provides suggestion functionality.

### FileSuggestion
Represents a suggested file with a relevance score and reasons for the suggestion.

### ContextSelector
Higher-level interface that automatically suggests files based on user queries, including parsing @mentions.

## How It Works

1. **Workspace Analysis**: Scans the workspace for source code files
2. **Import Extraction**: Uses regex patterns to extract imports from files
3. **Graph Building**: Creates a directed graph of file dependencies
4. **Scoring Algorithm**: Scores files based on:
   - Name/path matching the query
   - Recent editing activity
   - Dependency centrality
   - Import relationships
5. **Context Compression**: Uses heuristic algorithms to reduce context size while preserving important information

## Integration with DeepSeek CLI

This crate is designed to integrate with the DeepSeek CLI to provide:
- Automatic @file suggestions in the TUI
- Smart context window management
- Improved file relevance detection for better AI assistance

## Testing

Run tests with:
```bash
cargo test
```

## License

Same as the main DeepSeek CLI project.