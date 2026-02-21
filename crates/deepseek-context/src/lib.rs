//! Smart context management for DeepSeek CLI
//!
//! This module provides intelligent context management features:
//! - Automatic relevant file detection based on imports and dependencies
//! - Context window optimization and compression
//! - File relationship analysis and dependency graph building

use anyhow::Result;
use ignore::WalkBuilder;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Represents a file suggestion with relevance score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSuggestion {
    pub path: PathBuf,
    pub score: f32,
    pub reasons: Vec<String>,
}

/// Context manager for intelligent file selection
pub struct ContextManager {
    workspace_root: PathBuf,
    file_graph: DiGraph<PathBuf, ()>,
    node_indices: HashMap<PathBuf, NodeIndex>,
    import_patterns: HashMap<String, Regex>,
    recent_files: Vec<PathBuf>,
    max_recent_files: usize,
}

impl ContextManager {
    /// Create a new context manager for the given workspace
    pub fn new(workspace_root: impl AsRef<Path>) -> Result<Self> {
        let workspace_root = workspace_root.as_ref().to_path_buf();

        let import_patterns = Self::build_import_patterns();

        Ok(Self {
            workspace_root,
            file_graph: DiGraph::new(),
            node_indices: HashMap::new(),
            import_patterns,
            recent_files: Vec::new(),
            max_recent_files: 50,
        })
    }

    /// Build regex patterns for detecting imports in different languages
    fn build_import_patterns() -> HashMap<String, Regex> {
        let mut patterns = HashMap::new();

        let rust =
            Regex::new(r#"^\s*(?:pub\s+)?(?:use|crate|mod|extern\s+crate)\s+([\w_:]+)"#).unwrap();
        patterns.insert("rs".to_string(), rust);

        let javascript =
            Regex::new(r#"^\s*(?:import|export|require)\s*(?:\(|\{)?\s*['\"]([^'\"]+)['\"]"#)
                .unwrap();
        patterns.insert("js".to_string(), javascript.clone());
        patterns.insert("ts".to_string(), javascript.clone());
        patterns.insert("jsx".to_string(), javascript.clone());
        patterns.insert("tsx".to_string(), javascript);

        let python = Regex::new(r#"^\s*(?:import|from)\s+([\w\.]+)"#).unwrap();
        patterns.insert("py".to_string(), python);

        let java = Regex::new(r#"^\s*(?:import|package)\s+([\w\.]+)"#).unwrap();
        patterns.insert("java".to_string(), java);

        // Matches both:
        // import "fmt"
        // import (
        //   "fmt"
        // )
        let go = Regex::new(r#"^\s*(?:import\s+)?(?:[\w\.]+\s+)?["`]([^"`]+)["`]\s*$"#).unwrap();
        patterns.insert("go".to_string(), go);

        patterns
    }

    /// Analyze the workspace and build dependency graph
    pub fn analyze_workspace(&mut self) -> Result<()> {
        // Rebuild from scratch to avoid duplicate/stale nodes across re-analysis.
        self.file_graph = DiGraph::new();
        self.node_indices.clear();

        let walker = WalkBuilder::new(&self.workspace_root).hidden(false).build();

        // First pass: add all files as nodes
        for entry in walker {
            let entry = entry?;
            if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                let path = Self::normalize_existing_path(entry.path());
                if self.should_include_file(&path) {
                    let idx = self.file_graph.add_node(path.clone());
                    self.node_indices.insert(path, idx);
                }
            }
        }

        // Second pass: analyze imports and add edges
        for (path, &node_idx) in &self.node_indices {
            if let Ok(content) = std::fs::read_to_string(path) {
                let imports = self.extract_imports(path, &content);
                for import in imports {
                    if let Some(&import_idx) = self.node_indices.get(&import) {
                        self.file_graph.add_edge(node_idx, import_idx, ());
                    }
                }
            }
        }

        Ok(())
    }

    /// Number of indexed files in the dependency graph.
    pub fn file_count(&self) -> usize {
        self.node_indices.len()
    }

    /// Check if a file should be included in analysis
    fn should_include_file(&self, path: &Path) -> bool {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();

        // Include source code files
        matches!(
            ext.as_str(),
            "rs" | "js" | "ts" | "jsx" | "tsx" | "py" | "java" | "go" | "c" | "cpp" | "h" | "hpp"
        )
    }

    /// Extract imports from file content
    fn extract_imports(&self, path: &Path, content: &str) -> Vec<PathBuf> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        let mut imports = Vec::new();

        for line in content.lines() {
            if let Some(pattern) = self.import_patterns.get(ext.as_str()) {
                if let Some(caps) = pattern.captures(line) {
                    if let Some(import) = caps.get(1) {
                        let import_str = import.as_str();

                        // Try to resolve import to actual file
                        if let Some(resolved) = self.resolve_import(path, import_str) {
                            imports.push(resolved);
                        }
                    }
                }
            }
        }

        imports
    }

    /// Resolve import string to actual file path
    fn resolve_import(&self, from_path: &Path, import: &str) -> Option<PathBuf> {
        if from_path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("rs"))
            && let Some(path) = self.resolve_rust_import(from_path, import)
        {
            return Some(path);
        }

        let from_dir = from_path.parent()?;
        let import = import.trim().trim_matches('"').trim_matches('`');

        // Try different resolution strategies
        let candidates = [
            // Direct relative path
            from_dir.join(import),
            // With extension
            from_dir.join(format!("{}.rs", import)),
            from_dir.join(format!("{}.js", import)),
            from_dir.join(format!("{}.ts", import)),
            from_dir.join(format!("{}.py", import)),
            // Module/index file
            from_dir.join(import).join("mod.rs"),
            from_dir.join(import).join("index.js"),
            from_dir.join(import).join("index.ts"),
            from_dir.join(import).join("__init__.py"),
        ];

        for candidate in candidates {
            if candidate.exists() {
                return Some(Self::normalize_existing_path(&candidate));
            }
        }

        None
    }

    fn resolve_rust_import(&self, from_path: &Path, import: &str) -> Option<PathBuf> {
        let mut import = import.trim();
        let mut base_dir = from_path.parent()?.to_path_buf();

        if let Some(rest) = import.strip_prefix("crate::") {
            import = rest;
            base_dir = self.workspace_root.join("src");
        } else if let Some(rest) = import.strip_prefix("self::") {
            import = rest;
        } else {
            while let Some(rest) = import.strip_prefix("super::") {
                import = rest;
                base_dir = base_dir.parent()?.to_path_buf();
            }
        }

        let module_path = import.replace("::", "/");
        let candidates = [
            base_dir.join(format!("{module_path}.rs")),
            base_dir.join(&module_path).join("mod.rs"),
            base_dir.join(&module_path),
        ];

        for candidate in candidates {
            if candidate.exists() {
                return Some(Self::normalize_existing_path(&candidate));
            }
        }
        None
    }

    fn normalize_existing_path(path: &Path) -> PathBuf {
        path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
    }

    fn normalize_lookup_path(&self, path: &Path) -> PathBuf {
        let base = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.workspace_root.join(path)
        };
        Self::normalize_existing_path(&base)
    }

    /// Track a recently edited file
    pub fn track_recent_file(&mut self, path: PathBuf) {
        let path = self.normalize_lookup_path(&path);
        // Remove if already exists
        self.recent_files.retain(|p| *p != path);

        // Add to front
        self.recent_files.insert(0, path);

        // Trim if too many
        if self.recent_files.len() > self.max_recent_files {
            self.recent_files.truncate(self.max_recent_files);
        }
    }

    /// Suggest relevant files based on query and context
    pub fn suggest_relevant_files(&self, query: &str, limit: usize) -> Vec<FileSuggestion> {
        let mut suggestions = Vec::new();
        let normalized_query = query.trim().to_ascii_lowercase();

        // Score files based on various factors
        for (path, &node_idx) in &self.node_indices {
            let mut score = 0.0;
            let mut reasons = Vec::new();

            // 1. Check if file name matches query
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !normalized_query.is_empty()
                && file_name.to_ascii_lowercase().contains(&normalized_query)
            {
                score += 2.0;
                reasons.push("File name matches query".to_string());
            }

            // 2. Check if path contains query
            let path_str = path.to_string_lossy().to_ascii_lowercase();
            if !normalized_query.is_empty() && path_str.contains(&normalized_query) {
                score += 1.5;
                reasons.push("Path contains query".to_string());
            }

            // 3. Check recent files
            if self.recent_files.contains(path) {
                score += 1.0;
                reasons.push("Recently edited".to_string());
            }

            // 4. Check dependency centrality
            let in_degree = self
                .file_graph
                .edges_directed(node_idx, petgraph::Direction::Incoming)
                .count();
            let out_degree = self
                .file_graph
                .edges_directed(node_idx, petgraph::Direction::Outgoing)
                .count();
            let centrality = (in_degree + out_degree) as f32;
            if centrality > 0.0 {
                score += centrality * 0.1;
                reasons.push(format!("High dependency centrality ({})", centrality));
            }

            if score > 0.0 {
                suggestions.push(FileSuggestion {
                    path: path.clone(),
                    score,
                    reasons,
                });
            }
        }

        // Sort by score descending
        suggestions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Take top N
        suggestions.truncate(limit);

        suggestions
    }

    /// Get files related to a specific file
    pub fn get_related_files(&self, file_path: &Path, depth: usize) -> Vec<PathBuf> {
        let mut related = HashSet::new();
        let lookup_path = self.normalize_lookup_path(file_path);

        if let Some(&node_idx) = self.node_indices.get(&lookup_path) {
            // Get dependencies (files this file imports)
            let mut stack = vec![(node_idx, 0)];

            while let Some((current_idx, current_depth)) = stack.pop() {
                if current_depth > depth {
                    continue;
                }

                // Add to related
                let current_path = &self.file_graph[current_idx];
                related.insert(current_path.clone());

                // Add dependencies
                for edge in self
                    .file_graph
                    .edges_directed(current_idx, petgraph::Direction::Outgoing)
                {
                    let target_idx = edge.target();
                    let target_path = &self.file_graph[target_idx];
                    if !related.contains(target_path) {
                        stack.push((target_idx, current_depth + 1));
                    }
                }

                // Add dependents (files that import this file)
                for edge in self
                    .file_graph
                    .edges_directed(current_idx, petgraph::Direction::Incoming)
                {
                    let source_idx = edge.source();
                    let source_path = &self.file_graph[source_idx];
                    if !related.contains(source_path) {
                        stack.push((source_idx, current_depth + 1));
                    }
                }
            }
        }

        related.into_iter().collect()
    }

    /// Compress context to fit within token limit
    pub fn compress_context(&self, context: &str, max_tokens: usize) -> String {
        // Simple compression strategy: keep most relevant parts
        // In a real implementation, this would use token counting and LLM-based summarization

        let lines: Vec<&str> = context.lines().collect();
        let total_lines = lines.len();

        if total_lines <= max_tokens / 10 {
            // Rough estimate: 10 lines per token
            return context.to_string();
        }

        // Keep beginning and end, with some middle context
        let keep_start = (max_tokens / 30).min(total_lines / 4);
        let keep_end = (max_tokens / 30).min(total_lines / 4);
        let keep_middle = max_tokens / 30;

        let mut compressed = Vec::new();

        // Add start
        compressed.extend(lines.iter().take(keep_start).cloned());

        // Add middle section (every Nth line)
        let middle_start = total_lines / 3;
        let middle_end = total_lines * 2 / 3;
        let step = (middle_end - middle_start) / keep_middle.max(1);

        for i in (middle_start..middle_end).step_by(step.max(1)) {
            if i < lines.len() {
                compressed.push(lines[i]);
            }
        }

        // Add end
        compressed.extend(lines.iter().rev().take(keep_end).rev().cloned());

        // Add compression notice
        if compressed.len() < lines.len() {
            compressed.insert(
                0,
                "// Context compressed for brevity. Full context available upon request.",
            );
        }

        compressed.join("\n")
    }
}

/// Smart context selector for automatic @file suggestions
pub struct ContextSelector {
    manager: ContextManager,
    last_query: String,
}

impl ContextSelector {
    /// Create a new context selector
    pub fn new(workspace_root: impl AsRef<Path>) -> Result<Self> {
        let mut manager = ContextManager::new(workspace_root)?;
        manager.analyze_workspace()?;

        Ok(Self {
            manager,
            last_query: String::new(),
        })
    }

    /// Update context based on user query
    pub fn update_context(&mut self, query: &str) -> Vec<FileSuggestion> {
        self.last_query = query.to_string();

        if query.is_empty() {
            return Vec::new();
        }

        // Extract potential file references from query
        let file_pattern = Regex::new(r"@([\w\./\-_]+)").unwrap();
        let mut suggestions = Vec::new();

        for cap in file_pattern.captures_iter(query) {
            if let Some(file_ref) = cap.get(1) {
                let file_ref_str = file_ref.as_str();
                let file_suggestions = self.manager.suggest_relevant_files(file_ref_str, 5);
                suggestions.extend(file_suggestions);
            }
        }

        // If no @ mentions, suggest based on whole query
        if suggestions.is_empty() {
            suggestions = self.manager.suggest_relevant_files(query, 10);
        }

        suggestions
    }

    /// Get context manager reference
    pub fn manager(&self) -> &ContextManager {
        &self.manager
    }

    /// Track a recently edited file in selector state.
    pub fn track_recent_file(&mut self, path: PathBuf) {
        self.manager.track_recent_file(path);
    }

    /// Re-analyze workspace and rebuild dependency graph.
    pub fn analyze_workspace(&mut self) -> Result<()> {
        self.manager.analyze_workspace()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_context_manager_basic() -> Result<()> {
        let temp_dir = TempDir::new()?;

        // Create test files
        let main_rs = temp_dir.path().join("src").join("main.rs");
        fs::create_dir_all(temp_dir.path().join("src"))?;
        fs::write(&main_rs, "use crate::utils;\nfn main() {}\n")?;

        let utils_rs = temp_dir.path().join("src").join("utils.rs");
        fs::write(&utils_rs, "pub fn helper() {}\n")?;

        let mut manager = ContextManager::new(temp_dir.path())?;
        manager.analyze_workspace()?;

        // Test file suggestions
        let suggestions = manager.suggest_relevant_files("utils", 5);
        assert!(!suggestions.is_empty());

        // Test related files
        let related = manager.get_related_files(&main_rs, 1);
        let normalized_utils = utils_rs.canonicalize()?;
        assert!(related.contains(&normalized_utils));

        Ok(())
    }

    #[test]
    fn test_context_compression() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let manager = ContextManager::new(temp_dir.path())?;

        // Create a long context
        let mut long_context = String::new();
        for i in 0..100 {
            long_context.push_str(&format!("// Line {}\n", i));
        }

        let compressed = manager.compress_context(&long_context, 50);
        let compressed_lines = compressed.lines().count();

        // Should be significantly shorter
        assert!(compressed_lines < 100);
        assert!(compressed.contains("compressed"));

        Ok(())
    }
}
