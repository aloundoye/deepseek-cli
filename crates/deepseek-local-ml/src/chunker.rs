use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Strategy for splitting source files into chunks.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ChunkStrategy {
    /// Fixed-size overlapping line windows (always available).
    #[default]
    Line,
    /// AST-aware boundaries using tree-sitter (requires `local-ml` feature).
    /// Falls back to `Line` for unsupported languages.
    Semantic,
}

/// Configuration for source code chunking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkConfig {
    /// Lines per chunk.
    pub chunk_lines: usize,
    /// Overlap lines between adjacent chunks.
    pub chunk_overlap: usize,
    /// Glob patterns to include (empty = include all).
    pub include_globs: Vec<String>,
    /// Glob patterns to exclude.
    pub exclude_globs: Vec<String>,
    /// Maximum file size in bytes to process.
    pub max_file_bytes: u64,
    /// Chunking strategy: `Line` (default) or `Semantic` (AST-aware, requires `local-ml`).
    pub strategy: ChunkStrategy,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_lines: 50,
            chunk_overlap: 10,
            include_globs: Vec::new(),
            exclude_globs: Vec::new(),
            max_file_bytes: 1_048_576, // 1 MB
            strategy: ChunkStrategy::Line,
        }
    }
}

/// A chunk of source code from a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub file_path: PathBuf,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub content_hash: String,
    pub language: String,
}

/// Manifest tracking file hashes for incremental chunking.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkManifest {
    pub file_hashes: BTreeMap<String, String>,
    pub chunk_count: usize,
    pub timestamp: u64,
}

/// Chunk a single file into overlapping windows (line-based) or
/// AST-aware boundaries (semantic, requires `local-ml`).
pub fn chunk_file(path: &Path, config: &ChunkConfig) -> Result<Vec<Chunk>> {
    let content = std::fs::read_to_string(path)?;

    if is_binary_content(&content) {
        return Ok(Vec::new());
    }

    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Ok(Vec::new());
    }

    let language = detect_language(path);

    // Try semantic chunking if requested
    #[cfg(feature = "local-ml")]
    if matches!(config.strategy, ChunkStrategy::Semantic) {
        if let Some(chunks) = semantic_chunk_file(path, &content, &lines, &language, config) {
            return Ok(chunks);
        }
        // Fall through to line-based for unsupported languages
    }

    chunk_file_line_based(path, &lines, &language, config)
}

/// Line-based chunking with overlapping windows.
fn chunk_file_line_based(
    path: &Path,
    lines: &[&str],
    language: &str,
    config: &ChunkConfig,
) -> Result<Vec<Chunk>> {
    let file_path_str = path.to_string_lossy().to_string();
    let mut chunks = Vec::new();

    let step = if config.chunk_lines > config.chunk_overlap {
        config.chunk_lines - config.chunk_overlap
    } else {
        1
    };

    let mut start = 0;
    while start < lines.len() {
        let end = (start + config.chunk_lines).min(lines.len());
        let chunk_content = lines[start..end].join("\n");
        let content_hash = compute_hash(&chunk_content);
        let id = compute_chunk_id(&file_path_str, start, &content_hash);

        chunks.push(Chunk {
            id,
            file_path: path.to_path_buf(),
            start_line: start + 1, // 1-indexed
            end_line: end,
            content: chunk_content,
            content_hash,
            language: language.to_string(),
        });

        if end >= lines.len() {
            break;
        }
        start += step;
    }

    Ok(chunks)
}

/// Semantic chunking using tree-sitter AST boundaries.
///
/// Chunks at function/class/method/impl boundaries. If a single definition
/// exceeds `chunk_lines`, falls back to line-based splitting within that span.
/// Returns `None` if the language is not supported by tree-sitter.
#[cfg(feature = "local-ml")]
fn semantic_chunk_file(
    path: &Path,
    content: &str,
    lines: &[&str],
    language: &str,
    config: &ChunkConfig,
) -> Option<Vec<Chunk>> {
    use tree_sitter::{Language, Parser};

    let ts_language: Language = match language {
        "rust" => tree_sitter_rust::LANGUAGE.into(),
        "python" => tree_sitter_python::LANGUAGE.into(),
        "javascript" => tree_sitter_javascript::LANGUAGE.into(),
        "typescript" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "go" => tree_sitter_go::LANGUAGE.into(),
        "java" => tree_sitter_java::LANGUAGE.into(),
        _ => return None, // Unsupported — fall back to line-based
    };

    let mut parser = Parser::new();
    parser.set_language(&ts_language).ok()?;
    let tree = parser.parse(content.as_bytes(), None)?;
    let root = tree.root_node();

    // Collect top-level definition boundaries
    let definition_kinds = match language {
        "rust" => &[
            "function_item",
            "impl_item",
            "struct_item",
            "enum_item",
            "trait_item",
            "mod_item",
            "const_item",
            "static_item",
            "type_item",
        ][..],
        "python" => &["function_definition", "class_definition"][..],
        "javascript" | "typescript" => &[
            "function_declaration",
            "class_declaration",
            "export_statement",
            "lexical_declaration",
        ][..],
        "go" => &["function_declaration", "method_declaration", "type_declaration"][..],
        "java" => &[
            "class_declaration",
            "method_declaration",
            "interface_declaration",
        ][..],
        _ => return None,
    };

    // Walk the AST to find definition boundaries (as line ranges)
    let mut boundaries: Vec<(usize, usize)> = Vec::new(); // (start_line, end_line) 0-indexed
    let mut cursor = root.walk();

    for child in root.children(&mut cursor) {
        if definition_kinds.contains(&child.kind()) {
            let start = child.start_position().row;
            let end = child.end_position().row;
            boundaries.push((start, end));
        }
    }

    if boundaries.is_empty() {
        return None; // No definitions found — fall back to line-based
    }

    // Sort by start line and merge into chunks
    boundaries.sort_by_key(|b| b.0);

    let file_path_str = path.to_string_lossy().to_string();
    let mut chunks = Vec::new();
    let mut current_start = 0usize;

    for (def_start, def_end) in &boundaries {
        // Add any gap before this definition
        if *def_start > current_start {
            // Include preamble (imports, comments) as a chunk
            let gap_lines = &lines[current_start..*def_start];
            let gap_content: String = gap_lines.join("\n");
            if !gap_content.trim().is_empty() {
                let content_hash = compute_hash(&gap_content);
                let id = compute_chunk_id(&file_path_str, current_start, &content_hash);
                chunks.push(Chunk {
                    id,
                    file_path: path.to_path_buf(),
                    start_line: current_start + 1,
                    end_line: *def_start,
                    content: gap_content,
                    content_hash,
                    language: language.to_string(),
                });
            }
        }

        let def_line_count = def_end - def_start + 1;
        if def_line_count <= config.chunk_lines {
            // Definition fits in one chunk
            let end = (*def_end + 1).min(lines.len());
            let chunk_content = lines[*def_start..end].join("\n");
            let content_hash = compute_hash(&chunk_content);
            let id = compute_chunk_id(&file_path_str, *def_start, &content_hash);
            chunks.push(Chunk {
                id,
                file_path: path.to_path_buf(),
                start_line: def_start + 1,
                end_line: end,
                content: chunk_content,
                content_hash,
                language: language.to_string(),
            });
        } else {
            // Definition too large — split with overlapping windows within it
            let step = if config.chunk_lines > config.chunk_overlap {
                config.chunk_lines - config.chunk_overlap
            } else {
                1
            };
            let mut sub_start = *def_start;
            let def_end_line = (*def_end + 1).min(lines.len());
            while sub_start < def_end_line {
                let sub_end = (sub_start + config.chunk_lines).min(def_end_line);
                let chunk_content = lines[sub_start..sub_end].join("\n");
                let content_hash = compute_hash(&chunk_content);
                let id = compute_chunk_id(&file_path_str, sub_start, &content_hash);
                chunks.push(Chunk {
                    id,
                    file_path: path.to_path_buf(),
                    start_line: sub_start + 1,
                    end_line: sub_end,
                    content: chunk_content,
                    content_hash,
                    language: language.to_string(),
                });
                if sub_end >= def_end_line {
                    break;
                }
                sub_start += step;
            }
        }

        current_start = def_end + 1;
    }

    // Add trailing content after last definition
    if current_start < lines.len() {
        let trail_content = lines[current_start..].join("\n");
        if !trail_content.trim().is_empty() {
            let content_hash = compute_hash(&trail_content);
            let id = compute_chunk_id(&file_path_str, current_start, &content_hash);
            chunks.push(Chunk {
                id,
                file_path: path.to_path_buf(),
                start_line: current_start + 1,
                end_line: lines.len(),
                content: trail_content,
                content_hash,
                language: language.to_string(),
            });
        }
    }

    Some(chunks)
}

/// Chunk all files in a workspace, respecting .gitignore.
pub fn chunk_workspace(workspace: &Path, config: &ChunkConfig) -> Result<Vec<Chunk>> {
    let mut all_chunks = Vec::new();

    let walker = ignore::WalkBuilder::new(workspace)
        .hidden(true)
        .git_ignore(true)
        .build();

    for entry in walker.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        // Check file size
        if let Ok(meta) = path.metadata()
            && meta.len() > config.max_file_bytes
        {
            continue;
        }

        // Check exclude/include globs
        if should_skip_path(path, config) {
            continue;
        }

        match chunk_file(path, config) {
            Ok(chunks) => all_chunks.extend(chunks),
            Err(_) => continue, // Skip files that can't be read (binary, permission, etc.)
        }
    }

    Ok(all_chunks)
}

/// Incremental chunking: only re-chunk files that changed since the previous manifest.
pub fn chunk_workspace_incremental(
    workspace: &Path,
    config: &ChunkConfig,
    previous: &ChunkManifest,
) -> Result<(Vec<Chunk>, ChunkManifest)> {
    let mut all_chunks = Vec::new();
    let mut new_manifest = ChunkManifest {
        file_hashes: BTreeMap::new(),
        chunk_count: 0,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };

    let walker = ignore::WalkBuilder::new(workspace)
        .hidden(true)
        .git_ignore(true)
        .build();

    for entry in walker.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        if let Ok(meta) = path.metadata()
            && meta.len() > config.max_file_bytes
        {
            continue;
        }

        if should_skip_path(path, config) {
            continue;
        }

        let path_str = path.to_string_lossy().to_string();
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let file_hash = compute_hash(&content);
        new_manifest
            .file_hashes
            .insert(path_str.clone(), file_hash.clone());

        // Skip if unchanged
        if previous.file_hashes.get(&path_str) == Some(&file_hash) {
            continue;
        }

        match chunk_file(path, config) {
            Ok(chunks) => all_chunks.extend(chunks),
            Err(_) => continue,
        }
    }

    new_manifest.chunk_count = all_chunks.len();
    Ok((all_chunks, new_manifest))
}

/// Detect programming language from file extension.
pub fn detect_language(path: &Path) -> String {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();
    match ext {
        "rs" => "rust",
        "py" => "python",
        "js" | "jsx" => "javascript",
        "ts" | "tsx" => "typescript",
        "go" => "go",
        "java" => "java",
        "c" | "h" => "c",
        "cpp" | "cc" | "cxx" | "hpp" => "cpp",
        "cs" => "csharp",
        "rb" => "ruby",
        "swift" => "swift",
        "kt" | "kts" => "kotlin",
        "zig" => "zig",
        "hs" => "haskell",
        "ml" | "mli" => "ocaml",
        "sh" | "bash" | "zsh" => "shell",
        "toml" => "toml",
        "yaml" | "yml" => "yaml",
        "json" => "json",
        "md" | "markdown" => "markdown",
        _ => "unknown",
    }
    .to_string()
}

/// Check if content appears to be binary (contains null bytes).
pub fn is_binary_content(content: &str) -> bool {
    content.as_bytes().iter().take(8192).any(|&b| b == 0)
}

/// Check if a file path appears to be a binary file based on extension.
pub fn is_binary_file(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    matches!(
        ext.as_str(),
        "png"
            | "jpg"
            | "jpeg"
            | "gif"
            | "bmp"
            | "ico"
            | "svg"
            | "webp"
            | "mp3"
            | "mp4"
            | "avi"
            | "mov"
            | "wav"
            | "ogg"
            | "pdf"
            | "zip"
            | "tar"
            | "gz"
            | "bz2"
            | "xz"
            | "7z"
            | "rar"
            | "exe"
            | "dll"
            | "so"
            | "dylib"
            | "o"
            | "a"
            | "class"
            | "wasm"
            | "pyc"
            | "pyo"
    )
}

fn compute_hash(content: &str) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(content.as_bytes());
    format!("{:x}", hash)
}

fn compute_chunk_id(file_path: &str, start_line: usize, content_hash: &str) -> String {
    use sha2::{Digest, Sha256};
    let input = format!("{}:{}:{}", file_path, start_line, content_hash);
    let hash = Sha256::digest(input.as_bytes());
    format!("{:x}", hash)
}

fn should_skip_path(path: &Path, config: &ChunkConfig) -> bool {
    if is_binary_file(path) {
        return true;
    }

    let path_str = path.to_string_lossy();

    // Check exclude globs
    for pattern in &config.exclude_globs {
        if let Ok(glob) = glob::Pattern::new(pattern)
            && glob.matches(&path_str)
        {
            return true;
        }
    }

    // Check include globs (if specified, only include matching files)
    if !config.include_globs.is_empty() {
        let included = config.include_globs.iter().any(|pattern| {
            glob::Pattern::new(pattern)
                .map(|g| g.matches(&path_str))
                .unwrap_or(false)
        });
        if !included {
            return true;
        }
    }

    false
}
