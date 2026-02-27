use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_lines: 50,
            chunk_overlap: 10,
            include_globs: Vec::new(),
            exclude_globs: Vec::new(),
            max_file_bytes: 1_048_576, // 1 MB
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

/// Chunk a single file into overlapping windows.
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
            language: language.clone(),
        });

        if end >= lines.len() {
            break;
        }
        start += step;
    }

    Ok(chunks)
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
        if let Ok(meta) = path.metadata() {
            if meta.len() > config.max_file_bytes {
                continue;
            }
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

        if let Ok(meta) = path.metadata() {
            if meta.len() > config.max_file_bytes {
                continue;
            }
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
        if let Ok(glob) = glob::Pattern::new(pattern) {
            if glob.matches(&path_str) {
                return true;
            }
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
