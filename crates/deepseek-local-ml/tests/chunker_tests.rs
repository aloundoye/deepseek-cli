use deepseek_local_ml::chunker::*;
use std::fs;
use tempfile::TempDir;

fn create_test_workspace() -> TempDir {
    let dir = TempDir::new().unwrap();

    // Initialize as a git repo so .gitignore is respected by ignore::WalkBuilder
    std::process::Command::new("git")
        .args(["init", "-q"])
        .current_dir(dir.path())
        .output()
        .ok();

    // Create a Rust file with enough lines for multiple chunks
    let mut content = String::new();
    for i in 1..=120 {
        content.push_str(&format!("// Line {} of test file\n", i));
    }
    fs::write(dir.path().join("main.rs"), &content).unwrap();

    // Create a Python file
    let mut py_content = String::new();
    for i in 1..=60 {
        py_content.push_str(&format!("# Line {} of python file\n", i));
    }
    fs::write(dir.path().join("script.py"), &py_content).unwrap();

    // Create a binary file (contains null bytes)
    fs::write(dir.path().join("image.png"), b"\x89PNG\r\n\x1a\n\x00\x00").unwrap();

    // Create a .gitignore
    fs::write(dir.path().join(".gitignore"), "target/\n*.log\n").unwrap();

    // Create an ignored directory
    let target = dir.path().join("target");
    fs::create_dir_all(&target).unwrap();
    fs::write(target.join("output.rs"), "// ignored file\n").unwrap();

    // Create a .env file
    fs::write(dir.path().join(".env"), "SECRET_KEY=abc123\n").unwrap();

    dir
}

#[test]
fn chunk_file_produces_stable_ids() {
    let dir = TempDir::new().unwrap();
    let mut content = String::new();
    for i in 1..=80 {
        content.push_str(&format!("fn line_{}() {{}}\n", i));
    }
    let path = dir.path().join("test.rs");
    fs::write(&path, &content).unwrap();

    let config = ChunkConfig::default();
    let chunks1 = chunk_file(&path, &config).unwrap();
    let chunks2 = chunk_file(&path, &config).unwrap();

    assert!(!chunks1.is_empty());
    assert_eq!(chunks1.len(), chunks2.len());
    for (c1, c2) in chunks1.iter().zip(chunks2.iter()) {
        assert_eq!(c1.id, c2.id, "chunk IDs must be deterministic");
        assert_eq!(c1.content_hash, c2.content_hash);
    }
}

#[test]
fn chunk_overlap_correct() {
    let dir = TempDir::new().unwrap();
    let mut content = String::new();
    for i in 1..=120 {
        content.push_str(&format!("line {}\n", i));
    }
    let path = dir.path().join("test.txt");
    fs::write(&path, &content).unwrap();

    let config = ChunkConfig {
        chunk_lines: 50,
        chunk_overlap: 10,
        ..Default::default()
    };

    let chunks = chunk_file(&path, &config).unwrap();
    assert!(chunks.len() >= 2, "should have at least 2 chunks");

    // Check overlap between first two chunks
    let c0_end = chunks[0].end_line;
    let c1_start = chunks[1].start_line;
    let overlap = c0_end as i64 - c1_start as i64 + 1;
    assert_eq!(
        overlap, config.chunk_overlap as i64,
        "adjacent chunks must share exactly {} overlap lines",
        config.chunk_overlap
    );
}

#[test]
fn chunk_respects_gitignore() {
    let workspace = create_test_workspace();
    let config = ChunkConfig::default();
    let chunks = chunk_workspace(workspace.path(), &config).unwrap();

    // Ensure no chunks from target/ directory
    let target_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| c.file_path.to_string_lossy().contains("target/"))
        .collect();
    assert!(
        target_chunks.is_empty(),
        "gitignored files should not produce chunks"
    );
}

#[test]
fn chunk_skips_binary_files() {
    let workspace = create_test_workspace();
    let config = ChunkConfig::default();
    let chunks = chunk_workspace(workspace.path(), &config).unwrap();

    let png_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| c.file_path.to_string_lossy().contains("image.png"))
        .collect();
    assert!(
        png_chunks.is_empty(),
        "binary files should be skipped"
    );
}

#[test]
fn incremental_only_changed_files() {
    let workspace = create_test_workspace();
    let config = ChunkConfig::default();

    // First full chunk
    let chunks1 = chunk_workspace(workspace.path(), &config).unwrap();
    assert!(!chunks1.is_empty());

    // Build a manifest from the first run by using chunk_workspace_incremental
    // with an empty manifest (which will re-chunk everything and produce the manifest)
    let empty_manifest = ChunkManifest::default();
    let (_, manifest) =
        chunk_workspace_incremental(workspace.path(), &config, &empty_manifest).unwrap();

    // Run incremental with no changes — should produce no new chunks
    let (new_chunks, _) =
        chunk_workspace_incremental(workspace.path(), &config, &manifest).unwrap();
    assert!(
        new_chunks.is_empty(),
        "unchanged workspace should produce no new chunks, got {}",
        new_chunks.len()
    );

    // Modify one file and re-run
    fs::write(
        workspace.path().join("main.rs"),
        "// MODIFIED\nfn main() {}\n",
    )
    .unwrap();
    let (new_chunks2, _) =
        chunk_workspace_incremental(workspace.path(), &config, &manifest).unwrap();
    assert!(
        !new_chunks2.is_empty(),
        "modified file should produce new chunks"
    );
    // Only the modified file should have chunks
    for c in &new_chunks2 {
        assert!(
            c.file_path.to_string_lossy().contains("main.rs"),
            "only changed file should be re-chunked"
        );
    }
}

#[test]
fn detect_language_from_extensions() {
    assert_eq!(detect_language(std::path::Path::new("foo.rs")), "rust");
    assert_eq!(detect_language(std::path::Path::new("bar.py")), "python");
    assert_eq!(detect_language(std::path::Path::new("baz.ts")), "typescript");
    assert_eq!(detect_language(std::path::Path::new("qux.go")), "go");
    assert_eq!(detect_language(std::path::Path::new("nope.xyz")), "unknown");
}
