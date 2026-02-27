//! Integration tests for CLI commands and index operations (P8 Batch 6).

use deepseek_local_ml::chunker::ChunkConfig;
use deepseek_local_ml::embeddings::MockEmbeddings;
use deepseek_local_ml::retrieval::HybridRetriever;
use deepseek_local_ml::vector_index::SearchFilter;
use deepseek_local_ml::{PrivacyConfig, PrivacyPolicy, PrivacyRouter};
use std::sync::Arc;
use tempfile::TempDir;

#[test]
fn index_build_creates_vector_index() {
    let dir = TempDir::new().unwrap();
    let embeddings = Arc::new(MockEmbeddings::new(64));

    let ws = dir.path().join("workspace");
    std::fs::create_dir_all(&ws).unwrap();

    // Create enough content for chunking
    let mut content = String::new();
    for i in 1..=80 {
        content.push_str(&format!("fn function_{}() {{ /* body */ }}\n", i));
    }
    std::fs::write(ws.join("lib.rs"), &content).unwrap();

    let idx_path = dir.path().join("vector-index");
    let mut retriever = HybridRetriever::new(
        &idx_path,
        embeddings,
        None,
        0.7,
        ChunkConfig::default(),
    )
    .unwrap();

    let report = retriever.build_index(&ws).unwrap();
    assert!(report.chunks_indexed > 0, "should index chunks");
    assert!(report.files_processed > 0, "should process files");

    // Verify index directory was created
    assert!(idx_path.exists(), "vector index directory should exist");
}

#[test]
fn privacy_scan_finds_env_files() {
    let dir = TempDir::new().unwrap();
    let ws = dir.path().join("workspace");
    std::fs::create_dir_all(&ws).unwrap();

    // Create .env file with sensitive content
    std::fs::write(
        ws.join(".env"),
        "DATABASE_URL=postgres://user:password@localhost/db\nAPI_KEY=sk-secret123\n",
    )
    .unwrap();

    // Create a normal Rust file
    std::fs::write(ws.join("main.rs"), "fn main() { println!(\"hello\"); }\n").unwrap();

    let config = PrivacyConfig {
        enabled: true,
        sensitive_globs: vec!["*.env".to_string(), ".env".to_string()],
        sensitive_regex: vec![],
        policy: PrivacyPolicy::Redact,
        store_raw_in_logs: false,
    };
    let router = PrivacyRouter::new(config).unwrap();

    // .env should be detected as sensitive path
    assert!(router.is_sensitive_path(".env"), ".env should be sensitive");
    assert!(!router.is_sensitive_path("main.rs"), "main.rs should not be sensitive");

    // Content scanning should find secrets
    let env_content = std::fs::read_to_string(ws.join(".env")).unwrap();
    let matches = router.scan_content(&env_content);
    assert!(!matches.is_empty(), "should find sensitive content in .env");
}

#[test]
fn autocomplete_enable_disable_toggles() {
    // Test the config toggle logic used by the autocomplete command
    let mut cfg = deepseek_core::AppConfig::default();
    assert!(!cfg.local_ml.autocomplete.enabled, "default should be disabled");

    cfg.local_ml.autocomplete.enabled = true;
    assert!(cfg.local_ml.autocomplete.enabled);

    cfg.local_ml.autocomplete.enabled = false;
    assert!(!cfg.local_ml.autocomplete.enabled);

    // Model ID change
    cfg.local_ml.autocomplete.model_id = "custom-model".to_string();
    assert_eq!(cfg.local_ml.autocomplete.model_id, "custom-model");
}

#[test]
fn index_query_returns_results() {
    let dir = TempDir::new().unwrap();
    let embeddings = Arc::new(MockEmbeddings::new(64));

    let ws = dir.path().join("workspace");
    std::fs::create_dir_all(&ws).unwrap();

    let mut content = String::new();
    for i in 1..=60 {
        content.push_str(&format!("pub fn search_handler_{}() {{ /* search logic */ }}\n", i));
    }
    std::fs::write(ws.join("search.rs"), &content).unwrap();

    let mut retriever = HybridRetriever::new(
        &dir.path().join("idx"),
        embeddings,
        None,
        0.7,
        ChunkConfig::default(),
    )
    .unwrap();

    retriever.build_index(&ws).unwrap();

    let filter = SearchFilter::default();
    let results = retriever.search("search handler", 5, &filter).unwrap();
    assert!(!results.is_empty(), "query should return results");

    // Results should be ranked by score
    for i in 1..results.len() {
        assert!(
            results[i - 1].hybrid_score >= results[i].hybrid_score,
            "results should be sorted by score descending"
        );
    }
}
