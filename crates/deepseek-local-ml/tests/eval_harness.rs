//! Evaluation harness for retrieval quality (P8 Batch 6).
//!
//! Creates a fixture repository, runs retrieval queries, and measures
//! token estimates to prove retrieval reduces prompt tokens vs no-retrieval.

use deepseek_local_ml::chunker::ChunkConfig;
use deepseek_local_ml::embeddings::MockEmbeddings;
use deepseek_local_ml::retrieval::HybridRetriever;
use deepseek_local_ml::vector_index::SearchFilter;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tempfile::TempDir;

/// Metrics for a single evaluation run.
#[derive(Debug, Serialize, Deserialize)]
struct RunMetrics {
    query: String,
    result_count: usize,
    total_content_chars: usize,
    estimated_tokens: usize,
}

/// Full evaluation report comparing baseline vs retrieval-augmented.
#[derive(Debug, Serialize, Deserialize)]
struct EvalReport {
    baseline: RunMetrics,
    retrieval_first: RunMetrics,
    token_savings_pct: f64,
}

fn estimate_tokens(chars: usize) -> usize {
    // Rough estimate: ~4 chars per token
    chars / 4
}

fn create_fixture_workspace(dir: &std::path::Path) {
    let ws = dir.join("workspace");
    std::fs::create_dir_all(&ws).unwrap();

    // Create a realistic small codebase
    let mut main_content = String::from("// Main application entry point\n");
    for i in 1..=100 {
        main_content.push_str(&format!(
            "fn handler_{}() {{ /* handle request type {} */ }}\n",
            i, i
        ));
    }
    std::fs::write(ws.join("main.rs"), &main_content).unwrap();

    let mut lib_content = String::from("// Library module with utility functions\n");
    for i in 1..=80 {
        lib_content.push_str(&format!(
            "pub fn utility_{}(input: &str) -> String {{ input.to_uppercase() }}\n",
            i
        ));
    }
    std::fs::write(ws.join("lib.rs"), &lib_content).unwrap();

    let mut config_content = String::from("// Configuration and settings\n");
    for i in 1..=50 {
        config_content.push_str(&format!(
            "pub const CONFIG_{}: &str = \"value_{}\";\n",
            i, i
        ));
    }
    std::fs::write(ws.join("config.rs"), &config_content).unwrap();

    let mut test_content = String::from("// Test module\n#[cfg(test)]\nmod tests {\n");
    for i in 1..=60 {
        test_content.push_str(&format!(
            "    #[test] fn test_handler_{}() {{ assert!(true); }}\n",
            i
        ));
    }
    test_content.push_str("}\n");
    std::fs::write(ws.join("tests.rs"), &test_content).unwrap();
}

#[test]
fn eval_harness_produces_report() {
    let dir = TempDir::new().unwrap();
    create_fixture_workspace(dir.path());

    let ws = dir.path().join("workspace");
    let embeddings = Arc::new(MockEmbeddings::new(64));

    let mut retriever = HybridRetriever::new(
        &dir.path().join("idx"),
        embeddings,
        None,
        0.7,
        ChunkConfig::default(),
    )
    .unwrap();

    let build_report = retriever.build_index(&ws).unwrap();
    assert!(build_report.chunks_indexed > 0);
    assert!(build_report.files_processed > 0);

    // Baseline: full codebase content (no retrieval)
    let mut baseline_chars = 0;
    for entry in std::fs::read_dir(&ws).unwrap().flatten() {
        if entry.path().is_file() {
            if let Ok(content) = std::fs::read_to_string(entry.path()) {
                baseline_chars += content.len();
            }
        }
    }
    let baseline = RunMetrics {
        query: "handler implementation".to_string(),
        result_count: 4, // all files
        total_content_chars: baseline_chars,
        estimated_tokens: estimate_tokens(baseline_chars),
    };

    // Retrieval-augmented: only top-k chunks
    let filter = SearchFilter::default();
    let results = retriever
        .search("handler implementation", 5, &filter)
        .unwrap();
    let retrieval_chars: usize = results.iter().map(|r| r.chunk.content.len()).sum();
    let retrieval_first = RunMetrics {
        query: "handler implementation".to_string(),
        result_count: results.len(),
        total_content_chars: retrieval_chars,
        estimated_tokens: estimate_tokens(retrieval_chars),
    };

    let savings = if baseline.estimated_tokens > 0 {
        ((baseline.estimated_tokens as f64 - retrieval_first.estimated_tokens as f64)
            / baseline.estimated_tokens as f64)
            * 100.0
    } else {
        0.0
    };

    let report = EvalReport {
        baseline,
        retrieval_first,
        token_savings_pct: savings,
    };

    // Verify report structure
    assert!(report.baseline.estimated_tokens > 0, "baseline should have tokens");
    assert!(
        report.retrieval_first.estimated_tokens > 0,
        "retrieval should have tokens"
    );
    assert!(
        report.token_savings_pct >= 0.0,
        "should have non-negative savings"
    );

    // Serialization roundtrip
    let json = serde_json::to_string(&report).unwrap();
    let _: EvalReport = serde_json::from_str(&json).unwrap();
}

#[test]
fn retrieval_reduces_prompt_tokens() {
    let dir = TempDir::new().unwrap();
    create_fixture_workspace(dir.path());

    let ws = dir.path().join("workspace");
    let embeddings = Arc::new(MockEmbeddings::new(64));

    let mut retriever = HybridRetriever::new(
        &dir.path().join("idx"),
        embeddings,
        None,
        0.7,
        ChunkConfig::default(),
    )
    .unwrap();

    retriever.build_index(&ws).unwrap();

    // Baseline: count all characters in the workspace
    let mut total_workspace_chars = 0;
    for entry in std::fs::read_dir(&ws).unwrap().flatten() {
        if entry.path().is_file() {
            if let Ok(content) = std::fs::read_to_string(entry.path()) {
                total_workspace_chars += content.len();
            }
        }
    }
    let baseline_tokens = estimate_tokens(total_workspace_chars);

    // Retrieval: only relevant chunks
    let filter = SearchFilter::default();
    let results = retriever.search("utility function", 5, &filter).unwrap();
    let retrieval_chars: usize = results.iter().map(|r| r.chunk.content.len()).sum();
    let retrieval_tokens = estimate_tokens(retrieval_chars);

    assert!(
        retrieval_tokens < baseline_tokens,
        "retrieval ({} tokens) should use fewer tokens than full workspace ({} tokens)",
        retrieval_tokens,
        baseline_tokens
    );

    // Should save at least 20% of tokens (we're selecting 5 chunks out of many)
    let savings_pct =
        (baseline_tokens as f64 - retrieval_tokens as f64) / baseline_tokens as f64 * 100.0;
    assert!(
        savings_pct > 20.0,
        "should save >20% tokens, got {:.1}%",
        savings_pct
    );
}
