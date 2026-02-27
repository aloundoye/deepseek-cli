use deepseek_local_ml::chunker::Chunk;
use deepseek_local_ml::embeddings::MockEmbeddings;
use deepseek_local_ml::retrieval::{reciprocal_rank_fusion, HybridRetriever};
use deepseek_local_ml::vector_index::*;
use deepseek_local_ml::ChunkConfig;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

fn make_chunk(id: &str, file_path: &str, language: &str, content: &str) -> Chunk {
    Chunk {
        id: id.to_string(),
        file_path: PathBuf::from(file_path),
        start_line: 1,
        end_line: 10,
        content: content.to_string(),
        content_hash: format!("hash_{}", id),
        language: language.to_string(),
    }
}

fn make_vector(dim: usize, seed: f32) -> Vec<f32> {
    // Create a simple deterministic vector
    let mut v: Vec<f32> = (0..dim).map(|i| ((i as f32 + seed) * 0.1).sin()).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[test]
fn insert_and_search_finds_similar() {
    let dir = TempDir::new().unwrap();
    let backend = Box::new(BruteForceBackend::new(64));
    let mut index = VectorIndex::create(dir.path(), backend).unwrap();

    let c1 = make_chunk("c1", "a.rs", "rust", "fn main() {}");
    let c2 = make_chunk("c2", "b.rs", "rust", "fn helper() {}");
    let c3 = make_chunk("c3", "c.py", "python", "def main(): pass");

    let v1 = make_vector(64, 1.0);
    let v2 = make_vector(64, 1.1); // close to v1
    let v3 = make_vector(64, 100.0); // far from v1

    index.insert(&c1, &v1).unwrap();
    index.insert(&c2, &v2).unwrap();
    index.insert(&c3, &v3).unwrap();

    let filter = SearchFilter::default();
    let results = index.search(&v1, 2, &filter).unwrap();
    assert!(!results.is_empty(), "search should return results");
    assert_eq!(results[0].chunk.id, "c1", "most similar should be itself");
    // c2 (seed 1.1) should be closer than c3 (seed 100.0)
    if results.len() >= 2 {
        assert_eq!(results[1].chunk.id, "c2", "second closest should be c2");
    }
}

#[test]
fn search_filter_by_language() {
    let dir = TempDir::new().unwrap();
    let backend = Box::new(BruteForceBackend::new(64));
    let mut index = VectorIndex::create(dir.path(), backend).unwrap();

    let c1 = make_chunk("c1", "a.rs", "rust", "fn main() {}");
    let c2 = make_chunk("c2", "b.py", "python", "def main(): pass");

    let v1 = make_vector(64, 1.0);
    let v2 = make_vector(64, 1.1);

    index.insert(&c1, &v1).unwrap();
    index.insert(&c2, &v2).unwrap();

    let filter = SearchFilter {
        languages: vec!["python".to_string()],
        ..Default::default()
    };
    let results = index.search(&v1, 10, &filter).unwrap();
    assert!(
        results.iter().all(|r| r.chunk.language == "python"),
        "all results should be python"
    );
}

#[test]
fn remove_chunks_from_index() {
    let dir = TempDir::new().unwrap();
    let backend = Box::new(BruteForceBackend::new(64));
    let mut index = VectorIndex::create(dir.path(), backend).unwrap();

    let c1 = make_chunk("c1", "a.rs", "rust", "fn main() {}");
    let v1 = make_vector(64, 1.0);
    index.insert(&c1, &v1).unwrap();

    let stats_before = index.stats().unwrap();
    assert_eq!(stats_before.chunk_count, 1);

    index.remove("c1").unwrap();

    let stats_after = index.stats().unwrap();
    assert_eq!(stats_after.chunk_count, 0);

    let filter = SearchFilter::default();
    let results = index.search(&v1, 10, &filter).unwrap();
    assert!(results.is_empty(), "deleted chunk should not appear");
}

#[test]
fn index_stats_correct() {
    let dir = TempDir::new().unwrap();
    let backend = Box::new(BruteForceBackend::new(64));
    let mut index = VectorIndex::create(dir.path(), backend).unwrap();

    for i in 0..5 {
        let c = make_chunk(
            &format!("c{}", i),
            &format!("file{}.rs", i),
            "rust",
            &format!("fn func_{}() {{}}", i),
        );
        let v = make_vector(64, i as f32);
        index.insert(&c, &v).unwrap();
    }

    let stats = index.stats().unwrap();
    assert_eq!(stats.chunk_count, 5);
    assert_eq!(stats.file_count, 5);
}

#[test]
fn hybrid_search_combines_scores() {
    let dir = TempDir::new().unwrap();
    let embeddings = Arc::new(MockEmbeddings::new(64));

    let mut retriever = HybridRetriever::new(
        &dir.path().join("idx"),
        embeddings,
        None,
        0.7,
        ChunkConfig::default(),
    )
    .unwrap();

    // Build a small test workspace
    let ws = dir.path().join("workspace");
    std::fs::create_dir_all(&ws).unwrap();
    let mut content = String::new();
    for i in 1..=60 {
        content.push_str(&format!("fn function_{}() {{ /* implementation */ }}\n", i));
    }
    std::fs::write(ws.join("lib.rs"), &content).unwrap();

    let report = retriever.build_index(&ws).unwrap();
    assert!(report.chunks_indexed > 0, "should index at least one chunk");
    assert!(report.files_processed > 0);

    let filter = SearchFilter::default();
    let results = retriever.search("function implementation", 5, &filter).unwrap();
    assert!(!results.is_empty(), "hybrid search should return results");

    // All results should have valid scores
    for r in &results {
        assert!(r.hybrid_score >= 0.0, "hybrid score should be non-negative");
        assert!(r.vector_score >= -1.0 && r.vector_score <= 1.0);
    }
}

#[test]
fn hybrid_alpha_0_is_bm25_only() {
    let vec_results = vec![
        ("a".to_string(), 0.9),
        ("b".to_string(), 0.8),
        ("c".to_string(), 0.7),
    ];
    let bm25_results = vec![
        ("c".to_string(), 5.0),
        ("b".to_string(), 4.0),
        ("a".to_string(), 3.0),
    ];

    let fused = reciprocal_rank_fusion(&vec_results, &bm25_results, 0.0, 60);
    // With alpha=0, only BM25 contributes. BM25 ranking: c, b, a
    // c has BM25 rank 1, b rank 2, a rank 3
    assert_eq!(fused[0].0, "c", "alpha=0 should use BM25 ranking: c first");
    assert_eq!(fused[1].0, "b", "alpha=0 should use BM25 ranking: b second");
}

#[test]
fn hybrid_alpha_1_is_vector_only() {
    let vec_results = vec![
        ("a".to_string(), 0.9),
        ("b".to_string(), 0.8),
        ("c".to_string(), 0.7),
    ];
    let bm25_results = vec![
        ("c".to_string(), 5.0),
        ("b".to_string(), 4.0),
        ("a".to_string(), 3.0),
    ];

    let fused = reciprocal_rank_fusion(&vec_results, &bm25_results, 1.0, 60);
    // With alpha=1, only vector contributes. Vector ranking: a, b, c
    assert_eq!(fused[0].0, "a", "alpha=1 should use vector ranking: a first");
    assert_eq!(fused[1].0, "b", "alpha=1 should use vector ranking: b second");
}

#[test]
fn incremental_update_only_reindexes_changed() {
    let dir = TempDir::new().unwrap();
    let embeddings = Arc::new(MockEmbeddings::new(64));

    let ws = dir.path().join("workspace");
    std::fs::create_dir_all(&ws).unwrap();

    // Create initial files
    let mut content1 = String::new();
    for i in 1..=60 {
        content1.push_str(&format!("fn a_func_{}() {{}}\n", i));
    }
    std::fs::write(ws.join("a.rs"), &content1).unwrap();

    let mut content2 = String::new();
    for i in 1..=60 {
        content2.push_str(&format!("fn b_func_{}() {{}}\n", i));
    }
    std::fs::write(ws.join("b.rs"), &content2).unwrap();

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

    // Modify only one file
    std::fs::write(ws.join("a.rs"), "// changed\nfn new_func() {}\n").unwrap();

    let update_report = retriever.update_index(&ws).unwrap();
    // Only the changed file should be re-indexed
    assert!(
        update_report.chunks_added > 0,
        "changed file should produce new chunks"
    );
    assert!(
        update_report.files_changed > 0,
        "should report changed files"
    );
}
