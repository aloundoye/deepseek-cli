use anyhow::Result;
use codingbuddy_core::{Session, runtime_dir};
use ignore::WalkBuilder;
use notify::{
    Config as NotifyConfig, EventKind as NotifyEventKind, RecommendedWatcher, RecursiveMode,
    Watcher,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{NumericOptions, STORED, STRING, Schema, TEXT, Value};
use tantivy::{Index, doc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub baseline_commit: Option<String>,
    pub files: BTreeMap<String, String>,
    pub index_schema_version: u32,
    pub ignore_rules_hash: String,
    pub fresh: bool,
    pub corrupt: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub path: String,
    pub line: usize,
    pub excerpt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub freshness: String,
    pub results: Vec<QueryResult>,
}

pub struct IndexService {
    workspace: PathBuf,
}

impl IndexService {
    pub fn new(workspace: &Path) -> Result<Self> {
        fs::create_dir_all(runtime_dir(workspace).join("index"))?;
        Ok(Self {
            workspace: workspace.to_path_buf(),
        })
    }

    pub fn build(&self, session: &Session) -> Result<Manifest> {
        let mut files = BTreeMap::new();
        for path in workspace_file_paths(&self.workspace, true) {
            if !path.is_file() {
                continue;
            }
            let rel = path
                .strip_prefix(&self.workspace)?
                .to_string_lossy()
                .to_string();
            let bytes = fs::read(path)?;
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            files.insert(rel, format!("{:x}", hasher.finalize()));
        }

        let manifest = Manifest {
            baseline_commit: session.baseline_commit.clone(),
            files,
            index_schema_version: 1,
            ignore_rules_hash: "default-v1".to_string(),
            fresh: true,
            corrupt: false,
        };

        self.rebuild_tantivy(&manifest)?;

        let path = runtime_dir(&self.workspace).join("index/manifest.json");
        fs::write(path, serde_json::to_vec_pretty(&manifest)?)?;
        Ok(manifest)
    }

    pub fn update(&self, session: &Session) -> Result<Manifest> {
        self.build(session)
    }

    pub fn watch_and_update(
        &self,
        session: &Session,
        max_events: usize,
        timeout: Duration,
    ) -> Result<Manifest> {
        if self.status()?.is_none() {
            self.build(session)?;
        }

        let (tx, rx) = channel();
        let mut watcher = RecommendedWatcher::new(tx, NotifyConfig::default())?;
        watcher.watch(&self.workspace, RecursiveMode::Recursive)?;

        let deadline = Instant::now() + timeout;
        let mut seen = 0usize;
        let mut should_update = false;

        while seen < max_events {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let remaining = deadline.saturating_duration_since(now);
            match rx.recv_timeout(remaining) {
                Ok(Ok(event)) => {
                    let relevant_kind = matches!(
                        event.kind,
                        NotifyEventKind::Create(_)
                            | NotifyEventKind::Modify(_)
                            | NotifyEventKind::Remove(_)
                            | NotifyEventKind::Any
                    );
                    if !relevant_kind {
                        continue;
                    }
                    if event.paths.iter().any(|path| {
                        path.strip_prefix(&self.workspace)
                            .ok()
                            .is_none_or(|rel| !has_ignored_component(rel))
                    }) {
                        seen += 1;
                        should_update = true;
                    }
                }
                Ok(Err(_)) => continue,
                Err(_) => break,
            }
        }

        if should_update {
            return self.update(session);
        }

        match self.status()? {
            Some(manifest) if manifest.fresh => Ok(manifest),
            Some(_) => self.update(session),
            None => self.build(session),
        }
    }

    pub fn status(&self) -> Result<Option<Manifest>> {
        let path = runtime_dir(&self.workspace).join("index/manifest.json");
        if !path.exists() {
            return Ok(None);
        }
        let mut manifest: Manifest = serde_json::from_str(&fs::read_to_string(path)?)?;
        let mut fresh = true;
        let mut corrupt = false;
        for (rel, expected) in &manifest.files {
            let full = self.workspace.join(rel);
            if !full.exists() {
                corrupt = true;
                fresh = false;
                break;
            }
            let bytes = fs::read(&full)?;
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            let got = format!("{:x}", hasher.finalize());
            if &got != expected {
                fresh = false;
            }
        }
        let tantivy_dir = self.tantivy_dir();
        if !tantivy_dir.exists() {
            corrupt = true;
            fresh = false;
        }
        manifest.fresh = fresh && !corrupt;
        manifest.corrupt = corrupt;
        Ok(Some(manifest))
    }

    pub fn query(&self, q: &str, top_k: usize, scope: Option<&str>) -> Result<QueryResponse> {
        let status = self.status()?;
        let freshness = if let Some(m) = &status {
            if m.corrupt {
                "corrupt"
            } else if m.fresh {
                "fresh"
            } else {
                "stale"
            }
        } else {
            "stale"
        }
        .to_string();

        let results = match self.query_tantivy(q, top_k, scope) {
            Ok(results) => results,
            Err(_) => self.query_fallback(q, top_k, scope)?,
        };

        Ok(QueryResponse { freshness, results })
    }

    fn rebuild_tantivy(&self, manifest: &Manifest) -> Result<()> {
        let tantivy_dir = self.tantivy_dir();
        if tantivy_dir.exists() {
            fs::remove_dir_all(&tantivy_dir)?;
        }
        fs::create_dir_all(&tantivy_dir)?;

        let mut schema_builder = Schema::builder();
        let path_field = schema_builder.add_text_field("path", STRING | STORED);
        let content_field = schema_builder.add_text_field("content", TEXT | STORED);
        let schema = schema_builder.build();

        let index = Index::create_in_dir(&tantivy_dir, schema)?;
        let mut writer = index.writer(50_000_000)?;

        for rel in manifest.files.keys() {
            let full = self.workspace.join(rel);
            let content = match fs::read_to_string(&full) {
                Ok(c) => c,
                Err(_) => continue,
            };
            writer.add_document(doc!(path_field => rel.as_str(), content_field => content))?;
        }

        writer.commit()?;
        Ok(())
    }

    fn query_tantivy(
        &self,
        q: &str,
        top_k: usize,
        scope: Option<&str>,
    ) -> Result<Vec<QueryResult>> {
        let index = Index::open_in_dir(self.tantivy_dir())?;
        let schema = index.schema();
        let path_field = schema.get_field("path")?;
        let content_field = schema.get_field("content")?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let parser = QueryParser::for_index(&index, vec![path_field, content_field]);
        let query = parser.parse_query(q)?;
        let docs = searcher.search(
            &query,
            &TopDocs::with_limit(top_k.saturating_mul(4).max(10)),
        )?;

        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for (_score, doc_addr) in docs {
            let retrieved = searcher.doc::<tantivy::schema::TantivyDocument>(doc_addr)?;
            let Some(path) = retrieved
                .get_first(path_field)
                .and_then(|v| v.as_str())
                .map(ToString::to_string)
            else {
                continue;
            };
            if !seen.insert(path.clone()) {
                continue;
            }
            if let Some(s) = scope
                && !path.starts_with(s)
                && !path.contains(s)
            {
                continue;
            }

            let full = self.workspace.join(&path);
            let (line, excerpt) = extract_match_line(&full, q).unwrap_or((1, String::new()));
            out.push(QueryResult {
                path,
                line,
                excerpt,
            });
            if out.len() >= top_k {
                break;
            }
        }
        Ok(out)
    }

    fn query_fallback(
        &self,
        q: &str,
        top_k: usize,
        scope: Option<&str>,
    ) -> Result<Vec<QueryResult>> {
        let mut results = Vec::new();
        for path in workspace_file_paths(&self.workspace, true) {
            if !path.is_file() {
                continue;
            }
            let rel_path = path.strip_prefix(&self.workspace)?;
            if has_ignored_component(rel_path) {
                continue;
            }
            let rel = rel_path.to_string_lossy().to_string();
            if let Some(s) = scope
                && !rel.starts_with(s)
                && !rel.contains(s)
            {
                continue;
            }
            if let Ok(content) = fs::read_to_string(path) {
                for (idx, line) in content.lines().enumerate() {
                    if line.contains(q) {
                        results.push(QueryResult {
                            path: rel.clone(),
                            line: idx + 1,
                            excerpt: line.to_string(),
                        });
                        if results.len() >= top_k {
                            return Ok(results);
                        }
                    }
                }
            }
        }
        Ok(results)
    }

    fn tantivy_dir(&self) -> PathBuf {
        runtime_dir(&self.workspace).join("index/tantivy")
    }

    /// Build a chunk-level Tantivy index from pre-computed chunks.
    ///
    /// This is used by `HybridRetriever` to index chunks at the same granularity
    /// as the vector index, enabling proper BM25↔vector fusion via shared chunk IDs.
    pub fn build_chunk_index(&self, chunks: &[ChunkEntry]) -> Result<()> {
        let chunk_dir = self.chunk_tantivy_dir();
        if chunk_dir.exists() {
            fs::remove_dir_all(&chunk_dir)?;
        }
        fs::create_dir_all(&chunk_dir)?;

        let mut schema_builder = Schema::builder();
        let chunk_id_field = schema_builder.add_text_field("chunk_id", STRING | STORED);
        let path_field = schema_builder.add_text_field("path", STRING | STORED);
        let content_field = schema_builder.add_text_field("content", TEXT | STORED);
        let start_line_field = schema_builder.add_u64_field(
            "start_line",
            NumericOptions::default().set_stored().set_indexed(),
        );
        let end_line_field = schema_builder.add_u64_field(
            "end_line",
            NumericOptions::default().set_stored().set_indexed(),
        );
        let schema = schema_builder.build();

        let index = Index::create_in_dir(&chunk_dir, schema)?;
        let mut writer = index.writer(50_000_000)?;

        for chunk in chunks {
            writer.add_document(doc!(
                chunk_id_field => chunk.chunk_id.as_str(),
                path_field => chunk.path.as_str(),
                content_field => chunk.content.as_str(),
                start_line_field => chunk.start_line as u64,
                end_line_field => chunk.end_line as u64
            ))?;
        }

        writer.commit()?;
        Ok(())
    }

    /// Query the chunk-level Tantivy index, returning chunk IDs with BM25 scores.
    pub fn query_chunks(
        &self,
        q: &str,
        top_k: usize,
        scope: Option<&str>,
    ) -> Result<Vec<ChunkQueryResult>> {
        let chunk_dir = self.chunk_tantivy_dir();
        if !chunk_dir.exists() {
            return Ok(Vec::new());
        }

        let index = Index::open_in_dir(chunk_dir)?;
        let schema = index.schema();
        let chunk_id_field = schema.get_field("chunk_id")?;
        let path_field = schema.get_field("path")?;
        let content_field = schema.get_field("content")?;
        let start_line_field = schema.get_field("start_line")?;
        let end_line_field = schema.get_field("end_line")?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let parser = QueryParser::for_index(&index, vec![content_field]);
        let query = parser.parse_query(q)?;
        let docs = searcher.search(&query, &TopDocs::with_limit(top_k * 2))?;

        let mut results = Vec::new();
        for (score, doc_addr) in docs {
            let retrieved = searcher.doc::<tantivy::schema::TantivyDocument>(doc_addr)?;

            let chunk_id = retrieved
                .get_first(chunk_id_field)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let path = retrieved
                .get_first(path_field)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let start_line = retrieved
                .get_first(start_line_field)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            let end_line = retrieved
                .get_first(end_line_field)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;

            if let Some(s) = scope
                && !path.starts_with(s)
                && !path.contains(s)
            {
                continue;
            }

            results.push(ChunkQueryResult {
                chunk_id,
                path,
                start_line,
                end_line,
                score,
            });

            if results.len() >= top_k {
                break;
            }
        }

        Ok(results)
    }

    fn chunk_tantivy_dir(&self) -> PathBuf {
        runtime_dir(&self.workspace).join("index/tantivy_chunks")
    }
}

/// Entry for chunk-level Tantivy indexing.
#[derive(Debug, Clone)]
pub struct ChunkEntry {
    pub chunk_id: String,
    pub path: String,
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
}

/// Result from chunk-level BM25 query.
#[derive(Debug, Clone)]
pub struct ChunkQueryResult {
    pub chunk_id: String,
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub score: f32,
}

fn workspace_file_paths(workspace: &Path, respect_gitignore: bool) -> Vec<PathBuf> {
    let mut builder = WalkBuilder::new(workspace);
    builder.hidden(false);
    builder.follow_links(false);
    builder.parents(respect_gitignore);
    builder.git_ignore(respect_gitignore);
    builder.git_global(respect_gitignore);
    builder.git_exclude(respect_gitignore);
    builder.require_git(false);
    builder.add_custom_ignore_filename(".codingbuddyignore");

    let mut out = Vec::new();
    for entry in builder.build() {
        let Ok(entry) = entry else {
            continue;
        };
        if !entry
            .file_type()
            .map(|file_type| file_type.is_file())
            .unwrap_or(false)
        {
            continue;
        }
        let path = entry.path();
        let Ok(rel) = path.strip_prefix(workspace) else {
            continue;
        };
        if has_ignored_component(rel) {
            continue;
        }
        out.push(path.to_path_buf());
    }
    out
}

fn has_ignored_component(path: &Path) -> bool {
    path.components().any(|c| {
        let comp = c.as_os_str();
        comp == OsStr::new(".git")
            || comp == OsStr::new(".codingbuddy")
            || comp == OsStr::new("target")
    })
}

fn extract_match_line(path: &Path, needle: &str) -> Option<(usize, String)> {
    let content = fs::read_to_string(path).ok()?;
    for (idx, line) in content.lines().enumerate() {
        if line.contains(needle) {
            return Some((idx + 1, line.to_string()));
        }
    }
    Some((1, content.lines().next().unwrap_or_default().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use codingbuddy_core::{SessionBudgets, SessionState};
    use std::thread;
    use std::time::Duration;
    use uuid::Uuid;

    #[test]
    fn status_becomes_stale_when_file_changes() {
        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-index-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        fs::write(workspace.join("a.txt"), "hello").expect("seed file");

        let svc = IndexService::new(&workspace).expect("svc");
        let session = Session {
            session_id: Uuid::now_v7(),
            workspace_root: workspace.to_string_lossy().to_string(),
            baseline_commit: None,
            status: SessionState::Idle,
            budgets: SessionBudgets {
                per_turn_seconds: 10,
                max_think_tokens: 1000,
            },
            active_plan_id: None,
        };
        svc.build(&session).expect("build");
        fs::write(workspace.join("a.txt"), "changed").expect("mutate");
        let status = svc.status().expect("status").expect("manifest");
        assert!(!status.fresh);
    }

    #[test]
    fn query_uses_tantivy_index() {
        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-index-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        fs::write(workspace.join("main.rs"), "fn router_decision() {}\n").expect("seed file");

        let svc = IndexService::new(&workspace).expect("svc");
        let session = Session {
            session_id: Uuid::now_v7(),
            workspace_root: workspace.to_string_lossy().to_string(),
            baseline_commit: None,
            status: SessionState::Idle,
            budgets: SessionBudgets {
                per_turn_seconds: 10,
                max_think_tokens: 1000,
            },
            active_plan_id: None,
        };
        svc.build(&session).expect("build");
        let result = svc.query("router_decision", 5, None).expect("query");
        assert!(!result.results.is_empty());
    }

    #[test]
    fn watcher_rebuilds_after_file_change() {
        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-index-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        fs::write(workspace.join("watched.txt"), "before").expect("seed file");

        let svc = IndexService::new(&workspace).expect("svc");
        let session = Session {
            session_id: Uuid::now_v7(),
            workspace_root: workspace.to_string_lossy().to_string(),
            baseline_commit: None,
            status: SessionState::Idle,
            budgets: SessionBudgets {
                per_turn_seconds: 10,
                max_think_tokens: 1000,
            },
            active_plan_id: None,
        };
        svc.build(&session).expect("build");

        let workspace_for_write = workspace.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(200));
            fs::write(workspace_for_write.join("watched.txt"), "after").expect("mutate");
        });

        let manifest = svc
            .watch_and_update(&session, 1, Duration::from_secs(3))
            .expect("watch");
        assert!(manifest.fresh);
        let status = svc.status().expect("status").expect("manifest");
        assert!(status.fresh);
    }

    // ── Phase 11: Retrieval returns cited chunks ────────────────────────

    #[test]
    fn retrieval_results_include_file_path_and_line_citations() {
        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-index-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        fs::write(
            workspace.join("utils.rs"),
            "fn calculate_score(x: i32) -> i32 {\n    x * 2\n}\n",
        )
        .expect("seed file");

        let svc = IndexService::new(&workspace).expect("svc");
        let session = Session {
            session_id: Uuid::now_v7(),
            workspace_root: workspace.to_string_lossy().to_string(),
            baseline_commit: None,
            status: SessionState::Idle,
            budgets: SessionBudgets {
                per_turn_seconds: 10,
                max_think_tokens: 1000,
            },
            active_plan_id: None,
        };
        svc.build(&session).expect("build");

        let result = svc.query("calculate_score", 5, None).expect("query");
        assert!(!result.results.is_empty(), "should find at least one match");

        // Each QueryResult must carry file-path + line citation
        for hit in &result.results {
            assert!(
                !hit.path.is_empty(),
                "result must include a non-empty file path"
            );
            assert!(hit.line > 0, "result line number must be positive");
            assert!(
                !hit.excerpt.is_empty(),
                "result must include a non-empty excerpt"
            );
        }
    }

    #[test]
    fn chunk_level_index_returns_chunk_ids() {
        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-index-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        fs::write(
            workspace.join("lib.rs"),
            "fn compute() { 42 }\nfn helper() { 7 }\n",
        )
        .expect("seed");

        let svc = IndexService::new(&workspace).expect("svc");

        let chunks = vec![
            ChunkEntry {
                chunk_id: "chunk-001".to_string(),
                path: "lib.rs".to_string(),
                content: "fn compute() { 42 }".to_string(),
                start_line: 1,
                end_line: 1,
            },
            ChunkEntry {
                chunk_id: "chunk-002".to_string(),
                path: "lib.rs".to_string(),
                content: "fn helper() { 7 }".to_string(),
                start_line: 2,
                end_line: 2,
            },
        ];

        svc.build_chunk_index(&chunks).expect("build chunk index");

        let results = svc.query_chunks("compute", 5, None).expect("query chunks");
        assert!(!results.is_empty(), "should find chunk-level results");
        assert_eq!(
            results[0].chunk_id, "chunk-001",
            "should return the correct chunk_id"
        );
        assert_eq!(results[0].path, "lib.rs");
    }

    // ── Helper function tests ──

    #[test]
    fn has_ignored_component_detects_git_dir() {
        assert!(has_ignored_component(Path::new(".git/objects/abc")));
        assert!(has_ignored_component(Path::new("src/.git/config")));
    }

    #[test]
    fn has_ignored_component_detects_target_dir() {
        assert!(has_ignored_component(Path::new("target/debug/build")));
    }

    #[test]
    fn has_ignored_component_detects_codingbuddy_dir() {
        assert!(has_ignored_component(Path::new(".codingbuddy/config.toml")));
    }

    #[test]
    fn has_ignored_component_allows_normal_paths() {
        assert!(!has_ignored_component(Path::new("src/lib.rs")));
        assert!(!has_ignored_component(Path::new("crates/core/src/main.rs")));
    }

    #[test]
    fn extract_match_line_finds_matching_line() {
        let ws = std::env::temp_dir().join(format!("cb-idx-match-{}", Uuid::now_v7()));
        fs::create_dir_all(&ws).unwrap();
        let file = ws.join("sample.rs");
        fs::write(&file, "line one\nfn target_func() {}\nline three\n").unwrap();

        let (line, excerpt) = extract_match_line(&file, "target_func").unwrap();
        assert_eq!(line, 2);
        assert!(excerpt.contains("target_func"));
    }

    #[test]
    fn extract_match_line_defaults_to_first_line() {
        let ws = std::env::temp_dir().join(format!("cb-idx-default-{}", Uuid::now_v7()));
        fs::create_dir_all(&ws).unwrap();
        let file = ws.join("sample.txt");
        fs::write(&file, "first line\nsecond line\n").unwrap();

        let (line, _excerpt) = extract_match_line(&file, "nonexistent").unwrap();
        assert_eq!(line, 1);
    }

    #[test]
    fn extract_match_line_returns_none_for_missing_file() {
        assert!(extract_match_line(Path::new("/no/such/file.rs"), "x").is_none());
    }

    // ── IndexService edge cases ──

    #[test]
    fn status_returns_none_before_build() {
        let ws = std::env::temp_dir().join(format!("cb-idx-status-{}", Uuid::now_v7()));
        fs::create_dir_all(&ws).unwrap();
        let svc = IndexService::new(&ws).unwrap();
        assert!(svc.status().unwrap().is_none());
    }

    #[test]
    fn query_returns_stale_when_no_index() {
        let ws = std::env::temp_dir().join(format!("cb-idx-noindex-{}", Uuid::now_v7()));
        fs::create_dir_all(&ws).unwrap();
        fs::write(ws.join("test.txt"), "hello world").unwrap();
        let svc = IndexService::new(&ws).unwrap();

        let response = svc.query("hello", 5, None).unwrap();
        assert_eq!(response.freshness, "stale");
    }

    #[test]
    fn build_indexes_multiple_files() {
        let ws = std::env::temp_dir().join(format!("cb-idx-multi-{}", Uuid::now_v7()));
        fs::create_dir_all(&ws).unwrap();
        fs::write(ws.join("a.rs"), "fn alpha() {}").unwrap();
        fs::write(ws.join("b.rs"), "fn beta() {}").unwrap();
        fs::write(ws.join("c.rs"), "fn gamma() {}").unwrap();

        let svc = IndexService::new(&ws).unwrap();
        let session = Session {
            session_id: Uuid::now_v7(),
            workspace_root: ws.to_string_lossy().to_string(),
            baseline_commit: None,
            status: SessionState::Idle,
            budgets: SessionBudgets {
                per_turn_seconds: 10,
                max_think_tokens: 1000,
            },
            active_plan_id: None,
        };
        let manifest = svc.build(&session).unwrap();
        assert_eq!(manifest.files.len(), 3);
        assert!(manifest.fresh);
        assert!(!manifest.corrupt);
    }

    #[test]
    fn query_with_scope_filters_results() {
        let ws = std::env::temp_dir().join(format!("cb-idx-scope-{}", Uuid::now_v7()));
        fs::create_dir_all(ws.join("src")).unwrap();
        fs::create_dir_all(ws.join("tests")).unwrap();
        fs::write(ws.join("src/lib.rs"), "fn unique_target() {}").unwrap();
        fs::write(ws.join("tests/test.rs"), "fn unique_target() {}").unwrap();

        let svc = IndexService::new(&ws).unwrap();
        let session = Session {
            session_id: Uuid::now_v7(),
            workspace_root: ws.to_string_lossy().to_string(),
            baseline_commit: None,
            status: SessionState::Idle,
            budgets: SessionBudgets {
                per_turn_seconds: 10,
                max_think_tokens: 1000,
            },
            active_plan_id: None,
        };
        svc.build(&session).unwrap();

        let result = svc.query("unique_target", 10, Some("src/")).unwrap();
        for hit in &result.results {
            assert!(hit.path.starts_with("src/") || hit.path.contains("src/"));
        }
    }

    #[test]
    fn chunk_query_returns_empty_when_no_chunk_index() {
        let ws = std::env::temp_dir().join(format!("cb-idx-nochunk-{}", Uuid::now_v7()));
        fs::create_dir_all(&ws).unwrap();
        let svc = IndexService::new(&ws).unwrap();
        let results = svc.query_chunks("anything", 5, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn manifest_serialization_roundtrip() {
        let manifest = Manifest {
            baseline_commit: Some("abc123".to_string()),
            files: BTreeMap::from([("src/lib.rs".to_string(), "sha256hex".to_string())]),
            index_schema_version: 1,
            ignore_rules_hash: "v1".to_string(),
            fresh: true,
            corrupt: false,
        };
        let json = serde_json::to_string(&manifest).unwrap();
        let restored: Manifest = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.baseline_commit, manifest.baseline_commit);
        assert_eq!(restored.files.len(), 1);
        assert!(restored.fresh);
    }
}
