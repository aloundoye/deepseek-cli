use anyhow::Result;
use deepseek_core::{Session, runtime_dir};
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
use tantivy::schema::{STORED, STRING, Schema, TEXT, Value};
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
            if let Some(s) = scope {
                if !path.starts_with(s) && !path.contains(s) {
                    continue;
                }
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
            if let Some(s) = scope {
                if !rel.starts_with(s) && !rel.contains(s) {
                    continue;
                }
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
    builder.add_custom_ignore_filename(".deepseekignore");

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
            || comp == OsStr::new(".deepseek")
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
    use deepseek_core::{SessionBudgets, SessionState};
    use std::thread;
    use std::time::Duration;
    use uuid::Uuid;

    #[test]
    fn status_becomes_stale_when_file_changes() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-index-test-{}", Uuid::now_v7()));
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
            std::env::temp_dir().join(format!("deepseek-index-test-{}", Uuid::now_v7()));
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
            std::env::temp_dir().join(format!("deepseek-index-test-{}", Uuid::now_v7()));
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
            std::env::temp_dir().join(format!("deepseek-index-test-{}", Uuid::now_v7()));
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
}
