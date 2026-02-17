use anyhow::Result;
use deepseek_core::{Session, runtime_dir};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{STORED, STRING, Schema, TEXT, Value};
use tantivy::{Index, doc};
use walkdir::WalkDir;

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
        for entry in WalkDir::new(&self.workspace)
            .into_iter()
            .filter_entry(|e| match e.path().strip_prefix(&self.workspace) {
                Ok(rel) => !has_ignored_component(rel),
                Err(_) => true,
            })
            .filter_map(Result::ok)
        {
            let path = entry.path();
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

    pub fn query(&self, q: &str, top_k: usize) -> Result<QueryResponse> {
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

        let results = match self.query_tantivy(q, top_k) {
            Ok(results) => results,
            Err(_) => self.query_fallback(q, top_k)?,
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

    fn query_tantivy(&self, q: &str, top_k: usize) -> Result<Vec<QueryResult>> {
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

    fn query_fallback(&self, q: &str, top_k: usize) -> Result<Vec<QueryResult>> {
        let mut results = Vec::new();
        for entry in WalkDir::new(&self.workspace)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.path().is_file())
        {
            let path = entry.path();
            let rel_path = path.strip_prefix(&self.workspace)?;
            if has_ignored_component(rel_path) {
                continue;
            }
            let rel = rel_path.to_string_lossy().to_string();
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
        let result = svc.query("router_decision", 5).expect("query");
        assert!(!result.results.is_empty());
    }
}
