use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use codingbuddy_core::{EventKind, Session, runtime_dir};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchSet {
    pub patch_id: Uuid,
    pub base_sha256: String,
    pub unified_diff: String,
    pub created_at: String,
    pub applied: bool,
    pub conflicts: Vec<String>,
    #[serde(default)]
    pub target_files: Vec<String>,
    #[serde(default)]
    pub apply_attempts: u32,
    #[serde(default)]
    pub last_base_sha256: Option<String>,
    #[serde(default)]
    pub last_base_sha_match: Option<bool>,
    #[serde(default)]
    pub last_error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PatchStore {
    workspace: PathBuf,
    root: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GitApplyStrategy {
    /// Try plain apply first, then fallback to 3-way merge.
    Auto,
    /// Use git's 3-way strategy directly.
    ThreeWay,
}

impl PatchStore {
    pub fn new(workspace: &Path) -> Result<Self> {
        let root = runtime_dir(workspace).join("patches");
        fs::create_dir_all(&root)?;
        Ok(Self {
            workspace: workspace.to_path_buf(),
            root,
        })
    }

    pub fn stage(&self, diff: &str, base_blob: &[u8]) -> Result<PatchSet> {
        let target_files = extract_target_files(diff);
        let base_sha256 = if base_blob.is_empty() {
            hash_workspace_state(&self.workspace, &target_files)?
        } else {
            sha256_hex(base_blob)
        };

        let patch = PatchSet {
            patch_id: Uuid::now_v7(),
            base_sha256,
            unified_diff: diff.to_string(),
            created_at: Utc::now().to_rfc3339(),
            applied: false,
            conflicts: Vec::new(),
            target_files,
            apply_attempts: 0,
            last_base_sha256: None,
            last_base_sha_match: None,
            last_error: None,
        };
        self.write_patch(&patch)?;
        Ok(patch)
    }

    pub fn list(&self) -> Result<Vec<PatchSet>> {
        let mut patches: Vec<PatchSet> = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let path = entry?.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let content = fs::read_to_string(path)?;
                patches.push(serde_json::from_str(&content)?);
            }
        }
        patches.sort_by_key(|p| p.created_at.clone());
        Ok(patches)
    }

    pub fn apply(&self, workspace: &Path, patch_id: Uuid) -> Result<(bool, Vec<String>)> {
        self.apply_with_strategy(workspace, patch_id, GitApplyStrategy::ThreeWay)
    }

    pub fn apply_with_strategy(
        &self,
        workspace: &Path,
        patch_id: Uuid,
        strategy: GitApplyStrategy,
    ) -> Result<(bool, Vec<String>)> {
        let mut patch = self.read_patch(patch_id)?;
        patch.apply_attempts = patch.apply_attempts.saturating_add(1);
        let current_base_sha = hash_workspace_state(workspace, &patch.target_files)?;
        let base_matches = current_base_sha == patch.base_sha256;
        patch.last_base_sha256 = Some(current_base_sha.clone());
        patch.last_base_sha_match = Some(base_matches);

        let diff_path = self.root.join(format!("{}.diff", patch_id));

        let run_apply = |use_three_way: bool| -> Result<std::process::Output> {
            let mut cmd = Command::new("git");
            cmd.arg("apply");
            if use_three_way {
                cmd.arg("--3way");
            }
            cmd.arg(&diff_path);
            cmd.current_dir(workspace);
            cmd.output().context("failed to execute git apply")
        };

        let output = match strategy {
            GitApplyStrategy::ThreeWay => run_apply(true)?,
            GitApplyStrategy::Auto => {
                let plain = run_apply(false)?;
                if plain.status.success() {
                    plain
                } else {
                    let fallback = run_apply(true)?;
                    if fallback.status.success() {
                        fallback
                    } else {
                        let mut merged = fallback;
                        if !plain.stderr.is_empty() {
                            let mut stderr = String::from_utf8_lossy(&plain.stderr).to_string();
                            if !stderr.ends_with('\n') {
                                stderr.push('\n');
                            }
                            stderr.push_str("--- fallback --3way failed ---\n");
                            stderr.push_str(&String::from_utf8_lossy(&merged.stderr));
                            merged.stderr = stderr.into_bytes();
                        }
                        merged
                    }
                }
            }
        };

        if output.status.success() {
            patch.applied = true;
            patch.conflicts.clear();
            patch.last_error = None;
            self.write_patch(&patch)?;
            return Ok((true, vec![]));
        }

        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let mut conflicts: Vec<String> = stderr.lines().map(ToString::to_string).collect();
        if !base_matches {
            conflicts.insert(
                0,
                format!(
                    "base-sha mismatch: expected={} actual={}",
                    patch.base_sha256, current_base_sha
                ),
            );
        }
        patch.applied = false;
        patch.conflicts = conflicts;
        patch.last_error = Some(stderr);
        self.write_patch(&patch)?;
        Ok((false, patch.conflicts.clone()))
    }

    pub fn event_for_stage(
        &self,
        session: &Session,
        patch: &PatchSet,
        seq_no: u64,
    ) -> codingbuddy_core::EventEnvelope {
        codingbuddy_core::EventEnvelope {
            seq_no,
            at: Utc::now(),
            session_id: session.session_id,
            kind: EventKind::PatchStaged {
                patch_id: patch.patch_id,
                base_sha256: patch.base_sha256.clone(),
            },
        }
    }

    fn write_patch(&self, patch: &PatchSet) -> Result<()> {
        let json_path = self.root.join(format!("{}.json", patch.patch_id));
        let diff_path = self.root.join(format!("{}.diff", patch.patch_id));
        fs::write(&json_path, serde_json::to_vec_pretty(patch)?)?;
        fs::write(diff_path, &patch.unified_diff)?;
        Ok(())
    }

    fn read_patch(&self, patch_id: Uuid) -> Result<PatchSet> {
        let path = self.root.join(format!("{}.json", patch_id));
        if !path.exists() {
            return Err(anyhow!("unknown patch_id {}", patch_id));
        }
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn hash_workspace_state(workspace: &Path, files: &[String]) -> Result<String> {
    let mut hasher = Sha256::new();
    for rel in files {
        hasher.update(rel.as_bytes());
        hasher.update([0_u8]);
        let full = workspace.join(rel);
        if full.exists() {
            hasher.update([1_u8]);
            hasher.update(fs::read(full)?);
        } else {
            hasher.update([0_u8]);
        }
        hasher.update([255_u8]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn extract_target_files(diff: &str) -> Vec<String> {
    let mut files = BTreeSet::new();
    for line in diff.lines() {
        if let Some(path) = line.strip_prefix("+++ ") {
            if let Some(parsed) = parse_patch_path(path) {
                files.insert(parsed);
            }
        } else if let Some(path) = line.strip_prefix("--- ")
            && let Some(parsed) = parse_patch_path(path)
        {
            files.insert(parsed);
        }
    }
    files.into_iter().collect()
}

fn parse_patch_path(raw: &str) -> Option<String> {
    if raw == "/dev/null" {
        return None;
    }
    let normalized = raw
        .strip_prefix("a/")
        .or_else(|| raw.strip_prefix("b/"))
        .unwrap_or(raw);
    if normalized.is_empty() {
        return None;
    }
    Some(normalized.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_tracks_target_files_and_base_hash() {
        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-diff-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        fs::write(workspace.join("demo.txt"), "before\n").expect("seed");

        let store = PatchStore::new(&workspace).expect("store");
        let diff = "diff --git a/demo.txt b/demo.txt\n--- a/demo.txt\n+++ b/demo.txt\n@@ -1 +1 @@\n-before\n+after\n";
        let patch = store.stage(diff, &[]).expect("stage");

        assert_eq!(patch.target_files, vec!["demo.txt".to_string()]);
        assert!(!patch.base_sha256.is_empty());
    }

    #[test]
    fn apply_records_base_mismatch_metadata() {
        if !git_available() {
            return;
        }

        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-diff-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        git(&workspace, &["init"]).expect("git init");
        git(
            &workspace,
            &["config", "user.email", "deepseek@example.test"],
        )
        .expect("git config email");
        git(&workspace, &["config", "user.name", "CodingBuddy"]).expect("git config name");
        fs::write(workspace.join("demo.txt"), "before\n").expect("seed");
        git(&workspace, &["add", "."]).expect("git add");
        git(&workspace, &["commit", "-m", "init"]).expect("git commit");

        fs::write(workspace.join("demo.txt"), "after\n").expect("edit");
        let diff = git(&workspace, &["diff"]).expect("git diff");
        git(&workspace, &["checkout", "--", "demo.txt"]).expect("git checkout");

        let store = PatchStore::new(&workspace).expect("store");
        let patch = store.stage(&diff, &[]).expect("stage");
        fs::write(workspace.join("demo.txt"), "drifted\n").expect("drift");
        let _ = store.apply(&workspace, patch.patch_id).expect("apply");

        let refreshed = store
            .list()
            .expect("list")
            .into_iter()
            .find(|p| p.patch_id == patch.patch_id)
            .expect("patched record");
        assert_eq!(refreshed.last_base_sha_match, Some(false));
        assert!(refreshed.last_base_sha256.is_some());
    }

    fn git_available() -> bool {
        Command::new("git")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn git(workspace: &Path, args: &[&str]) -> Result<String> {
        let output = Command::new("git")
            .args(args)
            .current_dir(workspace)
            .output()
            .context("git command failed")?;
        if !output.status.success() {
            return Err(anyhow!(
                "git {:?} failed: {}",
                args,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    // ── extract_target_files tests ──

    #[test]
    fn extract_target_files_from_standard_diff() {
        let diff = "\
diff --git a/src/main.rs b/src/main.rs
--- a/src/main.rs
+++ b/src/main.rs
@@ -1 +1 @@
-old
+new
";
        let files = extract_target_files(diff);
        assert_eq!(files, vec!["src/main.rs"]);
    }

    #[test]
    fn extract_target_files_from_multi_file_diff() {
        let diff = "\
--- a/foo.rs
+++ b/foo.rs
@@ -1 +1 @@
-x
+y
--- a/bar.rs
+++ b/bar.rs
@@ -1 +1 @@
-a
+b
";
        let files = extract_target_files(diff);
        assert_eq!(files, vec!["bar.rs", "foo.rs"]); // BTreeSet sorts
    }

    #[test]
    fn extract_target_files_skips_dev_null() {
        let diff = "\
--- /dev/null
+++ b/new_file.rs
@@ -0,0 +1 @@
+content
";
        let files = extract_target_files(diff);
        assert_eq!(files, vec!["new_file.rs"]);
    }

    #[test]
    fn extract_target_files_empty_diff() {
        assert!(extract_target_files("").is_empty());
    }

    #[test]
    fn extract_target_files_deduplicates() {
        let diff = "\
--- a/same.rs
+++ b/same.rs
@@ -1 +1 @@
-a
+b
";
        let files = extract_target_files(diff);
        assert_eq!(files, vec!["same.rs"]);
    }

    // ── parse_patch_path tests ──

    #[test]
    fn parse_patch_path_strips_a_prefix() {
        assert_eq!(
            parse_patch_path("a/src/lib.rs"),
            Some("src/lib.rs".to_string())
        );
    }

    #[test]
    fn parse_patch_path_strips_b_prefix() {
        assert_eq!(
            parse_patch_path("b/src/lib.rs"),
            Some("src/lib.rs".to_string())
        );
    }

    #[test]
    fn parse_patch_path_no_prefix() {
        assert_eq!(
            parse_patch_path("src/lib.rs"),
            Some("src/lib.rs".to_string())
        );
    }

    #[test]
    fn parse_patch_path_dev_null_returns_none() {
        assert_eq!(parse_patch_path("/dev/null"), None);
    }

    #[test]
    fn parse_patch_path_empty_returns_none() {
        assert_eq!(parse_patch_path(""), None);
        // After stripping a/ prefix, nothing remains
        assert_eq!(parse_patch_path("a/"), None);
    }

    // ── sha256_hex tests ──

    #[test]
    fn sha256_hex_deterministic() {
        let h1 = sha256_hex(b"hello");
        let h2 = sha256_hex(b"hello");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // SHA-256 = 64 hex chars
    }

    #[test]
    fn sha256_hex_different_inputs() {
        assert_ne!(sha256_hex(b"a"), sha256_hex(b"b"));
    }

    // ── hash_workspace_state tests ──

    #[test]
    fn hash_workspace_state_changes_when_file_changes() {
        let ws = std::env::temp_dir().join(format!("cb-diff-hash-{}", Uuid::now_v7()));
        fs::create_dir_all(&ws).unwrap();
        fs::write(ws.join("f.txt"), "v1").unwrap();

        let h1 = hash_workspace_state(&ws, &["f.txt".to_string()]).unwrap();
        fs::write(ws.join("f.txt"), "v2").unwrap();
        let h2 = hash_workspace_state(&ws, &["f.txt".to_string()]).unwrap();

        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_workspace_state_handles_missing_files() {
        let ws = std::env::temp_dir().join(format!("cb-diff-missing-{}", Uuid::now_v7()));
        fs::create_dir_all(&ws).unwrap();

        // Should not panic on missing file — uses [0] sentinel
        let h = hash_workspace_state(&ws, &["nonexistent.rs".to_string()]).unwrap();
        assert!(!h.is_empty());
    }

    // ── PatchStore CRUD tests ──

    #[test]
    fn patch_store_list_returns_empty_initially() {
        let ws = std::env::temp_dir().join(format!("cb-diff-list-{}", Uuid::now_v7()));
        fs::create_dir_all(&ws).unwrap();
        let store = PatchStore::new(&ws).unwrap();
        assert!(store.list().unwrap().is_empty());
    }

    #[test]
    fn patch_store_stage_and_list_roundtrip() {
        let ws = std::env::temp_dir().join(format!("cb-diff-rt-{}", Uuid::now_v7()));
        fs::create_dir_all(&ws).unwrap();
        fs::write(ws.join("x.txt"), "orig").unwrap();

        let store = PatchStore::new(&ws).unwrap();
        let diff = "--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-orig\n+new\n";
        let patch = store.stage(diff, &[]).unwrap();

        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].patch_id, patch.patch_id);
        assert_eq!(listed[0].target_files, vec!["x.txt"]);
        assert!(!listed[0].applied);
    }

    #[test]
    fn patch_store_read_unknown_id_errors() {
        let ws = std::env::temp_dir().join(format!("cb-diff-unknown-{}", Uuid::now_v7()));
        fs::create_dir_all(&ws).unwrap();
        let store = PatchStore::new(&ws).unwrap();
        assert!(store.read_patch(Uuid::nil()).is_err());
    }

    #[test]
    fn patch_store_stage_with_base_blob() {
        let ws = std::env::temp_dir().join(format!("cb-diff-blob-{}", Uuid::now_v7()));
        fs::create_dir_all(&ws).unwrap();

        let store = PatchStore::new(&ws).unwrap();
        let diff = "--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-a\n+b\n";
        let patch = store.stage(diff, b"explicit base content").unwrap();

        // base_sha256 should be the hash of the explicit blob, not workspace files
        assert_eq!(patch.base_sha256, sha256_hex(b"explicit base content"));
    }
}
