use anyhow::Result;
use deepseek_core::ApplyStrategy;
use deepseek_diff::{GitApplyStrategy, PatchStore};
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::Path;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct ApplySuccess {
    pub patch_id: Uuid,
    pub changed_files: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyFailureClass {
    PatchMismatch,
}

#[derive(Debug, Clone)]
pub struct ApplyFailure {
    pub class: ApplyFailureClass,
    pub reason: String,
    pub conflicts: Vec<String>,
    pub changed_files: Vec<String>,
}

impl ApplyFailure {
    pub fn to_feedback(&self) -> String {
        let mut lines = vec![
            format!("classification={:?}", self.class),
            self.reason.clone(),
        ];
        if !self.changed_files.is_empty() {
            lines.push(format!("changed_files={}", self.changed_files.join(",")));
        }
        if self.conflicts.is_empty() {
            lines.join("\n")
        } else {
            lines.push(self.conflicts.join("\n"));
            lines.join("\n")
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiffStats {
    pub touched_files: usize,
    pub loc_delta: usize,
}

pub fn diff_stats(diff: &str) -> DiffStats {
    let touched_files = extract_target_files(diff).len();
    let mut loc_delta = 0usize;
    for line in diff.lines() {
        if line.starts_with("+++ ") || line.starts_with("--- ") {
            continue;
        }
        if line.starts_with('+') || line.starts_with('-') {
            loc_delta = loc_delta.saturating_add(1);
        }
    }
    DiffStats {
        touched_files,
        loc_delta,
    }
}

pub fn hash_text(text: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

pub fn hash_file(workspace: &Path, rel_path: &str) -> Result<String> {
    let path = workspace.join(rel_path);
    let content = std::fs::read_to_string(path)?;
    Ok(hash_text(&content))
}

pub fn extract_target_files(diff: &str) -> Vec<String> {
    let mut files = Vec::new();
    for line in diff.lines() {
        let path = if let Some(raw) = line.strip_prefix("+++ ") {
            parse_path(raw)
        } else if let Some(raw) = line.strip_prefix("--- ") {
            parse_path(raw)
        } else {
            None
        };
        if let Some(path) = path
            && !files.contains(&path)
        {
            files.push(path);
        }
    }
    files
}

pub fn ensure_repo_relative_path(path: &str) -> std::result::Result<(), ApplyFailure> {
    let rel = Path::new(path);
    if rel.is_absolute() {
        return Err(ApplyFailure {
            class: ApplyFailureClass::PatchMismatch,
            reason: format!("absolute paths are forbidden: {path}"),
            conflicts: vec![],
            changed_files: vec![path.to_string()],
        });
    }
    if rel
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(ApplyFailure {
            class: ApplyFailureClass::PatchMismatch,
            reason: format!("path escapes repository root: {path}"),
            conflicts: vec![],
            changed_files: vec![path.to_string()],
        });
    }
    Ok(())
}

pub fn apply_unified_diff(
    workspace: &Path,
    diff: &str,
    allowed_files: &HashSet<String>,
    expected_hashes: &HashMap<String, String>,
    apply_strategy: ApplyStrategy,
) -> std::result::Result<ApplySuccess, ApplyFailure> {
    if diff.trim().is_empty() {
        return Err(ApplyFailure {
            class: ApplyFailureClass::PatchMismatch,
            reason: "empty diff".to_string(),
            conflicts: vec![],
            changed_files: vec![],
        });
    }
    if !diff.contains("--- ") || !diff.contains("+++ ") || !diff.contains("@@") {
        return Err(ApplyFailure {
            class: ApplyFailureClass::PatchMismatch,
            reason: "invalid unified diff markers".to_string(),
            conflicts: vec![],
            changed_files: vec![],
        });
    }

    let target_files = extract_target_files(diff);
    if target_files.is_empty() {
        return Err(ApplyFailure {
            class: ApplyFailureClass::PatchMismatch,
            reason: "diff does not contain target files".to_string(),
            conflicts: vec![],
            changed_files: vec![],
        });
    }

    for path in &target_files {
        ensure_repo_relative_path(path)?;
        if path.starts_with(".git/") || path == ".git" {
            return Err(ApplyFailure {
                class: ApplyFailureClass::PatchMismatch,
                reason: format!(".git mutation forbidden: {path}"),
                conflicts: vec![],
                changed_files: target_files.clone(),
            });
        }
        if !allowed_files.contains(path) {
            return Err(ApplyFailure {
                class: ApplyFailureClass::PatchMismatch,
                reason: format!("diff targets undeclared file: {path}"),
                conflicts: vec![],
                changed_files: target_files.clone(),
            });
        }
        if let Some(expected) = expected_hashes.get(path)
            && let Ok(current) = hash_file(workspace, path)
            && expected != &current
        {
            return Err(ApplyFailure {
                class: ApplyFailureClass::PatchMismatch,
                reason: format!("stale editor context hash mismatch: {path}"),
                conflicts: vec![],
                changed_files: target_files.clone(),
            });
        }
    }

    let store = PatchStore::new(workspace).map_err(map_anyhow)?;
    let staged = store.stage(diff, &[]).map_err(map_anyhow)?;
    let strategy = match apply_strategy {
        ApplyStrategy::Auto => GitApplyStrategy::Auto,
        ApplyStrategy::ThreeWay => GitApplyStrategy::ThreeWay,
    };
    let (applied, conflicts) = store
        .apply_with_strategy(workspace, staged.patch_id, strategy)
        .map_err(map_anyhow)?;

    if !applied {
        return Err(ApplyFailure {
            class: ApplyFailureClass::PatchMismatch,
            reason: "patch apply failed".to_string(),
            conflicts,
            changed_files: target_files,
        });
    }

    Ok(ApplySuccess {
        patch_id: staged.patch_id,
        changed_files: staged.target_files,
    })
}

fn map_anyhow(err: anyhow::Error) -> ApplyFailure {
    ApplyFailure {
        class: ApplyFailureClass::PatchMismatch,
        reason: err.to_string(),
        conflicts: vec![],
        changed_files: vec![],
    }
}

fn parse_path(raw: &str) -> Option<String> {
    if raw == "/dev/null" {
        return None;
    }
    let normalized = raw
        .strip_prefix("a/")
        .or_else(|| raw.strip_prefix("b/"))
        .unwrap_or(raw)
        .trim();
    if normalized.is_empty() {
        return None;
    }
    Some(normalized.to_string())
}

pub fn _validate(
    workspace: &Path,
    diff: &str,
    allowed_files: &HashSet<String>,
    expected_hashes: &HashMap<String, String>,
    apply_strategy: ApplyStrategy,
) -> Result<()> {
    apply_unified_diff(
        workspace,
        diff,
        allowed_files,
        expected_hashes,
        apply_strategy,
    )
    .map(|_| ())
    .map_err(|e| anyhow::anyhow!(e.to_feedback()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepseek_core::ApplyStrategy;
    use std::collections::{HashMap, HashSet};
    use std::fs;

    #[test]
    fn parse_paths() {
        let diff = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-x\n+y\n";
        let files = extract_target_files(diff);
        assert_eq!(files, vec!["src/lib.rs".to_string()]);
    }

    #[test]
    fn rejects_parent_dir_escape() {
        let err = ensure_repo_relative_path("../secret").expect_err("should reject");
        assert!(err.reason.contains("repository root"));
    }

    #[test]
    fn rejects_outside_repo_root_paths_from_diff() {
        let temp = tempfile::tempdir().expect("tempdir");
        fs::write(temp.path().join("demo.txt"), "before\n").expect("seed");
        let diff = "--- a/../secret.txt\n+++ b/../secret.txt\n@@ -0,0 +1 @@\n+nope\n";
        let allowed = HashSet::from(["../secret.txt".to_string()]);
        let err = apply_unified_diff(
            temp.path(),
            diff,
            &allowed,
            &HashMap::new(),
            ApplyStrategy::Auto,
        )
        .expect_err("must reject root escape");
        assert!(err.reason.contains("repository root"));
    }

    #[test]
    fn rejects_stale_editor_hash_mismatch() {
        let temp = tempfile::tempdir().expect("tempdir");
        fs::write(temp.path().join("demo.txt"), "before\n").expect("seed");
        let diff = "--- a/demo.txt\n+++ b/demo.txt\n@@ -1 +1 @@\n-before\n+after\n";
        let allowed = HashSet::from(["demo.txt".to_string()]);
        let expected = HashMap::from([("demo.txt".to_string(), "deadbeef".to_string())]);
        let err = apply_unified_diff(temp.path(), diff, &allowed, &expected, ApplyStrategy::Auto)
            .expect_err("must reject stale mismatch");
        assert!(err.reason.contains("stale editor context hash mismatch"));
    }

    #[test]
    fn diff_stats_counts_lines_and_files() {
        let diff = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1,3 +1,3 @@\n context\n-old line\n+new line\n context\n";
        let stats = diff_stats(diff);
        assert_eq!(stats.touched_files, 1);
        assert_eq!(stats.loc_delta, 2); // one - and one +
    }

    #[test]
    fn diff_stats_multi_file() {
        let diff = "\
--- a/src/a.rs
+++ b/src/a.rs
@@ -1 +1 @@
-old_a
+new_a
--- a/src/b.rs
+++ b/src/b.rs
@@ -1 +1 @@
-old_b
+new_b
";
        let stats = diff_stats(diff);
        assert_eq!(stats.touched_files, 2);
        assert_eq!(stats.loc_delta, 4);
    }

    #[test]
    fn diff_stats_empty_diff() {
        let stats = diff_stats("");
        assert_eq!(stats.touched_files, 0);
        assert_eq!(stats.loc_delta, 0);
    }

    #[test]
    fn hash_text_deterministic() {
        let a = hash_text("hello world");
        let b = hash_text("hello world");
        assert_eq!(a, b);
        assert_eq!(a.len(), 16); // 16 hex chars
    }

    #[test]
    fn hash_text_differs_for_different_input() {
        let a = hash_text("hello");
        let b = hash_text("world");
        assert_ne!(a, b);
    }

    #[test]
    fn extract_target_files_deduplicates() {
        let diff = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-a\n+b\n";
        let files = extract_target_files(diff);
        assert_eq!(files, vec!["src/lib.rs"]);
    }

    #[test]
    fn extract_target_files_handles_dev_null() {
        let diff = "--- /dev/null\n+++ b/new_file.rs\n@@ -0,0 +1 @@\n+content\n";
        let files = extract_target_files(diff);
        assert_eq!(files, vec!["new_file.rs"]);
    }

    #[test]
    fn ensure_repo_relative_accepts_normal_paths() {
        assert!(ensure_repo_relative_path("src/lib.rs").is_ok());
        assert!(ensure_repo_relative_path("nested/deep/file.txt").is_ok());
    }

    #[test]
    fn ensure_repo_relative_rejects_absolute() {
        let err = ensure_repo_relative_path("/etc/passwd").unwrap_err();
        assert!(err.reason.contains("absolute"));
    }

    #[test]
    fn ensure_repo_relative_rejects_parent_traversal() {
        let err = ensure_repo_relative_path("foo/../../etc/passwd").unwrap_err();
        assert!(err.reason.contains("repository root"));
    }

    #[test]
    fn rejects_empty_diff() {
        let allowed = HashSet::new();
        let err =
            apply_unified_diff(Path::new("/tmp"), "", &allowed, &HashMap::new(), ApplyStrategy::Auto)
                .unwrap_err();
        assert!(err.reason.contains("empty diff"));
    }

    #[test]
    fn rejects_invalid_diff_markers() {
        let allowed = HashSet::new();
        let err = apply_unified_diff(
            Path::new("/tmp"),
            "not a diff at all",
            &allowed,
            &HashMap::new(),
            ApplyStrategy::Auto,
        )
        .unwrap_err();
        assert!(err.reason.contains("invalid unified diff markers"));
    }

    #[test]
    fn rejects_undeclared_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let diff = "--- a/secret.rs\n+++ b/secret.rs\n@@ -1 +1 @@\n-a\n+b\n";
        let allowed = HashSet::from(["other.rs".to_string()]);
        let err = apply_unified_diff(
            temp.path(),
            diff,
            &allowed,
            &HashMap::new(),
            ApplyStrategy::Auto,
        )
        .unwrap_err();
        assert!(err.reason.contains("undeclared file"));
    }

    #[test]
    fn rejects_git_dir_mutation() {
        let temp = tempfile::tempdir().expect("tempdir");
        let diff = "--- a/.git/config\n+++ b/.git/config\n@@ -1 +1 @@\n-a\n+b\n";
        let allowed = HashSet::from([".git/config".to_string()]);
        let err = apply_unified_diff(
            temp.path(),
            diff,
            &allowed,
            &HashMap::new(),
            ApplyStrategy::Auto,
        )
        .unwrap_err();
        assert!(err.reason.contains(".git mutation forbidden"));
    }

    #[test]
    fn apply_failure_to_feedback_format() {
        let failure = ApplyFailure {
            class: ApplyFailureClass::PatchMismatch,
            reason: "test reason".to_string(),
            conflicts: vec!["conflict1".to_string()],
            changed_files: vec!["file.rs".to_string()],
        };
        let feedback = failure.to_feedback();
        assert!(feedback.contains("PatchMismatch"));
        assert!(feedback.contains("test reason"));
        assert!(feedback.contains("conflict1"));
        assert!(feedback.contains("file.rs"));
    }
}
