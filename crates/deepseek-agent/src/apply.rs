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
}
