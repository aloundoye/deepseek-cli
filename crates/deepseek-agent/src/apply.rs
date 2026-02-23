use anyhow::Result;
use deepseek_diff::PatchStore;
use std::collections::HashSet;
use std::path::Path;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct ApplySuccess {
    pub patch_id: Uuid,
    pub changed_files: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ApplyFailure {
    pub reason: String,
    pub conflicts: Vec<String>,
}

impl ApplyFailure {
    pub fn to_feedback(&self) -> String {
        if self.conflicts.is_empty() {
            self.reason.clone()
        } else {
            format!("{}\n{}", self.reason, self.conflicts.join("\n"))
        }
    }
}

pub fn apply_unified_diff(
    workspace: &Path,
    diff: &str,
    allowed_files: &HashSet<String>,
) -> std::result::Result<ApplySuccess, ApplyFailure> {
    if diff.trim().is_empty() {
        return Err(ApplyFailure {
            reason: "empty diff".to_string(),
            conflicts: vec![],
        });
    }
    if !diff.contains("--- ") || !diff.contains("+++ ") || !diff.contains("@@") {
        return Err(ApplyFailure {
            reason: "invalid unified diff markers".to_string(),
            conflicts: vec![],
        });
    }

    let target_files = extract_target_files(diff);
    if target_files.is_empty() {
        return Err(ApplyFailure {
            reason: "diff does not contain target files".to_string(),
            conflicts: vec![],
        });
    }

    for path in &target_files {
        if path.starts_with('/') {
            return Err(ApplyFailure {
                reason: format!("absolute paths are forbidden: {path}"),
                conflicts: vec![],
            });
        }
        if path.starts_with(".git/") || path == ".git" {
            return Err(ApplyFailure {
                reason: format!(".git mutation forbidden: {path}"),
                conflicts: vec![],
            });
        }
        if !allowed_files.contains(path) {
            return Err(ApplyFailure {
                reason: format!("diff targets undeclared file: {path}"),
                conflicts: vec![],
            });
        }
    }

    let store = PatchStore::new(workspace).map_err(map_anyhow)?;
    let staged = store.stage(diff, &[]).map_err(map_anyhow)?;
    let (applied, conflicts) = store
        .apply(workspace, staged.patch_id)
        .map_err(map_anyhow)?;

    if !applied {
        return Err(ApplyFailure {
            reason: "git apply failed".to_string(),
            conflicts,
        });
    }

    Ok(ApplySuccess {
        patch_id: staged.patch_id,
        changed_files: staged.target_files,
    })
}

fn map_anyhow(err: anyhow::Error) -> ApplyFailure {
    ApplyFailure {
        reason: err.to_string(),
        conflicts: vec![],
    }
}

fn extract_target_files(diff: &str) -> Vec<String> {
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

pub fn _validate(workspace: &Path, diff: &str, allowed_files: &HashSet<String>) -> Result<()> {
    apply_unified_diff(workspace, diff, allowed_files)
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!(e.to_feedback()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_paths() {
        let diff = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-x\n+y\n";
        let files = extract_target_files(diff);
        assert_eq!(files, vec!["src/lib.rs".to_string()]);
    }
}
