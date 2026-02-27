//! Shared helpers used across agent modules.
//!
//! Functions relocated here from `architect.rs` during the pipeline removal
//! rewrite. `gather_context.rs` depends on `build_repo_map`.

use crate::repo_map;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Build a repo-map string for the given workspace.
///
/// Tries the tree-sitter-based repo map first; falls back to a simple
/// file-listing heuristic ranked by git status and prompt keyword overlap.
pub fn build_repo_map(
    workspace: &Path,
    user_prompt: &str,
    max_files: usize,
    additional_dirs: &[PathBuf],
) -> String {
    let rows = repo_map::build_repo_map(workspace, user_prompt, max_files, additional_dirs);
    if !rows.is_empty() {
        return repo_map::render_repo_map(&rows);
    }

    let mut files = tracked_files(workspace);
    for dir in additional_dirs {
        let joined = if dir.is_absolute() {
            dir.clone()
        } else {
            workspace.join(dir)
        };
        files.extend(list_files(&joined, Some(workspace)));
    }

    files.sort();
    files.dedup();

    let changed = changed_files(workspace);
    let tokens = prompt_tokens(user_prompt);

    let mut scored: Vec<(i32, String)> = files
        .into_iter()
        .map(|path| {
            let mut score = 0;
            if changed.contains(&path) {
                score += 100;
            }
            let lower = path.to_ascii_lowercase();
            for token in &tokens {
                if lower.contains(token) {
                    score += 10;
                }
            }
            if lower.ends_with("readme.md") || lower.ends_with("specs.md") {
                score += 5;
            }
            (score, path)
        })
        .collect();

    scored.sort_by(|a, b| b.cmp(a));

    let mut lines = Vec::new();
    for (count, (score, path)) in scored.into_iter().enumerate() {
        if count >= max_files.max(1) {
            break;
        }
        if score <= 0 && count >= max_files / 2 {
            break;
        }
        let size = workspace
            .join(&path)
            .metadata()
            .ok()
            .map(|m| m.len())
            .unwrap_or(0);
        lines.push(format!("- {path} ({size} bytes) score={score}"));
    }

    lines.join("\n")
}

fn tracked_files(workspace: &Path) -> Vec<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(workspace)
        .arg("ls-files")
        .output();
    if let Ok(out) = output
        && out.status.success()
    {
        return String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToString::to_string)
            .collect();
    }
    list_files(workspace, None)
}

fn list_files(root: &Path, workspace: Option<&Path>) -> Vec<String> {
    let mut out = Vec::new();
    let walker = ignore::WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .git_exclude(true)
        .add_custom_ignore_filename(".deepseekignore")
        .build();
    for entry in walker.flatten() {
        if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.into_path();
        let rel = if let Some(base) = workspace {
            path.strip_prefix(base).ok().map(|p| p.to_path_buf())
        } else {
            path.strip_prefix(root).ok().map(|p| p.to_path_buf())
        };
        if let Some(rel) = rel {
            out.push(rel.to_string_lossy().to_string());
        }
    }
    out
}

fn changed_files(workspace: &Path) -> HashSet<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(workspace)
        .args(["status", "--porcelain"])
        .output();
    let mut set = HashSet::new();
    if let Ok(out) = output
        && out.status.success()
    {
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            let trimmed = line.trim();
            if trimmed.len() >= 3 {
                set.insert(trimmed[3..].trim().to_string());
            }
        }
    }
    set
}

fn prompt_tokens(prompt: &str) -> HashSet<String> {
    prompt
        .split(|c: char| !c.is_ascii_alphanumeric())
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| s.len() >= 3)
        .collect()
}
