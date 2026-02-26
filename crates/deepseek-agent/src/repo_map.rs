use deepseek_context::ContextManager;
use deepseek_context::tags::{TagExtractor, tags_to_symbol_hints};
use deepseek_index::IndexService;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone)]
pub struct RepoMapEntry {
    pub path: String,
    pub score: f32,
    pub reasons: Vec<String>,
    pub symbol_hints: Vec<String>,
}

pub fn build_repo_map(
    workspace: &Path,
    prompt: &str,
    max_lines: usize,
    additional_dirs: &[PathBuf],
) -> Vec<RepoMapEntry> {
    let mut merged: BTreeMap<String, RepoMapEntry> = BTreeMap::new();

    if let Ok(mut manager) = ContextManager::new(workspace) {
        let _ = manager.analyze_workspace();
        for suggestion in manager.suggest_relevant_files(prompt, max_lines.max(8)) {
            let rel = make_relative(workspace, &suggestion.path);
            upsert_entry(
                &mut merged,
                rel.clone(),
                suggestion.score,
                suggestion.reasons,
                symbol_hints_for_file(&workspace.join(&rel)),
            );
        }
    }

    if let Ok(index) = IndexService::new(workspace)
        && let Ok(query) = index.query(prompt, max_lines.max(8), None)
    {
        for result in query.results {
            upsert_entry(
                &mut merged,
                result.path.clone(),
                1.0,
                vec![format!("index match line {}", result.line)],
                symbol_hints_for_file(&workspace.join(&result.path)),
            );
        }
    }

    for changed in changed_files(workspace) {
        upsert_entry(
            &mut merged,
            changed.clone(),
            2.0,
            vec!["changed file".to_string()],
            symbol_hints_for_file(&workspace.join(&changed)),
        );
    }

    for dir in additional_dirs {
        let joined = if dir.is_absolute() {
            dir.clone()
        } else {
            workspace.join(dir)
        };
        if let Ok(entries) = std::fs::read_dir(&joined) {
            for entry in entries.flatten().take(12) {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let rel = make_relative(workspace, &path);
                upsert_entry(
                    &mut merged,
                    rel.clone(),
                    0.6,
                    vec!["additional_dir".to_string()],
                    symbol_hints_for_file(&workspace.join(&rel)),
                );
            }
        }
    }

    let mut rows = merged.into_values().collect::<Vec<_>>();
    rows.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    });
    rows.truncate(max_lines.max(1));
    rows
}

pub fn render_repo_map(entries: &[RepoMapEntry]) -> String {
    let mut out = Vec::new();
    for entry in entries {
        let mut line = format!("- {} score={:.2}", entry.path, entry.score);
        if !entry.reasons.is_empty() {
            line.push_str(&format!(" reasons={}", entry.reasons.join("|")));
        }
        if !entry.symbol_hints.is_empty() {
            line.push_str(&format!(" symbols={}", entry.symbol_hints.join(",")));
        }
        out.push(line);
    }
    out.join("\n")
}

fn upsert_entry(
    merged: &mut BTreeMap<String, RepoMapEntry>,
    path: String,
    score: f32,
    reasons: Vec<String>,
    symbol_hints: Vec<String>,
) {
    let entry = merged.entry(path.clone()).or_insert_with(|| RepoMapEntry {
        path,
        score: 0.0,
        reasons: Vec::new(),
        symbol_hints: Vec::new(),
    });
    entry.score += score;
    for reason in reasons {
        if !entry.reasons.contains(&reason) {
            entry.reasons.push(reason);
        }
    }
    for hint in symbol_hints {
        if !entry.symbol_hints.contains(&hint) {
            entry.symbol_hints.push(hint);
        }
    }
}

fn make_relative(workspace: &Path, path: &Path) -> String {
    path.strip_prefix(workspace)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
}

/// Clear the global tag cache (used by `/map-refresh`).
pub fn clear_tag_cache() {
    let _ = with_tag_extractor(|e| {
        let _ = e.clear_cache();
    });
}

/// Global tag extractor with SQLite cache, lazily initialized.
/// Wrapped in Mutex because rusqlite::Connection is not Sync.
fn with_tag_extractor<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&TagExtractor) -> R,
{
    static EXTRACTOR: OnceLock<Mutex<Option<TagExtractor>>> = OnceLock::new();
    let mutex = EXTRACTOR.get_or_init(|| {
        let cache_dir = dirs_cache().unwrap_or_else(|| PathBuf::from(".deepseek/cache"));
        Mutex::new(TagExtractor::new(&cache_dir).ok())
    });
    let guard = mutex.lock().ok()?;
    let extractor = guard.as_ref()?;
    Some(f(extractor))
}

fn dirs_cache() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".deepseek").join("cache"))
}

fn symbol_hints_for_file(path: &Path) -> Vec<String> {
    // Try tree-sitter extraction first (supports 6 languages with caching)
    if let Some(hints) = with_tag_extractor(|extractor| {
        extractor
            .extract_tags(path)
            .ok()
            .filter(|tags| !tags.is_empty())
            .map(|tags| tags_to_symbol_hints(&tags, 8))
    })
        .flatten()
    {
        return hints;
    }

    // Fallback to naive regex extraction for unsupported languages
    let Ok(content) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for line in content.lines().take(240) {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("fn ") {
            out.push(rest.split('(').next().unwrap_or_default().to_string());
        } else if let Some(rest) = trimmed.strip_prefix("pub fn ") {
            out.push(rest.split('(').next().unwrap_or_default().to_string());
        } else if let Some(rest) = trimmed.strip_prefix("struct ") {
            out.push(
                rest.split_whitespace()
                    .next()
                    .unwrap_or_default()
                    .to_string(),
            );
        } else if let Some(rest) = trimmed.strip_prefix("class ") {
            out.push(
                rest.split_whitespace()
                    .next()
                    .unwrap_or_default()
                    .to_string(),
            );
        }
        if out.len() >= 4 {
            break;
        }
    }
    out
}

fn changed_files(workspace: &Path) -> Vec<String> {
    let Ok(output) = Command::new("git")
        .arg("-C")
        .arg(workspace)
        .args(["status", "--porcelain"])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let trimmed = line.trim();
        if trimmed.len() < 3 {
            continue;
        }
        out.push(trimmed[3..].trim().to_string());
    }
    out.sort();
    out.dedup();
    out
}
