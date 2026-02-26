use crate::ChatMode;
use crate::shared;
use deepseek_core::AppConfig;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const HEAVY_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".venv",
    "venv",
    "__pycache__",
    ".idea",
    ".vscode",
    ".cache",
];

const MANIFEST_NAMES: &[&str] = &[
    "Cargo.toml",
    "package.json",
    "pyproject.toml",
    "go.mod",
    "requirements.txt",
    "pom.xml",
    "build.gradle",
    "Gemfile",
];

#[derive(Debug, Clone, Default)]
pub struct AutoContextBootstrap {
    pub enabled: bool,
    pub repoish: bool,
    pub vague_codebase_prompt: bool,
    pub packet: String,
    pub repo_root: Option<PathBuf>,
    pub manifest_type: Option<String>,
    pub manifest_path: Option<String>,
    pub readme_path: Option<String>,
    pub readme_bytes: usize,
    pub root_tree_count: usize,
    pub repo_map_lines: usize,
    pub repo_map_estimated_tokens: usize,
    pub unavailable_reason: Option<String>,
}

#[derive(Debug, Clone)]
struct ManifestEntry {
    path: String,
    name: String,
    excerpt: String,
}

pub fn gather_for_prompt(
    workspace: &Path,
    cfg: &AppConfig,
    prompt: &str,
    mode: ChatMode,
    additional_dirs: &[PathBuf],
    repo_root_override: Option<&Path>,
) -> AutoContextBootstrap {
    if !cfg.agent_loop.context_bootstrap_enabled {
        return AutoContextBootstrap {
            enabled: false,
            ..Default::default()
        };
    }

    let explicit_repoish = is_repoish_prompt(prompt);
    let implicit_repoish = matches!(mode, ChatMode::Context | ChatMode::Code | ChatMode::Ask);
    let wants_context = explicit_repoish || implicit_repoish;

    if !wants_context {
        return AutoContextBootstrap {
            enabled: true,
            repoish: false,
            vague_codebase_prompt: false,
            packet: String::new(),
            repo_root: resolve_repo_root(workspace, repo_root_override),
            ..Default::default()
        };
    }

    let vague_codebase_prompt = is_vague_codebase_prompt(prompt);
    let Some(repo_root) = resolve_repo_root(workspace, repo_root_override) else {
        if explicit_repoish {
            return AutoContextBootstrap {
                enabled: true,
                repoish: true,
                vague_codebase_prompt,
                packet: String::new(),
                unavailable_reason: Some(
                    "No repository detected. Run from project root or pass --repo <path>.".to_string(),
                ),
                ..Default::default()
            };
        } else {
            return AutoContextBootstrap {
                enabled: true,
                repoish: false, // Don't trigger hard error inside loop/analysis
                vague_codebase_prompt: false,
                packet: String::new(),
                repo_root: None,
                ..Default::default()
            };
        }
    };
    let tree_entries = collect_tree_snapshot(
        &repo_root,
        cfg.agent_loop.context_bootstrap_max_tree_entries as usize,
        3,
    );
    let readme = read_readme_excerpt(
        &repo_root,
        cfg.agent_loop.context_bootstrap_max_readme_bytes as usize,
    );
    let manifests = discover_manifests(
        &repo_root,
        cfg.agent_loop.context_bootstrap_max_manifest_bytes as usize,
    );
    let primary_manifest = select_primary_manifest(&manifests).cloned();
    let project_types = infer_project_types(&manifests);
    let repo_map = build_repo_map_summary(
        &repo_root,
        prompt,
        cfg.agent_loop.context_bootstrap_max_repo_map_lines as usize,
        additional_dirs,
    );
    let repo_map_lines = repo_map.lines().count();
    let repo_map_estimated_tokens = estimate_tokens(&repo_map);

    let mut lines = vec!["AUTO_CONTEXT_BOOTSTRAP".to_string()];
    lines.push(format!("workspace={}", repo_root.display()));
    lines.push(format!("repoish_prompt=true mode={mode:?}"));
    lines.push(format!(
        "vague_codebase_prompt={}",
        if vague_codebase_prompt {
            "true"
        } else {
            "false"
        }
    ));

    lines.push("GIT_STATUS:".to_string());
    let git_status = git_status_snapshot(&repo_root);
    append_indented_lines(&mut lines, &git_status, "  ", 20);

    lines.push("ROOT_TREE_SNAPSHOT:".to_string());
    if tree_entries.is_empty() {
        lines.push("  (no files discovered)".to_string());
    } else {
        for entry in &tree_entries {
            lines.push(format!("  {entry}"));
        }
    }

    lines.push("PROJECT_TYPE:".to_string());
    if project_types.is_empty() {
        lines.push("  unknown".to_string());
    } else {
        lines.push(format!("  {}", project_types.join(", ")));
    }

    lines.push("MANIFEST_DISCOVERY:".to_string());
    if manifests.is_empty() {
        lines.push("  (none found)".to_string());
    } else {
        for manifest in manifests.iter().take(6) {
            lines.push(format!("  - {} ({})", manifest.path, manifest.name));
            append_indented_lines(&mut lines, &manifest.excerpt, "    ", 8);
        }
    }

    lines.push("README_EXCERPT:".to_string());
    if let Some((path, excerpt)) = &readme {
        lines.push(format!("  file={path}"));
        append_indented_lines(&mut lines, excerpt, "  ", 20);
    } else {
        lines.push("  (README not found)".to_string());
    }

    lines.push("REPO_MAP_SUMMARY:".to_string());
    if repo_map.trim().is_empty() {
        lines.push("  (empty)".to_string());
    } else {
        append_indented_lines(&mut lines, &repo_map, "  ", usize::MAX);
    }

    if vague_codebase_prompt {
        let audit = run_baseline_audit(
            &repo_root,
            cfg.agent_loop.context_bootstrap_max_audit_findings as usize,
            &manifests,
        );
        lines.push("BASELINE_AUDIT:".to_string());
        lines.push("  TODO_FIXME:".to_string());
        if audit.todo_fixme.is_empty() {
            lines.push("    - none found in bounded scan".to_string());
        } else {
            for finding in audit.todo_fixme {
                lines.push(format!("    - {finding}"));
            }
        }

        lines.push("  DEPENDENCIES:".to_string());
        if audit.dependencies.is_empty() {
            lines.push("    - no manifest dependencies detected".to_string());
        } else {
            for dep in audit.dependencies {
                lines.push(format!("    - {dep}"));
            }
        }

        lines.push("  TEST_SURFACE:".to_string());
        if audit.test_surface.is_empty() {
            lines.push("    - no explicit test surface found".to_string());
        } else {
            for test_line in audit.test_surface {
                lines.push(format!("    - {test_line}"));
            }
        }

        lines.push("  RISK_HOTSPOTS:".to_string());
        if audit.risk_hotspots.is_empty() {
            lines.push("    - none in bounded heuristic scan".to_string());
        } else {
            for hotspot in audit.risk_hotspots {
                lines.push(format!("    - {hotspot}"));
            }
        }
    }

    lines.push("AUTO_CONTEXT_BOOTSTRAP_END".to_string());

    AutoContextBootstrap {
        enabled: true,
        repoish: true,
        vague_codebase_prompt,
        packet: lines.join("\n"),
        repo_root: Some(repo_root),
        manifest_type: primary_manifest
            .as_ref()
            .map(|manifest| manifest.name.clone()),
        manifest_path: primary_manifest
            .as_ref()
            .map(|manifest| manifest.path.clone()),
        readme_path: readme.as_ref().map(|(path, _)| path.clone()),
        readme_bytes: readme
            .as_ref()
            .map(|(_, excerpt)| excerpt.len())
            .unwrap_or(0),
        root_tree_count: tree_entries.len(),
        repo_map_lines,
        repo_map_estimated_tokens,
        unavailable_reason: None,
    }
}

pub fn debug_context_enabled(debug_flag: bool) -> bool {
    if debug_flag {
        return true;
    }
    std::env::var("DEEPSEEK_DEBUG_CONTEXT")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

impl AutoContextBootstrap {
    pub fn debug_digest(&self, task_intent: &str, mode: ChatMode) -> String {
        let repo_root = self
            .repo_root
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<none>".to_string());
        let manifest = match (&self.manifest_type, &self.manifest_path) {
            (Some(kind), Some(path)) => format!("{kind} ({path})"),
            _ => "<none>".to_string(),
        };
        let readme = self.readme_path.as_deref().unwrap_or("<none>").to_string();
        let unavailable = self.unavailable_reason.as_deref().unwrap_or("<none>");
        format!(
            "[debug-context] intent={task_intent} mode={mode:?} repo_root={repo_root} manifest={manifest} readme={readme} readme_bytes={} tree_entries={} repo_map_lines={} repo_map_est_tokens={} unavailable_reason={}",
            self.readme_bytes,
            self.root_tree_count,
            self.repo_map_lines,
            self.repo_map_estimated_tokens,
            unavailable
        )
    }
}

fn is_repoish_prompt(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    let inspect_verbs = [
        "analyze", "analyse", "audit", "overview", "inspect", "check", "review",
    ];
    let repo_subjects = [
        "project",
        "repo",
        "repository",
        "codebase",
        "structure",
        "dependencies",
        "tests",
        "security",
        "quality",
        "readme",
        "this project",
        "this codebase",
    ];
    let has_inspect_verb = inspect_verbs.iter().any(|needle| lower.contains(needle));
    let has_repo_subject = repo_subjects.iter().any(|needle| lower.contains(needle));
    if has_inspect_verb && has_repo_subject {
        return true;
    }

    // Also check for explicit project-referencing phrases
    [
        "analyze this project",
        "analyze this codebase",
        "check the codebase",
        "check this project",
        "audit this project",
        "current project",
        "this repo",
        "this repository",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn is_vague_codebase_prompt(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    [
        "check codebase",
        "check the codebase",
        "check this codebase",
        "analyze this project",
        "analyze project",
        "analyze this codebase",
        "check this project",
        "audit this project",
        "audit codebase",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn resolve_repo_root(workspace: &Path, repo_root_override: Option<&Path>) -> Option<PathBuf> {
    if let Some(repo) = repo_root_override {
        if repo.is_dir() {
            return repo
                .canonicalize()
                .ok()
                .or_else(|| Some(repo.to_path_buf()));
        }
        return None;
    }

    let output = Command::new("git")
        .arg("-C")
        .arg(workspace)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let root = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if root.is_empty() {
        None
    } else {
        Some(PathBuf::from(root))
    }
}

fn collect_tree_snapshot(workspace: &Path, max_entries: usize, max_depth: usize) -> Vec<String> {
    let mut out = Vec::new();
    collect_tree_recursive(
        workspace,
        workspace,
        &mut out,
        max_entries.max(1),
        0,
        max_depth,
    );
    out
}

fn collect_tree_recursive(
    root: &Path,
    current: &Path,
    out: &mut Vec<String>,
    max_entries: usize,
    depth: usize,
    max_depth: usize,
) {
    if out.len() >= max_entries {
        return;
    }
    let Ok(entries) = fs::read_dir(current) else {
        return;
    };
    let mut entries = entries.filter_map(|entry| entry.ok()).collect::<Vec<_>>();
    entries.sort_by_key(|a| a.file_name());

    for entry in entries {
        if out.len() >= max_entries {
            break;
        }
        let path = entry.path();
        let rel = match path.strip_prefix(root) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let rel_str = rel.to_string_lossy();
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();

        if path.is_dir() {
            if HEAVY_DIRS.contains(&file_name.as_ref()) {
                continue;
            }
            out.push(format!("{rel_str}/"));
            if depth < max_depth {
                collect_tree_recursive(root, &path, out, max_entries, depth + 1, max_depth);
            }
            continue;
        }

        out.push(rel_str.to_string());
    }
}

fn read_readme_excerpt(workspace: &Path, max_bytes: usize) -> Option<(String, String)> {
    let candidates = ["README.md", "README", "readme.md", "README.rst"];
    for candidate in candidates {
        let path = workspace.join(candidate);
        if !path.exists() {
            continue;
        }
        if let Ok(bytes) = fs::read(&path) {
            let bounded = &bytes[..bytes.len().min(max_bytes.max(1))];
            let text = String::from_utf8_lossy(bounded).to_string();
            return Some((candidate.to_string(), text.trim().to_string()));
        }
    }
    None
}

fn discover_manifests(workspace: &Path, max_bytes: usize) -> Vec<ManifestEntry> {
    let mut out = Vec::new();
    discover_manifests_recursive(workspace, workspace, 0, 3, max_bytes.max(1), &mut out);
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

fn select_primary_manifest(manifests: &[ManifestEntry]) -> Option<&ManifestEntry> {
    for manifest_name in MANIFEST_NAMES {
        if let Some(entry) = manifests.iter().find(|entry| entry.name == *manifest_name) {
            return Some(entry);
        }
    }
    manifests.first()
}

fn discover_manifests_recursive(
    root: &Path,
    current: &Path,
    depth: usize,
    max_depth: usize,
    max_bytes: usize,
    out: &mut Vec<ManifestEntry>,
) {
    let Ok(entries) = fs::read_dir(current) else {
        return;
    };
    let mut entries = entries.filter_map(|entry| entry.ok()).collect::<Vec<_>>();
    entries.sort_by_key(|a| a.file_name());

    for entry in entries {
        let path = entry.path();
        let rel = match path.strip_prefix(root) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let rel_str = rel.to_string_lossy().to_string();
        let file_name = entry.file_name().to_string_lossy().to_string();

        if path.is_dir() {
            if HEAVY_DIRS.contains(&file_name.as_str()) {
                continue;
            }
            if depth < max_depth {
                discover_manifests_recursive(root, &path, depth + 1, max_depth, max_bytes, out);
            }
            continue;
        }

        if !MANIFEST_NAMES.contains(&file_name.as_str()) {
            continue;
        }

        let excerpt = fs::read(&path)
            .ok()
            .map(|bytes| {
                let bounded = &bytes[..bytes.len().min(max_bytes)];
                String::from_utf8_lossy(bounded).to_string()
            })
            .unwrap_or_default();

        out.push(ManifestEntry {
            path: rel_str,
            name: file_name,
            excerpt: excerpt.trim().to_string(),
        });
    }
}

fn infer_project_types(manifests: &[ManifestEntry]) -> Vec<String> {
    let mut kinds = BTreeSet::new();
    for manifest in manifests {
        match manifest.name.as_str() {
            "Cargo.toml" => {
                kinds.insert("Rust".to_string());
            }
            "package.json" => {
                kinds.insert("Node.js".to_string());
            }
            "pyproject.toml" | "requirements.txt" => {
                kinds.insert("Python".to_string());
            }
            "go.mod" => {
                kinds.insert("Go".to_string());
            }
            "pom.xml" | "build.gradle" => {
                kinds.insert("JVM".to_string());
            }
            "Gemfile" => {
                kinds.insert("Ruby".to_string());
            }
            _ => {}
        }
    }
    kinds.into_iter().collect()
}

fn build_repo_map_summary(
    workspace: &Path,
    prompt: &str,
    max_lines: usize,
    additional_dirs: &[PathBuf],
) -> String {
    let summary = shared::build_repo_map(workspace, prompt, max_lines.max(1), additional_dirs);
    summary
        .lines()
        .take(max_lines.max(1))
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Debug, Clone, Default)]
struct BaselineAudit {
    todo_fixme: Vec<String>,
    dependencies: Vec<String>,
    test_surface: Vec<String>,
    risk_hotspots: Vec<String>,
}

fn run_baseline_audit(
    workspace: &Path,
    max_findings: usize,
    manifests: &[ManifestEntry],
) -> BaselineAudit {
    let finding_cap = max_findings.max(1);
    BaselineAudit {
        todo_fixme: scan_todo_fixme(workspace, finding_cap),
        dependencies: summarize_dependencies(workspace, manifests, finding_cap),
        test_surface: summarize_test_surface(workspace, finding_cap),
        risk_hotspots: summarize_risk_hotspots(workspace, finding_cap),
    }
}

fn scan_todo_fixme(workspace: &Path, max_findings: usize) -> Vec<String> {
    let files = collect_text_files(workspace, 1500);
    let mut findings = Vec::new();
    for rel in files {
        if findings.len() >= max_findings {
            break;
        }
        let path = workspace.join(&rel);
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        let bounded = &bytes[..bytes.len().min(120_000)];
        let content = String::from_utf8_lossy(bounded);
        for (idx, line) in content.lines().enumerate() {
            if findings.len() >= max_findings {
                break;
            }
            let lower = line.to_ascii_lowercase();
            if !(lower.contains("todo") || lower.contains("fixme")) {
                continue;
            }
            let snippet = line.trim().chars().take(120).collect::<String>();
            findings.push(format!("{}:{} {}", rel, idx + 1, snippet));
        }
    }
    findings
}

fn summarize_dependencies(
    workspace: &Path,
    manifests: &[ManifestEntry],
    max_findings: usize,
) -> Vec<String> {
    let mut summary = Vec::new();
    for manifest in manifests {
        if summary.len() >= max_findings {
            break;
        }
        let path = workspace.join(&manifest.path);
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        let deps = parse_dependencies(&manifest.name, &content);
        if deps.is_empty() {
            continue;
        }
        let preview = deps.into_iter().take(8).collect::<Vec<_>>().join(", ");
        summary.push(format!("{}: {}", manifest.path, preview));
    }
    summary
}

fn parse_dependencies(manifest_name: &str, content: &str) -> Vec<String> {
    match manifest_name {
        "Cargo.toml" => parse_cargo_dependencies(content),
        "package.json" => parse_package_json_dependencies(content),
        "requirements.txt" => parse_requirements_dependencies(content),
        "go.mod" => parse_go_mod_dependencies(content),
        "pyproject.toml" => parse_pyproject_dependencies(content),
        _ => Vec::new(),
    }
}

fn parse_cargo_dependencies(content: &str) -> Vec<String> {
    let mut in_deps = false;
    let mut deps = BTreeSet::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_deps = matches!(
                trimmed,
                "[dependencies]" | "[dev-dependencies]" | "[workspace.dependencies]"
            );
            continue;
        }
        if !in_deps || trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let name = trimmed
            .split('=')
            .next()
            .unwrap_or_default()
            .split_whitespace()
            .next()
            .unwrap_or_default();
        if !name.is_empty() {
            deps.insert(name.to_string());
        }
    }
    deps.into_iter().collect()
}

fn parse_package_json_dependencies(content: &str) -> Vec<String> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(content) else {
        return Vec::new();
    };
    let mut deps = BTreeSet::new();
    for section in ["dependencies", "devDependencies", "peerDependencies"] {
        if let Some(map) = value.get(section).and_then(|v| v.as_object()) {
            for key in map.keys() {
                deps.insert(key.to_string());
            }
        }
    }
    deps.into_iter().collect()
}

fn parse_requirements_dependencies(content: &str) -> Vec<String> {
    let mut deps = BTreeSet::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let name = trimmed
            .split(|ch: char| ['=', '<', '>', ' ', '\t'].contains(&ch))
            .next()
            .unwrap_or_default()
            .trim();
        if !name.is_empty() {
            deps.insert(name.to_string());
        }
    }
    deps.into_iter().collect()
}

fn parse_go_mod_dependencies(content: &str) -> Vec<String> {
    let mut deps = BTreeSet::new();
    let mut in_require_block = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("require (") {
            in_require_block = true;
            continue;
        }
        if in_require_block && trimmed == ")" {
            in_require_block = false;
            continue;
        }
        if trimmed.starts_with("require ") {
            let name = trimmed
                .trim_start_matches("require")
                .split_whitespace()
                .next()
                .unwrap_or_default();
            if !name.is_empty() {
                deps.insert(name.to_string());
            }
            continue;
        }
        if in_require_block {
            let name = trimmed.split_whitespace().next().unwrap_or_default();
            if !name.is_empty() {
                deps.insert(name.to_string());
            }
        }
    }
    deps.into_iter().collect()
}

fn parse_pyproject_dependencies(content: &str) -> Vec<String> {
    let mut deps = BTreeSet::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if !trimmed.contains('"') {
            continue;
        }
        if !(trimmed.contains("dependencies") || trimmed.starts_with('"')) {
            continue;
        }
        for chunk in trimmed.split('"').skip(1).step_by(2) {
            let dep = chunk
                .split(['=', '<', '>', ' '])
                .next()
                .unwrap_or_default()
                .trim();
            if dep.is_empty() || dep.eq_ignore_ascii_case("dependencies") {
                continue;
            }
            deps.insert(dep.to_string());
        }
    }
    deps.into_iter().collect()
}

fn summarize_test_surface(workspace: &Path, max_findings: usize) -> Vec<String> {
    let mut files = collect_text_files(workspace, 3000)
        .into_iter()
        .filter(|path| {
            let lower = path.to_ascii_lowercase();
            lower.contains("/test")
                || lower.contains("/tests/")
                || lower.ends_with("_test.rs")
                || lower.ends_with(".test.ts")
                || lower.ends_with(".test.js")
                || lower.ends_with(".spec.ts")
                || lower.ends_with(".spec.js")
        })
        .collect::<Vec<_>>();
    files.sort();

    let mut out = Vec::new();
    out.push(format!(
        "test_files_detected={} (bounded scan)",
        files.len()
    ));
    if !files.is_empty() {
        let sample = files
            .into_iter()
            .take(max_findings.saturating_sub(1).max(1))
            .collect::<Vec<_>>()
            .join(", ");
        out.push(format!("sample: {sample}"));
    }

    let package_json = workspace.join("package.json");
    if package_json.exists()
        && let Ok(raw) = fs::read_to_string(package_json)
        && let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw)
        && let Some(test_script) = value
            .get("scripts")
            .and_then(|v| v.as_object())
            .and_then(|map| map.get("test"))
            .and_then(|v| v.as_str())
    {
        out.push(format!("package.json scripts.test: {test_script}"));
    }

    out.into_iter().take(max_findings).collect()
}

fn summarize_risk_hotspots(workspace: &Path, max_findings: usize) -> Vec<String> {
    let mut out = Vec::new();

    let changed = changed_files(workspace);
    if !changed.is_empty() {
        let preview = changed
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        out.push(format!("changed files: {preview}"));
    }

    let churn = git_churn(workspace, 120);
    if !churn.is_empty() {
        let preview = churn
            .into_iter()
            .take(5)
            .map(|(path, count)| format!("{path}({count})"))
            .collect::<Vec<_>>()
            .join(", ");
        out.push(format!("high churn (git log): {preview}"));
    }

    let mut sized = collect_text_files(workspace, 4000)
        .into_iter()
        .filter_map(|path| {
            workspace
                .join(&path)
                .metadata()
                .ok()
                .map(|meta| (path, meta.len()))
        })
        .collect::<Vec<_>>();
    sized.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if !sized.is_empty() {
        let preview = sized
            .into_iter()
            .take(4)
            .map(|(path, size)| format!("{}({}KB)", path, size / 1024))
            .collect::<Vec<_>>()
            .join(", ");
        out.push(format!("large files: {preview}"));
    }

    let critical = collect_text_files(workspace, 4000)
        .into_iter()
        .filter(|path| {
            let lower = path.to_ascii_lowercase();
            lower.contains("auth")
                || lower.contains("security")
                || lower.contains("payment")
                || lower.contains("migration")
                || lower.contains("workflow")
                || lower.contains("deploy")
        })
        .take(6)
        .collect::<Vec<_>>();
    if !critical.is_empty() {
        out.push(format!("critical-path candidates: {}", critical.join(", ")));
    }

    out.into_iter().take(max_findings).collect()
}

fn collect_text_files(workspace: &Path, max_files: usize) -> Vec<String> {
    let mut out = Vec::new();
    collect_text_files_recursive(workspace, workspace, &mut out, max_files.max(1));
    out.sort();
    out
}

fn collect_text_files_recursive(
    root: &Path,
    current: &Path,
    out: &mut Vec<String>,
    max_files: usize,
) {
    if out.len() >= max_files {
        return;
    }
    let Ok(entries) = fs::read_dir(current) else {
        return;
    };
    let mut entries = entries.filter_map(|entry| entry.ok()).collect::<Vec<_>>();
    entries.sort_by_key(|a| a.file_name());

    for entry in entries {
        if out.len() >= max_files {
            break;
        }
        let path = entry.path();
        let file_name = entry.file_name().to_string_lossy().to_string();

        if path.is_dir() {
            if HEAVY_DIRS.contains(&file_name.as_str()) {
                continue;
            }
            collect_text_files_recursive(root, &path, out, max_files);
            continue;
        }

        let rel = match path.strip_prefix(root) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let rel_str = rel.to_string_lossy().to_string();
        if !is_likely_text_file(&rel_str) {
            continue;
        }
        out.push(rel_str);
    }
}

fn is_likely_text_file(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    let extension = lower.rsplit('.').next().unwrap_or_default();
    matches!(
        extension,
        "rs" | "toml"
            | "md"
            | "txt"
            | "json"
            | "yaml"
            | "yml"
            | "js"
            | "ts"
            | "tsx"
            | "jsx"
            | "py"
            | "go"
            | "java"
            | "kt"
            | "c"
            | "cc"
            | "cpp"
            | "h"
            | "hpp"
            | "sh"
            | "sql"
    )
}

fn git_status_snapshot(workspace: &Path) -> String {
    let Ok(output) = Command::new("git")
        .arg("-C")
        .arg(workspace)
        .args(["status", "--porcelain", "-b"])
        .output()
    else {
        return "not a git workspace or git unavailable".to_string();
    };

    if !output.status.success() {
        return "not a git workspace or git unavailable".to_string();
    }

    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.is_empty() {
        "clean".to_string()
    } else {
        text
    }
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
    out
}

fn git_churn(workspace: &Path, max_commits: usize) -> Vec<(String, usize)> {
    let Ok(output) = Command::new("git")
        .arg("-C")
        .arg(workspace)
        .args([
            "log",
            "--name-only",
            "--pretty=format:",
            "-n",
            &max_commits.max(1).to_string(),
        ])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        *counts.entry(trimmed.to_string()).or_insert(0) += 1;
    }

    let mut pairs = counts.into_iter().collect::<Vec<_>>();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pairs
}

fn append_indented_lines(out: &mut Vec<String>, block: &str, prefix: &str, max_lines: usize) {
    let limit = max_lines.max(1);
    for line in block.lines().take(limit) {
        out.push(format!("{prefix}{}", line.trim_end()));
    }
}

fn estimate_tokens(text: &str) -> usize {
    text.len() / 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repoish_detection_covers_keywords() {
        // Keyword-based detection: needs an inspect verb + repo subject.
        assert!(is_repoish_prompt("Analyze this project"));
        assert!(is_repoish_prompt("review the codebase for issues"));
        assert!(is_repoish_prompt("audit this repository"));
        // Plain prompts without repo subjects should NOT match.
        assert!(!is_repoish_prompt("hello"));
        assert!(!is_repoish_prompt("What is 2+2?"));
    }

    #[test]
    fn vague_codebase_detection_matches_expected_phrases() {
        assert!(is_vague_codebase_prompt("check the codebase"));
        assert!(is_vague_codebase_prompt("analyze this project for risks"));
        assert!(!is_vague_codebase_prompt("fix failing test in src/lib.rs"));
    }
}
