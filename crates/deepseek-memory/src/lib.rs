use anyhow::{Result, anyhow};
use chrono::Utc;
use deepseek_core::runtime_dir;
use deepseek_store::{CheckpointRecord, Store, TranscriptExportRecord};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Markdown,
}

impl ExportFormat {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "json" => Some(Self::Json),
            "md" | "markdown" => Some(Self::Markdown),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Markdown => "md",
        }
    }
}

/// A git-backed shadow commit for lightweight checkpointing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowCommit {
    pub id: Uuid,
    pub ref_name: String,
    pub reason: String,
    pub created_at: String,
    pub git_backed: bool,
}

pub struct MemoryManager {
    workspace: PathBuf,
    store: Store,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoMemoryObservation {
    pub objective: String,
    pub summary: String,
    pub success: bool,
    #[serde(default)]
    pub patterns: Vec<String>,
    #[serde(default)]
    pub recorded_at: Option<String>,
}

impl MemoryManager {
    pub fn new(workspace: &Path) -> Result<Self> {
        Ok(Self {
            workspace: workspace.to_path_buf(),
            store: Store::new(workspace)?,
        })
    }

    pub fn memory_path(&self) -> PathBuf {
        self.workspace.join("DEEPSEEK.md")
    }

    pub fn global_memory_path() -> Option<PathBuf> {
        let home = std::env::var("HOME")
            .ok()
            .or_else(|| std::env::var("USERPROFILE").ok())?;
        Some(PathBuf::from(home).join(".deepseek/DEEPSEEK.md"))
    }

    pub fn ensure_initialized(&self) -> Result<PathBuf> {
        let path = self.memory_path();
        if path.exists() {
            return Ok(path);
        }
        let template = "# DEEPSEEK.md\n\nProject memory for DeepSeek CLI.\n\n## Conventions\n- Keep patches minimal and reviewable.\n- Run tests/lint before finalizing.\n";
        fs::write(&path, template)?;
        self.record_memory_version("init")?;
        Ok(path)
    }

    pub fn read_memory(&self) -> Result<String> {
        let path = self.ensure_initialized()?;
        Ok(fs::read_to_string(path)?)
    }

    pub fn read_combined_memory(&self) -> Result<String> {
        let mut chunks = Vec::new();

        // 1. Global memory: ~/.deepseek/DEEPSEEK.md
        if let Some(global) = Self::global_memory_path()
            && global.exists()
            && let Ok(text) = fs::read_to_string(&global)
        {
            let base_dir = global.parent().unwrap_or(Path::new("/"));
            let processed = process_imports(text.trim(), base_dir, 5);
            let trimmed = processed.trim();
            if !trimmed.is_empty() {
                chunks.push(format!("[global:{}]\n{}", global.display(), trimmed));
            }
        }

        // 2. Hierarchical DEEPSEEK.md files (root → cwd).
        let hierarchical = load_hierarchical_memory(&self.workspace);
        for (path, content) in &hierarchical {
            // Skip the workspace-level DEEPSEEK.md — we already load it below.
            if path == &self.memory_path() {
                continue;
            }
            let base_dir = path.parent().unwrap_or(Path::new("/"));
            let processed = process_imports(content, base_dir, 5);
            let trimmed = processed.trim();
            if !trimmed.is_empty() {
                chunks.push(format!("[memory:{}]\n{}", path.display(), trimmed));
            }
        }

        // 3. Project-level DEEPSEEK.md (workspace root).
        let project = self.read_memory()?;
        let base_dir = self
            .memory_path()
            .parent()
            .unwrap_or(Path::new("/"))
            .to_path_buf();
        let processed = process_imports(project.trim(), &base_dir, 5);
        let project_trimmed = processed.trim();
        if !project_trimmed.is_empty() {
            chunks.push(format!(
                "[project:{}]\n{}",
                self.memory_path().display(),
                project_trimmed
            ));
        }

        // 4. Project-local DEEPSEEK.local.md (gitignored).
        let local_path = self.workspace.join("DEEPSEEK.local.md");
        if local_path.exists()
            && let Ok(text) = fs::read_to_string(&local_path)
        {
            let processed = process_imports(text.trim(), &self.workspace, 5);
            let trimmed = processed.trim();
            if !trimmed.is_empty() {
                chunks.push(format!("[local:{}]\n{}", local_path.display(), trimmed));
            }
        }

        // 5. Rules directories.
        let rules = load_rules(&self.workspace);
        for rule in &rules {
            let level = if rule.user_level { "user-rule" } else { "rule" };
            chunks.push(format!(
                "[{}:{}]\n{}",
                level,
                rule.path.display(),
                rule.content
            ));
        }

        // 6. Auto-memory observations.
        if let Some(auto_path) = self.auto_memory_path()
            && auto_path.exists()
            && let Ok(text) = fs::read_to_string(&auto_path)
        {
            let snippet = tail_lines(&text, 200);
            let trimmed = snippet.trim();
            if !trimmed.is_empty() {
                chunks.push(format!("[auto:{}]\n{}", auto_path.display(), trimmed));
            }
        }
        Ok(chunks.join("\n\n"))
    }

    pub fn auto_memory_path(&self) -> Option<PathBuf> {
        let home = std::env::var("HOME")
            .ok()
            .or_else(|| std::env::var("USERPROFILE").ok())?;
        let canonical = self
            .workspace
            .canonicalize()
            .unwrap_or_else(|_| self.workspace.clone());
        let mut hasher = Sha256::new();
        hasher.update(canonical.to_string_lossy().as_bytes());
        let hash = format!("{:x}", hasher.finalize());
        Some(
            PathBuf::from(home)
                .join(".deepseek")
                .join("projects")
                .join(hash)
                .join("memory")
                .join("MEMORY.md"),
        )
    }

    pub fn append_auto_memory_observation(&self, observation: AutoMemoryObservation) -> Result<()> {
        let path = self
            .auto_memory_path()
            .ok_or_else(|| anyhow!("HOME/USERPROFILE is not set; cannot persist auto memory"))?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut existing = if path.exists() {
            fs::read_to_string(&path)?
        } else {
            format!(
                "# Auto Memory\n\nWorkspace: {}\n\n",
                self.workspace.to_string_lossy()
            )
        };

        let objective = truncate_line(observation.objective.trim(), 180);
        let summary = truncate_line(observation.summary.trim(), 220);
        let status = if observation.success {
            "success"
        } else {
            "failure"
        };
        let recorded_at = observation
            .recorded_at
            .unwrap_or_else(|| Utc::now().to_rfc3339());

        let patterns = if observation.patterns.is_empty() {
            infer_patterns(&objective, &summary, observation.success)
        } else {
            observation
                .patterns
                .into_iter()
                .map(|line| truncate_line(line.trim(), 160))
                .filter(|line| !line.is_empty())
                .take(8)
                .collect::<Vec<_>>()
        };

        existing.push_str(&format!("## {recorded_at} ({status})\n"));
        existing.push_str(&format!("- objective: {objective}\n"));
        existing.push_str(&format!("- summary: {summary}\n"));
        for pattern in patterns {
            existing.push_str(&format!("- pattern: {pattern}\n"));
        }
        existing.push('\n');

        let pruned = prune_auto_memory_lines(&existing, 260);
        fs::write(path, pruned)?;
        Ok(())
    }

    pub fn write_memory(&self, content: &str) -> Result<()> {
        let path = self.ensure_initialized()?;
        fs::write(path, content)?;
        self.record_memory_version("write")?;
        Ok(())
    }

    pub fn sync_memory_version(&self, note: &str) -> Result<Uuid> {
        self.record_memory_version(note)
    }

    fn record_memory_version(&self, note: &str) -> Result<Uuid> {
        let version_id = Uuid::now_v7();
        let path = self.memory_path();
        let content = fs::read_to_string(&path).unwrap_or_default();
        self.store.insert_memory_version(
            version_id,
            path.to_string_lossy().as_ref(),
            &content,
            note,
            &Utc::now().to_rfc3339(),
        )?;
        Ok(version_id)
    }

    pub fn create_checkpoint(&self, reason: &str) -> Result<CheckpointRecord> {
        let checkpoint_id = Uuid::now_v7();
        let checkpoint_root = runtime_dir(&self.workspace)
            .join("checkpoints")
            .join(checkpoint_id.to_string());
        let snapshot_root = checkpoint_root.join("fs");
        fs::create_dir_all(&snapshot_root)?;

        let mut files = 0_u64;
        for entry in WalkDir::new(&self.workspace)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.path().is_file())
        {
            let path = entry.path();
            let rel = path.strip_prefix(&self.workspace)?;
            if has_ignored_component(rel) {
                continue;
            }
            let dest = snapshot_root.join(rel);
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(path, dest)?;
            files += 1;
        }

        let metadata = serde_json::json!({
            "checkpoint_id": checkpoint_id,
            "reason": reason,
            "files": files,
            "created_at": Utc::now().to_rfc3339(),
        });
        fs::write(
            checkpoint_root.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )?;

        let record = CheckpointRecord {
            checkpoint_id,
            reason: reason.to_string(),
            snapshot_path: snapshot_root.to_string_lossy().to_string(),
            created_at: Utc::now().to_rfc3339(),
            files_count: files,
        };
        self.store.insert_checkpoint(&record)?;
        Ok(record)
    }

    pub fn rewind_to_checkpoint(&self, checkpoint_id: Uuid) -> Result<CheckpointRecord> {
        let record = self
            .store
            .load_checkpoint(checkpoint_id)?
            .ok_or_else(|| anyhow!("checkpoint not found: {checkpoint_id}"))?;

        let snapshot_root = PathBuf::from(&record.snapshot_path);
        if !snapshot_root.exists() {
            return Err(anyhow!(
                "checkpoint snapshot missing: {}",
                snapshot_root.display()
            ));
        }

        for entry in WalkDir::new(&self.workspace)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.path().is_file())
        {
            let path = entry.path();
            let rel = path.strip_prefix(&self.workspace)?;
            if has_ignored_component(rel) {
                continue;
            }
            fs::remove_file(path)?;
        }

        for entry in WalkDir::new(&snapshot_root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.path().is_file())
        {
            let path = entry.path();
            let rel = path.strip_prefix(&snapshot_root)?;
            let dest = self.workspace.join(rel);
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(path, dest)?;
        }

        Ok(record)
    }

    pub fn list_checkpoints(&self) -> Result<Vec<CheckpointRecord>> {
        self.store.list_checkpoints()
    }

    // ── Git Shadow Commits ───────────────────────────────────────────────────

    /// Create a shadow commit on hidden ref `refs/deepseek-shadow/<id>`.
    /// This is a lightweight alternative to full-file checkpointing that uses
    /// git's object store directly. Falls back to file-based checkpoint if
    /// not in a git repo.
    pub fn create_shadow_commit(&self, reason: &str) -> Result<ShadowCommit> {
        let commit_id = Uuid::now_v7();
        let ref_name = format!("refs/deepseek-shadow/{commit_id}");
        let workspace = &self.workspace;

        // Check if we're in a git repo
        let git_status = std::process::Command::new("git")
            .args(["rev-parse", "--git-dir"])
            .current_dir(workspace)
            .output();

        if git_status.is_err() || !git_status.as_ref().unwrap().status.success() {
            // Not a git repo — fall back to file-based checkpoint
            let record = self.create_checkpoint(reason)?;
            return Ok(ShadowCommit {
                id: commit_id,
                ref_name,
                reason: reason.to_string(),
                created_at: record.created_at,
                git_backed: false,
            });
        }

        // Stage all tracked + untracked files to a temporary index
        let stash_result = std::process::Command::new("git")
            .args(["stash", "create"])
            .current_dir(workspace)
            .output()?;

        let tree_sha = if stash_result.status.success() && !stash_result.stdout.is_empty() {
            // We have changes — use the stash tree
            let stash_sha = String::from_utf8_lossy(&stash_result.stdout).trim().to_string();
            // Get the tree from the stash commit
            let tree_out = std::process::Command::new("git")
                .args(["rev-parse", &format!("{stash_sha}^{{tree}}")])
                .current_dir(workspace)
                .output()?;
            String::from_utf8_lossy(&tree_out.stdout).trim().to_string()
        } else {
            // No changes — use HEAD tree
            let head_tree = std::process::Command::new("git")
                .args(["rev-parse", "HEAD^{tree}"])
                .current_dir(workspace)
                .output()?;
            if !head_tree.status.success() {
                return Err(anyhow!("no git HEAD found for shadow commit"));
            }
            String::from_utf8_lossy(&head_tree.stdout).trim().to_string()
        };

        // Create a commit object from the tree
        let commit_msg = format!("deepseek shadow: {reason} [{commit_id}]");
        let commit_out = std::process::Command::new("git")
            .args(["commit-tree", &tree_sha, "-m", &commit_msg])
            .current_dir(workspace)
            .output()?;

        if !commit_out.status.success() {
            return Err(anyhow!(
                "git commit-tree failed: {}",
                String::from_utf8_lossy(&commit_out.stderr)
            ));
        }

        let commit_sha = String::from_utf8_lossy(&commit_out.stdout).trim().to_string();

        // Create the hidden ref
        let ref_result = std::process::Command::new("git")
            .args(["update-ref", &ref_name, &commit_sha])
            .current_dir(workspace)
            .output()?;

        if !ref_result.status.success() {
            return Err(anyhow!(
                "git update-ref failed: {}",
                String::from_utf8_lossy(&ref_result.stderr)
            ));
        }

        let now = Utc::now().to_rfc3339();
        Ok(ShadowCommit {
            id: commit_id,
            ref_name,
            reason: reason.to_string(),
            created_at: now,
            git_backed: true,
        })
    }

    /// Restore workspace files from a shadow commit ref.
    pub fn restore_shadow_commit(&self, shadow_id: Uuid) -> Result<()> {
        let ref_name = format!("refs/deepseek-shadow/{shadow_id}");
        let workspace = &self.workspace;

        // Check if the ref exists
        let verify = std::process::Command::new("git")
            .args(["rev-parse", "--verify", &ref_name])
            .current_dir(workspace)
            .output()?;

        if !verify.status.success() {
            // Try file-based checkpoint fallback
            self.rewind_to_checkpoint(shadow_id)?;
            return Ok(());
        }

        let commit_sha = String::from_utf8_lossy(&verify.stdout).trim().to_string();

        // Restore files: checkout tree from shadow commit
        let checkout = std::process::Command::new("git")
            .args(["checkout", &commit_sha, "--", "."])
            .current_dir(workspace)
            .output()?;

        if !checkout.status.success() {
            return Err(anyhow!(
                "git checkout from shadow commit failed: {}",
                String::from_utf8_lossy(&checkout.stderr)
            ));
        }

        // Unstage the checkout so working tree is clean but not committed
        let _ = std::process::Command::new("git")
            .args(["reset", "HEAD", "."])
            .current_dir(workspace)
            .output();

        Ok(())
    }

    /// List all shadow commit refs.
    pub fn list_shadow_commits(&self) -> Result<Vec<ShadowCommit>> {
        let workspace = &self.workspace;
        let output = std::process::Command::new("git")
            .args([
                "for-each-ref",
                "--format=%(refname)\t%(subject)\t%(creatordate:iso-strict)",
                "refs/deepseek-shadow/",
            ])
            .current_dir(workspace)
            .output()?;

        if !output.status.success() {
            return Ok(Vec::new());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut commits = Vec::new();
        for line in stdout.lines() {
            let parts: Vec<&str> = line.splitn(3, '\t').collect();
            if parts.len() < 2 {
                continue;
            }
            let ref_name = parts[0].to_string();
            let subject = parts[1].to_string();
            let created_at = parts.get(2).unwrap_or(&"").to_string();

            // Extract UUID from ref name
            let id_str = ref_name.strip_prefix("refs/deepseek-shadow/").unwrap_or("");
            let id = Uuid::parse_str(id_str).unwrap_or(Uuid::nil());

            // Extract reason from subject: "deepseek shadow: <reason> [<id>]"
            let reason = subject
                .strip_prefix("deepseek shadow: ")
                .and_then(|s| s.rsplit_once(" [").map(|(r, _)| r.to_string()))
                .unwrap_or(subject);

            commits.push(ShadowCommit {
                id,
                ref_name,
                reason,
                created_at,
                git_backed: true,
            });
        }

        Ok(commits)
    }

    pub fn export_transcript(
        &self,
        format: ExportFormat,
        output: Option<&Path>,
        session_id: Option<Uuid>,
    ) -> Result<TranscriptExportRecord> {
        let session = if let Some(session_id) = session_id {
            session_id
        } else {
            self.store
                .load_latest_session()?
                .ok_or_else(|| anyhow!("no session found"))?
                .session_id
        };
        let projection = self.store.rebuild_from_events(session)?;
        let export_id = Uuid::now_v7();
        let out_path = output.map(Path::to_path_buf).unwrap_or_else(|| {
            runtime_dir(&self.workspace)
                .join("exports")
                .join(format!("{export_id}.{}", format.as_str()))
        });
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let body = match format {
            ExportFormat::Json => serde_json::to_string_pretty(&projection.transcript)?,
            ExportFormat::Markdown => {
                let mut md = String::from("# DeepSeek Transcript Export\n\n");
                for line in &projection.transcript {
                    md.push_str("- ");
                    md.push_str(line);
                    md.push('\n');
                }
                md
            }
        };
        fs::write(&out_path, body)?;

        let record = TranscriptExportRecord {
            export_id,
            session_id: session,
            format: format.as_str().to_string(),
            output_path: out_path.to_string_lossy().to_string(),
            created_at: Utc::now().to_rfc3339(),
        };
        self.store.insert_transcript_export(&record)?;
        Ok(record)
    }
}

// ── Rules Directory Loading ──────────────────────────────────────────────────

/// A loaded rule file from `.deepseek/rules/` or `~/.deepseek/rules/`.
#[derive(Debug, Clone)]
pub struct RuleFile {
    /// Source path of the rule file.
    pub path: PathBuf,
    /// Whether this is a user-level rule (from ~/.deepseek/rules/).
    pub user_level: bool,
    /// Glob patterns for path-specific activation. Empty = always active.
    pub path_patterns: Vec<String>,
    /// The rule content (markdown body after frontmatter).
    pub content: String,
}

/// Load all rule files from both project and user-level rules directories.
/// Returns rules in load order: user-level first, then project-level.
pub fn load_rules(workspace: &Path) -> Vec<RuleFile> {
    let mut rules = Vec::new();

    // User-level rules: ~/.deepseek/rules/*.md
    if let Some(home) = std::env::var("HOME")
        .ok()
        .or_else(|| std::env::var("USERPROFILE").ok())
    {
        let user_rules_dir = PathBuf::from(&home).join(".deepseek/rules");
        if user_rules_dir.is_dir() {
            load_rules_from_dir(&user_rules_dir, true, &mut rules);
        }
    }

    // Project-level rules: .deepseek/rules/*.md
    let project_rules_dir = workspace.join(".deepseek/rules");
    if project_rules_dir.is_dir() {
        load_rules_from_dir(&project_rules_dir, false, &mut rules);
    }

    rules
}

/// Filter rules to only those that are active for the given file paths being edited.
/// Rules with no path patterns are always active.
pub fn filter_active_rules<'a>(rules: &'a [RuleFile], active_paths: &[&str]) -> Vec<&'a RuleFile> {
    rules
        .iter()
        .filter(|rule| {
            if rule.path_patterns.is_empty() {
                return true; // No patterns = always active.
            }
            // At least one pattern must match at least one active path.
            rule.path_patterns.iter().any(|pattern| {
                if let Ok(pat) = glob::Pattern::new(pattern) {
                    active_paths
                        .iter()
                        .any(|p| pat.matches(p) || pat.matches_path(Path::new(p)))
                } else {
                    false
                }
            })
        })
        .collect()
}

fn load_rules_from_dir(dir: &Path, user_level: bool, out: &mut Vec<RuleFile>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
        })
        .collect();
    paths.sort();

    for path in paths {
        if let Ok(raw) = fs::read_to_string(&path) {
            let (frontmatter, body) = parse_rule_frontmatter(&raw);
            let path_patterns = frontmatter
                .get("paths")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            let content = body.trim().to_string();
            if !content.is_empty() {
                out.push(RuleFile {
                    path,
                    user_level,
                    path_patterns,
                    content,
                });
            }
        }
    }
}

/// Parse optional YAML frontmatter from a rule file.
/// Returns (frontmatter as JSON value, body after frontmatter).
fn parse_rule_frontmatter(raw: &str) -> (serde_json::Value, &str) {
    let trimmed = raw.trim_start();
    if !trimmed.starts_with("---") {
        return (serde_json::Value::Null, raw);
    }
    // Find closing ---
    if let Some(end) = trimmed[3..].find("\n---") {
        let yaml_str = &trimmed[3..3 + end].trim();
        let body = &trimmed[3 + end + 4..]; // skip past closing ---
        // Parse YAML as JSON (simple key: value pairs).
        let frontmatter = parse_simple_yaml(yaml_str);
        (frontmatter, body)
    } else {
        (serde_json::Value::Null, raw)
    }
}

/// Minimal YAML-like parser for frontmatter (supports key: value and key: [list]).
fn parse_simple_yaml(yaml: &str) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    let mut current_key: Option<String> = None;
    let mut current_list: Vec<serde_json::Value> = Vec::new();
    let mut in_list = false;

    for line in yaml.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // List item under current key.
        if in_list && trimmed.starts_with("- ") {
            current_list.push(serde_json::Value::String(trimmed[2..].trim().to_string()));
            continue;
        }

        // Save any pending list.
        if in_list {
            if let Some(ref key) = current_key {
                map.insert(key.clone(), serde_json::Value::Array(current_list.clone()));
            }
            in_list = false;
            current_list.clear();
        }

        if let Some(colon_pos) = trimmed.find(':') {
            let key = trimmed[..colon_pos].trim().to_string();
            let value_part = trimmed[colon_pos + 1..].trim();
            if value_part.is_empty() {
                // Could be start of a list.
                current_key = Some(key);
                in_list = true;
                current_list.clear();
            } else {
                map.insert(key, serde_json::Value::String(value_part.to_string()));
            }
        }
    }

    // Save final pending list.
    if in_list && let Some(ref key) = current_key {
        map.insert(key.clone(), serde_json::Value::Array(current_list));
    }

    serde_json::Value::Object(map)
}

// ── Hierarchical DEEPSEEK.md Loading ────────────────────────────────────────

/// Load DEEPSEEK.md files hierarchically from cwd up to filesystem root.
/// Returns content chunks in order: root → ... → cwd (higher-level first).
pub fn load_hierarchical_memory(workspace: &Path) -> Vec<(PathBuf, String)> {
    let mut paths = Vec::new();
    let mut dir = workspace.to_path_buf();

    loop {
        let deepseek_md = dir.join("DEEPSEEK.md");
        if deepseek_md.exists()
            && let Ok(content) = fs::read_to_string(&deepseek_md)
        {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                paths.push((deepseek_md, trimmed.to_string()));
            }
        }
        // Also check DEEPSEEK.local.md (gitignored, machine-local).
        let local_md = dir.join("DEEPSEEK.local.md");
        if local_md.exists()
            && let Ok(content) = fs::read_to_string(&local_md)
        {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                paths.push((local_md, trimmed.to_string()));
            }
        }

        if !dir.pop() {
            break;
        }
    }

    // Reverse so root-level appears first (higher precedence loads first).
    paths.reverse();
    paths
}

/// Process @import directives in memory content.
/// `@path/to/file` includes that file's contents inline.
/// Relative paths are resolved from the directory containing the source file.
/// Max recursion depth is 5 to prevent infinite loops.
pub fn process_imports(content: &str, base_dir: &Path, max_depth: u8) -> String {
    if max_depth == 0 {
        return content.to_string();
    }

    let mut result = String::with_capacity(content.len());
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(import_path) = trimmed.strip_prefix('@') {
            let import_path = import_path.trim();
            if import_path.is_empty() {
                result.push_str(line);
                result.push('\n');
                continue;
            }
            let resolved = base_dir.join(import_path);
            if resolved.exists() && resolved.is_file() {
                if let Ok(imported) = fs::read_to_string(&resolved) {
                    let import_dir = resolved.parent().unwrap_or(base_dir);
                    let processed = process_imports(&imported, import_dir, max_depth - 1);
                    result.push_str(&format!("<!-- imported from {} -->\n", resolved.display()));
                    result.push_str(processed.trim());
                    result.push('\n');
                } else {
                    result.push_str(&format!("<!-- import failed: {} -->\n", resolved.display()));
                }
            } else {
                result.push_str(&format!(
                    "<!-- import not found: {} -->\n",
                    resolved.display()
                ));
            }
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

fn has_ignored_component(path: &Path) -> bool {
    path.components().any(|component| {
        let value = component.as_os_str();
        value == OsStr::new(".git")
            || value == OsStr::new(".deepseek")
            || value == OsStr::new("target")
    })
}

fn truncate_line(input: &str, max_chars: usize) -> String {
    let mut out = input.trim().replace('\n', " ");
    if out.chars().count() <= max_chars {
        return out;
    }
    out = out.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
}

fn infer_patterns(objective: &str, summary: &str, success: bool) -> Vec<String> {
    let mut patterns = Vec::new();
    let objective_lc = objective.to_ascii_lowercase();
    let summary_lc = summary.to_ascii_lowercase();

    if objective_lc.contains("test") || summary_lc.contains("test") {
        patterns.push("Run targeted tests immediately after each change chunk.".to_string());
    }
    if objective_lc.contains("refactor") {
        patterns.push(
            "Refactors are safer when split into behavior-preserving checkpoints.".to_string(),
        );
    }
    if summary_lc.contains("verification") || summary_lc.contains("lint") {
        patterns.push(
            "Capture verification output explicitly to avoid repeated failure loops.".to_string(),
        );
    }
    if !success {
        patterns
            .push("On failure, revise plan scope before re-running the full pipeline.".to_string());
    }
    if patterns.is_empty() {
        patterns.push(
            "Prefer minimal, reviewable patches with explicit verification evidence.".to_string(),
        );
    }
    patterns
}

fn prune_auto_memory_lines(content: &str, max_lines: usize) -> String {
    let lines = content.lines().collect::<Vec<_>>();
    if lines.len() <= max_lines {
        let mut out = content.to_string();
        if !out.ends_with('\n') {
            out.push('\n');
        }
        return out;
    }
    let mut kept = Vec::new();
    if let Some(first) = lines.first().copied() {
        kept.push(first);
    }
    if lines.len() > 1 {
        kept.push(lines[1]);
    }
    let reserve = max_lines.saturating_sub(kept.len());
    let tail_start = lines.len().saturating_sub(reserve);
    kept.extend_from_slice(&lines[tail_start..]);
    let mut out = kept.join("\n");
    out.push('\n');
    out
}

fn tail_lines(content: &str, max_lines: usize) -> String {
    let lines = content.lines().collect::<Vec<_>>();
    if lines.len() <= max_lines {
        return content.to_string();
    }
    lines[lines.len().saturating_sub(max_lines)..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initializes_and_reads_memory() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-memory-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let manager = MemoryManager::new(&workspace).expect("manager");
        let content = manager.read_memory().expect("read");
        assert!(content.contains("DEEPSEEK.md"));
    }

    #[test]
    fn checkpoint_and_rewind_restores_files() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-memory-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        fs::write(workspace.join("a.txt"), "one").expect("seed");

        let manager = MemoryManager::new(&workspace).expect("manager");
        let checkpoint = manager.create_checkpoint("test").expect("checkpoint");
        fs::write(workspace.join("a.txt"), "two").expect("mutate");

        manager
            .rewind_to_checkpoint(checkpoint.checkpoint_id)
            .expect("rewind");
        let restored = fs::read_to_string(workspace.join("a.txt")).expect("restored");
        assert_eq!(restored, "one");
    }

    #[test]
    fn auto_memory_observations_are_persisted_and_loaded() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-memory-auto-{}", Uuid::now_v7()));
        let home = std::env::temp_dir().join(format!("deepseek-memory-home-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        fs::create_dir_all(&home).expect("home");

        let previous_home = std::env::var("HOME").ok();
        // SAFETY: test-only environment mutation.
        unsafe {
            std::env::set_var("HOME", &home);
        }

        let manager = MemoryManager::new(&workspace).expect("manager");
        manager
            .append_auto_memory_observation(AutoMemoryObservation {
                objective: "Refactor parser and run tests".to_string(),
                summary: "Verification failed on lint before tests".to_string(),
                success: false,
                patterns: vec![],
                recorded_at: Some("2026-02-19T00:00:00Z".to_string()),
            })
            .expect("append");

        let auto_path = manager.auto_memory_path().expect("auto path");
        assert!(auto_path.exists());
        let auto_text = fs::read_to_string(&auto_path).expect("auto memory");
        assert!(auto_text.contains("Refactor parser"));
        assert!(auto_text.contains("pattern:"));

        let combined = manager.read_combined_memory().expect("combined");
        assert!(combined.contains("[auto:"));
        assert!(combined.contains("Refactor parser"));

        match previous_home {
            Some(value) => {
                // SAFETY: test-only environment mutation.
                unsafe {
                    std::env::set_var("HOME", value);
                }
            }
            None => {
                // SAFETY: test-only environment mutation.
                unsafe {
                    std::env::remove_var("HOME");
                }
            }
        }
    }

    #[test]
    fn loads_rules_from_project_directory() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-rules-test-{}", Uuid::now_v7()));
        let rules_dir = workspace.join(".deepseek/rules");
        fs::create_dir_all(&rules_dir).expect("rules dir");
        fs::write(
            rules_dir.join("coding.md"),
            "Always use snake_case for variables.\n",
        )
        .expect("rule");
        fs::write(
            rules_dir.join("rust.md"),
            "---\npaths:\n  - src/**/*.rs\n---\nUse anyhow for errors.\n",
        )
        .expect("rule with frontmatter");

        let rules = load_rules(&workspace);
        assert_eq!(rules.len(), 2);
        // Sorted alphabetically: coding.md, rust.md
        assert!(rules[0].content.contains("snake_case"));
        assert!(rules[0].path_patterns.is_empty());
        assert!(rules[1].content.contains("anyhow"));
        assert_eq!(rules[1].path_patterns, vec!["src/**/*.rs"]);
    }

    #[test]
    fn filter_active_rules_matches_patterns() {
        let always = RuleFile {
            path: PathBuf::from("always.md"),
            user_level: false,
            path_patterns: vec![],
            content: "Always active".to_string(),
        };
        let rust_only = RuleFile {
            path: PathBuf::from("rust.md"),
            user_level: false,
            path_patterns: vec!["src/**/*.rs".to_string()],
            content: "Rust only".to_string(),
        };
        let js_only = RuleFile {
            path: PathBuf::from("js.md"),
            user_level: false,
            path_patterns: vec!["**/*.js".to_string()],
            content: "JS only".to_string(),
        };

        let rules = vec![always, rust_only, js_only];
        let active = filter_active_rules(&rules, &["src/main.rs"]);
        assert_eq!(active.len(), 2); // always + rust_only
        assert!(active.iter().any(|r| r.content == "Always active"));
        assert!(active.iter().any(|r| r.content == "Rust only"));
    }

    #[test]
    fn process_imports_resolves_files() {
        let dir = std::env::temp_dir().join(format!("deepseek-import-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&dir).expect("dir");
        fs::write(dir.join("extra.md"), "Extra content here").expect("extra");

        let input = "# Main\n@extra.md\nDone.";
        let output = process_imports(input, &dir, 5);
        assert!(output.contains("Extra content here"));
        assert!(output.contains("# Main"));
        assert!(output.contains("Done."));
    }

    #[test]
    fn process_imports_handles_missing_file() {
        let dir = std::env::temp_dir().join(format!("deepseek-import-missing-{}", Uuid::now_v7()));
        fs::create_dir_all(&dir).expect("dir");

        let input = "# Main\n@nonexistent.md\nDone.";
        let output = process_imports(input, &dir, 5);
        assert!(output.contains("import not found"));
        assert!(output.contains("Done."));
    }

    #[test]
    fn process_imports_respects_max_depth() {
        let dir = std::env::temp_dir().join(format!("deepseek-import-depth-{}", Uuid::now_v7()));
        fs::create_dir_all(&dir).expect("dir");
        // a.md imports b.md which imports a.md (circular).
        fs::write(dir.join("a.md"), "@b.md").expect("a");
        fs::write(dir.join("b.md"), "@a.md").expect("b");

        // Should terminate without infinite loop.
        let output = process_imports("@a.md", &dir, 3);
        assert!(!output.is_empty());
    }

    #[test]
    fn hierarchical_memory_loads_upward() {
        let base = std::env::temp_dir().join(format!("deepseek-hier-{}", Uuid::now_v7()));
        let child = base.join("project/sub");
        fs::create_dir_all(&child).expect("dirs");

        fs::write(base.join("DEEPSEEK.md"), "Root memory").expect("root");
        fs::write(base.join("project/DEEPSEEK.md"), "Project memory").expect("project");
        fs::write(child.join("DEEPSEEK.md"), "Sub memory").expect("sub");

        let loaded = load_hierarchical_memory(&child);
        assert!(loaded.len() >= 3);
        // First entry should be the highest-level one.
        assert!(loaded.first().unwrap().1.contains("Root memory"));
        assert!(loaded.last().unwrap().1.contains("Sub memory"));
    }

    #[test]
    fn parse_rule_frontmatter_extracts_paths() {
        let raw = "---\npaths:\n  - src/**/*.rs\n  - tests/**/*.rs\n---\nRule body here.\n";
        let (fm, body) = parse_rule_frontmatter(raw);
        let paths = fm["paths"].as_array().expect("array");
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0].as_str().unwrap(), "src/**/*.rs");
        assert!(body.contains("Rule body here"));
    }

    #[test]
    fn parse_rule_frontmatter_handles_no_frontmatter() {
        let raw = "Just plain content.";
        let (fm, body) = parse_rule_frontmatter(raw);
        assert!(fm.is_null());
        assert_eq!(body, raw);
    }
}
