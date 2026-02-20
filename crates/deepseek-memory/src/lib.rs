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
        if let Some(global) = Self::global_memory_path()
            && global.exists()
            && let Ok(text) = fs::read_to_string(&global)
        {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                chunks.push(format!("[global:{}]\n{}", global.display(), trimmed));
            }
        }
        let project = self.read_memory()?;
        let project_trimmed = project.trim();
        if !project_trimmed.is_empty() {
            chunks.push(format!(
                "[project:{}]\n{}",
                self.memory_path().display(),
                project_trimmed
            ));
        }
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
}
