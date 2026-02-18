use anyhow::{Result, anyhow};
use chrono::Utc;
use deepseek_core::runtime_dir;
use deepseek_store::{CheckpointRecord, Store, TranscriptExportRecord};
use serde::{Deserialize, Serialize};
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
        Ok(chunks.join("\n\n"))
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
}
