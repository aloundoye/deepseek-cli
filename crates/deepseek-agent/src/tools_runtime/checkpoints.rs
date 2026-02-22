use crate::*;

impl AgentEngine {
    pub fn restore_checkpoint(&self, checkpoint_id: &str) -> Result<Vec<String>> {
        let snapshot_dir = deepseek_core::runtime_dir(&self.workspace)
            .join("checkpoints")
            .join(checkpoint_id);
        if !snapshot_dir.exists() {
            return Err(anyhow!("checkpoint not found: {checkpoint_id}"));
        }

        let mut restored = Vec::new();
        for entry in std::fs::read_dir(&snapshot_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("path") {
                continue; // Skip mapping files
            }
            // Read the mapping file to find original path
            let mapping_path = path.with_extension("path");
            if let Ok(original) = std::fs::read_to_string(&mapping_path) {
                let original_path = PathBuf::from(original.trim());
                if std::fs::copy(&path, &original_path).is_ok() {
                    restored.push(original_path.to_string_lossy().to_string());
                }
            }
        }

        Ok(restored)
    }

    /// Conditionally create a checkpoint if the tool modifies files.
    pub(crate) fn emit_patch_events_if_any(
        &self,
        session_id: Uuid,
        call_name: &str,
        result: &deepseek_core::ToolResult,
    ) -> Result<()> {
        if call_name == "patch.stage"
            && result.success
            && let Some(id) = result.output.get("patch_id").and_then(|v| v.as_str())
        {
            let patch_id = Uuid::parse_str(id)?;
            let base_sha256 = result
                .output
                .get("base_sha256")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            self.emit(
                session_id,
                EventKind::PatchStagedV1 {
                    patch_id,
                    base_sha256,
                },
            )?;
        }
        if call_name == "patch.apply" && result.success {
            let patch_id = result
                .output
                .get("patch_id")
                .and_then(|v| v.as_str())
                .and_then(|s| Uuid::parse_str(s).ok())
                .unwrap_or_else(Uuid::now_v7);
            let applied = result
                .output
                .get("applied")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let conflicts = result
                .output
                .get("conflicts")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(ToString::to_string))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            self.emit(
                session_id,
                EventKind::PatchAppliedV1 {
                    patch_id,
                    applied,
                    conflicts,
                },
            )?;
        }
        Ok(())
    }

    /// Use the LLM to produce a high-quality compaction summary.
    /// Falls back to `summarize_chat_messages()` (truncation) on any failure.
    pub(crate) fn is_file_modifying_tool(name: &str) -> bool {
        matches!(
            name,
            "fs.write" | "fs.edit" | "multi_edit" | "patch.apply" | "notebook.edit"
        )
    }

    /// Create a checkpoint before file-modifying tools, snapshotting affected files.
    pub(crate) fn maybe_checkpoint(
        &self,
        session_id: Uuid,
        tool_name: &str,
        args: &serde_json::Value,
    ) {
        if Self::is_file_modifying_tool(tool_name) {
            self.create_checkpoint_for_tool(session_id, tool_name, args);
        }
    }

    pub(crate) fn create_checkpoint_for_tool(
        &self,
        session_id: Uuid,
        tool_name: &str,
        args: &serde_json::Value,
    ) {
        let mut files_to_snapshot = Vec::new();

        // Collect file paths from tool arguments
        if let Some(path) = args
            .get("file_path")
            .or_else(|| args.get("path"))
            .and_then(|v| v.as_str())
        {
            files_to_snapshot.push(PathBuf::from(path));
        }
        // multi_edit has an "edits" array with file_path entries
        if let Some(edits) = args.get("edits").and_then(|v| v.as_array()) {
            for edit in edits {
                if let Some(path) = edit.get("file_path").and_then(|v| v.as_str()) {
                    files_to_snapshot.push(PathBuf::from(path));
                }
            }
        }

        if files_to_snapshot.is_empty() {
            return;
        }

        // Create snapshot directory
        let checkpoint_id = Uuid::now_v7();
        let snapshot_dir = deepseek_core::runtime_dir(&self.workspace)
            .join("checkpoints")
            .join(checkpoint_id.to_string());
        if std::fs::create_dir_all(&snapshot_dir).is_err() {
            return;
        }

        let mut files_saved = 0u64;
        for file_path in &files_to_snapshot {
            let abs_path = if file_path.is_absolute() {
                file_path.clone()
            } else {
                self.workspace.join(file_path)
            };
            if !abs_path.exists() {
                continue; // New file — nothing to snapshot
            }
            // Save using SHA of path as filename to avoid collisions
            let hash = format!(
                "{:x}",
                sha2::Sha256::digest(abs_path.to_string_lossy().as_bytes())
            );
            let dest = snapshot_dir.join(&hash);
            if std::fs::copy(&abs_path, &dest).is_ok() {
                // Also write a mapping file
                if let Err(e) = std::fs::write(
                    snapshot_dir.join(format!("{hash}.path")),
                    abs_path.to_string_lossy().as_bytes(),
                ) {
                    self.observer
                        .warn_log(&format!("checkpoint: failed to write path mapping: {e}"));
                }
                files_saved += 1;
            }
        }

        if files_saved > 0
            && let Err(e) = self.emit(
                session_id,
                EventKind::CheckpointCreatedV1 {
                    checkpoint_id,
                    reason: format!("pre-{tool_name}"),
                    files_count: files_saved,
                    snapshot_path: snapshot_dir.to_string_lossy().to_string(),
                },
            )
        {
            self.observer
                .warn_log(&format!("checkpoint: failed to emit event: {e}"));
        }
    }
}
