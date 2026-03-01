use std::collections::BTreeMap;
use std::path::PathBuf;

/// Download/readiness status of a local model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelStatus {
    NotDownloaded,
    Downloading,
    Ready,
    Error(String),
}

/// Metadata about a locally managed model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_id: String,
    pub status: ModelStatus,
    pub cache_path: PathBuf,
}

/// Manages local model downloads and cache.
///
/// Models are stored under `cache_dir/<model_id>/`. The manager tracks status
/// per model and (when the `local-ml` feature is enabled) can download models
/// from Hugging Face Hub.
pub struct ModelManager {
    cache_dir: PathBuf,
    statuses: BTreeMap<String, ModelStatus>,
}

impl ModelManager {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            statuses: BTreeMap::new(),
        }
    }

    /// Get the current status of a model. Returns `NotDownloaded` if unknown.
    pub fn status(&self, model_id: &str) -> ModelStatus {
        if let Some(status) = self.statuses.get(model_id) {
            return status.clone();
        }
        let model_path = self.cache_dir.join(model_id);
        if model_path.is_dir() && has_model_files(&model_path) {
            return ModelStatus::Ready;
        }
        ModelStatus::NotDownloaded
    }

    /// List all known models and their statuses.
    pub fn list_models(&self) -> Vec<ModelInfo> {
        let mut models = Vec::new();
        // Check cached statuses
        for (model_id, status) in &self.statuses {
            models.push(ModelInfo {
                model_id: model_id.clone(),
                status: status.clone(),
                cache_path: self.cache_dir.join(model_id),
            });
        }
        // Scan cache directory for models not in the status map
        if let Ok(entries) = std::fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if !self.statuses.contains_key(&name) && entry.path().is_dir() {
                    let status = self.status(&name);
                    models.push(ModelInfo {
                        model_id: name,
                        status,
                        cache_path: entry.path(),
                    });
                }
            }
        }
        models
    }

    /// Ensure a model is downloaded and ready. Without `local-ml` feature,
    /// this only checks if the model directory exists.
    ///
    /// - `model_id`: internal identifier (used as cache directory name)
    /// - `hf_repo`: HuggingFace repository to download from
    /// - `files`: list of filenames to download from the repo
    pub fn ensure_model(
        &mut self,
        model_id: &str,
        hf_repo: &str,
        files: &[&str],
    ) -> anyhow::Result<PathBuf> {
        self.ensure_model_with_progress(model_id, hf_repo, files, |_, _| {})
    }

    /// Like `ensure_model`, but calls `progress_cb(file_index, total_files)` for each file.
    ///
    /// The callback fires with `(0, total)` before downloading the first file,
    /// `(1, total)` after the first file, etc. When the model is already ready,
    /// a single `(total, total)` is emitted.
    pub fn ensure_model_with_progress<F: Fn(usize, usize)>(
        &mut self,
        model_id: &str,
        hf_repo: &str,
        files: &[&str],
        progress_cb: F,
    ) -> anyhow::Result<PathBuf> {
        let model_path = self.cache_dir.join(model_id);
        match self.status(model_id) {
            ModelStatus::Ready => {
                progress_cb(files.len(), files.len());
                Ok(model_path)
            }
            ModelStatus::NotDownloaded => {
                #[cfg(feature = "local-ml")]
                {
                    self.download_model(model_id, hf_repo, files, &progress_cb)?;
                    Ok(model_path)
                }
                #[cfg(not(feature = "local-ml"))]
                {
                    let _ = (hf_repo, files, progress_cb);
                    anyhow::bail!(
                        "model '{}' not found and local-ml feature not enabled for download",
                        model_id
                    )
                }
            }
            ModelStatus::Downloading => {
                anyhow::bail!("model '{}' is currently downloading", model_id)
            }
            ModelStatus::Error(e) => {
                anyhow::bail!("model '{}' has error: {}", model_id, e)
            }
        }
    }

    #[cfg(feature = "local-ml")]
    fn download_model(
        &mut self,
        model_id: &str,
        hf_repo: &str,
        files: &[&str],
        progress_cb: &dyn Fn(usize, usize),
    ) -> anyhow::Result<()> {
        use std::fs;

        self.statuses
            .insert(model_id.to_string(), ModelStatus::Downloading);

        let model_path = self.cache_dir.join(model_id);
        fs::create_dir_all(&model_path)?;

        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(hf_repo.to_string());

        let total = files.len();
        for (i, filename) in files.iter().enumerate() {
            progress_cb(i, total);
            match repo.get(filename) {
                Ok(path) => {
                    let dest = model_path.join(filename);
                    if !dest.exists() {
                        fs::copy(path, dest)?;
                    }
                }
                Err(_) => continue,
            }
        }
        progress_cb(total, total);

        self.statuses
            .insert(model_id.to_string(), ModelStatus::Ready);
        Ok(())
    }
}

/// Check if a model directory contains weight files (config.json for SafeTensors
/// models, or any .gguf file for quantized models).
fn has_model_files(dir: &std::path::Path) -> bool {
    if dir.join("config.json").exists() {
        return true;
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry
                .path()
                .extension()
                .is_some_and(|ext| ext == "gguf")
            {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn model_manager_status_lifecycle() {
        let dir = TempDir::new().unwrap();
        let mgr = ModelManager::new(dir.path().to_path_buf());

        // Initially not downloaded
        assert_eq!(mgr.status("test-model"), ModelStatus::NotDownloaded);

        // Create the model directory with config.json → Ready (safetensors model)
        fs::create_dir_all(dir.path().join("test-model")).unwrap();
        fs::write(dir.path().join("test-model/config.json"), "{}").unwrap();
        assert_eq!(mgr.status("test-model"), ModelStatus::Ready);
    }

    #[test]
    fn status_detects_gguf_as_ready() {
        let dir = TempDir::new().unwrap();
        let mgr = ModelManager::new(dir.path().to_path_buf());

        // Create model dir with a .gguf file (no config.json)
        let model_dir = dir.path().join("gguf-model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model.Q4_K_M.gguf"), "fake-weights").unwrap();
        assert_eq!(mgr.status("gguf-model"), ModelStatus::Ready);
    }

    #[test]
    fn ensure_model_ready_returns_path() {
        let dir = TempDir::new().unwrap();
        let mut mgr = ModelManager::new(dir.path().to_path_buf());

        // Pre-create model dir
        let model_dir = dir.path().join("my-model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), "{}").unwrap();

        let path = mgr
            .ensure_model("my-model", "org/my-model", &["config.json"])
            .unwrap();
        assert_eq!(path, model_dir);
    }

    #[test]
    fn ensure_model_with_progress_calls_callback() {
        let dir = TempDir::new().unwrap();
        let mut mgr = ModelManager::new(dir.path().to_path_buf());

        // Pre-create model dir (already ready)
        let model_dir = dir.path().join("ready-model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), "{}").unwrap();

        let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let calls_clone = calls.clone();
        let files = &["config.json", "tokenizer.json", "model.safetensors"];
        let path = mgr
            .ensure_model_with_progress(
                "ready-model",
                "org/ready-model",
                files,
                move |current, total| {
                    calls_clone.lock().unwrap().push((current, total));
                },
            )
            .unwrap();
        assert_eq!(path, model_dir);
        let recorded = calls.lock().unwrap();
        // Already-ready model emits a single (total, total) completion signal
        assert_eq!(*recorded, vec![(3, 3)]);
    }

    #[test]
    fn list_models_shows_ready_and_missing() {
        let dir = TempDir::new().unwrap();
        let mgr = ModelManager::new(dir.path().to_path_buf());

        // Create one model dir
        fs::create_dir_all(dir.path().join("model-a")).unwrap();
        fs::write(dir.path().join("model-a/config.json"), "{}").unwrap();

        let models = mgr.list_models();
        assert!(
            models.iter().any(|info| info.model_id == "model-a"),
            "should list ready model"
        );
    }
}
