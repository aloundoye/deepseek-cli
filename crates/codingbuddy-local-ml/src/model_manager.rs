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
        // Check if the model directory exists and has files
        let model_path = self.cache_dir.join(model_id);
        if let Some(status) = self.statuses.get(model_id) {
            return status.clone();
        }
        if model_path.exists() && model_path.is_dir() {
            // Check if it has at least a config file
            if model_path.join("config.json").exists() {
                return ModelStatus::Ready;
            }
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
    pub fn ensure_model(&mut self, model_id: &str) -> anyhow::Result<PathBuf> {
        let model_path = self.cache_dir.join(model_id);
        match self.status(model_id) {
            ModelStatus::Ready => Ok(model_path),
            ModelStatus::NotDownloaded => {
                #[cfg(feature = "local-ml")]
                {
                    self.download_model(model_id)?;
                    Ok(model_path)
                }
                #[cfg(not(feature = "local-ml"))]
                {
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
    fn download_model(&mut self, model_id: &str) -> anyhow::Result<()> {
        use std::fs;

        self.statuses
            .insert(model_id.to_string(), ModelStatus::Downloading);

        let model_path = self.cache_dir.join(model_id);
        fs::create_dir_all(&model_path)?;

        // Use hf-hub to download the model
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(model_id.to_string());

        // Download essential files
        for filename in &["config.json", "tokenizer.json", "model.safetensors"] {
            match repo.get(filename) {
                Ok(path) => {
                    let dest = model_path.join(filename);
                    if !dest.exists() {
                        fs::copy(path, dest)?;
                    }
                }
                Err(_) => {
                    // Some files are optional (e.g., model might be split)
                    continue;
                }
            }
        }

        self.statuses
            .insert(model_id.to_string(), ModelStatus::Ready);
        Ok(())
    }

    /// Like `ensure_model`, but calls `progress_cb(file_index, total_files)` for each file.
    pub fn ensure_model_with_progress<F: Fn(usize, usize)>(
        &mut self,
        model_id: &str,
        progress_cb: F,
    ) -> anyhow::Result<PathBuf> {
        let model_path = self.cache_dir.join(model_id);
        match self.status(model_id) {
            ModelStatus::Ready => {
                progress_cb(1, 1);
                Ok(model_path)
            }
            ModelStatus::NotDownloaded => {
                #[cfg(feature = "local-ml")]
                {
                    self.download_model_with_progress(model_id, progress_cb)?;
                    Ok(model_path)
                }
                #[cfg(not(feature = "local-ml"))]
                {
                    let _ = progress_cb;
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
    fn download_model_with_progress<F: Fn(usize, usize)>(
        &mut self,
        model_id: &str,
        progress_cb: F,
    ) -> anyhow::Result<()> {
        use std::fs;

        self.statuses
            .insert(model_id.to_string(), ModelStatus::Downloading);

        let model_path = self.cache_dir.join(model_id);
        fs::create_dir_all(&model_path)?;

        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(model_id.to_string());

        let files = ["config.json", "tokenizer.json", "model.safetensors"];
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

        // Create the model directory to simulate a ready model
        fs::create_dir_all(dir.path().join("test-model")).unwrap();
        fs::write(dir.path().join("test-model/config.json"), "{}").unwrap();
        assert_eq!(mgr.status("test-model"), ModelStatus::Ready);
    }

    #[test]
    fn ensure_model_ready_returns_path() {
        let dir = TempDir::new().unwrap();
        let mut mgr = ModelManager::new(dir.path().to_path_buf());

        // Pre-create model dir
        let model_dir = dir.path().join("my-model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), "{}").unwrap();

        let path = mgr.ensure_model("my-model").unwrap();
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

        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let called_clone = called.clone();
        let path = mgr
            .ensure_model_with_progress("ready-model", move |_current, _total| {
                called_clone.store(true, std::sync::atomic::Ordering::Relaxed);
            })
            .unwrap();
        assert_eq!(path, model_dir);
        assert!(called.load(std::sync::atomic::Ordering::Relaxed));
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
