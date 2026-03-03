use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

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

/// Partial download state persisted to disk for resume support.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(feature = "local-ml")]
struct PartialDownloadState {
    model_id: String,
    hf_repo: String,
    /// SHA-256 digests of completed files (filename → digest).
    completed_digests: BTreeMap<String, String>,
}

/// Maximum number of retries per file download.
#[cfg(feature = "local-ml")]
const MAX_DOWNLOAD_RETRIES: usize = 6;

/// Manifest for a content-addressable model cache entry.
///
/// Maps logical filenames to SHA-256 digests stored in the `blobs/` directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub model_id: String,
    pub files: BTreeMap<String, String>,
}

/// Manages local model downloads and cache.
///
/// Uses a content-addressable storage scheme:
/// ```text
/// cache_dir/
///   blobs/
///     sha256-<digest>    -- immutable weight/config files
///   manifests/
///     <model_id>.json    -- maps filenames to blob digests
///   <model_id>/          -- legacy layout (migrated on first access)
/// ```
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

    fn blobs_dir(&self) -> PathBuf {
        self.cache_dir.join("blobs")
    }

    fn manifests_dir(&self) -> PathBuf {
        self.cache_dir.join("manifests")
    }

    /// Get the current status of a model. Returns `NotDownloaded` if unknown.
    pub fn status(&self, model_id: &str) -> ModelStatus {
        if let Some(status) = self.statuses.get(model_id) {
            return status.clone();
        }
        // Check manifest-based storage first
        if self.has_valid_manifest(model_id) {
            return ModelStatus::Ready;
        }
        // Fall back to legacy layout
        let model_path = self.cache_dir.join(model_id);
        if model_path.is_dir() && has_model_files(&model_path) {
            return ModelStatus::Ready;
        }
        ModelStatus::NotDownloaded
    }

    /// Check if a manifest exists and all referenced blobs are present.
    fn has_valid_manifest(&self, model_id: &str) -> bool {
        let manifest_path = self.manifests_dir().join(format!("{model_id}.json"));
        let Ok(data) = std::fs::read_to_string(&manifest_path) else {
            return false;
        };
        let Ok(manifest) = serde_json::from_str::<ModelManifest>(&data) else {
            return false;
        };
        manifest
            .files
            .values()
            .all(|digest| self.blobs_dir().join(format!("sha256-{digest}")).exists())
    }

    /// Read the manifest for a model, if it exists.
    pub fn read_manifest(&self, model_id: &str) -> Option<ModelManifest> {
        let manifest_path = self.manifests_dir().join(format!("{model_id}.json"));
        let data = std::fs::read_to_string(&manifest_path).ok()?;
        serde_json::from_str(&data).ok()
    }

    /// Verify integrity of a model by re-hashing all blobs against the manifest.
    pub fn verify_integrity(&self, model_id: &str) -> bool {
        let Some(manifest) = self.read_manifest(model_id) else {
            return false;
        };
        manifest.files.values().all(|expected_digest| {
            let blob_path = self.blobs_dir().join(format!("sha256-{expected_digest}"));
            match sha256_file(&blob_path) {
                Some(actual) => actual == *expected_digest,
                None => false,
            }
        })
    }

    /// Resolve a model's files to their actual paths.
    ///
    /// Checks manifest-based layout first, falls back to legacy `<model_id>/` dir.
    pub fn model_path(&self, model_id: &str) -> PathBuf {
        // Legacy layout path — files are accessed directly from here
        self.cache_dir.join(model_id)
    }

    /// Migrate a legacy model directory to content-addressable storage.
    ///
    /// Computes SHA-256 for each file, moves to `blobs/`, writes manifest.
    /// Returns the manifest on success.
    pub fn migrate_to_content_addressable(&self, model_id: &str) -> anyhow::Result<ModelManifest> {
        let legacy_dir = self.cache_dir.join(model_id);
        if !legacy_dir.is_dir() {
            anyhow::bail!("model '{}' directory not found", model_id);
        }

        let blobs = self.blobs_dir();
        let manifests = self.manifests_dir();
        std::fs::create_dir_all(&blobs)?;
        std::fs::create_dir_all(&manifests)?;

        let mut files = BTreeMap::new();
        for entry in std::fs::read_dir(&legacy_dir)?.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let filename = entry.file_name().to_string_lossy().to_string();
            let digest = sha256_file(&path)
                .ok_or_else(|| anyhow::anyhow!("failed to hash {}", path.display()))?;

            let blob_path = blobs.join(format!("sha256-{digest}"));
            if !blob_path.exists() {
                std::fs::copy(&path, &blob_path)?;
            }
            files.insert(filename, digest);
        }

        let manifest = ModelManifest {
            model_id: model_id.to_string(),
            files,
        };
        let manifest_path = manifests.join(format!("{model_id}.json"));
        std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;

        Ok(manifest)
    }

    /// List all known models and their statuses.
    pub fn list_models(&self) -> Vec<ModelInfo> {
        let mut models = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Check cached statuses
        for (model_id, status) in &self.statuses {
            seen.insert(model_id.clone());
            models.push(ModelInfo {
                model_id: model_id.clone(),
                status: status.clone(),
                cache_path: self.cache_dir.join(model_id),
            });
        }

        // Check manifests
        if let Ok(entries) = std::fs::read_dir(self.manifests_dir()) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "json")
                    && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                {
                    let model_id = stem.to_string();
                    if seen.insert(model_id.clone()) {
                        let status = self.status(&model_id);
                        models.push(ModelInfo {
                            model_id,
                            status,
                            cache_path: self.cache_dir.join(stem),
                        });
                    }
                }
            }
        }

        // Scan cache directory for legacy models not in the status map
        if let Ok(entries) = std::fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name == "blobs" || name == "manifests" {
                    continue;
                }
                if seen.insert(name.clone()) && entry.path().is_dir() {
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

    /// Path to the partial download state file for a model.
    #[cfg(feature = "local-ml")]
    fn partial_state_path(&self, model_id: &str) -> PathBuf {
        self.cache_dir.join(format!("{model_id}.partial"))
    }

    /// Load partial download state, if any.
    #[cfg(feature = "local-ml")]
    fn load_partial_state(&self, model_id: &str) -> Option<PartialDownloadState> {
        let path = self.partial_state_path(model_id);
        let data = std::fs::read_to_string(&path).ok()?;
        serde_json::from_str(&data).ok()
    }

    /// Save partial download state.
    #[cfg(feature = "local-ml")]
    fn save_partial_state(&self, state: &PartialDownloadState) -> anyhow::Result<()> {
        let path = self.partial_state_path(&state.model_id);
        std::fs::write(&path, serde_json::to_string(state)?)?;
        Ok(())
    }

    /// Remove partial download state (called on completion).
    #[cfg(feature = "local-ml")]
    fn remove_partial_state(&self, model_id: &str) {
        let _ = std::fs::remove_file(self.partial_state_path(model_id));
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

        let blobs = self.blobs_dir();
        let manifests = self.manifests_dir();
        fs::create_dir_all(&blobs)?;
        fs::create_dir_all(&manifests)?;

        // Resume from partial state if available
        let mut partial =
            self.load_partial_state(model_id)
                .unwrap_or_else(|| PartialDownloadState {
                    model_id: model_id.to_string(),
                    hf_repo: hf_repo.to_string(),
                    completed_digests: BTreeMap::new(),
                });

        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(hf_repo.to_string());

        let mut manifest_files = partial.completed_digests.clone();
        let total = files.len();
        for (i, filename) in files.iter().enumerate() {
            progress_cb(i, total);

            // Skip files already completed in a previous run
            if partial.completed_digests.contains_key(*filename) {
                continue;
            }

            // Download with retry
            let mut last_error = None;
            for attempt in 0..MAX_DOWNLOAD_RETRIES {
                if attempt > 0 {
                    eprintln!(
                        "[model_manager] retry {attempt}/{MAX_DOWNLOAD_RETRIES} for {filename}"
                    );
                }

                let result = repo.get(filename);

                match result {
                    Ok(path) => {
                        // Atomic copy: write to .partial temp, then rename
                        let dest = model_path.join(filename);
                        let partial_dest = model_path.join(format!("{filename}.partial"));
                        if !dest.exists() {
                            fs::copy(&path, &partial_dest)?;
                            fs::rename(&partial_dest, &dest)?;
                        }
                        // Store as content-addressable blob
                        if let Some(digest) = sha256_file(&dest) {
                            let blob_path = blobs.join(format!("sha256-{digest}"));
                            if !blob_path.exists() {
                                let partial_blob = blobs.join(format!("sha256-{digest}.partial"));
                                fs::copy(&dest, &partial_blob)?;
                                fs::rename(&partial_blob, &blob_path)?;
                            }
                            manifest_files.insert(filename.to_string(), digest.clone());
                            partial
                                .completed_digests
                                .insert(filename.to_string(), digest);
                        }
                        // Persist partial state after each file completes
                        let _ = self.save_partial_state(&partial);
                        last_error = None;
                        break;
                    }
                    Err(e) => {
                        eprintln!("[model_manager] failed to download {filename}: {e}");
                        last_error = Some(format!("{e}"));
                    }
                }
            }

            if let Some(err) = last_error {
                self.statuses
                    .insert(model_id.to_string(), ModelStatus::Error(err.clone()));
                anyhow::bail!(
                    "model download incomplete: failed to download {filename} after {MAX_DOWNLOAD_RETRIES} retries: {err}"
                );
            }
        }
        progress_cb(total, total);

        // Write manifest
        if !manifest_files.is_empty() {
            let manifest = ModelManifest {
                model_id: model_id.to_string(),
                files: manifest_files,
            };
            let manifest_path = manifests.join(format!("{model_id}.json"));
            fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
        }

        // Clean up partial state on success
        self.remove_partial_state(model_id);

        self.statuses
            .insert(model_id.to_string(), ModelStatus::Ready);
        Ok(())
    }
}

/// Compute SHA-256 digest of a file.
fn sha256_file(path: &Path) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Some(format!("{:x}", hasher.finalize()))
}

/// Check if a model directory contains weight files (config.json for SafeTensors
/// models, or any .gguf file for quantized models).
fn has_model_files(dir: &Path) -> bool {
    if dir.join("config.json").exists() {
        return true;
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.path().extension().is_some_and(|ext| ext == "gguf") {
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

    #[test]
    fn migrate_creates_manifest_and_blobs() {
        let dir = TempDir::new().unwrap();
        let mgr = ModelManager::new(dir.path().to_path_buf());

        // Create a legacy model directory
        let model_dir = dir.path().join("test-model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), r#"{"key": "value"}"#).unwrap();
        fs::write(model_dir.join("weights.gguf"), "fake-model-weights").unwrap();

        // Migrate
        let manifest = mgr.migrate_to_content_addressable("test-model").unwrap();
        assert_eq!(manifest.model_id, "test-model");
        assert_eq!(manifest.files.len(), 2);
        assert!(manifest.files.contains_key("config.json"));
        assert!(manifest.files.contains_key("weights.gguf"));

        // Blobs exist
        for digest in manifest.files.values() {
            let blob = dir.path().join("blobs").join(format!("sha256-{digest}"));
            assert!(blob.exists(), "blob should exist: {}", blob.display());
        }

        // Manifest file exists
        let manifest_path = dir.path().join("manifests/test-model.json");
        assert!(manifest_path.exists());

        // Status should still be Ready
        assert_eq!(mgr.status("test-model"), ModelStatus::Ready);
    }

    #[test]
    fn manifest_based_status_check() {
        let dir = TempDir::new().unwrap();
        let mgr = ModelManager::new(dir.path().to_path_buf());

        // Create manifest + blobs directly (simulating post-migration state)
        let blobs = dir.path().join("blobs");
        let manifests = dir.path().join("manifests");
        fs::create_dir_all(&blobs).unwrap();
        fs::create_dir_all(&manifests).unwrap();

        let content = b"test content";
        let digest = format!("{:x}", Sha256::digest(content));
        fs::write(blobs.join(format!("sha256-{digest}")), content).unwrap();

        let manifest = ModelManifest {
            model_id: "manifest-model".to_string(),
            files: BTreeMap::from([("model.gguf".to_string(), digest)]),
        };
        fs::write(
            manifests.join("manifest-model.json"),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();

        // Should be Ready via manifest (no legacy dir needed)
        assert_eq!(mgr.status("manifest-model"), ModelStatus::Ready);
    }

    #[test]
    fn verify_integrity_detects_corruption() {
        let dir = TempDir::new().unwrap();
        let mgr = ModelManager::new(dir.path().to_path_buf());

        // Create a model and migrate it
        let model_dir = dir.path().join("integrity-test");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("data.bin"), "original content").unwrap();
        mgr.migrate_to_content_addressable("integrity-test")
            .unwrap();

        // Integrity should pass
        assert!(mgr.verify_integrity("integrity-test"));

        // Corrupt a blob
        let manifest = mgr.read_manifest("integrity-test").unwrap();
        let digest = manifest.files.values().next().unwrap();
        let blob = dir.path().join("blobs").join(format!("sha256-{digest}"));
        fs::write(&blob, "corrupted!").unwrap();

        // Integrity should fail
        assert!(!mgr.verify_integrity("integrity-test"));
    }

    #[test]
    fn sha256_file_produces_correct_hash() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "hello world").unwrap();
        let digest = sha256_file(&path).unwrap();
        // Known SHA-256 of "hello world"
        assert_eq!(
            digest,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn list_models_includes_manifest_models() {
        let dir = TempDir::new().unwrap();
        let mgr = ModelManager::new(dir.path().to_path_buf());

        // Create manifest-only model
        let blobs = dir.path().join("blobs");
        let manifests = dir.path().join("manifests");
        fs::create_dir_all(&blobs).unwrap();
        fs::create_dir_all(&manifests).unwrap();

        let content = b"data";
        let digest = format!("{:x}", Sha256::digest(content));
        fs::write(blobs.join(format!("sha256-{digest}")), content).unwrap();

        let manifest = ModelManifest {
            model_id: "listed-model".to_string(),
            files: BTreeMap::from([("w.gguf".to_string(), digest)]),
        };
        fs::write(
            manifests.join("listed-model.json"),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();

        let models = mgr.list_models();
        assert!(
            models.iter().any(|m| m.model_id == "listed-model"),
            "should list manifest-based model"
        );
    }
}
