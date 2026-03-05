use crate::hardware::{LocalModelRuntimePolicy, detect_hardware, recommend_runtime_policy};
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
    /// SHA-256 digests of completed files (filename -> digest).
    completed_digests: BTreeMap<String, String>,
}

/// Maximum number of retries per file download.
#[cfg(feature = "local-ml")]
const MAX_DOWNLOAD_RETRIES: usize = 6;

/// If an individual file download takes longer than this without completing,
/// treat it as stalled and retry.
#[cfg(feature = "local-ml")]
const DOWNLOAD_STALL_TIMEOUT_SECS: u64 = 60;

/// Manifest for a content-addressable model cache entry.
///
/// Maps logical filenames to SHA-256 digests stored in the `blobs/` directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub model_id: String,
    pub files: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RuntimeSlotState {
    loaded_at_epoch_secs: u64,
    last_used_epoch_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeLifecycleEvent {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    pub at_epoch_secs: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct RuntimeLifecycleMetrics {
    pub total_slot_activations: u64,
    pub total_slot_reuses: u64,
    pub total_capacity_evictions: u64,
    pub total_idle_evictions: u64,
    pub total_queue_enqueued: u64,
    pub total_queue_completed: u64,
    pub total_queue_rejected: u64,
    pub total_queue_wait_timeouts: u64,
    pub total_memory_admission_denied: u64,
    pub total_memory_pressure_evictions: u64,
    pub total_runner_load_waits: u64,
    pub total_runner_reloads: u64,
    pub total_runner_load_failures: u64,
    pub max_observed_queue_depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct RuntimeSchedulerPolicySnapshot {
    pub max_concurrent_requests: usize,
    pub max_queue_depth: usize,
    pub max_queue_wait_ms: u64,
}

impl Default for RuntimeSchedulerPolicySnapshot {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 1,
            max_queue_depth: 4,
            max_queue_wait_ms: 45_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeLifecycleSnapshot {
    pub max_loaded_models: usize,
    pub keep_warm_secs: u64,
    pub aggressive_eviction: bool,
    #[serde(default)]
    pub scheduler: RuntimeSchedulerPolicySnapshot,
    pub warm_models: Vec<String>,
    pub metrics: RuntimeLifecycleMetrics,
    pub recent_events: Vec<RuntimeLifecycleEvent>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RuntimeStateFile {
    #[serde(default)]
    runtime_slots: BTreeMap<String, RuntimeSlotState>,
    #[serde(default)]
    metrics: RuntimeLifecycleMetrics,
    #[serde(default)]
    scheduler_policy: RuntimeSchedulerPolicySnapshot,
    #[serde(default)]
    recent_events: Vec<RuntimeLifecycleEvent>,
}

const MAX_RUNTIME_EVENTS: usize = 64;

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
    runtime_policy: LocalModelRuntimePolicy,
    runtime_scheduler_policy: RuntimeSchedulerPolicySnapshot,
    runtime_slots: BTreeMap<String, RuntimeSlotState>,
    runtime_metrics: RuntimeLifecycleMetrics,
    runtime_events: Vec<RuntimeLifecycleEvent>,
}

impl ModelManager {
    pub fn new(cache_dir: PathBuf) -> Self {
        let hardware = detect_hardware();
        let runtime_policy = recommend_runtime_policy(&hardware);
        Self::with_runtime_policy(cache_dir, runtime_policy)
    }

    pub fn with_runtime_policy(
        cache_dir: PathBuf,
        runtime_policy: LocalModelRuntimePolicy,
    ) -> Self {
        let persisted = load_runtime_state_file(&cache_dir).unwrap_or_default();
        let mut runtime_scheduler_policy = persisted.scheduler_policy;
        if runtime_scheduler_policy.max_concurrent_requests == 0 {
            runtime_scheduler_policy.max_concurrent_requests =
                runtime_policy.max_loaded_models.max(1);
        }
        if runtime_scheduler_policy.max_queue_depth == 0 {
            runtime_scheduler_policy.max_queue_depth = runtime_scheduler_policy
                .max_concurrent_requests
                .saturating_mul(4)
                .max(1);
        }
        if runtime_scheduler_policy.max_queue_wait_ms == 0 {
            runtime_scheduler_policy.max_queue_wait_ms =
                RuntimeSchedulerPolicySnapshot::default().max_queue_wait_ms;
        }
        Self {
            cache_dir,
            statuses: BTreeMap::new(),
            runtime_policy,
            runtime_scheduler_policy,
            runtime_slots: persisted.runtime_slots,
            runtime_metrics: persisted.metrics,
            runtime_events: persisted.recent_events,
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
        // Legacy layout path -- files are accessed directly from here
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

    /// Runtime lifecycle policy used for local model keep-warm behavior.
    pub fn runtime_policy(&self) -> LocalModelRuntimePolicy {
        self.runtime_policy
    }

    /// List currently warm runtime models.
    pub fn warm_runtime_models(&self) -> Vec<String> {
        self.runtime_slots.keys().cloned().collect()
    }

    /// Snapshot runtime lifecycle state (policy, warm models, metrics, events).
    pub fn runtime_snapshot(&self) -> RuntimeLifecycleSnapshot {
        RuntimeLifecycleSnapshot {
            max_loaded_models: self.runtime_policy.max_loaded_models,
            keep_warm_secs: self.runtime_policy.keep_warm_secs,
            aggressive_eviction: self.runtime_policy.aggressive_eviction,
            scheduler: self.runtime_scheduler_policy.clone(),
            warm_models: self.warm_runtime_models(),
            metrics: self.runtime_metrics.clone(),
            recent_events: self.runtime_events.clone(),
        }
    }

    /// Persist scheduler queue/backpressure policy for runtime diagnostics.
    pub fn record_runtime_scheduler_policy(
        &mut self,
        max_concurrent_requests: usize,
        max_queue_depth: usize,
        max_queue_wait_ms: u64,
    ) {
        let normalized = RuntimeSchedulerPolicySnapshot {
            max_concurrent_requests: max_concurrent_requests.max(1),
            max_queue_depth: max_queue_depth.max(1),
            max_queue_wait_ms: max_queue_wait_ms.max(1),
        };
        if self.runtime_scheduler_policy == normalized {
            return;
        }
        self.runtime_scheduler_policy = normalized.clone();
        self.push_runtime_event(RuntimeLifecycleEvent {
            kind: "scheduler_policy_updated".to_string(),
            model_id: None,
            at_epoch_secs: now_epoch_secs(),
            detail: Some(format!(
                "max_concurrent={},max_queue={},max_wait_ms={}",
                normalized.max_concurrent_requests,
                normalized.max_queue_depth,
                normalized.max_queue_wait_ms
            )),
        });
        self.persist_runtime_state();
    }

    /// Record queue depth when a scheduler enqueues a request.
    pub fn record_runtime_queue_enqueued(&mut self, queue_depth: usize) {
        self.runtime_metrics.total_queue_enqueued =
            self.runtime_metrics.total_queue_enqueued.saturating_add(1);
        self.runtime_metrics.max_observed_queue_depth = self
            .runtime_metrics
            .max_observed_queue_depth
            .max(queue_depth);
        self.push_runtime_event(RuntimeLifecycleEvent {
            kind: "request_queued".to_string(),
            model_id: None,
            at_epoch_secs: now_epoch_secs(),
            detail: Some(format!("depth={queue_depth}")),
        });
        self.persist_runtime_state();
    }

    /// Record completion of a queued request.
    pub fn record_runtime_queue_completed(&mut self) {
        self.runtime_metrics.total_queue_completed =
            self.runtime_metrics.total_queue_completed.saturating_add(1);
        self.push_runtime_event(RuntimeLifecycleEvent {
            kind: "request_completed".to_string(),
            model_id: None,
            at_epoch_secs: now_epoch_secs(),
            detail: None,
        });
        self.persist_runtime_state();
    }

    /// Record rejection due to queue/backpressure admission policy.
    pub fn record_runtime_queue_rejected(
        &mut self,
        active_requests: usize,
        queued_requests: usize,
        max_concurrent_requests: usize,
        max_queue_depth: usize,
    ) {
        self.runtime_metrics.total_queue_rejected =
            self.runtime_metrics.total_queue_rejected.saturating_add(1);
        self.push_runtime_event(RuntimeLifecycleEvent {
            kind: "request_rejected_queue_full".to_string(),
            model_id: None,
            at_epoch_secs: now_epoch_secs(),
            detail: Some(format!(
                "active={active_requests},queued={queued_requests},max_concurrent={max_concurrent_requests},max_queue={max_queue_depth}"
            )),
        });
        self.persist_runtime_state();
    }

    /// Record timeout while waiting in queue admission backpressure.
    pub fn record_runtime_queue_wait_timeout(
        &mut self,
        waited_ms: u64,
        active_requests: usize,
        queued_requests: usize,
        max_concurrent_requests: usize,
        max_queue_depth: usize,
    ) {
        self.runtime_metrics.total_queue_wait_timeouts = self
            .runtime_metrics
            .total_queue_wait_timeouts
            .saturating_add(1);
        self.push_runtime_event(RuntimeLifecycleEvent {
            kind: "request_timed_out_queue_wait".to_string(),
            model_id: None,
            at_epoch_secs: now_epoch_secs(),
            detail: Some(format!(
                "waited_ms={waited_ms},active={active_requests},queued={queued_requests},max_concurrent={max_concurrent_requests},max_queue={max_queue_depth}"
            )),
        });
        self.persist_runtime_state();
    }

    /// Record memory-based admission denial before loading a model runner.
    pub fn record_runtime_memory_admission_denied(
        &mut self,
        model_id: &str,
        available_mb: u64,
        reason: &str,
    ) {
        self.runtime_metrics.total_memory_admission_denied = self
            .runtime_metrics
            .total_memory_admission_denied
            .saturating_add(1);
        self.push_runtime_event(RuntimeLifecycleEvent {
            kind: "runner_admission_denied_memory".to_string(),
            model_id: Some(model_id.to_string()),
            at_epoch_secs: now_epoch_secs(),
            detail: Some(format!(
                "available_mb={available_mb}; {}",
                compact_detail(reason)
            )),
        });
        self.persist_runtime_state();
    }

    /// Evict one runner slot as pressure relief before retrying admission checks.
    pub fn evict_one_runtime_model_for_memory_pressure(
        &mut self,
        target_model_id: &str,
        available_mb: u64,
        attempt: usize,
    ) -> Option<String> {
        let victim = self
            .runtime_slots
            .iter()
            .filter(|(candidate, _)| candidate.as_str() != target_model_id)
            .min_by_key(|(_, slot)| (slot.last_used_epoch_secs, slot.loaded_at_epoch_secs))
            .map(|(candidate, _)| candidate.clone())
            .or_else(|| self.runtime_slots.keys().next().cloned())?;
        self.runtime_slots.remove(&victim);
        self.runtime_metrics.total_memory_pressure_evictions = self
            .runtime_metrics
            .total_memory_pressure_evictions
            .saturating_add(1);
        self.push_runtime_event(RuntimeLifecycleEvent {
            kind: "runner_evicted_memory_pressure".to_string(),
            model_id: Some(victim.clone()),
            at_epoch_secs: now_epoch_secs(),
            detail: Some(format!(
                "target={target_model_id},attempt={attempt},available_mb={available_mb}"
            )),
        });
        self.persist_runtime_state();
        Some(victim)
    }

    /// Record a runner reload attempt after a generation failure.
    pub fn record_runtime_runner_reload(&mut self, model_id: &str, reason: &str) {
        self.runtime_metrics.total_runner_reloads =
            self.runtime_metrics.total_runner_reloads.saturating_add(1);
        self.push_runtime_event(RuntimeLifecycleEvent {
            kind: "runner_reload".to_string(),
            model_id: Some(model_id.to_string()),
            at_epoch_secs: now_epoch_secs(),
            detail: Some(compact_detail(reason)),
        });
        self.persist_runtime_state();
    }

    /// Record a request waiting on the global runner load lane.
    pub fn record_runtime_runner_load_wait(&mut self, model_id: &str, blocked_by: &str) {
        self.runtime_metrics.total_runner_load_waits = self
            .runtime_metrics
            .total_runner_load_waits
            .saturating_add(1);
        self.push_runtime_event(RuntimeLifecycleEvent {
            kind: "runner_load_wait".to_string(),
            model_id: Some(model_id.to_string()),
            at_epoch_secs: now_epoch_secs(),
            detail: Some(format!("blocked_by={}", compact_detail(blocked_by))),
        });
        self.persist_runtime_state();
    }

    /// Record a terminal runner load/generation failure.
    pub fn record_runtime_runner_load_failure(&mut self, model_id: &str, reason: &str) {
        self.runtime_metrics.total_runner_load_failures = self
            .runtime_metrics
            .total_runner_load_failures
            .saturating_add(1);
        self.push_runtime_event(RuntimeLifecycleEvent {
            kind: "runner_load_failure".to_string(),
            model_id: Some(model_id.to_string()),
            at_epoch_secs: now_epoch_secs(),
            detail: Some(compact_detail(reason)),
        });
        self.persist_runtime_state();
    }

    /// Mark a model as recently used by the local runner and enforce slot caps.
    ///
    /// Returns evicted model ids (if any) from slot pressure.
    pub fn mark_runtime_used(&mut self, model_id: &str) -> Vec<String> {
        self.mark_runtime_used_at(model_id, now_epoch_secs())
    }

    /// Evict models that have been idle longer than the keep-warm window.
    pub fn evict_idle_runtime_models(&mut self) -> Vec<String> {
        self.evict_idle_runtime_models_at(now_epoch_secs())
    }

    fn mark_runtime_used_at(&mut self, model_id: &str, now_epoch_secs: u64) -> Vec<String> {
        if model_id.trim().is_empty() {
            return Vec::new();
        }

        match self.runtime_slots.get_mut(model_id) {
            Some(slot) => {
                slot.last_used_epoch_secs = now_epoch_secs;
                self.runtime_metrics.total_slot_reuses =
                    self.runtime_metrics.total_slot_reuses.saturating_add(1);
                self.push_runtime_event(RuntimeLifecycleEvent {
                    kind: "runner_reused".to_string(),
                    model_id: Some(model_id.to_string()),
                    at_epoch_secs: now_epoch_secs,
                    detail: None,
                });
            }
            None => {
                self.runtime_slots.insert(
                    model_id.to_string(),
                    RuntimeSlotState {
                        loaded_at_epoch_secs: now_epoch_secs,
                        last_used_epoch_secs: now_epoch_secs,
                    },
                );
                self.runtime_metrics.total_slot_activations = self
                    .runtime_metrics
                    .total_slot_activations
                    .saturating_add(1);
                self.push_runtime_event(RuntimeLifecycleEvent {
                    kind: "runner_activated".to_string(),
                    model_id: Some(model_id.to_string()),
                    at_epoch_secs: now_epoch_secs,
                    detail: None,
                });
            }
        }

        let cap = self.runtime_policy.max_loaded_models.max(1);
        let mut evicted = Vec::new();
        while self.runtime_slots.len() > cap {
            let victim = self
                .runtime_slots
                .iter()
                .filter(|(candidate, _)| candidate.as_str() != model_id)
                .min_by_key(|(_, slot)| (slot.last_used_epoch_secs, slot.loaded_at_epoch_secs))
                .map(|(candidate, _)| candidate.clone())
                .or_else(|| self.runtime_slots.keys().next().cloned());
            let Some(victim) = victim else {
                break;
            };
            self.runtime_slots.remove(&victim);
            evicted.push(victim);
        }

        if !evicted.is_empty() {
            self.runtime_metrics.total_capacity_evictions = self
                .runtime_metrics
                .total_capacity_evictions
                .saturating_add(evicted.len() as u64);
            for victim in &evicted {
                self.push_runtime_event(RuntimeLifecycleEvent {
                    kind: "runner_evicted_capacity".to_string(),
                    model_id: Some(victim.clone()),
                    at_epoch_secs: now_epoch_secs,
                    detail: Some(format!("cap={cap}")),
                });
            }
        }

        self.persist_runtime_state();

        evicted
    }

    fn evict_idle_runtime_models_at(&mut self, now_epoch_secs: u64) -> Vec<String> {
        let mut keep_warm = self.runtime_policy.keep_warm_secs.max(1);
        if self.runtime_policy.aggressive_eviction {
            keep_warm = keep_warm.saturating_div(2).max(1);
        }
        let threshold = now_epoch_secs.saturating_sub(keep_warm);
        let mut evicted = Vec::new();
        self.runtime_slots.retain(|model_id, slot| {
            let keep = slot.last_used_epoch_secs >= threshold;
            if !keep {
                evicted.push(model_id.clone());
            }
            keep
        });
        if !evicted.is_empty() {
            self.runtime_metrics.total_idle_evictions = self
                .runtime_metrics
                .total_idle_evictions
                .saturating_add(evicted.len() as u64);
            for victim in &evicted {
                self.push_runtime_event(RuntimeLifecycleEvent {
                    kind: "runner_evicted_idle".to_string(),
                    model_id: Some(victim.clone()),
                    at_epoch_secs: now_epoch_secs,
                    detail: Some(format!("threshold={threshold}")),
                });
            }
            self.persist_runtime_state();
        }
        evicted
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
                let _ = self.mark_runtime_used(model_id);
                let _ = self.evict_idle_runtime_models();
                Ok(model_path)
            }
            ModelStatus::NotDownloaded => {
                #[cfg(feature = "local-ml")]
                {
                    self.download_model(model_id, hf_repo, files, &progress_cb)?;
                    let _ = self.mark_runtime_used(model_id);
                    let _ = self.evict_idle_runtime_models();
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

    /// Download a single file with retry logic and stall detection.
    ///
    /// Returns `(filename, digest)` on success.
    #[cfg(feature = "local-ml")]
    fn download_single_file(
        repo: &hf_hub::api::sync::ApiRepo,
        filename: &str,
        model_path: &Path,
        blobs: &Path,
    ) -> Result<(String, String), String> {
        use std::fs;
        use std::time::Instant;

        let stall_timeout = std::time::Duration::from_secs(DOWNLOAD_STALL_TIMEOUT_SECS);
        let mut last_error = None;

        for attempt in 0..MAX_DOWNLOAD_RETRIES {
            if attempt > 0 {
                eprintln!("[model_manager] retry {attempt}/{MAX_DOWNLOAD_RETRIES} for {filename}");
            }

            let started = Instant::now();
            let result = repo.get(filename);

            // Stall detection: if the download took longer than the timeout and
            // still failed, log it explicitly so the user knows why we retried.
            if started.elapsed() > stall_timeout && result.is_err() {
                eprintln!(
                    "[model_manager] download of {filename} appears stalled (>{DOWNLOAD_STALL_TIMEOUT_SECS}s), retrying"
                );
            }

            match result {
                Ok(path) => {
                    // Atomic copy: write to .partial temp, then rename
                    let dest = model_path.join(filename);
                    let partial_dest = model_path.join(format!("{filename}.partial"));
                    if !dest.exists()
                        && let Err(e) = fs::copy(&path, &partial_dest)
                            .and_then(|_| fs::rename(&partial_dest, &dest))
                    {
                        last_error = Some(format!("copy/rename failed: {e}"));
                        continue;
                    }
                    // Store as content-addressable blob
                    if let Some(digest) = sha256_file(&dest) {
                        let blob_path = blobs.join(format!("sha256-{digest}"));
                        if !blob_path.exists() {
                            let partial_blob = blobs.join(format!("sha256-{digest}.partial"));
                            if let Err(e) = fs::copy(&dest, &partial_blob)
                                .and_then(|_| fs::rename(&partial_blob, &blob_path))
                            {
                                last_error = Some(format!("blob store failed: {e}"));
                                continue;
                            }
                        }
                        return Ok((filename.to_string(), digest));
                    }
                    last_error = Some("sha256 computation failed".to_string());
                }
                Err(e) => {
                    eprintln!("[model_manager] failed to download {filename}: {e}");
                    last_error = Some(format!("{e}"));
                }
            }
        }
        Err(last_error.unwrap_or_else(|| "unknown error".to_string()))
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
        use std::sync::Mutex;

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

        let mut manifest_files = partial.completed_digests.clone();
        let total = files.len();

        // Identify files that still need downloading
        let pending: Vec<&str> = files
            .iter()
            .filter(|f| !partial.completed_digests.contains_key(**f))
            .copied()
            .collect();

        // Report progress for already-completed files
        let completed_so_far = total - pending.len();
        if completed_so_far > 0 {
            progress_cb(completed_so_far, total);
        }

        if !pending.is_empty() {
            let api = hf_hub::api::sync::Api::new()?;
            let repo = api.model(hf_repo.to_string());

            // Parallel download using std::thread::scope.
            // Each thread gets its own reference to the repo handle (Send+Sync).
            type DownloadResult = Result<(String, String), (String, String)>;
            let results: Mutex<Vec<DownloadResult>> = Mutex::new(Vec::new());

            std::thread::scope(|s| {
                for filename in &pending {
                    let repo = &repo;
                    let model_path = &model_path;
                    let blobs = &blobs;
                    let results = &results;

                    s.spawn(move || {
                        let outcome = Self::download_single_file(repo, filename, model_path, blobs);
                        // Progress callback is not Sync, so we cannot call it from
                        // parallel threads. We record results and report progress
                        // after joining.
                        match outcome {
                            Ok(pair) => {
                                results.lock().unwrap().push(Ok(pair));
                            }
                            Err(e) => {
                                results.lock().unwrap().push(Err((filename.to_string(), e)));
                            }
                        }
                    });
                }
            });

            // Process results (single-threaded after thread::scope join)
            let results = results.into_inner().unwrap();
            for result in results {
                match result {
                    Ok((filename, digest)) => {
                        manifest_files.insert(filename.clone(), digest.clone());
                        partial.completed_digests.insert(filename, digest);
                        let _ = self.save_partial_state(&partial);
                    }
                    Err((filename, err)) => {
                        self.statuses
                            .insert(model_id.to_string(), ModelStatus::Error(err.clone()));
                        anyhow::bail!(
                            "model download incomplete: failed to download {filename} after {MAX_DOWNLOAD_RETRIES} retries: {err}"
                        );
                    }
                }
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

    fn runtime_state_path(&self) -> PathBuf {
        self.cache_dir.join("runtime_state.json")
    }

    fn push_runtime_event(&mut self, event: RuntimeLifecycleEvent) {
        self.runtime_events.push(event);
        if self.runtime_events.len() > MAX_RUNTIME_EVENTS {
            let drop_count = self.runtime_events.len() - MAX_RUNTIME_EVENTS;
            self.runtime_events.drain(..drop_count);
        }
    }

    fn persist_runtime_state(&self) {
        let state = RuntimeStateFile {
            runtime_slots: self.runtime_slots.clone(),
            metrics: self.runtime_metrics.clone(),
            scheduler_policy: self.runtime_scheduler_policy.clone(),
            recent_events: self.runtime_events.clone(),
        };
        let path = self.runtime_state_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp_path = path.with_extension("json.partial");
        if let Ok(payload) = serde_json::to_vec_pretty(&state) {
            let _ = std::fs::write(&tmp_path, payload);
            let _ = std::fs::rename(&tmp_path, &path);
        }
    }
}

fn load_runtime_state_file(cache_dir: &Path) -> Option<RuntimeStateFile> {
    let path = cache_dir.join("runtime_state.json");
    let raw = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
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

fn compact_detail(detail: &str) -> String {
    const MAX_DETAIL_CHARS: usize = 220;
    let trimmed = detail.trim();
    if trimmed.chars().count() <= MAX_DETAIL_CHARS {
        return trimmed.to_string();
    }
    let clipped: String = trimmed.chars().take(MAX_DETAIL_CHARS).collect();
    format!("{clipped}...")
}

fn now_epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
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

        // Create the model directory with config.json -> Ready (safetensors model)
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

    #[test]
    fn runtime_slots_respect_capacity() {
        let dir = TempDir::new().unwrap();
        let mut mgr = ModelManager::with_runtime_policy(
            dir.path().to_path_buf(),
            LocalModelRuntimePolicy {
                max_loaded_models: 2,
                keep_warm_secs: 600,
                aggressive_eviction: false,
            },
        );

        assert!(mgr.mark_runtime_used_at("model-a", 10).is_empty());
        assert!(mgr.mark_runtime_used_at("model-b", 11).is_empty());
        let evicted = mgr.mark_runtime_used_at("model-c", 12);

        assert_eq!(mgr.warm_runtime_models().len(), 2);
        assert_eq!(evicted.len(), 1);
        assert!(
            !mgr.warm_runtime_models().iter().any(|m| m == "model-a"),
            "least recently used model should be evicted first"
        );
    }

    #[test]
    fn runtime_slots_evict_idle_models() {
        let dir = TempDir::new().unwrap();
        let mut mgr = ModelManager::with_runtime_policy(
            dir.path().to_path_buf(),
            LocalModelRuntimePolicy {
                max_loaded_models: 3,
                keep_warm_secs: 30,
                aggressive_eviction: true,
            },
        );

        mgr.mark_runtime_used_at("model-a", 100);
        mgr.mark_runtime_used_at("model-b", 120);
        mgr.mark_runtime_used_at("model-c", 129);
        let evicted = mgr.evict_idle_runtime_models_at(131);

        assert_eq!(evicted, vec!["model-a".to_string()]);
        assert_eq!(mgr.warm_runtime_models().len(), 2);
        assert!(mgr.warm_runtime_models().iter().any(|m| m == "model-b"));
        assert!(mgr.warm_runtime_models().iter().any(|m| m == "model-c"));
    }

    #[test]
    fn runtime_snapshot_tracks_lifecycle_metrics() {
        let dir = TempDir::new().unwrap();
        let mut mgr = ModelManager::with_runtime_policy(
            dir.path().to_path_buf(),
            LocalModelRuntimePolicy {
                max_loaded_models: 1,
                keep_warm_secs: 30,
                aggressive_eviction: false,
            },
        );

        mgr.record_runtime_queue_enqueued(2);
        mgr.record_runtime_queue_completed();
        mgr.record_runtime_queue_rejected(1, 1, 1, 1);
        mgr.record_runtime_memory_admission_denied("model-a", 1024, "requires more memory");
        mgr.record_runtime_runner_load_wait("model-a", "model-b");
        mgr.record_runtime_runner_reload("model-a", "first generation failed");
        mgr.record_runtime_runner_load_failure("model-b", "backend init failed");
        assert!(mgr.mark_runtime_used_at("model-a", 100).is_empty());
        assert!(mgr.mark_runtime_used_at("model-a", 110).is_empty());
        let capacity_evicted = mgr.mark_runtime_used_at("model-b", 120);
        assert_eq!(capacity_evicted, vec!["model-a".to_string()]);
        let idle_evicted = mgr.evict_idle_runtime_models_at(200);
        assert_eq!(idle_evicted, vec!["model-b".to_string()]);

        let snapshot = mgr.runtime_snapshot();
        assert!(snapshot.warm_models.is_empty());
        assert_eq!(snapshot.metrics.total_slot_activations, 2);
        assert_eq!(snapshot.metrics.total_slot_reuses, 1);
        assert_eq!(snapshot.metrics.total_capacity_evictions, 1);
        assert_eq!(snapshot.metrics.total_idle_evictions, 1);
        assert_eq!(snapshot.metrics.total_queue_enqueued, 1);
        assert_eq!(snapshot.metrics.total_queue_completed, 1);
        assert_eq!(snapshot.metrics.total_queue_rejected, 1);
        assert_eq!(snapshot.metrics.total_memory_admission_denied, 1);
        assert_eq!(snapshot.metrics.total_runner_load_waits, 1);
        assert_eq!(snapshot.metrics.total_runner_reloads, 1);
        assert_eq!(snapshot.metrics.total_runner_load_failures, 1);
        assert_eq!(snapshot.metrics.max_observed_queue_depth, 2);
        assert!(!snapshot.recent_events.is_empty());
    }

    #[test]
    fn runtime_state_persists_across_restarts() {
        let dir = TempDir::new().unwrap();
        let policy = LocalModelRuntimePolicy {
            max_loaded_models: 2,
            keep_warm_secs: 600,
            aggressive_eviction: false,
        };

        {
            let mut mgr = ModelManager::with_runtime_policy(dir.path().to_path_buf(), policy);
            let _ = mgr.mark_runtime_used_at("persisted-model", 42);
            mgr.record_runtime_queue_enqueued(1);
        }

        let restored = ModelManager::with_runtime_policy(dir.path().to_path_buf(), policy);
        let snapshot = restored.runtime_snapshot();
        assert_eq!(
            snapshot.warm_models,
            vec!["persisted-model".to_string()],
            "warm model slots should survive process restart"
        );
        assert_eq!(snapshot.metrics.total_slot_activations, 1);
        assert_eq!(snapshot.metrics.total_queue_enqueued, 1);
    }
}
