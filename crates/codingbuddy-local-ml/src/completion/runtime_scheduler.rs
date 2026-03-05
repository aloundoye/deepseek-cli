use super::{GenOpts, LocalGenBackend};
use crate::ModelManager;
use crate::model_manager::RuntimeLifecycleSnapshot;
use anyhow::{Result, anyhow};
use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex};

/// Factory trait for loading local generation backends on demand.
pub trait BackendFactory: Send + Sync {
    fn load_backend(&self, model_id: &str) -> Result<Arc<dyn LocalGenBackend>>;
}

impl<F> BackendFactory for F
where
    F: Fn(&str) -> Result<Arc<dyn LocalGenBackend>> + Send + Sync,
{
    fn load_backend(&self, model_id: &str) -> Result<Arc<dyn LocalGenBackend>> {
        self(model_id)
    }
}

#[derive(Debug, Clone, Copy)]
struct RequestGateState {
    active_requests: usize,
    queued_requests: usize,
    max_concurrent_requests: usize,
}

/// Observability snapshot for the local runtime scheduler.
#[derive(Debug, Clone)]
pub struct LocalRuntimeSchedulerSnapshot {
    pub active_requests: usize,
    pub queued_requests: usize,
    pub max_concurrent_requests: usize,
    pub loaded_runners: Vec<String>,
    pub lifecycle: RuntimeLifecycleSnapshot,
}

struct SharedState {
    model_manager: Mutex<ModelManager>,
    runners: Mutex<BTreeMap<String, Arc<dyn LocalGenBackend>>>,
    gate: Mutex<RequestGateState>,
    gate_cv: Condvar,
    loader: Arc<dyn BackendFactory>,
}

/// Scheduler-style lifecycle manager for local generation runners.
///
/// Responsibilities:
/// - queue and gate requests with a bounded concurrent runner limit
/// - lazily load and cache generation runners by model id
/// - run keep-warm maintenance and idle/capacity evictions
/// - expose runtime snapshots for diagnostics/doctor output
#[derive(Clone)]
pub struct LocalRunnerLifecycleManager {
    shared: Arc<SharedState>,
}

struct RequestPermit {
    shared: Arc<SharedState>,
    active: bool,
}

impl Drop for RequestPermit {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        {
            let mut gate = self.shared.gate.lock().unwrap_or_else(|e| e.into_inner());
            gate.active_requests = gate.active_requests.saturating_sub(1);
            self.shared.gate_cv.notify_one();
        }

        let mut mgr = self
            .shared
            .model_manager
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        mgr.record_runtime_queue_completed();
    }
}

impl LocalRunnerLifecycleManager {
    /// Build a lifecycle manager using runtime-policy-derived concurrency.
    pub fn new(model_manager: ModelManager, loader: Arc<dyn BackendFactory>) -> Self {
        let max_concurrent = model_manager.runtime_policy().max_loaded_models.max(1);
        Self::with_limits(model_manager, loader, max_concurrent)
    }

    /// Build a lifecycle manager with an explicit concurrent request cap.
    pub fn with_limits(
        model_manager: ModelManager,
        loader: Arc<dyn BackendFactory>,
        max_concurrent_requests: usize,
    ) -> Self {
        Self {
            shared: Arc::new(SharedState {
                model_manager: Mutex::new(model_manager),
                runners: Mutex::new(BTreeMap::new()),
                gate: Mutex::new(RequestGateState {
                    active_requests: 0,
                    queued_requests: 0,
                    max_concurrent_requests: max_concurrent_requests.max(1),
                }),
                gate_cv: Condvar::new(),
                loader,
            }),
        }
    }

    /// Warm a model runner proactively.
    pub fn prewarm(&self, model_id: &str) -> Result<()> {
        let _ = self.ensure_runner(model_id)?;
        Ok(())
    }

    /// Generate text using a cached or lazily loaded runner.
    ///
    /// On generation failure, invalidates that runner and retries once with reload.
    pub fn generate(&self, model_id: &str, prompt: &str, opts: &GenOpts) -> Result<String> {
        let _permit = self.acquire_request_permit();
        let _ = self.maintenance_tick();

        let backend = self.ensure_runner(model_id)?;
        match backend.generate(prompt, opts) {
            Ok(output) => Ok(output),
            Err(first_error) => {
                self.invalidate_runner(model_id);
                let reloaded = self.ensure_runner(model_id)?;
                reloaded.generate(prompt, opts).map_err(|retry_error| {
                    anyhow!(
                        "generation failed for model '{model_id}': {first_error}; reload retry failed: {retry_error}"
                    )
                })
            }
        }
    }

    /// Perform keep-warm maintenance and evict idle runners.
    /// Returns model ids evicted due to idleness.
    pub fn maintenance_tick(&self) -> Vec<String> {
        let evicted = {
            let mut mgr = self
                .shared
                .model_manager
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            mgr.evict_idle_runtime_models()
        };
        self.remove_runners(&evicted);
        evicted
    }

    /// Return scheduler + runtime lifecycle diagnostics.
    pub fn snapshot(&self) -> LocalRuntimeSchedulerSnapshot {
        let gate = self.shared.gate.lock().unwrap_or_else(|e| e.into_inner());
        let loaded_runners = self
            .shared
            .runners
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        let lifecycle = self
            .shared
            .model_manager
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .runtime_snapshot();

        LocalRuntimeSchedulerSnapshot {
            active_requests: gate.active_requests,
            queued_requests: gate.queued_requests,
            max_concurrent_requests: gate.max_concurrent_requests,
            loaded_runners,
            lifecycle,
        }
    }

    fn acquire_request_permit(&self) -> RequestPermit {
        let queue_depth = {
            let mut gate = self.shared.gate.lock().unwrap_or_else(|e| e.into_inner());
            gate.queued_requests = gate.queued_requests.saturating_add(1);
            gate.queued_requests
        };

        {
            let mut mgr = self
                .shared
                .model_manager
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            mgr.record_runtime_queue_enqueued(queue_depth);
        }

        let mut gate = self.shared.gate.lock().unwrap_or_else(|e| e.into_inner());
        while gate.active_requests >= gate.max_concurrent_requests {
            gate = self
                .shared
                .gate_cv
                .wait(gate)
                .unwrap_or_else(|e| e.into_inner());
        }
        gate.queued_requests = gate.queued_requests.saturating_sub(1);
        gate.active_requests = gate.active_requests.saturating_add(1);

        RequestPermit {
            shared: Arc::clone(&self.shared),
            active: true,
        }
    }

    fn ensure_runner(&self, model_id: &str) -> Result<Arc<dyn LocalGenBackend>> {
        if model_id.trim().is_empty() {
            anyhow::bail!("model id cannot be empty");
        }

        if let Some(existing) = self
            .shared
            .runners
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(model_id)
            .cloned()
        {
            self.touch_runtime(model_id);
            return Ok(existing);
        }

        let loaded = self.shared.loader.load_backend(model_id)?;
        let runner = {
            let mut runners = self
                .shared
                .runners
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            runners
                .entry(model_id.to_string())
                .or_insert_with(|| Arc::clone(&loaded))
                .clone()
        };

        self.touch_runtime(model_id);
        Ok(runner)
    }

    fn touch_runtime(&self, model_id: &str) {
        let mut evicted = {
            let mut mgr = self
                .shared
                .model_manager
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let mut evicted = mgr.mark_runtime_used(model_id);
            evicted.extend(mgr.evict_idle_runtime_models());
            evicted
        };

        if !evicted.is_empty() {
            evicted.sort();
            evicted.dedup();
            self.remove_runners(&evicted);
        }
    }

    fn invalidate_runner(&self, model_id: &str) {
        self.remove_runners(&[model_id.to_string()]);
    }

    fn remove_runners(&self, model_ids: &[String]) {
        if model_ids.is_empty() {
            return;
        }

        let mut runners = self
            .shared
            .runners
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for model_id in model_ids {
            if let Some(runner) = runners.remove(model_id) {
                runner.cancel();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::MockGenerator;
    use crate::hardware::LocalModelRuntimePolicy;
    use std::thread;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    struct SleepGenerator {
        model_id: String,
        sleep_for: Duration,
    }

    impl SleepGenerator {
        fn new(model_id: String, sleep_for: Duration) -> Self {
            Self {
                model_id,
                sleep_for,
            }
        }
    }

    impl LocalGenBackend for SleepGenerator {
        fn generate(&self, _prompt: &str, _opts: &GenOpts) -> Result<String> {
            thread::sleep(self.sleep_for);
            Ok("ok".to_string())
        }

        fn model_id(&self) -> &str {
            &self.model_id
        }
    }

    #[test]
    fn scheduler_reuses_and_capacity_evicts_runners() {
        let dir = TempDir::new().unwrap();
        let manager = ModelManager::with_runtime_policy(
            dir.path().to_path_buf(),
            LocalModelRuntimePolicy {
                max_loaded_models: 1,
                keep_warm_secs: 300,
                aggressive_eviction: false,
            },
        );

        let loader: Arc<dyn BackendFactory> = Arc::new(|model_id: &str| {
            Ok(Arc::new(MockGenerator::new(format!("loaded:{model_id}")))
                as Arc<dyn LocalGenBackend>)
        });

        let scheduler = LocalRunnerLifecycleManager::new(manager, loader);
        scheduler.prewarm("model-a").unwrap();
        scheduler.prewarm("model-b").unwrap();

        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.loaded_runners, vec!["model-b".to_string()]);
        assert_eq!(snapshot.lifecycle.warm_models, vec!["model-b".to_string()]);
        assert_eq!(snapshot.lifecycle.metrics.total_slot_activations, 2);
        assert_eq!(snapshot.lifecycle.metrics.total_capacity_evictions, 1);
    }

    #[test]
    fn scheduler_tracks_queue_depth_when_saturated() {
        let dir = TempDir::new().unwrap();
        let manager = ModelManager::with_runtime_policy(
            dir.path().to_path_buf(),
            LocalModelRuntimePolicy {
                max_loaded_models: 1,
                keep_warm_secs: 300,
                aggressive_eviction: false,
            },
        );

        let loader: Arc<dyn BackendFactory> = Arc::new(|model_id: &str| {
            Ok(Arc::new(SleepGenerator::new(
                model_id.to_string(),
                Duration::from_millis(200),
            )) as Arc<dyn LocalGenBackend>)
        });

        let scheduler = LocalRunnerLifecycleManager::with_limits(manager, loader, 1);
        let opts = GenOpts::default();

        let s1 = scheduler.clone();
        let opts1 = opts.clone();
        let h1 = thread::spawn(move || s1.generate("model-a", "prompt-a", &opts1));

        thread::sleep(Duration::from_millis(30));

        let s2 = scheduler.clone();
        let opts2 = opts.clone();
        let h2 = thread::spawn(move || s2.generate("model-a", "prompt-b", &opts2));

        let start = Instant::now();
        let mut observed_queue = false;
        while start.elapsed() < Duration::from_millis(500) {
            if scheduler.snapshot().queued_requests > 0 {
                observed_queue = true;
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }

        assert!(observed_queue, "expected at least one queued request");
        assert!(h1.join().unwrap().is_ok());
        assert!(h2.join().unwrap().is_ok());

        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.active_requests, 0);
        assert_eq!(snapshot.queued_requests, 0);
        assert_eq!(snapshot.lifecycle.metrics.total_queue_enqueued, 2);
        assert_eq!(snapshot.lifecycle.metrics.total_queue_completed, 2);
        assert!(snapshot.lifecycle.metrics.max_observed_queue_depth >= 1);
    }

    #[test]
    fn scheduler_maintenance_evicts_idle_models() {
        let dir = TempDir::new().unwrap();
        let manager = ModelManager::with_runtime_policy(
            dir.path().to_path_buf(),
            LocalModelRuntimePolicy {
                max_loaded_models: 2,
                keep_warm_secs: 1,
                aggressive_eviction: false,
            },
        );

        let loader: Arc<dyn BackendFactory> = Arc::new(|model_id: &str| {
            Ok(Arc::new(MockGenerator::new(format!("loaded:{model_id}")))
                as Arc<dyn LocalGenBackend>)
        });

        let scheduler = LocalRunnerLifecycleManager::new(manager, loader);
        scheduler.prewarm("model-a").unwrap();

        let start = Instant::now();
        let mut evicted = Vec::new();
        while start.elapsed() < Duration::from_secs(4) {
            evicted = scheduler.maintenance_tick();
            if !evicted.is_empty() {
                break;
            }
            thread::sleep(Duration::from_millis(200));
        }

        assert_eq!(evicted, vec!["model-a".to_string()]);
        let snapshot = scheduler.snapshot();
        assert!(snapshot.loaded_runners.is_empty());
        assert!(snapshot.lifecycle.warm_models.is_empty());
        assert_eq!(snapshot.lifecycle.metrics.total_idle_evictions, 1);
    }
}
