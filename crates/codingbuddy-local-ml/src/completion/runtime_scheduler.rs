use super::{GenOpts, LocalGenBackend};
use crate::ModelManager;
use crate::hardware;
use crate::model_manager::{
    RunnerLifecycleState, RuntimeLifecycleSnapshot, RuntimeSchedulerLiveSnapshot,
};
use crate::model_registry;
use anyhow::{Result, anyhow};
use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

const RECOVERY_BACKOFF_MS: u64 = 250;

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
    max_queue_depth: usize,
    max_queue_wait_ms: u64,
}

#[derive(Debug, Clone, Default)]
struct LoadLaneState {
    active_model: Option<String>,
    waiting_requests: usize,
}

/// Observability snapshot for the local runtime scheduler.
#[derive(Debug, Clone)]
pub struct LocalRuntimeSchedulerSnapshot {
    pub active_requests: usize,
    pub queued_requests: usize,
    pub max_concurrent_requests: usize,
    pub max_queue_depth: usize,
    pub max_queue_wait_ms: u64,
    pub loading_model: Option<String>,
    pub loading_waiters: usize,
    pub loaded_runners: Vec<String>,
    pub lifecycle: RuntimeLifecycleSnapshot,
}

struct SharedState {
    model_manager: Mutex<ModelManager>,
    runners: Mutex<BTreeMap<String, Arc<dyn LocalGenBackend>>>,
    gate: Mutex<RequestGateState>,
    gate_cv: Condvar,
    load_lane: Mutex<LoadLaneState>,
    load_lane_cv: Condvar,
    loader: Arc<dyn BackendFactory>,
    memory_probe: Arc<dyn Fn() -> u64 + Send + Sync>,
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

struct LoadLanePermit {
    shared: Arc<SharedState>,
    model_id: String,
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
        drop(mgr);
        let scheduler = LocalRunnerLifecycleManager {
            shared: Arc::clone(&self.shared),
        };
        scheduler.refresh_live_snapshot();
    }
}

impl Drop for LoadLanePermit {
    fn drop(&mut self) {
        let mut lane = self
            .shared
            .load_lane
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if lane.active_model.as_deref() == Some(self.model_id.as_str()) {
            lane.active_model = None;
            self.shared.load_lane_cv.notify_all();
        }
        drop(lane);
        let scheduler = LocalRunnerLifecycleManager {
            shared: Arc::clone(&self.shared),
        };
        scheduler.refresh_live_snapshot();
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
        let max_concurrent_requests = max_concurrent_requests.max(1);
        let default_queue_depth = max_concurrent_requests.saturating_mul(4).max(1);
        Self::with_limits_and_queue(
            model_manager,
            loader,
            max_concurrent_requests,
            default_queue_depth,
        )
    }

    /// Build a lifecycle manager with explicit concurrent and queue limits.
    pub fn with_limits_and_queue(
        model_manager: ModelManager,
        loader: Arc<dyn BackendFactory>,
        max_concurrent_requests: usize,
        max_queue_depth: usize,
    ) -> Self {
        Self::with_limits_queue_and_wait(
            model_manager,
            loader,
            max_concurrent_requests,
            max_queue_depth,
            45_000,
        )
    }

    /// Build a lifecycle manager with explicit queue wait timeout.
    pub fn with_limits_queue_and_wait(
        model_manager: ModelManager,
        loader: Arc<dyn BackendFactory>,
        max_concurrent_requests: usize,
        max_queue_depth: usize,
        max_queue_wait_ms: u64,
    ) -> Self {
        Self::with_limits_queue_wait_and_memory_probe(
            model_manager,
            loader,
            max_concurrent_requests,
            max_queue_depth,
            max_queue_wait_ms,
            Arc::new(hardware::available_memory_mb),
        )
    }

    fn with_limits_queue_wait_and_memory_probe(
        model_manager: ModelManager,
        loader: Arc<dyn BackendFactory>,
        max_concurrent_requests: usize,
        max_queue_depth: usize,
        max_queue_wait_ms: u64,
        memory_probe: Arc<dyn Fn() -> u64 + Send + Sync>,
    ) -> Self {
        let max_concurrent_requests = max_concurrent_requests.max(1);
        let max_queue_depth = max_queue_depth.max(1);
        let max_queue_wait_ms = max_queue_wait_ms.max(1);
        let mut model_manager = model_manager;
        model_manager.record_runtime_scheduler_policy(
            max_concurrent_requests,
            max_queue_depth,
            max_queue_wait_ms,
        );
        Self {
            shared: Arc::new(SharedState {
                model_manager: Mutex::new(model_manager),
                runners: Mutex::new(BTreeMap::new()),
                gate: Mutex::new(RequestGateState {
                    active_requests: 0,
                    queued_requests: 0,
                    max_concurrent_requests,
                    max_queue_depth,
                    max_queue_wait_ms,
                }),
                gate_cv: Condvar::new(),
                load_lane: Mutex::new(LoadLaneState::default()),
                load_lane_cv: Condvar::new(),
                loader,
                memory_probe,
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
        let _permit = self.acquire_request_permit()?;
        let _ = self.maintenance_tick();

        let backend = self.ensure_runner(model_id)?;
        self.set_runner_state(model_id, RunnerLifecycleState::Busy, None);
        match backend.generate(prompt, opts) {
            Ok(output) => {
                self.set_runner_state(model_id, RunnerLifecycleState::Warm, None);
                Ok(output)
            }
            Err(first_error) => {
                let first_detail = first_error.to_string();
                {
                    let mut mgr = self
                        .shared
                        .model_manager
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    mgr.record_runtime_runner_reload(model_id, &first_detail);
                }
                self.set_runner_state(
                    model_id,
                    RunnerLifecycleState::Reloading,
                    Some(&first_detail),
                );
                self.invalidate_runner(model_id);
                let reloaded = self.ensure_runner(model_id)?;
                self.set_runner_state(model_id, RunnerLifecycleState::Busy, None);
                let retry_result = reloaded.generate(prompt, opts);
                if retry_result.is_ok() {
                    self.set_runner_state(model_id, RunnerLifecycleState::Warm, None);
                }
                retry_result.map_err(|retry_error| {
                    let retry_detail = retry_error.to_string();
                    {
                        let mut mgr = self
                            .shared
                            .model_manager
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        mgr.record_runtime_runner_load_failure(model_id, &retry_detail);
                    }
                    self.set_runner_state(
                        model_id,
                        RunnerLifecycleState::Failed,
                        Some(&retry_detail),
                    );
                    anyhow!(
                        "generation failed for model '{model_id}': {first_detail}; reload retry failed: {retry_detail}"
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
        let load_lane = self
            .shared
            .load_lane
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
            max_queue_depth: gate.max_queue_depth,
            max_queue_wait_ms: gate.max_queue_wait_ms,
            loading_model: load_lane.active_model.clone(),
            loading_waiters: load_lane.waiting_requests,
            loaded_runners,
            lifecycle,
        }
    }

    fn refresh_live_snapshot(&self) {
        let gate = self.shared.gate.lock().unwrap_or_else(|e| e.into_inner());
        let load_lane = self
            .shared
            .load_lane
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let active_requests = gate.active_requests;
        let queued_requests = gate.queued_requests;
        let loading_model = load_lane.active_model.clone();
        let loading_waiters = load_lane.waiting_requests;
        let loaded_runners = self
            .shared
            .runners
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        drop(gate);
        drop(load_lane);
        let mut mgr = self
            .shared
            .model_manager
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        mgr.record_runtime_live_snapshot(RuntimeSchedulerLiveSnapshot {
            active_requests,
            queued_requests,
            loading_model,
            loading_waiters,
            loaded_runners,
            updated_at_epoch_secs: 0,
        });
    }

    fn set_runner_state(&self, model_id: &str, state: RunnerLifecycleState, detail: Option<&str>) {
        let mut mgr = self
            .shared
            .model_manager
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        mgr.set_runner_state(model_id, state, detail);
    }

    fn clear_runner_state(&self, model_id: &str) {
        let mut mgr = self
            .shared
            .model_manager
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        mgr.clear_runner_state(model_id);
    }

    fn acquire_request_permit(&self) -> Result<RequestPermit> {
        let queue_depth = {
            let mut gate = self.shared.gate.lock().unwrap_or_else(|e| e.into_inner());
            let inflight = gate.active_requests.saturating_add(gate.queued_requests);
            let max_inflight = gate
                .max_concurrent_requests
                .saturating_add(gate.max_queue_depth);
            if inflight >= max_inflight {
                let active = gate.active_requests;
                let queued = gate.queued_requests;
                let max_concurrent = gate.max_concurrent_requests;
                let max_queue = gate.max_queue_depth;
                drop(gate);
                let mut mgr = self
                    .shared
                    .model_manager
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                mgr.record_runtime_queue_rejected(active, queued, max_concurrent, max_queue);
                drop(mgr);
                self.refresh_live_snapshot();
                anyhow::bail!(
                    "local runtime queue is full (active={active}, queued={queued}, max_concurrent={max_concurrent}, max_queue_depth={max_queue})"
                );
            }
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
        self.refresh_live_snapshot();

        let mut gate = self.shared.gate.lock().unwrap_or_else(|e| e.into_inner());
        let wait_started = Instant::now();
        let max_wait = Duration::from_millis(gate.max_queue_wait_ms.max(1));
        while gate.active_requests >= gate.max_concurrent_requests {
            let elapsed = wait_started.elapsed();
            if elapsed >= max_wait {
                gate.queued_requests = gate.queued_requests.saturating_sub(1);
                let active = gate.active_requests;
                let queued = gate.queued_requests;
                let max_concurrent = gate.max_concurrent_requests;
                let max_queue = gate.max_queue_depth;
                self.shared.gate_cv.notify_one();
                drop(gate);
                let mut mgr = self
                    .shared
                    .model_manager
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                mgr.record_runtime_queue_wait_timeout(
                    elapsed.as_millis() as u64,
                    active,
                    queued,
                    max_concurrent,
                    max_queue,
                );
                anyhow::bail!(
                    "local runtime queue wait timed out after {} ms (active={active}, queued={queued}, max_concurrent={max_concurrent}, max_queue_depth={max_queue})",
                    elapsed.as_millis()
                );
            }
            let remaining = max_wait.saturating_sub(elapsed);
            let (next_gate, _) = self
                .shared
                .gate_cv
                .wait_timeout(gate, remaining)
                .unwrap_or_else(|e| e.into_inner());
            gate = next_gate;
        }
        gate.queued_requests = gate.queued_requests.saturating_sub(1);
        gate.active_requests = gate.active_requests.saturating_add(1);
        drop(gate);
        self.refresh_live_snapshot();

        Ok(RequestPermit {
            shared: Arc::clone(&self.shared),
            active: true,
        })
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
            self.set_runner_state(model_id, RunnerLifecycleState::Warm, None);
            return Ok(existing);
        }

        let _load_lane = self.acquire_load_lane(model_id);

        if let Some(existing) = self
            .shared
            .runners
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(model_id)
            .cloned()
        {
            self.touch_runtime(model_id);
            self.set_runner_state(model_id, RunnerLifecycleState::Warm, None);
            return Ok(existing);
        }

        let mut available_mb = (self.shared.memory_probe)();
        let mut eviction_attempt = 0usize;
        loop {
            match model_registry::check_model_fits(model_id, available_mb) {
                Ok(()) => break,
                Err(reason) => {
                    let victim = {
                        let mut mgr = self
                            .shared
                            .model_manager
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        eviction_attempt = eviction_attempt.saturating_add(1);
                        mgr.evict_one_runtime_model_for_memory_pressure(
                            model_id,
                            available_mb,
                            eviction_attempt,
                        )
                    };
                    if let Some(evicted_model_id) = victim {
                        self.remove_runners(&[evicted_model_id]);
                        std::thread::sleep(Duration::from_millis(RECOVERY_BACKOFF_MS));
                        available_mb = (self.shared.memory_probe)();
                        continue;
                    }
                    let mut mgr = self
                        .shared
                        .model_manager
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    mgr.record_runtime_memory_admission_denied(model_id, available_mb, &reason);
                    anyhow::bail!("runtime admission denied for model '{model_id}': {reason}");
                }
            }
        }

        let current_state = self
            .shared
            .model_manager
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .runner_state(model_id);
        if current_state != Some(RunnerLifecycleState::Reloading) {
            self.set_runner_state(model_id, RunnerLifecycleState::Loading, None);
        }
        let loaded = match self.shared.loader.load_backend(model_id) {
            Ok(backend) => backend,
            Err(err) => {
                let detail = err.to_string();
                let mut mgr = self
                    .shared
                    .model_manager
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                mgr.record_runtime_runner_load_failure(model_id, &detail);
                drop(mgr);
                self.set_runner_state(model_id, RunnerLifecycleState::Failed, Some(&detail));
                return Err(err);
            }
        };
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
        self.set_runner_state(model_id, RunnerLifecycleState::Warm, None);
        Ok(runner)
    }

    fn acquire_load_lane(&self, model_id: &str) -> LoadLanePermit {
        let mut lane = self
            .shared
            .load_lane
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut recorded_wait = false;
        let mut blocked_by: Option<String> = None;

        loop {
            if lane.active_model.is_none() {
                if recorded_wait {
                    lane.waiting_requests = lane.waiting_requests.saturating_sub(1);
                }
                lane.active_model = Some(model_id.to_string());
                drop(lane);
                self.refresh_live_snapshot();

                if let Some(blocker) = blocked_by {
                    let mut mgr = self
                        .shared
                        .model_manager
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    mgr.record_runtime_runner_load_wait(model_id, &blocker);
                }

                return LoadLanePermit {
                    shared: Arc::clone(&self.shared),
                    model_id: model_id.to_string(),
                };
            }

            blocked_by = lane.active_model.clone();
            if !recorded_wait {
                lane.waiting_requests = lane.waiting_requests.saturating_add(1);
                recorded_wait = true;
                drop(lane);
                self.refresh_live_snapshot();
                lane = self
                    .shared
                    .load_lane
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
            }
            lane = self
                .shared
                .load_lane_cv
                .wait(lane)
                .unwrap_or_else(|e| e.into_inner());
        }
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
        self.remove_runners_preserving_state(&[model_id.to_string()]);
    }

    fn remove_runners(&self, model_ids: &[String]) {
        self.remove_runners_inner(model_ids, false);
    }

    fn remove_runners_preserving_state(&self, model_ids: &[String]) {
        self.remove_runners_inner(model_ids, true);
    }

    fn remove_runners_inner(&self, model_ids: &[String], preserve_state: bool) {
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
        drop(runners);
        if !preserve_state {
            for model_id in model_ids {
                self.clear_runner_state(model_id);
            }
        }
        self.refresh_live_snapshot();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::MockGenerator;
    use crate::hardware::LocalModelRuntimePolicy;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::thread;
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

    struct CancelSignalGenerator {
        model_id: String,
        on_cancel: Arc<dyn Fn() + Send + Sync>,
    }

    impl CancelSignalGenerator {
        fn new(model_id: String, on_cancel: Arc<dyn Fn() + Send + Sync>) -> Self {
            Self {
                model_id,
                on_cancel,
            }
        }
    }

    impl LocalGenBackend for CancelSignalGenerator {
        fn generate(&self, _prompt: &str, _opts: &GenOpts) -> Result<String> {
            Ok("ok".to_string())
        }

        fn model_id(&self) -> &str {
            &self.model_id
        }

        fn cancel(&self) {
            (self.on_cancel)();
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

        let scheduler = LocalRunnerLifecycleManager::with_limits_and_queue(manager, loader, 1, 2);
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
    fn scheduler_rejects_requests_when_queue_ceiling_reached() {
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
                Duration::from_millis(220),
            )) as Arc<dyn LocalGenBackend>)
        });

        let scheduler = LocalRunnerLifecycleManager::with_limits_and_queue(manager, loader, 1, 1);
        let opts = GenOpts::default();

        let s1 = scheduler.clone();
        let opts1 = opts.clone();
        let h1 = thread::spawn(move || s1.generate("model-a", "prompt-a", &opts1));

        thread::sleep(Duration::from_millis(20));

        let s2 = scheduler.clone();
        let opts2 = opts.clone();
        let h2 = thread::spawn(move || s2.generate("model-a", "prompt-b", &opts2));

        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(300) {
            if scheduler.snapshot().queued_requests >= 1 {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }

        let err = scheduler
            .generate("model-a", "prompt-c", &opts)
            .expect_err("third request should be rejected by queue ceiling");
        assert!(
            err.to_string().contains("queue is full"),
            "unexpected rejection message: {err}"
        );

        assert!(h1.join().unwrap().is_ok());
        assert!(h2.join().unwrap().is_ok());

        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.max_queue_depth, 1);
        assert_eq!(snapshot.lifecycle.metrics.total_queue_rejected, 1);
    }

    #[test]
    fn scheduler_rejects_memory_admission_with_low_probe() {
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

        let scheduler = LocalRunnerLifecycleManager::with_limits_queue_wait_and_memory_probe(
            manager,
            loader,
            1,
            2,
            45_000,
            Arc::new(|| 1024),
        );
        let err = scheduler
            .generate("qwen2.5-coder-7b", "prompt", &GenOpts::default())
            .expect_err("expected memory admission rejection");
        assert!(
            err.to_string().contains("runtime admission denied"),
            "unexpected error: {err}"
        );

        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.lifecycle.metrics.total_memory_admission_denied, 1);
    }

    #[test]
    fn scheduler_queue_wait_timeout_is_tracked() {
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
                Duration::from_millis(250),
            )) as Arc<dyn LocalGenBackend>)
        });

        let scheduler =
            LocalRunnerLifecycleManager::with_limits_queue_and_wait(manager, loader, 1, 2, 40);
        let opts = GenOpts::default();

        let s1 = scheduler.clone();
        let opts1 = opts.clone();
        let h1 = thread::spawn(move || s1.generate("model-a", "prompt-a", &opts1));
        thread::sleep(Duration::from_millis(15));

        let err = scheduler
            .generate("model-a", "prompt-b", &opts)
            .expect_err("queued request should time out");
        assert!(
            err.to_string().contains("queue wait timed out"),
            "unexpected timeout error: {err}"
        );
        assert!(h1.join().unwrap().is_ok());

        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.lifecycle.metrics.total_queue_wait_timeouts, 1);
        assert_eq!(snapshot.max_queue_wait_ms, 40);
    }

    #[test]
    fn scheduler_memory_pressure_evicts_and_retries_admission() {
        let dir = TempDir::new().unwrap();
        let manager = ModelManager::with_runtime_policy(
            dir.path().to_path_buf(),
            LocalModelRuntimePolicy {
                max_loaded_models: 2,
                keep_warm_secs: 300,
                aggressive_eviction: false,
            },
        );
        let available_mb = Arc::new(AtomicU64::new(1024));
        let available_mb_on_cancel = Arc::clone(&available_mb);
        let loader: Arc<dyn BackendFactory> = Arc::new(move |model_id: &str| {
            if model_id == "model-a" {
                let on_cancel: Arc<dyn Fn() + Send + Sync> = Arc::new({
                    let available_mb = Arc::clone(&available_mb_on_cancel);
                    move || {
                        available_mb.store(20_000, Ordering::SeqCst);
                    }
                });
                Ok(
                    Arc::new(CancelSignalGenerator::new(model_id.to_string(), on_cancel))
                        as Arc<dyn LocalGenBackend>,
                )
            } else {
                Ok(Arc::new(MockGenerator::new(format!("loaded:{model_id}")))
                    as Arc<dyn LocalGenBackend>)
            }
        });

        let scheduler = LocalRunnerLifecycleManager::with_limits_queue_wait_and_memory_probe(
            manager,
            loader,
            1,
            2,
            45_000,
            Arc::new({
                let available_mb = Arc::clone(&available_mb);
                move || available_mb.load(Ordering::SeqCst)
            }),
        );
        scheduler.prewarm("model-a").unwrap();
        let out = scheduler
            .generate("qwen2.5-coder-7b", "prompt", &GenOpts::default())
            .expect("memory pressure relief should allow admission");
        assert_eq!(out, "loaded:qwen2.5-coder-7b");

        let snapshot = scheduler.snapshot();
        assert_eq!(
            snapshot.lifecycle.metrics.total_memory_pressure_evictions,
            1
        );
        assert_eq!(snapshot.lifecycle.metrics.total_memory_admission_denied, 0);
        assert!(
            snapshot
                .loaded_runners
                .iter()
                .any(|model| model == "qwen2.5-coder-7b")
        );
    }

    #[test]
    fn scheduler_serializes_runner_loads_for_different_models() {
        let dir = TempDir::new().unwrap();
        let manager = ModelManager::with_runtime_policy(
            dir.path().to_path_buf(),
            LocalModelRuntimePolicy {
                max_loaded_models: 2,
                keep_warm_secs: 300,
                aggressive_eviction: false,
            },
        );
        let active_loads = Arc::new(AtomicUsize::new(0));
        let max_active_loads = Arc::new(AtomicUsize::new(0));
        let loader: Arc<dyn BackendFactory> = Arc::new({
            let active_loads = Arc::clone(&active_loads);
            let max_active_loads = Arc::clone(&max_active_loads);
            move |model_id: &str| {
                let current = active_loads.fetch_add(1, Ordering::SeqCst) + 1;
                max_active_loads.fetch_max(current, Ordering::SeqCst);
                thread::sleep(Duration::from_millis(120));
                active_loads.fetch_sub(1, Ordering::SeqCst);
                Ok(Arc::new(MockGenerator::new(format!("loaded:{model_id}")))
                    as Arc<dyn LocalGenBackend>)
            }
        });

        let scheduler = LocalRunnerLifecycleManager::with_limits_and_queue(manager, loader, 2, 2);
        let opts = GenOpts::default();

        let s1 = scheduler.clone();
        let opts1 = opts.clone();
        let h1 = thread::spawn(move || s1.generate("model-a", "prompt-a", &opts1));

        let load_start = Instant::now();
        while load_start.elapsed() < Duration::from_millis(500) {
            if scheduler.snapshot().loading_model.as_deref() == Some("model-a") {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }

        let s2 = scheduler.clone();
        let opts2 = opts.clone();
        let h2 = thread::spawn(move || s2.generate("model-b", "prompt-b", &opts2));

        let wait_start = Instant::now();
        let mut observed_waiter = false;
        while wait_start.elapsed() < Duration::from_millis(500) {
            let snapshot = scheduler.snapshot();
            if snapshot.loading_waiters > 0 {
                observed_waiter = true;
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }

        assert!(
            observed_waiter,
            "expected second request to wait on load lane"
        );
        assert!(h1.join().unwrap().is_ok());
        assert!(h2.join().unwrap().is_ok());
        assert_eq!(max_active_loads.load(Ordering::SeqCst), 1);

        let snapshot = scheduler.snapshot();
        assert!(snapshot.loading_model.is_none());
        assert_eq!(snapshot.loading_waiters, 0);
        assert_eq!(snapshot.lifecycle.metrics.total_runner_load_waits, 1);
    }

    #[test]
    fn scheduler_deduplicates_same_model_loads_while_loading() {
        let dir = TempDir::new().unwrap();
        let manager = ModelManager::with_runtime_policy(
            dir.path().to_path_buf(),
            LocalModelRuntimePolicy {
                max_loaded_models: 2,
                keep_warm_secs: 300,
                aggressive_eviction: false,
            },
        );
        let load_count = Arc::new(AtomicUsize::new(0));
        let loader: Arc<dyn BackendFactory> = Arc::new({
            let load_count = Arc::clone(&load_count);
            move |model_id: &str| {
                load_count.fetch_add(1, Ordering::SeqCst);
                thread::sleep(Duration::from_millis(120));
                Ok(Arc::new(MockGenerator::new(format!("loaded:{model_id}")))
                    as Arc<dyn LocalGenBackend>)
            }
        });

        let scheduler = LocalRunnerLifecycleManager::with_limits_and_queue(manager, loader, 2, 2);
        let opts = GenOpts::default();

        let s1 = scheduler.clone();
        let opts1 = opts.clone();
        let h1 = thread::spawn(move || s1.generate("model-a", "prompt-a", &opts1));

        let load_start = Instant::now();
        while load_start.elapsed() < Duration::from_millis(500) {
            if scheduler.snapshot().loading_model.as_deref() == Some("model-a") {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }

        let s2 = scheduler.clone();
        let opts2 = opts.clone();
        let h2 = thread::spawn(move || s2.generate("model-a", "prompt-b", &opts2));

        assert!(h1.join().unwrap().is_ok());
        assert!(h2.join().unwrap().is_ok());
        assert_eq!(load_count.load(Ordering::SeqCst), 1);
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
