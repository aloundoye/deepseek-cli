use crate::*;

impl AgentEngine {
    pub(crate) fn complete_with_cache(
        &self,
        session_id: Uuid,
        req: &LlmRequest,
    ) -> Result<deepseek_core::LlmResponse> {
        if self.cfg.scheduling.off_peak {
            let hour = Utc::now().hour() as u8;
            let start = self.cfg.scheduling.off_peak_start_hour;
            let end = self.cfg.scheduling.off_peak_end_hour;
            let in_window = in_off_peak_window(hour, start, end);
            if !in_window {
                let resume_after = next_off_peak_start(start);
                let mut reason = "outside_off_peak_window".to_string();
                if req.non_urgent && self.cfg.scheduling.defer_non_urgent {
                    let delay = seconds_until_off_peak_start(hour, start);
                    let capped_delay = if self.cfg.scheduling.max_defer_seconds == 0 {
                        delay
                    } else {
                        delay.min(self.cfg.scheduling.max_defer_seconds)
                    };
                    if capped_delay > 0 {
                        reason = format!("outside_off_peak_window_deferred_{}s", capped_delay);
                        thread::sleep(std::time::Duration::from_secs(capped_delay));
                    }
                }
                self.emit(
                    session_id,
                    EventKind::OffPeakScheduledV1 {
                        reason,
                        resume_after,
                    },
                )?;
            }
        }

        let cache_key = prompt_cache_key(&req.model, &req.prompt);
        if self.cfg.llm.prompt_cache_enabled
            && let Some(cached) = self.read_prompt_cache(&cache_key)?
        {
            self.store.insert_provider_metric(&ProviderMetricRecord {
                provider: "deepseek".to_string(),
                model: req.model.clone(),
                cache_key: Some(cache_key.clone()),
                cache_hit: true,
                latency_ms: 0,
                recorded_at: Utc::now().to_rfc3339(),
            })?;
            self.emit(
                session_id,
                EventKind::PromptCacheHitV1 {
                    cache_key,
                    model: req.model.clone(),
                },
            )?;
            return Ok(cached);
        }

        let started = Instant::now();
        let response = {
            let cb = self.stream_callback.lock().ok().and_then(|g| g.clone());
            if let Some(cb) = cb {
                self.llm.complete_streaming(req, cb)
            } else {
                self.llm.complete(req)
            }
        }?;
        let latency_ms = started.elapsed().as_millis() as u64;
        self.store.insert_provider_metric(&ProviderMetricRecord {
            provider: "deepseek".to_string(),
            model: req.model.clone(),
            cache_key: Some(cache_key.clone()),
            cache_hit: false,
            latency_ms,
            recorded_at: Utc::now().to_rfc3339(),
        })?;

        if self.cfg.llm.prompt_cache_enabled {
            self.write_prompt_cache(&cache_key, &response)?;
        }

        Ok(response)
    }

    pub(crate) fn prompt_cache_dir(&self) -> PathBuf {
        deepseek_core::runtime_dir(&self.workspace).join("prompt-cache")
    }

    pub(crate) fn write_prompt_cache(
        &self,
        cache_key: &str,
        response: &deepseek_core::LlmResponse,
    ) -> Result<()> {
        let dir = self.prompt_cache_dir();
        fs::create_dir_all(&dir)?;
        fs::write(
            dir.join(format!("{cache_key}.json")),
            serde_json::to_vec(response)?,
        )?;
        Ok(())
    }

    pub(crate) fn read_prompt_cache(
        &self,
        cache_key: &str,
    ) -> Result<Option<deepseek_core::LlmResponse>> {
        let path = self.prompt_cache_dir().join(format!("{cache_key}.json"));
        if !path.exists() {
            return Ok(None);
        }
        let raw = fs::read_to_string(path)?;
        Ok(Some(serde_json::from_str(&raw)?))
    }
}

pub(crate) fn seconds_until_off_peak_start(current_hour: u8, start_hour: u8) -> u64 {
    let now_minutes = (current_hour as u64) * 60;
    let start_minutes = (start_hour as u64) * 60;
    let minutes_until = if now_minutes <= start_minutes {
        start_minutes - now_minutes
    } else {
        (24 * 60 - now_minutes) + start_minutes
    };
    minutes_until * 60
}

pub(crate) fn prompt_cache_key(model: &str, prompt: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"deepseek:");
    hasher.update(model.as_bytes());
    hasher.update(b":");
    hasher.update(prompt.as_bytes());
    format!("{:x}", hasher.finalize())
}

pub(crate) fn next_off_peak_start(start_hour: u8) -> String {
    let now = Utc::now();
    let current = now.hour() as u8;
    let mut day_delta = 0_i64;
    if current >= start_hour {
        day_delta = 1;
    }
    let date = now.date_naive() + chrono::Duration::days(day_delta);
    if let Some(dt) = date.and_hms_opt(start_hour as u32, 0, 0) {
        return chrono::DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc).to_rfc3339();
    }
    now.to_rfc3339()
}

pub(crate) fn in_off_peak_window(hour: u8, start: u8, end: u8) -> bool {
    if start <= end {
        hour >= start && hour < end
    } else {
        hour >= start || hour < end
    }
}
