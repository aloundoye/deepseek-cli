use anyhow::Result;
use chrono::Utc;
use codingbuddy_core::{EventEnvelope, TelemetryConfig, runtime_dir};
use reqwest::blocking::Client;
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub struct Observer {
    log_path: PathBuf,
    telemetry: Option<TelemetrySink>,
    verbose: bool,
}

struct TelemetrySink {
    endpoint: String,
    client: Client,
}

impl Observer {
    pub fn new(workspace: &Path, telemetry_cfg: &TelemetryConfig) -> Result<Self> {
        let dir = runtime_dir(workspace);
        fs::create_dir_all(&dir)?;
        let telemetry = telemetry_sink(telemetry_cfg)?;
        Ok(Self {
            log_path: dir.join("observe.log"),
            telemetry,
            verbose: false,
        })
    }

    pub fn record_event(&self, event: &EventEnvelope) -> Result<()> {
        self.append_log_line(&format!(
            "{} EVENT {}",
            Utc::now().to_rfc3339(),
            serde_json::to_string(event)?
        ))?;
        self.emit_telemetry(
            "telemetry.event",
            json!({
                "session_id": event.session_id,
                "seq_no": event.seq_no,
                "kind": event.kind,
            }),
        )
    }

    /// Enable or disable verbose logging to stderr.
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Returns whether verbose mode is enabled.
    pub fn is_verbose(&self) -> bool {
        self.verbose
    }

    /// Log a message to stderr with `[deepseek]` prefix when verbose mode is on.
    pub fn verbose_log(&self, msg: &str) {
        if self.verbose {
            eprintln!("[deepseek] {msg}");
        }
    }

    /// Log a warning — always written to log file, and to stderr.
    pub fn warn_log(&self, msg: &str) {
        eprintln!("[deepseek WARN] {msg}");
        let _ = self.append_log_line(&format!("{} WARN {msg}", Utc::now().to_rfc3339()));
    }

    fn append_log_line(&self, line: &str) -> Result<()> {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;
        writeln!(f, "{line}")?;
        Ok(())
    }

    fn emit_telemetry(&self, name: &str, payload: serde_json::Value) -> Result<()> {
        let Some(sink) = &self.telemetry else {
            return Ok(());
        };

        let body = json!({
            "name": name,
            "at": Utc::now().to_rfc3339(),
            "payload": payload,
        });

        // Fire-and-forget: send telemetry in a background thread so it never
        // blocks the agent/TUI thread (the HTTP call can take up to 3 seconds).
        let client = sink.client.clone();
        let endpoint = sink.endpoint.clone();
        let log_path = self.log_path.clone();
        std::thread::spawn(move || {
            if let Err(err) = client.post(&endpoint).json(&body).send() {
                let line = format!("{} TELEMETRY_ERROR error={}", Utc::now().to_rfc3339(), err);
                let _ = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&log_path)
                    .and_then(|mut f| writeln!(f, "{line}"));
            }
        });
        Ok(())
    }
}

fn telemetry_sink(cfg: &TelemetryConfig) -> Result<Option<TelemetrySink>> {
    if !cfg.enabled {
        return Ok(None);
    }
    let Some(endpoint) = cfg.endpoint.clone() else {
        return Ok(None);
    };
    let client = Client::builder().timeout(Duration::from_secs(3)).build()?;
    Ok(Some(TelemetrySink { endpoint, client }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use codingbuddy_core::{EventKind, TelemetryConfig};
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;
    use uuid::Uuid;

    fn sample_event() -> EventEnvelope {
        EventEnvelope {
            seq_no: 1,
            at: Utc::now(),
            session_id: Uuid::now_v7(),
            kind: EventKind::TurnAdded {
                role: "user".to_string(),
                content: "hello".to_string(),
            },
        }
    }

    #[test]
    fn telemetry_disabled_does_not_require_endpoint() {
        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-observe-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("create workspace");
        let observer = Observer::new(
            &workspace,
            &TelemetryConfig {
                enabled: false,
                endpoint: None,
            },
        )
        .expect("observer");
        observer
            .record_event(&sample_event())
            .expect("record event");
    }

    #[test]
    fn telemetry_posts_when_enabled() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = vec![0_u8; 8192];
            let n = stream.read(&mut buf).expect("read request");
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK");
            request
        });

        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-observe-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("create workspace");
        let observer = Observer::new(
            &workspace,
            &TelemetryConfig {
                enabled: true,
                endpoint: Some(format!("http://{addr}/collect")),
            },
        )
        .expect("observer");
        observer
            .record_event(&sample_event())
            .expect("record event");
        let request = server.join().expect("join server");
        assert!(request.contains("POST /collect"));
        assert!(request.contains("telemetry.event"));
    }

    // ── Log file tests ──

    #[test]
    fn record_event_writes_to_log_file() {
        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-observe-log-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("create workspace");
        let observer = Observer::new(
            &workspace,
            &TelemetryConfig {
                enabled: false,
                endpoint: None,
            },
        )
        .expect("observer");
        observer.record_event(&sample_event()).expect("record");

        let log_content = fs::read_to_string(&observer.log_path).expect("read log");
        assert!(log_content.contains("EVENT"));
        assert!(log_content.contains("TurnAdded"));
    }

    #[test]
    fn multiple_events_append_to_log() {
        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-observe-multi-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("create workspace");
        let observer = Observer::new(
            &workspace,
            &TelemetryConfig {
                enabled: false,
                endpoint: None,
            },
        )
        .expect("observer");
        observer.record_event(&sample_event()).expect("record 1");
        observer.record_event(&sample_event()).expect("record 2");

        let log_content = fs::read_to_string(&observer.log_path).expect("read log");
        let event_lines: Vec<&str> = log_content
            .lines()
            .filter(|l| l.contains("EVENT"))
            .collect();
        assert_eq!(event_lines.len(), 2);
    }

    // ── Verbose / warn logging tests ──

    #[test]
    fn verbose_mode_defaults_to_off() {
        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-observe-verbose-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("create workspace");
        let observer = Observer::new(
            &workspace,
            &TelemetryConfig {
                enabled: false,
                endpoint: None,
            },
        )
        .expect("observer");
        assert!(!observer.is_verbose());
    }

    #[test]
    fn set_verbose_toggles_mode() {
        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-observe-toggle-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("create workspace");
        let mut observer = Observer::new(
            &workspace,
            &TelemetryConfig {
                enabled: false,
                endpoint: None,
            },
        )
        .expect("observer");
        observer.set_verbose(true);
        assert!(observer.is_verbose());
        observer.set_verbose(false);
        assert!(!observer.is_verbose());
    }

    #[test]
    fn warn_log_writes_to_log_file() {
        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-observe-warn-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("create workspace");
        let observer = Observer::new(
            &workspace,
            &TelemetryConfig {
                enabled: false,
                endpoint: None,
            },
        )
        .expect("observer");
        observer.warn_log("something went wrong");

        let log_content = fs::read_to_string(&observer.log_path).expect("read log");
        assert!(log_content.contains("WARN"));
        assert!(log_content.contains("something went wrong"));
    }

    // ── Event serialization tests ──

    #[test]
    fn event_serializes_with_session_id() {
        let event = sample_event();
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(json.contains(&event.session_id.to_string()));
    }

    #[test]
    fn telemetry_sink_requires_endpoint_when_enabled() {
        let sink = telemetry_sink(&TelemetryConfig {
            enabled: true,
            endpoint: None,
        })
        .expect("sink");
        assert!(sink.is_none(), "no endpoint → no sink even when enabled");
    }

    #[test]
    fn telemetry_sink_none_when_disabled() {
        let sink = telemetry_sink(&TelemetryConfig {
            enabled: false,
            endpoint: Some("http://example.com".to_string()),
        })
        .expect("sink");
        assert!(sink.is_none());
    }
}
