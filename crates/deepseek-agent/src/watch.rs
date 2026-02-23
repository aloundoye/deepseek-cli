//! Watch continuous mode: filesystem watcher that detects TODO(ai)/FIXME(ai)/AI:
//! comments and sends digest updates via channel.

use notify::{Config as NotifyConfig, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Digest of watch comment state at a point in time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchDigest {
    pub digest: u64,
    pub comment_count: usize,
    pub hints: String,
}

/// Scans a workspace for TODO(ai)/FIXME(ai)/AI: comments using ripgrep.
/// Returns None if no comments found or rg not available.
pub fn watch_scan(workspace: &Path) -> Option<WatchDigest> {
    let output = std::process::Command::new("rg")
        .args([
            "-n",
            "--hidden",
            "--glob",
            "!.git/**",
            "--glob",
            "!target/**",
            "--glob",
            "!.deepseek/**",
            "--ignore-file",
            ".deepseekignore",
            "TODO\\(ai\\)|FIXME\\(ai\\)|AI:",
            ".",
        ])
        .current_dir(workspace)
        .output()
        .ok()?;

    let text = String::from_utf8_lossy(&output.stdout);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let comment_count = trimmed.lines().count();
    let mut hasher = DefaultHasher::new();
    trimmed.hash(&mut hasher);
    let digest = hasher.finish();
    Some(WatchDigest {
        digest,
        comment_count,
        hints: trimmed.to_string(),
    })
}

/// Computes the diff between two digests.
/// Returns true if the comments changed.
pub fn watch_diff(old: Option<&WatchDigest>, new: Option<&WatchDigest>) -> bool {
    match (old, new) {
        (None, None) => false,
        (Some(_), None) | (None, Some(_)) => true,
        (Some(a), Some(b)) => a.digest != b.digest,
    }
}

/// Filesystem watcher daemon for continuous watch mode.
/// Uses `notify` to watch for file changes and debounces scan events.
pub struct WatchDaemon {
    _watcher: RecommendedWatcher,
    rx: mpsc::Receiver<WatchDigest>,
}

impl WatchDaemon {
    /// Start watching `workspace` for file changes.
    /// Sends `WatchDigest` updates whenever the comment state changes.
    pub fn start(workspace: &Path) -> anyhow::Result<Self> {
        let (tx, rx) = mpsc::channel();
        let workspace_path = workspace.to_path_buf();

        let (notify_tx, notify_rx) = std::sync::mpsc::channel();
        let mut watcher = RecommendedWatcher::new(notify_tx, NotifyConfig::default())?;
        watcher.watch(&workspace_path, RecursiveMode::Recursive)?;

        // Background thread: debounce filesystem events and scan for comments
        let scan_tx = tx;
        std::thread::spawn(move || {
            debounce_loop(&workspace_path, notify_rx, scan_tx);
        });

        Ok(Self {
            _watcher: watcher,
            rx,
        })
    }

    /// Non-blocking receive of the next digest update.
    pub fn try_recv(&self) -> Option<WatchDigest> {
        self.rx.try_recv().ok()
    }

    /// Blocking receive with timeout.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<WatchDigest> {
        self.rx.recv_timeout(timeout).ok()
    }
}

fn debounce_loop(
    workspace: &Path,
    notify_rx: mpsc::Receiver<Result<notify::Event, notify::Error>>,
    scan_tx: mpsc::Sender<WatchDigest>,
) {
    let debounce_duration = Duration::from_millis(500);
    let mut last_scan = Instant::now() - debounce_duration;
    let mut last_digest: Option<u64> = None;

    loop {
        // Wait for any filesystem event
        match notify_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(_event)) => {
                // Debounce: skip if we scanned recently
                if last_scan.elapsed() < debounce_duration {
                    // Drain remaining events within debounce window
                    while notify_rx.try_recv().is_ok() {}
                    std::thread::sleep(debounce_duration.saturating_sub(last_scan.elapsed()));
                }
            }
            Ok(Err(_)) | Err(mpsc::RecvTimeoutError::Timeout) => {
                // On error or timeout, do a periodic scan anyway
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        // Drain any accumulated events
        while notify_rx.try_recv().is_ok() {}

        last_scan = Instant::now();

        if let Some(digest) = watch_scan(workspace) {
            if last_digest != Some(digest.digest) {
                last_digest = Some(digest.digest);
                if scan_tx.send(digest).is_err() {
                    break; // receiver dropped
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watch_scan_empty_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let result = watch_scan(dir.path());
        assert!(result.is_none(), "empty workspace should have no comments");
    }

    #[test]
    fn watch_scan_finds_todo_ai() {
        // Skip if rg is not available on PATH
        let rg_available = std::process::Command::new("rg")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success());
        if !rg_available {
            eprintln!("skipping: rg not found on PATH");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("main.rs"),
            "fn main() {\n    // TODO(ai): implement this\n}\n",
        )
        .unwrap();
        let result = watch_scan(dir.path());
        assert!(result.is_some(), "should find TODO(ai) comment");
        let digest = result.unwrap();
        assert_eq!(digest.comment_count, 1);
        assert!(digest.hints.contains("TODO(ai)"));
    }

    #[test]
    fn watch_diff_detects_change() {
        let a = WatchDigest {
            digest: 123,
            comment_count: 1,
            hints: "a".to_string(),
        };
        let b = WatchDigest {
            digest: 456,
            comment_count: 2,
            hints: "b".to_string(),
        };
        assert!(!watch_diff(Some(&a), Some(&a)));
        assert!(watch_diff(Some(&a), Some(&b)));
        assert!(watch_diff(None, Some(&a)));
        assert!(watch_diff(Some(&a), None));
        assert!(!watch_diff(None, None));
    }
}
