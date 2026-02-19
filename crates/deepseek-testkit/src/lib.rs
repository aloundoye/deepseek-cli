use anyhow::Result;
use deepseek_agent::AgentEngine;
use std::path::Path;

pub fn run_replay_smoke(workspace: &Path) -> Result<String> {
    let engine = AgentEngine::new(workspace)?;
    let _ = engine.run_once("replay test", false)?;
    engine.resume()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[test]
    fn replay_smoke() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let workspace = std::env::temp_dir().join(format!("deepseek-testkit-replay-{suffix}"));
        fs::create_dir_all(&workspace).expect("workspace");
        let mock = start_mock_llm_server();
        configure_runtime_for_mock_llm(&workspace, &mock.endpoint);
        // SAFETY: process-local env setup in test before engine creation.
        unsafe {
            std::env::set_var("DEEPSEEK_API_KEY", "test-api-key");
        }
        let result = run_replay_smoke(&workspace);
        assert!(result.is_ok(), "replay smoke failed: {:?}", result.err());
    }

    struct MockLlmServer {
        endpoint: String,
        stop_tx: Option<mpsc::Sender<()>>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl Drop for MockLlmServer {
        fn drop(&mut self) {
            if let Some(tx) = self.stop_tx.take() {
                let _ = tx.send(());
            }
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn start_mock_llm_server() -> MockLlmServer {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock llm");
        listener
            .set_nonblocking(true)
            .expect("set nonblocking listener");
        let addr = listener.local_addr().expect("mock addr");
        let (tx, rx) = mpsc::channel::<()>();
        let handle = thread::spawn(move || {
            loop {
                if rx.try_recv().is_ok() {
                    break;
                }
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let _ = handle_mock_llm_connection(&mut stream);
                    }
                    Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });
        MockLlmServer {
            endpoint: format!("http://{addr}/chat/completions"),
            stop_tx: Some(tx),
            handle: Some(handle),
        }
    }

    fn configure_runtime_for_mock_llm(workspace: &Path, endpoint: &str) {
        let runtime = workspace.join(".deepseek");
        fs::create_dir_all(&runtime).expect("runtime");
        fs::write(
            runtime.join("settings.local.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "llm": {
                    "provider": "deepseek",
                    "endpoint": endpoint,
                    "api_key_env": "DEEPSEEK_API_KEY"
                }
            }))
            .expect("serialize settings"),
        )
        .expect("write settings");
    }

    fn handle_mock_llm_connection(stream: &mut TcpStream) -> std::io::Result<()> {
        let mut buffer = Vec::new();
        let mut chunk = [0u8; 1024];
        let mut header_end = None;
        while header_end.is_none() {
            let read = stream.read(&mut chunk)?;
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&chunk[..read]);
            header_end = find_subsequence(&buffer, b"\r\n\r\n").map(|idx| idx + 4);
            if buffer.len() > 1_048_576 {
                break;
            }
        }
        let header_len = header_end.unwrap_or(buffer.len());
        let content_length = parse_content_length(&buffer[..header_len]);
        let mut body = if header_len <= buffer.len() {
            buffer[header_len..].to_vec()
        } else {
            Vec::new()
        };
        while body.len() < content_length {
            let read = stream.read(&mut chunk)?;
            if read == 0 {
                break;
            }
            body.extend_from_slice(&chunk[..read]);
        }
        let prompt = extract_prompt_from_request_body(&body).unwrap_or_else(|| "test".to_string());
        let content = if prompt.to_ascii_lowercase().contains("plan") {
            "Generated plan: discover files, propose edits, verify with tests.".to_string()
        } else {
            format!("Mock response: {prompt}")
        };
        let payload = serde_json::json!({
            "choices": [
                {
                    "message": {
                        "content": content
                    }
                }
            ]
        })
        .to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            payload.len(),
            payload
        );
        stream.write_all(response.as_bytes())?;
        stream.flush()?;
        Ok(())
    }

    fn parse_content_length(headers: &[u8]) -> usize {
        let raw = String::from_utf8_lossy(headers);
        for line in raw.lines() {
            let mut parts = line.splitn(2, ':');
            let key = parts.next().unwrap_or_default().trim();
            if key.eq_ignore_ascii_case("content-length")
                && let Some(value) = parts.next()
                && let Ok(parsed) = value.trim().parse::<usize>()
            {
                return parsed;
            }
        }
        0
    }

    fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() || haystack.len() < needle.len() {
            return None;
        }
        haystack
            .windows(needle.len())
            .position(|window| window == needle)
    }

    fn extract_prompt_from_request_body(body: &[u8]) -> Option<String> {
        let value: serde_json::Value = serde_json::from_slice(body).ok()?;
        value
            .get("messages")
            .and_then(|v| v.as_array())
            .and_then(|rows| rows.last())
            .and_then(|row| row.get("content"))
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
    }
}
