use anyhow::Result;

/// Options for local text generation.
#[derive(Debug, Clone)]
pub struct GenOpts {
    pub max_tokens: u32,
    pub temperature: f32,
    pub stop_tokens: Vec<String>,
    pub timeout_ms: u64,
}

impl Default for GenOpts {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.2,
            stop_tokens: vec!["\n".to_string()],
            timeout_ms: 2000,
        }
    }
}

/// Backend trait for local text generation (autocomplete, ghost text).
///
/// Implementations run a small language model locally to produce completions.
/// The trait is object-safe for runtime backend swapping.
pub trait LocalGenBackend: Send + Sync {
    /// Generate a completion for the given prompt.
    fn generate(&self, prompt: &str, opts: &GenOpts) -> Result<String>;

    /// Generate with streaming callback. Default calls `generate` and emits the full result.
    fn generate_streaming(
        &self,
        prompt: &str,
        opts: &GenOpts,
        _on_token: &dyn Fn(&str),
    ) -> Result<String> {
        self.generate(prompt, opts)
    }

    /// Cancel any in-progress generation. Default is a no-op.
    fn cancel(&self) {}

    /// Model identifier string.
    fn model_id(&self) -> &str;
}

/// Mock generator that returns a fixed output, for testing.
pub struct MockGenerator {
    fixed_output: String,
}

impl MockGenerator {
    pub fn new(output: String) -> Self {
        Self {
            fixed_output: output,
        }
    }
}

impl LocalGenBackend for MockGenerator {
    fn generate(&self, _prompt: &str, _opts: &GenOpts) -> Result<String> {
        Ok(self.fixed_output.clone())
    }

    fn model_id(&self) -> &str {
        "mock-generator"
    }
}

/// Buffer for accumulating raw bytes and emitting valid UTF-8 strings.
///
/// Used by the Candle streaming backend (feature-gated) and tested without the feature.
///
/// Handles multi-byte UTF-8 sequences that may be split across token boundaries
/// during streaming inference. Bytes are accumulated until they form a valid
/// UTF-8 string, then emitted via `push()`. Call `flush()` at end of generation
/// to emit any remaining valid bytes (replacing invalid trailing bytes).
pub struct Utf8StreamBuffer {
    pending: Vec<u8>,
}

impl Default for Utf8StreamBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl Utf8StreamBuffer {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Append bytes and return a valid UTF-8 string if one can be formed.
    ///
    /// Partial multi-byte sequences at the end are retained in the buffer.
    pub fn push(&mut self, bytes: &[u8]) -> Option<String> {
        self.pending.extend_from_slice(bytes);

        // Find the longest prefix that is valid UTF-8.
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let result = s.to_string();
                self.pending.clear();
                if result.is_empty() {
                    None
                } else {
                    Some(result)
                }
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                if valid_up_to == 0 {
                    // No valid bytes yet — keep accumulating partial multi-byte sequences.
                    // If 4+ bytes are pending and still invalid (max UTF-8 sequence is 4 bytes),
                    // the leading byte is corrupt — drop it so subsequent bytes can be tried.
                    if self.pending.len() >= 4 {
                        self.pending.drain(..1);
                    }
                    None
                } else {
                    let valid =
                        String::from_utf8(self.pending.drain(..valid_up_to).collect()).ok()?;
                    Some(valid)
                }
            }
        }
    }

    /// Flush remaining bytes, replacing any invalid sequences.
    pub fn flush(&mut self) -> Option<String> {
        if self.pending.is_empty() {
            return None;
        }
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        if s.is_empty() { None } else { Some(s) }
    }
}

pub mod kv_cache;

#[cfg(feature = "local-ml")]
pub mod candle_backend;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_stream_buffer_handles_complete_ascii() {
        let mut buf = Utf8StreamBuffer::new();
        assert_eq!(buf.push(b"hello"), Some("hello".into()));
        assert_eq!(buf.flush(), None);
    }

    #[test]
    fn utf8_stream_buffer_handles_partial_sequences() {
        let mut buf = Utf8StreamBuffer::new();
        // é is U+00E9 = 0xC3 0xA9 in UTF-8
        // Push only the first byte — should not emit yet
        assert_eq!(buf.push(&[0xC3]), None);
        // Push the second byte — should emit "é"
        assert_eq!(buf.push(&[0xA9]), Some("é".into()));
        assert_eq!(buf.flush(), None);
    }

    #[test]
    fn utf8_stream_buffer_handles_mixed_sequences() {
        let mut buf = Utf8StreamBuffer::new();
        // "hé" = [0x68, 0xC3, 0xA9]
        // Push "h" + first byte of "é"
        assert_eq!(buf.push(&[0x68, 0xC3]), Some("h".into()));
        // Push second byte of "é"
        assert_eq!(buf.push(&[0xA9]), Some("é".into()));
    }

    #[test]
    fn utf8_stream_buffer_flush_replaces_invalid() {
        let mut buf = Utf8StreamBuffer::new();
        // Push a partial multi-byte sequence that never completes
        assert_eq!(buf.push(&[0xC3]), None);
        // Flush should emit replacement character
        let flushed = buf.flush();
        assert!(flushed.is_some());
        assert!(flushed.unwrap().contains('\u{FFFD}'));
    }

    #[test]
    fn utf8_stream_buffer_handles_three_byte_split() {
        let mut buf = Utf8StreamBuffer::new();
        // 中 is U+4E2D = 0xE4 0xB8 0xAD in UTF-8
        assert_eq!(buf.push(&[0xE4]), None);
        assert_eq!(buf.push(&[0xB8]), None);
        assert_eq!(buf.push(&[0xAD]), Some("中".into()));
    }
}
