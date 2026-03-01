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

#[cfg(feature = "local-ml")]
pub mod candle_backend;
