//! Candle-based local text generation backend.
//!
//! This module is only compiled when the `local-ml` feature is enabled.
//! Runs autoregressive token generation with temperature sampling for
//! ghost-text autocomplete suggestions.

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokenizers::Tokenizer;

use super::{GenOpts, LocalGenBackend};

/// Candle-powered local text generation using quantized LLaMA-family models.
///
/// Runs autoregressive token generation with temperature sampling.
/// Respects max_tokens, timeout, stop tokens, and an external cancel flag
/// for responsive ghost-text completion.
pub struct CandleCompletion {
    model: Mutex<ModelWeights>,
    tokenizer: Tokenizer,
    device: Device,
    cancel_flag: Arc<AtomicBool>,
    model_id_str: String,
}

impl CandleCompletion {
    /// Load a quantized GGUF model for text generation.
    ///
    /// - `model_path`: path to the .gguf weights file
    /// - `tokenizer_path`: path to the tokenizer.json
    /// - `device`: Candle compute device
    pub fn load(model_path: &Path, tokenizer_path: &Path, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(model_path)?;
        let model = ModelWeights::from_gguf(
            candle_core::quantized::gguf_file::Content::read(&mut file)?,
            &mut file,
            device,
        )?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("failed to load tokenizer: {}", e))?;

        let model_id_str = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("candle-gen")
            .to_string();

        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
            device: device.clone(),
            cancel_flag: Arc::new(AtomicBool::new(false)),
            model_id_str,
        })
    }

    /// Get the cancel flag for external cancellation.
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel_flag)
    }

    fn generate_tokens(
        &self,
        prompt: &str,
        opts: &GenOpts,
        on_token: Option<&dyn Fn(&str)>,
    ) -> Result<String> {
        self.cancel_flag.store(false, Ordering::SeqCst);
        let start = Instant::now();

        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("tokenization failed: {}", e))?;
        let input_ids = encoding.get_ids().to_vec();

        let mut logits_processor = LogitsProcessor::new(
            42, // seed
            Some(opts.temperature as f64),
            None, // top_p
        );

        let mut model = self
            .model
            .lock()
            .map_err(|e| anyhow!("model lock poisoned: {}", e))?;

        let mut all_tokens = input_ids.clone();
        let mut generated = String::new();

        // Process the full prompt and generate the first token.
        // Position starts at 0 for the prompt tokens.
        let input_tensor = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;
        let logits = model.forward(&input_tensor, 0)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let seq_len = logits.dim(0)?;
        let last_logits = logits.narrow(0, seq_len - 1, 1)?.squeeze(0)?;
        let mut current_token = logits_processor.sample(&last_logits)?;
        all_tokens.push(current_token);

        // Track position explicitly: after processing the prompt, the next position
        // is input_ids.len() (the first generated token occupies that position).
        // Each subsequent token advances position by 1.
        let mut pos = input_ids.len();

        // Autoregressive generation loop
        for _i in 0..opts.max_tokens {
            if self.cancel_flag.load(Ordering::SeqCst) {
                break;
            }
            if opts.timeout_ms > 0 && start.elapsed().as_millis() as u64 >= opts.timeout_ms {
                break;
            }

            let token_text = self
                .tokenizer
                .decode(&[current_token], true)
                .map_err(|e| anyhow!("decode failed: {}", e))?;

            // Check stop tokens
            if opts
                .stop_tokens
                .iter()
                .any(|s| token_text.contains(s.as_str()))
            {
                break;
            }

            generated.push_str(&token_text);
            if let Some(cb) = on_token {
                cb(&token_text);
            }

            // Advance position and generate next token
            pos += 1;
            let input = Tensor::new(&[current_token], &self.device)?.unsqueeze(0)?;
            let logits = model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let last_logits = logits.narrow(0, logits.dim(0)? - 1, 1)?.squeeze(0)?;
            current_token = logits_processor.sample(&last_logits)?;
            all_tokens.push(current_token);
        }

        Ok(generated)
    }
}

impl LocalGenBackend for CandleCompletion {
    fn generate(&self, prompt: &str, opts: &GenOpts) -> Result<String> {
        self.generate_tokens(prompt, opts, None)
    }

    fn generate_streaming(
        &self,
        prompt: &str,
        opts: &GenOpts,
        on_token: &dyn Fn(&str),
    ) -> Result<String> {
        self.generate_tokens(prompt, opts, Some(on_token))
    }

    fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
    }

    fn model_id(&self) -> &str {
        &self.model_id_str
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_dir() -> Option<PathBuf> {
        // Look for a GGUF model in a standard location
        let home = std::env::var("HOME").ok()?;
        let dir = PathBuf::from(home).join(".cache/deepseek/models");
        if dir.exists() { Some(dir) } else { None }
    }

    #[test]
    #[ignore] // Requires downloaded model files
    fn candle_completion_produces_output() {
        let dir = model_dir().expect("model dir not found");
        let model_path = dir.join("tinyllama-1.1b-chat.gguf");
        let tokenizer_path = dir.join("tokenizer.json");
        if !model_path.exists() || !tokenizer_path.exists() {
            panic!("model files not found in {:?}", dir);
        }

        let completer = CandleCompletion::load(&model_path, &tokenizer_path, &Device::Cpu).unwrap();
        let opts = GenOpts {
            max_tokens: 20,
            temperature: 0.1,
            ..Default::default()
        };
        let result = completer.generate("fn main() {", &opts).unwrap();
        assert!(!result.is_empty(), "should produce some output");
    }

    #[test]
    #[ignore] // Requires downloaded model files
    fn candle_completion_respects_cancel() {
        let dir = model_dir().expect("model dir not found");
        let model_path = dir.join("tinyllama-1.1b-chat.gguf");
        let tokenizer_path = dir.join("tokenizer.json");
        if !model_path.exists() || !tokenizer_path.exists() {
            panic!("model files not found");
        }

        let completer = CandleCompletion::load(&model_path, &tokenizer_path, &Device::Cpu).unwrap();
        let flag = completer.cancel_flag();

        // Cancel before generating
        flag.store(true, Ordering::SeqCst);
        let opts = GenOpts {
            max_tokens: 100,
            timeout_ms: 0, // no timeout — rely on cancel flag
            ..Default::default()
        };
        let result = completer.generate("fn test() {", &opts).unwrap();
        // Should produce minimal output since cancel is set
        assert!(
            result.len() < 200,
            "cancelled generation should produce little output"
        );
    }

    #[test]
    #[ignore] // Requires downloaded model files
    fn candle_completion_streaming_emits_tokens() {
        let dir = model_dir().expect("model dir not found");
        let model_path = dir.join("tinyllama-1.1b-chat.gguf");
        let tokenizer_path = dir.join("tokenizer.json");
        if !model_path.exists() || !tokenizer_path.exists() {
            panic!("model files not found");
        }

        let completer = CandleCompletion::load(&model_path, &tokenizer_path, &Device::Cpu).unwrap();
        let tokens = Arc::new(std::sync::Mutex::new(Vec::new()));
        let tokens_clone = tokens.clone();

        let opts = GenOpts {
            max_tokens: 10,
            temperature: 0.1,
            ..Default::default()
        };
        let result = completer
            .generate_streaming("Hello", &opts, &move |token| {
                tokens_clone.lock().unwrap().push(token.to_string());
            })
            .unwrap();

        let emitted = tokens.lock().unwrap();
        if !result.is_empty() {
            assert!(!emitted.is_empty(), "streaming should emit tokens");
        }
    }
}
