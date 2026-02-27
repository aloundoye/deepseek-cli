//! Candle-based embeddings backend using BERT models.
//!
//! This module is only compiled when the `local-ml` feature is enabled.
//! Supports models like all-MiniLM-L6-v2 (384d) and jina-code-v2 (768d).

use anyhow::{Result, anyhow, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use std::path::Path;
use tokenizers::Tokenizer;

use super::EmbeddingsBackend;

/// Candle-powered embeddings backend using a BERT-family model.
///
/// Tokenizes input text, runs a forward pass through the BERT model,
/// applies mean pooling over the token embeddings, and optionally L2-normalizes
/// the result for consistent cosine similarity behavior.
pub struct CandleEmbeddings {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dim: usize,
    normalize: bool,
}

impl CandleEmbeddings {
    /// Load a BERT model and tokenizer from local file paths.
    ///
    /// - `model_path`: path to the SafeTensors weights file
    /// - `config_path`: path to the model config.json
    /// - `tokenizer_path`: path to the tokenizer.json
    /// - `device`: Candle compute device (CPU/CUDA/Metal)
    /// - `normalize`: whether to L2-normalize output vectors
    pub fn load(
        model_path: &Path,
        config_path: &Path,
        tokenizer_path: &Path,
        device: &Device,
        normalize: bool,
    ) -> Result<Self> {
        let config_data = std::fs::read_to_string(config_path)?;
        let config: BertConfig = serde_json::from_str(&config_data)?;
        let dim = config.hidden_size;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, device)?
        };
        let model = BertModel::load(vb, &config)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("failed to load tokenizer: {}", e))?;

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            dim,
            normalize,
        })
    }

    /// Run mean pooling over the last hidden state, producing a single vector.
    fn mean_pool(&self, hidden: &Tensor, attention_mask: &Tensor) -> Result<Vec<f32>> {
        // hidden: (1, seq_len, hidden_size)
        // attention_mask: (1, seq_len)
        let mask = attention_mask
            .unsqueeze(2)?
            .to_dtype(DType::F32)?
            .broadcast_as(hidden.shape())?;
        let masked = (hidden * &mask)?;
        let summed = masked.sum(1)?; // (1, hidden_size)
        let counts = mask.sum(1)?; // (1, hidden_size)
        let pooled = (summed / counts)?;
        let pooled = pooled.squeeze(0)?; // (hidden_size,)

        let mut vec: Vec<f32> = pooled.to_vec1()?;

        if self.normalize {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut vec {
                    *v /= norm;
                }
            }
        }

        Ok(vec)
    }
}

impl EmbeddingsBackend for CandleEmbeddings {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenization failed: {}", e))?;

        let input_ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
        let attention_mask =
            Tensor::new(encoding.get_attention_mask(), &self.device)?.unsqueeze(0)?;
        let token_type_ids =
            Tensor::new(encoding.get_type_ids(), &self.device)?.unsqueeze(0)?;

        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        self.mean_pool(&hidden, &attention_mask)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // For small batches, per-item is simpler and avoids padding complexity.
        // A production implementation would pad and batch for GPU throughput.
        if texts.len() <= 4 {
            return texts.iter().map(|t| self.embed(t)).collect();
        }

        // Batch encoding with padding for larger batches
        let encodings: Vec<_> = texts
            .iter()
            .map(|t| {
                self.tokenizer
                    .encode(*t, true)
                    .map_err(|e| anyhow!("tokenization failed: {}", e))
            })
            .collect::<Result<Vec<_>>>()?;

        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        if max_len == 0 {
            bail!("all inputs produced empty tokenizations");
        }

        let mut results = Vec::with_capacity(texts.len());
        // Process individually — batched GPU inference would go here
        for encoding in &encodings {
            let input_ids =
                Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
            let attention_mask =
                Tensor::new(encoding.get_attention_mask(), &self.device)?.unsqueeze(0)?;
            let token_type_ids =
                Tensor::new(encoding.get_type_ids(), &self.device)?.unsqueeze(0)?;

            let hidden = self
                .model
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
            results.push(self.mean_pool(&hidden, &attention_mask)?);
        }

        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_id(&self) -> &str {
        "candle-bert"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_dir() -> Option<PathBuf> {
        // Look for model files in standard HF cache or a local test directory
        let home = std::env::var("HOME").ok()?;
        let cache = PathBuf::from(home).join(".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2");
        if cache.exists() {
            Some(cache)
        } else {
            None
        }
    }

    #[test]
    #[ignore] // Requires downloaded model files
    fn candle_embed_produces_vector() {
        let dir = model_dir().expect("model not found — run with model downloaded");
        let snapshots = std::fs::read_dir(dir.join("snapshots"))
            .expect("snapshots dir")
            .filter_map(|e| e.ok())
            .next()
            .expect("at least one snapshot");
        let snap = snapshots.path();

        let embeddings = CandleEmbeddings::load(
            &snap.join("model.safetensors"),
            &snap.join("config.json"),
            &snap.join("tokenizer.json"),
            &Device::Cpu,
            true,
        )
        .expect("load model");

        let vec = embeddings.embed("Hello world").expect("embed");
        assert_eq!(vec.len(), embeddings.dimension());
        // Normalized vector should have unit length
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "expected unit vector, got norm={}", norm);
    }

    #[test]
    #[ignore] // Requires downloaded model files
    fn candle_embed_deterministic() {
        let dir = model_dir().expect("model not found");
        let snap = std::fs::read_dir(dir.join("snapshots"))
            .expect("snapshots")
            .filter_map(|e| e.ok())
            .next()
            .expect("snapshot")
            .path();

        let embeddings = CandleEmbeddings::load(
            &snap.join("model.safetensors"),
            &snap.join("config.json"),
            &snap.join("tokenizer.json"),
            &Device::Cpu,
            true,
        )
        .expect("load");

        let v1 = embeddings.embed("test input").expect("embed1");
        let v2 = embeddings.embed("test input").expect("embed2");
        assert_eq!(v1, v2, "same input must produce same vector");
    }

    #[test]
    #[ignore] // Requires downloaded model files
    fn candle_embed_similar_texts_closer() {
        let dir = model_dir().expect("model not found");
        let snap = std::fs::read_dir(dir.join("snapshots"))
            .expect("snapshots")
            .filter_map(|e| e.ok())
            .next()
            .expect("snapshot")
            .path();

        let embeddings = CandleEmbeddings::load(
            &snap.join("model.safetensors"),
            &snap.join("config.json"),
            &snap.join("tokenizer.json"),
            &Device::Cpu,
            true,
        )
        .expect("load");

        let v_rust = embeddings.embed("fn main() { println!(\"hello\"); }").unwrap();
        let v_python = embeddings.embed("def main(): print('hello')").unwrap();
        let v_recipe = embeddings.embed("preheat the oven to 350 degrees").unwrap();

        // Cosine similarity (vectors are normalized, so dot product = cosine)
        let sim_code: f32 = v_rust.iter().zip(&v_python).map(|(a, b)| a * b).sum();
        let sim_unrelated: f32 = v_rust.iter().zip(&v_recipe).map(|(a, b)| a * b).sum();

        assert!(
            sim_code > sim_unrelated,
            "code snippets should be more similar ({}) than code vs recipe ({})",
            sim_code,
            sim_unrelated
        );
    }

    #[test]
    #[ignore] // Requires downloaded model files
    fn candle_embed_batch_matches_individual() {
        let dir = model_dir().expect("model not found");
        let snap = std::fs::read_dir(dir.join("snapshots"))
            .expect("snapshots")
            .filter_map(|e| e.ok())
            .next()
            .expect("snapshot")
            .path();

        let embeddings = CandleEmbeddings::load(
            &snap.join("model.safetensors"),
            &snap.join("config.json"),
            &snap.join("tokenizer.json"),
            &Device::Cpu,
            true,
        )
        .expect("load");

        let texts = ["hello world", "cargo test", "fn main()"];
        let batch = embeddings.embed_batch(&texts).unwrap();
        for (i, text) in texts.iter().enumerate() {
            let single = embeddings.embed(text).unwrap();
            let diff: f32 = batch[i]
                .iter()
                .zip(&single)
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(diff < 1e-5, "batch[{}] should match individual embed", i);
        }
    }
}
