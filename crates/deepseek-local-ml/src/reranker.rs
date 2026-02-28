use anyhow::Result;

/// Backend trait for cross-encoder reranking of retrieval results.
///
/// After initial retrieval (vector + BM25), a cross-encoder scores each
/// (query, document) pair jointly for higher precision. This is the single
/// biggest quality improvement for retrieval pipelines.
pub trait RerankerBackend: Send + Sync {
    /// Score (query, document) pairs. Returns relevance scores in the same order
    /// as the input documents. Higher = more relevant.
    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>>;

    /// Model identifier.
    fn model_id(&self) -> &str;
}

/// Mock reranker that preserves input order (returns scores by position).
/// For testing — first document gets highest score.
pub struct MockReranker;

impl MockReranker {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MockReranker {
    fn default() -> Self {
        Self::new()
    }
}

impl RerankerBackend for MockReranker {
    fn rerank(&self, _query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        // Return decreasing scores so input order is preserved
        Ok((0..documents.len())
            .map(|i| 1.0 - (i as f32 / documents.len().max(1) as f32))
            .collect())
    }

    fn model_id(&self) -> &str {
        "mock-reranker"
    }
}

#[cfg(feature = "local-ml")]
pub mod candle_backend {
    //! Cross-encoder reranker using Candle with MiniLM-based models.
    //!
    //! Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params) for
    //! high-quality passage reranking after initial retrieval.

    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::bert::{BertModel, Config as BertConfig};
    use tokenizers::Tokenizer;

    /// Candle-powered cross-encoder reranker.
    ///
    /// Unlike bi-encoders (which embed query and document independently),
    /// cross-encoders process (query, document) pairs jointly through the
    /// full transformer, producing much more accurate relevance scores.
    pub struct CandleReranker {
        model: BertModel,
        tokenizer: Tokenizer,
        device: Device,
        model_id_str: String,
    }

    impl CandleReranker {
        /// Load a cross-encoder model for reranking.
        pub fn load(
            model_path: &std::path::Path,
            config_path: &std::path::Path,
            tokenizer_path: &std::path::Path,
            device: &Device,
        ) -> Result<Self> {
            let config_data = std::fs::read_to_string(config_path)?;
            let config: BertConfig = serde_json::from_str(&config_data)?;

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, device)?
            };
            let model = BertModel::load(vb, &config)?;

            let tokenizer = Tokenizer::from_file(tokenizer_path)
                .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?;

            let model_id_str = model_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("candle-reranker")
                .to_string();

            Ok(Self {
                model,
                tokenizer,
                device: device.clone(),
                model_id_str,
            })
        }
    }

    impl RerankerBackend for CandleReranker {
        fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
            let mut scores = Vec::with_capacity(documents.len());

            for doc in documents {
                // Cross-encoders use [CLS] query [SEP] document [SEP] format
                let pair = format!("{} [SEP] {}", query, doc);
                let encoding = self
                    .tokenizer
                    .encode(pair, true)
                    .map_err(|e| anyhow::anyhow!("tokenization failed: {}", e))?;

                let input_ids =
                    Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
                let attention_mask =
                    Tensor::new(encoding.get_attention_mask(), &self.device)?.unsqueeze(0)?;
                let token_type_ids =
                    Tensor::new(encoding.get_type_ids(), &self.device)?.unsqueeze(0)?;

                let hidden = self
                    .model
                    .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

                // Use [CLS] token output as the relevance score
                let cls = hidden.narrow(1, 0, 1)?.squeeze(1)?.squeeze(0)?;
                let score: f32 = cls.to_vec1::<f32>()?[0];
                scores.push(score);
            }

            Ok(scores)
        }

        fn model_id(&self) -> &str {
            &self.model_id_str
        }
    }
}

/// Rerank a list of documents and return indices sorted by relevance (highest first).
pub fn rerank_indices(
    reranker: &dyn RerankerBackend,
    query: &str,
    documents: &[&str],
) -> Result<Vec<usize>> {
    let scores = reranker.rerank(query, documents)?;
    let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(indexed.into_iter().map(|(i, _)| i).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_reranker_preserves_order() {
        let reranker = MockReranker::new();
        let docs = vec!["first", "second", "third"];
        let scores = reranker.rerank("query", &docs).unwrap();
        assert_eq!(scores.len(), 3);
        // First doc should have highest score
        assert!(scores[0] > scores[1]);
        assert!(scores[1] > scores[2]);
    }

    #[test]
    fn rerank_indices_sorts_by_score() {
        let reranker = MockReranker::new();
        let docs = vec!["a", "b", "c"];
        let indices = rerank_indices(&reranker, "query", &docs).unwrap();
        // Mock reranker gives highest score to first doc, so order is preserved
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn mock_reranker_empty_docs() {
        let reranker = MockReranker::new();
        let scores = reranker.rerank("query", &[]).unwrap();
        assert!(scores.is_empty());
    }
}
