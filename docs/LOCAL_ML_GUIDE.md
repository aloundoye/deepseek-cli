# Local ML Guide

CodingBuddy includes an optional local ML layer that runs entirely on your machine. It serves as an **intelligence compensator** — improving the quality of information the LLM receives, not replacing it.

**Core value (keep & double down):**
- **Hybrid code retrieval** — automatically surfaces relevant code chunks before the LLM responds (every turn, not just the first)
- **Privacy scanning** — detects and redacts secrets in tool outputs before they reach the API
- **Cross-encoder reranking** — improves retrieval quality by re-scoring results with a learned model

**Supporting features:**
- **Ghost text** — inline code completions in the TUI, powered by local models
- **Local routing** — routes simple non-project questions to a local model (project-context queries always go to the API)

**Not supported:**
- **Speculative decoding** — removed from the public local-ml API because it is not runtime-wired. Focus remains on retrieval/privacy/reranking.

All of this is **off by default**. No models are bundled — they're downloaded from HuggingFace on first use.

---

## Quick Start

### 1. Enable local ML in your project config

```bash
mkdir -p .codingbuddy
cat > .codingbuddy/settings.json << 'EOF'
{
  "local_ml": {
    "enabled": true
  }
}
EOF
```

Or in your user config (`~/.codingbuddy/settings.json`) to enable globally.

### 2. That's it (mock mode)

With just `enabled: true`, the CLI uses **mock backends**:
- Embeddings: deterministic SHA-256 hashing (not semantic, but consistent)
- Retrieval: keyword/hash-based code chunk matching via BruteForce cosine similarity
- Privacy: fully functional 3-layer secret detection
- Ghost text: disabled (mock generator returns empty)

This works without downloading anything. It's useful for privacy scanning and basic retrieval.

### 3. Enable real ML models (optional)

For semantic embeddings and code completion, compile with the `local-ml` feature:

```bash
cargo build --release --bin codingbuddy --features local-ml
```

This pulls in Candle (Rust ML framework) and enables:
- **Real embeddings** via `jina-embeddings-v2-base-code` (~270MB download)
- **Real code completion** via quantized LLaMA models (~700MB–1.5GB download)
- **HNSW vector index** via Usearch (faster search for large codebases)

Models are downloaded automatically from HuggingFace Hub on first use and cached in `~/.cache/deepseek/`.

---

## Features in Detail

### Hybrid Retrieval

When you ask a question, the retrieval pipeline:

1. **Chunks** your workspace files into overlapping windows (configurable size)
2. **Indexes** chunks using vector embeddings + BM25 text search
3. **Searches** for chunks relevant to your query using Reciprocal Rank Fusion (RRF)
4. **Injects** the top results as context before the LLM sees your message

This means the LLM starts with relevant code context instead of being blind.

The index is built lazily on first query and updates incrementally when files change (SHA-256 change detection).

Retrieval fires on **every turn** of the conversation (not just the first), with budget based on remaining context window. This keeps multi-turn conversations grounded in actual code.

**Configuration:**

```json
{
  "local_ml": {
    "enabled": true,
    "retrieval": {
      "enabled": true,
      "max_results": 10,
      "context_budget_pct": 0.15,
      "rrf_k": 60.0
    },
    "chunker": {
      "window_size": 512,
      "overlap": 64,
      "max_file_size": 200000
    }
  }
}
```

### Privacy Router

Scans tool outputs (file reads, command results) for sensitive content before sending to the DeepSeek API. Three detection layers:

1. **Path-based**: files matching patterns like `.env*`, `*credentials*`, `*.pem`, `*.key`
2. **Content-based**: regex patterns for `api.?key`, `secret`, `password`, `token`
3. **Builtin patterns**: AWS keys (`AKIA*`), GitHub tokens (`ghp_*`), PEM blocks, connection strings

**Policies:**
- `BlockCloud` — prevents the content from reaching the API entirely
- `Redact` — replaces sensitive parts with `[REDACTED:type]`
- `LocalOnlySummary` — generates a local summary instead of sending raw content

**Configuration:**

```json
{
  "local_ml": {
    "enabled": true,
    "privacy": {
      "enabled": true,
      "policy": "Redact",
      "path_globs": [".env*", "*credentials*", "*.pem", "*.key"],
      "content_patterns": ["(?i)api.?key", "(?i)secret"]
    }
  }
}
```

**CLI commands:**

```bash
codingbuddy privacy scan              # Scan workspace for sensitive files
codingbuddy privacy policy            # Show current privacy policy
codingbuddy privacy redact-preview    # Preview what would be redacted
```

### Ghost Text (Autocomplete)

When enabled, the TUI shows inline code suggestions as you type (gray ghost text).

- **Tab**: Accept full suggestion
- **Alt+Right**: Accept one word
- **Esc**: Dismiss suggestion
- **200ms debounce**: Suggestions appear after you stop typing

With mock backends, ghost text is non-functional. With `--features local-ml`, it uses a local code completion model.

**Configuration:**

```json
{
  "local_ml": {
    "enabled": true,
    "autocomplete": {
      "enabled": true,
      "debounce_ms": 200,
      "max_tokens": 64,
      "temperature": 0.2
    }
  }
}
```

```bash
codingbuddy autocomplete enable       # Enable ghost text
codingbuddy autocomplete disable      # Disable ghost text
codingbuddy autocomplete model        # Show/change completion model
```

### Index Management

```bash
codingbuddy index build               # Build full-text + vector index
codingbuddy index status              # Show index stats
codingbuddy index --hybrid doctor     # Diagnose hybrid index issues
codingbuddy index --hybrid clean      # Remove and rebuild index
codingbuddy index query "search term" # Query the index directly
codingbuddy index watch               # Watch for file changes and auto-update
```

---

## Model Configuration

### Embeddings

| Setting | Default | Description |
|---------|---------|-------------|
| `local_ml.embeddings.model_id` | `jinaai/jina-embeddings-v2-base-code` | HuggingFace model ID |
| `local_ml.embeddings.dimension` | `384` | Vector dimension |
| `local_ml.embeddings.batch_size` | `32` | Batch size for embedding |

### Completion

| Setting | Default | Description |
|---------|---------|-------------|
| `local_ml.completion.model_id` | `qwen2.5-coder-3b` | Model ID for local code completion |
| `local_ml.completion.max_tokens` | `64` | Max tokens per completion |
| `local_ml.completion.temperature` | `0.2` | Sampling temperature |

### Cache

Models are stored in `~/.cache/deepseek/<model_id>/`. Set `local_ml.cache_dir` to change this.

The vector index is stored in `.codingbuddy/vector_index.sqlite` in your project directory.

### Memory-Aware Loading

When `local-ml` is enabled, the model loader checks available system memory before loading and selects a strategy:

| Strategy | Condition | Behavior |
|----------|-----------|----------|
| **Full** | Available >= 2x model size | Load at full context window |
| **ReducedContext** | Available >= 1x model size | Load with halved context window |
| **CpuOnly** | Available >= 0.5x model size | Force CPU-only (no GPU layers) |
| **Skip** | Available < 0.5x model size | Skip loading, fall back to mock with warning |

This prevents OOM crashes on memory-constrained machines. The check uses `sysctl` on macOS and `/proc/meminfo` on Linux.

### Parallel Downloads

Model files are downloaded concurrently using thread-scoped parallelism. A 60-second stall detection timer automatically retries individual files that stop making progress. Downloads resume from partial state if interrupted.

---

## Build Variants

| Build | Command | Embeddings | Completion | Vector Index | Size |
|-------|---------|-----------|------------|-------------|------|
| Default | `cargo build --release` | Mock (SHA-256) | Mock (empty) | BruteForce (O(n)) | ~15MB |
| Full ML | `cargo build --release --features local-ml` | Candle/BERT | Candle/GGUF | Usearch (HNSW) | ~25MB + models |

The `local-ml` feature adds ~10MB to the binary and requires model downloads on first use.

---

## Troubleshooting

**Retrieval returns irrelevant results:**
- Rebuild the index: `codingbuddy index --hybrid clean && codingbuddy index build`
- With mock embeddings, results are hash-based not semantic — enable `--features local-ml` for better quality

**Ghost text not appearing:**
- Verify config: `codingbuddy config show | grep -A5 local_ml`
- Must be in TUI mode (not `--json`)
- Mock generator returns empty — need `--features local-ml` for real completions

**Privacy false positives:**
- Adjust patterns in `local_ml.privacy.path_globs` and `local_ml.privacy.content_patterns`
- Preview with `codingbuddy privacy redact-preview`

**Model download fails:**
- Check internet connectivity and `~/.cache/deepseek/` permissions
- Set `HF_HOME` to change HuggingFace cache location
- Falls back to mock backends automatically on failure

**Vector index corruption:**
- Delete `.codingbuddy/vector_index.sqlite` and restart — lazy indexing rebuilds automatically
