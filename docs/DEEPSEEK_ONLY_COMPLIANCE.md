# DeepSeek-Only Compliance

This document certifies that the DeepSeek CLI project exclusively uses DeepSeek as its LLM provider. No other provider (OpenAI, Anthropic, Google Gemini, Azure, AWS Bedrock, Vertex AI, Ollama, or any third-party abstraction layer) is supported, referenced, or reachable from production code paths.

---

## 1. Provider Enforcement (Triple Validation)

Provider identity is validated at three independent layers. A non-DeepSeek provider value is rejected before any network call is made.

### 1a. `deepseek-llm` -- `complete()` (lines 323-328)

```rust
// crates/deepseek-llm/src/lib.rs:323-328
fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
    let provider = self.cfg.provider.to_ascii_lowercase();
    if provider != "deepseek" {
        return Err(anyhow!(
            "unsupported llm.provider='{}' (only 'deepseek' is supported)",
            self.cfg.provider
        ));
    }
    // ...
}
```

### 1b. `deepseek-llm` -- `complete_streaming()` (lines 346-351)

```rust
// crates/deepseek-llm/src/lib.rs:346-351
fn complete_streaming(&self, req: &LlmRequest, cb: StreamCallback) -> Result<LlmResponse> {
    let provider = self.cfg.provider.to_ascii_lowercase();
    if provider != "deepseek" {
        return Err(anyhow!(
            "unsupported llm.provider='{}' (only 'deepseek' is supported)",
            self.cfg.provider
        ));
    }
    // ...
}
```

### 1c. `deepseek-cli` -- `ensure_llm_ready_with_cfg()` (lines 1014-1023)

```rust
// crates/deepseek-cli/src/main.rs:1014-1023
fn ensure_llm_ready_with_cfg(cwd: Option<&Path>, cfg: &AppConfig, json_mode: bool) -> Result<()> {
    use std::io::IsTerminal;

    let provider = cfg.llm.provider.trim().to_ascii_lowercase();
    if provider != "deepseek" {
        return Err(anyhow!(
            "unsupported llm.provider='{}' (supported: deepseek)",
            cfg.llm.provider
        ));
    }
    // ...
}
```

### 1d. Default provider is `"deepseek"` (deepseek-core)

```rust
// crates/deepseek-core/src/lib.rs:744
impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            // ...
            provider: "deepseek".to_string(),
            // ...
        }
    }
}
```

---

## 2. Model Validation

### `normalize_deepseek_model()` (deepseek-core, lines 16-32)

Only DeepSeek model identifiers are accepted. All other strings return `None`, which is treated as an error upstream.

```rust
pub fn normalize_deepseek_model(model: &str) -> Option<&'static str> {
    let normalized = model.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "deepseek-chat" | "deepseek-v3.2" | "deepseek-v3.2-chat" | "v3.2" | "v3_2" => {
            Some(DEEPSEEK_V32_CHAT_MODEL)
        }
        "deepseek-reasoner"
        | "deepseek-v3.2-reasoner"
        | "reasoner"
        | "v3.2-reasoner"
        | "v3_2_reasoner" => Some(DEEPSEEK_V32_REASONER_MODEL),
        _ => None,
    }
}
```

Accepted models (all resolve to DeepSeek API model IDs):

| Alias(es) | Resolves to |
|---|---|
| `deepseek-chat`, `deepseek-v3.2`, `deepseek-v3.2-chat`, `v3.2`, `v3_2` | `deepseek-chat` |
| `deepseek-reasoner`, `deepseek-v3.2-reasoner`, `reasoner`, `v3.2-reasoner`, `v3_2_reasoner` | `deepseek-reasoner` |

### `normalize_deepseek_profile()` (deepseek-core)

Only one profile is accepted:

| Input(s) | Resolves to |
|---|---|
| `""`, `v3_2`, `v3.2`, `v32`, `deepseek-v3.2` | `v3_2` |

Any other profile string returns `None` and the request is rejected.

---

## 3. Endpoint Configuration

All API traffic targets a single, hardcoded default endpoint:

```rust
// crates/deepseek-core/src/lib.rs (LlmConfig::default)
endpoint: "https://api.deepseek.com/chat/completions".to_string(),
```

Authentication is via the `DEEPSEEK_API_KEY` environment variable (or the `llm.api_key` config field). The client sends a standard `Authorization: Bearer <key>` header. No other authentication scheme is implemented.

```rust
// crates/deepseek-core/src/lib.rs (LlmConfig::default)
api_key_env: "DEEPSEEK_API_KEY".to_string(),
```

---

## 4. Codebase Grep Audit -- Forbidden Identifiers Not Found

A comprehensive search of the codebase confirms that no competing provider names, multi-LLM abstractions, or alternative backend identifiers exist in production code:

| Search term | Result |
|---|---|
| `OpenAI` | Not found (except in this compliance document) |
| `Anthropic` | Not found |
| `Gemini` | Not found |
| `Azure` | Not found |
| `Bedrock` | Not found |
| `Vertex` | Not found |
| `Ollama` | Not found (only in test assertions rejecting it as a provider) |
| `LLMBackend` | Not found |
| `ModelRegistry` | Not found |
| `multi-provider` / `multi_provider` | Not found as a code abstraction |

The word `"provider"` appears only in the context of enforcing that the configured provider is `"deepseek"`.

---

## 5. DeepSeek Client Modules

All LLM interaction is implemented in a single crate with a single client struct:

**File:** `crates/deepseek-llm/src/lib.rs`

| Component | Description |
|---|---|
| `DeepSeekClient` struct | Holds `LlmConfig` and a `reqwest::blocking::Client` |
| `LlmClient` trait | Defines `complete()` and `complete_streaming()` -- the only LLM interface |
| `complete_inner()` | Non-streaming HTTP POST to the DeepSeek API |
| `complete_streaming_inner()` | Streaming SSE reader for DeepSeek's `data:` line protocol |
| `build_payload()` | Constructs the JSON request body with DeepSeek-specific fields (`reasoning_content`, model normalization) |
| `resolve_api_key()` | Reads `DEEPSEEK_API_KEY` env var or falls back to `llm.api_key` config |
| `resolve_request_model()` | Validates the model via `normalize_deepseek_model()` before any network call |
| SSE parsing | Handles DeepSeek-specific delta fields: `content`, `reasoning_content`, `tool_calls` |

Authentication is exclusively Bearer token (`self.client.post(&self.cfg.endpoint).bearer_auth(api_key)`). No OAuth, SigV4, or other auth scheme exists.

---

## 6. Dependencies -- No Third-Party LLM SDKs

The workspace `Cargo.toml` and all crate-level `Cargo.toml` files contain zero references to third-party LLM client libraries:

| Dependency | Present? |
|---|---|
| `openai-client` / `async-openai` | No |
| `anthropic-sdk` / `anthropic` | No |
| `aws-bedrock` / `aws-sdk-bedrockruntime` | No |
| `vertex-ai` / `google-cloud` | No |
| `ollama-rs` / `ollama` | No |

The only HTTP dependency is `reqwest` (with `blocking`, `json`, and `rustls` features), used directly by `DeepSeekClient` to call the DeepSeek API.

---

## 7. Test Coverage

Dedicated tests verify that non-DeepSeek configurations are rejected at every validation layer.

### `non_deepseek_provider_is_rejected()` (deepseek-llm, lines 821-843)

Iterates over `["openai", "anthropic", "custom", "local", "ollama"]` and asserts that each is rejected with `"only 'deepseek' is supported"`.

```rust
#[test]
fn non_deepseek_provider_is_rejected() {
    for provider in &["openai", "anthropic", "custom", "local", "ollama"] {
        let cfg = LlmConfig {
            provider: provider.to_string(),
            api_key: Some("test-key".to_string()),
            ..LlmConfig::default()
        };
        let client = DeepSeekClient::new(cfg).expect("client");
        let err = client
            .complete(&LlmRequest { /* ... */ })
            .expect_err("non-deepseek provider should be rejected");
        assert!(
            err.to_string().contains("only 'deepseek' is supported"),
            "provider '{provider}' should be rejected but got: {err}"
        );
    }
}
```

### `unsupported_model_is_rejected_before_network_call()` (deepseek-llm, lines 780-796)

Confirms that a model string like `"not-a-deepseek-model"` fails validation before any HTTP request is attempted.

```rust
#[test]
fn unsupported_model_is_rejected_before_network_call() {
    let cfg = LlmConfig {
        api_key: Some("test-key".to_string()),
        ..LlmConfig::default()
    };
    let client = DeepSeekClient::new(cfg).expect("client");
    let err = client
        .complete(&LlmRequest {
            model: "not-a-deepseek-model".to_string(),
            /* ... */
        })
        .expect_err("unsupported model should fail");
    assert!(err.to_string().contains("unsupported model"));
}
```

### `unsupported_profile_is_rejected()` (deepseek-llm, lines 760-777)

Verifies that a profile value of `"unknown"` is rejected with `"unsupported llm.profile"`.

```rust
#[test]
fn unsupported_profile_is_rejected() {
    let cfg = LlmConfig {
        profile: "unknown".to_string(),
        api_key: Some("test-key".to_string()),
        ..LlmConfig::default()
    };
    let client = DeepSeekClient::new(cfg).expect("client");
    let err = client
        .complete(&LlmRequest {
            model: "deepseek-chat".to_string(),
            /* ... */
        })
        .expect_err("unsupported profile should fail");
    assert!(err.to_string().contains("unsupported llm.profile"));
}
```

### Additional related tests

| Test name | What it verifies |
|---|---|
| `truly_unsupported_provider_is_rejected()` | Arbitrary non-deepseek provider string is rejected |
| `missing_api_key_is_rejected()` | Absent `DEEPSEEK_API_KEY` with no config fallback fails cleanly |
| `deepseek_model_normalization_is_case_and_whitespace_tolerant` (proptest) | Fuzz test confirms all known aliases normalize successfully regardless of casing/whitespace |

---

## Summary

DeepSeek CLI enforces single-provider compliance through:

1. **Triple runtime validation** -- provider checks at the CLI entry point, `complete()`, and `complete_streaming()`
2. **Closed model allowlist** -- `normalize_deepseek_model()` rejects anything outside the DeepSeek model family
3. **Closed profile allowlist** -- `normalize_deepseek_profile()` rejects non-DeepSeek profiles
4. **Hardcoded default endpoint** -- `https://api.deepseek.com/chat/completions`
5. **Single auth mechanism** -- Bearer token via `DEEPSEEK_API_KEY`
6. **Zero third-party LLM dependencies** -- no OpenAI, Anthropic, AWS, Google, or Ollama SDKs in `Cargo.toml`
7. **Comprehensive negative tests** -- explicit rejection tests for OpenAI, Anthropic, Ollama, custom providers, unsupported models, and unsupported profiles
