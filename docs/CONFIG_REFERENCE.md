# Configuration Reference

CodingBuddy merges configuration from these sources (later entries win):

1. Legacy fallback: `.codingbuddy/config.toml`
2. User config: `~/.codingbuddy/settings.json`
3. Project config: `.codingbuddy/settings.json`
4. Project-local overrides: `.codingbuddy/settings.local.json`

Run `codingbuddy config show` to view the merged configuration. API keys are redacted in output.

---

## `llm` — Model & Provider

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `base_model` | string | `"deepseek-chat"` | Default model for normal tool-use turns |
| `max_think_model` | string | `"deepseek-reasoner"` | Thinking model used for complex-task escalation, `/thinking`, and `think_deeply` |
| `provider` | string | `"deepseek"` | LLM provider identifier |
| `providers` | object | built-in map | Named provider definitions (`deepseek`, `openai-compatible`, `ollama`) |
| `base_url` | string | `"https://api.deepseek.com"` | Legacy top-level provider base URL fallback |
| `capability_overrides` | object | `{}` | Optional per-family/per-model capability overrides |
| `profile` | string | `"v3_2"` | Active model profile |
| `context_window_tokens` | int | `128000` | Maximum context window (raise for long-context use) |
| `temperature` | float? | per-model | Sampling temperature (auto-tuned per model family: deepseek 0.0, qwen 0.55, gemini 1.0, default 0.2). User value overrides. |
| `endpoint` | string | `"https://api.deepseek.com/chat/completions"` | API endpoint URL |
| `api_key` | string? | `null` | API key (prefer `api_key_env` or env var) |
| `api_key_env` | string | `"DEEPSEEK_API_KEY"` | Environment variable for API key |
| `fast_mode` | bool | `false` | Skip reasoning model in simple cases |
| `prompt_cache_enabled` | bool | `true` | Enable prompt caching |
| `timeout_seconds` | int | `60` | Per-request timeout |
| `max_retries` | int | `3` | Maximum retry attempts |
| `retry_base_ms` | int | `400` | Exponential backoff base |
| `stream` | bool | `true` | Enable streaming responses |

### `llm.providers.<id>`

Each named provider entry has these fields:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `kind` | string | provider id | Provider family (`deepseek`, `openai-compatible`, `ollama`) |
| `base_url` | string | none | Provider base URL used to construct chat/completions requests |
| `api_key_env` | string | none | Environment variable used to read the provider API key |
| `openai_compat_prefix` | bool | `false` | Add `/v1` prefix for SDK-compatible gateways |
| `payload_options` | object/null | `null` | Provider-wide request payload shim merged before request-level provider namespaces |
| `models.chat` | string | none | Chat/tool-use model id |
| `models.reasoner` | string? | `null` | Optional higher-thinking model id |

Payload merge order at request time:

1. provider `payload_options`
2. request `provider_options.default`
3. request `provider_options.<active-provider>`
4. request `provider_options.<model-family>`
5. request `provider_options.<exact-model>`

### `llm.capability_overrides`

Override registry keys:
- Family key: `<family>` or `<provider>@<family>` (example: `qwen`, `ollama@qwen`)
- Model key: exact model id or prefix wildcard, optionally provider scoped (example: `qwen2.5-coder:*`, `ollama@deepseek-r1:*`)

Override fields:
- `supports_tool_calling`
- `supports_tool_choice`
- `supports_parallel_tool_calls`
- `supports_reasoning_mode`
- `supports_thinking_config`
- `supports_streaming_tool_deltas`
- `supports_fim`
- `max_safe_tool_count`
- `preferred_edit_tool` (`fs-edit`, `multi-edit`, `patch-direct`)

## `agent_loop` — Agent Loop

Current runtime:
- Default path: tool-use loop (`think -> act -> observe`)
- Complex-task overlay: `Explore -> Plan -> Execute -> Verify` with per-phase tool filtering

The older `Architect -> Editor -> Apply -> Verify` pipeline has been removed. Some config keys from that era are still accepted for backward-compatible parsing, but they do not define the primary runtime path anymore.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tool_loop_max_turns` | int | `50` | Maximum LLM calls in the tool-use loop |
| `verify_timeout_seconds` | int | `60` | Timeout for verification commands |
| `apply_strategy` | string | `"auto"` | Patch apply strategy (`auto` or `three_way`) |

### `agent_loop.failure_classifier`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `repeat_threshold` | int | `2` | Consecutive same-error count to escalate |
| `similarity_threshold` | float | `0.8` | Jaccard similarity for "same error" detection |
| `fingerprint_lines` | int | `40` | Max error lines to fingerprint |

### `agent_loop.safety_gate`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_files_without_approval` | int | `8` | File count threshold for approval prompt |
| `max_loc_without_approval` | int | `600` | Line-of-code delta threshold for approval |

### `agent_loop.context_bootstrap_*`

Controls for automatic workspace context injection:

| Key | Default | Description |
|-----|---------|-------------|
| `context_bootstrap_enabled` | `true` | Enable automatic context gathering |
| `context_bootstrap_max_tree_entries` | `120` | Max directory tree entries |
| `context_bootstrap_max_readme_bytes` | `24000` | Max README content bytes |
| `context_bootstrap_max_manifest_bytes` | `16000` | Max package manifest bytes |
| `context_bootstrap_max_repo_map_lines` | `80` | Max repo-map lines |
| `context_bootstrap_max_audit_findings` | `20` | Max static analysis findings |

### `agent_loop` legacy compatibility keys

These keys remain in the schema so older configs still parse cleanly after the old pipeline removal. They are not the primary control surface for the current tool-use loop.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_iterations` | int | `6` | Legacy pipeline iteration cap; retained for compatibility |
| `architect_parse_retries` | int | `2` | Legacy architect-step parse retry count; retained for compatibility |
| `editor_parse_retries` | int | `2` | Legacy editor-step parse retry count; retained for compatibility |
| `max_editor_apply_retries` | int | `3` | Legacy editor/apply retry cap; retained for compatibility |
| `max_files_per_iteration` | int | `12` | Legacy per-iteration file cap from the removed pipeline |
| `max_file_bytes` | int | `200000` | Legacy pipeline context/file-size cap |
| `max_diff_bytes` | int | `400000` | Legacy pipeline diff-size cap |
| `max_context_requests_per_iteration` | int | `3` | Legacy pipeline context-fetch cap |
| `max_context_range_lines` | int | `400` | Legacy pipeline context line-range cap |

## `policy` — Permissions & Safety

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `approve_edits` | string | `"ask"` | Edit approval mode (`ask`, `always`, `never`) |
| `approve_bash` | string | `"ask"` | Bash approval mode |
| `allowlist` | string[] | `["rg","git status",...]` | Allowed command prefixes (supports `*` wildcards) |
| `block_paths` | string[] | `[".env",".ssh",...]` | Glob patterns for blocked paths |
| `redact_patterns` | string[] | `[...]` | Regex patterns for credential redaction |
| `sandbox_mode` | string | `"allowlist"` | Sandbox enforcement mode |
| `sandbox_wrapper` | string? | `null` | Optional OS sandbox command template |
| `lint_after_edit` | string? | `null` | Optional lint command to run after each edit |

## `plugins` — Plugin System

Enabled plugin commands are exposed to the agent runtime as callable tools named
`plugin__<plugin_id>__<command>`. Invoking one returns the rendered command prompt
template so it can be used in the active session.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable plugin discovery and execution |
| `search_paths` | string[] | `[".codingbuddy/plugins",".plugins"]` | Directories to scan for plugins |
| `enable_hooks` | bool | `false` | Enable plugin hook execution |

### `plugins.catalog`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable catalog sync |
| `index_url` | string | `".codingbuddy/plugins/catalog.json"` | URL or path to catalog index |
| `signature_key` | string | `"deepseek-local-dev-key"` | HMAC key for catalog signature verification |
| `refresh_hours` | int | `24` | Catalog refresh interval |

## `skills` — Prompt Skills

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `paths` | string[] | `[".codingbuddy/skills","~/.codingbuddy/skills"]` | Skill search directories |
| `hot_reload` | bool | `true` | Auto-reload skills on change |

## `usage` — Cost Tracking

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `show_statusline` | bool | `true` | Show cost in status line |
| `cost_per_million_input` | float | `0.27` | Input token cost (USD) |
| `cost_per_million_output` | float | `1.1` | Output token cost (USD) |

## `context` — Context Management

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `auto_compact_threshold` | float | `0.86` | Context window usage threshold for auto-compact |
| `compact_preview` | bool | `true` | Preview before compacting |

## `autopilot` — Unattended Loop

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_max_consecutive_failures` | int | `10` | Max consecutive failures before autopilot stops |
| `heartbeat_interval_seconds` | int | `5` | Status file heartbeat interval |
| `persist_checkpoints` | bool | `true` | Save checkpoints during autopilot runs |

## `replay` — Session Replay

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `strict_mode` | bool | `true` | Strict deterministic replay mode |

## `index` — Code Index

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable local code index |
| `engine` | string | `"tantivy"` | Index engine backend |
| `watch_files` | bool | `true` | Auto-rebuild index on file changes |

## `ui` — Interface

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_tui` | bool | `true` | Enable TUI in interactive terminals |
| `keybindings_path` | string | `"~/.codingbuddy/keybindings.json"` | Custom keybindings file |
| `reduced_motion` | bool | `false` | Disable animations |
| `statusline_mode` | string | `"minimal"` | Status line verbosity |
| `thinking_visibility` | string | `"concise"` | TUI reasoning display mode (`concise`, `raw`) |
| `phase_heartbeat_ms` | int | `5000` | Progress heartbeat interval while a phase is active |
| `mission_control_max_events` | int | `400` | Maximum retained mission-control timeline entries |
| `image_fallback` | string | `"open"` | Image display fallback (`open`, `path`, `none`) |
| `handoff_base_url` | string? | `null` | Optional base URL for teleport handoff links |

## `budgets` — Resource Limits

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_turn_duration_secs` | int | `300` | Maximum seconds per turn |
| `max_reasoner_tokens_per_session` | int | `1000000` | Thinking token budget per session |

## `theme` — Colors

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `primary` | string | `"Cyan"` | Primary accent color |
| `secondary` | string | `"Yellow"` | Secondary accent color |
| `error` | string | `"Red"` | Error highlight color |

## `local_ml` — Local ML Intelligence

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable local ML features (retrieval, privacy, autocomplete) |
| `device` | string | `"auto"` | Compute device (`auto`, `cpu`, `cuda`, `metal`) |
| `cache_dir` | string | `".codingbuddy/models"` | Directory for cached model weights and runtime artifacts |

### `local_ml.embeddings`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model_id` | string | `"jinaai/jina-embeddings-v2-base-code"` | HuggingFace model ID for embeddings |
| `dimension` | int | `384` | Embedding vector dimension |
| `batch_size` | int | `32` | Batch size for embedding computation |

### `local_ml.completion`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model_id` | string | `"qwen2.5-coder-3b"` | Model for local code completion |
| `max_tokens` | int | `64` | Max tokens for completion |
| `temperature` | float | `0.2` | Sampling temperature |

### `local_ml.autocomplete`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable ghost text in TUI (when local_ml is enabled) |
| `debounce_ms` | int | `200` | Debounce delay before triggering completion |
| `max_tokens` | int | `64` | Max tokens for autocomplete |
| `temperature` | float | `0.2` | Sampling temperature |

### `local_ml.privacy`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable 3-layer privacy detection on tool outputs |
| `path_globs` | string[] | `[".env*","*credentials*",...]` | File path patterns to flag as sensitive |
| `content_patterns` | string[] | `["(?i)api.?key","(?i)secret",...]` | Content regex patterns for sensitive data |
| `policy` | string | `"Redact"` | Default privacy policy (`BlockCloud`, `Redact`, `LocalOnlySummary`) |

### `local_ml.retrieval`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable hybrid retrieval (when local_ml is enabled) |
| `max_results` | int | `10` | Max chunks returned per query |
| `context_budget_pct` | float | `0.15` | Fraction of context window for retrieval results |
| `rrf_k` | float | `60.0` | Reciprocal Rank Fusion constant |

### `local_ml.chunker`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `window_size` | int | `512` | Chunk window size in tokens |
| `overlap` | int | `64` | Overlap between adjacent chunks |
| `max_file_size` | int | `200000` | Skip files larger than this |

## `lsp` — Post-Edit Validation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable post-edit diagnostics (runs language checks after `fs_edit`/`fs_write`) |
| `languages` | object | `{}` | Per-language enable/disable. Keys: `"rust"`, `"typescript"`, `"python"`, `"go"`. Missing languages default to enabled. |

Example — disable Python checks:

```json
{
  "lsp": {
    "enabled": true,
    "languages": {
      "python": false
    }
  }
}
```

Supported checks:
- **Rust**: `cargo check --message-format=json` (parsed JSON diagnostics)
- **TypeScript**: `tsc --noEmit --pretty false` (parsed TSC output)
- **Python**: `python3 -m py_compile` (syntax errors)
- **Go**: `go vet` (vet errors)

Gracefully skipped when the required toolchain is not installed.

## `experiments` — Feature Flags

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `visual_verification` | bool | `false` | Enable screenshot-based verification |
| `wasm_hooks` | bool | `false` | Enable WASM plugin hooks |

## `telemetry`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable anonymous telemetry |
| `endpoint` | string? | `null` | Custom telemetry endpoint |

## `scheduling` — Off-Peak

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `off_peak` | bool | `false` | Enable off-peak scheduling |
| `off_peak_start_hour` | int | `0` | Off-peak window start (UTC) |
| `off_peak_end_hour` | int | `6` | Off-peak window end (UTC) |
| `defer_non_urgent` | bool | `false` | Defer non-urgent requests |
| `max_defer_seconds` | int | `0` | Maximum defer duration |
