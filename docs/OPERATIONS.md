# CodingBuddy Operations Playbook

## Incident severity
- SEV1: data loss, unsafe command execution, or widespread CLI crash.
- SEV2: degraded model/runtime behavior with workaround.
- SEV3: non-blocking regression.

## Immediate response
1. Stop rollout and pin users to last known good release.
2. Capture failing command, logs (`.codingbuddy/observe.log`), and event stream (`.codingbuddy/events.jsonl`).
3. Open incident issue with timeline and affected versions.

## Common failure modes and responses
- LLM outage / rate limit:
  - Verify `DEEPSEEK_API_KEY` or `llm.api_key` is configured and endpoint/provider/profile settings are correct.
  - Confirm `llm.provider=deepseek`; unsupported providers fail fast.
  - Increase retry/backoff in `[llm]`.
  - For non-urgent workloads, consider `[scheduling].off_peak=true` and `[scheduling].defer_non_urgent=true`.
- Tool execution denied:
  - Review `[policy].allowlist` and approval modes.
  - Validate wildcard patterns (for example `npm *`) and remove broad patterns if unexpected commands pass.
  - Commands containing shell metacharacters (`&&`, `;`, `|`, backticks, `$(`) are blocked by policy.
  - Confirm no path traversal or secret-path attempts.
  - For healthcare deployments, tighten `[policy].block_paths` and `[policy].redact_patterns` for PHI/secret controls.
- Patch conflicts / base drift:
  - Inspect patch metadata in `.codingbuddy/patches/<id>.json`.
  - Re-stage from current workspace state.
- Index corruption:
  - Run `deepseek index build` to rebuild manifest + Tantivy index.
- Plugin faults:
  - Disable plugin via `deepseek plugins disable <id>`.
  - Re-run core workflow without plugins.
- Local ML failures:
  - **Embeddings backend unavailable**: Falls back to `MockEmbeddings` (deterministic SHA-256 hash, not semantic but provides consistent retrieval). Set `local_ml.enabled=false` to disable entirely.
  - **Vector index corruption**: Delete `.codingbuddy/vector_index.sqlite` and restart — lazy indexing rebuilds on next query.
  - **Candle model load failure**: Check `local_ml.cache_dir` permissions and disk space. Falls back to mock backend automatically.
  - **Retrieval returns irrelevant results**: Rebuild index with `deepseek index --hybrid doctor`. Check `local_ml.chunker.window_size` settings.
  - **Privacy false positives**: Adjust `local_ml.privacy.path_globs` and `local_ml.privacy.content_patterns` in project config.
  - **Ghost text not appearing**: Verify `local_ml.enabled=true` and `local_ml.autocomplete.enabled=true`. Check TUI is active (not `--json` mode).
- Bootstrap context failures:
  - **Context too large**: Reduce `agent_loop.context_bootstrap_max_tree_entries` or `context_bootstrap_max_repo_map_lines`.
  - **Context manager crash**: Set `agent_loop.context_bootstrap_enabled=false` to disable. File an issue with the error log.
- Compaction data loss:
  - LLM-based compaction preserves goals, progress, findings, and modified files using a structured template. If critical context is lost, lower `context.auto_compact_threshold` to delay compaction.
  - If LLM compaction fails (API error, timeout), it falls back to code-based extraction automatically.
- Agent profile mismatch:
  - If tools are unexpectedly unavailable, check if the wrong agent profile was selected. Ask/Context modes restrict to read-only tools. Planning keywords in Code mode restrict to plan-only tools.
  - MCP tools always pass through profile filters.
- Doom loop (agent repeating):
  - If the agent repeats identical tool calls 3+ times, corrective guidance is injected automatically.
  - If it persists, the circuit breaker will disable the tool after 3 failures.
  - Manual intervention: cancel and rephrase the request, or switch to a different chat mode.
- Step snapshot disk usage:
  - Snapshots accumulate in `runtime_dir/snapshots/`. In long sessions, this can grow. Clean manually if needed.
  - Snapshots only store content hashes and 50-line previews, not full file contents.

## Rollback strategy
- Releases are immutable and versioned; keep at least one prior stable release.
- Downgrade by reinstalling a previous tag:
  - macOS/Linux: `bash scripts/install.sh --version vPREVIOUS`
  - Windows: `./scripts/install.ps1 -Version vPREVIOUS`
- Package manager rollback:
  - Homebrew: `brew install deepseek@<version>` (or pin tap formula commit)
  - Winget: `winget install DeepSeek.DeepSeekCLI --version <version>`
- Verify rollback with:
  - `deepseek --json plan "rollback validation"`
  - `cargo test --workspace --all-targets`
  - `deepseek --json replay list --limit 20` (ensure replay ledger is intact after downgrade)

## Replay integrity checks
- `deepseek replay run --session-id <id> --deterministic` validates monotonic event sequencing.
- `deepseek replay list [--session-id <id>] [--limit N]` inspects recent replay cassette records and deterministic metadata.
- Strict replay fails closed when tool proposals/approvals do not have matching tool results.
- Replay strict mode never executes tools or provider calls; it rehydrates from event history only.

## Production drill cadence
- Weekly:
  - Run `performance-gates.yml` and inspect benchmark artifacts for regressions.
  - Run `security-gates.yml` and resolve any advisory/license failures.
- Before each release:
  - Run `release-readiness.yml` and attach artifacts to release notes.
  - Run `live-deepseek-smoke.yml` (with production key scope) and confirm first-token latency SLO.

## Telemetry policy
- Telemetry is opt-in only (`[telemetry].enabled=false` by default).
- If telemetry endpoint causes instability, disable telemetry and continue local logging.
