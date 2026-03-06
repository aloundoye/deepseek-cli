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
  - Confirm the selected provider entry under `llm.providers` has the right `base_url`, `models`, and optional `payload_options` for the gateway you are using.
  - Use `codingbuddy status --json` or `codingbuddy doctor --json` to inspect active compatibility transforms and the last applied compatibility shim.
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
  - Run `codingbuddy index build` to rebuild manifest + Tantivy index.
- Plugin faults:
  - Disable plugin via `codingbuddy plugins disable <id>`.
  - Re-run core workflow without plugins.
- Local ML failures:
  - **Embeddings backend unavailable**: Falls back to `MockEmbeddings` (deterministic SHA-256 hash, not semantic but provides consistent retrieval). Set `local_ml.enabled=false` to disable entirely.
  - **Vector index corruption**: Delete `.codingbuddy/vector_index.sqlite` and restart — lazy indexing rebuilds on next query.
  - **Candle model load failure**: Check `local_ml.cache_dir` permissions and disk space. Falls back to mock backend automatically.
  - **Retrieval returns irrelevant results**: Rebuild index with `codingbuddy index --hybrid doctor`. Check `local_ml.chunker.window_size` settings.
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
- Releases are immutable and versioned; keep at least one prior stable GitHub release.
- Downgrade by reinstalling a previous tag:
  - macOS/Linux: `bash scripts/install.sh --version vPREVIOUS --repo aloundoye/codingbuddy`
  - Windows: `./scripts/install.ps1 -Version vPREVIOUS -Repo aloundoye/codingbuddy`
- Package manager rollback:
  - No Homebrew or Winget automation exists in this repo today. If you distribute package-manager manifests externally, update those manually after the GitHub release is verified.
- Verify rollback with:
  - `codingbuddy --json status`
  - `cargo test --workspace --all-targets`
  - `codingbuddy replay list --limit 20` (ensure replay ledger is intact after downgrade)

## Replay integrity checks
- `codingbuddy replay run --session-id <id> --deterministic` validates monotonic event sequencing.
- `codingbuddy replay list [--session-id <id>] [--limit N]` inspects recent replay cassette records and deterministic metadata.
- Strict replay fails closed when tool proposals/approvals do not have matching tool results.
- Replay strict mode never executes tools or provider calls; it rehydrates from event history only.

## Workflow inventory
- `ci.yml`
  - Runs `cargo fmt --all -- --check`
  - Runs `cargo clippy --workspace --all-targets -- -D warnings`
  - Runs split test lanes on Linux and Windows
  - Runs macOS smoke build/status checks
  - Runs Linux conformance scripts
  - Runs installer dry-runs on Unix and Windows
- `release.yml`
  - Watches pushes to `main`
  - Skips if the version already has a GitHub release
  - Builds `codingbuddy` artifacts for Linux, macOS, and Windows
  - Publishes `checksums.txt`
  - Creates the `vX.Y.Z` tag and GitHub release
- `benchmark-live.yml`
  - Runs ignored live benchmark suites on a manual or nightly path
  - Produces DeepSeek live reports, optional reference-lane reports, and comparison artifacts when reference secrets are configured

## Production drill cadence
- Weekly manual checks:
  - Run `cargo test --workspace --all-targets`
  - Run `./scripts/run_coding_quality_benchmark.sh`
  - Review the latest live benchmark artifact from `benchmark-live.yml` when provider credentials are configured
  - On Linux, run:
    - `bash scripts/parity_regression_check.sh`
    - `bash scripts/runtime_conformance_scan.sh`
- Before each release:
  - Ensure `ci.yml` is green on `main`
  - Ensure `benchmark-live.yml` has a recent successful manual/nightly run if live provider credentials are configured
  - Confirm installer dry-runs either from CI artifacts/logs or locally:
    - `bash scripts/install.sh --dry-run --version vX.Y.Z --repo aloundoye/codingbuddy`
    - `./scripts/install.ps1 -DryRun -Version vX.Y.Z -Repo aloundoye/codingbuddy`
  - If production API credentials are available, run a bounded manual smoke in a disposable workspace. The live benchmark workflow is still non-blocking and does not replace a final release smoke.

## Telemetry policy
- Telemetry is opt-in only (`[telemetry].enabled=false` by default).
- If telemetry endpoint causes instability, disable telemetry and continue local logging.
