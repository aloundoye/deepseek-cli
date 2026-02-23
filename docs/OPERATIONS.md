# DeepSeek CLI Operations Playbook

## Incident severity
- SEV1: data loss, unsafe command execution, or widespread CLI crash.
- SEV2: degraded model/runtime behavior with workaround.
- SEV3: non-blocking regression.

## Immediate response
1. Stop rollout and pin users to last known good release.
2. Capture failing command, logs (`.deepseek/observe.log`), and event stream (`.deepseek/events.jsonl`).
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
  - Inspect patch metadata in `.deepseek/patches/<id>.json`.
  - Re-stage from current workspace state.
- Index corruption:
  - Run `deepseek index build` to rebuild manifest + Tantivy index.
- Plugin faults:
  - Disable plugin via `deepseek plugins disable <id>`.
  - Re-run core workflow without plugins.

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
