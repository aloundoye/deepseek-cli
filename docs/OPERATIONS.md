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
  - Verify fallback behavior (`offline_fallback=true` if desired).
  - Increase retry/backoff in `[llm]`.
- Tool execution denied:
  - Review `[policy].allowlist` and approval modes.
  - Confirm no path traversal or secret-path attempts.
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

## Telemetry policy
- Telemetry is opt-in only (`[telemetry].enabled=false` by default).
- If telemetry endpoint causes instability, disable telemetry and continue local logging.
