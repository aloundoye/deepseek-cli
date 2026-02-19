# DeepSeek CLI Production Readiness Checklist

Date: 2026-02-19

This checklist closes M11 launch gates from `PLANS.md` and provides a repeatable release signoff contract.

## 1) Mandatory CI Gates

- `ci.yml` (fmt/clippy/tests/build + CLI smoke)
- `replay-regression.yml` (deterministic replay cassette roundtrip)
- `performance-gates.yml` (benchmark SLO thresholds + strict SLO matrix gate)
- `security-gates.yml` (`cargo audit`, `cargo deny`, secret scan)
- `release-readiness.yml` (manual installer + rollback rehearsal evidence)
- `live-deepseek-smoke.yml` (secret-gated live DeepSeek API and first-token latency gate)

## 2) Launch Signoff Checklist

- [x] Workspace quality gate passes:
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace --all-targets`
  - `cargo build --workspace`
  - `cargo build --release --workspace`
- [x] Replay regression gate passes and cassette metadata is persisted (`deepseek replay list` non-empty after replay run).
- [x] Benchmark SLO gate passes (`--benchmark-max-p95-ms`, strict matrix compatibility).
- [x] Security gate passes (`cargo audit`, `cargo deny`, gitleaks).
- [x] Release readiness drill passes (installer dry-run + rollback rehearsal output).
- [ ] Live DeepSeek smoke passes with production-scoped key and first-token latency within budget.

## 3) Evidence Artifacts

- Replay evidence:
  - `replay_run.json`
  - `replay_list.json`
- Performance evidence:
  - `benchmark.json`
  - `matrix.json`
  - `matrix.md`
- Release drill evidence:
  - installer dry-run logs
  - rollback rehearsal JSON output
- Security evidence:
  - audit/deny logs
  - secret scan report
- Live smoke evidence:
  - ask/plan/autopilot JSON outputs
  - first-token latency measurement

## 4) Failure Policy

- Any failed mandatory gate blocks release tagging.
- Security advisories with no safe mitigation require explicit documented exception and owner signoff.
- Live smoke failures require retry only after root-cause note is attached in release prep issue.
