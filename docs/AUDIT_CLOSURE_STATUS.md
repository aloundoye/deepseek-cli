# Audit Closure Status

This is the canonical tracked closure document for the three-codebase audit. It replaces the ignored `plan.md` as the source of truth for what was implemented.

## Scope

Audit inputs:
- `deepseek-cli`
- `opencode`
- `ollama`

Closure targets:
- coding capability parity work for weaker models
- provider/gateway compatibility parity
- local runtime scheduling and lifecycle parity
- operator UX parity
- maintainability parity in the main hotspot files
- comparative measurement parity
- contract/docs/workflow parity

## Closure Summary

All six implementation batches from the audit closure plan are now represented in tracked code:

1. Provider and session parity
   - request-level `provider_options`
   - provider-level `payload_options`
   - ordered compatibility pipeline at the LLM boundary
   - unsupported input degradation instead of hard failure where safe
   - compatibility tracking in responses, events, status, and doctor surfaces

2. Local runtime parity
   - explicit live scheduler snapshot
   - explicit runner lifecycle states
   - serialized runner loading
   - persisted queue/load visibility and stale/live rendering rules
   - stricter eviction/recovery behavior under memory pressure

3. Operator UX parity
   - provider compatibility summaries in `status` and `doctor`
   - last-applied compatibility visible in operator JSON
   - runtime queue/load state visible in CLI and TUI status surfaces
   - failure-path context enriched with provider/family/transform metadata

4. Maintainability parity
   - `crates/codingbuddy-cli/src/commands/chat.rs` reduced below 2,000 LOC
   - `crates/codingbuddy-ui/src/lib.rs` reduced below 2,000 LOC
   - `crates/codingbuddy-tools/src/lib.rs` reduced below 2,000 LOC
   - new focused modules extracted for chat/UI/tools hot paths

5. Comparative measurement parity
   - deterministic suites: `coding-quality-core`, `coding-quality-repo`
   - ignored live suites for real providers/models
   - report metadata for provider/profile/lane
   - comparison artifacts with pass/build/test/cost/verification deltas
   - model-specific baseline path support under `docs/benchmarks/`

6. Final contract sweep
   - tracked live benchmark workflow at `.github/workflows/benchmark-live.yml`
   - tracked benchmark baselines for current deterministic suites
   - reconciled public docs/config samples with provider compatibility, runtime diagnostics, and benchmark lanes

## Hotspot Size Target

Current hotspot files are all below the audit target of 2,000 LOC:

- `crates/codingbuddy-cli/src/commands/chat.rs`
- `crates/codingbuddy-ui/src/lib.rs`
- `crates/codingbuddy-tools/src/lib.rs`

## Benchmark Baselines

Tracked deterministic baselines:
- `docs/benchmarks/coding-quality-core.scripted-tool-loop.baseline.json`
- `docs/benchmarks/coding-quality-repo.scripted-tool-loop.baseline.json`

Legacy compatibility baseline retained:
- `docs/benchmarks/coding_quality_baseline.json`

## Live Benchmark Path

Tracked non-blocking live path:
- `.github/workflows/benchmark-live.yml`

Behavior:
- manual or nightly execution
- DeepSeek live lane when `DEEPSEEK_API_KEY` is configured
- optional reference lane when reference secrets/vars are configured
- comparison artifact generation when both lanes run for the same suite

## Validation Contract

Required validation remains:
- `cargo test --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo xwin test --workspace --all-targets --target x86_64-pc-windows-msvc --no-run`
- native Linux CI execution as the source of truth for Linux runtime coverage

## Remaining Operational Caveat

Native Linux execution still depends on CI runners because this macOS host does not provide a working Linux C cross toolchain for full runtime validation.
