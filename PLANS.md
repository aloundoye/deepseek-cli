# DeepSeek CLI ExecPlan (100% Spec Completion)

## Authoritative Sources
1. `/Users/aloutndoye/Workspace/deepseek-cli/specs.md` (primary)
2. `/Users/aloutndoye/Workspace/deepseek-cli/README.md`
3. `/Users/aloutndoye/Workspace/deepseek-cli/docs/RELEASE.md`
4. `/Users/aloutndoye/Workspace/deepseek-cli/docs/OPERATIONS.md`
5. `/Users/aloutndoye/Workspace/deepseek-cli/config.example.toml`
6. `/Users/aloutndoye/Workspace/deepseek-cli/config.example.json`
7. `/Users/aloutndoye/Workspace/deepseek-cli/scripts/install.sh`
8. `/Users/aloutndoye/Workspace/deepseek-cli/scripts/install.ps1`
9. `/Users/aloutndoye/Workspace/deepseek-cli/.github/workflows/ci.yml`
10. `/Users/aloutndoye/Workspace/deepseek-cli/.github/workflows/release.yml`

Conflict rule: if behavior/docs differ, `specs.md` wins.

## Locked Defaults
- Strict online by default: `llm.offline_fallback=false`.
- Rich TUI default for interactive `deepseek chat`.
- Claude-first parity behavior benchmark, DeepSeek branding/contracts only.
- Rust `edition=2024` and `rust-version=1.93` remain pinned.

## Scope Checklist
- [x] 2.1 Core runtime parity
- [x] 2.2 Filesystem tools parity
- [x] 2.3 Git integration parity
- [x] 2.4 Slash command parity
- [x] 2.5 MCP parity
- [x] 2.6 Subagent parity
- [x] 2.7 UX shortcuts parity
- [x] 2.8 Config system parity
- [x] 2.9 Skills/hooks parity
- [x] 2.10 Advanced capabilities parity
- [x] 2.11 Permission/safety parity
- [x] 2.12 DeepSeek enhancements parity

## Milestones

### M0 Spec Freeze + Plan Tracker
Status: [x] Completed

Scope:
- Create/maintain this `PLANS.md` as living tracker.
- Normalize implementation/docs to DeepSeek-only wording.

Verification:
- `cargo check --workspace`
- `cargo test --workspace --all-targets`

### M1 Reliability Baseline + Dependency Refresh
Status: [x] Completed

Scope:
- Update workspace dependencies to latest compatible versions.
- Preserve Rust 1.93 compatibility and edition 2024.
- Keep workspace green (fmt/clippy/tests/build).

Verification:
- `cargo update`
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --all-targets`
- `cargo build --workspace`

### M2 Planner/Executor Production State Machine
Status: [x] Completed

Scope:
- Replace heuristic fallback path with schema-driven planner + deterministic repair loop.
- Enforce explicit session transitions and bounded revision policy.
- Add failure-injection coverage.

Verification:
- `cargo test -p deepseek-agent`
- replay/transition tests

### M3 Full TUI Slash/UX Parity
Status: [x] Completed

Scope:
- Make TUI path execute real slash behaviors (`/compact`, `/rewind`, `/teleport`, `/remote-env`, `/mcp`, `/memory`, `/status`, `/effort`).
- Keep keybindings and panes aligned with spec.
- Make TUI default in interactive chat.

Verification:
- `cargo test -p deepseek-ui`
- `cargo test -p deepseek-cli --test cli_json`

### M4 LLM Pipeline Hardening
Status: [x] Completed

Scope:
- Strict-online default and explicit diagnostics.
- Preserve retry/backoff/stream handling.
- Keep provider abstraction additive.

Verification:
- `cargo test -p deepseek-llm`
- `cargo test -p deepseek-cli --test cli_json`

### M5 MCP Runtime Completion
Status: [x] Completed

Scope:
- Replace stdio placeholder discovery with real stdio protocol handshake path.
- Keep HTTP discovery and add resilient fallback behavior.
- Emit change notices for dynamic tool changes.

Verification:
- `cargo test -p deepseek-mcp`

### M6 Tooling/Reference Resolver Hardening
Status: [x] Completed

Scope:
- Added explicit `@file:` / `@dir:` parsing plus line-range expansion in planner prompt references.
- Implemented `ignore`-based filesystem walking for `fs.glob`, `fs.grep`, and index scanning with `respectGitignore` behavior.
- Added binary/mime-safe read behavior tests and gitignore parity tests.
- Added auto-checkpoint creation for `fs.edit` and `fs.write` operations before mutation.

Verification:
- `cargo test -p deepseek-tools`
- `cargo test -p deepseek-agent`

### M7 Git Workflow Depth
Status: [x] Completed

Scope:
- Added branch/ahead/behind/staged/unstaged/conflict summaries for `deepseek git status`.
- Added conflict-assistant suggestions for `deepseek git resolve --strategy list`.
- Preserved additive JSON contract for automation.

Verification:
- `cargo test -p deepseek-cli --test cli_json`

### M8 Skills/Hooks/Plugins Reliability
Status: [x] Completed

Scope:
- Added `/skills` slash command parity in line mode and TUI (`list|reload|run`).
- Added TUI keybinding-file loading support (`~/.deepseek/keybindings.json` and overrides).
- Preserved plugin + hooks contracts and compatibility tests.

Verification:
- `cargo test -p deepseek-tools`
- `cargo test -p deepseek-skills`
- `cargo test -p deepseek-hooks`

### M9 Subagents + Guarded Autopilot
Status: [x] Completed

Scope:
- Added bounded retry budget per subagent task.
- Added deterministic role-first merge ordering and stable merged output summaries.
- Added subagent retry and deterministic merge tests.

Verification:
- `cargo test -p deepseek-subagent`
- `cargo test -p deepseek-cli --test cli_json`

### M10 Deterministic Replay + Perf/Profiling
Status: [x] Completed

Scope:
- Added strict replay validation for monotonic sequence and tool proposal/approval/result integrity.
- Replay now fails closed in strict mode when event artifacts are incomplete.
- Preserved profile/cost outputs and replay cassette persistence.
- Added non-urgent off-peak defer execution path with bounded/declarative scheduling controls.
- Added 1M context-window configuration + automatic transcript compaction near threshold.

Verification:
- `cargo test -p deepseek-store`
- `cargo test -p deepseek-cli --test cli_json`

### M11 Distribution/Install GA
Status: [x] Completed

Scope:
- Kept 6-target release artifact matrix with checksums/SBOM.
- Added CI target-build matrix for all 6 release targets.
- Added release provenance attestation step and retained Homebrew/Winget publish workflows.

Verification:
- workflow lint/smoke + installer dry-runs

### M12 Security/Ops/Docs GA Signoff
Status: [x] Completed

Scope:
- Added command-injection guardrails and wildcard allowlist support in policy engine.
- Strengthened path checks and dangerous command prefix denial.
- Added configurable `block_paths` + `redact_patterns` contracts with healthcare-friendly defaults.
- Added multilingual response setting propagation and global+project memory injection into planner context.
- Updated README/config/docs for slash skills, replay guarantees, off-peak defer behavior, context window compaction, provenance, and operations guidance.

Verification:
- full RC gate:
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace --all-targets`
  - `cargo build --workspace`
  - `cargo build --release --workspace`

### M13 Parity Gap Closure Loop
Status: [x] Completed (current iteration)

Scope:
- Added first-class permission controls with `deepseek permissions show|set` and `/permissions` in line/TUI chat.
- Added richer live autopilot control semantics via `deepseek autopilot status --follow --samples --interval-seconds`.
- Expanded benchmark parity by allowing external benchmark suites (`--benchmark-suite`) with per-case quality gates.

Verification:
- `cargo test -p deepseek-cli --test cli_json`
- `cargo test -p deepseek-ui`

### M14 Parity Gap Closure Loop (Reasoning/Subagents/Benchmarks)
Status: [x] Completed (current iteration)

Scope:
- Added bounded planner quality self-critique and repair retries before schema fallback.
- Added subagent delegated read-only tool probes (explore/plan/task role mapping) merged into output.
- Added benchmark quality-rate gating, baseline comparison/regression checks, and peer report ranking (`--benchmark-compare`).

Verification:
- `cargo test -p deepseek-agent`
- `cargo test -p deepseek-cli --test cli_json`

### M15 Parity Gap Closure Loop (Feedback Memory + Arbitration + Corpus Matrix)
Status: [x] Completed (current iteration)

Scope:
- Added planner verification-feedback memory in context and multi-pass feedback-alignment repair retries.
- Upgraded subagents from read-only probes to bounded delegated tool execution.
- Added subagent conflict-aware merge arbitration summaries for shared target files.
- Expanded benchmark peer comparison to include corpus mismatch warnings and per-case matrix.

Verification:
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --all-targets`
- `cargo build --workspace`
- `cargo build --release --workspace`

### M16 Parity Gap Closure Loop (Strategy Memory + Pack Workflow)
Status: [x] Completed (current iteration)

Scope:
- Added persisted planner strategy memory from successful runs and strategy-memory context injection in planning.
- Added scoped subagent delegated execution policies by role (bounded tool calls and task artifact writes).
- Added benchmark pack command workflow (`benchmark list-packs|show-pack|import-pack`) and `profile --benchmark-pack`.
- Extended benchmark side-by-side report with case-level matrix and corpus compatibility warnings.

Verification:
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --all-targets`
- `cargo build --workspace`
- `cargo build --release --workspace`

### M17 Parity Gap Closure Loop (Scored Strategies + Signed Scorecards)
Status: [x] Completed (current iteration)

Scope:
- Added planner strategy-memory scoring and pruning to suppress chronically low-performing strategy memories.
- Added approval-aware delegated subagent retries with bounded read-only fallback when delegated edits are blocked.
- Upgraded subagent merge arbitration with per-target scoring and rationale output for conflict resolution.
- Added seeded benchmark determinism plus signed corpus/execution manifests and benchmark scorecards.

Verification:
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --all-targets`
- `cargo build --workspace`
- `cargo build --release --workspace`

### M18 Parity Gap Closure Loop (Objective Memory + Matrix Parity)
Status: [x] Completed (current iteration)

Scope:
- Added long-horizon objective-outcome memory with confidence scoring/pruning and planning-context reinjection.
- Added matrix-scale benchmark execution (`benchmark run-matrix`) spanning benchmark packs and external suites.
- Added aggregate matrix scorecards (weighted success/quality/p95) with manifest/corpus compatibility warnings.
- Added matrix peer-comparison ranking with manifest-coverage and case-count diagnostics.

Verification:
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --all-targets`
- `cargo build --workspace`
- `cargo build --release --workspace`

## Data/Storage Contracts
- Canonical append-only event log: `.deepseek/events.jsonl`.
- Projections remain additive only.
- `schema_migrations` tracks forward-only schema updates.
- Replay strict mode: deterministic playback only; fail closed on missing cassette/event artifacts.
- Index freshness contract: `fresh|stale|corrupt` with deterministic rebuild on corruption.

## API/CLI Contracts
- Keep command families stable and additive.
- TUI default for interactive chat sessions.
- Slash command parity in both line and TUI modes.
- Config precedence remains:
  1. `~/.deepseek/settings.json`
  2. `.deepseek/settings.json`
  3. `.deepseek/settings.local.json`
  4. legacy `.deepseek/config.toml`

## Failure Modes / Retries / Rollback
- LLM outage/rate-limits: bounded exponential backoff, explicit surfaced failures, no silent offline switch.
- MCP outages: preserve cached toolset, mark stale, continue core runtime.
- Tool/patch conflicts: explicit events and retained artifacts.
- Index corruption: detect and rebuild deterministically.
- Plugin/hook faults: isolate and continue runtime.
- Release rollback: immutable versioned binaries and documented downgrade path.

## Risks and Chosen Defaults
- Risk: strict online default may break local test flows without key.
  - Default chosen: explicit failure; tests use offline_fallback override where needed.
- Risk: full MCP stdio parity depends on external server behavior variance.
  - Default chosen: standards-based handshake + resilient fallback to metadata tools.
- Risk: TUI parity regressions across terminals.
  - Default chosen: preserve line mode fallback and expand TUI snapshots.
