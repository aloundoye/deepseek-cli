# DeepSeek CLI Parity Gap Matrix (Codex / Aider / Claude Code)

Date: 2026-02-18

This matrix tracks the remaining gap between `specs.md` expectations and practical parity against Codex, Aider, and Claude Code.

## Current Gap Matrix

| Area | Gap | Status |
|---|---|---|
| Core runtime + session model | Claude/Codex-level long-horizon autonomous reliability (multi-turn decomposition and resilient replanning) still trails frontier behavior on hard tasks. | **Partially closed** (objective-outcome memory + long-horizon checkpoint-aware plan scoring + multi-pass plan repair) |
| File/Git tooling depth | Team/enterprise workflows still miss some higher-level orchestration behaviors (e.g., richer PR-native flows and broader conflict automation). | **Partially closed** |
| Live autopilot/background semantics | True in-TUI background handoff was missing for direct prompt/shell detachment. | **Mostly closed** (`Ctrl+B` now launches `/background run-agent` or `/background run-shell`, plus attach tailing and stop controls) |
| Subagent intelligence | Multi-agent conflict-free ownership/dependency scheduling was still too shallow. | **Mostly closed** (role/domain specialization memory + delegated retries + arbitration scoring + explicit phase/lane dependency planning) |
| Permission/safety governance | Team-managed non-overridable permissions were missing. | **Partially closed** (team policy overlay via `DEEPSEEK_TEAM_POLICY_PATH` / `~/.deepseek/team-policy.json` now enforced) |
| Advanced visual/sandbox parity | Visual artifact capture existed but lacked strict analyzers/gates; sandbox mode semantics needed stronger runtime enforcement. | **Partially closed** (`visual list|analyze --strict` + runtime shell sandbox enforcement for `read-only`/`workspace-write`) |
| Real-world benchmark parity | Need broader shared corpus + strict peer comparability at scale for direct Codex/Aider/Claude benchmarking. | **Partially closed** (`--benchmark-compare` + manifest/seed compatibility checks + signed scorecards/manifests + `benchmark run-matrix` aggregate parity runner + strict compare modes + expanded builtin `parity` pack) |

## Executed Plan (This Iteration)

1. Add permissions parity control plane across CLI + slash + persisted config.
2. Add richer live background control semantics for autopilot monitoring.
3. Add external benchmark suite ingestion with measurable per-case quality gates.
4. Validate with fmt/clippy/tests/build.

## Executed Plan (Latest Iteration)

1. Add bounded planner self-critique quality scoring and auto-repair prompt retries.
2. Add delegated subagent read-only tool probes by role (`explore|plan|task`).
3. Add benchmark quality-rate thresholds and baseline regression comparison support.
4. Validate with fmt/clippy/tests/build.

## Executed Plan (Newest Iteration)

1. Add verification-feedback memory into planner context and a second feedback-alignment repair pass.
2. Upgrade subagents to bounded delegated tool execution (including task-role artifact writes).
3. Add conflict-aware merge arbitration summaries when multiple subagents target the same file.
4. Expand side-by-side benchmark report to include corpus checks and per-case agent matrix.
5. Validate with fmt/clippy/tests/build.

## Executed Plan (Latest Iteration)

1. Persist successful planner strategies and re-inject them as strategy memory context.
2. Add scoped subagent execution policies (bounded delegated calls by role).
3. Add benchmark pack workflow (`benchmark list-packs|show-pack|import-pack`) and `--benchmark-pack`.
4. Expand benchmark side-by-side output with case-level matrix and corpus compatibility checks.
5. Validate with fmt/clippy/tests/build.

## Executed Plan (Current Iteration)

1. Add strategy-memory quality scoring and automatic pruning of low-performing planner memories.
2. Add approval-aware delegated subagent retries (write/approval fallback to bounded read-only probes).
3. Upgrade merge arbitration with per-target candidate scoring and rationale reporting.
4. Add deterministic benchmark seed selection plus signed corpus/execution manifests and scorecards.
5. Validate with fmt/clippy/tests/build.

## Executed Plan (Current Iteration +1)

1. Add objective-outcome long-horizon memory with confidence scoring, pruning, and planning-context injection.
2. Add matrix-scale benchmark runner (`benchmark run-matrix`) for broad corpus/pack parity runs.
3. Add aggregate matrix scorecards with weighted success/quality and compatibility warnings.
4. Add matrix peer comparison ranking with manifest-coverage and case-count diagnostics.
5. Validate with fmt/clippy/tests/build.

## Executed Plan (Current Iteration +2)

1. Add long-horizon checkpoint-aware replanning quality scoring tied to objective-outcome risk memory.
2. Add subagent role/domain specialization memory with confidence scoring + guidance feedback loops.
3. Add strict benchmark compatibility gates (`--benchmark-compare-strict`, `benchmark run-matrix --strict`) for peer parity enforcement.
4. Add team-managed non-overridable policy overlay support in policy engine.
5. Validate with fmt/clippy/tests/build.

## Executed Plan (Current Iteration +3)

1. Implement true TUI `Ctrl+B` background execution semantics (agent and shell job launch).
2. Add first-class background run commands (`background run-agent`, `background run-shell`) with log-tail attach payloads.
3. Add subagent dependency/ownership lane planner with phased conflict-minimizing execution.
4. Add tests for background launch/attach/stop JSON contracts and lane-planning behavior.
5. Validate with fmt/clippy/tests/build.

## Executed Plan (Current Iteration +4)

1. Enforce stronger shell sandbox semantics by mode (`read-only`, `workspace-write`) at tool runtime.
2. Add production visual verification control plane (`visual list|analyze`) with strict pass/fail quality gates.
3. Expand built-in benchmark parity corpus (`benchmark` builtin `parity` pack) for broader reproducible peer comparisons.
4. Extend slash/TUI flows with `/visual` command parity.
5. Validate with fmt/clippy/tests/build.

## Next Iteration Plan (Remaining Open Gaps)

1. Build/ingest larger shared real-world benchmark corpora from external/public task sets and publish reproducible Codex/Aider/Claude parity scorecards.
2. Extend visual verification beyond static heuristics to richer UI-diff/expectation analyzers.
3. Strengthen sandbox isolation from command-level policy to OS-level execution containment workflows.
