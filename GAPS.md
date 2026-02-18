# DeepSeek CLI Parity Gap Matrix (Codex / Aider / Claude Code)

Date: 2026-02-18

This matrix tracks the remaining gap between `specs.md` expectations and practical parity against Codex, Aider, and Claude Code.

## Current Gap Matrix

| Area | Gap | Status |
|---|---|---|
| Permission control UX | No first-class runtime permission management command and no slash control path. | **Closed in this iteration** (`deepseek permissions show|set`, `/permissions`) |
| Live autopilot/background semantics | Limited visibility during long runs; no live sampled status stream. | **Partially closed** (`autopilot status --follow --samples --interval-seconds`) |
| Benchmark realism | Benchmarks relied mostly on fixed built-in synthetic prompts. | **Partially closed** (`--benchmark-suite` external JSON/JSONL cases + quality gates + baseline comparison/regression checks) |
| Autonomous reasoning quality | Planner/executor still below frontier agents on deep, multi-hop, long-horizon reasoning and self-correction quality. | **Partially closed** (multi-pass quality + verification-feedback repair loops + strategy scoring/pruning + objective-outcome confidence memory injected into planning context) |
| Subagent intelligence | Subagents are model-backed, but still mostly advisory and not fully delegated tool-using specialists. | **Partially closed** (bounded delegated execution policies by role + approval-aware delegated retries + richer target arbitration scoring/rationales) |
| Real-world benchmark parity | No standardized side-by-side benchmark corpus + scorer against Codex/Aider/Claude on shared tasks. | **Partially closed** (`--benchmark-compare` ranking + per-case matrix + corpus mismatch warnings + importable packs + seeded signed scorecards/manifests + `benchmark run-matrix` aggregate parity runner) |

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

## Next Iteration Plan (Remaining Open Gaps)

1. Expand long-horizon autonomous reasoning with explicit multi-turn objective decomposition and checkpoint-aware replanning quality scores.
2. Add deeper subagent specialization signals (role-specific toolsets + quality feedback loops per subagent/team).
3. Establish a larger shared real-world benchmark corpus and ingest external Codex/Aider/Claude reports under strict manifest compatibility requirements.
