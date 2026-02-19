# DeepSeek CLI

DeepSeek CLI is a cross-platform coding agent for terminal workflows.
It provides planning, execution, patching, indexing, plugin/hooks extensibility, and guarded autopilot loops for long-running tasks.

## Highlights
- DeepSeek V3.2 model aliases: `deepseek-chat`, `deepseek-reasoner`
- DeepSeek profile control: `v3_2` (default) and opt-in `v3_2_speciale`
- Rust `edition = 2024`, `rust-version = 1.93.0`
- Session persistence and deterministic replay
- Cross-platform runtime (Linux, macOS, Windows)
- CLI JSON mode for automation (`--json`)
- Plugin, skills, hooks, MCP integrations
- Strict-online runtime (`DEEPSEEK_API_KEY` required)
- Rich terminal UI default for `deepseek chat`
- 1M context-window configuration with automatic context compaction near threshold
- Interactive approval prompts for gated tool calls in TTY sessions
- Multi-tool step execution from planner output (including alias/suffix tool syntax)

## Install

### GitHub release installer (recommended, no source checkout)

macOS/Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/aloutndoye/deepseek-cli/main/scripts/install.sh | bash -s -- --version latest
```

Windows (PowerShell):

```powershell
$script = Join-Path $env:TEMP "deepseek-install.ps1"
Invoke-WebRequest https://raw.githubusercontent.com/aloutndoye/deepseek-cli/main/scripts/install.ps1 -OutFile $script
& $script -Version latest
```

### Homebrew

```bash
brew tap <your-org>/deepseek
brew install deepseek
```

### Winget

```powershell
winget install DeepSeek.DeepSeekCLI
```

### Build from source

```bash
cargo build --release --bin deepseek
./target/release/deepseek --help
```

## Quickstart

```bash
export DEEPSEEK_API_KEY="<your-key>"
deepseek --json plan "Implement feature X"
deepseek chat
deepseek autopilot "Execute and verify task" --hours 4
```

Credential behavior:
- Interactive TTY commands prompt for `DEEPSEEK_API_KEY` when missing before first model call.
- Interactive key prompt offers to persist the key in `.deepseek/settings.local.json` (`llm.api_key`) for the current workspace.
- Non-interactive/JSON mode fails fast with a clear error instead of attempting a network call.
- `deepseek config show` redacts `llm.api_key` in both text and JSON output.

Execution behavior:
- Planner steps can execute multiple declared tools per step (up to 3), in declared order.
- Declared tool syntax supports `name:arg` and `name(arg)` forms (for example `bash.run:cargo test --workspace`).
- Subagents run model-backed analysis with delegated read-only tool probes and are recorded in transcript/status output.
- Task-role subagents now run bounded delegated tool execution (including isolated subagent artifact writes) with merge-arbitration summaries when multiple subagents target the same file.
- Delegated subagent execution now includes approval-aware retries with bounded read-only fallback paths.
- Autopilot supports live pause/resume control via pause files and `deepseek autopilot pause|resume`, plus live status sampling with `deepseek autopilot status --follow`.
- TUI `Ctrl+B` now backgrounds queued work directly: normal prompts launch background agent runs, and `!<command>` launches background shell jobs.
- Background jobs now support native starts for both agents and shell commands (`deepseek background run-agent|run-shell`) with log-tail attach payloads.
- Sandbox modes now enforce runtime semantics for shell commands (`read-only` blocks mutating commands, `workspace-write` blocks absolute/out-of-workspace path targets).
- Sandbox enforcement now also blocks risky network-egress shell commands in constrained modes.
- Isolated sandbox mode now supports OS-level containment wrappers via `policy.sandbox_wrapper` (template with `{workspace}` and `{cmd}` placeholders) plus platform auto-detection (`bwrap`/`firejail`/`sandbox-exec`).
- `deepseek permissions show|set` provides first-class policy control for approvals/allowlist/sandbox mode.
- Planner quality checks auto-repair weak plans with bounded retries before fallback.
- Planner self-evaluation now includes prior verification-failure memory to improve subsequent plan revisions.
- Successful/failed runs are persisted into planner strategy memory with score-based prioritization and pruning before future plan injection.
- Long-horizon objective outcomes are persisted with confidence/failure trends and reinjected into future planning context.
- Planner long-horizon quality checks now enforce decomposition + checkpoint/rollback guards for historically risky objectives.
- Subagent specialization memory persists per role/domain quality and injects guidance into future delegated runs.
- `deepseek profile --benchmark` runs planning benchmarks with latency/quality metrics, supports external suites, compares against baseline runs, and produces side-by-side peer ranking with case matrices.
- Benchmark mode supports thresholds and result export (`--benchmark-suite`, `--benchmark-min-success-rate`, `--benchmark-min-quality-rate`, `--benchmark-max-p95-ms`, `--benchmark-baseline`, `--benchmark-max-regression-ms`, `--benchmark-compare`, `--benchmark-output`), plus reproducible seeded runs with signed corpus/execution manifests and scorecards (`--benchmark-seed`, `--benchmark-signing-key-env`).
- `deepseek benchmark run-matrix <matrix.json>` executes multi-run parity matrices (packs/suites) with aggregate weighted scorecards and optional peer ranking (including strict compatibility gating).
- Matrix runs can now emit publish-ready Markdown scorecards and enforce required peer agents (`--report-output`, `--require-agent`).
- `deepseek benchmark sync-public <catalog.json|url>` imports public benchmark packs (JSON/JSONL sources, local or remote) into local pack storage with optional filtering/prefixing.
- `deepseek benchmark publish-parity` runs a parity matrix and emits publication bundles (`latest.json`, `latest.md`, timestamped reports, and `history.jsonl`) for recurring reporting.
- Built-in benchmark packs now include an expanded `parity` pack for broader reproducible peer-comparison coverage.
- Team-managed policy overlays can be enforced via `DEEPSEEK_TEAM_POLICY_PATH` (or `~/.deepseek/team-policy.json`) and cannot be broadened by local allowlists.
- `deepseek visual list|analyze` provides visual artifact inventory plus strict analysis gates for captured image/PDF artifacts.
- Visual analysis now supports baseline diffing and baseline manifest writeback (`--baseline`, `--write-baseline`) for regression gating.
- Visual analysis now supports semantic expectation rules (`--expect`) including artifact-count, MIME, glob, and image dimension checks.

## Command Overview

- `deepseek chat [--tools] [--tui]` (TUI is default in interactive terminals)
- `deepseek ask "<prompt>" [--tools]`
- `deepseek plan "<prompt>"`
- `deepseek run [session-id]`
- `deepseek profile [--benchmark] [--benchmark-cases N] [--benchmark-seed N] [--benchmark-suite <path>] [--benchmark-pack <name>] [--benchmark-signing-key-env ENV] [--benchmark-min-success-rate F] [--benchmark-min-quality-rate F] [--benchmark-max-p95-ms N] [--benchmark-baseline <path>] [--benchmark-max-regression-ms N] [--benchmark-compare <path>] [--benchmark-compare-strict] [--benchmark-output <path>]`
- `deepseek benchmark list-packs|show-pack <name>|import-pack <name> <source>|sync-public <catalog> [--only-pack <name[,name...]> --prefix <prefix>]|run-matrix <matrix.json> [--output <path>] [--report-output <path>] [--compare <path>] [--require-agent <agent[,agent...]>] [--strict]|publish-parity [--matrix <matrix.json> --output-dir <dir> --strict]`
- `deepseek autopilot "<prompt>" [--hours|--duration-seconds|--forever] [--max-iterations N]`
- `deepseek autopilot status [--follow --samples N --interval-seconds N]|pause|stop|resume`
- `deepseek diff`
- `deepseek apply [--patch-id <uuid>] --yes`
- `deepseek rewind [--to-checkpoint <uuid>] --yes`
- `deepseek export [--session <uuid>] [--format json|md] [--output <path>]`
- `deepseek memory show|edit|sync`
- `deepseek mcp add|list|get|remove`
- `deepseek git status|history|branch|checkout|commit|pr|resolve`
- `deepseek skills list|install|remove|run|reload`
- `deepseek replay run --session-id <uuid> --deterministic|list [--session-id <uuid>] [--limit N]`
- `deepseek background list|attach|stop|run-agent|run-shell`
- `deepseek visual list [--limit N]|analyze [--limit N --min-bytes N --min-artifacts N --min-image-artifacts N --baseline <path> --write-baseline <path> --expect <path> --max-new-artifacts N --max-missing-artifacts N --max-changed-artifacts N --strict]`
- `deepseek teleport [--session-id <uuid>] [--output <path>] [--import <path>]`
- `deepseek remote-env list|add|remove|check`
- `deepseek status`
- `deepseek usage [--session] [--day]`
- `deepseek compact [--from-turn N] [--yes]`
- `deepseek doctor`
- `deepseek index build|update|status|watch|query <q> [--top-k N]`
- `deepseek config edit|show`
- `deepseek permissions show|set [--approve-bash <ask|always|never>] [--approve-edits <ask|always|never>] [--sandbox-mode <mode>] [--allow <prefix>] [--clear-allowlist]`
- `deepseek plugins list|install|remove|enable|disable|inspect|catalog|search|verify|run`
- `deepseek clean [--dry-run]`

Global flag:
- `--json`

Interactive slash commands:
- `/help`, `/init`, `/clear`, `/compact`, `/memory`, `/config`, `/model`
- `/cost`, `/mcp`, `/rewind`, `/export`, `/plan`, `/teleport`, `/remote-env`
- `/status`, `/effort`, `/skills`
- `/permissions`, `/background`, `/visual`

## Configuration

Load order:
1. `~/.deepseek/settings.json`
2. `.deepseek/settings.json`
3. `.deepseek/settings.local.json`
4. legacy fallback: `.deepseek/config.toml`

Additional config files:
- `~/.deepseek/mcp.json`
- `~/.deepseek/mcp.local.json`
- `.mcp.json`
- `~/.deepseek/keybindings.json`

Sample config:
- `config.example.json`
- `config.example.toml`

LLM profile notes:
- `llm.profile = "v3_2"` uses the stable DeepSeek V3.2 aliases (`deepseek-chat`, `deepseek-reasoner`).
- `llm.profile = "v3_2_speciale"` is supported as an explicit opt-in compatibility profile.
- V3.2-Speciale is documented by DeepSeek as a limited release that ended on December 15, 2025; keep `v3_2` as the default for production.

Benchmark and visual examples:
- Public corpus catalog template: `.github/benchmark/public-catalog.example.json`
- Scheduled parity matrix template: `.github/benchmark/parity-matrix.json`
- SLO benchmark suite template: `.github/benchmark/slo-suite.json`
- SLO matrix template: `.github/benchmark/slo-matrix.json`
- CI parity publication workflow: `.github/workflows/parity-publication.yml`
- Replay regression workflow: `.github/workflows/replay-regression.yml`
- Performance gate workflow: `.github/workflows/performance-gates.yml`
- Security gate workflow: `.github/workflows/security-gates.yml`
- Live DeepSeek smoke workflow (secret-gated): `.github/workflows/live-deepseek-smoke.yml`
- Manual release readiness drill workflow: `.github/workflows/release-readiness.yml`

Hospital-focused safety recommendations:
- Require `DEEPSEEK_API_KEY` in runtime environments.
- Set `llm.context_window_tokens=1000000` (default) and tune `context.auto_compact_threshold`.
- Use strict `[policy].block_paths` and `[policy].redact_patterns` for PHI/secret redaction.
- Enable `[scheduling].off_peak=true` with `[scheduling].defer_non_urgent=true` for non-urgent autopilot runs.
- Keep `[telemetry].enabled=false` unless your compliance policy explicitly allows remote telemetry.

## Plugin/Skill Layout

```text
<plugin-root>/
  .deepseek-plugin/plugin.json
  commands/*.md
  agents/*.md
  skills/**/SKILL.md
  hooks/*
```

## Runtime Data

Runtime state is stored in `.deepseek/`:
- `events.jsonl` (append-only event log)
- `store.sqlite`
- `plans/`
- `patches/`
- `index/`
- `observe.log`

## Verification

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo build --workspace
cargo build --release --workspace
```

Extended production gates:

```bash
cargo run --bin deepseek -- --json replay list --limit 20
cargo run --bin deepseek -- --json profile --benchmark --benchmark-suite .github/benchmark/slo-suite.json --benchmark-cases 3 --benchmark-min-success-rate 1.0 --benchmark-min-quality-rate 1.0 --benchmark-max-p95-ms 2000
cargo run --bin deepseek -- --json benchmark run-matrix .github/benchmark/slo-matrix.json --strict
cargo audit --deny warnings
cargo deny check advisories bans licenses sources
```

## Docs
- Release/install process: `docs/RELEASE.md`
- Operations/rollback playbook: `docs/OPERATIONS.md`
- Production readiness checklist: `docs/PRODUCTION_READINESS.md`
