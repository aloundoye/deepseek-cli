# DeepSeek CLI

DeepSeek CLI is a terminal-native coding agent written in Rust. It combines chat, planning, tool execution, patch workflows, indexing, and long-running autopilot loops in one cross-platform CLI.

## Highlights
- DeepSeek model aliases: `deepseek-chat` and `deepseek-reasoner`
- Default profile: `v3_2`
- Interactive chat UI (TUI enabled by default in interactive terminals)
- Structured automation output via `--json`
- Session persistence and deterministic replay
- Plugin, skill, hook, and MCP integrations
- Permission and sandbox policy controls
- Built-in benchmark and parity tooling

## Install

### Option 1: Prebuilt release binary (recommended)

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

### Option 2: Build from source

```bash
cargo build --release --bin deepseek
./target/release/deepseek --help
```

## Quickstart

```bash
export DEEPSEEK_API_KEY="<your-api-key>"
deepseek chat
```

Non-interactive examples:

```bash
deepseek ask "Summarize this repository"
deepseek plan "Implement feature X and list risks"
deepseek autopilot "Execute plan and verify tests" --hours 2
```

Credential behavior:
- If the API key is missing in interactive TTY mode, DeepSeek CLI prompts for it before first model use.
- You can persist the key for the current workspace in `.deepseek/settings.local.json`.
- In non-interactive or JSON mode, missing credentials fail fast.
- `deepseek config show` redacts `llm.api_key`.

## Core Commands

Run `deepseek --help` for full details. The most used commands are:

- `deepseek chat`: interactive chat session (TUI when enabled and running in a TTY)
- `deepseek ask "<prompt>"`: one-shot response
- `deepseek plan "<prompt>"`: generate a plan without running it
- `deepseek run [session-id]`: continue execution for a session
- `deepseek autopilot "<prompt>"`: bounded unattended loop
- `deepseek review --diff|--staged|--pr <n>`: code review workflows
- `deepseek exec "<command>"`: policy-enforced shell execution
- `deepseek diff`, `deepseek apply`, `deepseek rewind`: patch/checkpoint lifecycle
- `deepseek status`, `deepseek context`, `deepseek usage`: runtime and cost visibility
- `deepseek index build|update|status|watch|query`: local code index operations
- `deepseek permissions show|set|dry-run`: safety policy inspection and tuning
- `deepseek replay run|list`: deterministic replay tooling
- `deepseek benchmark ...`, `deepseek profile --benchmark`: benchmark workflows
- `deepseek search "<query>"`: web search with provenance metadata
- `deepseek remote-env ...`: remote profile orchestration (`list|add|remove|check|exec|run-agent|logs`)
- `deepseek teleport ...`: handoff workflows (`export|import|link|consume`)
- `deepseek visual ...`: visual artifact workflows (`list|analyze|show`)

Useful global flags:
- `--json`: machine-readable output
- `--permission-mode ask|auto|plan`: per-run permission mode override

## Example Workflows

Plan and execute:

```bash
deepseek plan "Refactor auth middleware and keep behavior unchanged"
deepseek run
```

Autopilot with live status:

```bash
deepseek autopilot "Fix flaky tests in crates/deepseek-cli" --hours 1
deepseek autopilot status --follow
```

Review staged changes:

```bash
deepseek review --staged --focus correctness
```

Publish strict review findings to a PR:

```bash
deepseek review --pr 123 --publish --max-comments 20
```

Run a remote command over SSH profile:

```bash
deepseek remote-env exec <profile-id> --cmd "git status" --timeout-seconds 60
```

Create and consume one-time handoff links:

```bash
deepseek teleport link --ttl-minutes 30
deepseek teleport consume --handoff-id <id> --token <token>
```

Run matrix benchmark and emit report:

```bash
deepseek benchmark run-matrix .github/benchmark/slo-matrix.json --strict --report-output report.md
```

Generate shell completions:

```bash
deepseek completions --shell zsh > ~/.zsh/completions/_deepseek
```

## Configuration

DeepSeek CLI merges configuration in this order (later entries win):

1. Legacy fallback: `.deepseek/config.toml`
2. User config: `~/.deepseek/settings.json`
3. Project config: `.deepseek/settings.json`
4. Project-local overrides: `.deepseek/settings.local.json`

Useful commands:

```bash
deepseek config show
deepseek config edit
deepseek --setting-sources
```

Reference files:
- `config.example.json`
- `config.example.toml`

Notable architecture/runtime defaults:
- Edit execution uses a deterministic loop: Architect (`deepseek-reasoner`) -> Editor (`deepseek-chat`) -> Apply -> Verify.
- Unified diff is the only edit contract in the execution loop (`NEED_CONTEXT|path[:start-end]` is used when Editor needs more file context).
- Verify failures are classified deterministically (mechanical/repeated/design-mismatch) to decide Editor-first vs Architect-first retry.
- Apply safety gates require approval for oversized patches and create checkpoints around patch apply.
- Verify-pass emits a commit proposal and next actions; it never auto-commits.
- Optional team-lane composition (`--teammate-mode`) runs lane pipelines in isolated worktrees and merges lane patches deterministically.
- Tool/function-calling orchestration is not used as the core execution backbone.
- Analysis/review commands use a separate non-edit path and do not mutate files.
- Ask/context analysis prompts use deterministic `AUTO_CONTEXT_BOOTSTRAP_V1` repo context and return initial analysis before limited follow-up questions.
- Repo-oriented prompts without a detected repository return: `No repository detected. Run from project root or pass --repo <path>.`
- Chrome tooling is strict-live by default (`tools.chrome.allow_stub_fallback = false`).
- Terminal image fallback policy is configurable (`ui.image_fallback = "open|path|none"`).
- TUI visibility defaults to concise phase summaries with heartbeat progress (`ui.thinking_visibility = "concise"`, `ui.phase_heartbeat_ms = 5000`).
- Mission Control retention is bounded (`ui.mission_control_max_events`).
- Teleport link base URL is configurable (`ui.handoff_base_url`).

Common execution control flags:
- `--force-execute` (force full Architect->Editor->Apply->Verify loop)
- `--plan-only` (architect plan output only, no apply/verify)
- `--repo <path>` (override repository root for bootstrap/context)
- `--debug-context` (print deterministic pre-model context digest; also available via `DEEPSEEK_DEBUG_CONTEXT=1`)

Explicit commit workflow:
- `/stage`, `/unstage`, `/diff`, `/commit`, `/undo` are available in chat/TUI.
- `/commit` commits staged changes only (no implicit staging).
- `deepseek git commit --message "..."` also commits staged-only by default.

Workflow parity slash commands:
- Context/workspace controls: `/add`, `/drop`, `/read-only`, `/map`, `/map-refresh`
- Deterministic command helpers: `/run`, `/test`, `/lint`
- Web research helper: `/web`

Default model/profile behavior:
- `llm.provider = "deepseek"`
- `llm.profile = "v3_2"`
- `llm.base_model = "deepseek-chat"`
- `llm.max_think_model = "deepseek-reasoner"`
- `llm.context_window_tokens` defaults to `128000` (you can raise this in config, for example to 1M)

## Safety and Policy Controls

- Approval controls: `policy.approve_bash`, `policy.approve_edits`
- Command allowlist and blocked path patterns
- Redaction regex patterns for sensitive data
- Sandbox mode and optional OS sandbox wrapper
- Team-managed policy overlays (when configured) can lock settings

Inspect and update policy:

```bash
deepseek permissions show
deepseek permissions set --approve-bash ask --approve-edits ask
deepseek permissions dry-run bash.run
```

## Runtime Data

Project runtime state lives in `.deepseek/`, including:
- `events.jsonl` (append-only event log)
- `store.sqlite`
- `plans/`
- `patches/`
- `index/`
- `observe.log`

## Integrations

- IDE/Editor extensions: `extensions/`
  - VS Code starter extension: `extensions/vscode`
  - JetBrains starter plugin: `extensions/jetbrains`
  - Chrome native host bridge: `extensions/chrome`
- MCP management: `deepseek mcp ...`
- JSON-RPC server for IDE integration: `deepseek serve`

See `extensions/README.md` for setup.

## Development

Build and verify:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo build --workspace
cargo build --release --workspace
```

Additional production/CI gates used by this repo:

```bash
cargo run --bin deepseek -- --json replay list --limit 20
cargo run --bin deepseek -- --json profile --benchmark --benchmark-suite .github/benchmark/slo-suite.json --benchmark-cases 3 --benchmark-min-success-rate 1.0 --benchmark-min-quality-rate 1.0 --benchmark-max-p95-ms 2000
cargo run --bin deepseek -- --json benchmark run-matrix .github/benchmark/slo-matrix.json --strict
bash scripts/runtime_conformance_scan.sh
bash scripts/parity_regression_check.sh
```

## Docs

- Release process: `docs/RELEASE.md`
- Release checklist: `docs/RELEASE_CHECKLIST.md`
- Operations playbook: `docs/OPERATIONS.md`
- Production readiness: `docs/PRODUCTION_READINESS.md`
- Feature audit matrix: `docs/FEATURE_MATRIX.md`
