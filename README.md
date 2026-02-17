# DeepSeek CLI Agent

DeepSeek CLI is a Rust coding agent implementing the RFC in `/Users/aloutndoye/Workspace/deepseek-cli/specs.md` with:
- Plan-first execution loop
- Planner/Executor state machine with bounded recovery
- Safe tool host with approvals, allowlists, and path guards
- Auto model routing (`deepseek-chat` -> `deepseek-reasoner`)
- Append-only event log + deterministic replay projections
- Manifest-bound Tantivy indexing
- Plugin discovery/runtime and hooks

Compatibility baseline:
- DeepSeek V3.2 API aliases: `deepseek-chat` and `deepseek-reasoner`
- Rust edition `2024`
- Rust toolchain `1.93.0` (see `/Users/aloutndoye/Workspace/deepseek-cli/rust-toolchain.toml`)

## Install

### Option 1: Install from GitHub release binaries (recommended)

macOS/Linux:

```bash
bash scripts/install.sh --version latest
```

Windows (PowerShell):

```powershell
./scripts/install.ps1 -Version latest
```

### Option 2: Homebrew (tap)

```bash
brew tap <your-org>/deepseek
brew install deepseek
```

### Option 3: Winget

```powershell
winget install DeepSeek.DeepSeekCLI
```

### Option 4: Build from source (contributors)

```bash
cargo build --release --bin deepseek
./target/release/deepseek --help
```

The install scripts support:
- `--version` / `-Version` (`latest` or tag like `v0.1.0`)
- `--repo` / `-Repo` (GitHub `owner/repo`)
- `--install-dir` / `-InstallDir`
- `--target` / `-Target` (override binary target triple)
- `--dry-run` / `-DryRun`

Default release artifacts:
- `x86_64-unknown-linux-gnu`
- `aarch64-unknown-linux-gnu`
- `x86_64-apple-darwin`
- `aarch64-apple-darwin`
- `x86_64-pc-windows-msvc`
- `aarch64-pc-windows-msvc`

Uninstall:
- macOS/Linux: `rm <install-dir>/deepseek`
- Windows: `Remove-Item <InstallDir>\deepseek.exe`

## Quickstart

```bash
cargo test --workspace
cargo run --bin deepseek -- --json plan "Add config validation"
cargo run --bin deepseek -- chat
cargo run --bin deepseek -- autopilot "Implement and verify the refactor" --hours 6
```

## Commands

- `deepseek chat [--tools]`
- `deepseek ask "<prompt>" [--tools]`
- `deepseek plan "<prompt>"`
- `deepseek profile [--json]`
- `deepseek autopilot "<prompt>" [--hours N | --duration-seconds N | --forever] [--max-iterations N]`
- `deepseek autopilot status [--run-id <uuid>]`
- `deepseek autopilot stop [--run-id <uuid>]`
- `deepseek autopilot resume [--run-id <uuid>]`
- `deepseek run`
- `deepseek diff`
- `deepseek apply [--patch-id <uuid>] --yes`
- `deepseek rewind [--to-checkpoint <uuid>] --yes`
- `deepseek export [--session <uuid>] [--format json|md] [--output <path>]`
- `deepseek memory show|edit|sync`
- `deepseek status`
- `deepseek usage [--session] [--day]`
- `deepseek compact [--from-turn N] [--yes]`
- `deepseek doctor`
- `deepseek mcp add|list|get|remove`
- `deepseek index build|update|status|watch|query <q> [--top-k N]`
- `deepseek config edit|show`
- `deepseek plugins list|install|remove|enable|disable|inspect|catalog|search|verify|run`
- `deepseek clean [--dry-run]`

Global flag:
- `--json` for machine-readable output on key commands.

Interactive slash commands in `chat`:
- `/help`, `/init`, `/clear`, `/compact`, `/memory`, `/config`, `/model`
- `/cost`, `/mcp`, `/rewind`, `/export`, `/plan`, `/teleport`, `/remote-env`
- `/status`, `/effort`

Autopilot notes:
- Default runtime is 2 hours if no duration flag is provided.
- Use `--forever` for indefinite execution (stop manually).
- Use `--max-think true` (default) to force max-thinking model per iteration.
- Use `--tools true` (default) for full autonomous execution.
- Use `--stop-file <path>` (default `.deepseek/autopilot.stop`) for graceful external stop.
- Heartbeats are written to `.deepseek/autopilot.heartbeat.json` (or `--heartbeat-file <path>`).
- Use `--max-consecutive-failures <N>` to cap runaway failure loops.

## Plugin compatibility contract

Supported plugin layout:

```text
<plugin-root>/
  .deepseek-plugin/plugin.json
  commands/*.md
  agents/*.md
  skills/**/SKILL.md
  hooks/*
```

Example usage:

```bash
deepseek plugins install /path/to/plugin
deepseek plugins list
deepseek plugins inspect <plugin_id>
deepseek plugins run <plugin_id> <command_name> --input "target task"
```

Hooks run only when enabled in config (`[plugins].enable_hooks=true`).
Hook scripts receive:
- `DEEPSEEK_HOOK_PHASE` (`pretooluse` or `posttooluse`)
- `DEEPSEEK_TOOL_NAME`
- `DEEPSEEK_TOOL_ARGS_JSON`
- `DEEPSEEK_TOOL_RESULT_JSON` (post hooks)
- `DEEPSEEK_WORKSPACE`

## Runtime data layout

DeepSeek CLI writes runtime artifacts under `.deepseek/`:
- `.deepseek/events.jsonl` (canonical append-only event log)
- `.deepseek/store.sqlite` (projections/migrations)
- `.deepseek/plans/*.json`
- `.deepseek/patches/*.diff` + `*.json`
- `.deepseek/index/manifest.json` + `.deepseek/index/tantivy/`
- `.deepseek/observe.log`

`deepseek clean` prunes non-canonical artifacts (`patches`, Tantivy index files, observe log).

## Configuration

Configuration load order:
- `~/.deepseek/settings.json`
- `.deepseek/settings.json`
- `.deepseek/settings.local.json`
- legacy fallback: `.deepseek/config.toml`

Sample config files:
- `/Users/aloutndoye/Workspace/deepseek-cli/config.example.json` (current)
- `/Users/aloutndoye/Workspace/deepseek-cli/config.example.toml` (legacy fallback format)

Additional config contracts:
- `~/.deepseek/mcp.json`
- `~/.deepseek/mcp.local.json`
- `.mcp.json`
- `~/.deepseek/keybindings.json`

Example JSON config:

```json
{
  "llm": {
    "base_model": "deepseek-chat",
    "max_think_model": "deepseek-reasoner",
    "endpoint": "https://api.deepseek.com/chat/completions",
    "api_key_env": "DEEPSEEK_API_KEY"
  },
  "telemetry": {
    "enabled": false
  }
}
```

## Verification gates

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo build --workspace
```

CI matrix enforces the same gates on Linux, macOS, and Windows.

## Operations docs

- Release + installers: `/Users/aloutndoye/Workspace/deepseek-cli/docs/RELEASE.md`
- Rollback/incident playbook: `/Users/aloutndoye/Workspace/deepseek-cli/docs/OPERATIONS.md`
