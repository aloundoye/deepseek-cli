# DeepSeek CLI

DeepSeek CLI is a cross-platform coding agent for terminal workflows.
It provides planning, execution, patching, indexing, plugin/hooks extensibility, and guarded autopilot loops for long-running tasks.

## Highlights
- DeepSeek V3.2 model aliases: `deepseek-chat`, `deepseek-reasoner`
- Rust `edition = 2024`, `rust-version = 1.93.0`
- Session persistence and deterministic replay
- Cross-platform runtime (Linux, macOS, Windows)
- CLI JSON mode for automation (`--json`)
- Plugin, skills, hooks, MCP integrations

## Install

### GitHub release installer (recommended)

macOS/Linux:

```bash
bash scripts/install.sh --version latest
```

Windows (PowerShell):

```powershell
./scripts/install.ps1 -Version latest
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
cargo run --bin deepseek -- --json plan "Implement feature X"
cargo run --bin deepseek -- chat
cargo run --bin deepseek -- autopilot "Execute and verify task" --hours 4
```

## Command Overview

- `deepseek chat [--tools] [--tui]`
- `deepseek ask "<prompt>" [--tools]`
- `deepseek plan "<prompt>"`
- `deepseek run [session-id]`
- `deepseek autopilot "<prompt>" [--hours|--duration-seconds|--forever] [--max-iterations N]`
- `deepseek autopilot status|stop|resume`
- `deepseek diff`
- `deepseek apply [--patch-id <uuid>] --yes`
- `deepseek rewind [--to-checkpoint <uuid>] --yes`
- `deepseek export [--session <uuid>] [--format json|md] [--output <path>]`
- `deepseek memory show|edit|sync`
- `deepseek mcp add|list|get|remove`
- `deepseek git status|history|branch|checkout|commit|pr|resolve`
- `deepseek skills list|install|remove|run|reload`
- `deepseek replay run --session-id <uuid> --deterministic`
- `deepseek background list|attach|stop`
- `deepseek teleport [--session-id <uuid>] [--output <path>] [--import <path>]`
- `deepseek remote-env list|add|remove|check`
- `deepseek status`
- `deepseek usage [--session] [--day]`
- `deepseek compact [--from-turn N] [--yes]`
- `deepseek doctor`
- `deepseek index build|update|status|watch|query <q> [--top-k N]`
- `deepseek config edit|show`
- `deepseek plugins list|install|remove|enable|disable|inspect|catalog|search|verify|run`
- `deepseek clean [--dry-run]`

Global flag:
- `--json`

Interactive slash commands:
- `/help`, `/init`, `/clear`, `/compact`, `/memory`, `/config`, `/model`
- `/cost`, `/mcp`, `/rewind`, `/export`, `/plan`, `/teleport`, `/remote-env`
- `/status`, `/effort`

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

## Docs
- Release/install process: `docs/RELEASE.md`
- Operations/rollback playbook: `docs/OPERATIONS.md`
