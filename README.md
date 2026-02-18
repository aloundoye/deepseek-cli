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
- Strict-online default (`DEEPSEEK_API_KEY` required unless offline fallback is explicitly enabled)
- Rich terminal UI default for `deepseek chat`
- 1M context-window configuration with automatic context compaction near threshold

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

## Command Overview

- `deepseek chat [--tools] [--tui]` (TUI is default in interactive terminals)
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
- `/status`, `/effort`, `/skills`

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

Hospital-focused safety recommendations:
- Keep `llm.offline_fallback=false` and require `DEEPSEEK_API_KEY`.
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

## Docs
- Release/install process: `docs/RELEASE.md`
- Operations/rollback playbook: `docs/OPERATIONS.md`
