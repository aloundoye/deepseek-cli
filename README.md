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

### Option 1: Build from source

```bash
cargo build --release --bin deepseek
./target/release/deepseek --help
```

### Option 2: Install from release binaries

macOS/Linux:

```bash
bash scripts/install.sh --version latest
```

Windows (PowerShell):

```powershell
./scripts/install.ps1 -Version latest
```

The install scripts support:
- `--version` / `-Version` (`latest` or tag like `v0.1.0`)
- `--repo` / `-Repo` (GitHub `owner/repo`)
- `--install-dir` / `-InstallDir`
- `--target` / `-Target` (override binary target triple)
- `--dry-run` / `-DryRun`

Default release artifacts target `x86_64` on Linux/macOS/Windows.

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
- `deepseek autopilot "<prompt>" [--hours N | --duration-seconds N | --forever] [--max-iterations N]`
- `deepseek run`
- `deepseek diff`
- `deepseek apply [--patch-id <uuid>] --yes`
- `deepseek index build|update|status|query <q> [--top-k N]`
- `deepseek config edit|show`
- `deepseek plugins list|install|remove|enable|disable|inspect|run`
- `deepseek clean [--dry-run]`

Global flag:
- `--json` for machine-readable output on key commands.

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

See `/Users/aloutndoye/Workspace/deepseek-cli/config.example.toml`.

Key sections:
- `[llm]` endpoint/auth/retries/streaming/offline fallback
- `[router]` thresholds/weights/escalation caps
- `[policy]` approval modes, allowlist, sandbox mode
- `[plugins]` search paths and hook enablement
- `[telemetry]` opt-in (`enabled=false` by default)

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


Based on the search results, I've compiled a comprehensive feature list of everything Claude Code currently offers. This will serve as your roadmap for building a DeepSeek CLI that can match—and potentially surpass—Claude Code.

# 🎯 Complete Claude Code Feature Specification (2026)

## Core Architecture & Environment

| Category | Feature | Description |
|:---------|:--------|:------------|
| **Runtime** | Terminal-native CLI | Runs directly in terminal with full shell integration  |
| | Cross-platform support | macOS 13.0+, Ubuntu 20.04+/Debian 10+, Windows 10+ (WSL/Git Bash)  |
| | Multiple installation methods | Native binary installer (recommended), Homebrew, Winget  |
| | Session persistence | Resume conversations with `claude -c`  |
| **Context** | 1M token context window | Handle entire codebases (up to ~750k words)  |
| | Automatic context compression | Summarizes early content when approaching limits  |
| | Project-wide awareness | Understands entire codebase structure and dependencies  |
| | Checkpointing | Auto-saves file edits, reversible via `/rewind` or Escape+Escape  |
| **Performance** | Fast Mode | Optimized API config for lower latency (2.5x speedup)  |
| | Parallel tool calls | Multiple independent tool calls in single response  |
| | Background execution | Ctrl+B backgrounds long-running agents and commands  |

## File System Operations

| Tool | Capability | Key Features |
|:-----|:-----------|:-------------|
| **Read** | View file contents | Line numbers, supports images and PDFs  |
| **Write** | Create/overwrite files | Safer than bash redirection, tracked by checkpoints  |
| **Edit** | Targeted modifications | String replacement in existing files, precise line editing  |
| **Glob** | Pattern-based search | Find files by pattern (e.g., `**/*.ts`)  |
| **Grep** | Content search | Regex pattern matching across files  |
| **File References** | @-mentions | Reference files directly: `@src/auth.ts:42-58`  |
| | Directory references | `@src/components/` for whole directories  |

## Git Integration

| Feature | Capability |
|:--------|:-----------|
| **Commit operations** | Create commits, branches, and pull requests directly from CLI  |
| **History search** | Search through Git history  |
| **Merge resolution** | Resolve merge conflicts  |
| **Checkpointing note** | Bash commands (rm, mv, cp) NOT tracked - only file edits via dedicated tools  |

## Slash Commands (Complete Set)

| Command | Purpose |
|:--------|:--------|
| `/help` | Show all available commands  |
| `/init` | Scan project and create CLAUDE.md  |
| `/clear` | Clear conversation history  |
| `/compact` | Summarize conversation to save context  |
| `/memory` | Edit CLAUDE.md memory files  |
| `/config` | Open configuration interface  |
| `/model` | Switch between Claude models (Opus/Sonnet)  |
| `/cost` | Check current token usage  |
| `/mcp` | Manage MCP server connections  |
| `/rewind` | Rewind to previous checkpoint  |
| `/export` | Export conversation to file  |
| `/plan` | Enable plan mode for complex tasks  |
| `/teleport` | Resume sessions at claude.ai/code  |
| `/remote-env` | Configure remote sessions  |
| `/status` | Check current model and configuration  |
| `/effort` | Control thinking depth (low/medium/high/max)  |

## MCP (Model Context Protocol) Extensibility

| Aspect | Specification |
|:-------|:--------------|
| **Transport types** | HTTP (remote servers), Stdio (local processes), SSE (deprecated)  |
| **Installation scopes** | local (`~/.claude.json`), project (`.mcp.json`), user (`~/.claude.json`)  |
| **Management commands** | `claude mcp add/list/get/remove`  |
| **Dynamic updates** | `list_changed` notifications - servers update tools without reconnection  |
| **Popular integrations** | GitHub, Notion, PostgreSQL, Sentry, Fig, Airtable, Slack  |
| **Xcode integration** | Capture Previews for visual verification  |

## Subagent System

| Feature | Capability |
|:--------|:-----------|
| **Parallel agents** | Up to 7 subagents running simultaneously  |
| **Agent types** | Explore (research), Plan (task breakdown), Task (specialized)  |
| **Isolated contexts** | `context: fork` in skill frontmatter prevents state pollution  |
| **Agent Teams** | Multiple agents coordinate on different components (frontend, backend, testing)  |
| **Resilience** | Agents continue after permission denial, try alternative approaches  |

## Keyboard & UX

| Shortcut | Action |
|:---------|:-------|
| Escape | Stop current response  |
| Escape + Escape | Open rewind menu  |
| ↑ (Up Arrow) | Navigate past chats  |
| Ctrl + V | Paste images (not Cmd+V on macOS)  |
| Tab | Autocomplete file paths and commands  |
| Shift+Enter | Multi-line input (iTerm2, Kitty, Ghostty, WezTerm)  |
| Ctrl+B | Background agents and shell commands  |
| Ctrl+O | Transcript mode showing real-time thinking  |

## Configuration System

| File | Purpose |
|:-----|:--------|
| `~/.claude/settings.json` | User-level settings  |
| `.claude/settings.json` | Project-specific settings (shared)  |
| `.claude/settings.local.json` | Per-machine overrides (gitignored)  |
| `~/.claude.json` | MCP server configurations  |
| `.mcp.json` | Project-scoped MCP servers  |
| `~/.claude/keybindings.json` | Custom keyboard shortcuts  |
| `CLAUDE.md` | Project memory, conventions, workflows  |

## Customization & Skills

| Feature | Description |
|:--------|:------------|
| **Skills** | Reusable prompt templates in `~/.claude/skills/` or `.claude/skills/`  |
| **Hot reload** | Skills available immediately without restart  |
| **Skill discovery** | Visible in slash command menu by default  |
| **Hooks** | `PreToolUse`, `PostToolUse`, `Stop` logic for fine-grained control  |
| **Wildcard permissions** | `Bash(npm *)`, `Bash(*-h*)` for pattern-based access  |
| **respectGitignore** | Control @-mention file picker behavior  |

## Advanced Capabilities

| Feature | Description |
|:--------|:------------|
| **Visual verification** | Capture and analyze Xcode Previews for UI feedback  |
| **Multilingual output** | `language` setting for Japanese, Spanish, etc.  |
| **Adaptive thinking** | Model adjusts reasoning depth based on complexity  |
| **max effort parameter** | Highest level of reasoning for complex tasks  |
| **IS_DEMO mode** | Hides personal info for streaming/recording  |
| **Terminal command sandboxing** | Experimental safety for command execution  |
| **Auto-approval rules** | Configurable permissions to reduce prompts  |
| **Memory across projects** | Enterprise-wide `CLAUDE.md` for consistent conventions  |

## Permission & Safety System

| Level | Description |
|:-------|:------------|
| **User approval** | Destructive operations require confirmation (delete, force-push, etc.)  |
| **Scope awareness** | Authorization matches requested scope only  |
| **Blast radius consideration** | Checks reversibility before acting  |
| **Security hardening** | No command injection, XSS, SQL injection  |
| **Team permissions** | Managed permissions that cannot be overwritten locally  |

## 🚀 Opportunities to Surpass Claude Code

Based on this spec, here are areas where DeepSeek could potentially **exceed** Claude Code's capabilities:

1. **Cost efficiency**: DeepSeek's API is dramatically cheaper (16-20x less than Claude) . Build this into your value proposition.

2. **Prompt caching**: DeepSeek supports 90% discount on repeated input tokens —implement aggressive caching strategies.

3. **Off-peak pricing**: Consider implementing scheduled background tasks during cheaper rate periods.

4. **Open source ecosystem**: Your Rust implementation could become the reference open-source alternative, attracting community contributions.

5. **Multi-provider support**: Unlike Claude's lock-in, you could support multiple backends (DeepSeek, OpenAI, local models) [citation:10 shows Copilot already does this].

6. **Performance monitoring**: Add `/profile` or `/timeline` commands showing where time is spent.

7. **Integration marketplace**: Build a community repository of MCP-compatible plugins.

## 📚 Documentation References

For implementation details, study these sources:
- **MCP specification**: Anthropic's open standard 
- **Permission system**: GitHub prompt shows exact tool behavior 
- **Agent Teams**: Enable with `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` 
- **Checkpointing**: Only file edits tracked, not bash commands 

This spec gives you everything Claude Code offers as of February 2026. Start with the core infrastructure, then progressively add features based on user demand. The modular architecture I outlined earlier will let you build this systematically.

Which feature category would you like to tackle first? The MCP extensibility system would be a great differentiator to implement early.