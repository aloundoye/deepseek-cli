# DeepSeek CLI

DeepSeek CLI is a terminal-native coding agent written in Rust. It combines chat, planning, tool execution, patch workflows, indexing, and long-running autopilot loops in one cross-platform CLI.

## Highlights
- **Agent intelligence**: Adaptive complexity (Simple/Medium/Complex), planning protocols, error recovery, stuck detection
- **Bootstrap context**: Automatic project awareness (file tree, git status, repo map, dependency analysis) on first turn
- **Hybrid retrieval**: Vector + BM25 code search with RRF fusion — surfaces relevant code before the LLM responds
- **Privacy scanning**: 3-layer secret detection (path/content/builtin patterns) with redaction on tool outputs
- **Ghost text**: Local ML-powered inline completions in the TUI (Tab to accept, Alt+Right for one word)
- **Model routing**: Automatically routes complex tasks to `deepseek-reasoner`, simple tasks to `deepseek-chat`
- **14 lifecycle hooks**: SessionStart through TaskCompleted — extend behavior at every stage
- **Interactive TUI**: Vim mode, @file autocomplete, !bash prefix, syntax highlighting, keyboard shortcuts
- **Skills & subagents**: Forked execution, worktree isolation, custom agent definitions
- **MCP integration**: JSON-RPC stdio/http transports, prompts as slash commands
- **Permission engine**: 7 modes, glob allowlist/denylist, team-managed policy overlays
- **Session persistence**: JSONL event log + SQLite projections, deterministic replay
- **Local ML (opt-in)**: Candle-powered embeddings and code completion, HNSW vector index — runs fully offline

## Architecture

DeepSeek CLI is a Rust workspace organized into focused crates:

| Crate | Role |
|-------|------|
| `deepseek-cli` | CLI dispatch, argument parsing, 30+ subcommand handlers |
| `deepseek-agent` | Agent engine, tool-use loop, complexity classifier, prompt construction, team mode |
| `deepseek-core` | Shared types (`AppConfig`, `ChatRequest`, `StreamChunk`, `EventEnvelope`), config loading |
| `deepseek-llm` | LLM client (`LlmClient` trait), streaming, prompt cache |
| `deepseek-tools` | Tool definitions (enriched descriptions), plugin manager, shell runner, sandbox wrapping |
| `deepseek-policy` | Permission engine (denylist/allowlist), approval gates, output scanner, `ManagedSettings` |
| `deepseek-hooks` | 14 lifecycle events, `HookRuntime`, once/disabled fields, `PermissionDecision` |
| `deepseek-local-ml` | Local ML via Candle: embeddings, completion, chunking, vector index, hybrid retrieval, privacy router |
| `deepseek-store` | Session persistence (JSONL event log + SQLite projections) |
| `deepseek-memory` | Long-term memory, shadow commits, checkpoints |
| `deepseek-index` | Full-text code index (Tantivy), RAG retrieval with citations |
| `deepseek-mcp` | MCP server management (JSON-RPC stdio/http transports) |
| `deepseek-ui` | TUI rendering (ratatui/crossterm), autocomplete, vim mode, ML ghost text |
| `deepseek-diff` | Unified diff parsing, patch staging, and git-apply |
| `deepseek-context` | Context enrichment, dependency analysis (petgraph), file relevance scoring |
| `deepseek-skills` | Skill discovery, forked execution, frontmatter parsing |
| `deepseek-subagent` | Background tasks, worktree isolation, custom agent definitions |
| `deepseek-observe` | Structured logging |
| `deepseek-jsonrpc` | JSON-RPC server for IDE integration |
| `deepseek-chrome` | Chrome native host bridge |
| `deepseek-errors` | Error types |
| `deepseek-testkit` | Test utilities |

The default execution mode is the **tool-use loop** (think→act→observe):

```
User → LLM (with tools) → Tool calls → Results → LLM → ... → Final response
```

- The LLM freely decides which tools to call (file read/write, shell, search, etc.)
- Tools execute locally with policy-gated approval for write operations
- Checkpoints are created before destructive tool calls
- The loop continues until the LLM responds without tool calls (task complete)
- Thinking mode (`deepseek-reasoner`) can be enabled for complex reasoning
- Adaptive complexity: Simple/Medium/Complex classification with thinking budget escalation (8K→64K)
- Bootstrap context: automatic project awareness (tree, git status, repo map, manifests) on first turn
- Hybrid retrieval: vector + BM25 search with Reciprocal Rank Fusion (RRF)
- Semantic compaction: preserves file/error/decision context when compacting long conversations
- Error recovery: automatic guidance injection on failures, stuck detection after repeated errors
- Model routing: Complex+escalated tasks route to `deepseek-reasoner` automatically

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

Settings merge in order (later wins):

1. `~/.deepseek/settings.json` (user — all projects)
2. `.deepseek/settings.json` (project — shared with team)
3. `.deepseek/settings.local.json` (local overrides — gitignore this)

```bash
deepseek config show               # View merged config (keys redacted)
deepseek config edit               # Open in editor
```

Key settings:

```json
{
  "llm": {
    "base_model": "deepseek-chat",
    "max_think_model": "deepseek-reasoner",
    "context_window_tokens": 128000,
    "api_key_env": "DEEPSEEK_API_KEY"
  },
  "agent_loop": {
    "tool_loop_max_turns": 50,
    "context_bootstrap_enabled": true
  },
  "policy": {
    "approve_edits": "ask",
    "approve_bash": "ask",
    "allowlist": ["cargo *", "npm test", "git status"]
  },
  "local_ml": {
    "enabled": false
  }
}
```

See `docs/CONFIG_REFERENCE.md` for all options.

## Local ML (Optional)

An optional local intelligence layer — hybrid code retrieval, privacy scanning, and ghost text completions. No models bundled; downloads from HuggingFace on first use.

```bash
# Enable (mock backends, no download needed)
echo '{"local_ml": {"enabled": true}}' > .deepseek/settings.json

# Enable with real ML models (requires --features local-ml build)
cargo build --release --bin deepseek --features local-ml
```

| Feature | Mock Mode | Full ML Mode |
|---------|-----------|-------------|
| Code retrieval | Hash-based matching | Semantic (jina-code-v2, ~270MB) |
| Ghost text | Empty | Candle completion (~700MB–1.5GB) |
| Privacy scanning | Fully functional | Fully functional |
| Vector index | BruteForce O(n) | HNSW via Usearch |

```bash
deepseek privacy scan              # Find secrets in your project
deepseek privacy redact-preview    # Preview what gets redacted
deepseek index build               # Build hybrid index
deepseek index --hybrid doctor     # Diagnose index issues
```

See `docs/LOCAL_ML_GUIDE.md` for the full guide.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets    # 870 tests
cargo build --release --bin deepseek
```

## Docs

- **[User Guide](docs/USER_GUIDE.md)** — complete guide: chat modes, keyboard shortcuts, slash commands, tools, intelligence layers, permissions, hooks, skills, subagents, MCP, local ML, sessions, autopilot, code review, configuration, troubleshooting
- **[Local ML Guide](docs/LOCAL_ML_GUIDE.md)** — setup, models, retrieval, privacy, ghost text
- **[Configuration Reference](docs/CONFIG_REFERENCE.md)** — all settings with types and defaults
- **[Operations Playbook](docs/OPERATIONS.md)** — incident response, failure modes, rollback
- **[Release Guide](docs/RELEASE.md)** — release process, artifacts, installers
