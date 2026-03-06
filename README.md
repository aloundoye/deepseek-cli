# CodingBuddy

CodingBuddy is a terminal-native coding agent written in Rust. It combines chat, planning, tool execution, patch workflows, indexing, and long-running autopilot loops in one cross-platform CLI.

## Highlights
- **Agent intelligence**: Adaptive complexity (Simple/Medium/Complex), planning protocols, error recovery, stuck detection
- **Phase overlay for complex tasks**: Complex tasks stay in the tool-use loop but add Explore→Plan→Execute→Verify with per-phase tool filtering
- **Post-edit validation**: LSP-like diagnostics (`cargo check`, `tsc`, `py_compile`, `go vet`) fed back to the LLM for self-correction
- **Per-model system prompts**: Optimized prompts for DeepSeek chat/reasoner, Qwen (concise), and Gemini (methodical)
- **Agent profiles**: Task-specialized tool filtering (build/explore/plan) — reduces decision space for weaker models
- **Doom loop detection**: Detects and breaks repeated identical tool calls with corrective guidance
- **Bootstrap context**: Automatic project awareness (file tree, git status, repo map, dependency analysis) on first turn
- **Per-turn retrieval**: Vector + BM25 code search with RRF fusion — fires every turn, not just the first
- **Privacy scanning**: 3-layer secret detection (path/content/builtin patterns) with redaction on tool outputs
- **Ghost text**: Local ML-powered inline completions in the TUI (Tab to accept, Alt+Right for one word)
- **Provider compatibility boundary**: DeepSeek, OpenAI-compatible gateways, and Ollama are routed through capability-gated request/response shims instead of best-effort aliases
- **Operator diagnostics**: `status` and `doctor` expose provider compatibility transforms plus live local-runtime queue/load state
- **Model routing**: Automatically routes complex tasks to `deepseek-reasoner`, simple tasks to `deepseek-chat`
- **Step snapshots**: Per-tool-call file snapshots with content hashing for fine-grained undo
- **Default deny rules**: Built-in safety rules block dangerous operations (`rm -rf`, `git push --force`, `.env` edits)
- **14 lifecycle hooks**: SessionStart through TaskCompleted — extend behavior at every stage
- **Interactive TUI**: Vim mode, @file autocomplete, !bash prefix, syntax highlighting, keyboard shortcuts
- **Skills & subagents**: Forked execution, worktree isolation, custom agent definitions
- **MCP integration**: JSON-RPC stdio/http transports, prompts as slash commands
- **Permission engine**: 7 modes, glob allowlist/denylist, team-managed policy overlays, bypass mode
- **Session persistence**: JSONL event log + SQLite projections, deterministic replay
- **LLM compaction**: Structured LLM-based conversation compaction preserving goals, progress, and findings
- **Local ML (opt-in)**: Candle-powered embeddings and code completion, HNSW vector index, memory-aware loading — runs fully offline

## Architecture

CodingBuddy is a Rust workspace organized into focused crates:

| Crate | Role |
|-------|------|
| `codingbuddy-cli` | CLI dispatch, argument parsing, 24 subcommand handlers |
| `codingbuddy-agent` | Agent engine, tool-use loop, complexity classifier, prompt construction, phase loop, team mode |
| `codingbuddy-core` | Shared types (`AppConfig`, `ChatRequest`, `StreamChunk`, `TaskPhase`, `EventEnvelope`), config loading, session/task metadata |
| `codingbuddy-llm` | LLM client (`LlmClient` trait), streaming, prompt cache, cached API key resolution |
| `codingbuddy-tools` | Tool definitions (enriched descriptions), plugin manager, shell runner, sandbox wrapping |
| `codingbuddy-policy` | Permission engine (denylist/allowlist), approval gates, output scanner, `ManagedSettings`, default deny rules |
| `codingbuddy-hooks` | 14 lifecycle events, `HookRuntime`, once/disabled fields, `PermissionDecision` |
| `codingbuddy-lsp` | Post-edit validation: `cargo check`, `tsc`, `py_compile`, `go vet` diagnostics fed back to LLM |
| `codingbuddy-local-ml` | Local ML via Candle: embeddings, completion, chunking, vector index, hybrid retrieval, privacy router, memory-aware loading |
| `codingbuddy-store` | Session persistence (JSONL event log + SQLite projections) |
| `codingbuddy-memory` | Long-term memory, shadow commits, checkpoints, step snapshots |
| `codingbuddy-index` | Full-text code index (Tantivy), RAG retrieval with citations |
| `codingbuddy-mcp` | MCP server management (JSON-RPC stdio/http transports) |
| `codingbuddy-ui` | TUI rendering (ratatui/crossterm), autocomplete, vim mode, ML ghost text |
| `codingbuddy-diff` | Unified diff parsing, patch staging, and git-apply |
| `codingbuddy-context` | Context enrichment, dependency analysis (petgraph), file relevance scoring |
| `codingbuddy-skills` | Skill discovery, forked execution, frontmatter parsing |
| `codingbuddy-subagent` | Background tasks, worktree isolation, custom agent definitions |
| `codingbuddy-observe` | Structured logging |
| `codingbuddy-jsonrpc` | JSON-RPC server for IDE integration |
| `codingbuddy-chrome` | Chrome native host bridge |
| `codingbuddy-errors` | Error types |
| `codingbuddy-testkit` | Test utilities |

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
- Explicit phase overlay: Complex tasks stay in the tool-use loop while adding Explore→Plan→Execute→Verify with per-phase tool filtering
- Post-edit validation: LSP-like diagnostics after file edits, fed back to LLM for immediate self-correction
- Agent profiles: task-type tool filtering (build/explore/plan) reduces the model's decision space
- Doom loop detection: rolling hash window detects 3+ identical tool calls and injects corrective guidance
- Per-model system prompts: optimized for DeepSeek chat/reasoner, Qwen (concise), Gemini (methodical)
- Bootstrap context: automatic project awareness (tree, git status, repo map, manifests) on first turn
- Per-turn retrieval: vector + BM25 search with RRF, fires every turn with remaining-budget awareness
- LLM compaction: structured LLM-based summary (Goal/Completed/In Progress/Key Facts/Findings/Modified Files) with code-based fallback
- Step snapshots: before/after file state captured per tool call with SHA-256 hashing and revert support
- Error recovery: automatic guidance injection on failures, stuck detection after repeated errors
- Model routing: Complex+escalated tasks route to `deepseek-reasoner` automatically
- Default deny rules: built-in safety rules for dangerous operations (rm -rf, force push, .env edits)

This is a ReAct-style tool loop, not a hardcoded `Architect -> Editor -> Apply -> Verify` pipeline. For complex tasks, `Explore -> Plan -> Execute -> Verify` is an overlay on the same loop, not a separate runtime.

## Install

### Option 1: Prebuilt release binary (recommended)

macOS/Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/aloundoye/codingbuddy/main/scripts/install.sh | bash
```

Windows (PowerShell):

```powershell
$script = Join-Path $env:TEMP "codingbuddy-install.ps1"
Invoke-WebRequest https://raw.githubusercontent.com/aloundoye/codingbuddy/main/scripts/install.ps1 -OutFile $script
& $script
```

### Option 2: Build from source

```bash
cargo build --release --bin codingbuddy
./target/release/codingbuddy --help
```

## Quickstart

```bash
export DEEPSEEK_API_KEY="<your-api-key>"
codingbuddy chat
```

Non-interactive examples:

```bash
codingbuddy ask "Summarize this repository"
codingbuddy plan "Implement feature X and list risks"
codingbuddy autopilot "Execute plan and verify tests" --hours 2
```

Credential behavior:
- If the API key is missing in interactive TTY mode, CodingBuddy prompts for it before first model use.
- You can persist the key for the current workspace in `.codingbuddy/settings.local.json`.
- In non-interactive or JSON mode, missing credentials fail fast.
- `codingbuddy config show` redacts `llm.api_key`.

## Core Commands

Run `codingbuddy --help` for full details. The most used commands are:

- `codingbuddy chat`: interactive chat session (TUI when enabled and running in a TTY)
- `codingbuddy ask "<prompt>"`: one-shot response
- `codingbuddy plan "<prompt>"`: generate a plan without running it
- `codingbuddy run [session-id]`: continue execution for a session
- `codingbuddy autopilot "<prompt>"`: bounded unattended loop
- `codingbuddy review --diff|--staged|--pr <n>`: code review workflows
- `codingbuddy exec "<command>"`: policy-enforced shell execution
- `codingbuddy diff`, `codingbuddy apply`, `codingbuddy rewind`: patch/checkpoint lifecycle
- `codingbuddy status`, `codingbuddy doctor`, `codingbuddy context`, `codingbuddy usage`: runtime, provider compatibility, and cost visibility
- `codingbuddy index build|update|status|watch|query`: local code index operations
- `codingbuddy permissions show|set|dry-run`: safety policy inspection and tuning
- `codingbuddy replay run|list`: deterministic replay tooling
- `codingbuddy search "<query>"`: web search with provenance metadata
- `codingbuddy setup`: interactive setup wizard (provider, API key, local ML, privacy)

Useful global flags:
- `--json`: machine-readable output
- `--permission-mode ask|auto|plan`: per-run permission mode override

## Example Workflows

Plan and execute:

```bash
codingbuddy plan "Refactor auth middleware and keep behavior unchanged"
codingbuddy run
```

Autopilot with live status:

```bash
codingbuddy autopilot "Fix flaky tests in crates/codingbuddy-cli" --hours 1
codingbuddy autopilot status --follow
```

Review staged changes:

```bash
codingbuddy review --staged --focus correctness
```

Publish strict review findings to a PR:

```bash
codingbuddy review --pr 123 --publish --max-comments 20
```

Generate shell completions:

```bash
codingbuddy completions --shell zsh > ~/.zsh/completions/_codingbuddy
```

## Configuration

Settings merge in order (later wins):

1. `~/.codingbuddy/settings.json` (user — all projects)
2. `.codingbuddy/settings.json` (project — shared with team)
3. `.codingbuddy/settings.local.json` (local overrides — gitignore this)

```bash
codingbuddy config show               # View merged config (keys redacted)
codingbuddy config edit               # Open in editor
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
echo '{"local_ml": {"enabled": true}}' > .codingbuddy/settings.json

# Enable with real ML models (requires --features local-ml build)
cargo build --release --bin codingbuddy --features local-ml
```

| Feature | Mock Mode | Full ML Mode |
|---------|-----------|-------------|
| Code retrieval | Hash-based matching | Semantic (jina-code-v2, ~270MB) |
| Ghost text | Empty | Candle completion (~700MB–1.5GB) |
| Privacy scanning | Fully functional | Fully functional |
| Vector index | BruteForce O(n) | HNSW via Usearch |

```bash
codingbuddy privacy scan              # Find secrets in your project
codingbuddy privacy redact-preview    # Preview what gets redacted
codingbuddy index build               # Build hybrid index
codingbuddy index --hybrid doctor     # Diagnose index issues
```

See `docs/LOCAL_ML_GUIDE.md` for the full guide.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets    # 1,228 tests
cargo build --release --bin codingbuddy
```

## Docs

- **[User Guide](docs/USER_GUIDE.md)** — complete guide: chat modes, keyboard shortcuts, slash commands, tools, intelligence layers, permissions, hooks, skills, subagents, MCP, local ML, sessions, autopilot, code review, configuration, troubleshooting
- **[Local ML Guide](docs/LOCAL_ML_GUIDE.md)** — setup, models, retrieval, privacy, ghost text
- **[Configuration Reference](docs/CONFIG_REFERENCE.md)** — all settings with types and defaults
- **[Operations Playbook](docs/OPERATIONS.md)** — incident response, failure modes, rollback
- **[Release Guide](docs/RELEASE.md)** — release process, artifacts, installers
- **[Audit Closure Status](docs/AUDIT_CLOSURE_STATUS.md)** — tracked closure state for the three-codebase audit
