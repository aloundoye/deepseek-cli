# RFC (Comprehensive): DeepSeek CLI Agent in Rust – A Full-Featured Coding Assistant to Match and Surpass Claude Code use DeepSeek API

**Status:** Completed
**Date:** 2026-02-19
**Goal:** A production-grade, open-source coding agent CLI powered by DeepSeek, designed to **match and surpass Claude Code** in feature completeness, developer experience, safety, and cost-effectiveness. Incorporates best features from Claude Code, OpenAI Codex CLI, and Google Antigravity. This RFC defines the complete feature set, system architecture, and implementation roadmap.

---

## 1. Introduction

Claude Code has set a high bar for terminal-based AI coding assistants. DeepSeek’s models offer compelling advantages: significantly lower cost, a dedicated reasoning model (`deepseek-reasoner`), and an open ecosystem. By combining a thoughtful architecture with DeepSeek’s unique strengths, we can build a tool that not only replicates Claude Code’s capabilities but also introduces innovations like **automatic model escalation**, **deterministic replay**, and **aggressive cost optimization**.

This RFC outlines the complete feature set, system design, and incremental implementation plan for the **DeepSeek CLI Agent** (codename: `deepseek`). It incorporates every feature from Claude Code as of February 2026, plus DeepSeek-specific enhancements, ensuring a truly competitive offering.

---

## 2. Complete Feature Set

The following features are derived from Claude Code’s current (2026) capabilities, extended with DeepSeek-specific improvements.

### 2.1 Core Runtime & Environment

| Feature | Description |
|---------|-------------|
| **Terminal-native CLI** | Full shell integration, interactive REPL with rich TUI with Ratatui. |
| **Cross-platform** | macOS 13.0+, Ubuntu 20.04+/Debian 10+, Windows 10+ (WSL/Git Bash). |
| **Multiple installation methods** | Native binary, Homebrew, Winget, direct download. |
| **Session persistence** | Resume sessions with `deepseek run`; checkpointing via event log. |
| **Large context** | 128K token window (DeepSeek API limit), automatic context compression when nearing limits. |
| **Project-wide awareness** | Indexes entire codebase; understands file structure and dependencies. |
| **Checkpointing** | Auto-saves file edits; reversible via `/rewind` or keyboard shortcut. |
| **Fast Mode** | Optimized API parameters for lower latency (configurable). |
| **Parallel tool calls** | Execute multiple independent tool calls in a single LLM response. |
| **Background execution** | Long-running tasks can be backgrounded (Ctrl+B) and resumed. |
| **Print mode (`-p`)** | Non-interactive headless mode for CI/CD and scripting. Reads prompt from arg or stdin pipe. Supports `--output-format` (text/json/stream-json). |
| **Session continue** | `--continue` resumes last session; `--resume <ID>` resumes a specific session. Full conversation context is restored. |
| **Model/provider override** | `--model` and `--provider` flags for per-invocation overrides. |

### 2.2 File System Operations

| Tool | Description |
|------|-------------|
| `fs.read` | View file contents with line numbers; supports images/PDFs. |
| `fs.write` | Create/overwrite files (safer than bash redirection; checkpointed). |
| `fs.edit` | Targeted modifications (string replacement, line edits). |
| `fs.glob` | Pattern-based file search. |
| `fs.grep` | Content search with regex. |
| `web.fetch` | Fetch URL content, auto-extract text from HTML. Configurable timeout and max_bytes. |
| `web.search` | Web search returning structured results (title, URL, snippet, provenance). Cached with TTL. |
| `@file` references | Mention files directly: `@src/auth.ts:42-58`. |
| `@dir` references | Include whole directories. |

### 2.3 Git Integration

- Commit creation, branch management, pull requests from CLI.
- Git history search.
- Merge conflict resolution assistance.
- **Code review**: `deepseek review` analyzes diffs (`--diff`, `--staged`, `--pr NUMBER`) and provides structured feedback with severity levels (critical/warning/suggestion). Inspired by Codex CLI's `/review`.
- Checkpointing note: Only file edits (via dedicated tools) are auto-checkpointed; bash operations are not tracked.

### 2.4 Slash Commands (REPL)

| Command | Purpose |
|---------|---------|
| `/help` | Show all commands. |
| `/init` | Scan project and create `DEEPSEEK.md` (memory file). |
| `/clear` | Clear conversation history. |
| `/compact` | Summarize conversation to save context. |
| `/memory` | Edit `DEEPSEEK.md` memory files. |
| `/config` | Open configuration interface. |
| `/model` | Switch between models (chat/reasoner) manually. |
| `/cost` | Show token usage and estimated cost. |
| `/mcp` | Manage MCP server connections. |
| `/rewind` | Rewind to previous checkpoint. |
| `/export` | Export conversation to file. |
| `/plan` | Enable plan mode for complex tasks. |
| `/teleport` | Resume sessions at (future) web interface. |
| `/remote-env` | Configure remote sessions. |
| `/status` | Current model, mode, and configuration. |
| `/effort` | Control thinking depth (low/medium/high/max). |
| `/context` | Inspect context window: token breakdown by source (system prompt, conversation, tools, memory). |
| `/permissions` | View/change permission mode (ask/auto/locked). Dry-run evaluator shows what a tool call would produce under current mode. |
| `/sandbox` | View/configure sandbox mode (allowlist/isolated/off/workspace-write/read-only). |
| `/agents` | List running and completed subagents with status and output summaries. |
| `/tasks` | Open Mission Control: view/manage task queue, reorder priorities, inspect artifacts. |
| `/review` | Start a read-only code review pipeline. Presets: security, perf, style, PR-ready. Accepts `--diff`, `--staged`, `--pr N`, `--path P`, `--focus F`. |
| `/search` | Web search: query the web and include results as context. Cached with TTL. Provenance metadata attached to results. |
| `/terminal-setup` | Configure shell integration, prompt markers, and terminal capabilities. |
| `/keybindings` | Edit keyboard shortcuts interactively; opens `~/.deepseek/keybindings.json`. |
| `/doctor` | Run diagnostics: check API connectivity, config validity, tool availability, index health, disk space. |

### 2.5 MCP (Model Context Protocol) Extensibility

- **Transports:** HTTP (remote servers), Stdio (local processes).
- **Installation scopes:** user (`~/.deepseek/mcp.json`), project (`.mcp.json`), local (`~/.deepseek/mcp.local.json`).
- **Management commands:** `deepseek mcp add/list/get/remove`.
- **Dynamic updates:** Servers can notify of tool changes.
- **Community integrations:** GitHub, Notion, PostgreSQL, Sentry, Slack, etc.
- **Xcode integration (future):** Capture previews for visual verification.

### 2.6 Subagent System

- **Parallel agents:** Up to 7 subagents running simultaneously.
- **Agent types:**
  - *Explore*: Research and gather information.
  - *Plan*: Break down tasks into steps.
  - *Task*: Execute specific subtasks (e.g., refactor a module).
- **Isolated contexts:** Subagents operate in clean contexts, preventing state pollution.
- **Agent Teams:** Multiple agents coordinate on different components (frontend, backend, testing).
- **Resilience:** Agents continue after permission denials, trying alternative approaches.

### 2.7 User Experience & Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Escape | Stop current response. |
| Escape + Escape | Open rewind menu. |
| ↑ | Navigate past commands. |
| Ctrl+V | Paste images (where supported). |
| Tab | Autocomplete file paths and commands. |
| Shift+Enter | Multi-line input. |
| Ctrl+B | Background agents and shell commands. |
| Ctrl+O | Toggle transcript mode (show raw thinking). |
| Ctrl+C | Cancel current operation. |
| Shift+Tab | Cycle permission mode (ask → auto → locked). |
| Ctrl+T | Toggle Mission Control pane (tasks/subagents). |
| Ctrl+A | Toggle Artifacts pane. |

### 2.8 Configuration System

| File | Purpose |
|------|---------|
| `~/.deepseek/settings.json` | User-level settings. |
| `.deepseek/settings.json` | Project-specific settings (shared). |
| `.deepseek/settings.local.json` | Per-machine overrides (gitignored). |
| `~/.deepseek/mcp.json` | MCP server configurations. |
| `.mcp.json` | Project-scoped MCP servers. |
| `~/.deepseek/keybindings.json` | Custom keyboard shortcuts. |
| `DEEPSEEK.md` | Project memory, conventions, workflows. |

### 2.9 Customization & Skills

- **Skills:** Reusable prompt templates stored in `~/.deepseek/skills/` or `.deepseek/skills/`; appear in slash command menu.
- **Hot reload:** Skills available immediately without restart.
- **Hooks:** Extended lifecycle hooks for fine-grained control:
  - `SessionStart` — fired when a session begins or resumes.
  - `PreToolUse` — fired before tool execution (can block).
  - `PostToolUse` — fired after successful tool execution.
  - `PostToolUseFailure` — fired after tool execution fails.
  - `Stop` — fired when the agent completes a run (success or failure).
- **Wildcard permissions:** e.g., `Bash(npm *)` to allow any npm command without approval.
- **`respectGitignore`:** Control @-mention file picker behavior.

### 2.10 Advanced Capabilities

| Feature | Description |
|---------|-------------|
| **Visual verification** | Capture and analyze UI previews (Xcode, Flutter, etc.) – *future*. |
| **Multilingual output** | `language` setting for responses in Japanese, Spanish, etc. |
| **Adaptive thinking** | Model adjusts reasoning depth based on complexity. |
| **max effort parameter** | Highest level of reasoning for complex tasks. |
| **Terminal command sandboxing** | Experimental safety for command execution. |
| **Auto-approval rules** | Configurable permissions to reduce prompts. |
| **Memory across projects** | Enterprise-wide `DEEPSEEK.md` for consistent conventions. |
| **Auto-lint after edit** | Inspired by Aider: optional `lint_after_edit` config runs a linter automatically after `fs.edit`, including diagnostics in the tool result for self-healing. |
| **Web content fetching** | `web.fetch` tool retrieves URL content with HTML-to-text extraction, enabling documentation lookup and API exploration. |

### 2.11 Permission & Safety System

- **User approval:** Destructive operations require confirmation (delete, force-push, etc.).
- **Scope awareness:** Authorization matches requested scope only.
- **Blast radius consideration:** Checks reversibility before acting.
- **Security hardening:** No command injection, XSS, SQL injection.
- **Team permissions:** Managed permissions that cannot be overwritten locally.

### 2.12 Permission Modes

The CLI supports three runtime permission modes that govern how tool calls are authorized:

| Mode | Behavior |
|------|----------|
| **ask** (default) | Prompt the user for approval on each tool call that matches the policy gate (edits, bash, etc.). Read-only tools (fs.read, fs.glob, fs.grep, index.query) always pass. |
| **auto** | Auto-approve tool calls that match the allowlist. Calls not on the allowlist still prompt for approval. Ideal for workflows where the user trusts a known set of commands. |
| **locked** | Deny all non-read operations. No writes, no edits, no bash, no patch apply. Useful for review-only sessions or when exploring a codebase. |

**UX:**
- **Shift+Tab** cycles between modes at the REPL prompt: ask → auto → locked → ask.
- **Status bar** shows current mode as a colored indicator: `[ASK]` (yellow), `[AUTO]` (green), `[LOCKED]` (red).
- **Event log:** Every mode change is recorded as `PermissionModeChangedV1 { from, to }` in the event stream.
- **`/permissions` command:** Displays current mode, lists what each tool would do under the active mode (dry-run evaluator), and allows switching modes interactively.
- **Team policy override:** If `team-policy.json` sets `permission_mode`, it cannot be overridden locally.
- **Configuration:** `policy.permission_mode` in settings (default: `"ask"`).

### 2.13 DeepSeek-Specific Enhancements

| Enhancement | Description |
|--------------|-------------|
| **Cost efficiency** | ~16–20x cheaper than Claude; highlight in `/cost` command. |
| **Automatic “max thinking”** | Seamlessly escalate to `deepseek-reasoner` when complexity requires it. |
| **Prompt caching** | 90% discount on repeated input tokens; implement aggressive caching. |
| **Off-peak scheduling** | Option to defer non-urgent tasks to cheaper rate periods. |
| **Multi-provider support** | Pluggable backends via `llm.provider`: `deepseek` (default), `openai`, `anthropic`, `custom`, `local`, `ollama`. All use OpenAI-compatible chat format. Inspired by Codex CLI and Aider's model-agnostic approach. |
| **Performance monitoring** | `/profile` command showing time breakdown. |
| **Integration marketplace** | Community repository of MCP plugins. |

---

## 3. System Architecture

The architecture is a modular monolith with clear separation of concerns, designed to support the above features while remaining maintainable and testable.

```
crates/
  cli/          # Entry point, command parsing, print mode, review subcommand
  ui/           # TUI (ratatui + crossterm), theme, layout, keybindings
  core/         # Shared config types, session loop, scheduling
  agent/        # Planner, Executor, Subagent orchestration, session hooks
  llm/          # Multi-provider LLM client, streaming, retries, caching
  router/       # Model routing + auto max-think policy
  tools/        # Tool registry, sandboxed execution, web.fetch, auto-lint
  mcp/          # MCP server management and protocol handling
  subagent/     # Parallel subagent lifecycle and communication
  diff/         # Patch staging, application, conflict resolution
  index/        # Tantivy code index + manifest + file watcher
  store/        # SQLite + event log; projections; session queries
  policy/       # Approvals, allowlists, redaction, sandbox enforcement
  observe/      # Logs, metrics, tracing, cost tracking
  memory/       # DEEPSEEK.md management, project memory, cross-project conventions
  testkit/      # Replay harness, fake LLM, golden tests
  skills/       # Skill management and execution
  hooks/        # Extended lifecycle hooks (5 phases)
```

### 3.1 Core Components

| Component | Responsibility |
|-----------|----------------|
| **Agent Runtime** | Owns the session loop, manages state machine, persists events. |
| **Planner** | Creates/revises plans using LLM; may spawn subagents for exploration. |
| **Executor** | Executes plan steps via tools; requests plan refinement when needed. |
| **Subagent Manager** | Spawns, monitors, and communicates with parallel subagents. |
| **Model Router** | Decides which model (chat vs reasoner) to use per call; logs decisions. |
| **Tool Host** | Executes tools with policy enforcement, journaling, and sandboxing. |
| **MCP Client** | Discovers and invokes tools from MCP servers. |
| **Index Service** | Provides deterministic code search and repo snapshots. |
| **Event Store** | Append-only log of all events; source of truth for replay. |
| **Policy Engine** | Evaluates tool calls against allowlists, redaction rules, and user approvals. |

---

## 4. Detailed Design

### 4.1 Session Lifecycle and State Machine

The session state machine (as defined in the revised RFC) governs the overall flow. States: `Idle`, `Planning`, `ExecutingStep`, `AwaitingApproval`, `Verifying`, `Completed`, `Paused`, `Failed`. All transitions are recorded as events.

### 4.2 Planner and Executor

- **Planner** generates a JSON plan (see Section 3.2 of revised RFC). It may use the reasoner for complex planning tasks.
- **Executor** follows the plan, but can request **refinement** mid-step if ambiguity arises (e.g., search returns multiple candidates). Refinement requests go back to the planner with current context.

### 4.3 Subagent System

Subagents are lightweight agent instances running in parallel. Each has its own context fork (isolated from the main session). Communication happens via the subagent manager, which can merge results back into the main plan.

**Design:**

- Subagents are spawned with a specific goal (e.g., “explore API usage in module X”).
- They execute using the same tool set but with a clean context.
- Results are returned as structured data (e.g., findings, plan fragments).
- The main agent can incorporate these results into the next planning step.

**Agent Teams:** Multiple subagents can be coordinated as a team (e.g., frontend, backend, testing agents working on different parts of a feature). The manager ensures they don’t step on each other’s toes.

### 4.4 Model Router with Auto Max-Think

The router computes a complexity score based on:

- Prompt complexity (length, keywords).
- Repo complexity (touched files, index breadth).
- Failure history (consecutive tool failures, test failures).
- Planner confidence (self-reported).
- User intent hints (`/plan`, `/effort high`).
- Latency budget (interactive vs batch).

Thresholds and weights are configurable. The router logs each decision (including the feature vector snapshot) to the event log, enabling post-hoc analysis and tuning.

**Escalation retry:** If the chat model produces an invalid plan or stalls, the router automatically retries with the reasoner once.

### 4.5 Tool System

Tools are categorized as:

- **Built-in Rust tools:** `fs.read`, `fs.write`, `fs.edit`, `fs.glob`, `fs.grep`, `web.fetch`, `git.*`, `index.query`, etc.
- **MCP tools:** Dynamically loaded from MCP servers.
- **Shell commands:** Restricted via `bash.run` with allowlist and approval.

All tool calls are journaled (proposal, approval, result) in the event log. For `fs.edit`, we provide a high-level interface that accepts search/replace pairs and generates a unified diff internally using an LCS-based algorithm.

**`web.fetch`:** Fetches URL content via HTTP GET, strips HTML tags to plain text, and truncates to a configurable max byte limit. Useful for documentation lookup, API exploration, and web content analysis.

**`web.search`:** Performs a web search query and returns structured results with title, URL, snippet, and provenance metadata (source, timestamp). Results are cached with a configurable TTL (default 15 minutes) to avoid redundant queries. Distinct from `web.fetch`: search returns a list of results for discovery, while fetch retrieves a single URL's content. The `/search` slash command provides interactive access.

**Review pipeline constraints:** When `/review` or `deepseek review` is active, the tool host enters **read-only mode**: `fs.write`, `fs.edit`, `patch.stage`, `patch.apply`, `bash.run`, and `web.fetch` are all forbidden. Only read tools (`fs.read`, `fs.glob`, `fs.grep`, `index.query`, `git.diff`, `git.log`, `git.show`) are permitted. This ensures the review pipeline cannot modify the codebase.

**Auto-lint after edit:** When `policy.lint_after_edit` is configured (e.g., `"cargo fmt --check"`), the tool host automatically runs the linter after every `fs.edit` and includes diagnostics in the tool result JSON. This enables the LLM to self-heal formatting/lint issues without a separate tool call.

### 4.6 Patch Staging and Application

- Model never writes files directly.
- All edits go through `patch.stage` (or `fs.edit` which calls it).
- Staged patches are stored as unified diffs against a known file SHA.
- Applying a patch requires SHA match or merge strategy (3-way merge if git repo).
- Conflicts are presented to the user for resolution.

### 4.7 Indexing and Deterministic Snapshots

- Uses Tantivy for fast search.
- Manifest stores baseline commit (if git) or list of file SHAs.
- For non-git projects, initial index is built by walking the FS; a file watcher (`notify`) can incrementally update the index.
- Queries are tagged `fresh` or `stale` based on manifest staleness.

### 4.8 Event Store and Deterministic Replay

- All session events are appended to `events.jsonl` (canonical log).
- SQLite stores projections for fast querying (current state, conversation, etc.).
- **Replay mode:** Given a session ID, the system reads the event log and replays all LLM calls and tool results from stored events, **without re-executing** tools or calling the LLM again. This enables deterministic testing and debugging.

### 4.9 Permission and Policy Engine

- Policy rules are defined in TOML (see config example).
- For each tool call, the policy engine checks:
  - Is the tool allowed? (allowlist)
  - Are the arguments safe? (path traversal checks, secret patterns)
  - Does the tool require approval? (ask/allow/deny)
- Approvals are collected via TUI prompts (or auto-approved if configured).
- Redaction: before sending prompts to LLM, the system scans for secrets (API keys, tokens) based on regex patterns and redacts them (configurable).

### 4.10 Skills and Hooks

- Skills are stored as Markdown files with frontmatter (similar to Claude Code). They can include tool usage patterns.
- Hooks provide extended lifecycle control at five phases:
  - **SessionStart** — fired when a session begins or resumes. Useful for environment setup.
  - **PreToolUse** — fired before tool execution; can block the call (e.g., for custom approval logic).
  - **PostToolUse** — fired after successful tool execution; receives tool result.
  - **PostToolUseFailure** — fired after tool execution fails; receives error details. Enables custom error recovery or alerting.
  - **Stop** — fired when the agent completes a run (success or failure). Useful for cleanup, notifications, or metrics.
- Hooks can be written in Rust or via WASM (future, behind `experiments.wasm_hooks` flag).

### 4.11 MCP Integration

- MCP servers are managed via `deepseek mcp` commands.
- The MCP client discovers tools from servers and makes them available in the tool registry.
- Supports stdio and HTTP transports.
- Dynamic tool list updates via server notifications.

### 4.12 User Interface (TUI)

- **Real-time streaming:** LLM responses are streamed token-by-token to the terminal via SSE parsing. The `LlmClient::complete_streaming()` method reads the HTTP response line-by-line, invoking a callback for each `content` or `reasoning_content` delta. In print mode (`-p`), tokens are flushed to stdout immediately. In `--output-format stream-json`, each chunk is emitted as a JSON line (`{"type":"content","text":"..."}`) for programmatic consumption. The `AgentEngine` holds an optional `StreamCallback` that, when set, is used for the first LLM call in the run.
- Built with `ratatui` and `crossterm`.
- Layout:
  - Main conversation pane (scrollable, syntax highlighting).
  - Plan pane (collapsible) showing current plan and step progress.
  - Tool output pane (real-time logs from subagents or tool executions).
  - **Mission Control pane** (toggle via `/tasks` or Ctrl+T): task queue with status indicators, subagent swimlanes, priority reordering. Shows running/pending/completed tasks with elapsed time and token usage.
  - **Artifacts pane** (toggle via Ctrl+A): displays task artifacts from `.deepseek/artifacts/<task-id>/`. Each task bundle contains `plan.md`, `diff.patch`, `verification.md`. Artifacts are browsable and diffable inline.
  - Status bar: model, cost, pending approvals, background jobs, **permission mode indicator** (`[ASK]`/`[AUTO]`/`[LOCKED]`).
- Keyboard shortcuts as listed in section 2.7.
- Multi-line input with Shift+Enter.
- Autocomplete for commands and file paths.

### 4.13 CLI Commands (Non-Interactive)

In addition to the REPL, the CLI supports one-shot commands:

- `deepseek ask "<prompt>"` – single response (no tools by default).
- `deepseek plan "<prompt>"` – generate a plan and exit.
- `deepseek autopilot "<prompt>"` – run autonomous mode (may continue in background).
- `deepseek run <session-id>` – resume a session.
- `deepseek diff` – show staged changes.
- `deepseek apply` – apply staged patches.
- `deepseek index build|update|status|query` – manage index.
- `deepseek config edit|show` – edit configuration.
- `deepseek review [--diff|--staged|--pr N|--path P] [--focus F]` – structured code review with severity levels.
- `deepseek exec "<command>"` – execute a shell command under policy enforcement (allowlist, sandbox, approval) and print structured output.
- `deepseek tasks [list|show <id>|cancel <id>]` – manage the task queue from the command line.
- `deepseek doctor` – run diagnostics (API connectivity, config validation, tool/binary availability, index health, disk space, sandbox integrity).
- `deepseek search "<query>"` – web search from CLI, returns results with provenance metadata.

**Print mode (`-p`):** Any command can be run non-interactively with the `-p` flag. Reads prompt from trailing arguments or stdin pipe. Supports `--output-format` (`text`, `json`, `stream-json`). Designed for CI/CD pipelines, scripting, and integration with other tools.

**Session continuation:**
- `--continue` / `-c` resumes the most recent session, restoring full conversation context.
- `--resume <UUID>` / `-r <UUID>` resumes a specific session by ID.
- `--fork-session <UUID>` forks an existing session: clones conversation + state into a new session with a fresh event stream. The original session is untouched. File locking prevents concurrent writes to the same session.

**Per-invocation overrides:**
- `--model <name>` overrides `llm.base_model` for this run.
- `--provider <name>` overrides `llm.provider` for this run.

---

## 5. Implementation Milestones

### M1: Core REPL + Basic Tools (Weeks 1–4)
- Project setup, workspace structure.
- Basic REPL with chat (no tools) using `deepseek-chat`.
- Session persistence (event log, SQLite).
- Read-only tools: `fs.list`, `fs.read`, `git.status`.
- Manual model switching via `/model`.

### M2: Plan-First Mode + Patch Staging (Weeks 5–8)
- Planner generates JSON plans (using chat model).
- Patch staging with `fs.edit` and `patch.apply`.
- Approval flow for writes.
- Simple router with fixed thresholds.
- Add `/plan`, `/clear`, `/help` commands.

### M3: Verification Loops & Subagents (Weeks 9–12)
- Integration with test/lint tools (`run_tests`, `run_linter`).
- Verification step in plan execution.
- Basic subagent support: spawn, isolate, collect results.
- `/compact` command to summarize conversation.
- Background execution (Ctrl+B).

### M4: Auto Max-Think & Router Tuning (Weeks 13–16)
- Full router with feature vector and escalation logic.
- Router event logging and `/cost` command.
- Configurable thresholds; feedback collection.
- Add `/effort` command to influence router.

### M5: MCP Integration (Weeks 17–20)
- MCP client with stdio and HTTP support.
- `deepseek mcp` commands for server management.
- Dynamic tool discovery.
- Example MCP servers (GitHub, filesystem).

### M6: Skills, Hooks & Advanced Features (Weeks 21–24)
- Skill loading and execution.
- Pre/Post tool hooks.
- Wildcard permissions.
- `/memory` and `DEEPSEEK.md` support.
- Deterministic replay test harness.
- Performance monitoring (`/profile`).

### M7: Polishing & Ecosystem (Weeks 25–28)
- Cross-platform testing and packaging.
- Community plugin registry website.
- Documentation and tutorials.
- Auto-approval rules and team permissions.
- Visual verification (experimental).

---

## 6. Configuration Example (TOML)

```toml
[llm]
base_model = "deepseek-chat"
max_think_model = "deepseek-reasoner"
provider = "deepseek"              # deepseek | openai | anthropic | custom | local | ollama
profile = "v3_2"
context_window_tokens = 128000
temperature = 0.2
endpoint = "https://api.deepseek.com/chat/completions"
api_key_env = "DEEPSEEK_API_KEY"
fast_mode = false
language = "en"
prompt_cache_enabled = true
timeout_seconds = 60
max_retries = 3
retry_base_ms = 400
stream = true

[router]
auto_max_think = true
threshold_high = 0.72
escalate_on_invalid_plan = true
max_escalations_per_unit = 1
w1 = 0.2    # prompt_complexity
w2 = 0.15   # repo_complexity
w3 = 0.2    # failure_streak
w4 = 0.15   # planner_confidence
w5 = 0.2    # user_intent
w6 = 0.1    # latency_budget

[policy]
approve_edits = "ask"          # ask, allow, deny
approve_bash = "ask"
allowlist = ["rg", "git status", "git diff", "git show", "cargo test", "cargo fmt --check", "cargo clippy"]
block_paths = [".env", ".ssh", ".aws", ".gnupg", "**/id_*", "**/secret"]
redact_patterns = [
    '(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[''"]?[a-z0-9_\-]{8,}[''"]?',
    '\b\d{3}-\d{2}-\d{4}\b',
    '(?i)\b(mrn|medical_record_number|patient_id)\s*[:=]\s*[a-z0-9\-]{4,}\b',
]
sandbox_mode = "allowlist"     # allowlist | isolated | off
permission_mode = "ask"        # ask | auto | locked
# lint_after_edit = "cargo fmt --check"  # auto-lint after fs.edit

[plugins]
enabled = true
search_paths = [".deepseek/plugins", ".plugins"]
enable_hooks = false

[plugins.catalog]
enabled = true
index_url = ".deepseek/plugins/catalog.json"
signature_key = "deepseek-local-dev-key"
refresh_hours = 24

[skills]
paths = [".deepseek/skills", "~/.deepseek/skills"]
hot_reload = true

[usage]
show_statusline = true
cost_per_million_input = 0.27
cost_per_million_output = 1.10

[context]
auto_compact_threshold = 0.86
compact_preview = true

[autopilot]
default_max_consecutive_failures = 10
heartbeat_interval_seconds = 5
persist_checkpoints = true

[scheduling]
off_peak = false
off_peak_start_hour = 0
off_peak_end_hour = 6
defer_non_urgent = false
max_defer_seconds = 0

[replay]
strict_mode = true

[ui]
enable_tui = true
keybindings_path = "~/.deepseek/keybindings.json"
reduced_motion = false
statusline_mode = "minimal"

[experiments]
visual_verification = false
wasm_hooks = false

[telemetry]
enabled = false

[index]
enabled = true
engine = "tantivy"
watch_files = true

[budgets]
max_reasoner_tokens_per_session = 1000000
max_turn_duration_secs = 300

[theme]
primary = "Cyan"
secondary = "Yellow"
error = "Red"
```

---

## 7. Testing Strategy

### 7.1 Unit & Integration Tests
- Test each crate in isolation.
- Integration tests for end-to-end flows using fake LLM and tool mocks.

### 7.2 Deterministic Replay Tests
- Record real sessions (with user permission) into cassettes.
- Replay cassettes in CI to ensure changes don't break existing behavior.
- Golden files for expected plans and diffs.

### 7.3 Property-Based Testing
- Use `proptest` to generate sequences of user inputs and tool responses; verify state machine invariants.

### 7.4 Performance Benchmarks
- Benchmark index queries, tool execution, and LLM streaming latency.
- Set performance targets: p95 index query < 50ms, first token < 2s.

---

## 8. Conclusion

This RFC presents a complete blueprint for building a DeepSeek-powered coding agent that not only replicates every feature of Claude Code but also introduces DeepSeek-specific innovations. The modular architecture, emphasis on safety, deterministic replay, and incremental milestones ensure a path to a production-quality tool that can become the go-to open-source alternative.

With the Rust ecosystem’s maturity and the DeepSeek API’s cost advantage, we are well-positioned to deliver a tool that developers will love. Let’s build it.