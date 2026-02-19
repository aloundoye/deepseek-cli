# RFC (Comprehensive): DeepSeek CLI Agent in Rust – A Full-Featured Coding Assistant to Match and Surpass Claude Code use DeepSeek API

**Status:** Completed  
**Date:** 2026-02-17  
**Goal:** A production-grade, open-source coding agent CLI powered by DeepSeek, designed to **match and surpass Claude Code** in feature completeness, developer experience, safety, and cost-effectiveness. This RFC defines the complete feature set, system architecture, and implementation roadmap.

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
| **Large context** | 1M token window, automatic context compression when nearing limits. |
| **Project-wide awareness** | Indexes entire codebase; understands file structure and dependencies. |
| **Checkpointing** | Auto-saves file edits; reversible via `/rewind` or keyboard shortcut. |
| **Fast Mode** | Optimized API parameters for lower latency (configurable). |
| **Parallel tool calls** | Execute multiple independent tool calls in a single LLM response. |
| **Background execution** | Long-running tasks can be backgrounded (Ctrl+B) and resumed. |

### 2.2 File System Operations

| Tool | Description |
|------|-------------|
| `fs.read` | View file contents with line numbers; supports images/PDFs. |
| `fs.write` | Create/overwrite files (safer than bash redirection; checkpointed). |
| `fs.edit` | Targeted modifications (string replacement, line edits). |
| `fs.glob` | Pattern-based file search. |
| `fs.grep` | Content search with regex. |
| `@file` references | Mention files directly: `@src/auth.ts:42-58`. |
| `@dir` references | Include whole directories. |

### 2.3 Git Integration

- Commit creation, branch management, pull requests from CLI.
- Git history search.
- Merge conflict resolution assistance.
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
- **Hooks:** `PreToolUse`, `PostToolUse`, `Stop` logic for fine-grained control.
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

### 2.11 Permission & Safety System

- **User approval:** Destructive operations require confirmation (delete, force-push, etc.).
- **Scope awareness:** Authorization matches requested scope only.
- **Blast radius consideration:** Checks reversibility before acting.
- **Security hardening:** No command injection, XSS, SQL injection.
- **Team permissions:** Managed permissions that cannot be overwritten locally.

### 2.12 DeepSeek-Specific Enhancements

| Enhancement | Description |
|--------------|-------------|
| **Cost efficiency** | ~16–20x cheaper than Claude; highlight in `/cost` command. |
| **Automatic “max thinking”** | Seamlessly escalate to `deepseek-reasoner` when complexity requires it. |
| **Prompt caching** | 90% discount on repeated input tokens; implement aggressive caching. |
| **Off-peak scheduling** | Option to defer non-urgent tasks to cheaper rate periods. |
| **Multi-provider support** | Pluggable backends: DeepSeek, OpenAI, local models (via MCP). |
| **Performance monitoring** | `/profile` command showing time breakdown. |
| **Integration marketplace** | Community repository of MCP plugins. |

---

## 3. System Architecture

The architecture is a modular monolith with clear separation of concerns, designed to support the above features while remaining maintainable and testable.

```
crates/
  cli/          # Entry point, TUI (ratatui), command parsing
  core/         # Agent runtime, session loop, scheduling
  agent/        # Planner, Executor, Subagent orchestration
  llm/          # DeepSeek client, streaming, retries, caching
  router/       # Model routing + auto max-think policy
  tools/        # Tool registry, sandboxed execution, MCP client
  mcp/          # MCP server management and protocol handling
  subagent/     # Parallel subagent lifecycle and communication
  diff/         # Patch staging, application, conflict resolution
  index/        # Tantivy code index + manifest + file watcher
  store/        # SQLite + event log; projections; migrations
  policy/       # Approvals, allowlists, redaction, permissions
  observe/      # Logs, metrics, tracing, cost tracking
  testkit/      # Replay harness, fake LLM, golden tests
  skills/       # Skill management and execution
  hooks/        # Pre/post tool hooks
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

- **Built-in Rust tools:** `fs.read`, `fs.edit`, `fs.grep`, `git.*`, `index.query`, etc.
- **MCP tools:** Dynamically loaded from MCP servers.
- **Shell commands:** Restricted via `bash.run` with allowlist and approval.

All tool calls are journaled (proposal, approval, result) in the event log. For `fs.edit`, we provide a high-level interface that accepts search/replace pairs and generates a unified diff internally.

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
- Hooks: PreToolUse and PostToolUse allow custom logic (e.g., logging, modifying arguments) written in Rust or via WASM (future).

### 4.11 MCP Integration

- MCP servers are managed via `deepseek mcp` commands.
- The MCP client discovers tools from servers and makes them available in the tool registry.
- Supports stdio and HTTP transports.
- Dynamic tool list updates via server notifications.

### 4.12 User Interface (TUI)

- Built with `ratatui` and `crossterm`.
- Layout:
  - Main conversation pane (scrollable, syntax highlighting).
  - Plan pane (collapsible) showing current plan and step progress.
  - Tool output pane (real-time logs from subagents or tool executions).
  - Status bar: model, cost, pending approvals, background jobs.
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
temperature = 0.2
api_key = "${DEEPSEEK_API_KEY}"  # or read from env

[router]
auto_max_think = true
threshold_high = 0.72
escalate_on_invalid_plan = true
max_escalations_per_unit = 1
weights = { prompt_complexity = 0.3, failure_streak = 0.25, ambiguity = 0.2, plan_confidence = 0.25 }

[policy]
approve_edits = "ask"          # ask, allow, deny
approve_bash = "ask"
allowlist = ["rg", "git", "cargo test", "cargo fmt --check", "npm test"]
block_paths = [".env", "**/id_*", "**/secret"]
redact_patterns = [
    "(?i)(api[_-]?key|token|secret)\\s*[:=]\\s*['\"]?[a-z0-9_\\-]{16,}['\"]?"
]

[index]
enabled = true
engine = "tantivy"
watch_files = true

[budgets]
max_reasoner_tokens_per_session = 1000000
max_turn_duration_secs = 300

[theme]
# ratatui color theme
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