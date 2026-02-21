# PLAN.md — DeepSeek CLI: Full Claude Code Feature Parity

Based on deep analysis of all 18 crates (~25k LOC) and exhaustive comparison with Claude Code (Feb 2026).

---

## Architecture Reality: What Changed from the Original Specs

### The Big Shift: Plan-and-Execute → Chat-with-Tools

The original codebase was built around a **Plan-and-Execute** architecture:
1. `SchemaPlanner` generates a JSON plan (list of steps) from the user prompt
2. `SimpleExecutor` iterates the plan, calling hardcoded tools in sequence
3. Each step is a single tool call with fixed arguments

**What we actually built** is a **Chat-with-Tools** loop (like Claude Code):
1. User prompt + system prompt + `tool_definitions()` → DeepSeek API `/chat/completions`
2. Model streams response with text and/or `tool_calls`
3. Agent executes tools, sends results back as `role: "tool"` messages, loops
4. When model returns text without tool_calls → turn complete

**Why**: The Chat-with-Tools pattern is strictly superior — the LLM decides which tools to call, can self-correct, handles multi-step reasoning natively, and matches how modern coding agents (Claude Code, Cursor, Copilot) actually work.

**Impact**: `AgentEngine::chat_with_options()` is now the single entry point. `SchemaPlanner`/`SimpleExecutor`/`run_once_with_mode()` are deprecated dead code (zero call sites). Will be removed in Phase 15.

### Legacy Planner: Dead Code — Remove

| Aspect | Plan-and-Execute | Chat-with-Tools |
|--------|-----------------|-----------------|
| Tool selection | LLM must predict all steps upfront | LLM chooses tools dynamically per turn |
| Error recovery | None — plan fails, full retry | LLM sees error, adapts strategy |
| Multi-step reasoning | Limited to plan structure | Native chain-of-thought |
| Streaming | Not supported | Full streaming with tool call deltas |

**Plan mode** (like Claude Code) will be implemented as a Chat-with-Tools variant with restricted read-only tools — NOT as the old Planner architecture. See Phase 6.

---

## Phase 0: Critical Bugs (Blocks Basic Functionality) ✅

### 0.1 Fix tool definition parameter name mismatches ✅
### 0.2 Fix stream callback consumed and never re-installed ✅
### 0.3 Fix chat() not using session state transitions ✅
### 0.4 Fix chat() not loading prior turns on --continue/--resume ✅
### 0.5 Fix chat() not persisting memory observations ✅

---

## Phase 1: Make the Chat Loop Production-Ready ✅

### 1.1 Enrich system prompt ✅
### 1.2 Add context window management ✅
### 1.3 Wire all entry points to chat() ✅
### 1.4 Integrate model router in chat() — Hybrid Model Strategy ✅
### 1.5 Add tool execution feedback to stream ✅

---

## Phase 2: Tool System Completeness ✅

### 2.1 Add missing tools to tool_definitions() ✅
### 2.2 Fix fs_edit tool definition to match all edit modes ✅
### 2.3 Implement AskUserQuestion tool ✅
### 2.4 Implement task tracking tools (TodoWrite equivalent) ✅
### 2.5 Implement subagent/Task tool ✅
### 2.6 Improve tool output truncation ✅

---

## Phase 3: TUI / User Experience ✅

### 3.1 Improve markdown rendering ✅
### 3.2 Add tool call display in transcript ✅
### 3.3 Add cost/token tracking in status bar ✅
### 3.4 Fix spinner during tool execution ✅
### 3.5 Right panel toggle — SKIPPED
### 3.6 Improve slash command output ✅

---

## Phase 4: CLI Flags for Claude Code Parity ✅

### 4.1–4.8 All done ✅

---

## Phase 5: Conversation & Memory ✅

### 5.1 Multi-turn conversation persistence ✅
### 5.2 Improve error messages ✅
### 5.3 Implement proper token counting ✅
### 5.4 Cost tracking per conversation ✅

---

## Phase 6: Plan Mode & Subagent System ✅

> Claude Code's two most powerful orchestration features: Plan Mode (think-before-act) and Task/Subagents (parallel work delegation).

### 6.1 Implement Plan Mode — EnterPlanMode / ExitPlanMode tools
- Add `enter_plan_mode` tool definition: LLM can proactively enter plan mode
- Add `exit_plan_mode` tool definition with `allowedPrompts` parameter (bash permissions needed for execution)
- When plan mode active: restrict tool set to **read-only** only (fs_read, fs_glob, fs_grep, fs_list, git_status, git_diff, git_show, web_fetch, web_search, index_query)
- Block all write/execute tools (fs_write, fs_edit, bash_run, multi_edit, patch_stage, patch_apply, notebook_edit)
- LLM writes plan to a file (configurable via `plansDirectory` setting)
- User reviews plan → approves → tools unlocked for execution
- Shift+Tab keyboard shortcut to cycle: normal → plan → normal
- `/plan` slash command to enter plan mode from prompt
- `--permission-mode plan` flag already exists (Phase 4.1) — wire to this behavior

### 6.2 Wire subagent worker in CLI
- Call `engine.set_subagent_worker()` in CLI chat setup with a real worker closure
- Worker creates a child `AgentEngine` with its own context window
- Child agent gets reduced tool set based on subagent type
- ~20 LOC of glue — all infrastructure exists in deepseek-subagent

### 6.3 Subagent types matching Claude Code
- **Explore** agent: Read-only tools (Glob, Grep, Read, read-only Bash), uses fast model, thoroughness levels (quick/medium/thorough)
- **Plan** agent: Read-only tools, researches codebase and designs implementation
- **Bash** agent: Only Bash tool, for command execution in separate context
- **General-purpose** agent: All tools, for complex multi-step tasks
- Configure tool restrictions per agent type in the worker closure
- `subagent_type` parameter on `spawn_task` tool already exists

### 6.4 Subagent background execution
- `run_in_background` parameter: spawn subagent, return immediately with output file path
- Use existing `BackgroundJobStartedV1` events
- `task_output` tool: retrieve output from background subagent (like Claude Code's TaskOutput)
- `task_stop` tool: kill a running background subagent (like Claude Code's TaskStop)

### 6.5 Subagent resume & model selection
- `resume` parameter on `spawn_task`: continue a previous subagent with preserved context
- `model` parameter: override model per subagent (e.g., use fast model for Explore)
- `max_turns` parameter: limit subagent turns

### 6.6 Custom subagents from YAML files
- Load from `.deepseek/agents/`, `~/.deepseek/agents/`
- YAML frontmatter: `name`, `description`, `tools`, `disallowedTools`, `model`, `maxTurns`, `permissionMode`, `skills`, `hooks`, `memory`, `background`, `isolation`
- `isolation: worktree` — run in isolated git worktree
- Register in `spawn_task` tool's available types
- `--agents` CLI flag for dynamic subagent definitions (JSON)

---

## Phase 7: Missing LLM Tools ✅

> Tools that Claude Code's LLM can call but ours can't yet.

### 7.1 TaskGet & TaskList tools ✅ (done in Phase 6)

### 7.2 Skill tool (LLM invokes slash commands) ✅
- `skill` tool definition: LLM can invoke any registered skill
- Parameters: `skill` (name), `args` (optional arguments)
- Routes to SkillManager in deepseek-skills crate

### 7.3 Background task management tools ✅ (done in Phase 6)

### 7.4 KillShell tool + background bash ✅
- `kill_shell` tool: terminate a background bash process by shell_id
- `bash_run` now has `run_in_background` and `description` parameters
- Background bash processes tracked in AgentEngine, retrievable via `task_output`

---

## Phase 8: Full Hooks System ✅

> Claude Code has 14 hook events with 3 handler types. We have basic pre/post tool hooks with command handler only.

### 8.1 Expand hook events to match Claude Code ✅
- **SessionStart** — fires on session begin/resume/clear/compact (matcher: startup, resume, clear, compact)
- **UserPromptSubmit** — fires when user submits prompt (can block)
- **PreToolUse** — before tool execution (can allow/deny/ask) — we have this partially
- **PostToolUse** — after tool succeeds (feedback only) — we have this partially
- **PostToolUseFailure** — after tool fails (feedback only)
- **PermissionRequest** — when permission dialog shown (can allow/deny)
- **Notification** — on notification events (permission_prompt, idle_prompt, auth_success)
- **SubagentStart** — when subagent spawned (context injection)
- **SubagentStop** — when subagent finishes (can block)
- **Stop** — when main agent finishes responding (can block)
- **ConfigChange** — when config files change mid-session
- **PreCompact** — before context compaction (inject instructions)
- **SessionEnd** — session terminates (clear, logout, exit)
- **TaskCompleted** — task marked complete

### 8.2 Hook handler types ✅ (command handler implemented; prompt/agent handlers deferred to Phase 16)
- **command** (implemented): shell command, JSON on stdin, exit codes (0=allow, 2=block)
- **prompt**: single-turn LLM evaluation — deferred
- **agent**: multi-turn subagent with read-only tools — deferred

### 8.3 Hook input modification & decision control ✅
- `updatedInput`: PreToolUse hooks can modify tool parameters before execution
- `permissionDecision`: allow/deny/ask for PreToolUse and PermissionRequest
- `additionalContext`: inject context into many hook events
- `decision: "block"` at top level to block any hookable event

### 8.4 Async hooks (deferred)
- `async: true` flag for non-blocking background hook execution
- `once` flag: run only once per session (for skills)
- Environment persistence: SessionStart hooks can write to env file

### 8.5 Hook configuration locations ✅
- `~/.deepseek/settings.json` (all projects) — via `AppConfig.hooks` JSON field
- `.deepseek/settings.json` (project, shareable)
- `.deepseek/settings.local.json` (project, gitignored)
- Plugin `hooks/hooks.json`
- Skill/agent YAML frontmatter (scoped to component lifetime)

---

## Phase 9: Memory & Configuration Parity ✅

> Claude Code has hierarchical memory, modular rules, @imports, and path-specific rules.

### 9.1 Modular rules directory ✅
- `.deepseek/rules/*.md` — project rules (shareable via git)
- `~/.deepseek/rules/*.md` — user-level rules
- Auto-load all `.md` files from rules directories into system prompt
- Path-specific rules: `paths:` YAML frontmatter with glob patterns for conditional loading

### 9.2 Hierarchical DEEPSEEK.md loading ✅
- Load DEEPSEEK.md recursively upward from cwd to filesystem root
- Child directory DEEPSEEK.md files loaded on demand
- `DEEPSEEK.local.md` — project-local, gitignored
- Load order: managed policy → user global → project → project local

### 9.3 @import syntax in DEEPSEEK.md ✅
- `@path/to/file` includes file contents into memory context
- Recursive import with max depth 5
- Relative paths resolved from DEEPSEEK.md location

### 9.4 `#` key shortcut for quick memory add — DEFERRED to Phase 11
- Press `#` in TUI to quick-add a memory entry
- Opens mini-editor for the memory content
- Appends to appropriate DEEPSEEK.md file

### 9.5 Extended permission modes ✅
- Current: `ask`, `auto`, `plan`
- Add: `acceptEdits` — auto-accepts file edits, still prompts for commands
- Add: `dontAsk` — auto-denies unless pre-approved via allow rules

### 9.6 Permission rule glob patterns ✅
- Format: `Tool(specifier)` with glob patterns
- `Bash(npm run *)`, `Bash(git commit *)` — command-specific bash permissions
- `Read(//absolute/path)`, `Edit(src/**/*.rs)` — path-specific file permissions
- `WebFetch(domain:example.com)` — domain-specific web permissions
- `Task(AgentName)` — subagent-specific control
- Evaluation order: **deny > ask > allow** (first match wins)

### 9.7 Settings parity ✅
- `plansDirectory` — where plan files are stored
- `outputStyle` — system prompt style adjustment
- `language` — preferred response language
- `attribution` — git commit/PR attribution text
- `availableModels` — restrict model selection
- `cleanupPeriodDays` — session cleanup period
- `statusLine` — custom status line script
- `fileSuggestion` — custom `@` autocomplete script
- `spinnerVerbs` — custom spinner action verbs
- `respectGitignore` — exclude gitignored files from tool results

---

## Phase 10: CLI Flags & Slash Commands Full Parity

### 10.1 Missing CLI flags
- `--agent NAME` — specify a subagent for the session
- `--agents JSON` — define custom subagents dynamically
- `--chrome` / `--no-chrome` — Chrome browser integration toggle
- `--debug [categories]` — debug mode with category filtering
- `--disable-slash-commands` — disable all skills
- `--fallback-model MODEL` — fallback when primary overloaded
- `--init-only` — run initialization hooks and exit
- `--input-format [text|stream-json]` — input format
- `--json-schema SCHEMA` — validated JSON output matching schema (print mode)
- `--mcp-config PATH` — load MCP servers from JSON file
- `--no-session-persistence` — disable session persistence
- `--output-format [text|json|stream-json]` — output format (extend existing `--json`)
- `--permission-prompt-tool TOOL` — MCP tool for non-interactive permission handling
- `--plugin-dir PATH` — load plugins from directory (repeatable)
- `--session-id UUID` — use specific session UUID
- `--settings PATH` — path to settings JSON or JSON string
- `--strict-mcp-config` — only use MCP servers from `--mcp-config`
- `--system-prompt-file PATH` — replace system prompt from file
- `--append-system-prompt-file PATH` — append file contents to prompt
- `--tools LIST` — restrict built-in tools (`""` = none, `"default"` = all, or comma-separated)
- `--worktree` / `-w` — start in isolated git worktree

### 10.2 Missing slash commands
- `/copy` — copy last assistant response to clipboard
- `/debug [description]` — troubleshoot session via debug log analysis
- `/exit` — exit REPL (may already work via Ctrl+D)
- `/hooks` — interactive hooks manager
- `/rename NAME` — rename current session
- `/resume [session]` — resume by ID/name or interactive picker
- `/stats` — visualize daily usage, session history
- `/statusline` — configure status line UI
- `/theme` — change color theme
- `/usage` — show plan/subscription usage limits and rate limit status
- `/add-dir PATH` — add directory to context during session
- `/bug` — bundle logs + config + diagnostics for bug report
- `/pr_comments` — fetch and process GitHub PR comments
- `/release-notes` — generate release notes from commits
- `/login` / `/logout` — credential management

### 10.3 Custom slash commands
- Load from `.deepseek/commands/`, `~/.deepseek/commands/`
- YAML frontmatter + markdown body
- Variable substitution: `$ARGUMENTS`, `$WORKSPACE`, `$SESSION_ID`
- Optional shell execution step with policy enforcement
- `disable-model-invocation: true` — keep out of context until manually invoked
- `context: fork` — run in a subagent
- Register in slash command autocomplete

### 10.4 Structured output (print mode)
- `--json-schema` flag: LLM output validated against provided JSON schema
- Return structured JSON matching the schema
- Error if output doesn't validate

---

## Phase 11: TUI & Interaction Parity

> Keyboard shortcuts, vim mode, multiline input, checkpoints, and visual features from Claude Code.

### 11.1 Keyboard shortcuts
- `Ctrl+C` — cancel current generation (partially done)
- `Ctrl+D` — exit session
- `Ctrl+L` — clear terminal screen
- `Ctrl+R` — reverse search command history
- `Ctrl+V` — paste image from clipboard
- `Ctrl+B` — background running tasks
- `Ctrl+T` — toggle task list
- `Ctrl+F` — kill all background agents (press twice)
- `Esc+Esc` — rewind/summarize from selected message
- `Shift+Tab` — cycle permission modes (normal → plan → normal)
- `Alt+P` — switch model
- `Alt+T` — toggle extended thinking
- `#` — quick-add memory (see Phase 9.4)
- `!` prefix — bash mode (direct command execution without tool call)
- `@` — file path mention / autocomplete

### 11.2 Vim mode
- `/vim` slash command to enable (already exists)
- Full vi keybindings: normal/insert mode switching
- `hjkl` navigation, word motions (`w`, `e`, `b`), line motions (`0`, `$`, `^`)
- Text objects (`iw`, `aw`, `i"`, etc.)
- Operators (`d`, `c`, `y`, `p`), indent/dedent (`>>`, `<<`), `.` repeat
- Arrow key history navigation in normal mode

### 11.3 Multiline input
- `\ + Enter` — continue on next line
- `Option+Enter` (macOS) — newline
- `Shift+Enter` — newline (requires terminal setup)
- `Ctrl+J` — newline
- Paste mode: auto-detect multi-line paste

### 11.4 File edit checkpoints with undo
- Every file edit (`fs.write`, `fs.edit`, `multi_edit`) snapshots current file contents
- `Esc+Esc` to rewind to previous checkpoint
- `/rewind` command with checkpoint picker
- Checkpoints local to session, separate from git

### 11.5 Image support
- `Ctrl+V` paste images from clipboard
- Read tool handles image files (PNG, JPG) — already works via multimodal
- Images displayed inline in transcript where terminal supports

### 11.6 Prompt suggestions
- After each response, auto-suggest follow-up prompts
- Based on conversation history and git diff context
- Tab to accept, Enter to accept and submit
- Configurable via settings

### 11.7 Context visualization
- `/context` command: colored grid showing token usage breakdown
- Visual representation of what's consuming context (system prompt, messages, tool results)

### 11.8 Session management
- `/rename NAME` — rename current session
- `--fork-session` — branch off with new ID, copy full message history
- `--session-id UUID` — use specific session UUID
- `--no-session-persistence` — disable persistence

### 11.9 @ file mention autocomplete
- `@` prefix triggers file path autocomplete
- Tab completion with fuzzy matching
- Injects file contents into the prompt context
- Configurable via `fileSuggestion` setting for custom autocomplete

---

## Phase 12: MCP Full Integration

> Connect McpManager to the chat loop, implement serve mode, and add OAuth.

### 12.1 Dynamic MCP tool integration in chat()
- At chat loop startup: call `McpManager::discover_tools()` for all enabled servers
- Generate `ToolDefinition` for each MCP tool, add to `tool_definitions()` with `mcp__<server>__<tool>` naming
- Route tool calls with `mcp__` prefix to appropriate MCP server
- Handle tool schema translation (MCP → OpenAI function calling format)
- Error propagation across process boundaries

### 12.2 MCP tool search (MCPSearch)
- When MCP tools exceed 10% of context window: enable lazy loading
- `mcp_search` tool: LLM describes what tool it needs, searches MCP tool definitions on-demand
- Reduces token overhead ~85% for servers with many tools

### 12.3 MCP serve mode
- `deepseek mcp serve`: expose DeepSeek CLI as an MCP server
- Full JSON-RPC 2.0 server via stdio transport
- Expose all DeepSeek tools as MCP tools for other apps
- Use existing `deepseek-jsonrpc` crate

### 12.4 MCP OAuth 2.0 authentication
- Support OAuth flows for HTTP MCP servers
- `--client-id`, `--client-secret` on `mcp add`
- In-session OAuth via `/mcp` command
- Token refresh handling

### 12.5 MCP resources and prompts
- Resources: `@server:protocol://resource/path` mentions in prompts
- Prompts: `/mcp__<server>__<prompt>` slash commands from MCP servers
- Dynamic updates: `list_changed` notifications for live tool/prompt/resource refresh

### 12.6 MCP configuration improvements
- `.mcp.json` in project root (shareable via git)
- Environment variable expansion: `${VAR}` and `${VAR:-default}` in config
- Import from other tools: `mcp add-from-claude-desktop`
- `--strict-mcp-config` — only use servers from `--mcp-config`
- Per-MCP output token limits (warn at 10K, max 25K configurable)

### 12.7 Plugin tools as LLM-callable
- Generate `ToolDefinition` from plugin metadata
- Add plugin tools to `tool_definitions()` dynamically
- Route plugin tool calls to plugin execution system

---

## Phase 13: Permission & Sandbox Parity

### 13.1 Full permission mode set ✅ (done in Phase 9.5)
- `acceptEdits`, `dontAsk` modes implemented
- Remaining: `bypassPermissions` — skip all prompts (requires `--dangerously-skip-permissions` + `--allow-dangerously-skip-permissions`)

### 13.2 Granular permission rules with glob patterns ✅ (done in Phase 9.6)
- `Bash(npm run *)`, `Read(src/**/*.rs)`, `Edit(...)`, `WebFetch(domain:...)`, `Task(...)` all implemented
- Remaining: `mcp__<server>` / `mcp__<server>__<tool>` — MCP tool permissions (Phase 12)

### 13.3 Sandbox improvements
- OS-level filesystem and network isolation for Bash commands
- `sandbox.enabled`, `sandbox.autoAllowBashIfSandboxed`
- Network: `allowedDomains`, `allowLocalBinding`, `allowUnixSockets`
- `sandbox.excludedCommands` — commands that bypass sandbox

### 13.4 Managed settings (enterprise/team)
- System-wide paths: macOS `/Library/Application Support/DeepSeekCLI/managed-settings.json`
- `disableBypassPermissionsMode` — prevent bypass
- `allowManagedPermissionRulesOnly` — only managed rules apply
- `allowManagedHooksOnly` — only managed hooks load
- `allowedMcpServers` / `deniedMcpServers` — MCP server allowlists

---

## Phase 14: IDE Integration

### 14.1 Expand JSON-RPC server methods
- Session open/resume/fork, prompt execution, tool event streaming
- Patch preview/apply, diagnostics forwarding, task updates
- Fix session fork to copy message history

### 14.2 VS Code extension
- `extensions/vscode/` — full chat panel with streaming
- Checkpoint-based undo: track file edits, rewind to previous state
- @-mention files with line ranges from selection
- Parallel conversations in separate tabs
- Diff viewer for proposed changes
- Auto-accept or review-before-accept modes

### 14.3 JetBrains plugin
- Upgrade from status-only scaffold to full feature flow
- Chat interface in IDE terminal
- Opens proposed changes in IDE diff viewer

---

## Phase 15: Performance, Polish & Dead Code Removal

### 15.1 Reduce first-response latency
- Pre-build system prompt at startup
- Cache `tool_definitions()` (static data)
- Pre-warm HTTP connection pool in `DeepSeekClient`

### 15.2 Optimize context usage
- Summarize large tool results before adding to messages
- Drop reasoning_content from message history (display only)
- Compress repeated tool call patterns

### 15.3 Graceful degradation
- Invalid API key: clear setup instructions (partially done via `format_api_error()`)
- Network down: retry with backoff (partially done via `format_transport_error()`)
- Context exceeded: auto-compact and retry (done in Phase 1.2)
- Tool failure: include error in conversation (done in Phase 5.2)

### 15.4 Clean up dead code
- **Remove** `SchemaPlanner`, `SimpleExecutor`, `run_once_with_mode()` — deprecated, zero call sites
- **Remove** `Planner`/`Executor` traits and `revise_plan` — replaced by plan mode (Phase 6.1)
- **Remove** `RightPane` enum and `right_pane_collapsed` — unused after UI simplification
- Audit unused event types in the journal

### 15.5 Shell completions
- Verify `completions` command generates proper completions for bash/zsh/fish
- Include all subcommands and flags
- Test with common shells

### 15.6 PR review status in footer
- Clickable PR link in status bar with colored underline
- Green=approved, yellow=pending, red=changes requested, gray=draft, purple=merged
- Auto-refresh every 60 seconds via `gh` CLI

---

## Phase 16: Testing & Quality

### 16.1 Integration tests for chat() loop
- Single-turn text response (no tools)
- Multi-turn with tool calls (mock LLM with `deepseek-testkit`)
- Budget/turn limit enforcement
- Tool approval flow (approve and deny)
- Streaming callback delivery across multiple turns
- Context compaction triggers
- `ChatTurnV1` persistence and resume reconstruction
- Plan mode: enter/exit, tool restriction enforcement
- Subagent spawning and result collection

### 16.2 Parameter mapping tests
- For every tool in `tool_definitions()`: verify parameter names match `run_tool()` implementation
- For every tool in `map_tool_name()`: verify mapping is correct
- Fuzz test: random LLM-style arguments → verify parsing doesn't panic

### 16.3 End-to-end TUI tests
- Prompt submission → streaming → tool execution → final output
- Ctrl+C cancellation during streaming
- Slash command execution
- Approval dialog flow
- Plan mode toggle via Shift+Tab
- Vim mode key handling

### 16.4 Replay regression tests
- Record real API sessions as cassettes
- Replay against fake LLM
- Verify deterministic output

### 16.5 Hook tests
- All 14 hook events fire at correct times
- Command/prompt/agent handler types execute correctly
- Input modification and decision control
- Async hooks don't block main thread

### 16.6 MCP integration tests
- Tool discovery from MCP servers
- Tool call routing to correct server
- Error handling for unavailable servers
- OAuth token refresh

---

## Phase 17: Architectural Debt & Hardening

### 17.1 Split main.rs into modules
- `crates/deepseek-cli/src/main.rs` — ~9,500 lines, ~150 functions
- Extract:
  - `src/commands/mod.rs` — Clap structs, Commands enum, dispatch
  - `src/commands/chat.rs` — `run_chat()`, `run_chat_tui()`, TUI setup
  - `src/commands/autopilot.rs` — `run_autopilot()`, status, pause/resume
  - `src/commands/profile.rs` — `run_profile()`, `run_benchmark()`
  - `src/commands/patch.rs` — `run_diff()`, `run_apply()`
  - `src/commands/admin.rs` — config, permissions, plugins, clean, doctor
  - `src/output.rs` — `OutputFormatter` replacing `if cli.json` blocks
- Create `CliContext { cwd, json_mode, output }` to replace `json_mode: bool` threaded through signatures

### 17.2 Add type-safe tool names
- Create `ToolName` enum (~35 variants including new tools) in deepseek-core
- Replace `ToolCall.name: String` → `ToolCall.name: ToolName`
- Replace `map_tool_name()` with `ToolName::from_api_name()` / `ToolName::as_internal()`
- Replace string constants → `ToolName::is_read_only()`, `ToolName::is_agent_level()`

### 17.3 Restructure EventKind into sub-enums
- Current: 69+ variant flat enum
- Refactor into: `SessionEventV1`, `ToolEventV1`, `TaskEventV1`, `JobEventV1`, `PluginEventV1`, `HookEventV1`
- Replace string fields with enums: `role: ChatRole`, `stop_reason: StopReason`, `status: TaskStatus`

### 17.4 Add missing test coverage
- **deepseek-store** (2870 LOC, 2 tests): event journal, SQLite projections, concurrent access, ChatTurnV1
- **deepseek-router** (130 LOC, 1 test): boundary conditions, weight combinations
- **deepseek-hooks**: all handler types, failure modes, timeouts
- **deepseek-skills**: glob patterns, directory traversal, hot reload
- **deepseek-memory**: read/write roundtrip, auto-memory, cross-project
- **deepseek-diff**: SHA verification, conflict detection, 3-way merge

### 17.5 Fix error handling hygiene
- Replace `lock().unwrap()` → `.lock().expect("context")` or proper error propagation
- Replace `.ok()` on config → logging with "file not found" vs "parse error"
- Replace silent regex failures → startup validation
- Add `#[must_use]` where `Result` silently dropped

### 17.6 Consolidate config type safety
- `PolicyConfig.permission_mode: String` → `PermissionMode` enum
- `PolicyConfig.approve_edits: String` → `ApprovalMode` enum
- `PolicyConfig.sandbox_mode: String` → `SandboxMode` enum
- Validate at load time, not use time

### 17.7 Extract test helpers to deepseek-testkit
- `testkit::temp_workspace()` — shared temp workspace creation
- `testkit::temp_host()` — shared tool host for tests
- `testkit::fake_session()` — mock session for agent tests

---

## Git Workflow Protocol (Reference — Not a Phase)

> Matches Claude Code's git safety rules. Already partially implemented in system prompt.

- **NEVER** commit/push unless explicitly asked
- **NEVER** force push, reset --hard, checkout ., clean -f, branch -D unless explicitly asked
- **NEVER** skip hooks (--no-verify) unless asked
- **NEVER** amend commits unless asked — create NEW commits after hook failures
- **NEVER** use interactive flags (-i)
- Stage specific files (not `git add -A`)
- Warn on force push to main/master
- Don't commit secrets (.env, credentials.json)
- Co-Authored-By line on commits (configurable via `attribution` setting)
- PR creation via `gh pr create` with Summary + Test Plan sections

---

## Summary: Completion Status

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ Complete | Critical bugs fixed |
| 1 | ✅ Complete | Chat loop production-ready |
| 2 | ✅ Complete | Tool system complete |
| 3 | ✅ Complete | TUI / UX |
| 4 | ✅ Complete | CLI flags (basic set) |
| 5 | ✅ Complete | Conversation & memory |
| 6 | ✅ Complete | Plan mode & subagent system |
| 7 | ✅ Complete | Missing LLM tools (Skill, KillShell, background bash) |
| 8 | ✅ Complete | Full hooks system (14 events, command handler, wired into agent) |
| 9 | ✅ Complete | Memory & configuration parity (rules, hierarchy, @imports, modes, globs, settings) |
| 10 | Pending | CLI flags & slash commands full parity |
| 11 | Pending | TUI & interaction parity (vim, shortcuts, checkpoints) |
| 12 | Pending | MCP full integration |
| 13 | Pending | Permission & sandbox parity |
| 14 | Pending | IDE integration (VS Code, JetBrains) |
| 15 | Pending | Performance, polish & dead code removal |
| 16 | Pending | Testing & quality |
| 17 | Pending | Architectural debt & hardening |

### Feature Parity Gap Summary

| Category | We Have | Claude Code Has | Gap |
|----------|---------|----------------|-----|
| LLM Tools | ~40 | ~38 | Full parity ✅ |
| Slash Commands | 23 | ~30 | /copy, /debug, /hooks, /rename, /stats, /theme, /usage, /add-dir |
| Hook Events | 14 (all events) | 14 | Full parity ✅ (prompt/agent handlers deferred) |
| Permission Modes | 5 (ask/auto/plan/acceptEdits/dontAsk) | 5 | Full parity ✅ |
| Subagent Types | 5 (explore/plan/task/bash/custom) | 6+ | Full parity ✅ |
| Memory Features | Full hierarchy + rules + @import | Full hierarchy | Full parity ✅ (#shortcut deferred) |
| MCP | Discovery only | Full integration | Chat loop wiring, OAuth, search, serve |
| Keyboard Shortcuts | ~5 | ~16 | Shift+Tab, Alt+P, Alt+T, @, #, !, Esc+Esc |
| IDE | JSON-RPC stub | Full VS Code + JetBrains | Extensions needed |
