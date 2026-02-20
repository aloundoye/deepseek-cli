# PLAN.md — DeepSeek CLI: Exhaustive Task List for Claude Code Parity

Based on deep analysis of all 18 crates (~25k LOC) and comparison with Claude Code architecture.

---

## Phase 0: Critical Bugs (Blocks Basic Functionality) ✅

### 0.1 Fix tool definition parameter name mismatches ✅
- Fixed all parameter names in `tool_definitions()` to match what `run_tool()` actually reads from `call.args`

### 0.2 Fix stream callback consumed and never re-installed ✅
- Changed `StreamCallback` to `Arc<dyn Fn(StreamChunk) + Send + Sync>`, clone instead of take

### 0.3 Fix chat() not using session state transitions ✅
- Added `transition()` calls: `ExecutingStep` at start, `Completed` or `Failed` at end

### 0.4 Fix chat() not loading prior turns on --continue/--resume ✅
- On resume, loads `TurnAddedV1` events from store, reconstructs `Vec<ChatMessage>` with proper role mapping

### 0.5 Fix chat() not persisting memory observations ✅
- Added memory persistence at end of chat() loop

---

## Phase 1: Make the Chat Loop Production-Ready ✅

### 1.1 Enrich system prompt ✅
- Enriched `build_chat_system_prompt()` with tool usage guidelines, safety rules, git protocol, workspace info, DEEPSEEK.md content, current date/OS/shell/cwd, error recovery, and style guidelines

### 1.2 Add context window management ✅
- Added approximate token tracking and auto-compaction at 80% of context window
- Emits `ContextCompactedV1` event, truncates large tool outputs

### 1.3 Wire all entry points to chat() ✅
- Replaced all `run_once*` calls with `engine.chat()` / `engine.chat_with_options()`

### 1.4 Integrate model router in chat() — Hybrid Model Strategy ✅
- `router.select()` called per turn; below 0.72 → `deepseek-chat`, above → `deepseek-reasoner`
- Auto-escalation on failure

### 1.5 Add tool execution feedback to stream ✅
- Tool calls emit `StreamChunk::ContentDelta` with tool name, key args, completion duration, and truncated result preview

---

## Phase 2: Tool System Completeness ✅ COMPLETED

### 2.1 Add missing tools to tool_definitions() ✅
- Added 6 chrome tools (navigate, click, type_text, screenshot, read_console, evaluate) to `tool_definitions()` and `map_tool_name()`
- Added 4 agent-level tools (user_question, task_create, task_update, spawn_task)
- All tools now have definitions, mappings, and `AGENT_LEVEL_TOOLS` constant

### 2.2 Fix fs_edit tool definition to match all edit modes ✅
- Already uses Claude Code style (search/replace/all) — no changes needed

### 2.3 Implement AskUserQuestion tool ✅
- Added `UserQuestion` type and `UserQuestionHandler` callback in deepseek-core
- Agent intercepts `user_question` tool calls before LocalToolHost
- External handler (TUI mode) or fallback stdin prompt
- Returns JSON `{"answer": "..."}` or `{"cancelled": true}`

### 2.4 Implement task tracking tools (TodoWrite equivalent) ✅
- Added `task_create` tool: creates tasks in store with subject, description, priority
- Added `task_update` tool: updates task status (pending/in_progress/completed/failed)
- Emits `TaskCreatedV1` events, uses store's `insert_task()` and `update_task_status()`

### 2.5 Implement subagent/Task tool ✅
- Added `spawn_task` tool: spawns subagents via `SubagentManager`
- Supports explore/plan/task roles
- External worker function or fallback echo worker
- `set_subagent_worker()` setter for full agent capabilities

### 2.6 Improve tool output truncation ✅
- `truncate_tool_output()` now accepts tool name for tool-specific strategies
- `bash.run`: parses JSON output, keeps full stderr, truncates stdout to last 200 lines
- `fs.read`: uses 80 head + 80 tail lines with total line count
- Generic: keeps 100 head + 100 tail lines with omission count
- 4 new tests verify all truncation strategies

---

## Phase 3: TUI / User Experience ✅

### 3.1 Improve markdown rendering ✅
- Added `*italic*` support to `parse_inline_markdown()` with recursive nesting
- Added table rendering: pipe-delimited rows with `│` separators, separator rows dimmed
- Italic, bold, inline code, and combinations all work

### 3.2 Add tool call display in transcript ✅
- Added `ChatShell::push_tool_call(name, args_summary)` — creates `MessageKind::ToolCall` entry
- Added `ChatShell::push_tool_result(name, duration_ms, summary)` — creates `MessageKind::ToolResult` with auto-formatted duration (ms/s)
- Added `TuiStreamEvent::ToolCallStart` and `TuiStreamEvent::ToolCallEnd` events with full handling in event loop

### 3.3 Add cost/token tracking in status bar ✅
- Added session turn count display (`turn N`) to `render_statusline_spans()` — hidden when 0
- Context usage (K/K with color coding) and cost ($) were already displayed
- Added `ChatShell::push_cost_summary()` for `/cost` command output

### 3.4 Fix spinner during tool execution ✅
- Added `ToolCallStart`/`ToolCallEnd` stream events that properly set/clear `shell.active_tool`
- `ToolCallStart` both pushes transcript entry and activates spinner
- `ToolCallEnd` pushes result entry and clears spinner

### 3.5 Right panel toggle — SKIPPED
- Removed: right panel split disrupts the clean Claude-style full-width chat UI
- The existing `RightPane` enum and `Ctrl+O` cycle remain for future use if needed

### 3.6 Improve slash command output ✅
- Added `ChatShell::push_cost_summary(status)` — formats cost, tokens, context %, turns
- Added `ChatShell::push_status_summary(status)` — formats model, mode, approvals, tasks, jobs, autopilot, cost
- Added `ChatShell::push_model_info(model)` — displays current model

---

## Phase 4: CLI Flags for Claude Code Parity ✅

### 4.1 --permission-mode <ask|auto|plan> ✅
- Added `--permission-mode` flag, passed to `PolicyEngine` as override via `apply_cli_flags()`

### 4.2 --dangerously-skip-permissions ✅
- Sets permission mode to auto with all tools auto-approved

### 4.3 --allowed-tools / --disallowed-tools ✅
- Comma-separated tool names, filters `tool_definitions()` via `filter_tool_definitions()`
- Mutual exclusivity validated in `validate_cli_flags()`

### 4.4 --system-prompt / --append-system-prompt ✅
- Custom system prompt override or append via `ChatOptions`
- Mutual exclusivity validated

### 4.5 --add-dir <DIR> ✅
- Additional directories passed through `ChatOptions.additional_dirs`

### 4.6 --verbose ✅
- Enables detailed logging to stderr via `observer.set_verbose()`

### 4.7 --init as global flag ✅
- Added `--init` flag to initialize DEEPSEEK.md

### 4.8 Improve --print mode ✅
- Added `--no-input` for fully non-interactive mode

---

## Phase 5: Conversation & Memory ✅

### 5.1 Multi-turn conversation persistence ✅
- Added `ChatTurnV1` event kind: stores full structured `ChatMessage` (including `tool_calls` with IDs and `Tool` results with `tool_call_id`)
- On resume, prefers structured `chat_messages` from `ChatTurnV1`; falls back to legacy string transcript for older sessions
- `RebuildProjection.chat_messages: Vec<ChatMessage>` added to store

### 5.2 Improve error messages ✅
- Added `format_api_error()` in deepseek-llm: user-friendly messages for 401, 402, 429, 5xx with actionable suggestions
- Added `format_transport_error()`: timeout and connection errors with config hints
- Added `tool_error_hint()` in deepseek-tools: context-specific hints for fs.edit, fs.read, fs.write, bash.run failures
- Agent appends hints to failed tool results before sending back to LLM
- Improved budget/turn limit messages with `--max-budget-usd` / `--max-turns` suggestions

### 5.3 Implement proper token counting ✅
- Replaced `chars/4` approximation with word-based BPE heuristic in `estimate_tokens()`
- Short words (1-3 chars) → 1 token, medium (4-7) → 2, long (8-15) → 3, very long → chars/4
- Added per-message overhead (4 tokens framing) in `estimate_messages_tokens()`

### 5.4 Cost tracking per conversation ✅
- Cost ledger, per-session cost queries, and status bar display were already complete
- Added 80% budget warning via stream callback (once per session)
- Warning shows used/max cost, percentage, and remaining budget

---

## Phase 6: MCP & Extensibility

### 6.1 Implement MCP serve mode (only stub in codebase)
- File: `crates/deepseek-mcp/src/lib.rs` — `run_mcp_serve()`
- Currently: `eprintln!("MCP serve mode: listening on stdin/stdout (stub)")`
- **Implement**: Full JSON-RPC 2.0 server using `deepseek-jsonrpc` crate
- Expose all DeepSeek tools as MCP tools

### 6.2 Dynamic MCP tool integration in chat()
- When MCP servers are configured, discover their tools at startup
- Merge MCP tools into `tool_definitions()` dynamically
- Route tool calls to appropriate MCP server based on tool name prefix

### 6.3 Plugin tools in tool_definitions()
- Plugin system exists and works
- Plugins aren't exposed as LLM-callable tools
- **Fix**: Load plugin tools dynamically, add to tool_definitions()

---

## Phase 7: Custom Slash Commands

### 7.1 Custom command discovery
- Load from `.deepseek/commands/`, `~/.deepseek/commands/`
- Parse YAML frontmatter + markdown body
- Register in slash command autocomplete

### 7.2 Command execution
- Variable substitution: `$ARGUMENTS`, `$WORKSPACE`, `$SESSION_ID`
- Optional shell execution step
- Policy enforcement for any shell commands

### 7.3 Missing built-in slash commands
- `/bug` — bundle logs + config + diagnostics for bug report
- `/pr_comments` — fetch and process GitHub PR comments
- `/release-notes` — generate release notes from commits
- `/add-dir` — add directory to context
- `/login` / `/logout` — credential management

---

## Phase 8: IDE Integration

### 8.1 Expand JSON-RPC server methods
- File: `crates/deepseek-jsonrpc/src/lib.rs`
- Add: session open/resume/fork, prompt execution, tool event streaming
- Add: patch preview/apply, diagnostics forwarding, task updates

### 8.2 VS Code extension
- File: `extensions/vscode/` (if exists, or create)
- Full chat panel with streaming
- Inline actions: send selection/file/diagnostic to agent
- Diff preview for edits

### 8.3 JetBrains plugin
- Upgrade from status-only scaffold to full feature flow

---

## Phase 9: Testing & Quality

### 9.1 Integration tests for chat() loop
- Test: single-turn text response (no tools)
- Test: multi-turn with tool calls (mock LLM with `deepseek-testkit`)
- Test: budget/turn limit enforcement
- Test: tool approval flow (approve and deny)
- Test: streaming callback delivery across multiple turns
- Test: context compaction triggers

### 9.2 Parameter mapping tests
- For every tool in `tool_definitions()`: verify parameter names match `run_tool()` implementation
- For every tool in `map_tool_name()`: verify mapping is correct
- Fuzz test: random LLM-style arguments → verify parsing doesn't panic

### 9.3 End-to-end TUI tests
- Prompt submission → streaming → tool execution → final output
- Ctrl+C cancellation during streaming
- Slash command execution
- Approval dialog flow

### 9.4 Replay regression tests
- Record real API sessions as cassettes
- Replay against fake LLM
- Verify deterministic output

---

## Phase 10: Performance & Polish

### 10.1 Reduce first-response latency
- Pre-build system prompt at startup
- Cache `tool_definitions()` (static data)
- Pre-warm HTTP connection pool in `DeepSeekClient`

### 10.2 Optimize context usage
- Summarize large tool results before adding to messages
- Drop reasoning_content from message history (it's for display only)
- Compress repeated tool call patterns

### 10.3 Graceful degradation
- Invalid API key: clear setup instructions
- Network down: retry with backoff, then show error
- Context exceeded: auto-compact and retry
- Tool failure: include error in conversation, let LLM adapt

### 10.4 Clean up dead code
- `SimpleExecutor` — never instantiated, remove or document as example
- `Planner` trait `revise_plan` — only used in legacy path
- `RightPane` enum and `right_pane_collapsed` — now unused after UI simplification
- Unused event types in the journal (audit and remove or keep for future)

### 10.5 Shell completions
- Verify `completions` command generates proper completions for bash/zsh/fish
- Include subcommand and flag completions
- Test with common shells

---

## Phase 11: Architectural Debt & Hardening

### 11.1 Split main.rs into modules
- File: `crates/deepseek-cli/src/main.rs` — **9,563 lines, 149 functions**
- Extract into submodules:
  - `src/commands/mod.rs` — Clap structs, Commands enum, dispatch
  - `src/commands/chat.rs` — `run_chat()`, `run_chat_tui()`, TUI setup
  - `src/commands/autopilot.rs` — `run_autopilot()`, status, pause/resume
  - `src/commands/profile.rs` — `run_profile()`, `run_benchmark()`
  - `src/commands/patch.rs` — `run_diff()`, `run_apply()`
  - `src/commands/admin.rs` — config, permissions, plugins, clean, doctor
  - `src/output.rs` — `OutputFormatter` replacing 106 `if cli.json` blocks
- Create `CliContext { cwd, json_mode, output }` to replace `json_mode: bool` threaded through 40+ signatures

### 11.2 Add type-safe tool names
- Create `ToolName` enum (~30 variants) in deepseek-core
- Replace `ToolCall.name: String` → `ToolCall.name: ToolName`
- Replace `map_tool_name()` with `ToolName::from_api_name()` / `ToolName::as_internal()`
- Replace `REVIEW_BLOCKED_TOOLS: &[&str]` → `ToolName::is_read_only()`
- Replace `AGENT_LEVEL_TOOLS: &[&str]` → `ToolName::is_agent_level()`
- Update deepseek-tools, deepseek-agent, deepseek-policy

### 11.3 Restructure EventKind into sub-enums
- Current: 69-variant flat enum (260+ lines)
- Refactor into:
  - `SessionEventV1` — TurnAdded, StateChanged, Started, Resumed
  - `ToolEventV1` — Proposed, Approved, Denied, Result
  - `PlanEventV1` — Created, Revised, StepMarked
  - `TaskEventV1` — Created, Completed, Updated, Deleted
  - `JobEventV1` — Started, Resumed, Stopped
  - `PluginEventV1` — Installed, Removed, Enabled, Disabled, etc.
- Replace string fields with enums: `role: String` → `role: ChatRole`, `stop_reason: String` → `StopReason`, `status: String` → `TaskStatus`

### 11.4 Add missing test coverage
- **deepseek-store** (2870 LOC, 2 tests): event journal append/rebuild, SQLite projection consistency, concurrent access, event type migrations
- **deepseek-router** (130 LOC, 1 test): boundary conditions (threshold=0.72 exactly), low scores → base model, weight combinations, all-zero signals
- **deepseek-hooks** (129 LOC, 1 test): Windows PowerShell execution, hook failure modes, timeout handling
- **deepseek-skills** (235 LOC, 1 test): glob pattern edge cases, directory traversal, hot reload
- **deepseek-memory** (530 LOC, 3 tests): DEEPSEEK.md read/write roundtrip, auto-memory observation persistence, cross-project memory
- **deepseek-diff** (296 LOC, 2 tests): SHA verification, conflict detection, 3-way merge

### 11.5 Fix error handling hygiene
- Replace 26+ `lock().unwrap()` calls with `.lock().expect("context")` or proper error propagation
- Replace `.ok()` on config loading with logging: distinguish "file not found" from "parse error"
- Replace silent regex compilation failures in deepseek-policy with startup validation
- Add `#[must_use]` where `Result` return values are silently dropped

### 11.6 Consolidate config type safety
- Move `PolicyConfig.permission_mode: String` → `PermissionMode` enum (already exists in deepseek-policy)
- Move `PolicyConfig.approve_edits: String` → `ApprovalMode { Ask, Allow, Deny }` enum
- Move `PolicyConfig.sandbox_mode: String` → `SandboxMode` enum (already exists)
- Validate config at load time instead of at use time

### 11.7 Extract test helpers to deepseek-testkit
- Move temp workspace creation pattern (repeated in 5+ crates) to `testkit::temp_workspace()`
- Move `temp_host()` pattern from deepseek-tools tests to shared helper
- Add `testkit::fake_session()` for agent tests