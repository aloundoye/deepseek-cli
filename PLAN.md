# PLAN.md — DeepSeek CLI: Exhaustive Task List for Claude Code Parity

Based on deep analysis of all 18 crates (~25k LOC) and comparison with Claude Code architecture.

---

## Phase 0: Critical Bugs (Blocks Basic Functionality)

### 0.1 Fix tool definition parameter name mismatches
**Severity: CRITICAL — every tool call fails at runtime**
- File: `crates/deepseek-tools/src/lib.rs` — `tool_definitions()` function
- `tool_definitions()` uses `file_path` but `run_tool()` reads `path` from args
- `bash_run` definition says `command` but implementation reads `cmd`
- `fs_read` definition says `offset, limit` but code reads `start_line, max_bytes`
- `fs_edit` definition says `old_string, new_string` but code reads `search, replace` or `edits[]` or `start_line/end_line/replacement`
- `notebook_edit` says `cell_number, new_source` but code reads `cell_index, operation`
- **Fix**: Update all parameter names in `tool_definitions()` to match what `run_tool()` actually reads from `call.args`

### 0.2 Fix stream callback consumed and never re-installed
**Severity: CRITICAL — streaming dies after first tool call turn**
- File: `crates/deepseek-agent/src/lib.rs` — `chat()` method
- `self.stream_callback.lock().ok().and_then(|mut g| g.take())` consumes the callback
- After first LLM call, all subsequent turns fall back to non-streaming `complete_chat()`
- User sees first response stream, then nothing for all tool-call turns
- **Fix**: Change `StreamCallback` from `Box<dyn FnMut(StreamChunk) + Send>` to `Arc<dyn Fn(StreamChunk) + Send + Sync>` in `deepseek-core/src/lib.rs`, then clone instead of take

### 0.3 Fix chat() not using session state transitions
**Severity: HIGH — session state stays Idle, events incomplete**
- File: `crates/deepseek-agent/src/lib.rs` — `chat()` method
- Never calls `self.transition()` — session stays in `Idle` forever
- No `SessionStateChangedV1` events emitted during chat mode
- **Fix**: Transition to `ExecutingStep` at start, `Completed` or `Failed` at end

### 0.4 Fix chat() not loading prior turns on --continue/--resume
**Severity: HIGH — conversation continuity broken**
- File: `crates/deepseek-agent/src/lib.rs` — `chat()` method
- Always starts with empty `Vec<ChatMessage>` — ignores all prior conversation
- `--continue` and `--resume` flags exist in CLI but chat() discards history
- **Fix**: On resume, load `TurnAddedV1` events from store via `rebuild_from_events()`, reconstruct `Vec<ChatMessage>` with proper role mapping

### 0.5 Fix chat() not persisting memory observations
**Severity: MEDIUM — no learning across sessions**
- File: `crates/deepseek-agent/src/lib.rs` — `chat()` method
- Never calls `remember_successful_strategy()`, `remember_objective_outcome()`, or `append_auto_memory_observation()`
- The legacy `run_once_with_mode_and_priority()` does this at lines 906-954
- **Fix**: Add memory persistence at end of chat() loop

---

## Phase 1: Make the Chat Loop Production-Ready

### 1.1 Enrich system prompt
- File: `crates/deepseek-agent/src/lib.rs` — `build_chat_system_prompt()`
- Current prompt is ~150 tokens, too generic
- **Add**:
  - Tool usage guidelines with examples for each tool
  - Safety rules: read before edit, don't delete without asking, never force push
  - Git protocol: create new commits, never amend without asking, never push without asking
  - Workspace info: project structure from `fs.list`, language detection, recent git status
  - DEEPSEEK.md content (already partially done)
  - Current date, OS, shell, working directory
  - Error recovery: if tool fails, read the error and try a different approach
  - Style: be concise, use markdown, show file paths with line numbers

### 1.2 Add context window management
- File: `crates/deepseek-agent/src/lib.rs` — `chat()` method
- No token tracking in the chat loop — will exceed context window on long conversations
- **Implement**:
  - Track approximate token count of `messages` array (sum of content lengths / 4)
  - When approaching 80% of `context_window_tokens`: summarize old messages, keep system + recent
  - Emit `ContextCompactedV1` event
  - Keep tool results compact (truncate large outputs)

### 1.3 Wire all entry points to chat()
- File: `crates/deepseek-cli/src/main.rs`
- Multiple entry points still use legacy `run_once_with_mode()`:
  - `run_chat()` non-TUI REPL path (~line 2440)
  - `run_print_mode()` (~line 8928) — uses `engine.run_once()`
  - `ask` command handler (~line 936)
  - Skill execution (~line 7090, 7504)
  - Review command (~line 9155)
- **Fix**: Replace all `run_once*` calls with `engine.chat()` for the new architecture

### 1.4 Integrate model router in chat() — Hybrid Model Strategy
- File: `crates/deepseek-agent/src/lib.rs` — `chat()` method
- Currently hardcodes `self.cfg.llm.base_model` — never uses `WeightedRouter`
- **DeepSeek Model Strategy**: Use `deepseek-reasoner` as the primary "brain" for the agentic loop (complex multi-step reasoning, debugging, planning) and `deepseek-chat` for fast, low-complexity responses
- Both models support function calling and 128K context; reasoner generates CoT (up to 64K output), chat is direct (up to 8K output)
- Reasoner is superior for: self-correction during thinking, multi-part instruction following, long tool-calling chains
- Chat is superior for: speed, simple completions, boilerplate, documentation tasks
- **Fix**: Call `self.router.select()` per turn based on prompt complexity score (threshold 0.72)
  - Below threshold → `deepseek-chat` (fast, direct)
  - Above threshold → `deepseek-reasoner` (CoT, self-correcting)
  - Auto-escalation: if `deepseek-chat` fails or produces poor results, retry with `deepseek-reasoner`

### 1.5 Add tool execution feedback to stream
- File: `crates/deepseek-agent/src/lib.rs` — `chat()` tool execution section
- When tools run, user sees nothing until next LLM response
- **Implement**: Send `StreamChunk::ContentDelta` for each tool call with:
  - `[tool: fs_read] path=src/main.rs` — tool name + key args
  - `[tool: fs_read] done (1.2s)` — completion with duration
  - Truncated result preview for non-streaming tools

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

## Phase 3: TUI / User Experience

### 3.1 Improve markdown rendering
- File: `crates/deepseek-ui/src/lib.rs` — `style_transcript_line()`
- Current: basic heading bold, code block coloring
- **Add**: proper bullet/numbered lists, inline code backticks, bold/italic, tables
- Consider pulling in a ratatui markdown widget or building a simple parser

### 3.2 Add tool call display in transcript
- When LLM calls tools, push transcript entries showing:
  - Tool name and key arguments (dimmed/gray)
  - Result status (green check or red X)
  - Duration
- Update `TranscriptEntry` kinds or add new `MessageKind::ToolCall` / `MessageKind::ToolResult`

### 3.3 Add cost/token tracking in status bar
- File: `crates/deepseek-ui/src/lib.rs` — `render_statusline_spans()`
- Show: `tokens: 12.3k/128k | cost: $0.04 | turn 3`
- Wire into store's cost_ledger and token tracking

### 3.4 Fix spinner during tool execution
- File: `crates/deepseek-ui/src/lib.rs`
- `shell.active_tool` exists but may not update properly in chat() mode
- **Fix**: Send `TuiStreamEvent::ToolStarted(name)` / `TuiStreamEvent::ToolFinished` from agent to TUI via channel

### 3.5 Add optional right panel toggle
- The right panel (Plan/Tools/Mission/Artifacts) was removed for clean chat
- Add back as toggleable with Tab key (default: off)
- Show: active tools, recent tool calls, task list, cost breakdown

### 3.6 Improve slash command output
- `/cost` — show breakdown per session with input/output/cached tokens
- `/compact` — show tokens freed and new usage percentage
- `/status` — show model, permission mode, tools enabled, session info
- `/model` — show current model and allow switching

---

## Phase 4: CLI Flags for Claude Code Parity

### 4.1 --permission-mode <ask|auto|plan>
- File: `crates/deepseek-cli/src/main.rs` — `Cli` struct
- Pass to `PolicyEngine::from_app_config()` as override

### 4.2 --dangerously-skip-permissions
- Set permission mode to Auto with all tools auto-approved
- For CI/scripted usage

### 4.3 --allowed-tools / --disallowed-tools
- Accept comma-separated tool names
- Filter `tool_definitions()` before sending to LLM
- Block disallowed tools at execution time

### 4.4 --system-prompt / --append-system-prompt
- Allow custom system prompt override or append
- For specialized use cases (code review, documentation, etc.)

### 4.5 --add-dir <DIR>
- Add additional directories to workspace context
- Include in system prompt and allow tools to access them

### 4.6 --verbose
- Enable detailed logging to stderr
- Show: API requests/responses, tool calls, timing, token counts

### 4.7 --init as global flag
- Auto-initialize DEEPSEEK.md when running in a new project
- Currently only `/init` slash command

### 4.8 Improve --print mode
- Support stdin piping: `echo "fix the bug" | deepseek -p`
- Support `--no-input` for fully non-interactive
- Output only final response (not intermediate tool calls) unless --verbose

---

## Phase 5: Conversation & Memory

### 5.1 Multi-turn conversation persistence
- Load prior session turns into ChatMessage array on --continue/--resume
- Handle tool_calls and tool results from prior turns
- Implement conversation branching (fork at any point)

### 5.2 Improve error messages
- Replace all raw debug output with user-friendly messages
- API errors: "API key not set. Run `deepseek config edit` to add your DEEPSEEK_API_KEY"
- Tool failures: "fs_edit failed: old_string not found in file. Try reading the file first."
- Rate limits: "Rate limited. Retrying in 5s... (attempt 2/3)"
- Budget: "Budget limit reached ($0.50/$0.50). Use --max-budget-usd to increase."

### 5.3 Implement proper token counting
- Current `estimate_tokens()` uses chars/4 approximation
- Options: tiktoken-rs crate, or DeepSeek tokenizer API
- Critical for accurate context window management

### 5.4 Cost tracking per conversation
- Track input/output/cached tokens per turn
- Show cumulative cost in status bar
- Warn at 80% of budget limit
- Log to cost_ledger in store

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

│### **Potential Areas for Improvement**                                                                                    │
│                                                                                                                           │
│  1. **Documentation**: Architectural documentation could be more comprehensive                                            │
│  2. **Configuration**: Complex config system might be overwhelming for new users                                          │
│  3. **Testing**: More integration tests could improve reliability                                                         │
│  4. **Error Handling**: Could benefit from more structured error types                                                    │
│  5. **Performance**: Large workspaces might benefit from more aggressive caching                                          │
│                                                                                   