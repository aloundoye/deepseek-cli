# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepSeek CLI is an open-source coding agent CLI powered by DeepSeek, written in Rust. It uses edition 2024 with MSRV 1.93.0 (pinned in `rust-toolchain.toml`). The binary is called `deepseek`.

## Build & Test Commands

```bash
cargo fmt --all -- --check                                # Format check
cargo clippy --workspace --all-targets -- -D warnings     # Lint (warnings are errors)
cargo test --workspace --all-targets                      # Full test suite
cargo build --workspace                                   # Debug build
cargo build --release --bin deepseek                      # Release binary

# Single crate / single test
cargo test -p deepseek-core                               # One crate
cargo test -p deepseek-cli --test cli_json                # Specific integration test
cargo test --lib policy_allow                              # Test by name

# CLI smoke test
cargo run --bin deepseek -- --json status
```

CI runs on Ubuntu, macOS, and Windows. Cross-compilation targets: `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-apple-darwin`, `aarch64-apple-darwin`, `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc`.

## Workspace Architecture (18 crates)

```
deepseek-cli        CLI entry point, clap command parsing, TUI dispatch
deepseek-core       Core types: Session, AppConfig, EventEnvelope, state machine, traits
deepseek-agent      Agent runtime: Planner/Executor trait impls, subagent orchestration
deepseek-llm        DeepSeek API client (streaming, retries, prompt caching)
deepseek-router     Model routing with auto max-think complexity scoring
deepseek-tools      Tool registry, sandboxed execution, plugin management
deepseek-diff       Patch staging/application with SHA-256 base verification
deepseek-index      Tantivy-based code search + file manifest with notify watcher
deepseek-store      SQLite persistence, append-only event log (.deepseek/events.jsonl)
deepseek-policy     Permission engine, allowlists, redaction, sandbox modes
deepseek-observe    Logging, metrics, tracing, cost tracking
deepseek-mcp        Model Context Protocol server management
deepseek-subagent   Parallel subagent lifecycle (Explore/Plan/Task types, up to 7 concurrent)
deepseek-memory     DEEPSEEK.md management, objective-outcome memory, exports
deepseek-skills     Skill loading and execution
deepseek-hooks      Pre/post tool hooks
deepseek-ui         TUI (ratatui/crossterm), slash commands, keybindings
deepseek-testkit    Test utilities, replay harness, fake LLM
```

### Key dependency flow

`deepseek-cli` → `deepseek-agent` → `deepseek-core` (types/traits), `deepseek-llm` (API), `deepseek-tools` (execution), `deepseek-router` (model selection), `deepseek-store` (persistence), `deepseek-policy` (permissions), `deepseek-diff` (patches), `deepseek-index` (search)

## Core Abstractions

### Chat-with-Tools Architecture (Primary)

The agent uses **DeepSeek API function calling** for a conversational chat-with-tools loop:

1. User prompt + system prompt + tool definitions → DeepSeek API (`/chat/completions` with `tools` parameter)
2. Model streams response (text content and/or `tool_calls`)
3. If `tool_calls` present: execute each tool, send results back as `role: "tool"` messages, loop to step 1
4. If no `tool_calls`: return text response to user — conversation turn complete

**Key types** (`deepseek-core`):
- `ChatMessage` — tagged enum: `System`, `User`, `Assistant` (with optional `tool_calls`), `Tool` (with `tool_call_id`)
- `ChatRequest` — model, messages, tools, tool_choice, max_tokens, temperature
- `ToolDefinition` / `FunctionDefinition` — OpenAI-compatible function schemas
- `ToolChoice` — `auto`, `none`, or specific function
- `LlmToolCall` — id, name, arguments (as returned by the API)

**Agent entry point**: `AgentEngine::chat(prompt)` in `deepseek-agent` — the main chat-with-tools loop.

**Tool definitions**: `deepseek_tools::tool_definitions()` returns `Vec<ToolDefinition>` for all tools. `deepseek_tools::map_tool_name()` maps API function names (underscored: `fs_read`) to internal tool names (dotted: `fs.read`).

**LLM client**: `LlmClient::complete_chat_streaming()` / `complete_chat()` in `deepseek-llm` — sends `ChatRequest` with tools to the DeepSeek API and handles streaming tool call delta merging.

### Legacy Plan-and-Execute Architecture

The old `run_once_with_mode()` method still exists for backward compatibility. It uses `SchemaPlanner` to generate JSON plans and `SimpleExecutor` to run hardcoded tool calls. **New code should use `AgentEngine::chat()` instead.**

### Supporting Abstractions

**Event sourcing**: All state changes, tool proposals/approvals/results journaled to `.deepseek/events.jsonl` (append-only). SQLite projections in `.deepseek/store.sqlite` for queries.

**Tool flow**: `ToolCall` → `ToolProposal` (policy check) → `ApprovedToolCall` → `ToolResult` → event journal. Tools include `fs.read`, `fs.write`, `fs.edit`, `fs.grep`, `fs.glob`, `bash.run`, `git.*`, `web.*`, `notebook.*`, `multi_edit`, and plugins.

**Model router** (`deepseek-router`): Weighted complexity scoring toggles thinking mode on `deepseek-chat`. When the score crosses the threshold (default 0.72), the router sets `thinking_enabled=true`.

**Unified thinking + tools** (primary, DeepSeek V3.2): When `router.unified_thinking_tools = true` (default), `deepseek-chat` is called with both `thinking: {type: "enabled"}` and `tools: [...]` in a single API call. The model reasons via chain-of-thought and emits tool calls in one pass. Per V3.2 docs: `reasoning_content` is **kept** within the current tool loop (same user question) so the model retains its logical thread, and **stripped** from prior conversation turns (previous user questions) to save bandwidth. Temperature, top_p, and penalty params are omitted when thinking is enabled.

**Mode router** (`deepseek-agent::mode_router`): Selects between two execution modes per turn:
- **V3Autopilot** (default): `deepseek-chat` with unified thinking + tools in a single API call. Fast, handles most tasks.
- **R1DriveTools** (escalation): R1 (`deepseek-reasoner`) drives tools step-by-step via structured JSON intents (`tool_intent`, `delegate_patch`, `done`, `abort`). Orchestrator validates + executes via `ToolHost`. R1 receives `ObservationPack` context between steps. R1 budget capped at `r1_max_steps` (default 30).

**Escalation triggers** (V3Autopilot → R1DriveTools, checked by `decide_mode()` after each tool execution batch):
1. **Doom-loop** (highest priority): same tool signature (name + normalized args + exit code bucket) fails ≥ `doom_loop_threshold` (default 2) times.
2. **Repeated step failures**: consecutive failures ≥ `v3_max_step_failures` (default 5). V3 gets one mechanical recovery attempt for compile/lint/dependency errors before escalation (`v3_mechanical_recovery = true`).
3. **Ambiguous errors**: `ErrorClass::Ambiguous` — error cannot be classified.
4. **Blast radius exceeded**: ≥ `blast_radius_threshold` (default 10) files changed without verification.
5. **Cross-module failures**: errors span 2+ distinct modules.

**Policy doom-loop breaker** (`deepseek-agent/src/lib.rs`): When doom-loop detection fires and all failing signatures are `bash.run` policy errors (forbidden metacharacters, command not allowlisted), the agent injects a guidance message redirecting the model to built-in tools (`fs.glob`, `fs.grep`, `fs.read`) and resets failure trackers instead of escalating to R1 (which faces the same policy restrictions).

**Hysteresis**: Once escalated to R1, stays there until verification passes (`verify_passed_since_r1`), R1 returns `done`/`abort`, or R1 budget is exhausted (forced return to V3).

**R1 consultation** (`deepseek-agent::consultation`): Lightweight alternative to full R1DriveTools escalation. V3 asks R1 for targeted advice on a subproblem (error analysis, architecture, plan review, task decomposition). R1 returns text-only advice (no JSON intents, no tool execution). Advice is injected into V3's conversation as a tool result. V3 keeps control throughout.

**Plan → Execute → Verify discipline** (`deepseek-agent::plan_discipline`): For complex prompts (detected via keyword triggers), the agent generates a step-by-step plan before executing. `PlanState` tracks progress through `NotPlanned → Planned → Executing → Verifying → Completed`. Steps auto-advance when declared files are touched. Exit is gated by verification checkpoints (diagnostics, tests). Verification failures feed back into the loop.

Key modules in `deepseek-agent`:
- `mode_router.rs` — `AgentMode`, `ModeRouterConfig`, `FailureTracker`, `ToolSignature`, `decide_mode()`, escalation + hysteresis logic
- `protocol.rs` — R1 JSON envelope types (`R1Response`, `ToolIntent`, `DelegatePatch`), parsing + validation, V3 patch response parsing
- `observation.rs` — `ObservationPack`, `ErrorClass`, error classification, compact R1 context serialization
- `r1_drive.rs` — R1 drive-tools loop: R1 → JSON intent → tool execution → observation → loop
- `v3_patch.rs` — V3 as patch-only writer: produces unified diffs from R1's `delegate_patch` instructions
- `consultation.rs` — R1 consultation: lightweight text-only advice without full escalation
- `plan_discipline.rs` — Plan state machine, trigger detection, step tracking, verification gating

**Streaming text suppression** (`deepseek-llm`): When the model's streaming response contains both text content deltas and structured `tool_calls`, text fragments are suppressed from the TUI display once the first tool call delta arrives. After streaming completes, a `ClearStreamingText` signal removes any pre-tool-call text noise from the transcript. This prevents the choppy display of interleaved text fragments between tool calls.

**TUI** (`deepseek-ui`): Simple chat interface — full-width transcript with auto-scroll, input area, status bar. Streaming tokens displayed in real-time via `StreamCallback`. Handles `ClearStreamingText` to discard noise text when tool calls are present.

## Conventions

- `pub type Result<T> = anyhow::Result<T>` — used project-wide
- `thiserror` for domain-specific error types, `anyhow` for propagation with `.context()`
- All crates inherit `version`, `edition`, `rust-version`, `license` from `[workspace.package]`
- Workspace dependency management in root `Cargo.toml` `[workspace.dependencies]`
- DeepSeek-only provider: no fallback to other LLM providers
- API key via `DEEPSEEK_API_KEY` env var or `llm.api_key` config field
- Property-based tests with `proptest` for state machine invariants and config merging
- `--json` global flag for machine-readable output on all commands

## Configuration

Load order (later overrides earlier):
1. Legacy `.deepseek/config.toml`
2. `~/.deepseek/settings.json` (user-global)
3. `.deepseek/settings.json` (project-shared)
4. `.deepseek/settings.local.json` (machine-local, gitignored)

Config sections: `llm`, `router`, `policy`, `plugins`, `skills`, `usage`, `context`, `autopilot`, `scheduling`, `replay`, `ui`, `experiments`, `telemetry`, `index`, `budgets`, `theme`. See `config.example.toml` for all fields.

## Known DeepSeek API Limitations

### Thinking mode + tools (DeepSeek V3.2)

DeepSeek V3.2 (`deepseek-chat`) officially supports **thinking mode with function calling** in a single API call. This is the default mode (`router.unified_thinking_tools = true`).

**V3.2 rules for `reasoning_content` in tool loops:**
1. Within a tool loop (single user question): **keep `reasoning_content` in history** so the model retains its logical thread.
2. When a new user question starts: **clear `reasoning_content` from prior turns** to save bandwidth.
3. Temperature, top_p, presence_penalty, frequency_penalty are **unsupported with thinking mode** — omit them.
4. `deepseek-reasoner` (R1) still does **NOT** support function calling — use `deepseek-chat` with `thinking: {type: "enabled"}`.

**Implementation** (`deepseek-agent/src/lib.rs` — `chat_with_options()`):
- Thinking is enabled when `decision.thinking_enabled && (!options.tools || unified_thinking_tools)`.
- `reasoning_content` is stripped only from assistant messages *before* the last User message (prior turns), kept from current tool loop messages.

### DSML markup leak (safety net)

Older DeepSeek API versions could intermittently output **raw DSML markup** when thinking was combined with tools. The DSML rescue parser in `deepseek-llm/src/lib.rs` (`rescue_raw_tool_calls()`) remains as a safety net, parsing Format A (`<｜DSML｜invoke>`) and Format B (`<｜tool▁call▁begin｜>`) markup into proper `Vec<LlmToolCall>`.

**Workaround for complex failures**: The mode router automatically escalates from V3Autopilot to R1DriveTools when doom-loops, repeated failures (≥5), ambiguous errors, blast radius (≥10 files), or cross-module errors are detected. R1 drives tools via JSON intents with a step budget (`r1_max_steps`, default 30). For policy-related doom-loops (bash.run restrictions), the agent injects tool guidance and resets trackers instead of escalating to R1.

## Supply-Chain Security

`deny.toml` enforces license allowlist and bans yanked crates/unknown registries. Additional CI workflows: `security-gates.yml` (cargo audit + deny + gitleaks), `replay-regression.yml`, `performance-gates.yml`.
