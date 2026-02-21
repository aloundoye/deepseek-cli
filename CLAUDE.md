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

**Model router** (`deepseek-router`): Weighted complexity scoring toggles thinking mode on `deepseek-chat`. When the score crosses the threshold (default 0.72), the router sets `thinking_enabled=true` instead of switching models. However, **thinking mode is forcibly disabled when tools are active** (see "Known DeepSeek API Limitations" below). Thinking mode is only applied for pure text responses (no tools) and the legacy planner path. `deepseek-reasoner` does NOT support function calling and is only used for legacy planner paths without tools.

**TUI** (`deepseek-ui`): Simple chat interface — full-width transcript with auto-scroll, input area, status bar. Streaming tokens displayed in real-time via `StreamCallback`.

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

### Thinking mode + tools = DSML markup leak

When `thinking: {"type": "enabled", "budget_tokens": N}` is sent alongside `tools: [...]` in a `/chat/completions` request, the DeepSeek API intermittently outputs **raw DSML markup** (e.g. `<｜DSML｜function_calls>`, `<｜DSML｜invoke name="...">`) in the content stream instead of returning structured `tool_calls` in the response. This is a known upstream issue (tracked in sglang #14695, vllm #28219, vercel/ai #10778).

**Consequence**: Thinking mode and function calling are mutually exclusive in practice. The codebase enforces this in two places:

1. **`deepseek-agent/src/lib.rs` — `chat_with_options()`**: The `thinking` config is only built when `decision.thinking_enabled && !options.tools`. When tools are active, `thinking` is always `None` and `temperature` is set to `Some(0.0)`. The escalation retry path also skips thinking when tools are present.

2. **`deepseek-llm/src/lib.rs` — DSML rescue parser**: As a safety net, `rescue_raw_tool_calls()` detects DSML markup in response content and parses it into proper `Vec<LlmToolCall>`. This covers both Format A (`<｜DSML｜invoke>` blocks) and Format B (legacy `<｜tool▁call▁begin｜>` markers). The rescue runs on all three response paths: non-streaming, streaming payload, and live streaming. The streaming path also buffers DSML content to prevent raw markup from appearing in the TUI.

**What still gets thinking mode**: Pure text responses (no tools), the legacy planner path, and any turn where `options.tools = false`.

**If DeepSeek fixes this upstream**: Remove the `&& !options.tools` guard in `chat_with_options()` and the `tools.is_empty()` check in the escalation retry. The DSML rescue parser can be kept as a defensive fallback.

## Supply-Chain Security

`deny.toml` enforces license allowlist and bans yanked crates/unknown registries. Additional CI workflows: `security-gates.yml` (cargo audit + deny + gitleaks), `replay-regression.yml`, `performance-gates.yml`.
