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

**Session state machine** (`deepseek-core`): Idle → Planning → ExecutingStep → AwaitingApproval → Verifying → Completed/Failed/Paused. Transitions validated by `is_valid_session_state_transition()`.

**Event sourcing**: All state changes, tool proposals/approvals/results journaled to `.deepseek/events.jsonl` (append-only). SQLite projections in `.deepseek/store.sqlite` for queries.

**Tool flow**: `ToolCall` → `ToolProposal` (policy check) → `ApprovedToolCall` → `ToolResult` → event journal. Tools include `fs.read`, `fs.write`, `fs.edit`, `fs.grep`, `bash.run`, `git.*`, `patch.*`, MCP tools, and plugins.

**Patch staging**: Model edits go to `.deepseek/patches/<id>.json` as unified diffs with base SHA-256. Apply requires SHA match or 3-way merge.

**Model router** (`deepseek-router`): Weighted complexity scoring selects between `deepseek-chat` and `deepseek-reasoner`. Auto-escalation on invalid plans. Threshold default 0.72.

**Traits** in `deepseek-core`: `Planner`, `Executor`, `ModelRouter`, `ToolHost`.

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

## Supply-Chain Security

`deny.toml` enforces license allowlist and bans yanked crates/unknown registries. Additional CI workflows: `security-gates.yml` (cargo audit + deny + gitleaks), `replay-regression.yml`, `performance-gates.yml`.
