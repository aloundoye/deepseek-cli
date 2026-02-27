# DeepSeek CLI — Release Checklist & Verification Results

This document records the release checklist and verification results for DeepSeek CLI.

---

## Build Verification

| Check | Command | Result |
|-------|---------|--------|
| Format | `cargo fmt --all -- --check` | PASS (no output, exit 0) |
| Lint | `cargo clippy --workspace --all-targets -- -D warnings` | PASS (Finished dev profile, 0 warnings) |
| Tests | `cargo test --workspace --all-targets` | PASS (870 tests, 0 failures across 25 crates) |

---

## Test Summary by Crate

| Crate | Passed | Notes |
|-------|--------|-------|
| deepseek-cli | 174 | Integration tests: cli_json, subcommand handlers |
| deepseek-core | 60 | Includes proptest, event round-trip, config defaults |
| deepseek-agent | 56 | Tool-use loop, complexity, prompts, bootstrap, error recovery |
| deepseek-agent (integration) | 37 | tool_use_default, retrieval_wiring, runtime_conformance |
| deepseek-llm | 20 | LLM client, streaming, request building |
| deepseek-diff | 4 | Diff parsing, patch staging |
| deepseek-index | 2 | Tantivy index operations |
| deepseek-hooks | 14 | Lifecycle events, hook runtime |
| deepseek-mcp | 9 | MCP server management |
| deepseek-memory | 6 | Long-term memory, checkpoints |
| deepseek-observe | 1 | Structured logging |
| deepseek-policy | 21 | Permissions, team policy, managed settings |
| deepseek-skills | 8 | Skill discovery, forked execution |
| deepseek-store | 4 | Session persistence, event log |
| deepseek-subagent | 13 | Background tasks, worktree isolation |
| deepseek-testkit | 1 | Test utilities |
| deepseek-tools | 55 | Tool definitions, enriched descriptions |
| deepseek-ui | 29 | TUI rendering, keybindings, ghost text |
| deepseek-context | 41 | Dependency analysis, file suggestions |
| deepseek-local-ml | 60 | Chunker, retrieval, vector index, privacy, embeddings |
| deepseek-jsonrpc | 16 | JSON-RPC server |
| deepseek-chrome | 20 | Chrome native host bridge |
| deepseek-errors | 2 | Error types |
| **Total** | **870** | **0 failures across 25 crates** |

---

## CI Workflows (10 total)

| Workflow | Purpose |
|----------|---------|
| `ci.yml` | Multi-OS build/test (ubuntu, macos, windows) |
| `release.yml` | 6-target cross-compilation + checksums + SBOM |
| `security-gates.yml` | cargo audit + deny + gitleaks |
| `replay-regression.yml` | Deterministic replay validation |
| `performance-gates.yml` | P95 latency SLO |
| `live-deepseek-smoke.yml` | Daily DeepSeek API smoke test |
| `release-readiness.yml` | Pre-release gate |
| `homebrew.yml` | Auto formula generation |
| `winget.yml` | Auto manifest generation |
| `parity-publication.yml` | Weekly parity reports |

---

## Cross-Compilation Targets (6)

| Target Triple | Platform |
|---------------|----------|
| `x86_64-unknown-linux-gnu` | Linux x86_64 |
| `aarch64-unknown-linux-gnu` | Linux ARM64 |
| `x86_64-apple-darwin` | macOS x86_64 |
| `aarch64-apple-darwin` | macOS ARM64 (Apple Silicon) |
| `x86_64-pc-windows-msvc` | Windows x86_64 |
| `aarch64-pc-windows-msvc` | Windows ARM64 |

---

## Installers

| Installer | Details |
|-----------|---------|
| `scripts/install.sh` | Unix: auto-detect OS/arch, checksum verify, PATH guidance |
| `scripts/install.ps1` | Windows: PowerShell, checksum verify |
| Homebrew | Automated via `homebrew.yml` |
| Winget | Automated via `winget.yml` |

---

## Supply Chain

| Mechanism | Coverage |
|-----------|----------|
| `deny.toml` | License allowlist, registry policy, ban yanked crates |
| Security gates | cargo audit, cargo deny, gitleaks |

---

## Deferred Items

Smoke tests requiring the actual binary (`deepseek --version`, REPL commands) are deferred to integration/CI since they require an API key and a full binary build.
