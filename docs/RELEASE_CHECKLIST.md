# DeepSeek CLI — Release Checklist & Verification Results

This document records the release checklist and verification results for DeepSeek CLI.

---

## Build Verification

| Check | Command | Result |
|-------|---------|--------|
| Format | `cargo fmt --all -- --check` | PASS (no output, exit 0) |
| Lint | `cargo clippy --workspace --all-targets -- -D warnings` | PASS (Finished dev profile, 0 warnings) |
| Tests | `cargo test --workspace --all-targets` | PASS (165 tests, 0 failures across 18 crates) |

---

## Test Summary by Crate

| Crate | Passed | Notes |
|-------|--------|-------|
| deepseek-cli | 39 | Integration tests: cli_json 35 passed |
| deepseek-core | 5 | Includes proptest property-based tests, `new_event_types_round_trip_via_serde` |
| deepseek-agent | 3 | |
| deepseek-llm | 20 | |
| deepseek-diff | 4 | |
| deepseek-index | 2 | |
| deepseek-hooks | 2 | |
| deepseek-mcp | 2 | |
| deepseek-memory | 1 | |
| deepseek-observe | 1 | |
| deepseek-policy | 19 | Includes `team_policy_locks_permission_mode`, `team_policy_permission_mode_locked_flag` |
| deepseek-router | 1 | |
| deepseek-skills | 1 | |
| deepseek-store | 2 | |
| deepseek-subagent | 4 | |
| deepseek-testkit | 1 | |
| deepseek-tools | 15 | |
| deepseek-ui | 11 | Includes `default_keybindings_include_cycle_permission_mode` |
| **Total** | **165** | **0 failures** |

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
