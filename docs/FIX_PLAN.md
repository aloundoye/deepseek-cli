# FIX_PLAN.md — Spec Drift Remediation

**Date:** 2026-02-19
**Status:** COMPLETE — All 4 gaps fixed, tests green, audit re-run passed.

## Gaps (ordered by dependency)

### Gap 1: Missing Event Types (foundation — other features depend on events)

**Spec reference:** Section 4.8 "Event Store and Deterministic Replay" — "All session events are appended to events.jsonl"
Section 2.12 — "Every mode change is recorded as PermissionModeChangedV1"
Section 4.1 — "All transitions are recorded as events"

**Current behavior:** EventKind enum has 57 event types but is missing:
- `SessionStartedV1` (no explicit session-start event)
- `SessionResumedV1` (no explicit session-resume event)
- `ToolDeniedV1` (no explicit tool-denial event)

**Target behavior:** Add 3 new event variants to EventKind.

**Files to modify:**
- `crates/deepseek-core/src/lib.rs` — add event variants
- `crates/deepseek-store/src/lib.rs` — handle in `apply_projection()`

**Tests to add:**
- Unit test: verify new events serialize/deserialize round-trip
- Unit test: `rebuild_from_events` handles new event types

**Acceptance:** `cargo test -p deepseek-core` and `cargo test -p deepseek-store` pass.

---

### Gap 2: Team Policy `permission_mode` Lock

**Spec reference:** Section 2.12 — "If team-policy.json sets permission_mode, it cannot be overridden locally."

**Current behavior:** `TeamPolicyFile` struct doesn't include a `permission_mode` field. `TeamPolicyLocks` doesn't track `permission_mode_locked`. `apply_team_policy_override()` doesn't enforce it.

**Target behavior:** Add `permission_mode: Option<String>` to `TeamPolicyFile`, `permission_mode_locked: bool` to `TeamPolicyLocks`, and apply the override + lock.

**Files to modify:**
- `crates/deepseek-policy/src/lib.rs` — add field, lock logic, apply override

**Tests to add:**
- Unit test: team policy locks permission_mode, local config can't override

**Acceptance:** `cargo test -p deepseek-policy` passes with new test.

---

### Gap 3: Shift+Tab Permission Mode Cycling in TUI

**Spec reference:** Section 2.7 — "Shift+Tab: Cycle permission mode (ask → auto → locked)"
Section 2.12 — "Shift+Tab cycles between modes at the REPL prompt"

**Current behavior:** `PermissionMode::cycle()` exists in policy crate. Status bar renders `[ASK]`/`[AUTO]`/`[LOCKED]`. But no `BackTab` keybinding or event handler in TUI.

**Target behavior:** Add `cycle_permission_mode` keybinding (default: BackTab/Shift+Tab). Handle it in TUI event loop: cycle `status.permission_mode`, update info line.

**Files to modify:**
- `crates/deepseek-ui/src/lib.rs` — add keybinding field, default, handler, override, test

**Tests to add:**
- Unit test: verify BackTab keybinding exists in defaults
- Unit test: verify permission mode cycling through all 3 modes

**Acceptance:** `cargo test -p deepseek-ui` passes with new test.

---

### Gap 4: `/permissions` Dry-Run Evaluator

**Spec reference:** Section 2.4 — "/permissions: View/change permission mode (ask/auto/locked). Dry-run evaluator shows what a tool call would produce under current mode."

**Current behavior:** `PolicyEngine::dry_run()` exists and works. `/permissions` command supports show/set/bash/edits/sandbox/allow/clear-allowlist but NOT a dry-run subcommand.

**Target behavior:** Add `dry-run <tool-name>` subcommand to `/permissions`. Runs the policy engine's dry_run() and displays the result.

**Files to modify:**
- `crates/deepseek-cli/src/main.rs` — add DryRun variant to PermissionsCmd, parse it, run it

**Tests to add:**
- Integration test: `cargo test -p deepseek-cli --test cli_json` (verify dry-run output)

**Acceptance:** `cargo test -p deepseek-cli` passes.
