# SPEC_DRIFT_REPORT.md — DeepSeek CLI

**Date:** 2026-02-19
**Audit Result:** PASS — No remaining spec drift.

---

## Previously Identified Gaps (All Fixed)

### Gap 1: Missing Event Types — FIXED

**Spec reference:** Section 4.8 "All session events are appended to events.jsonl"

**What was missing:**
- `SessionStartedV1` — no explicit session-start event
- `SessionResumedV1` — no explicit session-resume event
- `ToolDeniedV1` — no explicit tool-denial event

**Fix applied:**
- Added 3 new variants to `EventKind` enum in `crates/deepseek-core/src/lib.rs:513-524`
- Added handling in `event_kind_name()` in `crates/deepseek-store/src/lib.rs:2757-2759`
- Added serde round-trip test: `new_event_types_round_trip_via_serde`

**Evidence:** `cargo test -p deepseek-core -- new_event_types` passes.

---

### Gap 2: Team Policy `permission_mode` Lock — FIXED

**Spec reference:** Section 2.12 "If team-policy.json sets permission_mode, it cannot be overridden locally."

**What was missing:**
- `TeamPolicyFile` had no `permission_mode` field
- `TeamPolicyLocks` had no `permission_mode_locked` flag
- `apply_team_policy_override()` didn't enforce permission_mode from team policy

**Fix applied:**
- Added `permission_mode: Option<String>` to `TeamPolicyFile` in `crates/deepseek-policy/src/lib.rs:450-451`
- Added `permission_mode_locked: bool` to `TeamPolicyLocks` in `crates/deepseek-policy/src/lib.rs:459`
- Updated `team_policy_locks()` to populate the new field
- Updated `apply_team_policy_override()` to apply permission_mode from team policy
- Updated `has_permission_locks()` to include permission_mode_locked
- Added `permission_mode_locked` to JSON output in CLI
- Added 2 tests: `team_policy_locks_permission_mode`, `team_policy_permission_mode_locked_flag`

**Evidence:** `cargo test -p deepseek-policy -- team_policy` passes.

---

### Gap 3: Shift+Tab Permission Mode Cycling in TUI — FIXED

**Spec reference:** Section 2.7 "Shift+Tab: Cycle permission mode (ask → auto → locked)"
Section 2.12 "Shift+Tab cycles between modes at the REPL prompt: ask → auto → locked → ask"

**What was missing:**
- No `cycle_permission_mode` field in `KeyBindings` struct
- No `BackTab` keybinding default
- No event handler in TUI event loop

**Fix applied:**
- Added `cycle_permission_mode: KeyEvent` to `KeyBindings` struct in `crates/deepseek-ui/src/lib.rs:647`
- Default: `KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT)` at line 684
- Added override support in `KeyBindingsFile` and `apply_overrides()`
- Added event handler in TUI loop: cycles ask → auto → locked → ask, updates status bar
- Made `status` parameter mutable in `run_tui_shell_with_bindings()`
- Added test: `default_keybindings_include_cycle_permission_mode`

**Evidence:** `cargo test -p deepseek-ui -- cycle_permission` passes.

---

### Gap 4: `/permissions` Dry-Run Evaluator — FIXED

**Spec reference:** Section 2.4 "/permissions: Dry-run evaluator shows what a tool call would produce under current mode."

**What was missing:**
- `/permissions` slash command had no dry-run subcommand
- `PolicyEngine::dry_run()` existed but was not exposed via CLI

**Fix applied:**
- Added `DryRun(PermissionsDryRunArgs)` variant to `PermissionsCmd` enum in `crates/deepseek-cli/src/main.rs:800`
- Added parsing for `/permissions dry-run <tool-name>` in `parse_permissions_cmd()`
- Added handler in `permissions_payload()` that constructs a ToolCall and calls `engine.dry_run()`
- Added display logic in `run_permissions()` for dry-run output
- Both JSON and text output modes supported

**Evidence:** `cargo clippy --workspace --all-targets -- -D warnings` passes; integration tests pass.

---

## Remaining Drift: None

All spec items are now at DONE status. The implementation matches specs.md with zero known gaps.
