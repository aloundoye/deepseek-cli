# DeepSeek CLI Architecture Spec

Updated: 2026-02-23

## 1. Scope
This spec defines the production agent architecture for edit-execution flows in DeepSeek CLI.

The runtime uses one deterministic pipeline:

`Architect (R1) -> Editor (V3) -> Apply -> Verify`

There is no legacy dual-mode routing/escalation path in the execution loop.

## 2. Why This Architecture
DeepSeek provides a capability split:
- `deepseek-reasoner` (R1): stronger reasoning, no tool calling.
- `deepseek-chat` (V3): better structured code-edit output and general execution support.

To avoid fragile tool-intent orchestration, DeepSeek CLI uses a deterministic edit contract. The models produce plans and diffs; the local harness owns filesystem mutation and verification.

## 3. Execution Model
### 3.1 Architect (R1)
Model: `llm.max_think_model`

Input:
- User request
- Minimal repository map
- Prior apply/verify failure feedback

Output contract (strict, line-oriented):

```text
ARCHITECT_PLAN_V1
PLAN|<step text>
FILE|<path>|<intent>
VERIFY|<command>
ACCEPT|<criterion>
NO_EDIT|true|<reason>        # optional
ARCHITECT_PLAN_END
```

Rules:
- No tool calls
- No JSON
- No diffs
- File paths must be workspace-relative

### 3.2 Editor (V3)
Model: `llm.base_model`

Input:
- Architect plan
- Exact content of architect-declared files only
- Prior apply/verify feedback

Output contract (strict):
- Unified diff payload only, or
- `NEED_FILE|path[:start-end]` lines

Rules:
- No commentary outside the diff/NEED_FILE contract
- No edits outside architect-declared files

### 3.3 Apply (deterministic local)
- Validate unified diff structure and size
- Enforce path safety:
  - no absolute paths
  - no `.git/` mutation
  - targets must be subset of architect file list
- Stage/apply via `deepseek-diff::PatchStore` (`git apply --3way`)
- Return structured apply failures for retry

### 3.4 Verify (deterministic local)
- Run architect-provided `VERIFY|` commands first
- Fallback to derived commands by project type when none are provided
- Verification success is based on command exit status (`status == 0`) and timeout checks
- On failure, summarize output and feed back into next architect pass

## 4. Loop Policy
Implemented in `crates/deepseek-agent/src/loop.rs`.

Per iteration:
1. Architect
2. Editor (with optional `NEED_FILE` round trips)
3. Apply
4. Verify

Defaults:
- `max_iterations = 6`
- `architect_parse_retries = 2`
- `editor_parse_retries = 2`
- `max_files_per_iteration = 12`
- `max_file_bytes = 200000`
- `max_diff_bytes = 400000`
- `verify_timeout_seconds = 60`

No hidden escalation path is used.

## 5. Analysis-Only Path
Non-edit commands (for example `deepseek review`) use a separate analysis API:
- `AgentEngine::analyze_with_options(...)`
- No apply step
- No verify step
- No filesystem mutation side effects

## 6. Streaming Contract
Execution emits phase lifecycle chunks:
- `ArchitectStarted`, `ArchitectCompleted`
- `EditorStarted`, `EditorCompleted`
- `ApplyStarted`, `ApplyCompleted`
- `VerifyStarted`, `VerifyCompleted`

These are consumed by CLI and TUI for real-time visibility.

## 7. Module Boundaries
Core modules:
- `crates/deepseek-agent/src/architect.rs`
- `crates/deepseek-agent/src/editor.rs`
- `crates/deepseek-agent/src/apply.rs`
- `crates/deepseek-agent/src/verify.rs`
- `crates/deepseek-agent/src/loop.rs`
- `crates/deepseek-agent/src/analysis.rs`

`chat_with_options` remains the compatibility entry point and dispatches to:
- loop execution when `tools=true`
- analysis path when `tools=false`

## 8. Configuration
### 8.1 Router config (`[router]`)
Router remains available for model selection heuristics in non-loop contexts:
- `auto_max_think`
- `threshold_high`
- `w1..w6`

Legacy escalation keys are removed.

### 8.2 Agent loop config (`[agent_loop]`)
- `max_iterations`
- `architect_parse_retries`
- `editor_parse_retries`
- `max_files_per_iteration`
- `max_file_bytes`
- `max_diff_bytes`
- `verify_timeout_seconds`

## 9. CLI Behavior
- `chat` and `ask` default to `--tools=true`.
- `--tools=false` forces analysis-only behavior.
- No `--allow-r1-drive-tools` flag exists in this architecture.

## 10. Removed Legacy Paths
Removed from execution path:
- Mode router/escalation loop
- R1 JSON-intent drive-tools flow
- DSML rescue parser used for tool-call salvage
- Legacy doom-loop rescue logic
- Legacy tool-call orchestration loop as primary execution mechanism

## 11. Test Requirements
Required integration scenarios for `deepseek-agent`:
- single file edit succeeds
- multi file edit succeeds
- patch apply failure recovers
- verification failure recovers

These scenarios are implemented under `crates/deepseek-agent/tests/architect_editor_loop.rs`.
