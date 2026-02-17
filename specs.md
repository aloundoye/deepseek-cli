# RFC: DeepSeek CLI Agent in Rust (Plan-first + Auto Max-Think)

**Status:** Draft
**Date:** 2026-02-13
**Goal:** A production-grade coding-agent CLI powered by DeepSeek, with **explicit planning**, **safe tool use**, and **automatic escalation to “max thinking”** (DeepSeek Reasoner) when needed—without the user manually switching models.

---

## 1) Product behavior requirements

### 1.1 Interaction loop

The CLI must natively support these behaviors:

* **Plan-first:** For most user requests, the agent produces a *Plan* (steps, file targets, tool usage), then executes step-by-step.
* **Two-phase cognition:**

  * **Planner**: makes/updates the plan, chooses which context to fetch, decides on tools.
  * **Executor**: performs actions (search/read/patch/test), guided by the plan.
* **Continuous verification:** After edits, the agent runs checks (tests/build/lint) if allowed, and iterates until goals met.
* **Approval gates:** file writes and shell commands require approval by default.
* **“Max thinking” automatically:** the agent escalates to DeepSeek Reasoner when tasks exceed a complexity threshold.

### 1.2 CLI commands (v1)

* `deepseek chat` — interactive session (default)
* `deepseek ask "<prompt>"` — one-shot response (no tools unless `--tools`)
* `deepseek plan "<prompt>"` — plan only
* `deepseek autopilot "<prompt>"` — long-running autonomous loop (hours or indefinite)
* `deepseek run` — resume session
* `deepseek diff` — show staged changes
* `deepseek apply` — apply staged patch sets (with approvals)
* `deepseek index build|update|status|query`
* `deepseek config edit|show`

### 1.3 “Max thinking” requirement

* Default model is **fast** (`deepseek-chat`).
* When “max thinking” is triggered, the agent transparently switches to **deepseek-reasoner** for that turn (or subtask), then may switch back.
* Escalation must be **observable** (log + UI hint), **auditable** (event log), and **bounded** (budget/time limits).

---

## 2) Architecture: modular monolith with Planner/Executor split

### 2.1 Module boundaries (crates)

```
crates/
  cli/          # clap + TUI/REPL
  core/         # agent runtime, session loop, scheduling
  agent/        # planner+executor logic, state machines
  llm/          # DeepSeek client, streaming, retries
  router/       # model routing + auto max-think policy
  tools/        # tool registry + sandboxed tool host
  diff/         # patch staging, apply, conflict resolution
  index/        # tantivy index + manifest
  store/        # sqlite + event log; projections; migrations
  policy/       # approvals, allowlists, redaction
  observe/      # logs/metrics/tracing
  testkit/      # replay harness, fake LLM, golden tests
```

### 2.2 Core runtime components

* **Agent Runtime (core):** owns the session loop + persistence boundaries.
* **Planner (agent):** produces plan artifacts + chooses tools.
* **Executor (agent):** executes plan steps via tools; stages patches.
* **Model Router (router):** decides **deepseek-chat vs deepseek-reasoner** per turn/subtask.
* **Tool Host (tools):** executes tools with policy + journaling.
* **Index Service (index):** deterministic code search + repo map.

---

## 3) Session model & artifacts

### 3.1 Entities

* **Session**: `{session_id, workspace_root, baseline_commit?, status, budgets, active_plan_id?}`
* **Turn**: user/assistant/tool messages + structured tool calls
* **Plan**: structured plan object (steps, dependencies, verification steps)
* **Step**: `{step_id, title, intent, required_tools, target_files?, done=false}`
* **PatchSet**: staged diffs + metadata; never write directly from the model
* **ToolInvocation**: request/approval/result with timestamps
* **RouterDecision**: `{decision_id, reason_codes[], selected_model, confidence}`

### 3.2 Plan artifact contract (v1)

Stored as JSON:

```json
{
  "plan_id": "uuidv7",
  "version": 1,
  "goal": "string",
  "assumptions": ["..."],
  "steps": [
    {
      "step_id": "uuidv7",
      "title": "Locate relevant modules",
      "intent": "search",
      "tools": ["index.query", "fs.search_rg", "fs.read"],
      "files": [],
      "done": false
    }
  ],
  "verification": ["cargo test", "cargo fmt --check"],
  "risk_notes": ["..."]
}
```

Rules:

* Planner may **revise** a plan (new plan version); old plan is preserved.
* Executor must mark steps done/failed with event emissions.

---

## 4) State machines

### 4.1 Session state machine

States:

* `Idle`
* `Planning`
* `ExecutingStep`
* `AwaitingApproval`
* `Verifying`
* `Completed`
* `Paused`
* `Failed`

Transitions:

* `Idle -> Planning` on first user input
* `Planning -> ExecutingStep` when plan created and accepted (implicit or explicit)
* `ExecutingStep -> AwaitingApproval` if tool/write needs user approval
* `ExecutingStep -> Planning` if plan must be revised (missing files, unexpected failures)
* `ExecutingStep -> Verifying` after patch staged/applied
* `Verifying -> ExecutingStep` if verification fails and fix is needed
* `Verifying -> Completed` when goals met
* Any -> `Failed` on unrecoverable errors

Invariants:

* Every transition emits `SessionStateChanged@v1` with monotonic `seq_no`
* Tool execution cannot occur in `Planning` unless it’s read-only context gathering

### 4.2 Planner/Executor sub-state machines

**Planner**

* `DraftPlan -> (Optional) GatherContext -> FinalizePlan -> EmitPlan`
* If context is insufficient, planner requests read-only tools

**Executor**

* `SelectNextStep -> ExecuteTools -> StagePatch -> (Optional) Apply -> Verify -> MarkStepDone`
* On failure: `RecordFailure -> ProposeRecovery -> (Optional) RevisePlan`

### 4.3 Model router state machine

States per “LLM call unit”:

* `Assess -> SelectModel -> CallLLM -> EvaluateOutcome -> (Optional) EscalateAndRetry`

Escalation rule: Only one automatic escalation retry per unit (prevents loops).

---

## 5) Automatic “Max Thinking” (deepseek-reasoner) design

### 5.1 Objectives

* Use fast model by default for responsiveness and cost.
* Automatically switch to reasoner for tasks requiring deeper reasoning:

  * multi-file architectural change
  * complex bug diagnosis
  * ambiguous requirements
  * repeated failure / low-confidence plan
  * tool errors that require strategy change

### 5.2 Router inputs (signals)

The router evaluates a **Feature Vector** `F` computed from:

* **Prompt complexity**

  * size of user request (tokens)
  * number of constraints (“must”, “guarantee”, “deterministic”, etc.)
* **Repo complexity**

  * number of touched files in last N steps
  * index results breadth
* **Failure history**

  * consecutive tool failures
  * verification failures (tests/lint) count
* **Uncertainty signals**

  * planner confidence score (self-reported numeric)
  * presence of unresolved questions
* **Latency budget**

  * interactive mode vs batch mode
* **User config**

  * `auto_max_think = on|off`
  * `max_think_threshold`

### 5.3 Router decision algorithm (explicit)

Compute score `S`:

```
S = w1*PromptComplexity
  + w2*RepoBreadth
  + w3*FailureStreak
  + w4*VerificationFailures
  + w5*LowConfidence
  + w6*AmbiguityFlags
```

Decision:

* if `S >= THRESHOLD_HIGH` => use `deepseek-reasoner`
* else use `deepseek-chat`

Retry escalation:

* If `deepseek-chat` output fails schema validation, produces an invalid plan, or stalls (no actionable steps), then:

  * escalate once to `deepseek-reasoner`
  * store `RouterEscalation@v1` event with reason codes

### 5.4 Planner/executor routing policy

* Planner is more likely to run on reasoner than executor.

  * Planning call uses `deepseek-reasoner` when:

    * user asks for “design”, “architecture”, “RFC”, “state machines”, “guarantees”
    * change spans > K files
* Executor typically uses chat, except:

  * repeated patch conflicts
  * repeated test failures
  * complex refactors requiring higher-level reasoning

### 5.5 Determinism & auditability

Each LLM call records:

* selected model
* router score + reasons
* budgets (token/time)
* prompt hash + tool schema hash
  So replays can explain “why max thinking triggered”.

---

## 6) Tools, patch staging, and safety (production-grade)

### 6.1 Tool set (v1)

Read-only:

* `fs.list(dir|glob)`
* `fs.read(path) -> {content, sha256}`
* `fs.search_rg(query, paths?, limit)`
* `git.status`, `git.diff`, `git.show(commit:path)`
* `index.query(q, top_k, filters)`

Write/stage:

* `patch.stage(unified_diff) -> {patch_id}`
* `patch.apply(patch_id) -> {applied, conflicts[]}`
* `fs.write(path, content, expected_sha256)` **(optional)**, but prefer patch apply

Exec (restricted):

* `bash.run(cmd, timeout, cwd)` allowlisted; approval gated

### 6.2 Patch staging contract

* Model never writes files directly.
* All edits must be represented as **unified diff** against a known base sha.
* Apply requires:

  * sha match (or a merge strategy)
  * policy approval
  * journaling of applied hunks

Conflict handling:

* If git available: attempt 3-way merge (`merge-base`, `apply --3way`).
* Else: mark conflict and require user.

### 6.3 Sandbox & policy

* Default: approvals for `bash.run`, `patch.apply`
* Allowlist patterns (configurable):

  * `rg`, `git status/diff/show`
  * `cargo test`, `cargo fmt`, `cargo clippy`
  * `npm test`, `pnpm test`, `pytest` (optional)
* Path constraints:

  * deny `..` escapes
  * deny reading common secret locations
* Secret redaction:

  * scan prompt payloads for key patterns; redact before sending

---

## 7) Storage contracts & deterministic rebuild

### 7.1 Storage tech

* SQLite (WAL) + append-only `events.jsonl`
* FS for artifacts (patches/blobs)
* Tantivy for index

### 7.2 Canonical event log

`events.jsonl` is the source of truth; projections rebuildable.

Event kinds (examples):

* `TurnAdded@v1`
* `PlanCreated@v1`, `PlanRevised@v1`, `StepMarked@v1`
* `RouterDecision@v1`, `RouterEscalation@v1`
* `ToolProposed@v1`, `ToolApproved@v1`, `ToolResult@v1`
* `PatchStaged@v1`, `PatchApplied@v1`
* `VerificationRun@v1`

### 7.3 Deterministic rebuild procedure

Given `session_id`:

1. Read `events.jsonl` sequentially; validate schema.
2. Rebuild projections:

   * conversation transcript
   * current plan state
   * staged patches and apply history
   * tool invocation table
   * router decisions timeline
3. **Replay mode (deterministic):**

   * tools are NOT re-run; results come from `ToolResult@v1`
   * LLM responses replay from stored streaming chunks if present
4. Produce deterministic outputs:

   * same plan
   * same staged diffs
   * same “why max thinking triggered” trail

---

## 8) Indexing guarantees & deterministic snapshotting

### 8.1 Manifest-bound index

`manifest.json` stores:

* baseline commit (if git)
* list of `(path, sha256)` or merkle root
* index schema version
* ignore rules hash

Guarantees:

* Queries declare whether results are `fresh` vs `stale`.
* If mismatch detected, index transitions to `Corrupt` and rebuild/update is triggered.

### 8.2 Deterministic index build algorithm (explicit)

If git repo:

1. Identify `baseline_commit` at session start.
2. Enumerate files via `git ls-tree -r --name-only baseline_commit`.
3. Read file blobs via `git show baseline_commit:path`.
4. Hash and index in stable path order.

Else:

1. Walk FS with ignore rules.
2. Read file bytes and compute sha.
3. If changes detected mid-build (sha mismatch), restart build.

---

## 9) Failure modes & recovery (with planner/executor behavior)

### 9.1 LLM failures

* retry on 429/5xx/timeouts with backoff
* if malformed plan/tool schema:

  * re-ask once with stricter schema instruction
  * escalate to reasoner if still invalid

### 9.2 Tool / verification failures

* executor records failure and proposes recovery step
* router may escalate planning for diagnosis:

  * “test failures 2x” => reasoner used for next planning call
* never silently keep retrying; bounded retries + require user awareness

### 9.3 Patch conflicts

* if conflict:

  * planner revises plan: “resolve conflicts” step
  * reasoner may be triggered for merge strategy

---

## 10) Performance targets

Interactive UX targets:

* first token streaming: p95 < 2s (network dependent)
* `rg` searches: p95 < 250ms (medium repo)
* index query: p95 < 50ms
* event append overhead: < 5ms per event

Budgets:

* per-turn time budget (configurable)
* “max thinking” token budget (hard cap) to prevent runaway usage

---

## 11) Implementation blueprint (key Rust traits)

### 11.1 Planner / Executor interfaces

```rust
pub trait Planner {
  fn create_plan(&self, ctx: PlanContext) -> Result<Plan>;
  fn revise_plan(&self, ctx: PlanContext, last_plan: &Plan, failure: Failure) -> Result<Plan>;
}

pub trait Executor {
  fn run_step(&self, ctx: ExecContext, step: &PlanStep) -> Result<StepOutcome>;
}
```

### 11.2 Router interface

```rust
pub trait ModelRouter {
  fn select(&self, unit: LlmUnit, signals: RouterSignals) -> RouterDecision;
}
```

### 11.3 Tool host interface

```rust
pub trait ToolHost {
  fn propose(&self, call: ToolCall) -> ToolProposal;
  fn execute(&self, approved: ApprovedToolCall) -> ToolResult;
}
```

---

## 12) Rollout milestones

### M1 — Plan-first + safe edits (foundational)

* session log + plan artifacts + approvals
* deepseek-chat streaming
* planner/executor loop
* patch stage/apply pipeline
* basic router with explicit thresholds

### M2 — Auto max-thinking + verification loops

* router signals from failures + plan confidence
* escalation retry (bounded)
* build/test tool integration (allowlisted)

### M3 — Deterministic indexing + replay

* manifest-bound tantivy index
* deterministic replay mode + golden tests

---

## Appendix A: Config (TOML) including auto max-think

```toml
[llm]
base_model = "deepseek-chat"
max_think_model = "deepseek-reasoner"
temperature = 0.2

[router]
auto_max_think = true
threshold_high = 0.72
escalate_on_invalid_plan = true
max_escalations_per_unit = 1

[policy]
approve_edits = "ask"
approve_bash = "ask"
allowlist = ["rg", "git", "cargo test", "cargo fmt --check"]

[index]
enabled = true
engine = "tantivy"
```
