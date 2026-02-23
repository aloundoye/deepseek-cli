# DeepSeek CLI Feature Matrix

Updated: 2026-02-23

This matrix reflects runtime behavior after the full Architect->Editor->Apply->Verify migration.

## 1. Core Agent Architecture

| Area | Status | Runtime Truth |
|---|---|---|
| Single execution path | DONE | Edit-execution uses only `Architect (R1) -> Editor (V3) -> Apply -> Verify`. |
| Legacy mode router in execution loop | REMOVED | No runtime fallback/escalation path is used for edit execution. |
| Legacy R1 tool-driving mode | REMOVED | No R1 JSON-intent drive-tools path in the execution loop. |
| Deterministic edit contract | DONE | Unified diff is the only accepted edit payload from Editor (with bounded `NEED_CONTEXT|...` requests). |
| Deterministic local apply | DONE | Diffs are validated and applied through `PatchStore` with path and scope checks. |
| Deterministic local verify | DONE | Verify success depends on actual command exit status and timeout checks. |
| Analysis-only non-edit path | DONE | `analyze_with_options` is used for review and no-edit workflows. |
| Core-loop subagent orchestration | DISABLED | Subagents are not spawned automatically inside the default single-lane core loop. |
| Team-lane orchestration (`--teammate-mode`) | DONE | Optional deterministic lane composition uses isolated worktrees, patch artifacts, ordered merge, and global verify. |

## 2. Contracts and Safety

| Area | Status | Runtime Truth |
|---|---|---|
| Architect contract enforcement | DONE | Strict `ARCHITECT_PLAN_V1` parsing with bounded repair retries. |
| Editor contract enforcement | DONE | Strict unified-diff or `NEED_CONTEXT` parsing with bounded repair retries. |
| File scope bounds | DONE | Max files/bytes per iteration and architect-file whitelist enforcement. |
| Path safety | DONE | Absolute paths, repo-root escapes, and `.git/` mutations are rejected at apply phase. |
| Failure classification | DONE | Verify failures are classified deterministically (`Mechanical`, `Repeated`, `DesignMismatch`) and routed accordingly. |
| Safety gate | DONE | Large patches require explicit approval before apply; checkpoints are created around apply. |
| Verification command policy | DONE | Verification commands run through policy + tool host; failures fed back into loop. |

## 3. Streaming and UI

| Area | Status | Runtime Truth |
|---|---|---|
| Phase streaming | DONE | Emits `Architect*`, `Editor*`, `Apply*`, `Verify*` stream chunks. |
| TUI phase visibility | DONE | TUI shows live per-iteration phase transitions and outcomes. |
| stream-json phase visibility | DONE | `--output-format stream-json` emits phase events. |

## 4. CLI and Config

| Area | Status | Runtime Truth |
|---|---|---|
| `chat/ask` tool default | DONE | `--tools=true` by default; `--tools=false` forces analysis path. |
| Execution overrides | DONE | `--force-execute` and `--plan-only` are available and mutually exclusive. |
| Removed legacy break-glass flag | DONE | `--allow-r1-drive-tools` is not part of current CLI behavior. |
| Loop configuration | DONE | `[agent_loop]` controls iteration/retry/bounds/verify timeout + classifier/context/safety knobs. |
| Router configuration cleanup | DONE | Active runtime/router config removed; only legacy read-compat event decode remains for historical logs. |
| Auto context bootstrap for ask/context | DONE | Repo-ish prompts inject deterministic `AUTO_CONTEXT_BOOTSTRAP_V1` packet, plus baseline audit for vague codebase checks. |
| Ask/context follow-up hardening | DONE | Ask/context responses provide initial analysis first and enforce max 1–2 targeted follow-ups (1 for vague codebase checks). |

## 5. Reliability and Integrations

| Area | Status | Runtime Truth |
|---|---|---|
| MCP stdio resource resolution | DONE | stdio `@server:uri` uses real `resources/read` path; failures are explicit markers. |
| Chrome strict-live default | DONE | No fake/stub success payloads unless explicitly configured for fallback. |
| Structured review output + publish | DONE | Strict findings schema with optional GitHub publish (`gh`) flow. |
| Remote environment orchestration | DONE | `remote-env exec|run-agent|logs` over SSH profiles. |
| Teleport one-time links | DONE | `teleport link` and `teleport consume` support secure one-time handoff flow. |
| Terminal image fallback | DONE | `visual show` supports inline/external/path fallback by policy. |
| CI nightly parity streak gate | DONE | Nightly journey report artifact + PR streak gate enforce 3 successful nightly runs. |

## 6. Compatibility Notes

| Topic | Current Position |
|---|---|
| Aider-style parity | Achieved for core edit execution model (architect/editor split + deterministic apply/verify). |
| Claude-like unified reasoning+tools in one model | Not applicable with DeepSeek split; runtime uses explicit two-role pipeline instead. |
| Tool-calling as execution backbone | Intentionally not used in core edit loop. |
