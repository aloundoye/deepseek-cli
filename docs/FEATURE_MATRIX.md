# DeepSeek CLI Feature Matrix

Updated: 2026-02-26

This matrix reflects runtime behavior after the tool-use loop migration (P3.5).

## 1. Core Agent Architecture

| Area | Status | Runtime Truth |
|---|---|---|
| Tool-use loop (default) | DONE | Code/Ask/Context modes use a fluid think→act→observe loop where the LLM decides which tools to call. |
| Pipeline mode (legacy) | DONE | Available via `--mode pipeline` or `--force-execute`. Uses `Architect (R1) → Editor (V3) → Apply → Verify`. |
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
| Verify-pass commit proposal | DONE | Emits `CommitProposal` stream/event payload with diffstat, verify status, and suggested message. |
| TUI phase visibility | DONE | TUI shows live per-iteration phase transitions and outcomes. |
| UI heartbeat progress | DONE | Active execution phases emit periodic in-progress heartbeat lines in mission control/status without layout changes. |
| Thinking visibility default | DONE | TUI defaults to concise phase summaries (`ui.thinking_visibility = concise`) rather than raw reasoning dumps. |
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
| Explicit no-repo error for repo-ish inspect | DONE | Returns `No repository detected. Run from project root or pass --repo <path>.` instead of generic clarification loops. |
| Ask/context follow-up hardening | DONE | Ask/context responses provide initial analysis first and enforce max 1–2 targeted follow-ups (1 for vague codebase checks). |
| Explicit commit intent workflow | DONE | Verify-pass never auto-commits; user commits explicitly via `/commit` or git subcommands. |
| Workflow parity slash commands | DONE | `/ask /code /architect /chat-mode /add /drop /read-only /map /map-refresh /run /test /lint /web /git /settings /load /save /paste /voice` are supported in chat surfaces. |
| Watch + URL assist flags | DONE | `--watch-files` injects bounded repo marker hints; `--detect-urls` enriches prompt context with bounded URL extracts. |

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
| Aider-style parity | Achieved for core edit execution model (architect/editor split + deterministic apply/verify) via pipeline mode. |
| Claude-like unified reasoning+tools in one model | Achieved via tool-use loop with thinking mode (`deepseek-reasoner` with tools). |
| Tool-calling as execution backbone | Default execution mode. LLM freely calls tools in a think→act→observe loop. |
