# Parity Target Matrix

This file defines closure-critical parity features and their hard test contract.
Every `Feature ID` must map to at least one automated test or CI gate.

| Feature ID | Domain | Requirement | Test Contract |
|---|---|---|---|
| core.runtime.single_path | core | Single deterministic runtime only | `scripts/runtime_conformance_scan.sh` |
| core.intent.inspect_edit_split | core | Deterministic `TaskIntent` split | `crates/deepseek-agent/tests/runtime_conformance.rs` |
| core.loop.architect_editor_apply_verify | core | Architect->Editor->Apply->Verify execution | `crates/deepseek-agent/tests/architect_editor_loop.rs` |
| core.verify.commit_proposal | core | Verify-pass emits commit proposal, no auto-commit | `crates/deepseek-agent/tests/architect_editor_loop.rs` |
| rpc.prompt_execute.real_runtime | rpc | JSON-RPC `prompt/execute` runs real runtime path | `crates/deepseek-jsonrpc/src/lib.rs` tests |
| rpc.prompt_stream.phase_events | rpc | `prompt/stream_next` exposes phase/event chunks | `crates/deepseek-jsonrpc/src/lib.rs` tests |
| rpc.context.debug_digest | rpc | `context/debug` returns deterministic digest | `crates/deepseek-jsonrpc/src/lib.rs` tests |
| team.auto_lane_trigger | team | Team lanes triggered by deterministic complexity | `crates/deepseek-agent/tests/team_orchestration.rs` |
| team.deterministic_merge_order | team | Lane merge order deterministic and orchestrator-only writes | `crates/deepseek-agent/tests/team_orchestration.rs` |
| workflow.slash.parity | workflow | `/add /drop /read-only /map /map-refresh /run /test /lint /web` available | `crates/deepseek-ui/src/lib.rs` parser tests + `crates/deepseek-cli/tests/cli_json.rs` |
| inspect.bootstrap.analysis_first | context | Repo-ish prompts bootstrap context and respond analysis-first | `crates/deepseek-agent/tests/runtime_conformance.rs` |
| ui.phase_visibility | ui | Phase start/end visibly streamed | `crates/deepseek-ui/src/lib.rs` tests |
| ui.heartbeat_no_silent_stall | ui | Active phase heartbeat prevents silent stalls | `crates/deepseek-ui/src/lib.rs` tests |
| ui.markdown.recovery | ui | Run-on/malformed markdown renders readable structure | `crates/deepseek-ui/src/lib.rs` markdown tests |
| ops.nightly_streak_gate | ops | 3-night parity streak gate enforced in CI | `.github/workflows/ci.yml` + `scripts/parity_streak_gate.py` |

