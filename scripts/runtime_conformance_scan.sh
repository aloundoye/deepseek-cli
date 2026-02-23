#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

fail=0

check_forbidden_symbol() {
  local symbol="$1"
  local matches
  matches=$(
    rg -n \
      --hidden \
      --glob '!target/**' \
      --glob '!.git/**' \
      --glob '!**/tests/**' \
      "$symbol" \
      crates \
      .github \
      config.example.toml \
      config.example.json \
      || true
  )
  if [[ -n "$matches" ]]; then
    echo "[conformance] forbidden symbol detected: $symbol"
    echo "$matches"
    fail=1
  fi
}

# Legacy runtime symbols that must not exist in active runtime code.
check_forbidden_symbol "mode_router"
check_forbidden_symbol "r1_drive_tools"
check_forbidden_symbol "R1DriveTools"
check_forbidden_symbol "dsml_rescue"
check_forbidden_symbol "contains_dsml_markup"
check_forbidden_symbol "DSML_MARKERS"
check_forbidden_symbol "dsml_buffering"
check_forbidden_symbol "deepseek-router"
check_forbidden_symbol "\"status\": \"queued\""
check_forbidden_symbol "Processing prompt:"

# Router event symbols are allowed only in deepseek-core compat parser.
router_matches=$(rg -n --hidden --glob '!target/**' --glob '!.git/**' "RouterDecisionV1|RouterEscalationV1" crates || true)
if [[ -n "$router_matches" ]]; then
  filtered=$(echo "$router_matches" | rg -v "crates/deepseek-core/src/lib.rs" || true)
  if [[ -n "$filtered" ]]; then
    echo "[conformance] router event symbols found outside compat shim:"
    echo "$filtered"
    fail=1
  fi
fi

# Router config must not be active in examples.
if rg -n "^\[router\]|\"router\"\s*:" config.example.toml config.example.json >/dev/null 2>&1; then
  echo "[conformance] router config block still present in config examples"
  rg -n "^\[router\]|\"router\"\s*:" config.example.toml config.example.json || true
  fail=1
fi

if [[ "$fail" -ne 0 ]]; then
  echo "[conformance] FAILED"
  exit 1
fi

echo "[conformance] PASS"
