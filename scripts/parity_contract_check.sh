#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MATRIX="docs/parity/PARITY_TARGET_MATRIX.md"
if [[ ! -f "$MATRIX" ]]; then
  echo "[parity-contract] missing $MATRIX"
  exit 1
fi

required_features=(
  "core.runtime.single_path"
  "core.intent.inspect_edit_split"
  "core.loop.architect_editor_apply_verify"
  "core.verify.commit_proposal"
  "rpc.prompt_execute.real_runtime"
  "rpc.prompt_stream.phase_events"
  "rpc.context.debug_digest"
  "team.auto_lane_trigger"
  "team.deterministic_merge_order"
  "workflow.slash.parity"
  "workflow.slash.mode_aliases"
  "workflow.git_passthrough"
  "workflow.profile.load_save"
  "workflow.voice.capability_probe"
  "workflow.watch_files.hints"
  "workflow.detect_urls.enrichment"
  "inspect.bootstrap.analysis_first"
  "ui.phase_visibility"
  "ui.heartbeat_no_silent_stall"
  "ui.markdown.recovery"
  "ops.nightly_streak_gate"
)

fail=0
matrix_features="$(awk -F'|' '/^\|/{if ($0 !~ /^\|---/) {gsub(/^ +| +$/,"",$2); if ($2 != "") print $2}}' "$MATRIX")"

for feature in "${required_features[@]}"; do
  if ! grep -qx "$feature" <<< "$matrix_features"; then
    echo "[parity-contract] missing required feature row: $feature"
    fail=1
  fi
done

while IFS= read -r line; do
  [[ "$line" =~ ^\| ]] || continue
  [[ "$line" =~ ^\|--- ]] && continue
  feature="$(echo "$line" | awk -F'|' '{gsub(/^ +| +$/,"",$2); print $2}')"
  contract="$(echo "$line" | awk -F'|' '{gsub(/^ +| +$/,"",$5); print $5}')"
  if [[ -z "$feature" ]]; then
    continue
  fi
  if [[ -z "$contract" || "$contract" == "TBD" ]]; then
    echo "[parity-contract] missing test contract for feature: $feature"
    fail=1
    continue
  fi

  # Extract obvious file paths and ensure they exist (bash 3 compatible).
  paths=$(echo "$contract" | rg -o "(scripts|crates|docs|\.github)/[A-Za-z0-9_./-]+" || true)
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    if [[ ! -e "$path" ]]; then
      echo "[parity-contract] missing contract target for $feature: $path"
      fail=1
    fi
  done <<< "$paths"
done < "$MATRIX"

if [[ "$fail" -ne 0 ]]; then
  echo "[parity-contract] FAILED"
  exit 1
fi

echo "[parity-contract] PASS"
