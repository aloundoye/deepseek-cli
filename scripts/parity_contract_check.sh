#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MATRIX="docs/parity/PARITY_TARGET_MATRIX.md"
if [[ ! -f "$MATRIX" ]]; then
  echo "[parity-contract] missing $MATRIX"
  exit 1
fi

fail=0
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
