#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_LABEL="${CODINGBUDDY_BENCHMARK_MODEL:-scripted-tool-loop}"
MODEL_SLUG="$(printf '%s' "$MODEL_LABEL" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9_-]+/-/g; s/^-+//; s/-+$//')"

echo "[coding-quality-benchmark] running deterministic suite..."
if [[ -n "${CODINGBUDDY_BENCHMARK_MODEL:-}" ]]; then
  echo "[coding-quality-benchmark] model label: ${CODINGBUDDY_BENCHMARK_MODEL}"
fi
if [[ -n "${CODINGBUDDY_BENCHMARK_BASELINE:-}" ]]; then
  echo "[coding-quality-benchmark] baseline override: ${CODINGBUDDY_BENCHMARK_BASELINE}"
fi
if [[ -n "${CODINGBUDDY_BENCHMARK_COMPARE_TO:-}" ]]; then
  echo "[coding-quality-benchmark] compare-to: ${CODINGBUDDY_BENCHMARK_COMPARE_TO}"
fi
cargo test -p codingbuddy-agent --test coding_quality_benchmark -- --nocapture

REPORT_PATH="$ROOT_DIR/.codingbuddy/benchmarks/coding-quality-core.${MODEL_SLUG}.latest.json"
if [[ -f "$REPORT_PATH" ]]; then
  echo "[coding-quality-benchmark] report: $REPORT_PATH"
else
  echo "[coding-quality-benchmark] warning: report file not found at expected path" >&2
fi
