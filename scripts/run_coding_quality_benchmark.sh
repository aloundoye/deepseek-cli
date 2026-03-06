#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_LABEL="${CODINGBUDDY_BENCHMARK_MODEL:-scripted-tool-loop}"
MODEL_SLUG="$(printf '%s' "$MODEL_LABEL" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9_-]+/-/g; s/^-+//; s/-+$//')"
SUITE_FILTER="${CODINGBUDDY_BENCHMARK_SUITE:-all}"
LIVE_MODE="${CODINGBUDDY_BENCHMARK_LIVE:-0}"

if [[ "$LIVE_MODE" == "1" ]]; then
  echo "[coding-quality-benchmark] running live suite(s)..."
else
  echo "[coding-quality-benchmark] running deterministic suite(s)..."
fi
echo "[coding-quality-benchmark] suite filter: ${SUITE_FILTER}"
if [[ -n "${CODINGBUDDY_BENCHMARK_MODEL:-}" ]]; then
  echo "[coding-quality-benchmark] model label: ${CODINGBUDDY_BENCHMARK_MODEL}"
fi
if [[ -n "${CODINGBUDDY_BENCHMARK_PROVIDER:-}" ]]; then
  echo "[coding-quality-benchmark] provider override: ${CODINGBUDDY_BENCHMARK_PROVIDER}"
fi
if [[ -n "${CODINGBUDDY_BENCHMARK_PROFILE:-}" ]]; then
  echo "[coding-quality-benchmark] profile override: ${CODINGBUDDY_BENCHMARK_PROFILE}"
fi
if [[ -n "${CODINGBUDDY_BENCHMARK_BASELINE:-}" ]]; then
  echo "[coding-quality-benchmark] baseline override: ${CODINGBUDDY_BENCHMARK_BASELINE}"
fi
if [[ -n "${CODINGBUDDY_BENCHMARK_COMPARE_TO:-}" ]]; then
  echo "[coding-quality-benchmark] compare-to: ${CODINGBUDDY_BENCHMARK_COMPARE_TO}"
fi

if [[ "$LIVE_MODE" == "1" ]]; then
  cargo test -p codingbuddy-agent --test coding_quality_benchmark -- --ignored --nocapture
else
  cargo test -p codingbuddy-agent --test coding_quality_benchmark -- --nocapture
fi

for SUITE in coding-quality-core coding-quality-repo; do
  if [[ "$SUITE_FILTER" != "all" && "$SUITE_FILTER" != "deterministic" && "$SUITE_FILTER" != "$SUITE" ]]; then
    continue
  fi
  REPORT_PATH="$ROOT_DIR/.codingbuddy/benchmarks/${SUITE}.${MODEL_SLUG}.latest.json"
  if [[ -f "$REPORT_PATH" ]]; then
    echo "[coding-quality-benchmark] report: $REPORT_PATH"
  else
    echo "[coding-quality-benchmark] warning: report file not found at expected path: $REPORT_PATH" >&2
  fi
done
