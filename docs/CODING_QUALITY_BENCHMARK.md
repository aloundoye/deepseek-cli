# Coding Quality Benchmark

This benchmark tracks whether coding-task behavior is improving or regressing over time.

It can also compare one model lane directly against another compatible report.

## Scope

The current deterministic suite covers 4 task classes:

- `edit-single-file`
- `debug-bugfix`
- `refactor-rename`
- `multi-file-update`

Each case records:

- pass/fail
- tool invocation count
- retry count (invocations above expected minimum)
- completion quality score (`1.0`, `0.5`, `0.0`)
- execution duration

## Run

Use the helper script:

```bash
./scripts/run_coding_quality_benchmark.sh
```

It runs:

```bash
cargo test -p codingbuddy-agent --test coding_quality_benchmark -- --nocapture
```

The test writes a report to:

```text
.codingbuddy/benchmarks/<suite>.<model>.latest.json
```

Override the report model label when you want the same suite to represent a real lane:

```bash
CODINGBUDDY_BENCHMARK_MODEL=deepseek-coder \
./scripts/run_coding_quality_benchmark.sh
```

## Baseline Gate

Default baseline file:

```text
docs/benchmarks/coding_quality_baseline.json
```

Optional overrides:

- `CODINGBUDDY_BENCHMARK_BASELINE=/path/to/baseline.json`
- `docs/benchmarks/<suite>.<model>.baseline.json`

Gate rule in the test:

- fail if pass-rate drops by more than `5.0` percentage points vs baseline
- fail if average completion quality score drops by more than `0.10`
- fail if average retries increase by more than `0.50`
- fail if baseline suite/model identity does not match current report

Update baseline only when an intentional quality shift is accepted.

## Comparative Mode

Compare the current run against another compatible report:

```bash
CODINGBUDDY_BENCHMARK_MODEL=deepseek-coder \
CODINGBUDDY_BENCHMARK_COMPARE_TO=/path/to/claude-code-report.json \
./scripts/run_coding_quality_benchmark.sh
```

Comparison rules:

- suite must match
- case ids must match
- case categories must match

When comparison is enabled, the test writes:

```text
.codingbuddy/benchmarks/<suite>.<current-model>.vs.<reference-model>.comparison.json
```

The comparison summary reports:

- pass-rate delta
- average quality delta
- average retry delta
- average duration delta
- improved case count
- regressed case count
