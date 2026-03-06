# Coding Quality Benchmark

This benchmark tracks whether coding-task behavior is improving or regressing over time.

It can also compare one model lane directly against another compatible report.

## Scope

There are two deterministic suites:

- `coding-quality-core`
  - `edit-single-file`
  - `debug-bugfix`
  - `refactor-rename`
  - `multi-file-update`
- `coding-quality-repo`
  - `repo-wide-rename-exports-tests`
  - `failing-unit-test-fix`
  - `compiler-build-error-fix`
  - `ambiguous-refactor-acceptance`
  - `tool-denial-recovery`
  - `compaction-pressure-follow-up-edit`

Each case records:

- pass/fail
- patch-applied
- build-passed
- tests-passed
- tool invocation count
- verification attempt count
- retry count (invocations above expected minimum)
- denied tool attempts
- compaction count
- completion quality score (`1.0`, `0.5`, `0.0`)
- execution duration
- input/output/cache token usage
- estimated cost

## Run

Use the helper script:

```bash
./scripts/run_coding_quality_benchmark.sh
```

By default it runs both deterministic suites:

```bash
cargo test -p codingbuddy-agent --test coding_quality_benchmark -- --nocapture
```

The test writes a report to:

```text
.codingbuddy/benchmarks/<suite>.<model>.latest.json
```

Run a single suite:

```bash
CODINGBUDDY_BENCHMARK_SUITE=coding-quality-repo \
./scripts/run_coding_quality_benchmark.sh
```

Override the report model label when you want the same suite to represent a named lane:

```bash
CODINGBUDDY_BENCHMARK_MODEL=deepseek-coder \
./scripts/run_coding_quality_benchmark.sh
```

Attach provider/profile metadata to the report:

```bash
CODINGBUDDY_BENCHMARK_MODEL=deepseek-coder \
CODINGBUDDY_BENCHMARK_PROVIDER=deepseek \
CODINGBUDDY_BENCHMARK_PROFILE=build \
./scripts/run_coding_quality_benchmark.sh
```

## Baseline Gate

Default baseline file:

```text
docs/benchmarks/<suite>.<model>.baseline.json
```

Optional overrides:

- `CODINGBUDDY_BENCHMARK_BASELINE=/path/to/baseline.json`
- `docs/benchmarks/<suite>.<model>.baseline.json`
- legacy fallback: `docs/benchmarks/coding_quality_baseline.json` for `coding-quality-core` + `scripted-tool-loop`

Gate rule in the test:

- fail if pass-rate drops by more than `5.0` percentage points vs baseline
- fail if average completion quality score drops by more than `0.10`
- fail if average retries increase by more than `0.50`
- fail if baseline suite/model identity does not match current report

Update baseline only when an intentional quality shift is accepted.

## Comparative Mode

Compare the current run against another compatible report:

```bash
CODINGBUDDY_BENCHMARK_SUITE=coding-quality-repo \
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
- patch/build/test pass-rate delta
- average quality delta
- average verification-attempt delta
- average retry delta
- average denied-tool delta
- average compaction delta
- average duration delta
- average input/output token delta
- average cost delta
- improved case count
- regressed case count

## Live Lane

Ignored live suites are available for manual or nightly runs. They reuse the same case IDs and categories as the deterministic suites.

Run a live suite:

```bash
CODINGBUDDY_BENCHMARK_LIVE=1 \
CODINGBUDDY_BENCHMARK_SUITE=coding-quality-core \
CODINGBUDDY_BENCHMARK_MODEL=deepseek-coder-live \
CODINGBUDDY_BENCHMARK_PROVIDER=deepseek \
CODINGBUDDY_BENCHMARK_PROFILE=build \
./scripts/run_coding_quality_benchmark.sh
```

Notes:

- live runs are `#[ignore]` and do not execute in the normal workspace test path
- the live report lane is recorded as `live`; deterministic runs record `scripted`
- for comparative reports, compare only reports from the same suite
- `.github/workflows/benchmark-live.yml` is the tracked manual/nightly path for these runs
