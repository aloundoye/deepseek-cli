use anyhow::{Result, anyhow};
use chrono::Utc;
use deepseek_agent::AgentEngine;
use deepseek_core::{AppConfig, EventKind, runtime_dir};
use deepseek_index::IndexService;
use deepseek_store::Store;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::context::*;
use crate::output::*;
use crate::{BenchmarkCmd, BenchmarkPublishParityArgs, BenchmarkRunMatrixArgs, ProfileArgs};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct BenchmarkCase {
    #[serde(default)]
    case_id: String,
    prompt: String,
    #[serde(default)]
    expected_keywords: Vec<String>,
    #[serde(default)]
    min_steps: Option<usize>,
    #[serde(default)]
    min_verification_steps: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct BenchmarkMatrixSpec {
    #[serde(default)]
    name: Option<String>,
    runs: Vec<BenchmarkMatrixRunSpec>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct BenchmarkMatrixRunSpec {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    pack: Option<String>,
    #[serde(default)]
    suite: Option<String>,
    #[serde(default)]
    cases: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub(crate) struct BenchmarkPublicCatalog {
    #[serde(default)]
    schema: Option<String>,
    #[serde(default)]
    packs: Vec<BenchmarkCatalogPack>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub(crate) struct BenchmarkCatalogPack {
    name: String,
    source: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    corpus_id: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct BenchmarkMetrics {
    agent: String,
    success_rate: f64,
    quality_rate: f64,
    p95_latency_ms: u64,
    executed_cases: u64,
    corpus_id: String,
    manifest_sha256: Option<String>,
    seed: Option<u64>,
    case_outcomes: HashMap<String, CaseOutcome>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CaseOutcome {
    ok: bool,
    quality_ok: bool,
    elapsed_ms: Option<u64>,
}

#[derive(Debug, Clone)]
pub(crate) struct MatrixPeerMetrics {
    agent: String,
    total_cases: u64,
    weighted_success_rate: f64,
    weighted_quality_rate: f64,
    worst_p95_latency_ms: u64,
    manifest_coverage: f64,
}

pub(crate) fn run_profile(cwd: &Path, args: ProfileArgs, json_mode: bool) -> Result<()> {
    let started = Instant::now();
    let cfg = AppConfig::ensure(cwd)?;
    let store = Store::new(cwd)?;
    let session = ensure_session_record(cwd, &store)?;
    let usage = store.usage_summary(Some(session.session_id), Some(24))?;
    let compactions = store.list_context_compactions(Some(session.session_id))?;
    let autopilot = store.load_latest_autopilot_run()?;
    let index = IndexService::new(cwd)?.status()?;
    let estimated_cost_usd = (usage.input_tokens as f64 / 1_000_000.0)
        * cfg.usage.cost_per_million_input
        + (usage.output_tokens as f64 / 1_000_000.0) * cfg.usage.cost_per_million_output;
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let profile_id = Uuid::now_v7();
    let summary = format!(
        "tokens={} cost_usd={:.6} compactions={} index_fresh={} autopilot={}",
        usage.input_tokens + usage.output_tokens,
        estimated_cost_usd,
        compactions.len(),
        index.as_ref().is_some_and(|m| m.fresh),
        autopilot
            .as_ref()
            .map(|run| run.status.as_str())
            .unwrap_or("none")
    );
    store.insert_profile_run(&deepseek_store::ProfileRunRecord {
        profile_id,
        session_id: session.session_id,
        summary: summary.clone(),
        elapsed_ms,
        created_at: Utc::now().to_rfc3339(),
    })?;
    append_control_event(
        cwd,
        EventKind::ProfileCapturedV1 {
            profile_id,
            summary: summary.clone(),
            elapsed_ms,
        },
    )?;

    let payload = json!({
        "profile_id": profile_id,
        "session_id": session.session_id,
        "summary": summary,
        "elapsed_ms": elapsed_ms,
        "usage": usage,
        "estimated_cost_usd": estimated_cost_usd,
        "compactions": compactions.len(),
        "autopilot": autopilot.map(|run| json!({
            "run_id": run.run_id,
            "status": run.status,
            "completed_iterations": run.completed_iterations,
            "failed_iterations": run.failed_iterations,
        })),
        "index": index,
    });

    let payload = if args.benchmark {
        ensure_llm_ready_with_cfg(Some(cwd), &cfg, json_mode)?;
        let engine = AgentEngine::new(cwd)?;
        let suite_path = args.benchmark_suite.as_deref().map(Path::new);
        if suite_path.is_some() && args.benchmark_pack.is_some() {
            return Err(anyhow!(
                "use either --benchmark-suite or --benchmark-pack, not both"
            ));
        }
        let bench = run_profile_benchmark(
            &engine,
            args.benchmark_cases.max(1),
            args.benchmark_seed,
            suite_path,
            args.benchmark_pack.as_deref(),
            cwd,
            &args.benchmark_signing_key_env,
        );
        match bench {
            Ok(bench) => {
                if let Some(path) = args.benchmark_output.as_deref() {
                    let output_path = PathBuf::from(path);
                    if let Some(parent) = output_path.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    fs::write(&output_path, serde_json::to_vec_pretty(&bench)?)?;
                }
                if let Some(min_success_rate) = args.benchmark_min_success_rate {
                    let executed = bench
                        .get("executed_cases")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as f64;
                    let succeeded =
                        bench.get("succeeded").and_then(|v| v.as_u64()).unwrap_or(0) as f64;
                    let success_rate = if executed <= f64::EPSILON {
                        0.0
                    } else {
                        succeeded / executed
                    };
                    if success_rate < min_success_rate {
                        return Err(anyhow!(
                            "benchmark success rate {:.3} below required minimum {:.3}",
                            success_rate,
                            min_success_rate
                        ));
                    }
                }
                if let Some(max_p95_ms) = args.benchmark_max_p95_ms {
                    let p95 = bench
                        .get("p95_latency_ms")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(u64::MAX);
                    if p95 > max_p95_ms {
                        return Err(anyhow!(
                            "benchmark p95 latency {}ms above allowed maximum {}ms",
                            p95,
                            max_p95_ms
                        ));
                    }
                }
                if let Some(min_quality_rate) = args.benchmark_min_quality_rate {
                    let quality_rate = benchmark_quality_rate(&bench);
                    if quality_rate < min_quality_rate {
                        return Err(anyhow!(
                            "benchmark quality rate {:.3} below required minimum {:.3}",
                            quality_rate,
                            min_quality_rate
                        ));
                    }
                }

                let baseline_comparison = if let Some(path) = args.benchmark_baseline.as_deref() {
                    let baseline_raw = fs::read_to_string(path)?;
                    let baseline_value: serde_json::Value = serde_json::from_str(&baseline_raw)?;
                    let baseline_bench = baseline_value.get("benchmark").unwrap_or(&baseline_value);
                    let comparison = compare_benchmark_runs(&bench, baseline_bench)?;
                    if let Some(max_regression_ms) = args.benchmark_max_regression_ms
                        && comparison
                            .get("p95_regression_ms")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0)
                            > max_regression_ms
                    {
                        return Err(anyhow!(
                            "benchmark p95 regression {}ms above allowed maximum {}ms",
                            comparison
                                .get("p95_regression_ms")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0),
                            max_regression_ms
                        ));
                    }
                    Some(comparison)
                } else {
                    None
                };
                let peer_comparison = if args.benchmark_compare.is_empty() {
                    None
                } else {
                    Some(compare_benchmark_with_peers(
                        &bench,
                        &args.benchmark_compare,
                    )?)
                };
                if args.benchmark_compare_strict
                    && let Some(comparison) = peer_comparison.as_ref()
                {
                    let corpus_warnings = comparison
                        .get("corpus_match_warnings")
                        .and_then(|v| v.as_array())
                        .map(|rows| rows.len())
                        .unwrap_or(0);
                    let manifest_warnings = comparison
                        .get("manifest_match_warnings")
                        .and_then(|v| v.as_array())
                        .map(|rows| rows.len())
                        .unwrap_or(0);
                    let seed_warnings = comparison
                        .get("seed_match_warnings")
                        .and_then(|v| v.as_array())
                        .map(|rows| rows.len())
                        .unwrap_or(0);
                    if corpus_warnings + manifest_warnings + seed_warnings > 0 {
                        return Err(anyhow!(
                            "benchmark compare strict mode failed: corpus_warnings={} manifest_warnings={} seed_warnings={}",
                            corpus_warnings,
                            manifest_warnings,
                            seed_warnings
                        ));
                    }
                }

                let mut object = payload.as_object().cloned().unwrap_or_default();
                object.insert("benchmark".to_string(), bench.clone());
                if let Some(comparison) = baseline_comparison {
                    object.insert("benchmark_comparison".to_string(), comparison);
                }
                if let Some(comparison) = peer_comparison {
                    object.insert("benchmark_peer_comparison".to_string(), comparison);
                }
                serde_json::Value::Object(object)
            }
            Err(err) => {
                let mut object = payload.as_object().cloned().unwrap_or_default();
                object.insert(
                    "benchmark".to_string(),
                    json!({
                        "ok": false,
                        "error": err.to_string(),
                    }),
                );
                serde_json::Value::Object(object)
            }
        }
    } else {
        payload
    };

    if json_mode {
        print_json(&payload)?;
    } else {
        println!(
            "profile={} elapsed_ms={} cost_usd={:.6} tokens={} compactions={}",
            profile_id,
            elapsed_ms,
            estimated_cost_usd,
            usage.input_tokens + usage.output_tokens,
            compactions.len()
        );
    }
    Ok(())
}

pub(crate) fn built_in_benchmark_cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase {
            case_id: "refactor-plan".to_string(),
            prompt: "Plan a safe refactor for a router module with rollback strategy.".to_string(),
            expected_keywords: vec!["refactor".to_string(), "verify".to_string()],
            min_steps: Some(2),
            min_verification_steps: Some(1),
        },
        BenchmarkCase {
            case_id: "test-triage".to_string(),
            prompt: "Create a stepwise plan to investigate flaky tests and isolate nondeterminism."
                .to_string(),
            expected_keywords: vec!["test".to_string()],
            min_steps: Some(2),
            min_verification_steps: Some(1),
        },
        BenchmarkCase {
            case_id: "security-hardening".to_string(),
            prompt:
                "Plan command-injection hardening for shell execution with verification criteria."
                    .to_string(),
            expected_keywords: vec!["verify".to_string(), "bash".to_string()],
            min_steps: Some(2),
            min_verification_steps: Some(1),
        },
        BenchmarkCase {
            case_id: "git-recovery".to_string(),
            prompt: "Plan recovery from a failed merge with conflict-resolution checkpoints."
                .to_string(),
            expected_keywords: vec!["git".to_string()],
            min_steps: Some(2),
            min_verification_steps: Some(1),
        },
        BenchmarkCase {
            case_id: "index-performance".to_string(),
            prompt: "Plan improvements for index query latency and deterministic cache behavior."
                .to_string(),
            expected_keywords: vec!["index".to_string()],
            min_steps: Some(2),
            min_verification_steps: Some(1),
        },
        BenchmarkCase {
            case_id: "mcp-reliability".to_string(),
            prompt: "Plan reliability improvements for MCP tool discovery and reconnect flows."
                .to_string(),
            expected_keywords: vec!["mcp".to_string()],
            min_steps: Some(2),
            min_verification_steps: Some(1),
        },
        BenchmarkCase {
            case_id: "autopilot-ops".to_string(),
            prompt: "Plan operational guardrails for long-running autopilot loops.".to_string(),
            expected_keywords: vec!["autopilot".to_string()],
            min_steps: Some(2),
            min_verification_steps: Some(1),
        },
        BenchmarkCase {
            case_id: "release-readiness".to_string(),
            prompt: "Plan release readiness checks across tests, lint, and packaging artifacts."
                .to_string(),
            expected_keywords: vec!["verify".to_string()],
            min_steps: Some(2),
            min_verification_steps: Some(1),
        },
    ]
}

pub(crate) fn built_in_parity_benchmark_cases() -> Vec<BenchmarkCase> {
    let mut cases = built_in_benchmark_cases();
    let extra = vec![
        (
            "frontend-a11y",
            "Plan accessibility hardening for keyboard navigation and focus traps.",
            vec!["verify", "test"],
        ),
        (
            "backend-migration",
            "Plan a staged database migration with rollback checkpoints and data validation.",
            vec!["verify", "checkpoint"],
        ),
        (
            "ci-flake-hunt",
            "Plan CI flake triage with deterministic replay instrumentation.",
            vec!["test", "verify"],
        ),
        (
            "incident-recovery",
            "Plan post-incident recovery actions with containment and follow-up verification.",
            vec!["verify", "rollback"],
        ),
        (
            "plugin-hardening",
            "Plan plugin sandbox hardening and hook failure isolation policies.",
            vec!["plugin", "verify"],
        ),
        (
            "docs-quality-gates",
            "Plan docs quality checks including link validation and changelog integrity.",
            vec!["verify", "docs"],
        ),
        (
            "remote-env-safety",
            "Plan remote environment safety checks before deployment commands.",
            vec!["remote", "verify"],
        ),
        (
            "tui-regression-pack",
            "Plan regression testing for TUI hotkeys and slash command parity.",
            vec!["tui", "test"],
        ),
        (
            "security-scan-gates",
            "Plan security scan gates and remediation prioritization for release branches.",
            vec!["security", "verify"],
        ),
        (
            "api-contract-evolution",
            "Plan API contract evolution with backward compatibility checkpoints.",
            vec!["verify", "api"],
        ),
        (
            "benchmark-governance",
            "Plan benchmark governance with manifest consistency and peer comparability controls.",
            vec!["benchmark", "manifest"],
        ),
        (
            "subagent-team-orchestration",
            "Plan dependency-aware subagent orchestration across frontend backend and testing teams.",
            vec!["subagent", "verify"],
        ),
    ];
    for (case_id, prompt, keywords) in extra {
        cases.push(BenchmarkCase {
            case_id: case_id.to_string(),
            prompt: prompt.to_string(),
            expected_keywords: keywords.into_iter().map(str::to_string).collect(),
            min_steps: Some(2),
            min_verification_steps: Some(1),
        });
    }
    cases
}

pub(crate) fn load_benchmark_cases(path: &Path) -> Result<Vec<BenchmarkCase>> {
    let raw = fs::read_to_string(path)?;
    parse_benchmark_cases(&raw, path.to_string_lossy().as_ref())
}

pub(crate) fn parse_benchmark_cases(raw: &str, source_hint: &str) -> Result<Vec<BenchmarkCase>> {
    let looks_like_jsonl = source_hint.to_ascii_lowercase().ends_with(".jsonl");
    let mut cases = if looks_like_jsonl {
        let mut parsed = Vec::new();
        for (line_no, line) in raw.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let case: BenchmarkCase = serde_json::from_str(trimmed).map_err(|err| {
                anyhow!(
                    "invalid benchmark case at {}:{}: {err}",
                    source_hint,
                    line_no + 1
                )
            })?;
            parsed.push(case);
        }
        parsed
    } else {
        let value: serde_json::Value = serde_json::from_str(raw)?;
        let items = if let Some(arr) = value.as_array() {
            arr.clone()
        } else if let Some(arr) = value.get("cases").and_then(|v| v.as_array()) {
            arr.clone()
        } else {
            return Err(anyhow!(
                "benchmark suite must be a JSON array or an object with `cases` array"
            ));
        };
        let mut parsed = Vec::new();
        for (idx, item) in items.into_iter().enumerate() {
            let case: BenchmarkCase = serde_json::from_value(item).map_err(|err| {
                anyhow!(
                    "invalid benchmark case at {}:{}: {err}",
                    source_hint,
                    idx + 1
                )
            })?;
            parsed.push(case);
        }
        parsed
    };

    for (idx, case) in cases.iter_mut().enumerate() {
        case.prompt = case.prompt.trim().to_string();
        case.expected_keywords = case
            .expected_keywords
            .iter()
            .map(|kw| kw.trim().to_ascii_lowercase())
            .filter(|kw| !kw.is_empty())
            .collect();
        if case.case_id.trim().is_empty() {
            case.case_id = format!("case-{}", idx + 1);
        }
    }
    cases.retain(|case| !case.prompt.is_empty());
    if cases.is_empty() {
        return Err(anyhow!("benchmark suite has no valid cases"));
    }
    Ok(cases)
}

pub(crate) fn is_remote_source(source: &str) -> bool {
    source.starts_with("http://") || source.starts_with("https://")
}

pub(crate) fn resolve_local_source_path(cwd: &Path, source: &str) -> PathBuf {
    let raw = source.trim();
    if let Some(path) = raw.strip_prefix("file://") {
        return PathBuf::from(path);
    }
    let candidate = PathBuf::from(raw);
    if candidate.is_absolute() {
        candidate
    } else {
        cwd.join(candidate)
    }
}

pub(crate) fn load_benchmark_cases_from_source(
    cwd: &Path,
    source: &str,
) -> Result<Vec<BenchmarkCase>> {
    if is_remote_source(source) {
        let client = Client::builder().timeout(Duration::from_secs(30)).build()?;
        let raw = client.get(source).send()?.error_for_status()?.text()?;
        return parse_benchmark_cases(&raw, source);
    }
    let path = resolve_local_source_path(cwd, source);
    load_benchmark_cases(&path)
}

pub(crate) fn built_in_benchmark_pack(name: &str) -> Option<(&'static str, Vec<BenchmarkCase>)> {
    match name.to_ascii_lowercase().as_str() {
        "core" => Some(("builtin-core", built_in_benchmark_cases())),
        "smoke" => Some((
            "builtin-smoke",
            built_in_benchmark_cases().into_iter().take(3).collect(),
        )),
        "ops" => Some((
            "builtin-ops",
            built_in_benchmark_cases()
                .into_iter()
                .filter(|case| {
                    matches!(
                        case.case_id.as_str(),
                        "mcp-reliability" | "autopilot-ops" | "release-readiness"
                    )
                })
                .collect(),
        )),
        "parity" => Some(("builtin-parity", built_in_parity_benchmark_cases())),
        _ => None,
    }
}

pub(crate) fn benchmark_pack_dir(cwd: &Path) -> PathBuf {
    runtime_dir(cwd).join("benchmark-packs")
}

pub(crate) fn sanitize_pack_name(name: &str) -> String {
    let sanitized = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string();
    if sanitized.is_empty() {
        "pack".to_string()
    } else {
        sanitized
    }
}

pub(crate) fn local_benchmark_pack_path(cwd: &Path, name: &str) -> PathBuf {
    benchmark_pack_dir(cwd).join(format!("{}.json", sanitize_pack_name(name)))
}

pub(crate) fn load_benchmark_pack(cwd: &Path, name: &str) -> Result<serde_json::Value> {
    if let Some((corpus_id, cases)) = built_in_benchmark_pack(name) {
        return Ok(json!({
            "name": name,
            "kind": "builtin",
            "source": corpus_id,
            "cases": cases,
        }));
    }
    let path = local_benchmark_pack_path(cwd, name);
    if !path.exists() {
        return Err(anyhow!("benchmark pack not found: {}", name));
    }
    let raw = fs::read_to_string(&path)?;
    let mut value: serde_json::Value = serde_json::from_str(&raw)?;
    if value.get("name").is_none() {
        value["name"] = json!(name);
    }
    if value.get("kind").is_none() {
        value["kind"] = json!("imported");
    }
    if value.get("source").is_none() {
        value["source"] = json!(path.display().to_string());
    }
    Ok(value)
}

pub(crate) fn load_benchmark_pack_cases(
    cwd: &Path,
    name: &str,
) -> Result<(String, Vec<BenchmarkCase>)> {
    if let Some((corpus_id, cases)) = built_in_benchmark_pack(name) {
        return Ok((corpus_id.to_string(), cases));
    }
    let pack = load_benchmark_pack(cwd, name)?;
    let cases_value = pack
        .get("cases")
        .and_then(|v| v.as_array())
        .cloned()
        .ok_or_else(|| anyhow!("benchmark pack {} missing cases array", name))?;
    let mut cases = Vec::new();
    for (idx, item) in cases_value.into_iter().enumerate() {
        let case: BenchmarkCase = serde_json::from_value(item).map_err(|err| {
            anyhow!(
                "invalid benchmark case in pack {} at index {}: {}",
                name,
                idx + 1,
                err
            )
        })?;
        cases.push(case);
    }
    let corpus_id = pack
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or(name)
        .to_string();
    Ok((corpus_id, cases))
}

pub(crate) fn list_benchmark_packs(cwd: &Path) -> Result<Vec<serde_json::Value>> {
    let mut out = Vec::new();
    for (name, source) in [
        ("core", "builtin-core"),
        ("smoke", "builtin-smoke"),
        ("ops", "builtin-ops"),
        ("parity", "builtin-parity"),
    ] {
        let count = built_in_benchmark_pack(name)
            .map(|(_, cases)| cases.len())
            .unwrap_or(0);
        out.push(json!({
            "name": name,
            "kind": "builtin",
            "source": source,
            "cases": count,
        }));
    }

    let dir = benchmark_pack_dir(cwd);
    if dir.exists() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            let raw = fs::read_to_string(&path)?;
            let value: serde_json::Value = serde_json::from_str(&raw).unwrap_or_default();
            let count = value
                .get("cases")
                .and_then(|v| v.as_array())
                .map(|rows| rows.len())
                .unwrap_or(0);
            let source = value
                .get("source")
                .and_then(|v| v.as_str())
                .map(ToString::to_string)
                .unwrap_or_else(|| path.to_string_lossy().to_string());
            out.push(json!({
                "name": value.get("name").and_then(|v| v.as_str()).unwrap_or(stem),
                "kind": value.get("kind").and_then(|v| v.as_str()).unwrap_or("imported"),
                "source": source,
                "cases": count,
            }));
        }
    }
    out.sort_by(|a, b| {
        a["name"]
            .as_str()
            .unwrap_or_default()
            .cmp(b["name"].as_str().unwrap_or_default())
    });
    Ok(out)
}

pub(crate) fn write_imported_benchmark_pack(
    cwd: &Path,
    name: &str,
    source: &str,
    kind: &str,
    cases: Vec<BenchmarkCase>,
    mut metadata: serde_json::Value,
) -> Result<(PathBuf, serde_json::Value)> {
    let dir = benchmark_pack_dir(cwd);
    fs::create_dir_all(&dir)?;
    let destination = dir.join(format!("{}.json", sanitize_pack_name(name)));
    let mut payload = json!({
        "name": name,
        "kind": kind,
        "source": source,
        "imported_at": Utc::now().to_rfc3339(),
        "cases": cases,
    });
    if let (Some(obj), Some(extra)) = (payload.as_object_mut(), metadata.as_object_mut()) {
        for (key, value) in std::mem::take(extra) {
            obj.insert(key, value);
        }
    }
    fs::write(&destination, serde_json::to_vec_pretty(&payload)?)?;
    Ok((destination, payload))
}

pub(crate) fn parse_public_benchmark_catalog(raw: &str) -> Result<BenchmarkPublicCatalog> {
    let value: serde_json::Value = serde_json::from_str(raw)?;
    if let Some(packs) = value.as_array() {
        let parsed = packs
            .iter()
            .cloned()
            .map(serde_json::from_value::<BenchmarkCatalogPack>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        return Ok(BenchmarkPublicCatalog {
            schema: Some("deepseek.benchmark.catalog.v1".to_string()),
            packs: parsed,
        });
    }
    let mut catalog: BenchmarkPublicCatalog = serde_json::from_value(value)?;
    if catalog.schema.is_none() {
        catalog.schema = Some("deepseek.benchmark.catalog.v1".to_string());
    }
    Ok(catalog)
}

pub(crate) fn resolve_catalog_source(
    cwd: &Path,
    catalog_source: &str,
    entry_source: &str,
) -> String {
    if is_remote_source(entry_source) {
        return entry_source.to_string();
    }
    let candidate = PathBuf::from(entry_source);
    if candidate.is_absolute() {
        return candidate.to_string_lossy().to_string();
    }
    if is_remote_source(catalog_source)
        && let Ok(base) = reqwest::Url::parse(catalog_source)
        && let Ok(joined) = base.join(entry_source)
    {
        return joined.to_string();
    }
    let catalog_path = resolve_local_source_path(cwd, catalog_source);
    let parent = catalog_path.parent().unwrap_or(cwd);
    parent.join(entry_source).to_string_lossy().to_string()
}

pub(crate) fn default_parity_matrix_spec() -> serde_json::Value {
    json!({
        "name": "parity-publication",
        "runs": [
            {"id": "parity-pack", "pack": "parity", "cases": 8, "seed": 211},
            {"id": "ops-pack", "pack": "ops", "cases": 3, "seed": 223},
            {"id": "smoke-pack", "pack": "smoke", "cases": 2, "seed": 227}
        ]
    })
}

pub(crate) fn ensure_parity_matrix_file(path: &Path) -> Result<bool> {
    if path.exists() {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        path,
        serde_json::to_vec_pretty(&default_parity_matrix_spec())?,
    )?;
    Ok(true)
}

pub(crate) fn run_profile_benchmark(
    engine: &AgentEngine,
    requested_cases: usize,
    benchmark_seed: Option<u64>,
    suite_path: Option<&Path>,
    pack_name: Option<&str>,
    cwd: &Path,
    signing_key_env: &str,
) -> Result<serde_json::Value> {
    let (source_kind, cases, corpus_id) = if let Some(path) = suite_path {
        (
            "suite".to_string(),
            load_benchmark_cases(path)?,
            path.display().to_string(),
        )
    } else if let Some(pack_name) = pack_name {
        let (corpus_id, cases) = load_benchmark_pack_cases(cwd, pack_name)?;
        (format!("pack:{pack_name}"), cases, corpus_id)
    } else {
        (
            "builtin".to_string(),
            built_in_benchmark_cases(),
            "builtin".to_string(),
        )
    };

    if cases.is_empty() {
        return Err(anyhow!("benchmark suite is empty"));
    }

    let total = requested_cases.min(cases.len()).max(1);
    let seed =
        benchmark_seed.unwrap_or_else(|| derive_benchmark_seed(&corpus_id, requested_cases, total));
    let corpus_manifest =
        build_benchmark_manifest(&corpus_id, &source_kind, &cases, seed, signing_key_env);
    let mut selected_cases = select_benchmark_cases(cases, total, seed);
    let selected_case_ids = selected_cases
        .iter()
        .map(|case| case.case_id.clone())
        .collect::<Vec<_>>();
    let mut records = Vec::new();
    let mut latencies = Vec::new();
    let mut succeeded = 0usize;

    for case in selected_cases.drain(..) {
        let started = Instant::now();
        #[allow(deprecated)]
        let result = engine.plan_only(&case.prompt);
        let elapsed_ms = started.elapsed().as_millis() as u64;
        latencies.push(elapsed_ms);
        match result {
            Ok(plan) => {
                let plan_text = serde_json::to_string(&plan)
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                let keyword_total = case.expected_keywords.len();
                let keyword_matches = case
                    .expected_keywords
                    .iter()
                    .filter(|kw| plan_text.contains(kw.as_str()))
                    .count();
                let min_steps = case.min_steps.unwrap_or(1);
                let min_verification_steps = case.min_verification_steps.unwrap_or(1);
                let min_steps_ok = plan.steps.len() >= min_steps;
                let min_verification_ok = plan.verification.len() >= min_verification_steps;
                let keywords_ok = keyword_total == 0 || keyword_matches == keyword_total;
                let case_ok = min_steps_ok && min_verification_ok && keywords_ok;
                if case_ok {
                    succeeded += 1;
                }
                records.push(json!({
                    "case_id": case.case_id,
                    "prompt_sha256": sha256_hex(case.prompt.as_bytes()),
                    "ok": case_ok,
                    "elapsed_ms": elapsed_ms,
                    "steps": plan.steps.len(),
                    "verification_steps": plan.verification.len(),
                    "min_steps_required": min_steps,
                    "min_verification_required": min_verification_steps,
                    "keyword_matches": keyword_matches,
                    "keyword_total": keyword_total,
                    "keywords_ok": keywords_ok,
                    "quality_ok": case_ok,
                }));
            }
            Err(err) => {
                records.push(json!({
                    "case_id": case.case_id,
                    "prompt_sha256": sha256_hex(case.prompt.as_bytes()),
                    "ok": false,
                    "elapsed_ms": elapsed_ms,
                    "error": err.to_string(),
                }));
            }
        }
    }

    let avg_ms = if latencies.is_empty() {
        0
    } else {
        latencies.iter().sum::<u64>() / (latencies.len() as u64)
    };
    latencies.sort_unstable();
    let p95_ms = if latencies.is_empty() {
        0
    } else {
        let idx = ((latencies.len() - 1) as f64 * 0.95).round() as usize;
        latencies[idx.min(latencies.len() - 1)]
    };
    let quality_passed = records
        .iter()
        .filter(|record| {
            if let Some(ok) = record.get("quality_ok").and_then(|v| v.as_bool()) {
                return ok;
            }
            record.get("ok").and_then(|v| v.as_bool()).unwrap_or(false)
        })
        .count();
    let quality_rate = if total == 0 {
        0.0
    } else {
        quality_passed as f64 / total as f64
    };
    let corpus_manifest_sha = corpus_manifest
        .get("manifest_sha256")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let execution_manifest = build_benchmark_execution_manifest(
        &corpus_manifest_sha,
        &selected_case_ids,
        seed,
        signing_key_env,
    );
    let scorecard = build_benchmark_scorecard(
        "deepseek-cli",
        &corpus_manifest_sha,
        &selected_case_ids,
        seed,
        succeeded as u64,
        total as u64,
        quality_rate,
        avg_ms,
        p95_ms,
        signing_key_env,
    );

    Ok(json!({
        "agent": "deepseek-cli",
        "generated_at": Utc::now().to_rfc3339(),
        "ok": succeeded == total,
        "suite": suite_path
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| source_kind.clone()),
        "pack": pack_name,
        "corpus_id": corpus_id,
        "seed": seed,
        "case_ids": selected_case_ids,
        "requested_cases": requested_cases,
        "executed_cases": total,
        "succeeded": succeeded,
        "failed": total - succeeded,
        "avg_latency_ms": avg_ms,
        "p95_latency_ms": p95_ms,
        "quality_rate": quality_rate,
        "corpus_manifest": corpus_manifest,
        "execution_manifest": execution_manifest,
        "scorecard": scorecard,
        "records": records,
    }))
}

pub(crate) fn derive_benchmark_seed(
    corpus_id: &str,
    requested_cases: usize,
    executed_cases: usize,
) -> u64 {
    let digest = sha256_hex(
        format!(
            "{}|requested:{}|executed:{}",
            corpus_id, requested_cases, executed_cases
        )
        .as_bytes(),
    );
    u64::from_str_radix(&digest[..16], 16).unwrap_or(0)
}

pub(crate) fn select_benchmark_cases(
    mut cases: Vec<BenchmarkCase>,
    total: usize,
    seed: u64,
) -> Vec<BenchmarkCase> {
    cases.sort_by(|a, b| {
        benchmark_case_rank(seed, a)
            .cmp(&benchmark_case_rank(seed, b))
            .then(a.case_id.cmp(&b.case_id))
    });
    cases.into_iter().take(total).collect()
}

pub(crate) fn benchmark_case_rank(seed: u64, case: &BenchmarkCase) -> String {
    sha256_hex(format!("{seed}|{}|{}", case.case_id, case.prompt).as_bytes())
}

pub(crate) fn build_benchmark_manifest(
    corpus_id: &str,
    source_kind: &str,
    cases: &[BenchmarkCase],
    seed: u64,
    signing_key_env: &str,
) -> serde_json::Value {
    let mut case_rows = cases
        .iter()
        .map(|case| {
            let mut keywords = case
                .expected_keywords
                .iter()
                .map(|kw| kw.trim().to_ascii_lowercase())
                .filter(|kw| !kw.is_empty())
                .collect::<Vec<_>>();
            keywords.sort();
            keywords.dedup();
            let constraints = json!({
                "expected_keywords": keywords,
                "min_steps": case.min_steps.unwrap_or(1),
                "min_verification_steps": case.min_verification_steps.unwrap_or(1),
            });
            json!({
                "case_id": case.case_id,
                "prompt_sha256": sha256_hex(case.prompt.as_bytes()),
                "constraints_sha256": hash_json_value(&constraints),
            })
        })
        .collect::<Vec<_>>();
    case_rows.sort_by(|a, b| {
        a.get("case_id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .cmp(
                b.get("case_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default(),
            )
    });
    let mut manifest = json!({
        "schema": "deepseek.benchmark.manifest.v1",
        "corpus_id": corpus_id,
        "source_kind": source_kind,
        "seed": seed,
        "case_count": case_rows.len(),
        "cases": case_rows,
    });
    let manifest_sha256 = hash_json_value(&manifest);
    let signature = sign_benchmark_hash(&manifest_sha256, signing_key_env);
    if let Some(object) = manifest.as_object_mut() {
        object.insert("manifest_sha256".to_string(), json!(manifest_sha256));
        object.insert("signature".to_string(), signature);
    }
    manifest
}

pub(crate) fn build_benchmark_execution_manifest(
    corpus_manifest_sha256: &str,
    case_ids: &[String],
    seed: u64,
    signing_key_env: &str,
) -> serde_json::Value {
    let mut manifest = json!({
        "schema": "deepseek.benchmark.execution.v1",
        "seed": seed,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "case_ids": case_ids,
        "case_count": case_ids.len(),
    });
    let manifest_sha256 = hash_json_value(&manifest);
    let signature = sign_benchmark_hash(&manifest_sha256, signing_key_env);
    if let Some(object) = manifest.as_object_mut() {
        object.insert("manifest_sha256".to_string(), json!(manifest_sha256));
        object.insert("signature".to_string(), signature);
    }
    manifest
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_benchmark_scorecard(
    agent: &str,
    corpus_manifest_sha256: &str,
    case_ids: &[String],
    seed: u64,
    succeeded: u64,
    executed_cases: u64,
    quality_rate: f64,
    avg_latency_ms: u64,
    p95_latency_ms: u64,
    signing_key_env: &str,
) -> serde_json::Value {
    let mut scorecard = json!({
        "schema": "deepseek.benchmark.scorecard.v1",
        "agent": agent,
        "seed": seed,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "case_ids": case_ids,
        "executed_cases": executed_cases,
        "succeeded": succeeded,
        "success_rate": if executed_cases == 0 { 0.0 } else { succeeded as f64 / executed_cases as f64 },
        "quality_rate": quality_rate,
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
    });
    let scorecard_sha256 = hash_json_value(&scorecard);
    let signature = sign_benchmark_hash(&scorecard_sha256, signing_key_env);
    if let Some(object) = scorecard.as_object_mut() {
        object.insert("scorecard_sha256".to_string(), json!(scorecard_sha256));
        object.insert("signature".to_string(), signature);
    }
    scorecard
}

pub(crate) fn hash_json_value(value: &serde_json::Value) -> String {
    let canonical = canonicalize_json(value);
    let rendered = serde_json::to_string(&canonical).unwrap_or_default();
    sha256_hex(rendered.as_bytes())
}

pub(crate) fn canonicalize_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys = map.keys().cloned().collect::<Vec<_>>();
            keys.sort();
            let mut ordered = serde_json::Map::new();
            for key in keys {
                if let Some(item) = map.get(&key) {
                    ordered.insert(key, canonicalize_json(item));
                }
            }
            serde_json::Value::Object(ordered)
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(canonicalize_json).collect())
        }
        _ => value.clone(),
    }
}

pub(crate) fn sign_benchmark_hash(hash: &str, signing_key_env: &str) -> serde_json::Value {
    let key_env = signing_key_env.trim();
    if key_env.is_empty() {
        return json!({
            "algorithm": "none",
            "key_env": "",
            "present": false,
        });
    }
    let secret = std::env::var(key_env)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    if let Some(secret) = secret {
        let digest = sha256_hex(format!("{secret}:{hash}").as_bytes());
        return json!({
            "algorithm": "sha256-keyed",
            "key_env": key_env,
            "present": true,
            "digest": digest,
        });
    }
    json!({
        "algorithm": "none",
        "key_env": key_env,
        "present": false,
    })
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub(crate) fn benchmark_quality_rate(bench: &serde_json::Value) -> f64 {
    let records = bench
        .get("records")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if records.is_empty() {
        return 0.0;
    }
    let passed = records
        .iter()
        .filter(|record| {
            if let Some(ok) = record.get("quality_ok").and_then(|v| v.as_bool()) {
                return ok;
            }
            record.get("ok").and_then(|v| v.as_bool()).unwrap_or(false)
        })
        .count() as f64;
    passed / records.len() as f64
}

pub(crate) fn compare_benchmark_runs(
    current: &serde_json::Value,
    baseline: &serde_json::Value,
) -> Result<serde_json::Value> {
    let current_p95 = current
        .get("p95_latency_ms")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("current benchmark missing p95_latency_ms"))?;
    let baseline_p95 = baseline
        .get("p95_latency_ms")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("baseline benchmark missing p95_latency_ms"))?;
    let current_success = current
        .get("succeeded")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let current_total = current
        .get("executed_cases")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let baseline_success = baseline
        .get("succeeded")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let baseline_total = baseline
        .get("executed_cases")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let current_quality = benchmark_quality_rate(current);
    let baseline_quality = benchmark_quality_rate(baseline);
    Ok(json!({
        "baseline_p95_latency_ms": baseline_p95,
        "current_p95_latency_ms": current_p95,
        "p95_regression_ms": current_p95.saturating_sub(baseline_p95),
        "p95_improvement_ms": baseline_p95.saturating_sub(current_p95),
        "baseline_success_rate": if baseline_total == 0 { 0.0 } else { baseline_success as f64 / baseline_total as f64 },
        "current_success_rate": if current_total == 0 { 0.0 } else { current_success as f64 / current_total as f64 },
        "baseline_quality_rate": baseline_quality,
        "current_quality_rate": current_quality,
        "quality_delta": current_quality - baseline_quality,
    }))
}

pub(crate) fn benchmark_metrics_from_value(
    value: &serde_json::Value,
    fallback_agent: &str,
) -> Result<BenchmarkMetrics> {
    let bench = value.get("benchmark").unwrap_or(value);
    let executed = bench
        .get("executed_cases")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("benchmark report missing executed_cases"))?;
    let succeeded = bench.get("succeeded").and_then(|v| v.as_u64()).unwrap_or(0);
    let p95_latency_ms = bench
        .get("p95_latency_ms")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("benchmark report missing p95_latency_ms"))?;
    let success_rate = if executed == 0 {
        0.0
    } else {
        succeeded as f64 / executed as f64
    };
    let quality_rate = benchmark_quality_rate(bench);
    let agent = value
        .get("agent")
        .and_then(|v| v.as_str())
        .or_else(|| bench.get("agent").and_then(|v| v.as_str()))
        .unwrap_or(fallback_agent)
        .to_string();
    let corpus_id = bench
        .get("corpus_id")
        .and_then(|v| v.as_str())
        .or_else(|| bench.get("suite").and_then(|v| v.as_str()))
        .unwrap_or("unknown")
        .to_string();
    let manifest_sha256 = bench
        .get("scorecard")
        .and_then(|v| v.get("corpus_manifest_sha256"))
        .and_then(|v| v.as_str())
        .or_else(|| {
            bench
                .get("corpus_manifest")
                .and_then(|v| v.get("manifest_sha256"))
                .and_then(|v| v.as_str())
        })
        .map(ToString::to_string);
    let seed = bench
        .get("scorecard")
        .and_then(|v| v.get("seed"))
        .and_then(|v| v.as_u64())
        .or_else(|| bench.get("seed").and_then(|v| v.as_u64()));
    let case_outcomes = benchmark_case_outcomes(bench);
    Ok(BenchmarkMetrics {
        agent,
        success_rate,
        quality_rate,
        p95_latency_ms,
        executed_cases: executed,
        corpus_id,
        manifest_sha256,
        seed,
        case_outcomes,
    })
}

pub(crate) fn benchmark_case_outcomes(bench: &serde_json::Value) -> HashMap<String, CaseOutcome> {
    let mut out = HashMap::new();
    for (idx, record) in bench
        .get("records")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .enumerate()
    {
        let case_id = record
            .get("case_id")
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("case-{}", idx + 1));
        let ok = record.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
        let quality_ok = record
            .get("quality_ok")
            .and_then(|v| v.as_bool())
            .unwrap_or(ok);
        let elapsed_ms = record.get("elapsed_ms").and_then(|v| v.as_u64());
        out.insert(
            case_id,
            CaseOutcome {
                ok,
                quality_ok,
                elapsed_ms,
            },
        );
    }
    out
}

pub(crate) fn compare_benchmark_with_peers(
    current_bench: &serde_json::Value,
    paths: &[String],
) -> Result<serde_json::Value> {
    let mut rows = Vec::new();
    rows.push(benchmark_metrics_from_value(
        &json!({"agent": "deepseek-cli", "benchmark": current_bench}),
        "deepseek-cli",
    )?);

    for path in paths {
        let raw = fs::read_to_string(path)?;
        let parsed: serde_json::Value = serde_json::from_str(&raw)?;
        let fallback_name = Path::new(path)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("peer");
        rows.push(benchmark_metrics_from_value(&parsed, fallback_name)?);
    }

    let mut corpus_warnings = Vec::new();
    let mut manifest_warnings = Vec::new();
    let mut seed_warnings = Vec::new();
    let canonical_corpus = rows
        .iter()
        .find(|row| row.agent == "deepseek-cli")
        .map(|row| row.corpus_id.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let canonical_manifest = rows
        .iter()
        .find(|row| row.agent == "deepseek-cli")
        .and_then(|row| row.manifest_sha256.clone());
    let canonical_seed = rows
        .iter()
        .find(|row| row.agent == "deepseek-cli")
        .and_then(|row| row.seed);
    for row in &rows {
        if row.corpus_id != canonical_corpus {
            corpus_warnings.push(format!(
                "agent {} corpus_id={} differs from deepseek-cli corpus_id={}",
                row.agent, row.corpus_id, canonical_corpus
            ));
        }
        if let Some(expected_manifest) = canonical_manifest.as_deref() {
            match row.manifest_sha256.as_deref() {
                Some(value) if value != expected_manifest => {
                    manifest_warnings.push(format!(
                        "agent {} manifest_sha256={} differs from deepseek-cli manifest_sha256={}",
                        row.agent, value, expected_manifest
                    ));
                }
                None => {
                    manifest_warnings.push(format!(
                        "agent {} missing manifest_sha256 for reproducible comparison",
                        row.agent
                    ));
                }
                _ => {}
            }
        } else if row.manifest_sha256.is_none() {
            manifest_warnings.push(format!(
                "agent {} missing manifest_sha256 for reproducible comparison",
                row.agent
            ));
        }
        if let Some(seed) = canonical_seed
            && row.seed.is_some_and(|value| value != seed)
        {
            seed_warnings.push(format!(
                "agent {} seed={} differs from deepseek-cli seed={}",
                row.agent,
                row.seed.unwrap_or_default(),
                seed
            ));
        }
    }

    let mut ranking = rows.clone();
    ranking.sort_by(|a, b| {
        b.quality_rate
            .partial_cmp(&a.quality_rate)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.success_rate
                    .partial_cmp(&a.success_rate)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(a.p95_latency_ms.cmp(&b.p95_latency_ms))
    });

    let current_rank = ranking
        .iter()
        .position(|row| row.agent == "deepseek-cli")
        .map(|idx| idx + 1)
        .unwrap_or(0);

    let mut case_ids = BTreeSet::new();
    for row in &rows {
        for case_id in row.case_outcomes.keys() {
            case_ids.insert(case_id.clone());
        }
    }
    let case_matrix = case_ids
        .into_iter()
        .map(|case_id| {
            let agents = rows
                .iter()
                .map(|row| {
                    let outcome = row.case_outcomes.get(&case_id).cloned();
                    json!({
                        "agent": row.agent.clone(),
                        "present": outcome.is_some(),
                        "ok": outcome.as_ref().map(|value| value.ok),
                        "quality_ok": outcome.as_ref().map(|value| value.quality_ok),
                        "elapsed_ms": outcome.and_then(|value| value.elapsed_ms),
                    })
                })
                .collect::<Vec<_>>();
            json!({
                "case_id": case_id,
                "agents": agents,
            })
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "current_rank": current_rank,
        "total_agents": ranking.len(),
        "corpus_id": canonical_corpus,
        "corpus_match_warnings": corpus_warnings,
        "ranking": ranking
            .into_iter()
            .map(|row| json!({
                "agent": row.agent,
                "quality_rate": row.quality_rate,
                "success_rate": row.success_rate,
                "p95_latency_ms": row.p95_latency_ms,
                "executed_cases": row.executed_cases,
                "corpus_id": row.corpus_id,
                "manifest_sha256": row.manifest_sha256,
                "seed": row.seed,
            }))
            .collect::<Vec<_>>(),
        "manifest_match_warnings": manifest_warnings,
        "seed_match_warnings": seed_warnings,
        "case_matrix": case_matrix,
    }))
}

pub(crate) fn run_benchmark_matrix(
    cwd: &Path,
    args: BenchmarkRunMatrixArgs,
    json_mode: bool,
) -> Result<()> {
    let payload = benchmark_matrix_payload(cwd, &args, json_mode)?;
    if let Some(output) = args.output.as_deref() {
        let output_path = PathBuf::from(output);
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(output_path, serde_json::to_vec_pretty(&payload)?)?;
    }
    if let Some(output) = args.report_output.as_deref() {
        let output_path = PathBuf::from(output);
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let report = payload
            .get("report_markdown")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        fs::write(output_path, report)?;
    }

    if json_mode {
        print_json(&payload)?;
    } else {
        println!(
            "benchmark matrix={} runs={} total_cases={} weighted_success_rate={:.3} weighted_quality_rate={:.3} worst_p95={}ms",
            payload["name"].as_str().unwrap_or_default(),
            payload["summary"]["total_runs"].as_u64().unwrap_or(0),
            payload["summary"]["total_cases"].as_u64().unwrap_or(0),
            payload["summary"]["weighted_success_rate"]
                .as_f64()
                .unwrap_or(0.0),
            payload["summary"]["weighted_quality_rate"]
                .as_f64()
                .unwrap_or(0.0),
            payload["summary"]["worst_p95_latency_ms"]
                .as_u64()
                .unwrap_or(0),
        );
    }
    Ok(())
}

pub(crate) fn benchmark_matrix_payload(
    cwd: &Path,
    args: &BenchmarkRunMatrixArgs,
    json_mode: bool,
) -> Result<serde_json::Value> {
    ensure_llm_ready(cwd, json_mode)?;
    let matrix_path = PathBuf::from(&args.matrix);
    let spec = load_benchmark_matrix_spec(&matrix_path)?;
    let matrix_name = spec.name.clone().unwrap_or_else(|| {
        matrix_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("benchmark-matrix")
            .to_string()
    });
    let matrix_dir = matrix_path.parent().unwrap_or(cwd);
    let engine = AgentEngine::new(cwd)?;

    let mut run_reports = Vec::new();
    for (idx, run) in spec.runs.iter().enumerate() {
        let run_id = run
            .id
            .clone()
            .filter(|id| !id.trim().is_empty())
            .unwrap_or_else(|| format!("run-{}", idx + 1));
        let cases = run.cases.unwrap_or(5).max(1);
        let suite_path = run
            .suite
            .as_deref()
            .map(|suite| resolve_matrix_suite_path(matrix_dir, suite));
        let benchmark = run_profile_benchmark(
            &engine,
            cases,
            run.seed,
            suite_path.as_deref(),
            run.pack.as_deref(),
            cwd,
            &args.signing_key_env,
        )?;
        run_reports.push(json!({
            "id": run_id,
            "pack": run.pack,
            "suite": run.suite,
            "cases": cases,
            "seed": benchmark.get("seed").and_then(|v| v.as_u64()).or(run.seed),
            "benchmark": benchmark,
        }));
    }

    let summary = aggregate_benchmark_matrix_results(&run_reports);
    if args.strict {
        let local_warnings = summary
            .get("compatibility_warnings")
            .and_then(|v| v.as_array())
            .map(|rows| rows.len())
            .unwrap_or(0);
        if local_warnings > 0 {
            return Err(anyhow!(
                "benchmark matrix strict mode failed: compatibility_warnings={}",
                local_warnings
            ));
        }
    }
    let peer_comparison = if args.compare.is_empty() {
        None
    } else {
        Some(compare_benchmark_matrix_with_peers(
            "deepseek-cli",
            &summary,
            &args.compare,
        )?)
    };
    if args.strict
        && let Some(peer) = peer_comparison.as_ref()
    {
        let coverage_warnings = peer
            .get("manifest_coverage_warnings")
            .and_then(|v| v.as_array())
            .map(|rows| rows.len())
            .unwrap_or(0);
        let case_warnings = peer
            .get("case_count_warnings")
            .and_then(|v| v.as_array())
            .map(|rows| rows.len())
            .unwrap_or(0);
        if coverage_warnings + case_warnings > 0 {
            return Err(anyhow!(
                "benchmark matrix strict mode failed: manifest_coverage_warnings={} case_count_warnings={}",
                coverage_warnings,
                case_warnings
            ));
        }
    }
    if !args.require_agent.is_empty() {
        let available_agents = collect_matrix_agents(peer_comparison.as_ref());
        let required = args
            .require_agent
            .iter()
            .map(|agent| agent.trim().to_ascii_lowercase())
            .filter(|agent| !agent.is_empty())
            .collect::<BTreeSet<_>>();
        let missing = required
            .iter()
            .filter(|agent| !available_agents.contains(*agent))
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(anyhow!(
                "benchmark matrix missing required agents: {} (available={})",
                missing.join(", "),
                available_agents.into_iter().collect::<Vec<_>>().join(", ")
            ));
        }
    }

    let mut payload = json!({
        "schema": "deepseek.benchmark.matrix.v1",
        "generated_at": Utc::now().to_rfc3339(),
        "agent": "deepseek-cli",
        "name": matrix_name,
        "source": matrix_path.display().to_string(),
        "runs": run_reports,
        "summary": summary,
    });
    if let Some(peer_comparison) = peer_comparison
        && let Some(object) = payload.as_object_mut()
    {
        object.insert("peer_comparison".to_string(), peer_comparison);
    }
    let report_markdown = render_benchmark_matrix_report(&payload);
    if let Some(object) = payload.as_object_mut() {
        object.insert("report_markdown".to_string(), json!(report_markdown));
    }
    Ok(payload)
}

pub(crate) fn load_benchmark_matrix_spec(path: &Path) -> Result<BenchmarkMatrixSpec> {
    let raw = fs::read_to_string(path)?;
    let mut spec: BenchmarkMatrixSpec = serde_json::from_str(&raw)?;
    if spec.runs.is_empty() {
        return Err(anyhow!("benchmark matrix has no runs"));
    }
    for (idx, run) in spec.runs.iter_mut().enumerate() {
        run.id = run
            .id
            .clone()
            .map(|id| id.trim().to_string())
            .filter(|id| !id.is_empty())
            .or_else(|| Some(format!("run-{}", idx + 1)));
        run.pack = run
            .pack
            .clone()
            .map(|pack| pack.trim().to_string())
            .filter(|pack| !pack.is_empty());
        run.suite = run
            .suite
            .clone()
            .map(|suite| suite.trim().to_string())
            .filter(|suite| !suite.is_empty());

        let has_pack = run.pack.is_some();
        let has_suite = run.suite.is_some();
        if has_pack == has_suite {
            return Err(anyhow!(
                "invalid matrix run {}: specify exactly one of `pack` or `suite`",
                idx + 1
            ));
        }
        run.cases = Some(run.cases.unwrap_or(5).max(1));
    }
    Ok(spec)
}

pub(crate) fn resolve_matrix_suite_path(base_dir: &Path, suite: &str) -> PathBuf {
    let candidate = PathBuf::from(suite);
    if candidate.is_absolute() {
        candidate
    } else {
        base_dir.join(candidate)
    }
}

pub(crate) fn aggregate_benchmark_matrix_results(
    run_reports: &[serde_json::Value],
) -> serde_json::Value {
    let mut total_cases = 0_u64;
    let mut total_succeeded = 0_u64;
    let mut weighted_quality_sum = 0.0_f64;
    let mut weighted_p95_sum = 0.0_f64;
    let mut worst_p95_latency_ms = 0_u64;
    let mut manifests = BTreeSet::new();
    let mut corpus_ids = BTreeSet::new();
    let mut seeds = BTreeSet::new();
    let mut signed_runs = 0_u64;
    let mut run_ids = Vec::new();

    for (idx, run) in run_reports.iter().enumerate() {
        let bench = run.get("benchmark").unwrap_or(run);
        let executed = bench
            .get("executed_cases")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let succeeded = bench.get("succeeded").and_then(|v| v.as_u64()).unwrap_or(0);
        let quality_rate = bench
            .get("quality_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or_else(|| benchmark_quality_rate(bench));
        let p95 = bench
            .get("p95_latency_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let run_id = run
            .get("id")
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("run-{}", idx + 1));
        run_ids.push(run_id);

        total_cases = total_cases.saturating_add(executed);
        total_succeeded = total_succeeded.saturating_add(succeeded);
        weighted_quality_sum += quality_rate * executed as f64;
        weighted_p95_sum += p95 as f64 * executed as f64;
        worst_p95_latency_ms = worst_p95_latency_ms.max(p95);

        if let Some(corpus_id) = bench.get("corpus_id").and_then(|v| v.as_str()) {
            corpus_ids.insert(corpus_id.to_string());
        }
        if let Some(seed) = bench.get("seed").and_then(|v| v.as_u64()) {
            seeds.insert(seed);
        }
        if let Some(manifest) = bench
            .get("scorecard")
            .and_then(|v| v.get("corpus_manifest_sha256"))
            .and_then(|v| v.as_str())
        {
            manifests.insert(manifest.to_string());
        }
        if bench
            .get("scorecard")
            .and_then(|v| v.get("signature"))
            .and_then(|v| v.get("present"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            signed_runs = signed_runs.saturating_add(1);
        }
    }

    let weighted_success_rate = if total_cases == 0 {
        0.0
    } else {
        total_succeeded as f64 / total_cases as f64
    };
    let weighted_quality_rate = if total_cases == 0 {
        0.0
    } else {
        weighted_quality_sum / total_cases as f64
    };
    let weighted_p95_latency_ms = if total_cases == 0 {
        0
    } else {
        (weighted_p95_sum / total_cases as f64).round() as u64
    };

    let mut compatibility_warnings = Vec::new();
    if manifests.len() > 1 {
        compatibility_warnings.push(format!(
            "matrix uses {} distinct corpus manifests; direct parity ranking is weaker",
            manifests.len()
        ));
    }
    if corpus_ids.len() > 1 {
        compatibility_warnings.push(format!(
            "matrix uses {} distinct corpus_ids",
            corpus_ids.len()
        ));
    }

    json!({
        "total_runs": run_reports.len(),
        "run_ids": run_ids,
        "total_cases": total_cases,
        "total_succeeded": total_succeeded,
        "total_failed": total_cases.saturating_sub(total_succeeded),
        "weighted_success_rate": weighted_success_rate,
        "weighted_quality_rate": weighted_quality_rate,
        "weighted_p95_latency_ms": weighted_p95_latency_ms,
        "worst_p95_latency_ms": worst_p95_latency_ms,
        "manifest_coverage": if run_reports.is_empty() { 0.0 } else { signed_runs as f64 / run_reports.len() as f64 },
        "manifest_sha256": manifests.into_iter().collect::<Vec<_>>(),
        "corpus_ids": corpus_ids.into_iter().collect::<Vec<_>>(),
        "seeds": seeds.into_iter().collect::<Vec<_>>(),
        "compatibility_warnings": compatibility_warnings,
    })
}

pub(crate) fn matrix_peer_metrics_from_value(
    value: &serde_json::Value,
    fallback_agent: &str,
) -> Result<MatrixPeerMetrics> {
    if let Some(summary) = value.get("summary") {
        let total_cases = summary
            .get("total_cases")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let weighted_success_rate = summary
            .get("weighted_success_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let weighted_quality_rate = summary
            .get("weighted_quality_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let worst_p95_latency_ms = summary
            .get("worst_p95_latency_ms")
            .or_else(|| summary.get("worst_p95_latency"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let manifest_coverage = summary
            .get("manifest_coverage")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let agent = value
            .get("agent")
            .and_then(|v| v.as_str())
            .or_else(|| summary.get("agent").and_then(|v| v.as_str()))
            .unwrap_or(fallback_agent)
            .to_string();
        return Ok(MatrixPeerMetrics {
            agent,
            total_cases,
            weighted_success_rate,
            weighted_quality_rate,
            worst_p95_latency_ms,
            manifest_coverage,
        });
    }

    let bench = benchmark_metrics_from_value(value, fallback_agent)?;
    Ok(MatrixPeerMetrics {
        agent: bench.agent,
        total_cases: bench.executed_cases,
        weighted_success_rate: bench.success_rate,
        weighted_quality_rate: bench.quality_rate,
        worst_p95_latency_ms: bench.p95_latency_ms,
        manifest_coverage: if bench.manifest_sha256.is_some() {
            1.0
        } else {
            0.0
        },
    })
}

pub(crate) fn compare_benchmark_matrix_with_peers(
    current_agent: &str,
    current_summary: &serde_json::Value,
    paths: &[String],
) -> Result<serde_json::Value> {
    let mut rows = Vec::new();
    rows.push(matrix_peer_metrics_from_value(
        &json!({"agent": current_agent, "summary": current_summary}),
        current_agent,
    )?);

    for path in paths {
        let raw = fs::read_to_string(path)?;
        let parsed: serde_json::Value = serde_json::from_str(&raw)?;
        let fallback = Path::new(path)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("peer");
        rows.push(matrix_peer_metrics_from_value(&parsed, fallback)?);
    }

    let canonical_cases = rows
        .iter()
        .find(|row| row.agent == current_agent)
        .map(|row| row.total_cases)
        .unwrap_or(0);
    let canonical_manifest_coverage = rows
        .iter()
        .find(|row| row.agent == current_agent)
        .map(|row| row.manifest_coverage)
        .unwrap_or(0.0);
    let mut coverage_warnings = Vec::new();
    let mut case_count_warnings = Vec::new();
    for row in &rows {
        if (row.manifest_coverage - canonical_manifest_coverage).abs() > 0.001 {
            coverage_warnings.push(format!(
                "agent {} manifest_coverage={:.3} differs from {} manifest_coverage={:.3}",
                row.agent, row.manifest_coverage, current_agent, canonical_manifest_coverage
            ));
        }
        if row.total_cases != canonical_cases {
            case_count_warnings.push(format!(
                "agent {} total_cases={} differs from {} total_cases={}",
                row.agent, row.total_cases, current_agent, canonical_cases
            ));
        }
    }

    let mut ranking = rows.clone();
    ranking.sort_by(|a, b| {
        b.weighted_quality_rate
            .partial_cmp(&a.weighted_quality_rate)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.weighted_success_rate
                    .partial_cmp(&a.weighted_success_rate)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(a.worst_p95_latency_ms.cmp(&b.worst_p95_latency_ms))
    });
    let current_rank = ranking
        .iter()
        .position(|row| row.agent == current_agent)
        .map(|idx| idx + 1)
        .unwrap_or(0);

    Ok(json!({
        "current_rank": current_rank,
        "total_agents": ranking.len(),
        "ranking": ranking
            .into_iter()
            .map(|row| json!({
                "agent": row.agent,
                "total_cases": row.total_cases,
                "weighted_success_rate": row.weighted_success_rate,
                "weighted_quality_rate": row.weighted_quality_rate,
                "worst_p95_latency_ms": row.worst_p95_latency_ms,
                "manifest_coverage": row.manifest_coverage,
            }))
            .collect::<Vec<_>>(),
        "manifest_coverage_warnings": coverage_warnings,
        "case_count_warnings": case_count_warnings,
    }))
}

pub(crate) fn collect_matrix_agents(
    peer_comparison: Option<&serde_json::Value>,
) -> BTreeSet<String> {
    let mut agents = BTreeSet::new();
    agents.insert("deepseek-cli".to_string());
    if let Some(peer) = peer_comparison {
        for row in peer
            .get("ranking")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
        {
            if let Some(agent) = row.get("agent").and_then(|v| v.as_str()) {
                agents.insert(agent.trim().to_ascii_lowercase());
            }
        }
    }
    agents
}

pub(crate) fn render_benchmark_matrix_report(payload: &serde_json::Value) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "# Benchmark Matrix Report: {}",
        payload["name"].as_str().unwrap_or("benchmark-matrix")
    ));
    lines.push(String::new());
    lines.push(format!(
        "- Generated at: {}",
        payload["generated_at"].as_str().unwrap_or_default()
    ));
    lines.push(format!(
        "- Source: {}",
        payload["source"].as_str().unwrap_or_default()
    ));
    lines.push(format!(
        "- Runs: {}",
        payload["summary"]["total_runs"].as_u64().unwrap_or(0)
    ));
    lines.push(format!(
        "- Cases: {}",
        payload["summary"]["total_cases"].as_u64().unwrap_or(0)
    ));
    lines.push(format!(
        "- Weighted success rate: {:.3}",
        payload["summary"]["weighted_success_rate"]
            .as_f64()
            .unwrap_or(0.0)
    ));
    lines.push(format!(
        "- Weighted quality rate: {:.3}",
        payload["summary"]["weighted_quality_rate"]
            .as_f64()
            .unwrap_or(0.0)
    ));
    lines.push(format!(
        "- Worst p95 latency: {} ms",
        payload["summary"]["worst_p95_latency_ms"]
            .as_u64()
            .unwrap_or(0)
    ));
    lines.push(String::new());

    lines.push("## Runs".to_string());
    lines.push(String::new());
    lines.push("| Run | Corpus | Cases | Success | Quality | p95 ms |".to_string());
    lines.push("|---|---:|---:|---:|---:|---:|".to_string());
    for run in payload["runs"].as_array().into_iter().flatten() {
        let bench = run.get("benchmark").unwrap_or(run);
        let corpus = bench
            .get("corpus_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let cases = bench
            .get("executed_cases")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let succeeded = bench.get("succeeded").and_then(|v| v.as_u64()).unwrap_or(0);
        let success_rate = if cases == 0 {
            0.0
        } else {
            succeeded as f64 / cases as f64
        };
        let quality_rate = bench
            .get("quality_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or_else(|| benchmark_quality_rate(bench));
        let p95 = bench
            .get("p95_latency_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        lines.push(format!(
            "| {} | {} | {} | {:.3} | {:.3} | {} |",
            run["id"].as_str().unwrap_or("run"),
            corpus,
            cases,
            success_rate,
            quality_rate,
            p95
        ));
    }
    lines.push(String::new());

    if let Some(peer) = payload.get("peer_comparison") {
        lines.push("## Peer Ranking".to_string());
        lines.push(String::new());
        lines.push(
            "| Rank | Agent | Cases | Weighted Success | Weighted Quality | Worst p95 ms |"
                .to_string(),
        );
        lines.push("|---:|---|---:|---:|---:|---:|".to_string());
        for (idx, row) in peer
            .get("ranking")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .enumerate()
        {
            lines.push(format!(
                "| {} | {} | {} | {:.3} | {:.3} | {} |",
                idx + 1,
                row["agent"].as_str().unwrap_or_default(),
                row["total_cases"].as_u64().unwrap_or(0),
                row["weighted_success_rate"].as_f64().unwrap_or(0.0),
                row["weighted_quality_rate"].as_f64().unwrap_or(0.0),
                row["worst_p95_latency_ms"].as_u64().unwrap_or(0),
            ));
        }
        let manifest_warnings = peer
            .get("manifest_coverage_warnings")
            .and_then(|v| v.as_array())
            .map(|rows| {
                rows.iter()
                    .filter_map(|row| row.as_str())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let case_warnings = peer
            .get("case_count_warnings")
            .and_then(|v| v.as_array())
            .map(|rows| {
                rows.iter()
                    .filter_map(|row| row.as_str())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if !manifest_warnings.is_empty() || !case_warnings.is_empty() {
            lines.push(String::new());
            lines.push("## Peer Compatibility Warnings".to_string());
            for warning in manifest_warnings {
                lines.push(format!("- {warning}"));
            }
            for warning in case_warnings {
                lines.push(format!("- {warning}"));
            }
        }
    }

    lines.join("\n")
}

pub(crate) fn run_benchmark(cwd: &Path, cmd: BenchmarkCmd, json_mode: bool) -> Result<()> {
    match cmd {
        BenchmarkCmd::ListPacks => {
            let packs = list_benchmark_packs(cwd)?;
            if json_mode {
                print_json(&json!(packs))?;
            } else if packs.is_empty() {
                println!("no benchmark packs found");
            } else {
                for pack in packs {
                    println!(
                        "{} ({}) cases={} source={}",
                        pack["name"].as_str().unwrap_or_default(),
                        pack["kind"].as_str().unwrap_or_default(),
                        pack["cases"].as_u64().unwrap_or(0),
                        pack["source"].as_str().unwrap_or_default(),
                    );
                }
            }
        }
        BenchmarkCmd::ShowPack(args) => {
            let pack = load_benchmark_pack(cwd, &args.name)?;
            if json_mode {
                print_json(&pack)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&pack)?);
            }
        }
        BenchmarkCmd::ImportPack(args) => {
            let cases = load_benchmark_cases_from_source(cwd, &args.source)?;
            let (destination, payload) = write_imported_benchmark_pack(
                cwd,
                &args.name,
                &args.source,
                "imported",
                cases,
                json!({}),
            )?;
            if json_mode {
                print_json(&json!({
                    "imported": true,
                    "name": payload["name"],
                    "cases": payload["cases"].as_array().map(|rows| rows.len()).unwrap_or(0),
                    "path": destination,
                }))?;
            } else {
                println!(
                    "imported benchmark pack {} with {} cases",
                    payload["name"].as_str().unwrap_or_default(),
                    payload["cases"]
                        .as_array()
                        .map(|rows| rows.len())
                        .unwrap_or(0),
                );
            }
        }
        BenchmarkCmd::SyncPublic(args) => {
            let raw = if is_remote_source(&args.catalog) {
                let client = Client::builder().timeout(Duration::from_secs(30)).build()?;
                client
                    .get(&args.catalog)
                    .send()?
                    .error_for_status()?
                    .text()?
            } else {
                let catalog_path = resolve_local_source_path(cwd, &args.catalog);
                fs::read_to_string(catalog_path)?
            };
            let catalog = parse_public_benchmark_catalog(&raw)?;
            let catalog_schema = catalog.schema.clone();
            let selected = args
                .only
                .iter()
                .map(|name| name.trim().to_ascii_lowercase())
                .filter(|name| !name.is_empty())
                .collect::<HashSet<_>>();
            let mut imported = Vec::new();
            for entry in catalog.packs {
                if entry.name.trim().is_empty() || entry.source.trim().is_empty() {
                    continue;
                }
                if !selected.is_empty() && !selected.contains(&entry.name.to_ascii_lowercase()) {
                    continue;
                }
                let target_name = args.prefix.as_deref().map_or_else(
                    || entry.name.clone(),
                    |prefix| format!("{}{}", prefix, entry.name),
                );
                let resolved_source = resolve_catalog_source(cwd, &args.catalog, &entry.source);
                let cases = load_benchmark_cases_from_source(cwd, &resolved_source)?;
                let (path, payload) = write_imported_benchmark_pack(
                    cwd,
                    &target_name,
                    &resolved_source,
                    entry.kind.as_deref().unwrap_or("public"),
                    cases,
                    json!({
                        "catalog": args.catalog,
                        "catalog_schema": catalog_schema,
                        "description": entry.description,
                        "corpus_id": entry.corpus_id,
                        "tags": entry.tags,
                    }),
                )?;
                imported.push(json!({
                    "name": payload["name"],
                    "kind": payload["kind"],
                    "source": payload["source"],
                    "cases": payload["cases"].as_array().map(|rows| rows.len()).unwrap_or(0),
                    "path": path,
                }));
            }
            let payload = json!({
                "catalog": args.catalog,
                "imported": imported,
                "count": imported.len(),
            });
            if json_mode {
                print_json(&payload)?;
            } else {
                println!(
                    "synced {} public benchmark packs from {}",
                    payload["count"].as_u64().unwrap_or(0),
                    args.catalog
                );
            }
        }
        BenchmarkCmd::PublishParity(args) => {
            return run_benchmark_publish_parity(cwd, args, json_mode);
        }
        BenchmarkCmd::RunMatrix(args) => {
            return run_benchmark_matrix(cwd, args, json_mode);
        }
    }
    Ok(())
}

pub(crate) fn run_benchmark_publish_parity(
    cwd: &Path,
    args: BenchmarkPublishParityArgs,
    json_mode: bool,
) -> Result<()> {
    let matrix_path = args
        .matrix
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| runtime_dir(cwd).join("benchmark-matrix-parity.json"));
    let matrix_created = ensure_parity_matrix_file(&matrix_path)?;
    let output_dir = args
        .output_dir
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| runtime_dir(cwd).join("reports/parity"));
    fs::create_dir_all(&output_dir)?;

    let matrix_args = BenchmarkRunMatrixArgs {
        matrix: matrix_path.to_string_lossy().to_string(),
        output: None,
        compare: args.compare.clone(),
        report_output: None,
        require_agent: args.require_agent.clone(),
        strict: args.strict,
        signing_key_env: args.signing_key_env.clone(),
    };
    let payload = benchmark_matrix_payload(cwd, &matrix_args, json_mode)?;

    let timestamp = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let stamped_json = output_dir.join(format!("{timestamp}.json"));
    let stamped_md = output_dir.join(format!("{timestamp}.md"));
    let latest_json = output_dir.join("latest.json");
    let latest_md = output_dir.join("latest.md");
    fs::write(&stamped_json, serde_json::to_vec_pretty(&payload)?)?;
    let report = payload
        .get("report_markdown")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    fs::write(&stamped_md, &report)?;
    fs::write(&latest_json, serde_json::to_vec_pretty(&payload)?)?;
    fs::write(&latest_md, &report)?;

    let history_path = output_dir.join("history.jsonl");
    let mut history = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&history_path)?;
    let history_entry = json!({
        "generated_at": Utc::now().to_rfc3339(),
        "matrix_path": matrix_path,
        "report_json": stamped_json,
        "report_markdown": stamped_md,
        "summary": payload.get("summary").cloned().unwrap_or(json!({})),
    });
    writeln!(history, "{}", serde_json::to_string(&history_entry)?)?;

    let result = json!({
        "published": true,
        "matrix_created": matrix_created,
        "matrix_path": matrix_path,
        "output_dir": output_dir,
        "stamped_json": stamped_json,
        "stamped_markdown": stamped_md,
        "latest_json": latest_json,
        "latest_markdown": latest_md,
        "history": history_path,
        "matrix_payload": payload,
    });
    if json_mode {
        print_json(&result)?;
    } else {
        println!(
            "published parity report: {} (latest: {})",
            result["stamped_markdown"].as_str().unwrap_or_default(),
            result["latest_markdown"].as_str().unwrap_or_default()
        );
    }
    Ok(())
}
