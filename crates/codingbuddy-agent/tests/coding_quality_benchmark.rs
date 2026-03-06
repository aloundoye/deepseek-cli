use anyhow::{Result, anyhow};
use codingbuddy_agent::{AgentEngine, ChatOptions};
use codingbuddy_core::{
    ChatRequest, FimRequest, LlmRequest, LlmResponse, LlmToolCall, StreamCallback, TokenUsage,
};
use codingbuddy_llm::LlmClient;
use codingbuddy_store::Store;
use codingbuddy_testkit::{
    CodingBenchmarkCaseResult, CodingBenchmarkGateThresholds, CodingBenchmarkReport, ScriptedLlm,
    compare_coding_benchmark_reports, evaluate_coding_benchmark_gate_with_thresholds,
    read_coding_benchmark_report, write_coding_benchmark_comparison_report,
    write_coding_benchmark_report,
};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
struct SharedScriptedLlm(Arc<ScriptedLlm>);

impl LlmClient for SharedScriptedLlm {
    fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        self.0.complete(req)
    }

    fn complete_streaming(&self, req: &LlmRequest, cb: StreamCallback) -> Result<LlmResponse> {
        self.0.complete_streaming(req, cb)
    }

    fn complete_chat(&self, req: &ChatRequest) -> Result<LlmResponse> {
        self.0.complete_chat(req)
    }

    fn complete_chat_streaming(
        &self,
        req: &ChatRequest,
        cb: StreamCallback,
    ) -> Result<LlmResponse> {
        self.0.complete_chat_streaming(req, cb)
    }

    fn complete_fim(&self, req: &FimRequest) -> Result<LlmResponse> {
        self.0.complete_fim(req)
    }

    fn complete_fim_streaming(&self, req: &FimRequest, cb: StreamCallback) -> Result<LlmResponse> {
        self.0.complete_fim_streaming(req, cb)
    }
}

fn tool_call_response(calls: Vec<(&str, &str, &str)>) -> LlmResponse {
    LlmResponse {
        text: String::new(),
        finish_reason: "tool_calls".to_string(),
        reasoning_content: String::new(),
        tool_calls: calls
            .into_iter()
            .map(|(id, name, args)| LlmToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments: args.to_string(),
            })
            .collect(),
        usage: Some(TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 40,
            ..Default::default()
        }),
        compatibility: None,
    }
}

fn text_response(text: &str) -> LlmResponse {
    LlmResponse {
        text: text.to_string(),
        finish_reason: "stop".to_string(),
        reasoning_content: String::new(),
        tool_calls: vec![],
        usage: Some(TokenUsage {
            prompt_tokens: 50,
            completion_tokens: 20,
            ..Default::default()
        }),
        compatibility: None,
    }
}

fn init_workspace(path: &Path) -> Result<()> {
    fs::create_dir_all(path.join("src"))?;
    fs::write(
        path.join("Cargo.toml"),
        "[package]\nname = \"codingbuddy_benchmark\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\n",
    )?;
    fs::write(path.join("src/main.rs"), "fn main() {}\n")?;
    fs::write(path.join("src/lib.rs"), "pub fn placeholder() {}\n")?;
    let init = std::process::Command::new("git")
        .args(["init", "-q"])
        .current_dir(path)
        .output()?;
    if !init.status.success() {
        return Err(anyhow!(
            "git init failed: {}",
            String::from_utf8_lossy(&init.stderr)
        ));
    }
    Ok(())
}

fn build_scripted_engine(path: &Path, responses: Vec<LlmResponse>) -> Result<AgentEngine> {
    let llm = Arc::new(ScriptedLlm::new(responses));
    let llm: Box<dyn LlmClient + Send + Sync> = Box::new(SharedScriptedLlm(llm));
    AgentEngine::new_with_llm(path, llm)
}

fn build_live_engine(path: &Path) -> Result<AgentEngine> {
    AgentEngine::new(path)
}

#[derive(Clone, Copy)]
enum BenchmarkLaneMode {
    Scripted,
    Live,
}

impl BenchmarkLaneMode {
    fn label(self) -> &'static str {
        match self {
            Self::Scripted => "scripted",
            Self::Live => "live",
        }
    }
}

struct BenchmarkCaseSpec {
    case_id: &'static str,
    category: &'static str,
    prompt: String,
    setup_files: Vec<(String, String)>,
    responses: Vec<LlmResponse>,
    expected_output_contains: String,
    expected_file_contains: Vec<(String, String)>,
    min_tool_invocations: usize,
    min_verification_attempts: usize,
    min_tool_denials: usize,
    min_compaction_events: usize,
    verification_commands: Vec<Vec<String>>,
    require_patch_applied: bool,
    require_build_success: bool,
    require_test_success: bool,
    context_window_tokens: Option<u64>,
    permission_mode: &'static str,
    approval_strategy: BenchmarkApprovalStrategy,
}

#[derive(Clone, Copy, Default)]
enum BenchmarkApprovalStrategy {
    #[default]
    None,
    DenyFirstThenAllow,
}

fn run_case(spec: BenchmarkCaseSpec, lane: BenchmarkLaneMode) -> Result<CodingBenchmarkCaseResult> {
    let temp = tempfile::tempdir()?;
    init_workspace(temp.path())?;

    for (rel, content) in &spec.setup_files {
        let file_path = temp.path().join(rel);
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(file_path, content)?;
    }

    let before = snapshot_workspace_files(temp.path())?;

    let mut engine = match lane {
        BenchmarkLaneMode::Scripted => {
            let mut scripted = spec.responses;
            scripted.push(text_response("Benchmark fallback completion."));
            build_scripted_engine(temp.path(), scripted)?
        }
        BenchmarkLaneMode::Live => build_live_engine(temp.path())?,
    };

    if let Some(provider) = benchmark_provider_name() {
        engine.cfg_mut().llm.provider = provider;
    }
    if let Some(profile) = benchmark_profile_name() {
        engine.cfg_mut().llm.profile = profile;
    }
    if let Some(context_window_tokens) = spec.context_window_tokens {
        engine.cfg_mut().llm.context_window_tokens = context_window_tokens;
    }

    engine.set_permission_mode(spec.permission_mode);
    match spec.approval_strategy {
        BenchmarkApprovalStrategy::None => {}
        BenchmarkApprovalStrategy::DenyFirstThenAllow => {
            let mut denied_once = false;
            engine.set_approval_handler(Box::new(move |_call| {
                if denied_once {
                    Ok(true)
                } else {
                    denied_once = true;
                    Ok(false)
                }
            }));
        }
    }
    engine.set_max_turns(Some(12));

    let started = Instant::now();
    let output = engine.chat_with_options(
        &spec.prompt,
        ChatOptions {
            tools: true,
            disable_team_orchestration: true,
            session_id: None,
            ..Default::default()
        },
    )?;
    let duration_ms = started.elapsed().as_millis();

    let store = Store::new(temp.path())?;
    let session_id = store
        .load_latest_session()?
        .ok_or_else(|| anyhow!("expected benchmark session to exist"))?
        .session_id;
    let projection = store.rebuild_from_events(session_id)?;
    let usage = store.usage_summary(Some(session_id), None)?;
    let estimated_cost_usd = store.total_session_cost(session_id)?;

    let patch_applied = before != snapshot_workspace_files(temp.path())?;
    let verification_results = run_verification_commands(temp.path(), &spec.verification_commands)?;
    let build_passed = verification_results
        .iter()
        .any(|run| command_looks_like_build(&run.command) && run.success);
    let tests_passed = verification_results
        .iter()
        .any(|run| command_looks_like_test(&run.command) && run.success);
    let verification_attempts = verification_results.len();
    let tool_invocations = projection.tool_invocations.len();
    let tool_denials = projection.denied_invocations.len();
    let compaction_events = projection.compaction_events;
    let retries = tool_invocations.saturating_sub(spec.min_tool_invocations);

    let mut notes = Vec::new();
    if !spec.expected_output_contains.is_empty() && !output.contains(&spec.expected_output_contains)
    {
        notes.push(format!(
            "output missing expected marker '{}'",
            spec.expected_output_contains
        ));
    }
    if tool_invocations < spec.min_tool_invocations {
        notes.push(format!(
            "tool invocations below minimum (got {tool_invocations}, expected at least {})",
            spec.min_tool_invocations
        ));
    }
    if verification_attempts < spec.min_verification_attempts {
        notes.push(format!(
            "verification attempts below minimum (got {verification_attempts}, expected at least {})",
            spec.min_verification_attempts
        ));
    }
    if tool_denials < spec.min_tool_denials {
        notes.push(format!(
            "tool denials below minimum (got {tool_denials}, expected at least {})",
            spec.min_tool_denials
        ));
    }
    if compaction_events < spec.min_compaction_events {
        notes.push(format!(
            "compaction events below minimum (got {compaction_events}, expected at least {})",
            spec.min_compaction_events
        ));
    }
    if spec.require_patch_applied && !patch_applied {
        notes.push("expected patch to be applied".to_string());
    }
    if spec.require_build_success && !build_passed {
        notes.push(format!(
            "expected successful build verification, got {:?}",
            verification_results
                .iter()
                .filter(|run| command_looks_like_build(&run.command))
                .map(|run| (&run.command, run.success))
                .collect::<Vec<_>>()
        ));
    }
    if spec.require_test_success && !tests_passed {
        notes.push(format!(
            "expected successful test verification, got {:?}",
            verification_results
                .iter()
                .filter(|run| command_looks_like_test(&run.command))
                .map(|run| (&run.command, run.success))
                .collect::<Vec<_>>()
        ));
    }

    for (rel, expected_snippet) in &spec.expected_file_contains {
        let actual = fs::read_to_string(temp.path().join(rel))
            .map_err(|err| anyhow!("failed to read {rel}: {err}"))?;
        if !actual.contains(expected_snippet) {
            notes.push(format!(
                "file '{rel}' missing expected snippet '{expected_snippet}'"
            ));
        }
    }

    let passed = notes.is_empty();
    let completion_quality_score = if passed {
        1.0
    } else if notes.len() == 1 {
        0.5
    } else {
        0.0
    };

    Ok(CodingBenchmarkCaseResult {
        case_id: spec.case_id.to_string(),
        category: spec.category.to_string(),
        passed,
        patch_applied,
        build_passed,
        tests_passed,
        tool_invocations,
        verification_attempts,
        retries,
        tool_denials,
        compaction_events,
        completion_quality_score,
        duration_ms,
        input_tokens: usage.input_tokens,
        cache_hit_tokens: usage.cache_hit_tokens,
        cache_miss_tokens: usage.cache_miss_tokens,
        output_tokens: usage.output_tokens,
        estimated_cost_usd,
        note: if notes.is_empty() {
            None
        } else {
            Some(notes.join("; "))
        },
    })
}

#[derive(Debug)]
struct CommandOutcome {
    command: String,
    success: bool,
}

fn run_verification_commands(
    workspace: &Path,
    commands: &[Vec<String>],
) -> Result<Vec<CommandOutcome>> {
    commands
        .iter()
        .map(|command| run_verification_command(workspace, command))
        .collect()
}

fn run_verification_command(workspace: &Path, argv: &[String]) -> Result<CommandOutcome> {
    let program = argv
        .first()
        .ok_or_else(|| anyhow!("verification command cannot be empty"))?;
    let status = std::process::Command::new(program)
        .args(&argv[1..])
        .current_dir(workspace)
        .status()?;
    Ok(CommandOutcome {
        command: argv.join(" "),
        success: status.success(),
    })
}

fn snapshot_workspace_files(root: &Path) -> Result<BTreeMap<String, Vec<u8>>> {
    fn visit(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name();
            if name == ".git" || name == ".codingbuddy" || name == "target" {
                continue;
            }
            if path.is_dir() {
                visit(root, &path, out)?;
                continue;
            }
            let rel = path
                .strip_prefix(root)
                .map_err(|err| anyhow!("relative path error: {err}"))?
                .to_string_lossy()
                .replace('\\', "/");
            out.insert(rel, fs::read(&path)?);
        }
        Ok(())
    }

    let mut out = BTreeMap::new();
    visit(root, root, &mut out)?;
    Ok(out)
}

fn command_looks_like_build(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    lower.contains("cargo check")
        || lower.contains("cargo build")
        || lower.contains("cargo test")
        || lower.contains("go test")
        || lower.contains("go build")
        || lower.contains("pytest")
        || lower.contains("npm test")
        || lower.contains("pnpm test")
        || lower.contains("yarn test")
}

fn command_looks_like_test(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    lower.contains("cargo test")
        || lower.contains("go test")
        || lower.contains("pytest")
        || lower.contains("npm test")
        || lower.contains("pnpm test")
        || lower.contains("yarn test")
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(Path::to_path_buf)
        .expect("workspace root")
}

fn benchmark_suite_selected(suite: &str) -> bool {
    std::env::var("CODINGBUDDY_BENCHMARK_SUITE")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .is_none_or(|value| {
            value == suite
                || value.eq_ignore_ascii_case("all")
                || value.eq_ignore_ascii_case("deterministic")
        })
}

fn benchmark_provider_name() -> Option<String> {
    std::env::var("CODINGBUDDY_BENCHMARK_PROVIDER")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn benchmark_profile_name() -> Option<String> {
    std::env::var("CODINGBUDDY_BENCHMARK_PROFILE")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn benchmark_model_name(default_label: &str) -> String {
    std::env::var("CODINGBUDDY_BENCHMARK_MODEL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default_label.to_string())
}

fn sanitize_benchmark_slug(input: &str) -> String {
    let mut slug = String::with_capacity(input.len());
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            slug.push(ch.to_ascii_lowercase());
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
    }
    slug.trim_matches('-').to_string()
}

fn benchmark_baseline_path(root: &Path, suite: &str, model: &str) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("CODINGBUDDY_BENCHMARK_BASELINE") {
        let path = PathBuf::from(path);
        return path.exists().then_some(path);
    }

    let model_specific = root.join("docs/benchmarks").join(format!(
        "{suite}.{}.baseline.json",
        sanitize_benchmark_slug(model)
    ));
    if model_specific.exists() {
        return Some(model_specific);
    }

    let legacy = root.join("docs/benchmarks/coding_quality_baseline.json");
    if suite == "coding-quality-core" && model == "scripted-tool-loop" && legacy.exists() {
        return Some(legacy);
    }

    None
}

fn maybe_compare_or_gate(report: &CodingBenchmarkReport) -> Result<()> {
    let root = repo_root();
    let output_dir = root.join(".codingbuddy/benchmarks");
    let report_path = write_coding_benchmark_report(&output_dir, report)?;
    println!("coding_quality_benchmark_report={}", report_path.display());
    println!("coding_quality_benchmark_suite={}", report.suite);
    println!("coding_quality_benchmark_model={}", report.model);
    if let Some(provider) = report.provider.as_deref() {
        println!("coding_quality_benchmark_provider={provider}");
    }
    if let Some(profile) = report.profile.as_deref() {
        println!("coding_quality_benchmark_profile={profile}");
    }
    if let Some(lane) = report.lane.as_deref() {
        println!("coding_quality_benchmark_lane={lane}");
    }

    if let Some(baseline_path) = benchmark_baseline_path(&root, &report.suite, &report.model) {
        let baseline = read_coding_benchmark_report(&baseline_path)?;
        if baseline.suite == report.suite {
            let gate = evaluate_coding_benchmark_gate_with_thresholds(
                report,
                &baseline,
                CodingBenchmarkGateThresholds {
                    max_pass_rate_drop_pct: 5.0,
                    max_quality_score_drop: 0.10,
                    max_avg_retries_increase: 0.50,
                },
            );
            assert!(
                gate.passed,
                "coding benchmark regression: suite={} compatible={} pass_rate current={:.1}% baseline={:.1}% delta={:.1}% allowed_drop={:.1}% quality current={:.3} baseline={:.3} delta={:.3} allowed_drop={:.3} retries current={:.3} baseline={:.3} delta={:.3} allowed_increase={:.3}",
                report.suite,
                gate.suite_model_compatible,
                gate.current_pass_rate_pct,
                gate.baseline_pass_rate_pct,
                gate.delta_pct,
                gate.allowed_drop_pct,
                gate.current_avg_completion_quality_score,
                gate.baseline_avg_completion_quality_score,
                gate.quality_delta,
                gate.max_quality_drop,
                gate.current_avg_retries,
                gate.baseline_avg_retries,
                gate.retries_delta,
                gate.max_retry_increase
            );
        }
    }

    if let Some(compare_to) = std::env::var_os("CODINGBUDDY_BENCHMARK_COMPARE_TO") {
        let compare_to = PathBuf::from(compare_to);
        let reference = read_coding_benchmark_report(&compare_to)?;
        if reference.suite == report.suite {
            let comparison = compare_coding_benchmark_reports(report, &reference);
            assert!(
                comparison.summary.comparable,
                "coding benchmark comparison incompatible: suite={} suite_compatible={} case_ids_compatible={} case_categories_compatible={} current_only_cases={:?} reference_only_cases={:?} category_mismatch_cases={:?}",
                report.suite,
                comparison.summary.suite_compatible,
                comparison.summary.case_ids_compatible,
                comparison.summary.case_categories_compatible,
                comparison.summary.current_only_cases,
                comparison.summary.reference_only_cases,
                comparison.summary.category_mismatch_cases,
            );
            let comparison_path =
                write_coding_benchmark_comparison_report(&output_dir, &comparison)?;
            println!(
                "coding_quality_benchmark_comparison={}",
                comparison_path.display()
            );
            println!(
                "coding_quality_benchmark_comparison_summary=suite={} current_model={} reference_model={} pass_rate_delta_pct={:.1} build_delta_pct={:.1} test_delta_pct={:.1} avg_quality_delta={:.3} avg_retries_delta={:.3} avg_verification_delta={:.3} avg_cost_delta_usd={:.6} improved_cases={} regressed_cases={}",
                comparison.suite,
                comparison.current_model,
                comparison.reference_model,
                comparison.summary.pass_rate_delta_pct,
                comparison.summary.build_pass_rate_delta_pct,
                comparison.summary.test_pass_rate_delta_pct,
                comparison.summary.avg_completion_quality_delta,
                comparison.summary.avg_retries_delta,
                comparison.summary.avg_verification_attempts_delta,
                comparison.summary.avg_estimated_cost_delta_usd,
                comparison.summary.improved_case_count,
                comparison.summary.regressed_case_count,
            );
        }
    }

    Ok(())
}

fn print_failed_cases(report: &CodingBenchmarkReport) {
    for case in report.cases.iter().filter(|case| !case.passed) {
        println!(
            "coding_quality_benchmark_failed_case suite={} case={} note={}",
            report.suite,
            case.case_id,
            case.note.as_deref().unwrap_or("n/a")
        );
    }
}

fn run_suite(
    suite: &str,
    default_model_label: &str,
    lane: BenchmarkLaneMode,
    cases: Vec<BenchmarkCaseSpec>,
) -> Result<CodingBenchmarkReport> {
    let results = cases
        .into_iter()
        .map(|case| run_case(case, lane))
        .collect::<Result<Vec<_>>>()?;
    let model = benchmark_model_name(default_model_label);
    Ok(CodingBenchmarkReport::from_case_results_with_metadata(
        suite,
        &model,
        benchmark_provider_name(),
        benchmark_profile_name(),
        Some(lane.label().to_string()),
        results,
    ))
}

fn core_suite_cases() -> Vec<BenchmarkCaseSpec> {
    vec![
        BenchmarkCaseSpec {
            case_id: "edit-single-file",
            category: "edit",
            prompt: "Add a print statement to main".to_string(),
            setup_files: vec![],
            responses: vec![
                tool_call_response(vec![(
                    "call_1",
                    "fs_edit",
                    r#"{"path":"src/main.rs","search":"fn main() {}","replace":"fn main() { println!(\"hi\"); }"}"#,
                )]),
                text_response("Applied edit to main.rs."),
            ],
            expected_output_contains: String::new(),
            expected_file_contains: vec![(
                "src/main.rs".to_string(),
                "println!(\"hi\");".to_string(),
            )],
            min_tool_invocations: 1,
            min_verification_attempts: 0,
            min_tool_denials: 0,
            min_compaction_events: 0,
            verification_commands: vec![],
            require_patch_applied: true,
            require_build_success: false,
            require_test_success: false,
            context_window_tokens: None,
            permission_mode: "bypassPermissions",
            approval_strategy: BenchmarkApprovalStrategy::None,
        },
        BenchmarkCaseSpec {
            case_id: "debug-bugfix",
            category: "debug",
            prompt: "Fix divide-by-zero in src/math.rs".to_string(),
            setup_files: vec![(
                "src/math.rs".to_string(),
                "pub fn divide(a: i32, b: i32) -> i32 { a / 0 }\n".to_string(),
            )],
            responses: vec![
                tool_call_response(vec![("call_1", "fs_read", r#"{"path":"src/math.rs"}"#)]),
                tool_call_response(vec![(
                    "call_2",
                    "fs_edit",
                    r#"{"path":"src/math.rs","search":"a / 0","replace":"a / b"}"#,
                )]),
                text_response("Fixed division bug."),
            ],
            expected_output_contains: "Fixed division".to_string(),
            expected_file_contains: vec![("src/math.rs".to_string(), "a / b".to_string())],
            min_tool_invocations: 2,
            min_verification_attempts: 0,
            min_tool_denials: 0,
            min_compaction_events: 0,
            verification_commands: vec![],
            require_patch_applied: true,
            require_build_success: false,
            require_test_success: false,
            context_window_tokens: None,
            permission_mode: "bypassPermissions",
            approval_strategy: BenchmarkApprovalStrategy::None,
        },
        BenchmarkCaseSpec {
            case_id: "refactor-rename",
            category: "refactor",
            prompt: "Rename calc_sum to sum_values in src/lib.rs".to_string(),
            setup_files: vec![(
                "src/lib.rs".to_string(),
                "pub fn calc_sum(a: i32, b: i32) -> i32 { a + b }\n".to_string(),
            )],
            responses: vec![
                tool_call_response(vec![(
                    "call_1",
                    "fs_edit",
                    r#"{"path":"src/lib.rs","search":"calc_sum","replace":"sum_values"}"#,
                )]),
                text_response("Refactor complete."),
            ],
            expected_output_contains: "Refactor complete".to_string(),
            expected_file_contains: vec![("src/lib.rs".to_string(), "sum_values".to_string())],
            min_tool_invocations: 1,
            min_verification_attempts: 0,
            min_tool_denials: 0,
            min_compaction_events: 0,
            verification_commands: vec![],
            require_patch_applied: true,
            require_build_success: false,
            require_test_success: false,
            context_window_tokens: None,
            permission_mode: "bypassPermissions",
            approval_strategy: BenchmarkApprovalStrategy::None,
        },
        BenchmarkCaseSpec {
            case_id: "multi-file-update",
            category: "multi-file",
            prompt: "Rename foo to bar in src/a.rs and src/b.rs".to_string(),
            setup_files: vec![
                (
                    "src/a.rs".to_string(),
                    "pub fn foo() -> i32 { 1 }\n".to_string(),
                ),
                (
                    "src/b.rs".to_string(),
                    "pub fn foo() -> i32 { 2 }\n".to_string(),
                ),
            ],
            responses: vec![
                tool_call_response(vec![(
                    "call_1",
                    "fs_edit",
                    r#"{"path":"src/a.rs","search":"foo","replace":"bar"}"#,
                )]),
                tool_call_response(vec![(
                    "call_2",
                    "fs_edit",
                    r#"{"path":"src/b.rs","search":"foo","replace":"bar"}"#,
                )]),
                text_response("Updated both files."),
            ],
            expected_output_contains: "Updated both files".to_string(),
            expected_file_contains: vec![
                ("src/a.rs".to_string(), "bar".to_string()),
                ("src/b.rs".to_string(), "bar".to_string()),
            ],
            min_tool_invocations: 2,
            min_verification_attempts: 0,
            min_tool_denials: 0,
            min_compaction_events: 0,
            verification_commands: vec![],
            require_patch_applied: true,
            require_build_success: false,
            require_test_success: false,
            context_window_tokens: None,
            permission_mode: "bypassPermissions",
            approval_strategy: BenchmarkApprovalStrategy::None,
        },
    ]
}

fn repo_suite_cases() -> Vec<BenchmarkCaseSpec> {
    let mut large_notes = String::from("pub const CONTEXT: &str = r#\"\n");
    for idx in 0..320 {
        let _ = writeln!(large_notes, "note line {idx}: {}", "context ".repeat(12));
    }
    large_notes.push_str("\"#;\n");

    vec![
        BenchmarkCaseSpec {
            case_id: "repo-wide-rename-exports-tests",
            category: "rename",
            prompt: "Rename calc_sum to sum_values across the library export and tests, then verify with cargo test.".to_string(),
            setup_files: vec![
                (
                    "src/lib.rs".to_string(),
                    "mod math;\npub use math::calc_sum;\n".to_string(),
                ),
                (
                    "src/math.rs".to_string(),
                    "pub fn calc_sum(a: i32, b: i32) -> i32 { a + b }\n".to_string(),
                ),
                (
                    "tests/math_tests.rs".to_string(),
                    "use codingbuddy_benchmark::calc_sum;\n\n#[test]\nfn sums_values() {\n    assert_eq!(calc_sum(2, 3), 5);\n}\n".to_string(),
                ),
            ],
            responses: vec![
                tool_call_response(vec![(
                    "call_1",
                    "fs_edit",
                    r#"{"path":"src/math.rs","search":"calc_sum","replace":"sum_values"}"#,
                )]),
                tool_call_response(vec![(
                    "call_2",
                    "fs_edit",
                    r#"{"path":"src/lib.rs","search":"calc_sum","replace":"sum_values"}"#,
                )]),
                tool_call_response(vec![(
                    "call_3",
                    "fs_edit",
                    r#"{"path":"tests/math_tests.rs","search":"calc_sum","replace":"sum_values"}"#,
                )]),
                tool_call_response(vec![(
                    "call_4",
                    "bash_run",
                    r#"{"command":"cargo test --quiet"}"#,
                )]),
                text_response("Repo rename verified."),
            ],
            expected_output_contains: "Repo rename verified".to_string(),
            expected_file_contains: vec![
                ("src/math.rs".to_string(), "sum_values".to_string()),
                ("src/lib.rs".to_string(), "sum_values".to_string()),
                ("tests/math_tests.rs".to_string(), "sum_values".to_string()),
            ],
            min_tool_invocations: 4,
            min_verification_attempts: 1,
            min_tool_denials: 0,
            min_compaction_events: 0,
            verification_commands: vec![vec![
                "cargo".to_string(),
                "test".to_string(),
                "--quiet".to_string(),
            ]],
            require_patch_applied: true,
            require_build_success: true,
            require_test_success: true,
            context_window_tokens: None,
            permission_mode: "bypassPermissions",
            approval_strategy: BenchmarkApprovalStrategy::None,
        },
        BenchmarkCaseSpec {
            case_id: "failing-unit-test-fix",
            category: "test-fix",
            prompt: "Fix the failing greeting test and verify with cargo test.".to_string(),
            setup_files: vec![
                (
                    "src/lib.rs".to_string(),
                    "pub fn greet(name: &str) -> String { format!(\"Hello {name}\") }\n".to_string(),
                ),
                (
                    "tests/greet.rs".to_string(),
                    "use codingbuddy_benchmark::greet;\n\n#[test]\nfn greet_includes_comma() {\n    assert_eq!(greet(\"Sam\"), \"Hello, Sam\");\n}\n".to_string(),
                ),
            ],
            responses: vec![
                tool_call_response(vec![("call_1", "fs_read", r#"{"path":"tests/greet.rs"}"#)]),
                tool_call_response(vec![(
                    "call_2",
                    "fs_edit",
                    r#"{"path":"src/lib.rs","search":"Hello {name}","replace":"Hello, {name}"}"#,
                )]),
                tool_call_response(vec![(
                    "call_3",
                    "bash_run",
                    r#"{"command":"cargo test --quiet"}"#,
                )]),
                text_response("Greeting test fixed."),
            ],
            expected_output_contains: "Greeting test fixed".to_string(),
            expected_file_contains: vec![("src/lib.rs".to_string(), "Hello, {name}".to_string())],
            min_tool_invocations: 3,
            min_verification_attempts: 1,
            min_tool_denials: 0,
            min_compaction_events: 0,
            verification_commands: vec![vec![
                "cargo".to_string(),
                "test".to_string(),
                "--quiet".to_string(),
            ]],
            require_patch_applied: true,
            require_build_success: true,
            require_test_success: true,
            context_window_tokens: None,
            permission_mode: "bypassPermissions",
            approval_strategy: BenchmarkApprovalStrategy::None,
        },
        BenchmarkCaseSpec {
            case_id: "compiler-build-error-fix",
            category: "build-fix",
            prompt: "Fix the compiler error in src/lib.rs and verify with cargo check.".to_string(),
            setup_files: vec![(
                "src/lib.rs".to_string(),
                "pub fn parse_port() -> i32 { missing_symbol }\n".to_string(),
            )],
            responses: vec![
                tool_call_response(vec![(
                    "call_1",
                    "fs_edit",
                    r#"{"path":"src/lib.rs","search":"missing_symbol","replace":"8080"}"#,
                )]),
                tool_call_response(vec![(
                    "call_2",
                    "bash_run",
                    r#"{"command":"cargo check --quiet"}"#,
                )]),
                text_response("Compiler error fixed."),
            ],
            expected_output_contains: "Compiler error fixed".to_string(),
            expected_file_contains: vec![("src/lib.rs".to_string(), "8080".to_string())],
            min_tool_invocations: 2,
            min_verification_attempts: 1,
            min_tool_denials: 0,
            min_compaction_events: 0,
            verification_commands: vec![vec![
                "cargo".to_string(),
                "check".to_string(),
                "--quiet".to_string(),
            ]],
            require_patch_applied: true,
            require_build_success: true,
            require_test_success: false,
            context_window_tokens: None,
            permission_mode: "bypassPermissions",
            approval_strategy: BenchmarkApprovalStrategy::None,
        },
        BenchmarkCaseSpec {
            case_id: "ambiguous-refactor-acceptance",
            category: "refactor",
            prompt: "Refactor read_name so it handles missing input cleanly, then prove it with cargo test.".to_string(),
            setup_files: vec![
                (
                    "src/lib.rs".to_string(),
                    "pub fn read_name(input: Option<&str>) -> String {\n    input.unwrap().trim().to_string()\n}\n".to_string(),
                ),
                (
                    "tests/name.rs".to_string(),
                    "use codingbuddy_benchmark::read_name;\n\n#[test]\nfn trims_present_name() {\n    assert_eq!(read_name(Some(\"  Ada  \")), \"Ada\");\n}\n\n#[test]\nfn missing_name_is_unknown() {\n    assert_eq!(read_name(None), \"unknown\");\n}\n".to_string(),
                ),
            ],
            responses: vec![
                tool_call_response(vec![("call_1", "fs_read", r#"{"path":"src/lib.rs"}"#)]),
                tool_call_response(vec![(
                    "call_2",
                    "fs_edit",
                    r#"{"path":"src/lib.rs","search":"input.unwrap().trim().to_string()","replace":"input.map(|value| value.trim().to_string()).filter(|value| !value.is_empty()).unwrap_or_else(|| \"unknown\".to_string())"}"#,
                )]),
                tool_call_response(vec![(
                    "call_3",
                    "bash_run",
                    r#"{"command":"cargo test --quiet"}"#,
                )]),
                text_response("Refactor accepted by tests."),
            ],
            expected_output_contains: String::new(),
            expected_file_contains: vec![("src/lib.rs".to_string(), "unknown".to_string())],
            min_tool_invocations: 3,
            min_verification_attempts: 1,
            min_tool_denials: 0,
            min_compaction_events: 0,
            verification_commands: vec![vec![
                "cargo".to_string(),
                "test".to_string(),
                "--quiet".to_string(),
            ]],
            require_patch_applied: true,
            require_build_success: true,
            require_test_success: true,
            context_window_tokens: None,
            permission_mode: "bypassPermissions",
            approval_strategy: BenchmarkApprovalStrategy::None,
        },
        BenchmarkCaseSpec {
            case_id: "tool-denial-recovery",
            category: "recovery",
            prompt: "Change src/lib.rs so status returns ready.".to_string(),
            setup_files: vec![(
                "src/lib.rs".to_string(),
                "pub fn status() -> &'static str { \"draft\" }\n".to_string(),
            )],
            responses: vec![
                tool_call_response(vec![(
                    "call_1",
                    "fs_edit",
                    r#"{"path":"src/lib.rs","search":"draft","replace":"ready"}"#,
                )]),
                tool_call_response(vec![(
                    "call_2",
                    "fs_edit",
                    r#"{"path":"src/lib.rs","search":"draft","replace":"ready"}"#,
                )]),
                text_response("Recovered after denied command."),
            ],
            expected_output_contains: "Recovered after denied command".to_string(),
            expected_file_contains: vec![("src/lib.rs".to_string(), "ready".to_string())],
            min_tool_invocations: 2,
            min_verification_attempts: 0,
            min_tool_denials: 1,
            min_compaction_events: 0,
            verification_commands: vec![],
            require_patch_applied: true,
            require_build_success: false,
            require_test_success: false,
            context_window_tokens: None,
            permission_mode: "ask",
            approval_strategy: BenchmarkApprovalStrategy::DenyFirstThenAllow,
        },
        BenchmarkCaseSpec {
            case_id: "compaction-pressure-follow-up-edit",
            category: "follow-up",
            prompt: "Rename summary_title to final_title in src/lib.rs.".to_string(),
            setup_files: vec![
                (
                    "src/lib.rs".to_string(),
                    "pub fn summary_title() -> &'static str { \"draft\" }\n".to_string(),
                ),
                ("src/notes.rs".to_string(), large_notes),
            ],
            responses: vec![
                tool_call_response(vec![("call_1", "fs_read", r#"{"path":"src/notes.rs"}"#)]),
                tool_call_response(vec![(
                    "call_2",
                    "fs_edit",
                    r#"{"path":"src/lib.rs","search":"summary_title","replace":"final_title"}"#,
                )]),
                text_response("Follow-up edit applied after compaction."),
            ],
            expected_output_contains: "Follow-up edit applied".to_string(),
            expected_file_contains: vec![
                ("src/lib.rs".to_string(), "final_title".to_string()),
            ],
            min_tool_invocations: 2,
            min_verification_attempts: 0,
            min_tool_denials: 0,
            min_compaction_events: 1,
            verification_commands: vec![],
            require_patch_applied: true,
            require_build_success: false,
            require_test_success: false,
            context_window_tokens: Some(400),
            permission_mode: "bypassPermissions",
            approval_strategy: BenchmarkApprovalStrategy::None,
        },
    ]
}

#[test]
fn coding_quality_core_benchmark_suite() -> Result<()> {
    if !benchmark_suite_selected("coding-quality-core") {
        return Ok(());
    }

    let report = run_suite(
        "coding-quality-core",
        "scripted-tool-loop",
        BenchmarkLaneMode::Scripted,
        core_suite_cases(),
    )?;
    print_failed_cases(&report);
    assert_eq!(report.summary.total_cases, 4);
    assert!(report.summary.pass_rate_pct >= 75.0);
    maybe_compare_or_gate(&report)
}

#[test]
fn coding_quality_repo_benchmark_suite() -> Result<()> {
    if !benchmark_suite_selected("coding-quality-repo") {
        return Ok(());
    }

    let report = run_suite(
        "coding-quality-repo",
        "scripted-tool-loop",
        BenchmarkLaneMode::Scripted,
        repo_suite_cases(),
    )?;
    print_failed_cases(&report);
    assert_eq!(report.summary.total_cases, 6);
    assert!(report.summary.patch_applied_rate_pct >= 80.0);
    assert!(report.summary.pass_rate_pct >= 66.0);
    maybe_compare_or_gate(&report)
}

#[test]
#[ignore = "requires live provider configuration"]
fn coding_quality_core_live_benchmark_suite() -> Result<()> {
    if !benchmark_suite_selected("coding-quality-core") {
        return Ok(());
    }

    let report = run_suite(
        "coding-quality-core",
        "live-active-model",
        BenchmarkLaneMode::Live,
        core_suite_cases(),
    )?;
    maybe_compare_or_gate(&report)
}

#[test]
#[ignore = "requires live provider configuration"]
fn coding_quality_repo_live_benchmark_suite() -> Result<()> {
    if !benchmark_suite_selected("coding-quality-repo") {
        return Ok(());
    }

    let report = run_suite(
        "coding-quality-repo",
        "live-active-model",
        BenchmarkLaneMode::Live,
        repo_suite_cases(),
    )?;
    maybe_compare_or_gate(&report)
}
