use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DebugAnalysisMode {
    Auto,
    Runtime,
    Test,
    Performance,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct DebugIssue {
    pub(crate) category: String,
    pub(crate) severity: String,
    pub(crate) title: String,
    pub(crate) evidence: String,
    pub(crate) suggestion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DebugAnalysis {
    pub(crate) mode: DebugAnalysisMode,
    pub(crate) issues: Vec<DebugIssue>,
    pub(crate) summary: String,
    pub(crate) next_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FrameworkSignal {
    pub(crate) name: String,
    pub(crate) ecosystem: String,
    pub(crate) confidence: f32,
    pub(crate) evidence: Vec<String>,
    pub(crate) recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FrameworkReport {
    pub(crate) detected: Vec<FrameworkSignal>,
    pub(crate) primary_ecosystem: String,
    pub(crate) recommendations: Vec<String>,
}

#[derive(Default)]
struct SignalAccumulator {
    ecosystem: String,
    confidence: f32,
    evidence: BTreeSet<String>,
    recommendations: BTreeSet<String>,
}

pub(crate) fn analyze_debug_text(text: &str, requested_mode: DebugAnalysisMode) -> DebugAnalysis {
    let mode = match requested_mode {
        DebugAnalysisMode::Auto => infer_debug_mode(text),
        other => other,
    };

    let mut issues = match mode {
        DebugAnalysisMode::Runtime => analyze_runtime_debug(text),
        DebugAnalysisMode::Test => analyze_test_debug(text),
        DebugAnalysisMode::Performance => analyze_performance_debug(text),
        DebugAnalysisMode::Auto => Vec::new(),
    };

    if issues.is_empty() {
        issues.push(DebugIssue {
            category: "analysis".to_string(),
            severity: "info".to_string(),
            title: "No clear issue signature found".to_string(),
            evidence: clip_line(text.lines().next().unwrap_or("")).to_string(),
            suggestion: "Provide more complete logs or use --mode runtime|test|performance."
                .to_string(),
        });
    }

    let mut next_steps = Vec::new();
    let mut seen = BTreeSet::new();
    for issue in &issues {
        if seen.insert(issue.suggestion.clone()) {
            next_steps.push(issue.suggestion.clone());
        }
    }

    DebugAnalysis {
        mode,
        summary: format!("Detected {} potential issue(s)", issues.len()),
        issues,
        next_steps,
    }
}

pub(crate) fn detect_frameworks(workspace: &Path) -> Result<FrameworkReport> {
    let mut signals: HashMap<String, SignalAccumulator> = HashMap::new();

    detect_js_frameworks(workspace, &mut signals);
    detect_python_frameworks(workspace, &mut signals);
    detect_rust_frameworks(workspace, &mut signals);
    detect_go_frameworks(workspace, &mut signals);

    let mut detected = signals
        .into_iter()
        .map(|(name, acc)| FrameworkSignal {
            name,
            ecosystem: acc.ecosystem,
            confidence: acc.confidence,
            evidence: acc.evidence.into_iter().collect(),
            recommendations: acc.recommendations.into_iter().collect(),
        })
        .collect::<Vec<_>>();

    detected.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.name.cmp(&b.name))
    });

    let mut ecosystem_counts: HashMap<String, usize> = HashMap::new();
    for signal in &detected {
        *ecosystem_counts
            .entry(signal.ecosystem.clone())
            .or_insert(0usize) += 1;
    }
    let primary_ecosystem = ecosystem_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(ecosystem, _)| ecosystem)
        .unwrap_or_else(|| "unknown".to_string());

    let mut recommendations = Vec::new();
    let mut seen = BTreeSet::new();
    for signal in &detected {
        for rec in &signal.recommendations {
            if seen.insert(rec.clone()) {
                recommendations.push(rec.clone());
            }
        }
    }

    Ok(FrameworkReport {
        detected,
        primary_ecosystem,
        recommendations,
    })
}

fn infer_debug_mode(text: &str) -> DebugAnalysisMode {
    let lower = text.to_ascii_lowercase();
    if lower.contains("test result: failed")
        || lower.contains("assertion failed")
        || lower.contains("assertionerror")
        || lower.contains("fail ")
        || lower.contains("... failed")
    {
        return DebugAnalysisMode::Test;
    }
    if lower.contains("latency")
        || lower.contains("throughput")
        || lower.contains("p95")
        || lower.contains("slow")
        || lower.contains("memory")
        || lower.contains("cpu")
    {
        return DebugAnalysisMode::Performance;
    }
    DebugAnalysisMode::Runtime
}

fn analyze_runtime_debug(text: &str) -> Vec<DebugIssue> {
    let mut issues = Vec::new();
    for line in text.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("panicked at")
            || lower.contains("thread '")
            || lower.contains("traceback")
            || lower.contains("exception")
        {
            push_issue(
                &mut issues,
                "runtime",
                "high",
                "Unhandled runtime failure",
                line,
                "Inspect the first stack frame in your code and add a focused repro test.",
            );
        }
        if lower.contains("permission denied") || lower.contains("operation not permitted") {
            push_issue(
                &mut issues,
                "permissions",
                "high",
                "Permission or sandbox failure",
                line,
                "Adjust permissions/sandbox policy or run the command with explicit approval.",
            );
        }
        if lower.contains("no such file")
            || lower.contains("not found")
            || lower.contains("module not found")
        {
            push_issue(
                &mut issues,
                "io",
                "medium",
                "Missing file or module",
                line,
                "Verify working directory, import path, and build artifacts before rerunning.",
            );
        }
        if lower.contains("connection refused")
            || lower.contains("timed out")
            || lower.contains("network is unreachable")
            || lower.contains("dns")
        {
            push_issue(
                &mut issues,
                "network",
                "medium",
                "Network dependency failure",
                line,
                "Check endpoint availability, credentials, and retry with backoff.",
            );
        }
        if lower.contains("out of memory") || lower.contains("oom") {
            push_issue(
                &mut issues,
                "memory",
                "high",
                "Out-of-memory condition",
                line,
                "Reduce workload size, stream data, or raise memory limits and profile allocations.",
            );
        }
        if lower.contains("nullpointerexception")
            || lower.contains("typeerror")
            || lower.contains("undefined is not")
        {
            push_issue(
                &mut issues,
                "nullability",
                "high",
                "Null/undefined access",
                line,
                "Guard nullable values and add explicit precondition checks at call boundaries.",
            );
        }
    }
    issues
}

fn analyze_test_debug(text: &str) -> Vec<DebugIssue> {
    let mut issues = Vec::new();
    let mut failed_tests = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        if trimmed.starts_with("test ") && trimmed.ends_with("... FAILED") {
            failed_tests.push(
                trimmed
                    .trim_start_matches("test ")
                    .trim_end_matches("... FAILED")
                    .trim()
                    .to_string(),
            );
        }
        if lower.starts_with("fail ") {
            failed_tests.push(trimmed.trim_start_matches("FAIL ").trim().to_string());
        }
        if lower.contains("assertion failed") || lower.contains("assertionerror") {
            push_issue(
                &mut issues,
                "assertion",
                "high",
                "Assertion mismatch",
                trimmed,
                "Compare expected/actual values and update either test fixture or implementation.",
            );
        }
        if lower.contains("snapshot") && lower.contains("failed") {
            push_issue(
                &mut issues,
                "snapshot",
                "medium",
                "Snapshot drift detected",
                trimmed,
                "Review snapshot diff; update snapshot only after validating intended UI behavior.",
            );
        }
    }

    if !failed_tests.is_empty() {
        let names = failed_tests
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        push_issue(
            &mut issues,
            "tests",
            "high",
            "Failing tests detected",
            &format!("Failed tests: {names}"),
            "Re-run only failing tests with verbose output and fix the first deterministic failure.",
        );
    }

    if text.to_ascii_lowercase().contains("test result: failed") {
        push_issue(
            &mut issues,
            "tests",
            "medium",
            "Overall test suite failed",
            "test result: FAILED",
            "Fix deterministic failures before flaky ones, then re-run the full suite.",
        );
    }

    issues
}

fn analyze_performance_debug(text: &str) -> Vec<DebugIssue> {
    let mut issues = Vec::new();
    let mut slowest_ms = 0.0f64;
    let mut memory_mb = 0.0f64;

    for line in text.lines() {
        let lower = line.to_ascii_lowercase();
        for token in line.split(|ch: char| {
            ch.is_whitespace()
                || matches!(
                    ch,
                    ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\''
                )
        }) {
            if let Some(ms) = parse_duration_ms(token)
                && ms > slowest_ms
            {
                slowest_ms = ms;
            }
            if let Some(mb) = parse_memory_mb(token)
                && mb > memory_mb
            {
                memory_mb = mb;
            }
        }

        if lower.contains("blocked for") || lower.contains("slow query") || lower.contains("n+1") {
            push_issue(
                &mut issues,
                "latency",
                "medium",
                "Potential high-latency hotspot",
                line,
                "Capture a profile around this code path and remove repeated expensive operations.",
            );
        }
    }

    if slowest_ms >= 1000.0 {
        push_issue(
            &mut issues,
            "latency",
            "high",
            "Very slow operation detected",
            &format!("max_duration_ms={slowest_ms:.1}"),
            "Add tracing spans around the hottest path and optimize I/O or algorithmic complexity.",
        );
    } else if slowest_ms >= 300.0 {
        push_issue(
            &mut issues,
            "latency",
            "medium",
            "Slow operation detected",
            &format!("max_duration_ms={slowest_ms:.1}"),
            "Investigate this path with targeted benchmarks and caching.",
        );
    }

    if memory_mb >= 1024.0 {
        push_issue(
            &mut issues,
            "memory",
            "high",
            "High memory footprint detected",
            &format!("peak_memory_mb={memory_mb:.1}"),
            "Profile allocations and reduce retained objects; stream or chunk large workloads.",
        );
    } else if memory_mb >= 512.0 {
        push_issue(
            &mut issues,
            "memory",
            "medium",
            "Elevated memory usage detected",
            &format!("peak_memory_mb={memory_mb:.1}"),
            "Track largest allocators and remove duplicate buffering.",
        );
    }

    issues
}

fn push_issue(
    issues: &mut Vec<DebugIssue>,
    category: &str,
    severity: &str,
    title: &str,
    evidence: &str,
    suggestion: &str,
) {
    if issues
        .iter()
        .any(|issue| issue.category == category && issue.title == title)
    {
        return;
    }
    issues.push(DebugIssue {
        category: category.to_string(),
        severity: severity.to_string(),
        title: title.to_string(),
        evidence: clip_line(evidence).to_string(),
        suggestion: suggestion.to_string(),
    });
}

fn clip_line(line: &str) -> &str {
    let trimmed = line.trim();
    if trimmed.len() <= 220 {
        return trimmed;
    }
    let mut end = 220usize;
    while !trimmed.is_char_boundary(end) {
        end = end.saturating_sub(1);
        if end == 0 {
            break;
        }
    }
    &trimmed[..end]
}

fn parse_duration_ms(token: &str) -> Option<f64> {
    let lower = token.trim().to_ascii_lowercase();
    if let Some(value) = lower.strip_suffix("ms") {
        return value.parse::<f64>().ok();
    }
    if let Some(value) = lower.strip_suffix('s') {
        return value.parse::<f64>().ok().map(|sec| sec * 1000.0);
    }
    None
}

fn parse_memory_mb(token: &str) -> Option<f64> {
    let lower = token.trim().to_ascii_lowercase();
    if let Some(value) = lower.strip_suffix("mb") {
        return value.parse::<f64>().ok();
    }
    if let Some(value) = lower.strip_suffix("gb") {
        return value.parse::<f64>().ok().map(|gb| gb * 1024.0);
    }
    None
}

fn detect_js_frameworks(workspace: &Path, signals: &mut HashMap<String, SignalAccumulator>) {
    let package_json = workspace.join("package.json");
    if package_json.exists() {
        let deps = read_package_dependency_names(&package_json);
        for dep in deps {
            match dep.as_str() {
                "next" => add_signal(
                    signals,
                    "Next.js",
                    "javascript",
                    0.98,
                    "package.json dependency: next",
                    &[
                        "Use App Router conventions consistently and keep route segments explicit.",
                        "Prefer Server Components for data-heavy views; isolate client components.",
                    ],
                ),
                "react" => add_signal(
                    signals,
                    "React",
                    "javascript",
                    0.94,
                    "package.json dependency: react",
                    &[
                        "Keep component state local; extract shared logic into custom hooks.",
                        "Use strict prop typing and memoization only on measured hot paths.",
                    ],
                ),
                "vue" => add_signal(
                    signals,
                    "Vue",
                    "javascript",
                    0.94,
                    "package.json dependency: vue",
                    &[
                        "Keep component contracts explicit with typed props/emits.",
                        "Avoid mixing options and composition APIs in the same feature area.",
                    ],
                ),
                "@angular/core" => add_signal(
                    signals,
                    "Angular",
                    "javascript",
                    0.95,
                    "package.json dependency: @angular/core",
                    &[
                        "Group features by domain module and keep DI boundaries explicit.",
                        "Prefer OnPush change detection for large component trees.",
                    ],
                ),
                "svelte" | "@sveltejs/kit" => add_signal(
                    signals,
                    "Svelte",
                    "javascript",
                    0.93,
                    "package.json dependency: svelte",
                    &[
                        "Keep stores scoped by feature and avoid global mutable state.",
                        "Use load functions for server data boundaries in SvelteKit.",
                    ],
                ),
                "nuxt" => add_signal(
                    signals,
                    "Nuxt",
                    "javascript",
                    0.93,
                    "package.json dependency: nuxt",
                    &[
                        "Use Nuxt server routes for backend adapters and keep composables pure.",
                        "Validate runtime config usage per environment.",
                    ],
                ),
                "express" => add_signal(
                    signals,
                    "Express",
                    "javascript",
                    0.9,
                    "package.json dependency: express",
                    &[
                        "Centralize error middleware and request validation.",
                        "Keep routing and business logic separated by domain modules.",
                    ],
                ),
                "fastify" => add_signal(
                    signals,
                    "Fastify",
                    "javascript",
                    0.9,
                    "package.json dependency: fastify",
                    &[
                        "Use schema-based validation for request/response contracts.",
                        "Register plugins per feature and avoid global decorators where possible.",
                    ],
                ),
                "@nestjs/core" => add_signal(
                    signals,
                    "NestJS",
                    "javascript",
                    0.92,
                    "package.json dependency: @nestjs/core",
                    &[
                        "Use module boundaries to isolate domains and providers.",
                        "Prefer DTO validation pipes for all external request surfaces.",
                    ],
                ),
                _ => {}
            }
        }
    }

    for (path, framework, evidence, confidence) in [
        ("next.config.js", "Next.js", "next.config.js present", 0.99),
        (
            "next.config.mjs",
            "Next.js",
            "next.config.mjs present",
            0.99,
        ),
        ("nuxt.config.ts", "Nuxt", "nuxt.config.ts present", 0.99),
        ("nuxt.config.js", "Nuxt", "nuxt.config.js present", 0.99),
        ("angular.json", "Angular", "angular.json present", 0.99),
        ("vite.config.ts", "Vite", "vite.config.ts present", 0.87),
        ("vite.config.js", "Vite", "vite.config.js present", 0.87),
    ] {
        if workspace.join(path).exists() {
            add_signal(
                signals,
                framework,
                "javascript",
                confidence,
                evidence,
                &["Keep build tooling and app framework versions aligned across the repo."],
            );
        }
    }
}

fn detect_python_frameworks(workspace: &Path, signals: &mut HashMap<String, SignalAccumulator>) {
    for file in [
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "Pipfile",
    ] {
        let path = workspace.join(file);
        if !path.exists() {
            continue;
        }
        let lower = fs::read_to_string(&path)
            .unwrap_or_default()
            .to_ascii_lowercase();
        if lower.contains("django") {
            add_signal(
                signals,
                "Django",
                "python",
                0.92,
                format!("{file} references django"),
                &[
                    "Keep settings modular per environment and centralize app-level middleware.",
                    "Use explicit queryset prefetch/select_related to control ORM performance.",
                ],
            );
        }
        if lower.contains("flask") {
            add_signal(
                signals,
                "Flask",
                "python",
                0.9,
                format!("{file} references flask"),
                &[
                    "Use blueprints per domain and avoid global mutable app state.",
                    "Standardize error handlers and request validation.",
                ],
            );
        }
        if lower.contains("fastapi") {
            add_signal(
                signals,
                "FastAPI",
                "python",
                0.93,
                format!("{file} references fastapi"),
                &[
                    "Use Pydantic models for all external request/response schemas.",
                    "Separate API router layer from service/business logic modules.",
                ],
            );
        }
        if lower.contains("pytest") {
            add_signal(
                signals,
                "Pytest",
                "python",
                0.82,
                format!("{file} references pytest"),
                &[
                    "Group fixtures by scope and keep integration fixtures isolated from unit tests.",
                ],
            );
        }
    }
}

fn detect_rust_frameworks(workspace: &Path, signals: &mut HashMap<String, SignalAccumulator>) {
    let cargo_toml = workspace.join("Cargo.toml");
    if !cargo_toml.exists() {
        return;
    }
    let lower = fs::read_to_string(&cargo_toml)
        .unwrap_or_default()
        .to_ascii_lowercase();
    for (token, name, confidence) in [
        ("axum", "Axum", 0.92),
        ("actix-web", "Actix Web", 0.92),
        ("rocket", "Rocket", 0.92),
        ("tokio", "Tokio", 0.88),
        ("serde", "Serde", 0.86),
    ] {
        if lower.contains(token) {
            add_signal(
                signals,
                name,
                "rust",
                confidence,
                format!("Cargo.toml references {token}"),
                &[
                    "Treat `Result` as a first-class API boundary and enrich errors with context.",
                    "Keep async/concurrency boundaries explicit; avoid hidden blocking calls.",
                ],
            );
        }
    }
}

fn detect_go_frameworks(workspace: &Path, signals: &mut HashMap<String, SignalAccumulator>) {
    let go_mod = workspace.join("go.mod");
    if !go_mod.exists() {
        return;
    }
    let lower = fs::read_to_string(&go_mod)
        .unwrap_or_default()
        .to_ascii_lowercase();
    for (token, name, confidence) in [
        ("github.com/gin-gonic/gin", "Gin", 0.92),
        ("github.com/labstack/echo", "Echo", 0.92),
        ("github.com/gofiber/fiber", "Fiber", 0.92),
    ] {
        if lower.contains(token) {
            add_signal(
                signals,
                name,
                "go",
                confidence,
                format!("go.mod references {token}"),
                &[
                    "Centralize middleware for auth/logging/recovery and keep handlers thin.",
                    "Use context-aware timeouts for outbound calls.",
                ],
            );
        }
    }
}

fn add_signal(
    signals: &mut HashMap<String, SignalAccumulator>,
    name: &str,
    ecosystem: &str,
    confidence: f32,
    evidence: impl Into<String>,
    recommendations: &[&str],
) {
    let entry = signals.entry(name.to_string()).or_default();
    if entry.ecosystem.is_empty() {
        entry.ecosystem = ecosystem.to_string();
    }
    entry.confidence = entry.confidence.max(confidence);
    entry.evidence.insert(evidence.into());
    for rec in recommendations {
        entry.recommendations.insert((*rec).to_string());
    }
}

fn read_package_dependency_names(path: &Path) -> BTreeSet<String> {
    let raw = match fs::read_to_string(path) {
        Ok(value) => value,
        Err(_) => return BTreeSet::new(),
    };
    let value: Value = match serde_json::from_str(&raw) {
        Ok(value) => value,
        Err(_) => return BTreeSet::new(),
    };
    let mut out = BTreeSet::new();
    for key in ["dependencies", "devDependencies", "peerDependencies"] {
        if let Some(obj) = value.get(key).and_then(|value| value.as_object()) {
            for dep_name in obj.keys() {
                out.insert(dep_name.to_ascii_lowercase());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn detect_frameworks_from_package_json() {
        let dir = TempDir::new().expect("temp dir");
        fs::write(
            dir.path().join("package.json"),
            r#"{
                "dependencies": {
                    "next": "^14.0.0",
                    "react": "^18.0.0",
                    "express": "^4.0.0"
                }
            }"#,
        )
        .expect("write package.json");

        let report = detect_frameworks(dir.path()).expect("framework report");
        assert!(report.detected.iter().any(|f| f.name == "Next.js"));
        assert!(report.detected.iter().any(|f| f.name == "React"));
        assert!(report.detected.iter().any(|f| f.name == "Express"));
        assert_eq!(report.primary_ecosystem, "javascript");
    }

    #[test]
    fn runtime_debug_analysis_detects_panic() {
        let input = "thread 'main' panicked at src/main.rs:12:5\npermission denied";
        let analysis = analyze_debug_text(input, DebugAnalysisMode::Runtime);
        assert!(analysis.issues.iter().any(|i| i.category == "runtime"));
        assert!(analysis.issues.iter().any(|i| i.category == "permissions"));
    }

    #[test]
    fn test_debug_analysis_detects_failed_tests() {
        let input = "test api::creates_user ... FAILED\nassertion failed: left == right\ntest result: FAILED";
        let analysis = analyze_debug_text(input, DebugAnalysisMode::Test);
        assert!(analysis.issues.iter().any(|i| i.category == "tests"));
        assert!(analysis.issues.iter().any(|i| i.category == "assertion"));
    }

    #[test]
    fn performance_debug_analysis_detects_slow_path() {
        let input = "query took 1450ms\npeak memory 900MB";
        let analysis = analyze_debug_text(input, DebugAnalysisMode::Performance);
        assert!(analysis.issues.iter().any(|i| i.category == "latency"));
        assert!(analysis.issues.iter().any(|i| i.category == "memory"));
    }
}
