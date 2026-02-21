use anyhow::{Result, anyhow};
use chrono::Utc;
use deepseek_store::Store;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::output::*;
use crate::{VisualAnalyzeArgs, VisualCmd, VisualListArgs};

pub(crate) fn parse_visual_cmd(args: Vec<String>) -> Result<VisualCmd> {
    if args.is_empty() || args[0].eq_ignore_ascii_case("list") {
        return Ok(VisualCmd::List(VisualListArgs { limit: 25 }));
    }
    let first = args[0].to_ascii_lowercase();
    match first.as_str() {
        "analyze" => Ok(VisualCmd::Analyze(VisualAnalyzeArgs {
            limit: 25,
            min_bytes: 128,
            min_artifacts: 1,
            min_image_artifacts: 1,
            strict: false,
            baseline: args.get(1).cloned(),
            write_baseline: None,
            max_new_artifacts: 0,
            max_missing_artifacts: 0,
            max_changed_artifacts: 0,
            expectations: None,
        })),
        _ => Err(anyhow!("use /visual list|analyze")),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VisualBaselineEntry {
    pub(crate) path: String,
    pub(crate) mime: String,
    pub(crate) size_bytes: u64,
    #[serde(default)]
    pub(crate) sha256: Option<String>,
    #[serde(default)]
    pub(crate) image_like: bool,
    #[serde(default)]
    pub(crate) width: Option<u32>,
    #[serde(default)]
    pub(crate) height: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VisualBaselineFile {
    pub(crate) schema: String,
    pub(crate) generated_at: String,
    pub(crate) workspace: String,
    pub(crate) artifacts: Vec<VisualBaselineEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VisualExpectationRule {
    pub(crate) path_glob: String,
    #[serde(default)]
    pub(crate) name: Option<String>,
    #[serde(default)]
    pub(crate) min_count: Option<usize>,
    #[serde(default)]
    pub(crate) max_count: Option<usize>,
    #[serde(default)]
    pub(crate) mime: Option<String>,
    #[serde(default)]
    pub(crate) min_bytes: Option<u64>,
    #[serde(default)]
    pub(crate) min_width: Option<u32>,
    #[serde(default)]
    pub(crate) min_height: Option<u32>,
    #[serde(default)]
    pub(crate) max_width: Option<u32>,
    #[serde(default)]
    pub(crate) max_height: Option<u32>,
    #[serde(default)]
    pub(crate) required_path_substrings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VisualExpectationFile {
    #[serde(default)]
    pub(crate) schema: String,
    #[serde(default)]
    pub(crate) rules: Vec<VisualExpectationRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VisualSemanticEntry {
    pub(crate) path: String,
    pub(crate) mime: String,
    pub(crate) size_bytes: u64,
    pub(crate) image_like: bool,
    pub(crate) exists: bool,
    pub(crate) width: Option<u32>,
    pub(crate) height: Option<u32>,
}

pub(crate) fn run_visual(cwd: &Path, cmd: VisualCmd, json_mode: bool) -> Result<()> {
    let payload = visual_payload(cwd, cmd)?;
    if json_mode {
        print_json(&payload)?;
        return Ok(());
    }

    if let Some(rows) = payload.as_array() {
        if rows.is_empty() {
            println!("no visual artifacts found");
            return Ok(());
        }
        for row in rows {
            println!(
                "{} {} exists={} size={} path={}",
                row["artifact_id"].as_str().unwrap_or_default(),
                row["mime"].as_str().unwrap_or_default(),
                row["exists"].as_bool().unwrap_or(false),
                row["size_bytes"].as_u64().unwrap_or(0),
                row["path"].as_str().unwrap_or_default(),
            );
        }
        return Ok(());
    }

    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(())
}

pub(crate) fn visual_payload(cwd: &Path, cmd: VisualCmd) -> Result<serde_json::Value> {
    let store = Store::new(cwd)?;
    match cmd {
        VisualCmd::List(args) => {
            let artifacts = store.list_visual_artifacts(args.limit.max(1))?;
            let rows = artifacts
                .into_iter()
                .map(|artifact| {
                    let full = resolve_visual_artifact_path(cwd, &artifact.path);
                    let metadata = fs::metadata(&full).ok();
                    json!({
                        "artifact_id": artifact.artifact_id,
                        "path": artifact.path,
                        "full_path": full,
                        "mime": artifact.mime,
                        "metadata": serde_json::from_str::<serde_json::Value>(&artifact.metadata_json).unwrap_or_default(),
                        "created_at": artifact.created_at,
                        "exists": metadata.is_some(),
                        "size_bytes": metadata.map(|m| m.len()).unwrap_or(0),
                    })
                })
                .collect::<Vec<_>>();
            Ok(json!(rows))
        }
        VisualCmd::Analyze(args) => {
            let artifacts = store.list_visual_artifacts(args.limit.max(1))?;
            let mut rows = Vec::new();
            let mut missing_paths = Vec::new();
            let mut tiny_artifacts = Vec::new();
            let mut image_like_count = 0usize;
            let mut existing_count = 0usize;
            let mut baseline_entries = Vec::new();
            let mut semantic_entries = Vec::new();

            for artifact in artifacts {
                let full = resolve_visual_artifact_path(cwd, &artifact.path);
                let artifact_path = artifact.path.clone();
                let artifact_mime = artifact.mime.clone();
                let metadata = fs::metadata(&full).ok();
                let exists = metadata.is_some();
                let size_bytes = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
                let sha256 = if exists {
                    fs::read(&full).ok().map(|bytes| sha256_hex(&bytes))
                } else {
                    None
                };
                if exists {
                    existing_count = existing_count.saturating_add(1);
                } else {
                    missing_paths.push(artifact_path.clone());
                }
                let image_like =
                    artifact_mime.starts_with("image/") || artifact_mime == "application/pdf";
                let (width, height) = if image_like && exists {
                    read_visual_dimensions(&full, &artifact_mime)
                        .map(|(w, h)| (Some(w), Some(h)))
                        .unwrap_or((None, None))
                } else {
                    (None, None)
                };
                if image_like {
                    image_like_count = image_like_count.saturating_add(1);
                    if exists && size_bytes < args.min_bytes {
                        tiny_artifacts.push(artifact_path.clone());
                    }
                }

                rows.push(json!({
                    "artifact_id": artifact.artifact_id,
                    "path": artifact_path,
                    "full_path": full,
                    "mime": artifact_mime,
                    "created_at": artifact.created_at,
                    "exists": exists,
                    "size_bytes": size_bytes,
                    "image_like": image_like,
                    "sha256": sha256,
                    "width": width,
                    "height": height,
                }));
                baseline_entries.push(VisualBaselineEntry {
                    path: artifact_path.clone(),
                    mime: artifact_mime.clone(),
                    size_bytes,
                    sha256,
                    image_like,
                    width,
                    height,
                });
                semantic_entries.push(VisualSemanticEntry {
                    path: artifact_path,
                    mime: artifact_mime,
                    size_bytes,
                    image_like,
                    exists,
                    width,
                    height,
                });
            }

            let total = rows.len();
            let mut warnings = Vec::new();
            if total < args.min_artifacts {
                warnings.push(format!(
                    "captured_artifacts={} below required minimum {}",
                    total, args.min_artifacts
                ));
            }
            if image_like_count < args.min_image_artifacts {
                warnings.push(format!(
                    "image_like_artifacts={} below required minimum {}",
                    image_like_count, args.min_image_artifacts
                ));
            }
            if !missing_paths.is_empty() {
                warnings.push(format!(
                    "{} artifact files are missing",
                    missing_paths.len()
                ));
            }
            if !tiny_artifacts.is_empty() {
                warnings.push(format!(
                    "{} artifacts are smaller than {} bytes",
                    tiny_artifacts.len(),
                    args.min_bytes
                ));
            }
            let mut baseline_diff = json!(null);
            let mut new_artifacts = Vec::new();
            let mut missing_from_current = Vec::new();
            let mut changed_artifacts = Vec::new();
            if let Some(path) = args.baseline.as_deref() {
                let baseline_path = PathBuf::from(path);
                let baseline = read_visual_baseline_file(&baseline_path)?;
                let current_by_path = baseline_entries
                    .iter()
                    .cloned()
                    .map(|entry| (entry.path.clone(), entry))
                    .collect::<HashMap<_, _>>();
                let baseline_by_path = baseline
                    .artifacts
                    .iter()
                    .cloned()
                    .map(|entry| (entry.path.clone(), entry))
                    .collect::<HashMap<_, _>>();

                for (path, current) in &current_by_path {
                    let Some(previous) = baseline_by_path.get(path) else {
                        new_artifacts.push(path.clone());
                        continue;
                    };
                    if current.sha256 != previous.sha256
                        || current.mime != previous.mime
                        || current.width != previous.width
                        || current.height != previous.height
                    {
                        changed_artifacts.push(path.clone());
                    }
                }
                for path in baseline_by_path.keys() {
                    if !current_by_path.contains_key(path) {
                        missing_from_current.push(path.clone());
                    }
                }
                new_artifacts.sort();
                missing_from_current.sort();
                changed_artifacts.sort();
                baseline_diff = json!({
                    "source": baseline_path,
                    "schema": baseline.schema,
                    "new_artifacts": new_artifacts,
                    "missing_artifacts": missing_from_current,
                    "changed_artifacts": changed_artifacts,
                });

                if baseline_diff["new_artifacts"]
                    .as_array()
                    .map(|rows| rows.len())
                    .unwrap_or(0)
                    > args.max_new_artifacts
                {
                    warnings.push(format!(
                        "new_artifacts={} above allowed maximum {}",
                        baseline_diff["new_artifacts"]
                            .as_array()
                            .map(|rows| rows.len())
                            .unwrap_or(0),
                        args.max_new_artifacts
                    ));
                }
                if baseline_diff["missing_artifacts"]
                    .as_array()
                    .map(|rows| rows.len())
                    .unwrap_or(0)
                    > args.max_missing_artifacts
                {
                    warnings.push(format!(
                        "missing_artifacts={} above allowed maximum {}",
                        baseline_diff["missing_artifacts"]
                            .as_array()
                            .map(|rows| rows.len())
                            .unwrap_or(0),
                        args.max_missing_artifacts
                    ));
                }
                if baseline_diff["changed_artifacts"]
                    .as_array()
                    .map(|rows| rows.len())
                    .unwrap_or(0)
                    > args.max_changed_artifacts
                {
                    warnings.push(format!(
                        "changed_artifacts={} above allowed maximum {}",
                        baseline_diff["changed_artifacts"]
                            .as_array()
                            .map(|rows| rows.len())
                            .unwrap_or(0),
                        args.max_changed_artifacts
                    ));
                }
            }
            let semantic = if let Some(path) = args.expectations.as_deref() {
                let expectation_path = PathBuf::from(path);
                let semantic = evaluate_visual_semantics(&semantic_entries, &expectation_path)?;
                let failures = semantic
                    .get("failures")
                    .and_then(|v| v.as_array())
                    .map(|rows| rows.len())
                    .unwrap_or(0);
                if failures > 0 {
                    warnings.push(format!("semantic_expectations_failed={}", failures));
                }
                json!(semantic)
            } else {
                json!(null)
            };
            let baseline_written = if let Some(path) = args.write_baseline.as_deref() {
                let output = PathBuf::from(path);
                write_visual_baseline_file(
                    &output,
                    cwd,
                    VisualBaselineFile {
                        schema: "deepseek.visual.baseline.v1".to_string(),
                        generated_at: Utc::now().to_rfc3339(),
                        workspace: cwd.to_string_lossy().to_string(),
                        artifacts: baseline_entries.clone(),
                    },
                )?;
                Some(output)
            } else {
                None
            };
            let ok = warnings.is_empty();
            let payload = json!({
                "ok": ok,
                "summary": {
                    "total_artifacts": total,
                    "existing_artifacts": existing_count,
                    "missing_artifacts": total.saturating_sub(existing_count),
                    "image_like_artifacts": image_like_count,
                    "min_bytes": args.min_bytes,
                    "min_artifacts": args.min_artifacts,
                    "min_image_artifacts": args.min_image_artifacts,
                },
                "warnings": warnings,
                "missing_paths": missing_paths,
                "tiny_artifacts": tiny_artifacts,
                "baseline_diff": baseline_diff,
                "semantic": semantic,
                "baseline_written": baseline_written,
                "artifacts": rows,
            });
            if args.strict && !ok {
                return Err(anyhow!(
                    "visual analysis failed: {}",
                    payload["warnings"]
                        .as_array()
                        .map(|rows| rows
                            .iter()
                            .filter_map(|row| row.as_str())
                            .collect::<Vec<_>>()
                            .join(" | "))
                        .unwrap_or_else(|| "unknown warning".to_string())
                ));
            }
            Ok(payload)
        }
    }
}

pub(crate) fn resolve_visual_artifact_path(cwd: &Path, artifact_path: &str) -> PathBuf {
    let candidate = PathBuf::from(artifact_path);
    if candidate.is_absolute() {
        candidate
    } else {
        cwd.join(candidate)
    }
}

pub(crate) fn read_visual_baseline_file(path: &Path) -> Result<VisualBaselineFile> {
    let raw = fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    if let Some(artifacts) = value.as_array() {
        let rows = artifacts
            .iter()
            .cloned()
            .map(serde_json::from_value::<VisualBaselineEntry>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        return Ok(VisualBaselineFile {
            schema: "deepseek.visual.baseline.v1".to_string(),
            generated_at: String::new(),
            workspace: String::new(),
            artifacts: rows,
        });
    }
    let mut baseline: VisualBaselineFile = serde_json::from_value(value)?;
    if baseline.schema.trim().is_empty() {
        baseline.schema = "deepseek.visual.baseline.v1".to_string();
    }
    Ok(baseline)
}

pub(crate) fn write_visual_baseline_file(
    path: &Path,
    cwd: &Path,
    mut baseline: VisualBaselineFile,
) -> Result<()> {
    if baseline.schema.trim().is_empty() {
        baseline.schema = "deepseek.visual.baseline.v1".to_string();
    }
    if baseline.workspace.trim().is_empty() {
        baseline.workspace = cwd.to_string_lossy().to_string();
    }
    if baseline.generated_at.trim().is_empty() {
        baseline.generated_at = Utc::now().to_rfc3339();
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(&baseline)?)?;
    Ok(())
}

pub(crate) fn read_visual_expectation_file(path: &Path) -> Result<VisualExpectationFile> {
    let raw = fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    if let Some(rules) = value.as_array() {
        let parsed = rules
            .iter()
            .cloned()
            .map(serde_json::from_value::<VisualExpectationRule>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        return Ok(VisualExpectationFile {
            schema: "deepseek.visual.expectation.v1".to_string(),
            rules: parsed,
        });
    }
    let mut parsed: VisualExpectationFile = serde_json::from_value(value)?;
    if parsed.schema.trim().is_empty() {
        parsed.schema = "deepseek.visual.expectation.v1".to_string();
    }
    Ok(parsed)
}

pub(crate) fn evaluate_visual_semantics(
    entries: &[VisualSemanticEntry],
    expectation_path: &Path,
) -> Result<serde_json::Value> {
    let expectations = read_visual_expectation_file(expectation_path)?;
    let mut failures = Vec::new();
    let mut results = Vec::new();
    let mut passed = 0usize;

    for (idx, rule) in expectations.rules.iter().enumerate() {
        let label = rule
            .name
            .clone()
            .unwrap_or_else(|| format!("rule-{}", idx + 1));
        let glob = glob::Pattern::new(&rule.path_glob)
            .map_err(|err| anyhow!("invalid expectation glob '{}': {}", rule.path_glob, err))?;
        let mut matched = entries
            .iter()
            .filter(|entry| glob.matches(&entry.path))
            .collect::<Vec<_>>();
        matched.sort_by(|a, b| a.path.cmp(&b.path));
        let existing = matched
            .iter()
            .filter(|entry| entry.exists)
            .copied()
            .collect::<Vec<_>>();
        let min_count = rule.min_count.unwrap_or(1);
        let mut rule_failures = Vec::new();
        if existing.len() < min_count {
            rule_failures.push(format!(
                "{} matched {} artifacts (minimum required {})",
                label,
                existing.len(),
                min_count
            ));
        }
        if let Some(max_count) = rule.max_count
            && existing.len() > max_count
        {
            rule_failures.push(format!(
                "{} matched {} artifacts (maximum allowed {})",
                label,
                existing.len(),
                max_count
            ));
        }

        for entry in existing {
            if let Some(expected_mime) = rule.mime.as_deref()
                && !entry.mime.eq_ignore_ascii_case(expected_mime)
            {
                rule_failures.push(format!(
                    "{}:{} mime {} did not match {}",
                    label, entry.path, entry.mime, expected_mime
                ));
            }
            if let Some(min_bytes) = rule.min_bytes
                && entry.size_bytes < min_bytes
            {
                rule_failures.push(format!(
                    "{}:{} size {} below minimum {}",
                    label, entry.path, entry.size_bytes, min_bytes
                ));
            }
            if let Some(min_width) = rule.min_width {
                match entry.width {
                    Some(width) if width >= min_width => {}
                    Some(width) => rule_failures.push(format!(
                        "{}:{} width {} below minimum {}",
                        label, entry.path, width, min_width
                    )),
                    None => rule_failures.push(format!(
                        "{}:{} width unavailable for semantic check",
                        label, entry.path
                    )),
                }
            }
            if let Some(min_height) = rule.min_height {
                match entry.height {
                    Some(height) if height >= min_height => {}
                    Some(height) => rule_failures.push(format!(
                        "{}:{} height {} below minimum {}",
                        label, entry.path, height, min_height
                    )),
                    None => rule_failures.push(format!(
                        "{}:{} height unavailable for semantic check",
                        label, entry.path
                    )),
                }
            }
            if let Some(max_width) = rule.max_width
                && let Some(width) = entry.width
                && width > max_width
            {
                rule_failures.push(format!(
                    "{}:{} width {} above maximum {}",
                    label, entry.path, width, max_width
                ));
            }
            if let Some(max_height) = rule.max_height
                && let Some(height) = entry.height
                && height > max_height
            {
                rule_failures.push(format!(
                    "{}:{} height {} above maximum {}",
                    label, entry.path, height, max_height
                ));
            }
            for needle in &rule.required_path_substrings {
                let normalized = needle.trim().to_ascii_lowercase();
                if normalized.is_empty() {
                    continue;
                }
                if !entry.path.to_ascii_lowercase().contains(&normalized) {
                    rule_failures.push(format!(
                        "{}:{} path missing required semantic token '{}'",
                        label, entry.path, normalized
                    ));
                }
            }
        }

        let ok = rule_failures.is_empty();
        if ok {
            passed = passed.saturating_add(1);
        } else {
            failures.extend(rule_failures.iter().cloned());
        }
        results.push(json!({
            "name": label,
            "path_glob": rule.path_glob,
            "matched": matched.len(),
            "matched_existing": matched.iter().filter(|entry| entry.exists).count(),
            "ok": ok,
            "failures": rule_failures,
        }));
    }

    Ok(json!({
        "source": expectation_path,
        "schema": expectations.schema,
        "rules_total": expectations.rules.len(),
        "rules_passed": passed,
        "rules_failed": expectations.rules.len().saturating_sub(passed),
        "failures": failures,
        "results": results,
    }))
}

pub(crate) fn read_visual_dimensions(path: &Path, mime: &str) -> Option<(u32, u32)> {
    let bytes = fs::read(path).ok()?;
    if mime.eq_ignore_ascii_case("image/png") {
        return parse_png_dimensions(&bytes);
    }
    if mime.eq_ignore_ascii_case("image/jpeg") || mime.eq_ignore_ascii_case("image/jpg") {
        return parse_jpeg_dimensions(&bytes);
    }
    if mime.eq_ignore_ascii_case("image/gif") {
        return parse_gif_dimensions(&bytes);
    }
    parse_png_dimensions(&bytes)
        .or_else(|| parse_jpeg_dimensions(&bytes))
        .or_else(|| parse_gif_dimensions(&bytes))
}

pub(crate) fn parse_png_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 24 {
        return None;
    }
    let signature = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    if bytes[..8] != signature {
        return None;
    }
    let width = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let height = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    Some((width, height))
}

pub(crate) fn parse_gif_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 10 {
        return None;
    }
    if !bytes.starts_with(b"GIF87a") && !bytes.starts_with(b"GIF89a") {
        return None;
    }
    let width = u16::from_le_bytes([bytes[6], bytes[7]]) as u32;
    let height = u16::from_le_bytes([bytes[8], bytes[9]]) as u32;
    Some((width, height))
}

pub(crate) fn parse_jpeg_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 4 || bytes[0] != 0xff || bytes[1] != 0xd8 {
        return None;
    }
    let mut idx = 2usize;
    while idx + 9 < bytes.len() {
        if bytes[idx] != 0xff {
            idx += 1;
            continue;
        }
        while idx < bytes.len() && bytes[idx] == 0xff {
            idx += 1;
        }
        if idx >= bytes.len() {
            break;
        }
        let marker = bytes[idx];
        idx += 1;
        if marker == 0xd8 || marker == 0xd9 || marker == 0x01 || (0xd0..=0xd7).contains(&marker) {
            continue;
        }
        if idx + 1 >= bytes.len() {
            break;
        }
        let segment_len = ((bytes[idx] as usize) << 8) | bytes[idx + 1] as usize;
        if segment_len < 2 || idx + segment_len > bytes.len() {
            break;
        }
        if matches!(
            marker,
            0xc0 | 0xc1
                | 0xc2
                | 0xc3
                | 0xc5
                | 0xc6
                | 0xc7
                | 0xc9
                | 0xca
                | 0xcb
                | 0xcd
                | 0xce
                | 0xcf
        ) && segment_len >= 7
        {
            let height = u16::from_be_bytes([bytes[idx + 3], bytes[idx + 4]]) as u32;
            let width = u16::from_be_bytes([bytes[idx + 5], bytes[idx + 6]]) as u32;
            return Some((width, height));
        }
        idx += segment_len;
    }
    None
}
