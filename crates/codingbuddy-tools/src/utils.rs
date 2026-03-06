use super::*;

pub(crate) fn should_skip_rel_path(path: &Path) -> bool {
    path.components().any(|c| {
        c.as_os_str() == ".git" || c.as_os_str() == ".codingbuddy" || c.as_os_str() == "target"
    })
}

pub(crate) fn walk_paths(root: &Path, workspace: &Path, respect_gitignore: bool) -> Vec<PathBuf> {
    let mut builder = WalkBuilder::new(root);
    builder.hidden(false);
    builder.follow_links(false);
    builder.parents(respect_gitignore);
    builder.git_ignore(respect_gitignore);
    builder.git_global(respect_gitignore);
    builder.git_exclude(respect_gitignore);
    builder.require_git(false);
    builder.add_custom_ignore_filename(".codingbuddyignore");

    let mut paths = Vec::new();
    for entry in builder.build() {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        let Ok(rel) = path.strip_prefix(workspace) else {
            continue;
        };
        if should_skip_rel_path(rel) {
            continue;
        }
        paths.push(path.to_path_buf());
    }
    paths
}

pub(crate) fn normalize_rel_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

pub(crate) fn is_binary(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    if bytes.contains(&0) {
        return true;
    }
    let sample = bytes.iter().take(8192);
    let non_text = sample
        .filter(|b| !(b.is_ascii() || **b == b'\n' || **b == b'\r' || **b == b'\t'))
        .count();
    non_text > 64
}

pub(crate) fn guess_mime(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "md" => "text/markdown",
        "txt" | "log" | "rs" | "toml" | "json" | "yaml" | "yml" | "js" | "ts" | "tsx" | "jsx"
        | "py" | "go" | "java" | "c" | "h" | "cpp" | "hpp" | "cs" | "sh" | "ps1" => "text/plain",
        "pdf" => "application/pdf",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        "ipynb" => "application/x-ipynb+json",
        _ => "application/octet-stream",
    }
}

pub(crate) fn collect_lines(
    content: &str,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> Vec<serde_json::Value> {
    let start = start_line.unwrap_or(1).max(1);
    let end = end_line.unwrap_or(usize::MAX).max(start);
    content
        .lines()
        .enumerate()
        .filter_map(|(idx, text)| {
            let line = idx + 1;
            if line < start || line > end {
                return None;
            }
            Some(json!({
                "line": line,
                "text": text
            }))
        })
        .collect()
}

pub(crate) fn extract_pdf_text(path: &Path, pages: Option<&str>) -> Result<String> {
    let full_text =
        pdf_extract::extract_text(path).map_err(|e| anyhow!("PDF extraction failed: {e}"))?;
    match pages {
        None => Ok(full_text),
        Some(range) => {
            let (start, end) = parse_page_range(range)?;
            let page_texts: Vec<&str> = full_text.split('\x0C').collect();
            let total = page_texts.len();
            if start > total {
                return Err(anyhow!("page {start} out of range ({total} pages)"));
            }
            let end_idx = end.min(total);
            Ok(page_texts[(start - 1)..end_idx].join("\n--- page break ---\n"))
        }
    }
}

pub(crate) fn parse_page_range(range: &str) -> Result<(usize, usize)> {
    if let Some((s, e)) = range.split_once('-') {
        let start: usize = s
            .trim()
            .parse()
            .map_err(|_| anyhow!("invalid range start"))?;
        let end: usize = e.trim().parse().map_err(|_| anyhow!("invalid range end"))?;
        if start == 0 || end < start {
            return Err(anyhow!("invalid page range"));
        }
        Ok((start, end))
    } else {
        let p: usize = range
            .trim()
            .parse()
            .map_err(|_| anyhow!("invalid page number"))?;
        if p == 0 {
            return Err(anyhow!("page must be >= 1"));
        }
        Ok((p, p))
    }
}

/// Extract readable text from HTML, stripping tags, scripts, and styles.
/// Uses the `scraper` crate for proper DOM-based extraction.
pub(crate) fn strip_html_tags(html: &str) -> String {
    use scraper::{Html, Selector};

    let document = Html::parse_document(html);

    let body_sel = Selector::parse("body").unwrap_or_else(|_| Selector::parse("*").unwrap());

    let mut out = String::with_capacity(html.len() / 2);

    if let Some(element) = document.select(&body_sel).next() {
        for text in element.text() {
            // Skip text that is inside script/style by checking ancestors
            // (scraper's .text() already skips script/style content in well-formed HTML)
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                out.push_str(trimmed);
                out.push('\n');
            }
        }
    }

    // If no <body> found, fall back to full document text
    if out.is_empty() {
        // No <body> found — extract text from root element
        // (.text() already skips script/style content in well-formed HTML)
        for text_node in document.root_element().text() {
            let trimmed = text_node.trim();
            if !trimmed.is_empty() {
                out.push_str(trimmed);
                out.push('\n');
            }
        }
    }

    // Collapse multiple blank lines
    let mut result = String::with_capacity(out.len());
    let mut blank_count = 0;
    for line in out.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            blank_count += 1;
            if blank_count <= 1 {
                result.push('\n');
            }
        } else {
            blank_count = 0;
            result.push_str(trimmed);
            result.push('\n');
        }
    }
    result.trim().to_string()
}

/// Parse search results from DuckDuckGo HTML response using CSS selectors.
/// Falls back to legacy string matching if selector-based parsing fails.
pub(crate) fn parse_search_results(html: &str, max_results: usize) -> Vec<serde_json::Value> {
    let results = parse_search_results_css(html, max_results);
    if results.is_empty() {
        // Fallback to legacy string-matching parser
        parse_search_results_legacy(html, max_results)
    } else {
        results
    }
}

/// CSS selector-based DuckDuckGo HTML parser (primary).
pub(crate) fn parse_search_results_css(html: &str, max_results: usize) -> Vec<serde_json::Value> {
    use scraper::{Html, Selector};

    let document = Html::parse_document(html);
    let mut results = Vec::new();

    // DuckDuckGo result containers
    let result_sel = match Selector::parse(".result, .web-result") {
        Ok(s) => s,
        Err(_) => return results,
    };
    let link_sel = Selector::parse(".result__a, .result-title-a, a.result__a")
        .unwrap_or_else(|_| Selector::parse("a").unwrap());
    let snippet_sel = Selector::parse(".result__snippet, .result-snippet")
        .unwrap_or_else(|_| Selector::parse(".snippet").unwrap());

    for element in document.select(&result_sel) {
        if results.len() >= max_results {
            break;
        }

        // Extract link and title
        if let Some(link_el) = element.select(&link_sel).next() {
            let url = link_el.value().attr("href").unwrap_or_default().to_string();
            let title: String = link_el.text().collect::<Vec<_>>().join(" ");
            let title = title.trim().to_string();

            // Extract snippet
            let snippet = if let Some(snippet_el) = element.select(&snippet_sel).next() {
                snippet_el
                    .text()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .trim()
                    .to_string()
            } else {
                String::new()
            };

            if !url.is_empty() && !title.is_empty() {
                results.push(json!({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }));
            }
        }
    }
    results
}

/// Legacy string-matching parser (fallback).
pub(crate) fn parse_search_results_legacy(
    html: &str,
    max_results: usize,
) -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    let lower = html.to_ascii_lowercase();
    let mut pos = 0;
    while results.len() < max_results {
        let link_marker = "class=\"result__a\"";
        let link_pos = match lower[pos..].find(link_marker) {
            Some(p) => pos + p,
            None => break,
        };
        let href_start = match html[..link_pos].rfind("href=\"") {
            Some(p) => p + 6,
            None => {
                pos = link_pos + link_marker.len();
                continue;
            }
        };
        let href_end = match html[href_start..].find('"') {
            Some(p) => href_start + p,
            None => {
                pos = link_pos + link_marker.len();
                continue;
            }
        };
        let url = html[href_start..href_end].to_string();
        let tag_close = match html[link_pos..].find('>') {
            Some(p) => link_pos + p + 1,
            None => {
                pos = link_pos + link_marker.len();
                continue;
            }
        };
        let tag_end = match html[tag_close..].find("</a>") {
            Some(p) => tag_close + p,
            None => {
                pos = link_pos + link_marker.len();
                continue;
            }
        };
        let title = strip_html_tags(&html[tag_close..tag_end]);
        let snippet_marker = "class=\"result__snippet\"";
        let snippet = if let Some(sp) = lower[tag_end..].find(snippet_marker) {
            let snippet_pos = tag_end + sp;
            let snippet_start = match html[snippet_pos..].find('>') {
                Some(p) => snippet_pos + p + 1,
                None => tag_end,
            };
            let snippet_end = match html[snippet_start..].find("</") {
                Some(p) => snippet_start + p,
                None => snippet_start,
            };
            strip_html_tags(&html[snippet_start..snippet_end])
        } else {
            String::new()
        };
        if !url.is_empty() && !title.is_empty() {
            results.push(json!({
                "title": title.trim(),
                "url": url,
                "snippet": snippet.trim(),
            }));
        }
        pos = tag_end;
    }
    results
}

/// URL-encode a string (percent encoding). Simple implementation for search queries.
pub(crate) fn url_encode_query(input: &str) -> String {
    let mut encoded = String::with_capacity(input.len() * 3);
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            b' ' => encoded.push('+'),
            _ => {
                encoded.push('%');
                encoded.push_str(&format!("{byte:02X}"));
            }
        }
    }
    encoded
}

pub(crate) fn generate_unified_diff(path: &str, before: &str, after: &str) -> String {
    let old_lines: Vec<&str> = before.lines().collect();
    let new_lines: Vec<&str> = after.lines().collect();
    let n = old_lines.len();
    let m = new_lines.len();

    // LCS dynamic programming table
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            if old_lines[i - 1] == new_lines[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    // Backtrack to produce edit operations: Equal / Delete / Insert
    #[derive(Clone, Copy, PartialEq)]
    enum Op {
        Equal,
        Delete,
        Insert,
    }
    let mut ops: Vec<(Op, usize)> = Vec::new(); // (op, line index in old or new)
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && old_lines[i - 1] == new_lines[j - 1] {
            ops.push((Op::Equal, i - 1));
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || dp[i][j - 1] >= dp[i - 1][j]) {
            ops.push((Op::Insert, j - 1));
            j -= 1;
        } else {
            ops.push((Op::Delete, i - 1));
            i -= 1;
        }
    }
    ops.reverse();

    // Group into hunks with 3 lines of context
    let context = 3usize;
    let mut hunks: Vec<(usize, usize)> = Vec::new(); // (start, end) indices into ops
    let mut hunk_start: Option<usize> = None;
    let mut last_change: Option<usize> = None;

    for (idx, &(op, _)) in ops.iter().enumerate() {
        if op != Op::Equal {
            if let Some(prev) = last_change {
                if idx.saturating_sub(prev) > context * 2 {
                    // Close previous hunk
                    let end = (prev + context).min(ops.len() - 1);
                    hunks.push((hunk_start.unwrap(), end));
                    hunk_start = Some(idx.saturating_sub(context));
                }
            } else {
                hunk_start = Some(idx.saturating_sub(context));
            }
            last_change = Some(idx);
        }
    }
    if let (Some(start), Some(prev)) = (hunk_start, last_change) {
        let end = (prev + context).min(ops.len() - 1);
        hunks.push((start, end));
    }

    let mut out = format!("--- a/{path}\n+++ b/{path}\n");
    for (hunk_start, hunk_end) in &hunks {
        // Count old/new line numbers at hunk boundaries
        let mut old_start = 1usize;
        let mut new_start = 1usize;
        for &(op, _) in ops.iter().take(*hunk_start) {
            match op {
                Op::Equal | Op::Delete => old_start += 1,
                _ => {}
            }
            match op {
                Op::Equal | Op::Insert => new_start += 1,
                _ => {}
            }
        }
        let mut old_count = 0usize;
        let mut new_count = 0usize;
        for &(op, _) in ops.iter().take(hunk_end + 1).skip(*hunk_start) {
            match op {
                Op::Equal | Op::Delete => old_count += 1,
                _ => {}
            }
            match op {
                Op::Equal | Op::Insert => new_count += 1,
                _ => {}
            }
        }
        out.push_str(&format!(
            "@@ -{},{} +{},{} @@\n",
            old_start, old_count, new_start, new_count
        ));
        for &(op, line_idx) in ops.iter().take(hunk_end + 1).skip(*hunk_start) {
            match op {
                Op::Equal => out.push_str(&format!(" {}\n", old_lines[line_idx])),
                Op::Delete => out.push_str(&format!("-{}\n", old_lines[line_idx])),
                Op::Insert => out.push_str(&format!("+{}\n", new_lines[line_idx])),
            }
        }
    }
    out
}

/// Run auto-diagnostics after an edit and inject results into the tool result JSON.
pub(crate) fn maybe_run_auto_diagnostics(
    runner: &dyn ShellRunner,
    workspace: &Path,
    target: Option<&str>,
    result: &mut serde_json::Value,
) {
    let Ok((cmd, source)) = detect_diagnostics_command(workspace, target) else {
        return;
    };
    let Ok(diag_result) = runner.run(&cmd, workspace, Duration::from_secs(30)) else {
        return;
    };
    let diagnostics = parse_diagnostics(&diag_result.stdout, &diag_result.stderr, source);
    if diagnostics.is_empty() {
        return;
    }
    result["diagnostics"] = json!(diagnostics);
    let msg: Vec<String> = diagnostics
        .iter()
        .map(|d| {
            let file = d.get("file").and_then(|v| v.as_str()).unwrap_or("?");
            let line = d.get("line").and_then(|v| v.as_u64()).unwrap_or(0);
            let text = d.get("message").and_then(|v| v.as_str()).unwrap_or("?");
            format!("  {file}:{line}: {text}")
        })
        .take(10)
        .collect();
    result["diagnostics_message"] = json!(format!(
        "Errors detected after edit — please fix:\n{}",
        msg.join("\n")
    ));
}

pub(crate) fn detect_diagnostics_command(
    workspace: &Path,
    target: Option<&str>,
) -> Result<(String, &'static str)> {
    if workspace.join("Cargo.toml").exists() {
        // cargo check always checks the whole workspace; target is unused for Rust.
        return Ok((
            "cargo check --message-format=json 2>&1".to_string(),
            "rustc",
        ));
    }
    if workspace.join("tsconfig.json").exists() {
        return Ok(("npx tsc --noEmit --pretty false 2>&1".to_string(), "tsc"));
    }
    if workspace.join("pyproject.toml").exists() || workspace.join("setup.py").exists() {
        let cmd = match target {
            Some(path) => format!("ruff check {path} --output-format json 2>&1"),
            None => "ruff check . --output-format json 2>&1".to_string(),
        };
        return Ok((cmd, "ruff"));
    }
    Err(anyhow!("no supported language project detected"))
}

pub(crate) fn parse_diagnostics(
    stdout: &str,
    stderr: &str,
    source: &str,
) -> Vec<serde_json::Value> {
    match source {
        "rustc" => parse_cargo_check_json(stdout),
        "tsc" => parse_tsc_output(stderr),
        "ruff" => parse_ruff_json(stdout),
        _ => vec![],
    }
}

pub(crate) fn parse_cargo_check_json(output: &str) -> Vec<serde_json::Value> {
    let mut diagnostics = Vec::new();
    for line in output.lines() {
        let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if entry.get("reason").and_then(|v| v.as_str()) != Some("compiler-message") {
            continue;
        }
        let Some(message) = entry.get("message") else {
            continue;
        };
        let level = message
            .get("level")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let text = message
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let spans = message.get("spans").and_then(|v| v.as_array());
        let (file, line_num, col) = if let Some(spans) = spans
            && let Some(span) = spans.first()
        {
            (
                span.get("file_name").and_then(|v| v.as_str()).unwrap_or(""),
                span.get("line_start").and_then(|v| v.as_u64()).unwrap_or(0),
                span.get("column_start")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
            )
        } else {
            ("", 0, 0)
        };
        diagnostics.push(json!({
            "level": level,
            "message": text,
            "file": file,
            "line": line_num,
            "column": col,
        }));
    }
    diagnostics
}

pub(crate) fn parse_tsc_output(output: &str) -> Vec<serde_json::Value> {
    let mut diagnostics = Vec::new();
    let re = regex::Regex::new(r"^(.+)\((\d+),(\d+)\):\s+error\s+(TS\d+):\s+(.+)$").ok();
    let Some(re) = re else {
        return diagnostics;
    };
    for line in output.lines() {
        if let Some(caps) = re.captures(line) {
            diagnostics.push(json!({
                "level": "error",
                "message": caps.get(5).map(|m| m.as_str()).unwrap_or(""),
                "file": caps.get(1).map(|m| m.as_str()).unwrap_or(""),
                "line": caps.get(2).map(|m| m.as_str()).unwrap_or("0").parse::<u64>().unwrap_or(0),
                "column": caps.get(3).map(|m| m.as_str()).unwrap_or("0").parse::<u64>().unwrap_or(0),
                "code": caps.get(4).map(|m| m.as_str()).unwrap_or(""),
            }));
        }
    }
    diagnostics
}

pub(crate) fn parse_ruff_json(output: &str) -> Vec<serde_json::Value> {
    let mut diagnostics = Vec::new();
    let Ok(entries) = serde_json::from_str::<Vec<serde_json::Value>>(output) else {
        return diagnostics;
    };
    for entry in entries {
        let file = entry.get("filename").and_then(|v| v.as_str()).unwrap_or("");
        let code = entry.get("code").and_then(|v| v.as_str()).unwrap_or("");
        let message = entry.get("message").and_then(|v| v.as_str()).unwrap_or("");
        let location = entry.get("location");
        let line_num = location
            .and_then(|l| l.get("row"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let col = location
            .and_then(|l| l.get("column"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        diagnostics.push(json!({
            "level": "warning",
            "message": message,
            "file": file,
            "line": line_num,
            "column": col,
            "code": code,
        }));
    }
    diagnostics
}
