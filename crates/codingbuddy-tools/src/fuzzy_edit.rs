use anyhow::{Result, anyhow};

/// Compute Levenshtein edit distance between two strings.
/// Normalized similarity between two strings (0.0 = completely different, 1.0 = identical).
fn levenshtein_similarity(a: &str, b: &str) -> f64 {
    strsim::normalized_levenshtein(a, b)
}

/// Result of a fuzzy match: byte offsets and the strategy name.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct FuzzyMatch {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) strategy: &'static str,
}

/// Try 7 fuzzy matching strategies in order when exact match fails.
/// Returns the first unambiguous match found.
fn fuzzy_match_search(content: &str, search: &str) -> Option<FuzzyMatch> {
    // Strategy 1: LineTrimmedReplacer — match lines ignoring leading/trailing whitespace per line
    if let Some(m) = fuzzy_line_trimmed(content, search) {
        return Some(m);
    }
    // Strategy 2: BlockAnchorReplacer — first+last lines as anchors, fuzzy middle
    if let Some(m) = fuzzy_block_anchor(content, search) {
        return Some(m);
    }
    // Strategy 3: WhitespaceNormalizedReplacer — collapse all whitespace to single spaces
    if let Some(m) = fuzzy_whitespace_normalized(content, search) {
        return Some(m);
    }
    // Strategy 4: IndentationFlexibleReplacer — strip common indentation prefix
    if let Some(m) = fuzzy_indentation_flexible(content, search) {
        return Some(m);
    }
    // Strategy 5: EscapeNormalizedReplacer — unescape \n, \t, \" before matching
    if let Some(m) = fuzzy_escape_normalized(content, search) {
        return Some(m);
    }
    // Strategy 6: TrimmedBoundaryReplacer — trim leading/trailing blank lines from search
    if let Some(m) = fuzzy_trimmed_boundary(content, search) {
        return Some(m);
    }
    // Strategy 7: ContextAwareReplacer — anchors + ≥50% middle line match rate
    if let Some(m) = fuzzy_context_aware(content, search) {
        return Some(m);
    }
    None
}

/// Convert a line-index range to byte offsets in the original content.
/// `line_idx` is the starting line, `line_count` is how many lines the match spans.
/// Each line's byte length includes the `\n` separator; the trailing newline of
/// the last matched line is excluded.
fn lines_to_byte_range(
    content_lines: &[&str],
    content_len: usize,
    line_idx: usize,
    line_count: usize,
) -> (usize, usize) {
    let start: usize = content_lines[..line_idx].iter().map(|l| l.len() + 1).sum();
    let end = start
        + content_lines[line_idx..line_idx + line_count]
            .iter()
            .map(|l| l.len() + 1)
            .sum::<usize>()
        - 1;
    (start, end.min(content_len))
}

/// Strategy 1: Match lines ignoring per-line leading/trailing whitespace.
pub(crate) fn fuzzy_line_trimmed(content: &str, search: &str) -> Option<FuzzyMatch> {
    let search_lines: Vec<&str> = search.lines().collect();
    if search_lines.is_empty() {
        return None;
    }
    let content_lines: Vec<&str> = content.lines().collect();
    if content_lines.len() < search_lines.len() {
        return None;
    }
    let trimmed_search: Vec<&str> = search_lines.iter().map(|l| l.trim()).collect();

    let mut matches = Vec::new();
    for i in 0..=content_lines.len() - search_lines.len() {
        let all_match =
            (0..search_lines.len()).all(|j| content_lines[i + j].trim() == trimmed_search[j]);
        if all_match {
            matches.push(i);
        }
    }

    if matches.len() != 1 {
        return None; // Ambiguous or no match
    }

    let line_idx = matches[0];
    let (start, end) =
        lines_to_byte_range(&content_lines, content.len(), line_idx, search_lines.len());

    Some(FuzzyMatch {
        start,
        end,
        strategy: "line_trimmed",
    })
}

/// Strategy 2: First+last lines as anchors, fuzzy-match middle (Levenshtein ≥ 0.3).
pub(crate) fn fuzzy_block_anchor(content: &str, search: &str) -> Option<FuzzyMatch> {
    let search_lines: Vec<&str> = search.lines().collect();
    if search_lines.len() < 3 {
        return None; // Need at least 3 lines for anchor strategy
    }
    let content_lines: Vec<&str> = content.lines().collect();
    if content_lines.len() < search_lines.len() {
        return None;
    }

    let first_trimmed = search_lines[0].trim();
    let last_trimmed = search_lines[search_lines.len() - 1].trim();
    if first_trimmed.is_empty() || last_trimmed.is_empty() {
        return None;
    }

    let mut matches = Vec::new();
    for i in 0..=content_lines.len() - search_lines.len() {
        let end_idx = i + search_lines.len() - 1;
        if content_lines[i].trim() != first_trimmed {
            continue;
        }
        if content_lines[end_idx].trim() != last_trimmed {
            continue;
        }
        // Check middle lines with Levenshtein similarity ≥ 0.3
        let middle_ok = (1..search_lines.len() - 1).all(|j| {
            levenshtein_similarity(content_lines[i + j].trim(), search_lines[j].trim()) >= 0.3
        });
        if middle_ok {
            matches.push(i);
        }
    }

    if matches.len() != 1 {
        return None;
    }

    let line_idx = matches[0];
    let (start, end) =
        lines_to_byte_range(&content_lines, content.len(), line_idx, search_lines.len());

    Some(FuzzyMatch {
        start,
        end,
        strategy: "block_anchor",
    })
}

/// Strategy 3: Collapse all whitespace to single spaces, then match.
pub(crate) fn fuzzy_whitespace_normalized(content: &str, search: &str) -> Option<FuzzyMatch> {
    let normalize = |s: &str| -> String {
        let mut result = String::with_capacity(s.len());
        let mut prev_ws = false;
        for ch in s.chars() {
            if ch.is_whitespace() {
                if !prev_ws {
                    result.push(' ');
                    prev_ws = true;
                }
            } else {
                result.push(ch);
                prev_ws = false;
            }
        }
        result
    };

    let norm_search = normalize(search);
    if norm_search.is_empty() {
        return None;
    }
    let norm_content = normalize(content);

    // Find all occurrences in normalized content
    let mut norm_matches = Vec::new();
    let mut start = 0;
    while let Some(pos) = norm_content[start..].find(&norm_search) {
        norm_matches.push(start + pos);
        start += pos + 1;
    }

    if norm_matches.len() != 1 {
        return None; // Ambiguous or no match
    }

    // Map normalized position back to original content position
    // Walk both strings in lockstep
    let norm_start = norm_matches[0];
    let norm_end = norm_start + norm_search.len();

    let mut norm_pos = 0;
    let mut orig_start = 0;
    let mut orig_end = 0;
    let mut prev_ws = false;

    for (i, ch) in content.char_indices() {
        if norm_pos == norm_start {
            orig_start = i;
        }
        if ch.is_whitespace() {
            if !prev_ws {
                norm_pos += 1; // One normalized space
                prev_ws = true;
            }
        } else {
            norm_pos += ch.len_utf8();
            prev_ws = false;
        }
        if norm_pos >= norm_end {
            orig_end = i + ch.len_utf8();
            break;
        }
    }
    if orig_end == 0 {
        orig_end = content.len();
    }

    Some(FuzzyMatch {
        start: orig_start,
        end: orig_end,
        strategy: "whitespace_normalized",
    })
}

/// Strategy 4: Strip common indentation prefix, match de-indented.
pub(crate) fn fuzzy_indentation_flexible(content: &str, search: &str) -> Option<FuzzyMatch> {
    let search_lines: Vec<&str> = search.lines().collect();
    if search_lines.is_empty() {
        return None;
    }

    // Find minimum indentation in search (ignoring empty lines)
    let search_indent = search_lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);

    let stripped_search: Vec<String> = search_lines
        .iter()
        .map(|l| {
            if l.trim().is_empty() {
                String::new()
            } else if l.len() >= search_indent {
                l[search_indent..].to_string()
            } else {
                l.to_string()
            }
        })
        .collect();

    let content_lines: Vec<&str> = content.lines().collect();
    if content_lines.len() < search_lines.len() {
        return None;
    }

    let mut matches = Vec::new();
    for i in 0..=content_lines.len() - search_lines.len() {
        // Find the actual indentation of this content block
        let block_indent = content_lines[i..i + search_lines.len()]
            .iter()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.len() - l.trim_start().len())
            .min()
            .unwrap_or(0);

        let all_match = (0..search_lines.len()).all(|j| {
            let cl = content_lines[i + j];
            let sl = &stripped_search[j];
            if cl.trim().is_empty() && sl.is_empty() {
                return true;
            }
            let cl_stripped = if cl.len() >= block_indent {
                &cl[block_indent..]
            } else {
                cl
            };
            cl_stripped == sl.as_str()
        });

        if all_match {
            matches.push(i);
        }
    }

    if matches.len() != 1 {
        return None;
    }

    let line_idx = matches[0];
    let (start, end) =
        lines_to_byte_range(&content_lines, content.len(), line_idx, search_lines.len());

    Some(FuzzyMatch {
        start,
        end,
        strategy: "indentation_flexible",
    })
}

/// Strategy 5: Unescape \n, \t, \" etc. before matching.
pub(crate) fn fuzzy_escape_normalized(content: &str, search: &str) -> Option<FuzzyMatch> {
    // Only apply if search actually contains escape sequences
    if !search.contains('\\') {
        return None;
    }

    let unescaped = search
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\"", "\"")
        .replace("\\'", "'")
        .replace("\\\\", "\\");

    if unescaped == search {
        return None; // No change after unescaping
    }

    // Try exact match with unescaped search
    let count = content.matches(&unescaped).count();
    if count != 1 {
        return None;
    }

    let pos = content.find(&unescaped)?;
    Some(FuzzyMatch {
        start: pos,
        end: pos + unescaped.len(),
        strategy: "escape_normalized",
    })
}

/// Strategy 6: Trim leading/trailing blank lines from search string.
pub(crate) fn fuzzy_trimmed_boundary(content: &str, search: &str) -> Option<FuzzyMatch> {
    let trimmed = search
        .trim_start_matches('\n')
        .trim_start_matches("\r\n")
        .trim_end_matches('\n')
        .trim_end_matches("\r\n");

    if trimmed == search || trimmed.is_empty() {
        return None; // No change after trimming, or empty
    }

    let count = content.matches(trimmed).count();
    if count != 1 {
        return None;
    }

    let pos = content.find(trimmed)?;
    Some(FuzzyMatch {
        start: pos,
        end: pos + trimmed.len(),
        strategy: "trimmed_boundary",
    })
}

/// Strategy 7: First+last lines as anchors, ≥50% middle line match rate.
pub(crate) fn fuzzy_context_aware(content: &str, search: &str) -> Option<FuzzyMatch> {
    let search_lines: Vec<&str> = search.lines().collect();
    if search_lines.len() < 3 {
        return None;
    }
    let content_lines: Vec<&str> = content.lines().collect();
    if content_lines.len() < search_lines.len() {
        return None;
    }

    let first_trimmed = search_lines[0].trim();
    let last_trimmed = search_lines[search_lines.len() - 1].trim();
    if first_trimmed.is_empty() || last_trimmed.is_empty() {
        return None;
    }

    let middle_count = search_lines.len() - 2;
    let required_matches = middle_count.div_ceil(2); // ≥50%

    let mut matches = Vec::new();
    for i in 0..=content_lines.len() - search_lines.len() {
        let end_idx = i + search_lines.len() - 1;
        if content_lines[i].trim() != first_trimmed {
            continue;
        }
        if content_lines[end_idx].trim() != last_trimmed {
            continue;
        }
        // Count middle lines that match (trimmed equality)
        let matched_middle = (1..search_lines.len() - 1)
            .filter(|&j| content_lines[i + j].trim() == search_lines[j].trim())
            .count();
        if matched_middle >= required_matches {
            matches.push(i);
        }
    }

    if matches.len() != 1 {
        return None;
    }

    let line_idx = matches[0];
    let (start, end) =
        lines_to_byte_range(&content_lines, content.len(), line_idx, search_lines.len());

    Some(FuzzyMatch {
        start,
        end,
        strategy: "context_aware",
    })
}

pub(crate) fn apply_single_edit(content: &mut String, edit: &serde_json::Value) -> Result<usize> {
    if let (Some(search), Some(replace)) = (
        edit.get("search").and_then(|v| v.as_str()),
        edit.get("replace").and_then(|v| v.as_str()),
    ) {
        let replace_all = edit.get("all").and_then(|v| v.as_bool()).unwrap_or(true);
        if replace_all {
            let count = content.matches(search).count();
            if count == 0 {
                // Fuzzy fallback for replace_all — try to find at least one match
                if let Some(fm) = fuzzy_match_search(content, search) {
                    content.replace_range(fm.start..fm.end, replace);
                    return Ok(1);
                }
                return Err(anyhow!("search pattern not found: {search}"));
            }
            *content = content.replace(search, replace);
            return Ok(count);
        }
        if let Some(pos) = content.find(search) {
            content.replace_range(pos..pos + search.len(), replace);
            return Ok(1);
        }
        // Fuzzy fallback chain for single replacement
        if let Some(fm) = fuzzy_match_search(content, search) {
            content.replace_range(fm.start..fm.end, replace);
            return Ok(1);
        }
        return Err(anyhow!("search pattern not found: {search}"));
    }

    if let (Some(start_line), Some(end_line), Some(replacement)) = (
        edit.get("start_line").and_then(|v| v.as_u64()),
        edit.get("end_line").and_then(|v| v.as_u64()),
        edit.get("replacement").and_then(|v| v.as_str()),
    ) {
        let start = start_line as usize;
        let end = end_line as usize;
        if start == 0 || end < start {
            return Err(anyhow!(
                "invalid line range: start_line={start_line} end_line={end_line}"
            ));
        }

        let had_trailing_newline = content.ends_with('\n');
        let mut lines = content.lines().map(ToString::to_string).collect::<Vec<_>>();
        if end > lines.len() {
            return Err(anyhow!(
                "line range out of bounds: end_line={end_line} file_lines={}",
                lines.len()
            ));
        }
        let replacement_lines = replacement
            .split('\n')
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        lines.splice((start - 1)..end, replacement_lines);
        *content = lines.join("\n");
        if had_trailing_newline {
            content.push('\n');
        }
        return Ok(1);
    }

    Err(anyhow!(
        "fs_edit requires either {{search, replace}} or {{start_line, end_line, replacement}}"
    ))
}
