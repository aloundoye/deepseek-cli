use anyhow::Result;
use ignore::WalkBuilder;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub(crate) struct PromptReference {
    pub(crate) raw: String,
    pub(crate) path: String,
    pub(crate) start_line: Option<usize>,
    pub(crate) end_line: Option<usize>,
    pub(crate) force_dir: bool,
}

pub(crate) fn expand_prompt_references(
    workspace: &Path,
    prompt: &str,
    respect_gitignore: bool,
) -> Result<String> {
    let refs = extract_prompt_references(prompt);
    if refs.is_empty() {
        return Ok(prompt.to_string());
    }

    let mut extra = String::new();
    extra.push_str("\n\n[Resolved references]\n");
    for reference in refs {
        let full = workspace.join(&reference.path);
        if !full.exists() {
            extra.push_str(&format!(
                "- {} -> missing ({})\n",
                reference.raw, reference.path
            ));
            continue;
        }

        if reference.force_dir || full.is_dir() {
            let mut shown = 0usize;
            extra.push_str(&format!(
                "- {} -> directory {}\n",
                reference.raw, reference.path
            ));
            for rel in walk_workspace_files(workspace, &full, respect_gitignore, 50) {
                extra.push_str(&format!("  - {rel}\n"));
                shown += 1;
            }
            if shown >= 50 {
                extra.push_str("  - ... (truncated)\n");
            }
            continue;
        }

        let bytes = fs::read(&full)?;
        if is_binary(&bytes) {
            extra.push_str(&format!(
                "- {} -> file {} (binary, {} bytes)\n",
                reference.raw,
                reference.path,
                bytes.len()
            ));
            continue;
        }

        let text = String::from_utf8(bytes)?;
        let rendered = render_referenced_lines(&text, reference.start_line, reference.end_line);
        extra.push_str(&format!(
            "```text\n# {}\n{}\n```\n",
            reference.path, rendered
        ));
    }

    Ok(format!("{prompt}{extra}"))
}

pub(crate) fn extract_prompt_references(prompt: &str) -> Vec<PromptReference> {
    prompt
        .split_whitespace()
        .filter_map(parse_prompt_reference)
        .collect()
}

pub(crate) fn parse_prompt_reference(token: &str) -> Option<PromptReference> {
    let trimmed = token.trim();
    if !trimmed.starts_with('@') {
        return None;
    }
    let body = trimmed
        .trim_start_matches('@')
        .trim_end_matches(|ch: char| {
            ch == ',' || ch == '.' || ch == ';' || ch == ')' || ch == ']' || ch == '>'
        });
    if body.is_empty() {
        return None;
    }

    let mut path = body.to_string();
    let mut start_line = None;
    let mut end_line = None;
    let mut force_dir = false;

    if let Some(rest) = body.strip_prefix("dir:") {
        path = rest.to_string();
        force_dir = true;
    } else if let Some(rest) = body.strip_prefix("file:") {
        path = rest.to_string();
    }
    if let Some(idx) = path.rfind(':') {
        let (candidate_path, range) = path.split_at(idx);
        let range = &range[1..];
        if let Some((start, end)) = parse_line_range(range) {
            path = candidate_path.to_string();
            start_line = Some(start);
            end_line = Some(end);
        }
    }

    Some(PromptReference {
        raw: trimmed.to_string(),
        path,
        start_line,
        end_line,
        force_dir,
    })
}

pub(crate) fn parse_line_range(range: &str) -> Option<(usize, usize)> {
    if range.is_empty() {
        return None;
    }
    if let Some((start, end)) = range.split_once('-') {
        let start = start.parse::<usize>().ok()?;
        let end = end.parse::<usize>().ok()?;
        if start == 0 || end < start {
            return None;
        }
        return Some((start, end));
    }
    let line = range.parse::<usize>().ok()?;
    if line == 0 {
        return None;
    }
    Some((line, line))
}

pub(crate) fn render_referenced_lines(
    content: &str,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> String {
    let start = start_line.unwrap_or(1).max(1);
    let end = end_line.unwrap_or(start + 200).max(start);
    let mut out = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        let line_no = idx + 1;
        if line_no < start || line_no > end {
            continue;
        }
        out.push(format!("{line_no:>5}: {line}"));
        if out.len() >= 200 {
            out.push("... (truncated)".to_string());
            break;
        }
    }
    out.join("\n")
}

pub(crate) fn is_binary(bytes: &[u8]) -> bool {
    bytes.contains(&0)
}

pub(crate) fn has_ignored_component(path: &Path) -> bool {
    path.components().any(|component| {
        let value = component.as_os_str();
        value == ".git" || value == ".deepseek" || value == "target"
    })
}

pub(crate) fn walk_workspace_files(
    workspace: &Path,
    root: &Path,
    respect_gitignore: bool,
    limit: usize,
) -> Vec<String> {
    let mut builder = WalkBuilder::new(root);
    builder.hidden(false);
    builder.follow_links(false);
    builder.parents(respect_gitignore);
    builder.git_ignore(respect_gitignore);
    builder.git_global(respect_gitignore);
    builder.git_exclude(respect_gitignore);
    builder.require_git(false);

    let mut out = Vec::new();
    for entry in builder.build() {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Ok(rel) = path.strip_prefix(workspace) else {
            continue;
        };
        if has_ignored_component(rel) {
            continue;
        }
        out.push(rel.to_string_lossy().replace('\\', "/"));
        if out.len() >= limit {
            break;
        }
    }
    out
}
