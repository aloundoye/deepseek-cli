use crate::planner::memory::{format_objective_outcomes, format_strategy_entries};
use crate::*;

impl AgentEngine {
    pub(crate) fn build_chat_system_prompt(&self, _user_prompt: &str) -> Result<String> {
        let now = Utc::now();

        // Try to get git branch
        let git_info = std::process::Command::new("git")
            .args(["branch", "--show-current"])
            .current_dir(&self.workspace)
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
                } else {
                    None
                }
            })
            .map(|branch| format!("Git branch: {branch}\n"))
            .unwrap_or_default();

        let mut parts = vec![format!(
            "{}\nDate: {}\n{git_info}",
            self.cached_system_prompt_base,
            now.format("%Y-%m-%d"),
        )];

        // Add DEEPSEEK.md / memory if available
        if let Ok(manager) = MemoryManager::new(&self.workspace)
            && let Ok(memory) = manager.read_combined_memory()
            && !memory.trim().is_empty()
        {
            parts.push(format!("\n\n[Project Memory]\n{memory}"));
        }

        // Add recent verification feedback
        let session = self.ensure_session()?;
        let verification_feedback = self
            .store
            .list_recent_verification_runs(session.session_id, 5)
            .unwrap_or_default()
            .into_iter()
            .filter(|run| !run.success)
            .take(3)
            .collect::<Vec<_>>();
        if !verification_feedback.is_empty() {
            parts.push(format!(
                "\n\n[Recent Verification Failures]\n{}",
                format_verification_feedback(&verification_feedback)
            ));
        }

        Ok(parts.join(""))
    }

    pub(crate) fn build_static_system_prompt_base(
        workspace: &Path,
        cfg: &AppConfig,
        policy: &PolicyEngine,
    ) -> String {
        let ws = workspace.to_string_lossy();

        // Detect project type
        let mut project_markers = Vec::new();
        for (file, lang) in &[
            ("Cargo.toml", "Rust"),
            ("package.json", "JavaScript/TypeScript"),
            ("pyproject.toml", "Python"),
            ("go.mod", "Go"),
            ("pom.xml", "Java/Maven"),
            ("build.gradle", "Java/Gradle"),
            ("Gemfile", "Ruby"),
            ("composer.json", "PHP"),
            ("CMakeLists.txt", "C/C++"),
            ("Makefile", "Make"),
        ] {
            if workspace.join(file).exists() {
                project_markers.push(*lang);
            }
        }
        let project_info = if project_markers.is_empty() {
            String::new()
        } else {
            format!("Project type: {}\n", project_markers.join(", "))
        };

        let shell = std::env::var("SHELL").unwrap_or_else(|_| "unknown".to_string());
        let permission_mode = policy.permission_mode();
        let base_model = &cfg.llm.base_model;
        let max_model = &cfg.llm.max_think_model;

        let mut base = format!(
            "You are DeepSeek, an AI coding assistant powering the `deepseek` CLI. \
             You help users with software engineering tasks including writing code, \
             debugging, refactoring, testing, and explaining codebases.\n\n\
             # Environment\n\
             Working directory: {ws}\n\
             Platform: {} ({})\n\
             Shell: {shell}\n\
             {project_info}\
             Models: {base_model} (fast) / {max_model} (reasoning)\n\
             Permission mode: {permission_mode:?}\n\n\
             # Permission Mode\n\
             Current mode is **{permission_mode:?}**.\n\
             - Ask: tool calls that modify files or run commands require user approval\n\
             - Auto: tool calls matching the allowlist are auto-approved\n\
             - Locked: all non-read operations are denied — read-only session\n\
             Adjust your approach accordingly. In Locked mode, only use read tools.\n\n\
             # Tool Usage Guidelines\n\
             - **fs_read**: Always read a file before editing it. Use `start_line`/`end_line` for large files. Supports images and PDFs (use `pages` for PDFs).\n\
             - **fs_edit**: Use exact `search` strings to make precise edits. The search string must appear verbatim in the file. Set `all: false` for first-occurrence-only.\n\
             - **fs_write**: Only for creating new files. Prefer `fs_edit` to modify existing files.\n\
             - **fs_glob**: Find files by pattern (e.g. `**/*.rs`). Use before grep to scope searches.\n\
             - **fs_grep**: Search file contents with regex. Use `case_sensitive: false` for case-insensitive. Use `glob` to filter file types.\n\
             - **bash_run**: Execute shell commands (git, build, test, etc.). Commands have a timeout (default 120s). \
               **Important restrictions**: Shell metacharacters (`&&`, `||`, `;`, backticks, `$()`) are FORBIDDEN and will be rejected. \
               A single pipeline (`|`) is allowed only when each segment is allowlisted. \
               Only allowlisted commands are permitted (e.g. `cargo`, `git`, `rg`, `find`, `head`). \
               For file discovery use `fs_glob`, for content search use `fs_grep`, for reading files use `fs_read`. \
               Prefer dedicated tools for exploration: avoid `ls`, `cat`, `tail`, `grep` via bash unless explicitly allowlisted.\n\
             - **multi_edit**: Batch edits across multiple files in one call. Each entry has a path and search/replace pairs.\n\
             - **git_status / git_diff / git_show**: Inspect repository state, diffs, and specific commits.\n\
             - **web_fetch**: Retrieve URL content as text. Use for documentation lookup.\n\
             - **web_search**: Search the web and return structured results.\n\
             - **index_query**: Full-text code search across the indexed codebase.\n\
             - **diagnostics_check**: Run language-specific diagnostics (cargo check, tsc, ruff, etc.).\n\
             - **patch_stage / patch_apply**: Stage and apply unified diffs with SHA verification.\n\
             - **notebook_read / notebook_edit**: Read and modify Jupyter notebooks.\n\n\
             Important: When you need to use a tool, invoke it through the function calling API.\n\
             Do NOT output tool names or arguments as text in your response.\n\n\
             # Safety Rules\n\
             - Always read a file before editing it — understand existing code first\n\
             - Never delete files, force-push, or run destructive commands without explicit user approval\n\
             - Stay within the working directory unless told otherwise\n\
             - Do not modify files outside the project without asking\n\
             - Do not introduce security vulnerabilities (command injection, XSS, SQL injection)\n\
             - Do not commit files containing secrets (.env, credentials, API keys)\n\n\
             # Git Protocol\n\
             - Create new commits — never amend unless explicitly asked\n\
             - Never push without user confirmation\n\
             - Use descriptive commit messages that explain the \"why\"\n\
             - Stage specific files by name, not `git add -A` or `git add .`\n\
             - Never use --force, --no-verify, or destructive git commands without asking\n\n\
             # Error Recovery\n\
             - If a tool call fails, read the error message carefully\n\
             - Try a different approach rather than repeating the same call\n\
             - Re-read the file after an edit failure to check current state\n\
             - If blocked, explain the issue to the user rather than brute-forcing\n\n\
             # Style\n\
             - Be concise and focused — avoid over-engineering\n\
             - Use markdown formatting in responses\n\
             - Reference files with path:line_number format\n\
             - When done, briefly explain what you changed and why",
            std::env::consts::OS,
            std::env::consts::ARCH,
        );

        // ── Enriched guidance: decision trees, workflows, and examples ──
        base.push_str(
            "\n\n# Tool Selection Decision Tree\n\
             When searching for code:\n\
             - Know the exact file path? → `fs_read`\n\
             - Know a regex pattern? → `fs_grep` (add `glob` to narrow file types)\n\
             - Know a filename pattern? → `fs_glob`\n\
             - Need semantic code search? → `index_query`\n\
             - Need to explore directory structure? → `fs_list`\n\n\
             When modifying code:\n\
             - Small targeted edit (one search/replace)? → `fs_edit`\n\
             - Multiple edits across files? → `multi_edit`\n\
             - Creating a brand new file? → `fs_write`\n\
             - Need to rename/move files? → `bash_run` with `git mv`\n\n\
             When debugging:\n\
             - Get exact error first → `bash_run` with build/test command\n\
             - Read the failing code → `fs_read` with relevant line range\n\
             - Search for related code → `fs_grep` for function/type names\n\
             - Apply fix → `fs_edit` with precise search string\n\
             - Verify fix → `bash_run` to rebuild/retest\n\n\
             # Workflow Patterns\n\n\
             ## Read-Before-Write (ALWAYS follow this)\n\
             1. `fs_read` the file to understand current content\n\
             2. Plan your edit based on the actual file content\n\
             3. `fs_edit` with an exact search string from the file\n\
             4. Verify with `diagnostics_check` or `bash_run`\n\n\
             ## Investigate-Before-Fix\n\
             1. `fs_grep` / `fs_glob` to find relevant files\n\
             2. `fs_read` to understand the code\n\
             3. `bash_run` to reproduce the issue\n\
             4. `fs_edit` to apply the fix\n\
             5. `bash_run` to verify the fix works\n\n\
             ## Multi-File Changes\n\
             1. Plan all changes before starting\n\
             2. Use `multi_edit` for related edits across 2+ files\n\
             3. Run `diagnostics_check` after all edits\n\
             4. Use `git_diff` to review changes before committing\n\n\
             # Error Recovery Guide\n\
             - `fs_edit` fails \"search string not found\" → `fs_read` the file first to get exact content\n\
             - `bash_run` times out → break into smaller commands or increase timeout\n\
             - Same approach fails twice → do NOT repeat it. Try a different strategy\n\
             - Stuck after 2-3 attempts → use `think_deeply` to get R1 reasoning advice\n\
             - Compilation error → read the error, find the exact line, fix it, rebuild\n\n\
             # Efficiency Guidelines\n\
             - Combine related reads: read relevant files before starting edits\n\
             - Use `fs_grep` with `glob` patterns to narrow searches (e.g., `glob: \"*.rs\"`)\n\
             - Prefer `fs_edit` over `fs_write` for existing files — it preserves unchanged content\n\
             - Use `multi_edit` when editing 2+ files for the same logical change\n\
             - If you need a tool not in your current set, use `tool_search` to discover it\n\n\
             # Plan-Aware Execution\n\
             When a <plan-context> message appears, it describes the current step in a structured plan:\n\
             - Focus on completing the current step before moving to the next\n\
             - Use the tools and files listed for that step\n\
             When a <verification-failure> message appears:\n\
             - Read the error carefully and identify the root cause\n\
             - Make targeted fixes to address the specific failure\n\
             - Do NOT repeat the same fix if it already failed",
        );

        // Add language preference if set.
        if !cfg.language.is_empty() {
            base.push_str(&format!("\n\n# Language\nRespond in **{}**.", cfg.language));
        }

        // Add output style directive.
        match cfg.output_style.as_str() {
            "concise" => {
                base.push_str(
                    "\n\n# Output Style\nBe extremely concise. Minimize explanation. Code only when possible.",
                );
            }
            "verbose" => {
                base.push_str(
                    "\n\n# Output Style\nBe thorough and detailed. Explain reasoning, trade-offs, and alternatives.",
                );
            }
            _ => {} // "normal" or unrecognized — no override
        }

        base
    }

    pub(crate) fn augment_prompt_context(&self, session_id: Uuid, prompt: &str) -> Result<String> {
        let projection = self.store.rebuild_from_events(session_id)?;
        let transcript = projection.transcript;
        let transcript_text = transcript.join("\n");
        let transcript_tokens = estimate_tokens(&transcript_text);
        let effective_window = self
            .cfg
            .llm
            .context_window_tokens
            .saturating_sub(self.cfg.context.reserved_overhead_tokens)
            .saturating_sub(self.cfg.context.response_budget_tokens);
        let threshold = (effective_window as f32
            * self.cfg.context.auto_compact_threshold.clamp(0.1, 1.0))
            as u64;

        let mut blocks = Vec::new();
        if transcript_tokens >= threshold && !transcript.is_empty() {
            if transcript.len() >= 20 && transcript.len().is_multiple_of(10) {
                let summary = summarize_transcript(&transcript, 30);
                let summary_id = Uuid::now_v7();
                let replay_pointer = format!(".deepseek/compactions/{summary_id}.md");
                let summary_path =
                    deepseek_core::runtime_dir(&self.workspace).join(&replay_pointer);
                if let Some(parent) = summary_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&summary_path, &summary)?;
                self.emit(
                    session_id,
                    EventKind::ContextCompactedV1 {
                        summary_id,
                        from_turn: 1,
                        to_turn: transcript.len() as u64,
                        token_delta_estimate: transcript_tokens as i64
                            - estimate_tokens(&summary) as i64,
                        replay_pointer,
                    },
                )?;
                self.tool_host.fire_session_hooks("contextcompacted");
                blocks.push(format!("[auto_compaction]\n{summary}"));
            }
        } else if !transcript.is_empty() {
            let recent = transcript
                .iter()
                .rev()
                .take(16)
                .cloned()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n");
            blocks.push(format!("[recent_transcript]\n{recent}"));
        }

        if let Ok(memory) = MemoryManager::new(&self.workspace)?.read_combined_memory()
            && !memory.trim().is_empty()
        {
            blocks.push(format!("[memory]\n{memory}"));
        }
        let strategy_entries = self.load_matching_strategies(prompt, 4).unwrap_or_default();
        if !strategy_entries.is_empty() {
            blocks.push(format!(
                "[strategy_memory]\n{}",
                format_strategy_entries(&strategy_entries)
            ));
        }
        let objective_entries = self
            .load_matching_objective_outcomes(prompt, 4)
            .unwrap_or_default();
        if !objective_entries.is_empty() {
            blocks.push(format!(
                "[objective_outcomes]\n{}",
                format_objective_outcomes(&objective_entries)
            ));
        }
        let verification_feedback = self
            .store
            .list_recent_verification_runs(session_id, 10)
            .unwrap_or_default()
            .into_iter()
            .filter(|run| !run.success)
            .take(6)
            .collect::<Vec<_>>();
        if !verification_feedback.is_empty() {
            blocks.push(format!(
                "[verification_feedback]\n{}",
                format_verification_feedback(&verification_feedback)
            ));
        }

        if blocks.is_empty() {
            Ok(prompt.to_string())
        } else {
            Ok(format!("{prompt}\n\n{}", blocks.join("\n\n")))
        }
    }
}

pub(crate) fn summarize_transcript(transcript: &[String], max_lines: usize) -> String {
    let mut lines = transcript
        .iter()
        .rev()
        .take(max_lines)
        .cloned()
        .collect::<Vec<_>>();
    lines.reverse();
    let rendered = lines
        .into_iter()
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.len() > 240 {
                format!("- {}...", &trimmed[..240])
            } else {
                format!("- {trimmed}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("Auto-compacted transcript summary:\n{rendered}")
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ChatSubagentSpawnDecision {
    pub(crate) should_spawn: bool,
    pub(crate) blocked_by_tools: bool,
    pub(crate) score: f32,
    pub(crate) task_budget: usize,
}
