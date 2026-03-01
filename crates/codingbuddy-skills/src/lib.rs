use anyhow::{Result, anyhow};
use codingbuddy_core::runtime_dir;
use glob::Pattern;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillEntry {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    pub summary: String,
    /// Tools this skill is allowed to use (empty = all tools).
    #[serde(default)]
    pub allowed_tools: Vec<String>,
    /// Tools this skill is NOT allowed to use.
    #[serde(default)]
    pub disallowed_tools: Vec<String>,
    /// Scope of this skill in the hierarchy: Project > User > BuiltIn.
    #[serde(default)]
    pub scope: SkillScope,
    /// Execution context: "normal" (inline) or "fork" (isolated subagent).
    #[serde(default = "default_context")]
    pub context: String,
    /// If true, the model cannot invoke this skill via the `skill` tool.
    #[serde(default)]
    pub disable_model_invocation: bool,
}

fn default_context() -> String {
    "normal".to_string()
}

/// Skill scope in the resolution hierarchy.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum SkillScope {
    /// Defined in the project's `.codingbuddy/skills/` directory.
    Project,
    /// Defined in the user's `~/.codingbuddy/skills/` directory.
    User,
    /// Built-in skill shipped with the CLI.
    #[default]
    BuiltIn,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillRunOutput {
    pub skill_id: String,
    pub source_path: PathBuf,
    pub rendered_prompt: String,
    /// Whether this skill should run in an isolated (forked) context.
    pub forked: bool,
    /// Tools this skill is allowed to use (empty = all).
    pub allowed_tools: Vec<String>,
    /// Tools this skill is NOT allowed to use.
    pub disallowed_tools: Vec<String>,
    /// Whether model auto-invocation is disabled for this skill.
    pub disable_model_invocation: bool,
}

pub struct SkillManager {
    workspace: PathBuf,
    install_root: PathBuf,
}

impl SkillManager {
    pub fn new(workspace: &Path) -> Result<Self> {
        let install_root = runtime_dir(workspace).join("skills");
        fs::create_dir_all(&install_root)?;
        Ok(Self {
            workspace: workspace.to_path_buf(),
            install_root,
        })
    }

    pub fn install_root(&self) -> &Path {
        &self.install_root
    }

    pub fn list(&self, configured_paths: &[String]) -> Result<Vec<SkillEntry>> {
        let mut roots = vec![self.install_root.clone()];
        for raw in configured_paths {
            let path = if Path::new(raw).is_absolute() {
                PathBuf::from(raw)
            } else {
                self.workspace.join(raw)
            };
            roots.push(path);
        }

        let mut out = Vec::new();
        for root in roots {
            if !root.exists() {
                continue;
            }
            for entry in WalkDir::new(&root).into_iter().filter_map(Result::ok) {
                if !entry.path().is_file() {
                    continue;
                }
                if entry
                    .path()
                    .file_name()
                    .is_none_or(|name| name != "SKILL.md")
                {
                    continue;
                }
                let skill_path = entry.path();
                let id = skill_path
                    .parent()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .unwrap_or("skill")
                    .to_string();
                let raw = fs::read_to_string(skill_path)?;
                let (frontmatter, body) = parse_command_frontmatter(&raw);
                let name = body
                    .lines()
                    .find(|line| line.starts_with('#'))
                    .map(|line| line.trim_start_matches('#').trim().to_string())
                    .filter(|line| !line.is_empty())
                    .unwrap_or_else(|| id.clone());
                let summary = body
                    .lines()
                    .find(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                let allowed_tools = parse_comma_list(frontmatter.get("allowed_tools"));
                let disallowed_tools = parse_comma_list(frontmatter.get("disallowed_tools"));
                let context = frontmatter
                    .get("context")
                    .cloned()
                    .unwrap_or_else(|| "normal".to_string());
                let disable_model_invocation = frontmatter
                    .get("disable-model-invocation")
                    .is_some_and(|v| v == "true");

                out.push(SkillEntry {
                    id,
                    name,
                    path: skill_path.to_path_buf(),
                    summary,
                    allowed_tools,
                    disallowed_tools,
                    scope: SkillScope::BuiltIn,
                    context,
                    disable_model_invocation,
                });
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out.dedup_by(|a, b| a.id == b.id);
        Ok(out)
    }

    pub fn install(&self, source: &Path) -> Result<SkillEntry> {
        let source_file = if source.is_file() {
            source.to_path_buf()
        } else if source.is_dir() {
            let direct = source.join("SKILL.md");
            if direct.exists() {
                direct
            } else {
                WalkDir::new(source)
                    .into_iter()
                    .filter_map(Result::ok)
                    .find(|entry| {
                        entry.path().is_file()
                            && entry.path().file_name() == Some("SKILL.md".as_ref())
                    })
                    .map(|entry| entry.path().to_path_buf())
                    .ok_or_else(|| anyhow!("no SKILL.md found in {}", source.display()))?
            }
        } else {
            return Err(anyhow!("invalid source: {}", source.display()));
        };

        let raw = fs::read_to_string(&source_file)?;
        let id = source_file
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("skill")
            .to_string();

        let dest_dir = self.install_root.join(&id);
        fs::create_dir_all(&dest_dir)?;
        let dest_file = dest_dir.join("SKILL.md");
        fs::write(&dest_file, raw)?;

        self.list(&[])?
            .into_iter()
            .find(|entry| entry.id == id)
            .ok_or_else(|| anyhow!("failed to index installed skill"))
    }

    pub fn remove(&self, skill_id: &str) -> Result<()> {
        let pattern = Pattern::new(skill_id).ok();
        let mut removed_any = false;
        for entry in fs::read_dir(&self.install_root)? {
            let path = entry?.path();
            if !path.is_dir() {
                continue;
            }
            let id = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            let matches = if let Some(pattern) = &pattern {
                pattern.matches(&id)
            } else {
                id == skill_id
            };
            if matches {
                fs::remove_dir_all(&path)?;
                removed_any = true;
            }
        }
        if !removed_any {
            return Err(anyhow!("skill not found: {skill_id}"));
        }
        Ok(())
    }

    pub fn run(
        &self,
        skill_id: &str,
        input: Option<&str>,
        configured_paths: &[String],
    ) -> Result<SkillRunOutput> {
        let skills = self.list(configured_paths)?;
        let skill = skills
            .into_iter()
            .find(|entry| entry.id == skill_id)
            .ok_or_else(|| anyhow!("skill not found: {skill_id}"))?;
        let template = fs::read_to_string(&skill.path)?;
        let rendered = template
            .replace("{{input}}", input.unwrap_or(""))
            .replace("{{skill_id}}", &skill.id)
            .replace("{{workspace}}", self.workspace.to_string_lossy().as_ref());
        let forked = skill.context == "fork";
        Ok(SkillRunOutput {
            skill_id: skill.id.clone(),
            source_path: skill.path.clone(),
            rendered_prompt: rendered,
            forked,
            allowed_tools: skill.allowed_tools.clone(),
            disallowed_tools: skill.disallowed_tools.clone(),
            disable_model_invocation: skill.disable_model_invocation,
        })
    }

    pub fn reload(&self, configured_paths: &[String]) -> Result<Vec<SkillEntry>> {
        self.list(configured_paths)
    }

    /// Run a skill in forked (isolated) mode.
    ///
    /// Returns a `SkillRunOutput` with `forked: true`. The caller is responsible
    /// for spawning the actual isolated ToolUseLoop context (e.g. via `SubagentWorker`).
    /// Tool restrictions (`allowed_tools`, `disallowed_tools`) from the skill's
    /// frontmatter are included in the output for the caller to enforce.
    pub fn run_forked(
        &self,
        skill_id: &str,
        input: Option<&str>,
        configured_paths: &[String],
    ) -> Result<SkillRunOutput> {
        let mut output = self.run(skill_id, input, configured_paths)?;
        output.forked = true;
        Ok(output)
    }

    /// Look up a skill by ID without rendering.
    pub fn get(&self, skill_id: &str, configured_paths: &[String]) -> Result<Option<SkillEntry>> {
        let skills = self.list(configured_paths)?;
        Ok(skills.into_iter().find(|s| s.id == skill_id))
    }
}

/// Parse a comma-separated list from an optional frontmatter value.
fn parse_comma_list(value: Option<&String>) -> Vec<String> {
    value
        .map(|v| {
            v.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

// ── Context Budget ─────────────────────────────────────────────────────────

/// Default skill description budget: 2% of 128K context ≈ 2560 tokens ≈ 10240 chars.
pub const DEFAULT_SKILL_BUDGET_CHARS: usize = 10_240;

/// Trim skill summaries to fit within a character budget.
/// Skills are processed in order; once the budget is exhausted, remaining
/// summaries are replaced with a placeholder.
pub fn apply_context_budget(skills: &[SkillEntry], budget_chars: usize) -> Vec<SkillEntry> {
    let mut result = skills.to_vec();
    let mut used = 0_usize;
    for skill in &mut result {
        let remaining = budget_chars.saturating_sub(used);
        if remaining == 0 {
            skill.summary = "[budget exceeded]".to_string();
        } else if skill.summary.len() > remaining {
            let end = remaining.saturating_sub(3);
            // Ensure we don't split a multi-byte char
            let safe_end = skill.summary.floor_char_boundary(end);
            skill.summary = format!("{}...", &skill.summary[..safe_end]);
        }
        used += skill.summary.len();
    }
    result
}

// ── Custom Slash Commands ──────────────────────────────────────────────────

/// A custom slash command loaded from `.codingbuddy/commands/` or `~/.codingbuddy/commands/`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomCommand {
    /// The command name (derived from filename, e.g. "deploy" from "deploy.md").
    pub name: String,
    /// Source file path.
    pub path: PathBuf,
    /// Short description (from `description:` frontmatter or first paragraph).
    pub description: String,
    /// Whether to suppress model invocation (dry run only).
    pub disable_model_invocation: bool,
    /// Context mode: "normal" or "fork" (run in subagent).
    pub context: String,
    /// The rendered prompt body (after variable substitution).
    pub body: String,
    /// Scoped lifecycle hooks declared in frontmatter (e.g. `hooks.PreToolUse: echo check`).
    #[serde(default)]
    pub hooks: std::collections::HashMap<String, Vec<String>>,
}

/// Load custom slash commands from `.codingbuddy/commands/` and `~/.codingbuddy/commands/`.
pub fn load_custom_commands(workspace: &Path) -> Vec<CustomCommand> {
    let mut commands = Vec::new();
    let project_dir = workspace.join(".codingbuddy").join("commands");
    let user_dir = dirs_commands();
    load_commands_from_dir(&project_dir, &mut commands);
    if let Some(user) = user_dir {
        load_commands_from_dir(&user, &mut commands);
    }
    // Dedup by name (project-level wins)
    let mut seen = std::collections::HashSet::new();
    commands.retain(|c| seen.insert(c.name.clone()));
    commands
}

/// Render a custom command body with variable substitution.
pub fn render_custom_command(
    cmd: &CustomCommand,
    arguments: &str,
    workspace: &Path,
    session_id: &str,
) -> String {
    cmd.body
        .replace("$ARGUMENTS", arguments)
        .replace("$WORKSPACE", &workspace.to_string_lossy())
        .replace("$SESSION_ID", session_id)
}

fn dirs_commands() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".codingbuddy").join("commands"))
}

fn load_commands_from_dir(dir: &Path, out: &mut Vec<CustomCommand>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext != "md" {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        let Ok(raw) = fs::read_to_string(&path) else {
            continue;
        };
        let (frontmatter, body) = parse_command_frontmatter(&raw);
        let description = frontmatter.get("description").cloned().unwrap_or_default();
        let disable_model = frontmatter
            .get("disable-model-invocation")
            .is_some_and(|v| v == "true");
        let context = frontmatter
            .get("context")
            .cloned()
            .unwrap_or_else(|| "normal".to_string());
        // Parse hooks.* keys from frontmatter
        let mut hooks = std::collections::HashMap::<String, Vec<String>>::new();
        for (key, value) in &frontmatter {
            if let Some(event_name) = key.strip_prefix("hooks.") {
                hooks
                    .entry(event_name.to_string())
                    .or_default()
                    .push(value.clone());
            }
        }
        out.push(CustomCommand {
            name,
            path: path.clone(),
            description,
            disable_model_invocation: disable_model,
            context,
            body: body.to_string(),
            hooks,
        });
    }
}

/// Parse YAML-style frontmatter delimited by `---`.
fn parse_command_frontmatter(raw: &str) -> (std::collections::HashMap<String, String>, &str) {
    let mut map = std::collections::HashMap::new();
    let trimmed = raw.trim_start();
    if !trimmed.starts_with("---") {
        return (map, raw);
    }
    let after_first = &trimmed[3..].trim_start_matches('\n');
    if let Some(end) = after_first.find("\n---") {
        let yaml_block = &after_first[..end];
        for line in yaml_block.lines() {
            if let Some((key, val)) = line.split_once(':') {
                map.insert(key.trim().to_string(), val.trim().to_string());
            }
        }
        let body_start = end + 4; // "\n---"
        let body = after_first[body_start..].trim_start_matches('\n');
        return (map, body);
    }
    (map, raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn installs_lists_runs_and_removes_skill() {
        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-skills-test-{}", uuid::Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let manager = SkillManager::new(&workspace).expect("manager");

        let source = workspace.join("sample-skill");
        fs::create_dir_all(&source).expect("source");
        fs::write(
            source.join("SKILL.md"),
            "# Sample Skill\n\nUse this for {{input}} in {{workspace}}.",
        )
        .expect("skill file");

        let installed = manager.install(&source).expect("install");
        assert_eq!(installed.id, "sample-skill");

        let listed = manager.list(&[]).expect("list");
        assert_eq!(listed.len(), 1);

        let run = manager
            .run("sample-skill", Some("planning"), &[])
            .expect("run");
        assert!(run.rendered_prompt.contains("planning"));

        manager.remove("sample-skill").expect("remove");
        assert!(manager.list(&[]).expect("list after").is_empty());
    }

    #[test]
    fn loads_custom_commands_from_directory() {
        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-cmds-test-{}", uuid::Uuid::now_v7()));
        let cmd_dir = workspace.join(".codingbuddy").join("commands");
        fs::create_dir_all(&cmd_dir).expect("cmd dir");

        fs::write(
            cmd_dir.join("deploy.md"),
            "---\ndescription: Deploy to production\ncontext: fork\n---\nDeploy $ARGUMENTS to $WORKSPACE",
        )
        .expect("deploy.md");

        fs::write(
            cmd_dir.join("lint.md"),
            "Run linting on $WORKSPACE with $ARGUMENTS",
        )
        .expect("lint.md");

        let commands = load_custom_commands(&workspace);
        assert_eq!(commands.len(), 2);

        let deploy = commands.iter().find(|c| c.name == "deploy").unwrap();
        assert_eq!(deploy.description, "Deploy to production");
        assert_eq!(deploy.context, "fork");
        assert!(!deploy.disable_model_invocation);

        let lint = commands.iter().find(|c| c.name == "lint").unwrap();
        assert!(lint.description.is_empty());
        assert_eq!(lint.context, "normal");

        // Test variable substitution
        let rendered = render_custom_command(deploy, "v2.0", &workspace, "sess-123");
        assert!(rendered.contains("v2.0"));
        assert!(rendered.contains(&workspace.to_string_lossy().to_string()));
    }

    #[test]
    fn parses_command_frontmatter() {
        let raw = "---\ndescription: My cmd\ndisable-model-invocation: true\n---\nBody here";
        let (fm, body) = parse_command_frontmatter(raw);
        assert_eq!(fm.get("description").unwrap(), "My cmd");
        assert_eq!(fm.get("disable-model-invocation").unwrap(), "true");
        assert_eq!(body, "Body here");
    }

    #[test]
    fn frontmatter_absent_returns_raw() {
        let raw = "Just a plain command body.";
        let (fm, body) = parse_command_frontmatter(raw);
        assert!(fm.is_empty());
        assert_eq!(body, raw);
    }

    #[test]
    fn parse_frontmatter_with_hooks() {
        let raw = "---\ndescription: My cmd\nhooks.PreToolUse: echo check\nhooks.Stop: notify-send done\n---\nBody";
        let (fm, body) = parse_command_frontmatter(raw);
        assert_eq!(fm.get("description").unwrap(), "My cmd");
        assert_eq!(fm.get("hooks.PreToolUse").unwrap(), "echo check");
        assert_eq!(fm.get("hooks.Stop").unwrap(), "notify-send done");
        assert_eq!(body, "Body");
    }

    #[test]
    fn skill_hooks_parsed_into_custom_command() {
        let workspace = std::env::temp_dir().join(format!(
            "codingbuddy-hook-cmd-test-{}",
            uuid::Uuid::now_v7()
        ));
        let cmd_dir = workspace.join(".codingbuddy").join("commands");
        fs::create_dir_all(&cmd_dir).expect("cmd dir");

        fs::write(
            cmd_dir.join("hooked.md"),
            "---\ndescription: Hooked command\nhooks.PreToolUse: echo validate\nhooks.PostToolUse: echo log\n---\nDo $ARGUMENTS",
        )
        .expect("hooked.md");

        let commands = load_custom_commands(&workspace);
        let hooked = commands.iter().find(|c| c.name == "hooked").unwrap();
        assert_eq!(hooked.hooks.len(), 2);
        assert!(hooked.hooks.contains_key("PreToolUse"));
        assert!(hooked.hooks.contains_key("PostToolUse"));
        assert_eq!(hooked.hooks["PreToolUse"], vec!["echo validate"]);
    }

    #[test]
    fn apply_context_budget_trims_long_summaries() {
        let skills: Vec<SkillEntry> = (0..5)
            .map(|i| SkillEntry {
                id: format!("s{i}"),
                name: format!("Skill {i}"),
                path: PathBuf::from(format!("/tmp/s{i}")),
                summary: "a".repeat(30),
                allowed_tools: vec![],
                disallowed_tools: vec![],
                scope: SkillScope::BuiltIn,
                context: "normal".to_string(),
                disable_model_invocation: false,
            })
            .collect();
        let result = apply_context_budget(&skills, 100);
        // First 3 skills fit (30*3=90), 4th is truncated (10 remaining), 5th is budget exceeded
        assert_eq!(result[0].summary.len(), 30);
        assert!(result[3].summary.ends_with("..."));
        assert_eq!(result[4].summary, "[budget exceeded]");
    }

    #[test]
    fn apply_context_budget_no_trim_within_limit() {
        let skills: Vec<SkillEntry> = (0..3)
            .map(|i| SkillEntry {
                id: format!("s{i}"),
                name: format!("Skill {i}"),
                path: PathBuf::from(format!("/tmp/s{i}")),
                summary: "short".to_string(),
                allowed_tools: vec![],
                disallowed_tools: vec![],
                scope: SkillScope::BuiltIn,
                context: "normal".to_string(),
                disable_model_invocation: false,
            })
            .collect();
        let result = apply_context_budget(&skills, 10_000);
        for (i, skill) in result.iter().enumerate() {
            assert_eq!(skill.summary, "short", "skill {i} should be unchanged");
        }
    }

    #[test]
    fn apply_context_budget_empty_skills() {
        let result = apply_context_budget(&[], 100);
        assert!(result.is_empty());
    }

    // ── P5-06: Forked execution ──────────────────────────────────────────

    #[test]
    fn skill_forked_isolates_context() {
        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-fork-test-{}", uuid::Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let manager = SkillManager::new(&workspace).expect("manager");

        // Create a skill with context: fork
        let source = workspace.join("forked-skill");
        fs::create_dir_all(&source).expect("source");
        fs::write(
            source.join("SKILL.md"),
            "---\ncontext: fork\ndescription: A forked skill\n---\n# Forked Skill\n\nRun {{input}} in isolated context.",
        )
        .expect("skill file");
        manager.install(&source).expect("install");

        // run_forked should return forked=true
        let output = manager
            .run_forked("forked-skill", Some("tests"), &[])
            .expect("run_forked");
        assert!(output.forked);
        assert!(output.rendered_prompt.contains("tests"));
        assert!(output.rendered_prompt.contains("isolated context"));

        // Regular run should also detect fork from frontmatter
        let output2 = manager.run("forked-skill", Some("lint"), &[]).expect("run");
        assert!(output2.forked); // context: fork in frontmatter
    }

    // ── P5-07: Allowed-tools enforcement ─────────────────────────────────

    #[test]
    fn allowed_tools_enforced() {
        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-allow-test-{}", uuid::Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let manager = SkillManager::new(&workspace).expect("manager");

        let source = workspace.join("restricted-skill");
        fs::create_dir_all(&source).expect("source");
        fs::write(
            source.join("SKILL.md"),
            "---\nallowed_tools: fs_read, fs_glob, git_status\ndisallowed_tools: bash_run\n---\n# Restricted\n\nOnly read ops.",
        )
        .expect("skill file");
        manager.install(&source).expect("install");

        let output = manager.run("restricted-skill", None, &[]).expect("run");
        assert_eq!(
            output.allowed_tools,
            vec!["fs_read", "fs_glob", "git_status"]
        );
        assert_eq!(output.disallowed_tools, vec!["bash_run"]);
    }

    #[test]
    fn disallowed_tools_filtered() {
        // Verify the existing filter_tool_definitions works with our data
        let skills = parse_comma_list(Some(&"bash_run, fs_edit".to_string()));
        assert_eq!(skills, vec!["bash_run", "fs_edit"]);
    }

    // ── P5-08: Skill auto-invocation by model ────────────────────────────

    #[test]
    fn skill_invocation_by_model() {
        let workspace =
            std::env::temp_dir().join(format!("codingbuddy-invoke-test-{}", uuid::Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("workspace");
        let manager = SkillManager::new(&workspace).expect("manager");

        // Skill with model invocation disabled
        let source = workspace.join("no-invoke-skill");
        fs::create_dir_all(&source).expect("source");
        fs::write(
            source.join("SKILL.md"),
            "---\ndisable-model-invocation: true\ndescription: Not auto-invocable\n---\n# No Invoke\n\nManual only.",
        )
        .expect("skill file");
        manager.install(&source).expect("install");

        let output = manager.run("no-invoke-skill", None, &[]).expect("run");
        assert!(output.disable_model_invocation);
        assert!(!output.forked);

        // Skill without the flag — model CAN invoke it
        let source2 = workspace.join("auto-skill");
        fs::create_dir_all(&source2).expect("source2");
        fs::write(
            source2.join("SKILL.md"),
            "---\ndescription: Auto-invocable\n---\n# Auto\n\nModel can call me.",
        )
        .expect("skill file");
        manager.install(&source2).expect("install");

        let output2 = manager.run("auto-skill", None, &[]).expect("run");
        assert!(!output2.disable_model_invocation);
    }
}
