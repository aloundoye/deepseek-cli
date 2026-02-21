use anyhow::{Result, anyhow};
use deepseek_core::runtime_dir;
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillRunOutput {
    pub skill_id: String,
    pub source_path: PathBuf,
    pub rendered_prompt: String,
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
                let name = raw
                    .lines()
                    .find(|line| line.starts_with('#'))
                    .map(|line| line.trim_start_matches('#').trim().to_string())
                    .filter(|line| !line.is_empty())
                    .unwrap_or_else(|| id.clone());
                let summary = raw
                    .lines()
                    .find(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
                    .unwrap_or_default()
                    .trim()
                    .to_string();

                out.push(SkillEntry {
                    id,
                    name,
                    path: skill_path.to_path_buf(),
                    summary,
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
        Ok(SkillRunOutput {
            skill_id: skill.id,
            source_path: skill.path,
            rendered_prompt: rendered,
        })
    }

    pub fn reload(&self, configured_paths: &[String]) -> Result<Vec<SkillEntry>> {
        self.list(configured_paths)
    }
}

// ── Custom Slash Commands ──────────────────────────────────────────────────

/// A custom slash command loaded from `.deepseek/commands/` or `~/.deepseek/commands/`.
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
}

/// Load custom slash commands from `.deepseek/commands/` and `~/.deepseek/commands/`.
pub fn load_custom_commands(workspace: &Path) -> Vec<CustomCommand> {
    let mut commands = Vec::new();
    let project_dir = workspace.join(".deepseek").join("commands");
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
        .map(|h| PathBuf::from(h).join(".deepseek").join("commands"))
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
        out.push(CustomCommand {
            name,
            path: path.clone(),
            description,
            disable_model_invocation: disable_model,
            context,
            body: body.to_string(),
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
            std::env::temp_dir().join(format!("deepseek-skills-test-{}", uuid::Uuid::now_v7()));
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
            std::env::temp_dir().join(format!("deepseek-cmds-test-{}", uuid::Uuid::now_v7()));
        let cmd_dir = workspace.join(".deepseek").join("commands");
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
}
