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
}
