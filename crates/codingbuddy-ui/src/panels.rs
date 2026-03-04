use crate::UiStatus;
use std::fs;
use std::path::Path;

pub fn render_mission_control_panel(status: &UiStatus, mission_control_lines: &[String]) -> String {
    let mut lines = if status.mission_control_snapshot.is_empty() {
        vec![
            "Mission Control".to_string(),
            "No persisted task snapshot is available for this session yet.".to_string(),
        ]
    } else {
        status.mission_control_snapshot.clone()
    };
    let recent_events: Vec<String> = mission_control_lines
        .iter()
        .rev()
        .take(6)
        .cloned()
        .collect();
    if !recent_events.is_empty() {
        lines.push(String::new());
        lines.push("Recent activity:".to_string());
        for event in recent_events.iter().rev() {
            lines.push(format!("  - {event}"));
        }
    }
    lines.push(String::new());
    if status.plan_state == "awaiting_approval" {
        lines.push(
            "Ctrl+Y approves the current plan. Alt+Y opens a rejection prompt with feedback."
                .to_string(),
        );
    }
    lines.push("Ctrl+T hides mission control.".to_string());
    lines.join("\n")
}

pub fn load_artifact_lines(workspace: &Path) -> Vec<String> {
    let artifacts_dir = workspace.join(".codingbuddy").join("artifacts");
    let mut lines = Vec::new();
    if !artifacts_dir.exists() {
        lines.push("No artifacts found.".to_string());
        lines.push(format!("Directory: {}", artifacts_dir.display()));
        return lines;
    }
    let mut entries: Vec<_> = fs::read_dir(&artifacts_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    entries.sort_by_key(|e| e.file_name());
    if entries.is_empty() {
        lines.push("No task artifacts found.".to_string());
        return lines;
    }
    for entry in entries {
        let task_dir = entry.path();
        let task_id = entry.file_name().to_string_lossy().to_string();
        lines.push(format!("## Task: {task_id}"));
        for name in &["plan.md", "diff.patch", "verification.md"] {
            let file_path = task_dir.join(name);
            if file_path.exists() {
                let size = fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);
                lines.push(format!("  {name} ({size} bytes)"));
                if let Ok(content) = fs::read_to_string(&file_path) {
                    for (i, line) in content.lines().take(5).enumerate() {
                        lines.push(format!("    {}", line));
                        if i == 4 {
                            lines.push("    ...".to_string());
                        }
                    }
                }
            }
        }
        if let Ok(files) = fs::read_dir(&task_dir) {
            for file in files.filter_map(|f| f.ok()) {
                let fname = file.file_name().to_string_lossy().to_string();
                if !["plan.md", "diff.patch", "verification.md"].contains(&fname.as_str()) {
                    let size = fs::metadata(file.path()).map(|m| m.len()).unwrap_or(0);
                    lines.push(format!("  {fname} ({size} bytes)"));
                }
            }
        }
        lines.push(String::new());
    }
    lines
}
