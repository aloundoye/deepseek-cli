use ratatui::style::Color;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UiStatus {
    pub model: String,
    #[serde(default)]
    pub provider: String,
    pub pending_approvals: usize,
    pub estimated_cost_usd: f64,
    pub background_jobs: usize,
    pub autopilot_running: bool,
    pub permission_mode: String,
    pub active_tasks: usize,
    #[serde(default)]
    pub workflow_phase: String,
    #[serde(default)]
    pub plan_state: String,
    #[serde(default)]
    pub context_used_tokens: u64,
    #[serde(default = "default_context_max")]
    pub context_max_tokens: u64,
    #[serde(default)]
    pub session_turns: usize,
    #[serde(default)]
    pub working_directory: String,
    #[serde(default)]
    pub pr_review_status: Option<String>,
    #[serde(default)]
    pub pr_url: Option<String>,
    #[serde(default)]
    pub agent_mode: String,
    #[serde(default)]
    pub mission_control_snapshot: Vec<String>,
    #[serde(default)]
    pub current_todo: String,
    #[serde(default)]
    pub current_step: String,
    #[serde(default)]
    pub running_subagents: usize,
    #[serde(default)]
    pub failed_subagents: usize,
    #[serde(default)]
    pub failed_tasks: usize,
    #[serde(default)]
    pub running_background_jobs: usize,
    #[serde(default)]
    pub capability_summary: String,
    #[serde(default)]
    pub provider_diagnostics_summary: String,
    #[serde(default)]
    pub runtime_diagnostics_summary: String,
    #[serde(default)]
    pub compaction_count: usize,
    #[serde(default)]
    pub replay_count: usize,
}

fn default_context_max() -> u64 {
    128_000
}

pub fn render_statusline(status: &UiStatus) -> String {
    let mode_indicator = match status.permission_mode.as_str() {
        "auto" => "[AUTO]",
        "plan" => "[PLAN]",
        "locked" => "[LOCKED]",
        _ => "[ASK]",
    };
    let tasks_part = if status.active_tasks > 0 {
        format!(" tasks={}", status.active_tasks)
    } else {
        String::new()
    };
    let phase_part = if !status.workflow_phase.is_empty() && status.workflow_phase != "idle" {
        format!(" phase={}", status.workflow_phase)
    } else {
        String::new()
    };
    let plan_part = if status.plan_state != "none" {
        format!(" plan={}", status.plan_state)
    } else {
        String::new()
    };
    let ctx_pct = if status.context_max_tokens > 0 {
        (status.context_used_tokens as f64 / status.context_max_tokens as f64 * 100.0) as u64
    } else {
        0
    };
    let ctx_part = if status.context_max_tokens > 0 {
        format!(
            " ctx={}K/{}K({}%)",
            status.context_used_tokens / 1000,
            status.context_max_tokens / 1000,
            ctx_pct
        )
    } else {
        String::new()
    };
    let review_part = match (&status.pr_review_status, &status.pr_url) {
        (Some(s), Some(url)) => format!(" review={s} {url}"),
        (Some(s), None) => format!(" review={s}"),
        _ => String::new(),
    };
    let todo_part = if status.current_todo.trim().is_empty() {
        String::new()
    } else {
        format!(" todo={}", status.current_todo)
    };
    let step_part = if status.current_step.trim().is_empty() {
        String::new()
    } else {
        format!(" step={}", status.current_step)
    };
    let subagent_part = if status.running_subagents > 0 || status.failed_subagents > 0 {
        format!(
            " subagents={}/{}",
            status.running_subagents, status.failed_subagents
        )
    } else {
        String::new()
    };
    let counters_part = if status.compaction_count > 0 || status.replay_count > 0 {
        format!(
            " compact={} replay={}",
            status.compaction_count, status.replay_count
        )
    } else {
        String::new()
    };
    let caps_part = if status.capability_summary.trim().is_empty() {
        String::new()
    } else {
        format!(" caps={}", status.capability_summary)
    };
    let agent_part = if !status.agent_mode.is_empty()
        && status.agent_mode != "ToolUseLoop"
        && status.agent_mode != "ArchitectEditorLoop"
    {
        format!(" agent={}", status.agent_mode)
    } else {
        String::new()
    };
    format!(
        "model={} {} approvals={} jobs={}{}{}{} autopilot={}{}{}{}{}{}{}{}{} cost=${:.4}",
        status.model,
        mode_indicator,
        status.pending_approvals,
        status.background_jobs,
        tasks_part,
        phase_part,
        plan_part,
        if status.autopilot_running {
            "running"
        } else {
            "idle"
        },
        ctx_part,
        step_part,
        todo_part,
        subagent_part,
        counters_part,
        caps_part,
        review_part,
        agent_part,
        status.estimated_cost_usd,
    )
}

pub(crate) fn review_badge(status: &str) -> (&'static str, Color) {
    match status {
        "approved" => (" REVIEW APPROVED ", Color::Green),
        "changes_requested" => (" REVIEW CHANGES ", Color::Red),
        "draft" => (" REVIEW DRAFT ", Color::Blue),
        "merged" => (" REVIEW MERGED ", Color::Cyan),
        _ => (" REVIEW PENDING ", Color::Yellow),
    }
}

pub(crate) fn render_statusline_spans(
    status: &UiStatus,
    active_tool: Option<&str>,
    spinner_frame: &str,
    scroll_pct: Option<usize>,
    vim_mode_label: Option<&str>,
    has_new_content_below: bool,
    is_thinking: bool,
) -> Vec<ratatui::text::Span<'static>> {
    use ratatui::style::{Modifier, Style};
    use ratatui::text::Span;

    let mode_color = match status.permission_mode.as_str() {
        "auto" => Color::Green,
        "plan" => Color::Blue,
        "locked" => Color::Red,
        _ => Color::Yellow,
    };
    let mode_label = match status.permission_mode.as_str() {
        "auto" => " AUTO ",
        "plan" => " PLAN ",
        "locked" => " LOCKED ",
        _ => " ASK ",
    };

    let mut spans = Vec::new();

    if is_thinking {
        spans.push(Span::styled(
            format!(" {} Thinking ", spinner_frame),
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ));
    } else if let Some(tool) = active_tool {
        spans.push(Span::styled(
            format!(" {} {} ", spinner_frame, tool),
            Style::default().fg(Color::Yellow),
        ));
    }

    spans.push(Span::styled(
        format!(" {} ", status.model),
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    ));
    if !status.provider.is_empty() && status.provider != status.model {
        spans.push(Span::styled(
            format!(" {} ", status.provider),
            Style::default().fg(Color::Blue),
        ));
    }
    spans.push(Span::raw(" "));
    spans.push(Span::styled(
        mode_label.to_string(),
        Style::default()
            .fg(Color::Black)
            .bg(mode_color)
            .add_modifier(Modifier::BOLD),
    ));

    if status.pending_approvals > 0 {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            format!(" {} pending ", status.pending_approvals),
            Style::default()
                .fg(Color::Black)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
    }

    if status.active_tasks > 0 {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            format!(" {} tasks ", status.active_tasks),
            Style::default().fg(Color::Magenta),
        ));
    }

    if !status.workflow_phase.is_empty() && status.workflow_phase != "idle" {
        let phase_color = match status.workflow_phase.as_str() {
            "explore" => Color::Cyan,
            "plan" => Color::Blue,
            "approval" => Color::Yellow,
            "execute" => Color::Green,
            "verify" => Color::Magenta,
            "completed" => Color::Green,
            "failed" => Color::Red,
            "paused" => Color::Gray,
            _ => Color::White,
        };
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            format!(" {} ", status.workflow_phase.to_ascii_uppercase()),
            Style::default()
                .fg(Color::Black)
                .bg(phase_color)
                .add_modifier(Modifier::BOLD),
        ));
    }

    if status.plan_state != "none" {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            format!(" PLAN:{} ", status.plan_state.to_ascii_uppercase()),
            Style::default().fg(Color::Blue),
        ));
    }

    if status.background_jobs > 0 {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            format!(" {} jobs ", status.background_jobs),
            Style::default().fg(Color::Blue),
        ));
    }

    if status.running_subagents > 0 || status.failed_subagents > 0 {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            format!(
                " subagents {}/{} ",
                status.running_subagents, status.failed_subagents
            ),
            Style::default().fg(Color::Magenta),
        ));
    }

    if !status.current_step.is_empty() {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            format!(" step {} ", status.current_step),
            Style::default().fg(Color::DarkGray),
        ));
    }

    if !status.current_todo.is_empty() {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            format!(" todo {} ", status.current_todo),
            Style::default().fg(Color::DarkGray),
        ));
    }

    if status.autopilot_running {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            " AUTOPILOT ".to_string(),
            Style::default()
                .fg(Color::Black)
                .bg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ));
    }

    if !status.agent_mode.is_empty()
        && status.agent_mode != "ToolUseLoop"
        && status.agent_mode != "ArchitectEditorLoop"
    {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            format!(" {} ", status.agent_mode),
            Style::default()
                .fg(Color::White)
                .bg(Color::Blue)
                .add_modifier(Modifier::BOLD),
        ));
    }

    if status.context_max_tokens > 0 {
        let pct =
            (status.context_used_tokens as f64 / status.context_max_tokens as f64 * 100.0) as u64;
        let ctx_color = if pct > 80 {
            Color::Red
        } else if pct > 60 {
            Color::Yellow
        } else {
            Color::Green
        };
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            format!(
                " {}K/{}K ",
                status.context_used_tokens / 1000,
                status.context_max_tokens / 1000
            ),
            Style::default().fg(ctx_color),
        ));
    }

    if status.session_turns > 0 {
        spans.push(Span::styled(
            format!(" turn {} ", status.session_turns),
            Style::default().fg(Color::DarkGray),
        ));
    }

    if let Some(review) = status.pr_review_status.as_deref() {
        let (label, bg) = review_badge(review);
        let badge_style = Style::default()
            .fg(Color::Black)
            .bg(bg)
            .add_modifier(Modifier::BOLD);
        spans.push(Span::raw(" "));
        if let Some(ref url) = status.pr_url {
            let linked = format!("\x1b]8;;{url}\x1b\\{label}\x1b]8;;\x1b\\");
            spans.push(Span::styled(linked, badge_style));
        } else {
            spans.push(Span::styled(label.to_string(), badge_style));
        }
    }

    spans.push(Span::styled(
        format!(" ${:.4} ", status.estimated_cost_usd),
        Style::default().fg(Color::DarkGray),
    ));

    if status.compaction_count > 0 || status.replay_count > 0 {
        spans.push(Span::styled(
            format!(" c/r {}/{} ", status.compaction_count, status.replay_count),
            Style::default().fg(Color::DarkGray),
        ));
    }

    if !status.capability_summary.is_empty() {
        spans.push(Span::styled(
            format!(" {} ", status.capability_summary),
            Style::default().fg(Color::DarkGray),
        ));
    }

    if let Some(mode) = vim_mode_label {
        spans.push(Span::styled(
            format!(" {} ", mode),
            Style::default()
                .fg(Color::Black)
                .bg(Color::Gray)
                .add_modifier(Modifier::BOLD),
        ));
    }

    if let Some(pct) = scroll_pct {
        spans.push(Span::styled(
            format!(" {}% ", pct),
            Style::default().fg(Color::DarkGray),
        ));
    }

    if has_new_content_below {
        spans.push(Span::styled(
            " \u{2193} new ".to_string(),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
    }

    spans
}
