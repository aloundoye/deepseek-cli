use anyhow::{Result, anyhow};
use deepseek_agent::{AgentEngine, ChatOptions};
use deepseek_core::{
    AppConfig, DEEPSEEK_PROFILE_V32_SPECIALE, DEEPSEEK_V32_SPECIALE_END_DATE, EventEnvelope,
    EventKind, Session, SessionBudgets, SessionState, normalize_deepseek_model,
    normalize_deepseek_profile, runtime_dir,
};
use deepseek_store::Store;
use serde_json::json;
use std::fs;
use std::path::Path;
use uuid::Uuid;

use crate::Cli;

/// Apply CLI-level engine overrides (permission mode, verbose, budget limits).
pub(crate) fn apply_cli_flags(engine: &mut AgentEngine, cli: &Cli) {
    if cli.dangerously_skip_permissions && cli.allow_dangerously_skip_permissions {
        engine.set_permission_mode("bypassPermissions");
    } else if let Some(ref mode) = cli.permission_mode {
        engine.set_permission_mode(mode);
    }
    if cli.verbose {
        engine.set_verbose(true);
    }
    engine.set_max_turns(cli.max_turns);
    engine.set_max_budget_usd(cli.max_budget_usd);
}

/// Wire the subagent worker so spawn_task creates real child agents.
pub(crate) fn wire_subagent_worker(engine: &AgentEngine, cwd: &Path) {
    let workspace = cwd.to_path_buf();
    engine.set_subagent_worker(std::sync::Arc::new(move |task| {
        let mut child = AgentEngine::new(&workspace)?;

        // If task carries a custom agent definition, use its config.
        if let Some(ref agent_def) = task.custom_agent {
            let opts = ChatOptions {
                tools: true,
                allowed_tools: if agent_def.tools.is_empty() {
                    None
                } else {
                    Some(agent_def.tools.clone())
                },
                disallowed_tools: if agent_def.disallowed_tools.is_empty() {
                    None
                } else {
                    Some(agent_def.disallowed_tools.clone())
                },
                system_prompt_override: Some(agent_def.prompt.clone()),
                ..Default::default()
            };
            child.set_max_turns(agent_def.max_turns.or(Some(50)));
            return child.chat_with_options(&task.goal, opts);
        }

        // Configure tool restrictions based on subagent role
        let opts = match task.role {
            deepseek_subagent::SubagentRole::Explore | deepseek_subagent::SubagentRole::Plan => {
                ChatOptions {
                    tools: true,
                    allowed_tools: Some(
                        deepseek_tools::PLAN_MODE_TOOLS
                            .iter()
                            .map(|s| s.to_string())
                            .collect(),
                    ),
                    ..Default::default()
                }
            }
            _ => ChatOptions {
                tools: true,
                ..Default::default()
            },
        };
        // Limit child agent turns to prevent runaway
        child.set_max_turns(Some(50));
        child.chat_with_options(&task.goal, opts)
    }));
}

/// Build ChatOptions from CLI flags.
pub(crate) fn chat_options_from_cli(cli: &Cli, tools: bool) -> ChatOptions {
    // --system-prompt-file overrides --system-prompt
    let sys_override = if let Some(ref path) = cli.system_prompt_file {
        fs::read_to_string(path).ok()
    } else {
        cli.system_prompt.clone()
    };
    // --append-system-prompt-file overrides --append-system-prompt
    let sys_append = if let Some(ref path) = cli.append_system_prompt_file {
        fs::read_to_string(path).ok()
    } else {
        cli.append_system_prompt.clone()
    };
    // --tools flag: "" = none, "default" = all, comma-separated = restrict
    let (effective_allowed, effective_disallowed) = if let Some(ref t) = cli.tools {
        if t.is_empty() {
            // No tools at all
            (Some(vec![]), None)
        } else if t == "default" {
            (None, None)
        } else {
            let list = t.split(',').map(|s| s.trim().to_string()).collect();
            (Some(list), None)
        }
    } else {
        (
            if cli.allowed_tools.is_empty() {
                None
            } else {
                Some(cli.allowed_tools.clone())
            },
            if cli.disallowed_tools.is_empty() {
                None
            } else {
                Some(cli.disallowed_tools.clone())
            },
        )
    };
    ChatOptions {
        tools,
        allowed_tools: effective_allowed,
        disallowed_tools: effective_disallowed,
        system_prompt_override: sys_override,
        system_prompt_append: sys_append,
        additional_dirs: cli.add_dir.clone(),
    }
}

pub(crate) fn ensure_llm_ready(cwd: &Path, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;
    ensure_llm_ready_with_cfg(Some(cwd), &cfg, json_mode)
}

pub(crate) fn ensure_llm_ready_with_cfg(
    cwd: Option<&Path>,
    cfg: &AppConfig,
    json_mode: bool,
) -> Result<()> {
    use std::io::IsTerminal;

    let provider = cfg.llm.provider.trim().to_ascii_lowercase();
    if provider != "deepseek" {
        return Err(anyhow!(
            "unsupported llm.provider='{}' (supported: deepseek)",
            cfg.llm.provider
        ));
    }
    let profile = normalize_deepseek_profile(&cfg.llm.profile).ok_or_else(|| {
        anyhow!(
            "unsupported llm.profile='{}' (supported: v3_2, v3_2_speciale)",
            cfg.llm.profile
        )
    })?;
    if normalize_deepseek_model(&cfg.llm.base_model).is_none() {
        return Err(anyhow!(
            "unsupported llm.base_model='{}' (supported aliases: deepseek-chat, deepseek-reasoner, deepseek-v3.2, deepseek-v3.2-speciale)",
            cfg.llm.base_model
        ));
    }
    if normalize_deepseek_model(&cfg.llm.max_think_model).is_none() {
        return Err(anyhow!(
            "unsupported llm.max_think_model='{}' (supported aliases: deepseek-chat, deepseek-reasoner)",
            cfg.llm.max_think_model
        ));
    }
    let base_lower = cfg.llm.base_model.trim().to_ascii_lowercase();
    if base_lower.contains("speciale") && profile != DEEPSEEK_PROFILE_V32_SPECIALE {
        return Err(anyhow!(
            "llm.base_model='{}' requires llm.profile='v3_2_speciale'",
            cfg.llm.base_model
        ));
    }
    if profile == DEEPSEEK_PROFILE_V32_SPECIALE && !json_mode {
        eprintln!(
            "warning: llm.profile=v3_2_speciale is documented as a limited release ending on {}. Use v3_2 if unavailable.",
            DEEPSEEK_V32_SPECIALE_END_DATE
        );
    }

    let env_key = cfg.llm.api_key_env.trim();
    if env_key.is_empty() {
        return Err(anyhow!(
            "llm.api_key_env is empty; set it in .deepseek/settings.json"
        ));
    }

    if std::env::var(env_key)
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
    {
        return Ok(());
    }

    if let Some(configured_key) = cfg
        .llm
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        // SAFETY: We set process-local environment for this CLI process before worker threads start.
        unsafe {
            std::env::set_var(env_key, configured_key);
        }
        return Ok(());
    }

    let interactive_tty = std::io::stderr().is_terminal();
    if json_mode || !interactive_tty {
        return Err(anyhow!("{} is required. Set it and retry.", env_key));
    }

    eprintln!(
        "API key is required to use provider '{}'.",
        cfg.llm.provider
    );
    let prompt = format!("Enter {}: ", env_key);
    let key = rpassword::prompt_password(prompt)?;
    let trimmed = key.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("received empty API key"));
    }
    // SAFETY: We set process-local environment for this CLI process before worker threads start.
    unsafe {
        std::env::set_var(env_key, trimmed);
    }
    if let Some(cwd) = cwd {
        maybe_persist_api_key(cwd, env_key, trimmed)?;
    }
    Ok(())
}

fn maybe_persist_api_key(cwd: &Path, env_key: &str, api_key: &str) -> Result<()> {
    use std::io::{IsTerminal, Write};

    if !(std::io::stdin().is_terminal() && std::io::stdout().is_terminal()) {
        return Ok(());
    }
    eprint!(
        "Save API key to {} for this workspace? [Y/n]: ",
        AppConfig::project_local_settings_path(cwd).display()
    );
    std::io::stderr().flush()?;
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    let normalized = answer.trim().to_ascii_lowercase();
    if matches!(normalized.as_str(), "n" | "no") {
        return Ok(());
    }

    let local_path = AppConfig::project_local_settings_path(cwd);
    if let Some(parent) = local_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut root = if local_path.exists() {
        let raw = fs::read_to_string(&local_path)?;
        serde_json::from_str::<serde_json::Value>(&raw).unwrap_or_else(|_| json!({}))
    } else {
        json!({})
    };
    if !root.is_object() {
        root = json!({});
    }
    let map = root
        .as_object_mut()
        .ok_or_else(|| anyhow!("settings.local.json root must be an object"))?;
    let llm_value = map.entry("llm".to_string()).or_insert_with(|| json!({}));
    if !llm_value.is_object() {
        *llm_value = json!({});
    }
    if let Some(llm) = llm_value.as_object_mut() {
        llm.insert("api_key".to_string(), json!(api_key));
        llm.insert("api_key_env".to_string(), json!(env_key));
    }
    fs::write(&local_path, serde_json::to_vec_pretty(&root)?)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&local_path)?.permissions();
        perms.set_mode(0o600);
        fs::set_permissions(&local_path, perms)?;
    }
    eprintln!("saved API key in {}", local_path.display());
    Ok(())
}

pub(crate) fn read_session_events(cwd: &Path, session_id: Uuid) -> Result<Vec<EventEnvelope>> {
    let path = runtime_dir(cwd).join("events.jsonl");
    let Ok(raw) = fs::read_to_string(path) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for line in raw.lines() {
        let Ok(event) = serde_json::from_str::<EventEnvelope>(line) else {
            continue;
        };
        if event.session_id == session_id {
            out.push(event);
        }
    }
    Ok(out)
}

pub(crate) fn append_control_event(cwd: &Path, kind: EventKind) -> Result<()> {
    let store = Store::new(cwd)?;
    let session = ensure_session_record(cwd, &store)?;
    let event = EventEnvelope {
        seq_no: store.next_seq_no(session.session_id)?,
        at: chrono::Utc::now(),
        session_id: session.session_id,
        kind,
    };
    store.append_event(&event)?;
    Ok(())
}

pub(crate) fn ensure_session_record(cwd: &Path, store: &Store) -> Result<Session> {
    if let Some(existing) = store.load_latest_session()? {
        return Ok(existing);
    }
    let cfg = AppConfig::load(cwd).unwrap_or_default();
    let session = Session {
        session_id: Uuid::now_v7(),
        workspace_root: cwd.to_string_lossy().to_string(),
        baseline_commit: None,
        status: SessionState::Idle,
        budgets: SessionBudgets {
            per_turn_seconds: cfg.budgets.max_turn_duration_secs,
            max_think_tokens: cfg.budgets.max_reasoner_tokens_per_session as u32,
        },
        active_plan_id: None,
    };
    store.save_session(&session)?;
    Ok(session)
}
