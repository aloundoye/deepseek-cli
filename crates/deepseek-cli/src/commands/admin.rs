use anyhow::{Result, anyhow};
use deepseek_agent::{AgentEngine, ChatOptions};
use deepseek_core::{AppConfig, EventKind, ToolCall, normalize_deepseek_profile, runtime_dir};
use deepseek_index::IndexService;
use deepseek_policy::{PolicyEngine, TeamPolicyLocks, team_policy_locks};
use deepseek_store::Store;
use deepseek_tools::PluginManager;
use serde_json::json;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use crate::commands::intelligence::{
    DebugAnalysisMode, FrameworkReport, analyze_debug_text, detect_frameworks,
};
use crate::commands::leadership::{LeadershipReport, build_leadership_report};
use crate::context::*;
use crate::output::*;
use crate::util::*;
use crate::{
    CleanArgs, ConfigCmd, DoctorArgs, DoctorModeArg, IndexCmd, PermissionModeArg, PermissionsCmd,
    PermissionsSetArgs, PluginCmd,
};

pub(crate) fn parse_permissions_cmd(args: Vec<String>) -> Result<PermissionsCmd> {
    if args.is_empty() || args[0].eq_ignore_ascii_case("show") {
        return Ok(PermissionsCmd::Show);
    }

    let first = args[0].to_ascii_lowercase();
    if first == "bash" {
        let mode = args
            .get(1)
            .ok_or_else(|| anyhow!("usage: /permissions bash <ask|always|never>"))?;
        return Ok(PermissionsCmd::Set(PermissionsSetArgs {
            approve_bash: Some(parse_permission_mode(mode)?),
            ..PermissionsSetArgs::default()
        }));
    }
    if first == "edits" {
        let mode = args
            .get(1)
            .ok_or_else(|| anyhow!("usage: /permissions edits <ask|always|never>"))?;
        return Ok(PermissionsCmd::Set(PermissionsSetArgs {
            approve_edits: Some(parse_permission_mode(mode)?),
            ..PermissionsSetArgs::default()
        }));
    }
    if first == "sandbox" {
        let mode = args
            .get(1)
            .ok_or_else(|| anyhow!("usage: /permissions sandbox <mode>"))?;
        return Ok(PermissionsCmd::Set(PermissionsSetArgs {
            sandbox_mode: Some(mode.clone()),
            ..PermissionsSetArgs::default()
        }));
    }
    if first == "allow" {
        let entry = args
            .iter()
            .skip(1)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(" ");
        if entry.trim().is_empty() {
            return Err(anyhow!("usage: /permissions allow <command-prefix>"));
        }
        return Ok(PermissionsCmd::Set(PermissionsSetArgs {
            allow: vec![entry],
            ..PermissionsSetArgs::default()
        }));
    }
    if first == "clear-allowlist" {
        return Ok(PermissionsCmd::Set(PermissionsSetArgs {
            clear_allowlist: true,
            ..PermissionsSetArgs::default()
        }));
    }

    if first == "dry-run" || first == "dryrun" {
        let tool_name = args
            .get(1)
            .ok_or_else(|| anyhow!("usage: /permissions dry-run <tool-name>"))?;
        return Ok(PermissionsCmd::DryRun(crate::PermissionsDryRunArgs {
            tool_name: tool_name.clone(),
        }));
    }

    if first != "set" {
        return Err(anyhow!(
            "use /permissions show|set|bash|edits|sandbox|allow|clear-allowlist|dry-run"
        ));
    }

    let mut parsed = PermissionsSetArgs::default();
    let mut idx = 1usize;
    while idx < args.len() {
        let token = args[idx].to_ascii_lowercase();
        match token.as_str() {
            "--approve-bash" | "approve-bash" => {
                idx += 1;
                let value = args.get(idx).ok_or_else(|| {
                    anyhow!("usage: /permissions set --approve-bash <ask|always|never>")
                })?;
                parsed.approve_bash = Some(parse_permission_mode(value)?);
            }
            "--approve-edits" | "approve-edits" => {
                idx += 1;
                let value = args.get(idx).ok_or_else(|| {
                    anyhow!("usage: /permissions set --approve-edits <ask|always|never>")
                })?;
                parsed.approve_edits = Some(parse_permission_mode(value)?);
            }
            "--sandbox-mode" | "sandbox-mode" => {
                idx += 1;
                let value = args
                    .get(idx)
                    .ok_or_else(|| anyhow!("usage: /permissions set --sandbox-mode <mode>"))?;
                parsed.sandbox_mode = Some(value.clone());
            }
            "--allow" | "allow" => {
                idx += 1;
                let value = args
                    .get(idx)
                    .ok_or_else(|| anyhow!("usage: /permissions set --allow <command-prefix>"))?;
                parsed.allow.push(value.clone());
            }
            "--clear-allowlist" | "clear-allowlist" => {
                parsed.clear_allowlist = true;
            }
            _ => {
                return Err(anyhow!(
                    "unknown permissions option: {} (expected --approve-bash/--approve-edits/--sandbox-mode/--allow/--clear-allowlist)",
                    args[idx]
                ));
            }
        }
        idx += 1;
    }

    Ok(PermissionsCmd::Set(parsed))
}

pub(crate) fn parse_permission_mode(value: &str) -> Result<PermissionModeArg> {
    match value.to_ascii_lowercase().as_str() {
        "ask" => Ok(PermissionModeArg::Ask),
        "always" => Ok(PermissionModeArg::Always),
        "never" => Ok(PermissionModeArg::Never),
        _ => Err(anyhow!(
            "invalid permission mode '{}' (expected ask|always|never)",
            value
        )),
    }
}

pub(crate) fn run_doctor(cwd: &Path, args: DoctorArgs, json_mode: bool) -> Result<()> {
    let payload = doctor_payload(cwd, &args)?;
    if json_mode {
        print_json(&payload)?;
    } else {
        println!(
            "doctor: os={} arch={} shell={}",
            payload["os"].as_str().unwrap_or_default(),
            payload["arch"].as_str().unwrap_or_default(),
            payload["shell"].as_str().unwrap_or_default()
        );
        println!(
            "toolchain: {} | {}",
            payload["toolchain"]["rustc"]
                .as_str()
                .unwrap_or("unavailable"),
            payload["toolchain"]["cargo"]
                .as_str()
                .unwrap_or("unavailable")
        );
        println!(
            "llm: profile={} base={} max={} endpoint={} api_key_env_set={} api_key_configured={}",
            payload["llm"]["profile"].as_str().unwrap_or_default(),
            payload["llm"]["base_model"].as_str().unwrap_or_default(),
            payload["llm"]["max_think_model"]
                .as_str()
                .unwrap_or_default(),
            payload["llm"]["endpoint"].as_str().unwrap_or_default(),
            payload["llm"]["api_key_env_set"].as_bool().unwrap_or(false),
            payload["llm"]["api_key_configured"]
                .as_bool()
                .unwrap_or(false),
        );
        println!(
            "plugins: enabled={} installed={}",
            payload["plugins"]["enabled"].as_u64().unwrap_or(0),
            payload["plugins"]["installed"].as_u64().unwrap_or(0)
        );
        let framework_names = payload["frameworks"]["detected"]
            .as_array()
            .map(|rows| {
                rows.iter()
                    .filter_map(|row| row["name"].as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        if framework_names.is_empty() {
            println!("frameworks: none detected");
        } else {
            println!(
                "frameworks: {} (primary={})",
                framework_names,
                payload["frameworks"]["primary_ecosystem"]
                    .as_str()
                    .unwrap_or("unknown")
            );
        }
        if let Some(leadership) = payload.get("leadership") {
            println!(
                "leadership: score={} ok={} integrations={} deployment_risks={}",
                leadership["readiness"]["score"].as_u64().unwrap_or(0),
                leadership["readiness"]["ok"].as_bool().unwrap_or(false),
                leadership["ecosystem"]["integrations"]
                    .as_array()
                    .map(|rows| rows.len())
                    .unwrap_or(0),
                leadership["deployment"]["risks"]
                    .as_array()
                    .map(|rows| rows.len())
                    .unwrap_or(0)
            );
        }
        if let Some(debug) = payload.get("debug_analysis")
            && !debug.is_null()
        {
            println!(
                "debug analysis: mode={} issues={}",
                debug["mode"].as_str().unwrap_or("unknown"),
                debug["issues"]
                    .as_array()
                    .map(|rows| rows.len())
                    .unwrap_or(0)
            );
        }
        if let Some(warnings) = payload["warnings"].as_array()
            && !warnings.is_empty()
        {
            println!("warnings:");
            for warning in warnings {
                if let Some(text) = warning.as_str() {
                    println!("- {text}");
                }
            }
        }
    }

    Ok(())
}

pub(crate) fn doctor_payload(cwd: &Path, args: &DoctorArgs) -> Result<serde_json::Value> {
    let cfg = AppConfig::ensure(cwd)?;
    let plugin_manager = PluginManager::new(cwd)?;
    let plugins = plugin_manager.list().unwrap_or_default();

    let runtime = runtime_dir(cwd);
    fs::create_dir_all(&runtime)?;
    let rustc = run_capture("rustc", &["--version"]);
    let cargo = run_capture("cargo", &["--version"]);
    let shell = std::env::var("SHELL")
        .ok()
        .or_else(|| std::env::var("ComSpec").ok())
        .unwrap_or_else(|| "unknown".to_string());
    let api_key_env_set = std::env::var(&cfg.llm.api_key_env)
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);
    let api_key_configured = cfg
        .llm
        .api_key
        .as_deref()
        .map(str::trim)
        .is_some_and(|value| !value.is_empty());
    let _profile = normalize_deepseek_profile(&cfg.llm.profile).unwrap_or("invalid");

    let checks = json!({
        "git": command_exists("git"),
        "rg": command_exists("rg"),
        "cargo": command_exists("cargo"),
        "shell": command_exists(shell.split(std::path::MAIN_SEPARATOR).next_back().unwrap_or("sh")),
    });

    let mut warnings = Vec::new();
    if !api_key_env_set && !api_key_configured {
        warnings.push(format!(
            "{} not set and llm.api_key not configured",
            cfg.llm.api_key_env
        ));
    }
    if checks["git"].as_bool() != Some(true) {
        warnings.push("git not found in PATH".to_string());
    }
    if checks["cargo"].as_bool() != Some(true) {
        warnings.push("cargo not found in PATH".to_string());
    }

    let frameworks = match detect_frameworks(cwd) {
        Ok(report) => report,
        Err(err) => {
            warnings.push(format!("framework detection failed: {err}"));
            FrameworkReport {
                detected: Vec::new(),
                primary_ecosystem: "unknown".to_string(),
                recommendations: Vec::new(),
            }
        }
    };
    let leadership = match build_leadership_report(cwd, 24) {
        Ok(report) => report,
        Err(err) => {
            warnings.push(format!("leadership analysis failed: {err}"));
            LeadershipReport::default()
        }
    };

    let debug_source = if let Some(file) = args.analyze_file.as_deref() {
        Some(json!({
            "kind": "file",
            "path": file,
        }))
    } else if args.analyze_text.is_some() {
        Some(json!({
            "kind": "inline",
            "path": serde_json::Value::Null,
        }))
    } else {
        None
    };

    let debug_analysis = if let Some(text) = args.analyze_text.as_deref() {
        Some(analyze_debug_text(text, debug_mode_from_arg(args.mode)))
    } else if let Some(path) = args.analyze_file.as_deref() {
        let content = fs::read_to_string(path).map_err(|err| {
            anyhow!(
                "failed to read --analyze-file {}: {}",
                Path::new(path).display(),
                err
            )
        })?;
        Some(analyze_debug_text(&content, debug_mode_from_arg(args.mode)))
    } else {
        None
    };

    let payload = json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "shell": shell,
        "workspace": cwd,
        "runtime_dir": runtime,
        "config_path": AppConfig::config_path(cwd),
        "config_paths": {
            "user": AppConfig::user_settings_path(),
            "project": AppConfig::project_settings_path(cwd),
            "project_local": AppConfig::project_local_settings_path(cwd),
            "legacy_toml": AppConfig::legacy_toml_path(cwd),
            "keybindings": AppConfig::keybindings_path(),
        },
        "binary_path": std::env::current_exe().ok(),
        "toolchain": {
            "rustc": rustc,
            "cargo": cargo,
        },
        "llm": {
            "endpoint": cfg.llm.endpoint,
            "profile": cfg.llm.profile,
            "api_key_env": cfg.llm.api_key_env,
            "api_key_env_set": api_key_env_set,
            "api_key_configured": api_key_configured,
            "base_model": cfg.llm.base_model,
            "max_think_model": cfg.llm.max_think_model,
        },
        "plugins": {
            "installed": plugins.len(),
            "enabled": plugins.iter().filter(|p| p.enabled).count(),
        },
        "checks": checks,
        "frameworks": frameworks,
        "leadership": leadership,
        "debug_source": debug_source,
        "debug_analysis": debug_analysis,
        "warnings": warnings,
    });

    Ok(payload)
}

fn debug_mode_from_arg(mode: DoctorModeArg) -> DebugAnalysisMode {
    match mode {
        DoctorModeArg::Auto => DebugAnalysisMode::Auto,
        DoctorModeArg::Runtime => DebugAnalysisMode::Runtime,
        DoctorModeArg::Test => DebugAnalysisMode::Test,
        DoctorModeArg::Performance => DebugAnalysisMode::Performance,
    }
}

pub(crate) fn run_index(cwd: &Path, cmd: IndexCmd, json_mode: bool) -> Result<()> {
    let service = IndexService::new(cwd)?;
    let store = Store::new(cwd)?;
    let session = ensure_session_record(cwd, &store)?;

    match cmd {
        IndexCmd::Build { hybrid } => {
            let manifest = service.build(&session)?;
            if hybrid {
                // Log that hybrid mode was requested (vector index built separately)
                if !json_mode {
                    println!("hybrid mode: BM25 index built; vector index via local-ml");
                }
            }
            if json_mode {
                print_json(&serde_json::json!({
                    "manifest": manifest,
                    "hybrid": hybrid,
                }))?;
            } else {
                println!("{}", serde_json::to_string_pretty(&manifest)?);
            }
        }
        IndexCmd::Update => {
            let manifest = service.update(&session)?;
            if json_mode {
                print_json(&manifest)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&manifest)?);
            }
        }
        IndexCmd::Status => {
            let status = service.status()?;
            if json_mode {
                print_json(&status)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&status)?);
            }
        }
        IndexCmd::Watch {
            events,
            timeout_seconds,
        } => {
            let manifest = service.watch_and_update(
                &session,
                events.max(1),
                Duration::from_secs(timeout_seconds.max(1)),
            )?;
            if json_mode {
                print_json(&manifest)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&manifest)?);
            }
        }
        IndexCmd::Query { q, top_k } => {
            let result = service.query(&q, top_k, None)?;
            if json_mode {
                print_json(&result)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
        }
        IndexCmd::Doctor => {
            let status = service.status()?;
            let index_path = cwd.join(".deepseek").join("index");
            let has_index = index_path.exists();
            let result = serde_json::json!({
                "index_exists": has_index,
                "index_path": index_path.to_string_lossy(),
                "status": status,
            });
            if json_mode {
                print_json(&result)?;
            } else {
                println!("Index diagnostics:");
                println!("  index exists: {}", has_index);
                println!("  index path: {}", index_path.display());
                println!("{}", serde_json::to_string_pretty(&status)?);
            }
        }
        IndexCmd::Clean => {
            let index_path = cwd.join(".deepseek").join("index");
            if index_path.exists() {
                std::fs::remove_dir_all(&index_path)?;
                if json_mode {
                    print_json(&serde_json::json!({"cleaned": true, "path": index_path.to_string_lossy()}))?;
                } else {
                    println!("Index cleaned: {}", index_path.display());
                }
            } else if json_mode {
                print_json(&serde_json::json!({"cleaned": false, "reason": "no index found"}))?;
            } else {
                println!("No index to clean.");
            }
        }
    }
    Ok(())
}

pub(crate) fn run_config(cwd: &Path, cmd: ConfigCmd, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;
    let cfg_path = AppConfig::config_path(cwd);

    match cmd {
        ConfigCmd::Show => {
            let display_cfg = redact_config_for_display(&cfg)?;
            if json_mode {
                print_json(&display_cfg)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&display_cfg)?);
            }
        }
        ConfigCmd::Edit => {
            let editor = std::env::var("EDITOR").unwrap_or_else(|_| default_editor().to_string());
            let status = Command::new(editor).arg(&cfg_path).status()?;
            if !status.success() {
                return Err(anyhow!("editor exited with non-zero status"));
            }
        }
    }
    Ok(())
}

pub(crate) fn run_permissions(cwd: &Path, cmd: PermissionsCmd, json_mode: bool) -> Result<()> {
    let payload = permissions_payload(cwd, cmd)?;
    if json_mode {
        print_json(&payload)?;
    } else if payload.get("dry_run").is_some() {
        println!(
            "dry-run: tool={} mode={} result={}",
            payload["dry_run"]["tool"].as_str().unwrap_or_default(),
            payload["dry_run"]["permission_mode"]
                .as_str()
                .unwrap_or_default(),
            payload["dry_run"]["result"].as_str().unwrap_or_default(),
        );
    } else {
        println!(
            "permissions: bash={} edits={} sandbox={} allowlist={}",
            payload["policy"]["approve_bash"]
                .as_str()
                .unwrap_or_default(),
            payload["policy"]["approve_edits"]
                .as_str()
                .unwrap_or_default(),
            payload["policy"]["sandbox_mode"]
                .as_str()
                .unwrap_or_default(),
            payload["policy"]["allowlist_entries"].as_u64().unwrap_or(0),
        );
        if payload["updated"].as_bool().unwrap_or(false) {
            println!(
                "updated project permissions at {}",
                AppConfig::config_path(cwd).display()
            );
        }
        if payload["team_policy"]["active"].as_bool().unwrap_or(false) {
            println!(
                "team policy lock active at {}",
                payload["team_policy"]["path"].as_str().unwrap_or_default()
            );
        }
    }
    Ok(())
}

pub(crate) fn permissions_payload(cwd: &Path, cmd: PermissionsCmd) -> Result<serde_json::Value> {
    let mut cfg = AppConfig::ensure(cwd)?;
    let mut updated = false;
    let team_locks = team_policy_locks();
    match cmd {
        PermissionsCmd::Show => {}
        PermissionsCmd::Set(args) => {
            if let Some(locks) = team_locks.as_ref() {
                let locked_fields = locked_permission_fields_for_set(&args, locks);
                if !locked_fields.is_empty() {
                    return Err(anyhow!(
                        "team policy at {} locks permissions fields: {}",
                        locks.path,
                        locked_fields.join(", ")
                    ));
                }
            }
            if let Some(mode) = args.approve_bash {
                let value = mode.to_approval_mode();
                if cfg.policy.approve_bash != value {
                    cfg.policy.approve_bash = value;
                    updated = true;
                }
            }
            if let Some(mode) = args.approve_edits {
                let value = mode.to_approval_mode();
                if cfg.policy.approve_edits != value {
                    cfg.policy.approve_edits = value;
                    updated = true;
                }
            }
            if let Some(mode) = args.sandbox_mode {
                let mode = mode.trim();
                if mode.is_empty() {
                    return Err(anyhow!("sandbox_mode cannot be empty"));
                }
                let parsed: deepseek_core::SandboxMode =
                    mode.parse().map_err(|e: anyhow::Error| anyhow!("{e}"))?;
                if cfg.policy.sandbox_mode != parsed {
                    cfg.policy.sandbox_mode = parsed;
                    updated = true;
                }
            }
            if args.clear_allowlist && !cfg.policy.allowlist.is_empty() {
                cfg.policy.allowlist.clear();
                updated = true;
            }
            for entry in args.allow {
                let trimmed = entry.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if !cfg
                    .policy
                    .allowlist
                    .iter()
                    .any(|existing| existing == trimmed)
                {
                    cfg.policy.allowlist.push(trimmed.to_string());
                    updated = true;
                }
            }
            if updated {
                cfg.save(cwd)?;
            }
        }
        PermissionsCmd::DryRun(args) => {
            let engine = PolicyEngine::from_app_config(&cfg.policy);
            let call = ToolCall {
                name: args.tool_name.clone(),
                args: serde_json::json!({}),
                requires_approval: false,
            };
            let result = engine.dry_run(&call);
            let verdict = format!("{:?}", result);
            return Ok(json!({
                "dry_run": {
                    "tool": args.tool_name,
                    "permission_mode": cfg.policy.permission_mode,
                    "result": verdict,
                }
            }));
        }
    }

    Ok(json!({
        "updated": updated,
        "policy": {
            "approve_bash": cfg.policy.approve_bash,
            "approve_edits": cfg.policy.approve_edits,
            "sandbox_mode": cfg.policy.sandbox_mode,
            "allowlist_entries": cfg.policy.allowlist.len(),
            "allowlist": cfg.policy.allowlist,
        },
        "team_policy": team_locks
            .as_ref()
            .map(|locks| json!({
                "active": true,
                "path": locks.path,
                "approve_edits_locked": locks.approve_edits_locked,
                "approve_bash_locked": locks.approve_bash_locked,
                "allowlist_locked": locks.allowlist_locked,
                "sandbox_mode_locked": locks.sandbox_mode_locked,
                "permission_mode_locked": locks.permission_mode_locked,
            }))
            .unwrap_or_else(|| json!({"active": false}))
    }))
}

pub(crate) fn locked_permission_fields_for_set(
    args: &PermissionsSetArgs,
    locks: &TeamPolicyLocks,
) -> Vec<&'static str> {
    let mut fields = Vec::new();
    if locks.approve_bash_locked && args.approve_bash.is_some() {
        fields.push("approve_bash");
    }
    if locks.approve_edits_locked && args.approve_edits.is_some() {
        fields.push("approve_edits");
    }
    if locks.sandbox_mode_locked && args.sandbox_mode.is_some() {
        fields.push("sandbox_mode");
    }
    if locks.allowlist_locked && (args.clear_allowlist || !args.allow.is_empty()) {
        fields.push("allowlist");
    }
    fields
}

pub(crate) fn run_plugins(cwd: &Path, cmd: PluginCmd, json_mode: bool) -> Result<()> {
    let manager = PluginManager::new(cwd)?;
    match cmd {
        PluginCmd::List(args) => {
            if args.discover {
                let cfg = AppConfig::load(cwd)?;
                let found = manager.discover(&cfg.plugins.search_paths)?;
                if json_mode {
                    print_json(&found)?;
                } else {
                    for p in found {
                        println!(
                            "{} {} {}",
                            p.manifest.id,
                            p.manifest.version,
                            p.root.display()
                        );
                    }
                }
            } else {
                let installed = manager.list()?;
                if json_mode {
                    print_json(&installed)?;
                } else {
                    for p in installed {
                        let state = if p.enabled { "enabled" } else { "disabled" };
                        println!("{} {} ({})", p.manifest.id, p.manifest.version, state);
                    }
                }
            }
        }
        PluginCmd::Install(args) => {
            let info = manager.install(Path::new(&args.source))?;
            append_control_event(
                cwd,
                EventKind::PluginInstalled {
                    plugin_id: info.manifest.id.clone(),
                    version: info.manifest.version.clone(),
                },
            )?;
            if json_mode {
                print_json(&info)?;
            } else {
                println!("installed {} {}", info.manifest.id, info.manifest.version);
            }
        }
        PluginCmd::Remove(args) => {
            manager.remove(&args.plugin_id)?;
            append_control_event(
                cwd,
                EventKind::PluginRemoved {
                    plugin_id: args.plugin_id.clone(),
                },
            )?;
            if json_mode {
                print_json(&json!({"removed": args.plugin_id}))?;
            } else {
                println!("removed {}", args.plugin_id);
            }
        }
        PluginCmd::Enable(args) => {
            manager.enable(&args.plugin_id)?;
            append_control_event(
                cwd,
                EventKind::PluginEnabled {
                    plugin_id: args.plugin_id.clone(),
                },
            )?;
            if json_mode {
                print_json(&json!({"enabled": args.plugin_id}))?;
            } else {
                println!("enabled {}", args.plugin_id);
            }
        }
        PluginCmd::Disable(args) => {
            manager.disable(&args.plugin_id)?;
            append_control_event(
                cwd,
                EventKind::PluginDisabled {
                    plugin_id: args.plugin_id.clone(),
                },
            )?;
            if json_mode {
                print_json(&json!({"disabled": args.plugin_id}))?;
            } else {
                println!("disabled {}", args.plugin_id);
            }
        }
        PluginCmd::Inspect(args) => {
            let info = manager.inspect(&args.plugin_id)?;
            if json_mode {
                print_json(&info)?;
            } else {
                println!(
                    "id={} version={} root={}",
                    info.manifest.id,
                    info.manifest.version,
                    info.root.display()
                );
                println!(
                    "commands={} agents={} skills={} hooks={}",
                    info.commands.len(),
                    info.agents.len(),
                    info.skills.len(),
                    info.hooks.len()
                );
            }
        }
        PluginCmd::Catalog => {
            let cfg = AppConfig::load(cwd)?;
            let catalog = manager
                .sync_catalog(&cfg.plugins.catalog)
                .or_else(|_| manager.search_catalog("", &cfg.plugins.catalog))?;
            append_control_event(
                cwd,
                EventKind::PluginCatalogSynced {
                    source: cfg.plugins.catalog.index_url,
                    total: catalog.len(),
                    verified_count: catalog.iter().filter(|p| p.verified).count(),
                },
            )?;
            if json_mode {
                print_json(&catalog)?;
            } else {
                for item in catalog {
                    println!(
                        "{} {} ({}) {}",
                        item.plugin_id,
                        item.version,
                        if item.verified {
                            "verified"
                        } else {
                            "unverified"
                        },
                        item.source
                    );
                }
            }
        }
        PluginCmd::Search(args) => {
            let cfg = AppConfig::load(cwd)?;
            let matches = manager.search_catalog(&args.query, &cfg.plugins.catalog)?;
            if json_mode {
                print_json(&matches)?;
            } else if matches.is_empty() {
                println!("no catalog results for '{}'", args.query);
            } else {
                for item in matches {
                    println!(
                        "{} {} ({}) - {}",
                        item.plugin_id,
                        item.version,
                        if item.verified {
                            "verified"
                        } else {
                            "unverified"
                        },
                        item.description
                    );
                }
            }
        }
        PluginCmd::Verify(args) => {
            let cfg = AppConfig::load(cwd)?;
            let result = manager.verify_catalog_plugin(&args.plugin_id, &cfg.plugins.catalog)?;
            append_control_event(
                cwd,
                EventKind::PluginVerified {
                    plugin_id: result.plugin_id.clone(),
                    verified: result.verified,
                    reason: result.reason.clone(),
                },
            )?;
            if json_mode {
                print_json(&result)?;
            } else {
                println!(
                    "{} verified={} reason={} source={}",
                    result.plugin_id, result.verified, result.reason, result.source
                );
            }
        }
        PluginCmd::Run(args) => {
            ensure_llm_ready(cwd, json_mode)?;
            let rendered = manager.render_command_prompt(
                &args.plugin_id,
                &args.command_name,
                args.input.as_deref(),
            )?;
            let engine = AgentEngine::new(cwd)?;
            let output = engine.chat_with_options(
                &rendered.prompt,
                ChatOptions {
                    tools: args.tools,
                    ..Default::default()
                },
            )?;
            if json_mode {
                print_json(&json!({
                    "plugin_id": rendered.plugin_id,
                    "command_name": rendered.command_name,
                    "source_path": rendered.source_path,
                    "tools": args.tools,
                    "max_think": args.max_think,
                    "output": output
                }))?;
            } else {
                println!(
                    "plugin command {}:{} ({})",
                    rendered.plugin_id,
                    rendered.command_name,
                    rendered.source_path.display()
                );
                println!("{output}");
            }
        }
    }
    Ok(())
}

pub(crate) fn run_clean(cwd: &Path, args: CleanArgs, json_mode: bool) -> Result<()> {
    let candidates = vec![
        cwd.join(".deepseek/patches"),
        cwd.join(".deepseek/observe.log"),
        cwd.join(".deepseek/index/tantivy"),
    ];
    let mut removed = Vec::new();
    for path in candidates {
        if !path.exists() {
            continue;
        }
        if !args.dry_run {
            if path.is_dir() {
                fs::remove_dir_all(&path)?;
            } else {
                fs::remove_file(&path)?;
            }
        }
        removed.push(path.to_string_lossy().to_string());
    }

    if json_mode {
        print_json(&json!({"dry_run": args.dry_run, "removed": removed}))?;
    } else if args.dry_run {
        println!("would remove:\n{}", removed.join("\n"));
    } else {
        println!("removed:\n{}", removed.join("\n"));
    }
    Ok(())
}
