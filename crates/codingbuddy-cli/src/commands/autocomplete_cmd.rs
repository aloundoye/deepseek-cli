//! CLI commands for local ML autocomplete management.

use anyhow::Result;
use codingbuddy_core::AppConfig;
use serde_json::json;
use std::path::Path;

use crate::output::print_json;

/// Autocomplete subcommands.
#[derive(clap::Subcommand)]
pub(crate) enum AutocompleteCmd {
    /// Enable local ML autocomplete.
    Enable,
    /// Disable local ML autocomplete.
    Disable,
    /// Show current autocomplete status.
    Status,
    /// Manage autocomplete model.
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },
}

#[derive(clap::Subcommand)]
pub(crate) enum ModelAction {
    /// Set the autocomplete model.
    Set {
        /// Model identifier (e.g. "deepseek-coder-1.3b-q4").
        model_id: String,
    },
}

pub(crate) fn run_autocomplete(cwd: &Path, cmd: AutocompleteCmd, json_mode: bool) -> Result<()> {
    let mut cfg = AppConfig::load(cwd).unwrap_or_default();

    match cmd {
        AutocompleteCmd::Enable => {
            cfg.local_ml.autocomplete.enabled = true;
            save_config(cwd, &cfg)?;
            if json_mode {
                print_json(&json!({"autocomplete": "enabled"}))?;
            } else {
                println!("Local ML autocomplete enabled.");
            }
        }
        AutocompleteCmd::Disable => {
            cfg.local_ml.autocomplete.enabled = false;
            save_config(cwd, &cfg)?;
            if json_mode {
                print_json(&json!({"autocomplete": "disabled"}))?;
            } else {
                println!("Local ML autocomplete disabled.");
            }
        }
        AutocompleteCmd::Status => {
            let status = json!({
                "enabled": cfg.local_ml.autocomplete.enabled,
                "model_id": cfg.local_ml.autocomplete.model_id,
                "debounce_ms": cfg.local_ml.autocomplete.debounce_ms,
                "timeout_ms": cfg.local_ml.autocomplete.timeout_ms,
                "max_tokens": cfg.local_ml.autocomplete.max_tokens,
            });
            if json_mode {
                print_json(&status)?;
            } else {
                println!("Autocomplete status:");
                println!("  enabled: {}", cfg.local_ml.autocomplete.enabled);
                println!("  model: {}", cfg.local_ml.autocomplete.model_id);
                println!("  debounce: {}ms", cfg.local_ml.autocomplete.debounce_ms);
                println!("  timeout: {}ms", cfg.local_ml.autocomplete.timeout_ms);
                println!("  max tokens: {}", cfg.local_ml.autocomplete.max_tokens);
            }
        }
        AutocompleteCmd::Model { action } => match action {
            ModelAction::Set { model_id } => {
                cfg.local_ml.autocomplete.model_id = model_id.clone();
                save_config(cwd, &cfg)?;
                if json_mode {
                    print_json(&json!({"model_id": model_id}))?;
                } else {
                    println!("Autocomplete model set to: {}", model_id);
                }
            }
        },
    }
    Ok(())
}

fn save_config(cwd: &Path, cfg: &AppConfig) -> Result<()> {
    let path = AppConfig::project_settings_path(cwd);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(cfg)?;
    std::fs::write(&path, json)?;
    Ok(())
}
