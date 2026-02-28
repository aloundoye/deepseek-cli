use anyhow::Result;
use deepseek_core::AppConfig;
use deepseek_local_ml::model_registry;
use deepseek_local_ml::{ModelManager, ModelStatus};
use serde_json::json;
use std::fs;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

use crate::SetupArgs;
use crate::output::print_json;

/// Marker file that records we already offered local ML setup.
const SETUP_MARKER: &str = ".setup_done";

/// Called from `run_chat` after API key is resolved. Offers local ML setup
/// once on first run, then writes a marker so it never asks again.
pub(crate) fn maybe_offer_local_ml(cwd: &Path, cfg: &AppConfig) -> Result<()> {
    let marker = AppConfig::project_settings_path(cwd)
        .parent()
        .map(|p| p.join(SETUP_MARKER))
        .unwrap_or_else(|| cwd.join(".deepseek").join(SETUP_MARKER));

    if marker.exists() {
        return Ok(());
    }

    // Only prompt in interactive terminals
    if !(std::io::stdin().is_terminal() && std::io::stdout().is_terminal()) {
        return Ok(());
    }

    println!();
    println!("Local ML runs models on your machine for:");
    println!("  - Code retrieval: surfaces relevant code before the LLM responds");
    println!("  - Privacy scanning: detects and redacts secrets before they reach the API");
    println!("  - Ghost text: inline code completions in the TUI");
    println!();

    let enable_ml = prompt_yes_no("Enable local ML? [Y/n]: ")?;

    if enable_ml {
        merge_local_ml_config(cwd, true, true)?;
        println!();
        download_required_models(cfg, false)?;
        println!();
    } else {
        // User declined — record that too so we don't ask again
        merge_local_ml_config(cwd, false, false)?;
        println!();
    }

    write_setup_marker(&marker)?;
    Ok(())
}

fn write_setup_marker(marker: &Path) -> Result<()> {
    if let Some(parent) = marker.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(marker, "")?;
    Ok(())
}

/// Run the setup wizard, `--local-ml` shortcut, or `--status` display.
pub(crate) fn run_setup(cwd: &Path, args: SetupArgs, json_mode: bool) -> Result<()> {
    if args.status {
        return run_status_display(cwd, json_mode);
    }
    if args.local_ml {
        return run_local_ml_shortcut(cwd, json_mode);
    }
    run_interactive_wizard(cwd, json_mode)
}

/// `deepseek setup --status` — show current setup state without prompts.
fn run_status_display(cwd: &Path, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;
    let api_key_set = has_api_key(&cfg);
    let models = model_download_status(&cfg);

    if json_mode {
        print_json(&json!({
            "api_key": if api_key_set { "configured" } else { "missing" },
            "local_ml": {
                "enabled": cfg.local_ml.enabled,
                "privacy_enabled": cfg.local_ml.privacy.enabled,
                "autocomplete_enabled": cfg.local_ml.autocomplete.enabled,
                "models": models,
            },
        }))?;
    } else {
        println!(
            "api_key: {}",
            if api_key_set { "configured" } else { "missing" }
        );
        println!(
            "local_ml: {}",
            if cfg.local_ml.enabled {
                "enabled"
            } else {
                "disabled"
            }
        );
        println!(
            "privacy: {}",
            if cfg.local_ml.privacy.enabled {
                "enabled"
            } else {
                "disabled"
            }
        );
        for (model_id, status) in &models {
            println!("model: {} ({})", model_id, status);
        }
    }
    Ok(())
}

/// `deepseek setup --local-ml` — non-interactive shortcut.
fn run_local_ml_shortcut(cwd: &Path, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;

    merge_local_ml_config(cwd, true, true)?;

    if !json_mode {
        println!("Local ML enabled.");
        println!("Privacy scanning enabled.");
    }

    // Download models immediately
    let download_results = download_required_models(&cfg, json_mode)?;

    if json_mode {
        print_json(&json!({
            "local_ml_enabled": true,
            "privacy_enabled": true,
            "models": download_results,
        }))?;
    } else {
        println!("\nConfig saved to {}", settings_path(cwd).display());
    }
    Ok(())
}

/// Full interactive wizard (3 steps).
fn run_interactive_wizard(cwd: &Path, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;
    let interactive = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();

    if !interactive || json_mode {
        return run_status_display(cwd, json_mode);
    }

    println!("Welcome to DeepSeek CLI setup!\n");

    // Step 1: API Key
    println!("[1/3] API Key");
    if has_api_key(&cfg) {
        println!("  API key is set.\n");
    } else {
        println!(
            "  API key not found. Set {} or run `deepseek chat` to be prompted.\n",
            cfg.llm.api_key_env
        );
    }

    // Step 2: Local ML
    println!("[2/3] Local ML");
    println!("  Local ML runs models on your machine for:");
    println!("  - Code retrieval: surfaces relevant code before the LLM responds");
    println!("  - Privacy scanning: detects and redacts secrets before they reach the API");
    println!("  - Ghost text: inline code completions in the TUI\n");

    let enable_ml = prompt_yes_no("  Enable local ML? [Y/n]: ")?;
    println!();

    // Step 3: Privacy Scanning
    println!("[3/3] Privacy Scanning");
    let enable_privacy = if enable_ml {
        prompt_yes_no("  Enable privacy scanning? [Y/n]: ")?
    } else {
        false
    };
    println!();

    // Write config and download models
    if enable_ml || enable_privacy {
        merge_local_ml_config(cwd, enable_ml, enable_privacy)?;
    }
    if enable_ml {
        download_required_models(&cfg, false)?;
    }

    println!("\nSetup complete! Run `deepseek chat` to start.");
    Ok(())
}

/// Download the default embedding and completion models, showing progress.
/// Returns a list of (model_id, outcome) pairs for JSON output.
fn download_required_models(cfg: &AppConfig, json_mode: bool) -> Result<Vec<(String, String)>> {
    let cache_dir = resolve_cache_dir(cfg);
    let mut manager = ModelManager::new(cache_dir);

    let embedding = model_registry::default_embedding_model();
    let completion = model_registry::default_completion_model();

    let models = [
        (embedding.model_id, embedding.display_name),
        (completion.model_id, completion.display_name),
    ];

    let mut results = Vec::new();

    for (model_id, display_name) in &models {
        let status = manager.status(model_id);
        if status == ModelStatus::Ready {
            if !json_mode {
                println!("  {} ({}) — already downloaded.", display_name, model_id);
            }
            results.push((model_id.to_string(), "ready".to_string()));
            continue;
        }

        if !json_mode {
            print!("  Downloading {} ({})...", display_name, model_id);
            std::io::stdout().flush()?;
        }

        match manager.ensure_model_with_progress(model_id, |current, total| {
            if !json_mode && total > 0 {
                print!(
                    "\r  Downloading {} ({})... [{}/{}]",
                    display_name, model_id, current, total
                );
                let _ = std::io::stdout().flush();
            }
        }) {
            Ok(_) => {
                if !json_mode {
                    println!(
                        "\r  Downloading {} ({})... done.       ",
                        display_name, model_id
                    );
                }
                results.push((model_id.to_string(), "downloaded".to_string()));
            }
            Err(e) => {
                if !json_mode {
                    println!(
                        "\r  Downloading {} ({})... skipped: {}       ",
                        display_name, model_id, e
                    );
                }
                results.push((model_id.to_string(), format!("error: {e}")));
            }
        }
    }

    Ok(results)
}

/// Check download status of the default models without downloading.
fn model_download_status(cfg: &AppConfig) -> Vec<(String, String)> {
    let cache_dir = resolve_cache_dir(cfg);
    let manager = ModelManager::new(cache_dir);

    let embedding = model_registry::default_embedding_model();
    let completion = model_registry::default_completion_model();

    vec![
        (
            embedding.model_id.to_string(),
            format!("{:?}", manager.status(embedding.model_id)),
        ),
        (
            completion.model_id.to_string(),
            format!("{:?}", manager.status(completion.model_id)),
        ),
    ]
}

/// Resolve the model cache directory from config.
fn resolve_cache_dir(cfg: &AppConfig) -> PathBuf {
    PathBuf::from(&cfg.local_ml.cache_dir)
}

/// Prompt user with a yes/no question. Returns true for Y/yes/empty, false for N/no.
fn prompt_yes_no(prompt: &str) -> Result<bool> {
    print!("{prompt}");
    std::io::stdout().flush()?;
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    let normalized = answer.trim().to_ascii_lowercase();
    Ok(!matches!(normalized.as_str(), "n" | "no"))
}

/// Merge local_ml keys into `.deepseek/settings.json` without clobbering other settings.
fn merge_local_ml_config(cwd: &Path, enabled: bool, privacy_enabled: bool) -> Result<()> {
    let path = settings_path(cwd);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut root = if path.exists() {
        let raw = fs::read_to_string(&path)?;
        serde_json::from_str::<serde_json::Value>(&raw).unwrap_or_else(|_| json!({}))
    } else {
        json!({})
    };
    if !root.is_object() {
        root = json!({});
    }

    let map = root
        .as_object_mut()
        .expect("root is guaranteed to be an object");
    let local_ml = map
        .entry("local_ml".to_string())
        .or_insert_with(|| json!({}));
    if !local_ml.is_object() {
        *local_ml = json!({});
    }
    if let Some(ml) = local_ml.as_object_mut() {
        ml.insert("enabled".to_string(), json!(enabled));
        let privacy = ml.entry("privacy".to_string()).or_insert_with(|| json!({}));
        if !privacy.is_object() {
            *privacy = json!({});
        }
        if let Some(p) = privacy.as_object_mut() {
            p.insert("enabled".to_string(), json!(privacy_enabled));
        }
    }

    fs::write(&path, serde_json::to_vec_pretty(&root)?)?;
    Ok(())
}

/// Project settings path (`.deepseek/settings.json`).
fn settings_path(cwd: &Path) -> PathBuf {
    AppConfig::project_settings_path(cwd)
}

/// Check whether the API key is available (env var or config).
fn has_api_key(cfg: &AppConfig) -> bool {
    let env_set = std::env::var(&cfg.llm.api_key_env)
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);
    let config_set = cfg
        .llm
        .api_key
        .as_deref()
        .map(str::trim)
        .is_some_and(|v| !v.is_empty());
    env_set || config_set
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn setup_merges_config_preserves_existing() {
        let tmp = TempDir::new().unwrap();
        let cwd = tmp.path();

        // Write pre-existing settings
        let dir = cwd.join(".deepseek");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        fs::write(
            &path,
            serde_json::to_vec_pretty(&json!({
                "llm": { "profile": "v3_2" },
                "custom_key": "preserve_me"
            }))
            .unwrap(),
        )
        .unwrap();

        // Merge local_ml config
        merge_local_ml_config(cwd, true, true).unwrap();

        // Read back and verify
        let raw = fs::read_to_string(&path).unwrap();
        let root: serde_json::Value = serde_json::from_str(&raw).unwrap();

        // Existing keys preserved
        assert_eq!(root["custom_key"], "preserve_me");
        assert_eq!(root["llm"]["profile"], "v3_2");

        // New keys written
        assert_eq!(root["local_ml"]["enabled"], true);
        assert_eq!(root["local_ml"]["privacy"]["enabled"], true);
    }

    #[test]
    fn model_download_status_returns_entries() {
        let cfg = AppConfig::default();
        let statuses = model_download_status(&cfg);
        assert_eq!(statuses.len(), 2);
        // Both should report NotDownloaded for a fresh config
        for (model_id, status) in &statuses {
            assert!(!model_id.is_empty());
            assert!(status.contains("NotDownloaded") || status.contains("Ready"));
        }
    }

    #[test]
    fn resolve_cache_dir_uses_config() {
        let cfg = AppConfig::default();
        let dir = resolve_cache_dir(&cfg);
        assert_eq!(dir, PathBuf::from(".deepseek/models"));
    }

    #[test]
    fn maybe_offer_skips_when_already_enabled() {
        let tmp = TempDir::new().unwrap();
        let cwd = tmp.path();
        let dir = cwd.join(".deepseek");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("settings.json"),
            serde_json::to_vec_pretty(&json!({ "local_ml": { "enabled": true } })).unwrap(),
        )
        .unwrap();

        let cfg = AppConfig::ensure(cwd).unwrap();
        // Should write the marker without prompting (non-interactive in test)
        maybe_offer_local_ml(cwd, &cfg).unwrap();

        // Marker should exist
        assert!(dir.join(SETUP_MARKER).exists());
    }

    #[test]
    fn maybe_offer_skips_when_marker_exists() {
        let tmp = TempDir::new().unwrap();
        let cwd = tmp.path();
        let dir = cwd.join(".deepseek");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("settings.json"), "{}").unwrap();
        fs::write(dir.join(SETUP_MARKER), "").unwrap();

        let cfg = AppConfig::ensure(cwd).unwrap();
        // Should return immediately — no prompt, no config change
        maybe_offer_local_ml(cwd, &cfg).unwrap();
    }
}
