use anyhow::Result;
use codingbuddy_core::AppConfig;
use codingbuddy_local_ml::model_registry;
use codingbuddy_local_ml::{ModelManager, ModelStatus};
use serde_json::json;
use std::fs;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

use crate::SetupArgs;
use crate::output::print_json;

/// Marker file that records we already ran first-time setup.
const SETUP_MARKER: &str = ".setup_done";

/// Called from `run_chat` on first run. Walks the user through the same
/// setup wizard as `codingbuddy setup` (provider, API key, local ML, privacy),
/// then writes a marker so it never asks again.
///
/// Returns `true` if the wizard ran and config was potentially modified.
pub(crate) fn maybe_first_time_setup(cwd: &Path, cfg: &AppConfig) -> Result<bool> {
    let marker = AppConfig::project_settings_path(cwd)
        .parent()
        .map(|p| p.join(SETUP_MARKER))
        .unwrap_or_else(|| cwd.join(".codingbuddy").join(SETUP_MARKER));

    if marker.exists() {
        return Ok(false);
    }

    // Only prompt in interactive terminals
    if !(std::io::stdin().is_terminal() && std::io::stdout().is_terminal()) {
        write_setup_marker(&marker)?;
        return Ok(false);
    }

    // If everything is already configured, just write the marker
    if has_api_key(cfg) && cfg.local_ml.enabled {
        write_setup_marker(&marker)?;
        return Ok(false);
    }

    println!();
    println!("Welcome to CodingBuddy! Let's get you set up.\n");

    run_wizard_steps(cwd, cfg)?;

    println!("Setup complete!\n");
    write_setup_marker(&marker)?;
    Ok(true)
}

fn write_setup_marker(marker: &Path) -> Result<()> {
    if let Some(parent) = marker.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(marker, "")?;
    Ok(())
}

/// Shared 4-step wizard logic used by both `maybe_first_time_setup` and `run_interactive_wizard`.
fn run_wizard_steps(cwd: &Path, cfg: &AppConfig) -> Result<()> {
    // Step 1: Provider selection
    println!("[1/4] Model Provider");
    println!("  1. DeepSeek API (default)");
    println!("  2. OpenAI-compatible (GLM-5, Qwen, Ollama, OpenRouter, custom endpoint)");
    println!();

    let provider_choice = prompt_choice("  Select [1-2]: ", 1, 2)?;

    if provider_choice == 2 {
        print!("  API base URL: ");
        std::io::stdout().flush()?;
        let mut base_url = String::new();
        std::io::stdin().read_line(&mut base_url)?;
        let base_url = base_url.trim().to_string();

        print!("  Model name: ");
        std::io::stdout().flush()?;
        let mut model_name = String::new();
        std::io::stdin().read_line(&mut model_name)?;
        let model_name = model_name.trim().to_string();

        print!("  API key env var (or empty for no auth): ");
        std::io::stdout().flush()?;
        let mut api_key_env = String::new();
        std::io::stdin().read_line(&mut api_key_env)?;
        let api_key_env = api_key_env.trim().to_string();

        merge_provider_config(
            cwd,
            "openai-compat",
            &base_url,
            &model_name,
            if api_key_env.is_empty() {
                None
            } else {
                Some(&api_key_env)
            },
        )?;
        println!("  Provider saved.\n");
    } else {
        println!("  Using DeepSeek API.\n");
    }

    // Step 2: API Key
    println!("[2/4] API Key");
    let env_var = if provider_choice == 2 {
        "OPENAI_API_KEY".to_string()
    } else {
        cfg.llm.active_provider().api_key_env.clone()
    };
    if has_api_key(cfg) {
        println!("  API key is set.\n");
    } else {
        println!(
            "  Set {} in your environment, or run `codingbuddy setup` to reconfigure.\n",
            env_var
        );
    }

    // Step 3: Local ML
    println!("[3/4] Local ML");
    println!("  Local ML runs models on your machine for:");
    println!("  - Code retrieval: surfaces relevant code before the LLM responds");
    println!("  - Privacy scanning: detects and redacts secrets before they reach the API");
    println!("  - Ghost text: inline code completions in the TUI\n");

    let enable_ml = prompt_yes_no("  Enable local ML? [Y/n]: ")?;
    println!();

    // Step 4: Privacy Scanning
    println!("[4/4] Privacy Scanning");
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
        download_required_models(cfg, false)?;
    }

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

/// `codingbuddy setup --status` — show current setup state without prompts.
fn run_status_display(cwd: &Path, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;
    let api_key_set = has_api_key(&cfg);
    let models = model_download_status(&cfg);
    let provider = cfg.llm.active_provider();

    if json_mode {
        print_json(&json!({
            "provider": cfg.llm.provider,
            "base_url": provider.base_url,
            "chat_model": provider.models.chat,
            "reasoner_model": provider.models.reasoner,
            "api_key": if api_key_set { "configured" } else { "missing" },
            "local_ml": {
                "enabled": cfg.local_ml.enabled,
                "privacy_enabled": cfg.local_ml.privacy.enabled,
                "autocomplete_enabled": cfg.local_ml.autocomplete.enabled,
                "models": models,
            },
        }))?;
    } else {
        println!("provider: {}", cfg.llm.provider);
        println!("base_url: {}", provider.base_url);
        println!("chat_model: {}", provider.models.chat);
        if let Some(ref reasoner) = provider.models.reasoner {
            println!("reasoner_model: {reasoner}");
        }
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

/// `codingbuddy setup --local-ml` — non-interactive shortcut.
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

/// Full interactive wizard (4 steps).
fn run_interactive_wizard(cwd: &Path, json_mode: bool) -> Result<()> {
    let cfg = AppConfig::ensure(cwd)?;
    let interactive = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();

    if !interactive || json_mode {
        return run_status_display(cwd, json_mode);
    }

    println!("Welcome to CodingBuddy setup!\n");

    run_wizard_steps(cwd, &cfg)?;

    println!("\nSetup complete! Run `codingbuddy chat` to start.");
    Ok(())
}

/// Download the default embedding and completion models, showing progress.
/// Returns a list of (model_id, outcome) pairs for JSON output.
fn download_required_models(cfg: &AppConfig, json_mode: bool) -> Result<Vec<(String, String)>> {
    let cache_dir = resolve_cache_dir(cfg);
    let mut manager = ModelManager::new(cache_dir);

    let embedding = model_registry::default_embedding_model();
    let completion = model_registry::default_completion_model();

    let models = [embedding, completion];

    let mut results = Vec::new();

    for entry in &models {
        let model_id = entry.model_id;
        let display_name = entry.display_name;
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

        let files = entry.download_files();
        match manager.ensure_model_with_progress(
            model_id,
            entry.hf_repo,
            &files,
            |current, total| {
                if !json_mode && total > 0 {
                    print!(
                        "\r  Downloading {} ({})... [{}/{}]",
                        display_name, model_id, current, total
                    );
                    let _ = std::io::stdout().flush();
                }
            },
        ) {
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

/// Prompt user to pick a number in [min, max]. Returns the chosen value.
fn prompt_choice(prompt: &str, min: u32, max: u32) -> Result<u32> {
    print!("{prompt}");
    std::io::stdout().flush()?;
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    let chosen: u32 = answer.trim().parse().unwrap_or(min);
    Ok(chosen.clamp(min, max))
}

/// Merge a custom provider into `.codingbuddy/settings.json`.
fn merge_provider_config(
    cwd: &Path,
    provider_name: &str,
    base_url: &str,
    model: &str,
    api_key_env: Option<&str>,
) -> Result<()> {
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

    let map = root.as_object_mut().expect("root is object");
    let llm = map.entry("llm".to_string()).or_insert_with(|| json!({}));
    if !llm.is_object() {
        *llm = json!({});
    }
    if let Some(llm_obj) = llm.as_object_mut() {
        llm_obj.insert("provider".to_string(), json!(provider_name));
        llm_obj.insert("base_url".to_string(), json!(base_url));
        llm_obj.insert("base_model".to_string(), json!(model));
        if let Some(env) = api_key_env {
            llm_obj.insert("api_key_env".to_string(), json!(env));
        }

        // Also save to providers map
        let providers = llm_obj
            .entry("providers".to_string())
            .or_insert_with(|| json!({}));
        if !providers.is_object() {
            *providers = json!({});
        }
        if let Some(p) = providers.as_object_mut() {
            p.insert(
                provider_name.to_string(),
                json!({
                    "base_url": base_url,
                    "api_key_env": api_key_env.unwrap_or(""),
                    "models": {
                        "chat": model,
                    }
                }),
            );
        }
    }

    fs::write(&path, serde_json::to_vec_pretty(&root)?)?;
    Ok(())
}

/// Merge local_ml keys into `.codingbuddy/settings.json` without clobbering other settings.
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

/// Project settings path (`.codingbuddy/settings.json`).
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
        let dir = cwd.join(".codingbuddy");
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
        assert_eq!(dir, PathBuf::from(".codingbuddy/models"));
    }

    #[test]
    fn first_time_setup_skips_when_fully_configured() {
        let tmp = TempDir::new().unwrap();
        let cwd = tmp.path();
        let dir = cwd.join(".codingbuddy");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("settings.json"),
            serde_json::to_vec_pretty(&json!({ "local_ml": { "enabled": true } })).unwrap(),
        )
        .unwrap();

        // Set API key so has_api_key returns true
        unsafe { std::env::set_var("DEEPSEEK_API_KEY", "test-key") };

        let cfg = AppConfig::ensure(cwd).unwrap();
        // Should write the marker without prompting (already configured + non-interactive)
        maybe_first_time_setup(cwd, &cfg).unwrap();

        // Marker should exist
        assert!(dir.join(SETUP_MARKER).exists());

        unsafe { std::env::remove_var("DEEPSEEK_API_KEY") };
    }

    #[test]
    fn first_time_setup_skips_when_marker_exists() {
        let tmp = TempDir::new().unwrap();
        let cwd = tmp.path();
        let dir = cwd.join(".codingbuddy");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("settings.json"), "{}").unwrap();
        fs::write(dir.join(SETUP_MARKER), "").unwrap();

        let cfg = AppConfig::ensure(cwd).unwrap();
        // Should return immediately — no prompt, no config change
        maybe_first_time_setup(cwd, &cfg).unwrap();
    }

    #[test]
    fn merge_provider_config_writes_provider_settings() {
        let tmp = TempDir::new().unwrap();
        let cwd = tmp.path();
        let dir = cwd.join(".codingbuddy");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("settings.json"), "{}").unwrap();

        merge_provider_config(
            cwd,
            "openai-compat",
            "http://localhost:11434/v1",
            "llama3",
            Some("OLLAMA_API_KEY"),
        )
        .unwrap();

        let path = dir.join("settings.json");
        let raw = fs::read_to_string(&path).unwrap();
        let root: serde_json::Value = serde_json::from_str(&raw).unwrap();

        assert_eq!(root["llm"]["provider"], "openai-compat");
        assert_eq!(root["llm"]["base_url"], "http://localhost:11434/v1");
        assert_eq!(root["llm"]["base_model"], "llama3");
        assert_eq!(root["llm"]["api_key_env"], "OLLAMA_API_KEY");
        assert_eq!(
            root["llm"]["providers"]["openai-compat"]["base_url"],
            "http://localhost:11434/v1"
        );
        assert_eq!(
            root["llm"]["providers"]["openai-compat"]["models"]["chat"],
            "llama3"
        );
    }

    #[test]
    fn merge_provider_config_preserves_existing_settings() {
        let tmp = TempDir::new().unwrap();
        let cwd = tmp.path();
        let dir = cwd.join(".codingbuddy");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("settings.json"),
            serde_json::to_vec_pretty(&json!({
                "local_ml": { "enabled": true },
                "custom": "preserved"
            }))
            .unwrap(),
        )
        .unwrap();

        merge_provider_config(cwd, "custom-llm", "http://my-llm:8000/v1", "my-model", None)
            .unwrap();

        let path = dir.join("settings.json");
        let raw = fs::read_to_string(&path).unwrap();
        let root: serde_json::Value = serde_json::from_str(&raw).unwrap();

        // Provider written
        assert_eq!(root["llm"]["provider"], "custom-llm");
        assert_eq!(root["llm"]["base_model"], "my-model");

        // Existing settings preserved
        assert_eq!(root["local_ml"]["enabled"], true);
        assert_eq!(root["custom"], "preserved");
    }
}
