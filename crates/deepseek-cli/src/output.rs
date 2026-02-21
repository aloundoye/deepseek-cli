use anyhow::Result;
use deepseek_core::AppConfig;
use serde::Serialize;
use serde_json::json;

pub(crate) fn print_json<T: Serialize>(value: &T) -> Result<()> {
    println!("{}", serde_json::to_string(value)?);
    Ok(())
}

pub(crate) fn redact_config_for_display(cfg: &AppConfig) -> Result<serde_json::Value> {
    let mut value = serde_json::to_value(cfg)?;
    if let Some(llm) = value.get_mut("llm").and_then(|entry| entry.as_object_mut())
        && llm.contains_key("api_key")
    {
        llm.insert("api_key".to_string(), json!("***REDACTED***"));
    }
    Ok(value)
}
