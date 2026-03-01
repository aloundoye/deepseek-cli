use anyhow::Result;
use codingbuddy_core::{AppConfig, ApprovedToolCall, ToolCall, ToolHost};
use codingbuddy_policy::PolicyEngine;
use codingbuddy_store::Store;
use codingbuddy_tools::LocalToolHost;
use serde_json::json;
use std::path::Path;

use crate::SearchArgs;
use crate::output::*;

pub(crate) fn run_search(cwd: &Path, args: SearchArgs, json_mode: bool) -> Result<()> {
    let _store = Store::new(cwd)?;
    let config = AppConfig::load(cwd).unwrap_or_default();
    let policy = PolicyEngine::from_app_config(&config.policy);
    let tool_host = LocalToolHost::new(cwd, policy)?;

    let call = ToolCall {
        name: "web.search".to_string(),
        args: json!({"query": args.query, "max_results": args.max_results}),
        requires_approval: false,
    };
    let proposal = tool_host.propose(call);
    let result = tool_host.execute(ApprovedToolCall {
        invocation_id: proposal.invocation_id,
        call: proposal.call,
    });

    if json_mode {
        print_json(&result.output)?;
    } else if let Some(results) = result
        .output
        .get("results")
        .and_then(|v: &serde_json::Value| v.as_array())
    {
        for (i, r) in results.iter().enumerate() {
            let title = r
                .get("title")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or("?");
            let url = r
                .get("url")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or("");
            let snippet = r
                .get("snippet")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or("");
            println!("{}. {}", i + 1, title);
            println!("   {url}");
            if !snippet.is_empty() {
                println!("   {snippet}");
            }
            println!();
        }
        let cached = result
            .output
            .get("cached")
            .and_then(|v: &serde_json::Value| v.as_bool())
            .unwrap_or(false);
        if cached {
            println!("(results from cache)");
        }
    } else {
        println!("No results found.");
    }
    Ok(())
}
