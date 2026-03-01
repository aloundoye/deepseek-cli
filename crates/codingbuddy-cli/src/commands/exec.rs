use anyhow::Result;
use codingbuddy_core::{AppConfig, ApprovedToolCall, ToolCall, ToolHost};
use codingbuddy_policy::PolicyEngine;
use codingbuddy_store::Store;
use codingbuddy_tools::LocalToolHost;
use serde_json::json;
use std::path::Path;

use crate::ExecArgs;
use crate::output::*;

pub(crate) fn run_exec(cwd: &Path, args: ExecArgs, json_mode: bool) -> Result<()> {
    let config = AppConfig::load(cwd).unwrap_or_default();
    let policy = PolicyEngine::from_app_config(&config.policy);

    // Check command against policy
    policy
        .check_command(&args.command)
        .map_err(|e| anyhow::anyhow!("policy denied command: {e}"))?;

    let _store = Store::new(cwd)?;
    let tool_host = LocalToolHost::new(cwd, policy)?;

    let call = ToolCall {
        name: "bash.run".to_string(),
        args: json!({"cmd": args.command, "timeout": args.timeout}),
        requires_approval: false,
    };
    let proposal = tool_host.propose(call);
    let result = tool_host.execute(ApprovedToolCall {
        invocation_id: proposal.invocation_id,
        call: proposal.call,
    });

    if json_mode {
        print_json(&json!({
            "command": args.command,
            "success": result.success,
            "output": result.output,
        }))?;
    } else {
        if let Some(stdout) = result
            .output
            .get("stdout")
            .and_then(|v: &serde_json::Value| v.as_str())
        {
            print!("{stdout}");
        }
        if let Some(stderr) = result
            .output
            .get("stderr")
            .and_then(|v: &serde_json::Value| v.as_str())
            && !stderr.is_empty()
        {
            eprint!("{stderr}");
        }
    }
    Ok(())
}
