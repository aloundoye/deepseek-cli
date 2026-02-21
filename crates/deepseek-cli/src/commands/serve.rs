use anyhow::{Result, anyhow};
use clap::CommandFactory;
use clap_complete::generate;
use std::io;

use crate::{Cli, CompletionsArgs, ServeArgs};

pub(crate) fn run_completions(args: CompletionsArgs) -> Result<()> {
    let mut cmd = Cli::command();
    generate(args.shell, &mut cmd, "deepseek", &mut io::stdout());
    Ok(())
}

pub(crate) fn run_serve(args: ServeArgs, json_mode: bool) -> Result<()> {
    match args.transport.as_str() {
        "stdio" => {
            if json_mode {
                println!(
                    "{}",
                    serde_json::json!({"status": "starting", "transport": "stdio"})
                );
            } else {
                eprintln!("deepseek: starting JSON-RPC server on stdio...");
            }
            let workspace = std::env::current_dir()?;
            let handler = deepseek_jsonrpc::IdeRpcHandler::new(&workspace)?;
            deepseek_jsonrpc::run_stdio_server(&handler)
        }
        other => Err(anyhow!(
            "unsupported transport '{}' (supported: stdio)",
            other
        )),
    }
}
