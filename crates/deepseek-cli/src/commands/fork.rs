use anyhow::{Result, anyhow};
use deepseek_store::Store;
use serde_json::json;
use std::path::Path;
use uuid::Uuid;

use crate::ForkArgs;
use crate::output::*;

pub(crate) fn run_fork(cwd: &Path, args: ForkArgs, json_mode: bool) -> Result<()> {
    let store = Store::new(cwd)?;
    let session_id = Uuid::parse_str(&args.session_id)?;

    // Try to acquire lock on source session to prevent concurrent forks
    let holder = format!("fork-{}", std::process::id());
    if !store.try_acquire_session_lock(session_id, &holder)? {
        return Err(anyhow!(
            "session {} is locked by another process",
            args.session_id
        ));
    }

    let forked = store.fork_session(session_id)?;
    store.release_session_lock(session_id, &holder)?;

    if json_mode {
        print_json(&json!({
            "forked_from": args.session_id,
            "new_session_id": forked.session_id.to_string(),
            "status": "idle",
        }))?;
    } else {
        println!(
            "Forked session {} \u{2192} {}",
            args.session_id, forked.session_id
        );
        println!(
            "New session is ready. Use --resume {} to continue.",
            forked.session_id
        );
    }
    Ok(())
}
