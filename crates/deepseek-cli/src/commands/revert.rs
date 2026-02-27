use anyhow::{Result, anyhow};
use deepseek_core::EventKind;
use deepseek_store::Store;
use serde_json::json;
use std::path::Path;

use crate::RevertArgs;
use crate::context::append_control_event;
use crate::output::print_json;

pub(crate) fn run_revert(cwd: &Path, args: RevertArgs, json_mode: bool) -> Result<()> {
    let store = Store::new(cwd)?;
    let session = store
        .load_latest_session()?
        .ok_or_else(|| anyhow!("no session active to revert"))?;

    let projection = store.rebuild_from_events(session.session_id)?;
    if projection.chat_messages.is_empty() {
        if json_mode {
            print_json(&json!({"status": "no_op", "reason": "empty_chat"}))?;
        } else {
            println!("no conversation turns available to revert.");
        }
        return Ok(());
    }

    let turns = args.turns.max(1);

    append_control_event(
        cwd,
        EventKind::TurnReverted {
            turns_dropped: turns,
        },
    )?;

    if json_mode {
        print_json(&json!({
            "status": "success",
            "turns_dropped": turns,
            "session_id": session.session_id
        }))?;
    } else {
        println!(
            "Reverted {} turn(s) from session {}.",
            turns, session.session_id
        );
    }
    Ok(())
}
