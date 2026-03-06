mod command_guard;
mod fuzzy_edit;
mod plugins;
mod sandbox;
mod shell;
pub mod tool_tiers;
pub mod validation;

pub use codingbuddy_core::ToolTier;
pub use tool_tiers::{
    ToolContextSignals, detect_signals, format_tool_search_results, search_extended_tools,
    tiered_tool_definitions, tool_search_definition, tool_tier,
};
pub use validation::{normalize_tool_args, normalize_tool_args_with_workspace, validate_tool_args};

use anyhow::{Result, anyhow};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use chrono::Utc;
use codingbuddy_chrome::{ChromeSession, ScreenshotFormat};
use codingbuddy_core::{
    AppConfig, ApprovedToolCall, EventEnvelope, EventKind, FunctionDefinition, ToolCall,
    ToolDefinition, ToolHost, ToolProposal, ToolResult,
};
use codingbuddy_diff::PatchStore;
use codingbuddy_hooks::{HookContext, HookRuntime};
use codingbuddy_index::IndexService;
use codingbuddy_memory::MemoryManager;
use codingbuddy_policy::PolicyEngine;
use codingbuddy_store::Store;
pub(crate) use command_guard::detect_container_environment;
use ignore::WalkBuilder;
pub use plugins::{
    CatalogPlugin, PluginCommandPrompt, PluginInfo, PluginManager, PluginVerifyResult,
    plugin_tool_definitions,
};
use plugins::{plugin_command_lookup_name, plugin_tool_api_name};
use serde_json::json;
use sha2::Digest;
pub use shell::{PlatformShellRunner, ShellRunResult, ShellRunner};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Duration;
use uuid::Uuid;

use fuzzy_edit::apply_single_edit;
#[cfg(test)]
use fuzzy_edit::{
    fuzzy_block_anchor, fuzzy_context_aware, fuzzy_escape_normalized, fuzzy_indentation_flexible,
    fuzzy_line_trimmed, fuzzy_trimmed_boundary, fuzzy_whitespace_normalized,
};
use sandbox::sandbox_wrap_command;
#[cfg(test)]
use sandbox::{build_bwrap_command, build_seatbelt_profile, seatbelt_wrap};

const DEFAULT_TIMEOUT_SECONDS: u64 = 120;
const READ_MAX_BYTES_DEFAULT: usize = 1_000_000;

mod catalog;
mod host;
mod utils;

pub use catalog::{
    AGENT_LEVEL_TOOLS, PLAN_MODE_TOOLS, filter_tool_definitions, map_tool_name, tool_definitions,
    tool_error_hint, validate_tool_args_schema,
};
pub use host::{LocalToolHost, McpExecutor};
#[cfg(test)]
pub(crate) use utils::{detect_diagnostics_command, parse_cargo_check_json, parse_tsc_output};

#[cfg(test)]
mod tests;
