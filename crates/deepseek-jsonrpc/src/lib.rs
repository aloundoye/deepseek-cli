use anyhow::{Result, anyhow};
use deepseek_core::{ChatMessage, EventKind, Session, SessionBudgets, SessionState};
use deepseek_store::Store;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Value,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

/// JSON-RPC 2.0 response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

// Standard JSON-RPC error codes.
pub const ERR_PARSE: i64 = -32700;
pub const ERR_METHOD_NOT_FOUND: i64 = -32601;
pub const ERR_INVALID_PARAMS: i64 = -32602;
pub const ERR_INTERNAL: i64 = -32603;

// Application-level error codes.
pub const ERR_SESSION_NOT_FOUND: i64 = -32000;
pub const ERR_TOOL_DENIED: i64 = -32001;
pub const ERR_PATCH_CONFLICT: i64 = -32002;

impl JsonRpcResponse {
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

/// Trait for handling JSON-RPC method dispatching.
pub trait RpcHandler {
    fn handle(&self, method: &str, params: Value) -> Result<Value>;
}

/// Run a JSON-RPC 2.0 server over stdio (newline-delimited JSON).
pub fn run_stdio_server(handler: &dyn RpcHandler) -> Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let reader = stdin.lock();
    let mut writer = stdout.lock();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<JsonRpcRequest>(trimmed) {
            Ok(req) => {
                if req.method == "shutdown" {
                    let resp = JsonRpcResponse::success(req.id, serde_json::json!({"ok": true}));
                    let out = serde_json::to_string(&resp)?;
                    writeln!(writer, "{out}")?;
                    writer.flush()?;
                    return Ok(());
                }
                match handler.handle(&req.method, req.params) {
                    Ok(result) => JsonRpcResponse::success(req.id, result),
                    Err(e) => JsonRpcResponse::error(req.id, ERR_INTERNAL, e.to_string()),
                }
            }
            Err(_) => JsonRpcResponse::error(Value::Null, ERR_PARSE, "Parse error"),
        };

        let out = serde_json::to_string(&response)?;
        writeln!(writer, "{out}")?;
        writer.flush()?;
    }

    Ok(())
}

/// A default handler that supports basic methods.
pub struct DefaultRpcHandler;

impl RpcHandler for DefaultRpcHandler {
    fn handle(&self, method: &str, _params: Value) -> Result<Value> {
        match method {
            "initialize" => Ok(serde_json::json!({
                "name": "deepseek-cli",
                "version": env!("CARGO_PKG_VERSION"),
                "capabilities": ["chat", "tool/execute", "status"]
            })),
            "status" => Ok(serde_json::json!({"status": "ready"})),
            "cancel" => Ok(serde_json::json!({"cancelled": true})),
            _ => Err(anyhow!("method not found: {method}")),
        }
    }
}

// ---------------------------------------------------------------------------
// IDE-oriented RPC handler with full session, prompt, tool, and patch methods
// ---------------------------------------------------------------------------

/// Tracks pending tool approvals for a session.
#[derive(Debug, Default)]
struct PendingToolApproval {
    /// Maps invocation_id → tool_name for tools awaiting approval.
    pending: HashMap<String, String>,
}

/// Full-featured IDE handler backed by a `Store`.
pub struct IdeRpcHandler {
    store: Arc<Store>,
    /// Per-session pending tool approvals.
    approvals: Mutex<HashMap<Uuid, PendingToolApproval>>,
    /// Per-session conversation histories (for prompt/execute streaming).
    conversations: Mutex<HashMap<Uuid, Vec<ChatMessage>>>,
}

impl IdeRpcHandler {
    pub fn new(workspace: &Path) -> Result<Self> {
        let store = Store::new(workspace)?;
        Ok(Self {
            store: Arc::new(store),
            approvals: Mutex::new(HashMap::new()),
            conversations: Mutex::new(HashMap::new()),
        })
    }

    pub fn from_store(store: Arc<Store>) -> Self {
        Self {
            store,
            approvals: Mutex::new(HashMap::new()),
            conversations: Mutex::new(HashMap::new()),
        }
    }

    // -- Session methods --

    fn handle_session_open(&self, params: Value) -> Result<Value> {
        let workspace = params
            .get("workspace_root")
            .and_then(|v| v.as_str())
            .unwrap_or(".");
        let session_id = Uuid::now_v7();
        let session = Session {
            session_id,
            workspace_root: workspace.to_string(),
            baseline_commit: params
                .get("baseline_commit")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            status: SessionState::Idle,
            budgets: SessionBudgets {
                per_turn_seconds: 300,
                max_think_tokens: 8192,
            },
            active_plan_id: None,
        };
        self.store.save_session(&session)?;
        Ok(serde_json::json!({
            "session_id": session_id.to_string(),
            "status": "idle",
            "workspace_root": workspace,
        }))
    }

    fn handle_session_resume(&self, params: Value) -> Result<Value> {
        let session_id = require_uuid(&params, "session_id")?;
        let session = self
            .store
            .load_session(session_id)?
            .ok_or_else(|| anyhow!("session not found: {session_id}"))?;

        // Rebuild conversation from event log.
        let projection = self.store.rebuild_from_events(session_id)?;
        let turn_count = projection.chat_messages.len();

        // Cache the conversation history.
        self.conversations
            .lock()
            .expect("conversations lock")
            .insert(session_id, projection.chat_messages);

        Ok(serde_json::json!({
            "session_id": session_id.to_string(),
            "status": format!("{:?}", session.status),
            "turn_count": turn_count,
            "workspace_root": session.workspace_root,
        }))
    }

    fn handle_session_fork(&self, params: Value) -> Result<Value> {
        let from_id = require_uuid(&params, "session_id")?;
        let forked = self.store.fork_session(from_id)?;

        // Copy conversation history to the new session.
        let conv = self.conversations.lock().expect("conversations lock");
        if let Some(messages) = conv.get(&from_id) {
            let cloned = messages.clone();
            drop(conv);
            self.conversations
                .lock()
                .expect("conversations lock")
                .insert(forked.session_id, cloned);
        }

        Ok(serde_json::json!({
            "session_id": forked.session_id.to_string(),
            "forked_from": from_id.to_string(),
            "status": "idle",
        }))
    }

    fn handle_session_list(&self, _params: Value) -> Result<Value> {
        // Query all sessions from the database ordered by updated_at desc.
        let conn = self.store.db()?;
        let mut stmt = conn.prepare(
            "SELECT session_id, workspace_root, status, updated_at
             FROM sessions ORDER BY updated_at DESC LIMIT 50",
        )?;
        let mut sessions = Vec::new();
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            sessions.push(serde_json::json!({
                "session_id": row.get::<_, String>(0)?,
                "workspace_root": row.get::<_, String>(1)?,
                "status": row.get::<_, String>(2)?,
                "updated_at": row.get::<_, String>(3)?,
            }));
        }
        Ok(serde_json::json!({ "sessions": sessions }))
    }

    // -- Prompt methods --

    fn handle_prompt_execute(&self, params: Value) -> Result<Value> {
        let session_id = require_uuid(&params, "session_id")?;
        let prompt = params
            .get("prompt")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing 'prompt' parameter"))?
            .to_string();

        // Verify session exists.
        let _session = self
            .store
            .load_session(session_id)?
            .ok_or_else(|| anyhow!("session not found: {session_id}"))?;

        // Store the user message in conversation cache.
        self.conversations
            .lock()
            .expect("conversations lock")
            .entry(session_id)
            .or_default()
            .push(ChatMessage::User {
                content: prompt.clone(),
            });

        // Record the prompt as a ChatTurnV1 event.
        let seq = self.store.next_seq_no(session_id)?;
        self.store.append_event(&deepseek_core::EventEnvelope {
            seq_no: seq,
            at: chrono::Utc::now(),
            session_id,
            kind: EventKind::ChatTurnV1 {
                message: ChatMessage::User {
                    content: prompt.clone(),
                },
            },
        })?;

        // Return a prompt_id for the IDE to track this execution.
        // Actual LLM execution is handled by the agent runtime which the IDE
        // polls via `prompt/status` or receives tool events.
        let prompt_id = Uuid::now_v7();
        Ok(serde_json::json!({
            "prompt_id": prompt_id.to_string(),
            "session_id": session_id.to_string(),
            "status": "queued",
            "prompt": prompt,
        }))
    }

    // -- Tool approval methods --

    fn handle_tool_approve(&self, params: Value) -> Result<Value> {
        let session_id = require_uuid(&params, "session_id")?;
        let invocation_id_str = params
            .get("invocation_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing 'invocation_id' parameter"))?;
        let invocation_id = Uuid::parse_str(invocation_id_str)?;

        // Remove from pending approvals.
        let mut approvals = self.approvals.lock().expect("approvals lock");
        let tool_name = approvals
            .get_mut(&session_id)
            .and_then(|pa| pa.pending.remove(invocation_id_str))
            .unwrap_or_default();

        // Record approval event.
        let seq = self.store.next_seq_no(session_id)?;
        self.store.append_event(&deepseek_core::EventEnvelope {
            seq_no: seq,
            at: chrono::Utc::now(),
            session_id,
            kind: EventKind::ToolApprovedV1 { invocation_id },
        })?;

        Ok(serde_json::json!({
            "approved": true,
            "invocation_id": invocation_id_str,
            "tool_name": tool_name,
        }))
    }

    fn handle_tool_deny(&self, params: Value) -> Result<Value> {
        let session_id = require_uuid(&params, "session_id")?;
        let invocation_id_str = params
            .get("invocation_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing 'invocation_id' parameter"))?;

        // Remove from pending approvals.
        let mut approvals = self.approvals.lock().expect("approvals lock");
        let tool_name = approvals
            .get_mut(&session_id)
            .and_then(|pa| pa.pending.remove(invocation_id_str))
            .unwrap_or_default();

        Ok(serde_json::json!({
            "denied": true,
            "invocation_id": invocation_id_str,
            "tool_name": tool_name,
        }))
    }

    // -- Patch methods --

    fn handle_patch_preview(&self, params: Value) -> Result<Value> {
        let session_id = require_uuid(&params, "session_id")?;
        let patch_id = require_uuid(&params, "patch_id")?;

        // Verify the patch is known in this session.
        let projection = self.store.rebuild_from_events(session_id)?;
        if !projection.staged_patches.contains(&patch_id) {
            return Err(anyhow!("patch not found: {patch_id}"));
        }

        let applied = projection.applied_patches.contains(&patch_id);
        Ok(serde_json::json!({
            "patch_id": patch_id.to_string(),
            "staged": true,
            "applied": applied,
        }))
    }

    fn handle_patch_apply(&self, params: Value) -> Result<Value> {
        let session_id = require_uuid(&params, "session_id")?;
        let patch_id = require_uuid(&params, "patch_id")?;

        // Record patch applied event.
        let seq = self.store.next_seq_no(session_id)?;
        self.store.append_event(&deepseek_core::EventEnvelope {
            seq_no: seq,
            at: chrono::Utc::now(),
            session_id,
            kind: EventKind::PatchAppliedV1 {
                patch_id,
                applied: true,
                conflicts: vec![],
            },
        })?;

        Ok(serde_json::json!({
            "patch_id": patch_id.to_string(),
            "applied": true,
            "conflicts": [],
        }))
    }

    // -- Diagnostics methods --

    fn handle_diagnostics_list(&self, params: Value) -> Result<Value> {
        let session_id = require_uuid(&params, "session_id")?;

        // Rebuild events and extract diagnostic-related entries from transcript.
        let projection = self.store.rebuild_from_events(session_id)?;
        let diagnostics: Vec<Value> = projection
            .transcript
            .iter()
            .filter(|line| {
                let lower = line.to_lowercase();
                lower.contains("diagnostic")
                    || lower.contains("error")
                    || lower.contains("warning")
                    || lower.contains("lint")
            })
            .map(|line| serde_json::json!({"message": line}))
            .collect();

        Ok(serde_json::json!({
            "session_id": session_id.to_string(),
            "diagnostics": diagnostics,
            "tool_invocations": projection.tool_invocations.len(),
        }))
    }

    // -- Task methods --

    fn handle_task_list(&self, params: Value) -> Result<Value> {
        let session_id = params
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(Uuid::parse_str)
            .transpose()?;
        let tasks = self.store.list_tasks(session_id)?;
        let task_list: Vec<Value> = tasks
            .iter()
            .map(|t| {
                serde_json::json!({
                    "task_id": t.task_id.to_string(),
                    "title": t.title,
                    "priority": t.priority,
                    "status": t.status,
                    "outcome": t.outcome,
                })
            })
            .collect();
        Ok(serde_json::json!({ "tasks": task_list }))
    }

    fn handle_task_update(&self, params: Value) -> Result<Value> {
        let task_id = require_uuid(&params, "task_id")?;
        let status = params
            .get("status")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing 'status' parameter"))?;
        let outcome = params.get("outcome").and_then(|v| v.as_str());

        self.store.update_task_status(task_id, status, outcome)?;

        Ok(serde_json::json!({
            "task_id": task_id.to_string(),
            "status": status,
            "updated": true,
        }))
    }

    /// Register a pending tool approval that the IDE can approve/deny.
    pub fn add_pending_approval(&self, session_id: Uuid, invocation_id: &str, tool_name: &str) {
        self.approvals
            .lock()
            .expect("approvals lock")
            .entry(session_id)
            .or_default()
            .pending
            .insert(invocation_id.to_string(), tool_name.to_string());
    }
}

impl RpcHandler for IdeRpcHandler {
    fn handle(&self, method: &str, params: Value) -> Result<Value> {
        match method {
            // Basic lifecycle
            "initialize" => Ok(serde_json::json!({
                "name": "deepseek-cli",
                "version": env!("CARGO_PKG_VERSION"),
                "capabilities": [
                    "session/open", "session/resume", "session/fork", "session/list",
                    "prompt/execute",
                    "tool/approve", "tool/deny",
                    "patch/preview", "patch/apply",
                    "diagnostics/list",
                    "task/list", "task/update",
                    "status", "cancel", "shutdown"
                ]
            })),
            "status" => Ok(serde_json::json!({"status": "ready", "handler": "ide"})),
            "cancel" => Ok(serde_json::json!({"cancelled": true})),

            // Session management
            "session/open" => self.handle_session_open(params),
            "session/resume" => self.handle_session_resume(params),
            "session/fork" => self.handle_session_fork(params),
            "session/list" => self.handle_session_list(params),

            // Prompt execution
            "prompt/execute" => self.handle_prompt_execute(params),

            // Tool approval/denial
            "tool/approve" => self.handle_tool_approve(params),
            "tool/deny" => self.handle_tool_deny(params),

            // Patch preview/apply
            "patch/preview" => self.handle_patch_preview(params),
            "patch/apply" => self.handle_patch_apply(params),

            // Diagnostics
            "diagnostics/list" => self.handle_diagnostics_list(params),

            // Task management
            "task/list" => self.handle_task_list(params),
            "task/update" => self.handle_task_update(params),

            _ => Err(anyhow!("method not found: {method}")),
        }
    }
}

/// Extract a UUID from a JSON params object.
fn require_uuid(params: &Value, field: &str) -> Result<Uuid> {
    let s = params
        .get(field)
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing '{field}' parameter"))?;
    Uuid::parse_str(s).map_err(|e| anyhow!("invalid UUID for '{field}': {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn jsonrpc_request_round_trip() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: "initialize".to_string(),
            params: json!({}),
        };
        let serialized = serde_json::to_string(&req).expect("serialize");
        let deserialized: JsonRpcRequest = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(deserialized.method, "initialize");
        assert_eq!(deserialized.id, json!(1));
    }

    #[test]
    fn parse_error_returns_code_32700() {
        let bad_json = "not json";
        let result = serde_json::from_str::<JsonRpcRequest>(bad_json);
        assert!(result.is_err());
        let resp = JsonRpcResponse::error(Value::Null, ERR_PARSE, "Parse error");
        assert_eq!(resp.error.as_ref().unwrap().code, -32700);
    }

    #[test]
    fn success_response_has_no_error() {
        let resp = JsonRpcResponse::success(json!(42), json!({"ok": true}));
        assert!(resp.error.is_none());
        assert_eq!(resp.result.unwrap()["ok"], true);
    }

    #[test]
    fn error_response_has_no_result() {
        let resp = JsonRpcResponse::error(json!(1), ERR_METHOD_NOT_FOUND, "method not found");
        assert!(resp.result.is_none());
        assert_eq!(resp.error.as_ref().unwrap().code, -32601);
    }

    #[test]
    fn default_handler_initialize() {
        let handler = DefaultRpcHandler;
        let result = handler.handle("initialize", json!({})).expect("initialize");
        assert_eq!(result["name"], "deepseek-cli");
    }

    #[test]
    fn default_handler_unknown_method() {
        let handler = DefaultRpcHandler;
        let err = handler
            .handle("nonexistent", json!({}))
            .expect_err("should fail");
        assert!(err.to_string().contains("method not found"));
    }

    #[test]
    fn ide_handler_initialize() {
        let tmp = tempfile::tempdir().unwrap();
        let handler = IdeRpcHandler::new(tmp.path()).unwrap();
        let result = handler.handle("initialize", json!({})).expect("initialize");
        assert_eq!(result["name"], "deepseek-cli");
        let caps = result["capabilities"].as_array().unwrap();
        assert!(caps.iter().any(|c| c == "session/open"));
        assert!(caps.iter().any(|c| c == "prompt/execute"));
        assert!(caps.iter().any(|c| c == "tool/approve"));
        assert!(caps.iter().any(|c| c == "patch/preview"));
    }

    #[test]
    fn ide_handler_session_open_and_list() {
        let tmp = tempfile::tempdir().unwrap();
        let handler = IdeRpcHandler::new(tmp.path()).unwrap();

        // Open a session.
        let result = handler
            .handle("session/open", json!({"workspace_root": "/tmp/project"}))
            .expect("session/open");
        let sid = result["session_id"].as_str().unwrap();
        assert!(!sid.is_empty());
        assert_eq!(result["status"], "idle");

        // List sessions.
        let list = handler
            .handle("session/list", json!({}))
            .expect("session/list");
        let sessions = list["sessions"].as_array().unwrap();
        assert!(!sessions.is_empty());
        assert!(sessions.iter().any(|s| s["session_id"] == sid));
    }

    #[test]
    fn ide_handler_session_fork() {
        let tmp = tempfile::tempdir().unwrap();
        let handler = IdeRpcHandler::new(tmp.path()).unwrap();

        // Open a session.
        let result = handler
            .handle("session/open", json!({"workspace_root": "/tmp/test"}))
            .unwrap();
        let sid = result["session_id"].as_str().unwrap();

        // Fork it.
        let forked = handler
            .handle("session/fork", json!({"session_id": sid}))
            .expect("session/fork");
        assert_eq!(forked["forked_from"], sid);
        assert_ne!(forked["session_id"], sid);
        assert_eq!(forked["status"], "idle");
    }

    #[test]
    fn ide_handler_session_resume() {
        let tmp = tempfile::tempdir().unwrap();
        let handler = IdeRpcHandler::new(tmp.path()).unwrap();

        let result = handler
            .handle("session/open", json!({"workspace_root": "/tmp/r"}))
            .unwrap();
        let sid = result["session_id"].as_str().unwrap();

        let resumed = handler
            .handle("session/resume", json!({"session_id": sid}))
            .expect("session/resume");
        assert_eq!(resumed["session_id"], sid);
        assert_eq!(resumed["turn_count"], 0);
    }

    #[test]
    fn ide_handler_prompt_execute() {
        let tmp = tempfile::tempdir().unwrap();
        let handler = IdeRpcHandler::new(tmp.path()).unwrap();

        let result = handler
            .handle("session/open", json!({"workspace_root": "/tmp/p"}))
            .unwrap();
        let sid = result["session_id"].as_str().unwrap();

        let exec = handler
            .handle(
                "prompt/execute",
                json!({"session_id": sid, "prompt": "explain this code"}),
            )
            .expect("prompt/execute");
        assert_eq!(exec["status"], "queued");
        assert_eq!(exec["prompt"], "explain this code");
        assert!(exec["prompt_id"].as_str().is_some());
    }

    #[test]
    fn ide_handler_tool_approve_deny() {
        let tmp = tempfile::tempdir().unwrap();
        let handler = IdeRpcHandler::new(tmp.path()).unwrap();

        let result = handler
            .handle("session/open", json!({"workspace_root": "/tmp/t"}))
            .unwrap();
        let sid_str = result["session_id"].as_str().unwrap();
        let sid = Uuid::parse_str(sid_str).unwrap();

        let inv_id = Uuid::now_v7();
        handler.add_pending_approval(sid, &inv_id.to_string(), "bash.run");

        // Approve the tool.
        let approved = handler
            .handle(
                "tool/approve",
                json!({
                    "session_id": sid_str,
                    "invocation_id": inv_id.to_string(),
                }),
            )
            .expect("tool/approve");
        assert_eq!(approved["approved"], true);
        assert_eq!(approved["tool_name"], "bash.run");

        // Deny a different one.
        let inv2 = Uuid::now_v7();
        handler.add_pending_approval(sid, &inv2.to_string(), "fs.write");
        let denied = handler
            .handle(
                "tool/deny",
                json!({
                    "session_id": sid_str,
                    "invocation_id": inv2.to_string(),
                }),
            )
            .expect("tool/deny");
        assert_eq!(denied["denied"], true);
        assert_eq!(denied["tool_name"], "fs.write");
    }

    #[test]
    fn ide_handler_task_list_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let handler = IdeRpcHandler::new(tmp.path()).unwrap();

        let result = handler.handle("task/list", json!({})).expect("task/list");
        let tasks = result["tasks"].as_array().unwrap();
        assert!(tasks.is_empty());
    }

    #[test]
    fn ide_handler_unknown_method() {
        let tmp = tempfile::tempdir().unwrap();
        let handler = IdeRpcHandler::new(tmp.path()).unwrap();
        let err = handler
            .handle("nonexistent", json!({}))
            .expect_err("should fail");
        assert!(err.to_string().contains("method not found"));
    }

    #[test]
    fn ide_handler_session_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let handler = IdeRpcHandler::new(tmp.path()).unwrap();
        let fake = Uuid::now_v7();
        let err = handler
            .handle("session/resume", json!({"session_id": fake.to_string()}))
            .expect_err("should fail");
        assert!(err.to_string().contains("session not found"));
    }

    #[test]
    fn require_uuid_helper() {
        let params = json!({"id": "not-a-uuid"});
        assert!(require_uuid(&params, "id").is_err());

        let valid = Uuid::now_v7();
        let params = json!({"id": valid.to_string()});
        assert_eq!(require_uuid(&params, "id").unwrap(), valid);

        let empty = json!({});
        assert!(require_uuid(&empty, "id").is_err());
    }
}
