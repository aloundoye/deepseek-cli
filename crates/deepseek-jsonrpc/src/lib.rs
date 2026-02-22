use anyhow::{Result, anyhow};
use deepseek_core::{ChatMessage, EventKind, Session, SessionBudgets, SessionState};
use deepseek_store::Store;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

// Enhanced context management
use deepseek_context::ContextManager;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HandoffDescriptor {
    schema: String,
    handoff_id: Uuid,
    bundle_path: String,
    token_hash: String,
    created_at: String,
    expires_at: String,
    used_at: Option<String>,
    session_id: Uuid,
}

/// Full-featured IDE handler backed by a `Store`.
pub struct IdeRpcHandler {
    store: Arc<Store>,
    /// Per-session pending tool approvals.
    approvals: Mutex<HashMap<Uuid, PendingToolApproval>>,
    /// Per-session conversation histories (for prompt/execute streaming).
    conversations: Mutex<HashMap<Uuid, Vec<ChatMessage>>>,
    /// Per-prompt queued partial message chunks for stream polling.
    prompt_streams: Mutex<HashMap<Uuid, Vec<String>>>,
    /// Whether partial-message streaming should be enabled by default.
    include_partial_messages: bool,
    /// Context manager for intelligent file suggestions.
    context_manager: Mutex<Option<ContextManager>>,
}

impl IdeRpcHandler {
    pub fn new(workspace: &Path) -> Result<Self> {
        let store = Store::new(workspace)?;

        // Initialize context manager (may fail if workspace is empty)
        let context_manager = match ContextManager::new(workspace) {
            Ok(mut manager) => {
                // Try to analyze workspace, but don't fail if it doesn't work
                let _ = manager.analyze_workspace();
                Some(manager)
            }
            Err(_) => None,
        };

        Ok(Self {
            store: Arc::new(store),
            approvals: Mutex::new(HashMap::new()),
            conversations: Mutex::new(HashMap::new()),
            prompt_streams: Mutex::new(HashMap::new()),
            include_partial_messages: env_flag_enabled("DEEPSEEK_INCLUDE_PARTIAL_MESSAGES"),
            context_manager: Mutex::new(context_manager),
        })
    }

    pub fn from_store(store: Arc<Store>) -> Self {
        Self {
            store,
            approvals: Mutex::new(HashMap::new()),
            conversations: Mutex::new(HashMap::new()),
            prompt_streams: Mutex::new(HashMap::new()),
            include_partial_messages: env_flag_enabled("DEEPSEEK_INCLUDE_PARTIAL_MESSAGES"),
            context_manager: Mutex::new(None),
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
        let include_partial = params
            .get("include_partial_messages")
            .and_then(|v| v.as_bool())
            .unwrap_or(self.include_partial_messages);
        if include_partial {
            let seed = format!("Processing prompt: {prompt}");
            let chunks = split_partial_chunks(&seed, 48);
            self.prompt_streams
                .lock()
                .expect("prompt_streams lock")
                .insert(prompt_id, chunks);
        }
        Ok(serde_json::json!({
            "prompt_id": prompt_id.to_string(),
            "session_id": session_id.to_string(),
            "status": "queued",
            "prompt": prompt,
            "partial_messages_enabled": include_partial,
            "stream": if include_partial {
                serde_json::json!({
                    "method": "prompt/stream_next",
                    "prompt_id": prompt_id.to_string(),
                    "cursor": 0,
                })
            } else {
                Value::Null
            },
        }))
    }

    fn handle_prompt_stream_next(&self, params: Value) -> Result<Value> {
        let prompt_id = require_uuid(&params, "prompt_id")?;
        let cursor = params.get("cursor").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let max_chunks = params
            .get("max_chunks")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;
        let max_chunks = max_chunks.clamp(1, 128);

        let mut streams = self.prompt_streams.lock().expect("prompt_streams lock");
        let Some(chunks) = streams.get(&prompt_id) else {
            return Ok(serde_json::json!({
                "prompt_id": prompt_id.to_string(),
                "chunks": [],
                "next_cursor": cursor,
                "done": true,
                "missing": true,
            }));
        };

        let start = cursor.min(chunks.len());
        let end = (start + max_chunks).min(chunks.len());
        let slice = &chunks[start..end];
        let out_chunks = slice
            .iter()
            .enumerate()
            .map(|(idx, delta)| {
                serde_json::json!({
                    "cursor": start + idx,
                    "type": "assistant_partial",
                    "delta": delta,
                })
            })
            .collect::<Vec<_>>();
        let done = end >= chunks.len();
        if done {
            streams.remove(&prompt_id);
        }

        Ok(serde_json::json!({
            "prompt_id": prompt_id.to_string(),
            "chunks": out_chunks,
            "next_cursor": end,
            "done": done,
        }))
    }

    fn handle_session_handoff_export(&self, params: Value) -> Result<Value> {
        let session_id = if let Some(session_id) = optional_uuid(&params, "session_id")? {
            session_id
        } else {
            self.store
                .load_latest_session()?
                .map(|s| s.session_id)
                .ok_or_else(|| anyhow!("no sessions available to export"))?
        };
        let session = self
            .store
            .load_session(session_id)?
            .ok_or_else(|| anyhow!("session not found: {session_id}"))?;
        let projection = self.store.rebuild_from_events(session_id)?;
        let bundle_id = Uuid::now_v7();
        let output_path = params
            .get("output_path")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                self.store
                    .root
                    .join("teleport")
                    .join(format!("{bundle_id}.json"))
            });
        if let Some(parent) = output_path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let payload = serde_json::json!({
            "schema": "deepseek.handoff.v1",
            "bundle_id": bundle_id.to_string(),
            "exported_at": chrono::Utc::now().to_rfc3339(),
            "session": {
                "session_id": session_id.to_string(),
                "workspace_root": session.workspace_root,
                "status": format!("{:?}", session.status),
            },
            "chat_messages": projection.chat_messages,
            "transcript": projection.transcript,
            "step_status": projection.step_status,
            "router_models": projection.router_models,
        });
        fs::write(&output_path, serde_json::to_vec_pretty(&payload)?)?;
        let seq = self.store.next_seq_no(session_id)?;
        self.store.append_event(&deepseek_core::EventEnvelope {
            seq_no: seq,
            at: chrono::Utc::now(),
            session_id,
            kind: EventKind::TeleportBundleCreatedV1 {
                bundle_id,
                path: output_path.to_string_lossy().to_string(),
            },
        })?;

        Ok(serde_json::json!({
            "bundle_id": bundle_id.to_string(),
            "session_id": session_id.to_string(),
            "path": output_path.to_string_lossy().to_string(),
            "turn_count": payload["chat_messages"].as_array().map_or(0, std::vec::Vec::len),
        }))
    }

    fn handle_session_handoff_import(&self, params: Value) -> Result<Value> {
        let bundle_path = params
            .get("bundle_path")
            .or_else(|| params.get("handoff_path"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing 'bundle_path' parameter"))?;
        let raw = fs::read_to_string(bundle_path)?;
        let payload: Value = serde_json::from_str(&raw)?;
        if payload
            .get("schema")
            .and_then(|v| v.as_str())
            .map(|value| value != "deepseek.handoff.v1")
            .unwrap_or(false)
        {
            return Err(anyhow!("unsupported handoff schema"));
        }

        let workspace_root = params
            .get("workspace_root")
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
            .or_else(|| {
                payload
                    .get("session")
                    .and_then(|s| s.get("workspace_root"))
                    .and_then(|v| v.as_str())
                    .map(ToString::to_string)
            })
            .unwrap_or_else(|| ".".to_string());
        let session_id = Uuid::now_v7();
        let session = Session {
            session_id,
            workspace_root,
            baseline_commit: None,
            status: SessionState::Idle,
            budgets: SessionBudgets {
                per_turn_seconds: 300,
                max_think_tokens: 8192,
            },
            active_plan_id: None,
        };
        self.store.save_session(&session)?;

        let chat_messages = payload
            .get("chat_messages")
            .cloned()
            .and_then(|v| serde_json::from_value::<Vec<ChatMessage>>(v).ok())
            .unwrap_or_default();
        for message in &chat_messages {
            let seq = self.store.next_seq_no(session_id)?;
            self.store.append_event(&deepseek_core::EventEnvelope {
                seq_no: seq,
                at: chrono::Utc::now(),
                session_id,
                kind: EventKind::ChatTurnV1 {
                    message: message.clone(),
                },
            })?;
        }
        let resume_import = params
            .get("resume")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        if resume_import {
            let seq = self.store.next_seq_no(session_id)?;
            self.store.append_event(&deepseek_core::EventEnvelope {
                seq_no: seq,
                at: chrono::Utc::now(),
                session_id,
                kind: EventKind::SessionResumedV1 {
                    session_id,
                    events_replayed: chat_messages.len() as u64,
                },
            })?;
        }
        self.conversations
            .lock()
            .expect("conversations lock")
            .insert(session_id, chat_messages.clone());

        Ok(serde_json::json!({
            "session_id": session_id.to_string(),
            "imported": true,
            "bundle_path": bundle_path,
            "turn_count": chat_messages.len(),
            "status": "idle",
            "resumed": resume_import,
        }))
    }

    fn handle_session_handoff_link_create(&self, params: Value) -> Result<Value> {
        let exported = self.handle_session_handoff_export(params.clone())?;
        let bundle_path = exported
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("handoff export did not return bundle path"))?;
        let session_id = exported
            .get("session_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("handoff export did not return session id"))?;
        let session_id = Uuid::parse_str(session_id)?;

        let handoff_id = Uuid::now_v7();
        let token = format!("{}{}", Uuid::now_v7().simple(), Uuid::now_v7().simple());
        let token_hash = sha256_hex(&token);
        let created_at = chrono::Utc::now();
        let ttl_minutes = params
            .get("ttl_minutes")
            .and_then(|v| v.as_u64())
            .unwrap_or(30)
            .clamp(1, 24 * 60);
        let expires_at = (created_at + chrono::Duration::minutes(ttl_minutes as i64)).to_rfc3339();

        let descriptor = HandoffDescriptor {
            schema: "deepseek.handoff_link.v1".to_string(),
            handoff_id,
            bundle_path: bundle_path.to_string(),
            token_hash,
            created_at: created_at.to_rfc3339(),
            expires_at: expires_at.clone(),
            used_at: None,
            session_id,
        };
        write_handoff_descriptor(self.store.root.as_path(), &descriptor)?;

        let seq = self.store.next_seq_no(session_id)?;
        self.store.append_event(&deepseek_core::EventEnvelope {
            seq_no: seq,
            at: chrono::Utc::now(),
            session_id,
            kind: EventKind::TeleportHandoffLinkCreatedV1 {
                handoff_id,
                session_id,
                expires_at: expires_at.clone(),
            },
        })?;

        let base_url = params
            .get("base_url")
            .and_then(|v| v.as_str())
            .unwrap_or("https://app.deepseek.com/handoff");
        let sep = if base_url.contains('?') { '&' } else { '?' };
        let link_url = format!("{base_url}{sep}handoff_id={handoff_id}&token={token}");

        Ok(serde_json::json!({
            "schema": "deepseek.handoff_link.v1",
            "handoff_id": handoff_id.to_string(),
            "bundle_path": bundle_path,
            "session_id": session_id.to_string(),
            "expires_at": expires_at,
            "token": token,
            "url": link_url,
        }))
    }

    fn handle_session_handoff_link_consume(&self, params: Value) -> Result<Value> {
        let handoff_id = require_uuid(&params, "handoff_id")?;
        let token = params
            .get("token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing 'token' parameter"))?;
        let mut descriptor = read_handoff_descriptor(self.store.root.as_path(), handoff_id)?;
        let now = chrono::Utc::now();

        let invalid_reason = if descriptor.used_at.is_some() {
            Some("already_used")
        } else if sha256_hex(token) != descriptor.token_hash {
            Some("token_mismatch")
        } else {
            let expires = chrono::DateTime::parse_from_rfc3339(&descriptor.expires_at)
                .map(|value| value.with_timezone(&chrono::Utc))
                .map_err(|_| anyhow!("invalid descriptor expiry timestamp"))?;
            if now > expires { Some("expired") } else { None }
        };

        if let Some(reason) = invalid_reason {
            let seq = self.store.next_seq_no(descriptor.session_id)?;
            self.store.append_event(&deepseek_core::EventEnvelope {
                seq_no: seq,
                at: chrono::Utc::now(),
                session_id: descriptor.session_id,
                kind: EventKind::TeleportHandoffLinkConsumedV1 {
                    handoff_id,
                    session_id: descriptor.session_id,
                    success: false,
                    reason: reason.to_string(),
                },
            })?;
            return Err(anyhow!("handoff token rejected: {reason}"));
        }

        let imported = self.handle_session_handoff_import(serde_json::json!({
            "bundle_path": descriptor.bundle_path,
            "resume": true,
        }))?;
        descriptor.used_at = Some(now.to_rfc3339());
        write_handoff_descriptor(self.store.root.as_path(), &descriptor)?;

        let seq = self.store.next_seq_no(descriptor.session_id)?;
        self.store.append_event(&deepseek_core::EventEnvelope {
            seq_no: seq,
            at: chrono::Utc::now(),
            session_id: descriptor.session_id,
            kind: EventKind::TeleportHandoffLinkConsumedV1 {
                handoff_id,
                session_id: descriptor.session_id,
                success: true,
                reason: "consumed".to_string(),
            },
        })?;

        Ok(serde_json::json!({
            "schema": "deepseek.handoff_link.v1",
            "handoff_id": handoff_id.to_string(),
            "consumed": true,
            "imported_session_id": imported["session_id"],
            "turn_count": imported["turn_count"],
        }))
    }

    fn handle_session_remote_resume(&self, params: Value) -> Result<Value> {
        if params.get("bundle_path").is_some() || params.get("handoff_path").is_some() {
            let imported = self.handle_session_handoff_import(params)?;
            return Ok(serde_json::json!({
                "remote": true,
                "mode": "handoff_import",
                "session_id": imported["session_id"],
                "status": imported["status"],
                "turn_count": imported["turn_count"],
            }));
        }
        let resumed = self.handle_session_resume(params.clone())?;
        let session_id = resumed
            .get("session_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing resumed session id"))?;
        let parsed = Uuid::parse_str(session_id)?;
        let events_replayed = resumed
            .get("turn_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let seq = self.store.next_seq_no(parsed)?;
        self.store.append_event(&deepseek_core::EventEnvelope {
            seq_no: seq,
            at: chrono::Utc::now(),
            session_id: parsed,
            kind: EventKind::SessionResumedV1 {
                session_id: parsed,
                events_replayed,
            },
        })?;

        Ok(serde_json::json!({
            "remote": true,
            "mode": "session_resume",
            "session_id": session_id,
            "status": resumed["status"],
            "turn_count": resumed["turn_count"],
            "workspace_root": resumed["workspace_root"],
        }))
    }

    fn handle_events_poll(&self, params: Value) -> Result<Value> {
        let session_id = require_uuid(&params, "session_id")?;
        let after_seq = params
            .get("after_seq")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as i64;
        let limit = limit.clamp(1, 500);

        let conn = self.store.db()?;
        let mut stmt = conn.prepare(
            "SELECT seq_no, at, kind, payload
             FROM events
             WHERE session_id = ?1 AND seq_no > ?2
             ORDER BY seq_no ASC
             LIMIT ?3",
        )?;
        let mut rows = stmt.query([
            session_id.to_string(),
            after_seq.to_string(),
            limit.to_string(),
        ])?;
        let mut events = Vec::new();
        let mut next_cursor = after_seq;
        while let Some(row) = rows.next()? {
            let seq_no = row.get::<_, i64>(0)?.max(0) as u64;
            let payload_raw = row.get::<_, String>(3)?;
            let payload_json = serde_json::from_str::<Value>(&payload_raw).unwrap_or(Value::Null);
            events.push(serde_json::json!({
                "seq_no": seq_no,
                "at": row.get::<_, String>(1)?,
                "kind": row.get::<_, String>(2)?,
                "payload": payload_json,
            }));
            next_cursor = seq_no;
        }
        let done = events.len() < limit as usize;

        Ok(serde_json::json!({
            "session_id": session_id.to_string(),
            "events": events,
            "next_cursor": next_cursor,
            "done": done,
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

    // -- Context management methods --

    fn handle_context_suggest(&self, params: Value) -> Result<Value> {
        let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");
        let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        let context_manager = self.context_manager.lock().expect("context_manager lock");

        if let Some(manager) = context_manager.as_ref() {
            let suggestions = manager.suggest_relevant_files(query, limit);
            let suggestions_json: Vec<Value> = suggestions
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "path": s.path.to_string_lossy(),
                        "score": s.score,
                        "reasons": s.reasons,
                    })
                })
                .collect();

            Ok(serde_json::json!({
                "suggestions": suggestions_json,
                "count": suggestions.len(),
                "query": query,
            }))
        } else {
            Ok(serde_json::json!({
                "suggestions": [],
                "count": 0,
                "query": query,
                "note": "Context analysis unavailable",
            }))
        }
    }

    fn handle_context_analyze(&self, _params: Value) -> Result<Value> {
        let mut context_manager = self.context_manager.lock().expect("context_manager lock");

        if let Some(manager) = context_manager.as_mut() {
            match manager.analyze_workspace() {
                Ok(()) => Ok(serde_json::json!({
                    "status": "analyzed",
                    "file_count": manager.file_count(),
                })),
                Err(e) => Ok(serde_json::json!({
                    "status": "error",
                    "error": e.to_string(),
                })),
            }
        } else {
            Ok(serde_json::json!({
                "status": "unavailable",
                "note": "Context manager not initialized",
            }))
        }
    }

    fn handle_context_compress(&self, params: Value) -> Result<Value> {
        let context = params.get("context").and_then(|v| v.as_str()).unwrap_or("");
        let max_tokens = params
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(4000) as usize;

        let context_manager = self.context_manager.lock().expect("context_manager lock");

        if let Some(manager) = context_manager.as_ref() {
            let compressed = manager.compress_context(context, max_tokens);
            let original_lines = context.lines().count();
            let compressed_lines = compressed.lines().count();

            Ok(serde_json::json!({
                "compressed": compressed,
                "original_lines": original_lines,
                "compressed_lines": compressed_lines,
                "compression_ratio": if original_lines > 0 {
                    (compressed_lines as f64) / (original_lines as f64)
                } else {
                    1.0
                },
            }))
        } else {
            // Fallback simple compression
            let lines: Vec<&str> = context.lines().collect();
            let total_lines = lines.len();
            let keep_lines = (max_tokens / 10).min(total_lines);

            let compressed = if total_lines > keep_lines {
                let start = keep_lines / 2;
                let end = keep_lines - start;
                let mut result: Vec<&str> = lines.iter().take(start).copied().collect();
                result.push("// ... context truncated ...");
                result.extend(lines.iter().rev().take(end).rev());
                result.join("\n")
            } else {
                context.to_string()
            };

            Ok(serde_json::json!({
                "compressed": compressed,
                "original_lines": total_lines,
                "compressed_lines": compressed.lines().count(),
                "compression_ratio": if total_lines > 0 {
                    (compressed.lines().count() as f64) / (total_lines as f64)
                } else {
                    1.0
                },
                "note": "Using fallback compression",
            }))
        }
    }

    fn handle_context_related(&self, params: Value) -> Result<Value> {
        let file_path = params
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing 'file_path' parameter"))?;
        let depth = params.get("depth").and_then(|v| v.as_u64()).unwrap_or(1) as usize;

        let path = Path::new(file_path);
        let context_manager = self.context_manager.lock().expect("context_manager lock");

        if let Some(manager) = context_manager.as_ref() {
            let related = manager.get_related_files(path, depth);
            let related_json: Vec<Value> = related
                .iter()
                .map(|p| serde_json::json!(p.to_string_lossy()))
                .collect();

            Ok(serde_json::json!({
                "file_path": file_path,
                "depth": depth,
                "related_files": related_json,
                "count": related.len(),
            }))
        } else {
            Ok(serde_json::json!({
                "file_path": file_path,
                "depth": depth,
                "related_files": [],
                "count": 0,
                "note": "Context analysis unavailable",
            }))
        }
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
                "partial_messages_enabled": self.include_partial_messages,
                "capabilities": [
                    "session/open", "session/resume", "session/fork", "session/list",
                    "session/remote_resume", "session/handoff_export", "session/handoff_import",
                    "session/handoff_link_create", "session/handoff_link_consume",
                    "prompt/execute", "prompt/stream_next",
                    "tool/approve", "tool/deny",
                    "patch/preview", "patch/apply",
                    "diagnostics/list", "events/poll",
                    "context/suggest", "context/analyze", "context/compress", "context/related",
                    "task/list", "task/update",
                    "status", "cancel", "shutdown"
                ]
            })),
            "status" => Ok(serde_json::json!({
                "status": "ready",
                "handler": "ide",
                "partial_messages_enabled": self.include_partial_messages,
            })),
            "cancel" => Ok(serde_json::json!({"cancelled": true})),

            // Session management
            "session/open" => self.handle_session_open(params),
            "session/resume" => self.handle_session_resume(params),
            "session/fork" => self.handle_session_fork(params),
            "session/list" => self.handle_session_list(params),
            "session/remote_resume" => self.handle_session_remote_resume(params),
            "session/handoff_export" => self.handle_session_handoff_export(params),
            "session/handoff_import" => self.handle_session_handoff_import(params),
            "session/handoff_link_create" => self.handle_session_handoff_link_create(params),
            "session/handoff_link_consume" => self.handle_session_handoff_link_consume(params),

            // Prompt execution
            "prompt/execute" => self.handle_prompt_execute(params),
            "prompt/stream_next" => self.handle_prompt_stream_next(params),

            // Tool approval/denial
            "tool/approve" => self.handle_tool_approve(params),
            "tool/deny" => self.handle_tool_deny(params),

            // Patch preview/apply
            "patch/preview" => self.handle_patch_preview(params),
            "patch/apply" => self.handle_patch_apply(params),

            // Diagnostics
            "diagnostics/list" => self.handle_diagnostics_list(params),
            "events/poll" => self.handle_events_poll(params),

            // Context management
            "context/suggest" => self.handle_context_suggest(params),
            "context/analyze" => self.handle_context_analyze(params),
            "context/compress" => self.handle_context_compress(params),
            "context/related" => self.handle_context_related(params),

            // Task management
            "task/list" => self.handle_task_list(params),
            "task/update" => self.handle_task_update(params),

            _ => Err(anyhow!("method not found: {method}")),
        }
    }
}

fn handoff_dir(store_root: &Path) -> PathBuf {
    store_root.join("teleport").join("handoffs")
}

fn handoff_descriptor_path(store_root: &Path, handoff_id: Uuid) -> PathBuf {
    handoff_dir(store_root).join(format!("{handoff_id}.json"))
}

fn write_handoff_descriptor(store_root: &Path, descriptor: &HandoffDescriptor) -> Result<()> {
    let path = handoff_descriptor_path(store_root, descriptor.handoff_id);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(descriptor)?)?;
    Ok(())
}

fn read_handoff_descriptor(store_root: &Path, handoff_id: Uuid) -> Result<HandoffDescriptor> {
    let path = handoff_descriptor_path(store_root, handoff_id);
    let raw = fs::read_to_string(&path)?;
    let descriptor = serde_json::from_str::<HandoffDescriptor>(&raw)?;
    Ok(descriptor)
}

fn sha256_hex(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Extract a UUID from a JSON params object.
fn require_uuid(params: &Value, field: &str) -> Result<Uuid> {
    let s = params
        .get(field)
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing '{field}' parameter"))?;
    Uuid::parse_str(s).map_err(|e| anyhow!("invalid UUID for '{field}': {e}"))
}

fn optional_uuid(params: &Value, field: &str) -> Result<Option<Uuid>> {
    let Some(raw) = params.get(field).and_then(|v| v.as_str()) else {
        return Ok(None);
    };
    Ok(Some(
        Uuid::parse_str(raw).map_err(|e| anyhow!("invalid UUID for '{field}': {e}"))?,
    ))
}

fn env_flag_enabled(key: &str) -> bool {
    match std::env::var(key) {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            !normalized.is_empty() && !matches!(normalized.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => false,
    }
}

fn split_partial_chunks(text: &str, max_chunk_bytes: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for token in text.split_whitespace() {
        let candidate_len = if current.is_empty() {
            token.len()
        } else {
            current.len() + 1 + token.len()
        };
        if candidate_len > max_chunk_bytes && !current.is_empty() {
            out.push(current.clone());
            current.clear();
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(token);
    }
    if !current.is_empty() {
        out.push(current);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
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
        assert!(caps.iter().any(|c| c == "prompt/stream_next"));
        assert!(caps.iter().any(|c| c == "session/handoff_export"));
        assert!(caps.iter().any(|c| c == "session/handoff_import"));
        assert!(caps.iter().any(|c| c == "session/handoff_link_create"));
        assert!(caps.iter().any(|c| c == "session/handoff_link_consume"));
        assert!(caps.iter().any(|c| c == "session/remote_resume"));
        assert!(caps.iter().any(|c| c == "events/poll"));
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
    fn ide_handler_prompt_partial_stream() {
        let tmp = tempfile::tempdir().unwrap();
        let handler = IdeRpcHandler::new(tmp.path()).unwrap();

        let opened = handler
            .handle("session/open", json!({"workspace_root": "/tmp/stream"}))
            .unwrap();
        let sid = opened["session_id"].as_str().unwrap();
        let exec = handler
            .handle(
                "prompt/execute",
                json!({
                    "session_id": sid,
                    "prompt": "stream this response please",
                    "include_partial_messages": true
                }),
            )
            .expect("prompt/execute");
        assert_eq!(exec["partial_messages_enabled"], true);
        let prompt_id = exec["prompt_id"].as_str().unwrap();

        let chunk_1 = handler
            .handle(
                "prompt/stream_next",
                json!({"prompt_id": prompt_id, "cursor": 0, "max_chunks": 1}),
            )
            .expect("prompt/stream_next first");
        assert_eq!(chunk_1["chunks"].as_array().unwrap().len(), 1);
        let next_cursor = chunk_1["next_cursor"].as_u64().unwrap();
        assert!(next_cursor >= 1);
        if !chunk_1["done"].as_bool().unwrap_or(false) {
            let chunk_2 = handler
                .handle(
                    "prompt/stream_next",
                    json!({"prompt_id": prompt_id, "cursor": next_cursor, "max_chunks": 100}),
                )
                .expect("prompt/stream_next second");
            assert_eq!(chunk_2["done"], true);
        }
    }

    #[test]
    fn ide_handler_handoff_export_import_remote_resume() {
        let tmp = tempfile::tempdir().unwrap();
        let handler = IdeRpcHandler::new(tmp.path()).unwrap();

        let opened = handler
            .handle("session/open", json!({"workspace_root": "/tmp/handoff"}))
            .unwrap();
        let sid = opened["session_id"].as_str().unwrap();
        let _ = handler
            .handle(
                "prompt/execute",
                json!({"session_id": sid, "prompt": "save this conversation"}),
            )
            .expect("prompt/execute");

        let export = handler
            .handle("session/handoff_export", json!({"session_id": sid}))
            .expect("handoff_export");
        let bundle_path = export["path"].as_str().unwrap();
        assert!(std::path::Path::new(bundle_path).exists());

        let imported = handler
            .handle(
                "session/handoff_import",
                json!({"bundle_path": bundle_path, "resume": true}),
            )
            .expect("handoff_import");
        assert_eq!(imported["imported"], true);
        let imported_id = imported["session_id"].as_str().unwrap();
        assert!(!imported_id.is_empty());

        let remote = handler
            .handle(
                "session/remote_resume",
                json!({"session_id": imported_id, "transport": "stdio"}),
            )
            .expect("remote_resume");
        assert_eq!(remote["remote"], true);
        assert_eq!(remote["session_id"], imported_id);
    }

    #[test]
    fn ide_handler_handoff_link_create_and_consume() {
        let tmp = tempfile::tempdir().unwrap();
        let handler = IdeRpcHandler::new(tmp.path()).unwrap();

        let opened = handler
            .handle(
                "session/open",
                json!({"workspace_root": "/tmp/handoff-link"}),
            )
            .unwrap();
        let sid = opened["session_id"].as_str().unwrap();
        let _ = handler
            .handle(
                "prompt/execute",
                json!({"session_id": sid, "prompt": "handoff link seed"}),
            )
            .expect("prompt/execute");

        let created = handler
            .handle(
                "session/handoff_link_create",
                json!({"session_id": sid, "ttl_minutes": 10}),
            )
            .expect("handoff_link_create");
        assert_eq!(created["schema"], "deepseek.handoff_link.v1");
        let handoff_id = created["handoff_id"].as_str().unwrap();
        let token = created["token"].as_str().unwrap();
        assert!(!handoff_id.is_empty());
        assert!(!token.is_empty());

        let consumed = handler
            .handle(
                "session/handoff_link_consume",
                json!({"handoff_id": handoff_id, "token": token}),
            )
            .expect("handoff_link_consume");
        assert_eq!(consumed["consumed"], true);
        assert!(consumed["imported_session_id"].as_str().is_some());
    }

    #[test]
    fn ide_handler_events_poll_returns_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let handler = IdeRpcHandler::new(tmp.path()).unwrap();
        let opened = handler
            .handle("session/open", json!({"workspace_root": "/tmp/events"}))
            .unwrap();
        let sid = opened["session_id"].as_str().unwrap();
        let _ = handler
            .handle(
                "prompt/execute",
                json!({"session_id": sid, "prompt": "hello"}),
            )
            .expect("prompt/execute");

        let poll = handler
            .handle(
                "events/poll",
                json!({"session_id": sid, "after_seq": 0, "limit": 10}),
            )
            .expect("events/poll");
        let events = poll["events"].as_array().unwrap();
        assert!(!events.is_empty());
        assert!(events[0]["seq_no"].as_u64().is_some());
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
