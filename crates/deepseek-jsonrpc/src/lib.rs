use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{BufRead, Write};

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
                    Err(e) => JsonRpcResponse::error(req.id, -32603, e.to_string()),
                }
            }
            Err(_) => JsonRpcResponse::error(Value::Null, -32700, "Parse error"),
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
        let resp = JsonRpcResponse::error(Value::Null, -32700, "Parse error");
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
        let resp = JsonRpcResponse::error(json!(1), -32601, "method not found");
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
}
