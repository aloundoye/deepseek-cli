use anyhow::{Result, anyhow};
use clap::CommandFactory;
use clap_complete::generate;
use codingbuddy_jsonrpc::{JsonRpcRequest, JsonRpcResponse, RpcHandler};
use serde_json::{Value, json};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::{Cli, CompletionsArgs, NativeHostArgs, ServeArgs};

const MAX_NATIVE_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

pub(crate) fn run_completions(args: CompletionsArgs) -> Result<()> {
    let mut cmd = Cli::command();
    generate(args.shell, &mut cmd, "codingbuddy", &mut io::stdout());
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
                eprintln!("codingbuddy: starting JSON-RPC server on stdio...");
            }
            let workspace = std::env::current_dir()?;
            let handler = codingbuddy_jsonrpc::IdeRpcHandler::new(&workspace)?;
            codingbuddy_jsonrpc::run_stdio_server(&handler)
        }
        other => Err(anyhow!(
            "unsupported transport '{}' (supported: stdio)",
            other
        )),
    }
}

pub(crate) fn run_native_host(cwd: &Path, args: NativeHostArgs) -> Result<()> {
    let workspace_root = args.workspace_root.unwrap_or_else(|| cwd.to_path_buf());
    let handler = codingbuddy_jsonrpc::IdeRpcHandler::new(&workspace_root)?;
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = BufWriter::new(stdout.lock());

    while let Some(raw) = read_native_message(&mut reader)? {
        let (response, should_shutdown) = handle_native_request(&handler, &raw);
        let encoded = serde_json::to_vec(&response)?;
        write_native_message(&mut writer, &encoded)?;
        if should_shutdown {
            break;
        }
    }

    Ok(())
}

fn handle_native_request(handler: &impl RpcHandler, raw: &[u8]) -> (JsonRpcResponse, bool) {
    let request = match parse_native_request(raw) {
        Ok(request) => request,
        Err(err) => {
            return (
                JsonRpcResponse::error(
                    Value::Null,
                    codingbuddy_jsonrpc::ERR_PARSE,
                    err.to_string(),
                ),
                false,
            );
        }
    };

    let should_shutdown = request.method == "shutdown";
    if should_shutdown {
        return (
            JsonRpcResponse::success(request.id, json!({"ok": true, "transport": "native"})),
            true,
        );
    }

    match handler.handle(&request.method, request.params) {
        Ok(result) => (JsonRpcResponse::success(request.id, result), false),
        Err(err) => (
            JsonRpcResponse::error(
                request.id,
                codingbuddy_jsonrpc::ERR_INTERNAL,
                err.to_string(),
            ),
            false,
        ),
    }
}

fn parse_native_request(raw: &[u8]) -> Result<JsonRpcRequest> {
    let value: Value = serde_json::from_slice(raw)?;
    if value.get("jsonrpc").is_none() && value.get("method").is_some() {
        let method = value
            .get("method")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("invalid request: 'method' must be a string"))?
            .to_string();
        let id = value.get("id").cloned().unwrap_or(Value::Null);
        let params = value.get("params").cloned().unwrap_or_else(|| json!({}));
        return Ok(JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id,
            method,
            params,
        });
    }
    Ok(serde_json::from_value(value)?)
}

fn read_native_message(reader: &mut impl Read) -> Result<Option<Vec<u8>>> {
    let mut len_buf = [0_u8; 4];
    match reader.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err.into()),
    }

    let len = u32::from_le_bytes(len_buf) as usize;
    if len == 0 {
        return Err(anyhow!("invalid native message length: 0"));
    }
    if len > MAX_NATIVE_MESSAGE_BYTES {
        return Err(anyhow!(
            "native message too large: {} bytes (max {})",
            len,
            MAX_NATIVE_MESSAGE_BYTES
        ));
    }

    let mut payload = vec![0_u8; len];
    reader.read_exact(&mut payload)?;
    Ok(Some(payload))
}

fn write_native_message(writer: &mut impl Write, payload: &[u8]) -> Result<()> {
    let len = u32::try_from(payload.len()).map_err(|_| anyhow!("native message too large"))?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(payload)?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoHandler;

    impl RpcHandler for EchoHandler {
        fn handle(&self, method: &str, params: Value) -> Result<Value> {
            Ok(json!({"method": method, "params": params}))
        }
    }

    #[test]
    fn parse_native_request_supports_simplified_shape() {
        let request = parse_native_request(br#"{"id":1,"method":"status","params":{"ok":true}}"#)
            .expect("request");
        assert_eq!(request.jsonrpc, "2.0");
        assert_eq!(request.method, "status");
        assert_eq!(request.id, json!(1));
        assert_eq!(request.params["ok"], true);
    }

    #[test]
    fn native_message_round_trip() {
        let payload = br#"{"jsonrpc":"2.0","id":1,"method":"status","params":{}}"#;
        let mut out = Vec::new();
        write_native_message(&mut out, payload).expect("write");
        let mut cursor = io::Cursor::new(out);
        let read_back = read_native_message(&mut cursor)
            .expect("read")
            .expect("payload");
        assert_eq!(read_back, payload);
    }

    #[test]
    fn handle_native_request_shutdown_short_circuits() {
        let (response, should_shutdown) = handle_native_request(
            &EchoHandler,
            br#"{"jsonrpc":"2.0","id":"x","method":"shutdown","params":{}}"#,
        );
        assert!(should_shutdown);
        assert!(response.error.is_none());
    }
}
