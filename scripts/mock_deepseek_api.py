#!/usr/bin/env python3
"""Minimal DeepSeek-compatible mock API server for CI smoke and reliability tests."""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class DeepSeekMockHandler(BaseHTTPRequestHandler):
    request_count = 0
    lock = threading.Lock()
    fail_first = 0
    fail_status = 503
    response_delay_ms = 0

    def log_message(self, fmt, *args):
        # Keep CI logs clean.
        return

    def do_POST(self):
        if self.path != "/chat/completions":
            self._json(404, {"error": "not_found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        payload = self._safe_json(raw_body)

        with self.lock:
            DeepSeekMockHandler.request_count += 1
            current_request = DeepSeekMockHandler.request_count

        if DeepSeekMockHandler.response_delay_ms > 0:
            time.sleep(DeepSeekMockHandler.response_delay_ms / 1000.0)

        if current_request <= DeepSeekMockHandler.fail_first:
            headers = {"Retry-After": "0"} if DeepSeekMockHandler.fail_status == 429 else None
            self._json(
                DeepSeekMockHandler.fail_status,
                {"error": "transient_mock_failure", "attempt": current_request},
                headers=headers,
            )
            return

        prompt = self._extract_prompt(payload)
        if "plan" in prompt.lower():
            content = "Generated plan: discover files, propose edits, verify with tests."
        elif "status" in prompt.lower():
            content = "Status: mock service healthy."
        else:
            content = f"Mock response: {prompt or 'ok'}"

        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": content,
                        "reasoning_content": "mock reasoning",
                        "tool_calls": [],
                    },
                }
            ]
        }
        self._json(200, response)

    def _safe_json(self, payload):
        try:
            return json.loads(payload.decode("utf-8") or "{}")
        except Exception:
            return {}

    def _extract_prompt(self, payload):
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return ""
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            return str(content)
        return ""

    def _json(self, status, payload, headers=None):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(encoded)


def main():
    parser = argparse.ArgumentParser(description="Run a local DeepSeek API mock server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--fail-first", type=int, default=0)
    parser.add_argument("--fail-status", type=int, default=503)
    parser.add_argument("--response-delay-ms", type=int, default=0)
    args = parser.parse_args()

    DeepSeekMockHandler.request_count = 0
    DeepSeekMockHandler.fail_first = max(args.fail_first, 0)
    DeepSeekMockHandler.fail_status = args.fail_status
    DeepSeekMockHandler.response_delay_ms = max(args.response_delay_ms, 0)

    server = ThreadingHTTPServer((args.host, args.port), DeepSeekMockHandler)
    print(
        f"mock deepseek api listening on http://{args.host}:{args.port}/chat/completions",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
