#!/usr/bin/env bash
set -euo pipefail

DEEPSEEK_BIN="${DEEPSEEK_BIN:-deepseek}"
exec "$DEEPSEEK_BIN" native-host "$@"
