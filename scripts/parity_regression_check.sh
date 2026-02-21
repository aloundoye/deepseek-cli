#!/usr/bin/env bash
set -euo pipefail

echo "Running section 10 parity regression checklist..."

# CLI parity flags / commands
cargo test -p deepseek-cli --test cli_json parity_flags_update_and_teleport_emit_json
cargo test -p deepseek-cli --test cli_json remote_env_check_performs_real_health_probe

# TUI parity (history search, ghost suggestions, Vim text objects/operators)
cargo test -p deepseek-ui reverse_search_and_ghost_helpers_work
cargo test -p deepseek-ui vim_text_object_bounds_support_word_and_quotes
cargo test -p deepseek-ui vim_operator_range_handles_change_delete_yank
cargo test -p deepseek-ui render_statusline_plan_mode_label
