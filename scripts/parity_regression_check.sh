#!/usr/bin/env bash
set -euo pipefail

echo "Running section 10 parity regression checklist..."

# Agent parity closures (bootstrap + teammate determinism fallback)
cargo test -p codingbuddy-agent --test analysis_bootstrap
cargo test -p codingbuddy-agent --test team_orchestration

# TUI parity (history search, ghost suggestions, Vim text objects/operators)
cargo test -p codingbuddy-ui reverse_search_and_ghost_helpers_work
cargo test -p codingbuddy-ui vim_text_object_bounds_support_word_and_quotes
cargo test -p codingbuddy-ui vim_operator_range_handles_change_delete_yank
cargo test -p codingbuddy-ui render_statusline_plan_mode_label

# Runtime conformance deny-list
bash scripts/runtime_conformance_scan.sh
