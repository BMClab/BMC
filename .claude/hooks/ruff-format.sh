#!/usr/bin/env bash
# PostToolUse(Edit|Write): format + lint-fix the edited Python file with Ruff.
# Skips marimo notebooks (reactive; reformatting would desync __marimo__/session caches).
set -euo pipefail

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')

[ -n "$file" ] || exit 0
case "$file" in
  *.py) ;;                          # only Python files
  *) exit 0 ;;
esac
case "$file" in
  */notebooks_marimo/*) exit 0 ;;   # leave marimo notebooks to the marimo editor
esac
[ -f "$file" ] || exit 0

uv run ruff format "$file" >/dev/null 2>&1 || true
uv run ruff check --fix "$file" >/dev/null 2>&1 || true
exit 0
