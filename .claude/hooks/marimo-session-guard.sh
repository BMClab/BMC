#!/usr/bin/env bash
# PostToolUse(Edit|Write): warn when editing a marimo notebook that has a committed
# session cache, so the cache isn't left stale before commit. (Cannot auto-regenerate:
# marimo only writes the cache from the editor — re-run all cells there.)
set -euo pipefail

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')

[ -n "$file" ] || exit 0
case "$file" in
  */notebooks_marimo/*.py) ;;
  *) exit 0 ;;
esac

base=$(basename "$file")
cache="$(dirname "$file")/__marimo__/session/${base}.json"
[ -f "$cache" ] || exit 0          # only warn when a tracked cache exists

msg="⚠️ Edited ${base}, which has a committed marimo session cache. Re-run all cells via 'uv run marimo edit ${file}' before committing so __marimo__/session/${base}.json stays in sync."
jq -nc --arg m "$msg" '{systemMessage: $m, hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext: $m}}'
exit 0
