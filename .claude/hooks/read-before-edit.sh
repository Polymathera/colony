#!/usr/bin/env bash
# PreToolUse hook for Edit/Write.
#
# Why this exists
# ---------------
# When the agent edits a file without first Reading the section it
# intends to change, it ships changes that conflict with adjacent
# code it never saw — duplicated lines, redundant guards, wrong-
# scope helpers. The Edit tool's own ``old_string`` match catches
# syntactic mismatches but NOT semantic ones (you can add a
# redundant line next to existing equivalent code without Edit
# noticing). This hook adds a mechanical "Read before Edit" check
# that survives across sessions.
#
# Blocks when:
#   1. ``file_path`` was never Read in this session, OR
#   2. (Edit only) no Read of ``file_path`` covered the line
#      containing the first line of ``old_string`` in the current
#      file content. Catches the "Read wrong section" failure mode.
#
# Allows when:
#   - Tool isn't Edit/Write -> pass
#   - file_path / transcript_path missing or transcript not yet a
#     file -> pass (defensive — happens on the very first tool call
#     in a session)
#   - Write of a non-existent file -> pass (creating new file; no
#     prior content to read)
#   - Edit's first line of old_string not found in file -> pass
#     (Edit will fail with its own error; not the hook's job to
#     duplicate that diagnostic)
#
# Exit codes:
#   0 = allow
#   2 = block (the stderr message is shown back to the agent as a
#       tool error, prompting it to Read the right range and retry).
#
# Tested with seven scenarios (see colony/.claude/hooks/README.md).
# Test G is the original incident that motivated this hook — it's
# faithfully reproduced and the hook blocks it cleanly.

set -euo pipefail

payload="$(cat)"
tool=$(jq -r '.tool_name // empty' <<<"$payload")
[[ "$tool" != "Edit" && "$tool" != "Write" ]] && exit 0

fp=$(jq -r '.tool_input.file_path // empty' <<<"$payload")
transcript=$(jq -r '.transcript_path // empty' <<<"$payload")
[[ -z "$fp" || -z "$transcript" || ! -f "$transcript" ]] && exit 0

# Write of a non-existent file is fine — no prior content to read.
[[ "$tool" == "Write" && ! -e "$fp" ]] && exit 0

# Every Read of this file_path in the transcript, with its (offset, limit).
# Defaults: offset=1, limit=999999 (full-file read covers everything).
reads_json=$(jq -c --arg fp "$fp" '
  select(.message.content)
  | .message.content[]?
  | select(.type=="tool_use" and .name=="Read" and .input.file_path==$fp)
  | {offset: (.input.offset // 1), limit: (.input.limit // 999999)}
' "$transcript" 2>/dev/null || true)

if [[ -z "$reads_json" ]]; then
  echo "[read-before-edit] BLOCK: $fp was never Read in this session. Use the Read tool first." >&2
  exit 2
fi

# Write doesn't have old_string; presence of any Read is enough.
[[ "$tool" == "Write" ]] && exit 0

# Edit: verify some Read covered the line containing old_string's first line.
old=$(jq -r '.tool_input.old_string // empty' <<<"$payload")
[[ -z "$old" || ! -f "$fp" ]] && exit 0
first_line=$(printf '%s\n' "$old" | head -1)
[[ -z "$first_line" ]] && exit 0

# Locate the edit's anchor line in the current file content.
edit_line=$(grep -nF -- "$first_line" "$fp" 2>/dev/null | head -1 | cut -d: -f1 || true)
# If first line not found in file, let Edit produce its own error.
[[ -z "$edit_line" ]] && exit 0

covered=0
while IFS= read -r r; do
  [[ -z "$r" ]] && continue
  off=$(jq -r '.offset' <<<"$r")
  lim=$(jq -r '.limit' <<<"$r")
  end=$((off + lim - 1))
  if [[ "$edit_line" -ge "$off" && "$edit_line" -le "$end" ]]; then
    covered=1
    break
  fi
done <<<"$reads_json"

if [[ "$covered" -eq 0 ]]; then
  echo "[read-before-edit] BLOCK: Edit hits line $edit_line of $fp but no Read covered that line range. Re-Read with appropriate offset/limit before editing." >&2
  exit 2
fi

exit 0
