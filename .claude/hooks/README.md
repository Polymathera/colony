# `.claude/hooks/` — Claude Code PreToolUse hooks for this repo

Hooks here are auto-discovered by Claude Code when invoked with
this repo's directory as the working directory (the hook path uses
`${CLAUDE_PROJECT_DIR}` so it resolves correctly after a fresh
`git clone`). No per-user install is required.

## `read-before-edit.sh`

Blocks Claude from editing a file unless it has Read the relevant
section in the current session.

### Why

Without this, Claude periodically ships edits that conflict with
adjacent code it never saw: duplicated lines, redundant guards,
helpers that recreate logic already present three lines down. The
`Edit` tool's `old_string` match catches *syntactic* mismatches —
it does not catch the *semantic* error of adding correct-on-its-own
code next to existing equivalent code. This hook closes that gap.

### How

On every `PreToolUse:Edit|Write`, the hook reads the session
transcript (`transcript_path` is provided on stdin per Claude Code's
hook protocol) and applies two checks:

1. **Existence**: was `file_path` ever Read by the `Read` tool in
   this session? If not → block with a "Read the file first" message.
2. **Range coverage** (Edit only): grep the current file for the
   first line of `old_string` to find the edit's anchor line, then
   verify at least one prior Read of this `file_path` covered that
   line. Reads with no `offset`/`limit` default to (1, 999999),
   meaning full-file reads always satisfy the check. If no Read
   covered the line → block with a "Re-Read with appropriate
   offset/limit" message.

Block exits with status 2 so the message surfaces back to Claude as
a tool error, prompting it to Read and retry.

### Test scenarios (all passing)

| # | Setup | Expected | Result |
|---|---|---|---|
| A | Edit a file never Read | block | ✅ block |
| B | Read offset=40 limit=20, Edit at line 50 | allow | ✅ allow |
| C | Read offset=10 limit=20, Edit at line 50 | block | ✅ block |
| D | Full-file Read (no offset/limit), Edit anywhere | allow | ✅ allow |
| E | Write to a file that doesn't exist yet | allow | ✅ allow |
| F | Read offset=820 limit=75, Edit at line 850 | allow | ✅ allow |
| G | Read offset=752 limit=40, Edit at line 850 | block | ✅ block |

Test G reproduces the exact pattern of the incident that motivated
this hook (`colony/distributed/stores/git.py`, May 2026).

### Limits — what this hook does NOT catch

- **Read the right section but ignored context**: the hook can't
  read minds. If the relevant context is in the Read range but the
  agent skipped over it, the hook still passes.
- **Stale Read**: a Read from 200 tool-uses ago still satisfies the
  existence check even if the file has since been modified by Bash.
  Could be added (count tool-uses since the last Read of this
  file) but currently isn't.
- **Edit racing a Bash modification**: if a `sed` / formatter runs
  between Read and Edit, the line numbers may have shifted. The
  hook checks the *current* file's line numbers, so a shifted line
  may still satisfy or falsely fail. Edit's own old_string match
  catches the false-allow case.

### Replicating this hook in sibling repos (cps, monorepo_opm_meg)

Copy `read-before-edit.sh` and `settings.json` into the target
repo's `.claude/`. The hook script uses `${CLAUDE_PROJECT_DIR}` so
it resolves to whatever repo Claude is invoked from — no path
edits needed. When you update the hook here, mirror the change
into the other repos (or symlink, but a clone-time copy is more
robust against missing siblings).
