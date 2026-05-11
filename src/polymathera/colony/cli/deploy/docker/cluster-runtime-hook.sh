#!/bin/bash
# Container-start hook for the L1-G image-extension mechanism.
#
# Reads a resolved-runtime JSON written by ``colony-env up``, pip-installs
# ``cluster.extensions.packages`` into a persistent overlay volume (cache-
# keyed by the hash in the JSON), then runs ``setup_commands`` followed by
# the role-specific ``{head,worker}_setup_commands``.
#
# The hook is idempotent: re-running with the same JSON skips pip install
# (hash file matches the JSON's hash). When the hash changes, the overlay
# is wiped and re-populated from scratch — pip install --target leaks stale
# files otherwise (no built-in uninstall for --target installs).
#
# Usage:
#   cluster-runtime-hook.sh <role>                — run hook only, exit 0.
#   cluster-runtime-hook.sh <role> -- <cmd...>    — run hook, then exec cmd.
#
# <role>: head | worker | dashboard. Picks which role-specific setup
# commands fire (general setup_commands always run).
#
# Reads:
#   COLONY_CLUSTER_RUNTIME_CONFIG    JSON path (default /etc/colony/cluster-runtime.json,
#                                    a read-only bind mount written by colony-env up).
#   COLONY_OVERLAY_DIR               Overlay install dir (default /opt/colony-overlay)
#
# Exports:
#   PYTHONPATH                       <overlay>:<previous PYTHONPATH>
#                                    so installed packages are importable.
#
# JSON shape:
#   { "hash": "<16hex>",
#     "pip_args": ["<version-spec>", ...],           # installed WITH deps
#     "pip_args_no_deps": ["<container-path>", ...], # installed --no-deps
#     "setup_commands": [...], "head_setup_commands": [...],
#     "worker_setup_commands": [...] }
#
# ``pip_args_no_deps`` is for path-source extensions: their pyproject's
# dependency tree (typically ``polymathera-colony`` for CPS-shaped
# extensions) is already satisfied by the base image, and ``pip install
# --target`` would otherwise trigger redundant overlay-installs of those
# deps with editable-vs-target conflicts.
#
# A missing JSON file is not an error — the hook treats it as "no
# extensions" and proceeds. ``colony-env up`` may write the file
# asynchronously w.r.t. container start, in which case the hook either
# sees the file (extensions installed) or proceeds without it (operator
# left ``cluster.extensions`` empty); both are valid.

set -euo pipefail

ROLE="${1:-}"
case "$ROLE" in
    head|worker|dashboard) ;;
    *)
        echo "[cluster-runtime-hook] usage: $0 <head|worker|dashboard> [-- cmd args...]" >&2
        exit 64
        ;;
esac
shift

# Drain "--" if present; remainder becomes the command to exec after the hook.
EXEC_CMD=()
if [[ "${1:-}" == "--" ]]; then
    shift
    EXEC_CMD=("$@")
fi

CONFIG_FILE="${COLONY_CLUSTER_RUNTIME_CONFIG:-/etc/colony/cluster-runtime.json}"
OVERLAY_DIR="${COLONY_OVERLAY_DIR:-/opt/colony-overlay}"
HASH_FILE="$OVERLAY_DIR/.installed-hash"

mkdir -p "$OVERLAY_DIR"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[cluster-runtime-hook] no config at $CONFIG_FILE; skipping extensions"
else
    # Use python's json module rather than jq — jq isn't guaranteed in the
    # base image, python is.
    HASH=$(python -c "import json; print(json.load(open('$CONFIG_FILE'))['hash'])")

    if [[ -f "$HASH_FILE" && "$(cat "$HASH_FILE")" == "$HASH" ]]; then
        echo "[cluster-runtime-hook] overlay hash $HASH matches; skipping pip install"
    else
        # Hash mismatch — wipe and reinstall. ``pip install --target`` does
        # not track / uninstall older entries; partial overlays poison
        # ``import`` resolution. Cheapest correct behaviour is wipe-then-
        # install on every hash change.
        find "$OVERLAY_DIR" -mindepth 1 -delete 2>/dev/null || true
        readarray -t PIP_ARGS < <(python -c "import json; print('\n'.join(json.load(open('$CONFIG_FILE'))['pip_args']))")
        readarray -t PIP_ARGS_NO_DEPS < <(python -c "import json; print('\n'.join(json.load(open('$CONFIG_FILE')).get('pip_args_no_deps', []) or []))")
        # Filter empty lines that readarray emits for empty input.
        PIP_ARGS=("${PIP_ARGS[@]/#/}"); PIP_ARGS=("${PIP_ARGS[@]/%/}")
        PIP_ARGS=($(printf '%s\n' "${PIP_ARGS[@]}" | awk 'NF'))
        PIP_ARGS_NO_DEPS=($(printf '%s\n' "${PIP_ARGS_NO_DEPS[@]}" | awk 'NF'))
        if [[ ${#PIP_ARGS[@]} -gt 0 ]]; then
            echo "[cluster-runtime-hook] installing ${#PIP_ARGS[@]} version-source extension(s) into $OVERLAY_DIR (hash $HASH)"
            pip install --target="$OVERLAY_DIR" "${PIP_ARGS[@]}"
        fi
        if [[ ${#PIP_ARGS_NO_DEPS[@]} -gt 0 ]]; then
            echo "[cluster-runtime-hook] installing ${#PIP_ARGS_NO_DEPS[@]} path-source extension(s) into $OVERLAY_DIR (--no-deps; hash $HASH)"
            pip install --target="$OVERLAY_DIR" --no-deps "${PIP_ARGS_NO_DEPS[@]}"
        fi
        echo "$HASH" > "$HASH_FILE"
    fi
fi

# Make installed packages importable. Prepend so newer extension versions
# win over any base-image-bundled copies; PYTHONPATH was set in
# Dockerfile.base to ``${APP_MOUNT_PATH}/src``.
export PYTHONPATH="$OVERLAY_DIR:${PYTHONPATH:-}"

# Setup commands. ``setup_commands`` runs on every node; role-specific
# variants fire only on the matching role. A missing key in the JSON is
# the same as an empty list. Lines are run with ``bash -c`` so the
# operator can use shell features (pipes, &&, env-var substitution).
run_commands() {
    local key="$1"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        return 0
    fi
    local cmds
    cmds=$(python -c "import json; print('\n'.join(json.load(open('$CONFIG_FILE')).get('$key', []) or []))")
    if [[ -z "$cmds" ]]; then
        return 0
    fi
    echo "[cluster-runtime-hook] running $key"
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        bash -c "$line"
    done <<< "$cmds"
}

run_commands "setup_commands"
case "$ROLE" in
    head) run_commands "head_setup_commands" ;;
    worker) run_commands "worker_setup_commands" ;;
    dashboard) ;;  # no role-specific setup commands for the dashboard
esac

if [[ ${#EXEC_CMD[@]} -gt 0 ]]; then
    exec "${EXEC_CMD[@]}"
fi
