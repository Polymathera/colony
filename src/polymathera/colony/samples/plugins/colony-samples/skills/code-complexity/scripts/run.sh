#!/usr/bin/env bash
# Thin wrapper around the Python complexity scanner. Forwards all
# --foo VALUE pairs the capability passes (path, threshold, top_n) to
# the scanner. Stdin/stdout are plain JSON so the LLM can parse them.
set -euo pipefail
exec python3 "$(dirname "$0")/complexity.py" "$@"
