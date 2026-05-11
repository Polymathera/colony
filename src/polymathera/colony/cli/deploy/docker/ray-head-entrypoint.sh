#!/bin/bash
# Colony ray-head entrypoint — starts Ray head node and deploys Colony cluster.
#
# 1. Start Ray head node (non-blocking — returns after daemon starts)
# 2. Wait for Ray to be ready
# 3. Deploy Colony cluster via `polymath deploy` and BLOCK (keeps actors alive)
#
# The `polymath deploy` process MUST stay alive because it owns the serving
# application actors. If it exits, Ray garbage-collects the actors.

set -e

# L1-G: install ``cluster.extensions.packages`` into the persistent overlay
# and run head-role setup commands BEFORE Ray starts, so any extension
# packages and any operator-supplied setup are visible to the Ray process
# tree. The hook is a no-op when no operator config is present. Run as a
# subprocess so its ``set -euo pipefail`` does not leak into this script;
# we explicitly export PYTHONPATH afterward so the overlay is importable
# in everything the head spawns.
HOOK="$(dirname "$0")/cluster-runtime-hook.sh"
OVERLAY_DIR="${COLONY_OVERLAY_DIR:-/opt/colony-overlay}"
if [ -x "$HOOK" ]; then
    "$HOOK" head
fi
export PYTHONPATH="$OVERLAY_DIR:${PYTHONPATH:-}"

echo "[colony] Starting Ray head node..."
ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --ray-client-server-port=10001 \
    --num-cpus=0

echo "[colony] Waiting for Ray to be ready..."
until ray status > /dev/null 2>&1; do
    sleep 1
done
echo "[colony] Ray head node ready."

# Auto-deploy Colony cluster.
#
# Choice of CLI --config path is now: an alternate /etc/colony/cluster.yaml
# (operator-mounted), otherwise the canonical /mnt/shared/config.yaml the
# colony-env CLI `docker cp`s in. Both ConfigurationManager and
# load_config_from_yaml tolerate the file being absent at boot — the
# manager waits up to ``wait_for_config_seconds`` for the docker-cp race
# (default 15 s, set in ConfigurationManager.__init__), then falls through
# to defaults + env vars cleanly. No shell-side wait loop required.
if [ "${COLONY_AUTO_DEPLOY:-true}" = "true" ]; then
    if [ -f "/etc/colony/cluster.yaml" ]; then
        CONFIG_FLAG="--config /etc/colony/cluster.yaml"
    else
        CONFIG_FLAG="--config /mnt/shared/config.yaml"
    fi

    echo "[colony] Deploying Colony cluster (blocking — keeps actors alive)..."
    # exec replaces the shell with polymath deploy, which blocks forever
    # after deployment to keep the serving app actors alive.
    exec python -m polymathera.colony.cli.polymath deploy --block $CONFIG_FLAG
else
    echo "[colony] Auto-deploy disabled (COLONY_AUTO_DEPLOY=false)."
    echo "[colony] Run 'polymath deploy --block' manually to deploy and keep alive."
    sleep infinity
fi
