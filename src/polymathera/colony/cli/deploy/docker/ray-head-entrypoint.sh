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

# Auto-deploy Colony cluster
if [ "${COLONY_AUTO_DEPLOY:-true}" = "true" ]; then
    CONFIG_FLAG=""
    if [ -f "/etc/colony/cluster.yaml" ]; then
        CONFIG_FLAG="--config /etc/colony/cluster.yaml"
    else
        # Wait for config file to be copied in by `colony-env up --config`
        # (docker cp happens after container starts). Timeout after 15s.
        echo "[colony] Waiting for config file..."
        for i in $(seq 1 15); do
            if [ -f "/mnt/shared/config.yaml" ]; then
                CONFIG_FLAG="--config /mnt/shared/config.yaml"
                break
            elif [ -f "/mnt/shared/cluster.yaml" ]; then
                CONFIG_FLAG="--config /mnt/shared/cluster.yaml"
                break
            fi
            sleep 1
        done
        if [ -z "$CONFIG_FLAG" ]; then
            echo "[colony] WARNING: No config file found after 15s. Deploying with defaults."
        else
            echo "[colony] Found config: $CONFIG_FLAG"
        fi
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
