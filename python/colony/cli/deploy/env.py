"""Environment variable builder for colony-env deployment.

Builds the env vars that polymath.py and PolymatheraApp need when running
inside the Docker cluster. Also collects API keys from the host environment
to pass through to containers.
"""

from __future__ import annotations

import os

from .config import DeployConfig


def build_container_env(config: DeployConfig) -> dict[str, str]:
    """Build environment variables for containers in the cluster.

    These are set in docker-compose.yml / K8s pod spec. They tell
    ConfigurationManager where Redis is and what environment we're in.
    """
    return {
        "REDIS_HOST": config.redis.host,
        "REDIS_PORT": str(config.redis.port),
        "RAY_CLUSTER_ENVIRONMENT": "true",
        "POLYMATHERA_ENV": "development",
    }


def collect_passthrough_env(config: DeployConfig) -> dict[str, str]:
    """Collect API keys and other env vars from the host to pass into containers.

    These are passed via `docker exec -e` when running polymath.py.
    Only includes vars that are actually set on the host.
    """
    env = {}
    for key in config.api_key_env_vars:
        val = os.environ.get(key)
        if val:
            env[key] = val
    return env
