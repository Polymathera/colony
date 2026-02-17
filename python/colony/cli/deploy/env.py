"""Environment variable builder for colony-env deployment.

Builds the env vars that polymath.py and PolymatheraApp need when running
inside the Docker cluster. Also collects API keys from the host environment
to pass through to containers.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import DeployConfig

# .env file location (same directory as .env.template)
_ENV_FILE = Path(__file__).parent / ".env"


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
    """Collect API keys and other env vars from the host **environment** to pass into containers.

    These are passed via `docker exec -e` when running polymath.py.
    Only includes vars that are actually exported in the current process
    environment.  For keys in a .env file that were not exported, use
    :func:`load_dotenv` instead.
    """
    env = {}
    for key in config.api_key_env_vars:
        val = os.environ.get(key)
        if val:
            env[key] = val
    return env


def load_dotenv(config: DeployConfig) -> dict[str, str]:
    """Read API keys from the deploy/.env file.

    Returns only the keys listed in ``config.api_key_env_vars`` that are
    present in the .env file with a non-empty value.  This allows users
    to put keys in the .env file without needing to ``export`` them.

    Format: one ``KEY=VALUE`` per line.  Lines starting with ``#`` and
    blank lines are skipped.  Quotes around values are stripped.
    """
    if not _ENV_FILE.is_file():
        return {}

    allowed = set(config.api_key_env_vars)
    env: dict[str, str] = {}
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'\"")
        if key in allowed and val:
            env[key] = val
    return env
