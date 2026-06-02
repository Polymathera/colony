"""Configuration dataclasses for colony-env deployment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RedisDeployConfig:
    """Redis deployment configuration."""

    host: str = "redis"  # Docker service name
    port: int = 6379
    host_port: int = 6379  # Port exposed to host
    image: str = "redis:7-alpine"
    container_name: str = "colony-redis"


@dataclass
class RayDeployConfig:
    """Ray cluster deployment configuration."""

    gcs_port: int = 6379  # Inside Docker network (no conflict with Redis)
    dashboard_port: int = 8265
    client_port: int = 10001
    head_num_cpus: int = 0  # Coordination only, no tasks on head
    workers: int = 1
    shm_size: str = "4gb"
    image: str = "colony:local"
    head_container_name: str = "colony-ray-head"


@dataclass
class DeployConfig:
    """Top-level deployment configuration."""

    mode: str = "compose"  # "compose" or "k8s"
    redis: RedisDeployConfig = field(default_factory=RedisDeployConfig)
    ray: RayDeployConfig = field(default_factory=RayDeployConfig)
    build: bool = True
    # Keys ``env.py:load_dotenv`` lifts from ``cli/deploy/.env`` into
    # ``os.environ`` so docker-compose's ``${VAR:-}`` placeholders
    # resolve at ``compose up`` time. Keep this list in sync with the
    # ``environment:`` blocks in ``docker/docker-compose.yml`` — a key
    # in compose but missing here silently resolves to empty (opaque
    # 401s downstream); a key here but missing from compose is read
    # into the operator shell but never reaches container runtime.
    api_key_env_vars: list[str] = field(
        default_factory=lambda: [
            # LLM providers
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            # PDF-extractor backends (Mistral OCR, LlamaParse) and
            # HuggingFace-pulled weights / self-hosted backends.
            "MISTRAL_API_KEY",
            "LLAMA_CLOUD_API_KEY",
            "HUGGING_FACE_HUB_TOKEN",
            # Git remote credentials.
            # GitHub: per-tenant App installation. ``GitHubAuthConfig``
            # reads App ID + private key from env (deploy-wide); the
            # per-tenant installation id lives on ``tenants`` (not env).
            # GitLab: still PAT-based until the GitLab integration
            # follows the same App pattern.
            "GITLAB_TOKEN",
            "GITHUB_APP_ID",
            "GITHUB_PRIVATE_KEY_PEM",
            # Same App, OAuth client for user-to-server flow.
            "GITHUB_APP_CLIENT_ID",
            "GITHUB_APP_CLIENT_SECRET",
            # WebSearchCapability / ColonyDocsCapability — Tavily.
            "TAVILY_API_KEY",
            # Slack-relay capability.
            "SLACK_APP_TOKEN",
            "SLACK_BOT_TOKEN",
        ]
    )
