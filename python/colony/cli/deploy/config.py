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
    api_key_env_vars: list[str] = field(
        default_factory=lambda: [
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "HUGGING_FACE_HUB_TOKEN",
        ]
    )
