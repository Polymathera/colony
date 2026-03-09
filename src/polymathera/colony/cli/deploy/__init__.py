"""Colony local deployment environment (colony-env).

Provides tools to spin up a local Ray cluster + Redis using Docker Compose,
enabling zero-friction testing of the Colony multi-agent framework.

Usage:
    colony-env up          # Build image and start Ray cluster + Redis
    colony-env run PATH    # Run polymath.py analysis inside the cluster
    colony-env status      # Show running services
    colony-env down        # Tear everything down
    colony-env doctor      # Check prerequisites
"""

from .config import DeployConfig, RayDeployConfig, RedisDeployConfig
from .manager import DeploymentManager

__all__ = [
    "DeployConfig",
    "RedisDeployConfig",
    "RayDeployConfig",
    "DeploymentManager",
]
