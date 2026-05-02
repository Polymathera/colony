"""Centralized deployment name configuration for polymathera.colony.

This module provides a centralized way to manage deployment names used throughout
the polymathera.colony system. Deployment names can be configured via:
1. Environment variables (highest priority)
2. JSON configuration file
3. Default values (lowest priority)

Usage:
    ```python
    from polymathera.colony.deployment_names import get_deployment_names

    # Get deployment names
    names = get_deployment_names()

    # Use in serving.get_deployment()
    vcm_handle = serving.get_deployment(app_name, names.vcm)
    ```

Environment Variables:
    POLYMATHERA_DEPLOYMENT_NAMES_FILE: Path to JSON config file
    POLYMATHERA_DEPLOYMENT_VCM: VCM deployment name
    POLYMATHERA_DEPLOYMENT_LLM_CLUSTER: LLM cluster deployment name
    POLYMATHERA_DEPLOYMENT_AGENT_SYSTEM: Agent system deployment name
    POLYMATHERA_DEPLOYMENT_TOOL_MANAGER: Tool manager deployment name
    POLYMATHERA_DEPLOYMENT_CONSCIOUSNESS_MANAGER: Consciousness manager deployment name
    POLYMATHERA_DEPLOYMENT_STANDALONE_AGENTS: Standalone agents deployment name
    POLYMATHERA_DEPLOYMENT_SESSION_MANAGER: Session manager deployment name
    POLYMATHERA_DEPLOYMENT_EMBEDDING: Embedding deployment name

Configuration File Format (JSON):
    ```json
    {
        "vcm": "vcm",
        "llm_cluster": "llm_cluster",
        "agent_system": "agent_system",
        "tool_manager": "tool_manager",
        "standalone_agents": "standalone_agents",
        "session_manager": "session_manager",
        "embedding": "embedding"
    }
    ```
"""
import json
import logging
import os
from typing import Any
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DeploymentNames(BaseModel):
    """Deployment names for polymathera.colony components.

    Each field represents a standard deployment name used throughout the system.
    These can be overridden via environment variables or JSON config file.
    """

    # Core infrastructure
    vcm: str = Field(
        default="vcm",
        description="Virtual Context Manager deployment name"
    )
    llm_cluster: str = Field(
        default="llm_cluster",
        description="LLM Cluster deployment name"
    )

    # Agent system
    agent_system: str = Field(
        default="agent_system",
        description="Agent System deployment name"
    )
    tool_manager: str = Field(
        default="tool_manager",
        description="Tool Manager deployment name"
    )
    standalone_agents: str = Field(
        default="standalone_agents",
        description="Standalone Agents deployment name"
    )
    session_manager: str = Field(
        default="session_manager",
        description="Session Manager deployment name"
    )

    # Additional services
    embedding: str = Field(
        default="embedding",
        description="Embedding service deployment name"
    )

    class Config:
        frozen = True  # Make immutable after creation


# Singleton instance
_deployment_names: DeploymentNames | None = None


def load_deployment_names_from_env() -> dict[str, Any]:
    """Load deployment names from environment variables.

    Returns:
        Dictionary of deployment names from environment variables
    """
    env_mapping = {
        "vcm": "POLYMATHERA_DEPLOYMENT_VCM",
        "llm_cluster": "POLYMATHERA_DEPLOYMENT_LLM_CLUSTER",
        "agent_system": "POLYMATHERA_DEPLOYMENT_AGENT_SYSTEM",
        "tool_manager": "POLYMATHERA_DEPLOYMENT_TOOL_MANAGER",
        "standalone_agents": "POLYMATHERA_DEPLOYMENT_STANDALONE_AGENTS",
        "session_manager": "POLYMATHERA_DEPLOYMENT_SESSION_MANAGER",
        "embedding": "POLYMATHERA_DEPLOYMENT_EMBEDDING",
    }

    config = {}
    for field_name, env_var in env_mapping.items():
        value = os.environ.get(env_var)
        if value:
            config[field_name] = value

    return config


def load_deployment_names_from_file(file_path: str) -> dict[str, Any]:
    """Load deployment names from JSON configuration file.

    Args:
        file_path: Path to JSON configuration file

    Returns:
        Dictionary of deployment names from file

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(file_path, "r") as f:
        config = json.load(f)

    return config


def get_deployment_names() -> DeploymentNames:
    """Get deployment names configuration.

    This function returns a singleton DeploymentNames instance loaded from:
    1. Environment variables (highest priority)
    2. JSON configuration file (if POLYMATHERA_DEPLOYMENT_NAMES_FILE is set)
    3. Default values (lowest priority)

    The instance is cached after first load for efficiency.

    Returns:
        DeploymentNames instance with configured names

    Example:
        ```python
        from polymathera.colony.deployment_names import get_deployment_names
        import polymathera.colony.distributed.ray_utils.serving as serving

        names = get_deployment_names()
        vcm_handle = serving.get_deployment(app_name, names.vcm)
        llm_cluster_handle = serving.get_deployment(app_name, names.llm_cluster)
        ```
    """
    global _deployment_names

    if _deployment_names is not None:
        return _deployment_names

    # Start with defaults
    config = {}

    # Try to load from file if specified
    config_file = os.environ.get("POLYMATHERA_DEPLOYMENT_NAMES_FILE")
    if not config_file:
        this_path = Path(__file__).parent.resolve()
        config_file = this_path / "deployment_names.json"

    if config_file and Path(config_file).is_file():
        try:
            file_config = load_deployment_names_from_file(config_file)
            config.update(file_config)
            logger.info(f"Loaded deployment names from file: {config_file}")
        except FileNotFoundError:
            logger.warning(f"Deployment names file not found: {config_file}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in deployment names file {config_file}: {e}")

    # Override with environment variables (highest priority)
    env_config = load_deployment_names_from_env()
    if env_config:
        config.update(env_config)
        logger.debug(f"Loaded {len(env_config)} deployment names from environment")

    # Create and cache instance
    _deployment_names = DeploymentNames(**config)

    logger.info(f"Deployment names configuration loaded: {_deployment_names.model_dump()}")

    return _deployment_names


def reset_deployment_names() -> None:
    """Reset the deployment names singleton (primarily for testing).

    This forces the next call to get_deployment_names() to reload from
    environment variables and configuration files.
    """
    global _deployment_names
    _deployment_names = None
