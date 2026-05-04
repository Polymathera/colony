"""Centralized deployment name configuration for polymathera.colony.

Per-deployment names are env-bound on the typed
:class:`DeploymentNames` ``ConfigComponent`` and resolved through the
shared ``ConfigurationManager`` (defaults → operator YAML → env vars).
The legacy ``POLYMATHERA_DEPLOYMENT_NAMES_FILE`` indirection is removed —
operator YAML at ``--config`` is the single override surface for files.

Environment Variables:
    POLYMATHERA_DEPLOYMENT_VCM
    POLYMATHERA_DEPLOYMENT_LLM_CLUSTER
    POLYMATHERA_DEPLOYMENT_AGENT_SYSTEM
    POLYMATHERA_DEPLOYMENT_TOOL_MANAGER
    POLYMATHERA_DEPLOYMENT_STANDALONE_AGENTS
    POLYMATHERA_DEPLOYMENT_SESSION_MANAGER
    POLYMATHERA_DEPLOYMENT_EMBEDDING

Resolution order (lowest → highest precedence):

1. **Pydantic field defaults** declared on :class:`DeploymentNames` below
   (e.g. ``vcm = "vcm"``). Source of truth shipped with colony — no separate
   defaults file.
2. **Operator YAML** loaded by ``ConfigurationManager`` from ``--config``,
   under the ``deployment_names:`` section::

       deployment_names:
         vcm: my_custom_vcm
         llm_cluster: my_custom_cluster

3. **Environment variables** ``POLYMATHERA_DEPLOYMENT_<NAME>`` (e.g.
   ``POLYMATHERA_DEPLOYMENT_VCM``).

Operator YAML is the single file-based override surface.
"""
import logging

from pydantic import ConfigDict, Field

from .distributed.config import (
    ConfigComponent,
    Tier,
    register_polymathera_config,
    tier_metadata,
)

logger = logging.getLogger(__name__)


def _name_field(default: str, env_var: str, description: str) -> Field:
    return Field(
        default=default,
        description=description,
        json_schema_extra={
            "env": env_var, "optional": True,
            **tier_metadata(tier=Tier.L1_OPERATOR),
        },
    )


@register_polymathera_config(path="deployment_names")
class DeploymentNames(ConfigComponent):
    """Deployment names for polymathera.colony components.

    Each field is a standard deployment name; operator YAML overrides via
    ``deployment_names: {<field>: ...}`` and env vars
    ``POLYMATHERA_DEPLOYMENT_<UPPER>`` keep working as before.
    """

    model_config = ConfigDict(frozen=True)  # Make immutable after creation

    vcm: str = _name_field(
        "vcm", "POLYMATHERA_DEPLOYMENT_VCM",
        "Virtual Context Manager deployment name",
    )
    llm_cluster: str = _name_field(
        "llm_cluster", "POLYMATHERA_DEPLOYMENT_LLM_CLUSTER",
        "LLM Cluster deployment name",
    )

    # Agent system
    agent_system: str = _name_field(
        "agent_system", "POLYMATHERA_DEPLOYMENT_AGENT_SYSTEM",
        "Agent System deployment name",
    )
    tool_manager: str = _name_field(
        "tool_manager", "POLYMATHERA_DEPLOYMENT_TOOL_MANAGER",
        "Tool Manager deployment name",
    )
    standalone_agents: str = _name_field(
        "standalone_agents", "POLYMATHERA_DEPLOYMENT_STANDALONE_AGENTS",
        "Standalone Agents deployment name",
    )
    session_manager: str = _name_field(
        "session_manager", "POLYMATHERA_DEPLOYMENT_SESSION_MANAGER",
        "Session Manager deployment name",
    )
    embedding: str = _name_field(
        "embedding", "POLYMATHERA_DEPLOYMENT_EMBEDDING",
        "Embedding service deployment name",
    )


def get_deployment_names() -> DeploymentNames:
    """Return the registered :class:`DeploymentNames` (defaults if uninit)."""
    from .distributed.config import get_component_or_default
    return get_component_or_default("deployment_names", DeploymentNames)


def reset_deployment_names() -> None:
    """No-op preserved for backward compat; resolution now goes through the
    shared ``ConfigurationManager`` cache, not a module-level singleton."""
