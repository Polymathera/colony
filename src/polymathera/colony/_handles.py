"""Deployment-handle accessors — extracted from ``colony/system.py`` to
break a real circular import cycle.

Why this module exists
----------------------

``colony/system.py`` top-level imports a heavy chain
(``cluster.config``, ``vcm.config``, ``agents.config``,
``agents.blueprint``, ``knowledge.cluster_config``). When 8+
deployments fire their ``@on_app_ready`` hooks against the same
``StandaloneAgentDeployment`` actor process and the first one happens
to be loading ``colony.system`` for the first time, a *second* hook
hits ``from ..system import get_llm_cluster`` while ``system.py`` is
mid-load, getting ``ImportError: cannot import name '...' from
partially initialized module``. The actor never recovers — every
later inference fails with "LLM cluster handle not initialized".

The accessors themselves don't need the heavy ``system.py`` graph;
each is a small wrapper around ``serving.get_deployment`` plus a
lazy import of its target deployment class. Pulled out here so
``agents/base.py`` (and any other framework code in the deployment
lifecycle path) can import them without dragging in the cycle.

``system.py`` re-exports these so existing callers keep working
without churn.
"""

from __future__ import annotations

import logging
from typing import Any, Type

from .distributed.ray_utils import serving
from .deployment_names import get_deployment_names

logger = logging.getLogger(__name__)


async def _get_deployment_by_name(
    name_attr: str,
    app_name: str | None = None,
    deployment_class: Type[Any] | None = None,
) -> serving.DeploymentHandle:
    """Resolve a deployment handle by its registered name.

    Shared body for the ``get_<role>`` accessors below. ``name_attr``
    is the attribute on ``DeploymentNames`` (e.g. ``"llm_cluster"``,
    ``"agent_system"``); ``deployment_class`` lets the serving layer
    type-check the returned handle.
    """

    try:
        names = await get_deployment_names()
        handle = serving.get_deployment(
            app_name or serving.get_my_app_name(),
            getattr(names, name_attr),
            deployment_class=deployment_class,
        )
        logger.debug(f"Connected to {name_attr} deployment: {getattr(names, name_attr)}")
        return handle
    except Exception as e:
        logger.error(f"{name_attr} deployment not found: {e}")
        raise


async def get_agent_system(app_name: str | None = None) -> serving.DeploymentHandle:
    """Handle to the ``AgentSystemDeployment`` (registry + lifecycle)."""

    from .agents.system import AgentSystemDeployment
    return await _get_deployment_by_name(
        "agent_system", app_name, deployment_class=AgentSystemDeployment,
    )


async def fetch_agent_info(
    agent_id: str,
    *,
    app_name: str | None = None,
) -> "AgentRegistrationInfo | None":  # noqa: F821 — fwd ref, lazy import below
    """Canonical lookup: ``AgentRegistrationInfo`` for ``agent_id`` from the
    :class:`AgentSystemDeployment` registry, or ``None`` when no record
    exists (agent terminated and reaped, or never existed).

    Single source of truth for "what is this agent's lifecycle state /
    type / capabilities right now?" — used by every call site that
    needs to read agent state from the registry.

    The returned :class:`AgentRegistrationInfo` is a fully typed
    Pydantic model. Callers MUST read its fields directly
    (``info.agent_type``, ``info.state``, ``info.capability_names``).
    NEVER use ``getattr(info, "...", default)`` — every field on the
    model is guaranteed to exist by construction, and a missing
    attribute would surface a real bug rather than be silently
    masked by a default per [[no-getattr-defaults]].

    Exceptions from the underlying registry call PROPAGATE. Callers
    that need a typed degraded fallback (e.g. "show state=UNKNOWN
    when the registry is unreachable") must catch + handle at their
    call site; the helper does not silently swallow.
    """

    from .agents.system import AgentSystemDeployment  # noqa: F401 — fwd-resolve

    agent_system = await get_agent_system(app_name=app_name)
    return await agent_system.get_agent_info(agent_id)


async def get_llm_cluster(app_name: str | None = None) -> serving.DeploymentHandle:
    """Handle to the ``LLMCluster`` deployment (routing front for LLMs)."""

    from .cluster.cluster import LLMCluster
    return await _get_deployment_by_name(
        "llm_cluster", app_name, deployment_class=LLMCluster,
    )


async def get_vcm(app_name: str | None = None) -> serving.DeploymentHandle:
    """Handle to the ``VirtualContextManager`` deployment."""

    from .vcm.manager import VirtualContextManager
    return await _get_deployment_by_name(
        "vcm", app_name, deployment_class=VirtualContextManager,
    )


async def get_standalone_agents(app_name: str | None = None) -> serving.DeploymentHandle:
    """Handle to the ``StandaloneAgentDeployment``."""

    from .agents.standalone import StandaloneAgentDeployment
    return await _get_deployment_by_name(
        "standalone_agents", app_name, deployment_class=StandaloneAgentDeployment,
    )


async def get_session_manager(app_name: str | None = None) -> serving.DeploymentHandle:
    """Handle to the ``SessionManagerDeployment``."""

    from .agents.sessions import SessionManagerDeployment
    return await _get_deployment_by_name(
        "session_manager", app_name, deployment_class=SessionManagerDeployment,
    )


async def get_embedding_deployment(app_name: str | None = None) -> serving.DeploymentHandle:
    """Handle to the embedding deployment (name from ``DeploymentNames.embedding``)."""

    return await _get_deployment_by_name("embedding", app_name)


def get_vllm_deployment(
    deployment_name: str, app_name: str | None = None,
) -> serving.DeploymentHandle:
    """Handle to a specific VLLM deployment by name.

    Sync (no ``await``) — unlike the per-role accessors, the caller
    already knows the deployment name and we skip the
    ``DeploymentNames`` round-trip.
    """

    from .cluster.vllm_deployment import VLLMDeployment

    try:
        handle = serving.get_deployment(
            app_name or serving.get_my_app_name(),
            deployment_name,
            deployment_class=VLLMDeployment,
        )
        logger.info(f"Connected to VLLM deployment: {deployment_name}")
        return handle
    except Exception as e:
        logger.error(f"VLLM deployment '{deployment_name}' not found: {e}")
        raise


def get_remote_llm_deployment(
    deployment_name: str, app_name: str | None = None,
) -> serving.DeploymentHandle:
    """Handle to a specific remote-LLM deployment (Anthropic, OpenRouter, ...) by name."""

    from .cluster.remote_deployment import RemoteLLMDeployment

    try:
        handle = serving.get_deployment(
            app_name or serving.get_my_app_name(),
            deployment_name,
            deployment_class=RemoteLLMDeployment,
        )
        logger.info(f"Connected to remote LLM deployment: {deployment_name}")
        return handle
    except Exception as e:
        logger.error(f"Remote LLM deployment '{deployment_name}' not found: {e}")
        raise
