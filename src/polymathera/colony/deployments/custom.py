"""Custom deployment protocol — Colony's plug-in surface for externally-
managed resources (HPC stacks, AWS-CDK, Slurm clusters, ...).

Colony defines the contract; extensions (e.g. ``polymathera-cps``) implement
concrete handlers. A handler is registered by name; operator YAML refers to
it via ``custom_deployments.<instance_name>.handler = "<registered_name>"``.

Lifecycle: :meth:`CustomDeployment.provision` brings the resource up and
returns a dict of runtime values (endpoints, credentials, ...). The handler
writes those values into the L4 ``runtime_overlays`` slot via
:meth:`DeploymentContext.write_runtime_overlay`, after which any other
component's ``cm.get_component_for(...)`` observes them automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..distributed.config import OverlayScope
from ..distributed.config.manager import ConfigurationManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeploymentContext:
    """Runtime context handed to a :class:`CustomDeployment` handler.

    ``name`` is the deployment-instance key from the operator YAML — it doubles
    as the L4 overlay scope key. ``tenant_id`` is set when the lifecycle is
    invoked on behalf of a tenant; left ``None`` for cluster-wide deployments.
    """

    name: str
    config_manager: ConfigurationManager
    tenant_id: str | None = None

    async def write_runtime_overlay(
        self, component_path: str, updates: dict[str, Any],
    ) -> None:
        """Write runtime values into L4 under this deployment's scope.

        Tier-checked by :meth:`ConfigurationManager.update_overlay` — fields
        whose declared tier is below ``L4_RUNTIME`` are rejected.
        """
        await self.config_manager.update_overlay(
            component_path, updates, scope=OverlayScope.runtime(self.name),
        )


@runtime_checkable
class CustomDeployment(Protocol):
    """Externally-managed resource Colony can provision / query / tear down."""

    name: str

    async def provision(self, ctx: DeploymentContext) -> None:
        """Bring the resource up and write any runtime values via ``ctx``."""
        ...

    async def query_state(self, ctx: DeploymentContext) -> dict[str, Any]:
        """Cheap, idempotent state probe used by health checks."""
        ...

    async def tear_down(self, ctx: DeploymentContext) -> None:
        """Best-effort teardown."""
        ...


# In-process registry. Last writer wins (with a warning) so a CPS extension
# can intentionally override a colony-shipped default; tests can re-register
# across sessions without raising.
_REGISTRY: dict[str, type[CustomDeployment]] = {}


def register_custom_deployment(name: str):
    """Decorator: register a handler class under ``name``."""

    def _decorator(cls: type[CustomDeployment]) -> type[CustomDeployment]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            logger.warning(
                "register_custom_deployment: %r already registered to %s; "
                "overwriting with %s",
                name, _REGISTRY[name].__name__, cls.__name__,
            )
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_custom_deployment_class(name: str) -> type[CustomDeployment] | None:
    return _REGISTRY.get(name)


def list_custom_deployments() -> list[str]:
    return sorted(_REGISTRY.keys())


__all__ = (
    "CustomDeployment",
    "DeploymentContext",
    "get_custom_deployment_class",
    "list_custom_deployments",
    "register_custom_deployment",
)
