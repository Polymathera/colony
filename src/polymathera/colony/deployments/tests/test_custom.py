"""Round-trip test: a stub :class:`CustomDeployment` writes via
``ctx.write_runtime_overlay`` and the values become visible through
``ConfigurationManager.get_component_for``.

Reuses the in-memory state backend from ``distributed/config/tests``.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, Field

from polymathera.colony.deployments import (
    CustomDeploymentsConfig,
    DeploymentContext,
    get_custom_deployment_class,
    register_custom_deployment,
)
from polymathera.colony.distributed.config import (
    ConfigComponent,
    ConfigOverlayState,
    OVERLAY_STATE_KEY,
    OverlayStore,
    Tier,
    register_polymathera_config,
    tier_metadata,
)
from polymathera.colony.distributed.config.manager import ConfigurationManager
from polymathera.colony.distributed.state_management import StateManager
from polymathera.colony.distributed.stores.state_base import (
    StateStorageBackend,
    StateStorageBackendFactory,
)


pytestmark = pytest.mark.asyncio


class _InMemConfig(BaseModel):
    retry_delay: float = 0.0


class _InMemBackend(StateStorageBackend):
    def __init__(self) -> None:
        self._store: dict[str, tuple[str, int]] = {}
        self._lock = asyncio.Lock()

    async def get_with_version(self, key):
        async with self._lock:
            return self._store.get(key, (None, 0))

    async def compare_and_swap(self, key, value, version):
        async with self._lock:
            entry = self._store.get(key)
            current = entry[1] if entry is not None else 0
            if current != version:
                return False
            self._store[key] = (value, version + 1)
            return True

    async def cleanup(self, key):
        async with self._lock:
            self._store.pop(key, None)


class _InMemFactory(StateStorageBackendFactory):
    def __init__(self) -> None:
        self.backend = _InMemBackend()

    def create_backend(self, config):
        return self.backend


@register_polymathera_config(path="custom_deployment_test_endpoints")
class _EndpointsConfig(ConfigComponent):
    """Stand-in for an HPC client config receiving values from a deployment."""

    scheduler_url: str = Field(
        default="",
        json_schema_extra=tier_metadata(tier=Tier.L4_RUNTIME),
    )


@register_custom_deployment("__test_stub_hpc")
class _StubHpc:
    """Stub handler — writes a fake scheduler URL into the L4 overlay."""

    name = "__test_stub_hpc"

    async def provision(self, ctx: DeploymentContext) -> None:
        await ctx.write_runtime_overlay(
            "custom_deployment_test_endpoints",
            {"scheduler_url": "http://stub-hpc:8080"},
        )

    async def query_state(self, ctx: DeploymentContext) -> dict:
        return {"status": "ready"}

    async def tear_down(self, ctx: DeploymentContext) -> None:
        return None


@pytest.fixture
async def cm():
    sm = StateManager(
        state_type=ConfigOverlayState,
        state_key=OVERLAY_STATE_KEY,
        config=_InMemConfig(),
        factory=_InMemFactory(),
    )
    await sm.initialize()
    manager = ConfigurationManager(overlay_store=OverlayStore(sm))
    await manager.initialize()
    return manager


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_registry_lookup_returns_stub_class() -> None:
    cls = get_custom_deployment_class("__test_stub_hpc")
    assert cls is _StubHpc


async def test_provision_writes_runtime_overlay_visible_via_get_component_for(cm) -> None:
    handler = _StubHpc()
    ctx = DeploymentContext(name="hpc_aero", config_manager=cm)

    # Pre-provision: default L1 value.
    pre = await cm.get_component_for("custom_deployment_test_endpoints")
    assert pre.scheduler_url == ""

    await handler.provision(ctx)

    composed = await cm.get_component_for("custom_deployment_test_endpoints")
    assert composed.scheduler_url == "http://stub-hpc:8080"


async def test_custom_deployments_config_default_empty(cm) -> None:
    cfg = cm.get_component("custom_deployments")
    assert isinstance(cfg, CustomDeploymentsConfig)
    assert cfg.deployments == {}


async def test_custom_deployments_config_via_l1_update(cm) -> None:
    cm.update_config({
        "custom_deployments": {
            "deployments": {
                "hpc_aero": {
                    "handler": "__test_stub_hpc",
                    "auto_provision": True,
                    "params": {"region": "us-west-2"},
                },
            },
        },
    })
    cfg = cm.get_component("custom_deployments")
    spec = cfg.deployments["hpc_aero"]
    assert spec.handler == "__test_stub_hpc"
    assert spec.auto_provision is True
    assert spec.params == {"region": "us-west-2"}
