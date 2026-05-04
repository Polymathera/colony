"""Round-trip tests for the StateManager-backed config overlay machinery.

Uses an in-memory ``StateStorageBackend`` so the tests do not require Redis.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, Field

from polymathera.colony.distributed.config import (
    ConfigComponent,
    ConfigOverlayState,
    Mutability,
    OVERLAY_STATE_KEY,
    OverlayScope,
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


# ---------------------------------------------------------------------------
# In-memory state backend
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test-only ConfigComponent with mixed-tier fields
# ---------------------------------------------------------------------------


@register_polymathera_config(path="overlay_test")
class _OverlayTestConfig(ConfigComponent):
    operator_field: str = Field(
        default="op-default",
        json_schema_extra=tier_metadata(tier=Tier.L1_OPERATOR),
    )
    tenant_field: str = Field(
        default="tenant-default",
        json_schema_extra=tier_metadata(
            tier=Tier.L2_TENANT, mutability=Mutability.RELOADABLE,
        ),
    )
    session_field: str = Field(
        default="session-default",
        json_schema_extra=tier_metadata(tier=Tier.L3_SESSION),
    )
    runtime_field: str = Field(
        default="runtime-default",
        json_schema_extra=tier_metadata(tier=Tier.L4_RUNTIME),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def overlay_store():
    sm = StateManager(
        state_type=ConfigOverlayState,
        state_key=OVERLAY_STATE_KEY,
        config=_InMemConfig(),
        factory=_InMemFactory(),
    )
    await sm.initialize()
    return OverlayStore(sm)


@pytest.fixture
async def cm(overlay_store):
    manager = ConfigurationManager(overlay_store=overlay_store)
    await manager.initialize()
    return manager


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_get_component_for_no_overlays_matches_get_component(cm) -> None:
    base = cm.get_component("overlay_test")
    composed = await cm.get_component_for("overlay_test")
    assert composed.model_dump() == base.model_dump()


async def test_tenant_overlay_round_trip(cm) -> None:
    await cm.update_overlay(
        "overlay_test",
        {"tenant_field": "tenant-overridden"},
        scope=OverlayScope.tenant("acme"),
    )
    composed = await cm.get_component_for("overlay_test", tenant_id="acme")
    assert composed.tenant_field == "tenant-overridden"
    assert composed.operator_field == "op-default"
    # Other tenants are unaffected.
    other = await cm.get_component_for("overlay_test", tenant_id="other")
    assert other.tenant_field == "tenant-default"


async def test_session_overlay_overrides_tenant(cm) -> None:
    await cm.update_overlay(
        "overlay_test",
        {"session_field": "tenant-set"},
        scope=OverlayScope.tenant("acme"),
    )
    await cm.update_overlay(
        "overlay_test",
        {"session_field": "session-set"},
        scope=OverlayScope.session("s1"),
    )
    composed = await cm.get_component_for(
        "overlay_test", tenant_id="acme", session_id="s1",
    )
    assert composed.session_field == "session-set"


async def test_runtime_overlay_visible_to_all(cm) -> None:
    await cm.update_overlay(
        "overlay_test",
        {"runtime_field": "rt"},
        scope=OverlayScope.runtime("cdk_stack_42"),
    )
    composed = await cm.get_component_for("overlay_test")
    assert composed.runtime_field == "rt"


async def test_tier_enforcement_rejects_lower_tier_write(cm) -> None:
    # operator_field is L1; tenant scope must not be allowed to write it.
    with pytest.raises(PermissionError):
        await cm.update_overlay(
            "overlay_test",
            {"operator_field": "evil"},
            scope=OverlayScope.tenant("acme"),
        )


async def test_update_overlay_without_store_raises() -> None:
    manager = ConfigurationManager()
    await manager.initialize()
    with pytest.raises(RuntimeError):
        await manager.update_overlay(
            "overlay_test", {"tenant_field": "x"},
            scope=OverlayScope.tenant("acme"),
        )


async def test_unknown_path_raises(cm) -> None:
    with pytest.raises(KeyError):
        await cm.update_overlay(
            "no_such_path", {"x": 1}, scope=OverlayScope.tenant("acme"),
        )
