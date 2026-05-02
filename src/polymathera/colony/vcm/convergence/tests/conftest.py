"""Shared guards + fixtures for the convergence-runtime tests.

Pytest's import-mode=importlib + namespace-package src layout can load
modules under two dotted names; modules with module-level Prometheus
``Counter()`` registration or SQLAlchemy ``Table`` declarations then
fail on the second import. ``src/conftest.py`` patches the registries
in ``pytest_configure``, but that hook fires after descendant
conftests' imports — so we install the same guards module-locally here.

Also provides an in-memory ``StateManager`` and a
``convergence_runtime`` fixture so the runtime can be exercised
without Redis. The fixture mocks the colony-scope blackboard so
``_emit_quiescence`` is a no-op, and tests assert on shared state's
``change_feed`` / ``last_episode`` (or override
``runtime._dispatch_via_blackboard`` to capture dispatches directly).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from prometheus_client import REGISTRY as _PROM_REGISTRY
from pydantic import BaseModel


_orig_register = _PROM_REGISTRY.register


def _safe_register(collector):
    try:
        _orig_register(collector)
    except ValueError:
        pass


_PROM_REGISTRY.register = _safe_register

try:
    from sqlalchemy.sql.schema import Table as _SqlTable

    _orig_new = _SqlTable._new.__func__

    def _tolerant_new(cls, *args, **kw):
        if args:
            name = args[0]
            metadata = args[1] if len(args) > 1 else kw.get("metadata")
            if metadata is not None and name in metadata.tables:
                return metadata.tables[name]
        return _orig_new(cls, *args, **kw)

    _SqlTable._new = classmethod(_tolerant_new)
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------------------
# In-memory state manager (test-only)
# ---------------------------------------------------------------------------


from polymathera.colony.distributed.state_management import StateManager  # noqa: E402
from polymathera.colony.distributed.stores.state_base import (  # noqa: E402
    StateStorageBackend,
    StateStorageBackendFactory,
)


class _InMemoryConfig(BaseModel):
    retry_delay: float = 0.0


class _InMemoryStateBackend(StateStorageBackend):
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


class _InMemoryStateBackendFactory(StateStorageBackendFactory):
    def __init__(self) -> None:
        self.backend = _InMemoryStateBackend()

    def create_backend(self, config):
        return self.backend


@pytest.fixture
async def state_manager():
    """Per-test in-memory ``StateManager`` over ``VirtualPageTableState``
    (which carries the embedded ``ConvergencePersistedState``)."""

    from polymathera.colony.vcm.models import VirtualPageTableState

    sm = StateManager(
        state_type=VirtualPageTableState,
        state_key="test:vcm:convergence",
        config=_InMemoryConfig(),
        factory=_InMemoryStateBackendFactory(),
    )
    await sm.initialize()
    return sm


@pytest.fixture
async def convergence_runtime(state_manager):
    """Initialised :class:`ConvergenceRuntime` backed by the in-memory
    state manager. The colony-scope blackboard handle is mocked so
    ``_emit_quiescence`` is a no-op; tests assert on shared state's
    ``change_feed`` / ``last_episode``, or override
    ``runtime._dispatch_via_blackboard`` to capture dispatches."""

    from polymathera.colony.vcm.convergence import ConvergenceRuntime

    rt = ConvergenceRuntime(state_manager=state_manager, app_name="test_app")
    rt._colony_blackboard = AsyncMock()
    rt._colony_blackboard.write = AsyncMock()
    rt._dispatch_via_blackboard = AsyncMock()
    yield rt
    # No cleanup() — mocked blackboard would error on stop.
