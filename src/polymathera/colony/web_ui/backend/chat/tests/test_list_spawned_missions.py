"""Tests for ``SessionOrchestratorCapability.list_spawned_missions``.

This action is the READ primitive the SessionAgent's LLM planner
calls before ``spawn_mission`` to answer "is there already a live
coordinator for this kind of work in my scope?". Reads
:class:`MissionExecutionLedger` directly so the answer reflects every
Ray worker in the cluster, not just this process.

The tests pin two behaviours:

- Live ledger entries surface through the action with the right
  shape (``agent_id`` / ``mission_type`` / ``mode`` / ``started_at``).
- The default scope is the SessionAgent's own SESSION bucket — a
  different session's entry must NOT leak in.

Run against an in-memory ledger (mirrors the fixture pattern in
``test_session_orchestrator_missions.py``) so no Polymathera + Redis
is required.
"""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

import pytest
from pydantic import BaseModel as _PydBM

from polymathera.colony.agents.configs import (
    MissionConcurrencyScope,
    MissionExecutionPolicy,
)
from polymathera.colony.agents.scopes import BlackboardScope
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring, execution_context,
)
from polymathera.colony.distributed.state_management import StateManager
from polymathera.colony.distributed.stores.state_base import (
    StateStorageBackend,
    StateStorageBackendFactory,
)
from polymathera.colony.web_ui.backend.chat.session_agent import (
    SessionOrchestratorCapability,
)


# ---------------------------------------------------------------------------
# Test fixtures — execution context + in-memory ledger.
# ---------------------------------------------------------------------------


@pytest.fixture
def _exec_ctx():
    """Execution context the action runs inside — mirrors the chat
    session's user-ring context."""

    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ) as ctx:
        yield ctx


class _Cfg(_PydBM):
    pass


class _InMemBackend(StateStorageBackend):
    """Single-process backend with Redis-style compare-and-swap.

    Each test gets its own backend instance via the ``_ledger`` fixture
    so ledger state never leaks across tests.
    """

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


@pytest.fixture
async def _ledger(monkeypatch):
    """Replace ``get_mission_execution_ledger`` with a fresh in-memory
    ledger per test. Returns the ledger so individual tests can seed
    entries before invoking the action."""

    ledger_mod = importlib.import_module(
        "polymathera.colony.agents.missions.execution_ledger"
    )

    sm = StateManager(
        ledger_mod.MissionLedgerState,
        state_key=ledger_mod.MissionLedgerState.get_state_key("test"),
        config=_Cfg(),
        factory=_InMemFactory(),
    )
    await sm.initialize()
    fake_ledger = ledger_mod.MissionExecutionLedger(state_manager=sm)

    async def _get(app_name=None):
        return fake_ledger

    monkeypatch.setattr(
        ledger_mod, "get_mission_execution_ledger", _get,
    )
    return fake_ledger


# ---------------------------------------------------------------------------
# Fake-agent helpers.
#
# Per [[no-getattr-defaults]]: the test fixtures must use SimpleNamespace
# (NOT bare MagicMock) so attribute access is grounded in the explicitly-
# named fields the production code reads — ``metadata.session_id``,
# ``metadata.parameters``. An auto-MagicMock would silently return
# MagicMock for missing fields and mask the very class of bug
# ``list_spawned_missions`` is meant to make impossible.
# ---------------------------------------------------------------------------


def _make_cap(_exec_ctx) -> SessionOrchestratorCapability:
    """Detached capability — same shape as the sibling
    ``test_session_orchestrator_missions`` fixture."""

    return SessionOrchestratorCapability(
        agent=None,
        scope=BlackboardScope.SESSION,
        namespace=SessionOrchestratorCapability.DEFAULT_NAMESPACE,
        capability_key="orchestrator_test",
        app_name="test_app",
    )


def _attach_fake_agent(
    cap: SessionOrchestratorCapability,
    *,
    session_id: str = "sess_test",
) -> SimpleNamespace:
    """Wire a synthetic agent onto ``cap``.

    The production ``list_spawned_missions`` reads
    ``parent_agent.metadata.session_id`` (via ``resolve_scope_id``);
    the SimpleNamespace exposes the field directly so the test
    grounds in the same attribute access the production code uses."""

    metadata = SimpleNamespace(
        parameters={},
        session_id=session_id,
    )
    agent = SimpleNamespace(
        agent_id="session_agent_xyz",
        metadata=metadata,
        get_capability_by_type=lambda _t: None,
    )
    cap._agent = agent
    return agent


async def _seed_entry(
    ledger,
    *,
    scope_id: str,
    mission_type: str,
    agent_id: str,
    mode: str | None = None,
) -> None:
    """Seed one live mission entry under ``(SESSION, scope_id,
    mission_type)``. Uses the unbounded policy so seeding multiple
    entries in the same bucket doesn't trip the singleton cap."""

    from polymathera.colony.agents.missions.execution_ledger import (
        RunningMissionKey,
    )

    from polymathera.colony.agents.missions.execution_ledger import (
        AdmissionAllowed,
    )

    policy = MissionExecutionPolicy(max_concurrent_instances=None)
    key = RunningMissionKey(
        scope=MissionConcurrencyScope.SESSION,
        scope_id=scope_id,
        mission_type=mission_type,
    )
    decision = await ledger.try_admit(
        key=key, mode=mode, policy=policy,
    )
    assert isinstance(decision, AdmissionAllowed)
    await ledger.register(
        reservation_id=decision.reservation_id,
        agent_id=agent_id, mode=mode,
    )


# ---------------------------------------------------------------------------
# list_spawned_missions — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_spawned_missions_returns_live_entries(
    _exec_ctx, _ledger, monkeypatch,
) -> None:
    """A coordinator registered for ``mission_type=project_planning``
    surfaces through the action with the full
    ``{agent_id, mission_type, mode, started_at}`` shape — the LLM
    branches on a non-empty ``missions`` list to reuse the live
    coordinator instead of spawning a duplicate."""

    cap = _make_cap(_exec_ctx)
    _attach_fake_agent(cap, session_id="sess_test")

    # Stub the mission registry to declare project_planning as
    # SESSION-scoped (matches the production policy).
    from polymathera.colony.agents import mission_registry as mr

    monkeypatch.setattr(mr, "get_mission_registry", lambda: {
        "project_planning": {
            "label": "Project Planning",
            "coordinator_v2": "pkg.mod.ProjectPlanningCoordinator",
            "execution_policy": {
                "max_concurrent_instances": 1,
                "concurrency_scope": "session",
            },
        },
    })

    await _seed_entry(
        _ledger,
        scope_id="sess_test",
        mission_type="project_planning",
        agent_id="coord_live_1",
        mode="decompose",
    )

    result = await cap.list_spawned_missions(
        mission_type="project_planning",
    )

    assert "error" not in result
    assert len(result["missions"]) == 1
    [entry] = result["missions"]
    assert entry["agent_id"] == "coord_live_1"
    assert entry["mission_type"] == "project_planning"
    assert entry["mode"] == "decompose"
    # started_at is a float (set by RunningMissionEntry's
    # default_factory=time.time) — direct attribute access, never
    # getattr default.
    assert isinstance(entry["started_at"], float)


# ---------------------------------------------------------------------------
# list_spawned_missions — scope isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_spawned_missions_scopes_by_session_id(
    _exec_ctx, _ledger, monkeypatch,
) -> None:
    """A coordinator running under a DIFFERENT session's bucket must
    not leak into this session's result. Default scope resolution
    walks ``parent_agent.metadata.session_id`` — no cross-session
    visibility unless the caller explicitly opts in."""

    cap = _make_cap(_exec_ctx)
    _attach_fake_agent(cap, session_id="sess_test")

    from polymathera.colony.agents import mission_registry as mr

    monkeypatch.setattr(mr, "get_mission_registry", lambda: {
        "project_planning": {
            "label": "Project Planning",
            "coordinator_v2": "pkg.mod.ProjectPlanningCoordinator",
            "execution_policy": {
                "max_concurrent_instances": 1,
                "concurrency_scope": "session",
            },
        },
    })

    # Seed one entry under this session and one under a different
    # session — same mission_type, different scope_id.
    await _seed_entry(
        _ledger,
        scope_id="sess_test",
        mission_type="project_planning",
        agent_id="coord_mine",
        mode=None,
    )
    await _seed_entry(
        _ledger,
        scope_id="sess_other",
        mission_type="project_planning",
        agent_id="coord_theirs",
        mode=None,
    )

    result = await cap.list_spawned_missions(
        mission_type="project_planning",
    )

    assert "error" not in result
    agent_ids = {m["agent_id"] for m in result["missions"]}
    assert agent_ids == {"coord_mine"}, (
        "Cross-session leak: list_spawned_missions returned "
        f"{agent_ids} but only coord_mine lives in sess_test."
    )
