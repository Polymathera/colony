"""Tests for ``SessionOrchestratorCapability._refresh_available_missions``.

Stage A (cps/STAGE_A_L1A_DYNAMIC_DISCOVERY_PLAN.md): the SessionAgent's
LLM planner reads ``metadata.parameters["available_missions"]`` to
decide which mission to spawn for a user's chat request. Without the
refresh implemented here, that dict is the static snapshot built at
session-create time from ``get_mission_registry()`` alone (entry-point
group + hardcoded builtins) — so missions authored under an L4 design
monorepo's ``.colony/missions/`` are invisible to the planner.

The refresh:

- Pulls fresh entries from
  ``RepoStateProvider.discovered_extensions.missions`` on every call.
- Merges them with ``get_mission_registry()`` (last-write-wins on
  collision, matching the convention for entry-point shadowing of
  builtins).
- Projects to the four-field shape
  ``sessions.py:create_session`` produces.
- Mutates ``self._agent.metadata.parameters["available_missions"]``
  in place.

These tests pin each rule. They run against a detached capability
with a synthetic agent + a stand-in for ``RepoStateProvider`` that
exposes a controllable ``discovered_extensions.missions`` dict.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.scopes import BlackboardScope
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring, execution_context,
)
from polymathera.colony.web_ui.backend.chat.session_agent import (
    SessionOrchestratorCapability,
)


@pytest.fixture
def _exec_ctx():
    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ) as ctx:
        yield ctx


# ---------------------------------------------------------------------------
# Mission spawn-gate fixtures.
#
# ``spawn_mission`` routes through ``admit_and_spawn``, which fetches
# the cluster-shared ledger via ``get_mission_execution_ledger``. In
# tests we don't want a live Polymathera + Redis; route the helper to
# an in-memory ledger so every test gets a deterministic, isolated
# spawn-gate view. The fixture is autouse so it applies to every test
# in this module — no test should observe ledger state leaking from
# its predecessor.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mission_ledger_stub(monkeypatch):
    """Replace the cluster-shared ledger with a fresh in-memory ledger
    per test. Returns the ledger so individual tests can inspect or
    pre-seed it.

    Also sets ``POLYMATHERA_SERVING_CURRENT_APP`` so
    ``admit_and_spawn`` 's call to ``serving.get_my_app_name()``
    succeeds — production runs inside a deployment context that
    sets this env var; tests must mirror that contract explicitly
    rather than letting the helper silently fall back to a default.
    """

    monkeypatch.setenv("POLYMATHERA_SERVING_CURRENT_APP", "test_app")

    import asyncio
    import importlib

    from pydantic import BaseModel as _PydBM

    ledger_mod = importlib.import_module(
        "polymathera.colony.agents.missions.execution_ledger"
    )
    from polymathera.colony.distributed.state_management import StateManager
    from polymathera.colony.distributed.stores.state_base import (
        StateStorageBackend,
        StateStorageBackendFactory,
    )

    class _Cfg(_PydBM):
        pass

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

    sm = StateManager(
        ledger_mod.MissionLedgerState,
        state_key=ledger_mod.MissionLedgerState.get_state_key("test"),
        config=_Cfg(),
        factory=_InMemFactory(),
    )

    # Lazy-initialise the backend on first call so the fixture
    # doesn't have to be ``async``. The first ``await
    # ledger.try_admit`` from a test triggers ``sm.initialize()``
    # via a small wrapper.
    initialised = False

    async def _ensure_initialised():
        nonlocal initialised
        if not initialised:
            await sm.initialize()
            initialised = True

    fake_ledger = ledger_mod.MissionExecutionLedger(state_manager=sm)

    # Wrap the ledger methods to ensure initialise-on-first-use so
    # we don't push the async setup onto every test.
    real_methods = {
        name: getattr(fake_ledger, name) for name in (
            "try_admit", "register", "release_reservation",
            "unregister", "snapshot",
        )
    }

    async def _wrap(name, *args, **kwargs):
        await _ensure_initialised()
        return await real_methods[name](*args, **kwargs)

    for name in real_methods:
        setattr(
            fake_ledger, name,
            (lambda nm: (
                lambda *a, **kw: _wrap(nm, *a, **kw)
            ))(name),
        )

    async def _get(app_name=None):
        return fake_ledger

    monkeypatch.setattr(
        ledger_mod, "get_mission_execution_ledger", _get,
    )
    return fake_ledger


class _SyntheticCoordinator:
    """Stand-in coordinator class for spawn-gate integration tests.

    Declares a :class:`MissionExecutionPolicy` with a one-instance
    cap + ``return_existing`` so a second spawn against the same
    session re-binds to the running coordinator regardless of mode.
    """

    from polymathera.colony.agents.configs import (
        MissionConcurrencyScope as _Scope,
        MissionExecutionPolicy as _Policy,
    )
    MISSION_EXECUTION_POLICY = _Policy(
        max_concurrent_instances=1,
        concurrency_scope=_Scope.SESSION,
        on_concurrency_violation="return_existing",
    )


def _make_pool_stub(*, created_agent_id: str = "child_xyz"):
    """Fake :class:`AgentPoolCapability` exposing the surfaces
    ``admit_and_spawn`` consults: ``resolve_agent_class`` and
    ``create_agent``.

    ``resolve_agent_class`` is mocked rather than letting MagicMock
    auto-create it — otherwise the auto-created stub returns another
    MagicMock instead of the real :class:`_SyntheticCoordinator`,
    and the spawn-gate falls back to the default
    :class:`MissionExecutionPolicy` (singleton per AGENT, no chains)
    instead of the test's intended policy.
    """

    pool = MagicMock()
    pool.resolve_agent_class = MagicMock(return_value=_SyntheticCoordinator)

    async def _create(**_kwargs):
        return {
            "agent_id": created_agent_id,
            "label": None,
            "created": True,
        }
    pool.create_agent = _create
    return pool


def _make_cap(_exec_ctx):
    """Build a detached SessionOrchestratorCapability — same shape the
    sibling ``test_human_approval_relay`` fixture uses."""

    cap = SessionOrchestratorCapability(
        agent=None,
        scope=BlackboardScope.SESSION,
        namespace=SessionOrchestratorCapability.DEFAULT_NAMESPACE,
        capability_key="orchestrator_test",
        app_name="test_app",
    )
    return cap


def _attach_fake_agent(
    cap: SessionOrchestratorCapability,
    *,
    design_monorepo_url: str | None = None,
    discovered_missions: dict | None = None,
    materialise_succeeds: bool = True,
) -> SimpleNamespace:
    """Wire a synthetic agent + RepoStateProvider onto ``cap``.

    ``discovered_missions``: when None, the capability behaves as if
    no L4 monorepo is mounted (no provider). When a dict, a fake
    provider returns those missions from ``discovered_extensions``.

    ``materialise_succeeds``: when False, simulate the lazy-clone
    failure path by making the fake provider's
    :meth:`ensure_materialized` return False (the public
    explicit-intent API; was previously a side-effect on
    ``current_branch``).
    """

    metadata = SimpleNamespace(parameters={})
    if design_monorepo_url:
        metadata.parameters["design_monorepo_url"] = design_monorepo_url

    if discovered_missions is None:
        agent = SimpleNamespace(
            agent_id="session_agent_xyz",
            metadata=metadata,
            get_capability_by_type=lambda _t: None,
        )
    else:
        provider = MagicMock()
        provider.ensure_materialized = MagicMock(return_value=materialise_succeeds)
        if materialise_succeeds:
            provider.discovered_extensions = SimpleNamespace(
                missions=discovered_missions,
            )
        else:
            # Failure mode: ensure_materialized returns False; the
            # helper still calls ``discovered_extensions`` (the
            # working_dir may be authoritative pre-seeded state),
            # so populate it with empty data so the test reflects
            # the "URL set, clone failed, nothing materialised"
            # scenario rather than "URL set, clone failed, but
            # somehow the disk has data".
            provider.discovered_extensions = SimpleNamespace(missions={})

        from polymathera.colony.design_monorepo import RepoStateProvider

        def _gcbt(t):
            return provider if t is RepoStateProvider else None

        agent = SimpleNamespace(
            agent_id="session_agent_xyz",
            metadata=metadata,
            get_capability_by_type=_gcbt,
        )
    cap._agent = agent
    return agent


# ---------------------------------------------------------------------------
# No design monorepo → static-only behaviour preserved
# ---------------------------------------------------------------------------


def test_no_design_monorepo_falls_back_to_entry_point_registry(
    _exec_ctx, monkeypatch,
) -> None:
    """When the agent has no design_monorepo_url configured, the
    refresh produces the entry-point + builtin registry with no L4
    additions — preserves the legacy session-create-time behaviour."""

    cap = _make_cap(_exec_ctx)
    agent = _attach_fake_agent(cap)

    # Stub the upstream registry so the test is independent of the
    # actual installed mission entry-points.
    from polymathera.colony.agents import mission_registry as mr

    monkeypatch.setattr(mr, "get_mission_registry", lambda: {
        "stub_builtin": {
            "label": "Stub Builtin",
            "description": "(stub)",
            "coordinator_v2": "pkg.mod.StubCoordinator",
            "worker": "pkg.mod.StubWorker",
        },
    })

    cap._refresh_available_missions()

    avail = agent.metadata.parameters["available_missions"]
    assert set(avail) == {"stub_builtin"}
    assert avail["stub_builtin"]["coordinator_class"] == "pkg.mod.StubCoordinator"
    assert avail["stub_builtin"]["worker_class"] == "pkg.mod.StubWorker"


def test_detached_capability_is_no_op(_exec_ctx) -> None:
    """A capability with ``self._agent is None`` (detached mode) must
    not raise — the refresh quietly skips."""

    cap = _make_cap(_exec_ctx)
    cap._agent = None
    # Should not raise.
    cap._refresh_available_missions()


# ---------------------------------------------------------------------------
# L4 missions merge in
# ---------------------------------------------------------------------------


def test_l4_missions_merge_into_registry(_exec_ctx, monkeypatch) -> None:
    """With a design_monorepo_url and an L4 mission discovered, the
    refresh merges the L4 entry into the registry alongside the
    builtins, projected to the four-field planner shape."""

    cap = _make_cap(_exec_ctx)
    agent = _attach_fake_agent(
        cap,
        design_monorepo_url="https://github.com/example/monorepo.git",
        discovered_missions={
            "opm_meg": {
                "label": "OPM-MEG Noise-Floor Design Analysis",
                "description": "L4 mission shipped under .colony/missions/",
                "coordinator_v2": "opm_meg_coordinator.OPMMEGCoordinator",
                "worker": (
                    "polymathera.cps.domains.technical.quantum_magnetometry"
                    ".agents.atomic_physics.AtomicPhysicsAgent"
                ),
            },
        },
    )
    from polymathera.colony.agents import mission_registry as mr

    monkeypatch.setattr(mr, "get_mission_registry", lambda: {
        "stub_builtin": {
            "label": "Stub Builtin",
            "description": "",
            "coordinator_v2": "pkg.mod.StubCoordinator",
            "worker": "pkg.mod.StubWorker",
        },
    })

    cap._refresh_available_missions()

    avail = agent.metadata.parameters["available_missions"]
    assert set(avail) == {"stub_builtin", "opm_meg"}
    assert avail["opm_meg"]["coordinator_class"].endswith("OPMMEGCoordinator")
    assert avail["opm_meg"]["label"] == "OPM-MEG Noise-Floor Design Analysis"


def test_l4_mission_shadows_colony_builtin_on_collision(
    _exec_ctx, monkeypatch, caplog,
) -> None:
    """When an L4 mission key collides with a colony-builtin / entry-
    point key, the L4 entry wins (last-write-wins) and the shadow is
    logged at warning — mirrors the convention in
    ``mission_registry.get_mission_registry``'s plugin-shadow
    behaviour."""

    cap = _make_cap(_exec_ctx)
    agent = _attach_fake_agent(
        cap,
        design_monorepo_url="https://github.com/example/monorepo.git",
        discovered_missions={
            "shared_key": {
                "label": "L4 Override",
                "coordinator_v2": "l4.OverrideCoordinator",
                "worker": "l4.OverrideWorker",
            },
        },
    )
    from polymathera.colony.agents import mission_registry as mr

    monkeypatch.setattr(mr, "get_mission_registry", lambda: {
        "shared_key": {
            "label": "Builtin",
            "coordinator_v2": "builtin.Coordinator",
            "worker": "builtin.Worker",
        },
    })

    with caplog.at_level("WARNING"):
        cap._refresh_available_missions()

    avail = agent.metadata.parameters["available_missions"]
    assert avail["shared_key"]["coordinator_class"] == "l4.OverrideCoordinator"
    assert any(
        "shadows a colony-builtin" in rec.message
        for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# Failure modes — must not break the planner
# ---------------------------------------------------------------------------


def test_clone_failure_falls_back_to_entry_points_only(
    _exec_ctx, monkeypatch,
) -> None:
    """If :meth:`RepoStateProvider.ensure_materialized` returns
    ``False`` (URL unreachable, auth failure, etc. — failures are
    logged inside ``ensure_materialized`` itself), the refresh
    surfaces only the entry-points / builtin registry. The user
    sees the underlying clone failure in container logs from
    ``ensure_materialized``'s WARNING, not from a duplicate log
    here."""

    cap = _make_cap(_exec_ctx)
    agent = _attach_fake_agent(
        cap,
        design_monorepo_url="https://github.com/example/unreachable.git",
        discovered_missions={"never_reached": {"label": "x"}},
        materialise_succeeds=False,
    )
    from polymathera.colony.agents import mission_registry as mr

    monkeypatch.setattr(mr, "get_mission_registry", lambda: {
        "stub_builtin": {
            "label": "Stub Builtin",
            "coordinator_v2": "pkg.mod.StubCoordinator",
            "worker": "pkg.mod.StubWorker",
        },
    })

    cap._refresh_available_missions()

    avail = agent.metadata.parameters["available_missions"]
    assert set(avail) == {"stub_builtin"}
    # The fake provider's ``ensure_materialized`` was called — the
    # public materialise API is the explicit-intent contract
    # ``get_l4_extensions`` depends on.
    from polymathera.colony.design_monorepo import RepoStateProvider
    provider = agent.get_capability_by_type(RepoStateProvider)
    provider.ensure_materialized.assert_called_once()


# ---------------------------------------------------------------------------
# spawn_mission action — the mission-aware dispatch that wraps
# AgentPoolCapability.create_agent so the LLM does not need to extract
# coordinator_class from the available_missions dict literal.
# ---------------------------------------------------------------------------


async def _run_spawn(cap, **kwargs):
    """spawn_mission is @action_executor-decorated, but the decorator
    just attaches metadata attributes and returns the original
    function — so we can call it directly on the bound instance."""
    return await cap.spawn_mission(**kwargs)


@pytest.mark.asyncio
async def test_spawn_mission_unknown_type_returns_error(
    _exec_ctx, monkeypatch,
) -> None:
    """Unknown ``mission_type`` returns a failure dict with an
    ``error`` naming the available keys — the LLM can branch on
    ``created=False`` and surface a useful message."""

    cap = _make_cap(_exec_ctx)
    _attach_fake_agent(cap, discovered_missions={
        "opm_meg": {"label": "x", "coordinator_v2": "p.Q"},
    }, design_monorepo_url="x")

    from polymathera.colony.agents import mission_registry as mr
    monkeypatch.setattr(mr, "get_mission_registry", lambda: {})

    result = await _run_spawn(cap, mission_type="never_registered")
    assert result["outcome"] == "error"
    assert result["created"] is False
    assert result["mission_gate"] is None
    assert result["agent_id"] is None
    assert "Unknown mission type" in result["error"]
    assert "opm_meg" in result["error"]


@pytest.mark.asyncio
async def test_spawn_mission_missing_coordinator_returns_error(
    _exec_ctx, monkeypatch,
) -> None:
    """A mission spec with neither coordinator_v1 nor coordinator_v2
    is unbootable — return a clean failure rather than crashing
    later in create_agent's resolver."""

    cap = _make_cap(_exec_ctx)
    _attach_fake_agent(cap, discovered_missions={
        "shapeless": {"label": "Shapeless mission"},
    }, design_monorepo_url="x")

    from polymathera.colony.agents import mission_registry as mr
    monkeypatch.setattr(mr, "get_mission_registry", lambda: {})

    result = await _run_spawn(cap, mission_type="shapeless")
    assert result["outcome"] == "error"
    assert result["created"] is False
    assert "no coordinator_v2" in result["error"]


@pytest.mark.asyncio
async def test_spawn_mission_dispatches_to_create_agent(
    _exec_ctx, monkeypatch,
) -> None:
    """Happy path: a registered mission resolves its
    ``coordinator_v2`` class path, builds metadata from
    ``self_concept`` + ``mission_params``, and dispatches to
    AgentPoolCapability.create_agent. Returns the spawned agent_id."""

    from polymathera.colony.agents.patterns.capabilities.agent_pool import (
        AgentPoolCapability,
    )

    cap = _make_cap(_exec_ctx)
    # Wire the SessionOrchestratorCapability's agent to expose a fake
    # AgentPoolCapability whose create_agent we observe.
    fake_pool = MagicMock()
    fake_pool.create_agent = MagicMock(return_value={
        "agent_id": "child_abc123",
        "label": None,
        "created": True,
    })

    async def _async_create(**kwargs):
        return fake_pool.create_agent(**kwargs)
    fake_pool.create_agent = _async_create
    # Track calls separately.
    captured: dict[str, dict] = {}

    async def _capture_create(**kwargs):
        captured["kwargs"] = kwargs
        return {"agent_id": "child_abc123", "label": None, "created": True}
    fake_pool.create_agent = _capture_create

    # The session has colony-scoped params on its metadata, but
    # spawn_mission deliberately does NOT thread them onto the
    # child's metadata — the central inheritance gate in
    # ``AgentPoolCapability.create_agent`` does that. This test
    # asserts spawn_mission's narrow contract; the inheritance
    # behaviour is pinned separately by
    # ``test_create_agent_inherits_colony_scoped_params`` (and
    # siblings) in ``tests/test_agent_pool.py``.
    metadata = SimpleNamespace(parameters={
        "design_monorepo_url": "https://github.com/acme/monorepo.git",
        "git_attribution": {
            "commit_principal": "colony", "commit_co_author": "user",
        },
        "github_identity": {
            "tenant_installation_id": "100",
            "user_github_login": "alice",
        },
        "available_missions": {},
        "available_tools": {"foo": "bar"},
    }, session_id="sess_test")
    cap._agent = SimpleNamespace(
        agent_id="session_agent_xyz",
        metadata=metadata,
        get_capability_by_type=lambda t: fake_pool if t is AgentPoolCapability else None,
    )

    from polymathera.colony.agents import mission_registry as mr
    monkeypatch.setattr(mr, "get_mission_registry", lambda: {
        "opm_meg": {
            "label": "OPM-MEG Noise-Floor",
            "description": "Spin up the OPM-MEG coordinator.",
            "coordinator_v2": "opm_meg_coordinator.OPMMEGCoordinator",
            "worker": "polymathera.cps.x.AtomicPhysicsAgent",
            # Real registry entries are ``MissionSelfConcept``-shaped
            # (extra="forbid", description/goals/constraints only).
            # ``spawn_mission`` is responsible for stamping the
            # ``agent_id`` and ``name`` that ``AgentSelfConcept`` requires.
            "self_concept": {
                "description": "Drives the OPM-MEG mission.",
                "goals": ["close the noise budget"],
                "constraints": ["cite every prediction"],
            },
        },
    })

    result = await _run_spawn(
        cap,
        mission_type="opm_meg",
        mission_params={"noise_floor_target_fT_rt_hz": 12.0},
    )

    assert result["created"] is True
    assert result["agent_id"] == "child_abc123"
    assert result["mission_type"] == "opm_meg"
    assert result["coordinator_class"] == "opm_meg_coordinator.OPMMEGCoordinator"
    assert result["label"] == "OPM-MEG Noise-Floor"

    # Verify the dispatch carried the right contract.
    sent = captured["kwargs"]
    assert sent["agent_type"] == "opm_meg_coordinator.OPMMEGCoordinator"
    sent_metadata = sent["metadata"]
    # The mission's self_concept reached the coordinator's metadata,
    # with ``agent_id`` blanked for ConsciousnessCapability and
    # ``name`` defaulted to the registry label.
    assert sent_metadata.self_concept is not None
    assert sent_metadata.self_concept.name == "OPM-MEG Noise-Floor"
    assert sent_metadata.self_concept.agent_id == ""
    assert sent_metadata.self_concept.description == "Drives the OPM-MEG mission."
    assert sent_metadata.self_concept.goals == ["close the noise budget"]
    assert sent_metadata.self_concept.constraints == ["cite every prediction"]
    # mission_params merged into metadata.parameters; mission_type tag
    # added by the action so the coordinator can self-identify.
    assert sent_metadata.parameters["noise_floor_target_fT_rt_hz"] == 12.0
    assert sent_metadata.parameters["mission_type"] == "opm_meg"
    # spawn_mission's narrow contract is: only mission_params +
    # mission_type. Colony/session-scoped keys from the parent's
    # metadata are the inheritance gate's responsibility (in
    # ``AgentPoolCapability.create_agent``), not spawn_mission's —
    # so they MUST NOT appear in what spawn_mission sends to the
    # pool. Pins that separation so the band-aid (a hardcoded
    # tuple in spawn_mission) can't sneak back.
    for k in (
        "design_monorepo_url", "git_attribution", "github_identity",
        "available_missions", "available_tools",
    ):
        assert k not in sent_metadata.parameters, (
            f"spawn_mission must not thread {k!r} onto the child's "
            f"metadata.parameters — the inheritance gate owns that."
        )


@pytest.mark.asyncio
async def test_spawn_mission_l4_mission_preferred_over_static_snapshot(
    _exec_ctx, monkeypatch,
) -> None:
    """The action re-reads the LIVE merged registry on every call —
    so an L4 mission added since the last static-snapshot refresh is
    spawnable. Pins the "live registry, not metadata snapshot" rule."""

    from polymathera.colony.agents.patterns.capabilities.agent_pool import (
        AgentPoolCapability,
    )

    cap = _make_cap(_exec_ctx)

    captured: dict[str, dict] = {}

    async def _capture_create(**kwargs):
        captured["kwargs"] = kwargs
        return {"agent_id": "child_new", "label": None, "created": True}

    fake_pool = MagicMock()
    fake_pool.create_agent = _capture_create

    # L4 provider returns a mission that does NOT appear in the
    # static snapshot.
    provider = MagicMock()
    provider.ensure_materialized = MagicMock(return_value=True)
    provider.discovered_extensions = SimpleNamespace(missions={
        "fresh_l4_mission": {
            "label": "Fresh L4",
            "coordinator_v2": "l4_pkg.L4Coordinator",
        },
    })

    from polymathera.colony.design_monorepo import RepoStateProvider

    def _gcbt(t):
        if t is AgentPoolCapability:
            return fake_pool
        if t is RepoStateProvider:
            return provider
        return None

    metadata = SimpleNamespace(parameters={
        "design_monorepo_url": "x",
        "available_missions": {},  # the snapshot is stale
    }, session_id="sess_test")
    cap._agent = SimpleNamespace(
        agent_id="session_agent_xyz",
        metadata=metadata,
        get_capability_by_type=_gcbt,
    )

    from polymathera.colony.agents import mission_registry as mr
    monkeypatch.setattr(mr, "get_mission_registry", lambda: {})

    result = await _run_spawn(cap, mission_type="fresh_l4_mission")

    assert result["created"] is True
    assert captured["kwargs"]["agent_type"] == "l4_pkg.L4Coordinator"


@pytest.mark.asyncio
async def test_spawn_mission_no_agent_pool_returns_error(
    _exec_ctx, monkeypatch,
) -> None:
    """If the SessionAgent lacks AgentPoolCapability (defensive
    against future deployments), spawn_mission surfaces a clean error
    instead of dispatching to None."""

    cap = _make_cap(_exec_ctx)

    metadata = SimpleNamespace(parameters={
        "design_monorepo_url": "x",
        "available_missions": {},
    }, session_id="sess_test")
    cap._agent = SimpleNamespace(
        agent_id="session_agent_xyz",
        metadata=metadata,
        get_capability_by_type=lambda _t: None,  # nothing mounted
    )

    from polymathera.colony.agents import mission_registry as mr
    monkeypatch.setattr(mr, "get_mission_registry", lambda: {
        "opm_meg": {
            "label": "x",
            "coordinator_v2": "p.Q",
        },
    })

    result = await _run_spawn(cap, mission_type="opm_meg")
    assert result["created"] is False
    assert "AgentPoolCapability" in result["error"]


def test_no_url_still_reads_discovered_extensions(
    _exec_ctx, monkeypatch,
) -> None:
    """Without a ``design_monorepo_url``,
    :meth:`RepoStateProvider.ensure_materialized` correctly returns
    ``False`` (its public behaviour: no URL → nothing to materialise).
    The helper still reads ``discovered_extensions`` — the working
    tree on disk is authoritative once present (e.g. operator
    pre-seeded, or a prior session left a checkout). In production,
    no URL plus no clone = empty discovered_extensions naturally;
    this test pins the "URL is the materialise trigger, not the
    discovery gate" contract."""

    provider = MagicMock()
    provider.ensure_materialized = MagicMock(return_value=False)
    provider.discovered_extensions = SimpleNamespace(
        missions={"from_disk": {
            "label": "From Disk",
            "coordinator_v2": "pkg.mod.FromDiskCoordinator",
            "worker": "pkg.mod.FromDiskWorker",
        }},
    )

    from polymathera.colony.design_monorepo import RepoStateProvider

    cap = _make_cap(_exec_ctx)
    metadata = SimpleNamespace(parameters={})  # no design_monorepo_url
    cap._agent = SimpleNamespace(
        agent_id="session_agent_xyz",
        metadata=metadata,
        get_capability_by_type=lambda t: provider if t is RepoStateProvider else None,
    )

    from polymathera.colony.agents import mission_registry as mr

    monkeypatch.setattr(mr, "get_mission_registry", lambda: {
        "stub_builtin": {
            "label": "Stub Builtin",
            "coordinator_v2": "pkg.mod.StubCoordinator",
            "worker": "pkg.mod.StubWorker",
        },
    })

    cap._refresh_available_missions()

    avail = metadata.parameters["available_missions"]
    # ensure_materialized was called (returned False; no live clone),
    # but discovered_extensions was read regardless because the on-
    # disk working tree is authoritative once it exists.
    provider.ensure_materialized.assert_called_once()
    assert set(avail) == {"stub_builtin", "from_disk"}


# ---------------------------------------------------------------------------
# Mission spawn-gate integration tests.
#
# These pin the contract that ``spawn_mission`` consults the cluster-
# shared ledger BEFORE dispatching to the pool — moved here from
# ``tests/test_agent_pool.py`` after the gate was lifted out of
# ``AgentPoolCapability.create_agent`` to keep that primitive
# mission-unaware. The pool now stays a generic agent-spawn
# capability; the mission concept lives in the orchestrator surface.
# ---------------------------------------------------------------------------


def _attach_for_gate_test(cap, *, pool, session_id: str = "s1"):
    """Mount the fake pool + the minimal agent shape ``admit_and_spawn``
    needs (``agent_id``, ``metadata.session_id``)."""

    from polymathera.colony.agents.patterns.capabilities.agent_pool import (
        AgentPoolCapability,
    )
    cap._agent = SimpleNamespace(
        agent_id="session_agent_xyz",
        metadata=SimpleNamespace(
            parameters={},
            session_id=session_id, colony_id="c1", tenant_id="t1",
        ),
        get_capability_by_type=lambda t: pool if t is AgentPoolCapability else None,
    )


def _register_test_mission(monkeypatch, mission_type: str = "test_mission"):
    from polymathera.colony.agents import mission_registry as mr
    monkeypatch.setattr(mr, "get_mission_registry", lambda: {
        mission_type: {
            "label": "Test Mission",
            "description": "Stub mission for gate-integration tests.",
            "coordinator_v2": "synthetic.SyntheticCoordinator",
            "worker": "synthetic.SyntheticWorker",
            "self_concept": {
                "description": "Synthetic coordinator.",
                "goals": ["x"],
                "constraints": [],
            },
        },
    })


@pytest.mark.asyncio
async def test_spawn_mission_admits_first_spawn(
    _exec_ctx, monkeypatch,
) -> None:
    """The first spawn of a mission lands a coordinator + registers
    in the cluster-shared ledger."""

    cap = _make_cap(_exec_ctx)
    pool = _make_pool_stub(created_agent_id="child_first")
    _attach_for_gate_test(cap, pool=pool)
    _register_test_mission(monkeypatch)

    result = await _run_spawn(
        cap, mission_type="test_mission",
        mission_params={"mode": "bootstrap"},
    )
    assert result["outcome"] == "spawned"
    assert result["created"] is True
    assert result["agent_id"] == "child_first"
    assert result["mission_gate"] is None


@pytest.mark.asyncio
async def test_spawn_mission_returns_existing_on_chained_mode(
    _exec_ctx, monkeypatch,
) -> None:
    """Second spawn declaring a different mode hands back the running
    agent_id under cap=1 + return_existing — the gate is mode-agnostic,
    so the LLM converges on one coordinator regardless of which mode
    it asked for second."""

    cap = _make_cap(_exec_ctx)
    pool = _make_pool_stub(created_agent_id="child_first")
    _attach_for_gate_test(cap, pool=pool)
    _register_test_mission(monkeypatch)

    first = await _run_spawn(
        cap, mission_type="test_mission",
        mission_params={"mode": "bootstrap"},
    )
    assert first["created"] is True

    second = await _run_spawn(
        cap, mission_type="test_mission",
        mission_params={"mode": "refresh"},
    )
    assert second["outcome"] == "return_existing"
    assert second["created"] is False
    assert second["mission_gate"] == "return_existing"
    assert second["agent_id"] == first["agent_id"]
    assert "returning the running instance" in second["reason"]


@pytest.mark.asyncio
async def test_spawn_mission_rejects_when_policy_says_reject(
    _exec_ctx, monkeypatch,
) -> None:
    """A coordinator whose policy is
    ``on_concurrency_violation=reject`` blocks the second spawn with
    a typed rejection the LLM can branch on."""

    from polymathera.colony.agents.configs import MissionExecutionPolicy

    class _RejectingCoordinator:
        MISSION_EXECUTION_POLICY = MissionExecutionPolicy(
            max_concurrent_instances=1,
            on_concurrency_violation="reject",
        )

    cap = _make_cap(_exec_ctx)
    pool = _make_pool_stub(created_agent_id="child_first")
    pool.resolve_agent_class = MagicMock(return_value=_RejectingCoordinator)
    _attach_for_gate_test(cap, pool=pool)
    _register_test_mission(monkeypatch)

    first = await _run_spawn(
        cap, mission_type="test_mission",
        mission_params={},
    )
    assert first["created"] is True

    second = await _run_spawn(
        cap, mission_type="test_mission",
        mission_params={},
    )
    assert second["outcome"] == "rejected"
    assert second["created"] is False
    assert second["mission_gate"] == "rejected"
    assert second["agent_id"] is None
    assert "currently in flight" in second["error"]
    assert second["suggested_action"]

