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
    assert result["created"] is False
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
