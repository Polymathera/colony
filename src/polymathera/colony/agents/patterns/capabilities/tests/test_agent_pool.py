"""Unit tests for ``AgentPoolCapability`` action surfaces.

We only exercise the LLM-action boundary where dict / model coercion
matters — the full spawn pipeline is integration-tested elsewhere.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from polymathera.colony.agents.models import AgentMetadata
from polymathera.colony.agents.patterns.capabilities.agent_pool import (
    AgentPoolCapability,
)
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring, execution_context,
)


@pytest.mark.asyncio
async def test_create_agent_coerces_dict_metadata_to_agent_metadata() -> None:
    """LLM-driven callers naturally pass ``metadata`` as a JSON dict
    (the REPL serialises kwargs). The action must coerce dict →
    :class:`AgentMetadata` before the blueprint enters the spawn
    pipeline; otherwise ``blueprint.metadata.tenant_id`` raises
    ``AttributeError`` inside :class:`AgentSystem.spawn_from_blueprint`
    because attribute access on a plain dict fails.

    This pins the coercion at the action boundary so the regression
    can't sneak back in.
    """

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        agent = MagicMock()
        agent.agent_id = "parent"
        agent.syscontext = MagicMock()
        agent.spawn_child_agents = AsyncMock(return_value=[])

        cap = AgentPoolCapability(agent=agent)
        cap._resolve_class = MagicMock(return_value=MagicMock(
            bind=MagicMock(return_value=MagicMock())
        ))

        result = await cap.create_agent(
            agent_type="polymathera.colony.agents.base.Agent",
            metadata={"tenant_id": "t", "parent_agent_id": "parent"},
        )

    bind_call = cap._resolve_class.return_value.bind
    bind_call.assert_called_once()
    forwarded = bind_call.call_args.kwargs["metadata"]
    assert isinstance(forwarded, AgentMetadata)
    assert forwarded.tenant_id == "t"
    assert forwarded.parent_agent_id == "parent"
    assert "agent_id" in result


@pytest.mark.asyncio
async def test_create_agent_passes_through_typed_metadata_unchanged(
) -> None:
    """When the caller already supplies a typed ``AgentMetadata`` (the
    Python-side path, not the LLM), the action MUST NOT re-wrap it —
    otherwise pinned ``syscontext`` / ``run_id`` etc. would be lost."""

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        agent = MagicMock()
        agent.agent_id = "parent"
        agent.syscontext = MagicMock()
        agent.spawn_child_agents = AsyncMock(return_value=[])

        cap = AgentPoolCapability(agent=agent)
        cap._resolve_class = MagicMock(return_value=MagicMock(
            bind=MagicMock(return_value=MagicMock())
        ))

        original = AgentMetadata(parent_agent_id="parent")
        await cap.create_agent(
            agent_type="polymathera.colony.agents.base.Agent",
            metadata=original,
        )

    forwarded = cap._resolve_class.return_value.bind.call_args.kwargs["metadata"]
    assert forwarded is original


# ---------------------------------------------------------------------------
# _resolve_class fallback to L1-A discovered agent registry
#
# Stage A (cps/STAGE_A_L1A_DYNAMIC_DISCOVERY_PLAN.md): L4 coordinator
# classes authored under <monorepo>/.colony/agents/<file>.py are loaded
# by L1-A's discover_agents into a dict keyed by class short-name, but
# the loader keeps them out of sys.modules — so the canonical
# ``importlib.import_module`` path in _resolve_class cannot find them.
# create_agent threads the parent agent's discovered_extensions.agents
# through _resolve_class as ``fallback_registry``; these tests pin the
# resolution semantics so the L4-via-chat path stays working.
# ---------------------------------------------------------------------------


def test_resolve_class_imports_pip_installed_class() -> None:
    """The primary path: a fully-qualified path to a pip-installed
    class resolves via ``importlib.import_module`` regardless of any
    fallback registry."""

    cls = AgentPoolCapability._resolve_class(
        "polymathera.colony.agents.models.AgentMetadata",
    )
    assert cls is AgentMetadata


def test_resolve_class_falls_back_to_registry_on_import_failure() -> None:
    """When ``importlib.import_module`` raises ``ImportError`` (the
    L4 case — module is not in sys.modules), _resolve_class looks up
    the class short-name in ``fallback_registry`` and returns the
    matching class."""

    class _SyntheticL4Coordinator:
        """Stand-in for an L4 Agent subclass discovered via L1-A."""

    cls = AgentPoolCapability._resolve_class(
        "synthetic_l4_coordinator.SyntheticL4Coordinator",
        fallback_registry={
            "SyntheticL4Coordinator": _SyntheticL4Coordinator,
        },
    )
    assert cls is _SyntheticL4Coordinator


def test_resolve_class_reraises_when_fallback_misses() -> None:
    """If both the importlib path and the fallback registry miss, the
    original ImportError surfaces — the caller must see why resolution
    failed (typo, missing extension, etc.) rather than a silently
    cleared error."""

    with pytest.raises((ImportError, AttributeError)):
        AgentPoolCapability._resolve_class(
            "definitely.not.a.module.NoSuchClass",
            fallback_registry={
                "DifferentClass": object,
            },
        )


def test_resolve_class_no_fallback_kwarg_is_backwards_compatible() -> None:
    """Pre-Stage-A callers (and the integration test pre-update) pass
    only the FQ path. The new optional kwarg defaults to None and the
    behaviour matches the pre-Stage-A static method."""

    with pytest.raises((ImportError, AttributeError)):
        AgentPoolCapability._resolve_class(
            "definitely.not.a.module.NoSuchClass",
        )


def test_resolve_class_importlib_wins_over_fallback() -> None:
    """If a name resolves via importlib AND is in the fallback
    registry, the importlib result wins — pip-installed classes are
    canonical; the fallback exists only to handle the gap, not to
    override it."""

    class _Decoy:
        pass

    cls = AgentPoolCapability._resolve_class(
        "polymathera.colony.agents.models.AgentMetadata",
        fallback_registry={"AgentMetadata": _Decoy},
    )
    assert cls is AgentMetadata
    assert cls is not _Decoy


@pytest.mark.asyncio
async def test_create_agent_uses_l4_fallback_for_l4_coordinator(
    monkeypatch,
) -> None:
    """End-to-end at the create_agent boundary: when the parent agent
    has a RepoStateProvider whose discovered_extensions.agents contains
    a class matching the requested agent_type's short-name, the spawn
    path resolves to that class — even though the fully-qualified
    module is not importable."""

    class _SyntheticOPMMEGCoordinator:
        @staticmethod
        def bind(**kwargs):
            return MagicMock(metadata=kwargs.get("metadata"))

    discovered = MagicMock()
    discovered.agents = {
        "SyntheticOPMMEGCoordinator": _SyntheticOPMMEGCoordinator,
    }

    fake_provider = MagicMock()
    fake_provider.discovered_extensions = discovered

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        agent = MagicMock()
        agent.agent_id = "parent"
        agent.syscontext = MagicMock()
        # Pin a real (empty) AgentMetadata so the central inheritance
        # gate in create_agent reads a plain dict for parent_params
        # rather than an auto-vivified MagicMock attribute (which
        # would leak MagicMock values into the spawned child's
        # metadata.parameters and break the real Agent.bind() that
        # this test exercises end-to-end).
        agent.metadata = AgentMetadata(
            parent_agent_id="parent",
        )
        agent.spawn_child_agents = AsyncMock(return_value=[
            MagicMock(child_agent_id="child_xyz"),
        ])
        # Mimic Agent.get_capability_by_type by returning the fake
        # provider for any RepoStateProvider lookup, None otherwise.
        from polymathera.colony.design_monorepo import RepoStateProvider

        def _gcbt(t):
            return fake_provider if t is RepoStateProvider else None

        agent.get_capability_by_type = _gcbt

        cap = AgentPoolCapability(agent=agent)
        result = await cap.create_agent(
            agent_type=(
                "synthetic_opm_meg_coordinator.SyntheticOPMMEGCoordinator"
            ),
            # Explicit metadata sidesteps the default-construction
            # path that would try to validate ``self.agent.syscontext``
            # (a MagicMock) into a real ExecutionContext.
            metadata={"tenant_id": "t", "parent_agent_id": "parent"},
        )

    assert result["agent_id"] == "child_xyz"
    assert result["created"] is True


@pytest.mark.asyncio
async def test_create_agent_works_without_repo_state_provider(
) -> None:
    """No RepoStateProvider mounted (detached / non-monorepo agent) →
    fallback registry is empty; pip-installed agent_type still
    resolves via importlib unchanged."""

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        agent = MagicMock()
        agent.agent_id = "parent"
        agent.syscontext = MagicMock()
        # See the previous test for why a real (empty) AgentMetadata
        # is wired here instead of relying on MagicMock auto-vivification.
        agent.metadata = AgentMetadata(
            parent_agent_id="parent",
        )
        agent.spawn_child_agents = AsyncMock(return_value=[
            MagicMock(child_agent_id="child_xyz"),
        ])
        agent.get_capability_by_type = lambda _t: None

        cap = AgentPoolCapability(agent=agent)
        result = await cap.create_agent(
            agent_type="polymathera.colony.agents.base.Agent",
            metadata={"tenant_id": "t", "parent_agent_id": "parent"},
        )

    assert result["created"] is True


# ---------------------------------------------------------------------------
# The central COLONY/SESSION-scoped parameter inheritance gate.
#
# Pins the contract that ``create_agent`` automatically threads
# colony/session-scoped ``metadata.parameters`` keys from the parent
# to the spawned child. Without this gate every spawn site (in the
# chat path, in the REST path, in any future path) would have to copy
# ``design_monorepo_url`` / ``git_attribution`` / ``github_identity``
# by hand — exactly the bug class
# ``colony/agent_metadata_parameter_spec_plan.md`` formalises out.


@pytest.fixture
def _stub_param_registry(monkeypatch):
    """Build an isolated registry containing two COLONY-scoped keys,
    one SESSION-scoped, one CALLER-scoped, and one AGENT-scoped;
    patch the accessor ``create_agent`` calls so the test is not
    coupled to which real capabilities happen to be imported."""

    from polymathera.colony.agents import metadata_parameters as mp

    reg = mp.MetadataParameterRegistry()
    reg.register(mp.ParameterSpec(
        name="design_monorepo_url",
        scope=mp.ParameterScope.COLONY, description="x",
    ))
    reg.register(mp.ParameterSpec(
        name="github_identity",
        scope=mp.ParameterScope.COLONY, description="x",
        json_type="object",
    ))
    reg.register(mp.ParameterSpec(
        name="available_tools",
        scope=mp.ParameterScope.SESSION, description="x",
        json_type="object", default_factory=dict,
    ))
    reg.register(mp.ParameterSpec(
        name="mode",
        scope=mp.ParameterScope.CALLER, description="x",
    ))
    reg.register(mp.ParameterSpec(
        name="agent_local",
        scope=mp.ParameterScope.AGENT, description="x",
    ))
    monkeypatch.setattr(mp, "get_metadata_parameter_registry", lambda: reg)
    return reg


def _spy_parent_agent(parent_params: dict) -> MagicMock:
    """Build a parent agent mock whose ``metadata.parameters`` is the
    given dict (a real dict, not a MagicMock auto-vivified attr —
    so the inheritance gate reads the seeded values, not auto-
    children)."""

    agent = MagicMock()
    agent.agent_id = "parent"
    agent.syscontext = MagicMock()
    # IMPORTANT: assign a real AgentMetadata so the gate's
    # ``getattr(self.agent.metadata, "parameters", None)`` returns a
    # plain dict rather than an auto-MagicMock.
    agent.metadata = AgentMetadata(
        parent_agent_id="parent",
    )
    agent.metadata.parameters = parent_params
    agent.spawn_child_agents = AsyncMock(return_value=[
        MagicMock(child_agent_id="child_xyz"),
    ])
    agent.get_capability_by_type = lambda _t: None
    return agent


@pytest.mark.asyncio
async def test_create_agent_inherits_colony_scoped_params(
    _stub_param_registry,
) -> None:
    """COLONY-scoped keys flow parent→child even when the caller's
    ``metadata`` dict carries no parameters."""

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        parent_params = {
            "design_monorepo_url": "https://github.com/acme/mono.git",
            "github_identity": {"tenant_installation_id": "100"},
        }
        agent = _spy_parent_agent(parent_params)
        cap = AgentPoolCapability(agent=agent)
        cap._resolve_class = MagicMock(return_value=MagicMock(
            bind=MagicMock(return_value=MagicMock()),
        ))

        await cap.create_agent(
            agent_type="polymathera.colony.agents.base.Agent",
            metadata={"tenant_id": "t", "parent_agent_id": "parent"},
        )

    forwarded = cap._resolve_class.return_value.bind.call_args.kwargs["metadata"]
    assert forwarded.parameters["design_monorepo_url"] == (
        "https://github.com/acme/mono.git"
    )
    assert forwarded.parameters["github_identity"] == {
        "tenant_installation_id": "100",
    }


@pytest.mark.asyncio
async def test_create_agent_inherits_session_scoped_params(
    _stub_param_registry,
) -> None:
    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        parent_params = {"available_tools": {"foo": "bar"}}
        agent = _spy_parent_agent(parent_params)
        cap = AgentPoolCapability(agent=agent)
        cap._resolve_class = MagicMock(return_value=MagicMock(
            bind=MagicMock(return_value=MagicMock()),
        ))

        await cap.create_agent(
            agent_type="polymathera.colony.agents.base.Agent",
            metadata={"tenant_id": "t", "parent_agent_id": "parent"},
        )

    forwarded = cap._resolve_class.return_value.bind.call_args.kwargs["metadata"]
    assert forwarded.parameters["available_tools"] == {"foo": "bar"}


@pytest.mark.asyncio
async def test_create_agent_does_not_inherit_caller_or_agent_scoped_params(
    _stub_param_registry,
) -> None:
    """CALLER-scoped keys would silently rebind the child's intent if
    inherited; AGENT-scoped keys would alias per-agent state across
    the spawn boundary. Both MUST stay on the parent."""

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        parent_params = {"mode": "bootstrap", "agent_local": "abc"}
        agent = _spy_parent_agent(parent_params)
        cap = AgentPoolCapability(agent=agent)
        cap._resolve_class = MagicMock(return_value=MagicMock(
            bind=MagicMock(return_value=MagicMock()),
        ))

        await cap.create_agent(
            agent_type="polymathera.colony.agents.base.Agent",
            metadata={"tenant_id": "t", "parent_agent_id": "parent"},
        )

    forwarded = cap._resolve_class.return_value.bind.call_args.kwargs["metadata"]
    assert "mode" not in forwarded.parameters
    assert "agent_local" not in forwarded.parameters


@pytest.mark.asyncio
async def test_create_agent_child_wins_on_collision(
    _stub_param_registry,
) -> None:
    """The LLM planner deliberately rebinding ``design_monorepo_url``
    (e.g. a mission probing a different repo) keeps the child's
    value; the parent's value is NOT silently overwritten."""

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        parent_params = {"design_monorepo_url": "https://acme/mono.git"}
        agent = _spy_parent_agent(parent_params)
        cap = AgentPoolCapability(agent=agent)
        cap._resolve_class = MagicMock(return_value=MagicMock(
            bind=MagicMock(return_value=MagicMock()),
        ))

        await cap.create_agent(
            agent_type="polymathera.colony.agents.base.Agent",
            metadata={
                "tenant_id": "t", "parent_agent_id": "parent",
                "parameters": {
                    "design_monorepo_url": "https://acme/probe.git",
                },
            },
        )

    forwarded = cap._resolve_class.return_value.bind.call_args.kwargs["metadata"]
    assert forwarded.parameters["design_monorepo_url"] == (
        "https://acme/probe.git"
    )


@pytest.mark.asyncio
async def test_create_agent_inherit_scoped_params_false_opts_out(
    _stub_param_registry,
) -> None:
    """Programmatic callers can opt out via the
    ``inherit_scoped_params=False`` kwarg (peeled out of
    ``agent_kwargs`` so it doesn't appear in the LLM-visible
    signature). Used by intentionally-detached spawns."""

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        parent_params = {"design_monorepo_url": "https://acme/mono.git"}
        agent = _spy_parent_agent(parent_params)
        cap = AgentPoolCapability(agent=agent)
        cap._resolve_class = MagicMock(return_value=MagicMock(
            bind=MagicMock(return_value=MagicMock()),
        ))

        await cap.create_agent(
            agent_type="polymathera.colony.agents.base.Agent",
            metadata={"tenant_id": "t", "parent_agent_id": "parent"},
            inherit_scoped_params=False,
        )

    forwarded = cap._resolve_class.return_value.bind.call_args.kwargs["metadata"]
    assert "design_monorepo_url" not in forwarded.parameters


# ---------------------------------------------------------------------------
# get_agent_status — authoritative state from AgentSystemDeployment.
#
# The prior implementation derived ``state="RUNNING"`` from
# ``agent_id in self.agent.child_agents.values()`` and never queried the
# registry. That left STOPPED / FAILED / SUSPENDED agents reported as
# ``"RUNNING"`` forever (run9 regression: SessionAgent polled this and
# told the user the coordinator was running for 17+ min AFTER it had
# cleanly STOPPED). Authoritative reads from
# :meth:`AgentSystemDeployment.get_agent_info` close the gap.
# ---------------------------------------------------------------------------


from unittest.mock import patch
from polymathera.colony.agents.models import (
    AgentRegistrationInfo,
    AgentState,
)
from polymathera.colony.agents.system import AgentSystemDeployment


def _make_pool_capability_with_handles(handles: dict[str, object]) -> AgentPoolCapability:
    """Build a ``AgentPoolCapability`` with pre-seeded
    ``_agent_handles``. The status path reads state from the
    registry, not the handles — but the handles dict still drives
    ``has_handle`` + the default ``agent_ids=None`` listing path, so
    the test fixture must populate it to exercise both."""

    agent = MagicMock()
    agent.agent_id = "parent"
    agent._app_name = "app-x"
    cap = AgentPoolCapability(agent=agent)
    cap._agent_handles = dict(handles)
    return cap


def _make_info(*, agent_id: str, state: AgentState) -> AgentRegistrationInfo:
    return AgentRegistrationInfo(
        agent_id=agent_id,
        agent_type="polymathera.colony.agents.base.Agent",
        state=state,
        tenant_id="t",
        colony_id="c",
    )


@pytest.mark.asyncio
async def test_get_agent_status_reports_stopped_from_registry() -> None:
    """The state must come from the registry, not from in-memory
    child-agent membership. The run9 bug: a stopped coordinator kept
    showing as ``RUNNING`` because the legacy code only checked
    ``agent_id in self.agent.child_agents.values()``. Pin the new
    contract — a registry ``state=STOPPED`` surfaces verbatim."""

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        cap = _make_pool_capability_with_handles({"child-1": object()})

        fake_system = AsyncMock(spec=AgentSystemDeployment)
        fake_system.get_agent_info.return_value = _make_info(
            agent_id="child-1", state=AgentState.STOPPED,
        )

        with patch(
            "polymathera.colony._handles.get_agent_system",
            new=AsyncMock(return_value=fake_system),
        ):
            result = await cap.get_agent_status(agent_ids=["child-1"])

        assert len(result["agents"]) == 1
        assert result["agents"][0]["state"] == "STOPPED"
        assert result["agents"][0]["agent_id"] == "child-1"
        assert result["agents"][0]["has_handle"] is True
        fake_system.get_agent_info.assert_awaited_once_with("child-1")


@pytest.mark.asyncio
async def test_get_agent_status_reports_running_from_registry() -> None:
    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        cap = _make_pool_capability_with_handles({"child-1": object()})

        fake_system = AsyncMock(spec=AgentSystemDeployment)
        fake_system.get_agent_info.return_value = _make_info(
            agent_id="child-1", state=AgentState.RUNNING,
        )

        with patch(
            "polymathera.colony._handles.get_agent_system",
            new=AsyncMock(return_value=fake_system),
        ):
            result = await cap.get_agent_status(agent_ids=["child-1"])

        assert result["agents"][0]["state"] == "RUNNING"


@pytest.mark.asyncio
async def test_get_agent_status_unregistered_for_missing_agent() -> None:
    """When the registry has no record (agent terminated and reaped,
    or never existed), surface ``UNREGISTERED`` so the LLM caller can
    branch on it — distinct from ``UNKNOWN`` (read-side error) and
    distinct from any real lifecycle state."""

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        cap = _make_pool_capability_with_handles({"ghost": object()})

        fake_system = AsyncMock(spec=AgentSystemDeployment)
        fake_system.get_agent_info.return_value = None

        with patch(
            "polymathera.colony._handles.get_agent_system",
            new=AsyncMock(return_value=fake_system),
        ):
            result = await cap.get_agent_status(agent_ids=["ghost"])

        assert result["agents"][0]["state"] == "UNREGISTERED"


@pytest.mark.asyncio
async def test_get_agent_status_unknown_on_registry_read_error() -> None:
    """Transient registry read-side errors must NOT crash the action
    — surface ``UNKNOWN`` so the LLM gets a typed envelope and can
    decide whether to retry, wait, or escalate."""

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        cap = _make_pool_capability_with_handles({"flaky": object()})

        fake_system = AsyncMock(spec=AgentSystemDeployment)
        fake_system.get_agent_info.side_effect = RuntimeError("transient")

        with patch(
            "polymathera.colony._handles.get_agent_system",
            new=AsyncMock(return_value=fake_system),
        ):
            result = await cap.get_agent_status(agent_ids=["flaky"])

        assert result["agents"][0]["state"] == "UNKNOWN"


@pytest.mark.asyncio
async def test_get_agent_status_filter_state_matches_registry_state() -> None:
    """``filter_state`` must compare against the registry-derived
    state. With the bogus old impl returning everything as ``RUNNING``,
    a filter of ``"STOPPED"`` would never match any agent. Pin that
    the filter now matches real states."""

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        cap = _make_pool_capability_with_handles(
            {"alive": object(), "dead": object()},
        )

        async def _info(agent_id: str) -> AgentRegistrationInfo:
            if agent_id == "alive":
                return _make_info(agent_id=agent_id, state=AgentState.RUNNING)
            return _make_info(agent_id=agent_id, state=AgentState.STOPPED)

        fake_system = AsyncMock(spec=AgentSystemDeployment)
        fake_system.get_agent_info.side_effect = _info

        with patch(
            "polymathera.colony._handles.get_agent_system",
            new=AsyncMock(return_value=fake_system),
        ):
            stopped = await cap.get_agent_status(
                agent_ids=["alive", "dead"], filter_state="STOPPED",
            )
            running = await cap.get_agent_status(
                agent_ids=["alive", "dead"], filter_state="RUNNING",
            )

        assert [a["agent_id"] for a in stopped["agents"]] == ["dead"]
        assert [a["agent_id"] for a in running["agents"]] == ["alive"]


@pytest.mark.asyncio
async def test_get_agent_status_defaults_to_all_handle_ids() -> None:
    """``agent_ids=None`` enumerates every entry in
    ``_agent_handles``. The set must come from the local handle
    cache (not the registry) so the action stays cheap for the
    common 'list my children' case."""

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        cap = _make_pool_capability_with_handles(
            {"c1": object(), "c2": object(), "c3": object()},
        )

        async def _info(agent_id: str) -> AgentRegistrationInfo:
            return _make_info(agent_id=agent_id, state=AgentState.RUNNING)

        fake_system = AsyncMock(spec=AgentSystemDeployment)
        fake_system.get_agent_info.side_effect = _info

        with patch(
            "polymathera.colony._handles.get_agent_system",
            new=AsyncMock(return_value=fake_system),
        ):
            result = await cap.get_agent_status()

        assert {a["agent_id"] for a in result["agents"]} == {"c1", "c2", "c3"}
        assert result["total"] == 3
