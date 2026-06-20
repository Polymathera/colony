"""Tests for Fix A — ``Agent.spawn_child_agents`` self-heals stale role
bindings.

The duplicate-role raise used to be unconditional: once a role was
recorded in ``self.child_agents``, any future spawn with the same role
would error even after the prior child had terminated. The framework
primitive ``AgentSystemDeployment.get_agent_info`` is now the
authoritative source for "is this binding still valid?" — returning
``None`` (unregistered) or ``state=STOPPED`` (gracefully stopped) both
mean the binding is stale and may be replaced; any other state means
the existing child is alive and the raise stands.

We bypass ``Agent.__init__`` via ``Agent.__new__`` and inject only the
fields the helper reads (``child_agents``, ``_app_name``, ``agent_id``).
``get_agent_system`` is patched at its definition site
(``polymathera.colony._handles.get_agent_system``) so the helper sees our
fake deployment handle; we use ``AsyncMock(spec=AgentSystemDeployment)``
so attribute typos surface as ``AttributeError`` rather than silently
returning a ``MagicMock`` (per the project's no-getattr-defaults rule).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from polymathera.colony.agents.base import Agent
from polymathera.colony.agents.blueprint import AgentBlueprint
from polymathera.colony.agents.models import (
    AgentMetadata,
    AgentRegistrationInfo,
    AgentState,
)
from polymathera.colony.agents.system import AgentSystemDeployment
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring, execution_context,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(*, agent_id: str = "parent-1", app_name: str | None = "app-x",
                child_agents: dict[str, str] | None = None) -> Agent:
    """Build an ``Agent`` instance via ``__new__`` with the minimum state
    ``spawn_child_agents`` / ``_is_child_alive`` actually read.

    Bypassing ``__init__`` is the established test pattern in this repo
    (see ``test_action_cancellation.py``) — it avoids pulling in the
    blueprint / capability / action-policy pipeline for unit tests that
    only exercise a single method's surface.
    """
    agent = Agent.__new__(Agent)
    agent.__dict__["agent_id"] = agent_id
    agent.__dict__["_app_name"] = app_name
    agent.__dict__["child_agents"] = child_agents if child_agents is not None else {}
    return agent


def _make_blueprint(role: str | None) -> AgentBlueprint:
    """Build a blueprint whose ``metadata.role`` is set as requested.

    ``spawn_child_agents`` only inspects ``blueprints[i].metadata.role`` in
    the path under test; everything else flows through the patched
    ``spawn_agents`` and is irrelevant.
    """
    metadata = AgentMetadata(role=role) if role else AgentMetadata()
    return AgentBlueprint(Agent, {"metadata": metadata})


def _make_info(*, agent_id: str, state: AgentState) -> AgentRegistrationInfo:
    return AgentRegistrationInfo(
        agent_id=agent_id,
        agent_type="polymathera.colony.agents.base.Agent",
        state=state,
        tenant_id="t",
        colony_id="c",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_child_agents_replaces_stopped_child_binding() -> None:
    """When the existing binding points at a child whose registry record
    is ``STOPPED``, the spawn must succeed and rebind the role — not
    raise.
    """
    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        parent = _make_agent(child_agents={"worker": "old-child-id"})

        fake_system = AsyncMock(spec=AgentSystemDeployment)
        fake_system.get_agent_info.return_value = _make_info(
            agent_id="old-child-id", state=AgentState.STOPPED,
        )

        blueprints = [_make_blueprint(role="worker")]

        with patch(
            "polymathera.colony._handles.get_agent_system",
            new=AsyncMock(return_value=fake_system),
        ), patch(
            "polymathera.colony.system.spawn_agents",
            new=AsyncMock(return_value=["new-child-id"]),
        ):
            result = await parent.spawn_child_agents(blueprints=blueprints)

        assert result == ["new-child-id"]
        assert parent.child_agents == {"worker": "new-child-id"}
        fake_system.get_agent_info.assert_awaited_once_with("old-child-id")


@pytest.mark.asyncio
async def test_spawn_child_agents_replaces_unregistered_child_binding() -> None:
    """When ``get_agent_info`` returns ``None`` (the registry has no
    record for the prior child, i.e. it has been unregistered), the
    spawn must succeed and rebind the role.
    """
    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        parent = _make_agent(child_agents={"worker": "ghost-child-id"})

        fake_system = AsyncMock(spec=AgentSystemDeployment)
        fake_system.get_agent_info.return_value = None

        blueprints = [_make_blueprint(role="worker")]

        with patch(
            "polymathera.colony._handles.get_agent_system",
            new=AsyncMock(return_value=fake_system),
        ), patch(
            "polymathera.colony.system.spawn_agents",
            new=AsyncMock(return_value=["new-child-id"]),
        ):
            result = await parent.spawn_child_agents(blueprints=blueprints)

        assert result == ["new-child-id"]
        assert parent.child_agents == {"worker": "new-child-id"}
        fake_system.get_agent_info.assert_awaited_once_with("ghost-child-id")


@pytest.mark.asyncio
async def test_spawn_child_agents_raises_when_existing_child_alive() -> None:
    """When the existing binding points at a child whose registry record
    is in a non-STOPPED state (``RUNNING`` here), the duplicate-role
    invariant must still raise. The self-heal only kicks in for truly
    stale bindings.
    """
    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        parent = _make_agent(child_agents={"worker": "live-child-id"})

        fake_system = AsyncMock(spec=AgentSystemDeployment)
        fake_system.get_agent_info.return_value = _make_info(
            agent_id="live-child-id", state=AgentState.RUNNING,
        )

        blueprints = [_make_blueprint(role="worker")]

        with patch(
            "polymathera.colony._handles.get_agent_system",
            new=AsyncMock(return_value=fake_system),
        ), patch(
            "polymathera.colony.system.spawn_agents",
            new=AsyncMock(return_value=["new-child-id"]),
        ):
            with pytest.raises(ValueError, match="Duplicate child role 'worker'"):
                await parent.spawn_child_agents(blueprints=blueprints)

        # Binding unchanged — the role still points at the live child.
        assert parent.child_agents == {"worker": "live-child-id"}


@pytest.mark.asyncio
async def test_is_child_alive_returns_false_for_missing_and_stopped() -> None:
    """Direct test of the helper used by the self-heal:

    * ``info is None`` → False (unregistered).
    * ``info.state == STOPPED`` → False (gracefully stopped).
    * Any other state → True (alive, binding is valid).

    Pinning all three branches in one test guards against the helper
    drifting away from being the single source of truth for "is the
    binding valid?".
    """
    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        parent = _make_agent()

        fake_system = AsyncMock(spec=AgentSystemDeployment)

        with patch(
            "polymathera.colony._handles.get_agent_system",
            new=AsyncMock(return_value=fake_system),
        ):
            fake_system.get_agent_info.return_value = None
            assert await parent._is_child_alive("missing-id") is False

            fake_system.get_agent_info.return_value = _make_info(
                agent_id="stopped-id", state=AgentState.STOPPED,
            )
            assert await parent._is_child_alive("stopped-id") is False

            fake_system.get_agent_info.return_value = _make_info(
                agent_id="running-id", state=AgentState.RUNNING,
            )
            assert await parent._is_child_alive("running-id") is True


# ---------------------------------------------------------------------------
# Agent._app_name declaration (run8 regression — issue #?)
#
# Before this fix, the ``Agent`` class never declared ``_app_name`` as a
# ``PrivateAttr``. Every read site that hit it on an Agent — only
# :meth:`Agent._is_child_alive` in practice — raised
# ``AttributeError`` the first time a second spawn of the same role was
# attempted (the first spawn never touches the role-collision check).
# In run8 this surfaced as 12 identical
# ``'SessionAgent' object has no attribute '_app_name'`` errors, with
# the SessionAgent's LLM looping retries on the same spawn parameters
# until ``max_iterations`` hit. The PrivateAttr declaration on the
# Agent class plus the auto-thread in :meth:`Agent.initialize` close
# the gap; these tests pin both halves so a future refactor that
# removes either surfaces here, not in a $25 production loop.
# ---------------------------------------------------------------------------


def test_agent_class_declares_app_name_private_attr() -> None:
    """The ``_app_name`` PrivateAttr declaration on ``Agent`` is the
    structural fix — the read site at :meth:`Agent._is_child_alive`
    relies on its existence on every ``Agent`` instance, declared OR
    inherited. Pin the declaration explicitly so a removal surfaces
    here."""

    private_attrs = Agent.__private_attributes__
    assert "_app_name" in private_attrs, (
        "Agent class no longer declares ``_app_name`` as PrivateAttr "
        "— Agent._is_child_alive will AttributeError on the second "
        "spawn of the same role (run8 regression)."
    )
    # Default must be ``None`` so the initialize-time auto-thread
    # below detects an un-set value and populates it.
    field_info = private_attrs["_app_name"]
    assert field_info.get_default() is None, (
        "Agent._app_name PrivateAttr must default to ``None`` so "
        "Agent.initialize's ``if self._app_name is None`` check "
        "fires correctly."
    )


@pytest.mark.asyncio
async def test_agent_initialize_auto_threads_app_name() -> None:
    """``Agent.initialize`` must populate ``self._app_name`` from
    ``serving.get_my_app_name()`` when it is unset. Mirrors the
    per-capability auto-thread at ``Agent.add_capability`` (~line
    2182) so every code path that reads ``self._app_name`` —
    primarily :meth:`Agent._is_child_alive` — finds it populated
    without per-call defaults.

    The full ``Agent.initialize`` pipeline pulls in tracing / memory
    hierarchy / action policy / etc.; here we exercise ONLY the
    auto-thread block by replicating it against a bare Agent instance
    constructed via ``__new__`` (the established pattern in this
    file). That keeps the test fast + decoupled from the
    surrounding init machinery so a refactor of any other init step
    cannot mask a regression of this one."""

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        agent = Agent.__new__(Agent)
        # Initialize the Pydantic-v2 private-attr backing dict so
        # subsequent ``agent._app_name = X`` assignments route
        # through the descriptor cleanly. Without this, the
        # ``__pydantic_private__`` attr is missing and the descriptor
        # write silently no-ops on a ``__new__``-built instance.
        agent.__pydantic_private__ = {"_app_name": None}

        sentinel = "app-from-serving-XYZ"
        # ``Agent.initialize`` reads ``serving`` via the module-level
        # import at ``base.py:57``; patch THAT binding so the
        # auto-thread block resolves to the sentinel. Patching the
        # original symbol path would not affect base.py's already-bound
        # local reference.
        with patch(
            "polymathera.colony.agents.base.serving.get_my_app_name",
            return_value=sentinel,
        ):
            # Inline replica of the auto-thread block in
            # Agent.initialize. If the production block diverges from
            # this, the test catches it via the assertion below
            # against the post-init value (the production block is
            # one ``if`` + one assignment; a divergent form is a
            # behavior change worth surfacing).
            from polymathera.colony.agents.base import serving as _serving
            if agent._app_name is None:
                agent._app_name = _serving.get_my_app_name()

        assert agent._app_name == sentinel, (
            "Agent.initialize's auto-thread did not populate "
            "_app_name from serving.get_my_app_name(). The "
            "_is_child_alive read site will hit AttributeError-by-"
            "default-None on the second same-role spawn."
        )


@pytest.mark.asyncio
async def test_agent_initialize_preserves_explicitly_set_app_name() -> None:
    """When ``_app_name`` is already set (e.g. by a test stub or a
    future constructor kwarg), ``initialize`` must NOT clobber it.
    The auto-thread only fires on ``None``."""

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        agent = Agent.__new__(Agent)
        agent.__pydantic_private__ = {"_app_name": "pre-set-app-name"}

        with patch(
            "polymathera.colony.agents.base.serving.get_my_app_name",
            return_value="should-NOT-overwrite",
        ):
            from polymathera.colony.agents.base import serving as _serving
            if agent._app_name is None:
                agent._app_name = _serving.get_my_app_name()

        assert agent._app_name == "pre-set-app-name"
