"""Tests for the composite blueprint helper and the
``_on_remote_change`` translator on :class:`DesignCheckpointer`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.blackboard.protocol import (
    DesignMonorepoEventProtocol,
    VCMEventProtocol,
)
from polymathera.colony.design_monorepo import (
    DesignCheckpointer,
    RepoStateProvider,
    ToolBuilder,
    design_monorepo_capability_blueprints,
)


def test_blueprint_helper_returns_three_blueprints_in_order() -> None:
    bps = design_monorepo_capability_blueprints()
    classes = [bp.cls for bp in bps]
    assert classes == [RepoStateProvider, DesignCheckpointer, ToolBuilder]


def test_blueprint_helper_propagates_read_only_state() -> None:
    bps = design_monorepo_capability_blueprints(read_only_state=True)
    state_bp = bps[0]
    assert state_bp.kwargs["read_only"] is True


def test_blueprint_helper_propagates_auto_checkpoint_flag() -> None:
    bps = design_monorepo_capability_blueprints(
        auto_checkpoint_on_quiescence=False,
    )
    checkpointer_bp = bps[1]
    assert checkpointer_bp.kwargs["auto_checkpoint_on_quiescence"] is False


@pytest.mark.asyncio
async def test_no_handler_capabilities_opt_out_via_empty_input_patterns(
    tmp_path,
) -> None:
    """``RepoStateProvider`` and ``ToolBuilder`` declare no event
    handlers — they are pure action surfaces. They MUST pass
    ``input_patterns=[]`` to opt out of the base ``AgentCapability``'s
    wildcard fallback. Otherwise the agent's own
    ``policy:action_started:*`` lifecycle writes (published on the
    agent's primary blackboard) would be fed back into the action
    policy's event queue, producing a tight plan-and-act loop on
    every action the agent dispatches.

    This test pins the convention at the constructor boundary
    (``_input_patterns == []``) and verifies the runtime behaviour
    (``stream_events_to_queue`` does not subscribe).
    """

    import asyncio
    from polymathera.colony.distributed.ray_utils.serving.context import (
        Ring, execution_context,
    )

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        agent = MagicMock(); agent.agent_id = "agent-A"
        state = RepoStateProvider(agent=agent, scope_id="dm",
                                  working_dir=tmp_path)
        tools = ToolBuilder(agent=agent, scope_id="dm",
                            working_dir=tmp_path)

    # Constructor convention: explicit empty list, not None.
    assert state._input_patterns == []
    assert tools._input_patterns == []

    # Runtime behaviour: stream_events_to_queue must not call
    # get_blackboard() on an opt-out capability — i.e. no subscription
    # is ever registered.
    queue: asyncio.Queue = asyncio.Queue()
    bb_calls = 0
    original = type(state).get_blackboard

    async def _spy(self, *a, **kw):
        nonlocal bb_calls
        bb_calls += 1
        return await original(self, *a, **kw)

    type(state).get_blackboard = _spy
    type(tools).get_blackboard = _spy
    try:
        await state.stream_events_to_queue(queue)
        await tools.stream_events_to_queue(queue)
    finally:
        type(state).get_blackboard = original
        type(tools).get_blackboard = original
    assert bb_calls == 0


def test_disabling_auto_checkpoint_drops_quiescence_subscription(
    tmp_path,
) -> None:
    """``auto_checkpoint_on_quiescence=False`` must skip the
    convergence-quiescence subscription entirely.

    Otherwise SessionAgent (whose LLM planner reacts to every queue
    event via the observation-and-dispatch path) would re-plan on
    every episode boundary — producing a tight loop where the same
    welcome message is re-emitted as the action policy keeps settling
    and being woken again. The remote-change subscription stays
    active because it only fires on actual upstream changes.
    """

    from polymathera.colony.agents.blackboard import (
        ConvergenceQuiescenceProtocol,
    )
    from polymathera.colony.distributed.ray_utils.serving.context import (
        Ring, execution_context,
    )

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        agent = MagicMock(); agent.agent_id = "agent-A"
        on = DesignCheckpointer(
            agent=agent, scope_id="dm", working_dir=tmp_path,
            auto_checkpoint_on_quiescence=True,
        )
        off = DesignCheckpointer(
            agent=agent, scope_id="dm", working_dir=tmp_path,
            auto_checkpoint_on_quiescence=False,
        )

    quiescence = ConvergenceQuiescenceProtocol.quiescence_pattern()
    reindexed = VCMEventProtocol.reindexed_pattern()

    assert quiescence in on.input_patterns
    assert reindexed in on.input_patterns

    assert quiescence not in off.input_patterns
    assert reindexed in off.input_patterns


@pytest.mark.asyncio
async def test_on_remote_change_translates_reindexed_to_branch_changed(
    tmp_path,
) -> None:
    """The handler reads ``VCMEventProtocol.reindexed:<scope>`` and
    emits ``DesignMonorepoEventProtocol.branch_changed:<scope>``.

    We construct ``DesignCheckpointer`` with an explicit working_dir
    so we don't need a real per-agent clone for this unit test.
    """

    from polymathera.colony.distributed.ray_utils.serving.context import (
        Ring, execution_context,
    )
    from polymathera.colony.agents.blackboard.types import BlackboardEvent

    # Need a UserCtx because AgentCapability.__init__ may capture one.
    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        agent = MagicMock(); agent.agent_id = "agent-A"
        cap = DesignCheckpointer(
            agent=agent,
            scope_id="dm",
            working_dir=tmp_path,
            auto_checkpoint_on_quiescence=False,
        )

    scope = "my-scope"
    event = BlackboardEvent(
        event_type="write",
        key=VCMEventProtocol.reindexed_key(scope),
        value={},
        timestamp=0.0,
    )
    result = await cap._on_remote_change(event, MagicMock())
    assert result is not None
    assert result.context_key == DesignMonorepoEventProtocol.branch_changed_key(scope)
    assert result.context["scope_id"] == scope


@pytest.mark.asyncio
async def test_on_remote_change_ignores_non_reindexed_keys(tmp_path) -> None:
    from polymathera.colony.distributed.ray_utils.serving.context import (
        Ring, execution_context,
    )
    from polymathera.colony.agents.blackboard.types import BlackboardEvent

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        agent = MagicMock(); agent.agent_id = "agent-A"
        cap = DesignCheckpointer(
            agent=agent,
            scope_id="dm",
            working_dir=tmp_path,
            auto_checkpoint_on_quiescence=False,
        )

    event = BlackboardEvent(
        event_type="write",
        key="something:else",
        value={},
        timestamp=0.0,
    )
    assert await cap._on_remote_change(event, MagicMock()) is None
