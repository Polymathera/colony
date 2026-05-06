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

        original = AgentMetadata(tenant_id="t", parent_agent_id="parent")
        await cap.create_agent(
            agent_type="polymathera.colony.agents.base.Agent",
            metadata=original,
        )

    forwarded = cap._resolve_class.return_value.bind.call_args.kwargs["metadata"]
    assert forwarded is original
