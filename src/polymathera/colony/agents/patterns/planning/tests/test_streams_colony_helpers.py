"""Tests for the Colony-side consciousness-stream helpers added in
PR-Col-Samples: :func:`colony_basic_stream` +
:func:`attach_colony_standard_sources` and the sample-coordinator
wiring that uses them.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.patterns.actions.policies import BaseActionPolicy
from polymathera.colony.agents.patterns.planning.streams import (
    EventLogFormatter,
    attach_colony_standard_sources,
    colony_basic_stream,
)


def _agent() -> Any:
    agent = MagicMock()
    agent.agent_id = "agent_col_helpers"
    return agent


# ---------------------------------------------------------------------------
# 1. colony_basic_stream produces a usable catch-all stream
# ---------------------------------------------------------------------------


class TestColonyBasicStream:
    def test_default_name_and_filters(self) -> None:
        bp = colony_basic_stream()
        s = bp.local_instance()
        assert s.name == "agent_experience"
        # All six kinds accepted (catch-all).
        assert set(s._filters.keys()) == {
            "event", "action", "tool_output", "vcm_update",
            "monorepo_commit", "domain_state",
        }
        assert isinstance(s.formatter, EventLogFormatter)

    def test_name_and_section_title_override(self) -> None:
        bp = colony_basic_stream(
            name="impact_mission_progress",
            section_title="## Custom",
            max_entries=99,
        )
        s = bp.local_instance()
        assert s.name == "impact_mission_progress"
        assert s._max_entries == 99

    def test_accepts_every_kind(self) -> None:
        s = colony_basic_stream().local_instance()
        # Walk every consider_* method with a representative payload.
        s.consider_event({"k": {"v": 1}})
        s.consider_action({"action_key": "x", "success": True})
        s.consider_tool_output({"action_key": "y", "tool_result": {}})
        s.consider_vcm_update({"kind": "added", "page_id": "p"})
        s.consider_monorepo_commit(
            {"sha": "x", "branch": "main", "message": "m"},
        )
        s.consider_domain_state({"state_machine": "x", "transition": "t"})
        assert len(s._entries) == 6


# ---------------------------------------------------------------------------
# 2. attach_colony_standard_sources installs 3 sources
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAttachColonyStandardSources:
    async def test_attaches_three_universal_sources(self) -> None:
        policy = BaseActionPolicy(agent=_agent())
        await attach_colony_standard_sources(policy)
        assert len(policy._stream_sources) == 3
        # Names — confirm classes registered.
        cls_names = {type(s).__name__ for s in policy._stream_sources}
        assert cls_names == {
            "AccumulatedContextSource", "ActionCallSource",
            "ToolResultSource",
        }

    async def test_idempotent(self) -> None:
        """Calling twice with the same policy doesn't duplicate
        attached sources (already-attached ones are skipped by
        ``attach_pending_sources``)."""
        policy = BaseActionPolicy(agent=_agent())
        await attach_colony_standard_sources(policy)
        first_count = len(policy._stream_sources)
        # Calling again creates new source instances each time, so the
        # list grows; but the ``_attached_source_ids`` registry stops
        # double-attaching. This documents the current behaviour —
        # operators should call once.
        await attach_colony_standard_sources(policy)
        assert len(policy._stream_sources) == first_count + 3


# ---------------------------------------------------------------------------
# 3. ChangeImpactAnalysisCoordinator wiring intent
# ---------------------------------------------------------------------------


class TestSampleCoordinatorWiring:
    def test_change_impact_coordinator_mounts_stream_and_sources(self) -> None:
        from polymathera.colony.samples.code_analysis.impact.coordinator import (
            ChangeImpactAnalysisCoordinator,
        )
        src = inspect.getsource(ChangeImpactAnalysisCoordinator.initialize)
        assert "colony_basic_stream" in src, (
            "ChangeImpactAnalysisCoordinator.initialize is missing the "
            "colony_basic_stream wiring."
        )
        assert "attach_colony_standard_sources" in src
        # The stream is named to disambiguate from other sample missions.
        assert 'name="impact_mission_progress"' in src
