"""Tests for the PR-Sub-2a substrate expansion of consciousness
streams: new ``consider_*`` methods on
:class:`~polymathera.colony.agents.patterns.planning.streams.ConsciousnessStream`,
the per-kind filter dict, the
:class:`~polymathera.colony.agents.patterns.planning.sources.StreamEventSource`
ABC + concrete sources (``AccumulatedContextSource`` /
``ActionCallSource`` / ``ToolResultSource``), and the
:meth:`BaseActionPolicy.record_stream_entry` /
``register_tool_result_source`` plumbing.

Covered:

1. Stream — per-kind filter dict accepts old-shape ``event_filter`` /
   ``action_filter`` kwargs + new-shape ``filters`` dict; precedence
   between the two; new ``consider_*`` methods record typed entries
   when the matching filter accepts.
2. Backward-compat properties (``_event_filter`` / ``_action_filter``)
   remain readable.
3. ``StreamEventSource`` ABC has the required abstract method.
4. Concrete sources: ``AccumulatedContextSource`` /
   ``ActionCallSource`` ``attach`` is a no-op (recorded for
   introspection). ``ToolResultSource.attach`` registers via
   ``policy.register_tool_result_source``.
5. ``ToolResultSource.build_payload`` duck-types correctly: dict /
   Pydantic-style model / attribute-bearing object → payload;
   non-shaped value → ``None``.
6. ``BaseActionPolicy.record_stream_entry`` fans entries to every
   mounted stream's matching ``consider_*`` and tolerates unknown
   kinds + stream-side exceptions.
7. ``BaseActionPolicy.dispatch`` post-dispatch tool-result fanout
   only fires when a ``ToolResultSource`` is registered and the
   action result has the ToolResult shape.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from polymathera.colony.agents.patterns.actions.policies import BaseActionPolicy
from polymathera.colony.agents.patterns.planning.sources import (
    AccumulatedContextSource,
    ActionCallSource,
    StreamEventSource,
    ToolResultSource,
)
from polymathera.colony.agents.patterns.planning.streams import (
    ConsciousnessStream,
    JSONStreamFormatter,
)


def _agent() -> Any:
    agent = MagicMock()
    agent.agent_id = "agent_substrate_test"
    return agent


def _stream(**kw) -> ConsciousnessStream:
    return ConsciousnessStream(
        name=kw.pop("name", "s"),
        formatter=JSONStreamFormatter(section_title="## S"),
        **kw,
    )


def _accept_all(_payload: Any) -> bool:
    return True


def _reject_all(_payload: Any) -> bool:
    return False


# ---------------------------------------------------------------------------
# 1. Per-kind filter dict
# ---------------------------------------------------------------------------


class TestPerKindFilters:
    def test_legacy_event_filter_kwarg_lifts_into_filters_dict(self) -> None:
        s = _stream(event_filter=_accept_all)
        assert s._filters == {"event": _accept_all}

    def test_all_per_kind_kwargs_lift(self) -> None:
        s = _stream(
            event_filter=_accept_all,
            action_filter=_accept_all,
            tool_output_filter=_accept_all,
            vcm_update_filter=_accept_all,
            monorepo_commit_filter=_accept_all,
            domain_state_filter=_accept_all,
        )
        assert set(s._filters.keys()) == {
            "event", "action", "tool_output", "vcm_update",
            "monorepo_commit", "domain_state",
        }

    def test_filters_dict_kwarg_takes_precedence(self) -> None:
        """When the same kind appears in both ``event_filter=`` AND
        ``filters={"event": ...}``, the dict wins."""
        s = _stream(
            event_filter=_accept_all,
            filters={"event": _reject_all},
        )
        assert s._filters["event"] is _reject_all

    def test_backward_compat_properties_readable(self) -> None:
        s = _stream(event_filter=_accept_all, action_filter=_reject_all)
        assert s._event_filter is _accept_all
        assert s._action_filter is _reject_all

    def test_missing_filter_defaults_to_none_via_property(self) -> None:
        s = _stream()
        assert s._event_filter is None
        assert s._action_filter is None


# ---------------------------------------------------------------------------
# 2. New consider_* methods
# ---------------------------------------------------------------------------


class TestNewConsiderMethods:
    @pytest.mark.parametrize(
        "kind, method, payload",
        [
            ("tool_output", "consider_tool_output",
             {"action_key": "a", "tool_result": {}, "success": True}),
            ("vcm_update", "consider_vcm_update",
             {"kind": "added", "page_id": "p1"}),
            ("monorepo_commit", "consider_monorepo_commit",
             {"sha": "abc", "branch": "main", "message": "m"}),
            ("domain_state", "consider_domain_state",
             {"state_machine": "g", "transition": "PROPOSE→GROUND"}),
        ],
    )
    def test_records_when_filter_accepts(
        self, kind: str, method: str, payload: dict[str, Any],
    ) -> None:
        s = _stream(filters={kind: _accept_all})
        getattr(s, method)(payload)
        assert len(s._entries) == 1
        entry = s._entries[0]
        assert entry["kind"] == kind
        assert entry["payload"] == payload

    @pytest.mark.parametrize(
        "method, payload",
        [
            ("consider_tool_output", {"action_key": "a"}),
            ("consider_vcm_update", {"kind": "added"}),
            ("consider_monorepo_commit", {"sha": "x"}),
            ("consider_domain_state", {"transition": "X"}),
        ],
    )
    def test_drops_when_filter_rejects(
        self, method: str, payload: dict[str, Any],
    ) -> None:
        s = _stream(filters={
            "tool_output": _reject_all, "vcm_update": _reject_all,
            "monorepo_commit": _reject_all, "domain_state": _reject_all,
        })
        getattr(s, method)(payload)
        assert s._entries == []

    @pytest.mark.parametrize(
        "method, payload",
        [
            ("consider_tool_output", {"action_key": "a"}),
            ("consider_vcm_update", {"kind": "added"}),
            ("consider_monorepo_commit", {"sha": "x"}),
            ("consider_domain_state", {"transition": "X"}),
        ],
    )
    def test_drops_when_filter_absent(
        self, method: str, payload: dict[str, Any],
    ) -> None:
        s = _stream()  # No filters at all
        getattr(s, method)(payload)
        assert s._entries == []

    def test_rolling_window_applies_to_new_kinds(self) -> None:
        s = _stream(
            tool_output_filter=_accept_all,
            max_entries=3,
        )
        for i in range(5):
            s.consider_tool_output({"i": i})
        assert len(s._entries) == 3
        assert [e["payload"]["i"] for e in s._entries] == [2, 3, 4]


# ---------------------------------------------------------------------------
# 3. StreamEventSource ABC + concrete sources
# ---------------------------------------------------------------------------


class TestStreamEventSourceABC:
    def test_abstract_method_required(self) -> None:
        with pytest.raises(TypeError):
            StreamEventSource()  # type: ignore[abstract]

    def test_subclass_with_attach_can_instantiate(self) -> None:
        class _X(StreamEventSource):
            async def attach(self, policy: Any) -> None:
                return None
        instance = _X()
        assert isinstance(instance, StreamEventSource)


class TestAccumulatedContextSourceAttach:
    @pytest.mark.asyncio
    async def test_attach_is_no_op(self) -> None:
        """The source records itself for introspection but does not
        install any new hook — the policy's existing event-handler
        post-step loop is the feed."""
        policy = BaseActionPolicy(agent=_agent())
        src = AccumulatedContextSource()
        await src.attach(policy)
        # No new hook to inspect; just confirm no exception.


class TestActionCallSourceAttach:
    @pytest.mark.asyncio
    async def test_attach_is_no_op(self) -> None:
        policy = BaseActionPolicy(agent=_agent())
        src = ActionCallSource()
        await src.attach(policy)


class TestToolResultSourceAttachAndBuild:
    @pytest.mark.asyncio
    async def test_attach_registers_with_policy(self) -> None:
        policy = BaseActionPolicy(agent=_agent())
        src = ToolResultSource()
        await src.attach(policy)
        assert src in policy._tool_result_sources

    def test_build_payload_accepts_dict_shaped_data(self) -> None:
        policy = BaseActionPolicy(agent=_agent())
        src = ToolResultSource()
        action_result = MagicMock()
        action_result.success = True
        action_result.data = {
            "payload": {"x": 1},
            "units": {},
            "provenance": {"tool_name": "t"},
            "extra_field": "ignored",
        }
        payload = src.build_payload(
            action_key="some_action", action_result=action_result,
            policy=policy,
        )
        assert payload is not None
        assert payload["action_key"] == "some_action"
        assert payload["success"] is True
        assert payload["agent_id"] == "agent_substrate_test"
        assert payload["tool_result"]["payload"] == {"x": 1}

    def test_build_payload_accepts_pydantic_model_shape(self) -> None:
        from pydantic import BaseModel
        class _FakeToolResult(BaseModel):
            payload: dict[str, Any]
            units: dict[str, str]
            provenance: dict[str, Any]
        policy = BaseActionPolicy(agent=_agent())
        src = ToolResultSource()
        action_result = MagicMock()
        action_result.success = True
        action_result.data = _FakeToolResult(
            payload={"y": 2}, units={"y": "m"}, provenance={"tool": "fake"},
        )
        payload = src.build_payload(
            action_key="k", action_result=action_result, policy=policy,
        )
        assert payload is not None
        assert payload["tool_result"]["payload"] == {"y": 2}

    def test_build_payload_returns_none_for_unshaped_data(self) -> None:
        policy = BaseActionPolicy(agent=_agent())
        src = ToolResultSource()
        action_result = MagicMock()
        action_result.success = True
        action_result.data = "just a string"
        assert src.build_payload(
            action_key="k", action_result=action_result, policy=policy,
        ) is None

    def test_build_payload_returns_none_when_data_missing(self) -> None:
        policy = BaseActionPolicy(agent=_agent())
        src = ToolResultSource()
        action_result = MagicMock()
        action_result.success = True
        action_result.data = None
        assert src.build_payload(
            action_key="k", action_result=action_result, policy=policy,
        ) is None

    def test_build_payload_returns_none_when_dict_missing_canonical_keys(
        self,
    ) -> None:
        policy = BaseActionPolicy(agent=_agent())
        src = ToolResultSource()
        action_result = MagicMock()
        action_result.success = True
        action_result.data = {"payload": {}, "units": {}}  # missing provenance
        assert src.build_payload(
            action_key="k", action_result=action_result, policy=policy,
        ) is None


# ---------------------------------------------------------------------------
# 4. BaseActionPolicy.record_stream_entry fanout + register_tool_result_source
# ---------------------------------------------------------------------------


class TestRecordStreamEntry:
    def test_fans_to_matching_consider_method(self) -> None:
        s1 = _stream(name="a", tool_output_filter=_accept_all)
        s2 = _stream(name="b", tool_output_filter=_accept_all)
        policy = BaseActionPolicy(
            agent=_agent(), consciousness_streams=[s1, s2],
        )
        policy.record_stream_entry(
            "tool_output", {"action_key": "x", "tool_result": {}, "success": True},
        )
        assert len(s1._entries) == 1
        assert len(s2._entries) == 1
        for s in (s1, s2):
            assert s._entries[0]["kind"] == "tool_output"

    def test_unknown_kind_is_logged_not_raised(self) -> None:
        s = _stream(name="s", event_filter=_accept_all)
        policy = BaseActionPolicy(
            agent=_agent(), consciousness_streams=[s],
        )
        policy.record_stream_entry("unknown_kind", {"x": 1})
        assert s._entries == []

    def test_stream_exception_does_not_poison_others(self) -> None:
        good = _stream(name="good", tool_output_filter=_accept_all)
        bad = MagicMock(spec=ConsciousnessStream)
        bad.name = "bad"
        bad.consider_tool_output = MagicMock(side_effect=RuntimeError("boom"))
        policy = BaseActionPolicy(
            agent=_agent(),
            # Order matters: bad first; the good stream after must
            # still receive the entry.
            consciousness_streams=[bad, good],
        )
        policy.record_stream_entry(
            "tool_output", {"action_key": "x"},
        )
        assert len(good._entries) == 1


class TestRegisterToolResultSource:
    def test_dedupes(self) -> None:
        src = ToolResultSource()
        policy = BaseActionPolicy(agent=_agent())
        policy.register_tool_result_source(src)
        policy.register_tool_result_source(src)
        assert policy._tool_result_sources.count(src) == 1


# ---------------------------------------------------------------------------
# 5. Post-dispatch tool-result fanout in BaseActionPolicy.dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPostDispatchFanout:
    async def _build_policy_with_dispatcher_stub(
        self,
        action_result_data: Any,
        streams: list[ConsciousnessStream],
        sources: list[Any],
    ) -> tuple[BaseActionPolicy, MagicMock]:
        """Construct a policy whose dispatcher returns a stubbed
        ``ActionResult`` carrying ``data=action_result_data``."""
        policy = BaseActionPolicy(
            agent=_agent(), consciousness_streams=streams,
        )
        for src in sources:
            await src.attach(policy)
        # Stub the dispatcher: skip _create_action_dispatcher path by
        # injecting a fake one with an async dispatch method.
        fake_result = MagicMock()
        fake_result.success = True
        fake_result.data = action_result_data
        policy._action_dispatcher = MagicMock()
        policy._action_dispatcher.dispatch = AsyncMock(return_value=fake_result)
        policy._create_action_dispatcher = AsyncMock()
        return policy, fake_result

    async def test_fanout_fires_when_tool_result_source_registered(
        self,
    ) -> None:
        s = _stream(name="s", tool_output_filter=_accept_all)
        src = ToolResultSource()
        policy, fake_result = await self._build_policy_with_dispatcher_stub(
            action_result_data={
                "payload": {"v": 1}, "units": {}, "provenance": {"tool": "t"},
            },
            streams=[s], sources=[src],
        )
        action = MagicMock()
        action.action_type = "compute"
        result = await policy.dispatch(action)
        assert result is fake_result
        # Stream received the tool_output entry.
        assert len(s._entries) == 1
        assert s._entries[0]["kind"] == "tool_output"
        assert s._entries[0]["payload"]["action_key"] == "compute"

    async def test_fanout_skips_when_no_tool_result_source(self) -> None:
        s = _stream(name="s", tool_output_filter=_accept_all)
        policy, _ = await self._build_policy_with_dispatcher_stub(
            action_result_data={
                "payload": {"v": 1}, "units": {}, "provenance": {"tool": "t"},
            },
            streams=[s], sources=[],
        )
        await policy.dispatch(MagicMock(action_type="compute"))
        # No ToolResultSource → no fanout.
        assert s._entries == []

    async def test_fanout_skips_unshaped_result(self) -> None:
        s = _stream(name="s", tool_output_filter=_accept_all)
        src = ToolResultSource()
        policy, _ = await self._build_policy_with_dispatcher_stub(
            action_result_data="not a tool result",
            streams=[s], sources=[src],
        )
        await policy.dispatch(MagicMock(action_type="compute"))
        assert s._entries == []
