"""Phase 1 tests for the event-priority architecture.

Covers the framework changes that make read-only status queries
(``/status``, ``/whatdoing``) responsive even while the agent's main
planning loop is awaiting a long-running action. See
``colony_docs/markdown/plans/design_event_priority_and_action_interruption.md``.

Each test isolates one layer:

- decorator metadata
- AgentCapability priority routing
- chat router slash-command classification
- get_status_snapshot output shape

The two-queue concurrent reader inside ``EventDrivenActionPolicy``
is exercised through a minimal in-process integration test that
spawns the loop, drops events into both queues, and asserts the
high-priority handler fires while the main loop is *blocked* awaiting
a slow action.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.patterns.events import (
    event_handler,
    EventProcessingResult,
    PROCESSED,
)


# ---------------------------------------------------------------------------
# Decorator metadata
# ---------------------------------------------------------------------------

class TestEventHandlerDecorator:

    def test_default_priority_is_normal(self):
        @event_handler(pattern="x:*")
        async def h(self, event, repl):
            return None

        assert h._is_event_handler is True
        assert h._event_pattern == "x:*"
        assert h._event_priority == "normal"

    def test_explicit_priority_high(self):
        @event_handler(pattern="x:*", priority="high")
        async def h(self, event, repl):
            return None

        assert h._event_priority == "high"

    def test_decorator_without_parens_defaults_normal(self):
        @event_handler
        async def h(self, event, repl):
            return None

        assert h._event_priority == "normal"

    def test_invalid_priority_rejected_at_decoration_time(self):
        with pytest.raises(ValueError, match="priority must be one of"):
            @event_handler(pattern="x:*", priority="urgent")
            async def h(self, event, repl):
                return None


# ---------------------------------------------------------------------------
# AgentCapability priority routing
# ---------------------------------------------------------------------------

class TestCapabilityPriorityPartitioning:
    """Verifies ``high_priority_patterns`` / ``normal_priority_patterns``
    walk the MRO and split by handler priority. We construct minimal
    capability classes inline so we don't depend on the production
    capabilities."""

    def _make_cap_class(self):
        # Local import so the import chain stays light for the
        # decorator-only tests above.
        from polymathera.colony.agents.base import AgentCapability

        class _Cap(AgentCapability):
            @event_handler(pattern="chat:user:*")
            async def normal_one(self, event, repl):
                return None

            @event_handler(pattern="chat:work:*")
            async def normal_two(self, event, repl):
                return None

            @event_handler(pattern="chat:control:*", priority="high")
            async def high_one(self, event, repl):
                return None

            @event_handler(pattern="chat:status:*", priority="high")
            async def high_two(self, event, repl):
                return None

            async def serialize_suspension_state(self, state):
                return state

            async def deserialize_suspension_state(self, state):
                return None

        return _Cap

    def test_partitions_split_handlers_by_priority(self):
        Cap = self._make_cap_class()
        cap = Cap.__new__(Cap)
        # Minimal init: the two properties only need self._input_patterns.
        cap._input_patterns = None
        normal = sorted(cap.normal_priority_patterns)
        high = sorted(cap.high_priority_patterns)
        assert normal == ["chat:user:*", "chat:work:*"]
        assert high == ["chat:control:*", "chat:status:*"]

    def test_explicit_input_patterns_treated_as_normal(self):
        """Backward-compat: when ``input_patterns`` was passed
        explicitly to ``__init__``, the override predates the
        priority lane and stays on the normal queue."""
        Cap = self._make_cap_class()
        cap = Cap.__new__(Cap)
        cap._input_patterns = ["legacy:foo:*"]
        assert cap.normal_priority_patterns == ["legacy:foo:*"]
        assert cap.high_priority_patterns == []


# ---------------------------------------------------------------------------
# stream_events_to_queue dual-queue routing
# ---------------------------------------------------------------------------

class _CaptureBlackboard:
    """Stub blackboard that records every ``stream_events_to_queue``
    call so the test can assert which pattern landed on which queue."""

    def __init__(self, scope_id: str):
        self.scope_id = scope_id
        self.backend_type = "memory"
        self.subscriptions: list[tuple[Any, str, frozenset[str]]] = []

    def stream_events_to_queue(self, queue, *, pattern, event_types):
        self.subscriptions.append(
            (queue, pattern, frozenset(event_types)),
        )


class TestStreamEventsToQueueRouting:

    def _make_cap(self, blackboard: _CaptureBlackboard):
        from polymathera.colony.agents.base import AgentCapability

        class _Cap(AgentCapability):
            @event_handler(pattern="normal:*")
            async def normal(self, event, repl):
                return None

            @event_handler(pattern="urgent:*", priority="high")
            async def urgent(self, event, repl):
                return None

            async def get_blackboard(self, **_):
                return blackboard

            async def serialize_suspension_state(self, state):
                return state

            async def deserialize_suspension_state(self, state):
                return None

        cap = _Cap.__new__(_Cap)
        cap._input_patterns = None
        cap._agent = None
        cap._blackboard = blackboard
        return cap

    def test_with_high_queue_routes_split(self):
        bb = _CaptureBlackboard("scope:x")
        cap = self._make_cap(bb)
        normal_q: asyncio.Queue = asyncio.Queue()
        high_q: asyncio.Queue = asyncio.Queue()
        asyncio.get_event_loop().run_until_complete(
            cap.stream_events_to_queue(normal_q, high_priority_queue=high_q)
        )
        # One subscription per pattern; no double subscription.
        normal_subs = [s for s in bb.subscriptions if s[1] == "normal:*"]
        high_subs = [s for s in bb.subscriptions if s[1] == "urgent:*"]
        assert len(normal_subs) == 1 and normal_subs[0][0] is normal_q
        assert len(high_subs) == 1 and high_subs[0][0] is high_q

    def test_without_high_queue_falls_back_to_normal(self):
        """Backward compatibility: callers that pass only one queue
        still work — high-priority patterns route into the normal
        queue (degraded but functional)."""
        bb = _CaptureBlackboard("scope:y")
        cap = self._make_cap(bb)
        normal_q: asyncio.Queue = asyncio.Queue()
        asyncio.get_event_loop().run_until_complete(
            cap.stream_events_to_queue(normal_q)
        )
        for _, pattern, _ in bb.subscriptions:
            # Only one queue was provided; both patterns land on it.
            pass
        targets = {s[1]: s[0] for s in bb.subscriptions}
        assert targets["normal:*"] is normal_q
        assert targets["urgent:*"] is normal_q


# ---------------------------------------------------------------------------
# Concurrent high-priority reader (in-process)
# ---------------------------------------------------------------------------

@dataclass
class _FakeEvent:
    """Minimal stand-in for ``BlackboardEvent`` (the production class
    is a Pydantic model with extra fields the loop doesn't need)."""

    key: str
    value: dict


class TestHighPriorityLoop:

    def test_high_priority_event_processed_while_main_loop_blocked(self):
        """End-to-end Phase 1 invariant: a high-priority event lands
        in the high queue and the dedicated reader processes it even
        while the main loop is suspended awaiting a slow action.

        Construction: we instantiate ``EventDrivenActionPolicy``
        directly with a stub agent. ``initialize`` spawns the
        concurrent reader. We then put events on both queues and
        assert the high handler runs while the simulated long action
        is still awaiting.
        """
        from polymathera.colony.agents.patterns.actions.policies import (
            EventDrivenActionPolicy,
        )

        # Track which handlers fired and on which task.
        fired: list[tuple[str, str]] = []  # (priority, key)

        # Use a real capability whose handlers self-register.
        from polymathera.colony.agents.base import AgentCapability

        class _Cap(AgentCapability):
            @event_handler(pattern="control:*", priority="high")
            async def on_control(self, event, _repl):
                fired.append(("high", event.key))
                return PROCESSED

            @event_handler(pattern="user:*")
            async def on_user(self, event, _repl):
                fired.append(("normal", event.key))
                return PROCESSED

            async def serialize_suspension_state(self, state):
                return state

            async def deserialize_suspension_state(self, state):
                return None

        async def run():
            # ``spec=`` restricts MagicMock auto-attribute creation so
            # ``_get_object_event_handlers``'s ``dir(cap)`` walk doesn't
            # pick the agent itself up as an event handler (a MagicMock
            # without ``spec`` falsely satisfies the
            # ``hasattr(method, '_is_event_handler')`` filter).
            agent = MagicMock(spec=["agent_id", "get_capabilities"])
            agent.agent_id = "agent-test"
            agent.get_capabilities = MagicMock(return_value=[])

            policy = EventDrivenActionPolicy.__new__(
                EventDrivenActionPolicy,
            )
            # Bypass normal init/dispatcher setup — we are unit-
            # testing the high-priority loop in isolation. Both
            # ``self._agent`` (used by AgentCapability semantics)
            # and ``self.agent`` (a plain attribute set by
            # ``ActionPolicy.__init__``) need to point at the agent
            # because internal helpers reference both.
            policy._agent = agent
            policy.agent = agent
            policy._action_map = None
            policy._action_providers = []
            policy.io = None
            policy._action_dispatcher = None
            policy._event_queue = asyncio.Queue()
            policy._high_priority_event_queue = asyncio.Queue()
            policy._high_priority_task = None
            policy._high_priority_restarts = 0
            policy._subscribed_callbacks = []
            policy._subscribed_providers = set()
            policy._consciousness_streams = []

            cap = _Cap.__new__(_Cap)
            cap._input_patterns = None
            cap._agent = agent
            cap._blackboard = None
            agent.get_capabilities = MagicMock(return_value=[cap])

            # Spawn the concurrent reader directly (bypassing
            # ``initialize``'s subscription path which needs a live
            # blackboard we don't want here).
            policy._high_priority_task = asyncio.create_task(
                policy._run_high_priority_loop(),
            )

            try:
                # Simulate the main loop: BLOCK awaiting a slow
                # action. Meanwhile, drop a high-priority event.
                async def slow_action():
                    await asyncio.sleep(0.5)

                main = asyncio.create_task(slow_action())

                # Drop both kinds of events. The normal one will sit
                # in the queue (no main-loop reader in this test);
                # the high one MUST be processed by the concurrent
                # reader while ``main`` is still awaiting.
                await policy._event_queue.put(
                    _FakeEvent("user:hello", {})
                )
                await policy._high_priority_event_queue.put(
                    _FakeEvent("control:status", {})
                )

                # Give the reader a tick to consume.
                await asyncio.sleep(0.1)

                # The slow action is still running — we have NOT
                # awaited it yet.
                assert not main.done()
                # But the high handler MUST have fired.
                assert ("high", "control:status") in fired
                # And the normal handler must NOT have fired (no main
                # loop reader in this isolated test).
                assert ("normal", "user:hello") not in fired

                await main
            finally:
                policy._high_priority_task.cancel()
                try:
                    await policy._high_priority_task
                except asyncio.CancelledError:
                    pass

        asyncio.get_event_loop().run_until_complete(run())

    def test_handler_immediate_action_is_ignored_with_warning(self, caplog):
        """High-priority handlers MUST NOT dispatch actions. If one
        returns an immediate_action, the loop logs a WARNING and
        discards it."""
        import logging
        from polymathera.colony.agents.patterns.actions.policies import (
            EventDrivenActionPolicy,
        )
        from polymathera.colony.agents.base import AgentCapability
        from polymathera.colony.agents.models import Action

        class _Cap(AgentCapability):
            @event_handler(pattern="bad:*", priority="high")
            async def bad_handler(self, event, _repl):
                return EventProcessingResult(
                    immediate_action=Action(
                        action_id="x", agent_id="x",
                        action_type="something",
                    ),
                )

            async def serialize_suspension_state(self, state):
                return state

            async def deserialize_suspension_state(self, state):
                return None

        async def run():
            # See the sibling test for the rationale on ``spec=`` —
            # without it, MagicMock auto-fakes ``_is_event_handler``
            # and gets misclassified as an event handler.
            agent = MagicMock(spec=["agent_id", "get_capabilities"])
            agent.agent_id = "agent-test"
            agent.get_capabilities = MagicMock(return_value=[])

            policy = EventDrivenActionPolicy.__new__(EventDrivenActionPolicy)
            policy._agent = agent
            policy.agent = agent
            policy._action_map = None
            policy._action_providers = []
            policy.io = None
            policy._action_dispatcher = None
            policy._event_queue = asyncio.Queue()
            policy._high_priority_event_queue = asyncio.Queue()
            policy._high_priority_task = None
            policy._high_priority_restarts = 0
            policy._subscribed_callbacks = []
            policy._subscribed_providers = set()
            policy._consciousness_streams = []

            cap = _Cap.__new__(_Cap)
            cap._input_patterns = None
            cap._agent = agent
            agent.get_capabilities = MagicMock(return_value=[cap])

            policy._high_priority_task = asyncio.create_task(
                policy._run_high_priority_loop(),
            )

            with caplog.at_level(logging.WARNING):
                try:
                    await policy._high_priority_event_queue.put(
                        _FakeEvent("bad:thing", {})
                    )
                    await asyncio.sleep(0.1)
                finally:
                    policy._high_priority_task.cancel()
                    try:
                        await policy._high_priority_task
                    except asyncio.CancelledError:
                        pass

            assert any(
                "read-only by contract" in rec.message
                for rec in caplog.records
            )

        asyncio.get_event_loop().run_until_complete(run())


# ---------------------------------------------------------------------------
# get_status_snapshot
# ---------------------------------------------------------------------------

class TestStatusSnapshot:

    def test_base_snapshot_has_identity_only(self):
        from polymathera.colony.agents.patterns.actions.policies import (
            BaseActionPolicy,
        )
        from polymathera.colony.agents.patterns.actions.llm_failure_backoff import (
            LLMFailureBackoff,
        )
        agent = MagicMock(); agent.agent_id = "a1"
        agent.metadata.idle_wait_counter = 0
        p = BaseActionPolicy.__new__(BaseActionPolicy)
        p.agent = agent
        p._llm_failure_backoff = LLMFailureBackoff(agent)
        snap = p.get_status_snapshot()
        assert snap["agent_id"] == "a1"
        assert snap["policy_class"] == "BaseActionPolicy"
        assert snap["llm_failure_backoff"]["in_backoff_streak"] is False

    def test_event_driven_snapshot_includes_queue_depth(self):
        from polymathera.colony.agents.patterns.actions.policies import (
            EventDrivenActionPolicy,
        )
        from polymathera.colony.agents.patterns.actions.llm_failure_backoff import (
            LLMFailureBackoff,
        )
        agent = MagicMock(); agent.agent_id = "a2"
        agent.metadata.idle_wait_counter = 0
        p = EventDrivenActionPolicy.__new__(EventDrivenActionPolicy)
        p.agent = agent
        p._llm_failure_backoff = LLMFailureBackoff(agent)
        p._event_queue = asyncio.Queue()
        p._high_priority_event_queue = asyncio.Queue()
        p._high_priority_task = None
        p._consciousness_streams = []
        snap = p.get_status_snapshot()
        assert snap["agent_id"] == "a2"
        assert snap["policy_class"] == "EventDrivenActionPolicy"
        assert snap["queue_depth_normal"] == 0
        assert snap["queue_depth_high"] == 0
        assert snap["high_priority_loop_running"] is False
        assert snap["has_pending_work"] is False


# ---------------------------------------------------------------------------
# Chat router slash-command classification
# ---------------------------------------------------------------------------

class TestChatRouterClassification:

    def test_slash_status_classified_as_control(self):
        from polymathera.colony.web_ui.backend.routers.chat import (
            _is_control_command, _extract_command,
        )
        assert _is_control_command("/status") is True
        assert _extract_command("/status") == "/status"

    def test_slash_with_args_still_classified(self):
        from polymathera.colony.web_ui.backend.routers.chat import (
            _is_control_command,
        )
        assert _is_control_command("/abort now please") is True

    def test_case_insensitive(self):
        from polymathera.colony.web_ui.backend.routers.chat import (
            _is_control_command,
        )
        assert _is_control_command("/STATUS") is True
        assert _is_control_command("/Cancel") is True

    def test_unknown_slash_command_stays_normal(self):
        from polymathera.colony.web_ui.backend.routers.chat import (
            _is_control_command,
        )
        # /help is a *normal*-priority rule-based command — handled
        # by the existing _handle_command path on the main loop.
        assert _is_control_command("/help") is False
        assert _is_control_command("/agents") is False

    def test_plain_message_not_classified(self):
        from polymathera.colony.web_ui.backend.routers.chat import (
            _is_control_command, _extract_command,
        )
        assert _is_control_command("hello") is False
        assert _extract_command("hello") is None
        assert _extract_command(" / ") is None  # space after slash

    def test_chat_protocol_control_keys_are_distinct_from_user_keys(self):
        from polymathera.colony.web_ui.backend.chat.chat_protocol import (
            SessionChatProtocol,
        )
        ctl = SessionChatProtocol.control_message_key("m1")
        usr = SessionChatProtocol.user_message_key("m1")
        assert ctl != usr
        assert ctl.startswith("chat:control:")
        assert usr.startswith("chat:user:")
        # Pattern subscription must be specific enough to not catch
        # both.
        import fnmatch
        ctl_pat = SessionChatProtocol.control_message_pattern()
        assert fnmatch.fnmatch(ctl, ctl_pat) is True
        assert fnmatch.fnmatch(usr, ctl_pat) is False
