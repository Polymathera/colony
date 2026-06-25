"""Tests for the ``wait_for_next_event`` action on
:class:`EventDrivenActionPolicy` — the consume-none idle primitive
the LLM calls from generated code when it has no work to do."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.patterns.actions.policies import (
    EventDrivenActionPolicy,
)
from polymathera.colony.distributed.ray_utils.serving.context import (
    user_execution_context,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _exec_ctx():
    with user_execution_context(
        tenant_id="tenant_test",
        colony_id="colony_test",
        session_id="session_test",
        origin="test",
    ) as ctx:
        yield ctx


class _AlwaysLiveCapability:
    """Test double for an :class:`AgentCapability` whose
    :meth:`is_awaiting_event` always returns ``True``. Used so the
    wait-primitive tests below exercise the queue/timeout mechanics
    without tripping the live-wake-source pre-check
    (:class:`NoLiveWakeSource`) on every call.

    ``capability_key`` is the unique-per-instance identifier the
    live-wake-source log site reads — set on every stub so the
    list-of-tuples log shape (not dict-keyed-by-class-name, which
    would collide on duplicates) has a non-empty key per entry."""

    def __init__(self, capability_key: str = "always_live") -> None:
        self.capability_key = capability_key

    def is_awaiting_event(self) -> bool:
        return True


class _FakeAgent:
    """Minimal agent stub for direct policy-method testing."""

    def __init__(self, agent_id: str = "agent-wait") -> None:
        self.agent_id = agent_id
        self.idle_wait_counter: int = 0
        self.metadata = MagicMock()
        self.metadata.goals = []
        self.metadata.role = "test"
        self.metadata.parameters = {}
        self.metadata.run_id = None
        self.writes: list[tuple[str, dict[str, Any]]] = []
        #: Mount a single always-live capability so the wait-primitive
        #: tests pass the DL2 pre-check. Tests that exercise the
        #: pre-check itself override this directly.
        self.capabilities: list[Any] = [_AlwaysLiveCapability()]

    async def get_blackboard(self, *, scope_id: str | None = None):
        agent = self

        class _BB:
            async def write(
                self,
                key: str,
                value: dict[str, Any],
                *,
                tags=None,
                metadata=None,
            ):
                agent.writes.append((key, value))

        return _BB()

    def get_capabilities(self):
        return self.capabilities


def _make_policy() -> EventDrivenActionPolicy:
    """Build a policy without subscribing capabilities; we test the
    wait action's interaction with the queue pair directly."""
    return EventDrivenActionPolicy(agent=_FakeAgent())


async def test_returns_immediately_when_normal_queue_nonempty() -> None:
    """Already-queued normal event ⇒ wait returns immediately."""
    policy = _make_policy()
    policy._queues.normal.put_nowait("ev")
    result = await asyncio.wait_for(
        policy.wait_for_next_event(), timeout=0.05,
    )
    assert result == {"ok": True, "timed_out": False}
    # Consume-none: event is still queued.
    assert policy._queues.normal.qsize() == 1


async def test_returns_immediately_when_high_queue_nonempty() -> None:
    """Already-queued high-priority event ⇒ wait returns immediately."""
    policy = _make_policy()
    policy._queues.high.put_nowait("ev-h")
    result = await asyncio.wait_for(
        policy.wait_for_next_event(), timeout=0.05,
    )
    assert result == {"ok": True, "timed_out": False}
    assert policy._queues.high.qsize() == 1


async def test_blocks_until_event_arrives_on_normal_lane() -> None:
    """Empty queues ⇒ the wait blocks; arrival wakes it."""
    policy = _make_policy()

    async def delayed_put() -> None:
        await asyncio.sleep(0.03)
        policy._queues.normal.put_nowait("ev")

    asyncio.create_task(delayed_put())
    result = await asyncio.wait_for(
        policy.wait_for_next_event(), timeout=1.0,
    )
    assert result == {"ok": True, "timed_out": False}


async def test_blocks_until_event_arrives_on_high_lane() -> None:
    """High-priority arrival also wakes the wait."""
    policy = _make_policy()

    async def delayed_put_high() -> None:
        await asyncio.sleep(0.03)
        policy._queues.high.put_nowait("ev-h")

    asyncio.create_task(delayed_put_high())
    result = await asyncio.wait_for(
        policy.wait_for_next_event(), timeout=1.0,
    )
    assert result == {"ok": True, "timed_out": False}


async def test_timeout_returns_timed_out_envelope() -> None:
    """No event before deadline ⇒ ``{ok: True, timed_out: True}``."""
    policy = _make_policy()
    result = await policy.wait_for_next_event(timeout_seconds=0.05)
    assert result == {"ok": True, "timed_out": True}


async def test_idle_wait_counter_pair_balanced_on_normal_wakeup() -> None:
    """Counter increments on entry and decrements on exit. Pairs
    with the existing ``idle_wait_counter`` accounting used by
    ``HumanApprovalCapability.get_response``."""
    policy = _make_policy()
    agent = policy.agent
    assert agent.idle_wait_counter == 0

    async def delayed_put() -> None:
        # Confirm counter is incremented while the wait is in
        # progress — sample mid-wait before delivering the event.
        await asyncio.sleep(0.02)
        assert agent.idle_wait_counter == 1
        policy._queues.normal.put_nowait("ev")

    asyncio.create_task(delayed_put())
    await policy.wait_for_next_event()
    assert agent.idle_wait_counter == 0


async def test_idle_wait_counter_balanced_on_timeout() -> None:
    """Counter returns to 0 even when the wait times out."""
    policy = _make_policy()
    agent = policy.agent
    assert agent.idle_wait_counter == 0
    await policy.wait_for_next_event(timeout_seconds=0.02)
    assert agent.idle_wait_counter == 0


async def test_idle_wait_counter_balanced_on_cancellation() -> None:
    """Counter returns to 0 when the wait task is cancelled — the
    /abort path relies on this so the agent's idle accounting does
    not leak when the user aborts an in-flight wait."""
    policy = _make_policy()
    agent = policy.agent
    waiter = asyncio.create_task(policy.wait_for_next_event())
    await asyncio.sleep(0.02)
    assert agent.idle_wait_counter == 1
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert agent.idle_wait_counter == 0


async def test_consume_none_event_is_not_pulled_from_queue() -> None:
    """The wait observes nonempty but does NOT consume; the next
    plan_step's drain (or any other consumer) sees the event
    exactly once."""
    policy = _make_policy()
    policy._queues.normal.put_nowait("ev-1")
    policy._queues.high.put_nowait("ev-h-1")
    await policy.wait_for_next_event()
    assert policy._queues.normal.qsize() == 1
    assert policy._queues.high.qsize() == 1


def test_action_is_discoverable_via_action_executor_decorator() -> None:
    """``ActionDispatcher._create_default_action_executors`` walks
    ``[self.agent, self.action_policy]`` (see ``dispatcher.py:877``)
    and discovers methods decorated with ``@action_executor`` via
    ``_action_input_schema`` / ``_action_output_schema`` attributes.
    Pin that ``wait_for_next_event`` carries those markers so a
    future refactor that drops the decorator surfaces here."""
    method = EventDrivenActionPolicy.wait_for_next_event
    assert callable(method)
    # The decorator at dispatcher.py:751-752 sets these attributes
    # on the wrapped function — they're the canonical signal the
    # dispatcher uses to discover action executors.
    assert hasattr(method, "_action_input_schema")
    assert hasattr(method, "_action_output_schema")


# ---------------------------------------------------------------------------
# DL2: live-wake-source pre-check
#
# Closes the run5 deadlock — a cell that called ``request_human_approval``
# whose input validator raised (no request published on BB), then
# unconditionally called ``wait_for_next_event``. Without the pre-check
# the agent waited forever because no response key would ever be
# written. With the pre-check the wait fails-fast with
# :class:`NoLiveWakeSource`, an :class:`ActionInputViolation` subclass
# the dispatcher re-raises unwrapped to abort the cell.
# ---------------------------------------------------------------------------


class _NeverLiveCapability:
    """Test double whose :meth:`is_awaiting_event` always returns
    ``False`` — represents a request-response capability with no
    outstanding requests AND no continuous subscription."""

    def __init__(self, capability_key: str = "never_live") -> None:
        self.capability_key = capability_key

    def is_awaiting_event(self) -> bool:
        return False


async def test_pre_check_raises_when_no_capability_is_live() -> None:
    from polymathera.colony.agents.patterns.actions.policies import (
        NoLiveWakeSource,
    )

    policy = _make_policy()
    # Override the default _AlwaysLiveCapability with a single
    # never-live one so the pre-check fires.
    policy.agent.capabilities = [_NeverLiveCapability()]
    with pytest.raises(NoLiveWakeSource):
        await policy.wait_for_next_event()


async def test_pre_check_raises_when_no_capabilities_mounted() -> None:
    """An agent with zero capabilities cannot possibly be awaiting
    an event. The empty ``any(...)`` evaluates to False so the
    pre-check fires."""

    from polymathera.colony.agents.patterns.actions.policies import (
        NoLiveWakeSource,
    )

    policy = _make_policy()
    policy.agent.capabilities = []
    with pytest.raises(NoLiveWakeSource):
        await policy.wait_for_next_event()


async def test_pre_check_passes_with_any_one_live_capability() -> None:
    """Mixed mount with one live + several inert capabilities ⇒
    the wait proceeds (and the queue-already-nonempty path returns
    the normal envelope). Confirms the ``any()`` shape."""

    policy = _make_policy()
    policy.agent.capabilities = [
        _NeverLiveCapability(capability_key="never_1"),
        _NeverLiveCapability(capability_key="never_2"),
        _AlwaysLiveCapability(capability_key="always_1"),
    ]
    policy._queues.normal.put_nowait("ev")
    result = await asyncio.wait_for(
        policy.wait_for_next_event(), timeout=0.05,
    )
    assert result == {"ok": True, "timed_out": False}


async def test_pre_check_does_not_collapse_same_class_capabilities(
    caplog,
) -> None:
    """``Agent._capabilities`` is keyed by ``capability_key``, not
    by class — multiple instances of the SAME capability class are
    legitimate (the canonical real-world case is the five
    ``MemoryCapability`` instances mounted on every agent under keys
    ``working`` / ``stm`` / ``ltm:episodic`` / ``ltm:semantic`` /
    ``ltm:procedural``). The live-wake-source log line MUST surface
    each instance independently — a dict-keyed-by-class-name would
    collapse them, hiding which instance was/wasn't live and making
    forensic recovery indirect.

    Pin via two same-class capabilities with different liveness:
    log emits BOTH ``capability_key``s, ``any_live`` is True, the
    wait proceeds."""

    import logging

    policy = _make_policy()
    # Two _NeverLiveCapability instances + one _AlwaysLiveCapability
    # — three same-class duplications-of-shape: two never-live with
    # different keys, one always-live. The log must show all three
    # entries.
    policy.agent.capabilities = [
        _NeverLiveCapability(capability_key="duplicate_class_a"),
        _NeverLiveCapability(capability_key="duplicate_class_b"),
        _AlwaysLiveCapability(capability_key="always_keep_alive"),
    ]
    policy._queues.normal.put_nowait("ev")
    with caplog.at_level(logging.INFO, logger="polymathera.colony"):
        await asyncio.wait_for(
            policy.wait_for_next_event(), timeout=0.05,
        )
    # Find the live_wake_check line.
    live_check_lines = [
        r for r in caplog.records
        if "live_wake_check" in r.getMessage()
    ]
    assert live_check_lines, (
        "live_wake_check log line missing — instrumentation regressed"
    )
    msg = live_check_lines[-1].getMessage()
    # All THREE capability_keys must appear — dict-keyed-by-class
    # would collapse the two _NeverLiveCapability instances into one
    # entry and the log would only show two distinct items, not three.
    assert "duplicate_class_a" in msg
    assert "duplicate_class_b" in msg
    assert "always_keep_alive" in msg


async def test_pre_check_exception_is_action_input_violation_subclass(
) -> None:
    """The dispatcher's two ``except Exception`` blocks were updated
    to re-raise :class:`ActionInputViolation` subclasses unwrapped
    (DL1 surface). :class:`NoLiveWakeSource` inherits from
    ``ActionInputViolation`` so this exception path threads
    through the same cell-abort mechanism, not a wrapped
    ``ActionResult(success=False)``."""

    from polymathera.colony.agents.patterns.actions.dispatcher import (
        ActionInputViolation,
    )
    from polymathera.colony.agents.patterns.actions.policies import (
        NoLiveWakeSource,
    )

    assert issubclass(NoLiveWakeSource, ActionInputViolation)

    policy = _make_policy()
    policy.agent.capabilities = []
    with pytest.raises(ActionInputViolation):
        await policy.wait_for_next_event()


async def test_pre_check_runs_before_idle_wait_counter_increment(
) -> None:
    """The pre-check happens BEFORE the ``idle_wait_counter += 1``
    statement, so a deadlock-detected wait does NOT leak counter
    state. Otherwise the agent's idle accounting would drift on
    every aborted cell."""

    from polymathera.colony.agents.patterns.actions.policies import (
        NoLiveWakeSource,
    )

    policy = _make_policy()
    policy.agent.capabilities = []
    assert policy.agent.idle_wait_counter == 0
    with pytest.raises(NoLiveWakeSource):
        await policy.wait_for_next_event()
    assert policy.agent.idle_wait_counter == 0
