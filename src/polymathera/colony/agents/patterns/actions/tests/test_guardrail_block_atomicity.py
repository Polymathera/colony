"""Tests for the guardrail-block atomicity contract on
:meth:`CodeGenerationActionPolicy._handle_guardrail_block`.

The LLM's code block is a sequenced plan the framework MUST execute
as a unit or reject as a unit. The prior behavior — returning
``ActionResult(success=False)`` from a blocked ``run()`` call — let
suffix calls in the same cell fire with their preconditions skipped.
The most visible regression: a blocked ``respond_to_user`` followed
by ``wait_for_next_event`` in the same cell put the agent into an
idle wait without ever responding to the user. The chat went silent;
the trace UI showed only the wait; no advisory reached the user.

These tests pin the four invariants that together implement the
atomicity contract:

1. The handler RAISES :class:`GuardrailBlockedError` so the cell's
   await site unwinds and suffix actions never fire.
2. ``BlockedDispatch`` + run-call-trace + block-streak state is
   recorded BEFORE the raise so the next iteration's planner prompt
   surfaces the rejection.
3. Lifecycle events fire on block (``policy:action_started`` +
   ``policy:action_completed`` with ``blocked=True``) so the trace
   UI shows the rejection as a real failed-action row instead of
   an invisible gap — silent failure is the bug we're fixing.
4. ``emits_lifecycle=False`` opt-outs still hold for blocks; the
   trace UI never gains a row for an action whose executor opted
   out of lifecycle emission.
"""

from __future__ import annotations

from typing import Any

import pytest

from polymathera.colony.agents.patterns.actions.code_constraints import (
    GuardrailDecision,
    RuntimeGuardrail,
)
from polymathera.colony.agents.patterns.actions.code_generation import (
    CodeGenerationActionPolicy,
)
from polymathera.colony.agents.patterns.planning.models import (
    BlockedDispatch,
    GuardrailBlockedError,
)
from polymathera.colony.distributed.ray_utils.serving.context import (
    user_execution_context,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _exec_ctx():
    """``BlockStreakTracker.track`` writes to a scoped blackboard and
    needs an active execution context to resolve the scope. Every
    test in this module exercises the track path, so set the context
    once."""

    with user_execution_context(
        tenant_id="tenant_test",
        colony_id="colony_test",
        session_id="session_test",
        origin="test",
    ) as ctx:
        yield ctx


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeBlackboard:
    """Captures every ``write(key, payload)`` call so tests can
    inspect lifecycle events and diagnostic emissions."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, dict[str, Any]]] = []

    async def write(
        self,
        key: str,
        value: dict[str, Any],
        *,
        tags: Any = None,
        metadata: Any = None,
    ) -> None:
        self.writes.append((key, value))


class _FakeAgent:
    """Minimal agent surface the policy's block-handler reads:
    ``agent_id`` (for lifecycle event payloads) and
    ``get_blackboard`` (for lifecycle emission + streak tracker)."""

    def __init__(self) -> None:
        self.agent_id = "agent-test"
        self.bb = _FakeBlackboard()

    async def get_blackboard(self, *, scope_id: str | None = None):
        return self.bb


class _FakeExecutor:
    """Surfaces only the ``emits_lifecycle`` attribute the block
    handler reads. Mirrors the real ``MethodWrapperActionExecutor``
    /``FunctionWrapperActionExecutor`` flag."""

    def __init__(self, *, emits_lifecycle: bool = True) -> None:
        self.emits_lifecycle = emits_lifecycle


class _FakeDispatcher:
    """Stubs ``find_executor`` so the handler's
    ``emits_lifecycle`` gate has a real value to read."""

    def __init__(
        self, executor_by_key: dict[str, _FakeExecutor] | None = None,
    ) -> None:
        self.executor_by_key = executor_by_key or {}

    def find_executor(self, action_key: str) -> _FakeExecutor | None:
        return self.executor_by_key.get(action_key)


class _AlwaysBlockGuardrail(RuntimeGuardrail):
    """Used to construct a policy whose runtime guardrail always
    blocks. The handler is exercised directly, but the policy's
    constructor still requires a guardrail."""

    async def check(self, action_key, params, call_history):
        return GuardrailDecision(
            allowed=False,
            reason="test block reason",
            suggestion="test suggestion",
        )


def _build_policy(
    *,
    executor_by_key: dict[str, _FakeExecutor] | None = None,
) -> tuple[CodeGenerationActionPolicy, _FakeAgent, _FakeDispatcher]:
    """Construct a policy + minimal deps for direct handler exercise."""

    agent = _FakeAgent()
    # The constructor reads ``agent.metadata.action_policy_config``
    # AND (PR2) ``agent.metadata.lifecycle_mode`` for the
    # effective_loop_max_iterations bypass. Wire both on the
    # minimal fake. Default to ONE_SHOT here so the cap-check
    # branch behaves like a production coordinator; tests that
    # care about CONTINUOUS lifecycle override before constructing
    # the policy.
    from polymathera.colony.agents.models import LifecycleMode
    agent.metadata = type("M", (), {})()
    agent.metadata.action_policy_config = {}
    agent.metadata.lifecycle_mode = LifecycleMode.ONE_SHOT

    policy = CodeGenerationActionPolicy(
        agent=agent,
        runtime_guardrail=_AlwaysBlockGuardrail(),
    )
    dispatcher = _FakeDispatcher(executor_by_key=executor_by_key)
    policy._action_dispatcher = dispatcher
    return policy, agent, dispatcher


def _block_decision() -> GuardrailDecision:
    return GuardrailDecision(
        allowed=False,
        reason="requires a recent 'get_agent_status' call",
        suggestion="Call get_agent_status before respond_to_user.",
    )


# ---------------------------------------------------------------------------
# 1. Atomic abort: handler RAISES the typed exception
# ---------------------------------------------------------------------------


async def test_handle_block_raises_guardrail_blocked_error() -> None:
    """Foundation of atomicity: the handler unwinds the cell at the
    await site so the cell's suffix actions can't fire. Pinned with
    ``pytest.raises`` since a regression to ``return ActionResult(...)``
    would silently re-open the gap that caused the silent
    ``respond_to_user`` → ``wait_for_next_event`` deadlock."""

    policy, _, _ = _build_policy()
    with pytest.raises(GuardrailBlockedError):
        await policy._handle_guardrail_block(
            action_key="X.gated",
            params={"k": "v"},
            decision=_block_decision(),
            repl_ns={},
        )


async def test_guardrail_blocked_error_exposes_typed_fields() -> None:
    """The exception carries the action_key, reason, and suggestion as
    typed attributes — not just buried in the message string — so
    downstream handlers (logging, retry, advisories) can structure
    their response without re-parsing the message."""

    policy, _, _ = _build_policy()
    decision = _block_decision()
    with pytest.raises(GuardrailBlockedError) as exc_info:
        await policy._handle_guardrail_block(
            action_key="SessionOrchestratorCapability.respond_to_user",
            params={"content": "..."},
            decision=decision,
            repl_ns={},
        )
    e = exc_info.value
    assert e.action_key == "SessionOrchestratorCapability.respond_to_user"
    assert e.reason == decision.reason
    assert e.suggestion == decision.suggestion
    # And the str message includes all three for human inspection.
    assert decision.reason in str(e)
    assert decision.suggestion in str(e)


async def test_guardrail_blocked_error_message_omits_empty_suggestion() -> None:
    """A guardrail may emit a block with no suggestion; the message
    must not render a dangling ``" "`` separator. Mirrors the
    existing ``BlockedDispatch`` rendering invariant."""

    policy, _, _ = _build_policy()
    decision = GuardrailDecision(
        allowed=False,
        reason="just no",
        suggestion="",
    )
    with pytest.raises(GuardrailBlockedError) as exc_info:
        await policy._handle_guardrail_block(
            action_key="X.gated",
            params={},
            decision=decision,
            repl_ns={},
        )
    e = exc_info.value
    assert e.suggestion == ""
    # The message ends with "just no." with no trailing space.
    assert str(e).endswith("just no.")


# ---------------------------------------------------------------------------
# 2. Recording happens BEFORE the raise so next iteration sees it
# ---------------------------------------------------------------------------


async def test_handle_block_appends_blocked_dispatch_before_raising() -> None:
    """The next-iteration planner-prompt advisory depends on
    ``_last_blocked_dispatches`` being populated. If the raise
    unwound BEFORE the append (an ordering bug), the LLM would see
    "your cell failed with GuardrailBlockedError" but no
    ``BlockedDispatch`` advisory — no suggestion, no recovery
    guidance, and the same block would repeat."""

    policy, _, _ = _build_policy()
    assert policy._last_blocked_dispatches == []
    decision = _block_decision()

    with pytest.raises(GuardrailBlockedError):
        await policy._handle_guardrail_block(
            action_key="SessionOrchestratorCapability.respond_to_user",
            params={"content": "agent-abc is running."},
            decision=decision,
            repl_ns={},
        )

    # The raise didn't unwind the append — the entry is here for
    # the next iteration's prompt build.
    assert len(policy._last_blocked_dispatches) == 1
    bd = policy._last_blocked_dispatches[0]
    assert isinstance(bd, BlockedDispatch)
    assert bd.action_key == "SessionOrchestratorCapability.respond_to_user"
    assert bd.reason == decision.reason
    assert bd.suggestion == decision.suggestion
    assert bd.params_preview == {"content": "agent-abc is running."}


async def test_handle_block_appends_run_call_trace_before_raising() -> None:
    """The trace entry must carry ``blocked=True`` so the existing
    ``RunCallTrace.successful_calls_to`` /
    ``test_view_validates_guardrail_blocked_entry`` invariants
    continue to hold. Also ensures the next-iteration prompt's
    run-call-trace section shows the block."""

    policy, _, _ = _build_policy()
    repl_ns: dict[str, Any] = {}
    with pytest.raises(GuardrailBlockedError):
        await policy._handle_guardrail_block(
            action_key="X.gated",
            params={"k": "v"},
            decision=_block_decision(),
            repl_ns=repl_ns,
        )

    assert len(policy._run_call_trace) == 1
    entry = policy._run_call_trace[0]
    assert entry["action_key"] == "X.gated"
    assert entry["success"] is False
    assert entry["blocked"] is True
    assert entry["parameters"] == {"k": "v"}
    # The REPL namespace's mirror is updated so the cell can see
    # the trace if it inspects ``_run_call_trace`` after a try/except
    # (defensive observability — not a primary surface, but pinned
    # because the existing code path always mirrored).
    assert repl_ns["_run_call_trace"] is policy._run_call_trace


async def test_handle_block_tracks_streak_before_raising() -> None:
    """Repeated blocks of the SAME action_key are a signal that the
    LLM is stuck; the streak tracker fires
    :data:`DIAGNOSTIC_GUARDRAIL_BLOCK_STREAK` so observers can
    intervene (e.g. plan_b prompt, hint advisory). Three blocks at
    the default threshold → one diagnostic write."""

    from polymathera.colony.agents.blackboard.protocol import (
        AgentDiagnosticProtocol,
        DIAGNOSTIC_GUARDRAIL_BLOCK_STREAK,
    )

    policy, agent, _ = _build_policy()
    for _ in range(3):
        with pytest.raises(GuardrailBlockedError):
            await policy._handle_guardrail_block(
                action_key="X.gated",
                params={},
                decision=_block_decision(),
                repl_ns={},
            )

    streak_writes = [
        (k, v) for k, v in agent.bb.writes
        if k.startswith("agent:diagnostic:")
    ]
    assert len(streak_writes) == 1
    parsed = AgentDiagnosticProtocol.parse_event_key(streak_writes[0][0])
    assert parsed["kind"] == DIAGNOSTIC_GUARDRAIL_BLOCK_STREAK
    assert streak_writes[0][1]["action_key"] == "X.gated"


# ---------------------------------------------------------------------------
# 3. Lifecycle events fire on block — visibility in the trace UI
# ---------------------------------------------------------------------------


async def test_handle_block_emits_started_and_completed_lifecycle_events() -> None:
    """The trace UI subscribes to ``policy:action_started:*`` /
    ``policy:action_completed:*``. Without these on a block, the
    blocked action was invisible: no trace row, no chat write, just
    a log warning the user never sees. The fix pairs both events
    with ``blocked=True`` in the payload so subscribers can render
    the rejection distinctly from a downstream dispatch failure."""

    policy, agent, _ = _build_policy(
        executor_by_key={
            "X.gated": _FakeExecutor(emits_lifecycle=True),
        },
    )
    decision = _block_decision()
    with pytest.raises(GuardrailBlockedError):
        await policy._handle_guardrail_block(
            action_key="X.gated",
            params={"k": "v"},
            decision=decision,
            repl_ns={},
        )

    started = [
        (k, v) for k, v in agent.bb.writes
        if k.startswith("policy:action_started:")
    ]
    completed = [
        (k, v) for k, v in agent.bb.writes
        if k.startswith("policy:action_completed:")
    ]
    assert len(started) == 1
    assert len(completed) == 1

    # Both events refer to the SAME synthetic action_id so a
    # subscriber that pairs started/completed by id (the chat-UI's
    # action banner does this) closes the row cleanly.
    sk, sp = started[0]
    ck, cp = completed[0]
    assert sp["action_id"] == cp["action_id"]
    assert sp["action_key"] == "X.gated"
    assert cp["action_key"] == "X.gated"
    assert sp["blocked"] is True
    assert cp["blocked"] is True
    # The completed event carries the failure semantics so existing
    # subscribers that filter on ``success is False`` still surface
    # the rejection.
    assert cp["success"] is False
    assert cp["cancelled"] is False
    assert decision.reason in cp["error"]
    assert cp["suggestion"] == decision.suggestion


async def test_handle_block_skips_lifecycle_when_emits_lifecycle_false() -> None:
    """The ``emits_lifecycle=False`` opt-out (used by idle waits and
    publish-only narrative emits) must still hold under a block —
    those actions don't suddenly appear in the trace UI just
    because they were blocked. Mirrors the dispatch path's gate at
    [code_generation.py: emits_lifecycle check]."""

    policy, agent, _ = _build_policy(
        executor_by_key={
            "X.idle_wait": _FakeExecutor(emits_lifecycle=False),
        },
    )
    with pytest.raises(GuardrailBlockedError):
        await policy._handle_guardrail_block(
            action_key="X.idle_wait",
            params={},
            decision=_block_decision(),
            repl_ns={},
        )

    started = [
        k for k, _ in agent.bb.writes
        if k.startswith("policy:action_started:")
    ]
    completed = [
        k for k, _ in agent.bb.writes
        if k.startswith("policy:action_completed:")
    ]
    assert started == []
    assert completed == []
    # The BlockedDispatch advisory still fires — opt-out is for
    # the UI banner only, not for the LLM-facing recovery surface.
    assert len(policy._last_blocked_dispatches) == 1


async def test_handle_block_unknown_executor_defaults_to_emitting() -> None:
    """``find_executor`` returning ``None`` for synthetic /
    test-injected actions defaults to "emit by default" — the
    conservative choice that avoids silent suppression for
    actions the dispatcher doesn't know about. Mirrors the same
    invariant in the dispatch path."""

    policy, agent, _ = _build_policy(
        executor_by_key={},  # unknown to find_executor
    )
    with pytest.raises(GuardrailBlockedError):
        await policy._handle_guardrail_block(
            action_key="X.never_registered",
            params={},
            decision=_block_decision(),
            repl_ns={},
        )

    started = [
        k for k, _ in agent.bb.writes
        if k.startswith("policy:action_started:")
    ]
    assert len(started) == 1


# ---------------------------------------------------------------------------
# 4. Truncation of giant params_preview survives the new path
# ---------------------------------------------------------------------------


async def test_handle_block_truncates_giant_params_preview() -> None:
    """A blocked dispatch may carry an enormous proposal (e.g. a
    full decomposition tree). The capture truncates to
    ``BLOCKED_DISPATCH_PARAMS_PREVIEW_BYTES`` so the in-memory
    history stays bounded across long-running coordinators. The
    refactor preserved this invariant; pinned here so a future
    rewrite doesn't drop it."""

    from polymathera.colony.agents.patterns.planning.models import (
        BLOCKED_DISPATCH_PARAMS_PREVIEW_BYTES,
    )

    policy, _, _ = _build_policy()
    giant = "x" * (BLOCKED_DISPATCH_PARAMS_PREVIEW_BYTES * 4)
    with pytest.raises(GuardrailBlockedError):
        await policy._handle_guardrail_block(
            action_key="X.gated",
            params={"payload": giant},
            decision=_block_decision(),
            repl_ns={},
        )

    bd = policy._last_blocked_dispatches[0]
    # The JSON-truncation strategy may produce either a parseable
    # subset (dict) or a string fallback; both must be bounded by
    # the cap.
    if isinstance(bd.params_preview, dict):
        size = len(str(bd.params_preview))
    else:
        size = len(bd.params_preview)
    assert size <= BLOCKED_DISPATCH_PARAMS_PREVIEW_BYTES
