"""Phase 2 + 3 tests for the action-interruption + /replace flow.

Covers:

- ``ActionResult.cancelled`` field shape.
- ``@action_executor(interruptible=True)`` metadata + wrapper
  propagation.
- ``ActionDispatcher.cancel_current_action()`` —
    - cancels an in-flight interruptible action and surfaces
      ``ActionResult(cancelled=True)``;
    - is a no-op (returns False) when no interruptible action is in
      flight or the in-flight one is non-interruptible;
    - distinguishes user-cancellation from outer-cancellation via
      the ``_cancellation_requested`` flag (outer cancel re-raises).
- ``BaseActionPolicy.abort_current()`` default delegates.
- ``CodeGenerationActionPolicy.abort_current()`` resets recovery
  state AND cancels in-flight codegen LLM call AND delegates.
- REPL ``execute()`` re-raises ``CancelledError`` (does not swallow
  it into a "success: False" result).
- Chat router classifies ``/abort``, ``/cancel``, ``/replace`` as
  high-priority commands.
- ``SessionOrchestratorCapability`` /abort handler calls the policy's
  ``abort_current``.
- ``SessionOrchestratorCapability`` /replace handler aborts and
  re-posts the request body to the normal lane.

The tests bypass ``__init__`` on production policy classes (using
``__new__``) and inject the minimum state each method actually reads,
mirroring the pattern already used in ``test_event_priority.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# ActionResult.cancelled field
# ---------------------------------------------------------------------------


class TestActionResultCancelledField:

    def test_default_is_false(self):
        from polymathera.colony.agents.models import ActionResult
        r = ActionResult(success=True)
        assert r.cancelled is False

    def test_explicit_true_round_trips(self):
        from polymathera.colony.agents.models import ActionResult
        r = ActionResult(
            success=False, completed=True, cancelled=True,
            error="Action cancelled: /abort",
            metadata={"cancellation_reason": "/abort"},
        )
        assert r.cancelled is True
        assert r.success is False
        assert r.completed is True
        # Round-trip via model_dump preserves the field.
        d = r.model_dump()
        assert d["cancelled"] is True


# ---------------------------------------------------------------------------
# action_executor decorator + wrapper propagation
# ---------------------------------------------------------------------------


class TestInterruptibleDecoratorMetadata:

    def test_default_interruptible_false(self):
        from polymathera.colony.agents.patterns.actions import action_executor

        @action_executor()
        async def fn(self):
            return None

        assert fn._action_interruptible is False

    def test_explicit_interruptible_true(self):
        from polymathera.colony.agents.patterns.actions import action_executor

        @action_executor(interruptible=True)
        async def fn(self):
            return None

        assert fn._action_interruptible is True

    def test_method_wrapper_propagates_interruptible(self):
        from polymathera.colony.agents.patterns.actions import action_executor
        from polymathera.colony.agents.patterns.actions.dispatcher import (
            MethodWrapperActionExecutor,
        )

        class _Holder:
            @action_executor(interruptible=True)
            async def go(self) -> None:
                return None

        h = _Holder()
        ex = MethodWrapperActionExecutor(
            object=h,
            method=_Holder.go,
            action_key="go",
            interruptible=getattr(_Holder.go, "_action_interruptible", False),
        )
        assert ex.interruptible is True

    def test_function_wrapper_propagates_interruptible(self):
        from polymathera.colony.agents.patterns.actions import action_executor
        from polymathera.colony.agents.patterns.actions.dispatcher import (
            FunctionWrapperActionExecutor,
        )

        @action_executor(interruptible=True)
        async def standalone() -> None:
            return None

        ex = FunctionWrapperActionExecutor(
            func=standalone,
            action_key="standalone",
            agent=MagicMock(),
            interruptible=getattr(standalone, "_action_interruptible", False),
        )
        assert ex.interruptible is True


# ---------------------------------------------------------------------------
# ActionDispatcher cancellable dispatch
# ---------------------------------------------------------------------------


def _make_dispatcher(executor_obj=None):
    """Construct an ActionDispatcher without firing its full ``__init__``
    machinery (which would scan capabilities, build action maps, etc.).

    Only the cancellation surfaces are exercised here; the executor is
    swapped in via a single ``ActionGroup`` by the calling test.
    """
    from polymathera.colony.agents.patterns.actions.dispatcher import (
        ActionDispatcher,
    )
    agent = MagicMock(); agent.agent_id = "agent-test"
    policy = MagicMock()
    d = ActionDispatcher.__new__(ActionDispatcher)
    d.agent = agent
    d.action_policy = policy
    d.action_map = []
    d.action_providers = []
    d._repl = None
    d._repl_discovered = True
    d._current_task = None
    d._current_action_id = None
    d._cancellation_requested = False
    d._cancellation_reason = None
    return d


class TestDispatcherCancellation:

    @pytest.mark.asyncio
    async def test_cancel_with_no_inflight_returns_false(self):
        d = _make_dispatcher()
        assert d.cancel_current_action(reason="/abort") is False
        assert d.has_interruptible_action_in_flight is False

    @pytest.mark.asyncio
    async def test_interruptible_action_cancelled_returns_cancelled_result(self):
        from polymathera.colony.agents.models import Action, ActionType
        from polymathera.colony.agents.patterns.actions import action_executor
        from polymathera.colony.agents.patterns.actions.dispatcher import (
            ActionGroup, MethodWrapperActionExecutor,
        )

        slow_started = asyncio.Event()

        class _Cap:
            @action_executor(interruptible=True)
            async def slow(self) -> str:
                slow_started.set()
                # Long enough that the test can land an abort first.
                await asyncio.sleep(60)
                return "done"

        cap = _Cap()
        ex = MethodWrapperActionExecutor(
            object=cap,
            method=_Cap.slow,
            action_key="cap.slow",
            interruptible=True,
        )
        d = _make_dispatcher()
        d.action_map = [ActionGroup(
            group_key="cap", description="x", executors={"cap.slow": ex},
        )]

        action = Action(
            action_id="a-1", agent_id="agent-test", action_type="cap.slow",
            parameters={},
        )
        dispatch_task = asyncio.create_task(d._dispatch_action(action))
        await slow_started.wait()
        assert d.has_interruptible_action_in_flight is True

        cancelled = d.cancel_current_action(reason="/abort")
        assert cancelled is True

        result = await dispatch_task
        assert result.cancelled is True
        assert result.success is False
        assert result.completed is True
        assert "abort" in (result.error or "").lower()
        # The dispatcher cleared its tracking after the dispatch resolved.
        assert d.has_interruptible_action_in_flight is False

    @pytest.mark.asyncio
    async def test_non_interruptible_action_cannot_be_cancelled(self):
        from polymathera.colony.agents.models import Action
        from polymathera.colony.agents.patterns.actions import action_executor
        from polymathera.colony.agents.patterns.actions.dispatcher import (
            ActionGroup, MethodWrapperActionExecutor,
        )

        started = asyncio.Event()

        class _Cap:
            @action_executor()  # interruptible defaults False
            async def quick(self) -> str:
                started.set()
                await asyncio.sleep(0.05)
                return "ok"

        cap = _Cap()
        ex = MethodWrapperActionExecutor(
            object=cap,
            method=_Cap.quick,
            action_key="cap.quick",
            interruptible=False,
        )
        d = _make_dispatcher()
        d.action_map = [ActionGroup(
            group_key="cap", description="x", executors={"cap.quick": ex},
        )]

        action = Action(
            action_id="a-2", agent_id="agent-test", action_type="cap.quick",
            parameters={},
        )
        task = asyncio.create_task(d._dispatch_action(action))
        await started.wait()
        # No tracked task — non-interruptible actions are never wrapped.
        assert d.has_interruptible_action_in_flight is False
        # cancel_current_action must report nothing-to-do.
        assert d.cancel_current_action(reason="/abort") is False
        # The action completes normally.
        result = await task
        assert result.success is True
        assert result.cancelled is False

    @pytest.mark.asyncio
    async def test_outer_cancellation_propagates(self):
        """When the OUTER awaiter is cancelled (not via cancel_current_action),
        ``CancelledError`` must propagate out — we must not silently turn
        an agent-shutdown cancellation into a cancelled ``ActionResult``."""
        from polymathera.colony.agents.models import Action
        from polymathera.colony.agents.patterns.actions import action_executor
        from polymathera.colony.agents.patterns.actions.dispatcher import (
            ActionGroup, MethodWrapperActionExecutor,
        )

        started = asyncio.Event()

        class _Cap:
            @action_executor(interruptible=True)
            async def slow(self) -> str:
                started.set()
                await asyncio.sleep(60)
                return "done"

        cap = _Cap()
        ex = MethodWrapperActionExecutor(
            object=cap, method=_Cap.slow, action_key="cap.slow",
            interruptible=True,
        )
        d = _make_dispatcher()
        d.action_map = [ActionGroup(
            group_key="cap", description="x", executors={"cap.slow": ex},
        )]

        action = Action(
            action_id="a-3", agent_id="agent-test", action_type="cap.slow",
            parameters={},
        )
        outer = asyncio.create_task(d._dispatch_action(action))
        await started.wait()
        # NB: we deliberately do NOT call cancel_current_action — we
        # cancel the outer task directly to simulate an agent-shutdown
        # path. The CancelledError must propagate out of dispatch.
        outer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer

    @pytest.mark.asyncio
    async def test_stale_cancel_request_does_not_kill_subsequent_action(self):
        """If a cancel signal arrives just as the targeted action
        completes naturally, the next action must NOT inherit the
        stale cancel — _cancellation_requested clears in the finally."""
        from polymathera.colony.agents.models import Action
        from polymathera.colony.agents.patterns.actions import action_executor
        from polymathera.colony.agents.patterns.actions.dispatcher import (
            ActionGroup, MethodWrapperActionExecutor,
        )

        class _Cap:
            @action_executor(interruptible=True)
            async def fast(self) -> str:
                return "ok"

        cap = _Cap()
        ex = MethodWrapperActionExecutor(
            object=cap, method=_Cap.fast, action_key="cap.fast",
            interruptible=True,
        )
        d = _make_dispatcher()
        d.action_map = [ActionGroup(
            group_key="cap", description="x", executors={"cap.fast": ex},
        )]

        # First action runs and completes naturally.
        a1 = Action(
            action_id="a1", agent_id="agent-test", action_type="cap.fast",
            parameters={},
        )
        r1 = await d._dispatch_action(a1)
        assert r1.success is True
        assert r1.cancelled is False
        # The flag must have been cleared even without an explicit cancel.
        assert d._cancellation_requested is False
        # Second action also runs cleanly.
        a2 = Action(
            action_id="a2", agent_id="agent-test", action_type="cap.fast",
            parameters={},
        )
        r2 = await d._dispatch_action(a2)
        assert r2.success is True
        assert r2.cancelled is False


# ---------------------------------------------------------------------------
# BaseActionPolicy passthrough + abort_current default
# ---------------------------------------------------------------------------


class TestBaseActionPolicyAbort:

    @pytest.mark.asyncio
    async def test_cancel_current_action_when_no_dispatcher_returns_false(self):
        from polymathera.colony.agents.patterns.actions.policies import (
            BaseActionPolicy,
        )
        p = BaseActionPolicy.__new__(BaseActionPolicy)
        p._action_dispatcher = None
        assert p.cancel_current_action(reason="/abort") is False
        assert (await p.abort_current(reason="/abort")) is False

    @pytest.mark.asyncio
    async def test_cancel_passthrough_to_dispatcher(self):
        from polymathera.colony.agents.patterns.actions.policies import (
            BaseActionPolicy,
        )
        p = BaseActionPolicy.__new__(BaseActionPolicy)
        p._action_dispatcher = MagicMock()
        p._action_dispatcher.cancel_current_action = MagicMock(return_value=True)
        p._action_dispatcher.has_interruptible_action_in_flight = True
        assert p.cancel_current_action(reason="/abort") is True
        p._action_dispatcher.cancel_current_action.assert_called_once_with(
            reason="/abort",
        )
        assert p.has_interruptible_action_in_flight is True


# ---------------------------------------------------------------------------
# CodeGenerationActionPolicy abort_current()
# ---------------------------------------------------------------------------


def _make_codegen_policy():
    """Build a CodeGenerationActionPolicy via __new__ with the minimum
    state ``abort_current`` reads. Avoids the heavy production __init__
    (which spawns a planner, registers capabilities, etc.)."""
    from polymathera.colony.agents.patterns.actions.code_generation import (
        CodeGenerationActionPolicy,
    )
    agent = MagicMock(); agent.agent_id = "agent-test"
    p = CodeGenerationActionPolicy.__new__(CodeGenerationActionPolicy)
    p._agent = agent
    p.agent = agent
    p._action_dispatcher = None
    p._consecutive_failures = 0
    p._error_history = []
    p._recovered_code = None
    p._current_codegen_task = None
    p._codegen_cancel_requested = False
    p.max_retries = 3
    return p


class TestCodeGenAbortCurrent:

    @pytest.mark.asyncio
    async def test_idle_returns_false(self):
        p = _make_codegen_policy()
        assert (await p.abort_current(reason="/abort")) is False

    @pytest.mark.asyncio
    async def test_resets_recovery_state(self):
        p = _make_codegen_policy()
        p._consecutive_failures = 2
        p._error_history = [{"code": "x", "error": "boom"}]
        p._recovered_code = "fixed = True"
        assert (await p.abort_current(reason="/abort")) is True
        assert p._consecutive_failures == 0
        assert p._error_history == []
        assert p._recovered_code is None

    @pytest.mark.asyncio
    async def test_cancels_in_flight_codegen_task(self):
        p = _make_codegen_policy()

        async def slow_llm():
            await asyncio.sleep(60)
            return "code = 1"

        p._current_codegen_task = asyncio.create_task(slow_llm())
        # Give the loop a tick to schedule it.
        await asyncio.sleep(0)

        result = await p.abort_current(reason="/abort")
        assert result is True
        assert p._codegen_cancel_requested is True
        # The task is now cancelled (or scheduled for cancellation).
        with pytest.raises(asyncio.CancelledError):
            await p._current_codegen_task

    @pytest.mark.asyncio
    async def test_delegates_to_super_for_dispatcher_action(self):
        p = _make_codegen_policy()
        p._action_dispatcher = MagicMock()
        p._action_dispatcher.cancel_current_action = MagicMock(return_value=True)
        # No in-flight LLM, no recovery state — only the dispatcher
        # returns True. abort_current must still report True.
        result = await p.abort_current(reason="/abort")
        assert result is True
        p._action_dispatcher.cancel_current_action.assert_called_once_with(
            reason="/abort",
        )


# ---------------------------------------------------------------------------
# REPL CancelledError handling
# ---------------------------------------------------------------------------


class TestREPLCancelledErrorPropagation:

    @pytest.mark.asyncio
    async def test_execute_does_not_swallow_cancelled_error(self):
        """The REPL's execute() must let CancelledError propagate. The
        dispatcher's outer try/except is what converts it into an
        ActionResult(cancelled=True); if execute() catches it (as a
        generic Exception sibling) the cancel turns into a misleading
        success=False result and the planner tries to recover."""
        from polymathera.colony.agents.patterns.actions.repl import (
            PolicyPythonREPL,
        )

        repl = PolicyPythonREPL.__new__(PolicyPythonREPL)
        # Patch _shell to a stub whose run_cell_async sleeps forever
        # so we can cancel it deterministically.
        repl._shell = MagicMock()

        async def slow(*_args, **_kwargs):
            await asyncio.sleep(60)

        repl._shell.run_cell_async = slow
        repl._shell.user_ns = {}
        repl._max_execution_time = 60.0

        # Stub the validate hook so we don't depend on its full logic.
        repl._validate_code = MagicMock()
        repl._pending_actions = []
        repl._variables = {}
        repl._code_history = []

        task = asyncio.create_task(repl.execute("x = 1"))
        # Give it a tick to enter the run_cell_async coroutine.
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ---------------------------------------------------------------------------
# Chat router classifies /abort, /cancel, /replace
# ---------------------------------------------------------------------------


class TestChatRouterPhase2Commands:

    def test_slash_abort_is_high_priority(self):
        from polymathera.colony.web_ui.backend.routers.chat import (
            _is_control_command, _extract_command,
        )
        assert _is_control_command("/abort") is True
        assert _extract_command("/abort") == "/abort"

    def test_slash_cancel_is_high_priority(self):
        from polymathera.colony.web_ui.backend.routers.chat import (
            _is_control_command,
        )
        assert _is_control_command("/cancel") is True

    def test_slash_replace_is_high_priority(self):
        from polymathera.colony.web_ui.backend.routers.chat import (
            _is_control_command, _extract_command,
        )
        assert _is_control_command("/replace fix the bug instead") is True
        assert _extract_command("/replace fix the bug instead") == "/replace"


# ---------------------------------------------------------------------------
# SessionOrchestratorCapability /abort + /replace handlers
# ---------------------------------------------------------------------------


def _make_orchestrator_with_policy(policy_mock):
    """Build a SessionOrchestratorCapability via __new__ with a stub
    agent whose ``action_policy`` is the supplied mock. ``_post_response``
    is patched into a recording stub so tests can assert what the user
    saw without needing a real blackboard."""
    from polymathera.colony.web_ui.backend.chat.session_agent import (
        SessionOrchestratorCapability,
    )
    cap = SessionOrchestratorCapability.__new__(SessionOrchestratorCapability)
    agent = MagicMock(); agent.agent_id = "agent-test"
    agent.action_policy = policy_mock
    cap._agent = agent
    cap.posted: list[tuple[str, dict]] = []  # type: ignore[attr-defined]

    async def _post(content, **extra):
        cap.posted.append((content, extra))  # type: ignore[attr-defined]

    cap._post_response = _post  # type: ignore[assignment]
    return cap


class TestSessionOrchestratorAbort:

    @pytest.mark.asyncio
    async def test_abort_with_no_policy_acks_clearly(self):
        cap = _make_orchestrator_with_policy(None)
        await cap._handle_abort_command(reason="/abort")
        assert len(cap.posted) == 1
        body, extra = cap.posted[0]
        assert "no action policy" in body.lower()
        assert extra.get("kind") == "control_ack"

    @pytest.mark.asyncio
    async def test_abort_calls_policy_abort_current_with_reason(self):
        policy = MagicMock()
        policy.abort_current = AsyncMock(return_value=True)
        cap = _make_orchestrator_with_policy(policy)
        await cap._handle_abort_command(reason="/abort")
        policy.abort_current.assert_awaited_once_with(reason="/abort")
        body, extra = cap.posted[0]
        assert "abort" in body.lower()
        assert extra.get("kind") == "control_ack"

    @pytest.mark.asyncio
    async def test_abort_when_idle_says_nothing_to_abort(self):
        policy = MagicMock()
        policy.abort_current = AsyncMock(return_value=False)
        cap = _make_orchestrator_with_policy(policy)
        await cap._handle_abort_command(reason="/abort")
        body, _ = cap.posted[0]
        assert "nothing to abort" in body.lower() or "idle" in body.lower()


class TestSessionOrchestratorReplace:

    @pytest.mark.asyncio
    async def test_empty_body_rejected(self):
        cap = _make_orchestrator_with_policy(None)
        await cap._handle_replace_command("/replace")
        body, extra = cap.posted[0]
        assert "requires a new request" in body.lower() or "usage" in body.lower()
        assert extra.get("kind") == "control_ack"

    @pytest.mark.asyncio
    async def test_replace_aborts_then_posts_user_message(self):
        from polymathera.colony.web_ui.backend.chat.chat_protocol import (
            SessionChatProtocol,
        )
        policy = MagicMock()
        policy.abort_current = AsyncMock(return_value=True)
        cap = _make_orchestrator_with_policy(policy)

        # Stub the blackboard write.
        bb = MagicMock()
        bb.write = AsyncMock()
        cap.get_blackboard = AsyncMock(return_value=bb)  # type: ignore[assignment]

        await cap._handle_replace_command("/replace fix the bug instead")
        # Aborted current.
        policy.abort_current.assert_awaited_once_with(reason="/replace")
        # Posted to chat:user:* with the body (sans "/replace ").
        bb.write.assert_awaited_once()
        key, payload = bb.write.await_args.args
        assert key.startswith(
            SessionChatProtocol.user_message_key("").rsplit(":", 1)[0],
        )
        assert payload["content"] == "fix the bug instead"
        # The control_ack confirms what happened.
        body, extra = cap.posted[-1]
        assert "queued" in body.lower()
        assert extra.get("kind") == "control_ack"

    @pytest.mark.asyncio
    async def test_replace_when_idle_still_posts_message(self):
        policy = MagicMock()
        policy.abort_current = AsyncMock(return_value=False)
        cap = _make_orchestrator_with_policy(policy)
        bb = MagicMock(); bb.write = AsyncMock()
        cap.get_blackboard = AsyncMock(return_value=bb)  # type: ignore[assignment]

        await cap._handle_replace_command("/replace do something else")
        # Even though abort returned False (nothing to abort), the
        # replacement message must still be queued — the user clearly
        # wants the new task to run.
        bb.write.assert_awaited_once()
        body, _ = cap.posted[-1]
        assert "idle" in body.lower() or "queued" in body.lower()
