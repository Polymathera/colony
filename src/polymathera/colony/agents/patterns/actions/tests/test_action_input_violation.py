"""DL1 + DL3 tests — the cell-aborting prevention surface and the
guardrail-waiver outstanding-id lifecycle that drives DL2's
``is_awaiting_event`` from the waiver capability.

DL1 (run5 deadlock-class prevention): both
:class:`MethodWrapperActionExecutor` and
:class:`FunctionWrapperActionExecutor` re-raise subclasses of
:class:`ActionInputViolation` UNWRAPPED so the LLM cell aborts at
the ``await run(...)`` site — same shape as
``GuardrailBlockedError``. Other exceptions are still wrapped in
``ActionResult(success=False)`` as before.

DL3: :class:`GuardrailWaiverCapability` tracks outstanding
``waiver_id``s and clears them on the typed response handler. This
lifecycle drives the capability's ``is_awaiting_event()`` override
so ``wait_for_next_event``'s DL2 pre-check counts the capability
as a live wake source while a decision is outstanding, and stops
counting it as soon as the response binding is produced.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from polymathera.colony.agents.blackboard import BlackboardEvent
from polymathera.colony.agents.blackboard.protocol import (
    GuardrailWaiverProtocol,
)
from polymathera.colony.agents.patterns.actions.dispatcher import (
    Action,
    ActionInputViolation,
    FunctionWrapperActionExecutor,
    MethodWrapperActionExecutor,
)
from polymathera.colony.agents.patterns.capabilities.guardrail_waiver import (
    GuardrailWaiverCapability,
)
from polymathera.colony.agents.patterns.capabilities.human_approval import (
    RequestHumanApprovalEmpty,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# DL1: ActionInputViolation marker re-raises unwrapped
# ---------------------------------------------------------------------------


def test_request_human_approval_empty_is_action_input_violation() -> None:
    """The run5 deadlock-triggering exception inherits from the new
    marker. Pinned so the parent class can't silently revert to a
    plain ``ValueError`` and start being swallowed again."""

    assert issubclass(RequestHumanApprovalEmpty, ActionInputViolation)
    assert issubclass(RequestHumanApprovalEmpty, ValueError)


class _RaisesInputViolationMethod:
    async def do(self) -> dict[str, Any]:
        raise RequestHumanApprovalEmpty("intentional from method")


def _make_action(action_type: str = "do") -> Action:
    return Action(
        action_id="t1",
        action_type=action_type,
        agent_id="a",
        parameters={},
    )


async def test_method_executor_reraises_action_input_violation_unwrapped(
) -> None:
    obj = _RaisesInputViolationMethod()
    executor = MethodWrapperActionExecutor(
        object=obj, method=obj.do.__func__, action_key="do",
    )
    with pytest.raises(ActionInputViolation):
        await executor.execute(_make_action("do"))


class _RaisesGenericMethod:
    async def do(self) -> dict[str, Any]:
        raise RuntimeError("generic boom")


async def test_method_executor_wraps_non_violation_in_action_result(
) -> None:
    """Asymmetry pin: only ``ActionInputViolation`` subclasses
    bypass the wrap. Generic exceptions still surface as
    ``success=False`` so existing recovery flows don't change."""

    obj = _RaisesGenericMethod()
    executor = MethodWrapperActionExecutor(
        object=obj, method=obj.do.__func__, action_key="do",
    )
    action = _make_action("do")
    result = await executor.execute(action)
    assert result.success is False
    assert "generic boom" in (result.error or "")


async def test_function_executor_reraises_action_input_violation_unwrapped(
) -> None:
    """Same twin path on the function-wrapper executor — both
    ``except`` blocks were updated to keep behaviour symmetric."""

    async def f() -> dict[str, Any]:
        raise RequestHumanApprovalEmpty("intentional from function")

    agent = MagicMock()
    executor = FunctionWrapperActionExecutor(
        func=f, action_key="f", agent=agent,
    )
    action = _make_action("f")
    with pytest.raises(ActionInputViolation):
        await executor.execute(action)


async def test_function_executor_wraps_non_violation_in_action_result(
) -> None:
    async def f() -> dict[str, Any]:
        raise RuntimeError("generic boom")

    agent = MagicMock()
    executor = FunctionWrapperActionExecutor(
        func=f, action_key="f", agent=agent,
    )
    action = _make_action("f")
    result = await executor.execute(action)
    assert result.success is False
    assert "generic boom" in (result.error or "")


async def test_input_violation_subclass_also_reraises() -> None:
    """The contract is ``except ActionInputViolation: raise``, so a
    subclass NOT defined in this module (e.g. ``NoLiveWakeSource``)
    flows through the same path. Confirms the asymmetric catch is
    polymorphic, not RequestHumanApprovalEmpty-specific."""

    class _MyViolation(ActionInputViolation):
        pass

    class _Obj:
        async def do(self) -> dict[str, Any]:
            raise _MyViolation("subclass")

    obj = _Obj()
    executor = MethodWrapperActionExecutor(
        object=obj, method=obj.do.__func__, action_key="do",
    )
    action = _make_action("do")
    with pytest.raises(_MyViolation):
        await executor.execute(action)


# ---------------------------------------------------------------------------
# DL3: GuardrailWaiverCapability outstanding-id lifecycle + is_awaiting_event
# ---------------------------------------------------------------------------


def _make_waiver_capability(
    agent_id: str = "agent-1",
) -> tuple[GuardrailWaiverCapability, MagicMock]:
    cap = GuardrailWaiverCapability.__new__(GuardrailWaiverCapability)
    cap._agent = SimpleNamespace(agent_id=agent_id)
    cap._outstanding_waiver_ids = set()
    bb = MagicMock()
    bb.write = AsyncMock()
    cap.get_blackboard = AsyncMock(return_value=bb)
    return cap, bb


async def test_is_awaiting_event_false_when_no_request_published() -> None:
    cap, _ = _make_waiver_capability()
    assert cap.is_awaiting_event() is False


async def test_is_awaiting_event_true_after_publish() -> None:
    cap, _ = _make_waiver_capability()
    result = await cap.request_guardrail_waiver(
        constraint_id="rule_x", justification="reason text",
    )
    assert result["ok"] is True
    assert cap.is_awaiting_event() is True


async def test_is_awaiting_event_clears_on_typed_response() -> None:
    """The response handler clears the matching waiver_id; once the
    set is empty the capability stops contributing to DL2's
    live-wake-source pre-check."""

    cap, _ = _make_waiver_capability()
    result = await cap.request_guardrail_waiver(
        constraint_id="rule_x", justification="reason text",
    )
    waiver_id = result["waiver_id"]
    assert cap.is_awaiting_event() is True

    event = BlackboardEvent(
        event_type="write",
        key=GuardrailWaiverProtocol.response_key(waiver_id),
        value={
            "waiver_id": waiver_id,
            "constraint_id": "rule_x",
            "approved": True,
            "decided_by": "user_abc",
            "reason": "ok",
        },
    )
    await cap._on_response(event, None)
    assert cap.is_awaiting_event() is False


async def test_is_awaiting_event_unaffected_by_alien_response_key() -> None:
    """A response key that doesn't parse via the typed protocol must
    NOT clear an outstanding waiver id — otherwise an unrelated BB
    write could silently flip the pre-check off and re-introduce
    the deadlock."""

    cap, _ = _make_waiver_capability()
    await cap.request_guardrail_waiver(
        constraint_id="rule_x", justification="reason text",
    )
    assert cap.is_awaiting_event() is True

    event = BlackboardEvent(
        event_type="write",
        key="chat:user:msg_X",
        value={"approved": True},
    )
    await cap._on_response(event, None)
    assert cap.is_awaiting_event() is True


async def test_is_awaiting_event_handles_multiple_outstanding_requests(
) -> None:
    """Two concurrent waivers ⇒ live-wake-source remains True until
    BOTH responses land. The pre-check then drops to False."""

    cap, _ = _make_waiver_capability()
    r1 = await cap.request_guardrail_waiver(
        constraint_id="rule_x", justification="r",
    )
    r2 = await cap.request_guardrail_waiver(
        constraint_id="rule_y", justification="r",
    )
    assert cap.is_awaiting_event() is True

    event1 = BlackboardEvent(
        event_type="write",
        key=GuardrailWaiverProtocol.response_key(r1["waiver_id"]),
        value={
            "waiver_id": r1["waiver_id"],
            "constraint_id": "rule_x",
            "approved": False,
            "decided_by": "u",
            "reason": "n",
        },
    )
    await cap._on_response(event1, None)
    assert cap.is_awaiting_event() is True  # r2 still outstanding

    event2 = BlackboardEvent(
        event_type="write",
        key=GuardrailWaiverProtocol.response_key(r2["waiver_id"]),
        value={
            "waiver_id": r2["waiver_id"],
            "constraint_id": "rule_y",
            "approved": True,
            "decided_by": "u",
            "reason": "ok",
        },
    )
    await cap._on_response(event2, None)
    assert cap.is_awaiting_event() is False


async def test_failed_request_does_not_register_outstanding_id() -> None:
    """A request that returns ``ok=False`` (e.g. empty
    ``constraint_id`` rejected by the input validator) MUST NOT
    leave a stranded entry in the outstanding-ids set — otherwise
    the agent would be considered awaiting an event that will never
    arrive (the inverse of run5)."""

    cap, bb = _make_waiver_capability()
    result = await cap.request_guardrail_waiver(
        constraint_id="   ", justification="r",
    )
    assert result["ok"] is False
    bb.write.assert_not_awaited()
    assert cap.is_awaiting_event() is False
