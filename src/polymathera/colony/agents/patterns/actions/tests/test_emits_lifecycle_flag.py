"""Tests for the ``emits_lifecycle`` flag on the ``@action_executor``
decorator and the executor wrappers it propagates through.

The flag is the framework-side hook that gates
``policy:action_started`` / ``policy:action_completed`` lifecycle
emission for actions whose semantics make a "running" badge wrong
(idle waits, publish-only narrative emits). The invariant
"idle waits are not work" lives at the single emission source via
this flag — every subscriber inherits the fix.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.patterns.actions.dispatcher import (
    ActionDispatcher,
    ActionGroup,
    FunctionWrapperActionExecutor,
    MethodWrapperActionExecutor,
    action_executor,
)
from polymathera.colony.agents.patterns.actions.policies import (
    EventDrivenActionPolicy,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Decorator-level pinning
# ---------------------------------------------------------------------------


def test_default_action_executor_emits_lifecycle() -> None:
    """Backwards-compatible default: ``@action_executor()`` produces a
    method with ``_action_emits_lifecycle = True`` — existing actions
    keep emitting."""

    @action_executor()
    async def my_action(self, x: int) -> int:  # noqa: D401, ARG001
        return x

    assert my_action._action_emits_lifecycle is True


def test_action_executor_emits_lifecycle_false_pinned() -> None:
    """The flag flows from the decorator to a method attribute the
    dispatcher's wrapper consumes. ``emits_lifecycle=False`` is the
    opt-out used by ``wait_for_next_event`` and the upcoming
    ``emit_mission_status``."""

    @action_executor(emits_lifecycle=False)
    async def idle_action(self) -> None:  # noqa: D401, ARG001
        return None

    assert idle_action._action_emits_lifecycle is False


def test_wait_for_next_event_declares_emits_lifecycle_false() -> None:
    """Regression pin: ``EventDrivenActionPolicy.wait_for_next_event``
    is the canonical idle primitive; it MUST declare
    ``emits_lifecycle=False`` so the spinner row stops lying about a
    wait being "running"."""

    method = EventDrivenActionPolicy.wait_for_next_event
    assert method._action_emits_lifecycle is False


# ---------------------------------------------------------------------------
# Executor wrappers propagate the flag
# ---------------------------------------------------------------------------


def test_method_wrapper_default_emits_lifecycle() -> None:
    """Constructed without the kwarg, the wrapper defaults to True so
    existing call sites that don't set the flag keep emitting."""

    wrapper = MethodWrapperActionExecutor(
        object=object(),
        method=lambda self: None,
        action_key="some.action",
    )
    assert wrapper.emits_lifecycle is True


def test_method_wrapper_emits_lifecycle_false_propagated() -> None:
    wrapper = MethodWrapperActionExecutor(
        object=object(),
        method=lambda self: None,
        action_key="some.action",
        emits_lifecycle=False,
    )
    assert wrapper.emits_lifecycle is False


def test_function_wrapper_default_emits_lifecycle() -> None:
    agent = MagicMock()
    wrapper = FunctionWrapperActionExecutor(
        func=lambda: None,
        action_key="some.action",
        agent=agent,
    )
    assert wrapper.emits_lifecycle is True


def test_function_wrapper_emits_lifecycle_false_propagated() -> None:
    agent = MagicMock()
    wrapper = FunctionWrapperActionExecutor(
        func=lambda: None,
        action_key="some.action",
        agent=agent,
        emits_lifecycle=False,
    )
    assert wrapper.emits_lifecycle is False


# ---------------------------------------------------------------------------
# Dispatcher.find_executor
# ---------------------------------------------------------------------------


def test_dispatcher_find_executor_returns_registered_executor() -> None:
    """``find_executor(action_key)`` returns the executor across all
    registered action groups — the lifecycle-emission gate uses this
    to look up the ``emits_lifecycle`` flag."""

    wrapper = MethodWrapperActionExecutor(
        object=object(),
        method=lambda self: None,
        action_key="grp.act",
        emits_lifecycle=False,
    )
    group = ActionGroup(
        group_key="grp",
        description="test",
        executors={"grp.act": wrapper},
    )
    dispatcher = ActionDispatcher.__new__(ActionDispatcher)
    dispatcher.action_map = [group]

    found = dispatcher.find_executor("grp.act")
    assert found is wrapper
    assert found.emits_lifecycle is False


def test_dispatcher_find_executor_returns_none_for_unknown_key() -> None:
    """Unknown keys return ``None``; the lifecycle gate treats this as
    "emit by default" — synthetic/test-injected actions opt in to
    emission via the safe default, not via silent suppression."""

    dispatcher = ActionDispatcher.__new__(ActionDispatcher)
    dispatcher.action_map = []
    assert dispatcher.find_executor("does.not.exist") is None


def test_dispatcher_find_executor_first_match_wins_across_groups() -> None:
    """Two groups registering the same key returns the first match
    (consistent with the existing ``get_plannable_actions`` traversal
    order)."""

    a = MethodWrapperActionExecutor(
        object=object(), method=lambda self: None,
        action_key="dup.act", emits_lifecycle=True,
    )
    b = MethodWrapperActionExecutor(
        object=object(), method=lambda self: None,
        action_key="dup.act", emits_lifecycle=False,
    )
    g1 = ActionGroup(group_key="g1", description="a", executors={"dup.act": a})
    g2 = ActionGroup(group_key="g2", description="b", executors={"dup.act": b})
    dispatcher = ActionDispatcher.__new__(ActionDispatcher)
    dispatcher.action_map = [g1, g2]

    found = dispatcher.find_executor("dup.act")
    assert found is a
