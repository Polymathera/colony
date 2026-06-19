"""Wiring contract for ``create_default_action_policy``.

The factory in ``defaults.py`` is the single bridge between an agent's
``action_policy_blueprints`` dict (set by missions like
``ProjectPlanningCoordinator`` to mount a typed
:class:`DecomposeCompletionValidator`) and the underlying
:class:`CodeGenerationActionPolicy`. Every blueprint key the policy's
constructor accepts MUST be forwarded — silently dropping one means
the mission's mounted policy is replaced by the default, and the
mount is dead code.

The 2026-06-18 head5 regression: the factory forwarded
``runtime_guardrail`` and ~12 other kwargs but never read
``completion_validator``. The ``DecomposeCompletionValidator`` mounted
by :class:`ProjectPlanningCoordinator` was silently discarded and
``LLMCompletionValidator`` ran instead — rubber-stamping a verbose
LLM-judged "success" message for a decompose run that had not drained
its in-scope set. This file pins the forwarding contract so the
regression cannot recur.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from polymathera.colony.agents.patterns.actions.code_constraints import (
    CompletionValidator,
)
from polymathera.colony.agents.patterns.actions.defaults import (
    create_default_action_policy,
)


pytestmark = pytest.mark.asyncio


class _SentinelValidator(CompletionValidator):
    """A typed sentinel so an ``assert policy._completion_validator is X``
    distinguishes 'caller's instance' from 'default constructed
    fresh'. Has no behavior — the test never invokes ``validate``."""

    async def validate(  # pragma: no cover — never called in this test
        self,
        agent: Any,
        goals: list[str],
        results: dict[str, Any],
        execution_context: Any,
    ) -> Any:
        raise NotImplementedError(
            "sentinel validator: forwarding test only, not a real validator"
        )


async def test_completion_validator_kwarg_is_forwarded() -> None:
    """When the caller passes ``completion_validator=X``, the factory
    must thread it through to
    :func:`create_code_generation_action_policy`. Before the fix,
    this kwarg was silently dropped; the policy was constructed with
    ``completion_validator=None`` and defaulted to
    ``LLMCompletionValidator()``."""

    sentinel = _SentinelValidator()
    agent = object()  # the inner factory is mocked, so the agent is never used

    with patch(
        "polymathera.colony.agents.patterns.actions.code_generation."
        "create_code_generation_action_policy",
        new=AsyncMock(return_value=object()),
    ) as inner:
        await create_default_action_policy(
            agent=agent,  # type: ignore[arg-type]
            completion_validator=sentinel,
        )

    inner.assert_called_once()
    forwarded = inner.call_args.kwargs.get("completion_validator")
    assert forwarded is sentinel, (
        "create_default_action_policy did not forward "
        "`completion_validator` to create_code_generation_action_policy. "
        "This is the head5 bug: DecomposeCompletionValidator silently "
        "discarded; LLMCompletionValidator runs as the fallback and "
        "rubber-stamps any LLM-judged 'success'."
    )


async def test_completion_validator_omitted_defaults_to_none() -> None:
    """When the caller omits ``completion_validator``, the factory
    must forward ``None`` (NOT a default validator). The downstream
    ``create_code_generation_action_policy`` applies its own default
    (``LLMCompletionValidator``); the factory must not pre-empt that
    decision by constructing a fresh validator here."""

    agent = object()

    with patch(
        "polymathera.colony.agents.patterns.actions.code_generation."
        "create_code_generation_action_policy",
        new=AsyncMock(return_value=object()),
    ) as inner:
        await create_default_action_policy(agent=agent)  # type: ignore[arg-type]

    inner.assert_called_once()
    assert inner.call_args.kwargs.get("completion_validator") is None
