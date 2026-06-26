"""R7-FIX-A regression: the LLM-exception swallow at
``code_generation.py:2313`` and ``minimal.py:141`` was removed so
that :class:`LLMInferenceError` raised inside ``plan_step``
PROPAGATES to ``EventDrivenActionPolicy.execute_iteration``'s outer
handler, which routes through :class:`LLMFailureBackoff`. Run7 had
~5,364 spin-loop retries against a permanently-open breaker because
the swallow bypassed the backoff; pinning here so it can't regress.

Source-level inspection — we don't need to construct a full agent +
LLM cluster to verify the contract; the contract is "no broad
``except Exception`` around the codegen LLM call that returns
``None``". A grep-based check catches a future refactor that
re-introduces the swallow without exercising the network path.
"""

from __future__ import annotations

import inspect

import pytest


def test_codegen_action_policy_does_not_swallow_llm_exception() -> None:
    """``CodeGenerationActionPolicy.plan_step`` must NOT contain a
    ``except Exception ... return None`` block around the codegen
    ``generate()`` call. The outer
    ``EventDrivenActionPolicy.execute_iteration`` catches
    :class:`LLMInferenceError` and runs the
    :class:`LLMFailureBackoff` sleep + idle-wait token; swallowing
    here bypasses that and produces tight spin loops against open
    breakers."""

    from polymathera.colony.agents.patterns.actions.code_generation import (
        CodeGenerationActionPolicy,
    )

    src = inspect.getsource(CodeGenerationActionPolicy.plan_step)
    # The exact bug pattern: an Exception-catching handler that
    # logs "code generation failed" and returns None silently.
    # Pin the specific shape so a partial-regression (re-introducing
    # a slightly-different swallow with the same effect) still trips.
    assert "code generation failed" not in src or "return None" not in src.split("code generation failed")[1][:200], (
        "CodeGenerationActionPolicy.plan_step appears to re-introduce "
        "the LLM-exception silent-swallow that bypasses "
        "EventDrivenActionPolicy.execute_iteration's LLMFailureBackoff. "
        "See run7 forensic / R7-FIX-A."
    )


def test_minimal_action_policy_does_not_swallow_llm_exception() -> None:
    """Same contract for :class:`MinimalActionPolicy`. Same root,
    same symptom."""

    from polymathera.colony.agents.patterns.actions.minimal import (
        MinimalActionPolicy,
    )

    src = inspect.getsource(MinimalActionPolicy.plan_step)
    assert "LLM inference failed" not in src or "return None" not in src.split("LLM inference failed")[1][:200], (
        "MinimalActionPolicy.plan_step appears to re-introduce the "
        "LLM-exception silent-swallow. See run7 forensic / R7-FIX-A."
    )


def test_event_driven_outer_handler_still_catches_llm_inference_error(
) -> None:
    """The outer handler in
    ``EventDrivenActionPolicy._execute_iteration_inner`` is the
    canonical catch site for :class:`LLMInferenceError` — verified
    by source inspection so a future refactor that drops the outer
    handler surfaces here, not silently in production. The public
    ``execute_iteration`` is now a thin shell delegating to the
    inner method; the catch lives in the inner."""

    from polymathera.colony.agents.patterns.actions.policies import (
        EventDrivenActionPolicy,
    )

    src = inspect.getsource(
        EventDrivenActionPolicy._execute_iteration_inner,
    )
    assert "except LLMInferenceError" in src, (
        "EventDrivenActionPolicy._execute_iteration_inner must catch "
        "LLMInferenceError + route through LLMFailureBackoff. "
        "Removing this handler would re-create the run7 spin loop."
    )
    assert "_llm_failure_backoff.handle_failure" in src
