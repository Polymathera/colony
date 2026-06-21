"""Tests for :func:`effective_loop_max_iterations` — the lifecycle-aware
rule that decides whether the agent loop's iteration cap is honored
or bypassed.

R11 forensic confirmed the silent-death pattern this rule prevents: a
SessionAgent that ran 20 iterations of LLM-judge-rejected
``respond_to_user`` drafts hit the default ``max_iterations=20`` cap
in the middle of a conversation; the user's next push-back message
arrived 16 minutes later and had no live agent in the chat session to
handle it.

The rule:

- CONTINUOUS lifecycle (long-lived service agents — SessionAgent,
  notification listeners, mention-routers, webhook watchers) → cap is
  always bypassed; loop runs until ``_stop_requested``.
- ONE_SHOT lifecycle (focused coordinators with a bounded goal) → cap
  is honored as a stuck-detection signal.

Pinning the rule at this level keeps it a one-line check on the loop
(no bloat) AND keeps it testable in isolation (no deployment-actor
setup required). See ``colony/MEMORY.md::primitives-not-pipelines``
for the framing principle: the rule lives at the loop, not at every
agent class.
"""

from __future__ import annotations

import logging

import pytest

from polymathera.colony.agents.base import effective_loop_max_iterations
from polymathera.colony.agents.models import LifecycleMode


# ---------------------------------------------------------------------------
# Rule: CONTINUOUS bypasses the cap
# ---------------------------------------------------------------------------


def test_continuous_bypasses_configured_cap() -> None:
    """A CONTINUOUS agent never inherits the configured cap; the
    function returns ``None`` so the loop falls through to
    ``_stop_requested``-only termination."""

    result = effective_loop_max_iterations(
        agent_id="agent-test",
        lifecycle_mode=LifecycleMode.CONTINUOUS,
        configured_max_iterations=20,
    )
    assert result is None


def test_continuous_bypasses_cap_even_when_caller_passes_none() -> None:
    """Already-``None`` cap stays ``None`` for CONTINUOUS — no
    spurious value introduced."""

    result = effective_loop_max_iterations(
        agent_id="agent-test",
        lifecycle_mode=LifecycleMode.CONTINUOUS,
        configured_max_iterations=None,
    )
    assert result is None


def test_continuous_bypass_logs_info_when_cap_was_configured(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Operator-visible signal that the cap is being intentionally
    bypassed. Without this log, a future operator inspecting why
    a CONTINUOUS agent ran past the configured ``max_iterations``
    would have to read the source to find out — that's the kind of
    silent override we explicitly avoid per
    ``colony/MEMORY.md::concise-diagnostics-no-speculation`` (the
    framework is observable by default)."""

    caplog.set_level(logging.INFO, logger="polymathera.colony.agents.base")
    effective_loop_max_iterations(
        agent_id="agent-test",
        lifecycle_mode=LifecycleMode.CONTINUOUS,
        configured_max_iterations=20,
    )
    msgs = [r.getMessage() for r in caplog.records]
    bypass_lines = [m for m in msgs if "CONTINUOUS" in m and "bypass" in m]
    assert bypass_lines, msgs
    assert "agent-test" in bypass_lines[0]
    assert "20" in bypass_lines[0]


def test_continuous_bypass_no_log_when_cap_was_already_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No-op cases don't spam the log — only emit the bypass
    message when the operator's configured value is being
    overridden."""

    caplog.set_level(logging.INFO, logger="polymathera.colony.agents.base")
    effective_loop_max_iterations(
        agent_id="agent-test",
        lifecycle_mode=LifecycleMode.CONTINUOUS,
        configured_max_iterations=None,
    )
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("bypassing" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# Rule: ONE_SHOT honors the configured cap
# ---------------------------------------------------------------------------


def test_one_shot_honors_configured_cap() -> None:
    """ONE_SHOT coordinators rely on ``max_iterations`` as their
    stuck-detection signal — the function passes the value through
    unchanged."""

    result = effective_loop_max_iterations(
        agent_id="agent-coord",
        lifecycle_mode=LifecycleMode.ONE_SHOT,
        configured_max_iterations=50,
    )
    assert result == 50


def test_one_shot_honors_none_when_unset() -> None:
    """A ONE_SHOT agent with no configured cap stays uncapped —
    the rule never introduces a cap that wasn't there."""

    result = effective_loop_max_iterations(
        agent_id="agent-coord",
        lifecycle_mode=LifecycleMode.ONE_SHOT,
        configured_max_iterations=None,
    )
    assert result is None


def test_one_shot_does_not_log_bypass(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The bypass log line is reserved for CONTINUOUS — ONE_SHOT
    must not produce it (would be misleading in operator-facing
    logs)."""

    caplog.set_level(logging.INFO, logger="polymathera.colony.agents.base")
    effective_loop_max_iterations(
        agent_id="agent-coord",
        lifecycle_mode=LifecycleMode.ONE_SHOT,
        configured_max_iterations=50,
    )
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("CONTINUOUS" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# Pure-function contract: no side effects beyond the log
# ---------------------------------------------------------------------------


def test_function_is_pure_and_idempotent() -> None:
    """Calling twice with the same inputs returns the same output —
    no hidden state, no caching surprises."""

    a = effective_loop_max_iterations(
        agent_id="agent-x",
        lifecycle_mode=LifecycleMode.ONE_SHOT,
        configured_max_iterations=42,
    )
    b = effective_loop_max_iterations(
        agent_id="agent-x",
        lifecycle_mode=LifecycleMode.ONE_SHOT,
        configured_max_iterations=42,
    )
    assert a == b == 42
