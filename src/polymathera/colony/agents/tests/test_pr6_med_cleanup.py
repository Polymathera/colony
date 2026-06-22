"""Tests for the MED cleanup tier (R12 B14-B20).

Currently covers:

- B19 — ``resolve_effective_max_iterations`` returns ``int | None``;
  callers can opt out of the 20-iteration schema default.
- B20 — ``signal_completion`` install gate now AND-combines
  ``_allow_self_termination`` with the lifecycle check; CONTINUOUS
  agents NEVER expose ``signal_completion`` to their LLM regardless
  of the per-policy default.
- B17 — backfill query cap surfaces a WARNING when hit (was: silent
  truncation; user saw only the most recent 500 rows with no signal).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from polymathera.colony.agents.configs import (
    MissionExecutionPolicy,
    resolve_effective_max_iterations,
)


# ---------------------------------------------------------------------------
# B19: resolve_effective_max_iterations accepts schema_default=None
# ---------------------------------------------------------------------------


def test_resolve_returns_int_when_caller_overrides() -> None:
    policy = MissionExecutionPolicy()
    assert resolve_effective_max_iterations(
        caller_override=7, policy=policy,
    ) == 7


def test_resolve_returns_policy_when_no_caller_override() -> None:
    policy = MissionExecutionPolicy(max_iterations=50)
    assert resolve_effective_max_iterations(
        caller_override=None, policy=policy,
    ) == 50


def test_resolve_returns_schema_default_20_by_default() -> None:
    """Backwards-compat: when neither caller nor policy sets the cap,
    fall back to 20. Pre-B19 behavior preserved."""

    policy = MissionExecutionPolicy()
    assert resolve_effective_max_iterations(
        caller_override=None, policy=policy,
    ) == 20


def test_resolve_returns_none_when_schema_default_explicitly_none() -> None:
    """B19 fix: callers that want "no cap" semantics (e.g. routing a
    CONTINUOUS coordinator through this helper) pass
    ``schema_default=None`` and get ``None`` back, which the agent
    loop's ``effective_loop_max_iterations`` honors as un-capped.
    Pre-B19 the function returned 20 here regardless."""

    policy = MissionExecutionPolicy()
    assert resolve_effective_max_iterations(
        caller_override=None,
        policy=policy,
        schema_default=None,
    ) is None


# ---------------------------------------------------------------------------
# B20: signal_completion install is gated on lifecycle too
# ---------------------------------------------------------------------------


def test_signal_completion_install_gated_on_lifecycle() -> None:
    """Source-pin: the namespace-install branch AND-combines
    ``_allow_self_termination`` with ``LifecycleMode.CONTINUOUS``.
    Without the lifecycle check, a future CONTINUOUS agent built
    via defaults.py:68 (which defaults allow_self_termination=True)
    becomes killable by its own LLM."""

    cg = (
        Path(__file__).resolve().parents[1]
        / "patterns" / "actions" / "code_generation.py"
    ).read_text(encoding="utf-8")
    # The two-key gate: both checks must appear together.
    assert "self._allow_self_termination" in cg
    assert "LifecycleMode.CONTINUOUS" in cg
    # Specifically the install branch combines them.
    assert "not _is_continuous" in cg


# ---------------------------------------------------------------------------
# B17: backfill cap surface
# ---------------------------------------------------------------------------


def test_backfill_cap_emits_warning_when_hit() -> None:
    """The 500-row backfill cap used to be silent. Pin the warning
    + the operator-readable message so a future refactor that drops
    it surfaces here."""

    ch = (
        Path(__file__).resolve().parents[2]
        / "web_ui" / "backend" / "routers" / "chat.py"
    ).read_text(encoding="utf-8")
    assert "_BACKFILL_CAP" in ch
    assert "_backfill_chat_history: hit limit" in ch
