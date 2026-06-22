"""Tests for PR2 (R12-ROOT-CAUSE-A + B1/B7/B13) — lifecycle-aware
iteration cap consulted by every policy.

Run12 confirmed the silent-death pattern at a SECOND layer: R11-Fix1
patched the OUTER agent-loop cap (effective_loop_max_iterations) to
bypass for CONTINUOUS-mode agents, but the INNER
``CodeGenerationActionPolicy.max_code_iterations`` (default 50) had
no such guard — the SessionAgent died at iter=50 in the middle of a
conversation. The R12 audit found at least 3 other independent
iteration caps with the same shape (MinimalActionPolicy, CacheAware
planner, EventDrivenActionPolicy event-handler done flag).

PR2 routes every per-policy cap through the SAME
:func:`effective_loop_max_iterations` helper so the lifecycle-aware
bypass applies UNIFORMLY. Plus a defense-in-depth branch in
``Agent.run_step``: when a policy reports ``policy_completed=True``
for a CONTINUOUS-mode agent, the loop downgrades to IDLE rather
than STOP — protecting against future cap-introductions that miss
the bypass at their own layer.

What we pin:

1. ``CodeGenerationActionPolicy.__init__`` routes
   ``max_code_iterations`` through ``effective_loop_max_iterations``
   so CONTINUOUS agents get ``None`` and never cap.
2. ``MinimalActionPolicy.__init__`` routes ``max_iterations`` through
   the same helper.
3. Both policies guard their cap-check on ``is not None``.
4. ``Agent.run_step`` downgrades ``policy_completed`` →
   ``AgentState.IDLE`` for CONTINUOUS agents.
"""

from __future__ import annotations

from pathlib import Path


_AGENTS_DIR = Path(__file__).resolve().parents[3]
# colony/src/polymathera/colony/agents
_CODE_GENERATION = _AGENTS_DIR / "patterns/actions/code_generation.py"
_MINIMAL = _AGENTS_DIR / "patterns/actions/minimal.py"
_BASE = _AGENTS_DIR / "base.py"


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# CodeGenerationActionPolicy (B1) — the cap that killed run12
# ---------------------------------------------------------------------------


def test_code_generation_imports_effective_loop_max_iterations() -> None:
    """Pin the import — without it the policy can't consult the
    lifecycle-aware helper."""

    src = _src(_CODE_GENERATION)
    assert "effective_loop_max_iterations" in src
    assert "from ...base import Agent, effective_loop_max_iterations" in src


def test_code_generation_routes_max_code_iterations_through_helper() -> None:
    """``__init__`` must store the helper-resolved value, not the
    raw kwarg. Pin the call shape so a refactor that bypasses the
    helper surfaces here."""

    src = _src(_CODE_GENERATION)
    assert (
        "self.max_code_iterations = effective_loop_max_iterations(" in src
    )
    # The helper is called with the canonical three kwargs.
    assert "configured_max_iterations=max_code_iterations" in src


def test_code_generation_cap_check_guards_on_not_none() -> None:
    """The cap check at plan_step must guard on
    ``self.max_code_iterations is not None`` since the helper
    returns ``None`` for CONTINUOUS-mode agents. Without the
    guard, ``int >= None`` raises TypeError and the agent crashes
    in a different way than the silent death it replaces — better
    than silence, but still not the intended behavior."""

    src = _src(_CODE_GENERATION)
    assert (
        "self.max_code_iterations is not None" in src
        and "self._code_iteration_count >= self.max_code_iterations" in src
    )


# ---------------------------------------------------------------------------
# MinimalActionPolicy (B7) — latent cap if anyone wires Minimal onto CONTINUOUS
# ---------------------------------------------------------------------------


def test_minimal_imports_effective_loop_max_iterations() -> None:
    src = _src(_MINIMAL)
    assert "from ...base import Agent, effective_loop_max_iterations" in src


def test_minimal_routes_max_iterations_through_helper() -> None:
    src = _src(_MINIMAL)
    assert "self._max_iterations = effective_loop_max_iterations(" in src
    assert "configured_max_iterations=max_iterations" in src


def test_minimal_cap_check_guards_on_not_none() -> None:
    src = _src(_MINIMAL)
    assert (
        "self._max_iterations is not None" in src
        and "self._iteration_count > self._max_iterations" in src
    )


# ---------------------------------------------------------------------------
# Agent.run_step defense-in-depth (B13) — CONTINUOUS never STOPs on policy_completed
# ---------------------------------------------------------------------------


def test_run_step_downgrades_policy_completed_for_continuous() -> None:
    """The agent loop's run_step must downgrade
    ``policy_completed=True`` to IDLE for CONTINUOUS-mode agents,
    rather than transitioning to STOPPED. Last line of defense
    against any future cap-introducer that misses the bypass at
    its own layer."""

    src = _src(_BASE)
    assert "self.metadata.lifecycle_mode == LifecycleMode.CONTINUOUS" in src
    # Pin that the CONTINUOUS branch routes to IDLE, not STOPPED.
    # The block we want pinned reads:
    #   if iteration_result.policy_completed:
    #       if ... == CONTINUOUS:
    #           ... self.state = AgentState.IDLE
    #       else:
    #           self.state = AgentState.STOPPED
    idx = src.find("if iteration_result.policy_completed:")
    assert idx > 0, "run_step's policy_completed branch not found"
    # Take a generous slice of the branch body.
    body = src[idx: idx + 2000]
    assert "CONTINUOUS" in body
    assert "AgentState.IDLE" in body
    assert "AgentState.STOPPED" in body
    # The IDLE assignment must appear BEFORE the STOPPED assignment
    # in the branch — that's how the if-else nests.
    idle_idx = body.find("self.state = AgentState.IDLE")
    stopped_idx = body.find("self.state = AgentState.STOPPED")
    assert 0 < idle_idx < stopped_idx, (
        "Expected the CONTINUOUS branch (→ IDLE) to appear before "
        "the ONE_SHOT branch (→ STOPPED) in run_step."
    )


def test_run_step_continuous_downgrade_emits_log() -> None:
    """The downgrade is operator-visible: a logger.info line
    explains why a CONTINUOUS agent's policy_completed signal was
    overridden. Pin the log shape so a future refactor that
    silences it surfaces here."""

    src = _src(_BASE)
    idx = src.find("if iteration_result.policy_completed:")
    body = src[idx: idx + 2000]
    assert "CONTINUOUS" in body and "policy_completed" in body
    # The log message contains both the agent_id and the IDLE
    # downgrade reason so an operator grepping the agent's log
    # can find every downgrade.
    assert "downgrading" in body
    assert "IDLE" in body
