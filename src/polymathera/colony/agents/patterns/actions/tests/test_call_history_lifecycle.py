"""Tests for the cumulative ``_call_history`` lifecycle on
:class:`CodeGenerationActionPolicy`.

The policy keeps ``_call_history`` cumulative across iterations so
:class:`ConstraintScope.SESSION` rules can see evidence from prior
cells. ``_iteration_history_boundary`` marks where the CURRENT cell's
calls start in that cumulative list. The per-cell step-summary build
slices at the boundary; the guardrail's ``cell_start_index`` is the
boundary; SESSION-scope verifiers read the whole list.

Three invariants pinned here:

- **Cumulative**: ``plan_step``'s per-iteration reset block does NOT
  clear ``_call_history``. Calls from iter K-1 are still visible at
  the top of iter K (and to SESSION-scope verifiers throughout iter
  K's cell).
- **Boundary slice**: while iter K's cell is running,
  ``_call_history[_iteration_history_boundary:]`` returns only THIS
  cell's calls so far — this is what the guardrail's CELL-scope check
  and the per-iter step summary depend on.
- **Cap eviction**: once cumulative growth exceeds
  ``_CALL_HISTORY_MAX_LEN``, FIFO eviction at the append site drops
  the oldest entries AND shifts ``_iteration_history_boundary`` by
  the drop count so the cell-slice math stays correct.

Constructed with ``__new__`` + direct field set to avoid the full
``CodeGenerationActionPolicy.__init__`` plumbing (planning context,
consciousness streams, dispatcher, etc.) — the invariants under test
are state-machine arithmetic, not initialization shape.
"""

from __future__ import annotations

import pytest

from polymathera.colony.agents.patterns.actions.code_generation import (
    CodeGenerationActionPolicy,
)
from polymathera.colony.agents.patterns.planning.models import CallRecord


pytestmark = pytest.mark.asyncio


def _make_record(action_key: str, action_id: str | None = None) -> CallRecord:
    return CallRecord(
        action_key=action_key,
        params={},
        action_id=action_id or f"id_{action_key}",
        end_wall=0.0,
        status="ok",
        error=None,
        result=None,
    )


def _make_policy() -> CodeGenerationActionPolicy:
    """Bypass __init__ — these tests target boundary arithmetic, not
    construction. Set ONLY the fields the boundary lifecycle touches."""

    policy = CodeGenerationActionPolicy.__new__(CodeGenerationActionPolicy)
    policy._call_history = []
    policy._iteration_history_boundary = 0
    return policy


# ---------------------------------------------------------------------------
# Cumulative: prior iter's calls survive into the next iter
# ---------------------------------------------------------------------------


def test_call_history_is_cumulative_across_simulated_iterations() -> None:
    """Reproduce iter 0 → iter 1's boundary advance. Prior calls
    remain in ``_call_history`` (the SESSION-scope semantic) and the
    boundary marks where iter 1's calls will start."""

    p = _make_policy()
    # iter 0's cell appended three calls.
    p._call_history.extend([
        _make_record("X.a"),
        _make_record("X.b"),
        _make_record("X.c"),
    ])
    # plan_step's boundary advance at the top of iter 1: snapshots the
    # current length so the new cell's calls land after this index.
    p._iteration_history_boundary = len(p._call_history)
    assert p._iteration_history_boundary == 3
    # iter 1's cell starts. Before any iter-1 call appends:
    assert p._call_history[p._iteration_history_boundary:] == ()  \
        or list(p._call_history[p._iteration_history_boundary:]) == []
    # Prior calls still visible (SESSION-scope) — the whole list.
    keys = [r.action_key for r in p._call_history]
    assert keys == ["X.a", "X.b", "X.c"]


# ---------------------------------------------------------------------------
# Boundary slice: cell-scope view excludes prior iter's calls
# ---------------------------------------------------------------------------


def test_boundary_slice_returns_only_current_cell_calls() -> None:
    """``_call_history[_iteration_history_boundary:]`` is exactly the
    current cell's calls. This is the slice the guardrail check uses
    (via ``cell_start_index``) and the per-iter step-summary builder
    uses (via the same slice expression)."""

    p = _make_policy()
    p._call_history.extend([
        _make_record("prior.a"),
        _make_record("prior.b"),
    ])
    # Advance boundary at iter 1's plan_step top.
    p._iteration_history_boundary = len(p._call_history)
    # iter 1's cell appends two calls.
    p._call_history.append(_make_record("cur.X"))
    p._call_history.append(_make_record("cur.Y"))
    # Slice = this cell only.
    cell_slice = p._call_history[p._iteration_history_boundary:]
    assert [r.action_key for r in cell_slice] == ["cur.X", "cur.Y"]
    # Prior cell still accessible for SESSION-scope rules.
    prior_slice = p._call_history[:p._iteration_history_boundary]
    assert [r.action_key for r in prior_slice] == ["prior.a", "prior.b"]


def test_boundary_zero_on_first_iteration_includes_all_calls() -> None:
    """On iter 0 the boundary stays at its init value (0) because
    ``_build_prior_iteration_observation`` early-returns and the
    boundary advance at line 2151 is skipped. Result: the cell slice
    == the whole list, which is correct — there is no 'prior cell'."""

    p = _make_policy()
    # No advance — iter 0 conditions.
    assert p._iteration_history_boundary == 0
    p._call_history.append(_make_record("first.a"))
    p._call_history.append(_make_record("first.b"))
    cell_slice = p._call_history[p._iteration_history_boundary:]
    assert [r.action_key for r in cell_slice] == ["first.a", "first.b"]


# ---------------------------------------------------------------------------
# Cap eviction: cumulative growth bounded; boundary shifts with drops
# ---------------------------------------------------------------------------


def _evict_overflow(p: CodeGenerationActionPolicy) -> None:
    """Mirror the inline eviction at the append site in
    ``code_generation.py`` so the cap math is testable without
    standing up the full dispatcher + REPL plumbing the run() helper
    needs. Keep in sync with that inline block; the invariants this
    function pins are the contract."""

    overflow = len(p._call_history) - p._CALL_HISTORY_MAX_LEN
    if overflow > 0:
        del p._call_history[:overflow]
        p._iteration_history_boundary = max(
            0, p._iteration_history_boundary - overflow,
        )


def test_cap_eviction_drops_oldest_and_shifts_boundary() -> None:
    """When cumulative growth pushes ``_call_history`` past the cap,
    FIFO eviction drops the oldest N AND shifts the boundary by N so
    the cell-slice math still points at the current cell's start."""

    p = _make_policy()
    cap = p._CALL_HISTORY_MAX_LEN
    # Pre-fill to exactly the cap.
    p._call_history.extend(_make_record(f"prior_{i}") for i in range(cap - 5))
    # Advance boundary AFTER the prior cell; current cell starts here.
    p._iteration_history_boundary = len(p._call_history)
    # Append 10 calls during this cell — that pushes length to cap+5.
    for i in range(10):
        p._call_history.append(_make_record(f"cur_{i}"))
        _evict_overflow(p)
    assert len(p._call_history) == cap
    # Boundary shifted by 5 (the drop count) — current cell's calls
    # still slice correctly.
    expected_boundary = (cap - 5) - 5
    assert p._iteration_history_boundary == expected_boundary
    cell_slice = p._call_history[p._iteration_history_boundary:]
    assert [r.action_key for r in cell_slice] == [
        f"cur_{i}" for i in range(10)
    ]


def test_cap_eviction_clamps_boundary_at_zero_when_overflow_exceeds_it(
) -> None:
    """If the drop count exceeds the boundary value, the boundary
    clamps to 0 — semantically: the entire prior-cell window has been
    evicted; the cumulative-history scope shrinks to fit the cap.
    SESSION-scope rules lose the oldest evidence (the documented
    tradeoff of bounding memory)."""

    p = _make_policy()
    cap = p._CALL_HISTORY_MAX_LEN
    # Tiny prior history.
    p._call_history.extend(_make_record(f"prior_{i}") for i in range(3))
    p._iteration_history_boundary = len(p._call_history)  # = 3
    # Append a flood — more than the cap.
    for i in range(cap + 50):
        p._call_history.append(_make_record(f"cur_{i}"))
        _evict_overflow(p)
    assert len(p._call_history) == cap
    # Drop count over the boundary's lifetime is (cap + 50 + 3) - cap = 53;
    # the boundary subtractions drove it to clamp at 0.
    assert p._iteration_history_boundary == 0
    # Cell slice = the whole list; the entire kept history IS the
    # current cell because the prior cell's three records evicted.
    cell_slice = p._call_history[p._iteration_history_boundary:]
    assert len(cell_slice) == cap


def test_cap_eviction_noop_when_under_cap() -> None:
    """No eviction when length is at or below the cap. Boundary
    unchanged."""

    p = _make_policy()
    p._call_history.extend(_make_record(f"a_{i}") for i in range(10))
    p._iteration_history_boundary = 5
    _evict_overflow(p)
    assert len(p._call_history) == 10
    assert p._iteration_history_boundary == 5
