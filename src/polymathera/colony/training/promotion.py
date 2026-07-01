"""Promotion gating for self-improving training loops.

A freshly trained candidate must beat the incumbent on a held-out
evaluation AND survive degeneracy guards before it is promoted (served).
These gates make an automatic train→serve loop *closed*: a candidate is
published only after clearing them, so a collapsed or reward-hacking
adapter can't silently promote itself. The guards are diversity +
degeneracy **heuristics, not a correctness proof** — the guard set and
the degeneracy predicate are injectable so callers add stronger checks.
The gate operates on :class:`EvalSummary` statistics, so it is
domain-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field


class EvalSummary(BaseModel):
    """Aggregate statistics over one model's held-out evaluation."""

    mean_reward: float = 0.0
    num_samples: int = 0
    distinct_ratio: float = 1.0   # unique completions / total — output diversity
    degenerate_ratio: float = 0.0  # degenerate completions / total
    label: str = ""


class GuardResult(BaseModel):
    name: str
    passed: bool
    reason: str = ""


Guard = Callable[[EvalSummary], GuardResult]


def collapse_guard(summary: EvalSummary, *, min_distinct_ratio: float = 0.5) -> GuardResult:
    """Fail when output diversity has collapsed (mode collapse)."""
    ok = summary.distinct_ratio >= min_distinct_ratio
    return GuardResult(
        name="collapse", passed=ok,
        reason="" if ok else
        f"output diversity collapsed: distinct_ratio {summary.distinct_ratio:.2f} < {min_distinct_ratio}",
    )


def reward_hack_guard(summary: EvalSummary, *, max_degenerate_ratio: float = 0.2) -> GuardResult:
    """Fail when degenerate completions are scoring reward — the reward is
    being gamed rather than the task solved."""
    ok = summary.degenerate_ratio <= max_degenerate_ratio
    return GuardResult(
        name="reward_hack", passed=ok,
        reason="" if ok else
        f"degenerate outputs scoring reward: degenerate_ratio {summary.degenerate_ratio:.2f} > {max_degenerate_ratio}",
    )


DEFAULT_GUARDS: tuple[Guard, ...] = (collapse_guard, reward_hack_guard)


class PromotionDecision(BaseModel):
    promote: bool
    reason: str
    candidate: EvalSummary
    incumbent: EvalSummary | None = None
    guard_results: list[GuardResult] = Field(default_factory=list)


def promotion_gate(
    candidate: EvalSummary,
    incumbent: EvalSummary | None = None,
    *,
    min_gain: float = 0.0,
    guards: Sequence[Guard] = DEFAULT_GUARDS,
) -> PromotionDecision:
    """Promote the candidate iff it passes every guard and beats the
    incumbent's mean reward by at least ``min_gain`` (first model with no
    incumbent promotes on guards alone)."""

    guard_results = [g(candidate) for g in guards]
    failed = [r for r in guard_results if not r.passed]
    if failed:
        return PromotionDecision(
            promote=False, reason="; ".join(r.reason for r in failed),
            candidate=candidate, incumbent=incumbent, guard_results=guard_results,
        )
    if incumbent is None:
        return PromotionDecision(
            promote=True, reason="no incumbent — first promotion",
            candidate=candidate, incumbent=None, guard_results=guard_results,
        )
    gain = candidate.mean_reward - incumbent.mean_reward
    promote = gain >= min_gain
    reason = (
        f"reward +{gain:.3f} >= min_gain {min_gain}" if promote
        else f"reward gain {gain:.3f} < min_gain {min_gain}"
    )
    return PromotionDecision(
        promote=promote, reason=reason,
        candidate=candidate, incumbent=incumbent, guard_results=guard_results,
    )


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return " ".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


def default_is_degenerate(text: str, *, min_length: int = 8) -> bool:
    """A completion is degenerate when it is too short or heavily repeats a
    handful of tokens. Replaceable via ``summarize_eval(is_degenerate=…)``."""
    stripped = text.strip()
    if len(stripped) < min_length:
        return True
    words = stripped.split()
    return len(words) >= 4 and len(set(words)) <= max(1, len(words) // 4)


def summarize_eval(
    completions: Sequence[Any],
    rewards: Sequence[float],
    *,
    label: str = "",
    is_degenerate: Callable[[str], bool] = default_is_degenerate,
) -> EvalSummary:
    """Build an :class:`EvalSummary` from raw eval completions and their
    verifiable rewards."""

    n = len(completions)
    if n == 0:
        return EvalSummary(label=label)
    texts = [_completion_text(c) for c in completions]
    distinct = len({t.strip() for t in texts}) / n
    degenerate = sum(1 for t in texts if is_degenerate(t)) / n
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    return EvalSummary(
        mean_reward=mean_reward, num_samples=n,
        distinct_ratio=distinct, degenerate_ratio=degenerate, label=label,
    )


__all__ = (
    "DEFAULT_GUARDS",
    "EvalSummary",
    "Guard",
    "GuardResult",
    "PromotionDecision",
    "collapse_guard",
    "default_is_degenerate",
    "promotion_gate",
    "reward_hack_guard",
    "summarize_eval",
)
