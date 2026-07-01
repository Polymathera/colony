"""The self-improving training loop, as a gated orchestrator.

``run_flywheel`` chains assemble → train → evaluate → **gate** → publish.
Every step is an injectable seam, so the orchestrator is generic (it
knows nothing about any particular corpus, trainer, or serving catalog)
and unit-testable. The gate is what makes the loop *closed*: a candidate
is published only if it clears :func:`promotion_gate` (beats the
incumbent and survives the degeneracy guards).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from .promotion import EvalSummary, PromotionDecision, promotion_gate


class FlywheelOutcome(BaseModel):
    model_id: str
    dataset_version: str
    promoted: bool
    published_name: str | None = None
    candidate_eval: EvalSummary
    decision: PromotionDecision


def run_flywheel(
    *,
    base_model: str,
    assemble: Callable[[], Any],
    train: Callable[[str, Any], Any],
    evaluate: Callable[[Any], EvalSummary],
    publish: Callable[[Any], str],
    incumbent_eval: EvalSummary | None = None,
    gate: Callable[..., PromotionDecision] = promotion_gate,
    min_gain: float = 0.0,
    model_id_of: Callable[[Any], str] = lambda m: m.model_id,
) -> FlywheelOutcome:
    """Run one closed iteration. ``assemble`` → dataset version;
    ``train(base_model, dataset)`` → candidate model; ``evaluate`` →
    its held-out summary; the gate decides; ``publish`` runs only on
    promotion."""

    dataset_version = assemble()
    candidate = train(base_model, dataset_version)
    candidate_eval = evaluate(candidate)
    decision = gate(candidate_eval, incumbent_eval, min_gain=min_gain)
    published = publish(candidate) if decision.promote else None
    return FlywheelOutcome(
        model_id=model_id_of(candidate),
        dataset_version=str(dataset_version),
        promoted=decision.promote,
        published_name=published,
        candidate_eval=candidate_eval,
        decision=decision,
    )


__all__ = ("FlywheelOutcome", "run_flywheel")
