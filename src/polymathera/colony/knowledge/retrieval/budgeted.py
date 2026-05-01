"""``BudgetedRetrievalAdapter`` — token-budget-aware retrieval.

Master §6.4 mode 4: "Fits results into a token budget and prioritizes
high-relevance / high-recency items." Same retrieval as
``ScopedRetrievalAdapter`` / ``GroundedRetrievalAdapter`` for the
ranking, but the result set is truncated by accumulated token count
rather than a fixed ``max_results``.

If ``query.max_tokens`` is None, falls back to a deterministic
default of 4096 tokens — a sensible inner-budget for "fit one
retrieval into a single LLM call".
"""

from __future__ import annotations

from datetime import timezone

from ...tools import (
    Determinism,
    HITLFrequency,
    HeadlessReadiness,
    Licensing,
    ToolSpec,
)
from ..models import RetrievalHit, RetrievalQuery, RetrievalResult
from .base import RetrievalAdapter


class BudgetedRetrievalAdapter(RetrievalAdapter):
    mode = "budgeted"
    spec = ToolSpec(
        name="retrieve_budgeted",
        version="0.1.0",
        domain="knowledge",
        backend="in_process",
        capabilities=("retrieve_budgeted",),
        determinism=Determinism.DETERMINISTIC,
        licensing=Licensing.APACHE_2_0,
        headless=HeadlessReadiness.NATIVE,
        hitl_frequency=HITLFrequency.AUTONOMOUS,
        interruptibility=False,
    )

    DEFAULT_MAX_TOKENS = 4096

    async def run(self, query: RetrievalQuery) -> RetrievalResult:
        if not query.text:
            return RetrievalResult(
                mode=self.mode, hits=(), total_candidates=0,
                extra={"reason": "budgeted retrieval requires query.text"},
            )
        budget = query.max_tokens or self.DEFAULT_MAX_TOKENS
        # Pull more candidates than ``max_results`` so the budget
        # filter has room to work.
        oversample = query.model_copy(
            update={"max_results": max(query.max_results * 4, 50)}
        )
        vectors = await self._deps.embedder.embed([query.text])
        if not vectors:
            return RetrievalResult(mode=self.mode, hits=(), total_candidates=0)
        candidates = list(
            await self._deps.vector_store.search(
                query_vector=vectors[0], query=oversample,
            )
        )
        # Re-sort by (-score, -recency) to prefer recent + relevant.
        candidates.sort(
            key=lambda h: (
                -h.score,
                -_recency_key(h),
                h.chunk.chunk_id,
            ),
        )
        accepted: list[RetrievalHit] = []
        used = 0
        for hit in candidates:
            chunk_tokens = max(1, hit.chunk.token_count)
            if used + chunk_tokens > budget and accepted:
                break
            if used + chunk_tokens > budget and not accepted:
                # The first hit is bigger than the entire budget.
                # Accept it anyway (truncating would defeat the point);
                # later hits are dropped.
                accepted.append(hit.model_copy(update={"rank": 0}))
                used += chunk_tokens
                break
            accepted.append(hit.model_copy(update={"rank": len(accepted)}))
            used += chunk_tokens
            if len(accepted) >= query.max_results:
                break
        return RetrievalResult(
            mode=self.mode,
            hits=tuple(accepted),
            total_candidates=len(candidates),
            used_tokens=used,
            extra={"budget_tokens": budget},
        )


def _recency_key(hit) -> float:
    eff = hit.chunk.effective_at
    if eff is None:
        return 0.0
    if eff.tzinfo is None:
        eff = eff.replace(tzinfo=timezone.utc)
    return eff.timestamp()


__all__ = ("BudgetedRetrievalAdapter",)
