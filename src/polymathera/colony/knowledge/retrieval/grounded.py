"""``GroundedRetrievalCapability`` — Tier 1–3 retrieval with enforced citations.

Master §6.4 mode 2: every hit must carry a verified ``CitationSpan``
(non-empty ``source_uri`` + character span). The adapter forces
``query.require_citations = True`` regardless of the caller's input,
so a chunk indexed without citations is never surfaced through this
mode (it would still be reachable via ``retrieve_scoped``).

Used by ``DesignRationaleAgent`` and ``KnowledgeCuratorAgent`` per
master §3.5; useful for any agent whose reply must show its work.
"""

from __future__ import annotations

from ...tools import (
    Determinism,
    HITLFrequency,
    HeadlessReadiness,
    Licensing,
    ToolSpec,
)
from ..models import CorpusTier, RetrievalQuery, RetrievalResult
from .base import RetrievalCapability


class GroundedRetrievalCapability(RetrievalCapability):
    mode = "grounded"
    spec = ToolSpec(
        name="retrieve_grounded",
        version="0.1.0",
        domain="knowledge",
        backend="in_process",
        capabilities=("retrieve_grounded",),
        determinism=Determinism.DETERMINISTIC,
        licensing=Licensing.APACHE_2_0,
        headless=HeadlessReadiness.NATIVE,
        hitl_frequency=HITLFrequency.AUTONOMOUS,
        interruptibility=False,
    )

    DEFAULT_TIERS: tuple[CorpusTier, ...] = (
        CorpusTier.TIER_1_FOUNDATIONS,
        CorpusTier.TIER_2_STANDARDS,
        CorpusTier.TIER_3_RESEARCH,
    )

    async def run(self, query: RetrievalQuery) -> RetrievalResult:
        if not query.text:
            return RetrievalResult(
                mode=self.mode, hits=(), total_candidates=0,
                extra={"reason": "grounded retrieval requires query.text"},
            )
        # Force grounded constraints.
        tiers = query.tiers or self.DEFAULT_TIERS
        bound = query.model_copy(
            update={"require_citations": True, "tiers": tiers},
        )
        vectors = await self._deps.embedder.embed([bound.text])
        if not vectors:
            return RetrievalResult(mode=self.mode, hits=(), total_candidates=0)
        hits = await self._deps.vector_store.search(
            query_vector=vectors[0], query=bound,
        )
        # Final safety check: every hit's citation must have a non-empty source_uri.
        verified = tuple(
            hit for hit in hits
            if hit.chunk.citation.source_uri
        )
        return RetrievalResult(
            mode=self.mode,
            hits=verified,
            total_candidates=len(hits),
            extra={
                "tiers": [t.value for t in tiers],
                "dropped_uncited": len(hits) - len(verified),
            },
        )


__all__ = ("GroundedRetrievalCapability")
