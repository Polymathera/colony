"""``ScopedRetrievalCapability`` — single-shard / single-source retrieval.

Master §6.4 mode 1: "Single Tier-4 shard; for tool-use agents." The
caller passes a ``source_prefix`` (e.g., ``"docs:k_wave:"``) and gets
back the top-N hits from that shard only. No cross-source mixing,
no citation enforcement.

This is the lightest mode — it's the one a tool-use agent calls when
asking "how do I set kgrid.makeTime in k-Wave?" and wants to read the
single-tool docs.
"""

from __future__ import annotations

from ...tools import (
    Determinism,
    HITLFrequency,
    HeadlessReadiness,
    Licensing,
    ToolSpec,
)
from ..models import RetrievalQuery, RetrievalResult
from .base import RetrievalCapability


class ScopedRetrievalCapability(RetrievalCapability):
    mode = "scoped"
    spec = ToolSpec(
        name="retrieve_scoped",
        version="0.1.0",
        domain="knowledge",
        backend="in_process",
        capabilities=("retrieve_scoped",),
        determinism=Determinism.DETERMINISTIC,
        licensing=Licensing.APACHE_2_0,
        headless=HeadlessReadiness.NATIVE,
        hitl_frequency=HITLFrequency.AUTONOMOUS,
        interruptibility=False,
    )

    async def run(self, query: RetrievalQuery) -> RetrievalResult:
        if not query.source_prefix:
            return RetrievalResult(
                mode=self.mode,
                hits=(),
                total_candidates=0,
                extra={"reason": "scoped retrieval requires a source_prefix"},
            )
        if not query.text:
            return RetrievalResult(
                mode=self.mode,
                hits=(),
                total_candidates=0,
                extra={"reason": "scoped retrieval requires query.text"},
            )
        vectors = await self._deps.embedder.embed([query.text])
        if not vectors:
            return RetrievalResult(mode=self.mode, hits=(), total_candidates=0)
        hits = await self._deps.vector_store.search(
            query_vector=vectors[0],
            query=query,
        )
        return RetrievalResult(
            mode=self.mode,
            hits=tuple(hits),
            total_candidates=len(hits),
        )


__all__ = ("ScopedRetrievalCapability")
