"""``StandardsRetrievalCapability`` — Tier-2 retrieval with rulemaking-state filter.

Master §6.4 mode 5: "Tier 2 with **rulemaking-state filter** (only
return clauses whose effective regulatory state matches the query
date)." Important for any deployment that produces regulatory
artefacts: a clause superseded by a later revision must NOT surface
when answering a question scoped to a date before the supersession.

Implementation:

- Filter to ``CorpusTier.TIER_2_STANDARDS`` (other tiers are dropped
  even if matching).
- Filter to chunks whose ``effective_at`` ≤ ``query.effective_at``,
  defaulting to ``datetime.now(timezone.utc)``.
- A chunk is *superseded* iff there's another chunk with the same
  ``source`` + ``section_path`` and a later ``effective_at`` ≤
  ``query.effective_at``. Superseded chunks are dropped.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from ...tools import (
    Determinism,
    HITLFrequency,
    HeadlessReadiness,
    Licensing,
    ToolSpec,
)
from ..models import CorpusTier, RetrievalHit, RetrievalQuery, RetrievalResult
from .base import RetrievalCapability


class StandardsRetrievalCapability(RetrievalCapability):
    mode = "standards"
    spec = ToolSpec(
        name="retrieve_standards",
        version="0.1.0",
        domain="knowledge",
        backend="in_process",
        capabilities=("retrieve_standards",),
        determinism=Determinism.DETERMINISTIC,
        licensing=Licensing.APACHE_2_0,
        headless=HeadlessReadiness.NATIVE,
        hitl_frequency=HITLFrequency.REVIEW_MILESTONES,
        interruptibility=False,
    )

    async def run(self, query: RetrievalQuery) -> RetrievalResult:
        if not query.text:
            return RetrievalResult(
                mode=self.mode, hits=(), total_candidates=0,
                extra={"reason": "standards retrieval requires query.text"},
            )
        as_of = query.effective_at or datetime.now(timezone.utc)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        # Force Tier-2 + larger candidate set so the supersession
        # filter has room to work.
        bound = query.model_copy(
            update={
                "tiers": (CorpusTier.TIER_2_STANDARDS,),
                "max_results": max(query.max_results * 3, 30),
                "effective_at": as_of,
            },
        )
        vectors = await self._deps.embedder.embed([bound.text])
        if not vectors:
            return RetrievalResult(mode=self.mode, hits=(), total_candidates=0)
        raw_hits = await self._deps.vector_store.search(
            query_vector=vectors[0], query=bound,
        )

        # Drop chunks whose ``effective_at`` is in the future relative
        # to ``as_of`` (not yet in force).
        in_force: list[RetrievalHit] = []
        for hit in raw_hits:
            eff = hit.chunk.effective_at
            if eff is None:
                # Treat as eternal (unversioned).
                in_force.append(hit)
                continue
            if eff.tzinfo is None:
                eff = eff.replace(tzinfo=timezone.utc)
            if eff <= as_of:
                in_force.append(hit)

        # Supersession: per (source, section_path), keep only the
        # latest ``effective_at`` ≤ as_of.
        latest_by_clause: dict[tuple[str, str], RetrievalHit] = {}
        for hit in in_force:
            key = (hit.chunk.source, hit.chunk.section_path)
            existing = latest_by_clause.get(key)
            if existing is None:
                latest_by_clause[key] = hit
                continue
            existing_eff = existing.chunk.effective_at
            current_eff = hit.chunk.effective_at
            if existing_eff is None and current_eff is None:
                # Tie: keep the higher-scoring hit.
                if hit.score > existing.score:
                    latest_by_clause[key] = hit
                continue
            if existing_eff is None:
                latest_by_clause[key] = hit
                continue
            if current_eff is None:
                continue
            if current_eff > existing_eff:
                latest_by_clause[key] = hit

        survivors = sorted(
            latest_by_clause.values(),
            key=lambda h: (-h.score, h.chunk.chunk_id),
        )[: query.max_results]

        return RetrievalResult(
            mode=self.mode,
            hits=tuple(
                hit.model_copy(update={"rank": i})
                for i, hit in enumerate(survivors)
            ),
            total_candidates=len(raw_hits),
            extra={
                "as_of": as_of.isoformat(),
                "dropped_not_yet_in_force": len(raw_hits) - len(in_force),
                "dropped_superseded": len(in_force) - len(latest_by_clause),
            },
        )


__all__ = ("StandardsRetrievalCapability")
