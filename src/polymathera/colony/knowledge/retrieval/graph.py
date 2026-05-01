"""``GraphRetrievalAdapter`` — knowledge-graph query mode.

Master §6.4 mode 3: cross-phase consistency-check / "tell me what
depends on the lithium choice"-style queries. Runs a Cypher-like
query against the ``GraphStore``; surfaces the matched subgraph as
the result, plus a ranked set of ``RetrievalHit``s for the *chunks*
that produced the matched edges (so the caller can show provenance).

Either ``query.graph_query`` or a default-constructed neighbourhood
walk based on ``query.text`` (interpreted as a node-id seed) is used;
the former is the explicit mode, the latter is convenience.
"""

from __future__ import annotations

from ...tools import (
    Determinism,
    HITLFrequency,
    HeadlessReadiness,
    Licensing,
    ToolSpec,
)
from ..models import RetrievalHit, RetrievalQuery, RetrievalResult
from ..stores.graph import GraphQueryResult, _node_id_for
from .base import RetrievalAdapter


class GraphRetrievalAdapter(RetrievalAdapter):
    mode = "graph"
    spec = ToolSpec(
        name="retrieve_graph",
        version="0.1.0",
        domain="knowledge",
        backend="in_process",
        capabilities=("retrieve_graph",),
        determinism=Determinism.DETERMINISTIC,
        licensing=Licensing.APACHE_2_0,
        headless=HeadlessReadiness.NATIVE,
        hitl_frequency=HITLFrequency.AUTONOMOUS,
        interruptibility=False,
    )

    async def run(self, query: RetrievalQuery) -> RetrievalResult:
        if self._deps.graph_store is None:
            return RetrievalResult(
                mode=self.mode, hits=(), total_candidates=0,
                extra={"reason": "no graph_store configured"},
            )
        if query.graph_query:
            try:
                gqr = await self._deps.graph_store.query(query.graph_query)
            except Exception as exc:  # noqa: BLE001
                return RetrievalResult(
                    mode=self.mode, hits=(), total_candidates=0,
                    extra={"error": f"graph query failed: {exc}"},
                )
            return _result_from_graph(gqr, query, self.mode)
        if query.text:
            seed = _node_id_for(query.text)
            gqr = await self._deps.graph_store.neighbours(
                seed,
                depth=int(query.extra.get("depth", 2)),
            )
            return _result_from_graph(gqr, query, self.mode, seed=seed)
        return RetrievalResult(
            mode=self.mode, hits=(), total_candidates=0,
            extra={"reason": "graph retrieval requires graph_query or text"},
        )


def _result_from_graph(
    gqr: GraphQueryResult,
    query: RetrievalQuery,
    mode: str,
    *,
    seed: str | None = None,
) -> RetrievalResult:
    # We do not turn graph nodes into ``RetrievalHit`` records (those
    # are chunk-shaped). Instead the result's ``extra`` carries the
    # subgraph; ``hits`` is empty unless the caller also wants chunk
    # provenance, which it can compute by looking up ``citation_uri``
    # on each edge.
    return RetrievalResult(
        mode=mode,
        hits=(),
        total_candidates=len(gqr.edges),
        extra={
            "seed": seed,
            "node_ids": [n.node_id for n in gqr.nodes],
            "edges": [
                {
                    "edge_id": e.edge_id,
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "predicate": e.predicate,
                    "confidence": e.confidence,
                    "citation_uri": e.citation_uri,
                }
                for e in gqr.edges
            ],
            "paths": [list(p) for p in gqr.paths],
        },
    )


__all__ = ("GraphRetrievalAdapter",)
