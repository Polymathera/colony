"""Master §6.4 retrieval modes, registered as C2 ``ToolAdapter``s.

Five modes:

- ``scoped`` — single-shard / single-source retrieval (Tier-4-style).
- ``grounded`` — Tier 1–3 retrieval with enforced citation tags;
  each hit carries a verified ``CitationSpan``.
- ``graph`` — knowledge-graph query using the ``GraphStore``'s small
  Cypher-like DSL.
- ``budgeted`` — fits hits into a token budget; ranks by cosine
  similarity then truncates by accumulated token count.
- ``standards`` — Tier-2 retrieval with rulemaking-state filter
  (``effective_at`` window).

Every mode is an ``RetrievalAdapter`` (subclass of C2's ``ToolAdapter``)
that plugs into a ``ToolRegistry``, so agents resolve retrieval the
same way they resolve any other tool capability.
"""

from __future__ import annotations

from .base import RetrievalAdapter, RetrievalDeps
from .budgeted import BudgetedRetrievalAdapter
from .graph import GraphRetrievalAdapter
from .grounded import GroundedRetrievalAdapter
from .scoped import ScopedRetrievalAdapter
from .standards import StandardsRetrievalAdapter


__all__ = (
    "RetrievalAdapter",
    "RetrievalDeps",
    "BudgetedRetrievalAdapter",
    "GraphRetrievalAdapter",
    "GroundedRetrievalAdapter",
    "ScopedRetrievalAdapter",
    "StandardsRetrievalAdapter",
)
