"""Master §6.4 retrieval modes, exposed as :class:`LocalToolCapability` subclasses.

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

Every mode is a :class:`RetrievalCapability` (subclass of
:class:`~polymathera.colony.agents.patterns.capabilities.tool.LocalToolCapability`)
that an agent mounts via the standard blueprint flow; the LLM planner
discovers it via the ``"tool"`` + ``"knowledge"`` + ``"retrieval"``
capability tags.

The legacy ``*RetrievalCapability`` names remain as deprecated aliases
for one release cycle so external imports keep working while
consumers migrate.
"""

from __future__ import annotations

from .base import RetrievalCapability, RetrievalCapability, RetrievalDeps
from .budgeted import BudgetedRetrievalCapability, BudgetedRetrievalCapability
from .graph import GraphRetrievalCapability, GraphRetrievalCapability
from .grounded import GroundedRetrievalCapability, GroundedRetrievalCapability
from .scoped import ScopedRetrievalCapability, ScopedRetrievalCapability
from .standards import StandardsRetrievalCapability, StandardsRetrievalCapability


__all__ = (
    # Current names
    "RetrievalCapability",
    "RetrievalDeps",
    "BudgetedRetrievalCapability",
    "GraphRetrievalCapability",
    "GroundedRetrievalCapability",
    "ScopedRetrievalCapability",
    "StandardsRetrievalCapability",
    # Deprecated aliases (pending removal)
    "RetrievalCapability",
    "BudgetedRetrievalCapability",
    "GraphRetrievalCapability",
    "GroundedRetrievalCapability",
    "ScopedRetrievalCapability",
    "StandardsRetrievalCapability",
)
