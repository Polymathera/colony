"""Knowledge-corpus stores: vector index + knowledge graph.

These are deliberately distinct from ``agents/patterns/memory/backends/``
(per master §3.2 — agent memory is a separate layer). The vector
store here is corpus-wide and citation-aware; the graph store is
RDF-shaped (subject / predicate / object).
"""

from __future__ import annotations

from .graph import (
    GraphEdge,
    GraphNode,
    GraphQueryResult,
    GraphStore,
    GraphStoreError,
    InMemoryGraphStore,
    KuzuGraphStore,
)
from .image import (
    ImageStore,
    ImageStoreError,
    InMemoryImageStore,
    LocalFsImageStore,
)
from .vector import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorStore,
    VectorStoreError,
)


__all__ = (
    "VectorStore",
    "VectorStoreError",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "GraphStore",
    "GraphStoreError",
    "GraphNode",
    "GraphEdge",
    "GraphQueryResult",
    "InMemoryGraphStore",
    "KuzuGraphStore",
    "ImageStore",
    "ImageStoreError",
    "InMemoryImageStore",
    "LocalFsImageStore",
)
