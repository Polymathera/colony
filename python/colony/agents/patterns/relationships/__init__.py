"""Relationship graph building from analysis results.

This package provides data models and extraction for entity relationships:
- Relationship: Typed edge between entities (pages, symbols, etc.)
- RelationshipGraph: In-memory graph of relationships with indexes
- RelationshipGraphBuilder: Extracts relationships from ScopeAwareResult

Behavioral graph operations (discovery, blackboard publish/subscribe,
query learning) live in PageGraphCapability.
"""

from .builder import RelationshipGraphBuilder, Relationship, RelationshipGraph

__all__ = [
    "RelationshipGraphBuilder",
    "Relationship",
    "RelationshipGraph",
]

