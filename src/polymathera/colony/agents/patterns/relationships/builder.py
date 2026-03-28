"""Relationship graph building from analysis results.

This module extracts relationships from analysis results and builds knowledge graphs.
Relationships can be:
- Dependencies (imports, function calls)
- Data flow (variable data flows)
- Similarity (semantic similarity, pattern instances)
- Traceability (requirement to code)
- Temporal (happens-before, causality)
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from ..scope import ScopeAwareResult
from ...blackboard.protocol import RelationshipProtocol


class Relationship(BaseModel):
    """Represents a relationship between two entities.

    Entities can be:
    - Pages/shards
    - Functions/classes
    - Modules/components
    - Requirements/implementations

    Can represent:
    - Code dependencies (ModuleA imports ModuleB)
    - Data flow (Variable x flows to variable y)
    - Traceability (Requirement R satisfied by Code C)
    - Similarity (Pattern P1 similar to Pattern P2)
    - Temporal (Event E1 happens before Event E2)
    """

    relationship_id: str = Field(
        default_factory=lambda: f"rel_{int(time.time() * 1000)}",
        description="Unique identifier"
    )

    source_id: str = Field(
        description="Source entity ID (page_id, symbol, etc.)"
    )

    target_id: str = Field(
        description="Target entity ID"
    )

    relationship_type: str = Field(
        description="Type of relationship (dependency, alias, dataflow, similarity, etc.)"
    )

    bidirectional: bool = Field(
        default=False,
        description="Whether relationship is bidirectional"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this relationship"
    )

    weight: float = Field(
        default=1.0,
        description="Relationship strength/weight"
    )

    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence supporting this relationship"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional relationship metadata"
    )

    created_at: float = Field(
        default_factory=time.time
    )

    discovered_by: str | None = Field(
        default=None,
        description="Agent ID that discovered this relationship"
    )

    def get_key(self) -> str:
        """Get unique key for this relationship.

        Returns:
            Unique relationship key
        """
        return RelationshipProtocol.relationship_key(self.source_id, self.target_id, self.relationship_type, namespace="relationships")

    @staticmethod
    def get_key_pattern(
        source_id: str | None = None,
        target_id: str | None = None,
        relationship_type: str | None = None
    ) -> str:
        """Get key pattern for querying relationships.

        Args:
            source_id: Optional filter by source
            target_id: Optional filter by target
            relationship_type: Optional filter by type

        Returns:
            Key pattern for querying relationships
        """
        # Build query pattern
        src = source_id or "*"
        tgt = target_id or "*"
        rel = relationship_type or "*"
        return RelationshipProtocol.relationship_key(src, tgt, rel, namespace="relationships")


class RelationshipGraph(BaseModel):
    """Graph of relationships between entities.

    Maintains a directed graph where:
    - Nodes are entities (pages, symbols, patterns, etc.)
    - Edges are relationships with type and confidence

    Supports:
    - Adding/removing relationships
    - Querying relationships by type, source, target
    - Graph traversal
    - Subgraph extraction
    """

    # TODO: Store this graph in a graph database or the blackboard for scalability

    graph_id: str = Field(
        default_factory=lambda: f"graph_{int(time.time() * 1000)}",
        description="Unique graph ID"
    )

    nodes: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Node metadata by node ID"
    )

    edges: dict[str, list[Relationship]] = Field(
        default_factory=dict,
        description="Edges from each node (rel_id -> relationships)"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Graph metadata"
    )

    by_source: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Source ID -> [rel_ids]"
    )

    by_target: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Target ID -> [rel_ids]"
    )

    by_type: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Type -> [rel_ids]"
    )

    def add_node(
        self,
        node_id: str,
        node_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add node to graph.

        Args:
            node_id: Node ID
            node_type: Optional node type (stored in metadata under ``"node_type"``)
            metadata: Optional node metadata
        """
        node_meta = metadata or {}
        if node_type is not None:
            node_meta["node_type"] = node_type
        if node_id not in self.nodes:
            self.nodes[node_id] = node_meta
        else:
            self.nodes[node_id].update(node_meta)
        if node_id not in self.edges:
            self.edges[node_id] = []

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        confidence: float = 1.0,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> Relationship:
        """Convenience method to create and add a relationship as an edge.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relationship_type: Type of relationship
            confidence: Confidence in this relationship (0.0-1.0)
            weight: Relationship strength/weight
            metadata: Additional edge metadata

        Returns:
            The created Relationship
        """
        rel = Relationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            confidence=confidence,
            weight=weight,
            metadata=metadata or {},
        )
        self.add_relationship(rel)
        return rel

    def add_relationship(self, relationship: Relationship) -> None:
        """Add relationship to graph.

        Args:
            relationship: Relationship to add
        """
        # Ensure nodes exist
        self.add_node(relationship.source_id)
        self.add_node(relationship.target_id)

        rel_id = relationship.relationship_id

        # Store relationship
        self.edges[rel_id] = relationship

        # Index by source
        def index_by_source(source_id: str, rel_id: str):
            if source_id not in self.by_source:
                self.by_source[source_id] = []
            self.by_source[source_id].append(rel_id)

        index_by_source(relationship.source_id, rel_id)
        if relationship.bidirectional:
            index_by_source(relationship.target_id, rel_id)

        # Index by target
        def index_by_target(target_id: str, rel_id: str):
            if target_id not in self.by_target:
                self.by_target[target_id] = []
            self.by_target[target_id].append(rel_id)

        index_by_target(relationship.target_id, rel_id)
        if relationship.bidirectional:
            index_by_target(relationship.source_id, rel_id)

        # Index by type
        if relationship.relationship_type not in self.by_type:
            self.by_type[relationship.relationship_type] = []
        self.by_type[relationship.relationship_type].append(rel_id)

    def remove_relationship(self, relationship_id: str) -> None:
        """Remove relationship from graph.

        Args:
            relationship_id: Relationship ID to remove
        """
        if relationship_id not in self.edges:
            return

        relationship = self.edges[relationship_id]

        # Remove from indexes
        self.by_source[relationship.source_id].remove(relationship_id)
        self.by_target[relationship.target_id].remove(relationship_id)
        self.by_type[relationship.relationship_type].remove(relationship_id)

        if relationship.bidirectional:
            self.by_source[relationship.target_id].remove(relationship_id)
            self.by_target[relationship.source_id].remove(relationship_id)

        # Remove from edges
        del self.edges[relationship_id]

    def get_outgoing_relationships(self, source_id: str, rel_type: str | None = None) -> list[Relationship]:
        """Get outgoing relationships from source.

        Args:
            source_id: Source entity ID
            rel_type: Optional filter by relationship type

        Returns:
            List of outgoing relationships
        """
        rel_ids = self.by_source.get(source_id, [])
        relationships: list[Relationship] = [self.edges[rid] for rid in rel_ids]

        if rel_type:
            relationships = [r for r in relationships if r.relationship_type == rel_type]

        return relationships

    def get_incoming_relationships(self, target_id: str, rel_type: str | None = None) -> list[Relationship]:
        """Get incoming relationships to target.

        Args:
            target_id: Target entity ID
            rel_type: Optional filter by relationship type

        Returns:
            List of incoming relationships
        """
        rel_ids = self.by_target.get(target_id, [])
        relationships: list[Relationship] = [self.edges[rid] for rid in rel_ids]

        if rel_type:
            relationships = [r for r in relationships if r.relationship_type == rel_type]

        return relationships

    def get_by_type(self, rel_type: str) -> list[Relationship]:
        """Get all relationships of a type.

        Args:
            rel_type: Relationship type

        Returns:
            List of relationships
        """
        rel_ids = self.by_type.get(rel_type, [])
        return [self.edges[rid] for rid in rel_ids]

    def traverse_forward(
        self,
        source_id: str,
        rel_type: str | None = None,
        max_depth: int = 3
    ) -> list[str]:
        """Traverse graph forward from source node.

        Args:
            source_id: Starting entity ID
            rel_type: Optional filter by relationship type
            max_depth: Maximum traversal depth

        Returns:
            List of reachable entity IDs
        """
        return self._traverse(
            start_id=source_id,
            rel_type=rel_type,
            max_depth=max_depth,
            backward=False
        )

    def traverse_backward(
        self,
        target_id: str,
        rel_type: str | None = None,
        max_depth: int = 3
    ) -> list[str]:
        """Traverse graph backward from target node.

        Args:
            target_id: Starting entity ID
            rel_type: Optional filter by relationship type
            max_depth: Maximum traversal depth
        Returns:
            List of reachable entity IDs
        """
        return self._traverse(
            start_id=target_id,
            rel_type=rel_type,
            max_depth=max_depth,
            backward=True
        )

    def _traverse(
        self,
        start_id: str,
        rel_type: str | None = None,
        max_depth: int = 3,
        backward: bool = False
    ) -> list[str]:
        """Traverse graph backward from target node.

        Args:
            start_id: Starting entity ID
            rel_type: Optional filter by relationship type
            max_depth: Maximum traversal depth
            backward: Whether to traverse backward

        Returns:
            List of reachable entity IDs
        """
        visited = set()
        queue = [(start_id, 0)]  # (entity_id, depth)
        reachable = []

        while queue:
            current_id, depth = queue.pop(0)

            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)
            if current_id != start_id:  # Don't include start
                reachable.append(current_id)

            if backward:
                # Get incoming relationships
                rels = self.get_incoming_relationships(current_id, rel_type)
            else:
                # Get outgoing relationships
                rels = self.get_outgoing_relationships(current_id, rel_type)

            # Add sources to queue
            for rel in rels:
                if backward:
                    if rel.source_id not in visited:
                        queue.append((rel.source_id, depth + 1))
                else:
                    if rel.target_id not in visited:
                        queue.append((rel.target_id, depth + 1))

        return reachable

    def merge_with(self, other: RelationshipGraph) -> None:
        """Merge another graph into this one.

        Args:
            other: Other graph to merge
        """

        # Merge nodes
        self.nodes.update(other.nodes)

        for relationship in other.edges.values():
            existing = [
                self.edges[e] for e in self.by_source[relationship.source_id]
                if self.edges[e].target_id == relationship.target_id and self.edges[e].relationship_type == relationship.relationship_type
            ]
            # if relationship.relationship_id not in self.edges:
            if not existing:
                self.add_relationship(relationship)
            elif relationship.confidence > any(e.confidence for e in existing):
                # Merge duplicate edges (take higher confidence)
                for e in existing:
                    self.remove_relationship(e.relationship_id)
                self.add_relationship(relationship)


class RelationshipGraphBuilder:
    """Builds relationship graphs from analysis results.

    Extracts relationships from various result types and constructs
    a unified relationship graph:
    - Dependency analysis → dependency edges
    - Data flow analysis → data flow edges
    - Traceability analysis → traceability links
    - Similarity analysis → similarity edges
    """

    def __init__(self) -> None:
        """Initialize relationship graph builder."""
        self.graph = RelationshipGraph()

    def get_graph(self) -> RelationshipGraph:
        """Get the current relationship graph.

        Returns:
            Relationship graph
        """
        return self.graph

    def add_relationships(
        self,
        relationships: list[Relationship]
    ) -> None:
        """Add relationships to graph.

        Args:
            relationships: Relationships to add
        """
        for rel in relationships:
            self.graph.add_relationship(rel)

    def update_graph(
        self,
        relationships: list[Relationship]
    ) -> None:
        """An alias for add_relationships.

        Args:
            relationships: Relationships to add
        """
        self.add_relationships(relationships)

    async def extract_relationships(
        self,
        result: ScopeAwareResult[Any]
    ) -> list[Relationship]:
        """Extract relationships from analysis result.

        Args:
            result: Analysis result

        Returns:
            List of extracted relationships
        """
        # TODO: The best option is to delegate to specialized extractors per result type

        relationships = []

        # Extract based on result type
        result_type = result.result_type or "unknown"

        if result_type == "dependency":
            relationships.extend(await self._extract_dependencies(result))
        elif result_type == "dataflow":
            relationships.extend(await self._extract_data_flows(result))
        elif result_type == "traceability":
            relationships.extend(await self._extract_traceability(result))
        elif result_type == "similarity":
            relationships.extend(await self._extract_similarity(result))

        # Extract from scope (related shards become relationships)
        relationships.extend(self._extract_from_scope(result))

        return relationships

    async def _extract_dependencies(
        self,
        result: ScopeAwareResult
    ) -> list[Relationship]:
        """Extract dependency relationships.

        Args:
            result: Dependency analysis result

        Returns:
            Dependency relationships
        """
        relationships = []

        # TODO: Extract from result content (format depends on result structure)
        # This is a placeholder - real implementation depends on result schema

        return relationships

    async def _extract_data_flows(
        self,
        result: ScopeAwareResult
    ) -> list[Relationship]:
        """Extract data flow relationships.

        Args:
            result: Data flow analysis result

        Returns:
            Data flow relationships
        """
        return []  # TODO: Implement data flow extraction

    async def _extract_traceability(
        self,
        result: ScopeAwareResult
    ) -> list[Relationship]:
        """Extract traceability relationships.

        Args:
            result: Traceability analysis result

        Returns:
            Traceability relationships
        """
        return []  # TODO: Implement traceability extraction

    async def _extract_similarity(
        self,
        result: ScopeAwareResult
    ) -> list[Relationship]:
        """Extract similarity relationships.

        Args:
            result: Similarity analysis result

        Returns:
            Similarity relationships
        """
        return []  # TODO: Implement similarity extraction

    def _extract_from_scope(
        self,
        result: ScopeAwareResult
    ) -> list[Relationship]:
        """Extract relationships from scope metadata.

        Args:
            result: Any result with scope

        Returns:
            Relationships from scope
        """
        relationships = []

        # Create relationships for related shards
        source_id = result.result_id  # TODO: Adjust based on actual source entity not result_id
        for related_shard in result.scope.related_shards:
            relationships.append(Relationship(
                relationship_id=f"rel_{source_id}_{related_shard}",
                source_id=source_id,
                target_id=related_shard,
                relationship_type="related",
                confidence=result.scope.confidence,
                evidence=[f"Scope metadata from {source_id}"]
            ))

        return relationships




