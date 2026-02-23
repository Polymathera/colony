"""Page graph capability for graph-based page traversal.

Provides @action_executor methods for traversing and updating the page
relationship graph. Wraps PageStorage graph operations.

Storage: Uses existing PageStorage.store_page_graph() / retrieve_page_graph().

Usage:
    # Add capability to agent
    page_graph_cap = PageGraphCapability(agent=self)
    self.add_capability(page_graph_cap)

    # ActionPolicy can now use these actions:
    # - get_neighbors(page_ids, direction, max_per_page)
    # - traverse(start_pages, strategy, max_depth)
    # - update_edge(source, target, weight_delta, relationship_type)
    # - compute_centrality(page_ids, metric)
    # - get_clusters(algorithm, min_size, max_size)
    # - find_path(source, target, max_hops)
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, TYPE_CHECKING
from overrides import override
import networkx as nx

from ...base import AgentCapability
from ...models import AgentSuspensionState
from ..actions.policies import action_executor

if TYPE_CHECKING:
    from ...base import Agent

logger = logging.getLogger(__name__)


class PageGraphCapability(AgentCapability):
    """Traverses and updates the page relationship graph.

    Wraps PageStorage graph operations as @action_executors.
    Graph is stored via PageStorage.store_page_graph() (EFS/S3).

    Primitives for:
    - Traversing graph (BFS, DFS, neighbors)
    - Updating edges (learning from queries)
    - Computing centrality metrics
    - Getting clusters

    Does NOT assume:
    - What relationships mean (dependencies, similarity, etc.)
    - How traversal is used (batching, prefetching, etc.)

    The ActionPolicy decides how to use these primitives.
    """

    def __init__(self, agent: Agent, scope_id: str | None = None):
        """Initialize page graph capability.

        Args:
            agent: Owning agent
            scope_id: Blackboard scope (defaults to agent_id)
        """
        super().__init__(agent=agent, scope_id=scope_id)
        self._page_graph: nx.DiGraph | None = None

    def get_action_group_description(self) -> str:
        return (
            "Page Graph — graph-based traversal and relationship management over VCM pages. "
            "Provides cache-aware traversal (BFS/DFS respecting working set), clustering for "
            "batch scheduling, and centrality metrics for page prioritization. "
            "Agnostic to relationship semantics — you decide how to use edges. "
            "Graph is loaded lazily from PageStorage and cached in memory. "
            "Use get_clusters for batch planning and compute_centrality for page importance ranking."
        )

    async def _get_page_graph(self) -> nx.DiGraph:
        """Load page graph via PageStorage."""
        # TODO: This needs to be loaded dynamically since the page graph can change.
        if self._page_graph is None:
            self._page_graph = await self.agent.load_page_graph()
        return self._page_graph

    async def _persist_graph(self) -> None:
        """Persist graph changes via PageStorage."""
        if self._page_graph is not None:
            # TODO - FIXME: This can introduce race conditions if multiple updates happen concurrently.
            # We may need a more robust graph storage/update mechanism for large graphs.
            page_storage = await self.agent.get_page_storage()
            await page_storage.store_page_graph(
                tenant_id=self.agent.tenant_id,
                group_id=self.agent.group_id,
                graph_data=self._page_graph
            )

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for PageGraphCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for PageGraphCapability")
        pass

    # === Action Executors ===

    @action_executor()
    async def get_neighbors(
        self,
        page_ids: list[str],
        direction: str = "both",
        max_per_page: int | None = None,
    ) -> dict[str, Any]:
        """Get graph neighbors of pages.

        Use for cache-aware traversal - load neighbors together for
        spatial locality in KV cache.

        Args:
            page_ids: Source pages to get neighbors for
            direction: Edge direction ("in", "out", "both")
            max_per_page: Limit neighbors per source page

        Returns:
            Dict with:
            - neighbors: Dict mapping page_id -> list of neighbor_ids
            - total_neighbors: Total neighbor count across all pages
        """
        graph = await self._get_page_graph()

        result = {}
        for page_id in page_ids:
            neighbors = []
            if page_id in graph:
                if direction in ("out", "both"):
                    neighbors.extend(list(graph.successors(page_id)))
                if direction in ("in", "both"):
                    neighbors.extend(list(graph.predecessors(page_id)))
                neighbors = list(set(neighbors))  # Deduplicate
                if max_per_page:
                    neighbors = neighbors[:max_per_page]
            result[page_id] = neighbors

        total = sum(len(n) for n in result.values())

        return {
            "neighbors": result,
            "total_neighbors": total,
        }

    @action_executor()
    async def traverse(
        self,
        start_pages: list[str],
        strategy: str = "bfs",
        max_depth: int = 2,
        max_nodes: int | None = None,
        prefer_cached: bool = False,
    ) -> dict[str, Any]:
        """Traverse graph from starting pages.

        Use for discovering related pages for prefetching or batching.

        LLM-controllable cache-awareness:
        - prefer_cached: When True, prioritize pages already in working set

        Args:
            start_pages: Pages to start traversal from
            strategy: Traversal strategy ("bfs" or "dfs")
            max_depth: Maximum traversal depth
            max_nodes: Maximum total nodes to return
            prefer_cached: If True, prioritize pages in working set (LLM decides when useful)

        Returns:
            Dict with:
            - visited: List of visited page IDs in traversal order
            - by_depth: Dict mapping depth -> list of page_ids at that depth
            - cache_hits: Number of visited pages that were in working set
            - cache_hit_ratio: Ratio of cache hits to total visited
        """
        graph = await self._get_page_graph()

        # Get working set if cache-aware traversal is enabled
        working_set_pages: set[str] = set()
        if prefer_cached:
            from .working_set import WorkingSetCapability
            ws_cap = self.agent.get_capability_by_type(WorkingSetCapability)
            if ws_cap:
                try:
                    ws_result = await ws_cap.get_working_set()
                    working_set_pages = set(ws_result.get("pages", []))
                except Exception as e:
                    logger.debug(f"Could not get working set for cache-aware traversal: {e}")

        visited = []
        visited_set = set()
        by_depth: dict[int, list[str]] = {}

        if prefer_cached and working_set_pages:
            # Priority-based traversal: cached pages get priority 0, non-cached get priority 1
            import heapq
            # Heap entries: (priority, depth, page_id)
            frontier = []
            for page_id in start_pages:
                if page_id in graph:
                    priority = 0 if page_id in working_set_pages else 1
                    heapq.heappush(frontier, (priority, 0, page_id))

            while frontier:
                priority, depth, page_id = heapq.heappop(frontier)

                if page_id in visited_set:
                    continue

                if depth > max_depth:
                    continue

                if max_nodes and len(visited) >= max_nodes:
                    break

                visited.append(page_id)
                visited_set.add(page_id)

                if depth not in by_depth:
                    by_depth[depth] = []
                by_depth[depth].append(page_id)

                # Add neighbors with cache-aware priority
                if depth < max_depth and page_id in graph:
                    for neighbor in list(graph.successors(page_id)) + list(graph.predecessors(page_id)):
                        if neighbor not in visited_set:
                            neighbor_priority = 0 if neighbor in working_set_pages else 1
                            heapq.heappush(frontier, (neighbor_priority, depth + 1, neighbor))
        else:
            # Standard BFS/DFS traversal
            frontier = [(page_id, 0) for page_id in start_pages if page_id in graph]

            while frontier:
                if strategy == "bfs":
                    page_id, depth = frontier.pop(0)
                else:  # dfs
                    page_id, depth = frontier.pop()

                if page_id in visited_set:
                    continue

                if depth > max_depth:
                    continue

                if max_nodes and len(visited) >= max_nodes:
                    break

                visited.append(page_id)
                visited_set.add(page_id)

                if depth not in by_depth:
                    by_depth[depth] = []
                by_depth[depth].append(page_id)

                # Add neighbors to frontier
                if depth < max_depth and page_id in graph:
                    for neighbor in list(graph.successors(page_id)) + list(graph.predecessors(page_id)):
                        if neighbor not in visited_set:
                            frontier.append((neighbor, depth + 1))

        # Compute cache statistics
        cache_hits = sum(1 for p in visited if p in working_set_pages) if working_set_pages else 0
        cache_hit_ratio = cache_hits / len(visited) if visited else 0.0

        return {
            "visited": visited,
            "by_depth": by_depth,
            "cache_hits": cache_hits,
            "cache_hit_ratio": cache_hit_ratio,
            "options_used": {
                "prefer_cached": prefer_cached,
            },
        }

    @action_executor()
    async def update_edge(
        self,
        source: str,
        target: str,
        weight_delta: float,
        relationship_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update edge weight in page graph.

        Use for learning page relationships from query patterns.

        Args:
            source: Source page ID
            target: Target page ID
            weight_delta: Weight change (positive = strengthen relationship)
            relationship_type: Optional relationship type label
            metadata: Additional edge metadata

        Returns:
            Dict with:
            - updated: Whether edge was updated
            - new_weight: New edge weight
            - edge_created: Whether this was a new edge
        """
        graph = await self._get_page_graph()
        metadata = metadata or {}

        edge_created = False
        if graph.has_edge(source, target):
            # Update existing edge
            edge_data = graph[source][target]
            new_weight = edge_data.get("weight", 0.5) + weight_delta
            new_weight = max(0.0, min(1.0, new_weight))  # Clamp to [0, 1]
            edge_data["weight"] = new_weight
            if relationship_type:
                rel_types = set(edge_data.get("relationship_types", []))
                rel_types.add(relationship_type)
                edge_data["relationship_types"] = list(rel_types)
            edge_data.update(metadata)
        else:
            # Create new edge
            new_weight = max(0.0, min(1.0, 0.5 + weight_delta))
            graph.add_edge(
                source,
                target,
                weight=new_weight,
                relationship_types=[relationship_type] if relationship_type else [],
                **metadata
            )
            edge_created = True

        # Persist changes
        # TODO: This can introduce race conditions if multiple updates happen concurrently.
        page_storage = await self.agent.get_page_storage()
        await page_storage.update_page_graph(
            tenant_id=self.agent.tenant_id,
            group_id=self.agent.group_id,
            page_relationships={
                (source, target): {
                    "weight": new_weight,
                    "relationship_type": relationship_type,
                    **metadata
                }
            }
        )

        return {
            "updated": True,
            "new_weight": new_weight,
            "edge_created": edge_created,
        }

    @action_executor()
    async def compute_centrality(
        self,
        page_ids: list[str] | None = None,
        metric: str = "degree",
    ) -> dict[str, Any]:
        """Compute centrality metrics for pages.

        Use for identifying important/central pages for initial working set
        or prioritization.

        Args:
            page_ids: Pages to compute centrality for (None = all pages)
            metric: Centrality metric ("degree", "pagerank", "betweenness")

        Returns:
            Dict with:
            - centrality: Dict mapping page_id -> centrality score
            - sorted_pages: Page IDs sorted by centrality (highest first)
        """
        graph = await self._get_page_graph()

        # Compute centrality
        if metric == "degree":
            centrality_values = dict(graph.degree())
        elif metric == "pagerank":
            if len(graph.nodes()) > 0:
                centrality_values = nx.pagerank(graph)
            else:
                centrality_values = {}
        elif metric == "betweenness":
            if len(graph.nodes()) > 0:
                centrality_values = nx.betweenness_centrality(graph)
            else:
                centrality_values = {}
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'degree', 'pagerank', or 'betweenness'.")

        # Filter to requested pages if specified
        if page_ids is not None:
            centrality = {p: centrality_values.get(p, 0) for p in page_ids}
        else:
            centrality = centrality_values

        # Sort by centrality
        sorted_pages = sorted(centrality.keys(), key=lambda p: centrality[p], reverse=True)

        return {
            "centrality": centrality,
            "sorted_pages": sorted_pages,
        }

    @action_executor()
    async def get_clusters(
        self,
        algorithm: str = "connected",
        min_size: int = 2,
        max_size: int | None = None,
    ) -> dict[str, Any]:
        """Get page clusters from graph.

        Use for batch-based processing of related pages.

        Wraps PageStorage.get_all_clusters().

        Args:
            algorithm: Clustering algorithm ("connected" for connected components)
            min_size: Minimum cluster size
            max_size: Maximum cluster size

        Returns:
            Dict with:
            - clusters: List of cluster dicts with cluster_id, page_ids, size
            - total_clusters: Number of clusters
        """
        clusters = []
        cluster_idx = 0

        page_storage = await self.agent.get_page_storage()

        async for cluster in page_storage.get_all_clusters(
            tenant_id=self.agent.tenant_id,
            group_id=self.agent.group_id,
            max_cluster_size=max_size or 100,
            min_cluster_size=min_size,
        ):
            clusters.append({
                "cluster_id": cluster.cluster_id,
                "page_ids": cluster.page_ids,
                "size": len(cluster.page_ids),
                "relationship_score": cluster.relationship_score,
                "cluster_type": cluster.cluster_type,
            })
            cluster_idx += 1

        return {
            "clusters": clusters,
            "total_clusters": len(clusters),
        }

    @action_executor()
    async def find_path(
        self,
        source: str,
        target: str,
        max_hops: int = 5,
    ) -> dict[str, Any]:
        """Find path between two pages.

        Use for understanding relationships and potential query routing.

        Args:
            source: Source page ID
            target: Target page ID
            max_hops: Maximum path length to search

        Returns:
            Dict with:
            - path: List of page IDs from source to target (empty if no path)
            - found: Whether a path was found
            - length: Path length (number of edges)
        """
        graph = await self._get_page_graph()

        if source not in graph or target not in graph:
            return {
                "path": [],
                "found": False,
                "length": 0,
                "reason": "source_or_target_not_in_graph",
            }

        try:
            # Use shortest_path with cutoff
            path = nx.shortest_path(
                graph.to_undirected(),  # Search in both directions
                source=source,
                target=target,
            )

            if len(path) - 1 > max_hops:
                return {
                    "path": [],
                    "found": False,
                    "length": 0,
                    "reason": f"path_too_long_{len(path) - 1}_hops",
                }

            return {
                "path": path,
                "found": True,
                "length": len(path) - 1,
            }
        except nx.NetworkXNoPath:
            return {
                "path": [],
                "found": False,
                "length": 0,
                "reason": "no_path_exists",
            }

    @action_executor()
    async def get_page_cluster(
        self,
        page_id: str,
        cluster_size: int = 10,
    ) -> dict[str, Any]:
        """Get cluster of pages related to a specific page.

        Use for finding related pages to load together for cache locality.

        Args:
            page_id: Page to find cluster for
            cluster_size: Desired cluster size

        Returns:
            Dict with:
            - cluster_pages: List of page IDs in cluster (including input page)
            - size: Cluster size
        """
        # TODO: Unify this with PageStorage.get_page_cluster()
        graph = await self._get_page_graph()

        if page_id not in graph:
            return {
                "cluster_pages": [page_id],
                "size": 1,
            }

        # BFS from page_id to get cluster
        cluster = [page_id]
        visited = {page_id}
        frontier = list(graph.successors(page_id)) + list(graph.predecessors(page_id))

        while frontier and len(cluster) < cluster_size:
            neighbor = frontier.pop(0)
            if neighbor in visited:
                continue
            visited.add(neighbor)
            cluster.append(neighbor)

            # Add neighbor's neighbors
            if len(cluster) < cluster_size:
                for n in list(graph.successors(neighbor)) + list(graph.predecessors(neighbor)):
                    if n not in visited:
                        frontier.append(n)

        return {
            "cluster_pages": cluster,
            "size": len(cluster),
        }

    # === Relationship Management (merged from relationships/ module) ===

    @action_executor()
    async def load_graph(self) -> dict[str, Any]:
        """Load and return the page graph.

        Returns:
            Dict with:
            - graph: The nx.DiGraph object
            - node_count: Number of nodes
            - edge_count: Number of edges
        """
        graph = await self._get_page_graph()
        return {
            "graph": graph,
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
        }

    @action_executor()
    async def apply_relationships(
        self,
        relationships: list[dict],
    ) -> dict[str, Any]:
        """Apply typed relationships as graph edges.

        Each relationship dict should match Relationship.model_dump() format
        (source_id, target_id, relationship_type, confidence, weight, etc.).

        Args:
            relationships: Serialized Relationship dicts

        Returns:
            Dict with:
            - applied: Number of relationships applied
            - new_edges: Number of new edges created
            - updated_edges: Number of existing edges updated
        """
        graph = await self._get_page_graph()
        new_edges = 0
        updated_edges = 0

        for rel_dict in relationships:
            source = rel_dict.get("source_id", "")
            target = rel_dict.get("target_id", "")
            rel_type = rel_dict.get("relationship_type", "unknown")
            confidence = rel_dict.get("confidence", 1.0)
            weight = rel_dict.get("weight", 1.0)
            evidence = rel_dict.get("evidence", [])
            metadata = rel_dict.get("metadata", {})

            if not source or not target:
                continue

            if graph.has_edge(source, target):
                # Update existing edge: merge relationship types, take max weight/confidence
                edge_data = graph[source][target]
                rel_types = set(edge_data.get("relationship_types", []))
                rel_types.add(rel_type)
                edge_data["relationship_types"] = list(rel_types)
                edge_data["weight"] = max(edge_data.get("weight", 0), weight)
                edge_data["confidence"] = max(edge_data.get("confidence", 0), confidence)
                edge_data.update(metadata)
                updated_edges += 1
            else:
                # Add new edge
                graph.add_edge(
                    source,
                    target,
                    weight=weight,
                    confidence=confidence,
                    relationship_types=[rel_type],
                    evidence=evidence,
                    **metadata,
                )
                new_edges += 1

        # Persist changes
        await self._persist_graph()

        return {
            "applied": len(relationships),
            "new_edges": new_edges,
            "updated_edges": updated_edges,
        }

    @action_executor()
    async def get_typed_relationships(
        self,
        source: str | None = None,
        target: str | None = None,
        relationship_type: str | None = None,
    ) -> dict[str, Any]:
        """Query edges matching filters, returning structured relationship data.

        Args:
            source: Optional source page filter
            target: Optional target page filter
            relationship_type: Optional relationship type filter

        Returns:
            Dict with:
            - relationships: List of matching relationship dicts
            - count: Number of matches
        """
        graph = await self._get_page_graph()
        results = []

        for u, v, data in graph.edges(data=True):
            if source and u != source:
                continue
            if target and v != target:
                continue
            if relationship_type:
                edge_types = data.get("relationship_types", [])
                if relationship_type not in edge_types:
                    continue

            results.append({
                "source_id": u,
                "target_id": v,
                "relationship_types": data.get("relationship_types", []),
                "weight": data.get("weight", 1.0),
                "confidence": data.get("confidence", 1.0),
                "metadata": {
                    k: val
                    for k, val in data.items()
                    if k not in ("weight", "confidence", "relationship_types", "evidence")
                },
            })

        return {
            "relationships": results,
            "count": len(results),
        }

    @action_executor()
    async def publish_relationships(
        self,
        relationships: list[dict],
    ) -> dict[str, Any]:
        """Write relationships to blackboard for cross-agent sharing.

        Args:
            relationships: Serialized Relationship dicts

        Returns:
            Dict with:
            - published: Number of relationships published
        """
        blackboard = await self.get_blackboard()
        published = 0

        for rel_dict in relationships:
            source = rel_dict.get("source_id", "")
            target = rel_dict.get("target_id", "")
            rel_type = rel_dict.get("relationship_type", "unknown")
            discovered_by = rel_dict.get("discovered_by")

            key = f"relationship:{source}:{target}:{rel_type}"
            await blackboard.write(
                key=key,
                value=rel_dict,
                tags={"relationship", rel_type, source, target},
                created_by=discovered_by or self.agent.agent_id,
            )
            published += 1

        return {"published": published}

    @action_executor()
    async def discover_cross_boundary(
        self,
        starting_page: str,
        relationship_type: str | None = None,
        max_hops: int = 3,
    ) -> dict[str, Any]:
        """Traverse existing typed edges via BFS from starting_page.

        Finds pages reachable by following existing edges of the given type
        in the persistent page graph. Returns visited pages and the edges
        traversed. Does NOT discover new relationships.

        To discover NEW relationships (analyze pages and add edges), use
        ``discover_new_relationships()`` which accepts a page analyzer callback.

        Args:
            starting_page: Page ID to start from
            relationship_type: Optional edge type filter
            max_hops: Maximum traversal depth

        Returns:
            Dict with:
            - visited: List of visited page IDs
            - discovered_relationships: List of existing edge dicts traversed
            - depth_reached: Maximum depth actually reached
        """
        graph = await self._get_page_graph()

        visited: list[str] = []
        visited_set: set[str] = set()
        discovered: list[dict] = []
        depth_reached = 0

        frontier = [(starting_page, 0)]

        while frontier:
            page_id, depth = frontier.pop(0)

            if page_id in visited_set or depth > max_hops:
                continue

            visited_set.add(page_id)
            visited.append(page_id)
            depth_reached = max(depth_reached, depth)

            if page_id not in graph:
                continue

            # Follow outgoing edges
            for neighbor in graph.successors(page_id):
                if neighbor in visited_set:
                    continue

                edge_data = graph[page_id][neighbor]
                edge_types = edge_data.get("relationship_types", [])

                if relationship_type and relationship_type not in edge_types:
                    continue

                discovered.append({
                    "source_id": page_id,
                    "target_id": neighbor,
                    "relationship_types": edge_types,
                    "weight": edge_data.get("weight", 1.0),
                })
                frontier.append((neighbor, depth + 1))

            # Follow incoming edges (bidirectional discovery)
            for predecessor in graph.predecessors(page_id):
                if predecessor in visited_set:
                    continue

                edge_data = graph[predecessor][page_id]
                edge_types = edge_data.get("relationship_types", [])

                if relationship_type and relationship_type not in edge_types:
                    continue

                discovered.append({
                    "source_id": predecessor,
                    "target_id": page_id,
                    "relationship_types": edge_types,
                    "weight": edge_data.get("weight", 1.0),
                })
                frontier.append((predecessor, depth + 1))

        return {
            "visited": visited,
            "discovered_relationships": discovered,
            "depth_reached": depth_reached,
        }

    async def discover_new_relationships(
        self,
        starting_page: str,
        page_analyzer: Callable[[str], Awaitable[list[dict]]],
        relationship_type: str | None = None,
        max_hops: int = 3,
    ) -> dict[str, Any]:
        """Discover NEW relationships by analyzing pages via BFS.

        Unlike ``discover_cross_boundary`` which only traverses existing edges,
        this method calls ``page_analyzer`` at each visited page to discover
        new relationships, adds them to the persistent graph, and follows
        newly created edges to expand the frontier.

        This is NOT an @action_executor because it requires a callable argument.
        Capabilities and policies that need programmatic discovery should call
        this method directly.

        Args:
            starting_page: Page ID to start from
            page_analyzer: Async callable that takes a page_id and returns
                a list of relationship dicts with keys: source_id, target_id,
                relationship_type, confidence, weight, metadata.
            relationship_type: Optional filter for which types to follow
            max_hops: Maximum traversal depth

        Returns:
            Dict with:
            - visited: List of visited page IDs
            - new_relationships: List of newly discovered relationship dicts
            - depth_reached: Maximum depth actually reached
            - edges_added: Number of new edges added to the persistent graph
        """
        graph = await self._get_page_graph()

        visited: list[str] = []
        visited_set: set[str] = set()
        new_relationships: list[dict] = []
        depth_reached = 0
        edges_added = 0

        frontier = [(starting_page, 0)]

        while frontier:
            page_id, depth = frontier.pop(0)

            if page_id in visited_set or depth > max_hops:
                continue

            visited_set.add(page_id)
            visited.append(page_id)
            depth_reached = max(depth_reached, depth)

            # Analyze this page to discover new relationships
            try:
                discovered = await page_analyzer(page_id)
            except Exception as e:
                logger.warning(
                    f"Page analyzer failed for {page_id}: {e}"
                )
                discovered = []

            # Add discovered relationships to the persistent graph
            for rel_dict in discovered:
                source = rel_dict.get("source_id", "")
                target = rel_dict.get("target_id", "")
                rel_type = rel_dict.get("relationship_type", "unknown")

                if not source or not target:
                    continue

                if relationship_type and rel_type != relationship_type:
                    continue

                new_relationships.append(rel_dict)

                # Add to persistent graph if edge doesn't exist
                if not graph.has_edge(source, target):
                    graph.add_edge(
                        source,
                        target,
                        weight=rel_dict.get("weight", 1.0),
                        confidence=rel_dict.get("confidence", 1.0),
                        relationship_types=[rel_type],
                        **rel_dict.get("metadata", {}),
                    )
                    edges_added += 1

                # Expand frontier to newly connected pages
                if target not in visited_set:
                    frontier.append((target, depth + 1))
                if source not in visited_set:
                    frontier.append((source, depth + 1))

            # Also follow existing edges of the requested type
            if page_id in graph:
                for neighbor in graph.successors(page_id):
                    if neighbor in visited_set:
                        continue
                    edge_data = graph[page_id][neighbor]
                    edge_types = edge_data.get("relationship_types", [])
                    if relationship_type and relationship_type not in edge_types:
                        continue
                    frontier.append((neighbor, depth + 1))

                for predecessor in graph.predecessors(page_id):
                    if predecessor in visited_set:
                        continue
                    edge_data = graph[predecessor][page_id]
                    edge_types = edge_data.get("relationship_types", [])
                    if relationship_type and relationship_type not in edge_types:
                        continue
                    frontier.append((predecessor, depth + 1))

        # Persist any new edges
        if edges_added > 0:
            await self._persist_graph()
            logger.info(
                f"Discovery from {starting_page}: visited {len(visited)} pages, "
                f"added {edges_added} new edges"
            )

        return {
            "visited": visited,
            "new_relationships": new_relationships,
            "depth_reached": depth_reached,
            "edges_added": edges_added,
        }

    @action_executor()
    async def record_query_resolution(
        self,
        source_page_id: str,
        target_page_id: str,
        query: str,
        success: bool,
        relevance_score: float = 1.0,
    ) -> dict[str, Any]:
        """Record query outcome to strengthen/weaken graph edges.

        Uses exponential moving average to update edge weights based
        on query success. Creates new edges for successful queries
        between previously unconnected pages.

        Args:
            source_page_id: Page that generated the query
            target_page_id: Page that was queried
            query: Query text
            success: Whether query found relevant content
            relevance_score: How relevant the target was (0.0-1.0)

        Returns:
            Dict with:
            - updated: Whether the graph was updated
            - edge_created: Whether a new edge was created
            - new_weight: Updated edge weight
        """
        if not success:
            return {"updated": False, "edge_created": False, "new_weight": 0.0}

        graph = await self._get_page_graph()
        edge_created = False

        if graph.has_edge(source_page_id, target_page_id):
            # Update existing edge (exponential moving average)
            edge_data = graph[source_page_id][target_page_id]
            old_weight = edge_data.get("weight", 0.5)
            new_weight = old_weight * 0.9 + relevance_score * 0.1
            edge_data["weight"] = new_weight
            edge_data["query_count"] = edge_data.get("query_count", 0) + 1
        else:
            # Create new discovered edge
            new_weight = relevance_score
            graph.add_edge(
                source_page_id,
                target_page_id,
                weight=new_weight,
                relationship_types=["discovered_dependency"],
                query_count=1,
            )
            edge_created = True

        # Persist changes
        # TODO: This can introduce race conditions if multiple updates happen concurrently.
        page_storage = await self.agent.get_page_storage()
        await page_storage.update_page_graph(
            tenant_id=self.agent.tenant_id,
            group_id=self.agent.group_id,
            page_relationships={
                (source_page_id, target_page_id): {
                    "weight": new_weight,
                    "relationship_type": "discovered_dependency",
                }
            },
        )

        return {
            "updated": True,
            "edge_created": edge_created,
            "new_weight": new_weight,
        }

