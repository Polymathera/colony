"""Wrapper for FileGrouper that exposes relationship graph for code analysis agents.

This wrapper provides access to the internal relationship graph built by FileGrouper,
allowing code analysis agents to understand file relationships without re-analyzing.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from .file_grouping import FileGrouper, FileGrouperConfig
from .analyzers.base import FileContentCache
from .tokenization import TokenManager
from polymathera.colony.utils import setup_logger

logger = setup_logger(__name__)


class FileGrouperWithGraph(FileGrouper):
    """FileGrouper wrapper that exposes the relationship graph.

    This wrapper allows code analysis agents to access the complete
    file relationship graph that was built during the grouping process.

    The graph contains:
    - Nodes: File paths
    - Edges: Relationships (imports, dependencies, semantic similarity, etc.)
    - Edge attributes:
        - weight: Relationship strength (0.0-1.0)
        - relationship_type: "import", "dependency", "semantic", "commit", "directory"
        - metadata: Type-specific metadata
        - is_cross_language: Whether this edge crosses language boundaries
        - source_language: Programming language of source file
        - target_language: Programming language of target file

    Example:
        ```python
        grouper = FileGrouperWithGraph(...)
        await grouper.initialize()

        # Group files (builds graph internally)
        groups = await grouper.group_files(repo_id, repo, files)

        # Access the relationship graph
        graph = grouper.get_relationship_graph()

        # Query graph for file relationships
        neighbors = list(graph.neighbors("path/to/file.py"))
        edge_data = graph.get_edge_data("file1.py", "file2.py")
        ```
    """

    def __init__(
        self,
        token_manager: TokenManager | None = None,
        config: FileGrouperConfig | None = None,
        file_content_cache: FileContentCache | None = None,
    ):
        super().__init__(
            token_manager=token_manager,
            config=config,
            file_content_cache=file_content_cache,
        )

        # Store the most recent graph built by group_files()
        self._last_graph: nx.DiGraph | None = None
        self._last_graph_metadata: dict[str, Any] = {}

    async def group_files(self, group_id, repo, files):
        """Override to ensure _last_graph is set even when loaded from cache."""
        result = await super().group_files(group_id, repo, files)

        # _build_graph sets _last_graph when the graph is built fresh.
        # When the base class loads a cached graph, _build_graph is NOT called
        # and _last_graph stays None. Load from the file graph cache in that case.
        if self._last_graph is None:
            commit_hash = repo.head.commit.hexsha
            cached = await self.file_graph_cache.get(
                key=f"{group_id}:{commit_hash}",
                version=self._get_graph_version(files),
            )
            if cached is not None:
                self._last_graph = cached
                logger.info(
                    f"Captured cached relationship graph: "
                    f"{cached.number_of_nodes()} nodes, {cached.number_of_edges()} edges"
                )

        return result

    def get_relationship_graph(self) -> nx.DiGraph | None:
        """Get the file relationship graph from the last group_files() call.

        Returns:
            NetworkX DiGraph with file relationships, or None if group_files()
            hasn't been called yet.

        Example:
            ```python
            graph = grouper.get_relationship_graph()
            if graph:
                # Get all files
                files = list(graph.nodes())

                # Get relationships for a specific file
                neighbors = list(graph.neighbors("path/to/file.py"))

                # Get edge metadata
                for neighbor in neighbors:
                    edge_data = graph.get_edge_data("path/to/file.py", neighbor)
                    print(f"Relationship: {edge_data['relationship_type']}")
                    print(f"Weight: {edge_data['weight']}")
                    print(f"Cross-language: {edge_data.get('is_cross_language', False)}")
            ```
        """
        return self._last_graph

    def get_graph_metadata(self) -> dict[str, Any]:
        """Get metadata about the last relationship graph.

        Returns:
            Dictionary with graph metadata:
                - cross_lang_bindings: List of files with cross-language relationships
                - circular_dependencies: List of detected circular dependency cycles
                - node_count: Number of files in graph
                - edge_count: Number of relationships in graph
                - graph_version: Version hash of the graph

        Example:
            ```python
            metadata = grouper.get_graph_metadata()
            print(f"Files: {metadata['node_count']}")
            print(f"Relationships: {metadata['edge_count']}")
            print(f"Cross-language files: {len(metadata['cross_lang_bindings'])}")
            ```
        """
        if self._last_graph is None:
            return {}

        return {
            **self._last_graph_metadata,
            "node_count": self._last_graph.number_of_nodes(),
            "edge_count": self._last_graph.number_of_edges(),
        }

    async def _build_graph(
        self,
        repo,
        files: list[str],
        cross_lang_bindings: set[str],
    ) -> nx.DiGraph:
        """Override to capture the built graph."""
        # Build graph using parent implementation
        graph = await super()._build_graph(repo, files, cross_lang_bindings)

        # Store graph and metadata
        self._last_graph = graph
        self._last_graph_metadata = {
            "cross_lang_bindings": list(cross_lang_bindings),
            "circular_dependencies": graph.graph.get("circular_dependencies", []),
            "graph_version": self._get_graph_version(files),
        }

        logger.info(
            f"Captured relationship graph: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )

        return graph

    def query_file_relationships(
        self,
        file_path: str,
        relationship_types: list[str] | None = None,
        min_weight: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Query relationships for a specific file.

        Args:
            file_path: Path to file to query
            relationship_types: Filter by relationship types (None = all types)
            min_weight: Minimum relationship weight (0.0-1.0)

        Returns:
            List of relationships, each with:
                - target: Target file path
                - relationship_type: Type of relationship
                - weight: Relationship strength
                - metadata: Type-specific metadata

        Example:
            ```python
            # Get all strong import relationships
            imports = grouper.query_file_relationships(
                "src/main.py",
                relationship_types=["import"],
                min_weight=0.5
            )
            for rel in imports:
                print(f"{rel['target']}: {rel['weight']:.2f}")
            ```
        """
        if self._last_graph is None:
            logger.warning("No graph available - call group_files() first")
            return []

        if file_path not in self._last_graph:
            logger.warning(f"File {file_path} not in graph")
            return []

        results = []
        for target in self._last_graph.neighbors(file_path):
            edge_data = self._last_graph.get_edge_data(file_path, target)
            edge_types = edge_data.get("relationship_types", [])

            # Filter by relationship type
            if relationship_types and not any(t in relationship_types for t in edge_types):
                continue

            # Filter by weight
            if edge_data["weight"] < min_weight:
                continue

            results.append({
                "target": target,
                "relationship_types": edge_types,
                "weight": edge_data["weight"],
                "metadata": edge_data.get("metadata", {}),
            })

        return results