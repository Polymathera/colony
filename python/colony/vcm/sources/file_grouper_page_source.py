

import logging
import pickle
from overrides import override
from typing import Any, AsyncIterator, Literal

import networkx as nx

from .context_page_source import ContextPageSource, PageCluster
from ..page_storage import PageStorage
from ...distributed import get_polymathera
from ...llms.sharding.file_grouping_wrapper import FileGrouperWithGraph
from ...llms.sharding.strategy_wrapper import GitRepoShardingWithMapping


logger = logging.getLogger(__name__)


class FileGrouperContextPageSource(ContextPageSource):
    """ContextPageSource backed by EFS/S3 storage using `FileGrouper`.

    This is NOT a Ray Deployment - just a regular class that uses
    distributed storage to persist page graphs.

    Maps `FileGrouper` concepts:
    - `FileGroup` → `PageCluster`
    - File relationship graph → Page graph
    - File paths → Page IDs
    """

    def __init__(
        self,
        group_id: str,  # Repo ID
        repo_path: str,  # Path to cloned repo
        tenant_id: str = "default",
        storage_backend_type: Literal["efs", "s3"] = "efs",
        storage_path: str = "colony/context_page_sources",
        file_grouper: FileGrouperWithGraph | None = None,
        sharding_strategy: GitRepoShardingWithMapping | None = None,
    ):
        """Initialize file-grouper-based context page source.

        Args:
            group_id: Repository or group identifier
            repo_path: Local path to git repository
            tenant_id: Tenant identifier
            storage_backend_type: Storage backend ("efs" or "s3")
            storage_path: Storage path prefix
            file_grouper: Optional FileGrouperWithGraph instance (created if None)
            sharding_strategy: Optional GitRepoShardingWithMapping instance (created if None)
        """
        self.group_id = group_id
        self.repo_path = repo_path
        self.tenant_id = tenant_id
        self.storage_backend_type = storage_backend_type
        self.storage_path = storage_path

        # File grouping and sharding wrappers (expose graph and file-to-page mapping)
        self.file_grouper = file_grouper
        self.sharding_strategy = sharding_strategy

        # Initialized in initialize()
        self.page_storage: PageStorage | None = None
        self.file_to_page: dict[str, str] = {}
        self.page_to_file: dict[str, str] = {}
        self.page_keys: dict[str, str] = {}  # page_id → summary (key)
        self.page_graph: nx.DiGraph | None = None
        self._file_relationship_graph: nx.DiGraph | None = None  # From FileGrouperWithGraph

    @override
    async def get_page_storage(self) -> PageStorage:
        """Get the page storage instance."""
        await self.initialize()
        return self.page_storage

    @override
    async def initialize(self) -> None:
        """Initialize storage and load/build page graph."""
        if self.page_storage is not None and self.page_graph is not None:
            return  # Already initialized

        # Get storage backend
        polymathera = get_polymathera()
        storage_backend = await polymathera.get_storage()

        # Initialize page storage
        self.page_storage = PageStorage(
            storage_backend=storage_backend,
            backend_type=self.storage_backend_type,
            storage_path=self.storage_path,
        )
        await self.page_storage.initialize()

        # Try to load existing graph
        graph_data = await self.page_storage.retrieve_page_graph(
            group_id=self.group_id,
            tenant_id=self.tenant_id
        )

        if graph_data:
            # Load existing graph
            graph_dict = pickle.loads(graph_data)
            self.page_graph = graph_dict["graph"]
            self.file_to_page = graph_dict["file_to_page"]
            self.page_to_file = graph_dict["page_to_file"]
            self.page_keys = graph_dict["page_keys"]
            logger.info(
                f"Loaded page graph for {self.group_id}: "
                f"{len(self.page_graph.nodes)} nodes, {len(self.page_graph.edges)} edges"
            )
        else:
            # Build new graph (requires FileGrouper - deferred to first usage)
            logger.info(f"No existing page graph for {self.group_id}, will build on first use")
            self.page_graph = nx.DiGraph()

    @override
    async def load_page_graph(self) -> nx.DiGraph:
        """Load page graph dynamically from PageStorage.

        This allows the agent and its components to load
        the page graph when needed, rather than
        passing the entire graph in metadata.
        """
        try:
            await self.initialize()

            if not self.page_storage:
                logger.debug("No page storage, using empty page graph")
                self.page_graph = nx.DiGraph()
                return self.page_graph

            if not self.group_id or not self.tenant_id:
                logger.warning("Missing group_id or tenant_id, creating empty page graph")
                self.page_graph = nx.DiGraph()
                return self.page_graph

            graph_data = await self.page_storage.retrieve_page_graph(
                group_id=self.group_id,
                tenant_id=self.tenant_id
            )

            if graph_data:
                graph_dict = pickle.loads(graph_data)
                self.page_graph = graph_dict["graph"]
                self.file_to_page = graph_dict["file_to_page"]
                self.page_to_file = graph_dict["page_to_file"]
                self.page_keys = graph_dict["page_keys"]
                logger.info(
                    f"Loaded page graph for {self.group_id}: "
                    f"{len(self.page_graph.nodes)} nodes, {len(self.page_graph.edges)} edges, "
                    f"{self.page_graph.number_of_edges()} relationships"
                )
            else:
                # Build new graph (requires FileGrouper - deferred to first usage)
                logger.info(f"No existing page graph for {self.group_id}, "
                            "will build on first use, creating empty graph")
                logger.warning("No existing page graph found in storage, creating empty graph")
                self.page_graph = nx.DiGraph()

            return self.page_graph

        except Exception as e:
            logger.debug(f"Failed to load page graph: {e}")
            self.page_graph = nx.DiGraph()
            return self.page_graph

    @override
    async def get_page_cluster(
        self,
        cluster_size: int = 10,
        cluster_type: str | None = None
    ) -> PageCluster:
        """Get a cluster of related pages."""
        # Simple implementation: use community detection or just take connected component
        if not self.page_graph or len(self.page_graph.nodes) == 0:
            raise RuntimeError("Page graph not initialized or empty")

        # Get strongly connected components
        components = list(nx.strongly_connected_components(self.page_graph))

        # Find component matching size
        for i, component in enumerate(components):
            if len(component) <= cluster_size:
                page_ids = list(component)
                return PageCluster(
                    cluster_id=f"{self.group_id}-cluster-{i}",
                    page_ids=page_ids,
                    relationship_score=0.8,  # TODO: Compute from graph
                    cluster_type=cluster_type or "connected_component",
                    metadata={"component_index": i}
                )

        # Fallback: take first N pages
        all_pages = list(self.page_graph.nodes)[:cluster_size]
        return PageCluster(
            cluster_id=f"{self.group_id}-cluster-fallback",
            page_ids=all_pages,
            relationship_score=0.5,
            cluster_type="fallback",
            metadata={}
        )

    @override
    async def get_all_clusters(
        self,
        max_cluster_size: int = 10,
        min_cluster_size: int = 2
    ) -> AsyncIterator[PageCluster]:
        """Iterate over all page clusters."""
        if not self.page_graph or len(self.page_graph.nodes) == 0:
            return

        # Get strongly connected components
        components = list(nx.strongly_connected_components(self.page_graph))

        for i, component in enumerate(components):
            if min_cluster_size <= len(component) <= max_cluster_size:
                page_ids = list(component)
                yield PageCluster(
                    cluster_id=f"{self.group_id}-cluster-{i}",
                    page_ids=page_ids,
                    relationship_score=0.8,
                    cluster_type="connected_component",
                    metadata={"component_index": i}
                )

    @override
    async def update_page_graph(
        self,
        page_relationships: dict[tuple[str, str], dict[str, Any]]
    ) -> None:
        """Update page graph and persist to storage."""
        # Update graph in memory
        for (src, tgt), rel_info in page_relationships.items():
            if self.page_graph.has_edge(src, tgt):
                edge_data = self.page_graph.get_edge_data(src, tgt)
                edge_data.update(rel_info)
            else:
                self.page_graph.add_edge(src, tgt, **rel_info)

        # Persist to storage
        await self._persist_graph()

    @override
    async def get_page_neighbors(
        self,
        page_id: str,
        max_neighbors: int = 5,
        relationship_types: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """Get nearest neighbor pages."""
        if not self.page_graph or page_id not in self.page_graph:
            return []

        # Get successors (pages referenced by this page)
        neighbors = []
        for neighbor in self.page_graph.successors(page_id):
            edge_data = self.page_graph.get_edge_data(page_id, neighbor)
            weight = edge_data.get("weight", 0.5)
            neighbors.append((neighbor, weight))

        # Sort by weight
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:max_neighbors]

    @override
    def get_config(self) -> dict[str, Any]:
        """Get configuration for recreating this source."""
        return {
            "group_id": self.group_id,
            "repo_path": self.repo_path,
            "tenant_id": self.tenant_id,
            "storage_backend_type": self.storage_backend_type,
            "storage_path": self.storage_path,
        }

    # === Helper Methods ===

    def _keyword_relevance(self, query: str, page_key: str) -> float:
        """Compute keyword-based relevance score."""
        query_words = set(query.lower().split())
        key_words = set(page_key.lower().split())
        if not query_words:
            return 0.0
        overlap = len(query_words & key_words)
        return overlap / len(query_words)

    async def _persist_graph(self) -> None:
        """Persist page graph to storage."""
        graph_dict = {
            "graph": self.page_graph,
            "file_to_page": self.file_to_page,
            "page_to_file": self.page_to_file,
            "page_keys": self.page_keys,
        }

        graph_data = pickle.dumps(graph_dict)
        await self.page_storage.store_page_graph(
            group_id=self.group_id,
            graph_data=graph_data,
            tenant_id=self.tenant_id
        )
        logger.info(f"Persisted page graph for {self.group_id}")

    # === File Relationship Graph Methods (from FileGrouperWithGraph) ===

    def get_file_relationship_graph(self) -> nx.DiGraph | None:
        """Get the file-to-file relationship graph from FileGrouper.

        Returns:
            NetworkX DiGraph with file relationships (imports, dependencies, etc.),
            or None if FileGrouper hasn't been used yet.

        Example:
            ```python
            graph = source.get_file_relationship_graph()
            if graph:
                # Find all files that import 'src/main.py'
                importers = list(graph.predecessors("src/main.py"))
            ```
        """
        if self.file_grouper:
            return self.file_grouper.get_relationship_graph()
        return self._file_relationship_graph

    def query_file_relationships(
        self,
        file_path: str,
        relationship_types: list[str] | None = None,
        min_weight: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Query file-to-file relationships for a specific file.

        Args:
            file_path: Path to file to query
            relationship_types: Filter by types ("import", "dependency", "semantic", etc.)
            min_weight: Minimum relationship weight (0.0-1.0)

        Returns:
            List of relationships with target, type, weight, and metadata

        Example:
            ```python
            # Find all strong import relationships for a file
            imports = source.query_file_relationships(
                "src/main.py",
                relationship_types=["import"],
                min_weight=0.5
            )
            for rel in imports:
                print(f"{rel['target']}: {rel['weight']:.2f}")
            ```
        """
        if self.file_grouper:
            return self.file_grouper.query_file_relationships(
                file_path, relationship_types, min_weight
            )
        return []

    # === File-to-Page Mapping Methods (from GitRepoShardingWithMapping) ===

    def get_page_for_file(self, file_path: str) -> str | None:
        """Get the page ID containing a specific file.

        Args:
            file_path: Path to the file

        Returns:
            Page ID containing the file, or None if file not found

        Example:
            ```python
            page_id = source.get_page_for_file("src/main.py")
            if page_id:
                print(f"File is in page {page_id}")
            ```
        """
        if self.sharding_strategy:
            return self.sharding_strategy.get_page_for_file(file_path)
        return self.file_to_page.get(file_path)

    def get_files_in_page(self, page_id: str) -> list[str]:
        """Get all files in a specific page.

        Args:
            page_id: ID of the page/shard

        Returns:
            List of file paths in the page

        Example:
            ```python
            files = source.get_files_in_page("page-123")
            print(f"Page contains {len(files)} files")
            for file in files:
                print(f"  - {file}")
            ```
        """
        if self.sharding_strategy:
            return self.sharding_strategy.get_files_in_page(page_id)
        return [self.page_to_file.get(page_id, [])]

    def get_file_to_page_mapping(self) -> dict[str, str]:
        """Get complete file-to-page mapping.

        Returns:
            Dictionary mapping file paths to page IDs

        Example:
            ```python
            mapping = source.get_file_to_page_mapping()
            for file_path, page_id in mapping.items():
                print(f"{file_path} → {page_id}")
            ```
        """
        if self.sharding_strategy:
            return self.sharding_strategy.get_file_to_page_map()
        return dict(self.file_to_page)

    def query_files_by_language(self, language: str) -> list[str]:
        """Find all files of a specific programming language.

        Args:
            language: Programming language name (e.g., "python", "javascript")

        Returns:
            List of file paths for that language

        Example:
            ```python
            py_files = source.query_files_by_language("python")
            print(f"Found {len(py_files)} Python files")
            ```
        """
        if self.sharding_strategy:
            return self.sharding_strategy.query_files_by_language(language)
        return []

