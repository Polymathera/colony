

import logging
import pickle
from overrides import override
from typing import Any, AsyncIterator, Literal
import networkx as nx

from colony.python.colony.system import get_vcm

from ...vcm.sources import ContextPageSource, ContextPageSourceFactory
from ...vcm.page_storage import PageStorage, PageStorageConfig
from ...distributed import get_polymathera
from .sharding.file_grouping_wrapper import FileGrouperWithGraph
from .sharding.strategy_wrapper import GitRepoShardingWithMapping


logger = logging.getLogger(__name__)

@ContextPageSourceFactory.register_new_source_type("file_grouper")
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
    async def initialize(self) -> None:
        """Initialize storage and load/build page graph."""
        if self.page_storage is not None and self.page_graph is not None:
            return  # Already initialized

        vcm_handle = get_vcm()
        config: PageStorageConfig | None = vcm_handle.get_page_storage_config()
        if not config:
            raise ValueError("Missing PageStorageConfig in VCM")

        self.page_storage = PageStorage(
            group_id = self.group_id,
            tenant_id = self.tenant_id,
            backend_type=config.backend_type,
            storage_path=config.storage_path,
            s3_bucket=config.s3_bucket,
        )
        await self.page_storage.initialize()

        # Try to load existing graph
        page_graph = await self.page_storage.retrieve_page_graph()

        if page_graph:
            self.page_graph = page_graph
            logger.info(
                f"Loaded page graph for {self.group_id}: "
                f"{len(self.page_graph.nodes)} nodes, {len(self.page_graph.edges)} edges"
            )
        else:
            # Build new graph (requires FileGrouper - deferred to first usage)
            logger.info(f"No existing page graph for {self.group_id}, will build on first use")
            self.page_graph = nx.DiGraph()

        self.file_to_page = await self.page_storage.retrieve_page_graph_level_data("file_to_page") or {}
        self.page_to_file = await self.page_storage.retrieve_page_graph_level_data("page_to_file") or {}
        self.page_keys = await self.page_storage.retrieve_page_graph_level_data("page_keys") or {}

    # === Helper Methods ===

    def _keyword_relevance(self, query: str, page_key: str) -> float:
        """Compute keyword-based relevance score."""
        query_words = set(query.lower().split())
        key_words = set(page_key.lower().split())
        if not query_words:
            return 0.0
        overlap = len(query_words & key_words)
        return overlap / len(query_words)

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
        # TODO: Why is this method not used anywhere?
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
        # TODO: Why is this method not used anywhere?
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
            self.file_to_page = self.sharding_strategy.get_file_to_page_map()
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

