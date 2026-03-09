

import logging
import pickle
from collections import defaultdict
from overrides import override
from typing import Any, AsyncIterator, Literal
import networkx as nx

from polymathera.colony.vcm.sources import ContextPageSource, ContextPageSourceFactory, BuilInContextPageSourceType
from polymathera.colony.vcm.models import MmapConfig, ContextPageId, VirtualContextPage
from polymathera.colony.vcm.page_storage import PageStorage, PageStorageConfig
from polymathera.colony.distributed import get_polymathera

from .sharding.file_grouping_wrapper import FileGrouperWithGraph
from .sharding.strategy_wrapper import GitRepoShardingWithMapping
from .sharding.prompting import IdentityPromptStrategy
from .sharding.strategy import GitRepoShardingStrategy


logger = logging.getLogger(__name__)

@ContextPageSourceFactory.register_new_source_type(BuilInContextPageSourceType.FILE_GROUPER.value)
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
        *,
        scope_id: str,  # Repo ID
        group_id: str,
        tenant_id: str,
        mmap_config: MmapConfig,
        origin_url: str,   # Git repo URL (https:// or file://)
        branch: str = "main",
        commit: str = "HEAD",
    ):
        """Initialize file-grouper-based context page source.

        Args:
            scope_id: Repository (or scope) identifier
            group_id: Group identifier for VCM address space
            tenant_id: Tenant identifier
            mmap_config: Configuration for memory-mapped page graph data
            origin_url: Git repository URL (https:// or file:// for local repos)
            branch: Git branch to check out
            commit: Git commit SHA (defaults to branch HEAD)
        """
        super().__init__(scope_id=scope_id, group_id=group_id, tenant_id=tenant_id, mmap_config=mmap_config)
        self.origin_url = origin_url
        self.branch = branch
        self.commit = commit

        # File grouping and sharding wrappers (expose graph and file-to-page mapping)
        self.file_grouper: FileGrouperWithGraph | None = None
        self.sharding_strategy: GitRepoShardingWithMapping | None = None

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

        from polymathera.colony.system import get_vcm
        vcm_handle = get_vcm()
        config: PageStorageConfig | None = await vcm_handle.get_page_storage_config()
        if not config:
            raise ValueError("Missing PageStorageConfig in VCM")

        self.page_storage = PageStorage(
            backend_type=config.backend_type,
            storage_path=config.storage_path,
            s3_bucket=config.s3_bucket,
        )
        await self.page_storage.initialize()

        # Try to load existing graph
        page_graph = await self.page_storage.retrieve_page_graph(
            group_id=self.group_id,
            tenant_id=self.tenant_id,
        )

        if page_graph and page_graph.number_of_nodes() > 0:
            self.page_graph = page_graph
            self.file_to_page = await self.page_storage.retrieve_page_graph_level_data(
                data_key="file_to_page",
                tenant_id=self.tenant_id,
                group_id=self.group_id
            ) or {}
            self.page_to_file = await self.page_storage.retrieve_page_graph_level_data(
                data_key="page_to_file",
                tenant_id=self.tenant_id,
                group_id=self.group_id
            ) or {}
            self.page_keys = await self.page_storage.retrieve_page_graph_level_data(
                data_key="page_keys",
                tenant_id=self.tenant_id,
                group_id=self.group_id
            ) or {}
            logger.info(
                f"Loaded page graph for {self.tenant_id}:{self.group_id}:{self.scope_id}: "
                f"{len(self.page_graph.nodes)} nodes, {len(self.page_graph.edges)} edges"
            )
        else:
            # No existing graph — build from repository
            logger.info(
                f"No existing page graph for {self.tenant_id}:{self.group_id}:{self.scope_id}, "
                f"building from repository at {self.origin_url} (branch={self.branch}, commit={self.commit})"
            )
            await self._build_and_persist_page_graph()

    async def _build_and_persist_page_graph(self) -> None:
        """Build page graph from repository and persist everything.

        Uses GitRepoShardingStrategy to:
        1. Analyze file relationships (imports, dependencies, etc.)
        2. Group related files into shards (pages)
        3. Build page-level relationship graph

        Then creates VirtualContextPages from shards and persists:
        - Page graph (networkx DiGraph)
        - File-to-page / page-to-file mappings
        - Individual pages (text for remote LLMs)
        """
        import git
        import time as _time

        build_start = _time.time()
        logger.info(
            f"Building page graph from {self.origin_url} "
            f"(branch={self.branch}, commit={self.commit})"
        )

        # Clone the repo through GitStorage so it lands under the managed
        # prefix.  This ensures normalize_file_path / denormalize_file_path
        # are idempotent for all file paths in the repository.
        # We call git_storage directly because the Storage wrapper's
        # _validate_repo_url() rejects file:// URLs via HttpUrl().
        polymathera = get_polymathera()
        storage = await polymathera.get_storage()
        repo_path = await storage.git_storage.clone_or_retrieve_repository(
            origin_url=self.origin_url,
            branch=self.branch,
            commit=self.commit,
            vmr_id=self.group_id,
        )
        logger.info(f"Repository cloned to {repo_path} ({_time.time() - build_start:.1f}s)")

        prompt_strategy = IdentityPromptStrategy()
        strategy = GitRepoShardingStrategy(
            prompt_strategy=prompt_strategy,
            config=None,  # Auto-loaded from Polymathera config system
        )
        await strategy.initialize()

        try:
            repo = git.Repo(str(repo_path))
            logger.info(f"Starting create_shards_with_graph ({_time.time() - build_start:.1f}s elapsed)")
            result = await strategy.create_shards_with_graph(
                group_id=self.group_id,
                repo=repo,
            )

            logger.info(
                f"Created {len(result.shards)} shards from {self.origin_url} "
                f"({_time.time() - build_start:.1f}s elapsed)"
            )

            # Page graph
            if result.page_graph and result.page_graph.number_of_nodes() > 0:
                self.page_graph = result.page_graph
                logger.info(
                    f"Page graph: {self.page_graph.number_of_nodes()} pages, "
                    f"{self.page_graph.number_of_edges()} cross-page relationships"
                )
            else:
                logger.warning("No page graph from sharding strategy, creating minimal graph")
                self.page_graph = nx.DiGraph()
                for shard in result.shards:
                    self.page_graph.add_node(shard.shard_id)

            # File-to-page and page-to-file mappings (normalized paths)
            self.file_to_page = result.file_to_page or {}
            if not self.file_to_page:
                polymathera = get_polymathera()
                for shard in result.shards:
                    for seg in shard.metadata.file_segments:
                        normalized = await polymathera.normalize_file_path(seg.file_path)
                        self.file_to_page[normalized] = shard.shard_id

            ptf: dict[str, list[str]] = defaultdict(list)
            for file_path, page_id in self.file_to_page.items():
                ptf[page_id].append(file_path)
            self.page_to_file = dict(ptf)

            # Page keys (page_id -> summary for relevance matching)
            # TODO: This should be page_id -> PageKey and should use
            # a KeyGenerator strategy to create keys, rather than just joining file names
            # TODO: Still key generation should be done in a lazy way in the KeyRegistry, not eagerly here at graph build time
            ### self.page_keys = {}
            ### for shard in result.shards:
            ###     file_names = [
            ###         seg.file_path.rsplit("/", 1)[-1]
            ###         for seg in shard.metadata.file_segments
            ###     ]
            ###     self.page_keys[shard.shard_id] = " ".join(file_names)

            # Create and persist VirtualContextPages
            logger.info(f"Persisting {len(result.shards)} pages ({_time.time() - build_start:.1f}s elapsed)")
            for shard in result.shards:
                page = VirtualContextPage(
                    page_id=shard.shard_id,
                    tokens=[],  # Remote deployments use text; vLLM path would tokenize separately
                    text=shard.raw_content,
                    size=shard.metadata.token_count or max(1, len(shard.raw_content) // 4),
                    metadata={
                        "source": FileGrouperContextPageSource.get_source_metadata(self.scope_id),
                        "files": [seg.file_path for seg in shard.metadata.file_segments],
                        "file_count": len(shard.metadata.file_segments),
                        "content_size_bytes": shard.metadata.content_size_bytes,
                    },
                    scope_id=self.scope_id,
                    group_id=self.group_id,
                    tenant_id=self.tenant_id,
                )
                await self.page_storage.store_page(page)

            logger.info(f"Stored {len(result.shards)} pages to PageStorage")

            # Persist graph and mappings
            await self.page_storage.store_page_graph(
                tenant_id=self.tenant_id,
                group_id=self.group_id,
                graph_data=self.page_graph
            )
            await self.page_storage.store_page_graph_level_data(
                tenant_id=self.tenant_id,
                group_id=self.group_id,
                data_key="file_to_page",
                graph_data=self.file_to_page
            )
            await self.page_storage.store_page_graph_level_data(
                tenant_id=self.tenant_id,
                group_id=self.group_id,
                data_key="page_to_file",
                graph_data=self.page_to_file
            )
            await self.page_storage.store_page_graph_level_data(
                tenant_id=self.tenant_id,
                group_id=self.group_id,
                data_key="page_keys",
                graph_data=self.page_keys
            )

            total_time = _time.time() - build_start
            logger.info(
                f"Persisted page graph for {self.tenant_id}:{self.group_id}:{self.scope_id}: "
                f"{self.page_graph.number_of_nodes()} pages, "
                f"{len(self.file_to_page)} files mapped "
                f"(total build time: {total_time:.1f}s)"
            )

        finally:
            await strategy.cleanup()

    # === Abstract method implementations (ContextPageSource) ===

    @override
    async def claim_orphaned_events(self) -> None:
        """No event stream for file-grouper source (files are static)."""
        pass

    @override
    async def shutdown(self) -> None:
        """Clean up resources."""
        self.page_storage = None
        self.page_graph = None

    @override
    async def get_page_id_for_record(self, record_id: str) -> ContextPageId | None:
        """Get page ID for a file path (record_id = file path)."""
        return self.file_to_page.get(record_id)

    @override
    async def get_record_ids_for_page(self, page_id: ContextPageId) -> list[str]:
        """Get file paths contained in a page."""
        return self.page_to_file.get(page_id, [])

    @override
    async def get_all_mapped_records(self) -> dict[str, ContextPageId]:
        """Get complete file-to-page mapping."""
        return dict(self.file_to_page)

    @override
    async def get_all_mapped_pages(self) -> dict[ContextPageId, list[str]]:
        """Get complete page-to-files mapping."""
        return dict(self.page_to_file)

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
        return self.page_to_file.get(page_id, [])

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

