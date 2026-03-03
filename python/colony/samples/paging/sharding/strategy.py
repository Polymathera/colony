from __future__ import annotations

import asyncio
import hashlib
import logging
import json
import string
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import git
import magic  # python-magic for file type detection
import networkx as nx
from circuitbreaker import circuit

from colony.distributed.caching.simple import CacheConfig
from colony.distributed.config import ConfigComponent, register_polymathera_config
from colony.distributed import get_polymathera
from colony.distributed.metrics.common import BaseMetricsMonitor
from colony.utils.git import validate_git_repository, configure_git_safety
from colony.utils.retry import standard_retry

from .types import RepositoryShard, ShardFileSegment, ShardMetadata, ShardingError
from .prompting import ShardedInferencePromptStrategy
from .code_splitting import CodeSplitter, CodeSplitterConfig, SplitStrategy
from .file_grouping import FileGrouperConfig
from .file_grouping_wrapper import FileGrouperWithGraph
from .tokenization import TokenizationConfig, TokenizationStrategy, TokenManager
from .analyzers.base import FileContentCache

logger = logging.getLogger(__name__)




################################################################################
# Resource management and incremental sharding
################################################################################


@register_polymathera_config()
class ShardResourceConfig(ConfigComponent):
    # Maximum memory usage per repo
    max_memory_mb: int = 1024

    # Maximum concurrent shards being processed
    max_concurrent_shards: int = 10

    # Timeout for processing a single shard
    shard_timeout_seconds: int = 300

    # Whether to enable incremental sharding
    enable_incremental: bool = True

    # Maximum file count for non-incremental processing
    max_files_non_incremental: int = 1000

    CONFIG_PATH: ClassVar[str] = "llms.sharding.strategy.shard_resource"


class IncrementalShardManager:
    """Manages incremental sharding of large repositories"""

    def __init__(self, config: ShardResourceConfig | None = None):
        self.config: ShardResourceConfig | None = config
        self.processed_files: set[str] = set()
        self.current_batch: list[git.Blob] = []
        self.memory_usage: int = 0

    async def initialize(self):
        self.config = await ShardResourceConfig.check_or_get_component(self.config)

    async def process_repo(self, repo: git.Repo) -> AsyncIterator[RepositoryShard]:
        """Process repository in batches to manage resources"""
        files = list(repo.head.commit.tree.traverse())

        if len(files) <= self.config.max_files_non_incremental:
            # Process normally for small repos
            async for shard in self._process_batch(files):  # TODO: Implement
                yield shard
            return

        # Process in batches for large repos
        for batch in self._create_batches(files):
            self.current_batch = batch
            async for shard in self._process_batch(batch):  # TODO: Implement
                yield shard

            # Update progress
            self.processed_files.update(f.path for f in batch)
            self.memory_usage = 0

    def _create_batches(self, files: list[git.Blob]) -> list[list[git.Blob]]:
        """Create batches of files for incremental processing"""
        batches = []
        current_batch = []
        current_size = 0

        for file in sorted(files, key=lambda f: f.size):
            if current_size + file.size > self.config.max_memory_mb * 1024 * 1024:
                batches.append(current_batch)
                current_batch = []
                current_size = 0

            current_batch.append(file)
            current_size += file.size

        if current_batch:
            batches.append(current_batch)

        return batches


################################################################################
# Sharding
################################################################################


@dataclass
class ShardingResult:
    """Result from repository sharding operation."""
    shards: list[RepositoryShard]
    page_graph: nx.DiGraph | None = None
    file_to_page: dict[str, str] | None = None


@dataclass
class FileSplitConfig:
    # Files larger than this will be split across shards
    large_file_threshold_bytes: int = 1024 * 1024  # 1MB

    # Minimum size for a shard to make splitting worthwhile
    min_shard_size_bytes: int = 1024 * 32  # 32KB

    # Whether to try keeping related code together
    preserve_context: bool = True

    # Maximum distance (in lines) to look for related code
    context_window_lines: int = 50


@register_polymathera_config()
class ShardingConfig(ConfigComponent):
    """Configuration for sharding strategy"""

    # Token limits
    max_tokens_per_shard: int = 4096  # Maximum tokens per shard
    max_tokens_per_file: int = 8192  # Maximum tokens for a single file
    target_tokens_per_shard: int = 3072  # Target tokens to leave room for prompts
    max_files_per_shard: int = 10

    # Concurrency settings
    max_concurrent_files: int = 10
    max_concurrent_shards: int = 5

    # Performance settings
    batch_size: int = 100
    timeout_seconds: int = 30

    # File handling
    skip_binary: bool = True
    binary_size_limit_bytes: int = 1024 * 1024  # 1MB
    replace_decoding_errors: bool = False
    binary_extensions: list[str] = [".bin", ".exe", ".dll", ".so", ".dylib"]

    # Resource management
    enable_circuit_breakers: bool = True

    # Cache settings
    cache_binary_files: bool = False
    cache_config: CacheConfig = CacheConfig(
        ttl_seconds=86400 * 30,
        max_size_mb=1024,
        compression_level=3,
        serialization_format="pickle",
        max_concurrent_ops=10,
        enable_compression=True,
        batch_size=100,
    )
    code_splitter_config: CodeSplitterConfig = CodeSplitterConfig()
    file_grouping_config: FileGrouperConfig = FileGrouperConfig()
    tokenization_config: TokenizationConfig = TokenizationConfig()
    shard_resource_config: ShardResourceConfig = ShardResourceConfig()

    CONFIG_PATH: ClassVar[str] = "llms.sharding.strategy"


# TODO: Performance Optimizations
# 1. Implement parallel tokenization for segments
# 2. Add token count caching at segment level
# 3. Optimize segment combining algorithm
# 4. Add streaming processing for very large files

# TODO: Resource Management
# 1. Add memory usage tracking
# 2. Implement adaptive batch sizing
# 3. Add resource-aware scheduling
# 4. Implement backpressure mechanisms

# TODO: Error Handling
# 1. Add more granular error recovery
# 2. Implement partial success handling
# 3. Add retry strategies for specific failures
# 4. Enhance error reporting

# TODO: Monitoring
# 1. Add detailed segment metrics
# 2. Track relationship preservation
# 3. Monitor token distribution
# 4. Add performance profiling

# TODO: Caching
# 1. Implement multi-level caching
# 2. Add predictive cache warming
# 3. Optimize cache key generation
# 4. Add cache eviction policies

# TODO: Features
# 1. Add support for incremental updates
# 2. Implement cross-shard references
# 3. Add content-aware splitting
# 4. Enhance relationship scoring

# TODO: Testing
# 1. Add performance benchmarks
# 2. Implement stress testing
# 3. Add edge case coverage
# 4. Test different configurations

# Questions to clarify:
# 1. Should we implement streaming processing for very large files?
# 2. How should we handle cross-references between shards?
# 3. What's the optimal caching strategy for different deployment scenarios?
# 4. How should we handle token budget across multiple repositories?
# 5. Should we implement predictive sharding based on usage patterns?
# 6. How should we handle repository updates and incremental processing?


class GitRepoShardCache:
    """Specialized cache for repository shards using TokenizedFileCache"""

    def __init__(
        self, config: CacheConfig | None = None, cache_binary_files: bool = False
    ):
        self.config = config
        self.cache_binary_files = cache_binary_files
        self.cache = None

    async def initialize(self):
        self.config = await CacheConfig.check_or_get_component(self.config)
        self.cache = await get_polymathera().create_distributed_simple_cache(
            namespace="shards",  # TODO: Does this need to be VMR-specific?
            config=self.config,
        )

    async def cleanup(self) -> None:
        from colony.utils import cleanup_dynamic_asyncio_tasks
        try:
            await cleanup_dynamic_asyncio_tasks(self, raise_exceptions=False)
            await self.cache.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up GitRepoShardCache tasks: {e}")

    async def get_shards(self, key: str) -> list[RepositoryShard] | None:
        """Get shard from cache"""
        logger.info(f"________ get_shards: 0 {key}")
        if not self.config.enable_caching:
            logger.info(f"________ get_shards: 1 cache not enabled")
            return None

        try:
            logger.info(f"________ get_shards: 1 cache enabled")
            shards = await self.cache.get(key)
            logger.info(f"________ get_shards: 2 - {len(shards) if shards else 0} shards")
            return shards
        except Exception as e:
            logger.error(f"Error retrieving shards: {e}")
            return None

    async def set_shards(self, key: str, shards: list[RepositoryShard]):
        """Store shard in cache with metrics and error handling"""
        try:
            # Skip binary files if configured
            logger.info(f"________ set_shards: 0 - {len(shards)} shards")
            if not self.cache_binary_files and any(shard.metadata.binary_files for shard in shards):
                logger.info(f"________ set_shards: 1 cache binary files not enabled")
                return

            # TODO: Track metrics
            # Store in cache
            logger.info(f"________ set_shards: 2 cache enabled")
            await self.cache.set(key, shards)
            logger.info(f"________ set_shards: 3 cache set")
        except Exception as e:
            logger.error(f"Error storing shards: {e}")



class GitRepoShardingMetricsMonitor(BaseMetricsMonitor):
    """Centralized monitoring for GitRepoShardingStrategy."""

    def __init__(self, enable_http_server: bool = True):
        # Initialize base class with HTTP server
        super().__init__(
            enable_http_server=enable_http_server,
            service_name="git-repo-sharding-metrics",
        )

        self.logger.info(f"Initializing GitRepoShardingMetricsMonitor instance {id(self)}...")

        # Use consistent label names across all metrics
        common_labels = ["repo_id"]

        self.shard_size = self.create_histogram(
            "repo_shard_size_bytes",
            "Size distribution of repository shards",
            buckets=[1024, 4096, 16384, 32768, 65536],
            #buckets=[1024, 10 * 1024, 100 * 1024, 1024 * 1024, 10 * 1024 * 1024],
            labelnames=common_labels
        )
        self.processing_time = self.create_histogram(
            "repo_sharding_duration_seconds",
            "Time taken to shard repository",
            labelnames=common_labels
        )
        self.shard_count = self.create_counter(
            "repo_shard_count",
            "Total number of shards created",
            labelnames=common_labels
        )
        self.binary_files = self.create_counter(
            "repo_binary_files",
            "Number of binary files encountered",
            labelnames=common_labels
        )
        self.processing_errors = self.create_counter(
            "repo_sharding_errors",
            "Number of errors during sharding",
            labelnames=common_labels
        )



class GitRepoShardingStrategy:
    """
    Shards a git repository into chunks suitable for LLM processing.

    Each repo shard is intended to capture enough contiguous code chunk from the repo
    (possibly spanning multiple files) that fits within one LLM context window (along with the prompt).
    TODO: Implement more sophisticated sharding strategies.

    Key features:
    - Handles binary files appropriately
    - Splits large files across shards
    - Maintains detailed metadata
    - Concurrent processing where beneficial
    - Robust error handling and retries
    - Metrics for monitoring
    - Configurable concurrency levels

    Features:
    1. Detailed Metadata:
        - Including file segments, sizes, mime types, and git commit info.
    2. Binary File Handling:
        - Proper binary file detection using multiple heuristics
        - Option to skip or include binary files
        - Metadata-only shards for binary files
    3. Concurrency Control:
        - Separate semaphores for file and shard processing
        - Configurable concurrency limits
        - Async processing with proper error handling
    4. Error Handling:
        - Retries for transient failures
        - Detailed error logging
        - Proper exception propagation
        - Encoding fallbacks for text files
    5. Monitoring:
        - Prometheus metrics for sizes, counts, errors
        - Processing time tracking
        - Binary file tracking
    6. Performance Considerations:
        - Efficient content splitting
        - Bounded memory usage
        - Concurrent processing where beneficial
    7. Language-Aware Splitting:
        - Uses tree-sitter for structural parsing of common languages
        - Falls back to Pygments for basic syntax awareness
        - Final fallback to line-based splitting
    8. Structural Boundaries:
        - Respects function/class definitions
        - Splits at statement boundaries when needed
        - Never splits mid-line
    9. Flexibility:
        - Supports multiple languages
        - Extensible for new languages
        - Configurable split points
    10. Robustness:
        - Graceful fallbacks if parsing fails
        - Handles unknown languages
        - Maintains size limits

    TODO/Questions:
    - Should we maintain file boundaries in shards or split mid-file?
    - How to handle file renames/moves between commits?
    - Should we cache shards to avoid reprocessing unchanged files?
    - How to handle very large repositories efficiently?
    - Should we implement incremental sharding for repo updates?
    - How to handle merge conflicts in file segments?

    Areas needing clarification:
    1. Caching Strategy:
        - Should we cache shards to avoid reprocessing unchanged files?
        - How to handle cache invalidation?
    2. File Boundaries:
        - Should shards respect file boundaries or split mid-file?
        - How to handle very large single files?
    3. Git Integration:
        - How to handle file renames/moves between commits?
        - Should we track file history?
    4. Resource Management:
        - How to handle very large repositories?
        - Should we implement incremental sharding?
    5. Content Processing:
        - Should we preprocess/normalize content?
        - How to handle special file types (e.g., notebooks)?
    6. Security:
        - How to handle sensitive content?
        - Should we implement content filtering?
    7. Language Support:
        - Which languages should be prioritized?
        - How to handle mixed language files?
        - Custom splitting rules per language?
    8. Performance:
        - Parser initialization cost
        - Caching parsed ASTs
        - Parallel processing of large files
    9. Edge Cases:
        - Comments and documentation blocks
        - Preprocessor directives
        - Template/macro code
    """

    def __init__(
        self,
        prompt_strategy: ShardedInferencePromptStrategy,
        config: ShardingConfig | None = None,
    ):
        self.prompt_strategy = prompt_strategy
        self.config: ShardingConfig | None = config
        self._token_manager = None
        # Code splitter for language-aware splitting
        self._code_splitter = None

        # Initialize locks and semaphores
        self.file_semaphore = None
        self.shard_semaphore = None

        # Metrics
        self.metrics = GitRepoShardingMetricsMonitor()

        # Initialize shard cache with proper configuration
        self.shard_cache = None
        self._file_content_cache: FileContentCache | None = None

        # Incremental sharding
        self.incremental_manager = None

        # Track processed commits
        self.processed_commits: dict[str, set[str]] = {}
        self.commit_history: dict[str, list[str]] = {}

        # Setup circuit breakers
        self._setup_circuit_breakers()

    async def initialize(self):
        self.config = await ShardingConfig.check_or_get_component(self.config)
        self._file_content_cache = FileContentCache()
        await self._file_content_cache.initialize()

        # Initialize locks and semaphores
        self.file_semaphore = asyncio.Semaphore(self.config.max_concurrent_files)
        self.shard_semaphore = asyncio.Semaphore(self.config.max_concurrent_shards)

        # Initialize shard cache with proper configuration
        self.shard_cache = GitRepoShardCache(
            self.config.cache_config,
            cache_binary_files=self.config.cache_binary_files,
        )
        await self.shard_cache.initialize()

        # Incremental sharding
        self.incremental_manager = IncrementalShardManager(
            self.config.shard_resource_config
        )
        await self.incremental_manager.initialize()

    async def get_token_manager(self) -> TokenManager:
        if self._token_manager is None:
            self._token_manager = TokenManager(
                file_content_cache=self._file_content_cache,
                config=self.config.tokenization_config,
            )
            await self._token_manager.initialize()
        return self._token_manager

    async def get_code_splitter(self) -> CodeSplitter:
        # Initialize code splitter if not exists
        if self._code_splitter is None:
            self.config.code_splitter_config.max_shard_size = self.config.max_tokens_per_shard
            self.config.code_splitter_config.max_workers = self.config.max_concurrent_files
            self._code_splitter = CodeSplitter(
                self.config.code_splitter_config
            )
            await self._code_splitter.initialize()
        return self._code_splitter

    async def cleanup(self):
        """Cleanup the sharding strategy"""
        # Cleanup: properly close the caches to stop monitoring tasks
        try:
            await self.shard_cache.cleanup()
            if self._token_manager is not None:
                await self._token_manager.cleanup()
        except Exception as e:
            logger.warning(f"Failed to cleanup sharding strategy cache: {e}")

    def _setup_circuit_breakers(self):
        """Setup circuit breakers for critical operations"""
        self._storage_breaker = circuit(
            failure_threshold=5, recovery_timeout=30, name="shard_storage"
        )

        self._processing_breaker = circuit(
            failure_threshold=3, recovery_timeout=15, name="shard_processing"
        )

    @staticmethod
    def _get_repo_files_sync(repo_path: str) -> list[str]:
        """Get all tracked files in the repository.

        Creates its own git.Repo instance so this can safely run in a
        background thread without sharing the parent repo's git subprocess.
        """
        repo = git.Repo(repo_path)
        base = Path(repo_path)
        return [
            str(base / blob.path)
            for blob in repo.head.commit.tree.traverse()
            if blob.type == "blob"
        ]

    @standard_retry(logger)
    async def create_shards_with_graph(
        self, group_id: str, repo: git.Repo
    ) -> ShardingResult:
        """
        Create shards from a git repository and extract the page relationship graph.

        This method builds shards using FileGrouperWithGraph and extracts the file relationship
        graph, converting it to a page-level graph for cache-aware scheduling by agents.

        Args:
            group_id: Unique identifier for the repository and its VMR context
            repo: GitPython repository object

        Returns:
            ShardingResult with shards, page_graph, and file_to_page mapping

        Raises:
            ShardingError: If sharding fails after retries
        """
        file_grouper = None
        page_graph = None
        file_to_page = None

        try:
            # Create shards (this will use FileGrouperWithGraph internally)
            shards, file_grouper = await self._create_shards_internal(group_id, repo)

            # Extract file relationship graph from FileGrouperWithGraph
            file_graph = file_grouper.get_relationship_graph()
            if file_graph:
                # Build page graph from file graph
                page_graph = await self._build_page_graph_from_file_graph(file_graph, shards)

                # Build file-to-page mapping using normalized paths (canonical form)
                polymathera = get_polymathera()
                file_to_page = {}
                for shard in shards:
                    for segment in shard.metadata.file_segments:
                        normalized = await polymathera.normalize_file_path(segment.file_path)
                        file_to_page[normalized] = shard.shard_id

                logger.info(
                    f"________ create_shards_with_graph: Extracted page graph: "
                    f"{page_graph.number_of_nodes()} pages, "
                    f"{page_graph.number_of_edges()} cross-page relationships"
                )
            else:
                logger.warning("________ create_shards_with_graph: No file graph available from FileGrouper")

            return ShardingResult(
                shards=shards,
                page_graph=page_graph,
                file_to_page=file_to_page
            )

        finally:
            if file_grouper:
                await file_grouper.cleanup()



    @standard_retry(logger)
    async def create_shards(
        self, group_id: str, repo: git.Repo
    ) -> list[RepositoryShard]:
        """
        Create shards from a git repository with retries on failure.
        Uses FileGrouper to cluster related files together before sharding.

        Each shard of code should be pinned to an LLM instance right after shards are created.
        Only the variable parts of the LLMs context windows is appended to the respective code shards of each LLM
        when run_prompt or _run_query call the LLM framework. So, the LLM framework should support the
        ability to keep the prefix of the prompt fixed and alter only the suffix extending that prefix.
        This will save a lot of time and energy with autoregressive LLMs.

        Args:
            group_id: Unique identifier for the repository and its VMR context
            repo: GitPython repository object

        Returns:
            List of RepositoryShard objects

        Raises:
            ShardingError: If sharding fails after retries
        """
        shards, _ = await self._create_shards_internal(group_id, repo)
        return shards

    async def _create_shards_internal(
        self, group_id: str, repo: git.Repo
    ) -> tuple[list[RepositoryShard], FileGrouperWithGraph]:
        """
        Internal method that creates shards and returns the file_grouper for graph extraction.

        Args:
            group_id: Unique identifier for the repository and its VMR context
            repo: GitPython repository object

        Returns:
            Tuple of (shards, file_grouper)

        Raises:
            ShardingError: If sharding fails
        """
        file_grouper = None
        try:
            # Configure Git safety settings to prevent ownership issues
            configure_git_safety(repo)

            # Validate Git repository health
            if not validate_git_repository(repo):
                logger.error(f"Git repository validation failed for {group_id}")
                raise ShardingError(f"Git repository is not in a healthy state: {repo.working_dir}")

            # TODO: Cache Invalidation: Ensure that the cache is invalidated or
            # updated appropriately when the repository changes. This implementation
            # assumes that the commit_hash is a sufficient identifier for
            # the current state of the repository.
            start_time = time.time()
            commit_hash = repo.head.commit.hexsha

            # Check if shards are already cached
            cache_key = f"{group_id}:{commit_hash}"
            logger.info(f"________ create_shards: 0 {cache_key}")
            cached_shards = await self._storage_breaker(self.shard_cache.get_shards)(cache_key)
            if cached_shards:
                logger.info(f"________ create_shards: 1 {len(cached_shards)} cached shards")
                # Return cached shards with file_grouper=None (no graph available from cache)
                return cached_shards, None

            # Get all tracked files.  Runs in a thread with its own git.Repo
            # to avoid blocking the event loop (GitPython's repo object is NOT
            # thread-safe — each instance needs its own git subprocess).
            repo_path = Path(repo.working_dir)
            logger.info(f"________ create_shards: 2.1 {repo_path}")
            files = await asyncio.to_thread(
                self._get_repo_files_sync, str(repo_path)
            )
            logger.info(f"________ create_shards: 2.2 {len(files)} files")

            # Initialize FileGrouper with appropriate config
            grouper_config = self.config.file_grouping_config
            logger.info(f"________ create_shards: 2.2 {grouper_config}")
            grouper_config.max_group_size = self.config.max_files_per_shard
            logger.info(f"________ create_shards: 2.3 {grouper_config.max_group_size}")
            grouper_config.max_group_tokens = self.config.target_tokens_per_shard
            logger.info(f"________ create_shards: 2.4 {grouper_config.max_group_tokens}")

            token_manager = await self.get_token_manager()
            file_grouper = FileGrouperWithGraph(
                token_manager=token_manager,
                config=grouper_config,
                file_content_cache=self._file_content_cache,
            )
            await file_grouper.initialize()
            logger.info(f"________ create_shards: 3 file grouper initialized")
            # Group related files together
            # TODO: Ensure this is cached
            file_groups = await file_grouper.group_files(group_id, repo, files)
            logger.info(f"________ create_shards: 4 {len(file_groups)} file groups")
            # Process groups concurrently with bounded concurrency.
            # Pass file paths directly from the group — no need to
            # re-traverse the git tree per group.
            tasks = []
            async with asyncio.TaskGroup() as tg:
                for group in file_groups:
                    tasks.append(
                        tg.create_task(
                            self._process_file_group(
                                group_id=group_id,
                                file_paths=group.files,
                                commit_hash=commit_hash,
                                repo_path=repo_path,
                            )
                        )
                    )
            logger.info(f"________ create_shards: 5 {len(tasks)} tasks")
            shards = [shard for task in tasks for shard in (task.result() or [])]
            logger.info(f"________ create_shards: 6 {len(shards)} shards")

            # Cache the created shards
            await self._storage_breaker(self.shard_cache.set_shards)(cache_key, shards)
            logger.info(f"________ create_shards: 7 Cached {len(shards)} shards for {cache_key}")
            # Update metrics
            duration = time.time() - start_time
            self.metrics.processing_time.labels(repo_id=group_id).observe(duration)
            self.metrics.shard_count.labels(repo_id=group_id).inc(len(shards))
            return shards, file_grouper

        except* asyncio.CancelledError:
            logger.warning("Sharding operation cancelled")
            raise
        except* Exception as e:
            self.metrics.processing_errors.labels(repo_id=group_id).inc()
            logger.error(f"Failed to create shards: {e!s}")
            raise ShardingError(f"Failed to create shards: {e!s}") from e

    def _decode_content(self, content: bytes) -> str | None:
        """Decode content to a string with fallback encodings"""
        if self.config.replace_decoding_errors:
            return content.decode("utf-8", errors="replace")
        # Decode text content
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            # Try alternate encodings
            for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    return content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            else:
                return None

    def _is_binary_file(self, file_path: str, content: bytes, mime_type: str) -> bool:
        """
        Determine if a file should be treated as binary.
        Uses multiple heuristics for reliable detection.
        """
        # Check extension
        if Path(file_path).suffix.lower() in self.config.binary_extensions:
            return True

        # Check mime type
        if not mime_type.startswith(("text/", "application/json")):
            return True

        # Check for null bytes
        if b"\x00" in content[:1024]:
            return True

        # Check if mostly non-printable characters
        printable = sum(c in string.printable.encode() for c in content[:1024])
        if printable / len(content[:1024]) < 0.8:
            return True

        return False

    async def _build_page_graph_from_file_graph(
        self,
        file_graph: nx.DiGraph,
        shards: list[RepositoryShard],
    ) -> nx.DiGraph:
        """
        Convert file-level relationship graph to page-level graph.

        The file graph (from FileGrouper) captures relationships between files (imports, dependencies, etc.).
        This method converts those relationships into page-level relationships for cache-aware scheduling.

        Args:
            file_graph: NetworkX DiGraph with file-to-file relationships
            shards: List of RepositoryShard objects (pages)

        Returns:
            NetworkX DiGraph with page-to-page relationships
        """
        polymathera = get_polymathera()

        # Build file-to-page mapping from shards.
        # Segment file paths are denormalized (absolute), but graph edges use
        # normalized paths (via polymathera.normalize_file_path in FileGrouper).
        # Normalize here so lookups match the graph edge endpoints.
        file_to_page: dict[str, str] = {}
        for shard in shards:
            for segment in shard.metadata.file_segments:
                normalized = await polymathera.normalize_file_path(segment.file_path)
                file_to_page[normalized] = shard.shard_id

        # Create page graph with nodes
        page_graph = nx.DiGraph()
        for shard in shards:
            page_graph.add_node(
                shard.shard_id,
                metadata={
                    "file_count": len(shard.metadata.file_segments),
                    "content_size": shard.metadata.content_size_bytes,
                    "files": [seg.file_path for seg in shard.metadata.file_segments],
                },
            )

        # Add edges for cross-page relationships
        # Aggregate file relationships into page relationships
        page_relationships: dict[tuple[str, str], dict[str, Any]] = {}

        for src_file, tgt_file, edge_data in file_graph.edges(data=True):
            src_page = file_to_page.get(src_file)
            tgt_page = file_to_page.get(tgt_file)

            # Only create edges for cross-page relationships (not within same page)
            if src_page and tgt_page and src_page != tgt_page:
                edge_key = (src_page, tgt_page)

                if edge_key not in page_relationships:
                    page_relationships[edge_key] = {
                        "relationship_types": set(),
                        "weight": 0.0,
                        "file_pairs": [],
                    }

                # Aggregate relationship data
                rel_data = page_relationships[edge_key]
                rel_data["file_pairs"].append((src_file, tgt_file))
                rel_data["weight"] += edge_data.get("weight", 1.0)

                # Collect relationship types (import, dependency, etc.)
                rel_type = edge_data.get("type", "unknown")
                rel_data["relationship_types"].add(rel_type)

        # Add aggregated edges to page graph
        for (src_page, tgt_page), rel_data in page_relationships.items():
            page_graph.add_edge(
                src_page,
                tgt_page,
                weight=rel_data["weight"],
                relationship_types=list(rel_data["relationship_types"]),
                file_pair_count=len(rel_data["file_pairs"]),
            )

        logger.info(
            f"Built page graph: {page_graph.number_of_nodes()} pages, "
            f"{page_graph.number_of_edges()} cross-page relationships"
        )

        return page_graph

    async def _create_shard_from_segments(
        self,
        group_id: str,
        segments: list[ShardFileSegment],
        content: str,
        commit_hash: str,
        token_count: int = 0,
    ) -> RepositoryShard:
        """Create a shard from the given segments."""
        async with self.shard_semaphore:
            try:
                metadata = ShardMetadata(
                    shard_id=await self._create_shard_id(
                        group_id, segments, commit_hash
                    ),
                    file_segments=segments,
                    content_size_bytes=len(content),
                    token_count=token_count,
                    creation_timestamp=time.time(),
                    git_commit_hash=commit_hash,
                )

                # Update metrics
                self.metrics.shard_size.labels(repo_id=group_id).observe(metadata.content_size_bytes)

                return RepositoryShard(
                    metadata=metadata,
                    raw_content=content,
                    annotated_content=await self.prompt_strategy.get_shard_setup_prompt(
                        shard_type="Code",
                        shard_content=content,
                        shard_metadata=metadata.model_dump(),
                    ),
                )

            except Exception as e:
                logger.error(f"Error creating shard: {e!s}")
                self.metrics.processing_errors.labels(repo_id=group_id).inc()
                raise

    async def _create_shard_id(
        self, group_id: str, segments: list[ShardFileSegment], commit_hash: str
    ) -> str:
        """Create deterministic shard ID from content and metadata."""
        # Create canonical representation of segments
        segment_data = []
        for seg in segments:
            normalized_file_path = await get_polymathera().normalize_file_path(
                seg.file_path
            )
            segment_data.append(
                f"{normalized_file_path}:{seg.start_line}:{seg.end_line}"
            )
        segment_str = ",".join(sorted(segment_data))

        # Combine all identifying information
        id_components = [
            f"group={group_id}",
            f"commit={commit_hash}",
            f"segments={segment_str}",
        ]

        # Create hash
        id_string = ":".join(id_components)
        return hashlib.sha256(id_string.encode()).hexdigest()

    async def _process_file_group(
        self,
        group_id: str,
        file_paths: list[str],
        commit_hash: str,
        repo_path: Path,
    ) -> list[RepositoryShard]:
        """
        Process a group of related files together, respecting token limits.
        Uses CodeSplitter for intelligent splitting and TokenManager for size control.

        Reads file content from the filesystem directly (no git.Blob objects),
        so there is no dependency on a shared git subprocess.
        """
        try:
            # Process files in group
            async with self.file_semaphore:
                # Process files and collect segments
                file_contents: list[tuple[str, str, str]] = []
                binary_files: set[str] = set()

                logger.debug(
                    f"_process_file_group: files={len(file_paths)} "
                    f"group_id={group_id} repo_path={repo_path}"
                )

                for file_path in file_paths:
                    try:
                        fp = Path(file_path)
                        if not fp.exists() or not fp.is_file():
                            logger.warning(f"File not found: {file_path}, skipping")
                            continue

                        content: bytes = await asyncio.to_thread(fp.read_bytes)

                        if not content:
                            logger.debug(f"Empty content for {file_path}, skipping")
                            continue

                        mime_type: str = await asyncio.to_thread(
                            lambda c=content: magic.from_buffer(c[:1024], mime=True)
                        )

                        is_binary = self._is_binary_file(file_path, content, mime_type)

                        if is_binary:
                            self.metrics.binary_files.labels(repo_id=group_id).inc()
                            if self.config.skip_binary:
                                continue
                            if (
                                self.config.binary_size_limit_bytes
                                and len(content) > self.config.binary_size_limit_bytes
                            ):
                                continue
                            binary_files.add(file_path)
                        else:
                            text_content = self._decode_content(content)
                            if text_content is None:
                                logger.warning(
                                    f"Could not decode {file_path}, treating as binary"
                                )
                                binary_files.add(file_path)
                                continue

                            # Get token count
                            token_manager = await self.get_token_manager()
                            token_count = await token_manager.get_file_token_count(
                                file_path, group_id, commit_hash
                            )

                            if token_count > self.config.max_tokens_per_file:
                                logger.debug(
                                    f"File too large: {file_path} ({token_count} tokens)"
                                )
                                continue

                            file_contents.append((file_path, text_content, mime_type))

                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        continue

                logger.debug(
                    f"_process_file_group: file_contents={len(file_contents)} "
                    f"binary={len(binary_files)} total_files={len(file_paths)}"
                )

                if not file_contents:
                    return []

                # Split files into segments
                code_splitter = await self.get_code_splitter()
                segment_groups = await code_splitter.split_files_3(
                    file_contents, max_shard_size=self.config.max_tokens_per_shard
                )

                # Create shards from segments
                shards = []
                current_segments = []
                current_tokens = 0

                # Pre-compute line counts per file for segment token estimation
                file_line_counts: dict[str, int] = {}
                for fp, text, _ in file_contents:
                    file_line_counts[fp] = text.count("\n") + 1

                token_manager = await self.get_token_manager()

                # TODO: Use parallel processing
                for segment_group in segment_groups:
                    group_content = []

                    for segment in segment_group:
                        # Estimate segment tokens by scaling file tokens by line ratio
                        file_tokens = await token_manager.get_file_token_count(
                            segment.file_path, group_id, commit_hash
                        )
                        num_file_lines = file_line_counts.get(segment.file_path, 1)
                        segment_lines = max(1, segment.end_line - segment.start_line)
                        segment_tokens = max(1, file_tokens * segment_lines // num_file_lines)

                        if (
                            current_tokens + segment_tokens
                            > self.config.target_tokens_per_shard
                        ):
                            # Create shard from current segments
                            if current_segments:
                                shard = await self._create_shard_from_segments(
                                    group_id=group_id,
                                    segments=current_segments,
                                    content="\n".join(group_content),
                                    commit_hash=commit_hash,
                                    # relationship_score=relationship_score,
                                    # group_metadata=group_metadata,
                                    token_count=current_tokens,
                                )
                                shards.append(shard)

                            # Reset accumulators
                            current_segments = []
                            current_tokens = 0
                            group_content = []

                        # Add segment to current group
                        current_segments.append(segment)
                        current_tokens += segment_tokens
                        group_content.append(segment.content)

                # Handle remaining segments
                if current_segments:
                    shard = await self._create_shard_from_segments(
                        group_id=group_id,
                        segments=current_segments,
                        content="\n".join(group_content),
                        commit_hash=commit_hash,
                        # relationship_score=relationship_score,
                        # group_metadata=group_metadata,
                        token_count=current_tokens,
                    )
                    shards.append(shard)

                return shards

        except Exception as e:
            logger.error(f"Error processing file group: {e!s}", exc_info=True)
            self.metrics.processing_errors.labels(repo_id=group_id).inc()
            raise

    async def create_shards_incremental(
        self, group_id: str, repo: git.Repo, base_commit: str | None = None
    ) -> AsyncIterator[RepositoryShard]:
        """
        Create shards incrementally by processing only changed files.

        Args:
            group_id: Repository group identifier
            repo: GitPython repository object
            base_commit: Previous commit hash to compare against
        """
        try:
            current_commit = repo.head.commit

            if base_commit:
                # Get changed files since base commit
                diff = repo.git.diff(base_commit, current_commit.hexsha, name_only=True)
                changed_files = set(diff.split("\n")) if diff else set()

                # Get files from renamed/moved paths
                renames = repo.git.diff(
                    base_commit,
                    current_commit.hexsha,
                    diff_filter="R",
                    name_status=True,
                )
                if renames:
                    for line in renames.split("\n"):
                        if line.startswith("R"):
                            _, old_path, new_path = line.split("\t")
                            changed_files.add(old_path)
                            changed_files.add(new_path)

                # Add files from merge commits
                if len(current_commit.parents) > 1:
                    for parent in current_commit.parents:
                        merge_diff = repo.git.diff(
                            parent.hexsha, current_commit.hexsha, name_only=True
                        )
                        changed_files.update(merge_diff.split("\n"))

                # Process only changed files
                changed_blobs = [
                    blob
                    for blob in current_commit.tree.traverse()
                    if blob.type == "blob" and blob.path in changed_files
                ]

                # Reuse cached shards for unchanged files
                cached_shards = await self._get_cached_shards(
                    group_id, base_commit, changed_files
                )

                # Yield cached shards first
                for shard in cached_shards:
                    yield shard

                # Process changed files incrementally
                async for shard in self.incremental_manager.process_repo(repo):
                    # Update cache
                    await self._cache_shard(group_id, current_commit.hexsha, shard)
                    yield shard

            else:
                # No base commit - process entire repo
                async for shard in self.incremental_manager.process_repo(repo):
                    await self._cache_shard(group_id, current_commit.hexsha, shard)
                    yield shard

            # Update commit history
            self._update_commit_history(group_id, current_commit.hexsha)

        except Exception as e:
            logger.error(f"Incremental sharding failed: {e}", exc_info=True)
            raise ShardingError(f"Incremental sharding failed: {e!s}") from e

    async def _get_cached_shards(
        self, group_id: str, commit_hash: str, changed_files: set[str]
    ) -> list[RepositoryShard]:
        """Get cached shards that don't contain changed files"""
        cache_key = f"{group_id}:{commit_hash}"
        cached_shards = await self.shard_cache.get_shards(cache_key)
        if not cached_shards:
            return []

        return [
            shard
            for shard in cached_shards
            if not any(
                segment.file_path in changed_files
                for segment in shard.metadata.file_segments
            )
        ]

    async def _cache_shard(
        self, group_id: str, commit_hash: str, shard: RepositoryShard
    ):
        """Cache a shard with proper metadata"""
        cache_key = f"{group_id}:{commit_hash}"
        existing = await self.shard_cache.get_shards(cache_key) or []
        existing.append(shard)
        await self.shard_cache.set_shards(cache_key, existing)

    def _update_commit_history(self, group_id: str, commit_hash: str):
        """Track commit history for better incremental processing"""
        if group_id not in self.commit_history:
            self.commit_history[group_id] = []
        self.commit_history[group_id].append(commit_hash)

        # Keep only recent history
        self.commit_history[group_id] = self.commit_history[group_id][-100:]
