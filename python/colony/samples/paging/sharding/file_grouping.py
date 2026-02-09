from __future__ import annotations

import asyncio
import itertools
import json
import os
import time
from collections import defaultdict
from enum import IntFlag
from pathlib import Path
from statistics import mean
from typing import Any, ClassVar, Iterable
import mimetypes
import functools

import git
import networkx as nx
import numpy as np
from community import community_louvain as community
from pydantic import BaseModel, Field
try:
    import xxhash # Optional dependency
except ImportError:
    xxhash = None

from ...metrics.common import BaseMetricsMonitor
from ...caching.simple import CacheConfig
from ...config import ConfigComponent, register_polymathera_config
from ....distributed import get_polymathera
from .analyzers.base import FileContentCache
from .analyzers.dependency import DependencyAnalyzer
from .analyzers.history import CommitAnalysisConfig, CommitHistoryAnalyzer
from .analyzers.imports import ImportAnalyzer, ImportConfig
from .analyzers.semantic import SemanticAnalyzer, SemanticAnalyzerConfig
from .languages.dependency import DependencyConfig
from .languages.file_grouping import (
    STRONG_LANGUAGE_BINDINGS,
    LanguageConfig,
    LanguageFeature,
)
from .tokenization import TokenManager
from ....utils import setup_logger

logger = setup_logger(__name__)

# Import centralized language detection
from .languages.extensions import detect_language as _detect_language_centralized


# TODO: Implement the remaining analyzer classes?
# TODO: Implement the analyzer classes?
# TODO: Add more sophisticated dependency analysis?
# TODO: Add more sophisticated semantic analysis strategies?

# TODO: Add more sophisticated directory relationship scoring?
# TODO: Implement batch size auto-tuning?

# TODO: Enhance the error recovery mechanisms?
# TODO: The comprehensive error handling and recovery system?
# TODO: Add more detailed metrics for directory analysis?
# TODO: Each part will maintain production-quality code with comprehensive error handling, metrics, and configurability.


# TODO: The enhanced concurrency control system?
# TODO: Each part will build on this foundation while maintaining production-readiness and configurability.

# TODO: Implement more optimization strategies?
# TODO: Enhance the pattern matching?

# TODO: The remaining language-specific optimizations (interface-based, partial classes)?
# TODO: Add more language configurations?
# TODO: Add more language-specific optimizations?
# TODO: Add more sophisticated grouping algorithms?

# TODO: Implement language-specific embedding models?

# TODO: Add more cross-language optimization strategies?
# TODO: The enhanced dependency analysis that handles cross-language dependencies?
# TODO: The language-specific analyzers for different types of cross-language bindings?
# TODO: The optimization strategies that preserve cross-language relationships?
# TODO: The cross-language optimization implementation?
# TODO: Add cross-language analysis?
# TODO: Add more cross-language relationship patterns?

# TODO: Enhance the caching system?
# TODO: Enhance the caching system for cross-language analysis?
# TODO: Enhance the caching system for semantic analysis?

# TODO: Cross-Language Relationship Management
# 1. Add versioning for cross-language binding rules
# 2. Implement validation for cross-language relationships
# 3. Add metrics for cross-language binding changes
# 4. Track relationship strength over time

# TODO: Graph Metadata Management
# 1. Add schema versioning for graph metadata
# 2. Implement metadata validation
# 3. Add metadata compression
# 4. Track metadata size metrics

# TODO: Cache Consistency
# 1. Add consistency checks for graph metadata
# 2. Implement repair mechanisms for corrupted metadata
# 3. Add metadata recovery strategies
# 4. Track metadata consistency metrics
"""
A multi-stage optimization process that includes:
1. Cross-language optimizations
2. Language-specific optimizations
3. Token-based optimizations


1. Group Splitting:
    - Smart splitting of large groups
    - Token-aware splitting
    - Relationship preservation
    - Fallback mechanisms

2. Optimization:
    - Dynamic sizing
    - Token count management
    - Relationship score calculation
    - Edge type tracking

3. Error Handling:
    - Graceful degradation
    - Multiple fallback levels
    - Comprehensive error logging

4. Performance:
    - Concurrent token counting
    - Efficient graph operations
    - Caching integration

1. Asynchronous Processing:
    - Made directory relationship analysis fully async
    - Added batch processing for memory efficiency
    - Concurrent processing of file pairs within batches
2. Resource Management:
    - Added semaphore control
    - Batch size configuration
    - Memory usage optimization
3. Error Handling:
    - Granular error tracking
    - Detailed logging
    - Metrics for each stage
4. Performance:
    - Parallel processing of file pairs
    - Batched processing for large repos
    - Configurable batch sizes

5. Language-Specific Features:
    - Scope detection
    - Dependency tracking
    - Import analysis
    - Context preservation

6. Optimization Strategies:
    - Module-based grouping
    - Interface-based grouping
    - Partial class handling

7. Pattern Matching:
    - Compiled regex patterns
    - Language-specific patterns
    - Efficient matching
8. Feature Detection:
    - Language feature sets
    - Capability checking
    - Extension mapping

1. Cross-Language Awareness:
    - Considers language pairs in similarity thresholds
    - Tracks cross-language semantic bindings
    - Adjusts weights based on language relationships
2. Language-Specific Processing:
    - Includes language context in embeddings
    - Different thresholds for same/cross-language pairs
    - Special handling for known language combinations
3. Performance Optimizations:
    - Efficient batch processing
    - Caching of embeddings
    - Language-aware threshold adjustments
4. Enhanced Detection:
    - API implementation matching
    - Configuration file relationships
    - Cross-language test files
    - Documentation relationships
    - Generated code detection
"""

class FileGraphCacheMetricsMonitor(BaseMetricsMonitor):
    """Base class for Prometheus metrics monitoring using node-global HTTP server."""

    def __init__(self,
                 enable_http_server: bool = True,
                 service_name: str = "service"):
        super().__init__(enable_http_server, service_name)

        self.logger.info(f"Initializing FileGraphCacheMetricsMonitor instance {id(self)}...")

        self.serialization_time = self.create_histogram(
            "graph_serialization_seconds",
            "Time spent serializing graphs",
            labelnames=["operation"],
        )
        self.graph_size = self.create_histogram(
            "graph_size_bytes",
            "Size of serialized graphs",
            buckets=[1000, 10000, 100000, 1000000],
        )
        self.active_operations = self.create_gauge(
            "graph_cache_operations",
            "Number of active cache operations",
            labelnames=["operation"],
        )
        self.cache_hits = self.create_counter(
            "graph_cache_hits_total",
            "Number of cache hits",
            labelnames=["operation"],
        )
        self.cache_misses = self.create_counter(
            "graph_cache_misses_total",
            "Number of cache misses",
            labelnames=["operation"],
        )


class FileGraphCache:
    """Cache for file relationship graphs with versioning support"""

    def __init__(self, config: CacheConfig | None = None):
        self.config = config

        # Use existing TokenizedFileCache with "graphs" type
        self.cache = None
        self.metrics = FileGraphCacheMetricsMonitor()

    async def initialize(self):
        self.config = await CacheConfig.check_or_get_component(self.config)

        # Use existing TokenizedFileCache with "graphs" type
        self.cache = await get_polymathera().create_distributed_simple_cache(
            namespace="relationship_graphs",  # TODO: Does this need to be VMR-specific?
            config=self.config,
        )

    async def cleanup(self) -> None:
        """Cleanup background tasks and resources"""
        try:
            await self.cache.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up FileGraphCache: {e}")

    async def get(self, key: str, version: str | None = None) -> nx.DiGraph | None:
        """Get graph from cache"""
        try:
            self.metrics.active_operations.labels(operation="get").inc()

            # Get versioned key
            cache_key = self._make_version_key(key, version)

            # Get from cache
            data = await self.cache.get(cache_key)
            if not data:
                return None

            if self.config.serialization_format == "json":
                graph = nx.node_link_graph(data)

            self.metrics.cache_hits.labels(operation="get").inc()
            return graph

        except Exception as e:
            logger.error(f"Error getting graph from cache: {e}")
            return None
        finally:
            self.metrics.active_operations.labels(operation="get").dec()

    async def set(
        self, key: str, graph: nx.DiGraph, version: str | None = None
    ) -> bool:
        """Store graph in cache"""
        try:
            self.metrics.active_operations.labels(operation="put").inc()

            # Serialize and compress
            if self.config.serialization_format == "json":
                graph = nx.node_link_data(graph)

            # Store with version
            cache_key = self._make_version_key(key, version)
            return await self.cache.set(cache_key, graph)

        except Exception as e:
            logger.error(f"Error storing graph in cache: {e}")
            return False
        finally:
            self.metrics.active_operations.labels(operation="put").dec()

    def _make_version_key(self, key: str, version: str | None) -> str:
        """Create versioned cache key"""
        if not version:
            return f"graph:{key}"
        return f"graph:{key}:v{version}"


###############################################################################
# Grouping Strategies
###############################################################################


class FileGroupingStrategy(IntFlag):
    """Strategies for grouping related files using bitwise flags"""

    NONE = 0  # No grouping
    IMPORTS = 1 << 0  # Group by import relationships
    DEPENDENCIES = 1 << 1  # Group by function/class dependencies
    DIRECTORY = 1 << 2  # Group by directory proximity
    COMMIT_HISTORY = 1 << 3  # Group by commit patterns
    SEMANTIC = 1 << 4  # Group by semantic similarity
    HYBRID = IMPORTS | DEPENDENCIES | DIRECTORY | COMMIT_HISTORY | SEMANTIC  # Combine all strategies


class FileGroup(BaseModel):
    """Group of related files"""

    files: list[str] = Field(default_factory=list)
    relationship_score: float = Field(default=0.0)
    group_type: str = Field(default="")
    metadata: dict = Field(default_factory=dict)




@register_polymathera_config()
class FileGrouperConfig(ConfigComponent):
    """Configuration for file grouping"""

    strategies: FileGroupingStrategy = Field(
        default=FileGroupingStrategy.HYBRID
    )

    # Relationship weights (0-1)
    import_weight: float = 0.8
    dependency_weight: float = 0.7
    directory_weight: float = 0.5
    commit_weight: float = 0.3
    semantic_weight: float = 0.6

    # Thresholds
    min_relationship_score: float = 0.2
    max_group_size: int = 10
    max_group_tokens: int = 8192  # For token-aware grouping

    # Performance settings
    max_concurrent_analysis: int = 4
    analysis_timeout: int = 30

    # Language-specific settings
    language_configs: dict[str, LanguageConfig] = Field(default_factory=dict)

    # Advanced features
    enable_semantic_grouping: bool = False
    enable_dynamic_sizing: bool = True
    enable_cross_language: bool = True

    # Optimization settings
    min_group_size: int = 2
    edge_removal_strategy: str = "betweenness"  # or "weight"
    community_resolution: float = 1.0
    semantic_threshold: float = 0.7
    semantic_cross_lang_threshold: float = 0.6
    semantic_batch_size: int = 50
    cross_lang_weight_multiplier: float = 1.2

    # Cache settings
    file_graph_cache_config: CacheConfig | None = Field(
        default=None
    )  # If None, the default will be automatically loaded from config manager
    dependency_cache_config: CacheConfig | None = Field(
        default=None
    )  # If None, the default will be automatically loaded from config manager
    imports_cache_config: CacheConfig | None = Field(
        default=None
    )  # If None, the default will be automatically loaded from config manager
    dependency_config: DependencyConfig | None = Field(
        default=None
    )  # If None, the default will be automatically loaded from config manager
    import_config: ImportConfig | None = Field(
        default=None
    )  # If None, the default will be automatically loaded from config manager
    semantic_config: SemanticAnalyzerConfig | None = Field(
        default=None
    )  # If None, the default will be automatically loaded from config manager
    commit_config: CommitAnalysisConfig | None = Field(
        default=None
    )  # If None, the default will be automatically loaded from config manager

    CONFIG_PATH: ClassVar[str] = "llms.file_grouping.grouping"


# TODO: Performance Optimizations
# 1. Implement partial graph updates
# 2. Add graph pruning for old versions
# 3. Optimize serialization format
# 4. Add streaming graph updates

# TODO: Cache Management
# 1. Add cache warming strategies
# 2. Implement cache eviction policies
# 3. Add cache consistency checks
# 4. Implement cache replication

# TODO: Graph Updates
# 1. Add conflict resolution for concurrent updates
# 2. Implement rollback mechanism
# 3. Add validation for LLM updates
# 4. Track update history

# TODO: Monitoring
# 1. Add graph complexity metrics
# 2. Track relationship quality
# 3. Monitor update patterns
# 4. Add performance profiling

class FileGrouperMetricsMonitor(BaseMetricsMonitor):
    """Base class for Prometheus metrics monitoring using node-global HTTP server."""

    def __init__(self,
                 enable_http_server: bool = True,
                 service_name: str = "service"):
        super().__init__(enable_http_server, service_name)

        self.logger.info(f"Initializing FileGrouperMetricsMonitor instance {id(self)}...")

        self.analysis_duration = self.create_histogram(
            "file_grouping_analysis_seconds",
            "Time spent analyzing file relationships",
            labelnames=["strategy"],
        )
        self.grouping_duration = self.create_histogram(
            "file_grouping_grouping_seconds",
            "Time spent grouping files",
            labelnames=["strategy"],
        )
        self.group_sizes = self.create_histogram(
            "file_grouping_group_sizes",
            "Distribution of group sizes"
        )
        self.relationship_scores = self.create_histogram(
            "file_grouping_relationship_scores",
            "Distribution of relationship scores",
        )
        self.cache_hits = self.create_counter(
            "file_grouping_cache_hits_total",
            "Number of cache hits",
            labelnames=["cache_type"]
        )
        self.errors = self.create_counter(
            "file_grouping_errors_total",
            "Number of errors during grouping",
            ["error_type"],
        )
        self.active_analyzers = self.create_gauge(
            "file_grouping_active_analyzers",
            "Number of active analyzers",
            labelnames=["type"]
        )
        self.circular_deps_total = self.create_counter(
            "file_grouping_circular_deps_total",
            "Total number of circular dependencies detected"
        )
        self.circular_deps_severity = self.create_histogram(
            "file_grouping_circular_deps_severity",
            "Severity scores of circular dependencies"
        )



class FileGrouper:
    """Groups related files for optimal LLM context"""

    def __init__(
        self,
        token_manager: TokenManager | None = None,
        config: FileGrouperConfig | None = None,
        file_content_cache: FileContentCache | None = None,
    ):
        logger.info(f"________ FileGrouper: 0 {token_manager} {config} {file_content_cache}")
        self.config = config
        logger.info(f"________ FileGrouper: 1 {self.config}")
        self.file_content_cache = file_content_cache
        logger.info(f"________ FileGrouper: 2 {self.file_content_cache}")
        self.token_manager = token_manager
        self.dependency_analyzer = None
        self.import_analyzer = None
        self.commit_analyzer = None
        self.semantic_analyzer = None
        self.file_graph_cache = None
        # Initialize analyzer caches
        self._import_analyzers = {}
        self._dependency_analyzers = {}

        # Caches - defer creation until first use
        self.dependency_cache = None
        self.imports_cache = None
        # Concurrency control
        self.semaphore = None
        # Metrics
        self.metrics = FileGrouperMetricsMonitor()
        logger.info(f"________ FileGrouper: 9")

    async def initialize(self):
        self.config = await FileGrouperConfig.check_or_get_component(self.config)
        if self.file_content_cache is None:
            self.file_content_cache = FileContentCache()
            await self.file_content_cache.initialize()

        # Analyzers
        self.dependency_analyzer = DependencyAnalyzer(
            file_content_cache=self.file_content_cache,
            config=self.config.dependency_config
        )
        await self.dependency_analyzer.initialize()
        logger.info(f"________ FileGrouper: 3")
        self.import_analyzer = ImportAnalyzer(
            file_content_cache=self.file_content_cache,
            config=self.config.import_config
        )
        await self.import_analyzer.initialize()
        logger.info(f"________ FileGrouper: 4")
        self.commit_analyzer = CommitHistoryAnalyzer(
            file_content_cache=self.file_content_cache,
            config=self.config.commit_config
        )
        await self.commit_analyzer.initialize()
        logger.info(f"________ FileGrouper: 5")
        self.semantic_analyzer = (
            SemanticAnalyzer(
                file_content_cache=self.file_content_cache,
                config=self.config.semantic_config
            )
            if self.config.enable_semantic_grouping
            else None
        )
        await self.semantic_analyzer.initialize()
        logger.info(f"________ FileGrouper: 6")

        # Initialize graph cache
        self.file_graph_cache = FileGraphCache(
            config=self.config.file_graph_cache_config,
        )
        await self.file_graph_cache.initialize()
        logger.info(f"________ FileGrouper: 7")
        # Concurrency control
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_analysis)
        logger.info(f"________ FileGrouper: 8")

    async def _ensure_caches_initialized(self) -> None:
        """Initialize caches if not already done."""
        # TODO: Add cache namespaces
        if self.dependency_cache is None:
            self.dependency_cache = await get_polymathera().create_distributed_simple_cache(
                namespace="dependency_graphs",  # TODO: Does this need to be VMR-specific?
                config=self.config.dependency_cache_config,
            )
        if self.imports_cache is None:
            self.imports_cache = await get_polymathera().create_distributed_simple_cache(
                namespace="import_graphs",  # TODO: Does this need to be VMR-specific?
                config=self.config.imports_cache_config,
            )

    async def cleanup(self) -> None:
        """Cleanup background tasks and resources"""
        from ....utils import cleanup_dynamic_asyncio_tasks

        try:
            await cleanup_dynamic_asyncio_tasks(self, raise_exceptions=False)
            if self.dependency_cache:
                await self.dependency_cache.cleanup()
            if self.imports_cache:
                await self.imports_cache.cleanup()
            await self.file_graph_cache.cleanup()
            await self.dependency_analyzer.cleanup()
            await self.import_analyzer.cleanup()
            if self.semantic_analyzer:
                await self.semantic_analyzer.cleanup()
            await self.commit_analyzer.cleanup()
            for analyzer in self._dependency_analyzers.values():
                await analyzer.cleanup()
            for analyzer in self._import_analyzers.values():
                await analyzer.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up FileGrouper: {e}")

    @functools.lru_cache(maxsize=1000)
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension and content using centralized registry"""
        try:
            # Use centralized language detection
            language = _detect_language_centralized(file_path)
            if language:
                return language

            # Fallback to MIME type detection
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type:
                if 'text' in mime_type:
                    return 'text'
                elif 'application/json' in mime_type:
                    return 'json'
                elif 'application/xml' in mime_type:
                    return 'xml'

            # Default to unknown
            return 'unknown'

        except Exception as e:
            logger.error(f"Error detecting language for {file_path}: {e}")
            return 'unknown'

    def _get_dependency_analyzer(self, language: str) -> DependencyAnalyzer | None:
        """Get the appropriate dependency analyzer for a language"""
        try:
            if language not in self._dependency_analyzers:
                config = self.config.language_configs.get(language)
                if config and hasattr(config, 'dependency_config'):
                    self._dependency_analyzers[language] = DependencyAnalyzer(
                        file_content_cache=self.file_content_cache,
                        config=config.dependency_config
                    )
                else:
                    # Use default dependency analyzer
                    self._dependency_analyzers[language] = self.dependency_analyzer
            return self._dependency_analyzers.get(language)
        except Exception as e:
            logger.error(f"Error getting dependency analyzer for {language}: {e}", exc_info=True)
            return None

    def _get_import_analyzer(self, language: str) -> ImportAnalyzer | None:
        """Get the appropriate import analyzer for a language"""
        try:
            if language not in self._import_analyzers:
                config = self.config.language_configs.get(language)
                if config and hasattr(config, 'import_config'):
                    self._import_analyzers[language] = ImportAnalyzer(
                        file_content_cache=self.file_content_cache,
                        config=config.import_config
                    )
                else:
                    # Use default import analyzer
                    self._import_analyzers[language] = self.import_analyzer
            return self._import_analyzers.get(language)
        except Exception as e:
            logger.error(f"Error getting import analyzer: {e}", exc_info=True)
            return None

    async def update_graph_with_llm_insights(
        self,
        group_id: str,
        commit_hash: str,
        new_file_relationships: dict[tuple[str, str], dict[str, Any]],
    ) -> bool:
        """Update relationship graph with LLM inference results.
        This graph is used later to group files into inference shards which
        will hopefully improve inference quality and reduce query traffic.
        NOTE: WARNING: The initial graph structure is derived from different kinds of
        semantic analysis which may not align with LLM inferences. The first LLM
        query traffic pattern will be informative. But subsequent LLM query traffic
        pattens might be random and misleading leading to deterioration of inference
        performance.

        TODO: Correlate LLM inference-based relationships with the initial graph structure to
        identify which other relationship types have most predictive power when it comes to
        LLM inference query traffic. These relationships should be prioritized in the graph
        clustering process during the initial file grouping process.
        """
        try:
            # Ensure caches are initialized
            await self._ensure_caches_initialized()

            # Get current graph
            graph = await self.file_graph_cache.get(key=f"{group_id}:{commit_hash}")
            if not graph:
                return False

            # Apply updates
            for file_pair, relationship in new_file_relationships.items():
                source, target = file_pair
                source = await get_polymathera().normalize_file_path(source)
                target = await get_polymathera().normalize_file_path(target)
                if graph.has_edge(source, target):
                    edge_data = graph.get_edge_data(source, target)
                    edge_data["llm_weight"] = relationship.get("weight", 0.5)
                    edge_data["llm_type"] = relationship.get("type")
                    edge_data["llm_confidence"] = relationship.get("confidence")

            # Store updated graph with new version
            return await self.file_graph_cache.set(
                key=f"{group_id}:{commit_hash}",
                graph=graph,
                version=f"llm_{int(time.time())}",
            )

        except Exception as e:
            logger.error(f"Error updating graph with LLM insights: {e}")
            return False

    async def group_files(
        self,
        group_id: str,
        repo: git.Repo,
        files: list[str],
    ) -> list[FileGroup]:
        """
        Group files based on configured strategy, properly handling cross-language relationships.
        Caches and reuses relationship graphs.

        Args:
            group_id: Unique identifier for the repository and its VMR context
            repo: Git repository object
            files: List of file paths to group

        Returns:
            List of FileGroup objects representing related files

        Note:
            This implementation preserves cross-language relationships by:
            1. Building a single relationship graph for all files
            2. Applying language-specific analysis in parallel
            3. Preserving cross-language bindings (imports, dependencies)
            4. Using language-aware community detection
        """
        logger.info(f"________ group_files: [{group_id}] Starting group_files for {len(files)} files")
        try:
            # Ensure caches are initialized
            await self._ensure_caches_initialized()
            logger.info(f"________ group_files: [{group_id}] Caches initialized.")

            start_time = time.time()
            commit_hash = repo.head.commit.hexsha

            # Get language configs for all files
            file_languages = {file: self._detect_language(file) for file in files}
            # logger.info(f"________ group_files: file_languages={json.dumps(file_languages, indent=4)}")

            # Try to get cached graph
            logger.info(f"________ group_files: [{group_id}] Attempting to get cached graph for commit {commit_hash}.")
            graph = await self.file_graph_cache.get(
                key=f"{group_id}:{commit_hash}", version=self._get_graph_version(files)
            )
            logger.info(f"________ group_files: [{group_id}] Cached graph is {'found' if graph else 'not found'}.")

            # Track files with known cross-language bindings
            cross_lang_bindings = set()

            if graph is None:
                logger.info(f"________ group_files: [{group_id}] Building new graph.")
                graph = await self._build_graph(
                    repo, files, cross_lang_bindings
                )
                logger.info(f"________ group_files: [{group_id}] Finished building graph. Caching it now.")

                # Store cross-language bindings in graph metadata
                graph.graph["cross_lang_bindings"] = list(cross_lang_bindings)

                # Cache the constructed graph
                await self.file_graph_cache.set(
                    key=f"{group_id}:{commit_hash}",
                    graph=graph,
                    version=self._get_graph_version(files),
                )
                logger.info(f"________ group_files: [{group_id}] Finished caching graph.")
            else:
                # Reconstruct cross-language bindings from graph
                cross_lang_bindings = set(graph.graph.get("cross_lang_bindings", []))

                # Verify and update cross-language bindings from edges
                graph_edges: Iterable[tuple[str, str, dict[str, Any]]] = graph.edges(data=True)
                for source, target, data in graph_edges:
                    if data.get("is_cross_language"):
                        cross_lang_bindings.add(source)
                        cross_lang_bindings.add(target)
                    elif (
                        "source_language" in data
                        and "target_language" in data
                        and data["source_language"] != data["target_language"]
                    ):
                        cross_lang_bindings.add(source)
                        cross_lang_bindings.add(target)

            # Find initial groups using language-aware community detection
            logger.info(f"________ group_files: [{group_id}] Clustering files.")
            initial_groups = await self._cluster_files_with_languages(
                graph, cross_lang_bindings
            )
            logger.info(f"[{group_id}] Finished clustering. Found {len(initial_groups)} initial groups.")

            # Apply language-specific optimizations while preserving cross-language relationships
            logger.info(f"________ group_files: [{group_id}] Optimizing groups.")
            optimized_groups = await self._optimize_groups_with_languages(
                initial_groups, cross_lang_bindings
            )
            logger.info(f"[{group_id}] Finished optimizing groups. Found {len(optimized_groups)} final groups.")

            duration = time.time() - start_time
            self.metrics.grouping_duration.labels(strategy="grouping").observe(duration)

            logger.info(f"________ group_files: [{group_id}] group_files completed successfully in {duration:.2f} seconds.")
            return optimized_groups

        except Exception as e:
            logger.error(
                f"Error grouping files: {e}",
                exc_info=True,
                extra={
                    "repo": repo.working_dir,
                    "group_id": group_id,
                    "file_count": len(files),
                    "has_cached_graph": graph is not None,
                },
            )
            self.metrics.errors.labels(error_type="grouping").inc()
            return self._fallback_grouping(files)

    async def _build_graph(
        self,
        repo: git.Repo,
        files: list[str],
        cross_lang_bindings: set[str],
    ) -> nx.DiGraph:
        """Build a relationship graph for a list of files"""
        logger.info(f"_________build_graph: [{repo.working_dir}] _build_graph started for {len(files)} files.")
        # Initialize relationship graph
        graph = nx.DiGraph()

        # Concurrent analysis tasks
        logger.info(f"_________build_graph: [{repo.working_dir}] Starting concurrent analysis tasks.")
        async with asyncio.TaskGroup() as tg:
            tasks = []

            if self.config.strategies & FileGroupingStrategy.IMPORTS:
                logger.info(f"_________build_graph: [{repo.working_dir}] Creating imports analysis task.")
                # Analyze all imports, including cross-language ones
                tasks.append(
                    tg.create_task(
                        self._add_import_relationships(
                            graph, files, cross_lang_bindings
                        )
                    )
                )

            if self.config.strategies & FileGroupingStrategy.DEPENDENCIES:
                logger.info(f"_________build_graph: [{repo.working_dir}] Creating dependencies analysis task.")
                # Analyze dependencies across languages
                tasks.append(
                    tg.create_task(
                        self._add_dependency_relationships(
                            graph, files, cross_lang_bindings
                        )
                    )
                )

            if self.config.strategies & FileGroupingStrategy.COMMIT_HISTORY:
                logger.info(f"_________build_graph: [{repo.working_dir}] Creating commit history analysis task.")
                tasks.append(
                    tg.create_task(self._add_commit_relationships(graph, repo, files))
                )

            if self.config.strategies & FileGroupingStrategy.SEMANTIC:
                logger.info(f"_________build_graph: [{repo.working_dir}] Creating semantic analysis task.")
                tasks.append(
                    tg.create_task(
                        self._add_semantic_relationships(
                            graph, files, cross_lang_bindings
                        )
                    )
                )

            if self.config.strategies & FileGroupingStrategy.DIRECTORY:
                logger.info(f"_________build_graph: [{repo.working_dir}] Creating directory analysis task.")
                tasks.append(
                    tg.create_task(self._add_directory_relationships(graph, files))
                )

        logger.info(f"_________build_graph: [{repo.working_dir}] Finished concurrent analysis tasks.")

        # Detect circular dependencies before grouping
        cycles = self._detect_circular_dependencies(graph)
        if cycles:
            logger.warning(
                f"Detected {len(cycles)} circular dependencies",
                extra={"cycles": cycles},
            )
            # Optionally store cycles in graph metadata for later use
            graph.graph["circular_dependencies"] = cycles

        self._print_graph_summary(graph)
        return graph

    def _print_graph_summary(self, graph: nx.DiGraph):
        """Print graph summary"""
        # Print graph summary
        logger.info(f"________ _print_graph_summary: Graph summary - Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

        # Print graph structure details
        if graph.number_of_nodes() > 0:
            # Show some example nodes and their connections
            sample_nodes = list(graph.nodes())[:5]  # First 5 nodes
            logger.info(f"________ _print_graph_summary: Sample nodes: {json.dumps(sample_nodes, indent=4)}")

            # Show some example edges with weights
            sample_edges = list(graph.edges(data=True))[:5]  # First 5 edges
            logger.info(f"________ _print_graph_summary: Sample edges: {json.dumps(sample_edges, indent=4)}")

            # Show graph density and connectivity
            density = nx.density(graph)
            logger.info(f"________ _print_graph_summary: Graph density: {density:.4f}")

            # Show connected components
            components = list(nx.strongly_connected_components(graph))
            logger.info(f"________ _print_graph_summary: Strongly connected components: {len(components)}")
            if components:
                largest_component = max(components, key=len)
                logger.info(f"________ _print_graph_summary: Largest component size: {len(largest_component)}")

    def _get_graph_version(self, files: list[str]) -> str:
        """Generate version string for graph caching"""
        if xxhash is None:
            return "unknown"
        # Hash file paths and modification times
        paths = sorted(files)
        return xxhash.xxh64(json.dumps(paths).encode()).hexdigest()

    def _fallback_split(self, files: list[str]) -> list[FileGroup]:
        """Simple size-based splitting as ultimate fallback"""
        try:
            chunk_size = self.config.max_group_size
            groups = []
            for i in range(0, len(files), chunk_size):
                chunk = files[i:i + chunk_size]
                groups.append(
                    FileGroup(
                        files=chunk,
                        relationship_score=0.0,
                        group_type="fallback_split",
                        metadata={"split_reason": "size_limit"},
                    )
                )
            return groups
        except Exception as e:
            logger.error(f"Error in fallback split: {e}")
            # Ultimate fallback: single-file groups
            return [
                FileGroup(
                    files=[f],
                    relationship_score=0.0,
                    group_type="fallback_single",
                    metadata={},
                )
                for f in files
            ]

    async def _optimize_cross_language_group(self, group: FileGroup) -> list[FileGroup]:
        """Optimize groups containing cross-language relationships"""
        try:
            # Identify language clusters within the group
            lang_clusters = defaultdict(list)
            for file in group.files:
                lang = self._detect_language(file)
                lang_clusters[lang].append(file)

            # If only one language, treat as regular group
            if len(lang_clusters) == 1:
                return [group]

            # Create language-based subgroups while preserving strong relationships
            subgroups = []

            # For each language cluster, create a subgroup
            for language, files in lang_clusters.items():
                if len(files) <= self.config.max_group_size:
                    subgroups.append(
                        FileGroup(
                            files=files,
                            relationship_score=group.relationship_score * 0.9,  # Slight penalty for splitting
                            group_type="cross_language_optimized",
                            metadata={
                                "parent_group": id(group),
                                "language": language,
                                "cross_language": True,
                            },
                        )
                    )
                else:
                    # Split large language clusters
                    split_groups = self._fallback_split(files)
                    for sg in split_groups:
                        sg.group_type = "cross_language_split"
                        sg.metadata.update({
                            "parent_group": id(group),
                            "language": language,
                        })
                    subgroups.extend(split_groups)

            return subgroups

        except Exception as e:
            logger.error(f"Error optimizing cross-language group: {e}")
            return [group]

    async def _optimize_by_tokens(self, group: FileGroup) -> list[FileGroup]:
        """Optimize group based on token limits"""
        try:
            total_tokens = await self._get_group_tokens(group.files)

            if total_tokens <= self.config.max_group_tokens:
                return [group]

            # Split by tokens using the existing method
            return await self._split_by_tokens(group)

        except Exception as e:
            logger.error(f"Error optimizing by tokens: {e}")
            return [group]

    async def _optimize_partial_classes(
        self,
        group: FileGroup,
        lang_config: LanguageConfig
    ) -> list[FileGroup]:
        """Optimize groups for languages with partial classes (like C#)"""
        try:
            # For languages with partial classes, group files that define the same class
            class_groups = defaultdict(list)

            for file in group.files:
                content = await self.file_content_cache.read_file(file)
                if content:
                    # Simple heuristic: look for class definitions
                    lines = content.split('\n')
                    for line in lines:
                        if 'partial class' in line.lower():
                            # Extract class name (simplified)
                            parts = line.split()
                            if 'class' in parts:
                                class_idx = parts.index('class')
                                if class_idx + 1 < len(parts):
                                    class_name = parts[class_idx + 1]
                                    class_groups[class_name].append(file)
                                    break
                    else:
                        # No partial class found, add to default group
                        class_groups['_default'].append(file)
                else:
                    class_groups['_default'].append(file)

            # Create groups from class clusters
            result = []
            for class_name, files in class_groups.items():
                result.append(
                    FileGroup(
                        files=files,
                        relationship_score=group.relationship_score,
                        group_type="partial_class_optimized",
                        metadata={
                            "parent_group": id(group),
                            "class_name": class_name,
                        },
                    )
                )

            return result

        except Exception as e:
            logger.error(f"Error optimizing partial classes: {e}")
            return [group]

    async def _optimize_interface_based(
        self,
        group: FileGroup,
        lang_config: LanguageConfig
    ) -> list[FileGroup]:
        """Optimize groups for interface-based languages"""
        try:
            # Group files by interface implementations
            interface_groups = defaultdict(list)

            for file in group.files:
                content = await self.file_content_cache.read_file(file)
                if content:
                    # Simple heuristic: look for interface implementations
                    if 'implements' in content.lower() or 'interface' in content.lower():
                        # Extract interface names (simplified)
                        lines = content.split('\n')
                        for line in lines:
                            if 'implements' in line.lower():
                                interface_groups['_implementations'].append(file)
                                break
                            elif 'interface' in line.lower() and 'class' not in line.lower():
                                interface_groups['_interfaces'].append(file)
                                break
                        else:
                            interface_groups['_default'].append(file)
                    else:
                        interface_groups['_default'].append(file)
                else:
                    interface_groups['_default'].append(file)

            # Create groups
            result = []
            for group_type, files in interface_groups.items():
                if files:
                    result.append(
                        FileGroup(
                            files=files,
                            relationship_score=group.relationship_score,
                            group_type="interface_optimized",
                            metadata={
                                "parent_group": id(group),
                                "interface_type": group_type,
                            },
                        )
                    )

            return result

        except Exception as e:
            logger.error(f"Error optimizing interface-based group: {e}")
            return [group]

    async def _add_import_relationships(
        self,
        graph: nx.DiGraph,
        files: list[str],
        cross_lang_bindings: set[str],
    ) -> None:
        """Add import relationships to graph with enhanced analysis"""
        try:
            start_time = time.time()

            async def process_file(source_fpath: str):
                try:
                    async with self.semaphore:
                        source_language = self._detect_language(source_fpath)
                        if not source_language:
                            return

                        content = await self.file_content_cache.read_file(source_fpath)
                        if not content:
                            return

                        # Get import analysis with metadata
                        imports = await self._get_import_analyzer(source_language).analyze_file(
                            source_fpath, content, source_language
                        )
                        logger.info(f"________ _add_import_relationships for {source_fpath}: {json.dumps(imports, indent=4, default=list)}")

                        # Add edges with metadata
                        external_imports = defaultdict(set)
                        for import_type, paths in imports.items():
                            if import_type == "metadata":
                                continue
                            for target_fpath in paths:
                                if target_fpath not in files:
                                    external_imports[source_fpath].add(target_fpath)
                                    continue
                                # Check if this is a cross-language import
                                target_language = self._detect_language(target_fpath)
                                is_cross_lang = target_language and target_language != source_language

                                normalized_source_fpath = await get_polymathera().normalize_file_path(source_fpath)
                                normalized_target_fpath = await get_polymathera().normalize_file_path(target_fpath)

                                if is_cross_lang:
                                    cross_lang_bindings.add(normalized_source_fpath)
                                    cross_lang_bindings.add(normalized_target_fpath)

                                # Get metadata with safe defaults
                                metadata = imports["metadata"]

                                weight = self._calculate_import_weight(
                                    import_type,
                                    metadata["import_style"],
                                    metadata["is_optional"],
                                    metadata["is_conditional"],
                                    is_cross_lang,
                                    source_language,
                                    target_language or 'unknown',
                                )

                                graph.add_edge(
                                    normalized_source_fpath,
                                    normalized_target_fpath,
                                    weight=weight,
                                    relationship_type="import",
                                    metadata={
                                        **metadata,
                                        "is_cross_language": is_cross_lang,
                                        "source_language": source_language,
                                        "target_language": target_language,
                                    },
                                )

                        logger.info(f"________ _add_import_relationships External Imports for {source_fpath}:{json.dumps(external_imports, indent=4, default=list)}")

                except Exception as e:
                    logger.error(
                        f"Error processing imports for {source_fpath}: {e}", exc_info=True
                    )
                    self.metrics.errors.labels(error_type="import_analysis").inc()

            # Process files concurrently with semaphore control
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(process_file(source_fpath)) for source_fpath in files]

            # Record duration metric
            duration = time.time() - start_time
            self.metrics.analysis_duration.labels(strategy="imports").observe(duration)

        except Exception as e:
            logger.error(f"Error adding import relationships: {e}", exc_info=True)
            self.metrics.errors.labels(error_type="import_analysis").inc()

    def _calculate_import_weight(
        self,
        import_type: str,
        import_style: str | None,
        is_optional: bool,
        is_conditional: bool,
        is_cross_lang: bool,
        source_lang: str,
        target_lang: str,
    ) -> float:
        """Calculate weight for import relationship"""
        type_multipliers = {
            "direct": 1.0,
            "types": 0.8,
            "runtime": 0.9,
            "cross_language": 0.7,
        }
        weight = self.config.import_weight * type_multipliers.get(import_type, 0.5)

        # Apply modifiers
        if is_optional:
            weight *= 0.8
        if is_conditional:
            weight *= 0.9
        if is_cross_lang:
            lang_pair = frozenset([source_lang, target_lang])
            if lang_pair in STRONG_LANGUAGE_BINDINGS:
                weight *= 1.2  # Boost for common language pairs
            else:
                weight *= 0.8  # Reduce for unusual combinations
        if import_style == "wildcard":
            weight *= 0.6
        elif import_style == "aliased":
            weight *= 0.9

        return min(weight, 1.0)  # Ensure weight doesn't exceed 1.0

    async def _add_dependency_relationships(
        self,
        graph: nx.DiGraph,
        files: list[str],
        cross_lang_bindings: set[str],
    ):
        """Add edges based on dependencies, including cross-language dependencies"""
        try:
            async with self.semaphore:
                with self.metrics.analysis_duration.labels(strategy="dependencies").time():
                    for source_fpath in files:
                        source_language = self._detect_language(source_fpath)
                        dependency_analyzer = self._get_dependency_analyzer(source_language)
                        if not dependency_analyzer:
                            continue

                        # Check cache
                        cache_key = f"deps:{source_fpath}"
                        deps = await self.dependency_cache.get(cache_key)

                        if deps is None:
                            # Analyze dependencies including cross-language ones
                            deps = await dependency_analyzer.analyze_file(
                                source_fpath,
                                content=None, # Let it read the file. It will use the file content cache.
                                language=source_language,
                                cross_language=True
                            )
                            await self.dependency_cache.set(cache_key, deps)
                        else:
                            self.metrics.cache_hits.labels(cache_type="dependencies").inc()

                        # Add edges for each dependency
                        for target_fpath, dep_info in deps.items():
                            if target_fpath not in files:
                                logger.info(f"________ _add_dependency_relationships:\n\t{source_fpath} -> {target_fpath} is not in repo files")
                                continue
                            target_language = self._detect_language(target_fpath)
                            is_cross_lang = target_language and target_language != source_language

                            normalized_source_fpath = await get_polymathera().normalize_file_path(source_fpath)
                            normalized_target_fpath = await get_polymathera().normalize_file_path(target_fpath)

                            if is_cross_lang:
                                cross_lang_bindings.add(normalized_source_fpath)
                                cross_lang_bindings.add(normalized_target_fpath)

                            weight = self._calculate_dependency_weight(
                                dep_info, source_language, target_language
                            )

                            graph.add_edge(
                                normalized_source_fpath,
                                normalized_target_fpath,
                                weight=weight,
                                relationship_type="dependency",
                                metadata={
                                    **dep_info,
                                    "is_cross_language": is_cross_lang,
                                    "source_language": source_language,
                                    "target_language": target_language or 'unknown',
                                },
                            )

        except Exception as e:
            logger.error(
                f"Error analyzing dependencies: {e}",
                exc_info=True,
                extra={"files": len(files)},
            )
            self.metrics.errors.labels(error_type="dependencies").inc()

    async def _add_directory_relationships(self, graph: nx.DiGraph, files: list[str]):
        """Add edges based on directory proximity asynchronously"""
        try:
            async with self.semaphore:
                with self.metrics.analysis_duration.labels(strategy="directory").time():
                    self.metrics.active_analyzers.labels(type="directory").inc()

                    # Process files in batches to avoid memory issues with large repos
                    batch_size = 1000  # Configurable
                    for i in range(0, len(files), batch_size):
                        batch = files[i : i + batch_size]
                        paths = [Path(f) for f in batch]
                        common = Path(os.path.commonpath(paths))

                        # Process file pairs concurrently within batch
                        async with asyncio.TaskGroup() as batch_tg:
                            # for path1, path2 in itertools.combinations(paths, 2): # TODO: Quadratic complexity
                            for j in range(len(paths)):
                                for k in range(j + 1, len(paths)):
                                    batch_tg.create_task(
                                        self._process_directory_pair(
                                            graph, paths[j], paths[k], common
                                        )
                                    )

                    self.metrics.active_analyzers.labels(type="directory").dec()

        except Exception as e:
            logger.error(
                f"Error analyzing directories: {e}",
                exc_info=True,
                extra={"files": len(files)},
            )
            self.metrics.errors.labels(error_type="directory").inc()

    async def _process_directory_pair(
        self, graph: nx.DiGraph, path1: Path, path2: Path, common: Path
    ):
        """Process a pair of files for directory relationships"""
        try:
            # Calculate proximity score based on shared path components
            rel1 = path1.relative_to(common)
            rel2 = path2.relative_to(common)
            shared = len(Path(os.path.commonpath([str(rel1), str(rel2)])).parts)
            max_depth = max(len(rel1.parts), len(rel2.parts))

            if max_depth > 0:
                score = shared / max_depth * self.config.directory_weight
                if score >= self.config.min_relationship_score:
                    normalized_fpath1 = await get_polymathera().normalize_file_path(str(path1))
                    normalized_fpath2 = await get_polymathera().normalize_file_path(str(path2))
                    graph.add_edge(
                        normalized_fpath1,
                        normalized_fpath2,
                        weight=score,
                        relationship_type="directory",
                        metadata={},
                    )

        except Exception as e:
            logger.error(
                f"Error processing directory pair: {e}",
                exc_info=True,
                extra={"path1": str(path1), "path2": str(path2)},
            )

    async def _cluster_files(self, graph: nx.DiGraph) -> list[FileGroup]:
        """Find optimal file groupings using community detection"""
        try:
            # Convert directed graph to undirected for community detection
            undirected_graph = graph.to_undirected()

            # Find communities
            communities = community.best_partition(undirected_graph)

            # Group files by community
            groups = defaultdict(list)
            scores = defaultdict(float)

            for file_path, community_id in communities.items():
                groups[community_id].append(file_path)

                # Calculate average relationship score
                out_edges: Iterable[tuple[str, str, dict[str, Any]]] = graph.edges(file_path, data=True)
                if out_edges:
                    total_weight = sum(e[2]["weight"] for e in out_edges)
                    scores[community_id] += total_weight / len(out_edges)

            # Create FileGroup objects
            result = []
            for community_id, files in groups.items():
                # Skip groups that are too large
                if len(files) > self.config.max_group_size:
                    # Split into smaller groups
                    subgroups = self._split_large_group(files, graph)
                    result.extend(subgroups)
                else:
                    result.append(
                        FileGroup(
                            files=files,
                            relationship_score=scores[community_id] / len(files),
                            group_type="community",
                            metadata={
                                "community_id": community_id,
                                "edge_types": self._get_edge_types(graph, files),
                            },
                        )
                    )

            # Record metrics
            for group in result:
                self.metrics.group_sizes.observe(len(group.files))
                self.metrics.relationship_scores.observe(group.relationship_score)

            return result

        except Exception as e:
            logger.error(f"Error clustering files: {e}")
            self.metrics.errors.labels(error_type="clustering").inc()
            return self._fallback_grouping(list(graph.nodes()))

    def _split_large_group(
        self, files: list[str], graph: nx.DiGraph
    ) -> list[FileGroup]:
        """Split large groups while preserving strong relationships"""
        try:
            # Create subgraph for these files
            subgraph = graph.subgraph(files).copy()

            # Calculate edge betweenness centrality
            edge_centrality = nx.edge_betweenness_centrality(subgraph)

            # Remove edges with high centrality until we get suitable sized components
            while True:
                if not edge_centrality:
                    break

                # Remove edge with highest centrality
                edge = max(edge_centrality.items(), key=lambda x: x[1])[0]
                subgraph.remove_edge(*edge)

                # Check components
                components = list(nx.connected_components(subgraph))
                if all(len(c) <= self.config.max_group_size for c in components):
                    break

                # Recalculate centrality if needed
                edge_centrality = nx.edge_betweenness_centrality(subgraph)

            # Create groups from components
            return [
                FileGroup(
                    files=list(component),
                    relationship_score=self._calculate_group_score(component, graph),
                    group_type="split_component",
                    metadata={"parent_size": len(files), "split_reason": "size"},
                )
                for component in nx.connected_components(subgraph)
            ]

        except Exception as e:
            logger.error(f"Error splitting large group: {e}")
            # Fallback to simple size-based splitting
            return self._fallback_split(files)

    def _calculate_group_score(self, files: set[str], graph: nx.DiGraph) -> float:
        """Calculate average relationship score for a group of files - OPTIMIZED"""
        try:
            # Optimized: avoid quadratic complexity by only checking existing edges
            scores = []
            ### for file1, file2 in itertools.combinations(
            ###     files, 2
            ### ):  # TODO: Quadratic complexity
            ###     if graph.has_edge(file1, file2):
            ###         edge_data = graph.get_edge_data(file1, file2)
            ###         scores.append(edge_data["weight"])
            total_possible = len(files) * (len(files) - 1) // 2

            if total_possible == 0:
                return 0.0

            # Only iterate through actual edges in the subgraph
            subgraph = graph.subgraph(files)
            for _, _, data in subgraph.edges(data=True):
                scores.append(data.get("weight", 0.0))

            # Calculate density-adjusted score
            edge_density = len(scores) / total_possible if total_possible > 0 else 0
            avg_weight = mean(scores) if scores else 0.0

            # Combine average weight with edge density for better scoring
            return avg_weight * (0.7 + 0.3 * edge_density)

        except Exception as e:
            logger.error(f"Error calculating group score: {e}")
            return 0.0

    def _get_edge_types(self, graph: nx.DiGraph, files: list[str]) -> dict[str, int]:
        """Get distribution of relationship types in a group - OPTIMIZED"""
        try:
            type_counts = defaultdict(int)
            ### for file1, file2 in itertools.combinations(files, 2):
            ###     if graph.has_edge(file1, file2):
            ###         edge_type = graph.get_edge_data(file1, file2)["type"]
            ###         type_counts[edge_type] += 1
            # Optimized: only check actual edges in subgraph
            subgraph = graph.subgraph(files)
            for _, _, data in subgraph.edges(data=True):
                edge_type = data.get("relationship_type", "unknown")
                type_counts[edge_type] += 1

            return dict(type_counts)

        except Exception as e:
            logger.error(f"Error getting edge types: {e}")
            return {}

    async def _optimize_groups(self, groups: list[FileGroup]) -> list[FileGroup]:
        """Optimize groups based on token limits and relationship strength"""
        try:
            if not self.config.enable_dynamic_sizing:
                return groups

            optimized_groups = []
            for group in groups:
                # Check token count if token manager is available
                if self.token_manager:
                    total_tokens = await self._get_group_tokens(group.files)

                    if total_tokens > self.config.max_group_tokens:
                        # Split based on tokens
                        subgroups = await self._split_by_tokens(group)
                        optimized_groups.extend(subgroups)
                        continue

                # Add original group if no splitting needed
                optimized_groups.append(group)

            return optimized_groups

        except Exception as e:
            logger.error(f"Error optimizing groups: {e}")
            return groups

    async def _get_group_tokens(self, files: list[str]) -> int:
        """Get total token count for a group of files"""
        try:
            total = 0
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(self.token_manager.get_file_token_count(f)) for f in files
                ]

            for task in tasks:
                total += await task

            return total

        except Exception as e:
            logger.error(f"Error getting group tokens: {e}")
            return float("inf")  # Conservative estimate

    async def _split_by_tokens(self, group: FileGroup) -> list[FileGroup]:
        """Split a group to respect token limits"""
        try:
            # Get token counts for all files
            file_tokens = {}
            async with asyncio.TaskGroup() as tg:
                tasks = {
                    f: tg.create_task(self.token_manager.get_file_token_count(f))
                    for f in group.files
                }

            for file, task in tasks.items():
                file_tokens[file] = await task

            # Sort files by token count
            sorted_files = sorted(
                group.files, key=lambda f: file_tokens[f], reverse=True
            )

            # Create new groups
            new_groups = []
            current_group = []
            current_tokens = 0

            for file in sorted_files:
                file_size = file_tokens[file]

                if (
                    current_tokens + file_size > self.config.max_group_tokens
                    and current_group
                ):
                    # Start new group
                    new_groups.append(
                        FileGroup(
                            files=current_group.copy(),
                            relationship_score=group.relationship_score,
                            group_type="token_split",
                            metadata={
                                "parent_group": id(group),
                                "token_count": current_tokens,
                            },
                        )
                    )
                    current_group = []
                    current_tokens = 0

                current_group.append(file)
                current_tokens += file_size

            # Add remaining files
            if current_group:
                new_groups.append(
                    FileGroup(
                        files=current_group,
                        relationship_score=group.relationship_score,
                        group_type="token_split",
                        metadata={
                            "parent_group": id(group),
                            "token_count": current_tokens,
                        },
                    )
                )

            return new_groups

        except Exception as e:
            logger.error(f"Error splitting by tokens: {e}")
            return [group]  # Return original group on error

    def _fallback_grouping(self, files: list[str]) -> list[FileGroup]:
        """Simple directory-based grouping as fallback"""
        try:
            # Group by directory
            by_dir = defaultdict(list)
            for file in files:
                by_dir[str(Path(file).parent)].append(file)

            # Create groups
            groups = []
            for dir_path, dir_files in by_dir.items():
                if len(dir_files) > self.config.max_group_size:
                    # Split large directories
                    chunk_size = self.config.max_group_size
                    for i in range(0, len(dir_files), chunk_size):
                        chunk = dir_files[i : i + chunk_size]
                        groups.append(
                            FileGroup(
                                files=chunk,
                                relationship_score=self.config.directory_weight,
                                group_type="fallback_directory",
                                metadata={"directory": dir_path},
                            )
                        )
                else:
                    groups.append(
                        FileGroup(
                            files=dir_files,
                            relationship_score=self.config.directory_weight,
                            group_type="fallback_directory",
                            metadata={"directory": dir_path},
                        )
                    )

            return groups

        except Exception as e:
            logger.error(f"Error in fallback grouping: {e}")
            # Ultimate fallback: single-file groups
            return [
                FileGroup(
                    files=[f],
                    relationship_score=0.0,
                    group_type="fallback_single",
                    metadata={},
                )
                for f in files
            ]

    async def _add_semantic_relationships(
        self,
        graph: nx.DiGraph,
        files: list[str],
        cross_lang_bindings: set[str],
    ):
        """
        Add edges based on semantic similarity, considering cross-language relationships.

        This is particularly useful for detecting:
        - Matching API implementations across languages
        - Related configuration files in different formats
        - Corresponding test files across language boundaries
        - Documentation in different languages
        - Generated code relationships
        """
        try:
            if not self.semantic_analyzer:
                return

            async with self.semaphore:
                with self.metrics.analysis_duration.labels(strategy="semantic").time():
                    self.metrics.active_analyzers.labels(type="semantic").inc()

                    # Process files in batches for efficiency
                    batch_size = self.config.semantic_batch_size
                    for i in range(0, len(files), batch_size):
                        batch_files = files[i : i + batch_size]

                        # Calculate similarities considering language pairs
                        pairwise_similarities = await self.semantic_analyzer.get_similarity_matrix(
                            batch_files,
                            [self._detect_language(file) for file in batch_files],
                            batch_size,
                        )
                        for (file1, file2), similarity in pairwise_similarities.items():
                            threshold = self._get_semantic_threshold(
                                self._detect_language(file1),
                                self._detect_language(file2)
                            )
                            if similarity < threshold:
                                continue

                            lang1 = self._detect_language(file1)
                            lang2 = self._detect_language(file2)
                            is_cross_lang = lang1 != lang2
                            normalized_file1 = await get_polymathera().normalize_file_path(file1)
                            normalized_file2 = await get_polymathera().normalize_file_path(file2)

                            # Track strong cross-language semantic relationships
                            if is_cross_lang and similarity >= self.config.semantic_cross_lang_threshold:
                                cross_lang_bindings.add(normalized_file1)
                                cross_lang_bindings.add(normalized_file2)

                            graph.add_edge(
                                normalized_file1,
                                normalized_file2,
                                weight=self._calculate_semantic_weight(lang1, lang2),
                                relationship_type="semantic",
                                metadata={
                                    "is_cross_language": is_cross_lang,
                                    "source_language": lang1,
                                    "target_language": lang2 or 'unknown',
                                },
                            )

                    self.metrics.active_analyzers.labels(type="semantic").dec()

        except Exception as e:
            logger.error(
                f"Error in semantic analysis: {e}",
                exc_info=True,
                extra={"files": len(files)},
            )
            self.metrics.errors.labels(error_type="semantic").inc()

    def _get_semantic_threshold(self, lang1: str, lang2: str) -> float:
        """
        Get semantic similarity threshold based on language pair.
        Different thresholds for same-language vs cross-language comparisons.
        """
        if lang1 == lang2:
            return self.config.semantic_threshold

        # Use lower threshold for known language pairs
        lang_pair = frozenset([lang1, lang2])
        if lang_pair in STRONG_LANGUAGE_BINDINGS:
            return self.config.semantic_cross_lang_threshold

        # Higher threshold for unusual language combinations
        return self.config.semantic_cross_lang_threshold * 1.2

    def _calculate_semantic_weight(self, lang1: str, lang2: str) -> float:
        """Calculate relationship weight for semantic similarity based on languages"""
        base_weight = self.config.semantic_weight

        # Adjust weight for cross-language relationships
        if lang1 != lang2:
            lang_pair = frozenset([lang1, lang2])
            if lang_pair in STRONG_LANGUAGE_BINDINGS:
                # Boost weight for common language pairs
                base_weight *= self.config.cross_lang_weight_multiplier
            else:
                # Reduce weight for unusual combinations
                base_weight *= 0.8

        return min(base_weight, 1.0)

    async def _add_commit_relationships(
        self, graph: nx.DiGraph, repo: git.Repo, files: list[str]
    ):
        """Add edges based on commit history patterns"""
        try:
            async with self.semaphore:
                with self.metrics.analysis_duration.labels(strategy="commits").time():
                    self.metrics.active_analyzers.labels(type="commits").inc()

                    # Analyze commit patterns
                    commit_patterns = await self.commit_analyzer.analyze_repo(
                        repo, files
                    )

                    # Add edges for co-committed files
                    for file1, file2, score in commit_patterns:
                        if score >= self.config.min_relationship_score:
                            normalized_file1 = await get_polymathera().normalize_file_path(file1)
                            normalized_file2 = await get_polymathera().normalize_file_path(file2)
                            graph.add_edge(
                                normalized_file1,
                                normalized_file2,
                                weight=score * self.config.commit_weight,
                                relationship_type="commit",
                                metadata={},
                            )

                    self.metrics.active_analyzers.labels(type="commits").dec()

        except Exception as e:
            logger.error(f"Error analyzing commit history: {e}")
            self.metrics.errors.labels(error_type="commits").inc()

    def _group_by_language(self, files: list[str]) -> dict[str, list[str]]:
        """Group files by programming language"""
        groups = defaultdict(list)
        for file in files:
            ext = Path(file).suffix.lower()
            if ext:  # Skip files without extension
                groups[ext].append(file)
        return dict(groups)

    async def _cluster_files_with_languages(
        self,
        graph: nx.DiGraph,
        cross_lang_bindings: set[str],
    ) -> list[FileGroup]:
        """
        Cluster files using language-aware community detection.
        Preserves cross-language relationships while optimizing for language-specific patterns.
        """
        try:
            # Adjust edge weights based on language relationships
            graph_copy = graph.copy()
            graph_edges: Iterable[tuple[str, str, dict[str, Any]]] = graph_copy.edges(data=True)
            for file1, file2, data in graph_edges:
                if file1 in cross_lang_bindings and file2 in cross_lang_bindings:
                    # Strengthen edges between known cross-language bindings
                    data["weight"] *= self.config.cross_lang_weight_multiplier

            # Convert directed graph to undirected for community detection
            undirected_graph = graph_copy.to_undirected()

            # Use language-aware community detection
            communities = community.best_partition(
                undirected_graph,
                resolution=self.config.community_resolution,
                random_state=42,  # For reproducibility
            )
            logger.info(f"________ _cluster_files_with_languages: communities={json.dumps(communities, indent=4)}")

            # Group files by community
            groups = defaultdict(list)
            scores = defaultdict(float)
            languages = defaultdict(set)

            for normalized_fpath, community_id in communities.items():
                fpath = await get_polymathera().denormalize_file_path(normalized_fpath)
                groups[community_id].append(fpath)
                languages[community_id].add(self._detect_language(fpath))

                # Calculate average relationship score
                out_edges: Iterable[tuple[str, str, dict[str, Any]]] = graph.edges(normalized_fpath, data=True)
                if out_edges:
                    total_weight = sum(e[2]["weight"] for e in out_edges)
                    scores[community_id] += total_weight / len(out_edges)

            # Create FileGroup objects with language metadata
            result = []
            for community_id, files in groups.items():
                result.append(
                    FileGroup(
                        files=files,
                        relationship_score=scores[community_id] / len(files),
                        group_type="language_aware_community",
                        metadata={
                            "community_id": community_id,
                            "languages": list(languages[community_id]),
                            "cross_language_bindings": len(
                                set(files) & cross_lang_bindings
                            ),
                            "edge_types": self._get_edge_types(graph, files),
                        },
                    )
                )
                logger.info(f"________ _cluster_files_with_languages: file group={result[-1].model_dump_json(indent=4)}")

            return result

        except Exception as e:
            logger.error(f"Error in language-aware clustering: {e}", exc_info=True)
            return self._fallback_grouping(list(graph.nodes()))

    async def _optimize_groups_with_languages(
        self,
        groups: list[FileGroup],
        cross_lang_bindings: set[str],
    ) -> list[FileGroup]:
        """
        Optimize groups while preserving cross-language relationships.
        Applies both language-specific and token-based optimizations.
        """
        try:
            optimized_groups = []

            for group in groups:
                # Get languages in this group
                group_languages = {self._detect_language(f) for f in group.files}

                # Check if group contains cross-language bindings
                has_cross_lang = bool(set(group.files) & cross_lang_bindings)

                if has_cross_lang:
                    # Apply cross-language specific optimizations
                    subgroups = await self._optimize_cross_language_group(group)
                else:
                    # Apply language-specific optimizations for single-language groups
                    if len(group_languages) == 1:
                        language = next(iter(group_languages))
                        lang_config = self.config.language_configs.get(language)
                        if lang_config:
                            subgroups = await self._optimize_language_specific(
                                group, lang_config
                            )
                        else:
                            subgroups = [group]
                    else:
                        subgroups = [group]

                # Apply token-based optimization if needed
                if self.token_manager and self.config.enable_dynamic_sizing:
                    for subgroup in subgroups:
                        token_groups = await self._optimize_by_tokens(subgroup)
                        optimized_groups.extend(token_groups)
                else:
                    optimized_groups.extend(subgroups)

            return optimized_groups

        except Exception as e:
            logger.error(f"Error optimizing groups: {e}", exc_info=True)
            return groups

    def _detect_circular_dependencies(self, graph: nx.DiGraph) -> list[list[str]]:
        """Detect and analyze circular dependencies"""
        try:
            # Find all cycles in the graph
            cycles: list[list[str]] = list(nx.simple_cycles(graph))

            # Filter and categorize cycles
            analyzed_cycles: list[dict[str, Any]] = []
            for cycle in cycles:
                cycle_info = {
                    "files": cycle,
                    "types": set(),  # relationship types in cycle
                    "languages": set(),  # languages involved
                    "severity": self._calculate_cycle_severity(graph, cycle),
                }

                # Analyze edges in cycle
                for i in range(len(cycle)):
                    src = cycle[i]
                    dst = cycle[(i + 1) % len(cycle)]
                    edge_data: dict[str, Any] = graph.edges[src, dst]

                    cycle_info["types"].add(edge_data["relationship_type"])
                    cycle_info["languages"].add(self._detect_language(src))

                analyzed_cycles.append(cycle_info)

            # Record metrics
            self.metrics.circular_deps_total.inc(len(analyzed_cycles))
            for cycle in analyzed_cycles:
                self.metrics.circular_deps_severity.observe(cycle["severity"])

            return analyzed_cycles

        except Exception as e:
            logger.error(f"Error detecting circular dependencies: {e}", exc_info=True)
            return []

    def _calculate_cycle_severity(self, graph: nx.DiGraph, cycle: list[str]) -> float:
        """Calculate severity of a circular dependency"""
        try:
            severity_factors = {
                "size": len(cycle) / 10,  # Larger cycles are worse
                "weight": sum(
                    graph.edges[cycle[i], cycle[(i + 1) % len(cycle)]]["weight"]
                    for i in range(len(cycle))
                )
                / len(cycle),
                "cross_language": len(set(self._detect_language(f) for f in cycle)) > 1,
                "relationship_types": len(
                    set(
                        graph.edges[cycle[i], cycle[(i + 1) % len(cycle)]][
                            "relationship_type"
                        ]
                        for i in range(len(cycle))
                    )
                ),
            }

            # Calculate weighted severity score
            return (
                severity_factors["size"] * 0.3
                + severity_factors["weight"] * 0.3
                + severity_factors["cross_language"] * 0.2
                + (severity_factors["relationship_types"] / 4) * 0.2
            )

        except Exception as e:
            logger.error(f"Error calculating cycle severity: {e}", exc_info=True)
            return 0.0

    async def _analyze_language_specific(
        self, file_path: str, content: str, lang_config: LanguageConfig
    ) -> dict[str, Any]:
        """Perform language-specific analysis"""
        try:
            results: dict[str, Any] = {
                "scopes": [],
                "dependencies": set(),
                "imports": set(),
                "contexts": {},
            }

            lines = content.splitlines()
            current_scope = None
            scope_stack: list[dict[str, Any]] = []

            for i, line in enumerate(lines):
                # Check for scope starts
                for scope_type, pattern in lang_config.scope_patterns.items():
                    if pattern.match(line):
                        scope = {
                            "type": scope_type,
                            "start": i,
                            "content": line,
                            "parent": current_scope,
                        }
                        scope_stack.append(scope)
                        current_scope = scope
                        break

                # Check for scope ends
                if line.strip().endswith("}") or line.strip() == "end":
                    if scope_stack:
                        scope = scope_stack.pop()
                        scope["end"] = i
                        results["scopes"].append(scope)
                        current_scope = scope_stack[-1] if scope_stack else None

                # Check for dependencies
                for dep_type, pattern in lang_config.dependency_patterns.items():
                    for match in pattern.finditer(line):
                        results["dependencies"].add((dep_type, match.group(1), i))

                # Check for imports
                for pattern in lang_config.import_patterns:
                    if pattern.match(line):
                        results["imports"].add(line.strip())

                # Check for context patterns
                for pattern, context_lines in lang_config.context_patterns.items():
                    if pattern.match(line):
                        results["contexts"][i] = context_lines

            return results

        except Exception as e:
            logger.error(f"Error in language-specific analysis: {e}")
            return {}

    async def _optimize_language_specific(
        self, group: FileGroup, lang_config: LanguageConfig
    ) -> list[FileGroup]:
        """Apply language-specific optimizations to a group"""
        try:
            # Check for language-specific features
            if LanguageFeature.MODULES in lang_config.features:
                return await self._optimize_module_based(group, lang_config)
            elif LanguageFeature.PARTIAL_CLASSES in lang_config.features:
                return await self._optimize_partial_classes(group, lang_config)
            elif LanguageFeature.INTERFACES in lang_config.features:
                return await self._optimize_interface_based(group, lang_config)

            return [group]

        except Exception as e:
            logger.error(f"Error in language-specific optimization: {e}")
            return [group]

    async def _optimize_module_based(
        self, group: FileGroup, lang_config: LanguageConfig
    ) -> list[FileGroup]:
        """Optimize groups based on module relationships"""
        try:
            # Analyze module dependencies
            module_deps = {}
            for file in group.files:
                analysis = await self._analyze_language_specific(
                    file, await self.file_content_cache.read_file(file), lang_config
                )
                module_deps[file] = analysis.get("imports", set())

            # Build module graph
            graph = nx.DiGraph()
            for file, imports in module_deps.items():
                normalized_file = await get_polymathera().normalize_file_path(file)
                graph.add_node(normalized_file)
                for imp in imports:
                    if imp in group.files:
                        normalized_imp = await get_polymathera().normalize_file_path(imp)
                        graph.add_edge(
                            normalized_file,
                            normalized_imp,
                            weight=1.0,
                            relationship_type="import",
                            metadata={},
                        )

            # Find strongly connected components
            components = list(nx.strongly_connected_components(graph))

            # Create groups from components
            return [
                FileGroup(
                    files=list(component),
                    relationship_score=group.relationship_score,
                    group_type="module_based",
                    metadata={
                        "parent_group": id(group),
                        "module_deps": len(nx.edges(graph.subgraph(component))),
                    },
                )
                for component in components
            ]

        except Exception as e:
            logger.error(f"Error in module-based optimization: {e}")
            return [group]


