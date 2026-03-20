from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict
from enum import IntFlag
from pathlib import Path
from statistics import mean
from typing import Any, ClassVar, Iterable
import mimetypes

import git
import networkx as nx
import numpy as np
from community import community_louvain as community
from pydantic import BaseModel, Field
try:
    import xxhash # Optional dependency
except ImportError:
    xxhash = None

from polymathera.colony.distributed.metrics.common import BaseMetricsMonitor
from polymathera.colony.distributed.caching.simple import CacheConfig
from polymathera.colony.distributed.config import ConfigComponent, register_polymathera_config
from polymathera.colony.distributed import get_initialized_polymathera
from polymathera.colony.utils import setup_logger

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

logger = setup_logger(__name__)

# Import centralized language detection
from .languages.extensions import detect_language as _detect_language_centralized


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
        # FileGraphCache serializes via nx.node_link_data() which produces
        # JSON-compatible dicts.  Override the default pickle format.
        if self.config.serialization_format != "json":
            self.config = self.config.model_copy(update={"serialization_format": "json"})

        # Use existing TokenizedFileCache with "graphs" type
        polymathera = await get_initialized_polymathera()
        self.cache = await polymathera.create_distributed_simple_cache(
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
            else:
                raise ValueError(f"Unsupported serialization format: {self.config.serialization_format}")

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

            if self.config.serialization_format == "json":
                data = nx.node_link_data(graph)
            else:
                raise ValueError(f"Unsupported serialization format: {self.config.serialization_format}")

            cache_key = self._make_version_key(key, version)
            return await self.cache.set(cache_key, data)

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
        self.config: FileGrouperConfig | None = config
        self.file_content_cache = file_content_cache
        self.token_manager = token_manager
        self.default_dependency_analyzer = None
        self.default_import_analyzer = None
        self.commit_analyzer = None
        self.semantic_analyzer = None
        self.file_graph_cache = None
        self._language_cache: dict[str, str] = {}
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

    async def initialize(self):
        self.config = await FileGrouperConfig.check_or_get_component(self.config)
        if self.file_content_cache is None:
            self.file_content_cache = FileContentCache()
            await self.file_content_cache.initialize()

        # Analyzers
        self.default_dependency_analyzer = DependencyAnalyzer(
            file_content_cache=self.file_content_cache,
            config=self.config.dependency_config
        )
        await self.default_dependency_analyzer.initialize()
        self.default_import_analyzer = ImportAnalyzer(
            file_content_cache=self.file_content_cache,
            config=self.config.import_config
        )
        await self.default_import_analyzer.initialize()
        self.commit_analyzer = CommitHistoryAnalyzer(
            file_content_cache=self.file_content_cache,
            config=self.config.commit_config
        )
        await self.commit_analyzer.initialize()
        if self.config.enable_semantic_grouping:
            self.semantic_analyzer = SemanticAnalyzer(
                file_content_cache=self.file_content_cache,
                config=self.config.semantic_config,
            )
            await self.semantic_analyzer.initialize()

        # Initialize graph cache
        self.file_graph_cache = FileGraphCache(
            config=self.config.file_graph_cache_config,
        )
        await self.file_graph_cache.initialize()
        # Concurrency control
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_analysis)

    async def _ensure_caches_initialized(self) -> None:
        """Initialize caches if not already done."""
        polymathera = await get_initialized_polymathera()
        # TODO: Add cache namespaces
        if self.dependency_cache is None:
            self.dependency_cache = await polymathera.create_distributed_simple_cache(
                namespace="dependency_graphs",  # TODO: Does this need to be VMR-specific?
                config=self.config.dependency_cache_config,
            )
        if self.imports_cache is None:
            self.imports_cache = await polymathera.create_distributed_simple_cache(
                namespace="import_graphs",  # TODO: Does this need to be VMR-specific?
                config=self.config.imports_cache_config,
            )

    async def cleanup(self) -> None:
        """Cleanup background tasks and resources"""
        from polymathera.colony.utils import cleanup_dynamic_asyncio_tasks

        try:
            await cleanup_dynamic_asyncio_tasks(self, raise_exceptions=False)
            if self.dependency_cache:
                await self.dependency_cache.cleanup()
            if self.imports_cache:
                await self.imports_cache.cleanup()
            await self.file_graph_cache.cleanup()
            await self.default_dependency_analyzer.cleanup()
            await self.default_import_analyzer.cleanup()
            if self.semantic_analyzer:
                await self.semantic_analyzer.cleanup()
            await self.commit_analyzer.cleanup()
            for analyzer in self._dependency_analyzers.values():
                await analyzer.cleanup()
            for analyzer in self._import_analyzers.values():
                await analyzer.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up FileGrouper: {e}")

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension using centralized registry."""
        cached = self._language_cache.get(file_path)
        if cached is not None:
            return cached

        try:
            language = _detect_language_centralized(file_path)
            if language:
                self._language_cache[file_path] = language
                return language

            # Fallback to MIME type detection
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type:
                if 'text' in mime_type:
                    result = 'text'
                elif 'application/json' in mime_type:
                    result = 'json'
                elif 'application/xml' in mime_type:
                    result = 'xml'
                else:
                    result = 'unknown'
                self._language_cache[file_path] = result
                return result

            self._language_cache[file_path] = 'unknown'
            return 'unknown'

        except Exception as e:
            logger.error(f"Error detecting language for {file_path}: {e}")
            return 'unknown'

    async def _get_dependency_analyzer(self, language: str) -> DependencyAnalyzer | None:
        """Get the appropriate dependency analyzer for a language"""
        try:
            if language not in self._dependency_analyzers:
                config = self.config.language_configs.get(language)
                if config and hasattr(config, 'dependency_config'):
                    self._dependency_analyzers[language] = DependencyAnalyzer(
                        file_content_cache=self.file_content_cache,
                        config=config.dependency_config
                    )
                    await self._dependency_analyzers[language].initialize()
                else:
                    # Use default dependency analyzer
                    self._dependency_analyzers[language] = self.default_dependency_analyzer
            return self._dependency_analyzers.get(language)
        except Exception as e:
            logger.error(f"Error getting dependency analyzer for {language}: {e}", exc_info=True)
            return None

    async def _get_import_analyzer(self, language: str) -> ImportAnalyzer | None:
        """Get the appropriate import analyzer for a language"""
        try:
            if language not in self._import_analyzers:
                config = self.config.language_configs.get(language)
                if config and hasattr(config, 'import_config'):
                    self._import_analyzers[language] = ImportAnalyzer(
                        file_content_cache=self.file_content_cache,
                        config=config.import_config
                    )
                    await self._import_analyzers[language].initialize()
                else:
                    # Use default import analyzer
                    self._import_analyzers[language] = self.default_import_analyzer
            return self._import_analyzers.get(language)
        except Exception as e:
            logger.error(f"Error getting import analyzer: {e}", exc_info=True)
            return None

    async def update_graph_with_llm_insights(
        self,
        colony_id: str,
        tenant_id: str,
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
            graph = await self.file_graph_cache.get(key=f"{tenant_id}:{colony_id}:{commit_hash}")
            if not graph:
                return False

            polymathera = await get_initialized_polymathera()

            # Apply updates
            for file_pair, relationship in new_file_relationships.items():
                source, target = file_pair
                source = await polymathera.normalize_file_path(source)
                target = await polymathera.normalize_file_path(target)
                if graph.has_edge(source, target):
                    edge_data = graph.get_edge_data(source, target)
                    edge_data["llm_weight"] = relationship.get("weight", 0.5)
                    edge_data["llm_type"] = relationship.get("type")
                    edge_data["llm_confidence"] = relationship.get("confidence")

            # Store updated graph with new version
            return await self.file_graph_cache.set(
                key=f"{tenant_id}:{colony_id}:{commit_hash}",
                graph=graph,
                version=f"llm_{int(time.time())}",
            )

        except Exception as e:
            logger.error(f"Error updating graph with LLM insights: {e}")
            return False

    async def group_files(
        self,
        colony_id: str,
        tenant_id: str,
        repo: git.Repo,
        files: list[str],
    ) -> list[FileGroup]:
        """
        Group files based on configured strategy, properly handling cross-language relationships.
        Caches and reuses relationship graphs.

        Args:
            colony_id: Unique identifier for the repository and its VMR context
            tenant_id: Tenant ID
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
        graph = None
        try:
            logger.info(f"[{tenant_id}:{colony_id}] group_files: starting for {len(files)} files")
            await self._ensure_caches_initialized()
            logger.info(f"[{tenant_id}:{colony_id}] group_files: caches initialized")

            start_time = time.time()
            commit_hash = repo.head.commit.hexsha

            # Detect languages for all files (populates cache)
            file_languages = {file: self._detect_language(file) for file in files}
            logger.info(f"[{tenant_id}:{colony_id}] group_files: detected languages for {len(file_languages)} files")

            # Try to get cached graph
            graph = await self.file_graph_cache.get(
                key=f"{tenant_id}:{colony_id}:{commit_hash}", version=self._get_graph_version(files)
            )

            # Track files with known cross-language bindings
            cross_lang_bindings = set()

            if graph is None:
                logger.info(f"[{tenant_id}:{colony_id}] group_files: no cached graph, building from scratch")
                graph = await self._build_graph(
                    repo, files, cross_lang_bindings
                )
                logger.info(
                    f"[{tenant_id}:{colony_id}] group_files: graph built: "
                    f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
                )

                # Store cross-language bindings in graph metadata
                graph.graph["cross_lang_bindings"] = list(cross_lang_bindings)

                # Cache the constructed graph
                await self.file_graph_cache.set(
                    key=f"{tenant_id}:{colony_id}:{commit_hash}",
                    graph=graph,
                    version=self._get_graph_version(files),
                )
                logger.info(f"[{tenant_id}:{colony_id}] group_files: graph cached")
            else:
                logger.info(
                    f"[{tenant_id}:{colony_id}] group_files: loaded cached graph: "
                    f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
                )
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
            logger.info(f"[{tenant_id}:{colony_id}] group_files: clustering files into communities")
            initial_groups = await self._cluster_files_with_languages(
                graph, cross_lang_bindings
            )
            logger.info(f"[{tenant_id}:{colony_id}] group_files: {len(initial_groups)} initial groups")

            # Apply language-specific optimizations while preserving cross-language relationships
            optimized_groups = await self._optimize_groups_with_languages(
                initial_groups, cross_lang_bindings
            )

            duration = time.time() - start_time
            self.metrics.grouping_duration.labels(strategy="grouping").observe(duration)
            logger.info(f"[{tenant_id}:{colony_id}] Grouped {len(files)} files into {len(optimized_groups)} groups in {duration:.2f}s")
            return optimized_groups

        except Exception as e:
            logger.error(
                f"Error grouping files: {e}",
                exc_info=True,
                extra={
                    "repo": repo.working_dir,
                    "tenant_id": tenant_id,
                    "colony_id": colony_id,
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
        graph = nx.DiGraph()

        enabled = self.config.strategies
        # Per-strategy timeout: allow each strategy a generous but finite budget.
        # This prevents any single strategy from hanging the whole pipeline.
        strategy_timeout_s = self.config.analysis_timeout * len(files) / 10  # scale with file count
        strategy_timeout_s = max(strategy_timeout_s, 60.0)  # at least 60s
        strategy_timeout_s = min(strategy_timeout_s, 600.0)  # at most 10 minutes

        async def _run_strategy(name: str, coro):
            start = time.time()
            logger.info(f"_build_graph: starting {name} strategy ({len(files)} files)")
            try:
                await asyncio.wait_for(coro, timeout=strategy_timeout_s)
                elapsed = time.time() - start
                logger.info(f"_build_graph: {name} completed in {elapsed:.1f}s")
            except asyncio.TimeoutError:
                elapsed = time.time() - start
                logger.warning(
                    f"_build_graph: {name} timed out after {elapsed:.1f}s "
                    f"(limit={strategy_timeout_s:.0f}s), skipping"
                )
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"_build_graph: {name} failed after {elapsed:.1f}s: {e}")

        async with asyncio.TaskGroup() as tg:
            if enabled & FileGroupingStrategy.IMPORTS:
                tg.create_task(
                    _run_strategy("imports", self._add_import_relationships(graph, files, cross_lang_bindings))
                )
            if enabled & FileGroupingStrategy.DEPENDENCIES:
                tg.create_task(
                    _run_strategy("dependencies", self._add_dependency_relationships(graph, files, cross_lang_bindings))
                )
            if enabled & FileGroupingStrategy.COMMIT_HISTORY:
                tg.create_task(
                    _run_strategy("commit_history", self._add_commit_relationships(graph, repo, files))
                )
            if enabled & FileGroupingStrategy.SEMANTIC:
                tg.create_task(
                    _run_strategy("semantic", self._add_semantic_relationships(graph, files, cross_lang_bindings))
                )
            if enabled & FileGroupingStrategy.DIRECTORY:
                tg.create_task(
                    _run_strategy("directory", self._add_directory_relationships(graph, files))
                )

        # Detect circular dependencies before grouping (CPU-bound, offload to thread with timeout)
        try:
            cycles = await asyncio.wait_for(
                asyncio.to_thread(self._detect_circular_dependencies, graph),
                timeout=30.0,
            )
            if cycles:
                logger.warning(f"Detected {len(cycles)} circular dependencies")
                graph.graph["circular_dependencies"] = cycles
        except asyncio.TimeoutError:
            logger.warning("Circular dependency detection timed out after 30s, skipping")

        self._print_graph_summary(graph)
        return graph

    def _print_graph_summary(self, graph: nx.DiGraph):
        """Log a concise graph summary at DEBUG level."""
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        if n_nodes == 0:
            logger.debug("Relationship graph is empty")
            return

        density = nx.density(graph)
        components = list(nx.strongly_connected_components(graph))
        largest = max(len(c) for c in components) if components else 0
        logger.debug(
            f"Relationship graph: {n_nodes} nodes, {n_edges} edges, "
            f"density={density:.4f}, {len(components)} SCCs (largest={largest})"
        )

    def _get_graph_version(self, files: list[str]) -> str:
        """Generate version string for graph caching.

        Includes both file paths and modification times so that the cache
        is invalidated when any file's content changes (even if paths are
        unchanged).  The graph is also keyed by commit hash in ``group_files``,
        which provides a further layer of invalidation for committed changes.
        """
        if xxhash is None:
            return "unknown"
        hasher = xxhash.xxh64()
        for fpath in sorted(files):
            hasher.update(fpath.encode())
            try:
                mtime = os.path.getmtime(fpath)
                hasher.update(str(mtime).encode())
            except OSError:
                pass
        return hasher.hexdigest()

    def _merge_edge_into_graph(
        self,
        graph: nx.DiGraph,
        source: str,
        target: str,
        weight: float,
        relationship_type: str,
        metadata: dict[str, Any],
    ) -> None:
        """Add or merge an edge into the graph.

        If the edge already exists (from a different analyzer), the
        relationship data is merged: types are accumulated and the
        maximum weight across all relationships is kept.
        """
        if graph.has_edge(source, target):
            edge = graph.edges[source, target]
            if relationship_type not in edge["relationship_types"]:
                edge["relationship_types"].append(relationship_type)
            edge["weight"] = max(edge["weight"], weight)
            # Merge metadata — keep existing keys, add new ones
            for k, v in metadata.items():
                if k not in edge["metadata"]:
                    edge["metadata"][k] = v
        else:
            graph.add_edge(
                source,
                target,
                weight=weight,
                relationship_types=[relationship_type],
                metadata=metadata,
            )

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
        """Optimize groups containing cross-language relationships.

        Cross-language groups are kept together (that's the whole point of
        detecting cross-language bindings). Only split if the group exceeds
        max_group_size — and even then, split by token budget rather than
        by language, to preserve cross-language co-location.
        """
        try:
            if len(group.files) <= self.config.max_group_size:
                return [group]

            # Group is too large — split by token budget, which preserves
            # relationships better than splitting by language.
            if self.token_manager:
                return await self._split_by_tokens(group)

            return self._fallback_split(group.files)

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

            polymathera = await get_initialized_polymathera()

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
                        import_analyzer = await self._get_import_analyzer(source_language)
                        imports = await import_analyzer.analyze_file(
                            source_fpath, content, source_language
                        )
                        logger.debug(f"Import analysis for {source_fpath}: {len(imports) - 1} categories")

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

                                normalized_source_fpath = await polymathera.normalize_file_path(source_fpath)
                                normalized_target_fpath = await polymathera.normalize_file_path(target_fpath)

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

                                self._merge_edge_into_graph(
                                    graph,
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

                        if external_imports:
                            logger.debug(f"External imports for {source_fpath}: {sum(len(v) for v in external_imports.values())} paths")

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

    def _calculate_dependency_weight(
        self,
        dep_info: dict[str, Any],
        source_lang: str,
        target_lang: str | None,
    ) -> float:
        """Calculate weight for a dependency relationship.

        Args:
            dep_info: Dependency metadata from DependencyAnalyzer (keys like
                      "type", "confidence", "cross_language").
            source_lang: Programming language of the source file.
            target_lang: Programming language of the target file (may be None).
        """
        type_multipliers = {
            "class": 1.0,
            "function": 0.9,
            "type": 0.8,
            "interface": 0.95,
            "inheritance": 1.0,
            "module": 0.85,
        }
        dep_type = dep_info.get("type", "unknown")
        weight = self.config.dependency_weight * type_multipliers.get(dep_type, 0.6)

        # Factor in analyzer confidence if available
        confidence = dep_info.get("confidence")
        if confidence is not None:
            weight *= float(confidence)

        # Cross-language modifier
        is_cross_lang = target_lang and target_lang != source_lang
        if is_cross_lang:
            lang_pair = frozenset([source_lang, target_lang])
            if lang_pair in STRONG_LANGUAGE_BINDINGS:
                weight *= self.config.cross_lang_weight_multiplier
            else:
                weight *= 0.8

        return min(weight, 1.0)

    async def _add_dependency_relationships(
        self,
        graph: nx.DiGraph,
        files: list[str],
        cross_lang_bindings: set[str],
    ):
        """Add edges based on dependencies, including cross-language dependencies."""
        try:
            start_time = time.time()
            polymathera = await get_initialized_polymathera()
            files_set = set(files)

            async def process_file(source_fpath: str):
                try:
                    async with self.semaphore:
                        source_language = self._detect_language(source_fpath)
                        dependency_analyzer = await self._get_dependency_analyzer(source_language)
                        if not dependency_analyzer:
                            return

                        cache_key = f"deps:{source_fpath}"
                        deps = await self.dependency_cache.get(cache_key)

                        if deps is None:
                            deps = await dependency_analyzer.analyze_file(
                                source_fpath,
                                content=None,
                                language=source_language,
                                cross_language=True,
                            )
                            await self.dependency_cache.set(cache_key, deps)
                        else:
                            self.metrics.cache_hits.labels(cache_type="dependencies").inc()

                        for target_fpath, dep_info in deps.items():
                            if target_fpath not in files_set:
                                continue
                            target_language = self._detect_language(target_fpath)
                            is_cross_lang = target_language and target_language != source_language

                            normalized_source = await polymathera.normalize_file_path(source_fpath)
                            normalized_target = await polymathera.normalize_file_path(target_fpath)

                            if is_cross_lang:
                                cross_lang_bindings.add(normalized_source)
                                cross_lang_bindings.add(normalized_target)

                            weight = self._calculate_dependency_weight(
                                dep_info, source_language, target_language
                            )
                            self._merge_edge_into_graph(
                                graph,
                                normalized_source,
                                normalized_target,
                                weight=weight,
                                relationship_type="dependency",
                                metadata={
                                    **dep_info,
                                    "is_cross_language": is_cross_lang,
                                    "source_language": source_language,
                                    "target_language": target_language or "unknown",
                                },
                            )
                except Exception as e:
                    logger.error(f"Error processing dependencies for {source_fpath}: {e}", exc_info=True)
                    self.metrics.errors.labels(error_type="dependency_analysis").inc()

            async with asyncio.TaskGroup() as tg:
                for source_fpath in files:
                    tg.create_task(process_file(source_fpath))

            duration = time.time() - start_time
            self.metrics.analysis_duration.labels(strategy="dependencies").observe(duration)

        except Exception as e:
            logger.error(
                f"Error analyzing dependencies: {e}",
                exc_info=True,
                extra={"files": len(files)},
            )
            self.metrics.errors.labels(error_type="dependencies").inc()

    async def _add_directory_relationships(self, graph: nx.DiGraph, files: list[str]):
        """Add edges based on directory proximity.

        Instead of comparing every O(n²) file pair, group files by directory
        and only create edges between files that share a directory ancestor.
        Files in the same directory get the strongest score; cousins get less.
        """
        try:
            start_time = time.time()
            polymathera = await get_initialized_polymathera()

            # Group files by their parent directory
            dir_to_files: dict[str, list[str]] = defaultdict(list)
            for fpath in files:
                dir_to_files[str(Path(fpath).parent)].append(fpath)

            # Add edges between files in the same directory (score = directory_weight)
            for dir_path, dir_files in dir_to_files.items():
                if len(dir_files) < 2:
                    continue
                normalized = {}
                for f in dir_files:
                    normalized[f] = await polymathera.normalize_file_path(f)

                for j in range(len(dir_files)):
                    for k in range(j + 1, len(dir_files)):
                        score = self.config.directory_weight
                        if score >= self.config.min_relationship_score:
                            self._merge_edge_into_graph(
                                graph,
                                normalized[dir_files[j]],
                                normalized[dir_files[k]],
                                weight=score,
                                relationship_type="directory",
                                metadata={},
                            )

            # Add weaker edges between sibling directories (one level up)
            parent_to_dirs: dict[str, list[str]] = defaultdict(list)
            for dir_path in dir_to_files:
                parent = str(Path(dir_path).parent)
                parent_to_dirs[parent].append(dir_path)

            for parent, child_dirs in parent_to_dirs.items():
                if len(child_dirs) < 2:
                    continue
                # Cross-directory score: half of directory_weight
                score = self.config.directory_weight * 0.5
                if score < self.config.min_relationship_score:
                    continue

                for j in range(len(child_dirs)):
                    for k in range(j + 1, len(child_dirs)):
                        # Pick one representative file from each directory
                        f1 = dir_to_files[child_dirs[j]][0]
                        f2 = dir_to_files[child_dirs[k]][0]
                        n1 = await polymathera.normalize_file_path(f1)
                        n2 = await polymathera.normalize_file_path(f2)
                        self._merge_edge_into_graph(
                            graph, n1, n2,
                            weight=score,
                            relationship_type="directory",
                            metadata={},
                        )

            duration = time.time() - start_time
            self.metrics.analysis_duration.labels(strategy="directory").observe(duration)

        except Exception as e:
            logger.error(
                f"Error analyzing directories: {e}",
                exc_info=True,
                extra={"files": len(files)},
            )
            self.metrics.errors.labels(error_type="directory").inc()

    def _split_large_group(
        self, files: list[str], graph: nx.DiGraph
    ) -> list[FileGroup]:
        """Split large groups while preserving strong relationships"""
        try:
            # Create undirected subgraph for these files (connected_components
            # requires an undirected graph)
            subgraph = graph.subgraph(files).to_undirected()

            # Calculate edge betweenness centrality
            edge_centrality = nx.edge_betweenness_centrality(subgraph)

            # Remove edges with high centrality until we get suitable sized components
            while edge_centrality:
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
        """Get distribution of relationship types in a group."""
        try:
            type_counts = defaultdict(int)
            subgraph = graph.subgraph(files)
            for _, _, data in subgraph.edges(data=True):
                for edge_type in data.get("relationship_types", []):
                    type_counts[edge_type] += 1

            return dict(type_counts)

        except Exception as e:
            logger.error(f"Error getting edge types: {e}")
            return {}

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

                            polymathera = await get_initialized_polymathera()
                            normalized_file1 = await polymathera.normalize_file_path(file1)
                            normalized_file2 = await polymathera.normalize_file_path(file2)

                            # Track strong cross-language semantic relationships
                            if is_cross_lang and similarity >= self.config.semantic_cross_lang_threshold:
                                cross_lang_bindings.add(normalized_file1)
                                cross_lang_bindings.add(normalized_file2)

                            self._merge_edge_into_graph(
                                graph,
                                normalized_file1,
                                normalized_file2,
                                weight=self._calculate_semantic_weight(similarity, lang1, lang2),
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

    def _calculate_semantic_weight(self, similarity: float, lang1: str, lang2: str) -> float:
        """Calculate relationship weight for semantic similarity.

        Args:
            similarity: Cosine similarity score (0.0-1.0) from the semantic analyzer.
            lang1: Language of file 1.
            lang2: Language of file 2.
        """
        # Start from similarity score scaled by the configured semantic weight
        weight = similarity * self.config.semantic_weight

        # Adjust for cross-language relationships
        if lang1 != lang2:
            lang_pair = frozenset([lang1, lang2])
            if lang_pair in STRONG_LANGUAGE_BINDINGS:
                # Boost weight for common language pairs
                weight *= self.config.cross_lang_weight_multiplier
            else:
                # Reduce weight for unusual combinations
                weight *= 0.8

        return min(weight, 1.0)

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
                            polymathera = await get_initialized_polymathera()
                            normalized_file1 = await polymathera.normalize_file_path(file1)
                            normalized_file2 = await polymathera.normalize_file_path(file2)
                            self._merge_edge_into_graph(
                                graph,
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

            # Use language-aware community detection (CPU-bound, offload to thread)
            communities = await asyncio.to_thread(
                community.best_partition,
                undirected_graph,
                resolution=self.config.community_resolution,
                random_state=42,  # For reproducibility
            )
            logger.info(f"Community detection: {len(set(communities.values()))} communities from {len(communities)} files")

            # Group files by community — track both denormalized (for FileGroup)
            # and normalized (for graph queries / cross_lang_bindings comparison)
            groups = defaultdict(list)           # community_id → [denormalized paths]
            normalized_groups = defaultdict(list)  # community_id → [normalized paths]
            scores = defaultdict(float)
            languages = defaultdict(set)

            polymathera = await get_initialized_polymathera()

            for normalized_fpath, community_id in communities.items():
                fpath = await polymathera.denormalize_file_path(normalized_fpath)
                groups[community_id].append(fpath)
                normalized_groups[community_id].append(normalized_fpath)
                languages[community_id].add(self._detect_language(fpath))

                # Calculate average relationship score
                edges = list(graph.edges(normalized_fpath, data=True))
                if edges:
                    total_weight = sum(e[2]["weight"] for e in edges)
                    scores[community_id] += total_weight / len(edges)

            # Create FileGroup objects with language metadata
            result = []
            for community_id, files in groups.items():
                norm_files = normalized_groups[community_id]
                result.append(
                    FileGroup(
                        files=files,
                        relationship_score=scores[community_id] / len(files),
                        group_type="language_aware_community",
                        metadata={
                            "community_id": community_id,
                            "languages": list(languages[community_id]),
                            "cross_language_bindings": len(
                                set(norm_files) & cross_lang_bindings
                            ),
                            "edge_types": self._get_edge_types(graph, norm_files),
                        },
                    )
                )

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

    def _detect_circular_dependencies(self, graph: nx.DiGraph) -> list[dict[str, Any]]:
        """Detect and analyze circular dependencies"""
        try:
            from itertools import islice
            # Limit cycle enumeration — simple_cycles can produce an exponential
            # number of cycles on dense graphs and hang forever.
            max_cycles = 100
            cycles: list[list[str]] = list(islice(nx.simple_cycles(graph), max_cycles))

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

                    cycle_info["types"].update(edge_data.get("relationship_types", []))
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
                "relationship_types": len({
                    rt
                    for i in range(len(cycle))
                    for rt in graph.edges[cycle[i], cycle[(i + 1) % len(cycle)]].get(
                        "relationship_types", []
                    )
                }),
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
        """Optimize groups for module-based languages.

        Module relationships are already captured in the main relationship
        graph (via ``_add_import_relationships``), and the community detection
        step groups files by those relationships. This method only needs to
        enforce size limits.
        """
        if len(group.files) <= self.config.max_group_size:
            return [group]
        return self._fallback_split(group.files)


