"""
The key improvements include:
1. Configurable Strategies:
    - Multiple splitting approaches
    - Adaptive strategy selection
    - Feature flags
2. Performance Optimizations:
    - Parallel processing
    - Multiple caching options
    - Skip large files
    - Timeout handling
3. Multi-file Support:
    - Related file grouping
    - Segment combining
    - Import tracking
4. Modularity:
    - Feature toggles
    - Fallback mechanisms
    - Strategy selection

Key improvements:
1. Multiple Splitting Strategies:
    - Simple line-based
    - Indentation-aware
    - Syntax-aware (Pygments)
    - Full AST parsing (tree-sitter)
2. Graceful Degradation:
    - Falls back to simpler strategies on failure
    - Handles parsing errors
3. Language Support:
    - Basic language detection
    - Language-specific splitting rules
    - Extensible for new languages
4. Size Management:
    - Respects maximum size limits
    - Byte-based size calculation
    - Smart split point selection

5. Configurability:
    - Per-language feature toggles
    - Performance tuning options
    - Custom patterns
    - Size thresholds
6. Performance:
    - Skip large nodes
    - Parallel processing option
    - Caching options
    - Timeout handling
7. Extensibility:
    - Easy to add new languages
    - Custom splitting rules
    - Project-specific configurations
8. Modularity:
    - Separate language rules
    - Feature toggles
    - Fallback mechanisms

1. Flexible Rule System:
    - Custom rule definitions
    - Priority-based processing
    - Language-specific rules
    - Context preservation
2. Advanced Features:
    - Custom conditions
    - Minimum segment sizes
    - Context lines
    - Pattern optimization
3. Error Handling:
    - Per-rule error handling
    - Fallback mechanisms
    - Logging and monitoring
4. Performance Considerations:
    - Compiled patterns
    - Efficient matching
    - Size-based optimization

5. Parallel Processing:
    - Thread pool for concurrent processing
    - Language-based file grouping
    - Configurable worker count
    - Resource monitoring
6. Performance Monitoring:
    - Prometheus metrics
    - Duration tracking
    - Cache hit rates
    - Error counting
    - Active worker tracking
7. Caching Strategy:
    - Two-level cache (local + distributed)
    - Content-based cache keys
    - TTL support
    - Error handling
9. Optimizations:
    - Language-based grouping
    - Cache locality
    - Configurable parameters
    - Resource management

"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import resource
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from statistics import mean, median
from typing import ClassVar

import psutil
from pydantic import Field
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token

from polymathera.colony.distributed.caching.simple import CacheConfig
from polymathera.colony.distributed.config import ConfigComponent, register_polymathera_config
from polymathera.colony.distributed import get_polymathera
from polymathera.colony.distributed.metrics.common import BaseMetricsMonitor
from polymathera.colony.utils import create_dynamic_asyncio_task, cleanup_dynamic_asyncio_tasks, call_async_in_executor

from .languages.code_splitting import (
    CustomRule,
    LanguageConfig,
    LanguageOptimization,
    LanguageRules,
    RuleManager,
    RuleMatch,
)
from .languages.utils import detect_language
from .types import ShardFileSegment

logger = logging.getLogger(__name__)


################################################################################
# Performance Monitoring
################################################################################


@dataclass
class ProcessingMetrics:
    """Metrics for code splitting performance"""

    file_count: int = 0
    total_size: int = 0
    processing_time: float = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0


@dataclass
class ResourceUsage:
    cpu_percent: float
    memory_percent: float
    io_wait: float
    thread_count: int
    open_files: int
    context_switches: int


class CodeSplitterMetricsMonitor(BaseMetricsMonitor):
    """Base class for Prometheus metrics monitoring using node-global HTTP server."""
    """Monitors and reports code splitting performance"""

    def __init__(self,
                 enable_http_server: bool = True,
                 service_name: str = "service"):
        super().__init__(enable_http_server, service_name)

        self.logger.info(f"Initializing CodeSplitterMetricsMonitor instance {id(self)}...")

        self.split_duration = self.create_histogram(
            "code_splitter_duration_seconds",
            "Time spent splitting code",
            labelnames=["language", "strategy"],
        )
        self.processing_duration = self.create_histogram(
            "code_splitter_processing_duration_seconds",
            "Time spent processing files",
            labelnames=["language", "strategy", "size_bucket"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        )
        self.file_size = self.create_histogram(
            "code_splitter_file_size_bytes",
            "Size of processed files",
            labelnames=["language"]
        )
        self.active_workers = self.create_gauge(
            "code_splitter_active_workers",
            "Number of active worker threads",
            labelnames=["language"]
        )

        # Error Metrics
        self.error_counter = self.create_counter(
            "code_splitter_errors_total",
            "Number of errors during splitting",
            labelnames=["error_type"]
        )
        self.errors = self.create_counter(
            "code_splitter_errors_by_language",
            "Number of errors by language",
            labelnames=["language"]
        )
        self.retries = self.create_counter(
            "code_splitter_retries_total",
            "Number of retries",
            labelnames=["error_type", "language"]
        )

        # Throughput Metrics
        self.throughput = self.create_counter(
            "code_splitter_throughput_bytes",
            "Bytes processed per second",
            labelnames=["language"]
        )

        # Queue Metrics
        self.queue_size = self.create_gauge(
            "code_splitter_queue_size",
            "Number of files waiting to be processed",
            labelnames=["language"]
        )
        self.queue_latency = self.create_histogram(
            "code_splitter_queue_latency_seconds",
            "Time spent in queue",
            labelnames=["language"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )

        # Cache Metrics
        self.cache_ops = self.create_counter(
            "code_splitter_cache_operations_total",
            "Cache operations",
            labelnames=["operation", "result", "level"]  # level: local, distributed
        )
        self.cache_size = self.create_gauge(
            "code_splitter_cache_size_bytes",
            "Cache size in bytes",
            labelnames=["level"]
        )

        # Recent performance tracking
        self.recent_durations = deque(maxlen=100)
        self.recent_throughput = deque(maxlen=100)

        self.process = psutil.Process()

        # Resource Gauges
        self.cpu_usage = self.create_gauge(
            "code_splitter_cpu_usage_percent",
            "CPU usage percentage",
            labelnames=["type"]  # system, process
        )
        self.memory_usage = self.create_gauge(
            "code_splitter_memory_usage_bytes",
            "Memory usage in bytes",
            labelnames=["type"]  # rss, vms
        )
        self.io_wait = self.create_gauge(
            "code_splitter_io_wait_percent",
            "IO wait percentage",
            labelnames=["type"]
        )
        self.file_handles = self.create_gauge(
            "code_splitter_open_files",
            "Number of open file handles",
            labelnames=["type"]
        )
        self.thread_count = self.create_gauge(
            "code_splitter_thread_count",
            "Number of active threads",
            labelnames=["type"]
        )

        # Resource Limits
        self.resource_limits = self.create_gauge(
            "code_splitter_resource_limits",
            "Resource limits",
            ["resource", "type"],  # soft, hard
        )

        # Worker Metrics
        self.worker_count = self.create_gauge(
            "code_splitter_worker_count",
            "Number of worker threads",
            labelnames=["state"]  # active, idle
        )
        self.worker_duration = self.create_histogram(
            "code_splitter_worker_duration_seconds",
            "Worker processing duration",
            labelnames=["worker_id"]
        )

        # Start monitoring task
        self.monitoring = False
        self.monitoring_task = None

    async def record_split(self, duration: float, language: str, strategy: str):
        self.split_duration.labels(language, strategy).observe(duration)

    async def record_file(self, size: int, language: str):
        self.file_size.labels(language).observe(size)

    async def record_cache_op(self, operation: str, hit: bool):
        self.cache_ops.labels(operation, "hit" if hit else "miss").inc()

    async def record_error(self, error_type: str):
        self.error_counter.labels(error_type).inc()

    def start_monitoring(self, interval: float = 1.0):
        """Start resource monitoring"""
        self.monitoring = True
        self.monitoring_task = create_dynamic_asyncio_task(self, self._monitor_resources(interval))

    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitoring_task:
            await cleanup_dynamic_asyncio_tasks(self.monitoring_task)

    async def _monitor_resources(self, interval: float):
        """Continuously monitor resource usage"""
        while self.monitoring:
            try:
                # Process metrics
                proc_info = self.process.as_dict(
                    attrs=[
                        "cpu_percent",
                        "memory_percent",
                        "io_counters",
                        "num_threads",
                        "num_fds",
                        "num_ctx_switches",
                    ]
                )

                # Update gauges
                self.cpu_usage.labels(type="process").set(proc_info["cpu_percent"])
                self.cpu_usage.labels(type="system").set(psutil.cpu_percent())

                self.memory_usage.labels(type="rss").set(self.process.memory_info().rss)
                self.memory_usage.labels(type="vms").set(self.process.memory_info().vms)

                self.thread_count.labels(type="process").set(proc_info["num_threads"])
                self.file_handles.labels(type="process").set(proc_info["num_fds"])

                # Update resource limits
                for resource_type in [resource.RLIMIT_NOFILE, resource.RLIMIT_AS]:
                    soft, hard = resource.getrlimit(resource_type)
                    self.resource_limits.labels(
                        resource=resource_type,
                        type="soft"
                    ).set(soft)
                    self.resource_limits.labels(
                        resource=resource_type,
                        type="hard"
                    ).set(hard)

            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")

            await asyncio.sleep(interval)


################################################################################
# Parallel Processing
################################################################################


class AdaptiveWorkerPool:
    """Adaptive thread pool that scales based on system load"""

    def __init__(
        self,
        initial_workers: int,
        min_workers: int,
        max_workers: int,
        metrics: CodeSplitterMetricsMonitor,
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = initial_workers
        self.metrics = metrics

        # Thread pool executor
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="code_splitter"
        )

        # Performance tracking
        self.performance_window = deque(maxlen=10)
        self.last_adjustment = time.time()
        self.adjustment_interval = 30  # seconds

    async def adjust_workers(self):
        """Adjust worker count based on system load"""
        current_time = time.time()
        if current_time - self.last_adjustment < self.adjustment_interval:
            return

        try:
            # Get current resource usage
            cpu_usage = self.metrics.cpu_usage.labels(type="process")._value.get()
            memory_usage = self.metrics.memory_usage.labels(type="rss")._value.get()

            # Calculate optimal worker count
            optimal_workers = self._calculate_optimal_workers(cpu_usage, memory_usage)

            # Apply changes gradually
            if optimal_workers > self.current_workers:
                self.current_workers = min(
                    self.current_workers + 2,  # Increase by 2 max
                    optimal_workers,
                    self.max_workers,
                )
            elif optimal_workers < self.current_workers:
                self.current_workers = max(
                    self.current_workers - 1,  # Decrease by 1 max
                    optimal_workers,
                    self.min_workers,
                )

            # Update metrics
            self.metrics.worker_count.labels(state="total").set(self.current_workers)

            self.last_adjustment = current_time

        except Exception as e:
            logger.error(f"Error adjusting workers: {e}")

    def _calculate_optimal_workers(self, cpu_usage: float, memory_usage: float) -> int:
        """Calculate optimal number of workers based on system load"""
        # Target CPU usage per worker (percentage)
        target_cpu_per_worker = 25

        # Available CPU capacity
        available_cpu = max(0, 80 - cpu_usage)  # Keep CPU below 80%

        # Available memory (keep below 80% usage)
        available_memory = max(0, 80 - memory_usage)

        # Calculate workers based on CPU and memory
        cpu_based_workers = int(available_cpu / target_cpu_per_worker)
        memory_based_workers = int(
            available_memory / 10
        )  # Assume 10% memory per worker

        # Take the minimum of the two
        return min(cpu_based_workers, memory_based_workers)

    async def get_worker_stats(self) -> dict:
        """Get current worker pool statistics"""
        return {
            "current_workers": self.current_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "performance": {
                "avg_duration": mean(self.performance_window)
                if self.performance_window
                else 0,
                "median_duration": median(self.performance_window)
                if self.performance_window
                else 0,
            },
        }

    async def process_files(
        self, files: list[tuple[str, str, str]], splitter: CodeSplitter
    ) -> dict[str, list[ShardFileSegment]]:
        """Process multiple files using adaptive worker pool"""
        results = {}
        pending_tasks = []
        active_workers = 0

        try:
            # Group files by language for better cache locality
            files_by_lang = self._group_by_language(files)

            # Process each language group
            for language, lang_files in files_by_lang.items():
                batch_size = self._calculate_batch_size(lang_files)

                # Process files in batches
                for i in range(0, len(lang_files), batch_size):
                    batch = lang_files[i : i + batch_size]

                    # Wait if we're at max workers
                    while active_workers >= self.current_workers:
                        done, pending_tasks = await asyncio.wait(
                            pending_tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                        active_workers -= len(done)

                        # Process completed tasks
                        for task in done:
                            file_path, segments = await task
                            results[file_path] = segments

                    # Submit new batch
                    for file_path, content in batch:
                        task = asyncio.create_task(
                            self._process_file(file_path, content, splitter)
                        )
                        pending_tasks.append(task)
                        active_workers += 1

                    # Update metrics
                    self.metrics.worker_count.labels(state="active").set(active_workers)

                    # Adjust worker count based on performance
                    await self.adjust_workers()

            # Wait for remaining tasks
            if pending_tasks:
                done, _ = await asyncio.wait(pending_tasks)
                for task in done:
                    file_path, segments = await task
                    results[file_path] = segments

        except Exception as e:
            logger.error(f"Error in worker pool: {e}")
            raise

        finally:
            self.metrics.worker_count.labels(state="active").set(0)

        return results

    async def _process_file(
        self, file_path: str, content: str, splitter: CodeSplitter
    ) -> tuple[str, list[ShardFileSegment]]:
        """Process a single file with performance tracking"""
        start_time = time.time()
        worker_id = threading.get_ident()

        try:
            # Try cache first
            cache_key = splitter.get_cache_key(file_path, content)
            cached_result = await splitter.cache.get(cache_key)

            if cached_result:
                return file_path, cached_result

            # Process file
            segments = await call_async_in_executor(
                self.executor,
                splitter._split_with_rules,
                content,
                file_path,
                detect_language(file_path),
                splitter.config.language_configs.get(detect_language(file_path), {}),
            )

            # Cache result
            await splitter.cache.set(cache_key, segments)

            # Record performance metrics
            duration = time.time() - start_time
            self.metrics.worker_duration.labels(worker_id=worker_id).observe(duration)
            self.performance_window.append(duration)

            return file_path, segments

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            raise

    def _calculate_batch_size(self, files: list[tuple[str, str]]) -> int:
        """Calculate optimal batch size based on current performance"""
        avg_duration = mean(self.performance_window) if self.performance_window else 1.0
        target_queue_time = 5.0  # seconds

        batch_size = max(
            1,
            min(
                int(target_queue_time / avg_duration * self.current_workers), len(files)
            ),
        )

        return batch_size

    def _group_by_language(
        self, files: list[tuple[str, str]]
    ) -> dict[str, list[tuple[str, str]]]:
        """Group files by programming language"""
        groups = {}
        for file_path, content in files:
            ext = Path(file_path).suffix.lower()
            if ext not in groups:
                groups[ext] = []
            groups[ext].append((file_path, content))
        return groups


@register_polymathera_config()
class ProcessingConfig(ConfigComponent):
    """Configuration for parallel processing"""

    max_workers: int = 4

    CONFIG_PATH: ClassVar[str] = "llms.sharding.code_splitter.processing"


class ParallelProcessor:
    """Handles parallel processing of multiple files"""

    def __init__(
        self,
        config: ProcessingConfig | None = None,
        metrics: CodeSplitterMetricsMonitor | None = None,
    ):
        self.config: ProcessingConfig | None = config
        self.metrics = metrics
        self.executor: ThreadPoolExecutor | None = None

    async def initialize(self) -> None:
        self.config = await ProcessingConfig.check_or_get_component(self.config)
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers, thread_name_prefix="code_splitter"
        )

    async def process_files(
        self, files: list[tuple[str, str, str]], splitter: CodeSplitter
    ) -> dict[str, list[ShardFileSegment]]:
        """Process multiple files in parallel"""
        results = {}
        futures = []

        # Group files by language for better cache locality
        files_by_lang = self._group_by_language(files)

        try:
            self.metrics.active_workers.labels(language=None).set(len(files))

            # Submit files for processing
            for file_path, content, mime_type in files:
                future = self.executor.submit(
                    self._process_single_file, file_path, content, mime_type, splitter
                )
                futures.append((file_path, future))

            # Collect results as they complete
            for file_path, future in futures:
                try:
                    segments = await asyncio.wrap_future(future)
                    results[file_path] = segments
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    await self.metrics.record_error(type(e).__name__)
                    results[file_path] = []

        finally:
            self.metrics.active_workers.labels(language=None).set(0)

        return results

    def _group_by_language(
        self, files: list[tuple[str, str]]
    ) -> dict[str, list[tuple[str, str]]]:
        """Group files by programming language"""
        groups = {}
        for file_path, content in files:
            lang = detect_language(file_path)
            if lang not in groups:
                groups[lang] = []
            groups[lang].append((file_path, content))
        return groups

    async def _process_single_file(
        self, file_path: str, content: str, mime_type: str, splitter: CodeSplitter
    ) -> list[ShardFileSegment]:
        """Process a single file with performance monitoring"""
        start_time = time.time()
        language = detect_language(file_path)

        try:
            # Try cache first
            cache_key = splitter.get_cache_key(file_path, content)
            cached_result = await splitter.cache.get(cache_key)

            if cached_result:
                await self.metrics.record_cache_op("get", True)
                return cached_result

            await self.metrics.record_cache_op("get", False)

            # Process file
            segments = await splitter.split_content(
                content, file_path, mime_type
            )  # TODO: Not implemented

            # Cache result
            await splitter.cache.set(cache_key, segments)

            # Record metrics
            duration = time.time() - start_time
            await self.metrics.record_split(
                duration, language, splitter.strategy
            )
            await self.metrics.record_file(len(content), language)

            return segments

        except Exception as e:
            await self.metrics.record_error(type(e).__name__)
            raise


################################################################################
# Code Splitter
################################################################################


class SplitStrategy(Enum):
    """Different strategies for code splitting with increasing complexity/accuracy"""

    SIMPLE_LINES = "lines"  # Simple line-based splitting
    INDENT_AWARE = "indent"  # Consider indentation levels
    SYNTAX_AWARE = "syntax"  # Use lexer for basic syntax
    FULL_PARSE = "parse"  # Full AST parsing
    HYBRID = "hybrid"  # Adaptive based on file size/type


@register_polymathera_config()
class CodeSplitterConfig(ConfigComponent):
    # General settings
    strategy: SplitStrategy = SplitStrategy.HYBRID
    max_shard_size: int = 1024 * 1024  # 1MB
    min_shard_size: int = 1024 * 32  # 32KB

    # Multi-file settings
    allow_multi_file: bool = True
    max_files_per_shard: int = 5
    prefer_related_files: bool = True

    # Performance settings
    initial_workers: int = 1
    min_workers: int = 1
    max_workers: int = 4
    parse_timeout: int = 30  # seconds

    # Language-specific settings
    language_configs: dict[str, LanguageConfig] = field(default_factory=dict)
    fallback_strategy: SplitStrategy = SplitStrategy.INDENT_AWARE

    # Feature flags
    enable_tree_sitter: bool = True
    enable_pygments: bool = True

    # Optimization flags
    skip_large_files: bool = True
    large_file_threshold: int = 1024 * 1024 * 5  # 5MB
    parallel_processing: bool = True

    cache_config: CacheConfig | None = Field(default=None)
    processing_config: ProcessingConfig | None = Field(default=None)

    CONFIG_PATH: ClassVar[str] = "llms.sharding.code_splitter"


class CodeSplitter:
    """Handles intelligent code splitting based on language and structure
    TODO: Add support for overlapping shards
    """

    def __init__(self, config: CodeSplitterConfig | None = None):
        self.config = config
        self.language_rules = LanguageRules()
        self.rule_manager = RuleManager()
        self.metrics = CodeSplitterMetricsMonitor()
        self.cache = None
        self.parallel_processor = None
        self.worker_pool = None

    async def initialize(self):
        self.config = await CodeSplitterConfig.check_or_get_component(self.config)
        self.cache = await get_polymathera().create_distributed_simple_cache(
            namespace="code_splitter:cache",  # TODO: Scope is global to all VMRs?
            config=self.config.cache_config,
        )
        self.parallel_processor = ParallelProcessor(
            self.config.processing_config,
            self.metrics
        )
        await self.parallel_processor.initialize()
        self.worker_pool = AdaptiveWorkerPool(
            initial_workers=self.config.initial_workers,
            min_workers=self.config.min_workers,
            max_workers=self.config.max_workers,
            metrics=self.metrics,
        )

        self._setup_parsers()
        self.metrics.start_monitoring()

    async def cleanup(self):
        await self.metrics.stop_monitoring()

    def _setup_parsers(self):
        """Initialize parsers based on configuration"""
        self.parsers = {}
        if self.config.enable_tree_sitter:
            self._init_tree_sitter()

    def _init_tree_sitter(self):
        """Initialize Tree-sitter parsers for supported languages"""
        try:
            import tree_sitter
            from tree_sitter import Language, Parser

            # Common languages that tree-sitter supports
            supported_languages = {
                'python': 'tree-sitter-python',
                'javascript': 'tree-sitter-javascript',
                'typescript': 'tree-sitter-typescript',
                'java': 'tree-sitter-java',
                'cpp': 'tree-sitter-cpp',
                'c': 'tree-sitter-c',
                'rust': 'tree-sitter-rust',
                'go': 'tree-sitter-go',
                'ruby': 'tree-sitter-ruby',
                'php': 'tree-sitter-php',
                'html': 'tree-sitter-html',
                'css': 'tree-sitter-css',
                'json': 'tree-sitter-json',
                'yaml': 'tree-sitter-yaml',
                'bash': 'tree-sitter-bash',
            }

            for lang_name, lib_name in supported_languages.items():
                try:
                    # Try to load the language library
                    # Note: In production, these would need to be pre-built and available
                    # For now, we'll gracefully handle missing languages
                    parser = Parser()
                    # This would require the tree-sitter language libraries to be built
                    # language = Language(lib_name, lang_name)
                    # parser.set_language(language)
                    # self.parsers[lang_name] = parser
                    logger.debug(f"Tree-sitter parser for {lang_name} would be initialized here")
                except Exception as e:
                    logger.debug(f"Could not load tree-sitter parser for {lang_name}: {e}")

        except ImportError:
            logger.warning("Tree-sitter not available, falling back to pygments")
            self.config.enable_tree_sitter = False

    def _calculate_scope_depth(self, content_prefix: str, opt: LanguageOptimization) -> int:
        """Calculate the scope depth at a given position in the code"""
        try:
            depth = 0
            in_string = False
            in_comment = False
            string_char = None

            i = 0
            while i < len(content_prefix):
                char = content_prefix[i]

                # Handle string literals
                if not in_comment and char in ['"', "'", '`']:
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        # Check if it's escaped
                        if i == 0 or content_prefix[i-1] != '\\':
                            in_string = False
                            string_char = None

                # Handle comments (simplified)
                elif not in_string:
                    if char == '#':  # Python, Ruby, Bash comments
                        in_comment = True
                    elif char == '\n':
                        in_comment = False
                    elif not in_comment:
                        # Count scope delimiters
                        if char in opt.scope_open_chars:
                            depth += 1
                        elif char in opt.scope_close_chars:
                            depth = max(0, depth - 1)

                i += 1

            return depth

        except Exception as e:
            logger.debug(f"Error calculating scope depth: {e}")
            return 0

    def get_cache_key(self, file_path: str, content: str) -> str:
        """Generate cache key for file content"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"split:{file_path}:{content_hash}"

    async def split_files_1(
        self, files: list[tuple[str, str]]
    ) -> dict[str, list[ShardFileSegment]]:
        """Split multiple files with adaptive worker pool"""
        start_time = time.time()

        try:
            # Update queue metrics
            self.metrics.queue_size.labels(language=None).set(len(files))

            # Process files
            results = await self.worker_pool.process_files(files, self)

            # Record metrics
            duration = time.time() - start_time
            total_bytes = sum(len(content) for _, content in files)

            self.metrics.throughput.labels(language=None).inc(total_bytes / duration)

            # Adjust worker pool
            await self.worker_pool.adjust_workers()

            return results

        except Exception as e:
            self.metrics.errors.labels(
                type=type(e).__name__, language="unknown"
            ).inc()
            raise
        finally:
            self.metrics.queue_size.labels(language=None).set(0)

    async def split_files_2(
        self, files: list[tuple[str, str, str]]
    ) -> dict[str, list[ShardFileSegment]]:
        """Split multiple files with parallel processing"""
        return await self.parallel_processor.process_files(files, self)

    async def split_files_3(
        self,
        files: list[tuple[str, str, str]],  # List of (path, content, mime_type)
        max_shard_size: int | None = None,
    ) -> list[list[ShardFileSegment]]:
        """
        Split multiple files into shards, potentially combining related files.
        Returns list of shards, where each shard is a list of file segments.
        """
        max_size = max_shard_size or self.config.max_shard_size

        # First pass: analyze files and create segments
        segments = []
        file_groups = {}  # Group related files

        if self.config.parallel_processing:
            # For parallel processing, we need to handle async calls properly
            tasks = [self._split_single_file(file_path, content, mime_type) for file_path, content, mime_type in files]
            segment_lists = await asyncio.gather(*tasks)
            segments = [s for sublist in segment_lists for s in sublist]
        else:
            for file_path, content, mime_type in files:
                file_segments = await self._split_single_file(file_path, content, mime_type)
                segments.extend(file_segments)

        # Group related segments if configured
        if self.config.prefer_related_files:
            segments = self._group_related_segments(segments)

        # Second pass: combine segments into shards
        return self._combine_segments(segments, max_size)

    async def _split_single_file(
        self, file_path: str, content: str, mime_type: str
    ) -> list[ShardFileSegment]:
        """Split a single file into segments"""
        # Check cache first
        cache_key = self.get_cache_key(file_path, content)
        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        # Skip large files if configured
        if (
            self.config.skip_large_files
            and len(content) > self.config.large_file_threshold
        ):
            logger.warning(f"Skipping large file: {file_path}")
            return []

        # Determine best strategy
        strategy = self._select_strategy(file_path, content, mime_type)

        # Split based on strategy
        try:
            segments = self._split_with_strategy(file_path, content, mime_type, strategy)

            # Cache result
            await self.cache.set(cache_key, segments)

            return segments

        except Exception as e:
            logger.error(f"Error splitting {file_path}: {e}")
            # Fallback to simple splitting
            return self._split_by_lines(content, file_path, mime_type)

    def _select_strategy(self, file_path: str, content: str, mime_type: str) -> SplitStrategy:
        """Select best splitting strategy based on file characteristics"""
        if self.config.strategy != SplitStrategy.HYBRID:
            return self.config.strategy

        # TODO: Implement smart strategy selection based on:
        # - File size
        # - File type
        # - Content complexity
        # - Available resources
        # - Previous processing times
        # Implement smart strategy selection
        file_size = len(content)
        file_ext = Path(file_path).suffix.lower()

        # For very small files, use simple line splitting
        if file_size < self.config.min_shard_size:
            return SplitStrategy.SIMPLE_LINES

        # For very large files, use more efficient strategies
        if file_size > self.config.large_file_threshold:
            return SplitStrategy.INDENT_AWARE

        # Use centralized language detection for code files
        from .languages.extensions import get_registry, FileCategory
        registry = get_registry()

        # For code files, use syntax-aware splitting
        if registry.is_code_file(file_path):
            if self.config.enable_tree_sitter and file_size < 1024 * 1024:  # 1MB
                return SplitStrategy.FULL_PARSE
            else:
                return SplitStrategy.SYNTAX_AWARE

        # For structured text (markup and config files), use indent-aware
        markup_extensions = registry.get_extensions_by_category(FileCategory.MARKUP)
        config_extensions = registry.get_extensions_by_category(FileCategory.CONFIG)
        structured_extensions = markup_extensions | config_extensions

        if file_ext in structured_extensions:
            return SplitStrategy.INDENT_AWARE

        # Default fallback
        return self.config.fallback_strategy

    def _split_with_strategy(
        self, file_path: str, content: str, mime_type: str, strategy: SplitStrategy
    ) -> list[ShardFileSegment]:
        """Split content using selected strategy"""
        if strategy == SplitStrategy.SIMPLE_LINES:
            return self._split_by_lines(content, file_path, mime_type)
        elif strategy == SplitStrategy.INDENT_AWARE:
            return self._split_by_indentation(content, file_path, mime_type)
        elif strategy == SplitStrategy.SYNTAX_AWARE:
            return self._split_with_pygments(content, file_path, mime_type)
        elif strategy == SplitStrategy.FULL_PARSE:
            return self._split_with_tree_sitter(content, file_path, mime_type)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _group_related_segments(self, segments: list[ShardFileSegment]) -> list[ShardFileSegment]:
        """Group segments based on imports and dependencies"""
        # TODO: Implement sophisticated grouping based on:
        # - Import statements
        # - Function calls
        # - Class hierarchies
        # - File proximity
        # - Semantic relationships
        # - Code complexity
        # - Previous processing times
        # Implement sophisticated grouping based on:
        # - Import statements
        # - Function calls
        if not segments:
            return segments

        # Create a simple grouping based on file proximity and imports
        grouped = []
        current_group = []

        for segment in segments:
            # Start new group if file changes or if we have imports that suggest separation
            if (current_group and
                (segment.file_path != current_group[-1].file_path or
                 len(current_group) >= self.config.max_files_per_shard)):
                grouped.extend(current_group)
                current_group = [segment]
            else:
                current_group.append(segment)

        # Add remaining segments
        if current_group:
            grouped.extend(current_group)

        return grouped

    def _combine_segments(
        self, segments: list[ShardFileSegment], max_size: int
    ) -> list[list[ShardFileSegment]]:
        """Combine segments into shards respecting size limits"""
        # TODO: Implement bin-packing algorithm to optimize shard sizes
        # TODO: Consider segment relationships when combining
        # TODO: Implement different combining strategies
        if not segments:
            return []

        shards = []
        current_shard = []
        current_size = 0

        for segment in segments:
            segment_size = len(segment.content.encode('utf-8'))

            # If adding this segment would exceed max size, start new shard
            if current_size + segment_size > max_size and current_shard:
                shards.append(current_shard)
                current_shard = [segment]
                current_size = segment_size
            else:
                current_shard.append(segment)
                current_size += segment_size

        # Add remaining segments
        if current_shard:
            shards.append(current_shard)

        return shards

    # TODO: Add more language support
    # TODO: Add more sophisticated caching
    # TODO: Add more metrics and monitoring
    # TODO: Add more optimization strategies
    # TODO: Add more error handling and recovery
    # TODO: Add more tests and validation
    # TODO: Add more documentation and examples
    # TODO: Add more logging and debugging
    # TODO: Add more profiling and optimization
    # TODO: Add more error handling and recovery
    # TODO: Add more tests and validation

    def _split_by_lines(self, content: str, file_path: str, mime_type: str) -> list[ShardFileSegment]:
        """Simple line-based splitting as fallback"""
        lines = content.splitlines(keepends=True)
        if not lines:
            return []

        segments = []
        current_lines = []
        current_size = 0
        start_line = 0

        for i, line in enumerate(lines):
            line_size = len(line.encode("utf-8"))  # Use byte size

            # Start new segment if adding this line would exceed max size
            if current_size + line_size > self.config.max_shard_size and current_lines:
                segments.append(
                    ShardFileSegment(
                        file_path=file_path,
                        start_line=start_line,
                        end_line=i,
                        content="".join(current_lines),
                        mime_type=mime_type,
                    )
                )
                current_lines = []
                current_size = 0
                start_line = i

            current_lines.append(line)
            current_size += line_size

        # Add remaining lines
        if current_lines:
            segments.append(
                ShardFileSegment(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=len(lines),
                    content="".join(current_lines),
                    mime_type=mime_type,
                )
            )

        return segments

    def _split_by_indentation(self, content: str, file_path: str, mime_type: str) -> list[ShardFileSegment]:
        """Split code based on indentation levels"""
        lines = content.splitlines(keepends=True)
        if not lines:
            return []

        segments = []
        current_lines = []
        current_size = 0
        start_line = 0
        base_indent = None

        def get_indent_level(line: str) -> int:
            return len(line) - len(line.lstrip())

        for i, line in enumerate(lines):
            line_size = len(line.encode("utf-8"))
            current_indent = get_indent_level(line)

            # Track base indentation level
            if line.strip() and base_indent is None:
                base_indent = current_indent

            # Start new segment if:
            # 1. Current segment would exceed max size, or
            # 2. We're at base indentation and previous content exists
            if (
                current_size + line_size > self.config.max_shard_size
                or (current_indent == base_indent and current_lines and line.strip())
            ) and current_lines:
                segments.append(
                    ShardFileSegment(
                        file_path=file_path,
                        start_line=start_line,
                        end_line=i,
                        content="".join(current_lines),
                        mime_type=mime_type,
                    )
                )
                current_lines = []
                current_size = 0
                start_line = i

            current_lines.append(line)
            current_size += line_size

        # Add remaining content
        if current_lines:
            segments.append(
                ShardFileSegment(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=len(lines),
                    content="".join(current_lines),
                    mime_type=mime_type,
                )
            )

        return segments

    def _split_with_pygments(self, content: str, file_path: str, mime_type: str) -> list[ShardFileSegment]:
        """Split code using Pygments lexer for syntax awareness"""
        try:
            lexer = guess_lexer_for_filename(file_path, content)
        except Exception as e:
            logger.warning(f"Failed to guess lexer for {file_path}: {e}")
            return self._split_by_indentation(content, file_path, mime_type)

        tokens = list(lexer.get_tokens_unprocessed(content))
        segments = []
        current_tokens = []
        current_size = 0
        start_pos = 0

        def is_major_token(token_type) -> bool:
            """Check if token type indicates major structural element"""
            return (
                token_type in Token.Keyword.Declaration
                or token_type in Token.Keyword.Namespace
                or str(token_type) in {"Token.Name.Class", "Token.Name.Function"}
            )

        for pos, token_type, value in tokens:
            token_size = len(value.encode("utf-8"))

            # Start new segment if:
            # 1. Current segment would exceed max size, or
            # 2. We're at a major structural element and have content
            if (
                current_size + token_size > self.config.max_shard_size
                or (is_major_token(token_type) and current_tokens)
            ) and current_tokens:
                segment_content = "".join(t[2] for t in current_tokens)
                start_line = content.count("\n", 0, start_pos)
                end_line = content.count("\n", 0, pos)

                segments.append(
                    ShardFileSegment(
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        content=segment_content,
                        mime_type=mime_type,
                    )
                )
                current_tokens = []
                current_size = 0
                start_pos = pos

            current_tokens.append((pos, token_type, value))
            current_size += token_size

        # Add remaining content
        if current_tokens:
            segment_content = "".join(t[2] for t in current_tokens)
            start_line = content.count("\n", 0, start_pos)
            end_line = content.count("\n", 0, len(content))

            segments.append(
                ShardFileSegment(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=segment_content,
                    mime_type=mime_type,
                )
            )

        return segments

    def _split_with_tree_sitter(
        self, content: str, file_path: str, mime_type: str
    ) -> list[ShardFileSegment]:
        """Split code using tree-sitter with language-specific rules"""
        try:
            language = detect_language(file_path)
            if not language or language not in self.parsers:
                return self._split_with_pygments(content, file_path, mime_type)

            # Get language-specific config
            lang_config = self.config.language_configs.get(
                language,
                LanguageConfig(),  # Use defaults if not specified
            )

            # Apply rules
            return self._split_with_rules(content, file_path, mime_type, language, lang_config)

        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
            return self._split_with_pygments(content, file_path, mime_type)

    def _split_with_rules(
        self, content: str, file_path: str, mime_type: str, language: str, lang_config: LanguageConfig
    ) -> list[ShardFileSegment]:
        """Split content using language-specific and custom rules"""
        try:
            # Get applicable rules
            rules = self.rule_manager.get_rules_for_language(language)

            # Find all potential split points
            matches = self._find_rule_matches(content, rules, file_path)

            ########################################################################
            # Apply language-specific optimizations
            opt = self.rule_manager.language_optimizer.get_optimization(language)
            if opt:
                # Filter out splits that would break language constructs
                matches = [
                    m
                    for m in matches
                    if not self._would_break_language_construct(m, content, opt)
                ]

                # Add required context
                matches = self._add_language_context(matches, content, opt)
            ########################################################################

            # Sort matches by priority and position
            matches.sort(key=lambda m: (-m.priority, m.start_pos))

            # Create segments based on matches
            return self._create_segments_from_matches(
                content, matches, file_path, mime_type, language, lang_config
            )

        except Exception as e:
            logger.error(f"Error in rule-based splitting for {file_path}: {e}")
            return self._split_with_pygments(content, file_path, mime_type)

    def _would_break_language_construct(
        self, match: RuleMatch, content: str, opt: LanguageOptimization
    ) -> bool:
        """Check if a split would break a language construct"""
        # Check scope depth
        scope_depth = self._calculate_scope_depth(content[: match.start_pos], opt)
        if scope_depth > 0 and scope_depth < opt.max_scope_depth:
            return True

        # Check merge patterns
        line = content[match.start_pos : match.end_pos].strip()
        return any(pattern.match(line) for pattern in opt.merge_patterns)

    def _add_language_context(
        self, matches: list[RuleMatch], content: str, opt: LanguageOptimization
    ) -> list[RuleMatch]:
        """Add required context lines to matches"""
        enhanced_matches = []
        lines = content.splitlines(keepends=True)

        for match in matches:
            context_lines = 0
            line = content[match.start_pos : match.end_pos].strip()

            # Check context patterns
            for pattern, required_context in opt.context_patterns.items():
                if pattern.match(line):
                    context_lines = max(context_lines, required_context)

            if context_lines > 0:
                # Adjust match to include context
                start_line = max(0, match.start_line - context_lines)
                match.start_pos = sum(len(line) for line in lines[:start_line])
                match.start_line = start_line

            enhanced_matches.append(match)

        return enhanced_matches

    def _find_rule_matches(
        self, content: str, rules: list[CustomRule], file_path: str
    ) -> list[RuleMatch]:
        """Find all matches for given rules in content"""
        matches = []
        lines = content.splitlines(keepends=True)

        for rule in rules:
            try:
                for match in rule.compiled_pattern.finditer(content):
                    start_pos = match.start()
                    end_pos = match.end()

                    # Get line numbers
                    start_line = content.count("\n", 0, start_pos)
                    end_line = content.count("\n", 0, end_pos)

                    # Add context lines if specified
                    if rule.min_context_lines > 0:
                        context_start = max(0, start_line - rule.min_context_lines)
                        context_end = min(len(lines), end_line + rule.min_context_lines)
                        context = "".join(lines[context_start:context_end])
                    else:
                        context = match.group(0)

                    # Check custom condition if specified
                    if rule.condition and not rule.condition(context, file_path):
                        continue

                    matches.append(
                        RuleMatch(
                            start_pos=start_pos,
                            end_pos=end_pos,
                            start_line=start_line,
                            end_line=end_line,
                            rule=rule,
                            context=context,
                            priority=rule.priority,
                        )
                    )

            except Exception as e:
                logger.warning(f"Error applying rule {rule.name}: {e}")
                continue

        return matches

    def _create_segments_from_matches(
        self,
        content: str,
        matches: list[RuleMatch],
        file_path: str,
        mime_type: str,
        language: str,
        lang_config: LanguageConfig,
    ) -> list[ShardFileSegment]:
        """Create file segments based on rule matches"""
        if not matches:
            return self._split_by_size(content, file_path, mime_type, lang_config.max_node_size)

        segments = []
        last_end = 0

        for match in matches:
            # Add content before match if it exists and is large enough
            if match.start_pos > last_end:
                pre_content = content[last_end : match.start_pos]
                if len(pre_content) >= lang_config.min_node_size:
                    segments.extend(
                        self._split_by_size(
                            pre_content,
                            file_path,
                            mime_type,
                            match.rule.max_size or lang_config.max_node_size,
                        )
                    )

            # Add matched content
            match_content = content[match.start_pos : match.end_pos]
            if len(match_content) >= lang_config.min_node_size:
                segments.append(
                    ShardFileSegment(
                        file_path=file_path,
                        start_line=match.start_line,
                        end_line=match.end_line,
                        content=match_content,
                        language=language,
                        mime_type=mime_type,
                    )
                )

            last_end = match.end_pos

        # Add remaining content
        if last_end < len(content):
            remaining = content[last_end:]
            if len(remaining) >= lang_config.min_node_size:
                segments.extend(
                    self._split_by_size(remaining, file_path, mime_type, lang_config.max_node_size)
                )

        return segments

    def _split_by_size(
        self, content: str, file_path: str, mime_type: str, max_size: int
    ) -> list[ShardFileSegment]:
        """Split content by size while respecting line boundaries"""
        if len(content) <= max_size:
            return [
                ShardFileSegment(
                    file_path=file_path,
                    start_line=content.count("\n", 0, 0),
                    end_line=content.count("\n", 0, len(content)),
                    content=content,
                    mime_type=mime_type,
                )
            ]

        return self._split_by_lines(content, file_path, mime_type)
