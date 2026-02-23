from __future__ import annotations
import asyncio
import hashlib
from abc import ABC, abstractmethod
from dataclasses import field
from typing import Any, ClassVar
from pydantic import Field
from pathlib import Path
import aiofiles
import time 
import xxhash
from circuitbreaker import circuit
from cachetools import TTLCache

from colony.distributed.caching.simple import CacheConfig
from colony.distributed.config import ConfigComponent, register_polymathera_config
from colony.distributed import get_polymathera
from colony.distributed.metrics.common import BaseMetricsMonitor
from colony.utils import setup_logger

from ..languages.utils import detect_language

logger = setup_logger(__name__)


@register_polymathera_config()
class FileContentCacheConfig(ConfigComponent):
    """Configuration for file content cache"""

    # Performance settings
    max_file_size_to_read_mb: float = Field(default=10.0, ge=0.0, description="Maximum file size to read in MB")
    max_file_size_to_cache_mb: float = Field(default=100.0, ge=0.0, description="Maximum file size to cache in MB")
    cache_ttl: int = Field(default=3600, ge=0, description="Time to live for cached file contents in seconds")
    cache_max_size: int = Field(default=10000, ge=0, description="Maximum number of cached file contents")

    CONFIG_PATH: ClassVar[str] = "llms.sharding.analyzers.file_content_cache"


class FileContentCacheMetricsMonitor(BaseMetricsMonitor):
    """Centralized monitoring for FileContentCache."""

    def __init__(self, enable_http_server: bool = True):
        super().__init__(enable_http_server=enable_http_server, service_name="file-content-cache")

        self.logger.info(f"Initializing FileContentCacheMetricsMonitor instance {id(self)}...")
        self.errors = self.create_counter(
            "file_content_cache_errors_total",
            "Number of errors during analysis",
            labelnames=["error_type"]
        )

        self.operation_duration = self.create_histogram(
            "file_content_cache_operation_duration_seconds",
            "Duration of file content cache operations",
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
            labelnames=["operation_type"]
        )


class FileContentCache:
    """File content cache to avoid re-reading files"""

    def __init__(self, config: FileContentCacheConfig | None = None):
        self.config: FileContentCacheConfig | None = config
        self.cache: TTLCache | None = None
        self.metrics = FileContentCacheMetricsMonitor()
        self._file_read_breaker = circuit(
            failure_threshold=4, recovery_timeout=20, name="file_read"
        )

    async def initialize(self):
        self.config = await FileContentCacheConfig.check_or_get_component(self.config)
        self.cache = TTLCache(
            maxsize=self.config.cache_max_size,
            ttl=self.config.cache_ttl,
            getsizeof=lambda x: len(x) if x is not None else 0,
        )

    async def cleanup(self):
        """Cleanup any resources used by the cache"""
        self.cache.clear()

    async def read_file(self, file_path: str) -> str | None:
        """Read file content with caching and error handling"""
        entry = await self._get_cache_entry(file_path)
        if entry is not None:
            return entry["content"]
        return None

    async def _read_file(self, file_path: str) -> str:
        """Read file content with metrics"""
        start_time = time.time()
        try:
            p = Path(file_path)
            if not p.is_file():
                logger.debug(f"Skipping non-file path: {file_path}")
                return None
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            ### with open(file_path, 'r', encoding='utf-8') as f:
            ###     content = f.read()
            return content
        except UnicodeDecodeError:
            # Try with latin-1 encoding as fallback
            ### with open(file_path, 'r', encoding='latin-1') as f:
            ###     content = f.read()
            logger.warning(f"Binary file skipped: {file_path}")
            self.metrics.errors.labels("encoding").inc()
            return None
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
            self.metrics.errors.labels("read").inc()
            return None
        finally:
            self.metrics.operation_duration.labels("file_read").observe(
                time.time() - start_time
            )

    async def get_file_hash(self, file_path: str) -> str | None:
        """Get file hash for file path"""
        entry = await self._get_cache_entry(file_path)
        if entry is not None:
            return entry["content_hash"]
        return None

    async def _get_cache_entry(self, file_path: str) -> dict[str, str] | None:
        """Get cache entry for file path"""
        try:
            # Check cache first
            if file_path not in self.cache:

                path = Path(file_path)
                if not path.exists():
                    logger.debug(f"File not found: {file_path}")
                    return None

                # Check file size
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb > self.config.max_file_size_to_read_mb:
                    logger.warning(f"File too large: {file_path} ({size_mb:.2f}MB)")
                    self.metrics.errors.labels("file_size").inc()
                    return None

                # Read file with proper encoding detection and circuit breaker if not in cache
                content = await self._file_read_breaker(self._read_file)(file_path)
                if content is None:
                    return None

                # Cache the content (with size limit to prevent memory issues)
                if len(content) < self.config.max_file_size_to_cache_mb * 1024 * 1024:
                    ### content_hash = xxhash.xxh64(content.encode()).hexdigest()
                    content_hash = hashlib.sha256(content.encode()).hexdigest()  # TODO: How to handle large files? How fast is this?

                    self.cache[file_path] = {
                        "content": content,
                        "content_hash": content_hash,
                    }

            return self.cache[file_path]

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            self.metrics.errors.labels("read").inc()
            return None



# DO NOT REGISTER THIS CONFIG - Register subclasses instead
class AnalyzerConfig(ConfigComponent):
    """Base configuration for all analyzers"""

    # Concurrency settings
    max_concurrent_files: int = 4
    timeout_seconds: int = 30
    batch_size: int = 32

    # Performance settings
    skip_large_files: bool = True

    # Language settings
    language_configs: dict[str, dict] = field(default_factory=dict)
    enable_cross_language: bool = True

    # Cache settings
    results_cache_config: CacheConfig | None = Field(default=None) # If None, the default will be automatically loaded from config manager


class BaseAnalyzerMetricsMonitor(BaseMetricsMonitor):
    """Base class for Prometheus metrics monitoring using node-global HTTP server."""

    def __init__(self,
                 analyzer_type: str,
                 enable_http_server: bool = True,
                 service_name: str = "service"):
        super().__init__(enable_http_server, service_name)
        self.analyzer_type = analyzer_type

        self.logger.info(f"Initializing BaseAnalyzerMetricsMonitor instance {id(self)}...")
        self.analysis_duration = self.create_histogram(
            f"{self.analyzer_type}_analysis_seconds",
            f"Time spent in {self.analyzer_type} analysis",
            ["language"],
        )
        self.cache_hits = self.create_counter(
            f"{self.analyzer_type}_cache_hits_total",
            "Number of cache hits",
            #labelnames=["language"]
        )
        self.cache_misses = self.create_counter(
            f"{self.analyzer_type}_cache_misses_total",
            "Number of cache misses"
        )
        self.cache_ops = self.create_counter(
            f"{self.analyzer_type}_cache_operations_total",
            "Cache operations",
            labelnames=["operation"]
        )
        self.cache_errors = self.create_counter(
            f"{self.analyzer_type}_cache_errors_total",
            "Number of cache errors",
            labelnames=["operation", "error_type"]
        )
        self.errors = self.create_counter(
            f"{self.analyzer_type}_errors_total",
            "Number of errors during analysis",
            labelnames=["error_type"]
        )
        self.active_analyzers = self.create_gauge(
            f"{self.analyzer_type}_active_analyzers",
            "Number of active analyzer instances",
            labelnames=["language"]
        )


class LocalResultsCache:
    """Local results cache for testing"""
    def __init__(self):
        self.cache = {}

    async def get(self, key: str) -> dict[str, Any] | None:
        return self.cache.get(key)

    async def set(self, key: str, value: dict[str, Any]):
        self.cache[key] = value

    async def cleanup(self):
        pass


class BaseAnalyzer(ABC):
    """Base class for all analyzers with shared functionality"""

    def __init__(self,
                 analyzer_type: str,
                 file_content_cache: FileContentCache):
        self.analyzer_type = analyzer_type
        self.config = None
        self.file_content_cache = file_content_cache
        self.semaphore = None
        self.results_cache = None
        self.base_metrics = BaseAnalyzerMetricsMonitor(self.analyzer_type)
        self._cache_initialized = False

        # Setup circuit breakers
        self._setup_circuit_breakers()

    async def initialize(self):
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_files)
        if self.config.running_locally():
            self.results_cache = LocalResultsCache()
        else:
            self.results_cache = await get_polymathera().create_distributed_simple_cache(
                namespace=f"llms:sharding:analyzers:{self.analyzer_type}",  # TODO: Does this need to be VMR-specific?
                config=self.config.results_cache_config,
            )

    def _setup_circuit_breakers(self):
        """Setup circuit breakers for critical operations"""
        self._cache_breaker = circuit(
            failure_threshold=3,
            recovery_timeout=15,
            name=f"{self.analyzer_type}_results_cache_breaker"
        )

    async def _get_cached_result(
        self, cache_key: str
    ) -> dict[str, Any] | None:
        """Get cached result if available"""
        try:
            cached_data = await self._cache_breaker(self.results_cache.get)(cache_key)
            if cached_data:
                self.base_metrics.cache_hits.inc()
                return cached_data
            else:
                self.base_metrics.cache_misses.inc()
                return None
        except Exception as e:
            logger.error(f"Error retrieving cached result: {e}", exc_info=True)
            self.base_metrics.cache_misses.inc()
            return None

    async def cleanup(self):
        """Cleanup any resources used by the analyzer"""
        from colony.utils import cleanup_dynamic_asyncio_tasks

        try:
            await cleanup_dynamic_asyncio_tasks(self, raise_exceptions=False)
            await self.results_cache.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up analyzer cache: {e}", exc_info=True)

    async def analyze_file(
        self, file_path: str, content: str | None = None, language: str | None = None, **kwargs
    ) -> dict[str, Any]:
        """Analyze a single file with caching and metrics"""
        try:
            async with self.semaphore:
                self.base_metrics.active_analyzers.labels(language).inc()

                # Get file content and language
                if not content:
                    content = await self.file_content_cache.read_file(file_path)

                if not language:
                    language = detect_language(file_path)

                if not content or not language:
                    return self._get_fallback_result()

                # Check cache first
                content_hash = await self.file_content_cache.get_file_hash(file_path)
                cache_key = self._make_cache_key(file_path, content_hash, language=language, **kwargs)
                cached_result = await self._get_cached_result(cache_key)

                if cached_result is not None:
                    return cached_result

                # Perform analysis with timeout
                with self.base_metrics.analysis_duration.labels(language).time():
                    result = await asyncio.wait_for(
                        self._analyze_file_impl(file_path, content, language, **kwargs),
                        timeout=self.config.timeout_seconds,
                    )

                await self._cache_breaker(self.results_cache.set)(cache_key, result)

                return result

        except asyncio.TimeoutError:
            logger.error(f"Analysis timeout for {file_path}")
            self.base_metrics.errors.labels("timeout").inc()
            return self._get_fallback_result()

        except Exception as e:
            logger.error(f"Analysis error for {file_path}: {e}", exc_info=True)
            self.base_metrics.errors.labels("analysis").inc()
            return self._get_fallback_result()

        finally:
            self.base_metrics.active_analyzers.labels(language).dec()

    @abstractmethod
    async def _analyze_file_impl(
        self, file_path: str, content: str, language: str | None = None, **kwargs
    ) -> dict[str, Any]:
        """Implement actual file analysis logic"""
        pass

    @abstractmethod
    def _get_fallback_result(self) -> dict[str, Any]:
        """Return safe fallback result on error"""
        pass

    def _make_cache_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate cache key for analysis results"""
        key_parts = list(args)
        key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        return ":".join(key_parts)
