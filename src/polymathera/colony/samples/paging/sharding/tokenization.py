from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, ClassVar

import aiofiles
import tiktoken
from circuitbreaker import circuit
from pydantic import Field
from tenacity import retry_if_exception_type

from polymathera.colony.distributed.caching.simple import CacheConfig, DistributedSimpleCache
from polymathera.colony.distributed.config import ConfigComponent, register_polymathera_config
from polymathera.colony.distributed import get_polymathera
from polymathera.colony.distributed.ray_utils import serving
from polymathera.colony.distributed.metrics.common import BaseMetricsMonitor
from polymathera.colony.utils.retry import standard_retry
from polymathera.colony.utils import run_method_once, setup_logger

from .languages.utils import detect_language
from .analyzers.base import FileContentCache

logger = setup_logger(__name__)


class TokenizationStrategy(Enum):
    LAZY = "lazy"  # Tokenize on demand
    EAGER = "eager"  # Tokenize immediately
    HYBRID = "hybrid"  # Smart caching based on usage patterns
    DISTRIBUTED = "distributed"  # Use distributed cache


@register_polymathera_config()
class TokenizationConfig(ConfigComponent):
    """Configuration for tokenization system"""

    strategy: TokenizationStrategy = TokenizationStrategy.HYBRID

    # Performance settings
    batch_size: int = 1000
    max_concurrent_tokenization: int = 10
    timeout_seconds: int = 30

    # Storage settings
    storage_backend: str = "postgres"  # postgres, redis, or filesystem
    storage_connection: str = ""

    # Cost control
    max_tokens_per_file: int = 100000
    skip_large_files: bool = True

    # Cache config
    token_cache_config: CacheConfig | None = Field(
        default=CacheConfig(
            enable_caching=True,
            base_ttl=3600,
            compression_level=3,
        )
    )  # If None, the default will be automatically loaded from config manager
    count_cache_config: CacheConfig | None = Field(
        default=CacheConfig(
            enable_caching=True,
            base_ttl=3600,
            compression_level=3,
        )
    )  # If None, the default will be automatically loaded from config manager

    CONFIG_PATH: ClassVar[str] = "llms.sharding.tokenization"



# TODO:
# 1. Add more sophisticated caching strategies:
#    - Predictive caching
#    - Cache warming
#    - Cache eviction policies
#
# 2. Implement distributed caching:
#    - Consistent hashing
#    - Cache synchronization
#    - Failure recovery
#
# 3. Add more storage backends:
#    - Cloud storage (S3, GCS)
#    - Distributed filesystems
#    - Custom solutions
#
# 4. Improve performance:
#    - Batch processing
#    - Streaming tokenization
#    - Parallel compression
#
# 5. Add monitoring:
#    - Cache hit rates
#    - Storage usage
#    - Tokenization latency
#    - Error rates
#
# 6. Implement cost control:
#    - Token budget per repo
#    - Storage quotas
#    - Rate limiting

class TokenManagerMetricsMonitor(BaseMetricsMonitor):
    """Centralized monitoring for TokenManager."""

    def __init__(self, enable_http_server: bool = True):
        super().__init__(enable_http_server=enable_http_server, service_name="token-manager")

        self.logger.info(f"Initializing TokenManagerMetricsMonitor instance {id(self)}...")

        self.token_counts = self.create_histogram(
            "token_manager_file_token_counts",
            "Distribution of file token counts",
            buckets=[100, 500, 1000, 2000, 5000, 10000, 20000, 50000],
            labelnames=["language"]
        )
        self.operation_duration = self.create_histogram(
            "token_manager_operation_duration_seconds",
            "Duration of token operations",
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
            labelnames=["operation_type"]
        )

        self.batch_size = self.create_histogram(
            "token_manager_batch_size",
            "Size of batch operations",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
            labelnames=["operation_type"]
        )
        self.errors = self.create_counter(
            "token_manager_errors_total",
            "Number of token manager errors",
            labelnames=["error_type", "operation"]
        )


class TokenManager:
    """High-level token management interface for FileGrouper

    This FileGrouper and TokenManager run as part of the GitRepoInferenceEngine which
    may be deployed and undeployed as a Ray actor many times as part of a processing
    loop that iterates over 1000s or even millions of git repos in multiple passes.
    A major rquirement for caching is that it survives deployment/undeployment cycles.
    Designed to work with Ray actors and persist cache across deployment cycles
    by leveraging DistributedSimpleCache with appropriate backends:
    - PostgreSQL for production on AWS
    - Redis for production alternatives
    - Filesystem for local development
    - LocalLRU for testing
    """

    def __init__(
        self,
        file_content_cache: FileContentCache,
        config: TokenizationConfig | None = None,
    ):
        self.config: TokenizationConfig | None = config
        self.file_content_cache = file_content_cache

        # Use DistributedSimpleCache with "tokens" type for persistence
        # Separate caches for tokens and counts
        # TODO: Add cache namespaces
        self.token_cache: DistributedSimpleCache | None = None
        self.count_cache: DistributedSimpleCache | None = None

        self._encoder = None
        self.semaphore = None

        # Setup metrics and circuit breakers
        self.metrics = TokenManagerMetricsMonitor()
        self._setup_circuit_breakers()

    async def initialize(self):
        self.config = await TokenizationConfig.check_or_get_component(self.config)

        # Use DistributedSimpleCache with "tokens" type for persistence
        # Separate caches for tokens and counts
        self.token_cache = (
            await get_polymathera().create_distributed_simple_cache(
                namespace="tokens",
                config=self.config.token_cache_config,
            )
        )
        self.count_cache = (
            await get_polymathera().create_distributed_simple_cache(
                namespace="token_counts",
                config=self.config.count_cache_config,
            )
        )
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_tokenization)

    async def cleanup(self):
        # Cleanup: properly close the caches to stop monitoring tasks
        try:
            await self.token_cache.cleanup()
            await self.count_cache.cleanup()
        except Exception as e:
            logger.warning(f"Failed to cleanup token manager caches: {e}")

    def _setup_circuit_breakers(self):
        """Setup circuit breakers for critical operations"""
        self._tokenization_breaker = circuit(
            failure_threshold=3, recovery_timeout=15, name="tokenization"
        )

    async def get_file_token_count(
        self,
        file_path: str,
        commit_hash: str | None = None,
    ) -> int:
        """Get token count for a file with persistent caching and fallbacks"""
        return await self.get_file_tokens(
            file_path, commit_hash, return_tokens=False
        )

    @run_method_once
    def _warn_file_too_large(self, extension: str) -> None:
        """Warn if a file is too large"""
        logger.warning(f"File too large with extension: {extension}")

    @standard_retry(logger, retry=retry_if_exception_type(IOError))
    async def get_file_tokens(
        self,
        file_path: str,
        commit_hash: str | None = None,
        return_tokens: bool = False,
    ) -> int | list[int]:
        """Get token count or tokens for a file with persistent caching and fallbacks"""
        ### if not hasattr(self, "_file_count"):
        ###     self._file_count = 0
        ### self._file_count += 1
        ### logger.info(f"________ get_file_tokens({commit_hash}:{file_path}) [{self._file_count}:{self.file_content_cache.cache.currsize}]")
        start_time = time.time()
        try:
            cache_key = self._make_cache_key(file_path, commit_hash)

            if return_tokens:
                # Try getting tokens from cache
                cached_tokens = await self.token_cache.get(cache_key)
                if cached_tokens is not None:
                    return [int(t) for t in cached_tokens.split(",")]
            else:
                # Try getting count from cache
                cached_count = await self.count_cache.get(cache_key)
                if cached_count is not None:
                    return int(cached_count)

            # Read file with circuit breaker and tokenize file if not in cache
            content = await self.file_content_cache.read_file(file_path)
            if content is None:
                return [] if return_tokens else 0

            # Get tokens with circuit breaker
            try:
                tokens = await self._tokenization_breaker(self._get_tokens)(content)
            except Exception as e:
                logger.error(f"Tokenization error: {e}")
                return (
                    [] if return_tokens else len(content) // 4
                )  # TODO: Fallback approximation

            token_count = len(tokens)

            # Cache the count (and tokens) BEFORE the size check so that
            # subsequent calls for the same file hit the cache instead of
            # re-tokenizing and re-raising every time.
            await self.token_cache.set(cache_key, ",".join(str(t) for t in tokens))
            await self.count_cache.set(cache_key, str(token_count))

            if (
                token_count > self.config.max_tokens_per_file
                and self.config.skip_large_files
            ):
                self._warn_file_too_large(file_path.split(".")[-1])
                return [] if return_tokens else 0
                # raise ValueError(f"File too large: {file_path} ({token_count} tokens)")

            # Update metrics
            self.metrics.token_counts.labels(detect_language(file_path)).observe(
                token_count
            )

            return tokens if return_tokens else token_count

        except Exception as e:
            logger.error(f"Error getting file tokens: {e}", exc_info=True)
            self.metrics.errors.labels(
                error_type=type(e).__name__, operation="get_tokens"
            ).inc()
            return [] if return_tokens else 0
        finally:
            self.metrics.operation_duration.labels("get_tokens").observe(
                time.time() - start_time
            )

            logger.debug(
                "TokenManager.get_file_tokens: done path=%s elapsed=%.2fs",
                file_path,
                time.time() - start_time,
            )

    async def _get_tokens(self, content: str) -> list[int]:
        """Get tokens using tiktoken"""
        async with self.semaphore:  # Limit concurrent tokenizations
            start_time = time.time()
            try:
                if self._encoder is None:
                    self._encoder = tiktoken.get_encoding("cl100k_base")
                # Disable special token checking to handle tokens like <|endoftext|> as normal text
                tokens = self._encoder.encode(content, disallowed_special=())
                logger.debug(f"________ _get_tokens: tokens: {tokens[:10]}...")
                return tokens
            finally:
                self.metrics.operation_duration.labels("tokenize").observe(
                    time.time() - start_time
                )

    async def get_files_batch(
        self,
        files: list[str],
        commit_hash: str | None = None,
    ) -> dict[str, int]:
        """Process multiple files efficiently using DistributedSimpleCache batching"""
        start_time = time.time()
        try:
            self.metrics.batch_size.labels("get_tokens").observe(len(files))

            # Generate cache keys for batch
            key_to_file = {
                self._make_cache_key(f, commit_hash): f for f in files
            }

            # Get cached values in batch
            cached_results = await self.count_cache.get_batch(list(key_to_file.keys()))

            # Process cache misses
            missing_files = {
                f for k, f in key_to_file.items() if cached_results.get(k) is None
            }

            if missing_files:
                new_results = {}
                for f in missing_files:
                    new_results[f] = await self.get_file_tokens(f, commit_hash)

                return {
                    **{
                        key_to_file[k]: int(v)
                        for k, v in cached_results.items()
                        if v is not None
                    },
                    **new_results,
                }
            else:
                return {key_to_file[k]: int(v) for k, v in cached_results.items()}

        except Exception as e:
            logger.error(f"Batch processing error: {e}", exc_info=True)
            self.metrics.errors.labels(
                error_type=type(e).__name__, operation="get_batch"
            ).inc()
            return {f: 0 for f in files}
        finally:
            self.metrics.operation_duration.labels("get_batch").observe(
                time.time() - start_time
            )

    def _make_cache_key(
        self, file_path: str, commit_hash: str | None
    ) -> str:
        """Generate consistent cache key"""
        return f"{file_path}:{commit_hash or 'latest'}"
