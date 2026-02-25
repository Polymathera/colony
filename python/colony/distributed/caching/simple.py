from __future__ import annotations

import asyncio
import builtins
import json
import logging
import pickle
import time
import uuid
import zlib
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, Literal, TypeVar

import msgpack
import numpy as np
from circuitbreaker import circuit
from redis.exceptions import RedisError

try:
    import zstandard as zstd
except ImportError:
    zstd = None

from ..config import ConfigComponent, register_polymathera_config
from ..metrics.cache import (
    CACHE_BATCH_SIZE,
    CACHE_COMPRESSION_RATIO,
    CACHE_ENTRY_AGE,
    CACHE_ERRORS,
    CACHE_EVICTIONS,
    CACHE_HEALTH,
    CACHE_HITS,
    CACHE_ITEM_SERIALIZATION_TIME,
    CACHE_ITEM_SIZE,
    CACHE_LATENCY,
    CACHE_MEMORY,
    CACHE_MISSES,
    CACHE_OPERATIONS,
    CACHE_SIZE,
)
from ..redis_utils import RedisClient
from ...utils.retry import standard_retry

logger = logging.getLogger(__name__)


retry_policy = standard_retry(logger)


T = TypeVar("T")

"""
TODO: Plan the caching hierarchy: Multi-level caching
- L1: Ray object store (in-memory)
- L2: Redis cluster
"""


class MultiLevelDistributedCache:
    """
    Multi-level caching using Ray's object store and Redis
    """

    def __init__(self, l1_cache: Any, l2_cache: Any):
        self.l1_cache = l1_cache
        self.l2_cache = l2_cache

    async def get(self, key: str) -> Any | None:
        if value := await self.l1_cache.get(key):
            return value

        if value := await self.l2_cache.get(key):
            await self.l1_cache.set(key, value)
            return value

        return None


@register_polymathera_config()
class CacheConfig(ConfigComponent):

    """Configuration for caching system"""

    enable_caching: bool = True
    max_concurrent_ops: int = 10
    enable_compression: bool = True
    batch_size: int = 100
    max_size_mb: int = 10000
    enable_adaptive_ttl: bool = False
    base_ttl: int = 86400  # 24 hours
    min_ttl: int = 3600  # 1 hour
    max_ttl: int = 86400  # 24 hours
    use_zstd: bool = True
    compression_level: int = 3
    serialization_format: Literal["json", "numpy", "pickle", "msgpack"] = "pickle"

    ### enable_versioning: bool = True
    ### version_retention_days: int = 30
    ### # Storage settings inherited from CacheConfig
    ### storage_backend: str = "postgres"  # postgres, redis, or filesystem
    ### storage_connection: str = ""

    CONFIG_PATH: ClassVar[str] = "caching.simple_cache"


class DistributedSimpleCache:
    """Redis-based distributed cache implementation with metadata support and adaptive TTL based on access patterns

    Handles caching of any serializable data with namespacing support.

    This class provides a clean abstraction over Redis operations with:
    1. Namespace isolation
    2. TTL support
    3. Metadata attachment (not implemented)
    4. Proper cleanup
    5. Metrics collection
    6. Scan-based iteration

    Key features:
    1. Namespace support to prevent key collisions
    2. Configurable TTL per cache instance
    3. Error handling and logging
    4. Generic type support
    5. Serialization handling
    6. Metadata support for rich caching
    7. Connection pooling for high concurrency
    8. Atomic operations support
    9. Set operations support
    10. Proper cleanup

    Implementation notes:
    - Uses Redis connection pooling to handle high concurrency efficiently
    - Pool size is configurable via config["redis_pool_size"]
    - Each operation gets a connection from the pool and returns it after use
    - Supports atomic operations for distributed locking
    - Maintains Redis semantics while providing a clean interface

    TODO: Collect metrics by namespace
    """

    def __init__(
        self,
        namespace: str,
        *,
        redis_client: RedisClient,
        labelnames: list[str] = [],
        config: CacheConfig | None = None,
    ):
        """Initialize the cache

        Args:
            namespace: Namespace to prevent key collisions
            redis_client: Redis client instance
            labelnames: List of label names for metrics
            config: Configuration for the cache
        """
        self.namespace = namespace
        self.instance_id = str(uuid.uuid4())[
            :8
        ]  # Use first 8 chars of UUID for readability
        logger.info(
            f"Creating cache instance {self.instance_id} for namespace {namespace}"
        )
        self.client = redis_client
        self.labelnames = labelnames
        self.config = config
        self.stats_key = self._build_namespaced_key("cache:stats")

        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(minutes=5)
        self.stats = defaultdict(int)
        self._monitoring_task = None
        self._shutdown_event = asyncio.Event()

        # Cache metrics
        self.metrics = None

    async def initialize(self):
        """Initialize the cache"""
        self.config = await CacheConfig.check_or_get_component(self.config)
        self.metrics = self._setup_metrics()
        self._start_monitoring_task()
        self._setup_circuit_breakers()

    def _build_namespaced_key(self, key: str) -> str:
        """Build fully qualified Redis key with namespace

        Args:
            key: User-provided key

        Returns:
            Namespaced Redis key
        """
        return f"{self.namespace}:{key}"

    def _build_cache_entry(
        self, value: T, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Build a cache entry combining value and metadata"""
        return {
            "value": value,
            "metadata": metadata or {},
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

    def _metadata_matches(
        self, stored_metadata: dict[str, Any], query_metadata: dict[str, Any]
    ) -> bool:
        """Check if stored metadata matches query metadata"""
        if not query_metadata:
            return True
        return all(
            key in stored_metadata and stored_metadata[key] == value
            for key, value in query_metadata.items()
        )

    def _setup_circuit_breakers(self):
        """Setup circuit breakers for critical operations"""
        self._storage_breaker = circuit(
            failure_threshold=5,
            recovery_timeout=30,
            name="distributed_simple_cache_storage",
        )
        self._compression_breaker = circuit(
            failure_threshold=3,
            recovery_timeout=15,
            name="distributed_simple_cache_compression",
        )
        self._batch_breaker = circuit(
            failure_threshold=4,
            recovery_timeout=20,
            name="distributed_simple_cache_batch",
        )

    def _format_extras(self, labels: dict[str, Any] = {}) -> str:
        """Format labels into a structured extras string for metrics.

        Args:
            labels: User-provided labels dictionary

        Returns:
            JSON string containing all labels, or empty string if no labels
        """
        if not labels:
            return ""
        try:
            # Filter out None values and convert all values to strings
            filtered_labels = {
                str(k): str(v) for k, v in labels.items() if v is not None
            }
            if filtered_labels:
                return json.dumps(filtered_labels, sort_keys=True)
            return ""
        except Exception as e:
            logger.error(f"Error formatting metric extras: {e}")
            return ""

    def _compress(self, value: Any, labels: dict[str, Any] = {}) -> bytes:
        if self.config.enable_compression:
            original_size = len(str(value).encode())
            if self.config.use_zstd and zstd is not None:
                compression = zstd.ZstdCompressor(level=self.config.compression_level)
                compressed = compression.compress(value)
            else:
                compressed = zlib.compress(value)

            ratio = original_size / len(compressed)
            self.metrics["compression_ratio"].labels(
                namespace=self.namespace,
                operation="compress",
                extras=self._format_extras(labels),
            ).observe(ratio)
            return compressed

        return value

    def _decompress(self, value: bytes) -> Any:
        if self.config.enable_compression:
            if self.config.use_zstd and zstd is not None:
                decompression = zstd.ZstdDecompressor()
                return decompression.decompress(value)
            else:
                return zlib.decompress(value)
        return value

    @staticmethod
    def _json_default(obj: Any) -> Any:
        """Handle non-JSON-serializable types (e.g., set → list)."""
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _serialize(self, data: Any) -> bytes | None:
        """Serialize value with proper error handling

        Args:
            data: Data to serialize

        Returns:
            Serialized bytes or None if serialization fails
        """
        with self.metrics["serialization_time"].labels(
            namespace=self.namespace, operation="serialize", extras=""
        ).time():
            try:
                if self.config.serialization_format == "json":
                    serialized = json.dumps(data, default=self._json_default).encode()
                elif self.config.serialization_format == "numpy":
                    if not isinstance(data, np.ndarray):
                        logger.error("Data is not a numpy array")
                        self.metrics["errors"].labels(
                            namespace=self.namespace,
                            operation="serialization",
                            extras="",
                        ).inc()
                        return None
                    serialized = data.tobytes()
                elif self.config.serialization_format == "msgpack":
                    serialized = msgpack.packb(data)
                else:
                    serialized = pickle.dumps(data)

                return serialized

            except (
                TypeError,
                ValueError,
                pickle.PickleError,
                msgpack.PackException,
            ) as e:
                logger.error(f"Serialization error: {e}")
                self.metrics["errors"].labels(
                    namespace=self.namespace, operation="serialization", extras=""
                ).inc()
                return None
            except Exception as e:
                logger.error(f"Unexpected serialization error: {e}")
                self.metrics["errors"].labels(
                    namespace=self.namespace, operation="serialization", extras=""
                ).inc()
                return None

    def _deserialize(self, data: bytes) -> Any | None:
        """Deserialize data"""
        with self.metrics["serialization_time"].labels(
            namespace=self.namespace, operation="deserialize", extras=""
        ).time():
            if self.config.serialization_format == "json":
                return json.loads(data.decode())  # Return dict
            elif self.config.serialization_format == "numpy":
                return np.frombuffer(data, dtype=np.float32)  # Return numpy array
            elif self.config.serialization_format == "msgpack":
                return msgpack.unpackb(data)
            else:
                return pickle.loads(data)  # Return object

    @asynccontextmanager
    async def pipeline(self) -> AsyncIterator[Any]:
        """Get an implementation-specific pipeline for batch operations."""
        async with self.client.get_pipeline() as (pipe, _):
            yield pipe

    @retry_policy
    async def execute_pipeline(self, pipe: Any) -> list[Any]:
        try:
            return await pipe.execute()
        except Exception as e:
            logger.error(f"Redis execute pipeline error: {e}")
            raise e

    @retry_policy
    async def acquire_lock(self, key: str, ttl: timedelta) -> bool:
        """
        Atomically acquire a lock using SET NX

        Args:
            key: Lock key
            ttl: Lock timeout

        Returns:
            True if lock acquired, False otherwise
        """
        try:
            async with self.client.get_redis_connection() as redis:
                return await redis.set(
                    self._build_namespaced_key(f"locks:{key}"),
                    "1",
                    ex=int(ttl.total_seconds()),
                    nx=True,  # Set only if key does not exist
                )
        except Exception as e:
            logger.error(f"Failed to acquire lock in cache {self.namespace}: {e}")
            return False

    @retry_policy
    async def release_lock(self, key: str):
        """Release a lock by deleting the key"""
        async with self.client.get_redis_connection() as redis:
            await redis.delete(self._build_namespaced_key(f"locks:{key}"))

    @retry_policy
    async def get(
        self,
        key: str,
        labels: dict[str, Any] = {},
    ) -> Any | None:
        if not self.config.enable_caching:
            return None

        start_time = time.time()
        extras = self._format_extras(labels)
        try:
            namespaced_key = self._build_namespaced_key(key)
            async with self.client.get_redis_connection() as redis:
                value = await redis.get(namespaced_key)
                if not value:
                    self.metrics["misses"].labels(
                        namespace=self.namespace, operation="get", extras=extras
                    ).inc()
                    return None
                self.metrics["hits"].labels(
                    namespace=self.namespace, operation="get", extras=extras
                ).inc()
                await self._update_access_stats(key)
                value = self._decompress(value)
                return self._deserialize(value)
        except (RedisError, ConnectionError) as e:
            logger.warning(f"Redis error in DistributedSimpleCache.get: {e!s}")
            self.metrics["errors"].labels(
                namespace=self.namespace, operation="get", extras=extras
            ).inc()
            raise  # Re-raise to trigger circuit breaker
        except pickle.PickleError as e:
            logger.warning(f"Pickle error in DistributedSimpleCache.get: {e!s}")
            self.metrics["errors"].labels(
                namespace=self.namespace, operation="deserialization", extras=extras
            ).inc()
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.metrics["errors"].labels(
                namespace=self.namespace, operation="get", extras=extras
            ).inc()
            return None
        finally:
            self.metrics["latency"].labels(
                namespace=self.namespace, operation="get", extras=extras
            ).observe(time.time() - start_time)

    @retry_policy
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        pipe: Any | None = None,
        labels: dict[str, Any] = {},
    ) -> bool:
        if not self.config.enable_caching:
            return False
        start_time = time.time()
        extras = self._format_extras(labels)
        try:
            self.metrics["operations"].labels(
                namespace=self.namespace,
                operation="set",
                status="attempt",
                extras=extras,
            ).inc()
            namespaced_key = self._build_namespaced_key(key)
            # Serialize value
            value_bytes = self._serialize(value)
            if not value_bytes:
                logger.error("Serialization failed")
                self.metrics["errors"].labels(
                    namespace=self.namespace, operation="serialization", extras=extras
                ).inc()
                return False

            # Check size limit on serialized data before compression
            value_size_mb = len(value_bytes) / (1024 * 1024)  # Convert to MB
            if value_size_mb > self.config.max_size_mb:
                logger.error(
                    f"Serialized value size ({value_size_mb:.2f}MB) exceeds limit ({self.config.max_size_mb}MB)"
                )
                self.metrics["errors"].labels(
                    namespace=self.namespace,
                    operation="size_limit_exceeded",
                    extras=extras,
                ).inc()
                return False
            # Compress if enabled
            compressed_bytes = self._compress(value_bytes, labels)
            if not compressed_bytes:
                logger.error("Compression failed")
                self.metrics["errors"].labels(
                    namespace=self.namespace, operation="compression", extras=extras
                ).inc()
                return False
            if ttl is None:
                ttl = await self._calculate_ttl(key)

            # Store the compressed value
            try:
                if pipe:
                    if ttl:
                        pipe.setex(namespaced_key, ttl, compressed_bytes)
                    else:
                        pipe.set(namespaced_key, compressed_bytes)
                else:
                    async with self.client.get_redis_connection() as redis:
                        if ttl:
                            await redis.setex(namespaced_key, ttl, compressed_bytes)
                        else:
                            await redis.set(namespaced_key, compressed_bytes)

                # Record successful operation
                self.metrics["operations"].labels(
                    namespace=self.namespace,
                    operation="set",
                    status="success",
                    extras=extras,
                ).inc()
                self.metrics["size"].labels(
                    namespace=self.namespace, operation="set", extras=extras
                ).inc(len(compressed_bytes))
                return True

            except (RedisError, ConnectionError) as e:
                logger.warning(f"Redis error in set operation: {e}")
                self.metrics["errors"].labels(
                    namespace=self.namespace, operation="set", extras=extras
                ).inc()
                raise  # Re-raise to trigger retry and circuit breaker

        except Exception as e:
            logger.error(f"Redis set error: {e}")
            self.metrics["errors"].labels(
                namespace=self.namespace, operation="set", extras=extras
            ).inc()
            return False
        finally:
            self.metrics["latency"].labels(
                namespace=self.namespace, operation="set", extras=extras
            ).observe(time.time() - start_time)

    @retry_policy
    async def delete(self, key: str, pipe: Any | None = None) -> None:
        if not self.config.enable_caching:
            return

        try:
            namespaced_key = self._build_namespaced_key(key)
            if pipe:
                pipe.delete(namespaced_key)
            else:
                async with self.client.get_redis_connection() as redis:
                    await redis.delete(namespaced_key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")

    @retry_policy
    async def clear(self, pipe: Any | None = None) -> None:
        """Delete all keys in the current namespace"""
        if not self.config.enable_caching:
            return

        try:
            namespaced_pattern = self._build_namespaced_key("*")
            # Get all keys matching the namespace pattern
            async with self.client.get_redis_connection() as redis:
                keys = await redis.keys(namespaced_pattern)
            if keys:
                if pipe:
                    # pipe.flushdb() # This will delete all keys in all namespaces
                    pipe.delete(*keys)
                else:
                    async with self.client.get_redis_connection() as redis:
                        # await redis.flushdb() # This will delete all keys in all namespaces
                        await redis.delete(*keys)
                self.metrics["size"].labels(self.namespace).set(0)
        except RedisError as e:
            logger.warning(f"Redis error in DistributedSimpleCache.clear: {e!s}")
            self.metrics["errors"].labels("redis", "clear").inc()
        except Exception as e:
            logger.error(f"Redis clear error: {e!s}", exc_info=True)
            self.metrics["errors"].labels(self.namespace, "clear").inc()

    @retry_policy
    async def cleanup(self) -> None:
        """Cleanup all resources for this namespace"""
        try:
            # First stop the monitoring task
            if self._monitoring_task is not None:
                # Signal shutdown
                self._shutdown_event.set()

                # Give the task a chance to exit gracefully
                try:
                    await asyncio.wait_for(self._monitoring_task, timeout=2)
                except asyncio.TimeoutError:
                    logger.warning(f"Monitoring task for cache {self.instance_id} did not exit gracefully, forcing cancellation")
                    # Force cancel if it doesn't exit gracefully
                    self._monitoring_task.cancel()
                    try:
                        await self._monitoring_task
                    except (asyncio.CancelledError, Exception) as e:
                        logger.debug(f"Monitoring task cancelled: {e}")
                finally:
                    self._monitoring_task = None

            # Then clean up Redis resources
            async with self.client.get_redis_connection() as redis:
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=f"{self.namespace}:*", count=100
                    )
                    if keys:
                        await redis.delete(*keys)
                    if cursor == 0:
                        break
        except RedisError as e:
            logger.error(f"Cache cleanup failed: {e!s}")
        except Exception as e:
            logger.error(f"Unexpected error during cleanup: {e}")

    @retry_policy
    async def get_batch_pattern(self, key_pattern: str) -> dict[str, Any]:
        """Get all values for keys matching a pattern

        Args:
            key_pattern: Pattern to match against keys (will be prefixed with namespace)

        Returns:
            Dictionary mapping matching keys to their values
        """
        if not self.config.enable_caching:
            return {}

        try:
            async with self.client.get_redis_connection() as redis:
                namespaced_key_pattern = self._build_namespaced_key(key_pattern)
                keys = await redis.keys(namespaced_key_pattern)
                if not keys:
                    return {}
                values = await redis.mget(keys)
                return await self._process_get_batch(keys, values)
        except Exception as e:
            logger.error(f"Redis batch get error: {e}")
            return {}

    @retry_policy
    async def get_batch(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values in a batch operation

        Args:
            keys: List of keys to retrieve

        Returns:
            Dictionary mapping keys to their values
        """
        if not self.config.enable_caching:
            return {}

        try:
            namespaced_keys = [self._build_namespaced_key(key) for key in keys]
            async with self.client.get_redis_connection() as redis:
                values = await redis.mget(namespaced_keys)
                return await self._process_get_batch(namespaced_keys, values)
        except Exception as e:
            logger.error(f"Redis batch get error: {e}")
            return {}

    @retry_policy
    async def scan_keys(self, pattern: str, count: int = 100) -> AsyncIterator[str]:
        """Scan keys matching pattern

        Args:
            pattern: Pattern to match against keys (will be prefixed with namespace)
            count: Number of keys to return per iteration

        Yields:
            Matching keys with namespace prefix removed
        """
        if not self.config.enable_caching:
            return

        try:
            namespaced_pattern = self._build_namespaced_key(pattern)
            async with self.client.get_redis_connection() as redis:
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=namespaced_pattern, count=count
                    )
                    for key in keys:
                        # Remove namespace prefix before yielding
                        yield key[len(self.namespace) + 1 :]
                    if cursor == 0:
                        break
        except RedisError as e:
            self.logger.error(f"Error scanning Redis keys with pattern {pattern}: {e}")
            raise

    @retry_policy
    async def get_all_keys(self) -> list[str]:
        """Get all keys in the cache namespace with the namespace prefix"""
        if not self.config.enable_caching:
            return []

        async with self.client.get_redis_connection() as redis:
            return await redis.keys(self._build_namespaced_key("*"))

    @retry_policy
    async def exists(self, key: str) -> bool:
        """Check if key exists

        Args:
            key: Redis key (will be prefixed with namespace)

        Returns:
            True if key exists, False otherwise
        """
        if not self.config.enable_caching:
            return False

        try:
            namespaced_key = self._build_namespaced_key(key)
            async with self.client.get_redis_connection() as redis:
                return await redis.exists(namespaced_key) > 0
        except RedisError as e:
            logger.error(f"Error checking existence of Redis hash {key}: {e}")
            raise

    async def _process_get_batch(
        self, keys: list[str], values: list[Any]
    ) -> dict[str, Any]:
        """Process batch get results

        Args:
            keys: List of namespaced keys
            values: List of raw values from Redis

        Returns:
            Dictionary mapping original keys to deserialized values
        """
        try:
            result = {}
            for key, value in zip(
                keys, values
            ):  # Remove strict=True to handle mismatches gracefully
                if value is not None:  # Only process non-None values
                    try:
                        # Extract original key by removing namespace prefix
                        original_key = (
                            key[len(self.namespace) + 1 :]
                            if key.startswith(f"{self.namespace}:")
                            else key
                        )
                        value = self._decompress(value)
                        deserialized = self._deserialize(value)
                        if deserialized is not None:
                            result[original_key] = deserialized
                            await self._update_access_stats(original_key)
                    except Exception as e:
                        logger.error(
                            f"Error processing batch result for key {key}: {e}"
                        )
                        continue  # Skip problematic keys but continue processing others
            return result
        except Exception as e:
            logger.error(f"Redis batch get error: {e}")
            # Only return empty dict for critical errors
            if isinstance(e, (RedisError, pickle.PickleError)):
                return {}
            raise  # Re-raise other exceptions

    @retry_policy
    async def set_batch(
        self,
        items: dict[str, Any],
        ttl: int | None = None,
        pipe: Any | None = None,
        labels: dict[str, Any] = {},
    ) -> bool:
        if not self.config.enable_caching:
            return False

        try:
            extras = self._format_extras(labels)
            values, ttls = {}, {}
            for key, value in items.items():
                ttls[key] = ttl if ttl else await self._calculate_ttl(key)
                values[key] = self._serialize(value)
                values[key] = self._compress(values[key], labels)

                # Check size limits
                if len(values[key]) > self.config.max_size_mb * 1024 * 1024:
                    logger.warning(
                        f"Value too large for cache: {len(values[key])} bytes"
                    )
                    return False

            async with self.client.get_pipeline() as (pipe, _):
                for key, value in items.items():
                    namespaced_key = self._build_namespaced_key(key)
                    if ttls[key]:
                        pipe.setex(namespaced_key, ttls[key], values[key])
                    else:
                        pipe.set(namespaced_key, values[key])
                await pipe.execute()

                self.metrics["size"].labels(
                    namespace=self.namespace, operation="set_batch", extras=extras
                ).inc(sum([len(value) for value in values.values()]))
                self.metrics["operations"].labels(
                    namespace=self.namespace,
                    operation="set_batch",
                    status="success",
                    extras=extras,
                ).inc()
                return True

        except Exception as e:
            logger.error(f"Redis batch set error: {e}")
            self.metrics["operations"].labels(
                namespace=self.namespace,
                operation="set_batch",
                status="failure",
                extras=self._format_extras(labels),
            ).inc()
            return False

    @retry_policy
    async def add_to_set(self, key: str, member: str, pipe: Any | None = None) -> None:
        try:
            namespaced_key = self._build_namespaced_key(key)
            if pipe:
                pipe.sadd(namespaced_key, member)
            else:
                async with self.client.get_redis_connection() as redis:
                    await redis.sadd(namespaced_key, member)
        except Exception as e:
            logger.error(f"Redis add to set error: {e}")

    @retry_policy
    async def remove_from_set(
        self, key: str, member: str, pipe: Any | None = None
    ) -> None:
        try:
            namespaced_key = self._build_namespaced_key(key)
            if pipe:
                pipe.srem(namespaced_key, member)
            else:
                async with self.client.get_redis_connection() as redis:
                    await redis.srem(namespaced_key, member)
        except Exception as e:
            logger.error(f"Redis remove from set error: {e}")

    @retry_policy
    async def get_set_members(self, key: str) -> builtins.set[str]:
        try:
            namespaced_key = self._build_namespaced_key(key)
            async with self.client.get_redis_connection() as redis:
                members = await redis.smembers(namespaced_key)
                return builtins.set(members) if members else builtins.set()
        except Exception as e:
            logger.error(f"Redis get set members error: {e}")
            return builtins.set()

    @retry_policy
    async def clear_set(self, key: str, pipe: Any | None = None) -> None:
        try:
            namespaced_key = self._build_namespaced_key(key)
            if pipe:
                pipe.delete(namespaced_key)
            else:
                async with self.client.get_redis_connection() as redis:
                    await redis.delete(namespaced_key)
        except Exception as e:
            logger.error(f"Redis clear set error: {e}")

    @retry_policy
    async def add_to_set_multiple(
        self, key: str, members: set[str], pipe: Any | None = None
    ) -> None:
        try:
            namespaced_key = self._build_namespaced_key(key)
            if pipe:
                pipe.sadd(namespaced_key, *members)
            else:
                async with self.client.get_redis_connection() as redis:
                    await redis.sadd(namespaced_key, *members)
        except Exception as e:
            logger.error(f"Redis set add multiple error: {e}")

    @retry_policy
    async def get_set_cardinality(self, key: str) -> int:
        """Get set cardinality"""
        try:
            namespaced_key = self._build_namespaced_key(key)
            async with self.client.get_redis_connection() as redis:
                return await redis.scard(namespaced_key)
        except Exception as e:
            logger.error(
                f"Failed to get set cardinality in cache {self.namespace}: {e}"
            )
            return 0

    @retry_policy
    async def add_to_sorted_set(
        self, key: str, member: str, score: float, pipe: Any | None = None
    ) -> None:
        try:
            namespaced_key = self._build_namespaced_key(key)
            if pipe:
                pipe.zadd(namespaced_key, {member: score})
            else:
                async with self.client.get_redis_connection() as redis:
                    await redis.zadd(namespaced_key, {member: score})
        except Exception as e:
            logger.error(f"Redis add to sorted set error: {e}")

    @retry_policy
    async def get_sorted_set_members(
        self, key: str, pipe: Any | None = None
    ) -> list[tuple[str, float]]:
        """Get all members of a sorted set with their scores"""
        try:
            namespaced_key = self._build_namespaced_key(key)
            if pipe:
                pipe.zrange(namespaced_key, 0, -1, withscores=True)
                return []
            else:
                async with self.client.get_redis_connection() as redis:
                    members = await redis.zrange(namespaced_key, 0, -1, withscores=True)
                    return [(member, score) for member, score in members]
        except Exception as e:
            logger.error(f"Redis get sorted set members error: {e}")
            return []

    @retry_policy
    async def get_dict_field_value(
        self, dict_name: str, key: str, pipe: Any | None = None
    ) -> Any | None:
        namespaced_dict_name = self._build_namespaced_key(dict_name)
        if pipe:
            pipe.hget(namespaced_dict_name, key)
        else:
            async with self.client.get_redis_connection() as redis:
                return await redis.hget(namespaced_dict_name, key)

    @retry_policy
    async def get_all_dict_fields(
        self, dict_name: str, pipe: Any | None = None
    ) -> dict[str, Any] | None:
        """Get all data from hash

        Args:
            dict_name: Redis key (will be prefixed with namespace)
            pipe: Optional pipeline for batch operations

        Returns:
            Dictionary of data
        """
        try:
            namespaced_key = self._build_namespaced_key(dict_name)
            async with self.client.get_redis_connection() as redis:
                return await redis.hgetall(namespaced_key)

        except RedisError as e:
            logger.error(f"Error getting data from Redis hash {dict_name}: {e}")
            raise

    @retry_policy
    async def set_dict_field_value(
        self, dict_name: str, key: str, value: Any, pipe: Any | None = None
    ) -> None:
        namespaced_dict_name = self._build_namespaced_key(dict_name)
        if pipe:
            pipe.hset(namespaced_dict_name, key, value)
        else:
            async with self.client.get_redis_connection() as redis:
                await redis.hset(namespaced_dict_name, key, value)

    @retry_policy
    async def set_dict_field_values(
        self,
        dict_name: str,
        data: dict[str, Any],
        ttl: int | None = None,
        pipe: Any | None = None,
    ) -> None:
        """Store data in hash

        Args:
            dict_name: Redis key (will be prefixed with namespace)
            data: Data to store
            ttl: Optional TTL in seconds
            pipe: Optional pipeline for batch operations
        """
        try:
            # Store data and metadata
            hash_data = dict(data)

            namespaced_key = self._build_namespaced_key(dict_name)
            if pipe:
                pipe.hmset(namespaced_key, hash_data)
                if ttl is not None:
                    pipe.expire(namespaced_key, ttl)
            else:
                async with self.client.get_redis_connection() as redis:
                    await redis.hmset(namespaced_key, hash_data)
                    # Set TTL if specified
                    if ttl is not None:
                        await redis.expire(namespaced_key, ttl)

        except RedisError as e:
            logger.error(f"Error storing data in Redis hash {dict_name}: {e!s}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error storing data in Redis hash {dict_name}: {e!s}"
            )
            raise

    @retry_policy
    async def delete_dict_field(
        self, dict_name: str, key: str, pipe: Any | None = None
    ) -> None:
        namespaced_dict_name = self._build_namespaced_key(dict_name)
        if pipe:
            pipe.hdel(namespaced_dict_name, key)
        else:
            async with self.client.get_redis_connection() as redis:
                await redis.hdel(namespaced_dict_name, key)

    @retry_policy
    async def clear_dict(self, dict_name: str, pipe: Any | None = None) -> None:
        namespaced_dict_name = self._build_namespaced_key(dict_name)
        if pipe:
            pipe.delete(namespaced_dict_name)
        else:
            async with self.client.get_redis_connection() as redis:
                await redis.delete(namespaced_dict_name)

    @retry_policy
    async def increment_dict_value(
        self, dict_name: str, key: str, amount: int = 1, pipe: Any | None = None
    ) -> None:
        namespaced_dict_name = self._build_namespaced_key(dict_name)
        if pipe:
            pipe.hincrby(namespaced_dict_name, key, amount)
        else:
            async with self.client.get_redis_connection() as redis:
                await redis.hincrby(namespaced_dict_name, key, amount)

    @retry_policy
    async def increment(
        self, key: str, amount: int = 1, pipe: Any | None = None
    ) -> int:
        try:
            namespaced_key = self._build_namespaced_key(key)
            if pipe:
                pipe.incrby(namespaced_key, amount)
            else:
                async with self.client.get_redis_connection() as redis:
                    return await redis.incrby(namespaced_key, amount)
        except Exception as e:
            logger.error(f"Redis increment error: {e}")
            return 0

    @retry_policy
    async def get_counter(self, key: str) -> int:
        try:
            key = self._build_namespaced_key(key)
            async with self.client.get_redis_connection() as redis:
                value = await redis.get(key)
                return int(value) if value else 0
        except Exception as e:
            logger.error(f"Redis get counter error: {e}")
            return 0

    @retry_policy
    async def get_stored_stats(self) -> dict[str, Any]:
        """Get Redis cache statistics"""
        try:
            async with self.client.get_redis_connection() as redis:
                stats = await redis.hgetall(self.stats_key)
                return {k: int(v) for k, v in stats.items()}
        except Exception as e:
            logger.error(f"Redis get stats error: {e}")
            return {}

    @retry_policy
    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "namespace": self.namespace,
            "base_ttl_seconds": self.config.base_ttl,
            "min_ttl_seconds": self.config.min_ttl,
            "max_ttl_seconds": self.config.max_ttl,
            "max_size_mb": self.config.max_size_mb,
            "enable_adaptive_ttl": self.config.enable_adaptive_ttl,
            "total_size": 0,
            "used_size": 0,
        }
        try:
            async with self.client.get_redis_connection() as redis:
                info = await redis.info()
                # logger.info(f"Redis info:\n{json.dumps(info, indent=2)}")
                # Get used memory directly in bytes
                used_memory = int(info.get("used_memory", 0))

                # Get total memory - use maxmemory if set, otherwise total system memory
                total_memory = int(info.get("maxmemory", 0))
                if total_memory == 0:  # No memory limit set
                    total_memory = used_memory  # Just use current usage as total

                stats.update(
                    {
                        "total_size": total_memory,
                        "used_size": used_memory,
                        "redis": {
                            "used_memory": used_memory,
                            "connected_clients": int(info.get("connected_clients", 0)),
                            "keyspace_hits": int(info.get("keyspace_hits", 0)),
                            "keyspace_misses": int(info.get("keyspace_misses", 0)),
                        },
                    }
                )
        except RedisError as e:
            logger.warning(f"Redis error getting stats: {e!s}")
            stats["redis"] = {"error": str(e)}

        return stats

    @retry_policy
    async def clean_expired(self) -> None:
        """Clean expired entries (Redis handles this automatically)"""
        # Redis handles TTL expiration automatically
        self.last_cleanup = datetime.now()

    @retry_policy
    async def update_stats(self) -> None:
        """Update Redis cache statistics"""
        try:
            async with self.client.get_pipeline() as (pipe, _):
                pipe.hincrby(self.stats_key, "total_gets", self.stats["gets"])
                pipe.hincrby(self.stats_key, "total_sets", self.stats["sets"])
                pipe.hincrby(
                    self.stats_key, "total_hits", self.stats["hits"]
                )  # self.stats["local_hits"] + self.stats["remote_hits"]
                pipe.hincrby(self.stats_key, "total_misses", self.stats["misses"])
                await pipe.execute()
        except Exception as e:
            logger.error(f"Redis update stats error: {e}")

    @retry_policy
    async def optimize(self) -> None:
        """Optimize Redis cache (no-op for now)"""
        # Redis handles memory management
        pass

    @retry_policy
    async def close(self):
        """Close Redis connections"""
        try:
            async with self.client.get_redis_connection() as redis:
                await redis.close()
                await redis.connection_pool.disconnect()
        except Exception as e:
            logger.error(f"Failed to close cache connections in {self.namespace}: {e}")

    async def _calculate_ttl(self, key: str) -> int | None:
        """Calculate adaptive TTL based on access patterns"""
        if not self.config.enable_adaptive_ttl:
            return self.config.base_ttl

        access_count = await self.get_counter(f"access_count:{key}")

        if access_count == 0:
            return self.config.base_ttl

        # Adjust TTL based on access frequency
        # Increase TTL for frequently accessed items
        if access_count > 10:
            ttl = min(self.config.base_ttl * 2, self.config.max_ttl)
        # Decrease TTL for rarely accessed items
        elif access_count < 3:
            ttl = max(self.config.base_ttl // 2, self.config.min_ttl)
        else:
            ttl = self.config.base_ttl

        return ttl

    async def _update_access_stats(self, key: str) -> None:
        """Update access statistics"""
        if self.config.enable_adaptive_ttl:
            await self.increment(f"access_count:{key}")

    async def _monitor_storage(self):
        """Monitor storage usage and performance"""
        try:
            while not self._shutdown_event.is_set():
                try:
                    stats = await self.get_stats()
                    used_size = stats.get("used_size", 0)
                    total_size = stats.get(
                        "total_size", 1
                    )  # Default to 1 to avoid division by zero

                    self.metrics["size"].labels(
                        namespace=self.namespace, operation="total", extras=""
                    ).set(total_size)

                    self.metrics["size"].labels(
                        namespace=self.namespace, operation="used", extras=""
                    ).set(used_size)

                    # Alert on high usage if we have valid sizes
                    if total_size > 0 and used_size / total_size > 0.9:
                        logger.warning(f"{self.namespace} cache usage above 90%")

                    try:
                        await asyncio.wait_for(self._shutdown_event.wait(), timeout=60)
                    except asyncio.TimeoutError:
                        continue  # Normal timeout, continue monitoring

                except Exception as e:
                    logger.error(f"Storage monitoring error: {e}")
                    try:
                        await asyncio.wait_for(self._shutdown_event.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        continue  # Normal timeout, retry monitoring
        except asyncio.CancelledError:
            logger.info(
                f"Monitoring task cancelled for cache instance {self.instance_id}"
            )
        finally:
            logger.info(
                f"Monitoring task stopped for cache instance {self.instance_id}"
            )

    def _start_monitoring_task(self):
        """Start background monitoring"""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(
                self._monitor_storage(), name=f"cache_monitor_{self.instance_id}"
            )

    def _setup_metrics(self) -> dict[str, Any]:
        """Setup cache metrics"""
        return {
            "hits": CACHE_HITS,
            "misses": CACHE_MISSES,
            "size": CACHE_SIZE,
            "serialization_time": CACHE_ITEM_SERIALIZATION_TIME,
            "item_size": CACHE_ITEM_SIZE,
            "operations": CACHE_OPERATIONS,
            "latency": CACHE_LATENCY,
            "errors": CACHE_ERRORS,
            "memory": CACHE_MEMORY,
            "evictions": CACHE_EVICTIONS,
            "age": CACHE_ENTRY_AGE,
            "batch_size": CACHE_BATCH_SIZE,
            "compression_ratio": CACHE_COMPRESSION_RATIO,
            "health": CACHE_HEALTH,
        }

    async def shutdown(self):
        """Gracefully shutdown the cache instance"""
        logger.info(f"Shutting down cache instance {self.instance_id}")
        await self.cleanup()

    async def __aenter__(self) -> DistributedSimpleCache:
        """Async context manager entry"""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit - ensures proper cleanup"""
        await self.cleanup()
