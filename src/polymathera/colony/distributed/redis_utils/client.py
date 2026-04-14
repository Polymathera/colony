from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, ClassVar

from circuitbreaker import circuit
from opentelemetry import metrics, trace
from pydantic import Field, field_validator
from redis.asyncio import Redis
from redis.asyncio.client import Pipeline, PubSub
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError, WatchError

from ..config import ConfigComponent, register_polymathera_config
from ..metrics.redis_om import RedisOMMetricsMonitor

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)


class RedisPersistenceMode(str, Enum):
    """Redis persistence configuration modes"""

    NONE = "none"
    RDB = "rdb"  # Snapshot-based
    AOF = "aof"  # Append-only file
    HYBRID = "hybrid"  # Both RDB and AOF


@register_polymathera_config()
class RedisConfig(ConfigComponent):
    """Configuration for Redis client with persistence and connection pooling."""

    redis_url: str | None = Field(default=None)
    redis_host: str | None = Field(
        default=None,
        json_schema_extra={"env": "REDIS_HOST"},
        #default_factory=lambda: os.getenv("REDIS_HOST", "localhost")
    )
    redis_port: int | None = Field(
        default=None,
        json_schema_extra={"env": "REDIS_PORT"},
        #default_factory=lambda: int(os.getenv("REDIS_PORT"))
    )
    redis_password: str | None = Field(
        default=None,
        json_schema_extra={"env": "REDIS_PASSWORD", "optional": True},
        #default_factory=lambda: os.getenv("REDIS_PASSWORD")
    )
    db: int = 0
    persistence_mode: RedisPersistenceMode = RedisPersistenceMode.NONE
    persistence_dir: str | None = Field(
        default=None, # "/data/redis"
        json_schema_extra={"env": "REDIS_PERSISTENCE_DIR", "optional": True},
    )
    rdb_save_frequency: int = 3600  # How often to save RDB snapshots (seconds)
    aof_fsync: str = "everysec"  # AOF fsync mode ('always', 'everysec', 'no')
    max_pool_size: int = 2500  # Each agent holds ~5 permanent PubSub connections (memory hierarchy)
    decode_responses: bool = False  # Keep raw bytes for pickle data
    redis_pool_timeout: int = 5  # Timeout for Redis connection pool
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    db_file_prefix: str = "vmr_knowledge_base"
    ttl: int = 86400  # 24 hour default TTL
    namespace: str = ""  # Namespace for Redis keys
    # Sentinel configuration
    sentinel_enabled: bool = False  # Whether to use Redis Sentinel
    sentinel_master: str | None = Field(default=None)  # Name of the master to monitor
    sentinel_hosts: list[tuple[str, int]] | None = Field(
        default=None
    )  # List of sentinel hosts
    sentinel_password: str | None = Field(default=None)  # Password for Sentinel auth
    sentinel_socket_timeout: float = 0.1  # Socket timeout for Sentinel connections
    sentinel_retry_interval: int = (
        1000  # Time between Sentinel connection retries in ms
    )

    CONFIG_PATH: ClassVar[str] = "redis"



    def get_redis_url(self) -> str:
        """Get Redis URL from config."""
        if self.redis_url:
            return self.redis_url

        if self.sentinel_enabled and self.sentinel_hosts and self.sentinel_master:
            # Format: redis://[[password@]host[:port]]/db
            # We'll use the first sentinel host as the initial connection point
            host, port = self.sentinel_hosts[0]
            auth = f":{self.sentinel_password}@" if self.sentinel_password else ""
            return f"redis://{auth}{host}:{port}/0"

        if self.redis_host is None or self.redis_port is None:
            raise ValueError("redis_host and redis_port must be configured when redis_url is not provided")

        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}"

    @field_validator("sentinel_master")
    def validate_sentinel_master(cls, v: str | None, info: Any) -> str | None:
        """Validate that sentinel_master is provided when sentinel_enabled is True."""
        if info.data.get("sentinel_enabled", False) and not v:
            raise ValueError(
                "sentinel_master must be provided when sentinel_enabled is True"
            )
        return v

    @field_validator("sentinel_hosts")
    def validate_sentinel_hosts(
        cls, v: list[tuple[str, int]] | None, info: Any
    ) -> list[tuple[str, int]] | None:
        """Validate that sentinel_hosts is provided when sentinel_enabled is True."""
        if info.data.get("sentinel_enabled", False) and not v:
            raise ValueError(
                "sentinel_hosts must be provided when sentinel_enabled is True"
            )
        return v


class RedisCircuitBreakerMonitor:
    """Monitor for Redis circuit breaker events with Prometheus metrics."""

    def __init__(self, namespace: str):
        self.namespace = namespace
        self.metrics = RedisOMMetricsMonitor()

    def record_success(self):
        """Record a successful operation."""
        self.metrics.REDIS_CIRCUIT_SUCCESSES.labels(namespace=self.namespace).inc()

    def record_failure(self, exc: Exception):
        """Record a failed operation."""
        self.metrics.REDIS_CIRCUIT_FAILURES.labels(namespace=self.namespace).inc()

    def record_state_change(self, state: str):
        """Record a circuit breaker state change."""
        state_value = {"open": 0, "half-open": 1, "closed": 2}[state]
        self.metrics.REDIS_CIRCUIT_STATE.labels(namespace=self.namespace).set(state_value)


class RedisClient:
    """
    A Redis client with connection pooling, persistence, and monitoring.

    Features:
    - Connection pooling with min/max connections
    - Configurable persistence (RDB, AOF, or hybrid)
    - Circuit breaker for fault tolerance
    - Health monitoring and metrics
    - Resource cleanup
    - Proper connection management
    - Sentinel support for high availability
    """

    def __init__(self, config: RedisConfig | None = None):
        """Initialize RedisClient with configuration."""
        self.config: RedisConfig | None = config

        # Initialize circuit breaker monitor
        self._circuit_monitor: RedisCircuitBreakerMonitor | None = None

        # ------------------------------------------------------------------
        # Connection-pool bookkeeping
        # ------------------------------------------------------------------

        self._active_connections: int = 0

        # One ConnectionPool per *event-loop* to guarantee that every
        # `redis.asyncio.Connection` is bound to the loop it will be used on.
        # Keyed by id(loop).
        self._loop_pools: dict[int, ConnectionPool] = {}

        # Concurrency throttling needs a semaphore per loop for the same
        # reason—an asyncio.Semaphore is bound to its creation loop.
        self._loop_semaphores: dict[int, asyncio.Semaphore] = {}

        # Health check configuration
        self._health_check_interval = None
        self._health_check_task: asyncio.Task | None = None
        self._last_health_check = time.time()
        self._healthy = True
        self.metrics = RedisOMMetricsMonitor()

    async def initialize(self) -> None:
        self.config = await RedisConfig.check_or_get_component(self.config)

        # Initialize circuit breaker monitor
        self._circuit_monitor = RedisCircuitBreakerMonitor(self.config.namespace)

        # ------------------------------------------------------------------
        # Connection-pool bookkeeping
        # ------------------------------------------------------------------

        self._active_connections: int = 0

        # One ConnectionPool per *event-loop* to guarantee that every
        # `redis.asyncio.Connection` is bound to the loop it will be used on.
        # Keyed by id(loop).
        self._loop_pools: dict[int, ConnectionPool] = {}

        # Concurrency throttling needs a semaphore per loop for the same
        # reason—an asyncio.Semaphore is bound to its creation loop.
        self._loop_semaphores: dict[int, asyncio.Semaphore] = {}

        # Statistics need a *canonical* pool for metrics that don't depend on
        # event-loop; we create one for the loop active during initialize().
        # Since we're in an async method, use get_running_loop()
        init_loop = asyncio.get_running_loop()
        self._loop_pools[id(init_loop)] = self._init_redis_pool_with_persistence()

        # Health check configuration
        self._health_check_interval = self.config.health_check_interval
        self._health_check_task: asyncio.Task | None = None
        self._last_health_check = time.time()
        self._healthy = True

        # Initialize metrics
        self.metrics.REDIS_POOL_SIZE.labels(namespace=self.config.namespace).set(self.config.max_pool_size)
        self.metrics.REDIS_ACTIVE_CONNECTIONS.labels(namespace=self.config.namespace).set(0)
        self.metrics.REDIS_CONNECTED.labels(namespace=self.config.namespace).set(1)  # Start as connected
        self.metrics.REDIS_HEALTH_CHECK.labels(
            status="success", namespace=self.config.namespace
        ).inc()  # Initial health check

    def _init_redis_pool_with_persistence(self) -> ConnectionPool:
        """Initialize Redis connection pool with configured persistence options"""
        persistence_mode = self.config.persistence_mode
        persistence_dir = self.config.persistence_dir

        # Ensure persistence directory exists
        if persistence_dir:
            if not os.path.exists(persistence_dir):
                os.makedirs(persistence_dir, exist_ok=True)
            else:
                logger.info(f"Using existing persistence directory: {persistence_dir}")
        elif persistence_mode != RedisPersistenceMode.NONE:
            raise ValueError("persistence_dir must be set if persistence_mode is not NONE")
            ### # Create a persistence directory if not provided
            ### # Use a temporary directory for simplicity
            ### import tempfile
            ### persistence_dir = tempfile.mkdtemp(prefix='data_redis_')
            ### os.environ['REDIS_PERSISTENCE_DIR'] = persistence_dir
            ### logger.info(f"Created persistence directory: {persistence_dir}")
        else:
            logger.info("Redis persistence disabled")

        # Basic Redis configuration
        pool_kwargs = {
            "max_connections": self.config.max_pool_size,
            "decode_responses": self.config.decode_responses,
            "retry_on_timeout": self.config.retry_on_timeout,
            "health_check_interval": self.config.health_check_interval,
            "socket_connect_timeout": self.config.redis_pool_timeout,
            "socket_keepalive": True,
            "retry_on_error": [TimeoutError],
        }

        # Add Sentinel-specific configuration if enabled
        if self.config.sentinel_enabled:
            pool_kwargs.update(
                {
                    "socket_timeout": self.config.sentinel_socket_timeout,
                    "retry_on_timeout": True,
                }
            )
            from redis.sentinel import Sentinel

            sentinel = Sentinel(
                self.config.sentinel_hosts,
                socket_timeout=self.config.sentinel_socket_timeout,
                password=self.config.sentinel_password,
                sentinel_kwargs={
                    "password": self.config.sentinel_password,
                    "socket_timeout": self.config.sentinel_socket_timeout,
                    "retry_on_timeout": True,
                },
            )
            master = sentinel.master_for(
                self.config.sentinel_master,
                socket_timeout=self.config.sentinel_socket_timeout,
                password=self.config.redis_password,
                db=self.config.db,
                **pool_kwargs,
            )
            return master.connection_pool

        redis_url = self.config.get_redis_url()
        if redis_url:
            pool = ConnectionPool.from_url(redis_url, **pool_kwargs)
        else:
            pool_kwargs.update(
                {
                    "host": self.config.redis_host,
                    "port": self.config.redis_port,
                    "db": self.config.db,
                }
            )
            if self.config.redis_password:
                pool_kwargs["password"] = self.config.redis_password
            pool = ConnectionPool(**pool_kwargs)

        # Configure persistence based on mode
        if persistence_mode != RedisPersistenceMode.NONE:
            try:
                # Create temporary client to configure persistence
                client = Redis(connection_pool=pool)

                # Set RDB configuration
                if persistence_mode in [
                    RedisPersistenceMode.RDB,
                    RedisPersistenceMode.HYBRID,
                ]:
                    save_frequency = self.config.rdb_save_frequency
                    asyncio.create_task(client.config_set("dir", persistence_dir))
                    asyncio.create_task(
                        client.config_set(
                            "dbfilename", f"{self.config.db_file_prefix}_dump.rdb"
                        )
                    )
                    asyncio.create_task(
                        client.config_set("save", f"{save_frequency} 1")
                    )

                # Set AOF configuration
                if persistence_mode in [
                    RedisPersistenceMode.AOF,
                    RedisPersistenceMode.HYBRID,
                ]:
                    asyncio.create_task(client.config_set("appendonly", "yes"))
                    asyncio.create_task(
                        client.config_set(
                            "appendfilename",
                            f"{self.config.db_file_prefix}_appendonly.aof",
                        )
                    )
                    asyncio.create_task(
                        client.config_set("appendfsync", self.config.aof_fsync)
                    )
                    asyncio.create_task(
                        client.config_set("auto-aof-rewrite-percentage", "100")
                    )
                    asyncio.create_task(
                        client.config_set("auto-aof-rewrite-min-size", "64mb")
                    )

                logger.info(f"Redis persistence configured: {persistence_mode.value}")

            except RedisError as e:
                logger.error(f"Failed to configure Redis persistence: {e}")
                raise

        return pool

    def _redis_conn_pool_semaphore(self) -> asyncio.Semaphore:
        # Ensure we use a semaphore that belongs to *this* loop
        loop = asyncio.get_running_loop()
        sem = self._loop_semaphores.get(id(loop))
        if sem is None:
            sem = asyncio.Semaphore(self.config.max_pool_size)
            self._loop_semaphores[id(loop)] = sem
        return sem

    def _redis_pool(self) -> ConnectionPool:
        # Fetch or create the ConnectionPool for this loop
        loop = asyncio.get_running_loop()
        pool = self._loop_pools.get(id(loop))
        if pool is None:
            pool = self._init_redis_pool_with_persistence()
            self._loop_pools[id(loop)] = pool
        return pool

    @asynccontextmanager
    async def get_redis_connection(self) -> AsyncGenerator[Redis, None]:
        """Get a connection from the pool with proper resource management."""
        start_time = time.time()
        try:
            async with self._redis_conn_pool_semaphore():
                wait_time = time.time() - start_time
                self.metrics.REDIS_OP_DURATION.labels(
                    operation="get_connection", namespace=self.config.namespace
                ).observe(wait_time)

                if wait_time > 1.0:  # Alert if waiting more than 1 second
                    self.metrics.REDIS_OP_COUNT.labels(
                        operation="connection_pool_exhausted",
                        status="error",
                        namespace=self.config.namespace,
                    ).inc()
                    logger.debug(
                        f"Connection pool exhausted. Waited {wait_time:.2f}s for connection"
                    )

                # Create a short-lived Redis client bound to *this* loop using
                # the loop-local pool.
                local_client = Redis(connection_pool=self._redis_pool())

                # ---- Book-keeping -------------------------------------------------
                self._active_connections += 1
                self.metrics.REDIS_ACTIVE_CONNECTIONS.labels(namespace=self.config.namespace).set(
                    self._active_connections
                )

                try:
                    yield local_client
                finally:
                    # Close the facade; this only returns its single connection
                    # to the pool – no network tear-down.
                    try:
                        await local_client.aclose()
                    except Exception as exc:  # pragma: no cover – best-effort
                        logger.debug("Error closing loop-local Redis client: %s", exc)

                    # # Debug: log loop IDs to investigate cross-loop usage
                    # if logger.isEnabledFor(logging.DEBUG):
                    #     try:
                    #         current_loop_id = id(asyncio.get_running_loop())
                    #     except RuntimeError:
                    #         current_loop_id = None

                    #     # Try to fetch one connection from the pool to inspect
                    #     try:
                    #         p = local_client.connection_pool
                    #         conn = p._available_connections[0] if p._available_connections else None  # type: ignore[attr-defined]
                    #         conn_loop_id = getattr(conn, "_loop", None) if conn else None
                    #     except Exception:
                    #         conn_loop_id = None

                    #     logger.debug(
                    #         "RedisClient debug – current_loop=%s, connection_loop=%s",
                    #         current_loop_id,
                    #         conn_loop_id,
                    #     )

                    # Decrement counter.
                    self._active_connections -= 1
                    self.metrics.REDIS_ACTIVE_CONNECTIONS.labels(namespace=self.config.namespace).set(
                        self._active_connections
                    )
        except asyncio.TimeoutError:
            self.metrics.REDIS_OP_COUNT.labels(
                operation="get_connection",
                status="timeout",
                namespace=self.config.namespace,
            ).inc()
            raise
        except Exception:
            self.metrics.REDIS_OP_COUNT.labels(
                operation="get_connection",
                status="error",
                namespace=self.config.namespace,
            ).inc()
            raise

    @circuit(
        failure_threshold=5,
        recovery_timeout=60,
        expected_exception=lambda exc_type, _: exc_type != WatchError,
        name="redis_circuit_breaker",
    )
    async def execute_with_semaphore(
        self, operation: Callable[[Redis], Awaitable[Any]]
    ) -> Any:
        """Execute operation with proper connection management and circuit breaker."""
        async with self.get_redis_connection() as conn:
            try:
                result = await operation(conn)
                self._circuit_monitor.record_success()
                return result
            except Exception as e:
                self._circuit_monitor.record_failure(e)
                raise

    @asynccontextmanager
    async def get_pipeline(self) -> AsyncGenerator[tuple[Pipeline, Redis], None]:
        """Get a Redis pipeline with proper connection management."""
        async with self.get_redis_connection() as conn:
            async with conn.pipeline() as pipe:
                yield pipe, conn

    @asynccontextmanager
    async def get_pubsub(self) -> AsyncGenerator[PubSub, None]:
        """Get a Redis pubsub connection with proper connection management."""
        async with self.get_redis_connection() as conn:
            pubsub = conn.pubsub()
            try:
                yield pubsub
            finally:
                await pubsub.close()

    async def get_connection_stats(self) -> dict[str, int]:
        """Get current connection pool statistics."""
        return {
            "active_connections": self._active_connections,
            "max_connections": self.config.max_pool_size,
            "available_connections": self.config.max_pool_size
            - self._active_connections,
        }

    async def start_health_checks(self):
        """Start health check monitoring."""
        if self._health_check_task is None:
            # Set initial health state
            self._healthy = True
            self.metrics.REDIS_CONNECTED.labels(namespace=self.config.namespace).set(1)
            self.metrics.REDIS_HEALTH_CHECK.labels(
                status="success", namespace=self.config.namespace
            ).inc()

            # Start health check loop
            self._health_check_task = asyncio.create_task(
                self._health_check_loop(),
                name=f"redis_health_check_{self.config.namespace}",
            )

    async def stop_health_checks(self):
        """Stop background health check task."""
        if self._health_check_task is not None:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

    async def _health_check_loop(self):
        """Background task for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    async def _perform_health_check(self) -> bool:
        """Perform health check and recovery if needed."""
        start_time = time.time()
        try:
            # Basic ping check
            await self.execute_with_semaphore(lambda redis: redis.ping())

            # Check connection pool health
            stats = await self.get_connection_stats()
            if stats["active_connections"] >= self.config.max_pool_size:
                logger.warning("Connection pool exhausted")
                self._healthy = False
                self.metrics.REDIS_HEALTH_CHECK.labels(
                    status="error", namespace=self.config.namespace
                ).inc()
                return False

            # Check memory usage
            info = await self.execute_with_semaphore(lambda redis: redis.info("memory"))
            used_memory = info.get("used_memory", 0)
            self.metrics.REDIS_MEMORY_USAGE.labels(namespace=self.config.namespace).set(used_memory)

            used_memory_percent = (
                info["used_memory"] / info["total_system_memory"] * 100
            )
            if used_memory_percent > 90:
                logger.warning(f"High memory usage: {used_memory_percent:.1f}%")
                self._healthy = False
                self.metrics.REDIS_HEALTH_CHECK.labels(
                    status="error", namespace=self.config.namespace
                ).inc()
                return False

            self._healthy = True
            self.metrics.REDIS_HEALTH_CHECK.labels(
                status="success", namespace=self.config.namespace
            ).inc()
            self.metrics.REDIS_CONNECTED.labels(namespace=self.config.namespace).set(1)
            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self._healthy = False
            self.metrics.REDIS_HEALTH_CHECK.labels(
                status="error", namespace=self.config.namespace
            ).inc()
            self.metrics.REDIS_CONNECTED.labels(namespace=self.config.namespace).set(0)

            # Attempt recovery
            await self._attempt_recovery()
            return False

        finally:
            duration = time.time() - start_time
            self.metrics.REDIS_OP_DURATION.labels(
                operation="health_check", namespace=self.config.namespace
            ).observe(duration)
            self._last_health_check = time.time()

    async def _attempt_recovery(self):
        """Attempt to recover from unhealthy state."""
        try:
            # Record recovery attempt
            self.metrics.REDIS_HEALTH_CHECK.labels(
                status="recovery_attempt", namespace=self.config.namespace
            ).inc()

            # Close all connections
            await self.redis.close()

            # Reinitialize Redis client
            self.redis_pool = self._init_redis_pool_with_persistence()
            self.redis = Redis(connection_pool=self.redis_pool)

            # Verify connection
            await self.execute_with_semaphore(lambda redis: redis.ping())

            self._healthy = True
            self.metrics.REDIS_HEALTH_CHECK.labels(
                status="recovery_success", namespace=self.config.namespace
            ).inc()
            logger.info("Successfully recovered Redis connection")

        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            self._healthy = False
            self.metrics.REDIS_HEALTH_CHECK.labels(
                status="recovery_failure", namespace=self.config.namespace
            ).inc()

    async def is_healthy(self) -> bool:
        """Check if the client is healthy."""
        # Perform health check if it hasn't been done recently
        if time.time() - self._last_health_check > self._health_check_interval:
            await self._perform_health_check()
        return self._healthy

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            await self.redis.flushdb()
            await self.redis.aclose()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def memory_usage(self) -> int:
        """Get Redis memory usage in bytes."""
        try:
            info = await self.redis.info("memory")
            return int(info["used_memory"])
        except Exception as e:
            logger.error(f"Failed to get Redis memory usage: {e}")
            return 0
