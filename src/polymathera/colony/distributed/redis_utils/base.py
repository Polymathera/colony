"""Base classes for distributed storage operations."""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from redis.asyncio import Redis
from redis.asyncio.client import Pipeline
from redis.exceptions import RedisError, WatchError

logger = logging.getLogger(__name__)

# Context variable to store the current pipeline
current_pipeline: contextvars.ContextVar[Pipeline | None] = contextvars.ContextVar(
    "current_pipeline", default=None
)

T = TypeVar("T")


class DistributedPipeline:
    """
    Pipeline for batching distributed storage operations.

    All operations within this context will be batched and executed atomically.
    The pipeline is stored in a context variable, making it available to any
    code executed within the context without passing it explicitly.

    Example:
        async with DistributedPipeline(redis) as pipe:
            await cache.set("key1", "value1")
            await set_store.add_member("set1", "member1")
            # Operations are batched
            results = await pipe.execute()
    """

    def __init__(self, redis: Redis):
        self.redis = redis
        self.pipeline: Pipeline | None = None
        self._token: contextvars.Token | None = None
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def __aenter__(self) -> AsyncGenerator[DistributedPipeline, None]:
        async with self._lock:
            self.pipeline = await self.redis.pipeline()
            self._token = current_pipeline.set(self.pipeline)
            try:
                yield self
            finally:
                if self._token:
                    current_pipeline.reset(self._token)
                self.pipeline = None

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Error in pipeline: {exc_val}")
            return False
        return True

    async def execute(self) -> list[Any]:
        """Execute all batched operations atomically."""
        if not self.pipeline:
            raise RuntimeError("No active pipeline")
        try:
            return await self.pipeline.execute()
        except RedisError as e:
            logger.error(f"Failed to execute pipeline: {e}")
            raise

    async def execute_watched(self) -> bool:
        """Execute pipeline and watch for changes."""
        if not self.pipeline:
            raise RuntimeError("No active pipeline")
        try:
            await self.pipeline.execute()
            return True
        except WatchError:
            return False

    async def watch(self, *keys: str) -> None:
        """Watch state for changes."""
        if not self.pipeline:
            raise RuntimeError("No active pipeline")
        await self.pipeline.watch(*keys)

    async def unwatch(self) -> None:
        """Unwatch state."""
        if not self.pipeline:
            raise RuntimeError("No active pipeline")
        await self.pipeline.unwatch()

    async def multi(self) -> None:
        """Start a multi-command transaction."""
        if not self.pipeline:
            raise RuntimeError("No active pipeline")
        await self.pipeline.multi()


class BaseStore:
    """Base class for all distributed stores."""

    def __init__(self, redis: Redis, namespace: str):
        self.redis = redis
        self.namespace = namespace
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _build_key(self, *parts: str) -> str:
        """Build namespaced key."""
        return f"{self.namespace}:{':'.join(parts)}"

    async def _get_pipeline(self) -> Pipeline:
        """Get current pipeline or create new one."""
        pipeline = current_pipeline.get()
        if pipeline is None:
            raise RuntimeError(
                "No active pipeline. Operations must be performed within a DistributedPipeline context."
            )
        return pipeline

    async def _execute_atomic(
        self, operation: str, func: callable, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute operation atomically, using pipeline if available. Otherwise, use Redis directly."""
        try:
            pipeline = current_pipeline.get()
            if pipeline is not None:
                return await func(pipeline, *args, **kwargs)
            return await func(self.redis, *args, **kwargs)
        except RedisError as e:
            self.logger.error(f"Failed to execute {operation}: {e}")
            raise
