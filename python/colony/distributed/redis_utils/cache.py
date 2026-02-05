"""Distributed cache implementation."""

import json
import pickle
from typing import Any, TypeVar

from redis.asyncio import Redis

from .base import BaseStore, Pipeline

T = TypeVar("T")

METADATA_KEY = "__cache_metadata__"


class DistributedCache(BaseStore):
    """
    Distributed cache with support for:
    - JSON and Pickle serialization
    - TTL management
    - Metadata attachment
    - Atomic operations
    - Namespace isolation
    """

    def __init__(
        self,
        redis: Redis,
        namespace: str,
        default_ttl: int | None = None,
        use_pickle: bool = True,
    ):
        """Initialize cache.

        Args:
            redis: Redis client
            namespace: Namespace for key isolation
            default_ttl: Default TTL in seconds
            use_pickle: Use Pickle serialization instead of JSON
        """
        super().__init__(redis, namespace)
        self.default_ttl = default_ttl
        self.use_pickle = use_pickle

    def get_all_keys(self, key: str) -> list[str]:
        """Get all keys associated with a resource."""
        return [self._build_key(key)]

    async def set(
        self,
        key: str,
        value: T,
        ttl: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store value with optional TTL and metadata.

        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live in seconds (overrides default_ttl)
            metadata: Optional metadata to attach
        """
        ttl = ttl if ttl is not None else self.default_ttl
        full_key = self._build_key(key)

        # Prepare value and metadata
        if self.use_pickle:
            if metadata:
                data = (value, metadata)
            else:
                data = value
            serialized = pickle.dumps(data)
        else:
            if metadata:
                data = {"value": value, METADATA_KEY: metadata}
            else:
                data = value
            serialized = json.dumps(data)

        async def _set(client: Redis | Pipeline, *args: Any) -> None:
            await client.set(full_key, serialized)
            if self.use_pickle:
                await client.set(full_key, serialized)
            else:
                await client.hset(full_key, mapping=serialized)
            if ttl is not None:
                await client.expire(full_key, ttl)

        if self.use_pickle:
            await self._execute_atomic("set", _set)
        else:
            await self._execute_atomic("hset", _set)

    async def get(
        self,
        key: str,
        with_metadata: bool = False,
    ) -> tuple[T | None, dict[str, Any] | None] | T | None:
        """Get value and optional metadata.

        Args:
            key: Cache key
            with_metadata: Whether to return metadata

        Returns:
            If with_metadata=True: Tuple of (value, metadata) or (None, None)
            If with_metadata=False: Just the value or None
        """
        full_key = self._build_key(key)

        async def _get(client: Redis | Pipeline, *args: Any) -> Any:
            if self.use_pickle:
                return await client.get(full_key)
            else:
                return await client.hgetall(full_key)

        data = await self._execute_atomic("get", _get)
        if not data:
            return (None, None) if with_metadata else None

        # TODO: Should we call data.decode() here?

        try:
            if self.use_pickle:
                deserialized = pickle.loads(data)
                if isinstance(deserialized, tuple) and len(deserialized) == 2:
                    return deserialized if with_metadata else deserialized[0]
                return (deserialized, None) if with_metadata else deserialized
            else:
                deserialized = json.loads(data)
                if isinstance(deserialized, dict) and METADATA_KEY in deserialized:
                    return (
                        (deserialized["value"], deserialized[METADATA_KEY])
                        if with_metadata
                        else deserialized["value"]
                    )
                return (deserialized, None) if with_metadata else deserialized
        except (json.JSONDecodeError, pickle.UnpicklingError) as e:
            self.logger.error(f"Failed to deserialize value for key {key}: {e}")
            return (None, None) if with_metadata else None

    async def delete(self, key: str) -> None:
        """Delete value from cache.

        Args:
            key: Cache key
        """
        full_key = self._build_key(key)

        async def _delete(client: Redis | Pipeline, *args: Any) -> None:
            await client.delete(full_key)

        await self._execute_atomic("delete", _delete)

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        full_key = self._build_key(key)

        async def _exists(client: Redis | Pipeline, *args: Any) -> bool:
            return await client.exists(full_key) > 0

        return await self._execute_atomic("exists", _exists)

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment value atomically.

        Args:
            key: Cache key
            amount: Amount to increment by

        Returns:
            New value after increment
        """
        full_key = self._build_key(key)

        async def _incr(client: Redis | Pipeline, *args: Any) -> int:
            return await client.incrby(full_key, amount)

        return await self._execute_atomic("increment", _incr)

    async def cleanup(self, pattern: str | None = None) -> None:
        """Delete all keys matching pattern in this namespace.

        Args:
            pattern: Optional pattern to match keys against
        """
        match_pattern = self._build_key(pattern) if pattern else f"{self.namespace}:*"

        async def _scan_delete(client: Redis | Pipeline, *args: Any) -> None:
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=match_pattern, count=100)
                if keys:
                    await client.delete(*keys)
                if cursor == 0:
                    break

        await self._execute_atomic("cleanup", _scan_delete)


