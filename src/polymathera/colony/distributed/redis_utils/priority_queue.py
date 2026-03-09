"""Distributed priority queue implementation."""

from collections.abc import AsyncIterator
from typing import Any

from redis.asyncio import Redis
from redis.asyncio.client import Pipeline

from .base import BaseStore


class DistributedPriorityQueue(BaseStore):
    """
    Distributed priority queue with support for:
    - Priority-based ordering
    - Atomic operations
    - Namespace isolation
    - Batch operations via pipeline
    - Optional TTL for items
    - Metadata attachment
    """

    def __init__(
        self,
        redis: Redis,
        namespace: str,
        default_ttl: int | None = None,
    ):
        """Initialize queue.

        Args:
            redis: Redis client
            namespace: Namespace for key isolation
            default_ttl: Default TTL in seconds for items
        """
        super().__init__(redis, namespace)
        self.default_ttl = default_ttl

    async def add_item(
        self,
        queue_name: str,
        item: str,
        priority: float,
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> None:
        """Add item to queue with priority.

        Args:
            queue_name: Name of the queue
            item: Item to add
            priority: Priority value (lower = higher priority)
            metadata: Optional metadata to attach
            ttl: Optional TTL in seconds (overrides default)
        """
        queue_key = self._build_key(queue_name)
        metadata_key = (
            self._build_key(queue_name, "metadata", item) if metadata else None
        )
        ttl = ttl if ttl is not None else self.default_ttl

        async def _add(client: Redis | Pipeline, *args: Any) -> None:
            # Add to sorted set
            await client.zadd(queue_key, {item: priority})

            # Set TTL on queue
            if ttl is not None:
                await client.expire(queue_key, ttl)

            # Store metadata if provided
            if metadata and metadata_key:
                await client.hset(metadata_key, mapping=metadata)
                if ttl is not None:
                    await client.expire(metadata_key, ttl)

        await self._execute_atomic("add_item", _add)

    async def add_items(
        self,
        queue_name: str,
        items: dict[str, float],
        metadata: dict[str, dict[str, Any]] | None = None,
        ttl: int | None = None,
    ) -> None:
        """Add multiple items with priorities.

        Args:
            queue_name: Name of the queue
            items: Dict mapping items to priorities
            metadata: Optional dict mapping items to metadata
            ttl: Optional TTL in seconds (overrides default)
        """
        if not items:
            return

        queue_key = self._build_key(queue_name)
        ttl = ttl if ttl is not None else self.default_ttl

        async def _add_multiple(client: Redis | Pipeline, *args: Any) -> None:
            # Add all items to sorted set
            await client.zadd(queue_key, items)

            # Set TTL on queue
            if ttl is not None:
                await client.expire(queue_key, ttl)

            # Store metadata if provided
            if metadata:
                for item, item_metadata in metadata.items():
                    metadata_key = self._build_key(queue_name, "metadata", item)
                    await client.hset(metadata_key, mapping=item_metadata)
                    if ttl is not None:
                        await client.expire(metadata_key, ttl)

        await self._execute_atomic("add_items", _add_multiple)

    async def remove_item(self, queue_name: str, item: str) -> None:
        """Remove item from queue.

        Args:
            queue_name: Name of the queue
            item: Item to remove
        """
        queue_key = self._build_key(queue_name)
        metadata_key = self._build_key(queue_name, "metadata", item)

        async def _remove(client: Redis | Pipeline, *args: Any) -> None:
            await client.zrem(queue_key, item)
            await client.delete(metadata_key)

        await self._execute_atomic("remove_item", _remove)

    async def get_item(
        self, queue_name: str, item: str
    ) -> tuple[float | None, dict[str, Any] | None]:
        """Get item's priority and metadata.

        Args:
            queue_name: Name of the queue
            item: Item to get

        Returns:
            Tuple of (priority, metadata) or (None, None) if not found
        """
        queue_key = self._build_key(queue_name)
        metadata_key = self._build_key(queue_name, "metadata", item)

        async def _get(
            client: Redis | Pipeline, *args: Any
        ) -> tuple[float | None, dict[str, Any] | None]:
            # Get priority
            score = await client.zscore(queue_key, item)

            # Get metadata if exists
            metadata = await client.hgetall(metadata_key)
            metadata = (
                {k.decode(): v.decode() for k, v in metadata.items()}
                if metadata
                else None
            )

            return score, metadata

        return await self._execute_atomic("get_item", _get)

    async def get_items_by_priority(
        self,
        queue_name: str,
        start: int = 0,
        end: int = -1,
        with_priorities: bool = False,
        with_metadata: bool = False,
    ) -> list[str] | list[tuple[str, float]] | list[tuple[str, float, dict[str, Any]]]:
        """Get items ordered by priority.

        Args:
            queue_name: Name of the queue
            start: Start index
            end: End index (-1 for all)
            with_priorities: Include priority values
            with_metadata: Include metadata

        Returns:
            List of items, optionally with priorities and metadata
        """
        queue_key = self._build_key(queue_name)

        async def _get_range(client: Redis | Pipeline, *args: Any) -> list[Any]:
            # Get items with scores
            items = await client.zrange(
                queue_key, start, end, withscores=True, score_cast_func=float
            )

            if not items:
                return []

            if not with_priorities and not with_metadata:
                return [item.decode() for item, _ in items]

            if with_priorities and not with_metadata:
                return [(item.decode(), score) for item, score in items]

            # Get metadata for each item
            result = []
            for item, score in items:
                item_str = item.decode()
                metadata_key = self._build_key(queue_name, "metadata", item_str)
                metadata = await client.hgetall(metadata_key)
                metadata = (
                    {k.decode(): v.decode() for k, v in metadata.items()}
                    if metadata
                    else {}
                )
                result.append((item_str, score, metadata))

            return result

        return await self._execute_atomic("get_range", _get_range)

    async def get_items_by_priority_batch(
        self,
        queue_name: str,
        batch_size: int = 100,
        with_priorities: bool = False,
        with_metadata: bool = False,
    ) -> AsyncIterator[list[Any]]:
        """Get items in batches to manage memory.

        Args:
            queue_name: Name of the queue
            batch_size: Size of each batch
            with_priorities: Include priority values
            with_metadata: Include metadata

        Yields:
            Batches of items, optionally with priorities and metadata
        """
        start = 0
        while True:
            batch = await self.get_items_by_priority(
                queue_name,
                start,
                start + batch_size - 1,
                with_priorities,
                with_metadata,
            )
            if not batch:
                break

            yield batch
            start += batch_size

    async def update_priority(
        self, queue_name: str, item: str, new_priority: float
    ) -> None:
        """Update item's priority.

        Args:
            queue_name: Name of the queue
            item: Item to update
            new_priority: New priority value
        """
        queue_key = self._build_key(queue_name)

        async def _update(client: Redis | Pipeline, *args: Any) -> None:
            await client.zadd(queue_key, {item: new_priority})

        await self._execute_atomic("update_priority", _update)

    async def get_highest_priority(
        self, queue_name: str
    ) -> tuple[str | None, float | None, dict[str, Any] | None]:
        """Get highest priority item.

        Args:
            queue_name: Name of the queue

        Returns:
            Tuple of (item, priority, metadata) or (None, None, None) if queue empty
        """
        items = await self.get_items_by_priority(
            queue_name, 0, 0, with_priorities=True, with_metadata=True
        )
        if not items:
            return None, None, None
        return items[0]

    async def count_items(self, queue_name: str) -> int:
        """Count items in queue.

        Args:
            queue_name: Name of the queue

        Returns:
            Number of items
        """
        queue_key = self._build_key(queue_name)

        async def _count(client: Redis | Pipeline, *args: Any) -> int:
            return await client.zcard(queue_key)

        return await self._execute_atomic("count_items", _count)

    async def cleanup(self, queue_name: str) -> None:
        """Remove queue and all associated metadata.

        Args:
            queue_name: Name of the queue to cleanup
        """
        queue_key = self._build_key(queue_name)
        metadata_pattern = self._build_key(queue_name, "metadata", "*")

        async def _cleanup(client: Redis | Pipeline, *args: Any) -> None:
            # Delete queue
            await client.delete(queue_key)

            # Scan and delete all metadata
            cursor = 0
            while True:
                cursor, keys = await client.scan(
                    cursor, match=metadata_pattern, count=100
                )
                if keys:
                    await client.delete(*keys)
                if cursor == 0:
                    break

        await self._execute_atomic("cleanup", _cleanup)
