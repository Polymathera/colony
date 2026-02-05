"""Distributed lock implementation."""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from redis.asyncio import Redis
from redis.asyncio.client import Pipeline

from .base import BaseStore


class DistributedLock(BaseStore):
    """
    Distributed lock with support for:
    - Lock acquisition with timeout
    - Lock renewal
    - Lock metadata
    - Atomic operations
    - Namespace isolation
    - Reentrant locking (same owner)
    """

    def __init__(
        self,
        redis: Redis,
        namespace: str,
        default_timeout: int = 30,
        renewal_interval: int = 10,
    ):
        """Initialize lock manager.

        Args:
            redis: Redis client
            namespace: Namespace for lock isolation
            default_timeout: Default lock timeout in seconds
            renewal_interval: How often to renew locks in seconds
        """
        super().__init__(redis, namespace)
        self.default_timeout = default_timeout
        self.renewal_interval = renewal_interval
        self._renewal_tasks: dict[str, asyncio.Task] = {}

    def get_all_keys(self, resource: str) -> list[str]:
        """Get all keys associated with a resource."""
        return [self._build_key(resource), self._build_key(resource, "metadata")]

    @asynccontextmanager
    async def acquire(
        self,
        resource: str,
        owner: str | None = None,
        timeout: int | None = None,
        metadata: dict[str, Any] | None = None,
        auto_renewal: bool = True,
    ) -> AsyncIterator[bool]:
        """Acquire lock with context manager.

        Args:
            resource: Resource to lock
            owner: Lock owner ID (generated if not provided)
            timeout: Lock timeout in seconds
            metadata: Optional lock metadata
            auto_renewal: Whether to auto-renew lock

        Yields:
            True if lock acquired, False otherwise
        """
        owner = owner or str(uuid.uuid4())
        timeout = timeout or self.default_timeout
        acquired = False

        try:
            acquired = await self._acquire_lock(resource, owner, timeout, metadata)
            if acquired and auto_renewal:
                await self._start_renewal(resource, owner, timeout, metadata)
            yield acquired
        finally:
            if acquired:
                if auto_renewal:
                    await self._stop_renewal(resource)
                await self.release(resource, owner)

    async def _acquire_lock(
        self,
        resource: str,
        owner: str,
        timeout: int,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Acquire lock with retries.

        Args:
            resource: Resource to lock
            owner: Lock owner ID
            timeout: Lock timeout in seconds
            metadata: Optional lock metadata

        Returns:
            True if lock acquired, False otherwise
        """
        lock_key = self._build_key(resource)
        metadata_key = self._build_key(resource, "metadata") if metadata else None
        expiry = int(time.time() + timeout)

        async def _acquire(client: Redis | Pipeline, *args: Any) -> bool:
            # Check if we already own the lock (reentrant)
            current_owner = await client.get(lock_key)
            if current_owner and current_owner.decode() == owner:
                # Extend timeout
                await client.expire(lock_key, timeout)
                if metadata_key:
                    await client.expire(metadata_key, timeout)
                return True

            # Try to acquire lock
            acquired = await client.set(
                lock_key,
                owner,
                nx=True,  # Only set if not exists
                ex=timeout,
            )

            if acquired and metadata:
                await client.hset(
                    metadata_key,
                    mapping={"owner": owner, "expiry": str(expiry), **metadata},
                )
                await client.expire(metadata_key, timeout)

            return bool(acquired)

        return await self._execute_atomic("acquire_lock", _acquire)

    async def release(self, resource: str, owner: str) -> bool:
        """Release lock if owned.

        Args:
            resource: Resource to unlock
            owner: Lock owner ID

        Returns:
            True if lock was released, False if not owned
        """
        lock_key = self._build_key(resource)
        metadata_key = self._build_key(resource, "metadata")

        async def _release(client: Redis | Pipeline, *args: Any) -> bool:
            # Only release if we own the lock
            current_owner = await client.get(lock_key)
            if not current_owner or current_owner.decode() != owner:
                return False

            # Release lock and metadata
            await client.delete(lock_key, metadata_key)
            return True

        return await self._execute_atomic("release_lock", _release)

    async def get_lock_info(
        self, resource: str
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Get lock owner and metadata.

        Args:
            resource: Resource to check

        Returns:
            Tuple of (owner, metadata) or (None, None) if not locked
        """
        lock_key = self._build_key(resource)
        metadata_key = self._build_key(resource, "metadata")

        async def _get_info(
            client: Redis | Pipeline, *args: Any
        ) -> tuple[str | None, dict[str, Any] | None]:
            owner = await client.get(lock_key)
            if not owner:
                return None, None

            metadata = await client.hgetall(metadata_key)
            return (
                owner.decode(),
                {k.decode(): v.decode() for k, v in metadata.items()}
                if metadata
                else None,
            )

        return await self._execute_atomic("get_lock_info", _get_info)

    async def _start_renewal(
        self,
        resource: str,
        owner: str,
        timeout: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Start background task to renew lock.

        Args:
            resource: Resource to renew lock for
            owner: Lock owner ID
            timeout: Lock timeout in seconds
            metadata: Optional lock metadata
        """
        if resource in self._renewal_tasks:
            return

        async def renew_lock() -> None:
            while True:
                try:
                    await asyncio.sleep(self.renewal_interval)
                    await self._acquire_lock(resource, owner, timeout, metadata)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Failed to renew lock: {e}")
                    break

        self._renewal_tasks[resource] = asyncio.create_task(renew_lock())

    async def _stop_renewal(self, resource: str) -> None:
        """Stop lock renewal task.

        Args:
            resource: Resource to stop renewal for
        """
        if resource in self._renewal_tasks:
            task = self._renewal_tasks.pop(resource)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def cleanup(self) -> None:
        """Cleanup all locks in this namespace."""
        pattern = f"{self.namespace}:*"

        async def _cleanup(client: Redis | Pipeline, *args: Any) -> None:
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)
                if keys:
                    await client.delete(*keys)
                if cursor == 0:
                    break

        await self._execute_atomic("cleanup", _cleanup)

        # Cancel all renewal tasks
        for resource in list(self._renewal_tasks.keys()):
            await self._stop_renewal(resource)
