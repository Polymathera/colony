"""Distributed set implementation."""

from collections.abc import AsyncIterator
from typing import Any

from redis.asyncio import Redis
from redis.asyncio.client import Pipeline

from .base import BaseStore


class DistributedSet(BaseStore):
    """
    Distributed set with support for:
    - Member management
    - Set operations (union, intersection)
    - Atomic operations
    - Namespace isolation
    - Batch operations via pipeline
    """

    async def add_member(self, set_name: str, member: str) -> None:
        """Add member to set.

        Args:
            set_name: Name of the set
            member: Member to add
        """
        full_key = self._build_key(set_name)

        async def _add(client: Redis | Pipeline, *args: Any) -> None:
            await client.sadd(full_key, member)

        await self._execute_atomic("add_member", _add)

    async def add_members(self, set_name: str, members: set[str]) -> None:
        """Add multiple members to set.

        Args:
            set_name: Name of the set
            members: Members to add
        """
        if not members:
            return

        full_key = self._build_key(set_name)

        async def _add_multiple(client: Redis | Pipeline, *args: Any) -> None:
            await client.sadd(full_key, *members)

        await self._execute_atomic("add_members", _add_multiple)

    async def remove_member(self, set_name: str, member: str) -> None:
        """Remove member from set.

        Args:
            set_name: Name of the set
            member: Member to remove
        """
        full_key = self._build_key(set_name)

        async def _remove(client: Redis | Pipeline, *args: Any) -> None:
            await client.srem(full_key, member)

        await self._execute_atomic("remove_member", _remove)

    async def remove_members(self, set_name: str, members: set[str]) -> None:
        """Remove multiple members from set.

        Args:
            set_name: Name of the set
            members: Members to remove
        """
        if not members:
            return

        full_key = self._build_key(set_name)

        async def _remove_multiple(client: Redis | Pipeline, *args: Any) -> None:
            await client.srem(full_key, *members)

        await self._execute_atomic("remove_members", _remove_multiple)

    async def get_members(self, set_name: str) -> set[str]:
        """Get all members of set.

        Args:
            set_name: Name of the set

        Returns:
            Set of members
        """
        full_key = self._build_key(set_name)

        async def _get_members(client: Redis | Pipeline, *args: Any) -> set[str]:
            members = await client.smembers(full_key)
            return {m.decode() for m in members}

        return await self._execute_atomic("get_members", _get_members)

    async def get_members_batch(
        self, set_name: str, batch_size: int = 100
    ) -> AsyncIterator[set[str]]:
        """Get members in batches to manage memory.

        Args:
            set_name: Name of the set
            batch_size: Size of each batch

        Yields:
            Batches of members
        """
        full_key = self._build_key(set_name)
        cursor = 0

        while True:

            async def _scan_members(
                client: Redis | Pipeline, *args: Any
            ) -> tuple[int, set[str]]:
                nonlocal cursor
                cursor, members = await client.sscan(full_key, cursor, count=batch_size)
                return cursor, {m.decode() for m in members}

            next_cursor, batch = await self._execute_atomic(
                "scan_members", _scan_members
            )
            if batch:
                yield batch

            cursor = next_cursor
            if cursor == 0:
                break

    async def get_union(self, set_names: list[str]) -> set[str]:
        """Get union of multiple sets.

        Args:
            set_names: Names of sets to union

        Returns:
            Union of all sets
        """
        if not set_names:
            return set()

        full_keys = [self._build_key(name) for name in set_names]

        async def _get_union(client: Redis | Pipeline, *args: Any) -> set[str]:
            members = await client.sunion(full_keys)
            return {m.decode() for m in members}

        return await self._execute_atomic("get_union", _get_union)

    async def get_intersection(self, set_names: list[str]) -> set[str]:
        """Get intersection of multiple sets.

        Args:
            set_names: Names of sets to intersect

        Returns:
            Intersection of all sets
        """
        if not set_names:
            return set()

        full_keys = [self._build_key(name) for name in set_names]

        async def _get_intersection(client: Redis | Pipeline, *args: Any) -> set[str]:
            members = await client.sinter(full_keys)
            return {m.decode() for m in members}

        return await self._execute_atomic("get_intersection", _get_intersection)

    async def move_member(self, source_set: str, dest_set: str, member: str) -> bool:
        """Move member from one set to another atomically.

        Args:
            source_set: Source set name
            dest_set: Destination set name
            member: Member to move

        Returns:
            True if member was moved, False if member wasn't in source set
        """
        source_key = self._build_key(source_set)
        dest_key = self._build_key(dest_set)

        async def _move(client: Redis | Pipeline, *args: Any) -> bool:
            return await client.smove(source_key, dest_key, member)

        return await self._execute_atomic("move_member", _move)

    async def count_members(self, set_name: str) -> int:
        """Count members in set.

        Args:
            set_name: Name of the set

        Returns:
            Number of members
        """
        full_key = self._build_key(set_name)

        async def _count(client: Redis | Pipeline, *args: Any) -> int:
            return await client.scard(full_key)

        return await self._execute_atomic("count_members", _count)

    async def is_member(self, set_name: str, member: str) -> bool:
        """Check if member exists in set.

        Args:
            set_name: Name of the set
            member: Member to check

        Returns:
            True if member exists, False otherwise
        """
        full_key = self._build_key(set_name)

        async def _is_member(client: Redis | Pipeline, *args: Any) -> bool:
            return await client.sismember(full_key, member)

        return await self._execute_atomic("is_member", _is_member)
