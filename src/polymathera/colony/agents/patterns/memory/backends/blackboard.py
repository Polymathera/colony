"""Blackboard-based storage backend for the memory system.

This is the default storage backend that delegates to EnhancedBlackboard.
It's suitable for all memory levels and provides:
- Distributed state via StateManager
- OCC transactions
- Event streaming
- Tag-based queries

Why default: The blackboard infrastructure already handles distributed state,
transactions, and event streaming. Using it as the storage backend avoids
duplicating this complexity.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TYPE_CHECKING

from ....blackboard.blackboard import EnhancedBlackboard
from ....blackboard.types import BlackboardEntry, BlackboardEvent, KeyPatternFilter

if TYPE_CHECKING:
    from ....base import Agent

logger = logging.getLogger(__name__)


class BlackboardStorageBackend:
    """Storage backend that delegates to EnhancedBlackboard.

    This is the default backend for MemoryCapability. It provides:
    - Key-value storage with rich metadata
    - Tag-based filtering
    - TTL support
    - OCC transactions (via the underlying blackboard)
    - Event streaming (via the underlying blackboard)

    The backend is scoped to a specific scope_id (e.g., "agent:abc123:stm").
    All operations are scoped to this namespace.

    Attributes:
        scope_id: Blackboard scope this backend manages
        blackboard: EnhancedBlackboard instance
        agent_id: Agent ID for attribution
    """

    def __init__(
        self,
        scope_id: str,
        blackboard: EnhancedBlackboard,
        agent_id: str | None = None,
    ):
        """Initialize blackboard storage backend.

        Args:
            scope_id: Scope ID for this backend (used as key prefix)
            blackboard: EnhancedBlackboard to delegate to
            agent_id: Agent ID for attribution on writes
        """
        self._scope_id = scope_id
        self.blackboard = blackboard
        self.agent_id = agent_id

    @property
    def scope_id(self) -> str:
        """Scope ID this backend is bound to."""
        return self._scope_id

    async def write(
        self,
        key: str,
        value: dict[str, Any],
        metadata: dict[str, Any],
        tags: set[str] | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        """Write an entry to the blackboard.

        Args:
            key: Storage key (should be in this scope's namespace)
            value: Serialized data (dict from model_dump())
            metadata: Entry metadata
            tags: Tags for categorization and filtering
            ttl_seconds: Time to live (None = no expiration)
        """
        # Session_id is automatically added to tags/metadata by EnhancedBlackboard.write()
        await self.blackboard.write(
            key=key,
            value=value,
            created_by=self.agent_id,
            tags=tags,
            ttl_seconds=ttl_seconds,
            metadata=metadata,
        )

        logger.debug(f"BlackboardStorageBackend: wrote {key}")

    async def read(self, key: str) -> BlackboardEntry | None:
        """Read a single entry by key.

        Args:
            key: Storage key

        Returns:
            Entry if found and not expired, None otherwise
        """
        entry = await self.blackboard.read_entry(key, agent_id=self.agent_id)
        return entry

    async def query(
        self,
        pattern: str | None = None,
        tags: set[str] | None = None,
        time_range: tuple[float, float] | None = None,
        limit: int = 100,
    ) -> list[BlackboardEntry]:
        """Query entries in this scope.

        Args:
            pattern: Key pattern (e.g., "scope:*"). If None, uses scope_id:*
            tags: Filter by tags (entries must have ALL tags)
            time_range: (start_timestamp, end_timestamp) filter
            limit: Maximum entries to return

        Returns:
            List of matching entries
        """
        # Use scope pattern if no pattern provided
        effective_pattern = pattern or "*"

        entries = await self.blackboard.query(
            namespace=effective_pattern,
            tags=tags,
            limit=limit,
        )

        # Apply time_range filter if specified
        if time_range is not None:
            start, end = time_range
            entries = [
                e for e in entries
                if start <= e.created_at <= end
            ]

        return entries

    async def delete(self, key: str) -> bool:
        """Delete an entry by key.

        Args:
            key: Storage key

        Returns:
            True if deleted, False if not found
        """
        # Check if exists first
        entry = await self.read(key)
        if entry is None:
            return False

        await self.blackboard.delete(key, agent_id=self.agent_id)
        logger.debug(f"BlackboardStorageBackend: deleted {key}")
        return True

    async def count(self) -> int:
        """Count total entries in this scope.

        Returns:
            Number of entries
        """
        entries = await self.query(limit=100000)  # High limit for counting
        return len(entries)

    async def clear(self) -> int:
        """Delete all entries in this scope.

        Returns:
            Number of entries deleted
        """
        entries = await self.query(limit=100000)
        count = 0

        for entry in entries:
            if await self.delete(entry.key):
                count += 1

        logger.info(f"BlackboardStorageBackend: cleared {count} entries from {self._scope_id}")
        return count

    async def stream_events_to_queue(
        self,
        event_queue: asyncio.Queue[BlackboardEvent],
        key_pattern: str,
        consumer_group: str | None = None,
        consumer_name: str | None = None,
    ) -> None:
        """Subscribe to write events matching a key pattern.

        When ``consumer_group`` is None (default), delegates to the underlying
        blackboard's pub-sub event streaming. All subscribers receive all events.
        All write() operations automatically emit events via the blackboard.

        When ``consumer_group`` is provided, uses Redis Streams consumer groups
        for exactly-once delivery across multiple consumers. This code path is
        used by the VCM's ``BlackboardContextPageSource`` to ensure each event
        is processed by exactly one VCM replica. Other subscribers (agents,
        capabilities) continue using standard pub-sub.

        Args:
            event_queue: Queue to receive BlackboardEvent objects
            key_pattern: Pattern to filter events (e.g., ``"scope:DataType:*"``)
            consumer_group: Optional consumer group name for exactly-once delivery.
            consumer_name: This consumer's name within the group (required if
                ``consumer_group`` is provided).
        """
        if consumer_group is not None:
            # Redis Streams code path for exactly-once delivery (VCM replicas).
            # Bypass blackboard's pub-sub and use XREADGROUP directly.
            success = await self.blackboard.stream_events_via_consumer_group(
                event_queue, consumer_group, consumer_name or "default",
            )
            if success:
                return

        # Standard pub-sub path (default for all non-VCM subscribers)
        self.blackboard.stream_events_to_queue(
            event_queue,
            KeyPatternFilter(pattern=key_pattern),
        )


class BlackboardStorageBackendFactory:
    """Factory for creating BlackboardStorageBackend instances.

    This implements the StorageBackendFactory protocol, allowing MemoryCapability
    to create storage backends for arbitrary scopes without hardcoding the
    BlackboardStorageBackend class.

    The factory uses the agent's get_blackboard() method to get or create
    blackboards for different scopes.
    """

    def __init__(self, agent: "Agent"):
        """Initialize the factory.

        Args:
            agent: Agent to create blackboards from
        """
        self._agent = agent

    async def create_for_scope(self, scope_id: str) -> BlackboardStorageBackend:
        """Create a BlackboardStorageBackend for the given scope.

        Args:
            scope_id: Scope ID to create backend for

        Returns:
            BlackboardStorageBackend instance for the scope
        """
        blackboard = await self._agent.get_blackboard(
            scope_id=scope_id,
        )
        return BlackboardStorageBackend(
            scope_id=scope_id,
            blackboard=blackboard,
            agent_id=self._agent.agent_id,
        )
