"""Storage backend abstraction for blackboard.

Backends are responsible for:
1. Storing and retrieving entries
2. Efficient querying with filters
3. Transactions with optimistic locking
4. TTL enforcement (backend-specific strategy)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .types import BlackboardEntry


@dataclass
class BlackboardTransaction:
    """Transaction context for atomic operations.

    Buffers operations and commits atomically.
    Supports optimistic locking via version tokens.

    Example:
        async with blackboard.transaction() as txn:
            val = await txn.read("counter")
            await txn.write("counter", val + 1)
            # Commits automatically on __aexit__
            # Rolls back on exception
    """

    backend: BlackboardBackend
    reads: dict[str, BlackboardEntry] = field(default_factory=dict)
    writes: dict[str, BlackboardEntry] = field(default_factory=dict)
    deletes: set[str] = field(default_factory=set)
    version_tokens: dict[str, str] = field(default_factory=dict)  # For optimistic locking
    committed: bool = False
    rolled_back: bool = False

    async def read_value(self, key: str) -> Any | None:
        """Read value within transaction."""
        entry = await self.read(key)
        return entry.value if entry else None

    async def write_value(self, key: str, value: Any) -> None:
        """Write value within transaction."""
        entry = BlackboardEntry(key=key, value=value)
        await self.write(key, entry)

    async def read(self, key: str) -> BlackboardEntry | None:
        """Read entry within transaction."""
        # Check local buffers first
        if key in self.writes:
            return self.writes[key]
        if key in self.deletes:
            return None

        # Cache backend read
        if key not in self.reads:
            entry, version = await self.backend.read_with_version(key)
            if entry:
                self.reads[key] = entry
                if version:
                    self.version_tokens[key] = version
        return self.reads.get(key)

    async def write(self, key: str, entry: BlackboardEntry) -> None:
        """Write entry within transaction."""
        self.writes[key] = entry
        self.deletes.discard(key)

    async def delete(self, key: str) -> None:
        """Delete entry within transaction."""
        self.deletes.add(key)
        self.writes.pop(key, None)

    async def commit(self) -> None:
        """Commit transaction atomically."""
        if self.committed or self.rolled_back:
            raise RuntimeError("Transaction already finalized")

        # Commit via backend (supports optimistic locking)
        await self.backend.commit_transaction(
            writes=self.writes,
            deletes=self.deletes,
            version_tokens=self.version_tokens,
        )
        self.committed = True

    async def rollback(self) -> None:
        """Rollback transaction."""
        if self.committed or self.rolled_back:
            raise RuntimeError("Transaction already finalized")

        self.reads.clear()
        self.writes.clear()
        self.deletes.clear()
        self.version_tokens.clear()
        self.rolled_back = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await self.commit()
        else:
            await self.rollback()
        return False


class BlackboardBackend(ABC):
    """Abstract storage backend.

    Each backend implements efficient operations for its storage mechanism:
    - InMemoryBackend: Fast dict operations
    - DistributedBackend: StateManager with single-transaction queries
    - RedisBackend: Redis native queries with indexing
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize backend."""
        ...

    @abstractmethod
    async def read(self, key: str) -> BlackboardEntry | None:
        """Read entry by key.

        Returns:
            Entry if exists, None otherwise
        """
        ...

    @abstractmethod
    async def read_with_version(self, key: str) -> tuple[BlackboardEntry | None, str | None]:
        """Read entry with version token for optimistic locking.

        Returns:
            (entry, version_token) tuple
        """
        ...

    @abstractmethod
    async def write(self, key: str, entry: BlackboardEntry) -> None:
        """Write entry."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete entry."""
        ...

    @abstractmethod
    async def list_keys(self) -> list[str]:
        """List all keys."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all data."""
        ...

    @abstractmethod
    async def query(
        self,
        namespace: str | None = None,
        tags: set[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[BlackboardEntry]:
        """Query entries efficiently using backend-specific mechanisms.

        Args:
            namespace: Namespace pattern (glob-style, e.g., "agent:*:results")
            tags: Tags to filter by (must match ALL tags)
            limit: Maximum number of entries
            offset: Pagination offset

        Returns:
            List of matching entries (backend handles filtering efficiently)
        """
        ...

    @abstractmethod
    async def get_statistics(self) -> dict[str, Any]:
        """Get backend-specific statistics.

        Returns:
            Dict with stats like entry_count, memory_bytes, etc.
        """
        ...

    async def read_batch(self, keys: list[str]) -> dict[str, BlackboardEntry]:
        """Read multiple entries efficiently.

        Default implementation - backends should override for better performance.

        Args:
            keys: List of keys to read

        Returns:
            Dict mapping keys to entries (missing keys are omitted)
        """
        result = {}
        for key in keys:
            entry = await self.read(key)
            if entry:
                result[key] = entry
        return result

    async def write_batch(self, entries: dict[str, BlackboardEntry]) -> None:
        """Write multiple entries efficiently.

        Default implementation - backends should override for better performance.

        Args:
            entries: Dict mapping keys to entries
        """
        for key, entry in entries.items():
            await self.write(key, entry)

    async def transaction(self) -> BlackboardTransaction:
        """Start transaction for atomic operations.

        Returns:
            Transaction context manager
        """
        return BlackboardTransaction(backend=self)

    async def commit_transaction(
        self,
        writes: dict[str, BlackboardEntry],
        deletes: set[str],
        version_tokens: dict[str, str],
    ) -> None:
        """Commit transaction atomically with optimistic locking.

        Args:
            writes: Entries to write
            deletes: Keys to delete
            version_tokens: Version tokens for optimistic locking

        Raises:
            ConcurrentModificationError: If version check fails
        """
        # Default implementation: apply operations sequentially
        # Backends can override for atomic batch operations

        # Check versions first (optimistic locking)
        for key, expected_version in version_tokens.items():
            if key in writes:
                current_entry, current_version = await self.read_with_version(key)
                if current_version and current_version != expected_version:
                    raise ConcurrentModificationError(
                        f"Key {key} was modified concurrently (expected version {expected_version}, got {current_version})"
                    )

        # Apply writes
        for key, entry in writes.items():
            await self.write(key, entry)

        # Apply deletes
        for key in deletes:
            await self.delete(key)


class ConcurrentModificationError(Exception):
    """Raised when optimistic locking detects concurrent modification."""

    pass
