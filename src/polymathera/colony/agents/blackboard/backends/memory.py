"""In-memory backend for local/testing use."""

import time
import uuid
from typing import Any

from ..backend import BlackboardBackend
from ..types import BlackboardEntry


class InMemoryBackend(BlackboardBackend):
    """Fast in-memory backend.

    - Stores everything in a dict
    - No persistence
    - Fast queries via direct dict iteration
    - Version tracking via UUID tokens
    """

    def __init__(self):
        self.data: dict[str, BlackboardEntry] = {}
        self.versions: dict[str, str] = {}  # key -> version token

    async def initialize(self) -> None:
        """Initialize backend (no-op for memory)."""
        pass

    async def read(self, key: str) -> BlackboardEntry | None:
        """Read entry."""
        entry = self.data.get(key)
        if entry and self._is_expired(entry):
            await self.delete(key)
            return None
        return entry

    async def read_with_version(self, key: str) -> tuple[BlackboardEntry | None, str | None]:
        """Read entry with version token."""
        entry = await self.read(key)
        version = self.versions.get(key) if entry else None
        return entry, version

    async def write(self, key: str, entry: BlackboardEntry) -> None:
        """Write entry and update version."""
        self.data[key] = entry
        self.versions[key] = str(uuid.uuid4())

    async def delete(self, key: str) -> None:
        """Delete entry."""
        self.data.pop(key, None)
        self.versions.pop(key, None)

    async def list_keys(self) -> list[str]:
        """List all keys."""
        # Clean up expired entries first
        await self._cleanup_expired()
        return list(self.data.keys())

    async def clear(self) -> None:
        """Clear all data."""
        self.data.clear()
        self.versions.clear()

    async def query(
        self,
        namespace: str | None = None,
        tags: set[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[BlackboardEntry]:
        """Query entries efficiently via dict iteration."""
        import fnmatch

        # Clean up expired entries first
        await self._cleanup_expired()

        results = []
        for key, entry in self.data.items():
            # Namespace filter (glob pattern)
            if namespace and not fnmatch.fnmatch(key, namespace):
                continue

            # Tags filter (must match ALL tags)
            if tags and not tags.issubset(entry.tags):
                continue

            results.append(entry)

        # Sort by updated_at (most recent first)
        results.sort(key=lambda e: e.updated_at, reverse=True)

        # Pagination
        return results[offset : offset + limit]

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics."""
        await self._cleanup_expired()

        oldest_age = None
        newest_age = None
        now = time.time()

        if self.data:
            oldest_entry = min(self.data.values(), key=lambda e: e.created_at)
            newest_entry = max(self.data.values(), key=lambda e: e.created_at)
            oldest_age = now - oldest_entry.created_at
            newest_age = now - newest_entry.created_at

        return {
            "backend_type": "InMemoryBackend",
            "entry_count": len(self.data),
            "oldest_entry_age": oldest_age,
            "newest_entry_age": newest_age,
        }

    def _is_expired(self, entry: BlackboardEntry) -> bool:
        """Check if entry is expired."""
        if entry.ttl_seconds is None:
            return False
        return time.time() - entry.created_at > entry.ttl_seconds

    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired_keys = [
            key
            for key, entry in self.data.items()
            if entry.ttl_seconds is not None and now - entry.created_at > entry.ttl_seconds
        ]
        for key in expired_keys:
            await self.delete(key)
