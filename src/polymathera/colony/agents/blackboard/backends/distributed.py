"""Distributed backend using StateManager."""
import time
from typing import Any

from pydantic import Field

from ....distributed import get_polymathera
from ....distributed.state_management import SharedState, StateManager
from ..backend import BlackboardBackend
from ..types import BlackboardEntry


class DistributedBackend(BlackboardBackend):
    """StateManager-based distributed backend.

    - All operations go through StateManager transactions
    - Query executes in ONE transaction (efficient)
    - Version tracking via StateManager's built-in versioning
    - TTL cleanup happens in query (not on every read)
    """

    def __init__(self, app_name: str, scope_id: str):
        self.app_name = app_name
        self.scope_id = scope_id
        self.state_manager: StateManager | None = None

    async def initialize(self) -> None:
        """Initialize StateManager."""

        class BlackboardState(SharedState):
            data: dict[str, dict[str, Any]] = Field(default_factory=dict)  # key -> entry dict

            @classmethod
            def get_state_key(cls, app_name: str, scope_id: str) -> str:
                return f"polymathera:serving:{app_name}:blackboard:{scope_id}"

        polymathera = get_polymathera()
        state_key = BlackboardState.get_state_key(self.app_name, self.scope_id)
        self.state_manager = await polymathera.get_state_manager(
            state_type=BlackboardState, state_key=state_key
        )

    async def read(self, key: str) -> BlackboardEntry | None:
        """Read entry."""
        async for state in self.state_manager.read_transaction():
            entry_dict = state.data.get(key)
            if not entry_dict:
                return None

            entry = self._dict_to_entry(entry_dict)

            # Check TTL
            if entry.ttl_seconds is not None:
                if time.time() - entry.created_at > entry.ttl_seconds:
                    # Expired - return None (cleanup happens in query or on next write)
                    return None

            return entry

    async def read_with_version(self, key: str) -> tuple[BlackboardEntry | None, str | None]:
        """Read entry with version token."""
        async for state in self.state_manager.read_transaction():
            entry_dict = state.data.get(key)
            if not entry_dict:
                return None, None

            entry = self._dict_to_entry(entry_dict)

            # Check TTL
            if entry.ttl_seconds is not None:
                if time.time() - entry.created_at > entry.ttl_seconds:
                    return None, None

            # Use entry version as token
            version_token = str(entry.version)
            return entry, version_token

    async def write(self, key: str, entry: BlackboardEntry) -> None:
        """Write entry."""
        async for state in self.state_manager.write_transaction():
            state.data[key] = self._entry_to_dict(entry)

    async def delete(self, key: str) -> None:
        """Delete entry."""
        async for state in self.state_manager.write_transaction():
            state.data.pop(key, None)

    async def list_keys(self) -> list[str]:
        """List all keys."""
        async for state in self.state_manager.read_transaction():
            return list(state.data.keys())

    async def clear(self) -> None:
        """Clear all data."""
        async for state in self.state_manager.write_transaction():
            state.data.clear()

    async def query(
        self,
        namespace: str | None = None,
        tags: set[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[BlackboardEntry]:
        """Query entries in ONE transaction (efficient).

        This is the key optimization - we read ALL data in one transaction
        and filter in-memory rather than multiple round-trips.
        """
        import fnmatch

        async for state in self.state_manager.read_transaction():
            now = time.time()
            results = []

            for key, entry_dict in state.data.items():
                entry = self._dict_to_entry(entry_dict)

                # Check TTL and skip expired
                if entry.ttl_seconds is not None:
                    if now - entry.created_at > entry.ttl_seconds:
                        continue

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
        async for state in self.state_manager.read_transaction():
            now = time.time()
            entry_count = 0
            oldest_age = None
            newest_age = None

            entries = []
            for entry_dict in state.data.values():
                entry = self._dict_to_entry(entry_dict)

                # Skip expired
                if entry.ttl_seconds is not None:
                    if now - entry.created_at > entry.ttl_seconds:
                        continue

                entries.append(entry)
                entry_count += 1

            if entries:
                oldest_entry = min(entries, key=lambda e: e.created_at)
                newest_entry = max(entries, key=lambda e: e.created_at)
                oldest_age = now - oldest_entry.created_at
                newest_age = now - newest_entry.created_at

            return {
                "backend_type": "DistributedBackend",
                "entry_count": entry_count,
                "oldest_entry_age": oldest_age,
                "newest_entry_age": newest_age,
            }

    async def read_batch(self, keys: list[str]) -> dict[str, BlackboardEntry]:
        """Read multiple entries efficiently in single transaction."""
        if not keys:
            return {}

        # Read all requested keys in ONE transaction
        async for state in self.state_manager.read_transaction():
            now = time.time()
            result = {}

            for key in keys:
                entry_dict = state.data.get(key)
                if not entry_dict:
                    continue

                entry = self._dict_to_entry(entry_dict)

                # Check TTL
                if entry.ttl_seconds is not None:
                    if now - entry.created_at > entry.ttl_seconds:
                        continue  # Skip expired

                result[key] = entry

            return result

    async def write_batch(self, entries: dict[str, BlackboardEntry]) -> None:
        """Write multiple entries efficiently in single transaction."""
        if not entries:
            return

        # Write all entries in ONE transaction
        async for state in self.state_manager.write_transaction():
            for key, entry in entries.items():
                state.data[key] = self._entry_to_dict(entry)

    async def commit_transaction(
        self,
        writes: dict[str, BlackboardEntry],
        deletes: set[str],
        version_tokens: dict[str, str],
    ) -> None:
        """Commit transaction atomically with optimistic locking."""
        from ..backend import ConcurrentModificationError

        async for state in self.state_manager.write_transaction():
            # Check versions first (optimistic locking)
            for key, expected_version in version_tokens.items():
                if key in writes:
                    entry_dict = state.data.get(key)
                    if entry_dict:
                        current_version = entry_dict.get("version", 0)
                        if str(current_version) != expected_version:
                            raise ConcurrentModificationError(
                                f"Key {key} was modified concurrently"
                            )

            # Apply writes
            for key, entry in writes.items():
                state.data[key] = self._entry_to_dict(entry)

            # Apply deletes
            for key in deletes:
                state.data.pop(key, None)

    def _entry_to_dict(self, entry: BlackboardEntry) -> dict[str, Any]:
        """Convert entry to dict for storage."""
        return {
            "key": entry.key,
            "value": entry.value,
            "version": entry.version,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
            "created_by": entry.created_by,
            "updated_by": entry.updated_by,
            "ttl_seconds": entry.ttl_seconds,
            "tags": list(entry.tags),
            "metadata": entry.metadata,
        }

    def _dict_to_entry(self, entry_dict: dict[str, Any]) -> BlackboardEntry:
        """Convert dict to entry."""
        return BlackboardEntry(
            key=entry_dict["key"],
            value=entry_dict["value"],
            version=entry_dict.get("version", 0),
            created_at=entry_dict.get("created_at", time.time()),
            updated_at=entry_dict.get("updated_at", time.time()),
            created_by=entry_dict.get("created_by"),
            updated_by=entry_dict.get("updated_by"),
            ttl_seconds=entry_dict.get("ttl_seconds"),
            tags=set(entry_dict.get("tags", [])),
            metadata=entry_dict.get("metadata", {}),
        )

