"""Redis backend using RedisOM for efficient indexing and querying."""
import time
from datetime import timedelta
from typing import Any

from ....distributed import get_polymathera
from ....distributed.redis_utils.client import RedisClient
from ....distributed.redis_utils.redis_om import IndexType, RedisIndex, RedisOM
from ..backend import BlackboardBackend, ConcurrentModificationError
from ..types import BlackboardEntry


# Define indexed entry model for Redis
IndexedBlackboardEntry = RedisIndex(
    prefix="blackboard_entries",
    exclude_nested=False,
    ttl=None,  # TTL handled per-entry
    query_builder_name="db",
    indices=[
        {"index_type": IndexType.SET, "field_path": "key"},
        {"index_type": IndexType.SET, "field_path": "created_by"},
        {"index_type": IndexType.SET, "field_path": "updated_by"},
        {"index_type": IndexType.SORTED, "field_path": "version"},
        {"index_type": IndexType.SORTED, "field_path": "created_at"},
        {"index_type": IndexType.SORTED, "field_path": "updated_at"},
        {"index_type": IndexType.SET, "field_path": "tags"},
    ],
)(BlackboardEntry)


class RedisBackend(BlackboardBackend):
    """Redis backend with efficient indexing and querying.

    Features:
    - RedisOM for schema and indexing
    - Native Redis queries for efficiency
    - Optimistic locking via version tokens
    - TTL handled by Redis natively
    - Batch operations via pipeline
    """

    def __init__(self, app_name: str, scope: str, scope_id: str):
        """Initialize Redis backend."""
        # TODO: Why both scope and scope_id needed? Keep just one?
        self.app_name = app_name
        self.scope = scope
        self.scope_id = scope_id
        self.namespace = f"{app_name}:blackboard:{scope}:{scope_id}"
        self.redis_client: RedisClient | None = None
        self.redis_om: RedisOM | None = None

    async def initialize(self) -> None:
        """Initialize Redis connection and RedisOM."""
        polymathera = get_polymathera()
        self.redis_client = await polymathera.get_redis_client()

        self.redis_om = RedisOM(
            redis_client=self.redis_client,
            namespace=self.namespace,
        )

    async def read(self, key: str) -> BlackboardEntry | None:
        """Read entry."""
        entry, _ = await self.redis_om.get(IndexedBlackboardEntry, key)
        return entry

    async def read_with_version(self, key: str) -> tuple[BlackboardEntry | None, str | None]:
        """Read entry with version token."""
        return await self.redis_om.get(IndexedBlackboardEntry, key)

    async def write(self, key: str, entry: BlackboardEntry) -> None:
        """Write entry with TTL."""
        ttl_seconds = entry.ttl_seconds if entry.ttl_seconds else None
        await self.redis_om.save(
            entry,
            key,
            update_if_exists=True,
            ttl=ttl_seconds,
            model_cls=IndexedBlackboardEntry,
        )

    async def delete(self, key: str) -> None:
        """Delete entry."""
        await self.redis_om.delete(IndexedBlackboardEntry, key)

    async def list_keys(self) -> list[str]:
        """List all keys."""
        return list(await self.redis_om.get_all_item_ids(IndexedBlackboardEntry))

    async def clear(self) -> None:
        """Clear all entries."""
        await self.redis_om.remove_all(IndexedBlackboardEntry)

    async def query(
        self,
        namespace: str | None = None,
        tags: set[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[BlackboardEntry]:
        """Query entries using Redis native queries.

        This is EFFICIENT - uses Redis indices, not full table scan.
        """
        import fnmatch

        db = IndexedBlackboardEntry.db

        # Build Redis query
        query_parts = []

        # Tags filter (Redis SET index)
        if tags:
            for tag in tags:
                query_parts.append(db.tags == tag)

        # Combine query parts
        if query_parts:
            redis_query = query_parts[0]
            for part in query_parts[1:]:
                redis_query = redis_query & part
        else:
            redis_query = None

        # Execute query and get all results
        if redis_query is not None:
            entries = []
            async for batch in self.redis_om.find_batches(
                model_cls=IndexedBlackboardEntry,
                query=redis_query,
                batch_size=1000,
            ):
                entries.extend(batch)
        else:
            # No Redis query - get all entries
            all_keys = await self.list_keys()
            entries = []
            for key in all_keys:
                entry = await self.read(key)
                if entry:
                    entries.append(entry)

        # Apply namespace filter (Redis doesn't support glob in queries, do it in-memory)
        if namespace:
            entries = [e for e in entries if fnmatch.fnmatch(e.key, namespace)]

        # Sort by updated_at (most recent first)
        entries.sort(key=lambda e: e.updated_at, reverse=True)

        # Pagination
        return entries[offset : offset + limit]

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics."""
        all_keys = await self.list_keys()
        entry_count = len(all_keys)

        oldest_age = None
        newest_age = None

        if entry_count > 0:
            # Sample entries to estimate ages
            sample_keys = all_keys[:min(100, entry_count)]
            entries = []
            for key in sample_keys:
                entry = await self.read(key)
                if entry:
                    entries.append(entry)

            if entries:
                now = time.time()
                oldest_entry = min(entries, key=lambda e: e.created_at)
                newest_entry = max(entries, key=lambda e: e.created_at)
                oldest_age = now - oldest_entry.created_at
                newest_age = now - newest_entry.created_at

        return {
            "backend_type": "RedisBackend",
            "entry_count": entry_count,
            "oldest_entry_age": oldest_age,
            "newest_entry_age": newest_age,
        }

    async def read_batch(self, keys: list[str]) -> dict[str, BlackboardEntry]:
        """Read multiple entries efficiently using Redis pipeline."""
        if not keys:
            return {}

        # Use RedisOM's get_batch for efficient pipeline reads
        results = await self.redis_om.get_batch(IndexedBlackboardEntry, keys)

        # Convert to dict, filtering out None entries
        batch_dict = {}
        for entry, _ in results:
            if entry:
                batch_dict[entry.key] = entry
        return batch_dict

    async def write_batch(self, entries: dict[str, BlackboardEntry]) -> None:
        """Write multiple entries efficiently using Redis pipeline."""
        if not entries:
            return

        entry_list = list(entries.values())
        ids = [e.key for e in entry_list]

        # Use RedisOM's save_batch for efficient pipeline writes
        await self.redis_om.save_batch(
            items=entry_list,
            ids=ids,
            update_if_exists=True,
            version_tokens=[],
            ttl=None,  # TTL is per-entry
            model_cls=IndexedBlackboardEntry,
        )

    async def commit_transaction(
        self,
        writes: dict[str, BlackboardEntry],
        deletes: set[str],
        version_tokens: dict[str, str],
    ) -> None:
        """Commit transaction atomically with optimistic locking."""
        # Use RedisOM's checkin mechanism for optimistic locking
        if writes:
            write_entries = list(writes.values())
            write_ids = [e.key for e in write_entries]
            write_versions = [version_tokens.get(e.key, "") for e in write_entries]

            results = await self.redis_om.checkin_items(
                items=write_entries,
                version_tokens=write_versions,
                ids=write_ids,
                model_cls=IndexedBlackboardEntry,
            )

            # Check for concurrent modifications
            for i, result in enumerate(results):
                if not result.updated and result.exists:
                    # Entry exists but wasn't updated = version mismatch
                    raise ConcurrentModificationError(
                        f"Key {write_ids[i]} was modified concurrently"
                    )

        # Apply deletes
        for key in deletes:
            await self.delete(key)