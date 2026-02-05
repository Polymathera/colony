"""Production-grade blackboard combining backend, events, and policies.

This is the main interface that users interact with. It orchestrates:
- Backend storage (memory, distributed, Redis)
- Event notifications (TRUE pub-sub via Redis)
- Policy enforcement (access, eviction, validation)
- Rich metadata (versioning, TTL, tags, audit trail)

Ambient transactions:
- `async with blackboard.transaction()` enables an *ambient transaction*.
- While an ambient transaction is active, `read/write/delete` transparently route
  through the active transaction buffers (no need to thread a txn handle).
- Use `read_tx/write_tx/delete_tx` to make transactional intent explicit.
"""
from __future__ import annotations

import time
import asyncio
import contextvars
from typing import Any, Awaitable, Callable, AsyncIterator

from ...utils import setup_logger
from .backend import BlackboardBackend, BlackboardTransaction, ConcurrentModificationError
from .backends import DistributedBackend, InMemoryBackend, RedisBackend
from .events import EventBus
from .types import (
    AccessPolicy,
    BlackboardEntry,
    BlackboardEvent,
    BlackboardScope,
    EventFilter,
    EventTypeFilter,
    CombinationFilter,
    KeyPatternFilter,
    EvictionPolicy,
    ValidationPolicy,
    NoOpAccessPolicy,
    LRUEvictionPolicy,
    NoOpValidationPolicy
)

logger = setup_logger(__name__)


# Ambient transaction stack for the current async context.
# The top-of-stack transaction is used by EnhancedBlackboard.read/write/delete to
# transparently route operations through transaction buffers.
_AMBIENT_BLACKBOARD_TX_STACK: contextvars.ContextVar[list[BlackboardTransaction]] = contextvars.ContextVar(
    "_AMBIENT_BLACKBOARD_TX_STACK",
    default=[],
)


def _get_ambient_tx() -> BlackboardTransaction | None:
    stack = _AMBIENT_BLACKBOARD_TX_STACK.get()
    return stack[-1] if stack else None



# TODO: Merge this with the BlackboardTransaction class.
class _AmbientBlackboardTransaction:
    """Wrap a `BlackboardTransaction` and make it ambient for the duration of the context.

    Backends commit writes/deletes directly, bypassing `EnhancedBlackboard.write/delete`.
    To keep event-driven policies consistent, this wrapper emits events after a
    successful commit.

    Once you make transactions “ambient”, you must also ensure:
    - Metadata/version semantics remain identical to non-transactional `write(...)`
    - Events are still emitted (otherwise event-driven policies break), even though
    backend commits bypass `EnhancedBlackboard.write(...)`.

    To keep event-driven architecture correct, the ambient transaction wrapper emits:
    - "`write`" events for all buffered writes
    - "`delete`" events for all buffered deletes
    **after** a successful `commit(...)`.
    """

    def __init__(self, blackboard: EnhancedBlackboard, txn: BlackboardTransaction):
        self.blackboard = blackboard
        self.txn = txn
        self._token: contextvars.Token[list[BlackboardTransaction]] | None = None

    # Expose underlying txn internals for integrations (e.g., ActionDispatcher conflict checks).
    @property
    def version_tokens(self) -> dict[str, str]:
        return self.txn.version_tokens

    @property
    def reads(self) -> dict[str, BlackboardEntry]:
        return self.txn.reads

    @property
    def writes(self) -> dict[str, BlackboardEntry]:
        return self.txn.writes

    @property
    def deletes(self) -> set[str]:
        return self.txn.deletes

    # Proxy common transaction methods (so callers can still do txn.read/txn.write).
    async def read(self, key: str):
        return await self.txn.read(key)

    async def write(self, key: str, entry: BlackboardEntry) -> None:
        return await self.txn.write(key, entry)

    async def delete(self, key: str) -> None:
        return await self.txn.delete(key)

    async def read_value(self, key: str) -> Any | None:
        return await self.txn.read_value(key)

    async def write_value(self, key: str, value: Any) -> None:
        return await self.txn.write_value(key, value)

    async def commit(self) -> None:
        return await self.txn.commit()

    async def rollback(self) -> None:
        return await self.txn.rollback()

    async def __aenter__(self):
        # Push onto ambient stack
        stack = list(_AMBIENT_BLACKBOARD_TX_STACK.get())
        stack.append(self.txn)
        self._token = _AMBIENT_BLACKBOARD_TX_STACK.set(stack)
        await self.txn.__aenter__()
        # Yield the underlying transaction to callers/dispatcher. The wrapper remains
        # responsible for ambient routing + event emission on __aexit__.
        return self.txn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            ok = await self.txn.__aexit__(exc_type, exc_val, exc_tb)
            if exc_type is None:
                await self._emit_events_after_commit()
            return ok
        finally:
            if self._token is not None:
                _AMBIENT_BLACKBOARD_TX_STACK.reset(self._token)
                self._token = None

    async def _emit_events_after_commit(self) -> None:
        if self.blackboard.event_bus is None:
            return

        # Writes: emit "write" events with old_value from reads (if present)
        for key, entry in self.txn.writes.items():
            old_entry = self.txn.reads.get(key)
            old_value = old_entry.value if old_entry else None
            await self.blackboard.emit_event(
                BlackboardEvent(
                    event_type="write",
                    key=key,
                    value=entry.value,
                    version=entry.version,
                    old_value=old_value,
                    agent_id=entry.updated_by,
                    tags=entry.tags,
                    metadata=entry.metadata or {},
                )
            )

        # Deletes: emit "delete" events with old_value from reads (if present)
        for key in self.txn.deletes:
            old_entry = self.txn.reads.get(key)
            old_value = old_entry.value if old_entry else None
            await self.blackboard.emit_event(
                BlackboardEvent(
                    event_type="delete",
                    key=key,
                    value=None,
                    version=old_entry.version if old_entry else 0,
                    old_value=old_value,
                    agent_id=None,
                    tags=old_entry.tags if old_entry else set(),
                    metadata={},
                )
            )


class EnhancedBlackboard:
    """Production-grade blackboard with all the bells and whistles.

    Features:
    - Pluggable backends (memory, distributed, Redis)
    - Event-driven notifications via Redis pub-sub
    - Policy-based customization (access, eviction, validation)
    - Transactions with optimistic locking
    - Rich metadata (TTL, tags, versioning)
    - Efficient backend-specific queries

    Example:
        ```python
        # Create blackboard with custom policies
        board = EnhancedBlackboard(
            app_name="my-app",
            scope=BlackboardScope.SHARED,
            scope_id="team-1",
            access_policy=MyAccessPolicy(),
            validation_policy=SchemaValidator(MySchema),
        )
        await board.initialize()

        # Write with metadata
        await board.write(
            "analysis_results",
            my_results,
            created_by="agent-123",
            tags={"analysis", "final"},
            ttl_seconds=3600,
        )

        # Subscribe to changes
        async def on_result_updated(event: BlackboardEvent):
            print(f"Result updated: {event.value}")

        board.subscribe(on_result_updated, filter=KeyPatternFilter("*_results"))

        # Atomic transaction
        async with board.transaction() as txn:
            counter = await txn.read("counter") or 0
            await txn.write("counter", counter + 1)

        # Query by namespace
        results = await board.query(namespace="agent:*:results")

        # Introspection
        stats = await board.get_statistics()
        print(f"Total entries: {stats['entry_count']}")
        ```
    """

    def __init__(
        self,
        app_name: str,
        scope: BlackboardScope = BlackboardScope.LOCAL,
        scope_id: str = "default",
        # Policy customization points
        access_policy: AccessPolicy | None = None,
        eviction_policy: EvictionPolicy | None = None,
        validation_policy: ValidationPolicy | None = None,
        # Backend selection
        backend: BlackboardBackend | None = None,
        backend_type: str | None = None,  # "memory", "distributed", "redis"
        # Event system
        enable_events: bool = True,
        max_event_queue_size: int = 1000,
        # Resource limits
        max_entries: int | None = None,
    ):
        """Initialize enhanced blackboard.

        Args:
            app_name: Application name for namespacing
            scope: Visibility scope (LOCAL, SHARED, GLOBAL)
            scope_id: Scope identifier for SHARED/GLOBAL scopes
            access_policy: Policy for access control
            eviction_policy: Policy for evicting entries when memory is constrained
            validation_policy: Policy for validating values before write
            backend: Backend instance (if None, created based on backend_type)
            backend_type: Type of backend to create ("memory", "distributed", "redis")
            enable_events: Whether to enable event system
            max_event_queue_size: Maximum size of event queue
            max_entries: Maximum number of entries (None = unlimited)
        """
        self.app_name = app_name
        self.scope = scope
        self.scope_id = scope_id

        # Policies (use defaults if not provided)

        self.access_policy = access_policy or NoOpAccessPolicy()
        self.eviction_policy = eviction_policy or LRUEvictionPolicy()
        self.validation_policy = validation_policy or NoOpValidationPolicy()

        # Storage backend
        self.backend = backend
        self.backend_type = backend_type

        # Event system
        self.event_bus = (
            EventBus(
                app_name=app_name,
                scope=scope.value,
                scope_id=scope_id,
                max_queue_size=max_event_queue_size,
                distributed=(scope != BlackboardScope.LOCAL),
            )
            if enable_events
            else None
        )

        # Resource limits
        self.max_entries = max_entries

        self._initialized = False
        self._subscribed_callbacks: list[Callable[[BlackboardEvent], None]] = []

    async def initialize(self) -> None:
        """Initialize blackboard."""
        if self._initialized:
            return

        # Create default backend if not provided
        if self.backend is None:
            # Determine backend type
            if self.backend_type is None:
                # Auto-select based on scope
                if self.scope == BlackboardScope.LOCAL:
                    self.backend_type = "memory"
                else:
                    self.backend_type = "distributed"

            # Create backend
            if self.backend_type == "memory":
                self.backend = InMemoryBackend()
            elif self.backend_type == "distributed":
                self.backend = DistributedBackend(
                    app_name=self.app_name, scope=self.scope.value, scope_id=self.scope_id
                )
            elif self.backend_type == "redis":
                self.backend = RedisBackend(
                    app_name=self.app_name, scope=self.scope.value, scope_id=self.scope_id
                )
            else:
                raise ValueError(f"Unknown backend type: {self.backend_type}")

        await self.backend.initialize()

        # Start event bus
        if self.event_bus:
            await self.event_bus.start()

        self._initialized = True
        logger.info(
            f"Initialized EnhancedBlackboard (app={self.app_name}, scope={self.scope}, "
            f"scope_id={self.scope_id}, backend={type(self.backend).__name__})"
        )

    async def write(
        self,
        key: str,
        value: Any,
        created_by: str | None = None,
        tags: set[str] | None = None,
        ttl_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write to blackboard with rich metadata.

        Args:
            key: Key to write
            value: Value to write
            created_by: Agent ID that created this entry
            tags: Tags for querying
            ttl_seconds: Time-to-live in seconds
            metadata: Additional metadata

        Raises:
            PermissionError: If access policy denies write
            ValidationError: If validation policy rejects value
        """
        await self.initialize()

        # Ambient transaction routing: buffer writes and commit atomically on __aexit__.
        if _get_ambient_tx() is not None:
            await self.write_tx(
                key=key,
                value=value,
                created_by=created_by,
                tags=tags,
                ttl_seconds=ttl_seconds,
                metadata=metadata,
            )
            return

        # Access control
        if self.access_policy and not await self.access_policy.can_write(
            created_by or "unknown", key, value, self.scope_id
        ):
            raise PermissionError(f"Access denied for write: {key}")

        # Validation
        if self.validation_policy:
            await self.validation_policy.validate(key, value, metadata or {})

        # Check resource limits
        await self._check_resource_limits()

        # Auto-include session_id for distributed traceability
        # This ensures ALL blackboard writes include session_id when available
        from ..sessions.context import get_current_session_id
        session_id = (metadata or {}).get("session_id") or get_current_session_id()

        effective_metadata = metadata or {}
        if session_id and "session_id" not in effective_metadata:
            effective_metadata = {**effective_metadata, "session_id": session_id}

        effective_tags = tags or set()
        if session_id:
            effective_tags = effective_tags | {f"session:{session_id}"}

        # Get existing entry from backend
        old_entry = await self.backend.read(key)
        old_value = old_entry.value if old_entry else None

        # Create new entry
        now = time.time()
        entry = BlackboardEntry(
            key=key,
            value=value,
            version=(old_entry.version + 1) if old_entry else 0,
            created_at=old_entry.created_at if old_entry else now,
            updated_at=now,
            created_by=old_entry.created_by if old_entry else created_by,
            updated_by=created_by,
            ttl_seconds=ttl_seconds,
            tags=effective_tags,
            metadata=effective_metadata,
        )

        # Store in backend
        await self.backend.write(key, entry)

        # Emit event with session_id included
        await self.emit_event(
            BlackboardEvent(
                event_type="write",
                key=key,
                value=value,
                version=entry.version,
                old_value=old_value,
                agent_id=created_by,
                tags=effective_tags,
                metadata=effective_metadata,
            )
        )

        logger.debug(f"Wrote to blackboard: {key} (version={entry.version})")

    async def read(self, key: str, agent_id: str | None = None) -> Any | None:
        """Read from blackboard.

        Args:
            key: Key to read
            agent_id: Agent ID requesting read (for access control)

        Returns:
            Value if exists and not expired, None otherwise

        Raises:
            PermissionError: If access policy denies read
        """
        await self.initialize()

        # Ambient transaction routing: read from transaction buffers when active.
        if _get_ambient_tx() is not None:
            return await self.read_tx(key, agent_id=agent_id)

        # Access control
        if self.access_policy and not await self.access_policy.can_read(
            agent_id or "unknown", key, self.scope_id
        ):
            raise PermissionError(f"Access denied for read: {key}")

        entry = await self.backend.read(key)
        if entry is None:
            return None

        # Check TTL (backend-specific, but we double-check here)
        if entry.ttl_seconds is not None:
            if time.time() - entry.created_at > entry.ttl_seconds:
                # Expired
                await self.delete(key, agent_id=agent_id)
                return None

        return entry.value

    async def read_entry(self, key: str, agent_id: str | None = None) -> BlackboardEntry | None:
        """Read full entry with metadata.

        Args:
            key: Key to read
            agent_id: Agent ID requesting read (for access control)

        Returns:
            BlackboardEntry if exists and not expired, None otherwise

        Raises:
            PermissionError: If access policy denies read
        """
        await self.initialize()

        # Access control
        if self.access_policy and not await self.access_policy.can_read(
            agent_id or "unknown", key, self.scope_id
        ):
            raise PermissionError(f"Access denied for read: {key}")

        entry = await self.backend.read(key)
        if entry is None:
            return None

        # Check TTL
        if entry.ttl_seconds is not None:
            if time.time() - entry.created_at > entry.ttl_seconds:
                # Expired
                await self.delete(key, agent_id=agent_id)
                return None

        return entry

    async def delete(self, key: str, agent_id: str | None = None) -> None:
        """Delete from blackboard.

        Args:
            key: Key to delete
            agent_id: Agent ID requesting delete (for access control)

        Raises:
            PermissionError: If access policy denies delete
        """
        await self.initialize()

        # Ambient transaction routing: buffer delete and commit atomically on __aexit__.
        if _get_ambient_tx() is not None:
            await self.delete_tx(key, agent_id=agent_id)
            return

        # Access control
        if self.access_policy and not await self.access_policy.can_delete(
            agent_id or "unknown", key, self.scope_id
        ):
            raise PermissionError(f"Access denied for delete: {key}")

        entry = await self.backend.read(key)
        if entry is None:
            return

        # Delete from backend
        await self.backend.delete(key)

        # Emit event
        await self.emit_event(
            BlackboardEvent(
                event_type="delete",
                key=key,
                value=None,
                version=entry.version,
                old_value=entry.value,
                tags=entry.tags,
                metadata=entry.metadata or {},
                agent_id=agent_id,
            )
        )

        logger.debug(f"Deleted from blackboard: {key}")

    async def list_keys(self) -> list[str]:
        """List all keys in blackboard.

        Returns:
            List of keys
        """
        await self.initialize()

        return await self.backend.list_keys()

    async def clear(self) -> None:
        """Clear all entries from blackboard."""
        await self.initialize()

        await self.backend.clear()

        # Emit event
        await self.emit_event(BlackboardEvent(event_type="clear"))

        logger.info("Cleared blackboard")

    async def query(
        self,
        namespace: str | None = None,
        tags: set[str] | None = None,  # TODO: Tags should be a dict with keys and values
        limit: int = 100,
        offset: int = 0,
    ) -> list[BlackboardEntry]:
        """Query blackboard entries using backend-specific efficient implementation.

        Args:
            namespace: Namespace pattern (glob-style, e.g., "agent:*:results")
            tags: Tags to filter by (must match ALL tags)
            limit: Maximum number of entries to return
            offset: Offset for pagination

        Returns:
            List of matching entries
        """
        await self.initialize()

        # Delegate to backend's efficient query implementation
        return await self.backend.query(
            namespace=namespace,
            tags=tags,
            limit=limit,
            offset=offset,
        )

    async def transaction(self) -> BlackboardTransaction: # TODO: Change return type to _AmbientBlackboardTransaction?
        """Start transaction for atomic operations.

        Returns:
            Transaction context manager

        Example:
            async with blackboard.transaction() as txn:
                val = await txn.read("counter")
                await txn.write("counter", val + 1)
        """
        await self.initialize()

        # Return an ambient transaction wrapper so blackboard.read/write/delete automatically
        # route through txn buffers while the context is active.
        return _AmbientBlackboardTransaction(self, BlackboardTransaction(backend=self.backend))

    # ---------------------------------------------------------------------
    # Explicit transactional helpers (require an ambient transaction)
    # ---------------------------------------------------------------------

    async def read_tx(self, key: str, agent_id: str | None = None) -> Any | None:
        """Read within the active ambient transaction.

        Raises:
            RuntimeError: if no ambient transaction is active.
        """
        tx = _get_ambient_tx()
        if tx is None:
            raise RuntimeError(
                "read_tx() requires an active ambient transaction (use `async with blackboard.transaction()`)."
            )

        if self.access_policy and not await self.access_policy.can_read(
            agent_id or "unknown", key, self.scope_id
        ):
            raise PermissionError(f"Access denied for read: {key}")

        entry = await tx.read(key)
        if entry is None:
            return None

        # TTL enforcement matches read()
        if entry.ttl_seconds is not None:
            if time.time() - entry.created_at > entry.ttl_seconds:
                await self.delete_tx(key, agent_id=agent_id)
                return None

        return entry.value

    async def write_tx(
        self,
        key: str,
        value: Any,
        created_by: str | None = None,
        tags: set[str] | None = None,
        ttl_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write within the active ambient transaction.

        This preserves blackboard semantics (version increment, metadata, policies),
        but defers persistence + event emission until commit (transaction exit).
        """
        tx = _get_ambient_tx()
        if tx is None:
            raise RuntimeError(
                "write_tx() requires an active ambient transaction (use `async with blackboard.transaction()`)."
            )

        # Access control + validation mirrors write()
        if self.access_policy and not await self.access_policy.can_write(
            created_by or "unknown", key, value, self.scope_id
        ):
            raise PermissionError(f"Access denied for write: {key}")
        if self.validation_policy:
            await self.validation_policy.validate(key, value, metadata or {})

        await self._check_resource_limits()

        # Ensure we read first so optimistic version token is captured for commit.
        old_entry = await tx.read(key)
        now = time.time()

        entry = BlackboardEntry(
            key=key,
            value=value,
            version=(old_entry.version + 1) if old_entry else 0,
            created_at=old_entry.created_at if old_entry else now,
            updated_at=now,
            created_by=old_entry.created_by if old_entry else created_by,
            updated_by=created_by,
            ttl_seconds=ttl_seconds,
            tags=tags or set(),
            metadata=metadata or {},
        )

        await tx.write(key, entry)

    async def delete_tx(self, key: str, agent_id: str | None = None) -> None:
        """Delete within the active ambient transaction."""
        tx = _get_ambient_tx()
        if tx is None:
            raise RuntimeError(
                "delete_tx() requires an active ambient transaction (use `async with blackboard.transaction()`)."
            )

        if self.access_policy and not await self.access_policy.can_delete(
            agent_id or "unknown", key, self.scope_id
        ):
            raise PermissionError(f"Access denied for delete: {key}")

        # Ensure we read first so old value is available for event emission, and
        # optimistic version token is captured for commit.
        await tx.read(key)
        await tx.delete(key)

    def subscribe(
        self,
        callback: Callable[[BlackboardEvent], Awaitable[None]],
        filter: EventFilter | None = None,
    ) -> None:
        """Subscribe to blackboard events.

        Args:
            callback: Async function to call when event occurs
            filter: Optional filter for events

        Example:
            async def on_complete(event: BlackboardEvent):
                print(f"Agent completed: {event.key}")

            blackboard.subscribe(on_complete, filter=KeyPatternFilter("*:complete"))
        """
        if self.event_bus is None:
            raise RuntimeError(
                "Event system not enabled. Set enable_events=True when creating blackboard."
            )

        self.event_bus.subscribe(callback, filter)

    def unsubscribe(self, callback: Callable[[BlackboardEvent], Awaitable[None]]) -> None:
        """Unsubscribe from blackboard events.

        Args:
            callback: Callback function to remove
        """
        if self.event_bus is None:
            raise RuntimeError("Event system not enabled.")

        self.event_bus.unsubscribe(callback)

    async def read_batch(self, keys: list[str], agent_id: str | None = None) -> dict[str, Any]:
        """Read multiple keys efficiently using backend batching.

        Args:
            keys: List of keys to read
            agent_id: Agent ID requesting read (for access control)

        Returns:
            Dict mapping keys to values (missing keys are omitted)
        """
        await self.initialize()

        # Access control check for all keys
        if self.access_policy:
            for key in keys:
                if not await self.access_policy.can_read(agent_id or "unknown", key, self.scope_id):
                    raise PermissionError(f"Access denied for read: {key}")

        # Use backend's efficient batch read
        entries = await self.backend.read_batch(keys)

        # Extract values and check TTL
        result = {}
        now = time.time()
        for key, entry in entries.items():
            # Check TTL
            if entry.ttl_seconds is not None:
                if now - entry.created_at > entry.ttl_seconds:
                    # Expired - delete it
                    await self.delete(key, agent_id=agent_id)
                    continue
            result[key] = entry.value

        return result

    async def write_batch(
        self,
        entries: dict[str, Any],
        created_by: str | None = None,
        tags: set[str] | None = None,
        ttl_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write multiple entries efficiently using backend batching.

        Args:
            entries: Dict mapping keys to values
            created_by: Agent ID that created these entries
            tags: Tags to apply to all entries
            ttl_seconds: TTL to apply to all entries
            metadata: Metadata to apply to all entries
        """
        await self.initialize()

        # Access control and validation for all entries
        if self.access_policy:
            for key, value in entries.items():
                if not await self.access_policy.can_write(
                    created_by or "unknown", key, value, self.scope_id
                ):
                    raise PermissionError(f"Access denied for write: {key}")

        if self.validation_policy:
            for key, value in entries.items():
                await self.validation_policy.validate(key, value, metadata or {})

        # Check resource limits
        await self._check_resource_limits()

        # Get existing entries for versioning and old values
        existing_entries = await self.backend.read_batch(list(entries.keys()))

        # Create BlackboardEntry objects
        now = time.time()
        blackboard_entries = {}
        old_values = {}

        for key, value in entries.items():
            old_entry = existing_entries.get(key)
            old_values[key] = old_entry.value if old_entry else None

            entry = BlackboardEntry(
                key=key,
                value=value,
                version=(old_entry.version + 1) if old_entry else 0,
                created_at=old_entry.created_at if old_entry else now,
                updated_at=now,
                created_by=old_entry.created_by if old_entry else created_by,
                updated_by=created_by,
                ttl_seconds=ttl_seconds,
                tags=tags or set(),
                metadata=metadata or {},
            )
            blackboard_entries[key] = entry

        # Write all entries using backend's batch write
        await self.backend.write_batch(blackboard_entries)

        # Emit events for all writes
        for key, value in entries.items():
            await self.emit_event(
                BlackboardEvent(
                    event_type="write",
                    key=key,
                    value=value,
                    version=blackboard_entries[key].version,
                    old_value=old_values[key],
                    agent_id=created_by,
                    tags=tags or set(),
                    metadata=metadata or {},
                )
            )

        logger.debug(f"Wrote {len(entries)} entries to blackboard in batch")

    async def get_statistics(self) -> dict[str, Any]:
        """Get blackboard statistics for monitoring.

        Returns:
            Dict with statistics
        """
        await self.initialize()

        # Get backend-specific statistics
        stats = await self.backend.get_statistics()

        # Add event system statistics
        if self.event_bus:
            stats["event_queue_size"] = self.event_bus.event_queue.qsize()
            stats["subscriber_count"] = len(self.event_bus.listeners)
        else:
            stats["event_queue_size"] = 0
            stats["subscriber_count"] = 0

        # Add configuration info
        stats["app_name"] = self.app_name
        stats["scope"] = self.scope.value
        stats["scope_id"] = self.scope_id
        stats["max_entries"] = self.max_entries

        return stats

    async def stop(self) -> None:
        """Stop event bus and cleanup resources."""
        if self.event_bus:
            await self.event_bus.stop()

    async def emit_event(self, event: BlackboardEvent) -> None:
        """Emit event to event bus."""
        if self.event_bus:
            await self.event_bus.emit(event)

    async def _check_resource_limits(self) -> None:
        """Check and enforce resource limits."""
        if self.max_entries is None:
            return

        # Get current entry count
        keys = await self.backend.list_keys()
        entry_count = len(keys)

        if entry_count >= self.max_entries:
            if self.eviction_policy:
                # Evict entries
                num_to_evict = max(1, entry_count // 10)  # Evict 10%

                # Read entries for eviction policy
                entries = {}
                for key in keys:
                    entry = await self.backend.read(key)
                    if entry:
                        entries[key] = entry

                keys_to_evict = await self.eviction_policy.get_eviction_candidates(
                    entries, num_to_evict
                )
                for key in keys_to_evict:
                    await self.delete(key)

                logger.info(f"Evicted {len(keys_to_evict)} entries due to resource limits")
            else:
                raise RuntimeError(f"Blackboard entry limit reached: {self.max_entries}")

    def stream_events_to_queue(
        self,
        event_queue: asyncio.Queue[BlackboardEvent] | None = None,
        filter: EventFilter | None = None,
        pattern: str = "*",
        event_types: set[str] | None = None,
    ) -> asyncio.Queue[BlackboardEvent]:
        """Subscribe to blackboard events, queueing them for plan_step.

        Pass the same event_queue to this method to reuse it across multiple subscriptions.

        Args:
            `event_queue`: Queue to stream events into
            `filter`: Optional event filter
            `pattern`: Key pattern to match events (glob-style)
            `event_types`: Set of event types to filter (e.g., {"write", "delete
        """
        if event_queue is None:
            event_queue = asyncio.Queue()

        async def queue_event(event: BlackboardEvent) -> None:
            try:
                event_queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"Event queue full, dropping event: {event.key}")

        # TODO: We actually need to support the conjunction of both filters here.
        # The pattern usually specifies the key pattern which can be as specific as
        # needed, while event_types filters the type of events which are generic
        # blackboard events (read, write, delete).
        if filter is not None:
            self.subscribe(queue_event, filter=filter)
        elif event_types is not None and pattern is not None:
            combined_filter = CombinationFilter(
                event_types=event_types,
                pattern=pattern
            )
            self.subscribe(queue_event, filter=combined_filter)
        elif pattern:
            self.subscribe(queue_event, filter=KeyPatternFilter(pattern))
        elif event_types is not None:
            self.subscribe(queue_event, filter=EventTypeFilter(event_types))

        self._subscribed_callbacks.append(queue_event)

        return event_queue

    async def stream_events(
        self,
        *,
        filter: EventFilter | None = None,
        pattern: str = "*",
        event_types: set[str] | None = None,
        queue_maxsize: int = 0,
        until: Callable[[], bool] | None = None,
        timeout: float | None = 10.0,
    ) -> AsyncIterator[BlackboardEvent]:
        """Yield blackboard events as an async iterator.

        This packages the common pattern:
        - create an asyncio.Queue
        - subscribe events into it
        - `async for` pull events from the queue forever
        - unsubscribe automatically on cancellation / exit

        This is useful for long-running background tasks:

            async def monitor():
                async for event in blackboard.stream_events(
                    filter=KeyPatternFilter("tenant:...:scope:*"),
                    until=lambda: self._stopped,
                ):
                    ...

        Args:
            filter: Optional event filter
            pattern: Key pattern to match events (glob-style). Used if `filter` is None.
            event_types: Optional set of event types to filter (e.g., {"write"}).
            queue_maxsize: If > 0, bound the internal queue and drop events when full.
            until: Optional callable that returns True to exit the stream.
            timeout: Optional timeout for waiting on events.

        Yields:
            BlackboardEvent objects as they arrive.
        """
        event_queue: asyncio.Queue[BlackboardEvent]
        if queue_maxsize > 0:
            event_queue = asyncio.Queue(maxsize=queue_maxsize)
        else:
            event_queue = asyncio.Queue()

        # Reuse stream_events_to_queue for subscription logic
        self.stream_events_to_queue(
            event_queue=event_queue,
            filter=filter,
            pattern=pattern,
            event_types=event_types,
        )

        if until is None:
            until = lambda: False

        try:
            while not until():
                try:
                    yield await asyncio.wait_for(event_queue.get(), timeout=timeout)

                except asyncio.TimeoutError:
                    continue
        finally:
            # Best-effort cleanup: remove the callback that was added by stream_events_to_queue
            if self._subscribed_callbacks:
                callback = self._subscribed_callbacks.pop()
                try:
                    self.unsubscribe(callback)
                except Exception:
                    pass

    async def stream_events_via_consumer_group(
        self,
        event_queue: asyncio.Queue[BlackboardEvent],
        consumer_group: str,
        consumer_name: str,
    ) -> bool:
        """Set up Redis Streams consumer group subscription.

        Creates the consumer group if it doesn't exist, then starts a
        background task that reads events via XREADGROUP.

        This is a VCM-only code path — ensures each event is delivered to
        exactly one consumer in the group (across VCM replicas).

        Args:
            event_queue: Queue to deliver events to.
            key_pattern: The stream name is derived from the key pattern
                (e.g., ``"scope:*"`` → stream ``"bb:events:{scope}"``)
            consumer_group: Consumer group name.
            consumer_name: This consumer's unique name in the group.
        """
        if self.event_bus is None:
            logger.warning(
                "Cannot set up consumer group: blackboard has no EventBus. "
                "Falling back to standard pub-sub."
            )
            return False

        streamed = await self.event_bus.stream_events_via_consumer_group(
            event_queue=event_queue,
            consumer_group=consumer_group,
            consumer_name=consumer_name,
        )

        if not streamed:
            logger.warning(
                "Cannot set up consumer group: EventBus has no Redis client. "
                "Falling back to standard pub-sub."
            )

        return streamed

