"""Core types and abstractions for the blackboard system."""
from __future__ import annotations

from abc import ABC, abstractmethod
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field



class BlackboardEntry(BaseModel):
    """Entry with metadata for debugging and audit trails.

    This is stored in the backend and contains all metadata about a value.
    Pydantic BaseModel so it is compatible with RedisIndex (RedisOM).
    """

    key: str
    value: Any = None
    version: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    created_by: str | None = None  # Agent ID
    updated_by: str | None = None  # Agent ID
    ttl_seconds: float | None = None  # Time-to-live
    tags: set[str] = Field(default_factory=set)  # For querying
    metadata: dict[str, Any] = Field(default_factory=dict)  # Extensible

    model_config = {"arbitrary_types_allowed": True}


@dataclass
class BlackboardEvent:
    """Event emitted on blackboard changes."""

    event_type: str  # "write", "delete", "clear"
    key: str | None  # None for clear events
    value: Any | None  # None for delete/clear events
    event_id: str = field(default_factory=lambda: f"blackboard_event_{uuid.uuid4().hex[:8]}")
    version: int = 0
    old_value: Any | None = None  # Previous value (for updates)
    timestamp: float = field(default_factory=time.time)
    agent_id: str | None = None
    tags: set[str] = field(default_factory=set)  # For querying
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Policy Protocols (Customization Points)
# ============================================================================


class AccessPolicy(ABC):
    """Policy for access control."""

    @abstractmethod
    async def can_read(self, agent_id: str, key: str, scope_id: str) -> bool:
        """Check if agent can read key."""
        ...

    @abstractmethod
    async def can_write(self, agent_id: str, key: str, value: Any, scope_id: str) -> bool:
        """Check if agent can write key."""
        ...

    @abstractmethod
    async def can_delete(self, agent_id: str, key: str, scope_id: str) -> bool:
        """Check if agent can delete key."""
        ...


class EvictionPolicy(ABC):
    """Policy for evicting entries when memory is constrained."""

    @abstractmethod
    async def get_eviction_candidates(
        self, entries: list[BlackboardEntry], num_to_evict: int
    ) -> list[str]:
        """Get keys to evict.

        Args:
            entries: List of all entries
            num_to_evict: Number of entries to evict

        Returns:
            List of keys to evict
        """
        ...


class ValidationPolicy(ABC):
    """Policy for validating values before write."""

    @abstractmethod
    async def validate(self, key: str, value: Any, metadata: dict[str, Any]) -> None:
        """Validate value. Raise ValidationError if invalid."""
        ...


# ============================================================================
# Event System
# ============================================================================


class EventFilter(ABC):
    """Filter for blackboard events."""

    @abstractmethod
    def matches(self, event: BlackboardEvent) -> bool:
        """Check if event matches filter."""
        ...


@dataclass
class KeyPatternFilter(EventFilter):
    """Filter events by key pattern (glob-style)."""

    pattern: str

    def matches(self, event: BlackboardEvent) -> bool:
        import fnmatch

        return event.key and fnmatch.fnmatch(event.key, self.pattern)


@dataclass
class EventTypeFilter(EventFilter):
    """Filter events by type."""

    event_types: set[str]

    def matches(self, event: BlackboardEvent) -> bool:
        return event.event_type in self.event_types


@dataclass
class AgentFilter(EventFilter):
    """Filter events by agent ID."""

    agent_ids: set[str]

    def matches(self, event: BlackboardEvent) -> bool:
        return event.agent_id in self.agent_ids



@dataclass
class TagFilter(EventFilter):
    """Filter events by tags.

    Matches events that contain ALL required tags (subset check).
    """

    required_tags: set[str]

    def matches(self, event: BlackboardEvent) -> bool:
        return bool(event.tags) and self.required_tags.issubset(event.tags)


@dataclass
class CombinationFilter(EventFilter):
    """Filter events by key pattern (glob-style) and event type."""

    pattern: str
    event_types: set[str]
    checker: Callable[[BlackboardEvent], bool] | None = None

    def matches(self, event: BlackboardEvent) -> bool:
        import fnmatch

        ret = event.key and fnmatch.fnmatch(event.key, self.pattern) and event.event_type in self.event_types
        if ret and self.checker and not self.checker(event):
            return False
        return ret





# ============================================================================
# Policy Implementations
# ============================================================================


class NoOpAccessPolicy(AccessPolicy):
    """Access policy that allows all operations (no restrictions)."""

    async def can_read(self, agent_id: str, key: str, scope_id: str) -> bool:
        """Allow all reads."""
        return True

    async def can_write(self, agent_id: str, key: str, value: Any, scope_id: str) -> bool:
        """Allow all writes."""
        return True

    async def can_delete(self, agent_id: str, key: str, scope_id: str) -> bool:
        """Allow all deletes."""
        return True


class LRUEvictionPolicy(EvictionPolicy):
    """Least Recently Used eviction policy."""

    async def get_eviction_candidates(
        self, entries: list[BlackboardEntry], num_to_evict: int
    ) -> list[str]:
        """Get least recently used entries.

        Args:
            entries: List of all entries
            num_to_evict: Number of entries to evict

        Returns:
            List of keys to evict (sorted by least recently used)
        """
        # Sort by updated_at ascending (oldest first)
        sorted_entries = sorted(entries, key=lambda e: e.updated_at)
        return [e.key for e in sorted_entries[:num_to_evict]]


class LFUEvictionPolicy(EvictionPolicy):
    """Least Frequently Used eviction policy.

    Note: Requires tracking access counts in metadata.
    """

    async def get_eviction_candidates(
        self, entries: list[BlackboardEntry], num_to_evict: int
    ) -> list[str]:
        """Get least frequently used entries.

        Args:
            entries: List of all entries
            num_to_evict: Number of entries to evict

        Returns:
            List of keys to evict (sorted by least frequently used)
        """
        # Sort by access count in metadata (default to 0 if not tracked)
        sorted_entries = sorted(entries, key=lambda e: e.metadata.get("access_count", 0))
        return [e.key for e in sorted_entries[:num_to_evict]]


class NoOpValidationPolicy(ValidationPolicy):
    """Validation policy that accepts all values (no validation)."""

    async def validate(self, key: str, value: Any, metadata: dict[str, Any]) -> None:
        """Accept all values without validation."""
        pass


class TypeValidationPolicy(ValidationPolicy):
    """Validation policy that enforces type constraints."""

    def __init__(self, type_constraints: dict[str, type | tuple[type, ...]]):
        """Initialize with type constraints.

        Args:
            type_constraints: Map of key patterns to expected types.
                Example: {"config.*": dict, "count.*": int}
        """
        self.type_constraints = type_constraints

    async def validate(self, key: str, value: Any, metadata: dict[str, Any]) -> None:
        """Validate value matches expected type for key pattern.

        Raises:
            ValueError: If value doesn't match expected type
        """
        import fnmatch

        for pattern, expected_type in self.type_constraints.items():
            if fnmatch.fnmatch(key, pattern):
                if not isinstance(value, expected_type):
                    raise ValueError(
                        f"Key '{key}' expects type {expected_type}, got {type(value)}"
                    )

