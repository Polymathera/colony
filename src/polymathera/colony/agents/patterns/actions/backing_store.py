"""BackingStore protocol for external data storage in REPL.

This module defines the `BackingStore` protocol - an interface for external
data storage used by PolicyPythonREPL to store large data that would bloat
the LLM context if stored directly.

Design principles:
- Dependency injection: Backing stores are injected, not hardcoded
- Async-first: All operations are async for non-blocking I/O
- Metadata support: Stores can track provenance and access patterns
- Key generation: Stores generate unique keys for variable storage

Example:
    ```python
    # Create backing store (injected dependency)
    store = BlackboardBackingStore(agent)

    # Use in REPL
    repl = PolicyPythonREPL(
        agent=agent,
        backing_stores={"blackboard": store}
    )

    # LLM planner decides storage type via StorageHint
    action = Action(
        action_type=ActionType.ANALYZE_PAGE,
        result_var="analysis_result",
        storage_hint=StorageHint(
            var_name="analysis_result",
            description="Page analysis with dependency graph",
            storage_type="reference",  # Store in backing store
            backing_store="blackboard",
        )
    )
    ```
"""

from __future__ import annotations

import time
from typing import Any, Literal, Protocol, TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

from ...blackboard.protocol import ActionPolicyProtocol

if TYPE_CHECKING:
    from ...base import Agent


class BackingStore(Protocol):
    """Protocol for external data storage used by REPL.

    Implementations provide persistent storage for large data that would
    bloat the LLM context if stored directly in the REPL namespace.

    Available implementations:
    - BlackboardBackingStore: Uses agent's EnhancedBlackboard
    - S3BackingStore: (Future) Uses S3 for very large artifacts
    - RedisBackingStore: (Future) Uses Redis for fast ephemeral storage
    """

    @property
    def name(self) -> str:
        """Unique name for this backing store (e.g., 'blackboard', 's3').

        Used in StorageHint.backing_store to identify the store.
        """
        ...

    async def store(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a value with optional metadata.

        Args:
            key: Unique key for the value (use generate_key() to create)
            value: Value to store (must be JSON-serializable for most backends)
            metadata: Optional metadata (provenance, description, etc.)
        """
        ...

    async def retrieve(self, key: str) -> Any:
        """Retrieve a value by key.

        Args:
            key: Key of the stored value

        Returns:
            The stored value, or None if not found

        Raises:
            KeyError: If key not found (implementation choice)
        """
        ...

    async def delete(self, key: str) -> bool:
        """Delete a value by key.

        Args:
            key: Key of the stored value

        Returns:
            True if deleted, False if not found
        """
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists.

        Args:
            key: Key to check

        Returns:
            True if exists, False otherwise
        """
        ...

    def generate_key(self, agent_id: str, var_name: str) -> str:
        """Generate a unique key for storing a variable.

        Args:
            agent_id: Agent ID for namespacing
            var_name: Variable name being stored

        Returns:
            Unique key string
        """
        ...


class BlackboardBackingStore:
    """BackingStore implementation using agent's EnhancedBlackboard.

    This is the default backing store for REPL. It uses the agent's
    blackboard for shared storage, which enables:
    - Persistence across agent suspensions
    - Cross-agent data sharing (with proper scoping)
    - Built-in TTL and eviction policies

    Example:
        ```python
        store = BlackboardBackingStore(agent, namespace="repl_data")
        await store.store("my_key", {"large": "data"})
        value = await store.retrieve("my_key")
        ```
    """

    def __init__(
        self,
        agent: "Agent",
        namespace: str = "repl",
    ):
        """Initialize BlackboardBackingStore.

        Args:
            agent: Agent whose blackboard to use
            namespace: Blackboard namespace for REPL data (default: "repl")
        """
        self._agent = agent
        self._namespace = namespace
        self._blackboard = None  # Lazy initialization

    @property
    def name(self) -> str:
        """Return store name."""
        return "blackboard"

    async def _get_blackboard(self):
        """Get or create blackboard instance."""
        if self._blackboard is None:
            from ...scopes import BlackboardScope, get_scope_prefix
            self._blackboard = await self._agent.get_blackboard(
                scope_id=get_scope_prefix(BlackboardScope.AGENT, self._agent, namespace=self._namespace),  # Namespaced by agent ID and namespace
            )
        return self._blackboard

    async def store(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store value in blackboard."""
        bb = await self._get_blackboard()
        await bb.write(
            key=key,
            value=value,
            agent_id=self._agent.agent_id,
            metadata=metadata or {},
        )

    async def retrieve(self, key: str) -> Any:
        """Retrieve value from blackboard."""
        bb = await self._get_blackboard()
        entry = await bb.read(key)
        if entry is None:
            return None
        return entry.value

    async def delete(self, key: str) -> bool:
        """Delete value from blackboard."""
        bb = await self._get_blackboard()
        try:
            await bb.delete(key)
            return True
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in blackboard."""
        bb = await self._get_blackboard()
        entry = await bb.read(key)
        return entry is not None

    def generate_key(self, agent_id: str, var_name: str) -> str:
        """Generate unique key for blackboard storage."""
        return ActionPolicyProtocol.repl_key(agent_id, var_name, time.time_ns())


class StorageHint(BaseModel):
    """LLM-provided hint for how to store an action result.

    The LLM planner explicitly decides storage strategy:
    - storage_type="value": Store directly in REPL namespace (default)
    - storage_type="reference": Store in backing store, keep only metadata

    This gives the LLM full control over context management rather than
    relying on auto-detection based on data size.

    Example:
        ```json
        {
            "action_type": "analyze_page",
            "parameters": {"page_id": "core/auth.py"},
            "result_var": "auth_analysis",
            "storage_hint": {
                "var_name": "auth_analysis",
                "description": "Detailed analysis of auth module with call graph",
                "storage_type": "reference",
                "backing_store": "blackboard"
            }
        }
        ```
    """

    var_name: str = Field(
        description="Variable name in REPL namespace"
    )
    description: str = Field(
        default="",
        description="Human-readable description for LLM context"
    )
    storage_type: Literal["value", "reference"] = Field(
        default="value",
        description="Storage strategy: 'value' for direct storage, 'reference' for backing store"
    )
    backing_store: str | None = Field(
        default=None,
        description="Backing store name (required if storage_type='reference')"
    )

    @field_validator("backing_store")
    @classmethod
    def validate_backing_store(cls, v: str | None, info) -> str | None:
        """Validate backing_store is provided when storage_type is 'reference'."""
        # Access other field values from info.data
        storage_type = info.data.get("storage_type", "value")
        if storage_type == "reference" and not v:
            raise ValueError("backing_store required when storage_type is 'reference'")
        return v
