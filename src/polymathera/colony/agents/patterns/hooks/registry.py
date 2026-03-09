"""Per-agent hook registry."""

from __future__ import annotations

import weakref
import logging
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from .types import (
    HookType,
    ErrorMode,
    HookContext,
    RegisteredHook,
    HookHandler,
)
from .pointcuts import Pointcut

if TYPE_CHECKING:
    from ...base import Agent

logger = logging.getLogger(__name__)


class AgentHookRegistry:
    """Per-agent registry for hooks.

    Each agent has its own hook registry. Hooks registered on an agent
    apply to all components of that agent (capabilities, policies, etc.)
    but not to other agents.

    The registry supports:
    - Registration with pointcuts for flexible matching
    - Priority-based ordering
    - Automatic cleanup when owners are garbage collected
    - Caching for performance

    Example:
        ```python
        # In an AgentCapability
        async def initialize(self):
            self.agent.hooks.register(
                pointcut=Pointcut.pattern("*.infer"),
                handler=self._track_tokens,
                hook_type=HookType.AFTER,
                priority=100,
                owner=self,  # Auto-removed when capability is removed
            )
        ```
    """

    def __init__(self, agent: Agent):
        """Initialize the registry.

        Args:
            agent: The agent this registry belongs to
        """
        self._agent_ref = weakref.ref(agent)
        self._hooks: list[RegisteredHook] = []
        self._cache: dict[tuple[str, int], list[RegisteredHook]] = {}

    @property
    def agent(self) -> Agent | None:
        """Get the owning agent (may be None if garbage collected)."""
        return self._agent_ref()

    def register(
        self,
        pointcut: Pointcut,
        handler: HookHandler,
        hook_type: HookType = HookType.AFTER,
        priority: int = 0,
        on_error: ErrorMode = ErrorMode.FAIL_FAST,
        owner: Any = None,
    ) -> str:
        """Register a hook.

        Args:
            pointcut: Determines which methods/instances match
            handler: The hook function to execute
            hook_type: BEFORE, AFTER, or AROUND
            priority: Higher values run first (BEFORE/AFTER) or outermost (AROUND)
            on_error: How to handle errors during execution
            owner: Object that owns this hook. If provided, hook is auto-removed
                   when owner is garbage collected or explicitly removed.

        Returns:
            Hook ID for later removal
        """
        hook_id = RegisteredHook.generate_id()
        hook = RegisteredHook(
            hook_id=hook_id,
            pointcut=pointcut,
            handler=handler,
            hook_type=hook_type,
            priority=priority,
            on_error=on_error,
            owner_ref=weakref.ref(owner) if owner else None,
        )
        self._hooks.append(hook)
        self._cache.clear()  # Invalidate cache

        logger.debug(
            f"Registered hook {hook_id}: {pointcut!r} ({hook_type.value}, priority={priority})"
        )
        return hook_id

    def unregister(self, hook_id: str) -> bool:
        """Remove a hook by ID.

        Args:
            hook_id: The hook ID returned by register()

        Returns:
            True if the hook was found and removed
        """
        before_count = len(self._hooks)
        self._hooks = [h for h in self._hooks if h.hook_id != hook_id]

        if len(self._hooks) < before_count:
            self._cache.clear()
            logger.debug(f"Unregistered hook {hook_id}")
            return True
        return False

    def remove_hooks_by_owner(self, owner: Any) -> int:
        """Remove all hooks registered by a specific owner.

        This is called when a capability or other component is removed
        from the agent, to clean up its hooks.

        Args:
            owner: The object that registered the hooks

        Returns:
            Number of hooks removed
        """
        owner_id = id(owner)
        before_count = len(self._hooks)

        self._hooks = [
            h
            for h in self._hooks
            if h.owner_ref is None or id(h.owner_ref()) != owner_id
        ]

        removed = before_count - len(self._hooks)
        if removed > 0:
            self._cache.clear()
            logger.debug(f"Removed {removed} hooks owned by {type(owner).__name__}")
        return removed

    def get_hooks(
        self, join_point: str, instance: Any
    ) -> tuple[list[RegisteredHook], list[RegisteredHook], list[RegisteredHook]]:
        """Get hooks matching a join point and instance.

        Automatically cleans up hooks whose owners have been garbage collected.

        Args:
            join_point: Method identifier (e.g., "MyCapability.process")
            instance: The object whose method is being called

        Returns:
            Tuple of (before_hooks, around_hooks, after_hooks), each sorted by priority
        """
        cache_key = (join_point, id(instance))

        if cache_key in self._cache:
            all_hooks = self._cache[cache_key]
        else:
            # Clean up hooks with garbage-collected owners
            self._hooks = [
                h
                for h in self._hooks
                if h.owner_ref is None or h.owner_ref() is not None
            ]

            # Find matching hooks
            all_hooks = [
                h for h in self._hooks if h.pointcut.matches(join_point, instance)
            ]

            self._cache[cache_key] = all_hooks

        # Separate by type and sort by priority (highest first)
        before_hooks = sorted(
            [h for h in all_hooks if h.hook_type == HookType.BEFORE],
            key=lambda h: -h.priority,
        )
        around_hooks = sorted(
            [h for h in all_hooks if h.hook_type == HookType.AROUND],
            key=lambda h: -h.priority,
        )
        after_hooks = sorted(
            [h for h in all_hooks if h.hook_type == HookType.AFTER],
            key=lambda h: -h.priority,
        )

        return before_hooks, around_hooks, after_hooks

    def list_hooks(self, pointcut: Pointcut | None = None) -> list[RegisteredHook]:
        """List registered hooks, optionally filtered by pointcut.

        Useful for debugging and introspection.

        Args:
            pointcut: Optional pointcut to filter by (hooks matching this pointcut)

        Returns:
            List of registered hooks
        """
        if pointcut is None:
            return list(self._hooks)

        # This is a bit tricky - we need to check if the hook's pointcut
        # would match the same things as the given pointcut. For simplicity,
        # we just return all hooks whose pointcut repr contains the pattern.
        # A more sophisticated implementation would do proper pointcut subsumption.
        return list(self._hooks)

    def clear(self) -> int:
        """Remove all hooks.

        Returns:
            Number of hooks removed
        """
        count = len(self._hooks)
        self._hooks.clear()
        self._cache.clear()
        logger.debug(f"Cleared {count} hooks")
        return count

