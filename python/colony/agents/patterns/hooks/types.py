"""Core types for the hook system."""

from __future__ import annotations

import uuid
import weakref
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Awaitable

if TYPE_CHECKING:
    from .pointcuts import Pointcut


class HookType(Enum):
    """Type of hook execution."""

    BEFORE = "before"
    """Execute before the method. Can modify args/kwargs."""

    AFTER = "after"
    """Execute after the method. Can modify return value."""

    AROUND = "around"
    """Wrap the method execution. Has full control."""


class ErrorMode(Enum):
    """How to handle errors in hook execution."""

    FAIL_FAST = "fail_fast"
    """First error aborts entire chain (default)."""

    CONTINUE = "continue"
    """Log error, continue to next hook."""

    SUPPRESS = "suppress"
    """Silently ignore errors."""


@dataclass
class HookContext:
    """Context passed to hook handlers.

    Attributes:
        join_point: The method being intercepted (e.g., "MyCapability.process")
        instance: The object whose method is being called
        args: Positional arguments to the method
        kwargs: Keyword arguments to the method
        metadata: Arbitrary data passed between hooks in the chain
    """

    join_point: str
    instance: Any
    args: tuple
    kwargs: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_args(self, args: tuple, kwargs: dict[str, Any]) -> HookContext:
        """Return a new context with modified arguments."""
        return HookContext(
            join_point=self.join_point,
            instance=self.instance,
            args=args,
            kwargs=kwargs,
            metadata=self.metadata,
        )


# Type aliases for hook handlers
BeforeHookHandler = Callable[[HookContext], Awaitable[HookContext]]
AfterHookHandler  = Callable[[HookContext, Any], Awaitable[Any]]
AroundHookHandler = Callable[[HookContext, Callable[[], Awaitable[Any]]], Awaitable[Any]]
HookHandler = BeforeHookHandler | AfterHookHandler | AroundHookHandler


@dataclass
class RegisteredHook:
    """A registered hook with its configuration.

    Attributes:
        hook_id: Unique identifier for this hook
        pointcut: Determines which methods/instances match
        handler: The hook function to execute
        hook_type: BEFORE, AFTER, or AROUND
        priority: Higher values run first (for BEFORE/AFTER) or outermost (for AROUND)
        on_error: How to handle errors during hook execution
        owner_ref: Weak reference to the object that registered this hook
    """

    hook_id: str
    pointcut: Pointcut
    handler: HookHandler
    hook_type: HookType = HookType.AFTER
    priority: int = 0
    on_error: ErrorMode = ErrorMode.FAIL_FAST
    owner_ref: weakref.ref | None = None

    @staticmethod
    def generate_id() -> str:
        """Generate a unique hook ID."""
        return f"hook_{uuid.uuid4().hex[:8]}"

