"""Decorators for the hook system.

- `@hookable`: Mark methods as interceptable (join points)
- `@register_hook`: Declaratively register a method as a hook handler
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, TypeVar, ParamSpec

from .types import HookContext, HookType, ErrorMode, RegisteredHook
from .pointcuts import Pointcut
from .registry import AgentHookRegistry

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


# Attribute name for storing hook registration metadata
_HOOK_REGISTRATION_ATTR = "_hook_registration"


def register_hook(
    pointcut: Pointcut,
    hook_type: HookType = HookType.AFTER,
    priority: int = 0,
    on_error: ErrorMode = ErrorMode.FAIL_FAST,
) -> Callable[[Callable], Callable]:
    """Decorator to declaratively register a method as a hook handler.

    The decorated method will be automatically registered as a hook when
    the owning capability is initialized (via `_auto_register_hooks()`).

    Args:
        pointcut: Determines which methods/instances match
        hook_type: BEFORE, AFTER, or AROUND
        priority: Higher values run first
        on_error: How to handle errors during execution

    Example:
        ```python
        class TokenTrackerCapability(AgentCapability):
            @register_hook(
                pointcut=Pointcut.pattern("*.infer"),
                hook_type=HookType.AFTER,
                priority=100,
            )
            async def track_tokens(self, ctx: HookContext, result: Any) -> Any:
                self.total_tokens += result.usage.total_tokens
                return result

            async def initialize(self):
                # Hooks are auto-registered by AgentCapability base class
                await super().initialize()
        ```

    Note:
        The method signature must match the hook type:
        - BEFORE: `async def handler(self, ctx: HookContext) -> HookContext`
        - AFTER: `async def handler(self, ctx: HookContext, result: T) -> T`
        - AROUND: `async def handler(self, ctx: HookContext, proceed: Callable) -> T`
    """

    def decorator(func: Callable) -> Callable:
        # Store registration metadata on the function
        setattr(
            func,
            _HOOK_REGISTRATION_ATTR,
            {
                "pointcut": pointcut,
                "hook_type": hook_type,
                "priority": priority,
                "on_error": on_error,
            },
        )
        return func

    return decorator


def discover_hook_handlers(obj: Any) -> list[tuple[Callable, dict]]:
    """Discover methods decorated with @register_hook on an object.

    Args:
        obj: Object to scan for hook handlers

    Returns:
        List of (method, registration_info) tuples
    """
    handlers = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(obj, name)
            if callable(attr) and hasattr(attr, _HOOK_REGISTRATION_ATTR):
                reg_info = getattr(attr, _HOOK_REGISTRATION_ATTR)
                handlers.append((attr, reg_info))
        except Exception:
            # Skip attributes that raise on access
            pass
    return handlers


def auto_register_hooks(obj: Any, owner: Any = None) -> list[str]:
    """Auto-register all @register_hook decorated methods on an object.

    Args:
        obj: Object with hook handler methods (typically an AgentCapability)
        owner: Owner for lifecycle management (defaults to obj)

    Returns:
        List of registered hook IDs
    """
    hook_registry = _get_hook_registry(obj)
    if hook_registry is None:
        return []

    owner = owner or obj
    hook_ids = []

    for handler, reg_info in discover_hook_handlers(obj):
        hook_id = hook_registry.register(
            pointcut=reg_info["pointcut"],
            handler=handler,
            hook_type=reg_info["hook_type"],
            priority=reg_info["priority"],
            on_error=reg_info["on_error"],
            owner=owner,
        )
        hook_ids.append(hook_id)
        logger.debug(
            f"Auto-registered hook {hook_id} from {type(obj).__name__}.{handler.__name__}"
        )

    return hook_ids


def hookable(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to mark a method as hookable.

    When a hookable method is called, it checks the owning agent's hook registry
    for matching hooks and executes them in the appropriate order:

    1. BEFORE hooks (highest priority first): Can modify args/kwargs
    2. AROUND hooks (highest priority = outermost): Wrap execution
    3. Original method execution
    4. AFTER hooks (highest priority first): Can modify return value

    The decorator automatically finds the owning agent by looking for `self.agent`
    on the instance. If no agent is found, the method executes without hooks.

    Example:
        ```python
        class MyCapability(AgentCapability):
            @hookable
            async def process(self, data: dict) -> dict:
                return {"processed": data}
        ```

    Note:
        - Only works on async methods (the primary pattern in this codebase)
        - The decorated method must be on an object with `self.agent` or be an Agent itself
    """

    @functools.wraps(func)
    async def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> R:
        # Find the owning agent
        hook_registry = _get_hook_registry(self)

        # If no hook registry, execute directly
        if hook_registry is None:
            return await func(self, *args, **kwargs)

        # Build join point identifier from the defining class, not the
        # runtime type.  func.__qualname__ is "ClassName.method_name" as set
        # at decoration time, so super() calls get their own distinct name.
        join_point = func.__qualname__
        # join_point = f"{type(self).__name__}.{func.__name__}"

        # Get matching hooks
        before_hooks, around_hooks, after_hooks = hook_registry.get_hooks(
            join_point, self
        )

        # If no hooks, execute directly (fast path)
        if not before_hooks and not around_hooks and not after_hooks:
            return await func(self, *args, **kwargs)

        # Build context
        ctx = HookContext(
            join_point=join_point,
            instance=self,
            args=args,
            kwargs=kwargs,
            metadata={},
        )

        # Execute hook chain
        return await _execute_hook_chain(
            ctx=ctx,
            before_hooks=before_hooks,
            around_hooks=around_hooks,
            after_hooks=after_hooks,
            original_func=func,
        )

    # Mark as hookable for introspection
    wrapper._is_hookable = True
    wrapper._original_func = func
    return wrapper


def _get_hook_registry(obj: Any) -> AgentHookRegistry | None:
    """Get the hook registry from an object.

    Looks for:
    1. obj itself if it has _hook_registry
    2. obj.agent (for capabilities, policies, etc.)

    Args:
        obj: The object to get the hook registry from

    Returns:
        The hook registry instance or None
    """
    # Check if obj itself is an Agent (has _hook_registry attribute)
    if hasattr(obj, "_hook_registry"):
        return obj._hook_registry

    # Check for agent attribute (capabilities, policies, dispatchers, etc.)
    agent = getattr(obj, "agent", None)
    if agent is not None and hasattr(agent, "_hook_registry"):
        return agent._hook_registry

    return None


async def _execute_hook_chain(
    ctx: HookContext,
    before_hooks: list[RegisteredHook],
    around_hooks: list[RegisteredHook],
    after_hooks: list[RegisteredHook],
    original_func: Callable,
) -> Any:
    """Execute the hook chain around the original function.

    Order of execution:
    1. BEFORE hooks (modify ctx.args/ctx.kwargs)
    2. AROUND hooks (wrap execution, outermost first)
    3. Original function
    4. AFTER hooks (modify result)
    """
    # 1. Execute BEFORE hooks
    for hook in before_hooks:
        try:
            ctx = await hook.handler(ctx)
        except Exception as e:
            if hook.on_error == ErrorMode.FAIL_FAST:
                raise
            elif hook.on_error == ErrorMode.CONTINUE:
                logger.warning(
                    f"Before hook {hook.hook_id} failed: {e}", exc_info=True
                )
            # SUPPRESS: silently ignore

    # 2. Build the execution chain with AROUND hooks
    async def core_execution() -> Any:
        return await original_func(ctx.instance, *ctx.args, **ctx.kwargs)

    # Wrap with around hooks (outermost = highest priority = first in list)
    execution = core_execution
    for hook in reversed(around_hooks):  # Reverse so highest priority is outermost
        execution = _wrap_with_around_hook(execution, hook, ctx)

    # 3. Execute (with around hooks wrapping)
    try:
        result = await execution()
    except Exception as e:
        # Around hooks may have their own error handling
        raise

    # 4. Execute AFTER hooks
    for hook in after_hooks:
        try:
            result = await hook.handler(ctx, result)
        except Exception as e:
            if hook.on_error == ErrorMode.FAIL_FAST:
                raise
            elif hook.on_error == ErrorMode.CONTINUE:
                logger.warning(
                    f"After hook {hook.hook_id} failed: {e}", exc_info=True
                )
            # SUPPRESS: silently ignore

    return result


def _wrap_with_around_hook(
    inner: Callable[[], Any],
    hook: RegisteredHook,
    ctx: HookContext,
) -> Callable[[], Any]:
    """Wrap an execution function with an around hook."""

    async def wrapped() -> Any:
        try:
            return await hook.handler(ctx, inner)
        except Exception as e:
            if hook.on_error == ErrorMode.FAIL_FAST:
                raise
            elif hook.on_error == ErrorMode.CONTINUE:
                logger.warning(
                    f"Around hook {hook.hook_id} failed: {e}", exc_info=True
                )
                # Fall through to inner execution
                return await inner()
            else:
                # SUPPRESS: silently execute inner
                return await inner()

    return wrapped

