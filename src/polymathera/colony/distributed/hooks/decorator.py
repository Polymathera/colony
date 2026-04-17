"""Decorators for the hook system.

- `@hookable`: Mark methods as interceptable (join points)
- `@hook_handler`: Declaratively register a method as a hook handler
- `@tracing`: Inject methods to specify domain keys for hook registries
- `install_hook_handlers()`: Auto-register all @hook_handler methods on an object
- `uninstall_hook_handlers()`: Remove all hook handlers owned by an object

Example usage:
```python
from polymathera.colony.distributed.hooks import hook_handler, hookable, tracing

@tracing(subscribe_key=lambda self: self.agent_id)
class MyCapability(AgentCapability):
    @hookable
    async def process(self, data: dict) -> dict:
        return {"processed": data}
```
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, TypeVar, ParamSpec

from .types import HookContext, HookType, ErrorMode, RegisteredHook
from .pointcuts import Pointcut
from .registry import HookRegistry

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


# Attribute name for storing hook registration metadata
_HOOK_REGISTRATION_ATTR = "_hook_registration"


def hook_handler(
    pointcut: Pointcut,
    hook_type: HookType = HookType.AFTER,
    priority: int = 0,
    on_error: ErrorMode = ErrorMode.FAIL_FAST,
) -> Callable[[Callable], Callable]:
    """Decorator to declaratively register a method as a hook handler.

    The decorated method will be automatically registered as a hook when
    the owning capability is initialized (via `_install_hooks()`).

    Args:
        pointcut: Determines which methods/instances match
        hook_type: BEFORE, AFTER, or AROUND
        priority: Higher values run first
        on_error: How to handle errors during execution

    Example:
        ```python
        class TokenTrackerCapability(AgentCapability):
            @hook_handler(
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


def _discover_hook_handlers(obj: Any) -> list[tuple[Callable, dict]]:
    """Discover methods decorated with @hook_handler on an object.

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


def tracing(
    publish_key: Callable[[Any], str] = None,
    subscribe_key: Callable[[Any], str] = None,
) -> Callable[[type], type]:
    """Class decorator to inject get_hookable_*_domain_key methods for hookable publishers and listeners."""
    
    def decorator(cls: type) -> type:
        if publish_key is not None:
            setattr(
                cls,
                "get_hookable_publication_domain_key",
                lambda self: publish_key(self),
            )
        if subscribe_key is not None:
            setattr(
                cls,
                "get_hooks_subscription_domain_key",
                lambda self: subscribe_key(self),
            )
        return cls

    return decorator


def install_hook_handlers(listener: Any) -> list[str]:
    """Install all @hook_handler decorated methods on an object.

    The listener's class needs to be decorated with @tracing(subscribe_key=...) to specify the domain key identifying the domain (i.e., the set of hookable methods and hook handlers
    handling them).

    Args:
        listener: The object containing the hook handlers. It is used for lifecycle
                management. When `listener` is garbage collected or explicitly removed,
                the hook handlers owned by it can be removed by calling `uninstall_hook_handlers(listener)`.

    Returns:
        List of registered hook IDs
    """
    domain_key = _get_hooks_subscription_domain_key(listener)
    hook_registry = get_hook_registry(domain_key)
    if hook_registry is None:
        return []

    hook_ids = []

    for handler, reg_info in _discover_hook_handlers(listener):
        hook_id = hook_registry.register(
            pointcut=reg_info["pointcut"],
            handler=handler,
            hook_type=reg_info["hook_type"],
            priority=reg_info["priority"],
            on_error=reg_info["on_error"],
            owner=listener,
        )
        hook_ids.append(hook_id)
        logger.debug(
            f"Auto-registered hook {hook_id} from {type(listener).__name__}.{handler.__name__}"
        )

    return hook_ids


def uninstall_hook_handlers(listener: Any) -> int:
    """Remove all hook handlers owned by a listener.

    This should be called when a listener (e.g., a capability) is removed from the agent,
    to clean up its hook handlers.

    The listener's class needs to be decorated with @tracing(subscribe_key=...) to specify the domain key identifying the domain (i.e., the set of hookable methods and hook handlers
    handling them).

    Args:
        listener: The object that owns the hook handlers to remove

    Returns:
        The number of hook handlers removed
    """
    domain_key = _get_hooks_subscription_domain_key(listener)
    hook_registry = get_hook_registry(domain_key)
    if hook_registry is None:
        return 0
    return hook_registry.uninstall_hook_handlers(listener)


def _get_hookable_publication_domain_key(obj: Any) -> Any:
    """Get the domain key that contains the hook registry that a hookable method publishes to.

    Args:
        obj: The publisher object to find the domain key for

    Returns:
        The domain key or None if no hook registry is available
    """
    if not hasattr(obj, "get_hookable_publication_domain_key"):
        raise ValueError(
            f"Publisher object of type {type(obj).__name__} does not implement get_hookable_publication_domain_key()"
        )
    return obj.get_hookable_publication_domain_key()


def _get_hooks_subscription_domain_key(obj: Any) -> Any:
    """Get the domain key that contains the hook registry for a hook to subscribe to.

    Args:
        obj: The listener object to find the domain key for

    Returns:
        The domain key or None if no hook registry is available
    """
    if not hasattr(obj, "get_hooks_subscription_domain_key"):
        raise ValueError(
            f"Listener object of type {type(obj).__name__} does not implement get_hooks_subscription_domain_key()"
        )
    return obj.get_hooks_subscription_domain_key()


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
        domain_key = _get_hookable_publication_domain_key(self)
        hook_registry = get_hook_registry(domain_key)

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


_hook_registries: dict[str, HookRegistry] = {}


def get_hook_registry(domain_key: str) -> HookRegistry | None:
    """Get or create the hook registry for a given domain key.

    Args:
        domain_key: The key identifying the domain (i.e., the set of hookable
                    methods and hooks handling them).

    Returns:
        The hook registry instance or None
    """
    if domain_key not in _hook_registries:
        _hook_registries[domain_key] = HookRegistry(domain_key=domain_key)
    return _hook_registries[domain_key]


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

