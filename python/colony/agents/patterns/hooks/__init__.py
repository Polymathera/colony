"""Hook system for aspect-oriented programming in agents.

This module provides a hook system for intercepting method calls on agent components
(capabilities, policies, dispatchers, etc.) without modifying the original code.

Key concepts:
- **Hookable methods**: Methods decorated with `@hookable` can be intercepted
- **Pointcuts**: Specify which methods/instances (join points) to hook (pattern, class, instance)
- **Hook types**: BEFORE (modify input), AFTER (modify output), AROUND (wrap execution)
- **Agent-scoped registry**: Hooks are registered per-agent, not globally

Example:
    ```python
    from polymathera.colony.agents.patterns.hooks import hookable, Pointcut, HookType

    class MyCapability(AgentCapability):
        @hookable
        async def process(self, data: dict) -> dict:
            return {"processed": data}

        async def initialize(self):
            # Register hook on all *.process methods in this agent
            self.agent.hooks.register(
                pointcut=Pointcut.pattern("*.process"),
                handler=self._log_process,
                hook_type=HookType.AFTER,
            )

        async def _log_process(self, ctx: HookContext, result: dict) -> dict:
            logger.info(f"Processed: {result}")
            return result
    ```
"""

from .types import (
    HookType,
    ErrorMode,
    HookContext,
    RegisteredHook,
)
from .pointcuts import (
    Pointcut,
    PatternPointcut,
    ClassPointcut,
    InstancePointcut,
    DecoratorPointcut,
    AndPointcut,
    OrPointcut,
    NotPointcut,
)
from .registry import AgentHookRegistry
from .decorator import hookable, register_hook, auto_register_hooks

__all__ = [
    # Types
    "HookType",
    "ErrorMode",
    "HookContext",
    "RegisteredHook",
    # Pointcuts
    "Pointcut",
    "PatternPointcut",
    "ClassPointcut",
    "InstancePointcut",
    "DecoratorPointcut",
    "AndPointcut",
    "OrPointcut",
    "NotPointcut",
    # Registry
    "AgentHookRegistry",
    # Decorators
    "hookable",
    "register_hook",
    "auto_register_hooks",
]

