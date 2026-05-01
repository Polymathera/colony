"""``ToolCapability`` — agent-facing surface over the tool registry.

Per master §3.3 the capability layer is "the vocabulary by which agents
specify what they want". Agents bind to a ``ToolCapability`` once;
the capability holds a default ``Preferences``, resolves an adapter
out of a ``ToolRegistry`` at dispatch time, builds a typed
``ToolCall``, invokes the adapter, and returns the typed
``ToolResult``. Agents never see backend-specific arguments.

Two related but distinct concepts to keep separate:

- ``ToolCapability`` (this module) — the design-time vocabulary that
  agents call. One capability key, possibly many adapters.
- ``AgentCapability`` (``agents/base.py``) — colony's existing actor-
  shaped wrapper for ``@action_executor``-decorated methods. An
  ``AgentCapability`` MAY hold one or more ``ToolCapability``
  instances and delegate to them inside its ``@action_executor``
  bodies; the names are similar but the abstractions are different.

A ``ToolCapability`` is intentionally light: a callable that can be
held by reference, passed to multiple agents, and reused across
sessions. It is not itself an ``AgentCapability``.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Mapping
from typing import Any

from .base import (
    CostModel,
    Preferences,
    ToolCall,
    ToolResult,
)
from .registry import NoAdapterAvailable, ToolRegistry


logger = logging.getLogger(__name__)


class ToolCapability:
    """Agent-facing wrapper around a capability + registry + preferences."""

    def __init__(
        self,
        *,
        name: str,
        registry: ToolRegistry,
        preferences: Preferences | None = None,
        description: str = "",
    ) -> None:
        if not name:
            raise ValueError("ToolCapability.name must be a non-empty string.")
        self._name = name
        self._registry = registry
        self._preferences = preferences or Preferences()
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def preferences(self) -> Preferences:
        return self._preferences

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    # ---- Invocation ---------------------------------------------------

    async def __call__(
        self,
        *,
        caller: str = "",
        trace_id: str | None = None,
        preferences: Preferences | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Resolve an adapter, invoke it, return the result.

        ``preferences`` overrides the capability's default for this
        single call (useful when an agent has a more-restrictive
        constraint than the deployment-wide default — e.g.,
        "this run must be deterministic"). Otherwise the capability's
        configured preferences apply.
        """

        prefs = preferences or self._preferences
        adapter = self._registry.resolve(self._name, prefs)
        call = ToolCall(
            call_id=f"call_{uuid.uuid4().hex[:12]}",
            capability=self._name,
            parameters=kwargs,
            caller=caller,
            trace_id=trace_id,
            started_at=time.time(),
        )
        try:
            result = await adapter.invoke(call)
        except Exception as exc:  # noqa: BLE001 - adapters raise typed errors
            logger.exception(
                "ToolCapability(%s): adapter %s raised during invoke",
                self._name, type(adapter).spec.name,
            )
            return ToolResult(
                call_id=call.call_id,
                adapter_name=type(adapter).spec.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        # If the adapter didn't fill in adapter_name / call_id, do it
        # for them — this is a common copy-paste mistake.
        if result.call_id != call.call_id or not result.adapter_name:
            result = result.model_copy(
                update={
                    "call_id": call.call_id,
                    "adapter_name": result.adapter_name or type(adapter).spec.name,
                },
            )
        return result

    # ---- Introspection ------------------------------------------------

    def available_adapters(
        self, preferences: Preferences | None = None,
    ) -> list[str]:
        """Names of every adapter that survives the hard filter.

        Used by ``BuildVsBuyAdvisor`` and by agents that want to surface
        "what would I be using?" in a chat reply before invoking.
        """

        return [
            type(a).spec.name
            for a in self._registry.resolve_all(
                self._name, preferences or self._preferences,
            )
        ]

    def is_available(self, preferences: Preferences | None = None) -> bool:
        try:
            self._registry.resolve(self._name, preferences or self._preferences)
        except NoAdapterAvailable:
            return False
        return True


__all__ = ("ToolCapability",)
