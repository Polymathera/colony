"""Stream event sources — pluggable feeders for
:class:`~polymathera.colony.agents.patterns.planning.streams.ConsciousnessStream`.

A ``StreamEventSource`` is anything that, once attached to a policy,
arranges to call ``policy.record_stream_entry(kind, payload)`` when it
has something to feed. Sources let new event kinds plug into the
existing per-agent stream substrate without each agent or stream
needing to know about them.

Three sources are pure policy-side reuse of the existing event /
action / tool-output feeds:

- :class:`AccumulatedContextSource` — refactor of the existing event
  feed from ``BaseActionPolicy.events_processed_callback``. Already
  wired today; this class just makes the source explicit so subclasses
  and tests can introspect it.
- :class:`ActionCallSource` — refactor of the existing action feed
  from ``BaseActionPolicy._feed_action_to_streams``. Same pattern.
- :class:`ToolResultSource` — installs a post-dispatch callback on the
  policy's action dispatcher. Every action whose return value is a
  typed :class:`~polymathera.cps.tools.tool_result.ToolResult` (or its
  ``dict_form`` equivalent in non-CPS callers) feeds a
  ``"tool_output"`` entry.

Two further sources surface **cross-process** events via the colony
blackboard. Both are :class:`AgentCapability` subclasses that subscribe
to typed protocol patterns and translate each event into a
``record_stream_entry`` call on the owning policy:

- :class:`VCMPageEventSource` — subscribes to
  :class:`~polymathera.colony.agents.blackboard.VCMPageEventProtocol`
  writes published by ``VirtualContextManager`` reconciler hooks; emits
  ``"vcm_update"`` stream entries.
- :class:`MonorepoCommitEventSource` — subscribes to
  :class:`~polymathera.colony.agents.blackboard.MonorepoCommitProtocol`
  writes published by ``BranchScopedCapabilityBase.fire_post_commit``;
  emits ``"monorepo_commit"`` stream entries.

Sources are constructed in agent setup (typically alongside the
streams they feed) and passed to ``policy.attach_source(source)``;
the policy invokes ``source.attach(policy)`` during ``initialize``.
For the two capability-shaped sources, ``attach`` also adds the source
to ``agent._capabilities`` (so the policy's event-handler discovery
finds the ``@event_handler`` methods) and registers a colony-scoped
blackboard subscription on the policy's event queue.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

from ...base import AgentCapability
from ...blackboard import (
    BlackboardEvent,
    MonorepoCommitProtocol,
    VCMPageEventProtocol,
)
from ..events import event_handler


if TYPE_CHECKING:  # pragma: no cover — type-only
    from ..actions.policies import BaseActionPolicy


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class StreamEventSource(ABC):
    """Anything that feeds typed entries to a policy's consciousness
    streams.

    Subclasses implement :meth:`attach` to arrange for the source to
    call ``policy.record_stream_entry(kind, payload)`` whenever it has
    something to feed. ``attach`` is called once per policy lifecycle
    (during ``BaseActionPolicy.initialize``).
    """

    @abstractmethod
    async def attach(self, policy: "BaseActionPolicy") -> None:
        """Bind the source to a policy.

        Implementations are free to install dispatcher hooks, register
        a capability on the agent + subscribe colony-scoped patterns,
        or spawn background tasks. The contract is only that
        ``policy.record_stream_entry`` gets called on the policy
        whenever this source has a typed entry.
        """
        ...

    async def detach(self, policy: "BaseActionPolicy") -> None:
        """Reverse :meth:`attach`. Default no-op for sources whose
        hooks die with the policy (i.e. don't outlive it). Sources
        that install process-wide callbacks should override this so
        teardown is clean."""
        return None


# ---------------------------------------------------------------------------
# Built-in sources for the pre-existing event + action feeds
# ---------------------------------------------------------------------------


class AccumulatedContextSource(StreamEventSource):
    """Source that re-uses the existing event-handler accumulated-
    context feed.

    The :class:`BaseActionPolicy` already calls
    ``stream.consider_event(accumulated_context)`` directly from
    ``events_processed_callback`` for backward compat. This class
    makes the source explicit (so introspection / configuration /
    testing can refer to it) but installs no new hook — the feeding
    happens in the policy's existing event-handler post-step path.
    """

    async def attach(self, policy: "BaseActionPolicy") -> None:
        # No-op — the policy's existing events_processed_callback
        # already invokes stream.consider_event for every mounted
        # stream. Recorded here as a sentinel so a caller can list
        # which sources a policy has.
        logger.debug(
            "AccumulatedContextSource attached to %s (no-op — uses "
            "policy's built-in event handler feed)",
            getattr(policy.agent, "agent_id", "<no-agent>"),
        )


class ActionCallSource(StreamEventSource):
    """Source that re-uses the existing action-call feed.

    Each policy that finishes dispatching an action calls
    ``self._feed_action_to_streams(call)`` which fans out to mounted
    streams. This class records the source as a sentinel so
    introspection works; no extra hook installed.
    """

    async def attach(self, policy: "BaseActionPolicy") -> None:
        logger.debug(
            "ActionCallSource attached to %s (no-op — uses policy's "
            "_feed_action_to_streams helper)",
            getattr(policy.agent, "agent_id", "<no-agent>"),
        )


# ---------------------------------------------------------------------------
# Tool outputs (typed ToolResult feed)
# ---------------------------------------------------------------------------


class ToolResultSource(StreamEventSource):
    """Source that surfaces typed ``ToolResult`` outputs from every
    action the policy dispatches.

    Installs an after-dispatch hook on the policy's
    :class:`BaseActionPolicy.dispatch`. After each ``dispatch``
    returns, this source inspects the :class:`ActionResult.data` — if
    it has the ``ToolResult``-like shape (a dict with ``payload`` /
    ``units`` / ``provenance`` keys, or a Pydantic model with those
    attributes), it builds a ``"tool_output"`` payload and calls
    ``policy.record_stream_entry``.

    Why duck-type rather than ``isinstance(ToolResult)``: the
    ``ToolResult`` class lives in CPS (``polymathera.cps.tools``).
    Importing it from Colony would invert the dependency direction
    that the rest of the architecture enforces (Colony is L1, CPS is
    L2). Duck-typing on the wire shape keeps this source Colony-pure.
    """

    _SHAPE_KEYS: frozenset[str] = frozenset({"payload", "units", "provenance"})

    def __init__(
        self,
        agent_id_extractor: Callable[["BaseActionPolicy"], str] | None = None,
    ):
        """
        Args:
            agent_id_extractor: Optional callable that produces the
                ``agent_id`` field stamped on each ``tool_output``
                entry's payload. Defaults to reading
                ``policy.agent.agent_id``.
        """
        self._agent_id_extractor = agent_id_extractor

    async def attach(self, policy: "BaseActionPolicy") -> None:
        policy.register_tool_result_source(self)
        logger.debug(
            "ToolResultSource attached to %s",
            getattr(policy.agent, "agent_id", "<no-agent>"),
        )

    def build_payload(
        self,
        action_key: str,
        action_result: Any,
        policy: "BaseActionPolicy",
    ) -> dict[str, Any] | None:
        """Return a ``"tool_output"`` payload dict for ``action_result``
        if it has the ToolResult shape, else ``None`` (skip)."""
        if action_result is None:
            return None
        data = getattr(action_result, "data", None)
        if data is None:
            return None
        tool_result_dict = self._coerce_to_tool_result_dict(data)
        if tool_result_dict is None:
            return None
        agent_id = (
            self._agent_id_extractor(policy)
            if self._agent_id_extractor is not None
            else getattr(policy.agent, "agent_id", "")
        )
        return {
            "action_key": action_key,
            "tool_result": tool_result_dict,
            "success": bool(getattr(action_result, "success", False)),
            "agent_id": agent_id,
        }

    def _coerce_to_tool_result_dict(self, value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            if self._SHAPE_KEYS.issubset(value.keys()):
                return value
            return None
        dumper = getattr(value, "model_dump", None)
        if callable(dumper):
            try:
                dumped = dumper(mode="json")
            except TypeError:
                dumped = dumper()
            if isinstance(dumped, dict) and self._SHAPE_KEYS.issubset(dumped.keys()):
                return dumped
            return None
        if all(hasattr(value, k) for k in self._SHAPE_KEYS):
            try:
                return {k: getattr(value, k) for k in self._SHAPE_KEYS}
            except Exception:  # noqa: BLE001
                return None
        return None


# ---------------------------------------------------------------------------
# Capability-shaped cross-process sources
#
# Both subclasses below sit on the AgentCapability + StreamEventSource
# diamond:
#
# - As AgentCapabilities, they own an @event_handler method that the
#   policy's event-dispatch loop calls (the policy walks
#   agent.get_capabilities() looking for @event_handler methods).
# - As StreamEventSources, they expose ``attach(policy)`` so they wire
#   into the same per-policy ``_stream_sources`` list the helper
#   factories (``cps_basic_stream``, ``colony_basic_stream``) feed.
#
# attach() binds the capability to its agent + subscribes the
# colony-scoped pattern onto the policy's event queue.  When a
# matching blackboard write fires, the event-handler method
# translates the payload into a ``record_stream_entry`` call on the
# policy.
#
# The colony-scoped subscription is critical:  ``VirtualContextManager``
# and ``BranchScopedCapabilityBase`` publish their events to the
# colony scope so every agent in the colony — across processes /
# replicas — can subscribe.  These capabilities OVERRIDE
# ``stream_events_to_queue`` to subscribe on a colony-scoped
# blackboard rather than the agent's own scope.
# ---------------------------------------------------------------------------


class ColonyScopedEventSource(AgentCapability, StreamEventSource):
    """Common scaffolding for the cross-process consciousness-stream sources.

    Subclasses live in Colony (``VCMPageEventSource``,
    ``MonorepoCommitEventSource``) and downstream packages (e.g. CPS's
    ``BudgetStateEventSource``) — this is a supported extension point,
    hence the public name + ``__all__`` export.

    A subclass sets :attr:`_PATTERN` to a colony-scoped
    ``BlackboardProtocol.event_pattern()`` and decorates exactly one
    ``@event_handler`` method that translates each matching blackboard
    write into a ``record_stream_entry`` call on the owning policy.
    """

    _PATTERN: str = "*"  # subclass overrides via Protocol.event_pattern()

    _PENDING_ATTACH_SCOPE: str = "__pending_attach__"

    def __init__(self) -> None:
        # AgentCapability is fully initialized with a placeholder
        # scope_id so accessing properties (capability_key,
        # input_patterns, …) before :meth:`attach` doesn't
        # AttributeError. ``attach`` rebinds the agent + scope.
        AgentCapability.__init__(
            self,
            agent=None,
            scope_id=self._PENDING_ATTACH_SCOPE,
        )
        self._policy: "BaseActionPolicy | None" = None

    async def attach(self, policy: "BaseActionPolicy") -> None:
        # Rebind the capability to the policy's agent now that one
        # is available. input_patterns auto-infers from the
        # subclass's @event_handler decorator.
        from ...scopes import ScopeUtils
        self._agent = policy.agent
        self.scope_id = ScopeUtils.get_agent_level_scope(policy.agent)
        self._policy = policy

        # Register on the agent so the policy's event-handler discovery
        # (EventDrivenActionPolicy._get_event_handlers) walks
        # agent.get_capabilities() and finds our @event_handler method.
        # events_only=True keeps the source out of the planner's action
        # surface — these capabilities expose no actions.
        policy.agent.add_capability(self, events_only=True)

        # Re-mark this provider as subscribed so a later
        # ``EventDrivenActionPolicy.initialize`` re-run doesn't
        # double-subscribe.  Also manually invoke the queue subscription
        # since the policy's initialize already ran for capabilities
        # added at agent-construction time.
        subscribed = getattr(policy, "_subscribed_providers", None)
        if subscribed is not None and id(self) not in subscribed:
            subscribed.add(id(self))
            await self.stream_events_to_queue(
                policy.get_event_queue(),
                high_priority_queue=getattr(
                    policy, "_high_priority_event_queue", None,
                ),
            )
        logger.debug(
            "%s attached to %s (colony-scoped subscription on %s)",
            type(self).__name__,
            getattr(policy.agent, "agent_id", "<no-agent>"),
            self._PATTERN,
        )

    async def detach(self, policy: "BaseActionPolicy") -> None:
        # The colony blackboard is owned by the agent manager's
        # per-scope pool (see _get_colony_blackboard) — we must not
        # stop it here; other capabilities on the same scope share it.
        # Just drop our reference + unsubscribe via the agent's
        # capability registry teardown.
        self._colony_blackboard = None
        self._policy = None

    # These sources hold no durable per-agent state — all event state
    # lives on the colony blackboard. Suspension is a no-op.
    async def serialize_suspension_state(self, state: Any) -> Any:
        return state

    async def deserialize_suspension_state(self, state: Any) -> None:
        return None

    async def stream_events_to_queue(
        self,
        event_queue: asyncio.Queue[BlackboardEvent],
        *,
        high_priority_queue: asyncio.Queue[BlackboardEvent] | None = None,
    ) -> None:
        """Override the default: subscribe on the COLONY-scoped
        blackboard rather than the agent's own scope.

        The producer (``VirtualContextManager._publish_page_event`` /
        ``BranchScopedCapabilityBase.fire_post_commit``) writes to the
        colony scope; subscribing on the agent's local scope would
        silently see nothing.
        """
        bb = await self._get_colony_blackboard()
        bb.stream_events_to_queue(
            event_queue,
            pattern=self._PATTERN,
            event_types={"write"},
        )


class VCMPageEventSource(ColonyScopedEventSource):
    """Surfaces VCM page-graph mutations (page added / evicted) to
    the policy's streams as ``vcm_update`` entries.

    Subscribes to :class:`VCMPageEventProtocol` writes on the
    colony-scoped blackboard. Producers are
    :meth:`VirtualContextManager._publish_page_event` (called from the
    ``_on_page_loaded`` / ``_on_page_evicted`` reconciler hooks).
    Cross-process: any VCM replica's reconciler write reaches every
    subscriber regardless of which process the agent runs in.
    """

    _PATTERN: str = VCMPageEventProtocol.event_pattern()

    def __init__(
        self,
        scope_prefix: str | None = None,
        page_prefix: str | None = None,
    ) -> None:
        """
        Args:
            scope_prefix: Reserved for future use — VCM page-events
                don't currently carry a scope_id, so this filter is
                a no-op at the source level. Keep ``None`` unless a
                later VCM revision starts including it.
            page_prefix: Only surface mutations whose ``page_id``
                starts with this prefix. ``None`` = no narrowing.
        """
        super().__init__()
        self._scope_prefix = scope_prefix
        self._page_prefix = page_prefix

    @event_handler(pattern=VCMPageEventProtocol.event_pattern())
    async def _on_vcm_page_event(
        self,
        event: BlackboardEvent,
        repl: Any,
    ) -> None:
        """Translate a colony-scoped VCMPageEventProtocol write into a
        ``vcm_update`` stream entry on the owning policy."""
        if self._policy is None:
            return None
        payload = self._build_payload(event.value or {})
        if payload is None:
            return None
        self._policy.record_stream_entry("vcm_update", payload)
        return None

    def _build_payload(self, value: dict[str, Any]) -> dict[str, Any] | None:
        page_id = value.get("page_id", "") or ""
        if (
            self._page_prefix is not None
            and not str(page_id).startswith(self._page_prefix)
        ):
            return None
        return {
            "kind": str(value.get("kind", "")),
            "page_id": str(page_id),
            "scope_id": str(value.get("scope_id", "") or ""),
            "page_source": str(value.get("deployment_name") or ""),
        }


class MonorepoCommitEventSource(ColonyScopedEventSource):
    """Surfaces design-monorepo commits (every successful
    ``BranchScopedCapabilityBase.fire_post_commit`` call) to the
    policy's streams as ``monorepo_commit`` entries.

    Subscribes to :class:`MonorepoCommitProtocol` writes on the
    colony-scoped blackboard. Producers are
    :meth:`BranchScopedCapabilityBase.fire_post_commit` called by
    every tier-2 ``checkpoint_*_to_repo`` action after
    :meth:`DesignMonorepoClient.commit_with_identity` returns.
    Cross-process: an ``IntegrationAgent`` working on
    ``design/budgets/`` sees commits from a peer agent working on
    ``design/beliefs/`` of the same branch even though they live in
    different Ray actors (master plan §5.2).
    """

    _PATTERN: str = MonorepoCommitProtocol.event_pattern()

    def __init__(
        self,
        branch: str | None = None,
        capability_fqn_prefix: str | None = None,
    ) -> None:
        """
        Args:
            branch: Only surface commits on this branch. ``None`` =
                all branches (stream filter still applies).
            capability_fqn_prefix: Only surface commits whose
                originating capability's FQN starts with this prefix
                (e.g. ``"polymathera.cps.experimentation"``). ``None`` =
                no narrowing.
        """
        super().__init__()
        self._branch = branch
        self._fqn_prefix = capability_fqn_prefix

    @event_handler(pattern=MonorepoCommitProtocol.event_pattern())
    async def _on_monorepo_commit(
        self,
        event: BlackboardEvent,
        repl: Any,
    ) -> None:
        """Translate a colony-scoped MonorepoCommitProtocol write into
        a ``monorepo_commit`` stream entry on the owning policy."""
        if self._policy is None:
            return None
        payload = self._filter_payload(event.value or {})
        if payload is None:
            return None
        self._policy.record_stream_entry("monorepo_commit", dict(payload))
        return None

    def _filter_payload(
        self, value: dict[str, Any],
    ) -> dict[str, Any] | None:
        if (
            self._branch is not None
            and value.get("branch") != self._branch
        ):
            return None
        if (
            self._fqn_prefix is not None
            and not str(value.get("capability_fqn", "")).startswith(
                self._fqn_prefix,
            )
        ):
            return None
        return value


__all__ = (
    "AccumulatedContextSource",
    "ActionCallSource",
    "ColonyScopedEventSource",
    "MonorepoCommitEventSource",
    "StreamEventSource",
    "ToolResultSource",
    "VCMPageEventSource",
)
