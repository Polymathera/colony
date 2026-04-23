"""Blackboard communication protocols.

A protocol defines the key format and scope level for a specific type of
structured communication on the blackboard. This is NOT limited to
request/result — it covers any pattern where writers and readers must
agree on key format:

- **Request/result**: Bidirectional, correlated by ``request_id``.
- **Streaming events**: Unidirectional stream correlated by ``request_id``.
- **State publication**: Unidirectional, no ``request_id`` (game state, working set).
- **Signal/notification**: Fire-and-forget (lifecycle events).
- **Command**: One-way instruction, no result expected.

Protocols are built on top of :mod:`~polymathera.colony.agents.scopes` —
they use ``ScopeUtils`` to compute scope IDs and ``format_key``/``pattern_key``
to construct keys within those scopes. **All keys are scope-relative**
(never include the ``scope_id`` prefix, since the blackboard partition
already provides isolation).

Each protocol also provides ``parse_*`` methods that extract structured
fields from keys, so callers never need raw ``str.split(":")`` parsing.

Usage::

    from polymathera.colony.agents.blackboard.protocol import AgentRunProtocol

    # Writer
    key = AgentRunProtocol.request_key(request_id="req_abc123")
    await blackboard.write(key, payload)

    # Subscriber
    pattern = AgentRunProtocol.request_pattern()
    blackboard.stream_events_to_queue(queue, pattern=pattern)

    # Parser (in event handler)
    request_id = AgentRunProtocol.parse_request_key(event.key)

    # AgentHandle.run() uses the protocol and namespace
    result = await handle.run(input_data, protocol=AgentRunProtocol, namespace="compliance")
"""

from __future__ import annotations

import re
from typing import ClassVar

from ..scopes import BlackboardScope, ScopeUtils


# ---------------------------------------------------------------------------
# Key validation
# ---------------------------------------------------------------------------

# Keys must not contain wildcards — those belong in patterns only.
_WILDCARD_CHARS = re.compile(r"[*?\[\]]")


class KeyValidationError(ValueError):
    """Raised when a blackboard key violates protocol constraints."""
    pass


def validate_key(key: str, scope_id: str = "") -> None:
    """Validate that a key is scope-relative and well-formed.

    Called by ``EnhancedBlackboard.write()``, ``read()``, and ``delete()``
    with the blackboard's own ``scope_id`` to catch scope leaks.

    Rules:
    - Must not be empty.
    - Must not start or end with ``:``.
    - Must not contain glob wildcard characters.
    - Must not contain the blackboard's ``scope_id`` (the key leaked
      the partition identifier).
    - Must not start with the global scope prefix (``polymathera:``).

    Args:
        key: The key to validate.
        scope_id: The blackboard's scope_id. If provided, validates that
            the key does not contain this scope_id as a substring.

    Raises:
        KeyValidationError: If the key is invalid.
    """
    if not key:
        raise KeyValidationError("Blackboard key must not be empty")

    if key.startswith(":") or key.endswith(":"):
        raise KeyValidationError(
            f"Blackboard key must not start or end with ':': {key!r}"
        )

    if _WILDCARD_CHARS.search(key):
        raise KeyValidationError(
            f"Blackboard key must not contain wildcard characters "
            f"(*, ?, [, ]): {key!r}. Use patterns for subscriptions."
        )

    # Check for scope_id leak — the key should never start with the
    # blackboard's own scope_id since that's the partition, not the key.
    if scope_id and key.startswith(f"{scope_id}:"):
        raise KeyValidationError(
            f"Blackboard key must be scope-relative — it must not contain "
            f"the blackboard's scope_id. Key {key!r} contains "
            f"scope_id {scope_id!r}. The scope_id is already set on "
            f"the blackboard partition; do not include it in the key."
        )

    # Check for global scope prefix leak
    global_prefix = f"{ScopeUtils.get_global_scope()}:"
    if key.startswith(global_prefix):
        raise KeyValidationError(
            f"Blackboard key must be scope-relative (no scope_id prefix). "
            f"Key {key!r} starts with {global_prefix!r}."
        )


def validate_pattern(pattern: str, scope_id: str = "") -> None:
    """Validate that a subscription pattern is scope-relative.

    Called by ``EnhancedBlackboard.stream_events_to_queue()`` with the
    blackboard's own ``scope_id``.

    Args:
        pattern: The glob pattern to validate.
        scope_id: The blackboard's scope_id. If provided, validates that
            the pattern does not contain this scope_id.

    Raises:
        KeyValidationError: If the pattern is invalid.
    """
    if not pattern:
        raise KeyValidationError("Pattern must not be empty")

    # Check for scope_id leak
    if scope_id and pattern.startswith(f"{scope_id}:"):
        raise KeyValidationError(
            f"Event pattern must be scope-relative — it must not contain "
            f"the blackboard's scope_id. Pattern {pattern!r} contains "
            f"scope_id {scope_id!r}."
        )

    # Check for global scope prefix leak
    global_prefix = f"{ScopeUtils.get_global_scope()}:"
    if pattern.startswith(global_prefix):
        raise KeyValidationError(
            f"Event pattern must be scope-relative (no scope_id prefix). "
            f"Pattern {pattern!r} starts with {global_prefix!r}."
        )


# ---------------------------------------------------------------------------
# Base protocol
# ---------------------------------------------------------------------------

class BlackboardProtocol:
    """Base class for blackboard communication protocols.

    Subclass this per interaction type — not per capability. Multiple
    capabilities can share the same protocol. One capability can support
    multiple protocols for different actions.

    Each protocol subclass provides three kinds of static methods:

    - **Key construction** (e.g., ``request_key``, ``result_key``):
      Produce the exact key string to pass to ``blackboard.write()``.
    - **Pattern construction** (e.g., ``request_pattern``, ``result_pattern``):
      Produce glob patterns for ``stream_events_to_queue()`` and ``@event_handler``.
    - **Key parsing** (e.g., ``parse_request_key``, ``parse_result_key``):
      Extract structured fields from a key, so callers never do raw
      ``str.split(":")`` parsing.

    All keys produced by protocols are scope-relative. Protocols that
    operate at colony-level scope include ``agent_id`` or other
    disambiguators in keys because multiple agents share the partition.
    Protocols at agent-level scope omit ``agent_id`` since the scope
    IS the agent's mailbox.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT
    """The scope level this protocol operates at."""


# ---------------------------------------------------------------------------
# Concrete protocols
# ---------------------------------------------------------------------------

class AgentRunProtocol(BlackboardProtocol):
    """Protocol for ``AgentHandle.run()`` <-> child agent communication.

    Key formats:

    - ``request:run:{request_id}``

    Example::

        key = AgentRunProtocol.request_key("req_abc")
        # -> "request:run:req_abc"

        class ComplianceCapability(AgentCapability):
            input_patterns = [AgentRunProtocol.request_pattern()]
            # -> ["request:run:*"]

            @event_handler(pattern=AgentRunProtocol.request_pattern())
            async def handle_request(self, event, repl):
                request_id = AgentRunProtocol.parse_request_key(event.key)
                ...
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    # --- Key construction ---

    @staticmethod
    def request_key(request_id: str) -> str:
        """Key for a request.

        Args:
            request_id: Unique request identifier.
        """
        return f"request:run:{request_id}"

    @staticmethod
    def result_key(request_id: str) -> str:
        """Key for a result."""
        return f"result:run:{request_id}"

    @staticmethod
    def event_key(request_id: str, event_name: str) -> str:
        """Key for a streaming event."""
        return f"event:run:{request_id}:{event_name}"

    # --- Pattern construction ---

    @staticmethod
    def request_pattern() -> str:
        """Pattern matching run requests.
        """
        return "request:run:*"

    @staticmethod
    def result_pattern() -> str:
        """Pattern matching run results."""
        return "result:run:*"

    @staticmethod
    def event_pattern(request_id: str) -> str:
        """Pattern matching streaming events for a specific request."""
        return f"event:run:{request_id}:*"

    # --- Key parsing ---

    @staticmethod
    def parse_request_key(key: str) -> str:
        """Extract request_id from a request key.

        Args:
            key: Key like ``"request:run:req_abc"``

        Returns:
            The request_id.
        """
        prefix = "request:run:"
        if not key.startswith(prefix):
            raise ValueError(f"Not an AgentRunProtocol request key: {key!r}")
        return key[len(prefix):]

    @staticmethod
    def parse_result_key(key: str) -> str:
        """Extract request_id from a result key."""
        prefix = "result:run:"
        if not key.startswith(prefix):
            raise ValueError(f"Not an AgentRunProtocol result key: {key!r}")
        return key[len(prefix):]

    @staticmethod
    def parse_event_key(key: str) -> tuple[str, str]:
        """Extract (request_id, event_name) from an event key."""
        prefix = "event:run:"
        if not key.startswith(prefix):
            raise ValueError(f"Not an AgentRunProtocol event key: {key!r}")
        rest = key[len(prefix):]
        parts = rest.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed AgentRunProtocol event key (expected request_id:event_name): {key!r}")
        return parts[0], parts[1]


class WorkAssignmentProtocol(BlackboardProtocol):
    """Protocol for coordinator <-> worker communication via ``AgentPoolCapability``.

    Operates at colony-level scope. ``agent_id`` IS needed in keys because
    multiple workers share the scope.

    Key types:

    - ``agent_id:{id}:work_assignment:{request_id}`` — coordinator assigns work
    - ``agent_id:{id}:result_type:{partial|final}`` — worker writes result
    - ``agent_id:{id}:broadcast:True`` — coordinator broadcasts to workers

    Example::

        # Coordinator assigns work
        key = WorkAssignmentProtocol.assignment_key(agent_id=worker_id, request_id=req_id)
        await colony_blackboard.write(key, work_unit)

        # Worker writes result
        key = WorkAssignmentProtocol.result_key(agent_id=self.agent.agent_id, result_type="final")
        await colony_blackboard.write(key, result)
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    # --- Key construction ---

    @staticmethod
    def assignment_key(agent_id: str, request_id: str) -> str:
        """Key for a work assignment targeting a specific worker."""
        return ScopeUtils.format_key(agent_id=agent_id, work_assignment=request_id)

    @staticmethod
    def result_key(agent_id: str, result_type: str = "final") -> str:
        """Key for a worker's result."""
        return ScopeUtils.format_key(agent_id=agent_id, result_type=result_type)

    @staticmethod
    def broadcast_key(agent_id: str) -> str:
        """Key for a broadcast to a specific worker."""
        return ScopeUtils.format_key(agent_id=agent_id, broadcast=True)

    # --- Pattern construction ---

    @staticmethod
    def assignment_pattern(agent_id: str | None = None) -> str:
        """Pattern matching work assignments (optionally for a specific worker)."""
        return ScopeUtils.pattern_key(agent_id=agent_id, work_assignment=None)

    @staticmethod
    def result_pattern(agent_id: str | None = None) -> str:
        """Pattern matching worker results (optionally from a specific worker)."""
        return ScopeUtils.pattern_key(agent_id=agent_id, result_type=None)

    @staticmethod
    def broadcast_pattern(agent_id: str | None = None) -> str:
        """Pattern matching broadcasts."""
        return ScopeUtils.pattern_key(agent_id=agent_id, broadcast=None)

    # --- Key parsing ---

    @staticmethod
    def parse_assignment_key(key: str) -> dict[str, str]:
        """Extract fields from an assignment key.

        Returns:
            Dict with ``agent_id`` and ``work_assignment`` fields.
        """
        return ScopeUtils.parse_key("", key)

    @staticmethod
    def parse_result_key(key: str) -> dict[str, str]:
        """Extract fields from a result key.

        Returns:
            Dict with ``agent_id`` and ``result_type`` fields.
        """
        return ScopeUtils.parse_key("", key)


class LifecycleSignalProtocol(BlackboardProtocol):
    """Protocol for agent lifecycle signals on colony control plane.

    Colony-scoped. ``agent_id`` in key prevents overwrites when multiple
    agents create/terminate concurrently (the blackboard is a KV store,
    not an append-only log).

    Used by ``MemoryLifecycleHooks`` (writer) and ``AgentMemoryRecycler``/
    ``CollectiveMemoryInitializer`` (readers) on
    ``MemoryScope.colony_control_plane("lifecycle")``.

    Key types:

    - ``agent_id:{id}:scope:agent_created`` — agent creation signal
    - ``agent_id:{id}:scope:agent_terminated`` — agent termination signal
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    # --- Key construction ---

    @staticmethod
    def created_key(agent_id: str) -> str:
        """Key for an agent creation signal."""
        return ScopeUtils.format_key(scope="agent_created", agent_id=agent_id)

    @staticmethod
    def terminated_key(agent_id: str) -> str:
        """Key for an agent termination signal."""
        return ScopeUtils.format_key(scope="agent_terminated", agent_id=agent_id)

    # --- Pattern construction ---

    @staticmethod
    def created_pattern() -> str:
        """Pattern matching all agent creation signals."""
        return ScopeUtils.pattern_key(scope="agent_created", agent_id=None)

    @staticmethod
    def terminated_pattern() -> str:
        """Pattern matching all agent termination signals."""
        return ScopeUtils.pattern_key(scope="agent_terminated", agent_id=None)

    # --- Key parsing ---

    @staticmethod
    def parse_key(key: str) -> dict[str, str]:
        """Extract fields from a lifecycle signal key.

        Returns:
            Dict with ``agent_id`` and ``scope`` (event type) fields.
        """
        return ScopeUtils.parse_key("", key)


class GameStateProtocol(BlackboardProtocol):
    """Protocol for game state publication in multi-agent games.

    Operates at colony scope (with game_id embedded in the scope_id, not the key).
    A single canonical game state key — all participants race to update it
    via optimistic concurrency control.

    Key types:

    - ``state:{game_id}`` — the canonical game state
    - ``result:game_complete`` — the game result after completion
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    # --- Key construction ---

    @staticmethod
    def state_key(game_id: str) -> str:
        """Key for the canonical game state."""
        return ScopeUtils.format_key(state=game_id)

    @staticmethod
    def result_key() -> str:
        """Key for the game result."""
        return ScopeUtils.format_key(result="game_complete")

    # --- Pattern construction ---

    @staticmethod
    def state_pattern() -> str:
        """Pattern matching game state updates."""
        return ScopeUtils.pattern_key(state=None)

    # --- Key parsing ---

    @staticmethod
    def parse_state_key(key: str) -> str:
        """Extract game_id from a state key.

        Returns:
            The game_id.
        """
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("state", "")


class GameInvitationProtocol(BlackboardProtocol):
    """Protocol for game invitation events at colony scope.

    Written by a coordinator to invite agents into a game.
    ``DynamicGameCapability`` listens for these events and
    auto-joins when the agent is listed as a participant.

    Key types:

    - ``game_invitation:{game_id}`` — invitation for a specific game
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    # --- Key construction ---

    @staticmethod
    def invitation_key(game_id: str) -> str:
        """Key for a game invitation."""
        return ScopeUtils.format_key(game_invitation=game_id)

    # --- Pattern construction ---

    @staticmethod
    def invitation_pattern() -> str:
        """Pattern matching all game invitation events."""
        return ScopeUtils.pattern_key(game_invitation=None)

    # --- Key parsing ---

    @staticmethod
    def parse_invitation_key(key: str) -> str:
        """Extract game_id from an invitation key."""
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("game_invitation", "")


class CritiqueProtocol(BlackboardProtocol):
    """Protocol for critique request/response exchange.

    Requester/responder IDs in keys for disambiguation.

    Key types (note: ``ScopeUtils.format_key`` sorts kwargs alphabetically):

    - Peer request: ``critique_request_from_peer:True:requester_id:{id}``
    - Parent-to-child: ``child_id:{id}:critique_request_from_parent:True``
    - Child-to-parent: ``critique_request_from_child:True:parent_id:{id}``
    - Response: ``critique_response_from:{responder}:requester_id:{requester}``

    Note: Because ``child_id`` sorts before ``critique_request_from_parent``,
    the parent-to-child key starts with ``child_id:``, NOT ``critique_request_*``.
    Use per-direction patterns, not a single glob.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    # --- Key construction ---

    @staticmethod
    def peer_request_key(requester_id: str) -> str:
        return ScopeUtils.format_key(requester_id=requester_id, critique_request_from_peer=True)

    @staticmethod
    def parent_to_child_request_key(child_id: str) -> str:
        return ScopeUtils.format_key(child_id=child_id, critique_request_from_parent=True)

    @staticmethod
    def child_to_parent_request_key(parent_id: str) -> str:
        return ScopeUtils.format_key(parent_id=parent_id, critique_request_from_child=True)

    @staticmethod
    def response_key(requester_id: str, responder_id: str) -> str:
        return ScopeUtils.format_key(requester_id=requester_id, critique_response_from=responder_id)

    # --- Pattern construction ---

    @staticmethod
    def peer_request_pattern() -> str:
        """Pattern matching peer critique requests."""
        return ScopeUtils.pattern_key(requester_id=None, critique_request_from_peer=True)

    @staticmethod
    def parent_to_child_request_pattern() -> str:
        """Pattern matching parent-to-child critique requests."""
        return ScopeUtils.pattern_key(child_id=None, critique_request_from_parent=True)

    @staticmethod
    def child_to_parent_request_pattern() -> str:
        """Pattern matching child-to-parent critique requests."""
        return ScopeUtils.pattern_key(parent_id=None, critique_request_from_child=True)

    @staticmethod
    def all_request_patterns() -> list[str]:
        """All three request patterns (peer, parent-to-child, child-to-parent).

        Use this when subscribing to all critique requests regardless of direction.
        """
        return [
            CritiqueProtocol.peer_request_pattern(),
            CritiqueProtocol.parent_to_child_request_pattern(),
            CritiqueProtocol.child_to_parent_request_pattern(),
        ]

    @staticmethod
    def response_pattern(requester_id: str | None = None) -> str:
        """Pattern matching critique responses (optionally for a specific requester)."""
        return ScopeUtils.pattern_key(requester_id=requester_id, critique_response_from=None)

    # --- Key parsing ---

    @staticmethod
    def parse_key(key: str) -> dict[str, str]:
        """Extract fields from any critique key.

        Returns:
            Dict of field names to values (e.g., ``{"requester_id": "...", "critique_request_from_peer": "True"}``).
        """
        return ScopeUtils.parse_key("", key)


class WorkingSetStateProtocol(BlackboardProtocol):
    """Protocol for publishing/observing VCM working set state.

    Colony-scoped. Single shared state keys — any agent can read,
    only ``WorkingSetCapability`` writes.

    Key types:

    - ``state:working_set:cluster`` — cluster-wide working set state
    - ``state:working_set:page_status`` — per-page status
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    _PREFIX = "state:working_set:"

    @staticmethod
    def cluster_state_key() -> str:
        """Key for cluster-wide working set state."""
        return f"{WorkingSetStateProtocol._PREFIX}cluster"

    @staticmethod
    def page_status_key() -> str:
        """Key for per-page status."""
        return f"{WorkingSetStateProtocol._PREFIX}page_status"

    @staticmethod
    def state_pattern() -> str:
        """Pattern matching all working set state updates."""
        return f"{WorkingSetStateProtocol._PREFIX}*"


class ConsistencyCheckProtocol(BlackboardProtocol):
    """Protocol for consistency check requests/results.

    Colony-scoped. ``request_id`` provides per-check uniqueness.

    Key types:

    - ``consistency_check_request:{request_id}`` — request
    - ``consistency_check_result:{request_id}`` — result (exact, not wildcard)
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    _REQUEST_PREFIX = "consistency_check_request:"
    _RESULT_PREFIX = "consistency_check_result:"

    # --- Key construction ---

    @staticmethod
    def request_key(request_id: str) -> str:
        return f"{ConsistencyCheckProtocol._REQUEST_PREFIX}{request_id}"

    @staticmethod
    def result_key(request_id: str) -> str:
        return f"{ConsistencyCheckProtocol._RESULT_PREFIX}{request_id}"

    # --- Pattern construction ---

    @staticmethod
    def request_pattern() -> str:
        return f"{ConsistencyCheckProtocol._REQUEST_PREFIX}*"

    @staticmethod
    def result_pattern() -> str:
        return f"{ConsistencyCheckProtocol._RESULT_PREFIX}*"

    # --- Key parsing ---

    @staticmethod
    def parse_request_key(key: str) -> str:
        """Extract request_id from a request key."""
        if not key.startswith(ConsistencyCheckProtocol._REQUEST_PREFIX):
            raise ValueError(f"Not a ConsistencyCheckProtocol request key: {key!r}")
        return key[len(ConsistencyCheckProtocol._REQUEST_PREFIX):]

    @staticmethod
    def parse_result_key(key: str) -> str:
        """Extract request_id from a result key."""
        if not key.startswith(ConsistencyCheckProtocol._RESULT_PREFIX):
            raise ValueError(f"Not a ConsistencyCheckProtocol result key: {key!r}")
        return key[len(ConsistencyCheckProtocol._RESULT_PREFIX):]


class GroundingProtocol(BlackboardProtocol):
    """Protocol for grounding requests/results.

    Agent-scoped. ``request_id`` provides per-request uniqueness.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def request_key(request_id: str) -> str:
        return ScopeUtils.format_key(grounding_request=request_id)

    @staticmethod
    def result_key(request_id: str) -> str:
        return ScopeUtils.format_key(grounding_result=request_id)

    @staticmethod
    def request_pattern() -> str:
        return ScopeUtils.pattern_key(grounding_request=None)

    @staticmethod
    def result_pattern() -> str:
        return ScopeUtils.pattern_key(grounding_result=None)

    @staticmethod
    def parse_request_key(key: str) -> str:
        """Extract request_id from a grounding request key."""
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("grounding_request", "")

    @staticmethod
    def parse_result_key(key: str) -> str:
        """Extract request_id from a grounding result key."""
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("grounding_result", "")


class GoalAlignmentProtocol(BlackboardProtocol):
    """Protocol for goal alignment requests and joint goal registrations.

    Colony-scoped.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    @staticmethod
    def request_key(request_id: str) -> str:
        return ScopeUtils.format_key(goal_alignment_request=request_id)

    @staticmethod
    def joint_goal_key(goal_id: str) -> str:
        return ScopeUtils.format_key(joint_goal_registration=goal_id)

    @staticmethod
    def request_pattern() -> str:
        return ScopeUtils.pattern_key(goal_alignment_request=None)

    @staticmethod
    def joint_goal_pattern() -> str:
        return ScopeUtils.pattern_key(joint_goal_registration=None)

    @staticmethod
    def parse_request_key(key: str) -> str:
        """Extract request_id from a goal alignment request key."""
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("goal_alignment_request", "")

    @staticmethod
    def result_key(request_id: str) -> str:
        """Key for a goal alignment result."""
        return f"goal_alignment_result:{request_id}"

    @staticmethod
    def result_pattern() -> str:
        """Pattern matching goal alignment results."""
        return "goal_alignment_result:*"

    @staticmethod
    def joint_goal_state_key(goal_id: str) -> str:
        """Key for a joint goal state entry."""
        return f"joint_goal:{goal_id}"


class PlanProtocol(BlackboardProtocol):
    """Protocol for plan publication on colony-wide plan blackboard.

    Colony-scoped (with ``:action_plans`` suffix on scope_id).
    ``agent_id`` in key because all agents share one plan blackboard.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    _APPROVAL_PREFIX = "plan_approval_request:"
    _NOTIFICATION_PREFIX = "agent_notification:"
    _SUBSCRIPTION_PREFIX = "plan_subscription:"

    @staticmethod
    def plan_key(agent_id: str) -> str:
        """Key for an agent's current plan."""
        return ScopeUtils.format_key(agent_id=agent_id)

    @staticmethod
    def plan_pattern(agent_id: str | None = None) -> str:
        """Pattern matching plans (optionally for a specific agent)."""
        return ScopeUtils.pattern_key(agent_id=agent_id)

    @staticmethod
    def approval_request_key(plan_id: str) -> str:
        return f"{PlanProtocol._APPROVAL_PREFIX}{plan_id}"

    @staticmethod
    def notification_key(agent_id: str, timestamp: float) -> str:
        return f"{PlanProtocol._NOTIFICATION_PREFIX}{agent_id}:{timestamp}"

    @staticmethod
    def subscription_key(plan_id: str, subscriber_id: str) -> str:
        return f"{PlanProtocol._SUBSCRIPTION_PREFIX}{plan_id}:{subscriber_id}"

    @staticmethod
    def parse_plan_key(key: str) -> str:
        """Extract agent_id from a plan key."""
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("agent_id", "")


class ErrorSignalProtocol(BlackboardProtocol):
    """Protocol for error signals.

    Used by coordinators to detect errors from worker agents.
    Agent-scoped (errors are written to the failing agent's own blackboard).
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    _PREFIX = "error:"

    @staticmethod
    def error_key(error_id: str) -> str:
        return f"{ErrorSignalProtocol._PREFIX}{error_id}"

    @staticmethod
    def error_pattern() -> str:
        return f"{ErrorSignalProtocol._PREFIX}*"

    @staticmethod
    def parse_error_key(key: str) -> str:
        """Extract error_id from an error key."""
        if not key.startswith(ErrorSignalProtocol._PREFIX):
            raise ValueError(f"Not an ErrorSignalProtocol key: {key!r}")
        return key[len(ErrorSignalProtocol._PREFIX):]


class DependencyQueryProtocol(BlackboardProtocol):
    """Protocol for inter-agent dependency queries.

    Used by impact analysis agents to query peers about dependencies
    across page boundaries. Colony-scoped since queries go between agents.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    @staticmethod
    def query_key(agent_id: str, query_id: str) -> str:
        return ScopeUtils.format_key(agent_id=agent_id, dependency_query=query_id)

    @staticmethod
    def query_pattern(agent_id: str | None = None) -> str:
        return ScopeUtils.pattern_key(agent_id=agent_id, dependency_query=None)

    @staticmethod
    def result_key(agent_id: str, query_id: str) -> str:
        return ScopeUtils.format_key(agent_id=agent_id, dependency_result=query_id)

    @staticmethod
    def parse_query_key(key: str) -> dict[str, str]:
        return ScopeUtils.parse_key("", key)


class ReputationProtocol(BlackboardProtocol):
    """Protocol for reputation tracking in multi-agent games.

    Colony-scoped. Multiple agents submit reputation update requests.
    A dedicated handler aggregates and publishes reputation scores.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    @staticmethod
    def update_request_key(requesting_agent_id: str) -> str:
        return ScopeUtils.format_key(reputation="update", requesting_agent_id=requesting_agent_id)

    @staticmethod
    def result_key() -> str:
        return ScopeUtils.format_key(reputation="result")

    @staticmethod
    def update_request_pattern() -> str:
        return ScopeUtils.pattern_key(reputation="update", requesting_agent_id=None)

    @staticmethod
    def state_pattern() -> str:
        """Pattern for game state changes that trigger reputation updates."""
        return ScopeUtils.pattern_key(state=None)

    @staticmethod
    def task_outcome_pattern() -> str:
        """Pattern for task outcomes that affect reputation."""
        return ScopeUtils.pattern_key(task_outcome=None)

    @staticmethod
    def all_input_patterns() -> list[str]:
        """All patterns this protocol monitors."""
        return [
            ReputationProtocol.update_request_pattern(),
            ReputationProtocol.state_pattern(),
            ReputationProtocol.task_outcome_pattern(),
        ]

    @staticmethod
    def parse_update_request_key(key: str) -> dict[str, str]:
        return ScopeUtils.parse_key("", key)

    @staticmethod
    def agent_reputation_key(agent_id: str) -> str:
        """Key for a specific agent's reputation record."""
        return f"agent:{agent_id}"


class ConsciousnessProtocol(BlackboardProtocol):
    """Protocol for consciousness state publication.

    Agent-scoped. Tracks consciousness level, attention focus,
    and meta-cognitive state.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def state_key(state_type: str) -> str:
        return ScopeUtils.format_key(consciousness=state_type)

    @staticmethod
    def state_pattern() -> str:
        return ScopeUtils.pattern_key(consciousness=None)

    @staticmethod
    def parse_state_key(key: str) -> str:
        """Extract state_type from a consciousness key."""
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("consciousness", "")


class ReflectionProtocol(BlackboardProtocol):
    """Protocol for reflection requests/results.

    Agent-scoped. Used to trigger and receive self-reflection analysis.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def request_key(request_id: str) -> str:
        return ScopeUtils.format_key(reflection_request=request_id)

    @staticmethod
    def result_key(request_id: str) -> str:
        return ScopeUtils.format_key(reflection_result=request_id)

    @staticmethod
    def request_pattern() -> str:
        return ScopeUtils.pattern_key(reflection_request=None)

    @staticmethod
    def result_pattern() -> str:
        return ScopeUtils.pattern_key(reflection_result=None)

    @staticmethod
    def parse_request_key(key: str) -> str:
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("reflection_request", "")

    @staticmethod
    def response_key(request_id: str) -> str:
        """Key for a reflection response."""
        return f"reflection_response:{request_id}"


class AnalysisResultProtocol(BlackboardProtocol):
    """Protocol for analysis result publication.

    Agent-scoped. Used by ``AdaptiveQueryGenerator`` to monitor
    completed analysis results and generate follow-up queries.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def result_key(result_id: str) -> str:
        return ScopeUtils.format_key(analysis_result=result_id)

    @staticmethod
    def result_pattern() -> str:
        return ScopeUtils.pattern_key(analysis_result=None)

    @staticmethod
    def parse_result_key(key: str) -> str:
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("analysis_result", "")


class ExplorationProtocol(BlackboardProtocol):
    """Protocol for query-driven exploration requests/results.

    Agent-scoped.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def request_key(request_id: str) -> str:
        return ScopeUtils.format_key(exploration_request=request_id)

    @staticmethod
    def request_pattern() -> str:
        return ScopeUtils.pattern_key(exploration_request=None)

    @staticmethod
    def parse_request_key(key: str) -> str:
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("exploration_request", "")


class IncrementalQueryProtocol(BlackboardProtocol):
    """Protocol for incremental query requests.

    Agent-scoped.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def request_key(request_id: str) -> str:
        return ScopeUtils.format_key(incremental_query_request=request_id)

    @staticmethod
    def request_pattern() -> str:
        return ScopeUtils.pattern_key(incremental_query_request=None)

    @staticmethod
    def parse_request_key(key: str) -> str:
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("incremental_query_request", "")


class MultiHopSearchProtocol(BlackboardProtocol):
    """Protocol for multi-hop search requests.

    Agent-scoped.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def request_key(request_id: str) -> str:
        return ScopeUtils.format_key(multi_hop_search_request=request_id)

    @staticmethod
    def request_pattern() -> str:
        return ScopeUtils.pattern_key(multi_hop_search_request=None)

    @staticmethod
    def parse_request_key(key: str) -> str:
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("multi_hop_search_request", "")


# ---------------------------------------------------------------------------
# Epistemic / game-theoretic protocols
# ---------------------------------------------------------------------------

class EpistemicProtocol(BlackboardProtocol):
    """Protocol for epistemic state (propositions, intentions) in multi-agent games."""

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    @staticmethod
    def proposition_key(proposition_id: str) -> str:
        return f"proposition:{proposition_id}"

    @staticmethod
    def intention_key(intention_id: str) -> str:
        return f"intention:{intention_id}"

    @staticmethod
    def joint_intention_key(intention_id: str) -> str:
        return f"joint_intention:{intention_id}"


# ---------------------------------------------------------------------------
# Memory / record protocols
# ---------------------------------------------------------------------------

class MemoryRecordProtocol(BlackboardProtocol):
    """Protocol for memory record storage."""

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def record_key(record_id: str) -> str:
        return f"memory_record:{record_id}"

    @staticmethod
    def consolidated_key(timestamp: int, count: int) -> str:
        return f"consolidated:{timestamp}:{count}"


class KeyRegistryProtocol(BlackboardProtocol):
    """Protocol for attention key registry entries."""

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def page_key(page_id: str) -> str:
        return f"page_id:{page_id}"

    @staticmethod
    def cluster_key(cluster_id: str) -> str:
        return f"cluster_id:{cluster_id}"


# ---------------------------------------------------------------------------
# VCM / result protocols
# ---------------------------------------------------------------------------

class VCMAnalysisProtocol(BlackboardProtocol):
    """Protocol for VCM analysis capability keys."""

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    @staticmethod
    def result_key(page_id: str) -> str:
        return f"vcm_result:{page_id}"

    @staticmethod
    def revisit_queue_key() -> str:
        return "vcm_revisit_queue"

    @staticmethod
    def outstanding_queries_key() -> str:
        return "vcm_outstanding_queries"

    @staticmethod
    def state_key() -> str:
        return "vcm_state"


class ResultStorageProtocol(BlackboardProtocol):
    """Protocol for ResultCapability storage."""

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    @staticmethod
    def partial_key(result_id: str) -> str:
        return f"results:partial:{result_id}"

    @staticmethod
    def index_key() -> str:
        return "results:index"


# ---------------------------------------------------------------------------
# Relationship / graph protocols
# ---------------------------------------------------------------------------

class RelationshipProtocol(BlackboardProtocol):
    """Protocol for relationship/page graph entries."""

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def relationship_key(source: str, target: str, rel_type: str) -> str:
        return f"relationship:{source}:{target}:{rel_type}"

    @staticmethod
    def relationship_pattern() -> str:
        return "relationship:*"


# ---------------------------------------------------------------------------
# Plan learning / action policy protocols
# ---------------------------------------------------------------------------

class PlanLearningProtocol(BlackboardProtocol):
    """Protocol for plan execution learning records."""

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def execution_key(plan_id: str) -> str:
        return f"execution:{plan_id}"

    @staticmethod
    def execution_pattern() -> str:
        return "execution:*"


class ActionPolicyProtocol(BlackboardProtocol):
    """Protocol for action policy internal state."""

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def iteration_key(namespace_prefix: str, iteration_num: int) -> str:
        return f"{namespace_prefix}:action_policy_iteration:{iteration_num}"

    @staticmethod
    def repl_key(agent_id: str, var_name: str, timestamp_ns: int) -> str:
        return f"repl:{agent_id}:{var_name}:{timestamp_ns}"


# ---------------------------------------------------------------------------
# Sample analysis protocols
# ---------------------------------------------------------------------------

class BasicAnalysisProtocol(BlackboardProtocol):
    """Protocol for basic code analysis sample keys."""

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def page_summary_key(page_id: str) -> str:
        return f"page_summary:{page_id}"

    @staticmethod
    def cluster_analysis_complete_key(agent_id: str) -> str:
        return f"cluster_analysis_complete:{agent_id}"

    @staticmethod
    def critique_key(agent_id: str) -> str:
        return f"critique:{agent_id}"

    @staticmethod
    def revision_request_key(agent_id: str) -> str:
        return f"revision_request:{agent_id}"


class ImpactAnalysisProtocol(BlackboardProtocol):
    """Protocol for impact analysis sample keys."""

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def impact_key(page_id: str) -> str:
        return f"impact:{page_id}"

    @staticmethod
    def test_coverage_key(page_id: str) -> str:
        return f"test_coverage:{page_id}"

    @staticmethod
    def dependency_graph_key(dep_key: str) -> str:
        return f"dependency_graph:{dep_key}"


class IntentAnalysisProtocol(BlackboardProtocol):
    """Protocol for intent analysis sample keys."""

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def intent_hierarchy_key(category: str) -> str:
        return f"intent_hierarchy:{category}"

    @staticmethod
    def intent_misalignments_key() -> str:
        return "intent_misalignments"


class HypothesisTrackingProtocol(BlackboardProtocol):
    """Protocol for hypothesis tracking state."""

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def hypotheses_key() -> str:
        """Key for the bulk tracked hypotheses dict."""
        return "tracked_hypotheses"

    @staticmethod
    def games_key() -> str:
        """Key for the hypothesis-to-game mappings dict."""
        return "hypothesis_games"


class ComplianceAnalysisProtocol(BlackboardProtocol):
    """Protocol for compliance analysis sample keys."""

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def obligation_key(obligation_id: str) -> str:
        return f"obligation:{obligation_id}"


class SlicingAnalysisProtocol(BlackboardProtocol):
    """Protocol for slicing analysis sample keys."""

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def interprocedural_resolutions_key() -> str:
        return "interprocedural_resolutions"


class InterruptionProtocol(BlackboardProtocol):
    """Protocol for external agent/session interruption.

    Enables the dashboard or CLI to interrupt running agents by writing
    to the colony-scoped blackboard. Agents with an ``@event_handler``
    for ``interrupt_pattern()`` react by cancelling, suspending, or
    applying configuration changes.

    Key format: ``interrupt:{agent_id}``

    Value schema::

        {
            "action": "cancel" | "suspend" | "reconfigure",
            "reason": "user requested via dashboard",
            "config": { ... }  // only for "reconfigure" action
        }

    Colony-scoped so any agent can be interrupted regardless of which
    deployment replica it's running on.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    _PREFIX = "interrupt:"

    @staticmethod
    def interrupt_key(agent_id: str) -> str:
        """Key for interrupting a specific agent."""
        return f"{InterruptionProtocol._PREFIX}{agent_id}"

    @staticmethod
    def interrupt_pattern() -> str:
        """Pattern matching all interruption events."""
        return f"{InterruptionProtocol._PREFIX}*"

    @staticmethod
    def parse_interrupt_key(key: str) -> str:
        """Extract agent_id from an interrupt key."""
        if not key.startswith(InterruptionProtocol._PREFIX):
            raise ValueError(f"Not an InterruptionProtocol key: {key!r}")
        return key[len(InterruptionProtocol._PREFIX):]

    @staticmethod
    def session_interrupt_key(session_id: str) -> str:
        """Key for interrupting all agents in a session."""
        return f"session_interrupt:{session_id}"

    @staticmethod
    def session_interrupt_pattern() -> str:
        """Pattern matching session-level interruption events."""
        return "session_interrupt:*"
