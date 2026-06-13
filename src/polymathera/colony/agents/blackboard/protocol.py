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


class HumanApprovalProtocol(BlackboardProtocol):
    """Protocol for typed human-approval requests/responses on the
    session blackboard.

    Operates at session scope so multiple agents in the same session
    share the topic, the SessionAgent can surface requests to the UI,
    and the Web UI HTTP endpoint can write responses back into the
    same scope. The ``request_id`` correlates a request with its
    response.

    Key formats:

    - ``human_approval:request:{request_id}`` — agent posts a typed
      ``HumanApprovalRequest`` payload.
    - ``human_approval:response:{request_id}`` — Web UI writes the
      user's typed ``HumanApprovalResponse`` payload.

    Example::

        # Agent capability publishes a request:
        key = HumanApprovalProtocol.request_key("appr_abc")
        await session_blackboard.write(key, request_payload)

        # Web UI HTTP endpoint posts the user's response:
        key = HumanApprovalProtocol.response_key("appr_abc")
        await session_blackboard.write(key, response_payload)

        # Agent's HumanApprovalCapability subscribes:
        @event_handler(pattern=HumanApprovalProtocol.response_pattern())
        async def on_response(self, event, repl):
            request_id = HumanApprovalProtocol.parse_response_key(event.key)
            ...
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.SESSION

    # --- Key construction ---

    @staticmethod
    def request_key(request_id: str) -> str:
        """Key for a human-approval request."""
        return f"human_approval:request:{request_id}"

    @staticmethod
    def response_key(request_id: str) -> str:
        """Key for the user's typed response."""
        return f"human_approval:response:{request_id}"

    @staticmethod
    def consumption_key(request_id: str) -> str:
        """Key marking an ``approve_once`` response as already consumed
        by one gated dispatch. Idempotent; presence-only payload."""
        return f"human_approval:consumed:{request_id}"

    # --- Pattern construction ---

    @staticmethod
    def request_pattern() -> str:
        """Pattern matching all human-approval requests in the session."""
        return "human_approval:request:*"

    @staticmethod
    def response_pattern() -> str:
        """Pattern matching all human-approval responses in the session."""
        return "human_approval:response:*"

    @staticmethod
    def consumption_pattern() -> str:
        return "human_approval:consumed:*"

    # --- Key parsing ---

    @staticmethod
    def parse_request_key(key: str) -> str:
        """Extract request_id from a request key."""
        prefix = "human_approval:request:"
        if not key.startswith(prefix):
            raise ValueError(f"Not a HumanApprovalProtocol request key: {key!r}")
        return key[len(prefix):]

    @staticmethod
    def parse_response_key(key: str) -> str:
        """Extract request_id from a response key."""
        prefix = "human_approval:response:"
        if not key.startswith(prefix):
            raise ValueError(f"Not a HumanApprovalProtocol response key: {key!r}")
        return key[len(prefix):]


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


class VCMEventProtocol(BlackboardProtocol):
    """Protocol for Virtual Context Manager lifecycle events.

    Emitted by ``VCMCapability`` when it drives the VCM on behalf of the
    agent (mapping or unmapping a scope, requesting page loads, and — in
    Phase 3 — re-indexing after a filesystem change). Other agents that
    want to react to VCM activity subscribe to the relevant pattern.

    Operates at the VCM capability's own partition (typically session
    scope). Because ``scope_id`` values used by the VCM contain colons,
    all key parsing uses ``str.split(":", 1)`` / ``split(":", 2)`` to
    preserve the remainder verbatim.

    Key types:

    - ``mapped:{scope_id}``         — a scope was just mapped
    - ``unmapped:{scope_id}``       — a scope was just unmapped
    - ``reindexed:{scope_id}``      — page graph rebuilt (watch-driven)
    - ``page_fault:{page_id}``      — an agent requested a page load
    - ``watch_fired:{watch_id}``    — a filesystem watcher triggered
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.SESSION

    _MAPPED_PREFIX: ClassVar[str] = "mapped:"
    _UNMAPPED_PREFIX: ClassVar[str] = "unmapped:"
    _REINDEXED_PREFIX: ClassVar[str] = "reindexed:"
    _PAGE_FAULT_PREFIX: ClassVar[str] = "page_fault:"
    _WATCH_FIRED_PREFIX: ClassVar[str] = "watch_fired:"

    # --- Key construction ---

    @staticmethod
    def mapped_key(scope_id: str) -> str:
        """Key for a 'scope mapped' event."""
        return f"{VCMEventProtocol._MAPPED_PREFIX}{scope_id}"

    @staticmethod
    def unmapped_key(scope_id: str) -> str:
        """Key for a 'scope unmapped' event."""
        return f"{VCMEventProtocol._UNMAPPED_PREFIX}{scope_id}"

    @staticmethod
    def reindexed_key(scope_id: str) -> str:
        """Key for a 'scope reindexed' event."""
        return f"{VCMEventProtocol._REINDEXED_PREFIX}{scope_id}"

    @staticmethod
    def page_fault_key(page_id: str) -> str:
        """Key for a 'page fault issued' event."""
        return f"{VCMEventProtocol._PAGE_FAULT_PREFIX}{page_id}"

    @staticmethod
    def watch_fired_key(watch_id: str) -> str:
        """Key for a 'filesystem watch fired' event."""
        return f"{VCMEventProtocol._WATCH_FIRED_PREFIX}{watch_id}"

    # --- Pattern construction ---

    @staticmethod
    def mapped_pattern() -> str:
        return f"{VCMEventProtocol._MAPPED_PREFIX}*"

    @staticmethod
    def unmapped_pattern() -> str:
        return f"{VCMEventProtocol._UNMAPPED_PREFIX}*"

    @staticmethod
    def reindexed_pattern() -> str:
        return f"{VCMEventProtocol._REINDEXED_PREFIX}*"

    @staticmethod
    def page_fault_pattern() -> str:
        return f"{VCMEventProtocol._PAGE_FAULT_PREFIX}*"

    @staticmethod
    def watch_fired_pattern() -> str:
        return f"{VCMEventProtocol._WATCH_FIRED_PREFIX}*"

    # --- Key parsing ---

    @staticmethod
    def parse_mapped_key(key: str) -> str:
        """Extract scope_id from a ``mapped:*`` key."""
        if not key.startswith(VCMEventProtocol._MAPPED_PREFIX):
            raise ValueError(f"Not a VCMEventProtocol mapped key: {key!r}")
        return key[len(VCMEventProtocol._MAPPED_PREFIX):]

    @staticmethod
    def parse_unmapped_key(key: str) -> str:
        if not key.startswith(VCMEventProtocol._UNMAPPED_PREFIX):
            raise ValueError(f"Not a VCMEventProtocol unmapped key: {key!r}")
        return key[len(VCMEventProtocol._UNMAPPED_PREFIX):]

    @staticmethod
    def parse_reindexed_key(key: str) -> str:
        if not key.startswith(VCMEventProtocol._REINDEXED_PREFIX):
            raise ValueError(f"Not a VCMEventProtocol reindexed key: {key!r}")
        return key[len(VCMEventProtocol._REINDEXED_PREFIX):]

    @staticmethod
    def parse_page_fault_key(key: str) -> str:
        if not key.startswith(VCMEventProtocol._PAGE_FAULT_PREFIX):
            raise ValueError(f"Not a VCMEventProtocol page_fault key: {key!r}")
        return key[len(VCMEventProtocol._PAGE_FAULT_PREFIX):]

    @staticmethod
    def parse_watch_fired_key(key: str) -> str:
        if not key.startswith(VCMEventProtocol._WATCH_FIRED_PREFIX):
            raise ValueError(f"Not a VCMEventProtocol watch_fired key: {key!r}")
        return key[len(VCMEventProtocol._WATCH_FIRED_PREFIX):]


class ConvergenceDispatchProtocol(BlackboardProtocol):
    """Protocol for ``ConvergenceRuntime`` -> ``ConvergenceCapability`` dispatch.

    The runtime writes one event per matched ``PageSubscription`` to the
    subscribing capability's blackboard scope under this key shape. The
    capability subscribes via ``@event_handler(pattern=...)`` and uses
    the embedded ``subscription_id`` to correlate the dispatch back to
    the subscription it owns.

    Operates at agent scope — each capability has its own dispatch
    mailbox on its primary blackboard. The runtime is the sole writer;
    the owning capability is the sole reader.

    Key format:

    - ``convergence:dispatch:{subscription_id}``
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    _PREFIX: ClassVar[str] = "convergence:dispatch:"

    @staticmethod
    def dispatch_key(subscription_id: str) -> str:
        """Key the runtime writes when a subscription matches."""
        return f"{ConvergenceDispatchProtocol._PREFIX}{subscription_id}"

    @staticmethod
    def dispatch_pattern() -> str:
        """Pattern matching every dispatch on the subscriber's scope."""
        return f"{ConvergenceDispatchProtocol._PREFIX}*"

    @staticmethod
    def parse_dispatch_key(key: str) -> str:
        """Extract ``subscription_id`` from a dispatch key."""
        prefix = ConvergenceDispatchProtocol._PREFIX
        if not key.startswith(prefix):
            raise ValueError(
                f"Not a ConvergenceDispatchProtocol key: {key!r}",
            )
        return key[len(prefix):]


class ConvergenceQuiescenceProtocol(BlackboardProtocol):
    """Protocol for ``ConvergenceRuntime`` quiescence events.

    The runtime emits one event per episode boundary onto the colony
    blackboard scope. Subscribers — typically ``DesignCheckpointer``
    auto-tagging an ``auto_quiescence_<iso8601>`` checkpoint, and
    future analogous "react when the design has settled" agents —
    listen via ``@event_handler(pattern=...)``.

    Operates at colony scope: quiescence is a colony-wide observable,
    not a per-agent one. The episode_id correlates the event with the
    counters carried in the payload.

    Key format:

    - ``convergence:quiescence:{episode_id}``
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    _PREFIX: ClassVar[str] = "convergence:quiescence:"

    @staticmethod
    def quiescence_key(episode_id: str) -> str:
        """Key the runtime writes when an episode reaches quiescence."""
        return f"{ConvergenceQuiescenceProtocol._PREFIX}{episode_id}"

    @staticmethod
    def quiescence_pattern() -> str:
        """Pattern matching every quiescence event on the colony scope."""
        return f"{ConvergenceQuiescenceProtocol._PREFIX}*"

    @staticmethod
    def parse_quiescence_key(key: str) -> str:
        """Extract ``episode_id`` from a quiescence key."""
        prefix = ConvergenceQuiescenceProtocol._PREFIX
        if not key.startswith(prefix):
            raise ValueError(
                f"Not a ConvergenceQuiescenceProtocol key: {key!r}",
            )
        return key[len(prefix):]


class DesignMonorepoEventProtocol(BlackboardProtocol):
    """Protocol for design-monorepo lifecycle events.

    Emitted by :class:`DesignCheckpointer` when it translates a raw
    ``VCMEventProtocol.page_changed`` event from the global VCM
    mapping into a coarser branch-update / merge / checkpoint event
    that the agent's action policy can react to. Subscribers
    typically watch ``branch_changed:*`` to detect upstream movement
    on the branch they are working from and trigger a checkout /
    merge / rebase against their per-agent local clone.

    Key formats:

    - ``branch_changed:{scope_id}``      — upstream commits landed on
      the tracked branch of this VCM scope.
    - ``branch_merged:{scope_id}``       — a fork branch was merged
      into the tracked branch.
    - ``checkpoint_emitted:{scope_id}``  — a checkpoint tag was
      created on this branch.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.SESSION

    _BRANCH_CHANGED_PREFIX: ClassVar[str] = "branch_changed:"
    _BRANCH_MERGED_PREFIX: ClassVar[str] = "branch_merged:"
    _CHECKPOINT_EMITTED_PREFIX: ClassVar[str] = "checkpoint_emitted:"
    _EXTENSION_AUTHORED_PREFIX: ClassVar[str] = "extension_authored:"
    _PROJECT_ARTIFACT_AUTHORED_PREFIX: ClassVar[str] = "project_artifact_authored:"
    _PROTECTED_OP_PENDING_PREFIX: ClassVar[str] = "protected_op:pending:"
    _PROTECTED_OP_OUTCOME_PREFIX: ClassVar[str] = "protected_op:outcome:"

    @staticmethod
    def branch_changed_key(scope_id: str) -> str:
        return f"{DesignMonorepoEventProtocol._BRANCH_CHANGED_PREFIX}{scope_id}"

    @staticmethod
    def branch_merged_key(scope_id: str) -> str:
        return f"{DesignMonorepoEventProtocol._BRANCH_MERGED_PREFIX}{scope_id}"

    @staticmethod
    def checkpoint_emitted_key(scope_id: str) -> str:
        return f"{DesignMonorepoEventProtocol._CHECKPOINT_EMITTED_PREFIX}{scope_id}"

    @staticmethod
    def extension_authored_key(surface: str, name: str) -> str:
        """L1-E authoring event. Tail is ``<surface>:<name>`` where
        ``surface`` is one of the five
        :data:`polymathera.colony.design_monorepo.manifest.DEFAULT_SURFACE_DIRS`
        keys and ``name`` is the agent-supplied extension name."""
        return (
            f"{DesignMonorepoEventProtocol._EXTENSION_AUTHORED_PREFIX}"
            f"{surface}:{name}"
        )

    @staticmethod
    def branch_changed_pattern() -> str:
        return f"{DesignMonorepoEventProtocol._BRANCH_CHANGED_PREFIX}*"

    @staticmethod
    def branch_merged_pattern() -> str:
        return f"{DesignMonorepoEventProtocol._BRANCH_MERGED_PREFIX}*"

    @staticmethod
    def checkpoint_emitted_pattern() -> str:
        return f"{DesignMonorepoEventProtocol._CHECKPOINT_EMITTED_PREFIX}*"

    @staticmethod
    def extension_authored_pattern() -> str:
        return f"{DesignMonorepoEventProtocol._EXTENSION_AUTHORED_PREFIX}*"

    @staticmethod
    def parse_branch_changed_key(key: str) -> str:
        prefix = DesignMonorepoEventProtocol._BRANCH_CHANGED_PREFIX
        if not key.startswith(prefix):
            raise ValueError(f"Not a DesignMonorepoEventProtocol branch_changed key: {key!r}")
        return key[len(prefix):]

    @staticmethod
    def parse_extension_authored_key(key: str) -> tuple[str, str]:
        """Inverse of :meth:`extension_authored_key` — returns
        ``(surface, name)``. ``name`` may contain ``:`` so the tail is
        split on the first ``:`` only."""
        prefix = DesignMonorepoEventProtocol._EXTENSION_AUTHORED_PREFIX
        if not key.startswith(prefix):
            raise ValueError(
                f"Not a DesignMonorepoEventProtocol extension_authored key: {key!r}",
            )
        tail = key[len(prefix):]
        if ":" not in tail:
            raise ValueError(
                f"Malformed extension_authored key (no surface:name tail): {key!r}",
            )
        surface, name = tail.split(":", 1)
        return surface, name

    @staticmethod
    def project_artifact_authored_key(action_kind: str, primary_path: str) -> str:
        """L1-F authoring event. Tail is ``<action_kind>:<primary_path>``
        where ``action_kind`` is one of
        :data:`polymathera.colony.design_monorepo.models.PROJECT_ACTION_KINDS`
        and ``primary_path`` is the working-tree-relative path of the
        primary file the action touched (the destination for
        ``move_file``)."""
        return (
            f"{DesignMonorepoEventProtocol._PROJECT_ARTIFACT_AUTHORED_PREFIX}"
            f"{action_kind}:{primary_path}"
        )

    @staticmethod
    def project_artifact_authored_pattern() -> str:
        return f"{DesignMonorepoEventProtocol._PROJECT_ARTIFACT_AUTHORED_PREFIX}*"

    @staticmethod
    def parse_project_artifact_authored_key(key: str) -> tuple[str, str]:
        """Inverse of :meth:`project_artifact_authored_key` — returns
        ``(action_kind, primary_path)``. Path may contain ``:`` so the
        tail is split on the first ``:`` only."""
        prefix = DesignMonorepoEventProtocol._PROJECT_ARTIFACT_AUTHORED_PREFIX
        if not key.startswith(prefix):
            raise ValueError(
                f"Not a DesignMonorepoEventProtocol project_artifact_authored key: {key!r}",
            )
        tail = key[len(prefix):]
        if ":" not in tail:
            raise ValueError(
                f"Malformed project_artifact_authored key: {key!r}",
            )
        kind, path = tail.split(":", 1)
        return kind, path

    # ---- Protected-branch gating (master §3.1 access-control) -----------

    @staticmethod
    def protected_op_pending_key(request_id: str) -> str:
        """Key for the persisted :class:`PendingProtectedOp` record an
        agent writes when a `DesignCheckpointer` action pauses on a
        human-approval gate. The ``request_id`` matches the paired
        :class:`HumanApprovalRequest`."""
        return (
            f"{DesignMonorepoEventProtocol._PROTECTED_OP_PENDING_PREFIX}"
            f"{request_id}"
        )

    @staticmethod
    def protected_op_pending_pattern() -> str:
        return f"{DesignMonorepoEventProtocol._PROTECTED_OP_PENDING_PREFIX}*"

    @staticmethod
    def parse_protected_op_pending_key(key: str) -> str:
        prefix = DesignMonorepoEventProtocol._PROTECTED_OP_PENDING_PREFIX
        if not key.startswith(prefix):
            raise ValueError(
                f"Not a DesignMonorepoEventProtocol protected_op pending key: {key!r}",
            )
        return key[len(prefix):]

    @staticmethod
    def protected_op_outcome_key(request_id: str) -> str:
        """Key for the :class:`ProtectedOpOutcome` written once the
        operator's response has been processed (approve → executed,
        reject → rejected, runtime failure → failed)."""
        return (
            f"{DesignMonorepoEventProtocol._PROTECTED_OP_OUTCOME_PREFIX}"
            f"{request_id}"
        )

    @staticmethod
    def protected_op_outcome_pattern() -> str:
        return f"{DesignMonorepoEventProtocol._PROTECTED_OP_OUTCOME_PREFIX}*"

    @staticmethod
    def parse_protected_op_outcome_key(key: str) -> str:
        prefix = DesignMonorepoEventProtocol._PROTECTED_OP_OUTCOME_PREFIX
        if not key.startswith(prefix):
            raise ValueError(
                f"Not a DesignMonorepoEventProtocol protected_op outcome key: {key!r}",
            )
        return key[len(prefix):]


class GitHubEventProtocol(BlackboardProtocol):
    """Protocol for GitHub webhook events surfaced on the blackboard.

    A future ``POST /api/v1/github/webhook`` endpoint normalizes
    incoming signed webhooks into writes at these keys; the
    ``GitHubCapability`` also writes ``audit:github:*`` records for
    mutations it performs. Other capabilities subscribe to the
    relevant pattern to react.

    Key formats — each tail contains the repo plus a numeric id (or
    project item id). Parsing splits on ``:`` with a fixed arity so
    callers avoid raw ``str.split`` logic.

    - ``github:issue_opened:{owner}/{repo}:{number}``
    - ``github:issue_commented:{owner}/{repo}:{number}``
    - ``github:issue_closed:{owner}/{repo}:{number}``
    - ``github:pr_opened:{owner}/{repo}:{number}``
    - ``github:pr_review_requested:{owner}/{repo}:{number}``
    - ``github:pr_merged:{owner}/{repo}:{number}``
    - ``github:project_item_changed:{project_id}:{item_id}``
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    _ISSUE_OPENED = "github:issue_opened:"
    _ISSUE_COMMENTED = "github:issue_commented:"
    _ISSUE_CLOSED = "github:issue_closed:"
    _PR_OPENED = "github:pr_opened:"
    _PR_REVIEW_REQUESTED = "github:pr_review_requested:"
    _PR_MERGED = "github:pr_merged:"
    _PROJECT_ITEM = "github:project_item_changed:"

    # --- Key construction ---

    @staticmethod
    def _issue_key(prefix: str, repo: str, number: int) -> str:
        return f"{prefix}{repo}:{number}"

    @staticmethod
    def issue_opened_key(repo: str, number: int) -> str:
        return GitHubEventProtocol._issue_key(
            GitHubEventProtocol._ISSUE_OPENED, repo, number,
        )

    @staticmethod
    def issue_commented_key(repo: str, number: int) -> str:
        return GitHubEventProtocol._issue_key(
            GitHubEventProtocol._ISSUE_COMMENTED, repo, number,
        )

    @staticmethod
    def issue_closed_key(repo: str, number: int) -> str:
        return GitHubEventProtocol._issue_key(
            GitHubEventProtocol._ISSUE_CLOSED, repo, number,
        )

    @staticmethod
    def pr_opened_key(repo: str, number: int) -> str:
        return GitHubEventProtocol._issue_key(
            GitHubEventProtocol._PR_OPENED, repo, number,
        )

    @staticmethod
    def pr_review_requested_key(repo: str, number: int) -> str:
        return GitHubEventProtocol._issue_key(
            GitHubEventProtocol._PR_REVIEW_REQUESTED, repo, number,
        )

    @staticmethod
    def pr_merged_key(repo: str, number: int) -> str:
        return GitHubEventProtocol._issue_key(
            GitHubEventProtocol._PR_MERGED, repo, number,
        )

    @staticmethod
    def project_item_key(project_id: str, item_id: str) -> str:
        return (
            f"{GitHubEventProtocol._PROJECT_ITEM}"
            f"{project_id}:{item_id}"
        )

    # --- Pattern construction ---

    @staticmethod
    def issue_opened_pattern() -> str:
        return f"{GitHubEventProtocol._ISSUE_OPENED}*"

    @staticmethod
    def issue_commented_pattern() -> str:
        return f"{GitHubEventProtocol._ISSUE_COMMENTED}*"

    @staticmethod
    def pr_opened_pattern() -> str:
        return f"{GitHubEventProtocol._PR_OPENED}*"

    @staticmethod
    def project_item_pattern() -> str:
        return f"{GitHubEventProtocol._PROJECT_ITEM}*"

    # --- Key parsing ---

    @staticmethod
    def _parse_issue(prefix: str, key: str) -> tuple[str, int]:
        if not key.startswith(prefix):
            raise ValueError(f"Not a GitHubEventProtocol key: {key!r}")
        tail = key[len(prefix):]
        # Repo names contain ``/`` (owner/repo); the number is the
        # final colon-separated segment. ``rsplit`` once isolates it
        # regardless of the repo's own colon count (never in practice).
        parts = tail.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed GitHubEventProtocol key: {key!r}")
        try:
            number = int(parts[1])
        except ValueError as e:
            raise ValueError(f"Malformed issue number in {key!r}") from e
        return parts[0], number

    @staticmethod
    def parse_issue_opened_key(key: str) -> tuple[str, int]:
        return GitHubEventProtocol._parse_issue(
            GitHubEventProtocol._ISSUE_OPENED, key,
        )

    @staticmethod
    def parse_issue_commented_key(key: str) -> tuple[str, int]:
        return GitHubEventProtocol._parse_issue(
            GitHubEventProtocol._ISSUE_COMMENTED, key,
        )

    @staticmethod
    def parse_pr_opened_key(key: str) -> tuple[str, int]:
        return GitHubEventProtocol._parse_issue(
            GitHubEventProtocol._PR_OPENED, key,
        )

    @staticmethod
    def parse_project_item_key(key: str) -> tuple[str, str]:
        if not key.startswith(GitHubEventProtocol._PROJECT_ITEM):
            raise ValueError(f"Not a GitHubEventProtocol key: {key!r}")
        tail = key[len(GitHubEventProtocol._PROJECT_ITEM):]
        parts = tail.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed GitHubEventProtocol key: {key!r}")
        return parts[0], parts[1]


class ActionPolicyLifecycleProtocol(BlackboardProtocol):
    """Generic lifecycle events emitted by every ``BaseActionPolicy``.

    The policy publishes:

    - ``policy:action_started:{action_id}`` — the policy is about to
      dispatch an action through its dispatcher. The payload carries
      the action key and any caller-supplied parameters.
    - ``policy:action_completed:{action_id}`` — the action returned.
      Payload carries success/error, wall time, and the action key.
    - ``policy:codegen_retry:{ts}`` — code generation produced
      invalid output. The policy is about to re-prompt the LLM with
      accumulated error feedback.
    - ``policy:codegen_failed:{ts}`` — code generation exhausted its
      retry budget. The user-visible meaning is "I gave up on the
      current request"; the agent is ready for the next prompt.

    These events are the **only** way the policy talks to the outside
    world about its progress. The policy does NOT know what a chat is,
    a UI is, or a tracing system is. Subscribers — capabilities,
    dashboards, log adapters — decide what to do with the events.

    Operates on the agent's **primary** blackboard scope so any
    ``@event_handler`` on a capability of that agent receives them
    via the standard event broadcast in ``EventDrivenActionPolicy``.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    _ACTION_STARTED = "policy:action_started:"
    _ACTION_COMPLETED = "policy:action_completed:"
    _CODEGEN_RETRY = "policy:codegen_retry:"
    _CODEGEN_FAILED = "policy:codegen_failed:"

    # --- Key construction ---

    @staticmethod
    def action_started_key(action_id: str) -> str:
        return f"{ActionPolicyLifecycleProtocol._ACTION_STARTED}{action_id}"

    @staticmethod
    def action_completed_key(action_id: str) -> str:
        return f"{ActionPolicyLifecycleProtocol._ACTION_COMPLETED}{action_id}"

    @staticmethod
    def codegen_retry_key(timestamp_ms: int) -> str:
        return f"{ActionPolicyLifecycleProtocol._CODEGEN_RETRY}{timestamp_ms}"

    @staticmethod
    def codegen_failed_key(timestamp_ms: int) -> str:
        return f"{ActionPolicyLifecycleProtocol._CODEGEN_FAILED}{timestamp_ms}"

    # --- Pattern construction ---

    @staticmethod
    def action_started_pattern() -> str:
        return f"{ActionPolicyLifecycleProtocol._ACTION_STARTED}*"

    @staticmethod
    def action_completed_pattern() -> str:
        return f"{ActionPolicyLifecycleProtocol._ACTION_COMPLETED}*"

    @staticmethod
    def codegen_retry_pattern() -> str:
        return f"{ActionPolicyLifecycleProtocol._CODEGEN_RETRY}*"

    @staticmethod
    def codegen_failed_pattern() -> str:
        return f"{ActionPolicyLifecycleProtocol._CODEGEN_FAILED}*"

    @staticmethod
    def all_pattern() -> str:
        """Match every lifecycle event."""
        return "policy:*"

    # --- Key parsing ---

    @staticmethod
    def parse_action_started_key(key: str) -> str:
        prefix = ActionPolicyLifecycleProtocol._ACTION_STARTED
        if not key.startswith(prefix):
            raise ValueError(f"Not an action_started key: {key!r}")
        return key[len(prefix):]

    @staticmethod
    def parse_action_completed_key(key: str) -> str:
        prefix = ActionPolicyLifecycleProtocol._ACTION_COMPLETED
        if not key.startswith(prefix):
            raise ValueError(f"Not an action_completed key: {key!r}")
        return key[len(prefix):]

    @staticmethod
    def is_codegen_retry_key(key: str) -> bool:
        return key.startswith(ActionPolicyLifecycleProtocol._CODEGEN_RETRY)

    @staticmethod
    def is_codegen_failed_key(key: str) -> bool:
        return key.startswith(ActionPolicyLifecycleProtocol._CODEGEN_FAILED)


# ---------------------------------------------------------------------------
# Consciousness-stream cross-agent event protocols
# (added for the PR-Sub-2b recovery — see
# ``cps/CONSCIOUSNESS_STREAMS_RECOVERY_AUDIT.md``)
# ---------------------------------------------------------------------------


class VCMPageEventProtocol(BlackboardProtocol):
    """Protocol for VCM page-graph mutations.

    ``VirtualContextManager._on_page_loaded`` /
    ``_on_page_evicted`` publish typed page-event records to the
    colony-scoped blackboard so any agent's stream source can
    consume them via ``@event_handler(pattern=VCMPageEventProtocol.event_pattern())``.

    Key format::

        vcm_page_event:{mutation_kind}:{page_id}:{millis}

    ``mutation_kind`` is one of ``added`` / ``evicted``.
    ``millis`` is included so re-emissions of the same page don't
    collide on the blackboard's key.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    _PREFIX: ClassVar[str] = "vcm_page_event:"

    # --- Key construction ---

    @staticmethod
    def event_key(mutation_kind: str, page_id: str, millis: int) -> str:
        # Sanitise components — page_id may legitimately contain ``:``
        # in some sources; replace with ``/`` so the key parser sees
        # exactly four colon-separated fields.
        safe_page = str(page_id).replace(":", "/")
        return f"{VCMPageEventProtocol._PREFIX}{mutation_kind}:{safe_page}:{millis}"

    # --- Pattern construction ---

    @staticmethod
    def event_pattern() -> str:
        return f"{VCMPageEventProtocol._PREFIX}*"

    @staticmethod
    def event_pattern_for_kind(mutation_kind: str) -> str:
        return f"{VCMPageEventProtocol._PREFIX}{mutation_kind}:*"

    # --- Key parsing ---

    @staticmethod
    def parse_event_key(key: str) -> dict[str, str]:
        """Return ``{"mutation_kind": ..., "page_id": ..., "millis": ...}``.

        Raises ``ValueError`` for malformed keys."""
        if not key.startswith(VCMPageEventProtocol._PREFIX):
            raise ValueError(f"Not a VCMPageEventProtocol key: {key!r}")
        rest = key[len(VCMPageEventProtocol._PREFIX):]
        parts = rest.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed VCMPageEvent key: {key!r}")
        kind_and_page, millis = parts
        sub = kind_and_page.split(":", 1)
        if len(sub) != 2:
            raise ValueError(f"Malformed VCMPageEvent key: {key!r}")
        return {
            "mutation_kind": sub[0],
            "page_id": sub[1].replace("/", ":"),
            "millis": millis,
        }


class MonorepoCommitProtocol(BlackboardProtocol):
    """Protocol for tier-2 design-monorepo commits.

    Every ``BranchScopedCapabilityBase`` subclass's tier-2 action
    (``checkpoint_*_to_repo``) calls ``await self.fire_post_commit(...)``
    after a successful ``DesignMonorepoClient.commit_with_identity``.
    ``fire_post_commit`` publishes a typed commit record to the
    colony-scoped blackboard so any agent's stream source can
    consume them via
    ``@event_handler(pattern=MonorepoCommitProtocol.event_pattern())``.

    Key format::

        monorepo_commit:{branch_safe}:{sha}

    ``branch_safe`` is the branch with ``/`` replaced by ``__`` so
    branch names like ``fork/experiment`` don't collide with the
    blackboard's ``:`` key separator semantics.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    _PREFIX: ClassVar[str] = "monorepo_commit:"

    # --- Key construction ---

    @staticmethod
    def event_key(branch: str, sha: str) -> str:
        branch_safe = str(branch).replace("/", "__")
        return f"{MonorepoCommitProtocol._PREFIX}{branch_safe}:{sha}"

    # --- Pattern construction ---

    @staticmethod
    def event_pattern() -> str:
        return f"{MonorepoCommitProtocol._PREFIX}*"

    @staticmethod
    def event_pattern_for_branch(branch: str) -> str:
        branch_safe = str(branch).replace("/", "__")
        return f"{MonorepoCommitProtocol._PREFIX}{branch_safe}:*"

    # --- Key parsing ---

    @staticmethod
    def parse_event_key(key: str) -> dict[str, str]:
        """Return ``{"branch": ..., "sha": ...}``."""
        if not key.startswith(MonorepoCommitProtocol._PREFIX):
            raise ValueError(f"Not a MonorepoCommit key: {key!r}")
        rest = key[len(MonorepoCommitProtocol._PREFIX):]
        parts = rest.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed MonorepoCommit key: {key!r}")
        branch_safe, sha = parts
        return {"branch": branch_safe.replace("__", "/"), "sha": sha}


class DesignContextMappedProtocol(BlackboardProtocol):
    """Protocol fired when the design-context materialiser finishes
    mapping a ``design_context_sources`` row through one of its
    ingestion paths.

    Three ingestion paths defined by the top-level design plan
    (``colony_docs/markdown/plans/design_top_level_design_process.md``
    §5): ``"vcm"`` (chunked pages, Phase 1), ``"kuzu"`` (LLM
    claim-extraction into the knowledge graph, Phase 3), and raw
    ``read_file`` (no event — agents read files directly). One event
    per (source_name, path) tuple. Subscribers (the system-design
    capability, the dashboard) react to know when context is fresh
    enough to query.

    Body fields the writer SHOULD include in the blackboard value
    payload (the protocol enforces only the key shape):

    - ``source_name`` (str): the ``design_context_sources`` row name
    - ``path`` (str): ``"vcm"`` | ``"kuzu"``
    - ``page_scope_id`` (str, vcm path only): scope id passed to
      ``mmap_application_scope``
    - ``num_files`` (int): file count matched by the row's globs
    - ``num_claims`` (int, kuzu path only): claims extracted into the KG
    - ``pinned`` (bool, vcm path only): whether pages were lock_page'd
    - ``materialized_at`` (float): unix seconds

    Key format::

        design_context_mapped:{source_name}:{path}:{millis}

    ``millis`` lets multiple re-materialisations of the same row co-exist
    on the blackboard with a natural ordering.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    _PREFIX: ClassVar[str] = "design_context_mapped:"

    # --- Key construction ---

    @staticmethod
    def event_key(source_name: str, path: str, millis: int) -> str:
        # source_name from the operator's repo_map.yaml is a free-form
        # identifier — guard against ``:`` collisions with the key
        # separator the same way other protocols do.
        safe_name = str(source_name).replace(":", "/")
        return (
            f"{DesignContextMappedProtocol._PREFIX}"
            f"{safe_name}:{path}:{millis}"
        )

    # --- Pattern construction ---

    @staticmethod
    def event_pattern() -> str:
        return f"{DesignContextMappedProtocol._PREFIX}*"

    @staticmethod
    def event_pattern_for_source(source_name: str) -> str:
        safe_name = str(source_name).replace(":", "/")
        return (
            f"{DesignContextMappedProtocol._PREFIX}"
            f"{safe_name}:*"
        )

    @staticmethod
    def event_pattern_for_path(path: str) -> str:
        return f"{DesignContextMappedProtocol._PREFIX}*:{path}:*"

    # --- Key parsing ---

    @staticmethod
    def parse_event_key(key: str) -> dict[str, str]:
        """Return ``{"source_name": ..., "path": ..., "millis": ...}``."""
        if not key.startswith(DesignContextMappedProtocol._PREFIX):
            raise ValueError(
                f"Not a DesignContextMapped key: {key!r}",
            )
        rest = key[len(DesignContextMappedProtocol._PREFIX):]
        # Right-anchored split: the trailing ``:millis`` and ``:path``
        # are well-formed; everything earlier belongs to ``source_name``
        # (which may contain ``/`` placeholders we restore below).
        parts = rest.rsplit(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Malformed DesignContextMapped key: {key!r}",
            )
        safe_name, path, millis = parts
        return {
            "source_name": safe_name.replace("/", ":"),
            "path": path,
            "millis": millis,
        }


class DesignInconsistencyProtocol(BlackboardProtocol):
    """Protocol fired when :meth:`SystemDesignCapability.find_inconsistencies`
    surfaces a claim-shaped finding in the design-context knowledge
    graph — either an explicit contradiction (claim with predicate
    ``contradicts`` / ``conflicts_with`` / ``is_incompatible_with``)
    or a hit from an operator-authored ``consistency_rule`` claim
    that the rule engine evaluates as firing.

    Phase P3c emits the contradiction-claim variant only — operator-
    authored rules require richer claim types than the deterministic
    extractor produces and become useful once Phase P3d's
    LLMClaimExtractor lands.

    Body fields the writer SHOULD include in the blackboard value
    payload (the protocol enforces only the key shape):

    - ``kind`` (str): ``contradiction`` | ``rule_finding`` | ``orphan``
    - ``subject`` / ``predicate`` / ``object`` (str): the source claim
    - ``citation_uri`` (str): ``design_context://<source>/<rel>``
    - ``source_name`` (str): the design_context_sources row name
    - ``file`` (str): the file path inside the source corpus
    - ``confidence`` (float)
    - ``rule_id`` (str, optional): when ``kind='rule_finding'``
    - ``detected_at`` (float): unix seconds

    Key format::

        design_inconsistency:{source_name}:{kind}:{millis}
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    _PREFIX: ClassVar[str] = "design_inconsistency:"

    @staticmethod
    def event_key(source_name: str, kind: str, millis: int) -> str:
        safe_name = str(source_name).replace(":", "/")
        safe_kind = str(kind).replace(":", "/")
        return (
            f"{DesignInconsistencyProtocol._PREFIX}"
            f"{safe_name}:{safe_kind}:{millis}"
        )

    @staticmethod
    def event_pattern() -> str:
        return f"{DesignInconsistencyProtocol._PREFIX}*"

    @staticmethod
    def event_pattern_for_source(source_name: str) -> str:
        safe_name = str(source_name).replace(":", "/")
        return f"{DesignInconsistencyProtocol._PREFIX}{safe_name}:*"

    @staticmethod
    def event_pattern_for_kind(kind: str) -> str:
        safe_kind = str(kind).replace(":", "/")
        return f"{DesignInconsistencyProtocol._PREFIX}*:{safe_kind}:*"

    @staticmethod
    def parse_event_key(key: str) -> dict[str, str]:
        if not key.startswith(DesignInconsistencyProtocol._PREFIX):
            raise ValueError(
                f"Not a DesignInconsistency key: {key!r}",
            )
        rest = key[len(DesignInconsistencyProtocol._PREFIX):]
        parts = rest.rsplit(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Malformed DesignInconsistency key: {key!r}",
            )
        safe_name, safe_kind, millis = parts
        return {
            "source_name": safe_name.replace("/", ":"),
            "kind": safe_kind.replace("/", ":"),
            "millis": millis,
        }


class BottleneckDetectedProtocol(BlackboardProtocol):
    """Protocol fired when :meth:`DesignProcessCapability.identify_bottlenecks`
    surfaces a workflow-state finding — an open issue stalled past
    the configured no-activity threshold, an open issue with a
    deep blocking chain, an under-assigned milestone past its
    halfway point, etc.

    Phase P5a ships the ``stalled_issue`` kind (built-in heuristic
    over ``GitHubCapability.list_issues``); richer kinds — including
    operator-authored ``bottleneck_rule``-claim-driven findings —
    surface once Phase P3d's :class:`LLMClaimExtractor` produces
    those typed claims from operator markdown.

    Body fields the writer SHOULD include in the blackboard value
    payload (the protocol enforces only the key shape):

    - ``kind`` (str): ``stalled_issue`` | ``rule_finding`` | …
    - ``severity`` (str): ``low`` | ``medium`` | ``high``
    - ``repo`` (str): ``owner/name``
    - ``issue_number`` (int): the affected issue
    - ``url`` (str): ``html_url``
    - ``summary`` (str): one-line description for the dashboard
    - ``suggested_remedies`` (list[str]): planner-facing hints
    - ``rule_id`` (str, optional): when ``kind='rule_finding'``
    - ``detected_at`` (float): unix seconds

    Key format::

        bottleneck_detected:{repo_safe}:{kind}:{millis}

    ``repo_safe`` is the repo with ``/`` replaced by ``__`` so
    repo names like ``polymathera/cps`` don't collide with the
    key separator.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    _PREFIX: ClassVar[str] = "bottleneck_detected:"

    @staticmethod
    def event_key(repo: str, kind: str, millis: int) -> str:
        repo_safe = str(repo).replace("/", "__")
        safe_kind = str(kind).replace(":", "/")
        return (
            f"{BottleneckDetectedProtocol._PREFIX}"
            f"{repo_safe}:{safe_kind}:{millis}"
        )

    @staticmethod
    def event_pattern() -> str:
        return f"{BottleneckDetectedProtocol._PREFIX}*"

    @staticmethod
    def event_pattern_for_repo(repo: str) -> str:
        repo_safe = str(repo).replace("/", "__")
        return f"{BottleneckDetectedProtocol._PREFIX}{repo_safe}:*"

    @staticmethod
    def event_pattern_for_kind(kind: str) -> str:
        safe_kind = str(kind).replace(":", "/")
        return f"{BottleneckDetectedProtocol._PREFIX}*:{safe_kind}:*"

    @staticmethod
    def parse_event_key(key: str) -> dict[str, str]:
        if not key.startswith(BottleneckDetectedProtocol._PREFIX):
            raise ValueError(
                f"Not a BottleneckDetected key: {key!r}",
            )
        rest = key[len(BottleneckDetectedProtocol._PREFIX):]
        parts = rest.rsplit(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Malformed BottleneckDetected key: {key!r}",
            )
        repo_safe, safe_kind, millis = parts
        return {
            "repo": repo_safe.replace("__", "/"),
            "kind": safe_kind.replace("/", ":"),
            "millis": millis,
        }


class RoadmapSyncProtocol(BlackboardProtocol):
    """Protocol fired when :meth:`DesignProcessCapability.sync_roadmap_with_github`
    completes — one event per sync run carrying the diff that was
    applied (or proposed for user review, depending on the sync
    mode).

    Phase P5c ships the sync action; this protocol lands earlier
    (P5a) so subscribers (the Colony Status panel, the operator
    notification stream) can register their handlers ahead of the
    first sync run.

    Body fields the writer SHOULD include:

    - ``direction`` (str): ``bidirectional`` | ``roadmap_to_github``
      | ``github_to_roadmap``
    - ``diff`` (list[dict]): per-change records the planner /
      dashboard can render
    - ``conflict_count`` (int): how many entries needed user mediation
    - ``ran_at`` (float): unix seconds

    Key format::

        roadmap_sync:{repo_safe}:{direction}:{millis}
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    _PREFIX: ClassVar[str] = "roadmap_sync:"

    @staticmethod
    def event_key(repo: str, direction: str, millis: int) -> str:
        repo_safe = str(repo).replace("/", "__")
        safe_dir = str(direction).replace(":", "/")
        return (
            f"{RoadmapSyncProtocol._PREFIX}"
            f"{repo_safe}:{safe_dir}:{millis}"
        )

    @staticmethod
    def event_pattern() -> str:
        return f"{RoadmapSyncProtocol._PREFIX}*"

    @staticmethod
    def event_pattern_for_repo(repo: str) -> str:
        repo_safe = str(repo).replace("/", "__")
        return f"{RoadmapSyncProtocol._PREFIX}{repo_safe}:*"

    @staticmethod
    def parse_event_key(key: str) -> dict[str, str]:
        if not key.startswith(RoadmapSyncProtocol._PREFIX):
            raise ValueError(
                f"Not a RoadmapSync key: {key!r}",
            )
        rest = key[len(RoadmapSyncProtocol._PREFIX):]
        parts = rest.rsplit(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Malformed RoadmapSync key: {key!r}",
            )
        repo_safe, safe_dir, millis = parts
        return {
            "repo": repo_safe.replace("__", "/"),
            "direction": safe_dir.replace("/", ":"),
            "millis": millis,
        }


class DesignSuggestionProtocol(BlackboardProtocol):
    """Protocol fired when :class:`SystemDesignCapability` surfaces a
    suggestion the planner should consider — an unverified hypothesis
    (Phase P3c via :meth:`audit_hypothesis_coverage`), a proposed
    design alternative (Phase P3d+ via ``propose_alternatives``), a
    missing literature reference, or a tool worth introducing.

    Body fields the writer SHOULD include in the blackboard value
    payload (the protocol enforces only the key shape):

    - ``kind`` (str): ``hypothesis_orphan`` | ``alternative`` |
      ``literature`` | ``tool``
    - ``target_claim_type`` (str): which extracted claim type this
      suggestion bears on (e.g. ``hypothesis``)
    - ``summary`` (str): one-line description
    - ``evidence`` (list[dict]): citations supporting the suggestion
      (each ``{citation_uri, snippet}``)
    - ``confidence`` (float)
    - ``detected_at`` (float): unix seconds

    Key format::

        design_suggestion:{source_name}:{kind}:{millis}
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    _PREFIX: ClassVar[str] = "design_suggestion:"

    @staticmethod
    def event_key(source_name: str, kind: str, millis: int) -> str:
        safe_name = str(source_name).replace(":", "/")
        safe_kind = str(kind).replace(":", "/")
        return (
            f"{DesignSuggestionProtocol._PREFIX}"
            f"{safe_name}:{safe_kind}:{millis}"
        )

    @staticmethod
    def event_pattern() -> str:
        return f"{DesignSuggestionProtocol._PREFIX}*"

    @staticmethod
    def event_pattern_for_source(source_name: str) -> str:
        safe_name = str(source_name).replace(":", "/")
        return f"{DesignSuggestionProtocol._PREFIX}{safe_name}:*"

    @staticmethod
    def event_pattern_for_kind(kind: str) -> str:
        safe_kind = str(kind).replace(":", "/")
        return f"{DesignSuggestionProtocol._PREFIX}*:{safe_kind}:*"

    @staticmethod
    def parse_event_key(key: str) -> dict[str, str]:
        if not key.startswith(DesignSuggestionProtocol._PREFIX):
            raise ValueError(
                f"Not a DesignSuggestion key: {key!r}",
            )
        rest = key[len(DesignSuggestionProtocol._PREFIX):]
        parts = rest.rsplit(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Malformed DesignSuggestion key: {key!r}",
            )
        safe_name, safe_kind, millis = parts
        return {
            "source_name": safe_name.replace("/", ":"),
            "kind": safe_kind.replace("/", ":"),
            "millis": millis,
        }


class MentionEventProtocol(BlackboardProtocol):
    """Protocol fired when :class:`MentionRoutingCapability` detects an
    ``@colony`` / ``@polymath`` mention in an inbound GitHub event's
    body text (issue body, PR body, or issue comment).

    P10 v1 emits one event per matched mention. Downstream subscribers:

    - :class:`InteractionLogCapability` mirrors each ``mention:*`` write
      into ``interaction_log`` so mentions are queryable via
      ``fetch_recent_activity`` / ``fetch_by_ref``.
    - Future P10 follow-up: an LLM-driven handler on the system
      ``SessionAgent`` that judges intent + responds via
      ``GitHubCapability.comment_on_issue``.

    Body fields the writer SHOULD include in the value payload
    (the protocol enforces only the key shape):

    - ``mention_kind`` (str): the matched handle — ``colony`` /
      ``polymath`` / ``colony-<name>`` / ``polymath-<name>``.
    - ``repo`` (str): ``owner/repo``.
    - ``issue_number`` (int): the issue (or PR) the mention is on.
    - ``comment_id`` (int | None): when the mention is in a comment;
      ``None`` when it's in an issue/PR body.
    - ``commenter_login`` (str | None): the GitHub user who typed the
      mention.
    - ``body`` (str): the full body text the mention was found in (the
      LLM-judge follow-up consumes this for intent inference).
    - ``html_url`` (str | None): a deep-link back to the comment / issue
      so dashboards / responders can surface it.

    Key format::

        mention:{owner}__{repo}:{issue_number}:{comment_id_or_zero}

    The repo's slash is encoded as ``__`` (same convention as
    :class:`MonorepoCommitProtocol`) so the key has a fixed colon-
    count of three. ``comment_id_or_zero`` is ``0`` when the mention
    is in an issue/PR body (no comment id to surface) — keeps the key
    shape uniform so pattern matching doesn't have to branch.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    _PREFIX: ClassVar[str] = "mention:"

    @staticmethod
    def event_key(
        repo: str, issue_number: int, comment_id: int | None = None,
    ) -> str:
        safe_repo = str(repo).replace("/", "__")
        comment_token = comment_id if comment_id is not None else 0
        return (
            f"{MentionEventProtocol._PREFIX}"
            f"{safe_repo}:{int(issue_number)}:{int(comment_token)}"
        )

    @staticmethod
    def event_pattern() -> str:
        return f"{MentionEventProtocol._PREFIX}*"

    @staticmethod
    def event_pattern_for_repo(repo: str) -> str:
        safe_repo = str(repo).replace("/", "__")
        return f"{MentionEventProtocol._PREFIX}{safe_repo}:*"

    @staticmethod
    def parse_event_key(key: str) -> dict[str, str]:
        if not key.startswith(MentionEventProtocol._PREFIX):
            raise ValueError(f"Not a MentionEvent key: {key!r}")
        rest = key[len(MentionEventProtocol._PREFIX):]
        parts = rest.split(":")
        if len(parts) != 3:
            raise ValueError(f"Malformed MentionEvent key: {key!r}")
        safe_repo, issue_number, comment_id = parts
        return {
            "repo": safe_repo.replace("__", "/"),
            "issue_number": issue_number,
            "comment_id": comment_id,
        }


class AgentDiagnosticProtocol(BlackboardProtocol):
    """Typed events for cross-agent visibility of internal failure
    patterns (guardrail block streaks, LLM failure streaks, polling
    timeouts, etc.). Producers: action policies, capabilities.
    Consumers: parents, observers via ``@event_handler``.

    Scope: SESSION — so a parent agent (SessionAgent) can subscribe
    once and see diagnostic events from every child coordinator
    spawned into the same session. The key carries ``agent_id`` so
    consumers can filter by producer.

    Key shape: ``agent:diagnostic:<agent_id>:<kind>:<sequence>``.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.SESSION

    _PREFIX = "agent:diagnostic:"

    @staticmethod
    def event_key(agent_id: str, kind: str, sequence: int) -> str:
        return f"{AgentDiagnosticProtocol._PREFIX}{agent_id}:{kind}:{sequence}"

    @staticmethod
    def event_pattern(agent_id: str | None = None) -> str:
        return f"{AgentDiagnosticProtocol._PREFIX}{agent_id or '*'}:*:*"

    @staticmethod
    def parse_event_key(key: str) -> dict[str, str]:
        if not key.startswith(AgentDiagnosticProtocol._PREFIX):
            raise ValueError(
                f"Not an AgentDiagnostic key: {key!r}",
            )
        rest = key[len(AgentDiagnosticProtocol._PREFIX):]
        parts = rest.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Malformed AgentDiagnostic key: {key!r}",
            )
        agent_id, kind, sequence = parts
        return {
            "agent_id": agent_id,
            "kind": kind,
            "sequence": sequence,
        }


#: Canonical sub-namespace under the session blackboard for chat
#: traffic — used both by the producer side (capabilities that emit
#: chat-relevant events) and the consumer side (the web_ui chat
#: router that listens). Single source of truth so the relay and the
#: writers cannot drift. Lives in ``agents/`` so capabilities in
#: ``agents/`` can reference it without depending on ``web_ui/``.
CHAT_BLACKBOARD_NAMESPACE = "session_chat"


class MissionStatusProtocol(BlackboardProtocol):
    """Coordinator-authored narrative status for one running mission.

    Producers: :class:`MissionStatusCapability` (mounted on every
    mission coordinator). Consumers: the chat WebSocket relay in
    ``web_ui/backend/routers/chat.py`` (snapshot-read on reconnect +
    streamed via ``stream_events``).

    The key is a SINGLETON per ``mission_id`` — the latest write
    replaces the prior status; the protocol does NOT retain a history.
    The ``mission_id`` is the coordinator's own ``agent_id`` —
    framework-known, never threaded by the LLM. Lifetime is
    framework-managed: cleared on mission terminal state or on a new
    mission with the same ``mission_id``.

    Scope: SESSION — narrative is per-session, mirrors the chat UI's
    scope.

    Key shape: ``chat:mission_status:<mission_id>``.

    NOTE: This protocol lives in ``agents/blackboard`` (not in
    ``web_ui/``) so any consumer in either layer can import it without
    a backwards dependency. The chat-protocol ``SessionChatProtocol``
    re-exposes the same methods for chat-side ergonomics but they
    delegate to this canonical owner.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.SESSION

    _PREFIX = "chat:mission_status:"

    @staticmethod
    def status_key(mission_id: str) -> str:
        return f"{MissionStatusProtocol._PREFIX}{mission_id}"

    @staticmethod
    def status_pattern(mission_id: str | None = None) -> str:
        if mission_id:
            return f"{MissionStatusProtocol._PREFIX}{mission_id}"
        return f"{MissionStatusProtocol._PREFIX}*"

    @staticmethod
    def parse_status_key(key: str) -> str:
        if not key.startswith(MissionStatusProtocol._PREFIX):
            raise ValueError(
                f"Not a MissionStatusProtocol key: {key!r}",
            )
        return key[len(MissionStatusProtocol._PREFIX):]


# Diagnostic kinds — open enum. Add a constant per new kind so the
# producer/consumer contract stays grep-able. v1 ships one kind; the
# rest are documented for future producers.
DIAGNOSTIC_GUARDRAIL_BLOCK_STREAK = "guardrail_block_streak"
DIAGNOSTIC_EMPTY_ITERATION_STREAK = "empty_iteration_streak"
# Reserved for future producers:
# DIAGNOSTIC_CODE_VALIDATION_STREAK = "code_validation_streak"
# DIAGNOSTIC_LLM_CALL_FAILURE_STREAK = "llm_call_failure_streak"
# DIAGNOSTIC_POLLING_TIMEOUT = "polling_timeout"
# DIAGNOSTIC_BUDGET_THRESHOLD = "budget_threshold_crossed"

# Diagnostics that are RELEVANT to their own producer's planner context.
# Most diagnostics describe "what this agent did" (already in the trace);
# ``empty_iteration_streak`` describes "what this agent must do next"
# (call ``wait_for_next_event``), which is exactly the planner-context
# surface.
SELF_RELEVANT_DIAGNOSTIC_KINDS: frozenset[str] = frozenset({
    DIAGNOSTIC_EMPTY_ITERATION_STREAK,
})
