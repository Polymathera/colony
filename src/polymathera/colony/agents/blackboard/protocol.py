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
    key = AgentRunProtocol.request_key(request_id="req_abc123", namespace="compliance")
    await blackboard.write(key, payload)

    # Subscriber
    pattern = AgentRunProtocol.request_pattern(namespace="compliance")
    blackboard.stream_events_to_queue(queue, pattern=pattern)

    # Parser (in event handler)
    request_id = AgentRunProtocol.parse_request_key(event.key, namespace="compliance")

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

    # Check for scope_id leak — the key should never contain the
    # blackboard's own scope_id since that's the partition, not the key.
    if scope_id and scope_id in key:
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
    if scope_id and scope_id in pattern:
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

    All protocol methods accept a ``namespace`` parameter for
    disambiguation when multiple capabilities share the same scope.
    The namespace is prepended to keys:
    ``{namespace}:{key}`` for keys, ``{namespace}:*`` prefix for patterns.

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

    @staticmethod
    def _ns(namespace: str, key: str) -> str:
        """Prepend namespace to a key."""
        return f"{namespace}:{key}"

    @staticmethod
    def _ns_pattern(namespace: str, pattern: str) -> str:
        """Prepend namespace to a pattern."""
        return f"{namespace}:{pattern}"


# ---------------------------------------------------------------------------
# Concrete protocols
# ---------------------------------------------------------------------------

class AgentRunProtocol(BlackboardProtocol):
    """Protocol for ``AgentHandle.run()`` <-> child agent communication.

    All methods accept a ``namespace`` parameter to scope keys
    within a shared blackboard partition. This prevents interference when
    multiple capabilities share a colony-level scope.

    Key formats:

    - Without namespace: ``request:run:{request_id}`` (agent-scoped, only one agent)
    - With namespace: ``request:run:compliance:{request_id}`` (colony-scoped, disambiguated)

    Example::

        # Agent-scoped (namespace still required for key construction)
        key = AgentRunProtocol.request_key("req_abc", namespace="analysis")
        # -> "request:run:analysis:req_abc"

        # Colony-scoped (namespace prevents cross-capability interference)
        key = AgentRunProtocol.request_key("req_abc", namespace="compliance")
        # -> "request:run:compliance:req_abc"

        # Capability declares its namespace
        class ComplianceCapability(AgentCapability):
            input_patterns = [AgentRunProtocol.request_pattern(namespace="compliance")]
            # -> ["request:run:compliance:*"]

            @event_handler(pattern=AgentRunProtocol.request_pattern(namespace="compliance"))
            async def handle_request(self, event, repl):
                request_id = AgentRunProtocol.parse_request_key(event.key, namespace="compliance")
                ...
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    # --- Key construction ---

    @staticmethod
    def request_key(request_id: str, namespace: str) -> str:
        """Key for a request.

        Args:
            request_id: Unique request identifier.
            namespace: Namespace for disambiguation in shared scopes
                (e.g., ``"compliance"``, ``"contracts"``).
        """
        ns = f"{namespace}:"
        return f"request:run:{ns}{request_id}"

    @staticmethod
    def result_key(request_id: str, namespace: str) -> str:
        """Key for a result."""
        ns = f"{namespace}:"
        return f"result:run:{ns}{request_id}"

    @staticmethod
    def event_key(request_id: str, event_name: str, namespace: str) -> str:
        """Key for a streaming event."""
        ns = f"{namespace}:"
        return f"event:run:{ns}{request_id}:{event_name}"

    # --- Pattern construction ---

    @staticmethod
    def request_pattern(namespace: str) -> str:
        """Pattern matching run requests.

        Args:
            namespace: Namespace to match requests for.
        """
        ns = f"{namespace}:"
        return f"request:run:{ns}*"

    @staticmethod
    def result_pattern(namespace: str) -> str:
        """Pattern matching run results."""
        ns = f"{namespace}:"
        return f"result:run:{ns}*"

    @staticmethod
    def event_pattern(request_id: str, namespace: str) -> str:
        """Pattern matching streaming events for a specific request."""
        ns = f"{namespace}:"
        return f"event:run:{ns}{request_id}:*"

    # --- Key parsing ---

    @staticmethod
    def parse_request_key(key: str, namespace: str) -> str:
        """Extract request_id from a request key.

        Args:
            key: Key like ``"request:run:req_abc"`` or ``"request:run:compliance:req_abc"``
            namespace: Expected namespace (must match what was used to write).

        Returns:
            The request_id.
        """
        ns = f"{namespace}:"
        prefix = f"request:run:{ns}"
        if not key.startswith(prefix):
            raise ValueError(f"Not an AgentRunProtocol request key (namespace={namespace!r}): {key!r}")
        return key[len(prefix):]

    @staticmethod
    def parse_result_key(key: str, namespace: str) -> str:
        """Extract request_id from a result key."""
        ns = f"{namespace}:"
        prefix = f"result:run:{ns}"
        if not key.startswith(prefix):
            raise ValueError(f"Not an AgentRunProtocol result key (namespace={namespace!r}): {key!r}")
        return key[len(prefix):]

    @staticmethod
    def parse_event_key(key: str, namespace: str) -> tuple[str, str]:
        """Extract (request_id, event_name) from an event key."""
        ns = f"{namespace}:"
        prefix = f"event:run:{ns}"
        if not key.startswith(prefix):
            raise ValueError(f"Not an AgentRunProtocol event key (namespace={namespace!r}): {key!r}")
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
        key = WorkAssignmentProtocol.assignment_key(agent_id=worker_id, request_id=req_id, namespace="pool")
        await colony_blackboard.write(key, work_unit)

        # Worker writes result
        key = WorkAssignmentProtocol.result_key(agent_id=self.agent.agent_id, namespace="pool", result_type="final")
        await colony_blackboard.write(key, result)
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    # --- Key construction ---

    @staticmethod
    def assignment_key(agent_id: str, request_id: str, namespace: str) -> str:
        """Key for a work assignment targeting a specific worker."""
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(agent_id=agent_id, work_assignment=request_id))

    @staticmethod
    def result_key(agent_id: str, namespace: str, result_type: str = "final") -> str:
        """Key for a worker's result."""
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(agent_id=agent_id, result_type=result_type))

    @staticmethod
    def broadcast_key(agent_id: str, namespace: str) -> str:
        """Key for a broadcast to a specific worker."""
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(agent_id=agent_id, broadcast=True))

    # --- Pattern construction ---

    @staticmethod
    def assignment_pattern(namespace: str, agent_id: str | None = None) -> str:
        """Pattern matching work assignments (optionally for a specific worker)."""
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(agent_id=agent_id, work_assignment=None))

    @staticmethod
    def result_pattern(namespace: str, agent_id: str | None = None) -> str:
        """Pattern matching worker results (optionally from a specific worker)."""
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(agent_id=agent_id, result_type=None))

    @staticmethod
    def broadcast_pattern(namespace: str, agent_id: str | None = None) -> str:
        """Pattern matching broadcasts."""
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(agent_id=agent_id, broadcast=None))

    # --- Key parsing ---

    @staticmethod
    def parse_assignment_key(key: str, namespace: str) -> dict[str, str]:
        """Extract fields from an assignment key.

        Returns:
            Dict with ``agent_id`` and ``work_assignment`` fields.
        """
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
        return ScopeUtils.parse_key("", key)

    @staticmethod
    def parse_result_key(key: str, namespace: str) -> dict[str, str]:
        """Extract fields from a result key.

        Returns:
            Dict with ``agent_id`` and ``result_type`` fields.
        """
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
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
    def created_key(agent_id: str, namespace: str) -> str:
        """Key for an agent creation signal."""
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(scope="agent_created", agent_id=agent_id))

    @staticmethod
    def terminated_key(agent_id: str, namespace: str) -> str:
        """Key for an agent termination signal."""
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(scope="agent_terminated", agent_id=agent_id))

    # --- Pattern construction ---

    @staticmethod
    def created_pattern(namespace: str) -> str:
        """Pattern matching all agent creation signals."""
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(scope="agent_created", agent_id=None))

    @staticmethod
    def terminated_pattern(namespace: str) -> str:
        """Pattern matching all agent termination signals."""
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(scope="agent_terminated", agent_id=None))

    # --- Key parsing ---

    @staticmethod
    def parse_key(key: str, namespace: str) -> dict[str, str]:
        """Extract fields from a lifecycle signal key.

        Returns:
            Dict with ``agent_id`` and ``scope`` (event type) fields.
        """
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
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
    def state_key(game_id: str, namespace: str) -> str:
        """Key for the canonical game state."""
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(state=game_id))

    @staticmethod
    def result_key(namespace: str) -> str:
        """Key for the game result."""
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(result="game_complete"))

    # --- Pattern construction ---

    @staticmethod
    def state_pattern(namespace: str) -> str:
        """Pattern matching game state updates."""
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(state=None))

    # --- Key parsing ---

    @staticmethod
    def parse_state_key(key: str, namespace: str) -> str:
        """Extract game_id from a state key.

        Returns:
            The game_id.
        """
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("state", "")


class CritiqueProtocol(BlackboardProtocol):
    """Protocol for critique request/response exchange.

    Colony-scoped with namespace suffix (e.g., ``{colony_scope}:critique``).
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
    def peer_request_key(requester_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(requester_id=requester_id, critique_request_from_peer=True))

    @staticmethod
    def parent_to_child_request_key(child_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(child_id=child_id, critique_request_from_parent=True))

    @staticmethod
    def child_to_parent_request_key(parent_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(parent_id=parent_id, critique_request_from_child=True))

    @staticmethod
    def response_key(requester_id: str, responder_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(requester_id=requester_id, critique_response_from=responder_id))

    # --- Pattern construction ---

    @staticmethod
    def peer_request_pattern(namespace: str) -> str:
        """Pattern matching peer critique requests."""
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(requester_id=None, critique_request_from_peer=True))

    @staticmethod
    def parent_to_child_request_pattern(namespace: str) -> str:
        """Pattern matching parent-to-child critique requests."""
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(child_id=None, critique_request_from_parent=True))

    @staticmethod
    def child_to_parent_request_pattern(namespace: str) -> str:
        """Pattern matching child-to-parent critique requests."""
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(parent_id=None, critique_request_from_child=True))

    @staticmethod
    def all_requests_pattern(namespace: str) -> str:
        """Pattern matching all critique requests."""
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key())

    @staticmethod
    def all_request_patterns(namespace: str) -> list[str]:
        """All three request patterns (peer, parent-to-child, child-to-parent).

        Use this when subscribing to all critique requests regardless of direction.
        """
        return [
            CritiqueProtocol.peer_request_pattern(namespace=namespace),
            CritiqueProtocol.parent_to_child_request_pattern(namespace=namespace),
            CritiqueProtocol.child_to_parent_request_pattern(namespace=namespace),
        ]

    @staticmethod
    def response_pattern(namespace: str, requester_id: str | None = None) -> str:
        """Pattern matching critique responses (optionally for a specific requester)."""
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(requester_id=requester_id, critique_response_from=None))

    # --- Key parsing ---

    @staticmethod
    def parse_key(key: str, namespace: str) -> dict[str, str]:
        """Extract fields from any critique key.

        Returns:
            Dict of field names to values (e.g., ``{"requester_id": "...", "critique_request_from_peer": "True"}``).
        """
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
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
    def cluster_state_key(namespace: str) -> str:
        """Key for cluster-wide working set state."""
        return BlackboardProtocol._ns(namespace, f"{WorkingSetStateProtocol._PREFIX}cluster")

    @staticmethod
    def page_status_key(namespace: str) -> str:
        """Key for per-page status."""
        return BlackboardProtocol._ns(namespace, f"{WorkingSetStateProtocol._PREFIX}page_status")

    @staticmethod
    def state_pattern(namespace: str) -> str:
        """Pattern matching all working set state updates."""
        return BlackboardProtocol._ns_pattern(namespace, f"{WorkingSetStateProtocol._PREFIX}*")


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
    def request_key(request_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, f"{ConsistencyCheckProtocol._REQUEST_PREFIX}{request_id}")

    @staticmethod
    def result_key(request_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, f"{ConsistencyCheckProtocol._RESULT_PREFIX}{request_id}")

    # --- Pattern construction ---

    @staticmethod
    def request_pattern(namespace: str) -> str:
        return BlackboardProtocol._ns_pattern(namespace, f"{ConsistencyCheckProtocol._REQUEST_PREFIX}*")

    @staticmethod
    def result_pattern(namespace: str) -> str:
        return BlackboardProtocol._ns_pattern(namespace, f"{ConsistencyCheckProtocol._RESULT_PREFIX}*")

    # --- Key parsing ---

    @staticmethod
    def parse_request_key(key: str, namespace: str) -> str:
        """Extract request_id from a request key."""
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
        if not key.startswith(ConsistencyCheckProtocol._REQUEST_PREFIX):
            raise ValueError(f"Not a ConsistencyCheckProtocol request key: {key!r}")
        return key[len(ConsistencyCheckProtocol._REQUEST_PREFIX):]

    @staticmethod
    def parse_result_key(key: str, namespace: str) -> str:
        """Extract request_id from a result key."""
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
        if not key.startswith(ConsistencyCheckProtocol._RESULT_PREFIX):
            raise ValueError(f"Not a ConsistencyCheckProtocol result key: {key!r}")
        return key[len(ConsistencyCheckProtocol._RESULT_PREFIX):]


class GroundingProtocol(BlackboardProtocol):
    """Protocol for grounding requests/results.

    Agent-scoped. ``request_id`` provides per-request uniqueness.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def request_key(request_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(grounding_request=request_id))

    @staticmethod
    def result_key(request_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(grounding_result=request_id))

    @staticmethod
    def request_pattern(namespace: str) -> str:
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(grounding_request=None))

    @staticmethod
    def result_pattern(namespace: str) -> str:
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(grounding_result=None))

    @staticmethod
    def parse_request_key(key: str, namespace: str) -> str:
        """Extract request_id from a grounding request key."""
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("grounding_request", "")

    @staticmethod
    def parse_result_key(key: str, namespace: str) -> str:
        """Extract request_id from a grounding result key."""
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("grounding_result", "")


class GoalAlignmentProtocol(BlackboardProtocol):
    """Protocol for goal alignment requests and joint goal registrations.

    Colony-scoped.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    @staticmethod
    def request_key(request_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(goal_alignment_request=request_id))

    @staticmethod
    def joint_goal_key(goal_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(joint_goal_registration=goal_id))

    @staticmethod
    def request_pattern(namespace: str) -> str:
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(goal_alignment_request=None))

    @staticmethod
    def joint_goal_pattern(namespace: str) -> str:
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(joint_goal_registration=None))

    @staticmethod
    def parse_request_key(key: str, namespace: str) -> str:
        """Extract request_id from a goal alignment request key."""
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("goal_alignment_request", "")


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
    def plan_key(agent_id: str, namespace: str) -> str:
        """Key for an agent's current plan."""
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(agent_id=agent_id))

    @staticmethod
    def plan_pattern(namespace: str, agent_id: str | None = None) -> str:
        """Pattern matching plans (optionally for a specific agent)."""
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(agent_id=agent_id))

    @staticmethod
    def approval_request_key(plan_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, f"{PlanProtocol._APPROVAL_PREFIX}{plan_id}")

    @staticmethod
    def notification_key(agent_id: str, timestamp: float, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, f"{PlanProtocol._NOTIFICATION_PREFIX}{agent_id}:{timestamp}")

    @staticmethod
    def subscription_key(plan_id: str, subscriber_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, f"{PlanProtocol._SUBSCRIPTION_PREFIX}{plan_id}:{subscriber_id}")

    @staticmethod
    def parse_plan_key(key: str, namespace: str) -> str:
        """Extract agent_id from a plan key."""
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
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
    def error_key(error_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, f"{ErrorSignalProtocol._PREFIX}{error_id}")

    @staticmethod
    def error_pattern(namespace: str) -> str:
        return BlackboardProtocol._ns_pattern(namespace, f"{ErrorSignalProtocol._PREFIX}*")

    @staticmethod
    def parse_error_key(key: str, namespace: str) -> str:
        """Extract error_id from an error key."""
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
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
    def query_key(agent_id: str, query_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(agent_id=agent_id, dependency_query=query_id))

    @staticmethod
    def query_pattern(namespace: str, agent_id: str | None = None) -> str:
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(agent_id=agent_id, dependency_query=None))

    @staticmethod
    def parse_query_key(key: str, namespace: str) -> dict[str, str]:
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
        return ScopeUtils.parse_key("", key)


class ReputationProtocol(BlackboardProtocol):
    """Protocol for reputation tracking in multi-agent games.

    Colony-scoped. Multiple agents submit reputation update requests.
    A dedicated handler aggregates and publishes reputation scores.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.COLONY

    @staticmethod
    def update_request_key(requesting_agent_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(reputation="update", requesting_agent_id=requesting_agent_id))

    @staticmethod
    def result_key(namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(reputation="result"))

    @staticmethod
    def update_request_pattern(namespace: str) -> str:
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(reputation="update", requesting_agent_id=None))

    @staticmethod
    def state_pattern(namespace: str) -> str:
        """Pattern for game state changes that trigger reputation updates."""
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(state=None))

    @staticmethod
    def task_outcome_pattern(namespace: str) -> str:
        """Pattern for task outcomes that affect reputation."""
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(task_outcome=None))

    @staticmethod
    def all_input_patterns(namespace: str) -> list[str]:
        """All patterns this protocol monitors."""
        return [
            ReputationProtocol.update_request_pattern(namespace=namespace),
            ReputationProtocol.state_pattern(namespace=namespace),
            ReputationProtocol.task_outcome_pattern(namespace=namespace),
        ]

    @staticmethod
    def parse_update_request_key(key: str, namespace: str) -> dict[str, str]:
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
        return ScopeUtils.parse_key("", key)


class ConsciousnessProtocol(BlackboardProtocol):
    """Protocol for consciousness state publication.

    Agent-scoped. Tracks consciousness level, attention focus,
    and meta-cognitive state.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def state_key(state_type: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(consciousness=state_type))

    @staticmethod
    def state_pattern(namespace: str) -> str:
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(consciousness=None))

    @staticmethod
    def parse_state_key(key: str, namespace: str) -> str:
        """Extract state_type from a consciousness key."""
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("consciousness", "")


class ReflectionProtocol(BlackboardProtocol):
    """Protocol for reflection requests/results.

    Agent-scoped. Used to trigger and receive self-reflection analysis.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def request_key(request_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(reflection_request=request_id))

    @staticmethod
    def result_key(request_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(reflection_result=request_id))

    @staticmethod
    def request_pattern(namespace: str) -> str:
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(reflection_request=None))

    @staticmethod
    def result_pattern(namespace: str) -> str:
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(reflection_result=None))

    @staticmethod
    def parse_request_key(key: str, namespace: str) -> str:
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("reflection_request", "")


class AnalysisResultProtocol(BlackboardProtocol):
    """Protocol for analysis result publication.

    Agent-scoped. Used by ``AdaptiveQueryGenerator`` to monitor
    completed analysis results and generate follow-up queries.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def result_key(result_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(analysis_result=result_id))

    @staticmethod
    def result_pattern(namespace: str) -> str:
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(analysis_result=None))

    @staticmethod
    def parse_result_key(key: str, namespace: str) -> str:
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("analysis_result", "")


class ExplorationProtocol(BlackboardProtocol):
    """Protocol for query-driven exploration requests/results.

    Agent-scoped.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def request_key(request_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(exploration_request=request_id))

    @staticmethod
    def request_pattern(namespace: str) -> str:
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(exploration_request=None))

    @staticmethod
    def parse_request_key(key: str, namespace: str) -> str:
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("exploration_request", "")


class IncrementalQueryProtocol(BlackboardProtocol):
    """Protocol for incremental query requests.

    Agent-scoped.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def request_key(request_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(incremental_query_request=request_id))

    @staticmethod
    def request_pattern(namespace: str) -> str:
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(incremental_query_request=None))

    @staticmethod
    def parse_request_key(key: str, namespace: str) -> str:
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("incremental_query_request", "")


class MultiHopSearchProtocol(BlackboardProtocol):
    """Protocol for multi-hop search requests.

    Agent-scoped.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    @staticmethod
    def request_key(request_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, ScopeUtils.format_key(multi_hop_search_request=request_id))

    @staticmethod
    def request_pattern(namespace: str) -> str:
        return BlackboardProtocol._ns_pattern(namespace, ScopeUtils.pattern_key(multi_hop_search_request=None))

    @staticmethod
    def parse_request_key(key: str, namespace: str) -> str:
        key = key[len(namespace) + 1:]  # Strip "{namespace}:" prefix
        parsed = ScopeUtils.parse_key("", key)
        return parsed.get("multi_hop_search_request", "")
