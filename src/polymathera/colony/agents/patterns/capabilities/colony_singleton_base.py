"""``ColonySingletonCapabilityBase`` — shared base for capabilities
mounted on the default colony system session's ``SessionAgent``
(per P8-0 foundation).

Extracted post-P11 when the third consumer arrived, per the
no-premature-abstractions discipline. Currently subclassed by:

- :class:`GitHubInboundCapability` (P8a) — colony GitHub poller.
- :class:`InteractionLogCapability` (P8b) — Postgres write-through
  for blackboard events.
- :class:`MentionRoutingCapability` (P10) — ``@colony`` / ``@polymath``
  mention parser.

The base owns ONLY the 3-way-duplicated shape:

1. ``__init__`` accepts ``agent``, ``scope`` (default COLONY), and
   a ``scope_id`` constructor passthrough. Derives ``scope_id`` via
   :func:`get_scope_prefix(scope, agent)` when not explicitly set —
   matching the ``GitHubCapability`` reference pattern that P5
   established. The ``scope_id`` passthrough is required for
   detached/test instances (the :class:`AgentCapability` base
   raises if neither ``agent`` nor ``scope_id`` is provided).
2. No-op ``serialize_suspension_state`` / ``deserialize_suspension_state``
   overrides — colony singletons own no agent-suspendable state
   (cursors, tables, blackboard events survive process restarts
   independently of agent suspension).

What's intentionally NOT here:

- The "read ``tenant_id`` / ``colony_id`` from agent metadata or
  quiesce" pattern. Two of the three current consumers
  (``GitHubInboundCapability`` + ``InteractionLogCapability``) have
  it; ``MentionRoutingCapability`` does NOT (it routes by scope, not
  by explicit tenant/colony lookup). Two consumers is below the
  extraction threshold — fold it in here when the third arrives.
- The ``_quiesced_reason`` field. Same reasoning — only the two
  Postgres-using subclasses track it.
- Lazy ``db_pool`` acquisition. Already extracted to
  :mod:`agents.utils.postgres` (``get_agent_db_pool``); subclasses
  call it directly when they need a pool.

Subclasses MUST forward ``agent`` / ``scope`` / ``scope_id`` /
``capability_key`` / ``app_name`` to ``super().__init__``; their
own constructor kwargs (db_pool, config, client, …) are
subclass-private.
"""

from __future__ import annotations

from ...base import Agent, AgentCapability
from ...scopes import BlackboardScope, get_scope_prefix


class ColonySingletonCapabilityBase(AgentCapability):
    """See module docstring."""

    def __init__(
        self,
        agent: Agent | None = None,
        scope: BlackboardScope = BlackboardScope.COLONY,
        *,
        scope_id: str | None = None,
        capability_key: str | None = None,
        app_name: str | None = None,
    ):
        if scope_id is None and agent is not None:
            scope_id = get_scope_prefix(scope, agent)
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            capability_key=capability_key,
            app_name=app_name,
        )

    async def serialize_suspension_state(self, state):  # type: ignore[no-untyped-def]
        """No-op — colony singletons own no agent-suspendable state."""
        return state

    async def deserialize_suspension_state(self, state):  # type: ignore[no-untyped-def]
        """No-op — see :meth:`serialize_suspension_state`."""
        return None


__all__ = ("ColonySingletonCapabilityBase",)
