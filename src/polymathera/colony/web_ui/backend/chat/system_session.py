"""Default colony session — the always-on ``SessionAgent`` host for
colony-singleton capabilities (P8: ``GitHubInboundCapability`` +
``InteractionLogCapability``; P9+ adds the webhook receiver, mention
routing, etc.).

Why a separate session at all? Per-tenant per-colony singletons need
a long-lived agent host that participates in the existing per-session
capability lifecycle (``initialize`` / ``shutdown`` / ``@event_handler``)
without inventing a new cluster-level capability primitive. The
``SessionAgent`` already provides exactly that surface. Bundling the
singletons onto a special session — one per colony, marked
``session_kind="system"`` — reuses every piece of existing
infrastructure (Traces visibility, blackboard scoping, capability mount
plumbing) with one schema-field addition.

This session is hidden from the chat-UI sessions list by default
(``GET /sessions/?include_system=false``); the chat-attach WebSocket
refuses to bind to it; the Traces tab still surfaces it. The
``SessionAgent`` itself uses a TRIMMED capability blueprint — only
``SessionOrchestratorCapability`` for the Session-model integration,
plus whichever colony-singleton capabilities the running phase needs
(P8a adds ``GitHubInboundCapability``, P8b adds
``InteractionLogCapability``, etc.). The 13+ per-user capabilities
mounted in :func:`routers.sessions.create_session` (UserPlugin,
SandboxedShell, WebSearch, …) make no sense for a no-user session
and are intentionally NOT mounted here.
"""

from __future__ import annotations

import logging
from typing import Any

from polymathera.colony.agents import AgentHandle, AgentMetadata
from polymathera.colony.distributed.ray_utils.serving.context import (
    ExecutionContext,
    Ring,
)
from polymathera.colony.agents.patterns.capabilities.github_inbound import (
    GitHubInboundCapability,
)
from polymathera.colony.agents.patterns.capabilities.interaction_log import (
    InteractionLogCapability,
)
from polymathera.colony.agents.patterns.capabilities.mention_routing import (
    MentionRoutingCapability,
)
from polymathera.colony.agents.scopes import BlackboardScope
from polymathera.colony.agents.self_concept import AgentSelfConcept

from .session_agent import SessionAgent, SessionOrchestratorCapability
from ..services.colony_connection import ColonyConnection


logger = logging.getLogger(__name__)


def _build_system_session_agent_metadata(
    *, tenant_id: str, colony_id: str, session_id: str,
) -> AgentMetadata:
    """Trimmed ``AgentMetadata`` for the system session.

    No per-user GitHub identity, no per-colony git-attribution, no
    design_monorepo_url — those are session-author-specific knobs the
    chat path threads in. The system session is not a chat session.
    Downstream readers that hit a missing key tolerate ``None``; this
    metadata is intentionally minimal.

    The caller passes ``tenant_id`` + ``colony_id`` + ``session_id``
    explicitly so we can build the syscontext without relying on
    whatever ``with execution_context(...)`` block happens to be in
    scope at construction time. The bootstrap path enters two nested
    contexts (one without session_id around session-create, one with
    session_id around the agent spawn), so an implicit syscontext
    snapshot can land in either; explicit construction is the
    durable shape.
    """

    return AgentMetadata(
        role="colony_system_session",
        syscontext=ExecutionContext(
            ring=Ring.USER,
            tenant_id=tenant_id,
            colony_id=colony_id,
            session_id=session_id,
            origin="colony_system_session",
        ),
        goals=[
            "Host always-on colony-singleton capabilities for this colony",
        ],
        self_concept=AgentSelfConcept(
            agent_id="",  # Overwritten by ConsciousnessCapability
            name="Colony System Session",
            role=(
                "the always-on host agent for colony-wide singleton "
                "capabilities (GitHub inbound polling, interaction "
                "log, future webhook + mention handlers)"
            ),
            description=(
                "You are NOT a chat agent — no user is attached to "
                "this session. You exist so colony-singleton "
                "capabilities have a long-lived ``SessionAgent`` host "
                "with the standard initialize / shutdown / "
                "@event_handler lifecycle. Capabilities mount on you "
                "via the system-session blueprint; their behaviour is "
                "driven by blackboard events, NOT by user messages. "
                "You are a proactive agent: when you have no work to do, "
                "end your code block with "
                "``await run('wait_for_next_event')`` to pause until the "
                "next event arrives, instead of burning empty planning "
                "iterations."
            ),
        ),
        parameters={
            "session_kind": "system",
        },
        action_policy_config={
            "allow_self_termination": False,
            "planning_capability_blueprints": [],
        },
    )


def build_system_session_agent_blueprint(
    *,
    tenant_id: str,
    colony_id: str,
    session_id: str,
) -> Any:
    """Build the trimmed ``SessionAgent.bind`` blueprint for a system
    session.

    Mounts only the minimum capability surface required to integrate
    with the Session / Traces model + the colony-singleton
    capabilities for the running phase. Subsequent phases extend the
    ``capability_blueprints`` list with one ``bind`` line each:

    - ``GitHubInboundCapability.bind(scope=COLONY)``.
    - ``InteractionLogCapability.bind(scope=COLONY)``.
    - webhook receiver lives in the dashboard process (route
      ``POST /api/v1/github/webhook``), not on this blueprint —
      receives directly + writes to colony blackboards via
      :func:`web_ui.backend.github_webhook.publisher.publish_to_tenant_colonies`.
    - ``MentionRoutingCapability.bind(scope=COLONY)``.

    Postgres-backed capabilities acquire their own per-process pool
    lazily inside ``initialize()`` from RDS_* env vars (set on every
    ray-worker by docker-compose.yml). Live asyncpg pools are NOT
    cloudpickle-serializable so they CAN'T ride through ``bind`` →
    cloudpickle → ray-worker — the lazy-acquisition pattern is the
    only sound shape.
    """

    metadata = _build_system_session_agent_metadata(
        tenant_id=tenant_id,
        colony_id=colony_id,
        session_id=session_id,
    )
    return SessionAgent.bind(
        agent_type=SessionAgent.model_fields["agent_type"].default,
        metadata=metadata,
        bound_pages=[],
        capability_blueprints=[
            # SessionOrchestratorCapability is the minimum required
            # for the SessionAgent to integrate with the Session /
            # Traces model. Colony-singleton capabilities get appended
            # below — see this function's docstring for the phasing.
            SessionOrchestratorCapability.bind(),
            # P8a: GitHub inbound polling — one tick loop per colony.
            # COLONY scope so emitted GitHubEventProtocol writes reach
            # user sessions in the same colony.
            GitHubInboundCapability.bind(
                scope=BlackboardScope.COLONY,
            ),
            # P8b: write-through mirror of colony-scoped blackboard
            # events to the ``interaction_log`` Postgres table. v1
            # subscribes to ``GitHubEventProtocol.*`` only — chat +
            # action-lifecycle coverage (SESSION/AGENT scopes) is a
            # follow-up. Mount AFTER the inbound poller so the order
            # of `@event_handler` registration on this agent matches
            # the order events are produced (cosmetic; doesn't affect
            # correctness, just trace clarity).
            InteractionLogCapability.bind(
                scope=BlackboardScope.COLONY,
            ),
            # P10: mention parser. Watches the same ``github:*`` events
            # InteractionLog sees + emits one ``MentionEventProtocol``
            # write per ``@colony`` / ``@polymath`` mention found in
            # the comment / issue / PR body. InteractionLog also
            # subscribes to ``mention:*`` so mentions land in the log
            # as their own rows.
            MentionRoutingCapability.bind(
                scope=BlackboardScope.COLONY,
            ),
        ],
        action_policy_blueprints={
            # No consciousness streams — the system session has no
            # conversation to maintain (chat-attach refuses to bind).
            "consciousness_streams": [],
        },
    )


async def ensure_system_session_for_colony(
    colony: ColonyConnection,
    *,
    tenant_id: str,
    colony_id: str,
) -> str | None:
    """Idempotently bootstrap the system session for a single colony.

    Two-phase, both safe to re-run:

    1. Call ``SessionManagerDeployment.ensure_system_session()`` — the
       endpoint is itself lookup-first-then-create and survives
       concurrent callers (write_transaction guard).
    2. If the returned session has no ``session_agent_id``, spawn the
       system-session ``SessionAgent`` via
       :func:`build_system_session_agent_blueprint` and call
       ``set_session_agent_id`` to claim it (the endpoint refuses to
       overwrite an existing id).

    Returns the system session's id, or ``None`` on bootstrap failure
    (logged at ERROR; callers continue with other colonies).
    """

    try:
        with colony.user_execution_context(
            tenant_id=tenant_id,
            colony_id=colony_id,
            origin="dashboard_system_session_bootstrap",
        ):
            sm = await colony.get_session_manager()
            session = await sm.ensure_system_session()

            session_id = getattr(session, "session_id", None) or (
                session.get("session_id") if isinstance(session, dict) else None
            )
            session_agent_id = getattr(session, "session_agent_id", None) or (
                session.get("session_agent_id")
                if isinstance(session, dict) else None
            )

            if not session_id:
                logger.error(
                    "ensure_system_session_for_colony: tenant=%s colony=%s "
                    "— session_manager returned no session_id",
                    tenant_id, colony_id,
                )
                return None

            if session_agent_id is not None:
                # SessionAgent already spawned in a prior bootstrap pass.
                return session_id

            blueprint = build_system_session_agent_blueprint(
                tenant_id=tenant_id,
                colony_id=colony_id,
                session_id=session_id,
            )
            # The blueprint spawn must run inside a session-scoped
            # execution context so the agent's scope_id resolves
            # correctly downstream.
            with colony.user_execution_context(
                tenant_id=tenant_id,
                colony_id=colony_id,
                session_id=session_id,
                origin="dashboard_system_session_bootstrap",
            ):
                handle = await AgentHandle.from_blueprint(
                    agent_blueprint=blueprint,
                    app_name=colony.app_name,
                )
            agent_id = handle.agent_id

            # Idempotent — returns the existing id if another caller
            # raced to set it. Discard the loser silently.
            settled = await sm.set_session_agent_id(
                session_id=session_id, agent_id=agent_id,
            )
            logger.info(
                "ensure_system_session_for_colony: tenant=%s colony=%s "
                "session=%s agent=%s",
                tenant_id, colony_id, session_id, settled,
            )
            return session_id
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "ensure_system_session_for_colony: tenant=%s colony=%s "
            "failed: %s",
            tenant_id, colony_id, exc, exc_info=True,
        )
        return None


async def ensure_system_sessions_for_all_colonies(
    colony: ColonyConnection,
) -> None:
    """Bootstrap a system session per colony at dashboard startup.

    Reads every colony in Postgres via
    :func:`auth_service.list_all_colonies` and calls
    :func:`ensure_system_session_for_colony` for each. Best-effort —
    a single colony's bootstrap failure does not stop the rest.

    Called once from ``main.lifespan`` after the schema-ensure step
    and again from ``routers.colonies.create_colony`` after a new
    colony lands. Both paths converge on the same per-colony helper,
    which is idempotent.
    """

    db_pool = colony._db_pool
    if db_pool is None:
        logger.warning(
            "ensure_system_sessions_for_all_colonies: db_pool unavailable; "
            "skipping system-session bootstrap",
        )
        return

    if not colony.is_connected:
        logger.warning(
            "ensure_system_sessions_for_all_colonies: colony cluster not "
            "connected; skipping system-session bootstrap",
        )
        return

    from ..auth import service as auth_service

    colonies = await auth_service.list_all_colonies(db_pool)
    if not colonies:
        logger.info(
            "ensure_system_sessions_for_all_colonies: no colonies in "
            "postgres; nothing to bootstrap",
        )
        return

    logger.info(
        "ensure_system_sessions_for_all_colonies: bootstrapping %d "
        "colony system session(s)",
        len(colonies),
    )
    for row in colonies:
        await ensure_system_session_for_colony(
            colony,
            tenant_id=row["tenant_id"],
            colony_id=row["colony_id"],
        )


__all__ = (
    "build_system_session_agent_blueprint",
    "ensure_system_session_for_colony",
    "ensure_system_sessions_for_all_colonies",
)
