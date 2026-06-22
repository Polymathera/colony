"""``InteractionLogCapability`` — colony-singleton Postgres-backed
write-through for blackboard events.

v1 (P8b) subscribes to ``GitHubEventProtocol.*`` only — the
colony-scoped GitHub inbound events that ``GitHubInboundCapability``
emits (P8a) + the equivalent webhook events P9 will emit. Chat
(``SessionChatProtocol`` — SESSION scope) and action lifecycle
(``ActionPolicyLifecycleProtocol`` — AGENT scope) are NOT in v1
because the capability is COLONY-scoped and can't receive
SESSION-/AGENT-scoped events natively. They land as follow-ups once
the originating capabilities republish to colony scope OR a
cross-scope subscription primitive lands.

Mounted on the system session's ``SessionAgent`` (P8-0 foundation),
same shape as ``GitHubInboundCapability``. The capability quiesces
silently if the agent process can't reach Postgres — the
SessionAgent stays up so other system-session capabilities still run.
"""

from __future__ import annotations

import logging
from typing import Any

from ...events import event_handler
from ....base import Agent
from ....blackboard.protocol import (
    AgentDiagnosticProtocol,
    BottleneckDetectedProtocol,
    DesignInconsistencyProtocol,
    GitHubEventProtocol,
    MentionEventProtocol,
)
from ....scopes import BlackboardScope
from ....utils.postgres import get_agent_db_pool
from ..colony_singleton_base import ColonySingletonCapabilityBase
from .service import insert_event


logger = logging.getLogger(__name__)


# Map GitHubEventProtocol key-prefix → (event_kind, key-parser).
# The dispatch table keeps the @event_handler body small and the
# unit tests can pin individual entries by feeding the right key.
_GITHUB_KEY_DISPATCH = (
    (
        GitHubEventProtocol._ISSUE_OPENED,
        "github_issue_event",
        GitHubEventProtocol.parse_issue_opened_key,
    ),
    (
        GitHubEventProtocol._ISSUE_CLOSED,
        "github_issue_event",
        lambda key: GitHubEventProtocol._parse_issue(
            GitHubEventProtocol._ISSUE_CLOSED, key,
        ),
    ),
    (
        GitHubEventProtocol._ISSUE_COMMENTED,
        "github_comment_event",
        GitHubEventProtocol.parse_issue_commented_key,
    ),
    (
        GitHubEventProtocol._PR_OPENED,
        "github_pr_event",
        GitHubEventProtocol.parse_pr_opened_key,
    ),
    (
        GitHubEventProtocol._PR_REVIEW_REQUESTED,
        "github_pr_event",
        lambda key: GitHubEventProtocol._parse_issue(
            GitHubEventProtocol._PR_REVIEW_REQUESTED, key,
        ),
    ),
    (
        GitHubEventProtocol._PR_MERGED,
        "github_pr_event",
        lambda key: GitHubEventProtocol._parse_issue(
            GitHubEventProtocol._PR_MERGED, key,
        ),
    ),
)


def _classify_github_key(key: str) -> tuple[str, str, int] | None:
    """Return ``(event_kind, repo, number)`` for a GitHubEventProtocol
    issue/PR key. ``None`` for keys we don't write-through in v1
    (e.g. ``github:project_item_changed`` — no issue number to
    surface as a ref)."""

    for prefix, event_kind, parser in _GITHUB_KEY_DISPATCH:
        if key.startswith(prefix):
            try:
                repo, number = parser(key)
            except ValueError:
                return None
            return event_kind, repo, number
    return None


class InteractionLogCapability(ColonySingletonCapabilityBase):
    """Colony-singleton write-through to ``interaction_log``.

    Args:
        agent: Owning ``SessionAgent`` (the system session's).
        scope: ``BlackboardScope.COLONY`` — the protocols this
            capability subscribes to write at colony scope so the
            capability must be mounted at colony scope to receive them.
        db_pool: Pre-built Postgres pool for test injection only.
            Production callers MUST NOT pass this — live asyncpg
            pools are not cloudpickle-serializable + would fail
            ``validate_serializable`` at bind time. Production
            acquires the pool lazily via
            :func:`agents.utils.postgres.get_agent_db_pool`.
        capability_key: Dispatcher key.
        app_name: Serving application name override.
    """

    def __init__(
        self,
        agent: Agent | None = None,
        scope: BlackboardScope = BlackboardScope.COLONY,
        *,
        scope_id: str | None = None,
        db_pool: Any = None,
        capability_key: str | None = None,
        app_name: str | None = None,
    ):
        super().__init__(
            agent=agent,
            scope=scope,
            scope_id=scope_id,
            capability_key=capability_key,
            app_name=app_name,
        )
        self._db_pool = db_pool
        self._tenant_id: str | None = None
        self._colony_id: str | None = None
        self._quiesced_reason: str | None = "not_initialized"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Pull tenant + colony from agent metadata, acquire the
        shared agent-process db_pool. Quiesces silently on either
        failure — write-through becomes a no-op, the SessionAgent
        stays up."""

        await super().initialize()

        # Read tenant + colony from the agent's syscontext-derived
        # typed properties — both are captured at metadata
        # construction time by ``AgentMetadata``'s
        # ``default_factory=serving.require_execution_context``
        # (the system-session bootstrap runs blueprint construction
        # inside ``colony.user_execution_context``).
        if self._agent is None:
            self._tenant_id = ""
            self._colony_id = ""
        else:
            self._tenant_id = self._agent.metadata.tenant_id
            self._colony_id = self._agent.metadata.colony_id
        if not self._tenant_id or not self._colony_id:
            self._quiesced_reason = "no_tenant_or_colony_in_syscontext"
            logger.warning(
                "InteractionLogCapability: tenant_id/colony_id absent "
                "from agent syscontext; write-through DISABLED.",
            )
            return

        if self._db_pool is None:
            try:
                self._db_pool = await get_agent_db_pool()
            except Exception as exc:  # noqa: BLE001
                self._quiesced_reason = "no_db_pool"
                logger.warning(
                    "InteractionLogCapability: failed to acquire "
                    "agent-process Postgres pool (%s); write-through "
                    "DISABLED. Requires the ray-worker container to "
                    "have RDS_* env vars set (docker-compose.yml).",
                    exc,
                )
                return

        self._quiesced_reason = None
        logger.info(
            "InteractionLogCapability: write-through enabled for "
            "tenant=%s colony=%s",
            self._tenant_id, self._colony_id,
        )

    # ------------------------------------------------------------------
    # Write-through — GitHubEventProtocol.*
    # ------------------------------------------------------------------

    @event_handler(pattern=GitHubEventProtocol.all_pattern())
    async def _on_github_event(self, event, _scope) -> None:  # type: ignore[no-untyped-def]
        """Mirror a ``GitHubEventProtocol.*`` write into ``interaction_log``.

        Quiesced no-op if ``initialize()`` couldn't reach Postgres or
        if the key shape is one we don't write-through in v1 (e.g.
        ``github:project_item_changed`` — no issue number to surface).
        """

        if self._quiesced_reason is not None:
            return

        key = event.key
        value = event.value or {}

        classified = _classify_github_key(key)
        if classified is None:
            return
        event_kind, repo, number = classified

        # Surface the issue/PR as a ref so cross-channel joins
        # ("show me everything that touched issue X") hit the GIN
        # index. ``owner/repo#N`` is the canonical short form GitHub
        # itself uses.
        issue_ref = f"{repo}#{number}"
        refs = [{"kind": "issue", "value": issue_ref}]
        channel_ref = (
            f"https://github.com/{repo}/issues/{number}"
        )

        # Best-effort: a single failed insert MUST NOT crash the
        # event handler (otherwise one bad row stops every future
        # event). Log + return.
        try:
            await insert_event(
                self._db_pool,
                tenant_id=self._tenant_id,
                colony_id=self._colony_id,
                channel="github",
                event_kind=event_kind,
                payload=dict(value),
                refs=refs,
                channel_ref=channel_ref,
                user_login=value.get("author_login"),
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "InteractionLogCapability: insert_event failed for "
                "key=%s — dropping event",
                key,
            )

    # ------------------------------------------------------------------
    # Write-through — MentionEventProtocol.* (P10)
    # ------------------------------------------------------------------

    @event_handler(pattern=MentionEventProtocol.event_pattern())
    async def _on_mention_event(self, event, _scope) -> None:  # type: ignore[no-untyped-def]
        """Mirror a :class:`MentionEventProtocol` write into
        ``interaction_log`` so mentions are queryable via
        ``fetch_recent_activity`` + ``fetch_by_ref``.

        Same best-effort + quiesce semantics as ``_on_github_event``.
        ``channel='github'`` because the mention originated from a
        GitHub event; ``event_kind='mention_event'`` discriminates
        it from raw issue/comment/PR rows in the log.
        """

        if self._quiesced_reason is not None:
            return

        key = event.key
        value = event.value or {}

        repo = value.get("repo")
        number = value.get("issue_number")
        if not isinstance(repo, str) or not isinstance(number, int):
            return

        # Two refs: the source issue (so "history of issue X" picks
        # up its mentions) + the matched handle (so "everything
        # @colony-roadmap was mentioned in" is a single GIN query).
        refs = [
            {"kind": "issue", "value": f"{repo}#{number}"},
        ]
        mention_kind = value.get("mention_kind")
        if isinstance(mention_kind, str) and mention_kind:
            refs.append({"kind": "mention", "value": mention_kind})

        channel_ref = value.get("html_url") or (
            f"https://github.com/{repo}/issues/{number}"
        )

        try:
            await insert_event(
                self._db_pool,
                tenant_id=self._tenant_id,
                colony_id=self._colony_id,
                channel="github",
                event_kind="mention_event",
                payload=dict(value),
                refs=refs,
                channel_ref=channel_ref,
                user_login=value.get("commenter_login"),
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "InteractionLogCapability: insert_event failed for "
                "mention key=%s — dropping event",
                key,
            )

    # ------------------------------------------------------------------
    # Write-through — alert protocols (P11)
    # ------------------------------------------------------------------

    @event_handler(pattern=BottleneckDetectedProtocol.event_pattern())
    async def _on_bottleneck_detected(self, event, _scope) -> None:  # type: ignore[no-untyped-def]
        """Mirror a :class:`BottleneckDetectedProtocol` write into
        ``interaction_log`` so the ColonyStatusPanel's "Alerts" tile
        can query recent bottlenecks via ``fetch_recent_activity``
        filtered by ``event_kind='bottleneck'``.

        Same quiesce / best-effort semantics as ``_on_github_event``.
        """
        await self._insert_alert(event, channel="internal", event_kind="bottleneck")

    @event_handler(pattern=DesignInconsistencyProtocol.event_pattern())
    async def _on_design_inconsistency(self, event, _scope) -> None:  # type: ignore[no-untyped-def]
        """Mirror a :class:`DesignInconsistencyProtocol` write into
        ``interaction_log``. Pair with ``_on_bottleneck_detected``
        so the alerts query returns both kinds in one tail."""
        await self._insert_alert(event, channel="internal", event_kind="inconsistency")

    # ------------------------------------------------------------------
    # Write-through — AgentDiagnosticProtocol
    # ------------------------------------------------------------------

    @event_handler(pattern=AgentDiagnosticProtocol.event_pattern())
    async def _on_agent_diagnostic(self, event, _scope) -> None:  # type: ignore[no-untyped-def]
        """Mirror an :class:`AgentDiagnosticProtocol` event into
        ``interaction_log`` so the health dashboard can tail recent
        agent diagnostics (session-agent crashes, GitHub inbound
        quiesce, etc.) via the standard ``fetch_recent_activity``
        query filtered by ``event_kind='agent_diagnostic'``.

        Producers writing on the colony blackboard:
        - ``session_agent_stopped`` from session_agent_lifecycle.py
        - ``github_inbound_quiesced`` from github_inbound/capability.py

        Both carry an ``agent_id`` + ``kind`` in the payload; the
        ``kind`` is surfaced as a ref so dashboard panels can filter
        by diagnostic type without parsing keys.
        """

        if self._quiesced_reason is not None:
            return

        value = event.value or {}
        refs: list[dict[str, str]] = []
        kind = value.get("kind")
        if isinstance(kind, str) and kind:
            refs.append({"kind": "diagnostic_kind", "value": kind})
        agent_id = value.get("agent_id")
        if isinstance(agent_id, str) and agent_id:
            refs.append({"kind": "agent_id", "value": agent_id})

        try:
            await insert_event(
                self._db_pool,
                tenant_id=self._tenant_id,
                colony_id=self._colony_id,
                channel="internal",
                event_kind="agent_diagnostic",
                payload=dict(value),
                refs=refs,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "InteractionLogCapability: insert_event failed for "
                "agent_diagnostic key=%s — dropping event",
                event.key,
            )

    async def _insert_alert(
        self, event, *, channel: str, event_kind: str,
    ) -> None:
        """Shared body for the two alert subscriptions.

        Both protocols carry their own payload shapes (BottleneckDetected
        carries ``repo`` + ``issue_number``; DesignInconsistency carries
        ``source_name`` + claim ids). We surface what we can as refs +
        store the full payload as JSONB — the dashboard renderer reads
        the payload directly for kind-specific display."""

        if self._quiesced_reason is not None:
            return

        value = event.value or {}
        refs: list[dict[str, str]] = []

        # Best-effort ref extraction — protocols may evolve their value
        # shapes; missing fields are silently skipped, the row still
        # lands with the raw payload preserved.
        repo = value.get("repo")
        issue_number = value.get("issue_number")
        if isinstance(repo, str) and isinstance(issue_number, int):
            refs.append({"kind": "issue", "value": f"{repo}#{issue_number}"})
        source_name = value.get("source_name")
        if isinstance(source_name, str) and source_name:
            refs.append({"kind": "source", "value": source_name})

        try:
            await insert_event(
                self._db_pool,
                tenant_id=self._tenant_id,
                colony_id=self._colony_id,
                channel=channel,
                event_kind=event_kind,
                payload=dict(value),
                refs=refs,
                channel_ref=value.get("url"),
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "InteractionLogCapability: insert_event failed for "
                "%s key=%s — dropping event",
                event_kind, event.key,
            )
