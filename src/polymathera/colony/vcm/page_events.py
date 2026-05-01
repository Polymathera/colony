"""Page-change events — the canonical signalling shape for VCM mutations.

Per the design-automation architecture (master §5.6 item 2), context
sources emit ``PageChangeEvent``s onto the colony blackboard topic
``vcm:page_events:*``. Capabilities and agents subscribe through the
existing ``EnhancedBlackboard.stream_events_to_queue`` machinery; the
convergence runtime subscribes to the topic centrally and dispatches
typed-predicate subscriptions on the basis of those events.

Five kinds, distinguished by ``PageChangeKind``. The shape is a single
discriminated union (one Pydantic model with a ``kind`` field) rather
than five subclasses — that keeps blackboard transport, JSON
serialisation, and pattern matching straightforward without the
overhead of polymorphic dispatch.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class PageChangeKind(str, Enum):
    """The five page-graph mutation kinds (master §5.6 item 2)."""

    PAGE_INVALIDATED = "page_invalidated"
    """The page no longer reflects its source — eviction + subscriber
    notification, but no replacement page exists yet."""

    PAGE_REPLACED = "page_replaced"
    """The source mutated and the new content has been re-paged. The
    old page id is retired; the new page id is its successor (and
    carries a ``superseded_by`` graph edge from the old to the new)."""

    PAGE_ADDED = "page_added"
    """A brand-new page entered the graph; no predecessor."""

    PAGE_GRAPH_EDGE_ADDED = "page_graph_edge_added"
    """A relationship between two existing pages was discovered or
    asserted. No page content changed."""

    PAGE_GRAPH_EDGE_REMOVED = "page_graph_edge_removed"
    """A previously-asserted edge was retracted (e.g., a ``cites`` edge
    on a paper that turned out to be a hallucinated reference). No
    page content changed."""


# ---------------------------------------------------------------------------
# Blackboard topic constants
# ---------------------------------------------------------------------------


PAGE_EVENTS_TOPIC_PREFIX = "vcm:page_events"
"""All page-change events are written under this prefix on the colony
blackboard scope. The runtime subscribes to ``vcm:page_events:*``.

Per-source subkeys follow the convention
``vcm:page_events:<source-id>:<kind>``, where ``<source-id>`` is a
free-form source-stable id (typically the source's ``scope_id``).
This lets a downstream consumer filter by source via the existing
``KeyPatternFilter``.
"""


CONVERGENCE_STATUS_KEY = "convergence:status"
"""Singleton key the runtime updates with the current
``ConvergenceStatus`` snapshot. Subscribed by the SessionAgent's
"converging / converged / cycling" indicator (master §5.4)."""


CONVERGENCE_QUIESCENCE_TOPIC = "convergence:quiescence"
"""Event the runtime emits each time a dispatch wave settles with no
new triggered work. Carries the episode id and the dispatch count.
The DesignCheckpointer waits for this before tagging a checkpoint
(master §8.1)."""


CONVERGENCE_CHANGE_FEED_KEY = "convergence:change_feed"
"""Singleton key the runtime updates with the bounded change-feed
(most recent N dispatches). Master §5.4 surface."""


CONVERGENCE_DISPATCH_PREFIX = "convergence:dispatch"
"""Per-subscription dispatch events. The runtime writes
``convergence:dispatch:<subscription_id>`` on the subscription's
declared scope, and the subscribing capability's ``@event_handler``
picks it up through the normal blackboard event machinery."""


# ---------------------------------------------------------------------------
# The event shape
# ---------------------------------------------------------------------------


class PageChangeEvent(BaseModel):
    """One page-graph mutation event.

    Carried as the ``value`` of a blackboard write under the
    ``vcm:page_events:*`` topic. The blackboard's ``BlackboardEvent``
    wraps this with timestamps, key, etc.; this is the *payload*
    semantically owned by the VCM source layer.

    Field discipline:

    - ``page_id`` is set for every kind; it is the *primary* page the
      event concerns (the new page id for ``page_replaced`` and
      ``page_added``; the affected page for the others; for edge
      events, one of the edge endpoints — the other is in
      ``related_page_ids[0]``).
    - ``related_page_ids`` carries auxiliary page references (the *old*
      page id for ``page_replaced``; the other edge endpoint for edge
      events).
    - ``edge_type`` is set only for edge events.
    - ``source`` is the source URI (e.g., ``git:<remote>:<branch>:<sha>``)
      so subscribers can filter by source without re-resolving the page.
    - ``data_type`` mirrors the page's typed metadata so subscribers
      can filter without resolving the page.
    - ``reason`` is free-form; used by ``page_invalidated`` to explain
      *why* (e.g., ``"source file deleted"``, ``"effective_at expired"``).
    - ``edit_diff`` is set on ``page_replaced`` when the source can
      cheaply summarise the change (a unified-diff text or a structural
      diff record). Empty when the source can't produce one cheaply.
    - ``occurred_at`` is when the source detected the change, not when
      the event reached the runtime.
    """

    model_config = ConfigDict(frozen=True)

    kind: PageChangeKind
    page_id: str
    related_page_ids: tuple[str, ...] = Field(default_factory=tuple)
    edge_type: str | None = None
    source: str
    data_type: str | None = None
    scope_id: str | None = None
    reason: str = ""
    edit_diff: str = ""
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    extra: dict[str, object] = Field(default_factory=dict)

    @classmethod
    def page_invalidated(
        cls, *, page_id: str, source: str, reason: str = "",
        data_type: str | None = None, scope_id: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> "PageChangeEvent":
        return cls(
            kind=PageChangeKind.PAGE_INVALIDATED,
            page_id=page_id,
            source=source,
            data_type=data_type,
            scope_id=scope_id,
            reason=reason,
            extra=extra or {},
        )

    @classmethod
    def page_replaced(
        cls, *, old_page_id: str, new_page_id: str, source: str,
        edit_diff: str = "", data_type: str | None = None,
        scope_id: str | None = None, extra: dict[str, object] | None = None,
    ) -> "PageChangeEvent":
        return cls(
            kind=PageChangeKind.PAGE_REPLACED,
            page_id=new_page_id,
            related_page_ids=(old_page_id,),
            source=source,
            data_type=data_type,
            scope_id=scope_id,
            edit_diff=edit_diff,
            extra=extra or {},
        )

    @classmethod
    def page_added(
        cls, *, page_id: str, source: str,
        data_type: str | None = None, scope_id: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> "PageChangeEvent":
        return cls(
            kind=PageChangeKind.PAGE_ADDED,
            page_id=page_id,
            source=source,
            data_type=data_type,
            scope_id=scope_id,
            extra=extra or {},
        )

    @classmethod
    def page_graph_edge_added(
        cls, *, source_page_id: str, target_page_id: str, edge_type: str,
        source: str, scope_id: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> "PageChangeEvent":
        return cls(
            kind=PageChangeKind.PAGE_GRAPH_EDGE_ADDED,
            page_id=source_page_id,
            related_page_ids=(target_page_id,),
            edge_type=edge_type,
            source=source,
            scope_id=scope_id,
            extra=extra or {},
        )

    @classmethod
    def page_graph_edge_removed(
        cls, *, source_page_id: str, target_page_id: str, edge_type: str,
        source: str, scope_id: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> "PageChangeEvent":
        return cls(
            kind=PageChangeKind.PAGE_GRAPH_EDGE_REMOVED,
            page_id=source_page_id,
            related_page_ids=(target_page_id,),
            edge_type=edge_type,
            source=source,
            scope_id=scope_id,
            extra=extra or {},
        )

    def topic_key(self, source_id: str) -> str:
        """Return the canonical blackboard key under which this event
        should be written: ``vcm:page_events:<source_id>:<kind>``."""

        return f"{PAGE_EVENTS_TOPIC_PREFIX}:{source_id}:{self.kind.value}"


__all__ = (
    "PageChangeEvent",
    "PageChangeKind",
    "PAGE_EVENTS_TOPIC_PREFIX",
    "CONVERGENCE_STATUS_KEY",
    "CONVERGENCE_QUIESCENCE_TOPIC",
    "CONVERGENCE_CHANGE_FEED_KEY",
    "CONVERGENCE_DISPATCH_PREFIX",
)
