"""``PageEventPublisher`` — small helper to publish ``PageChangeEvent``s
onto the colony scope's ``vcm:page_events:*`` topic.

Watchers do not interact with the convergence runtime directly; they
publish events on the blackboard, and the runtime's forwarder picks
them up. This decouples watchers from runtime initialisation order
and lets multiple watchers (one per source) operate independently.

The publisher takes an ``EnhancedBlackboard`` scoped to the colony
and the source-id label that prefixes the topic key.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..page_events import PageChangeEvent

if TYPE_CHECKING:
    from ...agents.blackboard import EnhancedBlackboard

logger = logging.getLogger(__name__)


class PageEventPublisher:
    """Writes ``PageChangeEvent``s to the colony's blackboard."""

    def __init__(self, blackboard: "EnhancedBlackboard", source_id: str) -> None:
        self._blackboard = blackboard
        self._source_id = source_id

    @property
    def source_id(self) -> str:
        return self._source_id

    async def publish(self, event: PageChangeEvent) -> None:
        key = event.topic_key(self._source_id)
        try:
            await self._blackboard.write(
                key,
                value=event.model_dump(mode="json"),
                tags={"vcm", "page_event", event.kind.value},
                metadata={
                    "source_id": self._source_id,
                    "page_id": event.page_id,
                    "kind": event.kind.value,
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "PageEventPublisher: failed to write %s for page %s",
                event.kind.value, event.page_id,
            )

    async def publish_many(self, events: list[PageChangeEvent]) -> None:
        for event in events:
            await self.publish(event)


__all__ = ("PageEventPublisher",)
