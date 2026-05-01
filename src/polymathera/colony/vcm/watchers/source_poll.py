"""``SourcePollWatcher`` — generic interval poll over any source.

Wraps a ``ContextPageSource`` that does not have a native ``watch()``
implementation but does expose a stable ``get_all_mapped_pages``
snapshot. The watcher snapshots periodically and emits the difference
as ``PageChangeEvent``s. This is the catch-all transport for sources
backed by API endpoints (arXiv RSS, PubMed E-utilities, supplier
catalogues) where the upstream system has no push notification.

Implementation:

1. Capture the source's ``get_all_mapped_pages`` snapshot.
2. Sleep ``poll_interval_s``.
3. Capture again; diff against the prior snapshot.
4. For each new page id: emit ``PageAdded``.
5. For each removed page id: emit ``PageInvalidated``.
6. For each retained page id where the record set changed (a different
   set of records mapped to the same page): emit ``PageReplaced``.

Sources whose snapshot is large (millions of pages) should not use
this watcher; the doc explicitly recommends webhook + git transport
for such cases. The polling watcher is for the long tail.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..page_events import PageChangeEvent
from ..sources.context_page_source import ContextPageSource


logger = logging.getLogger(__name__)


@dataclass
class SourcePollWatcherConfig:
    poll_interval_s: float = 60.0
    data_type: str | None = None


class SourcePollWatcher:
    """Generic poll-driven watcher for any ``ContextPageSource``."""

    static = False

    def __init__(
        self,
        *,
        source: ContextPageSource,
        scope_id: str,
        source_uri: str,
        config: SourcePollWatcherConfig | None = None,
    ) -> None:
        self._source = source
        self._scope_id = scope_id
        self._source_uri = source_uri
        self._config = config or SourcePollWatcherConfig()
        self._stopped = asyncio.Event()

    @property
    def scope_id(self) -> str:
        return self._scope_id

    def stop(self) -> None:
        self._stopped.set()

    async def watch(self) -> AsyncIterator[PageChangeEvent]:
        previous = await self._snapshot()
        while not self._stopped.is_set():
            try:
                await asyncio.wait_for(
                    self._stopped.wait(),
                    timeout=self._config.poll_interval_s,
                )
                break
            except asyncio.TimeoutError:
                pass
            try:
                current = await self._snapshot()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "SourcePollWatcher: snapshot failed for %s",
                    type(self._source).__name__,
                )
                continue
            for event in self._diff_to_events(previous, current):
                yield event
            previous = current

    async def _snapshot(self) -> dict[str, frozenset[str]]:
        mapping = await self._source.get_all_mapped_pages()
        return {
            page_id: frozenset(records or ())
            for page_id, records in mapping.items()
        }

    def _diff_to_events(
        self,
        old: dict[str, frozenset[str]],
        new: dict[str, frozenset[str]],
    ) -> list[PageChangeEvent]:
        out: list[PageChangeEvent] = []
        for page_id in new.keys() - old.keys():
            out.append(
                PageChangeEvent.page_added(
                    page_id=page_id,
                    source=self._source_uri,
                    data_type=self._config.data_type,
                    scope_id=self._scope_id,
                )
            )
        for page_id in old.keys() - new.keys():
            out.append(
                PageChangeEvent.page_invalidated(
                    page_id=page_id,
                    source=self._source_uri,
                    reason="source no longer maps any record to this page",
                    data_type=self._config.data_type,
                    scope_id=self._scope_id,
                )
            )
        for page_id in old.keys() & new.keys():
            if old[page_id] != new[page_id]:
                out.append(
                    PageChangeEvent.page_replaced(
                        old_page_id=page_id,
                        new_page_id=page_id,
                        source=self._source_uri,
                        data_type=self._config.data_type,
                        scope_id=self._scope_id,
                        extra={
                            "added_records": sorted(new[page_id] - old[page_id]),
                            "removed_records": sorted(old[page_id] - new[page_id]),
                        },
                    )
                )
        return out


__all__ = ("SourcePollWatcher", "SourcePollWatcherConfig")
