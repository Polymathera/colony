"""Tests for ``SourcePollWatcher`` against an in-memory ``ContextPageSource``."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from polymathera.colony.distributed.ray_utils.serving import (
    Ring,
    execution_context,
)
from polymathera.colony.vcm.models import ContextPageId, MmapConfig
from polymathera.colony.vcm.sources.context_page_source import ContextPageSource
from polymathera.colony.vcm.watchers import SourcePollWatcher, SourcePollWatcherConfig


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _exec_context():
    """Provide an execution context required by ``ContextPageSource.__init__``."""

    with execution_context(
        ring=Ring.USER, colony_id="c", tenant_id="t",
        session_id="sess", run_id="run", origin="test",
    ) as ctx:
        yield ctx


class _FakeSource(ContextPageSource):
    """Minimal ContextPageSource backed by a mutable dict."""

    def __init__(self, scope_id: str, mmap_config: MmapConfig) -> None:
        super().__init__(scope_id=scope_id, mmap_config=mmap_config)
        self._mapping: dict[ContextPageId, list[str]] = {}

    def set_mapping(self, mapping: dict[ContextPageId, list[str]]) -> None:
        self._mapping = {k: list(v) for k, v in mapping.items()}

    async def initialize(self) -> None:
        pass

    async def claim_orphaned_events(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def get_page_id_for_record(self, record_id: str) -> ContextPageId | None:
        for page_id, records in self._mapping.items():
            if record_id in records:
                return page_id
        return None

    async def get_record_ids_for_page(
        self, page_id: ContextPageId,
    ) -> list[str]:
        return list(self._mapping.get(page_id, ()))

    async def get_all_mapped_records(self) -> dict[str, ContextPageId]:
        out: dict[str, ContextPageId] = {}
        for page_id, records in self._mapping.items():
            for r in records:
                out[r] = page_id
        return out

    async def get_all_mapped_pages(self) -> dict[ContextPageId, list[str]]:
        return {k: list(v) for k, v in self._mapping.items()}


@pytest.fixture
def fake_source() -> _FakeSource:
    cfg = MmapConfig()
    return _FakeSource("scope-1", cfg)


async def _consume(watcher: SourcePollWatcher, n: int, timeout: float = 1.5):
    out: list = []
    async def _runner():
        async for event in watcher.watch():
            out.append(event)
            if len(out) >= n:
                watcher.stop()
                break
    await asyncio.wait_for(_runner(), timeout=timeout)
    return out


async def test_detects_added_pages(fake_source: _FakeSource) -> None:
    fake_source.set_mapping({"p1": ["r1"]})
    watcher = SourcePollWatcher(
        source=fake_source,
        scope_id="scope-1",
        source_uri="custom:test",
        config=SourcePollWatcherConfig(poll_interval_s=0.05),
    )

    async def mutator():
        await asyncio.sleep(0.1)
        fake_source.set_mapping({"p1": ["r1"], "p2": ["r2"]})

    mutator_task = asyncio.create_task(mutator())
    events = await _consume(watcher, n=1, timeout=1.0)
    await mutator_task
    assert events
    assert events[0].kind.value == "page_added"
    assert events[0].page_id == "p2"


async def test_detects_removed_pages(fake_source: _FakeSource) -> None:
    fake_source.set_mapping({"p1": ["r1"], "p2": ["r2"]})
    watcher = SourcePollWatcher(
        source=fake_source,
        scope_id="scope-1",
        source_uri="custom:test",
        config=SourcePollWatcherConfig(poll_interval_s=0.05),
    )

    async def mutator():
        await asyncio.sleep(0.1)
        fake_source.set_mapping({"p1": ["r1"]})

    mutator_task = asyncio.create_task(mutator())
    events = await _consume(watcher, n=1, timeout=1.0)
    await mutator_task
    assert events[0].kind.value == "page_invalidated"
    assert events[0].page_id == "p2"


async def test_detects_changed_record_set(fake_source: _FakeSource) -> None:
    fake_source.set_mapping({"p1": ["r1"]})
    watcher = SourcePollWatcher(
        source=fake_source,
        scope_id="scope-1",
        source_uri="custom:test",
        config=SourcePollWatcherConfig(poll_interval_s=0.05),
    )

    async def mutator():
        await asyncio.sleep(0.1)
        fake_source.set_mapping({"p1": ["r1", "r2"]})

    mutator_task = asyncio.create_task(mutator())
    events = await _consume(watcher, n=1, timeout=1.0)
    await mutator_task
    assert events[0].kind.value == "page_replaced"
    assert "r2" in events[0].extra["added_records"]
