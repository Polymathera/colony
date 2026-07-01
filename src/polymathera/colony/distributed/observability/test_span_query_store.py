"""SQL-shape tests for ``SpanQueryStore`` helpers used by the recorder
pipeline, over a fake pool."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest

from polymathera.colony.distributed.observability.store import SpanQueryStore

pytestmark = pytest.mark.asyncio


class _FakeConn:
    def __init__(self, *, fetch_result=None, fetchrow_result=None):
        self.fetch_result = fetch_result or []
        self.fetchrow_result = fetchrow_result
        self.fetch_calls: list[tuple] = []
        self.fetchrow_calls: list[tuple] = []

    async def fetch(self, sql: str, *args):
        self.fetch_calls.append((sql, args))
        return self.fetch_result

    async def fetchrow(self, sql: str, *args):
        self.fetchrow_calls.append((sql, args))
        return self.fetchrow_result


class _FakePool:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self._conn


async def test_list_recent_run_refs_returns_trace_run_pairs() -> None:
    conn = _FakeConn(fetch_result=[
        {"trace_id": "t1", "run_id": "r1"},
        {"trace_id": "t1", "run_id": "r2"},
    ])
    refs = await SpanQueryStore(_FakePool(conn)).list_recent_run_refs(1000.0)
    assert refs == [
        {"trace_id": "t1", "run_id": "r1"},
        {"trace_id": "t1", "run_id": "r2"},
    ]
    assert conn.fetch_calls[0][1] == (1000.0, 1000)  # since_wall + default limit


async def test_get_latest_infer_span_id_returns_span() -> None:
    conn = _FakeConn(fetchrow_result={"span_id": "span-7"})
    span_id = await SpanQueryStore(_FakePool(conn)).get_latest_infer_span_id(
        "t1", "a1", 100.0,
    )
    assert span_id == "span-7"
    assert conn.fetchrow_calls[0][1] == ("t1", "a1", 100.0)


async def test_get_latest_infer_span_id_none_when_absent() -> None:
    conn = _FakeConn(fetchrow_result=None)
    span_id = await SpanQueryStore(_FakePool(conn)).get_latest_infer_span_id(
        "t1", "a1", 100.0,
    )
    assert span_id is None
