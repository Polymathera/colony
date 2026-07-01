"""SQL-shape tests for :class:`SpanFeedbackStore` over a fake pool."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import pytest

from polymathera.colony.distributed.observability.feedback import (
    SpanFeedbackRatingError,
    SpanFeedbackStore,
)

pytestmark = pytest.mark.asyncio


class _FakeConn:
    def __init__(self, *, fetch_result: list[dict[str, Any]] | None = None):
        self.fetch_result = fetch_result or []
        self.execute_calls: list[tuple] = []
        self.fetch_calls: list[tuple] = []

    async def execute(self, sql: str, *args):
        self.execute_calls.append((sql, args))

    async def fetch(self, sql: str, *args):
        self.fetch_calls.append((sql, args))
        return self.fetch_result


class _FakePool:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self._conn


async def test_record_upserts_with_bound_columns() -> None:
    conn = _FakeConn()
    await SpanFeedbackStore(_FakePool(conn)).record(
        trace_id="t1", span_id="s1", author="u1", rating="up", note="good step",
    )
    assert len(conn.execute_calls) == 1
    sql, args = conn.execute_calls[0]
    assert "ON CONFLICT (span_id, author) DO UPDATE" in sql
    # Positional bind order: span_id, trace_id, author, rating, note.
    assert args == ("s1", "t1", "u1", "up", "good step")


async def test_record_rejects_unknown_rating() -> None:
    conn = _FakeConn()
    with pytest.raises(SpanFeedbackRatingError):
        await SpanFeedbackStore(_FakePool(conn)).record(
            trace_id="t1", span_id="s1", author="u1", rating="meh",
        )
    assert conn.execute_calls == []


async def test_get_for_trace_groups_by_span_and_floats_timestamp() -> None:
    ts = datetime(2026, 6, 29, 12, 0, tzinfo=timezone.utc)
    conn = _FakeConn(fetch_result=[
        {"span_id": "s1", "author": "u1", "rating": "up", "note": None, "updated_wall": ts},
        {"span_id": "s1", "author": "u2", "rating": "down", "note": "x", "updated_wall": ts},
        {"span_id": "s2", "author": "u1", "rating": "up", "note": None, "updated_wall": ts},
    ])
    grouped = await SpanFeedbackStore(_FakePool(conn)).get_for_trace("t1")
    assert set(grouped) == {"s1", "s2"}
    assert len(grouped["s1"]) == 2
    assert grouped["s1"][0] == {
        "author": "u1", "rating": "up", "note": None, "updated_wall": ts.timestamp(),
    }
    assert conn.fetch_calls[0][1] == ("t1",)
