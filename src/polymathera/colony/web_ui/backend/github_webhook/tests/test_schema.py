"""Tests for ``schema.record_delivery`` — the dedup primitive."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest

from polymathera.colony.web_ui.backend.github_webhook.schema import (
    record_delivery,
)


pytestmark = pytest.mark.asyncio


class _FakeConn:
    def __init__(self, *, fetchrow_result: Any = None):
        self.fetchrow_result = fetchrow_result
        self.fetchrow_calls: list[tuple] = []

    async def fetchrow(self, sql: str, *args):
        self.fetchrow_calls.append((sql, args))
        return self.fetchrow_result


class _FakePool:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self._conn


async def test_record_delivery_returns_true_for_new_delivery() -> None:
    """asyncpg returns one row from the RETURNING clause when the
    INSERT actually happened (no conflict)."""

    conn = _FakeConn(fetchrow_result={"delivery_id": "uuid-1"})
    pool = _FakePool(conn)
    inserted = await record_delivery(
        pool, delivery_id="uuid-1", event_type="issues",
    )
    assert inserted is True


async def test_record_delivery_returns_false_on_duplicate() -> None:
    """``ON CONFLICT DO NOTHING`` + ``RETURNING`` yields zero rows
    when the row already existed (GitHub retry of an already-handled
    delivery). ``fetchrow`` returns ``None`` in that case."""

    conn = _FakeConn(fetchrow_result=None)
    pool = _FakePool(conn)
    inserted = await record_delivery(
        pool, delivery_id="uuid-1", event_type="issues",
    )
    assert inserted is False


async def test_record_delivery_passes_bind_variables_in_order() -> None:
    """Pin the SQL shape — guards against a future refactor swapping
    column order."""

    conn = _FakeConn(fetchrow_result={"delivery_id": "x"})
    pool = _FakePool(conn)
    await record_delivery(
        pool, delivery_id="uuid-1", event_type="issues",
    )
    sql, args = conn.fetchrow_calls[0]
    assert "INSERT INTO github_webhook_deliveries" in sql
    assert "ON CONFLICT (delivery_id) DO NOTHING" in sql
    assert "RETURNING delivery_id" in sql
    assert args == ("uuid-1", "issues")
