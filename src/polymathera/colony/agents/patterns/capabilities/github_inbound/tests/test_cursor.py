"""Tests for ``cursor.get_cursor`` / ``cursor.bump_cursor``.

Pure SQL-shape tests against a fake asyncpg-like pool — no real
Postgres needed. The fake records every ``fetchrow`` + ``execute``
call and lets the test assert on the query shape + bind variables.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import pytest

from polymathera.colony.agents.patterns.capabilities.github_inbound.cursor import (
    Cursor,
    bump_cursor,
    get_cursor,
)


pytestmark = pytest.mark.asyncio


class _FakeConn:
    """Minimal asyncpg-shaped connection that records calls."""

    def __init__(self, *, fetchrow_result: Any = None):
        self.fetchrow_result = fetchrow_result
        self.fetchrow_calls: list[tuple] = []
        self.execute_calls: list[tuple] = []

    async def fetchrow(self, sql: str, *args):
        self.fetchrow_calls.append((sql, args))
        return self.fetchrow_result

    async def execute(self, sql: str, *args):
        self.execute_calls.append((sql, args))


class _FakePool:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self._conn


async def test_get_cursor_returns_default_when_row_absent() -> None:
    """No row → 2020-01-01 sentinel + ``last_seen_id=None``. The
    default bounds first-tick load on busy repos."""

    pool = _FakePool(_FakeConn(fetchrow_result=None))
    cursor = await get_cursor(
        pool, tenant_id="t1", colony_id="c1",
        repo="acme/widgets", channel="issues",
    )
    assert cursor.tenant_id == "t1"
    assert cursor.colony_id == "c1"
    assert cursor.repo == "acme/widgets"
    assert cursor.channel == "issues"
    assert cursor.last_updated == datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert cursor.last_seen_id is None


async def test_get_cursor_returns_row_when_present() -> None:
    """Row present → values from row."""

    when = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    pool = _FakePool(_FakeConn(fetchrow_result={
        "last_updated": when,
        "last_seen_id": "I_kwDOissue123",
    }))
    cursor = await get_cursor(
        pool, tenant_id="t1", colony_id="c1",
        repo="acme/widgets", channel="issues",
    )
    assert cursor.last_updated == when
    assert cursor.last_seen_id == "I_kwDOissue123"


async def test_get_cursor_passes_pk_bind_variables() -> None:
    """The fetchrow query must include all four PK columns as bind
    variables — guard against a future refactor accidentally
    dropping tenant_id/colony_id."""

    conn = _FakeConn(fetchrow_result=None)
    pool = _FakePool(conn)
    await get_cursor(
        pool, tenant_id="t1", colony_id="c1",
        repo="acme/widgets", channel="issues",
    )
    sql, args = conn.fetchrow_calls[0]
    assert "tenant_id = $1" in sql
    assert "colony_id = $2" in sql
    assert "repo = $3" in sql
    assert "channel = $4" in sql
    assert args == ("t1", "c1", "acme/widgets", "issues")


async def test_bump_cursor_upserts_with_on_conflict() -> None:
    """``bump_cursor`` runs an INSERT … ON CONFLICT DO UPDATE so the
    first tick (no row) and subsequent ticks (row exists) both work
    via the same path."""

    conn = _FakeConn()
    pool = _FakePool(conn)
    when = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    await bump_cursor(
        pool, tenant_id="t1", colony_id="c1",
        repo="acme/widgets", channel="issues",
        last_updated=when, last_seen_id="I_kwDOissue123",
    )
    assert len(conn.execute_calls) == 1
    sql, args = conn.execute_calls[0]
    assert "INSERT INTO github_poll_cursors" in sql
    assert "ON CONFLICT (tenant_id, colony_id, repo, channel)" in sql
    assert "DO UPDATE SET" in sql
    assert args == (
        "t1", "c1", "acme/widgets", "issues", when, "I_kwDOissue123",
    )


def test_cursor_is_frozen_dataclass() -> None:
    """Cursor is frozen — protects against accidental mutation in the
    poller (the cursor is read once per tick, used for filtering,
    then a NEW cursor row is upserted)."""

    cursor = Cursor(
        tenant_id="t", colony_id="c", repo="a/b", channel="issues",
        last_updated=datetime(2020, 1, 1, tzinfo=timezone.utc),
        last_seen_id=None,
    )
    with pytest.raises((AttributeError, Exception)):
        cursor.repo = "x/y"  # type: ignore[misc]
