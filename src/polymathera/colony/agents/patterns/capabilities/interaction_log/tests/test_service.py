"""Tests for ``service.insert_event`` / ``fetch_recent_activity`` /
``fetch_by_ref`` — pure SQL-shape tests against a fake asyncpg pool.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import pytest

from polymathera.colony.agents.patterns.capabilities.interaction_log.service import (
    fetch_by_ref,
    fetch_recent_activity,
    insert_event,
)


pytestmark = pytest.mark.asyncio


class _FakeConn:
    def __init__(
        self, *,
        fetch_result: list[dict[str, Any]] | None = None,
        fetchrow_result: dict[str, Any] | None = None,
    ):
        self.fetch_result = fetch_result or []
        self.fetchrow_result = fetchrow_result
        self.fetch_calls: list[tuple] = []
        self.fetchrow_calls: list[tuple] = []
        self.execute_calls: list[tuple] = []

    async def fetch(self, sql: str, *args):
        self.fetch_calls.append((sql, args))
        return self.fetch_result

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


async def test_insert_event_returns_new_id_and_passes_required_columns() -> None:
    """Insert returns the new ``id``; required columns are bound in
    the right positions."""

    conn = _FakeConn(fetchrow_result={"id": 42})
    pool = _FakePool(conn)
    new_id = await insert_event(
        pool,
        tenant_id="t1", colony_id="c1",
        channel="github", event_kind="github_issue_event",
        payload={"x": 1},
        refs=[{"kind": "issue", "value": "acme/widgets#7"}],
        channel_ref="https://github.com/acme/widgets/issues/7",
        user_login="alice",
    )
    assert new_id == 42

    sql, args = conn.fetchrow_calls[0]
    assert "INSERT INTO interaction_log" in sql
    assert "RETURNING id" in sql
    # Required + nullable args in declared order:
    # (tenant_id, colony_id, session_id, run_id, user_login,
    #  channel, channel_ref, event_kind, payload, refs)
    assert args[0] == "t1"
    assert args[1] == "c1"
    assert args[2] is None  # session_id
    assert args[3] is None  # run_id
    assert args[4] == "alice"
    assert args[5] == "github"
    assert args[6] == "https://github.com/acme/widgets/issues/7"
    assert args[7] == "github_issue_event"
    assert json.loads(args[8]) == {"x": 1}
    assert json.loads(args[9]) == [
        {"kind": "issue", "value": "acme/widgets#7"},
    ]


async def test_insert_event_defaults_empty_refs() -> None:
    """``refs=None`` → empty list ``[]`` in JSONB so the column's
    ``NOT NULL DEFAULT '[]'`` invariant is satisfied at the
    application level too."""

    conn = _FakeConn(fetchrow_result={"id": 1})
    pool = _FakePool(conn)
    await insert_event(
        pool,
        tenant_id="t1", colony_id="c1",
        channel="github", event_kind="x", payload={},
    )
    _, args = conn.fetchrow_calls[0]
    assert json.loads(args[9]) == []


async def test_insert_event_jsonb_encoder_handles_datetime() -> None:
    """Payload values that aren't native-JSON (e.g., datetime) are
    coerced via ``default=str`` so a single bad value can't crash
    the insert."""

    conn = _FakeConn(fetchrow_result={"id": 1})
    pool = _FakePool(conn)
    when = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    await insert_event(
        pool,
        tenant_id="t1", colony_id="c1",
        channel="github", event_kind="x",
        payload={"ts": when},
    )
    _, args = conn.fetchrow_calls[0]
    decoded = json.loads(args[8])
    assert isinstance(decoded["ts"], str)
    assert "2026-06-01" in decoded["ts"]


async def test_fetch_recent_activity_requires_tenant_and_colony() -> None:
    """The WHERE clause binds tenant_id + colony_id as $1 and $2."""

    conn = _FakeConn(fetch_result=[])
    pool = _FakePool(conn)
    await fetch_recent_activity(
        pool, tenant_id="t1", colony_id="c1", limit=10,
    )
    sql, args = conn.fetch_calls[0]
    assert "tenant_id = $1" in sql
    assert "colony_id = $2" in sql
    assert args[0] == "t1"
    assert args[1] == "c1"


async def test_fetch_recent_activity_user_login_filter_optional() -> None:
    """``user_login=None`` skips the filter; setting it adds it as
    the next bind arg."""

    conn = _FakeConn(fetch_result=[])
    pool = _FakePool(conn)
    await fetch_recent_activity(
        pool, tenant_id="t1", colony_id="c1",
        user_login="alice", limit=10,
    )
    sql, args = conn.fetch_calls[0]
    assert "user_login = $3" in sql
    assert args[2] == "alice"


async def test_fetch_recent_activity_decodes_jsonb_columns() -> None:
    """asyncpg returns JSONB as strings by default — the helper
    decodes ``payload`` + ``refs`` back to Python objects."""

    conn = _FakeConn(fetch_result=[{
        "id": 1, "ts": None,
        "tenant_id": "t1", "colony_id": "c1",
        "session_id": None, "run_id": None, "user_login": None,
        "channel": "github", "channel_ref": None,
        "event_kind": "x",
        "payload": '{"a": 1}',
        "refs": '[{"kind": "issue", "value": "r#1"}]',
    }])
    pool = _FakePool(conn)
    rows = await fetch_recent_activity(
        pool, tenant_id="t1", colony_id="c1",
    )
    assert rows[0]["payload"] == {"a": 1}
    assert rows[0]["refs"] == [{"kind": "issue", "value": "r#1"}]


async def test_fetch_by_ref_uses_gin_contains_query() -> None:
    """``refs @> $3::jsonb`` is the GIN-indexed contains query — the
    ``jsonb_path_ops`` opclass lets the planner use the index."""

    conn = _FakeConn(fetch_result=[])
    pool = _FakePool(conn)
    await fetch_by_ref(
        pool, tenant_id="t1", colony_id="c1",
        ref_kind="issue", ref_value="acme/widgets#7",
    )
    sql, args = conn.fetch_calls[0]
    assert "refs @> $3::jsonb" in sql
    needle = json.loads(args[2])
    assert needle == [{"kind": "issue", "value": "acme/widgets#7"}]
