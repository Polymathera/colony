"""Process-level Postgres pool for agent-side capabilities.

Capabilities that need direct SQL (P8a `GitHubInboundCapability`'s
cursor table, P8b `InteractionLogCapability`'s write-through, …)
CAN'T receive a live `asyncpg.Pool` through the
`AgentCapability.bind(...)` chain — pools wrap open socket connections
and fail `validate_serializable()` at cloudpickle time. The right
shape is lazy acquisition inside the capability's `initialize()`
via this module's singleton helper.

The helper reads the same `RDS_*` env vars `WebUIConfig` reads —
all five are set on every ray-worker container by
`docker-compose.yml`. Cached per-process so multiple capabilities
on the same worker share one pool.

History: introduced at P8a inline in
`agents/patterns/capabilities/github_inbound/capability.py`,
extracted here at P8b when `InteractionLogCapability` became the
second consumer.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any


logger = logging.getLogger(__name__)


_agent_db_pool: Any = None
_agent_db_pool_lock = asyncio.Lock()


async def get_agent_db_pool() -> Any:
    """Return the process-level `asyncpg.Pool`, creating it on first
    call. Concurrent first-callers funnel through the lock so only
    one pool is created per process.

    Raises whatever `asyncpg.create_pool` raises on failure — typically
    `ConnectionRefusedError` when Postgres is unreachable or
    `asyncpg.InvalidPasswordError` when `RDS_PASSWORD` is wrong.
    Callers handle the exception (most quiesce + WARN, since
    capability mounting should not crash the agent).
    """

    global _agent_db_pool
    if _agent_db_pool is not None:
        return _agent_db_pool
    async with _agent_db_pool_lock:
        if _agent_db_pool is not None:
            return _agent_db_pool
        import asyncpg
        _agent_db_pool = await asyncpg.create_pool(
            host=os.environ.get("RDS_HOST", "postgres"),
            port=int(os.environ.get("RDS_PORT", "5432")),
            user=os.environ.get("RDS_USER", "colony"),
            password=os.environ.get("RDS_PASSWORD", ""),
            database=os.environ.get("RDS_DB_NAME", "colony"),
            min_size=1, max_size=5,
        )
        return _agent_db_pool


async def reset_agent_db_pool_for_tests() -> None:
    """Drop the cached pool — only for test teardown to keep one test
    file's pool from leaking into the next. Production code never
    calls this."""

    global _agent_db_pool
    if _agent_db_pool is not None:
        try:
            await _agent_db_pool.close()
        except Exception:  # noqa: BLE001
            pass
        _agent_db_pool = None


__all__ = ("get_agent_db_pool", "reset_agent_db_pool_for_tests")
