"""Tests for ``web_ui/backend/chat/system_session.py`` — the P8-0
system-session bootstrap module.

The full bootstrap path exercises the live ``SessionManagerDeployment``
(a Polymathera serving deployment — Ray Core actor under the hood,
NOT Ray Serve); these tests pin only the pure orchestration logic
in the module — that the helper hits the right manager endpoints in
the right order, skips spawn when the session already has an agent,
and walks Postgres colonies in the cross-tenant case.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from polymathera.colony.web_ui.backend.chat.system_session import (
    ensure_system_session_for_colony,
    ensure_system_sessions_for_all_colonies,
)


pytestmark = pytest.mark.asyncio


def _stub_colony_connection(
    *,
    db_pool: object | None = MagicMock(),
    is_connected: bool = True,
    session_manager=None,
    agent_id: str = "agent_xyz",
) -> MagicMock:
    """Build a ``ColonyConnection`` stub with the two context managers
    + ``get_session_manager`` that ``system_session.py`` calls into."""

    from contextlib import contextmanager
    from polymathera.colony.distributed.ray_utils.serving.context import (
        Ring,
        execution_context,
    )

    colony = MagicMock()
    colony._db_pool = db_pool
    colony.is_connected = is_connected
    colony.app_name = "test-app"

    # Real execution_context inside the stub so ``AgentMetadata``'s
    # ``require_execution_context`` default factory finds a context
    # when ``build_system_session_agent_blueprint`` runs.
    @contextmanager
    def _user_ctx(**kwargs):
        with execution_context(
            ring=Ring.USER,
            tenant_id=kwargs.get("tenant_id"),
            colony_id=kwargs.get("colony_id"),
            session_id=kwargs.get("session_id"),
            origin=kwargs.get("origin"),
        ):
            yield

    @contextmanager
    def _kernel_ctx(**kwargs):
        with execution_context(
            ring=Ring.KERNEL,
            tenant_id=kwargs.get("tenant_id"),
            colony_id=kwargs.get("colony_id"),
            origin=kwargs.get("origin"),
        ):
            yield

    colony.user_execution_context = _user_ctx
    colony.kernel_execution_context = _kernel_ctx
    colony.get_session_manager = AsyncMock(return_value=session_manager)
    # ``ensure_system_session_for_colony`` calls AgentHandle.from_blueprint
    # — monkeypatch that in each test that exercises the spawn path.
    return colony


async def test_ensure_skips_spawn_when_session_agent_already_set(
    monkeypatch,
) -> None:
    """When ``SessionManagerDeployment.ensure_system_session`` returns
    a session that already has a ``session_agent_id``, the helper
    must NOT spawn a new agent (idempotent on restart)."""

    sm = SimpleNamespace(
        ensure_system_session=AsyncMock(return_value=SimpleNamespace(
            session_id="session_sys", session_agent_id="agent_existing",
        )),
        set_session_agent_id=AsyncMock(),
    )
    colony = _stub_colony_connection(session_manager=sm)

    # If from_blueprint is called, this raises — guarding "did not spawn".
    from polymathera.colony.agents import AgentHandle
    monkeypatch.setattr(
        AgentHandle, "from_blueprint",
        AsyncMock(side_effect=AssertionError("must not spawn")),
    )

    session_id = await ensure_system_session_for_colony(
        colony, tenant_id="t1", colony_id="c1",
    )
    assert session_id == "session_sys"
    sm.ensure_system_session.assert_awaited_once()
    sm.set_session_agent_id.assert_not_awaited()


async def test_ensure_spawns_agent_when_session_agent_is_none(
    monkeypatch,
) -> None:
    """First-time bootstrap: ``ensure_system_session`` returns a
    session with ``session_agent_id=None``; the helper spawns the
    SessionAgent + claims the id via ``set_session_agent_id``."""

    sm = SimpleNamespace(
        ensure_system_session=AsyncMock(return_value=SimpleNamespace(
            session_id="session_sys", session_agent_id=None,
        )),
        set_session_agent_id=AsyncMock(return_value="agent_new"),
    )
    colony = _stub_colony_connection(session_manager=sm)

    from polymathera.colony.agents import AgentHandle
    monkeypatch.setattr(
        AgentHandle, "from_blueprint",
        AsyncMock(return_value=SimpleNamespace(agent_id="agent_new")),
    )

    session_id = await ensure_system_session_for_colony(
        colony, tenant_id="t1", colony_id="c1",
    )
    assert session_id == "session_sys"
    sm.set_session_agent_id.assert_awaited_once_with(
        session_id="session_sys", agent_id="agent_new",
    )


async def test_ensure_returns_none_on_manager_failure(
    monkeypatch,
) -> None:
    """Best-effort: if ``ensure_system_session`` raises, the helper
    logs + returns ``None`` so the surrounding walker can continue
    with other colonies."""

    sm = SimpleNamespace(
        ensure_system_session=AsyncMock(side_effect=RuntimeError("nope")),
        set_session_agent_id=AsyncMock(),
    )
    colony = _stub_colony_connection(session_manager=sm)

    result = await ensure_system_session_for_colony(
        colony, tenant_id="t1", colony_id="c1",
    )
    assert result is None


async def test_walker_skips_when_db_pool_unavailable(
    monkeypatch,
) -> None:
    """``ensure_system_sessions_for_all_colonies`` short-circuits with
    a warning when the dashboard has no Postgres pool — used at
    startup before Postgres is configured."""

    colony = _stub_colony_connection(db_pool=None)
    await ensure_system_sessions_for_all_colonies(colony)
    # No exception = success; no manager calls either.
    colony.get_session_manager.assert_not_awaited()


async def test_walker_iterates_every_colony(monkeypatch) -> None:
    """The walker calls ``ensure_system_session_for_colony`` once per
    Postgres colony row, regardless of tenant. Per-colony failures
    don't stop the loop."""

    sm = SimpleNamespace(
        ensure_system_session=AsyncMock(return_value=SimpleNamespace(
            session_id="session_sys", session_agent_id="agent_x",
        )),
        set_session_agent_id=AsyncMock(),
    )
    colony = _stub_colony_connection(session_manager=sm)

    from polymathera.colony.web_ui.backend.auth import service as auth_service
    monkeypatch.setattr(
        auth_service, "list_all_colonies",
        AsyncMock(return_value=[
            {"colony_id": "c1", "tenant_id": "t1"},
            {"colony_id": "c2", "tenant_id": "t1"},
            {"colony_id": "c3", "tenant_id": "t2"},
        ]),
    )

    await ensure_system_sessions_for_all_colonies(colony)

    # One ensure_system_session call per colony row.
    assert sm.ensure_system_session.await_count == 3


async def test_walker_empty_db_logs_and_returns(monkeypatch) -> None:
    """No colonies in Postgres → no manager calls, no errors."""

    sm = SimpleNamespace(
        ensure_system_session=AsyncMock(),
        set_session_agent_id=AsyncMock(),
    )
    colony = _stub_colony_connection(session_manager=sm)

    from polymathera.colony.web_ui.backend.auth import service as auth_service
    monkeypatch.setattr(
        auth_service, "list_all_colonies", AsyncMock(return_value=[]),
    )

    await ensure_system_sessions_for_all_colonies(colony)
    sm.ensure_system_session.assert_not_awaited()
