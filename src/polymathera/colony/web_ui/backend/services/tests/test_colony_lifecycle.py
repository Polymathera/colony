"""Unit tests for :func:`provision_colony` — the single entry point
for landing a colony row.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from polymathera.colony.web_ui.backend.services import colony_lifecycle


@pytest.fixture
def _stub_session_manager_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default fixture: session_manager reports ready immediately."""
    monkeypatch.setattr(
        colony_lifecycle, "_wait_for_session_manager_ready",
        AsyncMock(return_value=True),
    )


@pytest.mark.asyncio
async def test_provision_colony_inserts_and_bootstraps(
    monkeypatch: pytest.MonkeyPatch, _stub_session_manager_ready: None,
) -> None:
    """Happy path: SQL insert succeeds, readiness wait passes, bootstrap runs."""
    create = AsyncMock(return_value={
        "colony_id": "colony_xyz", "name": "Workspace 1", "tenant_id": "tenant_abc",
    })
    monkeypatch.setattr(colony_lifecycle.auth_service, "create_colony", create)
    bootstrap = AsyncMock(return_value="session_systemxxxx")
    monkeypatch.setattr(
        colony_lifecycle, "ensure_system_session_for_colony", bootstrap,
    )

    colony = SimpleNamespace(_db_pool=object())
    result = await colony_lifecycle.provision_colony(
        colony, tenant_id="tenant_abc", name="Workspace 1",
        description="hi",
    )

    create.assert_awaited_once_with(
        colony._db_pool, tenant_id="tenant_abc", name="Workspace 1",
        description="hi",
        vcs_repo_id=None, vcs_repo_full_name=None, default_branch=None,
    )
    bootstrap.assert_awaited_once_with(
        colony, tenant_id="tenant_abc", colony_id="colony_xyz",
    )
    assert result["colony_id"] == "colony_xyz"


@pytest.mark.asyncio
async def test_provision_colony_skips_bootstrap_when_session_manager_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the session_manager never becomes ready within budget, the
    colony row stays committed, the bootstrap is skipped, and the
    lifespan walker retries on next restart."""
    monkeypatch.setattr(
        colony_lifecycle, "_wait_for_session_manager_ready",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        colony_lifecycle.auth_service, "create_colony",
        AsyncMock(return_value={
            "colony_id": "colony_xyz", "name": "x", "tenant_id": "tenant_abc",
        }),
    )
    bootstrap = AsyncMock()
    monkeypatch.setattr(
        colony_lifecycle, "ensure_system_session_for_colony", bootstrap,
    )

    colony = SimpleNamespace(_db_pool=object())
    result = await colony_lifecycle.provision_colony(
        colony, tenant_id="tenant_abc", name="x",
    )
    assert result["colony_id"] == "colony_xyz"
    bootstrap.assert_not_awaited()


@pytest.mark.asyncio
async def test_provision_colony_tolerates_bootstrap_failure(
    monkeypatch: pytest.MonkeyPatch, _stub_session_manager_ready: None,
) -> None:
    """Bootstrap exception MUST NOT fail the colony create — the row
    is already committed; lifespan walker is the safety net."""
    monkeypatch.setattr(
        colony_lifecycle.auth_service, "create_colony",
        AsyncMock(return_value={
            "colony_id": "colony_xyz", "name": "x", "tenant_id": "tenant_abc",
        }),
    )
    monkeypatch.setattr(
        colony_lifecycle, "ensure_system_session_for_colony",
        AsyncMock(side_effect=RuntimeError("ray cluster down")),
    )

    colony = SimpleNamespace(_db_pool=object())
    result = await colony_lifecycle.provision_colony(
        colony, tenant_id="tenant_abc", name="x",
    )
    assert result["colony_id"] == "colony_xyz"


@pytest.mark.asyncio
async def test_provision_colony_rejects_missing_db_pool() -> None:
    """No db_pool ⇒ surface a clean RuntimeError, not an AttributeError."""
    colony = SimpleNamespace(_db_pool=None)
    with pytest.raises(RuntimeError, match="db_pool"):
        await colony_lifecycle.provision_colony(
            colony, tenant_id="tenant_abc", name="x",
        )
