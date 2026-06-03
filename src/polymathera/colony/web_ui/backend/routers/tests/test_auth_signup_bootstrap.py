"""Unit test for the ``POST /auth/signup`` → ``provision_colony``
wiring.

Every code path that lands a colony row MUST go through
``services.colony_lifecycle.provision_colony`` (which composes the
SQL insert with the per-colony system ``SessionAgent`` bootstrap).
Signup is one of two such paths; the other is ``POST /colonies/``.

This test pins the contract: signup invokes ``provision_colony`` for
the default colony, with ``is_default=True`` and the tenant id that
``create_user`` returned. The bootstrap composition itself is
exercised in ``test_colony_lifecycle.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_signup_provisions_default_colony(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After ``create_user`` returns (user_id, tenant_id), signup must
    call ``provision_colony`` for the default colony — that's the
    only path that bootstraps the system session."""

    from polymathera.colony.web_ui.backend.routers import auth as auth_router

    provision = AsyncMock(return_value={
        "colony_id": "colony_default_xyz",
        "name": "Default",
        "tenant_id": "tenant_abc",
    })
    monkeypatch.setattr(
        "polymathera.colony.web_ui.backend.services.colony_lifecycle."
        "provision_colony",
        provision,
    )

    monkeypatch.setattr(
        auth_router.auth_service,
        "create_user",
        AsyncMock(return_value={
            "user_id": "user_abc",
            "tenant_id": "tenant_abc",
        }),
    )
    monkeypatch.setattr(
        auth_router.auth_service, "create_access_token",
        lambda *a, **k: "access.jwt",
    )
    monkeypatch.setattr(
        auth_router.auth_service, "create_refresh_token",
        lambda *a, **k: "refresh.jwt",
    )

    colony = SimpleNamespace(_db_pool=object())
    request = auth_router.SignupRequest(username="alice", password="secret123")
    response = SimpleNamespace(set_cookie=lambda **kw: None)

    result = await auth_router.signup(
        request=request, response=response, colony=colony,
    )

    provision.assert_awaited_once_with(
        colony,
        tenant_id="tenant_abc",
        name="Default",
        description="Auto-created default workspace",
        is_default=True,
    )
    assert result.default_colony_id == "colony_default_xyz"
    assert result.tenant_id == "tenant_abc"
