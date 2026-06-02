"""Tests for the ``POST /api/v1/github/webhook`` route.

Covers the HMAC + dedup + tenant lookup + dispatch pipeline against
a real FastAPI TestClient + stubbed dashboard dependencies. Goes
through the actual route handler — the failure mode here is the
ONE that would matter in production (HMAC wrong, route rejects;
duplicate delivery, 200 with status).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


WEBHOOK_SECRET = "test_secret_xyz"


def _sign(body: bytes, secret: str = WEBHOOK_SECRET) -> str:
    """Build the ``X-Hub-Signature-256`` header value GitHub would send."""
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _issues_payload() -> dict:
    return {
        "action": "opened",
        "issue": {
            "number": 42,
            "state": "open",
            "title": "found a bug",
            "body": "reproduces every time",
            "user": {"login": "alice"},
            "created_at": "2026-06-01T10:00:00Z",
            "updated_at": "2026-06-01T10:00:00Z",
            "closed_at": None,
            "html_url": "https://github.com/acme/widgets/issues/42",
        },
        "repository": {"full_name": "acme/widgets"},
        "installation": {"id": 12345},
    }


def _build_app(
    *,
    webhook_secret: str = WEBHOOK_SECRET,
    record_delivery_returns: bool = True,
    tenant_lookup_returns: dict | None = None,
    colonies_for_tenant: list[dict] | None = None,
    publish_returns: int = 1,
    db_pool_available: bool = True,
) -> tuple[FastAPI, dict[str, Any]]:
    """Build a minimal FastAPI app with the github_webhook router +
    every dependency mocked. Returns ``(app, capture)`` where
    ``capture`` records publisher calls so tests can assert on them.
    """

    from polymathera.colony.web_ui.backend.dependencies import get_colony
    from polymathera.colony.web_ui.backend.routers import github_webhook

    capture: dict[str, Any] = {"publish_calls": []}

    app = FastAPI()
    app.include_router(github_webhook.router, prefix="/api/v1")

    # Stub colony dependency: provide a ColonyConnection-shaped object
    # with .app_name + ._db_pool (the route's two reads).
    fake_colony = SimpleNamespace(
        app_name="test-app",
        _db_pool=object() if db_pool_available else None,
    )

    async def _get_colony_override():
        return fake_colony

    app.dependency_overrides[get_colony] = _get_colony_override

    # Stub the four module-level callables the route invokes:
    # - get_github_auth_config (returns webhook_secret)
    # - record_delivery (dedup)
    # - auth_service.get_tenant_by_installation_id
    # - auth_service.list_colonies
    # - publish_to_tenant_colonies
    from polymathera.colony.agents import configs as agents_configs
    from polymathera.colony.web_ui.backend.auth import (
        service as auth_service,
    )
    from polymathera.colony.web_ui.backend.github_webhook import (
        publisher as publisher_mod,
        schema as schema_mod,
    )

    # Stash originals so the test class teardown can restore them.
    capture["_originals"] = {
        "get_github_auth_config": agents_configs.get_github_auth_config,
        "record_delivery": schema_mod.record_delivery,
        "get_tenant_by_installation_id": (
            auth_service.get_tenant_by_installation_id
        ),
        "list_colonies": auth_service.list_colonies,
        "publish_to_tenant_colonies": (
            publisher_mod.publish_to_tenant_colonies
        ),
    }

    async def _fake_gh_auth():
        return SimpleNamespace(webhook_secret=webhook_secret)

    agents_configs.get_github_auth_config = _fake_gh_auth  # type: ignore[assignment]
    schema_mod.record_delivery = AsyncMock(  # type: ignore[assignment]
        return_value=record_delivery_returns,
    )
    auth_service.get_tenant_by_installation_id = AsyncMock(  # type: ignore[assignment]
        return_value=tenant_lookup_returns,
    )
    auth_service.list_colonies = AsyncMock(  # type: ignore[assignment]
        return_value=colonies_for_tenant or [],
    )

    async def _fake_publish(**kwargs):
        capture["publish_calls"].append(kwargs)
        return publish_returns

    publisher_mod.publish_to_tenant_colonies = _fake_publish  # type: ignore[assignment]

    # The route imports normalizer + publisher locally, so we also
    # need to patch the symbols on the router module if it imported
    # them by name. The router does:
    #   from ..github_webhook.publisher import publish_to_tenant_colonies
    #   from ..github_webhook.schema import record_delivery
    # so patches at the ROUTER module reach the binding the route
    # actually calls.
    from polymathera.colony.web_ui.backend.routers import (
        github_webhook as route_mod,
    )
    route_mod.record_delivery = schema_mod.record_delivery
    route_mod.publish_to_tenant_colonies = _fake_publish
    route_mod.auth_service = auth_service

    return app, capture


def _restore(capture: dict[str, Any]) -> None:
    from polymathera.colony.agents import configs as agents_configs
    from polymathera.colony.web_ui.backend.auth import (
        service as auth_service,
    )
    from polymathera.colony.web_ui.backend.github_webhook import (
        publisher as publisher_mod,
        schema as schema_mod,
    )
    from polymathera.colony.web_ui.backend.routers import (
        github_webhook as route_mod,
    )

    originals = capture["_originals"]
    agents_configs.get_github_auth_config = (
        originals["get_github_auth_config"]
    )
    schema_mod.record_delivery = originals["record_delivery"]
    auth_service.get_tenant_by_installation_id = (
        originals["get_tenant_by_installation_id"]
    )
    auth_service.list_colonies = originals["list_colonies"]
    publisher_mod.publish_to_tenant_colonies = (
        originals["publish_to_tenant_colonies"]
    )
    route_mod.record_delivery = originals["record_delivery"]
    route_mod.publish_to_tenant_colonies = (
        originals["publish_to_tenant_colonies"]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_receiver_503_when_secret_unset() -> None:
    """No ``GITHUB_WEBHOOK_SECRET`` configured → 503 with a
    setup-doc-pointer detail."""

    app, capture = _build_app(webhook_secret="")
    try:
        client = TestClient(app)
        body = json.dumps(_issues_payload()).encode()
        resp = client.post(
            "/api/v1/github/webhook",
            content=body,
            headers={
                "X-GitHub-Event": "issues",
                "X-GitHub-Delivery": "uuid-1",
                "X-Hub-Signature-256": _sign(body),
            },
        )
        assert resp.status_code == 503
        assert "GITHUB_WEBHOOK_SECRET" in resp.text
    finally:
        _restore(capture)


def test_receiver_401_on_bad_hmac() -> None:
    """Wrong signature → 401. No dedup write, no publish."""

    app, capture = _build_app()
    try:
        client = TestClient(app)
        body = json.dumps(_issues_payload()).encode()
        resp = client.post(
            "/api/v1/github/webhook",
            content=body,
            headers={
                "X-GitHub-Event": "issues",
                "X-GitHub-Delivery": "uuid-1",
                "X-Hub-Signature-256": "sha256=ffff",  # bogus
            },
        )
        assert resp.status_code == 401
        assert capture["publish_calls"] == []
    finally:
        _restore(capture)


def test_receiver_accepts_valid_delivery() -> None:
    """Happy path: HMAC valid + new delivery + tenant configured +
    colonies in tenant → 200 ``status: accepted`` + publish fired."""

    app, capture = _build_app(
        tenant_lookup_returns={"tenant_id": "t1"},
        colonies_for_tenant=[
            {"colony_id": "c1"},
            {"colony_id": "c2"},
        ],
        publish_returns=2,
    )
    try:
        client = TestClient(app)
        body = json.dumps(_issues_payload()).encode()
        resp = client.post(
            "/api/v1/github/webhook",
            content=body,
            headers={
                "X-GitHub-Event": "issues",
                "X-GitHub-Delivery": "uuid-2",
                "X-Hub-Signature-256": _sign(body),
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "status": "accepted", "colonies_written": "2",
        }
        assert len(capture["publish_calls"]) == 1
        call = capture["publish_calls"][0]
        assert call["tenant_id"] == "t1"
        assert call["colony_ids"] == ["c1", "c2"]
        assert call["delivery_id"] == "uuid-2"
    finally:
        _restore(capture)


def test_receiver_duplicate_delivery_returns_200_no_publish() -> None:
    """Retried delivery → ``record_delivery`` returns False → 200
    ``status: duplicate`` + no publish call."""

    app, capture = _build_app(record_delivery_returns=False)
    try:
        client = TestClient(app)
        body = json.dumps(_issues_payload()).encode()
        resp = client.post(
            "/api/v1/github/webhook",
            content=body,
            headers={
                "X-GitHub-Event": "issues",
                "X-GitHub-Delivery": "uuid-3",
                "X-Hub-Signature-256": _sign(body),
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "duplicate"}
        assert capture["publish_calls"] == []
    finally:
        _restore(capture)


def test_receiver_unhandled_event_type_returns_200_ignored() -> None:
    """``ping`` (or any non-v1 event type) → 200 ``status: ignored``
    so GitHub doesn't retry."""

    app, capture = _build_app()
    try:
        client = TestClient(app)
        body = b'{"zen": "Anything added dilutes everything else."}'
        resp = client.post(
            "/api/v1/github/webhook",
            content=body,
            headers={
                "X-GitHub-Event": "ping",
                "X-GitHub-Delivery": "uuid-ping",
                "X-Hub-Signature-256": _sign(body),
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "ignored", "event": "ping"}
        assert capture["publish_calls"] == []
    finally:
        _restore(capture)


def test_receiver_no_tenant_for_installation_returns_200_no_publish() -> None:
    """Installation id present but no tenant configured for it →
    200 ``status: no_tenant_for_installation``. Common during
    install/uninstall transitions."""

    app, capture = _build_app(tenant_lookup_returns=None)
    try:
        client = TestClient(app)
        body = json.dumps(_issues_payload()).encode()
        resp = client.post(
            "/api/v1/github/webhook",
            content=body,
            headers={
                "X-GitHub-Event": "issues",
                "X-GitHub-Delivery": "uuid-4",
                "X-Hub-Signature-256": _sign(body),
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_tenant_for_installation"
        assert capture["publish_calls"] == []
    finally:
        _restore(capture)
