"""Smoke + unit tests for the GitHub user-OAuth surface.

The full HTTP surface needs a live cluster (auth middleware + a real
Postgres pool); we exercise only what is testable in isolation:
route registration + the OAuth HTTP helpers in
:mod:`auth.github_oauth` with ``httpx.MockTransport``.
"""

from __future__ import annotations

import httpx
import pytest

from polymathera.colony.web_ui.backend.auth.github_oauth import (
    OAuthExchangeError,
    build_authorize_url,
    exchange_code_for_token,
    fetch_authenticated_identity,
)
from polymathera.colony.web_ui.backend.routers import (
    github_oauth as github_oauth_router,
)
from polymathera.colony.web_ui.backend.routers import tenants as tenants_router


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def test_github_oauth_router_registers_expected_paths() -> None:
    """Fail fast if a route disappears under a rename / refactor."""

    paths = {route.path for route in github_oauth_router.router.routes}
    assert paths == {
        "/auth/github/connect",
        "/auth/github/callback",
        "/users/me/github",  # both GET (read) and DELETE (disconnect)
    }


def test_tenants_router_registers_expected_paths() -> None:
    paths = {route.path for route in tenants_router.router.routes}
    assert paths == {"/tenants/me/github-installation"}


# ---------------------------------------------------------------------------
# build_authorize_url
# ---------------------------------------------------------------------------


def test_build_authorize_url_carries_required_params() -> None:
    url = build_authorize_url(
        client_id="Iv1.abc",
        redirect_uri="https://colony.example/api/v1/auth/github/callback",
        state="nonce-123",
    )
    assert url.startswith("https://github.com/login/oauth/authorize?")
    assert "client_id=Iv1.abc" in url
    assert "state=nonce-123" in url
    assert "redirect_uri=https" in url
    assert "scope=user%3Aemail" in url


def test_build_authorize_url_includes_custom_scopes() -> None:
    url = build_authorize_url(
        client_id="c", redirect_uri="r", state="s",
        scopes=("user:email", "read:org"),
    )
    assert "scope=user%3Aemail+read%3Aorg" in url


# ---------------------------------------------------------------------------
# exchange_code_for_token
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exchange_code_returns_access_token_on_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # GitHub expects form data; we sent ``data={...}``
        assert request.url.path == "/login/oauth/access_token"
        assert request.headers["accept"] == "application/json"
        return httpx.Response(200, json={
            "access_token": "ghu_xyz", "token_type": "bearer",
            "scope": "user:email",
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        token = await exchange_code_for_token(
            client_id="cid", client_secret="csec",
            code="thecode", redirect_uri="https://colony/x",
            http_client=client,
        )
    finally:
        await client.aclose()
    assert token == "ghu_xyz"


@pytest.mark.asyncio
async def test_exchange_code_raises_on_4xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(OAuthExchangeError, match="token exchange failed"):
            await exchange_code_for_token(
                client_id="cid", client_secret="csec",
                code="bad", redirect_uri="https://colony/x",
                http_client=client,
            )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_exchange_code_raises_on_error_in_json_body() -> None:
    """GitHub returns 200 with ``error`` field for bad codes / expired
    codes / wrong redirect_uri. Treat as an OAuthExchangeError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "error": "bad_verification_code",
            "error_description": "The code passed is incorrect or expired.",
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(OAuthExchangeError, match="bad_verification_code"):
            await exchange_code_for_token(
                client_id="c", client_secret="cs", code="x",
                redirect_uri="r", http_client=client,
            )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_exchange_code_raises_when_response_missing_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"token_type": "bearer"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(OAuthExchangeError, match="missing access_token"):
            await exchange_code_for_token(
                client_id="c", client_secret="cs", code="x",
                redirect_uri="r", http_client=client,
            )
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# fetch_authenticated_identity
# ---------------------------------------------------------------------------


def _identity_handler_factory(
    *,
    user_payload: dict | None = None,
    user_status: int = 200,
    emails_payload: list | None = None,
    emails_status: int = 200,
):
    """Returns a httpx handler that responds to /user + /user/emails
    with the canned payloads."""

    user_payload = user_payload if user_payload is not None else {
        "login": "anassar", "id": 42, "name": "Ali Nassar",
    }
    emails_payload = emails_payload if emails_payload is not None else [
        {"email": "noreply@github.com",
         "primary": False, "verified": True},
        {"email": "ali@example.com",
         "primary": True, "verified": True},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        assert request.headers["authorization"].startswith("Bearer ")
        if path == "/user":
            return httpx.Response(user_status, json=user_payload)
        if path == "/user/emails":
            return httpx.Response(emails_status, json=emails_payload)
        return httpx.Response(404, text=f"unexpected path: {path}")

    return handler


@pytest.mark.asyncio
async def test_fetch_identity_returns_primary_verified_email() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(_identity_handler_factory()),
    )
    try:
        identity = await fetch_authenticated_identity(
            user_token="tok", http_client=client,
        )
    finally:
        await client.aclose()
    assert identity["login"] == "anassar"
    assert identity["github_user_id"] == 42
    assert identity["name"] == "Ali Nassar"
    assert identity["primary_email"] == "ali@example.com"
    assert len(identity["verified_emails"]) == 2


@pytest.mark.asyncio
async def test_fetch_identity_falls_back_to_first_verified_when_no_primary() -> None:
    """If GitHub doesn't flag any address ``primary``, pick the first
    verified one rather than dropping the email entirely."""

    handler = _identity_handler_factory(emails_payload=[
        {"email": "a@example.com", "primary": False, "verified": True},
        {"email": "b@example.com", "primary": False, "verified": True},
    ])
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        identity = await fetch_authenticated_identity(
            user_token="tok", http_client=client,
        )
    finally:
        await client.aclose()
    assert identity["primary_email"] == "a@example.com"


@pytest.mark.asyncio
async def test_fetch_identity_filters_unverified_emails() -> None:
    handler = _identity_handler_factory(emails_payload=[
        {"email": "unverified@example.com",
         "primary": True, "verified": False},
        {"email": "verified@example.com",
         "primary": False, "verified": True},
    ])
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        identity = await fetch_authenticated_identity(
            user_token="tok", http_client=client,
        )
    finally:
        await client.aclose()
    # The unverified primary is dropped; only the verified one remains
    # → picked as the (sole) primary fallback.
    assert identity["primary_email"] == "verified@example.com"
    assert all(e["verified"] for e in identity["verified_emails"])
    assert len(identity["verified_emails"]) == 1


@pytest.mark.asyncio
async def test_fetch_identity_handles_missing_emails_scope() -> None:
    """If the App wasn't granted ``user:email``, /user/emails returns
    a 4xx and we proceed with no verified emails (the route layer
    decides whether to reject the connect)."""

    handler = _identity_handler_factory(
        emails_status=404, emails_payload={"message": "Not Found"},
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        identity = await fetch_authenticated_identity(
            user_token="tok", http_client=client,
        )
    finally:
        await client.aclose()
    assert identity["login"] == "anassar"
    assert identity["verified_emails"] == []
    assert identity["primary_email"] is None


@pytest.mark.asyncio
async def test_fetch_identity_raises_when_user_call_4xxes() -> None:
    handler = _identity_handler_factory(
        user_status=401, user_payload={"message": "Bad credentials"},
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(OAuthExchangeError, match="/user failed"):
            await fetch_authenticated_identity(
                user_token="tok", http_client=client,
            )
    finally:
        await client.aclose()
