"""GitHub App user-to-server OAuth — HTTP helpers.

The same GitHub App we use server-to-server (``GITHUB_APP_ID`` +
``GITHUB_PRIVATE_KEY_PEM`` → installation token) also exposes an
OAuth-style web flow that identifies an end user and returns a
short-lived user-to-server token. Colony uses this flow to verify the
Colony user's GitHub identity (login + verified emails) on the
"Connect GitHub" button — the token is used once during the callback,
then discarded. Colony never acts AS the user on GitHub.

Pure HTTP — no DB. The DB write side lives in :mod:`.service`
(``set_user_github_identity``).

References:
- https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-a-user-access-token-for-a-github-app
- https://docs.github.com/en/rest/users/users#get-the-authenticated-user
- https://docs.github.com/en/rest/users/emails#list-email-addresses-for-the-authenticated-user
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


_GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
_GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
_GITHUB_API_USER = "https://api.github.com/user"
_GITHUB_API_USER_EMAILS = "https://api.github.com/user/emails"


def build_authorize_url(
    *,
    client_id: str,
    redirect_uri: str,
    state: str,
    scopes: tuple[str, ...] = ("user:email",),
) -> str:
    """Render the URL the user's browser is redirected to so they
    approve at GitHub. ``state`` carries the CSRF token (random
    nonce); ``redirect_uri`` is Colony's own callback endpoint.

    Per the GitHub docs, App user-authorization grants the App's
    declared user permissions; the ``scope`` query parameter is
    informational (the actual scope is what the App registration
    requests). We include it for clarity in the consent screen.
    """

    from urllib.parse import urlencode
    return f"{_GITHUB_AUTHORIZE_URL}?" + urlencode({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "scope": " ".join(scopes),
    })


class OAuthExchangeError(RuntimeError):
    """Raised when GitHub rejects the authorisation code exchange."""


async def exchange_code_for_token(
    *,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
    http_client: httpx.AsyncClient | None = None,
) -> str:
    """Exchange the authorisation ``code`` for a user-to-server token.

    Raises :class:`OAuthExchangeError` when GitHub returns a 4xx or an
    error in the JSON body. Returns the raw access token string.

    ``http_client`` is injectable for tests. When ``None``, a fresh
    short-lived client is created + closed.
    """

    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }
    headers = {"Accept": "application/json"}

    if http_client is None:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
        ) as client:
            resp = await client.post(
                _GITHUB_TOKEN_URL, data=payload, headers=headers,
            )
    else:
        resp = await http_client.post(
            _GITHUB_TOKEN_URL, data=payload, headers=headers,
        )

    if resp.status_code >= 400:
        raise OAuthExchangeError(
            f"GitHub token exchange failed: {resp.status_code} "
            f"{resp.text[:200]}",
        )
    data = resp.json()
    if "error" in data:
        # GitHub returns 200 with an ``error`` field on bad codes,
        # expired codes, etc. Surface it as the same error type.
        raise OAuthExchangeError(
            f"GitHub token exchange returned error: "
            f"{data.get('error')}: {data.get('error_description', '')}",
        )
    token = data.get("access_token")
    if not token:
        raise OAuthExchangeError(
            "GitHub token exchange response missing access_token",
        )
    return token


async def fetch_authenticated_identity(
    *,
    user_token: str,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Use the user-to-server token to fetch the GitHub user's
    identity. Returns::

        {
            "login":            "anassar",
            "github_user_id":   123456,    # stable across renames
            "name":             "Ali ...", # display name or None
            "verified_emails":  [
                {"email": "...", "primary": True, "verified": True},
                ...
            ],
            "primary_email":    "ali@example.com",   # picked from the list
        }

    The primary email is the GitHub-verified ``primary`` address (or,
    if none is marked primary, the first verified one). When the user
    has no verified email at all, ``primary_email`` is ``None`` — the
    caller decides whether to reject the connect.
    """

    headers = {
        "Authorization": f"Bearer {user_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    if http_client is None:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
        ) as client:
            user_resp = await client.get(_GITHUB_API_USER, headers=headers)
            emails_resp = await client.get(
                _GITHUB_API_USER_EMAILS, headers=headers,
            )
    else:
        user_resp = await http_client.get(_GITHUB_API_USER, headers=headers)
        emails_resp = await http_client.get(
            _GITHUB_API_USER_EMAILS, headers=headers,
        )

    if user_resp.status_code >= 400:
        raise OAuthExchangeError(
            f"GET /user failed: {user_resp.status_code} "
            f"{user_resp.text[:200]}",
        )
    user_data = user_resp.json()

    # /user/emails requires user:email scope. If the App wasn't granted
    # that scope, this returns 404 — treat as no verified emails.
    verified_emails: list[dict[str, Any]] = []
    if emails_resp.status_code < 400:
        for entry in emails_resp.json():
            if entry.get("verified"):
                verified_emails.append({
                    "email": entry.get("email"),
                    "primary": bool(entry.get("primary")),
                    "verified": True,
                })

    primary_email: str | None = None
    for entry in verified_emails:
        if entry["primary"]:
            primary_email = entry["email"]
            break
    if primary_email is None and verified_emails:
        primary_email = verified_emails[0]["email"]

    return {
        "login": user_data.get("login"),
        "github_user_id": user_data.get("id"),
        "name": user_data.get("name"),
        "verified_emails": verified_emails,
        "primary_email": primary_email,
    }
