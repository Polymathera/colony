"""GitHub App authentication — JWT + installation access token flow.

A GitHub App signs a short-lived JWT (≤10 min) with its RSA private
key, then exchanges it for an installation-scoped access token that
lasts ≤1 hour. Agents never see the raw JWT or the installation token:
the capability caches it and refreshes automatically.

This module is intentionally SDK-free: ``PyJWT`` for signing,
``httpx`` for the token exchange. Swapping in ``githubkit`` later is
a one-file change — the capability only touches ``TokenCache``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
import jwt

logger = logging.getLogger(__name__)


_GITHUB_API = "https://api.github.com"
_JWT_TTL_S = 9 * 60           # under GitHub's 10-minute cap
_TOKEN_REFRESH_PAD_S = 5 * 60  # refresh 5 minutes before expiry


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class _CachedToken:
    token: str
    expires_at: float

    def is_fresh(self, *, now: float, pad_s: float = _TOKEN_REFRESH_PAD_S) -> bool:
        return self.expires_at - now > pad_s


# ---------------------------------------------------------------------------
# App-level JWT minting
# ---------------------------------------------------------------------------

class GitHubAppAuth:
    """Mint App-level JWTs.

    The JWT grants the App identity (not installation-scoped); the
    capability exchanges it for an installation token via
    ``TokenCache.get``.
    """

    def __init__(
        self,
        *,
        app_id: str,
        private_key_pem: str,
        clock: "callable[[], float] | None" = None,
    ):
        if not app_id:
            raise ValueError("GitHubAppAuth: app_id is required")
        if not private_key_pem:
            raise ValueError("GitHubAppAuth: private_key_pem is required")
        self._app_id = str(app_id)
        self._private_key = private_key_pem
        self._clock = clock or time.time

    def mint_jwt(self, *, ttl_s: int = _JWT_TTL_S) -> str:
        """Sign a new App JWT valid for ``ttl_s`` seconds.

        ``iat`` is backdated 60 s per GitHub's guidance to tolerate
        minor clock skew between this host and GitHub's servers.
        """
        now = int(self._clock())
        payload = {
            "iat": now - 60,
            "exp": now + int(ttl_s),
            "iss": self._app_id,
        }
        return jwt.encode(payload, self._private_key, algorithm="RS256")


# ---------------------------------------------------------------------------
# Installation token cache
# ---------------------------------------------------------------------------

class TokenCache:
    """Mint, cache, and refresh installation access tokens.

    A single capability instance serves concurrent action calls, so
    ``get()`` is lock-protected to prevent duplicate token mints under
    contention.
    """

    def __init__(
        self,
        *,
        app_auth: GitHubAppAuth,
        installation_id: str,
        client: httpx.AsyncClient,
        api_base: str = _GITHUB_API,
        clock: "callable[[], float] | None" = None,
    ):
        if not installation_id:
            raise ValueError("TokenCache: installation_id is required")
        self._app_auth = app_auth
        self._installation_id = str(installation_id)
        self._client = client
        self._api_base = api_base.rstrip("/")
        self._clock = clock or time.time
        self._cached: _CachedToken | None = None
        self._lock = asyncio.Lock()

    async def get(self, *, force_refresh: bool = False) -> str:
        """Return a fresh installation access token.

        The token is cached until it is within 5 minutes of expiry;
        callers can force a re-mint by setting ``force_refresh``.
        """
        async with self._lock:
            now = self._clock()
            if (
                not force_refresh
                and self._cached is not None
                and self._cached.is_fresh(now=now)
            ):
                return self._cached.token
            token, expires_at = await self._mint_installation_token()
            self._cached = _CachedToken(token=token, expires_at=expires_at)
            return token

    async def _mint_installation_token(self) -> tuple[str, float]:
        """Call GitHub's ``/access_tokens`` endpoint.

        Returns ``(token, expires_at_unix)``. Raises ``RuntimeError``
        on any non-2xx response so the caller can surface it as an
        action-level error dict.
        """
        app_jwt = self._app_auth.mint_jwt()
        url = (
            f"{self._api_base}/app/installations/"
            f"{self._installation_id}/access_tokens"
        )
        headers = {
            "Authorization": f"Bearer {app_jwt}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        resp = await self._client.post(url, headers=headers)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"GitHub installation-token exchange failed: "
                f"{resp.status_code} {resp.text[:200]}"
            )
        data: dict[str, Any] = resp.json()
        token = data.get("token")
        expires_iso = data.get("expires_at")
        if not token or not expires_iso:
            raise RuntimeError(
                f"GitHub returned an unexpected token payload: {data!r}"
            )
        # "2026-04-24T14:00:00Z" → unix epoch.
        from datetime import datetime, timezone
        try:
            expires_at = datetime.fromisoformat(
                expires_iso.replace("Z", "+00:00"),
            ).timestamp()
        except ValueError:
            # Fall back to a conservative one-hour window.
            expires_at = self._clock() + 3600.0
        logger.debug(
            "GitHubAppAuth: minted installation token (expires_at=%s)",
            expires_iso,
        )
        return token, expires_at

    def invalidate(self) -> None:
        """Drop the cached token. Next ``get()`` re-mints."""
        self._cached = None
