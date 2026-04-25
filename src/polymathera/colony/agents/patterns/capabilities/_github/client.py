"""Thin async GitHub REST + GraphQL client.

Wraps ``httpx`` with:

- Automatic installation-token injection (via ``TokenCache``).
- Distinct exception types (``NotFoundError``, ``RateLimitError``,
  ``GitHubError``) so the capability can surface each class as a
  distinct action-return shape.
- A single retry on ``401`` (re-mint token) and capped exponential
  backoff on secondary rate-limit signals.

GraphQL is exposed as a single ``graphql(query, variables)`` helper —
the capability builds the query strings at call sites, which keeps
this file independent of the set of queries the capability actually
uses.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

from .auth import TokenCache

logger = logging.getLogger(__name__)


_API_BASE = "https://api.github.com"
_GRAPHQL_URL = "https://api.github.com/graphql"
_DEFAULT_USER_AGENT = (
    "PolymatheraColony-GitHubCapability/1.0 "
    "(+https://polymathera.github.io/colony)"
)


class GitHubError(RuntimeError):
    """Generic GitHub API error. Carries the status code and body."""

    def __init__(self, message: str, *, status_code: int, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class NotFoundError(GitHubError):
    """404 response."""


class RateLimitError(GitHubError):
    """Primary or secondary rate limit exhausted."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class GitHubClient:
    """Async wrapper around the GitHub REST + GraphQL APIs.

    Args:
        tokens: ``TokenCache`` that returns an installation token.
        client: Optional pre-configured ``httpx.AsyncClient``. When
            ``None``, one is constructed with sensible defaults.
        max_retries: Maximum attempts on retryable failures.
        user_agent: ``User-Agent`` header value. GitHub rejects
            unrecognised UAs on some endpoints.
    """

    def __init__(
        self,
        *,
        tokens: TokenCache,
        client: httpx.AsyncClient | None = None,
        max_retries: int = 3,
        user_agent: str = _DEFAULT_USER_AGENT,
        api_base: str = _API_BASE,
        graphql_url: str = _GRAPHQL_URL,
    ):
        self._tokens = tokens
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0, read=30.0, write=10.0, pool=10.0,
            ),
        )
        self._max_retries = max_retries
        self._user_agent = user_agent
        self._api_base = api_base.rstrip("/")
        self._graphql_url = graphql_url

    async def close(self) -> None:
        await self._client.aclose()

    # --- Internal ---------------------------------------------------------

    async def _headers(self, *, force_fresh: bool = False) -> dict[str, str]:
        token = await self._tokens.get(force_refresh=force_fresh)
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": self._user_agent,
        }

    @staticmethod
    def _is_primary_rate_limited(resp: httpx.Response) -> bool:
        if resp.status_code != 403:
            return False
        return resp.headers.get("x-ratelimit-remaining") == "0"

    @staticmethod
    def _is_secondary_rate_limited(resp: httpx.Response) -> bool:
        if resp.status_code not in (403, 429):
            return False
        # GitHub flags secondary/abuse limits in the body rather than
        # a single header; we match both the header and the message.
        if resp.headers.get("retry-after"):
            return True
        text = resp.text.lower()
        return "secondary rate limit" in text or "abuse" in text

    async def _backoff(self, attempt: int, retry_after: float | None) -> None:
        if retry_after is not None:
            delay = max(0.0, float(retry_after))
        else:
            delay = min(30.0, 0.5 * (2 ** attempt)) + random.uniform(0.0, 0.25)
        await asyncio.sleep(delay)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
    ) -> httpx.Response:
        url = (
            path if path.startswith(("http://", "https://"))
            else f"{self._api_base}{path}"
        )
        attempt = 0
        while True:
            headers = await self._headers(force_fresh=False)
            resp = await self._client.request(
                method, url, headers=headers, params=params, json=json,
            )
            if resp.status_code == 401 and attempt == 0:
                # Stale token; force-refresh once.
                logger.debug(
                    "GitHubClient: 401 on %s %s; refreshing token",
                    method, url,
                )
                await self._tokens.get(force_refresh=True)
                attempt += 1
                continue
            if self._is_secondary_rate_limited(resp):
                if attempt >= self._max_retries:
                    raise RateLimitError(
                        "secondary rate limit exhausted",
                        status_code=resp.status_code, body=resp.text,
                    )
                retry_after = resp.headers.get("retry-after")
                retry_after_f = float(retry_after) if retry_after else None
                logger.info(
                    "GitHubClient: secondary rate limited on %s %s; "
                    "backing off (retry-after=%s)",
                    method, url, retry_after,
                )
                await self._backoff(attempt, retry_after_f)
                attempt += 1
                continue
            if self._is_primary_rate_limited(resp):
                raise RateLimitError(
                    "primary rate limit exhausted",
                    status_code=resp.status_code, body=resp.text,
                )
            return resp

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        if resp.status_code == 404:
            raise NotFoundError(
                f"{resp.request.method} {resp.request.url} → 404",
                status_code=404, body=resp.text,
            )
        raise GitHubError(
            f"{resp.request.method} {resp.request.url} → "
            f"{resp.status_code}",
            status_code=resp.status_code, body=resp.text,
        )

    # --- REST helpers -----------------------------------------------------

    async def get(self, path: str, **params: Any) -> Any:
        resp = await self._request("GET", path, params=params or None)
        self._raise_for_status(resp)
        return resp.json() if resp.text else None

    async def post(self, path: str, *, json: Any = None) -> Any:
        resp = await self._request("POST", path, json=json)
        self._raise_for_status(resp)
        return resp.json() if resp.text else None

    async def patch(self, path: str, *, json: Any = None) -> Any:
        resp = await self._request("PATCH", path, json=json)
        self._raise_for_status(resp)
        return resp.json() if resp.text else None

    async def put(self, path: str, *, json: Any = None) -> Any:
        resp = await self._request("PUT", path, json=json)
        self._raise_for_status(resp)
        return resp.json() if resp.text else None

    async def delete(self, path: str) -> None:
        resp = await self._request("DELETE", path)
        if resp.status_code == 404:
            # Deleting something gone is a no-op for our purposes.
            return
        self._raise_for_status(resp)

    async def get_raw(self, path: str, **params: Any) -> tuple[int, str, str]:
        """Return ``(status_code, content_type, body)`` for endpoints
        that return non-JSON content (e.g., ``Accept: diff``)."""
        resp = await self._request("GET", path, params=params or None)
        return (
            resp.status_code,
            resp.headers.get("content-type", ""),
            resp.text,
        )

    async def iter_paginated(
        self, path: str, *, page_size: int = 100, **params: Any,
    ):
        """Iterate every page of a paginated endpoint.

        GitHub signals pagination via the ``Link`` header; we don't
        bother parsing it — we stop when a page comes back short.
        """
        page = 1
        while True:
            resp = await self._request(
                "GET", path,
                params={**params, "per_page": page_size, "page": page},
            )
            self._raise_for_status(resp)
            items = resp.json() if resp.text else []
            if not isinstance(items, list):
                yield items
                return
            for item in items:
                yield item
            if len(items) < page_size:
                return
            page += 1

    # --- GraphQL ----------------------------------------------------------

    async def graphql(
        self, query: str, *, variables: dict[str, Any] | None = None,
    ) -> Any:
        resp = await self._request(
            "POST", self._graphql_url,
            json={"query": query, "variables": variables or {}},
        )
        self._raise_for_status(resp)
        data = resp.json()
        if "errors" in data and data["errors"]:
            # GraphQL often returns 200 with per-field errors; surface
            # the first one so the capability can show it to the LLM.
            first = data["errors"][0]
            raise GitHubError(
                f"GraphQL error: {first.get('message', '')}",
                status_code=200, body=str(data["errors"]),
            )
        return data.get("data")
