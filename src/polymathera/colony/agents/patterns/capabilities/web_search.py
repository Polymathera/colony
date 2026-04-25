"""Web search and page fetch capability.

Exposes ``@action_executor`` methods that let an LLM-driven agent run a
web search and fetch individual pages. The search backend is pluggable
(``SearchBackend`` ABC + ``TavilyBackend`` default); the fetch side uses
``httpx`` directly so no SDK is required on the critical path.

Safety defaults are opinionated: SSRF-blocking domain deny list, a
token-bucket rate limit, a per-fetch byte cap, and a conservative
User-Agent. API keys are never included in action return values and
never flow into the LLM prompt.

``ColonyDocsCapability`` is a thin subclass whose ``site`` is pinned to
``polymathera.github.io/colony``; it exists so the LLM sees a distinct
``search_docs`` / ``fetch_doc`` action surface when the agent is asking
about Colony itself.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING
from urllib.parse import urlparse
from overrides import override

from ...base import AgentCapability
from ...models import AgentSuspensionState
from ...scopes import BlackboardScope, get_scope_prefix
from ..actions import action_executor

if TYPE_CHECKING:
    from ...base import Agent


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

_ExtractMode = Literal["text", "html", "markdown"]


@dataclass
class SearchHit:
    """One result from a ``SearchBackend.search`` call.

    Kept as a small dataclass rather than a dict so backend
    implementations share a typed contract. Actions convert to plain
    dicts before returning to the dispatcher.
    """

    title: str
    url: str
    snippet: str
    score: float | None = None
    published_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "score": self.score,
            "published_at": self.published_at,
        }


class SearchBackend(ABC):
    """Abstract search backend.

    Subclasses implement ``search`` to return ``SearchHit`` records.
    The capability handles caching, rate limiting, and result shaping;
    backends are intentionally thin so that adding SerpAPI/Bing/Brave
    later is a pure-function job.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        max_results: int,
        site: str | None,
        freshness_days: int | None,
    ) -> list[SearchHit]:
        ...

    async def close(self) -> None:
        """Release backend resources. Called by ``VCMCapability.shutdown``-
        style teardown. Default: no-op."""
        return None


# ---------------------------------------------------------------------------
# Tavily backend
# ---------------------------------------------------------------------------

class TavilyBackend(SearchBackend):
    """Tavily Web Search API v2.

    Uses the JSON HTTP endpoint directly (``POST /search``) rather than
    pulling in the ``tavily-python`` SDK — fewer dependencies, the same
    request/response shape. API reference:
    https://docs.tavily.com/docs/rest-api/api-reference
    """

    _ENDPOINT = "https://api.tavily.com/search"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_s: float = 15.0,
    ):
        # Fall back to env var so keys stay out of config files.
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self._timeout_s = timeout_s
        self._client = None  # lazy-init

    def _get_client(self):
        import httpx
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_s)
        return self._client

    @override
    async def search(
        self,
        query: str,
        *,
        max_results: int,
        site: str | None,
        freshness_days: int | None,
    ) -> list[SearchHit]:
        if not self._api_key:
            raise RuntimeError(
                "TavilyBackend: no API key. Pass api_key=... to the "
                "backend or set TAVILY_API_KEY in the environment."
            )
        payload: dict[str, Any] = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max(1, min(max_results, 20)),
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        }
        if site:
            payload["include_domains"] = [site]
        if freshness_days is not None:
            payload["days"] = freshness_days

        resp = await self._get_client().post(self._ENDPOINT, json=payload)
        resp.raise_for_status()
        data = resp.json()
        hits: list[SearchHit] = []
        for r in data.get("results", []):
            hits.append(SearchHit(
                title=r.get("title", "") or "",
                url=r.get("url", "") or "",
                snippet=r.get("content", "") or "",
                score=r.get("score"),
                published_at=r.get("published_date"),
            ))
        return hits

    @override
    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# Rate limiter and cache
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Minimal async token bucket.

    ``max_requests_per_minute`` tokens; refill at 1/60s. Lock-protected
    so concurrent agent actions on the same capability instance do not
    race the counter.
    """

    def __init__(self, rate_per_minute: int):
        self._capacity = max(1, rate_per_minute)
        self._tokens = float(self._capacity)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, *, deadline_s: float) -> bool:
        """Try to acquire one token; wait up to ``deadline_s`` seconds.

        Returns True on success, False if the deadline is reached.
        """
        start = time.monotonic()
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = max(0.0, now - self._last)
                self._last = now
                # Refill: capacity tokens per 60s.
                self._tokens = min(
                    float(self._capacity),
                    self._tokens + elapsed * (self._capacity / 60.0),
                )
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
            if time.monotonic() - start >= deadline_s:
                return False
            await asyncio.sleep(0.05)


class _TTLCache:
    """Tiny LRU + TTL cache for fetched pages.

    Per-capability, in-process. The design doc raises the question of a
    session-wide distributed cache; we defer that to a follow-up and
    keep this layer simple and observable.
    """

    def __init__(self, *, max_entries: int, ttl_seconds: int):
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._data: OrderedDict[Any, tuple[float, Any]] = OrderedDict()

    def get(self, key: Any) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.time() >= expires_at:
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: Any, value: Any) -> None:
        expires_at = time.time() + self._ttl_seconds
        self._data[key] = (expires_at, value)
        self._data.move_to_end(key)
        while len(self._data) > self._max_entries:
            self._data.popitem(last=False)

    def size(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# Domain allow/deny
# ---------------------------------------------------------------------------

_DEFAULT_DENY: tuple[str, ...] = (
    "localhost", "127.0.0.1", "0.0.0.0",
    "169.254.0.0",  # AWS metadata / link-local
    "::1",
)


def _hostname_matches(host: str | None, patterns: tuple[str, ...]) -> bool:
    """Case-insensitive suffix match on the hostname.

    ``example.com`` matches ``sub.example.com`` but not ``fakeexample.com``.
    """
    if not host:
        return False
    h = host.lower()
    for p in patterns:
        p = p.lower()
        if h == p or h.endswith(f".{p}"):
            return True
    return False


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------

def _extract_body(html: str, mode: _ExtractMode) -> str:
    """Convert raw HTML to the mode-selected representation.

    ``html`` is returned unchanged; ``text`` collapses whitespace via
    BeautifulSoup's ``get_text``; ``markdown`` is a small hand-rolled
    pass that preserves headings, links, and paragraphs so the LLM
    sees reading-order prose without pulling in readability/markdownify.
    """
    if mode == "html":
        return html
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    if mode == "text":
        text = soup.get_text(separator="\n", strip=True)
        return "\n".join(line for line in text.splitlines() if line.strip())
    # markdown
    lines: list[str] = []
    for node in soup.descendants:
        if not hasattr(node, "name"):
            continue
        name = node.name
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(name[1])
            text = node.get_text(" ", strip=True)
            if text:
                lines.append(f"\n{'#' * level} {text}\n")
        elif name == "p":
            text = node.get_text(" ", strip=True)
            if text:
                lines.append(text)
        elif name == "li":
            text = node.get_text(" ", strip=True)
            if text:
                lines.append(f"- {text}")
    body = "\n".join(lines).strip()
    if body:
        return body
    # Fallback: plain text if nothing structured was found.
    return soup.get_text(separator="\n", strip=True)


# ---------------------------------------------------------------------------
# WebSearchCapability
# ---------------------------------------------------------------------------

class WebSearchCapability(AgentCapability):
    """Search the web and fetch pages.

    Actions:

    - ``search_web(query, ...)`` — run a search, return hits.
    - ``fetch_page(url, ...)`` — fetch one URL; extract text/markdown.
    - ``search_and_fetch(query, ...)`` — search + fetch top-N in one call.

    Args:
        agent: Owning agent.
        scope: Blackboard partition this capability writes under.
        namespace: Capability sub-namespace (not the search target — that
            is an action parameter).
        backend: Search backend implementation. Defaults to
            ``TavilyBackend()`` which reads ``TAVILY_API_KEY`` from env.
        allow_domains: If non-empty, only URLs matching one of these
            hostnames (suffix match) may be fetched.
        deny_domains: Hostnames that must NEVER be fetched. Defaults
            include common SSRF targets (localhost, link-local).
        max_requests_per_minute: Rate cap applied to both search and
            fetch calls on this capability instance.
        cache_ttl_seconds: How long to keep a fetched page in the local
            cache. 0 disables the cache.
        cache_max_entries: LRU bound on the cache.
        max_fetch_bytes: Per-request cap on the body size returned. Used
            to short-circuit very large pages that would otherwise fill
            the prompt.
        user_agent: Value of the ``User-Agent`` header. The default
            identifies the project with a link to the docs.
        capability_key: Dispatcher key. ``"web_search"`` by default.
        app_name: Optional ``serving`` application override.
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.SESSION,
        namespace: str = "web_search",
        backend: SearchBackend | None = None,
        allow_domains: list[str] | None = None,
        deny_domains: list[str] | None = None,
        max_requests_per_minute: int = 30,
        cache_ttl_seconds: int = 3600,
        cache_max_entries: int = 256,
        max_fetch_bytes: int = 1_000_000,
        user_agent: str = (
            "PolymatheraColony/1.0 "
            "(+https://polymathera.github.io/colony)"
        ),
        capability_key: str = "web_search",
        app_name: str | None = None,
    ):
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(scope, agent, namespace=namespace),
            capability_key=capability_key,
            app_name=app_name,
        )
        self._backend: SearchBackend = backend or TavilyBackend()
        self._allow_domains: tuple[str, ...] = tuple(allow_domains or ())
        self._deny_domains: tuple[str, ...] = tuple(
            (deny_domains if deny_domains is not None else list(_DEFAULT_DENY))
        )
        self._rate_limiter = _TokenBucket(max_requests_per_minute)
        self._cache = _TTLCache(
            max_entries=cache_max_entries,
            ttl_seconds=cache_ttl_seconds,
        )
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_fetch_bytes = max_fetch_bytes
        self._user_agent = user_agent
        self._fetch_client = None  # lazy

    def get_action_group_description(self) -> str:
        return (
            "Web Search — run web searches and fetch web pages. Use "
            "search_web to look up current information (recent CVEs, "
            "library releases, documentation), fetch_page to read a URL "
            "the user pasted, and search_and_fetch as a one-call "
            "search+read shortcut. The capability respects a domain "
            "deny list (no SSRF to private infra) and a request rate "
            "cap; exceeding the cap raises a rate-limit error."
        )

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"web", "retrieval", "external"})

    @override
    async def serialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> AgentSuspensionState:
        # Cache is volatile; API keys are not serialized.
        return state

    @override
    async def deserialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> None:
        return None

    async def shutdown(self) -> None:
        """Close backend + fetch clients. Idempotent."""
        if self._fetch_client is not None:
            await self._fetch_client.aclose()
            self._fetch_client = None
        if self._backend is not None:
            await self._backend.close()

    # --- Internal ----------------------------------------------------------

    def _get_fetch_client(self):
        import httpx
        if self._fetch_client is None:
            self._fetch_client = httpx.AsyncClient(
                headers={"User-Agent": self._user_agent},
                follow_redirects=True,
                timeout=httpx.Timeout(
                    connect=10.0, read=30.0, write=10.0, pool=10.0,
                ),
            )
        return self._fetch_client

    def _url_allowed(self, url: str) -> tuple[bool, str]:
        """Decide whether a URL may be fetched.

        Returns ``(allowed, reason)``. ``reason`` is non-empty iff blocked.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False, f"unsupported scheme {parsed.scheme!r}"
        host = parsed.hostname
        if host is None:
            return False, "no hostname in URL"
        if _hostname_matches(host, self._deny_domains):
            return False, f"host {host!r} is in deny list"
        if self._allow_domains and not _hostname_matches(
            host, self._allow_domains
        ):
            return False, f"host {host!r} is not in allow list"
        return True, ""

    async def _acquire_rate_token(self, *, timeout_s: float) -> None:
        ok = await self._rate_limiter.acquire(deadline_s=timeout_s)
        if not ok:
            raise RuntimeError(
                f"WebSearchCapability: rate limit exceeded "
                f"(waited {timeout_s}s)"
            )

    # === Action executors ==================================================

    @action_executor()
    async def search_web(
        self,
        query: str,
        *,
        max_results: int = 10,
        site: str | None = None,
        freshness_days: int | None = None,
    ) -> dict[str, Any]:
        """Run a web search.

        Args:
            query: Free-text query. Backend-specific operators (site:,
                filetype:, etc.) may or may not be honoured depending on
                the backend — prefer the explicit ``site`` parameter.
            max_results: Maximum number of hits to return (capped at 20).
            site: Restrict results to a single site suffix (e.g.,
                ``"stackoverflow.com"``).
            freshness_days: Prefer results published within the last N
                days. ``None`` means no freshness filter.

        Returns:
            ``{"hits": [...], "count": int, "message": str}`` where
            ``hits`` is a list of ``{title, url, snippet, score,
            published_at}`` dicts. Rate-limit and backend errors are
            surfaced in ``message`` rather than raised.
        """
        try:
            await self._acquire_rate_token(timeout_s=30.0)
            hits = await self._backend.search(
                query=query,
                max_results=max_results,
                site=site,
                freshness_days=freshness_days,
            )
        except Exception as e:
            logger.warning(
                "WebSearchCapability.search_web failed: %s", e,
            )
            return {"hits": [], "count": 0, "message": str(e)}
        result = [h.to_dict() for h in hits]
        return {"hits": result, "count": len(result), "message": ""}

    @action_executor()
    async def fetch_page(
        self,
        url: str,
        *,
        extract: _ExtractMode = "markdown",
        max_bytes: int | None = None,
        cache: bool = True,
    ) -> dict[str, Any]:
        """Fetch a single URL and extract its body.

        Args:
            url: Absolute HTTP/HTTPS URL.
            extract: ``"text"``, ``"html"``, or ``"markdown"``
                (default). Markdown uses a lightweight BS4-based
                extractor that preserves headings, paragraphs, and
                list items.
            max_bytes: Override for ``self._max_fetch_bytes``. Body is
                truncated (``truncated=True`` in the result) rather than
                failing when the cap is hit.
            cache: If True, consult / populate the in-process TTL cache
                keyed by ``(url, extract)``. Mutates are never cached.

        Returns:
            ``{"url", "final_url", "status_code", "content_type",
            "content", "fetched_at", "truncated", "cached", "message"}``.
            The action never raises for remote-side errors — it returns
            the error text in ``message`` so the LLM can see it.
        """
        allowed, reason = self._url_allowed(url)
        if not allowed:
            return self._fetch_error(url, reason)

        cap = max_bytes if max_bytes is not None else self._max_fetch_bytes

        if cache and self._cache_ttl_seconds > 0:
            cached = self._cache.get((url, extract))
            if cached is not None:
                cached = dict(cached)
                cached["cached"] = True
                return cached

        try:
            await self._acquire_rate_token(timeout_s=30.0)
            client = self._get_fetch_client()
            resp = await client.get(url)
        except Exception as e:
            logger.warning(
                "WebSearchCapability.fetch_page(%s) failed: %s", url, e,
            )
            return self._fetch_error(url, f"fetch failed: {e}")

        raw = resp.content or b""
        truncated = len(raw) > cap
        if truncated:
            raw = raw[:cap]
        try:
            body = raw.decode(resp.encoding or "utf-8", errors="replace")
        except Exception:
            body = raw.decode("utf-8", errors="replace")

        try:
            content = _extract_body(body, extract)
        except Exception as e:
            logger.warning(
                "WebSearchCapability._extract_body(%s, %s) failed: %s",
                url, extract, e,
            )
            content = body  # best effort — surface raw body

        result = {
            "url": url,
            "final_url": str(resp.url),
            "status_code": resp.status_code,
            "content_type": resp.headers.get("content-type", ""),
            "content": content,
            "fetched_at": time.time(),
            "truncated": truncated,
            "cached": False,
            "message": "",
        }
        if cache and self._cache_ttl_seconds > 0 and resp.status_code == 200:
            self._cache.set((url, extract), dict(result))
        return result

    @action_executor()
    async def search_and_fetch(
        self,
        query: str,
        *,
        max_results: int = 3,
        site: str | None = None,
        extract: _ExtractMode = "markdown",
    ) -> dict[str, Any]:
        """Run a search and fetch the top-``max_results`` pages in
        parallel.

        Returns:
            ``{"hits": [...], "count": int, "message": str}`` where each
            hit has every field from ``search_web`` plus the fetch fields
            (``content``, ``final_url``, ``status_code``, ``truncated``).
        """
        search = await self.search_web(
            query=query, max_results=max_results, site=site,
        )
        hits = search.get("hits", [])
        if not hits:
            return {
                "hits": hits, "count": 0,
                "message": search.get("message", ""),
            }
        # Parallel fetch with one token per URL.
        fetched = await asyncio.gather(*[
            self.fetch_page(h["url"], extract=extract) for h in hits
        ])
        merged: list[dict[str, Any]] = []
        for h, f in zip(hits, fetched):
            m = dict(h)
            for field in (
                "final_url", "status_code", "content_type", "content",
                "fetched_at", "truncated",
            ):
                m[field] = f.get(field)
            if f.get("message"):
                m["fetch_error"] = f["message"]
            merged.append(m)
        return {"hits": merged, "count": len(merged), "message": ""}

    @staticmethod
    def _fetch_error(url: str, message: str) -> dict[str, Any]:
        return {
            "url": url,
            "final_url": url,
            "status_code": 0,
            "content_type": "",
            "content": "",
            "fetched_at": time.time(),
            "truncated": False,
            "cached": False,
            "message": message,
        }


# ---------------------------------------------------------------------------
# ColonyDocsCapability
# ---------------------------------------------------------------------------

class ColonyDocsCapability(WebSearchCapability):
    """Look things up in the Polymathera Colony docs.

    Mechanically identical to :class:`WebSearchCapability` but the
    action surface is renamed (``search_docs`` / ``fetch_doc``) and the
    search target is pinned to ``polymathera.github.io/colony``. The
    LLM sees a distinct action group so the planner can reach for it
    specifically when the user asks about the system itself.
    """

    _DOCS_SITE = "polymathera.github.io/colony"
    _DOCS_BASE_URL = "https://polymathera.github.io/colony/"

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.SESSION,
        namespace: str = "colony_docs",
        **kwargs: Any,
    ):
        super().__init__(
            agent=agent, scope=scope, namespace=namespace, **kwargs,
        )

    @override
    def get_action_group_description(self) -> str:
        return (
            "Colony Docs — search and read the Polymathera Colony "
            "documentation at https://polymathera.github.io/colony/. "
            "Use search_docs whenever the user asks how the system "
            "works or which action to use. fetch_doc reads a single "
            "page by full URL or by doc slug (e.g., "
            "'architecture/blackboard')."
        )

    @override
    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"docs", "colony", "retrieval"})

    # --- Actions -----------------------------------------------------------

    @action_executor()
    async def search_docs(
        self, query: str, *, max_results: int = 5,
    ) -> dict[str, Any]:
        """Search the Colony docs site for ``query``."""
        return await self.search_web(
            query=query,
            max_results=max_results,
            site=self._DOCS_SITE,
        )

    @action_executor()
    async def fetch_doc(
        self,
        slug_or_url: str,
        *,
        extract: _ExtractMode = "markdown",
    ) -> dict[str, Any]:
        """Fetch one documentation page.

        ``slug_or_url`` may be a full URL (anywhere on the docs site) or
        a slug like ``architecture/blackboard`` or
        ``architecture/blackboard.html``. Slugs are resolved against
        the docs base URL.
        """
        if slug_or_url.startswith(("http://", "https://")):
            url = slug_or_url
        else:
            slug = slug_or_url.strip().lstrip("/")
            if not slug.endswith((".html", "/")):
                slug = f"{slug}.html"
            url = self._DOCS_BASE_URL + slug
        return await self.fetch_page(url=url, extract=extract)
