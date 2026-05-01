"""Unit tests for ``WebSearchCapability`` and ``ColonyDocsCapability``.

No network calls. The search backend is a stub; fetch goes through
``httpx.MockTransport`` so every HTTP response is deterministic.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import httpx
import pytest

from polymathera.colony.agents.patterns.capabilities.web_search import (
    ColonyDocsCapability,
    SearchBackend,
    SearchHit,
    WebSearchCapability,
    _extract_body,
    _hostname_matches,
    _TokenBucket,
    _TTLCache,
)
from polymathera.colony.agents.scopes import BlackboardScope
from polymathera.colony.distributed.ray_utils.serving.context import (
    execution_context,
    Ring,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _StubBackend(SearchBackend):
    """In-memory backend whose ``search`` echoes a scripted result."""

    def __init__(self):
        self.calls: list[dict] = []
        self.hits: list[SearchHit] = []
        self.raise_on_search: Exception | None = None

    async def search(self, query, *, max_results, site, freshness_days):
        self.calls.append({
            "query": query, "max_results": max_results,
            "site": site, "freshness_days": freshness_days,
        })
        if self.raise_on_search is not None:
            raise self.raise_on_search
        return list(self.hits)


def _make_capability(
    *,
    backend: SearchBackend | None = None,
    allow_domains: list[str] | None = None,
    deny_domains: list[str] | None = None,
    transport: httpx.BaseTransport | None = None,
    cache_ttl_seconds: int = 3600,
    max_requests_per_minute: int = 1000,
) -> WebSearchCapability:
    agent = MagicMock()
    agent.agent_id = "agent-test"
    cap = WebSearchCapability(
        agent=agent,
        scope=BlackboardScope.SESSION,
        backend=backend or _StubBackend(),
        allow_domains=allow_domains,
        deny_domains=deny_domains,
        max_requests_per_minute=max_requests_per_minute,
        cache_ttl_seconds=cache_ttl_seconds,
    )
    if transport is not None:
        cap._fetch_client = httpx.AsyncClient(
            transport=transport,
            headers={"User-Agent": "test"},
            follow_redirects=True,
        )
    return cap


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Pure-unit helpers
# ---------------------------------------------------------------------------

def test_hostname_suffix_match_semantics():
    assert _hostname_matches("example.com", ("example.com",)) is True
    assert _hostname_matches("sub.example.com", ("example.com",)) is True
    # Must not be a substring match.
    assert _hostname_matches("fakeexample.com", ("example.com",)) is False
    assert _hostname_matches("", ("example.com",)) is False
    assert _hostname_matches(None, ("example.com",)) is False


def test_token_bucket_admits_one_then_refuses_within_deadline():
    async def run():
        bucket = _TokenBucket(1)  # 1 req / 60s
        first = await bucket.acquire(deadline_s=0.0)
        second = await bucket.acquire(deadline_s=0.1)
        return first, second

    first, second = asyncio.get_event_loop().run_until_complete(run())
    assert first is True
    assert second is False


def test_ttl_cache_expires_entries():
    cache = _TTLCache(max_entries=4, ttl_seconds=0)
    cache.set("k", "v")
    # ttl_seconds=0 means "already expired" by the next get.
    time.sleep(0.01)
    assert cache.get("k") is None


def test_ttl_cache_evicts_oldest_over_capacity():
    cache = _TTLCache(max_entries=2, ttl_seconds=60)
    cache.set("a", 1); cache.set("b", 2); cache.set("c", 3)
    assert cache.get("a") is None  # evicted
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_extract_body_strips_scripts_and_style():
    html = """
    <html><head><style>.x{}</style></head>
    <body>
      <h1>Hello</h1>
      <script>alert(1)</script>
      <p>World</p>
    </body></html>
    """
    text = _extract_body(html, "text")
    assert "Hello" in text
    assert "World" in text
    assert "alert(1)" not in text


def test_extract_body_markdown_preserves_headings_and_paragraphs():
    html = "<h2>Title</h2><p>Body text.</p><ul><li>one</li><li>two</li></ul>"
    md = _extract_body(html, "markdown")
    assert "## Title" in md
    assert "Body text." in md
    assert "- one" in md


# ---------------------------------------------------------------------------
# search_web
# ---------------------------------------------------------------------------

def test_search_web_returns_hits_as_dicts():
    backend = _StubBackend()
    backend.hits = [
        SearchHit(title="T1", url="https://example.com/a",
                  snippet="s1", score=0.9),
        SearchHit(title="T2", url="https://example.com/b", snippet="s2"),
    ]
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(backend=backend)
        result = _run(cap.search_web(
            query="hello", max_results=5, site="example.com",
            freshness_days=7,
        ))
    assert result["count"] == 2
    assert result["hits"][0]["title"] == "T1"
    assert result["hits"][0]["score"] == 0.9
    # Kwargs forwarded faithfully.
    assert backend.calls == [{
        "query": "hello", "max_results": 5,
        "site": "example.com", "freshness_days": 7,
    }]


def test_search_web_degrades_to_error_dict_on_backend_failure():
    backend = _StubBackend()
    backend.raise_on_search = RuntimeError("boom")
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(backend=backend)
        result = _run(cap.search_web(query="x"))
    assert result == {"hits": [], "count": 0, "message": "boom"}


def test_search_web_surfaces_rate_limit_as_message():
    backend = _StubBackend()
    backend.hits = [SearchHit(title="T", url="https://x", snippet="")]
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(
            backend=backend, max_requests_per_minute=1,
        )
        # First call consumes the single token.
        assert _run(cap.search_web(query="a"))["count"] == 1
        # Second call can't get a token before the 30s deadline; we
        # shorten it here by patching the bucket to refuse immediately.
        cap._rate_limiter = _TokenBucket(1)
        cap._rate_limiter._tokens = 0.0
        cap._rate_limiter._last = time.monotonic()
        async def _no_wait(*, deadline_s):
            return False
        cap._rate_limiter.acquire = _no_wait  # type: ignore[assignment]
        result = _run(cap.search_web(query="b"))
    assert result["count"] == 0
    assert "rate limit" in result["message"].lower()


# ---------------------------------------------------------------------------
# fetch_page
# ---------------------------------------------------------------------------

def _fake_transport(
    responses: dict[str, httpx.Response],
    *,
    default_status: int = 404,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in responses:
            return responses[url]
        return httpx.Response(default_status, text="not found")
    return httpx.MockTransport(handler)


def test_fetch_page_blocks_private_hosts_by_default():
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability()
        result = _run(cap.fetch_page(url="http://localhost/secret"))
    assert result["status_code"] == 0
    assert "deny list" in result["message"]


def test_fetch_page_blocks_non_http_scheme():
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability()
        result = _run(cap.fetch_page(url="file:///etc/passwd"))
    assert result["status_code"] == 0
    assert "unsupported scheme" in result["message"]


def test_fetch_page_enforces_allow_list_when_provided():
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(allow_domains=["example.com"])
        result = _run(cap.fetch_page(url="https://other.com/page"))
    assert result["status_code"] == 0
    assert "not in allow list" in result["message"]


def test_fetch_page_extracts_markdown_from_html_response():
    html = "<html><body><h1>Title</h1><p>Body</p></body></html>"
    transport = _fake_transport({
        "https://example.com/a": httpx.Response(
            200, text=html,
            headers={"content-type": "text/html"},
        ),
    })
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(
            deny_domains=[],  # turn off default deny for test
            transport=transport,
        )
        result = _run(cap.fetch_page(url="https://example.com/a"))
    assert result["status_code"] == 200
    assert result["truncated"] is False
    assert "# Title" in result["content"]
    assert "Body" in result["content"]
    assert result["cached"] is False


def test_fetch_page_truncates_body_at_max_bytes():
    body = "<html><body>" + ("x" * 200) + "</body></html>"
    transport = _fake_transport({
        "https://example.com/big": httpx.Response(
            200, text=body, headers={"content-type": "text/html"},
        ),
    })
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(deny_domains=[], transport=transport)
        result = _run(cap.fetch_page(
            url="https://example.com/big", max_bytes=50,
        ))
    assert result["truncated"] is True


def test_fetch_page_caches_200_responses_by_default():
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(
            200, text="<html><body><p>Hi</p></body></html>",
            headers={"content-type": "text/html"},
        )

    transport = httpx.MockTransport(handler)
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(deny_domains=[], transport=transport)
        r1 = _run(cap.fetch_page(url="https://example.com/x"))
        r2 = _run(cap.fetch_page(url="https://example.com/x"))
    assert r1["cached"] is False
    assert r2["cached"] is True
    assert call_count["n"] == 1


def test_fetch_page_skips_cache_when_disabled():
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(
            200, text="<p>hi</p>", headers={"content-type": "text/html"},
        )

    transport = httpx.MockTransport(handler)
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(deny_domains=[], transport=transport)
        _run(cap.fetch_page(url="https://example.com/y", cache=False))
        _run(cap.fetch_page(url="https://example.com/y", cache=False))
    assert call_count["n"] == 2


def test_fetch_page_degrades_to_error_dict_on_transport_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("transport down", request=request)

    transport = httpx.MockTransport(handler)
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(deny_domains=[], transport=transport)
        result = _run(cap.fetch_page(url="https://example.com/z"))
    assert result["status_code"] == 0
    assert "transport down" in result["message"]


# ---------------------------------------------------------------------------
# search_and_fetch
# ---------------------------------------------------------------------------

def test_search_and_fetch_merges_search_hits_with_page_content():
    backend = _StubBackend()
    backend.hits = [
        SearchHit(title="T", url="https://example.com/hit", snippet="s"),
    ]
    transport = _fake_transport({
        "https://example.com/hit": httpx.Response(
            200, text="<p>content</p>",
            headers={"content-type": "text/html"},
        ),
    })
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(
            backend=backend, deny_domains=[], transport=transport,
        )
        result = _run(cap.search_and_fetch(query="q", max_results=1))
    assert result["count"] == 1
    merged = result["hits"][0]
    assert merged["title"] == "T"
    assert merged["status_code"] == 200
    assert "content" in merged["content"]


def test_search_and_fetch_short_circuits_when_search_is_empty():
    backend = _StubBackend()  # no hits, no error
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(backend=backend)
        result = _run(cap.search_and_fetch(query="q"))
    assert result == {"hits": [], "count": 0, "message": ""}


# ---------------------------------------------------------------------------
# ColonyDocsCapability
# ---------------------------------------------------------------------------

def test_colony_docs_pins_site_on_search_docs():
    backend = _StubBackend()
    backend.hits = [
        SearchHit(title="T", url="https://polymathera.github.io/colony/x",
                  snippet="s"),
    ]
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        agent = MagicMock(); agent.agent_id = "agent-test"
        cap = ColonyDocsCapability(
            agent=agent,
            scope=BlackboardScope.SESSION,
            backend=backend,
            deny_domains=[],
            cache_ttl_seconds=0,
        )
        _run(cap.search_docs(query="blackboard"))
    assert backend.calls[0]["site"] == "polymathera.github.io/colony"


def test_colony_docs_resolves_slug_to_full_url():
    # We don't need an HTTP response — we intercept fetch_page to
    # capture the URL the subclass built.
    captured: list[str] = []

    async def fake_fetch_page(self, url, *, extract="markdown",
                              max_bytes=None, cache=True):
        captured.append(url)
        return {
            "url": url, "final_url": url, "status_code": 200,
            "content_type": "text/html", "content": "",
            "fetched_at": 0.0, "truncated": False,
            "cached": False, "message": "",
        }

    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        agent = MagicMock(); agent.agent_id = "agent-test"
        cap = ColonyDocsCapability(
            agent=agent,
            scope=BlackboardScope.SESSION,
            backend=_StubBackend(),
        )
        # Monkey-patch the base class fetch_page so the subclass action
        # only has to prove its slug->URL resolution logic.
        cap.fetch_page = fake_fetch_page.__get__(cap, ColonyDocsCapability)

        _run(cap.fetch_doc(slug_or_url="architecture/blackboard"))
        _run(cap.fetch_doc(slug_or_url="/architecture/planning.html"))
        _run(cap.fetch_doc(
            slug_or_url="https://polymathera.github.io/colony/guides/colony-env.html",
        ))
    assert captured == [
        "https://polymathera.github.io/colony/architecture/blackboard.html",
        "https://polymathera.github.io/colony/architecture/planning.html",
        "https://polymathera.github.io/colony/guides/colony-env.html",
    ]


# ---------------------------------------------------------------------------
# Blueprint serialisation
# ---------------------------------------------------------------------------

def test_bind_round_trips_through_cloudpickle():
    # Ray's vendored cloudpickle — see comment in
    # test_github_capability for why standalone PyPI cloudpickle is
    # not the right import here.
    from ray import cloudpickle
    bp1 = WebSearchCapability.bind(
        scope=BlackboardScope.SESSION,
        allow_domains=["example.com"],
        deny_domains=["localhost"],
        max_requests_per_minute=50,
    )
    bp2 = ColonyDocsCapability.bind(scope=BlackboardScope.SESSION)
    for bp in (bp1, bp2):
        roundtripped = cloudpickle.loads(cloudpickle.dumps(bp))
        assert roundtripped.cls in (
            WebSearchCapability, ColonyDocsCapability,
        )


def test_action_executors_are_registered():
    import inspect
    ws_keys = {
        m._action_key for _, m in inspect.getmembers(
            WebSearchCapability, predicate=inspect.isfunction
        ) if getattr(m, "_action_key", None)
    }
    assert ws_keys == {"search_web", "fetch_page", "search_and_fetch"}

    docs_keys = {
        m._action_key for _, m in inspect.getmembers(
            ColonyDocsCapability, predicate=inspect.isfunction
        ) if getattr(m, "_action_key", None)
    }
    assert docs_keys == {
        "search_web", "fetch_page", "search_and_fetch",
        "search_docs", "fetch_doc",
    }
