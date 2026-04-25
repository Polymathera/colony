# `WebSearchCapability` and `ColonyDocsCapability`

`WebSearchCapability` exposes web search and page-fetch as `@action_executor` methods. `ColonyDocsCapability` is a thin subclass whose search target is pinned to <https://polymathera.github.io/colony/> so the LLM can answer system-help questions with citations.

Code: `polymathera.colony.agents.patterns.capabilities.WebSearchCapability` / `ColonyDocsCapability`.

## When to use

A session agent (or any LLM-driven agent) can otherwise only answer from training knowledge plus what's already on its blackboard. With this capability it can:

- Look up *current* information — recent CVEs, library releases, Stack Overflow answers.
- Browse a URL the user pasted into chat.
- Cite a Colony documentation page when the user asks how the system works.

## Action surface

| Action | Purpose |
|--------|---------|
| `search_web(query, max_results=10, site=None, freshness_days=None)` | Run a search via the configured backend. Returns ranked hits. |
| `fetch_page(url, extract="markdown", max_bytes=…, cache=True)` | Fetch one URL. `extract` ∈ `{text, html, markdown}` — markdown uses a lightweight BS4 reader (no `readability`/`markdownify` dep). |
| `search_and_fetch(query, max_results=3, site=None, extract="markdown")` | One-call search + parallel fetch of the top-N. |

`ColonyDocsCapability` adds:

| Action | Purpose |
|--------|---------|
| `search_docs(query, max_results=5)` | Search restricted to the Colony docs site. |
| `fetch_doc(slug_or_url, extract="markdown")` | Resolve `architecture/blackboard` → `https://polymathera.github.io/colony/architecture/blackboard.html`. |

Every action returns a stable shape (`{hits, count, message}` for search; `{url, content, status_code, truncated, cached, message}` for fetch). Errors degrade to messages — the LLM sees them.

## Backend abstraction

```python
class SearchBackend(ABC):
    async def search(self, query, *, max_results, site, freshness_days) -> list[SearchHit]: ...
    async def close(self): ...
```

Default: `TavilyBackend` — uses raw `httpx.AsyncClient` against `POST https://api.tavily.com/search`. No SDK dependency. API key from constructor or `TAVILY_API_KEY` env var.

To add SerpAPI / Bing / Brave, subclass `SearchBackend` and pass `backend=` to the capability blueprint.

## Safety defaults

- **SSRF deny list**: `localhost`, `127.0.0.1`, `0.0.0.0`, `169.254.0.0` (AWS metadata), `::1`. The `_url_allowed` check runs before any HTTP request.
- **Allow-list mode**: pass `allow_domains=["example.com"]` to flip to deny-by-default.
- **Scheme check**: only `http://` and `https://` are accepted; `file://`, `gopher://`, etc. are rejected.
- **Token-bucket rate limit**: `max_requests_per_minute` (default 30) covers both search and fetch on one capability instance.
- **`max_fetch_bytes`** truncates very large pages with `truncated=True` set on the result, so the LLM knows it's looking at a partial document.
- **Polite User-Agent**: identifies the project with a link to the docs.

## Caching

A small in-process LRU keyed on `(url, extract)`, TTL `cache_ttl_seconds` (default 1 h), bounded by `cache_max_entries`. Successful (`200`) responses populate the cache; mutations don't. The LLM sees `cached: true` on hits.

The cache is per-capability (not per-session). Distributed-cache + cross-capability access is a documented follow-up.

## Configuration

```python
WebSearchCapability.bind(
    scope=BlackboardScope.SESSION,
    backend=TavilyBackend(api_key=os.environ["TAVILY_API_KEY"]),
    deny_domains=["localhost", "127.0.0.1", "*.internal"],
    max_requests_per_minute=30,
    cache_ttl_seconds=3600,
)
ColonyDocsCapability.bind(scope=BlackboardScope.SESSION)
```

Both are wired into the session agent in `web_ui/backend/routers/sessions.py`. The Tavily key is read from the environment at action time, not at bind time, so a missing key fails the *first* `search_web` call rather than agent construction.

## Test surface

`tests/test_web_search_capability.py` (23 tests). Stubs the `SearchBackend` and uses `httpx.MockTransport` for fetches — no network. Covers: hostname suffix matching, token-bucket deadlines, TTL+LRU semantics, HTML→text/markdown extraction, SSRF block + scheme rejection + allow-list enforcement, body truncation, cache hit/miss/disabled, transport-error degradation, `search_and_fetch` merge, `ColonyDocsCapability` site pinning + slug-to-URL resolution.

## Open follow-ups

- **Distributed cache** keyed by tenant or session, backed by the working memory.
- **Robots.txt** honouring (default-on, opt-out via blueprint kwarg).
- **Per-tenant API key storage** in the Settings UI.
- **Additional backends** (SerpAPI, Bing, Brave) when needed.
