# Design: Remote LLM Deployments (Anthropic / OpenRouter)

## Motivation

`VLLMDeployment` requires GPU nodes. To test the Colony multi-agent framework on a **local Ray cluster without GPUs**, we need deployments that call remote LLM APIs (Anthropic, OpenRouter) while fitting into the existing VCM architecture as drop-in replacements for `VLLMDeployment`.

---

## 1. Colony VCM Architecture (Baseline)

The Virtual Context Manager (VCM) manages **billion-token contexts** by virtualizing them across a scalable but **limited** number of LLM instances. This is directly analogous to OS virtual memory:

| OS Virtual Memory | Colony VCM |
|-------------------|------------|
| Virtual address space (huge) | Total context pages (1B+ tokens) |
| Physical RAM (limited)       | LLM instance KV caches (limited) |
| Page frames | KV cache slots per replica |
| Page table  | `VirtualPageTable` (distributed, Redis-backed) |
| Page faults | `PageFault` → `process_page_faults_background()` |
| Page replacement (LRU) | `_select_pages_to_evict()` (LRU) |
| MMU | `VirtualContextManager` deployment |

**Key invariant**: Number of VCM pages >> Number of LLM instances.
Not all pages fit simultaneously. VCM schedules which pages are loaded where.

### How it works today (with `VLLMDeployment`)

```
VCM (cluster-level)                    VLLMDeployment replicas (per-instance)
┌──────────────────────────┐           ┌───────────────────────────┐
│ VirtualContextManager    │           │ Replica A (GPU 0)         │
│  - VirtualPageTable      │ load_page │  - loaded_pages: {P1, P3} │
│  - AllocationStrategy    │──────────>│  - vLLM engine (APC)      │
│  - PageFaultQueue        │           │  - KV cache capacity: 2M  │
│                          │ evict_page│  - KV cache used: 1.6M    │
│ allocate_pages():        │──────────>│                           │
│  - queries client_states │           └───────────────────────────┘
│  - decides WHERE to place│           ┌───────────────────────────┐
│  - evicts if needed      │           │ Replica B (GPU 1)         │
│  - calls load_page on    │ load_page │  - loaded_pages: {P2, P4} │
│    target replica        │──────────>│  - vLLM engine (APC)      │
│                          │           │  - KV cache capacity: 2M  │
│ process_page_faults():   │           │  - KV cache used: 1.2M    │
│  - pops from fault queue │           └───────────────────────────┘
│  - calls allocate_pages  │
└──────────────────────────┘
           ↑
           │ issue_page_fault()
           │
┌──────────┴───────────────┐
│ ContextAwareRouter /     │
│ PageAffinityRouter       │
│  - scores replicas by    │
│    page locality         │
│  - issues faults when    │
│    pages missing         │
└──────────────────────────┘
```

### Per-instance interface (what VCM calls)

VCM interacts with each deployment replica through these endpoints:

| Endpoint | Purpose | Called by |
|---|---|---|
| `load_page(page)` | Load a VCM page into this instance's cache | VCM `_load_page_on_client()` via `LLMCluster` |
| `evict_page(page_id)` | Remove a page from this instance's cache | VCM `_evict_page_from_client()` |
| `infer_with_context_composition(base_page_id, suffix_tokens, request)` | Inference using cached page + suffix | Agents (routed by `ContextAwareRouter`) |
| `get_state()` → `LLMClientState` | Report capacity, loaded pages, load | VCM allocation strategy, routers |

### Three layers of state

| Layer   | What | Where |
|---------|------|-------|
| Layer 1 | `VLLMDeploymentState` — per-deployment: `client_states`, `page_index` | `StateManager` (Redis-backed) |
| Layer 2 | `VirtualPageTableState` — global page table | VCM (`vcm/manager.py`) via Redis |
| Layer 3 | `loaded_pages: dict[ContextPageId, LoadedContextPage]` | In-process per replica |

### `VirtualContextPage` data model

```python
class VirtualContextPage(BaseModel):
    page_id: ContextPageId
    tokens: list[int]      # Token IDs from the model's tokenizer
    size: int              # Number of tokens
    metadata: dict         # Source file info, keywords, etc.
    group_id: str | None   # Spatial locality grouping
    tenant_id: str         # Multi-tenancy isolation
```

**Critical**: Pages store **token IDs** (`list[int]`), not text. `VLLMDeployment` consumes tokens directly; remote APIs consume **text**. This mismatch must be addressed (see Section 5).

---

## 2. The Remote Deployment Problem

### What changes

For Anthropic/OpenRouter deployments, each replica wraps a **remote API** instead of a local vLLM engine. The "KV cache" is Anthropic's **prefix cache**:

| Concept | `VLLMDeployment` | `RemoteLLMDeployment` (Anthropic) |
|---------|------------------|-----------------------------------|
| **Backing store** | Local GPU KV cache (APC) | Anthropic prefix cache (TTL-based) |
| **Capacity** | Physical GPU memory (known, queryable) | Unknown, TTL-based (simulated) |
| **Load page** | Dummy generation warms APC | Minimal request creates prefix cache entry |
| **Evict page** | Remove from tracking (vLLM LRU internally) | Remove from tracking (cache expires via TTL) |
| **Cache hit** | Free (KV blocks in GPU memory) | 0.1× input tokens (90% savings) |
| **Cache miss** | Recompute KV blocks | Full input price (1×) |
| **Warmup cost** | GPU compute time | 1.25× (5m TTL) or 2× (1h TTL) input tokens |
| **TTL** | None (persists until LRU-evicted) | 5 minutes or 1 hour (refreshed free on hit) |
| **Sharing** | vLLM shares KV blocks across concurrent requests | Same-prefix requests share cache |
| **Explicit eviction** | vLLM handles internally | Not possible — TTL expiry only |

### What does NOT change

The VCM architecture is **completely unchanged**. VCM doesn't care what backs each deployment — it only interacts through the deployment endpoints (`load_page`, `evict_page`, `get_state`, `infer_with_context_composition`).

- `AllocationStrategy` — same. Decides where pages go based on `LLMClientState`.
- `PageFaultQueue` — same. Queues missing pages, background task processes them.
- `ContextAwareRouter` — same. Routes to replicas with loaded pages.
- `PageAffinityRouter` — same. Issues faults when pages missing.
- `VirtualPageTable` — same. Tracks page locations globally.
- `LLMCluster` — same. Dispatches `load_page` to the right deployment/replica.

The remote deployment is a **transparent substitution** at the replica level.

---

## 3. Anthropic Prompt Caching API

### Key parameters

| Parameter | Value |
|---|---|
| Max cache breakpoints per request | **4** |
| Default TTL | **5 minutes** (refreshed free on each hit) |
| Extended TTL | **1 hour** (`"ttl": "1h"`) (refreshed free on each hit) |
| Cache read cost | **0.1×** base input price (90% savings) |
| 5-min cache write cost | **1.25×** base input price |
| 1-hour cache write cost | **2.0×** base input price |
| Min cacheable tokens (Sonnet 4/4.5) | **1,024** |
| Min cacheable tokens (Opus 4.5/4.6, Haiku 4.5) | **4,096** |
| Cache match type | **100% exact prefix match** |
| Manual eviction | **Not available** |
| Cache isolation | **Per organization** (per workspace starting Feb 2026) |
| Hierarchy | tools → system → messages (modifications invalidate downstream) |
| Longer TTL entries | Must appear before shorter TTL entries |
| Cache read tokens | NOT counted against rate limit |

### How prefix caching maps to VCM pages

Each VCM page loaded on an Anthropic deployment = one cached prefix. The prompt structure for `infer_with_context_composition(base_page_id, suffix_tokens)`:

```
[system prompt + cache_control]  →  Breakpoint 1 (stable across all requests)
[page text    + cache_control]  →  Breakpoint 2 (stable for same page)
[task suffix]                   →  Varies per agent/request
```

This uses 2 of 4 breakpoints. Multiple agents working on the same page share the cached prefix at 0.1× cost (analogous to vLLM sharing base KV blocks).

---

## 4. OpenRouter

- **OpenAI-compatible API** (`https://openrouter.ai/api/v1/chat/completions`)
- **No native caching** — passes through to underlying providers
- **For Claude models**: Anthropic `cache_control` is passed through → prefix
  caching works
- **Sticky routing** helps maintain warm caches on the same provider instance
- **Pricing**: Provider pricing + small markup; cache discounts reflected

---

## 5. Design Decision: Text vs Token IDs

### The problem

`VirtualContextPage.tokens` is `list[int]`. Remote APIs need **text**.

### Recommendation: Add `text` field to `VirtualContextPage`

```python
class VirtualContextPage(BaseModel):
    page_id: ContextPageId
    tokens: list[int]
    text: str | None = None      # NEW: source text for remote deployments
    size: int
    metadata: dict
    # ... rest unchanged
```

- `FileGrouperContextPageSource` already has the text before tokenizing — store
  both.
- `VLLMDeployment` ignores `text` (uses `tokens`).
- `RemoteLLMDeployment` requires `text` (falls back to `tokenizer.decode(tokens)`
  if `text` is None).
- Backward-compatible: `text` defaults to `None`.

---

## 6. Architecture

### Class hierarchy

```
                    ┌───────────────────────┐
                    │  RemoteLLMDeployment  │  (abstract base)
                    │  @serving.deployment  │
                    └──────────┬────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                                 │
  ┌───────────▼────────────┐     ┌──────────────▼───────────┐
  │ AnthropicLLMDeployment │     │ OpenRouterLLMDeployment  │
  │ (direct Anthropic API) │     │ (OpenAI-compat + cache   │
  │                        │     │  passthrough for Claude) │
  └────────────────────────┘     └──────────────────────────┘
```

### `RemoteLLMDeployment` — the `VLLMDeployment` analog

A `@serving.deployment` that implements the **same endpoint interface** as `VLLMDeployment`, making it a transparent drop-in from VCM's perspective:

```python
@serving.deployment
class RemoteLLMDeployment:
    """Remote LLM deployment — drop-in replacement for VLLMDeployment.

    Each replica of this deployment manages a LIMITED number of cached
    pages on the remote API, just as each VLLMDeployment replica manages
    a limited number of pages in GPU KV cache.

    VCM handles the cluster-level scheduling of which pages go where.
    """

    def __init__(self, config: RemoteLLMDeploymentConfig):
        # Page tracking (Layer 3 — same as VLLMDeployment)
        self.loaded_pages: dict[ContextPageId, CachedPageEntry] = {}

        # Simulated capacity (mirrors GPU KV cache capacity)
        self.max_cached_tokens: int = config.max_cached_tokens
        self.cached_tokens_used: int = 0

        # State management (Layer 1 — same pattern as VLLMDeployment)
        self.state_manager = None  # initialized in initialize()

        # Tokenizer for decode fallback
        self.tokenizer = None

    # --- Same endpoints as VLLMDeployment ---

    @serving.endpoint(router_class=TargetClientRouter, ...)
    async def load_page(self, page: VirtualContextPage) -> bool: ...

    @serving.endpoint
    async def evict_page(self, page_id: ContextPageId) -> bool: ...

    @serving.endpoint
    async def infer_with_context_composition(
        self, base_page_id: ContextPageId,
        suffix_tokens: list[int] | None = None,
        request: InferenceRequest | None = None, ...
    ) -> InferenceResponse: ...

    @serving.endpoint
    async def get_state(self) -> LLMClientState: ...

    # --- Abstract (subclass-specific) ---

    @abstractmethod
    async def _call_api(self, messages: list[dict], **kwargs) -> APIResponse: ...

    @abstractmethod
    def _build_cached_messages(
        self, page_text: str, suffix_text: str, system_prompt: str | None
    ) -> dict: ...
```

### Key architectural point

Each `RemoteLLMDeployment` replica manages a **limited** set of cached pages, just as each `VLLMDeployment` replica manages a limited KV cache. The VCM:

1. Queries `get_state()` on each replica to know capacity and loaded pages
2. Runs `AllocationStrategy.make_allocation_decisions()` to decide WHERE to place new pages
3. Calls `load_page()` on the chosen replica (via `LLMCluster`)
4. Calls `evict_page()` when capacity is needed
5. Updates `VirtualPageTable` (Layer 2) with page locations

The replica doesn't need to know about the global page space. It only manages its own local cache. **VCM is the virtual memory manager.**

---

## 7. Page Loading (Cache Warmup)

### `VLLMDeployment` approach (for reference)

`load_page()` warms Automatic Prefix Caching (APC) by running dummy generation (`max_tokens=1`) with the page tokens. After this, the KV blocks for those tokens are in GPU cache.

### `RemoteLLMDeployment` approach

Same strategy — send a minimal API request with `max_tokens=1` and `cache_control` markers to create a prefix cache entry:

```python
async def load_page(self, page: VirtualContextPage) -> bool:
    # 1. Check capacity (same pattern as VLLMDeployment)
    if self.cached_tokens_used + page.size > self.max_cached_tokens:
        evicted = await self._evict_pages_for_capacity(page.size)
        if not evicted:
            return False

    # 2. Get page text
    page_text = page.text or self.tokenizer.decode(page.tokens)

    # 3. Send minimal warmup request to create cache entry
    messages = self._build_cached_messages(
        page_text=page_text,
        suffix_text="Acknowledged.",
        system_prompt=self.system_prompt,
    )
    response = await self._call_api(messages, max_tokens=1)

    # 4. Track locally (Layer 3)
    self.loaded_pages[page.page_id] = CachedPageEntry(
        page=page,
        text=page_text,
        cached_tokens=page.size,
        ttl_expiry=time.time() + self.ttl_seconds,
        last_access=time.time(),
    )
    self.cached_tokens_used += page.size

    # 5. Update distributed state (Layer 1)
    async for state in self.state_manager.write_transaction():
        client_state = state.client_states.get(self.client_id)
        if client_state:
            client_state.kv_cache_used += page.size
            client_state.loaded_page_ids.add(page.page_id)
        state.register_page_load(page.page_id, self.client_id, page.tenant_id)

    # 6. Emit PageLoadedEvent (Layer 2 reconciliation)
    await self._emit_page_event(PageLoadedEvent(
        page_id=page.page_id,
        deployment_name=self.deployment_name,
        client_id=self.client_id,
        tenant_id=page.tenant_id,
        size=page.size,
        ...
    ))

    return True
```

**Cost of warmup** (Sonnet 4.5, 32K-token page, 1h TTL):
- Write: `32K × $6/MTok × 2.0 = $0.384`
- Subsequent reads: `32K × $6/MTok × 0.1 = $0.019` per request
- Break-even: after 2 requests

---

## 8. Inference with Context Composition

Maps directly to `VLLMDeployment.infer_with_context_composition()`:

```python
async def infer_with_context_composition(
    self, base_page_id, suffix_tokens, request
):
    entry = self.loaded_pages.get(base_page_id)
    if not entry:
        raise ValueError(f"Page {base_page_id} not loaded")

    # Refresh tracking
    entry.last_access = time.time()
    entry.ttl_expiry = time.time() + self.ttl_seconds

    suffix_text = self.tokenizer.decode(suffix_tokens) if suffix_tokens else ""

    messages = self._build_cached_messages(
        page_text=entry.text,
        suffix_text=suffix_text + "\n" + request.prompt,
        system_prompt=self.system_prompt,
    )
    response = await self._call_api(messages, max_tokens=request.max_tokens, ...)

    # Track cache metrics from response
    cache_hit = response.usage.cache_read_input_tokens > 0

    return InferenceResponse(
        request_id=request.request_id,
        generated_text=response.content[0].text,
        tokens_generated=response.usage.output_tokens,
        page_faults=[],
        metadata={
            "cache_read_tokens": response.usage.cache_read_input_tokens,
            "cache_write_tokens": response.usage.cache_creation_input_tokens,
            "provider": "anthropic",
        },
    )
```

**Concurrency**: Multiple agents analyzing the same page send requests with the same `[system + page]` prefix. Anthropic's cache serves all of them at 0.1× cost — analogous to vLLM's APC sharing base KV blocks.

### `AnthropicLLMDeployment` message format

```python
def _build_cached_messages(self, page_text, suffix_text, system_prompt=None):
    return {
        "system": [{
            "type": "text",
            "text": system_prompt or self.default_system_prompt,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }],
        "messages": [{"role": "user", "content": [
            {
                "type": "text",
                "text": page_text,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
            {
                "type": "text",
                "text": suffix_text,
                # No cache_control — varies per request
            },
        ]}],
    }
```

### `OpenRouterLLMDeployment` message format

```python
def _build_cached_messages(self, page_text, suffix_text, system_prompt=None):
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt or self.default_system_prompt,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": page_text,
                     "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": suffix_text},
                ],
            },
        ],
    }
```

---

## 9. Eviction

### What IS possible

- Remove from `loaded_pages` dict (Layer 3)
- Update `StateManager` (Layer 1)
- Publish `PageEvictedEvent` (Layer 2 reconciliation)
- Stop refreshing cache (let TTL expire naturally)

### What is NOT possible

- **Manual cache eviction on Anthropic's side**: No API exists.

### This is fine

The same pattern as `VLLMDeployment`: vLLM also handles cache eviction internally via LRU — `evict_page()` in `VLLMDeployment` mainly updates tracking state (comments in code: "*vLLM handles cache eviction automatically via its LRU policy. We cannot explicitly evict from vLLM's KV cache*.").

For the remote deployment, the cache expires via TTL instead of LRU. Stale cache entries on Anthropic's side are harmless — they don't cause incorrect behavior and save money if accidentally reused.

```python
async def evict_page(self, page_id: ContextPageId) -> bool:
    entry = self.loaded_pages.pop(page_id, None)
    if not entry:
        return True  # Already evicted

    self.cached_tokens_used -= entry.cached_tokens

    # Layer 1 update
    async for state in self.state_manager.write_transaction():
        client_state = state.client_states.get(self.client_id)
        if client_state:
            client_state.kv_cache_used -= entry.cached_tokens
            client_state.loaded_page_ids.discard(page_id)
        state.register_page_eviction(page_id, self.client_id)

    # Layer 2 event
    await self._emit_page_event(PageEvictedEvent(
        page_id=page_id, deployment_name=self.deployment_name,
        client_id=self.client_id, size=entry.cached_tokens,
        reason="manual",
    ))

    return True
```

---

## 10. TTL Management

### The problem

Anthropic caches expire after TTL of inactivity. `VLLMDeployment`'s KV cache has no TTL. A page that VCM thinks is "loaded" could have silently expired.

### Strategy: 1-hour TTL + lazy refresh + background keepalive

1. **Use 1-hour TTL** for all cache writes. Wide window before expiry.

2. **Each inference request is a free refresh**: Every `infer_with_context_composition` call refreshes the TTL on Anthropic's side at no extra cost. Active pages never expire.

3. **Background keepalive for idle pages**: Periodically send minimal requests (`max_tokens=1`) to refresh TTL for loaded-but-idle pages:

```python
@serving.periodic_health_check(interval_s=300.0)  # Every 5 minutes
async def _keepalive_cached_pages(self):
    """Refresh cache TTL for idle pages to prevent expiry."""
    now = time.time()
    for page_id, entry in list(self.loaded_pages.items()):
        idle_s = now - entry.last_access
        if idle_s > self.ttl_seconds * 0.8:  # 80% of TTL elapsed
            try:
                messages = self._build_cached_messages(
                    page_text=entry.text,
                    suffix_text="Acknowledge.",
                    system_prompt=self.system_prompt,
                )
                await self._call_api(messages, max_tokens=1)
                entry.last_access = time.time()
                entry.ttl_expiry = time.time() + self.ttl_seconds
            except Exception as e:
                logger.warning(f"Keepalive failed for page {page_id}: {e}")
                # Mark as potentially expired — VCM will handle page fault
                # if this page is requested
```

**Cost of keepalive**: ~0.1× page tokens per refresh (cache read cost). For a 32K-token page on Sonnet 4.5: `$0.019` per refresh. Negligible.

### TTL expiry as a "soft page fault"

If a page's cache expires despite keepalive (e.g., Anthropic evicts under heavy load), the next `infer_with_context_composition` call will result in a **cache miss** — Anthropic recomputes the prefix at 1× cost. This is equivalent to a vLLM APC cache miss (where vLLM recomputes the KV blocks).

The response is still correct; it's just more expensive. No explicit page fault is needed — the deployment handles it transparently.

---

## 11. Capacity Tracking (Simulated)

### The problem

`VLLMDeployment.get_state()` returns real GPU KV cache capacity/utilization. Remote APIs provide no equivalent.

### Strategy: Simulated capacity with configured limits

Track capacity locally based on configured limits. This gives VCM's `AllocationStrategy` the information it needs to make placement decisions:

```python
async def get_state(self) -> LLMClientState:
    return LLMClientState(
        client_id=self.client_id,
        deployment_name=self.deployment_name,
        model_name=self.config.model_name,
        kv_cache_capacity=self.max_cached_tokens,
        kv_cache_used=self.cached_tokens_used,
        loaded_page_ids=set(self.loaded_pages.keys()),
        last_heartbeat=time.time(),
        pending_requests=self._active_requests,
    )
```

The `BalancedAllocationStrategy` uses `kv_cache_capacity` and `kv_cache_used` to compute `capacity_score` and decide where to place pages. The simulated values make this work identically to `VLLMDeployment`.

The `max_cached_tokens` config parameter controls how many VCM pages this replica accepts before VCM must evict. This is the analog of GPU memory size.

---

## 12. Configuration

### `RemoteLLMDeploymentConfig`

New config class alongside `LLMDeploymentConfig`:

```python
class RemoteLLMDeploymentConfig(BaseModel):
    """Configuration for a remote LLM deployment (Anthropic / OpenRouter)."""
    deployment_id: str | None = None
    model_name: str                          # e.g., "claude-sonnet-4-20250514"
    provider: Literal["anthropic", "openrouter"]
    api_key_env_var: str = "ANTHROPIC_API_KEY"

    # Simulated capacity (analog of GPU KV cache size)
    max_cached_pages: int = 50
    max_cached_tokens: int = 2_000_000

    # Caching
    system_prompt: str | None = None
    ttl: Literal["5m", "1h"] = "1h"

    # Rate limiting
    max_concurrent_requests: int = 10

    # Scaling
    num_replicas: int = 1
    min_replicas: int | None = None
    max_replicas: int | None = None

    # Routing (same as LLMDeploymentConfig)
    default_router_class: str = "ContextAwareRouter"

    # OpenRouter-specific
    openrouter_provider_order: list[str] | None = None
```

### Integration into `ClusterConfig`

```python
class ClusterConfig(BaseModel):
    app_name: str
    vllm_deployments: list[LLMDeploymentConfig] = []
    remote_deployments: list[RemoteLLMDeploymentConfig] = []  # NEW

    def add_deployments_to_app(self, app: serving.Application, top_level: bool):
        # ... existing VLLMDeployment registration ...

        # Remote deployments — same pattern
        for rconf in self.remote_deployments:
            deployment_name = rconf.get_deployment_name()
            if rconf.provider == "anthropic":
                deployment_cls = AnthropicLLMDeployment
            elif rconf.provider == "openrouter":
                deployment_cls = OpenRouterLLMDeployment
            else:
                raise ValueError(f"Unknown provider: {rconf.provider}")

            router_class = get_routing_policy_class(rconf.default_router_class)
            app.add_deployment(
                deployment_cls.bind(config=rconf),
                name=deployment_name,
                default_router_class=router_class,
                autoscaling_config={...},
                ray_actor_options={"num_gpus": 0},  # No GPUs needed
            )
```

VCM treats remote deployments identically to `VLLMDeployments`. The`AllocationStrategy` sees them as clients with `kv_cache_capacity` and `kv_cache_used` — it doesn't know (or care) whether they're backed by GPUs or API calls.

---

## 13. Routing

### No changes needed

The existing routing infrastructure works as-is:

- **`ContextAwareRouter`**: Scores replicas by page locality (`PAGE_HIT_WEIGHT=100`), load (`LOAD_WEIGHT=10`), capacity (`CAPACITY_WEIGHT=5`). Works because remote replicas report `loaded_page_ids` and `kv_cache_used` through `get_state()` / `VLLMDeploymentState`.

- **`PageAffinityRouter`**: Strict page affinity — issues page faults via VCM when required pages are missing. Works because page faults route through `VCM.allocate_pages()`, which calls `load_page()` on the target replica regardless of whether it's VLLM or remote.

- **`TargetClientRouter`**: Used for directed `load_page()` calls. Works because the routing mechanism is deployment-agnostic.

- **`RequirementBasedRouter`**: Cluster-level deployment selection based on model family, context window, etc. Remote deployments register with the same metadata.


### Multi-replica remote deployments

Multiple replicas of a remote deployment each maintain their own`loaded_pages` dict. Anthropic's prefix cache is content-based and organization-scoped, so if two replicas warm the same page, the second warmup is a cache read (0.1× cost) — Anthropic handles this automatically.

`ContextAwareRouter` routes requests to the replica with the page already loaded, avoiding redundant warmup. This is the same optimization it provides for `VLLMDeployment` replicas.

---

## 14. The "Session IDs" Option

The user's original request mentioned:

> "the AnthropicLLMDeployment can either use different session IDs for different VCM pages (the VCM context manager needs to manage assignment of a limited number of active sessions to an unlimited number of VCM pages) or just ignore caching information altogether"

### How this maps to the design

The "session ID" concept maps to the Anthropic prefix cache identity. Each unique prefix (system prompt + page content) effectively creates a distinct "session" in Anthropic's cache. The VCM manages assignment of unlimited VCM pages to limited cache slots through its existing allocation machinery:

| User's concept | Implementation |
|----------------|----------------|
| "Different session IDs for different VCM pages" | Each `load_page()` call creates a distinct prefix cache entry (identified by content) |
| "Limited number of active sessions" | `RemoteLLMDeploymentConfig.max_cached_tokens` limits how many pages fit per replica |
| "Unlimited number of VCM pages" | `VirtualPageTable` tracks all pages; `AllocationStrategy` decides which are loaded |
| "VCM manages assignment" | `BalancedAllocationStrategy.make_allocation_decisions()` — unchanged |

### The "ignore caching" option

For a simpler implementation (no prefix caching):

```python
class SimpleRemoteLLMDeployment(RemoteLLMDeployment):
    """Remote deployment without caching — every request pays full input cost."""

    async def load_page(self, page: VirtualContextPage) -> bool:
        # Just track the page locally, no warmup request
        self.loaded_pages[page.page_id] = CachedPageEntry(
            page=page,
            text=page.text or self.tokenizer.decode(page.tokens),
            cached_tokens=page.size,
        )
        # ... update state layers ...
        return True

    def _build_cached_messages(self, page_text, suffix_text, system_prompt):
        # No cache_control markers
        return {
            "system": [{"type": "text", "text": system_prompt}],
            "messages": [{"role": "user", "content": page_text + "\n" + suffix_text}],
        }
```

This is functionally correct — VCM still works, routing still works, page faults still work. The only difference is cost: every request pays full input price instead of 0.1× for cached prefixes.

---

## 15. Summary: What Changes and What Doesn't

### Changes (new code)

| Component | Change |
|---|---|
| `VirtualContextPage` | Add `text: str \| None` field |
| `FileGrouperContextPageSource` | Populate `text` field |
| `RemoteLLMDeploymentConfig` | New config class in `cluster/config.py` |
| `RemoteLLMDeployment` | New base class in `cluster/remote_deployment.py` |
| `AnthropicLLMDeployment` | Subclass in `cluster/anthropic_deployment.py` |
| `OpenRouterLLMDeployment` | Subclass in `cluster/openrouter_deployment.py` |
| `ClusterConfig` | Add `remote_deployments` field + registration |

### Unchanged (no modifications needed)

| Component | Why unchanged |
|---|---|
| `VirtualContextManager` | Interacts through deployment endpoints — deployment-agnostic |
| `AllocationStrategy` | Uses `LLMClientState` — works with simulated capacity |
| `VirtualPageTable` | Tracks page locations — doesn't care about backing store |
| `ContextAwareRouter` | Scores replicas by `loaded_page_ids` — same |
| `PageAffinityRouter` | Issues page faults — same |
| `LLMCluster` | Dispatches to deployments — handles any deployment type |
| Page fault mechanism | All unchanged — works through `allocate_pages()` |

---

## 16. Implementation Plan

### Phase 1: Core infrastructure
1. Add `text: str | None` to `VirtualContextPage`
2. Update `FileGrouperContextPageSource` to populate `text`
3. Create `RemoteLLMDeploymentConfig` in `cluster/config.py`
4. Create `RemoteLLMDeployment` base class in `cluster/remote_deployment.py`

### Phase 2: Anthropic deployment
5. Create `AnthropicLLMDeployment` in `cluster/anthropic_deployment.py`
6. Implement `load_page`, `evict_page`, `infer_with_context_composition`
7. Implement keepalive loop for TTL refresh
8. Unit tests with mocked Anthropic API

### Phase 3: OpenRouter deployment
9. Create `OpenRouterLLMDeployment` in `cluster/openrouter_deployment.py`
10. Implement cache_control passthrough for Claude models
11. Handle provider routing preferences

### Phase 4: Integration
12. Update `ClusterConfig.add_deployments_to_app()`
13. Update `polymath` CLI to support remote deployment configs
14. Integration tests on local Ray cluster (no GPUs)


