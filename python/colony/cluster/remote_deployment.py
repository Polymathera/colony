"""Remote LLM deployment — drop-in replacement for VLLMDeployment.

Each replica of this deployment manages a LIMITED number of cached pages on
a remote API (Anthropic, OpenRouter), just as each VLLMDeployment replica
manages a limited number of pages in GPU KV cache.

VCM handles the cluster-level scheduling of which pages go where.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..distributed.ray_utils import serving
from ..vcm.events import (
    PageEvent,
    PageEvictedEvent,
    PageLoadedEvent,
    PageLoadFailedEvent,
)
from ..vcm.models import ContextPageId, VirtualContextPage
from .models import (
    ContextPageState,
    InferenceRequest,
    InferenceResponse,
    LLMClientId,
    LLMClientState,
    LoadedContextPage,
    KVCacheMetrics,
    VLLMDeploymentState,
)
from .remote_config import RemoteLLMDeploymentConfig

logger = logging.getLogger(__name__)


@dataclass
class CachedPageEntry:
    """Tracks a page cached on this remote deployment replica."""

    page: VirtualContextPage
    text: str
    cached_tokens: int
    ttl_expiry: float
    last_access: float
    access_count: int = 0


@dataclass
class APIResponse:
    """Normalized response from a remote LLM API call."""

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0
    raw_response: Any = None


@serving.deployment
class RemoteLLMDeployment:
    """Remote LLM deployment — drop-in replacement for VLLMDeployment.

    Each replica manages a LIMITED number of cached pages on the remote API,
    just as each VLLMDeployment replica manages a limited number of pages
    in GPU KV cache. VCM handles cluster-level scheduling of which pages
    go where.

    Subclasses must implement:
    - _call_api(messages, **kwargs) -> APIResponse
    - _build_cached_messages(page_text, suffix_text, system_prompt) -> dict
    - _initialize_client() -> None
    """

    def __init__(self, config: RemoteLLMDeploymentConfig):
        self.config = config

        # Page tracking (Layer 3 — same as VLLMDeployment)
        self.loaded_pages: dict[ContextPageId, CachedPageEntry] = {}

        # Simulated capacity (mirrors GPU KV cache capacity)
        self.max_cached_tokens: int = config.max_cached_tokens
        self.cached_tokens_used: int = 0

        # State management (Layer 1 — same pattern as VLLMDeployment)
        self.state_manager = None
        self.client_id: LLMClientId | None = None

        # KV cache metrics (same structure as VLLMDeployment)
        self.kv_metrics = KVCacheMetrics(
            kv_cache_capacity_tokens=self.max_cached_tokens
        )

        # Concurrency control
        self._active_requests = 0
        self._request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        # TTL configuration
        self.ttl_seconds = config.get_ttl_seconds()

        # Event publishing infrastructure (initialized in initialize())
        self.redis_client = None
        self.redis_om = None
        self.event_namespace: str | None = None

        # Tokenizer for decode fallback (when page.text is None)
        self.tokenizer = None

    @serving.initialize_deployment
    async def initialize(self) -> None:
        """Initialize the remote deployment and set up state.

        Called automatically by the serving framework after deployment.
        """
        import ray

        logger.info(
            f"Initializing RemoteLLMDeployment for {self.config.model_name} "
            f"(provider={self.config.provider})"
        )

        # Generate client ID
        self.client_id = LLMClientId(
            f"remote-{self.config.provider}-{ray.get_runtime_context().get_actor_id()}"
        )

        # Initialize the API client (subclass-specific)
        await self._initialize_client()

        # Initialize tokenizer for decode fallback
        from .tokenization import get_tokenizer_for_model
        try:
            self.tokenizer = await get_tokenizer_for_model(
                model_name_or_path=self.config.model_name,
                model_name=self.config.model_name,
            )
            logger.info(f"Initialized tokenizer for {self.config.model_name}")
        except Exception as e:
            logger.warning(
                f"Failed to initialize tokenizer for {self.config.model_name}: {e}. "
                "Pages without text field will not be usable."
            )

        # Initialize deployment-level StateManager (same pattern as VLLMDeployment)
        app_name = serving.get_my_app_name()
        deployment_name = serving.get_my_deployment_name()
        logger.info(f"Setting up StateManager for {app_name}.{deployment_name}")

        from ..distributed import get_polymathera
        polymathera = get_polymathera()
        self.state_manager = await polymathera.get_state_manager(
            state_type=VLLMDeploymentState,
            state_key=VLLMDeploymentState.get_state_key(app_name, deployment_name),
        )

        # Register this client in deployment state
        initial_state = LLMClientState(
            client_id=self.client_id,
            deployment_name=deployment_name,
            app_name=app_name,
            model_name=self.config.model_name,
            kv_cache_capacity=self.max_cached_tokens,
            kv_cache_used=0,
            loaded_page_ids=set(),
            last_heartbeat=time.time(),
        )
        async for state in self.state_manager.write_transaction():
            state.client_states[self.client_id] = initial_state

        # Initialize Redis and event publishing infrastructure
        try:
            self.redis_client = await polymathera.get_redis_client()
            self.event_namespace = f"remote_events:{deployment_name}"

            from ..distributed.redis_utils import RedisOM
            self.redis_om = RedisOM(
                redis_client=self.redis_client,
                namespace=self.event_namespace,
            )

            topics = {"vcm_page_events": {}}
            await self.redis_om.initialize_topics(topics)
            logger.info(f"Initialized Redis event publishing for {self.event_namespace}")
        except Exception as e:
            logger.warning(
                f"Failed to initialize Redis event publishing: {e}. "
                "Events will not be emitted."
            )
            self.redis_client = None
            self.redis_om = None

        logger.info(
            f"RemoteLLMDeployment {self.client_id} initialized "
            f"(capacity={self.max_cached_tokens} tokens, ttl={self.config.ttl})"
        )

    # -------------------------------------------------------------------------
    # Endpoints (same interface as VLLMDeployment)
    # -------------------------------------------------------------------------

    @serving.endpoint(router_class="TargetClientRouter")
    async def load_page(self, page: VirtualContextPage) -> bool:
        """Load a context page by creating a prefix cache entry on the remote API.

        Sends a minimal warmup request (max_tokens=1) with cache_control markers
        to create a prefix cache entry. This is the analog of VLLMDeployment
        warming up Automatic Prefix Caching (APC) via dummy generation.

        Args:
            page: Context page to load

        Returns:
            True if page was loaded successfully
        """
        try:
            # Check if page is already loaded
            if page.page_id in self.loaded_pages:
                entry = self.loaded_pages[page.page_id]
                entry.last_access = time.time()
                entry.ttl_expiry = time.time() + self.ttl_seconds
                entry.access_count += 1
                logger.info(f"Page {page.page_id} already loaded in {self.client_id}")
                return True

            # Check capacity
            if self.cached_tokens_used + page.size > self.max_cached_tokens:
                evicted = await self._evict_pages_for_capacity(page.size)
                if not evicted:
                    logger.error(f"Failed to evict pages for {page.page_id}")
                    return False

            # Get page text
            page_text = page.text
            if page_text is None:
                if self.tokenizer is None:
                    logger.error(
                        f"Page {page.page_id} has no text and no tokenizer available"
                    )
                    return False
                page_text = self.tokenizer.decode(page.tokens)

            # Send minimal warmup request to create cache entry
            messages = self._build_cached_messages(
                page_text=page_text,
                suffix_text="Acknowledged.",
                system_prompt=self.config.system_prompt,
            )
            response = await self._call_api(messages, max_tokens=1)

            # Track locally (Layer 3)
            now = time.time()
            self.loaded_pages[page.page_id] = CachedPageEntry(
                page=page,
                text=page_text,
                cached_tokens=page.size,
                ttl_expiry=now + self.ttl_seconds,
                last_access=now,
            )
            self.cached_tokens_used += page.size

            # Update distributed state (Layer 1)
            kv_cache_used = 0
            if self.state_manager:
                async for state in self.state_manager.write_transaction():
                    client_state = state.client_states.get(self.client_id)
                    if client_state:
                        client_state.kv_cache_used += page.size
                        client_state.loaded_page_ids.add(page.page_id)
                        kv_cache_used = client_state.kv_cache_used
                    state.register_page_load(
                        page.page_id, self.client_id, page.tenant_id
                    )

            logger.info(
                f"Loaded page {page.page_id} into {self.client_id} "
                f"(size={page.size}, capacity_used={kv_cache_used}/{self.max_cached_tokens}, "
                f"warmup_cost=${response.cost_usd:.4f})"
            )

            # Emit PageLoadedEvent for VCM reconciliation
            load_duration_ms = (time.time() - now) * 1000
            deployment_name = serving.get_my_deployment_name()
            await self._emit_page_event(
                PageLoadedEvent(
                    page_id=page.page_id,
                    deployment_name=deployment_name,
                    client_id=self.client_id,
                    tenant_id=page.tenant_id,
                    timestamp=time.time(),
                    size=page.size,
                    kv_cache_slot=0,  # No physical KV slot for remote
                    load_duration_ms=load_duration_ms,
                )
            )

            return True

        except Exception as e:
            logger.error(
                f"Failed to load page {page.page_id}: {e}", exc_info=True
            )
            deployment_name = serving.get_my_deployment_name()
            await self._emit_page_event(
                PageLoadFailedEvent(
                    page_id=page.page_id,
                    deployment_name=deployment_name,
                    client_id=self.client_id,
                    tenant_id=page.tenant_id,
                    timestamp=time.time(),
                    error=str(e),
                    error_type=type(e).__name__,
                )
            )
            return False

    @serving.endpoint
    async def evict_page(self, page_id: ContextPageId) -> bool:
        """Evict a page from tracking. The remote cache expires via TTL.

        Same pattern as VLLMDeployment: vLLM handles cache eviction via LRU,
        we handle cache eviction via TTL expiry. In both cases, evict_page()
        mainly updates tracking state.

        Args:
            page_id: ID of the page to evict

        Returns:
            True if page was evicted
        """
        try:
            entry = self.loaded_pages.pop(page_id, None)
            if not entry:
                return True  # Already evicted

            self.cached_tokens_used -= entry.cached_tokens
            tenant_id = entry.page.tenant_id

            # Update distributed state (Layer 1)
            if self.state_manager:
                async for state in self.state_manager.write_transaction():
                    client_state = state.client_states.get(self.client_id)
                    if client_state:
                        client_state.kv_cache_used -= entry.cached_tokens
                        client_state.loaded_page_ids.discard(page_id)
                    state.register_page_eviction(page_id, self.client_id)

            logger.info(f"Evicted page {page_id} from {self.client_id}")

            # Emit PageEvictedEvent for VCM reconciliation
            deployment_name = serving.get_my_deployment_name()
            await self._emit_page_event(
                PageEvictedEvent(
                    page_id=page_id,
                    deployment_name=deployment_name,
                    client_id=self.client_id,
                    tenant_id=tenant_id,
                    timestamp=time.time(),
                    size=entry.cached_tokens,
                    reason="manual",
                )
            )

            return True

        except Exception as e:
            logger.error(f"Failed to evict page {page_id}: {e}", exc_info=True)
            return False

    @serving.endpoint
    async def infer_with_context_composition(
        self,
        base_page_id: ContextPageId,
        suffix_tokens: list[int] | None = None,
        request: InferenceRequest | None = None,
        max_concurrent_per_page: int | None = None,
    ) -> InferenceResponse:
        """Inference using cached page prefix + task-specific suffix.

        This is the PRIMARY API for context composition, identical in signature
        to VLLMDeployment.infer_with_context_composition().

        Multiple agents using the same page share the cached prefix at 0.1x cost
        (analogous to vLLM sharing base KV blocks via APC).

        Args:
            base_page_id: ID of the loaded base page
            suffix_tokens: Optional task-specific tokens to append
            request: Optional InferenceRequest
            max_concurrent_per_page: Unused (kept for API compatibility)

        Returns:
            InferenceResponse with generated text and cost metadata
        """
        start_time = time.time()

        # Validate base page is loaded
        entry = self.loaded_pages.get(base_page_id)
        if not entry:
            raise ValueError(
                f"Page {base_page_id} not loaded in {self.client_id}"
            )

        # Refresh tracking
        entry.last_access = time.time()
        entry.ttl_expiry = time.time() + self.ttl_seconds
        entry.access_count += 1

        # Build suffix text
        suffix_text = ""
        if suffix_tokens and self.tokenizer:
            suffix_text = self.tokenizer.decode(suffix_tokens)

        # Merge with request prompt if provided
        prompt_text = ""
        if request and request.prompt:
            prompt_text = request.prompt

        if suffix_text and prompt_text:
            full_suffix = f"{suffix_text}\n{prompt_text}"
        elif suffix_text:
            full_suffix = suffix_text
        else:
            full_suffix = prompt_text or "Continue."

        # Create default request if not provided
        if request is None:
            request = InferenceRequest(
                request_id=f"compose-{base_page_id}-{uuid.uuid4().hex[:8]}",
                prompt=full_suffix,
            )

        # Build messages with cache control markers
        messages = self._build_cached_messages(
            page_text=entry.text,
            suffix_text=full_suffix,
            system_prompt=self.config.system_prompt,
        )

        # Call API with concurrency control
        async with self._request_semaphore:
            self._active_requests += 1
            try:
                response = await self._call_api(
                    messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    json_schema=request.json_schema,
                )
            finally:
                self._active_requests -= 1

        latency_ms = (time.time() - start_time) * 1000

        # Track cache metrics
        cache_hit = response.cache_read_input_tokens > 0
        suffix_size = len(suffix_tokens) if suffix_tokens else 0
        self.kv_metrics.record_request(
            page_id=base_page_id,
            suffix_size=suffix_size,
            cache_hit=cache_hit,
        )

        return InferenceResponse(
            request_id=request.request_id,
            generated_text=response.content,
            tokens_generated=response.output_tokens,
            page_faults=[],
            latency_ms=latency_ms,
            metadata={
                "base_page_id": base_page_id,
                "suffix_size": suffix_size,
                "cache_hit": cache_hit,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cache_read_tokens": response.cache_read_input_tokens,
                "cache_write_tokens": response.cache_creation_input_tokens,
                "cost_usd": response.cost_usd,
                "provider": self.config.provider,
                "model": self.config.model_name,
            },
        )

    @serving.endpoint
    async def get_state(self) -> LLMClientState:
        """Get current state of this remote LLM client.

        Returns simulated capacity information for VCM's AllocationStrategy.

        Returns:
            Current client state
        """
        current_state = None
        async for dep_state in self.state_manager.write_transaction():
            client_state = dep_state.client_states.get(self.client_id)
            if client_state:
                client_state.last_heartbeat = time.time()
                client_state.pending_requests = self._active_requests
                current_state = client_state.model_copy(deep=True)

        if current_state is None:
            raise RuntimeError(
                f"Client state not found in StateManager for {self.client_id}"
            )

        return current_state

    # -------------------------------------------------------------------------
    # TTL management
    # -------------------------------------------------------------------------

    @serving.periodic_health_check(interval_s=300.0)
    async def _keepalive_cached_pages(self) -> None:
        """Refresh cache TTL for idle pages to prevent expiry.

        Runs every 5 minutes. Sends minimal requests (max_tokens=1) to
        refresh TTL for pages that are approaching expiry. Cost is negligible:
        cache read at 0.1x page tokens per refresh.
        """
        now = time.time()
        refreshed = 0

        for page_id, entry in list(self.loaded_pages.items()):
            idle_s = now - entry.last_access
            # Refresh if 80% of TTL has elapsed
            if idle_s > self.ttl_seconds * 0.8:
                try:
                    messages = self._build_cached_messages(
                        page_text=entry.text,
                        suffix_text="Acknowledge.",
                        system_prompt=self.config.system_prompt,
                    )
                    await self._call_api(messages, max_tokens=1)
                    entry.last_access = time.time()
                    entry.ttl_expiry = time.time() + self.ttl_seconds
                    refreshed += 1
                except Exception as e:
                    logger.warning(f"Keepalive failed for page {page_id}: {e}")

        if refreshed > 0:
            logger.info(f"Refreshed TTL for {refreshed} idle pages")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    async def _evict_pages_for_capacity(self, required_size: int) -> bool:
        """Evict pages to free up required capacity using LRU policy.

        Args:
            required_size: Number of tokens needed

        Returns:
            True if sufficient capacity was freed
        """
        pages_by_lru = sorted(
            self.loaded_pages.values(),
            key=lambda e: e.last_access,
        )

        freed_size = 0
        evicted_pages = []
        available = self.max_cached_tokens - self.cached_tokens_used

        for entry in pages_by_lru:
            if available + freed_size >= required_size:
                break
            await self.evict_page(entry.page.page_id)
            freed_size += entry.cached_tokens
            evicted_pages.append(entry.page.page_id)

        if evicted_pages:
            logger.info(
                f"Evicted {len(evicted_pages)} pages to free {freed_size} tokens: "
                f"{evicted_pages}"
            )

        return (available + freed_size) >= required_size

    async def _emit_page_event(self, event: PageEvent) -> bool:
        """Emit a page lifecycle event to Redis pub/sub.

        Same pattern as VLLMDeployment._emit_page_event().
        """
        if not self.redis_om:
            logger.debug(f"Skipping event emission (no Redis): {event.event_type}")
            return False

        try:
            success = await self.redis_om.update_state_topic(
                topic="vcm_page_events",
                updates={
                    "event_type": event.event_type,
                    "event_data": event.model_dump(),
                    "timestamp": event.timestamp,
                },
                replace_all=False,
                update_type="update",
            )
            if success:
                logger.debug(
                    f"Emitted {event.event_type} for page {event.page_id} "
                    f"on {event.deployment_name}/{event.client_id}"
                )
            return success
        except Exception as e:
            logger.warning(f"Failed to emit page event: {e}")
            return False

    # -------------------------------------------------------------------------
    # Abstract methods (subclass-specific)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def _initialize_client(self) -> None:
        """Initialize the API client. Called during initialize()."""
        ...

    @abstractmethod
    async def _call_api(
        self,
        messages: dict[str, Any],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        json_schema: dict[str, Any] | None = None,
    ) -> APIResponse:
        """Call the remote LLM API.

        Args:
            messages: Message dict (format depends on provider)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            json_schema: Optional JSON schema for structured output

        Returns:
            Normalized APIResponse
        """
        ...

    @abstractmethod
    def _build_cached_messages(
        self,
        page_text: str,
        suffix_text: str,
        system_prompt: str | None,
    ) -> dict[str, Any]:
        """Build messages with cache control markers.

        Args:
            page_text: The page text (cached prefix)
            suffix_text: The suffix text (varies per request)
            system_prompt: Optional system prompt (cached at first breakpoint)

        Returns:
            Message dict suitable for _call_api()
        """
        ...
