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
from ..distributed.ray_utils.rate_limit import RateLimitConfig, TokenBucketRateLimiter
from ..distributed.hooks import tracing, hookable
from ..distributed.observability import TracingConfig
from ..vcm.events import (
    PageEvent,
    PageEvictedEvent,
    PageLoadedEvent,
    PageLoadFailedEvent,
)
from ..vcm.models import (
    ContextPageId,
    VirtualContextPage,
    VirtualPageTableState,
)
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
from .routing import TargetClientRouter
from ..agents.base import AgentManagerBase
from .config import LLMDeploymentConfig

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
    appended_tokens: list[int] = field(default_factory=list)
    appended_size: int = 0
    max_append_capacity: int = 0

    def __post_init__(self):
        self.max_append_capacity = self.page.size - len(self.page.tokens)

    def can_append(self, num_tokens: int) -> bool:
        return self.appended_size + num_tokens <= self.max_append_capacity

    def has_appended_context(self) -> bool:
        return self.appended_size > 0


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




@tracing(
    publish_key=lambda self: f"deployment:{getattr(self, '_deployment_id', 'remote')}",
    subscribe_key=lambda self: f"deployment:{getattr(self, '_deployment_id', 'remote')}",
)
@serving.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_queue_length": 5,
    },
    ray_actor_options={
        "num_gpus": 0,
    },
)
class RemoteLLMDeployment(AgentManagerBase):
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

    def __init__(
        self,
        config: RemoteLLMDeploymentConfig,
        deployment_config: LLMDeploymentConfig | None = None,
    ):
        # Initialize AgentManagerBase
        super().__init__(deployment_config=deployment_config)

        self.config = config

        # Page tracking (Layer 3 — same as VLLMDeployment)
        # Keyed by page_ref ("page_id:colony_id:tenant_id") for easier reconciliation with VCM state
        self.loaded_pages: dict[str, CachedPageEntry] = {}

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

        # Concurrency tracking (no semaphore — let the API provider's own
        # rate limits and the httpx connection pool control throughput).
        self._active_requests = 0
        self._rate_limiter = TokenBucketRateLimiter(RateLimitConfig(
            requests_per_second=config.throttle_rps,
            burst_size=config.throttle_burst,
        ))

        # TTL configuration
        self.ttl_seconds = config.get_ttl_seconds()

        # Event publishing infrastructure (initialized in initialize())
        self.redis_client = None
        self.redis_om = None
        self.event_namespace: str | None = None

        # Tokenizer for decode fallback (when page.text is None)
        self.tokenizer = None

        # Tracing
        self._deployment_id = config.get_deployment_name()
        self._tracing_facility = None

    @serving.initialize_deployment
    async def initialize(self) -> None:
        """Initialize the remote deployment and set up state.

        Called automatically by the serving framework after deployment.
        """
        await super().initialize()  # Initialize AgentManagerBase (capabilities, etc.)

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
            loaded_page_ids={},
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

            topics = {"vcm_page_events": {}}  # TODO: Make this configurable
            await self.redis_om.initialize_topics(topics)
            logger.info(f"Initialized Redis event publishing for {self.event_namespace}")
        except Exception as e:
            logger.warning(
                f"Failed to initialize Redis event publishing: {e}. "
                "Events will not be emitted."
            )
            self.redis_client = None
            self.redis_om = None

        # Initialize distributed tracing
        import os
        tracing_enabled = os.environ.get("TRACING_ENABLED", "").lower() in ("true", "1", "yes")
        if tracing_enabled:
            from .observability import ClusterTracingFacility
            self._tracing_facility = ClusterTracingFacility(
                config=TracingConfig(enabled=True),
                owner=self,
                service_name="RemoteLLMDeployment",
                deployment_name=self._deployment_id,
            )
            await self._tracing_facility.initialize()

        logger.info(
            f"RemoteLLMDeployment {self.client_id} initialized "
            f"(capacity={self.max_cached_tokens} tokens, ttl={self.config.ttl})"
        )

    @serving.on_app_ready
    async def on_ready(self):
        """Discover sibling deployment handles after all deployments are started: LLMCluster, VCM, AgentSystem, ToolManager, ConsciousnessManager."""
        await self.discover_handles()
        logger.info(f"RemoteLLMDeployment {self.client_id} handle discovery complete")

    async def get_replica_metadata(self) -> dict[str, Any]:
        """Report metadata for proxy routing (called by serving framework)."""
        return {"client_id": self.client_id}

    # -------------------------------------------------------------------------
    # Endpoints (same interface as VLLMDeployment)
    # -------------------------------------------------------------------------

    @serving.endpoint(
        router_class=TargetClientRouter,
        router_kwargs={"strip_routing_params": ["target_client_id"]}
    )
    @hookable
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
            serving.ensure_context(page.page_id, page.syscontext)

            # Check if page is already loaded
            page_key = VirtualPageTableState.get_page_ref(page.page_id)
            if page_key in self.loaded_pages:
                entry = self.loaded_pages[page_key]
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
            self._active_requests += 1
            try:
                response = await self._call_api(messages, max_tokens=1)
            finally:
                self._active_requests -= 1

            # Track locally (Layer 3)
            now = time.time()
            self.loaded_pages[page_key] = CachedPageEntry(
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
                        client_state.loaded_page_ids[page_key] = page.size
                        kv_cache_used = client_state.kv_cache_used
                    state.register_page_load(
                        page_id=page.page_id,
                        client_id=self.client_id,
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
            page_key = VirtualPageTableState.get_page_ref(page_id)
            entry: CachedPageEntry = self.loaded_pages.pop(page_key, None)
            if not entry:
                return True  # Already evicted

            serving.ensure_context(entry.page.page_id, entry.page.syscontext)

            self.cached_tokens_used -= entry.cached_tokens

            # Update distributed state (Layer 1)
            if self.state_manager:
                async for state in self.state_manager.write_transaction():
                    client_state: LLMClientState = state.client_states.get(self.client_id)
                    if client_state:
                        client_state.kv_cache_used -= entry.cached_tokens
                        client_state.loaded_page_ids.pop(page_key, None)
                    state.register_page_eviction(page_id, self.client_id)

            logger.info(f"Evicted page {page_key} from {self.client_id}")

            # Emit PageEvictedEvent for VCM reconciliation
            deployment_name = serving.get_my_deployment_name()
            await self._emit_page_event(
                PageEvictedEvent(
                    page_id=page_id,
                    deployment_name=deployment_name,
                    client_id=self.client_id,
                    timestamp=time.time(),
                    size=entry.cached_tokens,
                    reason="manual",
                )
            )

            return True

        except Exception as e:
            logger.error(f"Failed to evict page {page_key}: {e}", exc_info=True)
            return False

    @serving.endpoint
    @hookable
    async def infer_with_suffix(
        self,
        base_page_id: ContextPageId,
        request: InferenceRequest,
        suffix_tokens: list[int] | None = None,
        max_concurrent_per_page: int | None = None,
    ) -> InferenceResponse:
        """Inference using cached page prefix + task-specific suffix.

        This is the PRIMARY API for context composition, identical in signature
        to VLLMDeployment.infer_with_suffix().

        Multiple agents using the same page share the cached prefix at 0.1x cost
        (analogous to vLLM sharing base KV blocks via APC).

        Args:
            base_page_id: ID of the loaded base page
            suffix_tokens: Optional task-specific tokens to append
            request: InferenceRequest
            max_concurrent_per_page: Unused (kept for API compatibility)

        Returns:
            InferenceResponse with generated text and cost metadata
        """
        serving.ensure_context(request.request_id, request.syscontext)

        start_time = time.time()

        # Check if page is already loaded
        page_key = VirtualPageTableState.get_page_ref(base_page_id)
        entry = self.loaded_pages.get(page_key)
        if not entry:
            raise ValueError(
                f"Page {page_key} not loaded in {self.client_id}"
            )

        serving.ensure_context(entry.page.page_id, entry.page.syscontext)

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

        # Build messages with cache control markers
        messages = self._build_cached_messages(
            page_text=entry.text,
            suffix_text=full_suffix,
            system_prompt=self.config.system_prompt,
        )

        await self._throttle()
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

    @serving.endpoint
    @hookable
    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Perform inference with automatic context page loading.

        This endpoint mirrors VLLMDeployment.infer(). It checks which required
        pages are loaded, builds the prompt from their cached text, and calls
        the remote API.

        Args:
            request: Inference request with prompt and required page IDs

        Returns:
            Inference response with generated text and page fault info
        """
        serving.ensure_context(request.request_id, request.syscontext)

        start_time = time.time()
        page_faults = []

        try:
            # Update state: increment pending requests
            async for dep_state in self.state_manager.write_transaction():
                client_state = dep_state.client_states.get(self.client_id)
                if client_state:
                    client_state.pending_requests += 1
                    client_state.total_requests += 1

                    for page_id in request.context_page_ids:
                        if not client_state.has_page(page_id):
                            page_faults.append(page_id)
                            logger.warning(
                                f"Page fault: {page_id} not loaded in {self.client_id}"
                            )

            # Build prompt from loaded page texts + request prompt
            page_context_parts = []
            for page_id in request.context_page_ids:
                # Check if page is already loaded
                page_key = VirtualPageTableState.get_page_ref(page_id)

                entry = self.loaded_pages.get(page_key)
                if entry:
                    serving.ensure_context(entry.page.page_id, entry.page.syscontext)
                    page_context_parts.append(entry.text)
                    entry.last_access = time.time()
                    entry.ttl_expiry = time.time() + self.ttl_seconds
                    entry.access_count += 1

            prompt_text = request.prompt
            if page_context_parts:
                context_block = "\n\n".join(page_context_parts)
                prompt_text = f"{context_block}\n\n{prompt_text}"

            # Build messages (use _build_cached_messages with page context as prefix)
            messages = self._build_cached_messages(
                page_text="\n\n".join(page_context_parts) if page_context_parts else "",
                suffix_text=request.prompt,
                system_prompt=self.config.system_prompt,
            )

            await self._throttle()
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

            result = InferenceResponse(
                request_id=request.request_id,
                generated_text=response.content,
                tokens_generated=response.output_tokens,
                page_faults=page_faults,
                latency_ms=latency_ms,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cache_read_tokens": response.cache_read_input_tokens,
                    "cache_write_tokens": response.cache_creation_input_tokens,
                    "cost_usd": response.cost_usd,
                    "provider": self.config.provider,
                    "model": self.config.model_name,
                },
            )

            logger.info(
                f"Inference completed: request_id={request.request_id}, "
                f"tokens={response.output_tokens}, latency={latency_ms:.2f}ms, "
                f"page_faults={len(page_faults)}, cost=${response.cost_usd:.4f}"
            )

            return result

        except Exception as e:
            async for dep_state in self.state_manager.write_transaction():
                client_state = dep_state.client_states.get(self.client_id)
                if client_state:
                    client_state.error_count += 1
                    client_state.last_error = str(e)
            logger.error(
                f"Inference failed for request {request.request_id}: {e}",
                exc_info=True,
            )
            raise

        finally:
            async for dep_state in self.state_manager.write_transaction():
                client_state = dep_state.client_states.get(self.client_id)
                if client_state:
                    client_state.pending_requests -= 1

    @serving.endpoint
    async def migrate_page(
        self,
        page_id: ContextPageId,
        target_deployment_name: str,
        target_client_id: LLMClientId,
    ) -> bool:
        """Migrate a page from this deployment to another deployment.

        Loads the page in the target deployment, then evicts it from this one.
        Same pattern as VLLMDeployment.migrate_page().

        Args:
            page_id: ID of page to migrate
            target_deployment_name: Name of the target deployment
            target_client_id: ID of target LLM client

        Returns:
            True if migration succeeded
        """
        try:
            # Check if page is already loaded
            page_key = VirtualPageTableState.get_page_ref(page_id)
            if page_key not in self.loaded_pages:
                raise ValueError(f"Page {page_key} not loaded in {self.client_id}")

            entry = self.loaded_pages[page_key]
            page = entry.page
            serving.ensure_context(page.page_id, page.syscontext)

            logger.info(
                f"Migrating page {page_key} from {self.client_id} to {target_client_id}"
            )

            from ..system import get_llm_cluster
            llm_cluster_handle = get_llm_cluster()

            try:
                success = await llm_cluster_handle.load_page(
                    page=page,
                    deployment_name=target_deployment_name,
                    client_id=target_client_id,
                )
                if not success:
                    raise RuntimeError("Target deployment failed to load page")
            except Exception as e:
                logger.error(f"Failed to load page in target deployment: {e}")
                raise RuntimeError(f"Page migration failed: {e}") from e

            await self.evict_page(page_id)

            logger.info(
                f"Successfully migrated page {page_key} from "
                f"{self.client_id} to {target_client_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Error during page migration: {e}", exc_info=True)
            raise

    @serving.endpoint
    async def append_context_to_page(
        self,
        page_id: ContextPageId,
        tokens: list[int],
    ) -> bool:
        """Append task-specific context to a loaded page.

        For remote deployments, appended tokens are tracked locally and composed
        into messages during inference (unlike vLLM where they modify KV cache).

        Args:
            page_id: ID of the page to append to
            tokens: Token IDs to append

        Returns:
            True if successful
        """
        try:
            # Check if page is already loaded
            page_key = VirtualPageTableState.get_page_ref(page_id)
            if page_key not in self.loaded_pages:
                logger.error(f"Cannot append to page {page_key}: page not loaded")
                return False

            entry = self.loaded_pages[page_key]
            serving.ensure_context(entry.page.page_id, entry.page.syscontext)

            if not entry.can_append(len(tokens)):
                logger.error(
                    f"Cannot append {len(tokens)} tokens to page {page_key}: "
                    f"would exceed capacity (current: {entry.appended_size}, "
                    f"max: {entry.max_append_capacity})"
                )
                return False

            entry.appended_tokens.extend(tokens)
            entry.appended_size += len(tokens)

            logger.info(
                f"Appended {len(tokens)} tokens to page {page_key} "
                f"(total appended: {entry.appended_size}/{entry.max_append_capacity})"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to append context to page {page_key}: {e}", exc_info=True
            )
            return False

    @serving.endpoint
    async def remove_appended_context(
        self,
        page_id: ContextPageId,
    ) -> bool:
        """Remove appended context from a loaded page, resetting to base page.

        Args:
            page_id: ID of the page to reset

        Returns:
            True if successful
        """
        try:
            page_key = VirtualPageTableState.get_page_ref(page_id)
            if page_key not in self.loaded_pages:
                logger.warning(
                    f"Cannot remove appended context: page {page_key} not loaded"
                )
                return True

            entry = self.loaded_pages[page_key]
            serving.ensure_context(entry.page.page_id, entry.page.syscontext)

            if not entry.has_appended_context():
                logger.debug(f"Page {page_key} has no appended context to remove")
                return True

            removed_size = entry.appended_size
            entry.appended_tokens.clear()
            entry.appended_size = 0

            logger.info(
                f"Removed {removed_size} appended tokens from page {page_key}, "
                f"reset to base page ({entry.cached_tokens} tokens)"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to remove appended context from page {page_key}: {e}",
                exc_info=True,
            )
            return False

    @serving.endpoint
    async def get_kv_metrics(self) -> KVCacheMetrics:
        """Get current KV cache metrics.

        Returns:
            KVCacheMetrics with current statistics
        """
        async for dep_state in self.state_manager.read_transaction():
            client_state = dep_state.client_states.get(self.client_id)
            if client_state:
                self.kv_metrics.update_kv_cache_usage(
                    client_state.kv_cache_used,
                    client_state.kv_cache_capacity,
                )

        return self.kv_metrics.model_copy(deep=True)

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

        for page_key, entry in list(self.loaded_pages.items()):
            idle_s = now - entry.last_access
            # Refresh if 80% of TTL has elapsed
            if idle_s > self.ttl_seconds * 0.8:
                try:
                    messages = self._build_cached_messages(
                        page_text=entry.text,
                        suffix_text="Acknowledge.",
                        system_prompt=self.config.system_prompt,
                    )
                    self._active_requests += 1
                    try:
                        await self._call_api(messages, max_tokens=1)
                    finally:
                        self._active_requests -= 1
                    entry.last_access = time.time()
                    entry.ttl_expiry = time.time() + self.ttl_seconds
                    refreshed += 1
                except Exception as e:
                    logger.warning(f"Keepalive failed for page {page_key}: {e}")

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

            with serving.restore_execution_context(entry.page.syscontext):
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
                topic="vcm_page_events",  # TODO: Make this configurable
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

    async def _throttle(self) -> None:
        """Rate-limit before acquiring a semaphore slot."""
        await self._rate_limiter.acquire()

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
        top_p: float | None = None,
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

    async def cleanup(self):
        """Cleanup resources on shutdown."""
        # TODO: Implement any necessary cleanup, such as closing Redis connections
