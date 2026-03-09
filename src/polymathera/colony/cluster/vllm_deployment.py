"""vLLM deployment implementation for the LLM cluster.

This module provides the VLLMDeployment class which wraps a vLLM instance
as a polymathera.ray_utils.serving deployment with context caching capabilities.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

if TYPE_CHECKING:
    from .registry import QuantizationMethod

from .gpus import GPUMetricsCollector, get_gpu_memory_info
from ..distributed.ray_utils import serving
from ..agents.base import AgentManagerBase
from .circuit_breakers import inference_circuit, page_loading_circuit
from .model_loader import S3ModelLoader
from .routing import TargetClientRouter
from .models import (
    ContextPageState,
    InferenceRequest,
    InferenceResponse,
    KVCacheMetrics,
    LLMClientId,
    LLMClientState,
    LoadedContextPage,
)
from .config import LLMDeploymentConfig
from ..vcm.models import VirtualContextPage, ContextPageId
from ..vcm.events import PageEvent, PageLoadedEvent, PageEvictedEvent, PageLoadFailedEvent
from .registry import ModelRegistry
from .tokenization import get_tokenizer_for_model, HuggingFaceTokenizer, TiktokenTokenizer
from ..system import get_llm_cluster

logger = logging.getLogger(__name__)


def _calculate_kv_cache_capacity(
    gpu_id: int = 0,
    gpu_memory_utilization: float = 0.9,
    kv_cache_fraction: float = 0.4,
    bytes_per_token: int = 2,
) -> int:
    """Calculate KV cache capacity based on available GPU memory.

    Args:
        gpu_id: GPU device ID
        gpu_memory_utilization: Fraction of GPU memory vLLM will use
        kv_cache_fraction: Fraction of vLLM's memory to use for KV cache
        bytes_per_token: Bytes per token for KV cache (depends on quantization)

    Returns:
        KV cache capacity in tokens

    Note:
        vLLM uses approximately:
        - Model weights: ~50-60% of allocated memory
        - KV cache: ~40% of allocated memory
        - Activation memory: remaining

        For a 16GB GPU with 0.9 utilization and fp16:
        - Total available: 14.4GB
        - KV cache: ~5.76GB (40%)
        - At 2 bytes/token (fp16): ~3M tokens
        - At 1 byte/token (fp8/int8): ~6M tokens
    """
    try:
        memory_info = get_gpu_memory_info(gpu_id)
        total_memory_bytes = memory_info['total']

        if total_memory_bytes == 0:
            logger.warning(f"GPU {gpu_id} has no memory info, using default 128k KV cache")
            return 128 * 1024

        # Calculate memory vLLM will use
        vllm_memory = total_memory_bytes * gpu_memory_utilization

        # Calculate KV cache memory
        kv_cache_memory = vllm_memory * kv_cache_fraction

        # Convert to tokens based on quantization
        kv_cache_tokens = int(kv_cache_memory / bytes_per_token)

        logger.info(
            f"Calculated KV cache capacity for GPU {gpu_id}: "
            f"{kv_cache_tokens:,} tokens ({kv_cache_memory / 1e9:.2f}GB / "
            f"{total_memory_bytes / 1e9:.2f}GB total, {bytes_per_token} bytes/token)"
        )

        return kv_cache_tokens

    except Exception as e:
        logger.warning(f"Failed to calculate KV cache capacity: {e}, using default 128k")
        return 128 * 1024


@serving.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_queue_length": 5,
    },
    ray_actor_options={
        "num_gpus": 1,  # Default to single GPU, override in production
    },
)
class VLLMDeployment(AgentManagerBase):
    """vLLM-based LLM deployment with context caching.

    This class wraps a vLLM AsyncLLMEngine and provides:
    1. Context page management (load/evict pages in KV cache)
    2. Continuous batching inference (handled by vLLM internally)
    3. Page migration between deployments
    4. Health monitoring and statistics

    The deployment is designed to work with colony.distributed.ray_utils.serving
    and can be deployed as part of an Application with custom routing.

    Architecture:
    - Each VLLMDeployment instance manages a single vLLM engine
    - The KV cache is divided into slots for context pages
    - Pages can be preloaded and reused across multiple requests
    - Custom routing directs requests to instances with required pages
    - vLLM handles continuous batching automatically for efficiency

    Example:
        ```python
        from colony.distributed.ray_utils import serving
        from polymathera.colony import VLLMDeployment

        # Deploy vLLM instance
        app = serving.Application(name="llm-cluster")
        app.add_deployment(
            VLLMDeployment.bind(
                model_name="meta-llama/Llama-3.1-8B",
                kv_cache_capacity=256*1024,
            ),
            ray_actor_options={"num_gpus": 2},
        )
        await app.start()
        ```
    """

    def __init__(
        self,
        model_name: str,
        kv_cache_capacity: int | None = None,
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        gpu_id: int = 0,
        quantization: str | None = None,
        s3_bucket: str | None = None,
        s3_retry_attempts: int = 10,
        deployment_config: LLMDeploymentConfig | None = None,
    ):
        """Initialize VLLMDeployment.

        Args:
            model_name: HuggingFace model name or path to model
            kv_cache_capacity: Maximum KV cache size in tokens (auto-calculated if None)
            max_model_len: Maximum model context length (from registry if None)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory for vLLM (default: 0.9)
            trust_remote_code: Whether to trust remote code when loading model
            gpu_id: GPU device ID for this deployment (default: 0)
            quantization: Quantization method (awq, gptq, fp8, etc.) or None for auto-detect
            s3_bucket: S3 bucket name for model loading (None to load from HuggingFace)
            s3_retry_attempts: Number of retry attempts for S3 downloads (default: 10)

        Note:
            vLLM handles continuous batching automatically. No need for manual
            BatchQueueManager - just submit requests concurrently and vLLM will
            batch them optimally.
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.gpu_id = gpu_id
        self.quantization = quantization
        self.s3_bucket = s3_bucket
        self.s3_retry_attempts = s3_retry_attempts

        # Get model parameters from registry
        model_params = ModelRegistry.get_model(model_name)
        if model_params:
            self.max_model_len = max_model_len or model_params.context_window
            self.model_params = model_params
            logger.info(
                f"Loaded model config from registry: {model_name}, "
                f"context_window={model_params.context_window}"
            )
        else:
            # Fallback if model not in registry
            self.max_model_len = max_model_len or 32 * 1024
            self.model_params = None
            logger.warning(
                f"Model {model_name} not in registry, using max_model_len={self.max_model_len}"
            )

        # Determine bytes per token for KV cache calculation
        if model_params and quantization:
            # Try to parse quantization string to enum
            from .registry import QuantizationMethod
            try:
                quant_method = QuantizationMethod[quantization.upper().replace("-", "_")]
                bytes_per_token = model_params.get_bytes_per_token(quant_method)
            except (KeyError, AttributeError):
                bytes_per_token = 2  # Default to fp16
                logger.warning(f"Unknown quantization method: {quantization}, using 2 bytes/token")
        elif model_params:
            bytes_per_token = model_params.get_bytes_per_token()
        else:
            bytes_per_token = 2  # Default to fp16

        # Calculate KV cache capacity based on GPU memory if not provided
        if kv_cache_capacity is None:
            self.kv_cache_capacity = _calculate_kv_cache_capacity(
                gpu_id=gpu_id,
                gpu_memory_utilization=gpu_memory_utilization,
                bytes_per_token=bytes_per_token,
            )
        else:
            self.kv_cache_capacity = kv_cache_capacity

        # Will be set during initialization
        self.engine: AsyncLLMEngine | None = None
        self.client_id: LLMClientId | None = None
        self.tokenizer: HuggingFaceTokenizer | TiktokenTokenizer | None = None
        self.gpu_metrics: GPUMetricsCollector | None = None
        self.state_manager = None  # StateManager for deployment state

        # Page management
        self.loaded_pages: dict[ContextPageId, LoadedContextPage] = {}
        self.next_kv_slot = 0

        # Concurrency control for context composition
        self._page_semaphores: dict[ContextPageId, asyncio.Semaphore] = {}
        self._max_concurrent_per_page = 20  # Configurable per-page concurrency limit

        # KV cache metrics
        self.kv_metrics = KVCacheMetrics(
            kv_cache_capacity_tokens=self.kv_cache_capacity
        )

        # Monitoring
        self._monitoring_task: asyncio.Task | None = None
        self._monitoring_interval = 30.0  # seconds

        # Event publishing infrastructure (initialized in initialize())
        self.redis_client = None  # RedisClient
        self.redis_om = None  # RedisOM for event publishing
        self.event_namespace: str | None = None

        # Initialize AgentManagerBase
        super().__init__(deployment_config=deployment_config)

    @serving.initialize_deployment
    async def initialize(self):
        """Initialize the vLLM engine and set up state.

        This is called automatically by the serving framework after deployment.
        """
        logger.info(f"Initializing VLLMDeployment with model {self.model_name}")

        # Generate client ID
        import ray
        self.client_id = LLMClientId(f"vllm-{ray.get_runtime_context().get_actor_id()}")

        # Download model from S3 if specified
        model_path = self.model_name
        tokenizer_path = self.model_name
        if self.s3_bucket:
            loader = S3ModelLoader(
                bucket=self.s3_bucket,
                model_name=self.model_name,
                retry_attempts=self.s3_retry_attempts
            )
            downloaded_path = await loader.download_and_extract()
            if downloaded_path:
                model_path = downloaded_path
                tokenizer_path = downloaded_path  # Use local path for tokenizer too
                logger.info(f"Using local model and tokenizer from S3: {downloaded_path}")
            else:
                logger.warning(f"Failed to download model from S3, falling back to HuggingFace: {self.model_name}")

        # Initialize tokenizer BEFORE vLLM engine (async, non-blocking)
        self.tokenizer = await get_tokenizer_for_model(
            model_name_or_path=tokenizer_path,
            model_name=self.model_name  # Pass original name for registry lookup
        )
        logger.info(f"Initialized tokenizer for {self.model_name}: {type(self.tokenizer).__name__}")

        # Initialize vLLM engine
        logger.info(f"Initializing vLLM engine for model {model_path}")
        engine_args = AsyncEngineArgs(
            model=model_path,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=self.trust_remote_code,
            gpu_memory_utilization=self.gpu_memory_utilization,
            quantization=self.quantization,  # vLLM quantization parameter
            # Enable prefix caching for context reuse
            enable_prefix_caching=True,
        )

        logger.info(f"Engine arguments prepared: {engine_args}")
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"vLLM engine initialized for model {self.model_name}")

        # Initialize GPU metrics collector
        try:
            self.gpu_metrics = GPUMetricsCollector(gpu_id=self.gpu_id)
            logger.info(f"GPU metrics collector initialized for GPU {self.gpu_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU metrics collector: {e}")
            self.gpu_metrics = None

        # Initialize deployment-level StateManager
        app_name = serving.get_my_app_name()
        deployment_name = serving.get_my_deployment_name()
        logger.info(f"Setting up StateManager for {app_name}.{deployment_name}")

        from ..distributed import get_polymathera
        from .models import VLLMDeploymentState
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
            model_name=self.model_name,
            kv_cache_capacity=self.kv_cache_capacity,
            kv_cache_used=0,
            loaded_page_ids=set(),
            last_heartbeat=time.time(),
        )
        async for state in self.state_manager.write_transaction():
            state.client_states[self.client_id] = initial_state

        # Initialize Redis and event publishing infrastructure
        try:
            self.redis_client = await polymathera.get_redis_client()
            self.event_namespace = f"vllm_events:{deployment_name}"

            from ..distributed.redis_utils import RedisOM
            self.redis_om = RedisOM(
                redis_client=self.redis_client,
                namespace=self.event_namespace,
            )

            # Initialize event topics
            topics = {"vcm_page_events": {}}
            await self.redis_om.initialize_topics(topics)

            logger.info(f"Initialized Redis event publishing for {self.event_namespace}")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis event publishing: {e}. Events will not be emitted.")
            self.redis_client = None
            self.redis_om = None

        #Start periodic GPU monitoring
        self._monitoring_task = asyncio.create_task(self._periodic_gpu_monitoring())

        logger.info(f"VLLMDeployment {self.client_id} initialized with state tracking")

        # Initialize AgentManagerBase
        await super().initialize()

    @serving.on_app_ready
    async def on_ready(self):
        """Discover sibling deployment handles after all deployments are started."""
        await self.discover_handles()
        logger.info(f"VLLMDeployment {self.client_id} handle discovery complete")

    @serving.endpoint
    @inference_circuit
    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Perform inference with automatic context page loading.

        This endpoint:
        1. Checks which required pages are loaded
        2. Loads missing pages into KV cache (may evict others)
        3. Performs inference using vLLM
        4. Returns results with page fault information

        Args:
            request: Inference request with prompt and required page IDs

        Returns:
            Inference response with generated text and page fault info
        """
        start_time = time.time()
        page_faults = []

        try:
            # Update state in StateManager
            async for dep_state in self.state_manager.write_transaction():
                client_state = dep_state.client_states.get(self.client_id)
                if client_state:
                    client_state.pending_requests += 1
                    client_state.total_requests += 1

                    # Check for missing pages (page faults)
                    for page_id in request.context_page_ids:
                        if not client_state.has_page(page_id):
                            page_faults.append(page_id)
                            logger.warning(f"Page fault: {page_id} not loaded in {self.client_id}")

            # Load missing pages before inference
            # NOTE: We load pages by warming up vLLM's prefix cache
            for page_id in page_faults:
                # Try to find the page in our cluster state
                # For now, we'll just log - the router should have pre-loaded pages
                logger.warning(
                    f"Page fault during inference: {page_id} not in cache. "
                    f"Router should have loaded this page beforehand."
                )

            # Build sampling params
            # Note: vLLM supports guided decoding via guided_json parameter (in newer versions)
            sampling_params_dict = {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
            }

            # Add JSON schema for structured output if provided
            # Try with guided_json first (newer vLLM), fall back without it
            if request.json_schema:
                sampling_params_dict["guided_json"] = request.json_schema
                try:
                    sampling_params = SamplingParams(**sampling_params_dict)
                    logger.info(
                        f"Using guided JSON generation for request {request.request_id}"
                    )
                except TypeError as e:
                    # guided_json not supported in this vLLM version
                    logger.warning(
                        f"guided_json parameter not supported in this vLLM version, "
                        f"falling back to standard generation: {e}"
                    )
                    del sampling_params_dict["guided_json"]
                    sampling_params = SamplingParams(**sampling_params_dict)
            else:
                sampling_params = SamplingParams(**sampling_params_dict)

            # Prepare full prompt with context pages
            # Reconstruct prompt with loaded page tokens (vLLM will use prefix cache)
            full_prompt = request.prompt
            if request.context_page_ids:
                # Prepend page tokens to prompt (vLLM will cache this prefix)
                page_tokens = []
                for page_id in request.context_page_ids:
                    if page_id in self.loaded_pages:
                        loaded_page = self.loaded_pages[page_id]
                        page_tokens.extend(loaded_page.page.tokens)
                        loaded_page.access_count += 1
                        loaded_page.last_access_time = time.time()

                # Convert tokens back to text for vLLM
                # NOTE: This is a simplified approach - production should use tokenizer
                if page_tokens:
                    # For now, we trust that the prompt already includes context
                    # In production, we'd decode page_tokens and prepend to prompt
                    pass

            # Generate using vLLM with automatic prefix caching
            # vLLM will automatically cache and reuse the prompt prefix across requests
            # engine.generate() returns an async generator that yields intermediate results
            final_output = None
            async for output in self.engine.generate(
                prompt=full_prompt,
                sampling_params=sampling_params,
                request_id=request.request_id,
            ):
                final_output = output

            # Extract generated text from final output
            generated_text = final_output.outputs[0].text if final_output and final_output.outputs else ""
            tokens_generated = len(final_output.outputs[0].token_ids) if final_output and final_output.outputs else 0

            latency_ms = (time.time() - start_time) * 1000

            response = InferenceResponse(
                request_id=request.request_id,
                generated_text=generated_text,
                tokens_generated=tokens_generated,
                page_faults=page_faults,
                latency_ms=latency_ms,
            )

            logger.info(
                f"Inference completed: request_id={request.request_id}, "
                f"tokens={tokens_generated}, latency={latency_ms:.2f}ms, "
                f"page_faults={len(page_faults)}"
            )

            return response

        except Exception as e:
            async for dep_state in self.state_manager.write_transaction():
                client_state = dep_state.client_states.get(self.client_id)
                if client_state:
                    client_state.error_count += 1
                    client_state.last_error = str(e)
            logger.error(f"Inference failed for request {request.request_id}: {e}", exc_info=True)
            raise

        finally:
            async for dep_state in self.state_manager.write_transaction():
                client_state = dep_state.client_states.get(self.client_id)
                if client_state:
                    client_state.pending_requests -= 1

    @serving.endpoint(
        router_class=TargetClientRouter,
        router_kwargs={"strip_routing_params": ["target_client_id"]}
    )
    @page_loading_circuit
    async def load_page(self, page: VirtualContextPage) -> bool:
        """Load a context page into the KV cache.

        This endpoint loads a context page into the KV cache, making it available
        for subsequent inference requests. If the cache is full, it may evict
        the least recently used page.

        Args:
            page: Context page to load

        Returns:
            True if page was loaded successfully, False otherwise
        """
        try:
            # Check if page is already loaded
            if page.page_id in self.loaded_pages:
                loaded_page = self.loaded_pages[page.page_id]
                if loaded_page.state == ContextPageState.LOADED:
                    logger.info(f"Page {page.page_id} already loaded in {self.client_id}")
                    loaded_page.access_count += 1
                    loaded_page.last_access_time = time.time()
                    return True

            # Check if we have capacity (read from StateManager)
            kv_cache_used = 0
            async for dep_state in self.state_manager.read_transaction():
                client_state = dep_state.client_states.get(self.client_id)
                if client_state:
                    kv_cache_used = client_state.kv_cache_used

            if kv_cache_used + page.size > self.kv_cache_capacity:
                # Need to evict pages to make room
                evicted = await self._evict_pages_for_capacity(page.size)
                if not evicted:
                    logger.error(f"Failed to evict pages for {page.page_id}")
                    return False

            # Allocate KV cache slot
            kv_slot = self.next_kv_slot
            self.next_kv_slot += 1

            # Create loaded page entry
            loaded_page = LoadedContextPage(
                page=page,
                client_id=self.client_id,
                state=ContextPageState.LOADING,
                kv_cache_slot=kv_slot,
                load_time=time.time(),
                last_access_time=time.time(),
                access_count=0,
            )

            self.loaded_pages[page.page_id] = loaded_page

            # TODO: Integrate with vLLM's actual KV cache management
            # TODO: Do we need to pin the page in memory?
            # TODO: Do we even have to warm up the cache, or that will happen the first time we use it?
            # Load the page into vLLM's KV cache via prefix caching
            # We warm up the cache by running a dummy generation with the page tokens
            await self._warm_up_page_in_cache(page, loaded_page)

            loaded_page.state = ContextPageState.LOADED

            # Update state in StateManager and register page load
            if self.state_manager:
                kv_cache_used = 0
                async for state in self.state_manager.write_transaction():
                    client_state = state.client_states.get(self.client_id)
                    if client_state:
                        client_state.kv_cache_used += page.size
                        client_state.loaded_page_ids.add(page.page_id)
                        kv_cache_used = client_state.kv_cache_used
                    state.register_page_load(page.page_id, self.client_id, page.tenant_id)

            logger.info(
                f"Loaded page {page.page_id} into {self.client_id} "
                f"(slot={kv_slot}, size={page.size}, capacity_used={kv_cache_used}/{self.kv_cache_capacity})"
            )

            # Emit PageLoadedEvent for VCM reconciliation
            load_duration_ms = (time.time() - loaded_page.load_time) * 1000
            deployment_name = serving.get_my_deployment_name()
            await self._emit_page_event(PageLoadedEvent(
                page_id=page.page_id,
                deployment_name=deployment_name,
                client_id=self.client_id,
                tenant_id=page.tenant_id,
                timestamp=time.time(),
                size=page.size,
                kv_cache_slot=kv_slot,
                load_duration_ms=load_duration_ms,
            ))

            return True

        except Exception as e:
            logger.error(f"Failed to load page {page.page_id}: {e}", exc_info=True)

            # Emit PageLoadFailedEvent
            deployment_name = serving.get_my_deployment_name()
            await self._emit_page_event(PageLoadFailedEvent(
                page_id=page.page_id,
                deployment_name=deployment_name,
                client_id=self.client_id,
                tenant_id=page.tenant_id,
                timestamp=time.time(),
                error=str(e),
                error_type=type(e).__name__,
            ))

            return False

    @serving.endpoint
    async def evict_page(self, page_id: ContextPageId) -> bool:
        """Evict a context page from the KV cache.

        Args:
            page_id: ID of the page to evict

        Returns:
            True if page was evicted successfully, False otherwise
        """
        try:
            if page_id not in self.loaded_pages:
                logger.warning(f"Page {page_id} not found in {self.client_id}")
                return True  # Already evicted

            loaded_page = self.loaded_pages[page_id]
            loaded_page.state = ContextPageState.EVICTING

            # NOTE: vLLM handles cache eviction automatically via its LRU policy
            # We cannot explicitly evict from vLLM's KV cache - it manages this internally
            # Our evict_page() mainly serves to update our tracking state
            # vLLM will evict the cached prefix when it runs out of space

            # Update state in StateManager
            if self.state_manager:
                async for state in self.state_manager.write_transaction():
                    client_state = state.client_states.get(self.client_id)
                    if client_state:
                        client_state.kv_cache_used -= loaded_page.page.size
                        client_state.loaded_page_ids.discard(page_id)
                    state.register_page_eviction(page_id, self.client_id)

            loaded_page.state = ContextPageState.EVICTED
            page_size = loaded_page.page.size
            tenant_id = loaded_page.page.tenant_id
            del self.loaded_pages[page_id]

            logger.info(f"Evicted page {page_id} from {self.client_id}")

            # Emit PageEvictedEvent for VCM reconciliation
            deployment_name = serving.get_my_deployment_name()
            await self._emit_page_event(PageEvictedEvent(
                page_id=page_id,
                deployment_name=deployment_name,
                client_id=self.client_id,
                tenant_id=tenant_id,
                timestamp=time.time(),
                size=page_size,
                reason="manual",  # TODO: Add reason parameter to evict_page()
            ))

            return True

        except Exception as e:
            logger.error(f"Failed to evict page {page_id}: {e}", exc_info=True)
            return False

    @serving.endpoint
    async def get_state(self) -> LLMClientState:
        """Get current state of this LLM client.

        Returns:
            Current client state including loaded pages and metrics
        """
        # Update heartbeat and return state from StateManager
        current_state = None
        async for dep_state in self.state_manager.write_transaction():
            client_state = dep_state.client_states.get(self.client_id)
            if client_state:
                client_state.last_heartbeat = time.time()
                current_state = client_state.model_copy(deep=True)

        if current_state is None:
            raise RuntimeError(f"Client state not found in StateManager for {self.client_id}")

        return current_state

    async def _evict_pages_for_capacity(self, required_size: int) -> bool:
        """Evict pages to free up required capacity.

        Uses LRU (Least Recently Used) eviction policy.

        Args:
            required_size: Number of tokens needed

        Returns:
            True if sufficient capacity was freed, False otherwise
        """
        # Sort pages by last access time (LRU)
        pages_by_lru = sorted(
            self.loaded_pages.values(),
            key=lambda p: p.last_access_time,
        )

        freed_size = 0
        evicted_pages = []

        for loaded_page in pages_by_lru:
            # Read available capacity from StateManager
            available_capacity = 0
            async for dep_state in self.state_manager.read_transaction():
                client_state = dep_state.client_states.get(self.client_id)
                if client_state:
                    available_capacity = client_state.get_available_cache_capacity()

            if available_capacity + freed_size >= required_size:
                break

            # Evict this page
            await self.evict_page(loaded_page.page.page_id)
            freed_size += loaded_page.page.size
            evicted_pages.append(loaded_page.page.page_id)

        logger.info(
            f"Evicted {len(evicted_pages)} pages to free {freed_size} tokens: {evicted_pages}"
        )

        # Read final available capacity from StateManager
        available_capacity = 0
        async for dep_state in self.state_manager.read_transaction():
            client_state = dep_state.client_states.get(self.client_id)
            if client_state:
                available_capacity = client_state.get_available_cache_capacity()

        return available_capacity >= required_size

    async def _warm_up_page_in_cache(self, page: VirtualContextPage, loaded_page: LoadedContextPage) -> None:
        """Warm up vLLM's prefix cache by loading page tokens.

        This method runs a dummy generation with the page tokens as the prompt prefix,
        causing vLLM's prefix caching to store the KV cache for these tokens.

        Args:
            page: Context page to load
            loaded_page: LoadedContextPage tracking entry
        """
        try:
            # Decode page tokens to text
            page_text = self.tokenizer.decode(page.tokens)

            # Run dummy generation with minimal output to warm up cache
            # vLLM will cache the KV states for the page_text prefix
            warmup_params = SamplingParams(
                max_tokens=1,  # Minimal generation
                temperature=0.0,  # Deterministic
            )

            # Fire and forget - we don't care about the output
            _ = await self.engine.generate(
                prompt=page_text,
                sampling_params=warmup_params,
                request_id=f"warmup-{page.page_id}",
            )

            logger.debug(
                f"Warmed up vLLM cache for page {page.page_id} "
                f"({page.size} tokens)"
            )

        except Exception as e:
            logger.warning(
                f"Failed to warm up cache for page {page.page_id}: {e}. "
                f"Page will still be tracked but may cause cache miss on first use."
            )

    async def _periodic_gpu_monitoring(self):
        """Periodically collect and log GPU metrics."""
        while True:
            try:
                await asyncio.sleep(self._monitoring_interval)

                if self.gpu_metrics:
                    metrics = self.gpu_metrics.collect()
                    if metrics:
                        logger.info(
                            f"GPU {metrics.gpu_id} ({metrics.gpu_name}): "
                            f"Memory: {metrics.memory_used_mb:.1f}/{metrics.memory_total_mb:.1f}MB "
                            f"({metrics.memory_utilization_pct:.1f}%), "
                            f"GPU Util: {metrics.gpu_utilization_pct:.1f}%"
                            + (f", Temp: {metrics.temperature_celsius}°C" if metrics.temperature_celsius else "")
                            + (f", Power: {metrics.power_draw_watts:.1f}W" if metrics.power_draw_watts else "")
                        )
            except asyncio.CancelledError:
                logger.info("GPU monitoring task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in GPU monitoring: {e}", exc_info=True)

    async def _find_deployment_name(
        self,
        target_client_id: LLMClientId,
    ) -> bool:
        target_deployment_name = None
        async for dep_state in self.state_manager.read_transaction():
            # Search across all client states to find the target client
            for client_state in dep_state.client_states.values():
                if client_state.client_id == target_client_id:
                    target_deployment_name = client_state.deployment_name
                    break

        if not target_deployment_name:
            raise ValueError(f"Target client {target_client_id} not found in cluster state")
        return target_deployment_name

    @serving.endpoint
    async def migrate_page(
        self,
        page_id: ContextPageId,
        target_deployment_name: str,
        target_client_id: LLMClientId,
    ) -> bool:
        """Migrate a page from this deployment to another deployment.

        This method implements page migration for distributed context management,
        allowing pages to be moved between deployments based on access patterns.

        The migration process:
        1. Verify page is loaded in this deployment
        2. Get target deployment handle
        3. Load page in target deployment
        4. Evict from source deployment (this one)

        Args:
            page_id: ID of page to migrate
            target_client_id: ID of target LLM client

        Returns:
            True if migration succeeded, False otherwise

        Raises:
            ValueError: If page not found or target client not found
            RuntimeError: If migration fails
        """
        try:
            # Check if page is loaded here
            if page_id not in self.loaded_pages:
                raise ValueError(f"Page {page_id} not loaded in {self.client_id}")

            loaded_page = self.loaded_pages[page_id]
            page = loaded_page.page

            logger.info(
                f"Migrating page {page_id} from {self.client_id} to {target_client_id}"
            )

            # Get LLMCluster handle
            llm_cluster_handle = get_llm_cluster()

            # Load page in target deployment with specific client targeting
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

            # Evict from this deployment
            await self.evict_page(page_id)

            logger.info(
                f"Successfully migrated page {page_id} from {self.client_id} to {target_client_id}"
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
        """Append task-specific context to a loaded page's KV cache.

        This method enables dynamic context composition by temporarily appending
        tokens (e.g., agent instructions, tool descriptions, current state) to
        a base page without modifying the immutable VirtualContextPage.

        The appended context is stored in the KV cache and can be removed after
        inference to prepare for the next request with different task context.

        Args:
            page_id: ID of the page to append to
            tokens: Token IDs to append (task-specific context)

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If page not loaded or insufficient capacity

        Example:
            ```python
            # Load base page (code from file.py)
            await vllm.load_page(base_page)

            # Append agent instructions
            task_tokens = tokenize("You are a code reviewer. Check for bugs...")
            await vllm.append_context_to_page("file.py", task_tokens)

            # Run inference
            response = await vllm.infer(request)

            # Remove appended context for next agent
            await vllm.remove_appended_context("file.py")
            ```

        TODO (VCM Integration):
            - Implement vLLM KV cache manipulation to append tokens
            - vLLM prefix caching needs to support this pattern:
              * Cache base page tokens permanently
              * Allow temporary suffix that can be replaced
            - Investigate vLLM's KV cache API for partial updates
            - Add validation for capacity limits
            - Consider chunked appending for large contexts
            - Integrate with agent system for automatic lifecycle
            - Add metrics tracking (append count, average size, etc.)
            - Handle concurrent appends (lock page during modification)
            - Support multiple append "layers" (base + agent + tool + state)

        Design Notes:
            This is critical for agent efficiency because:
            - Base context (code files) can be reused across many agents
            - Each agent appends its own instructions without duplicating base
            - Maximizes KV cache utilization (one base, many tasks)
            - Reduces page loading overhead dramatically

            The design must evolve with VCM and agent system requirements.
            Keep implementation flexible and well-documented.
        """
        try:
            # Check if page is loaded
            if page_id not in self.loaded_pages:
                logger.error(f"Cannot append to page {page_id}: page not loaded")
                return False

            loaded_page = self.loaded_pages[page_id]

            # Check capacity
            if not loaded_page.can_append(len(tokens)):
                logger.error(
                    f"Cannot append {len(tokens)} tokens to page {page_id}: "
                    f"would exceed capacity (current: {loaded_page.appended_size}, "
                    f"max: {loaded_page.max_append_capacity})"
                )
                return False

            # TODO (VCM Integration): Implement actual vLLM KV cache append
            # For now, just update tracking
            loaded_page.appended_tokens.extend(tokens)
            loaded_page.appended_size += len(tokens)

            logger.info(
                f"Appended {len(tokens)} tokens to page {page_id} "
                f"(total appended: {loaded_page.appended_size}/{loaded_page.max_append_capacity})"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to append context to page {page_id}: {e}", exc_info=True)
            return False

    @serving.endpoint
    async def remove_appended_context(
        self,
        page_id: ContextPageId,
    ) -> bool:
        """Remove appended context from a loaded page, resetting to base page.

        This removes all temporarily appended tokens, leaving only the base
        (immutable) VirtualContextPage in the KV cache.

        Args:
            page_id: ID of the page to reset

        Returns:
            True if successful, False otherwise

        Example:
            ```python
            # After inference, prepare page for next agent
            await vllm.remove_appended_context("file.py")

            # Append different agent instructions
            new_task_tokens = tokenize("You are a refactoring expert...")
            await vllm.append_context_to_page("file.py", new_task_tokens)
            ```

        TODO (VCM Integration):
            - Implement vLLM KV cache truncation to original page size
            - Verify KV cache state after removal
            - Add safety checks (ensure base page integrity)
            - Consider lazy removal (mark for removal, cleanup on next access)
            - Integrate with agent lifecycle management
            - Add metrics for removal latency
            - Handle removal failures gracefully

        Design Notes:
            This is called after each inference request to prepare the page
            for reuse by different agents. Must be fast and reliable.
        """
        try:
            # Check if page is loaded
            if page_id not in self.loaded_pages:
                logger.warning(f"Cannot remove appended context: page {page_id} not loaded")
                return True  # Not an error if already removed

            loaded_page = self.loaded_pages[page_id]

            # Check if there's appended context
            if not loaded_page.has_appended_context():
                logger.debug(f"Page {page_id} has no appended context to remove")
                return True

            # TODO (VCM Integration): Implement actual vLLM KV cache truncation
            # For now, just update tracking
            removed_size = loaded_page.appended_size
            loaded_page.appended_tokens.clear()
            loaded_page.appended_size = 0

            logger.info(
                f"Removed {removed_size} appended tokens from page {page_id}, "
                f"reset to base page ({len(loaded_page.page.tokens)} tokens)"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to remove appended context from page {page_id}: {e}", exc_info=True)
            return False

    def _get_or_create_page_semaphore(
        self,
        page_id: ContextPageId,
        max_concurrent: int,
    ) -> asyncio.Semaphore:
        """Get or create a semaphore for per-page concurrency control.

        Args:
            page_id: Page ID
            max_concurrent: Maximum concurrent requests for this page

        Returns:
            Semaphore for this page
        """
        if page_id not in self._page_semaphores:
            self._page_semaphores[page_id] = asyncio.Semaphore(max_concurrent)
        return self._page_semaphores[page_id]

    def _check_kv_cache_capacity(self, base_size: int, suffix_size: int) -> bool:
        """Check if we have sufficient KV cache capacity for a request.

        This is a simple heuristic based on current usage. In production,
        we'd query vLLM's actual KV cache state.

        Args:
            base_size: Size of base page in tokens
            suffix_size: Size of suffix in tokens

        Returns:
            True if request can be admitted, False otherwise
        """
        # Assume base might already be cached (optimistic)
        estimated_needed = suffix_size

        # Check if base page is loaded
        # If not loaded, we'll need space for it too
        # (This is a simplification - vLLM may have cached the prefix already)
        estimated_needed = base_size + suffix_size

        # Get current KV cache usage from metrics
        current_used = self.kv_metrics.kv_cache_used_tokens
        available = self.kv_cache_capacity - current_used

        # Keep 10% buffer for safety
        safe_available = available * 0.9

        return estimated_needed <= safe_available

    @serving.endpoint
    @inference_circuit
    async def infer_with_context_composition(
        self,
        base_page_id: ContextPageId,
        suffix_tokens: list[int] | None = None,
        request: InferenceRequest | None = None,
        max_concurrent_per_page: int | None = None,
    ) -> InferenceResponse:
        """Inference with dynamic context composition and concurrency control.

        This endpoint leverages vLLM's Automatic Prefix Caching (APC) to enable
        highly concurrent inference with multiple agents using the same base page
        but different task-specific suffixes.

        Key Features:
        - **Automatic KV Sharing**: vLLM shares base page KV blocks across requests
        - **Per-Page Concurrency Limits**: Prevents resource monopolization
        - **Memory-Based Admission Control**: Rejects requests if capacity exceeded
        - **Zero Coordination Overhead**: vLLM handles all synchronization internally

        Concurrency Model:
        - Multiple concurrent calls with same base_page_id are SAFE
        - vLLM automatically shares base page KV blocks (hash-based lookup)
        - Each request gets its own suffix KV blocks (allocated on demand)
        - No application-level locks needed - vLLM handles everything

        Memory Overhead: O(num_concurrent × suffix_size)
        NOT O(num_concurrent × (base_size + suffix_size))

        Args:
            base_page_id: ID of the loaded base page
            suffix_tokens: Optional task-specific tokens to append (e.g., agent instructions)
            request: Optional InferenceRequest (if None, creates default)
            max_concurrent_per_page: Override default per-page concurrency limit

        Returns:
            InferenceResponse with generated text and metadata

        Raises:
            ValueError: If base page not loaded
            ResourceExhausted: If insufficient KV cache capacity

        Example:
            ```python
            # Load base page once
            base_page = VirtualContextPage(page_id="file.py", tokens=[...])
            await vllm.load_page(base_page)

            # 10 concurrent agents - all safe!
            async def run_agent(agent_id: int):
                suffix = tokenize(f"Agent {agent_id}: analyze this code...")
                return await vllm.infer_with_context_composition(
                    base_page_id="file.py",
                    suffix_tokens=suffix,
                    request=InferenceRequest(request_id=f"agent-{agent_id}", prompt="")
                )

            results = await asyncio.gather(*[run_agent(i) for i in range(10)])

            # vLLM automatically:
            # - Shares file.py KV blocks across all 10 requests (1× base memory)
            # - Allocates separate suffix blocks per agent (10× suffix memory)
            ```

        Design Notes:
            This is the PRIMARY API for context composition. The append/remove
            endpoints are optional utilities - this endpoint provides request-scoped
            composition that leverages vLLM's built-in prefix caching.

            See SPECS_*.md for full design rationale and concurrency analysis.
        """
        queue_start_time = time.time()
        start_time = queue_start_time  # Will update after acquiring semaphore

        try:
            # Validate base page is loaded
            if base_page_id not in self.loaded_pages:
                raise ValueError(f"Base page {base_page_id} not loaded in {self.client_id}")

            loaded_page = self.loaded_pages[base_page_id]
            base_tokens = loaded_page.page.tokens
            suffix_size = len(suffix_tokens) if suffix_tokens else 0

            # Layer 1: Per-page concurrency limit (prevent monopolization)
            max_concurrent = max_concurrent_per_page or self._max_concurrent_per_page
            semaphore = self._get_or_create_page_semaphore(base_page_id, max_concurrent)

            # Try to acquire semaphore - will block if at limit
            async with semaphore:
                # Update metrics: track concurrent requests
                self.kv_metrics.update_concurrency(base_page_id, delta=+1)

                # Record queue time
                queue_time_ms = (time.time() - queue_start_time) * 1000
                if queue_time_ms > 0:
                    # Update queue metrics (simplified - real impl would use moving average)
                    self.kv_metrics.requests_queued += 1
                    total_queue_time = self.kv_metrics.avg_queue_time_ms * (self.kv_metrics.requests_queued - 1)
                    self.kv_metrics.avg_queue_time_ms = (total_queue_time + queue_time_ms) / self.kv_metrics.requests_queued

                start_time = time.time()  # Reset after queue wait

                # Layer 2: Memory-based admission control
                if not self._check_kv_cache_capacity(len(base_tokens), suffix_size):
                    self.kv_metrics.requests_rejected += 1
                    self.kv_metrics.update_concurrency(base_page_id, delta=-1)

                    raise ValueError(
                        f"Insufficient KV cache capacity for request: "
                        f"base={len(base_tokens)} tokens, suffix={suffix_size} tokens, "
                        f"available≈{self.kv_cache_capacity - self.kv_metrics.kv_cache_used_tokens} tokens"
                    )

                # Layer 3: vLLM's internal queuing (max_num_seqs)
                # vLLM handles this automatically - no action needed

                # Construct full token sequence: base + suffix
                if suffix_tokens:
                    full_tokens = base_tokens + suffix_tokens
                else:
                    full_tokens = base_tokens

                # Decode to text for vLLM
                # TODO: How much overhead does this unnecessary decoding add?
                full_prompt = self.tokenizer.decode(full_tokens)

                # Create or use provided inference request
                if request is None:
                    import uuid
                    request = InferenceRequest(
                        request_id=f"compose-{base_page_id}-{uuid.uuid4().hex[:8]}",
                        prompt=full_prompt,
                    )
                else:
                    # Override prompt with composed prompt
                    request = request.model_copy(update={"prompt": full_prompt})

                # Build sampling params
                sampling_params = SamplingParams(
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                )

                # Submit to vLLM - automatic prefix caching!
                # vLLM will:
                # 1. Hash the base_tokens portion
                # 2. Check if KV blocks for base_tokens already exist (cache lookup)
                # 3. If yes: reuse existing blocks (ZERO memory/compute cost for base!)
                # 4. If no: compute and cache KV blocks for base_tokens
                # 5. Always compute fresh KV blocks for suffix_tokens
                # 6. Generate response
                # 7. Free suffix KV blocks when request completes (auto cleanup)

                final_output = None
                async for output in self.engine.generate(
                    prompt=full_prompt,
                    sampling_params=sampling_params,
                    request_id=request.request_id,
                ):
                    final_output = output

                # Extract response
                generated_text = final_output.outputs[0].text if final_output and final_output.outputs else ""
                tokens_generated = len(final_output.outputs[0].token_ids) if final_output and final_output.outputs else 0

                latency_ms = (time.time() - start_time) * 1000

                # Update metrics
                # Note: We can't easily detect cache hits from vLLM's API
                # In production, we'd integrate with vLLM's metrics endpoint
                cache_hit = base_page_id in self.loaded_pages  # Heuristic: assume hit if page loaded
                self.kv_metrics.record_request(
                    page_id=base_page_id,
                    suffix_size=suffix_size,
                    cache_hit=cache_hit,
                )

                # Update KV cache utilization (rough estimate)
                # In production, query vLLM's actual KV cache state
                estimated_used = self.kv_metrics.kv_cache_used_tokens + suffix_size
                self.kv_metrics.update_kv_cache_usage(estimated_used, self.kv_cache_capacity)

                # Update page access stats
                loaded_page.access_count += 1
                loaded_page.last_access_time = time.time()

                # Decrement concurrency counter
                self.kv_metrics.update_concurrency(base_page_id, delta=-1)

                response = InferenceResponse(
                    request_id=request.request_id,
                    generated_text=generated_text,
                    tokens_generated=tokens_generated,
                    page_faults=[],  # No faults - base page was pre-loaded
                    latency_ms=latency_ms,
                    metadata={
                        "base_page_id": base_page_id,
                        "suffix_size": suffix_size,
                        "queue_time_ms": queue_time_ms,
                        "cache_hit": cache_hit,
                    },
                )

                logger.info(
                    f"Context composition inference completed: "
                    f"request_id={request.request_id}, "
                    f"base={base_page_id}, suffix={suffix_size} tokens, "
                    f"generated={tokens_generated} tokens, "
                    f"latency={latency_ms:.2f}ms, queue={queue_time_ms:.2f}ms"
                )

                return response

        except ValueError as e:
            # Resource exhaustion or invalid input - don't retry
            logger.warning(f"Request rejected: {e}")
            raise

        except Exception as e:
            # Unexpected error
            logger.error(
                f"Inference with context composition failed: "
                f"base={base_page_id}, suffix_size={suffix_size}, error={e}",
                exc_info=True
            )
            # Ensure concurrency counter is decremented
            self.kv_metrics.update_concurrency(base_page_id, delta=-1)
            raise

    @serving.endpoint
    async def get_kv_metrics(self) -> KVCacheMetrics:
        """Get current KV cache metrics.

        Returns:
            KVCacheMetrics with current statistics
        """
        # Update KV cache usage from state manager
        async for dep_state in self.state_manager.read_transaction():
            client_state = dep_state.client_states.get(self.client_id)
            if client_state:
                self.kv_metrics.update_kv_cache_usage(
                    client_state.kv_cache_used,
                    client_state.kv_cache_capacity,
                )

        return self.kv_metrics.model_copy(deep=True)

    async def _emit_page_event(self, event: PageEvent) -> bool:
        """Emit a page lifecycle event to Redis pub/sub.

        Args:
            event: PageEvent to publish

        Returns:
            True if event published successfully, False otherwise
        """
        if not self.redis_om:
            # Redis not configured, skip event emission (degraded mode)
            logger.debug(f"Skipping event emission (no Redis): {event.event_type}")
            return False

        try:
            # Publish event to "vcm_page_events" topic
            success = await self.redis_om.update_state_topic(
                topic="vcm_page_events",
                updates={
                    "event_type": event.event_type,
                    "event_data": event.model_dump(),
                    "timestamp": event.timestamp,
                },
                replace_all=False,  # Don't replace entire topic
                update_type="update",
            )

            if success:
                logger.debug(
                    f"Emitted {event.event_type} for page {event.page_id} "
                    f"on {event.deployment_name}/{event.client_id}"
                )
            else:
                logger.warning(
                    f"Failed to emit {event.event_type} for page {event.page_id}"
                )

            return success
        except Exception as e:
            logger.error(f"Error emitting event {event.event_type}: {e}", exc_info=True)
            return False

    async def cleanup(self):
        """Cleanup resources on shutdown."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        if self.gpu_metrics:
            self.gpu_metrics.cleanup()
