"""Core data models for the LLM cluster layer.

This module defines the fundamental data structures used throughout the LLM cluster,
including context pages, client state, and routing information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..distributed.state_management import SharedState
from ..vcm.models import VirtualContextPage, ContextPageId

LLMClientId = str
"""Unique identifier for an LLM client instance."""


class ContextPageState(str, Enum):
    """State of a context page in an LLM client's KV cache."""

    LOADING = "loading"  # Page is being loaded into KV cache
    LOADED = "loaded"  # Page is fully loaded and ready
    EVICTING = "evicting"  # Page is being evicted from KV cache
    EVICTED = "evicted"  # Page has been evicted


@dataclass
class LoadedContextPage:
    """Represents a context page that is currently loaded in an LLM client's KV cache.

    This class supports dynamic context composition by allowing task-specific context
    to be temporarily appended to the base (immutable) VirtualContextPage. The appended
    context can be removed after inference to prepare for the next request.

    Attributes:
        page: The base context page (immutable)
        client_id: ID of the LLM client where this page is loaded
        state: Current state of the page
        kv_cache_slot: Slot index in the KV cache where this page is stored
        load_time: Timestamp when the page was loaded
        last_access_time: Timestamp of last access
        access_count: Number of times this page has been accessed
        appended_tokens: Temporary tokens appended to base page (task-specific context)
        appended_size: Size of appended portion in tokens
        max_append_capacity: Maximum tokens that can be appended (page.size - len(page.tokens))

    Usage:
        ```python
        # Load base page
        loaded_page = LoadedContextPage(page=base_page, ...)

        # Append task-specific context (e.g., agent instructions)
        task_tokens = tokenize("You are a code reviewer. Analyze this file...")
        loaded_page.append_context(task_tokens)

        # Use page for inference
        await vllm.infer(request)

        # Remove appended context after inference
        loaded_page.remove_appended_context()
        ```

    TODO (VCM Integration):
        - Implement vLLM KV cache manipulation for append/remove operations
        - Add automatic lifecycle management via agent system
        - Add validation for append capacity limits
        - Consider compression/summarization for large appended contexts
    """

    page: VirtualContextPage
    client_id: LLMClientId
    state: ContextPageState
    kv_cache_slot: int
    load_time: float
    last_access_time: float
    access_count: int = 0

    # Dynamic context composition
    # These fields support appending task-specific context to the base page
    appended_tokens: list[int] = field(default_factory=list)
    """Temporary tokens appended to base page (e.g., agent instructions, tool descriptions)"""

    appended_size: int = 0
    """Size of appended portion in tokens"""

    max_append_capacity: int = field(default=0)
    """Maximum tokens that can be appended without exceeding page.size"""

    def __post_init__(self):
        """Initialize max_append_capacity based on page capacity and actual tokens."""
        # Calculate how much space is reserved for appending
        # page.size is the total capacity, len(page.tokens) is the base content
        self.max_append_capacity = self.page.size - len(self.page.tokens)

    def can_append(self, num_tokens: int) -> bool:
        """Check if num_tokens can be appended without exceeding capacity.

        Args:
            num_tokens: Number of tokens to append

        Returns:
            True if there's sufficient capacity
        """
        return self.appended_size + num_tokens <= self.max_append_capacity

    def get_total_size(self) -> int:
        """Get total size including base page and appended context.

        Returns:
            Total tokens (base + appended)
        """
        return len(self.page.tokens) + self.appended_size

    def has_appended_context(self) -> bool:
        """Check if this page has appended context.

        Returns:
            True if context has been appended
        """
        return self.appended_size > 0


class LLMClientState(BaseModel):
    """State of an LLM client deployment.

    Tracks the current state of an LLM client including loaded pages,
    resource usage, and health metrics.
    """

    client_id: LLMClientId
    """Unique identifier for this client"""

    deployment_name: str
    """Name of the deployment (for service discovery)"""

    app_name: str
    """Name of the application this client belongs to"""

    model_name: str
    """Name/path of the LLM model"""

    kv_cache_capacity: int
    """Maximum number of tokens the KV cache can hold"""

    kv_cache_used: int = 0
    """Number of tokens currently in KV cache"""

    loaded_page_ids: set[ContextPageId] = Field(default_factory=set)
    """Set of page IDs loaded in this client's KV cache"""

    pending_requests: int = 0
    """Number of pending inference requests"""

    total_requests: int = 0
    """Total number of requests processed"""

    error_count: int = 0
    """Number of errors encountered"""

    is_healthy: bool = True
    """Whether this client is healthy"""

    last_heartbeat: float | None = None
    """Timestamp of last heartbeat"""

    last_error: str | None = None
    """Last error message if any"""

    def get_available_cache_capacity(self) -> int:
        """Get available KV cache capacity in tokens."""
        return self.kv_cache_capacity - self.kv_cache_used

    def has_page(self, page_id: ContextPageId) -> bool:
        """Check if a page is loaded in this client."""
        return page_id in self.loaded_page_ids


class InferenceRequest(BaseModel):
    """Request for LLM inference.

    Attributes:
        request_id: Unique identifier for this request
        prompt: The prompt text
        context_page_ids: IDs of context pages required for this request
        requirements: Optional requirements for deployment selection
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        json_schema: Optional JSON schema for structured output (vLLM guided decoding)
        metadata: Additional request metadata
    """

    request_id: str
    prompt: str
    context_page_ids: list[ContextPageId] = Field(default_factory=list)
    requirements: LLMClientRequirements | None = Field(
        default=None,
        description="Requirements for deployment selection and multi-tenancy"
    )
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    json_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON schema for structured output generation (vLLM guided decoding)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_tenant_id(self) -> str:
        """Get tenant ID from requirements or return default.

        Returns:
            Tenant ID for this request
        """
        if self.requirements:
            return self.requirements.tenant_id
        return "default"


class InferenceResponse(BaseModel):
    """Response from LLM inference.

    Attributes:
        request_id: ID of the corresponding request
        generated_text: Generated text
        tokens_generated: Number of tokens generated
        page_faults: IDs of pages that weren't loaded (page faults)
        latency_ms: Inference latency in milliseconds
        metadata: Additional response metadata
    """

    request_id: str
    generated_text: str
    tokens_generated: int
    page_faults: list[ContextPageId] = Field(default_factory=list)
    latency_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClusterStatistics(BaseModel):
    """Statistics for the entire LLM cluster.

    Attributes:
        total_clients: Total number of LLM clients
        healthy_clients: Number of healthy clients
        total_kv_cache_capacity: Total KV cache capacity across all clients (in tokens)
        total_kv_cache_used: Total KV cache used across all clients (in tokens)
        total_pages_loaded: Total number of unique pages loaded across all clients
        total_requests: Total requests processed
        total_errors: Total errors encountered
        average_latency_ms: Average inference latency
        page_hit_rate: Percentage of requests that had all required pages loaded
    """

    total_clients: int
    healthy_clients: int
    total_kv_cache_capacity: int
    total_kv_cache_used: int
    total_pages_loaded: int
    total_requests: int
    total_errors: int
    average_latency_ms: float
    page_hit_rate: float


class VLLMDeploymentState(SharedState):
    """State for a single vLLM deployment across all its replicas.

    Tracks pages and clients within one deployment to avoid transaction
    contention between different deployments.
    Tracks state across all replicas within one deployment, including:
    - Client health and capacity per replica
    - Page locations (which pages are loaded on which replicas)
    - Deployment-wide metrics

    Each deployment has its own StateManager to avoid transaction contention.

    ------------------------------------------------------------------------
    Important Note:
    ------------------------------------------------------------------------
    We maintain the states of all replicas within a
    single deployment in one SharedState instance. This is because
    the request router needs to make decisions based on the
    combined state of all replicas in a deployment ATOMICALLY. If we split
    the state across multiple instances, it could lead to inconsistent
    routing decisions and degraded performance.
    ------------------------------------------------------------------------
    """

    # Client states indexed by client ID
    client_states: dict[LLMClientId, LLMClientState] = Field(default_factory=dict)

    # Page location index: maps page_id to list of client_ids that have it loaded
    page_index: dict[ContextPageId, list[LLMClientId]] = Field(default_factory=dict)

    # Reverse index: maps client_id to set of page_ids loaded on that client
    client_page_index: dict[LLMClientId, set[ContextPageId]] = Field(default_factory=dict)

    # Deployment-wide metrics
    total_requests: int = 0
    total_errors: int = 0
    total_page_faults: int = 0

    # Tenant tracking
    tenant_pages: dict[str, set[ContextPageId]] = Field(default_factory=dict)

    @staticmethod
    def get_state_key(app_name: str, deployment_name: str) -> str:
        """Get the state key for this deployment."""
        return f"polymathera:serving:{app_name}:llm_cluster:{deployment_name}.state"

    def register_page_load(
        self,
        page_id: ContextPageId,
        client_id: LLMClientId,
        tenant_id: str = "default",
    ) -> None:
        """Register that a page has been loaded on a client."""
        if page_id not in self.page_index:
            self.page_index[page_id] = []
        if client_id not in self.page_index[page_id]:
            self.page_index[page_id].append(client_id)

        if client_id not in self.client_page_index:
            self.client_page_index[client_id] = set()
        self.client_page_index[client_id].add(page_id)

        if tenant_id not in self.tenant_pages:
            self.tenant_pages[tenant_id] = set()
        self.tenant_pages[tenant_id].add(page_id)

    def register_page_eviction(
        self,
        page_id: ContextPageId,
        client_id: LLMClientId,
    ) -> None:
        """Register that a page has been evicted from a client."""
        if page_id in self.page_index:
            if client_id in self.page_index[page_id]:
                self.page_index[page_id].remove(client_id)
            if not self.page_index[page_id]:
                del self.page_index[page_id]

        if client_id in self.client_page_index:
            self.client_page_index[client_id].discard(page_id)
            if not self.client_page_index[client_id]:
                del self.client_page_index[client_id]

    def find_clients_with_page(self, page_id: ContextPageId) -> list[LLMClientId]:
        """Find all clients that have a specific page loaded."""
        return self.page_index.get(page_id, []).copy()




class LLMClusterState(SharedState):
    """Distributed state for cluster-wide metrics across all LLM deployments.

    This class tracks only aggregate, cluster-level metrics that are written
    by the LLMCluster. Per-deployment state (client health, page locations, etc.)
    is tracked in VLLMDeploymentState instances.

    Why separate from VLLMDeploymentState?
    - Cluster metrics (total_requests, total_errors) aggregate across deployments
    - Per-deployment metrics are in VLLMDeploymentState to avoid transaction contention
    - Cross-deployment queries iterate through per-deployment states

    All state mutations are atomic and thread-safe through the StateManager.
    """

    # Cluster-wide metrics (aggregated across all deployments)
    total_requests: int = 0
    total_errors: int = 0
    total_page_faults: int = 0

    @staticmethod
    def get_state_key(app_name: str) -> str:
        """Get the state key for this cluster."""
        return f"polymathera:serving:{app_name}:llm_cluster.state"



class KVCacheMetrics(BaseModel):
    """Metrics for KV cache efficiency and utilization.

    Tracks KV cache performance including prefix caching effectiveness,
    memory utilization, and concurrency patterns. These metrics are critical
    for understanding vLLM's automatic prefix caching behavior and optimizing
    context composition strategies.

    Attributes:
        total_requests: Total inference requests processed
        cache_hit_count: Requests that benefited from prefix cache hits
        cache_miss_count: Requests that required full KV computation
        cache_hit_rate: Percentage of requests with cache hits
        total_kv_blocks_allocated: Total KV blocks allocated (lifetime)
        current_kv_blocks_used: Currently used KV blocks
        kv_cache_capacity_tokens: Total KV cache capacity in tokens
        kv_cache_used_tokens: Currently used KV cache in tokens
        kv_cache_utilization: Current utilization percentage (0-100)
        concurrent_requests_per_page: Active requests per page
        avg_suffix_size_tokens: Average suffix size in tokens
        requests_queued: Requests waiting for concurrency slot
        requests_rejected: Requests rejected due to capacity
        avg_queue_time_ms: Average time spent waiting in queue
    """

    # Cache effectiveness
    total_requests: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    cache_hit_rate: float = 0.0

    # KV cache utilization
    total_kv_blocks_allocated: int = 0
    current_kv_blocks_used: int = 0
    kv_cache_capacity_tokens: int = 0
    kv_cache_used_tokens: int = 0
    kv_cache_utilization: float = 0.0

    # Concurrency metrics
    concurrent_requests_per_page: dict[ContextPageId, int] = Field(default_factory=dict)
    avg_suffix_size_tokens: float = 0.0
    requests_queued: int = 0
    requests_rejected: int = 0
    avg_queue_time_ms: float = 0.0

    # Per-page statistics
    page_request_counts: dict[ContextPageId, int] = Field(default_factory=dict)
    page_suffix_sizes: dict[ContextPageId, list[int]] = Field(default_factory=dict)

    def record_request(
        self,
        page_id: ContextPageId | None,
        suffix_size: int,
        cache_hit: bool,
    ) -> None:
        """Record a request for metrics tracking.

        Args:
            page_id: Base page ID used (None if no composition)
            suffix_size: Size of appended suffix in tokens
            cache_hit: Whether prefix cache was hit
        """
        self.total_requests += 1

        if cache_hit:
            self.cache_hit_count += 1
        else:
            self.cache_miss_count += 1

        if self.total_requests > 0:
            self.cache_hit_rate = self.cache_hit_count / self.total_requests

        if page_id:
            self.page_request_counts[page_id] = self.page_request_counts.get(page_id, 0) + 1

            if page_id not in self.page_suffix_sizes:
                self.page_suffix_sizes[page_id] = []
            self.page_suffix_sizes[page_id].append(suffix_size)

        # Update average suffix size
        all_sizes = [s for sizes in self.page_suffix_sizes.values() for s in sizes]
        if all_sizes:
            self.avg_suffix_size_tokens = sum(all_sizes) / len(all_sizes)

    def update_concurrency(self, page_id: ContextPageId, delta: int) -> None:
        """Update concurrent request count for a page.

        Args:
            page_id: Page ID
            delta: Change in concurrent requests (+1 for new, -1 for completed)
        """
        current = self.concurrent_requests_per_page.get(page_id, 0)
        new_count = max(0, current + delta)

        if new_count == 0:
            self.concurrent_requests_per_page.pop(page_id, None)
        else:
            self.concurrent_requests_per_page[page_id] = new_count

    def update_kv_cache_usage(self, used_tokens: int, capacity_tokens: int) -> None:
        """Update KV cache utilization metrics.

        Args:
            used_tokens: Currently used KV cache in tokens
            capacity_tokens: Total KV cache capacity in tokens
        """
        self.kv_cache_used_tokens = used_tokens
        self.kv_cache_capacity_tokens = capacity_tokens

        if capacity_tokens > 0:
            self.kv_cache_utilization = (used_tokens / capacity_tokens) * 100


class LLMClientRequirements(BaseModel):
    """Requirements for selecting an LLM deployment for inference.

    Inspired by LiteLLM's routing configuration, this model specifies
    constraints and preferences for deployment selection at the request level.

    The router uses these requirements to select an appropriate deployment
    from the available pool, with support for fallbacks and multi-tenant isolation.

    Example:
        ```python
        requirements = LLMClientRequirements(
            min_context_window=32000,
            model_family="llama",
            tenant_id="customer-123",
            isolation_level="shared",
            requires_structured_output=True,
        )

        request = InferenceRequest(
            request_id="req-1",
            prompt="Generate JSON...",
            requirements=requirements,
        )
        ```
    """

    # Model constraints
    min_context_window: int | None = Field(
        default=None,
        description="Minimum context window size in tokens"
    )
    max_context_window: int | None = Field(
        default=None,
        description="Maximum context window size in tokens"
    )
    model_family: str | None = Field(
        default=None,
        description="Model family (e.g., 'llama', 'mistral', 'qwen')"
    )
    min_model_size: str | None = Field(
        default=None,
        description="Minimum model size (e.g., '7B', '13B', '70B')"
    )

    # Quantization preferences
    preferred_quantization: list[str] | None = Field(
        default=None,
        description="Preferred quantization methods in order of preference (e.g., ['awq', 'gptq', 'fp8'])"
    )

    # Performance constraints
    max_latency_ms: float | None = Field(
        default=None,
        description="Maximum acceptable latency in milliseconds"
    )
    min_throughput_rps: float | None = Field(
        default=None,
        description="Minimum required throughput in requests per second"
    )

    # Capability requirements
    requires_structured_output: bool = Field(
        default=False,
        description="Whether structured output (JSON schema) support is required"
    )
    requires_function_calling: bool = Field(
        default=False,
        description="Whether function calling capability is required"
    )

    # Multi-LoRA support
    lora_adapter_id: str | None = Field(
        default=None,
        description="Specific LoRA adapter to use (for multi-LoRA serving)"
    )

    # Deployment preferences (for multi-model routing)
    preferred_deployment_ids: list[str] | None = Field(
        default=None,
        description="Explicitly preferred deployment IDs in order"
    )
    fallback_deployment_ids: list[str] | None = Field(
        default=None,
        description="Fallback deployment IDs if preferred unavailable"
    )

    # Multi-tenancy and isolation
    tenant_id: str = Field(
        default="default",
        description="Tenant ID for multi-tenancy isolation"
    )
    run_id: str | None = Field(
        default=None,
        description="AgentRun ID for finer-grained tracking within tenant"
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for session-level tracking and isolation"
    )
    isolation_level: Literal["shared", "isolated"] = Field(
        default="shared",
        description="'shared' (tenant-isolated KV cache) or 'isolated' (dedicated instance)"
    )

    def matches_deployment(
        self,
        deployment_name: str,
        model_name: str,
        context_window: int,
        quantization: str | None,
        capabilities: set[str],
    ) -> bool:
        """Check if a deployment matches these requirements.

        Args:
            deployment_name: Name of the deployment
            model_name: Model name/path
            context_window: Context window size
            quantization: Quantization method
            capabilities: Set of capability strings

        Returns:
            True if deployment matches, False otherwise
        """
        # Check explicit deployment preference
        if self.preferred_deployment_ids:
            if deployment_name not in self.preferred_deployment_ids:
                return False

        # Check context window
        if self.min_context_window and context_window < self.min_context_window:
            return False
        if self.max_context_window and context_window > self.max_context_window:
            return False

        # Check model family
        if self.model_family:
            model_lower = model_name.lower()
            if self.model_family.lower() not in model_lower:
                return False

        # Check quantization preference
        if self.preferred_quantization and quantization:
            if quantization not in self.preferred_quantization:
                return False

        # Check capabilities
        if self.requires_structured_output and "structured_output" not in capabilities:
            return False
        if self.requires_function_calling and "function_calling" not in capabilities:
            return False

        return True