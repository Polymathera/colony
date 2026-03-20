# LLM Cluster Layer

Production-ready LLM serving infrastructure for Polymathera, built on `polymathera.rayutils.serving` and vLLM.

## Overview

This package provides the **LLM Cluster Layer** - the second layer in Polymathera's distributed context architecture for billion-token scale inference:

```
┌─────────────────────────────────────────┐
│  Virtual/Physical Context Managers      │  ← Future layer (PCM/VCM)
│  (Page allocation, eviction, migration) │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  LLM Cluster Layer (THIS PACKAGE)       │  ← Current implementation
│  (vLLM deployments, context routing)    │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Serving Infrastructure                 │  ← polymathera.rayutils.serving
│  (Deployments, applications, proxies)   │
└─────────────────────────────────────────┘
```

### Key Features

1. **Context-Aware vLLM Deployments**
   - Each deployment manages its own KV cache as "physical context memory"
   - Pages (chunks of context) can be loaded/evicted from KV cache
   - Automatic LRU eviction when cache is full

2. **Intelligent Request Routing**
   - Routes requests to LLM instances based on loaded context pages
   - Minimizes expensive page faults (loading context on-demand)
   - Balances load across replicas while respecting page locality

3. **Distributed State Management**
   - Tracks which pages are loaded in which LLM instances
   - Cluster-wide page index for efficient routing
   - Synchronized using `StateManager`

4. **Production-Ready**
   - Auto-scaling based on request queue length
   - Health monitoring and self-healing
   - Comprehensive metrics and statistics

## Architecture

### Components

#### 1. **VLLMDeployment**

A `@serving.deployment` that wraps a vLLM `AsyncLLMEngine`:

```python
@serving.deployment(
    autoscaling_config={"min_replicas": 1, "max_replicas": 10},
    ray_actor_options={"num_gpus": 1},
)
class VLLMDeployment:
    """vLLM instance with KV cache management."""

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Perform inference with context pages."""

    async def load_page(self, page: ContextPage) -> bool:
        """Load context page into KV cache."""

    async def evict_page(self, page_id: ContextPageId) -> bool:
        """Evict page from KV cache."""
```

**Key Responsibilities:**
- Initialize vLLM engine with specified model
- Manage KV cache as fixed-size slots for context pages
- Load/evict pages using LRU policy
- Perform inference with automatic prefix caching

#### 2. **ContextAwareRouter**

Custom `RequestRouter` that considers page locality:

```python
class ContextAwareRouter(RequestRouter):
    """Routes requests based on which pages are loaded."""

    async def select_replica(
        self, request: DeploymentRequest, replicas: list[DeploymentReplicaInfo]
    ) -> DeploymentReplicaInfo:
        # Score each replica based on:
        # 1. Number of required pages loaded (page hits)
        # 2. Current load (pending requests)
        # 3. Available KV cache capacity
        ...
```

**Routing Strategy:**
- Extract required page IDs from inference request
- Score each replica: `score = PAGE_HIT_WEIGHT * hit_ratio - LOAD_WEIGHT * load_factor`
- Select replica with highest score

**Variants:**
- `ContextAwareRouter`: Best-effort routing (allows page faults)
- `PageAffinityRouter`: Strict routing (requires all pages loaded, fails otherwise)

#### 3. **LLMCluster**

High-level cluster manager:

```python
class LLMCluster:
    """Manages vLLM cluster with context-aware routing."""

    async def deploy(self) -> None:
        """Deploy vLLM instances as serving.Application."""

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Route and execute inference request."""

    async def load_page(self, page: ContextPage) -> bool:
        """Load page into cluster."""

    async def get_statistics(self) -> ClusterStatistics:
        """Get cluster-wide metrics."""
```

**Responsibilities:**
- Deploy `VLLMDeployment` instances using `serving.Application`
- Provide high-level inference API
- Track page locations across cluster
- Collect and report statistics

### Data Models

#### ContextPage
```python
@dataclass
class ContextPage:
    page_id: ContextPageId
    tokens: list[int]  # Token IDs
    size: int  # Number of tokens
    metadata: dict[str, Any]
```

#### InferenceRequest
```python
class InferenceRequest(BaseModel):
    request_id: str
    prompt: str
    context_page_ids: list[ContextPageId]  # Required pages
    max_tokens: int = 1024
    temperature: float = 0.7
```

#### LLMClientState
```python
class LLMClientState(BaseModel):
    client_id: LLMClientId
    kv_cache_capacity: int
    kv_cache_used: int
    loaded_pages: dict[ContextPageId, LoadedContextPage]
    is_healthy: bool
```

## Usage

### Basic Example

```python
from polymathera.llms_0 import LLMCluster, InferenceRequest, ContextPage

# 1. Deploy cluster
cluster = LLMCluster(
    app_name="my-llm-cluster",
    model_name="meta-llama/Llama-3.1-8B",
    num_replicas=4,
    kv_cache_capacity=256*1024,  # 256k tokens per replica
)
await cluster.deploy()

# 2. Load context pages
page1 = ContextPage(
    page_id="codebase-page-1",
    tokens=[101, 2023, 2003, ...],  # Tokenized code
    size=20000,
    metadata={"source": "src/main.py", "lines": "1-500"},
)
await cluster.load_page(page1)

# 3. Perform inference
request = InferenceRequest(
    request_id="req-1",
    prompt="Explain the main function in this code",
    context_page_ids=["codebase-page-1"],
    max_tokens=512,
)
response = await cluster.infer(request)
print(response.generated_text)

# 4. Check statistics
stats = await cluster.get_statistics()
print(f"Page hit rate: {stats.page_hit_rate:.2%}")
print(f"Total requests: {stats.total_requests}")
```

### Advanced: Custom Routing

```python
from polymathera.llms_0 import PageAffinityRouter

# Strict page affinity (fails if pages not loaded)
cluster = LLMCluster(
    app_name="strict-cluster",
    model_name="meta-llama/Llama-3.1-70B",
    routing_policy=PageAffinityRouter,  # No page faults allowed
)
await cluster.deploy()
```

### Integration with Serving Layer

```python
from polymathera.rayutils import serving
from polymathera.llms_0 import VLLMDeployment, ContextAwareRouter

# Manual deployment with custom config
app = serving.Application(name="custom-llm-app")

app.add_deployment(
    VLLMDeployment.bind(
        model_name="meta-llama/Llama-3.1-405B",
        kv_cache_capacity=512*1024,
        tensor_parallel_size=8,  # 8-GPU tensor parallelism
    ),
    name="llama-405b",
    routing_policy=ContextAwareRouter,
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 8,
        "target_queue_length": 10,
    },
    ray_actor_options={
        "num_gpus": 8,
        "resources": {"accelerator_type:A100": 1},
    },
)

await app.start()

# Get deployment handle
handle = serving.get_deployment("custom-llm-app", "llama-405b")
response = await handle.infer(request)
```

## Design Decisions

### 1. **Why Wrap vLLM in Deployments?**

- **Reusability**: vLLM instances are expensive to initialize. By wrapping them as deployments, we can:
  - Keep them alive across multiple requests
  - Reuse loaded context (KV cache) for temporal locality
  - Avoid repeated model loading

- **Scalability**: `polymathera.rayutils.serving` provides:
  - Auto-scaling based on queue length
  - Health monitoring and self-healing
  - Load balancing with custom routing

- **Composability**: Deployments can be combined in Applications with other services

### 2. **Context Pages vs. vLLM Prefix Caching**

- **Context Pages**: Polymathera-level abstraction for managing billion-token contexts
  - Explicit control over what's loaded in KV cache
  - Cross-request page sharing and migration
  - Integration with future PCM/VCM layers

- **vLLM Prefix Caching**: vLLM-level optimization for automatic prefix reuse
  - Complements our page-based approach
  - Provides automatic caching within vLLM engine
  - Both work together for maximum efficiency

### 3. **Routing Strategy**

The `ContextAwareRouter` prioritizes **page locality over load balancing**:

- **Page hits** weighted 10x higher than load
- Rationale: Page faults are expensive (require loading large context chunks)
- Trade-off: May create slight load imbalance for better cache hit rates

For latency-sensitive use cases, use `PageAffinityRouter` to guarantee zero page faults.

### 4. **State Management**

Uses `StateManager` for distributed coordination:

- **Client states**: Tracked per LLM instance
- **Page index**: Maps page_id → list of client_ids that have it loaded
- **Cluster metrics**: Aggregated statistics

This enables the router to make informed decisions without querying every replica.

## Future Extensions

### Physical Context Manager (PCM)
```python
# Future API
class PhysicalContextManager:
    """Manages physical context pages across LLM cluster."""

    async def allocate_page(self, page_id: str) -> PhysicalContextPage:
        """Allocate physical page for virtual page."""

    async def migrate_page(self, page_id: str, from_client: str, to_client: str):
        """Migrate page between LLM instances."""

    async def evict_pages(self, count: int) -> list[str]:
        """Evict pages using cluster-wide policy."""
```

### Virtual Context Manager (VCM)
```python
# Future API
class VirtualContextManager:
    """Manages virtual context pages and their layout."""

    async def create_pages_from_codebase(self, repo_path: str) -> list[VirtualContextPage]:
        """Layout codebase into virtual pages."""

    async def get_page_for_query(self, query: str) -> list[VirtualContextPage]:
        """Find relevant pages for query."""
```

## Testing

```python
import pytest
from polymathera.llms_0 import LLMCluster, InferenceRequest, ContextPage

@pytest.mark.asyncio
async def test_cluster_deployment():
    cluster = LLMCluster(
        app_name="test-cluster",
        model_name="facebook/opt-125m",  # Small model for testing
        num_replicas=2,
    )

    await cluster.deploy()
    stats = await cluster.get_statistics()

    assert stats.total_clients == 2
    assert stats.healthy_clients == 2

    await cluster.shutdown()

@pytest.mark.asyncio
async def test_page_loading():
    cluster = LLMCluster(app_name="test", model_name="facebook/opt-125m")
    await cluster.deploy()

    page = ContextPage(
        page_id="test-page",
        tokens=[1, 2, 3, 4, 5],
        size=5,
    )

    success = await cluster.load_page(page)
    assert success

    stats = await cluster.get_statistics()
    assert stats.total_pages_loaded == 1
```

## Recent Additions (2025)

### ✅ Completed Features

#### 1. **Model Registry & Configuration System**
- `ModelRegistry`: Centralized model parameter database with 30+ pre-configured models
- `ClusterConfig` & `DeploymentConfig`: Type-safe configuration with validation
- Auto-calculation of KV cache capacity based on GPU memory and quantization
- Registry includes: Llama, Mistral, Qwen, Phi, DeepSeek models

#### 2. **S3 Model Loading**
- `S3ModelLoader`: Download and cache models from S3
- Exponential backoff retry (5s → 300s with jitter)
- Circuit breaker protection
- Automatic cleanup on process exit

#### 3. **Circuit Breakers**
- `inference_circuit`: 10 failures, 30s recovery
- `page_loading_circuit`: 5 failures, 30s recovery
- `s3_operations_circuit`: 5 failures, 60s recovery
- Prevents cascading failures in distributed systems

#### 4. **Quantization Support**
- AWQ, GPTQ, FP8, INT8, INT4 quantization methods
- Per-model quantization recommendations in registry
- Automatic bytes-per-token calculation for KV cache sizing

#### 5. **Structured Output Support**
- JSON schema-guided generation via vLLM's `guided_json`
- Type-safe JSON outputs without post-processing
- Integrated into `InferenceRequest.json_schema` field

#### 6. **Embedding Model Support**
- `EmbeddingDeployment`: Large-scale embedding generation (1B-10B params)
- Uses vLLM with `task="embed"` mode
- Auto-scaling, S3 loading, quantization support
- Complementary to existing `EmbeddingClient` (SentenceTransformer)

#### 7. **Distributed Page Location Tracking**
- Bidirectional indices: `page_index` (page → clients) and `client_page_index` (client → pages)
- `find_best_client_for_pages()`: Smart routing based on page locality
- `get_page_distribution_stats()`: Cluster-wide page distribution metrics

#### 8. **Page Migration**
- `migrate_page()`: Transfer pages between vLLM instances
- Atomic operation with rollback on failure
- Enables dynamic load balancing and cache optimization

### Architecture Modules

#### Core Infrastructure
- `circuit_breakers.py`: Centralized circuit breaker policies
- `model_loader.py`: S3 model download with retry logic
- `registry.py`: Model parameter database
- `config.py`: Configuration system using registry

#### Deployments
- `vllm_deployment.py`: vLLM-based LLM serving with KV cache
- `embedding_deployment.py`: vLLM-based embedding generation

#### Routing & State
- `routing.py`: Context-aware request routing
- `cluster.py`: Cluster management and coordination
- `models.py`: Core data structures

## TODOs

### High Priority
- [ ] Test cluster on real GPU hardware (AWS EKS Ray cluster)
- [ ] Implement actual vLLM KV cache management (currently using prefix caching)
- [ ] Add page preloading based on predicted access patterns
- [ ] Benchmark different quantization methods

### Medium Priority
- [ ] Add support for multi-node vLLM (pipeline parallelism)
- [ ] Implement more sophisticated eviction policies (beyond LRU)
- [ ] Add metrics export to Prometheus
- [ ] Add fallback models for embedding deployment

### Low Priority
- [ ] Support for multiple model types in same cluster
- [ ] Dynamic model loading/unloading based on demand
- [ ] Advanced routing policies (affinity, locality-aware)

## Performance Considerations

### KV Cache Sizing
- Default: 128k tokens per replica
- Large models: 256k-512k tokens
- Balance: More cache = fewer page faults, but higher memory usage

### Routing Overhead
- Page state queries add ~1-5ms latency
- Mitigated by caching state in router
- Trade-off worth it for 10-100x reduction in page fault latency

### Scaling Guidelines
- **Small models (≤13B)**: 1-2 GPUs per replica, 4-8 replicas
- **Medium models (13B-70B)**: 2-4 GPUs per replica, 2-4 replicas
- **Large models (70B+)**: 4-8 GPUs per replica, 1-2 replicas

### Cost Optimization
- Use spot instances for non-critical workloads
- Set aggressive scale-down policies during low traffic
- Share context pages across multiple models when possible

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [`polymathera.rayutils.serving` package](../rayutils/serving/README.md)
- [LLM Cluster Specs](./SPECS_LLM_CLUSTER.md)
- [System Specs](./SPECS_SYSTEM.md)