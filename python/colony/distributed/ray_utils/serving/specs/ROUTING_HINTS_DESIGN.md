# Routing Hints Architecture Design

## Problem Statement

The current routing system has a fundamental flaw: routers must **inspect DeploymentRequest arguments** to determine routing strategy. This is fragile because:

1. **Tight coupling**: Routers must know the internal structure of method arguments
2. **No declarative configuration**: Can't specify routing policy per method
3. **Fragile pattern matching**: `_extract_inference_request()` uses isinstance checks on positional args
4. **No type safety**: Easy to break when refactoring method signatures
5. **Limited extensibility**: Hard to add new routing strategies

### Current Flow (Fragile)

```
Caller → DeploymentHandle.method(args)
    → wrap in DeploymentRequest(method_name, args, kwargs)
    → DeploymentProxyRayActor.handle_request(request)
    → router.route_request(request, replicas)
    → router._extract_inference_request(request)  # ⚠️ FRAGILE!
    → peek into request.args[0], check isinstance(InferenceRequest)
    → extract context_page_ids from args
    → route based on pages
```

## Solution: First-Class Routing Hints

Make routing information **first-class** and **declarative** by:

1. **Routing metadata** attached directly to `DeploymentRequest`
2. **Declarative policies** via `@endpoint(routing_policy=...)`
3. **Automatic extraction** from typed method arguments
4. **Type-safe dispatch** using routing hints

### Proposed Flow (Robust)

```
Caller → DeploymentHandle.method(args)
    → extract routing hints from args using endpoint config
    → wrap in DeploymentRequest(method_name, args, kwargs, routing_hints)
    → DeploymentProxyRayActor.handle_request(request)
    → select router based on request.routing_hints.policy
    → router.route_request(request, replicas)
    → router uses request.routing_hints.context_page_ids  # ✓ EXPLICIT!
    → route based on hints
```

## Design Components

### 1. RoutingPolicy Enum

Define standard routing policies as an enum:

```python
class RoutingPolicy(str, Enum):
    """Standard routing policies for deployments."""

    ROUND_ROBIN = "round_robin"
    """Simple round-robin across replicas."""

    LEAST_LOADED = "least_loaded"
    """Route to replica with lowest load (default)."""

    CONTEXT_AWARE = "context_aware"
    """Route based on context page locality (LLM-specific)."""

    PAGE_AFFINITY = "page_affinity"
    """Strict affinity - only route to replicas with ALL required pages."""

    CUSTOM = "custom"
    """Custom router implementation."""
```

### 2. RoutingHints Dataclass

First-class routing metadata:

```python
@dataclass
class RoutingHints:
    """Routing metadata for request routing.

    This class carries routing information extracted from method arguments,
    allowing routers to make intelligent routing decisions without inspecting
    the original arguments.
    """

    policy: RoutingPolicy | None = None
    """Routing policy to use for this request."""

    context_page_ids: list[str] | None = None
    """Required context page IDs (for context-aware routing)."""

    tenant_id: str | None = None
    """Tenant ID for multi-tenancy isolation."""

    requirements: 'LLMClientRequirements | None' = None
    """LLM client requirements (for requirement-based routing)."""

    affinity_key: str | None = None
    """Generic affinity key for custom routing strategies."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional routing metadata."""

    @staticmethod
    def from_inference_request(
        inference_req: 'InferenceRequest',
        policy: RoutingPolicy | None = None
    ) -> 'RoutingHints':
        """Create routing hints from InferenceRequest."""
        return RoutingHints(
            policy=policy,
            context_page_ids=list(inference_req.context_page_ids),
            tenant_id=inference_req.requirements.tenant_id if inference_req.requirements else None,
            requirements=inference_req.requirements,
        )
```

### 3. Enhanced DeploymentRequest

Add routing_hints field:

```python
class DeploymentRequest(BaseModel):
    """Request to a deployment endpoint."""

    request_id: str
    method_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    metadata: dict[str, Any]

    routing_hints: RoutingHints | None = None  # NEW
    """Routing hints extracted from method arguments."""
```

### 4. Enhanced @endpoint Decorator

Accept routing_policy parameter:

```python
@serving.endpoint(routing_policy=RoutingPolicy.CONTEXT_AWARE)
async def infer(self, request: InferenceRequest) -> InferenceResponse:
    """Inference endpoint with context-aware routing."""
    ...

@serving.endpoint(routing_policy=RoutingPolicy.PAGE_AFFINITY)
async def infer_with_strict_affinity(self, request: InferenceRequest) -> InferenceResponse:
    """Inference with strict page affinity."""
    ...

@serving.endpoint()  # defaults to LEAST_LOADED
async def health_check(self) -> str:
    """Simple health check with default routing."""
    ...
```

### 5. Endpoint Configuration Storage

Store routing policy per method in `DeploymentConfig`:

```python
@dataclass
class EndpointMetadata:
    """Metadata for a deployment endpoint."""
    name: str
    routing_policy: RoutingPolicy
    # Future: rate limiting, timeout, priority, etc.

class DeploymentConfig:
    """Configuration for a deployment."""

    def __init__(self, ...):
        ...
        self.endpoint_metadata: dict[str, EndpointMetadata] = {}
        """Map of method_name -> EndpointMetadata."""

    def register_endpoint(
        self,
        method_name: str,
        routing_policy: RoutingPolicy | None = None
    ) -> None:
        """Register endpoint with routing policy."""
        self.endpoint_metadata[method_name] = EndpointMetadata(
            name=method_name,
            routing_policy=routing_policy or RoutingPolicy.LEAST_LOADED,
        )

    def get_endpoint_metadata(self, method_name: str) -> EndpointMetadata | None:
        """Get endpoint metadata by method name."""
        return self.endpoint_metadata.get(method_name)
```

### 6. RoutingHintExtractor

Utility to extract routing hints from method args:

```python
class RoutingHintExtractor:
    """Extracts routing hints from method arguments."""

    @staticmethod
    def extract(
        method_name: str,
        args: tuple,
        kwargs: dict,
        endpoint_metadata: EndpointMetadata | None
    ) -> RoutingHints:
        """Extract routing hints from method arguments.

        Args:
            method_name: Name of the method being called
            args: Positional arguments
            kwargs: Keyword arguments
            endpoint_metadata: Endpoint configuration (routing policy, etc.)

        Returns:
            RoutingHints with extracted information
        """
        # Get routing policy from endpoint config
        policy = endpoint_metadata.routing_policy if endpoint_metadata else RoutingPolicy.LEAST_LOADED

        # Look for InferenceRequest in args (position 0 is common pattern)
        inference_req = RoutingHintExtractor._find_inference_request(args, kwargs)

        if inference_req:
            return RoutingHints.from_inference_request(inference_req, policy=policy)

        # No special routing hints, return policy only
        return RoutingHints(policy=policy)

    @staticmethod
    def _find_inference_request(args: tuple, kwargs: dict) -> 'InferenceRequest | None':
        """Find InferenceRequest in args or kwargs."""
        from polymathera.llms_0.cluster.models import InferenceRequest

        # Check first positional arg
        if args and isinstance(args[0], InferenceRequest):
            return args[0]

        # Check common kwarg names
        for key in ['request', 'inference_request', 'req']:
            if key in kwargs and isinstance(kwargs[key], InferenceRequest):
                return kwargs[key]

        return None
```

### 7. Updated DeploymentHandle

Extract and attach routing hints when creating request:

```python
class DeploymentHandle:
    """Handle for remotely calling deployment methods."""

    def __init__(self, ..., deployment_config: DeploymentConfig):
        ...
        self.deployment_config = deployment_config

    async def __call_method__(self, method_name: str, *args, **kwargs) -> Any:
        """Call a deployment method with routing hints extraction."""

        # Extract routing hints using endpoint configuration
        endpoint_metadata = self.deployment_config.get_endpoint_metadata(method_name)
        routing_hints = RoutingHintExtractor.extract(
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            endpoint_metadata=endpoint_metadata
        )

        # Create request with routing hints
        request = DeploymentRequest(
            request_id=str(uuid.uuid4()),
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            routing_hints=routing_hints,  # ✓ Attached!
        )

        # Send to proxy
        response = await self.proxy_handle.handle_request(request)
        ...
```

### 8. Router Selection in Proxy

Use routing hints to select router:

```python
class DeploymentProxyRayActor:
    """Proxy actor with router selection based on routing hints."""

    def __init__(self, ..., routing_policy: Type[RequestRouter] | None = None):
        """Initialize with default and specialized routers."""
        # Default router
        self.default_router = routing_policy() if routing_policy else LeastLoadedRouter()

        # Registry of specialized routers
        self.specialized_routers: dict[RoutingPolicy, RequestRouter] = {
            RoutingPolicy.ROUND_ROBIN: RoundRobinRouter(),
            RoutingPolicy.LEAST_LOADED: LeastLoadedRouter(),
            # Context-aware routers initialized lazily
        }

    def _get_router(self, routing_hints: RoutingHints | None) -> RequestRouter:
        """Select router based on routing hints."""
        if not routing_hints or not routing_hints.policy:
            return self.default_router

        policy = routing_hints.policy

        # Use specialized router if available
        if policy in self.specialized_routers:
            return self.specialized_routers[policy]

        # For context-aware/page-affinity, need lazy initialization
        if policy == RoutingPolicy.CONTEXT_AWARE:
            if policy not in self.specialized_routers:
                from polymathera.llms_0.cluster.routing import ContextAwareRouter
                self.specialized_routers[policy] = ContextAwareRouter()
            return self.specialized_routers[policy]

        if policy == RoutingPolicy.PAGE_AFFINITY:
            if policy not in self.specialized_routers:
                from polymathera.llms_0.cluster.routing import PageAffinityRouter
                self.specialized_routers[policy] = PageAffinityRouter()
            return self.specialized_routers[policy]

        # Fallback to default
        return self.default_router

    async def handle_request(self, request: DeploymentRequest) -> DeploymentResponse:
        """Handle request with router selection."""
        ...
        # Select router based on routing hints
        router = self._get_router(request.routing_hints)

        # Route request
        replica = await router.route_request(request, healthy_replicas)
        ...
```

### 9. Updated Routers

Routers now use routing hints instead of extracting from args:

```python
class ContextAwareRouter(RequestRouter):
    """Context-aware router using routing hints."""

    async def route_request(
        self,
        request: DeploymentRequest,
        replicas: list[DeploymentReplicaInfo],
    ) -> DeploymentReplicaInfo:
        """Route based on context page locality using routing hints."""

        # Extract routing hints (no more fragile _extract_inference_request!)
        if not request.routing_hints or not request.routing_hints.context_page_ids:
            # No context pages, fallback to least-loaded
            return min(replicas, key=lambda r: r.queue_length + r.in_flight_requests)

        required_pages = set(request.routing_hints.context_page_ids)

        # Read client states and score replicas
        ...
```

## Benefits

### 1. Type Safety
- Routing hints are explicitly typed
- No more `isinstance()` checks on arbitrary args
- IDEs can provide autocomplete and type checking

### 2. Declarative Configuration
- `@endpoint(routing_policy=...)` is self-documenting
- Easy to see which methods use which routing strategies
- No hidden magic in router implementations

### 3. Extensibility
- Easy to add new routing policies (just add to enum)
- Custom routers can define their own hint fields
- No need to modify existing routers when adding new methods

### 4. Separation of Concerns
- **Decorator**: Declares routing policy
- **Extractor**: Extracts hints from args (knows about arg types)
- **Proxy**: Selects router (orchestrates)
- **Router**: Routes using hints (no arg inspection)

### 5. Testability
- Can test routers with synthetic RoutingHints
- No need to construct full InferenceRequest objects
- Easy to mock routing behavior

### 6. Performance
- Extraction happens once at handle level
- Routers don't re-parse args
- Can cache extracted hints if needed

## Migration Path

### Phase 1: Add Infrastructure (Non-Breaking)
1. Add `RoutingPolicy`, `RoutingHints` to `serving/models.py`
2. Add `routing_hints` field to `DeploymentRequest` (optional)
3. Add `routing_policy` parameter to `@endpoint` (optional)
4. Add `RoutingHintExtractor` utility

### Phase 2: Update Handle and Proxy
5. Update `DeploymentHandle` to extract routing hints
6. Update `DeploymentProxyRayActor` to use hints for router selection

### Phase 3: Update Routers (Breaking for Custom Routers)
7. Update `ContextAwareRouter` to use `routing_hints`
8. Update `PageAffinityRouter` to use `routing_hints`
9. Deprecate `_extract_inference_request()` methods

### Phase 4: Update Endpoints
10. Add `routing_policy` to `@endpoint` decorators in LLM cluster
11. Remove fallback `_extract_inference_request()` code

## Usage Examples

### Example 1: Basic Endpoint with Context-Aware Routing

```python
@serving.deployment(name="VLLMDeployment")
class VLLMDeployment:

    @serving.endpoint(routing_policy=RoutingPolicy.CONTEXT_AWARE)
    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Inference with automatic context-aware routing."""
        # Routing hints automatically extracted:
        # - policy=CONTEXT_AWARE
        # - context_page_ids=request.context_page_ids
        # - tenant_id=request.requirements.tenant_id
        # - requirements=request.requirements
        ...

    @serving.endpoint(routing_policy=RoutingPolicy.PAGE_AFFINITY)
    async def infer_strict(self, request: InferenceRequest) -> InferenceResponse:
        """Inference with strict page affinity (all pages must be loaded)."""
        ...

    @serving.endpoint()  # defaults to LEAST_LOADED
    async def health_check(self) -> str:
        """Simple health check with default routing."""
        return "healthy"
```

### Example 2: Custom Routing Hints

```python
@dataclass
class CustomRoutingHints(RoutingHints):
    """Extended routing hints for custom routing."""
    priority: int = 0
    deadline_ms: int | None = None
    sla_tier: str = "standard"

class CustomRouter(RequestRouter):
    """Custom router using extended hints."""

    async def route_request(
        self,
        request: DeploymentRequest,
        replicas: list[DeploymentReplicaInfo],
    ) -> DeploymentReplicaInfo:
        hints = request.routing_hints

        # Use custom fields
        if isinstance(hints, CustomRoutingHints):
            if hints.priority > 10:
                # Route high-priority to least-loaded
                return min(replicas, key=lambda r: r.queue_length)

        # Default routing
        return replicas[0]
```

### Example 3: Testing with Synthetic Hints

```python
async def test_context_aware_routing():
    """Test router with synthetic routing hints."""
    router = ContextAwareRouter()

    # Create synthetic hints (no need for full InferenceRequest)
    hints = RoutingHints(
        policy=RoutingPolicy.CONTEXT_AWARE,
        context_page_ids=["page-1", "page-2"],
        tenant_id="tenant-a",
    )

    request = DeploymentRequest(
        request_id="test-1",
        method_name="infer",
        args=(),
        kwargs={},
        routing_hints=hints,
    )

    replica = await router.route_request(request, replicas)
    assert replica is not None
```

## Summary

This design transforms routing from **implicit and fragile** to **explicit and robust**:

| Aspect | Before | After |
|--------|--------|-------|
| **Configuration** | Implicit in router code | Declarative `@endpoint(routing_policy=...)` |
| **Hint Extraction** | Routers peek into args | Automatic extraction at handle level |
| **Type Safety** | `isinstance()` checks | Typed `RoutingHints` dataclass |
| **Extensibility** | Modify routers | Add to `RoutingPolicy` enum |
| **Testability** | Need full request objects | Synthetic `RoutingHints` |
| **Performance** | Re-parse args per router | Extract once, reuse |
| **Coupling** | Tight (routers know arg structure) | Loose (routers use hints) |

The result is a **production-ready, maintainable, and extensible** routing system that will scale as we add more routing strategies and deployment types.