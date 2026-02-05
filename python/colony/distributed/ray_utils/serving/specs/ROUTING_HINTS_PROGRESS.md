# Routing Hints Implementation Progress

## Overview

This document tracks the implementation of first-class routing hints for the serving framework, eliminating the fragile `_extract_inference_request()` pattern in favor of declarative, type-safe routing.

## ✅ Completed Work

### 1. **Comprehensive Design Document** ✓
**File**: `ROUTING_HINTS_DESIGN.md`

- Problem analysis and architectural overview
- Complete component design (RoutingPolicy, RoutingHints, extractors, etc.)
- Usage examples and migration path
- Benefits analysis comparing before/after

### 2. **Core Data Models** ✓
**File**: `models.py`

**Added RoutingPolicy Enum:**
```python
class RoutingPolicy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CONTEXT_AWARE = "context_aware"      # LLM-specific
    PAGE_AFFINITY = "page_affinity"       # Strict affinity
    CUSTOM = "custom"
```

**Added RoutingHints Dataclass:**
```python
@dataclass
class RoutingHints:
    policy: RoutingPolicy | None
    context_page_ids: list[str] | None
    tenant_id: str | None
    requirements: LLMClientRequirements | None
    affinity_key: str | None
    metadata: dict[str, Any]

    @staticmethod
    def from_inference_request(...)  # Automatic extraction helper
```

**Enhanced DeploymentRequest:**
```python
class DeploymentRequest(BaseModel):
    ...
    routing_hints: RoutingHints | None = None  # ✨ NEW FIELD
```

### 3. **Enhanced @endpoint Decorator** ✓
**File**: `decorators.py`

The decorator now accepts routing_policy parameter:

```python
# Without parameters (backward compatible)
@serving.endpoint
async def health_check(self) -> str:
    return "healthy"

# With routing policy
@serving.endpoint(routing_policy=RoutingPolicy.CONTEXT_AWARE)
async def infer(self, request: InferenceRequest) -> InferenceResponse:
    # Automatically routed based on context pages
    ...
```

**DeploymentConfig Enhancement:**
- Added `endpoint_routing_policies` dict to store per-method routing policies
- Added `register_endpoint()` method
- Added `get_endpoint_routing_policy()` method
- Automatic discovery of endpoint routing policies during deployment decoration

### 4. **RoutingHintExtractor Utility** ✓
**File**: `routing_hints_extractor.py`

Extracts routing hints from method arguments without fragile pattern matching:

```python
class RoutingHintExtractor:
    @staticmethod
    def extract(
        method_name: str,
        args: tuple,
        kwargs: dict,
        routing_policy: RoutingPolicy | None,
    ) -> RoutingHints:
        # Finds InferenceRequest in args/kwargs
        # Extracts context_page_ids, tenant_id, requirements
        # Returns structured RoutingHints
        ...
```

**Features:**
- Type-safe extraction (no isinstance checks in routers)
- Handles InferenceRequest in common positions (first arg, common kwarg names)
- Fallback to basic hints when InferenceRequest not found
- Graceful handling of import errors

### 5. **Enhanced DeploymentHandle** ✓
**File**: `handle.py`

The handle now extracts and attaches routing hints before sending requests:

**Changes:**
- Added `deployment_class` parameter to constructor
- Added `_endpoint_routing_policies` cache
- Added `_get_endpoint_routing_policy()` helper method
- Updated `call_method()` to extract routing hints using `RoutingHintExtractor`
- Routing hints automatically attached to all `DeploymentRequest` objects

**Flow:**
```python
# 1. User calls method
result = await handle.infer(inference_request)

# 2. Handle extracts routing policy from endpoint config
routing_policy = RoutingPolicy.CONTEXT_AWARE

# 3. Handle extracts hints from arguments
hints = RoutingHintExtractor.extract(
    method_name="infer",
    args=(inference_request,),
    kwargs={},
    routing_policy=routing_policy,
)

# 4. Request created with hints attached
request = DeploymentRequest(
    method_name="infer",
    args=(inference_request,),
    routing_hints=hints,  # ✨ Hints attached!
)

# 5. Proxy uses hints for routing (see pending work below)
```

## 🔄 Remaining Work

### 6. **Update DeploymentProxyRayActor** (Pending)
**File**: `proxy.py`
**Status**: Not started

**What needs to be done:**
1. Add router registry (mapping `RoutingPolicy` → `RequestRouter`)
2. Add `_get_router()` method that selects router based on `request.routing_hints.policy`
3. Update `handle_request()` to use `_get_router()` instead of hardcoded default router
4. Lazy-initialize specialized routers (ContextAwareRouter, PageAffinityRouter)

**Pseudocode:**
```python
class DeploymentProxyRayActor:
    def __init__(self, ...):
        self.default_router = LeastLoadedRouter()
        self.specialized_routers: dict[RoutingPolicy, RequestRouter] = {
            RoutingPolicy.ROUND_ROBIN: RoundRobinRouter(),
            RoutingPolicy.LEAST_LOADED: LeastLoadedRouter(),
        }

    def _get_router(self, routing_hints: RoutingHints | None) -> RequestRouter:
        if not routing_hints or not routing_hints.policy:
            return self.default_router

        policy = routing_hints.policy

        # Return specialized router or initialize lazily
        if policy == RoutingPolicy.CONTEXT_AWARE:
            if policy not in self.specialized_routers:
                from polymathera.llms_0.cluster.routing import ContextAwareRouter
                self.specialized_routers[policy] = ContextAwareRouter()
            return self.specialized_routers[policy]

        # ... similar for PAGE_AFFINITY ...

        return self.default_router

    async def handle_request(self, request: DeploymentRequest):
        # Select router based on routing hints
        router = self._get_router(request.routing_hints)

        # Route request
        replica = await router.route_request(request, healthy_replicas)
        ...
```

### 7. **Update ContextAwareRouter** (Pending)
**File**: `polymathera/llms_0/cluster/routing.py`
**Status**: Not started

**What needs to be done:**
1. Remove `_extract_inference_request()` method
2. Update `route_request()` to use `request.routing_hints.context_page_ids` directly
3. Use `request.routing_hints.tenant_id` for tenant isolation
4. Use `request.routing_hints.requirements` for requirement matching

**Before (fragile):**
```python
async def route_request(self, request: DeploymentRequest, replicas: list) -> DeploymentReplicaInfo:
    inference_req = self._extract_inference_request(request)  # ❌ Fragile!
    if not inference_req or not inference_req.context_page_ids:
        return fallback_routing()

    required_pages = set(inference_req.context_page_ids)
    ...
```

**After (robust):**
```python
async def route_request(self, request: DeploymentRequest, replicas: list) -> DeploymentReplicaInfo:
    if not request.routing_hints or not request.routing_hints.context_page_ids:
        return fallback_routing()

    required_pages = set(request.routing_hints.context_page_ids)  # ✅ Type-safe!
    ...
```

### 8. **Update PageAffinityRouter** (Pending)
**File**: `polymathera/llms_0/cluster/routing.py`
**Status**: Not started

**What needs to be done:**
1. Remove `_extract_inference_request()` method
2. Update `route_request()` to use `request.routing_hints.context_page_ids`
3. Ensure strict affinity logic works with hints (fail if not all pages loaded)

**Similar changes to ContextAwareRouter but with stricter validation.**

## 📋 Testing Plan

Once all components are implemented, test the following scenarios:

### Test 1: Context-Aware Routing
```python
@serving.deployment()
class VLLMDeployment:
    @serving.endpoint(routing_policy=RoutingPolicy.CONTEXT_AWARE)
    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        ...

# Test that requests with context_page_ids route to replicas with those pages loaded
request = InferenceRequest(
    request_id="test-1",
    prompt="Generate code",
    context_page_ids=["page-1", "page-2"],
)
handle = serving.get_deployment("MyApp", deployment_class=VLLMDeployment)
response = await handle.infer(request)

# Verify: Request routed to replica with pages loaded (check StateManager)
```

### Test 2: Page Affinity Routing
```python
@serving.endpoint(routing_policy=RoutingPolicy.PAGE_AFFINITY)
async def infer_strict(self, request: InferenceRequest) -> InferenceResponse:
    ...

# Test that requests fail if no replica has ALL required pages
```

### Test 3: Default Routing (Backward Compatibility)
```python
@serving.endpoint  # No routing policy specified
async def health_check(self) -> str:
    return "healthy"

# Test that defaults to LEAST_LOADED routing
```

### Test 4: Routing Without InferenceRequest
```python
@serving.endpoint(routing_policy=RoutingPolicy.ROUND_ROBIN)
async def process(self, data: str) -> str:
    return f"processed: {data}"

# Test that falls back to basic routing when no InferenceRequest present
```

## 📊 Benefits Achieved

| Aspect | Before | After |
|--------|--------|-------|
| **Configuration** | Implicit in router code | Declarative `@endpoint(routing_policy=...)` |
| **Hint Extraction** | Routers peek into args | Automatic extraction at handle level |
| **Type Safety** | `isinstance()` checks everywhere | Typed `RoutingHints` dataclass |
| **Extensibility** | Modify routers for new methods | Add to `RoutingPolicy` enum |
| **Testability** | Need full InferenceRequest objects | Synthetic `RoutingHints` work |
| **Performance** | Re-parse args per router | Extract once, reuse hints |
| **Coupling** | Tight (routers know arg structure) | Loose (routers use hints) |
| **Maintainability** | Fragile pattern matching | Explicit, documented contracts |

## 🎯 Next Steps

**To complete the implementation:**

1. **Update DeploymentProxyRayActor** (proxy.py)
   - Add router selection based on routing hints
   - Implement lazy initialization of specialized routers

2. **Update ContextAwareRouter** (llms_0/cluster/routing.py)
   - Remove `_extract_inference_request()`
   - Use `request.routing_hints` directly

3. **Update PageAffinityRouter** (llms_0/cluster/routing.py)
   - Remove `_extract_inference_request()`
   - Use `request.routing_hints` directly

4. **Test End-to-End**
   - Write integration tests
   - Verify routing works correctly
   - Test backward compatibility

5. **Update Documentation**
   - Update SPECS_LLM_CLUSTER.md with new routing architecture
   - Add examples to endpoint decorator docstrings
   - Document migration path for custom routers

## 🔍 Files Modified

- ✅ `polymathera/rayutils/serving/models.py` - Added RoutingPolicy, RoutingHints, enhanced DeploymentRequest
- ✅ `polymathera/rayutils/serving/decorators.py` - Enhanced @endpoint decorator, DeploymentConfig
- ✅ `polymathera/rayutils/serving/routing_hints_extractor.py` - NEW utility class
- ✅ `polymathera/rayutils/serving/handle.py` - Enhanced DeploymentHandle with routing hint extraction
- ⏳ `polymathera/rayutils/serving/proxy.py` - PENDING (router selection)
- ⏳ `polymathera/llms_0/cluster/routing.py` - PENDING (update routers)

## 📚 Documentation Created

- ✅ `ROUTING_HINTS_DESIGN.md` - Comprehensive design document
- ✅ `ROUTING_HINTS_PROGRESS.md` - This progress document
- ✅ `PAGE_TYPES_RELATIONSHIP.md` - VirtualContextPage vs LoadedContextPage (earlier work)

## 💡 Key Design Decisions

1. **Routing hints are optional**: If `routing_hints` is None, fall back to default routing
2. **Policy defaults to LEAST_LOADED**: Safe, predictable default behavior
3. **Extraction happens at handle level**: Centralized, consistent extraction logic
4. **Lazy router initialization**: Only create specialized routers when needed
5. **Backward compatibility**: Existing code without `routing_policy` continues to work
6. **Type-safe throughout**: From decorator to router, everything is strongly typed

## ⚠️ Migration Notes

**For custom routers:**
- Update `route_request()` signature to use `request.routing_hints`
- Remove any arg inspection or `_extract_*()` helper methods
- Use typed fields from `RoutingHints` instead of casting args

**For deployment methods:**
- Optionally add `routing_policy` to `@endpoint` decorators
- No changes required for existing code (backward compatible)

**For tests:**
- Can use synthetic `RoutingHints` instead of full requests
- Easier to test routing logic in isolation