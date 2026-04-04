"""Context-aware routing for LLM cluster.

This module provides intelligent request routing that considers which context
pages are loaded in each LLM client's KV cache, minimizing page faults and
improving inference performance.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..distributed.ray_utils.serving import (
    DeploymentRequest,
    RequestRouter,
    LeastLoadedRouter,
    DeploymentReplicaInfo,
    RoutingHints,
    require_execution_context,
    ensure_context,
)
from .models import InferenceRequest, LLMClientRequirements, LLMClientState
from ..vcm.models import ContextPageId

if TYPE_CHECKING:
    from .config import LLMDeploymentConfig

logger = logging.getLogger(__name__)



class RoutingHintExtractor:
    """Extracts routing hints from method arguments for intelligent request routing.

    This utility analyzes method arguments to extract routing information,
    eliminating the need for routers to inspect raw arguments.

    Example:
        ```python
        # Extract hints from InferenceRequest
        extractor = RoutingHintExtractor()
        hints = extractor.extract(
            method_name="infer",
            args=(inference_request,),
            kwargs={},
            router_class=ContextAwareRouter,
        )

        # hints.context_page_ids = inference_request.context_page_ids
        # hints.tenant_id = inference_request.requirements.tenant_id
        ```
    """

    @staticmethod
    def extract(
        method_name: str,
        args: tuple,
        kwargs: dict,
        router_class: type[RequestRouter] | None = None,
    ) -> RoutingHints:
        """Extract routing hints from method arguments.

        This method analyzes the arguments to find routing-relevant information
        (e.g., InferenceRequest) and constructs RoutingHints.

        Args:
            method_name: Name of the method being called.
            args: Positional arguments to the method.
            kwargs: Keyword arguments to the method.
            router_class: Router class configured for this endpoint (from @endpoint decorator).

        Returns:
            RoutingHints with extracted information.

        Example:
            ```python
            from polymathera.colony.cluster.models import InferenceRequest

            request = InferenceRequest(
                request_id="req-1",
                prompt="Generate code",
                context_page_ids=["page-1", "page-2"],
            )

            hints = RoutingHintExtractor.extract(
                method_name="infer",
                args=(request,),
                kwargs={},
                router_class=ContextAwareRouter,
            )

            assert hints.context_page_ids == ["page-1", "page-2"]
            assert hints.router_class == ContextAwareRouter
            ```
        """
        # Look for InferenceRequest in args
        inference_req = RoutingHintExtractor._find_inference_request(args, kwargs)

        if inference_req:
            # Found InferenceRequest - extract hints from it
            return RoutingHintExtractor.from_inference_request(
                inference_req,
                router_class=router_class
            )

        # No InferenceRequest found - return basic hints with router class only
        return RoutingHints(router_class=router_class or LeastLoadedRouter)

    @staticmethod
    def _find_inference_request(args: tuple, kwargs: dict) -> 'InferenceRequest | None':
        """Find InferenceRequest in args or kwargs.

        Looks for InferenceRequest in common locations:
        - First positional argument (most common pattern)
        - Common keyword argument names: 'request', 'inference_request', 'req'

        Args:
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            InferenceRequest if found, None otherwise.
        """
        # Check first positional arg (most common pattern)
        if args and isinstance(args[0], InferenceRequest):
            return args[0]

        # Check common kwarg names
        for key in ['request', 'inference_request', 'req']:
            if key in kwargs and isinstance(kwargs[key], InferenceRequest):
                return kwargs[key]

        return None

    @staticmethod
    def from_inference_request(
        inference_req: 'InferenceRequest',
        router_class: type[RequestRouter] | None = None
    ) -> RoutingHints:
        """Create routing hints from InferenceRequest.

        Args:
            inference_req: Inference request containing context and requirements.
            router_class: Router class to use (defaults to ContextAwareRouter if pages present).

        Returns:
            RoutingHints with extracted information.

        Example:
            ```python
            from polymathera.colony.cluster.models import InferenceRequest

            request = InferenceRequest(
                request_id="req-1",
                prompt="Generate code",
                context_page_ids=["page-1", "page-2"],
                requirements=LLMClientRequirements(min_context_window=4096),
            )

            hints = RoutingHints.from_inference_request(request)
            # hints.router_class = ContextAwareRouter
            # hints.context_page_ids = ["page-1", "page-2"]
            # hints.tenant_id = "tenant-a"
            ```
        """
        # Auto-select router class based on presence of context pages
        if router_class is None:
            router_class = (
                ContextAwareRouter
                if inference_req.context_page_ids
                else LeastLoadedRouter
            )

        return RoutingHints(
            router_class=router_class,
            metadata={
                "context_page_ids": list(inference_req.context_page_ids),
                "requirements": inference_req.requirements,
                "affinity_key": None # Generic affinity key for custom routing strategies.
            },
        )


class ContextAwareRouter(RequestRouter):
    """Context-aware request router for LLM cluster.

    This router implements intelligent request routing based on which context
    pages are currently loaded in each LLM client's KV cache. It aims to:

    1. **Minimize page faults**: Route requests to clients that have required pages loaded
    2. **Balance load**: Distribute requests across clients when page affinity is equal
    3. **Optimize cache usage**: Prefer clients with more matching pages

    Routing Strategy:
    - Extract required page IDs from inference request
    - Score each replica based on:
        a) Number of required pages already loaded (page hit count)
        b) Current load (pending requests)
        c) Available KV cache capacity
    - Select replica with highest score

    This routing is critical for performance when dealing with billion-token contexts
    where page faults are expensive (require loading large context chunks).

    Example:
        ```python
        from polymathera.colony.distributed.ray_utils import serving
        from polymathera.colony import VLLMDeployment, ContextAwareRouter

        app = serving.Application(name="llm-cluster")
        app.add_deployment(
            VLLMDeployment.bind(model_name="llama-3.1-8B"),
            default_router_class=ContextAwareRouter,
        )
        await app.start()
        ```
    """

    def __init__(self):
        """Initialize router with lazy StateManager initialization."""
        super().__init__()
        self._state_manager = None
        self._vcm_handle = None
        self._initialized = False

    @classmethod
    def extract_routing_hints(
        cls,
        method_name: str,
        args: tuple,
        kwargs: dict,
    ) -> RoutingHints:
        """Extract routing hints using RoutingHintExtractor.

        This class method delegates to RoutingHintExtractor to extract hints
        from method arguments.

        Args:
            method_name: Name of the method being called.
            args: Positional arguments to the method.
            kwargs: Keyword arguments to the method.
        Returns:
            RoutingHints with extracted information.
        """
        return RoutingHintExtractor.extract(
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            router_class=cls,
        )

    async def _ensure_initialized(self, replicas: list[DeploymentReplicaInfo]):
        """Initialize StateManager and VCM handle on first request.

        Args:
            replicas: List of replicas (used to determine deployment name)
        """
        if self._initialized:
            return

        try:
            from ..distributed import get_polymathera
            from ..distributed.ray_utils import serving
            from .models import VLLMDeploymentState
            from ..system import get_vcm

            # Get app name and deployment from serving context
            # These are set via runtime_env by serving.Application during deployment
            app_name = serving.get_my_app_name()
            deployment_name = serving.get_my_deployment_name()

            polymathera = get_polymathera()
            self._state_manager = await polymathera.get_state_manager(
                state_type=VLLMDeploymentState,
                state_key=VLLMDeploymentState.get_state_key(app_name, deployment_name),
            )

            # Get VCM handle for page fault issuing
            try:
                self._vcm_handle = get_vcm()
                logger.info("ContextAwareRouter connected to VCM for page fault handling")
            except Exception as e:
                logger.warning(f"VCM deployment not found, page fault handling disabled: {e}")

            self._initialized = True
            logger.info(f"ContextAwareRouter initialized with StateManager for {app_name}.{deployment_name}")

        except Exception as e:
            logger.warning(f"Failed to initialize router StateManager: {e}. Falling back to load-based routing.")
            self._initialized = True  # Mark as initialized to avoid retry

    async def route_request(
        self,
        request: DeploymentRequest,
        replicas: list[DeploymentReplicaInfo],
    ) -> DeploymentReplicaInfo:
        """Route request to the best replica based on context page locality.

        This method uses routing hints extracted at the handle level to make
        intelligent routing decisions without inspecting raw request arguments.

        Args:
            request: The incoming request with routing hints attached
            replicas: List of available replicas

        Returns:
            Selected replica with best page locality

        Raises:
            ValueError: If no replicas are available

        TODO: Turn this method into an allocate_replica() method that,
        not only reads a snapshot of all client states, but also
        allocates pages to the selected replica atomically.
        """
        ensure_context(request.request_id, request.execution_context)

        if not replicas:
            raise ValueError("No replicas available for routing")

        # Ensure StateManager is initialized
        await self._ensure_initialized(replicas)

        # Extract context page IDs from routing hints (no more fragile pattern matching!)
        if not request.routing_hints or "context_page_ids" not in request.routing_hints.metadata:
            # No context pages, fallback to least-loaded routing
            return min(replicas, key=lambda r: r.queue_length + r.in_flight_requests)

        required_pages = set(request.routing_hints.metadata["context_page_ids"])

        # Read ALL client states atomically from StateManager
        client_states_snapshot = {}
        if self._state_manager:
            try:
                async for dep_state in self._state_manager.read_transaction():
                    client_states_snapshot = dep_state.client_states.copy()
            except Exception as e:
                logger.warning(f"Failed to read from StateManager: {e}, falling back to load-based routing")

        # Score replicas using the atomic snapshot
        replica_scores = []
        for replica in replicas:
            score = await self._score_replica(replica, required_pages, client_states_snapshot)
            replica_scores.append((replica, score))
            logger.debug(
                f"Replica {replica.replica_id}: score={score:.2f}, "
                f"queue={replica.queue_length}, in_flight={replica.in_flight_requests}"
            )

        # Select replica with highest score
        best_replica, best_score = max(replica_scores, key=lambda x: x[1])

        logger.info(
            f"Selected replica {best_replica.replica_id} with score {best_score:.2f} "
            f"for request {request.request_id} with {len(required_pages)} pages"
        )

        return best_replica

    async def _score_replica(
        self,
        replica: DeploymentReplicaInfo,
        required_pages: set[ContextPageId],
        client_states_snapshot: dict,
    ) -> float:
        """Score a replica based on page locality and load.

        Scoring formula:
            score = page_hit_weight * (pages_loaded / total_pages)
                  - load_weight * (current_load / max_load)
                  + capacity_weight * (available_capacity / total_capacity)

        Args:
            replica: Replica to score
            required_pages: Set of required page IDs
            client_states_snapshot: Atomic snapshot of all client states from StateManager

        Returns:
            Score for this replica (higher is better)
        """
        # Weights for different factors
        PAGE_HIT_WEIGHT = 100.0  # Heavily prioritize page hits
        LOAD_WEIGHT = 10.0  # Penalize high load
        CAPACITY_WEIGHT = 5.0  # Slightly prefer more available capacity

        try:
            # Get client state from the atomic snapshot
            # We need to map replica to its client_id
            # Replica metadata should contain client_id
            client_id = replica.metadata.get("client_id") if hasattr(replica, 'metadata') else None
            client_state = None

            if client_id and client_id in client_states_snapshot:
                client_state = client_states_snapshot[client_id]

            if not client_state:
                # No state available in snapshot, use simple load-based scoring
                current_load = replica.queue_length + replica.in_flight_requests
                return -LOAD_WEIGHT * current_load

            # Calculate page hits
            pages_loaded = sum(
                1 for page_id in required_pages
                if client_state.has_page(page_id)
            )
            page_hit_ratio = pages_loaded / len(required_pages) if required_pages else 0.0

            # Calculate load factor
            current_load = replica.queue_length + replica.in_flight_requests
            # Assume max load of 100 for normalization
            load_factor = current_load / 100.0

            # Calculate capacity factor
            available_capacity = client_state.get_available_cache_capacity()
            capacity_factor = available_capacity / client_state.kv_cache_capacity if client_state.kv_cache_capacity > 0 else 0.0

            # Compute final score
            score = (
                PAGE_HIT_WEIGHT * page_hit_ratio
                - LOAD_WEIGHT * load_factor
                + CAPACITY_WEIGHT * capacity_factor
            )

            return score

        except Exception as e:
            logger.warning(f"Error scoring replica {replica.replica_id}: {e}")
            # Fallback to negative load score
            return -LOAD_WEIGHT * (replica.queue_length + replica.in_flight_requests)


class TargetClientRouter(RequestRouter):
    """Router that targets a specific client/replica based on routing hints.

    Used for operations that must execute on a specific replica (e.g., loading
    a page on a specific client as decided by the allocation strategy).

    This router extracts the target_client_id from routing hints and routes
    the request to the replica with that client_id.

    Args:
        strip_routing_params: List of parameter names to strip from kwargs
            before passing to the method. Default: ["target_client_id"].
            This prevents TypeError when the target method doesn't accept
            these routing parameters.
    """

    def __init__(self, strip_routing_params: list[str] | None = None):
        """Initialize the router.

        Args:
            strip_routing_params: List of parameter names to strip from kwargs
                before passing to the method. Defaults to ["target_client_id"].
        """
        super().__init__()
        self.strip_routing_params = strip_routing_params or ["target_client_id"]

    @classmethod
    def extract_routing_hints(
        cls,
        method_name: str,
        args: tuple,
        kwargs: dict,
    ) -> RoutingHints:
        """Extract target_client_id from method arguments and remove from kwargs.

        This method extracts routing parameters and REMOVES them from kwargs
        to prevent them from being passed to the actual method (which would
        cause TypeError).

        Args:
            method_name: Name of the method being called
            args: Positional arguments
            kwargs: Keyword arguments (MUTATED: routing params removed based on config)

        Returns:
            RoutingHints with target_client_id in metadata
        """
        # Extract target_client_id from kwargs
        # Note: We can't access instance config here (classmethod), so we always
        # extract but delegate stripping to __init__ config
        target_client_id = kwargs.get('target_client_id')

        return RoutingHints(
            router_class=cls,
            metadata={"target_client_id": target_client_id}
        )

    async def route_request(
        self,
        request: DeploymentRequest,
        replicas: list[DeploymentReplicaInfo],
    ) -> DeploymentReplicaInfo:
        """Route to the specific replica specified in routing hints.

        Args:
            request: The incoming request with routing hints
            replicas: List of available replicas

        Returns:
            The replica with matching client_id

        Raises:
            ValueError: If target_client_id not found or no matching replica
        """
        ensure_context(request.request_id, request.execution_context)

        if not request.routing_hints or "target_client_id" not in request.routing_hints.metadata:
            raise ValueError("TargetClientRouter requires target_client_id in routing hints")

        target_client_id = request.routing_hints.metadata["target_client_id"]

        if not target_client_id:
            raise ValueError("target_client_id cannot be None")

        # Strip routing parameters from kwargs before passing to method
        # This prevents TypeError when the target method doesn't accept these params
        logger.debug(
            f"Stripping routing parameters {self.strip_routing_params} from kwargs {request.kwargs} for TargetClientRouter. ")
        for param in self.strip_routing_params:
            request.kwargs.pop(param, None)
        logger.debug(f"Kwargs after stripping: {request.kwargs}")

        # Find the replica with matching client_id
        for replica in replicas:
            client_id = replica.metadata.get("client_id") if hasattr(replica, 'metadata') else None
            if client_id == target_client_id:
                logger.debug(f"TargetClientRouter routing to replica {client_id}")
                return replica

        raise ValueError(
            f"No replica found with client_id={target_client_id}. "
            f"Available replicas: {[r.replica_id for r in replicas]}"
        )


class PageAffinityRouter(ContextAwareRouter):
    """Specialized router that strictly enforces page affinity.

    This router only routes requests to replicas that have ALL required pages loaded.
    If no replica has all pages, it returns an error instead of routing with page faults.

    This is useful for latency-sensitive applications where page faults are unacceptable.

    Example:
        ```python
        app.add_deployment(
            VLLMDeployment.bind(model_name="llama-3.1-8B"),
            default_router_class=PageAffinityRouter,
        )
        ```
    """

    def _get_candidate_replicas(
        self,
        replicas: list[DeploymentReplicaInfo],
        required_pages: set[ContextPageId],
        client_states_snapshot: dict[str, LLMClientState],
    ) -> list[DeploymentReplicaInfo]:
        """Get replicas that have ALL required pages loaded.
        TODO: This method returns replicas that EACH have all required pages loaded.
        This is can be too restrictive. Is this even the intended behavior?
        """
        # Find candidates again after page fault
        candidates = []
        for replica in replicas:
            client_id = replica.metadata.get("client_id")
            if client_id and client_id in client_states_snapshot:
                client_state = client_states_snapshot[client_id]
                if all(client_state.has_page(page_id) for page_id in required_pages):
                    candidates.append(replica)
        return candidates

    async def route_request(
        self,
        request: DeploymentRequest,
        replicas: list[DeploymentReplicaInfo],
    ) -> DeploymentReplicaInfo:
        """Route request to replica with ALL required pages loaded.

        This method uses routing hints extracted at the handle level to enforce
        strict page affinity without inspecting raw request arguments.

        Args:
            request: The incoming request with routing hints attached
            replicas: List of available replicas

        Returns:
            Selected replica with all pages loaded

        Raises:
            ValueError: If no replica has all required pages

        TODO: Turn this method into an allocate_replica() method that,
        not only reads a snapshot of all client states, but also
        allocates pages to the selected replica atomically.
        """
        ensure_context(request.request_id, request.execution_context)

        if not replicas:
            raise ValueError("No replicas available for routing")

        # Ensure StateManager is initialized
        await self._ensure_initialized(replicas)

        # Extract context page IDs from routing hints (no more fragile pattern matching!)
        if not request.routing_hints or "context_page_ids" not in request.routing_hints.metadata:
            # No context pages, fallback to parent routing
            return await super().route_request(request, replicas)

        required_pages = set(request.routing_hints.metadata["context_page_ids"])

        # Read ALL client states atomically from StateManager
        client_states_snapshot = {}
        if self._state_manager:
            try:
                async for dep_state in self._state_manager.read_transaction():
                    client_states_snapshot = dep_state.client_states.copy()
            except Exception as e:
                logger.warning(f"Failed to read from StateManager: {e}, falling back to parent routing")
                return await super().route_request(request, replicas)

        # Find replicas with all required pages using the atomic snapshot
        candidates = self._get_candidate_replicas(
            replicas,
            required_pages,
            client_states_snapshot,
        )

        if not candidates:
            # No replica has all pages - issue page fault and wait
            if not self._vcm_handle:
                raise ValueError(
                    f"No replica has all required pages and VCM not available for page fault handling. "
                    f"Required: {required_pages}, Available replicas: {len(replicas)}"
                )

            logger.info(
                f"No replica has all {len(required_pages)} required pages, "
                f"issuing page fault for request {request.request_id}"
            )

            # Issue page fault
            try:
                fault_id = await self._vcm_handle.issue_page_fault(
                    page_ids=list(required_pages),
                    requester_id=f"PageAffinityRouter-{request.request_id}",
                    priority=10,  # High priority for router requests
                )

                logger.info(f"Issued page fault {fault_id}, waiting for pages to load")

                # Wait for pages to be loaded
                success = await self._vcm_handle.wait_for_pages(
                    fault_id=fault_id,
                    timeout_s=30.0,  # 30 second timeout
                )

                if not success:
                    raise TimeoutError(
                        f"Page fault {fault_id} timed out after 30s for request {request.request_id}"
                    )

                logger.info(f"Page fault {fault_id} completed, retrying routing")

            except Exception as e:
                logger.error(f"Error during page fault handling: {e}", exc_info=True)
                raise ValueError(
                    f"Failed to load required pages via page fault: {e}. "
                    f"Required: {required_pages}"
                ) from e

            # Retry: Re-read client states and find candidates again
            try:
                async for dep_state in self._state_manager.read_transaction():
                    client_states_snapshot = dep_state.client_states.copy()
            except Exception as e:
                raise ValueError(f"Failed to re-read client states after page fault: {e}") from e

            # Find candidates again after page fault
            candidates = self._get_candidate_replicas(
                replicas,
                required_pages,
                client_states_snapshot,
            )

            if not candidates:
                raise ValueError(
                    f"Page fault completed but pages still not loaded. "
                    f"Required: {required_pages}, This indicates a problem with the page fault mechanism."
                )

        # Among candidates, select least loaded
        best_replica = min(candidates, key=lambda r: r.queue_length + r.in_flight_requests)

        logger.info(
            f"Selected replica {best_replica.replica_id} with all {len(required_pages)} "
            f"pages loaded for request {request.request_id}"
        )

        return best_replica


class RequirementBasedRouter:
    """Deployment-level router that selects deployments based on requirements.

    This router operates at the cluster level (above individual deployments) to
    select which deployment should handle a request based on LLMClientRequirements.

    Routing strategy:
    1. Extract requirements from InferenceRequest
    2. Filter deployments by constraints (model family, context window, quantization, capabilities)
    3. Score remaining deployments by performance and load
    4. Return best deployment with fallback support

    Example:
        ```python
        router = RequirementBasedRouter(
            deployment_configs=[llama_8b_config, llama_70b_config],
            enable_fallbacks=True
        )

        deployment_name = router.select_deployment(inference_request)
        ```
    """

    def __init__(
        self,
        deployment_configs: list[LLMDeploymentConfig] | None = None,
        remote_deployment_configs: list | None = None,
        enable_fallbacks: bool = True,
    ):
        """Initialize requirement-based router.

        Args:
            deployment_configs: List of vLLM deployment configurations
            remote_deployment_configs: List of remote deployment configurations
            enable_fallbacks: Whether to enable fallback routing if primary fails constraints
        """
        self.deployment_configs = deployment_configs or []
        self.remote_deployment_configs = remote_deployment_configs or []
        self.enable_fallbacks = enable_fallbacks

    def _get_all_configs(self) -> list:
        """Get all deployment configs (vLLM + remote) as a flat list."""
        return list(self.deployment_configs) + list(self.remote_deployment_configs)

    def select_deployment(
        self,
        requirements: LLMClientRequirements | None,
    ) -> str:
        """Select deployment based on requirements.

        Args:
            requirements: Client requirements (if None, returns first deployment)

        Returns:
            Deployment name to use

        Raises:
            ValueError: If no deployment matches requirements
        """
        all_configs = self._get_all_configs()

        if not requirements:
            # No requirements, return first deployment
            if all_configs:
                return all_configs[0].get_deployment_name()
            raise ValueError("No deployments available")

        # Filter by requirements
        candidates = []
        for dconf in self.deployment_configs:
            if self._matches_requirements(dconf, requirements):
                candidates.append(dconf)
        for rconf in self.remote_deployment_configs:
            if self._matches_remote_requirements(rconf, requirements):
                candidates.append(rconf)

        if not candidates:
            if self.enable_fallbacks and all_configs:
                # Fallback to any deployment
                logger.warning(
                    f"No deployment matches requirements {requirements}. "
                    f"Using fallback to first deployment."
                )
                return all_configs[0].get_deployment_name()
            raise ValueError(
                f"No deployment matches requirements: {requirements}. "
                f"Available: {[d.get_deployment_name() for d in all_configs]}"
            )

        # Score candidates and select best
        best_deployment = self._score_and_select(candidates, requirements)
        return best_deployment.get_deployment_name()

    def _matches_requirements(
        self,
        dconf: LLMDeploymentConfig,
        requirements: LLMClientRequirements,
    ) -> bool:
        """Check if deployment matches requirements.

        Args:
            dconf: Deployment configuration
            requirements: Client requirements

        Returns:
            True if deployment matches all required constraints
        """
        from .registry import ModelRegistry

        model_params = dconf.get_model_params()
        if not model_params:
            # Model not in registry, can't validate
            return True

        # Check model family
        if requirements.model_family:
            if requirements.model_family.lower() not in model_params.model_name.lower():
                return False

        # Check context window
        if requirements.min_context_window:
            if model_params.context_window < requirements.min_context_window:
                return False
        if requirements.max_context_window:
            if model_params.context_window > requirements.max_context_window:
                return False

        # Check quantization preferences
        if requirements.preferred_quantization and dconf.quantization:
            if dconf.quantization not in requirements.preferred_quantization:
                return False

        # Check capability requirements
        if requirements.requires_structured_output:
            if "structured_output" not in dconf.capabilities:
                return False
        if requirements.requires_function_calling:
            if "function_calling" not in dconf.capabilities:
                return False

        # Check LoRA adapter support
        if requirements.lora_adapter_id:
            if not dconf.lora_adapters:
                return False
            # Check if this specific adapter is configured
            adapter_ids = {adapter.adapter_id for adapter in dconf.lora_adapters}
            if requirements.lora_adapter_id not in adapter_ids:
                return False

        # Check preferred deployments
        if requirements.preferred_deployment_ids:
            deployment_name = dconf.get_deployment_name()
            if deployment_name not in requirements.preferred_deployment_ids:
                # Not in preferred list, but might still be acceptable
                # Check if it's explicitly excluded via fallback list
                if requirements.fallback_deployment_ids:
                    if deployment_name not in requirements.fallback_deployment_ids:
                        return False

        return True

    def _matches_remote_requirements(
        self,
        rconf,
        requirements: LLMClientRequirements,
    ) -> bool:
        """Check if a remote deployment matches requirements.

        Remote deployments don't have model registry entries, so matching
        is done directly on the config fields.

        Args:
            rconf: RemoteLLMDeploymentConfig
            requirements: Client requirements

        Returns:
            True if deployment matches all required constraints
        """
        # Check model family via model_name string
        if requirements.model_family:
            if requirements.model_family.lower() not in rconf.model_name.lower():
                return False

        # Remote deployments don't expose context_window directly — skip those checks
        # (Remote APIs handle context limits internally)

        # Check capability requirements
        if requirements.requires_structured_output:
            if "structured_output" not in rconf.capabilities:
                return False
        if requirements.requires_function_calling:
            if "function_calling" not in rconf.capabilities:
                return False

        # LoRA adapters not supported on remote deployments
        if requirements.lora_adapter_id:
            return False

        # Check preferred deployments
        if requirements.preferred_deployment_ids:
            deployment_name = rconf.get_deployment_name()
            if deployment_name not in requirements.preferred_deployment_ids:
                if requirements.fallback_deployment_ids:
                    if deployment_name not in requirements.fallback_deployment_ids:
                        return False

        return True

    def _score_and_select(
        self,
        candidates: list[LLMDeploymentConfig],
        requirements: LLMClientRequirements,
    ) -> LLMDeploymentConfig:
        """Score candidates and select best deployment.

        Args:
            candidates: List of candidate deployments
            requirements: Client requirements

        Returns:
            Best deployment configuration
        """
        # Simple scoring for now: prefer deployments in preferred list
        if requirements.preferred_deployment_ids:
            for dconf in candidates:
                if dconf.get_deployment_name() in requirements.preferred_deployment_ids:
                    return dconf

        # Otherwise return first candidate
        return candidates[0]


def get_routing_policy_class(policy_name: str) -> type[RequestRouter]:
    """Get routing policy class by name.

    Args:
        policy_name: Name of the routing policy (e.g., "ContextAwareRouter", "PageAffinityRouter", "RoundRobin")

    Returns:
        Routing policy class

    Raises:
        ValueError: If policy name is unknown
    """
    policies = {
        "ContextAwareRouter": ContextAwareRouter,
        "PageAffinityRouter": PageAffinityRouter,
        "RoundRobin": RequestRouter,  # Base class provides round-robin
    }

    if policy_name not in policies:
        raise ValueError(
            f"Unknown routing policy: {policy_name}. "
            f"Available policies: {list(policies.keys())}"
        )

    return policies[policy_name]

