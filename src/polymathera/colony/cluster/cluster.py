"""LLM cluster management.

This module provides the LLMCluster class for managing a cluster of vLLM deployments
with context-aware routing and distributed state management.
"""

from __future__ import annotations

import logging
from typing import Any

from ..distributed import get_polymathera
from ..distributed.state_management import StateManager
from ..distributed.ray_utils import serving
from ..distributed.hooks import tracing, hookable
from ..distributed.observability.models import SpanKind
from .config import ClusterConfig
from .models import (
    ClusterStatistics,
    InferenceRequest,
    InferenceResponse,
    LLMClientId,
    LLMClientState,
    LLMClusterState,
)
from ..vcm.models import VirtualContextPage, ContextPageId
logger = logging.getLogger(__name__)


@tracing(
    publish_key=lambda self: "deployment:llm_cluster",
    subscribe_key=lambda self: "deployment:llm_cluster",
)
@serving.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 1,  # Single cluster manager
        "target_queue_length": 10,
    },
)
class LLMCluster:
    """Manager for a cluster of vLLM deployments.

    The LLMCluster provides high-level management of vLLM instances deployed
    using colony.distributed.ray_utils.serving. It handles:

    1. **Cluster Deployment**: Deploy vLLM instances with custom configurations
    2. **Context Management**: Track which pages are loaded in which clients
    3. **Intelligent Routing**: Route requests based on page locality
    4. **Health Monitoring**: Monitor client health and performance
    5. **Statistics**: Collect and report cluster-wide metrics

    Architecture:
    - Uses serving.Application to deploy VLLMDeployment instances
    - Uses ContextAwareRouter for intelligent request routing
    - Uses distributed StateManager for cluster-wide state
    - Provides high-level APIs for inference and page management

    This layer sits between:
    - **Below**: colony.distributed.ray_utils.serving (deployment infrastructure)
    - **Above**: Virtual Context Manager (VCM) - to be implemented

    Example:
        ```python
        from polymathera.colony import LLMCluster

        # Create and deploy cluster
        cluster = LLMCluster(
            app_name="llm-cluster",
            model_name="meta-llama/Llama-3.1-8B",
            num_replicas=4,
            top_level=True,
        )
        await cluster.deploy()

        # Perform inference
        request = InferenceRequest(
            request_id="req-1",
            prompt="Explain quantum computing",
            context_page_ids=["page-1", "page-2"],
        )
        response = await cluster.infer(request)
        ```
    """

    def __init__(self, config: ClusterConfig, top_level: bool = False) -> None:
        """Initialize LLM cluster.

        Args:
            config: Complete cluster configuration

        Example:
            ```python
            from polymathera.colony.cluster import ClusterConfig, LLMDeploymentConfig

            # Create deployment configs from registry
            llama_8b = LLMDeploymentConfig.from_model_registry(
                model_name="meta-llama/Llama-3.1-8B",
                tensor_parallel_size=2,
                num_replicas=4,
            )
            llama_70b = LLMDeploymentConfig.from_model_registry(
                model_name="meta-llama/Llama-3.1-70B",
                tensor_parallel_size=4,
                num_replicas=2,
            )

            # Create cluster config
            config = ClusterConfig(
                app_name="llm-cluster",
                vllm_deployments=[llama_8b, llama_70b],
            )

            cluster = LLMCluster(config=config, top_level=True)
            ```
        """
        self.config: ClusterConfig = config
        self.top_level = top_level

        # Validate all deployment configurations
        for dconf in self.config.vllm_deployments:
            warnings = dconf.validate_against_registry()
            for warning in warnings:
                logger.warning(f"Deployment {dconf.get_deployment_name()}: {warning}")

        # Store app name
        self.app_name = self.config.app_name

        # Will be set during deployment
        self.app: serving.Application | None = None
        self.vllm_deployment_handles: dict[str, serving.DeploymentHandle] = {}
        self.remote_deployment_handles: dict[str, serving.DeploymentHandle] = {}
        self.embedding_deployment_handle: serving.DeploymentHandle | None = None
        self.state_manager: StateManager | None = None
        self.deployment_state_managers: dict[str, StateManager] = {}

    @serving.initialize_deployment
    async def initialize(self):
        """Initialize LLMCluster self-contained state (top_level=False).

        Sets up state managers and local resources. Cross-deployment handle
        discovery is deferred to on_ready() via @on_app_ready, which runs
        after all sibling deployments have been started.
        """
        if self.top_level:
            # When top_level=True, deploy() handles rest of initialization
            return

        # Get app name from environment
        self.app_name = serving.get_my_app_name()
        logger.info(f"Initializing LLMCluster deployment for app '{self.app_name}'")

        # Initialize state managers (self-contained — no cross-deployment calls)
        polymathera = get_polymathera()
        cluster_state_key = LLMClusterState.get_state_key(self.app_name)
        self.state_manager = await polymathera.get_state_manager(
            state_type=LLMClusterState,
            state_key=cluster_state_key,
        )

        # Initialize distributed tracing
        import os
        tracing_enabled = os.environ.get("TRACING_ENABLED", "").lower() in ("true", "1", "yes")
        if tracing_enabled:
            from ..distributed.observability import TracingConfig
            from .observability import ClusterTracingFacility
            self._tracing_facility = ClusterTracingFacility(
                config=TracingConfig(enabled=True),
                owner=self,
                service_name="LLMCluster",
                deployment_name="llm_cluster",
                pointcuts=[("*.infer", SpanKind.INFER)],
            )
            await self._tracing_facility.initialize()

        logger.info("LLMCluster deployment initialized (awaiting app ready for handle discovery)")

    @serving.on_app_ready
    async def on_ready(self):
        """Discover sibling deployment handles after all deployments are started.

        This runs after every deployment in the application has been started,
        so it is safe to call serving.get_deployment() for sibling deployments.
        """
        if self.top_level:
            return

        logger.info("LLMCluster app ready — discovering deployment handles")
        await self._discover_deployment_handles()

        logger.info(
            f"LLMCluster handle discovery complete: "
            f"{len(self.vllm_deployment_handles)} vLLM, "
            f"{len(self.remote_deployment_handles)} remote deployments"
        )

    async def cleanup_state_managers(self) -> None:
        """Cleanup all deployment state managers."""
        # Cleanup state managers for cluster and all deployments
        # Cluster-level state
        cluster_state_key = LLMClusterState.get_state_key(self.app_name)
        logger.info(f"Cleaning up cluster state: {cluster_state_key}")
        try:
            await self.state_manager.cleanup()
            logger.info(f"Cleaned up cluster state: {cluster_state_key}")
        except Exception as e:
            logger.warning(f"Failed to cleanup cluster state {cluster_state_key}: {e}")

        for deployment_name, deployment_state_manager in self.deployment_state_managers.items():
            logger.info(f"Cleaning up deployment state: {deployment_state_manager.state_key}")
            try:
                await deployment_state_manager.cleanup()
                logger.info(f"Cleaned up deployment state: {deployment_state_manager.state_key}")
            except Exception as e:
                logger.warning(f"Failed to cleanup deployment state {deployment_state_manager.state_key}: {e}")

    async def deploy(self) -> None:
        """Deploy the LLM cluster.

        This creates a serving.Application with multiple VLLMDeployment instances
        (one per configured model) and starts them with their configured routing policies.
        All deployment parameters are derived from the cluster configuration.
        """
        if not self.top_level:
            raise RuntimeError("LLMCluster.deploy() can only be called on top-level clusters")

        logger.info(
            f"Deploying LLM cluster '{self.app_name}' with {len(self.config.vllm_deployments)} "
            f"vLLM deployment(s)"
        )

        # Initialize distributed state managers
        polymathera = get_polymathera()

        # Cluster-level state manager
        if self.state_manager is None:
            cluster_state_key = LLMClusterState.get_state_key(self.app_name)
            self.state_manager = await polymathera.get_state_manager(
                state_type=LLMClusterState,
                state_key=cluster_state_key,
            )

        # Cleanup existing deployments and states if requested
        if self.config.cleanup_on_init:
            logger.info(f"Cleanup requested: removing existing application '{self.app_name}' and its states")

            await serving.Application.cleanup(self.app_name)

            await self.cleanup_state_managers()

            logger.info(f"Cleanup complete for application '{self.app_name}'")

        # Create application
        self.app = serving.Application(name=self.app_name)

        # Use config to add deployments (top_level=True means don't deploy LLMCluster itself)
        self.config.add_deployments_to_app(self.app, top_level=True)

        # Start application
        await self.app.start()

        await self._discover_deployment_handles()

        logger.info(
            f"LLM cluster '{self.app_name}' deployed successfully with "
            f"{len(self.vllm_deployment_handles)} vLLM deployment(s)"
        )

    async def _discover_deployment_handles(self) -> None:
        """Discover all LLM deployment handles and initialize state managers.

        Called by:
        - deploy() when top_level=True (after manual deployment)
        - on_ready() when top_level=False (after automatic deployment)
        """
        # NOTE: Imported here (not at module level) to break circular import chain
        from ..system import get_embedding_deployment, get_vllm_deployment, get_remote_llm_deployment
        from ..utils.retry import create_retry_with_logging

        # Get deployment handles for all vLLM deployments
        for dconf in self.config.vllm_deployments:
            deployment_name = dconf.get_deployment_name()
            self.vllm_deployment_handles[deployment_name] = get_vllm_deployment(
                deployment_name,
                self.app_name,
            )
            logger.debug(f"Connected to VLLM deployment: {deployment_name}")

        # Get deployment handles for all remote deployments.
        # Retry because remote deployments may still be starting up.
        # The retry decorator is applied to a local function (not a class method)
        # because tenacity captures _thread._local which Ray can't serialize.
        @create_retry_with_logging(logger, stop_attempts=30, wait_min=2, wait_max=2)
        async def _get_remote_handle(deployment_name: str):
            return get_remote_llm_deployment(deployment_name, self.app_name)

        for rconf in self.config.remote_deployments:
            deployment_name = rconf.get_deployment_name()
            self.remote_deployment_handles[deployment_name] = await _get_remote_handle(
                deployment_name
            )
            logger.debug(f"Connected to remote deployment: {deployment_name}")

        # Get embedding deployment handle if any embedding backend is configured (GPU, API, or SentenceTransformers)
        if self.config.embedding_config or self.config.remote_embedding_config or self.config.st_embedding_config:
            self.embedding_deployment_handle = get_embedding_deployment(self.app_name)
            logger.debug("Connected to embedding deployment")

        # Per-deployment state managers (for reading deployment states)
        await self._initialize_deployment_state_managers()

    async def _initialize_deployment_state_managers(self) -> None:
        # Per-deployment state managers (for reading deployment states)
        total_deployments = len(self.config.vllm_deployments) + len(self.config.remote_deployments)
        if len(self.deployment_state_managers) > 0:
            if len(self.deployment_state_managers) != total_deployments:
                raise RuntimeError("Deployment state managers must match configured deployments")
            return  # Already initialized

        from .models import VLLMDeploymentState
        # Initialize distributed state managers
        polymathera = get_polymathera()

        self.deployment_state_managers = {}
        # vLLM deployments
        for dconf in self.config.vllm_deployments:
            deployment_name = dconf.get_deployment_name()
            deployment_state_manager = await polymathera.get_state_manager(
                state_type=VLLMDeploymentState,
                state_key=VLLMDeploymentState.get_state_key(self.app_name, deployment_name),
            )
            self.deployment_state_managers[deployment_name] = deployment_state_manager

        # Remote deployments (reuse VLLMDeploymentState — same structure)
        for rconf in self.config.remote_deployments:
            deployment_name = rconf.get_deployment_name()
            deployment_state_manager = await polymathera.get_state_manager(
                state_type=VLLMDeploymentState,
                state_key=VLLMDeploymentState.get_state_key(self.app_name, deployment_name),
            )
            self.deployment_state_managers[deployment_name] = deployment_state_manager

    async def shutdown(self) -> None:
        """Shutdown the LLM cluster.

        This stops the serving application and cleans up resources.
        """
        logger.info(f"Shutting down LLM cluster '{self.app_name}'")
        if self.app:
            if self.top_level:
                await self.app.stop()
            self.app = None
        self.vllm_deployment_handles = {}
        self.remote_deployment_handles = {}
        self.embedding_deployment_handle = None
        logger.info(f"LLM cluster '{self.app_name}' shut down successfully")

    @serving.endpoint
    @hookable
    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Perform inference using the cluster.

        The request is routed to an appropriate vLLM deployment based on requirements,
        then to a specific replica based on context-aware routing.

        Args:
            request: Inference request with optional requirements

        Returns:
            Inference response

        Raises:
            RuntimeError: If cluster is not deployed
            ValueError: If no deployment matches requirements or tenant validation fails
        """
        self._check_initialized()

        all_handles = {**self.vllm_deployment_handles, **self.remote_deployment_handles}
        if not all_handles:
            raise RuntimeError("Cluster not deployed. Call deploy() first.")

        # Validate tenant access if multi-tenancy is enabled
        if self.config.enable_tenant_isolation:
            await self._validate_tenant_access(request)

        # Select deployment based on requirements
        deployment_name = self._select_deployment_for_request(request)
        deployment_handle = all_handles[deployment_name]

        logger.debug(
            f"Routing request {request.request_id} to deployment '{deployment_name}' "
            f"(syscontext: {request.syscontext.to_dict()})"
        )

        # Route and execute inference
        response = await deployment_handle.infer(request)

        # Update cluster state
        await self._update_cluster_state(request, response)

        return response

    @serving.endpoint
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using the cluster's embedding deployment.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (one per input text)

        Raises:
            RuntimeError: If cluster is not deployed or no embedding deployment configured
        """
        self._check_initialized()

        if not self.embedding_deployment_handle:
            raise RuntimeError(
                "No embedding deployment configured. "
                "Provide embedding_config when creating the cluster."
            )

        return await self.embedding_deployment_handle.embed(texts)

    @serving.endpoint
    async def load_page(
        self,
        page: VirtualContextPage,
        deployment_name: str | None = None,
        client_id: str | None = None,
    ) -> bool:
        """Load a context page into the cluster.

        The page will be loaded into one or more deployments. If client_id is specified,
        the page is loaded on that specific replica; otherwise it goes to any replica.

        Args:
            page: Context page to load
            deployment_name: Specific deployment to load into (if None, loads into all deployments)
            client_id: Specific client/replica to load onto (requires deployment_name)

        Returns:
            True if page was loaded successfully

        Raises:
            RuntimeError: If cluster is not deployed
            ValueError: If tenant validation fails or client_id specified without deployment_name
        """
        self._check_initialized()

        all_handles = {**self.vllm_deployment_handles, **self.remote_deployment_handles}
        if not all_handles:
            raise RuntimeError("Cluster not deployed. Call deploy() first.")

        # Validate tenant access
        if self.config.enable_tenant_isolation:
            if page.syscontext.tenant_id not in page.allowed_tenant_ids:
                # Auto-add page creator to allowed tenants
                page.allowed_tenant_ids.add(page.syscontext.tenant_id)

        # If client_id specified, deployment_name is required
        if client_id and not deployment_name:
            raise ValueError("client_id requires deployment_name to be specified")

        # Determine which deployments to load into
        if deployment_name:
            if deployment_name not in all_handles:
                raise ValueError(f"Deployment '{deployment_name}' not found in cluster")
            deployment_handles = {deployment_name: all_handles[deployment_name]}
        else:
            # Load into all deployments for maximum availability
            deployment_handles = all_handles

        # Load page into selected deployment(s)
        # Each deployment's replicas will track internally which specific replica has the page
        success = False
        for dep_name, handle in deployment_handles.items():
            try:
                if client_id:
                    # Load on specific client - pass target_client_id for router
                    result = await handle.load_page(page, target_client_id=client_id)
                else:
                    # Load on any replica
                    result = await handle.load_page(page)

                if result:
                    success = True
                    if client_id:
                        logger.debug(f"Loaded page {page.page_id} onto {dep_name}/{client_id}")
                    else:
                        logger.debug(f"Loaded page {page.page_id} into deployment '{dep_name}'")
            except Exception as e:
                logger.warning(f"Failed to load page {page.page_id} into deployment '{dep_name}': {e}")

        return success

    @serving.endpoint
    async def evict_page(
        self,
        page_id: ContextPageId,
        deployment_name: str | None = None,
    ) -> bool:
        """Evict a context page from the cluster.

        The page will be evicted from all replicas across specified deployment(s).

        Args:
            page_id: ID of page to evict
            deployment_name: Specific deployment to evict from (if None, evicts from all deployments)

        Returns:
            True if page was evicted successfully from at least one deployment

        Raises:
            RuntimeError: If cluster is not deployed
        """
        self._check_initialized()

        all_handles = {**self.vllm_deployment_handles, **self.remote_deployment_handles}
        if not all_handles:
            raise RuntimeError("Cluster not deployed. Call deploy() first.")

        # Determine which deployments to evict from
        if deployment_name:
            deployment_handles = {deployment_name: all_handles[deployment_name]}
        else:
            # Evict from all deployments
            deployment_handles = all_handles

        syscontext = serving.require_execution_context()

        # Evict page from selected deployment(s)
        success = False
        for dep_name, handle in deployment_handles.items():
            try:
                result = await handle.evict_page(page_id)
                if result:
                    success = True
                    logger.debug(f"Evicted page {page_id}: syscontext: {syscontext.to_dict()} from deployment '{dep_name}'")
            except Exception as e:
                logger.warning(f"Failed to evict page {page_id}: syscontext: {syscontext.to_dict()} from deployment '{dep_name}': {e}")

        # Note: Page tracking is internal to each deployment's replicas
        # The ContextAwareRouter queries replica state directly

        return success

    @serving.endpoint
    async def get_statistics(self) -> ClusterStatistics:
        """Get cluster-wide statistics by aggregating from all deployments.

        Returns:
            Cluster statistics including health, capacity, and performance metrics
        """
        self._check_initialized()

        # Aggregate statistics from all deployment states
        total_clients = 0
        healthy_clients = 0
        total_kv_capacity = 0
        total_kv_used = 0
        total_pages = set()  # Use set to deduplicate pages across deployments
        total_requests = 0
        total_errors = 0
        total_page_faults = 0

        # Query each deployment's state
        # TODO: Parallelize this loop
        for deployment_name, state_manager in self.deployment_state_managers.items():
            async for dep_state in state_manager.read_transaction():
                # Aggregate client stats
                for client_state in dep_state.client_states.values():
                    total_clients += 1
                    if client_state.is_healthy:
                        healthy_clients += 1
                    total_kv_capacity += client_state.kv_cache_capacity
                    total_kv_used += client_state.kv_cache_used

                # Aggregate deployment metrics
                total_requests += dep_state.total_requests
                total_errors += dep_state.total_errors
                total_page_faults += dep_state.total_page_faults

                # Track unique pages (page may be in multiple deployments)
                total_pages.update(dep_state.page_index.keys())

        # Get cluster-level stats
        async for cluster_state in self.state_manager.read_transaction():
            # Cluster state tracks cross-deployment requests
            if cluster_state.total_requests > total_requests:
                total_requests = cluster_state.total_requests
            if cluster_state.total_errors > total_errors:
                total_errors = cluster_state.total_errors

        # Calculate metrics
        avg_latency = 0.0  # TODO: Track latency properly
        page_hit_rate = (
            1.0 - (total_page_faults / total_requests)
            if total_requests > 0
            else 0.0
        )

        return ClusterStatistics(
            total_clients=total_clients,
            healthy_clients=healthy_clients,
            total_kv_cache_capacity=total_kv_capacity,
            total_kv_cache_used=total_kv_used,
            total_pages_loaded=len(total_pages),
            total_requests=total_requests,
            total_errors=total_errors,
            average_latency_ms=avg_latency,
            page_hit_rate=page_hit_rate,
        )

    @serving.endpoint(ring=serving.Ring.KERNEL)
    async def get_all_deployment_names(self) -> list[str]:
        """Get names of all deployments (vLLM + remote) in the cluster.

        Returns:
            List of deployment names
        """
        self._check_initialized()

        handles = list(self.vllm_deployment_handles.keys()) + list(self.remote_deployment_handles.keys())
        if not self.top_level and not handles:
            raise RuntimeError("Cluster's @on_app_ready endpoint must be called to discover deployment handles.")

        return handles

    @serving.endpoint
    async def get_all_client_states(self) -> dict[LLMClientId, LLMClientState]:
        """Get client states from all deployments.

        Returns a flat mapping from client_id to LLMClientState, aggregated
        across all deployments (vLLM + remote). Each deployment replica
        registers exactly one client, so client_id uniquely identifies a
        replica.

        Used by VCM's allocation strategy which expects
        ``dict[client_id, LLMClientState]``.

        Returns:
            Flat dict: {client_id: LLMClientState}
        """
        self._check_initialized()

        all_client_states: dict[LLMClientId, LLMClientState] = {}

        for state_manager in self.deployment_state_managers.values():
            async for dep_state in state_manager.read_transaction():
                all_client_states.update(dep_state.client_states)

        return all_client_states

    async def _update_cluster_state(
        self,
        request: InferenceRequest,
        response: InferenceResponse,
    ) -> None:
        """Update cluster state after an inference request.

        Args:
            request: The inference request
            response: The inference response
        """
        async for state in self.state_manager.write_transaction():
            state.total_requests += 1
            if response.page_faults:
                state.total_page_faults += len(response.page_faults)

    def _check_initialized(self):
        if self.top_level and not self.app:
            raise RuntimeError("Cluster not deployed. Call deploy() first.")

        if not self.state_manager:
            raise RuntimeError("Cluster not initialized. State manager not available.")

    @serving.endpoint(ring=serving.Ring.KERNEL)
    async def select_deployment(
        self,
        requirements: Any | None = None,  # LLMClientRequirements - avoid circular import
    ) -> tuple[str, str]:
        """Select an LLM deployment based on requirements.

        Public endpoint for other components (e.g., AgentSystem) to select
        which deployment to use based on LLMClientRequirements.

        Args:
            requirements: LLMClientRequirements or None

        Returns:
            Tuple of (deployment_name, deployment_kind) where kind is
            ``"vllm"`` or ``"remote"``.

        Raises:
            ValueError: If no deployment matches requirements

        Example:
            ```python
            deployment_name, deployment_kind = await llm_cluster.select_deployment(
                LLMClientRequirements(
                    model_family="llama",
                    min_context_window=32000,
                )
            )
            ```
        """
        self._check_initialized()

        from .routing import RequirementBasedRouter

        # Create router for deployment selection
        router = RequirementBasedRouter(
            vllm_deployment_configs=self.config.vllm_deployments,
            remote_deployment_configs=self.config.remote_deployments,
            enable_fallbacks=self.config.enable_fallbacks,
        )

        # Select deployment based on requirements
        deployment_name = router.select_deployment(requirements)

        # Determine kind from config
        remote_names = {r.get_deployment_name() for r in self.config.remote_deployments}
        kind = "remote" if deployment_name in remote_names else "vllm"

        return deployment_name, kind

    def _select_deployment_for_request(self, request: InferenceRequest) -> str:
        """Select deployment for an inference request based on requirements.

        Args:
            request: Inference request with optional requirements

        Returns:
            Deployment name to use

        Raises:
            ValueError: If no deployment matches requirements
        """
        from .routing import RequirementBasedRouter

        # Create router for deployment selection (includes both vLLM and remote)
        router = RequirementBasedRouter(
            vllm_deployment_configs=self.config.vllm_deployments,
            remote_deployment_configs=self.config.remote_deployments,
            enable_fallbacks=self.config.enable_fallbacks,
        )

        # Select deployment based on requirements
        return router.select_deployment(request.requirements)

    async def _validate_tenant_access(self, request: InferenceRequest) -> None:
        """Validate tenant access for pages referenced in request.

        Args:
            request: Inference request

        Raises:
            ValueError: If tenant doesn't have access to required pages
        """
        if not request.context_page_ids:
            # No pages to validate
            return

        # Check each referenced page for tenant access
        # For now, this is a stub - full implementation requires loading page metadata
        # TODO: Implement full tenant validation by loading page metadata and checking allowed_tenant_ids
        logger.debug(
            f"Tenant validation for syscontext {request.syscontext.to_dict()}: "
            f"{len(request.context_page_ids)} pages (validation stub)"
        )
