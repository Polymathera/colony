"""Deployment proxy actor for managing replicas and routing requests."""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Type

import ray
from prometheus_client import Counter, Histogram

from .autoscaler import DeploymentAutoscaler
from .health import DeploymentHealthMonitor
from .models import (
    AutoscalingConfig,
    DeploymentReplicaInfo,
    DeploymentRequest,
    DeploymentResponse,
    LoggingConfig,
    ApplicationRegistry,
    RoutingHints,
    LeastLoadedRouter,
    RequestRouter,
    RoundRobinRouter
)

logger = logging.getLogger(__name__)

# Metrics
request_total = Counter(
    "deployment_request_total",
    "Total number of requests to deployment",
    ["deployment_name", "method", "status"],
)
request_duration = Histogram(
    "deployment_request_duration_seconds",
    "Duration of deployment requests",
    ["deployment_name", "method"],
)


class DeploymentProxyRayActor:
    """Proxy actor that manages deployment replicas and routes requests.

    This actor serves as the ingress point for a deployment, handling:
    - Request routing across replicas
    - Health monitoring and replica management
    - Autoscaling based on load
    - Fault tolerance and error handling
    """

    def __init__(
        self,
        deployment_name: str,
        deployment_class: Type[Any],
        app_name: str,
        deployment_init_args: tuple[Any, ...] | None = None,
        deployment_init_kwargs: dict[str, Any] | None = None,
        default_router_class: Type[RequestRouter] | None = None,
        autoscaling_config: AutoscalingConfig | None = None,
        ray_actor_options: dict[str, Any] | None = None,
        logging_config: LoggingConfig | None = None,
    ):
        """Initialize the deployment proxy.

        Args:
            deployment_name: Name of this deployment.
            deployment_class: The wrapped deployment class to instantiate as replicas.
            app_name: Name of the application this deployment belongs to.
            deployment_init_args: Positional arguments for deployment __init__.
            deployment_init_kwargs: Keyword arguments for deployment __init__.
            default_router_class: Custom routing policy class (defaults to LeastLoadedRouter).
            autoscaling_config: Autoscaling configuration.
            ray_actor_options: Ray actor options for replicas.
            logging_config: Logging configuration for proxy and replicas.
        """
        # Configure logging from LoggingConfig or fall back to environment variable
        if logging_config:
            logging_config.apply()
        else:
            log_level = os.environ.get("POLYMATHERA_LOG_LEVEL", "INFO").upper()
            logging.basicConfig(
                level=getattr(logging, log_level, logging.INFO),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                force=True,
            )

        self.deployment_name = deployment_name
        self.deployment_class = deployment_class
        self.app_name = app_name
        self.deployment_init_args = deployment_init_args or ()
        self.deployment_init_kwargs = deployment_init_kwargs or {}
        self.ray_actor_options = ray_actor_options or {}
        self.autoscaling_config = autoscaling_config or AutoscalingConfig()
        self.logging_config = logging_config

        # Initialize default router and registry for specialized routers
        default_router_class = default_router_class or LeastLoadedRouter
        self.default_router: RequestRouter = default_router_class()

        # Router registry for specialized routing policies
        # Maps router class name -> RequestRouter instance
        self.specialized_routers: dict[str, RequestRouter] = {
            RoundRobinRouter.__qualname__: RoundRobinRouter(),
            LeastLoadedRouter.__qualname__: LeastLoadedRouter(),
        }

        # Replica management
        self.replicas: list[DeploymentReplicaInfo] = []
        self._replica_lock = asyncio.Lock()

        # Per-replica queues for request queueing when max_concurrency is reached
        # Maps replica_id -> asyncio.Queue[DeploymentRequest]
        self._replica_queues: dict[str, asyncio.Queue[DeploymentRequest]] = {}

        # Track completion events for requests in flight
        # Maps request_id -> asyncio.Event
        self._request_completion_events: dict[str, asyncio.Event] = {}

        # Per-replica periodic health check tasks
        # Maps replica_id -> list[asyncio.Task]
        self._replica_health_check_tasks: dict[str, list[asyncio.Task]] = {}

        # Initialize health monitor
        self.health_monitor = DeploymentHealthMonitor(deployment_name)

        # Initialize autoscaler
        self.autoscaler = DeploymentAutoscaler(
            deployment_name=deployment_name,
            config=self.autoscaling_config,
            scale_callback=self._scale_to_target,
        )

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []

        logger.info(f"Initialized deployment proxy for {deployment_name}")

    async def initialize(self) -> None:
        """Initialize the deployment with minimum replicas and start monitoring."""
        # Create initial replicas
        logger.info(
            f"Initializing deployment {self.deployment_name} with "
            f"{self.autoscaling_config.min_replicas} replicas"
        )
        await self._scale_to_target(self.autoscaling_config.min_replicas)
        logger.info(f"Deployment {self.deployment_name} scaled to minimum replicas")

        # Start health monitoring
        logger.info(f"Starting health monitor for deployment {self.deployment_name}")
        await self.health_monitor.start(self.replicas)
        logger.info(f"Health monitor started for deployment {self.deployment_name}")

        # Start autoscaling
        await self.autoscaler.start()

        # Start background autoscaling check loop
        self._background_tasks.append(
            asyncio.create_task(self._autoscaling_check_loop())
        )

        logger.info(
            f"Deployment {self.deployment_name} initialized with "
            f"{len(self.replicas)} replicas"
        )

    def _get_router(self, request: DeploymentRequest) -> RequestRouter:
        """Select router based on routing hints.

        This method selects the appropriate router based on the routing policy
        specified in the routing hints. Specialized routers are lazily initialized
        on first use. If router_kwargs are provided in hints, a new instance is
        created with those kwargs.

        Args:
            request: The DeploymentRequest being routed.

        Returns:
            RequestRouter instance to use for routing this request.

        Example:
            ```python
            # Request with context-aware routing
            req.routing_hints = RoutingHints(router_class=ContextAwareRouter, metadata={"context_page_ids": ["page-1"]})
            router = self._get_router(req)  # Returns ContextAwareRouter

            # Request with custom router kwargs
            req.routing_hints = RoutingHints(
                router_class=TargetClientRouter,
                router_kwargs={"strip_routing_params": ["target_client_id"]}
            )
            router = self._get_router(req)  # Returns TargetClientRouter configured with kwargs

            # Request with no hints
            router = self._get_router(None)  # Returns default_router
            ```
        """
        method_name: str = request.method_name
        routing_hints: RoutingHints | None = request.routing_hints
        # No hints or no router class - use default
        if not routing_hints or not routing_hints.router_class:
            return self.default_router

        router_class = routing_hints.router_class
        router_kwargs = routing_hints.router_kwargs or {}

        # Check if we have this router in registry
        key = f"{router_class.__qualname__}:{method_name}"
        if key in self.specialized_routers:
            return self.specialized_routers[key]

        # Lazy initialization for LLM-specific routers
        logger.debug(f"Creating {router_class} with kwargs {router_kwargs} for "
                     f"{self.deployment_name}, endpoint {method_name}")
        try:
            router = router_class(**router_kwargs)
            self.specialized_routers[key] = router
            return router
        except Exception as e:
            logger.warning(
                f"Could not instantiate {router_class} for {self.deployment_name}, endpoint {method_name}: {e}. "
                f"Falling back to default router."
            )
            return self.default_router

    async def _autoscaling_check_loop(self) -> None:
        """Background task that periodically checks autoscaling."""
        while True:
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds

                async with self._replica_lock:
                    target_replicas = await self.autoscaler.check_and_scale(self.replicas)
                    if target_replicas is not None:
                        await self._scale_to_target(target_replicas)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in autoscaling check loop: {e}", exc_info=True)

    async def handle_request(self, request: DeploymentRequest) -> DeploymentResponse:
        """Handle an incoming request by routing it to a replica.

        This method implements request queueing when max_concurrency is exceeded.

        Args:
            request: The request to handle.

        Returns:
            Response from the replica.
        """
        method_name = request.method_name

        try:
            # Get healthy replicas
            async with self._replica_lock:
                healthy_replicas = [r for r in self.replicas if r.is_healthy]

            if not healthy_replicas:
                error_msg = f"No healthy replicas available for {self.deployment_name}"
                logger.error(error_msg)
                request_total.labels(
                    deployment_name=self.deployment_name,
                    method=method_name,
                    status="no_replicas",
                ).inc()
                return DeploymentResponse.error(
                    request_id=request.request_id,
                    error=Exception(error_msg),
                )

            # Select router based on routing hints
            router = self._get_router(request)

            # Route request to a replica
            replica = await router.route_request(request, healthy_replicas)

            # Add request to replica's queue
            # The queue processor will handle it
            replica.queue_length += 1
            await self._replica_queues[replica.replica_id].put(request)

            # Wait for the request to complete
            completion_event = asyncio.Event()
            self._request_completion_events[request.request_id] = completion_event
            await completion_event.wait()

            # Get the response (stored in request metadata by processor)
            response = request.metadata.get("__response__")
            if response is None:
                raise RuntimeError(f"No response found for request {request.request_id}")

            return response

        except Exception as e:
            logger.error(
                f"Error handling request to {self.deployment_name}.{method_name}: {e}",
                exc_info=True,
            )
            request_total.labels(
                deployment_name=self.deployment_name,
                method=method_name,
                status="error",
            ).inc()
            return DeploymentResponse.error(request_id=request.request_id, error=e)

    async def _process_replica_queue(self, replica: DeploymentReplicaInfo) -> None:
        """Process requests from a replica's queue.

        This background task pulls requests from the replica's queue and executes them
        concurrently, respecting the max_concurrency limit.

        Args:
            replica: The replica to process requests for.
        """
        replica_id = replica.replica_id
        queue = self._replica_queues[replica_id]
        max_concurrency = self.autoscaling_config.max_concurrency

        while True:
            try:
                # Get next request from queue
                request = await queue.get()

                # Decrement queue length
                replica.queue_length -= 1

                # Wait if we've reached max concurrency
                if max_concurrency is not None:
                    while replica.in_flight_requests >= max_concurrency:
                        await asyncio.sleep(0.01)  # Small delay before checking again

                # Increment in-flight count
                replica.in_flight_requests += 1

                # Process the request concurrently (don't await here)
                asyncio.create_task(
                    self._execute_request_on_replica(replica, request)
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Error in queue processor for replica {replica_id}: {e}",
                    exc_info=True,
                )

    async def _execute_request_on_replica(
        self,
        replica: DeploymentReplicaInfo,
        request: DeploymentRequest,
    ) -> None:
        """Execute a single request on a replica.

        Args:
            replica: The replica to execute on.
            request: The request to execute.
        """
        try:
            start_time = time.time()
            logger.debug(
                f"Executing request {request.request_id} on replica {replica.replica_id}"
            )
            # Call the replica's actor method
            response = await replica.actor_handle.__handle_request__.remote(request)
            logger.debug(
                f"Completed request {request.request_id} on replica {replica.replica_id}"
            )

            # Record metrics
            duration = time.time() - start_time
            request_duration.labels(
                deployment_name=self.deployment_name,
                method=request.method_name,
            ).observe(duration)
            request_total.labels(
                deployment_name=self.deployment_name,
                method=request.method_name,
                status="success",
            ).inc()

        except Exception as e:
            logger.error(
                f"Error processing request {request.request_id} on replica {replica.replica_id}: {e}",
                exc_info=True,
            )
            response = DeploymentResponse.error(
                request_id=request.request_id,
                error=e,
            )
            request_total.labels(
                deployment_name=self.deployment_name,
                method=request.method_name,
                status="error",
            ).inc()

        finally:
            # Decrement in-flight count
            replica.in_flight_requests -= 1

        # Store response in request metadata
        request.metadata["__response__"] = response

        # Signal completion
        completion_event = self._request_completion_events.pop(request.request_id, None)
        if completion_event:
            completion_event.set()

    async def _call_lifecycle_hooks(self, actor_handle: Any) -> None:
        """Call lifecycle hooks on a replica after creation.

        This discovers and calls methods decorated with @initialize_deployment
        and @post_initialize_deployment on the replica.

        Args:
            actor_handle: The Ray actor handle for the replica.
        """
        # Discover initialization methods from the deployment class
        init_methods = []

        for attr_name in dir(self.deployment_class):
            try:
                attr = getattr(self.deployment_class, attr_name)
                if callable(attr) and getattr(attr, "__initialize_deployment__", False):
                    init_methods.append(attr_name)
            except AttributeError:
                pass

        # Call each initialization method on the replica
        for method_name in init_methods:
            try:
                logger.info(f"Calling {method_name} on replica {actor_handle}")
                method = getattr(actor_handle, method_name)
                await method.remote()
                logger.info(f"Completed {method_name} on replica")
            except Exception as e:
                logger.error(
                    f"Error calling lifecycle hook {method_name} on replica: {e}",
                    exc_info=True
                )
                raise  # Re-raise to fail replica creation if initialization fails

    async def _call_cleanup_hooks(self, actor_handle: Any) -> None:
        """Call cleanup hooks on a replica before destruction.

        This discovers and calls methods decorated with @cleanup_deployment
        on the replica.

        Args:
            actor_handle: The Ray actor handle for the replica.
        """
        # Discover cleanup methods from the deployment class
        cleanup_methods = []

        for attr_name in dir(self.deployment_class):
            try:
                attr = getattr(self.deployment_class, attr_name)
                if callable(attr) and getattr(attr, "__cleanup_deployment__", False):
                    cleanup_methods.append(attr_name)
            except AttributeError:
                pass

        # Call each cleanup method on the replica
        for method_name in cleanup_methods:
            try:
                logger.info(f"Calling {method_name} on replica {actor_handle}")
                method = getattr(actor_handle, method_name)
                await method.remote()
                logger.info(f"Completed {method_name} on replica")
            except Exception as e:
                logger.error(
                    f"Error calling cleanup hook {method_name} on replica: {e}",
                    exc_info=True
                )
                # Don't re-raise - we still want to kill the replica even if cleanup fails

    def _discover_periodic_health_checks(self) -> list[tuple[str, float]]:
        """Discover methods decorated with @periodic_health_check.

        Returns:
            List of tuples (method_name, interval_s) for each periodic health check method.
        """
        health_check_methods = []

        for attr_name in dir(self.deployment_class):
            try:
                attr = getattr(self.deployment_class, attr_name)
                if callable(attr) and getattr(attr, "__periodic_health_check__", False):
                    interval_s = getattr(attr, "__health_check_interval_s__", 30.0)
                    health_check_methods.append((attr_name, interval_s))
            except AttributeError:
                pass

        return health_check_methods

    async def _run_periodic_health_check(
        self,
        replica: DeploymentReplicaInfo,
        method_name: str,
        interval_s: float
    ) -> None:
        """Run a periodic health check method on a replica.

        This is a background task that loops forever, calling the health check
        method at regular intervals.

        Args:
            replica: The replica to run the health check on.
            method_name: Name of the health check method to call.
            interval_s: Interval in seconds between health checks.
        """
        replica_id = replica.replica_id
        logger.info(
            f"Starting periodic health check '{method_name}' for replica {replica_id} "
            f"with interval {interval_s}s"
        )

        while True:
            try:
                # Wait for the interval
                await asyncio.sleep(interval_s)

                # Call the health check method on the replica
                logger.debug(
                    f"Calling periodic health check '{method_name}' on replica {replica_id}"
                )
                method = getattr(replica.actor_handle, method_name)
                await method.remote()
                logger.debug(
                    f"Completed periodic health check '{method_name}' on replica {replica_id}"
                )

            except asyncio.CancelledError:
                # Task is being cancelled (replica removal or shutdown)
                logger.info(
                    f"Periodic health check '{method_name}' cancelled for replica {replica_id}"
                )
                break

            except Exception as e:
                # Log error but continue the loop
                logger.error(
                    f"Error in periodic health check '{method_name}' for replica {replica_id}: {e}",
                    exc_info=True
                )
                # Continue looping - errors shouldn't stop periodic health checks

    def _start_periodic_health_checks(self, replica: DeploymentReplicaInfo) -> None:
        """Start periodic health check tasks for a replica.

        Args:
            replica: The replica to start health checks for.
        """
        replica_id = replica.replica_id

        # Discover periodic health check methods
        health_check_methods = self._discover_periodic_health_checks()

        if not health_check_methods:
            logger.debug(f"No periodic health checks defined for {self.deployment_name}")
            return

        logger.info(
            f"Starting {len(health_check_methods)} periodic health check(s) for replica {replica_id}"
        )

        # Create a task for each health check method
        tasks = []
        for method_name, interval_s in health_check_methods:
            task = asyncio.create_task(
                self._run_periodic_health_check(replica, method_name, interval_s)
            )
            tasks.append(task)

        # Track the tasks
        self._replica_health_check_tasks[replica_id] = tasks

    async def _stop_periodic_health_checks(self, replica_id: str) -> None:
        """Stop periodic health check tasks for a replica.

        Args:
            replica_id: The replica ID to stop health checks for.
        """
        tasks = self._replica_health_check_tasks.pop(replica_id, [])

        if not tasks:
            return

        logger.info(f"Stopping {len(tasks)} periodic health check task(s) for replica {replica_id}")

        # Cancel all tasks
        for task in tasks:
            task.cancel()

        # Wait for all tasks to finish
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _scale_to_target(self, target_count: int) -> None:
        """Scale deployment to target replica count.

        Args:
            target_count: Desired number of replicas.
        """
        async with self._replica_lock:
            current_count = len(self.replicas)

            if target_count == current_count:
                return

            if target_count > current_count:
                # Scale up: add replicas
                to_add = target_count - current_count
                logger.info(
                    f"Scaling up {self.deployment_name} by {to_add} replicas "
                    f"({current_count} -> {target_count})"
                )
                await self._add_replicas(to_add)

            else:
                # Scale down: remove replicas
                to_remove = current_count - target_count
                logger.info(
                    f"Scaling down {self.deployment_name} by {to_remove} replicas "
                    f"({current_count} -> {target_count})"
                )
                await self._remove_replicas(to_remove)

    async def _add_replicas(self, count: int) -> None:
        """Add new replicas to the deployment.

        Args:
            count: Number of replicas to add.
        """
        logger.info(f"Adding {count} replicas to {self.deployment_name}")
        for _ in range(count):
            replica_id = ApplicationRegistry.get_new_deployment_replica_id(self.app_name, self.deployment_name)

            try:
                # Prepare actor options with runtime_env for app name and deployment name
                actor_options = self.ray_actor_options.copy()

                # Set runtime_env with app name and logging configuration
                runtime_env = actor_options.get("runtime_env", {})
                env_vars = runtime_env.get("env_vars", {})
                env_vars["POLYMATHERA_SERVING_CURRENT_APP"] = self.app_name
                env_vars["POLYMATHERA_SERVING_CURRENT_DEPLOYMENT"] = self.deployment_name
                env_vars["POLYMATHERA_SERVING_CURRENT_REPLICA_ID"] = replica_id

                # Propagate logging config if set, otherwise fall back to env var
                if self.logging_config:
                    import json
                    from dataclasses import asdict
                    env_vars["POLYMATHERA_LOGGING_CONFIG"] = json.dumps(asdict(self.logging_config))
                else:
                    # Fallback to legacy environment variable
                    log_level = os.environ.get("POLYMATHERA_LOG_LEVEL")
                    if log_level:
                        env_vars["POLYMATHERA_LOG_LEVEL"] = log_level

                runtime_env["env_vars"] = env_vars
                actor_options["runtime_env"] = runtime_env
                actor_options["name"] = replica_id

                # Use the same namespace as the proxy actor
                actor_options["namespace"] = ApplicationRegistry.get_ray_actor_namespace(self.app_name)

                # Create Ray actor for this replica with bound init args/kwargs
                actor_handle = (
                    ray.remote(self.deployment_class)
                    .options(**actor_options)
                    .remote(*self.deployment_init_args, **self.deployment_init_kwargs)
                )

                # Call initialization lifecycle hooks on the replica
                await self._call_lifecycle_hooks(actor_handle)

                # Create replica info
                replica = DeploymentReplicaInfo(
                    replica_id=replica_id,
                    actor_handle=actor_handle,
                    is_healthy=True,
                    last_health_check=time.time(),
                )

                self.replicas.append(replica)

                # Create queue for this replica
                self._replica_queues[replica_id] = asyncio.Queue()

                # Start queue processor for this replica
                self._background_tasks.append(
                    asyncio.create_task(self._process_replica_queue(replica))
                )

                # Start periodic health checks for this replica
                self._start_periodic_health_checks(replica)

                logger.info(f"Added replica {replica_id} to {self.deployment_name}")

            except Exception as e:
                logger.error(f"Failed to add replica {replica_id}: {e}", exc_info=True)

    async def _remove_replicas(self, count: int) -> None:
        """Remove replicas from the deployment.

        Args:
            count: Number of replicas to remove.
        """
        # Remove replicas from the end (LIFO)
        # Prefer removing unhealthy replicas first
        replicas_to_remove = []

        # First, remove unhealthy replicas
        unhealthy = [r for r in self.replicas if not r.is_healthy]
        replicas_to_remove.extend(unhealthy[: min(count, len(unhealthy))])

        # If we need more, remove healthy replicas with least load
        remaining = count - len(replicas_to_remove)
        if remaining > 0:
            healthy = sorted(
                [r for r in self.replicas if r.is_healthy],
                key=lambda r: r.queue_length + r.in_flight_requests,
            )
            replicas_to_remove.extend(healthy[:remaining])

        # Cleanup and kill the selected replicas
        for replica in replicas_to_remove:
            try:
                # Stop periodic health checks for this replica
                await self._stop_periodic_health_checks(replica.replica_id)

                # Call cleanup hooks before killing
                await self._call_cleanup_hooks(replica.actor_handle)

                # Kill the replica
                ray.kill(replica.actor_handle)
                self.replicas.remove(replica)
                logger.info(f"Removed replica {replica.replica_id} from {self.deployment_name}")
            except Exception as e:
                logger.error(
                    f"Failed to remove replica {replica.replica_id}: {e}",
                    exc_info=True,
                )

    async def get_stats(self) -> dict[str, Any]:
        """Get deployment statistics.

        Returns:
            Dictionary with deployment stats.
        """
        async with self._replica_lock:
            healthy_count = sum(1 for r in self.replicas if r.is_healthy)
            total_queue = sum(r.queue_length for r in self.replicas)
            total_in_flight = sum(r.in_flight_requests for r in self.replicas)

            return {
                "deployment_name": self.deployment_name,
                "total_replicas": len(self.replicas),
                "healthy_replicas": healthy_count,
                "total_queue_length": total_queue,
                "total_in_flight": total_in_flight,
                "autoscaling_config": {
                    "min_replicas": self.autoscaling_config.min_replicas,
                    "max_replicas": self.autoscaling_config.max_replicas,
                    "target_queue_length": self.autoscaling_config.target_queue_length,
                },
            }

    async def get_all_replica_property(self, property_name: str) -> dict[str, Any]:
        """Get a property value from all replicas.

        This queries all healthy replicas for a specific property marked with
        @serving.replica_property(name). The property must be defined on the
        deployment class and decorated with @replica_property.

        Args:
            property_name: Name of the property to query (as defined in decorator)

        Returns:
            Dictionary mapping replica_id to property value.

        Raises:
            ValueError: If no method found with the specified property name

        Example:
            # Deployment has:
            # @serving.endpoint
            # @serving.replica_property("resource_usage")
            # async def get_resource_usage(self) -> dict[str, Any]: ...

            # Query the property:
            usage = await proxy.get_all_replica_property("resource_usage")
            # Returns: {
            #     "replica_0": {"cpu": 2.5, "memory": 1024, ...},
            #     "replica_1": {"cpu": 1.2, "memory": 512, ...},
            # }
        """
        replica_properties = {}

        async with self._replica_lock:
            if not self.replicas:
                return replica_properties

            # Get property-to-method mapping from first healthy replica
            method_name = None
            for replica in self.replicas:
                if replica.is_healthy:
                    try:
                        property_mapping = await replica.actor_handle.__get_replica_properties__.remote()
                        method_name = property_mapping.get(property_name)
                        if method_name:
                            break
                    except Exception as e:
                        logger.warning(
                            f"Failed to get property mapping from replica {replica.replica_id}: {e}"
                        )
                        continue

            if method_name is None:
                raise ValueError(
                    f"No method found with @replica_property('{property_name}') "
                    f"on deployment {self.deployment_name}"
                )

            # Query each healthy replica for the property
            for replica in self.replicas:
                if not replica.is_healthy:
                    continue

                try:
                    # Call the method on the replica
                    method = getattr(replica.actor_handle, method_name)
                    property_value = await method.remote()
                    replica_properties[replica.replica_id] = property_value
                except Exception as e:
                    logger.warning(
                        f"Failed to get property '{property_name}' from replica {replica.replica_id}: {e}"
                    )

        return replica_properties

    async def shutdown(self) -> None:
        """Shutdown the deployment and cleanup resources."""
        logger.info(f"Shutting down deployment {self.deployment_name}")

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Stop monitoring
        await self.autoscaler.stop()
        await self.health_monitor.stop()

        # Cleanup and kill all replicas
        async with self._replica_lock:
            for replica in self.replicas:
                try:
                    # Stop periodic health checks for this replica
                    await self._stop_periodic_health_checks(replica.replica_id)

                    # Call cleanup hooks before killing
                    await self._call_cleanup_hooks(replica.actor_handle)

                    # Kill the replica
                    ray.kill(replica.actor_handle)
                except Exception as e:
                    logger.error(f"Error killing replica {replica.replica_id}: {e}")

            self.replicas.clear()

        logger.info(f"Deployment {self.deployment_name} shutdown complete")
