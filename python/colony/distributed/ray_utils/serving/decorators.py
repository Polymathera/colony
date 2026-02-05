"""Decorators for defining deployments and endpoints."""

import asyncio
import functools
import inspect
import logging
import os
from typing import Any, Callable, Type

from .models import AutoscalingConfig, DeploymentRequest, DeploymentResponse, LoggingConfig
from .router import RequestRouter

logger = logging.getLogger(__name__)

# Module-level registry for deployment classes
_DEPLOYMENT_REGISTRY: dict[str, Type[Any]] = {}


def endpoint(
    func: Callable | None = None,
    *,
    router_class: Type[RequestRouter] | None = None,
    router_kwargs: dict[str, Any] | None = None,
) -> Callable:
    """Mark a method as a deployment endpoint that can be called remotely.

    This decorator can be used with or without parameters:

    - Without parameters: `@endpoint`
    - With parameters: `@endpoint(router_class=RoundRobinRouter, router_kwargs={...})`

    Args:
        func: The method to mark as an endpoint (when used without parentheses).
        router_class: Router class for this endpoint (defaults to LeastLoadedRouter).
        The router_class must have a class method to extract routing hints from the method name, arguments and router class.
        router_kwargs: Keyword arguments to pass to router constructor. Allows per-endpoint
        router configuration.

    Returns:
        The decorated method with endpoint metadata.

    Examples:
        ```python
        @serving.deployment()
        class VLLMDeployment:
            # Simple endpoint with default routing
            @serving.endpoint
            async def health_check(self) -> str:
                return "healthy"

            # Endpoint with context-aware routing
            @serving.endpoint(router_class=ContextAwareRouter)
            async def infer(self, request: InferenceRequest) -> InferenceResponse:
                # Routes to replicas with required context pages loaded
                ...

            # Endpoint with target-specific routing that strips routing params
            @serving.endpoint(
                router_class=TargetClientRouter,
                router_kwargs={"strip_routing_params": ["target_client_id"]}
            )
            async def load_page(self, page: VirtualContextPage) -> bool:
                # Routes to specific replica, strips target_client_id from kwargs
                ...
        ```
    **TODO**: Maybe we can just pass `router_obj` instead of class and kwargs separately?
    """
    # TODO: Use functools.wraps to preserve method metadata?
    def decorator(f: Callable) -> Callable:
        f.__is_endpoint__ = True  # type: ignore
        f.__router_class__ = router_class  # type: ignore
        f.__router_kwargs__ = router_kwargs or {}  # type: ignore
        return f

    # Support both @endpoint and @endpoint(...)
    if func is None:
        # Called with parameters: @endpoint(router_class=...)
        return decorator
    else:
        # Called without parameters: @endpoint
        return decorator(func)


def replica_property(name: str) -> Callable[[Callable], Callable]:
    """Mark a method as a replica property that can be queried from all replicas.

    This decorator marks a method that returns a property value for the replica.
    The property can then be queried across all replicas using
    `DeploymentHandle.get_all_replica_property(name)`.

    This decorator automatically marks the method as an @endpoint, so you don't
    need to add @endpoint explicitly.

    Use this for:
    - Resource usage metrics (CPU, memory, agent count, etc.)
    - Replica-specific state information
    - Health or status indicators
    - Any read-only property that varies per replica

    Args:
        name: The property name (used to query the property).

    Returns:
        Decorator function.

    Example:
        ```python
        @serving.deployment()
        class MyDeployment:
            @serving.replica_property("resource_usage")
            async def get_resource_usage(self) -> dict[str, Any]:
                return {
                    "cpu_used": self.cpu_usage,
                    "memory_used": self.memory_usage,
                    "agents": len(self.agents),
                }

        # Query from another deployment or client
        handle = serving.get_deployment("MyApp", "MyDeployment")
        all_usage = await handle.get_all_replica_property("resource_usage")
        # Returns: {
        #     "replica_0": {"cpu_used": 2.5, "memory_used": 1024, "agents": 10},
        #     "replica_1": {"cpu_used": 1.2, "memory_used": 512, "agents": 5},
        # }
        ```

    Note:
        - The method is automatically marked as an endpoint (remotely callable)
        - The method should be lightweight and return quickly
        - Multiple properties can be defined on the same deployment
        - Property names must be unique within a deployment
    """
    def decorator(func: Callable) -> Callable:
        # Mark as endpoint (remotely callable)
        func.__is_endpoint__ = True  # type: ignore
        # Mark as replica property
        func.__is_replica_property__ = True  # type: ignore
        func.__replica_property_name__ = name  # type: ignore
        return func

    return decorator


def initialize_deployment(func: Callable) -> Callable:
    """Mark a method to run after replica creation.

    This decorator marks a method to be called automatically after a replica
    is created (via __init__). This is the right place to initialize resources
    that require async operations or coordination with other actors.

    The method will be called:
    - After __init__ completes successfully
    - Before the replica is added to the active replica pool
    - For every replica created (including autoscaled replicas)

    Use this for:
    - Initializing async resources (state managers, connections, etc.)
    - Loading models or data
    - Registering with cluster state

    Args:
        func: The async method to run during initialization.

    Returns:
        The decorated method.

    Example:
        ```python
        @serving.deployment()
        class MyDeployment:
            def __init__(self, model_name: str):
                self.model_name = model_name
                self.state_manager = None

            @serving.initialize_deployment
            async def initialize(self):
                # Initialize async resources
                self.state_manager = await get_state_manager()
                self.model = await load_model(self.model_name)
        ```
    """
    func.__initialize_deployment__ = True  # type: ignore
    return func


def cleanup_deployment(func: Callable) -> Callable:
    """Mark a method to run before replica destruction.

    This decorator marks a method to be called automatically before a replica
    is destroyed (shutdown, scale-down, or failure). This is the right place
    to clean up resources, save state, or deregister from cluster state.

    The method will be called:
    - Before the replica is removed from the active replica pool
    - Before ray.kill() is called on the replica
    - For every replica removed (including autoscaled down replicas)

    Use this for:
    - Closing connections and releasing resources
    - Saving state or checkpoints
    - Deregistering from cluster state
    - Flushing buffers or logs

    Args:
        func: The async method to run during cleanup.

    Returns:
        The decorated method.

    Example:
        ```python
        @serving.deployment()
        class MyDeployment:
            @serving.cleanup_deployment
            async def cleanup(self):
                # Clean up resources
                await self.state_manager.deregister()
                await self.model.save_checkpoint()
                await self.connection.close()
        ```
    """
    func.__cleanup_deployment__ = True  # type: ignore
    return func


def periodic_health_check(interval_s: float = 30.0) -> Callable[[Callable], Callable]:
    """Mark a method to run periodically for health checks and cleanup tasks.

    This decorator marks a method to be called automatically at regular intervals
    while the replica is running. This is the right place to perform health checks,
    cleanup tasks, state validation, or periodic maintenance.

    The method will be called:
    - Periodically at the specified interval (default: 30 seconds)
    - For each replica independently
    - Starting after the replica is added to the active replica pool
    - Continuing until the replica is removed or shutdown

    Use this for:
    - Performing health checks (e.g., connection validation)
    - Periodic cleanup tasks (e.g., cache eviction, temp file cleanup)
    - State validation and self-healing
    - Metrics collection and reporting
    - Resource usage monitoring

    Args:
        interval_s: Interval in seconds between executions (default: 30.0).

    Returns:
        Decorator function.

    Example:
        ```python
        @serving.deployment()
        class MyDeployment:
            def __init__(self):
                self.connection = None
                self.cache = {}

            @serving.initialize_deployment
            async def initialize(self):
                self.connection = await create_connection()

            @serving.periodic_health_check(interval_s=30.0)
            async def health_check(self):
                # Validate connection is still alive
                if not await self.connection.ping():
                    logger.warning("Connection dead, reconnecting...")
                    self.connection = await create_connection()

                # Clean up old cache entries
                now = time.time()
                self.cache = {
                    k: v for k, v in self.cache.items()
                    if now - v['timestamp'] < 300
                }
        ```

    Note:
        - The method should be async
        - Errors are logged but do not stop the periodic execution or crash the replica
        - Multiple methods can be decorated with different intervals
        - The first execution happens after the specified interval (not immediately)
    """
    def decorator(func: Callable) -> Callable:
        func.__periodic_health_check__ = True  # type: ignore
        func.__health_check_interval_s__ = interval_s  # type: ignore
        return func

    return decorator


class DeploymentConfig:
    """Configuration for a deployment, attached to the class."""

    def __init__(
        self,
        name: str | None = None,
        router_class: Type[RequestRouter] | None = None,
        autoscaling_config: dict[str, Any] | None = None,
        ray_actor_options: dict[str, Any] | None = None,
        max_concurrency: int | None = None,
        logging_config: LoggingConfig | None = None,
    ):
        self.name = name
        self.router_class = router_class
        self.autoscaling_config = autoscaling_config
        self.ray_actor_options = ray_actor_options
        self.max_concurrency = max_concurrency
        self.logging_config = logging_config
        # Endpoint metadata: method_name -> router_class
        self.endpoint_router_classes: dict[str, Type[RequestRouter] | None] = {}
        # Endpoint router kwargs: method_name -> router_kwargs
        self.endpoint_router_kwargs: dict[str, dict[str, Any]] = {}

    def register_endpoint(
        self,
        method_name: str,
        router_class: Type[RequestRouter] | None = None,
        router_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Register endpoint with routing policy.

        Args:
            method_name: Name of the endpoint method.
            router_class: Router class for this endpoint (None = use default).
            router_kwargs: Router constructor kwargs for this endpoint.
        """
        self.endpoint_router_classes[method_name] = router_class
        self.endpoint_router_kwargs[method_name] = router_kwargs or {}

    def get_endpoint_router_class(self, method_name: str) -> Type[RequestRouter] | None:
        """Get router class for an endpoint.

        Args:
            method_name: Name of the endpoint method.

        Returns:
            Router class if configured, None otherwise.
        """
        return self.endpoint_router_classes.get(method_name)

    def get_endpoint_router_kwargs(self, method_name: str) -> dict[str, Any]:
        """Get router kwargs for an endpoint.

        Args:
            method_name: Name of the endpoint method.

        Returns:
            Router kwargs if configured, empty dict otherwise.
        """
        return self.endpoint_router_kwargs.get(method_name, {})


def deployment(
    name: str | None = None,
    router_class: Type[RequestRouter] | None = None,
    autoscaling_config: dict[str, Any] | None = None,
    ray_actor_options: dict[str, Any] | None = None,
    max_concurrency: int | None = None,
    logging_config: LoggingConfig | None = None,
) -> Callable[[Type[Any]], Type[Any]]:
    """Decorator to mark a class as a deployment.

    This wraps the user's service class with additional functionality:
    - Request handling and routing
    - Health checking
    - Error handling with tracebacks
    - Lifecycle methods

    Args:
        name: Deployment name (defaults to class name).
        router_class: Custom router class.
        autoscaling_config: Autoscaling configuration dict.
        ray_actor_options: Ray actor options dict.
        max_concurrency: Maximum concurrent requests per replica (None = unlimited).
        logging_config: Logging configuration for deployment actors.

    Returns:
        Decorator function.

    Example:
        ```python
        @deployment(
            name="MyService",
            autoscaling_config={"min_replicas": 2, "max_replicas": 10},
            logging_config=LoggingConfig(level="DEBUG")
        )
        class MyService:
            @endpoint
            async def process(self, data: str) -> str:
                return f"Processed: {data}"
        ```
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        # Store deployment config on the class
        deployment_name = name or cls.__name__
        config = DeploymentConfig(
            name=deployment_name,
            router_class=router_class,
            autoscaling_config=autoscaling_config,
            ray_actor_options=ray_actor_options,
            max_concurrency=max_concurrency,
            logging_config=logging_config,
        )
        cls.__deployment_config__ = config  # type: ignore

        # Discover and register endpoint routing policies
        for attr_name in dir(cls):
            try:
                attr = getattr(cls, attr_name)
                if callable(attr) and getattr(attr, "__is_endpoint__", False):
                    router_class_value = getattr(attr, "__router_class__", None)
                    router_kwargs_value = getattr(attr, "__router_kwargs__", {})
                    config.register_endpoint(attr_name, router_class_value, router_kwargs_value)
            except AttributeError:
                pass

        # Register the deployment class
        _DEPLOYMENT_REGISTRY[deployment_name] = cls

        # Create wrapped class that extends the original
        class WrappedDeployment(cls):  # type: ignore
            """Wrapped deployment class with added functionality."""

            def __init__(self, *args, **kwargs):
                """Initialize the deployment with request handling."""
                # Configure logging from runtime_env or fall back to environment variable
                logging_config_dict = os.environ.get("POLYMATHERA_LOGGING_CONFIG")
                if logging_config_dict:
                    import json
                    logging_cfg = LoggingConfig(**json.loads(logging_config_dict))
                    logging_cfg.apply()
                else:
                    # Fallback to legacy environment variable
                    log_level = os.environ.get("POLYMATHERA_LOG_LEVEL", "INFO").upper()
                    logging.basicConfig(
                        level=getattr(logging, log_level, logging.INFO),
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        force=True,
                    )

                super().__init__(*args, **kwargs)
                self._endpoints = self._discover_endpoints()
                logger.info(
                    f"Initialized deployment replica with endpoints: "
                    f"{list(self._endpoints.keys())}"
                )

            def _discover_endpoints(self) -> dict[str, Callable]:
                """Discover all endpoint methods on this class."""
                endpoints = {}
                for attr_name in dir(self):
                    try:
                        attr = getattr(self, attr_name)
                        if callable(attr) and getattr(attr, "__is_endpoint__", False):
                            endpoints[attr_name] = attr
                    except AttributeError:
                        pass
                return endpoints

            def _discover_replica_properties(self) -> dict[str, str]:
                """Discover all replica property methods on this class.

                Returns:
                    Dictionary mapping property_name -> method_name
                """
                properties = {}
                for attr_name in dir(self):
                    try:
                        attr = getattr(self, attr_name)
                        if callable(attr) and getattr(attr, "__is_replica_property__", False):
                            property_name = getattr(attr, "__replica_property_name__", None)
                            if property_name:
                                properties[property_name] = attr_name
                    except AttributeError:
                        pass
                return properties

            async def __handle_request__(
                self, request: DeploymentRequest
            ) -> DeploymentResponse:
                """Handle an incoming request.

                This method is called by the proxy actor to execute a request
                on this replica.

                Args:
                    request: The request to handle.

                Returns:
                    Response with result or error.
                """
                try:
                    # Validate method exists and is an endpoint
                    if request.method_name not in self._endpoints:
                        raise ValueError(
                            f"Method '{request.method_name}' is not a registered endpoint. "
                            f"Available endpoints: {list(self._endpoints.keys())}"
                        )

                    # Get the endpoint method
                    method = self._endpoints[request.method_name]

                    # Call the method with provided arguments
                    result = method(*request.args, **request.kwargs)

                    # Await if coroutine
                    if inspect.iscoroutine(result):
                        result = await result

                    # Return successful response
                    return DeploymentResponse.success(
                        request_id=request.request_id,
                        result=result,
                    )

                except Exception as e:
                    logger.error(
                        f"Error handling request {request.request_id} "
                        f"to method {request.method_name}: {e}",
                        exc_info=True,
                    )
                    # Return error response with traceback
                    return DeploymentResponse.error(
                        request_id=request.request_id,
                        error=e,
                    )

            async def __ping__(self) -> str:
                """Health check endpoint.

                Returns:
                    Status message.
                """
                return "healthy"

            async def __get_replica_properties__(self) -> dict[str, str]:
                """Get replica property mapping (internal).

                Returns:
                    Dictionary mapping property_name -> method_name
                """
                return self._discover_replica_properties()

            @classmethod
            def bind(cls, *args, **kwargs) -> "BoundDeployment":
                """Bind deployment with constructor arguments.

                This returns a BoundDeployment object that can be added to an Application.

                Args:
                    *args: Positional arguments for deployment constructor.
                    **kwargs: Keyword arguments for deployment constructor.

                Returns:
                    BoundDeployment instance.
                """
                return BoundDeployment(cls, args, kwargs)

        # Preserve class name and module
        WrappedDeployment.__name__ = cls.__name__
        WrappedDeployment.__qualname__ = cls.__qualname__
        WrappedDeployment.__module__ = cls.__module__

        # Copy over deployment config
        WrappedDeployment.__deployment_config__ = config  # type: ignore

        # Update registry with wrapped class
        _DEPLOYMENT_REGISTRY[deployment_name] = WrappedDeployment

        return WrappedDeployment

    return decorator


class BoundDeployment:
    """A deployment class bound with constructor arguments.

    This is returned by the .bind() method and can be passed to Application.
    """

    def __init__(self, deployment_class: Type[Any], args: tuple, kwargs: dict):
        """Initialize bound deployment.

        Args:
            deployment_class: The wrapped deployment class.
            args: Constructor args.
            kwargs: Constructor kwargs.
        """
        self.deployment_class = deployment_class
        self.args = args
        self.kwargs = kwargs
        self.config: DeploymentConfig = deployment_class.__deployment_config__  # type: ignore


def get_deployment_class(name: str) -> Type[Any] | None:
    """Get a registered deployment class by name.

    Args:
        name: Name of the deployment.

    Returns:
        The deployment class if found, None otherwise.
    """
    return _DEPLOYMENT_REGISTRY.get(name)


def list_deployments() -> list[str]:
    """List all registered deployment names.

    Returns:
        List of deployment names.
    """
    return list(_DEPLOYMENT_REGISTRY.keys())