"""Core data models for the serving framework."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
import traceback
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from enum import Enum
from typing import Any, Literal
import uuid

from ...state_management import SharedState


logger = logging.getLogger(__name__)



class RequestRouter(ABC):
    """Base class for request routing policies.

    Subclass this to implement custom routing logic for deployments.
    The router determines which replica should handle each incoming request.
    """

    def __init__(self):
        """Initialize the router."""
        pass

    @abstractmethod
    async def route_request(
        self,
        request: DeploymentRequest,
        replicas: list[DeploymentReplicaInfo],
    ) -> DeploymentReplicaInfo:
        """Route a request to a replica.

        Args:
            request: The incoming request to route.
            replicas: Available healthy replicas.

        Returns:
            The replica that should handle this request.

        Raises:
            ValueError: If no suitable replica can be found.
        """
        pass


class RoundRobinRouter(RequestRouter):
    """Round-robin load balancer across replicas."""

    def __init__(self):
        super().__init__()
        self._index = 0

    async def route_request(
        self,
        request: DeploymentRequest,
        replicas: list[DeploymentReplicaInfo],
    ) -> DeploymentReplicaInfo:
        """Route request using round-robin selection."""
        if not replicas:
            raise ValueError("No healthy replicas available")

        replica = replicas[self._index % len(replicas)]
        self._index += 1
        return replica


class LeastLoadedRouter(RequestRouter):
    """Route requests to the replica with the least load.

    Load is measured as: queue_length + in_flight_requests
    """

    async def route_request(
        self,
        request: DeploymentRequest,
        replicas: list[DeploymentReplicaInfo],
    ) -> DeploymentReplicaInfo:
        """Route request to the replica with lowest load."""
        if not replicas:
            raise ValueError("No healthy replicas available")

        # Calculate load for each replica
        def get_load(replica: DeploymentReplicaInfo) -> int:
            return replica.queue_length + replica.in_flight_requests

        # Select replica with minimum load
        return min(replicas, key=get_load)


@dataclass
class RoutingHints:
    """Routing metadata for intelligent request routing.

    This class carries routing information extracted from method arguments,
    allowing routers to make intelligent routing decisions without inspecting
    the original arguments.

    Routing hints are automatically extracted by DeploymentHandle based on
    endpoint configuration and attached to DeploymentRequest.

    Example:
        ```python
        # Automatically extracted from InferenceRequest
        hints = RoutingHints(
            router_class=ContextAwareRouter,
            router_kwargs={"strip_routing_params": ["target_client_id"]},
            metadata={
                "context_page_ids": ["page-1", "page-2", "page-3"],
                "tenant_id": "tenant-a",
                "requirements": LLMClientRequirements(min_context_window=4096)
            },
        )

        # Used by router for intelligent routing
        request = DeploymentRequest(
            method_name="infer",
            args=(inference_request,),
            routing_hints=hints,
        )
        ```
    """

    router_class: type[RequestRouter] | None = LeastLoadedRouter
    """Router class to use for this request."""

    router_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to pass to router constructor.

    Allows per-endpoint router configuration. For example, TargetClientRouter
    can be configured to strip routing parameters from method kwargs before
    passing them to the actual method.
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional routing metadata for custom routers."""


class DeploymentRequest(BaseModel):
    """Request to a deployment endpoint.

    Encapsulates a remote method call with routing metadata for intelligent
    request routing across replicas.
    """

    request_id: str
    """Unique identifier for this request."""

    method_name: str
    """Name of the method to call on the deployment."""

    args: tuple[Any, ...] = ()
    """Positional arguments for the method."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments for the method."""

    routing_hints: RoutingHints | None = None
    """Routing hints for intelligent request routing.

    Extracted automatically by DeploymentHandle based on endpoint configuration
    and method arguments. Used by routers to make smart routing decisions without
    inspecting the original args/kwargs.

    Example:
        ```python
        # Hints extracted from InferenceRequest argument
        request = DeploymentRequest(
            method_name="infer",
            args=(inference_request,),
            routing_hints=RoutingHints(
                router_class=ContextAwareRouter,
                metadata={"context_page_ids": ["page-1", "page-2"]},
            ),
        )
        ```
    """

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Optional metadata for routing and monitoring."""


class DeploymentResponseStatus(str, Enum):
    """Status of a deployment response."""

    SUCCESS = "success"
    ERROR = "error"


class DeploymentResponse(BaseModel):
    """Response from a deployment endpoint."""

    request_id: str
    """Request ID this response corresponds to."""

    status: DeploymentResponseStatus
    """Status of the request execution."""

    result: Any = None
    """Result value if successful."""

    error: str | None = None
    """Error message if failed."""

    error_type: str | None = None
    """Exception type name if failed (e.g., 'ResourceExhausted', 'ValueError')."""

    error_module: str | None = None
    """Exception module path if failed (e.g., 'polymathera.colony.agents.base', 'builtins')."""

    traceback: str | None = None
    """Traceback if failed."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Optional metadata (e.g., timing, resource usage)."""

    @classmethod
    def with_success(cls, request_id: str, result: Any, metadata: dict[str, Any] | None = None) -> DeploymentResponse:
        """Create a successful response."""
        return cls(
            request_id=request_id,
            status=DeploymentResponseStatus.SUCCESS,
            result=result,
            metadata=metadata or {},
        )

    @classmethod
    def with_error(cls, request_id: str, error: Exception, metadata: dict[str, Any] | None = None) -> DeploymentResponse:
        """Create an error response.

        Preserves the exception type and module so it can be re-raised on the client side.
        """
        return cls(
            request_id=request_id,
            status=DeploymentResponseStatus.ERROR,
            error=str(error),
            error_type=error.__class__.__name__,
            error_module=error.__class__.__module__,
            traceback=traceback.format_exc(),
            metadata=metadata or {},
        )


@dataclass
class DeploymentReplicaInfo:
    """Information about a deployment replica."""

    replica_id: str
    """Unique identifier for this replica."""

    actor_handle: Any
    """Ray actor handle for this replica."""

    queue_length: int = 0
    """Current queue length for this replica."""

    in_flight_requests: int = 0
    """Number of requests currently being processed."""

    last_health_check: float | None = None
    """Timestamp of last successful health check."""

    is_healthy: bool = True
    """Whether this replica is healthy."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional metadata for custom routing."""


@dataclass
class LoggingConfig:
    """Configuration for logging in deployments and actors."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    """Logging level."""

    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    """Log message format."""

    logs_dir: str | None = None
    """Directory to store the logs. Default to None, which means driver logs will not be stored persistently and Ray actor logs will still be available in the default Ray logs ('/tmp/ray/session_latest/logs/serve/...')."""

    def apply(self) -> None:
        """Apply this logging configuration to the current process."""
        import logging

        logging.basicConfig(
            level=getattr(logging, self.level, logging.INFO),
            format=self.format,
            force=True,  # Override any existing configuration
        )


@dataclass
class AutoscalingConfig:
    """Configuration for deployment autoscaling."""

    min_replicas: int = 1
    """Minimum number of replicas to maintain."""

    max_replicas: int = 10
    """Maximum number of replicas to maintain."""

    target_queue_length: int = 5
    """Target queue length per replica for autoscaling."""

    scale_up_cooldown_s: float = 10.0
    """Cooldown period after scaling up."""

    scale_down_cooldown_s: float = 30.0
    """Cooldown period after scaling down."""

    max_concurrency: int | None = None
    """Maximum concurrent requests per replica. None = unlimited."""

    def __post_init__(self):
        """Validate configuration."""
        if self.min_replicas < 1:
            raise ValueError("min_replicas must be >= 1")
        if self.max_replicas < self.min_replicas:
            raise ValueError("max_replicas must be >= min_replicas")
        if self.target_queue_length < 1:
            raise ValueError("target_queue_length must be >= 1")
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1 or None")


@dataclass
class HealthCheckConfig:
    """Configuration for deployment health checks."""

    interval_s: float = 10.0
    """Interval between health checks in seconds."""

    timeout_s: float = 5.0
    """Timeout for each health check ping in seconds."""

    max_consecutive_failures: int = 3
    """Number of consecutive failures before marking a replica unhealthy."""


class DeploymentProxyInfo(BaseModel):
    """Information about a deployment proxy in an application."""

    deployment_name: str
    """Name of the deployment class."""

    proxy_actor_name: str
    """Name of the proxy actor managing this deployment."""


class ApplicationInfo(BaseModel):
    """Information about a running serving application."""

    app_name: str
    """Name of the application."""

    deployments: dict[str, DeploymentProxyInfo] = Field(default_factory=dict)
    """Map of deployment_name to DeploymentProxyInfo."""

    created_at: float
    """Timestamp when the application was created."""


class ApplicationRegistry(SharedState):
    """Distributed registry of all serving applications in the cluster."""

    applications: dict[str, ApplicationInfo] = Field(default_factory=dict)
    """Map of app_name to ApplicationInfo."""

    def get_app(self, app_name: str) -> ApplicationInfo | None:
        """Get application info by name."""
        return self.applications.get(app_name)

    def register_app(self, app_info: ApplicationInfo) -> None:
        """Register a new application."""
        self.applications[app_info.app_name] = app_info

    def unregister_app(self, app_name: str) -> ApplicationInfo | None:
        """Unregister an application and return its info."""
        return self.applications.pop(app_name, None)

    def list_apps(self) -> list[ApplicationInfo]:
        """List all registered applications."""
        return list(self.applications.values())

    @staticmethod
    def get_deployment_proxy_actor_name(app_name: str, deployment_name: str) -> str:
        """Get the Ray actor name for a deployment proxy.

        Args:
            app_name: Name of the application.
            deployment_name: Name of the deployment.
        Returns:
            The Ray actor name for the deployment proxy.
        """
        return f"polymathera-serving-apps-{app_name}-{deployment_name}-proxy"

    @staticmethod
    def get_new_deployment_replica_id(app_name: str, deployment_name: str) -> str:
        return f"polymathera-serving-apps-{app_name}-{deployment_name}-replica-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def get_ray_actor_namespace(app_name: str) -> str:
        """Get the Ray actor namespace for this application."""
        return f"polymathera.serving.{app_name}"

    @staticmethod
    def get_deployment_proxy_actor_name_prefix(app_name:str) -> str:
        """Get the prefix for deployment proxy actor names in this application."""
        return f"polymathera-serving-apps-{app_name}-"

    @staticmethod
    def get_deployment_proxy_actor_name_suffix() -> str:
        """Get the suffix for deployment proxy actor names in this application."""
        return "-proxy"
