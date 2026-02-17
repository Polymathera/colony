"""Polymathera Serving Framework.

A lightweight, production-ready serving framework built on Ray Core for
building scalable, fault-tolerant services.

Key Features:
- Declarative deployment API with @deployment decorator
- Automatic fault tolerance and health monitoring
- Built-in autoscaling based on load
- Flexible request routing (round-robin, least-loaded, custom)
- Service discovery and inter-deployment communication
- Comprehensive observability with Prometheus metrics

Example:
    ```python
    from colony.distributed.ray_utils import serving

    @serving.deployment(
        autoscaling_config={
            "min_replicas": 2,
            "max_replicas": 10,
        }
    )
    class MyService:
        @serving.endpoint
        async def process(self, data: str) -> str:
            return f"Processed: {data}"

    # Create application
    app = serving.Application(name="MyApp")
    app.add_deployment(MyService.bind())

    # Run application
    await app.start()

    # Call from another deployment
    handle = serving.get_deployment("MyApp", "MyService")
    result = await handle.process("test data")
    ```
"""

# Core decorators
from .decorators import (
    DeploymentConfig,
    deployment,
    endpoint,
    replica_property,
    initialize_deployment,
    on_app_ready,
    cleanup_deployment,
    periodic_health_check,
)

# Application and handles
from .application import Application
from .handle import (
    DeploymentHandle,
    get_deployment,
    get_my_app_name,
    get_my_deployment_name,
    get_my_replica_id,
)

# Models (for type hints and advanced usage)
from .models import (
    ApplicationInfo,
    ApplicationRegistry,
    AutoscalingConfig,
    DeploymentProxyInfo,
    DeploymentReplicaInfo,
    DeploymentRequest,
    DeploymentResponse,
    DeploymentResponseStatus,
    LoggingConfig,
    RoutingHints,
    # Routing (for custom routing policies)
    LeastLoadedRouter,
    RequestRouter,
    RoundRobinRouter,
)


# Internal components (exposed for advanced usage and testing)
from .autoscaler import DeploymentAutoscaler
from .health import DeploymentHealthMonitor
from .proxy import DeploymentProxyRayActor

__all__ = [
    # Decorators
    "DeploymentConfig",
    "deployment",
    "endpoint",
    "replica_property",
    "initialize_deployment",
    "on_app_ready",
    "cleanup_deployment",
    "periodic_health_check",
    # Application
    "Application",
    # Handles
    "DeploymentHandle",
    "get_deployment",
    "get_my_app_name",
    "get_my_deployment_name",
    "get_my_replica_id",
    # Models
    "ApplicationInfo",
    "ApplicationRegistry",
    "AutoscalingConfig",
    "DeploymentProxyInfo",
    "LoggingConfig",
    "DeploymentRequest",
    "DeploymentResponse",
    "DeploymentResponseStatus",
    "DeploymentReplicaInfo",
    "RoutingHints",
    # Routing
    "RequestRouter",
    "RoundRobinRouter",
    "LeastLoadedRouter",
    # Internal components
    "DeploymentProxyRayActor",
    "DeploymentHealthMonitor",
    "DeploymentAutoscaler",
]