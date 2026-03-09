# Polymathera Serving Framework

A lightweight, production-ready serving framework built on Ray Core for building scalable, fault-tolerant services.

## Features

- **Declarative API**: Use simple decorators to define deployments and endpoints
- **Automatic Fault Tolerance**: Built-in health monitoring and automatic replica restart
- **Autoscaling**: Automatic scaling based on request queue length and load
- **Flexible Routing**: Choose from built-in routers or implement custom routing logic
- **Service Discovery**: Easy inter-deployment communication
- **Observable**: Comprehensive Prometheus metrics for monitoring

## Key Features Implemented

✅ Declarative API - Simple @deployment and @endpoint decorators
✅ Fault Tolerance - Health monitoring with automatic unhealthy replica detection
✅ Autoscaling - Automatic scaling based on queue length with configurable cooldowns
✅ Flexible Routing - Built-in round-robin and least-loaded routers + custom router support
✅ Service Discovery - get_deployment() for inter-deployment communication
✅ Observability - Prometheus metrics throughout (requests, latency, replicas, health checks)
✅ Error Handling - Application errors caught and returned with tracebacks
✅ Lifecycle Hooks - Pre/post initialization decorators
✅ Production Ready - Comprehensive logging, error handling, graceful shutdown

## Architecture Highlights

- Class transformation: The @deployment decorator wraps user classes with request handling, health checking, and error handling
- Registry-based discovery: No serialization overhead - classes are registered locally
- Proxy pattern: Each deployment has a proxy actor that manages replicas
- Component-based design: Separate health monitor and autoscaler components
- Async-first: Built for async/await with proper coroutine handling



## Quick Start

### 1. Define a Deployment

```python
from polymathera.rayutils import serving

@serving.deployment(
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 10,
        "target_queue_length": 5,
    },
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0,
    }
)
class MyService:
    def __init__(self):
        self.counter = 0

    @serving.endpoint
    async def process(self, data: str) -> str:
        self.counter += 1
        return f"Processed: {data} (count: {self.counter})"
```

### 2. Create an Application

```python
import asyncio
import ray

async def main():
    # Initialize Ray
    ray.init()

    # Create application
    app = serving.Application(name="MyApp")
    app.add_deployment(MyService.bind())

    # Start application
    await app.start()

    # Call the deployment
    handle = serving.get_deployment("MyApp", "MyService")
    result = await handle.process("test data")
    print(result)

    # Stop application
    await app.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Concepts

### Deployments

A deployment is a collection of replicas of a service class. Use the `@deployment` decorator to mark a class as a deployment:

```python
@serving.deployment(
    name="MyDeployment",  # Optional, defaults to class name
    routing_policy=CustomRouter,  # Optional, defaults to LeastLoadedRouter
    autoscaling_config={...},  # Optional autoscaling configuration
    ray_actor_options={...},  # Optional Ray actor options
)
class MyDeployment:
    pass
```

### Endpoints

Mark methods as endpoints using the `@endpoint` decorator. Only endpoints can be called remotely:

```python
@serving.endpoint
async def my_method(self, arg: str) -> str:
    return f"Result: {arg}"
```

### Autoscaling

Configure autoscaling for each deployment:

```python
autoscaling_config={
    "min_replicas": 1,          # Minimum replicas (default: 1)
    "max_replicas": 10,         # Maximum replicas (default: 10)
    "target_queue_length": 5,   # Target queue per replica (default: 5)
    "scale_up_cooldown_s": 10,  # Scale-up cooldown (default: 10s)
    "scale_down_cooldown_s": 30,# Scale-down cooldown (default: 30s)
}
```

The autoscaler monitors request queue length and automatically adjusts the number of replicas to maintain the target queue length.

### Request Routing

Built-in routing policies:

- **LeastLoadedRouter** (default): Routes to replica with lowest queue + in-flight requests
- **RoundRobinRouter**: Simple round-robin across replicas

Implement custom routing by subclassing `RequestRouter`:

```python
class CustomRouter(serving.RequestRouter):
    async def route_request(
        self,
        request: serving.DeploymentRequest,
        replicas: list[serving.DeploymentReplicaInfo],
    ) -> serving.DeploymentReplicaInfo:
        # Your custom routing logic
        return replicas[0]
```

### Service Discovery

Deployments can call each other using `get_deployment`:

```python
@serving.deployment
class ServiceA:
    @serving.endpoint
    async def call_b(self):
        # Get handle to ServiceB
        service_b = serving.get_deployment(
            serving.get_my_app_name(),
            deployment_class=ServiceB,
        )
        # Call ServiceB
        result = await service_b.some_method()
        return result
```

### Lifecycle Hooks

Use lifecycle decorators for initialization phases:

```python
@serving.deployment
class MyService:
    @serving.pre_initialize_deployment
    async def pre_init(self):
        # Runs before deployment initialization
        pass

    @serving.initialize_deployment
    async def init(self):
        # Runs during deployment initialization
        pass

    @serving.post_initialize_deployment
    async def post_init(self):
        # Runs after deployment initialization
        pass
```

## Architecture

### Components

1. **DeploymentProxyRayActor**: Single entry point for each deployment, handles:
   - Request routing to replicas
   - Health monitoring
   - Autoscaling decisions
   - Fault tolerance

2. **DeploymentHealthMonitor**: Monitors replica health with periodic health checks

3. **DeploymentAutoscaler**: Adjusts replica count based on load metrics

4. **RequestRouter**: Routes requests to appropriate replicas

5. **DeploymentHandle**: Client-side proxy for calling deployments

### Request Flow

```
Client
  ↓
DeploymentHandle
  ↓
DeploymentProxyRayActor (routes request via RequestRouter)
  ↓
Replica (wrapped deployment class)
  ↓
Response back to client
```

## Monitoring

The framework exports Prometheus metrics:

- `deployment_request_total`: Total requests per deployment/method
- `deployment_request_duration_seconds`: Request latency
- `deployment_current_replicas`: Current replica count
- `deployment_health_check_total`: Health check results
- `deployment_autoscaling_decisions_total`: Autoscaling actions

## Examples

See `example.py` for complete working examples including:
- Simple deployments
- Inter-deployment communication
- Custom routing policies
- Application management

## Comparison with Ray Serve

This framework is built on Ray Core (not Ray Serve) and provides:

- ✅ Simpler, more transparent architecture
- ✅ No HTTP overhead (pure Ray actor communication)
- ✅ Class-based deployments with local registration (no serialization/deserialization)
- ✅ Built-in Prometheus metrics
- ✅ Explicit lifecycle management
- ❌ No HTTP ingress (use Ray Serve if you need HTTP)
- ❌ No model composition patterns (build your own)

## Implementation Details

### Fault Tolerance

- Replicas are Ray actors with automatic restart (`max_restarts`)
- Health monitor marks unresponsive replicas as unhealthy
- Unhealthy replicas are excluded from routing
- Application errors are caught and returned with tracebacks
- Replica crashes trigger autoscaler to maintain replica count

### Performance Considerations

- Request routing is O(1) for RoundRobinRouter, O(n) for LeastLoadedRouter
- Health checks run every 10s (configurable)
- Autoscaling checks run every 5s (configurable)
- No HTTP serialization overhead - pure Python object passing via Ray

### Production Readiness

- Comprehensive error handling and logging
- Prometheus metrics for observability
- Configurable timeouts and cooldowns
- Graceful shutdown with resource cleanup
- Ray actor fault tolerance built-in

## Run the Example

To run the example, use the following command:

```bash
POLYMATHERA_LOG_LEVEL=DEBUG poetry run python polymathera/rayutils/serving/example.py 2>&1
```
This will start a Ray cluster, deploy the example services, and demonstrate inter-service communication.
