
# Execution Abstractions

This document describes the high-level abstractions and components for Polymathera's scalable, fault-tolerant, and efficient serving framework to build services on top of Ray Core.


## Decorators and Run APIs

- Similar to `ray.serve.deployment`, `@polymathera.rayutils.serving.deployment` decorator is applied to the user-defined service class to be deployed to define fault-tolerant deployments of identical replicas of a service. It can take the following optional arguments:
    - **Name**: This is name of the deployment. If not provided, it defaults to the class name. It can be used for service discovery.
    - **Routing policy**: This is a class that subclasses `polymathera.rayutils.serving.RequestRouter` and implements a custom request routing policy by overriding the `route_request` method. If not provided, it defaults to a built-in load balancer that load-balances requests across replicas.
    - **Autoscaling config**: This is a dictionary that specifies the autoscaling configuration for the deployment. It can include the following keys:
        - `min_replicas`: The minimum number of replicas to maintain (default: 1).
        - `max_replicas`: The maximum number of replicas to maintain (default: 10).
        - `target_queue_length`: The target length of the request queue per replica (default: 5).
        - `scale_up_cooldown_s`: The cooldown period (in seconds) after scaling up before another scale-up can occur (default: 10).
        - `scale_down_cooldown_s`: The cooldown period (in seconds) after scaling down before another scale-down can occur (default: 30).
    - **Ray actor options**: This is a dictionary that specifies the Ray actor options for the deployment replicas. It can include the following keys:
        - `num_cpus`: The number of CPUs to allocate to each replica (default: 1).
        - `num_gpus`: The number of GPUs to allocate to each replica (default: 0).
        - `memory`: The amount of memory to allocate to each replica (default: None).
        - `resources`: A dictionary of custom resources to allocate to each replica (default: None).

A service class decorated with `@polymathera.rayutils.serving.deployment` can have multiple methods that can be called remotely, each decorated with the `@polymathera.serving.endpoint` decorator.

The `@polymathera.serving.deployment` decorator inspects the decorated class and generates a new class that extends the original class with additional methods for:
- **Request handling**: This includes an initialization method that sets up the request queue and a method that reads requests from the queue and routes them to the appropriate decorated method of the decorated service class.
- **Fault tolerance**: It adds error handling (catching application errors and reporting them to requesters), retry logic to the methods of the original class and liveness checks with the proxy actor (configurable intermittency settings).
- **Autoscaling**: This adds methods to monitor request load and the current resource utilization for the autoscaler running in the proxy actor to query and make autoscaling decisions.

- Unlike Ray's internal use of serialization/deserialization of deployment classes, we can just register the decorated class with a Polymathera service registry that is available to all deployments in their runtime environment. If a deployment class is not available in the local registry at any node, an error will be sent back to the deployment proxy that is then escalated to the caller.


## Runtime Components

#### Deployment Proxy Actor

The `polymathera.rayutils.serving.DeploymentProxyRayActor` runs on some node in the Ray cluster. It acts as a single entry point (ingress) for all requests to the deployment and is responsible for implementing:
- **Routing policy**: routing them to the appropriate replica.
- **Fault tolerance** and deployment monitoring: ensuring that the deployment remains healthy and can recover from failures.
- **Autoscaling**: automatically adjusting the number of replicas based on the incoming request load.


#### Autoscaling

The `polymathera.rayutils.serving.DeploymentAutoscaler` runs as a component of the `DeploymentProxyRayActor` to monitor the request queue length and the number of in-flight requests and adjust the number of replicas up or down accordingly.

    - **Autoscaling**: This adds methods to monitor request load and the current resource utilization for the autoscaler running in the proxy actor to query and make autoscaling decisions.

#### Request Routing Policy
The routing policy runs as a component of the `DeploymentProxyRayActor` to determine how requests are routed to replicas. It can be either:
- A built-in **load balancer** that is responsible for load-balancing requests across replicas or pushing requests to a shared (distributed) queue created as follows:

```python
self.queue_store = await get_polymathera().create_distributed_work_queue(
    self.queue_prefix
)
```


- A **custom request router** that subclasses `polymathera.rayutils.serving.RequestRouter` and overrides a specific routing policy in the `route_request` method which maps requests to replicas (e.g., prefix-aware routing, locality-aware routing, routing table, etc.). In this case, replicas have private request queues. Note that `distributed_work_queue.DistributedQueueStore` can (or needs to) support multiple querues using replica-specific queue prefixes.
  - The custom routing policy may not find a suitable replica to handle the request (e.g., the required context page is not loaded in the KV cache of any replica), in which case it will schedule the loading of the required context into a replica before handling the routing request, or it will queue the request until the required context is loaded into a replica according to some custom prioritization scheme specific to the custom request router.





#### Fault tolerance
The `polymathera.rayutils.serving.DeploymentHealthMonitor` runs as a component of the `DeploymentProxyRayActor` to monitor the health of replicas (e.g., checking their liveness and responsiveness or reachability) and restart them if they become unresponsive or fail. (configurable intermittency settings)

Application errors (like exceptions in model evaluation code) are caught and returned with the traceback information to the requester. The replica will be able to continue to handle other requests.



#### Service discovery and usage

We can use Ray's built-in service discovery mechanism (e.g., finding actors by name) to find and connect to other services in the cluster through their proxy actor.
- The service discovery API `polymathera.serving.get_deployment` allows callers to discover other deployments by class type, name and/or other metadata (e.g., tags, labels, etc.). It returns a `polymathera.serving.DeploymentHandle` that acts as a local proxy to the specified deployment and allows callers to call into these deployments through their proxy actors.


- **Compositions of deployments**: This discovery API is the mechanism that allows deployments to call into each other (and even into themselves) and form a arbitrary compositions of deployments.

- **Method interface**: The call `deployment_handle.method_name(args)` provides a simple method interface for interacting with the associated deployment, where `method_name` is the name of the method being called. So, `DeploymentHandle.__getattr__` ensures that `method_name` is defined on the deployed class and has the correct signature, then validates the arguments and calls `DeploymentProxyRayActor.handle_request.remote` with a `DeploymentRequest` object. The `DeploymentProxyRayActor.handle_request` eventually returns a `DeploymentResponse` that `DeploymentHandle` unpacks and returns the result to the local caller. The `DeploymentRequest` and `DeploymentResponse` classes represent requests and responses between deployments, proxies and deployment handles. They are hidden from deployment callers and handlers (classes decorated with `@polymathera.serving.deployment`).

> Very low priority: <s>**HTTP interface** to the ingress proxy actor for anywhere in the cluster to call into the deployment</s>.



#### Applications

An application is a collection of deployments that work together to provide a service. It is defined using the `polymathera.serving.Application` class. An application can have multiple deployments, each with its own proxy actor, routing policy, autoscaling configuration, and resource requirements. The application is responsible for starting and stopping the deployments and ensuring that they are healthy and available.




> The problem is happening when the `DeploymentProxyRayActor.__init__` (which can be running on the head node or a CPU-only worker node) tries to deserialize one of the deployment classes `VLLMDeployment` or `EmbeddingDeployment` passed to it as an argument, which triggers an import chain leading to vllm code which is not available in the CPU-only worker image because you forgot to update its pytorch and ray base image. So, the question now (relevant to the `polymathera.serving` library) is whether we should:
- Force the `DeploymentProxyRayActor` to run on a node with similar resource requirements as the deployment class (e.g., a GPU) without consuming any of those resources (e.g., by using a `custom_gpu` resource) and
- Avoid passing the deployment class itself to the `DeploymentProxyRayActor` and instead pass just its name so that it can look it up in the global registry of deployment classes that is available to all nodes in the cluster.

**TODO**: The proxy is deployment-specific, so co-locating it with deployment nodes makes sense. Use custom_gpu resource:
- GPU nodes advertise `custom_gpu`: 1 in their ray start resources
- Proxy actors request `custom_gpu: 0.001` to schedule on GPU nodes without consuming GPU
- Pros: Clean resource-based scheduling, no wasted dependencies on CPU nodes
- Cons: Proxy must run on GPU nodes (but doesn't waste GPU since it only uses 0.001 of `custom_gpu`)


Just add a `custom_gpu: 1` resource to GPU nodes in the Ray config and note in a comment that it will be used to schedule actors on GPU nodes without consuming GPU resources. When we decide to implement Option 1, we will only need to edit the source code without the need to redeploy the cluster.
This `rayproject/ray:2.49.0-py311-cu128` base image has CUDA inside. If we don't run any CUDA code (which I am not sure vllm imports won't do) in that container, will this actually work on a CPU-only node (no GPU or nvidia docker toolkit)? If you notice, `deployment/docker/Dockerfile.ray-cpu` installs the CPU version of pytorch and faiss to avoid needing CUDA in the container, just to allow imports of these libraries to work on CPU nodes. So, will this approach even work? Take a step back and think about this architectural decision. What are the options, pros and cons?


**Application Configuration and Deployment Names**

> - [X] In all `serving.deployments` classes in `polymathera/llms_0` and their components, they use hardcoded `deployment_name` strings to get handles to other deployments via `serving.get_deployment()`. This is brittle and error-prone. Instead, we should use configuration objects (perhaps an `ApplicationConfig` loaded from a JSON file at the root directory `polymathera/llms_0`) or environment variables to pass these names around. This will make it easier to change the names and avoid typos.


## Example

Here is a full example of a deployment with a custom request router, fault tolerance and autoscaling:

```python
import polymathera.serving
import asyncio

class MyCustomRequestRouter(polymathera.serving.RequestRouter):
    def __init__(self):
        super().__init__()
        # Data structures for replica states and routing decisions
        pass

    async def route_request(
        self,
        request: polymathera.serving.DeploymentRequest,
        replicas: list[polymathera.serving.DeploymentReplicaInfo]
    ) -> asyncio.Future[polymathera.serving.DeploymentReplicaInfo]:
        # Custom routing logic here
        # For example, route based on some request metadata or state of replicas
        # Return an asyncio.Future[polymathera.serving.DeploymentReplicaInfo] object
        # that will eventually resolve into a DeploymentReplicaInfo object representing
        # the selected replica.
        pass



# Also, to avoid ordering issues during startup, deployment happens in phases.
# Each deployed class can perform phase-specific initialization by decorating
# methods with the `@pre_initialize_deployment`, `@initialize_deployment`, and `@post_initialize_deployment` decorators.
# Each one of these methods is called on all replicas of all deployments in an application
# in some arbitrary (but deterministic) order (if the deployed class has these methods).
@polymathera.serving.deployment(
    name="MyDeployment1",
    routing_policy=MyCustomRequestRouter,  # Custom request router class
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_queue_length": 5,
        "scale_up_cooldown_s": 10,
        "scale_down_cooldown_s": 30,
    },
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
)
class MyDeployment1:
    def __init__(self):
        # Initialization code here
        pass

    @polymathera.serving.pre_initialize_deployment
    async def pre_initialize(self):
        # Code to run before the deployment is initialized
        pass

    @polymathera.serving.initialize_deployment
    async def initialize(self):
        # Code to run during the deployment initialization
        pass

    @polymathera.serving.post_initialize_deployment
    async def post_initialize(self):
        # Code to run after the deployment is initialized
        pass

    @polymathera.serving.endpoint
    async def my_method(self, arg1: str, arg2: int) -> str:
        # Method implementation here
        return f"Hello {arg1}, you passed {arg2}"


@polymathera.serving.deployment
class MyDeployment2:
    def __init__(self):
        # Initialization code here
        pass

    @polymathera.serving.endpoint
    async def another_method(self, arg: str) -> str:
        # Method implementation here
        d1 = polymathera.serving.get_deployment(
            polymathera.serving.get_my_app_name(),
            deployment_class=MyDeployment1,
            "MyDeployment1"
        )
        result = await d1.my_method(arg, 42)
        return f"{result}. Goodbye {arg} from MyDeployment2!"



# There is no single-ingress restriction.
# Each deployment has its own ingress proxy actor.

app = polymathera.serving.Application(name="MyApp")
d1 = app.add_deployment(
    MyDeployment1.bind()
)

# You can specify actor options and autoscaling config per deployment.
d2 = app.add_deployment(
    MyDeployment2.bind(),
    name="MyDeployment2",
    routing_policy=MyCustomRequestRouter,  # Custom request router class
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_queue_length": 5,
        "scale_up_cooldown_s": 10,
        "scale_down_cooldown_s": 30,
    },
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
)

app.run()


```
