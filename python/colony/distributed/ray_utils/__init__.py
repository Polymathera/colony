import asyncio
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import ray
from ray.util.state import get_actor
from ray.exceptions import RayActorError, GetTimeoutError
from prometheus_client import Counter, Histogram
from pydantic import BaseModel
from ray.util.placement_group import (
    PlacementGroup,
    placement_group,
    placement_group_table,
    remove_placement_group,  # Import the function
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from ..schema.base_types import NodeId
from .network_monitor import NetworkMonitor
from .params import (
    ray_global_code_analysis_config,
    ray_local_code_analysis_config,
    ray_params,
    ray_repo_stats_collector_config,
    ray_vmr_exec_config,
)
from .topology_aware_scheduler import TopologyAwareScheduler

# Ray accepted actor option keys (kept in one place to filter YAML-driven options)
ACCEPTED_RAY_ACTOR_OPTIONS: set[str] = {
    "label_selector",
    "accelerator_type",
    "memory",
    "name",
    "num_cpus",
    "num_gpus",
    "object_store_memory",
    "placement_group",
    "placement_group_bundle_index",
    "placement_group_capture_child_tasks",
    "resources",
    "runtime_env",
    "scheduling_strategy",
    "_metadata",
    "enable_task_events",
    "_labels",
    "concurrency_groups",
    "lifetime",
    "max_concurrency",
    "max_restarts",
    "max_task_retries",
    "max_pending_calls",
    "namespace",
    "get_if_exists",
}

_all_params = {
    "ray_params": ray_params,
    "VmrExecutiveAgent": ray_vmr_exec_config,
    "RepoStatsCollectorAgent": ray_repo_stats_collector_config,
    "GlobalCodeAnalysisAgent": ray_global_code_analysis_config,
    "LocalCodeAnalysisAgent": ray_local_code_analysis_config,
}


class WorkerDeploymentOptions(BaseModel):
    """Options for deploying distributed workers"""

    model_config = {"arbitrary_types_allowed": True}

    num_cpus: float | None = None
    num_gpus: float | None = None
    memory: int | None = None  # In bytes
    object_store_memory: int | None = None
    resources: dict[str, float] | None = None
    accelerator_type: str | None = None
    placement_group: PlacementGroupSchedulingStrategy | None = None
    runtime_env: dict[str, Any] | None = None
    max_concurrency: int | None = None


logger = logging.getLogger(__name__)


# Metrics
actor_creation_attempts = Counter(
    "actor_creation_attempts_total",
    "Number of actor creation attempts",
    ["environment", "actor_class", "status"],
)
actor_method_calls = Histogram(
    "actor_method_duration_seconds",
    "Duration of actor method calls",
    ["environment", "actor_class", "method"],
)
actor_errors = Counter(
    "actor_errors_total",
    "Number of actor errors",
    ["environment", "actor_class", "error_type"],
)



def ray_logging_setup_func():
    """Set up Ray logging configuration for worker processes.

    This function is designed to be serializable for Ray's worker_process_setup_hook.
    It must be importable and self-contained.
    """
    import logging
    import sys
    import os

    # Enable ANSI color support across platforms
    try:
        import colorama
        colorama.init(autoreset=False, convert=True, strip=False)
    except ImportError:
        # If colorama isn't available, enable ANSI on Windows manually
        if os.name == 'nt':
            os.system('')  # Enable ANSI escape sequences on Windows

    # Ensure stdout can handle colors
    if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        # Set environment variables to force color output
        os.environ['TERM'] = os.environ.get('TERM', 'xterm-256color')
        os.environ['FORCE_COLOR'] = '1'

    # Configure Ray component loggers to suppress excessive output
    # Reduce noise for most Ray components, but keep Serve/ASGI logs at INFO to surface replica errors
    noise_loggers = [
        "ray",
        "ray.data",
        "ray.tune",
        # "ray.serve",
        "ray.rllib",
        "ray.train",
    ]
    for logger_name in noise_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Surface Serve and ASGI stacktraces in driver logs
    logging.getLogger("ray.serve").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("starlette").setLevel(logging.INFO)

    # Configure the root logger with basic handler that preserves colors
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)


def get_gpu_requirement(model_name):
    # This is a simplified example. In practice, you'd have a more comprehensive mapping.
    gpu_requirements = {
        "gpt2": 1,
        "gpt2-medium": 2,
        "gpt2-large": 4,
        "gpt2-xl": 8,
        # Add more models and their GPU requirements
    }
    return gpu_requirements.get(model_name, 1)  # Default to 1 if not specified


class ActorEnvironment(str, Enum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


@dataclass
class ActorCustomConfig:
    custom_disk_space: int = 10  # 10 GB
    num_gpus: int = 1


@dataclass
class ActorConfig:
    namespace: str | None = None
    name: str | None = None
    placement_group: PlacementGroup | None = None
    resources: dict[str, float] | None = None
    memory: int | None = None
    num_cpus: int | None = None
    num_gpus: int | None = None
    runtime_env: dict | None = None
    custom_config: ActorCustomConfig | None = None
    scheduling_strategy: str | None = None
    max_restarts: int = 3
    restart_delay: float = 1.0
    method_timeout: float = 30.0
    concurrency_groups: list[str] | None = None
    lifetime: str | None = None
    max_concurrency: int | None = None
    max_task_retries: int | None = None
    max_pending_calls: int | None = None
    get_if_exists: bool | None = None


class ActorError(Exception):
    """Base class for Actor-related exceptions"""
    pass


class ActorInitializationError(ActorError):
    """Raised when actor initialization fails"""
    pass


def get_ray_node_id() -> str:
    """Get the Ray node ID where this actor is running."""
    try:
        return ray.get_runtime_context().get_node_id()
    except Exception: # Broad except to catch RayNotInitializedError if called outside Ray actor
        return "unknown"


def get_ray_actor_id() -> str:
    """Get the Ray actor ID where this actor is running."""
    try:
        return ray.get_runtime_context().get_actor_id()

    except Exception: # Broad except to catch RayNotInitializedError if called outside Ray actor
        return "unknown"

polymathera_ray_actor_ping_method = "polymathera_ray_actor_ping"

def polymathera_ray_actor(cls: type[Any]) -> type[Any]:
    """Decorator to inject a ping method into a Ray actor to check if it's alive.
    Ray does propagate exceptions raised during actor method calls back to the caller,
    but only when you call `ray.get()` on the result.
    YourActor.remote() returns immediately with an ActorHandle. The `__init__` runs asynchronously later.
    Crucially, the exception is not automatically propagated back to the code that called `.remote()`.
    The original call that created the handle has already returned successfully.
    While it might seem intuitive for `__init__` exceptions to propagate like method
    exceptions, Ray's asynchronous design necessitates the explicit health check pattern
    to detect initialization failures reliably.
    """
    orig_init = cls.__init__
    def wrapped_init(self, *args, **kwargs):
        try:
            orig_init(self, *args, **kwargs)
            logger.info(f"Actor {cls.__name__} (id={get_ray_actor_id()}) initialized successfully on node {get_ray_node_id()}")
        except Exception as e:
            logger.error(f"Actor {cls.__name__} init failed: {e}", exc_info=True)
            raise  # Re-raise to let Ray mark as DEAD

    cls.__init__ = wrapped_init

    async def ping(self):
        return f"{cls.__name__} pong"

    def repr(self):
        """Used by Ray's logging system to distinguish between log messages from different Actors."""
        return f"{cls.__name__}(id={get_ray_actor_id()})" # (id={id(self)})"

    setattr(cls, polymathera_ray_actor_ping_method, ping)
    setattr(cls, "__repr__", repr)

    def get_node_id(self):
        return get_ray_node_id()
    setattr(cls, "get_node_id", get_node_id)
    return cls




# async def is_actor_alive(cls: type[Any], actor_handle: ray.actor.ActorHandle, timeout_seconds: float = 30.0) -> bool:
#     """Checks if a Ray actor is alive and responsive."""
#     if not hasattr(cls, polymathera_ray_actor_ping_method):
#         raise ActorInitializationError(f"Actor {cls.__name__} does not have a ping method: {polymathera_ray_actor_ping_method}")
#     if not actor_handle:
#         return False

#     try:
#         # Call the ping method with a timeout
#         logger.info(f"_____ Trying to ping actor {actor_handle}")
#         await asyncio.wait_for(getattr(actor_handle, polymathera_ray_actor_ping_method).remote(), timeout=timeout_seconds)
#         logger.info(f"_____ Pong received from actor {actor_handle}")
#         # If the call succeeds without timeout, the actor is alive
#         return True
#     except (RayActorError, asyncio.TimeoutError, GetTimeoutError) as e:
#         # RayActorError means the actor process has died.
#         # TimeoutError/GetTimeoutError means the actor didn't respond in time (could be dead or stuck).
#         logger.error(f"Actor check failed for handle {actor_handle}: {type(e).__name__} - {e}")

#         # For RayActorError, suggest checking Ray logs for the original constructor error
#         if isinstance(e, RayActorError):
#             logger.error(f"Actor {cls.__name__} died during initialization. Check Ray worker logs for the original constructor exception.")

#         return False
#     except Exception as e:
#         # Catch other unexpected errors
#         logger.error(f"Unexpected error checking actor {actor_handle}: {type(e).__name__} - {e}")
#         return False


async def is_actor_alive(cls: type[Any], actor_handle: ray.actor.ActorHandle, max_retries: int = 7, backoff_seconds: float = 4.0) -> bool:
    """Robust check if Ray actor is alive, using state API with retries."""
    actor_id = actor_handle._actor_id.hex()
    for attempt in range(1, max_retries + 1):
        try:
            state = ray.util.state.get_actor(actor_id)
            if state.state == "ALIVE":
                # Confirm with ping for responsiveness
                await asyncio.wait_for(getattr(actor_handle, polymathera_ray_actor_ping_method).remote(), timeout=5.0)
                logger.info(f"Actor {cls.__name__}(id={actor_id}) is ALIVE and responsive (attempt {attempt})")
                return True
            elif state.state in ["DEAD", "FAILED"]:
                logger.error(f"Actor {cls.__name__}(id={actor_id}) is {state.state}: {state.exit_detail}")
                return False
            else:
                logger.warning(f"Actor {cls.__name__}(id={actor_id}) in state {state.state} (attempt {attempt}) - retrying...")
        except Exception as e:
            logger.warning(f"State check failed (attempt {attempt}): {e} - retrying...")

        await asyncio.sleep(backoff_seconds * (2 ** (attempt - 1)))  # Exponential backoff

    logger.error(f"Actor {cls.__name__}(id={actor_id}) not alive after {max_retries} attempts")
    return False


def get_actor_error_details(actor_handle: ray.actor.ActorHandle) -> dict:
    """Fetch detailed error info for an actor using state API."""
    try:
        actor_id = actor_handle._actor_id.hex()
        state = ray.util.state.get_actor(actor_id)
        logger.info(f"Actor {actor_id} state: {state}")
        if state.state == "DEAD":
            return {
                "exit_detail": state.exit_detail,
                "error_time": state.error_time,
                "restart_count": state.restart_count
            }
        return {"status": state.state}
    except Exception as e:
        logger.error(f"Failed to get actor details: {e}")
        return {}

class _PolymatheraRayResources:
    """Additional custom Ray resources that can be specified at Ray startup."""

    def __init__(
        self,
        locality_aware_scheduling: bool = False,
        datacenter_aware_scheduling: bool = False,
    ):
        self.locality_aware_scheduling: bool = locality_aware_scheduling
        self.datacenter_aware_scheduling: bool = datacenter_aware_scheduling
        self._monitor = None
        self._monitor_task = None
        self.placement_groups: dict[NodeId, PlacementGroup] = {}

    async def get_available_gpus(self) -> int:
        return int(ray.available_resources().get("GPU", 0))

    def get_deployment_options(self, key: str) -> dict[str, Any]:
        if key in _all_params:
            # Copy to avoid mutating the global params
            raw_options = dict(_all_params[key])
            if "type" in raw_options and raw_options["type"] == "WorkerDeploymentOptions":
                raw_options.pop("type")
                # Validate known structure, then filter to only Ray-accepted keys
                try:
                    WorkerDeploymentOptions.model_validate(raw_options)
                except Exception:
                    # Ignore validation failures here; we'll filter keys regardless
                    pass

            # Filter to only accepted Ray actor options
            options: dict[str, Any] = {
                k: v for k, v in raw_options.items() if k in ACCEPTED_RAY_ACTOR_OPTIONS
            }
            return options
        return {}

    def get(self) -> dict[str, Any]:
        """Prefer nodes with low latency to current node"""
        current_node_id = ray.get_runtime_context().node_id
        return {NetworkMonitor.get_latency_resource_name(current_node_id): 10}

    async def initialize(
        self,
        locality_aware_scheduling: bool = False,
        datacenter_aware_scheduling: bool = False,
    ):
        topology_map = {"dc1": ["node1", "node2"], "dc2": ["node3", "node4"]}
        custom_scheduler = TopologyAwareScheduler(topology_map)

        ray_config = ray_params.get("ray_config", {})
        ray.init(_scheduler=custom_scheduler, **ray_config)

        self.locality_aware_scheduling = locality_aware_scheduling
        self.datacenter_aware_scheduling = datacenter_aware_scheduling
        # Dynamic custom resources:
        # For custom resources like disk space, Ray won't automatically enforce or
        # manage these resources. It's up to your application logic to respect these allocations.
        # You'll need to implement your own tracking and management of actual disk usage.
        # For the dynamic resources that depend on runtime context, add these after Ray initialization:
        # ray.get_runtime_context() can only be called after ray.init()
        current_node_id = ray.get_runtime_context().node_id
        ray.experimental.set_resource(
            NetworkMonitor.get_latency_resource_name(current_node_id), 10.0
        )

        self._monitor = NetworkMonitor.remote()
        self._monitor_task = self._monitor.run.remote()
        # Wait for the monitor to update resources
        await asyncio.sleep(65)

    async def cleanup(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._monitor:
            await asyncio.wait_for(ray.kill(self._monitor), timeout=10)

        # Remove placement groups
        for pg in self.placement_groups.values():
            remove_placement_group(pg)

    async def get_llm_deployment_args(
        self, model_name, node_id, scale_to_gpu_count: bool = False
    ):
        # Create or get placement group for this node
        pg = self.placement_groups.get(node_id)
        if not pg:
            pg = await self._create_placement_group_for_node(node_id)

        options = {}
        ### if scale_to_gpu_count:
        ###     # Scale the deployment based on available GPUs
        ###     # NOTE: Always read the most up-to-date GPU count because the number of GPUs can change dynamically
        ###     available_gpus = int(ray.available_resources().get("GPU", 0))
        ###     options["num_replicas"] = available_gpus
        options["num_gpus"] = get_gpu_requirement(model_name)
        # Add latency requirement to deployment options
        options["resources"] = {
            # TODO: Adjust threshold as needed
            NetworkMonitor.get_latency_resource_name(node_id): 10.0
        }
        return options

    async def _create_placement_group_for_node(self, node_id: str) -> PlacementGroup:
        """Create a placement group requesting the generic node_latency resource.
        TODO: Create a placement group that prefers nodes with low latency to target node
        """
        # Create a placement group with custom scheduling strategy
        pg = placement_group(
            bundles=[{
                "GPU": 1,  # Still require GPU
                "node_latency": 1 # Request the generic resource defined at startup
            }],
            # Remove the dynamic latency resource requirement
            # bundles=[{
            #     "GPU": 1,  # Resource requirement for the deployment
            #     # Additional custom resources can be specified at Ray startup
            #     NetworkMonitor.get_latency_resource_name(
            #         node_id
            #     ): 10,  # Prefer nodes with low latency to current node
            # }],
            strategy="SPREAD",  # or "PACK" or "STRICT_SPREAD" depending on your needs
            name=f"llm_deployment_pg_{node_id}",
        )
        logger.info(f"Creating placement group {pg.bundle_specs}...")
        await pg.ready()
        self.placement_groups[node_id] = pg
        return pg

    async def get_latency_to_node(
        self, source_node_id: str, target_node_id: str
    ) -> float:
        """Get the latency from the source node to the target node.

        Args:
            source_node_id: ID of the source node
            target_node_id: ID of the target node to measure latency to

        Returns:
            float: Latency in seconds between the nodes. Returns inf if latency is unknown.
        """
        latency_resource_name = f"latency_from_{source_node_id}:{target_node_id}"
        return float(ray.available_resources().get(latency_resource_name, float("inf")))

    async def get_vmr_agent_options(self, vmr_id: str | None = None) -> dict[str, Any]:
        # TODO: Keep track of which nodes have which repositories and using that
        # information to make placement decisions.
        if not self.locality_aware_scheduling:
            # No custom resources needed when locality scheduling is disabled
            return {}

        # Create a placement group for this task
        current_node_id = ray.get_runtime_context().node_id
        options = {
            "resources": {NetworkMonitor.get_latency_resource_name(current_node_id): 10}
        }
        if not self.datacenter_aware_scheduling:
            # Get the node ID where the repositories for this sub_vmr were last accessed
            preferred_node = current_node_id
            pg = placement_group(
                [
                    {
                        "CPU": 1,
                        "custom_disk_space": 10,
                        NetworkMonitor.get_latency_resource_name(
                            preferred_node
                        ): 10,  # Prefer nodes with low latency to current node
                    }
                ],
                strategy="STRICT_SPREAD",
            )
            await pg.ready()

            # Create the agent with custom placement
            options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True
            )
            options["runtime_env"] = {"working_dir": preferred_node}
            return options

        # Get the current node ID
        current_node_id = ray.get_runtime_context().node_id
        # Get the current node's rack and datacenter
        current_node_info = ray.nodes()
        current_node = [n for n in current_node_info if n["NodeID"] == current_node_id][
            0
        ]
        current_rack = current_node["Resources"].get("rack", 0)
        current_datacenter = current_node["Resources"].get("datacenter", 0)

        # Create a placement group that prefers the same rack and datacenter
        pg = placement_group(
            [
                {
                    "CPU": 1,
                    "custom_disk_space": 10,
                    f"rack_{current_rack}": 0.01,  # Soft constraint
                    f"datacenter_{current_datacenter}": 0.01,  # Soft constraint
                }
            ],
            strategy="STRICT_SPREAD",
        )
        await pg.ready()

        # Create the agent with custom placement
        options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_capture_child_tasks=True
        )
        return options

    async def create_or_discover_vmr_agent(
        self,
        cls: type[Any],
        name: str,
        actor_config: ActorConfig | None = None,
        create_if_not_found: bool = True,
        **kwargs,
    ) -> ray.actor.ActorHandle:
        options = await self.get_vmr_agent_options()

        return await self.create_or_discover_actor(
            cls, name, actor_config, create_if_not_found, options, **kwargs
        )

    async def discover_actor(self, name: str) -> ray.actor.ActorHandle:
        try:
            actor_ref = ray.get_actor(name)
            logger.info(f"Found existing actor {name}")
            return actor_ref
        except ValueError:
            logger.info(f"Actor {name} not found")
            return None

    async def create_or_discover_actor(
        self,
        cls: type[Any],
        name: str,
        actor_config: ActorConfig | None = None,
        create_if_not_found: bool = True,
        options: dict[str, Any] | None = None,
        ray_actor_options_key: str | None = None,
        namespace: str | None = None,
        get_if_exists: bool | None = None,
        **kwargs,
    ) -> ray.actor.ActorHandle:
        """Create new actor or discover existing one."""
        # Get and validate environment
        env = os.environ.get("POLYMATHERA_ENVIRONMENT", "prod").lower()
        try:
            environment = ActorEnvironment(env)
        except ValueError:
            logger.warning(f"Invalid environment '{env}', defaulting to 'prod'")
            environment = ActorEnvironment.PROD

        try:
            # Try to discover existing actor
            try:
                # Prefer namespace-aware lookup if provided
                if namespace is not None:
                    actor_ref = await asyncio.to_thread(
                        ray.get_actor, name, namespace
                    )
                else:
                    actor_ref = await asyncio.to_thread(ray.get_actor, name)
                logger.info(f"Found existing actor {name}")
                return actor_ref
            except ValueError:
                # Actor doesn't exist - this is expected, proceed to create it
                logger.info(f"Actor {name} not found, will create new one")
                pass

            # Actor not found
            if not create_if_not_found:
                return None

            if options is None:
                options = {}
            if ray_actor_options_key is None:
                ray_actor_options_key = cls.__name__
            # Merge YAML-driven options, filtered to accepted keys
            options.update(self.get_deployment_options(ray_actor_options_key))

            # Attach namespace/get_if_exists if provided by caller
            if namespace is not None:
                options["namespace"] = namespace
            if get_if_exists is not None:
                options["get_if_exists"] = get_if_exists

            # Check if GPU requirements can be satisfied
            available_gpus = ray.available_resources().get('GPU', 0)
            requested_gpus = options.get('num_gpus', 0)
            if requested_gpus > 0 and available_gpus < requested_gpus:
                # In production, fail fast with clear error
                raise ActorInitializationError(
                    f"Cannot create actor {name}: Requested {requested_gpus} GPUs but only {available_gpus} available. "
                    f"Ensure the Ray cluster has sufficient GPU resources or modify the actor configuration."
                )

            # Create new actor
            actor_ref = await self._initialize_actor(
                environment, cls, name, actor_config, options, **kwargs
            )
            logger.info(f"__________ Actor {name}, actor handle: {actor_ref}")
            if not await is_actor_alive(cls, actor_ref):
                error_details = get_actor_error_details(actor_ref)
                raise ActorInitializationError(f"Actor {name} is not alive: {error_details}")
            actor_creation_attempts.labels(
                environment=environment, actor_class=cls.__name__, status="success"
            ).inc()
            return actor_ref
        except Exception as e:
            actor_creation_attempts.labels(
                environment=environment, actor_class=cls.__name__, status="failure"
            ).inc()
            actor_errors.labels(
                environment=environment,
                actor_class=cls.__name__,
                error_type=type(e).__name__,
            ).inc()
            raise ActorInitializationError(
                f"Failed to initialize/discover actor: {e}"
            ) from e

    async def _initialize_actor(
        self,
        environment: ActorEnvironment,
        cls: type[Any],
        name: str,
        actor_config: ActorConfig | None = None,
        options: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Initialize either local object or Ray actor based on environment"""
        # if environment == ActorEnvironment.DEV:
        #     return cls(**kwargs) # Local object
        # else:
        #     pass

        # Configure Ray actor
        if options is None:
            options = {}
        options["name"] = name
        if actor_config:
            options["max_restarts"] = actor_config.max_restarts
            options["max_task_retries"] = actor_config.max_restarts

            # Add optional configurations
            # if actor_config.custom_config:
            #     options["resources"] = {
            #         "custom_disk_space": actor_config.custom_config.custom_disk_space,
            #         "num_gpus": actor_config.custom_config.num_gpus,
            #     }
            if actor_config.resources:
                if "resources" in options:
                    options["resources"].update(actor_config.resources)
                else:
                    options["resources"] = actor_config.resources
            if actor_config.memory:
                options["memory"] = actor_config.memory
            if actor_config.num_cpus:
                options["num_cpus"] = actor_config.num_cpus
            if actor_config.runtime_env:
                options["runtime_env"] = actor_config.runtime_env
            if actor_config.num_gpus:
                options["num_gpus"] = actor_config.num_gpus
            # Propagate additional supported options
            if actor_config.lifetime:
                options["lifetime"] = actor_config.lifetime
            if actor_config.max_concurrency:
                options["max_concurrency"] = actor_config.max_concurrency

        # Ensure only accepted Ray options are passed at the final call site too
        safe_options = {k: v for k, v in (options or {}).items() if k in ACCEPTED_RAY_ACTOR_OPTIONS}
        logger.info(f"__________ Initializing actor {name} with options: {safe_options}")
        return ray.remote(cls).options(**safe_options).remote(**kwargs)

    async def cleanup_actor(self, actor_ref: ray.actor.ActorHandle):
        """Cleanup actor resources"""
        try:
            await actor_ref.cleanup.remote()
            ray.kill(actor_ref)
        except Exception as e:
            logger.error(f"Failed to cleanup actor: {e}")
            raise

    def _options_for_resource(self, resource_name: str) -> dict[str, Any]:
        """Get Ray remote options for this resource."""
        return {
            # TODO: Get this from config and tune it.
            "resources": {
                resource_name: 0.01
            },  # Small value to act as a placement hint (soft constraint).
            "scheduling_strategy": "SPREAD",  # Spread actors across available nodes
        }

    async def _setup_localized_resource(
        self,
        resource_name: str,
        previous_node_id: NodeId | None = None,
        previous_last_used: float | None = None,
        previous_node_load: float | None = None,
        max_node_load: float | None = None,
        least_loaded_node_id: NodeId | None = None,
        select_least_loaded: bool = False,
    ):
        """Set up a unique custom resource for each repo for placement hints.
        Get the best node for a repo considering multiple factors.

        TODO: It would be beneficial to allocate a new GitRepoInferenceEngine actor to the same
        node where the same repo was previously processed. This can help with:
        - File system caching - The node's OS will likely have parts of the repo cached in memory
        - Reduced network traffic - Less data needs to be transferred over the network
        - Better locality - Related computations stay together

        TODO: You could also make this more sophisticated by:
        - Adding time-based decay to preferences
        - Considering node load/capacity
        - Adding fallback strategies
        - Caching repo data explicitly
        """
        # If we have a previous location for this repo
        if previous_node_id:
            # Check if previous location is still valid
            current_time = time.time()
            time_since_last_use = current_time - previous_last_used
            if (
                time_since_last_use < 3600
            ):  # 1 hour timeout (TODO: Make this configurable)
                if previous_node_load < max_node_load:
                    # Add the custom resource to that node
                    ray.get_runtime_context().node.set_resources(
                        {resource_name: 1.0}, node_id=previous_node_id
                    )
                    return

        if select_least_loaded:
            ray.get_runtime_context().node.set_resources(
                {resource_name: 1.0}, node_id=least_loaded_node_id
            )
            return

        # For new repos, add resource to all nodes
        for node_id in ray.nodes():
            ray.get_runtime_context().node.set_resources(
                {resource_name: 1.0}, node_id=node_id
            )

    async def create_actor_with_localized_resource(
        self,
        cls: type[Any],
        name: str,
        resource_name: str,
        actor_params: dict[str, Any],
        previous_node_id: NodeId | None = None,
        previous_last_used: float | None = None,
        previous_node_load: float | None = None,
        max_node_load: float | None = None,
        least_loaded_node_id: NodeId | None = None,
        select_least_loaded: bool = False,
    ) -> ray.actor.ActorHandle:
        """Create a new actor instance with proper placement."""
        await self._setup_localized_resource(
            resource_name,
            previous_node_id,
            previous_last_used,
            previous_node_load,
            max_node_load,
            least_loaded_node_id,
            select_least_loaded,
        )

        # Create actor with placement preferences
        actor = (
            ray.remote(cls)
            .options(name=name, **self._options_for_resource(resource_name))
            .remote(**actor_params)
        )

        return actor


polymathera_ray_cluster = _PolymatheraRayResources()

# Import cleanup utilities for easy access
from .cleanup import (
    RayActorCleanupUtility,
    ActorCleanupConfig,
    CleanupStrategy,
    cleanup_test_actors,
    cleanup_worker_handles,
    emergency_cleanup_namespace,
)
