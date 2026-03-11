import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import ray
from prometheus_client import Counter, Histogram
from pydantic import BaseModel
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .params import (
    ray_global_code_analysis_config,
    ray_local_code_analysis_config,
    ray_params,
    ray_repo_stats_collector_config,
    ray_vmr_exec_config,
)

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


# Metrics — use REGISTRY to avoid duplicate registration errors in tests
from prometheus_client import REGISTRY as _REGISTRY

def _get_or_create_counter(name, description, labelnames):
    if name in _REGISTRY._names_to_collectors:
        return _REGISTRY._names_to_collectors[name]
    return Counter(name, description, labelnames)

def _get_or_create_histogram(name, description, labelnames):
    if name in _REGISTRY._names_to_collectors:
        return _REGISTRY._names_to_collectors[name]
    return Histogram(name, description, labelnames)

actor_creation_attempts = _get_or_create_counter(
    "actor_creation_attempts",
    "Number of actor creation attempts",
    ["environment", "actor_class", "status"],
)
actor_method_calls = _get_or_create_histogram(
    "actor_method_duration_seconds",
    "Duration of actor method calls",
    ["environment", "actor_class", "method"],
)
actor_errors = _get_or_create_counter(
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


