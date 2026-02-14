from __future__ import annotations

import asyncio
import logging
import os
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Awaitable
import weakref

import ray
from pydantic import BaseModel

from .caching.simple import CacheConfig, DistributedSimpleCache
from .config.manager import ConfigurationManager, EnvironmentType
from .ray_utils import serving
from .redis_utils import RedisClient
from .configs import (
    StorageConfig,
    SystemConfig,
)
from .state_management import StateManager
from .sys_info import get_sys_info
from ..utils import setup_logger

logger = setup_logger(__name__)


class PolymatheraApp:
    """
    This class provides a common interface for all Polymathera agents to access shared resources
    and interact with the Polymathera system.

    This class only provides support for distributed configuration, storage and
    state management. Any other functionality (e.g., inference, vector stores, security,
    observability, messaging, chat, authentication, databases, ETL, etc.) can be added separately based on
    these core capabilities (configuration, storage, and state management).

    This class forms the basis for self-awareness and self-modeling in Polymathera agents, since
    it allows the agent to observe its own embodiment in the execution environment (e.g., its
    file system, network access, process, memory, etc.).
    """

    def __init__(
        self,
        config_path: str | None = None,
        environment: EnvironmentType | None = None,
        head: bool = False,
    ):
        self._id: str = str(uuid.uuid4())
        self._head = head
        self._redis_client = None

        # Registry to track all created cache instances for cleanup
        self._cache_registry: dict[str, DistributedSimpleCache] = set()
        self._shared_simple_cache = None
        # Cache for StateManager instances (keyed by state_key)
        self._state_managers: dict[str, StateManager] = {}

        # Initialize configuration manager
        self._config_manager = ConfigurationManager(
            config_path=config_path,
            environment=environment,
            distributed=False,  # Disable distributed configuration
        )

        # Initialize core attributes
        self.sys_config: SystemConfig | None = None

        self.name = None
        self.version = None
        self.architecture = None
        self.sys_info = {}

        self._storage = None

        self._initialized = False

    async def setup_ray(
        self,
        worker_env_vars: dict[str, str],
        working_dir: str | None = None,
        py_modules: list = None,
        ray_logging_setup_func: Callable[[], None] | None = None,
    ) -> None:
        """Initialize Ray cluster.

        This should be called on the driver node before any Ray operations are performed.

        Args:
            worker_env_vars: Environment variables to propagate to Ray worker processes
            working_dir: Directory to set as working directory for Ray workers (also distributed to workers).
                Used to distribute all files to workers (including autoscaled ones).
                This ensures workers have the latest code from the driver, not stale code from Docker image.
            ray_logging_setup_func: Optional function to set up logging in Ray worker processes
            py_modules: List of Python modules to be available in worker processes (adds to PYTHONPATH)
                All the polymathera.colony modules will be included by default, so this is only needed for external dependencies that are not in the Docker image.
        """
        if ray.is_initialized():
            return

        # Get environment variables for Ray workers
        logger.info(f"Propagating {len(worker_env_vars)} environment variables to Ray workers")
        logger.info(f"Worker env vars: {list(worker_env_vars.keys())}")

        # Try to connect to existing Ray cluster (on EKS)
        try:
            py_modules = py_modules or []
            from ... import colony
            py_modules.append(colony)  # Ensure polymathera.colony modules are included in worker processes

            # Try to connect to the cluster
            ray.init(
                address="auto",
                ignore_reinit_error=True,
                _system_config={"health_check_timeout_ms": 5000},
                ### # Enable structured JSON logging with actor/task metadata
                ### logging_config=ray.LoggingConfig(
                ###     encoding="JSON",
                ###     log_level="INFO", # "DEBUG", # Lower to DEBUG for init details
                ###     # Include additional metadata for debugging
                ###     additional_log_standard_attrs=['name', 'funcName', 'module', 'lineno'] #, 'process', 'levelname', 'exc_info']
                ### ),
                # Use log_to_driver=True for development/debugging
                log_to_driver=True,
                runtime_env={
                    "worker_process_setup_hook": ray_logging_setup_func,
                    # Ensure polymathera module is available in worker processes (adds to PYTHONPATH)
                    "py_modules": py_modules,
                    # Set working directory and distribute all files to workers (including autoscaled ones)
                    # This ensures workers have the latest code from the driver, not stale code from Docker image
                    "working_dir": working_dir,
                    # Propagate environment variables to worker processes
                    "env_vars": worker_env_vars,
                    # In a cluster environment, we can provide extra dependencies for Ray
                    # "pip": ["boto3", "opensearch-py", "torch", "transformers", "accelerate", "bitsandbytes", "optimum"]
                },
            )
            logger.info(f"Connected to Ray cluster: {ray.cluster_resources()}")

        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            raise

        # Don't shutdown if we connected to existing cluster
        # ray.shutdown()

    async def get_config_manager(self) -> ConfigurationManager:
        await self.initialize()
        return self._config_manager

    async def get_storage(self):
        await self.initialize()
        if self._storage is None:
            from .storage import Storage

            config = await StorageConfig.check_or_get_component(self.sys_config.storage)
            self._storage = Storage(config)
            await self._storage.initialize()
        return self._storage

    async def get_state_manager(
        self, state_type: type[BaseModel], state_key: str
    ) -> StateManager:
        """Get a state manager for a specific state type.

        Returns the same StateManager instance for calls with the same state_key
        to avoid initializing many instances in the same process.
        """
        await self.initialize()

        # Return cached instance if available
        if state_key in self._state_managers:
            return self._state_managers[state_key]

        # Create new StateManager instance
        logger.info(
            f"Creating new StateManager for state_key='{state_key}'. "
            f"State storage config:\n{self.sys_config.distributed_state.storage.model_dump_json(indent=2)}"
        )
        state_manager = StateManager(
            state_type, state_key, self.sys_config.distributed_state.storage
        )
        await state_manager.initialize()

        # Cache the instance
        self._state_managers[state_key] = state_manager

        return state_manager

    async def get_redis_client(self):
        if self._redis_client is None:
            self._redis_client = RedisClient(self.sys_config.redis)
            await self._redis_client.initialize()
        return self._redis_client

    async def get_shared_simple_cache(self) -> DistributedSimpleCache:
        """Get a shared distributed cache instance"""
        if self._shared_simple_cache is None:
            # Create a shared cache with a default namespace
            self._shared_simple_cache = await self.create_distributed_simple_cache(
                namespace="shared"
            )
        return self._shared_simple_cache

    async def create_distributed_simple_cache(
        self, namespace: str, config: CacheConfig | None = None
    ) -> DistributedSimpleCache:
        """Create a distributed cache instance"""
        config = await CacheConfig.check_or_get_component(config)
        cache = DistributedSimpleCache(
            namespace=namespace,
            redis_client=await self.get_redis_client(),
            config=config,
        )
        await cache.initialize()

        # Register the cache for cleanup using weak reference to avoid circular references
        self._cache_registry[id(cache)] = cache

        # Use weak reference callback to remove from registry when cache is garbage collected
        weakref.finalize(cache, self._cache_registry.pop, id(cache))

        return cache

    async def get_or_create_ray_service(
        self,
        svc_class: type,
        service_name: str,
    ) -> serving.DeploymentHandle:
        """
        Get or create a shared deployment for any service class using polymathera.rayutils.serving.
        This service is shared across agents and persists for the lifetime
        of the Ray cluster to avoid repeated deployment overhead.

        Uses Application.start() which handles distributed state management atomically.

        Args:
            svc_class: The service class to deploy (must be decorated with @serving.deployment)
            service_name: Name for the service (used as app name)

        Returns:
            DeploymentHandle for the service
        """
        # Use service_name as app name
        app_name = service_name

        # Check if class has bind method and deployment config
        if not hasattr(svc_class, 'bind'):
            raise AttributeError(
                f"Service class '{svc_class.__name__}' does not have a 'bind' method. "
                f"Make sure it is decorated with @serving.deployment."
            )

        if not hasattr(svc_class, '__deployment_config__'):
            raise AttributeError(
                f"Service class '{svc_class.__name__}' does not have __deployment_config__. "
                f"Make sure it is decorated with @serving.deployment."
            )

        # Get deployment name from decorator config (NOT from class name)
        # This respects the 'name' parameter in @serving.deployment(name=...)
        deployment_config = svc_class.__deployment_config__
        deployment_name = deployment_config.name or svc_class.__name__

        logger.info(
            f"Service class '{svc_class.__name__}' has deployment name '{deployment_name}' "
            f"from decorator config"
        )

        # Try to get existing deployment using distributed service discovery
        try:
            handle = serving.get_deployment(app_name, deployment_name)
            logger.info(f"Found existing deployment '{deployment_name}' in app '{app_name}'")
            return handle
        except Exception as e:
            logger.info(f"Deployment '{deployment_name}' in app '{app_name}' not found: {e}. Will create it.")

        # Deployment doesn't exist, create it using Application
        # Application.start() handles distributed state management atomically
        logger.info(f"Creating new application '{app_name}' with deployment '{deployment_name}'")

        try:
            # Create application and add deployment
            app = serving.Application(name=app_name)
            logger.info(f"Created Application instance for '{app_name}'")

            bound_deployment = svc_class.bind()
            logger.info(f"Successfully called bind() on {svc_class.__name__}")

            app.add_deployment(bound_deployment)
            logger.info(f"Added deployment to application")

            # Start the application (handles distributed state atomically)
            logger.info(f"Starting application '{app_name}'...")
            await app.start()
            logger.info(f"Application '{app_name}' started successfully")

            # Get deployment handle using distributed service discovery
            handle = serving.get_deployment(app_name, deployment_name)
            logger.info(f"Successfully retrieved deployment handle for '{deployment_name}'")
            return handle

        except Exception as e:
            logger.error(
                f"Failed to create/start application '{app_name}' with deployment '{deployment_name}': {e}",
                exc_info=True
            )
            raise

    async def delete_ray_service(self, service_name: str) -> bool:
        """
        Delete the Ray service (using service_name as Polymathera Ray serving app name).
        This function is useful for cleanup during tests to reduce resource usage
        and associated costs.

        Uses Application.stop_by_name_if_exists() which handles distributed state management.

        Returns:
            bool: True if service was successfully deleted or didn't exist, False if deletion failed
        """
        return await serving.Application.stop_by_name_if_exists(service_name)

    async def get_git_file_storage_prefix(self):
        storage = await self.get_storage()
        return await storage.git_storage.get_root_path()

    async def normalize_file_path(self, file_path: str) -> str:
        """Remove the git file storage prefix from the file path.
        This is useful for many distributed operations:
        - To make IDs of file-related objects (e.g., shards) deterministic and dependent only on the
          object's intrinsic properties (not Polymathera's file system mount point).
        - To make persistent data (e.g., dependency graphs) that use file paths as identifiers
          in their keys consistent across executions.
        """
        prefix = str(await self.get_git_file_storage_prefix())
        return file_path.replace(prefix, "")

    async def denormalize_file_path(self, normalized_file_path: str) -> str:
        """Denormalize a normalized file path"""
        prefix = await self.get_git_file_storage_prefix()
        # Remove leading slash from normalized_file_path if present to avoid it being treated as absolute
        clean_normalized_path = normalized_file_path.lstrip('/')
        fpath = str(prefix / clean_normalized_path)
        return fpath

    @property
    def id(self):
        return self._id  # Read-only property

    async def initialize(self) -> None:
        """Initialize all subsystems"""
        # logger.info(f"=========== Initializing Polymathera {'head' if self._head else 'worker'} node ===========")
        if self._initialized:
            # logger.info(f"=========== Polymathera {'head' if self._head else 'worker'} node already initialized ===========")
            return
        logger.info(f"=========== Initializing Polymathera {'head' if self._head else 'worker'} node ===========")
        logger.info(f"Stack trace: {traceback.format_exc()}")

        # Initialize configuration manager first
        await self._config_manager.initialize()
        config_manager = self._config_manager

        # Initialize core attributes
        self.sys_config = await config_manager.check_or_get_component(
            SystemConfig.CONFIG_PATH,
            SystemConfig,
        )
        logger.info(
            f"State storage config:\n{self.sys_config.distributed_state.storage.model_dump_json(indent=2)}"
        )

        self.name = self.sys_config.name
        self.version = self.sys_config.version
        self.architecture = self.sys_config.architecture

        self._initialized = True

    async def stop(self):
        """Stop the Polymathera system"""
        logger.info("Stopping Polymathera system and cleaning up resources...")

        # Clean up all registered cache instances
        if self._cache_registry:
            logger.info(f"Cleaning up {len(self._cache_registry)} cache instances...")
            cleanup_tasks = []
            for cache in list(self._cache_registry.values()):  # Create copy to avoid modification during iteration
                try:
                    cleanup_tasks.append(cache.cleanup())
                except Exception as e:
                    logger.error(f"Error creating cleanup task for cache {cache.namespace}: {e}")

            if cleanup_tasks:
                try:
                    await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                    logger.info("All cache instances cleaned up successfully")
                except Exception as e:
                    logger.error(f"Error during cache cleanup: {e}")

            self._cache_registry.clear()

        await self._config_manager.cleanup()  # Cleanup config manager resources

        logger.info("Polymathera system stopped successfully")






# TODO: Unify the configuration source so that all services are aligned
#  because the config is used to find files, etc.
# Initialize the Polymathera system
is_head = os.getenv("POLYMATHERA_HEAD", "false").lower() == "true"
logger.info(f"Initializing Polymathera {'head' if is_head else 'worker'} node")

polymathera = PolymatheraApp(
    config_path=os.getenv("POLYMATHERA_CONFIG"),
    environment=EnvironmentType(
        os.getenv("POLYMATHERA_ENV", EnvironmentType.DEVELOPMENT)
    ),
    head=is_head,  # Determine head status from environment
)



async def initialize_polymathera():
    """Initialize the Polymathera system asynchronously"""
    await polymathera.initialize()


# Initialize polymathera based on context, handling both sync and async contexts
# from ..utils import run_sync
# init_task = run_sync(initialize_polymathera())
