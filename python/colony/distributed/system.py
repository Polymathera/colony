from __future__ import annotations

import asyncio
import logging
import os
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
import weakref

import ray
from pydantic import BaseModel

from ..caching.simple import CacheConfig, DistributedSimpleCache

# from .orchestrator import Orchestrator, ResourceManager, VmrExecutionResourceAllocator
from ..config.manager import ConfigurationManager, EnvironmentType
from .ray_utils import serving
from .redis_utils.client import RedisClient
from .configs import (
    StorageConfig,
    SystemConfig,
)
from .state_management import StateManager
from .sys_info import get_sys_info
from ..utils import setup_logger

logger = setup_logger(__name__)


class _PolymatheraApiStub:
    """
    This class provides a common interface for all Polymathera agents to access shared resources
    and interact with the Polymathera system.
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
        self._cache_registry: set[DistributedSimpleCache] = set()

        # Initialize configuration manager
        self._config_manager = ConfigurationManager(
            config_path=config_path,
            environment=environment,
            distributed=True,  # Enable distributed configuration
        )

        # Initialize core attributes
        self.sys_config: SystemConfig | None = None

        self.name = None
        self.version = None
        self.architecture = None
        self.sys_info = {}

        # Initialize services with configuration
        self._service_registry = None
        self._observability = None
        self._security_manager = None
        self._messaging = None
        self._storage = None
        self._auth_service = None
        self._lb_registry = None
        self._chat_service = None
        self._distributed_config_manager = None
        self._object_registry = None
        self._embedding_client = None
        self._vector_store = None
        self._vector_etl_client = None
        # self.resource_manager = ResourceManager(self.config)
        # self.orchestrator = Orchestrator(self.config)
        self._knowledge_system = None
        self._shared_simple_cache = None

        # self.resource_allocator = VmrExecutionResourceAllocator(dist_config_manager=self._config_manager)
        self._initialized = False

        # Cache for StateManager instances (keyed by state_key)
        self._state_managers: dict[str, StateManager] = {}

    async def get_config_manager(self) -> ConfigurationManager:
        await self.initialize()
        return self._config_manager

    def get_knowledge_system_name(self):
        return "PolymatheraKnowledgeSystem"

    async def get_github_client(self):
        await self.initialize()
        from ..gitutils.clients import GitHubClient
        gh = GitHubClient()
        await gh.initialize()
        return gh

    async def get_gitlab_client(self):
        await self.initialize()
        from ..gitutils.clients import GitLabClient
        gl = GitLabClient()
        await gl.initialize()
        return gl

    async def get_knowledge_system(self) -> ray.actor.ActorHandle | None:
        # TODO: Make this return a local object instead of an actor handle
        await self.initialize()
        from ..memory.system import AgentMemorySystem

        knowledge_system = await polymathera_ray_cluster.create_or_discover_actor(
            AgentMemorySystem,
            name=self.get_knowledge_system_name(),
            create_if_not_found=self._head,
            # Constructor keyword arguments
            config=self.sys_config.memory,
        )
        if self._knowledge_system is None:
            if self._head:
                await knowledge_system.initialize.remote()
            self._knowledge_system = knowledge_system
        return knowledge_system

    async def get_storage(self):
        await self.initialize()
        if self._storage is None:
            from .storage import Storage

            config = await StorageConfig.check_or_get_component(self.sys_config.storage)
            self._storage = Storage(config)
            await self._storage.initialize()
        return self._storage

    async def get_service_registry(self):
        await self.initialize()
        if self._service_registry is None:
            from .registries import ServiceRegistry

            config = await ServiceRegistryConfig.check_or_get_component(
                self.sys_config.service_registry
            )
            self._service_registry = ServiceRegistry(config)
            await self._service_registry.initialize()
        return self._service_registry

    async def get_messaging(self):
        await self.initialize()
        if self._messaging is None:
            from .messaging import SystemMessaging

            config = await SystemMessagingConfig.check_or_get_component(
                self.sys_config.messaging
            )
            self._messaging = SystemMessaging(config)
            await self._messaging.initialize()
        return self._messaging

    async def get_security_manager(self):
        await self.initialize()
        if self._security_manager is None:
            from .security import SecurityManager

            config = await SecurityManagerConfig.check_or_get_component(
                self.sys_config.security
            )
            self._security_manager = SecurityManager(config)
            await self._security_manager.initialize()
        return self._security_manager

    async def get_auth_service(self):
        await self.initialize()
        if self._auth_service is None:
            from .security import AuthService

            config = await AuthServiceConfig.check_or_get_component(
                self.sys_config.auth_service
            )
            self._auth_service = AuthService(config)
            await self._auth_service.initialize()
        return self._auth_service

    async def get_observability(self):
        await self.initialize()
        if self._observability is None:
            from .observe import Observability

            config = await ObservabilityConfig.check_or_get_component(
                self.sys_config.observability
            )
            self._observability = Observability(config)
            await self._observability.initialize()
        return self._observability

    async def get_load_balancer_registry(self):
        await self.initialize()
        if self._lb_registry is None:
            from .load import LoadBalancerRegistry

            self._lb_registry = LoadBalancerRegistry(
                self.sys_config.load_balancer_domain
            )
            await self._lb_registry.initialize()
        return self._lb_registry

    async def get_chat_service(self):
        await self.initialize()
        if self._chat_service is None:
            from .chat import UserChatService

            config = await UserChatServiceConfig.check_or_get_component(self.sys_config.chat)
            self._chat_service = UserChatService(config)
            await self._chat_service.initialize()
            await self._chat_service.start()
        return self._chat_service

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
        self._cache_registry.add(cache)

        # Use weak reference callback to remove from registry when cache is garbage collected
        weakref.finalize(cache, self._cache_registry.discard, cache)

        return cache

    async def create_distributed_work_queue(
        self, queue_prefix: str, worker_id: str | None = None
    ):
        """Create a distributed work queue"""
        from .distributed_work_queue import DistributedQueueStore

        dqs = DistributedQueueStore(
            await self.get_redis_client(),
            namespace=queue_prefix,
            worker_id=worker_id
        )
        await dqs.initialize()
        return dqs

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

    async def get_embedding_client(self):
        await self.initialize()
        if self._embedding_client is None:
            from ..llms.inference.cluster.embedding import EmbeddingClient
            self._embedding_client = EmbeddingClient(self.sys_config.embedding)
            await self._embedding_client.initialize()
        return self._embedding_client

    async def get_vector_store(self):
        await self.initialize()
        if self._vector_store is None:
            from ..vectors.stores import VectorSearchBackendFactory
            self._vector_store = await VectorSearchBackendFactory.create_backend(self.sys_config.vector_store)
        return self._vector_store

    async def get_vector_etl_client(self):
        await self.initialize()
        if self._vector_etl_client is None:
            from ..vectors.etl import VectorStoreETLClient, VectorStoreETLConfig
            config = await VectorStoreETLConfig.check_or_get_component(self.sys_config.vector_etl)
            self._vector_etl_client = VectorStoreETLClient(config)
            await self._vector_etl_client.initialize()
        return self._vector_etl_client

    async def get_llm_client(self, **kwargs) -> ray.actor.ActorHandle:
        """Get an LLM client"""
        inference_manager = await self.get_inference_manager()
        cluster = await inference_manager.get_llm_cluster.remote()
        return await cluster.get_llm_client.remote(**kwargs)

    async def get_inference_manager(self) -> ray.actor.ActorHandle:
        await self.initialize()

        # Localize imports to break import cycle and isolate during unit testing
        from ..llms.inference.manager import MultiTenantInferenceManager

        inference_manager_name = self.get_multi_tenant_inference_manager_actor_name()
        remote_manager = await polymathera_ray_cluster.create_or_discover_actor(
            MultiTenantInferenceManager,
            name=inference_manager_name,
            create_if_not_found=True,
            # Constructor keyword arguments
            config=self.sys_config.inference_manager,
        )
        await remote_manager.initialize.remote()
        return remote_manager

    def get_multi_tenant_inference_manager_actor_name(
        self, vmr_id: str | None = None
    ) -> str:
        # if vmr_id:
        #     return f"multi_tenant_inference_manager:{vmr_id}"
        # TODO: Get it from the config manager
        return "multi_tenant_inference_manager"

    async def get_repo_ids(
        self, vmr_id: str, origin_url: str, branch: str = "main"
    ) -> list[str]:
        # TODO: Implement this
        pass

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

    def connect(self, *args, **kwargs: Any):
        pass

    def disconnect(self, *args, **kwargs: Any):
        pass

    def init(self, *args, **kwargs: Any):
        pass

    def shutdown(self, *args, **kwargs: Any):
        pass

    @property
    def id(self):
        return self._id  # Read-only property

    async def register(self):
        from .registries import ServiceInfo

        # Register this microservice with the service registry
        self.sys_info = await get_sys_info()
        service_info = ServiceInfo(
            id=self.id,
            name=self.__class__.__name__,
            host=await self.get_host(),
            port=await self.get_port(),
            health_check_endpoint="/health",
            version=await self.get_version(),
            capabilities=await self.get_capabilities(),
            node_address=await self.get_node_address(),
            container_id=await self.get_container_id(),
            image=await self.get_image(),
            environment=await self.get_environment(),
            last_heartbeat=datetime.now().timestamp(),
            status="active",
            resource_usage=await self.get_resource_usage(),
            dependencies=await self.get_dependencies(),
            api_endpoints=await self.get_api_endpoints(),
            metrics_endpoint=await self.get_metrics_endpoint(),
            logs_endpoint=await self.get_logs_endpoint(),
            tracing_endpoint=await self.get_tracing_endpoint(),
            config=await self.get_config(),  # TODO: Remove this
            tags=await self.get_tags(),  # TODO: Remove this
        )
        try:
            await self._service_registry.register_service(
                self.__class__.__name__, service_info
            )
            logger.info(
                f"Successfully registered {self.__class__.__name__} with the service registry"
            )
        except Exception as e:
            logger.error(
                f"Failed to register {self.__class__.__name__} with the service registry: {e!s}"
            )
            raise

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

        # Initialize core attributes
        self.sys_config = await self._config_manager.check_or_get_component(
            SystemConfig.CONFIG_PATH,
            SystemConfig,
        )
        logger.info(
            f"State storage config:\n{self.sys_config.distributed_state.storage.model_dump_json(indent=2)}"
        )

        self.name = self.sys_config.name
        self.version = self.sys_config.version
        self.architecture = self.sys_config.architecture

        # # Initialize all other subsystems
        # await self._service_registry.initialize()
        # await self._messaging.initialize()
        # await self._storage.initialize()

        # load_balancer_address = await self.load_balancer.get_address()
        # await self.lb_registry.register_load_balancer(load_balancer_address)

        # await self.register()
        # await asyncio.gather(self._start_heartbeat(), self._process_system_messages())

        self._initialized = True

    async def _process_system_messages(self):
        async for message in self.system_messaging.receive_system_messages():
            await self.handle_system_message(message)

    async def handle_system_message(self, message: str):
        # Handle system messages
        logger.info(f"Received system message: {message}")

    async def stop(self):
        """Stop the Polymathera system"""
        logger.info("Stopping Polymathera system and cleaning up resources...")

        # Clean up all registered cache instances
        if self._cache_registry:
            logger.info(f"Cleaning up {len(self._cache_registry)} cache instances...")
            cleanup_tasks = []
            for cache in list(self._cache_registry):  # Create copy to avoid modification during iteration
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
        if self._chat_service is not None:
            await self._chat_service.stop()

        logger.info("Polymathera system stopped successfully")






# TODO: Unify the configuration source so that all services are aligned
#  because the config is used to find files, etc.
# Initialize the Polymathera system
is_head = os.getenv("POLYMATHERA_HEAD", "false").lower() == "true"
logger.info(f"Initializing Polymathera {'head' if is_head else 'worker'} node")

polymathera = _PolymatheraApiStub(
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
