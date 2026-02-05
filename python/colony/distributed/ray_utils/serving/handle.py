"""Deployment handles for calling deployments as clients."""

import logging
import uuid
from typing import Any, Type

import ray

from .models import (
    DeploymentConfig,
    DeploymentRequest,
    DeploymentResponse,
    DeploymentResponseStatus,
    ApplicationRegistry,
)
from .router import RequestRouter

logger = logging.getLogger(__name__)

# Global registry of active deployment proxies in this process
_DEPLOYMENT_PROXIES: dict[str, Any] = {}


class DeploymentHandle:
    """Handle for calling into a deployment.

    This acts as a local proxy that allows calling deployment methods
    as if they were local async functions. It handles request creation,
    routing through the proxy actor, and unpacking responses.

    Example:
        ```python
        handle = get_deployment("MyApp", "MyDeployment")
        result = await handle.my_method(arg1="value")
        ```
    """

    def __init__(self, app_name: str, deployment_name: str, proxy_actor_handle: Any, deployment_class: Type[Any] | None = None):
        """Initialize deployment handle.

        Args:
            app_name: Name of the application.
            deployment_name: Name of the deployment.
            proxy_actor_handle: Ray actor handle for the deployment proxy.
            deployment_class: Optional deployment class for accessing configuration.
        """
        self.app_name = app_name
        self.deployment_name = deployment_name
        self._proxy_actor_handle = proxy_actor_handle
        self._deployment_class = deployment_class
        self._endpoint_router_classes: dict[str, type[RequestRouter] | None] | None = None
        logger.debug(
            f"Created deployment handle for {app_name}/{deployment_name}"
        )

    def _get_endpoint_router_class(self, method_name: str) -> type[RequestRouter] | None:
        """Get router class for an endpoint (cached).

        Args:
            method_name: Name of the endpoint method.

        Returns:
            Router class if configured, None otherwise.
        """
        # Lazy-load endpoint routing classes from deployment class
        if self._endpoint_router_classes is None and self._deployment_class is not None:
            if hasattr(self._deployment_class, "__deployment_config__"):
                config: DeploymentConfig = self._deployment_class.__deployment_config__
                self._endpoint_router_classes = config.endpoint_router_classes
            else:
                self._endpoint_router_classes = {}

        # Return class if available
        if self._endpoint_router_classes:
            return self._endpoint_router_classes.get(method_name)

        return None

    def _get_endpoint_router_kwargs(self, method_name: str) -> dict[str, Any]:
        """Get router kwargs for an endpoint.

        Args:
            method_name: Name of the endpoint method.

        Returns:
            Router kwargs if configured, empty dict otherwise.
        """
        if self._deployment_class is not None:
            if hasattr(self._deployment_class, "__deployment_config__"):
                config: DeploymentConfig = self._deployment_class.__deployment_config__
                return config.get_endpoint_router_kwargs(method_name)

        return {}

    def _get_exception_class(self, error_type: str, error_module: str | None) -> Type[Exception] | None:
        """Dynamically import and return exception class.

        Uses importlib to dynamically import the exception class from its module.
        This is more robust and extensible than hardcoding exception mappings.

        Args:
            error_type: Name of the exception class (e.g., 'ResourceExhausted')
            error_module: Module path (e.g., 'polymathera.colony.agents.base', 'builtins')

        Returns:
            Exception class if successfully imported, None otherwise

        Security:
            Only imports from trusted prefixes (polymathera., builtins) to prevent
            arbitrary code execution.
        """
        if not error_module:
            logger.debug(f"No module info for exception type {error_type}")
            return None

        # Security: Only allow imports from trusted modules
        trusted_prefixes = ('polymathera.', 'builtins', 'ray.', 'asyncio.')
        if not any(error_module.startswith(prefix) for prefix in trusted_prefixes):
            logger.warning(
                f"Refusing to import exception from untrusted module: {error_module}.{error_type}"
            )
            return None

        try:
            import importlib
            module = importlib.import_module(error_module)
            exception_class = getattr(module, error_type, None)

            if exception_class and issubclass(exception_class, Exception):
                logger.debug(f"Successfully imported {error_module}.{error_type}")
                return exception_class
            else:
                logger.warning(
                    f"{error_module}.{error_type} is not a valid Exception class"
                )
                return None

        except (ImportError, AttributeError) as e:
            logger.warning(
                f"Failed to import exception {error_module}.{error_type}: {e}"
            )
            return None

    def __getattr__(self, method_name: str) -> Any:
        """Get a callable for invoking a deployment method.

        Args:
            method_name: Name of the method to call.

        Returns:
            Async function that invokes the method.
        """
        # Avoid infinite recursion for private attributes
        if method_name.startswith("_"):
            raise AttributeError(f"Attribute '{method_name}' not found")

        async def call_method(*args, **kwargs) -> Any:
            """Call the deployment method with given arguments.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.

            Returns:
                Result from the deployment method.

            Raises:
                Exception: If the deployment method raised an error.
            """
            # Get endpoint router class and kwargs
            router_class = self._get_endpoint_router_class(method_name)
            router_kwargs = self._get_endpoint_router_kwargs(method_name)

            # Extract routing hints from arguments
            if router_class is None or not hasattr(router_class, "extract_routing_hints"):
                routing_hints = None
            else:
                routing_hints = router_class.extract_routing_hints(
                    method_name=method_name,
                    args=args,
                    kwargs=kwargs,
                )
                # Add router_kwargs to routing hints if configured
                if router_kwargs and routing_hints is not None:
                    routing_hints.router_kwargs = router_kwargs

            # Create request with routing hints
            request = DeploymentRequest(
                request_id=str(uuid.uuid4()),
                method_name=method_name,
                args=args,
                kwargs=kwargs,
                routing_hints=routing_hints,
            )

            # Send request to proxy actor
            try:
                response: DeploymentResponse = await self._proxy_actor_handle.handle_request.remote(
                    request
                )
            except Exception as e:
                logger.error(
                    f"Error calling {self.deployment_name}.{method_name}: {e}",
                    exc_info=True,
                )
                raise

            # Check response status and unpack
            if response.status == DeploymentResponseStatus.SUCCESS:
                return response.result
            else:
                # Re-raise the error from the deployment with original exception type
                error_msg = (
                    f"Error in {self.deployment_name}.{method_name}: "
                    f"{response.error}\n"
                )
                if response.traceback:
                    error_msg += f"\nRemote traceback:\n{response.traceback}"

                logger.error(error_msg)

                # Try to re-raise with the original exception type
                if response.error_type and response.error_module:
                    exception_class = self._get_exception_class(
                        response.error_type, response.error_module
                    )
                    if exception_class:
                        raise exception_class(error_msg)

                # Fallback to RuntimeError if type unknown
                raise RuntimeError(error_msg)

        return call_method

    async def scale_deployment(self, target_replicas: int) -> bool:
        """Scale this deployment to target replica count.

        This is a management operation for programmatic scaling in response
        to application-level events (e.g., ResourceExhausted).

        Args:
            target_replicas: Desired number of replicas

        Returns:
            True if scaling was initiated successfully

        Raises:
            ValueError: If target_replicas is invalid

        Example:
            ```python
            handle = serving.get_deployment("MyApp", "MyDeployment")
            current_count = await handle.get_replica_count()
            await handle.scale_deployment(current_count + 1)
            ```
        TODO: Using two separate calls to get_replica_count then scale_deployment
        is not atomic. Consider adding an optional parameter to scale_deployment
        to scale by delta instead of absolute count.
        """
        if target_replicas < 1:
            raise ValueError(f"target_replicas must be >= 1, got {target_replicas}")

        try:
            await self._proxy_actor_handle._scale_to_target.remote(target_replicas)
            logger.info(
                f"Initiated scaling of {self.app_name}/{self.deployment_name} "
                f"to {target_replicas} replicas"
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to scale {self.app_name}/{self.deployment_name}: {e}",
                exc_info=True
            )
            return False

    async def get_replica_count(self) -> int:
        """Get current number of replicas for this deployment.

        Returns:
            Current replica count

        Example:
            ```python
            handle = serving.get_deployment("MyApp", "MyDeployment")
            count = await handle.get_replica_count()
            print(f"Current replica count: {count}")
            ```
        """
        replicas_info = await self._proxy_actor_handle.get_replicas_info.remote()
        return len(replicas_info)

    async def get_all_replica_property(self, property_name: str) -> dict[str, Any]:
        """Get a property value from all replicas of this deployment.

        This queries all healthy replicas for a specific property marked with
        @serving.replica_property(name). The property must be defined on the
        deployment class and decorated with @replica_property.

        Args:
            property_name: Name of the property to query (as defined in decorator).

        Returns:
            Dictionary mapping replica_id to property value.

        Raises:
            ValueError: If no method found with the specified property name.

        Example:
            ```python
            # Deployment has:
            # @serving.endpoint
            # @serving.replica_property("resource_usage")
            # async def get_resource_usage(self) -> dict[str, Any]: ...

            # Query the property:
            handle = serving.get_deployment("MyApp", "MyDeployment")
            usage = await handle.get_all_replica_property("resource_usage")
            # Returns: {
            #     "replica_0": {"cpu": 2.5, "memory": 1024, ...},
            #     "replica_1": {"cpu": 1.2, "memory": 512, ...},
            # }
            ```
        """
        return await self._proxy_actor_handle.get_all_replica_property.remote(property_name)


def get_deployment(
    app_name: str,
    deployment_name: str | None = None,
    deployment_class: Type[Any] | None = None,
) -> DeploymentHandle:
    """Get a handle to call into a deployment.

    This function discovers a deployment by name or class and returns
    a handle that can be used to call its methods.

    Args:
        app_name: Name of the application.
        deployment_name: Name of the deployment (optional if deployment_class provided).
        deployment_class: Deployment class (optional if deployment_name provided).

    Returns:
        DeploymentHandle for calling the deployment.

    Raises:
        ValueError: If deployment cannot be found.

    Example:
        ```python
        # Get by name
        handle = get_deployment("MyApp", deployment_name="MyDeployment")

        # Get by class
        handle = get_deployment("MyApp", deployment_class=MyDeployment)

        # Call methods
        result = await handle.my_method(arg="value")
        ```
    """
    # Determine deployment name
    if deployment_name is None and deployment_class is not None:
        if hasattr(deployment_class, "__deployment_config__"):
            deployment_name = deployment_class.__deployment_config__.name
        else:
            deployment_name = deployment_class.__name__
    elif deployment_name is None:
        raise ValueError("Either deployment_name or deployment_class must be provided")

    # Check cache first
    cache_key = f"{app_name}/{deployment_name}"
    if cache_key in _DEPLOYMENT_PROXIES:
        proxy_handle = _DEPLOYMENT_PROXIES[cache_key]
    else:
        proxy_namespace  = ApplicationRegistry.get_ray_actor_namespace(app_name)
        proxy_actor_name = ApplicationRegistry.get_deployment_proxy_actor_name(app_name, deployment_name)
        try:
            proxy_handle = ray.get_actor(proxy_actor_name, namespace=proxy_namespace)
            _DEPLOYMENT_PROXIES[cache_key] = proxy_handle
        except ValueError as e:
            raise ValueError(
                f"Deployment '{deployment_name}' not found in application '{app_name}'.\n{e}\n"
                f"Make sure the application is running and the deployment name is correct."
            ) from e

    return DeploymentHandle(app_name, deployment_name, proxy_handle, deployment_class)


def get_my_app_name() -> str:
    """Get the name of the current application.

    This is a helper function for deployments that need to discover other
    deployments in the same application.

    Returns:
        Name of the current application.

    Raises:
        RuntimeError: If not called from within a deployment.
    """
    # This would need to be set by the Application when it starts deployments
    # For now, we'll use an environment variable or task-local storage
    import os

    app_name = os.environ.get("POLYMATHERA_SERVING_CURRENT_APP")
    if not app_name:
        raise RuntimeError(
            "get_my_app_name() must be called from within a deployment context. "
            "The POLYMATHERA_SERVING_CURRENT_APP environment variable is not set."
        )
    return app_name

def get_my_deployment_name() -> str:
    """Get the name of the current deployment.

    This is a helper function for deployments that need to identify themselves.

    Returns:
        Name of the current deployment.

    Raises:
        RuntimeError: If not called from within a deployment.
    """
    import os

    # Try environment variable first (set by Application during deployment)
    deployment_name = os.environ.get("POLYMATHERA_SERVING_CURRENT_DEPLOYMENT")
    if not deployment_name:
        raise RuntimeError(
            "get_my_deployment_name() must be called from within a deployment context. "
            "The POLYMATHERA_SERVING_CURRENT_DEPLOYMENT environment variable is not set."
        )
    return deployment_name


def get_my_replica_id() -> str:
    """Get the replica ID of the current deployment instance.

    This is a helper function for deployments that need to identify their replica.

    Returns:
        Replica ID of the current deployment instance.
    Raises:
        RuntimeError: If not called from within a deployment.
    """
    import os

    # Try environment variable first (set by Application during deployment)
    replica_id = os.environ.get("POLYMATHERA_SERVING_CURRENT_REPLICA_ID")
    if not replica_id:
        raise RuntimeError(
            "get_my_replica_id() must be called from within a deployment context. "
            "The POLYMATHERA_SERVING_CURRENT_REPLICA_ID environment variable is not set."
        )
    return replica_id
