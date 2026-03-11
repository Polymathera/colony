"""Application for managing multiple deployments."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Type

import ray

from .decorators import BoundDeployment
from .models import ApplicationInfo, ApplicationRegistry, AutoscalingConfig, DeploymentProxyInfo, HealthCheckConfig, LoggingConfig, RequestRouter
from .proxy import DeploymentProxyRayActor
from ...state_management import StateManager

logger = logging.getLogger(__name__)


class DeploymentInfo:
    """Information about a deployment in an application.

    This is a thin wrapper around BoundDeployment that holds:
    1. The bound deployment (class + args + base config from decorator)
    2. Instance-specific config overrides (optional)
    """

    def __init__(
        self,
        bound_deployment: BoundDeployment,
        name: str | None = None,
        default_router_class: Type[RequestRouter] | None = None,
        autoscaling_config: dict[str, Any] | None = None,
        ray_actor_options: dict[str, Any] | None = None,
        max_concurrency: int | None = None,
        logging_config: LoggingConfig | None = None,
        health_check_config: dict[str, Any] | None = None,
    ):
        """Initialize deployment info.

        Args:
            bound_deployment: Bound deployment with class and args.
            name: Optional override for deployment name.
            default_router_class: Optional override for default router class.
            autoscaling_config: Optional override for autoscaling config.
            ray_actor_options: Optional override for ray actor options.
            max_concurrency: Optional override for max concurrent requests per replica.
            logging_config: Optional override for logging configuration.
            health_check_config: Optional override for health check config.
        """
        self.bound_deployment = bound_deployment

        # Store instance-specific overrides as a separate config
        self.instance_config_overrides = {
            "name": name,
            "default_router_class": default_router_class,
            "autoscaling_config": autoscaling_config,
            "ray_actor_options": ray_actor_options,
            "max_concurrency": max_concurrency,
            "logging_config": logging_config,
            "health_check_config": health_check_config,
        }

        # Will be set when deployment is started
        self.proxy_actor_handle: Any = None

    @property
    def deployment_class(self):
        """Get the deployment class from bound deployment."""
        return self.bound_deployment.deployment_class

    @property
    def args(self):
        """Get constructor args from bound deployment."""
        return self.bound_deployment.args

    @property
    def kwargs(self):
        """Get constructor kwargs from bound deployment."""
        return self.bound_deployment.kwargs

    @property
    def base_config(self):
        """Get the base config from the deployment decorator."""
        return self.bound_deployment.config

    @property
    def name(self) -> str:
        """Get the effective deployment name (instance override or decorator config or class name)."""
        return (
            self.instance_config_overrides.get("name")
            or self.base_config.name
            or self.deployment_class.__name__
        )

    @property
    def default_router_class(self) -> Type[RequestRouter] | None:
        """Get the effective routing policy."""
        return self.instance_config_overrides.get("default_router_class") or self.base_config.router_class

    @property
    def ray_actor_options(self) -> dict[str, Any]:
        """Get the effective ray actor options."""
        return self.instance_config_overrides.get("ray_actor_options") or self.base_config.ray_actor_options or {}

    @property
    def logging_config(self) -> LoggingConfig | None:
        """Get the effective logging config."""
        return self.instance_config_overrides.get("logging_config") or self.base_config.logging_config

    @property
    def autoscaling_config(self) -> AutoscalingConfig:
        """Get the effective autoscaling config (merged base + overrides)."""
        # Build autoscaling config with max_concurrency
        base_autoscaling = self.base_config.autoscaling_config or {}
        override_autoscaling = self.instance_config_overrides.get("autoscaling_config") or {}
        merged_autoscaling = {**base_autoscaling, **override_autoscaling}

        # Handle max_concurrency override
        instance_max_concurrency = self.instance_config_overrides.get("max_concurrency")
        final_max_concurrency = instance_max_concurrency if instance_max_concurrency is not None else self.base_config.max_concurrency
        if final_max_concurrency is not None:
            merged_autoscaling["max_concurrency"] = final_max_concurrency

        return AutoscalingConfig(**merged_autoscaling)

    @property
    def health_check_config(self) -> HealthCheckConfig:
        """Get the effective health check config (merged base + overrides)."""
        base_hc = self.base_config.health_check_config or {}
        override_hc = self.instance_config_overrides.get("health_check_config") or {}
        merged_hc = {**base_hc, **override_hc}
        return HealthCheckConfig(**merged_hc)


class Application:
    """Application that manages multiple deployments.

    An Application is a collection of deployments that work together.
    It handles starting, stopping, and managing the lifecycle of all
    deployments.

    Example:
        ```python
        app = Application(name="MyApp")

        app.add_deployment(MyDeployment1.bind())
        app.add_deployment(
            MyDeployment2.bind(),
            autoscaling_config={"min_replicas": 2}
        )

        await app.run()
        ```
    """

    def __init__(self, name: str, health_check_interval_s: float = 30.0):
        """Initialize application.

        Args:
            name: Name of the application.
            health_check_interval_s: Interval in seconds between health checks of proxy actors.
        """
        self.name = name
        self.deployments: list[DeploymentInfo] = []
        self._running = False
        self._state_manager: StateManager | None = None
        self._health_check_interval_s = health_check_interval_s
        self._health_monitor_task: asyncio.Task | None = None
        logger.info(f"Created application '{name}'")

    @staticmethod
    async def create_state_manager() -> StateManager:
        """Get or create the state manager for application registry."""
        from ... import get_polymathera
        polymathera = get_polymathera()
        state_manager = await polymathera.get_state_manager(
            state_type=ApplicationRegistry,
            state_key="polymathera.colony.distributed.ray_utils.serving.apps"
        )
        return state_manager

    async def _get_state_manager(self):
        """Get or create the state manager for application registry."""
        if self._state_manager is None:
            self._state_manager = await Application.create_state_manager()
        return self._state_manager

    def add_deployment(
        self,
        bound_deployment: BoundDeployment,
        name: str | None = None,
        default_router_class: Type[RequestRouter] | None = None,
        autoscaling_config: dict[str, Any] | None = None,
        ray_actor_options: dict[str, Any] | None = None,
        max_concurrency: int | None = None,
        logging_config: LoggingConfig | None = None,
        health_check_config: dict[str, Any] | None = None,
    ) -> Application:
        """Add a deployment to the application.

        Args:
            bound_deployment: Bound deployment from Class.bind().
            name: Optional override for deployment name.
            default_router_class: Optional override for default router class.
            autoscaling_config: Optional override for autoscaling config.
            ray_actor_options: Optional override for ray actor options.
            max_concurrency: Optional override for max concurrent requests per replica.
            logging_config: Optional override for logging configuration.
            health_check_config: Optional override for health check config.

        Returns:
            Self for chaining.
        """
        deployment_info = DeploymentInfo(
            bound_deployment=bound_deployment,
            name=name,
            default_router_class=default_router_class,
            autoscaling_config=autoscaling_config,
            ray_actor_options=ray_actor_options,
            max_concurrency=max_concurrency,
            logging_config=logging_config,
            health_check_config=health_check_config,
        )
        self.deployments.append(deployment_info)
        logger.info(f"Added deployment '{deployment_info.name}' to application '{self.name}'")
        return self

    async def start(self) -> None:
        """Start all deployments in the application.

        This creates proxy actors for each deployment and initializes them.
        Uses distributed state management to ensure atomic app creation across the cluster.
        """
        if self._running:
            logger.warning(f"Application '{self.name}' is already running")
            return

        logger.info(f"Starting application '{self.name}' with {len(self.deployments)} deployments")

        # Get state manager
        state_manager = await self._get_state_manager()

        # Register app in distributed state atomically
        async for registry in state_manager.write_transaction(max_retries=3):
            # Check if app already exists
            app_info = registry.get_app(self.name)
            if app_info:
                logger.info(f"Application '{self.name}' already registered, updating deployments")
            else:
                # Create new app info
                app_info = ApplicationInfo(
                    app_name=self.name,
                    deployments={},
                    created_at=time.time()
                )

            # Start all deployments
            proxy_namespace = ApplicationRegistry.get_ray_actor_namespace(self.name)
            for deployment_info in self.deployments:
                proxy_actor_name = ApplicationRegistry.get_deployment_proxy_actor_name(self.name, deployment_info.name)

                # Check if deployment already exists in registry
                if deployment_info.name in app_info.deployments:
                    logger.info(f"Deployment '{deployment_info.name}' already exists in registry, verifying actor exists")

                    # Verify the proxy actor actually exists in Ray (in the app's namespace)
                    try:
                        ray.get_actor(proxy_actor_name, namespace=proxy_namespace)
                        logger.info(f"Proxy actor '{proxy_actor_name}' verified to exist, skipping deployment start")
                        continue
                    except ValueError:
                        logger.warning(
                            f"Deployment '{deployment_info.name}' exists in registry but proxy actor "
                            f"'{proxy_actor_name}' not found in Ray. Will recreate the actor."
                        )
                        # Fall through to start the deployment

                # Start the deployment
                await self._start_deployment(deployment_info)

                # Register deployment in app info
                app_info.deployments[deployment_info.name] = DeploymentProxyInfo(
                    deployment_name=deployment_info.name,
                    proxy_actor_name=proxy_actor_name
                )

            # Register/update app in registry
            registry.register_app(app_info)
            logger.info(f"Registered application '{self.name}' in distributed state")

        # Notify all deployments that the application is ready.
        # This triggers @on_app_ready hooks, which can safely discover
        # sibling deployment handles since all proxies are now running.
        logger.info(
            f"\n"
            f"        ╔══════════════════════════════════════════════════════╗\n"
            f"        ║  🔮 Notifying all deployments that application       ║\n"
            f"        ║  '{self.name}' is ready                              ║\n"
            f"        ╚══════════════════════════════════════════════════════╝"
        )
        for deployment_info in self.deployments:
            if deployment_info.proxy_actor_handle is not None:
                try:
                    await deployment_info.proxy_actor_handle.notify_app_ready.remote()
                except Exception as e:
                    logger.error(
                        f"Error notifying deployment '{deployment_info.name}' of app ready: {e}",
                        exc_info=True,
                    )

        # Start health monitoring background task
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        logger.info(f"Started health monitor for application '{self.name}'")

        self._running = True
        logger.info(f"Application '{self.name}' started successfully")

    async def _start_deployment(self, deployment_info: DeploymentInfo) -> None:
        """Start a single deployment.

        Args:
            deployment_info: Information about the deployment to start.
        """
        logger.info(f"Starting deployment '{deployment_info.name}'")

        # Prepare actor options with runtime_env for app name and deployment name
        proxy_ray_actor_options = {
            "runtime_env": {
                "env_vars": {
                    "POLYMATHERA_SERVING_CURRENT_APP": self.name,
                    "POLYMATHERA_SERVING_CURRENT_DEPLOYMENT": deployment_info.name
                }
            }
        }

        # Create proxy actor with explicit namespace for this app
        proxy_actor_name = ApplicationRegistry.get_deployment_proxy_actor_name(self.name, deployment_info.name)
        proxy_namespace = ApplicationRegistry.get_ray_actor_namespace(self.name)
        proxy_actor = (
            ray.remote(DeploymentProxyRayActor)
            .options(name=proxy_actor_name, namespace=proxy_namespace, **proxy_ray_actor_options)
            .remote(
                deployment_name=deployment_info.name,
                deployment_class=deployment_info.deployment_class,
                app_name=self.name,
                deployment_init_args=deployment_info.args,
                deployment_init_kwargs=deployment_info.kwargs,
                default_router_class=deployment_info.default_router_class,
                autoscaling_config=deployment_info.autoscaling_config,
                ray_actor_options=deployment_info.ray_actor_options,
                logging_config=deployment_info.logging_config,
                health_check_config=deployment_info.health_check_config,
            )
        )

        # Initialize the proxy (creates initial replicas and starts monitoring)
        await proxy_actor.initialize.remote()

        deployment_info.proxy_actor_handle = proxy_actor
        logger.info(f"Deployment '{deployment_info.name}' started")

    async def _health_monitor_loop(self) -> None:
        """Background task that monitors health of all proxy actors and reconciles state.

        This ensures that:
        1. Dead proxy actors are detected and recreated
        2. Distributed state stays synchronized with actual Ray actor state
        3. The application self-heals from failures
        """
        logger.info(f"Health monitor started for application '{self.name}'")

        while self._running:
            try:
                await asyncio.sleep(self._health_check_interval_s)

                if not self._running:
                    break

                logger.debug(f"Running health check for application '{self.name}'")

                # Check each deployment
                proxy_namespace = ApplicationRegistry.get_ray_actor_namespace(self.name)
                for deployment_info in self.deployments:
                    proxy_actor_name = ApplicationRegistry.get_deployment_proxy_actor_name(self.name, deployment_info.name)

                    try:
                        # Try to get the actor handle (in app's namespace)
                        proxy_handle = ray.get_actor(proxy_actor_name, namespace=proxy_namespace)

                        # Verify it's actually alive by calling get_stats
                        await proxy_handle.get_stats.remote()

                        # Actor is healthy
                        logger.debug(f"Proxy actor '{proxy_actor_name}' is healthy")

                    except (ValueError, Exception) as e:
                        # Actor is dead or unreachable
                        logger.error(
                            f"Proxy actor '{proxy_actor_name}' is dead or unreachable: {e}. "
                            f"Attempting to recreate..."
                        )

                        try:
                            # Try to kill the old actor if it still exists (but is unresponsive)
                            try:
                                old_actor = ray.get_actor(proxy_actor_name, namespace=proxy_namespace)
                                ray.kill(old_actor, no_restart=True)
                                logger.info(f"Killed unresponsive proxy actor '{proxy_actor_name}'")
                                # Give Ray time to clean up the actor registration
                                await asyncio.sleep(1.0)
                            except ValueError:
                                # Actor doesn't exist, no need to kill
                                logger.debug(f"Actor '{proxy_actor_name}' not registered, no need to kill")
                            except Exception as kill_error:
                                logger.warning(f"Error killing old actor '{proxy_actor_name}': {kill_error}")

                            # Recreate the deployment
                            await self._start_deployment(deployment_info)

                            # Update distributed state
                            state_manager = await self._get_state_manager()
                            async for registry in state_manager.write_transaction(max_retries=3):
                                app_info = registry.get_app(self.name)
                                if app_info:
                                    app_info.deployments[deployment_info.name] = DeploymentProxyInfo(
                                        deployment_name=deployment_info.name,
                                        proxy_actor_name=proxy_actor_name
                                    )
                                    registry.register_app(app_info)

                            logger.info(f"Successfully recreated proxy actor '{proxy_actor_name}'")

                        except Exception as recreate_error:
                            logger.error(
                                f"Failed to recreate proxy actor '{proxy_actor_name}': {recreate_error}",
                                exc_info=True
                            )

            except Exception as e:
                logger.error(f"Error in health monitor loop for application '{self.name}': {e}", exc_info=True)

        logger.info(f"Health monitor stopped for application '{self.name}'")

    async def stop(self) -> None:
        """Stop all deployments and cleanup resources.

        Unregisters the application from distributed state.
        """
        if not self._running:
            logger.warning(f"Application '{self.name}' is not running")
            return

        logger.info(f"Stopping application '{self.name}'")

        # Signal health monitor to stop
        self._running = False

        # Wait for health monitor to finish
        if self._health_monitor_task:
            try:
                await asyncio.wait_for(self._health_monitor_task, timeout=5.0)
                logger.info("Health monitor task stopped successfully")
            except asyncio.TimeoutError:
                logger.warning("Health monitor task did not stop within timeout, cancelling")
                self._health_monitor_task.cancel()
                try:
                    await self._health_monitor_task
                except asyncio.CancelledError:
                    pass

        # Stop all deployments
        for deployment_info in self.deployments:
            await self._stop_deployment(deployment_info)

        # Unregister from distributed state
        try:
            state_manager = await self._get_state_manager()
            async for registry in state_manager.write_transaction(max_retries=3):
                app_info = registry.unregister_app(self.name)
                if app_info:
                    logger.info(f"Unregistered application '{self.name}' from distributed state")
                else:
                    logger.warning(f"Application '{self.name}' was not found in distributed state")
        except Exception as e:
            logger.error(f"Error unregistering application '{self.name}' from distributed state: {e}")

        logger.info(f"Application '{self.name}' stopped")

    @staticmethod
    async def stop_by_name_if_exists(app_name: str) -> bool:
        """Stop an application by name if it exists in the cluster.

        This is a static method that looks up the application in distributed state
        and stops all its deployments.

        Args:
            app_name: Name of the application to stop.

        Returns:
            bool: True if application was stopped or didn't exist, False if stopping failed.
        """
        try:
            state_manager = await Application.create_state_manager()

            # Check if app exists in distributed state
            app_info = None
            async for registry in state_manager.read_transaction(max_retries=3):
                app_info = registry.get_app(app_name)
                break

            if not app_info:
                logger.info(f"Application '{app_name}' does not exist, nothing to stop")
                return True

            # App exists, shutdown deployments and unregister
            logger.info(f"Stopping application '{app_name}'")

            async for registry in state_manager.write_transaction(max_retries=3):
                app_info = registry.get_app(app_name)
                if not app_info:
                    logger.warning(f"Application '{app_name}' was already deleted")
                    return True

                # Shutdown all deployments using the app's namespace
                proxy_namespace = ApplicationRegistry.get_ray_actor_namespace(app_name)
                for deploy_info in app_info.deployments.values():
                    try:
                        proxy_actor = ray.get_actor(deploy_info.proxy_actor_name, namespace=proxy_namespace)
                        await proxy_actor.shutdown.remote()
                        ray.kill(proxy_actor)
                        logger.info(f"Killed proxy actor '{deploy_info.proxy_actor_name}'")
                    except Exception as e:
                        logger.warning(f"Error killing proxy actor '{deploy_info.proxy_actor_name}': {e}")

                # Unregister the app
                registry.unregister_app(app_name)
                logger.info(f"Unregistered application '{app_name}' from distributed state")

            logger.info(f"Application '{app_name}' stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to stop application '{app_name}': {e}", exc_info=True)
            return False

    async def _stop_deployment(self, deployment_info: DeploymentInfo) -> None:
        """Stop a single deployment.

        Args:
            deployment_info: Information about the deployment to stop.
        """
        if deployment_info.proxy_actor_handle is None:
            return

        logger.info(f"Stopping deployment '{deployment_info.name}'")

        try:
            # Call shutdown on proxy actor
            await deployment_info.proxy_actor_handle.shutdown.remote()

            # Kill proxy actor
            ray.kill(deployment_info.proxy_actor_handle)

        except Exception as e:
            logger.error(f"Error stopping deployment '{deployment_info.name}': {e}")

        deployment_info.proxy_actor_handle = None
        logger.info(f"Deployment '{deployment_info.name}' stopped")

    async def get_deployment_stats(self, deployment_name: str) -> dict[str, Any]:
        """Get statistics for a specific deployment.

        Args:
            deployment_name: Name of the deployment.

        Returns:
            Dictionary with deployment statistics.

        Raises:
            ValueError: If deployment not found.
        """
        for deployment_info in self.deployments:
            if deployment_info.name == deployment_name:
                if deployment_info.proxy_actor_handle is None:
                    raise ValueError(f"Deployment '{deployment_name}' is not running")

                return await deployment_info.proxy_actor_handle.get_stats.remote()

        raise ValueError(f"Deployment '{deployment_name}' not found in application")

    async def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all deployments.

        Returns:
            Dictionary mapping deployment names to their stats.
        """
        stats = {}
        for deployment_info in self.deployments:
            if deployment_info.proxy_actor_handle is not None:
                try:
                    stats[deployment_info.name] = (
                        await deployment_info.proxy_actor_handle.get_stats.remote()
                    )
                except Exception as e:
                    logger.error(
                        f"Error getting stats for deployment '{deployment_info.name}': {e}"
                    )
                    stats[deployment_info.name] = {"error": str(e)}

        return stats

    def run(self) -> None:
        """Run the application (blocking).

        This is a convenience method that starts the application and blocks
        until interrupted.
        """
        try:
            # Start the application
            asyncio.run(self.start())

            # Keep running until interrupted
            logger.info(f"Application '{self.name}' is running. Press Ctrl+C to stop.")
            asyncio.get_event_loop().run_forever()

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")

        finally:
            # Cleanup
            asyncio.run(self.stop())

    @staticmethod
    async def cleanup(app_name: str) -> None:
        """Cleanup Ray actors used by the application and stop the application if it exists.

        Args:
            app_name: Name of the application.
        """
        # Kill all Ray actors matching this application name pattern
        # This is more robust than stop_by_name_if_exists because it doesn't rely on
        # application state (which may not exist if previous deployment failed mid-way)
        await Application._cleanup_ray_actors(app_name)

        # Try to stop application through normal path (for cleanup of registered state)
        stopped = await Application.stop_by_name_if_exists(app_name)
        if stopped:
            logger.info(f"Stopped existing application '{app_name}'")

    @staticmethod
    def _get_all_remaining_proxy_actor_names(app_name: str) -> list[str]:
        """Get names of all remaining proxy actors for this application.

        Returns:
            List of actor names
        """
        import ray

        actor_names = []
        try:
            # Use the app's namespace
            # Filter for proxy actors in this app's namespace
            app_namespace = ApplicationRegistry.get_ray_actor_namespace(app_name)
            proxy_prefix  = ApplicationRegistry.get_deployment_proxy_actor_name_prefix(app_name)
            proxy_suffix  = ApplicationRegistry.get_deployment_proxy_actor_name_suffix()

            # List actors in this app's namespace
            all_actors = ray.util.list_named_actors(all_namespaces=True)

            for actor_info in all_actors:
                if isinstance(actor_info, dict):
                    actor_name = actor_info.get("name", "")
                    actor_namespace = actor_info.get("namespace", "")
                    if (actor_name.startswith(proxy_prefix) and
                        actor_name.endswith(proxy_suffix) and
                        actor_namespace == app_namespace):
                        actor_names.append(actor_name)

        except Exception as e:
            logger.warning(f"Error listing proxy actors for application '{app_name}': {e}")

        return actor_names

    @staticmethod
    async def _cleanup_ray_actors(app_name: str) -> None:
        """Kill all proxy actors belonging to this application.

        Calls shutdown() on each proxy to kill replicas, then kills the proxy.
        Uses the app's namespace for consistent actor lookup.
        """
        import ray

        try:
            proxies_to_kill = Application._get_all_remaining_proxy_actor_names(app_name)
            if not proxies_to_kill:
                logger.info(f"No proxy actors found for application '{app_name}'")
                return

            logger.info(f"Found {len(proxies_to_kill)} proxy actors for application '{app_name}'")

            app_namespace = ApplicationRegistry.get_ray_actor_namespace(app_name)

            killed_count = 0
            for actor_name in proxies_to_kill:
                try:
                    actor_handle = ray.get_actor(actor_name, namespace=app_namespace)
                    logger.info(f"Killing proxy actor: {actor_name}")

                    # Call shutdown to kill replicas
                    try:
                        await actor_handle.shutdown.remote()
                        logger.info(f"Shutdown called on proxy actor: {actor_name}")
                    except Exception as e:
                        logger.warning(f"Shutdown failed for {actor_name}: {e}")

                    # Kill the proxy
                    ray.kill(actor_handle, no_restart=True)
                    killed_count += 1
                    logger.info(f"Killed proxy actor: {actor_name}")
                except Exception as e:
                    logger.warning(f"Failed to kill {actor_name}: {e}")

            logger.info(f"Killed {killed_count}/{len(proxies_to_kill)} proxy actors")

            # Wait for Ray to fully clean up actor registrations
            # ray.kill() is async, so we need to verify actors are actually gone
            if killed_count > 0:
                import asyncio
                logger.info("Waiting for Ray to fully clean up actor registrations...")
                max_wait_time = 10.0  # seconds
                check_interval = 0.5  # seconds
                elapsed = 0.0

                while elapsed < max_wait_time:
                    await asyncio.sleep(check_interval)
                    elapsed += check_interval

                    # Check if any actors still exist
                    remaining_actors = Application._get_all_remaining_proxy_actor_names(app_name)

                    if not remaining_actors:
                        logger.info(f"All actors cleaned up after {elapsed:.1f}s")
                        break

                    logger.debug(f"Still waiting for {len(remaining_actors)} actors to be cleaned up...")

                if remaining_actors:
                    logger.warning(
                        f"Timed out waiting for actors to be cleaned up after {max_wait_time}s. "
                        f"Remaining: {remaining_actors}"
                    )

        except Exception as e:
            logger.warning("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            logger.warning(f"Error during cleanup: {e}")
            logger.warning("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            # Don't fail deployment if cleanup fails

