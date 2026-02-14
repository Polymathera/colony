from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any, TypeVar, TYPE_CHECKING

import yaml

from ..redis_utils.client import RedisConfig

if TYPE_CHECKING:
    from ..stores.state_etcd import EtcdStorage

from .configs import (
    AWS_CONFIG_PATH,
    CLOUD_SYSTEM_CONFIG_PATH,
    KAFKA_CONFIG_PATH,
    MONITORING_CONFIG_PATH,
    SECURITY_CONFIG_PATH,
    AWSConfig,
    CloudSystemConfig,
    ConfigComponent,
    EnvironmentType,
    KafkaConfig,
    MonitoringConfig,
    PolymatheraConfig,
    SecurityConfig,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=ConfigComponent)


class ConfigurationManager:
    """
    Centralized configuration management for the Polymathera distributed system.

    Features:
    - Hierarchical configuration with multiple sources
    - Environment-specific configs
    - Secret management
    - Dynamic configuration updates
    - Configuration validation
    - Distributed configuration synchronization

    Configuration Sources (in order of precedence):
    1. Environment variables (highest)
    2. Command line arguments
    3. Local overrides
    4. Distributed store (etcd)
    5. Environment-specific config files
    6. Base config file
    7. Default values (lowest)

    - Initial Node Bootstrap: The first node to start up will:
        - Load configuration from local sources
        - Check if configuration exists in etcd
        - If no configuration exists in etcd, initialize it with the current config
        - Start watching for configuration changes
    - Other Nodes: When additional nodes start up, they will:
        - Load basic configuration to connect to etcd
        - Get the full configuration from etcd
        - Apply local overrides and environment variables
        - Start watching for configuration changes
    - Configuration Updates:
        - Changes to the distributed configuration in etcd are automatically propagated to all nodes
        - Each node maintains its local overrides and environment variables
        - Changes are applied atomically using etcd's transaction support
    - Security Considerations:
        - Sensitive values (API keys, passwords) should be provided through environment variables
        - etcd should be configured with proper authentication and TLS
        - Each environment (dev/staging/prod) has its own configuration namespace
    - Monitoring and Debugging:
        - Configuration changes are logged
        - Future enhancement: implement pub/sub for configuration updates
        - Metrics for configuration sync status
        - Health checks for etcd connectivity
    - This design ensures that:
        - Configuration is consistently distributed across all nodes
        - Local overrides are still possible when needed
        - Sensitive information is properly handled
        - Configuration changes are atomic and reliable
        - The system is resilient to etcd connectivity issues
        - Configuration updates are real-time across the cluster

    /etc/polymathera/                # System-wide configs
    ├── config.yml                   # Base config
    ├── config.production.yml        # Environment-specific
    ├── config.d/                    # Config fragments
    │   ├── logging.yml
    │   ├── security.yml
    │   └── redis.yml
    └── secrets/                     # Sensitive configs
        └── credentials.yml

    $HOME/.polymathera/              # User-specific configs
    ├── config.yml
    └── config.d/

    ./config/                        # Project/repo configs
    ├── config.yml                   # Default development config
    ├── config.production.yml
    ├── config.staging.yml
    └── examples/                    # Example configs
        └── config.example.yml

    **Configuring Subobjects - Best Practices**:
    There are two main patterns, each with their use cases:

    A. **Parent Passing Configuration (Dependency Injection)**:
    ```python
    # Preferred for most cases
    class Parent:
        def __init__(self, config: PolymatheraConfig):
            self.child = Child(config.child_config)
            self.other_child = OtherChild(config.other_child_config)

    class Child:
        def __init__(self, config: ChildConfig):
            self.config = config
    ```

    Advantages:
    - Clear dependencies
    - Better testability
    - Explicit configuration flow
    - Better control over configuration timing
    - Easier to mock in tests

    B. **Self-Configuration Pattern**:
    ```python
    # Use when components need dynamic config updates
    class DynamicService:
        def __init__(self):
            from polymathera.distributed import get_polymathera
            self.config_manager = await get_polymathera().get_config_manager()
            self.config = self.get_config_manager().get_child_config()
            self.watch_config_changes()

        async def watch_config_changes(self):
            async for config in self.get_config_manager().watch("service.*"):
                await self.reconfigure(config)
    ```

    Advantages:
    - Components can update their config dynamically
    - Works well with distributed systems
    - Simpler implementation for dynamic components

    C. **Hybrid Approach for Complex Systems**:
    ```python
    class HybridService:
        def __init__(self, static_config: StaticConfig):
            # Static configuration via dependency injection
            self.static_config = static_config

            # Dynamic configuration via self-configuration
            self.config_manager = get_polymathera().get_config_manager()
            self.watch_dynamic_config()
    ```

    ```python
    async def watch_redis_config():
        async for new_host in config_manager.watch("redis.host"):
            print(f"Redis host changed to: {new_host}")
            await reconnect_redis(new_host)

    # Watch entire kafka config
    async def watch_kafka_config():
        async for kafka_config in config_manager.watch("kafka.*"):
            print(f"Kafka config updated: {kafka_config}")
            await reconfigure_kafka(kafka_config)
    ```

    Key features of this design:
    - Registration System:
        - Components register their config classes with paths
        - Supports nested paths (e.g., "caching.redis")
        - Automatic schema generation
    - Type Safety:
        - All configs are Pydantic models
        - Type hints for config retrieval
        - Validation on load
    - Flexibility:
        - Components can self-configure or accept config from parent
        - Default values in config classes
        - Support for nested configurations
    - Schema Generation:
        - JSON Schema generation for documentation
        - YAML schema for validation
        - Automatic documentation of config structure
    - Dynamic Updates:
        - Watch for config changes
        - Type-safe config updates
        - Distributed configuration via etcd
    """

    def __init__(
        self,
        config_path: str | None = None,
        environment: EnvironmentType | None = None,
        overrides: dict[str, Any] | None = None,
        distributed: bool = False,
    ):
        """Initialize configuration manager
        NOTE: Config field names must be in all lowercase (and may contain underscores) to
        match the environment variable names and, hence, be configurable via environment variables.
        """
        self.config_path = config_path
        self.environment = environment or EnvironmentType.DEVELOPMENT
        self.overrides = overrides or {}
        self.distributed = distributed
        self._initialized = False  # Prevent re-initialization

        # TODO: Make this configurable
        # Config locations in order of precedence (lowest to highest)
        self.config_locations = [
            "/etc/polymathera",  # System-wide
            os.path.expanduser("~/.polymathera"),  # User-specific
            os.path.join(os.getcwd(), "config"),  # Project-specific
            os.path.join(os.getenv("APP_MOUNT_PATH", "/app"), "config"),
        ]

        if config_path:
            self.config_locations.append(config_path)

        self.config_locations = list(set(self.config_locations))

        self._config: PolymatheraConfig | None = None
        self._etcd_storage: EtcdStorage | None = None
        self._watch_task = None
        self._config_update_handlers = []

    @property
    def config(self) -> PolymatheraConfig | None:
        """Get the current configuration"""
        if self._config is None:
            raise RuntimeError(
                f"Configuration not loaded for config manager: ObjId={id(self)}"
            )
        return self._config

    async def initialize(self) -> None:
        """Initialize the configuration manager and start watching for changes"""
        if self._initialized:
            return

        if self.distributed:
            await self._init_etcd()

        await self._load_config()

        self._initialized = True

    async def _init_etcd(self) -> None:
        """Initialize etcd storage"""
        try:
            from ..stores.state_etcd import EtcdStorage
        except ImportError as e:
            raise ImportError(
                "etcd3 is required for distributed mode. "
                "Install it with: pip install etcd3  "
                "(or use poetry install --extras distributed)"
            ) from e

        # Initialize etcd storage with basic config to bootstrap
        self._etcd_storage = EtcdStorage(
            host=os.environ["POLYMATHERA_ETCD_HOST"],
            port=int(os.environ["POLYMATHERA_ETCD_PORT"]),
        )
        # Start watching for config changes
        self._watch_task = asyncio.create_task(self._watch_config_changes())
        # Push current config to etcd if we're the first node
        await self._maybe_initialize_distributed_config()

    def get_component(self, path: str) -> ConfigComponent | None:
        """Get a configuration component by its path"""
        return self.config.get_component(path)

    def get_component_by_type(self, config_cls: type[T]) -> T | None:
        """Get a configuration component by its class"""
        return self.config.get_component_by_type(config_cls)

    async def check_or_get_component(
        self, path: str, cls: type[T], config: T | None = None
    ) -> T:
        """Check if a component exists or get it"""
        logger.debug(f"Checking or getting component: {path}")

        if config is None:
            logger.debug(f"Getting pre-existing config for path: {path}")
            config = self.get_component(path)

        if config is None:
            raise ValueError(f"No component found for path: {path}")

        if not isinstance(config, cls):
            raise ValueError(f"Config must be a {cls.__name__}, got {type(config)}")

        return config

    def get_schema(self, format: str = "json") -> str:
        """Get configuration schema in the specified format"""
        schema = PolymatheraConfig.model_json_schema()

        if format.lower() == "yaml":
            return yaml.dump(schema)
        return json.dumps(schema, indent=2)

    def save_config(self, path: str) -> None:
        """Save the current configuration to a file"""
        if not self._config:
            return

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save with proper YAML formatting
        with open(path, "w") as f:
            yaml.dump(
                self.config.model_dump(),
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    async def _maybe_initialize_distributed_config(self) -> None:
        """Initialize distributed config if it doesn't exist"""
        if not self._etcd_storage:
            return

        # Try to get the distributed config
        value = await self._get_distributed_config()

        if value is None:
            # No distributed config exists yet, we're the first node
            # Push our config to etcd
            logger.debug("Initializing distributed configuration in etcd")
            await self._push_config_to_etcd()

    def _get_config_key(self) -> str:
        """Get the etcd key for the current environment's config"""
        return f"/polymathera/config/{self.environment}"

    async def _get_distributed_config(self) -> dict[str, Any] | None:
        """Get configuration from etcd"""
        if not self._etcd_storage:
            return None

        try:
            config_key = self._get_config_key()
            value, _ = await self._etcd_storage.get_with_version(config_key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Failed to get distributed config: {e}")
            return None

    async def _push_config_to_etcd(self) -> None:
        """Push current configuration to etcd"""
        if not self._etcd_storage or not self._config:
            return

        try:
            config_key = self._get_config_key()
            config_json = self.config.model_dump_json()
            # Get current version first
            _, version = await self._etcd_storage.get_with_version(config_key)
            # Use compare-and-swap to safely update
            success = await self._etcd_storage.compare_and_swap(
                config_key, config_json, version
            )
            if success:
                logger.debug("Successfully pushed configuration to etcd")
            else:
                logger.warning("Failed to push config - version mismatch")
        except Exception as e:
            logger.error(f"Failed to push config to etcd: {e}")

    def _load_config_file(self, path: str) -> dict[str, Any]:
        """Load a YAML/JSON config file"""
        if not os.path.exists(path):
            return {}

        try:
            with open(path) as f:
                if path.endswith((".yaml", ".yml")):
                    return yaml.safe_load(f) or {}
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config file {path}: {e}")
            return {}

    def _load_config_directory(self, directory: str) -> dict[str, Any]:
        """Load all config files from a directory"""
        config = {}
        if not os.path.exists(directory) or not os.path.isdir(directory):
            return config

        try:
            for filename in sorted(os.listdir(directory)):
                if filename.endswith((".yaml", ".yml", ".json")):
                    path = os.path.join(directory, filename)
                    file_config, _ = self._flatten_config(self._load_config_file(path))
                    config = self._deep_merge(config, file_config)
        except Exception as e:
            logger.warning(f"Failed to load config directory {directory}: {e}")
        return config

    def _deep_merge(self, dict1: dict, dict2: dict) -> dict:
        """Deep merge two dictionaries"""
        if not isinstance(dict1, dict):
            dict1 = {}
        if not isinstance(dict2, dict):
            dict2 = {}

        result = dict1.copy()
        for key, value in dict2.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _flatten_config(
        self, config: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Convert flat config_dict to nested structure if needed
        components_dict = {}
        version_data = config.pop("version", {})

        for path, value in config.items():
            if "." in path:
                # Handle nested paths
                current = components_dict
                parts = path.split(".")
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = value
            else:
                # Top-level components
                components_dict[path] = value

        return components_dict, version_data

    async def _load_config(self) -> None:
        """Load configuration from all sources in order of precedence
        NOTE: Config field names must be in all lowercase (and may contain underscores) to
        match the environment variable names and, hence, be configurable via environment variables.
        """
        logger.info(
            "\n\n##############\n"
            + f"Loading configuration for config manager: ObjId={id(self)}"
            + "\n##############\n\n"
        )
        # Start with default config (not an empty config)
        config_dict = PolymatheraConfig().model_dump()

        components_dict, version_data = self._flatten_config(config_dict)

        # Load from config locations in order
        for location in self.config_locations:
            # Expand user path if needed
            location = os.path.expanduser(location)

            # Load main config file (try both yaml and json)
            for filename in ["config.yml", "config.yaml", "config.json"]:
                config_path = os.path.join(location, filename)
                if os.path.exists(config_path):
                    logger.info(f"Loading config file: {config_path}")
                    file_components_dict, _ = self._flatten_config(
                        self._load_config_file(config_path)
                    )
                    components_dict = self._deep_merge(
                        components_dict, file_components_dict
                    )
                    break

            # Load environment-specific config
            if self.environment:
                for filename in [
                    f"config.{self.environment}.yml",
                    f"config.{self.environment}.yaml",
                    f"config.{self.environment}.json",
                ]:
                    config_path = os.path.join(location, filename)
                    if os.path.exists(config_path):
                        logger.info(
                            f"Loading environment-specific config file: {config_path}"
                        )
                        file_components_dict, _ = self._flatten_config(
                            self._load_config_file(config_path)
                        )
                        components_dict = self._deep_merge(
                            components_dict, file_components_dict
                        )
                        break

            # Load config.d directory
            conf_d = os.path.join(location, "config.d")
            if os.path.exists(conf_d):
                logger.info(f"Loading config.d directory: {conf_d}")
                file_components_dict, _ = self._flatten_config(
                    self._load_config_directory(conf_d)
                )
                components_dict = self._deep_merge(
                    components_dict, file_components_dict
                )

        # If distributed mode is enabled, try to get config from etcd
        if self.distributed and self._etcd_storage:
            distributed_config_dict = await self._get_distributed_config()
            if distributed_config_dict:
                distributed_components_dict, _ = self._flatten_config(
                    distributed_config_dict
                )
                components_dict = self._deep_merge(
                    components_dict, distributed_components_dict
                )
                # version_data = self._deep_merge(version_data, distributed_version_data)

        # Apply environment variables (highest priority except overrides)
        self._apply_env_vars(components_dict)

        # Apply manual overrides (highest priority)
        if self.overrides:
            components_dict = self._deep_merge(components_dict, self.overrides)

        # Create final config dictionary with proper structure
        final_config = {"version": version_data, **components_dict}

        # Create config object
        self._config = PolymatheraConfig(**final_config)
        logger.info(
            "\n\n##############\n"
            + f"Configuration loaded for config manager: ObjId={id(self)}"
            + "\n##############\n\n"
        )
        # print(self.config.model_dump_json(indent=2))
        # for path, component in self.config._components.items():
        #     print(f"{path}: " + component.model_dump_json(indent=2))

    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

        if self._etcd_storage:
            config_key = self._get_config_key()
            await self._etcd_storage.cleanup(config_key)

    def _apply_env_vars(self, components_dict: dict[str, Any]) -> None:
        """Apply environment variables to configuration
        NOTE: Config field names must be in all lowercase (and may contain underscores) to
        match the environment variable names and, hence, be configurable via environment variables.
        Environment variables must have the format `f"POLYMATHERA_{config_path}_{field_name}"`
        """
        # Find longest matching registered config path
        matching_paths = ConfigComponent.match_config_path_pattern("*")
        matching_paths.sort(key=len, reverse=True)
        matching_paths = [path.replace(".", "_") for path in matching_paths]

        env_prefix = "POLYMATHERA_"
        for key, value in os.environ.items():
            if not key.startswith(env_prefix):
                continue

            # Remove prefix and convert env var to lowercase path format
            env_path = key[len(env_prefix) :].lower()

            for config_path in matching_paths:
                # Check if config path is a prefix of the env var path
                if env_path.startswith(config_path.replace(".", "_")):
                    # Get the remaining path after the config path
                    remaining = env_path[len(config_path.replace(".", "_")) + 1 :]
                    if remaining:
                        # Navigate to the config object
                        current = components_dict
                        for part in config_path.split("."):
                            current = current.setdefault(part, {})
                        # Set the field value, converting to appropriate type
                        try:
                            # Try to interpret as JSON first
                            current[remaining] = json.loads(value)
                        except json.JSONDecodeError:
                            # Fall back to string if not valid JSON
                            current[remaining] = value
                    break

    def get_aws_config(self) -> AWSConfig:
        """Get AWS-specific configuration"""
        return self.get_component(AWS_CONFIG_PATH)

    def get_redis_config(self) -> RedisConfig:
        """Get Redis configuration"""
        return self.get_component(RedisConfig.CONFIG_PATH)

    def get_kafka_config(self) -> KafkaConfig:
        """Get Kafka configuration"""
        return self.get_component(KAFKA_CONFIG_PATH)

    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return self.get_component(MONITORING_CONFIG_PATH)

    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        return self.get_component(SECURITY_CONFIG_PATH)

    def get_cloud_system_config(self) -> CloudSystemConfig:
        """Get cloud system configuration"""
        return self.get_component(CLOUD_SYSTEM_CONFIG_PATH)

    def update_config(self, updates: dict[str, Any]) -> None:
        """Update configuration with new values"""
        if not self._config:
            return

        # Get current config as dict
        config_dict = self.config.model_dump()

        # Apply updates
        config_dict = self._deep_merge(config_dict, updates)

        # Create new config object
        self._config = PolymatheraConfig(**config_dict)

        # Push updates to etcd if in distributed mode
        if self.distributed and self._etcd_storage:
            asyncio.create_task(self._push_config_to_etcd())

        # Notify subscribers
        self._notify_config_updated()

    def get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """Get the value of a feature flag"""
        return self.config.feature_flags.get(flag_name, default)

    def get_custom_setting(self, setting_name: str, default: Any = None) -> Any:
        """Get a custom setting value"""
        return self.config.custom_settings.get(setting_name, default)

    async def _watch_config_changes_old(self) -> None:
        """Watch for configuration changes in etcd (Older, potentially blocking version)"""
        if not self._etcd_storage:
            return

        config_key = self._get_config_key()

        try:
            async for event in await self._etcd_storage.watch(config_key):
                if event.type == "PUT":
                    # Configuration was updated
                    new_config = json.loads(event.value.decode())
                    self._update_config_from_distributed(new_config)
                    logger.info("Configuration updated from etcd")
        except Exception as e:
            logger.error(f"Error watching config changes: {e}")
            # Retry watching after a delay
            await asyncio.sleep(5)
            # Avoid restarting the task here if the main loop handles it
            # self._watch_task = asyncio.create_task(self._watch_config_changes())

    async def _watch_config_changes(self) -> None:
        """Watch for configuration changes in etcd using the simplified async for loop."""
        if not self.distributed or not self._etcd_storage:
            logger.warning("Distributed mode or etcd storage not enabled, skipping watch.")
            return

        config_key = self._get_config_key()
        logger.info(f"Starting etcd configuration watch on key '{config_key}'...")

        # Loop to restart watch if it fails/exits
        while True: # Loop indefinitely to keep watching
            try:
                async for event in await self._etcd_storage.watch(config_key):
                    if event and event.type == "PUT":
                        try:
                            # Decode the new configuration data
                            config_data = json.loads(event.value.decode())
                            logger.info(f"Received config update from etcd for key '{config_key}'")
                            # Update the internal configuration state
                            self._update_config_from_distributed(config_data)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode config update from etcd for key '{config_key}'")
                        except Exception as e:
                            logger.exception(f"Error processing etcd config update for key '{config_key}': {e}")
            except asyncio.CancelledError:
                logger.info(f"Etcd configuration watch task cancelled for key '{config_key}'.")
                await self._etcd_storage.stop_watch(config_key)
                break # Exit the while loop
            except Exception as e:
                logger.exception(f"Error in etcd watch loop for key '{config_key}': {e}. Retrying in 5s...")
                # Wait before retrying to prevent tight loops

            # If the loop finishes or has an error, wait before restarting
            logger.warning(f"Etcd watch stream ended or failed for key '{config_key}'. Restarting watch in 5s...")
            await asyncio.sleep(5)

    def _update_config_from_distributed(self, new_config: dict[str, Any]) -> None:
        """Update local configuration from distributed source"""
        try:
            # Merge with current config keeping local overrides
            config_dict = new_config.copy()

            # Apply environment variables (highest priority)
            self._apply_env_vars(config_dict)

            # Apply local overrides
            config_dict.update(self.overrides)

            # Update config object
            self._config = PolymatheraConfig(**config_dict)

            # Notify subscribers if we implement that feature
            self._notify_config_updated()
        except Exception as e:
            logger.error(f"Failed to update config from distributed source: {e}")

    def _notify_config_updated(self) -> None:
        """Notify subscribers about configuration updates"""
        if not self._config:
            return

        old_config = {} if not hasattr(self, "_last_config") else self._last_config
        new_config = self.config.model_dump()

        # Store current config for next comparison
        self._last_config = new_config

        # Notify all handlers
        for handler in self._config_update_handlers:
            try:
                handler(old_config, new_config)
            except Exception as e:
                logger.error(f"Error in config update handler: {e}")

    async def watch(self, path: str) -> AsyncIterator[Any]:
        """
        Watch for changes to a specific configuration path.

        Args:
            path: Dot-separated path to watch (e.g. `"redis.host"` or `"kafka.*"`)

        Returns:
            `AsyncIterator` yielding new values when the watched path changes

        Example:
            ```python
            async for new_value in config_manager.watch("redis.host"):
                print(f"Redis host changed to: {new_value}")
            ```
        """
        if not self.distributed or not self._etcd_storage:
            raise RuntimeError(
                "Configuration watching requires distributed mode with etcd"
            )

        # Convert dot notation to parts
        path_parts = path.split(".")

        # Queue for receiving config updates
        queue = asyncio.Queue()

        def _handle_config_update(
            old_config: dict[str, Any], new_config: dict[str, Any]
        ) -> None:
            """Handle configuration updates and push relevant changes to queue"""
            try:
                # Get old and new values at path
                old_value = old_config
                new_value = new_config

                for part in path_parts:
                    if part == "*":
                        # Wildcard - yield the entire remaining config
                        if old_value != new_value:
                            asyncio.create_task(queue.put(new_value))
                        return
                    elif isinstance(old_value, dict) and isinstance(new_value, dict):
                        old_value = old_value.get(part, {})
                        new_value = new_value.get(part, {})
                    else:
                        # Path doesn't exist
                        return

                # If we get here and values differ, push to queue
                if old_value != new_value:
                    asyncio.create_task(queue.put(new_value))

            except Exception as e:
                logger.error(f"Error handling config update for path {path}: {e}")

        # Register update handler
        self._config_update_handlers.append(_handle_config_update)

        try:
            while True:
                value = await queue.get()
                yield value
        finally:
            # Clean up handler when iterator is closed
            self._config_update_handlers.remove(_handle_config_update)
