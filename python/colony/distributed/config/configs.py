from __future__ import annotations

import enum
import fnmatch
import hashlib
import json
import logging
import os
import re
import types
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, TypeVar
import traceback

from packaging import version
from pydantic import BaseModel, Field, PrivateAttr, model_validator, ConfigDict

print("\nModule loading - about to define ConfigComponent")


class ConfigRegistry:
    """Registry that's aware of module reloads and handles duplicate registrations"""

    def __init__(self):
        self._registry: dict[str, tuple[type[ConfigComponent], int]] = {}
        self._default_paths: dict[tuple[type[ConfigComponent], int], str] = {}
        self._logger = logging.getLogger(__name__)

    def register(self, path: str, config_cls: type[ConfigComponent]) -> None:
        cls_id = id(config_cls)
        cls_qualname = f"{config_cls.__module__}.{config_cls.__qualname__}"
        existing = self._registry.get(path)

        self._logger.debug(
            f"Attempting to register config class: {cls_qualname} (id: {cls_id}) at path: {path}"
        )

        if existing is not None:
            existing_cls, existing_id = existing
            existing_qualname = f"{existing_cls.__module__}.{existing_cls.__qualname__}"

            if (
                existing_id != cls_id
                and existing_cls.__qualname__ == config_cls.__qualname__
            ):
                # Same class name but different ID - this is a reload
                self._logger.debug(
                    f"Detected module reload: Updating registration for {cls_qualname}\n"
                    f"  Old ID: {existing_id}, New ID: {cls_id}\n"
                    f"  Path: {path}"
                )
                # Update registration with new class
                self._registry[path] = (config_cls, cls_id)
                self._default_paths[(config_cls, cls_id)] = path
                return
            elif existing_cls.__qualname__ != config_cls.__qualname__:
                # Actually different classes trying to use same path
                self._logger.error(
                    f"Path conflict detected:\n"
                    f"  Path: {path}\n"
                    f"  Existing: {existing_qualname} (id: {existing_id})\n"
                    f"  New: {cls_qualname} (id: {cls_id})"
                )
                raise ValueError(
                    f"Path {path} is already registered to {existing_qualname}"
                )

        self._logger.debug(
            f"Registering new config class: {cls_qualname} at path: {path}"
        )
        self._registry[path] = (config_cls, cls_id)
        self._default_paths[(config_cls, cls_id)] = path

    def get_class(self, path: str) -> type[ConfigComponent] | None:
        if path not in self._registry:
            self._logger.debug(f"No config class registered at path: {path}")
            return None
        cls, _ = self._registry[path]
        cls_qualname = f"{cls.__module__}.{cls.__qualname__}"
        self._logger.debug(f"Found config class {cls_qualname} at path: {path}")
        return cls

    def get_path(self, config_cls: type[ConfigComponent]) -> str | None:
        cls_id = id(config_cls)
        path = self._default_paths.get((config_cls, cls_id))
        cls_qualname = f"{config_cls.__module__}.{config_cls.__qualname__}"
        if path is None:
            self._logger.debug(
                f"No path registered for config class: {cls_qualname} (id: {cls_id})"
            )
        else:
            self._logger.debug(
                f"Found registered path {path} for config class: {cls_qualname}"
            )
        return path

    def clear(self) -> None:
        self._logger.debug("Clearing config registry")
        self._registry.clear()
        self._default_paths.clear()


# Global registry instance
_config_registry = ConfigRegistry()

logger = logging.getLogger(__name__)


class ConfigComponent(BaseModel):
    """Base class for all configuration components that can be registered"""

    # Instance variables for path tracking
    _config_path: str | None = PrivateAttr(default=None)
    _parent_path: str | None = PrivateAttr(default=None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
    def running_locally(cls) -> bool:
        """Check if the code is running locally"""
        return os.environ.get("POLYMATHERA_RUNNING_LOCALLY", "false").lower() == "true"

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ == "ConfigComponent":
            logger.debug(f"\nConfigComponent itself being subclassed with id {id(cls)}")
        else:
            logger.debug(
                f"\nConfigComponent subclass loaded:"
                f"\n  name: {cls.__name__}"
                f"\n  qualname: {cls.__qualname__}"
                f"\n  module: {cls.__module__}"
                f"\n  id: {id(cls)}"
            )

    @classmethod
    def register_config(cls, path: str | None = None) -> Callable[[type[T]], type[T]]:
        """
        Decorator to register a configuration class with its default path

        Args:
            path: Dot-separated path where default config can be found.
            If not provided, the class must have a CONFIG_PATH class variable.

        Example:
            ```python
            @ConfigComponent.register_config("redis")
            class RedisConfig(ConfigComponent):
                host: str = "localhost"
                port: int = 6379
            ```
        """

        def decorator(config_cls: type[T]) -> type[T]:
            config_path = path  # Capture path in closure
            if config_path is None:
                if not hasattr(config_cls, "CONFIG_PATH"):
                    raise ValueError(
                        f"Class {config_cls.__name__} must have a CONFIG_PATH class variable"
                    )
                config_path = config_cls.CONFIG_PATH

            # Register with the registry manager instead of class variables
            _config_registry.register(config_path, config_cls)

            # Set path attributes on the class
            parts = config_path.split(".")
            config_cls._config_path = config_path
            config_cls._parent_path = ".".join(parts[:-1]) if len(parts) > 1 else None

            return config_cls

        return decorator

    @classmethod
    def get_registered_configs(cls) -> dict[str, type[ConfigComponent]]:
        """Get all registered configuration classes with their paths"""
        return {path: entry[0] for path, entry in _config_registry._registry.items()}

    @classmethod
    def get_default_path(cls, config_cls: type[ConfigComponent]) -> str | None:
        """Get the default path for a configuration class"""
        return _config_registry.get_path(config_cls)

    @classmethod
    def match_config_path_pattern(cls, path_pattern: str) -> list[str]:
        """Check if a path matches any registered config path pattern"""
        return [
            path
            for path in _config_registry._registry.keys()
            if fnmatch.fnmatch(path, path_pattern)
        ]

    @classmethod
    def clear_registry(cls) -> None:
        """Clear the component registry - use with caution, mainly for testing"""
        _config_registry.clear()

    @model_validator(mode="after")
    def validate_component(self) -> ConfigComponent:
        """Validate the component after all fields are set"""
        # Get validation methods (methods starting with validate_)
        validation_methods = [
            method
            for name, method in self.__class__.__dict__.items()
            if name.startswith("validate_")
            and callable(method)
            and name != "validate_component"
        ]

        # Run all validation methods
        for method in validation_methods:
            try:
                method(self)
            except Exception as e:
                raise ValueError(f"Validation failed: {str(e)}")

        return self

    @classmethod
    async def check_or_get_component(cls, config: ConfigComponent | None = None) -> ConfigComponent:
        if not hasattr(cls, "CONFIG_PATH"):
            raise AttributeError(
                f"{cls.__name__} must define a class variable CONFIG_PATH to use check_or_get_component."
            )

        if cls.running_locally():
            return cls() if config is None else config

        from .. import get_polymathera
        config_manager = await get_polymathera().get_config_manager()

        return await config_manager.check_or_get_component(cls.CONFIG_PATH, cls, config)


# Define TypeVar after ConfigComponent class
T = TypeVar("T", bound=ConfigComponent)


def register_polymathera_config(
    path: str | None = None
) -> Callable[[type[T]], type[T]]:
    """Decorator to register a configuration class with its default path"""
    return ConfigComponent.register_config(path)


class ConfigSource(Enum):
    """Sources of configuration in order of precedence (highest to lowest)"""

    ENVIRONMENT = "environment"
    COMMAND_LINE = "command_line"
    CONFIG_FILE = "config_file"
    ETCD = "etcd"
    DEFAULT = "default"


class EnvironmentType(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


AWS_CONFIG_PATH = "aws"


@register_polymathera_config(path=AWS_CONFIG_PATH)
class AWSConfig(ConfigComponent):
    """AWS-specific configuration"""

    region: str = Field(default="us-east-1", json_schema_extra={"env": "AWS_REGION"})
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None

    # Service-specific configs
    dynamodb_endpoint: str | None = None
    s3_endpoint: str | None = None
    sqs_endpoint: str | None = None

    # Resource configs
    lambda_memory: int = 1024
    lambda_timeout: int = 300
    fargate_cpu: int = 256
    fargate_memory: int = 512


KAFKA_CONFIG_PATH = "kafka"


@register_polymathera_config(path=KAFKA_CONFIG_PATH)
class KafkaConfig(ConfigComponent):
    """Kafka configuration"""

    bootstrap_servers: list[str] = Field(default_factory=lambda: ["localhost:9092"])
    topic_prefix: str = "polymathera"
    consumer_group: str = "polymathera-consumer"
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: str | None = None
    sasl_username: str | None = None
    sasl_password: str | None = None


MONITORING_CONFIG_PATH = "monitoring"


@register_polymathera_config(path=MONITORING_CONFIG_PATH)
class MonitoringConfig(ConfigComponent):
    """Monitoring and observability configuration"""

    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    metrics_interval: int = 60
    tracing_sample_rate: float = 0.1
    prometheus_port: int = 9090
    jaeger_endpoint: str | None = None


RESOURCE_LIMITS_CONFIG_PATH = "resource_limits"


@register_polymathera_config(path=RESOURCE_LIMITS_CONFIG_PATH)
class ResourceLimits(ConfigComponent):
    """Resource limits for components"""

    max_memory_mb: int = 1024
    max_cpu_cores: float = 1.0
    max_disk_gb: int = 10
    max_network_mbps: int = 100


SECURITY_CONFIG_PATH = "security"


@register_polymathera_config(path=SECURITY_CONFIG_PATH)
class SecurityConfig(ConfigComponent):
    """Security configuration"""

    enable_encryption: bool = True
    enable_authentication: bool = True
    jwt_secret: str | None = None
    ssl_cert_path: str | None = None
    ssl_key_path: str | None = None
    allowed_origins: list[str] = Field(default_factory=list)
    api_keys: dict[str, str] = Field(default_factory=dict)


CLOUD_SYSTEM_CONFIG_PATH = "cloud_system"


@register_polymathera_config(path=CLOUD_SYSTEM_CONFIG_PATH)
class CloudSystemConfig(ConfigComponent):
    """System-wide configuration"""

    environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    deployment_type: str = "kubernetes"  # kubernetes, ecs, ec2, etc.
    region: str = "us-east-1"
    availability_zones: list[str] = Field(default_factory=list)
    vpc_id: str | None = None
    subnet_ids: list[str] = Field(default_factory=list)
    security_group_ids: list[str] = Field(default_factory=list)


FEATURE_FLAGS_CONFIG_PATH = "feature_flags"


@register_polymathera_config(path=FEATURE_FLAGS_CONFIG_PATH)
class FeatureFlagsConfig(ConfigComponent):
    """Configuration for feature flags"""

    feature_flags: dict[str, bool] = Field(default_factory=dict)


CUSTOM_SETTINGS_CONFIG_PATH = "custom_settings"


@register_polymathera_config(path=CUSTOM_SETTINGS_CONFIG_PATH)
class CustomSettingsConfig(ConfigComponent):
    """Configuration for custom settings"""

    custom_settings: dict[str, Any] = Field(default_factory=dict)


class ConfigVersion(BaseModel):
    """Version information for a configuration"""

    schema_version: str = Field(
        default="1.0.0", description="Version of the configuration schema"
    )
    content_version: str = Field(description="Hash of the configuration content")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When this version was created"
    )
    created_by: str | None = Field(
        default=None, description="Who/what created this version"
    )
    description: str | None = Field(
        default=None, description="Description of changes in this version"
    )
    parent_version: str | None = Field(
        default=None, description="Content version of the parent configuration"
    )

    def __lt__(self, other: ConfigVersion) -> bool:
        """Compare versions based on schema version and creation time"""
        if not isinstance(other, ConfigVersion):
            return NotImplemented

        # First compare schema versions
        self_schema = version.parse(self.schema_version)
        other_schema = version.parse(other.schema_version)

        if self_schema != other_schema:
            return self_schema < other_schema

        # If schema versions are equal, compare creation times
        return self.created_at < other.created_at

    def is_compatible_with(self, other: ConfigVersion) -> bool:
        """Check if this version is compatible with another version"""
        self_schema = version.parse(self.schema_version)
        other_schema = version.parse(other.schema_version)

        # Major version must match for compatibility
        return self_schema.major == other_schema.major


# DO NOT REGISTER THIS CLASS WITH A PATH
class PolymatheraConfig(ConfigComponent):
    """
    Root configuration class that can hold any registered ConfigComponent
    at its registered path

    It supports nested configuration components with automatic default loading:
    - Automatic Initialization: Nested config components are automatically initialized when their parent is loaded
    - Default Values: Uses registered default paths when available
    - Fallback to Defaults: Creates new instances with default values when no registered config exists
    - Type Safety: Properly handles Optional types and type checking
    - Registry Integration: Integrates with the existing component registry
    """

    _components: dict[str, ConfigComponent] = PrivateAttr(default_factory=dict)
    version: ConfigVersion = Field(
        default_factory=lambda: ConfigVersion(
            schema_version="1.0.0",
            content_version="0" * 40,
            created_at=datetime.utcnow(),
        )
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def __init__(self, **data):
        # Extract version data if present
        version_data = data.pop("version", {})

        super().__init__(**data)
        self._components = {}

        # Initialize all registered components with their default paths
        for path, config_cls in self.get_registered_configs().items():
            # Get config data from the path in the input data
            config_data = self._get_nested_dict_value(data, path, {})
            # Initialize nested components and store in components dict
            self._components[path] = self._initialize_config_component(config_data, config_cls)

        self.version = None
        self.update_version(version_data)

    def update_version(self, version_data: dict[str, Any] = {}) -> None:
        """Update the version information"""
        new_version_data = {}
        if self.version is not None:
            new_version_data = self.version.model_dump()
            new_version_data.update(version_data)
        if not new_version_data.get("content_version"):
            new_version_data["content_version"] = self.calculate_content_version()

        env_vars = self._read_env_vars(ConfigVersion)
        self._deep_update(new_version_data, env_vars)

        self.version = ConfigVersion(**new_version_data)

    def _get_nested_dict_value(self, d: dict, path: str, default: Any) -> Any:
        """Get a value from a nested dictionary using dot notation"""
        current = d
        for part in path.split("."):
            if not isinstance(current, dict):
                return default
            current = current.get(part, {})
        return current

    def _read_env_vars(self, cls: type[BaseModel]) -> dict[str, Any]:
        """Read environment variables for a component. This is not recursive because config
        components are created first before their subcomponents are initialized separately."""
        env_vars = {}

        # Check if field should be read from the environment
        for field_name, field in cls.model_fields.items():
            if field.json_schema_extra is None or "env" not in field.json_schema_extra:
                continue

            # Get the environment variable value
            env_var = field.json_schema_extra["env"]
            if env_var not in os.environ:
                if field.json_schema_extra.get("optional", False):
                    logger.warning(
                        f"Field {field_name} from optional environment variable {env_var} not found"
                    )
                    continue
                else:
                    raise ValueError(
                        f"Field {field_name} from environment variable {env_var} not found"
                    )

            # Get the field type, handling Optional types
            field_type = field.annotation
            field_parser = field.json_schema_extra.get("parser", None)
            if isinstance(field_type, types.UnionType) and hasattr(field_type, "__args__"):
                # Handle Optional[T] by getting the first non-None type
                field_type = next(
                    (t for t in field_type.__args__ if t is not type(None)), None
                )
            elif (field_type is list or field_type is dict) and field_parser is None:
                raise ValueError(f"Field {field_name} is a list or dict, which is not supported for environment-configurable fields.")

            value = field_parser(os.environ[env_var]) if field_parser else field_type(os.environ[env_var])  # Ensure the type is correct
            env_vars[field_name] = value
            logger.debug(
                f"Field {field_name} from environment variable {env_var} found: {value}"
            )

        return env_vars

    def _deep_update(
        self, base_dict: dict[str, Any], update_dict: dict[str, Any]
    ) -> None:
        """Recursively update a dictionary with another dictionary, merging nested structures"""
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _initialize_config_component(self, config_data: dict[str, Any], config_cls: type[ConfigComponent]) -> ConfigComponent:
        """
        Recursively initialize nested ConfigComponent fields that are None
        with their default configurations, and ensure all nested ConfigComponents
        (whether previously initialized or not) have their own nested components initialized.
        """
        # Read environment variables to be validated when the component is initialized
        env_vars = self._read_env_vars(config_cls)
        config_data.update(env_vars)
        component = config_cls(**config_data) # This validates the config data

        # print(f">>>>>> Initializing nested components for: {config_cls.__name__}")
        # Get all fields of the component
        for field_name, field in config_cls.model_fields.items():
            field_value = getattr(component, field_name, None)

            # Get the field type, handling Optional types
            field_type = field.annotation
            # print(f"\tField {field_name}:")
            # print(f"\t\tType: {field_type}")
            # print(f"\t\tType class: {field_type.__class__}")
            # print(f"\t\tType dir: {dir(field_type)}")
            # print(f"\t\tValue: {field_value}")

            if isinstance(field_type, types.UnionType) and hasattr(field_type, "__args__"):
                # Handle Optional[T] by getting the first non-None type
                field_type = next(
                    (t for t in field_type.__args__ if t is not type(None)), None
                )
                # logger.info(f"\tFound Optional field: {field_name}: {field_type} = {field_value}")
            elif field_type is list and hasattr(field_type, "__args__"):
                # Handle List types
                field_type = field_type.__args__[0]
                if field_value is None:
                    field_value = []
                    setattr(component, field_name, field_value)
                # Reinitialize list items (to include env vars) if they are ConfigComponents
                if isinstance(field_type, type) and issubclass(field_type, ConfigComponent):
                    assert hasattr(field_type, "model_fields"), f"Field type {field_type} has no model_fields"
                    for i, item in enumerate(field_value):
                        if isinstance(item, dict):
                            field_value[i] = self._initialize_config_component(item, field_type)
                        elif isinstance(item, ConfigComponent):
                            field_value[i] = self._initialize_config_component(item.model_dump(), field_type)
                        else:
                            raise ValueError(f"Unexpected item type: {type(item)}. Expected {field_type} or dict.")
                continue
            elif field_type is dict:
                # Handle Dict types
                if field_value is None:
                    field_value = {}
                    setattr(component, field_name, field_value)
                # TODO: Handle Dict elements of ConfigComponent type.
                continue

            if field_type is None:
                continue

            # Check if field type is a ConfigComponent
            if isinstance(field_type, type) and issubclass(field_type, ConfigComponent):
                assert hasattr(field_type, "model_fields"), f"Field type {field_type} has no model_fields"
                path = self.get_default_path(field_type)
                if field_value is None:
                    # Initialize uninitialized field
                    if path and path in self._components:
                        # Use existing component if available
                        field_value = self._components[path]
                    else:
                        # Create new instance with defaults and env vars applied
                        field_value = self._initialize_config_component({}, field_type)
                        if path:
                            self._components[path] = field_value

                    # Set the field value
                    setattr(component, field_name, field_value)
                    # logger.debug(f">>>>>> Initialized nested component: {path}.{field_name} = {field_value}")
                elif isinstance(field_value, dict):
                    # Convert dict to component instance
                    field_value = self._initialize_config_component(field_value, field_type)
                    setattr(component, field_name, field_value)
                elif isinstance(field_value, ConfigComponent):
                    # Reinitialize the component with its default values and env vars applied
                    field_value = self._initialize_config_component(field_value.model_dump(), field_type)
                    setattr(component, field_name, field_value)

        return component

    def get_component(self, path: str) -> ConfigComponent | None:
        """Get a configuration component by its path

        Searches for components using the following precedence:
        1. Exact path match
        2. Most specific wildcard path match (longest matching prefix)
        3. Most specific registered path that matches the search path

        Args:
            path: Configuration path, may contain wildcards

        Returns:
            Matching ConfigComponent if found, None otherwise
        """
        # First try exact match
        component = self._components.get(path)
        if component is not None:
            return component

        # Get all registered paths that could match
        matching_paths = []
        search_pattern = path.replace(".", r"\.").replace("*", ".*")
        search_regex = re.compile(f"^{search_pattern}$")

        for registered_path in self._components:
            # Check if registered path matches search path
            if search_regex.match(registered_path):
                matching_paths.append(registered_path)
            # Check if search path matches registered path with wildcards
            registered_pattern = registered_path.replace(".", r"\.").replace("*", ".*")
            registered_regex = re.compile(f"^{registered_pattern}$")
            if registered_regex.match(path):
                matching_paths.append(registered_path)

        # Sort by specificity (longest non-wildcard prefix first)
        matching_paths.sort(key=lambda p: (-len(p.split("*")[0]), -len(p)))

        # Return first match if any found
        if matching_paths:
            component = self._components[matching_paths[0]]
            return component

        return None

    def get_component_by_type(
        self, config_cls: type[ConfigComponent]
    ) -> ConfigComponent | None:
        """Get a configuration component by its class"""
        path = self.get_default_path(config_cls)
        if path:
            return self.get_component(path)
        return None

    def calculate_content_version(self) -> str:
        """Calculate a hash of the configuration content"""
        # Sort components by path for consistent hashing
        sorted_components = sorted(self._components.items(), key=lambda x: x[0])

        # Create a dictionary of component dumps
        content = {}
        for path, component in sorted_components:
            try:
                content[path] = component.model_dump_json()
            except Exception as e:
                raise ValueError(
                    f"Error dumping component {path} ({component.__class__.__name__}): {e}"
                )

        # Convert to JSON string with sorted keys
        content_json = json.dumps(content, sort_keys=True)

        # Calculate SHA-1 hash
        return hashlib.sha1(content_json.encode()).hexdigest()

    def is_compatible_with(self, other: PolymatheraConfig) -> bool:
        """Check if this configuration is compatible with another configuration"""
        return self.version.is_compatible_with(other.version)

    def has_changes_from(self, other: PolymatheraConfig) -> bool:
        """Check if this configuration has changes compared to another configuration"""
        return self.version.content_version != other.version.content_version

    def get_changed_components(
        self, other: PolymatheraConfig
    ) -> dict[str, tuple[Any, Any]]:
        """Get components that changed between this and another configuration"""
        changes = {}
        all_paths = set(self._components.keys()) | set(other._components.keys())

        for path in all_paths:
            self_component = self._components.get(path)
            other_component = other._components.get(path)

            if self_component is None or other_component is None:
                # Component added or removed
                changes[path] = (other_component, self_component)
            elif self_component.model_dump() != other_component.model_dump():
                # Component changed
                changes[path] = (other_component, self_component)

        return changes

    def create_new_version(
        self,
        schema_version: str | None = None,
        description: str | None = None,
        created_by: str | None = None,
    ) -> PolymatheraConfig:
        """Create a new version of this configuration"""
        config_dict = self.model_dump()

        # Update version information
        version_data = {
            "schema_version": schema_version or self.version.schema_version,
            "parent_version": self.version.content_version,
            "description": description,
            "created_by": created_by,
        }

        config_dict["version"] = version_data
        return PolymatheraConfig(**config_dict)

    @model_validator(mode="after")
    def validate_version(self) -> PolymatheraConfig:
        """Validate version information"""
        # Ensure content version matches actual content
        calculated_version = self.calculate_content_version()
        if self.version.content_version != calculated_version:
            self.version.content_version = calculated_version
        return self

    def model_dump(self) -> dict[str, Any]:
        """Convert config to dictionary preserving the hierarchy"""
        result = {"version": self.version.model_dump()}

        for path, component in self._components.items():
            current = result
            parts = path.split(".")

            # Create nested structure
            for part in parts[:-1]:
                current = current.setdefault(part, {})

            # Set component data
            current[parts[-1]] = component.model_dump()

        return result

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        """Generate JSON schema for the entire configuration"""
        # Generate base schema with refs
        base_schema = super().model_json_schema()

        # Create a new schema with just the properties we need
        schema = {
            "type": "object",
            "properties": {},
            "$defs": base_schema.get("$defs", {}),
        }
        if "title" in base_schema:
            schema["title"] = base_schema["title"]
        if "description" in base_schema:
            schema["description"] = base_schema["description"]

        # Copy version schema from base
        if "version" in base_schema.get("properties", {}):
            schema["properties"]["version"] = base_schema["properties"]["version"]

        # Add all registered components to the schema
        for path, config_cls in cls.get_registered_configs().items():
            current = schema["properties"]
            parts = path.split(".")

            # Create nested structure
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {"type": "object", "properties": {}}
                current = current[part]["properties"]

            # Add component schema
            current[parts[-1]] = config_cls.model_json_schema()
            # component_schema = config_cls.model_json_schema()
            # # Add component definitions to $defs
            # if '$defs' in component_schema:
            #     schema['$defs'].update(component_schema['$defs'])
            # # Add component ref
            # current[parts[-1]] = {'$ref': f'#/$defs/{config_cls.__name__}'}

        return schema
