from __future__ import annotations

import json
import logging
import os
from typing import Any, TypeVar

import yaml

from .configs import ConfigComponent, PolymatheraConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=ConfigComponent)


class ConfigurationManager:
    """Centralized configuration manager for Polymathera.

    Resolution order (lowest → highest):
        1. Defaults declared on each ``ConfigComponent``.
        2. Optional YAML/JSON file at ``config_path`` (one file, no search).
        3. Environment variables, applied via ``json_schema_extra["env"]``
           bindings on individual fields. The ``POLYMATHERA_<path>_<field>``
           catch-all in ``_apply_env_vars`` covers the rest.

    Components register themselves with ``@register_polymathera_config()`` at
    import time and are discovered into the root ``PolymatheraConfig`` model.
    Runtime updates flow through ``update_config()`` and notify the subscribers
    registered in ``_config_update_handlers``.
    """

    def __init__(self, config_path: str | None = None):
        """Initialize the configuration manager.

        ``config_path`` is the single YAML/JSON file to load (after defaults,
        before env-var overrides). When ``None``, only defaults + env vars apply.

        NOTE: Config field names must be lowercase (with underscores) so they
        match the env-var suffixes used by ``_apply_env_vars``.
        """
        self.config_path = config_path
        self._initialized = False  # Prevent re-initialization
        self._config: PolymatheraConfig | None = None
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
        """Load configuration. Idempotent."""
        if self._initialized:
            return

        await self._load_config()

        self._initialized = True

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
        """Convert flat ``"a.b.c"`` paths into nested dicts; pop ``version`` aside."""
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
        """Load configuration: defaults ⊕ optional YAML at ``config_path`` ⊕ env vars.

        NOTE: Config field names must be lowercase (with underscores) so they match
        the environment-variable suffixes used by ``_apply_env_vars``.
        """
        logger.info(
            f"Loading configuration for config manager: ObjId={id(self)}"
        )
        # Start with default config (not an empty config)
        config_dict = PolymatheraConfig().model_dump()

        components_dict, version_data = self._flatten_config(config_dict)

        if self.config_path:
            logger.info(f"Loading config file: {self.config_path}")
            file_components_dict, _ = self._flatten_config(
                self._load_config_file(self.config_path)
            )
            components_dict = self._deep_merge(
                components_dict, file_components_dict
            )

        # Apply environment variables (highest priority except overrides)
        self._apply_env_vars(components_dict)

        self._config = PolymatheraConfig(version=version_data, **components_dict)

    def _apply_env_vars(self, components_dict: dict[str, Any]) -> None:
        """Apply ``POLYMATHERA_<config_path>_<field>`` environment variables.

        Field names must be lowercase (with underscores) so they round-trip
        cleanly through env-var naming. Values are JSON-decoded when possible.
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

    def update_config(self, updates: dict[str, Any]) -> None:
        """Update configuration with new values and notify subscribers."""
        if not self._config:
            return

        # Get current config as dict
        config_dict = self.config.model_dump()

        # Apply updates
        config_dict = self._deep_merge(config_dict, updates)

        # Create new config object
        self._config = PolymatheraConfig(**config_dict)

        # Notify subscribers
        self._notify_config_updated()

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

