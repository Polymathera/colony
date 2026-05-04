from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, TypeVar

import yaml

from .configs import ConfigComponent, PolymatheraConfig
from .extensions import discover_config_components
from .overlays import (
    OverlayScope,
    OverlayStore,
    assert_writable_at_scope,
    compose_overlays,
)

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

    def __init__(
        self,
        config_path: str | None = None,
        overlay_store: OverlayStore | None = None,
        wait_for_config_seconds: float = 15.0,
    ):
        """Initialize the configuration manager.

        ``config_path`` is the single YAML/JSON file to load (after defaults,
        before env-var overrides). When ``None``, only defaults + env vars apply.

        ``overlay_store`` (optional) is the StateManager-backed sink for
        L2/L3/L4 overlays — see :mod:`.overlays`. When ``None``, only the
        in-process L1 view is available; ``get_component_for`` collapses to
        ``get_component`` and ``update_overlay`` raises.

        ``wait_for_config_seconds`` is the deadline ``initialize()`` waits for
        ``config_path`` to materialise before falling through to defaults +
        env vars. The default tolerates the colony-env ``docker cp`` race
        (file copied into the shared volume *after* containers start) without
        each consumer having to re-implement the loop. Set to ``0`` to skip
        the wait — useful for tests and for processes that intentionally run
        without an operator YAML. Ignored when ``config_path`` is ``None`` or
        when the file already exists.

        NOTE: Config field names must be lowercase (with underscores) so they
        match the env-var suffixes used by ``_apply_env_vars``.
        """
        self.config_path = config_path
        self._overlay_store = overlay_store
        self._wait_for_config_seconds = wait_for_config_seconds
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

    @property
    def is_initialized(self) -> bool:
        """Whether ``initialize()`` has populated the in-memory config."""
        return self._initialized

    async def initialize(self) -> None:
        """Load configuration. Idempotent.

        Discovers extension-supplied ``ConfigComponent``s before loading so any
        components they register participate in defaults + YAML resolution.
        It waits up to ``wait_for_config_seconds`` for ``config_path`` to
        materialise (a no-op when ``config_path`` is ``None`` or the file already
        exists), then loads defaults + YAML + env-var overrides.
        """
        if self._initialized:
            return

        discover_config_components()
        await self._wait_for_config_path()
        await self._load_config()

        self._initialized = True

    async def _wait_for_config_path(self) -> None:
        """Poll until ``self.config_path`` exists or the deadline passes.

        Tolerates the ``colony-env up`` ``docker cp`` race: the operator YAML
        is copied into the shared volume *after* containers start, so a
        Python process that calls ``initialize()`` early would otherwise
        miss it. On timeout we log a warning and let ``_load_config`` fall
        through to the defaults + env-var path — same behaviour as if no
        ``config_path`` had been set, plus a visible warning so an operator
        can diagnose the missing file.
        """
        path = self.config_path
        timeout = self._wait_for_config_seconds
        if not path or timeout <= 0 or os.path.exists(path):
            return

        poll_interval = max(0.05, min(0.5, timeout / 30))
        deadline = asyncio.get_event_loop().time() + timeout
        logger.info(
            f"Waiting up to {timeout:.1f}s for config file to appear at {path}"
        )
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(poll_interval)
            if os.path.exists(path):
                logger.info(f"Config file found at {path}")
                return
        logger.warning(
            f"Config file at {path} did not appear within {timeout:.1f}s; "
            f"continuing with defaults + env vars only."
        )

    def set_config_path(self, path: str | None) -> None:
        """Update ``config_path`` and mark the manager dirty so the next
        :meth:`initialize` call (or an explicit awaitable returned by the
        caller) re-reads from the new file.

        Used by the CLI to bridge ``--config`` into the global manager when
        the file path is only known after the typer command parses its args.
        """
        if path == self.config_path:
            return
        self.config_path = path
        self._initialized = False

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
        """Update L1 configuration with new values in-process and notify subscribers."""
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

    async def get_component_for(
        self,
        path: str,
        *,
        tenant_id: str | None = None,
        session_id: str | None = None,
    ) -> ConfigComponent | None:
        """Return ``path`` composed across L1 + tenant + session + runtime overlays.

        Equivalent to :meth:`get_component` when no ``overlay_store`` is wired
        or when neither ``tenant_id`` nor ``session_id`` is supplied (and no
        runtime overlays exist). Re-instantiates the component class from the
        merged dict so its field validators run on the composed view.
        """
        base = self.get_component(path)
        if base is None or self._overlay_store is None:
            return base

        state = await self._overlay_store.read()
        merged = compose_overlays(
            base.model_dump(),
            state,
            path=path,
            tenant_id=tenant_id,
            session_id=session_id,
        )
        return type(base)(**merged)

    async def update_overlay(
        self, path: str, updates: dict[str, Any], *, scope: OverlayScope,
    ) -> None:
        """Write ``updates`` to overlay ``scope`` for component ``path``.

        Tier-checked against the field metadata declared on the component:
        a write fails fast if any updated field's declared tier is below the
        overlay's tier. Subscribers are notified after the write commits.
        """
        if self._overlay_store is None:
            raise RuntimeError(
                "ConfigurationManager: overlay_store not configured; "
                "construct with overlay_store=OverlayStore(...) to enable overlays."
            )
        component = self.get_component(path)
        if component is None:
            raise KeyError(f"unknown config component path: {path}")
        assert_writable_at_scope(type(component), updates, scope)
        await self._overlay_store.write(scope, path, updates)
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


def get_component_or_default(path: str, cls: type[T]) -> T:
    """Return the registered component if the global manager has loaded it; else defaults.

    Sync-safe shim for capability constructors that cannot await the manager's
    initialize coroutine. Falls back to ``cls()`` when the global ``polymathera``
    singleton has not been initialized yet (e.g. unit tests constructing
    capabilities directly without booting the full app).
    """
    try:
        from .. import get_polymathera as _get_polymathera
        cm = _get_polymathera().config_manager
    except Exception:  # noqa: BLE001 — distributed/system import side effects
        return cls()
    if not cm.is_initialized:
        return cls()
    component = cm.get_component(path)
    return component if isinstance(component, cls) else cls()

