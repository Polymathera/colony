"""Registry for remote LLM deployment classes keyed by provider name.

Replaces the hardcoded ``if rconf.provider == "anthropic" / "openrouter"``
switch in :mod:`cluster.config`. Built-in providers register themselves at
module-import time via :func:`register_remote_llm_provider`; extensions register
the same way (typically inside a ``polymathera.config_components`` entry
point — see :mod:`distributed.config.extensions`).

Builtins are lazy-imported on first miss so CPU-only environments without
the optional ``vllm`` extra never pay for unused module loads.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .remote_deployment import RemoteLLMDeployment

logger = logging.getLogger(__name__)


_REGISTRY: dict[str, type[RemoteLLMDeployment]] = {}


# Provider name → fully-qualified module path. Imported on first lookup miss
# to populate the registry; the imported module's class is expected to call
# ``register_remote_llm_provider`` at import time.
_BUILTIN_LAZY_MODULES: dict[str, str] = {
    "anthropic": "polymathera.colony.cluster.anthropic_deployment",
    "openrouter": "polymathera.colony.cluster.openrouter_deployment",
    "vllm": "polymathera.colony.cluster.vllm_remote_deployment",
}


def register_remote_llm_provider(name: str):
    """Decorator: register a :class:`RemoteLLMDeployment` subclass under ``name``.

    Last writer wins (with a warning) so an extension can intentionally
    override a colony-shipped default.
    """

    def _decorator(cls):
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            logger.warning(
                "register_remote_llm_provider: %r already bound to %s; overwriting with %s",
                name, _REGISTRY[name].__name__, cls.__name__,
            )
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_remote_llm_deployment_class(name: str) -> type[RemoteLLMDeployment]:
    """Return the registered deployment class for provider ``name``.

    Lazy-imports the matching builtin module on first miss. Raises
    :class:`ValueError` if the provider is unknown.
    """
    cls = _REGISTRY.get(name)
    if cls is not None:
        return cls
    module_path = _BUILTIN_LAZY_MODULES.get(name)
    if module_path is not None:
        importlib.import_module(module_path)
        cls = _REGISTRY.get(name)
        if cls is not None:
            return cls
    raise ValueError(
        f"Unknown remote provider: {name!r}. Registered: "
        f"{sorted(set(_REGISTRY) | set(_BUILTIN_LAZY_MODULES))}"
    )


def list_remote_llm_providers() -> list[str]:
    """All currently-registered provider names (does not force lazy imports)."""
    return sorted(_REGISTRY.keys())


__all__ = (
    "get_remote_llm_deployment_class",
    "list_remote_llm_providers",
    "register_remote_llm_provider",
)
