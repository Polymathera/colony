import importlib as _importlib

from .remote_embedding_config import RemoteEmbeddingConfig, get_embedding_pricing
from .remote_embedding_deployment import (
    RemoteEmbeddingDeployment,
    OpenAICompatibleEmbeddingDeployment,
    GeminiEmbeddingDeployment
)
from .st_embedding import (
    STEmbeddingDeployment,
    STEmbeddingModel,
    STEmbeddingDeploymentConfig,
    TextChunkerBase,
)

# EmbeddingDeployment depend on vllm (GPU optional dep).
# Lazy-loaded via __getattr__ so CPU-only environments work.
_VLLM_NAMES = {
    "EmbeddingDeployment": ".embedding_deployment",
}



__all__ = [
    # Config and utils
    "RemoteEmbeddingConfig",
    "get_embedding_pricing",
    # Deployments
    "RemoteEmbeddingDeployment",
    "OpenAICompatibleEmbeddingDeployment",
    "GeminiEmbeddingDeployment",
    "EmbeddingDeployment",
    "STEmbeddingDeployment",
    "STEmbeddingModel",
    "STEmbeddingDeploymentConfig",
    "TextChunkerBase",
]


def __getattr__(name: str):
    if name in _VLLM_NAMES:
        mod = _importlib.import_module(_VLLM_NAMES[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(globals()) + list(_VLLM_NAMES)

