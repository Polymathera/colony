"""LLM serving layer for Polymathera.

This package provides LLM cluster management, context-aware routing, and integration
with vLLM for high-performance inference.

Components:
    - cluster: LLM cluster layer built on distributed.ray_utils.serving
    - deployment_names: Centralized deployment name configuration

Imports are lazy (PEP 562) so that lightweight submodules (e.g. the colony-env
CLI) can be loaded without pulling in heavy GPU dependencies like vLLM/PyTorch.
"""

import importlib as _importlib

_CLUSTER_NAMES = {
    "ClusterStatistics",
    "ContextAwareRouter",
    "VirtualContextPage",
    "ContextPageId",
    "ContextPageState",
    "InferenceRequest",
    "InferenceResponse",
    "LLMClientId",
    "LLMClientState",
    "LLMCluster",
    "LLMClusterState",
    "LoadedContextPage",
    "PageAffinityRouter",
    "VLLMDeployment",
    "RemoteLLMDeployment",
    "AnthropicLLMDeployment",
    "OpenRouterLLMDeployment",
}

_DEPLOYMENT_NAMES = {
    "DeploymentNames",
    "get_deployment_names",
}

__all__ = [
    # Cluster management
    "LLMCluster",
    "LLMClusterState",
    # Deployments
    "VLLMDeployment",
    "RemoteLLMDeployment",
    "AnthropicLLMDeployment",
    "OpenRouterLLMDeployment",
    # Routing
    "ContextAwareRouter",
    "PageAffinityRouter",
    # Models
    "VirtualContextPage",
    "ContextPageId",
    "ContextPageState",
    "LoadedContextPage",
    "InferenceRequest",
    "InferenceResponse",
    "LLMClientId",
    "LLMClientState",
    "ClusterStatistics",
    # Configuration
    "DeploymentNames",
    "get_deployment_names",
]


def __getattr__(name: str):
    if name in _CLUSTER_NAMES:
        mod = _importlib.import_module(".cluster", __name__)
        return getattr(mod, name)
    if name in _DEPLOYMENT_NAMES:
        mod = _importlib.import_module(".deployment_names", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(globals()) + __all__
