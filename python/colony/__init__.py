"""LLM serving layer for Polymathera.

This package provides LLM cluster management, context-aware routing, and integration
with vLLM for high-performance inference.

Components:
    - cluster: LLM cluster layer built on distributed.ray_utils.serving
    - deployment_names: Centralized deployment name configuration
"""

# Re-export cluster components at package level for convenience
from .cluster import (
    ClusterStatistics,
    ContextAwareRouter,
    VirtualContextPage,
    ContextPageId,
    ContextPageState,
    InferenceRequest,
    InferenceResponse,
    LLMClientId,
    LLMClientState,
    LLMCluster,
    LLMClusterState,
    LoadedContextPage,
    PageAffinityRouter,
    VLLMDeployment,
)
from .deployment_names import DeploymentNames, get_deployment_names

__all__ = [
    # Cluster management
    "LLMCluster",
    "LLMClusterState",
    # Deployments
    "VLLMDeployment",
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
