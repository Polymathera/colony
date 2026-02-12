"""LLM Cluster Layer for Polymathera.

This package provides a production-ready LLM serving infrastructure built on top of
colony.distributed.ray_utils.serving and vLLM. It enables:

1. **Context-Aware Serving**: Deploy vLLM instances with KV cache management for
   extremely long contexts (billion-token scale)

2. **Intelligent Routing**: Route inference requests based on which context pages
   are loaded in each LLM's KV cache, minimizing expensive page faults

3. **Distributed State**: Track page locations and client states across the cluster
   using distributed state management

4. **Scalability**: Auto-scaling vLLM replicas with load-based policies

Architecture Layers:
    - **Bottom**: colony.distributed.ray_utils.serving (deployment infrastructure)
    - **This Layer**: LLM cluster with context caching and routing
    - **Above (Future)**: Virtual Context Manager (VCM)

Quick Start:
    ```python
    from polymathera.colony.cluster import LLMCluster, InferenceRequest

    # Deploy cluster
    cluster = LLMCluster(
        app_name="my-llm-cluster",
        model_name="meta-llama/Llama-3.1-8B",
        num_replicas=4,
        top_level=True,
    )
    await cluster.deploy()

    # Perform inference
    request = InferenceRequest(
        request_id="req-1",
        prompt="Explain quantum computing",
        context_page_ids=["page-1", "page-2"],
    )
    response = await cluster.infer(request)
    print(response.generated_text)
    ```

Components:
    - VLLMDeployment: vLLM-based deployment with context caching
    - ContextAwareRouter: Intelligent routing based on page locality
    - LLMCluster: High-level cluster management
    - Models: Data structures for pages, requests, responses, state
"""

from .cluster import LLMCluster
from .config import ClusterConfig, LLMDeploymentConfig, LoRAAdapterConfig
from .embedding_deployment import EmbeddingDeployment
from .models import (
    ClusterStatistics,
    ContextPageState,
    InferenceRequest,
    InferenceResponse,
    LLMClientId,
    LLMClientRequirements,
    LLMClientState,
    LoadedContextPage,
    VLLMDeploymentState,
    LLMClusterState
)
from .registry import (
    LLMBackend,
    LLMCapability,
    LLMModelParameters,
    LLMSize,
    ModelRegistry,
    QuantizationMethod,
)
from .remote_config import RemoteLLMDeploymentConfig
from .remote_deployment import RemoteLLMDeployment
from .anthropic_deployment import AnthropicLLMDeployment
from .openrouter_deployment import OpenRouterLLMDeployment
from .routing import ContextAwareRouter, PageAffinityRouter
from .vllm_deployment import VLLMDeployment

__all__ = [
    # Cluster management
    "LLMCluster",
    "LLMClusterState",
    # Configuration
    "ClusterConfig",
    "LLMDeploymentConfig",
    "LoRAAdapterConfig",
    "RemoteLLMDeploymentConfig",
    # Deployments
    "VLLMDeployment",
    "EmbeddingDeployment",
    "RemoteLLMDeployment",
    "AnthropicLLMDeployment",
    "OpenRouterLLMDeployment",
    # Routing
    "ContextAwareRouter",
    "PageAffinityRouter",
    # Models
    "ContextPageState",
    "LoadedContextPage",
    "InferenceRequest",
    "InferenceResponse",
    "LLMClientId",
    "LLMClientRequirements",
    "LLMClientState",
    "ClusterStatistics",
    "VLLMDeploymentState",
    # Registry
    "ModelRegistry",
    "LLMModelParameters",
    "LLMCapability",
    "LLMSize",
    "LLMBackend",
    "QuantizationMethod",
]
