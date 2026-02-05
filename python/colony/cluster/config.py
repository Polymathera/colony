"""Cluster configuration using model registry.

This module provides configuration classes for LLM cluster deployment,
integrating with the model registry to provide validated, model-specific configurations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from ..distributed.ray_utils import serving


if TYPE_CHECKING:
    from .registry import LLMModelParameters, QuantizationMethod

from .registry import ModelRegistry

logger = logging.getLogger(__name__)


class LoRAAdapterConfig(BaseModel):
    """Configuration for a LoRA adapter in multi-LoRA serving.

    This configuration specifies a LoRA adapter that can be dynamically
    loaded and used with a base model for fine-tuned inference.

    Example:
        ```python
        adapter = LoRAAdapterConfig(
            adapter_id="customer-123-finance",
            adapter_name="my-org/llama-finance-adapter",
            base_model_name="meta-llama/Llama-3.1-8B",
            rank=16,
            alpha=32,
        )
        ```
    """

    adapter_id: str = Field(description="Unique identifier for this LoRA adapter")
    adapter_name: str = Field(description="HuggingFace adapter name or path")
    base_model_name: str = Field(description="Base model this adapter is for")

    # LoRA configuration
    rank: int = Field(
        default=8,
        ge=1,
        le=256,
        description="LoRA rank (r parameter)"
    )
    alpha: int = Field(
        default=16,
        ge=1,
        description="LoRA alpha scaling parameter"
    )
    target_modules: list[str] | None = Field(
        default=None,
        description="Target modules for LoRA (e.g., ['q_proj', 'v_proj'])"
    )

    # Storage
    s3_bucket: str | None = Field(
        default=None,
        description="S3 bucket for adapter weights"
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (owner, version, etc.)"
    )


class DeploymentConfig(BaseModel):
    """Configuration for a single vLLM deployment instance.

    This configuration is derived from model registry parameters and
    cluster-level settings, with validation to ensure compatibility.
    """

    # Deployment identification
    deployment_id: str | None = Field(
        default=None,
        description="Unique ID for this deployment (auto-generated from model name if None)"
    )

    # Model configuration
    model_name: str = Field(description="HuggingFace model name or path")
    quantization: str | None = Field(
        default=None,
        description="Quantization method (awq, gptq, fp8, etc.)"
    )

    # Resource allocation
    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs for tensor parallelism"
    )
    gpu_memory_utilization: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Fraction of GPU memory to use"
    )

    # Capacity configuration (auto-calculated if None)
    kv_cache_capacity: int | None = Field(
        default=None,
        description="KV cache capacity in tokens (auto-calculated if None)"
    )
    max_model_len: int | None = Field(
        default=None,
        description="Maximum model context length (from registry if None)"
    )

    # Capabilities
    capabilities: set[str] = Field(
        default_factory=lambda: {"structured_output"},
        description="Model capabilities (e.g., structured_output, function_calling)"
    )

    # Multi-LoRA serving
    lora_adapters: list[LoRAAdapterConfig] | None = Field(
        default=None,
        description="LoRA adapters for multi-LoRA serving (STUB - not yet implemented)"
    )

    # Multi-tenancy
    dedicated_tenant_ids: set[str] = Field(
        default_factory=set,
        description="Tenant IDs that require dedicated deployment (Option B isolation)"
    )

    # S3 model loading
    s3_bucket: str | None = Field(
        default=None,
        description="S3 bucket for model loading"
    )
    s3_retry_attempts: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Number of S3 download retry attempts"
    )

    # Other settings
    trust_remote_code: bool = Field(
        default=True,
        description="Whether to trust remote code when loading model"
    )
    gpu_id: int = Field(
        default=0,
        ge=0,
        description="GPU device ID for this deployment"
    )

    # Scaling configuration (per-deployment)
    num_replicas: int = Field(
        default=2,
        ge=1,
        description="Initial number of replicas for this deployment"
    )
    min_replicas: int | None = Field(
        default=None,
        ge=1,
        description="Minimum replicas for autoscaling (defaults to num_replicas)"
    )
    max_replicas: int | None = Field(
        default=None,
        ge=1,
        description="Maximum replicas for autoscaling (defaults to 2 * num_replicas)"
    )
    target_queue_length: int = Field(
        default=5,
        ge=1,
        description="Target queue length for autoscaling"
    )

    # Routing within deployment (replica selection)
    default_router_class: str = Field(
        default="ContextAwareRouter",
        description="Routing policy within deployment replicas (ContextAwareRouter, PageAffinityRouter, RoundRobin)"
        " unless overridden by routing hints in @serving.endpoint."
    )

    # Agent capacity limits (for deployments that host agents)
    max_agents_per_replica: int = Field(
        default=100,
        ge=1,
        description="Maximum number of agents per replica (prevents resource exhaustion)"
    )
    max_cpu_cores_per_replica: float = Field(
        default=8.0,
        ge=0.1,
        description="Total CPU cores available per replica for agents"
    )
    max_memory_mb_per_replica: int = Field(
        default=16384,  # 16GB
        ge=512,
        description="Total memory (MB) available per replica for agents"
    )
    max_gpu_cores_per_replica: float = Field(
        default=0.0,
        ge=0.0,
        description="Total GPU cores available per replica for agents (0.0 = no GPU)"
    )
    max_gpu_memory_mb_per_replica: int = Field(
        default=0,
        ge=0,
        description="Total GPU memory (MB) available per replica for agents"
    )

    def model_post_init(self, __context):
        """Set default min/max replicas if not provided (Pydantic v2)."""
        if self.min_replicas is None:
            self.min_replicas = self.num_replicas
        if self.max_replicas is None:
            self.max_replicas = self.num_replicas * 2

    def get_deployment_name(self) -> str:
        """Get deployment name for this configuration.

        Returns:
            deployment_id if set, otherwise auto-generated from model name
        """
        if self.deployment_id:
            return self.deployment_id
        # Auto-generate: convert "meta-llama/Llama-3.1-8B" to "vllm-llama-3-1-8b"
        return f"vllm-{self.model_name.lower().replace('/', '-').replace('.', '-')}"

    @field_validator("model_name")
    @classmethod
    def validate_model_in_registry(cls, v: str) -> str:
        """Validate that model exists in registry."""
        model_params = ModelRegistry.get_model(v)
        if model_params is None:
            logger.warning(
                f"Model {v} not found in registry. "
                "Using default configuration. Consider adding to registry."
            )
        return v

    @field_validator("quantization")
    @classmethod
    def validate_quantization(cls, v: str | None, info) -> str | None:
        """Validate quantization method is supported by model."""
        if v is None:
            return v

        model_name = info.data.get("model_name")
        if not model_name:
            return v

        model_params = ModelRegistry.get_model(model_name)
        if model_params is None:
            # Model not in registry, can't validate
            return v

        # Try to parse quantization method
        from .registry import QuantizationMethod
        try:
            quant_method = QuantizationMethod[v.upper().replace("-", "_")]

            # Check if model supports this quantization
            if quant_method not in model_params.supported_quantizations:
                logger.warning(
                    f"Model {model_name} may not support {v} quantization. "
                    f"Supported: {[q.name.lower() for q in model_params.supported_quantizations]}"
                )
        except KeyError:
            logger.warning(f"Unknown quantization method: {v}")

        return v

    @classmethod
    def from_model_registry(
        cls,
        model_name: str,
        tensor_parallel_size: int = 1,
        quantization: str | None = None,
        s3_bucket: str | None = None,
        **overrides,
    ) -> "DeploymentConfig":
        """Create deployment config from model registry.

        Args:
            model_name: Model name to look up in registry
            tensor_parallel_size: Number of GPUs for tensor parallelism
            quantization: Quantization method (uses model default if None)
            s3_bucket: S3 bucket for model loading
            **overrides: Additional configuration overrides

        Returns:
            DeploymentConfig instance with registry-derived settings

        Example:
            ```python
            config = DeploymentConfig.from_model_registry(
                model_name="meta-llama/Llama-3.1-8B",
                tensor_parallel_size=2,
                quantization="awq",
                s3_bucket="my-models"
            )
            ```
        """
        model_params = ModelRegistry.get_model(model_name)

        if model_params is None:
            logger.warning(
                f"Model {model_name} not found in registry. "
                "Using default configuration."
            )
            return cls(
                model_name=model_name,
                tensor_parallel_size=tensor_parallel_size,
                quantization=quantization,
                s3_bucket=s3_bucket,
                **overrides
            )

        # Use model's default quantization if not specified
        if quantization is None and model_params.default_quantization:
            from .registry import QuantizationMethod
            if model_params.default_quantization != QuantizationMethod.NONE:
                quantization = model_params.default_quantization.name.lower()

        # Build configuration from registry
        config_dict = {
            "model_name": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "quantization": quantization,
            "s3_bucket": s3_bucket,
            "max_model_len": model_params.context_window,
            # Note: kv_cache_capacity will be auto-calculated based on GPU memory
        }

        # Apply overrides
        config_dict.update(overrides)

        return cls(**config_dict)

    def get_model_params(self) -> LLMModelParameters | None:
        """Get model parameters from registry.

        Returns:
            LLMModelParameters if model is in registry, None otherwise
        """
        return ModelRegistry.get_model(self.model_name)

    def validate_against_registry(self) -> list[str]:
        """Validate configuration against model registry.

        Returns:
            List of validation warnings (empty if all valid)
        """
        warnings = []

        model_params = self.get_model_params()
        if model_params is None:
            warnings.append(f"Model {self.model_name} not found in registry")
            return warnings

        # Validate GPU requirements
        if self.tensor_parallel_size < model_params.num_gpus:
            warnings.append(
                f"Model {model_params.model_name} recommends {model_params.num_gpus} GPUs, "
                f"but only {self.tensor_parallel_size} configured"
            )

        # Validate quantization
        if self.quantization:
            from .registry import QuantizationMethod
            try:
                quant = QuantizationMethod[self.quantization.upper().replace("-", "_")]
                if quant not in model_params.supported_quantizations:
                    warnings.append(
                        f"Quantization {self.quantization} not in supported list: "
                        f"{[q.name.lower() for q in model_params.supported_quantizations]}"
                    )
            except KeyError:
                warnings.append(f"Unknown quantization: {self.quantization}")

        return warnings


class ClusterConfig(BaseModel):
    """Configuration for an LLM cluster deployment.

    Supports multiple vLLM deployments for different models/tasks with
    requirement-based routing and per-deployment autoscaling.

    Routing architecture:
    - Level 1 (Cluster): deployment_routing_policy selects which deployment
    - Level 2 (Deployment): default_router_class (or endpoint-specific router class) selects which replica within deployment
    """

    # Application configuration
    app_name: str = Field(description="Name for the serving application")

    # Multi-deployment configuration
    vllm_deployments: list[DeploymentConfig] = Field(
        description="List of vLLM deployment configurations for different models"
    )

    # Optional embedding deployment
    embedding_config: DeploymentConfig | None = Field(
        default=None,
        description="Optional configuration for embedding model deployment"
    )

    # Deployment-level routing (which deployment to use)
    deployment_routing_policy: str = Field(
        default="RequirementBasedRouter",
        description="Routing policy for selecting deployment (RequirementBasedRouter, RoundRobin)"
    )
    enable_fallbacks: bool = Field(
        default=True,
        description="Enable fallback routing if primary deployment unavailable"
    )

    # Multi-tenancy
    enable_tenant_isolation: bool = Field(
        default=True,
        description="Enable tenant isolation via tenant_id in KV cache hash"
    )
    isolated_tenants: set[str] = Field(
        default_factory=set,
        description="Tenants requiring dedicated instances (not yet implemented)"
    )

    # Cleanup configuration
    cleanup_on_init: bool = Field(
        default=False,
        description="Cleanup all existing deployments and states before initializing (useful for testing)"
    )

    def validate_config(self) -> None:
        # Validate all deployment configurations
        for dconf in self.vllm_deployments:
            warnings = dconf.validate_against_registry()
            for warning in warnings:
                logger.warning(f"Deployment {dconf.get_deployment_name()}: {warning}")

    def add_deployments_to_app(self, app: serving.Application, top_level: bool) -> None:
        """Add LLMCluster and all VLLMDeployment instances to the application.

        This is called by higher-level configs (e.g., PolymatheraClusterConfig) to
        recursively build the full deployment hierarchy.

        Args:
            app: The serving Application to add deployments to
            top_level: If True, don't deploy LLMCluster itself (it's on the driver)
        """
        logger.info(
            f"Deploying LLM cluster '{self.app_name}' to existing application "
            f"with {len(self.vllm_deployments)} vLLM deployment(s)"
        )
        from ..cluster import LLMCluster
        from .vllm_deployment import VLLMDeployment
        from .embedding_deployment import EmbeddingDeployment
        from .routing import get_routing_policy_class

        # Add LLMCluster deployment (unless it's top-level on driver)
        if not top_level:
            app.add_deployment(
                LLMCluster.bind(config=self, top_level=False),
                name="llm_cluster"
            )

        # Add each vLLM deployment
        for dconf in self.vllm_deployments:
            deployment_name = dconf.get_deployment_name()
            logger.info(
                f"Adding deployment '{deployment_name}': model={dconf.model_name}, "
                f"replicas={dconf.num_replicas}, tensor_parallel_size={dconf.tensor_parallel_size}"
            )

            # Get routing policy class for this deployment
            # Parse default_router_class (could be "ContextAwareRouter" or other policy names)
            default_router_class = get_routing_policy_class(dconf.default_router_class)

            app.add_deployment(
                VLLMDeployment.bind(
                    model_name=dconf.model_name,
                    kv_cache_capacity=dconf.kv_cache_capacity,
                    max_model_len=dconf.max_model_len,
                    tensor_parallel_size=dconf.tensor_parallel_size,
                    gpu_memory_utilization=dconf.gpu_memory_utilization,
                    trust_remote_code=dconf.trust_remote_code,
                    gpu_id=dconf.gpu_id,
                    quantization=dconf.quantization,
                    s3_bucket=dconf.s3_bucket,
                    s3_retry_attempts=dconf.s3_retry_attempts,
                ),
                name=deployment_name,
                default_router_class=default_router_class,
                autoscaling_config={
                    "min_replicas": dconf.min_replicas,
                    "max_replicas": dconf.max_replicas,
                    "target_queue_length": dconf.target_queue_length,
                },
                ray_actor_options={
                    "num_gpus": dconf.tensor_parallel_size,
                },
            )

        # Add embedding deployment if configured
        if self.embedding_config:
            econf = self.embedding_config
            logger.info(f"Adding embedding deployment: {econf.model_name}")
            app.add_deployment(
                EmbeddingDeployment.bind(
                    model_name=econf.model_name,
                    tensor_parallel_size=econf.tensor_parallel_size,
                    gpu_memory_utilization=econf.gpu_memory_utilization,
                    trust_remote_code=econf.trust_remote_code,
                    gpu_id=econf.gpu_id,
                    quantization=econf.quantization,
                    s3_bucket=econf.s3_bucket,
                    s3_retry_attempts=econf.s3_retry_attempts,
                ),
                name="embedding",
                ray_actor_options={
                    "num_gpus": econf.tensor_parallel_size,
                },
            )
