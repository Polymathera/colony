"""Configuration for remote LLM deployments (Anthropic / OpenRouter).

This module provides configuration classes for remote LLM deployments that
call external APIs instead of running local GPU-based vLLM engines. These
deployments are drop-in replacements for VLLMDeployment from the VCM's
perspective.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Pricing per million tokens (USD) for common models.
# Used for cost estimation in InferenceResponse.metadata.
# Format: {model_name_prefix: {input, output, cache_read, cache_write_5m, cache_write_1h}}
ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,       # 0.1x input
        "cache_write_5m": 3.75,   # 1.25x input
        "cache_write_1h": 6.0,    # 2.0x input
    },
    "claude-opus-4": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.50,       # 0.1x input
        "cache_write_5m": 18.75,  # 1.25x input
        "cache_write_1h": 30.0,   # 2.0x input
    },
    "claude-haiku-4": {
        "input": 0.80,
        "output": 4.0,
        "cache_read": 0.08,       # 0.1x input
        "cache_write_5m": 1.0,    # 1.25x input
        "cache_write_1h": 1.6,    # 2.0x input
    },
}


def get_pricing_for_model(model_name: str) -> dict[str, float] | None:
    """Look up pricing for a model name by prefix match.

    Args:
        model_name: Full model name (e.g., "claude-sonnet-4-20250514")

    Returns:
        Pricing dict if found, None otherwise
    """
    for prefix, pricing in ANTHROPIC_PRICING.items():
        if model_name.startswith(prefix):
            return pricing
    return None


class RemoteLLMDeploymentConfig(BaseModel):
    """Configuration for a remote LLM deployment (Anthropic / OpenRouter).

    This is the remote equivalent of LLMDeploymentConfig. Each replica wraps
    a remote API instead of a local vLLM engine, with simulated KV cache
    capacity based on configured limits.

    Example:
        ```python
        config = RemoteLLMDeploymentConfig(
            model_name="claude-sonnet-4-20250514",
            provider="anthropic",
            max_cached_tokens=2_000_000,
            ttl="1h",
        )
        ```
    """

    # Deployment identification
    deployment_id: str | None = Field(
        default=None,
        description="Unique ID for this deployment (auto-generated from model name if None)"
    )

    # Model and provider
    model_name: str = Field(
        description="Model name for the API (e.g., 'claude-sonnet-4-20250514')"
    )
    provider: Literal["anthropic", "openrouter"] = Field(
        description="LLM API provider"
    )
    api_key_env_var: str = Field(
        default="ANTHROPIC_API_KEY",
        description="Environment variable containing the API key"
    )

    # Simulated capacity (analog of GPU KV cache size)
    # VCM's AllocationStrategy uses these to make placement decisions.
    max_cached_pages: int = Field(
        default=50,
        ge=1,
        description="Maximum number of pages this replica can cache"
    )
    max_cached_tokens: int = Field(
        default=2_000_000,
        ge=1000,
        description="Simulated KV cache capacity in tokens (controls how many VCM pages fit)"
    )

    # Caching configuration
    system_prompt: str | None = Field(
        default=None,
        description="System prompt to use (cached at first breakpoint)"
    )
    ttl: Literal["5m", "1h"] = Field(
        default="1h",
        description="Cache TTL for Anthropic prefix caching (1h recommended for cost savings)"
    )

    # Rate limiting
    max_concurrent_requests: int = Field(
        default=10,
        ge=1,
        description="Maximum concurrent API requests per replica"
    )

    # Scaling configuration (mirrors LLMDeploymentConfig)
    num_replicas: int = Field(
        default=1,
        ge=1,
        description="Initial number of replicas"
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
        description="Routing policy within deployment replicas"
    )

    # Capabilities (for RequirementBasedRouter matching)
    capabilities: set[str] = Field(
        default_factory=lambda: {"structured_output"},
        description="Model capabilities"
    )

    # OpenRouter-specific
    openrouter_provider_order: list[str] | None = Field(
        default=None,
        description="Preferred provider order for OpenRouter (e.g., ['anthropic', 'google'])"
    )

    def model_post_init(self, __context: Any) -> None:
        """Set default min/max replicas if not provided."""
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
        # Auto-generate: "remote-claude-sonnet-4-20250514"
        return f"remote-{self.model_name.lower().replace('/', '-').replace('.', '-')}"

    def get_ttl_seconds(self) -> int:
        """Get TTL in seconds."""
        return 300 if self.ttl == "5m" else 3600

    def get_pricing(self) -> dict[str, float] | None:
        """Get pricing for this model.

        Returns:
            Pricing dict if model is known, None otherwise
        """
        return get_pricing_for_model(self.model_name)
