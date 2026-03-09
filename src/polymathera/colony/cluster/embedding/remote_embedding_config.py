"""Configuration for remote embedding deployments (OpenAI / Gemini / OpenRouter).

This module provides configuration for embedding deployments that call external
APIs instead of running local GPU-based vLLM embedding engines. These deployments
are drop-in replacements for EmbeddingDeployment from the LLMCluster's perspective.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Pricing per million tokens (USD) for common embedding models.
# Format: {model_name_prefix: price_per_million_tokens}
EMBEDDING_PRICING: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}

# Provider-specific defaults
_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai": {
        "api_key_env_var": "OPENAI_API_KEY",
    },
    "gemini": {
        "api_key_env_var": "GOOGLE_API_KEY",
        "max_batch_size": 100,
    },
    "openrouter": {
        "api_key_env_var": "OPENROUTER_API_KEY",
    },
}


def get_embedding_pricing(model_name: str) -> float | None:
    """Look up embedding pricing by model name prefix match.

    Args:
        model_name: Full model name (e.g., "text-embedding-3-small")

    Returns:
        Price per million tokens if found, None otherwise.
    """
    for prefix, price in EMBEDDING_PRICING.items():
        if model_name.startswith(prefix):
            return price
    return None


class RemoteEmbeddingConfig(BaseModel):
    """Configuration for a remote API-based embedding deployment.

    Mutually exclusive with ``LLMDeploymentConfig`` for GPU-based embedding.
    Both register under the deployment name ``"embedding"`` so
    ``LLMCluster.embed()`` works transparently with either backend.

    Example::

        config = RemoteEmbeddingConfig(
            model_name="text-embedding-3-small",
            provider="openai",
        )
    """

    # Model and provider
    model_name: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name for the API",
    )
    provider: Literal["openai", "gemini", "openrouter"] = Field(
        description="Embedding API provider",
    )
    api_key_env_var: str | None = Field(
        default=None,
        description=(
            "Environment variable containing the API key. "
            "Defaults per provider: OPENAI_API_KEY, GOOGLE_API_KEY, OPENROUTER_API_KEY."
        ),
    )

    # Model parameters
    dimensions: int | None = Field(
        default=None,
        description=(
            "Output embedding dimensions (None = model default). "
            "Supported by OpenAI text-embedding-3-small/large."
        ),
    )
    max_batch_size: int = Field(
        default=2048,
        ge=1,
        description="Maximum texts per API call (OpenAI: 2048, Gemini: 100)",
    )

    # Concurrency
    max_concurrent_requests: int = Field(
        default=10,
        ge=1,
        description="Maximum concurrent API requests per replica",
    )

    # Scaling
    num_replicas: int = Field(default=1, ge=1)
    min_replicas: int | None = Field(default=None, ge=1)
    max_replicas: int | None = Field(default=None, ge=1)
    target_queue_length: int = Field(default=5, ge=1)

    def model_post_init(self, __context: Any) -> None:
        """Apply provider-specific defaults."""
        defaults = _PROVIDER_DEFAULTS.get(self.provider, {})

        if self.api_key_env_var is None:
            self.api_key_env_var = defaults.get("api_key_env_var", "OPENAI_API_KEY")

        # Gemini has a lower batch limit than OpenAI/OpenRouter.
        # Only override if the user didn't explicitly set a value.
        if "max_batch_size" in defaults and self.max_batch_size == 2048:
            self.max_batch_size = defaults["max_batch_size"]

        if self.min_replicas is None:
            self.min_replicas = self.num_replicas
        if self.max_replicas is None:
            self.max_replicas = self.num_replicas * 2
