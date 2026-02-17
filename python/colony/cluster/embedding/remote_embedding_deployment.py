"""Remote embedding deployments — drop-in replacement for EmbeddingDeployment.

Provides API-based embedding via OpenAI, Google Gemini, or OpenRouter.
These register as deployment name ``"embedding"`` (same as GPU-based
``EmbeddingDeployment``) and expose the same ``embed()`` endpoint so
``LLMCluster`` works unchanged.

Providers:
  - ``OpenAICompatibleEmbeddingDeployment``: OpenAI and OpenRouter (same API)
  - ``GeminiEmbeddingDeployment``: Google Gemini (google-genai SDK)

Requires (depending on provider):
  - ``pip install openai``   for OpenAI / OpenRouter
  - ``pip install google-genai``  for Google Gemini
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import abstractmethod

from ...distributed.ray_utils import serving
from ..circuit_breakers import inference_circuit
from ..models import LLMClientId, LLMClientState
from .remote_embedding_config import RemoteEmbeddingConfig, get_embedding_pricing

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@serving.deployment
class RemoteEmbeddingDeployment:
    """Base class for remote API-based embedding deployments.

    Subclasses must implement:
      - ``_initialize_client()``
      - ``_call_embed_api(texts) -> list[list[float]]``
    """

    def __init__(self, config: RemoteEmbeddingConfig):
        self.config = config
        self._request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        # Monitoring counters
        # TODO: consider moving to a more robust monitoring solution (e.g. Prometheus metrics) since these won't aggregate across replicas of this deployment
        self._total_requests: int = 0
        self._total_texts: int = 0
        self._total_tokens: int = 0
        self._total_errors: int = 0
        self._last_error: str | None = None

        # Set during initialize()
        self.client_id: LLMClientId | None = None

    @serving.initialize_deployment
    async def initialize(self) -> None:
        """Initialize the embedding deployment.

        Called automatically by the serving framework after deployment.
        """
        import ray

        logger.info(
            f"Initializing {self.__class__.__name__} "
            f"(model={self.config.model_name}, provider={self.config.provider})"
        )

        self.client_id = LLMClientId(
            f"embed-{self.config.provider}-{ray.get_runtime_context().get_actor_id()}"
        )

        await self._initialize_client()

        logger.info(f"{self.__class__.__name__} {self.client_id} initialized")

    @serving.endpoint
    @inference_circuit
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for input texts.

        Handles batching transparently when ``len(texts) > max_batch_size``.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        if not texts:
            return []

        start_time = time.time()
        self._total_requests += 1
        self._total_texts += len(texts)

        try:
            batch_size = self.config.max_batch_size
            if len(texts) <= batch_size:
                embeddings = await self._call_with_semaphore(texts)
            else:
                batches = [
                    texts[i : i + batch_size]
                    for i in range(0, len(texts), batch_size)
                ]
                batch_results = await asyncio.gather(
                    *(self._call_with_semaphore(batch) for batch in batches)
                )
                embeddings = [emb for result in batch_results for emb in result]

            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Embedding completed: {len(texts)} texts, "
                f"latency={latency_ms:.1f}ms"
            )
            return embeddings

        except Exception as e:
            self._total_errors += 1
            self._last_error = str(e)
            logger.error(f"Embedding failed: {e}", exc_info=True)
            raise

    @serving.endpoint
    async def get_state(self) -> LLMClientState:
        """Get current state of this embedding deployment.

        Returns:
            Basic client state with monitoring counters.
        """
        return LLMClientState(
            client_id=self.client_id or LLMClientId("embed-uninitialized"),
            deployment_name=serving.get_my_deployment_name(),
            app_name=serving.get_my_app_name(),
            model_name=self.config.model_name,
            kv_cache_capacity=0,
            kv_cache_used=0,
            pending_requests=0,
            total_requests=self._total_requests,
            error_count=self._total_errors,
            last_heartbeat=time.time(),
            last_error=self._last_error,
        )

    async def _call_with_semaphore(self, texts: list[str]) -> list[list[float]]:
        """Call the embedding API with concurrency control."""
        async with self._request_semaphore:
            return await self._call_embed_api(texts)

    @abstractmethod
    async def _initialize_client(self) -> None:
        """Initialize the API client. Called during ``initialize()``."""
        ...

    @abstractmethod
    async def _call_embed_api(self, texts: list[str]) -> list[list[float]]:
        """Call the embedding API for a single batch of texts.

        Args:
            texts: Batch of texts (length <= max_batch_size).

        Returns:
            List of embedding vectors, one per input text.
        """
        ...


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


class OpenAICompatibleEmbeddingDeployment(RemoteEmbeddingDeployment):
    """Embedding via OpenAI or OpenRouter (OpenAI-compatible API).

    For OpenRouter, the same ``openai.AsyncOpenAI`` client is used with
    ``base_url`` pointing to ``https://openrouter.ai/api/v1``.

    Requires: ``pip install openai``
    """

    def __init__(self, config: RemoteEmbeddingConfig):
        super().__init__(config)
        self._client = None  # openai.AsyncOpenAI
        self._pricing = get_embedding_pricing(config.model_name)

    async def _initialize_client(self) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for "
                f"{self.__class__.__name__}. Install with: pip install openai"
            )

        api_key = os.environ.get(self.config.api_key_env_var)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable "
                f"'{self.config.api_key_env_var}'. "
                f"Set it with: export {self.config.api_key_env_var}=your-key"
            )

        kwargs: dict = {"api_key": api_key}
        if self.config.provider == "openrouter":
            kwargs["base_url"] = OPENROUTER_BASE_URL

        self._client = openai.AsyncOpenAI(**kwargs)
        logger.info(
            f"Initialized OpenAI-compatible client for {self.config.model_name} "
            f"(provider={self.config.provider})"
        )

    async def _call_embed_api(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict = {
            "model": self.config.model_name,
            "input": texts,
        }
        if self.config.dimensions is not None:
            kwargs["dimensions"] = self.config.dimensions

        response = await self._client.embeddings.create(**kwargs)

        if response.usage:
            self._total_tokens += response.usage.prompt_tokens

        # response.data is ordered by index; sort defensively
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]


class GeminiEmbeddingDeployment(RemoteEmbeddingDeployment):
    """Embedding via Google Gemini (google-genai SDK).

    Requires: ``pip install google-genai``
    """

    def __init__(self, config: RemoteEmbeddingConfig):
        super().__init__(config)
        self._client = None  # google.genai.Client

    async def _initialize_client(self) -> None:
        try:
            from google import genai  # noqa: F811
        except ImportError:
            raise ImportError(
                "The 'google-genai' package is required for "
                "GeminiEmbeddingDeployment. Install with: pip install google-genai"
            )

        api_key = os.environ.get(self.config.api_key_env_var)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable "
                f"'{self.config.api_key_env_var}'. "
                f"Set it with: export {self.config.api_key_env_var}=your-key"
            )

        self._client = genai.Client(api_key=api_key)
        logger.info(f"Initialized Gemini client for {self.config.model_name}")

    async def _call_embed_api(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.aio.models.embed_content(
            model=self.config.model_name,
            contents=texts,
        )
        return [e.values for e in result.embeddings]
