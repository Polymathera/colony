from __future__ import annotations
import json
from typing import ClassVar
from typing import Any

import asyncio
import time
from collections.abc import Iterator
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
import numpy as np
from sklearn.manifold import TSNE
from circuitbreaker import circuit
from sentence_transformers import SentenceTransformer

from ...distributed.ray_utils import serving
from ...distributed.config import ConfigComponent, register_polymathera_config
from ...distributed.metrics.common import BaseMetricsMonitor
from ...utils import setup_logger
from ..circuit_breakers import inference_circuit


logger = setup_logger(__name__)



class STEmbeddingModel(Enum):
    """Available embedding models with different speed/quality tradeoffs"""

    CODEBERT = "microsoft/codebert-base"  # High quality, slower
    MINILM = "all-MiniLM-L6-v2"  # Medium quality, faster
    SPECTER = "allenai/specter"  # Research papers, good for docs
    LABSE = "google/labse"  # Multilingual support
    CODESEARCH = "github/codebert-base"  # Code search specific


@register_polymathera_config()
class STEmbeddingDeploymentConfig(ConfigComponent):
    """Configuration for the embedding client"""
    # Model selection
    model_name: STEmbeddingModel = STEmbeddingModel.MINILM  # Changed from CODEBERT which has vulnerability issue in `torch.load`
    fallback_model: STEmbeddingModel = STEmbeddingModel.MINILM

    # Performance settings
    enable_gpu: bool = True
    batch_size: int = 32
    max_concurrent_embeddings: int = 8

    # Content processing
    max_content_length: int = 2048
    min_content_length: int = 10
    skip_binary_files: bool = True

    # Chunking configuration
    chunk_large_files: bool = True
    max_chunks_per_file: int = 10

    CONFIG_PATH: ClassVar[str] = "llms.inference.cluster.embedding"


# TODO: Add batch processing methods
# TODO: Add embedding model auto-selection



class TextChunkerBase(ABC):
    @abstractmethod
    def chunk_content(self, content: str) -> Iterator[str]:
        """Split content into semantic chunks"""
        pass


class STEmbeddingClientMetricsMonitor(BaseMetricsMonitor):
    """Metrics for the embedding client"""

    def __init__(self,
                 enable_http_server: bool = True,
                 service_name: str = "EmbeddingClient"):
        super().__init__(enable_http_server, service_name)

        self.logger.info(f"Initializing EmbeddingClientMetricsMonitor instance {id(self)}...")
        self.embedding_duration = self.create_histogram(
            "embedding_client_embedding_duration_seconds",
            "Time to generate embeddings",
            ["model", "status"],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
        )
        self.embedding_batch_size = self.create_histogram(
            "embedding_client_embedding_batch_size",
            "Size of embedding batches",
            buckets=[1, 5, 10, 25, 50, 100],
        )
        self.model_usage = self.create_counter(
            "embedding_client_model_usage_total",
            "Number of times a model was used",
            labelnames=["model"],
        )
        self.model_fallbacks = self.create_counter(
            "embedding_client_model_fallbacks_total",
            "Number of times fallback model was used",
        )
        self.content_length = self.create_histogram(
            "embedding_client_content_length_chars",
            "Length of processed content",
            buckets=[100, 500, 1000, 2000, 5000, 10000],
        )
        self.gpu_memory = self.create_gauge(
            "embedding_client_gpu_memory_bytes",
            "GPU memory usage"
        )


@dataclass
class EmbeddingResult:
    embedding: np.ndarray
    num_chunks: int
    models_used: set[str]


@serving.deployment
class STEmbeddingDeployment:
    """Deployment for embedding text content using SentenceTransformer models.

    Key Recommendations Summary:
    1. Include rich type context
    2. Use hierarchical text structure - Type | Layer | Content | Evidence
    3. Domain-aware preprocessing - Normalize technical terms, expand abbreviations
    4. Multi-level embeddings - Separate embeddings for content, metadata, relationships, etc.
    5. Semantic preprocessing - Clean and structure text before embedding
    6. Quality validation - Monitor embedding quality and relationships
    7. Consistent vocabulary - Normalize technical terms across the codebase

    This approach will significantly improve semantic relationship capture while maintaining the flexibility to handle diverse KnowledgeTypeVar content types.

    Features:
    - Multiple embedding models with fallbacks
    - Batched processing
    - GPU acceleration
    - Caching with TTL
    - Circuit breakers
    - Comprehensive metrics
    """

    def __init__(self, config: STEmbeddingDeploymentConfig | None = None):
        self.config: STEmbeddingDeploymentConfig | None = config

        # Setup concurrency control
        self.gpu_memory_lock = asyncio.Lock()
        self.gpu_memory_threshold = 0.9  # 90% max GPU memory usage
        self.embedding_semaphore: asyncio.Semaphore | None = None

        # Initialize models
        self.device = None
        self._primary_model = None  # Lazy loading
        self._fallback_model = None  # Lazy loading

        # Model usage tracking
        self.local_model_usage_stats = {"primary": 0, "fallback": 0}

        # Setup metrics
        self.metrics: STEmbeddingClientMetricsMonitor | None = None

    @serving.initialize_deployment
    async def initialize(self) -> None:
        from ..gpus import is_cuda_available

        self.config = await STEmbeddingDeploymentConfig.check_or_get_component(self.config)
        self.embedding_semaphore = asyncio.Semaphore(self.config.max_concurrent_embeddings)

        # Initialize models using centralized CUDA detection
        cuda_available = is_cuda_available()
        self.device = (
            "cuda" if cuda_available and self.config.enable_gpu else "cpu"
        )
        self.metrics = STEmbeddingClientMetricsMonitor()

        # Setup circuit breakers
        self._setup_circuit_breakers()

    async def cleanup(self) -> None:
        """Cleanup background tasks and resources"""
        from ...utils import cleanup_dynamic_asyncio_tasks
        from ..gpus import clear_gpu_cache, is_cuda_available

        try:
            # Clean up models
            if self._primary_model:
                del self._primary_model
                self._primary_model = None
            if self._fallback_model:
                del self._fallback_model
                self._fallback_model = None

            # Clear GPU cache using centralized utility
            if is_cuda_available():
                clear_gpu_cache()

            await cleanup_dynamic_asyncio_tasks(self, raise_exceptions=False)
        except Exception as e:
            logger.warning(f"Error cleaning up EmbeddingClient: {e}")

    def _setup_circuit_breakers(self):
        """Setup circuit breakers for critical operations"""
        self._embedding_breaker = circuit(
            failure_threshold=5, recovery_timeout=30, name="embedding_generation"
        )

    @serving.endpoint
    @inference_circuit
    async def embed(
        self,
        texts: list[str],
        chunker: TextChunkerBase | None = None,
    ) -> list[np.ndarray]:
        """Generate embeddings for input texts"""

        results = []
        for text in texts:
            # Track content length
            self.metrics.content_length.observe(len(text))

            if not text:
                embedding_dim = await self._get_embedding_dimension()
                results.append(np.zeros(embedding_dim))
                continue

            # Generate embedding
            chunks = []
            models_used = set()
            if not self.config.chunk_large_files or not chunker:
                final_embedding, model_used = await self._generate_embedding(text)
                chunks = [text]
                models_used.add(model_used)
            else:
                # Chunk content if needed
                chunks = list(chunker.chunk_content(text))
                if not chunks:
                    embedding_dim = await self._get_embedding_dimension()
                    results.append(np.zeros(embedding_dim))
                    return [ np.zeros(embedding_dim) for _ in texts ]

                # Generate embeddings for chunks
                chunk_embeddings = []
                # Limit chunks to prevent memory issues - use reasonable default
                max_chunks = min(len(chunks), self.config.max_chunks_per_file)
                for chunk in chunks[:max_chunks]:
                    embedding, model_used = await self._generate_embedding(chunk)
                    models_used.add(model_used)
                    chunk_embeddings.append(embedding)

                # Combine chunk embeddings
                final_embedding = await self._combine_chunk_embeddings(chunk_embeddings)
                results.append(final_embedding)

        return results

    async def _combine_chunk_embeddings(
        self, embeddings: list[np.ndarray]
    ) -> np.ndarray:
        """Combine multiple chunk embeddings into a single embedding

        Strategies:
        1. Weighted average based on chunk importance
        2. Max pooling across chunks
        3. Attention-based combination
        """
        try:
            if not embeddings:
                embedding_dim = await self._get_embedding_dimension()
                return np.zeros(embedding_dim)

            if len(embeddings) == 1:
                return embeddings[0]

            # Convert to numpy array
            embedding_array = np.array(embeddings)

            # Strategy 1: Weighted average (default)
            weights = np.ones(len(embeddings)) / len(embeddings)
            combined = np.average(embedding_array, axis=0, weights=weights)

            # Normalize the result
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm

            return combined

        except Exception as e:
            logger.error(f"Error combining embeddings: {e}")
            if embeddings:
                return embeddings[0]
            else:
                embedding_dim = await self._get_embedding_dimension()
                return np.zeros(embedding_dim)

    async def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension from the model"""
        model = await self._get_primary_model()
        if model is None:
            model = await self._get_fallback_model()
        return model.get_sentence_embedding_dimension() if model else 384  # Use 384 as default for MiniLM

    async def _generate_embedding(self, content: str, oom_retry: bool = True) -> tuple[np.ndarray, str]:
        """Generate embedding with fallback and metrics and memory management"""
        start_time = time.time()

        async with self.embedding_semaphore:
            try:
                # Manage GPU memory before processing
                await self._manage_gpu_memory()

                # Try primary model
                primary_model = await self._get_primary_model()
                if primary_model:
                    try:
                        embedding = await self._embedding_breaker(self._embed_with_model)(
                            primary_model, content
                        )

                        self.metrics.embedding_duration.labels(
                            model=self.config.model_name.value, status="success"
                        ).observe(time.time() - start_time)

                        self.metrics.model_usage.labels(
                            model=self.config.model_name.value
                        ).inc()

                        self.local_model_usage_stats["primary"] += 1
                        return embedding, self.config.model_name.value

                    except RuntimeError as e:
                        if "out of memory" in str(e) and oom_retry:
                            await self._reduce_gpu_memory()
                            # Retry once after memory reduction. Avoid infinite recursion
                            # by setting oom_retry to False.
                            return await self._generate_embedding(content, oom_retry=False)
                        logger.warning(f"Primary model failed: {e}")
                        raise # Jump to fallback model

            except Exception as e:
                logger.warning(f"Primary model failed: {e}")  # Fall through to fallback

            # Try fallback model
            model_used = "fallback"
            self.metrics.model_fallbacks.inc()

            try:
                fallback_model = await self._get_fallback_model()
                if fallback_model:
                    embedding = await self._embedding_breaker(self._embed_with_model)(
                        fallback_model, content
                    )

                    self.metrics.embedding_duration.labels(
                        model=self.config.fallback_model.value, status="success"
                    ).observe(time.time() - start_time)

                    self.metrics.model_usage.labels(
                        model=self.config.fallback_model.value
                    ).inc()

                    self.local_model_usage_stats["fallback"] += 1
                    return embedding, self.config.fallback_model.value

            except Exception as e:
                logger.error(f"Fallback model failed: {e}")
                self.metrics.embedding_duration.labels(
                    model=self.config.fallback_model.value, status="error"
                ).observe(time.time() - start_time)

            # Return zero vector as last resort
            # Use default embedding dimension for fallback
            embedding_dim = await self._get_embedding_dimension()
            return np.zeros(embedding_dim), "none"

    async def _embed_with_model(
        self, model: SentenceTransformer, content: str
    ) -> np.ndarray:
        """Generate embedding using specified model"""
        embedding = await asyncio.to_thread(
            model.encode, content, convert_to_tensor=True, show_progress_bar=False
        )
        return embedding.cpu().numpy()

    async def _manage_gpu_memory(self):
        """Manage GPU memory usage and model loading using centralized GPU utilities."""
        from ..gpus import get_gpu_memory_info, manage_gpu_memory, is_cuda_available

        async with self.gpu_memory_lock:
            try:
                if not is_cuda_available() or not self.config.enable_gpu:
                    return

                # Get current memory usage using centralized utilities
                memory_info = get_gpu_memory_info(device_id=0)
                memory_allocated = memory_info['allocated']
                memory_total = memory_info['total']
                memory_ratio = memory_allocated / memory_total if memory_total > 0 else 0

                # Update metrics (keeping all three as before)
                self.metrics.gpu_memory.labels("ratio").set(memory_ratio)
                self.metrics.gpu_memory.labels("allocated").set(memory_allocated)
                self.metrics.gpu_memory.labels("total").set(memory_total)

                # Memory management actions using centralized utility
                within_threshold = manage_gpu_memory(
                    threshold=self.gpu_memory_threshold,
                    device_id=0,
                    clear_cache_on_high=False  # We'll handle this manually below
                )

                if not within_threshold:
                    logger.warning(f"High GPU memory usage: {memory_ratio:.2%}")
                    await self._reduce_gpu_memory()

            except Exception as e:
                logger.error(f"GPU memory management error: {e}")
                self.device = "cpu"  # Fallback to CPU

    async def _reduce_gpu_memory(self):
        """Reduce GPU memory usage when threshold is exceeded."""
        from ..gpus import clear_gpu_cache

        try:
            # Clear cache using centralized utility
            clear_gpu_cache(device_id=0)

            # Move less frequently used model to CPU if both are loaded
            if self._primary_model and self._fallback_model:
                # Check usage metrics to decide which model to offload
                global_primary_usage = (
                    self.metrics.model_usage
                    .labels(self.config.model_name.value)
                    .get()
                )
                global_fallback_usage = (
                    self.metrics.model_usage
                    .labels(self.config.fallback_model.value)
                    .get()
                )
                # Check usage stats to decide which model to offload
                primary_usage = self.local_model_usage_stats["primary"]
                fallback_usage = self.local_model_usage_stats["fallback"]

                if primary_usage < fallback_usage:
                    self._primary_model = self._primary_model.cpu()
                    logger.info("Moved primary model to CPU")
                else:
                    self._fallback_model = self._fallback_model.cpu()
                    logger.info("Moved fallback model to CPU")

        except Exception as e:
            logger.error(f"Error reducing GPU memory: {e}")

    async def _load_model_to_device(
        self, model_name: str
    ) -> SentenceTransformer | None:
        """Load model to appropriate device with memory management"""
        try:
            # Check GPU memory before loading
            await self._manage_gpu_memory()

            # Load model
            model = await asyncio.to_thread(lambda: SentenceTransformer(model_name))

            # Move to appropriate device
            if self.device == "cuda":
                try:
                    model = model.to(self.device)
                except RuntimeError as e:
                    logger.warning(f"GPU memory error loading model: {e}")
                    self.device = "cpu"
                    model = model.to("cpu")

            return model

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None

    async def _get_primary_model(self) -> SentenceTransformer | None:
        """Lazy load primary model"""
        if self._primary_model is None:
            self._primary_model = await self._load_model_to_device(self.config.model_name.value)
        return self._primary_model

    async def _get_fallback_model(self) -> SentenceTransformer | None:
        """Lazy load fallback model"""
        if self._fallback_model is None:
            self._fallback_model = await self._load_model_to_device(
                self.config.fallback_model.value
            )
        return self._fallback_model

    def __del__(self):
        """Cleanup GPU memory on deletion using centralized utilities."""
        from ..gpus import clear_gpu_cache, is_cuda_available

        try:
            if is_cuda_available():
                clear_gpu_cache()
        except Exception as e:
            logger.error(f"Error cleaning up GPU memory: {e}")




