"""Embedding deployment implementation for the LLM cluster.

This module provides the EmbeddingDeployment class which wraps a vLLM instance
configured for embedding generation.

● Key Differences Between the Two Embedding Implementations

  polymathera/llms/inference/cluster/embedding.py (EmbeddingClient)

  Underlying Technology:
  - SentenceTransformer (Hugging Face transformers library)
  - Smaller, lighter-weight models (MiniLM, CodeBERT, SPECTER, etc.)
  - PyTorch-based with direct model inference

  Architecture:
  - Standalone client class (not Ray-based)
  - Runs locally in the same process
  - Manual GPU memory management
  - Lazy model loading

  Capabilities:
  - ✅ Multiple predefined models with automatic fallback
  - ✅ Text preprocessing and chunking support
  - ✅ Chunk embedding aggregation (weighted average)
  - ✅ Circuit breakers for fault tolerance
  - ✅ Rich metrics (Prometheus)
  - ✅ GPU memory management with OOM recovery
  - ✅ Configurable batch sizes
  - ❌ No distributed serving
  - ❌ No auto-scaling
  - ❌ No S3 model loading
  - ❌ Limited to smaller embedding models

  Use Cases:
  - Code embedding for semantic search
  - Document/paper embedding (SPECTER)
  - Smaller-scale embedding tasks
  - Local development and testing
  - When you need specific models like CodeBERT for code
  - Quick prototyping without distributed infrastructure

  ---
  polymathera/colony/cluster/embedding_deployment.py (EmbeddingDeployment)

  Underlying Technology:
  - vLLM AsyncLLMEngine with task="embed" mode
  - Large embedding models (e.g., e5-mistral-7b-instruct - 7B parameters)
  - Optimized for high-throughput, production serving

  Architecture:
  - Ray Serve deployment (distributed)
  - Runs as a Ray actor with auto-scaling
  - Professional serving infrastructure
  - Integrated with colony.distributed.ray_utils.serving framework

  Capabilities:
  - ✅ Massive models (multi-billion parameter embeddings)
  - ✅ Distributed serving with load balancing
  - ✅ Auto-scaling (1-5 replicas based on queue length)
  - ✅ S3 model loading with exponential backoff retry
  - ✅ Tensor parallelism for multi-GPU models
  - ✅ Circuit breakers for fault tolerance
  - ✅ Quantization support (AWQ, GPTQ, FP8)
  - ✅ GPU metrics monitoring
  - ✅ State tracking via LLMClientState
  - ✅ Production-ready with Ray Serve integration
  - ❌ No text chunking/preprocessing
  - ❌ No fallback models
  - ❌ Limited to vLLM-supported embedding models

  Use Cases:
  - Production embedding service at scale
  - Large embedding models (7B+ parameters)
  - High-throughput requirements
  - Distributed workloads across multiple GPUs/nodes
  - When you need auto-scaling based on load
  - Integration with existing LLM cluster infrastructure
  - S3-based model distribution

  ---
  Summary Table

  | Feature         | SentenceTransformerEmbeddingDeployment (SentenceTransformer)   | EmbeddingDeployment (vLLM)           |
  |-----------------|-----------------------------------------|--------------------------------------|
  | Model Size      | Small-medium (100M-400M params)         | Large (1B-10B+ params)               |
  | Technology      | SentenceTransformer/PyTorch             | vLLM AsyncLLMEngine                  |
  | Deployment      | Local/In-process                        | Distributed Ray Serve                |
  | Scaling         | Manual                                  | Auto-scaling (1-5 replicas)          |
  | GPU Support     | Single GPU                              | Multi-GPU tensor parallelism         |
  | Throughput      | Low-Medium                              | High (optimized batching)            |
  | S3 Loading      | ❌                                      | ✅                                  |
  | Chunking        | ✅                                      | ❌                                  |
  | Fallback Models | ✅                                      | ❌                                  |
  | Quantization    | ❌                                      | ✅ (AWQ, GPTQ, FP8)                 |
  | Best For        | Code search, local dev, specific models | Production, large models, high scale |

"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from vllm import AsyncEngineArgs, AsyncLLMEngine, PoolingParams, PoolingOutput

from ..gpus import GPUMetricsCollector
from ...distributed.ray_utils import serving
from ..circuit_breakers import inference_circuit
from ..model_loader import S3ModelLoader
from ..models import LLMClientId, LLMClientState
from ..registry import ModelRegistry

logger = logging.getLogger(__name__)


@serving.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 5,
        "target_queue_length": 10,
    },
    ray_actor_options={
        "num_gpus": 1,
    },
)
class EmbeddingDeployment:
    """vLLM-based embedding model deployment.

    This class wraps a vLLM AsyncLLMEngine configured for embedding generation.
    It provides efficient batch embedding generation with GPU acceleration.

    Example:
        ```python
        from polymathera.colony.distributed.ray_utils import serving
        from polymathera.colony import EmbeddingDeployment

        # Deploy embedding instance
        app = serving.Application(name="embedding-service")
        app.add_deployment(
            EmbeddingDeployment.bind(
                model_name="intfloat/e5-mistral-7b-instruct",
            ),
            ray_actor_options={"num_gpus": 1},
        )
        await app.start()
        ```
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        gpu_id: int = 0,
        quantization: str | None = None,
        s3_bucket: str | None = None,
        s3_retry_attempts: int = 10,
    ):
        """Initialize EmbeddingDeployment.

        Args:
            model_name: HuggingFace embedding model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory for vLLM (default: 0.9)
            trust_remote_code: Whether to trust remote code when loading model
            gpu_id: GPU device ID for this deployment (default: 0)
            quantization: Quantization method (awq, gptq, fp8, etc.) or None
            s3_bucket: S3 bucket name for model loading (None to load from HuggingFace)
            s3_retry_attempts: Number of retry attempts for S3 downloads (default: 10)
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.gpu_id = gpu_id
        self.quantization = quantization
        self.s3_bucket = s3_bucket
        self.s3_retry_attempts = s3_retry_attempts

        # Get model parameters from registry
        model_params = ModelRegistry.get_model(model_name)
        if model_params:
            self.model_params = model_params
            logger.info(
                f"Loaded embedding model config from registry: {model_name}"
            )
        else:
            self.model_params = None
            logger.warning(f"Embedding model {model_name} not in registry")

        # Will be set during initialization
        self.engine: AsyncLLMEngine | None = None
        self.client_id: LLMClientId | None = None
        self.state: LLMClientState | None = None
        self.gpu_metrics: GPUMetricsCollector | None = None

        # Monitoring
        self._monitoring_task: asyncio.Task | None = None
        self._monitoring_interval = 30.0  # seconds

    @serving.initialize_deployment
    async def initialize(self):
        """Initialize the vLLM embedding engine and set up state.

        This is called automatically by the serving framework after deployment.
        """
        logger.info(f"Initializing EmbeddingDeployment with model {self.model_name}")

        # Generate client ID
        import ray
        self.client_id = LLMClientId(f"embed-{ray.get_runtime_context().get_actor_id()}")

        # Download model from S3 if specified
        model_path = self.model_name
        if self.s3_bucket:
            loader = S3ModelLoader(
                bucket=self.s3_bucket,
                model_name=self.model_name,
                retry_attempts=self.s3_retry_attempts
            )
            downloaded_path = await loader.download_and_extract()
            if downloaded_path:
                model_path = downloaded_path
                logger.info(f"Using local embedding model from S3: {downloaded_path}")
            else:
                logger.warning(
                    f"Failed to download model from S3, falling back to HuggingFace: {self.model_name}"
                )

        # Initialize vLLM engine in embedding mode
        engine_args = AsyncEngineArgs(
            model=model_path,
            task="embed",  # Enable embedding mode
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=self.trust_remote_code,
            gpu_memory_utilization=self.gpu_memory_utilization,
            quantization=self.quantization,
            enforce_eager=True,  # Recommended for embeddings
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Initialize GPU metrics collector
        try:
            self.gpu_metrics = GPUMetricsCollector(gpu_id=self.gpu_id)
            logger.info(f"GPU metrics collector initialized for GPU {self.gpu_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU metrics collector: {e}")
            self.gpu_metrics = None

        # Initialize state
        app_name = serving.get_my_app_name()
        self.state = LLMClientState(
            client_id=self.client_id,
            deployment_name=self.__class__.__name__,
            app_name=app_name,
            model_name=self.model_name,
            kv_cache_capacity=0,  # Embeddings don't use KV cache
            kv_cache_used=0,
            loaded_pages={},
            last_heartbeat=time.time(),
        )

        # Start periodic GPU monitoring
        self._monitoring_task = asyncio.create_task(self._periodic_gpu_monitoring())

        logger.info(f"EmbeddingDeployment {self.client_id} initialized successfully")

    @serving.endpoint
    @inference_circuit
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for input texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (one per input text)

        Example:
            ```python
            texts = ["Hello world", "Another text"]
            embeddings = await deployment.embed(texts)
            # embeddings: [[0.1, 0.2, ...], [0.3, 0.4, ...]]
            ```
        """
        start_time = time.time()

        try:
            # Update state
            self.state.pending_requests += len(texts)
            self.state.total_requests += len(texts)

            # Generate embeddings using vLLM's encode method
            # engine.encode() returns an AsyncGenerator, we need to iterate through it
            embeddings = []

            for text in texts:
                # Each text gets its own request with a unique ID
                request_id = str(uuid.uuid4())

                # Create async generator for this request
                # PoolingParams must specify task='embed' for vLLM v1
                results_generator = self.engine.encode(
                    prompt=text,
                    pooling_params=PoolingParams(task='embed'),
                    request_id=request_id
                )

                # Iterate through the async generator to get the final output
                final_output = None
                async for request_output in results_generator:
                    final_output = request_output

                # Extract embedding vector from final output
                # final_output is an EmbeddingRequestOutput
                # final_output.outputs is a PoolingOutput with .data attribute (torch.Tensor)
                if final_output and final_output.outputs:
                    embedding_data = final_output.outputs.data
                    # Convert to list if it's a tensor
                    if hasattr(embedding_data, 'tolist'):
                        embedding = embedding_data.tolist()
                    else:
                        embedding = list(embedding_data) if not isinstance(embedding_data, list) else embedding_data
                    embeddings.append(embedding)
                else:
                    logger.error(f"No output received for text: {text[:50]}...")
                    embeddings.append([])

            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Embedding completed: {len(texts)} texts, "
                f"latency={latency_ms:.2f}ms, "
                f"avg_latency_per_text={latency_ms / len(texts):.2f}ms"
            )

            return embeddings

        except Exception as e:
            self.state.error_count += len(texts)
            self.state.last_error = str(e)
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            raise

        finally:
            self.state.pending_requests -= len(texts)

    @serving.endpoint
    async def get_state(self) -> LLMClientState:
        """Get current state of this embedding client.

        Returns:
            Current client state including metrics
        """
        self.state.last_heartbeat = time.time()
        return self.state

    async def _periodic_gpu_monitoring(self):
        """Periodically collect and log GPU metrics."""
        while True:
            try:
                await asyncio.sleep(self._monitoring_interval)

                if self.gpu_metrics:
                    metrics = self.gpu_metrics.collect()
                    if metrics:
                        logger.info(
                            f"GPU {metrics.gpu_id} ({metrics.gpu_name}): "
                            f"Memory: {metrics.memory_used_mb:.1f}/{metrics.memory_total_mb:.1f}MB "
                            f"({metrics.memory_utilization_pct:.1f}%), "
                            f"GPU Util: {metrics.gpu_utilization_pct:.1f}%"
                            + (f", Temp: {metrics.temperature_celsius}°C" if metrics.temperature_celsius else "")
                            + (f", Power: {metrics.power_draw_watts:.1f}W" if metrics.power_draw_watts else "")
                        )
            except asyncio.CancelledError:
                logger.info("GPU monitoring task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in GPU monitoring: {e}", exc_info=True)

    async def cleanup(self):
        """Cleanup resources on shutdown."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        if self.gpu_metrics:
            self.gpu_metrics.cleanup()