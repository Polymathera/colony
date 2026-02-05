"""Model registry for LLM cluster.

This module provides a registry of supported LLM models with their capabilities,
resource requirements, and configuration parameters. The registry enables:

1. Model discovery based on requirements (size, capabilities, backend)
2. Resource planning (GPU count, VRAM requirements)
3. Easy model switching via configuration
4. Validation of model-tokenizer compatibility

The registry is designed to be extensible for new models without code changes.
"""

from enum import Enum, auto
from pydantic import BaseModel, Field


class LLMCapability(Enum):
    """Capabilities that an LLM model can support."""
    CODE_ANALYSIS = auto()
    CODE_GENERATION = auto()
    REASONING = auto()
    STRUCTURED_OUTPUT = auto()
    MATH = auto()
    CHAT = auto()
    EMBEDDINGS = auto()


class LLMSize(Enum):
    """Size/scale categories for LLM models."""
    SMALL = auto()   # e.g., 1B-8B parameters
    MEDIUM = auto()  # e.g., 13B-30B parameters
    LARGE = auto()   # e.g., 70B-100B parameters
    HUGE = auto()    # e.g., 180B+ parameters
    UNKNOWN = auto() # Closed-source models


class LLMBackend(Enum):
    """Backend type for serving the LLM."""
    VLLM = auto()        # vLLM for high-throughput inference
    HUGGINGFACE = auto() # HuggingFace transformers
    THIRD_PARTY = auto() # External APIs (OpenAI, Anthropic, etc.)


class QuantizationMethod(Enum):
    """Quantization methods supported for model inference."""
    NONE = auto()      # No quantization (fp16/bf16)
    AWQ = auto()       # Activation-aware Weight Quantization (int4)
    GPTQ = auto()      # GPTQ quantization (int4/int8)
    FP8 = auto()       # FP8 quantization
    INT8 = auto()      # INT8 quantization
    INT4 = auto()      # INT4 quantization
    BNB_4BIT = auto()  # BitsAndBytes 4-bit (HuggingFace only)
    BNB_8BIT = auto()  # BitsAndBytes 8-bit (HuggingFace only)


class LLMModelParameters(BaseModel):
    """Parameters defining an LLM model's capabilities and requirements."""

    # Model identification
    model_name: str = Field(description="Full model name (e.g., 'meta-llama/Llama-3.1-8B')")

    # Size and resource requirements
    size: LLMSize = Field(description="Size category of the model")
    size_bp: float = Field(default=0, description="Size in billions of parameters")
    num_gpus: float = Field(default=0, description="Number of A100-80GB GPUs required (fractional allowed)")
    num_vram_gb: int = Field(default=0, description="VRAM required at 4-bit quantization (GB)")

    # Model architecture (for KV cache calculations)
    n_layers: int = Field(default=32, description="Number of transformer layers")
    d_model: int = Field(default=4096, description="Hidden dimension size")

    # Capabilities
    capabilities: set[LLMCapability] = Field(description="Capabilities supported by the model")
    backend: LLMBackend = Field(description="Preferred backend for this model")

    # Context and output limits
    context_window: int = Field(description="Maximum context window size (tokens)")
    max_output_tokens: int = Field(description="Maximum output tokens supported")

    # Quantization configuration
    default_quantization: QuantizationMethod = Field(
        default=QuantizationMethod.NONE,
        description="Default quantization method for this model"
    )
    supported_quantizations: set[QuantizationMethod] = Field(
        default_factory=lambda: {QuantizationMethod.NONE},
        description="Quantization methods supported by this model"
    )

    # Additional features
    supports_structured_output: bool = Field(default=False, description="Whether model supports structured output (JSON, etc.)")
    fine_tuned_for: str | None = Field(default=None, description="Specific task the model is fine-tuned for")

    # Performance characteristics
    typical_latency_ms: int = Field(default=1000, description="Typical inference latency (ms)")
    max_batch_size: int = Field(default=32, description="Maximum recommended batch size")

    def get_bytes_per_token(self, quantization: QuantizationMethod | None = None) -> int:
        """Get bytes per token for KV cache based on model architecture and quantization.

        The KV cache stores key and value vectors for each token across all layers:
        - For each token position, we store K and V vectors for every layer
        - Each K or V vector has dimension d_model
        - Size per token = 2 (K+V) × n_layers × d_model × bytes_per_element

        Example for Llama-3.1-8B with fp16:
        - 32 layers × 4096 dim × 2 bytes × 2 (K+V) = 524,288 bytes = 512 KB per token

        Args:
            quantization: Quantization method for KV cache elements (uses default if None)

        Returns:
            Bytes per token for KV cache storage
        """
        quant = quantization or self.default_quantization

        # Bytes per element based on quantization of KV cache
        # Note: Weight quantization (AWQ/GPTQ) typically doesn't quantize KV cache
        # INT4/AWQ/GPTQ/BNB_4BIT quantize the model weights, NOT the KV cache.
        # The KV cache stores activations (key and value vectors computed during forward pass), not weights. Most quantization methods:
        # - Quantize weights to 4-bit for storage/computation efficiency
        # - Keep KV cache in fp16/bf16 (2 bytes) because:
        #   - Activation quantization is much harder and degrades quality significantly
        #   - KV cache needs to be recomputed or loaded on every forward pass
        #   - vLLM and most inference engines keep KV cache in fp16 even with quantized weights
        # Exception: FP8 and INT8 quantization methods (like vLLM's fp8 KV cache support) DO quantize the KV cache to 1 byte per element.

        bytes_per_element_map = {
            QuantizationMethod.NONE: 2,      # fp16/bf16: 2 bytes per element
            QuantizationMethod.FP8: 1,       # fp8: 1 byte per element
            QuantizationMethod.INT8: 1,      # int8: 1 byte per element
            QuantizationMethod.BNB_8BIT: 1,  # int8: 1 byte per element
            QuantizationMethod.INT4: 2,      # int4 weights, but KV cache usually stays fp16
            QuantizationMethod.AWQ: 2,       # int4 weights, KV cache stays fp16
            QuantizationMethod.GPTQ: 2,      # int4 weights, KV cache stays fp16
            QuantizationMethod.BNB_4BIT: 2,  # int4 weights, KV cache stays fp16
        }

        bytes_per_element = bytes_per_element_map.get(quant, 2)

        # Calculate total KV cache size per token
        # 2 (key + value) × n_layers × d_model × bytes_per_element
        kv_cache_bytes_per_token = 2 * self.n_layers * self.d_model * bytes_per_element

        return kv_cache_bytes_per_token


class ModelRegistry:
    """Registry of supported LLM models.

    This registry maintains a catalog of models with their specifications,
    enabling intelligent model selection based on requirements.
    """

    # Catalog of supported models
    _models: list[LLMModelParameters] = [
        # Llama 3.1 series - Open source, versatile
        LLMModelParameters(
            model_name="meta-llama/Llama-3.1-8B",
            size=LLMSize.SMALL,
            size_bp=8.03,
            num_gpus=1,
            num_vram_gb=8,
            n_layers=32,
            d_model=4096,
            capabilities={
                LLMCapability.CODE_ANALYSIS,
                LLMCapability.CODE_GENERATION,
                LLMCapability.REASONING,
                LLMCapability.MATH,
                LLMCapability.CHAT,
            },
            backend=LLMBackend.VLLM,
            context_window=128 * 1024,
            max_output_tokens=4 * 1024,
            supports_structured_output=True,
            typical_latency_ms=500,
            max_batch_size=64,
        ),
        LLMModelParameters(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            size=LLMSize.SMALL,
            size_bp=8.03,
            num_gpus=1,
            num_vram_gb=8,
            n_layers=32,
            d_model=4096,
            capabilities={
                LLMCapability.CODE_ANALYSIS,
                LLMCapability.CODE_GENERATION,
                LLMCapability.REASONING,
                LLMCapability.MATH,
                LLMCapability.CHAT,
                LLMCapability.STRUCTURED_OUTPUT,
            },
            backend=LLMBackend.VLLM,
            context_window=128 * 1024,
            max_output_tokens=4 * 1024,
            supports_structured_output=True,
            typical_latency_ms=500,
            max_batch_size=64,
        ),
        LLMModelParameters(
            model_name="meta-llama/Llama-3.1-70B",
            size=LLMSize.LARGE,
            size_bp=70.6,
            num_gpus=2,
            num_vram_gb=70,
            n_layers=80,
            d_model=8192,
            capabilities={
                LLMCapability.CODE_ANALYSIS,
                LLMCapability.CODE_GENERATION,
                LLMCapability.REASONING,
                LLMCapability.MATH,
                LLMCapability.CHAT,
            },
            backend=LLMBackend.VLLM,
            context_window=128 * 1024,
            max_output_tokens=4 * 1024,
            supports_structured_output=True,
            typical_latency_ms=2000,
            max_batch_size=32,
        ),
        LLMModelParameters(
            model_name="meta-llama/Llama-3.1-70B-Instruct",
            size=LLMSize.LARGE,
            size_bp=70.6,
            num_gpus=2,
            num_vram_gb=70,
            n_layers=80,
            d_model=8192,
            capabilities={
                LLMCapability.CODE_ANALYSIS,
                LLMCapability.CODE_GENERATION,
                LLMCapability.REASONING,
                LLMCapability.MATH,
                LLMCapability.CHAT,
                LLMCapability.STRUCTURED_OUTPUT,
            },
            backend=LLMBackend.VLLM,
            context_window=128 * 1024,
            max_output_tokens=4 * 1024,
            supports_structured_output=True,
            typical_latency_ms=2000,
            max_batch_size=32,
        ),
        LLMModelParameters(
            model_name="meta-llama/Llama-3.1-405B",
            size=LLMSize.HUGE,
            size_bp=405,
            num_gpus=8,
            num_vram_gb=405,
            n_layers=126,
            d_model=16384,
            capabilities={
                LLMCapability.CODE_ANALYSIS,
                LLMCapability.CODE_GENERATION,
                LLMCapability.REASONING,
                LLMCapability.MATH,
                LLMCapability.CHAT,
                LLMCapability.STRUCTURED_OUTPUT,
            },
            backend=LLMBackend.VLLM,
            context_window=128 * 1024,
            max_output_tokens=4 * 1024,
            supports_structured_output=True,
            typical_latency_ms=5000,
            max_batch_size=16,
        ),

        # Llama 3.2 series - Smaller, efficient models
        LLMModelParameters(
            model_name="meta-llama/Llama-3.2-1B",
            size=LLMSize.SMALL,
            size_bp=1.24,
            num_gpus=1,
            num_vram_gb=2,
            n_layers=16,
            d_model=2048,
            capabilities={
                LLMCapability.CODE_ANALYSIS,
                LLMCapability.CHAT,
            },
            backend=LLMBackend.VLLM,
            context_window=128 * 1024,
            max_output_tokens=2 * 1024,
            supports_structured_output=False,
            typical_latency_ms=200,
            max_batch_size=128,
        ),
        LLMModelParameters(
            model_name="meta-llama/Llama-3.2-3B",
            size=LLMSize.SMALL,
            size_bp=3.21,
            num_gpus=1,
            num_vram_gb=4,
            n_layers=28,
            d_model=3072,
            capabilities={
                LLMCapability.CODE_ANALYSIS,
                LLMCapability.CODE_GENERATION,
                LLMCapability.CHAT,
            },
            backend=LLMBackend.VLLM,
            context_window=128 * 1024,
            max_output_tokens=4 * 1024,
            supports_structured_output=True,
            typical_latency_ms=300,
            max_batch_size=96,
        ),

        # Test models - Small models for testing and development
        LLMModelParameters(
            model_name="facebook/opt-125m",
            size=LLMSize.SMALL,
            size_bp=0.125,
            num_gpus=1,
            num_vram_gb=1,
            n_layers=12,
            d_model=768,
            capabilities={
                LLMCapability.CODE_GENERATION,
                LLMCapability.CHAT,
            },
            backend=LLMBackend.VLLM,
            context_window=2048,
            max_output_tokens=1024,
            supports_structured_output=False,
            typical_latency_ms=100,
            max_batch_size=256,
        ),
        LLMModelParameters(
            model_name="intfloat/e5-small-v2",
            size=LLMSize.SMALL,
            size_bp=0.033,
            num_gpus=1,
            num_vram_gb=1,
            n_layers=12,
            d_model=384,
            capabilities={
                LLMCapability.EMBEDDINGS,
            },
            backend=LLMBackend.VLLM,
            context_window=512,
            max_output_tokens=0,  # Embedding model doesn't generate text
            supports_structured_output=False,
            typical_latency_ms=50,
            max_batch_size=512,
        ),

        # Third-party API models
        LLMModelParameters(
            model_name="anthropic/claude-3.7-sonnet",
            size=LLMSize.UNKNOWN,
            capabilities={
                LLMCapability.CODE_ANALYSIS,
                LLMCapability.CODE_GENERATION,
                LLMCapability.REASONING,
                LLMCapability.MATH,
                LLMCapability.CHAT,
                LLMCapability.STRUCTURED_OUTPUT,
            },
            backend=LLMBackend.THIRD_PARTY,
            context_window=200 * 1024,
            max_output_tokens=128 * 1024,
            supports_structured_output=True,
            typical_latency_ms=3000,
            max_batch_size=1,
        ),
        LLMModelParameters(
            model_name="openai/gpt-4.5",
            size=LLMSize.UNKNOWN,
            capabilities={
                LLMCapability.CODE_ANALYSIS,
                LLMCapability.CODE_GENERATION,
                LLMCapability.REASONING,
                LLMCapability.MATH,
                LLMCapability.CHAT,
                LLMCapability.STRUCTURED_OUTPUT,
            },
            backend=LLMBackend.THIRD_PARTY,
            context_window=256 * 1024,
            max_output_tokens=32 * 1024,
            supports_structured_output=True,
            typical_latency_ms=2000,
            max_batch_size=1,
        ),
    ]

    @classmethod
    def get_model(cls, model_name: str) -> LLMModelParameters | None:
        """Get model parameters by name.

        Args:
            model_name: Full model name

        Returns:
            Model parameters if found, None otherwise
        """
        for model in cls._models:
            if model.model_name == model_name:
                return model
        return None

    @classmethod
    def find_models(
        cls,
        size: LLMSize | None = None,
        capabilities: set[LLMCapability] | None = None,
        backend: LLMBackend | None = None,
        max_gpus: float | None = None,
    ) -> list[LLMModelParameters]:
        """Find models matching the given criteria.

        Args:
            size: Required size category
            capabilities: Required capabilities (model must have ALL)
            backend: Required backend type
            max_gpus: Maximum number of GPUs available

        Returns:
            List of matching models, sorted by size (smallest first)
        """
        matches = []

        for model in cls._models:
            # Filter by size
            if size is not None and model.size != size:
                continue

            # Filter by capabilities (model must have ALL required capabilities)
            if capabilities is not None:
                if not capabilities.issubset(model.capabilities):
                    continue

            # Filter by backend
            if backend is not None and model.backend != backend:
                continue

            # Filter by GPU availability
            if max_gpus is not None and model.num_gpus > max_gpus:
                continue

            matches.append(model)

        # Sort by size (smallest first for efficiency)
        matches.sort(key=lambda m: m.size_bp if m.size_bp > 0 else float('inf'))

        return matches

    @classmethod
    def register_model(cls, model: LLMModelParameters) -> None:
        """Register a new model in the registry.

        This allows users to add custom models without modifying this file.

        Args:
            model: Model parameters to register
        """
        # Check if model already exists
        existing = cls.get_model(model.model_name)
        if existing:
            # Replace existing model
            cls._models = [m for m in cls._models if m.model_name != model.model_name]

        cls._models.append(model)

    @classmethod
    def list_all_models(cls) -> list[LLMModelParameters]:
        """Get list of all registered models.

        Returns:
            List of all model parameters
        """
        return cls._models.copy()