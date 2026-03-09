"""Tokenization utilities for LLM cluster.

This module provides tokenization functionality with support for multiple backends.
The implementation uses tiktoken for generic tokenization and HuggingFace tokenizers
for model-specific tokenization.

Key features:
- Consistent API across different tokenizer backends
- Model-tokenizer compatibility validation
- Efficient token counting without full encoding
- Support for special tokens handling
- Async tokenizer loading to avoid blocking
"""

import asyncio
import logging
from typing import Protocol

import tiktoken
from transformers import AutoTokenizer

from .registry import LLMModelParameters, LLMBackend, ModelRegistry

logger = logging.getLogger(__name__)


class TokenizerProtocol(Protocol):
    """Protocol defining the tokenizer interface."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text without full encoding."""
        ...


class TiktokenTokenizer:
    """Tokenizer using tiktoken (OpenAI's tokenizer).

    This is used for:
    - Generic tokenization when model-specific tokenizer not available
    - Third-party API models that use tiktoken-compatible tokenization
    - Fast token counting

    Uses cl100k_base encoding (GPT-4/GPT-3.5-turbo compatible).
    """

    def __init__(self):
        """Initialize tiktoken tokenizer."""
        self._encoding = tiktoken.get_encoding("cl100k_base")
        logger.info("Initialized tiktoken tokenizer with cl100k_base encoding")

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode

        Returns:
            List of token IDs

        Note:
            Uses disallowed_special=() to treat special tokens as normal text
        """
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode

        Returns:
            Decoded text
        """
        return self._encoding.decode(token_ids)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.encode(text))

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._encoding.n_vocab


class HuggingFaceTokenizer:
    """Tokenizer using HuggingFace transformers.

    This is used for:
    - Model-specific tokenization
    - Models served via vLLM or HuggingFace backends
    - When exact tokenizer-model compatibility is required
    """

    def __init__(self, model_name_or_path: str, tokenizer_obj):
        """Initialize HuggingFace tokenizer.

        Args:
            model_name_or_path: Name or path of the model
            tokenizer_obj: Pre-loaded tokenizer object

        Note:
            Use from_pretrained() class method to create instances.
        """
        self.model_name = model_name_or_path
        self._tokenizer = tokenizer_obj

        logger.info(f"Initialized HuggingFace tokenizer for {model_name_or_path}")

    @classmethod
    async def from_pretrained(cls, model_name_or_path: str) -> "HuggingFaceTokenizer":
        """Load tokenizer from pretrained model (async).

        Args:
            model_name_or_path: Model name (e.g., 'meta-llama/Llama-3.1-8B') or local path

        Returns:
            HuggingFaceTokenizer instance

        Note:
            This method uses asyncio.to_thread() to avoid blocking the event loop
            during potentially slow tokenizer loading from HuggingFace Hub or disk.
        """
        logger.info(f"Loading HuggingFace tokenizer from {model_name_or_path}")

        # Load tokenizer in thread pool to avoid blocking
        tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained, # Took 8 seconds to load facebook/opt-125m tokenizer
            model_name_or_path
        )

        # Add pad token if missing (common issue with some models)
        if tokenizer.pad_token is None:
            logger.warning(
                f"Tokenizer for {model_name_or_path} missing pad_token, adding '[PAD]'"
            )
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        return cls(model_name_or_path, tokenizer)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens (BOS/EOS)

        Returns:
            List of token IDs
        """
        return self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors=None  # Return list, not tensor
        )

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text
        """
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for
            add_special_tokens: Whether to count special tokens

        Returns:
            Number of tokens
        """
        return len(self.encode(text, add_special_tokens=add_special_tokens))

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self._tokenizer)


async def get_tokenizer_for_model(
    model_name_or_path: str,
    model_name: str | None = None,
) -> TiktokenTokenizer | HuggingFaceTokenizer:
    """Get appropriate tokenizer for a model (async).

    This function automatically selects the right tokenizer backend based on
    the model's characteristics:
    - Third-party API models → tiktoken
    - vLLM/HuggingFace models → HuggingFace tokenizer

    Args:
        model_name_or_path: Model name or local path to model directory
        model_name: Original model name (if model_name_or_path is a local path)

    Returns:
        Tokenizer instance (tiktoken or HuggingFace)

    Example:
        ```python
        # Load from HuggingFace Hub
        tokenizer = await get_tokenizer_for_model("meta-llama/Llama-3.1-8B")

        # Load from local path (e.g., after S3 download)
        tokenizer = await get_tokenizer_for_model(
            "/tmp/model_123",
            model_name="meta-llama/Llama-3.1-8B"
        )
        ```

    Note:
        Uses asyncio.to_thread() to avoid blocking during tokenizer loading.
    """
    # Determine which model name to use for registry lookup
    registry_lookup_name = model_name or model_name_or_path

    # Look up model in registry
    model_params = ModelRegistry.get_model(registry_lookup_name)

    if model_params is None:
        logger.warning(
            f"Model {registry_lookup_name} not found in registry, defaulting to tiktoken"
        )
        return TiktokenTokenizer()

    # Use tiktoken for third-party API models
    if model_params.backend == LLMBackend.THIRD_PARTY:
        logger.info(f"Using tiktoken for third-party model {registry_lookup_name}")
        return TiktokenTokenizer()

    # Use HuggingFace tokenizer for vLLM and HuggingFace models
    logger.info(f"Using HuggingFace tokenizer for {registry_lookup_name}")
    try:
        # Load from the provided path (could be HuggingFace name or local path)
        return await HuggingFaceTokenizer.from_pretrained(model_name_or_path)
    except Exception as e:
        logger.error(
            f"Failed to load HuggingFace tokenizer from {model_name_or_path}: {e}"
        )
        logger.warning("Falling back to tiktoken")
        return TiktokenTokenizer()


def validate_tokenizer_model_compatibility(
    tokenizer: TiktokenTokenizer | HuggingFaceTokenizer,
    model_params: LLMModelParameters,
) -> bool:
    """Validate that tokenizer is compatible with model.

    Args:
        tokenizer: Tokenizer to validate
        model_params: Model parameters to validate against

    Returns:
        True if compatible

    Raises:
        ValueError: If incompatible tokenizer-model pair detected
    """
    # Tiktoken should only be used with third-party models
    if isinstance(tokenizer, TiktokenTokenizer):
        if model_params.backend != LLMBackend.THIRD_PARTY:
            logger.warning(
                f"Using tiktoken for {model_params.model_name} "
                f"(backend: {model_params.backend.name}). "
                "Consider using model-specific tokenizer for better accuracy."
            )
        return True

    # HuggingFace tokenizer should match the model
    if isinstance(tokenizer, HuggingFaceTokenizer):
        if tokenizer.model_name != model_params.model_name:
            raise ValueError(
                f"Tokenizer model mismatch: "
                f"tokenizer={tokenizer.model_name}, "
                f"model={model_params.model_name}"
            )
        return True

    return True

