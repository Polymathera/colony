"""Claim extractors that turn ``Chunk``s into typed ``Claim``s.

Two implementations:

- ``DeterministicClaimExtractor`` — pure-Python, rule-based; deterministic
  for tests and as a smoke-test in deployments without an LLM.
- ``LLMClaimExtractor`` — typed Pydantic-schema extraction over
  colony's existing LLM cluster. The extractor's ``SCHEMA`` is the
  decoder-level contract honored natively by every deployment
  (Anthropic tool-use, vLLM ``guided_json``, OpenRouter
  ``response_format``); the real-LLM wiring is a
  :data:`TypedLLMCallable` the ``Ingestor`` constructor supplies.
"""

from __future__ import annotations

from .claims import (
    ClaimExtractor,
    ClaimList,
    DeterministicClaimExtractor,
    ExtractedClaim,
    ExtractionPrompt,
    LLMClaimExtractor,
    TypedLLMCallable,
)


__all__ = (
    "ClaimExtractor",
    "ClaimList",
    "DeterministicClaimExtractor",
    "ExtractedClaim",
    "ExtractionPrompt",
    "LLMClaimExtractor",
    "TypedLLMCallable",
)
