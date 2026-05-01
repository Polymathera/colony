"""Claim extractors that turn ``Chunk``s into typed ``Claim``s.

Two implementations:

- ``DeterministicClaimExtractor`` — pure-Python, rule-based; deterministic
  for tests and as a smoke-test in deployments without an LLM.
- ``LLMClaimExtractor`` — typed Pydantic-schema extraction over
  colony's existing LLM cluster. The interface is here; the
  real-LLM wiring (a deployment handle for ``LLMCluster``) lives
  in the ``Ingestor`` constructor — the extractor itself takes a
  callable that returns a JSON-shaped extraction.
"""

from __future__ import annotations

from .claims import (
    ClaimExtractor,
    DeterministicClaimExtractor,
    ExtractionPrompt,
    LLMCallable,
    LLMClaimExtractor,
)


__all__ = (
    "ClaimExtractor",
    "DeterministicClaimExtractor",
    "ExtractionPrompt",
    "LLMCallable",
    "LLMClaimExtractor",
)
