"""``Embedder`` protocol + two implementations.

``InMemoryEmbedder`` is a deterministic hash-based embedder used in
unit tests — it produces a 64-d vector from a SHA-256 of the input,
normalised, so ``cosine_similarity(emb(a), emb(a)) == 1`` and
distinct inputs produce distinct vectors. No model download, no
network.

``ColonyEmbeddingClient`` wraps the existing
``cluster.embedding.EmbeddingDeployment`` so production pipelines
embed via colony's vLLM-backed embedding service. The wrapper takes
the deployment handle as a constructor argument so the caller decides
how to acquire it (typically via ``polymathera.colony.system.get_embedding_deployment``).
"""

from __future__ import annotations

import hashlib
import logging
import struct
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Async callable that produces a fixed-dim vector for each input
    string. Production embedders return floats; tests may use any
    numeric tuple."""

    embedder_id: str
    dimensions: int

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one vector per input ``text``, in input order."""
        ...


# ---------------------------------------------------------------------------
# Deterministic in-memory embedder (tests)
# ---------------------------------------------------------------------------


class InMemoryEmbedder:
    """Hash-based deterministic embedder. 64-d float vectors, L2-normalised.

    The vector for a given input is reproducible across processes
    (SHA-256 of UTF-8 bytes), and distinct inputs produce vectors with
    different content but the same dimensionality. Vector quality is
    *not* semantic — this exists for unit tests, not retrieval quality
    benchmarks.
    """

    embedder_id = "polymathera.knowledge:inmem_sha256_64d"
    dimensions = 64

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return tuple(self._one(text) for text in texts)

    def _one(self, text: str) -> tuple[float, ...]:
        # 64 floats from a stretched SHA-256 by hashing chained.
        seed = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
        floats: list[float] = []
        block = seed
        while len(floats) < self.dimensions:
            block = hashlib.sha256(block).digest()
            for i in range(0, len(block), 4):
                if len(floats) >= self.dimensions:
                    break
                # Map 4 bytes → float in [-1, 1].
                (value,) = struct.unpack(">i", block[i : i + 4])
                floats.append(value / 0x7FFFFFFF)
        # L2 normalize.
        norm = sum(v * v for v in floats) ** 0.5
        if norm == 0.0:
            return tuple(floats)
        return tuple(v / norm for v in floats)


# ---------------------------------------------------------------------------
# Production embedder — wraps colony's existing EmbeddingDeployment.
# ---------------------------------------------------------------------------


class ColonyEmbeddingClient:
    """Embedder implementation that delegates to colony's
    ``EmbeddingDeployment`` (``cluster.embedding.embedding_deployment``).

    The deployment handle's interface (per ``embedding_deployment.py``)
    exposes an async ``encode(texts: list[str]) -> list[list[float]]``
    method through the standard ``DeploymentHandle`` rpc. Since this
    is colony-runtime code, we call the deployment directly; the
    handle is supplied at construction.
    """

    def __init__(
        self,
        *,
        deployment_handle: Any,
        embedder_id: str,
        dimensions: int,
    ) -> None:
        self._handle = deployment_handle
        self.embedder_id = embedder_id
        self.dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return ()
        try:
            result = await self._handle.encode(list(texts))
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "ColonyEmbeddingClient: encode failed for %d texts (%s)",
                len(texts), exc,
            )
            raise
        # The deployment returns list[list[float]]; normalize to tuple-of-tuples.
        return tuple(tuple(float(v) for v in row) for row in result)


__all__ = ("Embedder", "InMemoryEmbedder", "ColonyEmbeddingClient")
