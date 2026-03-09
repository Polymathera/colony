from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, ClassVar
from pydantic import Field
from overrides import override

import numpy as np

from polymathera.colony.distributed.config import register_polymathera_config, ConfigComponent
from polymathera.colony.distributed import get_polymathera
from polymathera.colony.distributed.caching.simple import CacheConfig
from polymathera.colony.distributed.metrics.common import BaseMetricsMonitor
from polymathera.colony.cluster.embedding import TextChunkerBase
from polymathera.colony.system import get_llm_cluster
from polymathera.colony.utils import setup_logger, cleanup_dynamic_asyncio_tasks

from .base import AnalyzerConfig, BaseAnalyzer, FileContentCache

logger = setup_logger(__name__)


@register_polymathera_config()
class ChunkingConfig(ConfigComponent):
    """Configuration for content chunking"""

    # Maximum chunk size in characters
    max_chunk_size: int = 2048

    # Minimum chunk size to process
    min_chunk_size: int = 100

    # Overlap between chunks (percentage)
    chunk_overlap: float = 0.1

    # Whether to preserve function/class boundaries
    respect_code_blocks: bool = True

    # Maximum chunks per file
    max_chunks_per_file: int = 10

    # Patterns for code block detection
    block_start_patterns: dict[str, str] = Field(
        default_factory=lambda: {
            "python": r"(class|def|async def)\s+\w+.*:$",
            "javascript": r"(class|function|const.*=>)\s+\w+.*{$",
            "java": r"(class|interface|enum|public|private|protected).*{$",
            "typescript": r"(class|interface|function|const.*=>)\s+\w+.*{$",
            "go": r"func\s+\w+.*{$",
            # TODO: Add more languages
            # TODO: Reuse code from polymathera.colony.samples.paging.sharding.code_splitting
        }
    )

    # Performance thresholds
    cross_language_threshold: float = 0.6

    # Language-specific settings
    language_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "python": 1.0,
            "javascript": 0.9,
            "java": 0.9,
            "typescript": 0.9,
            "go": 0.8,
            # TODO: Add more languages
        }
    )

    CONFIG_PATH: ClassVar[str] = "llms.inference.cluster.embedding.chunking"


@register_polymathera_config()
class SemanticAnalyzerConfig(AnalyzerConfig):
    # Content processing
    max_content_length: int = 2048
    min_content_length: int = 10

    # Chunking configuration
    chunk_large_files: bool = True
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)

    semantic_cache_config: CacheConfig | None = Field(
        default=None
    )  # If None, the default will be automatically loaded from config manager

    CONFIG_PATH: ClassVar[str] = "llms.sharding.analyzers.semantic"




# TODO: Add batch processing methods
# TODO: Add more sophisticated language handling
# TODO: Add cross-language similarity adjustments
# TODO: Add embedding model auto-selection
# TODO: Add more preprocessing options

# TODO:
# 1. Add more sophisticated chunk combination strategies:
#    - Attention-based combination
#    - Hierarchical pooling
#    - Semantic importance weighting
# 2. Enhance code block detection:
#    - More language patterns
#    - Nested block handling
#    - Smarter boundary detection
# 3. Add chunk optimization:
#    - Dynamic chunk sizing
#    - Content-aware splitting
#    - Semantic boundary detection
# 4. Implement additional features:
#    - Chunk similarity analysis
#    - Cross-chunk relationships
#    - Chunk importance scoring



class LanguageAwareTextChunker(TextChunkerBase):
    def __init__(self, language: str, config: ChunkingConfig | None = None):
        self.config: ChunkingConfig | None = config
        self.language = language

    async def initialize(self):
        self.config = await ChunkingConfig.check_or_get_component(self.config)

    @override
    def chunk_content(self, content: str) -> Iterator[str]:
        """Split content into semantic chunks

        Strategies:
        1. Code block aware: Respects function/class boundaries
        2. Sliding window: For languages without clear block patterns
        3. Hybrid: Combines both approaches when possible
        """
        try:
            if (
                self.config.respect_code_blocks
                and self.language in self.config.block_start_patterns
            ):
                yield from self._chunk_by_code_blocks(content)
            else:
                yield from self._chunk_by_sliding_window(content)
        except Exception as e:
            logger.error(f"Chunking error: {e}")
            yield content[: self.config.max_chunk_size]

    def _chunk_by_code_blocks(self, content: str) -> Iterator[str]:
        """Split content by code block boundaries"""
        try:
            # Get language-specific pattern
            pattern = self.config.block_start_patterns[self.language]

            # Find all block boundaries
            lines = content.splitlines()
            current_chunk = []
            current_size = 0

            for line in lines:
                line_size = len(line) + 1  # +1 for newline

                # Check if line starts a new block
                if re.match(pattern, line.strip()):
                    # Yield current chunk if it's large enough
                    if current_size >= self.config.min_chunk_size:
                        yield "\n".join(current_chunk)
                    current_chunk = []
                    current_size = 0

                current_chunk.append(line)
                current_size += line_size

                # Split if chunk gets too large
                if current_size >= self.config.max_chunk_size:
                    yield "\n".join(current_chunk)
                    current_chunk = []
                    current_size = 0

            # Yield remaining chunk
            if current_chunk and current_size >= self.config.min_chunk_size:
                yield "\n".join(current_chunk)

        except Exception as e:
            logger.error(f"Code block chunking error: {e}")
            yield content[: self.config.max_chunk_size]

    def _chunk_by_sliding_window(self, content: str) -> Iterator[str]:
        """Split content using sliding window approach"""
        try:
            chunk_size = self.config.max_chunk_size
            overlap_size = int(chunk_size * self.config.chunk_overlap)

            # Split into chunks with overlap
            start = 0
            while start < len(content):
                end = start + chunk_size

                # Find a good break point
                if end < len(content):
                    # Try to break at newline
                    break_point = content.rfind("\n", start, end)
                    if break_point > start:
                        end = break_point

                chunk = content[start:end]
                if len(chunk) >= self.config.min_chunk_size:
                    yield chunk

                start = end - overlap_size

        except Exception as e:
            logger.error(f"Sliding window chunking error: {e}")
            yield content[: self.config.max_chunk_size]


def _combine_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """Combine chunk embeddings via mean-pooling with L2 normalization."""
    if not embeddings:
        return np.zeros(768)
    if len(embeddings) == 1:
        return embeddings[0]
    combined = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    return combined


def get_similar_pairs(
    embeddings: dict[str, np.ndarray],
    batch_size: int = 100,
) -> dict[tuple[str, str], float]:
    """Compute pairwise cosine similarities between file embeddings.

    Args:
        embeddings: Mapping of file path to embedding vector.
        batch_size: Number of rows to process at once (controls peak memory).

    Returns:
        Dict mapping ``(file_a, file_b)`` to cosine similarity for every
        unique pair (upper-triangle, ``file_a < file_b``).
    """
    if len(embeddings) < 2:
        return {}

    files = sorted(embeddings.keys())
    n = len(files)

    # Stack into matrix and L2-normalize rows for cosine similarity via dot product
    matrix = np.array([embeddings[f] for f in files], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    matrix = matrix / norms

    similarities: dict[tuple[str, str], float] = {}

    # Process in row-batches to limit memory for large file sets
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # (batch_rows, dim) @ (dim, n) -> (batch_rows, n)
        sim_block = matrix[start:end] @ matrix.T
        for i_local, i_global in enumerate(range(start, end)):
            # Only upper triangle: j > i_global
            for j in range(i_global + 1, n):
                similarities[(files[i_global], files[j])] = float(sim_block[i_local, j])

    return similarities


class SemanticAnalyzerMetricsMonitor(BaseMetricsMonitor):
    """Base class for Prometheus metrics monitoring using node-global HTTP server."""

    def __init__(self,
                 enable_http_server: bool = True,
                 service_name: str = "service"):
        super().__init__(enable_http_server, service_name)

        self.logger.info(f"Initializing SemanticAnalyzerMetricsMonitor instance {id(self)}...")

        self.cache_hits = self.create_counter(
            "semantic_cache_hits_total",
            "Number of cache hits",
            labelnames=["cache_type"]
        )


class SemanticAnalyzer(BaseAnalyzer):
    """Analyzes semantic relationships between code files that offers language-aware analysis
    """

    def __init__(self, file_content_cache: FileContentCache, config: SemanticAnalyzerConfig | None = None):
        super().__init__("semantic", file_content_cache)
        self.config = config
        self.semantic_cache = None
        self.metrics = SemanticAnalyzerMetricsMonitor()

    async def initialize(self):
        self.config = await SemanticAnalyzerConfig.check_or_get_component(self.config)
        await super().initialize()

    async def _ensure_caches_initialized(self) -> None:
        """Initialize caches if not already done."""
        # TODO: Add cache namespaces
        if self.semantic_cache is None:
            self.semantic_cache = await get_polymathera().create_distributed_simple_cache(
                namespace="semantic_file_embeddings",  # TODO: Does this need to be VMR-specific?
                config=self.config.semantic_cache_config,
            )

    async def cleanup(self) -> None:
        """Cleanup background tasks and resources"""
        await super().cleanup()

        try:
            await cleanup_dynamic_asyncio_tasks(self, raise_exceptions=False)
            if self.semantic_cache:
                await self.semantic_cache.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up SemanticAnalyzer: {e}")

    async def get_file_embedding(self, file_path: str, language: str | None = None) -> np.ndarray:
        """Generate embedding for file content, with optional chunking."""
        content = await self.file_content_cache.read_file(file_path)
        return await self._embed_with_chunking(content, language)

    async def _embed_with_chunking(self, content: str, language: str | None = None) -> np.ndarray:
        """Embed text, chunking first if configured and content is long."""
        embedder = get_llm_cluster()

        if self.config.chunk_large_files and len(content) > self.config.max_content_length:
            chunker = LanguageAwareTextChunker(language, self.config.chunking)
            await chunker.initialize()
            chunks = list(chunker.chunk_content(content))
            if not chunks:
                return np.zeros(768)  # Fallback dimension
            chunks = chunks[: self.config.chunking.max_chunks_per_file]
            raw = await embedder.embed(chunks)
            embeddings = [np.array(v) for v in raw]
            return _combine_embeddings(embeddings)

        raw = await embedder.embed([content])
        return np.array(raw[0])

    @override
    async def _analyze_file_impl(
        self, file_path: str, content: str, language: str | None = None, **kwargs
    ) -> dict[str, Any]:
        """Generate embeddings for file content"""
        embedding = await self._embed_with_chunking(content, language)
        content_hash = await self.file_content_cache.get_file_hash(file_path)

        result = {
            "embedding": embedding,
            "language": language,
            "file_path": file_path,
            "content_hash": content_hash,
        }

        return result

    def _get_fallback_result(self) -> dict[str, Any]:
        """Return fallback result when analysis fails"""
        return {
            "embedding": np.zeros(768),  # TODO: Make this configurable and model-specific.
            "language": None,
            "file_path": None,
            "content_hash": None,
            "num_chunks": 0,
            "models_used": []
        }

    async def get_similarity_matrix(self, file_paths: list[str], languages: list[str], batch_size: int = 100) -> dict[tuple[str, str], float]:
        """Get pairs of file paths with high similarity."""
        # Get semantic embeddings with language context
        # TODO: Parallelize this loop
        embeddings = {
            file: await self._get_embedding_cached(file, languages[i])
            for i, file in enumerate(file_paths)
        }
        # Calculate pairwise cosine similarities
        return get_similar_pairs(embeddings, batch_size)

    async def _get_embedding_cached(self, file: str, language: str) -> np.ndarray:
        # Ensure caches are initialized
        await self._ensure_caches_initialized()

        cache_key = f"semantic:{file}"
        embedding = await self.semantic_cache.get(cache_key)
        if embedding is not None:
            self.metrics.cache_hits.labels(cache_type="semantic").inc()
        else:
            # Include language information in semantic analysis
            embedding = await self.get_file_embedding(
                file, language=language
            )
            await self.semantic_cache.set(cache_key, embedding)
        return embedding


