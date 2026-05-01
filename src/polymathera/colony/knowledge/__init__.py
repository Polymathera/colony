"""Polymathera Colony — Knowledge & retrieval framework (Phase C1a).

Master §3.2 / §6.3 / §6.4. The colony-generic, in-process knowledge
layer:

- **Models** (``models.py``): typed shapes for documents, sections,
  chunks, claims, retrieval queries / hits / results, ingestion records.
- **Format detection** (``formats.py``): extension + magic-byte
  routing.
- **Readers** (``readers/``): ``FormatReader`` ABC + concrete
  in-process readers for plain text, Markdown, HTML, JSONL, CSV,
  source code, Jupyter notebooks, and PDF (pypdf fallback).
- **Chunking** (``chunking.py``): paragraph-aware ``ProseChunker``
  + ``CodeChunker`` that delegates to the existing
  ``LanguageAwareTextChunker`` (no duplication).
- **Extractors** (``extractors/``): ``ClaimExtractor`` ABC +
  ``DeterministicClaimExtractor`` (rule-based) + ``LLMClaimExtractor``
  (typed-schema, binds to colony's LLM cluster via injection).
- **Embedder** (``embedder.py``): ``Embedder`` Protocol +
  ``InMemoryEmbedder`` (deterministic SHA-256-based, for tests) +
  ``ColonyEmbeddingClient`` (wraps the existing
  ``cluster.embedding.EmbeddingDeployment``).
- **Stores** (``stores/``): ``VectorStore`` ABC +
  ``InMemoryVectorStore`` (full impl) + ``QdrantVectorStore`` stub;
  ``GraphStore`` ABC + ``InMemoryGraphStore`` (with a tiny Cypher-like
  DSL) + ``KuzuGraphStore`` stub.
- **Ingestor** (``ingestion.py``): orchestrator that runs the master
  §6.3 pipeline end to end with sampled human-review queueing.
- **Retrieval** (``retrieval/``): five master §6.4 retrieval modes
  registered as Phase C2 ``ToolAdapter``s
  (``ScopedRetrievalAdapter``, ``GroundedRetrievalAdapter``,
  ``GraphRetrievalAdapter``, ``BudgetedRetrievalAdapter``,
  ``StandardsRetrievalAdapter``).

This is C1a — the in-process / mock-friendly framework. C1b lands the
real Qdrant + Kùzu + GROBID Docker wiring + the corpus-management
dashboard surface.

The framework is colony-generic (master §1.4): nothing here carries
design-engineering-shaped semantics; CPS-shared knowledge curation
agents (`KnowledgeCuratorAgent`) and per-domain corpus catalogues
(QS Tier-1 textbook canon, RACER vehicle-dynamics literature, …) layer
on top in `cps/`.
"""

from __future__ import annotations

from .chunking import (
    ChunkerConfig,
    CodeChunker,
    ProseChunker,
    TokenCounter,
    default_token_counter,
)
from .embedder import (
    ColonyEmbeddingClient,
    Embedder,
    InMemoryEmbedder,
)
from .extractors import (
    ClaimExtractor,
    DeterministicClaimExtractor,
    ExtractionPrompt,
    LLMCallable,
    LLMClaimExtractor,
)
from .formats import (
    EXTENSION_MAP,
    SOURCE_CODE_LANGUAGE,
    detect_format,
    language_for_source_code,
)
from .bulk_acquisition import (
    AcquiredSource,
    AcquirerStrategy,
    AcquisitionEntry,
    BulkAcquisitionCapability,
    BulkAcquisitionError,
    BulkAcquisitionReport,
    CorpusManifest,
    LocalPathAcquirer,
    ManifestEntry,
    _TODO_ArxivAcquirer,
    _TODO_DoiAcquirer,
    _TODO_HttpAcquirer,
    _TODO_IeeeXploreAcquirer,
    _TODO_NeuroImageAcquirer,
    _TODO_SaeMobilusAcquirer,
    _TODO_SemanticScholarAcquirer,
)
from .ingestion import HumanReviewQueueCallback, Ingestor
from .models import (
    Chunk,
    CitationSpan,
    Claim,
    CorpusTier,
    EmbeddedChunk,
    IngestionPolicy,
    IngestionRecord,
    IngestionStatus,
    KnowledgeFormat,
    ParsedSection,
    RawDocument,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResult,
    deterministic_source_uri,
    tier_priority,
)
from .readers import (
    CsvReader,
    FormatReader,
    FormatReaderError,
    GrobidPdfReader,
    HtmlReader,
    JsonlReader,
    JupyterReader,
    MarkdownReader,
    PdfReader,
    PlainTextReader,
    ReaderRegistry,
    SourceCodeReader,
    default_registry,
    default_registry_with_grobid,
)
from .retrieval import (
    BudgetedRetrievalAdapter,
    GraphRetrievalAdapter,
    GroundedRetrievalAdapter,
    RetrievalAdapter,
    RetrievalDeps,
    ScopedRetrievalAdapter,
    StandardsRetrievalAdapter,
)
from .stores import (
    GraphEdge,
    GraphNode,
    GraphQueryResult,
    GraphStore,
    GraphStoreError,
    InMemoryGraphStore,
    InMemoryVectorStore,
    KuzuGraphStore,
    QdrantVectorStore,
    VectorStore,
    VectorStoreError,
)


__all__ = (
    # Models
    "KnowledgeFormat",
    "CorpusTier",
    "tier_priority",
    "IngestionPolicy",
    "CitationSpan",
    "RawDocument",
    "ParsedSection",
    "Chunk",
    "Claim",
    "EmbeddedChunk",
    "RetrievalQuery",
    "RetrievalHit",
    "RetrievalResult",
    "IngestionStatus",
    "IngestionRecord",
    "deterministic_source_uri",
    # Formats
    "EXTENSION_MAP",
    "SOURCE_CODE_LANGUAGE",
    "detect_format",
    "language_for_source_code",
    # Readers
    "FormatReader",
    "FormatReaderError",
    "ReaderRegistry",
    "PlainTextReader",
    "MarkdownReader",
    "HtmlReader",
    "JsonlReader",
    "CsvReader",
    "SourceCodeReader",
    "JupyterReader",
    "PdfReader",
    "GrobidPdfReader",
    "default_registry",
    "default_registry_with_grobid",
    # Chunking
    "ChunkerConfig",
    "ProseChunker",
    "CodeChunker",
    "TokenCounter",
    "default_token_counter",
    # Extractors
    "ClaimExtractor",
    "DeterministicClaimExtractor",
    "ExtractionPrompt",
    "LLMCallable",
    "LLMClaimExtractor",
    # Embedder
    "Embedder",
    "InMemoryEmbedder",
    "ColonyEmbeddingClient",
    # Stores
    "VectorStore",
    "VectorStoreError",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "GraphStore",
    "GraphStoreError",
    "GraphNode",
    "GraphEdge",
    "GraphQueryResult",
    "InMemoryGraphStore",
    "KuzuGraphStore",
    # Ingestor
    "Ingestor",
    "HumanReviewQueueCallback",
    # Bulk acquisition (master §6.6)
    "AcquirerStrategy",
    "AcquiredSource",
    "AcquisitionEntry",
    "BulkAcquisitionCapability",
    "BulkAcquisitionError",
    "BulkAcquisitionReport",
    "CorpusManifest",
    "LocalPathAcquirer",
    "ManifestEntry",
    "_TODO_ArxivAcquirer",
    "_TODO_DoiAcquirer",
    "_TODO_HttpAcquirer",
    "_TODO_IeeeXploreAcquirer",
    "_TODO_NeuroImageAcquirer",
    "_TODO_SaeMobilusAcquirer",
    "_TODO_SemanticScholarAcquirer",
    # Retrieval
    "RetrievalAdapter",
    "RetrievalDeps",
    "ScopedRetrievalAdapter",
    "GroundedRetrievalAdapter",
    "GraphRetrievalAdapter",
    "BudgetedRetrievalAdapter",
    "StandardsRetrievalAdapter",
)
