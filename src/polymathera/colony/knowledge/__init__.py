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
  (``ScopedRetrievalCapability``, ``GroundedRetrievalCapability``,
  ``GraphRetrievalCapability``, ``BudgetedRetrievalCapability``,
  ``StandardsRetrievalCapability``).

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
    MarkdownChunker,
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
from .acquirers import (
    AcquiredSource,
    AcquirerRegistry,
    AcquirerStrategy,
    default_registry as default_acquirer_registry,
    _TODO_ArxivAcquirer,
    _TODO_DoiAcquirer,
    _TODO_HttpAcquirer,
    _TODO_IeeeXploreAcquirer,
    _TODO_SaeMobilusAcquirer,
    _TODO_SemanticScholarAcquirer,
)
from .ingestion import HumanReviewQueueCallback, Ingestor
from .monorepo_persisted_ingestor import (
    MonorepoPersistedIngestor,
    SidecarManifest,
)
from .models import (
    Chunk,
    CitationSpan,
    Claim,
    CorpusTier,
    EmbeddedChunk,
    FigureRef,
    IngestionPolicy,
    IngestionRecord,
    IngestionStatus,
    KnowledgeFormat,
    ParsedSection,
    RawDocument,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResult,
    SourceSummary,
    deterministic_source_uri,
    tier_priority,
)
from .readers import (
    AnthropicPdfReader,
    CsvReader,
    FormatReader,
    FormatReaderError,
    GeminiPdfReader,
    GrobidMetadataReader,
    GrobidPdfReader,
    HtmlReader,
    JsonlReader,
    JupyterReader,
    LlamaParsePdfReader,
    LlamaParseTier,
    MarkdownReader,
    MistralOcrPdfReader,
    PdfExtractorBackend,
    PdfReader,
    PlainTextReader,
    ReaderRegistry,
    SourceCodeReader,
    default_registry,
    default_registry_with_grobid,
    default_registry_with_pdf_extractor,
)
from .retrieval import (
    # New names (canonical):
    BudgetedRetrievalCapability,
    GraphRetrievalCapability,
    GroundedRetrievalCapability,
    RetrievalCapability,
    RetrievalDeps,
    ScopedRetrievalCapability,
    StandardsRetrievalCapability,
    # Deprecated aliases (pending removal):
    BudgetedRetrievalCapability,
    GraphRetrievalCapability,
    GroundedRetrievalCapability,
    RetrievalCapability,
    ScopedRetrievalCapability,
    StandardsRetrievalCapability,
)
from .stores import (
    GraphEdge,
    GraphNode,
    GraphQueryResult,
    GraphStore,
    GraphStoreError,
    ImageStore,
    ImageStoreError,
    InMemoryGraphStore,
    InMemoryImageStore,
    InMemoryVectorStore,
    KuzuGraphStore,
    LocalFsImageStore,
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
    "FigureRef",
    "Chunk",
    "Claim",
    "EmbeddedChunk",
    "RetrievalQuery",
    "RetrievalHit",
    "RetrievalResult",
    "SourceSummary",
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
    "GrobidMetadataReader",
    "MistralOcrPdfReader",
    "AnthropicPdfReader",
    "GeminiPdfReader",
    "LlamaParsePdfReader",
    "LlamaParseTier",
    "PdfExtractorBackend",
    "default_registry",
    "default_registry_with_grobid",
    "default_registry_with_pdf_extractor",
    # Chunking
    "ChunkerConfig",
    "ProseChunker",
    "MarkdownChunker",
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
    "ImageStore",
    "ImageStoreError",
    "InMemoryImageStore",
    "LocalFsImageStore",
    # Ingestor + monorepo-persisted wrapper
    "Ingestor",
    "HumanReviewQueueCallback",
    "MonorepoPersistedIngestor",
    "SidecarManifest",
    # Acquirers (remote-source fetch strategies for repo_map.yaml)
    "AcquirerStrategy",
    "AcquiredSource",
    "AcquirerRegistry",
    "default_acquirer_registry",
    "_TODO_ArxivAcquirer",
    "_TODO_DoiAcquirer",
    "_TODO_HttpAcquirer",
    "_TODO_IeeeXploreAcquirer",
    "_TODO_SaeMobilusAcquirer",
    "_TODO_SemanticScholarAcquirer",
    # Retrieval (current names)
    "RetrievalCapability",
    "RetrievalDeps",
    "ScopedRetrievalCapability",
    "GroundedRetrievalCapability",
    "GraphRetrievalCapability",
    "BudgetedRetrievalCapability",
    "StandardsRetrievalCapability",
    # Retrieval (deprecated aliases — pending removal)
    "RetrievalCapability",
    "ScopedRetrievalCapability",
    "GroundedRetrievalCapability",
    "GraphRetrievalCapability",
    "BudgetedRetrievalCapability",
    "StandardsRetrievalCapability",
)
