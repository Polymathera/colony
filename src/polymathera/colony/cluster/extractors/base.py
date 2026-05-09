"""``PdfExtractorDeployment`` ABC + shared extract-result types.

Per the multimodal-pdf-ingestion design (§§7-8), every self-hosted
layout-aware PDF extractor (Marker, Docling, MinerU) is a Ray
``serving.deployment`` exposing one endpoint::

    @serving.endpoint
    async def extract(
        self, *, pdf_bytes: bytes, options: ExtractOptions | None = None,
    ) -> ExtractResult: ...

The reader (e.g. ``MarkerPdfReader``) holds a
:class:`~polymathera.colony.distributed.ray_utils.serving.DeploymentHandle`
to one of these and calls ``extract`` per source. The deployment
returns a :class:`ExtractResult` whose markdown carries figure
references (``![label](mem://<id>)``) keyed against the
:class:`FigureBlob`s in ``figures``; the reader stores each blob via
the active :class:`~polymathera.colony.knowledge.stores.image.ImageStore`,
rewrites the markdown to the resulting ``colony-image://<sha>``
URIs, and emits :class:`~polymathera.colony.knowledge.models.ParsedSection`
objects.

Hosted readers (``MistralOcrPdfReader``, ``AnthropicPdfReader``)
bypass this ABC — they hit the vendor's HTTP endpoint directly —
but use the same :class:`ExtractResult` / :class:`FigureBlob` /
:class:`ExtractOptions` shapes so the reader-side code that walks
markdown + image references is identical regardless of backend.

The ABC itself is deliberately minimal: it carries no Ray-serve
imports or autoscaling config. Each concrete deployment subclass
applies its own ``@serving.deployment`` decorator with backend-
specific resource hints (GPU count, replica budget, autoscaling).
That keeps this module importable from non-cluster code (readers,
tests) without dragging the Ray runtime in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PdfExtractorError(RuntimeError):
    """Raised when an extractor cannot produce a result.

    Subclasses MAY raise more specific errors (auth failure, rate
    limit, malformed PDF) — readers catch the base class and surface
    the message to the ingestor's per-record error string.
    """


# ---------------------------------------------------------------------------
# Wire types
# ---------------------------------------------------------------------------


class ExtractOptions(BaseModel):
    """Per-call extraction options shared by every backend.

    Backends MAY ignore options they do not support; readers SHOULD
    pass the defaults unless the operator overrode something via the
    cluster config or KB router. Adding a new option here is a
    forwards-compatible change as long as a sensible default is
    supplied — older deployments will receive the field and ignore
    it via Pydantic ``model_config["extra"] = "allow"``.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    extract_images: bool = True
    """When True (the default), the extractor returns image bytes
    for figures / diagrams / table-as-image regions. When False,
    ``ExtractResult.figures`` is empty and the markdown contains
    only text — useful for cheap "is this PDF readable at all?"
    probes."""

    table_format: Literal["markdown", "html", "inline"] = "markdown"
    """How tables are serialised in the returned markdown:
    ``"markdown"`` (GFM tables, standard chunk-friendly),
    ``"html"`` (when row/colspan matters),
    ``"inline"`` (let the backend pick — Mistral OCR's default)."""

    pages: tuple[int, ...] | None = None
    """1-indexed page selector. ``None`` means all pages. Lets the
    operator A/B a single problematic page through the dashboard
    without re-running the whole document."""


class FigureBlob(BaseModel):
    """One image / table / diagram region returned by an extractor.

    The reader stores ``image_bytes`` via the :class:`ImageStore`
    and produces a :class:`~polymathera.colony.knowledge.models.FigureRef`
    pointing at the resulting URI. ``label`` is the in-text reference
    (``"Fig. 3"`` / ``"Table 2"``) — used both as a fallback caption
    hint and as the key the reader uses to rewrite markdown image
    references from the extractor's local ID space (``img-0.jpeg``)
    to ``colony-image://<sha>``.
    """

    model_config = ConfigDict(frozen=True)

    blob_id: str
    """Extractor-local ID. Stable within a single ``extract`` call;
    used as the key for markdown rewriting. NOT used downstream of
    the reader."""

    image_bytes: bytes
    mime: str = "image/png"
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    label: str = ""
    caption_hint: str = ""
    kind: Literal["figure", "table", "diagram", "equation", "other"] = "figure"


class ExtractResult(BaseModel):
    """Output of one ``PdfExtractorDeployment.extract`` call.

    Markdown is *page-aware* — the extractor SHOULD insert a
    ``\\n\\n<!-- page: N -->\\n\\n`` separator between pages so the
    reader can split into one :class:`~polymathera.colony.knowledge.models.ParsedSection`
    per page without re-parsing layout. Backends that cannot easily
    emit per-page boundaries MAY return the whole document as a
    single page; the reader degrades to one section for the document.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    markdown: str
    """Concatenated per-page markdown. See class docstring re: page
    separators."""

    figures: tuple[FigureBlob, ...] = Field(default_factory=tuple)

    backend: str = ""
    """Free-form name of the producing backend (``"marker"``,
    ``"mistral_ocr"``, …). Recorded in
    ``Chunk.extra["metadata_origin"]`` so retrieval can trace which
    extractor produced a given chunk."""

    page_count: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class PdfExtractorDeployment(ABC):
    """Contract every self-hosted PDF extractor deployment fulfils.

    Implementations apply ``@serving.deployment`` themselves (with
    backend-appropriate autoscaling / GPU options), then expose
    :meth:`extract` as a ``@serving.endpoint``. The ABC here exists
    so reader code can type-check against the protocol without
    knowing which backend it has.

    Subclasses MUST:

    - Set the class attribute ``backend_name`` to a unique stable
      identifier (``"marker"``, ``"docling"``, ``"mineru"``). The
      cluster config's ``pdf_extractor.backend`` selector matches on
      this string.
    - Implement :meth:`extract` with the signature documented below.
    - Apply ``@serving.deployment`` to the concrete class with the
      appropriate ray actor options.
    """

    #: Stable identifier for this backend. Subclasses override.
    backend_name: str = ""

    @abstractmethod
    async def extract(
        self,
        *,
        pdf_bytes: bytes,
        options: ExtractOptions | None = None,
    ) -> ExtractResult:
        """Convert ``pdf_bytes`` to markdown + figure blobs.

        Args:
            pdf_bytes: Raw PDF payload. Implementations MUST NOT
                require a filename; the reader hands them bytes
                straight from ``RawDocument.bytes_``.
            options: Per-call options. ``None`` means use backend
                defaults (extract images, GFM tables, all pages).

        Returns:
            :class:`ExtractResult` with concatenated markdown and
            figure blobs. The reader is responsible for storing
            blobs via the :class:`ImageStore` and rewriting markdown
            references.

        Raises:
            PdfExtractorError: On any backend-side failure. Readers
                catch this and surface ``str(exc)`` into the
                ingestion record's ``error`` field.
        """
