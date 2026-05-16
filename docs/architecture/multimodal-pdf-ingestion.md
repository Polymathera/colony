# Multimodal PDF Ingestion

Status: **Phase 1+2 — iteration 3.1 landed (2026-05-08)**. All four
hosted readers (Mistral OCR / Anthropic / Gemini / LlamaParse)
plus three self-hosted deployments (Marker / Docling / MinerU)
selectable via the top-level `knowledge:` YAML section.
GROBID demoted to a metadata-only sibling reader. Cost/quality
tradeoff is first-class — every backend exposes a tier knob via
its reader constructor, forwarded verbatim through the `options`
block. Knowledge-stack knobs (PDF extractor, image dir, Qdrant
URL, GROBID URL) live in the typed `KnowledgeConfig` and flow to
every Ray worker via `POLYMATHERA_CONFIG`; env-var passthrough is
limited to bootstrap and secrets.
Owner: knowledge layer
Related: [knowledge-capabilities.md](knowledge-capabilities.md), [literature-context-page-source.md](literature-context-page-source.md)

This document is the working design for replacing GROBID as the
content extractor in the knowledge layer, lifting figures and tables
into the data model as first-class entities, and growing toward a
full multimodal pipeline (figure captioning, multimodal retrieval,
agent vision system).

It is also the **progress tracker** — each phase has a checklist;
update it as work lands.

---

## 1. Problem statement

The knowledge layer currently routes every PDF through
[`GrobidPdfReader`](../../src/polymathera/colony/knowledge/readers/grobid_pdf.py)
when `default_registry_with_grobid` is the active reader registry
(see [`readers/__init__.py:49-77`](../../src/polymathera/colony/knowledge/readers/__init__.py#L49-L77)).
GROBID was designed for **bibliographic** extraction: header,
authors, affiliations, references. Its full-text segmentation
identifies figures, tables, and formulas mostly so it can route
surrounding prose correctly — not so it can faithfully reconstruct
them. The maintainer has publicly acknowledged that formula, figure,
and table extraction "is really bad" and that has not changed
materially.

Concretely, in the operator's KB tab today:
- Equations come through as broken plain text ("∇ × E = -∂B/∂t" lost
  to "∇× E= − ∂B∂t" or worse)
- Tables collapse to whitespace-shuffled prose
- Figure captions and bodies are torn apart
- No image bytes are extracted at all

CPS designs (the colony's primary use case) are inherently
multimodal: control diagrams, schematics, datasheets, simulation
plots, mechanical CAD views, oscilloscope traces. A text-only
extractor cannot cover this corpus.

## 2. Goals / non-goals

### Goals

- Replace GROBID as the **content** extractor with a layout-aware
  reader that produces clean Markdown (LaTeX equations preserved,
  GFM tables, figure references) — selectable between **Marker**,
  **Docling**, and **MinerU** per cluster config so the operator can
  evaluate them on real corpora.
- Lift figures into the data model as first-class
  `FigureRef(image_uri, bbox, page, caption_hint)` so chunkers,
  retrievers, and agents can address them by ID.
- Provide a **third-party-API path** for CPU-only environments
  (Anthropic native PDF, Mistral OCR) so local testing isn't gated
  on a 20s-per-page CPU run of Marker/Docling/MinerU.
- Preserve GROBID for what it's actually good at — header/author/
  affiliation/bibliography metadata — by routing those into
  `ParsedSection.extra` rather than the chunk text.
- Add a `VisionDeployment` that wraps `AnthropicLLMDeployment` for
  per-figure captioning at ingest time. Captions are stored as
  sibling `Chunk`s with `data_type="figure_caption"` so plain Qdrant
  text search finds them.
- Wire image bytes into a content-addressed `ImageStore` under
  `/mnt/shared/` (no S3 dep, fits the existing
  `colony-shared` volume).

### Non-goals (this design doc)

- ColPali / page-image multi-vector retrieval (Phase 5, deferred —
  expensive, only worthwhile after caption-first is measured).
- Per-document-type pipelines for datasheets / patents / standards
  (Phase 4, deferred — needs throughput data first).
- A self-hosted Qwen2-VL captioner (deferred — local-CPU testing
  uses Anthropic; Phase 3 ships with hosted only).
- Replacing the agent's planner with a multimodal LLM as policy
  itself (Phase 6, distinct workstream).

## 3. Operator decisions (locked)

These are settled — they shape the architecture below.

| # | Decision | Owner | Date |
|---|---|---|---|
| 1 | Three self-hosted extractors as separate `serving.deployment`s: **Marker**, **Docling**, **MinerU**. The cluster config selects which to deploy. | operator | 2026-05-08 |
| 2 | Captioner v1 = `AnthropicVisionDeployment` (wraps the existing `AnthropicLLMDeployment`). No GPUs in local testing. | operator | 2026-05-08 |
| 3 | Image store backend = local filesystem under `/mnt/shared/colony-images/<sha256[:2]>/<sha256>.{ext}`. Reuses the `colony-shared` Docker volume. | operator | 2026-05-08 |
| 4 | Phase 1 scope = reader swap + figures-in-data-model (Phase 1+2 ship together). | operator | 2026-05-08 |
| 5 | Plus: a hosted-API extractor path so CPU testing doesn't hinge on Marker/Docling/MinerU latency. See §5 for the survey and §6 for the recommended primary. | operator | 2026-05-08 |
| 6 | **Marker is GPL-3.0 — keep it behind a build flag.** It is registered as the `marker` backend in :func:`default_registry_with_pdf_extractor`, but is not installed by the default `colony:local` image. Operators opt in by adding `marker` to the `knowledge_marker` poetry extra and rebuilding. Docling (MIT) and MinerU (AGPL-3.0) ship by default. | operator | 2026-05-08 |
| 7 | Markdown chunker question (open #2 in §11): **subclass `ProseChunker` to treat fenced code blocks, GFM tables, and `$...$` math as atomic units.** Defer the implementation until row 1.14 hits a real corpus where blocks split across chunks; ProseChunker today already preserves paragraph boundaries which is sufficient for Mistral OCR's per-page emission. Locked direction; follow-up implementation tracked in iteration 2. | operator | 2026-05-08 |

## 4. Architecture overview

```
                    ┌──────────────────────────────────────────────┐
                    │          Ingestor (knowledge.ingestion)      │
                    └─────┬──────────────────────────────────┬─────┘
                          │                                  │
                          ▼                                  ▼
            ┌──────────────────────────┐    ┌─────────────────────────────┐
            │   FormatReader (PDF)     │    │  ImageStore (NEW)           │
            │   one of:                │    │  • InMemoryImageStore       │
            │   • MarkerPdfReader      │◄──►│  • LocalFsImageStore        │
            │   • DoclingPdfReader     │    │      /mnt/shared/colony-    │
            │   • MinerUPdfReader      │    │      images/<sha[:2]>/<sha> │
            │   • AnthropicPdfReader   │    └─────────────────────────────┘
            │   • MistralOcrPdfReader  │
            │   (selectable)           │
            └──────────────┬───────────┘
                           │
                  ParsedSection(text=md, figures=(FigureRef, ...))
                           │
                           ▼
            ┌──────────────────────────┐
            │   Chunker                │  produces Chunk + propagates
            └──────────────┬───────────┘  figure refs into chunk.extra
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌───────────────┐         ┌─────────────────────┐
     │  Embedder     │         │  Captioner (NEW)    │
     │  → vector     │         │  AnthropicVision    │
     └───────┬───────┘         │  → caption text     │
             │                 └────────┬────────────┘
             ▼                          ▼
     ┌──────────────────────────────────────────────────┐
     │  Vector store — text chunks + figure_caption     │
     │  chunks (separate data_type, linked by figure_id)│
     └──────────────────────────────────────────────────┘
```

Three new components, three modified ones:

**New**:
- `ImageStore` ABC + `InMemoryImageStore`, `LocalFsImageStore`
- `FigureRef` model
- `AnthropicVisionDeployment` (or extend `AnthropicLLMDeployment`)
- `MarkerPdfReader`, `DoclingPdfReader`, `MinerUPdfReader` plus
  `AnthropicPdfReader`, `MistralOcrPdfReader`
- `MarkerExtractorDeployment`, `DoclingExtractorDeployment`,
  `MinerUExtractorDeployment` (Ray serving deployments)
- Per-cluster `pdf_extractor` config field

**Modified**:
- `ParsedSection` gains `figures: tuple[FigureRef, ...]` and `format` (`"text"` vs `"markdown"`)
- `Chunk.extra` gains documented keys `figure_ids: list[str]`,
  `metadata_origin: "grobid" | "marker" | ...`
- `RetrievalHit` gains optional `figures: tuple[FigureRef, ...]` so
  the agent's planner can fetch image URIs alongside text

## 5. Library / API survey

### Self-hosted (Phase 1 deployments — picked by operator decision #1)

| Tool | Strength | Output | CPU latency (rough, A4 page) | License |
|---|---|---|---|---|
| **Marker** | General-purpose; equations via Texify; broad format coverage (PDF/PPTX/DOCX/XLSX/EPUB) | Markdown / JSON / HTML | 5–15 s/page CPU, sub-1 s GPU | GPL-3.0 (commercial license available) |
| **Docling** | IBM Research; strong table model (TableFormer); LangChain/LlamaIndex integrations | `DoclingDocument` → Markdown | 10–30 s/page CPU | MIT |
| **MinerU** | OpenDataLab; excellent on heavy/complex layouts and CJK | Markdown / JSON | 8–20 s/page CPU | AGPL-3.0 |

CPU latency is the operator's pain point — a 50-page paper on a
3-worker dev cluster is ≥5 minutes. None of these scale linearly
with cores; they are GPU-shaped tools.

### Hosted APIs — cost / quality tradeoff matrix

The operator picks a row, not a vendor. The framework supports
seven hosted backends across four cost / quality tiers; pricing is
per-page in USD, all batch / cache discounts available where the
vendor offers them.

| Tier | Backend (`pdf_extractor.backend=…`) | Tier knob (`options=…`) | Per-page cost | Latency | Image bytes? | Notes |
|---|---|---|---|---|---|---|
| **Cheap-OCR** | `mistral_ocr` | (none — single model) | **$0.001 batch / $0.002 std** | <1 s | ✅ | Pure OCR; cheapest by 10× ([Mistral OCR 3](https://mistral.ai/news/mistral-ocr-3)). |
| **Cheap-OCR** | `llamaparse` | `tier=fast` | $0.00125 | 5-15 s | ✅ (presigned) | No-AI text extract; competitive with Mistral on simple PDFs ([LlamaParse pricing](https://developers.llamaindex.ai/llamaparse/general/pricing/)). |
| **Balanced multimodal** | `gemini` | `model=gemini-2.5-flash` (default) | **~$0.003** | 2-5 s | ❌ | Native PDF, fast multimodal reasoning; the recommended balanced default ([Gemini docs](https://ai.google.dev/gemini-api/docs/document-processing), [pricing](https://ai.google.dev/gemini-api/docs/pricing)). |
| **Balanced multimodal** | `llamaparse` | `tier=cost_effective` (LlamaParse default) | $0.00375 | 5-15 s | ✅ (presigned) | LLM-assisted layout reasoning, RAG-tuned markdown. |
| **Premium multimodal** | `anthropic` | `model=claude-sonnet-4-5` (default) | ~$0.009 | 3-8 s | ❌ | Best Anthropic visual reasoning; prompt caching halves cost on re-runs ([Anthropic docs](https://platform.claude.com/docs/en/build-with-claude/pdf-support)). |
| **Premium multimodal** | `gemini` | `model=gemini-2.5-pro` | ~$0.010 | 5-15 s | ❌ | Premium multimodal quality, native PDF. |
| **Highest-fidelity** | `llamaparse` | `tier=agentic` | $0.0125 | 15-60 s | ✅ (presigned) | Sonnet-class agent walks the document; best semantic structure preservation. |
| **Highest-fidelity** | `llamaparse` | `tier=agentic_plus` | $0.056 | 30-120 s | ✅ (presigned) | Highest-fidelity tier; reserve for visually-dense docs the agentic tier missed. |

Other surveyed vendors that didn't make the lineup:

- **[Reducto](https://llms.reducto.ai/document-parser-comparison)** — ~$0.015/page entry. Best on RD-TableBench (90.2%) but enterprise-leaning (SOC 2, on-prem); overkill for our v1. Easy to add as another backend later if a corpus warrants it.

### Self-hosted deployments — tier 5 ("highest fidelity, no vendor cost")

| Tool | Strength | Output | CPU latency (A4 page) | License |
|---|---|---|---|---|
| **Marker** | General-purpose; equations via Texify; broad format coverage | Markdown / JSON / HTML | 5-15 s CPU, sub-1 s GPU | GPL-3.0 (behind `knowledge_marker` extra) |
| **Docling** | IBM Research; strong table model (TableFormer) | `DoclingDocument` → Markdown | 10-30 s CPU | MIT |
| **MinerU** | OpenDataLab; excellent on visually-dense / CJK | Markdown / JSON | 8-20 s CPU | AGPL-3.0 |

GPU-shaped tools — none scale linearly with cores. Operators run
them in production where they have the GPU budget; CPU is for
correctness validation only.

### Picking a row

- **Just bulk-ingest a known-clean corpus**: `mistral_ocr` (default).
- **Mixed corpus where you want LLM-assisted layout reasoning**: `gemini` with `model=gemini-2.5-flash` — the cost / quality sweet spot.
- **Visually-dense documents the cheap tiers miss**: `gemini` with `model=gemini-2.5-pro`, or `anthropic`.
- **Citation-graph + RAG-tuned markdown for academic literature**: `llamaparse` with `tier=cost_effective` or `agentic`.
- **Self-hosted, no vendor cost**: `docling` (default GPU tool), `mineru` (heaviest layouts), `marker` (broadest format support, GPL).

Per-document-type routing (Phase 4) eventually picks the right
tier per doc class — datasheets through `mistral_ocr`, complex
patents through `llamaparse:agentic`, etc. Until then, operators
pin one backend at deploy time and override per-call via the KB
tab's `extractor_override`.

For both, the operator only pays per-page when they choose to use
the hosted path. The default cluster config keeps the self-hosted
extractor selected; the hosted readers are explicit
opt-ins via the `pdf_extractor` config field.

## 6. Data model changes

### 6.1 New: `FigureRef`

Lives in `knowledge/models.py`:

```python
class FigureRef(BaseModel):
    """One figure / image / table-as-image extracted from a document.

    The reader emits these alongside the section text. The chunker
    forwards the IDs into ``Chunk.extra["figure_ids"]`` so retrievers
    and the agent's planner can fetch image bytes by URI.
    """

    model_config = ConfigDict(frozen=True)

    figure_id: str = Field(default_factory=lambda: f"fig_{uuid.uuid4().hex[:16]}")
    image_uri: str
    """Content-addressed URI returned by ``ImageStore.put`` —
    typically ``colony-image://<sha256>``. Resolves via the active
    ``ImageStore`` instance bound to ``RetrievalDeps``."""

    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    """``(x0, y0, x1, y1)`` in PDF user-space units, when the reader
    can supply them."""

    caption_hint: str = ""
    """Caption text the reader extracted from below the figure (or
    similar). Used as a hint for the captioner; not authoritative."""

    kind: Literal["figure", "table", "diagram", "equation", "other"] = "figure"

    extra: dict[str, Any] = Field(default_factory=dict)
```

### 6.2 Modified: `ParsedSection`

```python
class ParsedSection(BaseModel):
    section_path: str = ""
    heading: str = ""
    text: str
    citation: CitationSpan
    figures: tuple[FigureRef, ...] = Field(default_factory=tuple)   # NEW
    format: Literal["text", "markdown"] = "text"                    # NEW
    extra: dict[str, Any] = Field(default_factory=dict)
```

`format="markdown"` flags sections whose `text` carries Markdown the
chunker should preserve through chunking (don't split mid-table,
don't break equations across chunks). Existing readers default to
`"text"`; the new readers emit `"markdown"`.

### 6.3 Modified: `Chunk.extra` documented keys

| Key | Type | Set by | Meaning |
|---|---|---|---|
| `figure_ids` | `list[str]` | chunker | Figures referenced by this chunk's text. |
| `metadata_origin` | `str` | reader | Which reader produced the section (`"grobid"`, `"marker"`, `"mistral_ocr"`, …). |
| `bibliographic` | `dict` | metadata reader (GROBID) | Authors, affiliations, citations parsed from the doc header. |

### 6.4 New: `ImageStore`

Lives in `knowledge/stores/image.py`. Same shape as `VectorStore` —
ABC + two impls.

```python
class ImageStore(ABC):
    @abstractmethod
    async def put(self, payload: bytes, *, mime: str = "image/png") -> str:
        """Persist ``payload`` and return a content-addressed URI."""

    @abstractmethod
    async def get(self, uri: str) -> bytes | None:
        """Resolve a URI returned by ``put``."""

    @abstractmethod
    async def has(self, uri: str) -> bool: ...

    @abstractmethod
    async def delete(self, uri: str) -> bool: ...

    @abstractmethod
    async def stat(self, uri: str) -> dict[str, Any] | None:
        """Return ``{size, mime, created_at}`` for the image, or
        ``None`` if absent."""
```

`LocalFsImageStore` writes to `/mnt/shared/colony-images/<sha[:2]>/<sha>.{ext}`,
indexes mime + size in a sidecar JSON. Sharded by sha-prefix so a
single directory never grows past a few hundred entries.

`InMemoryImageStore` keeps everything in a `dict[str, bytes]` for
tests.

`RetrievalDeps` and `Ingestor` both gain an `image_store` field
(blueprint chain extended, same pattern as the vector store).

### 6.5 Modified: `RetrievalHit`

```python
class RetrievalHit(BaseModel):
    chunk: Chunk
    score: float = 0.0
    rank: int = 0
    explanation: str = ""
    figures: tuple[FigureRef, ...] = Field(default_factory=tuple)  # NEW
```

The `figures` are denormalised onto the hit so the agent can pull
images without a second round-trip. `KnowledgeRetrievalCapability`
populates them by joining `chunk.extra["figure_ids"]` against a
small per-source figure index (built at ingest time).

## 7. Reader interface

The existing `FormatReader` ABC already has the right shape — it
returns `Sequence[ParsedSection]`. Image-emitting readers stash
bytes via the `ImageStore` and reference them in `FigureRef`s
inside the section. **No new abstract method is needed** — readers
already have access to `RawDocument.metadata` and can pull the
`ImageStore` from a constructor injection.

```python
class MarkerPdfReader(FormatReader):
    handles = (KnowledgeFormat.PDF,)

    def __init__(self, *, deployment_handle, image_store):
        self._handle = deployment_handle  # Ray serving handle
        self._image_store = image_store

    async def read_async(self, document: RawDocument) -> Sequence[ParsedSection]:
        # 1. ship the bytes to MarkerExtractorDeployment.extract(...)
        result = await self._handle.extract(pdf_bytes=document.bytes_)
        # 2. for each figure in result.images, store via image_store.put(...)
        # 3. rewrite markdown image refs to colony-image:// URIs
        # 4. emit ParsedSection(text=md, figures=tuple(refs), format="markdown")
        ...
```

The `read()` synchronous path raises `NotImplementedError` for
deployment-backed readers; the registry already prefers
`read_async()` when available.

## 8. Serving deployments

Per operator decision #1, three sibling deployments under
`cluster/extractors/`:

```
cluster/extractors/
  __init__.py
  base.py              # PdfExtractorDeployment ABC
  marker_deployment.py
  docling_deployment.py
  mineru_deployment.py
```

Each exposes:

```python
@serving.endpoint
async def extract(
    self, *, pdf_bytes: bytes, options: ExtractOptions | None = None,
) -> ExtractResult:
    """Returns ExtractResult(markdown, figures=[FigureBlob, ...])."""
```

`FigureBlob = (image_bytes, mime, page, bbox, caption_hint, kind)`.
Reader code stores them via `ImageStore.put(...)` and converts to
`FigureRef`.

The hosted-API readers (`AnthropicPdfReader`,
`MistralOcrPdfReader`) **do not need their own deployment** — they
hit external endpoints directly through the existing
`AnthropicLLMDeployment` / a new lightweight `MistralOcrClient`.

### Cluster config selector

```yaml
# cluster_config.yaml
knowledge:
  pdf_extractor:
    backend: marker  # marker | docling | mineru | anthropic | mistral_ocr
    # Self-hosted backends:
    marker:
      replicas: 1
      gpus_per_replica: 0       # 1 in production
    docling: { replicas: 1 }
    mineru: { replicas: 1 }
    # Hosted-API backends:
    anthropic:
      model: claude-sonnet-4-6
      use_files_api: true       # avoid re-uploading repeated PDFs
      prompt_cache: true
    mistral_ocr:
      api_base: https://api.mistral.ai/v1
      batch: false              # set true for $0.001/page corpus runs
```

`PolymatheraClusterConfig.add_deployments_to_app` reads `knowledge.pdf_extractor.backend` and only instantiates the matching deployment class. The other two binaries don't get pulled into the image (Dockerfile keeps them behind extras).

## 9. Captioner deployment (Phase 3)

`AnthropicVisionDeployment` extends `AnthropicLLMDeployment`. New
endpoint:

```python
@serving.endpoint
async def caption_figure(
    self, *, image_uri: str, hint: str = "", style: CaptionStyle = "cps",
) -> str:
    """Return a structured caption for the figure at ``image_uri``.

    ``style="cps"`` uses a CPS-tuned prompt:
      describe the figure type, list any labeled components,
      identify any equations or units, transcribe any axis labels
      and legends, identify the apparent purpose.
    """
```

The ingestor calls `caption_figure` per extracted `FigureRef`, writes the caption back as a sibling chunk with `data_type="figure_caption"` and `chunk.extra["figure_id"]`. PlainQdrant text search then surfaces figures by description. No second collection.

## 10. Phase plan

### Phase 1+2 — Reader swap + figures in the data model (in progress)

**Goal**: a clean PDF → markdown-with-figure-refs pipeline behind a config switch, with images persisted to disk and round-tripped to the chat UI's KB tab.

| # | Task | Files | Owner | Status |
|---|---|---|---|---|
| 1.1 | Add `FigureRef` model + `figures` + `format` fields on `ParsedSection` | `knowledge/models.py` | claude | ✅ |
| 1.2 | Add `ImageStore` ABC + `InMemoryImageStore` + `LocalFsImageStore` (sharded layout, sidecar meta, atomic writes, `@blueprint`) | `knowledge/stores/image.py` (new) | claude | ✅ |
| 1.3 | Wire `image_store` into `RetrievalDeps` + `Ingestor` (blueprint chain); env-driven `KB_IMAGE_DIR` selector in `set_knowledge_deps` + `default_*_blueprint` factories | `knowledge/retrieval/base.py`, `knowledge/ingestion.py`, `knowledge/deps.py` | claude | ✅ |
| 1.4 | `PdfExtractorDeployment` ABC + `ExtractResult` / `FigureBlob` / `ExtractOptions` types (no Ray-serve deps in the ABC — concrete deployments apply `@serving.deployment` themselves) | `cluster/extractors/base.py` (new), `cluster/extractors/__init__.py` | claude | ✅ |
| 1.5 | `MarkerExtractorDeployment` (behind the `knowledge_marker` poetry extra per decision #6; lazy ML-model load on first ``extract``; PIL → PNG re-encode for figures; CPU + GPU autoscaling profiles; `@serving.deployment` on the class; `@serving.endpoint` on `extract`) | `cluster/extractors/marker_deployment.py` (new) | claude | ✅ |
| 1.6 | `DoclingExtractorDeployment` (default `knowledge` extra; lazy `DocumentConverter` build with table structure + image generation enabled; `DocumentStream` upload path so we never write the PDF to disk; figures re-encoded as PNG) | `cluster/extractors/docling_deployment.py` (new) | claude | ✅ |
| 1.7 | `MinerUExtractorDeployment` (default `knowledge` extra; lazy `magic_pdf` import; tempdir-per-call so concurrent extracts don't collide; `parse_method` toggle for OCR / TXT / auto) | `cluster/extractors/mineru_deployment.py` (new) | claude | ✅ |
| 1.8 | One generic `RemotePdfExtractorReader(backend=…, image_store=…)` instead of three near-identical reader classes — the deployment is the per-backend specialisation, the reader is backend-agnostic glue. Resolves the deployment handle lazily on first call (so the reader is picklable across the Ray boundary); `ExtractResult` dict-shape coercion via `model_validate` for cloudpickle round-trips; `<!-- page: N -->` markers split sections; figure refs propagate per-page | `knowledge/readers/remote_pdf.py` (new) | claude | ✅ |
| 1.9 | `AnthropicPdfReader` (hosted, uses the `anthropic` SDK directly via `messages.create` with a base64 document content block; `<!-- page: N -->` page markers in the prompt → one `ParsedSection` per marker; `figures=()` since Anthropic returns no image bytes; `prompt_cache` toggle; picklable api-key resolution) | `knowledge/readers/anthropic_pdf.py` (new) | claude | ✅ |
| 1.9b | `GeminiPdfReader` (hosted, `google-genai` SDK; ``model`` is the cost/quality tier knob — `gemini-2.5-flash` ≈ $0.003/page balanced default, `gemini-2.5-pro` ≈ $0.010/page premium; same page-marker prompt as Anthropic; `cached_content_name` for the 90% repeat-input discount; api-key resolves from `GOOGLE_API_KEY`; picklable) | `knowledge/readers/gemini_pdf.py` (new) | claude | ✅ |
| 1.9c | `LlamaParsePdfReader` (hosted, full LlamaParse v2 REST: upload → poll → result with `expand=markdown,images_content_metadata,metadata` → presigned-URL image fetch; `tier` knob (`fast` / `cost_effective` / `agentic` / `agentic_plus`); per-call timeout + poll-interval bounds; image-fetch failures degrade to skipped figures rather than failed document; picklable) | `knowledge/readers/llamaparse_pdf.py` (new) | claude | ✅ |
| 1.10 | `MistralOcrPdfReader` (hosted; full HTTP impl: Files API → signed URL → `/v1/ocr`; base64 image decode; markdown image-ref rewrite to `colony-image://`; picklable; no instance-level `httpx.AsyncClient`) | `knowledge/readers/mistral_ocr_pdf.py` (new) | claude | ✅ |
| 1.11 | `KnowledgeConfig` + `PdfExtractorConfig` Pydantic models (`@register_polymathera_config(path="knowledge")`); top-level `knowledge.pdf_extractor` YAML section with `backend`, free-form `options` dict (forwarded as backend kwargs), `replicas`, `num_gpus`; `add_deployments_to_app` sets `KB_PDF_EXTRACTOR` env var on every worker AND brings up the matching `*ExtractorDeployment` for self-hosted backends via the lazy registry; missing-library import errors logged + skipped (env var still applied); hosted backends are no-ops at deploy time (vendor-direct calls) | `knowledge/cluster_config.py` (new), `system.py` | claude | ✅ |
| 1.12 | `default_registry_with_pdf_extractor(backend=...)` factory in readers; env-driven `KB_PDF_EXTRACTOR` selector in `set_knowledge_deps` (auto-fallback when the selected backend is reserved-but-unimplemented) | `knowledge/readers/__init__.py`, `knowledge/deps.py` | claude | ✅ |
| 1.13 | GROBID demote: `GrobidPdfReader.mode="metadata_only"` skips body extraction and emits ONLY title + abstract sections with a structured `extra["bibliographic"]` payload (authors with affiliations, parsed reference list); `GrobidMetadataReader` is the operator-facing alias with header / citation consolidation enabled by default. Multi-reader pass (running GROBID-metadata alongside the body extractor) deferred to its own iteration — needs a clean Ingestor refactor | `knowledge/readers/grobid_pdf.py` | claude | ✅ |
| 1.14 | Chunker propagates `figure_ids` through `chunk.extra` (matches `colony-image://` URIs against `section.figures`); also forwards `metadata_origin` provenance; both `ProseChunker` + `CodeChunker`; empty-extra preserved for plain-text sections | `knowledge/chunking.py` | claude | ✅ |
| 1.15 | KB tab renders multimodal chunks: full markdown via `react-markdown` + `remark-gfm`, inline figure previews via `colony-image://<sha>` → `/api/v1/kb/images/<sha>` rewrite, expandable `<pre>` blocks for long fenced code, "📷 N" badge per chunk for figure-id count, `metadata_origin` badge for extractor provenance; backend exposes `GET /kb/images/{sha}` with `Cache-Control: immutable` (content-addressed, never changes), 400 on non-hex sha, 404 on missing | `web_ui/frontend/src/components/kb/ChunkMarkdown.tsx` (new), `KnowledgeBaseTab.tsx`, `web_ui/backend/routers/kb.py` | claude | ✅ |
| 1.16 | KB ingest endpoint accepts `extractor_override` so operator can A/B from the dashboard | `web_ui/backend/routers/kb.py` | claude | ✅ |
| 1.17 | Tests (76 new across iterations 1+2, all passing): `ImageStore` contract + sharded layout + idempotency + sidecar fallback + blueprint pickle, `MistralOcrPdfReader` end-to-end with mocked `httpx.MockTransport` + auth-header assertions + 3-call ordering + malformed-image tolerance + error paths, `AnthropicPdfReader` with monkeypatched `anthropic.AsyncAnthropic` + page-marker round-trip + cache-control toggle + custom-prompt + empty-response handling, chunker `figure_ids` propagation per-chunk + `metadata_origin` forwarding + plain-text empty-extra invariant, `MarkdownChunker` keeps fenced code / GFM tables / `$$...$$` / `\\[...\\]` atomic + oversized-block preserved + ingestor wiring, `deps.py` env-driven `KB_PDF_EXTRACTOR` selection + fallback for unknown / unimplemented backends, `/kb/images/{sha}` round-trip + 400 hex-only validation + 404 on miss + `Cache-Control: immutable` | `knowledge/tests/test_image_store.py`, `test_mistral_ocr_pdf.py`, `test_anthropic_pdf.py`, `test_markdown_chunker.py`, `test_chunker_figures.py`, `test_deps_pdf_extractor.py`, `web_ui/backend/routers/tests/test_kb.py` | claude | ✅ |
| 1.18 | Re-ingest the existing literature corpus with `KB_PDF_EXTRACTOR=mistral_ocr` (cheapest fast path) and confirm KB tab shows multi-line markdown with linked figures | manual / dashboard | operator | ⬜ |
| 1.19 | Iteration 3.1 — knowledge-stack knobs absorbed into typed `KnowledgeConfig`. New `QdrantConfig`, `GrobidConfig`, and `image_dir` field; `knowledge/deps.py`, `web_ui/backend/routers/kb.py`, `design_monorepo/capabilities.py` read via `get_component_or_default("knowledge", KnowledgeConfig)` instead of `os.environ`. `RUNTIME_ENV_PREFIXES` pruned to bootstrap (`POLYMATH`, `RAY_`, `REDIS_`) + LLM-provider secrets; `KB_*`, `QDRANT_*`, `GROBID_*` removed from the proxy allowlist and from `docker-compose.yml`. `add_deployments_to_app` no longer writes to `os.environ`. Supersedes the env-var paths in rows 1.3, 1.11, 1.12 — every Ray worker re-reads the YAML via `POLYMATHERA_CONFIG` and resolves the typed config in its own process | `knowledge/cluster_config.py`, `knowledge/deps.py`, `web_ui/backend/routers/kb.py`, `design_monorepo/capabilities.py`, `distributed/ray_utils/serving/proxy.py`, `cli/deploy/docker/docker-compose.yml`, `cli/deploy/.env.template`, `configs/example.yaml` | claude | ✅ |

### Iteration 1 wrap-up (2026-05-08)

What landed in this iteration: rows **1.1, 1.2, 1.3, 1.4, 1.10, 1.12, 1.14, 1.16, 1.17**. Net new SLOC ≈ 1,500 (production) + 700 (tests). 47 new tests, 353 in the relevant suites green, no regressions.

What an operator can now do without writing code:

```bash
# Iteration 1 shipped an env-var-driven path on every container that
# runs ingestion (ray-head, ray-workers, dashboard). Iteration 3.1
# replaced this with the typed KnowledgeConfig (see the iteration 3.1
# wrap-up below); the snippet is preserved verbatim as historical
# context.
export KB_PDF_EXTRACTOR=mistral_ocr
export MISTRAL_API_KEY=<key>
export KB_IMAGE_DIR=/mnt/shared/colony-images   # optional; falls back to in-mem
```

After redeploy, `set_knowledge_deps()` resolves a registry whose
PDF reader is `MistralOcrPdfReader` wired to a `LocalFsImageStore`
under `KB_IMAGE_DIR`. Every PDF ingest goes through Mistral OCR;
figures land in `/mnt/shared/colony-images/<sha[:2]>/<sha>.{ext}`;
`Chunk.extra["figure_ids"]` lists the figures each chunk
references. The KB tab renders chunk text correctly (existing
markdown renderer); inline figure previews wait for row 1.15.

What is intentionally NOT in this iteration:
- Self-hosted Marker / Docling / MinerU deployments (rows 1.5-1.8) — each needs its own Docker variant + library.
- `AnthropicPdfReader` (row 1.9) — Mistral covers the hosted path; Anthropic adds a second hosted backend in iteration 2.
- KB tab figure rendering (row 1.15) — frontend work; chunks already carry `figure_ids` so the renderer pickup is mechanical.
- `cluster_config.yaml` integration (row 1.11) — env var path is sufficient for now; YAML wiring lands with the self-hosted deployments.
- GROBID metadata-only mode (row 1.13) — GROBID still runs as today; the new readers shadow it via the registry's last-write-wins rule.

### Iteration 2 wrap-up (2026-05-08)

What landed: rows **1.9, 1.15** plus operator decision **#7**'s `MarkdownChunker` implementation. Net new SLOC ≈ 1,300 (production: AnthropicPdfReader, MarkdownChunker, ChunkMarkdown.tsx, /kb/images endpoint) + 950 (tests). 29 new tests this iteration, 76 cumulative across 1+2; 382 passing in the relevant suites with zero regressions. Frontend type-clean.

What an operator can now do without writing code:

```bash
# Two hosted backends are live; pick one with KB_PDF_EXTRACTOR.
# Both rely on the same KB_IMAGE_DIR for figure persistence (Mistral
# emits image bytes; Anthropic doesn't, but the env var is harmless
# either way).
export KB_PDF_EXTRACTOR=anthropic     # or mistral_ocr
export ANTHROPIC_API_KEY=<key>        # or MISTRAL_API_KEY=<key>
export KB_IMAGE_DIR=/mnt/shared/colony-images
```

After redeploy, the KB tab renders each chunk as full markdown with:
- LaTeX math (`$inline$` / `$$display$$`) shown verbatim — picked up by any downstream KaTeX integration; today renders as code-styled text.
- GFM tables rendered natively.
- Fenced code blocks shown in a max-height collapsible `<pre>` with an inline expand toggle.
- Figure references resolved through `/api/v1/kb/images/{sha}` so images appear inline (Mistral OCR path); the Anthropic path describes figures in prose since the API returns no image bytes.
- A "📷 N" badge per chunk when `figure_ids` is non-empty.
- A `metadata_origin` badge naming the extractor that produced the chunk (`mistral_ocr` / `anthropic` / …).

`MarkdownChunker` keeps fenced code, GFM tables, and display math (`$$...$$`, `\\[...\\]`) atomic — a 200-line function emerges as one (oversized) chunk rather than being torn apart. The `Ingestor` picks the chunker per section: `format="markdown"` → `MarkdownChunker`; `format="text"` → `ProseChunker` (existing behaviour preserved).

**Acceptance**: opening any chunk in the KB tab for a paper shows properly-formatted markdown (LaTeX equations as `$...$`, tables as GFM tables, figures as inline previews resolved through `/api/v1/kb/images/<sha>`).

What is intentionally NOT in this iteration (deferred to iteration 3):
- Self-hosted Marker / Docling / MinerU deployments (rows 1.5-1.8) — each needs its own Docker variant + library; not blocking on a CPU dev box where they would be too slow to use anyway.
- `cluster_config.yaml` integration (row 1.11) — env vars cover the live path; YAML wiring lands with the self-hosted deployments.
- GROBID demote (row 1.13) — GROBID still runs as today; the new readers already shadow it via the registry's last-write-wins rule, so behaviour is correct. The structural rewrite into a sibling metadata-only reader needs a multi-reader Ingestor pass and is a clean slice of work on its own.

### Iteration 3 wrap-up (2026-05-08)

What landed: rows **1.5, 1.6, 1.7, 1.8, 1.9b (NEW), 1.9c (NEW), 1.11, 1.13**, plus the `knowledge_marker` poetry extra and the cost/quality matrix rewrite of §5. Net new SLOC ≈ 2,800 (production: 4 readers + 3 deployments + cluster config + GROBID demote) + 1,500 (tests). 47 new tests this iteration, 123 cumulative across 1+2+3; **428 passing** in the relevant suites with zero regressions. Frontend type-clean.

The two **bonus rows (1.9b / 1.9c)** were added in response to the operator's "we need cost/quality tradeoffs, not just cheapest" point — Gemini covers the balanced-multimodal price/quality sweet spot at $0.003/page, LlamaParse covers the highest-fidelity end at $0.05/page agentic-plus. The operator now has **seven hosted backend × tier combinations** plus three self-hosted deployments to A/B against one another via the `pdf_extractor.backend` YAML field and `/kb/ingest`'s `extractor_override`.

> **Note (superseded by iteration 3.1):** iteration 3 originally
> shipped a dual env-var + YAML path with `KB_PDF_EXTRACTOR` writes
> on the driver and a wide proxy allowlist forwarding the knob to
> workers. Iteration 3.1 deletes that workaround — knowledge-stack
> knobs are typed fields on `KnowledgeConfig` and reach workers via
> `POLYMATHERA_CONFIG`. The runbook below reflects iteration 3.1.

### Iteration 3.1 wrap-up (2026-05-08)

What landed: knowledge-stack knobs absorbed into typed
`ConfigComponent`s. New `QdrantConfig`, `GrobidConfig`, and
`image_dir` field on `KnowledgeConfig`. `knowledge/deps.py`,
`web_ui/backend/routers/kb.py`, `design_monorepo/capabilities.py`
all read via `get_component_or_default("knowledge",
KnowledgeConfig)` instead of `os.environ.get`. Proxy
`RUNTIME_ENV_PREFIXES` pruned to bootstrap (`POLYMATH`, `RAY_`,
`REDIS_`) plus secrets (LLM provider key prefixes); `KB_`,
`QDRANT_`, `GROBID_` removed. Compose env block stripped of the
config-knob passthroughs; `.env.template` carries only secrets.
`example.yaml` carries the docker-stack URLs as canonical YAML
defaults. Regression tests rewritten to lock the new contract.
**320 passing** (knowledge + serving + cluster + dashboard
routers); zero regressions outside live-Redis-only tests.

What an operator can now do without writing code:

```yaml
# operator.yaml
knowledge:
  pdf_extractor:
    backend: gemini             # cost/quality tier 2
    options:
      model: gemini-2.5-flash   # ~$0.003/page balanced default
  image_dir: /mnt/shared/colony-images
  qdrant:
    url: http://qdrant:6333
    collection: colony_knowledge
  grobid:
    url: http://grobid:8070

# Or, for the highest-fidelity tier:
#   pdf_extractor:
#     backend: llamaparse
#     options:
#       tier: agentic_plus

# Or, for self-hosted:
#   pdf_extractor:
#     backend: docling
#     replicas: 2
#     num_gpus: 1
```

`add_deployments_to_app` brings up the matching `*ExtractorDeployment`
for self-hosted backends; hosted backends are no-ops at deploy
time. Every Ray worker re-reads the same YAML via its own
`ConfigurationManager` (loaded from `POLYMATHERA_CONFIG`), so
driver + actors agree on the configured backends without env-var
passthrough.

`extractor_override` on `/kb/ingest` continues to accept the same
backend names — operators A/B between Mistral / Gemini / Anthropic
/ LlamaParse on a single doc from the dashboard without
redeploying.

What is NOT in this iteration:
- Multi-reader Ingestor pass (running GROBID metadata-only ALONGSIDE the layout-aware body extractor). Needs a small Ingestor refactor and an explicit cross-reader merge contract; deferred until we have a corpus where citation-graph + layout-aware-body together is genuinely worth it.
- Phase 3 (figure captioning), Phase 4 (per-document-type routing), Phase 5 (ColPali), Phase 6 (multimodal at synthesis). Each is its own design iteration tracked separately in §10.

**Operator runbook** (the actual sequence to test iteration 3.1):

1. **Regenerate `poetry.lock`** — already done in iteration 3; `pyproject.toml` gained `docling`, `magic-pdf`, `google-genai` in the `knowledge` extra and the lock was refreshed. If you pull a future commit that touches deps, run `poetry lock --no-update` before rebuilding.

2. **Set API keys for the backend(s) you want to use.** API keys are secrets, so they live in `.env` (compose forwards them as env vars) or your host shell, not in YAML. Pick the rows you need:

   ```bash
   # In .env (recommended) or exported in your host shell.
   ANTHROPIC_API_KEY=...
   OPENAI_API_KEY=...
   GOOGLE_API_KEY=...
   MISTRAL_API_KEY=...
   LLAMA_CLOUD_API_KEY=...
   ```

3. **Pick a backend in YAML.** Edit your operator config (or a copy) under the top-level `knowledge.pdf_extractor` key:

   ```yaml
   knowledge:
     pdf_extractor:
       backend: gemini
       options:
         model: gemini-2.5-flash
   ```

   The default `configs/example.yaml` already has docker-stack URLs for `qdrant` / `grobid` / `image_dir`. Override anything else you need.

4. **`colony-env down && colony-env up --workers 3 --config operator.yaml`** — rebuilds the image (poetry installs the new deps from the refreshed lock), brings up the stack, mounts the YAML at `/mnt/shared/config.yaml`. Every container's `POLYMATHERA_CONFIG` points at it, so driver + workers + dashboard all resolve the same `KnowledgeConfig`.

5. **Re-ingest the literature corpus from the chat UI.** The chunks land in Qdrant via the chosen reader; the KB tab renders multi-line markdown with inline figure previews (Mistral / LlamaParse) or prose figure descriptions (Anthropic / Gemini).

6. **Operator A/B from the dashboard.** `/kb/ingest` accepts `extractor_override` to swap backends per call without redeploying — flip Mistral → Gemini → LlamaParse on the same PDF and compare.

**Special cases:**

- **Marker (GPL-3.0)** is intentionally NOT poetry-managed because its `openai <2.0` pin conflicts with the colony's `openai ^2.21.0`. Operators who specifically want Marker run `docker exec colony-ray-head pip install --no-deps marker-pdf` (and the same on workers) inside the live containers, then set `pdf_extractor.backend: marker` in YAML. The deployment lazy-imports `marker`, so its absence is a clean error rather than a startup crash.
- **Self-hosted backends on CPU** are slow (5-30 s/page). The ML libraries (`docling`, `magic-pdf`) load on first call and download model weights — first ingest of a doc takes minutes. Subsequent ingests are normal-speed. Use the hosted backends for fast iteration; reserve self-hosted for correctness validation against your specific corpus.
- **YAML didn't load?** If a worker resolves the default `KnowledgeConfig` (in-memory vector store, `mistral_ocr` backend) when you expected your YAML overrides, check that `POLYMATHERA_CONFIG` points at the file inside the container and that `colony-env up --config` was passed.

### Phase 3 — Caption-first multimodality

| # | Task | Files | Owner | Status |
|---|---|---|---|---|
| 3.1 | `AnthropicVisionDeployment` adds `caption_figure(image_uri, hint, style)` | `cluster/anthropic_deployment.py` (extend) | claude | ⬜ |
| 3.2 | Ingestor optionally captions every `FigureRef` after `image_store.put` | `knowledge/ingestion.py` | claude | ⬜ |
| 3.3 | Caption stored as `Chunk(data_type="figure_caption", source=<doc>, extra={figure_id})` | `knowledge/ingestion.py` | claude | ⬜ |
| 3.4 | `RetrievalHit.figures` populated by joining `chunk.extra["figure_ids"]` against the figure index | `knowledge/retrieval/*.py` | claude | ⬜ |
| 3.5 | Caption prompt template + style registry (`cps`, `paper`, `datasheet`, `patent`) | `cluster/anthropic_deployment.py` | claude | ⬜ |
| 3.6 | Dashboard shows caption next to figure preview in KB tab | `web_ui/frontend/.../KnowledgeBaseTab.tsx` | claude | ⬜ |
| 3.7 | Tests: caption round-trip with mocked deployment, figure-search recall on a 5-doc fixture | `knowledge/tests/test_captioner.py` | claude | ⬜ |

### Phase 4 — Per-document-type routing (deferred)

| # | Task | Status |
|---|---|---|
| 4.1 | `IngestionRouter` chooses an extractor per `data_type` (papers→Marker, datasheets→Marker+Camelot, patents→Marker+xref-preserver, standards→Marker+clause splitter). | ⬜ |
| 4.2 | `data_type` detection heuristic at ingest time (path-based + reader-emitted hints). | ⬜ |
| 4.3 | Datasheet table specialist: pull tables into a structured store keyed by datasheet section. | ⬜ |
| 4.4 | Patent xref preserver: keep `Fig. 3 / element 42` semantics intact through chunking. | ⬜ |

Defer until Phase 1+2 ship and we have throughput data.

### Phase 5 — ColPali / page-image multi-vector retrieval (deferred, optional)

| # | Task | Status |
|---|---|---|
| 5.1 | Decision: only proceed if caption-first measurably underperforms on visually-dense subcorpora (define a small eval set). | ⬜ |
| 5.2 | Second `VectorStore` collection in Qdrant multi-vector mode. | ⬜ |
| 5.3 | `ColPaliEmbedder` wraps a hosted endpoint (or vLLM-served ColQwen2.5 in production). | ⬜ |
| 5.4 | `RetrievalAdapter` fuses caption-first text scores with page-level ColPali scores via reciprocal rank fusion. | ⬜ |

### Phase 6 — Multimodal at synthesis (separate workstream)

| # | Task | Status |
|---|---|---|
| 6.1 | Agent's planner can pass `RetrievalHit.figures` image URIs to a vision-capable model in its loop. | ⬜ |
| 6.2 | `VisionDeployment` reused as the agent's vision system (different prompt template, same backend). | ⬜ |
| 6.3 | Sim/visualization-result understanding: agent runs visualization code → captures plot bytes → captions via `VisionDeployment` → reasons over caption + numeric metrics. | ⬜ |

## 11. Open questions

1. **Marker license**. **Resolved 2026-05-08 (decision #6).** Marker stays behind a `knowledge_marker` poetry extra; not in the default `colony:local` image. Docling (MIT) and MinerU (AGPL-3.0) ship by default.
2. **Chunker boundary preservation for markdown**. **Direction locked 2026-05-08 (decision #7).** Subclass `ProseChunker` into a `MarkdownChunker` that treats fenced code, GFM tables, and `$...$` math as atomic units. Implementation deferred to iteration 2 — Mistral OCR emits one section per page so markdown blocks stay within section boundaries today, and the paragraph-aware ProseChunker is sufficient for the iteration-1 smoke test (verified by `test_chunker_figures.py::test_prose_chunker_propagates_figure_ids_per_chunk`).
3. **Mistral OCR rate limits**. The cookbook claims ~2,000 pages/min
   but the public tier has unspecified rate caps. Phase 1 stays
   sequential per-doc and parallel across docs; revisit if
   throughput hits a ceiling.
4. **Image store eviction policy**. None for v1 — local FS, operator
   manages. Phase 4 may want size-bounded LRU.
5. **Figure cross-references**. Marker / Mistral emit markdown like
   `![Fig. 3](image_xx.png)`. The reader needs to map "Fig. 3" back
   to the `figure_id` so chunks that mention "Fig. 3" can populate
   `figure_ids`. Plan: maintain a (label → `figure_id`) table per
   document during the section walk.

## 12. Tracking discipline

- **Update the checklists in §10 as work lands.** A row is ⬜
  → 🟨 (PR open) → ✅ (merged + on a deployed cluster).
- **Each PR references this doc** in the commit body so the chain is
  navigable.
- **Acceptance criteria are concrete** — Phase 1+2 is done when
  the operator can re-ingest the literature corpus and the KB tab
  shows formatted markdown with linked figure previews. Anything
  short of that is not "done."

---

## Sources

- [Anthropic — PDF support docs](https://platform.claude.com/docs/en/build-with-claude/pdf-support)
- [Mistral — OCR 3 announcement](https://mistral.ai/news/mistral-ocr-3)
- [LlamaIndex — LlamaParse pricing](https://www.llamaindex.ai/pricing)
- [Reducto — document parser comparison](https://llms.reducto.ai/document-parser-comparison)
- [Firecrawl — best PDF parsers 2026](https://www.firecrawl.dev/blog/best-pdf-parsers)
- [Labellerr — Mistral OCR vs alternatives](https://www.labellerr.com/blog/mistralocr-did-it-do-what-it-claim/)
