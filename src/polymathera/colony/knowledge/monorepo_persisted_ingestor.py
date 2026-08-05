"""``MonorepoPersistedIngestor`` — :class:`Ingestor` wrapper that
persists per-source extraction outputs as a sidecar next to the source
file in a design monorepo.

For each PDF input at ``<dir>/<stem>.pdf`` the wrapper maintains a
sidecar directory ``<dir>/.ingested/<stem>/``:

- ``extracted.md`` — concatenated reader markdown across sections,
  with ``<!-- page: N -->`` separators between sections. Plain git
  (text). Users can read or edit it. Image references are RELATIVE
  paths into the sidecar's ``images/`` directory, so the committed
  markdown renders anywhere the repo is cloned.
- ``images/<sha256>.<ext>`` — figure bytes referenced by
  ``extracted.md``, copied out of the process's :class:`ImageStore`
  at extraction time. Content-addressed filenames match the store's
  own URIs, so round-trips are stable.
- ``ingestion.json`` — Pydantic-validated :class:`SidecarManifest`:
  pdf_sha256, extracted_md_sha256 (content hash of ``extracted.md``
  as written — the user-edit detector), extractor backend label,
  extracted_at timestamp, section_count, page_count, image_count,
  source_uri. Plain git (small JSON).

On the skip path the wrapper re-ingests from the sidecar: relative
image refs are rehydrated into the process's :class:`ImageStore`
(idempotent — content-addressed) and rewritten back to
``colony-image://`` URIs so the downstream pipeline sees exactly what
a fresh extraction would have produced. A sidecar whose image files
are missing is treated as corrupt and falls back to full
re-extraction (self-healing cache).

The wrapper avoids re-paying the reader's extraction cost on
re-ingest of an unchanged PDF: if ``ingestion.json``'s ``pdf_sha256``
matches the current PDF (and its ``extractor`` matches the configured
backend), the wrapper skips the reader and feeds the
on-disk ``extracted.md`` directly into :meth:`Ingestor.ingest_text`
for chunking + embedding. When ``extracted.md``'s content hash
differs from the manifest's ``extracted_md_sha256`` (user-edited
markdown), the wrapper trusts the edit and ingests the edited
markdown — chunking + embedding re-run; the reader does not.
(Content hashes, not mtimes: git checkout order makes mtimes
meaningless on fresh clones.)

For non-PDF inputs (markdown, source code, plain text, etc.) the
wrapper delegates to :meth:`Ingestor.ingest_file` directly — the
underlying file already IS the readable artifact, a sidecar would
duplicate it.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .formats import detect_format
from .ingestion import Ingestor
from .models import (
    CorpusTier,
    IngestionPolicy,
    IngestionRecord,
    IngestionStatus,
    KnowledgeFormat,
    ParsedSection,
    RawDocument,
)
from .readers.base import FormatReaderError, ReaderRegistry
from .stores.image import ImageStore, ext_for_mime, mime_for_ext


logger = logging.getLogger(__name__)


SIDECAR_DIRNAME = ".ingested"
EXTRACTED_MD_NAME = "extracted.md"
INGESTION_JSON_NAME = "ingestion.json"
IMAGES_DIRNAME = "images"

# Separator the wrapper emits between sections in ``extracted.md``. The
# same form the multimodal readers already use for page boundaries, so
# round-trip parses cleanly through the markdown reader on the skip
# path.
SECTION_SEPARATOR = "\n\n<!-- section -->\n\n"

_SHA256_BLOCKSIZE = 1 << 20

#: ``colony-image://<sha256-hex>`` URIs as multimodal readers emit them.
_COLONY_IMAGE_URI_RE = re.compile(r"colony-image://([0-9a-f]{64})")

#: Relative sidecar refs as :func:`_externalize_figures` writes them —
#: matched only inside markdown link/image parentheses so prose that
#: merely mentions ``images/`` is untouched.
_RELATIVE_IMAGE_REF_RE = re.compile(
    r"(?<=\()" + IMAGES_DIRNAME + r"/([0-9a-f]{64})(\.[A-Za-z0-9]+)(?=\))"
)


class SidecarImageError(RuntimeError):
    """A sidecar figure operation could not complete — bytes missing
    from the :class:`ImageStore` at externalize time, or an image file
    missing from the sidecar at rehydrate time."""


class SidecarManifest(BaseModel):
    """Schema for ``<sidecar>/ingestion.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    source_uri: str
    pdf_sha256: str
    extractor: str = ""
    """Label of the reader / pipeline that produced ``extracted.md``.
    Free-form (``"anthropic"``, ``"mistral_ocr"``, ``"grobid"``, ...);
    compared against the wrapper's configured ``extractor_label`` by
    :meth:`MonorepoPersistedIngestor._should_skip_extraction` — a
    mismatch invalidates the skip-cache so a backend switch re-pays
    for quality instead of serving the old backend's extraction."""
    extracted_at: str
    """ISO-8601 UTC timestamp."""
    section_count: int
    page_count: int = 0
    """Best-effort; sourced from section.extra['page'] when present."""
    extracted_md_sha256: str = ""
    """Content hash of ``extracted.md`` exactly as the wrapper wrote
    it. The user-edit detector: a differing hash on a later ingest
    means a human changed the markdown and the edit is trusted.
    Empty (pre-content-hash sidecars) means unknown provenance — the
    wrapper re-extracts rather than trust it."""
    image_count: int = 0
    """Number of figure files persisted under ``images/``."""


class MonorepoPersistedIngestor:
    """:class:`Ingestor` wrapper with ``.ingested/`` sidecar persistence
    for PDF inputs. See the module docstring for the persistence
    contract.
    """

    def __init__(
        self,
        ingestor: Ingestor,
        readers: ReaderRegistry,
        *,
        extractor_label: str = "",
    ) -> None:
        self._ingestor = ingestor
        self._readers = readers
        self._extractor_label = extractor_label

    async def ingest_file(
        self,
        path: str | Path,
        *,
        tier: CorpusTier = CorpusTier.UNTIERED,
        data_type_override: str | None = None,
        source_uri: str | None = None,
        policy: IngestionPolicy = IngestionPolicy.SKIP_IF_PRESENT,
    ) -> IngestionRecord:
        """Ingest a file with sidecar persistence when the input is a PDF.

        Returns the underlying :class:`IngestionRecord`. The sidecar
        write happens before chunking/embedding so the markdown + metadata
        are persisted even if downstream steps fail.
        """

        path_obj = Path(path)
        if not path_obj.is_file():
            return _fail_record(
                source_uri=source_uri or path_obj.as_uri(),
                error=f"File not found: {path_obj}",
            )

        # Non-PDF: delegate. The underlying file IS the readable artifact;
        # a sidecar would duplicate it.
        if path_obj.suffix.lower() != ".pdf":
            return await self._ingestor.ingest_file(
                path_obj,
                tier=tier,
                data_type_override=data_type_override,
                source_uri=source_uri,
                policy=policy,
            )

        canonical_uri = source_uri or path_obj.as_uri()
        sidecar_dir = path_obj.parent / SIDECAR_DIRNAME / path_obj.stem
        extracted_md_path = sidecar_dir / EXTRACTED_MD_NAME
        manifest_path = sidecar_dir / INGESTION_JSON_NAME

        pdf_sha256 = _sha256_file(path_obj)

        # Decide whether to skip the reader step entirely.
        skip_extraction, skip_reason = self._should_skip_extraction(
            pdf_sha256=pdf_sha256,
            extracted_md_path=extracted_md_path,
            manifest_path=manifest_path,
        )

        if skip_extraction:
            sidecar_md = extracted_md_path.read_text(encoding="utf-8")
            try:
                rehydrated_md = await _internalize_figures(
                    sidecar_md, sidecar_dir, self._ingestor.image_store,
                )
            except SidecarImageError as exc:
                # Corrupt or unusable sidecar (missing image files, no
                # store to rehydrate into): self-heal by re-extracting
                # instead of ingesting markdown with dangling refs.
                logger.warning(
                    "MonorepoPersistedIngestor: sidecar for %s is not "
                    "usable (%s) — falling back to full re-extraction.",
                    path_obj, exc,
                )
            else:
                logger.info(
                    "MonorepoPersistedIngestor: skipping extraction for "
                    "%s — %s",
                    path_obj, skip_reason,
                )
                return await self._ingestor.ingest_text(
                    rehydrated_md,
                    source_uri=canonical_uri,
                    fmt=KnowledgeFormat.MARKDOWN,
                    tier=tier,
                    data_type_override=data_type_override,
                    policy=policy,
                )

        # Full extraction: run the PDF reader, write sidecar, then
        # chunk + embed via ingest_text (cheap downstream).
        try:
            sections = await self._run_readers(path_obj, canonical_uri)
        except FormatReaderError as exc:
            return _fail_record(
                source_uri=canonical_uri,
                error=f"reader rejected PDF: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "MonorepoPersistedIngestor: reader failed on %s", path_obj,
            )
            return _fail_record(
                source_uri=canonical_uri,
                error=f"reader failed: {exc}",
            )

        if not sections:
            return _fail_record(
                source_uri=canonical_uri,
                error="reader returned no sections",
            )

        extracted_md = _sections_to_markdown(sections)

        # Persist figure bytes into the sidecar and relativize the
        # markdown's image refs BEFORE writing anything — a sidecar
        # with dangling ``colony-image://`` refs must never land in
        # git. The ORIGINAL markdown (store URIs) feeds the pipeline
        # below, matching what a sidecar-less ingest would see.
        try:
            sidecar_md, image_count = await _externalize_figures(
                extracted_md, sidecar_dir, self._ingestor.image_store,
            )
        except SidecarImageError as exc:
            return _fail_record(source_uri=canonical_uri, error=str(exc))

        manifest = SidecarManifest(
            source_uri=canonical_uri,
            pdf_sha256=pdf_sha256,
            extractor=self._extractor_label,
            extracted_at=datetime.now(timezone.utc).isoformat(),
            section_count=len(sections),
            page_count=_count_pages(sections),
            extracted_md_sha256=_sha256_bytes(sidecar_md.encode("utf-8")),
            image_count=image_count,
        )
        self._write_sidecar(
            sidecar_dir=sidecar_dir,
            extracted_md=sidecar_md,
            manifest=manifest,
        )

        return await self._ingestor.ingest_text(
            extracted_md,
            source_uri=canonical_uri,
            fmt=KnowledgeFormat.MARKDOWN,
            tier=tier,
            data_type_override=data_type_override,
            policy=policy,
        )

    # ---- Internals --------------------------------------------------------

    def _should_skip_extraction(
        self,
        *,
        pdf_sha256: str,
        extracted_md_path: Path,
        manifest_path: Path,
    ) -> tuple[bool, str]:
        """Return ``(skip, reason)``. Content-hash based throughout —
        mtimes are meaningless across git clones (checkout writes
        files in arbitrary order).

        Skip when either:
        - ``extracted.md``'s content hash differs from the manifest's
          ``extracted_md_sha256`` (user-edited markdown — the human's
          content wins over every other signal), or
        - manifest.pdf_sha256 matches the current PDF (cache hit) AND
          manifest.extractor matches this wrapper's configured
          ``extractor_label`` — a sidecar produced by a different
          (or unrecorded) backend is re-extracted so switching the
          configured extractor re-pays for quality instead of
          silently serving the old backend's output.

        Re-extract otherwise, including when the manifest is missing,
        unreadable, or predates content hashing (unknown provenance).
        """

        if not extracted_md_path.is_file():
            return False, "no sidecar"

        if not manifest_path.is_file():
            return False, "no manifest"

        try:
            manifest = SidecarManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8"),
            )
        except Exception:  # noqa: BLE001 — corrupt JSON / schema drift
            return False, "manifest unreadable"

        if not manifest.extracted_md_sha256:
            return False, (
                "manifest predates content hashing — unknown provenance"
            )

        try:
            current_md_sha256 = _sha256_file(extracted_md_path)
        except OSError:
            return False, "extracted.md unreadable"

        # User-edited markdown takes precedence — trust the edit,
        # even over an extractor-label or PDF change.
        if current_md_sha256 != manifest.extracted_md_sha256:
            return True, "extracted.md edited (content hash differs)"

        if (
            self._extractor_label
            and manifest.extractor != self._extractor_label
        ):
            return False, (
                f"extractor changed: "
                f"{manifest.extractor or '<unrecorded>'} -> "
                f"{self._extractor_label}"
            )

        if manifest.pdf_sha256 == pdf_sha256:
            return True, "pdf_sha256 unchanged"

        return False, "pdf_sha256 changed"

    async def _run_readers(
        self, pdf_path: Path, source_uri: str,
    ) -> list[ParsedSection]:
        """Run the configured PDF readers and return their sections.

        Mirrors the per-reader try/skip pattern in
        :meth:`Ingestor.ingest_document` so a transient reader failure
        on one backend doesn't poison the whole extraction.
        """

        payload = pdf_path.read_bytes()
        fmt = detect_format(path=pdf_path, payload=payload)
        document = RawDocument(
            source_uri=source_uri,
            detected_format=fmt,
            payload=payload,
            metadata={"path": str(pdf_path), "size_bytes": len(payload)},
        )
        readers = self._readers.readers_for(fmt)
        if not readers:
            raise FormatReaderError(
                f"No reader registered for format {fmt.value}.",
            )

        sections: list[ParsedSection] = []
        last_error: Exception | None = None
        for reader in readers:
            try:
                reader_sections = await reader.read_async(document)
            except Exception as exc:  # noqa: BLE001
                # A REQUIRED reader (the operator-configured body
                # extractor) failing fails the whole file — accepting
                # the surviving free readers' sections would write a
                # degraded sidecar whose pdf_sha256 then blocks the
                # paid extraction from ever being retried.
                if self._readers.is_required(fmt, reader):
                    logger.error(
                        "MonorepoPersistedIngestor: required reader %s "
                        "failed on %s",
                        type(reader).__name__, source_uri, exc_info=exc,
                    )
                    raise FormatReaderError(
                        f"required reader {type(reader).__name__} "
                        f"failed: {exc}. File NOT ingested and no "
                        f"sidecar written (no fallback to other "
                        f"readers); re-run the ingest once the "
                        f"backend is restored.",
                    ) from exc
                last_error = exc
                if isinstance(exc, FormatReaderError):
                    logger.warning(
                        "MonorepoPersistedIngestor: reader %s rejected "
                        "%s (%s)",
                        type(reader).__name__, source_uri, exc,
                    )
                else:
                    logger.exception(
                        "MonorepoPersistedIngestor: reader %s failed on %s",
                        type(reader).__name__, source_uri,
                    )
                continue
            sections.extend(reader_sections)

        if not sections and last_error is not None:
            if isinstance(last_error, FormatReaderError):
                raise last_error
            raise FormatReaderError(f"all readers failed: {last_error}")
        return sections

    def _write_sidecar(
        self,
        *,
        sidecar_dir: Path,
        extracted_md: str,
        manifest: SidecarManifest,
    ) -> None:
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        md_path = sidecar_dir / EXTRACTED_MD_NAME
        json_path = sidecar_dir / INGESTION_JSON_NAME
        md_path.write_text(extracted_md, encoding="utf-8")
        json_path.write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        # Observability: the sidecar persistence is the operator-
        # visible artifact of a successful ingest, but until this
        # log was added the substance was unobservable from logs.
        logger.info(
            "MonorepoPersistedIngestor: sidecar persisted "
            "source_uri=%s extractor=%s sections=%d pages=%d "
            "images=%d md_bytes=%d",
            manifest.source_uri,
            manifest.extractor or "<unknown>",
            manifest.section_count,
            manifest.page_count,
            manifest.image_count,
            len(extracted_md),
        )


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(_SHA256_BLOCKSIZE)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def _externalize_figures(
    markdown: str,
    sidecar_dir: Path,
    image_store: ImageStore | None,
) -> tuple[str, int]:
    """Copy every ``colony-image://`` figure referenced by ``markdown``
    into ``<sidecar_dir>/images/`` and rewrite the refs to relative
    paths. Returns ``(rewritten_markdown, image_count)``.

    Raises :class:`SidecarImageError` when any referenced bytes cannot
    be fetched (store unavailable or bytes missing) — a sidecar with
    dangling image refs must never be written; the 2026-08-03 run
    committed 911 of them.
    """

    uris = list(dict.fromkeys(_COLONY_IMAGE_URI_RE.findall(markdown)))
    if not uris:
        return markdown, 0
    if image_store is None:
        raise SidecarImageError(
            f"markdown references {len(uris)} colony-image figures but "
            f"no ImageStore is configured — cannot persist them into "
            f"the sidecar.",
        )

    # Fetch + validate EVERY referenced figure before touching the
    # filesystem — a failure must leave no partial sidecar behind.
    fetched: dict[str, tuple[str, bytes]] = {}  # uri -> (filename, bytes)
    for sha in uris:
        uri = f"colony-image://{sha}"
        payload = await image_store.get(uri)
        if payload is None:
            raise SidecarImageError(
                f"figure bytes for {uri} are not in the ImageStore — "
                f"refusing to write a sidecar with dangling image refs.",
            )
        info = await image_store.stat(uri)
        mime = str((info or {}).get("mime", "application/octet-stream"))
        fetched[uri] = (f"{sha}{ext_for_mime(mime)}", payload)

    images_dir = sidecar_dir / IMAGES_DIRNAME
    images_dir.mkdir(parents=True, exist_ok=True)
    replacements: dict[str, str] = {}
    for uri, (filename, payload) in fetched.items():
        (images_dir / filename).write_bytes(payload)
        replacements[uri] = f"{IMAGES_DIRNAME}/{filename}"

    rewritten = _COLONY_IMAGE_URI_RE.sub(
        lambda m: replacements[f"colony-image://{m.group(1)}"], markdown,
    )
    return rewritten, len(replacements)


async def _internalize_figures(
    markdown: str,
    sidecar_dir: Path,
    image_store: ImageStore | None,
) -> str:
    """Inverse of :func:`_externalize_figures` for the skip path: load
    each relative ``images/<sha>.<ext>`` ref from the sidecar, put the
    bytes into the process's :class:`ImageStore` (idempotent —
    content-addressed) and rewrite the ref back to its
    ``colony-image://`` URI, so downstream chunking sees exactly what
    a fresh extraction would produce.

    Raises :class:`SidecarImageError` when a referenced image file is
    missing (corrupt sidecar — caller falls back to re-extraction) or
    when no store is configured to receive the bytes.
    """

    refs = list(dict.fromkeys(_RELATIVE_IMAGE_REF_RE.findall(markdown)))
    if not refs:
        return markdown
    if image_store is None:
        raise SidecarImageError(
            f"sidecar references {len(refs)} figure files but no "
            f"ImageStore is configured to rehydrate them into.",
        )

    replacements: dict[str, str] = {}
    for sha, ext in refs:
        rel = f"{IMAGES_DIRNAME}/{sha}{ext}"
        image_path = sidecar_dir / IMAGES_DIRNAME / f"{sha}{ext}"
        if not image_path.is_file():
            raise SidecarImageError(
                f"sidecar image {rel} is missing from {sidecar_dir}.",
            )
        payload = image_path.read_bytes()
        uri = await image_store.put(payload, mime=mime_for_ext(ext))
        if uri != f"colony-image://{sha}":
            # Content-addressed round-trip broke: the file's bytes no
            # longer hash to its filename. Keep the TRUE uri (the
            # store's) so downstream lookups resolve, but flag the
            # integrity drift.
            logger.warning(
                "MonorepoPersistedIngestor: sidecar image %s hashes to "
                "%s (filename says %s) — file was modified after "
                "extraction.",
                image_path, uri, sha,
            )
        replacements[rel] = uri

    return _RELATIVE_IMAGE_REF_RE.sub(
        lambda m: replacements[f"{IMAGES_DIRNAME}/{m.group(1)}{m.group(2)}"],
        markdown,
    )


def _sections_to_markdown(sections: list[ParsedSection]) -> str:
    parts: list[str] = []
    for section in sections:
        heading = section.heading.strip()
        text = section.text
        if heading and not text.lstrip().startswith("#"):
            parts.append(f"# {heading}\n\n{text}")
        else:
            parts.append(text)
    return SECTION_SEPARATOR.join(parts) + "\n"


def _count_pages(sections: list[ParsedSection]) -> int:
    """Distinct page count from the typed
    :attr:`CitationSpan.page_number` field — the convention every PDF
    reader actually implements (Mistral / Anthropic / pypdf / remote
    all stamp it; an earlier version of this helper read
    ``section.extra['page']``, which NO reader sets, so every manifest
    recorded ``page_count: 0``). Returns 0 when no section carries a
    page number (page-less readers advertise ``page_number=None``)."""

    pages: set[int] = set()
    for section in sections:
        page_number = section.citation.page_number
        if page_number is not None:
            pages.add(page_number)
    return len(pages)


def _fail_record(*, source_uri: str, error: str) -> IngestionRecord:
    return IngestionRecord(
        source_uri=source_uri,
        status=IngestionStatus.FAILED,
        error=error,
    )


__all__ = (
    "EXTRACTED_MD_NAME",
    "IMAGES_DIRNAME",
    "INGESTION_JSON_NAME",
    "MonorepoPersistedIngestor",
    "SECTION_SEPARATOR",
    "SIDECAR_DIRNAME",
    "SidecarImageError",
    "SidecarManifest",
)
