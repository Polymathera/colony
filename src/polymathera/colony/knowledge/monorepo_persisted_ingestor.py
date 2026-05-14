"""``MonorepoPersistedIngestor`` — :class:`Ingestor` wrapper that
persists per-source extraction outputs as a sidecar next to the source
file in a design monorepo.

For each PDF input at ``<dir>/<stem>.pdf`` the wrapper maintains a
sidecar directory ``<dir>/.ingested/<stem>/``:

- ``extracted.md`` — concatenated reader markdown across sections,
  with ``<!-- page: N -->`` separators between sections. Plain git
  (text). Users can read or edit it.
- ``ingestion.json`` — Pydantic-validated :class:`SidecarManifest`:
  pdf_sha256, extractor backend label, extracted_at timestamp,
  section_count, page_count, source_uri. Plain git (small JSON).

The wrapper avoids re-paying the reader's extraction cost on
re-ingest of an unchanged PDF: if ``ingestion.json``'s ``pdf_sha256``
matches the current PDF, the wrapper skips the reader and feeds the
on-disk ``extracted.md`` directly into :meth:`Ingestor.ingest_text`
for chunking + embedding. When ``extracted.md`` is newer than the
PDF (user-edited markdown), the wrapper trusts the edit and ingests
the edited markdown — chunking + embedding re-run; the reader does
not.

For non-PDF inputs (markdown, source code, plain text, etc.) the
wrapper delegates to :meth:`Ingestor.ingest_file` directly — the
underlying file already IS the readable artifact, a sidecar would
duplicate it.

**Known gap — figure persistence.** Markdown image references emitted
by multimodal PDF readers point at ``colony-image://`` URIs resolved
through the :class:`ImageStore`. Today the wrapper does not copy
those bytes into the sidecar's ``images/`` subdirectory. Figure bytes
remain in whichever ``ImageStore`` the cluster is configured with
(``LocalFsImageStore`` over ``knowledge.image_dir``). Persisting
figures into the sidecar + rewriting the markdown's image refs to
relative paths is a focused follow-up; the current wrapper unblocks
the markdown + metadata persistence the operator pays the most for.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

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


logger = logging.getLogger(__name__)


SIDECAR_DIRNAME = ".ingested"
EXTRACTED_MD_NAME = "extracted.md"
INGESTION_JSON_NAME = "ingestion.json"

# Separator the wrapper emits between sections in ``extracted.md``. The
# same form the multimodal readers already use for page boundaries, so
# round-trip parses cleanly through the markdown reader on the skip
# path.
SECTION_SEPARATOR = "\n\n<!-- section -->\n\n"

_SHA256_BLOCKSIZE = 1 << 20


class SidecarManifest(BaseModel):
    """Schema for ``<sidecar>/ingestion.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    source_uri: str
    pdf_sha256: str
    extractor: str = ""
    """Label of the reader / pipeline that produced ``extracted.md``.
    Free-form (``"anthropic"``, ``"mistral_ocr"``, ``"grobid"``, ...);
    used for diagnostics + future cache invalidation when a backend's
    extraction quality changes."""
    extracted_at: str
    """ISO-8601 UTC timestamp."""
    section_count: int
    page_count: int = 0
    """Best-effort; sourced from section.extra['page'] when present."""


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
            pdf_path=path_obj,
            pdf_sha256=pdf_sha256,
            extracted_md_path=extracted_md_path,
            manifest_path=manifest_path,
        )

        if skip_extraction:
            logger.info(
                "MonorepoPersistedIngestor: skipping extraction for %s — %s",
                path_obj, skip_reason,
            )
            extracted_md = extracted_md_path.read_text(encoding="utf-8")
            return await self._ingestor.ingest_text(
                extracted_md,
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
        manifest = SidecarManifest(
            source_uri=canonical_uri,
            pdf_sha256=pdf_sha256,
            extractor=self._extractor_label,
            extracted_at=datetime.now(timezone.utc).isoformat(),
            section_count=len(sections),
            page_count=_count_pages(sections),
        )
        self._write_sidecar(
            sidecar_dir=sidecar_dir,
            extracted_md=extracted_md,
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
        pdf_path: Path,
        pdf_sha256: str,
        extracted_md_path: Path,
        manifest_path: Path,
    ) -> tuple[bool, str]:
        """Return ``(skip, reason)``.

        Skip when either:
        - manifest.pdf_sha256 matches the current PDF (cache hit), or
        - ``extracted.md`` mtime > PDF mtime (user-edited markdown).

        Otherwise re-extract.
        """

        if not extracted_md_path.is_file():
            return False, "no sidecar"

        # User-edited markdown takes precedence — trust the edit.
        try:
            md_mtime = extracted_md_path.stat().st_mtime
            pdf_mtime = pdf_path.stat().st_mtime
        except OSError:
            return False, "stat failed"
        if md_mtime > pdf_mtime:
            return True, "extracted.md edited after PDF"

        if not manifest_path.is_file():
            return False, "no manifest"

        try:
            manifest = SidecarManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8"),
            )
        except Exception:  # noqa: BLE001 — corrupt JSON / schema drift
            return False, "manifest unreadable"

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
            except FormatReaderError as exc:
                last_error = exc
                logger.warning(
                    "MonorepoPersistedIngestor: reader %s rejected %s (%s)",
                    type(reader).__name__, source_uri, exc,
                )
                continue
            except Exception as exc:  # noqa: BLE001
                last_error = exc
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
        (sidecar_dir / EXTRACTED_MD_NAME).write_text(
            extracted_md, encoding="utf-8",
        )
        (sidecar_dir / INGESTION_JSON_NAME).write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
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


_PAGE_NUMBER_RE = re.compile(r"^\s*page\s*[:=]\s*(\d+)\s*$", re.IGNORECASE)


def _count_pages(sections: list[ParsedSection]) -> int:
    """Best-effort page count from section.extra['page'] (per reader
    convention). Returns 0 when no section advertises a page number."""

    pages: set[int] = set()
    for section in sections:
        raw = section.extra.get("page")
        if isinstance(raw, int):
            pages.add(raw)
        elif isinstance(raw, str):
            m = _PAGE_NUMBER_RE.match(raw)
            if m:
                pages.add(int(m.group(1)))
    return len(pages)


def _fail_record(*, source_uri: str, error: str) -> IngestionRecord:
    return IngestionRecord(
        source_uri=source_uri,
        status=IngestionStatus.FAILED,
        error=error,
    )


__all__ = (
    "EXTRACTED_MD_NAME",
    "INGESTION_JSON_NAME",
    "MonorepoPersistedIngestor",
    "SECTION_SEPARATOR",
    "SIDECAR_DIRNAME",
    "SidecarManifest",
)
