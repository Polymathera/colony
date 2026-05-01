"""Detect a ``KnowledgeFormat`` from a path / payload / mime hint.

The detector runs three checks in order:

1. Explicit hint (caller supplied ``mime_hint``).
2. File extension (``.pdf``, ``.md``, ``.ipynb``, …).
3. Magic-byte sniff for binary formats (PDF starts with ``%PDF-``,
   Parquet ends with ``PAR1``, HDF5 starts with ``\\x89HDF``, …).

Anything that fails all three is ``KnowledgeFormat.UNKNOWN`` — the
caller chooses whether to fall back to ``PlainTextReader`` or refuse
ingestion.

The detector is colony-generic; CPS adds domain-specific extensions
(``.reqif`` is in this list because the dossier-shared reqif-merge
driver needs it; ``.dwg`` etc. live in cps).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .models import KnowledgeFormat


# ---------------------------------------------------------------------------
# Extension table
# ---------------------------------------------------------------------------


EXTENSION_MAP: dict[str, KnowledgeFormat] = {
    ".txt": KnowledgeFormat.PLAIN_TEXT,
    ".log": KnowledgeFormat.PLAIN_TEXT,
    ".md": KnowledgeFormat.MARKDOWN,
    ".markdown": KnowledgeFormat.MARKDOWN,
    ".rst": KnowledgeFormat.MARKDOWN,
    ".html": KnowledgeFormat.HTML,
    ".htm": KnowledgeFormat.HTML,
    ".pdf": KnowledgeFormat.PDF,
    ".docx": KnowledgeFormat.DOCX,
    ".ipynb": KnowledgeFormat.JUPYTER,
    ".jsonl": KnowledgeFormat.JSONL,
    ".ndjson": KnowledgeFormat.JSONL,
    ".csv": KnowledgeFormat.CSV,
    ".tsv": KnowledgeFormat.CSV,
    ".parquet": KnowledgeFormat.PARQUET,
    ".pq": KnowledgeFormat.PARQUET,
    ".h5": KnowledgeFormat.HDF5,
    ".hdf5": KnowledgeFormat.HDF5,
    ".reqif": KnowledgeFormat.REQIF,
    ".reqifz": KnowledgeFormat.REQIF,
    # Source code — open-set; the source-code reader picks the language.
    ".py": KnowledgeFormat.SOURCE_CODE,
    ".pyi": KnowledgeFormat.SOURCE_CODE,
    ".c": KnowledgeFormat.SOURCE_CODE,
    ".h": KnowledgeFormat.SOURCE_CODE,
    ".cpp": KnowledgeFormat.SOURCE_CODE,
    ".hpp": KnowledgeFormat.SOURCE_CODE,
    ".cc": KnowledgeFormat.SOURCE_CODE,
    ".rs": KnowledgeFormat.SOURCE_CODE,
    ".jl": KnowledgeFormat.SOURCE_CODE,
    ".go": KnowledgeFormat.SOURCE_CODE,
    ".java": KnowledgeFormat.SOURCE_CODE,
    ".js": KnowledgeFormat.SOURCE_CODE,
    ".ts": KnowledgeFormat.SOURCE_CODE,
    ".tsx": KnowledgeFormat.SOURCE_CODE,
    ".jsx": KnowledgeFormat.SOURCE_CODE,
    ".scala": KnowledgeFormat.SOURCE_CODE,
    ".kt": KnowledgeFormat.SOURCE_CODE,
    ".rb": KnowledgeFormat.SOURCE_CODE,
    ".swift": KnowledgeFormat.SOURCE_CODE,
    ".m": KnowledgeFormat.SOURCE_CODE,
    ".sh": KnowledgeFormat.SOURCE_CODE,
}


SOURCE_CODE_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".rs": "rust",
    ".jl": "julia",
    ".go": "go",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".scala": "scala",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".swift": "swift",
    ".m": "objectivec",
    ".sh": "bash",
}


# ---------------------------------------------------------------------------
# Magic-byte sniffer
# ---------------------------------------------------------------------------


_MAGIC: tuple[tuple[bytes, KnowledgeFormat], ...] = (
    (b"%PDF-", KnowledgeFormat.PDF),
    (b"PK\x03\x04", KnowledgeFormat.DOCX),  # zip wrapper — disambiguated below
    (b"\x89HDF", KnowledgeFormat.HDF5),
)


def _sniff_magic(payload: bytes) -> KnowledgeFormat | None:
    if not payload:
        return None
    head = payload[:8]
    for marker, fmt in _MAGIC:
        if head.startswith(marker):
            # PK\x03\x04 (zip) disambiguates: DOCX zips contain
            # ``[Content_Types].xml`` with a specific Office string.
            if fmt is KnowledgeFormat.DOCX:
                if b"word/" in payload[:4096] or b"[Content_Types].xml" in payload[:4096]:
                    return KnowledgeFormat.DOCX
                return None  # pure zip, not docx
            return fmt
    if payload.endswith(b"PAR1"):
        return KnowledgeFormat.PARQUET
    return None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def detect_format(
    *,
    path: str | Path | None = None,
    payload: bytes | str | None = None,
    mime_hint: str | None = None,
) -> KnowledgeFormat:
    """Detect the format of a source.

    At least one of ``path`` / ``payload`` / ``mime_hint`` must be
    supplied; pass as many as available to maximise accuracy.
    """

    if mime_hint:
        m = _from_mime(mime_hint)
        if m is not KnowledgeFormat.UNKNOWN:
            return m

    if path is not None:
        p = Path(path)
        ext = p.suffix.lower()
        if ext in EXTENSION_MAP:
            return EXTENSION_MAP[ext]

    if isinstance(payload, (bytes, bytearray)):
        sniffed = _sniff_magic(bytes(payload))
        if sniffed is not None:
            return sniffed
    elif isinstance(payload, str):
        # A text payload with no path / mime hint defaults to plain text.
        # Markdown / HTML detection from raw text alone is unreliable; if
        # the caller wants a different reader they pass the hint.
        if path is None:
            return KnowledgeFormat.PLAIN_TEXT

    return KnowledgeFormat.UNKNOWN


def language_for_source_code(path: str | Path) -> str:
    """Best-effort programming-language id for a source-code file.

    Defaults to ``"text"`` when the extension isn't in
    ``SOURCE_CODE_LANGUAGE``.
    """

    return SOURCE_CODE_LANGUAGE.get(Path(path).suffix.lower(), "text")


_MIME_MAP: dict[str, KnowledgeFormat] = {
    "text/plain": KnowledgeFormat.PLAIN_TEXT,
    "text/markdown": KnowledgeFormat.MARKDOWN,
    "text/html": KnowledgeFormat.HTML,
    "application/pdf": KnowledgeFormat.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": KnowledgeFormat.DOCX,
    "application/x-ipynb+json": KnowledgeFormat.JUPYTER,
    "application/x-jupyter-notebook": KnowledgeFormat.JUPYTER,
    "application/x-ndjson": KnowledgeFormat.JSONL,
    "application/jsonl": KnowledgeFormat.JSONL,
    "text/csv": KnowledgeFormat.CSV,
    "application/vnd.apache.parquet": KnowledgeFormat.PARQUET,
    "application/x-hdf5": KnowledgeFormat.HDF5,
    "application/reqif+xml": KnowledgeFormat.REQIF,
}


def _from_mime(mime: str) -> KnowledgeFormat:
    return _MIME_MAP.get(mime.split(";")[0].strip().lower(), KnowledgeFormat.UNKNOWN)


__all__ = (
    "EXTENSION_MAP",
    "SOURCE_CODE_LANGUAGE",
    "detect_format",
    "language_for_source_code",
)
