"""``GrobidPdfReader`` — PDF reader backed by GROBID's full-text API.

Per master §6.3, GROBID is the canonical PDF-to-structured-text path:
``processFulltextDocument`` returns TEI XML carrying the document's
section hierarchy (``<div type="section">``), per-section text, page
numbers (``<pb n="..."/>``), figures, tables, and a normalised
bibliography. This reader parses that TEI XML into the framework's
``ParsedSection`` records.

When GROBID is *not* available, callers should keep using the
in-process ``PdfReader`` (pypdf) fallback — the registry helper
``default_registry_with_grobid`` does this swap explicitly so the
choice is visible at the call site.

The reader uses ``httpx`` for the HTTP request (already a colony
dependency via ``dashboard``/``cpu`` extras) and ``xml.etree.ElementTree``
for parsing (stdlib, no extra deps).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from ..models import CitationSpan, KnowledgeFormat, ParsedSection, RawDocument
from .base import FormatReader, FormatReaderError


logger = logging.getLogger(__name__)


# TEI namespace — every TEI element is in this namespace.
_TEI_NS = "http://www.tei-c.org/ns/1.0"
_NS = {"tei": _TEI_NS}


class GrobidPdfReader(FormatReader):
    """PDF reader that calls GROBID's ``/api/processFulltextDocument``.

    Caller-supplied ``RawDocument.payload`` MUST be ``bytes``. The
    reader posts the bytes as ``multipart/form-data`` with the field
    name GROBID expects (``input``) and parses the returned TEI XML.

    Constructor takes the GROBID base URL (``http://grobid:8070``) and
    an optional ``timeout_s`` (default 60 s — GROBID processing can
    take seconds for non-trivial papers).
    """

    handles = (KnowledgeFormat.PDF,)

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float = 60.0,
        consolidate_header: int = 0,
        consolidate_citations: int = 0,
        include_raw_citations: bool = False,
    ) -> None:
        if not base_url:
            raise ValueError("GrobidPdfReader requires a non-empty base_url.")
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._consolidate_header = int(consolidate_header)
        self._consolidate_citations = int(consolidate_citations)
        self._include_raw_citations = bool(include_raw_citations)

    @property
    def base_url(self) -> str:
        return self._base_url

    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        """Sync entry point; runs the async HTTP call via
        ``httpx.Client``. ``read_async`` overrides for callers in an
        event loop."""

        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:
            raise FormatReaderError(
                "GrobidPdfReader requires the 'httpx' package; install via "
                "`pip install polymathera-colony[dashboard]` or "
                "`pip install httpx`.",
            ) from exc
        if document.is_text:
            raise FormatReaderError(
                f"GrobidPdfReader expected bytes for {document.source_uri}; "
                "got text.",
            )
        with httpx.Client(timeout=self._timeout_s) as client:
            response = client.post(
                f"{self._base_url}/api/processFulltextDocument",
                files={
                    "input": (
                        Path(document.source_uri).name or "document.pdf",
                        document.bytes_,
                        "application/pdf",
                    ),
                },
                data=self._form_fields(),
            )
        if response.status_code != 200:
            raise FormatReaderError(
                f"GROBID returned HTTP {response.status_code} for "
                f"{document.source_uri}: {response.text[:512]!r}",
            )
        return self._parse_tei(response.text, document.source_uri)

    async def read_async(
        self, document: RawDocument,
    ) -> Sequence[ParsedSection]:
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:
            raise FormatReaderError(
                "GrobidPdfReader requires the 'httpx' package.",
            ) from exc
        if document.is_text:
            raise FormatReaderError(
                f"GrobidPdfReader expected bytes for {document.source_uri}; "
                "got text.",
            )
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(
                f"{self._base_url}/api/processFulltextDocument",
                files={
                    "input": (
                        Path(document.source_uri).name or "document.pdf",
                        document.bytes_,
                        "application/pdf",
                    ),
                },
                data=self._form_fields(),
            )
        if response.status_code != 200:
            raise FormatReaderError(
                f"GROBID returned HTTP {response.status_code} for "
                f"{document.source_uri}: {response.text[:512]!r}",
            )
        return self._parse_tei(response.text, document.source_uri)

    def _form_fields(self) -> dict[str, Any]:
        return {
            "consolidateHeader": str(self._consolidate_header),
            "consolidateCitations": str(self._consolidate_citations),
            "includeRawCitations": "1" if self._include_raw_citations else "0",
            "segmentSentences": "0",
        }

    # ---- TEI parser ----------------------------------------------------

    def _parse_tei(
        self, tei_xml: str, source_uri: str,
    ) -> Sequence[ParsedSection]:
        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError as exc:
            raise FormatReaderError(
                f"GrobidPdfReader: malformed TEI XML for {source_uri}: {exc}",
            ) from exc

        sections: list[ParsedSection] = []
        char_offset = 0

        # Header (title + abstract) becomes section 0/0 and 0/1 when
        # present — useful as the "page 1" handle for a paper.
        title_text = self._extract_text(
            root.find(".//tei:teiHeader//tei:titleStmt/tei:title", _NS),
        )
        if title_text:
            char_end = char_offset + len(title_text)
            sections.append(
                ParsedSection(
                    section_path="0/0",
                    heading="title",
                    text=title_text,
                    citation=CitationSpan(
                        source_uri=source_uri,
                        section_path="0/0",
                        char_start=char_offset,
                        char_end=char_end,
                        page_number=1,
                    ),
                    extra={"role": "title"},
                )
            )
            char_offset = char_end + 2

        abstract_el = root.find(".//tei:teiHeader//tei:abstract", _NS)
        if abstract_el is not None:
            abstract_text = self._concat_paragraph_text(abstract_el)
            if abstract_text:
                char_end = char_offset + len(abstract_text)
                sections.append(
                    ParsedSection(
                        section_path="0/1",
                        heading="abstract",
                        text=abstract_text,
                        citation=CitationSpan(
                            source_uri=source_uri,
                            section_path="0/1",
                            char_start=char_offset,
                            char_end=char_end,
                            page_number=1,
                        ),
                        extra={"role": "abstract"},
                    )
                )
                char_offset = char_end + 2

        body = root.find(".//tei:text/tei:body", _NS)
        if body is not None:
            char_offset = self._walk_body(
                body, sections, source_uri, char_offset,
                section_path_prefix=(),
                counters=[0] * 10,
                current_page=1,
            )

        return tuple(sections)

    def _walk_body(
        self,
        node: ET.Element,
        sections: list[ParsedSection],
        source_uri: str,
        char_offset: int,
        *,
        section_path_prefix: tuple[int, ...],
        counters: list[int],
        current_page: int,
    ) -> int:
        """Recursive walk over the body's nested ``<div>`` sections.

        Each ``<div>`` represents a section; the depth in the tree
        becomes the section_path. ``<head>`` is the section's heading;
        ``<p>`` children are the body paragraphs. ``<pb n="..."/>``
        tags update the running page number.
        """

        depth = len(section_path_prefix)
        for child in list(node):
            tag = _local(child.tag)
            if tag == "pb":
                page_n = child.attrib.get("n")
                if page_n and page_n.isdigit():
                    current_page = int(page_n)
                continue
            if tag != "div":
                continue
            counters[depth] += 1
            for d in range(depth + 1, len(counters)):
                counters[d] = 0
            new_prefix = section_path_prefix + (counters[depth],)
            section_path = "/".join(str(c) for c in new_prefix if c > 0)

            head_el = child.find("tei:head", _NS)
            heading = self._extract_text(head_el) if head_el is not None else ""

            paragraphs: list[str] = []
            for grand in list(child):
                gtag = _local(grand.tag)
                if gtag == "pb":
                    pn = grand.attrib.get("n")
                    if pn and pn.isdigit():
                        current_page = int(pn)
                elif gtag == "p":
                    paragraphs.append(self._concat_paragraph_text(grand))
                elif gtag == "head":
                    continue  # already used
                # nested <div>s are handled by recursion below

            text_parts: list[str] = []
            if heading:
                text_parts.append(heading)
            text_parts.extend(p for p in paragraphs if p)
            text = "\n\n".join(text_parts)

            char_start = char_offset
            if text:
                char_end = char_offset + len(text)
                sections.append(
                    ParsedSection(
                        section_path=section_path,
                        heading=heading,
                        text=text,
                        citation=CitationSpan(
                            source_uri=source_uri,
                            section_path=section_path,
                            char_start=char_start,
                            char_end=char_end,
                            page_number=current_page,
                        ),
                        extra={"depth": depth + 1},
                    )
                )
                char_offset = char_end + 2

            # Recurse into nested divs.
            char_offset = self._walk_body(
                child, sections, source_uri, char_offset,
                section_path_prefix=new_prefix,
                counters=counters,
                current_page=current_page,
            )
        return char_offset

    @staticmethod
    def _extract_text(element: ET.Element | None) -> str:
        if element is None:
            return ""
        return "".join(element.itertext()).strip()

    @staticmethod
    def _concat_paragraph_text(paragraph: ET.Element) -> str:
        return GrobidPdfReader._extract_text(paragraph)


def _local(tag: str) -> str:
    """Return the local-name of a possibly-namespaced tag."""

    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


__all__ = ("GrobidPdfReader",)
