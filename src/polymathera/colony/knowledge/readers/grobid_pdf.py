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

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float = 60.0,
        consolidate_header: int = 0,
        consolidate_citations: int = 0,
        include_raw_citations: bool = False,
        mode: str = "full",
    ) -> None:
        """Construct the GROBID reader.

        Args:
            base_url: GROBID base URL (``http://grobid:8070``).
            timeout_s: Per-call HTTP timeout.
            consolidate_header / consolidate_citations / include_raw_citations:
                GROBID form-field passthroughs.
            mode: ``"full"`` (default, legacy behaviour) emits one
                section per body section plus title + abstract.
                ``"metadata_only"`` skips body extraction and emits
                ONLY the title + abstract + bibliographic envelope —
                used when a layout-aware reader (Mistral OCR,
                Marker, Docling, MinerU, Anthropic, Gemini,
                LlamaParse) is the canonical body extractor and
                GROBID is along just to harvest the citation graph.
                The "metadata_only" sections carry an
                ``extra["bibliographic"]`` payload with authors,
                affiliations, and the parsed reference list so
                downstream consumers (the Phase 4 cross-reference
                preserver, future citation-graph features) can hook
                in.
        """
        super().__init__(handles=(KnowledgeFormat.PDF,))
        if not base_url:
            raise ValueError("GrobidPdfReader requires a non-empty base_url.")
        if mode not in ("full", "metadata_only"):
            raise ValueError(
                f"GrobidPdfReader.mode must be 'full' or 'metadata_only'; "
                f"got {mode!r}.",
            )
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._consolidate_header = int(consolidate_header)
        self._consolidate_citations = int(consolidate_citations)
        self._include_raw_citations = bool(include_raw_citations)
        self._mode = mode

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

        if self._mode == "full":
            body = root.find(".//tei:text/tei:body", _NS)
            if body is not None:
                char_offset = self._walk_body(
                    body, sections, source_uri, char_offset,
                    section_path_prefix=(),
                    counters=[0] * 10,
                    current_page=1,
                )

        # In ``metadata_only`` mode (and harmlessly in ``full`` too),
        # attach the parsed bibliographic envelope to the FIRST
        # section's ``extra["bibliographic"]`` so downstream
        # citation-graph consumers can read it without re-parsing
        # the TEI. Authors, affiliations, and the references list
        # are the operator-visible value of GROBID's continued
        # presence after the body-extraction demote.
        if sections:
            bibliographic = self._extract_bibliographic(root)
            if bibliographic:
                first = sections[0]
                merged_extra = dict(first.extra)
                merged_extra["bibliographic"] = bibliographic
                merged_extra.setdefault("metadata_origin", "grobid")
                sections[0] = first.model_copy(update={"extra": merged_extra})

        return tuple(sections)

    def _extract_bibliographic(
        self, root: ET.Element,
    ) -> dict[str, Any]:
        """Return the citation-graph payload GROBID is genuinely
        good at: authors, affiliations, parsed references.

        Empty dict if nothing useful was found — keeps
        ``extra["bibliographic"]`` from carrying a noisy envelope on
        documents GROBID couldn't fingerprint.
        """
        out: dict[str, Any] = {}
        authors: list[dict[str, Any]] = []
        for analytic_author in root.findall(
            ".//tei:teiHeader//tei:analytic//tei:author", _NS,
        ):
            persname = analytic_author.find("tei:persName", _NS)
            full_name = self._concat_paragraph_text(persname) if persname is not None else ""
            affiliation_el = analytic_author.find(".//tei:affiliation", _NS)
            affiliation = (
                self._concat_paragraph_text(affiliation_el)
                if affiliation_el is not None
                else ""
            )
            if full_name or affiliation:
                authors.append(
                    {"name": full_name, "affiliation": affiliation},
                )
        if authors:
            out["authors"] = authors

        references: list[dict[str, Any]] = []
        for biblio in root.findall(
            ".//tei:listBibl//tei:biblStruct", _NS,
        ):
            title_el = biblio.find(".//tei:title", _NS)
            title_text = self._extract_text(title_el)
            year_el = biblio.find(".//tei:date[@type='published']", _NS)
            year_text = year_el.get("when") if year_el is not None else None
            if title_text:
                references.append(
                    {"title": title_text, "year": year_text},
                )
        if references:
            out["references"] = references
        return out

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


class GrobidMetadataReader(GrobidPdfReader):
    """Convenience alias: ``GrobidPdfReader(mode="metadata_only")``.

    Use this when a layout-aware reader (Mistral OCR, Marker,
    Docling, MinerU, Anthropic, Gemini, LlamaParse) is the canonical
    body extractor and GROBID is along solely to harvest the
    citation graph. The reader emits the title + abstract sections
    with ``extra["bibliographic"]`` populated; no body text.

    Today the reader cannot run *alongside* the body extractor in
    the same registry — :class:`ReaderRegistry` is one reader per
    format. The Phase 4 multi-reader Ingestor pass (tracked
    separately in design doc §10) is what unlocks running both.
    Until that ships, the operator's choice is body-extractor OR
    metadata-reader; this class exists so the metadata path is
    one explicit class with the right defaults.
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float = 60.0,
        consolidate_header: int = 1,
        consolidate_citations: int = 1,
        include_raw_citations: bool = True,
    ) -> None:
        super().__init__(
            base_url=base_url,
            timeout_s=timeout_s,
            consolidate_header=consolidate_header,
            consolidate_citations=consolidate_citations,
            include_raw_citations=include_raw_citations,
            mode="metadata_only",
        )


__all__ = ("GrobidPdfReader", "GrobidMetadataReader")
