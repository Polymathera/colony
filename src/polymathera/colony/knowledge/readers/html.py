"""HTML reader: one ``ParsedSection`` per ``<h1>``…``<h6>`` heading.

Pure-stdlib using ``html.parser`` so we don't depend on
``beautifulsoup``. Strips out ``<script>`` and ``<style>`` blocks
entirely, decodes HTML entities, and emits one section per heading
(numeric hierarchy, same shape as the markdown reader).
"""

from __future__ import annotations

from collections.abc import Sequence
from html import unescape
from html.parser import HTMLParser

from ..models import CitationSpan, KnowledgeFormat, ParsedSection, RawDocument
from .base import FormatReader


_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_BLOCK_TAGS = frozenset(
    {"p", "div", "section", "article", "li", "tr", "td", "th", "br", "pre", "blockquote"}
)
_SKIP_TAGS = frozenset({"script", "style", "noscript"})


class _SectionedExtractor(HTMLParser):
    def __init__(self, source_uri: str) -> None:
        super().__init__(convert_charrefs=True)
        self._source_uri = source_uri
        self._skip_depth = 0
        self._heading_level: int | None = None
        self._heading_buf: list[str] = []
        self._body_buf: list[str] = []
        self._counters: list[int] = [0] * 7
        self._sections: list[ParsedSection] = []
        self._open_section_path = "0"
        self._open_heading = ""
        self._char_offset = 0
        self._section_char_start = 0

    def handle_starttag(self, tag: str, attrs):  # type: ignore[override]
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag in _HEADING_TAGS:
            # Flush whatever's buffered.
            self._flush_section()
            level = int(tag[1])
            self._counters[level] += 1
            for d in range(level + 1, 7):
                self._counters[d] = 0
            self._open_section_path = "/".join(
                str(self._counters[i])
                for i in range(1, level + 1)
                if self._counters[i] > 0
            )
            self._heading_level = level
            self._heading_buf = []
            self._body_buf = []
            self._section_char_start = self._char_offset
        elif tag in _BLOCK_TAGS:
            # Insert a paragraph break in the body buffer.
            if self._heading_level is not None:
                self._body_buf.append("\n\n")

    def handle_endtag(self, tag: str):  # type: ignore[override]
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag in _HEADING_TAGS and self._heading_level is not None:
            self._open_heading = "".join(self._heading_buf).strip()
            self._heading_level = None
            self._heading_buf = []

    def handle_data(self, data: str):  # type: ignore[override]
        self._char_offset += len(data)
        if self._skip_depth > 0:
            return
        if self._heading_level is not None and self._heading_buf is not None:
            self._heading_buf.append(unescape(data))
            return
        self._body_buf.append(unescape(data))

    def _flush_section(self) -> None:
        body = "".join(self._body_buf).strip()
        heading = self._open_heading
        if not body and not heading:
            return
        text = (
            f"{heading}\n\n{body}" if heading and body else heading or body
        )
        self._sections.append(
            ParsedSection(
                section_path=self._open_section_path,
                heading=heading,
                text=text,
                citation=CitationSpan(
                    source_uri=self._source_uri,
                    section_path=self._open_section_path,
                    char_start=self._section_char_start,
                    char_end=self._char_offset,
                ),
            )
        )
        self._body_buf = []
        self._open_heading = ""

    def finalise(self) -> tuple[ParsedSection, ...]:
        self._flush_section()
        return tuple(self._sections)


class HtmlReader(FormatReader):
    handles = (KnowledgeFormat.HTML,)

    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        text = document.text
        if not text:
            return ()
        extractor = _SectionedExtractor(document.source_uri)
        try:
            extractor.feed(text)
            extractor.close()
        except Exception:  # noqa: BLE001 - HTMLParser is forgiving but log.
            return tuple(extractor.finalise()) if extractor._sections else ()
        return extractor.finalise()


__all__ = ("HtmlReader",)
