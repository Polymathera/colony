"""Markdown reader: one ``ParsedSection`` per heading.

Walks the document line-by-line, splitting on ATX headings (``#``…
``######``). Each heading opens a new section; the ``section_path`` is
the numeric heading hierarchy (e.g. ``"3/3.1/3.1.4"``). Inline content
between headings is the section's body.

Pure-stdlib implementation — does not depend on the ``markdown``
library — because we want the *structure* (heading tree), not the
HTML render. Reusing the heading-detection regex avoids a dependency
that other readers (the GROBID-backed PDF reader) would re-import
anyway.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..models import CitationSpan, KnowledgeFormat, ParsedSection, RawDocument
from .base import FormatReader


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


class MarkdownReader(FormatReader):
    handles = (KnowledgeFormat.MARKDOWN,)

    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        text = document.text
        if not text:
            return ()
        lines = text.splitlines(keepends=True)

        # Track per-level heading counters for the section_path.
        counters: list[int] = [0] * 7  # index 1..6 used; 0 unused

        sections: list[ParsedSection] = []
        current_heading = ""
        current_level = 0
        current_lines: list[str] = []
        current_section_path = "0"
        current_char_start = 0
        offset = 0

        def flush(end_offset: int) -> None:
            body = "".join(current_lines).strip()
            if not body and not current_heading:
                return
            sections.append(
                ParsedSection(
                    section_path=current_section_path,
                    heading=current_heading,
                    text=(
                        f"{current_heading}\n\n{body}"
                        if current_heading and body
                        else current_heading or body
                    ),
                    citation=CitationSpan(
                        source_uri=document.source_uri,
                        section_path=current_section_path,
                        char_start=current_char_start,
                        char_end=end_offset,
                    ),
                )
            )

        for raw_line in lines:
            line = raw_line.rstrip("\n").rstrip("\r")
            m = _HEADING_RE.match(line) if line.startswith("#") else None
            if m:
                # New heading — flush the previous section first.
                flush(end_offset=offset)
                current_lines = []
                level = len(m.group(1))
                # Increment this level's counter; reset deeper levels.
                counters[level] += 1
                for d in range(level + 1, 7):
                    counters[d] = 0
                current_section_path = "/".join(
                    str(counters[i]) for i in range(1, level + 1) if counters[i] > 0
                )
                current_level = level
                current_heading = m.group(2).strip()
                current_char_start = offset
            else:
                current_lines.append(raw_line)
            offset += len(raw_line)

        flush(end_offset=offset)
        return tuple(sections)


__all__ = ("MarkdownReader",)
