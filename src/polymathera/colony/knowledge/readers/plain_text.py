"""Plain-text reader: one ``ParsedSection`` per blank-line block.

Treats each chunk of text separated by ``\\n\\n`` as a section. The
section heading is the first line truncated to 80 chars; this is
heuristic and meant for log-file / readme-style text where there's no
structural markup.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..models import CitationSpan, KnowledgeFormat, ParsedSection, RawDocument
from .base import FormatReader


class PlainTextReader(FormatReader):
    handles = (KnowledgeFormat.PLAIN_TEXT,)

    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        text = document.text
        sections: list[ParsedSection] = []
        offset = 0
        for idx, raw_block in enumerate(text.split("\n\n")):
            block = raw_block.strip()
            if not block:
                offset += len(raw_block) + 2
                continue
            char_start = text.find(raw_block, offset)
            char_end = char_start + len(raw_block)
            heading = block.splitlines()[0][:80]
            sections.append(
                ParsedSection(
                    section_path=f"{idx}",
                    heading=heading,
                    text=block,
                    citation=CitationSpan(
                        source_uri=document.source_uri,
                        section_path=f"{idx}",
                        char_start=char_start,
                        char_end=char_end,
                    ),
                )
            )
            offset = char_end + 2
        if not sections and text:
            sections.append(
                ParsedSection(
                    section_path="0",
                    heading=text[:80].splitlines()[0] if text else "",
                    text=text,
                    citation=CitationSpan(
                        source_uri=document.source_uri,
                        section_path="0",
                        char_start=0,
                        char_end=len(text),
                    ),
                )
            )
        return sections


__all__ = ("PlainTextReader",)
