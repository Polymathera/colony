"""JSONL / NDJSON reader: one ``ParsedSection`` per line.

Each non-empty line is parsed as JSON. Malformed lines are kept as
plain-text sections so the ingestor can flag them for review rather
than failing the whole file. Each section's heading is the value at
``"title"`` if present, else the first 80 chars of the JSON dump.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from ..models import CitationSpan, KnowledgeFormat, ParsedSection, RawDocument
from .base import FormatReader


class JsonlReader(FormatReader):
    def __init__(self) -> None:
        super().__init__(handles=(KnowledgeFormat.JSONL,))

    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        text = document.text
        sections: list[ParsedSection] = []
        offset = 0
        for idx, raw_line in enumerate(text.splitlines(keepends=True)):
            line = raw_line.rstrip("\n").rstrip("\r")
            stripped = line.strip()
            if not stripped:
                offset += len(raw_line)
                continue
            char_start = offset
            char_end = offset + len(line)
            heading = ""
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                heading = f"malformed:{idx}"
                section_text = line
            else:
                if isinstance(obj, dict):
                    if "title" in obj and isinstance(obj["title"], str):
                        heading = obj["title"][:80]
                    elif "name" in obj and isinstance(obj["name"], str):
                        heading = obj["name"][:80]
                section_text = json.dumps(obj, ensure_ascii=False, sort_keys=True)
                if not heading:
                    heading = section_text[:80]
            sections.append(
                ParsedSection(
                    section_path=f"{idx}",
                    heading=heading,
                    text=section_text,
                    citation=CitationSpan(
                        source_uri=document.source_uri,
                        section_path=f"{idx}",
                        char_start=char_start,
                        char_end=char_end,
                    ),
                    extra={"row_index": idx},
                )
            )
            offset += len(raw_line)
        return tuple(sections)


__all__ = ("JsonlReader",)
