"""CSV / TSV reader: one ``ParsedSection`` per row, plus a header section.

Header row becomes the first ``ParsedSection`` (section_path ``0``);
each subsequent row is a section keyed by row index. The reader
detects ``\\t`` vs. ``,`` from the file extension when available; for
in-memory payloads it sniffs by counting separators in the first line.

CSV is intentionally read row-at-a-time without ``pandas`` so the
reader has no heavy dependency.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence

from ..models import CitationSpan, KnowledgeFormat, ParsedSection, RawDocument
from .base import FormatReader


class CsvReader(FormatReader):
    handles = (KnowledgeFormat.CSV,)

    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        text = document.text
        if not text:
            return ()

        # Sniff separator: prefer tab if more tabs than commas in line 1.
        first_line = text.split("\n", 1)[0]
        delimiter = "\t" if first_line.count("\t") > first_line.count(",") else ","

        sections: list[ParsedSection] = []
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows = list(reader)
        if not rows:
            return ()

        # Char offset bookkeeping: re-scan the source to find each line's
        # start. Cheaper than tracking per-cell offsets.
        line_offsets: list[int] = [0]
        for line in text.splitlines(keepends=True)[:-1]:
            line_offsets.append(line_offsets[-1] + len(line))

        header = rows[0]
        sections.append(
            ParsedSection(
                section_path="0",
                heading="header",
                text=delimiter.join(header),
                citation=CitationSpan(
                    source_uri=document.source_uri,
                    section_path="0",
                    char_start=0,
                    char_end=line_offsets[1] if len(line_offsets) > 1 else len(text),
                ),
                extra={"columns": header, "delimiter": delimiter},
            )
        )

        for i, row in enumerate(rows[1:], start=1):
            char_start = line_offsets[i] if i < len(line_offsets) else len(text)
            char_end = (
                line_offsets[i + 1]
                if i + 1 < len(line_offsets)
                else len(text)
            )
            section_text_parts = []
            for col, val in zip(header, row):
                section_text_parts.append(f"{col}: {val}")
            sections.append(
                ParsedSection(
                    section_path=f"{i}",
                    heading=row[0][:80] if row else f"row {i}",
                    text="\n".join(section_text_parts) if section_text_parts else delimiter.join(row),
                    citation=CitationSpan(
                        source_uri=document.source_uri,
                        section_path=f"{i}",
                        char_start=char_start,
                        char_end=char_end,
                    ),
                    extra={"row_index": i, "columns": header},
                )
            )
        return tuple(sections)


__all__ = ("CsvReader",)
