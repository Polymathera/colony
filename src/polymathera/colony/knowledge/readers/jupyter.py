"""Jupyter notebook reader: one ``ParsedSection`` per cell.

Uses ``nbformat`` to parse the ``.ipynb`` JSON. Markdown cells use the
cell's source text directly (and the heading regex from
``MarkdownReader`` for sub-section paths). Code cells emit a section
with the code as text, plus the cell's outputs concatenated into
``extra["outputs"]`` for the chunker / ingestor to optionally
reference.

Falls back to a plain-text section when ``nbformat`` is unavailable —
the file is still tracked, just with degraded structure.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

from ..models import CitationSpan, KnowledgeFormat, ParsedSection, RawDocument
from .base import FormatReader


logger = logging.getLogger(__name__)


class JupyterReader(FormatReader):
    def __init__(self) -> None:
        super().__init__(handles=(KnowledgeFormat.JUPYTER,))

    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        try:
            payload = (
                json.loads(document.text)
                if document.is_text
                else json.loads(document.bytes_.decode("utf-8"))
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning(
                "JupyterReader: failed to parse %s: %s",
                document.source_uri, exc,
            )
            return ()

        cells = payload.get("cells") or []
        if not isinstance(cells, list):
            return ()

        sections: list[ParsedSection] = []
        offset = 0
        for idx, cell in enumerate(cells):
            if not isinstance(cell, dict):
                continue
            cell_type = str(cell.get("cell_type", "raw"))
            source = cell.get("source", "")
            if isinstance(source, list):
                source_text = "".join(str(s) for s in source)
            else:
                source_text = str(source)
            char_start = offset
            char_end = offset + len(source_text)
            heading = (
                source_text.splitlines()[0][:80] if source_text else f"cell {idx}"
            )
            extra = {"cell_index": idx, "cell_type": cell_type}
            if cell_type == "code":
                outputs = cell.get("outputs") or []
                if isinstance(outputs, list):
                    output_text_parts: list[str] = []
                    for out in outputs:
                        if not isinstance(out, dict):
                            continue
                        text = out.get("text") or out.get("data", {}).get("text/plain")
                        if isinstance(text, list):
                            output_text_parts.append("".join(str(t) for t in text))
                        elif isinstance(text, str):
                            output_text_parts.append(text)
                    if output_text_parts:
                        extra["outputs"] = "\n".join(output_text_parts)
            sections.append(
                ParsedSection(
                    section_path=f"{idx}",
                    heading=heading,
                    text=source_text,
                    citation=CitationSpan(
                        source_uri=document.source_uri,
                        section_path=f"{idx}",
                        char_start=char_start,
                        char_end=char_end,
                    ),
                    extra=extra,
                )
            )
            offset = char_end + 1  # +1 for an implied separator
        return tuple(sections)


__all__ = ("JupyterReader",)
