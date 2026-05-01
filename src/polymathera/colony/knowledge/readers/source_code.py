"""Source-code reader: one ``ParsedSection`` per top-level
function / class / method block (best-effort, regex-driven).

Reuses the existing ``LanguageAwareTextChunker`` (``samples/paging/
sharding/analyzers/semantic.py``) for the actual block-detection
patterns at chunking time. Here we just split into top-level blocks;
the chunker handles deeper splitting.

For unknown languages the reader falls through to a single-section
result containing the full file text.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from ..formats import language_for_source_code
from ..models import CitationSpan, KnowledgeFormat, ParsedSection, RawDocument
from .base import FormatReader


# Per-language top-level block heading regex. Covers Python / C-family /
# Rust / Julia / Go. Bound to one-line heading patterns so we never
# parse a body — the chunker does that.
_BLOCK_HEADER: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^(class|def|async\s+def)\s+([A-Za-z_][A-Za-z0-9_]*)"),
    "c": re.compile(
        r"^(?:[A-Za-z_][A-Za-z0-9_*\s]+\s+)*([A-Za-z_][A-Za-z0-9_]*)\s*\("
    ),
    "cpp": re.compile(
        r"^(?:[A-Za-z_][A-Za-z0-9_:*\s<>]+\s+)*([A-Za-z_][A-Za-z0-9_]*)\s*\("
    ),
    "rust": re.compile(r"^(pub\s+)?(fn|struct|enum|trait|impl)\s+([A-Za-z_][A-Za-z0-9_]*)"),
    "julia": re.compile(r"^(function|struct|module)\s+([A-Za-z_][A-Za-z0-9_!]*)"),
    "go": re.compile(r"^func\s+(?:\([^)]*\)\s+)?([A-Za-z_][A-Za-z0-9_]*)"),
    "java": re.compile(
        r"^(?:public|protected|private|static|\s)+\s*[A-Za-z_<>\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
    ),
    "javascript": re.compile(
        r"^(?:export\s+)?(?:async\s+)?(class|function)\s+([A-Za-z_][A-Za-z0-9_$]*)"
    ),
    "typescript": re.compile(
        r"^(?:export\s+)?(?:async\s+)?(class|function|interface|type)\s+([A-Za-z_][A-Za-z0-9_$]*)"
    ),
}


class SourceCodeReader(FormatReader):
    handles = (KnowledgeFormat.SOURCE_CODE,)

    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        text = document.text
        if not text:
            return ()
        # Resolve the language id from the source URI's extension when
        # available; default to ``"text"``.
        ext_hint = document.metadata.get("language")
        if ext_hint:
            language = str(ext_hint)
        else:
            language = language_for_source_code(document.source_uri)
        pattern = _BLOCK_HEADER.get(language)

        if pattern is None or not text.strip():
            return (
                ParsedSection(
                    section_path="0",
                    heading=Path(document.source_uri).name or "file",
                    text=text,
                    citation=CitationSpan(
                        source_uri=document.source_uri,
                        section_path="0",
                        char_start=0,
                        char_end=len(text),
                    ),
                    extra={"language": language},
                ),
            )

        # Split by top-level block headers. Indented (continued)
        # definitions are kept inside their parent block.
        sections: list[ParsedSection] = []
        lines = text.splitlines(keepends=True)
        block_starts: list[int] = []
        block_headings: list[str] = []
        offset = 0
        offsets: list[int] = []
        for line in lines:
            offsets.append(offset)
            if not line.startswith((" ", "\t")):
                m = pattern.match(line.rstrip("\n").rstrip("\r"))
                if m:
                    block_starts.append(offset)
                    # Use the highest-index named group as the heading.
                    block_headings.append(
                        next(
                            (g for g in reversed(m.groups()) if g),
                            line.strip()[:80],
                        )
                    )
            offset += len(line)
        offsets.append(offset)

        if not block_starts:
            return (
                ParsedSection(
                    section_path="0",
                    heading=Path(document.source_uri).name or "file",
                    text=text,
                    citation=CitationSpan(
                        source_uri=document.source_uri,
                        section_path="0",
                        char_start=0,
                        char_end=len(text),
                    ),
                    extra={"language": language},
                ),
            )

        # Optional preamble (imports, top-level constants) before the
        # first block.
        if block_starts[0] > 0:
            sections.append(
                ParsedSection(
                    section_path="0",
                    heading="preamble",
                    text=text[: block_starts[0]],
                    citation=CitationSpan(
                        source_uri=document.source_uri,
                        section_path="0",
                        char_start=0,
                        char_end=block_starts[0],
                    ),
                    extra={"language": language},
                )
            )

        for i, (start, heading) in enumerate(
            zip(block_starts, block_headings), start=1
        ):
            end = block_starts[i] if i < len(block_starts) else len(text)
            sections.append(
                ParsedSection(
                    section_path=f"{i}",
                    heading=heading,
                    text=text[start:end],
                    citation=CitationSpan(
                        source_uri=document.source_uri,
                        section_path=f"{i}",
                        char_start=start,
                        char_end=end,
                    ),
                    extra={"language": language},
                )
            )
        return tuple(sections)


__all__ = ("SourceCodeReader",)
