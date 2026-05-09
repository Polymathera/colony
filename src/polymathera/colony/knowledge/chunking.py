"""Chunkers that turn ``ParsedSection``s into retrieval-sized ``Chunk``s.

Two strategies, picked by content type:

- ``ProseChunker`` — paragraph-aware sliding window for prose
  (paper sections, standards clauses, manual passages). Targets a
  configurable token budget (default 800), with a configurable overlap
  (default 100) so long documents have continuity across chunks.
- ``CodeChunker`` — wraps the existing
  ``LanguageAwareTextChunker`` (``samples/paging/sharding/analyzers/
  semantic.py``) which already does code-block-aware splitting for
  every language colony supports. We reuse it directly rather than
  duplicate.

Token counts are produced by ``tiktoken`` via the
``token_counter_factory`` callback. The default factory returns a
``cl100k_base`` counter (matching colony's default LLM tokenizer);
tests pass a deterministic word-count counter.

The boundary: chunkers DO NOT load files, embed, or store. They take
``ParsedSection`` and emit ``Chunk``. The ``Ingestor`` (next module)
glues the stages together.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .models import (
    Chunk,
    CitationSpan,
    CorpusTier,
    ParsedSection,
)
from .stores.image import URI_SCHEME as _IMAGE_URI_SCHEME


logger = logging.getLogger(__name__)


# Matches every ``colony-image://<sha>`` URI a layout-aware reader
# may have inlined into the section markdown (via Mistral / Marker /
# Docling / MinerU image references). The chunker scans each chunk
# for these URIs and resolves them against the section's
# ``figures`` to populate ``Chunk.extra["figure_ids"]``.
_IMAGE_URI_RE = re.compile(
    r"" + re.escape(_IMAGE_URI_SCHEME) + r"://[A-Za-z0-9]+",
)


def _chunk_extra_for(
    section: ParsedSection, chunk_text: str,
) -> dict[str, object]:
    """Build the ``Chunk.extra`` payload for a chunk derived from
    ``section``.

    Two responsibilities, both small enough to live inline rather
    than grow into their own class:

    1. **Provenance.** Forward
       ``section.extra["metadata_origin"]`` so downstream consumers
       can tell which extractor produced the chunk (``mistral_ocr``,
       ``marker``, …) without re-resolving the source.
    2. **Figure linkage.** Scan the chunk text for
       ``colony-image://<sha>`` URIs (markdown image references the
       layout-aware reader inlined) and record the matching
       :class:`~polymathera.colony.knowledge.models.FigureRef` IDs
       under ``figure_ids``. The agent's planner pulls these via
       :class:`RetrievalHit.figures` to fetch the bytes when it
       wants multimodal context for an answer.

    Returns an empty dict for plain-text chunks with no figures and
    no provenance metadata — keeps existing serialised chunks
    bit-identical so the move to the multimodal pipeline doesn't
    invalidate caches.
    """

    extra: dict[str, object] = {}

    origin = section.extra.get("metadata_origin")
    if origin:
        extra["metadata_origin"] = origin

    if section.figures:
        # Build a uri → figure_id index once per chunk; section
        # figure lists are typically small (a handful per page) so
        # the dict construction is cheap. We could cache it on the
        # section but ParsedSection is frozen and the chunker is
        # called once per section in the typical pipeline.
        uri_to_id = {f.image_uri: f.figure_id for f in section.figures}
        seen_ids: list[str] = []
        for uri in _IMAGE_URI_RE.findall(chunk_text):
            fid = uri_to_id.get(uri)
            if fid is None or fid in seen_ids:
                continue
            seen_ids.append(fid)
        if seen_ids:
            extra["figure_ids"] = seen_ids

    return extra


TokenCounter = Callable[[str], int]
"""Function that estimates the token count of a piece of text. Defaults
to a length-based approximation; production wires it to ``tiktoken``."""


def default_token_counter() -> TokenCounter:
    """Return a tiktoken-cl100k_base counter when ``tiktoken`` is
    available, else fall back to a 4-chars-per-token approximation
    (a reasonable heuristic for English prose)."""

    try:
        import tiktoken  # type: ignore[import-not-found]

        encoder = tiktoken.get_encoding("cl100k_base")

        def _count(text: str) -> int:
            try:
                return len(encoder.encode(text, disallowed_special=()))
            except Exception:  # noqa: BLE001
                return max(1, len(text) // 4)

        return _count
    except ImportError:
        logger.info("tiktoken not installed; using 4-chars-per-token approximation")
        return lambda text: max(1, len(text) // 4)


@dataclass(frozen=True)
class ChunkerConfig:
    """Tuning knobs for prose / code chunkers."""

    target_tokens: int = 800
    """Target token count per chunk. ProseChunker fills near this
    before emitting; CodeChunker uses it as the max."""

    overlap_tokens: int = 100
    """Sliding-window overlap (in tokens) for ProseChunker."""

    min_tokens: int = 32
    """Drop a chunk smaller than this (avoids one-sentence chunks
    polluting the index)."""

    max_tokens: int = 2_400
    """Hard cap; chunks above this are split."""


class ProseChunker:
    """Paragraph-aware sliding-window chunker for prose."""

    _PARAGRAPH_RE = re.compile(r"\n\s*\n")

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._config = config or ChunkerConfig()
        self._count = token_counter or default_token_counter()

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split ``text`` into paragraph-shaped units the chunker
        treats as atomic when packing into target-sized chunks.

        Default behaviour is blank-line-separated paragraphs. Subclasses
        (notably :class:`MarkdownChunker`) override to keep fenced
        code blocks, GFM tables, and display math intact through
        chunking."""

        return [p.strip() for p in self._PARAGRAPH_RE.split(text) if p.strip()]

    def chunk(
        self,
        section: ParsedSection,
        *,
        data_type: str = "paper_section",
        source: str | None = None,
        tier: CorpusTier = CorpusTier.UNTIERED,
        language: str | None = None,
    ) -> Sequence[Chunk]:
        """Split ``section.text`` into chunks with paragraph-aware
        boundaries and ``overlap_tokens`` of overlap.

        ``source`` defaults to ``section.citation.source_uri`` so
        retrievers can filter by source without re-resolving the
        section."""

        text = section.text
        if not text or not text.strip():
            return ()

        cfg = self._config
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return ()

        para_tokens = [self._count(p) for p in paragraphs]
        chunks: list[Chunk] = []
        sources_uri = source or section.citation.source_uri

        i = 0
        char_offset = section.citation.char_start
        while i < len(paragraphs):
            current_paras: list[str] = []
            current_tokens = 0
            j = i
            while j < len(paragraphs):
                p_tokens = para_tokens[j]
                if current_tokens + p_tokens > cfg.target_tokens and current_paras:
                    break
                if current_tokens + p_tokens > cfg.max_tokens and current_paras:
                    break
                current_paras.append(paragraphs[j])
                current_tokens += p_tokens
                j += 1
                if current_tokens >= cfg.target_tokens:
                    break

            if not current_paras:
                # A single paragraph blew through max_tokens; split by
                # sentence to fit.
                long_paragraph = paragraphs[i]
                for sub_text, sub_tokens in self._split_long_paragraph(
                    long_paragraph, cfg,
                ):
                    chunks.append(
                        self._make_chunk(
                            sub_text, sub_tokens, char_offset, section,
                            data_type=data_type, source=sources_uri,
                            tier=tier, language=language,
                        )
                    )
                    char_offset += len(sub_text)
                i += 1
                continue

            chunk_text = "\n\n".join(current_paras)
            tokens = current_tokens
            if tokens >= cfg.min_tokens or len(chunks) == 0:
                chunks.append(
                    self._make_chunk(
                        chunk_text, tokens, char_offset, section,
                        data_type=data_type, source=sources_uri,
                        tier=tier, language=language,
                    )
                )

            # Move forward by ``j - i`` paragraphs minus the overlap.
            # Count paragraphs while accumulated tokens < overlap,
            # *including* the boundary paragraph that crosses the
            # threshold so consecutive chunks share content.
            consumed = j - i
            overlap_paragraphs = 0
            running = 0
            if cfg.overlap_tokens > 0:
                for k in range(j - 1, i - 1, -1):
                    overlap_paragraphs += 1
                    running += para_tokens[k]
                    if running >= cfg.overlap_tokens:
                        break
                # Cap so we always make progress (advance ≥ 1).
                if overlap_paragraphs >= consumed:
                    overlap_paragraphs = consumed - 1
            advance = max(1, consumed - overlap_paragraphs)
            char_offset += sum(len(p) + 2 for p in paragraphs[i : i + advance])
            i += advance

        return tuple(chunks)

    def _make_chunk(
        self,
        text: str,
        tokens: int,
        char_offset: int,
        section: ParsedSection,
        *,
        data_type: str,
        source: str,
        tier: CorpusTier,
        language: str | None,
    ) -> Chunk:
        end = min(char_offset + len(text), section.citation.char_end)
        return Chunk(
            text=text,
            token_count=tokens,
            section_path=section.section_path,
            citation=CitationSpan(
                source_uri=section.citation.source_uri,
                section_path=section.section_path,
                char_start=char_offset,
                char_end=end,
                page_number=section.citation.page_number,
            ),
            data_type=data_type,
            source=source,
            tier=tier,
            language=language,
            extra=_chunk_extra_for(section, text),
        )

    def _split_long_paragraph(
        self, paragraph: str, cfg: ChunkerConfig,
    ) -> Iterable[tuple[str, int]]:
        """Sentence-aware split for a paragraph above ``max_tokens``."""

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        buffer: list[str] = []
        running = 0
        for s in sentences:
            t = self._count(s)
            if running + t > cfg.target_tokens and buffer:
                yield (" ".join(buffer), running)
                buffer = []
                running = 0
            buffer.append(s)
            running += t
            if running >= cfg.max_tokens:
                yield (" ".join(buffer), running)
                buffer = []
                running = 0
        if buffer and running >= cfg.min_tokens:
            yield (" ".join(buffer), running)


class MarkdownChunker(ProseChunker):
    """Paragraph-aware chunker that keeps Markdown blocks atomic.

    Sibling of :class:`ProseChunker` for sections whose ``format``
    field is ``"markdown"`` (Mistral OCR, Anthropic, Marker, Docling,
    MinerU outputs). Three constructs MUST NOT be split mid-block —
    a half-table, a half-fenced-code-block, or a half-display-math
    expression is unreadable to both humans and downstream LLMs:

    1. Fenced code blocks (```` ```...``` ````, including
       language-tagged ```` ```python ... ``` ````, and the rarer
       tilde-fence ``~~~...~~~``).
    2. GFM tables — a header row of pipes followed by a separator
       row of pipes-and-dashes, followed by zero or more body rows.
    3. Display math — ``$$...$$`` blocks (multi-line) and the
       ``\\[...\\]`` LaTeX form.

    The chunker walks the input line-by-line, packs lines belonging
    to one of the above into a single "paragraph" (atomic unit), and
    splits prose on blank lines like the parent class. The result is
    fed to :meth:`ProseChunker.chunk`'s existing target-token
    packing loop unchanged.

    A block that exceeds ``max_tokens`` is preserved as a single
    oversized chunk rather than split — better an oversized chunk
    than a corrupted code listing or torn equation. Operators with
    pathological inputs can subclass and override
    :meth:`_split_paragraphs` to add their own splitter.
    """

    # ``^\s*`` allows leading indentation. Backtick / tilde fences
    # MUST balance: a ``` opens a block; the next ``` closes it.
    _FENCE_OPEN_RE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})")
    # GFM table separator row: ``|---|---|`` with optional alignment
    # colons and surrounding pipes.
    _TABLE_SEP_RE = re.compile(
        r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$",
    )
    # First row of a pipe table — at least one ``|`` separating cells.
    _TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$|^\s*[^|]+(\s*\|\s*[^|]+)+\s*$")
    # Display-math openers. ``$$`` may stand alone on a line or open a
    # multi-line block; ``\[`` opens, ``\]`` closes.
    _DISPLAY_MATH_DOLLAR_RE = re.compile(r"^\s*\$\$\s*$")
    _DISPLAY_MATH_BRACKET_OPEN_RE = re.compile(r"^\s*\\\[\s*$")
    _DISPLAY_MATH_BRACKET_CLOSE_RE = re.compile(r"^\s*\\\]\s*$")

    def _split_paragraphs(self, text: str) -> list[str]:
        lines = text.splitlines()
        out: list[str] = []
        prose: list[str] = []

        def flush_prose() -> None:
            if not prose:
                return
            joined = "\n".join(prose).strip()
            if joined:
                # Re-split prose on blank lines so paragraph-level
                # packing still works inside non-block runs. We
                # round-trip through the parent regex to keep the
                # one source of truth for "what is a paragraph
                # boundary."
                for piece in self._PARAGRAPH_RE.split(joined):
                    piece = piece.strip()
                    if piece:
                        out.append(piece)
            prose.clear()

        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]

            # 1. Fenced code blocks.
            m = self._FENCE_OPEN_RE.match(line)
            if m:
                fence = m.group("fence")
                flush_prose()
                block_lines = [line]
                i += 1
                # Close on a matching fence (same backticks /
                # tildes, possibly longer per CommonMark) — tolerate
                # an EOF without closer by treating the rest of the
                # file as the block.
                close_re = re.compile(r"^\s*" + re.escape(fence[0]) + r"{3,}\s*$")
                while i < n:
                    block_lines.append(lines[i])
                    if close_re.match(lines[i]):
                        i += 1
                        break
                    i += 1
                block = "\n".join(block_lines).strip()
                if block:
                    out.append(block)
                continue

            # 2. Display math: $$ on its own line.
            if self._DISPLAY_MATH_DOLLAR_RE.match(line):
                flush_prose()
                block_lines = [line]
                i += 1
                while i < n:
                    block_lines.append(lines[i])
                    if self._DISPLAY_MATH_DOLLAR_RE.match(lines[i]):
                        i += 1
                        break
                    i += 1
                block = "\n".join(block_lines).strip()
                if block:
                    out.append(block)
                continue

            # 2b. Display math: \[ ... \].
            if self._DISPLAY_MATH_BRACKET_OPEN_RE.match(line):
                flush_prose()
                block_lines = [line]
                i += 1
                while i < n:
                    block_lines.append(lines[i])
                    if self._DISPLAY_MATH_BRACKET_CLOSE_RE.match(lines[i]):
                        i += 1
                        break
                    i += 1
                block = "\n".join(block_lines).strip()
                if block:
                    out.append(block)
                continue

            # 3. GFM table — a row of pipes followed immediately by
            # a separator row. We require BOTH lines to identify a
            # table; a stray pipe in prose ("a|b") doesn't trigger.
            if (
                self._TABLE_ROW_RE.match(line)
                and i + 1 < n
                and self._TABLE_SEP_RE.match(lines[i + 1])
            ):
                flush_prose()
                block_lines = [line, lines[i + 1]]
                i += 2
                while i < n and self._TABLE_ROW_RE.match(lines[i]):
                    block_lines.append(lines[i])
                    i += 1
                block = "\n".join(block_lines).strip()
                if block:
                    out.append(block)
                continue

            # Default: accumulate into the current prose run.
            prose.append(line)
            i += 1

        flush_prose()
        return out


class CodeChunker:
    """Code chunker that delegates to the existing
    ``LanguageAwareTextChunker`` (``samples/paging/sharding/
    analyzers/semantic.py``).

    Reuse, not duplication: the chunker already handles function /
    class boundaries for every language colony's sharding layer
    supports. We wrap it to produce ``Chunk`` records with the right
    citation + metadata.
    """

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._config = config or ChunkerConfig()
        self._count = token_counter or default_token_counter()

    def chunk(
        self,
        section: ParsedSection,
        *,
        language: str = "text",
        data_type: str = "code",
        source: str | None = None,
        tier: CorpusTier = CorpusTier.UNTIERED,
    ) -> Sequence[Chunk]:
        text = section.text
        if not text or not text.strip():
            return ()

        sources_uri = source or section.citation.source_uri
        chunks_text = list(self._delegate_chunk(text, language))
        if not chunks_text:
            # Delegate may drop everything for short inputs (its
            # ``min_chunk_size`` is sized for production code files).
            # Fall back to the sliding-window primitive so short
            # source files still produce at least one chunk.
            chunks_text = list(self._fallback_sliding_window(text))
        if not chunks_text:
            return ()

        results: list[Chunk] = []
        char_offset = section.citation.char_start
        for raw in chunks_text:
            tokens = self._count(raw)
            end = min(char_offset + len(raw), section.citation.char_end)
            results.append(
                Chunk(
                    text=raw,
                    token_count=tokens,
                    section_path=section.section_path,
                    citation=CitationSpan(
                        source_uri=section.citation.source_uri,
                        section_path=section.section_path,
                        char_start=char_offset,
                        char_end=end,
                        page_number=section.citation.page_number,
                    ),
                    data_type=data_type,
                    source=sources_uri,
                    tier=tier,
                    language=language,
                    extra=_chunk_extra_for(section, raw),
                )
            )
            char_offset = end
        return tuple(results)

    def _delegate_chunk(self, text: str, language: str) -> Iterable[str]:
        """Run ``LanguageAwareTextChunker`` synchronously.

        That class is async-initialised because its config is loaded
        from the colony component registry; for in-process use we call
        the underlying ``_chunk_by_code_blocks`` / ``_chunk_by_sliding_
        window`` methods directly with a config snapshot. This avoids
        bringing up the async config-loader machinery in unit tests.
        """

        try:
            from ..samples.paging.sharding.analyzers.semantic import (  # type: ignore
                ChunkingConfig as _ShardingChunkingConfig,
                LanguageAwareTextChunker as _LangAwareChunker,
            )
        except ImportError:
            yield from self._fallback_sliding_window(text)
            return

        chunker = _LangAwareChunker(language=language)
        # Skip async initialise: hand it a config object directly.
        try:
            chunker.config = _ShardingChunkingConfig()
        except Exception:  # noqa: BLE001
            yield from self._fallback_sliding_window(text)
            return
        try:
            yield from chunker.chunk_content(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CodeChunker: delegate failed (%s); falling back to sliding window.",
                exc,
            )
            yield from self._fallback_sliding_window(text)

    def _fallback_sliding_window(self, text: str) -> Iterable[str]:
        cfg = self._config
        # Token budget → approximate char budget at 4 chars/token.
        char_budget = cfg.target_tokens * 4
        overlap_chars = cfg.overlap_tokens * 4
        start = 0
        while start < len(text):
            end = min(start + char_budget, len(text))
            if end < len(text):
                # Prefer line breaks.
                line_break = text.rfind("\n", start, end)
                if line_break > start:
                    end = line_break
            piece = text[start:end]
            if piece.strip():
                yield piece
            if end >= len(text):
                break
            start = max(end - overlap_chars, start + 1)


__all__ = (
    "ChunkerConfig",
    "ProseChunker",
    "MarkdownChunker",
    "CodeChunker",
    "TokenCounter",
    "default_token_counter",
)
