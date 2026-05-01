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


logger = logging.getLogger(__name__)


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
        paragraphs = [p.strip() for p in self._PARAGRAPH_RE.split(text) if p.strip()]
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
    "CodeChunker",
    "TokenCounter",
    "default_token_counter",
)
