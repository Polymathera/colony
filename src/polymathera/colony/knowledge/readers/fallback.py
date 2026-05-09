"""``FallbackPdfReader`` — primary reader with a typed-error fallback.

Today's only consumer is the Mistral OCR 1000-page hard limit:
operator picks ``mistral_ocr`` for cost, but DICOM-class documents
(2000+ pages) blow past Mistral's cap and fail with a typed
:class:`PdfTooManyPagesError`. Wrapping the primary in
``FallbackPdfReader(primary, fallback=GeminiPdfReader(...))`` routes
those documents to a backend without that limit instead of failing
the ingest.

The fallback only fires on :class:`PdfTooManyPagesError`. Other
``FormatReaderError`` subclasses (auth failures, malformed
responses, etc.) propagate unchanged — those genuinely indicate the
ingest should fail rather than silently swap backends.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from ..models import ParsedSection, RawDocument
from .base import FormatReader, PdfTooManyPagesError


logger = logging.getLogger(__name__)


class FallbackPdfReader(FormatReader):
    """Wrap two readers; route to ``fallback`` only on
    :class:`PdfTooManyPagesError`.

    ``handles`` is inherited from ``primary`` so the registry indexes
    the wrapper exactly like the primary would have been indexed.
    """

    def __init__(
        self, primary: FormatReader, fallback: FormatReader,
    ) -> None:
        super().__init__(handles=primary.handles)
        self._primary = primary
        self._fallback = fallback

    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        # Sync path: most multimodal readers override ``read_async``
        # directly and leave ``read`` raising. Defer to ``read_async``
        # via the framework's worker-thread bridge so callers can use
        # either entry point.
        try:
            return self._primary.read(document)
        except PdfTooManyPagesError as exc:
            logger.info(
                "FallbackPdfReader: %s rejected %s (%d > %d pages); "
                "routing to %s",
                type(self._primary).__name__, document.source_uri,
                exc.page_count or -1, exc.max_pages or -1,
                type(self._fallback).__name__,
            )
            return self._fallback.read(document)

    async def read_async(
        self, document: RawDocument,
    ) -> Sequence[ParsedSection]:
        try:
            return await self._primary.read_async(document)
        except PdfTooManyPagesError as exc:
            logger.info(
                "FallbackPdfReader: %s rejected %s (%s > %s pages); "
                "routing to %s",
                type(self._primary).__name__, document.source_uri,
                exc.page_count, exc.max_pages,
                type(self._fallback).__name__,
            )
            return await self._fallback.read_async(document)


__all__ = ("FallbackPdfReader",)
