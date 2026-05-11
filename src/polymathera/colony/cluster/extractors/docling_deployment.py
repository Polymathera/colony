"""Docling self-hosted PDF extractor as a Ray serving deployment.

`Docling <https://github.com/docling-project/docling>`_ is IBM
Research's layout-aware document converter — strong on tables via
TableFormer, ships with first-class LangChain / LlamaIndex
integrations (which we don't use directly, but the underlying
extraction quality benefits the same way).

MIT-licensed — included in the default ``knowledge`` poetry extra
(no build flag) so the operator can flip
``knowledge.pdf_extractor.backend: docling`` in
YAML without a rebuild.

Library imports are lazy so a Ray worker without the Docling
package can still bring up the agent system; the failure surfaces
as a typed :class:`PdfExtractorError` only when the operator
actually invokes the deployment.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

from ...distributed.ray_utils import serving

from .base import (
    ExtractOptions,
    ExtractResult,
    FigureBlob,
    PdfExtractorDeployment,
    PdfExtractorError,
)


logger = logging.getLogger(__name__)


@serving.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_queue_length": 2,
    },
    ray_actor_options={"num_gpus": 0},
)
class DoclingExtractorDeployment(PdfExtractorDeployment):
    """Docling-backed PDF extractor."""

    backend_name = "docling"

    def __init__(
        self,
        *,
        do_ocr: bool = True,
        do_table_structure: bool = True,
    ) -> None:
        # Capture config; defer the Docling pipeline build to first
        # call. Building the pipeline triggers model downloads on
        # first run, which can take a minute and is wasted work if
        # nothing ever calls extract.
        self._do_ocr = bool(do_ocr)
        self._do_table_structure = bool(do_table_structure)
        self._converter: Any | None = None
        self._init_lock = asyncio.Lock()

    async def _ensure_converter(self) -> None:
        if self._converter is not None:
            return
        async with self._init_lock:
            if self._converter is not None:
                return

            def _build() -> Any:
                try:
                    from docling.datamodel.base_models import (  # type: ignore[import-not-found]
                        InputFormat,
                    )
                    from docling.datamodel.pipeline_options import (  # type: ignore[import-not-found]
                        PdfPipelineOptions,
                    )
                    from docling.document_converter import (  # type: ignore[import-not-found]
                        DocumentConverter,
                        PdfFormatOption,
                    )
                except ImportError as exc:
                    raise PdfExtractorError(
                        "DoclingExtractorDeployment requires the 'docling' "
                        "package. Install via the knowledge poetry extra "
                        "(`poetry install --extras knowledge`).",
                    ) from exc

                opts = PdfPipelineOptions()
                opts.do_ocr = self._do_ocr
                opts.do_table_structure = self._do_table_structure
                # Generate page-level images so figure extraction has
                # bytes to work with — Docling's default is to skip
                # image generation for performance.
                if hasattr(opts, "generate_picture_images"):
                    opts.generate_picture_images = True

                logger.info(
                    "DoclingExtractorDeployment: building DocumentConverter "
                    "(do_ocr=%s, do_table_structure=%s)",
                    self._do_ocr, self._do_table_structure,
                )
                return DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
                    },
                )

            self._converter = await asyncio.to_thread(_build)

    @serving.endpoint
    async def extract(
        self,
        *,
        pdf_bytes: bytes,
        options: ExtractOptions | None = None,
    ) -> ExtractResult:
        await self._ensure_converter()
        opts = options or ExtractOptions()

        def _convert() -> tuple[str, list[FigureBlob], int]:
            try:
                from docling.datamodel.base_models import (  # type: ignore[import-not-found]
                    DocumentStream,
                    InputFormat,
                )
            except ImportError as exc:
                raise PdfExtractorError(
                    "DoclingExtractorDeployment: docling import failed at "
                    "extract time (incompatible version?).",
                ) from exc

            buf = io.BytesIO(pdf_bytes)
            stream = DocumentStream(name="document.pdf", stream=buf)
            try:
                result = self._converter.convert(
                    source=stream, raises_on_error=True,
                )
            except Exception as exc:  # noqa: BLE001
                raise PdfExtractorError(
                    f"DoclingExtractorDeployment: convert failed: "
                    f"{type(exc).__name__}: {exc}",
                ) from exc

            doc = result.document
            markdown = doc.export_to_markdown()
            figures = self._figures_from_docling(doc)
            page_count = len(getattr(doc, "pages", {}) or {})
            return markdown, figures, page_count

        try:
            markdown, figures, page_count = await asyncio.to_thread(_convert)
        except PdfExtractorError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PdfExtractorError(
                f"DoclingExtractorDeployment.extract failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        return ExtractResult(
            markdown=markdown or "",
            figures=tuple(figures),
            backend=self.backend_name,
            page_count=page_count,
        )

    @staticmethod
    def _figures_from_docling(document: Any) -> list[FigureBlob]:
        """Walk ``document.pictures`` and re-encode each picture as
        PNG bytes. Docling pictures expose either a PIL ``image`` or
        a ``self_ref`` URI to the rendered tile — we handle both.
        """
        try:
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise PdfExtractorError(
                "DoclingExtractorDeployment: Pillow is required to "
                "re-encode extracted pictures.",
            ) from exc

        pictures = getattr(document, "pictures", None) or ()
        blobs: list[FigureBlob] = []
        for idx, picture in enumerate(pictures):
            image = getattr(picture, "image", None)
            pil_image = None
            if image is not None:
                pil_image = getattr(image, "pil_image", None)
            # Newer Docling versions expose ``get_image(doc=...)``.
            if pil_image is None and hasattr(picture, "get_image"):
                try:
                    pil_image = picture.get_image(doc=document)
                except Exception:  # noqa: BLE001
                    pil_image = None
            if pil_image is None or not isinstance(pil_image, Image.Image):
                continue
            buf = io.BytesIO()
            try:
                pil_image.save(buf, format="PNG")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "DoclingExtractorDeployment: failed to encode picture "
                    "%d as PNG (%s); skipping",
                    idx, exc,
                )
                continue
            page = None
            prov = getattr(picture, "prov", None) or ()
            if prov:
                page_no_attr = getattr(prov[0], "page_no", None)
                if isinstance(page_no_attr, int):
                    page = page_no_attr
            blob_id = (
                getattr(picture, "self_ref", None) or f"picture-{idx}"
            )
            blobs.append(
                FigureBlob(
                    blob_id=str(blob_id),
                    image_bytes=buf.getvalue(),
                    mime="image/png",
                    page=page,
                    label=str(blob_id),
                    kind="figure",
                ),
            )
        return blobs


__all__ = ("DoclingExtractorDeployment",)
