"""Marker self-hosted PDF extractor as a Ray serving deployment.

`Marker <https://github.com/datalab-to/marker>`_ converts PDFs (and
PPTX / DOCX / XLSX / EPUB) to Markdown / JSON with strong equation
preservation via Texify. CPU-capable (slow: ~5-15 s/page); pinned
to a single GPU per replica in production. Runs as one
:class:`PdfExtractorDeployment` per replica so VCM can autoscale
based on extraction queue length.

GPL-3.0 license — Marker stays behind the ``knowledge_marker``
poetry extra (operator decision #6 in the design doc) and is NOT
in the default ``colony:local`` image. To opt in, add the extra to
the ``poetry install`` line in ``Dockerfile.local`` and rebuild.

The deployment is **lazy** about the heavy ML imports: ``marker``
and its model artefacts only load on the first ``extract`` call.
Construction is cheap so the cluster can deploy this as a sibling
of Docling / MinerU and route per ``cluster_config.yaml``.
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
    ray_actor_options={
        # Default to CPU; operators with GPUs override via the
        # cluster config's ``ray_actor_options.num_gpus`` field.
        "num_gpus": 0,
    },
)
class MarkerExtractorDeployment(PdfExtractorDeployment):
    """Marker-backed PDF extractor."""

    backend_name = "marker"

    def __init__(
        self,
        *,
        max_pages: int | None = None,
        langs: list[str] | None = None,
        batch_multiplier: int = 1,
    ) -> None:
        # Capture config; defer the heavy model load to first call.
        self._max_pages = max_pages
        self._langs = list(langs or [])
        self._batch_multiplier = int(batch_multiplier)
        self._model_lst: Any | None = None
        self._init_lock = asyncio.Lock()

    async def _ensure_models(self) -> None:
        if self._model_lst is not None:
            return
        async with self._init_lock:
            if self._model_lst is not None:
                return

            def _load() -> Any:
                try:
                    from marker.models import create_model_dict  # type: ignore[import-not-found]
                except ImportError as exc:
                    raise PdfExtractorError(
                        "MarkerExtractorDeployment requires the 'marker-pdf' "
                        "package. Install via the knowledge_marker poetry "
                        "extra (`poetry install --extras knowledge_marker`).",
                    ) from exc
                logger.info("MarkerExtractorDeployment: loading marker models…")
                return create_model_dict()

            self._model_lst = await asyncio.to_thread(_load)

    @serving.endpoint
    async def extract(
        self,
        *,
        pdf_bytes: bytes,
        options: ExtractOptions | None = None,
    ) -> ExtractResult:
        await self._ensure_models()
        opts = options or ExtractOptions()

        def _convert() -> tuple[str, dict[str, Any], dict[str, Any]]:
            try:
                from marker.converters.pdf import PdfConverter  # type: ignore[import-not-found]
                from marker.output import text_from_rendered  # type: ignore[import-not-found]
            except ImportError as exc:
                raise PdfExtractorError(
                    "MarkerExtractorDeployment: marker.converters.pdf import "
                    "failed (incompatible marker-pdf version?).",
                ) from exc

            converter = PdfConverter(
                artifact_dict=self._model_lst,
                config={
                    "max_pages": self._max_pages,
                    "languages": self._langs or None,
                    "batch_multiplier": self._batch_multiplier,
                    "output_format": "markdown",
                },
            )
            rendered = converter(io.BytesIO(pdf_bytes))
            text, _, images = text_from_rendered(rendered)
            metadata = getattr(rendered, "metadata", {}) or {}
            return text, images, metadata

        try:
            markdown, images, metadata = await asyncio.to_thread(_convert)
        except PdfExtractorError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PdfExtractorError(
                f"MarkerExtractorDeployment.extract failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        figures = self._figures_from_marker_images(images)
        return ExtractResult(
            markdown=markdown or "",
            figures=tuple(figures),
            backend=self.backend_name,
            page_count=int(metadata.get("page_count") or 0),
            extra={"raw_metadata": metadata},
        )

    @staticmethod
    def _figures_from_marker_images(
        images: dict[str, Any] | None,
    ) -> list[FigureBlob]:
        """Marker's renderer returns images as ``{filename: PIL.Image}``.
        We re-encode each image as PNG bytes (the most universally
        renderable mime) and build a :class:`FigureBlob` keyed by
        filename — the reader uses that filename to rewrite markdown
        image references to ``colony-image://<sha>``.
        """
        if not images:
            return []
        try:
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise PdfExtractorError(
                "MarkerExtractorDeployment: Pillow is required to "
                "re-encode extracted images.",
            ) from exc

        blobs: list[FigureBlob] = []
        for filename, image in images.items():
            if not isinstance(image, Image.Image):
                continue
            buf = io.BytesIO()
            try:
                image.save(buf, format="PNG")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "MarkerExtractorDeployment: failed to encode %s as PNG "
                    "(%s); skipping figure",
                    filename, exc,
                )
                continue
            blobs.append(
                FigureBlob(
                    blob_id=str(filename),
                    image_bytes=buf.getvalue(),
                    mime="image/png",
                    label=str(filename),
                    kind="figure",
                ),
            )
        return blobs


__all__ = ("MarkerExtractorDeployment",)
