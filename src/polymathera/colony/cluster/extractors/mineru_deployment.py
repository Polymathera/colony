"""MinerU self-hosted PDF extractor as a Ray serving deployment.

`MinerU <https://github.com/opendatalab/MinerU>`_ (OpenDataLab,
Shanghai AI Lab) is the strongest open-source extractor for
visually-dense / complex layouts and CJK content. Backed by
PaddleOCR and OpenDataLab's layout models.

AGPL-3.0 — included in the default ``knowledge`` poetry extra
(decision #6) since the colony framework itself is Apache-2.0 and
the AGPL terms only constrain network-served modifications, which
are an operator-deployment-time concern.

Lazy-imports its ML backend (the ``magic_pdf`` package, MinerU's
underlying engine) on first call. Construction stays cheap so the
deployment can sit dormant until the operator flips
``knowledge.pdf_extractor.backend: mineru`` in
YAML.
"""

from __future__ import annotations

import asyncio
import io
import logging
import shutil
import tempfile
from pathlib import Path
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
class MinerUExtractorDeployment(PdfExtractorDeployment):
    """MinerU-backed PDF extractor."""

    backend_name = "mineru"

    def __init__(self, *, parse_method: str = "auto") -> None:
        # ``parse_method`` is MinerU's pick between ``ocr`` (run OCR
        # on every page), ``txt`` (trust the embedded text layer),
        # or ``auto`` (use OCR only when the text layer is empty).
        # ``auto`` is the right default for mixed corpora.
        self._parse_method = parse_method
        self._initialised = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialised(self) -> None:
        if self._initialised:
            return
        async with self._init_lock:
            if self._initialised:
                return

            def _import() -> None:
                try:
                    import magic_pdf  # type: ignore[import-not-found]  # noqa: F401
                    from magic_pdf.config.constants import (  # type: ignore[import-not-found]
                        SupportedPdfParseMethod,  # noqa: F401
                    )
                except ImportError as exc:
                    raise PdfExtractorError(
                        "MinerUExtractorDeployment requires the 'magic-pdf' "
                        "package (MinerU). Install via the knowledge poetry "
                        "extra (`poetry install --extras knowledge`).",
                    ) from exc
                logger.info("MinerUExtractorDeployment: magic_pdf importable.")

            await asyncio.to_thread(_import)
            self._initialised = True

    @serving.endpoint
    async def extract(
        self,
        *,
        pdf_bytes: bytes,
        options: ExtractOptions | None = None,
    ) -> ExtractResult:
        await self._ensure_initialised()
        opts = options or ExtractOptions()

        def _convert() -> tuple[str, list[FigureBlob], int]:
            try:
                from magic_pdf.data.dataset import PymuDocDataset  # type: ignore[import-not-found]
                from magic_pdf.data.read_api import read_local_pdfs  # type: ignore[import-not-found]
                from magic_pdf.model.doc_analyze_by_custom_model import (  # type: ignore[import-not-found]
                    doc_analyze,
                )
                from magic_pdf.config.constants import (  # type: ignore[import-not-found]
                    SupportedPdfParseMethod,
                )
            except ImportError as exc:
                raise PdfExtractorError(
                    "MinerUExtractorDeployment: magic_pdf import failed at "
                    "extract time (incompatible version?).",
                ) from exc

            # MinerU's API expects an on-disk path + a working
            # directory it can scribble images into. Use a private
            # tempdir per call so concurrent extracts don't collide.
            with tempfile.TemporaryDirectory(prefix="mineru-") as workdir:
                workdir_path = Path(workdir)
                pdf_path = workdir_path / "input.pdf"
                pdf_path.write_bytes(pdf_bytes)
                images_dir = workdir_path / "images"
                images_dir.mkdir(exist_ok=True)

                # Resolve the parse method enum from our string
                # config. ``auto`` is the safe default.
                method_map = {
                    "auto": SupportedPdfParseMethod.OCR,
                    "ocr": SupportedPdfParseMethod.OCR,
                    "txt": SupportedPdfParseMethod.TXT,
                }
                method = method_map.get(
                    self._parse_method, SupportedPdfParseMethod.OCR,
                )

                try:
                    datasets = read_local_pdfs(str(pdf_path))
                    if not datasets:
                        raise PdfExtractorError(
                            "MinerUExtractorDeployment: read_local_pdfs "
                            "returned empty dataset list",
                        )
                    dataset: PymuDocDataset = datasets[0]
                    inference = doc_analyze(
                        dataset, ocr=(method == SupportedPdfParseMethod.OCR),
                    )
                    pipe = inference.pipe_ocr_mode(images_dir) if (
                        method == SupportedPdfParseMethod.OCR
                    ) else inference.pipe_txt_mode(images_dir)
                    markdown = pipe.get_markdown(
                        images_dir.relative_to(workdir_path).as_posix() + "/",
                    )
                except PdfExtractorError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise PdfExtractorError(
                        f"MinerUExtractorDeployment: pipeline failed: "
                        f"{type(exc).__name__}: {exc}",
                    ) from exc

                # Collect all extracted images from the working
                # directory before the tempdir disappears. Order by
                # filename so callers see a stable enumeration.
                figures: list[FigureBlob] = []
                for image_path in sorted(images_dir.iterdir()):
                    if not image_path.is_file():
                        continue
                    try:
                        payload = image_path.read_bytes()
                    except OSError as exc:
                        logger.warning(
                            "MinerUExtractorDeployment: failed to read "
                            "extracted image %s (%s)", image_path, exc,
                        )
                        continue
                    mime = _mime_for_suffix(image_path.suffix)
                    figures.append(
                        FigureBlob(
                            blob_id=image_path.name,
                            image_bytes=payload,
                            mime=mime,
                            label=image_path.name,
                            kind="figure",
                        ),
                    )
                page_count = len(getattr(dataset, "data_list", []) or [])

            return markdown or "", figures, page_count

        try:
            markdown, figures, page_count = await asyncio.to_thread(_convert)
        except PdfExtractorError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PdfExtractorError(
                f"MinerUExtractorDeployment.extract failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        return ExtractResult(
            markdown=markdown,
            figures=tuple(figures),
            backend=self.backend_name,
            page_count=page_count,
        )


def _mime_for_suffix(suffix: str) -> str:
    """Map a filename suffix to a sensible mime. Falls back to PNG
    for unknown suffixes — MinerU emits PNGs by default."""
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix.lower(), "image/png")


__all__ = ("MinerUExtractorDeployment",)
