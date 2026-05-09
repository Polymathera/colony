"""Self-hosted PDF extractor deployments.

The :class:`PdfExtractorDeployment` ABC is the contract every
self-hosted layout-aware extractor (Marker, Docling, MinerU)
implements. Hosted-API readers
(:class:`~polymathera.colony.knowledge.readers.mistral_ocr_pdf.MistralOcrPdfReader`,
:class:`~polymathera.colony.knowledge.readers.anthropic_pdf.AnthropicPdfReader`,
:class:`~polymathera.colony.knowledge.readers.gemini_pdf.GeminiPdfReader`,
:class:`~polymathera.colony.knowledge.readers.llamaparse_pdf.LlamaParsePdfReader`)
do **not** instantiate a deployment — they hit the vendor's HTTP
endpoint directly — but they reuse the same
:class:`ExtractResult` / :class:`FigureBlob` / :class:`ExtractOptions`
types so reader code stays uniform across backends.

The three concrete deployments are imported lazily so a process
that doesn't deploy them (the dashboard, a worker pool that only
runs the hosted readers) doesn't pay the import cost. The lazy
proxies preserve `MarkerExtractorDeployment` etc. as importable
symbols for type hints and ``cluster_config.yaml`` resolution.

See :doc:`/architecture/multimodal-pdf-ingestion` for the design.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .base import (
    ExtractOptions,
    ExtractResult,
    FigureBlob,
    PdfExtractorDeployment,
    PdfExtractorError,
)


if TYPE_CHECKING:  # pragma: no cover - import-only for type hints
    from .docling_deployment import DoclingExtractorDeployment
    from .marker_deployment import MarkerExtractorDeployment
    from .mineru_deployment import MinerUExtractorDeployment


# Map of backend name → (module-relative-name, class-name). Used by
# the cluster config resolver to instantiate the right deployment
# at startup without ever importing the libraries the operator did
# not opt into.
_DEPLOYMENT_REGISTRY: dict[str, tuple[str, str]] = {
    "marker": (".marker_deployment", "MarkerExtractorDeployment"),
    "docling": (".docling_deployment", "DoclingExtractorDeployment"),
    "mineru": (".mineru_deployment", "MinerUExtractorDeployment"),
}


def get_deployment_class(backend: str) -> type[PdfExtractorDeployment]:
    """Resolve a backend name to its deployment class via lazy import.

    Raises :class:`PdfExtractorError` with a clear message if the
    backend is unknown or its underlying library is missing.
    """
    entry = _DEPLOYMENT_REGISTRY.get(backend)
    if entry is None:
        raise PdfExtractorError(
            f"unknown self-hosted PDF extractor backend {backend!r}; "
            f"choose one of {sorted(_DEPLOYMENT_REGISTRY)}",
        )
    module_name, class_name = entry
    try:
        module = importlib.import_module(module_name, package=__name__)
    except ImportError as exc:
        raise PdfExtractorError(
            f"{backend!r} deployment requires its underlying library; "
            f"import failed: {exc}",
        ) from exc
    return getattr(module, class_name)


def __getattr__(name: str) -> Any:
    """PEP 562 lazy module attribute — allows ``from
    polymathera.colony.cluster.extractors import MarkerExtractorDeployment``
    without paying the marker / docling / mineru import cost up
    front. Each subclass is loaded only when actually referenced.
    """
    for backend, (module_name, class_name) in _DEPLOYMENT_REGISTRY.items():
        if name == class_name:
            module = importlib.import_module(module_name, package=__name__)
            return getattr(module, class_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    "ExtractOptions",
    "ExtractResult",
    "FigureBlob",
    "PdfExtractorDeployment",
    "PdfExtractorError",
    "DoclingExtractorDeployment",
    "MarkerExtractorDeployment",
    "MinerUExtractorDeployment",
    "get_deployment_class",
)
