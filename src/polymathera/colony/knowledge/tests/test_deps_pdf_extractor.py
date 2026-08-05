"""Tests for the config-driven PDF extractor selection in
``knowledge.deps``.

The contract: ``set_knowledge_deps()`` reads
:class:`KnowledgeConfig.pdf_extractor` from the typed config tree
(via :func:`get_component_or_default`) and wires the singleton
:class:`Ingestor` with a reader registry whose PDF reader matches
the configured backend. When registry construction fails (unknown
backend, missing extras), the framework falls back to the in-process
:class:`PdfReader`.

Tests inject the typed config by patching
``knowledge.deps._knowledge_config`` — the same path the runtime
takes via :func:`get_component_or_default`, only short-circuited so
unit tests don't need to boot the global ``ConfigurationManager``.
"""

from __future__ import annotations

import pytest

from polymathera.colony.knowledge import deps as deps_module
from polymathera.colony.knowledge.cluster_config import (
    KnowledgeConfig,
    PdfExtractorConfig,
)
from polymathera.colony.knowledge.deps import (
    get_default_ingestor,
    reset_knowledge_deps,
    set_knowledge_deps,
)
from polymathera.colony.knowledge.models import KnowledgeFormat
from polymathera.colony.knowledge.readers import (
    MistralOcrPdfReader,
    PdfReader,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_knowledge_deps()
    yield
    reset_knowledge_deps()


def _patch_config(monkeypatch: pytest.MonkeyPatch, cfg: KnowledgeConfig) -> None:
    monkeypatch.setattr(deps_module, "_knowledge_config", lambda: cfg)


def test_default_pdf_reader_is_mistral_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default ``KnowledgeConfig`` selects ``mistral_ocr`` —
    the cheap, multimodal default. (Was text-only ``PdfReader`` in
    iteration 2; iteration 3 shipped the multimodal default.)"""
    _patch_config(monkeypatch, KnowledgeConfig())
    set_knowledge_deps()
    ing = get_default_ingestor()
    pdf_reader = ing._readers.reader_for(KnowledgeFormat.PDF)
    assert isinstance(pdf_reader, MistralOcrPdfReader)


def test_pdf_extractor_mistral_ocr_swaps_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(monkeypatch, KnowledgeConfig(
        pdf_extractor=PdfExtractorConfig(backend="mistral_ocr"),
    ))
    set_knowledge_deps()
    ing = get_default_ingestor()
    pdf_reader = ing._readers.reader_for(KnowledgeFormat.PDF)
    assert isinstance(pdf_reader, MistralOcrPdfReader)


def test_pdf_extractor_wires_image_store_into_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(monkeypatch, KnowledgeConfig(
        pdf_extractor=PdfExtractorConfig(backend="mistral_ocr"),
    ))
    deps = set_knowledge_deps()
    ing = get_default_ingestor()
    pdf_reader = ing._readers.reader_for(KnowledgeFormat.PDF)
    assert isinstance(pdf_reader, MistralOcrPdfReader)
    # The reader's image_store must be the same instance as the
    # singleton's so figures land where retrieval expects them.
    assert pdf_reader._image_store is deps.image_store


def test_registry_construction_failure_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the operator-configured backend cannot be honoured
    (missing extras, unknown name), bring-up FAILS LOUD instead of
    silently degrading every PDF to the free text-only reader —
    a degraded ingest would also poison the skip-caches."""
    import polymathera.colony.knowledge.readers as readers_module

    _patch_config(monkeypatch, KnowledgeConfig(
        pdf_extractor=PdfExtractorConfig(backend="mistral_ocr"),
    ))

    def _unbuildable(**_kw):
        raise ValueError("mistralai extras not installed")

    monkeypatch.setattr(
        readers_module, "default_registry_with_pdf_extractor", _unbuildable,
    )
    with pytest.raises(RuntimeError, match="could not be honoured"):
        set_knowledge_deps()


def test_pdf_extractor_self_hosted_wires_remote_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A self-hosted backend (marker / docling / mineru) routes
    through :class:`RemotePdfExtractorReader` — the deployment-
    backed generic reader. The reader resolves the
    ``DeploymentHandle`` lazily on first use, so this test
    succeeds without the underlying ML libraries installed: it
    only asserts the registry registered the right reader class."""
    from polymathera.colony.knowledge.readers import RemotePdfExtractorReader

    _patch_config(monkeypatch, KnowledgeConfig(
        pdf_extractor=PdfExtractorConfig(backend="marker"),
    ))
    set_knowledge_deps()
    ing = get_default_ingestor()
    pdf_reader = ing._readers.reader_for(KnowledgeFormat.PDF)
    assert isinstance(pdf_reader, RemotePdfExtractorReader)
    assert pdf_reader.backend == "marker"


def test_pdf_extractor_options_forwarded_as_backend_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pdf_extractor.options`` is forwarded as ``backend_kwargs``
    to the reader constructor — the operator's tier knob (e.g.
    ``model: gemini-2.5-pro``) reaches the reader without code
    changes."""
    _patch_config(monkeypatch, KnowledgeConfig(
        pdf_extractor=PdfExtractorConfig(
            backend="mistral_ocr",
            options={"model": "mistral-ocr-2509", "table_format": "markdown"},
        ),
    ))
    set_knowledge_deps()
    ing = get_default_ingestor()
    pdf_reader = ing._readers.reader_for(KnowledgeFormat.PDF)
    assert isinstance(pdf_reader, MistralOcrPdfReader)
    assert pdf_reader._model == "mistral-ocr-2509"
