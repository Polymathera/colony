"""Tests for :class:`KnowledgeConfig`, :class:`PdfExtractorConfig`,
:class:`QdrantConfig`, and :class:`GrobidConfig`.

Knowledge-stack knobs are typed fields on :class:`KnowledgeConfig`.
Workers re-read the same config from their own
:class:`ConfigurationManager` (loaded from ``POLYMATHERA_CONFIG``).
"""

from __future__ import annotations

import pytest

from polymathera.colony.knowledge.cluster_config import (
    GrobidConfig,
    KnowledgeConfig,
    PdfExtractorConfig,
    QdrantConfig,
)


# ---------------------------------------------------------------------------
# PdfExtractorConfig
# ---------------------------------------------------------------------------


def test_default_backend_is_mistral_ocr() -> None:
    cfg = PdfExtractorConfig()
    assert cfg.backend == "mistral_ocr"
    assert cfg.options == {}
    assert cfg.is_hosted()
    assert not cfg.is_self_hosted()


def test_hosted_backend_classification() -> None:
    for backend in ("mistral_ocr", "anthropic", "gemini", "llamaparse"):
        cfg = PdfExtractorConfig(backend=backend)
        assert cfg.is_hosted()
        assert not cfg.is_self_hosted()


def test_self_hosted_backend_classification() -> None:
    for backend in ("marker", "docling", "mineru"):
        cfg = PdfExtractorConfig(backend=backend)
        assert cfg.is_self_hosted()
        assert not cfg.is_hosted()


def test_options_dict_accepts_arbitrary_kwargs() -> None:
    """``options`` is the operator's pass-through to the reader's
    constructor — anything goes, schema-side validation lives in
    the reader."""
    cfg = PdfExtractorConfig(
        backend="llamaparse",
        options={"tier": "agentic", "timeout_s": 600.0},
    )
    assert cfg.options["tier"] == "agentic"
    assert cfg.options["timeout_s"] == 600.0


def test_extra_fields_rejected() -> None:
    """``extra="forbid"`` keeps the YAML schema honest — typos
    fail loudly instead of silently dropping config."""
    with pytest.raises(Exception):
        PdfExtractorConfig(typo_field="oops")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# QdrantConfig + GrobidConfig — typed sub-components
# ---------------------------------------------------------------------------


def test_qdrant_defaults_to_in_memory() -> None:
    """Empty ``url`` selects :class:`InMemoryVectorStore` — fine for
    tests / single-process dev. Operator opts in to Qdrant by setting
    the URL in YAML."""
    cfg = QdrantConfig()
    assert cfg.url == ""
    assert cfg.collection == "colony_knowledge"


def test_qdrant_extra_fields_rejected() -> None:
    with pytest.raises(Exception):
        QdrantConfig(typo_field="oops")  # type: ignore[call-arg]


def test_grobid_defaults_to_disabled() -> None:
    """Empty ``url`` keeps the text-only fallback reader. Operator
    opts in by setting the URL in YAML."""
    cfg = GrobidConfig()
    assert cfg.url == ""


# ---------------------------------------------------------------------------
# KnowledgeConfig — composition
# ---------------------------------------------------------------------------


def test_knowledge_config_default_shape() -> None:
    """Every nested component has a well-formed default; the whole
    tree is constructible without YAML (the bare-defaults path used
    by :func:`get_component_or_default` when the manager has not
    been initialized)."""
    cfg = KnowledgeConfig()
    assert isinstance(cfg.pdf_extractor, PdfExtractorConfig)
    assert isinstance(cfg.qdrant, QdrantConfig)
    assert isinstance(cfg.grobid, GrobidConfig)
    assert cfg.image_dir == ""
    assert cfg.qdrant.url == ""
    assert cfg.grobid.url == ""
    assert cfg.pdf_extractor.backend == "mistral_ocr"


def test_knowledge_config_from_yaml_shape() -> None:
    """Mirror the ``knowledge`` YAML shape an
    operator writes — single source of truth, no env vars."""
    cfg = KnowledgeConfig(
        pdf_extractor={
            "backend": "gemini",
            "options": {"model": "gemini-2.5-pro"},
        },
        image_dir="/mnt/shared/colony-images",
        qdrant={"url": "http://qdrant:6333", "collection": "colony_knowledge"},
        grobid={"url": "http://grobid:8070"},
    )
    assert cfg.pdf_extractor.backend == "gemini"
    assert cfg.pdf_extractor.options == {"model": "gemini-2.5-pro"}
    assert cfg.image_dir == "/mnt/shared/colony-images"
    assert cfg.qdrant.url == "http://qdrant:6333"
    assert cfg.qdrant.collection == "colony_knowledge"
    assert cfg.grobid.url == "http://grobid:8070"


# ---------------------------------------------------------------------------
# add_deployments_to_app — no env-var side effects
# ---------------------------------------------------------------------------


class _FakeApplication:
    """Stand-in for ``serving.Application``. We only need
    ``add_deployment`` here — ``KnowledgeConfig`` never reads back
    from it."""

    def __init__(self) -> None:
        self.deployments: list[tuple[str, object]] = []

    def add_deployment(self, deployment: object, name: str) -> None:
        self.deployments.append((name, deployment))


def test_hosted_backend_registers_no_deployment() -> None:
    """Hosted backends call vendor endpoints directly — there's no
    Ray deployment to bring up at cluster start."""
    cfg = KnowledgeConfig(pdf_extractor=PdfExtractorConfig(backend="anthropic"))
    app = _FakeApplication()
    cfg.add_deployments_to_app(app, top_level=False)
    assert app.deployments == []


def test_self_hosted_backend_handles_missing_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the operator selects a self-hosted backend whose
    underlying library isn't installed, the deployment-resolve
    path raises :class:`PdfExtractorError`. ``add_deployments_to_app``
    catches it, logs, and proceeds with no deployment registered."""
    from polymathera.colony.cluster import extractors as ext_mod
    from polymathera.colony.cluster.extractors.base import PdfExtractorError

    def _raise(_backend: str) -> object:
        raise PdfExtractorError("library missing in test env")

    monkeypatch.setattr(ext_mod, "get_deployment_class", _raise)

    cfg = KnowledgeConfig(
        pdf_extractor=PdfExtractorConfig(
            backend="docling", replicas=2, num_gpus=0,
        ),
    )
    app = _FakeApplication()
    cfg.add_deployments_to_app(app, top_level=False)
    assert app.deployments == []


def test_self_hosted_backend_registers_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the deployment class resolves cleanly, the config adds
    it to the app under the class name, forwarding ``options`` as
    bind kwargs."""
    from polymathera.colony.cluster import extractors as ext_mod

    class _FakeBound:
        def __init__(self, **kwargs: object) -> None:
            self.bind_kwargs = kwargs

    class _FakeDeploymentClass:
        __name__ = "FakeMarkerExtractorDeployment"

        def bind(self, **kwargs: object) -> _FakeBound:
            return _FakeBound(**kwargs)

    fake_cls = _FakeDeploymentClass()
    monkeypatch.setattr(
        ext_mod, "get_deployment_class", lambda backend: fake_cls,
    )

    cfg = KnowledgeConfig(
        pdf_extractor=PdfExtractorConfig(
            backend="marker",
            options={"max_pages": 50, "batch_multiplier": 2},
        ),
    )
    app = _FakeApplication()
    cfg.add_deployments_to_app(app, top_level=False)

    assert len(app.deployments) == 1
    name, bound = app.deployments[0]
    assert name == "FakeMarkerExtractorDeployment"
    assert isinstance(bound, _FakeBound)
    assert bound.bind_kwargs == {"max_pages": 50, "batch_multiplier": 2}


# ---------------------------------------------------------------------------
# Lazy deployment registry
# ---------------------------------------------------------------------------


def test_get_deployment_class_unknown_backend_raises() -> None:
    from polymathera.colony.cluster.extractors import (
        PdfExtractorError, get_deployment_class,
    )

    with pytest.raises(PdfExtractorError, match="unknown"):
        get_deployment_class("definitely-not-a-backend")


# ---------------------------------------------------------------------------
# Worker-side resolution contract
# ---------------------------------------------------------------------------
#
# These tests guard the iteration-3.1 fix: workers read
# ``KnowledgeConfig`` from the typed config tree, NOT from
# ``os.environ``. The chain is:
#
#   YAML (``knowledge``)
#     → driver loads via ConfigurationManager.initialize()
#     → POLYMATHERA_CONFIG forwards to actor runtime_env.env_vars
#     → worker's PolymatheraApp singleton loads SAME YAML
#     → worker calls get_component_or_default("knowledge", KnowledgeConfig)
#     → set_knowledge_deps reads the typed config to pick the reader


def test_worker_resolution_contract_uses_get_component_or_default() -> None:
    """The fast-path used by ``knowledge/deps.py``:
    :func:`get_component_or_default` is sync, returns defaults if
    the manager has not been initialized, and is the documented
    sync-safe accessor for capability constructors."""

    from polymathera.colony.distributed.config import get_component_or_default

    cfg = get_component_or_default("knowledge", KnowledgeConfig)
    assert isinstance(cfg, KnowledgeConfig)
    # When the manager is uninitialized in this unit-test process,
    # ``get_component_or_default`` returns bare defaults — which is
    # the safe fallback workers see before YAML is loaded.
    assert isinstance(cfg.pdf_extractor, PdfExtractorConfig)
    assert isinstance(cfg.qdrant, QdrantConfig)
    assert isinstance(cfg.grobid, GrobidConfig)
