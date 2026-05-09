"""Knowledge-layer configuration plumbed through PolymatheraClusterConfig.

Operators declare which PDF extractor backend to bring up at
cluster bring-up time via the YAML ``polymathera_cluster.knowledge``
section. The config decides:

1. Which (if any) self-hosted ``*ExtractorDeployment`` to add to
   the serving Application — Marker / Docling / MinerU.
2. Which reader registry every Ray worker resolves at process
   start, via :func:`set_knowledge_deps`. Workers re-read the same
   ``KnowledgeConfig`` from the global :class:`ConfigurationManager`
   (each process loads the YAML pointed at by ``POLYMATHERA_CONFIG``);
   no env-var passthrough required.

YAML shape::

    polymathera_cluster:
      knowledge:
        pdf_extractor:
          backend: docling          # mistral_ocr | anthropic | gemini |
                                    # llamaparse | marker | docling | mineru
          options:                  # forwarded as backend_kwargs to the
                                    # reader; tier knobs live here
            tier: cost_effective    # for llamaparse
          # Self-hosted-only options:
          replicas: 1
          num_gpus: 0
        image_dir: "/mnt/shared/colony-images"
        qdrant:
          url: "http://qdrant:6333"
          collection: "colony_knowledge"
        grobid:
          url: "http://grobid:8070"

Hosted backends (Mistral / Anthropic / Gemini / LlamaParse) need
no Ray deployment — only the reader registration. Self-hosted
backends also bring up their ``*ExtractorDeployment``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import ConfigDict, Field

from ..distributed.config import (
    ConfigComponent,
    Mutability,
    Tier,
    register_polymathera_config,
    tier_metadata,
)
from ..distributed.ray_utils import serving


logger = logging.getLogger(__name__)


PdfExtractorBackend = Literal[
    "mistral_ocr",
    "anthropic",
    "gemini",
    "llamaparse",
    "marker",
    "docling",
    "mineru",
]


_HOSTED_BACKENDS = frozenset({"mistral_ocr", "anthropic", "gemini", "llamaparse"})
_SELF_HOSTED_BACKENDS = frozenset({"marker", "docling", "mineru"})


class PdfExtractorConfig(ConfigComponent):
    """Per-backend configuration for the PDF extractor.

    ``options`` is a free-form dict the operator uses to pin a
    backend's tier / model / endpoint. The framework forwards it
    verbatim to the reader's constructor as ``backend_kwargs``,
    so the schema is naturally extensible — adding a new tier knob
    is one line in the reader, no config-component change needed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    backend: PdfExtractorBackend = Field(
        default="mistral_ocr",
        description=(
            "Active PDF extractor. Hosted backends need only an API "
            "key; self-hosted backends require the corresponding "
            "deployment to be brought up at cluster start."
        ),
        json_schema_extra=tier_metadata(
            tier=Tier.L1_OPERATOR, mutability=Mutability.RELOADABLE,
        ),
    )

    options: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Backend-specific kwargs forwarded to the reader. "
            "Examples: ``{\"tier\": \"agentic\"}`` for LlamaParse, "
            "``{\"model\": \"gemini-2.5-pro\"}`` for Gemini, "
            "``{\"prompt_cache\": false}`` for Anthropic."
        ),
        json_schema_extra=tier_metadata(
            tier=Tier.L1_OPERATOR, mutability=Mutability.RELOADABLE,
        ),
    )

    # Self-hosted-only knobs — ignored when ``backend`` is a hosted
    # vendor.
    replicas: int = Field(
        default=1, ge=1,
        description="Number of replicas of the self-hosted deployment.",
        json_schema_extra=tier_metadata(tier=Tier.L1_OPERATOR),
    )
    num_gpus: float = Field(
        default=0.0, ge=0.0,
        description=(
            "GPUs per replica for self-hosted deployments. ``0`` "
            "uses CPU; production typically wants ``1``."
        ),
        json_schema_extra=tier_metadata(tier=Tier.L1_OPERATOR),
    )

    def is_self_hosted(self) -> bool:
        return self.backend in _SELF_HOSTED_BACKENDS

    def is_hosted(self) -> bool:
        return self.backend in _HOSTED_BACKENDS


class QdrantConfig(ConfigComponent):
    """Qdrant vector-store connection for knowledge-layer storage.

    Empty ``url`` (the default) keeps :class:`InMemoryVectorStore` —
    fine for tests and single-process dev. Operators of the
    docker-compose stack set ``url`` to ``http://qdrant:6333`` in YAML
    so ingested chunks survive process restarts and live in the same
    collection across driver and worker actors.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    url: str = Field(
        default="",
        description=(
            "Qdrant HTTP endpoint. Empty selects the in-memory "
            "fallback. Docker-compose: ``http://qdrant:6333``."
        ),
        json_schema_extra=tier_metadata(
            tier=Tier.L1_OPERATOR, mutability=Mutability.RELOADABLE,
        ),
    )
    collection: str = Field(
        default="colony_knowledge",
        description="Collection name shared by curation and retrieval.",
        json_schema_extra=tier_metadata(
            tier=Tier.L1_OPERATOR, mutability=Mutability.RELOADABLE,
        ),
    )


class GrobidConfig(ConfigComponent):
    """GROBID PDF-to-TEI extractor connection.

    Empty ``url`` (the default) means GROBID is not available — the
    text-only fallback reader handles PDFs. Docker-compose stacks set
    ``url`` to ``http://grobid:8070``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    url: str = Field(
        default="",
        description=(
            "GROBID HTTP endpoint. Empty disables the GROBID "
            "metadata reader. Docker-compose: ``http://grobid:8070``."
        ),
        json_schema_extra=tier_metadata(
            tier=Tier.L1_OPERATOR, mutability=Mutability.RELOADABLE,
        ),
    )


@register_polymathera_config(path="knowledge")
class KnowledgeConfig(ConfigComponent):
    """Knowledge-layer cluster config.

    Single source of truth for PDF extraction, vector store, image
    store, and GROBID endpoint selection. Each Ray worker re-reads
    this from its own ``ConfigurationManager`` (loaded from the YAML
    at ``POLYMATHERA_CONFIG``), so the driver and every actor agree
    on the configured backends without env-var passthrough.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pdf_extractor: PdfExtractorConfig = Field(
        default_factory=PdfExtractorConfig,
        description="PDF extraction backend + tier configuration.",
    )

    image_dir: str = Field(
        default="",
        description=(
            "Directory for :class:`LocalFsImageStore`. Empty selects "
            "the in-memory fallback. Docker-compose: "
            "``/mnt/shared/colony-images`` so figures survive "
            "container restarts and are visible to the dashboard's "
            "KB tab."
        ),
        json_schema_extra=tier_metadata(
            tier=Tier.L1_OPERATOR, mutability=Mutability.RELOADABLE,
        ),
    )

    qdrant: QdrantConfig = Field(
        default_factory=QdrantConfig,
        description="Qdrant vector-store connection (in-memory if empty).",
    )

    grobid: GrobidConfig = Field(
        default_factory=GrobidConfig,
        description="GROBID PDF-to-TEI endpoint (text-only fallback if empty).",
    )

    def add_deployments_to_app(
        self, app: serving.Application, top_level: bool,
    ) -> None:
        """Bring up the configured self-hosted ``*ExtractorDeployment``.

        For hosted backends this is a no-op on the Ray side — the reader
        hits the vendor endpoint directly, and worker-side
        :func:`set_knowledge_deps` resolves it from the same
        :class:`KnowledgeConfig` (via the global ConfigurationManager).

        For self-hosted backends the corresponding deployment class is
        resolved via the lazy registry in
        :mod:`polymathera.colony.cluster.extractors` and bound with the
        operator's ``options`` + ``num_gpus`` overrides.
        """
        cfg = self.pdf_extractor

        if not cfg.is_self_hosted():
            logger.info(
                "KnowledgeConfig: hosted PDF extractor %r — no Ray "
                "deployment to add (workers resolve the reader from "
                "KnowledgeConfig directly).",
                cfg.backend,
            )
            return

        from ..cluster.extractors import get_deployment_class

        try:
            deployment_cls = get_deployment_class(cfg.backend)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "KnowledgeConfig: failed to resolve deployment class for "
                "backend=%r (%s); skipping deployment registration. "
                "Install the corresponding poetry extra and redeploy.",
                cfg.backend, exc,
            )
            return

        deployment_name = deployment_cls.__name__
        ray_actor_options: dict[str, Any] = {}
        if cfg.num_gpus:
            ray_actor_options["num_gpus"] = cfg.num_gpus
        bound = deployment_cls.bind(**cfg.options).options(
            num_replicas=cfg.replicas,
            ray_actor_options=ray_actor_options or None,
        ) if hasattr(deployment_cls.bind(**cfg.options), "options") else (
            deployment_cls.bind(**cfg.options)
        )
        app.add_deployment(bound, name=deployment_name)
        logger.info(
            "KnowledgeConfig: added %s (replicas=%d, num_gpus=%g, "
            "options=%r)",
            deployment_name, cfg.replicas, cfg.num_gpus, cfg.options,
        )


__all__ = (
    "GrobidConfig",
    "KnowledgeConfig",
    "PdfExtractorBackend",
    "PdfExtractorConfig",
    "QdrantConfig",
)
