"""``BulkAcquisitionCapability`` — master §6.6 corpus bulk acquisition.

Master §6.6 names a capability that downstream applications use to
load expected literature into the corpus **without trampling user-
seeded content**. This module implements the durable spine of that
capability:

- A typed :class:`CorpusManifest` schema — operator-authored YAML
  listing every external source to acquire. Path is operator-chosen;
  the action takes it as an explicit argument. The suggested
  convention for operators who keep manifests inside a Colony design
  monorepo is ``.colony/corpora/<name>.manifest.yaml`` (alongside
  ``.colony/repo_map.yaml`` and ``.colony/manifest.json``); the
  framework does not enforce this.
- An :class:`AcquirerStrategy` ABC — the integration point for the
  per-method acquirers (HTTP, arXiv, DOI, IEEE Xplore, SAE Mobilus,
  Semantic Scholar). The only real implementation today is
  :class:`LocalPathAcquirer` (the file is already on disk because the
  operator pre-seeded it). The rest are ``_TODO_<source>Acquirer``
  stubs that raise ``NotImplementedError`` with the build-effort
  estimate, so the framework's missing integrations are *visible*
  rather than silent.
- :class:`BulkAcquisitionCapability` — the agent-callable
  ``@action_executor`` surface. Defaults to
  :data:`IngestionPolicy.SKIP_IF_PRESENT`, the safe-by-default
  contract: re-running on the same manifest never overwrites or
  re-ingests already-acquired sources unless the operator opts in.

Acronyms expanded for non-specialist readers:

- DOI — Digital Object Identifier (a Crossref-issued canonical URL).
- IEEE Xplore — the IEEE's literature portal.
- SAE Mobilus — Society of Automotive Engineers literature portal.
- arXiv — the open-access preprint server.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from overrides import override
from pydantic import BaseModel, ConfigDict, Field

from ..agents.base import Agent, AgentCapability
from ..agents.models import AgentSuspensionState
from ..agents.patterns.actions import action_executor
from .ingestion import Ingestor
from .models import (
    CorpusTier,
    IngestionPolicy,
    IngestionRecord,
    IngestionStatus,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest schema (durable contract)
# ---------------------------------------------------------------------------


_MANIFEST_SCHEMA_VERSION = 1
"""Bump when the manifest schema changes in a way that needs a
migration. Migrations live alongside this constant. Schema-version
checks happen in :meth:`CorpusManifest.from_yaml`."""


class ManifestEntry(BaseModel):
    """One source in a corpus manifest.

    The ``method`` field names which :class:`AcquirerStrategy` handles
    the entry (e.g., ``"local_path"`` is handled by
    :class:`LocalPathAcquirer`). ``acquirer_args`` is per-method
    free-form; e.g. for ``"local_path"`` it carries ``{"path":
    "/abs/path/to/file.pdf"}``. The eventual HTTP / arXiv / DOI
    acquirers will define their own argument keys.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    uri: str
    """Canonical ``source_uri`` to register the chunks under. Stable
    across re-acquisitions — this is the identity the
    :class:`Ingestor`'s idempotency contract keys on."""

    tier: CorpusTier = CorpusTier.UNTIERED
    """Tier under which the chunks should be stored. Honoured per the
    ``IngestionPolicy.UPGRADE_TIER`` rule when re-acquiring an
    already-known source: a higher tier upgrades, a lower tier is
    silently preserved. See
    :func:`~polymathera.colony.knowledge.models.tier_priority`."""

    method: str
    """The :class:`AcquirerStrategy` ``method`` key (e.g.,
    ``"local_path"``, ``"http_url"``, ``"arxiv_id"``)."""

    acquirer_args: dict[str, Any] = Field(default_factory=dict)
    """Per-method arguments forwarded to the acquirer."""

    expected_sha256: str = ""
    """Optional SHA-256 digest of the raw bytes. When non-empty the
    capability verifies the acquired file matches before handing off
    to the ingestor — protects against silent corruption / midstream
    swap."""

    notes: str = ""
    """Free-form human-readable provenance / citation note. Round-
    tripped through YAML so a reviewer can read why a paper is in the
    corpus without leaving the manifest."""

    data_type_override: str | None = None
    """If set, forwarded to ``Ingestor.ingest_file`` to override the
    auto-detected ``data_type`` (e.g., classify a PDF as
    ``"standard_clause"`` rather than ``"paper_section"``)."""

    extra: dict[str, Any] = Field(default_factory=dict)
    """Forward-compatibility surface — unknown YAML keys land here so
    new manifest fields don't break older readers."""


class CorpusManifest(BaseModel):
    """A list of external sources to acquire and ingest.

    Operator-authored YAML; path is supplied explicitly to
    :meth:`BulkAcquisitionCapability.acquire_manifest`. The framework
    does not prescribe a location, but operators who keep manifests
    inside a Colony design monorepo will find it natural to put them
    alongside ``.colony/repo_map.yaml`` and ``.colony/manifest.json``
    — e.g. ``.colony/corpora/<name>.manifest.yaml``. Multiple
    manifests can coexist (``papers``, ``standards``, …); each one
    has its own ``domain`` label.

    The schema is intentionally minimal so it can absorb future
    fields without breaking older consumers — the
    ``schema_version`` field gates migrations.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    schema_version: int = _MANIFEST_SCHEMA_VERSION
    domain: str
    """The domain identity (e.g., ``"quantum"``, ``"racer"``,
    ``"fusion"``). Used by the report and by future per-domain hooks
    (rate limiting, credentialing)."""

    description: str = ""
    entries: tuple[ManifestEntry, ...] = ()
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_yaml(self) -> str:
        import yaml  # type: ignore[import-not-found]

        # ``mode="json"`` so ``CorpusTier`` (a ``str``-Enum) and
        # ``Path`` round-trip as their string forms — PyYAML's default
        # representer can't serialize arbitrary Python objects.
        return yaml.safe_dump(
            self.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
        )

    @classmethod
    def from_yaml(cls, text: str) -> "CorpusManifest":
        import yaml  # type: ignore[import-not-found]

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("CorpusManifest YAML must be a mapping")
        version = int(data.get("schema_version", 1))
        if version > _MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"CorpusManifest schema_version {version} is newer than "
                f"this colony version supports ({_MANIFEST_SCHEMA_VERSION}). "
                f"Upgrade polymathera-colony or downgrade the manifest.",
            )
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Acquirer strategy
# ---------------------------------------------------------------------------


class AcquirerStrategy(ABC):
    """Materialize a :class:`ManifestEntry` to a local file the
    :class:`Ingestor` can read.

    Concrete strategies are keyed by ``method``. The
    :class:`BulkAcquisitionCapability` constructor accepts a list of
    strategies and dispatches per entry.
    """

    @property
    @abstractmethod
    def method(self) -> str:
        """The ``method`` key this strategy handles (e.g.,
        ``"local_path"``, ``"http_url"``)."""

    @abstractmethod
    async def acquire(
        self, entry: ManifestEntry, *, cache_dir: Path,
    ) -> "AcquiredSource":
        """Resolve ``entry`` to a local file. ``cache_dir`` is a
        per-capability scratch directory acquirers may use to stash
        downloaded bytes (the :class:`LocalPathAcquirer` ignores it).
        """


class AcquiredSource(BaseModel):
    """Output of :meth:`AcquirerStrategy.acquire` — the ingestor reads
    from ``local_path``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str
    local_path: Path
    fetched_bytes: int = 0
    cached: bool = False
    """``True`` if the file was already on disk and the strategy
    didn't fetch fresh bytes (e.g., :class:`LocalPathAcquirer` always
    returns ``cached=True``; an HTTP acquirer returns ``True`` when
    its cache hit was a match)."""


class LocalPathAcquirer(AcquirerStrategy):
    """The operator has already placed the file on disk — typical for
    a pre-seeded corpus. ``acquirer_args["path"]`` carries the
    absolute path. No network access.
    """

    @property
    @override
    def method(self) -> str:
        return "local_path"

    @override
    async def acquire(
        self, entry: ManifestEntry, *, cache_dir: Path,
    ) -> AcquiredSource:
        raw_path = entry.acquirer_args.get("path")
        if not raw_path:
            raise BulkAcquisitionError(
                f"local_path acquirer requires acquirer_args['path']; "
                f"missing for entry {entry.uri!r}",
            )
        path = Path(str(raw_path)).expanduser()
        if not path.is_file():
            raise BulkAcquisitionError(
                f"local_path acquirer: file not found for entry "
                f"{entry.uri!r}: {path}",
            )
        return AcquiredSource(
            uri=entry.uri,
            local_path=path,
            fetched_bytes=path.stat().st_size,
            cached=True,
        )


# ---- TODO stubs (master §6.6 — drop in real impls per dossier §II.N) ----


class _TODOAcquirer(AcquirerStrategy):
    """Base for unimplemented acquirers. Each ``_TODO_<source>Acquirer``
    raises a ``NotImplementedError`` carrying the dossier reference so
    the gap is visible. They subclass :class:`AcquirerStrategy` so
    they're discoverable and can be wired into manifests today (with
    a deferred error) — i.e., a manifest can be authored before its
    acquirer ships."""

    _METHOD: str = ""
    _DOSSIER_REF: str = ""
    _BUILD_EFFORT: str = ""

    @property
    @override
    def method(self) -> str:
        return self._METHOD

    @override
    async def acquire(
        self, entry: ManifestEntry, *, cache_dir: Path,
    ) -> AcquiredSource:
        raise NotImplementedError(
            f"{type(self).__name__} is a placeholder. Master §6.6 / dossier "
            f"{self._DOSSIER_REF}. Build effort: {self._BUILD_EFFORT}. "
            f"Until implemented, fall back to LocalPathAcquirer (download "
            f"the file manually, list it under method='local_path').",
        )


class _TODO_HttpAcquirer(_TODOAcquirer):
    """Method ``"http_url"`` — fetch bytes from
    ``acquirer_args['url']`` via ``httpx`` with content-hash
    verification + on-disk cache. Master §6.6.1.
    """

    _METHOD = "http_url"
    _DOSSIER_REF = "master §6.6.1"
    _BUILD_EFFORT = "1-2 days"


class _TODO_ArxivAcquirer(_TODOAcquirer):
    """Method ``"arxiv_id"`` — fetch a paper by arXiv identifier
    (``acquirer_args['arxiv_id']``) via the arXiv API. Master §6.6.1
    + QS dossier §II.N Tier 3 (most arXiv physics.atom-ph and
    eess.SP papers). arXiv = open-access preprint server.
    """

    _METHOD = "arxiv_id"
    _DOSSIER_REF = "master §6.6.1, QS dossier §II.N Tier 3"
    _BUILD_EFFORT = "2-3 days"


class _TODO_DoiAcquirer(_TODOAcquirer):
    """Method ``"doi"`` — resolve a DOI (Digital Object Identifier)
    via Crossref + Unpaywall fallback (when the publication is
    open-access). Master §6.6.1 + QS dossier §II.N Tier 3.
    """

    _METHOD = "doi"
    _DOSSIER_REF = "master §6.6.1, QS dossier §II.N Tier 3"
    _BUILD_EFFORT = "3-5 days"


class _TODO_IeeeXploreAcquirer(_TODOAcquirer):
    """Method ``"ieee_xplore"`` — IEEE Xplore (the IEEE's literature
    portal) requires institutional API access. Master §6.6.2.
    """

    _METHOD = "ieee_xplore"
    _DOSSIER_REF = "master §6.6.2"
    _BUILD_EFFORT = "2-3 days plus institutional API key"


class _TODO_SaeMobilusAcquirer(_TODOAcquirer):
    """Method ``"sae_mobilus"`` — SAE Mobilus = Society of Automotive
    Engineers literature portal. Subscription-gated. Master §6.6.2 +
    Racer dossier (RACER program needs SAE journals).
    """

    _METHOD = "sae_mobilus"
    _DOSSIER_REF = "master §6.6.2, Racer dossier"
    _BUILD_EFFORT = "2-3 days plus institutional API key"


class _TODO_SemanticScholarAcquirer(_TODOAcquirer):
    """Method ``"semantic_scholar"`` — Semantic Scholar's open API for
    paper metadata + open-access PDFs when available. Master §6.6.1.
    """

    _METHOD = "semantic_scholar"
    _DOSSIER_REF = "master §6.6.1"
    _BUILD_EFFORT = "2-3 days"


class _TODO_NeuroImageAcquirer(_TODOAcquirer):
    """Method ``"neuroimage"`` — NeuroImage is the canonical clinical
    MEG (Magnetoencephalography) journal; subscription-gated. QS
    dossier §II.N Tier 3 mentions it explicitly.
    """

    _METHOD = "neuroimage"
    _DOSSIER_REF = "QS dossier §II.N Tier 3"
    _BUILD_EFFORT = "2-3 days plus institutional access"


# ---------------------------------------------------------------------------
# Capability + report
# ---------------------------------------------------------------------------


class BulkAcquisitionError(RuntimeError):
    """Raised by acquirers when the source cannot be materialised."""


class AcquisitionEntry(BaseModel):
    """One row in a :class:`BulkAcquisitionReport`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str
    tier: CorpusTier
    method: str
    outcome: str
    """One of: ``"ingested"``, ``"skipped_present"``,
    ``"tier_upgraded"``, ``"hash_mismatch"``, ``"fetch_failed"``,
    ``"ingest_failed"``, ``"unsupported_method"``."""

    chunks_produced: int = 0
    document_hash: str = ""
    error: str = ""
    cached: bool = False
    fetched_bytes: int = 0


class BulkAcquisitionReport(BaseModel):
    """The result of running
    :meth:`BulkAcquisitionCapability.acquire_manifest` over a manifest.
    Aggregates per-source outcomes plus rolled-up counts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: str
    total: int
    ingested: int
    skipped_present: int
    tier_upgraded: int
    failed: int
    entries: tuple[AcquisitionEntry, ...]


class BulkAcquisitionCapability(AgentCapability):
    """Orchestrates corpus bulk acquisition (master §6.6).

    The capability is a thin coordinator: per :class:`ManifestEntry`,
    it picks an :class:`AcquirerStrategy` by ``method``, materializes
    the file, optionally verifies the SHA-256, and hands off to the
    :class:`Ingestor` under the configured :class:`IngestionPolicy`.
    The default policy is :data:`IngestionPolicy.SKIP_IF_PRESENT` so
    bulk-acquisition cannot trample user-seeded content.

    Wire it into a :class:`KnowledgeCuratorAgent` (or any agent that
    owns the ingestor) by giving the capability the same Ingestor
    reference:

    .. code-block:: python

        bulk = BulkAcquisitionCapability(
            agent=curator,
            ingestor=curator._ingestor,
            acquirers=[LocalPathAcquirer()],
            cache_dir=Path("/var/cache/colony/corpus"),
        )

    Detached use (no agent, e.g. from a CLI command) is also
    supported — pass ``agent=None, scope_id="bulk_acquisition"``.
    """

    def __init__(
        self,
        agent: Agent | None = None,
        scope_id: str | None = None,
        *,
        ingestor: Ingestor,
        acquirers: Sequence[AcquirerStrategy] = (),
        cache_dir: Path | str = "/tmp/polymathera_corpus_cache",
        capability_key: str | None = None,
        app_name: str | None = None,
    ) -> None:
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            input_patterns=[],
            capability_key=capability_key,
            app_name=app_name,
        )
        self._ingestor = ingestor
        self._cache_dir = Path(cache_dir).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Strategies registered by ``method`` — last write wins so a
        # caller can inject a real acquirer that shadows a TODO stub.
        self._acquirers: dict[str, AcquirerStrategy] = {}
        for strategy in acquirers:
            self._acquirers[strategy.method] = strategy
        if "local_path" not in self._acquirers:
            # Local-path is always available — the user-seeded path.
            self._acquirers["local_path"] = LocalPathAcquirer()

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"knowledge", "ingestion", "bulk_acquisition"})

    def register_acquirer(self, strategy: AcquirerStrategy) -> None:
        """Add or replace a per-method acquirer at runtime."""

        self._acquirers[strategy.method] = strategy

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    # ---- Action surface ---------------------------------------------------

    @action_executor(
        planning_summary=(
            "Acquire every entry in a corpus manifest into the corpus, "
            "respecting an IngestionPolicy. Default policy is "
            "SKIP_IF_PRESENT: never trample user-seeded content."
        ),
    )
    async def acquire_manifest(
        self,
        manifest_path: str,
        *,
        policy: IngestionPolicy = IngestionPolicy.SKIP_IF_PRESENT,
    ) -> BulkAcquisitionReport:
        manifest = CorpusManifest.from_yaml(
            Path(manifest_path).read_text(encoding="utf-8"),
        )
        return await self.acquire_corpus(manifest, policy=policy)

    @action_executor(
        planning_summary=(
            "Acquire a single corpus entry. Returns the typed outcome "
            "(ingested / skipped_present / tier_upgraded / failed)."
        ),
    )
    async def acquire_one(
        self,
        entry: ManifestEntry,
        *,
        policy: IngestionPolicy = IngestionPolicy.SKIP_IF_PRESENT,
    ) -> AcquisitionEntry:
        return await self._acquire_entry(entry, policy=policy)

    async def acquire_corpus(
        self,
        manifest: CorpusManifest,
        *,
        policy: IngestionPolicy = IngestionPolicy.SKIP_IF_PRESENT,
    ) -> BulkAcquisitionReport:
        """Run the manifest end-to-end. Public Python API for callers
        that already hold a :class:`CorpusManifest` instance."""

        rows: list[AcquisitionEntry] = []
        for entry in manifest.entries:
            row = await self._acquire_entry(entry, policy=policy)
            rows.append(row)
        return _summarise(manifest.domain, tuple(rows))

    # ---- Internals --------------------------------------------------------

    async def _acquire_entry(
        self,
        entry: ManifestEntry,
        *,
        policy: IngestionPolicy,
    ) -> AcquisitionEntry:
        strategy = self._acquirers.get(entry.method)
        if strategy is None:
            return AcquisitionEntry(
                uri=entry.uri,
                tier=entry.tier,
                method=entry.method,
                outcome="unsupported_method",
                error=(
                    f"no acquirer registered for method {entry.method!r}; "
                    f"known: {sorted(self._acquirers)}"
                ),
            )

        try:
            acquired = await strategy.acquire(entry, cache_dir=self._cache_dir)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "BulkAcquisition: acquirer %s failed for %s",
                type(strategy).__name__, entry.uri,
            )
            return AcquisitionEntry(
                uri=entry.uri,
                tier=entry.tier,
                method=entry.method,
                outcome="fetch_failed",
                error=str(exc),
            )

        if entry.expected_sha256:
            actual = _sha256_file(acquired.local_path)
            if actual.lower() != entry.expected_sha256.lower():
                return AcquisitionEntry(
                    uri=entry.uri,
                    tier=entry.tier,
                    method=entry.method,
                    outcome="hash_mismatch",
                    error=(
                        f"expected_sha256 {entry.expected_sha256!r}; "
                        f"actual {actual!r}"
                    ),
                    cached=acquired.cached,
                    fetched_bytes=acquired.fetched_bytes,
                )

        try:
            record = await self._ingestor.ingest_file(
                acquired.local_path,
                tier=entry.tier,
                source_uri=entry.uri,
                data_type_override=entry.data_type_override,
                policy=policy,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "BulkAcquisition: ingestor failed for %s", entry.uri,
            )
            return AcquisitionEntry(
                uri=entry.uri,
                tier=entry.tier,
                method=entry.method,
                outcome="ingest_failed",
                error=str(exc),
                cached=acquired.cached,
                fetched_bytes=acquired.fetched_bytes,
            )
        return _outcome_from_record(entry, record, acquired)

    # ---- Suspension hooks (no in-memory state worth persisting) -----------

    @override
    async def serialize_suspension_state(
        self, state: AgentSuspensionState,
    ) -> AgentSuspensionState:
        return state

    @override
    async def deserialize_suspension_state(
        self, state: AgentSuspensionState,
    ) -> None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _outcome_from_record(
    entry: ManifestEntry,
    record: IngestionRecord,
    acquired: AcquiredSource,
) -> AcquisitionEntry:
    base = {
        "uri": entry.uri,
        "tier": entry.tier,
        "method": entry.method,
        "chunks_produced": record.chunks_produced,
        "document_hash": record.document_hash,
        "cached": acquired.cached,
        "fetched_bytes": acquired.fetched_bytes,
    }
    if record.status is IngestionStatus.COMPLETED:
        return AcquisitionEntry(outcome="ingested", **base)
    if record.status is IngestionStatus.SKIPPED_ALREADY_PRESENT:
        return AcquisitionEntry(outcome="skipped_present", **base)
    if record.status is IngestionStatus.TIER_UPGRADED:
        return AcquisitionEntry(outcome="tier_upgraded", **base)
    return AcquisitionEntry(
        outcome="ingest_failed",
        error=record.error or f"unexpected status {record.status.value}",
        **base,
    )


def _summarise(
    domain: str, rows: tuple[AcquisitionEntry, ...],
) -> BulkAcquisitionReport:
    by_outcome: Mapping[str, int] = {
        outcome: sum(1 for r in rows if r.outcome == outcome)
        for outcome in {row.outcome for row in rows}
    }
    return BulkAcquisitionReport(
        domain=domain,
        total=len(rows),
        ingested=by_outcome.get("ingested", 0),
        skipped_present=by_outcome.get("skipped_present", 0),
        tier_upgraded=by_outcome.get("tier_upgraded", 0),
        failed=(
            by_outcome.get("fetch_failed", 0)
            + by_outcome.get("ingest_failed", 0)
            + by_outcome.get("hash_mismatch", 0)
            + by_outcome.get("unsupported_method", 0)
        ),
        entries=rows,
    )


def _sha256_file(path: Path, *, chunk_size: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            blk = fh.read(chunk_size)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


__all__ = (
    "AcquirerStrategy",
    "AcquiredSource",
    "AcquisitionEntry",
    "BulkAcquisitionCapability",
    "BulkAcquisitionError",
    "BulkAcquisitionReport",
    "CorpusManifest",
    "LocalPathAcquirer",
    "ManifestEntry",
    "_TODO_ArxivAcquirer",
    "_TODO_DoiAcquirer",
    "_TODO_HttpAcquirer",
    "_TODO_IeeeXploreAcquirer",
    "_TODO_NeuroImageAcquirer",
    "_TODO_SaeMobilusAcquirer",
    "_TODO_SemanticScholarAcquirer",
)
