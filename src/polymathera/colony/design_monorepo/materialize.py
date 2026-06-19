"""Bridge between :class:`RepoMap` and the two materialisation
operations it drives:

- :func:`materialize_vcm_sources` — issues one
  ``mmap_application_scope`` per row in ``vcm_sources:`` (VCM
  mapping; the dashboard's "Map to VCM" button + the CLI auto-deploy
  flow).
- :func:`materialize_knowledge_sources` — walks each row in
  ``knowledge_sources:`` and feeds matching files (or acquired
  remote sources) to the process-singleton :class:`Ingestor` via
  :class:`MonorepoPersistedIngestor`, persisting extraction outputs
  as ``.ingested/`` sidecars next to each source. Used by the
  SessionAgent's ``ingest_repo_map_literature`` action and the
  dashboard's "Ingest Knowledge" button.

Both accept ``enabled_sources`` so the dashboard's per-section
checkbox lists can filter the rows actually materialised. The two
operations are orthogonal — the same path can be VCM-mapped,
KB-ingested, both, or neither, and the dashboard exposes a separate
button per operation.

The knowledge-source materialiser does NOT commit. Persisting the
sidecars + any acquired files into git is the caller's job (the
agent path uses :meth:`DesignMonorepoClient.commit_with_identity`
after a successful materialisation). One commit per invocation is
the discipline; the materialiser stays unconcerned with identity.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from polymathera.colony.distributed import get_polymathera
from polymathera.colony.distributed.ray_utils import serving
from polymathera.colony.knowledge.acquirers import (
    AcquiredSource,
    AcquirerRegistry,
    default_registry as default_acquirer_registry,
)
from polymathera.colony.knowledge.models import (
    CorpusTier,
    IngestionRecord,
    IngestionStatus,
)
from polymathera.colony.knowledge.monorepo_persisted_ingestor import (
    SIDECAR_DIRNAME,
    MonorepoPersistedIngestor,
)
from polymathera.colony.vcm.models import MmapConfig

from .design_context_renewer import DesignContextLockRenewer
from .repo_map import DesignContextSource, KnowledgeSource, RepoMap
from ._internal import DESIGN_CONTEXT_URI_SCHEME


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Knowledge materialisation — typed return
# ---------------------------------------------------------------------------


class AcquisitionOutcome(BaseModel):
    """Per-row outcome of the acquirer step for an acquirer-shaped
    ``knowledge_sources`` entry. Local-paths rows do not appear here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    """``KnowledgeSource.name``."""

    method: str
    """``acquirer.method`` — the registry key used."""

    outcome: str
    """One of ``acquired`` / ``cached`` / ``todo_stub`` /
    ``unsupported_method`` / ``fetch_failed``."""

    local_path: str = ""
    """Absolute path of the acquired file on disk, when outcome is
    ``acquired`` or ``cached``."""

    fetched_bytes: int = 0
    error: str = ""


class KnowledgeMaterialisationReport(BaseModel):
    """Aggregated output of :func:`materialize_knowledge_sources`.

    The agent and dashboard surface different facets of this:

    - ``records`` — per-file :class:`IngestionRecord`s from
      :class:`Ingestor` / :class:`MonorepoPersistedIngestor`.
    - ``acquisitions`` — per-row acquirer outcomes. Empty when all
      enabled rows are paths-shaped.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    records: tuple[IngestionRecord, ...] = ()
    acquisitions: tuple[AcquisitionOutcome, ...] = ()

    @property
    def ingested_count(self) -> int:
        return sum(
            1 for r in self.records if r.status is IngestionStatus.COMPLETED
        )

    @property
    def skipped_count(self) -> int:
        return sum(
            1 for r in self.records
            if r.status in (
                IngestionStatus.SKIPPED_ALREADY_PRESENT,
                IngestionStatus.TIER_UPGRADED,
            )
        )

    @property
    def failed_count(self) -> int:
        return sum(
            1 for r in self.records if r.status is IngestionStatus.FAILED
        )


# ---------------------------------------------------------------------------
# VCM materialisation (unchanged)
# ---------------------------------------------------------------------------


def materialize_scope_ids(repo_map: RepoMap, base_scope_id: str) -> list[str]:
    """Pick one ``scope_id`` per ``vcm_sources`` row.

    The single-row default fallback (``RepoMap.default_for_unmapped_repo``)
    keeps using the caller-supplied ``base_scope_id`` so existing
    one-source-per-repo deployments are unaffected. Any
    ``repo_map.yaml`` with named rows composes
    ``f"{base_scope_id}:{row.name}"``.
    """

    sources = repo_map.vcm_sources
    if len(sources) == 1 and sources[0].name == "default":
        return [base_scope_id]
    return [f"{base_scope_id}:{s.name}" for s in sources]


async def materialize_vcm_sources(
    *,
    vcm_handle: Any,
    origin_url: str,
    branch: str,
    commit: str,
    base_scope_id: str,
    mmap_config: MmapConfig,
    enabled_sources: set[str] | None = None,
) -> list[Any]:
    """Clone the design monorepo, load its repo map, and issue one
    ``mmap_application_scope`` call per row in ``vcm_sources:``.

    ``mmap_config`` provides the deployment-wide defaults; each row
    may override individual fields (``flush_threshold``,
    ``flush_token_budget``, ``pinned``) via
    :meth:`VcmSource.to_mmap_config_overrides`.

    ``enabled_sources``, when not ``None``, restricts mapping to rows
    whose ``name`` is in the set. The default (``None``) maps every
    row.

    Returns the list of mmap results in row order. Failures on a
    single row are logged and skipped — the rest still materialise —
    so a typo in one row does not block the whole map.
    """

    polymathera = get_polymathera()
    storage = await polymathera.get_storage()
    colony_id = serving.get_colony_id()
    repo_path = await storage.git_storage.clone_or_retrieve_repository(
        origin_url=origin_url,
        branch=branch,
        commit=commit,
        vmr_id=colony_id,
    )
    repo_root = Path(str(repo_path))
    repo_map = RepoMap.load(repo_root)
    scope_ids = materialize_scope_ids(repo_map, base_scope_id)

    results: list[Any] = []
    for spec, scope_id in zip(repo_map.vcm_sources, scope_ids, strict=True):
        if enabled_sources is not None and spec.name not in enabled_sources:
            continue
        try:
            kwargs = spec.to_mmap_kwargs(
                repo_root=repo_root,
                scope_id=scope_id,
                fallback_origin_url=origin_url,
                fallback_branch=branch,
                fallback_commit=commit,
            )
            overrides = spec.to_mmap_config_overrides()
            effective_config = (
                mmap_config.model_copy(update=overrides) if overrides else mmap_config
            )
            result = await vcm_handle.mmap_application_scope(
                config=effective_config, **kwargs,
            )
            results.append(result)
        except Exception:  # noqa: BLE001
            logger.exception(
                "materialize_vcm_sources: row %r (scope_id=%s) failed; "
                "continuing with the remaining rows.",
                spec.name, scope_id,
            )
    return results


# ---------------------------------------------------------------------------
# Knowledge materialisation
# ---------------------------------------------------------------------------


async def materialize_knowledge_sources(
    *,
    repo_map: RepoMap,
    repo_root: Path,
    enabled_sources: set[str] | None = None,
    acquirer_registry: AcquirerRegistry | None = None,
    extractor_label: str = "",
) -> KnowledgeMaterialisationReport:
    """Walk ``repo_map.knowledge_sources`` and ingest each row.

    Behaviour by row shape:

    - **Local** (``paths`` set): glob-walk ``repo_root``, skipping
      anything under a ``.ingested/`` sidecar subtree, and ingest each
      match via :class:`MonorepoPersistedIngestor`. PDF inputs land
      in a ``.ingested/`` sidecar next to the source; non-PDFs flow
      straight through the ingestor.
    - **Remote** (``acquirer`` set): look up the strategy in
      ``acquirer_registry`` (default = :func:`default_acquirer_registry`),
      run it into ``<repo_root>/<row.destination>/``, then ingest the
      returned file via the same :class:`MonorepoPersistedIngestor`.
      Failures (registry miss, ``NotImplementedError`` from a TODO
      stub, network errors) land in the report's ``acquisitions``
      list at row level — they never poison the other rows.

    ``enabled_sources``, when not ``None``, restricts ingestion to
    rows whose ``name`` is in the set. The default (``None``) ingests
    every row.

    Per-file ingestion errors are logged at WARNING and don't fail
    the whole call — partial progress beats no progress.

    Does **not** commit. Callers that want one-commit-per-invocation
    persistence (the agent's ``ingest_repo_map_literature`` action)
    invoke :meth:`DesignMonorepoClient.commit_with_identity` after
    this returns.
    """

    from polymathera.colony.knowledge.deps import get_default_ingestor

    ingestor = get_default_ingestor()
    mpi = MonorepoPersistedIngestor(
        ingestor, ingestor.readers, extractor_label=extractor_label,
    )
    registry = acquirer_registry or default_acquirer_registry()

    records: list[IngestionRecord] = []
    acquisitions: list[AcquisitionOutcome] = []

    for source in repo_map.knowledge_sources:
        if enabled_sources is not None and source.name not in enabled_sources:
            continue

        if source.acquirer is not None:
            acquired = await _run_acquirer(
                registry=registry,
                source=source,
                repo_root=repo_root,
                acquisitions=acquisitions,
            )
            if acquired is None:
                continue
            try:
                rec = await mpi.ingest_file(
                    acquired.local_path,
                    tier=source.tier,
                    data_type_override=source.profile,
                    source_uri=_acquired_source_uri(source, acquired),
                )
                records.append(rec)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "materialize_knowledge_sources: ingest failed for "
                    "acquired source %s (row %r)",
                    acquired.local_path, source.name,
                )
            continue

        # Local-paths row.
        for abs_path in _iter_matching_files(repo_root, source):
            try:
                rec = await mpi.ingest_file(
                    abs_path,
                    tier=source.tier,
                    data_type_override=source.profile,
                )
                records.append(rec)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "materialize_knowledge_sources: ingest failed for %s "
                    "(row %r)", abs_path, source.name,
                )

    return KnowledgeMaterialisationReport(
        records=tuple(records),
        acquisitions=tuple(acquisitions),
    )


async def _run_acquirer(
    *,
    registry: AcquirerRegistry,
    source: KnowledgeSource,
    repo_root: Path,
    acquisitions: list[AcquisitionOutcome],
) -> AcquiredSource | None:
    """Run an acquirer-shaped row and record the outcome.

    Returns the :class:`AcquiredSource` on success so the caller can
    feed its ``local_path`` into the ingestor. Returns ``None`` on
    any failure (the row's outcome is already in ``acquisitions``)
    so the caller skips ingestion for that row.
    """

    assert source.acquirer is not None and source.destination is not None
    method = source.acquirer.method
    strategy = registry.get(method)
    if strategy is None:
        acquisitions.append(
            AcquisitionOutcome(
                name=source.name,
                method=method,
                outcome="unsupported_method",
                error=(
                    f"no acquirer registered for method {method!r}; "
                    f"known: {list(registry.methods())}"
                ),
            ),
        )
        return None

    destination_dir = repo_root / source.destination
    destination_dir.mkdir(parents=True, exist_ok=True)

    try:
        acquired = await strategy.acquire(
            args=source.acquirer.args,
            destination_dir=destination_dir,
        )
    except NotImplementedError as exc:
        acquisitions.append(
            AcquisitionOutcome(
                name=source.name,
                method=method,
                outcome="todo_stub",
                error=str(exc),
            ),
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "materialize_knowledge_sources: acquirer %s failed for row %r",
            method, source.name,
        )
        acquisitions.append(
            AcquisitionOutcome(
                name=source.name,
                method=method,
                outcome="fetch_failed",
                error=str(exc),
            ),
        )
        return None

    acquisitions.append(
        AcquisitionOutcome(
            name=source.name,
            method=method,
            outcome="cached" if acquired.cached else "acquired",
            local_path=str(acquired.local_path),
            fetched_bytes=acquired.fetched_bytes,
        ),
    )
    return acquired


def _acquired_source_uri(
    source: KnowledgeSource, acquired: AcquiredSource,
) -> str:
    """Pick the ``source_uri`` to register the ingested chunks under.

    Prefers the acquirer's ``metadata['source_uri']`` (canonical
    remote identity — ``arxiv:2407.12345v1``, ``doi:10.xxxx/...``)
    over the local file URI so re-ingesting from a different on-disk
    copy still hits the idempotency cache. Falls back to the local
    file URI when the acquirer didn't supply one."""

    meta_uri = acquired.metadata.get("source_uri")
    if isinstance(meta_uri, str) and meta_uri:
        return meta_uri
    return acquired.local_path.as_uri()


def _iter_matching_files(repo_root: Path, source: "KnowledgeSource"):
    """Yield absolute paths under ``repo_root`` whose forward-slash
    relative path matches ``source``. Walks the tree once; the
    ``KnowledgeSource.matches`` call delegates to :class:`pathspec`.

    Skips anything under a ``.ingested/`` subtree so the sidecar
    ``extracted.md`` / ``ingestion.json`` we wrote in a previous
    ingest doesn't get re-ingested as primary source content."""

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if SIDECAR_DIRNAME in path.parts:
            continue
        try:
            rel = path.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        if source.matches(rel):
            yield path


# ---------------------------------------------------------------------------
# Design-context materialisation
# ---------------------------------------------------------------------------


class DesignContextRowOutcome(BaseModel):
    """Per-row outcome of :func:`materialize_design_context_sources`.

    One row per processed ``design_context_sources`` entry per path.
    Phase 1 ships the VCM (path-2) materialiser; Phase 3 adds the
    Kuzu (path-1) KG ingestion outcomes with the same shape
    (``path='kuzu'``). ``status`` semantics is per-path:

    - ``path='vcm'`` — pass-through of ``MmapResult.status``
      (``mapped`` / ``already_mapped`` / ``error``); ``pinned``
      reflects whether ``pin_in_vcm`` was honoured.
    - ``path='kuzu'`` — ``completed`` when at least one file
      ingested cleanly, ``partial`` when some files failed,
      ``error`` when none ingested (or no graph store wired),
      ``skipped`` when ``include_kuzu=False``. ``num_claims``
      is the delta of :class:`IngestionRecord.claims_extracted`
      summed across the row's files.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_name: str
    path: str = Field(description="'vcm' or 'kuzu'.")
    scope_id: str = Field(
        default="",
        description=(
            "VCM scope id (path='vcm'); empty for path='kuzu' "
            "(claims are keyed by source_uri in the graph store, "
            "not by a VCM scope)."
        ),
    )
    status: str = Field(
        description=(
            "Per-path outcome status — see the class docstring for "
            "per-path semantics."
        ),
    )
    num_files: int = 0
    num_claims: int = Field(
        default=0,
        description=(
            "Total ``IngestionRecord.claims_extracted`` summed across "
            "this row's files (path='kuzu' only)."
        ),
    )
    pinned: bool = False
    error: str = ""


class DesignContextMaterialisationReport(BaseModel):
    """Aggregated output of :func:`materialize_design_context_sources`.

    With ``include_kuzu=True`` (the default), one VCM row and one
    Kuzu row per declared ``design_context_sources`` entry. With
    ``include_kuzu=False``, only the VCM rows.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: tuple[DesignContextRowOutcome, ...] = ()

    @property
    def vcm_rows(self) -> tuple[DesignContextRowOutcome, ...]:
        return tuple(r for r in self.rows if r.path == "vcm")

    @property
    def kuzu_rows(self) -> tuple[DesignContextRowOutcome, ...]:
        return tuple(r for r in self.rows if r.path == "kuzu")

    @property
    def mapped_count(self) -> int:
        """VCM rows whose mmap succeeded (mapped or already_mapped)."""

        return sum(
            1 for r in self.vcm_rows
            if r.status in ("mapped", "already_mapped")
        )

    @property
    def pinned_count(self) -> int:
        return sum(1 for r in self.vcm_rows if r.pinned)

    @property
    def ingested_count(self) -> int:
        """Kuzu rows that ingested at least one file cleanly."""

        return sum(
            1 for r in self.kuzu_rows
            if r.status in ("completed", "partial")
        )

    @property
    def total_claims_extracted(self) -> int:
        """Sum of claims extracted across all Kuzu rows."""

        return sum(r.num_claims for r in self.kuzu_rows)

    @property
    def failed_count(self) -> int:
        return sum(
            1 for r in self.rows
            if r.status == "error" or r.error
        )


async def materialize_design_context_sources(
    *,
    vcm_handle: Any,
    repo_map: RepoMap,
    repo_root: Path,
    base_scope_id: str,
    origin_url: str,
    branch: str,
    commit: str,
    mmap_config: MmapConfig,
    renewer: DesignContextLockRenewer | None = None,
    enabled_sources: set[str] | None = None,
    include_kuzu: bool = True,
    ingestor: Any = None,
) -> DesignContextMaterialisationReport:
    """Materialise every row in ``repo_map.design_context_sources``
    through the two non-raw ingestion paths of §5:

    - **Path 2 (VCM, always)**: issue ``mmap_application_scope(
      source_type='literature', ...)`` with the row's path globs;
      if ``pin_in_vcm=True`` and a ``renewer`` is provided, list the
      materialised pages via ``get_pages_for_scope`` and ``lock_page``
      each one for ``pin_lock_duration_days * 86400`` seconds, then
      ``renewer.register`` so the lock refreshes before expiry.

    - **Path 1 (Kuzu KG, when ``include_kuzu=True``)**: for each
      matching file, call ``ingestor.ingest_file(path,
      source_uri=f"design_context://{row.name}/{rel}")``. The
      ingestor's deterministic claim extractor (rule-based ``is_a``
      pattern — see :class:`DeterministicClaimExtractor`) emits
      ``Claim`` instances whose ``CitationSpan.source_uri`` starts
      with ``"design_context://{row.name}/"``, which the downstream
      ``find_inconsistencies`` / ``search_design_context(path='kuzu')``
      actions filter on. LLM-extracted claims arrive in Phase P3d
      when an :class:`LLMClaimExtractor` is wired into the singleton
      Ingestor.

    Failures on a single row are logged + recorded in the report
    and do NOT block subsequent rows — partial materialisation
    beats no materialisation. Per-file ingestion failures within a
    row degrade the row to ``status='partial'`` (some files
    ingested, some didn't) or ``status='error'`` (zero ingested).

    The caller is responsible for emitting any blackboard events
    (``DesignContextMappedProtocol``); this function intentionally
    stays decoupled from the blackboard to match the shape of
    :func:`materialize_vcm_sources` and to keep the function pure /
    testable. The ``RepoStateProvider.materialize_design_context``
    action wires events after this returns — one event per
    (source, path) tuple.

    ``enabled_sources``, when not ``None``, restricts materialisation
    to rows whose ``name`` is in the set.

    ``include_kuzu=False`` opts out of path-1 entirely (a kuzu row
    is still emitted per source with ``status='skipped'`` so the
    caller's per-row blackboard emission stays consistent across
    invocations).

    ``ingestor``, when not ``None``, overrides
    :func:`polymathera.colony.knowledge.deps.get_default_ingestor`
    (used by tests to inject mocks). The default Ingestor is wired
    in ``set_knowledge_deps`` with the configured graph store
    (``knowledge.graph_db_path`` → KuzuGraphStore, or InMemory).
    """

    # Per-row fan-out. Each row has independent VCM-mapping + Kùzu KG
    # paths; they don't share mutable state at this level (the per-file
    # gather inside ``_materialize_kuzu_path`` already caps Anthropic
    # concurrency at 10 via its own semaphore, and Kùzu writes serialize
    # at ``threading.RLock``). Running rows in parallel collapses the
    # outer wall-time too.

    async def _materialize_one(
        src: DesignContextSource,
    ) -> tuple[DesignContextRowOutcome, DesignContextRowOutcome]:
        files = list(_iter_design_context_files(repo_root, src))
        num_files = len(files)

        # ----------------- Path 2: VCM mapping --------------------------
        vcm_outcome = await _materialize_vcm_path(
            vcm_handle=vcm_handle,
            src=src,
            num_files=num_files,
            base_scope_id=base_scope_id,
            origin_url=origin_url,
            branch=branch,
            commit=commit,
            mmap_config=mmap_config,
            renewer=renewer,
        )

        # ----------------- Path 1: Kuzu KG ingestion --------------------
        kuzu_outcome = await _materialize_kuzu_path(
            src=src,
            files=files,
            repo_root=repo_root,
            include_kuzu=include_kuzu,
            ingestor=ingestor,
        )
        return vcm_outcome, kuzu_outcome

    selected_sources = [
        s for s in repo_map.design_context_sources
        if enabled_sources is None or s.name in enabled_sources
    ]
    per_row = await asyncio.gather(
        *(_materialize_one(s) for s in selected_sources),
    )

    outcomes: list[DesignContextRowOutcome] = []
    for vcm_outcome, kuzu_outcome in per_row:
        outcomes.append(vcm_outcome)
        outcomes.append(kuzu_outcome)

    return DesignContextMaterialisationReport(rows=tuple(outcomes))


async def _materialize_vcm_path(
    *,
    vcm_handle: Any,
    src: DesignContextSource,
    num_files: int,
    base_scope_id: str,
    origin_url: str,
    branch: str,
    commit: str,
    mmap_config: MmapConfig,
    renewer: DesignContextLockRenewer | None,
) -> DesignContextRowOutcome:
    """Path-2 (VCM page materialisation) for one ``design_context_sources``
    row. Extracted from the materialiser body so the per-path
    error-handling story is contained."""

    scope_id = f"{base_scope_id}:design_context.{src.name}"

    try:
        result = await vcm_handle.mmap_application_scope(
            config=mmap_config,
            scope_id=scope_id,
            source_type="literature",
            origin_url=origin_url,
            branch=branch,
            commit=commit,
            include_globs=list(src.paths),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "materialize_design_context_sources: mmap failed for "
            "row %r (scope_id=%s); continuing with the next row.",
            src.name, scope_id,
        )
        return DesignContextRowOutcome(
            source_name=src.name,
            path="vcm",
            scope_id=scope_id,
            status="error",
            num_files=num_files,
            pinned=False,
            error=str(exc),
        )

    mmap_status = getattr(result, "status", "unknown")

    # Pin path — only attempted when the row asked for it AND the
    # caller provided a renewer. Without a renewer, locks would
    # expire silently; better to no-op + log than to leave the
    # operator with a false sense of pinning.
    pinned = False
    pin_error = ""
    if src.pin_in_vcm:
        if renewer is None:
            pin_error = "pin_in_vcm requested but no renewer provided"
            logger.warning(
                "materialize_design_context_sources: row %r has "
                "pin_in_vcm=True but no DesignContextLockRenewer "
                "was passed — pages will materialise but NOT be "
                "pinned. Caller must wire a renewer for pinning "
                "to take effect.",
                src.name,
            )
        else:
            try:
                pinned = await _pin_scope_pages(
                    vcm_handle=vcm_handle,
                    renewer=renewer,
                    source_name=src.name,
                    scope_id=scope_id,
                    lock_duration_s=(
                        float(src.pin_lock_duration_days) * 86400.0
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                pin_error = f"pin_failed: {exc}"
                logger.exception(
                    "materialize_design_context_sources: pinning "
                    "failed for row %r (scope %s) — mmap succeeded "
                    "but pages are NOT locked.",
                    src.name, scope_id,
                )

    return DesignContextRowOutcome(
        source_name=src.name,
        path="vcm",
        scope_id=scope_id,
        status=mmap_status,
        num_files=num_files,
        pinned=pinned,
        error=pin_error,
    )


async def _materialize_kuzu_path(
    *,
    src: DesignContextSource,
    files: list[Path],
    repo_root: Path,
    include_kuzu: bool,
    ingestor: Any,
) -> DesignContextRowOutcome:
    """Path-1 (Kuzu KG claim extraction) for one
    ``design_context_sources`` row. Per-file
    :meth:`Ingestor.ingest_file` invocations with the
    ``design_context://`` source-URI scheme; results aggregated into
    a single row outcome.

    Returns a row with ``status='skipped'`` when
    ``include_kuzu=False`` (so the caller's per-row blackboard
    emission stays consistent), ``status='completed'`` when all files
    ingested cleanly, ``status='partial'`` when some failed but at
    least one succeeded, ``status='error'`` when none succeeded or
    the ingestor singleton couldn't be resolved.
    """

    num_files = len(files)
    if not include_kuzu:
        return DesignContextRowOutcome(
            source_name=src.name,
            path="kuzu",
            scope_id="",
            status="skipped",
            num_files=num_files,
            num_claims=0,
            pinned=False,
            error="",
        )

    if ingestor is None:
        try:
            # Lazy import — the singleton lives in the knowledge package,
            # and importing at the top of this module would tighten the
            # coupling between design_monorepo and knowledge.
            from polymathera.colony.knowledge.deps import (
                get_default_ingestor,
            )
            ingestor = get_default_ingestor()
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "materialize_design_context_sources: cannot resolve the "
                "default Ingestor for row %r; KG ingestion skipped.",
                src.name,
            )
            return DesignContextRowOutcome(
                source_name=src.name,
                path="kuzu",
                scope_id="",
                status="error",
                num_files=num_files,
                num_claims=0,
                pinned=False,
                error=f"ingestor_unavailable: {exc}",
            )

    if num_files == 0:
        return DesignContextRowOutcome(
            source_name=src.name,
            path="kuzu",
            scope_id="",
            status="completed",
            num_files=0,
            num_claims=0,
            pinned=False,
            error="",
        )

    # Bounded-concurrency fan-out. Each ``ingest_file`` issues one
    # ``LLMClaimExtractor`` Anthropic call (~10s) + a Kùzu graph write.
    # The Anthropic deployment already caps concurrent inference at
    # ``max_concurrent_requests=10`` (remote_config.py) and shapes the
    # rate via its token bucket; KuzuGraphStore serializes writes via
    # ``threading.RLock`` (stores/graph.py:501). So gathering the file-
    # level coroutines is safe: the deployment + the store are the
    # actual throttles. The semaphore here just keeps the file-level
    # gather from queuing thousands of pending coroutines on huge repos.
    _INGEST_CONCURRENCY = 10
    sem = asyncio.Semaphore(_INGEST_CONCURRENCY)

    async def _ingest_one(file_path: Path) -> tuple[str, Any | None, str | None]:
        try:
            rel = file_path.relative_to(repo_root).as_posix()
        except ValueError:
            rel = str(file_path)
        source_uri = (
            f"{DESIGN_CONTEXT_URI_SCHEME}://{src.name}/{rel}"
        )
        async with sem:
            try:
                record = await ingestor.ingest_file(
                    file_path,
                    # Hand-authored design context is the foundational
                    # design layer for the project — high retrieval weight,
                    # consistent with how the master pipeline treats
                    # tier-1 foundational sources.
                    tier=CorpusTier.TIER_1_FOUNDATIONS,
                    source_uri=source_uri,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "materialize_design_context_sources: ingest_file failed "
                    "for %s in row %r; continuing with the next file.",
                    file_path, src.name,
                )
                return rel, None, str(exc)
        return rel, record, None

    results = await asyncio.gather(
        *(_ingest_one(p) for p in files), return_exceptions=False,
    )

    total_claims = 0
    successes = 0
    failures: list[str] = []
    for rel, record, exc_msg in results:
        if exc_msg is not None:
            failures.append(f"{rel}: {exc_msg}")
            continue
        status = getattr(record, "status", None)
        # ``IngestionStatus`` enum values: pending, parsing, completed,
        # skipped_already_present, tier_upgraded, failed. Treat the
        # success-set as ``completed`` + the two skip variants
        # (the source was previously ingested → claims already in KG;
        # no new claims to count for this run but the file is
        # represented in the graph).
        status_value = getattr(status, "value", str(status))
        if status_value in (
            IngestionStatus.COMPLETED.value,
            IngestionStatus.SKIPPED_ALREADY_PRESENT.value,
            IngestionStatus.TIER_UPGRADED.value,
        ):
            successes += 1
            total_claims += getattr(record, "claims_extracted", 0) or 0
        else:
            failures.append(
                f"{rel}: status={status_value} "
                f"error={getattr(record, 'error', '') or ''}",
            )

    if successes == num_files:
        row_status = "completed"
        error_msg = ""
    elif successes > 0:
        row_status = "partial"
        # Cap to avoid bloating the response with hundreds of file
        # errors; the planner can re-run with ``enabled_sources``
        # to isolate the problem files.
        error_msg = "; ".join(failures[:5])
        if len(failures) > 5:
            error_msg += f" (and {len(failures) - 5} more)"
    else:
        row_status = "error"
        error_msg = "; ".join(failures[:5])

    return DesignContextRowOutcome(
        source_name=src.name,
        path="kuzu",
        scope_id="",
        status=row_status,
        num_files=num_files,
        num_claims=total_claims,
        pinned=False,
        error=error_msg,
    )


async def _pin_scope_pages(
    *,
    vcm_handle: Any,
    renewer: DesignContextLockRenewer,
    source_name: str,
    scope_id: str,
    lock_duration_s: float,
) -> bool:
    """Apply the initial round of locks + register the scope for
    renewal. Returns ``True`` when at least one page was locked
    successfully (i.e. there's something to pin). An empty scope
    (no pages yet — e.g. mapping just registered, content still
    loading) still registers with the renewer so its first tick
    picks up newly-materialised pages."""

    pages = await vcm_handle.get_pages_for_scope(scope_id=scope_id)
    locked_any = False
    for page in pages:
        page_id = page.get("page_id")
        if not page_id:
            continue
        try:
            await vcm_handle.lock_page(
                page_id=page_id,
                locked_by=f"design_context.{source_name}",
                lock_duration_s=lock_duration_s,
                reason=f"design_context pin ({source_name})",
            )
            locked_any = True
        except Exception:  # noqa: BLE001 — renewer will retry on next tick
            logger.exception(
                "_pin_scope_pages: lock_page failed for page %r in "
                "scope %r; renewer will retry on its next tick.",
                page_id, scope_id,
            )

    await renewer.register(
        source_name=source_name,
        scope_id=scope_id,
        lock_duration_s=lock_duration_s,
    )
    # Return True even if zero pages were locked this round, because
    # we DID register with the renewer — the next tick will catch
    # newly-materialised pages. The distinction "locked nothing yet
    # but registered" is more useful to callers than "no pages found
    # so report unpinned".
    return locked_any or len(pages) == 0


def _iter_design_context_files(repo_root: Path, source: DesignContextSource):
    """Yield absolute paths under ``repo_root`` matching this row.

    Mirrors :func:`_iter_matching_files` but skips the sidecar-dir
    filter (design context has no sidecars to dodge). Used for
    ``num_files`` reporting; the actual page materialisation happens
    inside VCM's literature ``ContextPageSource`` which walks the
    same globs."""

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        if source.matches(rel):
            yield path


__all__ = (
    "AcquisitionOutcome",
    "DesignContextMaterialisationReport",
    "DesignContextRowOutcome",
    "KnowledgeMaterialisationReport",
    "materialize_design_context_sources",
    "materialize_knowledge_sources",
    "materialize_scope_ids",
    "materialize_vcm_sources",
)
