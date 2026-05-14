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
from polymathera.colony.knowledge.models import IngestionRecord, IngestionStatus
from polymathera.colony.knowledge.monorepo_persisted_ingestor import (
    SIDECAR_DIRNAME,
    MonorepoPersistedIngestor,
)
from polymathera.colony.vcm.models import MmapConfig

from .repo_map import KnowledgeSource, RepoMap


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


__all__ = (
    "AcquisitionOutcome",
    "KnowledgeMaterialisationReport",
    "materialize_knowledge_sources",
    "materialize_scope_ids",
    "materialize_vcm_sources",
)
