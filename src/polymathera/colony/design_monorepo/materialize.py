"""Bridge between :class:`RepoMap` and the two materialisation
operations it drives:

- :func:`materialize_vcm_sources` — issues one
  ``mmap_application_scope`` per row in ``vcm_sources:`` (VCM
  mapping; the dashboard's "Map to VCM" button + the CLI auto-deploy
  flow).
- :func:`materialize_knowledge_sources` — walks each row in
  ``knowledge_sources:`` and feeds matching files to the
  process-singleton :class:`Ingestor` (KB ingestion; the dashboard's
  "Ingest Knowledge" button + the SessionAgent's
  ``ingest_repo_map_literature`` action).

Both accept ``enabled_sources`` so the dashboard's per-section
checkbox lists can filter the rows actually materialised. The two
operations are orthogonal — the same path can be VCM-mapped,
KB-ingested, both, or neither, and the dashboard exposes a separate
button per operation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from polymathera.colony.distributed import get_polymathera
from polymathera.colony.distributed.ray_utils import serving
from polymathera.colony.vcm.models import MmapConfig

from .repo_map import RepoMap


logger = logging.getLogger(__name__)


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


async def materialize_knowledge_sources(
    *,
    repo_map: RepoMap,
    repo_root: Path,
    enabled_sources: set[str] | None = None,
) -> list[Any]:
    """Walk ``repo_map.knowledge_sources`` and ingest every matching
    file via the process-singleton :class:`Ingestor`.

    ``enabled_sources``, when not ``None``, restricts ingestion to
    rows whose ``name`` is in the set. The default (``None``) ingests
    every row.

    Skips files larger than the readers' size limit (the readers
    enforce this themselves; we don't pre-filter). Returns the list
    of :class:`IngestionRecord` so callers (dashboard preview,
    SessionAgent action) can surface per-file outcomes.
    """

    from polymathera.colony.knowledge.deps import get_default_ingestor

    ingestor = get_default_ingestor()
    records: list[Any] = []
    for source in repo_map.knowledge_sources:
        if enabled_sources is not None and source.name not in enabled_sources:
            continue
        for abs_path in _iter_matching_files(repo_root, source):
            try:
                rec = await ingestor.ingest_file(
                    abs_path,
                    data_type_override=source.profile,
                )
                records.append(rec)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "materialize_knowledge_sources: ingest failed for %s",
                    abs_path,
                )
    return records


def _iter_matching_files(repo_root: Path, source: "KnowledgeSource"):
    """Yield absolute paths under ``repo_root`` whose forward-slash
    relative path matches ``source``. Walks the tree once; the
    ``KnowledgeSource.matches`` call delegates to :class:`pathspec`."""

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        if source.matches(rel):
            yield path


# Forward references so the type annotations above resolve without a
# runtime cycle through the heavy ``RepoMap`` import path.
if False:  # pragma: no cover
    from .repo_map import KnowledgeSource, RepoMap  # noqa: F401


__all__ = (
    "materialize_knowledge_sources",
    "materialize_scope_ids",
    "materialize_vcm_sources",
)
