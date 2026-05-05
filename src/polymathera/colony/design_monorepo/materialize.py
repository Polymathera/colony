"""Bridge between :class:`RepoMap` and the VCM's ``mmap_application_scope``
endpoint, plus the KB-side seeder for ``knowledge_routing``.

Both entry points (CLI ``run_integration_test`` and the dashboard
``/vcm/map-repo`` router) call :func:`materialize_repo_map` instead of
issuing a single ``mmap_application_scope`` directly. The function
clones the design monorepo via :class:`GitFileStorage` (idempotent),
reads ``.colony/repo_map.yaml`` (or falls back to a single
default source), issues one VCM mapping per source row, and finally
walks ``knowledge_routing`` to seed the knowledge base with files
the user has flagged for ingestion (literature committed to the
monorepo; chat-driven acquisition is a separate path).
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
    """Pick one ``scope_id`` per source.

    The single-source default fallback (``RepoMap.default_for_unmapped_repo``)
    keeps using the caller-supplied ``base_scope_id`` so existing
    one-source-per-repo deployments are unaffected. Any
    ``repo_map.yaml`` with named sources composes
    ``f"{base_scope_id}:{source.name}"``.
    """

    sources = repo_map.sources
    if len(sources) == 1 and sources[0].name == "default":
        return [base_scope_id]
    return [f"{base_scope_id}:{s.name}" for s in sources]


async def materialize_repo_map(
    *,
    vcm_handle: Any,
    origin_url: str,
    branch: str,
    commit: str,
    base_scope_id: str,
    mmap_config: MmapConfig,
) -> list[Any]:
    """Clone the design monorepo, load its repo map, and issue one
    ``mmap_application_scope`` call per source row.

    Returns the list of mmap results in source order. Failures on a
    single source row are logged and skipped — the rest still
    materialise — so a typo in one row does not block the whole map.
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
    for spec, scope_id in zip(repo_map.sources, scope_ids, strict=True):
        try:
            kwargs = spec.to_mmap_kwargs(
                repo_root=repo_root,
                scope_id=scope_id,
                fallback_origin_url=origin_url,
                fallback_branch=branch,
                fallback_commit=commit,
            )
            result = await vcm_handle.mmap_application_scope(
                config=mmap_config, **kwargs,
            )
            results.append(result)
        except Exception:  # noqa: BLE001
            logger.exception(
                "materialize_repo_map: source %r (scope_id=%s) failed; "
                "continuing with the remaining sources.",
                spec.name, scope_id,
            )

    if repo_map.knowledge_routing:
        try:
            await materialize_knowledge_routing(
                repo_map=repo_map, repo_root=repo_root,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "materialize_repo_map: knowledge_routing materialisation "
                "failed; VCM mappings remain in place.",
            )
    return results


async def materialize_knowledge_routing(
    *,
    repo_map: "RepoMap",
    repo_root: Path,
) -> list[Any]:
    """Walk ``repo_map.knowledge_routing`` and ingest every matching
    file via the process-singleton :class:`Ingestor`.

    Skips files larger than :data:`MAX_INGEST_BYTES` (the ingestor's
    own readers do this too — early-exit avoids reading the bytes).
    Returns the list of :class:`IngestionRecord` so callers (e.g.,
    the dashboard preview) can surface per-file outcomes.
    """

    from polymathera.colony.knowledge.deps import get_default_ingestor

    ingestor = get_default_ingestor()
    records: list[Any] = []
    for route in repo_map.knowledge_routing:
        # ``ingest_to: vcm`` rows are documentation-only — they
        # declare a path was intentionally promoted to VCM and the
        # KB materialiser must skip it. The corresponding VCM
        # mapping comes from a ``sources:`` row.
        if route.ingest_to != "knowledge_base":
            continue
        for abs_path in _iter_matching_files(repo_root, route):
            try:
                rec = await ingestor.ingest_file(
                    abs_path,
                    data_type_override=route.profile,
                )
                records.append(rec)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "materialize_knowledge_routing: ingest failed for %s",
                    abs_path,
                )
    return records


def _iter_matching_files(repo_root: Path, route: "KnowledgeRoute"):
    """Yield absolute paths under ``repo_root`` whose forward-slash
    relative path matches ``route``. Walks the tree once; the
    ``KnowledgeRoute.matches`` call delegates to :class:`pathspec`."""

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        if route.matches(rel):
            yield path


# Forward references so the type annotations above resolve without a
# runtime cycle through the heavy ``RepoMap`` import path.
if False:  # pragma: no cover
    from .repo_map import KnowledgeRoute, RepoMap  # noqa: F401


__all__ = (
    "materialize_knowledge_routing",
    "materialize_repo_map",
    "materialize_scope_ids",
)
