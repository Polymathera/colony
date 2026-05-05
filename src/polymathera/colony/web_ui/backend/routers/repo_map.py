"""Read-only endpoints backing the dashboard's "Design Monorepo" tab.

Three endpoints, all ``Ring.USER`` and gated by ``require_auth``:

- ``GET /repo-map``           — current ``.colony/repo_map.yaml`` plus
                                its parsed sources (or the default
                                fallback when the file is absent).
- ``GET /repo-map/tree``      — directory tree of the cloned design
                                monorepo, annotated with the source
                                each path resolves to.
- ``POST /repo-map/preview``  — dry-run: list the
                                ``mmap_application_scope`` kwargs the
                                materialiser would issue, without
                                executing them.

All three accept the same ``origin_url`` / ``branch`` / ``commit``
parameters as ``/vcm/map-repo`` so the tab can preview a repo before
the user maps it. Cloning goes through the colony's
``GitFileStorage`` (idempotent — no-op when the repo is already on
``/mnt/shared``).

Write-side (``PUT /repo-map`` to commit a new YAML back to the
monorepo) is intentionally deferred — it needs auth and branch-policy
decisions that belong with the operator, not the framework.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth.middleware import require_auth
from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection


logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RepoMapResponse(BaseModel):
    """Returned by ``GET /repo-map``."""

    origin_url: str
    branch: str
    commit: str
    has_repo_map_file: bool = Field(
        description="True when ``.colony/repo_map.yaml`` exists in the clone.",
    )
    raw_yaml: str | None = Field(
        default=None,
        description=(
            "Raw file contents when present. ``None`` when the repo "
            "uses the default fallback. The frontend renders this in "
            "a read-only Monaco editor."
        ),
    )
    sources: list[dict[str, Any]] = Field(
        description=(
            "Parsed source rows (Pydantic model_dump) — what the "
            "materialiser would feed to ``mmap_application_scope``."
        ),
    )


class RepoTreeNode(BaseModel):
    """One node in the repo file tree."""

    path: str
    is_dir: bool
    children: list["RepoTreeNode"] = Field(default_factory=list)


RepoTreeNode.model_rebuild()


class RepoTreeResponse(BaseModel):
    """Returned by ``GET /repo-map/tree``."""

    origin_url: str
    branch: str
    commit: str
    root: RepoTreeNode


class PreviewedSource(BaseModel):
    """One materialised source row."""

    name: str
    scope_id: str
    mmap_kwargs: dict[str, Any]


class RepoMapPreviewRequest(BaseModel):
    origin_url: str
    branch: str = "main"
    commit: str = "HEAD"
    base_scope_id: str | None = None


class RepoMapPreviewResponse(BaseModel):
    origin_url: str
    base_scope_id: str
    sources: list[PreviewedSource]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_TREE_MAX_NODES = 5000
_TREE_MAX_DEPTH = 8


async def _clone_or_retrieve(
    *, origin_url: str, branch: str, commit: str,
) -> Path:
    """Idempotent clone via ``GitFileStorage``. Same call the
    page-source ``initialize`` issues, so this is a no-op when the
    repo is already on the shared volume."""

    from polymathera.colony.distributed import get_polymathera
    from polymathera.colony.distributed.ray_utils import serving

    polymathera = get_polymathera()
    storage = await polymathera.get_storage()
    colony_id = serving.get_colony_id()
    repo_path = await storage.git_storage.clone_or_retrieve_repository(
        origin_url=origin_url,
        branch=branch,
        commit=commit,
        vmr_id=colony_id,
    )
    return Path(str(repo_path))


def _walk_tree(root: Path, *, max_nodes: int, max_depth: int) -> RepoTreeNode:
    """Build a bounded directory tree rooted at ``root``.

    Skips ``.git/`` because it is huge and not interesting for the
    repo-map UI. Bounded by ``max_nodes`` + ``max_depth`` so a
    pathological repo cannot stall the dashboard.
    """

    counter = {"n": 0}

    def _build(p: Path, depth: int) -> RepoTreeNode:
        counter["n"] += 1
        rel = p.relative_to(root).as_posix() or "."
        node = RepoTreeNode(path=rel, is_dir=p.is_dir())
        if not p.is_dir() or depth >= max_depth:
            return node
        try:
            entries = sorted(
                (c for c in p.iterdir() if c.name != ".git"),
                key=lambda c: (not c.is_dir(), c.name),
            )
        except OSError:
            return node
        for child in entries:
            if counter["n"] >= max_nodes:
                break
            node.children.append(_build(child, depth + 1))
        return node

    return _build(root, depth=0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/repo-map", response_model=RepoMapResponse)
async def get_repo_map(
    origin_url: str = Query(..., description="Git URL of the design monorepo."),
    branch: str = Query("main"),
    commit: str = Query("HEAD"),
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> RepoMapResponse:
    """Return the design monorepo's ``.colony/repo_map.yaml`` (or the
    default fallback when absent), parsed into the typed schema.

    Idempotent — clones into ``/mnt/shared`` once, reads the file, no
    side effects on the repo.
    """

    if not colony.is_connected:
        raise HTTPException(status_code=503, detail="Cluster not connected.")

    from polymathera.colony.design_monorepo.repo_map import (
        REPO_MAP_DIR,
        REPO_MAP_FILENAME,
        RepoMap,
    )

    try:
        repo_path = await _clone_or_retrieve(
            origin_url=origin_url, branch=branch, commit=commit,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Clone failed: {e}") from e

    yaml_path = repo_path / REPO_MAP_DIR / REPO_MAP_FILENAME
    has_file = yaml_path.is_file()
    raw_yaml = yaml_path.read_text(encoding="utf-8") if has_file else None
    try:
        repo_map = RepoMap.load(repo_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return RepoMapResponse(
        origin_url=origin_url,
        branch=branch,
        commit=commit,
        has_repo_map_file=has_file,
        raw_yaml=raw_yaml,
        sources=[s.model_dump(mode="json") for s in repo_map.sources],
    )


@router.get("/repo-map/tree", response_model=RepoTreeResponse)
async def get_repo_tree(
    origin_url: str = Query(...),
    branch: str = Query("main"),
    commit: str = Query("HEAD"),
    max_nodes: int = Query(_TREE_MAX_NODES, le=20_000, ge=10),
    max_depth: int = Query(_TREE_MAX_DEPTH, le=20, ge=1),
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> RepoTreeResponse:
    """Return a bounded directory tree of the cloned monorepo.

    The tab uses this to populate the left-hand pane. Bounds prevent
    a runaway monorepo from stalling the dashboard; the operator
    can request more depth via ``max_depth``.
    """

    if not colony.is_connected:
        raise HTTPException(status_code=503, detail="Cluster not connected.")
    try:
        repo_path = await _clone_or_retrieve(
            origin_url=origin_url, branch=branch, commit=commit,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Clone failed: {e}") from e

    root = _walk_tree(repo_path, max_nodes=max_nodes, max_depth=max_depth)
    return RepoTreeResponse(
        origin_url=origin_url, branch=branch, commit=commit, root=root,
    )


@router.post("/repo-map/preview", response_model=RepoMapPreviewResponse)
async def preview_repo_map(
    request: RepoMapPreviewRequest,
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> RepoMapPreviewResponse:
    """Dry-run the materialiser: load the repo map, resolve every
    source row's mmap kwargs, return them — without issuing any
    ``mmap_application_scope`` calls. Useful before the operator
    clicks "Map" in the VCM tab.
    """

    if not colony.is_connected:
        raise HTTPException(status_code=503, detail="Cluster not connected.")

    from polymathera.colony.agents import ScopeUtils
    from polymathera.colony.design_monorepo.materialize import (
        materialize_scope_ids,
    )
    from polymathera.colony.design_monorepo.repo_map import RepoMap

    base_scope_id = request.base_scope_id or ScopeUtils.get_colony_level_scope()
    try:
        repo_path = await _clone_or_retrieve(
            origin_url=request.origin_url,
            branch=request.branch,
            commit=request.commit,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Clone failed: {e}") from e
    try:
        repo_map = RepoMap.load(repo_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    scope_ids = materialize_scope_ids(repo_map, base_scope_id)

    previews: list[PreviewedSource] = []
    for spec, scope_id in zip(repo_map.sources, scope_ids, strict=True):
        try:
            kwargs = spec.to_mmap_kwargs(
                repo_root=repo_path,
                scope_id=scope_id,
                fallback_origin_url=request.origin_url,
                fallback_branch=request.branch,
                fallback_commit=request.commit,
            )
        except Exception as e:  # noqa: BLE001
            kwargs = {"error": str(e)}
        previews.append(
            PreviewedSource(name=spec.name, scope_id=scope_id, mmap_kwargs=kwargs),
        )

    return RepoMapPreviewResponse(
        origin_url=request.origin_url,
        base_scope_id=base_scope_id,
        sources=previews,
    )
